import threading
import time

from src.background_jobs import BackgroundJobManager


def _wait_for_terminal(manager, pool, key, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = manager.snapshot(pool, key)
        if snapshot.state in {"done", "error", "cancelled"}:
            return snapshot
        time.sleep(0.01)
    raise AssertionError("Background job did not finish in time")


def test_submission_is_non_blocking_and_result_arrives_later():
    manager = BackgroundJobManager(
        worker_limits={"default": 1},
        completed_ttl=60,
    )
    release = threading.Event()

    def slow_lookup():
        release.wait(timeout=1)
        return {"status": "ok"}

    started = time.monotonic()
    manager.get_or_submit("default", "slow", slow_lookup)
    elapsed = time.monotonic() - started

    try:
        assert elapsed < 0.2
        assert manager.snapshot("default", "slow").state == "running"
        release.set()
        snapshot = _wait_for_terminal(manager, "default", "slow")
        assert snapshot.state == "done"
        assert snapshot.result == {"status": "ok"}
    finally:
        release.set()
        manager.shutdown()


def test_same_job_key_is_submitted_only_once():
    manager = BackgroundJobManager(
        worker_limits={"default": 2},
        completed_ttl=60,
    )
    calls = 0
    lock = threading.Lock()

    def lookup():
        nonlocal calls
        with lock:
            calls += 1
        return "complete"

    try:
        manager.get_or_submit("default", ("url", "same"), lookup)
        manager.get_or_submit("default", ("url", "same"), lookup)
        snapshot = _wait_for_terminal(
            manager,
            "default",
            ("url", "same"),
        )
        assert snapshot.result == "complete"
        assert calls == 1
    finally:
        manager.shutdown()


def test_worker_exception_is_exposed_without_raising_in_ui_thread():
    manager = BackgroundJobManager(
        worker_limits={"default": 1},
        completed_ttl=60,
    )

    def broken_lookup():
        raise RuntimeError("service unavailable")

    try:
        manager.get_or_submit("default", "broken", broken_lookup)
        snapshot = _wait_for_terminal(manager, "default", "broken")
        assert snapshot.state == "error"
        assert "service unavailable" in snapshot.error
    finally:
        manager.shutdown()
