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


def test_pending_queue_is_bounded_and_accepts_work_again_after_completion():
    manager = BackgroundJobManager(
        worker_limits={"default": 1},
        pending_limits={"default": 1},
        completed_ttl=60,
    )
    release = threading.Event()

    try:
        assert manager.get_or_submit(
            "default", "first", lambda: release.wait(timeout=1)
        ) is True
        assert manager.get_or_submit(
            "default", "rejected", lambda: "never"
        ) is False
        assert manager.snapshot("default", "rejected").state == "missing"

        release.set()
        assert _wait_for_terminal(manager, "default", "first").state == "done"
        assert manager.get_or_submit(
            "default", "next", lambda: "accepted"
        ) is True
        assert _wait_for_terminal(manager, "default", "next").result == "accepted"
    finally:
        release.set()
        manager.shutdown()


def test_progress_is_published_in_thread_safe_snapshots():
    manager = BackgroundJobManager(
        worker_limits={"default": 1},
        completed_ttl=60,
    )
    release = threading.Event()

    def lookup():
        manager.report_progress(
            "default",
            ("job", "session-a"),
            {"current": 2, "total": 4},
        )
        release.wait(timeout=1)
        return "complete"

    try:
        manager.get_or_submit(
            "default",
            ("job", "session-a"),
            lookup,
        )
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            snapshot = manager.snapshot(
                "default",
                ("job", "session-a"),
            )
            if snapshot.progress:
                break
            time.sleep(0.01)

        assert snapshot.state == "running"
        assert snapshot.progress == {"current": 2, "total": 4}
    finally:
        release.set()
        manager.shutdown()


def test_cancel_session_cancels_queued_and_marks_running_jobs():
    manager = BackgroundJobManager(
        worker_limits={"default": 1},
        pending_limits={"default": 2},
        completed_ttl=60,
    )
    release = threading.Event()
    running_key = ("job", "session-a", "running")
    queued_key = ("job", "session-a", "queued")

    try:
        manager.get_or_submit(
            "default",
            running_key,
            lambda: release.wait(timeout=1),
        )
        manager.get_or_submit(
            "default",
            queued_key,
            lambda: "should not run",
        )

        assert manager.cancel_session("session-a") == 2
        assert manager.snapshot("default", running_key).state == "cancelled"
        assert manager.snapshot("default", queued_key).state == "cancelled"
        assert manager.is_cancellation_requested(
            "default",
            running_key,
        ) is True
    finally:
        release.set()
        manager.shutdown()
