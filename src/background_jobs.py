"""Thread-safe background execution for slow, independent analysis lookups."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import BoundedSemaphore, RLock
from time import monotonic
from typing import Any, Callable, Hashable


@dataclass(frozen=True)
class JobSnapshot:
    state: str
    result: Any = None
    error: str = ""


@dataclass
class _JobRecord:
    future: Future
    touched_at: float


class BackgroundJobManager:
    """Deduplicate jobs and expose non-blocking snapshots to the UI thread."""

    def __init__(
        self,
        *,
        worker_limits: dict[str, int] | None = None,
        pending_limits: dict[str, int] | None = None,
        completed_ttl: float = 1800,
    ):
        limits = dict(worker_limits) if worker_limits is not None else {
            "virustotal": 2,
            "abuseipdb": 4,
            "geolocation": 4,
            "llm": 2,
            "bert": 1,
            "default": 4,
        }
        self._executors = {
            name: ThreadPoolExecutor(
                max_workers=max(1, int(limit)),
                thread_name_prefix=f"fishstop-{name}",
            )
            for name, limit in limits.items()
        }
        if "default" not in self._executors:
            self._executors["default"] = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="fishstop-default",
            )
            limits["default"] = 4
        configured_pending = pending_limits or {}
        self._capacity = {
            name: BoundedSemaphore(
                value=max(
                    1,
                    int(configured_pending.get(name, max(4, int(limits[name]) * 8))),
                )
            )
            for name in self._executors
        }
        self._completed_ttl = max(1.0, float(completed_ttl))
        self._jobs: dict[tuple[str, Hashable], _JobRecord] = {}
        self._lock = RLock()

    def get_or_submit(
        self,
        pool: str,
        key: Hashable,
        function: Callable,
        *args,
        **kwargs,
    ) -> bool:
        job_key = (pool, key)
        now = monotonic()
        with self._lock:
            self._cleanup_locked(now)
            record = self._jobs.get(job_key)
            if record is not None:
                record.touched_at = now
                return True
            selected_pool = pool if pool in self._executors else "default"
            capacity = self._capacity[selected_pool]
            if not capacity.acquire(blocking=False):
                return False
            executor = self._executors[selected_pool]
            try:
                future = executor.submit(function, *args, **kwargs)
            except Exception:
                capacity.release()
                raise
            future.add_done_callback(lambda _future: capacity.release())
            self._jobs[job_key] = _JobRecord(future=future, touched_at=now)
            return True

    def snapshot(self, pool: str, key: Hashable) -> JobSnapshot:
        job_key = (pool, key)
        now = monotonic()
        with self._lock:
            record = self._jobs.get(job_key)
            if record is None:
                return JobSnapshot("missing")
            record.touched_at = now
            future = record.future
        if future.cancelled():
            return JobSnapshot("cancelled")
        if not future.done():
            return JobSnapshot("running")
        try:
            return JobSnapshot("done", result=future.result())
        except Exception as exc:
            return JobSnapshot("error", error=str(exc))

    def _cleanup_locked(self, now: float) -> None:
        stale = [
            key
            for key, record in self._jobs.items()
            if record.future.done()
            and now - record.touched_at >= self._completed_ttl
        ]
        for key in stale:
            self._jobs.pop(key, None)

    def shutdown(self, *, wait: bool = True) -> None:
        for executor in self._executors.values():
            executor.shutdown(wait=wait, cancel_futures=True)
