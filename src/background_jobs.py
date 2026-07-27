"""Thread-safe background execution for slow, independent analysis lookups."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock
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
        completed_ttl: float = 1800,
    ):
        limits = worker_limits or {
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
    ) -> None:
        job_key = (pool, key)
        now = monotonic()
        with self._lock:
            self._cleanup_locked(now)
            record = self._jobs.get(job_key)
            if record is not None:
                record.touched_at = now
                return
            executor = self._executors.get(pool, self._executors["default"])
            self._jobs[job_key] = _JobRecord(
                future=executor.submit(function, *args, **kwargs),
                touched_at=now,
            )

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
