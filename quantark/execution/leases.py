"""Resource lease accounting (spec section 11 — Phase 1 serial subset).

Admission control bounds ADMITTED estimates, not OS memory (spec section 11
as amended). Phase 1 enforces the in-flight task slot and the artifact-cache
byte pool; task-scratch and worker pools arrive with parallel backends in
Phase 2. A closed manager rejects all further acquisitions.
"""
import threading

from quantark.execution.errors import ResourceBudgetExceeded
from quantark.execution.policy import ResourceBudget

__all__ = ["ResourceLeaseManager"]


class ResourceLeaseManager:
    def __init__(self, budget: ResourceBudget):
        self._budget = budget
        self._lock = threading.Lock()
        self._in_flight = 0
        self._pools: dict = {}
        self._capacities = {
            "artifact_cache": budget.artifact_cache_bytes,
            "draw_cache": budget.draw_cache_bytes,
            "task_scratch": budget.total_memory_bytes,
        }
        self._closed = False

    def task_slot(self):
        return _TaskSlot(self)

    def lease_bytes(self, n: int, pool: str) -> None:
        with self._lock:
            if self._closed:
                raise ResourceBudgetExceeded(
                    "ResourceLeaseManager is closed; no post-close acquisitions"
                )
            capacity = self._capacities.get(pool)
            used = self._pools.get(pool, 0)
            if capacity is not None and used + n > capacity:
                raise ResourceBudgetExceeded(
                    f"pool {pool!r}: lease of {n} bytes exceeds capacity "
                    f"{capacity} (in use: {used})"
                )
            self._pools[pool] = used + n

    def release_bytes(self, n: int, pool: str) -> None:
        with self._lock:
            self._pools[pool] = max(0, self._pools.get(pool, 0) - n)

    def pool_bytes(self, pool: str) -> int:
        with self._lock:
            return self._pools.get(pool, 0)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._pools.clear()
            self._in_flight = 0

    def _acquire_slot(self) -> None:
        with self._lock:
            if self._closed:
                raise ResourceBudgetExceeded(
                    "ResourceLeaseManager is closed; no post-close acquisitions"
                )
            if self._in_flight >= self._budget.max_in_flight:
                raise ResourceBudgetExceeded(
                    f"max_in_flight={self._budget.max_in_flight} exhausted"
                )
            self._in_flight += 1

    def _release_slot(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)


class _TaskSlot:
    def __init__(self, mgr):
        self._mgr = mgr

    def __enter__(self):
        self._mgr._acquire_slot()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._mgr._release_slot()
        return False
