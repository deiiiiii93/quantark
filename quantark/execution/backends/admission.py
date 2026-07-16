"""Shared batch-task admission (spec section 11).

Both backends use the same primitive (code-gate finding 2026-07-16: serial
must not bypass byte admission): a task holds a task slot plus its
``est_task_peak_bytes`` lease while executing; on completion the task lease
is swapped for an ``est_outcome_bytes`` lease held until the reducer has
consumed the outcome.
"""

__all__ = ["AdmissionLeases"]

_POOL = "task_scratch"


class AdmissionLeases:
    def __init__(self, lease_manager, plan):
        self._mgr = lease_manager
        self._task = plan.est_task_peak_bytes
        self._out = plan.est_outcome_bytes

    def _move(self, n, sign):
        if self._mgr is not None and n is not None:
            if sign > 0:
                self._mgr.lease_bytes(n, _POOL)
            else:
                self._mgr.release_bytes(n, _POOL)

    def start_task(self):
        if self._mgr is None:
            return None
        slot = self._mgr.task_slot()
        slot.__enter__()
        try:
            self._move(self._task, +1)
        except BaseException:
            slot.__exit__(None, None, None)
            raise
        return slot

    def finish_task(self, slot, *, to_outcome):
        if slot is not None:
            slot.__exit__(None, None, None)
        self._move(self._task, -1)
        if to_outcome:
            self._move(self._out, +1)

    def yield_outcome(self):
        self._move(self._out, -1)
