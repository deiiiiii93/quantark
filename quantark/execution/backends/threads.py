"""Bounded threaded backend (spec sections 8.2, 11, 12.2).

Admission contract (hardened at the 2026-07-16 plan gate):

- Every executing batch holds a per-task slot from the lease manager, so
  ``ResourceBudget.max_in_flight`` bounds CONCURRENT BATCH EXECUTION, not
  merely dispatches. The kernel clamps ``window <= max_in_flight``.
- ``est_task_peak_bytes`` is leased while a task executes; on completion it
  is swapped for an ``est_outcome_bytes`` lease held until the ordered
  iterator yields that outcome to the reducer — buffered outcomes stay
  charged (pathwise-IID outcomes carry per-path totals).
- Submission gates on ``len(pending) + len(buffered) < window``: total
  retained work is at most ``window`` items. Deadlock-free because tasks
  are submitted in index order, so whenever anything is buffered the next
  canonical index is already pending.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

__all__ = ["iter_ordered"]

_POOL = "task_scratch"


class _Leases:
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


def iter_ordered(plan, execute, workers, window, lease_manager=None,
                 observer=None):
    tasks = list(plan.tasks)
    leases = _Leases(lease_manager, plan)

    def run(task):
        slot = leases.start_task()
        try:
            outcome = execute(task)
        except BaseException:
            leases.finish_task(slot, to_outcome=False)
            raise
        leases.finish_task(slot, to_outcome=True)
        return outcome

    buffered: dict = {}
    next_index = 0
    submitted = 0
    pending: dict = {}

    def submit_up_to_window(pool):
        nonlocal submitted
        while (submitted < len(tasks)
               and len(pending) + len(buffered) < window):
            future = pool.submit(run, tasks[submitted])
            pending[future] = tasks[submitted].batch_index
            submitted += 1
            if observer is not None:
                observer(len(pending), len(buffered))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            submit_up_to_window(pool)
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    buffered[index] = future.result()  # raises on failure
                    if observer is not None:
                        observer(len(pending), len(buffered))
                while next_index in buffered:
                    outcome = buffered.pop(next_index)
                    leases.yield_outcome()
                    yield next_index, outcome
                    next_index += 1
                submit_up_to_window(pool)
        except BaseException:
            for future in pending:
                future.cancel()
            # Running tasks cannot be interrupted (pool shutdown waits on
            # them regardless); wait so their outcome leases are visible,
            # then sweep every charged-but-unconsumed outcome.
            done, _ = wait(pending)
            for future in done:
                if not future.cancelled() and future.exception() is None:
                    leases.yield_outcome()
            for _ in range(len(buffered)):
                leases.yield_outcome()
            raise
