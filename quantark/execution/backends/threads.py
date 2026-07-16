"""Bounded threaded backend (spec sections 8.2, 11, 12.2).

Admission contract (hardened at the 2026-07-16 plan and code gates):

- Every executing batch holds a per-task slot from the lease manager, so
  ``ResourceBudget.max_in_flight`` bounds CONCURRENT BATCH EXECUTION, not
  merely dispatches. The kernel clamps ``window <= max_in_flight``.
- ``est_task_peak_bytes`` is leased while a task executes; on completion it
  is swapped for an ``est_outcome_bytes`` lease held until the reducer has
  consumed that outcome — buffered outcomes stay charged (pathwise-IID
  outcomes carry per-path totals). Both backends share the primitive in
  ``admission.py``.
- Submission gates on ``len(pending) + len(buffered) < window``: total
  retained work is at most ``window`` items. Deadlock-free because tasks
  are submitted in index order, so whenever anything is buffered the next
  canonical index is already pending.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from quantark.execution.backends.admission import AdmissionLeases

__all__ = ["iter_ordered"]


def iter_ordered(plan, execute, workers, window, lease_manager=None,
                 observer=None):
    tasks = list(plan.tasks)
    leases = AdmissionLeases(lease_manager, plan)

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
                    try:
                        yield next_index, outcome
                    finally:
                        leases.yield_outcome()
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
