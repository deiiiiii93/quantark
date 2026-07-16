"""Serial backend (spec section 12.1): the compatibility reference.

Uses the SAME admission primitive as the threaded backend (code-gate
finding 2026-07-16): each batch holds a task slot plus its byte lease while
executing, and the outcome lease is released only after the reducer has
consumed the yielded outcome (the ``finally`` also covers generator close).
"""
from quantark.execution.backends.admission import AdmissionLeases

__all__ = ["iter_ordered"]


def iter_ordered(plan, execute, lease_manager=None):
    leases = AdmissionLeases(lease_manager, plan)
    for task in plan.tasks:
        slot = leases.start_task()
        try:
            outcome = execute(task)
        except BaseException:
            leases.finish_task(slot, to_outcome=False)
            raise
        leases.finish_task(slot, to_outcome=True)
        try:
            yield task.batch_index, outcome
        finally:
            leases.yield_outcome()
