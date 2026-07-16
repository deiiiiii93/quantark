"""Serial backend (spec section 12.1): the compatibility reference.

Each batch still holds one task slot while executing, so a shared lease
manager sees the same admission accounting as the threaded backend.
"""
import contextlib

__all__ = ["iter_ordered"]


def iter_ordered(plan, execute, lease_manager=None):
    for task in plan.tasks:
        slot = (lease_manager.task_slot() if lease_manager is not None
                else contextlib.nullcontext())
        with slot:
            outcome = execute(task)
        yield task.batch_index, outcome
