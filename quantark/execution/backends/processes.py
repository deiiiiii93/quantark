"""Bounded spawn-process backend for scenario cells (spec section 12.3).

Same bounded-window / buffered-reassembly discipline as ``threads.py``.
Tested with Python's spawn start method (the macOS default), which is
also the only start method this backend requests.

Fail-fast with error payloads (plan-gate finding 5): worker failures
arrive as SUCCESSFUL futures carrying an ``{"error": ...}`` payload, so
``future.result()`` never raises for them. The completion loop inspects
each completed payload IMMEDIATELY — before buffering it and before any
further submission — and, under ``fail_fast``, cancels pending futures
and raises ``TaskExecutionError`` even when the failing cell's position
is later than unfinished earlier cells.
"""
import multiprocessing
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from quantark.execution.errors import TaskExecutionError

__all__ = ["iter_ordered"]


def iter_ordered(cells, spec_payload, workers, window, *, fail_fast=True,
                 engine_factory_id=None, observer=None):
    """Yield ``(position, outcome_payload)`` in caller order."""
    from quantark.execution.scenario.worker import run_worker_cell

    mp_context = multiprocessing.get_context("spawn")
    buffered: dict = {}
    pending: dict = {}
    submitted = 0
    next_index = 0
    window = max(1, window)

    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp_context
    ) as pool:

        def submit_up_to_window():
            nonlocal submitted
            while (submitted < len(cells)
                   and len(pending) + len(buffered) < window):
                future = pool.submit(
                    run_worker_cell, spec_payload, cells[submitted],
                    engine_factory_id,
                )
                pending[future] = submitted
                submitted += 1
                if observer is not None:
                    observer(submitted)

        submit_up_to_window()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                payload = future.result()  # raises only on infra failure
                error = payload.get("error")
                if error and fail_fast:
                    for other in pending:
                        other.cancel()
                    raise TaskExecutionError(
                        "scenario cell "
                        f"{payload.get('scenario_id', index)!r} failed in a "
                        f"worker process: {error['type']}: {error['message']}"
                        " (fail-fast: pending cells cancelled, submission "
                        "stopped)"
                    )
                buffered[index] = payload
            while next_index in buffered:
                yield next_index, buffered.pop(next_index)
                next_index += 1
            submit_up_to_window()
