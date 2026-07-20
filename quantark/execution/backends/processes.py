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

__all__ = ["WorkerInfrastructureError", "iter_ordered"]


class WorkerInfrastructureError(TaskExecutionError):
    """A worker PROCESS died (killed/OOM/broken pool) — distinct from a
    cell raising inside a healthy worker. Only these failures are
    retry-eligible (spec section 15), using the identical serialized
    payloads. Carries the positions already yielded so a retry driver can
    resubmit exactly the remainder."""

    def __init__(self, message, *, completed_positions=(),
                 in_flight_positions=(), cancelled_positions=()):
        super().__init__(message)
        self.completed_positions = tuple(completed_positions)
        self.in_flight_positions = tuple(in_flight_positions)
        self.cancelled_positions = tuple(cancelled_positions)


def iter_ordered(cells, spec_payload, workers, window, *, fail_fast=True,
                 engine_factory_id=None, observer=None, positions=None):
    """Yield ``(position, outcome_payload)`` in caller order.

    ``spec_payload`` is either a single worker-spec payload dict shared by
    every cell, or a LIST parallel to ``cells`` (multi-plan packing, spec
    2026-07-20: each cell travels with its own base's worker spec).

    ``positions`` maps local cell indices to caller positions (used by the
    infrastructure-retry driver to resubmit a remainder); defaults to
    identity.
    """
    from quantark.execution.scenario.worker import run_worker_cell

    positions = list(range(len(cells))) if positions is None else list(positions)
    per_cell_specs = isinstance(spec_payload, list)
    if per_cell_specs and len(spec_payload) != len(cells):
        raise TaskExecutionError(
            "per-cell spec payload list must align with cells: "
            f"{len(spec_payload)} != {len(cells)}"
        )
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
                    run_worker_cell,
                    (spec_payload[submitted] if per_cell_specs
                     else spec_payload),
                    cells[submitted],
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
                try:
                    payload = future.result()
                except Exception as exc:  # infra: killed worker/broken pool
                    for other in pending:
                        other.cancel()
                    raise WorkerInfrastructureError(
                        f"worker process failure while executing scenario "
                        f"cell at position {positions[index]}: "
                        f"{type(exc).__name__}: {exc} (completed="
                        f"{next_index}, in-flight cancelled="
                        f"{len(pending)}, unsubmitted="
                        f"{len(cells) - submitted})",
                        completed_positions=[
                            positions[i] for i in range(next_index)
                        ],
                        in_flight_positions=[positions[index]] + [
                            positions[i] for i in pending.values()
                        ],
                        cancelled_positions=[
                            positions[i] for i in range(submitted, len(cells))
                        ],
                    ) from exc
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
                yield positions[next_index], buffered.pop(next_index)
                next_index += 1
            submit_up_to_window()
