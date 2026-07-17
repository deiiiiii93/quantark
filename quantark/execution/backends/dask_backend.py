"""Dask scenario backend over the same plans and reducers (spec 12.4).

Translates ``ScenarioPlan`` cells into Dask futures running the SAME
``run_worker_cell`` entry point the spawn backend uses — worker
verification, CallableRef reconstruction, and the error-payload protocol
are identical, and results flow through the same caller-order reassembly.
There is no separate numerical implementation.

Scope notes (Phase 5):

- ``BatchPlan``-over-Dask is deliberately NOT implemented: fixed-batch MC
  bodies close over live prepared engines by design (the Phase 2
  contract), so they are not process-serializable; scenario-level
  parallelism is the productized Dask surface.
- Legacy Snowball/Phoenix ``use_dask=True`` behavior is untouched,
  including its availability warning/fallback semantics (spec 12.4).
- An EXPLICIT ``dask`` backend request with dask unavailable raises
  ``CapabilityError`` — never a silent fallback.
"""
from quantark.execution.contracts import (
    FrameworkErrorInfo,
    PricingFailure,
    ScenarioOutcome,
)
from quantark.execution.diagnostics import RunDiagnostics
from quantark.execution.errors import CapabilityError, TaskExecutionError
from quantark.execution.scenario import registries
from quantark.execution.scenario.contracts import BaseInputsRef

__all__ = ["available", "iter_ordered", "run_plan_dask"]


def available() -> bool:
    try:
        import distributed  # noqa: F401
    except ImportError:
        return False
    return True


def _require_distributed():
    try:
        import distributed
    except ImportError as exc:
        raise CapabilityError(
            "the dask scenario backend was explicitly requested but "
            "dask.distributed is not installed; explicit requests never "
            "silently fall back (spec section 12.4). Install "
            "'dask[distributed]' or select another backend."
        ) from exc
    return distributed


def iter_ordered(cells, spec_payload, workers, *, fail_fast=True,
                 engine_factory_id=None, client=None):
    """Yield ``(position, outcome_payload)`` in caller order.

    A caller-supplied ``client`` is borrowed (never closed); without one,
    a short-lived process-worker LocalCluster is created.
    """
    distributed = _require_distributed()
    from quantark.execution.scenario.worker import run_worker_cell

    owned_cluster = None
    owned_client = None
    if client is None:
        owned_cluster = distributed.LocalCluster(
            n_workers=workers, threads_per_worker=1, processes=True,
            dashboard_address=None,
        )
        owned_client = distributed.Client(owned_cluster)
        client = owned_client
    try:
        futures = [
            client.submit(
                run_worker_cell, spec_payload, cell, engine_factory_id,
                key=f"scenario-cell-{index}-{cell['cell_fingerprint']}",
                pure=False,
            )
            for index, cell in enumerate(cells)
        ]
        buffered: dict = {}
        next_index = 0
        for future, payload in distributed.as_completed(
            futures, with_results=True
        ):
            index = futures.index(future)
            error = payload.get("error")
            if error and fail_fast:
                for other in futures:
                    if not other.done():
                        other.cancel()
                raise TaskExecutionError(
                    "scenario cell "
                    f"{payload.get('scenario_id', index)!r} failed in a "
                    f"dask worker: {error['type']}: {error['message']} "
                    "(fail-fast: pending cells cancelled)"
                )
            buffered[index] = payload
            while next_index in buffered:
                yield next_index, buffered.pop(next_index)
                next_index += 1
    finally:
        if owned_client is not None:
            owned_client.close()
        if owned_cluster is not None:
            owned_cluster.close()


def run_plan_dask(plan, base, engine_factory, context, *,
                  collect_errors: bool = False, client=None) -> list:
    """Execute a ScenarioPlan on Dask workers (spec 12.4)."""
    _require_distributed()
    from quantark.execution.scenario.worker import (
        build_worker_spec,
        worker_spec_to_payload,
        _cell_payload,
    )

    if plan.base_kind != "inputs_ref" or not isinstance(base, BaseInputsRef):
        raise CapabilityError(
            "the dask scenario backend requires a BaseInputsRef base "
            "(registered factory); live request objects cannot cross a "
            "worker boundary"
        )
    if engine_factory is not None and not isinstance(engine_factory, str):
        raise CapabilityError(
            "the dask scenario backend requires a REGISTERED engine "
            "factory id (string)"
        )
    for cell in plan.cells:
        if registries.get_runner(cell.runner_id).value_kind != "float":
            raise CapabilityError(
                f"runner {cell.runner_id!r} is value_kind='native' and "
                "cannot return its value across a worker boundary "
                "(plan-gate finding 4)"
            )
    if not plan.cells:
        return []

    policy = context.execution_policy.scenario
    budget = context.resource_budget
    workers = max(1, min(policy.workers, budget.max_processes,
                         len(plan.cells)))
    spec = build_worker_spec(plan, base, context, workers)
    spec_payload = worker_spec_to_payload(spec)
    cell_payloads = [_cell_payload(cell) for cell in plan.cells]

    results: list = [None] * len(plan.cells)
    iterator = iter_ordered(
        cell_payloads, spec_payload, workers,
        fail_fast=not collect_errors,
        engine_factory_id=plan.engine_factory_id,
        client=client,
    )
    from quantark.execution.scenario.worker import _lists_to_pairs

    for position, payload in iterator:
        cell = plan.cells[position]
        error = payload.get("error")
        if error:
            results[position] = PricingFailure(
                item_id=cell.scenario_id,
                error=FrameworkErrorInfo(
                    error_type=error["type"], message=error["message"]
                ),
                diagnostics=RunDiagnostics(
                    adapter_id=f"scenario:{cell.runner_id}"
                ),
            )
            continue
        results[position] = ScenarioOutcome(
            scenario_id=cell.scenario_id,
            value=payload["value"],
            normalized_economics=_lists_to_pairs(payload["economics"]),
            diagnostics=RunDiagnostics(
                adapter_id=f"scenario:{cell.runner_id}",
                timings=(
                    ("execute_seconds", payload.get("elapsed_seconds", 0.0)),
                ),
            ),
            manifest_fingerprint=payload.get("manifest_fingerprint"),
        )
    sink = context.diagnostics_sink
    if sink is not None:
        sink.emit(
            RunDiagnostics(
                adapter_id="scenario-runner",
                records=(
                    f"scenario:cells={len(plan.cells)}",
                    "scenario:deduped=0",
                    f"scenario:backend=dask:{workers}",
                    f"scenario:plan={plan.plan_id}",
                ),
            )
        )
    return results
