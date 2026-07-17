"""Backend-independent scenario plan execution (spec sections 13.3, 15).

Built-in runner ``request/v1`` dispatches a transformed ``PricingRequest``
through the existing kernel — it is ``value_kind="native"`` and therefore
serial/threads-only (native value objects do not cross process
boundaries; plan-gate finding 4).

Identical-cell deduplication (plan-gate finding 3): cells with equal
``cell_fingerprint`` execute once, but only the scenario-INDEPENDENT
execution payload is cached — every cell receives its OWN
``ScenarioOutcome`` carrying that cell's ``scenario_id``, so caller
identity and validator pairing survive.

Failure semantics (spec section 15): fail-fast by default — the first
cell failure stops submission and re-raises the native exception.
``collect_errors=True`` converts each failure into a ``PricingFailure``
and continues. Cancellation is honored between cells.
"""
import time

from quantark.execution.cache.fingerprint import try_fingerprint
from quantark.execution.contracts import (
    FrameworkErrorInfo,
    PricingFailure,
    ScenarioOutcome,
)
from quantark.execution.diagnostics import RunDiagnostics
from quantark.execution.errors import CapabilityError, TaskExecutionError
from quantark.execution.scenario import registries
from quantark.execution.scenario.planner import resolve_base

__all__ = ["ResolvedCellInputs", "run_plan", "run_request_cell"]


class ResolvedCellInputs:
    """What a runner receives: the base inputs, the transformer output,
    and the engine for this cell (None for workflow runners that build
    their own engines from payload parameters)."""

    __slots__ = ("base_inputs", "transformed", "engine")

    def __init__(self, base_inputs, transformed, engine=None):
        self.base_inputs = base_inputs
        self.transformed = transformed
        self.engine = engine


def run_request_cell(cell, resolved, child_context):
    """Dispatch the transformed request through the kernel (spec 13.3)."""
    from quantark.execution.kernel import ExecutionKernel

    request = resolved.transformed
    outcome = ExecutionKernel.dispatch(resolved.engine, request, child_context)
    manifest_fp = try_fingerprint(outcome.manifest)
    return outcome.value, outcome.normalized_economics, manifest_fp


registries.register_runner("request/v1", run_request_cell, value_kind="native")


def _resolve_engine_factory(engine_factory):
    if engine_factory is None:
        return None
    if isinstance(engine_factory, str):
        return registries.get_factory(engine_factory).fn
    return engine_factory


def _check_cancelled(context) -> None:
    token = context.cancellation_token
    if token is not None and token.cancelled():
        raise TaskExecutionError(
            "scenario execution cancelled between cells (spec section 15)"
        )


def execute_cell(cell, base_inputs, engine_factory, context):
    """One cell: transform -> build engine -> registered runner.

    Returns the scenario-independent execution payload
    ``(value, economics, manifest_fp, diagnostics)``.
    """
    registration = registries.get_transformer(cell.transformer_id)
    runner = registries.get_runner(cell.runner_id)
    parameters = dict(cell.parameters)
    transformed = registration.fn(base_inputs, parameters)
    factory = _resolve_engine_factory(engine_factory)
    engine = factory(parameters) if factory is not None else None
    resolved = ResolvedCellInputs(base_inputs, transformed, engine)
    start = time.perf_counter()
    value, economics, manifest_fp = runner.fn(cell, resolved, context.child())
    elapsed = time.perf_counter() - start
    diagnostics = RunDiagnostics(
        adapter_id=f"scenario:{cell.runner_id}",
        timings=(("execute_seconds", elapsed),),
    )
    return value, economics, manifest_fp, diagnostics


def _failure(cell, exc) -> PricingFailure:
    return PricingFailure(
        item_id=cell.scenario_id,
        error=FrameworkErrorInfo(
            error_type=type(exc).__name__, message=str(exc)
        ),
        diagnostics=RunDiagnostics(adapter_id=f"scenario:{cell.runner_id}"),
    )


def _emit_summary(context, plan, deduped: int) -> None:
    sink = context.diagnostics_sink
    if sink is None:
        return
    sink.emit(
        RunDiagnostics(
            adapter_id="scenario-runner",
            records=(
                f"scenario:cells={len(plan.cells)}",
                f"scenario:deduped={deduped}",
                f"scenario:plan={plan.plan_id}",
            ),
        )
    )


def _run_serial(plan, base_inputs, engine_factory, context, collect_errors):
    results: list = [None] * len(plan.cells)
    payload_cache: dict = {}
    deduped = 0
    for cell in plan.cells:
        _check_cancelled(context)
        key = cell.cell_fingerprint
        if key is not None and key in payload_cache:
            deduped += 1
            payload = payload_cache[key]
        else:
            try:
                payload = execute_cell(
                    cell, base_inputs, engine_factory, context
                )
            except Exception as exc:  # noqa: BLE001 - typed into the failure
                if not collect_errors:
                    raise
                results[cell.position] = _failure(cell, exc)
                continue
            if key is not None:
                payload_cache[key] = payload
        value, economics, manifest_fp, diagnostics = payload
        results[cell.position] = ScenarioOutcome(
            scenario_id=cell.scenario_id,
            value=value,
            normalized_economics=economics,
            diagnostics=diagnostics,
            manifest_fingerprint=manifest_fp,
        )
    _emit_summary(context, plan, deduped)
    return results


def _run_threads(plan, base_inputs, engine_factory, context, collect_errors):
    """Bounded thread pool over cells. Allowed only for ``request/v1``
    cells: every cell builds its OWN engine from the factory, so no engine
    instance is shared across threads (spec section 12.2)."""
    from concurrent.futures import ThreadPoolExecutor

    for cell in plan.cells:
        if cell.runner_id != "request/v1":
            raise CapabilityError(
                "threads scenario backend supports only request/v1 cells; "
                f"cell {cell.scenario_id!r} uses {cell.runner_id!r}"
            )
    budget = context.resource_budget
    requested = context.execution_policy.scenario.workers
    workers = max(1, min(requested, budget.max_threads, len(plan.cells)))

    def run_one(cell):
        _check_cancelled(context)
        return execute_cell(cell, base_inputs, engine_factory, context)

    results: list = [None] * len(plan.cells)
    deduped = 0  # thread path executes every cell (no cross-thread dedupe)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_one, cell): cell for cell in plan.cells
        }
        first_error = None
        for future, cell in futures.items():
            try:
                payload = future.result()
            except Exception as exc:  # noqa: BLE001 - typed into the failure
                if collect_errors:
                    results[cell.position] = _failure(cell, exc)
                    continue
                if first_error is None:
                    first_error = exc
                continue
            value, economics, manifest_fp, diagnostics = payload
            results[cell.position] = ScenarioOutcome(
                scenario_id=cell.scenario_id,
                value=value,
                normalized_economics=economics,
                diagnostics=diagnostics,
                manifest_fingerprint=manifest_fp,
            )
        if first_error is not None:
            raise first_error
    _emit_summary(context, plan, deduped)
    return results


def run_plan(plan, base, engine_factory, context, *,
             collect_errors: bool = False):
    """Execute an immutable ScenarioPlan; outcomes in caller order."""
    backend = context.execution_policy.scenario.backend
    _, base_inputs, _ = resolve_base(base)
    if backend == "serial":
        return _run_serial(
            plan, base_inputs, engine_factory, context, collect_errors
        )
    if backend == "threads":
        return _run_threads(
            plan, base_inputs, engine_factory, context, collect_errors
        )
    if backend == "processes":
        from quantark.execution.scenario import worker as worker_mod

        return worker_mod.run_plan_processes(
            plan, base, engine_factory, context, collect_errors=collect_errors
        )
    if backend == "dask":
        from quantark.execution.backends import dask_backend

        return dask_backend.run_plan_dask(
            plan, base, engine_factory, context, collect_errors=collect_errors
        )
    raise CapabilityError(
        f"unknown scenario backend {backend!r}; explicit requests never "
        "silently fall back (spec section 3.3)"
    )
