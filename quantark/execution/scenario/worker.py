"""Spawn-safe worker reconstruction and the process scenario path
(spec section 12.3).

``run_worker_cell`` is the top-level spawn/Dask entry point. A fresh
worker: (1) rebuilds the ``WorkerSpec`` from its JSON payload, (2) IMPORTS
every ``CallableRef`` module — executing that module's ``register_*``
calls — and verifies each imported object IS the registered one, (3)
verifies the environment fingerprints, and only then (4) resolves base
inputs through the registered factory and executes the cell on an
explicit, inner-serial child context. Steps 2-3 fail with
``CapabilityError``/``DeterminismViolation`` BEFORE any numerical work.

Per-process child contexts are cached in ``_CHILD_CONTEXTS`` keyed by the
worker-spec payload. This is per-process construction of IMMUTABLE
services (context, leases, caches), not a mutable worker global carrying
run state — the spec 3.3 ban targets the latter (the solution script's
``_WORKER`` dict of live curves/products rebuilt per initializer is
exactly what ``BaseInputsRef`` + registered factories replace).

Worker failures return as ERROR PAYLOADS in successful futures; the
process backend inspects them on completion (before buffering or further
submission) so fail-fast still stops admission (plan-gate finding 5).
"""
import importlib
import json
import time

from quantark.execution.backends.processes import (
    iter_ordered as processes_iter_ordered,
)
from quantark.execution.contracts import (
    FrameworkErrorInfo,
    PricingFailure,
    ScenarioOutcome,
)
from quantark.execution.diagnostics import RunDiagnostics
from quantark.execution.errors import CapabilityError, DeterminismViolation
from quantark.execution.manifest import build_versions
from quantark.execution.scenario import registries
from quantark.execution.scenario.contracts import (
    SCENARIO_SCHEMA_VERSION,
    BaseInputsRef,
    CallableRef,
    ScenarioCell,
    WorkerSpec,
)

__all__ = [
    "build_worker_spec",
    "payload_to_worker_spec",
    "run_plan_processes",
    "run_worker_cell",
    "verify_worker_environment",
    "worker_spec_to_payload",
]

_DTYPE_TAG = "float64"


def _expected_environment() -> tuple:
    versions = dict(build_versions())
    return (
        ("schema", SCENARIO_SCHEMA_VERSION),
        ("quantark", versions["quantark"]),
        ("numpy", versions["numpy"]),
        ("scipy", versions["scipy"]),
        ("dtype", _DTYPE_TAG),
    )


def build_worker_spec(plan, base_ref, context, workers: int) -> WorkerSpec:
    """Explicit child policy/budget (spec 12.5): inner-serial, nested off,
    parent byte budgets divided across workers. Children never re-read
    environment variables."""
    budget = context.resource_budget

    def _share(total):
        return None if total is None else int(total // max(1, workers))

    refs = [registries.callable_ref("factory", base_ref.factory_id)]
    if plan.engine_factory_id is not None:
        refs.append(registries.callable_ref("factory", plan.engine_factory_id))
    seen = {(r.kind, r.ref_id) for r in refs}
    for cell in plan.cells:
        for kind, ref_id in (
            ("transformer", cell.transformer_id), ("runner", cell.runner_id)
        ):
            if (kind, ref_id) not in seen:
                refs.append(registries.callable_ref(kind, ref_id))
                seen.add((kind, ref_id))

    return WorkerSpec(
        schema_version=SCENARIO_SCHEMA_VERSION,
        base_ref=base_ref,
        callable_refs=tuple(refs),
        child_policy_values=(
            ("batch.backend", "serial"),
            ("scenario.backend", "serial"),
            ("nested_execution", False),
            ("fail_fast", True),
        ),
        child_budget_values=(
            ("max_processes", 1),
            ("max_threads", 1),
            ("max_in_flight", 1),
            ("artifact_cache_bytes", _share(budget.artifact_cache_bytes)),
            ("draw_cache_bytes", _share(budget.draw_cache_bytes)),
            ("total_memory_bytes", _share(budget.total_memory_bytes)),
        ),
        expected=_expected_environment(),
    )


def _pairs_to_lists(pairs) -> list:
    return [[key, value] for key, value in pairs]


def _lists_to_pairs(entries) -> tuple:
    return tuple((key, _deep_tuple(value)) for key, value in entries)


def _deep_tuple(value):
    if isinstance(value, list):
        return tuple(_deep_tuple(entry) for entry in value)
    return value


def worker_spec_to_payload(spec: WorkerSpec) -> dict:
    payload = {
        "schema_version": spec.schema_version,
        "base_ref": {
            "factory_id": spec.base_ref.factory_id,
            "payload": _pairs_to_lists(spec.base_ref.payload),
        },
        "callable_refs": [
            [r.kind, r.ref_id, r.module, r.qualname, r.schema_version]
            for r in spec.callable_refs
        ],
        "child_policy_values": _pairs_to_lists(spec.child_policy_values),
        "child_budget_values": _pairs_to_lists(spec.child_budget_values),
        "expected": _pairs_to_lists(spec.expected),
    }
    registries.check_worker_payload(payload)
    return payload


def payload_to_worker_spec(payload: dict) -> WorkerSpec:
    return WorkerSpec(
        schema_version=payload["schema_version"],
        base_ref=BaseInputsRef(
            factory_id=payload["base_ref"]["factory_id"],
            payload=_lists_to_pairs(payload["base_ref"]["payload"]),
        ),
        callable_refs=tuple(
            CallableRef(*entry) for entry in payload["callable_refs"]
        ),
        child_policy_values=_lists_to_pairs(payload["child_policy_values"]),
        child_budget_values=_lists_to_pairs(payload["child_budget_values"]),
        expected=_lists_to_pairs(payload["expected"]),
    )


def verify_worker_environment(spec: WorkerSpec) -> None:
    """Schema/version verification BEFORE preparation (spec 12.3)."""
    if spec.schema_version != SCENARIO_SCHEMA_VERSION:
        raise CapabilityError(
            f"worker spec schema {spec.schema_version!r} does not match "
            f"this worker's {SCENARIO_SCHEMA_VERSION!r}; readers reject "
            "unknown schema versions rather than guessing (spec section 22)"
        )
    local = dict(_expected_environment())
    for name, expected_value in spec.expected:
        if name == "schema":
            continue
        if local.get(name) != expected_value:
            raise DeterminismViolation(
                f"worker environment mismatch for {name!r}: parent expected "
                f"{expected_value!r}, worker has {local.get(name)!r}; "
                "failing before numerical execution (spec 12.3)"
            )


def _import_and_verify_refs(spec: WorkerSpec) -> None:
    for ref in spec.callable_refs:
        try:
            module = importlib.import_module(ref.module)
        except ImportError as exc:
            raise CapabilityError(
                f"worker cannot import {ref.module!r} for {ref.kind} "
                f"{ref.ref_id!r}: {exc}"
            ) from exc
        obj = module
        try:
            for part in ref.qualname.split("."):
                obj = getattr(obj, part)
        except AttributeError as exc:
            raise CapabilityError(
                f"{ref.module}:{ref.qualname} no longer resolves for "
                f"{ref.kind} {ref.ref_id!r}: {exc}"
            ) from exc
        registration = registries.get_registration(ref.kind, ref.ref_id)
        if registration.fn is not obj:
            raise CapabilityError(
                f"{ref.kind} {ref.ref_id!r} is registered to a different "
                f"object than {ref.module}:{ref.qualname}; refusing to "
                "execute mismatched code"
            )


_CHILD_CONTEXTS: dict = {}


def _child_context(cache_key: str, spec: WorkerSpec):
    context = _CHILD_CONTEXTS.get(cache_key)
    if context is not None:
        return context
    from quantark.execution.cache.artifacts import PreparedArtifactCache
    from quantark.execution.cache.draws import DrawRepository
    from quantark.execution.context import PricingRunContext
    from quantark.execution.diagnostics import InMemoryDiagnosticsSink
    from quantark.execution.leases import ResourceLeaseManager
    from quantark.execution.policy import (
        DeterminismPolicy,
        ExecutionPolicy,
        ExecutorSelection,
        ResourceBudget,
    )

    budget_values = dict(spec.child_budget_values)
    budget = ResourceBudget(
        max_processes=budget_values["max_processes"],
        max_threads=budget_values["max_threads"],
        max_in_flight=budget_values["max_in_flight"],
        artifact_cache_bytes=budget_values["artifact_cache_bytes"],
        draw_cache_bytes=budget_values["draw_cache_bytes"],
        total_memory_bytes=budget_values["total_memory_bytes"],
    )
    policy_values = dict(spec.child_policy_values)
    policy = ExecutionPolicy(
        batch=ExecutorSelection(backend=policy_values["batch.backend"]),
        scenario=ExecutorSelection(backend=policy_values["scenario.backend"]),
        nested_execution=policy_values["nested_execution"],
        fail_fast=policy_values["fail_fast"],
    )
    leases = ResourceLeaseManager(budget)
    cache = PreparedArtifactCache(leases)
    repo = DrawRepository(leases)
    context = PricingRunContext(
        execution_policy=policy,
        resource_budget=budget,
        determinism_policy=DeterminismPolicy(),
        diagnostics_sink=InMemoryDiagnosticsSink(),
        artifact_cache=cache,
        lease_manager=leases,
        draw_repository=repo,
        config_snapshot=(("policy", "worker_spec"),),
    )
    _CHILD_CONTEXTS[cache_key] = context
    return context


def _cell_payload(cell) -> dict:
    payload = {
        "scenario_id": cell.scenario_id,
        "position": cell.position,
        "transformer_id": cell.transformer_id,
        "runner_id": cell.runner_id,
        "parameters": _pairs_to_lists(cell.parameters),
        "invalidate_all": cell.invalidate_all,
        "cell_fingerprint": cell.cell_fingerprint,
    }
    registries.check_worker_payload(payload)
    return payload


def _cell_from_payload(payload: dict) -> ScenarioCell:
    return ScenarioCell(
        scenario_id=payload["scenario_id"],
        position=payload["position"],
        transformer_id=payload["transformer_id"],
        runner_id=payload["runner_id"],
        parameters=_lists_to_pairs(payload["parameters"]),
        mutation_tags=frozenset(),
        changed_tags=frozenset(),
        invalidate_all=payload["invalidate_all"],
        cell_fingerprint=payload["cell_fingerprint"],
        group_key=(payload["runner_id"], payload["transformer_id"]),
    )


def run_worker_cell(spec_payload: dict, cell_payload: dict,
                    engine_factory_id: str | None) -> dict:
    """Top-level spawn/Dask task: reconstruct, verify, execute one cell."""
    scenario_id = cell_payload.get("scenario_id", "<unknown>")
    try:
        spec = payload_to_worker_spec(spec_payload)
        _import_and_verify_refs(spec)
        verify_worker_environment(spec)
        factory = registries.get_factory(spec.base_ref.factory_id)
        base_inputs = factory.fn(dict(spec.base_ref.payload))
        cell = _cell_from_payload(cell_payload)
        context = _child_context(
            json.dumps(spec_payload, sort_keys=True), spec
        )
        from quantark.execution.scenario.runner import execute_cell

        start = time.perf_counter()
        value, economics, manifest_fp, _diagnostics = execute_cell(
            cell, base_inputs, engine_factory_id, context
        )
        elapsed = time.perf_counter() - start
        economics_payload = _pairs_to_lists(economics)
        registries.check_worker_payload(economics_payload)
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
        return {
            "scenario_id": scenario_id,
            "value": float(value) if numeric else None,
            "economics": economics_payload,
            "manifest_fingerprint": manifest_fp,
            "elapsed_seconds": elapsed,
            "error": None,
        }
    except BaseException as exc:  # noqa: BLE001 - serialized to the parent
        return {
            "scenario_id": scenario_id,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def run_plan_processes(plan, base, engine_factory, context, *,
                       collect_errors: bool = False) -> list:
    """Execute a ScenarioPlan on spawn processes (spec 12.3)."""
    if plan.base_kind != "inputs_ref" or not isinstance(base, BaseInputsRef):
        raise CapabilityError(
            "the processes scenario backend requires a BaseInputsRef base "
            "(registered factory); live request objects cannot cross a "
            "process boundary and explicit requests never silently fall "
            "back (spec section 3.3)"
        )
    if engine_factory is not None and not isinstance(engine_factory, str):
        raise CapabilityError(
            "the processes scenario backend requires a REGISTERED engine "
            "factory id (string); live callables cannot cross a process "
            "boundary"
        )
    for cell in plan.cells:
        if registries.get_runner(cell.runner_id).value_kind != "float":
            raise CapabilityError(
                f"runner {cell.runner_id!r} is value_kind='native' and "
                "cannot return its value across a process boundary; "
                "process/dask cells require value_kind='float' runners "
                "(plan-gate finding 4)"
            )
    if not plan.cells:
        return []

    policy = context.execution_policy.scenario
    budget = context.resource_budget
    workers = max(1, min(policy.workers, budget.max_processes,
                         len(plan.cells)))
    window = max(workers, policy.max_in_flight or workers)

    spec = build_worker_spec(plan, base, context, workers)
    spec_payload = worker_spec_to_payload(spec)
    cell_payloads = [_cell_payload(cell) for cell in plan.cells]

    results: list = [None] * len(plan.cells)
    iterator = processes_iter_ordered(
        cell_payloads, spec_payload, workers, window,
        fail_fast=not collect_errors,
        engine_factory_id=plan.engine_factory_id,
    )
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
                    f"scenario:backend=processes:{workers}",
                    f"scenario:plan={plan.plan_id}",
                ),
            )
        )
    return results
