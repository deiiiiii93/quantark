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
    from quantark.execution.manifest import platform_tag

    versions = dict(build_versions())
    return (
        ("schema", SCENARIO_SCHEMA_VERSION),
        ("python", versions["python"]),
        ("quantark", versions["quantark"]),
        ("numpy", versions["numpy"]),
        ("scipy", versions["scipy"]),
        ("platform", platform_tag()),
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

    # Minimal, explicit sys.path roots so backends whose workers do NOT
    # inherit the parent's runtime sys.path (Dask) can import the refs.
    import_paths: list = []
    for ref in refs:
        module = importlib.import_module(ref.module)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        import pathlib

        root = pathlib.Path(module_file).resolve().parent
        for _ in range(ref.module.count(".")):
            root = root.parent
        root_str = str(root)
        if root_str not in import_paths:
            import_paths.append(root_str)

    # Canonical payload normalization (code-gate finding 2026-07-17): the
    # spec carries the SAME list->tuple normalization the JSON path
    # applies, so it round-trips equal and parent/worker factories see
    # identical value types.
    base_ref = BaseInputsRef(
        factory_id=base_ref.factory_id,
        payload=_lists_to_pairs(_pairs_to_lists(base_ref.payload)),
    )
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
        expected=_expected_environment() + (
            ("base", plan.resolved_base_fingerprint or "unverifiable"),
        ),
        import_paths=tuple(import_paths),
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
        "import_paths": list(spec.import_paths),
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
        import_paths=tuple(payload.get("import_paths", ())),
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
        if name in ("schema", "base"):
            continue
        if local.get(name) != expected_value:
            raise DeterminismViolation(
                f"worker environment mismatch for {name!r}: parent expected "
                f"{expected_value!r}, worker has {local.get(name)!r}; "
                "failing before numerical execution (spec 12.3)"
            )


def _import_and_verify_refs(spec: WorkerSpec) -> None:
    import sys

    for path in spec.import_paths:
        if path not in sys.path:
            sys.path.append(path)
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
        if registration.schema_version != ref.schema_version:
            raise CapabilityError(
                f"{ref.kind} {ref.ref_id!r} schema version mismatch: the "
                f"plan references {ref.schema_version!r} but this worker "
                f"registers {registration.schema_version!r} (code-gate "
                "finding 2026-07-17)"
            )


def _normalized_payload(payload_pairs) -> dict:
    from quantark.execution.scenario.planner import normalized_payload_dict

    return normalized_payload_dict(payload_pairs)


def _verify_rebuilt_base(spec: WorkerSpec, base_inputs) -> None:
    """The rebuilt base must match the parent's resolved-base fingerprint
    (code-gate finding 2026-07-17): a factory reading drifting local state
    would otherwise price different cells against different bases while
    the plan fingerprint stayed constant. ``unverifiable`` bases (no safe
    canonicalizer) are allowed through by design and recorded as such at
    planning time."""
    from quantark.execution.cache.fingerprint import try_fingerprint

    expected = dict(spec.expected).get("base")
    if expected in (None, "unverifiable"):
        return
    rebuilt = try_fingerprint(base_inputs)
    if rebuilt != expected:
        raise DeterminismViolation(
            f"worker factory {spec.base_ref.factory_id!r} rebuilt a base "
            "that does not match the parent's resolved-base fingerprint; "
            "the factory is not deterministic (fail closed, spec 12.3)"
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
        # Schema gate FIRST, on the raw payload: an unknown schema version
        # must be rejected before any parsing, sys.path mutation, or
        # payload-selected imports can run their side effects in this
        # (possibly long-lived) worker (spec section 22; code-gate
        # 2026-07-18).
        raw_schema = spec_payload.get("schema_version")
        if raw_schema != SCENARIO_SCHEMA_VERSION:
            raise CapabilityError(
                f"worker spec schema {raw_schema!r} does not match this "
                f"worker's {SCENARIO_SCHEMA_VERSION!r}; readers reject "
                "unknown schema versions rather than guessing (spec "
                "section 22)"
            )
        spec = payload_to_worker_spec(spec_payload)
        _import_and_verify_refs(spec)
        verify_worker_environment(spec)
        factory = registries.get_factory(spec.base_ref.factory_id)
        base_inputs = factory.fn(
            _normalized_payload(spec.base_ref.payload)
        )
        _verify_rebuilt_base(spec, base_inputs)
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


def _verify_spawn_safe_main() -> None:
    """Fail closed BEFORE creating a pool when spawn children cannot
    re-import the parent's ``__main__``. Under the spawn start method,
    multiprocessing records ``__main__.__file__`` (when ``__spec__`` is
    unset) and every worker re-executes that file at bootstrap; a
    pseudo-path main (``python - <<EOF`` heredocs record ``<stdin>``) or a
    since-deleted file kills EVERY worker with a FileNotFoundError the
    pool only surfaces as an opaque ``BrokenProcessPool``. Mains run via
    ``-m``/runpy (``__spec__`` set), ``-c`` and REPLs (no ``__file__``),
    and real script files are all left alone."""
    import os
    import sys

    main_module = sys.modules.get("__main__")
    if main_module is None:
        return
    if getattr(getattr(main_module, "__spec__", None), "name", None):
        return
    main_path = getattr(main_module, "__file__", None)
    if main_path is None:
        return
    if (not os.path.basename(main_path).startswith("<")
            and os.path.exists(main_path)):
        return
    raise CapabilityError(
        "the processes scenario backend cannot start spawn workers from "
        f"this __main__ module: {main_path!r} is not a re-importable "
        "file, so every spawn child would die at bootstrap re-running it "
        "(multiprocessing re-executes the parent's __main__ under the "
        "spawn start method). Run from a real script file guarded by "
        "`if __name__ == '__main__':`, or use the serial/threads backend "
        "for stdin/heredoc sessions"
    )


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
    _verify_spawn_safe_main()

    policy = context.execution_policy.scenario
    budget = context.resource_budget
    clamp_records: list = []
    requested_workers = policy.workers
    workers = max(1, min(requested_workers, budget.max_processes,
                         len(plan.cells)))
    if workers < requested_workers:
        clamp_records.append(
            f"clamp:scenario.workers={requested_workers}->{workers}"
        )
    # max_in_flight is a HARD admission bound on concurrent cells
    # (code-gate finding 2026-07-17): the window never exceeds it.
    requested_window = policy.max_in_flight or workers
    window = max(1, min(requested_window, budget.max_in_flight))
    if window < requested_window:
        clamp_records.append(
            f"clamp:scenario.window={requested_window}->{window}"
        )
    workers = min(workers, window)

    spec = build_worker_spec(plan, base, context, workers)
    spec_payload = worker_spec_to_payload(spec)
    cell_payloads = [_cell_payload(cell) for cell in plan.cells]

    results: list = [None] * len(plan.cells)

    def _consume(payload, position) -> None:
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
            return
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

    # Infrastructure retry (spec section 15; code-gate finding 2026-07-17):
    # a KILLED worker (OOM, crash, broken pool) is not a cell failure —
    # collect_errors does not swallow it. With retries configured, the
    # remaining cells are resubmitted with IDENTICAL serialized payloads
    # on a fresh pool; numerical/user-code failures are never retried.
    from quantark.execution.backends.processes import (
        WorkerInfrastructureError,
    )

    remaining = list(range(len(plan.cells)))
    attempts_left = max(0, context.execution_policy.retries)
    while True:
        iterator = processes_iter_ordered(
            [cell_payloads[i] for i in remaining], spec_payload,
            min(workers, max(1, len(remaining))), window,
            fail_fast=not collect_errors,
            engine_factory_id=plan.engine_factory_id,
            positions=remaining,
        )
        try:
            for position, payload in iterator:
                _consume(payload, position)
            break
        except WorkerInfrastructureError:
            if attempts_left <= 0:
                raise
            attempts_left -= 1
            # everything not yet reduced into results is resubmitted —
            # including buffered-but-unyielded completions the dead pool
            # dropped
            remaining = [i for i in remaining if results[i] is None]
            clamp_records.append(
                f"retry:infrastructure={len(remaining)} cells resubmitted"
            )
            if not remaining:
                break
    sink = context.diagnostics_sink
    if sink is not None:
        sink.emit(
            RunDiagnostics(
                adapter_id="scenario-runner",
                records=tuple(clamp_records) + (
                    f"scenario:cells={len(plan.cells)}",
                    "scenario:deduped=0",
                    f"scenario:backend=processes:{workers}",
                    f"scenario:plan={plan.plan_id}",
                ),
            )
        )
    return results
