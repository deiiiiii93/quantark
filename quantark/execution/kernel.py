"""Canonical execution lifecycle (spec section 7).

Phase 0 implemented the serial subset: validate -> normalize -> execute ->
normalize output -> emit diagnostics/manifest. Phase 1 added preparation and
leases. Phase 2 adds the fixed-batch path: when the resolved adapter
implements ``plan_batches``, the kernel builds an immutable BatchPlan, runs
it through the serial or bounded-thread backend, and feeds outcomes to the
adapter's reducer in canonical batch-index order. The kernel never
statically imports asset code; engines reach it via ``BaseEngine.execute``'s
lazy import or through ``PricingSession``.

Dispatch-slot narrowing (plan-gate hardening 2026-07-16): for batch plans
the backends acquire one task slot PER EXECUTING BATCH, so
``ResourceBudget.max_in_flight`` bounds concurrent batch execution; the
kernel therefore does not also hold its dispatch-wide slot around a batch
plan. Non-batch dispatches keep the Phase-1 dispatch-wide slot.

Phase 3 adds the adaptive path: when the resolved adapter implements
``plan_adaptive`` and returns a non-None ``AdaptivePlan``, the kernel runs
``execute_adaptive`` (sequential compatibility mode, spec section 8.4)
under the dispatch-wide slot, stamps the plan fingerprint into the
manifest, and records the complete checkpoint-trace fingerprint in
diagnostics; a None plan falls through to the native legacy call.
"""
import time

from quantark.execution.backends import serial, threads
from quantark.execution.cache.fingerprint import try_fingerprint
from quantark.execution.contracts import PricingOutcome
from quantark.execution.diagnostics import RunDiagnostics
from quantark.execution.errors import CapabilityError
from quantark.execution.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ReproducibilityManifest,
    build_versions,
    platform_tag,
)
from quantark.execution.policy import policy_values
from quantark.execution.registry import build_default_registry

__all__ = ["ExecutionKernel"]

_module_default_registry = None


def _default_registry():
    global _module_default_registry
    if _module_default_registry is None:
        registry = build_default_registry()
        registry.freeze()
        _module_default_registry = registry
    return _module_default_registry


def _run_batch_plan(adapter, plan, prepared, context, caps, clamp_records):
    backend = context.execution_policy.batch.backend
    if backend == "threads":
        if not caps.prepared_state_thread_safe:
            raise CapabilityError(
                f"adapter {caps.adapter_id!r} prepared state is not "
                "thread-safe; threads backend rejected"
            )
        budget = context.resource_budget
        requested = context.execution_policy.batch.workers
        workers = max(1, min(requested, budget.max_threads, plan.num_batches))
        if workers < requested:
            clamp_records.append(
                f"clamp:batch.workers={requested}->{workers}"
            )
        # window is bounded by max_in_flight: batch tasks hold per-task
        # slots, so admission control binds on CONCURRENT BATCH EXECUTION.
        requested_window = (
            context.execution_policy.batch.max_in_flight or workers
        )
        window = max(1, min(requested_window, budget.max_in_flight))
        if window < requested_window:
            clamp_records.append(
                f"clamp:batch.window={requested_window}->{window}"
            )
        indexed = threads.iter_ordered(
            plan, lambda t: adapter.execute_batch(t, prepared, context),
            workers=workers, window=window,
            lease_manager=context.lease_manager,
        )
    else:
        indexed = serial.iter_ordered(
            plan, lambda t: adapter.execute_batch(t, prepared, context),
            lease_manager=context.lease_manager,
        )
    outcomes = (outcome for _, outcome in indexed)
    return adapter.reduce_batches(outcomes, plan, prepared, context)


class ExecutionKernel:
    @staticmethod
    def dispatch(engine, request, context) -> PricingOutcome:
        registry = context.adapter_registry or _default_registry()
        adapter = registry.resolve(engine)
        caps = adapter.capabilities()

        backend = context.execution_policy.batch.backend
        if backend not in caps.supported_backends:
            raise CapabilityError(
                f"backend {backend!r} not supported by adapter "
                f"{caps.adapter_id!r}; supported: "
                f"{sorted(caps.supported_backends)}"
            )
        adapter.validate(engine, request)

        normalized = adapter.normalize(engine, request)
        batch_capable = hasattr(adapter, "plan_batches")
        lease_manager = context.lease_manager
        # Batch plans lease per-task slots inside the backends instead of
        # one dispatch-wide slot (see module docstring).
        slot = (
            lease_manager.task_slot()
            if lease_manager is not None and not batch_capable
            else None
        )
        prepared = None
        plan_fingerprint = None
        clamp_records: list = []
        batch_count = None
        prep_seconds = 0.0
        start = time.perf_counter()
        try:
            if slot is not None:
                slot.__enter__()
            if hasattr(adapter, "prepare"):
                t_prep = time.perf_counter()
                prepared = adapter.prepare(engine, request, context)
                prep_seconds = time.perf_counter() - t_prep
            executed = False
            if hasattr(adapter, "plan_adaptive"):
                adaptive_plan = adapter.plan_adaptive(
                    engine, request, prepared, context
                )
                if adaptive_plan is not None:
                    plan_fingerprint = try_fingerprint(adaptive_plan)
                    value, economics, trace = adapter.execute_adaptive(
                        engine, adaptive_plan, prepared, context
                    )
                    batch_count = len(trace)
                    clamp_records.append(
                        f"adaptive:batches_used={len(trace)}"
                    )
                    clamp_records.append(
                        "adaptive:stopped_early="
                        f"{len(trace) < adaptive_plan.max_batches}"
                    )
                    # Complete-trace evidence (plan-gate finding
                    # 2026-07-16): a deterministic fingerprint of every
                    # checkpoint value, so two runs with different batch
                    # means can never produce identical records.
                    clamp_records.append(
                        f"adaptive:trace_fingerprint={try_fingerprint(trace)}"
                    )
                    executed = True
            if not executed and batch_capable:
                plan = adapter.plan_batches(engine, request, prepared, context)
                plan_fingerprint = try_fingerprint(plan)
                batch_count = plan.num_batches
                value, economics = _run_batch_plan(
                    adapter, plan, prepared, context, caps, clamp_records
                )
                if hasattr(adapter, "project_operation"):
                    value = adapter.project_operation(value, request.operation)
            elif not executed:
                value, economics = adapter.execute_native(
                    engine, request, normalized, context, prepared=prepared
                )
        finally:
            if prepared is not None:
                for handle in prepared.handles:
                    handle.close()
            if slot is not None:
                slot.__exit__(None, None, None)
        elapsed = time.perf_counter() - start

        records = tuple(clamp_records)
        cache = context.artifact_cache
        if cache is not None:
            records += tuple(
                f"cache:{k}={v}" for k, v in sorted(cache.stats().items())
            )
        draw_repo = context.draw_repository
        if draw_repo is not None:
            records += tuple(
                f"draws:{k}={v}" for k, v in sorted(draw_repo.stats().items())
            )

        manifest = ReproducibilityManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            request_fingerprint=normalized.fingerprint,
            plan_fingerprint=plan_fingerprint,
            adapter_id=caps.adapter_id,
            adapter_version=caps.adapter_version,
            engine_class_path=normalized.engine_class_path,
            versions=build_versions(),
            platform=platform_tag(),
            resolved_policy=policy_values(
                context.execution_policy,
                context.resource_budget,
                context.determinism_policy,
            ),
            preparation_fingerprint=(
                prepared.fingerprint if prepared is not None else None
            ),
        )
        timings = (
            ("execute_seconds", elapsed),
            ("prepare_seconds", prep_seconds),
        )
        if batch_count is not None:
            timings += (("batch_count", float(batch_count)),)
        diagnostics = RunDiagnostics(
            adapter_id=caps.adapter_id,
            timings=timings,
            policy_sources=context.config_snapshot,
            records=records,
        )
        outcome = PricingOutcome(
            value=value,
            normalized_economics=economics,
            diagnostics=diagnostics,
            manifest=manifest,
        )
        sink = context.diagnostics_sink
        if sink is not None:
            sink.emit(diagnostics)
        return outcome
