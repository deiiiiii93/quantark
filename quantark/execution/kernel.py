"""Canonical execution lifecycle, serial subset (spec section 7).

Phase 0 implements lifecycle steps: validate -> normalize -> execute ->
normalize output -> emit diagnostics/manifest. Resource leases, plan
fingerprints, and parallel backends arrive in later phases. The kernel never
statically imports asset code; engines reach it via ``BaseEngine.execute``'s
lazy import or through ``PricingSession``.
"""
import time

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
        lease_manager = context.lease_manager
        slot = lease_manager.task_slot() if lease_manager is not None else None
        prepared = None
        prep_seconds = 0.0
        start = time.perf_counter()
        try:
            if slot is not None:
                slot.__enter__()
            if hasattr(adapter, "prepare"):
                t_prep = time.perf_counter()
                prepared = adapter.prepare(engine, request, context)
                prep_seconds = time.perf_counter() - t_prep
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

        records = ()
        cache = context.artifact_cache
        if cache is not None:
            records = tuple(
                f"cache:{k}={v}" for k, v in sorted(cache.stats().items())
            )

        manifest = ReproducibilityManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            request_fingerprint=normalized.fingerprint,
            plan_fingerprint=None,
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
        diagnostics = RunDiagnostics(
            adapter_id=caps.adapter_id,
            timings=(
                ("execute_seconds", elapsed),
                ("prepare_seconds", prep_seconds),
            ),
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
