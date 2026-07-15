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
        start = time.perf_counter()
        value, economics = adapter.execute_native(
            engine, request, normalized, context
        )
        elapsed = time.perf_counter() - start

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
        )
        diagnostics = RunDiagnostics(
            adapter_id=caps.adapter_id,
            timings=(("execute_seconds", elapsed),),
            policy_sources=context.config_snapshot,
            records=(),
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
