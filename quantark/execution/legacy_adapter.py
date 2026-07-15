"""Serial compatibility adapter: routes framework requests to legacy methods.

This is the terminal fallback of adapter resolution (spec section 6.2). It
performs no caching, no batching, and no parallel submission, so its shallow
normalized snapshot (``snapshot_complete=False``, no fingerprint) is safe:
the raw objects are used exactly as a direct legacy call would use them, on
the calling thread. Native legacy exceptions propagate unwrapped.

Serial-boundary note (spec sections 12.4, 17.1): "serial" here means the
FRAMEWORK submits no parallel work. Engine-internal legacy parallelism —
Snowball/Phoenix ``use_dask=True``, ``QUANTARK_DCN_MC_WORKERS`` threads —
is engine-owned preserved behavior and passes through unchanged, exactly as
a direct call would run it. It is outside the framework's (Phase-0-empty)
budget claims; later phases route it through budgeted plans. A session
therefore never alters, rejects, or silently disables an engine's own
configured execution.
"""
from quantark.execution.contracts import (
    EngineCapabilities,
    NormalizedPricingRequest,
    OutputKind,
    PricingOperation,
    PricingRequest,
)
from quantark.execution.errors import CapabilityError

__all__ = ["ADAPTER_ID", "ADAPTER_VERSION", "LegacyPriceAdapter"]

ADAPTER_ID = "legacy-price"
ADAPTER_VERSION = "0"

# The outputs set is the adapter's GUARANTEE, not a description of the native
# payload. Legacy result objects are heterogeneous (e.g. DCNPDEResult has no
# error-estimate field), so this adapter can verify only PV across all
# engines. Requesting any other OutputKind raises CapabilityError rather than
# returning a success that silently omits a requested output; the full native
# object is still returned as ``value``, with whatever fields it natively has.
# Per-engine output bundles arrive with the native adapters (Phases 2/4).
_OP_OUTPUTS = {
    PricingOperation.PRICE: frozenset({OutputKind.PV}),
    PricingOperation.PRICE_DETAILED: frozenset({OutputKind.PV}),
    PricingOperation.EVENT_STATS: frozenset({OutputKind.PV}),
}


class LegacyPriceAdapter:
    def __init__(self, call_shape: str):
        if call_shape not in ("product_env", "env_bound"):
            raise CapabilityError(f"unknown call shape {call_shape!r}")
        self.call_shape = call_shape

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            operations=frozenset(_OP_OUTPUTS),
            output_kinds=frozenset({OutputKind.PV}),
            supported_backends=frozenset({"serial"}),
            fixed_planning=None,
            prepared_state_thread_safe=False,
            instance_reentrant=False,
            process_reconstructable=False,
            deterministic_reduction=True,
            peak_memory_estimate="unavailable",
            adapter_id=ADAPTER_ID,
            adapter_version=ADAPTER_VERSION,
        )

    def validate(self, engine, request: PricingRequest) -> None:
        allowed = _OP_OUTPUTS.get(request.operation)
        if allowed is None:
            raise CapabilityError(
                f"operation {request.operation} unsupported by {ADAPTER_ID}"
            )
        extra = request.outputs - allowed
        if extra:
            raise CapabilityError(
                f"outputs {sorted(k.value for k in extra)} unsupported for "
                f"operation {request.operation.value} via {ADAPTER_ID}"
            )
        if self.call_shape == "product_env" and request.pricing_env is None:
            raise CapabilityError(
                "pricing_env is required for product_env engines"
            )

    def normalize(self, engine, request: PricingRequest) -> NormalizedPricingRequest:
        cls = type(engine)
        return NormalizedPricingRequest(
            engine_class_path=f"{cls.__module__}.{cls.__qualname__}",
            operation=request.operation,
            outputs=request.outputs,
            operation_options=tuple(sorted(request.operation_options)),
            product_ref=request.product,
            pricing_env_ref=request.pricing_env,
            snapshot_complete=False,
            fingerprint=None,
        )

    def execute_native(self, engine, request, normalized, context):
        op = request.operation
        if op is PricingOperation.PRICE:
            value = self._call_price(engine, request)
            return value, (("pv", float(value)),)
        if op is PricingOperation.PRICE_DETAILED:
            value = self._call_detailed(engine, request)
            return value, (("pv", _extract_pv(value)),)
        if op is PricingOperation.EVENT_STATS:
            value = self._call_event_stats(engine, request)
            return value, (("pv", _extract_pv(value)),)
        raise CapabilityError(f"operation {op} unsupported by {ADAPTER_ID}")

    def _call_price(self, engine, request):
        if self.call_shape == "env_bound":
            return engine.price(request.product)
        return engine.price(request.product, request.pricing_env)

    def _call_detailed(self, engine, request):
        if self.call_shape == "env_bound":
            method = getattr(engine, "price_with_details", None)
            if method is None:
                raise CapabilityError(
                    f"{type(engine).__qualname__} has no price_with_details"
                )
            return method(request.product)
        method = getattr(engine, "price_detailed", None)
        if method is None:
            raise CapabilityError(
                f"{type(engine).__qualname__} has no price_detailed"
            )
        return method(request.product, request.pricing_env)

    def _call_event_stats(self, engine, request):
        method = getattr(engine, "calculate_event_stats", None)
        if method is None:
            raise CapabilityError(
                f"{type(engine).__qualname__} has no calculate_event_stats"
            )
        stats = method(request.product, request.pricing_env)
        if stats is None:
            raise CapabilityError(
                f"{type(engine).__qualname__} does not support event stats"
            )
        return stats


def _extract_pv(value):
    for attr in ("pv", "price", "npv"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, (int, float)):
            return float(candidate)
    if isinstance(value, (int, float)):
        return float(value)
    return None
