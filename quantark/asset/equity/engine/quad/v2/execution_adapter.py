"""Serial framework integration preserving explicit lifecycle state."""

from dataclasses import replace

from quantark.execution.contracts import PricingOperation, OutputKind
from quantark.execution.errors import CapabilityError
from quantark.execution.legacy_adapter import LegacyPriceAdapter


class QuadV2ExecutionAdapter(LegacyPriceAdapter):
    """Prepared spot batching is exposed by V2's public context API.

    Framework scheduling remains serial; no parallel or grid-exposure
    capability is claimed by this adapter.
    """

    def __init__(self):
        super().__init__(call_shape="product_env")

    def capabilities(self):
        return replace(
            super().capabilities(),
            operations=frozenset(
                {PricingOperation.PRICE, PricingOperation.EVENT_STATS}
            ),
            output_kinds=frozenset(
                {OutputKind.PV, OutputKind.EVENT_STATS, OutputKind.CASHFLOWS}
            ),
            adapter_id="quad-v2-serial",
            adapter_version="1",
        )

    def validate(self, engine, request):
        if request.operation not in {
            PricingOperation.PRICE,
            PricingOperation.EVENT_STATS,
        }:
            raise CapabilityError(
                "QUAD V2 supports PRICE and EVENT_STATS framework operations"
            )
        allowed = (
            {OutputKind.PV}
            if request.operation is PricingOperation.PRICE
            else {OutputKind.PV, OutputKind.EVENT_STATS, OutputKind.CASHFLOWS}
        )
        if request.outputs - set(allowed):
            raise CapabilityError(
                "requested output is not supported by the QUAD V2 serial adapter"
            )
        options = dict(request.operation_options)
        if len(options) != len(request.operation_options) or options.keys() - {
            "event_phase"
        }:
            raise CapabilityError(
                "QUAD V2 accepts only one event_phase operation option"
            )
        if options.get("event_phase", "before") not in {"before", "after"}:
            raise CapabilityError("event_phase must be before or after")
        super().validate(engine, replace(request, outputs=frozenset({OutputKind.PV})))

    def _call_price(self, engine, request):
        return engine.price(
            request.product,
            request.pricing_env,
            lifecycle_state=request.lifecycle_state,
            **dict(request.operation_options),
        )

    def _call_event_stats(self, engine, request):
        return engine.calculate_event_stats(
            request.product,
            request.pricing_env,
            lifecycle_state=request.lifecycle_state,
            **dict(request.operation_options),
        )
