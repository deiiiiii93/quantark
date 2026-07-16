"""Execution-framework adapter for the FX local-vol PDE solver (Phase 4).

``FxLocalVolPDESolver.price`` rebuilds the Dupire surface on EVERY direct
call; the session adapter serves it from the artifact cache (same
descriptor kind and capture-once/reverify contract as the equity/DCN
adapters) and clones the engine around it. exact=True registration:
unknown subclasses fall to the legacy adapter.
"""
from dataclasses import dataclass

from quantark.execution.contracts import (
    EngineCapabilities,
    OutputKind,
    PreparedState,
    PricingOperation,
)
from quantark.execution.errors import CapabilityError
from quantark.execution.legacy_adapter import LegacyPriceAdapter
from quantark.execution.prep.dupire import dupire_surface_state
from quantark.execution.prep.verify import capture_market, verify_market

__all__ = ["FxLVPDESessionAdapter"]

ADAPTER_ID = "fx-pde-prepared"
ADAPTER_VERSION = "1"


@dataclass(frozen=True)
class _PreparedFxPDE:
    clone: object
    capture: object = None  # MarketCapture; verified after execution


def _fx_market_fields(fx_env) -> tuple:
    return (
        fx_env.vol_surface,
        fx_env.spot_quote,
        fx_env.domestic_curve,
        fx_env.foreign_curve,
    )


def _surface_nbytes(surface) -> int:
    total = 0
    for value in vars(surface).values():
        nbytes = getattr(value, "nbytes", None)
        if isinstance(nbytes, int):
            total += nbytes
    return total or (1 << 20)


def _estimate_surface_bytes(vol_surface) -> int:
    iv_nbytes = getattr(getattr(vol_surface, "iv_grid", None), "nbytes", None)
    if isinstance(iv_nbytes, int) and iv_nbytes > 0:
        return 8 * iv_nbytes + (256 << 10)
    return 8 << 20


class FxLVPDESessionAdapter(LegacyPriceAdapter):
    def __init__(self):
        super().__init__(call_shape="product_env")

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            operations=frozenset({PricingOperation.PRICE}),
            output_kinds=frozenset({OutputKind.PV}),
            supported_backends=frozenset({"serial"}),
            fixed_planning=None,
            prepared_state_thread_safe=False,
            instance_reentrant=False,
            process_reconstructable=False,
            deterministic_reduction=True,
            peak_memory_estimate="conservative",
            adapter_id=ADAPTER_ID,
            adapter_version=ADAPTER_VERSION,
        )

    def validate(self, engine, request) -> None:
        if request.operation is not PricingOperation.PRICE:
            raise CapabilityError(
                f"operation {request.operation} unsupported by {ADAPTER_ID}"
            )
        extra = request.outputs - frozenset({OutputKind.PV})
        if extra:
            raise CapabilityError(
                f"outputs {sorted(k.value for k in extra)} unsupported via "
                f"{ADAPTER_ID}"
            )
        if request.pricing_env is None:
            raise CapabilityError("fx_env is required for FX PDE engines")

    def prepare(self, engine, request, context) -> PreparedState:
        from quantark.volmodels.localvol import build_dupire_local_vol

        fx_env = request.pricing_env
        spot = float(fx_env.effective_spot())
        # Capture once. The effective-spot float rides in the fingerprint
        # tuple (it folds in spot_days / market-forward adjustments); the
        # identity slot for it passes the captured value through recapture,
        # while replacement of the underlying quote/curves is caught by the
        # object-identity slots and in-place mutation by re-fingerprinting.
        inputs = (
            fx_env.vol_surface,
            fx_env.spot_quote,
            fx_env.domestic_curve,
            fx_env.foreign_curve,
            ("effective_spot", spot),
        )
        foreign = inputs[3]

        def div_fn(t):
            return float(foreign.get_rate(t))

        def builder():
            return build_dupire_local_vol(
                inputs[0], spot=spot, rate_curve=inputs[2], div_yield=div_fn,
            )

        surface_state = dupire_surface_state(
            prebuilt=engine._prebuilt,
            inputs=inputs,
            recapture=lambda: (
                fx_env.vol_surface,
                fx_env.spot_quote,
                fx_env.domestic_curve,
                fx_env.foreign_curve,
                inputs[4],
            ),
            builder=builder,
            context=context,
            estimate_bytes=_estimate_surface_bytes(inputs[0]),
            measure=_surface_nbytes,
        )
        clone = type(engine)(
            params=engine.params,
            grid_size=engine.grid_size,
            time_steps=engine.time_steps,
            theta=engine.theta,
            local_vol_surface=surface_state.payload,
        )
        return PreparedState(
            payload=_PreparedFxPDE(
                clone=clone,
                capture=capture_market(
                    _fx_market_fields(fx_env), request.product
                ),
            ),
            descriptors=surface_state.descriptors,
            fingerprint=surface_state.fingerprint,
            byte_estimate=surface_state.byte_estimate,
            handles=surface_state.handles,
        )

    def execute_native(self, engine, request, normalized, context, prepared=None):
        clone = prepared.payload.clone if prepared is not None else engine
        result = super().execute_native(clone, request, normalized, context)
        # End-to-end mutation guard (code-gate finding 2026-07-16).
        if prepared is not None and prepared.payload.capture is not None:
            verify_market(
                prepared.payload.capture,
                _fx_market_fields(request.pricing_env),
                request.product,
            )
        return result
