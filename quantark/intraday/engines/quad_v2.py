"""QUAD V2 autocallable route: exact Gaussian interval moments on the intraday clock."""
from __future__ import annotations

from math import isfinite

from quantark.intraday.engines.base import TERMINATED_POINT_GREEKS, EnginePriceOutcome, PointGreeks

_DIAGNOSTIC_KEYS = ("nodes", "cells", "events", "states", "model", "backends", "continuous")


class QuadV2Route:
    def price(self, ctx, engine) -> EnginePriceOutcome:
        num = ctx.numerical
        if num.terminated:
            return EnginePriceOutcome(0.0, "terminated", {"reason": "lifecycle state is not alive; only the ledger remains"}, {})
        components = engine.price_components(num.product, ctx.pricing_env, lifecycle_state=num.lifecycle_state,
                                             event_phase=ctx.phase.value)
        pending = float(components.get("pending", 0.0))
        contingent = float(components["price"]) - pending
        numerical = {"reconciliation_error": float(components.get("reconciliation_error", 0.0))}
        # price_components leaves the context it just evaluated in the engine's prepared cache
        prepared = getattr(engine, "_prepared", None)
        diagnostics = dict(getattr(prepared, "diagnostics", {}) or {})
        numerical.update({k: diagnostics[k] for k in _DIAGNOSTIC_KEYS if k in diagnostics})
        backends = diagnostics.get("backends", ())
        method = "quad_v2_fft" if any("fft" in str(b) for b in backends) else "quad_v2_direct"
        return EnginePriceOutcome(contingent, method, numerical,
                                  {k: float(components[k]) for k in ("ko", "coupon", "terminal") if k in components})

    def point_greeks(self, ctx, engine) -> PointGreeks:
        num = ctx.numerical
        if num.terminated:
            return TERMINATED_POINT_GREEKS
        res = engine.calculate_point_greeks(num.product, ctx.pricing_env, lifecycle_state=num.lifecycle_state,
                                            event_phase=ctx.phase.value)
        delta, gamma = float(res["delta"]), float(res["gamma"])
        if not (isfinite(delta) and isfinite(gamma)):
            return PointGreeks(None, None, "undefined", "payoff discontinuity of an unfixed event at the query spot",
                               "kernel_derivative")
        return PointGreeks(delta, gamma, "ok", "", "kernel_derivative")
