"""QUAD V2 autocallable route: exact Gaussian interval moments on the intraday clock."""
from __future__ import annotations


from quantark.intraday.engines.base import (TERMINATED_POINT_GREEKS, EnginePriceOutcome, PointGreeks, decided_at_valuation,
                                            decided_point_greeks, decided_price_outcome, point_greeks_from_estimates)

_DIAGNOSTIC_KEYS = ("nodes", "cells", "events", "states", "model", "backends", "continuous")


class QuadV2Route:
    def price(self, ctx, engine) -> EnginePriceOutcome:
        num = ctx.numerical
        if num.terminated:
            return EnginePriceOutcome(0.0, "terminated", {"reason": "lifecycle state is not alive; only the ledger remains"}, {},
                                      exact=True)
        if decided_at_valuation(ctx):
            # QUAD V2's contract compiler has no zero-maturity twin (the product validates a positive maturity)
            return decided_price_outcome(ctx)
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
        """The kernel's analytic spot derivatives of the discretised continuation value.

        Differentiating the quadrature exactly does not remove its discretisation error; how close
        this derivative is to the true one at a given ``cells_per_sd`` is measured offline.
        """
        num = ctx.numerical
        if num.terminated:
            return TERMINATED_POINT_GREEKS
        if decided_at_valuation(ctx):
            return decided_point_greeks(ctx)
        res = engine.calculate_point_greeks(num.product, ctx.pricing_env, lifecycle_state=num.lifecycle_state,
                                            event_phase=ctx.phase.value)
        return point_greeks_from_estimates(float(res["delta"]), float(res["gamma"]), "kernel_derivative")
