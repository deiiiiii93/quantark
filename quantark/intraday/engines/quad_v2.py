"""QUAD V2 autocallable route: exact Gaussian interval moments on the intraday clock."""
from __future__ import annotations

import dataclasses
from copy import deepcopy

from quantark.intraday.engines.base import (TERMINATED_POINT_GREEKS, EnginePriceOutcome, PointGreeks,
                                            point_greeks_from_estimates)
from quantark.intraday.events import EventPhase

_DIAGNOSTIC_KEYS = ("nodes", "cells", "events", "states", "model", "backends", "continuous")
#: Relative spot step of the point-Greek stencil on a claim decided at the valuation instant.
_DECIDED_STENCIL = 1e-6


def _decided_at_valuation(ctx) -> bool:
    """Every remaining event sits at the valuation instant (the maturity close) and has not happened yet."""
    return ctx.numerical.maturity_tau == 0.0 and ctx.phase is EventPhase.BEFORE and not ctx.numerical.terminated


def _decided_value(ctx) -> float:
    """The contingent value of a claim whose every remaining event is decided on the known spot now.

    It is what the lifecycle books when those events are fixed at the spot: the pending ledger they leave plus the
    cash paid at the instant, less the ledger already pending -- the identity that relates BEFORE and AFTER at every
    other fixing instant. QUAD V2's contract compiler has no zero-maturity twin (the product validates a positive
    maturity), so the route resolves the decided claim through the runtime's own event resolution instead of
    re-stating the payoff. The ledger is discounted on this context's market, so a bumped cell prices its own.
    """
    from quantark.asset.equity.engine.settlement_support import pending_receivable_pv
    from quantark.intraday.context import resolve_context
    from quantark.intraday.fixings import Fixing
    from quantark.intraday.timestamp import to_utc

    now = to_utc(ctx.valuation_timestamp)
    instants = sorted({e.timestamp for e in ctx.numerical.remaining_events}, key=to_utc)
    history = tuple(f for f in ctx.request.fixings if to_utc(f.timestamp) != now)
    decided = resolve_context(dataclasses.replace(
        ctx.request, event_phase=EventPhase.AFTER,
        fixings=history + tuple(Fixing(t, float(ctx.spot)) for t in instants)))

    def pending(state):
        return float(pending_receivable_pv(state, ctx.pricing_env)) if state is not None else 0.0

    received = float(decided.numerical.paid_cash) - float(ctx.numerical.paid_cash)
    return pending(decided.numerical.lifecycle_state) + received - pending(ctx.numerical.lifecycle_state)


class QuadV2Route:
    def price(self, ctx, engine) -> EnginePriceOutcome:
        num = ctx.numerical
        if num.terminated:
            return EnginePriceOutcome(0.0, "terminated", {"reason": "lifecycle state is not alive; only the ledger remains"}, {},
                                      exact=True)
        if _decided_at_valuation(ctx):
            return EnginePriceOutcome(_decided_value(ctx), "decided_at_valuation",
                                      {"reason": "every remaining event is at the valuation instant: decided on the spot"},
                                      {}, exact=True)
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
        if _decided_at_valuation(ctx):
            return self._decided_point_greeks(ctx, engine)
        res = engine.calculate_point_greeks(num.product, ctx.pricing_env, lifecycle_state=num.lifecycle_state,
                                            event_phase=ctx.phase.value)
        return point_greeks_from_estimates(float(res["delta"]), float(res["gamma"]), "kernel_derivative")

    def _decided_point_greeks(self, ctx, engine) -> PointGreeks:
        """Central differences of the exact decided value at h = 1e-6 S; undefined when the stencil reaches the
        level of an event decided at this instant (one side of it knocks, the other does not). At a payoff kink
        (the strike of a knocked-in claim) the difference is the average of the one-sided slopes."""
        from quantark.intraday.greeks import with_pricing_env

        spot = float(ctx.spot)
        h = _DECIDED_STENCIL * spot
        for event in ctx.numerical.remaining_events:
            if event.barrier is not None and abs(spot - float(event.barrier)) <= h:
                return PointGreeks(None, None, "undefined",
                                   f"the difference stencil reaches the level {float(event.barrier):g} of "
                                   f"{event.event_id}, decided at this instant", "decided_at_valuation")

        def at(s):
            env = deepcopy(ctx.pricing_env)
            env.spot_quote.spot = s
            return self.price(with_pricing_env(ctx, env, f"decided_spot:{s!r}"), engine).contingent_pv

        base, up, down = self.price(ctx, engine).contingent_pv, at(spot + h), at(spot - h)
        return point_greeks_from_estimates((up - down) / (2.0 * h), (up - 2.0 * base + down) / (h * h),
                                           "decided_at_valuation")
