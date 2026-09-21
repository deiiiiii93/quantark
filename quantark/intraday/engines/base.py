"""Route contract: a route prices the resolved context's twin through one engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Protocol, Tuple


@dataclass(frozen=True)
class EnginePriceOutcome:
    contingent_pv: float
    method: str                                   # e.g. "analytical_bs_effective_variance", "quad_v2_direct"
    numerical: Mapping[str, object] = field(default_factory=dict)
    components: Mapping[str, float] = field(default_factory=dict)   # engine-native components when available
    records: Tuple[str, ...] = ()
    #: The engine instance that produced the value when the route priced on a
    #: refined clone; the kernel parity dispatch must use the same instance.
    engine_used: object = field(default=None, compare=False)
    #: True when the value carries no discretisation or sampling error: a closed form the route has proven
    #: exact here, or a claim already decided (a fixed ledger, an outcome decided on the known spot). A finite
    #: difference of exact values is exact as a MOVE; any other value carries its own discretisation or sampling error.
    exact: bool = False

    def __post_init__(self):
        object.__setattr__(self, "numerical", MappingProxyType(dict(self.numerical)))
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))
        object.__setattr__(self, "records", tuple(self.records))


@dataclass(frozen=True)
class PointGreeks:
    """Spot derivatives of the conditional price function at the valuation spot.

    ``status`` and ``reason`` apply to both outputs unless ``statuses`` / ``reasons`` override one:
    a finite delta survives a gamma that could not be computed.
    """

    delta: Optional[float]
    gamma: Optional[float]
    status: str                   # "ok" | "undefined" | "failed"
    reason: str
    evidence: str                 # "kernel_derivative" | "closed_form" | "closed_form_fd" | "grid_stencil" | "paired_rqmc" | "terminated"
    uncertainty: Mapping[str, float] = field(default_factory=dict)
    statuses: Mapping[str, str] = field(default_factory=dict)
    reasons: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "uncertainty", MappingProxyType(dict(self.uncertainty)))
        object.__setattr__(self, "statuses", MappingProxyType(dict(self.statuses)))
        object.__setattr__(self, "reasons", MappingProxyType(dict(self.reasons)))
        for name in ("delta", "gamma"):
            if self.status_of(name) != "ok":
                object.__setattr__(self, name, None)

    def status_of(self, name: str) -> str:
        return self.statuses.get(name, self.status)

    def reason_of(self, name: str) -> str:
        return self.reasons.get(name, self.reason)


def point_greeks_from_estimates(delta: float, gamma: float, evidence: str, *, note: str = "",
                                uncertainty: Optional[Mapping[str, float]] = None) -> PointGreeks:
    """Validate each estimate on its own: a finite one is ``ok``, a non-finite one is ``failed``.

    ``undefined`` is never inferred from a non-finite number here. A payoff discontinuity at the query
    spot is established separately, before the route is asked (``greeks.discontinuity_at_spot``).
    """
    from math import isfinite
    statuses, reasons = {}, {}
    for name, value in (("delta", delta), ("gamma", gamma)):
        if not isfinite(value):
            statuses[name] = "failed"
            reasons[name] = f"non-finite {name} estimate ({value!r}); no discontinuity is established at the query spot"
    return PointGreeks(delta, gamma, "ok", note, evidence, uncertainty or {}, statuses, reasons)


TERMINATED_POINT_GREEKS = PointGreeks(0.0, 0.0, "ok", "", "terminated")


#: Relative spot step of the point-Greek stencil on a claim decided at the valuation instant.
DECIDED_STENCIL = 1e-6
_DECIDED_REASON = "every remaining event is at the valuation instant: decided on the spot"


def decides_at_valuation(product) -> bool:
    """The autocallables whose maturity close the routes resolve themselves: the snowball, the Phoenix and the
    knock-out-reset snowball, matched exactly (a subclass does not inherit the resolution). The Monte Carlo and PDE
    routes also serve barrier, one-touch and digital claims, which are not resolved here."""
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
    from quantark.asset.equity.product.option.snowball_option import SnowballOption

    return type(product) in (SnowballOption, PhoenixOption, KnockOutResetSnowballOption)


def _ko_reset(product) -> bool:
    from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption

    return type(product) is KnockOutResetSnowballOption


def decided_at_valuation(ctx) -> bool:
    """Every remaining event sits at the valuation instant (the maturity close) and has not happened yet.

    No route may hand this state to its day-level engine: those engines' zero-maturity shortcuts mean "the lifecycle
    has already processed today", which is the AFTER phase, and under BEFORE the last observation is still pending.
    Monte Carlo priced a knocked-in claim at its rebate that way, and the PDE solver a knocked-out one (2026-09-19).
    """
    from quantark.intraday.events import EventPhase

    return ctx.numerical.maturity_tau == 0.0 and ctx.phase is EventPhase.BEFORE and not ctx.numerical.terminated


def decided_value(ctx) -> float:
    """The contingent value of a claim whose every remaining event is decided on the known spot now.

    It is what the lifecycle books when those events are fixed at the spot: the pending ledger they leave plus the
    cash paid at the instant, less the ledger already pending -- the identity that relates BEFORE and AFTER at every
    other fixing instant. The route resolves the decided claim through the runtime's own event resolution instead of
    re-stating the payoff, so every route agrees by construction. The ledger is discounted on this context's
    market, so a bumped cell prices its own. The knock-out-reset snowball is the exception: the runtime cannot
    replay it, so its decided claim is read off the twin (``_decided_value_ko_reset``).
    """
    import dataclasses

    if _ko_reset(ctx.numerical.product):
        return _decided_value_ko_reset(ctx)

    from quantark.asset.equity.engine.settlement_support import pending_receivable_pv
    from quantark.intraday.context import resolve_context
    from quantark.intraday.events import EventPhase
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


def _ko_reset_decision(ctx):
    state = ctx.numerical.lifecycle_state
    return ctx.numerical.product.decide_observations_at_valuation(
        float(ctx.spot), ctx.pricing_env, knocked_in=bool(getattr(state, "knocked_in", False)))


def _decided_value_ko_reset(ctx) -> float:
    """The knock-out-reset snowball has no lifecycle replay in the runtime (the daily tracker observes only its
    pre-KI schedule), so the BEFORE / AFTER identity above cannot be evaluated for it. The decided claim is read
    off the float-time twin instead: its own decision on the schedule the knock-in state puts in force, paid on the
    twin's payment times, which are the timeline's."""
    from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing

    twin, env = ctx.numerical.product, ctx.pricing_env
    decision = _ko_reset_decision(ctx)

    def discounted(cash, payment_time):
        if payment_time is not None and payment_time > 0.0:
            return float(cash) * float(env.get_discount_factor(payment_time))
        return float(cash)

    if decision.knocked_out:
        return discounted(decision.ko_record.payoff, decision.ko_record.settlement_time)
    redemption = twin.get_payoff(float(ctx.spot), env, knocked_in=decision.knocked_in)
    return discounted(redemption, resolve_terminal_timing(twin, env).payment_time)


def decided_price_outcome(ctx) -> EnginePriceOutcome:
    return EnginePriceOutcome(decided_value(ctx), "decided_at_valuation", {"reason": _DECIDED_REASON}, {}, exact=True)


def decided_point_greeks(ctx) -> PointGreeks:
    """Central differences of the exact decided value at h = 1e-6 S; undefined when the stencil reaches the
    level of an event decided at this instant (one side of it knocks, the other does not). At a payoff kink
    (the strike of a knocked-in claim) the difference is the average of the one-sided slopes."""
    from copy import deepcopy

    from quantark.intraday.greeks import with_pricing_env

    spot = float(ctx.spot)
    h = DECIDED_STENCIL * spot
    for event in ctx.numerical.remaining_events:
        if event.barrier is not None and abs(spot - float(event.barrier)) <= h:
            return PointGreeks(None, None, "undefined",
                               f"the difference stencil reaches the level {float(event.barrier):g} of "
                               f"{event.event_id}, decided at this instant", "decided_at_valuation")

    def at(s):
        env = deepcopy(ctx.pricing_env)
        env.spot_quote.spot = s
        return decided_value(with_pricing_env(ctx, env, f"decided_spot:{s!r}"))

    base, up, down = decided_value(ctx), at(spot + h), at(spot - h)
    return point_greeks_from_estimates((up - down) / (2.0 * h), (up - 2.0 * base + down) / (h * h),
                                       "decided_at_valuation")


class IntradayEngineRoute(Protocol):
    def price(self, ctx, engine) -> EnginePriceOutcome: ...
