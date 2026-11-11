"""The engine fold contract and the shared folded trade value (ship spec 2026-10-21 D11, D13, Contracts)."""
from copy import deepcopy

import numpy as np
import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures.greeks.theta_shift import (
    ThetaShift,
    bumped_trade_value,
    bumped_trade_value_folded,
)
from quantark.cashleg import (
    DeterministicLeg,
    EventDistribution,
    EventType,
    FixedPayoffLeg,
    FoldMode,
    FoldOverlay,
    LegDirection,
    PaymentTrigger,
    PricingResult,
    TradeGreeks,
)
from quantark.cashleg.folding import _event_structure, folded_trade_value
from quantark.execution.errors import CapabilityError
from quantark.util.enum import OptionType

from test_cashleg._autocallable_helpers import make_env, make_snowball
from test_cashleg.test_folding import _accrual, _autocallable_leg


class _WeightsEngine(BaseEngine):
    """An engine whose event weights are fixed: the event route and the folded route must agree."""

    supports_cash_leg_folding = True

    def __init__(self, npv=950_000.0):
        super().__init__(None)
        self.npv = npv
        self.solves = 0

    def _weights(self, product, env):
        n = len(_event_structure(product, env).ko)
        p_ko = np.linspace(0.05, 0.15, n)
        p_ki = 0.2
        return p_ko, 1.0 - p_ko.sum() - p_ki, p_ki

    def price(self, product, pricing_env, **_):
        self.solves += 1
        return self.npv * pricing_env.get_discount_factor(1.0)

    def price_with_events(self, product, pricing_env, emit_distribution=True, streams=None, **_):
        dist = _event_structure(product, pricing_env).distribution(*self._weights(product, pricing_env))
        return PricingResult(npv=self.price(product, pricing_env), event_distribution=dist)

    def price_folded(self, product, pricing_env, overlay, *, lifecycle_state=None):
        if overlay.is_empty:
            return self.price(product, pricing_env)
        s = _event_structure(product, pricing_env)
        amounts = overlay.bind_ko([(x.schedule, x.determination_time, x.payment_time) for x in s.ko])
        p_ko, p_no, p_ki = self._weights(product, pricing_env)
        df = pricing_env.get_discount_factor
        extra = sum(p * a * df(x.payment_time) for p, a, x in zip(p_ko, amounts, s.ko))
        extra += p_no * overlay.bind_terminal("no_ki", s.no_ki.determination_time, s.no_ki.payment_time) \
            * df(s.no_ki.payment_time)
        extra += p_ki * overlay.bind_terminal("ki", s.ki.determination_time, s.ki.payment_time) \
            * df(s.ki.payment_time)
        return self.price(product, pricing_env) + extra


def _legs(product, env):
    s = _event_structure(product, env)
    times = s.distribution(np.zeros(len(s.ko)), 1.0, 0.0).stream_times[EventType.KO]
    return [
        DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=5_000.0, payment_time=0.1, name="premium"),
        _accrual(),
        FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=7_000.0, trigger=PaymentTrigger.AT_KO),
        _autocallable_leg(times),
    ]


def _event_route(engine, product, env, legs, q):
    result = engine.price_with_events(product, env)
    per_leg = {leg.leg_id: leg.value(result.event_distribution, env, 100.0) for leg in legs}
    return result.npv, per_leg, q * result.npv + sum(per_leg.values())


def test_base_engine_refuses_to_fold():
    assert BaseEngine.supports_cash_leg_folding is False
    option = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    overlay = FoldOverlay(ko=(), maturity_no_ki=None, maturity_ki=None)
    with pytest.raises(CapabilityError, match="BlackScholesEngine"):
        BlackScholesEngine().price_folded(option, make_env(), overlay)


def test_trade_greeks_gain_empty_leg_greeks_by_default():
    assert TradeGreeks(product={}, total={}, leg_pvs={}).leg_greeks == {}


@pytest.mark.parametrize("q", [1.0, -2.5])
def test_total_mode_is_one_folded_solve_matching_the_event_route(q):
    env, product = make_env(), make_snowball(ko_dates=(0.25, 0.5, 0.75, 1.0))
    engine, legs = _WeightsEngine(), _legs(product, env)
    _, _, expected = _event_route(engine, product, env, legs, q)
    engine.solves = 0
    value = folded_trade_value(engine, product, env, legs, quantity=q, unit_notional=100.0,
                               mode=FoldMode.TOTAL)
    assert engine.solves == 1
    assert value.npv is None and value.per_leg == {}
    assert value.total == pytest.approx(expected, rel=1e-12)


def test_drilldown_reports_product_and_each_leg():
    env, product = make_env(), make_snowball(ko_dates=(0.25, 0.5, 0.75, 1.0))
    engine, legs, q = _WeightsEngine(), _legs(product, env), 2.0
    npv, per_leg, expected = _event_route(engine, product, env, legs, q)
    engine.solves = 0
    value = folded_trade_value(engine, product, env, legs, quantity=q, unit_notional=100.0,
                               mode=FoldMode.DRILLDOWN)
    assert engine.solves == 2 + 3                     # plain + total + one per event-dependent leg
    assert value.npv == npv
    assert value.total == pytest.approx(expected, rel=1e-12)
    assert value.per_leg.keys() == per_leg.keys()
    for leg_id, pv in per_leg.items():
        assert value.per_leg[leg_id] == pytest.approx(pv, rel=1e-10, abs=1e-6)
    assert q * value.npv + sum(value.per_leg.values()) == pytest.approx(value.total, rel=1e-12)


def test_event_independent_legs_alone_cost_one_plain_solve():
    env, product = make_env(), make_snowball()
    engine = _WeightsEngine()
    legs = [DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=5.0, payment_time=0.2)]
    value = folded_trade_value(engine, product, env, legs, quantity=1.0, unit_notional=100.0,
                               mode=FoldMode.TOTAL)
    assert engine.solves == 1
    assert value.total == pytest.approx(engine.price(product, env) - 5.0 * env.get_discount_factor(0.2),
                                        rel=1e-14)


# --- D13: theta branches -------------------------------------------------------------------

def _shift(product, env, **kw):
    return ThetaShift(deepcopy(product), deepcopy(env), **kw)


def test_theta_settled_value_with_legs_refuses():
    env, product = make_env(), make_snowball()
    shift = _shift(product, env, settled_value=10.0)
    with pytest.raises(CapabilityError):
        bumped_trade_value_folded(_WeightsEngine(), shift, _legs(product, env), quantity=1.0,
                                  unit_of=lambda e: 100.0, mode=FoldMode.TOTAL)


@pytest.mark.parametrize("mode", [FoldMode.TOTAL, FoldMode.DRILLDOWN])
def test_theta_live_branch_matches_the_event_route(mode):
    env, product = make_env(), make_snowball(ko_dates=(0.25, 0.5, 0.75, 1.0))
    legs, q = _legs(product, env), -1.5
    shift = _shift(product, env, fixed_pv=12.0, time_bump=0.0)   # product unshifted, so the legs stay put
    npv, leg_sum = bumped_trade_value(_WeightsEngine(), shift, legs, streams=None, unit_of=lambda e: 100.0)
    value = bumped_trade_value_folded(_WeightsEngine(), shift, legs, quantity=q,
                                      unit_of=lambda e: 100.0, mode=mode)
    assert value.total == pytest.approx(q * npv + leg_sum, rel=1e-12)
    if mode is FoldMode.DRILLDOWN:
        assert value.npv == npv
        assert sum(value.per_leg.values()) == pytest.approx(leg_sum, rel=1e-10)


def test_theta_ended_branch_reuses_the_event_route():
    env, product = make_env(), make_snowball(ko_dates=(0.25, 0.5, 0.75, 1.0))
    legs, q = _legs(product, env), 2.0
    shift = _shift(product, env, time_bump=1.0 / 365.0,
                   ended_at_t0=(product, env, 0.5, 1.0 / 365.0))
    npv, leg_sum = bumped_trade_value(_WeightsEngine(), shift, legs, streams=None, unit_of=lambda e: 100.0)
    value = bumped_trade_value_folded(_WeightsEngine(), shift, legs, quantity=q,
                                      unit_of=lambda e: 100.0, mode=FoldMode.DRILLDOWN)
    assert value.total == q * npv + leg_sum
    assert value.npv == npv
    assert sum(value.per_leg.values()) == pytest.approx(leg_sum, rel=1e-12)
