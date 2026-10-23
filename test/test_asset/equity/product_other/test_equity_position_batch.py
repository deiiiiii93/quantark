"""EquityPosition.batch_trade_values / batch_trade_risk (ship spec 2026-10-23 C2, D8, D10, AC5).

Element i equals the scalar ``get_trade_value(envs[i])`` / ``get_trade_risk(envs[i], gc)``: on a QUAD V2 engine
the calls run inside one spot-batch session (equal within the V2 tolerance), on every other engine they are the
plain scalar calls (bitwise).
"""
import copy
from datetime import datetime

import pytest

from quantark.asset.equity.engine import BlackScholesEngine
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.engine.quad.v2 import PhoenixQuadEngineV2, SnowballQuadEngineV2
from quantark.asset.equity.param import PDEParams, QuadParams
from quantark.asset.equity.product import EuropeanVanillaOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.cashleg.autocallable_leg import AccrualBasis, AutocallableCashLeg, AutocallableLegType
from quantark.cashleg.base import LegDirection
from quantark.cashleg.event_distribution import EventType
from quantark.cashleg import FoldMode
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio import EquityPosition
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.option_enums import ObservationType

VAL = datetime(2026, 3, 16)
GREEKS = ("price", "delta", "gamma", "vega", "theta", "rho", "dividend_rho")
KO, KI, COUPON_LEVEL = 103.0, 75.0, 80.0
#: valued ON an observation date: t = 0 is decided on each env's own spot
TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)
LAG = SettlementConvention(lag=0.02, lag_unit=SettlementLagUnit.YEAR_FRACTION)


def _env(spot=100.0, vol=0.25, rate=0.03):
    return PricingEnvironment(spot_quote=SpotQuote(spot), vol_surface=FlatVolSurface(vol), rate_curve=FlatRateCurve(rate),
                              div_yield=ContinuousDividendYield(0.01), valuation_date=VAL)


def _schedule(level, times=TIMES):
    return ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=level) for t in times])


def _barrier(ko_rate):
    return BarrierConfig(ko_barrier=KO, ko_rate=ko_rate, ko_observation_type=ObservationType.DISCRETE,
                         ko_observation_schedule=_schedule(KO), ki_barrier=KI,
                         ki_observation_type=ObservationType.DISCRETE, ki_observation_schedule=_schedule(KI))


def _snowball():
    return SnowballOption(initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=10_000.0,
                          barrier_config=_barrier(0.15), settlement_convention=LAG,
                          accrual_config=AccrualConfig(is_annualized=False),
                          payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False))


def _phoenix():
    return PhoenixOption(initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=10_000.0,
                         barrier_config=_barrier(0.0), settlement_convention=LAG,
                         coupon_config=CouponBarrierConfig(coupon_barrier=COUPON_LEVEL, coupon_rate=0.02,
                                                           memory_coupon=True, fixed_coupon_year_fraction=0.25),
                         accrual_config=AccrualConfig(is_annualized=False, is_annualized_coupon=True),
                         payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True))


def _rebate_leg():
    return AutocallableCashLeg(direction=LegDirection.BUYER_RECEIVES, leg_type=AutocallableLegType.REBATE,
                               notional=1_000.0, rate=0.05, terminal_accrual_factor=1.0,
                               terminal_settlement_time=1.0, accrual_basis=AccrualBasis.KO_MATURITY,
                               terminal_events=frozenset({EventType.MATURITY_NO_KO}))


def _position(product, engine, *, quantity=-2.0, legs=()):
    return EquityPosition(product=product, quantity=quantity, entry_price=0.0, underlying="X", engine=engine,
                          entry_timestamp=VAL, cash_legs=list(legs))


#: spots straddling KI (70: knocked in at t0), the coupon level (78 / 90) and KO (105: knocked out at t0),
#: plus vol / rate shocked states at the base spot
ENVS = [_env(s) for s in (70.0, 78.0, 90.0, 100.0, 105.0)] + [_env(vol=0.30), _env(rate=0.04)]


def _scalar(position, envs, gc):
    """The reference: each env on a fresh copy of the position."""
    values = [copy.deepcopy(position).get_trade_value(e) for e in envs]
    risks = [copy.deepcopy(position).get_trade_risk(e, gc).total for e in envs]
    return values, risks


#: pinned ceilings, ~10x the measured batch-vs-scalar gaps at CP1 (PV 5e-12, first order 1e-10, gamma 1e-8
#: relative; default V2 params) and inside AC5's caps (1e-9 / 1e-9 / 1e-6)
PV_REL, FIRST_REL, GAMMA_REL = 1e-10, 1e-9, 1e-7


def _assert_values_close(got, want, notional):
    for g, w in zip(got, want):
        assert abs(g - w) <= PV_REL * max(abs(w), notional), (g, w)


def _assert_close(got, want, notional, spot=100.0):
    """Relative ceilings per greek, with a near-zero floor of 1e-10 x notional in that greek's own units."""
    for name in want:
        rel = {"price": PV_REL, "gamma": GAMMA_REL}.get(name, FIRST_REL)
        unit = {"delta": spot, "gamma": spot ** 2}.get(name, 1.0)
        tol = rel * abs(want[name]) + 1e-10 * notional / unit
        assert abs(got[name] - want[name]) <= tol, (name, got[name], want[name])


@pytest.mark.parametrize("make_product,make_engine", [(_snowball, SnowballQuadEngineV2), (_phoenix, PhoenixQuadEngineV2)])
@pytest.mark.parametrize("legs,fold", [((), None), ((_rebate_leg(),), None)])
def test_quad_v2_batch_matches_scalar_across_branches(make_product, make_engine, legs, fold):
    gc = GreeksCalculator()
    position = _position(make_product(), make_engine(), legs=legs)
    position.cash_leg_folding = fold
    want_values, want_risks = _scalar(position, ENVS, gc)
    batch = copy.deepcopy(position)
    got_values = batch.batch_trade_values(ENVS)
    got_risks = [r.total for r in batch.batch_trade_risk(ENVS, gc)]
    notional = abs(position.get_actual_notional(_env()))
    _assert_values_close(got_values, want_values, notional)
    for g, w in zip(got_risks, want_risks):
        _assert_close(g, w, notional)


@pytest.mark.parametrize("mode", [FoldMode.TOTAL, FoldMode.DRILLDOWN])
def test_quad_v2_batch_with_folded_legs_matches_scalar(mode):
    from test_cashleg._autocallable_helpers import make_env, make_snowball
    from test_cashleg.test_folding import _leg_set, _route_dist

    env, product = make_env(), make_snowball(ko_dates=(0.25, 0.5, 0.75, 1.0))
    legs = _leg_set(_route_dist(SnowballPDESolver(PDEParams()), product, env).event_times)
    gc = GreeksCalculator()
    envs = []
    for f in (0.9, 1.0, 1.1):
        e = copy.deepcopy(env)
        e.spot_quote.spot *= f
        envs.append(e)
    position = _position(product, SnowballQuadEngineV2(), quantity=2.0, legs=legs)
    position.cash_leg_folding = mode
    want_values, want_risks = _scalar(position, envs, gc)
    batch = copy.deepcopy(position)
    notional = abs(position.get_actual_notional(env))
    _assert_values_close(batch.batch_trade_values(envs), want_values, notional)
    for g, w in zip(batch.batch_trade_risk(envs, gc), want_risks):
        _assert_close(g.total, w, notional, spot=env.spot)


@pytest.mark.parametrize("make_engine,product", [
    (lambda: SnowballPDESolver(PDEParams()), _snowball()),
    (lambda: SnowballQuadEngine(params=QuadParams()), _snowball()),
    (BlackScholesEngine, EuropeanVanillaOption(strike=100.0, maturity=0.5, option_type=OptionType.CALL)),
])
def test_other_engines_batch_is_the_scalar_loop(make_engine, product):
    gc = GreeksCalculator()
    position = _position(product, make_engine())
    envs = [_env(s) for s in (90.0, 100.0, 110.0)]
    want_values = [position.get_trade_value(e) for e in envs]
    want_risks = [position.get_trade_risk(e, gc).total for e in envs]
    assert position.batch_trade_values(envs) == want_values
    assert [r.total for r in position.batch_trade_risk(envs, gc)] == want_risks


def test_batch_keeps_order_and_leaves_the_position_alone():
    gc = GreeksCalculator()
    position = _position(_snowball(), SnowballQuadEngineV2(), legs=(_rebate_leg(),))
    product_before, legs_before = copy.deepcopy(position.product), copy.deepcopy(position.cash_legs)
    envs = [_env(s) for s in (110.0, 90.0, 100.0)]
    values = position.batch_trade_values(envs)
    reversed_values = position.batch_trade_values(envs[::-1])
    assert values == reversed_values[::-1] or all(
        abs(a - b) <= 1e-9 * max(abs(b), 1.0) for a, b in zip(values, reversed_values[::-1]))
    assert position.product.__dict__.keys() == product_before.__dict__.keys()
    assert position.product.barrier_config == product_before.barrier_config
    assert len(position.cash_legs) == len(legs_before)
    assert position.engine._batch is None
    assert position.batch_trade_values([]) == [] and position.batch_trade_risk([], gc) == []
