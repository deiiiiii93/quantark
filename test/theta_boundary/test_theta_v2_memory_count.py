"""Theta hands QUAD V2 ONE consistent memory debt when the caller's lifecycle state carries a coupon-memory count.

QUAD V2 takes the arrears owed at the pricing date from either carrier: a lifecycle state's COUNT of missed periods,
or the contract's AMOUNT (``initial_coupon_arrears``); with both set they must state the same debt (spec D24, T4b).
Theta decides the as-of observation on the spot (``decide_observations_at_valuation``) from the contract's amount
alone, so with the debt on the count it (a) carried only the period missed now into the bumped contract's amount
while the state still said k -- two carriers disagreeing, which V2 refuses since T4b -- and (b) on a coupon paid now
left the count stale (the released arrears paid again later) and paid no arrears in the carried receivable.

Now the decision starts from the count's debt and the bumped valuation receives it on ONE carrier: the amount on the
shifted contract, the state's count released. Independent checks: the same trade stated by its amount alone (the
carrier QUAD V1 reads) has the same theta on QUAD V1, within twice the AC28 V1-V2 price allowance (theta is a
difference of two prices); and V2's bumped value equals V2 priced directly on the hand-built t1 contract plus the
hand-computed receivable.
"""

from copy import deepcopy
from datetime import datetime

import pytest

from quantark.asset.equity.engine.quad import PhoenixQuadEngineV2
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.lifecycle import AutocallableLifecycleState
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option import PhoenixOption
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.riskmeasures.greeks import bump_envs
from quantark.asset.equity.riskmeasures.greeks.theta_shift import bumped_value, shift_for_theta
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.exceptions import ValidationError

#: the as-of observation and four later ones; the coupon is 0.12 x 0.25 x 100 = 3.0 a period (fixed fraction)
OBS = [0.0, 0.25, 0.5, 0.75, 1.0]
COUPON = 3.0
COUNT = 2                                   # two periods missed before the as-of date: 6.0 owed
OWED = COUNT * COUPON
LAG = 0.1
#: the AC28 V1-V2 price allowance at V1 grid 4001 (test_phoenix_quad_v2_arrears.V1_PV_ALLOWANCE), twice: a theta
#: is the difference of two prices, each within it
GRID_POINTS = 4001
THETA_ALLOWANCE = 2 * 1.5e-4
EXACT = 1e-10

#: the coupon barrier is 95 on every record, the knock-out 110: the as-of spot pays the coupon now (100) or misses it (90)
COUPON_BARRIER = 95.0
PAID_NOW, MISSED_NOW = 100.0, 90.0


def _env(spot=PAID_NOW):
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=0.25),
                              rate_curve=FlatRateCurve(rate=0.05), div_yield=ContinuousDividendYield(div_yield=0.01),
                              valuation_date=datetime(2026, 1, 1))


def _phoenix(pay_type=CouponPayType.INSTANT, arrears=0.0, obs=OBS, maturity=1.0):
    return PhoenixOption(
        initial_price=100.0, strike=100.0, maturity=maturity,
        barrier_config=BarrierConfig(ko_barrier=110.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(obs), ki_barrier=70.0,
                                     ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True),
        coupon_config=CouponBarrierConfig(coupon_barrier=COUPON_BARRIER, coupon_rate=0.12, coupon_pay_type=pay_type,
                                          memory_coupon=True, fixed_coupon_year_fraction=0.25,
                                          initial_coupon_arrears=arrears),
        payoff_config=PayoffConfig(include_principal=True),
        settlement_convention=SettlementConvention(lag=LAG, lag_unit=SettlementLagUnit.YEAR_FRACTION))


def _count():
    return AutocallableLifecycleState(coupon_memory_count=COUNT)


def _step(env):
    bumped_date, time_bump, _ = bump_envs.advance_theta_bump(env, 1, "calendar_days")
    return time_bump, bumped_date


def _theta(engine, product, env, state=None):
    time_bump, bumped_date = _step(env)
    shift = shift_for_theta(product, env, time_bump, bumped_date, lifecycle_state=state)
    base = engine.price(product, env) if state is None else engine.price(product, env, lifecycle_state=state)
    return bumped_value(engine, shift) - float(base), shift


CASES = [
    pytest.param(MISSED_NOW, CouponPayType.INSTANT, id="missed-now-instant"),
    pytest.param(MISSED_NOW, CouponPayType.EXPIRY, id="missed-now-expiry"),
    pytest.param(PAID_NOW, CouponPayType.INSTANT, id="paid-now-instant"),
]


@pytest.mark.parametrize("spot, pay_type", CASES)
def test_v2_theta_on_a_count_is_the_theta_of_the_same_debt_stated_by_its_amount_on_v1(spot, pay_type):
    env = _env(spot)
    on_count, _ = _theta(PhoenixQuadEngineV2(), _phoenix(pay_type), env, _count())
    on_amount, _ = _theta(PhoenixQuadEngine(params=QuadParams(grid_points=GRID_POINTS)),
                          _phoenix(pay_type, arrears=OWED), env)
    assert abs(on_count - on_amount) <= THETA_ALLOWANCE, f"V2 (count) {on_count:+.6e} vs V1 (amount) {on_amount:+.6e}"


@pytest.mark.parametrize("spot, pay_type", CASES)
def test_v2_bumped_value_is_the_hand_built_t1_contract(spot, pay_type):
    """t1 = t0 + one calendar day: the as-of record gone, the rest moved by the bump. Missed now: 6.0 + 3.0 owed on the
    contract, nothing on the count. Paid now: nothing owed; 3.0 + 6.0 paid at the record's settlement (t0 + 0.1)."""
    env = _env(spot)
    _, shift = _theta(PhoenixQuadEngineV2(), _phoenix(pay_type), env, _count())
    time_bump, bumped_date = _step(env)
    env_t1 = deepcopy(env)
    env_t1.valuation_date = bumped_date
    missed = spot == MISSED_NOW
    t1 = _phoenix(pay_type, arrears=OWED + COUPON if missed else 0.0,
                  obs=[t - time_bump for t in OBS[1:]], maturity=1.0 - time_bump)
    receivable = 0.0 if missed else (COUPON + OWED) * float(env_t1.get_discount_factor(LAG - time_bump))
    expected = float(PhoenixQuadEngineV2().price(t1, env_t1)) + receivable
    assert getattr(shift.lifecycle_state, "coupon_memory_count", 0) == 0
    assert bumped_value(PhoenixQuadEngineV2(), shift) == pytest.approx(expected, abs=EXACT)


def test_a_count_without_an_observation_now_is_carried_as_before():
    """Nothing decided on the as-of date: the caller's state goes on unchanged (its count still carries the debt)."""
    env = _env()
    product = _phoenix(obs=OBS[1:])
    state = _count()
    time_bump, bumped_date = _step(env)
    shift = shift_for_theta(product, env, time_bump, bumped_date, lifecycle_state=state)
    assert shift.lifecycle_state is state
    assert shift.product.coupon_config.initial_coupon_arrears == 0.0


def test_a_count_and_a_different_amount_refuse_in_theta_too():
    env = _env(MISSED_NOW)
    time_bump, bumped_date = _step(env)
    with pytest.raises(ValidationError, match="initial_coupon_arrears"):
        shift_for_theta(_phoenix(arrears=COUPON), env, time_bump, bumped_date, lifecycle_state=_count())
