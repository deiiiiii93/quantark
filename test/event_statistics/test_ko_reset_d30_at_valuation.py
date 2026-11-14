"""D30 completion (controller ruling, 2026-11-14): under EXPIRY coupon pay a schedule-1 knock-out DECIDED AT VALUATION
is paid at the D30/D33 first-schedule payment -- the note that ends with schedule 1 -- in the two places that still read
another time:

- theta's at-valuation knock-out (``theta_shift._autocallable_outcome``) read the record's raw settlement (the
  observation plus its lag), so a note whose schedule 1 runs on past the knock-out was "paid within the bump" and its
  ended value dropped to zero; it is paid at ``first_schedule_payment_time`` (schedule 1's maturity plus the delay, or
  the explicit settlement date), as every grid engine pays it;
- Monte Carlo's ``_ko_reset_decided_knock_out`` (reached only at zero time to maturity) paid at the product terminal;
  it pays at the first-schedule payment. At zero time to maturity schedule 1 ends at the product terminal (one
  maturity), so the value is unchanged: the tests pin MC == QUAD V2 == PDE 1D == cash x DF(first-schedule payment), with
  and without a settlement date.

A second-schedule knock-out and INSTANT pay are outside the class and unchanged.
"""
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.quad.v2 import KOResetSnowballQuadEngineV2
from quantark.asset.equity.engine.settlement_support import first_schedule_payment_time
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.riskmeasures.greeks.theta_shift import carried_ended_value, shift_for_theta
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import CouponPayType
from quantark.util.enum.option_enums import ObservationType, PostKOScheduleMode as M

V = datetime(2026, 3, 16)
CASH = 115.0                                          # principal 100 + 15% (not annualized)


def env(spot=110.0, valuation=V):
    return PricingEnvironment(rate_curve=FlatRateCurve(0.03), valuation_date=valuation, spot_quote=SpotQuote(spot),
                              vol_surface=FlatVolSurface(0.25), div_yield=ContinuousDividendYield(0.01))


def _times(times, level):
    return ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=level) for t in times])


def _dates(dates, level):
    return ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=level) for d in dates])


def note(pre, post, *, pay=CouponPayType.EXPIRY, dated=False, settlement_date=None, convention=None):
    sched = _dates if dated else _times
    kw = (dict(initial_date=V - timedelta(days=365), exercise_date=max(post), settlement_date=settlement_date)
          if dated else dict(maturity=max(post)))
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=sched(pre, 103.0), ki_barrier=75.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=sched(pre, 75.0)),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=sched(post, 95.0)),
        accrual_config=AccrualConfig(is_annualized=False, coupon_pay_type=pay),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True), post_ko_mode=M.ABSOLUTE,
        settlement_convention=convention, **kw)


# --- theta ----------------------------------------------------------------------------------------------------------

THETA_CASES = {
    # schedule 1 knocks out at t0 (record at 0) and runs on to 0.5; schedule 2 to 1.0
    "numeric": lambda: note((0.0, 0.25, 0.5), (0.25, 0.5, 0.75, 1.0)),
    "numeric_lag": lambda: note((0.0, 0.25, 0.5), (0.25, 0.5, 0.75, 1.0),
                                convention=SettlementConvention(lag=5 / 365, lag_unit=SettlementLagUnit.YEAR_FRACTION)),
    "dated_settlement_date": lambda: note(
        (V, V + timedelta(days=91), V + timedelta(days=182)),
        (V + timedelta(days=91), V + timedelta(days=182), V + timedelta(days=365)),
        dated=True, settlement_date=V + timedelta(days=372)),
}


@pytest.mark.parametrize("case", tuple(THETA_CASES))
def test_theta_pays_an_at_valuation_schedule_one_knock_out_at_the_first_schedule_payment(case):
    product, e = THETA_CASES[case](), env()
    shift = shift_for_theta(product, e, 1 / 365, V + timedelta(days=1))
    assert shift.ended_at_t0 is not None
    expected = float(first_schedule_payment_time(product, e))
    assert expected > 1 / 365                                      # schedule 1 runs on: not paid within the bump
    assert shift.ended_at_t0[2] == expected
    value_t0 = float(KOResetSnowballQuadEngineV2().price(product, e))
    carried = carried_ended_value(shift, value_t0)
    assert carried > 0.0                                           # base: 0.0 (the record's settlement at t0)


def test_theta_instant_pay_keeps_the_record_settlement():
    product, e = note((0.0, 0.25, 0.5), (0.25, 0.5, 0.75, 1.0), pay=CouponPayType.INSTANT), env()
    shift = shift_for_theta(product, e, 1 / 365, V + timedelta(days=1))
    assert shift.ended_at_t0[2] == 0.0


# --- Monte Carlo at zero time to maturity -----------------------------------------------------------------------------


def _at_its_end(product, e):
    """Age a contract to its last instant (every record kept at time 0)."""
    product.time_shift(float(product.maturity), V, e, keep_landing=True)
    return product


MC_CASES = {
    "numeric": lambda e: _at_its_end(note((0.25,), (0.25,)), e),
    "numeric_lag": lambda e: _at_its_end(note((0.25,), (0.25,),
                                              convention=SettlementConvention(lag=5 / 365, lag_unit=SettlementLagUnit.YEAR_FRACTION)), e),
    "dated_settlement_date": lambda e: note((V,), (V,), dated=True, settlement_date=V + timedelta(days=7)),
}


@pytest.mark.parametrize("case", tuple(MC_CASES))
def test_mc_at_zero_maturity_pays_the_first_schedule_payment_like_the_grid_engines(case):
    e = env()
    product = MC_CASES[case](e)
    expected = CASH * float(e.get_discount_factor(float(first_schedule_payment_time(product, e))))
    mc = float(SnowballMCEngine(params=MCParams(num_paths=64, seed=1)).price(product, e))
    v2 = float(KOResetSnowballQuadEngineV2().price(MC_CASES[case](e), e))
    pde = float(KOResetSnowballPDESolver(PDEParams()).price(MC_CASES[case](e), e))
    assert mc == pytest.approx(expected, rel=1e-13, abs=1e-13)
    assert v2 == pytest.approx(expected, rel=1e-13, abs=1e-13)
    assert pde == pytest.approx(expected, rel=1e-13, abs=1e-13)
