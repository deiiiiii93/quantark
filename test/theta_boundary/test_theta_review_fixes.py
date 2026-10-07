"""Final-review fixes to the theta step (spec 2026-10-07 R4; user ruling: theta excludes only cash already booked)."""
import pytest

from pde_routes.cases import env_const
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.asset.equity.riskmeasures.greeks import bump_envs
from quantark.asset.equity.riskmeasures.greeks.theta_shift import _carried_to_t1, bumped_value, shift_for_theta
from quantark.execution.errors import CapabilityError
from quantark.util.enum import AccumulatorKnockOutType, ObservationType, OptionType

DISC = ObservationType.DISCRETE


def bump(env):
    bumped_date, tb, _ = bump_envs.advance_theta_bump(env, 1, "calendar_days")
    return bumped_date, tb


def test_dcn_theta_runs():
    """C1: a BaseEquityProduct that is not an option took no keep_landing (TypeError)."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))
    from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn
    from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
    g = GreeksCalculator().calculate(product=make_dcn(DCN_A), pricing_env=flat_env(**FLAT),
                                     engine=DCNMCEngine(num_paths=2000, seed=42), greeks=["theta"])
    assert g["theta"] == g["theta"]                       # finite, no TypeError


def _accumulator(dates, **kw):
    from quantark.asset.equity.product.option import AccumulatorOption
    return AccumulatorOption(strike=95.0, knock_out_barrier=105.0, initial_price=100.0, maturity=dates[-1],
                             option_type=OptionType.CALL, notional=1_000_000.0, observation_type=DISC,
                             observation_dates=list(dates), **kw)


def test_accumulator_knocked_out_at_t0_is_its_booked_accrual_and_the_rebate_receivable():
    """C2: a TERMINATION breach on a t0 fixing ends the contract; the bumped value is NOT the alive contract."""
    from quantark.asset.equity.engine.analytical.accumulator_analytical_engine import AccumulatorAnalyticalEngine
    past = [(-2 / 365, 99.0), (-1 / 365, 101.0)]
    p = _accumulator([0.0] + [k / 52 for k in range(1, 9)], past_observations=past, knock_out_rebate_rate=0.01)
    eng, env = AccumulatorAnalyticalEngine(), env_const(106.0)
    bd, tb = bump(env)
    shift = shift_for_theta(p, env, tb, bd)
    assert shift.terminated
    booked = p.get_realized_accrual()                      # 4 x 10,000 + 6 x 10,000: settled on observation dates
    rebate = p.get_knock_out_rebate_cash() * p.contract_multiplier
    # the rebate settles on the t0 fixing's own settlement date (t0, no lag): paid within the bump
    assert bumped_value(eng, shift) == pytest.approx(booked + _carried_to_t1(rebate, 0.0, env, shift.env, tb), abs=1e-9)
    assert bumped_value(eng, shift) == pytest.approx(booked, abs=1e-9)


def test_accumulator_t0_fixing_paid_within_the_bump_leaves_the_value():
    """I2: the t0 fixing is engine-determined cash, not booked: paid at t0 it is gone at t1 (no longer carried into
    past_observations, where the engine counted it at face on both dates)."""
    from quantark.asset.equity.engine.analytical.accumulator_analytical_engine import AccumulatorAnalyticalEngine
    later = [k / 52 for k in range(1, 9)]
    eng, env = AccumulatorAnalyticalEngine(), env_const(98.0)
    bd, tb = bump(env)
    shift = shift_for_theta(_accumulator([0.0] + later), env, tb, bd)
    assert not shift.product.past_observations
    alive = _accumulator(later)
    alive.time_shift(tb, bd, shift.env, keep_landing=True)
    assert bumped_value(eng, shift) == pytest.approx(float(eng.price(alive, shift.env)), abs=1e-9)


def test_accumulator_t0_fixing_settled_at_expiry_is_a_receivable():
    from quantark.asset.equity.engine.analytical.accumulator_analytical_engine import AccumulatorAnalyticalEngine
    later = [k / 52 for k in range(1, 9)]
    eng, env = AccumulatorAnalyticalEngine(), env_const(98.0)
    bd, tb = bump(env)
    p = _accumulator([0.0] + later, settlement_at_expiry=True)
    shift = shift_for_theta(p, env, tb, bd)
    T = p.get_maturity(env)
    assert shift.fixed_pv == pytest.approx(p.get_observation_payoff(98.0) * float(shift.env.get_discount_factor(T - tb)),
                                           rel=1e-12)


def test_ko_reset_whose_schedule_1_ends_at_t0_is_a_contract_ended_at_t0():
    """I1: schedule 1 ends at t0, no knock-out, no knock-in: the contract matured at t0 (rule 2); theta raised."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from test_quad_empty_side import make, env
    from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
    from quantark.asset.equity.param import PDEParams
    p = make("ko_reset", (0.0,), (0.0, 0.5, 1.0), (0.25, 0.5, 1.0))
    e = env(90.0)
    bd, tb = bump(e)
    shift = shift_for_theta(p, e, tb, bd)
    assert shift.terminated
    eng = KOResetSnowballPDESolver(PDEParams())
    assert bumped_value(eng, shift) == pytest.approx(0.0, abs=1e-12)       # paid at t0, no settlement lag
    GreeksCalculator().calculate(product=p, pricing_env=e, engine=eng, greeks=["theta"])


def test_phoenix_expiry_coupon_decided_at_t0_is_refused():
    """I3: an EXPIRY-pay coupon rolls up to the note's end (a knock-out or maturity); no state carries it."""
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1]))
    from dataclasses import replace
    from test_autocallable_observation_at_valuation import NOW_AND_LATER, PAYS, env, phoenix
    from quantark.util.enum.option_enums import CouponPayType
    p = phoenix(NOW_AND_LATER)
    p.coupon_config = replace(p.coupon_config, coupon_pay_type=CouponPayType.EXPIRY)
    p.accrual_config = replace(p.accrual_config, coupon_pay_type=CouponPayType.EXPIRY)
    e = env(PAYS)
    bd, tb = bump(e)
    with pytest.raises(CapabilityError, match="EXPIRY"):
        shift_for_theta(p, e, tb, bd)


@pytest.mark.parametrize("residue", [-1e-17, 0.0, 1e-17])
def test_cash_paid_on_t1_is_kept_whatever_the_float_residue(residue):
    """Minor 1 (re-graded): cash paid ON t1 is still in the value (DF 1 at the new as-of); a float residue decided."""
    env = env_const()
    bd, tb = bump(env)
    from copy import deepcopy
    env1 = deepcopy(env); env1.valuation_date = bd
    pv0 = 100.0 * float(env.get_discount_factor(tb))
    assert _carried_to_t1(pv0, tb + residue, env, env1, tb) == pytest.approx(100.0, rel=1e-12)
