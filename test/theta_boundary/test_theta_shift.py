"""Theta: the landing record kept, the as-of outcome carried, an ended schedule priced (spec R1, R3, R4)."""
import pytest

from pde_routes.cases import env_const
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.asset.equity.riskmeasures.greeks.theta_shift import bumped_value, shift_for_theta
from quantark.util.enum import BarrierType, ObservationType, OptionType

D = 1 / 252
DISC = ObservationType.DISCRETE


def theta(product, env, engine):
    return GreeksCalculator().calculate(product=product, pricing_env=env, engine=engine, greeks=["theta"])["theta"]


def bumped(product, env, engine, time_bump=D):
    from quantark.asset.equity.riskmeasures.greeks import bump_envs
    bumped_date, tb, _ = bump_envs.advance_theta_bump(env, 1, "calendar_days")
    return shift_for_theta(product, env, tb, bumped_date), tb


def accumulator(dates, past=None):
    from quantark.asset.equity.product.option import AccumulatorOption
    return AccumulatorOption(strike=95.0, knock_out_barrier=105.0, initial_price=100.0, maturity=dates[-1],
                             option_type=OptionType.CALL, notional=1_000_000.0, observation_type=DISC,
                             observation_dates=list(dates), past_observations=past)


def test_theta_is_the_bumped_value_less_the_base():
    from quantark.asset.equity.engine.analytical.accumulator_analytical_engine import AccumulatorAnalyticalEngine
    eng, env = AccumulatorAnalyticalEngine(), env_const()
    p = accumulator([k / 365 for k in range(1, 64)])          # calendar-day bump lands on the first record
    shift, _ = bumped(p, env, eng)
    assert shift.product.observation_schedule.records[0].observation_time == 0.0
    assert theta(p, env, eng) == pytest.approx(bumped_value(eng, shift) - float(eng.price(p, env)), abs=1e-9)


def test_accumulator_theta_no_longer_drops_an_observation():
    """The old bump dropped the observation landing on t1; now it is kept and settled on the spot. Measured on this
    env: V0 472,997, old V1 492,393 (theta +19,397), new V1 542,393 (theta +69,397) = old + (100 - 95) x 10,000."""
    from copy import deepcopy
    from quantark.asset.equity.engine.analytical.accumulator_analytical_engine import AccumulatorAnalyticalEngine
    eng, env = AccumulatorAnalyticalEngine(), env_const()
    p = accumulator([k / 365 for k in range(1, 64)])
    shift, tb = bumped(p, env, eng)
    dropped = deepcopy(p)
    dropped.time_shift(tb, shift.env.valuation_date, shift.env, keep_landing=False)
    assert len(dropped.observation_schedule.records) == len(shift.product.observation_schedule.records) - 1
    kept_less_dropped = bumped_value(eng, shift) - float(eng.price(dropped, shift.env))
    assert kept_less_dropped == pytest.approx(p.get_observation_payoff(env.spot), rel=1e-12)


def test_accumulator_t0_fixing_is_cash_not_a_booked_fixing():
    """Final review I2: the t0 fixing settles on the spot as engine-determined cash; it is no longer appended to
    past_observations (counted at face on both dates, theta ~0); paid at t0 it leaves the value."""
    from quantark.asset.equity.engine.analytical.accumulator_analytical_engine import AccumulatorAnalyticalEngine
    eng, env = AccumulatorAnalyticalEngine(), env_const(98.0)
    p = accumulator([0.0] + [k / 365 for k in range(1, 64)])
    shift, tb = bumped(p, env, eng)
    assert not shift.product.past_observations
    assert shift.fixed_pv == 0.0


def test_snowball_t0_knock_in_is_carried():
    from quantark.asset.equity.engine.pde import SnowballPDESolver
    from quantark.asset.equity.product.option import SnowballOption
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig
    s = SnowballOption(initial_price=100.0, strike=100.0, maturity=1.0,
                       barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_dates=[0.5, 1.0],
                                                    ki_barrier=75.0, ki_observation_dates=[0.0, 0.25]))
    eng, env = SnowballPDESolver(), env_const(70.0)
    shift, _ = bumped(s, env, eng)
    assert shift.product._otc_lifecycle_knocked_in            # the autocallable engines' carried knock-in
    alive = SnowballOption(initial_price=100.0, strike=100.0, maturity=1.0,
                           barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_dates=[0.5, 1.0],
                                                        ki_barrier=75.0, ki_observation_dates=[0.0, 0.25]))
    alive.time_shift(shift.product.maturity and 1.0 - shift.product.maturity, None, env_const(70.0), keep_landing=True)
    assert bumped_value(eng, shift) < float(eng.price(alive, shift.env))      # knocked in: worth less than alive


def test_t0_knock_out_carries_the_receivable():
    from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
    from quantark.asset.equity.product.option import BarrierOption
    p = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT,
                      maturity=1.0, rebate=5.0, observation_type=DISC, observation_dates=[0.0, 0.5, 1.0])
    eng, env = BarrierAnalyticalEngine(), env_const(125.0)
    shift, tb = bumped(p, env, eng)
    assert shift.terminated
    # the rebate is paid at the terminal payment time (no pay_at_hit): a receivable after the bump
    assert bumped_value(eng, shift) == pytest.approx(5.0 * float(shift.env.get_discount_factor(1.0 - tb)), rel=1e-12)


def test_barrier_t0_knock_in_is_carried_as_the_vanilla():
    from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
    from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
    from quantark.asset.equity.product.option import BarrierOption, EuropeanVanillaOption
    p = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_IN,
                      maturity=1.0, rebate=5.0, observation_type=DISC, observation_dates=[0.0, 0.5, 1.0])
    eng, env = BarrierAnalyticalEngine(), env_const(125.0)
    shift, tb = bumped(p, env, eng)
    assert shift.product._otc_lifecycle_knocked_in
    vanilla = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0 - tb)
    assert bumped_value(eng, shift) == pytest.approx(float(BlackScholesEngine().price(vanilla, shift.env)), abs=1e-10)


def test_ended_schedule_theta_is_carry_not_zero():
    from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
    from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
    from quantark.asset.equity.product.option import BarrierOption, EuropeanVanillaOption
    p = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT,
                      maturity=0.5, observation_type=DISC, observation_dates=[0.4 / 365, 0.8 / 365])
    eng, env = BarrierAnalyticalEngine(), env_const()
    shift, tb = bumped(p, env, eng)
    assert shift.product.observation_schedule.monitoring_ended
    vanilla = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=0.5 - tb)
    want = float(BlackScholesEngine().price(vanilla, shift.env)) - float(eng.price(p, env))
    assert theta(p, env, eng) == pytest.approx(want, abs=1e-10)
    assert theta(p, env, eng) != 0.0


def test_range_accrual_records_shift_and_the_landing_fixing_is_decided_on_the_spot():
    from quantark.asset.equity.product.option import RangeAccrualOption
    from quantark.asset.equity.product.option.range_accrual_config import RangeAccrualConfig

    def make():
        return RangeAccrualOption(initial_price=100.0, maturity=0.5,
                                  range_config=RangeAccrualConfig(lower_barrier=90.0, upper_barrier=110.0,
                                                                  accrual_rate=0.1),
                                  observation_times=[k / 365 for k in range(1, 30)])
    p = make()
    p.time_shift(1 / 365, None, env_const(100.0), keep_landing=True)
    recs = p.get_observation_records()
    assert recs[0].observation_time == 0.0 and recs[0].observed_in_range is True
    assert recs[1].observation_time == pytest.approx(1 / 365)
    assert p.maturity == pytest.approx(0.5 - 1 / 365)


def _ko_at_t0():
    from quantark.asset.equity.product.option import BarrierOption
    return BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT,
                         maturity=1.0, rebate=5.0, observation_type=DISC, observation_dates=[0.0, 0.5, 1.0])


def _alive():
    from quantark.asset.equity.product.option import BarrierOption
    return BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT,
                         maturity=1.0, observation_type=DISC, observation_dates=[0.25, 0.5, 0.75, 1.0])


def test_a_contract_ended_at_t0_has_no_spot_or_vol_greeks_at_t1():
    """t0 knocked it out: at t1 it is a fixed receivable, so charm / color / vega decay are minus the base greeks."""
    from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
    from quantark.asset.equity.riskmeasures.greeks import numerical as nm
    calc, eng, env_ko = GreeksCalculator(), BarrierAnalyticalEngine(), env_const(125.0)
    p = _ko_at_t0()
    _, delta, gamma = nm.get_delta_gamma(calc, p, env_ko, eng, float(eng.price(p, env_ko)))
    vega = nm.numerical_vega(calc, p, env_ko, eng)
    assert nm.numerical_charm(calc, p, env_ko, eng) == pytest.approx(-delta, abs=1e-12)
    assert nm.numerical_color(calc, p, env_ko, eng) == pytest.approx(-gamma, abs=1e-12)
    assert nm.numerical_vega_theta(calc, p, env_ko, eng) == pytest.approx(-vega, abs=1e-12)


def test_the_carried_receivable_has_no_gamma_or_vega():
    """A t0 coupon paid after t1 rides in the advanced price, never in the advanced gamma or vega bumps."""
    from dataclasses import replace
    from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
    from quantark.asset.equity.riskmeasures.greeks import numerical as nm
    calc, eng, env = GreeksCalculator(), BarrierAnalyticalEngine(), env_const()
    p = _alive()
    plain = {}
    color, veta = (nm.numerical_color(calc, p, env, eng, memo=plain),
                   nm.numerical_vega_theta(calc, p, env, eng, memo=plain))
    seeded = {}
    scenario = dict(nm.time_scenario(calc, p, env, None, seeded))
    scenario["shift"] = replace(scenario["shift"], fixed_pv=1_000.0)
    seeded["time_adv:bare"] = scenario
    assert nm.numerical_theta_from_scenario(calc, p, env, eng, None, scenario, None, seeded) == pytest.approx(
        nm.numerical_theta(calc, p, env, eng) + 1_000.0, abs=1e-9)
    assert nm.numerical_color(calc, p, env, eng, memo=seeded) == pytest.approx(color, abs=1e-12)
    assert nm.numerical_vega_theta(calc, p, env, eng, memo=seeded) == pytest.approx(veta, abs=1e-12)
