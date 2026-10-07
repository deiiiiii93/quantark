"""A contract whose monitoring has ended (spec 3.3): every engine prices the remaining contract."""
import pytest

from pde_routes.cases import env_const
from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.product.option import (BarrierOption, DoubleBarrierOption, DoubleOneTouchOption,
                                                  DoubleSharkfinOption, EuropeanVanillaOption, OneTouchOption,
                                                  SingleSharkfinOption)
from quantark.util.enum import (BarrierDirection, BarrierType, DoubleBarrierType, ObservationType, OptionType,
                                TouchType)

D = 1 / 252
T = 0.5
TB = T - 2 * D
DISC = ObservationType.DISCRETE


def ended(p):
    p.time_shift(2 * D, None, env_const(), keep_landing=True)     # the only record was at D: monitoring has ended
    return p


def vanilla(strike, call, multiplier=1.0):
    return float(BlackScholesEngine().price(
        EuropeanVanillaOption(strike=strike, option_type=OptionType.CALL if call else OptionType.PUT, maturity=TB,
                              contract_multiplier=multiplier), env_const()))


def df():
    return float(env_const().get_discount_factor(TB))


CASES = {
    "uo_call": (lambda: ended(BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0,
                                            barrier_type=BarrierType.UP_OUT, maturity=T, rebate=5.0,
                                            participation_rate=0.8, contract_multiplier=10.0,
                                            observation_type=DISC, observation_dates=[D])),
                lambda: 0.8 * vanilla(100.0, True, 10.0)),
    "ui_call": (lambda: ended(BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0,
                                            barrier_type=BarrierType.UP_IN, maturity=T, rebate=5.0,
                                            observation_type=DISC, observation_dates=[D])),
                lambda: 5.0 * df()),
    "dko_put": (lambda: ended(DoubleBarrierOption(100.0, OptionType.PUT, 130.0, 80.0, DoubleBarrierType.KNOCK_OUT,
                                                  maturity=T, rebate=5.0, observation_type=DISC,
                                                  observation_dates=[D])),
                lambda: vanilla(100.0, False)),
    "dki_call": (lambda: ended(DoubleBarrierOption(100.0, OptionType.CALL, 130.0, 80.0, DoubleBarrierType.KNOCK_IN,
                                                   maturity=T, rebate=5.0, observation_type=DISC,
                                                   observation_dates=[D])),
                 lambda: 5.0 * df()),
    "one_touch": (lambda: ended(OneTouchOption(barrier=120.0, barrier_direction=BarrierDirection.UP, maturity=T,
                                               rebate=5.0, touch_type=TouchType.ONE_TOUCH, observation_type=DISC,
                                               observation_dates=[D])),
                  lambda: 0.0),
    "no_touch": (lambda: ended(OneTouchOption(barrier=120.0, barrier_direction=BarrierDirection.UP, maturity=T,
                                              rebate=5.0, touch_type=TouchType.NO_TOUCH, observation_type=DISC,
                                              observation_dates=[D])),
                 lambda: 5.0 * df()),
    "double_no_touch": (lambda: ended(DoubleOneTouchOption(upper_barrier=130.0, lower_barrier=80.0, maturity=T,
                                                           rebate=5.0, touch_type=TouchType.DOUBLE_NO_TOUCH,
                                                           observation_type=DISC, observation_dates=[D])),
                        lambda: 5.0 * df()),
    "double_one_touch": (lambda: ended(DoubleOneTouchOption(upper_barrier=130.0, lower_barrier=80.0, maturity=T,
                                                            rebate=5.0, touch_type=TouchType.DOUBLE_ONE_TOUCH,
                                                            observation_type=DISC, observation_dates=[D])),
                         lambda: 0.0),
    "sharkfin_call": (lambda: ended(SingleSharkfinOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0,
                                                         maturity=T, participation_rate=0.8, knock_out_rebate=5.0,
                                                         no_hit_rebate=3.0, observation_type=DISC,
                                                         observation_dates=[D])),
                      lambda: 0.8 * (vanilla(100.0, True) - vanilla(130.0, True)) + 3.0 * df()),
    "double_sharkfin_put": (lambda: ended(DoubleSharkfinOption(strike=100.0, option_type=OptionType.PUT,
                                                               lower_barrier=80.0, upper_barrier=130.0, maturity=T,
                                                               participation_rate=0.8, knock_out_rebate=5.0,
                                                               no_hit_rebate=3.0, observation_type=DISC,
                                                               observation_dates=[D])),
                            lambda: 0.8 * (vanilla(100.0, False) - vanilla(80.0, False)) + 3.0 * df()),
}


def engines_for(name):
    from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
    from quantark.asset.equity.engine.analytical.double_barrier_option_engine import DoubleBarrierOptionAnalyticalEngine
    from quantark.asset.equity.engine.analytical.double_sharkfin_option_analytical_engine import (
        DoubleSharkfinOptionAnalyticalEngine)
    from quantark.asset.equity.engine.analytical.one_touch_analytical_engine import OneTouchAnalyticalEngine
    from quantark.asset.equity.engine.analytical.single_sharkfin_option_analytical_engine import (
        SingleSharkfinOptionAnalyticalEngine)
    from quantark.asset.equity.engine.mc.barrier_option_mc_engine import BarrierOptionMCEngine
    from quantark.asset.equity.engine.mc.double_sharkfin_option_mc_engine import DoubleSharkfinOptionMCEngine
    from quantark.asset.equity.engine.mc.single_sharkfin_option_mc_engine import SingleSharkfinOptionMCEngine
    from quantark.asset.equity.param import MCParams
    mc = MCParams(num_paths=2048, time_steps=4, seed=7)
    return {
        "uo_call": [BarrierAnalyticalEngine(), BarrierOptionMCEngine(params=mc)],
        "ui_call": [BarrierAnalyticalEngine(), BarrierOptionMCEngine(params=mc)],
        "dko_put": [DoubleBarrierOptionAnalyticalEngine()],
        "dki_call": [DoubleBarrierOptionAnalyticalEngine()],
        "one_touch": [OneTouchAnalyticalEngine()],
        "no_touch": [OneTouchAnalyticalEngine()],
        "double_no_touch": [],
        "double_one_touch": [],
        "sharkfin_call": [SingleSharkfinOptionAnalyticalEngine(), SingleSharkfinOptionMCEngine(params=mc)],
        "double_sharkfin_put": [DoubleSharkfinOptionAnalyticalEngine(), DoubleSharkfinOptionMCEngine(params=mc)],
    }[name]


@pytest.mark.parametrize("name", sorted(CASES))
def test_ended_monitoring_prices_the_remaining_contract(name):
    make, truth = CASES[name]
    for engine in engines_for(name):
        assert float(engine.price(make(), env_const())) == pytest.approx(truth(), abs=1e-10), type(engine).__name__


def test_remaining_legs_of_a_knocked_in_barrier_are_the_vanilla():
    from quantark.asset.equity.engine.monitoring_ended import remaining_replication
    make, _ = CASES["ui_call"]
    assert remaining_replication(make(), knocked_in=True).price(env_const()) == pytest.approx(vanilla(100.0, True),
                                                                                          abs=1e-10)


def test_ended_accumulator_prices_its_realized_accrual():
    from quantark.asset.equity.engine.analytical.accumulator_analytical_engine import AccumulatorAnalyticalEngine
    from quantark.asset.equity.engine.mc.accumulator_mc_engine import AccumulatorMCEngine
    from quantark.asset.equity.param import MCParams
    from quantark.asset.equity.product.option import AccumulatorOption

    def make():
        p = AccumulatorOption(strike=95.0, knock_out_barrier=105.0, initial_price=100.0, maturity=T,
                              option_type=OptionType.CALL, notional=1_000_000.0, observation_type=DISC,
                              observation_dates=[D], past_observations=[(-D, 98.0)])
        return ended(p)
    a = float(AccumulatorAnalyticalEngine().price(make(), env_const()))
    m = float(AccumulatorMCEngine(params=MCParams(num_paths=1024, time_steps=4, seed=3)).price(make(), env_const()))
    assert a == pytest.approx(m, abs=1e-8)
