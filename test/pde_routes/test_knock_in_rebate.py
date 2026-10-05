"""A knock-in's rebate (spec 3C decision 8; §7.2 rows 'knock-in rebate, single / double', 'aggregation and the
composite'; Review Focus 3 and 4)."""
import math

import pytest

from pde_routes.barrier_reference import barrier_reference, knock_out
from pde_routes.cases import CONST, barrier, black_scholes, env_const, obs_dates, schedule
from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
from quantark.asset.equity.engine.analytical.double_barrier_option_engine import DoubleBarrierOptionAnalyticalEngine
from quantark.asset.equity.engine.pde import BarrierPDESolver, DoubleBarrierPDESolver
from quantark.asset.equity.product.option import BarrierOption, DoubleBarrierOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.util.enum import BarrierType, DoubleBarrierType, ObservationAggregation, ObservationType, OptionType
from quantark.util.exceptions import ValidationError

VOL, R, Q = CONST["vol"], CONST["rate"], CONST["div"]
M = dict(vol=VOL, r=R, q=Q)


@pytest.mark.parametrize("kind,call,level,spot", [("UI", True, 130.0, 100.0), ("DI", False, 85.0, 100.0),
                                                  ("UI", True, 130.0, 120.0)])
def test_continuous_single_against_the_closed_form(kind, call, level, spot):
    p = barrier(kind, call, level, rebate=5.0)
    exact = BarrierAnalyticalEngine().price(p, env_const(spot))
    assert abs(BarrierPDESolver().price(p, env_const(spot)) - exact) <= 1e-3        # measured 4.9e-4; today -3.75


@pytest.mark.parametrize("kind,call,level", [("UI", True, 130.0), ("DI", False, 85.0)])
def test_monthly_single_against_the_exact_recursion(kind, call, level):
    p = barrier(kind, call, level, obs="monthly", rebate=5.0)
    exact = barrier_reference(kind=kind, call=call, strike=100.0, level=level, rebate=5.0, at_hit=False,
                              obs=obs_dates("monthly"), spots=[100.0], **M)[100.0][0]
    assert abs(BarrierPDESolver().price(p, env_const(100.0)) - exact) <= 1e-3        # today -4.85


def test_in_out_parity_with_the_rebate():
    """knock-in(R) + knock-out(R paid at maturity) = vanilla + R DF(T_pay), path by path."""
    env = env_const(100.0)
    ki, ko = barrier("UI", True, 130.0, rebate=5.0), barrier("UO", True, 130.0, rebate=5.0)
    total = BarrierPDESolver().price(ki, env) + BarrierPDESolver().price(ko, env)
    assert abs(total - (black_scholes(True, 100.0, 100.0, 1.0)[0] + 5.0 * math.exp(-R))) <= 1e-3


@pytest.mark.parametrize("call", [True, False])
def test_continuous_double_against_closed_form_plus_no_touch(call):
    import json
    from pathlib import Path
    nt = {tuple(c["args"]): c["reference"] for c in
          json.loads((Path(__file__).parent / "no_touch_reference.json").read_text())["cells"]["engine"]}
    p0 = DoubleBarrierOption(100.0, OptionType.CALL if call else OptionType.PUT, 130.0, 80.0, DoubleBarrierType.KNOCK_IN,
                             maturity=1.0, rebate=0.0, observation_type=ObservationType.CONTINUOUS)
    p5 = DoubleBarrierOption(100.0, OptionType.CALL if call else OptionType.PUT, 130.0, 80.0, DoubleBarrierType.KNOCK_IN,
                             maturity=1.0, rebate=5.0, observation_type=ObservationType.CONTINUOUS)
    env = env_const(100.0)
    exact = DoubleBarrierOptionAnalyticalEngine().price(p0, env) + 5.0 * math.exp(-R) * nt[(100.0, 1.0, R, Q, VOL, 80.0, 130.0)]
    assert abs(DoubleBarrierPDESolver().price(p5, env) - exact) <= 1e-3               # measured 2.5e-4; today -2.22


def test_monthly_double_against_the_extended_recursion():
    obs = obs_dates("monthly")
    p = DoubleBarrierOption(100.0, OptionType.CALL, 130.0, 80.0, DoubleBarrierType.KNOCK_IN, maturity=1.0, rebate=5.0,
                            observation_type=ObservationType.DISCRETE, observation_dates=obs)
    ko = knock_out(payoff="call", strike=100.0, lower=80.0, upper=130.0, cash=0.0, at_hit=False, obs=obs,
                   spots=[100.0], **M)[0][100.0][0]
    surv = knock_out(payoff="one", strike=100.0, lower=80.0, upper=130.0, cash=0.0, at_hit=False, obs=obs,
                     spots=[100.0], **M)[0][100.0][0]
    exact = black_scholes(True, 100.0, 100.0, 1.0)[0] - ko + 5.0 * surv
    assert abs(DoubleBarrierPDESolver().price(p, env_const(100.0)) - exact) <= 1e-3   # §7.2 'new'


def test_mismatched_record_payoff_raises():
    sch = ObservationSchedule(records=[ObservationRecord(observation_time=0.5, payoff=3.0),
                                       ObservationRecord(observation_time=1.0)])
    p = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=BarrierType.UP_IN,
                      maturity=1.0, rebate=5.0, observation_type=ObservationType.DISCRETE, observation_schedule=sch)
    with pytest.raises(ValidationError, match="one rebate"):
        BarrierPDESolver().price(p, env_const(100.0))


def test_accumulate_knock_in_raises():
    p = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=BarrierType.UP_IN,
                      maturity=1.0, rebate=5.0, observation_type=ObservationType.DISCRETE,
                      observation_schedule=schedule(obs_dates("monthly"), ObservationAggregation.ACCUMULATE))
    with pytest.raises(ValidationError, match="first-hit"):
        BarrierPDESolver().price(p, env_const(126.1))                                 # today -23.67


def test_default_payoff_schedule_is_not_refused():
    """Review Focus 4: records built with the product's rebate as their default payoff, and rebate 0 with payoff 0."""
    for rebate in (5.0, 0.0):
        p = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=BarrierType.UP_IN,
                          maturity=1.0, rebate=rebate, observation_type=ObservationType.DISCRETE,
                          observation_schedule=ObservationSchedule.from_legacy(observation_dates=obs_dates("monthly"),
                                                                               default_barrier=130.0,
                                                                               default_payoff=rebate))
        BarrierPDESolver().price(p, env_const(100.0))


def test_knock_in_decided_by_record_level():
    """Review Focus 3: a valuation-date record at 120 on a knock-in with barrier 130, spot 125: knocked in."""
    sch = ObservationSchedule(records=[ObservationRecord(observation_time=0.0, barrier=120.0, payoff=5.0),
                                       ObservationRecord(observation_time=0.5), ObservationRecord(observation_time=1.0)])
    p = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=BarrierType.UP_IN,
                      maturity=1.0, rebate=5.0, observation_type=ObservationType.DISCRETE, observation_schedule=sch)
    from quantark.asset.equity.engine.pde import EuropeanPDESolver
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    van = EuropeanPDESolver().price(EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0),
                                    env_const(125.0))
    assert BarrierPDESolver().price(p, env_const(125.0)) == van


def test_rebate_free_knock_in_is_bit_identical():
    """With R = 0 the composite is today's, bit for bit (spec §5.2 (a))."""
    from pde_bitwise.capture_book import knock_in_products
    for name, (product, _) in knock_in_products().items():
        assert product.rebate == 0.0, name
