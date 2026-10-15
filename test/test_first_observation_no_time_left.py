"""The projection through the first observation when that observation is the maturity (no time left after it).

Valued on its penultimate record (the record stays on the valuation instant, the spot does not breach it), a
DISCRETE contract is decided once more, at the maturity record: a European payoff decided at T. The continuation
after that record is the terminal payoff itself, whose kinks (the strike, a sharkfin's cap) are the expectation's
panel edges, so the two Gauss-Legendre orders agree on each smooth piece. Before this, a knock-out with the strike
inside the alive region refused (orders 8 and 12 differed by ~2e-3) where main 1e32c27c priced it.

The references are closed forms of that European payoff: Black-Scholes calls / puts and cash digitals at T.
"""
from __future__ import annotations

import math
import warnings

import pytest
from scipy.stats import norm

import test_discrete_barrier_between_observations as T
from quantark.asset.equity.engine.analytical.accumulator_analytical_engine import AccumulatorAnalyticalEngine
from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
from quantark.asset.equity.engine.analytical.double_barrier_option_engine import DoubleBarrierOptionAnalyticalEngine
from quantark.asset.equity.engine.analytical.double_sharkfin_option_analytical_engine import (
    DoubleSharkfinOptionAnalyticalEngine,
)
from quantark.asset.equity.engine.analytical.one_touch_analytical_engine import OneTouchAnalyticalEngine
from quantark.asset.equity.engine.analytical.single_sharkfin_option_analytical_engine import (
    SingleSharkfinOptionAnalyticalEngine,
)
from quantark.asset.equity.product.option import (
    AccumulatorOption,
    BarrierOption,
    DoubleBarrierOption,
    DoubleSharkfinOption,
    OneTouchOption,
    SingleSharkfinOption,
)
from quantark.util.enum import (
    AccumulatorKnockOutType,
    BarrierDirection,
    BarrierType,
    DoubleBarrierType,
    ObservationType,
    OptionType,
    TouchType,
)

R, Q, VOL = 0.03, 0.01, 0.20
TAU = 1.0 / 12.0        # the last record (the maturity) after aging onto the eleventh
SPOT = 105.0
D = dict(maturity=1.0, observation_type=ObservationType.DISCRETE)
#: the expectation's own error check is Tolerance.RELATIVE (1e-9); the rule is far tighter on smooth pieces
ABS = 1e-10


def _on_penultimate_record(product):
    product.time_shift(11 / 12, T.VALUATION, T.env(100.0), keep_landing=True)
    return product


def _price(engine_cls, product, spot=SPOT):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(engine_cls().price(_on_penultimate_record(product), T.env(spot)))


# --- European closed forms at T = TAU (DF = exp(-r T)) -----------------------------------------------------------------

def _d2(level, spot=SPOT):
    return (math.log(spot / level) + (R - Q - 0.5 * VOL * VOL) * TAU) / (VOL * math.sqrt(TAU))


def _call(strike, spot=SPOT):
    d2 = _d2(strike, spot)
    d1 = d2 + VOL * math.sqrt(TAU)
    return spot * math.exp(-Q * TAU) * norm.cdf(d1) - strike * math.exp(-R * TAU) * norm.cdf(d2)


def _put(strike, spot=SPOT):
    d2 = _d2(strike, spot)
    d1 = d2 + VOL * math.sqrt(TAU)
    return strike * math.exp(-R * TAU) * norm.cdf(-d2) - spot * math.exp(-Q * TAU) * norm.cdf(-d1)


def _above(level, spot=SPOT):
    """DF(T) P[S_T >= level]."""
    return math.exp(-R * TAU) * norm.cdf(_d2(level, spot))


def _below(level, spot=SPOT):
    """DF(T) P[S_T <= level]."""
    return math.exp(-R * TAU) * norm.cdf(-_d2(level, spot))


def _call_below(strike, cap, spot=SPOT):
    """DF(T) E[(S_T - K)+ 1{S_T < cap}]."""
    return _call(strike, spot) - _call(cap, spot) - (cap - strike) * _above(cap, spot)


def _put_above(strike, floor, spot=SPOT):
    """DF(T) E[(K - S_T)+ 1{S_T > floor}]."""
    return _put(strike, spot) - _put(floor, spot) - (strike - floor) * _below(floor, spot)


REFERENCES = {
    # up-and-out call K 100, L 120, rebate 1 at the terminal payment time
    "UOC": (BarrierAnalyticalEngine, lambda: BarrierOption(
        strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT, rebate=1.0,
        observation_schedule=T._single_schedule(120.0), **D),
        _call_below(100.0, 120.0) + 1.0 * _above(120.0)),
    # down-and-out call K 100, L 85: the strike inside the alive region
    "DOC": (BarrierAnalyticalEngine, lambda: BarrierOption(
        strike=100.0, option_type=OptionType.CALL, barrier=85.0, barrier_type=BarrierType.DOWN_OUT, rebate=0.0,
        observation_schedule=T._single_schedule(85.0), **D),
        _call(100.0)),
    # up-and-in put K 130, L 120, rebate 2 if never knocked in: the strike inside the BREACHED region
    "UIP_K130": (BarrierAnalyticalEngine, lambda: BarrierOption(
        strike=130.0, option_type=OptionType.PUT, barrier=120.0, barrier_type=BarrierType.UP_IN, rebate=2.0,
        observation_schedule=T._single_schedule(120.0), **D),
        _put(130.0) - _put(120.0) - (130.0 - 120.0) * _below(120.0) + 2.0 * _below(120.0)),
    # double knock-out call K 100, corridor 75..130, rebate 1 at the terminal payment time
    "DKO": (DoubleBarrierOptionAnalyticalEngine, lambda: DoubleBarrierOption(
        strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0, lower_barrier=75.0,
        barrier_type=DoubleBarrierType.KNOCK_OUT, rebate=1.0, observation_schedule=T._double_schedule(130.0, 75.0),
        **D),
        _call_below(100.0, 130.0) + 1.0 * (_above(130.0) + _below(75.0))),
    # single sharkfin call K 100, L 120, knock-out cash 0.5 paid at the hit (the maturity record), no-hit cash 3
    "SSF_hit": (SingleSharkfinOptionAnalyticalEngine, lambda: SingleSharkfinOption(
        strike=100.0, option_type=OptionType.CALL, barrier=120.0, knock_out_rebate=0.5, no_hit_rebate=3.0,
        pay_at_hit=True, observation_schedule=T._single_schedule(120.0), **D),
        _call_below(100.0, 120.0) + 3.0 * _below(120.0) + 0.5 * _above(120.0)),
    # double sharkfin call K 100, corridor 75..130, knock-out cash 0.5 at the hit, no-hit cash 3
    "DSF_hit": (DoubleSharkfinOptionAnalyticalEngine, lambda: DoubleSharkfinOption(
        strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0, lower_barrier=75.0, knock_out_rebate=0.5,
        no_hit_rebate=3.0, pay_at_hit=True, observation_schedule=T._double_schedule(130.0, 75.0), **D),
        _call_below(100.0, 130.0) + 3.0 * (math.exp(-R * TAU) - _above(130.0) - _below(75.0))
        + 0.5 * (_above(130.0) + _below(75.0))),
    # double sharkfin put K 100, corridor 75..130, knock-out cash 0.5 at the terminal payment time, no-hit cash 3
    "DSF_put": (DoubleSharkfinOptionAnalyticalEngine, lambda: DoubleSharkfinOption(
        strike=100.0, option_type=OptionType.PUT, upper_barrier=130.0, lower_barrier=75.0, knock_out_rebate=0.5,
        no_hit_rebate=3.0, observation_schedule=T._double_schedule(130.0, 75.0), **D),
        _put_above(100.0, 75.0) + 3.0 * (math.exp(-R * TAU) - _above(130.0) - _below(75.0))
        + 0.5 * (_above(130.0) + _below(75.0))),
}


@pytest.mark.parametrize("name", list(REFERENCES))
def test_the_maturity_record_decides_a_european_payoff(name):
    engine_cls, build, reference = REFERENCES[name]
    assert _price(engine_cls, build()) == pytest.approx(reference, abs=ABS, rel=0.0)


@pytest.mark.parametrize("spot", [70.0, 75.5, 99.9, 100.0, 100.1, 119.99, 129.9, 140.0])
def test_double_knock_out_on_the_penultimate_record_at_any_spot(spot):
    product = DoubleBarrierOption(
        strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0, lower_barrier=75.0,
        barrier_type=DoubleBarrierType.KNOCK_OUT, rebate=1.0, observation_schedule=T._double_schedule(130.0, 75.0),
        **D)
    if spot <= 75.0 or spot >= 130.0:
        pytest.skip("breached by today's record")
    reference = _call_below(100.0, 130.0, spot) + 1.0 * (_above(130.0, spot) + _below(75.0, spot))
    assert _price(DoubleBarrierOptionAnalyticalEngine, product, spot) == pytest.approx(reference, abs=ABS, rel=0.0)


# --- every kind prices on its penultimate record -----------------------------------------------------------------------

def _kinds():
    s, d = T._single_schedule, T._double_schedule
    out = []
    for bt in BarrierType:
        for ot in (OptionType.CALL, OptionType.PUT):
            level = 120.0 if bt in (BarrierType.UP_OUT, BarrierType.UP_IN) else 85.0
            for strike in (80.0, 100.0, 125.0):
                for rebate, at_hit in ((0.0, False), (2.0, False), (2.0, True)):
                    if at_hit and bt in (BarrierType.UP_IN, BarrierType.DOWN_IN):
                        continue
                    out.append((f"{bt.name}-{ot.name}-K{strike}-R{rebate}{'-hit' if at_hit else ''}",
                                BarrierAnalyticalEngine,
                                (lambda bt=bt, ot=ot, level=level, strike=strike, rebate=rebate, at_hit=at_hit:
                                 BarrierOption(strike=strike, option_type=ot, barrier=level, barrier_type=bt,
                                               rebate=rebate, pay_at_hit=at_hit, observation_schedule=s(level), **D))))
    for tt in (TouchType.ONE_TOUCH, TouchType.NO_TOUCH):
        for up, level in ((True, 120.0), (False, 85.0)):
            out.append((f"{tt.name}-{'up' if up else 'dn'}", OneTouchAnalyticalEngine,
                        (lambda tt=tt, up=up, level=level: OneTouchOption(
                            barrier=level, barrier_direction=BarrierDirection.UP if up else BarrierDirection.DOWN,
                            rebate=5.0, touch_type=tt, observation_schedule=s(level), **D))))
    for dbt in DoubleBarrierType:
        for ot in (OptionType.CALL, OptionType.PUT):
            out.append((f"{dbt.name}-{ot.name}", DoubleBarrierOptionAnalyticalEngine,
                        (lambda dbt=dbt, ot=ot: DoubleBarrierOption(
                            strike=100.0, option_type=ot, upper_barrier=130.0, lower_barrier=75.0, barrier_type=dbt,
                            rebate=1.0, observation_schedule=d(130.0, 75.0), **D))))
    for ot, level in ((OptionType.CALL, 120.0), (OptionType.PUT, 85.0)):
        for at_hit in (False, True):
            out.append((f"SSF-{ot.name}-{at_hit}", SingleSharkfinOptionAnalyticalEngine,
                        (lambda ot=ot, level=level, at_hit=at_hit: SingleSharkfinOption(
                            strike=100.0, option_type=ot, barrier=level, knock_out_rebate=0.5, no_hit_rebate=3.0,
                            pay_at_hit=at_hit, observation_schedule=s(level), **D))))
            out.append((f"DSF-{ot.name}-{at_hit}", DoubleSharkfinOptionAnalyticalEngine,
                        (lambda ot=ot, at_hit=at_hit: DoubleSharkfinOption(
                            strike=100.0, option_type=ot, upper_barrier=130.0, lower_barrier=75.0,
                            knock_out_rebate=0.5, no_hit_rebate=3.0, pay_at_hit=at_hit,
                            observation_schedule=d(130.0, 75.0), **D))))
    for kot in AccumulatorKnockOutType:
        out.append((f"ACC-{kot.name}", AccumulatorAnalyticalEngine,
                    (lambda kot=kot: AccumulatorOption(
                        strike=95.0, knock_out_barrier=110.0, option_type=OptionType.CALL, maturity=1.0,
                        daily_share_accumulation=1.0, gearing=2.0, knock_out_type=kot,
                        observation_dates=[k / 12 for k in range(1, 13)]))))
    return out


KINDS = _kinds()


@pytest.mark.parametrize("name,engine_cls,build", KINDS, ids=[k[0] for k in KINDS])
@pytest.mark.parametrize("spot", [90.0, 100.0, 105.0])
def test_every_kind_prices_on_its_penultimate_record(name, engine_cls, build, spot):
    value = _price(engine_cls, build(), spot)
    assert math.isfinite(value)
