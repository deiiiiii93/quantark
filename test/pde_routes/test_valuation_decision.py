"""The decision at the valuation instant (spec 3C decision 25; §7.2 rows 'valuation-date hit test',
'valuation-date record' (i) and (ii), 'valuation-date jump' at the facade; Review Focus 1 and 2)."""
import math
from datetime import datetime

import numpy as np
import pytest
from scipy.stats import norm

from pde_routes.barrier_reference import knock_out
from pde_routes.cases import CONST, VAL_DATE, black_scholes, env_const, obs_dates, schedule
from quantark.asset.equity.engine.pde import (BarrierPDESolver, DoubleBarrierPDESolver, DoubleOneTouchPDESolver,
                                              OneTouchPDESolver)
from quantark.asset.equity.product.option import BarrierOption, DoubleBarrierOption
from quantark.asset.equity.product.option.double_one_touch_option import DoubleOneTouchOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
from quantark.util.enum import (BarrierDirection, BarrierType, DoubleBarrierType, ObservationAggregation,
                                ObservationType, OptionType, TouchType)

VOL, R, Q = CONST["vol"], CONST["rate"], CONST["div"]
M = dict(vol=VOL, r=R, q=Q)
FIRST, ACC = ObservationAggregation.STOP_FIRST_HIT, ObservationAggregation.ACCUMULATE
INSIDE = 1e-3                    # the §7.2 bound away from a firing record
FIRES = 1e-12                    # where a record fires: its cash on the curve


def p_above(spot, level, t):
    mu = R - Q - 0.5 * VOL * VOL
    return float(norm.cdf((math.log(spot / level) + mu * t) / (VOL * math.sqrt(t))))


# --- (i) a record on the valuation date with its own level, cash and payment time (§4.11) -------------------

from pde_routes.cases import record_product                                                  # noqa: E402,F401


@pytest.mark.parametrize("spot", [119.9, 119.99, 120.0, 125.0, 129.99, 130.0, 135.0])
def test_valuation_record_carries_its_own_level_cash_and_payment(spot):
    env = env_const(spot)
    solver = BarrierPDESolver()
    price, g = solver.price(record_product(), env), solver.calculate_greeks(record_product(), env)
    if spot >= 120.0:
        cash = 7.0 * math.exp(-R * 0.25)                         # 6.947696, from the record's own payment time
        assert abs(price - cash) <= FIRES and abs(g["price"] - cash) <= FIRES
        assert g["delta"] == 0.0 and g["gamma"] == 0.0
    else:
        ref = knock_out(payoff="call", strike=100.0, lower=None, upper=130.0, cash=5.0, at_hit=True, obs=[0.5, 1.0],
                        spots=[spot], **M)[0][spot]
        assert abs(price - ref[0]) <= INSIDE                       # today -3.7e-2 / -5.2e-2
        assert abs(g["delta"] - ref[1]) <= 2e-4                    # today -0.27


# --- spot beyond a discretely monitored barrier, no record due (§4.9) ----------------------------------

def single(bt, call, level, rebate, dates, mode=FIRST):
    kw = dict(observation_type=ObservationType.DISCRETE)
    if mode is FIRST:
        kw["observation_dates"] = list(dates)
    else:
        kw["observation_schedule"] = schedule(list(dates), mode, barrier=level, payoff=rebate)
    return BarrierOption(strike=100.0, option_type=OptionType.CALL if call else OptionType.PUT, barrier=level,
                         barrier_type=bt, maturity=1.0, rebate=rebate, **kw)


MONTHLY, LATE = obs_dates("monthly"), [k / 12.0 for k in range(7, 13)]
HIT_CELLS = [
    ("UO call 130 rb5 s135", single(BarrierType.UP_OUT, True, 130.0, 5.0, MONTHLY), 135.0,
     dict(payoff="call", lower=None, upper=130.0, cash=5.0, obs=MONTHLY)),
    ("UO call 130 rb5 s131", single(BarrierType.UP_OUT, True, 130.0, 5.0, MONTHLY), 131.0,
     dict(payoff="call", lower=None, upper=130.0, cash=5.0, obs=MONTHLY)),
    ("UO call 130 rb5 late s135", single(BarrierType.UP_OUT, True, 130.0, 5.0, LATE), 135.0,
     dict(payoff="call", lower=None, upper=130.0, cash=5.0, obs=LATE)),
    ("DO put 85 rb3 s80", single(BarrierType.DOWN_OUT, False, 85.0, 3.0, MONTHLY), 80.0,
     dict(payoff="put", lower=85.0, upper=None, cash=3.0, obs=MONTHLY)),
]


@pytest.mark.parametrize("name,product,spot,ref", HIT_CELLS, ids=[c[0] for c in HIT_CELLS])
def test_spot_beyond_a_discrete_barrier_is_solved(name, product, spot, ref):
    exact = knock_out(strike=100.0, at_hit=False, spots=[spot], **ref, **M)[0][spot][0]
    assert abs(BarrierPDESolver().price(product, env_const(spot)) - exact) <= INSIDE      # today -0.20 to -1.08


@pytest.mark.parametrize("bt,call,level,spot", [(BarrierType.UP_IN, True, 130.0, 135.0),
                                                 (BarrierType.DOWN_IN, False, 85.0, 80.0)])
def test_knock_in_beyond_a_discrete_barrier_is_the_composite(bt, call, level, spot):
    up = bt is BarrierType.UP_IN
    ko = knock_out(payoff="call" if call else "put", strike=100.0, lower=None if up else level,
                   upper=level if up else None, cash=0.0, at_hit=False, obs=MONTHLY, spots=[spot], **M)[0][spot][0]
    exact = black_scholes(call, spot, 100.0, 1.0)[0] - ko
    got = BarrierPDESolver().price(single(bt, call, level, 0.0, MONTHLY), env_const(spot))
    assert abs(got - exact) <= INSIDE                                                   # today +0.596 / +9.8e-2


def test_double_knock_out_and_one_touch_beyond_a_barrier():
    dko = DoubleBarrierOption(100.0, OptionType.CALL, 130.0, 80.0, DoubleBarrierType.KNOCK_OUT, maturity=1.0, rebate=5.0,
                              observation_type=ObservationType.DISCRETE, observation_dates=MONTHLY)
    exact = knock_out(payoff="call", strike=100.0, lower=80.0, upper=130.0, cash=5.0, at_hit=False, obs=MONTHLY,
                      spots=[135.0], **M)[0][135.0][0]
    assert abs(DoubleBarrierPDESolver().price(dko, env_const(135.0)) - exact) <= INSIDE   # today -8.6e-2
    ot = OneTouchOption(barrier=130.0, barrier_direction=BarrierDirection.UP, maturity=1.0, rebate=1.0,
                        payment_at_hit=False, touch_type=TouchType.ONE_TOUCH, observation_type=ObservationType.DISCRETE,
                        observation_dates=MONTHLY)
    exact = knock_out(payoff="zero", strike=100.0, lower=None, upper=130.0, cash=1.0, at_hit=False, obs=MONTHLY,
                      spots=[135.0], **M)[0][135.0][0]
    assert abs(OneTouchPDESolver().price(ot, env_const(135.0)) - exact) <= INSIDE          # today +7.9e-2


def test_accumulate_schedule_beyond_its_level():
    """Monthly, barrier 130, every payoff 5 paid at maturity: value = call + 5 e^{-rT} sum_k P[S_k >= 130]."""
    p = single(BarrierType.UP_OUT, True, 130.0, 5.0, MONTHLY, ACC)
    closed = black_scholes(True, 135.0, 100.0, 1.0)[0] + 5.0 * math.exp(-R) * sum(p_above(135.0, 130.0, t) for t in MONTHLY)
    assert abs(BarrierPDESolver().price(p, env_const(135.0)) - closed) <= INSIDE         # today -67.7 (72.55)


# --- the valuation-date jump at the facade (§4.11): observations 0 / 0.5 / 1 -------------------------------

def jump_products():
    sch = lambda mode=FIRST, **kw: schedule([0.0, 0.5, 1.0], mode, **kw)            # noqa: E731
    return {
        "up_in": BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=BarrierType.UP_IN,
                               maturity=1.0, observation_type=ObservationType.DISCRETE,
                               observation_schedule=sch(payoff=0.0)),
        "one_touch": OneTouchOption(barrier=130.0, barrier_direction=BarrierDirection.UP, maturity=1.0, rebate=5.0,
                                    payment_at_hit=False, touch_type=TouchType.ONE_TOUCH,
                                    observation_type=ObservationType.DISCRETE, observation_schedule=sch()),
        "accumulate": BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0,
                                    barrier_type=BarrierType.UP_OUT, maturity=1.0, rebate=5.0,
                                    observation_type=ObservationType.DISCRETE, observation_schedule=sch(ACC)),
    }


def jump_exact(name, spot):
    """(price, delta) against the later dates (0.5, 1); the valuation-date record applied at the spot."""
    later = [0.5, 1.0]
    if name == "up_in":
        if spot >= 130.0:
            bs = black_scholes(True, spot, 100.0, 1.0)
            return bs[0], bs[1]
        ko = knock_out(payoff="call", strike=100.0, lower=None, upper=130.0, cash=0.0, at_hit=False, obs=later,
                       spots=[spot], **M)[0][spot]
        bs = black_scholes(True, spot, 100.0, 1.0)
        return bs[0] - ko[0], bs[1] - ko[1]
    if name == "one_touch":
        if spot >= 130.0:
            return 5.0 * math.exp(-R), 0.0
        ref = knock_out(payoff="zero", strike=100.0, lower=None, upper=130.0, cash=5.0, at_hit=False, obs=later,
                        spots=[spot], **M)[0][spot]
        return ref[0], ref[1]
    bs = black_scholes(True, spot, 100.0, 1.0)                                         # accumulate
    h = 1e-4 * spot
    f = lambda s: 5.0 * math.exp(-R) * (p_above(s, 130.0, 0.5) + p_above(s, 130.0, 1.0))  # noqa: E731
    now = 5.0 * math.exp(-R) if spot >= 130.0 else 0.0
    return bs[0] + f(spot) + now, bs[1] + (f(spot + h) - f(spot - h)) / (2 * h)


SOLVER = {"up_in": BarrierPDESolver, "one_touch": OneTouchPDESolver, "accumulate": BarrierPDESolver}


@pytest.mark.parametrize("name", ["up_in", "one_touch", "accumulate"])
@pytest.mark.parametrize("spot", [129.0, 129.9, 129.99, 130.0, 131.0])
def test_valuation_date_jump_at_the_facade(name, spot):
    g = SOLVER[name]().calculate_greeks(jump_products()[name], env_const(spot))
    price, delta = jump_exact(name, spot)
    assert abs(g["price"] - price) <= 1.5e-3                       # today +2.5 just inside
    assert abs(g["delta"] - delta) <= 2e-4                         # today +13


def test_double_knock_in_and_double_one_touch_jump_at_the_facade():
    dki = DoubleBarrierOption(100.0, OptionType.CALL, 130.0, 80.0, DoubleBarrierType.KNOCK_IN, maturity=1.0,
                              observation_type=ObservationType.DISCRETE,
                              observation_schedule=schedule([0.0, 0.5, 1.0], FIRST, double=True, payoff=0.0))
    for spot in (80.01, 80.1, 129.9, 129.99):
        ko = knock_out(payoff="call", strike=100.0, lower=80.0, upper=130.0, cash=0.0, at_hit=False, obs=[0.5, 1.0],
                       spots=[spot], **M)[0][spot]
        bs = black_scholes(True, spot, 100.0, 1.0)
        g = DoubleBarrierPDESolver().calculate_greeks(dki, env_const(spot))
        assert abs(g["price"] - (bs[0] - ko[0])) <= 1.5e-3 and abs(g["delta"] - (bs[1] - ko[1])) <= 2e-4
    dot = DoubleOneTouchOption(upper_barrier=130.0, lower_barrier=80.0, maturity=1.0, rebate=5.0, payment_at_hit=False,
                               touch_type=TouchType.DOUBLE_ONE_TOUCH, observation_type=ObservationType.DISCRETE,
                               observation_schedule=schedule([0.0, 0.5, 1.0], FIRST, double=True))
    for spot in (80.01, 129.99):
        ref = knock_out(payoff="zero", strike=100.0, lower=80.0, upper=130.0, cash=5.0, at_hit=False, obs=[0.5, 1.0],
                        spots=[spot], **M)[0][spot]
        g = DoubleOneTouchPDESolver().calculate_greeks(dot, env_const(spot))
        assert abs(g["price"] - ref[0]) <= 1.5e-3 and abs(g["delta"] - ref[1]) <= 2e-4


# --- Review Focus 1 and 2 ----------------------------------------------------------------------------------

def test_dated_product_on_its_observation_date():
    """Records by date, valued on the first one: the record's own cash, nothing written into the nodes."""
    sch = ObservationSchedule(records=[ObservationRecord(observation_date=VAL_DATE, barrier=120.0, payoff=7.0),
                                       ObservationRecord(observation_date=datetime(2026, 7, 1)),
                                       ObservationRecord(observation_date=datetime(2027, 1, 1))], aggregation_mode=FIRST)
    p = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=BarrierType.UP_OUT,
                      exercise_date=datetime(2027, 1, 1), rebate=5.0, observation_type=ObservationType.DISCRETE,
                      observation_schedule=sch)
    env = env_const(125.0)
    solver = BarrierPDESolver()
    cash = 7.0 * env.get_discount_factor(solver._terminal_payment_time(p, env))
    assert abs(solver.price(p, env) - cash) <= FIRES
    g = solver.calculate_greeks(p, env)
    assert abs(g["price"] - cash) <= FIRES and g["delta"] == 0.0


def test_bump_context_folds_the_decision_across_a_record_level():
    """A bump context's frozen layout and a bumped spot that crosses the valuation record's level: the clone
    folds the decision as a fresh solver does (no raise, no value written into the nodes)."""
    env = env_const(119.4)
    solver = BarrierPDESolver()
    ctx = solver.create_bump_context(record_product(), env)
    assert ctx.price(record_product(), env) == solver.price(record_product(), env)
    up = env_const(119.4 * 1.01)                                     # 120.594: the record fires
    assert ctx.price(record_product(), up) == BarrierPDESolver().price(record_product(), up)
    assert abs(ctx.price(record_product(), up) - 7.0 * math.exp(-R * 0.25)) <= FIRES


# --- §7.2 'valuation-date record' (ii): the 168 cells of §4.12 -------------------------------------------------
# Cell builder and references copied from the 3C evidence `w3c_record_levels_lag_curves.py`, verbatim, with the
# term recursion from `pde_routes.barrier_reference` and the kinked curves from `pde_routes.cases`.
from pde_routes.barrier_reference import knock_out_term                                      # noqa: E402
from pde_routes.cases import kinked                                                          # noqa: E402
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit          # noqa: E402

D = ObservationType.DISCRETE


def curves(env):
    df = lambda t: 1.0 if t <= 0.0 else float(env.get_discount_factor(float(t)))                                                        # noqa: E731
    growth = lambda t: 1.0 if t <= 0.0 else float(np.exp((env.get_rate(float(t)) - env.get_div_yield(float(t))) * float(t)))            # noqa: E731
    return df, growth


def records(double, with_valuation_record):
    own = dict(upper_barrier=120.0, lower_barrier=85.0) if double else dict(barrier=120.0)
    std = dict(upper_barrier=130.0, lower_barrier=80.0) if double else dict(barrier=130.0)
    recs = [ObservationRecord(observation_time=0.5, payoff=5.0, **std), ObservationRecord(observation_time=1.0, payoff=5.0, **std)]
    if with_valuation_record:
        recs.insert(0, ObservationRecord(observation_time=0.0, payoff=7.0, **own))
    return ObservationSchedule(records=recs)


def conv(lag):
    return dict(settlement_convention=SettlementConvention(lag=lag, lag_unit=SettlementLagUnit.YEAR_FRACTION)) if lag else {}


FAMILIES = {
    "up-and-out call": (BarrierPDESolver, False, "call", 5.0, 7.0,
                        lambda s, lag: BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=BarrierType.UP_OUT, maturity=1.0, rebate=5.0,
                                                     observation_type=D, observation_schedule=s, **conv(lag))),
    "double knock-out call": (DoubleBarrierPDESolver, True, "call", 5.0, 7.0,
                              lambda s, lag: DoubleBarrierOption(strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0, lower_barrier=80.0,
                                                                 barrier_type=DoubleBarrierType.KNOCK_OUT, maturity=1.0, rebate=5.0, observation_type=D,
                                                                 observation_schedule=s, **conv(lag))),
    "one-touch": (OneTouchPDESolver, False, "zero", 5.0, 7.0,
                  lambda s, lag: OneTouchOption(barrier=130.0, barrier_direction=BarrierDirection.UP, maturity=1.0, rebate=5.0, payment_at_hit=False,
                                                touch_type=TouchType.ONE_TOUCH, observation_type=D, observation_schedule=s, **conv(lag))),
    "no-touch": (OneTouchPDESolver, False, "one", 0.0, 0.0,
                 lambda s, lag: OneTouchOption(barrier=130.0, barrier_direction=BarrierDirection.UP, maturity=1.0, rebate=5.0, payment_at_hit=False,
                                               touch_type=TouchType.NO_TOUCH, observation_type=D, observation_schedule=s, **conv(lag))),
    "double one-touch": (DoubleOneTouchPDESolver, True, "zero", 5.0, 7.0,
                         lambda s, lag: DoubleOneTouchOption(upper_barrier=130.0, lower_barrier=80.0, maturity=1.0, rebate=5.0, payment_at_hit=False,
                                                             touch_type=TouchType.DOUBLE_ONE_TOUCH, observation_type=D, observation_schedule=s, **conv(lag))),
    "double no-touch": (DoubleOneTouchPDESolver, True, "one", 0.0, 0.0,
                        lambda s, lag: DoubleOneTouchOption(upper_barrier=130.0, lower_barrier=80.0, maturity=1.0, rebate=5.0, payment_at_hit=False,
                                                            touch_type=TouchType.DOUBLE_NO_TOUCH, observation_type=D, observation_schedule=s, **conv(lag))),
}

MARKETS = {"flat": lambda s: env_const(spot=s), "kinked": kinked}
SWEEP = [(market, lag, name) for market in MARKETS for lag in (0.0, 0.25) for name in FAMILIES]


@pytest.mark.parametrize("market,lag,name", SWEEP, ids=[f"{m}-{lag}-{n}" for m, lag, n in SWEEP])
def test_record_levels_lags_and_curves(market, lag, name):
    solver, double, payoff, later_cash, record_cash, make = FAMILIES[name]
    spots = (84.0, 85.0, 85.01, 100.0, 119.99, 120.0, 125.0, 135.0) if double else (100.0, 119.9, 119.99, 120.0, 125.0, 135.0)
    scale = 5.0 if payoff == "one" else 1.0                  # a no-touch pays its rebate of 5 on survival
    for spot in spots:
        env = MARKETS[market](spot)
        df, growth = curves(env)
        fired = spot >= 120.0 or (double and spot <= 85.0)
        price = float(solver().price(make(records(double, True), lag), env))
        if fired:
            assert abs(price - record_cash * df(1.0 + lag)) <= FIRES, (spot, price)
        else:
            ref = knock_out_term(payoff=payoff, strike=100.0, lower=80.0 if double else None, upper=130.0, cash=later_cash, at_hit=False,
                                 obs=[0.5, 1.0], spots=[spot], vol=VOL, df=df, growth=growth, terminal_pay=1.0 + lag)[0][spot]
            assert abs(price - scale * ref[0]) <= INSIDE, (spot, price, scale * ref[0])
