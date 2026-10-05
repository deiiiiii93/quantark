"""The value at maturity (spec 3C decision 30; §7.2 'maturity' row (i)-(vi))."""
import dataclasses
import math

import pytest

from pde_routes.cases import env_const
from quantark.asset.equity.engine.pde import (AmericanPDESolver, BarrierPDESolver, DoubleBarrierPDESolver,
                                              DoubleOneTouchPDESolver, EuropeanPDESolver, OneTouchPDESolver)
from quantark.asset.equity.product.option import BarrierOption, DoubleBarrierOption, EuropeanVanillaOption
from quantark.asset.equity.product.option.american_option import AmericanOption
from quantark.asset.equity.product.option.double_one_touch_option import DoubleOneTouchOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.one_touch_option import OneTouchOption
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.util.enum import (BarrierDirection, BarrierType, DoubleBarrierType, ObservationAggregation,
                                ObservationType, OptionType, TouchType)

ACC = ObservationAggregation.ACCUMULATE
C, D, X = ObservationType.CONTINUOUS, ObservationType.DISCRETE, ObservationType.EXPIRY


# --- product builders, copied from the 3C evidence `w2_maturity_payment_lag.py:29-75` (touches pay 10, barriers
# rebate 5; a schedule keeps one record, at T) ---------------------------------------------------------------

def conv():
    return SettlementConvention(lag=0.25, lag_unit=SettlementLagUnit.YEAR_FRACTION)


def at(make, obs_type, T):
    """The product with `T` left: a schedule keeps one record, at T."""
    kw = dict(maturity=1.0, observation_type=obs_type, settlement_convention=conv())
    if obs_type is D:
        kw["observation_dates"] = [1.0]
    p = make(kw)
    if obs_type is D:
        own = (dict(upper_barrier=p.upper_barrier, lower_barrier=p.lower_barrier, payoff=p.rebate) if hasattr(p, "upper_barrier")
               else dict(barrier=p.barrier, payoff=p.rebate))
        p.observation_schedule = ObservationSchedule(records=[ObservationRecord(observation_time=T, **own)])
        if getattr(p, "observation_dates", None) is not None:
            p.observation_dates = [T]
    p.maturity = T
    return p


def single(bt, at_hit=False, participation=1.0):
    return lambda kw: BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=bt, rebate=5.0, pay_at_hit=at_hit,
                                    participation_rate=participation, **kw)


def double(bt):
    return lambda kw: DoubleBarrierOption(strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0, lower_barrier=80.0, barrier_type=bt, rebate=5.0, **kw)


def touch(tt):
    return lambda kw: OneTouchOption(barrier=130.0, barrier_direction=BarrierDirection.UP, rebate=10.0, payment_at_hit=False, touch_type=tt, **kw)


def touch2(tt):
    return lambda kw: DoubleOneTouchOption(upper_barrier=130.0, lower_barrier=80.0, rebate=10.0, payment_at_hit=False, touch_type=tt, **kw)

LAG, RATE = 0.25, 0.03
DDF = math.exp(-RATE * LAG)                                  # 0.992528
C, D, X = ObservationType.CONTINUOUS, ObservationType.DISCRETE, ObservationType.EXPIRY
SOLVERS = {"single": BarrierPDESolver, "double": DoubleBarrierPDESolver, "touch": OneTouchPDESolver,
           "touch2": DoubleOneTouchPDESolver}


def three_ops(cls, product, spot):
    """price, calculate_greeks and the session's npv at one spot; no call may raise."""
    env = env_const(spot)
    g = cls().calculate_greeks(product, env)
    return cls().price(product, env), g, cls()._session_outputs(product, env).npv


# (ii) the 36 cells of §4.12: value = amount x DF(payment), delta = slope x participation x the same DF
CELLS = [
    # (family, maker, obs, spot, value, delta)
    ("single", single(BarrierType.UP_OUT), C, 125.0, 25.0 * DDF, DDF),
    ("single", single(BarrierType.UP_OUT), C, 135.0, 5.0 * DDF, 0.0),
    ("single", single(BarrierType.UP_OUT), D, 125.0, 25.0 * DDF, DDF),
    ("single", single(BarrierType.UP_OUT), D, 135.0, 5.0 * DDF, 0.0),
    ("single", single(BarrierType.UP_IN), D, 135.0, 35.0 * DDF, DDF),
    ("single", single(BarrierType.UP_IN), D, 125.0, 5.0 * DDF, 0.0),
    ("touch", touch(TouchType.NO_TOUCH), D, 125.0, 10.0 * DDF, 0.0),
    ("touch", touch(TouchType.NO_TOUCH), D, 135.0, 0.0, 0.0),
    ("touch", touch(TouchType.ONE_TOUCH), D, 135.0, 10.0 * DDF, 0.0),
    ("touch", touch(TouchType.ONE_TOUCH), D, 125.0, 0.0, 0.0),
]


@pytest.mark.parametrize("family,make,obs,spot,value,delta", CELLS)
def test_terminal_value_carries_the_payment_discount(family, make, obs, spot, value, delta):
    product = at(make, obs, 0.0)
    price, g, npv = three_ops(SOLVERS[family], product, spot)
    for v in (price, g["price"], npv):
        assert abs(v - value) <= 1e-12                         # today 25.000000 for 24.813201, 35 for 34.738482
    assert abs(g["delta"] - delta) <= 1e-12 and g["gamma"] == 0.0


def test_participation_at_maturity():
    p = at(lambda kw: BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0,
                                    barrier_type=BarrierType.UP_OUT, rebate=5.0, participation_rate=0.5, **kw), C, 0.0)
    price, g, npv = three_ops(BarrierPDESolver, p, 125.0)
    assert abs(price - 12.5 * DDF) <= 1e-12 and abs(g["delta"] - 0.5 * DDF) <= 1e-12     # today 25 for 12.41


def test_maturity_record_with_its_own_level_and_cash():
    p = at(single(BarrierType.UP_OUT), D, 0.0)
    p.observation_schedule = ObservationSchedule(records=[ObservationRecord(observation_time=0.0, barrier=120.0, payoff=7.0)])
    price, g, npv = three_ops(BarrierPDESolver, p, 125.0)
    assert abs(price - 7.0 * DDF) <= 1e-12 and g["delta"] == 0.0                   # today 25


def test_schedule_without_a_maturity_record_does_not_test_the_barrier():
    p = at(single(BarrierType.UP_OUT), D, 0.0)
    # no record at maturity: the record carries the product's level (resolve(require_single=True) needs one) and
    # lies after maturity (the resolver refuses a negative determination time)
    p.observation_schedule = ObservationSchedule(records=[ObservationRecord(observation_time=0.5, barrier=130.0, payoff=5.0)])
    price, _, _ = three_ops(BarrierPDESolver, p, 135.0)
    assert abs(price - 35.0 * DDF) <= 1e-12                                        # today 5


@pytest.mark.parametrize("mode", [ObservationAggregation.ACCUMULATE, ObservationAggregation.STOP_FIRST_HIT])
@pytest.mark.parametrize("spot", [135.0, 125.0])
def test_touch_and_knock_out_operations_agree_with_the_march(mode, spot):
    """(iii): the three operations equal the fold of the march's own maturity record (a no-touch's is SET 0)."""
    for family, make in (("touch", touch(TouchType.NO_TOUCH)), ("touch2", touch2(TouchType.DOUBLE_NO_TOUCH)),
                         ("touch", touch(TouchType.ONE_TOUCH)), ("single", single(BarrierType.UP_OUT))):
        product = at(make, D, 0.0)
        product.observation_schedule = dataclasses.replace(product.observation_schedule, aggregation_mode=mode)
        price, g, npv = three_ops(SOLVERS[family], product, spot)
        assert price == g["price"] == npv


@pytest.mark.parametrize("family,make,obs,spot", [("single", single(BarrierType.UP_OUT), D, 125.0),
                                                   ("single", single(BarrierType.UP_OUT), D, 135.0),
                                                   ("touch", touch(TouchType.ONE_TOUCH), D, 135.0)])
def test_continuity_with_the_march_an_hour_before(family, make, obs, spot):
    """(iv): the value at maturity equals the product marched an hour before it, to the hour's carry."""
    cls = SOLVERS[family]
    at_t = cls().price(at(make, obs, 0.0), env_const(spot))
    before = cls().price(at(make, obs, 1e-4), env_const(spot))
    assert abs(at_t - before) <= 5e-4                        # today a jump of 0.187 on the surviving knock-out


def test_session_grid_at_maturity_fails_closed_and_curve_reprices():
    product = at(single(BarrierType.UP_OUT), C, 0.0)
    out = BarrierPDESolver()._session_outputs(product, env_const(125.0), want_grid=True)
    assert out.solution is None
    rows = BarrierPDESolver().calculate_spot_greeks_curve(product, env_const(125.0), [125.0, 135.0])
    assert abs(rows[0]["price"] - 25.0 * DDF) <= 1e-12 and abs(rows[1]["price"] - 5.0 * DDF) <= 1e-12


def test_expired_knock_in_prices():
    for obs in (C, D):
        p = at(single(BarrierType.UP_IN), obs, 0.0)
        assert abs(BarrierPDESolver().price(p, env_const(135.0)) - 35.0 * DDF) <= 1e-12     # today raises
    d = at(double(DoubleBarrierType.KNOCK_IN), C, 0.0)
    assert abs(DoubleBarrierPDESolver().price(d, env_const(135.0)) - 35.0 * DDF) <= 1e-12


def test_continuous_hit_keeps_its_hit_timing():
    """A rebate paid at hit, a terminal payment date of its own: the hit pays a lag after the hit (4.962640)."""
    p = at(single(BarrierType.UP_OUT, at_hit=True), C, 0.0)
    want = BarrierPDESolver().adapter._rebate_at_valuation(p, env_const(135.0))
    assert BarrierPDESolver().price(p, env_const(135.0)) == want


def test_expired_european_session_outputs():
    p = at(lambda kw: EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL,
                                            **{k: v for k, v in kw.items() if k != "observation_type"}), C, 0.0)
    assert abs(EuropeanPDESolver()._session_outputs(p, env_const(125.0)).npv - 25.0 * DDF) <= 1e-12   # today 25


def test_expired_american_greeks_take_the_payment_discount():
    """P10, the extension the user kept (2026-10-05)."""
    p = at(lambda kw: AmericanOption(100.0, OptionType.CALL, **{k: v for k, v in kw.items()
                                                                 if k != "observation_type"}), C, 0.0)
    env = env_const(125.0)
    g = AmericanPDESolver().calculate_greeks(p, env)
    assert abs(g["price"] - AmericanPDESolver().price(p, env)) <= 1e-12             # today 25 for 24.813201
    assert abs(g["delta"] - DDF) <= 1e-12
    assert abs(AmericanPDESolver()._session_outputs(p, env).npv - g["price"]) <= 1e-12


# --- §4.12's 36 cells, copied from the 3C evidence `w2b_terminal_read.py` (`at_maturity`, `dated`, `CASES`) -------

def default_record(p):
    """A record's own fields, as `from_legacy` fills them from the product."""
    if hasattr(p, "upper_barrier"):
        return dict(upper_barrier=p.upper_barrier, lower_barrier=p.lower_barrier, payoff=p.rebate)
    return dict(barrier=p.barrier, payoff=p.rebate)


def at_maturity(make, obs_type, record=None, mode=None):
    kw = dict(maturity=1.0, observation_type=obs_type, settlement_convention=conv())
    if obs_type is D:
        kw["observation_dates"] = [1.0]
    p = make(kw)
    if obs_type is D:
        rec = dict(record or default_record(p))
        sched = dict(records=[ObservationRecord(observation_time=0.0, **rec)])
        if mode is not None:
            sched["aggregation_mode"] = mode
        p.observation_schedule = ObservationSchedule(**sched)
        if getattr(p, "observation_dates", None) is not None:
            p.observation_dates = [0.0]
    p.maturity = 0.0
    return p


def dated(make):
    """Continuous, at maturity, with the lag convention AND an explicit settlement date 400 days out:
    a hit is paid 0.25 year after the hit, the terminal amount on the settlement date."""
    from datetime import timedelta
    from pde_routes.cases import VAL_DATE

    p = make(dict(maturity=1.0, observation_type=C, settlement_convention=conv(), settlement_date=VAL_DATE + timedelta(days=400)))
    p.maturity = 0.0
    return p


KO, KI = BarrierType.UP_OUT, BarrierType.UP_IN
DKO, DKI = DoubleBarrierType.KNOCK_OUT, DoubleBarrierType.KNOCK_IN
W2B = [  # name, solver, product factory, {spot: (amount, slope)}
    ("up-and-out, continuous", BarrierPDESolver, lambda: at_maturity(single(KO), C), {125.0: (25.0, 1.0), 135.0: (5.0, 0.0)}),
    ("up-and-out, a maturity record", BarrierPDESolver, lambda: at_maturity(single(KO), D), {125.0: (25.0, 1.0), 135.0: (5.0, 0.0)}),
    ("up-and-out, rebate at hit, continuous", BarrierPDESolver, lambda: at_maturity(single(KO, True), C), {125.0: (25.0, 1.0), 135.0: (5.0, 0.0)}),
    ("up-and-out, participation 0.5", BarrierPDESolver, lambda: at_maturity(single(KO, participation=0.5), C), {125.0: (12.5, 0.5), 135.0: (5.0, 0.0)}),
    ("up-and-out, record at 120 paying 7", BarrierPDESolver, lambda: at_maturity(single(KO), D, dict(barrier=120.0, payoff=7.0)), {125.0: (7.0, 0.0), 115.0: (15.0, 1.0)}),
    ("up-and-out, ACCUMULATE, a maturity record", BarrierPDESolver, lambda: at_maturity(single(KO), D, dict(barrier=130.0, payoff=5.0), ACC), {125.0: (25.0, 1.0), 135.0: (40.0, 1.0)}),
    ("up-and-in, continuous", BarrierPDESolver, lambda: at_maturity(single(KI), C), {135.0: (35.0, 1.0), 125.0: (5.0, 0.0)}),
    ("up-and-in, a maturity record", BarrierPDESolver, lambda: at_maturity(single(KI), D), {135.0: (35.0, 1.0), 125.0: (5.0, 0.0)}),
    ("up-and-in, participation 0.5", BarrierPDESolver, lambda: at_maturity(single(KI, participation=0.5), C), {135.0: (17.5, 0.5), 125.0: (5.0, 0.0)}),
    ("double knock-out, continuous", DoubleBarrierPDESolver, lambda: at_maturity(double(DKO), C), {125.0: (25.0, 1.0), 135.0: (5.0, 0.0), 75.0: (5.0, 0.0)}),
    ("double knock-out, EXPIRY", DoubleBarrierPDESolver, lambda: at_maturity(double(DKO), X), {125.0: (25.0, 1.0), 135.0: (5.0, 0.0)}),
    ("double knock-in, continuous", DoubleBarrierPDESolver, lambda: at_maturity(double(DKI), C), {135.0: (35.0, 1.0), 125.0: (5.0, 0.0)}),
    ("one-touch up 130, 10 at expiry", OneTouchPDESolver, lambda: at_maturity(touch(TouchType.ONE_TOUCH), C), {135.0: (10.0, 0.0), 125.0: (0.0, 0.0)}),
    ("no-touch up 130, 10 at expiry", OneTouchPDESolver, lambda: at_maturity(touch(TouchType.NO_TOUCH), C), {125.0: (10.0, 0.0), 135.0: (0.0, 0.0)}),
    ("double one-touch 80/130", DoubleOneTouchPDESolver, lambda: at_maturity(touch2(TouchType.DOUBLE_ONE_TOUCH), C), {135.0: (10.0, 0.0), 125.0: (0.0, 0.0)}),
    ("double no-touch 80/130", DoubleOneTouchPDESolver, lambda: at_maturity(touch2(TouchType.DOUBLE_NO_TOUCH), C), {125.0: (10.0, 0.0), 135.0: (0.0, 0.0)}),
    # a rebate paid at hit, with a terminal payment date of its own (400 days out): hit and terminal timing differ
    ("up-and-out, rebate at hit, settlement date", BarrierPDESolver, lambda: dated(single(KO, True)), {135.0: ("hit", 0.0)}),
    ("up-and-out, rebate at expiry, settlement date", BarrierPDESolver, lambda: dated(single(KO)), {135.0: ("terminal", 0.0)}),
    ("one-touch paid at hit, settlement date", OneTouchPDESolver, lambda: dated(lambda kw: OneTouchOption(
        barrier=130.0, barrier_direction=BarrierDirection.UP, rebate=5.0, payment_at_hit=True, touch_type=TouchType.ONE_TOUCH, **kw)), {135.0: ("hit", 0.0)}),
]
SPECIAL = {"hit": 5.0 * math.exp(-RATE * LAG), "terminal": 5.0 * math.exp(-RATE * 400.0 / 365.0)}
W2B_CELLS = [(name, solver, make, spot, amount, slope) for name, solver, make, truth in W2B
             for spot, (amount, slope) in truth.items()]


@pytest.mark.parametrize("name,solver,make,spot,amount,slope", W2B_CELLS,
                         ids=[f"{c[0]}@{c[3]}" for c in W2B_CELLS])
def test_terminal_read_cells(name, solver, make, spot, amount, slope):
    """§7.2 'maturity' row (ii): §4.12's 36 cells, amount x DF(payment) in price, Greeks and the session npv."""
    want, want_delta = (SPECIAL[amount], 0.0) if isinstance(amount, str) else (amount * DDF, slope * DDF)
    price, g, npv = three_ops(solver, make(), spot)
    for v in (price, g["price"], npv):
        assert abs(v - want) <= 1e-12, (v, want)
    assert abs(g["delta"] - want_delta) <= 1e-12 and g["gamma"] == 0.0


def test_knock_in_multiplier_at_maturity():
    one, ten = at_maturity(single(KI), C), at_maturity(single(KI), C)
    ten.contract_multiplier = 10.0
    for spot in (135.0, 125.0):
        a, b = BarrierPDESolver().price(one, env_const(spot)), BarrierPDESolver().price(ten, env_const(spot))
        assert abs(b - 10.0 * a) <= 1e-12 * max(1.0, abs(10.0 * a))
