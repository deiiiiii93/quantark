"""W18c: the analytical barrier routes read the schedule record's level and payoff on a DISCRETE schedule (spec
2026-10-13 follow-up queue I4, contract C6, AC8, amendment A8).

W18b (user approval 2026-10-10) made the Black-Scholes barrier MC decide at the valuation instant on the deciding
record's level and pay its payoff (``valuation_barrier_terms``). The analytical engines -- barrier, one-touch,
double barrier, single and double sharkfin -- still read ``product.barrier`` (``upper_barrier`` / ``lower_barrier``)
in their hit pre-checks, immediate knock-out cash and forward closed form. Now:

* every decision at the valuation instant (hit, expired branch, immediate knock-out cash) reads the deciding record:
  its level(s) and its payoff;
* forward pricing uses the level(s) every REMAINING record (t > 0) shares -- it may differ from ``product.barrier``;
* remaining records with different levels have no closed form: ``ObservationSchedule.assert_analytical_ready``
  raises ``ValidationError``, as it does for mixed payoffs (A8);
* CONTINUOUS / EXPIRY monitoring and plain schedules (records at ``product.barrier``) are unchanged.

A contract with ``product.barrier`` B and every record at L is the same contract as the one booked with B = L: the
engines price both bitwise equal.
"""
from __future__ import annotations

import warnings
from datetime import datetime

import pytest

from quantark.asset.equity.engine import barrier_schedule_rules
from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
from quantark.asset.equity.engine.analytical.double_barrier_option_engine import DoubleBarrierOptionAnalyticalEngine
from quantark.asset.equity.engine.analytical.double_sharkfin_option_analytical_engine import (
    DoubleSharkfinOptionAnalyticalEngine,
)
from quantark.asset.equity.engine.analytical.one_touch_analytical_engine import OneTouchAnalyticalEngine
from quantark.asset.equity.engine.analytical.single_sharkfin_option_analytical_engine import (
    SingleSharkfinOptionAnalyticalEngine,
)
from quantark.asset.equity.engine.mc import barrier_option_mc_engine
from quantark.asset.equity.product.option import (
    BarrierOption,
    DoubleBarrierOption,
    DoubleSharkfinOption,
    OneTouchOption,
    SingleSharkfinOption,
)
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import (
    BarrierDirection,
    BarrierType,
    DoubleBarrierType,
    ObservationType,
    OptionType,
    TouchType,
)
from quantark.util.exceptions import ValidationError

VALUATION = datetime(2025, 1, 6)
#: twelve monthly observations on a regular grid (the analytical barrier shift needs one)
TIMES = [k / 12 for k in range(1, 13)]
TOL = dict(rel=1e-12, abs=1e-12)
MIXED = "differ across the remaining observation records"


def env(spot: float) -> PricingEnvironment:
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=0.20),
                              rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=VALUATION)


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


def _schedule(levels, payoff=None, double=False) -> ObservationSchedule:
    extra = {} if payoff is None else {"payoff": payoff}
    if double:
        return ObservationSchedule(records=[
            ObservationRecord(observation_time=t, upper_barrier=u, lower_barrier=lo, **extra)
            for t, (u, lo) in zip(TIMES, levels)])
    return ObservationSchedule(records=[
        ObservationRecord(observation_time=t, barrier=lv, **extra) for t, lv in zip(TIMES, levels)])


def _aged(product, years: float, keep_landing: bool = True):
    """Aged ``years`` (past records dropped); ``keep_landing`` keeps a record landing on the valuation instant."""
    product.time_shift(years, VALUATION, env(100.0), keep_landing=keep_landing)
    return product


# ======================================================================================================================
# the neutral rules module (C6, A8)
# ======================================================================================================================

def _barrier(levels, barrier=120.0, barrier_type=BarrierType.UP_OUT, payoff=None, pay_at_hit=False,
             option_type=OptionType.CALL) -> BarrierOption:
    return BarrierOption(strike=100.0, option_type=option_type, barrier=barrier, barrier_type=barrier_type,
                         rebate=5.0, pay_at_hit=pay_at_hit, maturity=1.0, observation_type=ObservationType.DISCRETE,
                         observation_schedule=_schedule(levels, payoff))


def _double(levels, upper=130.0, lower=75.0, barrier_type=DoubleBarrierType.KNOCK_OUT, payoff=None):
    return DoubleBarrierOption(strike=100.0, option_type=OptionType.CALL, upper_barrier=upper, lower_barrier=lower,
                               barrier_type=barrier_type, rebate=2.0, maturity=1.0,
                               observation_type=ObservationType.DISCRETE,
                               observation_schedule=_schedule(levels, payoff, double=True))


def test_valuation_barrier_terms_moved_to_the_neutral_rules_module_and_stays_importable_from_the_mc_module():
    assert barrier_option_mc_engine.valuation_barrier_terms is barrier_schedule_rules.valuation_barrier_terms


def test_valuation_barrier_terms_reads_the_deciding_record():
    product = _aged(_barrier([115.0] * 12, payoff=4.5), 0.25)
    assert barrier_schedule_rules.valuation_barrier_terms(product, env(100.0)) == (115.0, 4.5)


def test_valuation_double_barrier_terms_reads_the_deciding_record():
    product = _aged(_double([(125.0, 80.0)] * 12, payoff=1.5), 0.25)
    assert barrier_schedule_rules.valuation_double_barrier_terms(product, env(100.0)) == (125.0, 80.0, 1.5)


def test_valuation_double_barrier_terms_outside_discrete_monitoring_reads_the_product():
    product = DoubleBarrierOption(strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0, lower_barrier=75.0,
                                  barrier_type=DoubleBarrierType.KNOCK_OUT, rebate=2.0, maturity=1.0,
                                  observation_type=ObservationType.CONTINUOUS)
    assert barrier_schedule_rules.valuation_double_barrier_terms(product, env(100.0)) == (130.0, 75.0, 2.0)


def test_mixed_remaining_levels_have_no_closed_form():
    product = _barrier([115.0, 120.0] * 6)
    with pytest.raises(ValidationError, match=MIXED):
        product.observation_schedule.assert_analytical_ready(default_payoff=5.0, pricing_env=env(100.0))


def test_mixed_remaining_double_levels_have_no_closed_form():
    product = _double([(125.0, 80.0), (125.0, 75.0)] * 6)
    with pytest.raises(ValidationError, match=MIXED):
        product.observation_schedule.assert_analytical_ready(default_payoff=2.0, pricing_env=env(100.0))


def test_a_record_on_the_valuation_instant_is_not_a_remaining_record():
    """The t = 0 record is decided at valuation: its level may differ from the uniform remaining level."""
    product = _aged(_barrier([118.0, 118.0, 110.0] + [115.0] * 9), 0.25)
    product.observation_schedule.assert_analytical_ready(default_payoff=5.0, pricing_env=env(100.0))


# ======================================================================================================================
# the analytical engines (AC8 analytical half)
# ======================================================================================================================

def _touch(levels, barrier, direction, touch_type, payoff=None, pay_at_hit=False) -> OneTouchOption:
    return OneTouchOption(barrier=barrier, barrier_direction=direction, maturity=1.0, rebate=5.0,
                          payment_at_hit=pay_at_hit, touch_type=touch_type, observation_type=ObservationType.DISCRETE,
                          observation_schedule=_schedule(levels, payoff))


def _single_sharkfin(levels, barrier, option_type, payoff=None, pay_at_hit=False) -> SingleSharkfinOption:
    return SingleSharkfinOption(strike=100.0, option_type=option_type, barrier=barrier, maturity=1.0,
                                knock_out_rebate=3.0, no_hit_rebate=1.0, pay_at_hit=pay_at_hit,
                                observation_type=ObservationType.DISCRETE,
                                observation_schedule=_schedule(levels, payoff))


def _double_sharkfin(levels, upper, lower, option_type, payoff=None, pay_at_hit=False) -> DoubleSharkfinOption:
    return DoubleSharkfinOption(strike=100.0, option_type=option_type, upper_barrier=upper, lower_barrier=lower,
                                maturity=1.0, knock_out_rebate=3.0, no_hit_rebate=1.0, pay_at_hit=pay_at_hit,
                                observation_type=ObservationType.DISCRETE,
                                observation_schedule=_schedule(levels, payoff, double=True))


class Kind:
    """One contract kind: ``build(product_level, record_level, payoff=None)`` with the record level on all twelve
    records, its engine, the spot strictly between the two levels (beyond the records' level only), the spot inside
    both, and whether a hit now pays knock-out cash (a knock-in or a no-touch does not)."""

    def __init__(self, engine, build, product_level, record_level, between, inside, knocks_out=True):
        self.engine, self.build = engine, build
        self.product_level, self.record_level = product_level, record_level
        self.between, self.inside, self.knocks_out = between, inside, knocks_out


def _single(build_fn):
    return lambda level, record_level, payoff=None: build_fn([record_level] * 12, level, payoff=payoff)


def _double_kind(build_fn):
    return lambda level, record_level, payoff=None: build_fn([record_level] * 12, *level, payoff=payoff)


KINDS = {
    "barrier_up_out_call": Kind(BarrierAnalyticalEngine, _single(
        lambda lv, b, **kw: _barrier(lv, b, BarrierType.UP_OUT, **kw)), 120.0, 115.0, 117.0, 110.0),
    "barrier_up_out_call_pay_at_hit": Kind(BarrierAnalyticalEngine, _single(
        lambda lv, b, **kw: _barrier(lv, b, BarrierType.UP_OUT, pay_at_hit=True, **kw)), 120.0, 115.0, 117.0, 110.0),
    "barrier_up_in_call": Kind(BarrierAnalyticalEngine, _single(
        lambda lv, b, **kw: _barrier(lv, b, BarrierType.UP_IN, **kw)), 120.0, 115.0, 117.0, 110.0, False),
    "barrier_down_out_put": Kind(BarrierAnalyticalEngine, _single(
        lambda lv, b, **kw: _barrier(lv, b, BarrierType.DOWN_OUT, option_type=OptionType.PUT, **kw)),
        85.0, 90.0, 87.0, 95.0),
    "barrier_down_in_put": Kind(BarrierAnalyticalEngine, _single(
        lambda lv, b, **kw: _barrier(lv, b, BarrierType.DOWN_IN, option_type=OptionType.PUT, **kw)),
        85.0, 90.0, 87.0, 95.0, False),
    "one_touch_up": Kind(OneTouchAnalyticalEngine, _single(
        lambda lv, b, **kw: _touch(lv, b, BarrierDirection.UP, TouchType.ONE_TOUCH, **kw)), 120.0, 115.0, 117.0,
        110.0),
    "one_touch_up_pay_at_hit": Kind(OneTouchAnalyticalEngine, _single(
        lambda lv, b, **kw: _touch(lv, b, BarrierDirection.UP, TouchType.ONE_TOUCH, pay_at_hit=True, **kw)),
        120.0, 115.0, 117.0, 110.0),
    "no_touch_down": Kind(OneTouchAnalyticalEngine, _single(
        lambda lv, b, **kw: _touch(lv, b, BarrierDirection.DOWN, TouchType.NO_TOUCH, **kw)), 85.0, 90.0, 87.0, 95.0,
        False),
    "double_knock_out": Kind(DoubleBarrierOptionAnalyticalEngine, _double_kind(
        lambda lv, u, lo, **kw: _double(lv, u, lo, **kw)), (130.0, 75.0), (125.0, 80.0), 127.0, 100.0),
    "double_knock_in": Kind(DoubleBarrierOptionAnalyticalEngine, _double_kind(
        lambda lv, u, lo, **kw: _double(lv, u, lo, DoubleBarrierType.KNOCK_IN, **kw)), (130.0, 75.0), (125.0, 80.0),
        77.0, 100.0, False),
    "single_sharkfin_call": Kind(SingleSharkfinOptionAnalyticalEngine, _single(
        lambda lv, b, **kw: _single_sharkfin(lv, b, OptionType.CALL, **kw)), 120.0, 115.0, 117.0, 110.0),
    "single_sharkfin_put_pay_at_hit": Kind(SingleSharkfinOptionAnalyticalEngine, _single(
        lambda lv, b, **kw: _single_sharkfin(lv, b, OptionType.PUT, pay_at_hit=True, **kw)), 85.0, 90.0, 87.0, 95.0),
    "double_sharkfin_call": Kind(DoubleSharkfinOptionAnalyticalEngine, _double_kind(
        lambda lv, u, lo, **kw: _double_sharkfin(lv, u, lo, OptionType.CALL, **kw)), (130.0, 75.0), (125.0, 80.0),
        127.0, 100.0),
    "double_sharkfin_put_pay_at_hit": Kind(DoubleSharkfinOptionAnalyticalEngine, _double_kind(
        lambda lv, u, lo, **kw: _double_sharkfin(lv, u, lo, OptionType.PUT, pay_at_hit=True, **kw)),
        (130.0, 75.0), (125.0, 80.0), 77.0, 100.0),
}

#: (aged years, keep the record landing on the valuation instant, which spot): the cells
CELLS = {
    "forward_from_inception": (0.0, False, "inside"),
    "forward_mid_life": (0.30, False, "inside"),
    "record_now_inside": (0.25, True, "inside"),
    "record_now_between": (0.25, True, "between"),
    "between_observations_beyond_the_records": (0.30, False, "between"),
    "expired_between": (1.0, True, "between"),
    "expired_inside": (1.0, True, "inside"),
}


def _price(kind: Kind, level, record_level, cell, payoff=None):
    years, keep, which = CELLS[cell]
    product = kind.build(level, record_level, payoff=payoff)
    if years:
        _aged(product, years, keep_landing=keep)
    spot = kind.between if which == "between" else kind.inside
    return _quiet(lambda: kind.engine().price(product, env(spot)))


@pytest.mark.parametrize("cell", list(CELLS))
@pytest.mark.parametrize("name", list(KINDS))
def test_records_at_another_level_price_as_the_contract_booked_at_that_level(name, cell):
    """``product.barrier`` B, every record at L: bitwise the contract booked at L -- hit, expired and pay-at-hit
    decisions on L, the forward closed form on L."""
    kind = KINDS[name]
    priced = _price(kind, kind.product_level, kind.record_level, cell)
    booked = _price(kind, kind.record_level, kind.record_level, cell)
    assert priced == booked


@pytest.mark.parametrize("name", [n for n, k in KINDS.items() if k.knocks_out and "pay_at_hit" not in n])
def test_an_immediate_knock_out_pays_the_record_payoff(name):
    """A record on the valuation instant breached at its level L (short of ``product.barrier``) and paying 2.5:
    the contract pays 2.5 -- at the terminal payment time (the double barrier: undiscounted, as its rebate)."""
    kind = KINDS[name]
    priced = _price(kind, kind.product_level, kind.record_level, "record_now_between", payoff=2.5)
    expected = 2.5 if name == "double_knock_out" else 2.5 * env(100.0).get_discount_factor(0.75)
    assert priced == pytest.approx(expected, **TOL)


@pytest.mark.parametrize("name", [n for n, k in KINDS.items() if k.knocks_out and "pay_at_hit" in n])
def test_an_immediate_hit_paid_knock_out_pays_the_record_payoff(name):
    """Hit-paid: the record's payoff instead of the product's cash, at the same payment time."""
    kind = KINDS[name]
    paid = _price(kind, kind.product_level, kind.record_level, "record_now_between", payoff=2.5)
    reference = _price(kind, kind.record_level, kind.record_level, "record_now_between")
    product_cash = 3.0 if "sharkfin" in name else 5.0
    assert paid == pytest.approx(reference * 2.5 / product_cash, **TOL)


@pytest.mark.parametrize("cell", ["record_now_between", "expired_between"])
@pytest.mark.parametrize("name", [n for n in KINDS if n.startswith("barrier_")])
def test_the_decision_at_the_valuation_instant_matches_the_black_scholes_barrier_mc(name, cell):
    """The W18b reference: on a record at the valuation instant (mid-life, and on the exercise date), breached at
    its level L short of ``product.barrier``, the analytical value equals the Black-Scholes barrier MC's settled
    decision -- the record payoff on a knock-out, the vanilla on a knock-in (MC within 4 s.e.)."""
    from quantark.asset.equity.engine.mc import BarrierOptionMCEngine
    from quantark.asset.equity.param import MCParams

    kind = KINDS[name]
    payoff = 2.5 if kind.knocks_out else None
    analytical = _price(kind, kind.product_level, kind.record_level, cell, payoff=payoff)
    years, keep, _ = CELLS[cell]
    product = _aged(kind.build(kind.product_level, kind.record_level, payoff=payoff), years, keep_landing=keep)
    mc = BarrierOptionMCEngine(params=MCParams(seed=11, num_paths=20000, time_steps=73))
    value = _quiet(lambda: mc.price(product, env(kind.between)))
    if kind.knocks_out or cell == "expired_between":
        assert analytical == pytest.approx(value, **TOL)
    else:
        assert abs(analytical - value) <= 4.0 * mc.get_last_std_error()


def _set_level(record, level) -> None:
    if isinstance(level, tuple):
        record.upper_barrier, record.lower_barrier = level
    else:
        record.barrier = level


@pytest.mark.parametrize("name", list(KINDS))
def test_mixed_remaining_levels_refuse_the_forward_closed_form(name):
    kind = KINDS[name]
    product = kind.build(kind.product_level, kind.record_level)
    for record in product.observation_schedule.records[1::2]:
        _set_level(record, kind.product_level)
    with pytest.raises(ValidationError, match=MIXED):
        _quiet(lambda: kind.engine().price(product, env(kind.inside)))


@pytest.mark.parametrize("name", [n for n, k in KINDS.items() if k.knocks_out])
def test_mixed_remaining_levels_still_decide_a_knock_out_at_the_valuation_instant(name):
    """The record on the valuation instant is breached: decided now on its level, no forward closed form needed."""
    kind = KINDS[name]
    product = kind.build(kind.product_level, kind.record_level)
    for record in product.observation_schedule.records[1::2]:
        _set_level(record, kind.product_level)
    _aged(product, 0.25, keep_landing=True)          # the landing record (index 2) keeps the record level
    priced = _quiet(lambda: kind.engine().price(product, env(kind.between)))
    booked = _price(kind, kind.record_level, kind.record_level, "record_now_between")
    assert priced == booked


# ======================================================================================================================
# unchanged: CONTINUOUS / EXPIRY monitoring and plain schedules (records at the product's levels)
# ======================================================================================================================

def _monitored(product, observation_type):
    product.observation_type = observation_type
    product.observation_schedule = None
    product.observation_dates = None
    product.validate()
    return product


_UP, _CORRIDOR = [120.0] * 12, [(130.0, 75.0)] * 12
#: name -> (engine, build, spot); the plain contracts the values below were taken from on 796b667a (before W18c)
PLAIN = {
    "barrier_uoc": (BarrierAnalyticalEngine, lambda: _barrier(_UP), 110.0),
    "barrier_uic": (BarrierAnalyticalEngine, lambda: _barrier(_UP, barrier_type=BarrierType.UP_IN), 110.0),
    "barrier_uoc_pah": (BarrierAnalyticalEngine, lambda: _barrier(_UP, pay_at_hit=True), 110.0),
    "one_touch": (OneTouchAnalyticalEngine, lambda: _touch(_UP, 120.0, BarrierDirection.UP, TouchType.ONE_TOUCH),
                  110.0),
    "double_ko": (DoubleBarrierOptionAnalyticalEngine, lambda: _double(_CORRIDOR), 100.0),
    "single_sharkfin": (SingleSharkfinOptionAnalyticalEngine, lambda: _single_sharkfin(_UP, 120.0, OptionType.CALL),
                        110.0),
    "double_sharkfin": (DoubleSharkfinOptionAnalyticalEngine,
                        lambda: _double_sharkfin(_CORRIDOR, 130.0, 75.0, OptionType.CALL), 100.0),
}
BEFORE = {
    ("CONTINUOUS", "barrier_uoc"): 3.9447645256683375,
    ("CONTINUOUS", "barrier_uic"): 16.360994025370896,
    ("CONTINUOUS", "barrier_uoc_pah"): 4.014494123154026,
    ("CONTINUOUS", "one_touch"): 3.2195556223690733,
    ("CONTINUOUS", "double_ko"): 3.7460445429593667,
    ("CONTINUOUS", "single_sharkfin"): 2.9834766857954014,
    ("CONTINUOUS", "double_sharkfin"): 4.716490076507875,
    ("EXPIRY", "barrier_uoc"): 4.925371919411028,
    ("EXPIRY", "barrier_uic"): 15.380386631628205,
    ("EXPIRY", "barrier_uoc_pah"): 4.925371919411028,
    ("EXPIRY", "one_touch"): 1.6097778111845367,
    ("EXPIRY", "double_ko"): 5.177742879048815,
    ("EXPIRY", "single_sharkfin"): 4.9299507662488145,
    ("EXPIRY", "double_sharkfin"): 6.148188412597324,
    ("DISCRETE", "barrier_uoc"): 4.1452590567598735,
    ("DISCRETE", "barrier_uic"): 16.16049949427936,
    ("DISCRETE", "barrier_uoc_pah"): 4.195874026007574,
    ("DISCRETE", "one_touch"): 2.6510157507019025,
    ("DISCRETE", "double_ko"): 4.526659284923278,
    ("DISCRETE", "single_sharkfin"): 3.52509513988724,
    ("DISCRETE", "double_sharkfin"): 5.497104818471787,
}


@pytest.mark.parametrize("observation,name", list(BEFORE))
def test_continuous_expiry_and_plain_schedules_are_unchanged(observation, name):
    """The pre-W18c values (bitwise on the machine that took them; ``TOL`` here for other architectures)."""
    engine, build, spot = PLAIN[name]
    product = build()
    if observation != "DISCRETE":
        product = _monitored(product, ObservationType[observation])
    assert _quiet(lambda: engine().price(product, env(spot))) == pytest.approx(BEFORE[observation, name], **TOL)


#: plain schedules decided now: (label, aged years, keep the landing record, spot) -> pre-W18c value
PLAIN_DECIDED_NOW = {
    ("barrier_uoc_hit", 0.25, True, 121.0): 4.888756185966682,
    ("barrier_uoc_between_observations", 0.30, False, 121.0): 4.895228788316507,
    ("barrier_uoc_pah_hit", 0.25, True, 121.0): 5.0,
    ("one_touch_hit", 0.25, True, 121.0): 4.888756185966682,
    ("one_touch_between_observations", 0.30, False, 121.0): 4.325607738235779,
    ("double_ko_hit", 0.25, True, 131.0): 2.0,
    ("double_ko_between_observations", 0.30, False, 131.0): 3.014316895136673,
    ("single_sharkfin_hit", 0.25, True, 121.0): 2.933253711580009,
    ("double_sharkfin_hit", 0.25, True, 131.0): 2.933253711580009,
    ("double_sharkfin_between_observations", 0.30, False, 131.0): 3.9935358597061317,
}


@pytest.mark.parametrize("case", list(PLAIN_DECIDED_NOW), ids=[c[0] for c in PLAIN_DECIDED_NOW])
def test_plain_schedules_decided_at_the_valuation_instant_are_unchanged(case):
    label, years, keep, spot = case
    name = label.rsplit("_hit", 1)[0] if label.endswith("_hit") else label.split("_between")[0]
    engine, build, _ = PLAIN[name]
    product = _aged(build(), years, keep_landing=keep)
    assert _quiet(lambda: engine().price(product, env(spot))) == pytest.approx(PLAIN_DECIDED_NOW[case], **TOL)
