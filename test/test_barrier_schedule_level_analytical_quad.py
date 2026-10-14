"""W18c: the analytical barrier routes read the schedule record's level and payoff on a DISCRETE schedule (spec
2026-10-13 follow-up queue I4, contract C6, AC8, amendment A8).

W18b (user approval 2026-10-10) made the Black-Scholes barrier MC decide at the valuation instant on the deciding
record's level and pay its payoff (``valuation_barrier_terms``). The analytical engines -- barrier, one-touch,
double barrier, single and double sharkfin -- still read ``product.barrier`` (``upper_barrier`` / ``lower_barrier``)
in their hit pre-checks, immediate knock-out cash and forward closed form. Now:

* every decision at the valuation instant (hit, expired branch, immediate knock-out cash) reads the deciding record:
  its level(s) and its payoff -- the record ON the valuation instant; with no record now nothing is decided and the
  contract is priced forward (A15, user decision 2026-10-14);
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

from golden_compare import GOLDEN_ABS_TOL, GOLDEN_REL_TOL
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
#: frozen literals cross architectures (test/golden_compare.py); TOL stays for same-run comparisons
GOLDEN = dict(rel=GOLDEN_REL_TOL, abs=GOLDEN_ABS_TOL)
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


def _set_cash(product, cash: float) -> None:
    """The product's own knock-out cash (``rebate``, a sharkfin's ``knock_out_rebate``), set after booking."""
    if hasattr(product, "knock_out_rebate"):
        product.knock_out_rebate = cash
    else:
        product.rebate = cash


def _price(kind: Kind, level, record_level, cell, payoff=None, product_cash=None):
    years, keep, which = CELLS[cell]
    product = kind.build(level, record_level, payoff=payoff)
    if product_cash is not None:
        _set_cash(product, product_cash)
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


#: the cells priced forward (the knock-out cash of a future hit enters) and one decided now
FORWARD_CELLS = ["forward_from_inception", "forward_mid_life", "record_now_inside",
                 "between_observations_beyond_the_records", "record_now_between"]


@pytest.mark.parametrize("product_cash", [None, 0.0], ids=["product_cash", "zero_product_cash"])
@pytest.mark.parametrize("cell", FORWARD_CELLS)
@pytest.mark.parametrize("name", [n for n, k in KINDS.items() if k.knocks_out])
def test_a_future_hit_pays_the_record_payoff(name, cell, product_cash):
    """Every record at L paying 4.5, the product's own knock-out cash 5 (a sharkfin's 3), or 0: a future hit pays
    the hit record's payoff (the MC's ``payoffs[first_idx]``, the PDE) -- bitwise the contract booked at L with cash
    4.5. A zero product cash does not skip the knock-out cash leg (the single sharkfin's touch leg)."""
    kind = KINDS[name]
    priced = _price(kind, kind.product_level, kind.record_level, cell, payoff=4.5, product_cash=product_cash)
    booked = _price(kind, kind.record_level, kind.record_level, cell, payoff=4.5, product_cash=4.5)
    assert priced == booked


def test_a_zero_cash_single_sharkfin_knocked_out_now_pays_the_record_payoff():
    """``knock_out_rebate`` 0, records paying 2.5: the record on the valuation instant breached at 115 pays 2.5 at
    the terminal payment time."""
    kind = KINDS["single_sharkfin_call"]
    priced = _price(kind, kind.product_level, kind.record_level, "record_now_between", payoff=2.5, product_cash=0.0)
    assert priced == pytest.approx(2.5 * env(100.0).get_discount_factor(0.75), **TOL)


@pytest.mark.parametrize("name", [n for n, k in KINDS.items() if k.knocks_out and "pay_at_hit" not in n])
def test_an_immediate_knock_out_pays_the_record_payoff(name):
    """A record on the valuation instant breached at its level L (short of ``product.barrier``) and paying 2.5:
    the contract pays 2.5 at the terminal payment time (the double barrier too since T17 concern 3: it paid 2.5
    undiscounted)."""
    kind = KINDS[name]
    priced = _price(kind, kind.product_level, kind.record_level, "record_now_between", payoff=2.5)
    expected = 2.5 * env(100.0).get_discount_factor(0.75)
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
    """The pre-W18c values (bitwise on the machine that took them; ``GOLDEN`` tolerance here for other architectures)."""
    engine, build, spot = PLAIN[name]
    product = build()
    if observation != "DISCRETE":
        product = _monitored(product, ObservationType[observation])
    assert _quiet(lambda: engine().price(product, env(spot))) == pytest.approx(BEFORE[observation, name], **GOLDEN)


#: plain schedules decided now: (label, aged years, keep the landing record, spot) -> pre-W18c value
PLAIN_DECIDED_NOW = {
    ("barrier_uoc_hit", 0.25, True, 121.0): 4.888756185966682,
    ("barrier_uoc_between_observations", 0.30, False, 121.0): 4.895228788316507,
    ("barrier_uoc_pah_hit", 0.25, True, 121.0): 5.0,
    ("one_touch_hit", 0.25, True, 121.0): 4.888756185966682,
    ("one_touch_between_observations", 0.30, False, 121.0): 4.325607738235779,
    ("double_ko_hit", 0.25, True, 131.0): 1.9555024743866727,    # 2.0 (undiscounted) before T17 concern 3
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
    assert _quiet(lambda: engine().price(product, env(spot))) == pytest.approx(PLAIN_DECIDED_NOW[case], **GOLDEN)


# ======================================================================================================================
# the QUAD adapters (AC8 QUAD half, T6): barrier, one-touch, double barrier
# ======================================================================================================================
#
# QUAD decides at the valuation instant as the Black-Scholes barrier MC does (W18b): a record ON the valuation instant
# decides, on its level(s), and a knock-out pays its payoff. With no record now nothing is decided, whatever the spot:
# the lattice prices it forward (A15, user decision 2026-10-14; it superseded the W18b "next observation" pre-check).
# The forward lattice reads each record (``resolve_schedule``); mixed levels price natively (no refusal).

def _quad():
    from quantark.asset.equity.engine.quad import DiscreteQuadEngine
    from quantark.asset.equity.param import QuadParams

    return DiscreteQuadEngine(params=QuadParams(grid_points=401))


#: the QUAD kinds: (analytical kind giving the contract and spots) -- the sharkfins have no QUAD adapter
def _double_touch(levels, upper, lower, payoff=None, touch_type=None, pay_at_hit=False):
    from quantark.asset.equity.product.option import DoubleOneTouchOption

    return DoubleOneTouchOption(upper_barrier=upper, lower_barrier=lower, maturity=1.0, rebate=5.0,
                                payment_at_hit=pay_at_hit, touch_type=touch_type or TouchType.DOUBLE_ONE_TOUCH,
                                observation_type=ObservationType.DISCRETE,
                                observation_schedule=_schedule(levels, payoff, double=True))


#: the QUAD-only kinds: the double one-touch / no-touch (no analytical engine prices them)
QUAD_ONLY_KINDS = {
    "double_one_touch": Kind(None, _double_kind(lambda lv, u, lo, **kw: _double_touch(lv, u, lo, **kw)),
                             (130.0, 75.0), (125.0, 80.0), 127.0, 100.0),
    "double_one_touch_pay_at_hit": Kind(None, _double_kind(
        lambda lv, u, lo, **kw: _double_touch(lv, u, lo, pay_at_hit=True, **kw)), (130.0, 75.0), (125.0, 80.0),
        77.0, 100.0),
    "double_no_touch": Kind(None, _double_kind(
        lambda lv, u, lo, **kw: _double_touch(lv, u, lo, touch_type=TouchType.DOUBLE_NO_TOUCH, **kw)),
        (130.0, 75.0), (125.0, 80.0), 127.0, 100.0, False),
}
#: the QUAD kinds: the analytical kinds QUAD prices (the sharkfins have no QUAD adapter) and the double touches
QK = {**{n: k for n, k in KINDS.items() if not n.startswith(("single_sharkfin", "double_sharkfin"))},
      **QUAD_ONLY_KINDS}
QUAD_KINDS = list(QK)

#: (aged years, keep the landing record, which spot); an uncrossed record ON the valuation instant is not a QUAD cell:
#: the lattice refuses it ("t_grid must be strictly increasing", pre-existing, the QUAD twin of I5)
QUAD_CELLS = {
    "forward_from_inception": (0.0, False, "inside"),
    "forward_mid_life": (0.30, False, "inside"),
    "beyond_the_next_record_at_inception": (0.0, False, "between"),
    "beyond_the_next_record_mid_life": (0.30, False, "between"),
    "record_now_between": (0.25, True, "between"),
    "expired_between": (1.0, True, "between"),
    "expired_inside": (1.0, True, "inside"),
}


def _quad_product(kind: Kind, level, record_level, cell, payoff=None, product_cash=None):
    years, keep, which = QUAD_CELLS[cell]
    product = kind.build(level, record_level, payoff=payoff)
    if product_cash is not None:
        _set_cash(product, product_cash)
    if years:
        _aged(product, years, keep_landing=keep)
    return product, (kind.between if which == "between" else kind.inside)


def _quad_price(kind: Kind, level, record_level, cell, payoff=None, product_cash=None):
    product, spot = _quad_product(kind, level, record_level, cell, payoff, product_cash)
    return _quiet(lambda: _quad().price(product, env(spot)))


@pytest.mark.parametrize("cell", list(QUAD_CELLS))
@pytest.mark.parametrize("name", QUAD_KINDS)
def test_quad_records_at_another_level_price_as_the_contract_booked_at_that_level(name, cell):
    """``product.barrier`` B, every record at L: QUAD prices bitwise the contract booked at L."""
    kind = QK[name]
    assert _quad_price(kind, kind.product_level, kind.record_level, cell) == _quad_price(
        kind, kind.record_level, kind.record_level, cell)


@pytest.mark.parametrize("cell", ["beyond_the_next_record_at_inception", "beyond_the_next_record_mid_life",
                                  "record_now_between", "expired_between"])
@pytest.mark.parametrize("name", [n for n in QUAD_KINDS if n.startswith("barrier_")])
def test_quad_decision_at_the_valuation_instant_matches_the_black_scholes_barrier_mc(name, cell):
    """The W18b reference, spot beyond the records' level L but short of ``product.barrier``. On a record (now, or
    the exercise date) a knock-out pays the record payoff (2.5) as the MC does, and a knock-in is the vanilla (QUAD's
    quadrature vanilla vs the MC's, within 4 s.e.) or, on the exercise date, the payoff. Beyond the next record with
    no record now nothing is decided (A15): QUAD's lattice and the MC's paths price forward, within 4 s.e."""
    from quantark.asset.equity.engine.mc import BarrierOptionMCEngine
    from quantark.asset.equity.param import MCParams

    kind = QK[name]
    payoff = 2.5 if kind.knocks_out else None
    product, spot = _quad_product(kind, kind.product_level, kind.record_level, cell, payoff=payoff)
    quad_value = _quiet(lambda: _quad().price(product, env(spot)))
    product, spot = _quad_product(kind, kind.product_level, kind.record_level, cell, payoff=payoff)
    mc = BarrierOptionMCEngine(params=MCParams(seed=11, num_paths=20000, time_steps=73))
    mc_value = _quiet(lambda: mc.price(product, env(spot)))
    decided_now = cell in ("record_now_between", "expired_between")
    if (kind.knocks_out and decided_now) or cell == "expired_between":
        assert quad_value == pytest.approx(mc_value, **TOL)
    else:
        assert abs(quad_value - mc_value) <= 4.0 * mc.get_last_std_error()


@pytest.mark.parametrize("cell", ["beyond_the_next_record_at_inception", "record_now_between", "expired_between"])
@pytest.mark.parametrize("name", [n for n in QUAD_KINDS if QK[n].knocks_out])
def test_quad_knock_out_pays_the_record_payoff(name, cell):
    """Records paying 2.5: a knock-out (one-touch) decided at the valuation instant (a record now, the exercise
    date) pays 2.5, and one priced forward beyond the next record (no record now, A15) pays 2.5 on a future hit --
    bitwise the contract booked with cash 2.5."""
    kind = QK[name]
    priced = _quad_price(kind, kind.product_level, kind.record_level, cell, payoff=2.5)
    booked = _quad_price(kind, kind.record_level, kind.record_level, cell, payoff=2.5, product_cash=2.5)
    assert priced == booked
    assert priced != _quad_price(kind, kind.record_level, kind.record_level, cell)


@pytest.mark.parametrize("cell", ["forward_from_inception", "forward_mid_life"])
@pytest.mark.parametrize("name", [n for n in QUAD_KINDS if QK[n].knocks_out])
def test_quad_zero_product_cash_still_prices_the_record_payoff(name, cell):
    """The product's knock-out cash 0, records paying 4.5: the forward lattice pays the records' payoff (its gate no
    longer reads only the product's rebate) -- bitwise the contract booked with cash 4.5."""
    kind = QK[name]
    priced = _quad_price(kind, kind.record_level, kind.record_level, cell, payoff=4.5, product_cash=0.0)
    booked = _quad_price(kind, kind.record_level, kind.record_level, cell, payoff=4.5, product_cash=4.5)
    assert priced == booked


@pytest.mark.parametrize("name", QUAD_KINDS)
def test_quad_prices_mixed_levels_natively(name):
    """QUAD has no closed-form restriction: mixed remaining levels price (finite, positive), no refusal."""
    import math

    kind = QK[name]
    product = kind.build(kind.product_level, kind.record_level)
    for record in product.observation_schedule.records[1::2]:
        _set_level(record, kind.product_level)
    value = _quiet(lambda: _quad().price(product, env(kind.inside)))
    assert math.isfinite(value) and value > 0.0


@pytest.mark.parametrize("record_level", ["product_level", "record_level"])
@pytest.mark.parametrize("name", QUAD_KINDS)
def test_quad_an_uncrossed_record_at_the_valuation_instant_leaves_the_lattice(name, record_level):
    """I5 on QUAD (A14): the record on the valuation instant, not crossed by the spot, is decided "not hit" there
    and leaves the lattice -- bitwise the contract aged past it (it raised "t_grid must be strictly increasing")."""
    kind = QK[name]
    level = getattr(kind, record_level)
    on_record = _aged(kind.build(kind.product_level, level), 0.25, keep_landing=True)
    aged_past = _aged(kind.build(kind.product_level, level), 0.25, keep_landing=False)
    e = env(kind.inside)
    assert _quiet(lambda: _quad().price(on_record, e)) == _quiet(lambda: _quad().price(aged_past, e))


def _sole_record(kind: Kind, keep_landing: bool):
    """The contract with only its 3-month record, aged onto it (``keep_landing``) or past it (monitoring ended)."""
    product = kind.build(kind.product_level, kind.record_level)
    product.observation_schedule.records = [product.observation_schedule.records[2]]
    return _aged(product, 0.25, keep_landing=keep_landing)


@pytest.mark.parametrize("name", QUAD_KINDS)
def test_quad_a_sole_uncrossed_record_at_the_valuation_instant_leaves_the_never_hit_value(name):
    """Nothing is left to observe: the never-hit value (A14) -- bitwise the contract whose monitoring has ended (its
    remaining-contract replication: a knock-out the vanilla, a knock-in or no-touch its rebate, a touch nothing)."""
    kind = QK[name]
    e = env(kind.inside)
    on_record = _quiet(lambda: _quad().price(_sole_record(kind, True), e))
    assert on_record == _quiet(lambda: _quad().price(_sole_record(kind, False), e))


@pytest.mark.parametrize("name", QUAD_KINDS)
def test_quad_two_records_on_the_valuation_instant_stay_refused(name):
    """A repeated observation time is refused, never silently filtered (T3 CP1)."""
    from copy import deepcopy

    kind = QK[name]
    product = _aged(kind.build(kind.product_level, kind.record_level), 0.25, keep_landing=True)
    records = product.observation_schedule.records
    records.insert(0, deepcopy(records[0]))
    with pytest.raises(ValidationError, match="strictly increasing"):
        _quiet(lambda: _quad().price(product, env(kind.inside)))


# ======================================================================================================================
# every route refuses a knock-in whose records pay other than the rebate (W18 Q3, Ruling R19)
# ======================================================================================================================

KNOCK_IN_RECORD_PAYOFF = {
    "barrier_analytical_up_in": (BarrierAnalyticalEngine, lambda: _barrier(_UP_IN_LEVELS, 120.0, BarrierType.UP_IN,
                                                                             payoff=4.5), 110.0, 5.0),
    "barrier_analytical_down_in": (BarrierAnalyticalEngine, lambda: _barrier(
        [85.0] * 12, 85.0, BarrierType.DOWN_IN, payoff=4.5, option_type=OptionType.PUT), 95.0, 5.0),
    "double_barrier_analytical_knock_in": (DoubleBarrierOptionAnalyticalEngine, lambda: _double(
        [(130.0, 75.0)] * 12, barrier_type=DoubleBarrierType.KNOCK_IN, payoff=1.5), 100.0, 2.0),
    "double_barrier_quad_knock_in": (_quad, lambda: _double(
        [(130.0, 75.0)] * 12, barrier_type=DoubleBarrierType.KNOCK_IN, payoff=1.5), 100.0, 2.0),
}
_UP_IN_LEVELS = [120.0] * 12


@pytest.mark.parametrize("name", list(KNOCK_IN_RECORD_PAYOFF))
def test_a_knock_in_whose_records_pay_other_than_the_rebate_is_refused(name):
    engine, build, spot, rebate = KNOCK_IN_RECORD_PAYOFF[name]
    paid = 1.5 if "double" in name else 4.5
    with pytest.raises(ValidationError, match=(rf"a knock-in has one rebate \({rebate}\); a schedule record pays "
                                               rf"{paid} \(spec 3C §5\.2 \(a\)\)")):
        _quiet(lambda: engine().price(build(), env(spot)))


@pytest.mark.parametrize("name", list(KNOCK_IN_RECORD_PAYOFF))
def test_a_knock_in_whose_records_pay_the_rebate_still_prices(name):
    import math

    engine, build, spot, rebate = KNOCK_IN_RECORD_PAYOFF[name]
    product = build()
    for record in product.observation_schedule.records:
        record.payoff = rebate
    assert math.isfinite(_quiet(lambda: engine().price(product, env(spot))))


# ======================================================================================================================
# QUAD unchanged on plain schedules and EXPIRY monitoring (pre-T6 values, 49019aa4)
# ======================================================================================================================

_QUAD_PLAIN = {
    "barrier_uoc": (lambda: _barrier([120.0] * 12), 110.0, 121.0),
    "barrier_uic": (lambda: _barrier([120.0] * 12, barrier_type=BarrierType.UP_IN), 110.0, 121.0),
    "barrier_uoc_pah": (lambda: _barrier([120.0] * 12, pay_at_hit=True), 110.0, 121.0),
    "barrier_dop": (lambda: _barrier([85.0] * 12, 85.0, BarrierType.DOWN_OUT, option_type=OptionType.PUT), 95.0,
                    84.0),
    "one_touch": (lambda: _touch([120.0] * 12, 120.0, BarrierDirection.UP, TouchType.ONE_TOUCH), 110.0, 121.0),
    "no_touch": (lambda: _touch([85.0] * 12, 85.0, BarrierDirection.DOWN, TouchType.NO_TOUCH), 95.0, 84.0),
    "double_ko": (lambda: _double([(130.0, 75.0)] * 12), 100.0, 131.0),
    "double_ki": (lambda: _double([(130.0, 75.0)] * 12, barrier_type=DoubleBarrierType.KNOCK_IN), 100.0, 131.0),
}
#: cell -> (aged years, keep the landing record, spot "inside" / "hit")
_QUAD_PLAIN_CELLS = {
    "DISCRETE": (0.0, False, "inside"),
    "DISCRETE_mid": (0.30, False, "inside"),
    "DISCRETE_hit": (0.25, True, "hit"),
    "DISCRETE_expired": (1.0, True, "inside"),
    "DISCRETE_expired_hit": (1.0, True, "hit"),
    "EXPIRY": (0.0, False, "inside"),
}
QUAD_BEFORE = {
    ('DISCRETE', 'barrier_uoc'): 4.077778296932585,
    ('DISCRETE_mid', 'barrier_uoc'): 4.470149971687532,
    ('DISCRETE_hit', 'barrier_uoc'): 4.888756185966682,
    ('DISCRETE_expired', 'barrier_uoc'): 10.0,
    ('DISCRETE_expired_hit', 'barrier_uoc'): 5.0,
    ('EXPIRY', 'barrier_uoc'): 4.925371919411028,
    ('DISCRETE', 'barrier_uic'): 16.228761998030937,
    ('DISCRETE_mid', 'barrier_uic'): 14.406170434218108,
    ('DISCRETE_hit', 'barrier_uic'): 23.399122925181604,
    ('DISCRETE_expired', 'barrier_uic'): 5.0,
    ('DISCRETE_expired_hit', 'barrier_uic'): 21.0,
    ('EXPIRY', 'barrier_uic'): 15.381168375552495,
    ('DISCRETE', 'barrier_uoc_pah'): 4.1268494885139235,
    ('DISCRETE_mid', 'barrier_uoc_pah'): 4.4968220699763854,
    ('DISCRETE_hit', 'barrier_uoc_pah'): 5.0,
    ('DISCRETE_expired', 'barrier_uoc_pah'): 10.0,
    ('DISCRETE_expired_hit', 'barrier_uoc_pah'): 5.0,
    ('EXPIRY', 'barrier_uoc_pah'): 4.925371919411028,
    ('DISCRETE', 'barrier_dop'): 3.374765744809814,
    ('DISCRETE_mid', 'barrier_dop'): 3.526535496399273,
    ('DISCRETE_hit', 'barrier_dop'): 4.888756185966682,
    ('DISCRETE_expired', 'barrier_dop'): 5.0,
    ('DISCRETE_expired_hit', 'barrier_dop'): 5.0,
    ('EXPIRY', 'barrier_dop'): 3.6915797140083377,
    ('DISCRETE', 'one_touch'): 2.6718663389898367,
    ('DISCRETE_mid', 'one_touch'): 2.3343845560290353,
    ('DISCRETE_hit', 'one_touch'): 4.888756185966682,
    ('DISCRETE_expired', 'one_touch'): 0.0,
    ('DISCRETE_expired_hit', 'one_touch'): 5.0,
    ('EXPIRY', 'one_touch'): 1.6097778111845367,
    ('DISCRETE', 'no_touch'): 2.554375009991321,
    ('DISCRETE_mid', 'no_touch'): 2.9699390061977775,
    ('DISCRETE_hit', 'no_touch'): 0.0,
    ('DISCRETE_expired', 'no_touch'): 5.0,
    ('DISCRETE_expired_hit', 'no_touch'): 0.0,
    ('EXPIRY', 'no_touch'): 3.4496348894447455,
    ('DISCRETE', 'double_ko'): 5.1624911886851566,
    ('DISCRETE_mid', 'double_ko'): 5.646571886050389,
    ('DISCRETE_hit', 'double_ko'): 1.9555024743866727,
    ('DISCRETE_expired', 'double_ko'): 0.0,
    ('DISCRETE_expired_hit', 'double_ko'): 2.0,
    ('EXPIRY', 'double_ko'): 6.00231331644959,
    ('DISCRETE', 'double_ki'): 5.605721900746439,
    ('DISCRETE_mid', 'double_ki'): 3.6003500096404997,
    ('DISCRETE_hit', 'double_ki'): 32.651873420278896,
    ('DISCRETE_expired', 'double_ki'): 2.0,
    ('DISCRETE_expired_hit', 'double_ki'): 31.0,
    ('EXPIRY', 'double_ki'): 4.765899772982006,
}


@pytest.mark.parametrize("cell,name", list(QUAD_BEFORE))
def test_quad_plain_schedules_and_expiry_monitoring_are_unchanged(cell, name):
    build, inside, hit = _QUAD_PLAIN[name]
    years, keep, which = _QUAD_PLAIN_CELLS[cell]
    product = build()
    if cell == "EXPIRY":
        product = _monitored(product, ObservationType.EXPIRY)
    elif years:
        _aged(product, years, keep_landing=keep)
    spot = hit if which == "hit" else inside
    assert _quiet(lambda: _quad().price(product, env(spot))) == pytest.approx(QUAD_BEFORE[cell, name], **GOLDEN)


#: plain double one-touch / no-touch on QUAD (records at 130/75), pre-T6-follow-up values (179da8b6):
#: (touch type, hit-paid) -> {cell: value}; cells as ``_QUAD_PLAIN_CELLS`` (spot 100 inside, 131 hit)
DOUBLE_TOUCH_BEFORE = {
    ("DOUBLE_ONE_TOUCH", False): {"DISCRETE": 1.237339494728514, "DISCRETE_mid": 0.686059220544024,
                                  "DISCRETE_hit": 4.888756185966682,
                                  "DISCRETE_expired": 0.0, "DISCRETE_expired_hit": 5.0, "EXPIRY": 0.8246314413957159},
    ("DOUBLE_ONE_TOUCH", True): {"DISCRETE": 1.2487415733305751, "DISCRETE_mid": 0.689481565224967,
                                 "DISCRETE_hit": 5.0, "DISCRETE_expired": 0.0,
                                 "DISCRETE_expired_hit": 5.0, "EXPIRY": 0.8246314413957159},
    ("DOUBLE_NO_TOUCH", False): {"DISCRETE": 3.6148881730140268, "DISCRETE_mid": 4.2100356023032734,
                                 "DISCRETE_hit": 0.0, "DISCRETE_expired": 5.0,
                                 "DISCRETE_expired_hit": 0.0, "EXPIRY": 4.027596226346825},
}


@pytest.mark.parametrize("cell", list(_QUAD_PLAIN_CELLS))
@pytest.mark.parametrize("touch,hit_paid", list(DOUBLE_TOUCH_BEFORE))
def test_quad_plain_double_touch_is_unchanged(touch, hit_paid, cell):
    years, keep, which = _QUAD_PLAIN_CELLS[cell]
    product = _double_touch([(130.0, 75.0)] * 12, 130.0, 75.0, touch_type=TouchType[touch], pay_at_hit=hit_paid)
    if cell == "EXPIRY":
        product = _monitored(product, ObservationType.EXPIRY)
    elif years:
        _aged(product, years, keep_landing=keep)
    spot = 131.0 if which == "hit" else 100.0
    assert _quiet(lambda: _quad().price(product, env(spot))) == pytest.approx(
        DOUBLE_TOUCH_BEFORE[touch, hit_paid][cell], **GOLDEN)


#: A15 (user decision 2026-10-14, T17): the plain schedules aged 0.30 (no record on the valuation instant, the next
#: at 1/3) with the spot beyond the next record's level are priced FORWARD by the lattice. Before T17 they were decided
#: at once on the next record (W18b / T6): name -> (pre-T17 value, A15 value). The Black-Scholes PDE agrees within
#: 2e-4 except the double barrier with a rebate (0.05: a QUAD / PDE rebate gap its inside cells share, pre-existing).
QUAD_A15_BEYOND = {
    "barrier_uoc": (4.896094822847298, 4.869282182116993),
    "barrier_uic": (23.202892679985446, 23.22970532071575),
    "barrier_uoc_pah": (5.0, 4.947535814554486),
    "barrier_dop": (4.896094822847298, 4.717181787602059),
    "one_touch": (4.896094822847298, 4.302710336340313),
    "no_touch": (0.0, 0.5319679607803076),
    "double_ko": (1.9584379291389191, 3.0769000717546238),
    "double_ki": (32.511223884158724, 31.39276174154302),
}
#: (touch type, hit-paid) -> (pre-T17 value, A15 value), spot 131
DOUBLE_TOUCH_A15_BEYOND = {
    ("DOUBLE_ONE_TOUCH", False): (4.896094822847298, 4.2930175553665935),
    ("DOUBLE_ONE_TOUCH", True): (5.0, 4.370896899871662),
    ("DOUBLE_NO_TOUCH", False): (0.0, 0.6030772674807041),
}


@pytest.mark.parametrize("name", list(QUAD_A15_BEYOND))
def test_quad_plain_schedules_beyond_the_next_record_price_forward(name):
    build, _, hit = _QUAD_PLAIN[name]
    product = _aged(build(), 0.30, keep_landing=False)
    before, after = QUAD_A15_BEYOND[name]
    value = _quiet(lambda: _quad().price(product, env(hit)))
    assert value == pytest.approx(after, **GOLDEN)
    assert abs(value - before) > 1e-3


@pytest.mark.parametrize("touch,hit_paid", list(DOUBLE_TOUCH_A15_BEYOND))
def test_quad_plain_double_touch_beyond_the_next_record_prices_forward(touch, hit_paid):
    product = _double_touch([(130.0, 75.0)] * 12, 130.0, 75.0, touch_type=TouchType[touch], pay_at_hit=hit_paid)
    _aged(product, 0.30, keep_landing=False)
    before, after = DOUBLE_TOUCH_A15_BEYOND[touch, hit_paid]
    value = _quiet(lambda: _quad().price(product, env(131.0)))
    assert value == pytest.approx(after, **GOLDEN)
    assert abs(value - before) > 1e-3
