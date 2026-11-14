"""REBASED KO-reset cash legs end to end (unified event statistics spec AC16, D13; DESIGN §5).

Every cash-leg type on every trigger -- ``AccrualLeg`` (both ``KOBehavior`` values, both survival bases, both
payment conventions), ``FixedPayoffLeg`` (every ``PaymentTrigger``), ``AutocallableCashLeg`` (KO basis with each
terminal-bucket choice and ``PvFormula``; COUPON basis) -- valued through ``price_with_events`` on an un-knocked
REBASED KO-reset (Monte Carlo only, DESIGN §4.9a) and on an already-knocked-in one (priced, as the KO-reset tracker
prices it, through its ABSOLUTE copy), against a direct valuation of the same leg on the price's own paths.

The direct valuation reads nothing of the statistics or the distribution: ``mc_same_path_reference.ko_reset_paths``
takes the paths of a plain ``price()`` (same engine configuration, same seed) and decides each path's outcome, its
determination and settlement time, and its first knock-in from the contract; each leg's cash is then summed path by
path from the leg's own definition. Both sides use the same paths, so they agree to round-off.

The legs sit on the D13 deterministic candidate axis (controller ruling R2): every first-schedule date and every
knock-in date + offset, built here from the contract. A leg booked on the contract's dates alone lists no candidate
time and keeps refusing (the T6 ruling; the cash-leg bank's REBASED records).

A knock-in already in force at the valuation is a past event (controller ruling R1, DESIGN §4.9c): the AT_KI leg
takes nothing from the distribution on a knocked-in note (an unpaid past knock-in payment is a receivable, valued
where the producer emits one), while the statistics keep the carried knock-in as DESIGN §4.5's row at 0.

The folded route refuses a REBASED note (DESIGN §4.11: its settlement is knock-in dependent); on the knocked-in
note's ABSOLUTE copy every foldable leg folds and equals the same direct valuation.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from quantark.asset.equity.product.option.accrual import knock_in_anchor_for
from quantark.cashleg import (
    AccrualBasis,
    AccrualLeg,
    AutocallableCashLeg,
    AutocallableLegType,
    BaseAmount,
    BaseAmountMode,
    DeterministicLeg,
    EventType,
    FixedPayoffLeg,
    KOBehavior,
    LegDirection,
    LegSchedule,
    PaymentConvention,
    PaymentTrigger,
    PvFormula,
    SurvivalBasis,
)
from quantark.cashleg.folding import FoldMode, folded_trade_value
from quantark.execution.errors import CapabilityError
from quantark.util.enum.option_enums import PostKOScheduleMode
from quantark.util.exceptions import ValidationError

BANK_DIR = Path(__file__).resolve().parent / "event_stats_baseline"


def _load_cases():
    name = "_cashleg_event_stats_baseline_cases"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, BANK_DIR / "cases.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cases = _load_cases()

#: round-off between a leg on the distribution and the same leg summed path by path (relative to the leg's scale)
ROUND_OFF = 1e-12
#: the knocked-in note: the bank's contract aged 0.6y, knocked in on its second knock-in record (now at -0.1)
AGE, KNOCK_IN_RECORD, KNOCK_IN_TIME = 0.6, 1, -0.1
STATES = ("unknocked", "knocked_in")


def _product(state: str):
    """The bank's REBASED KO-reset; knocked in: the same contract aged ``AGE`` years (schedule 1 left at 0.15 and
    0.4), knocked in on its second knock-in record (``KNOCK_IN_TIME``), as the KO-reset tracker prices it -- the
    ABSOLUTE contract it has become (``with_absolute_schedule_2``: schedule 2 at -0.1 + offsets, to 0.9) carrying
    the knock-in (``_otc_lifecycle_knocked_in``)."""
    product = cases._ko_reset(PostKOScheduleMode.REBASED)
    if state == "unknocked":
        return product
    env = cases.flat_env()
    record = product.barrier_config.ki_observation_schedule.records[KNOCK_IN_RECORD]
    anchor = knock_in_anchor_for(record, KNOCK_IN_RECORD)
    product.time_shift(AGE, env.valuation_date, env)
    converted = product.with_absolute_schedule_2(KNOCK_IN_TIME, anchor=anchor)
    converted._otc_lifecycle_knocked_in = True
    return converted


def _engine():
    return cases.ENGINES["mc_snowball"].build()


def _times(schedule) -> list:
    return [float(r.observation_time) for r in schedule.records]


def axis_rows(state: str) -> list:
    """The knock-out candidate axis from the contract alone (D13), as (time, schedule) rows in the event route's
    order. Un-knocked: every first-schedule date and every knock-in date + every schedule-2 offset (one row per
    time). Knocked in: the ABSOLUTE copy's two schedules merged by time, the first schedule's row first at a tie
    (its schedule 1, which the carried knock-in has replaced for good, stays on the grid at zero mass)."""
    product = _product(state)
    pre = _times(product.barrier_config.ko_observation_schedule)
    post = _times(product.post_barrier_config.ko_observation_schedule)
    if state == "knocked_in":
        return sorted([(t, "pre") for t in pre] + [(t, "post") for t in post], key=lambda row: (row[0], row[1] != "pre"))
    ki = _times(product.barrier_config.ki_observation_schedule)
    return [(t, "any") for t in sorted(set(pre) | {k + o for k in ki for o in post})]


def candidate_axis(state: str) -> np.ndarray:
    return np.array([t for t, _ in axis_rows(state)], dtype=float)


def _accrual_periods(state: str):
    """Quarterly to 2y (un-knocked) or the knocked-in note's schedule-2 periods to its 0.9 end: every boundary on the
    candidate axis, so the survival read at each one is a path count."""
    if state == "unknocked":
        ends = np.arange(1, 9) * 0.25
    else:
        ends = np.array(_times(_product(state).post_barrier_config.ko_observation_schedule), dtype=float)
    return np.r_[0.0, ends[:-1]], ends


def legs(state: str) -> dict:
    """Every leg type x trigger on ``state``'s candidate axis (names are the leg ids)."""
    starts, ends = _accrual_periods(state)

    def accrual(behavior=KOBehavior.TRUNCATE_AT_KO, basis=SurvivalBasis.ENTER_PERIOD,
                convention=PaymentConvention.AT_PERIOD_END, pay_shift=0.0):
        return AccrualLeg(direction=LegDirection.BUYER_RECEIVES, rate=0.04,
                          base=BaseAmount(value=200_000.0, mode=BaseAmountMode.ABSOLUTE),
                          schedule=LegSchedule(period_starts=starts, period_ends=ends, payment_times=ends + pay_shift),
                          ko_behavior=behavior, survival_basis=basis, payment_convention=convention)

    obs = candidate_axis(state)
    settle = obs + 0.01

    def autocallable(leg_type, formula, terminal, basis=AccrualBasis.KO_MATURITY):
        return AutocallableCashLeg(
            direction=LegDirection.BUYER_PAYS, leg_type=leg_type, notional=1_000_000.0, rate=0.05,
            observation_schedule=tuple(obs), accrual_factors=tuple(np.linspace(0.3, 1.0, obs.size)),
            settlement_schedule=tuple(settle), terminal_accrual_factor=1.1,
            terminal_settlement_time=float(settle[-1]) + 0.01, pv_formula=formula, accrual_basis=basis,
            terminal_events=terminal)

    both = frozenset({EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI})
    out = {
        "deterministic": DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=5_000.0, payment_time=0.1),
        "accrual_truncate_enter": accrual(),
        "accrual_truncate_complete": accrual(basis=SurvivalBasis.COMPLETE_PERIOD, pay_shift=0.02),
        "accrual_truncate_at_maturity": accrual(convention=PaymentConvention.AT_MATURITY),
        "accrual_full_schedule": accrual(behavior=KOBehavior.PAY_FULL_SCHEDULE),
    }
    for trigger, offset, amount in ((PaymentTrigger.AT_KO, 2, 7_000.0), (PaymentTrigger.AT_KI, 3, 6_000.0),
                                    (PaymentTrigger.AT_MATURITY_NO_KO, 0, 2_000.0),
                                    (PaymentTrigger.AT_MATURITY_WITH_KI, 1, 4_000.0),
                                    (PaymentTrigger.AT_MATURITY_ANY, 3, 3_000.0)):
        out[f"fixed_{trigger.value}"] = FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=amount,
                                                       trigger=trigger, payment_offset_days=offset)
    out.update({
        "autocallable_margin_nmp_both": autocallable(AutocallableLegType.MARGIN, PvFormula.NOTIONAL_MINUS_PAYOFF, both),
        "autocallable_backend_premium_no_ko": autocallable(AutocallableLegType.BACKEND_PREMIUM, PvFormula.NORMAL,
                                                           frozenset({EventType.MATURITY_NO_KO})),
        "autocallable_rebate_with_ki": autocallable(AutocallableLegType.REBATE, PvFormula.NORMAL,
                                                    frozenset({EventType.MATURITY_WITH_KI})),
        "autocallable_minimum_return_both": autocallable(AutocallableLegType.MINIMUM_RETURN, PvFormula.NORMAL, both),
        "autocallable_backend_interest_coupon": autocallable(AutocallableLegType.BACKEND_INTEREST, PvFormula.NORMAL,
                                                             both, basis=AccrualBasis.COUPON),
    })
    return {name: replace(leg, leg_id=name, name=name) for name, leg in out.items()}


LEG_NAMES = tuple(legs("unknocked"))
#: legs a KO-reset cannot value: it has no coupon stream (DESIGN §4.3: ``coupons is None``), on either state
NO_COUPON_STREAM = ("autocallable_backend_interest_coupon",)


# --- the two sides ------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def event_route(state: str):
    """``price_with_events`` on a fresh engine: (npv, distribution, price() on another fresh engine, env)."""
    env = cases.flat_env()
    result = _engine().price_with_events(_product(state), env)
    return result.npv, result.event_distribution, _engine().price(_product(state), env), env


@lru_cache(maxsize=None)
def paths(state: str) -> dict:
    """The price's own paths, every outcome decided from the contract (``ko_reset_paths``), independent of the
    statistics."""
    from test_asset.equity.engine_mc import mc_same_path_reference

    return mc_same_path_reference.ko_reset_paths(_engine(), _product(state), cases.flat_env())


def _df(env, times) -> np.ndarray:
    return np.array([env.get_discount_factor(float(t)) for t in np.atleast_1d(times)], dtype=float)


def _alive(state: str, ref, t: float) -> float:
    """P(alive at t) on the paths, by each layout's survival definition (at the valuation every path is alive):

    * REBASED (the termination distribution's absolute survival, D13 / T9): not determined by t -- every outcome
      (knock-out, maturity, knocked-in settlement) leaves at its own determination;
    * ABSOLUTE (``ObservationTable.survival``, P(alive after the row); spec Contracts / F5): a knock-out leaves at
      its row; the never-knocked-in maturity leaves at its determination only when that precedes the last row; the
      knocked-in settlement, at the contract's end, never leaves."""
    if t <= 0.0:
        return 1.0
    event, det = ref["event"], ref["det"]
    if state == "unknocked":
        return float(np.count_nonzero(det > t)) / ref["n"]
    last_row = float(candidate_axis(state)[-1])
    gone = (np.isin(event, ("ko_pre", "ko_post")) & (det <= t)) | (
        (event == "maturity") & (det <= t) & (det < last_row))
    return float(np.count_nonzero(~gone)) / ref["n"]


def direct_value(state: str, name: str) -> float:
    """``name``'s value summed path by path on the price's own paths, from the leg's own definition."""
    leg = legs(state)[name]
    ref = paths(state)
    env = event_route(state)[3]
    n, event, det, settle = ref["n"], ref["event"], ref["det"], ref["settle"]
    sign = leg.sign()
    if isinstance(leg, DeterministicLeg):
        return sign * leg.amount * float(_df(env, leg.payment_time)[0])
    if isinstance(leg, AccrualLeg):
        frac = np.asarray(leg.schedule.period_ends - leg.schedule.period_starts, dtype=float)
        if leg.ko_behavior is KOBehavior.PAY_FULL_SCHEDULE:
            alive = np.ones(frac.size)
        else:
            reference = (leg.schedule.period_starts if leg.survival_basis is SurvivalBasis.ENTER_PERIOD
                         else leg.schedule.period_ends)
            alive = np.array([_alive(state, ref, float(t)) for t in reference])
        if leg.payment_convention is PaymentConvention.AT_MATURITY:
            # the leg's own definition: every period settles at the never-knocked-in maturity settlement, the
            # contract's first-schedule end (this contract settles with no lag: every matured path says so)
            maturity_pay = float(_product(state).get_pre_maturity_time(env))
            assert np.all(settle[event == "maturity"] == maturity_pay)
            pays = np.full(frac.size, maturity_pay)
        else:
            pays = np.asarray(leg.schedule.payment_times, dtype=float)
        return sign * leg.rate * leg.base.value * float(np.sum(frac * alive * _df(env, pays)))
    if isinstance(leg, FixedPayoffLeg):
        offset = leg.payment_offset_days / 365.0
        if leg.trigger is PaymentTrigger.AT_KI:
            if state == "knocked_in":
                return 0.0                  # R1: the knock-in is past; nothing from the distribution
            first = ref["ki_first"]
            return sign * leg.amount * float(np.sum(_df(env, ref["ki_time"][first] + offset))) / n
        events = {PaymentTrigger.AT_KO: ("ko_pre", "ko_post"), PaymentTrigger.AT_MATURITY_NO_KO: ("maturity",),
                  PaymentTrigger.AT_MATURITY_WITH_KI: ("ki_settlement",),
                  PaymentTrigger.AT_MATURITY_ANY: ("maturity", "ki_settlement")}[leg.trigger]
        mask = np.isin(event, events)
        return sign * leg.amount * float(np.sum(_df(env, settle[mask] + offset))) / n
    assert isinstance(leg, AutocallableCashLeg) and leg.accrual_basis is AccrualBasis.KO_MATURITY
    rows = axis_rows(state)
    af = np.asarray(leg.accrual_factors, dtype=float)
    ss_df = _df(env, leg.settlement_schedule)
    contingent = 0.0
    for p in np.flatnonzero(np.isin(event, ("ko_pre", "ko_post"))):
        schedule = "any" if state == "unknocked" else event[p][3:]
        # every knock-out lands on exactly one candidate row of its own schedule
        (i,) = [k for k, (t, tag) in enumerate(rows) if tag == schedule and t == det[p]]
        contingent += af[i] * ss_df[i]
    terminal_df = float(_df(env, leg.terminal_settlement_time)[0])
    if EventType.MATURITY_NO_KO in leg.terminal_events:
        contingent += leg.terminal_accrual_factor * terminal_df * np.count_nonzero(event == "maturity")
    if EventType.MATURITY_WITH_KI in leg.terminal_events:
        knocked = event == "ki_settlement"
        if state == "unknocked":
            # REBASED: the knocked-in branch settles with the note, at its own path-dependent date (D13)
            contingent += leg.terminal_accrual_factor * float(np.sum(_df(env, settle[knocked])))
        else:
            contingent += leg.terminal_accrual_factor * terminal_df * np.count_nonzero(knocked)
    r = leg.notional * leg.rate * contingent / n
    return sign * r if leg.pv_formula is PvFormula.NORMAL else sign * (leg.notional - r)


def _scale(leg) -> float:
    for attr in ("notional", "amount"):
        if getattr(leg, attr, 0.0):
            return float(getattr(leg, attr))
    return float(leg.base.value * leg.rate * 2.0)


# --- the tests ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("state", STATES)
def test_the_two_sides_price_the_same_paths(state):
    """``npv`` is ``price()`` bitwise; the distribution is REBASED exactly on the un-knocked note; the paths are the
    price's (their mean payoff is the price)."""
    npv, dist, price, env = event_route(state)
    assert npv == price
    assert dist.rebased is (state == "unknocked")
    ref = paths(state)
    assert ref["n"] == cases._MC["num_paths"]
    # this contract settles every outcome at its determination (no lag), so each payment time the direct valuation
    # uses is the contract's own, decided from the paths -- not read from the kernel's settlement arithmetic
    np.testing.assert_array_equal(ref["settle"], ref["det"])
    if state == "unknocked":
        np.testing.assert_array_equal(np.unique(dist.stream_times[EventType.KO]), candidate_axis(state))
    else:
        np.testing.assert_array_equal(dist.stream_times[EventType.KO], candidate_axis(state))


@pytest.mark.parametrize("name", [n for n in LEG_NAMES if n not in NO_COUPON_STREAM])
@pytest.mark.parametrize("state", STATES)
def test_event_route_leg_equals_the_same_path_valuation(state, name):
    _, dist, _, env = event_route(state)
    leg = legs(state)[name]
    live = leg.value(dist, env, cases.UNIT)
    expected = direct_value(state, name)
    assert abs(live - expected) <= ROUND_OFF * _scale(leg), (state, name, live, expected)


@pytest.mark.parametrize("state", STATES)
def test_the_event_route_is_not_trivially_zero(state):
    """Each event-dependent leg carries cash from more than one outcome on these paths (the comparison is not 0 = 0),
    except the AT_KI leg on the knocked-in note (R1)."""
    ref = paths(state)
    outcomes = {k: int(np.count_nonzero(ref["event"] == k)) for k in ("ko_pre", "ko_post", "ki_settlement", "maturity")}
    if state == "unknocked":
        assert all(count > 0 for count in outcomes.values()), outcomes
        assert np.count_nonzero(ref["ki_first"]) > 0
        assert len(set(ref["settle"][ref["event"] == "ki_settlement"].tolist())) > 1   # path-dependent settlement
    else:
        assert outcomes["ko_post"] > 0 and outcomes["ki_settlement"] > 0 and outcomes["ko_pre"] == 0
    for name in LEG_NAMES:
        if name in NO_COUPON_STREAM or name == "deterministic":
            continue
        if state == "knocked_in" and name in ("fixed_at_ki", "fixed_at_maturity_no_ko",
                                               "autocallable_backend_premium_no_ko"):
            continue
        assert direct_value(state, name) != 0.0, (state, name)


@pytest.mark.parametrize("state", STATES)
def test_a_knock_in_leg_follows_the_knock_in_rule(state):
    """Un-knocked: the KI stream is the first knock-in before any knock-out on its own axis (the KI dates), paid at
    the KI date (D8). Knocked in (R1): the statistics keep DESIGN §4.5's carried row (t = 0, P = 1), but the
    distribution holds no knock-in event -- the knock-in is past -- so the AT_KI leg is exactly 0."""
    _, dist, _, env = event_route(state)
    leg = legs(state)["fixed_at_ki"]
    if state == "unknocked":
        ki_dates = np.array(_times(_product(state).barrier_config.ki_observation_schedule))
        np.testing.assert_array_equal(dist.stream_times[EventType.KI], ki_dates)
        ref = paths(state)
        counts = np.array([np.count_nonzero(ref["ki_first"] & (ref["ki_time"] == t)) for t in ki_dates]) / ref["n"]
        assert np.max(np.abs(dist.probabilities[EventType.KI] - counts)) <= 1e-15
        return
    stats = _engine().calculate_event_stats(_product(state), env)
    np.testing.assert_array_equal(stats.ki_table.times, [0.0])
    np.testing.assert_array_equal(stats.ki_table.probability, [1.0])
    assert stats.knock_in.probability == 1.0 and stats.knock_in.expected_time == 0.0
    assert dist.knock_in_carried
    assert dist.stream_times[EventType.KI].size == 0 and dist.probabilities[EventType.KI].size == 0
    assert leg.value(dist, env, cases.UNIT) == 0.0


@pytest.mark.parametrize("state", STATES)
def test_a_coupon_basis_leg_refuses_a_ko_reset(state):
    """A KO-reset has no coupon table (``coupons is None``), so no COUPON stream: the COUPON-basis leg refuses on
    both states, with the reason (a value would be a stand-in)."""
    _, dist, _, env = event_route(state)
    assert EventType.COUPON not in dist.probabilities
    with pytest.raises(ValidationError, match="coupon"):
        legs(state)["autocallable_backend_interest_coupon"].value(dist, env, cases.UNIT)


def test_a_rebased_leg_off_the_candidate_axis_refuses():
    """R2: an AutocallableCashLeg booked on the contract's first-schedule dates only lists no knock-in date + offset,
    so a REBASED knock-out at such a time has no schedule entry: refused, never interpolated (T6 ruling)."""
    _, dist, _, env = event_route("unknocked")
    leg = legs("unknocked")["autocallable_rebate_with_ki"]
    pre = np.array(_times(_product("unknocked").barrier_config.ko_observation_schedule))
    booked = replace(leg, observation_schedule=tuple(pre), accrual_factors=tuple(np.linspace(0.3, 1.0, pre.size)),
                     settlement_schedule=tuple(pre + 0.01))
    with pytest.raises(ValidationError, match="schedule entry"):
        booked.value(dist, env, cases.UNIT)


@pytest.mark.parametrize("name", LEG_NAMES)
def test_the_folded_route_refuses_a_rebased_note(name):
    """DESIGN §4.11: folding a REBASED note stays refused (its settlement is knock-in dependent); a KI or coupon
    leg is refused by the fold binder before that."""
    engine, product, env = _engine(), _product("unknocked"), cases.flat_env()
    leg = legs("unknocked")[name]
    with pytest.raises(CapabilityError, match="REBASED|cannot be folded"):
        folded_trade_value(engine, product, env, [leg], quantity=1.0, unit_notional=cases.UNIT, mode=FoldMode.TOTAL)


#: on the knocked-in note's ABSOLUTE copy the fold binder takes every leg but a knock-in or coupon stream
_FOLDABLE = tuple(n for n in LEG_NAMES if n not in ("fixed_at_ki", *NO_COUPON_STREAM))


@lru_cache(maxsize=None)
def _knocked_in_drilldown():
    engine, product, env = _engine(), _product("knocked_in"), cases.flat_env()
    chosen = [legs("knocked_in")[n] for n in _FOLDABLE]
    return folded_trade_value(engine, product, env, chosen, quantity=1.0, unit_notional=cases.UNIT,
                              mode=FoldMode.DRILLDOWN)


@pytest.mark.parametrize("name", _FOLDABLE)
def test_the_folded_route_on_the_knocked_in_note_equals_the_same_path_valuation(name):
    """Folded into the same Monte Carlo solve (same seed, same paths), each leg's DRILLDOWN value and its TOTAL fold
    less the plain npv equal the direct valuation to round-off."""
    leg = legs("knocked_in")[name]
    expected = direct_value("knocked_in", name)
    folded = _knocked_in_drilldown()
    assert folded.npv == event_route("knocked_in")[2]
    assert abs(folded.per_leg[name] - expected) <= ROUND_OFF * _scale(leg), (name, folded.per_leg[name], expected)
    engine, product, env = _engine(), _product("knocked_in"), cases.flat_env()
    total = folded_trade_value(engine, product, env, [leg], quantity=1.0, unit_notional=cases.UNIT,
                               mode=FoldMode.TOTAL).total
    assert abs((total - folded.npv) - expected) <= ROUND_OFF * _scale(leg), (name, total - folded.npv, expected)


@pytest.mark.parametrize("name", ("fixed_at_ki", *NO_COUPON_STREAM))
def test_the_fold_binder_refuses_knock_in_and_coupon_legs_on_the_knocked_in_note(name):
    engine, product, env = _engine(), _product("knocked_in"), cases.flat_env()
    with pytest.raises(CapabilityError, match="cannot be folded"):
        folded_trade_value(engine, product, env, [legs("knocked_in")[name]], quantity=1.0, unit_notional=cases.UNIT,
                           mode=FoldMode.TOTAL)
