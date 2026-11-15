"""Spec D32 / AC35: opt-in accrual to the payment date (``AccrualLeg.survival_until``).

``SurvivalAnchor.DETERMINATION`` (the default) is spec D29 unchanged: a knock-out leaves survival at its own row, a
maturity-type termination is alive at its determination and gone right after (pinned by
``test_d29_determination_time.py``). ``SurvivalAnchor.PAYMENT``: every termination -- knock-out, never-knocked-in
maturity, knocked-in settlement, each REBASED path's own settlement -- is alive until its own PAYMENT time and gone
after it. The survival is then the step ``1 - sum(mass paid strictly before t)`` read from the distribution's
per-stream probabilities and payment times; nothing is interpolated, so a paid note is never revived. A distribution
without the payment time of a termination stream refuses a PAYMENT read (no fallback).

The oracles here are written from that definition (or from the price's own paths), never from the reader under test.
"""
from __future__ import annotations

import importlib.util
import pickle
import sys
import warnings
from dataclasses import replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from intraday.conftest import dated_snowball
from quantark.asset.equity.engine.quad import SnowballQuadEngineV2
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.cashleg import SurvivalAnchor
from quantark.cashleg.accrual_leg import AccrualLeg, KOBehavior, PaymentConvention, SurvivalBasis
from quantark.cashleg.base import LegDirection
from quantark.cashleg.base_amount import BaseAmount, BaseAmountMode
from quantark.cashleg.booking import LegCashBook, RealizedOutcome
from quantark.cashleg.event_distribution import EventDistribution, EventType
from quantark.cashleg.leg_schedule import LegSchedule
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum.option_enums import PostKOScheduleMode
from quantark.util.exceptions import ValidationError
from test_asset.equity.engine_quad.test_quad_v2_event_statistics import SSE, dated_env

warnings.filterwarnings("ignore", category=DeprecationWarning)

EPS = 1e-9
ZERO_TOL = 1e-10            # Tolerance.ZERO: the readers' time tie
ROUND_OFF = 1e-12
PAYMENT, DETERMINATION = SurvivalAnchor.PAYMENT, SurvivalAnchor.DETERMINATION


# --- the definition, as an oracle ------------------------------------------------------------------------------------

def _terminations(dist):
    """(payment time, mass) of every termination row, read from the distribution's own streams."""
    rows = []
    for event in (EventType.KO, EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI):
        if event not in dist.probabilities:
            continue
        p = np.atleast_1d(np.asarray(dist.probabilities[event], dtype=float))
        pay = np.atleast_1d(np.asarray(dist.payment_times[event], dtype=float))
        rows += list(zip(pay.tolist(), p.tolist()))
    return rows


def paid_survival(dist, t, *, entering=False):
    """P(alive at t) under PAYMENT: a termination is alive at its own payment time and gone after it; ``entering``:
    going into a period that starts at t (a note paid at t does not enter)."""
    if t < 0.0:
        return 1.0
    gone = 0.0
    for pay, mass in _terminations(dist):
        if pay < t - ZERO_TOL or (entering and abs(pay - t) <= ZERO_TOL):
            gone += mass
    return 1.0 - gone


# --- a hand-built distribution ---------------------------------------------------------------------------------------

KO_T = np.array([0.25, 0.5, 0.75])
KO_PAY = np.array([0.27, 0.5, 0.80])               # T+k, same-day, T+k
KO_P = np.array([0.1, 0.2, 0.1])
MAT_T, MAT_PAY = 1.0, 1.05
P_NO_KI, P_KI = 0.4, 0.2


def _hand(**overrides):
    kwargs = dict(
        probabilities={EventType.KO: KO_P, EventType.MATURITY_NO_KO: P_NO_KI, EventType.MATURITY_WITH_KI: P_KI},
        stream_times={EventType.KO: KO_T},
        survival_times=np.r_[0.0, KO_T, MAT_T],
        survival_probability=np.array([1.0, 0.9, 0.7, 0.6, 0.6]),
        payment_times={EventType.KO: KO_PAY, EventType.MATURITY_NO_KO: MAT_PAY, EventType.MATURITY_WITH_KI: MAT_PAY},
        determination_times={EventType.MATURITY_NO_KO: MAT_T, EventType.MATURITY_WITH_KI: MAT_T},
        terminal_times=np.array([MAT_T, MAT_T]),
        terminal_probability=np.array([P_NO_KI, P_KI]),
    )
    kwargs.update(overrides)
    return EventDistribution(**kwargs)


PROBES = sorted({*np.linspace(-0.1, 1.2, 131).tolist(), *KO_T.tolist(), *KO_PAY.tolist(), MAT_T, MAT_PAY,
                 *(KO_PAY + EPS).tolist(), MAT_PAY + EPS, MAT_T + EPS, 0.0})


def test_payment_survival_is_the_paid_step_at_every_time():
    dist = _hand()
    for t in PROBES:
        assert dist.survival_at(t, anchor=PAYMENT) == pytest.approx(paid_survival(dist, t), abs=ROUND_OFF), t
        assert dist.survival_after(t, anchor=PAYMENT) == pytest.approx(
            paid_survival(dist, t, entering=True), abs=ROUND_OFF), t


def test_a_knock_out_is_alive_until_its_payment_and_gone_after():
    dist = _hand()
    # the T+k knock-out at 0.25 paid 0.27: gone at its row under DETERMINATION, alive until 0.27 under PAYMENT
    assert dist.survival_at(0.25) == pytest.approx(0.9, abs=ROUND_OFF)
    assert dist.survival_at(0.25, anchor=PAYMENT) == pytest.approx(1.0, abs=ROUND_OFF)
    assert dist.survival_at(0.27, anchor=PAYMENT) == pytest.approx(1.0, abs=ROUND_OFF)
    assert dist.survival_at(0.27 + EPS, anchor=PAYMENT) == pytest.approx(0.9, abs=ROUND_OFF)
    assert dist.survival_after(0.27, anchor=PAYMENT) == pytest.approx(0.9, abs=ROUND_OFF)
    # a knock-out paid on its own date is alive at that instant under PAYMENT (it is not yet paid before it)
    assert dist.survival_at(0.5, anchor=PAYMENT) == pytest.approx(0.9, abs=ROUND_OFF)
    assert dist.survival_after(0.5, anchor=PAYMENT) == pytest.approx(0.7, abs=ROUND_OFF)


def test_the_matured_mass_stays_until_its_payment_and_drops_after():
    dist = _hand()
    matured = P_NO_KI + P_KI
    # DETERMINATION (D29): alive at 1.0, gone right after although unpaid
    assert dist.survival_at(MAT_T) == pytest.approx(matured, abs=ROUND_OFF)
    assert dist.survival_at(MAT_T + EPS) == pytest.approx(0.0, abs=ROUND_OFF)
    # PAYMENT: the matured mass is alive through its payment and gone after it
    for t in (MAT_T, MAT_T + EPS, 0.5 * (MAT_T + MAT_PAY), MAT_PAY):
        assert dist.survival_at(t, anchor=PAYMENT) == pytest.approx(matured, abs=ROUND_OFF), t
    assert dist.survival_after(MAT_PAY, anchor=PAYMENT) == pytest.approx(0.0, abs=ROUND_OFF)
    for t in (MAT_PAY + EPS, 1.2, 50.0):
        assert dist.survival_at(t, anchor=PAYMENT) == pytest.approx(0.0, abs=ROUND_OFF), t


def test_payment_survival_never_revives_a_paid_note():
    dist = _hand()
    grid = np.linspace(-0.05, 1.3, 2701)
    values = np.array([dist.survival_at(float(t), anchor=PAYMENT) for t in grid])
    assert np.all(np.diff(values) <= ROUND_OFF)
    after = np.array([dist.survival_after(float(t), anchor=PAYMENT) for t in grid])
    assert np.all(after <= values + ROUND_OFF)


def test_the_default_anchor_is_determination_and_reads_as_before():
    dist = _hand()
    for t in PROBES:
        assert dist.survival_at(t) == dist.survival_at(t, anchor=DETERMINATION)
        assert dist.survival_after(t) == dist.survival_after(t, anchor=DETERMINATION)


def test_an_unknown_anchor_is_refused():
    with pytest.raises(ValidationError, match="SurvivalAnchor"):
        _hand().survival_at(0.5, anchor="payment")


@pytest.mark.parametrize("missing", [EventType.KO, EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI])
def test_a_payment_read_without_payment_times_raises(missing):
    pays = {e: v for e, v in _hand().payment_times.items() if e is not missing}
    dist = _hand(payment_times=pays)
    dist.survival_at(0.5)                          # DETERMINATION does not need them
    with pytest.raises(ValidationError, match=missing.value):
        dist.survival_at(0.5, anchor=PAYMENT)
    with pytest.raises(ValidationError, match=missing.value):
        dist.survival_after(0.5, anchor=PAYMENT)


def test_a_path_dependent_knock_out_payment_raises():
    pays = {e: v for e, v in _hand().payment_times.items() if e is not EventType.KO}
    dist = _hand(payment_times=pays, path_dependent_payment_events=frozenset({EventType.KO}))
    with pytest.raises(ValidationError, match="path-dependent"):
        dist.survival_at(0.5, anchor=PAYMENT)


def test_the_trivial_distribution_reads_under_payment():
    dist = EventDistribution.trivial(1.0)
    assert dist.survival_at(1.0, anchor=PAYMENT) == 1.0
    assert dist.survival_after(1.0, anchor=PAYMENT) == pytest.approx(0.0, abs=ROUND_OFF)


def test_a_terminated_note_refuses_a_payment_read():
    with pytest.raises(ValidationError, match="terminated"):
        EventDistribution.ended().survival_at(0.5, anchor=PAYMENT)


# --- the leg ---------------------------------------------------------------------------------------------------------

def _leg(ends, basis=SurvivalBasis.COMPLETE_PERIOD, anchor=None, *, starts=None, pays=None, **extra):
    ends = np.asarray(ends, dtype=float)
    starts = np.r_[0.0, ends[:-1]] if starts is None else np.asarray(starts, dtype=float)
    kwargs = {} if anchor is None else {"survival_until": anchor}
    return AccrualLeg(
        rate=0.04, base=BaseAmount(value=1_000_000.0, mode=BaseAmountMode.ABSOLUTE),
        schedule=LegSchedule(period_starts=starts, period_ends=ends,
                             payment_times=ends.copy() if pays is None else np.asarray(pays, dtype=float)),
        day_count=DayCountConvention.ACT_365, payment_convention=PaymentConvention.AT_PERIOD_END,
        ko_behavior=extra.pop("ko_behavior", KOBehavior.TRUNCATE_AT_KO), survival_basis=basis,
        direction=LegDirection.BUYER_RECEIVES, **kwargs, **extra)


def test_the_field_defaults_to_determination():
    assert _leg([0.5, 1.0]).survival_until is SurvivalAnchor.DETERMINATION


def test_the_field_must_be_a_survival_anchor():
    with pytest.raises(ValidationError, match="SurvivalAnchor"):
        _leg([0.5, 1.0], anchor="payment")


def test_a_full_schedule_leg_refuses_a_payment_anchor_it_would_never_read():
    with pytest.raises(ValidationError, match="PAY_FULL_SCHEDULE"):
        _leg([0.5, 1.0], anchor=PAYMENT, ko_behavior=KOBehavior.PAY_FULL_SCHEDULE)


def test_a_payment_leg_reads_the_maturity_streams_too():
    assert _leg([0.5, 1.0]).required_event_types() == frozenset({EventType.KO})
    assert _leg([0.5, 1.0], anchor=PAYMENT).required_event_types() == frozenset(
        {EventType.KO, EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI})


def _df(env, t):
    return float(env.get_discount_factor(float(t)))


class _Env:
    def get_discount_factor(self, t):
        return float(np.exp(-0.03 * t))


@pytest.mark.parametrize("basis", list(SurvivalBasis))
def test_the_payment_leg_values_on_the_paid_survival(basis):
    dist, env = _hand(), _Env()
    ends = np.array([0.25, 0.27, 0.5, 0.75, 0.8, 1.0, 1.05, 1.1])
    leg = _leg(ends, basis, PAYMENT)
    starts = leg.schedule.period_starts
    alive = ([paid_survival(dist, float(s), entering=True) for s in starts]
             if basis is SurvivalBasis.ENTER_PERIOD else [paid_survival(dist, float(e)) for e in ends])
    expected = 0.04 * 1_000_000.0 * float(np.sum((ends - starts) * np.array(alive)
                                                 * np.array([_df(env, t) for t in ends])))
    assert leg.value(dist, env, 0.0) == pytest.approx(expected, rel=ROUND_OFF)
    assert leg.value(dist, env, 0.0) != pytest.approx(_leg(ends, basis).value(dist, env, 0.0), rel=1e-6)


def test_the_payment_leg_raises_without_payment_times():
    pays = {e: v for e, v in _hand().payment_times.items() if e is not EventType.MATURITY_NO_KO}
    with pytest.raises(ValidationError, match="maturity_no_ko"):
        _leg([0.5, 1.0], anchor=PAYMENT).value(_hand(payment_times=pays), _Env(), 0.0)


# --- serialization and booking ---------------------------------------------------------------------------------------

def test_the_field_round_trips_through_pickle_replace_and_time_shift():
    leg = _leg([0.25, 0.5, 0.75, 1.0], anchor=PAYMENT)
    # pickle of an object this test just built (no untrusted input): the field survives process boundaries
    back = pickle.loads(pickle.dumps(leg))
    assert back.survival_until is PAYMENT
    dist, env = _hand(), _Env()
    assert back.value(dist, env, 0.0) == leg.value(dist, env, 0.0)
    assert replace(leg, rate=0.05).survival_until is PAYMENT
    assert leg.time_shift(0.3).survival_until is PAYMENT
    assert _leg([0.25, 0.5]).time_shift(0.1).survival_until is DETERMINATION


def test_the_booked_leg_keeps_the_field():
    leg = _leg([0.25, 0.5, 0.75, 1.0], anchor=PAYMENT)
    book = LegCashBook([leg], 1.0, datetime(2026, 3, 16), snap=lambda t: round(t * 365.0) / 365.0)
    assert book.legs[0].survival_until is PAYMENT
    assert book.legs_at_run_start()[0].survival_until is PAYMENT


D = 1.0 / 365.0
KO_DAY, KO_SETTLE_DAY = 182, 186


@pytest.mark.parametrize("basis, anchor, paid_periods", [
    # periods [0,91] [91,182] [182,184] [184,186] [186,273]; knock-out observed day 182, paid day 186
    (SurvivalBasis.ENTER_PERIOD, DETERMINATION, [0, 1]),          # D29: [182, ...) starts on the KO date
    (SurvivalBasis.COMPLETE_PERIOD, DETERMINATION, [0]),          # D29: [91, 182] ends on the KO date
    (SurvivalBasis.ENTER_PERIOD, PAYMENT, [0, 1, 2, 3]),          # alive just after 182 and 184; paid at 186
    (SurvivalBasis.COMPLETE_PERIOD, PAYMENT, [0, 1, 2, 3]),       # periods ending on or before the payment
])
def test_booking_truncates_a_knock_out_at_the_anchor(basis, anchor, paid_periods):
    days = np.array([0, 91, 182, 184, 186, 273])
    leg = _leg(days[1:] * D, basis, anchor, starts=days[:-1] * D)
    book = LegCashBook([leg], 1.0, datetime(2026, 3, 16))
    outcome = RealizedOutcome.ko(KO_DAY * D, KO_SETTLE_DAY * D, maturity_settlement_time=367 * D)
    assert sorted(i for (_, i), _a, _t in book.termination_flows(outcome)) == paid_periods


# --- a dated note settling after its determination (V2), knock-outs paid T+3, maturity T+5 ---------------------------

def _lagged_product():
    product = dated_snowball(SSE, datetime(2026, 3, 16))
    records = product.barrier_config.ko_observation_schedule.records
    product.barrier_config = replace(product.barrier_config, ko_observation_schedule=ObservationSchedule(records=[
        ObservationRecord(observation_date=r.observation_date, barrier=r.barrier,
                          settlement_date=SSE.trading_days_after(r.observation_date, 3)) for r in records]))
    product.settlement_date = SSE.trading_days_after(product.exercise_date, 5)
    product.validate()
    return product


@lru_cache(maxsize=None)
def _lagged():
    market = dated_env()
    stats = SnowballQuadEngineV2().calculate_event_stats(_lagged_product(), market)
    return stats, EventDistribution.from_statistics(stats), market


def test_the_lagged_note_really_settles_after_its_determinations():
    stats, dist, _ = _lagged()
    obs = stats.observations
    assert np.all(obs.payment_times > obs.times + 2.0 / 365.0)
    assert stats.maturity.payment_time > stats.maturity.time + 5.0 / 365.0 - ROUND_OFF
    assert stats.maturity.probability + stats.ki_settlement.probability > 0.2
    np.testing.assert_array_equal(dist.payment_times[EventType.KO], obs.payment_times)


def test_payment_survival_is_determination_survival_plus_the_unpaid_knock_outs_up_to_determination():
    """Up to the maturity's determination the matured mass reads exactly as under DETERMINATION; the only difference
    is the knock-outs already determined and not yet paid."""
    stats, dist, _ = _lagged()
    obs = stats.observations
    for t in dist.survival_times[dist.survival_times <= stats.maturity.time + ZERO_TOL]:
        pending = float(np.sum(obs.probability[(obs.times <= t + ZERO_TOL) & (obs.payment_times >= t - ZERO_TOL)]))
        assert dist.survival_at(float(t), anchor=PAYMENT) == pytest.approx(
            dist.survival_at(float(t)) + pending, abs=ROUND_OFF), t


def test_the_matured_mass_is_alive_until_its_payment_and_gone_after():
    stats, dist, _ = _lagged()
    obs = stats.observations
    determination, payment = stats.maturity.time, stats.maturity.payment_time
    matured = stats.maturity.probability + stats.ki_settlement.probability + stats.probability_residual
    for t in (determination, determination + EPS, 0.5 * (determination + payment), payment):
        unpaid_ko = float(np.sum(obs.probability[obs.payment_times >= t - ZERO_TOL]))
        assert dist.survival_at(t, anchor=PAYMENT) == pytest.approx(matured + unpaid_ko, abs=ROUND_OFF), t
    assert dist.survival_at(payment, anchor=PAYMENT) == pytest.approx(matured, abs=ROUND_OFF)
    assert dist.survival_after(payment, anchor=PAYMENT) == pytest.approx(0.0, abs=ROUND_OFF)
    assert dist.survival_at(payment + EPS, anchor=PAYMENT) == pytest.approx(0.0, abs=ROUND_OFF)


def test_a_complete_period_leg_ending_on_the_settlement_date_pays_on_the_matured_mass():
    stats, dist, market = _lagged()
    payment = stats.maturity.payment_time
    matured = stats.maturity.probability + stats.ki_settlement.probability + stats.probability_residual
    first = _leg([0.5], anchor=PAYMENT).value(dist, market, 0.0)
    to_settlement = _leg([0.5, payment], anchor=PAYMENT).value(dist, market, 0.0)
    expected = 0.04 * 1_000_000.0 * (payment - 0.5) * matured * _df(market, payment)
    assert to_settlement - first == pytest.approx(expected, rel=1e-10)
    # DETERMINATION (D29, CP8 I1) pays nothing on that period
    determination_leg = _leg([0.5, payment]).value(dist, market, 0.0)
    assert determination_leg == pytest.approx(_leg([0.5]).value(dist, market, 0.0), rel=ROUND_OFF, abs=1e-9)
    assert to_settlement > determination_leg + 1_000.0


def test_a_knock_out_paid_t_plus_k_is_alive_until_its_payment():
    stats, dist, market = _lagged()
    obs = stats.observations
    j = int(np.argmax(obs.probability[:-1]))
    det, pay, p = float(obs.times[j]), float(obs.payment_times[j]), float(obs.probability[j])
    assert p > 1e-3 and pay > det + ROUND_OFF
    assert dist.survival_at(pay, anchor=PAYMENT) - dist.survival_at(pay + EPS, anchor=PAYMENT) == pytest.approx(
        p, abs=ROUND_OFF)
    assert dist.survival_at(det, anchor=PAYMENT) - dist.survival_at(det) == pytest.approx(p, abs=ROUND_OFF)
    # a period ending on the knock-out's payment date completes on it under PAYMENT only
    leg_pay = _leg([det, pay], anchor=PAYMENT).value(dist, market, 0.0)
    leg_det = _leg([det, pay]).value(dist, market, 0.0)
    # both periods move: [0, det] ends on the knock-out's determination (alive under PAYMENT, it is unpaid), and
    # [det, pay] ends on its payment
    expected_gap = 0.04 * 1_000_000.0 * sum(
        (end - start) * _df(market, end) * (paid_survival(dist, end) - dist.survival_at(end))
        for start, end in ((0.0, det), (det, pay)))
    assert leg_pay - leg_det == pytest.approx(expected_gap, rel=1e-10)
    assert paid_survival(dist, pay) - dist.survival_at(pay) >= p - ROUND_OFF


# --- REBASED: each path's own settlement (MC, the price's own paths) -------------------------------------------------

BANK_DIR = Path(__file__).resolve().parent / "event_stats_baseline"


def _bank_cases():
    name = "_cashleg_event_stats_baseline_cases"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, BANK_DIR / "cases.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@lru_cache(maxsize=None)
def _rebased():
    from test_asset.equity.engine_mc import mc_same_path_reference

    cases = _bank_cases()
    env = cases.flat_env()
    product = cases._ko_reset(PostKOScheduleMode.REBASED)
    dist = cases.ENGINES["mc_snowball"].build().price_with_events(product, env).event_distribution
    paths = mc_same_path_reference.ko_reset_paths(cases.ENGINES["mc_snowball"].build(),
                                                  cases._ko_reset(PostKOScheduleMode.REBASED), env)
    return dist, paths, env


def _path_alive(paths, t, *, entering=False):
    """PAYMENT on the paths: a path is alive until its own settlement (payment) and gone after it."""
    if t < 0.0:
        return 1.0
    settle = paths["settle"]
    alive = settle > t + ZERO_TOL if entering else settle >= t - ZERO_TOL
    return float(np.count_nonzero(alive)) / paths["n"]


def test_rebased_per_path_settlements_are_honoured():
    dist, paths, env = _rebased()
    assert dist.rebased
    knocked = paths["event"] == "ki_settlement"
    assert np.unique(paths["settle"][knocked]).size > 1          # the knocked-in settlement varies by path
    for t in sorted({*np.unique(paths["settle"]).tolist(), *np.linspace(0.0, 2.2, 45).tolist()}):
        assert dist.survival_at(t, anchor=PAYMENT) == pytest.approx(_path_alive(paths, t), abs=ROUND_OFF), t
        assert dist.survival_after(t, anchor=PAYMENT) == pytest.approx(
            _path_alive(paths, t, entering=True), abs=ROUND_OFF), t


@pytest.mark.parametrize("basis", list(SurvivalBasis))
def test_rebased_payment_legs_equal_the_path_by_path_value(basis):
    dist, paths, env = _rebased()
    ends = np.arange(1, 10) * 0.25
    leg = _leg(ends, basis, PAYMENT)
    starts = leg.schedule.period_starts
    alive = ([_path_alive(paths, float(s), entering=True) for s in starts] if basis is SurvivalBasis.ENTER_PERIOD
             else [_path_alive(paths, float(e)) for e in ends])
    direct = 0.04 * 1_000_000.0 * float(np.sum((ends - starts) * np.array(alive)
                                               * np.array([_df(env, t) for t in ends])))
    assert leg.value(dist, env, 0.0) == pytest.approx(direct, rel=ROUND_OFF)
    if basis is SurvivalBasis.COMPLETE_PERIOD:
        # this contract pays every termination on its determination date: only a period ending on a knock-out
        # or settlement date tells the anchors apart (alive at its payment instant under PAYMENT)
        assert leg.value(dist, env, 0.0) > _leg(ends, basis).value(dist, env, 0.0) + 100.0
