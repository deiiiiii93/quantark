"""``EventDistribution.from_statistics`` on hand-built ``EventStatistics`` (unified event statistics T6).

Per-stream axes (DESIGN §4.11): KO and COUPON share the observation grid, KI has the KI table's own
axis (never dropped for a length differing from the KO grid), scalar outcomes keep their determination
times. D16: the probability residual is absorbed into MATURITY_NO_KO and carried visibly. D13: a
REBASED KO-reset maps its absolute termination distribution to array streams, MATURITY_WITH_KI
included, and legs value each event at its own time. Receivables are ledger rows, not events.
No engine runs here.
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.event_stats import (
    CashLedger,
    CouponTable,
    EventStatistics,
    KnockInStatistics,
    KnockInTable,
    ObservationTable,
    OutcomeRow,
    PostKIOffsetTable,
    ReceivablesTable,
    StatisticsProvenance,
    TerminationDistribution,
    build_cash_ledger,
    build_rebased_cash_ledger,
)
from quantark.cashleg import (
    AccrualLeg,
    AutocallableCashLeg,
    AutocallableLegType,
    BaseAmount,
    BaseAmountMode,
    EventDistribution,
    EventType,
    FixedPayoffLeg,
    KOBehavior,
    LegDirection,
    LegSchedule,
    PaymentTrigger,
    PvFormula,
)
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import NumericalError, ValidationError

PROV = StatisticsProvenance(statistics_model="hand_built", price_model="hand_built")


def _env(rate: float = 0.03) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=rate), div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2024, 1, 1),
    )


def _df(t: float) -> float:
    return math.exp(-0.03 * t)


# --- hand-built statistics ---------------------------------------------------------------------------


def _observations(times, probability, *, survival=None, is_ko=None, is_coupon=False, schedule=None,
                  pay_lag=0.01) -> ObservationTable:
    times = np.asarray(times, dtype=float)
    probability = np.asarray(probability, dtype=float)
    n = times.size
    if survival is None:
        survival = 1.0 - np.cumsum(probability)
    return ObservationTable(
        times=times, payment_times=times + pay_lag,
        is_ko_obs=np.ones(n, dtype=bool) if is_ko is None else np.asarray(is_ko, dtype=bool),
        is_coupon_obs=np.full(n, bool(is_coupon)), schedule=schedule,
        probability=probability, survival=np.asarray(survival, dtype=float),
        discounted_cash=probability * 100.0 * 0.99, cash_to_pay=probability * 100.0,
        payment_is_path_dependent=np.zeros(n, dtype=bool),
    )


def _row(time, probability, *, pay_lag=0.02, cash=100.0) -> OutcomeRow:
    return OutcomeRow(time=time, payment_time=time + pay_lag, probability=probability,
                      discounted_cash=probability * cash * 0.97, cash_to_pay=probability * cash)


def _ki_table(times, probability) -> KnockInTable:
    times = np.asarray(times, dtype=float)
    probability = np.asarray(probability, dtype=float)
    return KnockInTable(times=times, payment_times=times, probability=probability,
                        survival=1.0 - np.cumsum(probability), interpolated=np.zeros(times.size, dtype=bool))


def _stats(*, family="snowball", observations, coupons=None, maturity, ki_settlement, ki_table,
           receivables=None, unavailable=None, path_dependent_cash=None, knock_in_probability=None) -> EventStatistics:
    receivables = ReceivablesTable.empty() if receivables is None else receivables
    ledger = build_cash_ledger(observations=observations, coupons=coupons, maturity=maturity,
                               ki_settlement=ki_settlement, receivables=receivables,
                               path_dependent_cash=path_dependent_cash)
    p_ki = ki_settlement.probability if knock_in_probability is None else knock_in_probability
    unavailable = {} if unavailable is None else dict(unavailable)
    if p_ki == 0.0:
        unavailable["knock_in.expected_time"] = "undefined: P(knock in) = 0"
    return EventStatistics(
        product_family=family, pv=100.0, observations=observations, coupons=coupons, maturity=maturity,
        ki_settlement=ki_settlement,
        knock_in=KnockInStatistics(probability=p_ki, probability_no_ko=ki_settlement.probability,
                                   expected_time=0.4 if p_ki > 0.0 else None, expected_loss=1.0),
        ki_table=ki_table, receivables=receivables, ledger=ledger, expected_lifetime=0.6, provenance=PROV,
        unavailable=unavailable,
    )


KO_TIMES = (0.25, 0.5, 0.75, 1.0)
KO_P = (0.30, 0.10, 0.05, 0.05)
#: a continuous-KI table with its own (daily-like) axis, longer than the KO grid
KI_TIMES = (0.1, 0.2, 0.3, 0.45, 0.6, 0.8)
KI_P = (0.01, 0.02, 0.03, 0.02, 0.01, 0.01)


def _snowball(*, maturity_p=0.42, ki_p=0.08, ki_table="table", receivables=None) -> EventStatistics:
    table = _ki_table(KI_TIMES, KI_P) if ki_table == "table" else ki_table
    unavailable = {"ki_table": "not requested"} if table is None else None
    p_ki = sum(KI_P) if table is None else float(np.sum(table.probability))
    return _stats(observations=_observations(KO_TIMES, KO_P), maturity=_row(1.0, maturity_p),
                  ki_settlement=_row(1.0, ki_p), ki_table=table, receivables=receivables,
                  unavailable=unavailable, knock_in_probability=p_ki)


# --- the KI stream on its own axis -------------------------------------------------------------------


def test_ki_stream_keeps_its_own_axis_when_its_length_differs_from_the_ko_grid():
    """The base bug: the legacy autocallable converter dropped the KI stream unless it had the KO grid's length."""
    stats = _snowball()
    dist = EventDistribution.from_statistics(stats)
    np.testing.assert_array_equal(dist.probabilities[EventType.KI], np.asarray(KI_P))
    np.testing.assert_array_equal(dist.stream_times[EventType.KI], np.asarray(KI_TIMES))
    np.testing.assert_array_equal(dist.stream_times[EventType.KO], np.asarray(KO_TIMES))
    # D8: a KI row is paid at its observation time; the leg adds its own settlement offset
    np.testing.assert_array_equal(dist.payment_times_for(EventType.KI), np.asarray(KI_TIMES))


def test_at_ki_leg_pays_on_the_ki_axis_plus_its_own_offset():
    dist = EventDistribution.from_statistics(_snowball())
    leg = FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=6_000.0, trigger=PaymentTrigger.AT_KI,
                         payment_offset_days=2)
    expected = 6_000.0 * sum(p * _df(t + 2.0 / 365.0) for t, p in zip(KI_TIMES, KI_P))
    assert leg.value(dist, _env(), 0.0) == pytest.approx(expected, rel=1e-13)


def test_an_empty_ki_table_is_an_empty_ki_stream_and_an_unavailable_one_is_no_stream():
    empty = EventDistribution.from_statistics(_snowball(ki_p=0.0, maturity_p=0.5, ki_table=KnockInTable.empty()))
    assert empty.probabilities[EventType.KI].size == 0
    leg = FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=6_000.0, trigger=PaymentTrigger.AT_KI)
    assert leg.value(empty, _env(), 0.0) == 0.0          # computed: no knock-in can happen

    pruned = EventDistribution.from_statistics(_snowball(ki_table=None))
    assert EventType.KI not in pruned.probabilities
    with pytest.raises(ValidationError, match="knock_in"):
        leg.value(pruned, _env(), 0.0)


def _knocked_in(*, ki_table="carried") -> EventStatistics:
    """A note already knocked in: DESIGN §4.5's single carried row (t = 0, P = 1) in the statistics."""
    table = _ki_table([0.0], [1.0]) if ki_table == "carried" else ki_table
    unavailable = {"ki_table": "not requested"} if table is None else None
    return _stats(observations=_observations(KO_TIMES, KO_P), maturity=_row(1.0, 0.0),
                  ki_settlement=_row(1.0, 0.5), ki_table=table, unavailable=unavailable, knock_in_probability=1.0)


def test_a_carried_knock_in_is_past_so_the_ki_stream_is_empty():
    """T23 ruling R1 (DESIGN §4.9c): a knock-in in force at the valuation happened before it; cash it decided is
    paid or a receivable, never an event of the remaining life. The statistics keep §4.5's carried row, the
    distribution holds no knock-in event, and the AT_KI leg takes nothing from it."""
    stats = _knocked_in()
    dist = EventDistribution.from_statistics(stats, knock_in_carried=True)
    assert dist.knock_in_carried
    assert dist.stream_times[EventType.KI].size == 0 and dist.probabilities[EventType.KI].size == 0
    leg = FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=6_000.0, trigger=PaymentTrigger.AT_KI,
                         payment_offset_days=2)
    assert leg.value(dist, _env(), 0.0) == 0.0
    # every other stream is the uncarried conversion's
    plain = EventDistribution.from_statistics(stats)
    assert not plain.knock_in_carried
    np.testing.assert_array_equal(plain.probabilities[EventType.KI], [1.0])
    for event_type in (EventType.KO, EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI):
        np.testing.assert_array_equal(dist.probabilities[event_type], plain.probabilities[event_type])
    np.testing.assert_array_equal(dist.survival_probability, plain.survival_probability)


def test_a_carried_knock_in_needs_the_carried_row_and_a_pruned_table_stays_no_stream():
    """A carried knock-in whose KI table is anything but the single row (0, 1) is inconsistent: refused, never
    re-read. A pruned KI table is no KI stream, carried or not."""
    with pytest.raises(ValidationError, match="carried knock-in"):
        EventDistribution.from_statistics(_snowball(), knock_in_carried=True)
    pruned = EventDistribution.from_statistics(_knocked_in(ki_table=None), knock_in_carried=True)
    assert EventType.KI not in pruned.probabilities and pruned.knock_in_carried


def test_ko_stream_and_scalar_outcomes():
    stats = _snowball()
    dist = EventDistribution.from_statistics(stats)
    np.testing.assert_array_equal(dist.probabilities[EventType.KO], np.asarray(KO_P))
    np.testing.assert_array_equal(dist.payment_times_for(EventType.KO), np.asarray(KO_TIMES) + 0.01)
    assert dist.probabilities[EventType.MATURITY_WITH_KI] == stats.ki_settlement.probability
    assert dist.determination_times[EventType.MATURITY_NO_KO] == 1.0
    assert dist.determination_times[EventType.MATURITY_WITH_KI] == 1.0
    assert dist.payment_times_for(EventType.MATURITY_NO_KO) == pytest.approx(1.02)
    assert dist.provenance == PROV
    # survival on the observation axis, 1 before the first observation
    assert dist.survival_at(0.1) == pytest.approx(1.0 - 0.30 * 0.1 / 0.25)
    assert dist.survival_at(0.5) == pytest.approx(0.60)


# --- D16: the residual --------------------------------------------------------------------------------


@pytest.mark.parametrize("residual", [2.0e-4, -3.0e-3, 0.0])
def test_residual_is_absorbed_into_maturity_no_ko_and_carried(residual):
    """Never raised on, even far beyond ``Tolerance.PROBABILITY``; visible on the distribution."""
    stats = _snowball(maturity_p=0.42 - residual)
    assert stats.probability_residual == pytest.approx(residual, abs=1e-15)
    dist = EventDistribution.from_statistics(stats)
    assert dist.probability_residual == stats.probability_residual
    assert dist.probabilities[EventType.MATURITY_NO_KO] == pytest.approx(0.42, abs=1e-15)
    total = sum(float(np.sum(dist.probabilities[e]))
                for e in (EventType.KO, EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI))
    assert total == pytest.approx(1.0, abs=1e-15)


# --- receivables are ledger rows, not events ---------------------------------------------------------


def test_receivables_are_not_events():
    receivables = ReceivablesTable(source=("coupon", "ko"), decision_times=[-0.02, 0.0], payment_times=[0.03, 0.01],
                                   discounted_cash=[1.9, 104.0], cash_to_pay=[2.0, 105.0])
    with_receivables = EventDistribution.from_statistics(_snowball(receivables=receivables))
    without = EventDistribution.from_statistics(_snowball())
    assert set(with_receivables.probabilities) == set(without.probabilities)
    for event in without.probabilities:
        np.testing.assert_array_equal(np.asarray(with_receivables.probabilities[event]),
                                      np.asarray(without.probabilities[event]))
    np.testing.assert_array_equal(with_receivables.survival_probability, without.survival_probability)
    np.testing.assert_array_equal(with_receivables.stream_times[EventType.KO], without.stream_times[EventType.KO])


# --- KO and COUPON share the observation grid ---------------------------------------------------------


def _phoenix(*, path_dependent=False, coupon_probability="computed") -> EventStatistics:
    obs = _observations(KO_TIMES, KO_P, is_coupon=True)
    n = obs.size
    pay = obs.times + (1.02 - obs.times if path_dependent else 0.015)
    cash = np.array([0.5, 0.4, 0.3, 0.2])
    probability = np.array([0.9, 0.6, 0.5, 0.4]) if coupon_probability == "computed" else None
    coupons = CouponTable(probability=probability, survival=np.r_[1.0, obs.survival[:-1]], payment_times=pay,
                          discounted_cash=cash, cash_to_pay=cash * 1.01,
                          payment_is_path_dependent=np.full(n, path_dependent))
    path_cash = None
    if path_dependent:
        path_cash = CashLedger(determination_times=[1.0], payment_times=[1.02], discounted_cash=[cash.sum()],
                               cash_to_pay=[cash.sum() * 1.01])
    unavailable = {} if probability is not None else {"coupons.probability": "not requested"}
    return _stats(family="phoenix", observations=obs, coupons=coupons, maturity=_row(1.0, 0.42),
                  ki_settlement=_row(1.0, 0.08), ki_table=_ki_table(KI_TIMES, KI_P), unavailable=unavailable,
                  path_dependent_cash=path_cash, knock_in_probability=sum(KI_P))


def test_coupon_stream_shares_the_observation_grid():
    dist = EventDistribution.from_statistics(_phoenix())
    np.testing.assert_array_equal(dist.stream_times[EventType.COUPON], dist.stream_times[EventType.KO])
    np.testing.assert_array_equal(dist.probabilities[EventType.COUPON], [0.9, 0.6, 0.5, 0.4])
    np.testing.assert_array_equal(dist.payment_times_for(EventType.COUPON), np.asarray(KO_TIMES) + 0.015)


def test_path_dependent_coupon_payment_is_refused_and_a_pruned_coupon_is_no_stream():
    dist = EventDistribution.from_statistics(_phoenix(path_dependent=True))
    assert EventType.COUPON in dist.path_dependent_payment_events
    with pytest.raises(NotImplementedError, match="path-dependent"):
        dist.payment_times_for(EventType.COUPON)
    assert EventType.COUPON not in EventDistribution.from_statistics(_phoenix(coupon_probability=None)).probabilities


def test_coupon_only_rows_stay_on_the_ko_axis_with_zero_probability():
    """DCN lock-up rows: coupon observations that are not KO observations keep the padded KO axis."""
    obs = _observations(KO_TIMES, (0.0, 0.0, 0.2, 0.1), is_ko=(False, False, True, True), is_coupon=True)
    n = obs.size
    coupons = CouponTable(probability=np.full(n, 0.5), survival=np.r_[1.0, obs.survival[:-1]],
                          payment_times=obs.times, discounted_cash=np.full(n, 0.1), cash_to_pay=np.full(n, 0.1),
                          payment_is_path_dependent=np.zeros(n, dtype=bool))
    stats = _stats(family="dcn", observations=obs, coupons=coupons, maturity=_row(1.0, 0.5),
                   ki_settlement=_row(1.0, 0.2), ki_table=None, unavailable={"ki_table": "not requested"},
                   knock_in_probability=0.25)
    dist = EventDistribution.from_statistics(stats)
    np.testing.assert_array_equal(dist.stream_times[EventType.KO], np.asarray(KO_TIMES))
    np.testing.assert_array_equal(dist.probabilities[EventType.KO], [0.0, 0.0, 0.2, 0.1])


def test_knock_out_at_valuation_keeps_the_observation_at_zero():
    obs = _observations((0.0, 0.5, 1.0), (1.0, 0.0, 0.0), pay_lag=0.01)
    stats = _stats(observations=obs, maturity=_row(1.0, 0.0), ki_settlement=_row(1.0, 0.0),
                   ki_table=KnockInTable.empty(), knock_in_probability=0.0)
    dist = EventDistribution.from_statistics(stats)
    assert dist.survival_times[0] == 0.0
    assert dist.survival_at(0.0) == 0.0          # post-observation survival of the t = 0 knock-out
    assert dist.survival_at(-0.25) == 1.0


# --- KO-reset ABSOLUTE: both schedules on the KO axis ------------------------------------------------


def test_ko_reset_absolute_keeps_both_schedules_and_its_survival():
    times = (0.5, 0.5, 1.0, 1.0, 1.5, 2.0)
    schedule = ("pre", "post", "pre", "post", "post", "post")
    p = np.array([0.3, 0.0, 0.1, 0.02, 0.03, 0.02])
    survival = 1.0 - np.cumsum(p)
    survival[2:] -= 0.33            # F5: the tables drop the pre-maturity mass at its determination (t = 1.0)
    obs = _observations(times, p, survival=survival, schedule=schedule)
    stats = _stats(family="ko_reset", observations=obs, maturity=_row(1.0, 0.33), ki_settlement=_row(2.0, 0.20),
                   ki_table=None, unavailable={"ki_table": "not requested"})
    dist = EventDistribution.from_statistics(stats)
    np.testing.assert_array_equal(dist.stream_times[EventType.KO], np.asarray(times))
    # spec D29: one node per time (the value after its last row); the matured notes are alive at 1.0 and gone after
    np.testing.assert_array_equal(dist.survival_times, [0.0, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(dist.survival_probability, [1.0, 0.7, 0.58, 0.22, 0.20], rtol=0, atol=1e-15)
    assert dist.survival_after(1.0) == pytest.approx(0.25, abs=1e-15)
    assert dist.survival_at(2.5) == pytest.approx(0.0, abs=1e-15)    # the knocked-in notes settled at 2.0
    assert dist.determination_times == {EventType.MATURITY_NO_KO: 1.0, EventType.MATURITY_WITH_KI: 2.0}
    assert dist.payment_times_for(EventType.MATURITY_WITH_KI) == pytest.approx(2.02)


# --- REBASED (spec D13) -------------------------------------------------------------------------------

#: (time, event, probability); payment one hundredth of a year later. Sums to 1.
REBASED_ROWS = (
    (0.25, "ko_pre", 0.30),
    (0.50, "ko_pre", 0.10),
    (0.50, "ko_post", 0.05),
    (0.75, "ko_post", 0.04),
    (1.00, "ko_post", 0.03),
    (1.00, "maturity", 0.28),
    (1.25, "ki_settlement", 0.12),
    (1.50, "ki_settlement", 0.08),
)


def _rebased() -> EventStatistics:
    times = np.array([r[0] for r in REBASED_ROWS])
    probability = np.array([r[2] for r in REBASED_ROWS])
    termination = TerminationDistribution(
        times=times, payment_times=times + 0.01, event=tuple(r[1] for r in REBASED_ROWS),
        probability=probability, discounted_cash=probability * 99.0, cash_to_pay=probability * 100.0,
        survival_times=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        # spec D29: a knock-out leaves at its row, maturity / knocked-in settlement right after their time
        survival=[0.70, 0.55, 0.51, 0.48, 0.20, 0.08],
    )
    offsets = PostKIOffsetTable(offsets=[0.25, 0.5, 0.75, 1.0], kind=("ko_post", "ko_post", "ko_post", "ki_settlement"),
                                probability=[0.05, 0.04, 0.03, 0.20], discounted_cash=[4.9, 3.9, 2.9, 19.0],
                                cash_to_pay=[5.0, 4.0, 3.0, 20.0])
    obs = _observations((0.25, 0.5), (0.30, 0.10), schedule=("pre", "pre"))
    receivables = ReceivablesTable.empty()
    return EventStatistics(
        product_family="ko_reset", pv=99.0, observations=obs, coupons=None, maturity=_row(1.0, 0.28, pay_lag=0.01),
        ki_settlement=_row(1.5, 0.20, pay_lag=0.01),
        knock_in=KnockInStatistics(probability=0.35, probability_no_ko=0.20, expected_time=0.5, expected_loss=3.0),
        ki_table=_ki_table((0.25, 0.5), (0.2, 0.15)), post_ki_offsets=offsets, termination=termination,
        receivables=receivables, ledger=build_rebased_cash_ledger(termination=termination, receivables=receivables),
        expected_lifetime=0.8, provenance=PROV,
    )


def test_rebased_maps_the_termination_distribution_to_array_streams():
    dist = EventDistribution.from_statistics(_rebased())
    assert dist.rebased
    ko = [(t, p) for t, e, p in REBASED_ROWS if e in ("ko_pre", "ko_post")]
    np.testing.assert_array_equal(dist.stream_times[EventType.KO], [t for t, _ in ko])
    np.testing.assert_array_equal(dist.probabilities[EventType.KO], [p for _, p in ko])
    # MATURITY_WITH_KI is an ARRAY stream on REBASED: path-dependent settlement dates
    with_ki = dist.probabilities[EventType.MATURITY_WITH_KI]
    assert isinstance(with_ki, np.ndarray)
    np.testing.assert_array_equal(with_ki, [0.12, 0.08])
    np.testing.assert_array_equal(dist.stream_times[EventType.MATURITY_WITH_KI], [1.25, 1.5])
    np.testing.assert_array_equal(dist.payment_times_for(EventType.MATURITY_WITH_KI), [1.26, 1.51])
    assert dist.probabilities[EventType.MATURITY_NO_KO] == pytest.approx(0.28, abs=1e-15)
    assert dist.payment_times_for(EventType.MATURITY_NO_KO) == pytest.approx(1.01)
    # survival is the termination distribution's absolute curve (D29: the 1.25 settlement alive at 1.25, gone after)
    assert dist.survival_at(1.25) == pytest.approx(0.20)
    assert dist.survival_after(1.25) == pytest.approx(0.08)
    assert dist.survival_at(0.125) == pytest.approx(0.85)


def _rebased_leg(schedule=(0.25, 0.5, 0.75, 1.0)) -> AutocallableCashLeg:
    obs = tuple(schedule)
    return AutocallableCashLeg(
        direction=LegDirection.BUYER_PAYS, leg_type=AutocallableLegType.MARGIN, notional=1_000_000.0, rate=0.05,
        observation_schedule=obs, accrual_factors=tuple(0.9 * t for t in obs),
        settlement_schedule=tuple(t + 0.02 for t in obs), terminal_accrual_factor=1.1, terminal_settlement_time=1.03,
        pv_formula=PvFormula.NORMAL,
        terminal_events=frozenset({EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI}),
    )


def test_rebased_autocallable_leg_accrues_to_each_event_time():
    """Each knock-out is valued with the leg's own entry for the event's time; each knocked-in settlement
    pays the terminal branch with the note's actual settlement."""
    dist = EventDistribution.from_statistics(_rebased())
    leg = _rebased_leg()
    contingent = sum(0.9 * t * p * _df(t + 0.02) for t, e, p in REBASED_ROWS if e in ("ko_pre", "ko_post"))
    contingent += 1.1 * 0.28 * _df(1.03)
    contingent += sum(1.1 * p * _df(t + 0.01) for t, e, p in REBASED_ROWS if e == "ki_settlement")
    expected = -1_000_000.0 * 0.05 * contingent
    assert leg.value(dist, _env(), 0.0) == pytest.approx(expected, rel=1e-13)


def test_rebased_autocallable_leg_refuses_an_event_time_off_its_schedule():
    dist = EventDistribution.from_statistics(_rebased())
    with pytest.raises(ValidationError, match="0.75"):
        _rebased_leg(schedule=(0.25, 0.5, 1.0)).value(dist, _env(), 0.0)


def test_rebased_fixed_payoff_legs_pay_at_each_settlement():
    dist = EventDistribution.from_statistics(_rebased())
    env = _env()
    with_ki = FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=4_000.0,
                             trigger=PaymentTrigger.AT_MATURITY_WITH_KI)
    assert with_ki.value(dist, env, 0.0) == pytest.approx(4_000.0 * (0.12 * _df(1.26) + 0.08 * _df(1.51)), rel=1e-13)
    any_maturity = FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=3_000.0,
                                  trigger=PaymentTrigger.AT_MATURITY_ANY, payment_offset_days=3)
    off = 3.0 / 365.0
    expected = 3_000.0 * (0.28 * _df(1.01 + off) + 0.12 * _df(1.26 + off) + 0.08 * _df(1.51 + off))
    assert any_maturity.value(dist, env, 0.0) == pytest.approx(expected, rel=1e-13)
    at_ko = FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=7_000.0, trigger=PaymentTrigger.AT_KO)
    ko = [(t, p) for t, e, p in REBASED_ROWS if e in ("ko_pre", "ko_post")]
    assert at_ko.value(dist, env, 0.0) == pytest.approx(7_000.0 * sum(p * _df(t + 0.01) for t, p in ko), rel=1e-13)


def test_rebased_accrual_leg_truncates_on_the_absolute_survival():
    dist = EventDistribution.from_statistics(_rebased())
    leg = AccrualLeg(direction=LegDirection.BUYER_RECEIVES, rate=0.04,
                     base=BaseAmount(value=200_000.0, mode=BaseAmountMode.ABSOLUTE),
                     schedule=LegSchedule(period_starts=np.array([0.0, 0.5, 1.0]),
                                          period_ends=np.array([0.5, 1.0, 1.5]),
                                          payment_times=np.array([0.5, 1.0, 1.5])),
                     ko_behavior=KOBehavior.TRUNCATE_AT_KO)
    expected = 0.04 * 200_000.0 * sum(0.5 * s * _df(t) for s, t in ((1.0, 0.5), (0.55, 1.0), (0.20, 1.5)))
    assert leg.value(dist, _env(), 0.0) == pytest.approx(expected, rel=1e-13)


# --- constructing a distribution directly -------------------------------------------------------------


def test_an_array_stream_needs_its_own_axis():
    with pytest.raises(NumericalError, match="axis"):
        EventDistribution(probabilities={EventType.KO: np.array([0.4]), EventType.MATURITY_NO_KO: 0.6},
                          survival_times=np.array([0.0, 1.0]), survival_probability=np.array([1.0, 0.6]))


def test_maturity_with_ki_is_scalar_unless_rebased():
    with pytest.raises(NumericalError, match="MATURITY_WITH_KI|maturity_with_ki"):
        EventDistribution(
            probabilities={EventType.MATURITY_NO_KO: 0.6, EventType.MATURITY_WITH_KI: np.array([0.4])},
            stream_times={EventType.MATURITY_WITH_KI: np.array([1.0])},
            survival_times=np.array([0.0, 1.0]), survival_probability=np.array([1.0, 1.0]),
        )


# --- CP2: a terminated note; flags carried onto the distribution ---------------------------------------


def _terminated() -> EventStatistics:
    empty = _observations((), ())
    receivables = ReceivablesTable(source=("ko",), decision_times=[-0.02], payment_times=[0.03],
                                   discounted_cash=[104.0], cash_to_pay=[105.0])
    return _stats(observations=empty, maturity=_row(1.0, 0.0), ki_settlement=_row(1.0, 0.0),
                  ki_table=KnockInTable.empty(), receivables=receivables, knock_in_probability=0.0)


def test_a_terminated_note_is_an_empty_distribution_flagged_terminated():
    stats = _terminated()
    assert stats.terminated
    dist = EventDistribution.from_statistics(stats)
    assert dist.terminated
    assert dist.probabilities == {} and dist.stream_times == {}
    assert dist.probability_residual == 0.0


def test_event_dependent_legs_refuse_a_terminated_note_and_event_free_legs_still_value():
    from quantark.cashleg import DeterministicLeg, PaymentConvention

    dist = EventDistribution.from_statistics(_terminated())
    env = _env()
    period = dict(direction=LegDirection.BUYER_RECEIVES, rate=0.04,
                  base=BaseAmount(value=1.0, mode=BaseAmountMode.ABSOLUTE),
                  schedule=LegSchedule(period_starts=np.array([0.0]), period_ends=np.array([0.5]),
                                       payment_times=np.array([0.5])))
    refused = [
        *(FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=1.0, trigger=t) for t in PaymentTrigger),
        _rebased_leg(),
        AccrualLeg(ko_behavior=KOBehavior.TRUNCATE_AT_KO, **period),
        AccrualLeg(ko_behavior=KOBehavior.PAY_FULL_SCHEDULE, payment_convention=PaymentConvention.AT_MATURITY,
                   **period),
    ]
    for leg in refused:
        with pytest.raises(ValidationError, match="terminated"):
            leg.value(dist, env, 0.0)
    premium = DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=5.0, payment_time=0.1)
    assert premium.value(dist, env, 0.0) == pytest.approx(-5.0 * _df(0.1))
    full = AccrualLeg(ko_behavior=KOBehavior.PAY_FULL_SCHEDULE, **period)      # reads no event
    assert full.value(dist, env, 0.0) == pytest.approx(0.04 * 0.5 * _df(0.5))


def test_a_flagged_ki_table_carries_its_flag_onto_the_ki_stream():
    stats = _stats(observations=_observations(KO_TIMES, KO_P), maturity=_row(1.0, 0.42),
                   ki_settlement=_row(1.0, 0.08), ki_table=_ki_table(KO_TIMES, (0.01, 0.02, 0.03, 0.04)),
                   unavailable={"ki_table": "transition: legacy producer"}, knock_in_probability=0.1)
    dist = EventDistribution.from_statistics(stats)
    assert EventType.KI in dist.probabilities
    assert dist.flagged_streams[EventType.KI] == "transition: legacy producer"
    assert EventType.KI not in EventDistribution.from_statistics(_snowball()).flagged_streams
