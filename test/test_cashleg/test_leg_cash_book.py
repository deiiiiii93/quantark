"""Multi-day cash-leg booking (ship spec 2026-10-23 portfolio-trade-value D11–D14; AC9a, AC9b, AC14 book level).

``LegCashBook`` rolls the ORIGINAL legs by the run time ``τ_d = whole days / 365`` from the anchor, pays each
flow once at nominal on the first day the shared landing predicate says it is no longer ahead, and on
termination turns every unpaid flow into a fixed receivable by explicit per-leg rules.
"""
import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantark.cashleg import (
    AccrualLeg,
    BaseAmount,
    BaseAmountMode,
    DeterministicLeg,
    EventDistribution,
    EventType,
    KOBehavior,
    LegDirection,
    LegSchedule,
)
from quantark.cashleg.accrual_leg import PaymentConvention, SurvivalBasis
from quantark.cashleg.autocallable_leg import AccrualBasis, AutocallableCashLeg, AutocallableLegType, PvFormula
from quantark.cashleg.booking import LegCashBook, RealizedOutcome, run_time
from quantark.cashleg.fixed_payoff_leg import FixedPayoffLeg, PaymentTrigger
from quantark.execution.errors import CapabilityError
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError

ANCHOR = datetime(2026, 3, 16)
RATE = 0.03
D = 1 / 365.0
QUARTERS = [91, 182, 273, 365]          # observation / period-boundary days
UNIT = 1_000_000.0


def _env(day=0):
    return PricingEnvironment(spot_quote=SpotQuote(spot=100.0, asset_name="X"), vol_surface=FlatVolSurface(0.2),
                              rate_curve=FlatRateCurve(RATE), div_yield=ContinuousDividendYield(0.01),
                              valuation_date=ANCHOR + timedelta(days=day))


def _df(t):
    return math.exp(-RATE * t)


def _accrual(ko, basis, convention=PaymentConvention.AT_PERIOD_END, name=None):
    starts = np.array([0] + QUARTERS[:-1]) * D
    ends = np.array(QUARTERS) * D
    return AccrualLeg(direction=LegDirection.BUYER_RECEIVES, rate=0.04, name=name,
                      base=BaseAmount(value=0.5, mode=BaseAmountMode.NOTIONAL_FRACTION),
                      schedule=LegSchedule(period_starts=starts, period_ends=ends, payment_times=ends.copy()),
                      payment_convention=convention, ko_behavior=ko, survival_basis=basis)


def _rebate():
    obs = tuple(d * D for d in QUARTERS)
    return AutocallableCashLeg(direction=LegDirection.BUYER_RECEIVES, leg_type=AutocallableLegType.REBATE,
                               notional=10_000.0, rate=0.05, observation_schedule=obs, accrual_factors=obs,
                               settlement_schedule=tuple(t + 2 * D for t in obs), terminal_accrual_factor=1.0,
                               terminal_settlement_time=367 * D, accrual_basis=AccrualBasis.KO_MATURITY,
                               terminal_events=frozenset({EventType.MATURITY_NO_KO}))


def _legs():
    return {
        "fee": DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=250.0, payment_time=30 * D),
        "trunc_enter": _accrual(KOBehavior.TRUNCATE_AT_KO, SurvivalBasis.ENTER_PERIOD),
        "trunc_complete": _accrual(KOBehavior.TRUNCATE_AT_KO, SurvivalBasis.COMPLETE_PERIOD),
        "full_complete": _accrual(KOBehavior.PAY_FULL_SCHEDULE, SurvivalBasis.COMPLETE_PERIOD),
        "at_maturity": _accrual(KOBehavior.TRUNCATE_AT_KO, SurvivalBasis.ENTER_PERIOD, PaymentConvention.AT_MATURITY),
        "ko_payoff": FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=1_000.0,
                                    trigger=PaymentTrigger.AT_KO, payment_offset_days=2),
        "rebate": _rebate(),
    }


def _coupon(i):
    return 0.04 * 0.5 * UNIT * (QUARTERS[i] - ([0] + QUARTERS)[i]) * D


KO_DAY, KO_SETTLE_DAY, MAT_SETTLE_DAY = 182, 184, 367


def _expected_flows():
    """(leg, day paid, nominal) for the path that knocks out on day 182 (an observation and a period boundary)."""
    c = _coupon
    return sorted([
        ("fee", 30, -250.0),
        ("trunc_enter", 91, c(0)), ("trunc_enter", 182, c(1)),               # [182, 273) starts ON the KO date
        ("trunc_complete", 91, c(0)),                                         # [91, 182] ends ON the KO date
        ("full_complete", 91, c(0)), ("full_complete", 182, c(1)),
        ("full_complete", 273, c(2)), ("full_complete", 365, c(3)),
        ("at_maturity", MAT_SETTLE_DAY, c(0)), ("at_maturity", MAT_SETTLE_DAY, c(1)),  # periods 0, 1 survived
        ("ko_payoff", KO_SETTLE_DAY + 2, 1_000.0),
        ("rebate", KO_SETTLE_DAY, 10_000.0 * 0.05 * KO_DAY * D),
    ], key=lambda f: (f[1], f[0]))


def _ko():
    return RealizedOutcome.ko(KO_DAY * D, KO_SETTLE_DAY * D, maturity_settlement_time=MAT_SETTLE_DAY * D)


def _run(path_days):
    legs = _legs()
    book = LegCashBook(list(legs.values()), UNIT, ANCHOR)
    names = {leg.leg_id: name for name, leg in legs.items()}
    paid = {}
    days = {}
    for day in path_days:
        date = ANCHOR + timedelta(days=day)
        result = book.terminate(_ko(), date, _env(day)) if day == KO_DAY else book.roll(date, _env(day))
        days[day] = result
        for (leg_id, _idx), amount in book.paid_today:
            paid.setdefault(names[leg_id], []).append((day, amount))
    return book, days, paid


PATH = [0, 10, 30, 60, 91, 120, KO_DAY, 183, KO_SETTLE_DAY, 186, 200, 273, 300, 365, MAT_SETTLE_DAY, 400]


def test_run_time_is_whole_days_over_365():
    assert run_time(ANCHOR, ANCHOR + timedelta(days=91, hours=15)) == 91 / 365.0
    assert run_time(ANCHOR.replace(hour=9), ANCHOR + timedelta(days=1)) == 1 / 365.0


def test_every_flow_is_paid_once_at_nominal_on_its_day():
    _book, days, paid = _run(PATH)
    got = sorted(((name, day, amount) for name, flows in paid.items() for day, amount in flows),
                 key=lambda f: (f[1], f[0]))
    want = _expected_flows()
    assert [(n, d) for n, d, _ in got] == [(n, d) for n, d, _ in want]
    for (_, _, a), (_, _, b) in zip(got, want):
        assert a == pytest.approx(b, rel=1e-15)
    assert sum(r.paid_cash for r in days.values()) == pytest.approx(sum(a for _, _, a in want), rel=1e-14)


def test_receivable_pv_after_termination_accretes_to_nominal():
    _book, days, _paid = _run(PATH)
    for day in (183, 200, 300):
        tau = day * D
        pending = [(d, a) for _, d, a in _expected_flows() if d > day]
        want = sum(a * _df(d * D - tau) for d, a in pending)
        assert days[day].receivable_pv == pytest.approx(want, rel=1e-13)
    assert days[400].receivable_pv == 0.0


def test_live_legs_before_termination_and_none_after():
    book, days, _paid = _run([0, 10])
    live = {type(l).__name__ for l in days[10].live_legs}
    assert "DeterministicLeg" in live and "AutocallableCashLeg" in live
    fee = next(l for l in days[10].live_legs if isinstance(l, DeterministicLeg))
    assert fee.payment_time == 30 * D - 10 * D
    _book2, days2, _ = _run(PATH)
    assert days2[183].live_legs == ()


def test_day_zero_payment_and_landing_tolerance():
    fee0 = DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=50.0, payment_time=0.0)
    near = DeterministicLeg(direction=LegDirection.BUYER_RECEIVES, amount=7.0, payment_time=10 * D + 1e-11)
    book = LegCashBook([fee0, near], UNIT, ANCHOR, terminable=False)
    day0 = book.roll(ANCHOR, _env(0))
    assert day0.paid_cash == -50.0
    assert [l.leg_id for l in day0.live_legs] == [near.leg_id]       # paid at 0 leaves the live legs
    assert book.roll(ANCHOR + timedelta(days=9), _env(9)).paid_cash == 0.0
    assert book.roll(ANCHOR + timedelta(days=10), _env(10)).paid_cash == 7.0  # within 1e-11 of τ: paid


def test_a_skipped_payment_day_pays_on_the_next_run_day():
    fee = DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=250.0, payment_time=30 * D)
    book = LegCashBook([fee], UNIT, ANCHOR, terminable=False)
    book.roll(ANCHOR + timedelta(days=29), _env(29))
    assert book.roll(ANCHOR + timedelta(days=33), _env(33)).paid_cash == -250.0
    assert book.roll(ANCHOR + timedelta(days=34), _env(34)).paid_cash == 0.0


def test_maturity_outcomes():
    legs = _legs()
    for knocked_in in (False, True):
        book = LegCashBook(list(legs.values()), UNIT, ANCHOR)
        flows = {(leg_id, i): (a, t) for (leg_id, i), a, t in book.termination_flows(
            RealizedOutcome.maturity(knocked_in, MAT_SETTLE_DAY * D))}
        rebate = [a for (lid, _), (a, _) in flows.items() if lid == legs["rebate"].leg_id]
        assert rebate == ([] if knocked_in else [10_000.0 * 0.05 * 1.0])   # terminal_events = {NO_KO}
        ko = [a for (lid, _), (a, _) in flows.items() if lid == legs["ko_payoff"].leg_id]
        assert ko == []
        full = [a for (lid, _), (a, _) in flows.items() if lid == legs["trunc_enter"].leg_id]
        assert len(full) == 4                                                # every period pays at maturity


def test_roll_refuses_going_back_and_double_termination():
    book = LegCashBook(list(_legs().values()), UNIT, ANCHOR)
    book.roll(ANCHOR + timedelta(days=10), _env(10))
    with pytest.raises(ValidationError):
        book.roll(ANCHOR + timedelta(days=5), _env(5))
    book.terminate(_ko(), ANCHOR + timedelta(days=KO_DAY), _env(KO_DAY))
    with pytest.raises(ValidationError):
        book.terminate(_ko(), ANCHOR + timedelta(days=KO_DAY), _env(KO_DAY))


def test_snap_moves_leg_times_and_keeps_amounts():
    from quantark.asset.equity.lifecycle.snapping import snap_time
    off = AccrualLeg(direction=LegDirection.BUYER_RECEIVES, rate=0.04,
                     base=BaseAmount(value=1_000.0, mode=BaseAmountMode.ABSOLUTE),
                     schedule=LegSchedule(period_starts=np.array([0.0, 0.25]), period_ends=np.array([0.25, 0.5]),
                                          payment_times=np.array([0.25, 0.5])))
    book = LegCashBook([off], UNIT, ANCHOR, snap=snap_time, terminable=False)
    snapped = book.legs[0]
    np.testing.assert_array_equal(snapped.schedule.payment_times, [91 / 365.0, round(182.5) / 365.0])
    assert list(snapped.accrual_fractions()) == [0.25, 0.25]
    assert snapped.leg_id == off.leg_id


# --- AC9b: the explicit rules agree with leg.value on a probability-1 distribution -------------------------------

def _realized_distribution(outcome):
    """A complete probability-1 distribution on the observation grid (every stream the legs read)."""
    obs = np.array(QUARTERS, dtype=float) * D
    settle = obs + 2 * D
    ko = np.zeros(len(obs))
    survival = np.ones(len(obs) + 1)
    probabilities = {EventType.MATURITY_NO_KO: 0.0, EventType.MATURITY_WITH_KI: 0.0}
    if outcome.kind.name == "KO":
        k = int(np.argmin(np.abs(obs - outcome.event_time)))
        ko[k] = 1.0
        survival[k + 1:] = 0.0
    else:
        probabilities[EventType.MATURITY_WITH_KI if outcome.knocked_in else EventType.MATURITY_NO_KO] = 1.0
    probabilities[EventType.KO] = ko
    return EventDistribution(event_times=obs, event_dates=None, probabilities=probabilities,
                             survival_probability=survival,
                             payment_times={EventType.KO: settle,
                                            EventType.MATURITY_NO_KO: MAT_SETTLE_DAY * D,
                                            EventType.MATURITY_WITH_KI: MAT_SETTLE_DAY * D})


OUTCOMES = [RealizedOutcome.ko(k * D, (k + 2) * D, maturity_settlement_time=MAT_SETTLE_DAY * D) for k in QUARTERS[:3]]
OUTCOMES += [RealizedOutcome.maturity(False, MAT_SETTLE_DAY * D), RealizedOutcome.maturity(True, MAT_SETTLE_DAY * D)]


@pytest.mark.parametrize("outcome", OUTCOMES, ids=lambda o: f"{o.kind.name}-{o.event_time:.3f}-{o.knocked_in}")
@pytest.mark.parametrize("name", sorted(_legs()))
def test_rule_flows_match_leg_value_on_a_probability_one_distribution(name, outcome):
    leg = _legs()[name]
    if name == "fee":
        return  # a deterministic leg has no contingent flows; covered by the paid-flow tests
    env = _env(0)
    book = LegCashBook([leg], UNIT, ANCHOR)
    flows = book.termination_flows(outcome)
    pv = sum(a * env.get_discount_factor(t) for _key, a, t in flows)
    assert pv == pytest.approx(leg.value(_realized_distribution(outcome), env, UNIT), rel=1e-12, abs=1e-12)


# --- AC14 (book level): refusals ---------------------------------------------------------------------------------

@pytest.mark.parametrize("leg", [
    FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=1.0, trigger=PaymentTrigger.AT_KI),
    AutocallableCashLeg(direction=LegDirection.BUYER_RECEIVES, leg_type=AutocallableLegType.MARGIN, notional=1.0,
                        rate=0.1, terminal_accrual_factor=1.0, terminal_settlement_time=1.0,
                        pv_formula=PvFormula.NOTIONAL_MINUS_PAYOFF),
    AutocallableCashLeg(direction=LegDirection.BUYER_RECEIVES, leg_type=AutocallableLegType.REBATE, notional=1.0,
                        rate=0.1, terminal_accrual_factor=1.0, terminal_settlement_time=1.0,
                        accrual_basis=AccrualBasis.COUPON),
    _accrual(KOBehavior.TRUNCATE_AT_KO, SurvivalBasis.ENTER_PERIOD, PaymentConvention.AT_KO),
], ids=["fixed-ki", "notional-minus-payoff", "coupon-basis", "accrual-at-ko"])
def test_unsupported_legs_are_refused(leg):
    with pytest.raises(CapabilityError):
        LegCashBook([leg], UNIT, ANCHOR)


@pytest.mark.parametrize("name", ["at_maturity", "ko_payoff", "rebate"])
def test_termination_paying_legs_need_a_terminable_position(name):
    with pytest.raises(CapabilityError):
        LegCashBook([_legs()[name]], UNIT, ANCHOR, terminable=False)
    LegCashBook([_legs()["fee"], _legs()["trunc_enter"]], UNIT, ANCHOR, terminable=False)
