"""T8 — QuantLib cross-check of the dated bootstrap (AC17).

Skipped when QuantLib is not importable (CI); executed here with QuantLib 1.44.

Conventions are matched so that every day-count fraction and curve time agrees
to the last ulp, and any DF difference ≫ 1e-10 would be a genuine bootstrap or
interpolation disagreement, never a convention artefact:

* time axis ``Actual365Fixed`` on both sides (``act365f`` = ``(d − VD).days/365``);
* ``NullCalendar`` + ``Unadjusted`` (ours: ``CalendarType.NONE`` +
  ``BusinessDayConvention.UNADJUSTED``), settlement 0 days, so the swaps'
  effective date is the valuation date and no payment date rolls;
* deposits at whole-day tenors ``Period(n, Days)`` ↔ ``DepositQuote(maturity_date
  = VD + n days, day_count=ACT_365)``, so ``DF = 1/(1 + r·n/365)``;
* annual fixed and floating legs with a 1Y Ibor index (fixing days 0,
  ``Actual365Fixed``) ↔ ``SwapQuote(payment_frequency=ANNUAL, fixed/float day
  count ACT_365)`` on a ``RateIndex`` with ``tenor_months=12``,
  ``fixing_lag_days=0``; both schedules are generated backward from maturity
  on whole-year anniversaries, and the floating index window equals the accrual
  window, so each forward is ``(P(t_a)/P(t_b) − 1)/dcf`` on both sides;
* ``PiecewiseLogLinearDiscount`` ↔ ``LOG_LINEAR_DF`` (log DF linear in t through
  DF(0) = 1) and ``PiecewiseLinearZero`` ↔ ``LINEAR_ZERO`` (zero rate linear in t,
  flat before the first pillar: QuantLib's ``ZeroYield::updateGuess`` copies the
  first pillar's rate to the reference-date node; ``TermInterpolator`` holds
  ``level(t) = y_1`` for ``t < T_1``).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

ql = pytest.importorskip("QuantLib")

from dateutil.relativedelta import relativedelta  # noqa: E402

from quantark.param import (  # noqa: E402
    BootstrapConfig,
    CurveInterpolation,
    DepositQuote,
    RateIndex,
    SwapQuote,
    bootstrap_discount_curve,
    reprice_quotes,
)
from quantark.util.calendar import BusinessDayConvention, CalendarType, DayCountConvention  # noqa: E402
from quantark.util.enum import PaymentFrequency  # noqa: E402

VD = datetime(2026, 11, 5)
DEPOSIT_DAYS_RATES = [(30, 0.0180), (91, 0.0190), (182, 0.0205)]
SWAP_YEARS_RATES = [(1, 0.0220), (2, 0.0245), (3, 0.0262), (5, 0.0285), (7, 0.0298), (10, 0.0306)]

IBOR_1Y = RateIndex(
    name="IBOR_1Y_ACT365",
    tenor_months=12,
    day_count_convention=DayCountConvention.ACT_365,
    fixing_lag_days=0,
    calendar_type=CalendarType.NONE,
    currency="CNY",
    description="1Y term index with QuantLib-matched conventions",
)

TOL = 1e-10


def _our_quotes():
    deposits = [
        DepositQuote(rate=r, maturity_date=VD + timedelta(days=n), day_count=DayCountConvention.ACT_365, label=f"{n}D")
        for n, r in DEPOSIT_DAYS_RATES
    ]
    swaps = [
        SwapQuote(
            fixed_rate=s,
            index=IBOR_1Y,
            maturity_date=VD + relativedelta(years=k),
            payment_frequency=PaymentFrequency.ANNUAL,
            fixed_day_count=DayCountConvention.ACT_365,
            float_day_count=DayCountConvention.ACT_365,
            calendar_type=CalendarType.NONE,
            business_day_convention=BusinessDayConvention.UNADJUSTED,
            label=f"{k}Y",
        )
        for k, s in SWAP_YEARS_RATES
    ]
    return deposits + swaps


def _ql_helpers():
    cal = ql.NullCalendar()
    dc = ql.Actual365Fixed()
    ibor = ql.IborIndex("Ibor1Y", ql.Period(1, ql.Years), 0, ql.CNYCurrency(), cal, ql.Unadjusted, False, dc)
    helpers = [
        ql.DepositRateHelper(ql.QuoteHandle(ql.SimpleQuote(r)), ql.Period(n, ql.Days), 0, cal, ql.Unadjusted, False, dc)
        for n, r in DEPOSIT_DAYS_RATES
    ]
    helpers += [
        ql.SwapRateHelper(ql.QuoteHandle(ql.SimpleQuote(s)), ql.Period(k, ql.Years), cal, ql.Annual, ql.Unadjusted, dc, ibor)
        for k, s in SWAP_YEARS_RATES
    ]
    return cal, dc, helpers


def _ql_curve(kind):
    ql.Settings.instance().evaluationDate = ql.Date(VD.day, VD.month, VD.year)
    cal, dc, helpers = _ql_helpers()
    curve = kind(0, cal, helpers, dc)
    curve.enableExtrapolation()
    return curve


def _sample_times(our_curve):
    pillars = list(our_curve.tenors)
    mids = [0.5 * (a + b) for a, b in zip([0.0] + pillars[:-1], pillars)]
    return pillars, mids


@pytest.mark.parametrize(
    "ql_kind, scheme",
    [
        (lambda *a: ql.PiecewiseLogLinearDiscount(*a), CurveInterpolation.LOG_LINEAR_DF),
        (lambda *a: ql.PiecewiseLinearZero(*a), CurveInterpolation.LINEAR_ZERO),
    ],
    ids=["log-linear-discount", "linear-zero"],
)
def test_ac17_dfs_match_quantlib_at_pillars_and_midpoints(ql_kind, scheme):
    quotes = _our_quotes()
    ours = bootstrap_discount_curve(quotes, BootstrapConfig(interpolation=scheme), valuation_date=VD)
    assert max(abs(r) for r in reprice_quotes(quotes, ours, valuation_date=VD)) <= 1e-12
    theirs = _ql_curve(ql_kind)

    pillars, mids = _sample_times(ours)
    # the pillar dates agree: our act365f times are QuantLib's Actual365Fixed times
    ql_times = [theirs.dayCounter().yearFraction(theirs.referenceDate(), d) for d in theirs.dates()[1:]]
    assert ql_times == pytest.approx(pillars, abs=0.0)

    worst = 0.0
    for t in pillars + mids:
        d = abs(float(ours.get_discount_factor(t)) - theirs.discount(t))
        worst = max(worst, d)
        assert d <= TOL, (t, d)
    assert worst <= TOL


def test_ac17_zero_rates_match_quantlib_before_first_pillar():
    quotes = _our_quotes()
    ours = bootstrap_discount_curve(quotes, BootstrapConfig(interpolation=CurveInterpolation.LINEAR_ZERO), valuation_date=VD)
    theirs = _ql_curve(lambda *a: ql.PiecewiseLinearZero(*a))
    for t in (0.01, 0.04, 30 / 365 / 2):
        ours_z = float(ours.get_rate(t))
        theirs_z = theirs.zeroRate(t, ql.Continuous).rate()
        assert abs(ours_z - theirs_z) <= TOL
