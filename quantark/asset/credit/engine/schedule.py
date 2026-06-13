"""
Premium-leg coupon schedule construction for CDS-type products.

Shared by the reduced-form single-name engine and the basket-CDS Monte-Carlo
engine so both build identical schedules — in particular the final (possibly
short) **stub** coupon that ends exactly at maturity. Omitting the stub drops
the premium of any contract shorter than one full coupon period and can leave
the risky annuity (RPV01) non-positive.

Two construction modes are provided:

* :func:`cds_coupon_schedule` — *tenor-based*. Builds the schedule from a float
  ``maturity`` (years from valuation). Used for un-seasoned pricing and by the
  basket engine, which does not yet support roll-down.
* :func:`cds_coupon_schedule_asof` — *date-based*. Builds the full contractual
  schedule between ``effective_date`` and ``maturity_date`` and returns only the
  coupons that fall **after** ``as_of_date``, with payment times re-expressed
  relative to the as-of date. This is what makes a CDS price as a seasoned trade
  when the valuation clock advances (roll-down / carry).
"""
from datetime import datetime
from typing import List, Tuple

# Tolerance for deciding whether a regular coupon date coincides with maturity.
_EPS = 1e-9
_DAYS_PER_YEAR = 365.0
_SECONDS_PER_DAY = 86_400.0


def year_fraction_act365(start: datetime, end: datetime) -> float:
    """
    Signed ACT/365 year fraction between two dates (the project's
    ``CALENDAR_DAYS`` convention, ``actual_days / 365``).

    Unlike :func:`quantark.util.calendar.calculate_year_fraction` this permits a
    zero or negative result (``end`` on or before ``start``), which the as-of
    pricing path needs when the valuation date sits before the effective date or
    exactly on a coupon date.
    """
    return (end - start).total_seconds() / (_DAYS_PER_YEAR * _SECONDS_PER_DAY)


def _contractual_times(maturity: float, payment_freq: int) -> List[float]:
    """
    Regular ``1 / payment_freq`` coupon times on ``(0, maturity]`` ending with a
    (possibly short) stub exactly at ``maturity``.
    """
    interval = 1.0 / payment_freq
    times: List[float] = []
    t = interval
    while t < maturity - _EPS:
        times.append(t)
        t += interval
    times.append(maturity)  # final coupon ends exactly at maturity (stub if short)
    return times


def cds_coupon_schedule(maturity: float, payment_freq: int) -> List[Tuple[float, float]]:
    """
    Return ``[(payment_time, accrual_fraction), ...]`` for a CDS premium leg.

    Coupons fall on regular ``1 / payment_freq`` intervals. The final period is
    a (possibly short) stub ending exactly at ``maturity`` so contracts shorter
    than one full coupon period still carry their stub premium and the accrual
    fraction of each coupon is the actual time since the previous coupon.

    Args:
        maturity: Time to maturity in years (> 0).
        payment_freq: Premium payments per year (> 0).

    Returns:
        Ordered list of ``(payment_time, accrual_fraction)`` pairs covering
        ``(0, maturity]`` with no overlap or gap.
    """
    schedule: List[Tuple[float, float]] = []
    prev = 0.0
    for pay in _contractual_times(maturity, payment_freq):
        schedule.append((pay, pay - prev))
        prev = pay
    return schedule


def cds_coupon_schedule_asof(
    effective_date: datetime,
    maturity_date: datetime,
    payment_freq: int,
    as_of_date: datetime,
) -> List[Tuple[float, float]]:
    """
    Remaining premium-leg coupons for a seasoned CDS, as of ``as_of_date``.

    The contractual coupon calendar is fixed at ``effective_date`` (regular
    ``1 / payment_freq`` intervals with a stub at ``maturity_date``). Coupons
    whose contractual payment date is on or before ``as_of_date`` have settled
    and are dropped. Each surviving coupon is returned as
    ``(payment_time, accrual_fraction)`` where:

    * ``payment_time`` is years from ``as_of_date`` to the coupon date (so the
      caller discounts/survives from the as-of date), and
    * ``accrual_fraction`` is the **full contractual** period since the previous
      coupon date — the premium that accrues since the previous contractual
      coupon and is paid in full if the name survives to the payment date, even
      when ``as_of_date`` sits mid-period.

    When ``as_of_date`` is *before* ``effective_date`` (a forward-starting CDS)
    ``as_of`` is negative and every coupon is still outstanding, with payment
    times pushed out by the forward-start delay. Returns an empty list only when
    the contract has fully matured.
    """
    total = year_fraction_act365(effective_date, maturity_date)
    as_of = year_fraction_act365(effective_date, as_of_date)

    schedule: List[Tuple[float, float]] = []
    prev = 0.0
    for pay in _contractual_times(total, payment_freq):
        accrual = pay - prev
        prev = pay
        if pay > as_of + _EPS:  # coupon still to be paid
            schedule.append((pay - as_of, accrual))
    return schedule


def contractual_coupon_dates(
    effective_date: datetime,
    maturity_date: datetime,
    payment_freq: int,
) -> List[Tuple[datetime, float]]:
    """
    Absolute contractual coupon dates and their accrual fractions.

    Returns ``[(payment_date, accrual_fraction), ...]`` over the whole life of
    the trade. Used by the dynamic-scenario and backtest cashflow ledgers to
    detect which coupons settle as the valuation date advances (so paid coupons
    are booked as realized cash rather than silently vanishing from PV).
    """
    from datetime import timedelta

    dates: List[Tuple[datetime, float]] = []
    prev = 0.0
    for pay in _contractual_times(
        year_fraction_act365(effective_date, maturity_date), payment_freq
    ):
        accrual = pay - prev
        prev = pay
        pay_date = effective_date + timedelta(days=pay * _DAYS_PER_YEAR)
        dates.append((pay_date, accrual))
    return dates
