"""D29 counts a settled note as gone from its DETERMINATION time, not its payment time (CP8 ruling I1).

A knock-out leaves survival at its own observation row although its cash settles later; a maturity-type termination
(never-knocked-in maturity, knocked-in settlement) does the same: alive at its determination, gone just after it, even
while its redemption is still unpaid. Consequence (owner-visible, documented in the CHANGELOG Known issues): a
COMPLETE_PERIOD leg whose last period ends on a settlement date after the determination pays nothing on the matured
mass.
"""
from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pytest

from intraday.conftest import dated_snowball
from quantark.asset.equity.engine.quad import SnowballQuadEngineV2
from quantark.cashleg.accrual_leg import AccrualLeg, KOBehavior, PaymentConvention, SurvivalBasis
from quantark.cashleg.base import LegDirection
from quantark.cashleg.base_amount import BaseAmount, BaseAmountMode
from quantark.cashleg.event_distribution import EventDistribution
from quantark.cashleg.leg_schedule import LegSchedule
from quantark.util.calendar.day_counter import DayCountConvention
from test_asset.equity.engine_quad.test_quad_v2_event_statistics import SSE, dated_env

warnings.filterwarnings("ignore", category=DeprecationWarning)

EPS = 1e-9


def _settlement_lagged():
    """A dated Snowball whose maturity settles five trading days after its determination (2027-03-16 -> 03-23)."""
    product = dated_snowball(SSE, datetime(2026, 3, 16))
    product.settlement_date = SSE.trading_days_after(product.exercise_date, 5)
    product.validate()
    market = dated_env()
    stats = SnowballQuadEngineV2().calculate_event_stats(product, market)
    return stats, EventDistribution.from_statistics(stats), market


def _leg(ends, basis):
    starts = np.r_[0.0, ends[:-1]]
    return AccrualLeg(
        rate=0.04, base=BaseAmount(value=1_000_000.0, mode=BaseAmountMode.ABSOLUTE),
        schedule=LegSchedule(period_starts=starts, period_ends=np.asarray(ends), payment_times=np.asarray(ends)),
        day_count=DayCountConvention.ACT_365, payment_convention=PaymentConvention.AT_PERIOD_END,
        ko_behavior=KOBehavior.TRUNCATE_AT_KO, survival_basis=basis, direction=LegDirection.BUYER_RECEIVES)


def test_a_matured_note_is_gone_right_after_its_determination_not_its_payment():
    stats, dist, _ = _settlement_lagged()
    determination, payment = stats.maturity.time, stats.maturity.payment_time
    assert payment > determination + 5.0 / 365.0 - 1e-12          # the settlement lag is real
    matured = stats.maturity.probability + stats.ki_settlement.probability
    assert matured > 0.2
    # alive at its determination (the node keeps maturity-type mass), gone just after, and still gone while unpaid
    assert dist.survival_at(determination) == pytest.approx(matured, abs=1e-12)
    for t in (determination + EPS, 0.5 * (determination + payment), payment):
        assert dist.survival_at(t) == pytest.approx(0.0, abs=1e-12)
    assert dist.survival_after(determination) == pytest.approx(0.0, abs=1e-12)


def test_a_complete_period_leg_ending_on_the_settlement_date_pays_nothing_on_the_matured_mass():
    stats, dist, market = _settlement_lagged()
    payment = stats.maturity.payment_time
    to_settlement = _leg([0.5, payment], SurvivalBasis.COMPLETE_PERIOD)
    first_period_only = _leg([0.5], SurvivalBasis.COMPLETE_PERIOD)
    # the period ending on the settlement date completes on no path: matured notes left at determination, the rest
    # knocked out earlier
    assert to_settlement.value(dist, market, 0.0) == pytest.approx(first_period_only.value(dist, market, 0.0),
                                                                   rel=1e-12, abs=1e-9)
    # ... while the same period ending AT the determination still pays on the matured mass
    to_determination = _leg([0.5, stats.maturity.time], SurvivalBasis.COMPLETE_PERIOD)
    assert to_determination.value(dist, market, 0.0) > first_period_only.value(dist, market, 0.0) + 1_000.0
