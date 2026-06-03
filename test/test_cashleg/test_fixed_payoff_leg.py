import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
from datetime import datetime

import numpy as np
import pytest

from cashleg.base import LegDirection
from cashleg.event_distribution import EventDistribution, EventType
from cashleg.fixed_payoff_leg import FixedPayoffLeg, PaymentTrigger
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment
from util.exceptions import ValidationError


def _env(rate=0.05):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )


def test_at_ko_pv_equals_amount_times_p_ko_times_df():
    dist = EventDistribution(
        event_times=np.array([0.25, 0.5, 1.0]),
        event_dates=None,
        probabilities={
            EventType.KO: np.array([0.3, 0.2, 0.1]),
            EventType.MATURITY_NO_KO: 0.4,
        },
        survival_probability=np.array([1.0, 0.7, 0.5, 0.4]),
    )
    leg = FixedPayoffLeg(
        amount=10_000.0,
        trigger=PaymentTrigger.AT_KO,
        direction=LegDirection.BUYER_RECEIVES,
    )
    pv = leg.value(dist, _env(rate=0.05), position_notional=0.0)
    expected = 10_000.0 * (
        0.3 * math.exp(-0.05 * 0.25)
        + 0.2 * math.exp(-0.05 * 0.5)
        + 0.1 * math.exp(-0.05 * 1.0)
    )
    assert pv == pytest.approx(expected, rel=1e-9)


def test_at_maturity_no_ko_pv():
    dist = EventDistribution(
        event_times=np.array([0.5, 1.0]),
        event_dates=None,
        probabilities={
            EventType.KO: np.array([0.3, 0.0]),
            EventType.MATURITY_NO_KO: 0.7,
        },
        survival_probability=np.array([1.0, 0.7, 0.7]),
    )
    leg = FixedPayoffLeg(
        amount=50_000.0,
        trigger=PaymentTrigger.AT_MATURITY_NO_KO,
        direction=LegDirection.BUYER_RECEIVES,
    )
    pv = leg.value(dist, _env(rate=0.05), position_notional=0.0)
    expected = 50_000.0 * 0.7 * math.exp(-0.05)
    assert pv == pytest.approx(expected, rel=1e-9)


def test_missing_trigger_in_distribution_raises():
    leg = FixedPayoffLeg(
        amount=1.0,
        trigger=PaymentTrigger.AT_KI,
        direction=LegDirection.BUYER_RECEIVES,
    )
    with pytest.raises(ValidationError, match="trigger"):
        leg.value(EventDistribution.trivial(1.0), _env(), position_notional=0.0)


def test_payment_offset_days_shifts_discount():
    dist = EventDistribution(
        event_times=np.array([1.0]),
        event_dates=None,
        probabilities={EventType.KO: np.array([1.0])},
        survival_probability=np.array([1.0, 0.0]),
    )
    leg = FixedPayoffLeg(
        amount=1_000.0,
        trigger=PaymentTrigger.AT_KO,
        payment_offset_days=365,
        direction=LegDirection.BUYER_RECEIVES,
    )
    pv = leg.value(dist, _env(rate=0.05), position_notional=0.0)
    assert pv == pytest.approx(1_000.0 * math.exp(-0.05 * 2.0), rel=1e-9)
