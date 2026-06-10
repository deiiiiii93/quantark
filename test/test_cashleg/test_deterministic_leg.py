import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
from datetime import datetime

import pytest

from quantark.cashleg.base import LegDirection
from quantark.cashleg.deterministic_leg import DeterministicLeg
from quantark.cashleg.event_distribution import EventDistribution
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment


def _env(rate: float = 0.05):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )


def test_front_premium_pv_equals_amount():
    leg = DeterministicLeg(
        amount=1_000_000.0,
        payment_time=0.0,
        direction=LegDirection.BUYER_PAYS,
        name="Front Premium",
    )
    pv = leg.value(EventDistribution.trivial(1.0), _env(), position_notional=0.0)
    assert pv == pytest.approx(-1_000_000.0, abs=1e-6)


def test_backend_premium_pv_discounted():
    leg = DeterministicLeg(
        amount=1_000_000.0,
        payment_time=1.0,
        direction=LegDirection.BUYER_RECEIVES,
    )
    pv = leg.value(EventDistribution.trivial(1.0), _env(rate=0.05), 0.0)
    assert pv == pytest.approx(1_000_000.0 * math.exp(-0.05), rel=1e-9)


def test_does_not_require_event_distribution():
    leg = DeterministicLeg(
        amount=1.0, payment_time=0.0, direction=LegDirection.BUYER_PAYS
    )
    assert leg.requires_event_distribution() is False
