import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime

import pytest

from cashleg.base import LegDirection
from cashleg.deterministic_leg import DeterministicLeg
from cashleg.event_distribution import EventDistribution
from cashleg.leg_valuator import LegPV, TradeValueBreakdown, value_leg
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from priceenv import PricingEnvironment


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )


def test_value_leg_delegates_to_leg_value_method():
    leg = DeterministicLeg(
        amount=1000.0,
        payment_time=0.0,
        direction=LegDirection.BUYER_RECEIVES,
    )
    pv = value_leg(leg, EventDistribution.trivial(1.0), _env(), 0.0)
    assert pv == pytest.approx(1000.0)


def test_trade_value_breakdown_total_sums_components():
    leg_pv = LegPV(name="Premium", direction=LegDirection.BUYER_PAYS, pv=-100.0)
    breakdown = TradeValueBreakdown(product_npv=500.0, leg_pvs={"leg-1": leg_pv})
    assert breakdown.total == 400.0


def test_trade_value_breakdown_empty_legs():
    breakdown = TradeValueBreakdown(product_npv=500.0, leg_pvs={})
    assert breakdown.total == 500.0
