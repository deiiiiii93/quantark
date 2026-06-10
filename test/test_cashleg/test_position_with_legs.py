import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math
from datetime import datetime

import pytest

from asset.equity.engine.analytical import BlackScholesEngine
from asset.equity.product.option import EuropeanVanillaOption
from asset.equity.riskmeasures import GreeksCalculator
from cashleg import DeterministicLeg, LegDirection
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from portfolio.equity.position import EquityPosition
from priceenv import PricingEnvironment
from util.enum import OptionType


def _env(rate=0.05):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )


def _option():
    return EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )


def test_get_trade_value_includes_premium_leg():
    option = _option()
    engine = BlackScholesEngine()
    env = _env()
    product_pv = engine.price(option, env)

    pos = EquityPosition(
        product=option,
        quantity=1.0,
        entry_price=product_pv,
        underlying="SPX",
        engine=engine,
        entry_timestamp=datetime(2026, 1, 1),
        cash_legs=[
            DeterministicLeg(
                amount=100.0,
                payment_time=0.0,
                direction=LegDirection.BUYER_PAYS,
                name="Premium",
            ),
        ],
    )
    assert pos.get_trade_value(env) == pytest.approx(product_pv - 100.0, rel=1e-9)


def test_get_trade_value_breakdown_attributes_per_leg():
    pos = EquityPosition(
        product=_option(),
        quantity=2.0,
        entry_price=5.0,
        underlying="SPX",
        engine=BlackScholesEngine(),
        entry_timestamp=datetime(2026, 1, 1),
        cash_legs=[
            DeterministicLeg(
                amount=100.0,
                payment_time=0.0,
                direction=LegDirection.BUYER_PAYS,
                name="Premium",
            ),
            DeterministicLeg(
                amount=50.0,
                payment_time=1.0,
                direction=LegDirection.BUYER_RECEIVES,
                name="Backend",
            ),
        ],
    )
    breakdown = pos.get_trade_value_breakdown(_env())
    assert len(breakdown.leg_pvs) == 2
    premium_pv = next(v.pv for v in breakdown.leg_pvs.values() if v.name == "Premium")
    backend_pv = next(v.pv for v in breakdown.leg_pvs.values() if v.name == "Backend")
    assert premium_pv == pytest.approx(-100.0 * 2.0, rel=1e-9)
    assert backend_pv == pytest.approx(50.0 * math.exp(-0.05) * 2.0, rel=1e-9)


def test_two_deterministic_legs_of_same_type_both_priced():
    pos = EquityPosition(
        product=_option(),
        quantity=1.0,
        entry_price=5.0,
        underlying="SPX",
        engine=BlackScholesEngine(),
        entry_timestamp=datetime(2026, 1, 1),
        cash_legs=[
            DeterministicLeg(
                amount=100.0,
                payment_time=0.0,
                direction=LegDirection.BUYER_PAYS,
                name="Premium A",
            ),
            DeterministicLeg(
                amount=100.0,
                payment_time=0.0,
                direction=LegDirection.BUYER_PAYS,
                name="Premium B",
            ),
        ],
    )
    breakdown = pos.get_trade_value_breakdown(_env())
    assert len(breakdown.leg_pvs) == 2
    assert sum(v.pv for v in breakdown.leg_pvs.values()) == pytest.approx(-200.0)


def test_get_trade_greeks_for_position_with_only_premium_leg():
    engine = BlackScholesEngine()
    option = _option()
    env = _env()
    pos = EquityPosition(
        product=option,
        quantity=1.0,
        entry_price=5.0,
        underlying="SPX",
        engine=engine,
        entry_timestamp=datetime(2026, 1, 1),
        cash_legs=[
            DeterministicLeg(
                amount=100.0,
                payment_time=0.5,
                direction=LegDirection.BUYER_PAYS,
                name="Mid Premium",
            ),
        ],
    )
    calc = GreeksCalculator()
    greeks = pos.get_trade_greeks(env, calc)

    product_only = EquityPosition(
        product=option,
        quantity=1.0,
        entry_price=5.0,
        underlying="SPX",
        engine=engine,
        entry_timestamp=datetime(2026, 1, 1),
    )
    product_greeks = product_only.get_greeks(env, calc)
    assert greeks["delta"] == pytest.approx(product_greeks["delta"], rel=1e-3)
