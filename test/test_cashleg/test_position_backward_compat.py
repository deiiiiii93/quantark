import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime

from asset.equity.engine.analytical import BlackScholesEngine
from asset.equity.product.option import EuropeanVanillaOption
from param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from portfolio.equity.position import EquityPosition
from priceenv import PricingEnvironment
from util.enum import OptionType


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )


def _position():
    return EquityPosition(
        product=EuropeanVanillaOption(
            strike=100.0, option_type=OptionType.CALL, maturity=1.0
        ),
        quantity=10.0,
        entry_price=5.0,
        underlying="SPX",
        engine=BlackScholesEngine(),
        entry_timestamp=datetime(2026, 1, 1),
    )


def test_position_constructs_without_cash_legs():
    assert _position().cash_legs == []


def test_get_market_value_unchanged_without_legs():
    pos = _position()
    mv = pos.get_market_value(_env())
    assert mv > 0.0
    assert mv == 10.0 * pos.engine.price(pos.product, _env())


def test_positional_position_id_argument_still_works():
    option = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )
    pos = EquityPosition(
        option,
        1.0,
        5.0,
        "SPX",
        BlackScholesEngine(),
        datetime(2026, 1, 1),
        "manual-id",
    )
    assert pos.position_id == "manual-id"
    assert pos.cash_legs == []
