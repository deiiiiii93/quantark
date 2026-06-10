import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.cashleg.event_distribution import EventType, PricingResult
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 1, 1),
    )


def test_default_returns_pricing_result_with_trivial_distribution():
    option = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )
    engine = BlackScholesEngine()

    result = engine.price_with_events(option, _env())
    assert isinstance(result, PricingResult)
    assert result.npv == pytest.approx(engine.price(option, _env()), rel=1e-12)
    assert result.event_distribution is not None
    assert result.event_distribution.probabilities[EventType.MATURITY_NO_KO] == 1.0


def test_emit_distribution_false_still_returns_pricing_result():
    option = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )
    result = BlackScholesEngine().price_with_events(
        option, _env(), emit_distribution=False
    )
    assert result.event_distribution is not None
