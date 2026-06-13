"""
Tests that the FX portfolio feeds the ISDA SIMM v2.6 initial-margin engine.

The same FXPortfolio used by the VaR / stress / dynamic-scenario / backtest
modules also implements the SIMM ``SIMMSensitivityProvider`` protocol, so it
flows through SIMMPortfolioAdapter -> SIMMCalculator with no extra glue.
"""
from datetime import datetime

import pytest

from quantark.asset.fx.engine.analytical import FxDeltaOneEngine, GarmanKohlhagenEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.deltaone import FxForward
from quantark.asset.fx.product.option import FxVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio import FxPosition, FXPosition, FXPortfolio
from quantark.priceenv import FxPricingEnvironment
from quantark.simm import SIMMConfig
from quantark.simm.engines.aggregation import SIMMCalculator
from quantark.simm.engines.base import SIMMSensitivityProvider
from quantark.simm.engines.portfolio_adapter import SIMMPortfolioAdapter
from quantark.util.enum import OptionType


def _book():
    env = FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12), spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=0.10))
    pf = FXPortfolio(portfolio_name="fx", pricing_environments={"EURUSD": env})
    pf.add_position(
        product=FxVanillaOption(currency_pair=CurrencyPair("EUR", "USD"), strike=1.25,
                                option_type=OptionType.CALL, maturity=1.0,
                                notional_foreign=1_000_000.0),
        quantity=1.0, entry_price=0.0, underlying="EURUSD", engine=GarmanKohlhagenEngine())
    pf.add_position(
        product=FxForward(currency_pair=CurrencyPair("EUR", "USD"), notional_base=1_000_000.0,
                          contract_rate=1.21, maturity_date=datetime(2027, 6, 14)),
        quantity=1.0, entry_price=0.0, underlying="EURUSD", engine=FxDeltaOneEngine())
    return pf


def test_fx_position_is_simm_provider():
    assert issubclass(FXPosition, SIMMSensitivityProvider) or hasattr(
        FXPosition, "get_simm_sensitivities")
    assert FxPosition is FXPosition


def test_fx_portfolio_generates_simm_sensitivities_and_margin():
    pf = _book()
    config = SIMMConfig(calculation_currency="USD", calculate_delta=True,
                        calculate_vega=True, calculate_curvature=True,
                        derive_curvature_from_vega=True)
    sens = SIMMPortfolioAdapter(config).portfolio_to_sensitivities(pf)

    risk_types = {type(s).__name__ for s in sens.sensitivities}
    assert "FXDeltaSensitivity" in risk_types  # forward + option carry FX delta
    assert "FXVegaSensitivity" in risk_types   # the option carries vega

    result = SIMMCalculator(config).calculate(sens)
    assert result.total_margin > 0


def test_fxposition_alias_keyword_construction():
    """The SIMM-correctness code constructs FxPosition without an explicit underlying."""
    pos = FxPosition(
        product=FxVanillaOption(currency_pair=CurrencyPair("EUR", "USD"), strike=1.25,
                                option_type=OptionType.CALL, maturity=1.0,
                                notional_foreign=1_000_000.0),
        quantity=2.0, engine=GarmanKohlhagenEngine())
    assert pos.underlying == "EURUSD"  # derived from the product's currency pair


def test_simm_margin_rejects_non_usd_quote_when_calc_ccy_usd():
    """Quote currency must equal the calculation currency for built-in FX sensitivities."""
    env = FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12), spot_quote=SpotQuote(spot=150.0),
        domestic_curve=FlatRateCurve(rate=0.001), foreign_curve=FlatRateCurve(rate=0.05),
        vol_surface=FlatVolSurface(volatility=0.10))
    pf = FXPortfolio(portfolio_name="fx", pricing_environments={"USDJPY": env})
    pf.add_position(
        product=FxVanillaOption(currency_pair=CurrencyPair("USD", "JPY"), strike=155.0,
                                option_type=OptionType.CALL, maturity=1.0,
                                notional_foreign=1_000_000.0),
        quantity=1.0, entry_price=0.0, underlying="USDJPY", engine=GarmanKohlhagenEngine())
    config = SIMMConfig(calculation_currency="USD", calculate_delta=True)
    with pytest.raises(Exception):
        SIMMPortfolioAdapter(config).portfolio_to_sensitivities(pf)
