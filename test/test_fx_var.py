"""
Tests for FX Value-at-Risk (parametric / historical / Monte Carlo).

The FX VaR engines compute dollar sensitivities by finite-difference
revaluation of the two-rate FxPricingEnvironment, then combine them with the
historical covariance of per-pair factor changes (spot return, vol change,
domestic-rate shift, foreign-rate shift).
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quantark.asset.fx.engine.analytical import FxDeltaOneEngine, GarmanKohlhagenEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.deltaone import FxForward
from quantark.asset.fx.product.option import FxVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.portfolio.fx import FXPortfolio
from quantark.priceenv import FxPricingEnvironment
from quantark.util.enum import OptionType
from quantark.var import VaRConfig


def _env(spot=1.20, r_dom=0.05, r_for=0.03, vol=0.10):
    return FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 12),
        spot_quote=SpotQuote(spot=spot),
        domestic_curve=FlatRateCurve(rate=r_dom),
        foreign_curve=FlatRateCurve(rate=r_for),
        vol_surface=FlatVolSurface(volatility=vol),
    )


def _forward_portfolio():
    pf = FXPortfolio(portfolio_name="fx", pricing_environments={"EURUSD": _env()})
    pf.add_position(
        product=FxForward(currency_pair=CurrencyPair("EUR", "USD"),
                          notional_base=1_000_000.0, contract_rate=1.20,
                          maturity_date=datetime(2027, 6, 14)),
        quantity=1.0, entry_price=0.0, underlying="EURUSD", engine=FxDeltaOneEngine(),
    )
    return pf


def _option_portfolio():
    pf = FXPortfolio(portfolio_name="fx", pricing_environments={"EURUSD": _env()})
    pf.add_position(
        product=FxVanillaOption(currency_pair=CurrencyPair("EUR", "USD"), strike=1.25,
                                option_type=OptionType.CALL, maturity=1.0,
                                notional_foreign=1_000_000.0),
        quantity=1.0, entry_price=0.0, underlying="EURUSD", engine=GarmanKohlhagenEngine(),
    )
    return pf


def _spot_only_history(days=300, sigma=0.008, seed=7):
    """Levels DataFrame where only EURUSD spot moves (vol/rates flat)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, sigma, days)
    spot = 1.20 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2025-01-01", periods=days, freq="B")
    return pd.DataFrame({
        "EURUSD_spot": spot,
        "EURUSD_vol": np.full(days, 0.10),
        "EURUSD_dom_rate": np.full(days, 0.05),
        "EURUSD_for_rate": np.full(days, 0.03),
    }, index=idx)


def test_fx_parametric_var_matches_closed_form_for_linear_forward():
    from quantark.var import FXParametricVaREngine

    df = _spot_only_history()
    config = VaRConfig(confidence_level=0.99, lookback_days=250)
    engine = FXParametricVaREngine(config)
    result = engine.calculate_var(_forward_portfolio(), df)

    # For a pure spot-linear forward with only spot varying, VaR ~= z*|s_spot|*sigma.
    from scipy import stats
    z = stats.norm.ppf(0.99)
    spot_returns = df["EURUSD_spot"].pct_change().dropna().tail(250)
    sigma = spot_returns.std()
    # dollar spot sensitivity ~ notional in domestic (forward delta ~ df_for*N*spot)
    assert result.var > 0
    # closed form within 10%
    s_spot = result.config_summary["sensitivities"]["EURUSD_spot_return"]
    assert result.var == pytest.approx(z * abs(s_spot) * sigma, rel=0.10)


def test_fx_parametric_var_scales_with_sqrt_holding_period():
    from quantark.var import FXParametricVaREngine

    df = _spot_only_history()
    base = FXParametricVaREngine(VaRConfig(confidence_level=0.99, lookback_days=250,
                                           holding_period=1)).calculate_var(
        _forward_portfolio(), df)
    tenday = FXParametricVaREngine(VaRConfig(confidence_level=0.99, lookback_days=250,
                                             holding_period=10)).calculate_var(
        _forward_portfolio(), df)
    assert tenday.var == pytest.approx(base.var * np.sqrt(10), rel=1e-6)


def test_fx_historical_var_positive_and_cvar_ge_var():
    from quantark.var import FXHistoricalVaREngine

    df = _spot_only_history()
    result = FXHistoricalVaREngine(
        VaRConfig(confidence_level=0.99, lookback_days=250)
    ).calculate_var(_option_portfolio(), df)
    assert result.var > 0
    assert result.cvar >= result.var


def test_fx_monte_carlo_var_close_to_parametric_for_linear_book():
    from quantark.var import FXMonteCarloVaREngine, FXParametricVaREngine

    df = _spot_only_history()
    config = VaRConfig(confidence_level=0.99, lookback_days=250,
                       mc_num_simulations=20_000, mc_seed=42)
    par = FXParametricVaREngine(config).calculate_var(_forward_portfolio(), df)
    mc = FXMonteCarloVaREngine(config).calculate_var(_forward_portfolio(), df)
    assert mc.var == pytest.approx(par.var, rel=0.15)


def test_fx_parametric_factor_attribution_sums_to_var():
    from quantark.var import FXParametricVaREngine

    df = _spot_only_history()
    config = VaRConfig(confidence_level=0.99, lookback_days=250,
                       calculate_factor_var=True)
    result = FXParametricVaREngine(config).calculate_var(_option_portfolio(), df)
    assert result.factor_var is not None
    assert sum(result.factor_var.values()) == pytest.approx(result.var, rel=1e-6)


def test_fx_var_supports_fx_portfolio_only():
    from quantark.var import FXParametricVaREngine

    engine = FXParametricVaREngine()
    assert engine.supports_portfolio(_forward_portfolio()) is True
    assert engine.supports_portfolio(object()) is False
