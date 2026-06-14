import numpy as np
import pytest
from datetime import datetime
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment, FxPricingEnvironment
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.analytical import HestonAnalyticalEngine
from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
from quantark.asset.fx.engine.analytical import FxHestonAnalyticalEngine
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import EngineType, HestonAnalyticalMethod
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.heston.analytical_kernel import heston_call_price


PARAMS = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)


def _eq_env(r=0.03, q=0.01):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(r), valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(0.2),
        div_yield=ContinuousDividendYield(q),
    )


def test_equity_heston_engine_matches_kernel():
    env = _eq_env(0.03, 0.01)
    opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    eng = HestonAnalyticalEngine(PARAMS, method=EngineType.ANALYTICAL(HestonAnalyticalMethod.GATHERAL))
    expected = heston_call_price(100.0, 100.0, 1.0, PARAMS, 0.03, 0.01)
    assert eng.price(opt, env) == pytest.approx(expected, abs=1e-9)


def test_equity_heston_contract_multiplier_scales():
    env = _eq_env()
    opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0,
                                contract_multiplier=1000.0)
    eng = HestonAnalyticalEngine(PARAMS)
    assert eng.price(opt, env) == pytest.approx(
        1000.0 * heston_call_price(100.0, 100.0, 1.0, PARAMS, 0.03, 0.01), abs=1e-6)


def test_equity_heston_greeks_no_vega():
    env = _eq_env()
    opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    g = HestonAnalyticalEngine(PARAMS).calculate_greeks(opt, env)
    assert set(["price", "delta", "gamma", "theta", "rho"]).issubset(g.keys())
    assert "vega" not in g
    assert 0.0 < g["delta"] < 1.0 + 1e-9


def _fx_env(rd=0.03, rf=0.01):
    return FxPricingEnvironment(
        valuation_date=datetime(2026, 1, 1), spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rd), foreign_curve=FlatRateCurve(rf),
        vol_surface=FlatVolSurface(0.1),
    )


def test_fx_heston_matches_equity_when_carry_equal():
    # With q == r_f and unit notional/participation/annualization, FX == equity kernel value.
    eq_unit = heston_call_price(1.20, 1.20, 1.0, PARAMS, 0.03, 0.01)
    env = _fx_env(0.03, 0.01)
    opt = FxVanillaOption(strike=1.20, option_type=OptionType.CALL, maturity=1.0,
                          notional_foreign=1.0)
    fx = FxHestonAnalyticalEngine(PARAMS).price(opt, env)
    assert fx == pytest.approx(eq_unit, abs=1e-9)


def test_fx_heston_sizing_and_restrictions():
    env = _fx_env()
    opt = FxVanillaOption(strike=1.20, option_type=OptionType.CALL, maturity=1.0,
                          notional_foreign=1_000_000.0)
    base = FxHestonAnalyticalEngine(PARAMS).price(opt, env)
    assert base == pytest.approx(
        1_000_000.0 * heston_call_price(1.20, 1.20, 1.0, PARAMS, 0.03, 0.01), rel=1e-9)
    env.market_forward = 1.25
    from quantark.util.exceptions import PricingError
    with pytest.raises(PricingError):
        FxHestonAnalyticalEngine(PARAMS).price(opt, env)


def test_fx_heston_greeks_no_vega():
    env = _fx_env()
    opt = FxVanillaOption(strike=1.20, option_type=OptionType.CALL, maturity=1.0,
                          notional_foreign=1_000_000.0)
    g = FxHestonAnalyticalEngine(PARAMS).calculate_greeks(opt, env)
    assert set(["price", "delta", "gamma", "rho_dom", "rho_for"]).issubset(g.keys())
    assert "vega" not in g
