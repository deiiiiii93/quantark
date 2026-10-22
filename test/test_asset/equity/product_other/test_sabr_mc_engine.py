from datetime import datetime

import pytest

from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.mc import SABRMCEngine
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.param import MCParams
from quantark.param import SpotQuote, FlatRateCurve
from quantark.param.vol import SABRVolSurface, FlatVolSurface
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import SABRMCScheme
from quantark.util.exceptions import MarketDataError


def _env(surface, r=0.0):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(rate=r),
        valuation_date=datetime(2026, 6, 24),
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=surface,
    )


@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
def test_sabr_mc_logeuler_matches_hagan_analytical(K):
    # r=0 so spot==forward; the analytical path is BS with the SABR smile vol.
    sabr = SABRVolSurface.from_params(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4, maturity=1.0)
    env = _env(sabr, r=0.0)
    opt = EuropeanVanillaOption(strike=K, option_type=OptionType.CALL, maturity=1.0)
    analytic = BlackScholesEngine().price(opt, env)
    mc = SABRMCEngine(params=MCParams(num_paths=400_000, time_steps=100, seed=11)).price(opt, env)
    assert abs(mc - analytic) < 0.25  # within MC error


@pytest.mark.parametrize("K", [90.0, 100.0, 110.0])
def test_sabr_mc_quadexp_beta1_matches_hagan_at_coarse_grid(K):
    # beta=1 QUADEXP is exact conditional lognormal: matches Hagan with FEW steps.
    sabr = SABRVolSurface.from_params(alpha=0.2, beta=1.0, rho=-0.4, nu=0.5, maturity=1.0)
    env = _env(sabr, r=0.0)
    opt = EuropeanVanillaOption(strike=K, option_type=OptionType.CALL, maturity=1.0)
    analytic = BlackScholesEngine().price(opt, env)
    mc = SABRMCEngine(
        params=MCParams(num_paths=400_000, time_steps=8, seed=5),  # only 8 steps
        scheme=SABRMCScheme.QUADEXP,
    ).price(opt, env)
    assert abs(mc - analytic) < 0.3


def test_sabr_mc_rejects_non_sabr_surface():
    env = _env(FlatVolSurface(volatility=0.2))
    opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    with pytest.raises(MarketDataError):
        SABRMCEngine().price(opt, env)
