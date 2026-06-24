import warnings
import pytest
from quantark.param.vol import FlatVolSurface, SABRVolSurface
from quantark.param.vol.collapse_guard import SmileCollapseWarning, guard_constant_vol
from quantark.util.exceptions import PricingError


def _sabr():
    return SABRVolSurface.from_params(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4, maturity=1.0)


def test_flat_surface_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        guard_constant_vol(FlatVolSurface(volatility=0.2), "EuropeanMCEngine")


def test_smile_surface_warns_by_default():
    with pytest.warns(SmileCollapseWarning):
        guard_constant_vol(_sabr(), "EuropeanMCEngine")


def test_smile_surface_raises_in_strict_mode():
    with pytest.raises(PricingError):
        guard_constant_vol(_sabr(), "EuropeanMCEngine", strict=True)


def test_euro_mc_warns_on_smile_surface():
    from datetime import datetime
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.asset.equity.engine.mc import EuropeanMCEngine
    from quantark.asset.equity.param import MCParams
    from quantark.param import SpotQuote, FlatRateCurve
    from quantark.priceenv import PricingEnvironment
    from quantark.util.enum import OptionType

    env = PricingEnvironment(
        rate_curve=FlatRateCurve(rate=0.02),
        valuation_date=datetime(2026, 6, 24),
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=_sabr(),
    )
    opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    eng = EuropeanMCEngine(params=MCParams(num_paths=2000, time_steps=10))
    with pytest.warns(SmileCollapseWarning):
        eng.price(opt, env)
