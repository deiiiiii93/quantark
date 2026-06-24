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
