import numpy as np
import pytest

from quantark.param.vol import SABRVolSurface
from quantark.param.vol.sabr.diagnostics import check_arbitrage, butterfly_density
from quantark.util.exceptions import ValidationError


def _benign():
    return SABRVolSurface.from_params(alpha=0.2, beta=0.5, rho=-0.3, nu=0.3, maturity=1.0)


def test_benign_sabr_slice_is_butterfly_arbitrage_free():
    s = _benign()
    strikes = np.linspace(70, 130, 25)
    g = butterfly_density(s, T=1.0, strikes=strikes, spot=100.0)
    assert np.all(g >= -1e-8)


def test_check_arbitrage_reports_ok_for_benign_surface():
    s = _benign()
    report = check_arbitrage(s, strikes=np.linspace(70, 130, 25),
                             maturities=[1.0], spot=100.0)
    assert report.butterfly_ok is True
    assert report.calendar_ok is True


def test_extreme_volofvol_flags_butterfly_arbitrage():
    # High vol-of-vol + steep skew induces a negative-density region in Hagan.
    bad = SABRVolSurface.from_params(alpha=0.3, beta=0.5, rho=-0.9, nu=4.0, maturity=1.0)
    report = check_arbitrage(bad, strikes=np.linspace(60, 160, 80), maturities=[1.0], spot=100.0)
    assert report.butterfly_ok is False
    assert report.min_density < 0.0


def test_construct_with_check_arbitrage_raises_on_bad_slice():
    with pytest.raises(ValidationError):
        SABRVolSurface(
            slices={1.0: {"alpha": 0.3, "beta": 0.5, "rho": -0.9, "nu": 4.0}},
            check_arbitrage=True,
        )
