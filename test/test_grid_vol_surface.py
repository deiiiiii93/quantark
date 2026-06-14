import numpy as np
import pytest
from quantark.param import GridVolSurface
from quantark.util.exceptions import ValidationError


def _flat_surface(vol=0.2):
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    maturities = [0.25, 0.5, 1.0, 2.0]
    return GridVolSurface(strikes, maturities, np.full((len(maturities), len(strikes)), vol))


def test_flat_grid_returns_constant():
    assert _flat_surface(0.2).get_vol(105.0, 0.75, spot=100.0) == pytest.approx(0.2, abs=1e-12)


def test_total_variance_interpolation_in_time():
    strikes = [90.0, 110.0]
    maturities = [0.5, 1.0]
    iv = np.array([[0.2, 0.2], [0.3, 0.3]])
    surf = GridVolSurface(strikes, maturities, iv)
    w075 = np.interp(0.75, [0.5, 1.0], [0.2**2 * 0.5, 0.3**2 * 1.0])
    assert surf.get_vol(100.0, 0.75, spot=100.0) == pytest.approx(np.sqrt(w075 / 0.75), abs=1e-9)


def test_flat_extrapolation_in_strike():
    surf = _flat_surface(0.2)
    assert surf.get_vol(60.0, 0.5, spot=100.0) == pytest.approx(0.2, abs=1e-12)
    assert surf.get_vol(200.0, 0.5, spot=100.0) == pytest.approx(0.2, abs=1e-12)


def test_validation_rejects_shape_mismatch():
    with pytest.raises(ValidationError):
        GridVolSurface([90.0, 100.0], [0.5, 1.0], np.zeros((2, 3)))


def test_validation_rejects_nonincreasing_axes():
    with pytest.raises(ValidationError):
        GridVolSurface([100.0, 90.0], [0.5, 1.0], np.full((2, 2), 0.2))


def test_validation_rejects_nonpositive_vol():
    with pytest.raises(ValidationError):
        GridVolSurface([90.0, 100.0], [0.5, 1.0], np.array([[0.2, 0.0], [0.2, 0.2]]))
