import numpy as np
import pytest
from quantark.param import GridVolSurface, FlatRateCurve
from quantark.util.exceptions import ValidationError, NumericalError
from quantark.volmodels.localvol.surface import LocalVolSurface
from quantark.volmodels.localvol.dupire import build_dupire_local_vol


def test_local_vol_surface_exact_bilinear_and_clamp():
    K = np.array([80.0, 100.0, 120.0])
    T = np.array([0.5, 1.0])
    lv = np.array([[0.18, 0.20, 0.22], [0.19, 0.21, 0.23]])  # (nT, nK)
    surf = LocalVolSurface(strike_grid=K, time_grid=T, lv_grid=lv)
    assert surf.local_vol(100.0, 0.5) == pytest.approx(0.20, abs=1e-12)
    # exact bilinear at S=90 (between 80,100), t=0.75 (between 0.5,1.0):
    # row T=0.5: interp(90)= (0.18+0.20)/2 = 0.19 ; row T=1.0: (0.19+0.21)/2 = 0.20
    # time interp at 0.75: (0.19+0.20)/2 = 0.195
    assert surf.local_vol(90.0, 0.75) == pytest.approx(0.195, abs=1e-12)
    assert surf.local_vol(40.0, 0.5) == pytest.approx(0.18, abs=1e-12)
    assert surf.local_vol(200.0, 1.0) == pytest.approx(0.23, abs=1e-12)


def test_local_vol_surface_rejects_nonpositive():
    with pytest.raises(ValidationError):
        LocalVolSurface(np.array([80.0, 100.0]), np.array([0.5, 1.0]),
                        np.array([[0.2, 0.0], [0.2, 0.2]]))


def _flat_iv_surface(n_strike=7, n_mat=5, vol=0.20):
    strikes = list(np.round(100.0 * np.exp(np.linspace(-0.6, 0.6, n_strike)), 4))
    maturities = list(np.linspace(0.2, 2.0, n_mat))
    return GridVolSurface(strikes, maturities, np.full((n_mat, n_strike), vol))


def _max_abs_error(surf_lv, target):
    err = 0.0
    for t in surf_lv.time_grid:
        for k in surf_lv.strike_grid:
            err = max(err, abs(float(surf_lv.local_vol(k, t)) - target))
    return err


def test_flat_iv_round_trip_is_exact_zero_rates():
    lv = build_dupire_local_vol(
        _flat_iv_surface(7, 5, 0.20), spot=100.0,
        rate_curve=FlatRateCurve(0.0), div_yield=lambda t: 0.0,
    )
    assert _max_abs_error(lv, 0.20) < 1e-9


def test_flat_iv_round_trip_is_exact_with_carry():
    lv = build_dupire_local_vol(
        _flat_iv_surface(9, 6, 0.25), spot=100.0,
        rate_curve=FlatRateCurve(0.03), div_yield=lambda t: 0.01,
    )
    assert _max_abs_error(lv, 0.25) < 1e-9


def test_dupire_rejects_calendar_arbitrage_by_default():
    strikes = [80.0, 100.0, 120.0]
    maturities = [0.5, 1.0, 1.5]
    iv = np.array([[0.30, 0.30, 0.30], [0.20, 0.20, 0.20], [0.10, 0.10, 0.10]])
    surf = GridVolSurface(strikes, maturities, iv)
    with pytest.raises(NumericalError):
        build_dupire_local_vol(surf, spot=100.0, rate_curve=FlatRateCurve(0.0), div_yield=lambda t: 0.0)


def test_dupire_requires_three_nodes_each_axis():
    surf = GridVolSurface([90.0, 100.0], [0.5, 1.0], np.full((2, 2), 0.2))
    with pytest.raises(ValidationError):
        build_dupire_local_vol(surf, spot=100.0, rate_curve=FlatRateCurve(0.0), div_yield=lambda t: 0.0)


def test_floor_is_opt_in():
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    maturities = [0.3, 0.6, 1.0]
    iv = np.full((3, 5), 0.2)
    iv[1, 2] = 0.6  # spike at one interior node
    surf = GridVolSurface(strikes, maturities, iv)
    with pytest.raises(NumericalError):
        build_dupire_local_vol(surf, spot=100.0, rate_curve=FlatRateCurve(0.0), div_yield=lambda t: 0.0)
    lv = build_dupire_local_vol(
        surf, spot=100.0, rate_curve=FlatRateCurve(0.0), div_yield=lambda t: 0.0,
        validate_arbitrage=False, vol_floor=0.05,
    )
    assert np.all(lv.lv_grid >= 0.05 - 1e-12)
