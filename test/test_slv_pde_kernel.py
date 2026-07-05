import numpy as np
import pytest
from quantark.param import GridVolSurface, FlatRateCurve
from quantark.util.enum.engine_enums import ADIScheme, LeverageCalibrationMethod
from quantark.util.exceptions import ValidationError, NumericalError
from quantark.volmodels.black_scholes import bs_call_price
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv import (
    LeverageSurface, calibrate_leverage_surface, price_european_slv_pde,
)


def _flat_lv(vol=0.2):
    strikes = list(100.0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    mats = list(np.linspace(0.1, 2.0, 6))
    return build_dupire_local_vol(GridVolSurface(strikes, mats, np.full((6, 9), vol)),
                                  spot=100.0, rate_curve=FlatRateCurve(0.0), div_yield=lambda t: 0.0)


def _const(T, M, r, q):
    return np.full(M, T / M), np.full(M, r), np.full(M, q)


# Feller-satisfied so the MC variance mean ~ theta and leverage ~ 1 (cleaner cross-check).
P = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.2, rho=-0.5)


def test_slv_pde_with_unit_leverage_matches_bs():
    # Leverage identically 1 + flat local vol => SLV PDE reduces to Heston PDE with the
    # local-vol leg = constant; with v0=theta and the variance-vol scaling it reprices BS-ish.
    K = np.array([1.0, 1e6]); Tg = np.array([0.0, 100.0])
    lev = LeverageSurface(Tg, K, np.full((2, 2), 1.0))  # L == 1 everywhere
    # With L=1, eta=0 (frozen v=v0=0.04) the local vol is sqrt(v0)=0.2 -> BS(0.2).
    price = price_european_slv_pde(100.0, 100.0, True, 1.0, P, lev, r=0.03, carry=0.01,
                                   eta=0.0, n_x=200, n_v=80, n_t=80)
    assert price == pytest.approx(bs_call_price(100.0, 100.0, 1.0, 0.2, 0.03, 0.01), abs=0.1)


def test_slv_pde_consumes_mc_calibrated_leverage_reprices_vanilla():
    # End-to-end: calibrate leverage by MC on a flat LV surface, then price by backward PDE.
    lv = _flat_lv(0.2)
    dt, rf, cf = _const(1.0, 60, 0.0, 0.0)
    lev = calibrate_leverage_surface(100.0, P, lv, dt, rf, cf, eta=1.0,
                                     method=LeverageCalibrationMethod.MC_BINNING,
                                     num_paths=60_000, num_bins=20, seed=5)
    price = price_european_slv_pde(100.0, 100.0, True, 1.0, P, lev, r=0.0, carry=0.0,
                                   eta=1.0, n_x=220, n_v=90, n_t=100)
    # SLV calibrated to a flat surface must reprice the flat-vol vanilla (within grid+calib error).
    assert price == pytest.approx(bs_call_price(100.0, 100.0, 1.0, 0.2, 0.0, 0.0), abs=0.4)


def test_slv_pde_validation():
    K = np.array([50.0, 200.0]); Tg = np.array([0.0, 2.0])
    lev = LeverageSurface(Tg, K, np.full((2, 2), 1.0))
    with pytest.raises(ValidationError):
        price_european_slv_pde(100.0, 100.0, True, 1.0, P, lev, 0.0, 0.0, n_x=2)
    with pytest.raises(ValidationError):
        price_european_slv_pde(100.0, 100.0, True, 1.0, P, lev, 0.0, 0.0, scheme=ADIScheme.MCS)


def test_slv_adi_price_regression_pinned_for_batch_swap():
    # Captured from the pre-batching implementation (per-system Python Thomas), WS-B2.
    # The batched sweep preserves per-system arithmetic order -> 1e-13 relative.
    params = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.5, rho=-0.7)
    lev = LeverageSurface(time_grid=np.array([0.0, 0.5, 1.0]),
                          strike_grid=np.array([60.0, 100.0, 160.0]),
                          leverage_grid=np.array([[1.3, 1.0, 0.9],
                                                  [1.25, 1.0, 0.92],
                                                  [1.2, 1.0, 0.95]]))
    p = price_european_slv_pde(100.0, 100.0, True, 1.0, params, lev, 0.03, 0.01,
                               n_x=120, n_v=60, n_t=60)
    assert np.isclose(p, 9.008727342033493, rtol=1e-13)
