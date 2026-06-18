"""
Tests for the SABR Hagan smile, calibration, and SABRVolSurface adapter.
"""

import numpy as np
import pytest

from quantark.param.vol import SABRVolSurface
from quantark.param.vol.sabr import (
    calibrate_sabr_slice,
    sabr_atm_implied_vol_black,
    sabr_generate_vol_surface,
    sabr_implied_vol_black,
    sabr_implied_vol_black_shifted,
)
from quantark.util.exceptions import ValidationError

# Reference SABR parameters used across tests.
F0 = 100.0
T0 = 1.0
ALPHA, BETA, RHO, NU = 0.20, 0.5, -0.3, 0.4


def _reference_hagan(F, K, T, alpha, beta, rho, nu):
    """Independent textbook Hagan (2002) lognormal IV, for cross-checking."""
    import math

    one_mb = 1.0 - beta
    if abs(F - K) < 1e-12:
        fb = F ** one_mb
        A = 1.0 + (
            (one_mb**2) * alpha**2 / (24.0 * F ** (2.0 * one_mb))
            + rho * beta * nu * alpha / (4.0 * fb)
            + (2.0 - 3.0 * rho**2) * nu**2 / 24.0
        ) * T
        return alpha / fb * A
    log_fk = math.log(F / K)
    fk_b = (F * K) ** (0.5 * one_mb)
    z = (nu / alpha) * fk_b * log_fk
    chi = math.log((math.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
    denom = fk_b * (
        1.0 + (one_mb**2) * log_fk**2 / 24.0 + (one_mb**4) * log_fk**4 / 1920.0
    )
    A = 1.0 + (
        (one_mb**2) * alpha**2 / (24.0 * (F * K) ** one_mb)
        + rho * beta * nu * alpha / (4.0 * fk_b)
        + (2.0 - 3.0 * rho**2) * nu**2 / 24.0
    ) * T
    return alpha / denom * (z / chi) * A


def test_matches_textbook_hagan_across_strikes():
    """Corrected formula must agree with an independent textbook Hagan impl."""
    strikes = [60.0, 80.0, 95.0, 110.0, 130.0]
    for K in strikes:
        got = float(sabr_implied_vol_black(F0, K, T0, ALPHA, BETA, RHO, NU))
        ref = _reference_hagan(F0, K, T0, ALPHA, BETA, RHO, NU)
        assert got == pytest.approx(ref, rel=1e-9)


def test_beta_one_does_not_crash_and_matches_reference():
    """beta=1 (lognormal SABR) previously divided by zero; must now work."""
    strikes = [80.0, 100.0, 120.0]
    for K in strikes:
        got = float(sabr_implied_vol_black(F0, K, T0, 0.20, 1.0, RHO, NU))
        assert np.isfinite(got) and got > 0.0
        ref = _reference_hagan(F0, K, T0, 0.20, 1.0, RHO, NU)
        assert got == pytest.approx(ref, rel=1e-9)


def test_atm_general_formula_matches_atm_specialization():
    """At K=F the general-strike formula must equal the ATM specialization."""
    general = float(sabr_implied_vol_black(F0, F0, T0, ALPHA, BETA, RHO, NU))
    atm = float(sabr_atm_implied_vol_black(F0, T0, ALPHA, BETA, RHO, NU))
    assert general == pytest.approx(atm, rel=1e-10)


def test_implied_vol_is_positive_across_strikes():
    strikes = np.linspace(70.0, 140.0, 15)
    vols = sabr_implied_vol_black(F0, strikes, T0, ALPHA, BETA, RHO, NU)
    assert np.all(vols > 0.0)
    assert np.all(np.isfinite(vols))


def test_negative_correlation_produces_downward_skew():
    """rho < 0 should give higher vols for low strikes than high strikes."""
    low = float(sabr_implied_vol_black(F0, 80.0, T0, ALPHA, BETA, RHO, NU))
    high = float(sabr_implied_vol_black(F0, 120.0, T0, ALPHA, BETA, RHO, NU))
    assert low > high


def test_shifted_sabr_wrapper_matches_shift_kwarg():
    shift = 5.0
    a = float(sabr_implied_vol_black_shifted(F0, 90.0, T0, ALPHA, BETA, RHO, NU, shift))
    b = float(sabr_implied_vol_black(F0, 90.0, T0, ALPHA, BETA, RHO, NU, shift=shift))
    assert a == pytest.approx(b, rel=1e-12)


def test_shifted_sabr_handles_low_strikes():
    """With a shift, strikes near/below zero forward remain finite."""
    vols = sabr_implied_vol_black(
        10.0, np.array([1.0, 5.0, 10.0, 20.0]), T0, ALPHA, BETA, RHO, NU, shift=5.0
    )
    assert np.all(np.isfinite(vols))
    assert np.all(vols > 0.0)


def test_calibration_round_trip_recovers_smile():
    """Calibrate against a SABR-generated smile and recover a low-error fit."""
    strikes = np.linspace(80.0, 120.0, 11)
    true_vols = sabr_implied_vol_black(F0, strikes, T0, ALPHA, BETA, RHO, NU)

    res = calibrate_sabr_slice(F0, strikes, T0, true_vols, beta=BETA)

    fitted = sabr_implied_vol_black(
        F0, strikes, T0, res["alpha"], res["beta"], res["rho"], res["nu"]
    )
    # The fit should reproduce the input smile to within a tight vol tolerance.
    assert np.max(np.abs(fitted - true_vols)) < 5e-3
    assert res["mse"] < 1e-5
    assert res["beta"] == BETA


def test_calibration_rejects_mismatched_lengths():
    with pytest.raises(ValidationError):
        calibrate_sabr_slice(F0, [90.0, 100.0], T0, [0.2])


def test_calibration_rejects_all_nonfinite_vols():
    with pytest.raises(ValidationError):
        calibrate_sabr_slice(F0, [90.0, 100.0, 110.0], T0, [np.nan, np.nan, np.nan])


def test_calibration_tolerates_partial_nonfinite_vols():
    """A single NaN quote must not poison the fit; finite quotes still calibrate."""
    strikes = np.linspace(80.0, 120.0, 11)
    true_vols = sabr_implied_vol_black(F0, strikes, T0, ALPHA, BETA, RHO, NU)
    polluted = true_vols.copy()
    polluted[3] = np.nan  # one bad quote

    res = calibrate_sabr_slice(F0, strikes, T0, polluted, beta=BETA)
    assert np.isfinite(res["mse"])
    assert res["mse"] < 1e-4  # not the inf/default-rho degenerate result
    assert -1.0 <= res["rho"] <= 1.0


def test_generate_surface_keys_and_values():
    strikes = [90.0, 100.0, 110.0]
    mats = [0.5, 1.0]
    surf = sabr_generate_vol_surface(F0, strikes, mats, ALPHA, BETA, RHO, NU)
    assert len(surf) == len(strikes) * len(mats)
    assert all(v > 0.0 for v in surf.values())


def test_sabr_vol_surface_matches_hagan_at_pillar():
    surface = SABRVolSurface.from_params(
        ALPHA, BETA, RHO, NU, maturity=T0, forward=F0
    )
    got = surface.get_vol(strike=95.0, time_to_maturity=T0, spot=F0)
    expected = float(sabr_implied_vol_black(F0, 95.0, T0, ALPHA, BETA, RHO, NU))
    assert got == pytest.approx(expected, rel=1e-12)


def test_sabr_vol_surface_forward_callable():
    """A spot->forward callable is honored when converting spot to forward."""
    surface = SABRVolSurface.from_params(
        ALPHA, BETA, RHO, NU, maturity=T0, forward=lambda s, t: s * 1.02
    )
    got = surface.get_vol(strike=100.0, time_to_maturity=T0, spot=100.0)
    expected = float(sabr_implied_vol_black(102.0, 100.0, T0, ALPHA, BETA, RHO, NU))
    assert got == pytest.approx(expected, rel=1e-12)


def test_sabr_vol_surface_interpolates_between_pillars():
    surface = SABRVolSurface(
        slices={
            0.5: {"alpha": 0.20, "beta": BETA, "rho": -0.4, "nu": 0.5},
            2.0: {"alpha": 0.25, "beta": BETA, "rho": -0.2, "nu": 0.3},
        },
        forward=F0,
    )
    # Parameters interpolate linearly in maturity; vol at T=1.25 (midpoint) uses
    # interpolated (alpha, rho, nu).
    mid = surface.get_vol(strike=100.0, time_to_maturity=1.25, spot=F0)
    alpha_mid = np.interp(1.25, [0.5, 2.0], [0.20, 0.25])
    rho_mid = np.interp(1.25, [0.5, 2.0], [-0.4, -0.2])
    nu_mid = np.interp(1.25, [0.5, 2.0], [0.5, 0.3])
    expected = float(
        sabr_implied_vol_black(F0, 100.0, 1.25, alpha_mid, BETA, rho_mid, nu_mid)
    )
    assert mid == pytest.approx(expected, rel=1e-12)


def test_sabr_vol_surface_validates_params():
    with pytest.raises(ValidationError):
        SABRVolSurface(slices={1.0: {"alpha": 0.2, "beta": BETA, "rho": 2.0, "nu": 0.4}})
    with pytest.raises(ValidationError):
        SABRVolSurface(slices={})
