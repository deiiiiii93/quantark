"""
Tests for the param-layer Vanna-Volga FX smile construction and surface.
"""

import math

import numpy as np
import pytest

from quantark.param.vol import VannaVolgaVolSurface
from quantark.param.vol.vannavolga import (
    DeltaConvention,
    FXEnv,
    GKInput,
    SmileQuotes,
    compute_omega,
    greeks_gk,
    price_gk,
    strikes_25d,
)
from quantark.util.exceptions import ValidationError

# Standard EURUSD-like case.
ENV = FXEnv(spot=1.20, rd=0.02, rf=0.01, tau=1.0)
QUOTES = SmileQuotes(sigma_atm=0.10, rr25=-0.01, bf25_2vol=0.0025)


def test_put_call_parity():
    """GK call - put = DF_for*S - DF_dom*K."""
    x = GKInput(ENV.spot, 1.25, ENV.rd, ENV.rf, 0.10, ENV.tau)
    call = price_gk(True, x)
    put = price_gk(False, x)
    parity = ENV.df_for * ENV.spot - ENV.df_dom * 1.25
    assert (call - put) == pytest.approx(parity, abs=1e-12)


def test_vega_vanna_volga_match_finite_difference():
    strike, sigma = 1.22, 0.10
    base = greeks_gk(True, GKInput(ENV.spot, strike, ENV.rd, ENV.rf, sigma, ENV.tau))

    h = 1e-5
    p_up = price_gk(True, GKInput(ENV.spot, strike, ENV.rd, ENV.rf, sigma + h, ENV.tau))
    p_dn = price_gk(True, GKInput(ENV.spot, strike, ENV.rd, ENV.rf, sigma - h, ENV.tau))
    vega_fd = (p_up - p_dn) / (2 * h)
    assert base["vega"] == pytest.approx(vega_fd, rel=1e-4)

    # Vanna = d(vega)/dS
    def vega_at_spot(s):
        return greeks_gk(True, GKInput(s, strike, ENV.rd, ENV.rf, sigma, ENV.tau))["vega"]

    hs = 1e-5
    vanna_fd = (vega_at_spot(ENV.spot + hs) - vega_at_spot(ENV.spot - hs)) / (2 * hs)
    assert base["vanna"] == pytest.approx(vanna_fd, rel=1e-4)

    # Volga = d(vega)/dsigma
    def vega_at_vol(v):
        return greeks_gk(True, GKInput(ENV.spot, strike, ENV.rd, ENV.rf, v, ENV.tau))["vega"]

    volga_fd = (vega_at_vol(sigma + h) - vega_at_vol(sigma - h)) / (2 * h)
    assert base["volga"] == pytest.approx(volga_fd, rel=1e-3)


def test_smile_quotes_rr_bf_identities():
    sigma_25p, sigma_25c = QUOTES.sigma_25d()
    assert (sigma_25c - sigma_25p) == pytest.approx(QUOTES.rr25, abs=1e-15)
    assert (0.5 * (sigma_25c + sigma_25p) - QUOTES.sigma_atm) == pytest.approx(
        QUOTES.bf25_2vol, abs=1e-15
    )


def test_strikes_25d_have_target_deltas():
    conv = DeltaConvention.SPOT
    kp, kc = strikes_25d(QUOTES.sigma_atm, ENV, conv)
    from quantark.param.vol.vannavolga import bs_delta

    assert bs_delta(kp, False, ENV, QUOTES.sigma_atm, conv) == pytest.approx(-0.25, abs=1e-6)
    assert bs_delta(kc, True, ENV, QUOTES.sigma_atm, conv) == pytest.approx(0.25, abs=1e-6)


def test_compute_omega_finite_and_matrix_invertible():
    omega, A = compute_omega(ENV, QUOTES, DeltaConvention.SPOT)
    assert omega.shape == (3,)
    assert np.all(np.isfinite(omega))
    assert abs(np.linalg.det(A)) > 1e-12


def test_compute_omega_satisfies_calibration_condition():
    """Omega must satisfy A @ Omega = I (instruments-as-rows orientation)."""
    from quantark.param.vol.vannavolga import strike_for_delta, rr_bf_costs

    omega, A = compute_omega(ENV, QUOTES, DeltaConvention.SPOT)
    sigma_25p, sigma_25c = QUOTES.sigma_25d()
    # Wing strikes solved with their own quoted vols (as compute_omega does).
    kp = strike_for_delta(-0.25, False, ENV, sigma_25p, DeltaConvention.SPOT)
    kc = strike_for_delta(+0.25, True, ENV, sigma_25c, DeltaConvention.SPOT)
    rr_cost, bf_cost = rr_bf_costs(ENV, QUOTES.sigma_atm, kc, kp, sigma_25c, sigma_25p)
    I = np.array([0.0, rr_cost, bf_cost])
    np.testing.assert_allclose(A @ omega, I, atol=1e-12)


def test_vv_surface_reprices_25delta_wing_vols():
    """VV surface must reproduce the quoted wing vols at the true 25d strikes."""
    from quantark.param.vol.vannavolga import strike_for_delta

    surf = VannaVolgaVolSurface(ENV, QUOTES, DeltaConvention.SPOT)
    sigma_25p, sigma_25c = QUOTES.sigma_25d()
    kp = strike_for_delta(-0.25, False, ENV, sigma_25p, DeltaConvention.SPOT)
    kc = strike_for_delta(+0.25, True, ENV, sigma_25c, DeltaConvention.SPOT)
    # Second-order VV reprices the wings to within a few basis points.
    assert surf.get_vol(kp, ENV.tau, ENV.spot) == pytest.approx(sigma_25p, abs=2e-3)
    assert surf.get_vol(kc, ENV.tau, ENV.spot) == pytest.approx(sigma_25c, abs=2e-3)


def test_strikes_25d_premium_adjusted_convention():
    """Premium-adjusted call delta is hump-shaped; solver must still bracket."""
    kp, kc = strikes_25d(QUOTES.sigma_atm, ENV, DeltaConvention.SPOT_PREM)
    from quantark.param.vol.vannavolga import bs_delta

    assert bs_delta(kp, False, ENV, QUOTES.sigma_atm, DeltaConvention.SPOT_PREM) == pytest.approx(
        -0.25, abs=1e-6
    )
    assert bs_delta(kc, True, ENV, QUOTES.sigma_atm, DeltaConvention.SPOT_PREM) == pytest.approx(
        0.25, abs=1e-6
    )
    assert kp < ENV.forward < kc  # OTM put below, OTM call above forward


def test_rho_signs_match_finite_difference():
    strike, sigma = 1.22, 0.10
    g = greeks_gk(True, GKInput(ENV.spot, strike, ENV.rd, ENV.rf, sigma, ENV.tau))
    # Call: domestic rho > 0, foreign rho < 0.
    assert g["rho_dom"] > 0.0
    assert g["rho_for"] < 0.0
    h = 1e-6
    up = price_gk(True, GKInput(ENV.spot, strike, ENV.rd + h, ENV.rf, sigma, ENV.tau))
    dn = price_gk(True, GKInput(ENV.spot, strike, ENV.rd - h, ENV.rf, sigma, ENV.tau))
    assert g["rho_dom"] == pytest.approx((up - dn) / (2 * h), rel=1e-4)


def test_zero_vol_gk_returns_intrinsic_not_negative():
    """OTM zero-vol call must be 0, ITM call its discounted forward intrinsic."""
    otm = price_gk(True, GKInput(ENV.spot, 2.0, ENV.rd, ENV.rf, 0.0, ENV.tau))
    assert otm == pytest.approx(0.0, abs=1e-12)
    itm = price_gk(True, GKInput(ENV.spot, 1.0, ENV.rd, ENV.rf, 0.0, ENV.tau))
    fwd = ENV.forward
    assert itm == pytest.approx(ENV.df_dom * (fwd - 1.0), rel=1e-12)
    assert itm >= 0.0


def test_surface_reduces_to_flat_when_no_smile():
    """RR=BF=0 => VV adjustment vanishes => smile is flat at sigma_atm."""
    flat_quotes = SmileQuotes(sigma_atm=0.10, rr25=0.0, bf25_2vol=0.0)
    surf = VannaVolgaVolSurface(ENV, flat_quotes, DeltaConvention.SPOT)
    for k in (1.05, 1.15, 1.20, 1.25, 1.35):
        assert surf.get_vol(k, ENV.tau, ENV.spot) == pytest.approx(0.10, abs=5e-5)


def test_surface_inversion_correct_for_deep_strikes():
    """Flat smile must invert to exactly sigma_atm even very deep OTM/ITM."""
    flat = SmileQuotes(sigma_atm=0.10, rr25=0.0, bf25_2vol=0.0)
    surf = VannaVolgaVolSurface(ENV, flat, DeltaConvention.SPOT)
    for k in (3.0 * ENV.spot, 5.0 * ENV.spot, 0.2 * ENV.spot):
        assert surf.get_vol(k, ENV.tau, ENV.spot) == pytest.approx(0.10, abs=1e-6)


def test_surface_recovers_atm_vol_near_atm_strike():
    surf = VannaVolgaVolSurface(ENV, QUOTES, DeltaConvention.SPOT)
    k_atm = ENV.forward * math.exp(0.5 * QUOTES.sigma_atm**2 * ENV.tau)
    vol_atm = surf.get_vol(k_atm, ENV.tau, ENV.spot)
    # At the ATM strike the VV smile should be very close to sigma_atm.
    assert vol_atm == pytest.approx(QUOTES.sigma_atm, abs=2e-3)


def test_surface_reflects_risk_reversal_skew():
    """Negative RR (rr25<0) => puts richer => low-strike vol > high-strike vol."""
    surf = VannaVolgaVolSurface(ENV, QUOTES, DeltaConvention.SPOT)
    low = surf.get_vol(1.05, ENV.tau, ENV.spot)
    high = surf.get_vol(1.35, ENV.tau, ENV.spot)
    assert low > high


def test_surface_rejects_bad_inputs():
    with pytest.raises(ValidationError):
        VannaVolgaVolSurface(ENV, SmileQuotes(-0.1, 0.0, 0.0), DeltaConvention.SPOT)
    surf = VannaVolgaVolSurface(ENV, QUOTES, DeltaConvention.SPOT)
    with pytest.raises(ValidationError):
        surf.get_vol(-1.0, ENV.tau, ENV.spot)


def test_smile_quotes_rejects_negative_derived_wing():
    """SmileQuotes(.1, -.25, 0) yields a negative call wing and must be rejected."""
    with pytest.raises(ValidationError):
        SmileQuotes(0.10, -0.25, 0.0)


def test_zero_vol_delta_respects_moneyness():
    from quantark.param.vol.vannavolga import bs_delta

    # Deterministic OTM call (strike well above forward) has ~zero delta.
    otm_call = bs_delta(2.0, True, ENV, 0.0, DeltaConvention.SPOT)
    assert otm_call == pytest.approx(0.0, abs=1e-12)
    # Deterministic ITM put (strike well above forward) has negative delta.
    itm_put = bs_delta(2.0, False, ENV, 0.0, DeltaConvention.SPOT)
    assert itm_put < 0.0


def test_delta_convention_accepts_legacy_strings():
    assert DeltaConvention("spot") is DeltaConvention.SPOT
    assert DeltaConvention("fwd_prem") is DeltaConvention.FWD_PREM


def _reference_gk(is_call, S, K, rd, rf, sigma, tau):
    """Independent textbook Garman-Kohlhagen price, for cross-checking."""
    from scipy.stats import norm

    F = S * math.exp((rd - rf) * tau)
    df_dom = math.exp(-rd * tau)
    df_for = math.exp(-rf * tau)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * tau) / (sigma * math.sqrt(tau))
    d2 = d1 - sigma * math.sqrt(tau)
    if is_call:
        return S * df_for * norm.cdf(d1) - K * df_dom * norm.cdf(d2)
    return K * df_dom * norm.cdf(-d2) - S * df_for * norm.cdf(-d1)


def test_param_gk_matches_textbook_garman_kohlhagen():
    """Param-layer self-contained GK must match an independent GK formula.

    This guards against the param-layer GK copy drifting from standard GK
    (the engine-layer GarmanKohlhagenEngine remains the production source of
    truth; it uses the same closed form).
    """
    strike, sigma = 1.25, 0.10
    for is_call in (True, False):
        got = price_gk(is_call, GKInput(ENV.spot, strike, ENV.rd, ENV.rf, sigma, ENV.tau))
        ref = _reference_gk(is_call, ENV.spot, strike, ENV.rd, ENV.rf, sigma, ENV.tau)
        assert got == pytest.approx(ref, rel=1e-12)


def test_rebound_reanchors_spot_and_rates():
    surf = VannaVolgaVolSurface(ENV, QUOTES, DeltaConvention.SPOT)
    bumped = surf.rebound(spot=1.25, rd=0.025, rf=0.012, tau=1.0)
    assert bumped is not surf
    assert bumped.env.spot == 1.25
    assert bumped.env.rd == 0.025
    assert bumped.env.rf == 0.012
    # quotes + convention are preserved (intrinsic smile data)
    assert bumped.quotes == QUOTES
    assert bumped.conv == surf.conv
    # original is untouched (immutable market data)
    assert surf.env.spot == 1.20


def test_with_quotes_shifts_all_three_quotes():
    surf = VannaVolgaVolSurface(ENV, QUOTES, DeltaConvention.SPOT)
    shifted = SmileQuotes(
        sigma_atm=QUOTES.sigma_atm + 0.01,
        rr25=QUOTES.rr25 + 0.01,
        bf25_2vol=QUOTES.bf25_2vol + 0.01,
    )
    bumped = surf.with_quotes(shifted)
    assert bumped is not surf
    assert bumped.quotes == shifted
    assert bumped.env.spot == surf.env.spot  # anchor unchanged
    assert surf.quotes == QUOTES               # original untouched
