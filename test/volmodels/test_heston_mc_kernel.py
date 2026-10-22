import numpy as np
import pytest
from golden_compare import GOLDEN_REL_TOL
from quantark.util.enum.engine_enums import HestonMCScheme
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams, heston_call_price
from quantark.volmodels.heston.mc_kernel import (
    _simulate_terminal_spot,
    price_european_heston_mc,
    simulate_heston_spot_nodes,
)


P = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)


def _const(T, M, r, q):
    return np.full(M, T / M), np.full(M, r), np.full(M, q)


@pytest.mark.parametrize(
    "scheme",
    [
        HestonMCScheme.EULER,
        HestonMCScheme.EULERLOG,
        HestonMCScheme.FULL_TRUNCATION_EULER,
        HestonMCScheme.QUADEXP,
        HestonMCScheme.QUADEXP_M,
    ],
)
def test_recorded_path_terminal_matches_terminal_kernel(scheme):
    dt, rf, cf = _const(1.0, 8, 0.03, 0.01)
    rng = np.random.default_rng(19)
    z_var = rng.standard_normal((64, 8))
    z_ind = rng.standard_normal((64, 8))
    u_var = rng.random((64, 8))

    nodes = simulate_heston_spot_nodes(
        100.0,
        P,
        scheme,
        dt,
        rf,
        cf,
        z_var,
        z_ind,
        u_var,
        record_steps=np.asarray([0, 8]),
    )
    terminal = _simulate_terminal_spot(
        100.0, P, scheme, dt, rf, cf, z_var, z_ind, u_var
    )

    assert nodes.shape == (64, 2)
    assert np.array_equal(nodes[:, -1], terminal)


@pytest.mark.parametrize("scheme", [HestonMCScheme.EULER, HestonMCScheme.EULERLOG, HestonMCScheme.QUADEXP])
def test_mc_converges_to_analytic(scheme):
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    dt, rf, cf = _const(T, 100, r, q)
    price, se = price_european_heston_mc(
        s0, k, True, P, dt, rf, cf, disc_factor=float(np.exp(-r * T)),
        scheme=scheme, num_paths=200_000, seed=7, use_antithetic=True, return_stderr=True,
    )
    analytic = heston_call_price(s0, k, T, P, r, q)
    assert abs(price - analytic) < 4 * se + 0.05


def test_quadexp_accurate_few_steps():
    # QE is accurate even with coarse time stepping (its design goal).
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.01
    dt, rf, cf = _const(T, 12, r, q)
    price = price_european_heston_mc(
        s0, k, True, P, dt, rf, cf, disc_factor=float(np.exp(-r * T)),
        scheme=HestonMCScheme.QUADEXP, num_paths=300_000, seed=3, use_antithetic=True,
    )
    assert price == pytest.approx(heston_call_price(s0, k, T, P, r, q), abs=0.1)


def test_put_via_parity_mc():
    s0, k, T, r, q = 100.0, 105.0, 1.0, 0.02, 0.0
    dt, rf, cf = _const(T, 100, r, q)
    kw = dict(s0=s0, strike=k, params=P, step_dt=dt, r_fwd=rf, carry_fwd=cf,
              disc_factor=float(np.exp(-r * T)), scheme=HestonMCScheme.QUADEXP,
              num_paths=200_000, seed=11, use_antithetic=True)
    c = price_european_heston_mc(is_call=True, **kw)
    p = price_european_heston_mc(is_call=False, **kw)
    parity = s0 * np.exp(-q * T) - k * np.exp(-r * T)
    assert (c - p) == pytest.approx(parity, abs=0.1)


def test_reproducible_and_validation():
    dt, rf, cf = _const(1.0, 50, 0.0, 0.0)
    kw = dict(s0=100.0, strike=100.0, is_call=True, params=P, step_dt=dt, r_fwd=rf,
              carry_fwd=cf, disc_factor=1.0, scheme=HestonMCScheme.QUADEXP,
              num_paths=10_000, seed=42)
    assert price_european_heston_mc(**kw) == price_european_heston_mc(**kw)
    with pytest.raises(ValidationError):
        price_european_heston_mc(s0=-1.0, strike=100.0, is_call=True, params=P, step_dt=dt,
                                 r_fwd=rf, carry_fwd=cf, disc_factor=1.0, num_paths=100)


def test_quadexp_deterministic_vol_is_rho_independent():
    # sigma->0: variance deterministic => price independent of rho, equals BS.
    from quantark.volmodels.black_scholes import bs_call_price
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.0, 0.0
    dt, rf, cf = _const(T, 100, r, q)
    bs = bs_call_price(s0, k, T, np.sqrt(0.04), r, q)
    for rho in (0.0, -0.7, -0.99):
        p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=1e-10, rho=rho)
        price, se = price_european_heston_mc(
            s0, k, True, p, dt, rf, cf, disc_factor=1.0, scheme=HestonMCScheme.QUADEXP,
            num_paths=200_000, seed=4, use_antithetic=True, return_stderr=True)
        assert abs(price - bs) < 4 * se + 0.02


def test_quadexp_zero_kappa_matches_analytic():
    # kappa=0, sigma>0: variance is a driftless CIR; QE moments use the k->0 limit.
    p = HestonParams(v0=0.04, kappa=0.0, theta=0.04, sigma=0.3, rho=-0.5)
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.02, 0.0
    dt, rf, cf = _const(T, 200, r, q)
    price, se = price_european_heston_mc(
        s0, k, True, p, dt, rf, cf, disc_factor=float(np.exp(-r * T)),
        scheme=HestonMCScheme.QUADEXP, num_paths=300_000, seed=8,
        use_antithetic=True, return_stderr=True)
    analytic = heston_call_price(s0, k, T, p, r, q)
    assert abs(price - analytic) < 4 * se + 0.05


# --- QE exponential-branch sign fix (found during WS-B4, 2026-07-05) ---

def test_quadexp_martingale_feller_violated():
    # Feller 2*kappa*theta/sigma^2 = 0.48 (violated): psi > 1.5 fires the exponential
    # branch constantly. Pre-fix, Psi^{-1} was negated -> every branch-B draw clamped
    # to 0 -> variance collapse -> E[S_T]/forward ~ 1.033 at 100 steps. Post-fix the
    # martingale property holds to MC error.
    params = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.5, rho=-0.7)
    M, n = 100, 200_000
    rng = np.random.default_rng(42)
    z_var = rng.standard_normal((n, M))
    z_ind = rng.standard_normal((n, M))
    u = rng.random((n, M))
    dt = np.full(M, 1.0 / M)
    s = _simulate_terminal_spot(100.0, params, HestonMCScheme.QUADEXP, dt,
                                np.full(M, 0.03), np.full(M, 0.01), z_var, z_ind, u)
    assert abs(np.mean(s) / (100.0 * np.exp(0.02)) - 1.0) < 3e-3


def test_quadexp_converges_to_analytic_feller_violated():
    from quantark.volmodels.heston.analytical_kernel import heston_call_price
    params = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.5, rho=-0.7)
    M = 100
    dt = np.full(M, 1.0 / M)
    p = price_european_heston_mc(100.0, 100.0, True, params, dt,
                                 np.full(M, 0.03), np.full(M, 0.01),
                                 disc_factor=float(np.exp(-0.03)),
                                 scheme=HestonMCScheme.QUADEXP,
                                 num_paths=200_000, seed=42)
    analytic = heston_call_price(100.0, 100.0, 1.0, params, 0.03, 0.01)
    assert abs(p - analytic) < 0.10          # pre-fix error was ~2.3


# --- WS-B4: u_var drawn only for QUADEXP; z/u streams pinned across the change ---

@pytest.mark.parametrize("scheme,use_antithetic,pinned", [
    (HestonMCScheme.EULER, False, 8.05990821865451),
    (HestonMCScheme.EULER, True, 8.090175350910078),
    (HestonMCScheme.EULERLOG, False, 8.071218418656654),
    (HestonMCScheme.EULERLOG, True, 8.1013457393631),
    (HestonMCScheme.QUADEXP, False, 8.145397651538548),
    (HestonMCScheme.QUADEXP, True, 8.119291373836068),
])
def test_seed_stability_pinned_across_u_var_change(scheme, use_antithetic, pinned):
    # Captured BEFORE u_var became QUADEXP-only. u draws happen after z draws, so
    # removing them for EULER/EULERLOG cannot perturb the z-stream, and QUADEXP's
    # u-stream is unchanged. Exact equality: the random streams must not move.
    params = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.5, rho=-0.7)
    dt = np.full(12, 1.0 / 12.0)
    p = price_european_heston_mc(100.0, 100.0, True, params, dt,
                                 np.full(12, 0.03), np.full(12, 0.01),
                                 disc_factor=float(np.exp(-0.03)), scheme=scheme,
                                 num_paths=20_000, seed=42, use_antithetic=use_antithetic)
    # Pinned same-machine; x86_64 CI drifts from the ARM64 freeze host by the
    # last ULP over 20k paths, so compare within cross-arch tolerance. The
    # invariant under test (z/u streams unmoved) survives a ~1e-9 rel window.
    assert p == pytest.approx(pinned, rel=GOLDEN_REL_TOL)
