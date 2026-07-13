"""DCN two-surface PDE solver tests: solver unit gates, event-operator
ordering, and the MC<->PDE cross-validation gate (acceptance criterion #3)."""
import numpy as np
import pytest
from scipy.stats import norm

from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.engine.pde.dcn_pde_solver import (
    DCNPDEEngine,
    _operator_coeffs,
    _sinh_grid,
    _theta_step,
    apply_dcn_events,
)
from quantark.asset.equity.product.option.dcn_option import DCNDirection

from dcn_fixtures import (
    DCN_A,
    FLAT,
    SAMPLE_DCN_A,
    SAMPLE_DCN_B,
    flat_env,
    make_dcn,
)

# Decided validation setting: >= 2^17 Sobol paths, and stderr must be
# < 2bp*N (raise paths, never widen the gate). DCN-B's loss leg carries more
# variance, so it needs one more doubling than DCN-A.
GATE_SETTINGS = {"DCN-A": (2 ** 20, 16), "DCN-B": (2 ** 21, 32)}


def _bs_put(s, k, t, r, q, sigma):
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    return k * np.exp(-r * t) * norm.cdf(-d2) - s * np.exp(-q * t) * norm.cdf(-d1)


def test_theta_scheme_prices_european_put():
    # bare solver loop, no events: must reproduce Black-Scholes (with q)
    r, q, sigma = FLAT["r"], FLAT["q"], FLAT["sigma"]
    s0, k, t_mat = 6000.0, 6600.0, 2.0
    n_steps = 500
    x = _sinh_grid(np.log(0.05 * s0), np.log(4.0 * s0), np.log(0.75 * s0),
                   801, 0.15)
    s_grid = np.exp(x)
    v = np.maximum(k - s_grid, 0.0)
    dt = t_mat / n_steps
    lo, mid, hi = _operator_coeffs(x, r, q, sigma)
    for i in range(n_steps):
        if i < 2:  # Rannacher start
            v = _theta_step(v, lo, mid, hi, dt / 2.0, 1.0)
            v = _theta_step(v, lo, mid, hi, dt / 2.0, 1.0)
        else:
            v = _theta_step(v, lo, mid, hi, dt, 0.5)
    pde = float(np.interp(np.log(s0), x, v))
    ref = _bs_put(s0, k, t_mat, r, q, sigma)
    assert pde == pytest.approx(ref, rel=1e-3)


def test_event_operator_ordering():
    # backward order: coupon injection -> KO overwrite -> KI projection
    s_grid = np.array([4000.0, 4600.0, 5000.0, 6100.0])
    ki_mask = s_grid <= 4500.0
    cpn_mask = s_grid >= 4800.0
    ko_mask = s_grid >= 6000.0
    v0 = np.zeros(4)
    v1 = np.full(4, -100.0)
    v0_out, v1_out = apply_dcn_events(
        v0, v1, cpn_mask, ko_mask, ki_mask, coupon_amount=5.0, ko_amount=7.0
    )
    assert v0_out[3] == 7.0 and v1_out[3] == 7.0   # KO wins over coupon
    assert v0_out[2] == 5.0                        # coupon region, no KO
    assert v0_out[0] == -100.0                     # KI projection onto v1
    assert v0_out[1] == 0.0                        # no event region
    assert v1_out[0] == -100.0 and v1_out[2] == -100.0  # v1 never gets coupons


@pytest.mark.parametrize(
    "contract,ident,direction",
    [
        (SAMPLE_DCN_A, "DCN-A", DCNDirection.BUYER),
        (SAMPLE_DCN_B, "DCN-B", DCNDirection.SELLER),
    ],
    ids=["DCN-A", "DCN-B"],
)
def test_mc_pde_cross_validation_gate(contract, ident, direction):
    p = make_dcn(contract, direction=direction)
    env = flat_env(**FLAT, spot=contract["initial_price"])
    paths, batches = GATE_SETTINGS[ident]
    mc = DCNMCEngine(
        num_paths=paths, num_batches=batches, seed=42
    ).price_detailed(p, env)
    pde = DCNPDEEngine(num_space_nodes=2401).price(p, env)
    notional = contract["notional"]
    assert mc.std_error < 2e-4 * notional, "stderr gate: raise paths, not tolerance"
    gate = max(3.0 * mc.std_error, 5e-4 * notional)
    assert abs(mc.pv - pde) < gate, (
        f"|MC-PDE| = {abs(mc.pv - pde):.2f} exceeds gate {gate:.2f} "
        f"(MC={mc.pv:.2f} +/- {mc.std_error:.2f}, PDE={pde:.2f})"
    )


def test_pde_buyer_seller_negation_exact():
    env = flat_env(**FLAT)
    b = DCNPDEEngine().price(make_dcn(DCN_A), env)
    s = DCNPDEEngine().price(make_dcn(DCN_A, direction=DCNDirection.SELLER), env)
    assert s == -b


def test_pde_knocked_in_seed_prices_v1():
    env = flat_env(**FLAT)
    base = DCNPDEEngine().price(make_dcn(DCN_A), env)
    ki = DCNPDEEngine().price(make_dcn(DCN_A, knocked_in_at_valuation=True), env)
    assert ki < base  # V1 <= V0 pointwise: no coupons + short put


def test_pde_valuation_date_breach_selects_knocked_in_surface():
    env = flat_env(**FLAT, spot=4000.0)  # below the 4500 KI barrier
    observed_today = DCNPDEEngine().price(make_dcn(DCN_A), env)
    seeded = DCNPDEEngine().price(
        make_dcn(DCN_A, knocked_in_at_valuation=True), env
    )
    assert observed_today == seeded


def test_pde_deterministic():
    env = flat_env(**FLAT)
    p = make_dcn(DCN_A)
    assert DCNPDEEngine().price(p, env) == DCNPDEEngine().price(p, env)
