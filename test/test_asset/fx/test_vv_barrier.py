"""
Tests for the engine-layer Vanna-Volga barrier pricing.
"""

import math

import numpy as np
import pytest

from quantark.asset.fx.engine.analytical.vannavolga import (
    BarrierGamma,
    BarrierPrices,
    enforce_double_barrier_arbitrage,
    enforce_single_barrier_arbitrage,
    gamma_fet,
    gamma_surv,
    no_touch_price,
    one_touch_hit_prob,
    price_vv_one_touch,
    survival_probability_single,
)
from quantark.param.vol.vannavolga import FXEnv, SmileQuotes
from quantark.util.exceptions import ValidationError

ENV = FXEnv(spot=1.10, rd=0.01, rf=0.005, tau=1.0)
QUOTES = SmileQuotes(sigma_atm=0.10, rr25=-0.005, bf25_2vol=0.002)


def test_one_touch_prob_in_unit_interval():
    for H in (1.15, 1.20, 1.30):
        p = one_touch_hit_prob(ENV.spot, H, 0.10, ENV.tau, ENV.rd - ENV.rf, is_up=True)
        assert 0.0 <= p <= 1.0


def test_one_touch_prob_increases_as_barrier_approaches_spot():
    near = one_touch_hit_prob(ENV.spot, 1.12, 0.10, ENV.tau, ENV.rd - ENV.rf, is_up=True)
    far = one_touch_hit_prob(ENV.spot, 1.40, 0.10, ENV.tau, ENV.rd - ENV.rf, is_up=True)
    assert near > far


def test_no_touch_price_relation():
    H = 1.25
    p_hit = one_touch_hit_prob(ENV.spot, H, 0.10, ENV.tau, ENV.rd - ENV.rf, is_up=True)
    nt = no_touch_price(ENV.spot, H, 0.10, ENV.tau, ENV.rd, ENV.rd - ENV.rf, is_up=True)
    assert nt == pytest.approx(math.exp(-ENV.rd * ENV.tau) * (1.0 - p_hit), rel=1e-12)


def test_survival_probability_bounds_and_monotonicity():
    near = survival_probability_single(ENV.spot, 1.12, ENV.rd, ENV.rf, 0.10, ENV.tau, is_up=True)
    far = survival_probability_single(ENV.spot, 1.50, ENV.rd, ENV.rf, 0.10, ENV.tau, is_up=True)
    assert 0.0 <= near <= 1.0
    assert 0.0 <= far <= 1.0
    # Survival is higher when the barrier is further from spot.
    assert far > near


def test_gamma_surv_in_unit_interval():
    g = gamma_surv(ENV, None, 1.25, 0.10)
    assert 0.0 <= g <= 1.0


def test_gamma_fet_pde_in_unit_interval():
    g = gamma_fet(ENV, None, 1.25, 0.10, method="pde", nS=101, nT=100)
    assert 0.0 <= g <= 1.0


def test_gamma_fet_no_barrier_returns_one():
    assert gamma_fet(ENV, None, None, 0.10) == 1.0


def test_gamma_fet_pde_and_mc_agree():
    """The deterministic PDE and seeded MC FET estimates should be close."""
    g_pde = gamma_fet(ENV, None, 1.30, 0.10, method="pde", nS=201, nT=200)
    rng = np.random.default_rng(12345)
    g_mc = gamma_fet(ENV, None, 1.30, 0.10, method="mc", num_paths=40000, steps=400, rng=rng)
    assert g_mc == pytest.approx(g_pde, abs=0.05)


def test_gamma_fet_rejects_unknown_method():
    with pytest.raises(ValidationError):
        gamma_fet(ENV, None, 1.25, 0.10, method="bogus")


def test_single_barrier_arbitrage_clamps():
    pr = BarrierPrices(vanilla=0.05, ko=0.06, wko=-0.01)
    out = enforce_single_barrier_arbitrage(pr)
    assert out.ko is not None and out.wko is not None
    assert 0.0 <= out.ko <= out.vanilla
    assert out.ko <= out.wko <= out.vanilla


def test_double_barrier_arbitrage_clamps():
    pr = BarrierPrices(vanilla=0.05, ko=0.04, dko=0.10)
    out = enforce_double_barrier_arbitrage(pr, ko1=0.03, ko2=0.045)
    assert out.dko is not None
    assert out.dko <= 0.03  # min(ko1, ko2)


def test_price_vv_one_touch_returns_sensible_values():
    res = price_vv_one_touch(ENV, QUOTES, barrier=1.25, is_up=True, gamma_type=BarrierGamma.SURV)
    assert 0.0 <= res.gamma <= 1.0
    assert np.isfinite(res.bstv) and np.isfinite(res.vv)
    # One-touch price is a (discounted) probability: within [0, 1].
    assert 0.0 <= res.bstv <= 1.0
    assert res.omega.shape == (3,)


def test_price_vv_one_touch_fet_method_runs():
    res = price_vv_one_touch(
        ENV, QUOTES, barrier=1.25, is_up=True, gamma_type=BarrierGamma.FET, fet_method="pde"
    )
    assert np.isfinite(res.vv)


def test_price_vv_one_touch_rejects_bad_barrier():
    with pytest.raises(ValidationError):
        price_vv_one_touch(ENV, QUOTES, barrier=-1.0, is_up=True)


def test_price_vv_one_touch_clamped_to_arbitrage_bounds():
    """Large VV corrections must not push the price outside [0, DF_dom]."""
    env = FXEnv(spot=1.20, rd=0.01, rf=0.005, tau=1.0)
    quotes = SmileQuotes(sigma_atm=0.10, rr25=-0.20, bf25_2vol=0.01)  # extreme RR
    res = price_vv_one_touch(env, quotes, barrier=1.10, is_up=False, gamma_type=BarrierGamma.SURV)
    df_dom = math.exp(-env.rd * env.tau)
    assert 0.0 <= res.vv <= df_dom


def test_double_barrier_capped_by_vanilla_without_ko_bounds():
    pr = BarrierPrices(vanilla=1.0, dko=2.0)
    out = enforce_double_barrier_arbitrage(pr, ko1=None, ko2=None)
    assert out.dko == pytest.approx(1.0)


def test_gamma_fet_mc_zero_when_spot_already_breached():
    """Spot at/beyond the barrier => first-exit time 0 for both PDE and MC."""
    env = FXEnv(spot=1.30, rd=0.01, rf=0.005, tau=1.0)
    rng = np.random.default_rng(7)
    g_mc = gamma_fet(env, None, 1.25, 0.10, method="mc", num_paths=2000, steps=50, rng=rng)
    assert g_mc == 0.0


def test_one_touch_prob_stable_for_low_vol_high_carry():
    """Low vol + strong carry must not overflow the (H/S)^(2mu) tail term."""
    p = one_touch_hit_prob(1.0, 1.5, 0.01, 1.0, 0.2, is_up=True)
    assert 0.0 <= p <= 1.0
    assert math.isfinite(p)


def test_gamma_fet_zero_vol_one_sided_deterministic():
    """Zero-vol one-sided FET must reflect the drift hitting the barrier."""
    env = FXEnv(spot=1.0, rd=0.20, rf=0.0, tau=2.0)
    # Drift b=0.2 reaches ln(1.2)/0.2 = 0.912y < 2y, so lambda<tau => gamma<1.
    g = gamma_fet(env, None, 1.2, 0.0, method="pde")
    assert 0.0 < g < 1.0
    # Closed-form deterministic check (both measures share the zero-vol path).
    t_hit = math.log(1.2) / 0.2
    assert g == pytest.approx(t_hit / env.tau, rel=1e-9)


def test_gamma_fet_pde_mc_agree_with_drift():
    """Even with stronger drift the widened-domain PDE should track MC."""
    env = FXEnv(spot=1.0, rd=0.08, rf=0.0, tau=1.0)
    g_pde = gamma_fet(env, None, 1.25, 0.12, method="pde", nS=301, nT=300)
    rng = np.random.default_rng(99)
    g_mc = gamma_fet(env, None, 1.25, 0.12, method="mc", num_paths=40000, steps=400, rng=rng)
    assert g_mc == pytest.approx(g_pde, abs=0.05)


def test_double_barrier_negative_ko_bound_clamped():
    pr = BarrierPrices(vanilla=1.0, dko=0.2)
    out = enforce_double_barrier_arbitrage(pr, ko1=-0.1, ko2=None)
    assert out.dko is not None and out.dko >= 0.0
    assert out.dko == pytest.approx(0.0)


def test_one_touch_greeks_stable_for_tiny_vol():
    """Tiny sigma must keep the FD bump positive (no deterministic-branch leak)."""
    from quantark.asset.fx.engine.analytical.vannavolga import numeric_greeks_ot

    g = numeric_greeks_ot(ENV, 0.0004, 1.25, True)
    assert all(math.isfinite(v) for v in g.values())


def test_matured_one_touch_returns_immediate_payoff():
    env0 = FXEnv(spot=1.30, rd=0.01, rf=0.005, tau=0.0)
    res = price_vv_one_touch(env0, QUOTES, barrier=1.25, is_up=True)
    # Already breached up barrier at expiry: pays 1 (undiscounted at tau=0).
    assert res.bstv == pytest.approx(1.0, abs=1e-12)
    assert res.vv == res.bstv


def test_negative_tau_rejected():
    env_neg = FXEnv(spot=1.10, rd=0.01, rf=0.005, tau=-0.5)
    with pytest.raises(ValidationError):
        price_vv_one_touch(env_neg, QUOTES, barrier=1.25, is_up=True)


def test_survival_probability_at_expiry():
    # Unbreached at expiry -> survival 1.
    assert survival_probability_single(1.10, 1.25, 0.01, 0.005, 0.10, 0.0, is_up=True) == 1.0
    # Already breached -> survival 0.
    assert survival_probability_single(1.30, 1.25, 0.01, 0.005, 0.10, 0.0, is_up=True) == 0.0


def test_one_touch_zero_vol_deterministic_crossing():
    """Zero-vol path with positive drift can still touch an upper barrier."""
    # b = rd - rf > 0 so the forward drifts up through the barrier.
    env = FXEnv(spot=1.10, rd=0.20, rf=0.0, tau=2.0)
    b = env.rd - env.rf
    s_T = env.spot * math.exp(b * env.tau)
    assert s_T > 1.20  # deterministic terminal crosses the barrier
    p = one_touch_hit_prob(env.spot, 1.20, 0.0, env.tau, b, is_up=True)
    assert p == 1.0
    # A barrier above the terminal level is never touched.
    p_no = one_touch_hit_prob(env.spot, 2.0, 0.0, env.tau, b, is_up=True)
    assert p_no == 0.0
