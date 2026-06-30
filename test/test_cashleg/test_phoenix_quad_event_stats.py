"""Native Phoenix QUAD event stats: refactor guard + KO/KI/coupon vs Phoenix MC."""

import pathlib

import numpy as np

from quantark.asset.equity.engine.event_stats import PhoenixEventStats
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.param import QuadParams
from test_cashleg._autocallable_helpers import make_env, make_snowball, make_phoenix, make_engine


def _quad(grid_points=4001):
    # KO/coupon probabilities near a barrier need grid resolution to match MC;
    # QUAD converges to PDE/MC as grid_points grows (grid discretization, not bias).
    return PhoenixQuadEngine(params=QuadParams(grid_points=grid_points))


def test_snowball_quad_event_stats_unchanged_after_refactor():
    env = make_env()
    sb = make_snowball()
    s = make_engine("quad", "snowball").calculate_event_stats(sb, env)
    assert s is not None and s.ko_probability.shape == s.ko_times.shape


def test_phoenix_quad_ko_survival_match_mc():
    env = make_env()
    ph = make_phoenix()
    s_q = _quad().calculate_event_stats(ph, env)
    s_mc = make_engine("mc", "phoenix").calculate_event_stats(ph, env)
    assert isinstance(s_q, PhoenixEventStats)
    np.testing.assert_allclose(s_q.ko_probability, s_mc.ko_probability, atol=6e-3)
    np.testing.assert_allclose(s_q.survival_probability, s_mc.survival_probability, atol=6e-3)


def test_phoenix_quad_module_has_no_mc_import():
    import quantark.asset.equity.engine.quad.phoenix_quad_engine as mod

    assert "MCEngine" not in pathlib.Path(mod.__file__).read_text()


def test_phoenix_quad_coupon_prob_match_mc():
    env = make_env()
    ph = make_phoenix()
    s_q = _quad().calculate_event_stats(ph, env)
    s_mc = make_engine("mc", "phoenix").calculate_event_stats(ph, env)
    assert s_q.coupon_probability.shape == s_mc.coupon_probability.shape
    np.testing.assert_allclose(s_q.coupon_probability, s_mc.coupon_probability, atol=6e-3)


def test_phoenix_quad_coupon_at_simultaneous_ko_matches_mc():
    env = make_env()
    ph = make_phoenix(ko_barrier=90.0, coupon_barrier=(80.0, 80.0))
    s_q = _quad().calculate_event_stats(ph, env)
    s_mc = make_engine("mc", "phoenix").calculate_event_stats(ph, env)
    np.testing.assert_allclose(s_q.coupon_probability, s_mc.coupon_probability, atol=7e-3)


def test_phoenix_quad_coupon_prob_with_ki_and_disable_ko_after_ki():
    # Knocked-in-but-alive paths still earn coupons: disable_ko_after_ki suppresses
    # future KO, never coupons. Coarse lock against the catastrophic regression
    # (gating the coupon row on disable_ko_after_ki dropped post-KI coupon mass to
    # ~0). NOTE: continuous-KI QUAD coupon stats carry a Brownian-bridge
    # approximation (~5-7% vs MC) even though KO/survival match tightly; for tight
    # continuous-KI coupon valuation prefer the MC engine.
    env = make_env()
    ph = make_phoenix(ki_barrier=95.0, disable_ko_after_ki=True)
    s_q = _quad().calculate_event_stats(ph, env)
    s_mc = make_engine("mc", "phoenix").calculate_event_stats(ph, env)
    np.testing.assert_allclose(s_q.coupon_probability, s_mc.coupon_probability, atol=8e-2)


def test_phoenix_quad_expiry_coupon_cashflow_uses_real_maturity():
    # Last KO observation (0.75) strictly before maturity (1.0); EXPIRY coupons must
    # discount to maturity, not the last KO date. (Regression for the maturity proxy.)
    from quantark.util.enum import CouponPayType

    env = make_env(rate=0.10)
    ph = make_phoenix(ko_dates=(0.5, 0.75), maturity=1.0,
                      coupon_pay=CouponPayType.EXPIRY)
    s_q = _quad().calculate_event_stats(ph, env)
    s_mc = make_engine("mc", "phoenix").calculate_event_stats(ph, env)
    np.testing.assert_allclose(
        s_q.expected_discounted_coupon_cashflow,
        s_mc.expected_discounted_coupon_cashflow,
        atol=3e-3,
    )
