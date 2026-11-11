"""Native Phoenix PDE event stats: refactor guard + KO/KI/coupon vs Phoenix MC (unified EventStatistics, T12)."""

import pathlib

import numpy as np

from quantark.asset.equity.engine.event_stats import EventStatistics
from test_cashleg._autocallable_helpers import make_env, make_snowball, make_phoenix, make_engine


def test_snowball_pde_event_stats_unchanged_after_refactor():
    env = make_env()
    sb = make_snowball()
    s = make_engine("pde", "snowball").calculate_event_stats(sb, env)
    assert s is not None
    assert s.observations.probability.shape == s.observations.times.shape
    assert 0.0 <= float(np.sum(s.observations.probability)) <= 1.0 + 1e-9


def test_phoenix_pde_ko_survival_match_mc():
    env = make_env()
    ph = make_phoenix()
    s_pde = make_engine("pde", "phoenix").calculate_event_stats(ph, env)
    s_mc = make_engine("mc", "phoenix").calculate_event_stats(ph, env)
    assert isinstance(s_pde, EventStatistics) and s_pde.product_family == "phoenix"
    # two knock-out rows and two coupon rows at their settlements, then the maturity and knock-in settlement rows
    assert s_pde.ledger.determination_times.shape == (6,)
    assert s_pde.ledger.payment_times.shape == (6,)
    assert abs(s_pde.pv - s_pde.ledger.discounted_cash.sum()) < 1e-8
    np.testing.assert_allclose(s_pde.observations.times, s_mc.observations.times, atol=1e-9)
    np.testing.assert_allclose(s_pde.observations.probability, s_mc.observations.probability, atol=5e-3)
    np.testing.assert_allclose(s_pde.observations.survival, s_mc.observations.survival, atol=5e-3)


def test_phoenix_pde_module_has_no_mc_import():
    import quantark.asset.equity.engine.pde.phoenix_pde_solver as mod

    assert "MCEngine" not in pathlib.Path(mod.__file__).read_text()


def test_phoenix_pde_coupon_prob_match_mc():
    env = make_env()
    ph = make_phoenix()
    s_pde = make_engine("pde", "phoenix").calculate_event_stats(ph, env)
    s_mc = make_engine("mc", "phoenix").calculate_event_stats(ph, env)
    assert s_pde.coupons.probability.shape == s_mc.coupons.probability.shape
    np.testing.assert_allclose(s_pde.coupons.probability, s_mc.coupons.probability, atol=5e-3)


def test_phoenix_pde_coupon_at_simultaneous_ko_matches_mc():
    # KO barrier below coupon barrier => KO region is within coupon-pay region:
    # every KO observation is also a coupon trigger. Locks "coupon counts at KO".
    env = make_env()
    ph = make_phoenix(ko_barrier=90.0, coupon_barrier=(80.0, 80.0))
    s_pde = make_engine("pde", "phoenix").calculate_event_stats(ph, env)
    s_mc = make_engine("mc", "phoenix").calculate_event_stats(ph, env)
    np.testing.assert_allclose(s_pde.coupons.probability, s_mc.coupons.probability, atol=6e-3)
