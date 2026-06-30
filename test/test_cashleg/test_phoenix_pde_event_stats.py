"""Native Phoenix PDE event stats: refactor guard + KO/KI/coupon vs Phoenix MC."""

import pathlib

import numpy as np

from quantark.asset.equity.engine.event_stats import PhoenixEventStats
from test_cashleg._autocallable_helpers import make_env, make_snowball, make_phoenix, make_engine


def test_snowball_pde_event_stats_unchanged_after_refactor():
    env = make_env()
    sb = make_snowball()
    s = make_engine("pde", "snowball").calculate_event_stats(sb, env)
    assert s is not None
    assert s.ko_probability.shape == s.ko_times.shape
    assert 0.0 <= float(np.sum(s.ko_probability)) <= 1.0 + 1e-9


def test_phoenix_pde_ko_survival_match_mc():
    env = make_env()
    ph = make_phoenix()
    s_pde = make_engine("pde", "phoenix").calculate_event_stats(ph, env)
    s_mc = make_engine("mc", "phoenix").calculate_event_stats(ph, env)
    assert isinstance(s_pde, PhoenixEventStats)
    np.testing.assert_allclose(s_pde.ko_times, s_mc.ko_times, atol=1e-9)
    np.testing.assert_allclose(s_pde.ko_probability, s_mc.ko_probability, atol=5e-3)
    np.testing.assert_allclose(
        s_pde.survival_probability, s_mc.survival_probability, atol=5e-3
    )


def test_phoenix_pde_module_has_no_mc_import():
    import quantark.asset.equity.engine.pde.phoenix_pde_solver as mod

    assert "MCEngine" not in pathlib.Path(mod.__file__).read_text()
