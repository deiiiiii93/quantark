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
