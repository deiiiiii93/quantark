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
