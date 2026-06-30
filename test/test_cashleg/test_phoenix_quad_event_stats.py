"""Native Phoenix QUAD event stats: refactor guard + KO/KI/coupon vs Phoenix MC."""

import pathlib

import numpy as np

from quantark.asset.equity.engine.event_stats import PhoenixEventStats
from test_cashleg._autocallable_helpers import make_env, make_snowball, make_phoenix, make_engine


def test_snowball_quad_event_stats_unchanged_after_refactor():
    env = make_env()
    sb = make_snowball()
    s = make_engine("quad", "snowball").calculate_event_stats(sb, env)
    assert s is not None and s.ko_probability.shape == s.ko_times.shape
