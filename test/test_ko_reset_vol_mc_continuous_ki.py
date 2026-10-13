"""Local-vol / Heston / SLV MC price a continuous-knock-in KO-reset (the W17 side ticket, I6).

The KO-reset path hands the Brownian bridge the paths sliced to the knock-in horizon (schedule 1), while the vol
engines record the per-step log-variance over the whole contractual grid; the strict shape check refused every such
contract with a PricingError. The KI horizon is now passed to the variance lookup, which slices the record to it.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from test_dated_expiry_settles import HESTON, SLV_LEVERAGE
from test_mc_time_node_merge import _price, env, ko_reset

from quantark.asset.equity.engine.mc import SnowballMCEngine
from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (
    HestonSLVSnowballMCEngine,
    HestonSnowballMCEngine,
    LocalVolSnowballMCEngine,
)
from quantark.asset.equity.param import MCParams

ROUTES = {
    "lv": (lambda: LocalVolSnowballMCEngine(params=MCParams(seed=7, num_paths=256)), True),
    "heston": (lambda: HestonSnowballMCEngine(HESTON, params=MCParams(seed=7, num_paths=256)), False),
    "slv": (lambda: HestonSLVSnowballMCEngine(HESTON, params=MCParams(seed=7, num_paths=256),
                                              leverage_surface=SLV_LEVERAGE), True),
}


@pytest.mark.parametrize("route", list(ROUTES))
@pytest.mark.parametrize("day", [0, 26, 120])
def test_vol_mc_prices_a_continuous_ki_ko_reset_and_the_twins_agree(route, day):
    factory, grid = ROUTES[route]
    dated = _price(ko_reset, factory, grid, "dated", day)
    numeric = _price(ko_reset, factory, grid, "numeric", day)
    assert np.isfinite(dated) and dated == pytest.approx(numeric, rel=1e-10), (dated, numeric)


def test_flat_local_vol_matches_black_scholes_within_four_standard_errors():
    """A flat 20% IV grid makes the local vol 20%: the LV MC and the BS MC estimate one value."""
    bs = SnowballMCEngine(params=MCParams(seed=11, num_paths=20000))
    lv = LocalVolSnowballMCEngine(params=MCParams(seed=12, num_paths=20000))
    v_bs = bs.price(ko_reset("dated", 0), env(100.0, 0))
    v_lv = lv.price(ko_reset("dated", 0), env(100.0, 0, grid=True))
    se = math.hypot(bs.get_last_std_error(), lv.get_last_std_error())
    assert abs(v_lv - v_bs) <= 4.0 * se, (v_lv, v_bs, se)
