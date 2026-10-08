"""The MC REBASED KO-reset route: accrued offset, V1 at each path's own maturity, V1 on that path's tenor
(spec 2026-10-08 section 5, review round 2 finding 4)."""
import numpy as np
import pytest

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.util.enum.option_enums import PostKOScheduleMode

from .conftest import env
from .test_ko_reset_product import D, ko_reset


def _mc():
    return SnowballMCEngine(params=MCParams(seed=11, num_paths=16384))


def test_an_aged_rebased_knock_out_keeps_the_accrued_offset():
    """One knock-in date, so the REBASED contract IS the ABSOLUTE one at 91 days + offsets."""
    rebased = ko_reset(PostKOScheduleMode.REBASED, [91 * D, 365 * D], ki_times=(91 * D,), annualized=True,
                       accrued_offset=0.1)
    absolute = ko_reset(PostKOScheduleMode.ABSOLUTE, [182 * D, 456 * D], ki_times=(91 * D,), annualized=True,
                        accrued_offset=0.1, maturity=456 * D)
    assert _mc().price(rebased, env(90.0)) == pytest.approx(_mc().price(absolute, env(90.0)), rel=1e-12)


def test_an_annualized_rebased_v1_prices():
    product = ko_reset(PostKOScheduleMode.REBASED, [91 * D, 365 * D], annualized=True)
    assert np.isfinite(_mc().price(product, env(80.0)))


def test_v1_is_read_at_each_paths_own_maturity_on_its_tenor():
    """A path knocked in at 91 days matures at 91 + 365; the grid runs on to 182 + 365."""
    product = ko_reset(PostKOScheduleMode.REBASED, [91 * D, 365 * D], annualized=True)
    engine = _mc()
    pricing_env = env(80.0)
    grid = engine._build_time_grid_ko_reset(product, pricing_env, product.get_max_maturity_time(pricing_env))
    times = grid["all_times"]
    paths = np.full((1, times.size + 1), 70.0)               # knocked in at 91 days, never above 95
    maturity_node = int(np.searchsorted(times, 91 * D + 365 * D))
    assert maturity_node < times.size - 1
    paths[0, maturity_node + 1] = 60.0
    paths[0, -1] = 40.0                                       # the grid's last node: another path's maturity
    payoffs, _, stats, _ = engine._compute_payoffs_ko_reset(product, pricing_env, paths, grid, 0.03,
                                                            float(times[-1]), 0.25, 1)
    assert stats["v1_count"] == 1
    assert payoffs[0] == pytest.approx(product.get_maturity_payoff_v1(60.0, pricing_env, contract_tenor=456 * D))
