"""A 2D solver's statistics refusal is not masked by the replay's MC fallback (spec 3B §5.6, decision 8)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from replay_golden import fixtures  # noqa: E402

from quantark.asset.equity.engine.event_stats import StatisticsRefused  # noqa: E402
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import HestonSnowballPDESolver  # noqa: E402
from quantark.backtest.replay.single import AutocallableBacktestEngine  # noqa: E402
from quantark.volmodels.heston import HestonParams  # noqa: E402

HP = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.6)


def test_mc_fallback_does_not_swallow_a_2d_refusal():
    config = fixtures.make_scalar_bsm_config()
    config.engine_config.event_stats_fallback = "mc"
    engine = AutocallableBacktestEngine(config)
    # a default 2D solver: statistics refused unless statistics='flat_vol_surrogate'
    engine._inner._replays[0].event_stats_engine = HestonSnowballPDESolver(HP, n_x=48, n_v=18, n_t=16)
    with pytest.raises(StatisticsRefused, match="flat_vol_surrogate"):
        engine.run()
