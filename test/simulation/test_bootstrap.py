from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.simulation.paths.bootstrap import StationaryBlockBootstrap, stationary_block_indices
from quantark.backtest.simulation.paths.history import PathHistory
from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID, StartState, trading_calendar
from quantark.util.exceptions import ValidationError

from .test_history import _frames, RATE


@pytest.fixture()
def history() -> PathHistory:
    spot, vol, futures = _frames(n_days=60)
    rng = np.random.default_rng(0)
    spot["spot"] = 6000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(spot))))
    vol["volatility"] = 0.2 + np.cumsum(rng.normal(0.0, 0.002, len(vol)))
    return PathHistory.from_frames(spot=spot, vol=vol, futures=futures, rate=RATE, tenor_grid=DEFAULT_TENOR_GRID)


def _gen(history, **overrides):
    kwargs = dict(mean_block_days=5, demean_returns=False, annual_drift=0.0, vol_floor=0.05,
                  carry_mode="changes", start=history.snapshot(), calendar=trading_calendar(date(2025, 1, 2), 40))
    kwargs.update(overrides)
    return StationaryBlockBootstrap(history, **kwargs)


def test_block_indices_are_circular_runs_with_the_requested_mean_length():
    rng = np.random.default_rng(1)
    idx = stationary_block_indices(rng, n_source=50, n_paths=200, n_days=100, mean_block_days=10)
    assert idx.shape == (200, 100) and idx.min() >= 0 and idx.max() < 50
    continues = (idx[:, 1:] == (idx[:, :-1] + 1) % 50).mean()
    assert continues == pytest.approx(0.9, abs=0.02)     # p(new block) = 1/10
    idx1 = stationary_block_indices(rng, n_source=50, n_paths=200, n_days=100, mean_block_days=1)
    assert (idx1[:, 1:] == (idx1[:, :-1] + 1) % 50).mean() == pytest.approx(0.02, abs=0.02)  # iid apart from chance


def test_generate_is_seed_reproducible_and_records_meta(history):
    gen = _gen(history)
    a = gen.generate(4, 40, seed=7)
    b = gen.generate(4, 40, seed=7)
    c = gen.generate(4, 40, seed=8)
    assert a.fingerprint() == b.fingerprint() != c.fingerprint()
    assert a.meta["generator"] == "stationary_block_bootstrap" and a.meta["seed"] == 7
    assert a.meta["mean_block_days"] == 5 and a.meta["history_fingerprint"] == history.source_fingerprint
    assert a.n_paths == 4 and a.n_days == 40 and a.dates.equals(trading_calendar(date(2025, 1, 2), 40))


def test_paths_start_from_the_start_state_and_integrate_resampled_changes(history):
    gen = _gen(history, mean_block_days=10_000)      # one block: the history replayed in order from a random start
    mp = gen.generate(1, 20, seed=3)
    changes = history.changes()
    # find the start row from the first spot change
    first = np.log(mp.spot[0, 1] / mp.spot[0, 0])
    k = int(np.argmin(np.abs(changes[:, 0] - first)))
    assert mp.spot[0, 0] == pytest.approx(history.snapshot().spot)
    for j in range(1, 20):
        row = changes[(k + j - 1) % len(changes)]
        assert np.log(mp.spot[0, j] / mp.spot[0, j - 1]) == pytest.approx(row[0], abs=1e-12)
        assert mp.atm_vol[0, j] - mp.atm_vol[0, j - 1] == pytest.approx(row[1], abs=1e-12)
        assert mp.carry[0, j] - mp.carry[0, j - 1] == pytest.approx(row[3:], abs=1e-12)


def test_demean_and_drift_set_the_mean_log_return(history):
    gen = _gen(history, demean_returns=True, annual_drift=0.0504, mean_block_days=1,
               calendar=trading_calendar(date(2025, 1, 2), 60))
    mp = gen.generate(400, 60, seed=11)
    mean_daily = np.log(mp.spot[:, -1] / mp.spot[:, 0]).mean() / 59
    assert mean_daily == pytest.approx(0.0504 / 252, abs=4e-4)


def test_vol_floor_binds_on_generated_days_and_is_counted(history):
    gen = _gen(history, vol_floor=0.30)                # above the whole history: floor binds everywhere
    mp = gen.generate(3, 10, seed=5)
    # day 0 is the declared start state, not a simulated day: it is never floored
    assert mp.atm_vol[:, 0] == pytest.approx(history.snapshot().atm_vol)
    assert np.all(mp.atm_vol[:, 1:] >= 0.30)
    assert mp.meta["vol_floor_hits"] > 0


def test_levels_mode_resamples_carry_levels(history):
    gen = _gen(history, carry_mode="levels", mean_block_days=1)
    mp = gen.generate(2, 15, seed=2)
    rows = {tuple(np.round(r, 12)) for r in history.carry}
    for i in range(2):
        for j in range(1, 15):
            assert tuple(np.round(mp.carry[i, j], 12)) in rows


def test_parameters_are_validated(history):
    with pytest.raises(ValidationError):
        _gen(history, mean_block_days=0)
    with pytest.raises(ValidationError):
        _gen(history, carry_mode="random")
    with pytest.raises(ValidationError):
        _gen(history).generate(0, 10, seed=1)
