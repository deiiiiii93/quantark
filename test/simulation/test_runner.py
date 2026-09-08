from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantark.backtest.simulation.engine import EnsembleBacktestEngine
from quantark.backtest.simulation.paths.gbm import ConstantVol, GBMPaths
from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID, StartState, trading_calendar
from quantark.backtest.simulation.runner import batch_ranges, concat_results, run_ensemble

from .conftest import RATE, SPOT, VOL, ensemble_config, flat_carry, ladder_pricing

START = date(2024, 1, 2)
FLOAT_COLUMNS = ("total_pnl", "product_mtm", "delta", "futures_contracts", "cash", "pending_receivable_pv")


def _paths(n_paths=6, n_days=8, sigma=0.6, seed=5):
    start = StartState(spot=SPOT, atm_vol=VOL, rate=RATE, carry=flat_carry(DEFAULT_TENOR_GRID))
    gen = GBMPaths(start=start, calendar=trading_calendar(START, n_days), mu=0.0, sigma=sigma,
                   vol_rule=ConstantVol(VOL), carry_schedule=None, rate=RATE, tenor_grid=DEFAULT_TENOR_GRID)
    return gen.generate(n_paths, n_days, seed=seed)


def _trades(results):
    """Per path, chronological: the whole run logs day-major, a batched run batch-major."""
    rows = [{k: v for k, v in t.items() if k != "date"} for t in results.trades]
    return sorted(rows, key=lambda t: (t["path"], t["day"]))         # stable: same-day order kept


def _events(results):
    def clean(e):
        d = dict(e.__dict__)
        d["barrier"] = None if np.isnan(d["barrier"]) else d["barrier"]   # nan != nan
        return d
    return sorted((clean(e) for e in results.events), key=lambda e: (e["path"], e["day"], e["product"]))


def _same(a, b):
    for name in FLOAT_COLUMNS:
        assert np.array_equal(getattr(a.cube, name), getattr(b.cube, name)), name
    assert a.cube.active_contract == b.cube.active_contract
    assert np.array_equal(a.last_day, b.last_day) and np.array_equal(a.initial_book_value, b.initial_book_value)
    assert _trades(a) == _trades(b)
    assert _events(a) == _events(b)


def test_batch_ranges():
    assert batch_ranges(7, 3) == [range(0, 3), range(3, 6), range(6, 7)]
    assert batch_ranges(7, None) == [range(0, 7)]
    assert batch_ranges(2, 5) == [range(0, 2)]


def test_batching_is_bit_inert_in_process():
    paths = _paths()
    whole = EnsembleBacktestEngine(ensemble_config()).run(paths)
    batched = run_ensemble(ensemble_config(batch_paths=2), paths)
    _same(whole, batched)
    assert batched.manifest["path_fingerprint"] == paths.fingerprint()
    assert len(batched.manifest["batches"]) == 3
    assert batched.manifest["engine_calls"] == sum(b["engine_calls"] for b in batched.manifest["batches"])


def test_a_spawn_pool_gives_the_same_numbers():
    paths = _paths()
    whole = EnsembleBacktestEngine(ensemble_config()).run(paths)
    pooled = run_ensemble(ensemble_config(workers=2, batch_paths=2), paths)
    _same(whole, pooled)


def test_the_ladder_batches_share_states_through_the_disk_tier(tmp_path):
    paths = _paths()
    cfg = ensemble_config(pricing=ladder_pricing(disk_dir=str(tmp_path)), batch_paths=3)
    first = run_ensemble(cfg, paths)
    second = run_ensemble(cfg, paths)
    assert second.manifest["engine_calls"] == 0
    _same(first, second)


def test_concat_remaps_path_indices():
    paths = _paths(n_paths=4)
    a = EnsembleBacktestEngine(ensemble_config()).run(paths.take([0, 1]))
    b = EnsembleBacktestEngine(ensemble_config()).run(paths.take([2, 3]))
    whole = concat_results([a, b], path_fingerprint=paths.fingerprint())
    assert whole.n_paths == 4
    assert {t["path"] for t in whole.trades} <= {0, 1, 2, 3}
    assert any(t["path"] >= 2 for t in whole.trades) or not b.trades
    assert whole.path_states(3).equals(b.path_states(1))
