from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantark.backtest.replay import ReplayProduct
from quantark.backtest.simulation.engine import EnsembleBacktestEngine, EnsembleResults
from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID, StartState, trading_calendar
from quantark.backtest.simulation.paths.gbm import ConstantVol, GBMPaths
from quantark.util.exceptions import ValidationError

from .conftest import RATE, SPOT, VOL, ensemble_config, flat_carry, short_snowball

START = date(2024, 1, 2)


def _paths(n_paths: int = 4, n_days: int = 8, sigma: float = 0.25):
    start = StartState(spot=SPOT, atm_vol=VOL, rate=RATE, carry=flat_carry(DEFAULT_TENOR_GRID))
    gen = GBMPaths(start=start, calendar=trading_calendar(START, n_days), mu=0.0, sigma=sigma,
                   vol_rule=ConstantVol(VOL), carry_schedule=None, rate=RATE,
                   tenor_grid=DEFAULT_TENOR_GRID)
    return gen.generate(n_paths, n_days, seed=3)


def test_the_run_produces_a_cube_of_the_right_shape():
    paths = _paths()
    results = EnsembleBacktestEngine(ensemble_config()).run(paths)
    assert isinstance(results, EnsembleResults)
    assert results.n_paths == paths.n_paths
    assert results.cube.total_pnl.shape == (paths.n_paths, paths.n_days)
    assert len(results.cube.active_contract) == paths.n_days
    frame = results.path_states(0)
    assert {"date", "portfolio_value", "total_pnl", "futures_contracts", "alive"} <= set(frame.columns)


def test_the_accounting_identity_holds_on_every_day_of_every_path():
    results = EnsembleBacktestEngine(ensemble_config()).run(_paths())
    cube = results.cube
    identity = cube.portfolio_value - results.initial_book_value[:, None]
    assert cube.total_pnl == pytest.approx(identity, abs=1e-9)


def test_portfolio_value_is_its_own_parts():
    results = EnsembleBacktestEngine(ensemble_config()).run(_paths())
    cube = results.cube
    assert cube.portfolio_value == pytest.approx(
        cube.product_mtm + cube.hedge_mtm + cube.cash + cube.pending_receivable_pv, abs=1e-9
    )
    assert cube.cash == pytest.approx(cube.cashflows - cube.transaction_costs, abs=1e-9)


def test_the_dividend_object_is_built_once_per_distinct_env_key():
    # a held carry curve and a flat rate: every path shares one env_key a day,
    # so the build count is the number of days the loop actually executed
    results = EnsembleBacktestEngine(ensemble_config()).run(_paths(n_paths=6))
    assert results.manifest["dividend_builds"] == results.manifest["days_run"]
    assert results.manifest["days_run"] >= 1


def test_a_dead_path_carries_no_product_mark_and_no_hedge():
    # A deep crash knocks in but never knocks out; a strong rally knocks out.
    paths = _paths(n_paths=8, sigma=1.2)
    results = EnsembleBacktestEngine(ensemble_config()).run(paths)
    dead = ~results.cube.alive
    assert results.cube.product_mtm[dead] == pytest.approx(0.0)
    assert results.cube.delta[dead] == pytest.approx(0.0)
    assert results.cube.futures_contracts[dead] == pytest.approx(0.0)


def test_the_hedge_targets_the_book_delta_each_day():
    cfg = ensemble_config()
    results = EnsembleBacktestEngine(cfg).run(_paths())
    cube = results.cube
    alive = cube.alive
    expected = np.round(-cube.delta / cfg.hedge.multiplier)
    assert cube.futures_contracts[alive] == pytest.approx(expected[alive])


def test_a_path_stops_updating_once_it_settles():
    paths = _paths(n_paths=6, sigma=1.2)
    results = EnsembleBacktestEngine(ensemble_config()).run(paths)
    for i in range(results.n_paths):
        last = int(results.last_day[i])
        if last < paths.n_days - 1:
            assert results.cube.total_pnl[i, last:] == pytest.approx(results.cube.total_pnl[i, last])
        assert len(results.path_states(i)) == last + 1


def test_an_initial_price_on_the_product_replaces_the_day_zero_mark():
    cfg = ensemble_config(products=[
        ReplayProduct(product=short_snowball(), quantity=-1.0, position_id=1,
                      has_lifecycle=True, initial_price=0.0)
    ])
    results = EnsembleBacktestEngine(cfg).run(_paths())
    assert results.initial_book_value == pytest.approx(0.0)
    assert results.cube.total_pnl[:, 0] == pytest.approx(results.cube.portfolio_value[:, 0], abs=1e-9)


def test_a_calendar_too_short_for_the_product_fails_closed():
    with pytest.raises(ValidationError):
        EnsembleBacktestEngine(ensemble_config()).run(_paths(n_days=3))


def test_a_short_calendar_is_allowed_when_data_end_is_declared():
    results = EnsembleBacktestEngine(ensemble_config(allow_data_end=True)).run(_paths(n_days=3))
    assert results.manifest["data_end_paths"] == results.n_paths


def test_the_manifest_records_the_run():
    results = EnsembleBacktestEngine(ensemble_config()).run(_paths())
    manifest = results.manifest
    assert manifest["path_fingerprint"] and manifest["engine_fingerprint"]
    assert manifest["gate"]["mode"] == "exact"
    assert manifest["cache"]["hits"] >= 0 and manifest["engine_calls"] > 0
