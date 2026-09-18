from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from quantark.backtest.replay import ReplayProduct
from quantark.backtest.simulation.config import PricingProviderConfig
from quantark.backtest.simulation.engine import EnsembleBacktestEngine, EnsembleResults
from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID, StartState, trading_calendar
from quantark.backtest.simulation.paths.gbm import ConstantVol, GBMPaths
from quantark.util.exceptions import ValidationError

from .conftest import (RATE, SPOT, VOL, ensemble_config, flat_carry, ladder_pricing, short_snowball,
                       surface_pricing)

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


def test_the_dividend_object_is_built_once_per_distinct_environment():
    # A held carry curve and a flat rate: on day 0 every path shares one
    # (rate, spot, carry) row, so one build serves the batch; afterwards the
    # GBM spots differ and each path needs its own (the replay's basis
    # arithmetic is not spot-free at the last ulp, see _day_market).
    paths = _paths(n_paths=6)
    results = EnsembleBacktestEngine(ensemble_config()).run(paths)
    days_run = results.manifest["days_run"]
    assert days_run >= 2
    assert results.manifest["dividend_builds"] == 1 + (days_run - 1) * paths.n_paths


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
        ReplayProduct(product=short_snowball(), quantity=-1000.0, position_id=1,
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


def test_ladder_and_surface_runs_keep_the_accounting_identity():
    for pricing in (ladder_pricing(), surface_pricing()):
        results = EnsembleBacktestEngine(ensemble_config(pricing=pricing)).run(_paths(n_paths=6))
        cube = results.cube
        assert cube.total_pnl == pytest.approx(cube.portfolio_value - results.initial_book_value[:, None], abs=1e-6)
        assert results.manifest["mode"] == pricing.mode
        assert results.manifest["gate"]["mode"] == pricing.mode and results.manifest["gate"]["passed"]


def test_the_surface_run_makes_far_fewer_engine_calls():
    exact = EnsembleBacktestEngine(ensemble_config()).run(_paths(n_paths=6))
    surface = EnsembleBacktestEngine(ensemble_config(pricing=surface_pricing())).run(_paths(n_paths=6))
    assert surface.manifest["solves"] >= 1
    assert surface.manifest["solves"] < exact.manifest["engine_calls"] / 4


def test_a_failed_gate_aborts_the_cell_with_its_report():
    from quantark.backtest.simulation.config import GateConfig
    from quantark.backtest.simulation.pricing.base import GateFailure

    coarse = ladder_pricing(spot_step=0.05)
    strict = PricingProviderConfig(provider="repricing", cache=coarse.cache, spot_step=0.05, vol_step=0.0, q_step=0.0,
                                   gate=GateConfig(sample_states=8, pv_tolerance_bp=1e-6, delta_tolerance_hands=1e-6))
    with pytest.raises(GateFailure) as excinfo:
        EnsembleBacktestEngine(ensemble_config(pricing=strict)).run(_paths(n_paths=4))
    assert excinfo.value.report.mode == "ladder" and not excinfo.value.report.passed


def test_a_disk_cache_serves_a_second_run(tmp_path):
    pricing = ladder_pricing(disk_dir=str(tmp_path))
    first = EnsembleBacktestEngine(ensemble_config(pricing=pricing)).run(_paths(n_paths=4))
    second = EnsembleBacktestEngine(ensemble_config(pricing=pricing)).run(_paths(n_paths=4))
    assert second.manifest["engine_calls"] == 0
    assert second.manifest["cache"]["disk"]["disk_hits"] > 0
    assert second.cube.total_pnl == pytest.approx(first.cube.total_pnl)


def test_the_env_key_separates_two_hedge_contracts_on_the_same_market_row():
    """The active contract fixes the dividend, so it must reach the state key.

    ``StateKey`` deliberately carries nothing about the hedge, on the
    grounds that the hedge does not price the product.  It does here: the
    dividend is implied by inverting the ACTIVE futures contract, so two
    cells whose roll policies pick different contracts price different
    dividends off one market row.  Were the key blind to that, a shared
    disk cache would serve one cell's prices to the other.
    """
    from quantark.backtest.simulation.carry import day_chain

    paths = _paths(n_paths=3, n_days=8)
    engine = EnsembleBacktestEngine(ensemble_config())
    chain = day_chain(paths, 3)
    rate = paths.rate[:, 3]
    assert len(chain.contracts) >= 2, "the test needs two listed contracts to choose between"

    near, far = 0, len(chain.contracts) - 1
    key_near, div_near, _, _, _ = engine._day_market(paths, 3, chain, near, chain.contracts[near], rate)
    key_far, div_far, _, _, _ = engine._day_market(paths, 3, chain, far, chain.contracts[far], rate)

    # The two contracts really do imply different dividends on this row ...
    assert div_near[0].get_yield(1.0) != div_far[0].get_yield(1.0)
    # ... so no state may share a key between them.
    assert not set(key_near.tolist()) & set(key_far.tolist())
