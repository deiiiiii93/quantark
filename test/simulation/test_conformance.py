from __future__ import annotations

from datetime import date

import pytest

from quantark.backtest.replay import ReplayProduct
from quantark.backtest.simulation.conformance import OracleReport, run_oracle
from quantark.backtest.simulation.paths.gbm import ConstantVol, GBMPaths
from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID, StartState, trading_calendar
from quantark.backtest.strategy.futures_delta_strategy import AutocallableDeltaHedgeStrategy

from .conftest import RATE, SPOT, VOL, ensemble_config, flat_carry, pde_engine_config, short_snowball

START = date(2024, 1, 2)


def _paths(n_paths: int = 5, n_days: int = 8, sigma: float = 0.6):
    start = StartState(spot=SPOT, atm_vol=VOL, rate=RATE, carry=flat_carry(DEFAULT_TENOR_GRID))
    gen = GBMPaths(start=start, calendar=trading_calendar(START, n_days), mu=0.0, sigma=sigma,
                   vol_rule=ConstantVol(VOL), carry_schedule=None, rate=RATE,
                   tenor_grid=DEFAULT_TENOR_GRID)
    return gen.generate(n_paths, n_days, seed=11)


@pytest.mark.parametrize("path_index", [0, 1, 2, 3, 4])
def test_every_path_matches_the_replay_engine_exactly(path_index):
    report = run_oracle(ensemble_config(), _paths(), path_index)
    assert isinstance(report, OracleReport)
    assert report.passed, report.summary()
    assert report.exact_columns_match
    assert report.trade_mismatches == 0
    assert report.max_pv_gap == 0.0
    assert report.max_delta_gap == 0.0
    assert report.max_total_pnl_gap == 0.0


def test_a_term_dividend_source_also_matches():
    cfg = ensemble_config(engine_config=pde_engine_config(
        dividend_source="futures_curve", futures_curve_extrapolation="flat_q",
        futures_curve_min_tenor_days=1,
    ))
    report = run_oracle(cfg, _paths(n_paths=2), 1)
    assert report.passed, report.summary()


def test_a_traded_initial_price_matches():
    cfg = ensemble_config(products=[
        ReplayProduct(product=short_snowball(), quantity=-1000.0, position_id=1,
                      has_lifecycle=True, initial_price=0.0)
    ])
    report = run_oracle(cfg, _paths(n_paths=2), 0)
    assert report.passed, report.summary()


def test_a_two_product_book_matches():
    cfg = ensemble_config(products=[
        ReplayProduct(product=short_snowball(), quantity=-1000.0, position_id=1, has_lifecycle=True),
        ReplayProduct(product=short_snowball(ko_days=(3, 5)), quantity=-2000.0, position_id=2,
                      has_lifecycle=True),
    ])
    report = run_oracle(cfg, _paths(n_paths=2), 0)
    assert report.passed, report.summary()


def test_a_deliberate_disagreement_is_reported_not_hidden():
    # A different hedge ratio in the simulation must show up as a mismatch.
    cfg = ensemble_config(strategy=AutocallableDeltaHedgeStrategy(
        delta_threshold=0.0, hedge_ratio=0.5, target_delta=0.0))
    report = run_oracle(cfg, _paths(n_paths=1), 0, _replay_strategy=AutocallableDeltaHedgeStrategy(
        delta_threshold=0.0, hedge_ratio=1.0, target_delta=0.0))
    assert not report.passed
    assert report.first_mismatch is not None


def test_the_report_summary_names_the_path_and_the_day_count():
    report = run_oracle(ensemble_config(), _paths(n_paths=1), 0)
    text = report.summary()
    assert "path 0" in text and str(report.days) in text
