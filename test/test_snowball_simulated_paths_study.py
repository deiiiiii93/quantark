"""Tests for ``example/snowball_simulated_paths``: every stage on synthetic frames.

The study's real inputs are local, untracked caches; these tests build a
synthetic history and a one-month snowball so the whole pipeline -- paths,
fair coupon, cells, oracle spot check, report -- runs in seconds.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID, MarketPath, trading_calendar

REPO = Path(__file__).resolve().parents[1]
STUDY_DIR = REPO / "example" / "snowball_simulated_paths"
RATE = 0.02


def _load(name: str):
    path = STUDY_DIR / name
    spec = importlib.util.spec_from_file_location(f"snowball_simulated_paths_{name.replace('.py', '')}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


C = _load("_common.py")
S01 = _load("01_build_paths.py")


def synthetic_frames(n_days: int = 80):
    """Spot, vol and a four-contract IM chain with a flat -10% carry, like the plan-1 tests."""
    dates = trading_calendar(date(2024, 1, 2), n_days)
    rng = np.random.default_rng(0)
    spot = 6000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, n_days)))
    spot_df = pd.DataFrame({"date": dates, "spot": spot})
    vol_df = pd.DataFrame({"date": dates, "volatility": 0.22 + 0.02 * np.sin(np.arange(n_days) / 9.0)})
    chain = [("IM2401", pd.Timestamp("2024-01-19")), ("IM2402", pd.Timestamp("2024-02-23")),
             ("IM2403", pd.Timestamp("2024-03-15")), ("IM2406", pd.Timestamp("2024-06-21")),
             ("IM2409", pd.Timestamp("2024-09-20"))]
    rows = []
    for k, d in enumerate(dates):
        for code, expiry in chain:
            t = (expiry - d).days / 365.0
            if t <= 0:
                continue
            rows.append({"date": d, "contract": code, "futures_price": float(spot[k]) * np.exp(-0.10 * t),
                         "expiry_date": expiry, "multiplier": 200.0})
    return spot_df, vol_df, pd.DataFrame(rows)


def test_build_history_and_paths_from_frames():
    spot, vol, futures = synthetic_frames()
    history = S01.build_history(spot, vol, futures, rate=RATE)
    assert history.n_days == 80 and history.tenor_grid.shape == DEFAULT_TENOR_GRID.shape
    bootstrap, stress = S01.build_paths(history, n_paths=4, n_days=30, seed=7, mean_block_days=5,
                                        annual_drift=0.0, vol_floor=0.08, carry_mode="changes")
    assert (bootstrap.n_paths, bootstrap.n_days) == (4, 30) and stress.n_days == 30 and stress.n_paths == 5
    assert bootstrap.dates[0] > history.dates[-1] and bootstrap.dates.equals(stress.dates)
    assert bootstrap.spot[:, 0] == pytest.approx(history.spot[-1])
    assert bootstrap.meta["seed"] == 7 and stress.meta["scenario_names"][0].startswith("crash_into_ki")
    again, _ = S01.build_paths(history, n_paths=4, n_days=30, seed=7, mean_block_days=5,
                               annual_drift=0.0, vol_floor=0.08, carry_mode="changes")
    assert again.fingerprint() == bootstrap.fingerprint()


def test_paths_are_written_with_a_manifest_and_read_back(tmp_path):
    spot, vol, futures = synthetic_frames()
    history = S01.build_history(spot, vol, futures, rate=RATE)
    bootstrap, stress = S01.build_paths(history, n_paths=3, n_days=20, seed=1, mean_block_days=5,
                                        annual_drift=0.0, vol_floor=0.08, carry_mode="changes")
    manifest = S01.write_paths(tmp_path, bootstrap, stress, history=history)
    assert manifest["bootstrap_fingerprint"] == bootstrap.fingerprint()
    assert manifest["history_fingerprint"] == history.source_fingerprint
    b, s, m = S01.load_paths(tmp_path)
    assert b.fingerprint() == bootstrap.fingerprint() and s.fingerprint() == stress.fingerprint() and m == manifest


def test_the_study_reuses_the_q_study_and_names_its_cells():
    assert C.Q.Q_MODELS["term_opt_tail"].dividend_source == "futures_curve"
    assert C.cell_name("term_flat_q", "far") == "term_flat_q__far"
    cfg = C.engine_config("term_opt_tail", "pde", quad_grid=101)
    assert cfg.dividend_source == "futures_curve" and cfg.futures_curve_extrapolation == "surface_forward_carry"
    assert cfg.futures_curve_min_tenor_days == 1
    assert C.engine_config("flat_active", "quad", quad_grid=101).dividend_source is None
