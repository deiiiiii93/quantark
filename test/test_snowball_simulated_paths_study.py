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


C = _load("_sim_common.py")
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
    assert C.engine_config(C.MODELS[0], "quad", quad_grid=101).dividend_source is None


S02 = _load("02_ensemble_fleet.py")

#: The one-month fixture: a 90% knock-in (the study's 75% is unreachable in a
#: month, which makes the fair coupon exactly zero), and a narrow life-surface
#: domain that still covers the 30% stress crash -- the PDE mesh costs 10 ms
#: per new spot on it against about a second on the study's wide envelope.
FIXTURE_KI_PCT = 0.90
FIXTURE_SPOT_RANGE = (0.60, 1.60)
CELL = dict(cost_bp=1.0, workers=1, batch_paths=None, quad_grid=101, spot_range=FIXTURE_SPOT_RANGE)
#: The fixture's gate, not the study's.  Measured on this seed: day 17 of
#: bootstrap path 3 sits 0.02% above the 90% knock-in barrier five days from
#: maturity, where the alive value kinks and the surface reads linearly
#: across the kink -- 85.7 bp and 140.1 hands against the exact engine (the
#: next worst state, 0.2% above the barrier, 13.1 bp / 57.2 hands).  The
#: study's gate (25 bp / 2 hands) would fail this cell, which
#: ``test_the_gate_records_the_near_barrier_gap`` pins; the fixture widens the
#: budget so the pipeline runs and the numbers are recorded, not hidden.
FIXTURE_GATE = dict(sample_states=64, pv_tolerance_bp=120.0, delta_tolerance_hands=200.0)
#: The QUAD ladder reads the same kink across 0.25% nodes: 12.41 bp and 0.97
#: hands measured against the study's 10 bp / 2 hands.
FIXTURE_LADDER_GATE = dict(sample_states=64, pv_tolerance_bp=20.0, delta_tolerance_hands=2.0)


def fixture_terms(dates):
    import dataclasses
    return dataclasses.replace(S02.study_terms(dates, maturity_months=1, lockout_months=1), ki_pct=FIXTURE_KI_PCT)


@pytest.fixture(scope="module")
def tiny_fleet(tmp_path_factory):
    """A one-month snowball on four bootstrap paths: one cell on the life surface, a ladder check, an oracle."""
    out = tmp_path_factory.mktemp("fleet")
    spot, vol, futures = synthetic_frames()
    history = S01.build_history(spot, vol, futures, rate=RATE)
    bootstrap, stress = S01.build_paths(history, n_paths=4, n_days=40, seed=2, mean_block_days=5,
                                        annual_drift=0.0, vol_floor=0.08, carry_mode="changes")
    S01.write_paths(out, bootstrap, stress, history=history)
    terms = fixture_terms(bootstrap.dates)
    coupon = S02.fair_coupon(bootstrap, terms, model="term_flat_q", quad_grid=101)
    product = C.Q.build_product(terms, float(bootstrap.spot[0, 0]), coupon.coupon)
    runs = {}
    # The oracle spot-checks path 1, which never comes near the knock-in
    # barrier.  Paths 0 and 3 sit within 0.2% of it for a day, where the
    # widened gate allows the hedge to differ by up to 140 hands; a day of
    # that moves the path's total P&L by 200-560 bp, far past the oracle's
    # P&L budget, which is the PV budget (a sequence gap, not a state gap).
    for model, hedge in ((C.MODELS[0], "front"), ("term_flat_q", "front")):
        cell = C.cell_name(model, hedge)
        cfg = S02.cell_config(product, model, hedge, provider="life_surface", gate_override=FIXTURE_GATE, **CELL)
        runs[cell] = S02.run_cell(bootstrap, cfg, out / "cells" / cell, resume=False, oracle_paths=[1])
        stress_cfg = S02.cell_config(product, model, hedge, provider="life_surface", gate_override=FIXTURE_GATE, **CELL)
        runs[cell + "__stress"] = S02.run_cell(stress, stress_cfg, out / "cells" / f"{cell}__stress",
                                               resume=False, oracle_paths=[])
        ladder_cfg = S02.cell_config(product, model, hedge, provider="ladder", gate_override=FIXTURE_LADDER_GATE, **CELL)
        runs[cell + "__ladder_quad"] = S02.run_cell(bootstrap.take([0, 1]), ladder_cfg,
                                                    out / "cells" / f"{cell}__ladder_quad", resume=False, oracle_paths=[1])
    C.write_json(out / "fleet_manifest.json", {"coupon": coupon.summary(), "terms": terms.summary(), "runs": runs})
    return out, runs, coupon


def test_the_fair_coupon_prices_the_product_to_zero_at_the_start_state(tiny_fleet):
    _, _, coupon = tiny_fleet
    assert coupon.converged and abs(coupon.pv) <= coupon.tolerance and 0.0 < coupon.coupon < 2.0


def test_every_cell_run_is_persisted_gated_and_oracle_checked(tiny_fleet):
    out, runs, _ = tiny_fleet
    from quantark.backtest.simulation.results import EnsembleResults

    for name, run in runs.items():
        assert not run["skipped"] and not run["failed"] and run["gate"]["passed"], name
        results = EnsembleResults.from_dir(out / "cells" / name)
        assert results.manifest["mode"] == ("ladder" if name.endswith("ladder_quad") else "life_surface")
        assert (out / "cells" / name / "config.json").exists()
        for report in run["oracle"]:
            assert report["passed"] and report["exact_columns_match"], (name, report)
    assert len(runs[C.BASELINE_CELL]["oracle"]) == 1 and runs[C.BASELINE_CELL + "__stress"]["oracle"] == []


def test_the_gate_records_the_near_barrier_gap(tiny_fleet):
    """The fixture's widened gate records what the study's gate would have refused (see FIXTURE_GATE)."""
    _, runs, _ = tiny_fleet
    gate = runs[C.BASELINE_CELL]["gate"]
    assert gate["passed"] and gate["sampled"] > 0
    assert gate["max_pv_gap_bp"] > C.GATE_SURFACE["pv_tolerance_bp"]
    assert gate["max_delta_gap_hands"] > C.GATE_SURFACE["delta_tolerance_hands"]
    assert gate["max_pv_gap_bp"] <= FIXTURE_GATE["pv_tolerance_bp"]


def test_resume_skips_a_run_whose_config_matches(tiny_fleet):
    out, _, coupon = tiny_fleet
    bootstrap, _, _ = S01.load_paths(out)
    terms = fixture_terms(bootstrap.dates)
    product = C.Q.build_product(terms, float(bootstrap.spot[0, 0]), coupon.coupon)
    cfg = S02.cell_config(product, C.MODELS[0], "front", provider="life_surface", gate_override=FIXTURE_GATE, **CELL)
    again = S02.run_cell(bootstrap, cfg, out / "cells" / C.BASELINE_CELL, resume=True, oracle_paths=[0])
    assert again["skipped"]
    other = S02.cell_config(product, C.MODELS[0], "front", provider="life_surface", gate_override=FIXTURE_GATE,
                            **{**CELL, "cost_bp": 2.0})
    assert S02.config_fingerprint(other, bootstrap) != S02.config_fingerprint(cfg, bootstrap)


def test_the_cell_config_states_every_choice(tiny_fleet):
    out, _, coupon = tiny_fleet
    bootstrap, _, _ = S01.load_paths(out)
    terms = fixture_terms(bootstrap.dates)
    product = C.Q.build_product(terms, float(bootstrap.spot[0, 0]), coupon.coupon)
    cfg = S02.cell_config(product, "term_opt_tail", "far", provider="ladder", cost_bp=1.0, workers=2,
                          batch_paths=100, quad_grid=201)
    assert cfg.pricing.mode == "ladder" and cfg.engine_config.futures_curve_extrapolation == "surface_forward_carry"
    assert cfg.products[0].initial_price == 0.0 and cfg.products[0].quantity == C.Q.PRODUCT_QUANTITY
    assert type(cfg.hedge.roll_policy).__name__ == "FarContractRollPolicy"
    assert (cfg.workers, cfg.batch_paths) == (2, 100) and cfg.metadata["model"] == "term_opt_tail"
    assert cfg.metadata["spot_range"] == list(C.SURFACE_SPOT_RANGE)
    exact = S02.cell_config(product, "term_flat_q", "front", provider="exact", cost_bp=0.0, workers=1,
                            batch_paths=None, quad_grid=101)
    assert exact.pricing.mode == "exact" and exact.pricing.gate.sample_states == 0
    tol = S02.oracle_tolerances(cfg)
    assert tol["contracts_tolerance"] == C.GATE_LADDER["delta_tolerance_hands"]
    assert S02.oracle_tolerances(exact) == {"pv_tolerance": 0.0, "delta_tolerance": 0.0, "contracts_tolerance": 0.0}


S03 = _load("03_report.py")


def test_aggregate_reduces_every_cell_and_pairs_them(tiny_fleet):
    out, runs, _ = tiny_fleet
    agg = S03.aggregate(out, es_level=0.25, historical_dir=None)
    assert set(agg["cells"]) == {C.BASELINE_CELL, "term_flat_q__front"}
    cell = agg["cells"]["term_flat_q__front"]
    assert set(S03.HEADLINE_MEASURES) <= set(cell["distributions"])
    d = cell["distributions"]["terminal_pnl_bp"]
    assert d["n_paths"] == 4 and "q50" in d["quantiles"] and d["es_level"] == 0.25
    assert [p["variant"] for p in agg["paired"]] == ["term_flat_q__front"] and agg["paired"][0]["base"] == C.BASELINE_CELL
    assert agg["paired"][0]["measures"]["terminal_pnl_bp"]["n"] == 4
    assert {row["cell"] for row in agg["stress"]} == {C.BASELINE_CELL, "term_flat_q__front"}
    assert len(agg["stress"]) == 10 and all(row["scenario"] for row in agg["stress"])
    check = {row["cell"]: row for row in agg["engine_check"]}
    assert check["term_flat_q__front"]["n"] == 2 and "terminal_pnl_bp" in check["term_flat_q__front"]["measures"]
    assert agg["gates"][C.BASELINE_CELL]["passed"] and agg["historical"]["available"] is False


def test_tables_and_report_are_written(tiny_fleet, tmp_path):
    out, _, _ = tiny_fleet
    agg = S03.aggregate(out, es_level=0.25, historical_dir=None)
    S03.write_tables(agg, tmp_path)
    for name in ("fleet_cells.json", "fleet_paired.csv", "stress_table.csv", "engine_check.csv", "fleet_summary.json"):
        assert (tmp_path / name).exists(), name
    # the tables ship in a public repo: no absolute path of this machine in them
    assert str(out) not in (tmp_path / "fleet_summary.json").read_text()
    html = S03.build_report(agg)
    assert "<html" in html and "term_flat_q__front" in html and C.BASELINE_CELL in html
    assert "expected shortfall" in html.lower() and "historical" in html.lower()
    (tmp_path / "report.html").write_text(html)


def test_the_historical_location_uses_the_library_measures(tiny_fleet, tmp_path):
    """A fake q-study run directory: one inception, one cell, frames in the study's layout."""
    out, _, _ = tiny_fleet
    from quantark.backtest.simulation.results import EnsembleResults

    results = EnsembleResults.from_dir(out / "cells" / "term_flat_q__front")
    run_dir = tmp_path / "runs" / "2024-01" / "term_flat_q__front"
    run_dir.mkdir(parents=True)
    states = results.path_states(0).set_index("date")
    states.to_csv(run_dir / "states.csv")
    trades = results.path_trades(0)
    (trades.set_index("date") if len(trades) else trades).to_csv(run_dir / "trades.csv")
    for name in ("greeks", "rebalances", "actions"):
        pd.DataFrame(index=pd.DatetimeIndex([], name="date")).to_csv(run_dir / f"{name}.csv")
    C.write_json(run_dir / "run_summary.json", {"inception": "2024-01-02", "model": "term_flat_q", "hedge": "front",
                                                "notional": results.notional})
    agg = S03.aggregate(out, es_level=0.25, historical_dir=tmp_path)
    hist = agg["historical"]
    assert hist["available"] and len(hist["rows"]) == 1
    row = hist["rows"][0]
    assert row["cell"] == "term_flat_q__front" and row["inception"] == "2024-01-02"
    assert row["terminal_pnl_bp"] == pytest.approx(results.summary["terminal_pnl_bp"].iloc[0])
    assert 0.0 <= row["terminal_pnl_percentile"] <= 100.0


def test_engine_config_rejects_a_model_whose_carry_contract_is_not_the_hedge(monkeypatch):
    """The simulation inverts the ACTIVE contract; a q-study model with its own carry contract would silently disagree."""
    from types import SimpleNamespace
    from quantark.util.exceptions import ValidationError

    base = C.Q.Q_MODELS["term_flat_q"]
    far = SimpleNamespace(dividend_source=base.dividend_source, extrapolation=base.extrapolation,
                          min_tenor_days=base.min_tenor_days, dividend_policy="far")
    monkeypatch.setitem(C.Q.Q_MODELS, "flat_from_far_fake", far)
    with pytest.raises(ValidationError, match="carry contract"):
        C.engine_config("flat_from_far_fake", "quad", quad_grid=101)


def test_a_run_that_misses_its_gate_is_recorded_not_raised(tiny_fleet, tmp_path):
    """The fleet records a failed gate in run.json and persists no results, then carries on with the next run."""
    out, _, coupon = tiny_fleet
    bootstrap, _, _ = S01.load_paths(out)
    terms = fixture_terms(bootstrap.dates)
    product = C.Q.build_product(terms, float(bootstrap.spot[0, 0]), coupon.coupon)
    strict = dict(sample_states=64, pv_tolerance_bp=0.0, delta_tolerance_hands=0.0)
    cfg = S02.cell_config(product, C.MODELS[0], "front", provider="life_surface", gate_override=strict, **CELL)
    run = S02.run_cell(bootstrap.take([1]), cfg, tmp_path / "strict", resume=False, oracle_paths=[0])
    assert run["failed"] and not run["skipped"] and not run["gate"]["passed"] and run["oracle"] == []
    assert not (tmp_path / "strict" / "manifest.json").exists() and (tmp_path / "strict" / "run.json").exists()
    assert S02.parse_args(["--provider", "exact"]).provider == "exact"


def test_an_oracle_path_whose_rerun_misses_the_gate_is_a_failed_oracle_entry(tiny_fleet, tmp_path, monkeypatch):
    """The oracle re-runs the ensemble on one path with its own gate; a failure there is recorded, not raised."""
    from quantark.backtest.simulation.pricing.base import GateFailure, GateReport

    out, _, coupon = tiny_fleet
    bootstrap, _, _ = S01.load_paths(out)
    terms = fixture_terms(bootstrap.dates)
    product = C.Q.build_product(terms, float(bootstrap.spot[0, 0]), coupon.coupon)
    report = GateReport(mode="ladder", sampled=5, max_pv_gap_bp=19.1, max_delta_gap_hands=0.6, passed=False)

    def failing_oracle(*args, **kwargs):
        raise GateFailure(report)

    monkeypatch.setattr(S02, "run_oracle", failing_oracle)
    cfg = S02.cell_config(product, C.MODELS[0], "front", provider="ladder", gate_override=FIXTURE_LADDER_GATE, **CELL)
    run = S02.run_cell(bootstrap.take([1]), cfg, tmp_path / "oracle_gate", resume=False, oracle_paths=[0])
    assert not run["failed"] and run["gate"]["passed"]
    assert run["oracle"] == [{"path": 0, "passed": False, "gate": report.as_dict()}]
    assert (tmp_path / "oracle_gate" / "run.json").exists()
    # A run that died in its oracle has results and a matching config but no run.json: resume
    # reuses the persisted results and runs only the oracle, never the ensemble again.
    (tmp_path / "oracle_gate" / "run.json").unlink()
    monkeypatch.setattr(S02, "run_ensemble", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ensemble re-run")))
    calls = []
    monkeypatch.setattr(S02, "run_oracle", lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(GateFailure(report)))
    again = S02.run_cell(bootstrap.take([1]), cfg, tmp_path / "oracle_gate", resume=True, oracle_paths=[0])
    assert len(calls) == 1 and not again["skipped"] and again["resumed_results"] and not again["failed"]
    assert (tmp_path / "oracle_gate" / "run.json").exists()


def test_a_failed_run_is_skipped_on_resume(tiny_fleet, tmp_path):
    out, _, coupon = tiny_fleet
    bootstrap, _, _ = S01.load_paths(out)
    product = C.Q.build_product(fixture_terms(bootstrap.dates), float(bootstrap.spot[0, 0]), coupon.coupon)
    strict = dict(sample_states=64, pv_tolerance_bp=0.0, delta_tolerance_hands=0.0)
    cfg = S02.cell_config(product, C.MODELS[0], "front", provider="life_surface", gate_override=strict, **CELL)
    first = S02.run_cell(bootstrap.take([1]), cfg, tmp_path / "strict", resume=False, oracle_paths=[])
    assert first["failed"] and (tmp_path / "strict" / "config.json").exists()
    again = S02.run_cell(bootstrap.take([1]), cfg, tmp_path / "strict", resume=True, oracle_paths=[])
    assert again["failed"] and again["skipped"]


def test_the_cli_runs_every_cell_of_the_quick_grid(tiny_fleet, tmp_path):
    """Stage 02's main over two cells on exact QUAD: every run gets a run.json (the oracle list survives the first cell)."""
    import shutil
    out, _, _ = tiny_fleet
    shutil.copytree(out / "paths", tmp_path / "paths")
    rc = S02.main(["--out-dir", str(tmp_path), "--provider", "exact", "--cells", f"{C.MODELS[0]}:front", "term_flat_q:front",
                   "--check-paths", "0", "--oracle-paths", "1", "--quad-grid", "101",
                   "--maturity-months", "1", "--lockout-months", "1"])
    assert rc == 0
    for name in (C.BASELINE_CELL, C.BASELINE_CELL + "__stress", "term_flat_q__front", "term_flat_q__front__stress"):
        run = C.read_json(tmp_path / "cells" / name / "run.json")
        assert not run["failed"] and run["gate"]["max_pv_gap_bp"] == 0.0, name
    fleet = C.read_json(tmp_path / "fleet_manifest.json")
    assert len(fleet["runs"]) == 4 and fleet["runs"]["term_flat_q__front"]["oracle"][0]["passed"]
