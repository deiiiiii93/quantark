"""Tests for example/mo_volmodels/13_aggregate_and_report.py (Phase 5).

Builds a synthetic fleet on disk (manifest + per-run CSV/JSON artifacts) and
checks that the aggregator computes the metrics it claims, that the paired
comparison really is paired, and that the report renders - including the
partial-result and no-edge cases, which are the ones most likely to be
quietly mis-stated.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "example/mo_volmodels/13_aggregate_and_report.py"

spec = importlib.util.spec_from_file_location("aggregate_and_report_13", MODULE_PATH)
s13 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = s13
spec.loader.exec_module(s13)

NOTIONAL = 50_000_000.0


# ---------------------------------------------------------------------------
# Synthetic fleet builder
# ---------------------------------------------------------------------------

def _write_run(
    root: Path,
    inception: str,
    variant: str,
    *,
    total_pnl: float,
    costs: float = 25_000.0,
    residual_delta: float = 10_000.0,
    n_days: int = 5,
    n_trades: int = 4,
    lifecycle=None,
    calibration=None,
):
    run_dir = root / "runs" / inception / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2023-05-04", periods=n_days, freq="D")

    # Column sets AND accounting identities mirror what
    # AutocallableBacktestEngine actually writes, so both the completeness
    # verifier and the sanity checker are exercised against a faithful run:
    #   total_pnl        = product_pnl + hedge_pnl - transaction_costs
    #   cash             = cashflows - transaction_costs
    #   portfolio_value  = product_mtm + hedge_mtm + cash
    #   futures_contracts= cumulative traded quantity
    gross = [(total_pnl + costs) * (i + 1) / n_days for i in range(n_days)]
    costs_cum = [costs * (i + 1) / n_days for i in range(n_days)]
    product_pnl = [0.6 * g for g in gross]
    hedge_pnl = [0.4 * g for g in gross]
    cash = [-c for c in costs_cum]  # cashflows are 0 in the fixture
    pnl = [p + h - c for p, h, c in zip(product_pnl, hedge_pnl, costs_cum)]
    position = [float(min(i + 1, n_trades)) for i in range(n_days)]
    pd.DataFrame(
        {
            "portfolio_value": [p + h + c for p, h, c in zip(product_pnl, hedge_pnl, cash)],
            "product_mtm": product_pnl,
            "hedge_mtm": hedge_pnl,
            "cash": cash,
            "cashflows": [0.0] * n_days,
            "product_pnl": product_pnl,
            "hedge_pnl": hedge_pnl,
            "transaction_costs": costs_cum,
            "total_pnl": pnl,
            "spot": [6700.0] * n_days,
            "volatility": [0.21] * n_days,
            "rate": [0.02] * n_days,
            "basis_yield": [-0.08] * n_days,
            "implied_q": [0.09] * n_days,
            "pricing_q": [0.09] * n_days,
            "active_contract": ["IM2306"] * n_days,
            "futures_price": [6650.0] * n_days,
            "futures_ttm": [0.12] * n_days,
            "futures_multiplier": [200.0] * n_days,
            "futures_contracts": position,
            "alive": [True] * n_days,
            "knocked_in": [False] * n_days,
            "knocked_out": [False] * n_days,
            "matured": [False] * n_days,
        },
        index=dates,
    ).to_csv(run_dir / "states.csv")

    pd.DataFrame(
        {
            "price": [1.0] * n_days,
            "delta": [-0.5] * n_days,
            "gamma": [-0.001] * n_days,
            "product_position_delta": [-500.0] * n_days,
            "product_position_gamma": [-1.0] * n_days,
            "post_hedge_delta_cash_1pct": [residual_delta] * n_days,
            "pre_hedge_delta_cash_1pct": [residual_delta * 5.0] * n_days,
            "gamma_cash_1pct": [-2_000.0] * n_days,
            "vega": [-1_000.0] * n_days,
            "theta": [50.0] * n_days,
        },
        index=dates,
    ).to_csv(run_dir / "greeks.csv")

    pd.DataFrame(
        {
            "trade_type": ["buy"] * n_trades,
            "instrument_type": ["futures"] * n_trades,
            "contract": ["IM2306"] * n_trades,
            "quantity": [1.0] * n_trades,
            "price": [6650.0] * n_trades,
            "multiplier": [200.0] * n_trades,
            "notional": [1_200_000.0] * n_trades,
            "transaction_cost": [costs / max(n_trades, 1)] * n_trades,
            "reason": ["rebalance"] * n_trades,
        },
        index=pd.date_range("2023-05-04", periods=n_trades, freq="D"),
    ).to_csv(run_dir / "trades.csv")

    (run_dir / "calibration_records.json").write_text(json.dumps(calibration or []))
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "inception": inception,
                "variant": variant,
                "coupon": 0.15,
                "vol_model_solver": "mc" if variant.startswith("heston") else "pde",
                "elapsed_seconds": 100.0,
                "lifecycle": lifecycle
                or {
                    "knocked_out": False,
                    "knocked_in": False,
                    "matured": False,
                    "censored_at_data_end": True,
                },
            }
        )
    )


def _write_manifest(root: Path, runs, *, variants, inceptions, failures=None):
    (root).mkdir(parents=True, exist_ok=True)
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config": {"notional": NOTIONAL, "rate": 0.02, "variants": variants},
                "term_sheet": {
                    "underlying": "000852.SH",
                    "tenor_months": 36,
                    "ko_pct": 1.03,
                    "ki_pct": 0.75,
                    "lockout_months": 3,
                    "product_quantity": -1.0,
                    "coupon": "solved per inception under flat BSM (Gate G4)",
                },
                "hedge_costs": {
                    "model": "CompleteCostModel",
                    "proportional_rate": 5e-5,
                    "spread_bps": 1.0,
                },
                "gate_g2": {
                    "routes": {"heston": "mc", "heston_slv": "mc"},
                    "evidence_sha256": "abc123def456",
                },
                "inceptions": [
                    {
                        "inception": i,
                        "initial_spot": 6733.97,
                        "coupon": 0.15,
                        "atm_vol_at_inception": 0.21,
                        "futures_implied_q": 0.09,
                        "coupon_solution": {"iterations": 1, "pv": 0.0, "solved": True},
                    }
                    for i in inceptions
                ],
                "runs": runs,
                "failures": failures or [],
                "counts": {
                    "inceptions": len(inceptions),
                    "variants": len(variants),
                    "runs_expected": len(inceptions) * len(variants),
                    "runs_completed": len(runs),
                    "runs_failed": len(failures or []),
                    "censored_at_data_end": len(runs),
                    "knocked_out": 0,
                    "matured": 0,
                },
            }
        )
    )


@pytest.fixture
def fleet(tmp_path):
    """Two inceptions x three variants, with a known PnL ordering."""
    root = tmp_path / "fleet"
    inceptions = ["2023-05-04", "2023-06-01"]
    variants = ["flat_bsm", "ts_bsm", "heston"]
    # flat_bsm baseline; ts_bsm beats it by +0.2% of notional on both dates;
    # heston loses by -0.1% on both.  Paired deltas are therefore exact.
    pnl = {
        ("2023-05-04", "flat_bsm"): 500_000.0,
        ("2023-05-04", "ts_bsm"): 600_000.0,
        ("2023-05-04", "heston"): 450_000.0,
        ("2023-06-01", "flat_bsm"): 200_000.0,
        ("2023-06-01", "ts_bsm"): 300_000.0,
        ("2023-06-01", "heston"): 150_000.0,
    }
    runs = []
    for inception in inceptions:
        for variant in variants:
            calibration = None
            if variant == "heston":
                calibration = [
                    {
                        "surface_sha": f"sha{i}",
                        "overall_rmse_iv": 0.004 + 0.001 * i,
                        "bound_hits": ["kappa"] if i % 2 == 0 else [],
                        "feller_satisfied": i % 3 != 0,
                    }
                    for i in range(5)
                ]
            _write_run(
                root,
                inception,
                variant,
                total_pnl=pnl[(inception, variant)],
                residual_delta=5_000.0 if variant == "ts_bsm" else 10_000.0,
                calibration=calibration,
            )
            runs.append({"inception": inception, "variant": variant})
    _write_manifest(root, runs, variants=variants, inceptions=inceptions)
    return root


# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------

def test_pnl_is_read_from_the_last_day_not_averaged(fleet):
    agg = s13.aggregate(fleet)
    row = next(
        r for r in agg["per_run"]
        if r["inception"] == "2023-05-04" and r["variant"] == "flat_bsm"
    )
    assert row["total_pnl"] == pytest.approx(500_000.0)
    assert row["total_pnl_pct_notional"] == pytest.approx(1.0)


def test_cost_drag_is_expressed_against_notional(fleet):
    agg = s13.aggregate(fleet)
    row = agg["per_run"][0]
    assert row["transaction_costs"] == pytest.approx(25_000.0)
    assert row["cost_drag_pct_notional"] == pytest.approx(100.0 * 25_000.0 / NOTIONAL)


def test_residual_delta_uses_post_hedge_not_pre_hedge(fleet):
    """Hedge quality must measure what is LEFT after rebalancing."""
    agg = s13.aggregate(fleet)
    row = next(r for r in agg["per_run"] if r["variant"] == "ts_bsm")
    assert row["residual_delta_cash_rms"] == pytest.approx(5_000.0)
    assert row["pre_hedge_delta_cash_rms"] == pytest.approx(25_000.0)


def test_max_drawdown_is_zero_for_a_monotonically_rising_pnl(fleet):
    agg = s13.aggregate(fleet)
    assert all(r["pnl_max_drawdown"] == pytest.approx(0.0) for r in agg["per_run"])


def test_max_drawdown_detects_a_real_peak_to_trough():
    assert s13._max_drawdown([0.0, 100.0, 40.0, 90.0]) == pytest.approx(-60.0)
    assert s13._max_drawdown([]) is None


def test_metrics_never_invent_values_for_missing_columns(tmp_path):
    run_dir = tmp_path / "runs" / "2023-05-04" / "flat_bsm"
    run_dir.mkdir(parents=True)
    pd.DataFrame({"spot": [1.0]}).to_csv(run_dir / "states.csv")
    frames = s13.load_run_frames(run_dir)
    m = s13.metrics_for_run(
        inception="2023-05-04", variant="flat_bsm", notional=NOTIONAL, frames=frames
    )
    assert m["total_pnl"] is None
    assert m["total_pnl_pct_notional"] is None
    assert m["residual_delta_cash_rms"] is None
    assert m["n_trades"] == 0


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------

def test_paired_deltas_are_same_inception_differences(fleet):
    agg = s13.aggregate(fleet)
    pairs = {(p["inception"], p["variant"]): p for p in agg["paired_vs_baseline"]}
    # ts_bsm beat flat_bsm by 100k on both dates = +0.2% of notional.
    assert pairs[("2023-05-04", "ts_bsm")]["d_pnl_pct_notional"] == pytest.approx(0.2)
    assert pairs[("2023-06-01", "ts_bsm")]["d_pnl_pct_notional"] == pytest.approx(0.2)
    assert pairs[("2023-05-04", "heston")]["d_pnl_pct_notional"] == pytest.approx(-0.1)


def test_baseline_is_not_compared_against_itself(fleet):
    agg = s13.aggregate(fleet)
    assert all(p["variant"] != "flat_bsm" for p in agg["paired_vs_baseline"])


def test_win_rates_count_the_right_direction(fleet):
    agg = s13.aggregate(fleet)
    summary = agg["paired_summary"]
    assert summary["ts_bsm"]["pnl_win_rate"] == pytest.approx(1.0)
    assert summary["heston"]["pnl_win_rate"] == pytest.approx(0.0)
    # ts_bsm leaves LESS residual delta -> it wins on hedge quality.
    assert summary["ts_bsm"]["hedge_win_rate"] == pytest.approx(1.0)
    assert summary["heston"]["hedge_win_rate"] == pytest.approx(0.0)


def test_inception_without_a_baseline_run_is_skipped_not_imputed():
    per_run = [
        {"inception": "2023-05-04", "variant": "ts_bsm", "total_pnl_pct_notional": 1.0,
         "cost_drag_pct_notional": 0.05, "residual_delta_cash_rms_pct_notional": 0.02,
         "n_trades": 4, "lifecycle": {}},
    ]
    assert s13.paired_comparisons(per_run) == []


def test_paired_diff_is_none_when_either_side_is_missing():
    assert s13._diff(None, 1.0) is None
    assert s13._diff(1.0, None) is None
    assert s13._diff(float("nan"), 1.0) is None
    assert s13._diff(3.0, 1.0) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Calibration quality
# ---------------------------------------------------------------------------

def test_calibration_quality_surfaces_bound_hits_and_feller(fleet):
    agg = s13.aggregate(fleet)
    row = next(r for r in agg["per_run"] if r["variant"] == "heston")
    calib = row["calibration"]
    assert calib["n_records"] == 5
    assert calib["n_unique_surfaces"] == 5
    # i = 0, 2, 4 carry bound hits
    assert calib["n_bound_hits"] == 3
    assert calib["bound_hit_fraction"] == pytest.approx(3 / 5)
    # feller_satisfied is False when i % 3 == 0 -> i = 0, 3
    assert calib["n_feller_violated"] == 2
    assert calib["rmse_iv"]["mean"] == pytest.approx(0.006)


def test_variants_without_calibration_report_zero_records(fleet):
    agg = s13.aggregate(fleet)
    row = next(r for r in agg["per_run"] if r["variant"] == "flat_bsm")
    assert row["calibration"] == {"n_records": 0}


# ---------------------------------------------------------------------------
# Aggregation + outputs
# ---------------------------------------------------------------------------

def test_variant_summary_covers_every_variant(fleet):
    agg = s13.aggregate(fleet)
    assert set(agg["variant_summary"]) == {"flat_bsm", "ts_bsm", "heston"}
    assert agg["variant_summary"]["ts_bsm"]["n_runs"] == 2
    assert agg["variant_summary"]["ts_bsm"]["pnl_pct_notional"]["mean"] == pytest.approx(0.9)


def test_variants_are_reported_in_the_canonical_order(fleet):
    assert s13.aggregate(fleet)["variants"] == ["flat_bsm", "ts_bsm", "heston"]


def test_missing_run_artifacts_are_reported_not_silently_dropped(fleet):
    import shutil

    shutil.rmtree(fleet / "runs" / "2023-06-01" / "heston")
    agg = s13.aggregate(fleet)
    assert agg["missing_runs"] == [{"inception": "2023-06-01", "variant": "heston"}]
    assert len(agg["per_run"]) == 5


def test_aggregate_is_actionable_when_the_fleet_never_ran(tmp_path):
    with pytest.raises(FileNotFoundError, match="run stage 12"):
        s13.aggregate(tmp_path)


def test_tables_and_report_are_written(fleet, tmp_path):
    agg = s13.aggregate(fleet)
    out = tmp_path / "agg"
    paths = s13.write_tables(agg, out)
    assert set(paths) == {"per_run", "variant_summary", "paired"}
    per_run = pd.read_csv(paths["per_run"])
    assert len(per_run) == 6
    assert {"pnl_pct_notional", "cost_drag_pct_notional", "censored"} <= set(per_run.columns)
    summary = pd.read_csv(paths["variant_summary"])
    assert len(summary) == 3


def test_report_renders_the_headline_tables(fleet):
    html = s13.build_report(s13.aggregate(fleet))
    assert html.startswith("<!doctype html>")
    assert "Does vol-model sophistication pay" in html
    for label in ("Flat BSM", "TS BSM", "Heston"):
        assert label in html
    assert "paired" in html.lower()
    assert "abc123def456"[:16] in html, "gate evidence must be cited"
    assert "flat in total variance" in html, "extrapolation caveat must be stated"


def test_report_flags_a_partial_run(fleet):
    """A run missing whole variants must say so, loudly, at the top."""
    html = s13.build_report(s13.aggregate(fleet))
    assert "PARTIAL RESULT" in html
    assert "Local Vol" in html and "Heston-SLV" in html  # named as absent


def test_report_states_a_null_result_as_a_null_result(tmp_path):
    """An edge inside two standard errors must NOT be reported as an edge."""
    root = tmp_path / "noedge"
    inceptions = [f"2023-{m:02d}-01" for m in range(1, 7)]
    variants = ["flat_bsm", "ts_bsm"]
    runs = []
    # ts_bsm alternates +/- around flat_bsm: mean edge ~0, large spread.
    for i, inception in enumerate(inceptions):
        _write_run(root, inception, "flat_bsm", total_pnl=500_000.0)
        _write_run(
            root, inception, "ts_bsm",
            total_pnl=500_000.0 + (400_000.0 if i % 2 else -400_000.0),
        )
        runs += [
            {"inception": inception, "variant": "flat_bsm"},
            {"inception": inception, "variant": "ts_bsm"},
        ]
    _write_manifest(root, runs, variants=variants, inceptions=inceptions)

    agg = s13.aggregate(root)
    edge = agg["paired_summary"]["ts_bsm"]["d_pnl_pct_notional"]
    # Constructed: alternating +/-0.8% of notional -> mean ~0, wide spread.
    assert abs(edge["mean"]) < 0.05
    assert edge["stdev"] > 0.5
    assert abs(edge["mean"]) < 2.0 * edge["stdev"] / (edge["n"] ** 0.5)

    html = s13.build_report(agg)
    assert "<em>not</em> distinguishable from zero" in html
    assert "outside two standard errors" not in html


def test_report_calls_a_real_edge_a_real_edge(tmp_path):
    """The mirror case: a consistent edge must be reported as significant."""
    root = tmp_path / "edge"
    inceptions = [f"2023-{m:02d}-01" for m in range(1, 7)]
    runs = []
    for inception in inceptions:
        _write_run(root, inception, "flat_bsm", total_pnl=500_000.0)
        _write_run(root, inception, "ts_bsm", total_pnl=600_000.0)  # +0.2% every time
        runs += [
            {"inception": inception, "variant": "flat_bsm"},
            {"inception": inception, "variant": "ts_bsm"},
        ]
    _write_manifest(root, runs, variants=["flat_bsm", "ts_bsm"], inceptions=inceptions)

    html = s13.build_report(s13.aggregate(root))
    assert "outside two standard errors" in html
    assert "<em>not</em> distinguishable from zero" not in html


def test_report_survives_an_empty_fleet(tmp_path):
    root = tmp_path / "empty"
    _write_manifest(root, [], variants=[], inceptions=[])
    agg = s13.aggregate(root)
    html = s13.build_report(agg)
    assert "<!doctype html>" in html


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def test_helpers_ignore_none_and_nan():
    values = [1.0, None, float("nan"), 3.0]
    assert s13._mean(values) == pytest.approx(2.0)
    assert s13._median(values) == pytest.approx(2.0)
    assert s13._rms([3.0, 4.0]) == pytest.approx(3.5355339, rel=1e-6)
    assert s13._distribution(values)["n"] == 2


def test_helpers_return_none_rather_than_zero_when_empty():
    assert s13._mean([]) is None
    assert s13._median([]) is None
    assert s13._stdev([1.0]) is None
    assert s13._rms([]) is None
    dist = s13._distribution([])
    assert dist["n"] == 0 and dist["mean"] is None and dist["min"] is None


def test_formatter_renders_missing_values_as_a_dash():
    assert s13._fmt(None) == "&mdash;"
    assert s13._fmt(float("nan")) == "&mdash;"
    assert s13._fmt(1234.5678, 2) == "1,234.57"


# ---------------------------------------------------------------------------
# Output completeness (Task 6.2)
# ---------------------------------------------------------------------------

def test_a_well_formed_run_is_complete(fleet):
    check = s13.verify_run_completeness(
        fleet / "runs" / "2023-05-04" / "flat_bsm", variant="flat_bsm"
    )
    assert check["complete"], check["issues"]
    assert check["n_days"] == 5


def test_missing_greeks_file_is_reported(fleet):
    run_dir = fleet / "runs" / "2023-05-04" / "flat_bsm"
    (run_dir / "greeks.csv").unlink()
    check = s13.verify_run_completeness(run_dir, variant="flat_bsm")
    assert not check["complete"]
    assert any("greeks_path" in issue for issue in check["issues"])
    assert check["categories"]["greeks_path"] is False


def test_missing_pnl_columns_are_reported(fleet):
    run_dir = fleet / "runs" / "2023-05-04" / "flat_bsm"
    frame = pd.read_csv(run_dir / "states.csv", index_col=0).drop(columns=["hedge_pnl"])
    frame.to_csv(run_dir / "states.csv")
    check = s13.verify_run_completeness(run_dir, variant="flat_bsm")
    assert not check["complete"]
    assert any("hedge_pnl" in issue for issue in check["issues"])


def test_row_count_mismatch_between_states_and_greeks_is_caught(fleet):
    run_dir = fleet / "runs" / "2023-05-04" / "flat_bsm"
    greeks = pd.read_csv(run_dir / "greeks.csv", index_col=0).iloc[:2]
    greeks.to_csv(run_dir / "greeks.csv")
    check = s13.verify_run_completeness(run_dir, variant="flat_bsm")
    assert not check["complete"]
    assert any("greeks.csv has 2 rows" in issue for issue in check["issues"])


def test_declared_day_count_must_match_the_states_frame(fleet):
    check = s13.verify_run_completeness(
        fleet / "runs" / "2023-05-04" / "flat_bsm", variant="flat_bsm", expected_days=99
    )
    assert not check["complete"]
    assert any("run summary claims 99" in issue for issue in check["issues"])


def test_empty_trades_are_allowed(fleet):
    """A run can end before any rebalance clears the rounding threshold."""
    run_dir = fleet / "runs" / "2023-05-04" / "flat_bsm"
    empty = pd.read_csv(run_dir / "trades.csv", index_col=0).iloc[:0]
    empty.to_csv(run_dir / "trades.csv")
    check = s13.verify_run_completeness(run_dir, variant="flat_bsm")
    assert check["complete"], check["issues"]


def test_calibrated_variant_without_records_is_incomplete(fleet):
    run_dir = fleet / "runs" / "2023-05-04" / "heston"
    (run_dir / "calibration_records.json").write_text("[]")
    check = s13.verify_run_completeness(run_dir, variant="heston")
    assert not check["complete"]
    assert any("calibration_records" in issue for issue in check["issues"])


def test_bsm_variant_is_not_asked_for_calibration_records(fleet):
    run_dir = fleet / "runs" / "2023-05-04" / "flat_bsm"
    assert not (json.loads((run_dir / "calibration_records.json").read_text()))
    check = s13.verify_run_completeness(run_dir, variant="flat_bsm")
    assert check["complete"]
    assert check["categories"]["calibration_records"] is True


def test_lv_surface_records_required_only_for_lv_bearing_variants(fleet):
    """localvol and heston_slv carry a leverage/LV surface; plain heston does not."""
    run_dir = fleet / "runs" / "2023-05-04" / "heston"
    # heston records have no lv_min/lv_max, and must still pass.
    check = s13.verify_run_completeness(run_dir, variant="heston")
    assert check["complete"], check["issues"]

    # The same records under the localvol variant must fail the LV category.
    check_lv = s13.verify_run_completeness(run_dir, variant="localvol")
    assert not check_lv["complete"]
    assert any("lv_surface_records" in issue for issue in check_lv["issues"])


def test_fleet_completeness_rolls_up_every_run(fleet):
    agg = s13.aggregate(fleet)
    comp = agg["completeness"]
    assert comp["n_runs_checked"] == 6
    assert comp["all_complete"], comp["incomplete"]
    assert comp["n_incomplete"] == 0


def test_fleet_completeness_reports_a_broken_run(fleet):
    (fleet / "runs" / "2023-06-01" / "ts_bsm" / "greeks.csv").unlink()
    comp = s13.aggregate(fleet)["completeness"]
    assert not comp["all_complete"]
    assert comp["n_incomplete"] == 1
    assert comp["incomplete"][0]["variant"] == "ts_bsm"


# ---------------------------------------------------------------------------
# Outcome-concentration caveat
# ---------------------------------------------------------------------------

def _all_ko_fleet(tmp_path, *, ki=False):
    root = tmp_path / "allko"
    inceptions = ["2023-05-04", "2023-06-01", "2023-07-03"]
    runs = []
    for inception in inceptions:
        for variant in ("flat_bsm", "ts_bsm"):
            _write_run(
                root, inception, variant, total_pnl=400_000.0,
                lifecycle={
                    "knocked_out": True,
                    "knocked_in": ki,
                    "matured": False,
                    "censored_at_data_end": False,
                },
            )
            runs.append({"inception": inception, "variant": variant})
    _write_manifest(root, runs, variants=["flat_bsm", "ts_bsm"], inceptions=inceptions)
    return root


def test_all_knock_out_sample_is_called_out(tmp_path):
    """A sample with only KO outcomes cannot speak to knocked-in maturity."""
    html = s13.build_report(s13.aggregate(_all_ko_fleet(tmp_path)))
    assert "Every trade in this sample knocked out" in html
    assert "knocked in at maturity" in html
    assert "no edge on" in html


def test_ki_then_ko_count_is_reported(tmp_path):
    html = s13.build_report(s13.aggregate(_all_ko_fleet(tmp_path, ki=True)))
    assert "3 knocked in at some point" in html


def test_mixed_outcome_sample_gets_no_special_warning(fleet):
    """The default fixture is censored, not all-KO - no banner expected."""
    html = s13.build_report(s13.aggregate(fleet))
    assert "Every trade in this sample knocked out" not in html


# ---------------------------------------------------------------------------
# Run sanity invariants (Gate G3)
# ---------------------------------------------------------------------------

def test_consistent_run_passes_every_invariant(fleet):
    report = s13.sanity_check_run(fleet / "runs" / "2023-05-04" / "flat_bsm")
    assert report["sane"], report["issues"]
    assert report["checks"]["pnl_identity_max_abs"] < 1e-6


def _break_states(run_dir, column, mutate):
    frame = pd.read_csv(run_dir / "states.csv", index_col=0)
    frame[column] = mutate(frame[column])
    frame.to_csv(run_dir / "states.csv")


def test_broken_pnl_decomposition_is_caught(fleet):
    run_dir = fleet / "runs" / "2023-05-04" / "flat_bsm"
    _break_states(run_dir, "hedge_pnl", lambda s: s + 12_345.0)
    report = s13.sanity_check_run(run_dir)
    assert not report["sane"]
    assert any("PnL identity" in i for i in report["issues"])


def test_broken_portfolio_identity_is_caught(fleet):
    run_dir = fleet / "runs" / "2023-05-04" / "flat_bsm"
    _break_states(run_dir, "cash", lambda s: s + 999.0)
    report = s13.sanity_check_run(run_dir)
    assert not report["sane"]
    assert any("identity" in i for i in report["issues"])


def test_decreasing_cumulative_costs_are_caught(fleet):
    run_dir = fleet / "runs" / "2023-05-04" / "flat_bsm"
    _break_states(run_dir, "transaction_costs", lambda s: s[::-1].values)
    report = s13.sanity_check_run(run_dir)
    assert not report["sane"]
    assert any("costs decrease" in i or "identity" in i for i in report["issues"])


def test_cost_reconciliation_against_trades(fleet):
    run_dir = fleet / "runs" / "2023-05-04" / "flat_bsm"
    trades = pd.read_csv(run_dir / "trades.csv", index_col=0)
    trades["transaction_cost"] = trades["transaction_cost"] * 3.0
    trades.to_csv(run_dir / "trades.csv")
    report = s13.sanity_check_run(run_dir)
    assert not report["sane"]
    assert any("cost reconciliation" in i for i in report["issues"])


def test_position_tracking_mismatch_is_caught(fleet):
    run_dir = fleet / "runs" / "2023-05-04" / "flat_bsm"
    _break_states(run_dir, "futures_contracts", lambda s: s * 0.0 + 77.0)
    report = s13.sanity_check_run(run_dir)
    assert not report["sane"]
    assert any("position tracking" in i for i in report["issues"])


def test_resurrected_trade_is_caught(fleet):
    run_dir = fleet / "runs" / "2023-05-04" / "flat_bsm"
    _break_states(run_dir, "alive", lambda s: [True, False, False, True, True])
    report = s13.sanity_check_run(run_dir)
    assert not report["sane"]
    assert any("came back to life" in i for i in report["issues"])


def test_unset_knockout_flag_is_caught(fleet):
    run_dir = fleet / "runs" / "2023-05-04" / "flat_bsm"
    _break_states(run_dir, "knocked_out", lambda s: [False, True, True, False, False])
    report = s13.sanity_check_run(run_dir)
    assert not report["sane"]
    assert any("turns back off" in i for i in report["issues"])


def test_knockout_without_a_ko_action_row_is_caught(fleet):
    run_dir = fleet / "runs" / "2023-05-04" / "flat_bsm"
    _break_states(run_dir, "knocked_out", lambda s: [False, False, False, False, True])
    report = s13.sanity_check_run(run_dir)
    assert not report["sane"]
    assert any("no action log" in i or "no KO action row" in i for i in report["issues"])


def test_hedge_that_increases_delta_is_caught(fleet):
    """A sign error in the rebalance would show up exactly here."""
    run_dir = fleet / "runs" / "2023-05-04" / "flat_bsm"
    greeks = pd.read_csv(run_dir / "greeks.csv", index_col=0)
    greeks["post_hedge_delta_cash_1pct"] = greeks["pre_hedge_delta_cash_1pct"] * 2.0
    greeks.to_csv(run_dir / "greeks.csv")
    report = s13.sanity_check_run(run_dir)
    assert not report["sane"]
    assert any("increased |delta|" in i for i in report["issues"])


def test_nan_in_a_headline_column_is_caught(fleet):
    run_dir = fleet / "runs" / "2023-05-04" / "flat_bsm"
    _break_states(run_dir, "spot", lambda s: [6700.0, float("nan"), 6700.0, 6700.0, 6700.0])
    report = s13.sanity_check_run(run_dir)
    assert not report["sane"]
    assert any("NaN" in i for i in report["issues"])


def test_missing_run_is_reported_as_unsound(tmp_path):
    report = s13.sanity_check_run(tmp_path / "nope")
    assert not report["sane"]
    assert "states.csv missing" in report["issues"]


def test_fleet_sanity_rolls_up(fleet):
    comp = s13.aggregate(fleet)["completeness"]
    assert comp["all_sane"], comp["sanity_failures"]
    assert comp["n_sane"] == 6


def test_fleet_sanity_reports_the_offending_run(fleet):
    _break_states(
        fleet / "runs" / "2023-06-01" / "heston", "hedge_pnl", lambda s: s + 1_000.0
    )
    comp = s13.aggregate(fleet)["completeness"]
    assert not comp["all_sane"]
    assert len(comp["sanity_failures"]) == 1
    assert comp["sanity_failures"][0]["variant"] == "heston"
