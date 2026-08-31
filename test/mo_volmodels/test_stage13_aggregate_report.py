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
    maturity_date=None,
    front_load: float = 0.0,
    product_pnl_final=None,
    settle: bool = False,
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
    # front_load moves gross PnL onto day 1: 0.0 accrues it linearly over the
    # replay (all of it earned while hedging), 1.0 books it all at inception
    # (a pure mark-to-model, nothing added afterwards).
    shape = [
        front_load + (1.0 - front_load) * (i + 1) / n_days for i in range(n_days)
    ]
    gross = [(total_pnl + costs) * f for f in shape]
    costs_cum = [costs * (i + 1) / n_days for i in range(n_days)]
    # product_pnl_final pins where the CONTRACT leg ends and lets the hedge leg
    # absorb the rest.  A shared realized path forces exactly that shape: two
    # models of one trade settle the SAME coupon and differ only in what their
    # hedges earned, so the contract leg must be settable independently of the
    # total.  Default keeps the historic 60/40 split.
    if product_pnl_final is None:
        product_pnl = [0.6 * g for g in gross]
    else:
        product_pnl = [float(product_pnl_final) * f for f in shape]
    hedge_pnl = [g - p for g, p in zip(gross, product_pnl)]
    # settle=True terminates the trade: the mark moves into cashflows on the
    # last day and product_pnl (their sum) does NOT move -- settlement is a
    # transfer, not income.  settle=False leaves the run censored with the
    # whole value still marked to model.
    cashflows = [0.0] * n_days
    product_mtm = list(product_pnl)
    if settle and n_days:
        cashflows[-1] = product_pnl[-1]
        product_mtm[-1] = 0.0
    cash = [f - c for f, c in zip(cashflows, costs_cum)]
    pnl = [p + h - c for p, h, c in zip(product_pnl, hedge_pnl, costs_cum)]
    position = [float(min(i + 1, n_trades)) for i in range(n_days)]
    pd.DataFrame(
        {
            "portfolio_value": [
                m + h + c for m, h, c in zip(product_mtm, hedge_pnl, cash)
            ],
            "product_mtm": product_mtm,
            "hedge_mtm": hedge_pnl,
            "cash": cash,
            "cashflows": cashflows,
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
                "maturity_date": maturity_date or f"{int(inception[:4]) + 3}{inception[4:]}",
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
    assert calib["rmse_iv"]["mean"] == pytest.approx(0.006)
    # These records carry no feller_ratio, so no regime can be ranked -- the
    # screen must say "unknown", not report a reassuring zero violated.
    assert calib["feller_buckets"]["unknown"] == 5
    assert calib["feller_violated_fraction"] is None
    assert calib["sigma_collapse_fraction"] is None
    # feller_satisfied is False when i % 3 == 0 -> i = 0, 3.  A flag saying
    # unsatisfied under an enforcing calibration is an enforcement breach,
    # which is a different claim from a ratio-ranked regime.
    assert calib["n_enforcement_breaches"] == 2


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


# ---------------------------------------------------------------------------
# Feller regime screen
#
# The pool this study calibrates against is enforce_feller=True, so
# ``feller_satisfied`` is True on every record BY CONSTRUCTION -- 257/257 in
# output/mo_daily_calibration/calibration_manifest.json.  Screening on that
# boolean can only ever report "clean".  Three of those same 257 fits carry
# ratios of 7.9e3, 1.7e5 and 2.3e5 with sigma pinned at its 0.001 lower bound:
# sigma-collapse, a deterministic-variance degenerate that spec section
# 7A.10(3) says must be flagged, never averaged into a `heston` result.
# The screen therefore has to read ``feller_ratio``.
# ---------------------------------------------------------------------------

def _enforced_records():
    """Records as the enforced calibration really emits them.

    Every one satisfies Feller; the ratios still span all three regimes.
    """
    ratios = [0.31, 1.00002, 1.00003, 4.894, 226022.0]
    return [
        {
            "surface_sha": f"sha{i}",
            "overall_rmse_iv": 0.004,
            "feller_ratio": ratio,
            "feller_satisfied": True,
        }
        for i, ratio in enumerate(ratios)
    ]


def test_the_feller_screen_reads_the_ratio_not_the_enforced_flag():
    calib = s13.calibration_quality(_enforced_records())
    # Screening the boolean would report zero of everything here.
    assert calib["feller_buckets"]["violated"] == 1
    assert calib["feller_buckets"]["boundary"] == 3
    assert calib["feller_buckets"]["degenerate"] == 1
    assert calib["n_feller_violated"] == 1
    assert calib["feller_violated_fraction"] == pytest.approx(1 / 5)


def test_sigma_collapse_dates_are_counted_not_averaged_away():
    calib = s13.calibration_quality(_enforced_records())
    assert calib["n_sigma_collapse"] == 1
    assert calib["sigma_collapse_fraction"] == pytest.approx(1 / 5)
    # The ratio distribution must expose the collapse magnitude, not hide it
    # behind a mean: the max is the marker.
    assert calib["feller_ratio"]["max"] == pytest.approx(226022.0)
    assert calib["feller_ratio"]["n"] == 5


def test_records_without_a_ratio_are_unknown_and_do_not_dilute_the_fractions():
    records = _enforced_records() + [
        {"surface_sha": "lv0", "overall_rmse_iv": 0.004},
        {"surface_sha": "lv1", "overall_rmse_iv": 0.004},
    ]
    calib = s13.calibration_quality(records)
    assert calib["feller_buckets"]["unknown"] == 2
    # Denominator is the records that carry a ratio, never all seven.
    assert calib["sigma_collapse_fraction"] == pytest.approx(1 / 5)
    assert calib["feller_violated_fraction"] == pytest.approx(1 / 5)


def test_a_variant_that_never_carries_a_ratio_reports_none_not_zero():
    records = [{"surface_sha": "lv0", "overall_rmse_iv": 0.004} for _ in range(3)]
    calib = s13.calibration_quality(records)
    assert calib["feller_buckets"]["unknown"] == 3
    assert calib["sigma_collapse_fraction"] is None
    assert calib["feller_violated_fraction"] is None


def test_an_enforcement_breach_is_surfaced_as_its_own_finding():
    """`violated` should be empty under enforcement; if it is not, say so.

    Per the rebaseline plan: "if it is not, the enforcement did not take and
    that is a finding".  A record whose own flag says unsatisfied is that
    finding, and it is distinct from a ratio-bucket count.
    """
    records = _enforced_records()
    records[0] = dict(records[0], feller_satisfied=False)
    calib = s13.calibration_quality(records)
    assert calib["n_enforcement_breaches"] == 1


def test_stage13_agrees_with_stage11_on_the_feller_cut_points():
    """The cut points are measured, and two stages must not drift apart.

    Stage 11 is an implementation input to the Stage 16 certification hash, so
    it is not refactored to share this constant; this test is what makes the
    duplication safe.
    """
    gate_path = ROOT / "example/mo_volmodels/11_pde_convergence_gate.py"
    gate_spec = importlib.util.spec_from_file_location("gate_11_cuts", gate_path)
    gate = importlib.util.module_from_spec(gate_spec)
    sys.modules[gate_spec.name] = gate
    gate_spec.loader.exec_module(gate)

    assert s13.FELLER_VIOLATED_BELOW == gate.FELLER_VIOLATED_BELOW
    assert s13.FELLER_DEGENERATE_ABOVE == gate.FELLER_DEGENERATE_ABOVE
    for ratio in (0.31, 0.5, 1.00002, 9.657, 10.0, 226022.0, None, float("nan")):
        assert s13.feller_bucket(ratio) == gate.feller_bucket(ratio)


def test_pooled_calibration_carries_the_sigma_collapse_fraction():
    entries = [
        s13.calibration_quality(_enforced_records()),
        s13.calibration_quality(_enforced_records()[:3]),
    ]
    pooled = s13._pooled_calibration(entries)
    assert pooled["n_records"] == 8
    # 1/5 and 0/3 -> the mean of the per-run fractions, as bound_hit does.
    assert pooled["sigma_collapse_fraction"] == pytest.approx((1 / 5 + 0.0) / 2)


def test_report_renders_the_sigma_collapse_column(tmp_path):
    """A degenerate date must be visible in the rendered report.

    Computing the metric and not printing it would leave the screen exactly
    as invisible as the enforced boolean it replaces.
    """
    root = tmp_path / "fleet"
    for variant in ("flat_bsm", "heston"):
        calibration = _enforced_records() if variant == "heston" else None
        _write_run(
            root, "2023-05-04", variant, total_pnl=500_000.0, calibration=calibration
        )
    _write_manifest(
        root,
        [{"inception": "2023-05-04", "variant": v} for v in ("flat_bsm", "heston")],
        variants=["flat_bsm", "heston"],
        inceptions=["2023-05-04"],
    )
    html = s13.build_report(s13.aggregate(root))
    assert "&sigma;-collapse" in html
    assert "max Feller ratio" in html
    # 1 of 5 heston records is degenerate, and its ratio is the max.
    assert "20.0%" in html
    assert "226,022.0000" in html


def test_the_screen_derives_the_ratio_for_slv_records_that_nest_their_heston_fit():
    """heston_slv carries its Heston fit nested, without a ratio.

    A real fleet record for `heston_slv` is
    {..., "heston": {"kappa":…, "theta":…, "sigma":…, "rho":…, "v0":…}} --
    the five raw parameters and no `feller_ratio`.  Reading only the top
    level would report "unknown" for every SLV date and leave sigma-collapse
    invisible for one of the two certified 2-D variants, which is the exact
    blind spot this screen exists to close.  SLV inherits the Heston fit, so
    it inherits its regime.
    """
    collapsed = {"kappa": 2.046, "theta": 0.05524, "sigma": 0.001, "rho": -0.3, "v0": 0.03}
    ordinary = {"kappa": 3.0, "theta": 0.0246, "sigma": 0.3844, "rho": -0.26, "v0": 0.028}
    records = [
        {"surface_sha": "slv0", "variant": "heston_slv", "heston": ordinary},
        {"surface_sha": "slv1", "variant": "heston_slv", "heston": collapsed},
    ]
    calib = s13.calibration_quality(records)
    assert calib["feller_buckets"]["unknown"] == 0
    assert calib["n_sigma_collapse"] == 1
    assert calib["sigma_collapse_fraction"] == pytest.approx(0.5)
    # 2*2.046*0.05524 / 0.001**2
    assert calib["feller_ratio"]["max"] == pytest.approx(226_042.08, rel=1e-6)


def test_a_nested_fit_with_no_usable_sigma_is_unknown_not_silently_ranked():
    records = [
        {"surface_sha": "slv0", "heston": {"kappa": 2.0, "theta": 0.05, "sigma": 0.0}},
        {"surface_sha": "slv1", "heston": {"kappa": 2.0, "theta": 0.05}},
        {"surface_sha": "slv2", "heston": "not-a-mapping"},
    ]
    calib = s13.calibration_quality(records)
    assert calib["feller_buckets"]["unknown"] == 3
    assert calib["sigma_collapse_fraction"] is None


def test_an_explicit_ratio_wins_over_the_nested_parameters():
    """The calibrator's own ratio is authoritative where it exists."""
    records = [
        {
            "surface_sha": "h0",
            "feller_ratio": 1.00002,
            "heston": {"kappa": 2.0, "theta": 0.05, "sigma": 0.001},
        }
    ]
    calib = s13.calibration_quality(records)
    assert calib["feller_buckets"]["boundary"] == 1
    assert calib["n_sigma_collapse"] == 0


def _one_variant_fleet(root, calibration):
    _write_run(root, "2023-05-04", "flat_bsm", total_pnl=500_000.0)
    _write_run(root, "2023-05-04", "heston", total_pnl=450_000.0, calibration=calibration)
    _write_manifest(
        root,
        [{"inception": "2023-05-04", "variant": v} for v in ("flat_bsm", "heston")],
        variants=["flat_bsm", "heston"],
        inceptions=["2023-05-04"],
    )
    return root


def test_an_enforcement_breach_is_called_out_in_the_report(tmp_path):
    """`enforce_feller=True` means breaches must be 0; say so when they aren't.

    A finding that lands only in the JSON is half-hidden.  A record whose own
    flag says unsatisfied means the constraint did not take on that date,
    which invalidates the assumption the whole screen rests on.
    """
    records = _enforced_records()
    records[1] = dict(records[1], feller_satisfied=False)
    html = s13.build_report(s13.aggregate(_one_variant_fleet(tmp_path / "a", records)))
    assert "FELLER ENFORCEMENT BREACH" in html
    assert "Heston" in html


def test_a_clean_run_carries_no_breach_banner(tmp_path):
    html = s13.build_report(
        s13.aggregate(_one_variant_fleet(tmp_path / "b", _enforced_records()))
    )
    assert "FELLER ENFORCEMENT BREACH" not in html


def test_a_malformed_stated_ratio_falls_through_instead_of_crashing():
    """Aggregation must not die on one bad record in a 143-CPU-hour fleet."""
    records = [
        {
            "surface_sha": "h0",
            "feller_ratio": "n/a",
            "heston": {"kappa": 3.0, "theta": 0.0246, "sigma": 0.3844},
        }
    ]
    calib = s13.calibration_quality(records)
    assert calib["feller_buckets"]["boundary"] == 1


# ---------------------------------------------------------------------------
# Certificate span audit
#
# Report-only: the ADI Greek certificate's admitted verdict is an AGGREGATE
# mean signed bias over seven archetypes against a 0.1-contract bound (each
# cell individually only reached 0.5), so it cannot be decomposed into
# per-date permissions.  The audit says whether the fleet's visited states
# stay inside the regime span those archetypes straddle, and names the ones
# that do not.  It gates nothing.
# ---------------------------------------------------------------------------

ORDINARY_FIT = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.30, "rho": -0.55}
# 2024-10-10, ratio 35,048 -- 18x past the sigma_collapse archetype at 1,898.
OUT_OF_SPAN_FIT = {
    "v0": 0.15899, "kappa": 2.068, "theta": 0.018037, "sigma": 0.001459, "rho": -0.00404,
}


def _dated(fits):
    return [
        {"date": day, "surface_sha": f"sha{i}", "overall_rmse_iv": 0.004, **fit}
        for i, (day, fit) in enumerate(fits)
    ]


def _span_fleet(root, records, variant="heston"):
    _write_run(root, "2023-05-04", "flat_bsm", total_pnl=500_000.0)
    _write_run(
        root, "2023-05-04", variant, total_pnl=450_000.0, calibration=records,
        maturity_date="2026-05-06",
    )
    _write_manifest(
        root,
        [{"inception": "2023-05-04", "variant": v} for v in ("flat_bsm", variant)],
        variants=["flat_bsm", variant],
        inceptions=["2023-05-04"],
    )
    return root


def test_a_heston_run_carries_a_certificate_span_audit(tmp_path):
    records = _dated([("2023-05-04", ORDINARY_FIT), ("2024-10-10", OUT_OF_SPAN_FIT)])
    agg = s13.aggregate(_span_fleet(tmp_path / "a", records))
    row = next(r for r in agg["per_run"] if r["variant"] == "heston")
    span = row["certificate_span"]
    assert span["n_states"] == 2
    assert span["n_out_of_span"] == 1
    assert span["out_of_span"][0]["label"] == "2024-10-10"
    assert span["covered"] is False


def test_the_span_audit_uses_the_trades_real_remaining_maturity(tmp_path):
    """Inception day of a 3Y trade is 1,098 days out -- 3.0062y at ACT/365.25.

    If the audit passed a placeholder maturity instead of the trade's own,
    this state would be reported out of span on a day-count artefact.
    """
    agg = s13.aggregate(
        _span_fleet(tmp_path / "b", _dated([("2023-05-04", ORDINARY_FIT)]))
    )
    row = next(r for r in agg["per_run"] if r["variant"] == "heston")
    span = row["certificate_span"]
    assert span["n_out_of_span"] == 0
    assert span["covered"] is True


def test_a_variant_the_certificate_does_not_cover_carries_no_span_audit(tmp_path):
    """flat_bsm never runs an ADI solver, so the certificate is not about it."""
    agg = s13.aggregate(
        _span_fleet(tmp_path / "c", _dated([("2023-05-04", ORDINARY_FIT)]))
    )
    row = next(r for r in agg["per_run"] if r["variant"] == "flat_bsm")
    assert row["certificate_span"] is None


def test_the_fleet_pools_the_span_audit_and_names_the_dates(tmp_path):
    records = _dated([("2023-05-04", ORDINARY_FIT), ("2024-10-10", OUT_OF_SPAN_FIT)])
    agg = s13.aggregate(_span_fleet(tmp_path / "d", records))
    fleet = agg["certificate_span"]
    assert fleet["covered"] is False
    assert fleet["n_out_of_span"] == 1
    assert fleet["dates_out_of_span"] == ["2024-10-10"]
    assert fleet["variants"] == ["heston"]


def test_the_report_banners_an_out_of_span_state(tmp_path):
    records = _dated([("2023-05-04", ORDINARY_FIT), ("2024-10-10", OUT_OF_SPAN_FIT)])
    html = s13.build_report(s13.aggregate(_span_fleet(tmp_path / "e", records)))
    assert "OUTSIDE THE CERTIFIED REGIME SPAN" in html
    assert "2024-10-10" in html
    # The banner must not read as a pricing failure: nothing was gated, and
    # the dates stay in the hedge path.
    assert "Nothing was gated" in html
    assert "pricing is unaffected" in html
    assert "remain in the hedge path" in html


def test_a_covered_fleet_gets_no_span_banner(tmp_path):
    html = s13.build_report(
        s13.aggregate(_span_fleet(tmp_path / "f", _dated([("2023-05-04", ORDINARY_FIT)])))
    )
    assert "OUTSIDE THE CERTIFIED REGIME SPAN" not in html


# ---------------------------------------------------------------------------
# Inception mark vs hedging period
#
# Every variant prices the SAME contract, whose coupon was solved so flat BSM
# values it at zero, and the contract is booked at zero -- so day 1 marks each
# model's disagreement with that solve instantly.  That is a valuation opinion,
# not hedging skill, and on the real fleet the two halves carry OPPOSITE signs
# for localvol and heston_slv (+0.5 / +0.8 at inception, -1.9 / -1.6 hedging).
# Blending them understates both, which is what these tests pin.
# ---------------------------------------------------------------------------


def test_pnl_splits_into_inception_and_hedging(fleet):
    agg = s13.aggregate(fleet)
    for row in agg["per_run"]:
        assert row["pnl_inception_pct_notional"] + row[
            "pnl_hedging_pct_notional"
        ] == pytest.approx(row["total_pnl_pct_notional"], abs=1e-12)


def test_inception_component_is_day_one_not_an_average(fleet):
    """The day-1 row IS the mark; anything else silently blends in hedging."""
    agg = s13.aggregate(fleet)
    row = next(
        r for r in agg["per_run"]
        if r["inception"] == "2023-05-04" and r["variant"] == "flat_bsm"
    )
    states = pd.read_csv(
        fleet / "runs" / "2023-05-04" / "flat_bsm" / "states.csv", index_col=0
    )
    assert row["pnl_inception"] == pytest.approx(states["total_pnl"].iloc[0])
    assert row["pnl_hedging"] == pytest.approx(
        states["total_pnl"].iloc[-1] - states["total_pnl"].iloc[0]
    )


def test_paired_deltas_decompose_the_same_way(fleet):
    agg = s13.aggregate(fleet)
    for pair in agg["paired_vs_baseline"]:
        assert pair["d_pnl_inception_pct_notional"] + pair[
            "d_pnl_hedging_pct_notional"
        ] == pytest.approx(pair["d_pnl_pct_notional"], abs=1e-12)


def test_hedging_win_rate_ignores_the_day_one_mark(tmp_path):
    """A variant can win on the total and lose on every hedging path.

    This is exactly the localvol/heston_slv shape: a positive inception mark
    covering a negative hedging edge.  The blended win rate would call it a
    draw; the hedging win rate must call it 0%.
    """
    root = tmp_path / "masked"
    inceptions = ["2023-05-04", "2023-06-01"]
    for inception in inceptions:
        # Baseline: nothing on day 1, all of it earned while hedging.
        _write_run(root, inception, "flat_bsm", total_pnl=500_000.0, front_load=0.0)
        # Challenger: books a big day-1 mark, then hedges worse, ending ahead.
        _write_run(root, inception, "localvol", total_pnl=600_000.0, front_load=1.0)
    runs = [
        {"inception": i, "variant": v}
        for i in inceptions for v in ("flat_bsm", "localvol")
    ]
    _write_manifest(
        root, runs, variants=["flat_bsm", "localvol"], inceptions=inceptions
    )
    agg = s13.aggregate(root)
    paired = agg["paired_summary"]["localvol"]
    assert paired["d_pnl_pct_notional"]["mean"] > 0.0        # wins on the total
    assert paired["pnl_win_rate"] == pytest.approx(1.0)
    assert paired["d_pnl_hedging_pct_notional"]["mean"] < 0.0  # loses while hedging
    assert paired["pnl_hedging_win_rate"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# PnL decomposition, Cut B: BY COMPONENT
#
#   total = open mark + cashflows + hedge - costs
#
# A different seam through the same total, answering "where did the money come
# from" where Cut A answers "when was it booked".  The two are NOT additive:
# every Cut-B term landing after day 1 already sits inside Cut A's hedging
# half, so summing terms across the cuts double-counts most of the trade.
# ---------------------------------------------------------------------------


def test_pnl_splits_by_component_too(fleet):
    agg = s13.aggregate(fleet)
    for row in agg["per_run"]:
        assert (
            row["pnl_open_mark_pct_notional"]
            + row["pnl_cashflows_pct_notional"]
            + row["pnl_hedge_pct_notional"]
            - row["cost_drag_pct_notional"]
        ) == pytest.approx(row["total_pnl_pct_notional"], abs=1e-12)


def test_the_two_cuts_are_not_additive(tmp_path):
    """Mixing terms across the cuts double-counts, and must not look right.

    Cut A's hedging half already contains the coupon and the hedge, so
    inception + hedging + cashflows + costs is not the total -- it is the
    total plus the whole contract leg again.  Pinned because the four names
    read like one additive list and reviewers will try to add them.
    """
    root = tmp_path / "cuts"
    _write_run(
        root, "2023-05-04", "flat_bsm",
        total_pnl=500_000.0, product_pnl_final=-3_000_000.0, settle=True,
    )
    _write_manifest(
        root,
        [{"inception": "2023-05-04", "variant": "flat_bsm"}],
        variants=["flat_bsm"],
        inceptions=["2023-05-04"],
    )
    row = s13.aggregate(root)["per_run"][0]
    total = row["total_pnl_pct_notional"]

    # Each cut, on its own, is exact.
    assert row["pnl_inception_pct_notional"] + row[
        "pnl_hedging_pct_notional"
    ] == pytest.approx(total, abs=1e-12)

    # The naive blend of the two is not, and is off by the contract leg.
    blended = (
        row["pnl_inception_pct_notional"]
        + row["pnl_hedging_pct_notional"]
        + row["pnl_cashflows_pct_notional"]
        - row["cost_drag_pct_notional"]
    )
    assert blended != pytest.approx(total, abs=1e-6)


def test_settlement_moves_the_mark_into_cash_without_creating_pnl(tmp_path):
    """The knock-out coupon is not income -- it is the mark turning into cash.

    Settling a seven-figure coupon must leave total PnL untouched: on the real
    fleet the coupon lands as a ~6.9M cashflow while product_pnl moves 52k,
    which is one ordinary day of drift.  A PnL that jumped on the coupon date
    would mean the accrual was wrong.
    """
    root = tmp_path / "settle"
    for variant, settle in (("flat_bsm", False), ("ts_bsm", True)):
        _write_run(
            root, "2023-05-04", variant,
            total_pnl=500_000.0, product_pnl_final=-8_000_000.0, settle=settle,
        )
    _write_manifest(
        root,
        [{"inception": "2023-05-04", "variant": v} for v in ("flat_bsm", "ts_bsm")],
        variants=["flat_bsm", "ts_bsm"],
        inceptions=["2023-05-04"],
    )
    rows = {r["variant"]: r for r in s13.aggregate(root)["per_run"]}
    censored, settled = rows["flat_bsm"], rows["ts_bsm"]

    # Same total either way -- settlement changed nothing about the money.
    assert settled["total_pnl"] == pytest.approx(censored["total_pnl"])
    # It only moved WHERE the value sits.
    assert settled["pnl_cashflows"] == pytest.approx(-8_000_000.0)
    assert settled["pnl_open_mark"] == pytest.approx(0.0)
    assert censored["pnl_cashflows"] == pytest.approx(0.0)
    assert censored["pnl_open_mark"] == pytest.approx(-8_000_000.0)


def test_only_a_censored_run_carries_an_open_mark(tmp_path):
    """A non-zero open mark is the marker of PnL that is still an opinion."""
    root = tmp_path / "openmark"
    _write_run(root, "2023-05-04", "flat_bsm", total_pnl=100_000.0, settle=True)
    _write_run(root, "2023-06-01", "flat_bsm", total_pnl=100_000.0, settle=False)
    _write_manifest(
        root,
        [{"inception": i, "variant": "flat_bsm"} for i in ("2023-05-04", "2023-06-01")],
        variants=["flat_bsm"],
        inceptions=["2023-05-04", "2023-06-01"],
    )
    audit = s13.aggregate(root)["pnl_decomposition"]
    assert audit["n_runs"] == 2
    assert audit["n_runs_with_open_mark"] == 1


def test_decomposition_audit_checks_both_cuts(fleet):
    audit = s13.aggregate(fleet)["pnl_decomposition"]
    for cut in ("time_cut", "component_cut"):
        assert audit[cut]["n_checked"] == audit["n_runs"]
        assert audit[cut]["n_unchecked"] == 0
        assert audit[cut]["max_abs_residual"] == pytest.approx(0.0, abs=1e-6)


def test_a_missing_term_is_unchecked_not_a_satisfied_identity():
    """Absent terms must never be reported as an identity that held."""
    assert s13._residual(10.0, [4.0, 6.0]) == pytest.approx(0.0)
    assert s13._residual(10.0, [4.0, 5.0]) == pytest.approx(1.0)
    assert s13._residual(10.0, [4.0, None]) is None
    assert s13._residual(None, [4.0, 6.0]) is None
    assert s13._residual(10.0, [4.0, float("nan")]) is None
    assert s13._residual(10.0, [12.0], minus=[2.0]) == pytest.approx(0.0)


def test_component_terms_reach_the_csv_and_still_add_up(fleet, tmp_path):
    paths = s13.write_tables(s13.aggregate(fleet), tmp_path / "out")
    frame = pd.read_csv(paths["per_run"])
    recombined = (
        frame["pnl_open_mark_pct_notional"]
        + frame["pnl_cashflows_pct_notional"]
        + frame["pnl_hedge_pct_notional"]
        - frame["cost_drag_pct_notional"]
    )
    assert recombined.sub(frame["pnl_pct_notional"]).abs().max() < 1e-12

    summary = pd.read_csv(paths["variant_summary"])
    means = (
        summary["pnl_open_mark_pct_mean"]
        + summary["pnl_cashflows_pct_mean"]
        + summary["pnl_hedge_pct_mean"]
        - summary["cost_drag_pct_mean"]
    )
    assert means.sub(summary["pnl_pct_mean"]).abs().max() < 1e-12


def test_a_shared_path_collapses_the_paired_edge_to_hedge_minus_cost(tmp_path):
    """The whole point of pairing: the contract terms cancel exactly.

    Variants of one inception settle the SAME coupon on the SAME date, so
    under Cut B their contract-side terms are identical and the paired edge is
    arithmetically (hedge - cost).  This is what licenses the claim that the
    study measures hedging and nothing else.
    """
    root = tmp_path / "shared"
    inceptions = ["2023-05-04", "2023-06-01"]
    for inception in inceptions:
        # Same contract leg, different hedge outcomes -- a shared realized path.
        _write_run(
            root, inception, "flat_bsm",
            total_pnl=400_000.0, product_pnl_final=-5_000_000.0, settle=True,
        )
        _write_run(
            root, inception, "localvol",
            total_pnl=650_000.0, product_pnl_final=-5_000_000.0, settle=True,
        )
    _write_manifest(
        root,
        [{"inception": i, "variant": v} for i in inceptions
         for v in ("flat_bsm", "localvol")],
        variants=["flat_bsm", "localvol"],
        inceptions=inceptions,
    )
    agg = s13.aggregate(root)
    for pair in agg["paired_vs_baseline"]:
        assert pair["d_pnl_cashflows_pct_notional"] == pytest.approx(0.0, abs=1e-12)
        assert pair["d_pnl_open_mark_pct_notional"] == pytest.approx(0.0, abs=1e-12)
        # ... which leaves exactly hedge minus cost.
        assert pair["d_pnl_hedge_pct_notional"] - pair[
            "d_cost_drag_pct_notional"
        ] == pytest.approx(pair["d_pnl_pct_notional"], abs=1e-12)

    summary = agg["paired_summary"]["localvol"]
    assert summary["contract_terms_max_abs"] == pytest.approx(0.0, abs=1e-12)
    assert "entirely hedge PnL minus trading cost" in s13.build_report(agg)


def test_an_open_mark_stops_the_paired_edge_from_collapsing(tmp_path):
    """A censored pair leaves a mark each model values differently.

    Then part of the paired edge IS a valuation disagreement, and the report
    must say so instead of claiming the pairing removed the contract.
    """
    root = tmp_path / "censored"
    _write_run(root, "2023-05-04", "flat_bsm", total_pnl=400_000.0, settle=False)
    _write_run(root, "2023-05-04", "localvol", total_pnl=650_000.0, settle=False)
    _write_manifest(
        root,
        [{"inception": "2023-05-04", "variant": v} for v in ("flat_bsm", "localvol")],
        variants=["flat_bsm", "localvol"],
        inceptions=["2023-05-04"],
    )
    agg = s13.aggregate(root)
    summary = agg["paired_summary"]["localvol"]
    assert summary["contract_terms_max_abs"] > 0.0
    html = s13.build_report(agg)
    assert "valuation disagreement rather than hedging" in html
    assert "entirely hedge PnL minus trading cost" not in html


def test_report_carries_both_cuts(fleet):
    html = s13.build_report(s13.aggregate(fleet))
    assert "two cuts of the same total" in html
    assert "Cut A &mdash; by time" in html
    assert "Cut B &mdash; by component" in html
    assert "Do not add across the cuts" in html
    assert "The knock-out coupon is not income" in html
    # The identity residuals are stated, not merely computed.
    assert "worst residual" in html


# ---------------------------------------------------------------------------
# Completeness checks
# ---------------------------------------------------------------------------


def test_a_knocked_out_run_is_complete_despite_a_short_states_file(tmp_path):
    """states.csv holds days REPLAYED, not days in the window.

    Every run in this study knocks out early and terminates, so comparing row
    count against the window length reported every single one as incomplete.
    """
    root = tmp_path / "ko"
    _write_run(root, "2023-05-04", "flat_bsm", total_pnl=100_000.0, n_days=5)
    runs = [{
        "inception": "2023-05-04",
        "variant": "flat_bsm",
        "n_days": 727,                       # window
        "metrics": {"days_replayed": 5},     # actually replayed
    }]
    _write_manifest(root, runs, variants=["flat_bsm"], inceptions=["2023-05-04"])
    manifest = json.loads((root / "run_manifest.json").read_text())
    comp = s13.verify_fleet_completeness(root, manifest)
    assert comp["n_complete"] == 1, comp["incomplete"]


def test_more_rows_than_the_window_is_still_an_error(tmp_path):
    """The window remains the upper bound -- a replay cannot exceed it."""
    root = tmp_path / "toolong"
    _write_run(root, "2023-05-04", "flat_bsm", total_pnl=100_000.0, n_days=5)
    runs = [{
        "inception": "2023-05-04",
        "variant": "flat_bsm",
        "n_days": 3,
        "metrics": {"days_replayed": 5},
    }]
    _write_manifest(root, runs, variants=["flat_bsm"], inceptions=["2023-05-04"])
    manifest = json.loads((root / "run_manifest.json").read_text())
    comp = s13.verify_fleet_completeness(root, manifest)
    assert comp["n_complete"] == 0
    assert any("more than the 3 days" in i for c in comp["incomplete"] for i in c["issues"])


def test_slv_is_checked_for_its_leverage_surface_not_a_dupire_one(tmp_path):
    """heston_slv records leverage_min/max; demanding lv_min/max fails a good run."""
    root = tmp_path / "slv"
    _write_run(
        root, "2023-05-04", "heston_slv", total_pnl=100_000.0,
        calibration=[{"leverage_min": 0.43, "leverage_max": 1.67, "eta": 1.0}],
    )
    check = s13.verify_run_completeness(
        root / "runs" / "2023-05-04" / "heston_slv", variant="heston_slv"
    )
    assert check["complete"], check["issues"]
    assert check["categories"]["leverage_surface_records"] is True


def test_localvol_still_needs_its_dupire_surface_stats(tmp_path):
    root = tmp_path / "lv"
    _write_run(
        root, "2023-05-04", "localvol", total_pnl=100_000.0,
        calibration=[{"leverage_min": 0.43, "leverage_max": 1.67}],  # wrong keys here
    )
    check = s13.verify_run_completeness(
        root / "runs" / "2023-05-04" / "localvol", variant="localvol"
    )
    assert not check["complete"]
    assert any("lv_min" in i for i in check["issues"])
