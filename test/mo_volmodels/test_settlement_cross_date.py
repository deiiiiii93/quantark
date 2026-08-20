"""Focused contracts for official MO settlement history and cross-date gating."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from quantark.volmodels.black_scholes import bs_call_price


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "example" / "mo_volmodels"
sys.path.insert(0, str(EXAMPLE))


def _load_numbered(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage01 = _load_numbered(
    "01_fetch_mo_settlement_history.py", "mo_settlement_history_contract"
)
stage10 = _load_numbered(
    "10_calibration_diagnostics.py", "mo_cross_date_diagnostics_contract"
)


def _csv_bytes(rows: list[str]) -> bytes:
    header = (
        "合约代码,今开盘,最高价,最低价,成交量,成交金额,持仓量,持仓变化,"
        "今收盘,今结算,前结算,涨跌1,涨跌2,Delta"
    )
    return ("\n".join([header, *rows]) + "\n").encode("gb18030")


def test_official_parser_preserves_close_settlement_liquidity_and_date_maturity() -> None:
    payload = _csv_bytes(
        [
            "IM2608,6800,6810,6790,12,100,40,1,6801,6802,6799,2,3,0",
            "MO2608-C-6400,884.8,931,573,555,3738.958,254,254,681.2,681.4,873.2,-192,-192,0.6860",
            "MO2608-P-6400,100,110,90,321,999,222,4,101.2,101.4,99,2,3,-0.3140",
            "小计,,,,,,,,,,,,,",
        ]
    )

    snapshot = stage01.parse_cffex_csv(payload, "20260720")

    assert snapshot["source_class"] == "official_cffex_eod_settlement"
    assert snapshot["price_field"] == "settlement"
    assert snapshot["source_sha256"] == hashlib.sha256(payload).hexdigest()
    assert snapshot["record_count"] == 2
    assert snapshot["ignored_row_count"] == 2
    expiry = snapshot["expiries"][0]
    assert expiry["expiry_date"] == "2026-08-21"
    assert expiry["calendar_days"] == 32
    assert expiry["T_years"] == pytest.approx(32 / 365)
    call = next(row for row in expiry["quotes"] if row["type"] == "C")
    assert call["close"] == 681.2
    assert call["settlement"] == 681.4
    assert call["volume"] == 555
    assert call["oi"] == 254


def test_parser_rejects_missing_required_column_and_duplicate_contract() -> None:
    missing = "合约代码,今收盘\nMO2608-C-6400,1\n".encode("gb18030")
    with pytest.raises(ValueError, match="missing required columns"):
        stage01.parse_cffex_csv(missing, "20260720")

    row = "MO2608-C-6400,1,1,1,1,1,1,1,1,1,1,1,1,0.5"
    with pytest.raises(ValueError, match="duplicate MO contract"):
        stage01.parse_cffex_csv(_csv_bytes([row, row]), "20260720")


def test_snapshot_writer_rejects_non_standard_nan(tmp_path: Path) -> None:
    snapshot = stage01.parse_cffex_csv(
        _csv_bytes(
            [
                "MO2608-C-6400,1,1,1,1,1,1,1,1,1,1,1,1,0.5",
                "MO2608-P-6400,1,1,1,1,1,1,1,1,1,1,1,1,-0.5",
            ]
        ),
        "20260720",
    )
    snapshot["record_count"] = math.nan

    with pytest.raises(ValueError, match="Out of range float"):
        stage01.write_snapshot(snapshot, tmp_path)

    assert not list(tmp_path.iterdir())


def test_dragon_boat_holiday_rolls_mo2606_expiry_to_next_trading_day() -> None:
    payload = _csv_bytes(
        [
            "MO2606-C-6400,1,1,1,1,1,1,1,1,1,1,1,1,0.5",
            "MO2606-P-6400,1,1,1,1,1,1,1,1,1,1,1,1,-0.5",
        ]
    )

    snapshot = stage01.parse_cffex_csv(payload, "20260615")

    assert snapshot["expiries"][0]["expiry_date"] == "2026-06-22"
    assert snapshot["expiries"][0]["calendar_days"] == 7
    assert stage10._third_friday("2606").isoformat() == "2026-06-22"
    assert snapshot["expiry_calendar"]["frozen_overrides"]["2606"] == "2026-06-22"


def _settlement_snapshot(
    trade_date: str = "2026-07-20", *, digest_seed: str = "base"
) -> dict:
    expiries = [
        ("2608", "2026-08-21"),
        ("2609", "2026-09-18"),
        ("2612", "2026-12-18"),
        ("2703", "2027-03-19"),
        ("2706", "2027-06-18"),
    ]
    valuation = __import__("datetime").date.fromisoformat(trade_date)
    out = {
        "schema_version": 1,
        "trade_date": trade_date,
        "source_class": stage10.SOURCE_CLASS,
        "source_url": "https://example.test/cffex.csv",
        "source_sha256": hashlib.sha256(digest_seed.encode()).hexdigest(),
        "price_field": stage10.PRICE_FIELD,
        "expiries": [],
    }
    forward = 100.0
    # 100.0 is deliberately ABSENT from this grid. Stage 10 splits OTM puts from
    # OTM calls with `strike < forward`, and that forward is OLS-REGRESSED from
    # the quotes -- so a strike sitting exactly ON the forward changes side with
    # the last ULP of the regression. That is what made
    # test_expiry_without_both_liquid_otm_wings_is_explicitly_excluded pass on
    # ARM64 (strike 100 -> call, put wing empty, excluded) and fail on x86_64
    # (strike 100 -> put, put wing non-empty, not excluded). 105.0 stays on the
    # grid because another test keys on it.
    strikes = (
        [80.0 + 2.5 * index for index in range(8)]      # 80 .. 97.5, OTM puts
        + [101.0]                                        # clear of the forward
        + [102.5 + 2.5 * index for index in range(8)]    # 102.5 .. 120, OTM calls
    )
    for contract_month, expiry_date in expiries:
        expiry = __import__("datetime").date.fromisoformat(expiry_date)
        maturity = (expiry - valuation).days / 365.0
        rate = 0.02
        discount_factor = math.exp(-rate * maturity)
        quotes = []
        for strike in strikes:
            call = bs_call_price(forward, strike, maturity, 0.25, rate, rate)
            put = call - discount_factor * (forward - strike)
            for option_type, price in (("C", call), ("P", put)):
                quotes.append(
                    {
                        "contract": f"MO{contract_month}-{option_type}-{strike:g}",
                        "type": option_type,
                        "strike": strike,
                        "close": price,
                        "settlement": price,
                        "volume": 10,
                        "oi": 100,
                    }
                )
        out["expiries"].append(
            {
                "contract_month": contract_month,
                "expiry_date": expiry_date,
                "calendar_days": (expiry - valuation).days,
                "T_years": maturity,
                "quotes": quotes,
            }
        )
    return out


def test_raw_node_builder_has_no_interpolation_and_equal_expiry_weight() -> None:
    snapshot = _settlement_snapshot()

    nodes, metadata = stage10.build_calibration_nodes(snapshot)

    assert metadata["expiry_count"] == 5
    assert metadata["node_count"] == 85
    assert len(metadata["node_keys"]) == 85
    assert metadata["total_objective_weight"] == pytest.approx(5.0)
    assert metadata["parity_quality"]["rmse_points_range"][1] < 1e-10
    assert all(row["two_sided_otm_wings"] is True for row in metadata["per_expiry"])
    assert all(row["put_node_count"] > 0 for row in metadata["per_expiry"])
    assert all(row["call_node_count"] > 0 for row in metadata["per_expiry"])
    assert metadata["static_arbitrage"]["repair_applied"] is False
    assert metadata["static_arbitrage"]["non_increasing_call_violations"] == 0
    assert metadata["static_arbitrage"]["convex_slope_violations"] == 0
    for row in metadata["parity_quality"]["evaluated_expiries"]:
        sensitivity = row["near_atm_sensitivity"]
        assert sensitivity["status"] == "measured"
        assert sensitivity["subset_pair_count"] == 9
        assert sensitivity["forward_relative_difference_vs_full_ols"] == pytest.approx(
            0.0, abs=1e-12
        )
    assert all(node["market_iv"] == pytest.approx(0.25, abs=3e-7) for node in nodes)
    for expiry in {node["expiry_date"] for node in nodes}:
        assert sum(node["weight"] for node in nodes if node["expiry_date"] == expiry) == pytest.approx(1.0)


def test_coverage_gate_rejects_too_few_expiries_or_nodes() -> None:
    snapshot = _settlement_snapshot()
    snapshot["expiries"] = snapshot["expiries"][:4]
    with pytest.raises(stage10.CoverageError, match="need >= 5 expiries"):
        stage10.build_calibration_nodes(snapshot)


def test_expiry_without_both_liquid_otm_wings_is_explicitly_excluded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _settlement_snapshot()
    first = snapshot["expiries"][0]
    for quote in first["quotes"]:
        if quote["type"] == "P" and quote["strike"] < 100.0:
            quote["volume"] = 0
    monkeypatch.setattr(stage10, "MIN_EXPIRIES", 4)
    monkeypatch.setattr(stage10, "MIN_NODES", 60)

    _nodes, metadata = stage10.build_calibration_nodes(snapshot)

    exclusion = next(
        row for row in metadata["excluded_expiries"] if row["contract_month"] == "2608"
    )
    assert exclusion["reason"] == "missing_liquid_otm_wing"
    assert exclusion["put_node_count"] == 0
    assert exclusion["call_node_count"] > 0


def test_implausible_parity_pillar_is_excluded_before_iv_calibration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _settlement_snapshot()
    first = snapshot["expiries"][0]
    for quote in first["quotes"]:
        if quote["type"] == "C":
            quote["settlement"] += 0.2 * quote["strike"]
    monkeypatch.setattr(stage10, "MIN_EXPIRIES", 4)
    monkeypatch.setattr(stage10, "MIN_NODES", 60)

    _nodes, metadata = stage10.build_calibration_nodes(snapshot)

    exclusion = next(
        row for row in metadata["excluded_expiries"] if row["contract_month"] == "2608"
    )
    assert exclusion["reason"] == "parity_quality_gate_failed"
    assert abs(exclusion["implied_rate"]) > 0.10
    assert "2608" in metadata["parity_quality"]["failed_contract_months"]


def test_raw_static_arbitrage_violations_are_reported_without_repair_or_gating() -> None:
    snapshot = _settlement_snapshot()
    for quote in snapshot["expiries"][0]["quotes"]:
        if quote["strike"] == 105.0:
            # Bump call and put equally: parity C-P is unchanged, but the retained
            # call-equivalent strike sequence is no longer monotone/convex.
            quote["settlement"] += 5.0

    nodes, metadata = stage10.build_calibration_nodes(snapshot)

    assert len(nodes) == 85
    diagnostic = metadata["static_arbitrage"]
    assert diagnostic["repair_applied"] is False
    assert (
        diagnostic["non_increasing_call_violations"]
        + diagnostic["convex_slope_violations"]
        > 0
    )
    assert "2608" in diagnostic["affected_contract_months"]


def test_identification_jacobian_applies_sqrt_objective_row_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes = [{"weight": 0.25}, {"weight": 1.0}]
    observed = {}

    def fake_model(_nodes, _params):
        return __import__("numpy").array([2.0, 4.0])

    def fake_jacobian(model_values, parameters, lower, upper):
        observed["values"] = model_values(__import__("numpy").asarray(parameters))
        return {"matrix": [[1.0] * 5, [2.0] * 5], "svd": {}}

    monkeypatch.setattr(stage10, "_model_ivs", fake_model)
    monkeypatch.setattr(stage10.hd, "finite_difference_model_jacobian", fake_jacobian)

    result = stage10._identification_jacobian(
        nodes, [0.05, 1.0, 0.05, 0.4, -0.5]
    )

    assert observed["values"].tolist() == [1.0, 4.0]
    assert result["row_weighting"]["policy"] == "sqrt_equal_total_weight_per_expiry"
    assert result["unweighted_market_iv_matrix"] == [
        [2.0] * 5,
        [2.0] * 5,
    ]


def test_calibration_config_freezes_optimizer_and_best_start_policy() -> None:
    config = stage10.calibration_config(max_nfev=123)

    assert config["optimizer_tolerances"] == {
        "xtol": 1e-6,
        "ftol": 1e-6,
        "gtol": 1e-6,
    }
    assert config["enforce_feller"] is False
    assert config["best_start_selection_policy"] == (
        "minimum_weighted_rmse_among_optimizer_success_true"
    )


def _evidence(
    trade_date: str,
    *,
    digest_seed: str | None = None,
    source_class: str = stage10.SOURCE_CLASS,
    max_nfev: int = 10,
    kappa: float = 1.5,
) -> dict:
    digest_seed = digest_seed or trade_date
    params = {"v0": 0.05, "kappa": kappa, "theta": 0.06, "sigma": 0.4, "rho": -0.5}
    hits = {
        name: {"lower": False, "upper": False} for name in stage10.PARAMETER_NAMES
    }
    return {
        "schema_version": 1,
        "trade_date": trade_date,
        "source_class": source_class,
        "source_sha256": hashlib.sha256(digest_seed.encode()).hexdigest(),
        "price_field": stage10.PRICE_FIELD,
        "config": stage10.calibration_config(max_nfev=max_nfev),
        "node_universe": {
            "node_count": 100,
            "expiry_count": 5,
            "per_expiry": [],
            "parity_quality": {
                "evaluated_expiries": [
                    {
                        "contract_month": "2608",
                        "parity_rmse_points": 1.0,
                        "parity_rmse_forward_ratio": 0.0001,
                        "implied_rate": 0.02,
                        "discount_factor": 0.99,
                        "quality_gate_passed": True,
                    }
                ]
            },
        },
        "best": {
            "success": True,
            "params": params,
            "weighted_rmse_iv": 0.02,
            "feller_ratio": 1.125,
            "bound_hits": hits,
        },
        "jacobian": {
            "svd": {
                "fixed_economic": {
                    "condition_number": 500.0,
                    "policy_effective_rank": 5,
                }
            }
        },
    }


def test_strict_gate_sorts_dates_and_rejects_source_duplicates_and_config_drift() -> None:
    good_late = _evidence("2026-07-20")
    good_early = _evidence("2026-04-30")
    wrong_source = _evidence("2026-05-15", source_class="intraday_sina_midpoint")
    duplicate_date = _evidence("2026-04-30", digest_seed="duplicate-date")
    duplicate_hash = _evidence("2026-06-15", digest_seed="2026-07-20")
    config_drift = _evidence("2026-06-30", max_nfev=11)

    report = stage10.aggregate_evidence(
        [good_late, good_early, wrong_source, duplicate_date, duplicate_hash, config_drift]
    )

    assert [row["trade_date"] for row in report["included"]] == [
        "2026-04-30",
        "2026-07-20",
    ]
    reasons = {row["reason"] for row in report["exclusions"]}
    assert reasons == {
        "source_class_mismatch",
        "duplicate_trade_date",
        "duplicate_source_sha256",
        "calibration_config_mismatch",
    }


def test_direct_aggregate_config_vote_is_independent_of_input_order() -> None:
    rows = [
        _evidence("2026-04-30", max_nfev=10),
        _evidence("2026-05-15", max_nfev=10),
        _evidence("2026-06-15", max_nfev=11),
    ]

    forward = stage10.aggregate_evidence(rows)
    reversed_report = stage10.aggregate_evidence(list(reversed(rows)))

    assert forward == reversed_report
    assert forward["strict_comparability_gate"]["required_config"]["max_nfev"] == 10
    assert [row["trade_date"] for row in forward["included"]] == [
        "2026-04-30",
        "2026-05-15",
    ]
    assert forward["exclusions"][0]["reason"] == "calibration_config_mismatch"


def test_build_report_uses_deterministic_mocked_fits_and_genuine_dates(tmp_path: Path) -> None:
    for compact, iso in (("20260720", "2026-07-20"), ("20260430", "2026-04-30")):
        snapshot = _settlement_snapshot(iso, digest_seed=compact)
        # The mock only needs identity fields; it deliberately avoids an optimizer in this test.
        (tmp_path / f"mo_settlement_snapshot_{compact}.json").write_text(
            json.dumps(snapshot), encoding="utf-8"
        )

    def mocked_calibrator(snapshot: dict, *, max_nfev: int) -> dict:
        day_factor = int(snapshot["trade_date"][-2:]) / 100.0
        return _evidence(
            snapshot["trade_date"],
            digest_seed=snapshot["trade_date"].replace("-", ""),
            max_nfev=max_nfev,
            kappa=1.0 + day_factor,
        )

    first = stage10.build_cross_date_report(
        ["20260720", "20260430"], tmp_path, max_nfev=17, calibrator=mocked_calibrator
    )
    second = stage10.build_cross_date_report(
        ["20260720", "20260430"], tmp_path, max_nfev=17, calibrator=mocked_calibrator
    )

    assert first == second
    assert [row["trade_date"] for row in first["included"]] == [
        "2026-04-30",
        "2026-07-20",
    ]
    assert first["verdicts"][0]["status"] == "insufficient"


def test_artifact_writer_emits_json_csv_and_plot(tmp_path: Path) -> None:
    report = stage10.aggregate_evidence(
        [_evidence("2026-04-30"), _evidence("2026-07-20")]
    )

    paths = stage10.write_artifacts(report, tmp_path, "study")

    assert paths["json"] == tmp_path / "mo_calibration_diagnostics_study.json"
    assert paths["json"].is_file()
    assert paths["csv"].is_file()
    assert paths["plot"] is not None and paths["plot"].is_file()
    saved = json.loads(paths["json"].read_text())
    assert saved["source_class"] == stage10.SOURCE_CLASS
    assert len(saved["included"]) == 2


def test_report_writer_rejects_non_standard_nan_before_side_effects(tmp_path: Path) -> None:
    report = stage10.aggregate_evidence([_evidence("2026-04-30")])
    report["stability"]["bad_number"] = math.nan

    with pytest.raises(ValueError, match="Out of range float"):
        stage10.write_artifacts(report, tmp_path, "nan")

    assert not list(tmp_path.iterdir())


def test_all_candidates_excluded_keeps_parity_static_evidence_without_crashing() -> None:
    details = {
        "parity_quality": {
            "evaluated_expiries": [
                {
                    "contract_month": "2608",
                    "parity_rmse_points": 20.0,
                    "parity_rmse_forward_ratio": 0.002,
                    "implied_rate": -0.20,
                    "discount_factor": 1.02,
                    "quality_gate_passed": False,
                    "near_atm_sensitivity": {"status": "not_measured"},
                }
            ]
        },
        "static_arbitrage": {
            "repair_applied": False,
            "non_increasing_call_violations": 1,
            "convex_slope_violations": 2,
            "affected_contract_months": ["2612"],
        },
    }

    report = stage10.aggregate_evidence(
        [],
        extra_exclusions=[
            {
                "trade_date": "2026-07-20",
                "reason": "coverage_gate_failed",
                "details": details,
            }
        ],
    )

    assert report["included"] == []
    assert report["stability"]["parity_quality"]["failed_pillars"]
    assert report["stability"]["static_arbitrage"]["affected_expiry_count"] == 1
    verdicts = {row["name"]: row["status"] for row in report["verdicts"]}
    assert verdicts["parameter_stability"] == "insufficient"
    assert verdicts["local_identification"] == "insufficient"
    assert verdicts["raw_settlement_static_arbitrage"] == "warning"
