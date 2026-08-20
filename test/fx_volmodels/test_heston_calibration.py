"""Focused contracts for the CFETS raw-node Heston research stages."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "example" / "fx_volmodels"
sys.path.insert(0, str(EXAMPLE))

import _fx_common as fx  # noqa: E402

from quantark.volmodels.heston import HestonParams  # noqa: E402


def _load_numbered_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage04 = _load_numbered_module("04_heston_calibration.py", "fx_heston_calibration_contract")
stage06 = _load_numbered_module(
    "06_calibration_diagnostics.py", "fx_heston_diagnostics_contract"
)


_DAYS = {
    "1D": 1,
    "1W": 7,
    "2W": 14,
    "3W": 21,
    "1M": 30,
    "2M": 60,
    "3M": 90,
    "6M": 180,
    "9M": 270,
    "1Y": 365,
    "18M": 548,
    "2Y": 730,
    "3Y": 1095,
}
_PARAMS = {"v0": 0.0010, "kappa": 2.0, "theta": 0.0012, "sigma": 0.03, "rho": -0.25}


def _snapshot(trade_date: str = "2026-07-20") -> dict:
    slices = []
    forward = 7.10
    for tenor in fx.TENOR_ORDER:
        maturity = _DAYS[tenor] / 365.0
        log_step = max(0.0015, 0.55 * 0.035 * math.sqrt(maturity))
        ratios = [math.exp(multiplier * log_step) for multiplier in (-2, -1, 0, 1, 2)]
        mids = (0.0345, 0.0330, 0.0320, 0.0315, 0.0320)
        slices.append(
            {
                "tenor": tenor,
                "maturity": maturity,
                "expiry_date": "2027-12-31",
                "domestic_rate": 0.02,
                "foreign_rate": 0.03,
                "forward": forward,
                "quotes": [
                    {
                        "pillar": pillar,
                        "delta": fx.PILLAR_DELTA[pillar],
                        "strike": forward * ratio,
                        "bid_iv": mid - 0.0005,
                        "mid_iv": mid,
                        "ask_iv": mid + 0.0005,
                    }
                    for pillar, ratio, mid in zip(fx.PILLAR_ORDER, ratios, mids)
                ],
            }
        )
    return {
        "schema_version": 1,
        "trade_date": trade_date,
        "quote_time": "16:00",
        "currency_pair": "USD.CNY",
        "source_class": "synthetic_test_fixture",
        "spot": forward,
        "limitations": ["synthetic fixture"],
        "slices": slices,
    }


def _best_fit(*, hard: bool = False, rank: int = 5, condition: float | None = 500.0) -> dict:
    params = dict(_PARAMS)
    if hard:
        params["sigma"] = math.sqrt(2.0 * params["kappa"] * params["theta"])
    return {
        "success": True,
        "message": "synthetic convergence",
        "optimizer": "SLSQP" if hard else "least_squares",
        "nfev": 20,
        "params": params,
        "feller_ratio": 1.0 if hard else 2.0 * params["kappa"] * params["theta"] / params["sigma"] ** 2,
        "feller_margin": 0.0 if hard else 2.0 * params["kappa"] * params["theta"] - params["sigma"] ** 2,
        "rmse_iv": 0.0002 if hard else 0.0001,
        "rmse_vol_points": 0.02 if hard else 0.01,
        "mae_vol_points": 0.015 if hard else 0.008,
        "max_abs_vol_points": 0.04 if hard else 0.03,
        "inside_nonzero_public_band_pct": 80.0,
        "rows": [],
        "_rank": rank,
        "_condition": condition,
    }


def _calibration_artifact(
    snapshot: dict,
    tag: str,
    *,
    universe: str = "core",
    starts: int = 5,
    rank: int = 5,
    condition: float | None = 500.0,
) -> dict:
    modes = {}
    for mode, hard in (("free", False), ("hard_feller", True)):
        best = _best_fit(hard=hard, rank=rank, condition=condition)
        mode_rank = best.pop("_rank")
        mode_condition = best.pop("_condition")
        modes[mode] = {
            "best": best,
            "multistart": {"valid": starts, "near_best": 1},
            "fits": [{key: value for key, value in best.items() if key != "rows"}],
            "jacobian": {
                "scaled": {
                    "condition_number": mode_condition,
                    "numerical_rank": mode_rank,
                }
            },
        }
    return {
        "schema_version": 1,
        "tag": tag,
        "trade_date": snapshot["trade_date"],
        "quote_time": snapshot["quote_time"],
        "currency_pair": "USD.CNY",
        "config": {
            "calibration_target": "raw_CFETS_five_delta_mid_IVs",
            "normalization": "forward=1, domestic_rate=foreign_rate=0",
            "method": "lewis",
            "weight_mode": "equal",
            "starts": starts,
            "max_nfev": 500,
            "bounds": [list(fx.HESTON_BOUNDS[0]), list(fx.HESTON_BOUNDS[1])],
            "feller_modes": ["free", "hard_feller"],
        },
        "universes": {
            universe: {
                "node_count": len(fx.TENOR_SETS[universe]) * len(fx.PILLAR_ORDER),
                "node_keys": stage06.expected_node_keys(universe),
                **modes,
                "hard_feller_fit_penalty": {"rmse_vol_points": 0.01},
            }
        },
    }


def _write_inputs(data_dir: Path, tag: str, snapshot: dict, calibration: dict) -> None:
    fx.write_json(data_dir / f"cfets_usdcny_snapshot_{tag}.json", snapshot)
    fx.write_json(data_dir / f"cfets_usdcny_heston_{tag}.json", calibration)


def test_universes_use_only_raw_five_node_slices() -> None:
    snapshot = _snapshot()
    assert len(fx.iter_nodes(snapshot, "core")) == 30
    assert len(fx.iter_nodes(snapshot, "liquid")) == 45
    assert len(fx.iter_nodes(snapshot, "full")) == 60
    assert {
        node["pillar"] for node in fx.iter_nodes(snapshot, "full")
    } == set(fx.PILLAR_ORDER)


def test_failed_solver_is_never_promoted_over_successful_fit() -> None:
    fits = [
        {"success": False, "rmse_iv": 1e-12, "message": "failed despite low objective"},
        {"success": True, "rmse_iv": 2e-4, "message": "usable"},
        {"success": True, "rmse_iv": 3e-4, "message": "worse"},
    ]
    assert stage04._best_successful(fits)["message"] == "usable"


def test_hard_feller_starts_are_feasible_and_all_five_are_summarised(monkeypatch) -> None:
    import quantark.volmodels.heston as native_heston

    received = []

    def fake_calibrate_heston(*, initial, enforce_feller, **_kwargs):
        assert enforce_feller is True
        received.append(initial)
        return SimpleNamespace(
            success=True,
            message="synthetic",
            optimizer="SLSQP",
            nfev=1,
            params=initial,
            feller_margin=2.0 * initial.kappa * initial.theta - initial.sigma**2,
        )

    def fake_diagnostics(_nodes, result):
        return {
            "success": True,
            "message": result.message,
            "rmse_iv": result.params.sigma,
            "rmse_vol_points": 100.0 * result.params.sigma,
            "params": {
                name: getattr(result.params, name) for name in stage04.PARAMETER_NAMES
            },
        }

    monkeypatch.setattr(native_heston, "calibrate_heston", fake_calibrate_heston)
    monkeypatch.setattr(fx, "heston_fit_diagnostics", fake_diagnostics)
    fits = fx.calibrate_heston_multistart(
        _snapshot(), tenor_set="core", hard_feller=True, starts=5
    )

    assert len(received) == len(fits) == 5
    assert all(2.0 * p.kappa * p.theta - p.sigma**2 >= -1e-15 for p in received)
    assert fx.summarise_multistart(fits)["valid"] == 5


def test_finite_difference_jacobian_uses_lewis_model_ivs_and_reports_svd(monkeypatch) -> None:
    nodes = fx.iter_nodes(_snapshot(), "core")
    params = HestonParams(**_PARAMS)
    calls = 0
    original = fx.heston_model_ivs

    def counted_model_ivs(call_nodes, call_params):
        nonlocal calls
        calls += 1
        return original(call_nodes, call_params)

    monkeypatch.setattr(fx, "heston_model_ivs", counted_model_ivs)
    result = stage04.finite_difference_iv_jacobian(nodes, params)

    assert result["method"] == "finite_difference_of_Lewis_implied_vols"
    assert result["shape"] == [30, 5]
    assert calls == 11  # one base plus central up/down perturbations
    assert len(result["raw"]["singular_values"]) == 5
    assert len(result["scaled"]["singular_values"]) == 5
    assert 0 <= result["scaled"]["numerical_rank"] <= 5


def test_report_keeps_free_and_hard_feller_results_separate(monkeypatch) -> None:
    calls = []

    def fake_mode_report(snapshot, universe, *, hard_feller, **_kwargs):
        calls.append((universe, hard_feller))
        best = _best_fit(hard=hard_feller)
        best.pop("_rank")
        best.pop("_condition")
        return {
            "best": best,
            "multistart": {"valid": 5},
            "fits": [],
            "jacobian": {"scaled": {"condition_number": 500.0, "numerical_rank": 5}},
        }

    monkeypatch.setattr(stage04, "_mode_report", fake_mode_report)
    report = stage04.build_calibration_report(_snapshot(), tag="sample")

    assert set(report["universes"]) == {"core", "liquid", "full"}
    assert [report["universes"][name]["node_count"] for name in ("core", "liquid", "full")] == [
        30,
        45,
        60,
    ]
    assert calls == [
        ("core", False),
        ("core", True),
        ("liquid", False),
        ("liquid", True),
        ("full", False),
        ("full", True),
    ]
    assert report["universes"]["core"]["hard_feller_fit_penalty"]["rmse_vol_points"] > 0


def test_cross_date_gate_explicitly_excludes_a_missing_raw_node(tmp_path: Path) -> None:
    good = _snapshot("2026-07-18")
    bad = _snapshot("2026-07-20")
    one_month = next(row for row in bad["slices"] if row["tenor"] == "1M")
    one_month["quotes"] = [q for q in one_month["quotes"] if q["pillar"] != "25P"]
    _write_inputs(tmp_path, "good", good, _calibration_artifact(good, "good"))
    _write_inputs(tmp_path, "bad", bad, _calibration_artifact(bad, "bad"))

    report = stage06.build_cross_date_report(["good", "bad"], tmp_path, "core")

    assert [row["tag"] for row in report["included"]] == ["good"]
    exclusion = report["exclusions"][0]
    assert exclusion["reason"] == "non_comparable_snapshot_nodes"
    assert ["1M", "25P"] in exclusion["missing_nodes"]


def test_cross_date_gate_explicitly_excludes_config_drift(tmp_path: Path) -> None:
    first = _snapshot("2026-07-18")
    second = _snapshot("2026-07-20")
    _write_inputs(tmp_path, "first", first, _calibration_artifact(first, "first", starts=5))
    _write_inputs(tmp_path, "second", second, _calibration_artifact(second, "second", starts=4))

    report = stage06.build_cross_date_report(["first", "second"], tmp_path, "core")

    assert [row["tag"] for row in report["included"]] == ["first"]
    assert report["exclusions"][0]["reason"] == "calibration_config_mismatch"
    assert report["exclusions"][0]["expected_config"]["starts"] == 5
    assert report["exclusions"][0]["actual_config"]["starts"] == 4


def test_rank_deficiency_is_a_warning_even_when_condition_is_undefined(tmp_path: Path) -> None:
    snapshot = _snapshot()
    calibration = _calibration_artifact(snapshot, "weak", rank=4, condition=None)
    _write_inputs(tmp_path, "weak", snapshot, calibration)

    report = stage06.build_cross_date_report(["weak"], tmp_path, "core")
    verdict = next(item for item in report["verdicts"] if item["name"] == "local_identification")
    assert verdict["status"] == "warning"
    assert verdict["evidence"]["minimum_scaled_jacobian_rank"] == 4


def test_diagnostics_writes_tagged_json_csv_and_plot(tmp_path: Path) -> None:
    first = _snapshot("2026-07-18")
    second = _snapshot("2026-07-20")
    _write_inputs(tmp_path, "first", first, _calibration_artifact(first, "first"))
    second_artifact = _calibration_artifact(second, "second")
    second_artifact["universes"]["core"]["free"]["best"]["params"]["rho"] = -0.35
    _write_inputs(tmp_path, "second", second, second_artifact)
    report = stage06.build_cross_date_report(["first", "second"], tmp_path, "core")

    paths = stage06.write_artifacts(report, tmp_path, "study")

    assert paths["json"] == tmp_path / "cfets_usdcny_diagnostics_study.json"
    assert paths["json"].exists()
    assert paths["csv"].exists()
    assert paths["plot"] is not None and paths["plot"].exists()
