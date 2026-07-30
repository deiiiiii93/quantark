import importlib.util
import hashlib
import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "example/mo_volmodels/10_explainer.py"
SPEC = importlib.util.spec_from_file_location("mo_stage10_explainer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EXPLAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPLAINER)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixtures(data_dir: Path, tag: str = "sample") -> dict[str, dict]:
    fetched_at = "2026-07-06T15:00:00"
    maturities = [0.10, 0.50]
    smoothing = {
        "method": "sabr_calendar_projected",
        "beta": 1.0,
        "grid_size": 25,
        "calendar_adjusted_nodes": 0,
        "raw_grid_rmse_iv": 0.005,
        "raw_points_rmse_iv": 0.004,
    }
    snapshot_expiries = []
    surface_expiries = []
    for expiry_date, maturity, forward, vols in (
        ("2026-08-21", 0.10, 99.0, (0.25, 0.20, 0.23)),
        ("2027-01-15", 0.50, 96.0, (0.28, 0.22, 0.24)),
    ):
        quotes = []
        for strike, vol in zip((90.0, 100.0, 110.0), vols):
            for option_type in ("C", "P"):
                quotes.append(
                    {
                        "strike": strike,
                        "type": option_type,
                        "last": 4.0 + vol,
                        "bid": 3.9 + vol,
                        "ask": 4.1 + vol,
                        "volume": 20,
                        "oi": 20,
                    }
                )
        snapshot_expiries.append(
            {"expiry_date": expiry_date, "T_years": maturity, "quotes": quotes}
        )
        surface_expiries.append(
            {
                "expiry_date": expiry_date,
                "T": maturity,
                "r": 0.02,
                "q": 0.08,
                "forward": forward,
                "df": 0.99,
                "points": [[90.0, vols[0]], [100.0, vols[1]], [110.0, vols[2]]],
            }
        )

    snapshot = {
        "fetched_at": fetched_at,
        "market_open": True,
        "underlying": {"code": "000852.SH", "spot": 100.0},
        "expiries": snapshot_expiries,
    }
    surface = {
        "s0": 100.0,
        "fetched_at": fetched_at,
        "strikes": [90.0, 100.0, 110.0],
        "maturities": maturities,
        "iv_grid": [[0.25, 0.20, 0.23], [0.28, 0.22, 0.24]],
        "per_expiry": surface_expiries,
    }

    def model_rows(prepared: tuple[float, float], raw: tuple[float, float]) -> list[dict]:
        return [
            {"T": maturity, "rmse_iv": prep, "raw_rmse_iv": raw_value}
            for maturity, prep, raw_value in zip(maturities, prepared, raw)
        ]

    params = {"v0": 0.04, "kappa": 3.0, "theta": 0.04, "sigma": 0.7, "rho": -0.2}
    feller = 2.0 * params["kappa"] * params["theta"] / params["sigma"] ** 2
    bounds = [[1e-6, 0.001, 0.0001, 0.001, -0.95], [0.5, 3.0, 0.5, 0.7, 0.0]]
    bound_hits = {
        name: {
            "lower": False,
            "upper": name in {"kappa", "sigma"},
        }
        for name in EXPLAINER.PARAMETERS
    }

    def svd(*, condition: float = 20.0, rank: int = 5) -> dict:
        vectors = []
        for index, name in enumerate(EXPLAINER.PARAMETERS):
            components = {parameter: 0.0 for parameter in EXPLAINER.PARAMETERS}
            components[name] = 1.0
            vectors.append(
                {
                    "index": index,
                    "singular_value": 1.0 / (index + 1),
                    "relative_singular_value": 1.0 / (index + 1),
                    "components": components,
                }
            )
        return {
            "singular_values": [1.0, 0.6, 0.3, 0.1, 0.05],
            "relative_singular_values": [1.0, 0.6, 0.3, 0.1, 0.05],
            "condition_number": condition,
            "numerical_rank": 5,
            "policy_effective_rank": rank,
            "right_singular_vectors": vectors,
        }

    jacobian = {
        "shape": [6, 5],
        "parameter_order": list(EXPLAINER.PARAMETERS),
        "excludes_feller_penalty": True,
        "base_parameters": params,
        "svd": {"fixed_economic": svd()},
    }
    bootstrap_quantiles = {
        name: {
            "q05": value * 0.9 if value >= 0 else value * 1.1,
            "q50": value,
            "q95": value * 1.1 if value >= 0 else value * 0.9,
        }
        for name, value in params.items()
    }
    bootstrap = {
        "requested_replicates": 4,
        "successful_replicates": 4,
        "failed_replicates": 0,
        "is_statistical_confidence_interval": False,
        "parameter_quantiles": bootstrap_quantiles,
        "bound_hit_rates": {
            name: {"lower": 0.0, "upper": 1.0 if name in {"kappa", "sigma"} else 0.0,
                   "either": 1.0 if name in {"kappa", "sigma"} else 0.0}
            for name in EXPLAINER.PARAMETERS
        },
        "replicates": [{"index": index, "success": True} for index in range(4)],
    }
    heston = {
        "params": params,
        "feller": feller,
        "cost": 0.01,
        "success": True,
        "overall_rmse_iv": 0.012,
        "raw_overall_rmse_iv": 0.015,
        "per_expiry": model_rows((0.010, 0.014), (0.013, 0.017)),
        "target_smoothing": smoothing,
        "calibration_spec": {
            "parameter_order": list(EXPLAINER.PARAMETERS),
            "node_count": 6,
            "bounds": bounds,
            "s0": 100.0,
            "surface_provenance": {
                "kind": "file_sha256",
                "filename": f"mo_iv_surface_{tag}.json",
                "sha256": hashlib.sha256(json.dumps(surface).encode("utf-8")).hexdigest(),
            },
        },
        "bound_hits": bound_hits,
        "node_rows": [{"index": index} for index in range(6)],
        "jacobian": jacobian,
        "bootstrap": bootstrap,
    }
    localvol = {
        "overall_rmse_iv": 0.020,
        "raw_overall_rmse_iv": 0.023,
        "per_expiry": model_rows((0.018, 0.022), (0.020, 0.026)),
        "target_smoothing": smoothing,
        "lv_min": 0.15,
        "lv_max": 0.42,
    }
    slv = {
        "overall_rmse_iv": 0.021,
        "raw_overall_rmse_iv": 0.024,
        "per_expiry": model_rows((0.019, 0.023), (0.021, 0.027)),
        "target_smoothing": smoothing,
        "leverage_min": 0.7,
        "leverage_max": 1.4,
    }
    diagnostic_config = {
        "source_class": EXPLAINER.SYNTHETIC_DIAGNOSTIC_SOURCE,
        "price_field": "settlement",
        "bounds": bounds,
        "weighting": "equal_total_weight_per_expiry",
    }

    def diagnostic_date(index: int, *, rank: int = 5) -> dict:
        fitted = {**params, "rho": params["rho"] + 0.1 * index}
        return {
            "trade_date": f"2026-07-{6 + index:02d}",
            "source_class": EXPLAINER.SYNTHETIC_DIAGNOSTIC_SOURCE,
            "source_sha256": f"{index + 1:064x}",
            "price_field": "settlement",
            "config": diagnostic_config,
            "node_universe": {"node_count": 90 + index, "expiry_count": 5},
            "best": {
                "success": True,
                "params": fitted,
                "weighted_rmse_iv": 0.01 + 0.005 * index,
                "feller_ratio": 0.95 + 0.1 * index,
                "bound_hits": bound_hits,
            },
            "jacobian": {"svd": {"fixed_economic": svd(condition=20.0 + 50 * index, rank=rank)}},
        }

    diagnostics = {
        "source_class": EXPLAINER.SYNTHETIC_DIAGNOSTIC_SOURCE,
        "price_field": "settlement",
        "strict_comparability_gate": {
            "required_source_class": EXPLAINER.SYNTHETIC_DIAGNOSTIC_SOURCE,
            "required_price_field": "settlement",
            "unique_trade_dates": True,
            "unique_source_sha256": True,
            "required_config": diagnostic_config,
        },
        "included": [diagnostic_date(0), diagnostic_date(1, rank=4)],
        "exclusions": [{"tag": "synthetic-reject", "reason": "coverage gate"}],
        "stability": {
            "parameters": {
                name: {
                    "min": min(params[name], diagnostic_date(1)["best"]["params"][name]),
                    "max": max(params[name], diagnostic_date(1)["best"]["params"][name]),
                    "cv_abs_mean": 0.6 if name == "rho" else 0.1,
                }
                for name in EXPLAINER.PARAMETERS
            },
            "weighted_rmse_iv": {"min": 0.01, "max": 0.015},
            "feller_ratio": {"min": 0.95, "max": 1.05},
            "bound_hit_frequency": {
                name: 1.0 if name in {"kappa", "sigma"} else 0.0
                for name in EXPLAINER.PARAMETERS
            },
            "identification": {
                "condition_number_range": [20.0, 70.0],
                "minimum_policy_effective_rank": 4,
                "scale_policy": "fixed_economic",
            },
        },
        "verdicts": [{"name": "synthetic", "status": "warning"}],
    }
    payloads = {
        "mo_snapshot": snapshot,
        "mo_iv_surface": surface,
        "mo_reprice_localvol": localvol,
        "mo_calib_heston": heston,
        "mo_reprice_slv": slv,
        "mo_calibration_diagnostics": diagnostics,
    }
    for stem, payload in payloads.items():
        _write(data_dir / f"{stem}_{tag}.json", payload)
    return payloads


def test_explainer_generates_complete_offline_document(tmp_path: Path) -> None:
    _fixtures(tmp_path)

    output = EXPLAINER.generate(tmp_path, "sample")
    document = output.read_text(encoding="utf-8")

    assert output == tmp_path / "mo_calibration_explainer_sample.html"
    assert document.startswith("<!doctype html>")
    assert document.rstrip().endswith("</html>")
    assert len(re.findall(r'<section id="s[1-8]">', document)) == 8
    assert "SYNTHETIC TEST FIXTURE" in document.upper()
    assert "Raw smile explorer" in document
    assert "Official-settlement stability explorer" in document
    assert "Intraday tenor-error explorer" in document
    assert 'id="smileCanvas"' in document
    assert 'id="rmseCanvas"' in document
    assert 'id="stabilityCanvas"' in document
    assert "Fallback evidence table — all raw OTM surface points" in document
    assert "Fallback evidence table — every admitted settlement date" in document
    assert "Fallback evidence table — every intraday expiry and model error" in document
    assert "100.00" in document  # artifact-backed spot
    assert "1.5000 vol pts" in document  # 0.015 decimal IV converted to vol points
    assert "0.4000 vol pts" in document  # point-smoothing RMSE conversion
    assert "violated in the saved fit" in document  # data-driven Feller reading
    assert "Fitted-bound hits" in document
    assert "Seeded maturity-stratified multiplier bootstrap" in document
    assert "not a confidence interval" in document
    assert "study-defined 0.10-vol-point" in document
    assert "Cross-date evidence" in document and "2 DATES" in document
    assert "Optional scenarios absent" in document
    assert "window.__MO_REPORT_DATA__" in document

    assert not re.search(r"<(?:script|img|link)\b[^>]+(?:src|href)=[\"']https?://", document, re.I)
    assert "@import" not in document
    assert "fetch(" not in document
    assert "XMLHttpRequest" not in document


@pytest.mark.parametrize(
    "stem",
    (
        "mo_snapshot",
        "mo_iv_surface",
        "mo_reprice_localvol",
        "mo_calib_heston",
        "mo_reprice_slv",
        "mo_calibration_diagnostics",
    ),
)
def test_explainer_fails_closed_when_core_artifact_is_missing(tmp_path: Path, stem: str) -> None:
    _fixtures(tmp_path)
    (tmp_path / f"{stem}_sample.json").unlink()

    with pytest.raises(FileNotFoundError, match=stem):
        EXPLAINER.generate(tmp_path, "sample")

    assert not (tmp_path / "mo_calibration_explainer_sample.html").exists()


def test_explainer_rejects_snapshot_surface_drift(tmp_path: Path) -> None:
    _fixtures(tmp_path)
    path = tmp_path / "mo_iv_surface_sample.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["fetched_at"] = "2026-07-07T15:00:00"
    _write(path, payload)

    with pytest.raises(ValueError, match="fetched_at drift"):
        EXPLAINER.generate(tmp_path, "sample")


def test_explainer_rejects_model_maturity_drift(tmp_path: Path) -> None:
    _fixtures(tmp_path)
    path = tmp_path / "mo_reprice_slv_sample.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["per_expiry"][1]["T"] = 0.55
    _write(path, payload)

    with pytest.raises(ValueError, match="slv maturity vector drift"):
        EXPLAINER.generate(tmp_path, "sample")


def test_explainer_rejects_smoothing_fingerprint_drift(tmp_path: Path) -> None:
    _fixtures(tmp_path)
    path = tmp_path / "mo_reprice_localvol_sample.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["target_smoothing"]["beta"] = 0.7
    _write(path, payload)

    with pytest.raises(ValueError, match="smoothing fingerprint drift"):
        EXPLAINER.generate(tmp_path, "sample")


def test_explainer_rejects_inconsistent_feller(tmp_path: Path) -> None:
    _fixtures(tmp_path)
    path = tmp_path / "mo_calib_heston_sample.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["feller"] = 1.0
    _write(path, payload)

    with pytest.raises(ValueError, match="Feller ratio does not match"):
        EXPLAINER.generate(tmp_path, "sample")


def test_explainer_requires_jacobian_and_bootstrap_evidence(tmp_path: Path) -> None:
    _fixtures(tmp_path)
    path = tmp_path / "mo_calib_heston_sample.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("jacobian")
    _write(path, payload)

    with pytest.raises(ValueError, match="Jacobian/SVD evidence"):
        EXPLAINER.generate(tmp_path, "sample")


def test_explainer_rejects_duplicate_cross_date_source_hashes(tmp_path: Path) -> None:
    _fixtures(tmp_path)
    path = tmp_path / "mo_calibration_diagnostics_sample.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["included"][1]["source_sha256"] = payload["included"][0]["source_sha256"]
    _write(path, payload)

    with pytest.raises(ValueError, match="duplicate source hashes"):
        EXPLAINER.generate(tmp_path, "sample")


def test_explainer_rejects_cross_date_source_class_mixing(tmp_path: Path) -> None:
    _fixtures(tmp_path)
    path = tmp_path / "mo_calibration_diagnostics_sample.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["included"][1]["source_class"] = EXPLAINER.OFFICIAL_SETTLEMENT_SOURCE
    _write(path, payload)

    with pytest.raises(ValueError, match="source_class drift"):
        EXPLAINER.generate(tmp_path, "sample")


def test_explainer_rejects_cross_date_drift_from_required_config(tmp_path: Path) -> None:
    _fixtures(tmp_path)
    path = tmp_path / "mo_calibration_diagnostics_sample.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload["included"]:
        row["config"] = {**row["config"], "max_nfev": 999}
    _write(path, payload)

    with pytest.raises(ValueError, match="drift from required config"):
        EXPLAINER.generate(tmp_path, "sample")


def test_explainer_rejects_malformed_optional_artifact(tmp_path: Path) -> None:
    payloads = _fixtures(tmp_path)
    smoothing = payloads["mo_calib_heston"]["target_smoothing"]
    _write(
        tmp_path / "mo_barrier_sample.json",
        {
            "spec": {"s0": 100.0, "iv_smoothing": smoothing},
            "models": {"Local Vol": {"mc": 1.0, "pde": 1.0}},
        },
    )

    with pytest.raises(ValueError, match="barrier artifact missing model rows"):
        EXPLAINER.generate(tmp_path, "sample")


def test_explainer_requires_optional_smoothing_identity(tmp_path: Path) -> None:
    _fixtures(tmp_path)
    _write(
        tmp_path / "mo_barrier_sample.json",
        {
            "spec": {"s0": 100.0},
            "models": {
                "BSM (flat ATM)": {"mc": 1.0, "pde": 1.0},
                "Local Vol": {"mc": 1.0, "pde": 1.0},
                "Heston": {"mc": 1.0, "pde": 1.0},
                "SLV": {"mc": 1.0, "pde": 1.0},
            },
        },
    )

    with pytest.raises(ValueError, match="barrier spec requires iv_smoothing metadata"):
        EXPLAINER.generate(tmp_path, "sample")


def test_explainer_labels_futures_close_as_asynchronous(tmp_path: Path) -> None:
    _fixtures(tmp_path)
    _write(
        tmp_path / "im_futures_sample.json",
        {
            "valuation_date": "2026-07-06",
            "fetched_for_demo": "2026-07-08",
            "quotes": [
                {
                    "expiry_date": "2026-08-21",
                    "maturity": 0.10,
                    "close": 99.1,
                }
            ],
        },
    )

    document = EXPLAINER.generate(tmp_path, "sample").read_text(encoding="utf-8")

    assert "asynchronous same-date IM daily closes" in document
    assert "not synchronized executable-price validation" in document
    assert "futures artifact was fetched on 2026-07-08" in document


def test_explainer_rejects_unsafe_tag(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tag must contain"):
        EXPLAINER.load_artifacts(tmp_path, "../sample")
