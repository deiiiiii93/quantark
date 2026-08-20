import importlib.util
import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "example/fx_volmodels/07_explainer.py"
SPEC = importlib.util.spec_from_file_location("fx_stage07_explainer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EXPLAINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPLAINER)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _quote(pillar: str, strike: float, mid: float) -> dict:
    return {
        "pillar": pillar,
        "strike": strike,
        "bid_iv": mid - 0.0002,
        "mid_iv": mid,
        "ask_iv": mid + 0.0002,
    }


def _fixtures(data_dir: Path, tag: str = "sample") -> None:
    pillars = ("10P", "25P", "ATM", "25C", "10C")
    slices = []
    model_rows = []
    for tenor, maturity, forward, mids in (
        ("1M", 1.0 / 12.0, 7.1120, (0.0340, 0.0320, 0.0300, 0.0290, 0.0295)),
        ("3M", 0.25, 7.0980, (0.0360, 0.0335, 0.0310, 0.0300, 0.0308)),
    ):
        strikes = [forward * ratio for ratio in (0.97, 0.985, 1.0, 1.015, 1.03)]
        quotes = [_quote(pillar, strike, mid) for pillar, strike, mid in zip(pillars, strikes, mids)]
        slices.append(
            {
                "tenor": tenor,
                "maturity": maturity,
                "expiry_date": "2026-08-21" if tenor == "1M" else "2026-10-21",
                "forward": forward,
                "domestic_rate": 0.014,
                "foreign_rate": 0.043,
                "raw_quotes": quotes,
            }
        )
        for quote in quotes:
            model_rows.append(
                {
                    "tenor": tenor,
                    "pillar": quote["pillar"],
                    "model_iv": quote["mid_iv"] + 0.00005,
                }
            )

    common = {
        "schema_version": 1,
        "trade_date": "2026-07-20",
        "currency_pair": "USD.CNY",
    }
    surface = {
        **common,
        "quote_time": "16:00",
        "spot": 7.1098,
        "tenor_set": "core",
        "observed_node_count": 10,
        "strikes": [6.88, 7.10, 7.31],
        "maturities": [1.0 / 12.0, 0.25],
        "iv_grid": [[0.034, 0.030, 0.0295], [0.036, 0.031, 0.0308]],
        "slices": slices,
        "surface_preparation": {
            "method": "per-tenor SABR plus calendar projection",
            "grid_size": 3,
            "raw_five_delta_sabr_rmse_iv": 0.00018,
            "calendar_adjusted_nodes": 2,
        },
        "limitations": [
            "Public bid/mid/ask are composite outputs, not executable quotes.",
            "The prepared grid is interpolation, not additional observed liquidity.",
        ],
    }
    localvol = {
        **common,
        "prepared_target_fit": {"rmse_vol_points": 0.0060},
        "raw_composite_fit": {
            "rmse_vol_points": 0.0110,
            "in_prepared_domain": {"rmse_vol_points": 0.0090},
        },
        "limitations": ["Local-vol differentiation depends on the prepared surface."],
    }
    best = {
        "success": True,
        "message": "converged",
        "params": {"v0": 0.0009, "kappa": 1.4, "theta": 0.0010, "sigma": 0.11, "rho": -0.24},
        "feller_ratio": 0.2314,
        "rmse_vol_points": 0.0210,
        "inside_nonzero_public_band_pct": 80.0,
        "rows": model_rows,
    }
    heston = {
        **common,
        "tag": tag,
        "quote_time": "16:00",
        "universes": {
            "core": {
                "node_count": 10,
                "free": {"best": best, "fits": [best]},
                "hard_feller": {"best": best, "fits": [best]},
            }
        },
        "limitations": ["Feller compliance is a diagnostic, not a promotion rule."],
    }
    slv = {
        **common,
        "overall_rmse_vol_points": 0.0780,
        "limitations": ["SLV repricing does not create new market nodes."],
    }

    def history_row(row_tag: str, date: str, kappa: float, rmse: float) -> dict:
        return {
            "tag": row_tag,
            "trade_date": date,
            "node_keys": [[tenor, pillar] for tenor in ("1M", "3M") for pillar in pillars],
            "modes": {
                "free": {
                    "params": {**best["params"], "kappa": kappa},
                    "rmse_vol_points": rmse,
                    "feller_ratio": 0.23,
                    "jacobian_condition": 1250.0 + 100.0 * kappa,
                }
            },
        }

    diagnostics = {
        "schema_version": 1,
        "universe": "core",
        "requested_tags": ["20260718", "20260720"],
        "included": [
            history_row("20260718", "2026-07-18", 1.2, 0.024),
            history_row("20260720", "2026-07-20", 1.4, 0.021),
        ],
        "exclusions": [],
        "stability": {"free": {"rmse_range": [0.021, 0.024]}},
        "verdicts": [
            {
                "status": "conditional",
                "name": "USD/CNY Heston readiness",
                "evidence": {"complete_dates": 2},
                "interpretation": "Fit is measurable; executable-history promotion remains open.",
            }
        ],
    }

    payloads = {
        "cfets_usdcny_surface": surface,
        "cfets_usdcny_localvol": localvol,
        "cfets_usdcny_heston": heston,
        "cfets_usdcny_slv": slv,
        "cfets_usdcny_diagnostics": diagnostics,
    }
    for stem, payload in payloads.items():
        _write(data_dir / f"{stem}_{tag}.json", payload)


def test_explainer_generates_complete_offline_document(tmp_path: Path) -> None:
    _fixtures(tmp_path)

    output = EXPLAINER.generate(tmp_path, "sample")
    document = output.read_text(encoding="utf-8")

    assert output == tmp_path / "fx_calibration_explainer_sample.html"
    assert document.startswith("<!doctype html>")
    assert document.rstrip().endswith("</html>")
    assert len(re.findall(r'<section id="s[1-8]">', document)) == 8
    assert "Public composite · not executable history" in document
    assert "PUBLIC COMPOSITE" in document.upper()
    assert "Tenor smile explorer" in document
    assert "Parameter stability explorer" in document
    assert 'id="smileCanvas"' in document
    assert 'id="stabilityCanvas"' in document
    assert "Fallback evidence table — all market and model nodes" in document
    assert "Fallback evidence table — every included calibration date" in document
    assert "7.1098" in document  # artifact-backed spot
    assert "0.0210 vol pts" in document  # artifact-backed Heston RMSE
    assert "0.0090 vol pts" in document  # in-domain raw-composite local-vol RMSE
    assert "Fit is measurable; executable-history promotion remains open." in document
    assert "2026-07-18" in document and "2026-07-20" in document

    # The generated page opens offline: no runtime stylesheets, scripts, images,
    # imports, or data fetches.  Internal section anchors are allowed.
    assert not re.search(r"<(?:script|img|link)\b[^>]+(?:src|href)=[\"']https?://", document, re.I)
    assert "@import" not in document
    assert "fetch(" not in document
    assert "XMLHttpRequest" not in document
    assert "window.__FX_REPORT_DATA__" in document


def test_explainer_fails_closed_when_an_artifact_is_missing(tmp_path: Path) -> None:
    _fixtures(tmp_path)
    (tmp_path / "cfets_usdcny_slv_sample.json").unlink()

    with pytest.raises(FileNotFoundError, match="cfets_usdcny_slv_sample.json"):
        EXPLAINER.generate(tmp_path, "sample")

    assert not (tmp_path / "fx_calibration_explainer_sample.html").exists()


def test_explainer_rejects_cross_artifact_date_drift(tmp_path: Path) -> None:
    _fixtures(tmp_path)
    path = tmp_path / "cfets_usdcny_localvol_sample.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["trade_date"] = "2026-07-19"
    _write(path, payload)

    with pytest.raises(ValueError, match="inconsistent trade_date"):
        EXPLAINER.generate(tmp_path, "sample")


def test_explainer_rejects_unsafe_tag(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tag must contain"):
        EXPLAINER.load_artifacts(tmp_path, "../sample")


def test_explainer_does_not_invent_missing_market_numbers(tmp_path: Path) -> None:
    _fixtures(tmp_path)
    path = tmp_path / "cfets_usdcny_surface_sample.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["slices"][0]["raw_quotes"][0]["bid_iv"]
    _write(path, payload)

    with pytest.raises(ValueError, match="surface quote 1M 10P is invalid"):
        EXPLAINER.generate(tmp_path, "sample")

    assert not (tmp_path / "fx_calibration_explainer_sample.html").exists()
