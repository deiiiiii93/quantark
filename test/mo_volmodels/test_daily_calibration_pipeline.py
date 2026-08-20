"""Operational contract tests for the daily MO calibration orchestrator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "example/mo_volmodels/14_daily_calibration_pipeline.py"
SPEC = importlib.util.spec_from_file_location("mo_daily_calibration_pipeline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
daily = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = daily
SPEC.loader.exec_module(daily)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _spot_csv(path: Path, dates: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "date,spot\n" + "".join(f"{value},6000\n" for value in dates),
        encoding="utf-8",
    )


def _run_args(history: Path, runtime: Path, *extra: str):
    args = daily.build_parser().parse_args(
        [
            "run",
            "--as-of",
            "2026-07-22",
            "--history-dir",
            str(history),
            "--runtime-dir",
            str(runtime),
            "--stage01-python",
            sys.executable,
            "--stage-python",
            sys.executable,
            *extra,
        ]
    )
    daily.validate_args(args)
    return args


def _current_artifacts(history: Path, runtime: Path, args) -> None:
    tag = "20260722"
    _spot_csv(history / "csi1000_spot.csv", ["2026-07-21", "2026-07-22"])
    raw = history / "settlement_csv" / f"{tag}_1.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(b"frozen")
    _write_json(
        history / "settlement_manifest.json",
        {"records": [{"date": tag, "status": "ok"}]},
    )
    surface = history / "iv_surface" / f"mo_iv_surface_{tag}.json"
    _write_json(surface, {})
    _write_json(
        history / "surface_manifest.json",
        {
            "records": [
                {
                    "date": tag,
                    "status": "ok",
                    "artifact_sha256": "surface-sha",
                }
            ]
        },
    )
    config = daily.calibration_config_payload(args)
    _write_json(
        runtime / "calibration_manifest.json",
        {
            "schema_version": 1,
            "baseline_date": tag,
            "records": [
                {
                    "date": tag,
                    "surface_sha": "surface-sha",
                    "status": "ok",
                    "config": config,
                    "variants": {
                        variant: {"status": "ok"} for variant in daily.VARIANTS
                    },
                }
            ],
        },
    )


def test_bootstrap_selects_latest_only_then_only_new_dates() -> None:
    surfaces = {
        tag: {"date": tag, "status": "ok", "artifact_sha256": f"sha-{tag}"}
        for tag in ("20260720", "20260721", "20260722")
    }
    config = {"preset": "frozen"}
    selected = daily.select_calibration_dates(
        surfaces,
        {},
        config=config,
        backfill=False,
        max_dates=None,
    )
    assert selected == ["20260722"]

    calibrations = {
        "20260722": {
            "date": "20260722",
            "surface_sha": "sha-20260722",
            "status": "ok",
            "config": config,
            "variants": {
                variant: {"status": "ok"} for variant in daily.VARIANTS
            },
        }
    }
    surfaces["20260723"] = {
        "date": "20260723",
        "status": "ok",
        "artifact_sha256": "sha-20260723",
    }
    selected = daily.select_calibration_dates(
        surfaces,
        calibrations,
        config=config,
        backfill=False,
        max_dates=None,
        baseline_date="20260722",
    )
    assert selected == ["20260723"]


def test_backfill_is_explicit_and_respects_cap() -> None:
    surfaces = {
        tag: {"date": tag, "status": "ok", "artifact_sha256": f"sha-{tag}"}
        for tag in ("20260720", "20260721", "20260722")
    }
    assert daily.select_calibration_dates(
        surfaces,
        {},
        config={},
        backfill=True,
        max_dates=2,
    ) == ["20260720", "20260721"]


def test_backfill_respects_inclusive_date_window() -> None:
    surfaces = {
        tag: {"date": tag, "status": "ok", "artifact_sha256": f"sha-{tag}"}
        for tag in ("20260720", "20260721", "20260722", "20260723")
    }
    assert daily.select_calibration_dates(
        surfaces,
        {},
        config={},
        backfill=True,
        max_dates=None,
        start_date="20260721",
        end_date="20260722",
    ) == ["20260721", "20260722"]


def test_backfill_date_window_validation() -> None:
    parser = daily.build_parser()
    bounded = parser.parse_args(
        [
            "run",
            "--backfill-calibrations",
            "--calibration-start",
            "2026-07-22",
            "--calibration-end",
            "20260721",
        ]
    )
    with pytest.raises(SystemExit, match="must be <="):
        daily.validate_args(bounded)

    missing_opt_in = parser.parse_args(
        ["run", "--calibration-start", "2026-07-21"]
    )
    with pytest.raises(SystemExit, match="require --backfill-calibrations"):
        daily.validate_args(missing_opt_in)


def _heston_record(
    trade_date: str,
    *,
    v0: float,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
) -> dict:
    return {
        "date": trade_date,
        "status": "ok",
        "variants": {
            "heston": {
                "status": "ok",
                "record": {
                    "v0": v0,
                    "kappa": kappa,
                    "theta": theta,
                    "sigma": sigma,
                    "rho": rho,
                },
            }
        },
    }


def test_structural_ewma_uses_raw_history_and_keeps_v0_daily() -> None:
    records = {
        "20260720": _heston_record(
            "20260720",
            v0=0.04,
            kappa=1.0,
            theta=0.10,
            sigma=0.20,
            rho=-0.10,
        ),
        "20260721": _heston_record(
            "20260721",
            v0=0.09,
            kappa=2.0,
            theta=0.20,
            sigma=0.30,
            rho=-0.20,
        ),
    }
    prior = daily.structural_ewma_before(
        records,
        before_date="20260722",
        span=5,
    )
    assert prior is not None
    assert prior["alpha"] == pytest.approx(1.0 / 3.0)
    assert prior["observation_count"] == 2
    assert prior["parameters"]["kappa"] == pytest.approx(4.0 / 3.0)
    assert prior["parameters"]["theta"] == pytest.approx(2.0 / 15.0)

    raw_today = {
        "v0": 0.16,
        "kappa": 2.5,
        "theta": 0.25,
        "sigma": 0.40,
        "rho": -0.30,
    }
    updated = daily.update_structural_ewma(
        prior,
        raw_today,
        trade_date="20260722",
        span=5,
    )
    slv_heston = daily.combine_daily_v0_and_structure(raw_today, updated)
    assert slv_heston["v0"] == pytest.approx(0.16)
    assert slv_heston["kappa"] == pytest.approx(
        (1.0 / 3.0) * 2.5 + (2.0 / 3.0) * (4.0 / 3.0)
    )
    assert daily.heston_feller_diagnostics(slv_heston)[
        "feller_satisfied"
    ] is True
    assert updated["last_source_date"] == "20260722"


def test_temporal_record_prefers_explicit_raw_heston() -> None:
    record = _heston_record(
        "20260722",
        v0=0.04,
        kappa=3.0,
        theta=0.30,
        sigma=0.60,
        rho=-0.30,
    )
    record["temporal_scheme"] = {
        "raw_heston": {
            "v0": 0.05,
            "kappa": 1.5,
            "theta": 0.15,
            "sigma": 0.35,
            "rho": -0.15,
        }
    }
    raw = daily.raw_heston_from_calibration_record(record)
    assert raw is not None
    assert raw["kappa"] == pytest.approx(1.5)


def test_temporal_opt_in_changes_config_but_default_payload_stays_legacy(
    tmp_path: Path,
) -> None:
    default_args = _run_args(tmp_path / "history-a", tmp_path / "runtime-a")
    default_payload = daily.calibration_config_payload(default_args)
    assert "temporal_scheme" not in default_payload

    temporal_args = _run_args(
        tmp_path / "history-b",
        tmp_path / "runtime-b",
        "--temporal-smoothing",
    )
    temporal_payload = daily.calibration_config_payload(temporal_args)
    assert temporal_payload["temporal_scheme"]["structural_ewma_span"] == 5
    assert temporal_payload["temporal_scheme"]["structural_ewma_alpha"] == (
        pytest.approx(1.0 / 3.0)
    )
    assert temporal_payload["temporal_scheme"][
        "heston_temporal_regularization"
    ] == pytest.approx(0.1)


def test_new_scheme_bootstraps_latest_without_implicit_backfill() -> None:
    surfaces = {
        tag: {"date": tag, "status": "ok", "artifact_sha256": f"sha-{tag}"}
        for tag in ("20260720", "20260721", "20260722")
    }
    old_config = {"scheme": "independent"}
    records = {
        tag: {
            "date": tag,
            "surface_sha": f"sha-{tag}",
            "status": "ok",
            "config": old_config,
            "variants": {
                variant: {"status": "ok"} for variant in daily.VARIANTS
            },
        }
        for tag in surfaces
    }
    assert daily.select_calibration_dates(
        surfaces,
        records,
        config={"scheme": "temporal"},
        backfill=False,
        max_dates=None,
        baseline_date="20260720",
    ) == ["20260722"]


def test_temporal_cli_validation() -> None:
    parser = daily.build_parser()
    bad_span = parser.parse_args(
        ["run", "--temporal-smoothing", "--structural-ewma-span", "0"]
    )
    with pytest.raises(SystemExit, match="span must be >= 1"):
        daily.validate_args(bad_span)

    bad_weight = parser.parse_args(
        [
            "run",
            "--temporal-smoothing",
            "--heston-temporal-regularization",
            "-0.1",
        ]
    )
    with pytest.raises(SystemExit, match="finite and non-negative"):
        daily.validate_args(bad_weight)


def test_status_distinguishes_current_excluded_and_stale_cache(tmp_path: Path) -> None:
    history = tmp_path / "history"
    runtime = tmp_path / "runtime"
    args = _run_args(history, runtime)
    _current_artifacts(history, runtime, args)
    paths = daily.PipelinePaths(history, runtime)

    current = daily.build_freshness_status(
        paths=paths,
        as_of=daily.date(2026, 7, 22),
    )
    assert current["overall_status"] == "current"
    assert current["freshness"]["calibration_lag_trading_days"] == 0

    surface_manifest = json.loads(
        (history / "surface_manifest.json").read_text(encoding="utf-8")
    )
    surface_manifest["records"][0].update(
        status="excluded", reason="static_arbitrage"
    )
    _write_json(history / "surface_manifest.json", surface_manifest)
    excluded = daily.build_freshness_status(
        paths=paths,
        as_of=daily.date(2026, 7, 22),
    )
    assert excluded["overall_status"] == "surface_excluded"

    surface_manifest["records"][0].update(status="ok", reason=None)
    _write_json(history / "surface_manifest.json", surface_manifest)
    stale = daily.build_freshness_status(
        paths=paths,
        as_of=daily.date(2026, 7, 31),
    )
    assert stale["overall_status"] == "market_cache_stale"
    assert stale["freshness"]["spot_cache_calendar_age_days"] == 9


def test_refresh_stage_suppresses_false_holiday_staleness(tmp_path: Path) -> None:
    history = tmp_path / "history"
    runtime = tmp_path / "runtime"
    args = _run_args(history, runtime)
    _current_artifacts(history, runtime, args)
    paths = daily.PipelinePaths(history, runtime)
    stage = daily.StageResult(
        name="refresh_market_cache",
        command=["refresh"],
        started_at=daily.iso_utc(),
        completed_at=daily.iso_utc(),
        elapsed_seconds=0.1,
        returncode=0,
        stdout_tail="",
        stderr_tail="",
    )
    status = daily.build_freshness_status(
        paths=paths,
        as_of=daily.date(2026, 7, 31),
        stages=[stage],
    )
    assert status["overall_status"] == "current"
    assert status["freshness"]["market_refresh_completed_this_run"] is True


def test_lock_is_non_blocking_and_reports_owner(tmp_path: Path) -> None:
    lock = tmp_path / "pipeline.lock"
    with daily.acquire_lock(lock):
        with pytest.raises(daily.LockBusy, match="already running"):
            with daily.acquire_lock(lock):
                pass


def test_pipeline_success_from_current_artifacts(tmp_path: Path) -> None:
    history = tmp_path / "history"
    runtime = tmp_path / "runtime"
    args = _run_args(
        history,
        runtime,
        "--skip-market-refresh",
        "--skip-settlement-fetch",
        "--skip-surface-build",
    )
    _current_artifacts(history, runtime, args)
    code, status = daily.run_pipeline(args)
    assert code == daily.EXIT_CURRENT
    assert status["overall_status"] == "current"
    persisted = json.loads((runtime / "status.json").read_text(encoding="utf-8"))
    assert persisted["overall_status"] == "current"


def test_pipeline_failure_is_persisted_fail_closed(tmp_path: Path) -> None:
    history = tmp_path / "history"
    runtime = tmp_path / "runtime"
    args = _run_args(history, runtime)

    def fail_stage(_name, _command):
        raise daily.PipelineError("synthetic stage failure")

    code, status = daily.run_pipeline(args, stage_runner=fail_stage)
    assert code == daily.EXIT_FAILED
    assert status["overall_status"] == "failed"
    assert status["last_error"]["error_type"] == "PipelineError"
    persisted = json.loads((runtime / "status.json").read_text(encoding="utf-8"))
    assert "synthetic stage failure" in persisted["last_error"]["message"]
