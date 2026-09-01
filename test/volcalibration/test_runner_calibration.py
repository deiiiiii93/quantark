import json
from datetime import date
from pathlib import Path

import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.config import (
    CalibrationRunConfig,
    RunConfig,
    SurfaceBuildConfig,
    UnderlyingConfig,
)
from quantark.volcalibration.runner import (
    calibration_record_is_current,
    combine_daily_v0_and_structure,
    heston_feller_diagnostics,
    run_calibration_stage,
    run_surface_stage,
    select_calibration_dates,
    structural_ewma_before,
    update_structural_ewma,
)
from quantark.volcalibration.snapshot import QuoteSnapshot
from quantark.volcalibration.store import StoreLayout

CONFIG = CalibrationRunConfig().manifest_payload()
VARIANTS = ("localvol", "heston", "heston_slv")
SAMPLE = Path("example/mo_volmodels/data/mo_settlement_snapshot_20260430.json")


def _cal_record(tag, sha, status="ok", config=None, variants=VARIANTS):
    return {
        "date": tag,
        "status": status,
        "surface_sha": sha,
        "config": config or CONFIG,
        "variants": {v: {"status": "ok", "record": {}} for v in variants},
    }


def test_a_record_is_current_only_when_sha_config_and_every_variant_agree():
    surface = {"artifact_sha256": "abc"}
    assert calibration_record_is_current(
        _cal_record("20260430", "abc"), surface, CONFIG, VARIANTS
    )
    assert not calibration_record_is_current(
        _cal_record("20260430", "xyz"), surface, CONFIG, VARIANTS
    )
    stale_config = _cal_record("20260430", "abc", config={**CONFIG, "slv_n_x": 81})
    assert not calibration_record_is_current(stale_config, surface, CONFIG, VARIANTS)
    partial = _cal_record("20260430", "abc")
    partial["variants"]["heston_slv"] = {"status": "failed"}
    assert not calibration_record_is_current(partial, surface, CONFIG, VARIANTS)


def test_a_localvol_only_run_is_not_judged_stale_for_lacking_heston():
    """A run that asked for one variant must not rebuild forever."""
    surface = {"artifact_sha256": "abc"}
    record = _cal_record("20260430", "abc", variants=("localvol",))
    assert calibration_record_is_current(record, surface, CONFIG, ("localvol",))
    assert not calibration_record_is_current(record, surface, CONFIG, VARIANTS)


def test_incremental_selection_does_not_bootstrap_a_multi_year_backfill():
    surfaces = {
        d: {"status": "ok", "artifact_sha256": d}
        for d in ("20260101", "20260102", "20260103")
    }
    selected = select_calibration_dates(
        surfaces,
        {},
        config=CONFIG,
        backfill=False,
        max_dates=None,
        baseline_date=None,
        start_date=None,
        end_date=None,
        variants=VARIANTS,
    )
    assert selected == ["20260103"], "an empty manifest calibrates only the latest date"


def test_backfill_selects_every_stale_admitted_date():
    surfaces = {
        d: {"status": "ok", "artifact_sha256": d}
        for d in ("20260101", "20260102", "20260103")
    }
    selected = select_calibration_dates(
        surfaces,
        {},
        config=CONFIG,
        backfill=True,
        max_dates=None,
        baseline_date=None,
        start_date=None,
        end_date=None,
        variants=VARIANTS,
    )
    assert selected == ["20260101", "20260102", "20260103"]


def test_a_date_window_bounds_the_backfill():
    surfaces = {
        d: {"status": "ok", "artifact_sha256": d}
        for d in ("20260101", "20260102", "20260103")
    }
    selected = select_calibration_dates(
        surfaces,
        {},
        config=CONFIG,
        backfill=True,
        max_dates=None,
        baseline_date=None,
        start_date="20260102",
        end_date="20260102",
        variants=VARIANTS,
    )
    assert selected == ["20260102"]


def test_temporal_smoothing_invalidates_every_later_date():
    """Each fit is regularized toward an EWMA of all prior fits, so
    recalibrating one date makes the whole suffix stale."""
    surfaces = {
        d: {"status": "ok", "artifact_sha256": d}
        for d in ("20260101", "20260102", "20260103")
    }
    # 0102 is stale (wrong sha); 0101 and 0103 look fine on their own.
    records = {
        "20260101": _cal_record("20260101", "20260101"),
        "20260102": _cal_record("20260102", "stale"),
        "20260103": _cal_record("20260103", "20260103"),
    }
    kwargs = dict(
        config=CONFIG,
        backfill=True,
        max_dates=None,
        baseline_date=None,
        start_date=None,
        end_date=None,
        variants=VARIANTS,
    )
    assert select_calibration_dates(surfaces, records, **kwargs) == ["20260102"]
    assert select_calibration_dates(
        surfaces, records, temporal_smoothing=True, **kwargs
    ) == ["20260102", "20260103"]


def test_excluded_surfaces_are_never_calibrated():
    surfaces = {"20260101": {"status": "excluded", "reason": "static_arbitrage"}}
    assert (
        select_calibration_dates(
            surfaces,
            {},
            config=CONFIG,
            backfill=True,
            max_dates=None,
            baseline_date=None,
            start_date=None,
            end_date=None,
            variants=VARIANTS,
        )
        == []
    )


def test_feller_diagnostics_report_the_ratio_and_the_verdict():
    ok = heston_feller_diagnostics({"kappa": 2.0, "theta": 0.04, "sigma": 0.2})
    assert ok["feller_ratio"] == pytest.approx(2 * 2.0 * 0.04 / 0.04)
    assert ok["feller_satisfied"]
    assert not heston_feller_diagnostics(
        {"kappa": 0.1, "theta": 0.01, "sigma": 0.9}
    )["feller_satisfied"]


def test_the_ewma_is_recursive_and_seeds_from_the_first_observation():
    raw = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.2, "rho": -0.5}
    first = update_structural_ewma(None, raw, trade_date="20260101", span=19)
    assert first["parameters"]["kappa"] == pytest.approx(2.0)
    assert first["observation_count"] == 1
    second = update_structural_ewma(
        first, {**raw, "kappa": 3.0}, trade_date="20260102", span=19
    )
    alpha = 2.0 / 20.0
    assert second["parameters"]["kappa"] == pytest.approx(
        alpha * 3.0 + (1 - alpha) * 2.0
    )
    assert second["first_source_date"] == "20260101"
    assert second["last_source_date"] == "20260102"
    assert second["observation_count"] == 2


def test_a_feller_violating_history_seed_fails_closed():
    records = {
        "20260101": {
            "status": "ok",
            "variants": {
                "heston": {
                    "status": "ok",
                    "record": {
                        "v0": 0.04,
                        "kappa": 0.1,
                        "theta": 0.01,
                        "sigma": 0.9,
                        "rho": -0.5,
                    },
                }
            },
        }
    }
    with pytest.raises(ValidationError):
        structural_ewma_before(records, before_date="20260102", span=19)


def test_the_slv_vector_carries_today_s_v0_over_smoothed_structure():
    raw = {"v0": 0.09, "kappa": 3.0, "theta": 0.05, "sigma": 0.3, "rho": -0.6}
    state = {"parameters": {"kappa": 2.0, "theta": 0.04, "sigma": 0.2, "rho": -0.5}}
    assert combine_daily_v0_and_structure(raw, state) == {
        "v0": 0.09,
        "kappa": 2.0,
        "theta": 0.04,
        "sigma": 0.2,
        "rho": -0.5,
    }


@pytest.mark.slow
def test_localvol_calibration_writes_a_manifest_record(tmp_path):
    """One real fit through the real calibrator, localvol only."""
    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
    config = RunConfig(
        name="t",
        underlying=UnderlyingConfig("000852.SH", "listed_strike", "settlement"),
        history_dir=tmp_path,
        runtime_dir=tmp_path,
        spot_csv=None,
        surface=SurfaceBuildConfig(),
        calibration=CalibrationRunConfig(variants=("localvol",)),
        workers=1,
    )
    layout = StoreLayout.from_config(config)
    snap = QuoteSnapshot.from_legacy_settlement(
        payload,
        trade_date=date(2026, 4, 30),
        spot=8381.947,
        symbol="000852.SH",
        source_sha256=payload["source_sha256"],
        source_url=payload.get("source_url"),
    )
    layout.snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.snapshot_path("20260430").write_text(json.dumps(snap.to_payload()))

    surfaces = run_surface_stage(layout, config)
    records = run_calibration_stage(layout, config, surface_records=surfaces)
    assert records["20260430"]["status"] == "ok"
    assert records["20260430"]["variants"]["localvol"]["status"] == "ok"
    assert records["20260430"]["surface_sha"] == surfaces["20260430"]["artifact_sha256"]
    assert layout.calibration_manifest.is_file()

    # Re-running is a no-op: the record is current.
    again = run_calibration_stage(layout, config, surface_records=surfaces)
    assert again["20260430"]["started_at"] == records["20260430"]["started_at"]
