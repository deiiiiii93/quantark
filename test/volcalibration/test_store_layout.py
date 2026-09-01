import json

import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.config import (
    CalibrationRunConfig,
    RunConfig,
    SurfaceBuildConfig,
    UnderlyingConfig,
)
from quantark.volcalibration.store import (
    StoreLayout,
    atomic_write_json,
    load_calibration_manifest,
    load_surface_manifest,
    read_json,
    save_calibration_manifest,
    save_surface_manifest,
    surface_record,
)


def _config(tmp_path, runtime=None):
    return RunConfig(
        name="t",
        underlying=UnderlyingConfig("000852.SH", "listed_strike", "settlement"),
        history_dir=tmp_path / "history",
        runtime_dir=runtime or (tmp_path / "history"),
        spot_csv=None,
        surface=SurfaceBuildConfig(),
        calibration=CalibrationRunConfig(),
        workers=1,
    )


def test_layout_splits_history_from_runtime(tmp_path):
    layout = StoreLayout.from_config(_config(tmp_path, runtime=tmp_path / "out"))
    assert (
        layout.snapshot_path("20260430") == tmp_path / "history/snapshots/20260430.json"
    )
    assert (
        layout.artifact_path("20260430")
        == tmp_path / "history/iv_surface/mo_iv_surface_20260430.json"
    )
    assert layout.surface_manifest == tmp_path / "history/surface_manifest.json"
    assert layout.calibration_manifest == tmp_path / "out/calibration_manifest.json"
    assert layout.calibration_cache == tmp_path / "out/calibration_cache"
    assert layout.status == tmp_path / "out/status.json"
    assert layout.lock == tmp_path / "out/pipeline.lock"


def test_a_single_root_store_needs_only_one_key(tmp_path):
    layout = StoreLayout.from_config(_config(tmp_path))
    assert layout.history_dir == layout.runtime_dir


def test_available_snapshot_dates_ignores_non_date_files(tmp_path):
    layout = StoreLayout.from_config(_config(tmp_path))
    assert layout.available_snapshot_dates() == []
    layout.snapshots_dir.mkdir(parents=True)
    for name in ("20260501.json", "20260430.json", "README.json"):
        (layout.snapshots_dir / name).write_text("{}")
    assert layout.available_snapshot_dates() == ["20260430", "20260501"]


def test_atomic_write_leaves_no_temp_file_and_reads_back(tmp_path):
    target = tmp_path / "deep" / "payload.json"
    atomic_write_json(target, {"b": 2, "a": 1})
    assert json.loads(target.read_text()) == {"a": 1, "b": 2}
    assert sorted(p.name for p in target.parent.iterdir()) == ["payload.json"]
    assert read_json(target, default=None) == {"a": 1, "b": 2}
    assert read_json(tmp_path / "absent.json", default={"records": []}) == {
        "records": []
    }


def test_a_corrupt_manifest_raises_rather_than_defaulting(tmp_path):
    layout = StoreLayout.from_config(_config(tmp_path))
    layout.surface_manifest.parent.mkdir(parents=True, exist_ok=True)
    layout.surface_manifest.write_text("{ not json")
    with pytest.raises(ValidationError):
        load_surface_manifest(layout)


def test_saving_the_surface_manifest_preserves_foreign_top_level_blocks(tmp_path):
    layout = StoreLayout.from_config(_config(tmp_path))
    config = _config(tmp_path)
    atomic_write_json(
        layout.surface_manifest,
        {
            "schema_version": 1,
            "records": [],
            "study_admission": {"vol_model_backtest": {"excluded_dates": ["20240930"]}},
        },
    )
    records = {
        "20260430": surface_record(
            "20260430", status="ok", n_expiries=6, artifact_sha256="a"
        )
    }
    save_surface_manifest(
        layout,
        records,
        config=config.surface.fingerprint_payload(),
        window={"start": "20260430", "end": "20260430"},
        price_field="settlement",
        source_class="official_cffex_eod_settlement",
    )
    payload, by_date = load_surface_manifest(layout)
    # exclude_thin_surfaces.py writes study_admission; the builder must not eat it
    assert payload["study_admission"]["vol_model_backtest"]["excluded_dates"] == [
        "20240930"
    ]
    assert payload["price_field"] == "settlement"
    assert payload["source"] == "official_cffex_eod_settlement"
    assert payload["config"]["min_common_strikes"] == 3
    assert payload["gap_policy"]
    assert list(by_date) == ["20260430"]


def test_calibration_manifest_round_trips_and_sorts(tmp_path):
    layout = StoreLayout.from_config(_config(tmp_path))
    base, records = load_calibration_manifest(layout)
    assert records == {}
    assert base["bootstrap_policy"] == "latest_admitted_surface_only"
    base["baseline_date"] = "20260430"
    save_calibration_manifest(
        layout,
        base,
        {"20260501": {"date": "20260501"}, "20260430": {"date": "20260430"}},
        config=CalibrationRunConfig().manifest_payload(),
    )
    payload, by_date = load_calibration_manifest(layout)
    assert [r["date"] for r in payload["records"]] == ["20260430", "20260501"]
    assert payload["baseline_date"] == "20260430"
    assert sorted(by_date) == ["20260430", "20260501"]


def test_an_unsupported_calibration_schema_is_refused(tmp_path):
    layout = StoreLayout.from_config(_config(tmp_path))
    atomic_write_json(layout.calibration_manifest, {"schema_version": 99, "records": []})
    with pytest.raises(ValidationError):
        load_calibration_manifest(layout)


def test_a_record_without_a_date_is_refused(tmp_path):
    layout = StoreLayout.from_config(_config(tmp_path))
    atomic_write_json(
        layout.surface_manifest, {"schema_version": 1, "records": [{"status": "ok"}]}
    )
    with pytest.raises(ValidationError) as exc:
        load_surface_manifest(layout)
    assert "date" in str(exc.value)
