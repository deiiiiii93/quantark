import json
from datetime import date
from pathlib import Path

import pytest

from quantark.volcalibration.config import (
    CalibrationRunConfig,
    RunConfig,
    SurfaceBuildConfig,
    UnderlyingConfig,
)
from quantark.volcalibration.runner import (
    plan_surface_dates,
    run_surface_stage,
    surface_record_is_current,
)
from quantark.volcalibration.snapshot import QuoteSnapshot
from quantark.volcalibration.store import (
    BUILDER_SCHEMA_VERSION,
    StoreLayout,
    builder_fingerprint,
    load_surface_manifest,
    surface_record,
)

SAMPLE = Path("example/mo_volmodels/data/mo_settlement_snapshot_20260430.json")
TAG = "20260430"
SPOT = 8381.947  # the s0 the stored 20260430 artifact was built with


def _config(history_dir, **surface_kwargs):
    return RunConfig(
        name="t",
        underlying=UnderlyingConfig("000852.SH", "listed_strike", "settlement"),
        history_dir=Path(history_dir),
        runtime_dir=Path(history_dir),
        spot_csv=None,
        surface=SurfaceBuildConfig(**surface_kwargs),
        calibration=CalibrationRunConfig(),
        workers=1,
    )


@pytest.fixture
def store(tmp_path):
    """A store holding one canonical snapshot for 2026-04-30."""
    config = _config(tmp_path)
    layout = StoreLayout.from_config(config)
    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
    snap = QuoteSnapshot.from_legacy_settlement(
        payload,
        trade_date=date(2026, 4, 30),
        spot=SPOT,
        symbol="000852.SH",
        source_sha256=payload["source_sha256"],
        source_url=payload.get("source_url"),
    )
    layout.snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.snapshot_path(TAG).write_text(
        json.dumps(snap.to_payload()), encoding="utf-8"
    )
    return config, layout, snap


def test_a_fresh_store_builds_and_records_provenance(store):
    config, layout, snap = store
    records = run_surface_stage(layout, config)
    assert records[TAG]["status"] == "ok"
    assert layout.artifact_path(TAG).is_file()
    assert records[TAG]["snapshot_sha256"] == snap.sha256
    assert records[TAG]["price_field"] == "settlement"
    assert records[TAG]["builder_fingerprint"] == builder_fingerprint(
        config.surface.fingerprint_payload()
    )
    assert records[TAG]["provenance"] == "verified"
    assert records[TAG]["n_expiries"] == 6


def test_the_manifest_records_the_snapshot_s_own_source_class(store):
    config, layout, _ = store
    run_surface_stage(layout, config)
    payload, _records = load_surface_manifest(layout)
    assert payload["source"] == "official_cffex_eod_settlement"
    assert payload["price_field"] == "settlement"


def test_a_second_run_skips_and_leaves_the_bytes_identical(store):
    config, layout, _ = store
    run_surface_stage(layout, config)
    before = layout.artifact_path(TAG).read_bytes()
    mtime = layout.artifact_path(TAG).stat().st_mtime_ns
    assert (
        plan_surface_dates(
            layout, load_surface_manifest(layout)[1], config, tags=[TAG], force=False
        )
        == []
    )
    run_surface_stage(layout, config)
    assert layout.artifact_path(TAG).read_bytes() == before
    assert layout.artifact_path(TAG).stat().st_mtime_ns == mtime


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda r: r.update(snapshot_sha256="different"), id="republished_snapshot"
        ),
        pytest.param(lambda r: r.update(price_field="mid_or_last"), id="price_field"),
        pytest.param(
            lambda r: r.update(builder_fingerprint="stale"), id="builder_config"
        ),
        pytest.param(
            lambda r: r.update(builder_schema_version=BUILDER_SCHEMA_VERSION + 1),
            id="builder_code",
        ),
    ],
)
def test_each_invalidation_field_forces_a_rebuild(store, mutate):
    config, layout, _ = store
    records = run_surface_stage(layout, config)
    record = dict(records[TAG])
    mutate(record)
    assert plan_surface_dates(
        layout, {TAG: record}, config, tags=[TAG], force=False
    ) == [TAG]


def test_a_grandfathered_record_is_trusted_not_rebuilt(store):
    config, layout, _ = store
    record = surface_record(
        TAG,
        status="ok",
        n_expiries=6,
        artifact_sha256="a",
        provenance="grandfathered",
    )
    assert surface_record_is_current(
        record,
        snapshot_sha="anything",
        price_field="settlement",
        fingerprint="anything",
    )
    assert (
        plan_surface_dates(layout, {TAG: record}, config, tags=[TAG], force=False) == []
    )


def test_a_foreign_exclusion_is_never_resurrected(store):
    """exclude_thin_surfaces.py's record must survive a normal run (plan DP-1)."""
    config, layout, _ = store
    record = surface_record(
        TAG,
        status="excluded",
        reason="insufficient_expiries_for_dupire",
        n_expiries=2,
    )
    assert (
        plan_surface_dates(layout, {TAG: record}, config, tags=[TAG], force=False) == []
    )
    assert plan_surface_dates(
        layout, {TAG: record}, config, tags=[TAG], force=True
    ) == [TAG]


def test_a_builder_exclusion_is_retried_when_the_snapshot_changes(store):
    config, layout, _ = store
    record = surface_record(
        TAG,
        status="excluded",
        reason="static_arbitrage",
        snapshot_sha256="old",
        price_field="settlement",
        fingerprint=builder_fingerprint(config.surface.fingerprint_payload()),
    )
    assert plan_surface_dates(
        layout, {TAG: record}, config, tags=[TAG], force=False
    ) == [TAG]


def test_a_moved_knob_rebuilds_because_the_fingerprint_moves(store):
    config, layout, _ = store
    records = run_surface_stage(layout, config)
    moved = _config(layout.history_dir, max_abs_implied_rate=0.05)
    assert plan_surface_dates(layout, records, moved, tags=[TAG], force=False) == [TAG]


def test_planning_defaults_to_every_snapshot_on_disk(store):
    config, layout, _ = store
    assert plan_surface_dates(layout, {}, config) == [TAG]
