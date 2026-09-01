import pytest

from quantark.volcalibration.store import (
    BUILDER_SCHEMA_VERSION,
    builder_fingerprint,
    migrate_manifest,
    serialize_artifact,
    surface_record,
)

LEGACY_MANIFEST = {
    "schema_version": 1,
    "source": "official_cffex_eod_settlement",
    "price_field": "settlement",
    "gap_policy": "consumers carry forward previous admitted surface",
    "config": {
        "sabr_beta": 1.0,
        "min_expiries": 2,
        "min_strikes_per_expiry": 5,
        "min_common_strikes": 5,
        "artifact_schema_version": 1,
    },
    "records": [
        {
            "date": "20240930",
            "status": "excluded",
            "reason": "insufficient_expiries",
            "detail": "1 expiry",
            "n_expiries": 1,
            "artifact_sha256": None,
        },
        {
            "date": "20260430",
            "status": "ok",
            "reason": None,
            "detail": None,
            "n_expiries": 4,
            "artifact_sha256": "abc123",
        },
    ],
}


def test_serialize_is_deterministic_and_nan_free():
    art = {"b": 1, "a": [1.0, 2.0]}
    first, second = serialize_artifact(art), serialize_artifact(art)
    assert first == second
    assert first.endswith(b"\n")
    assert b'"a"' in first and first.index(b'"a"') < first.index(b'"b"')
    with pytest.raises(ValueError):
        serialize_artifact({"x": float("nan")})


def test_fingerprint_is_order_insensitive_but_value_sensitive():
    a = builder_fingerprint({"sabr_beta": 1.0, "min_expiries": 2})
    b = builder_fingerprint({"min_expiries": 2, "sabr_beta": 1.0})
    c = builder_fingerprint({"sabr_beta": 0.5, "min_expiries": 2})
    assert a == b
    assert a != c
    assert len(a) == 64


def test_migration_populates_records_without_touching_artifacts():
    migrated = migrate_manifest(LEGACY_MANIFEST)
    ok = next(r for r in migrated["records"] if r["date"] == "20260430")
    assert ok["price_field"] == "settlement"
    assert ok["builder_fingerprint"] == builder_fingerprint(LEGACY_MANIFEST["config"])
    assert ok["builder_schema_version"] == BUILDER_SCHEMA_VERSION
    assert ok["artifact_sha256"] == "abc123"  # untouched
    assert ok["provenance"] == "grandfathered"
    assert ok["snapshot_sha256"] is None


def test_migration_is_idempotent():
    once = migrate_manifest(LEGACY_MANIFEST)
    twice = migrate_manifest(once)
    assert once == twice


def test_new_records_are_verified():
    rec = surface_record(
        "20260901",
        status="ok",
        n_expiries=5,
        artifact_sha256="deadbeef",
        snapshot_sha256="feedface",
        price_field="settlement",
        fingerprint=builder_fingerprint({"sabr_beta": 1.0}),
    )
    assert rec["provenance"] == "verified"
    assert rec["builder_schema_version"] == BUILDER_SCHEMA_VERSION
