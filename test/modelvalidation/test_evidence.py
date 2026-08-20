"""Tests for evidence hashing, atomic writes, and identity-gated checkpoints."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from quantark.util.exceptions import ValidationError
from quantark.modelvalidation.evidence import (
    SCHEMA_VERSION,
    CheckpointStore,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    evidence_projection,
    identity_hash,
    projected_sha256,
    read_json,
    validate_durable_root,
)


def test_canonical_json_is_order_independent():
    a = canonical_json({"b": 1, "a": 2})
    b = canonical_json({"a": 2, "b": 1})
    assert a == b
    assert a == '{"a":2,"b":1}'


def test_canonical_json_handles_numpy_scalars():
    text = canonical_json({"x": np.float64(1.5), "n": np.int64(3)})
    assert json.loads(text) == {"x": 1.5, "n": 3}


def test_canonical_json_rejects_unserializable():
    with pytest.raises(TypeError):
        canonical_json({"x": object()})


def test_projection_drops_volatile_and_hash_stable():
    payload = {
        "a": 1,
        "wall_clock_seconds": 3.2,
        "nested": [{"timestamp": "x", "b": 2}],
    }
    projected = evidence_projection(payload)
    assert "wall_clock_seconds" not in projected
    assert "timestamp" not in projected["nested"][0]
    assert projected["nested"][0]["b"] == 2

    first = projected_sha256(payload)
    payload["wall_clock_seconds"] = 99.0
    assert projected_sha256(payload) == first  # volatile change -> same hash

    payload["a"] = 2
    assert projected_sha256(payload) != first  # real change -> new hash


def test_projection_does_not_mutate_input():
    payload = {"a": 1, "wall_clock_seconds": 3.2}
    evidence_projection(payload)
    assert "wall_clock_seconds" in payload


def test_projection_excludes_the_hash_field_itself():
    payload = {"a": 1}
    digest = projected_sha256(payload)
    payload["projected_sha256"] = digest
    # Re-hashing a stamped payload must reproduce the same digest.
    assert projected_sha256(payload) == digest


def test_identity_hash_is_stable_and_discriminating():
    first = identity_hash({"case": "ordinary", "seed": 7})
    assert first == identity_hash({"seed": 7, "case": "ordinary"})
    assert first != identity_hash({"case": "ordinary", "seed": 8})


def test_atomic_write_and_read_round_trip(tmp_path):
    path = tmp_path / "certificate.json"
    atomic_write_json(path, {"schema": SCHEMA_VERSION, "x": [1, 2, 3]})
    assert read_json(path) == {"schema": SCHEMA_VERSION, "x": [1, 2, 3]}
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_creates_parent_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "out.json"
    atomic_write_json(path, {"ok": True})
    assert read_json(path) == {"ok": True}


def test_atomic_write_text_round_trip(tmp_path):
    path = tmp_path / "report.md"
    atomic_write_text(path, "# Report\n")
    assert path.read_text(encoding="utf-8") == "# Report\n"


def test_durable_root_refuses_system_temp_roots():
    """Banked evidence must never be rooted somewhere the OS may reclaim."""
    for bad in (Path("/tmp"), Path(tempfile.gettempdir()), Path("/tmp/modelvalidation")):
        with pytest.raises(ValidationError):
            validate_durable_root(bad)


def test_durable_root_allows_nested_working_directories(tmp_path):
    """pytest's tmp_path is a managed nested directory, not a temp root."""
    assert validate_durable_root(tmp_path) == tmp_path.resolve()
    assert validate_durable_root(tmp_path / "output" / "study")


def test_checkpoint_round_trip(tmp_path):
    store = CheckpointStore(tmp_path)
    identity = {"case": "ordinary", "seed": 7}
    store.save("reference", "ordinary", identity, {"v": 1})
    assert store.load("reference", "ordinary", identity) == {"v": 1}


def test_checkpoint_identity_mismatch_quarantines(tmp_path):
    store = CheckpointStore(tmp_path)
    store.save("reference", "ordinary", {"case": "ordinary", "seed": 7}, {"v": 1})

    stale = store.load("reference", "ordinary", {"case": "ordinary", "seed": 8})
    assert stale is None
    assert (tmp_path / "reference" / "ordinary.json.stale").exists()
    assert not (tmp_path / "reference" / "ordinary.json").exists()


def test_checkpoint_missing_returns_none(tmp_path):
    store = CheckpointStore(tmp_path)
    assert store.load("reference", "absent", {"a": 1}) is None


def test_checkpoint_rejects_unsafe_keys(tmp_path):
    store = CheckpointStore(tmp_path)
    for bad_key in ("../escape", "with/slash", "", "space key"):
        with pytest.raises(ValidationError):
            store.save("reference", bad_key, {"a": 1}, {"v": 1})


def test_checkpoint_overwrite_updates_payload(tmp_path):
    store = CheckpointStore(tmp_path)
    identity = {"case": "ordinary", "seed": 7}
    store.save("reference", "ordinary", identity, {"batches": 1})
    store.save("reference", "ordinary", identity, {"batches": 2})
    assert store.load("reference", "ordinary", identity) == {"batches": 2}


def test_checkpoint_corrupt_file_is_quarantined(tmp_path):
    store = CheckpointStore(tmp_path)
    identity = {"case": "ordinary"}
    store.save("reference", "ordinary", identity, {"v": 1})
    (tmp_path / "reference" / "ordinary.json").write_text("{not json", encoding="utf-8")

    assert store.load("reference", "ordinary", identity) is None
    assert (tmp_path / "reference" / "ordinary.json.stale").exists()
