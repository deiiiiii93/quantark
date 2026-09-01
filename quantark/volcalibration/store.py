"""On-disk contracts: store layout, artifact serialization, both manifests.

Artifact bytes are frozen (spec 5.3): their sha256 feeds the calibration cache
key, so no field may be added to or removed from the artifact body.  Anything
new -- resume metadata, provenance status -- goes in a manifest record.  The
body's existing ``source_sha256`` is what makes the legacy migration a per-date
recovery rather than a guess.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from quantark.util.exceptions import ValidationError

# Bump whenever a change to normalization, smoothing or admission CODE alters
# builder output.  Config changes are covered by the fingerprint; code changes
# are not, so this constant is the only mechanism that invalidates artifacts
# after a builder upgrade.  Same obligation as _CACHE_SCHEMA_VERSION in
# quantark/volcalibration/calibrate.py.
BUILDER_SCHEMA_VERSION = 1

PROVENANCE_VERIFIED = "verified"
PROVENANCE_GRANDFATHERED = "grandfathered"

SURFACE_MANIFEST_SCHEMA_VERSION = 1
CALIBRATION_MANIFEST_SCHEMA_VERSION = 1
GAP_POLICY = "consumers carry forward previous admitted surface"
BOOTSTRAP_POLICY = "latest_admitted_surface_only"


def serialize_artifact(artifact: Mapping[str, Any]) -> bytes:
    """Deterministic artifact bytes: sorted keys, no NaN, trailing newline."""
    return (
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def artifact_path(output_dir, trade_date: str) -> Path:
    """Canonical artifact filename for one trade date."""
    return Path(output_dir) / f"mo_iv_surface_{trade_date}.json"


def builder_fingerprint(surface_config: Mapping[str, Any]) -> str:
    """sha256 over canonical JSON of the resolved surface-builder config."""
    canonical = json.dumps(
        dict(surface_config), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def surface_record(
    trade_date: str,
    *,
    status: str,
    reason: Optional[str] = None,
    detail: Optional[str] = None,
    n_expiries: int = 0,
    artifact_sha256: Optional[str] = None,
    snapshot_sha256: Optional[str] = None,
    snapshot_content_sha256: Optional[str] = None,
    symbol: Optional[str] = None,
    price_field: Optional[str] = None,
    fingerprint: Optional[str] = None,
    provenance: str = PROVENANCE_VERIFIED,
) -> Dict[str, Any]:
    """Build one surface-manifest record.

    ``snapshot_sha256`` is the *vendor source* digest -- the CSV the snapshot
    was parsed from -- which is what legacy records can recover from the
    artifact body.  It does not cover everything the build consumed: the spot
    comes from a separate cache, so a corrected spot leaves it unchanged.
    ``snapshot_content_sha256`` closes that by digesting the canonical snapshot
    file itself, and ``symbol`` records which underlying the record is about.
    Both are newer than the migration, so a legacy record simply does not carry
    them (see ``surface_record_is_current``).
    """
    return {
        "date": str(trade_date),
        "status": str(status),
        "reason": reason,
        "detail": detail,
        "n_expiries": int(n_expiries),
        "artifact_sha256": artifact_sha256,
        "snapshot_sha256": snapshot_sha256,
        "snapshot_content_sha256": snapshot_content_sha256,
        "symbol": symbol,
        "price_field": price_field,
        "builder_fingerprint": fingerprint,
        "builder_schema_version": BUILDER_SCHEMA_VERSION,
        "provenance": provenance,
    }


def _snapshot_sha_from_artifact(iv_surface_dir, trade_date: str) -> Optional[str]:
    """Read one artifact's own ``source_sha256``; None if unreadable.

    The artifact body records the sha of the snapshot it was built from, so the
    source CSV does not need to still exist for a date to migrate to
    ``verified``.
    """
    if iv_surface_dir is None:
        return None
    path = artifact_path(iv_surface_dir, trade_date)
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError):
        return None
    sha = payload.get("source_sha256")
    return str(sha) if sha else None


def migrate_manifest(
    manifest: Mapping[str, Any], *, iv_surface_dir=None
) -> Dict[str, Any]:
    """One-time, no-rebuild migration of legacy surface-manifest records.

    Most of the missing metadata is already in the manifest, one level up: the
    top-level ``price_field`` and ``config`` block ARE the settings those
    artifacts were built with.  Copying them down is lossless and touches no
    artifact.

    ``snapshot_sha256`` comes from each artifact's own ``source_sha256`` field
    when ``iv_surface_dir`` is given, so an admitted date migrates to
    ``verified`` without needing its source CSV.  Anything still unrecoverable
    -- a missing or unreadable artifact, or an ``excluded`` date that never had
    one -- leaves the record ``grandfathered``, which is never treated as a
    mismatch and never triggers a rebuild: rebuilding would destroy the very
    bytes the cohort pins depend on.
    """
    out = dict(manifest)
    config = dict(out.get("config", {}))
    fingerprint = builder_fingerprint(config) if config else None
    top_price_field = out.get("price_field")

    migrated = []
    for raw in out.get("records", []):
        rec = dict(raw)
        if rec.get("snapshot_sha256") is None:
            rec["snapshot_sha256"] = _snapshot_sha_from_artifact(
                iv_surface_dir, str(rec.get("date", ""))
            )
        if rec.get("price_field") is None:
            rec["price_field"] = top_price_field
        if rec.get("builder_fingerprint") is None:
            rec["builder_fingerprint"] = fingerprint
        rec.setdefault("builder_schema_version", BUILDER_SCHEMA_VERSION)
        if rec.get("provenance") is None:
            recoverable = all(
                rec.get(key) is not None
                for key in ("snapshot_sha256", "price_field", "builder_fingerprint")
            )
            rec["provenance"] = (
                PROVENANCE_VERIFIED if recoverable else PROVENANCE_GRANDFATHERED
            )
        migrated.append(rec)

    out["records"] = migrated
    return out


# ------------------------------------------------------------- store layout


@dataclass(frozen=True)
class StoreLayout:
    """Where a run's inputs, artifacts and manifests live.

    Two roots, not one: the MO deployment keeps surfaces beside the example
    suite's data and calibration output under ``output/``.  ``runtime_dir``
    defaults to ``history_dir``, so a single-root store stays a single key in
    the YAML.
    """

    history_dir: Path
    runtime_dir: Path

    @classmethod
    def from_config(cls, config) -> "StoreLayout":
        return cls(Path(config.history_dir), Path(config.runtime_dir))

    @property
    def snapshots_dir(self) -> Path:
        return self.history_dir / "snapshots"

    @property
    def surface_dir(self) -> Path:
        return self.history_dir / "iv_surface"

    @property
    def surface_manifest(self) -> Path:
        return self.history_dir / "surface_manifest.json"

    @property
    def calibration_cache(self) -> Path:
        return self.runtime_dir / "calibration_cache"

    @property
    def calibration_manifest(self) -> Path:
        return self.runtime_dir / "calibration_manifest.json"

    @property
    def status(self) -> Path:
        return self.runtime_dir / "status.json"

    @property
    def lock(self) -> Path:
        return self.runtime_dir / "pipeline.lock"

    @property
    def history_lock(self) -> Path:
        """Guards the surface artifacts and their manifest.

        Separate from ``lock`` because split roots let two configs share one
        ``history_dir`` while holding different ``runtime_dir`` locks; without
        this they could replace the same artifacts concurrently and leave a
        manifest sha pointing at the other process's file.
        """
        return self.history_dir / "surface.lock"

    def lock_paths(self) -> List[Path]:
        """Every lock a run must hold, in a fixed order to avoid deadlock.

        Always two distinct files, even in a single-root store: they guard
        different resources (the shared surface history, this run's calibration
        output) and are named differently, so acquiring both never contends
        with itself.
        """
        return [self.history_lock, self.lock]

    def snapshot_path(self, trade_date: str) -> Path:
        return self.snapshots_dir / f"{trade_date}.json"

    def artifact_path(self, trade_date: str) -> Path:
        return artifact_path(self.surface_dir, trade_date)

    def available_snapshot_dates(self) -> List[str]:
        """Trade-date tags with a snapshot on disk, ascending."""
        if not self.snapshots_dir.is_dir():
            return []
        return sorted(
            p.stem for p in self.snapshots_dir.glob("*.json") if p.stem.isdigit()
        )


# ------------------------------------------------------------------ file IO


def _iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path, *, default):
    """Read a JSON file, returning ``default`` only when it does not exist.

    An unreadable or malformed file raises: silently falling back to a default
    would rewrite a corrupt manifest as an empty one and orphan every artifact
    it recorded.
    """
    target = Path(path)
    if not target.is_file():
        return default
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read {target}: {exc}") from exc


def atomic_write_bytes(path, data: bytes) -> None:
    """Write bytes through a same-directory temp file and one ``os.replace``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=target.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path, payload) -> None:
    """Deterministic, atomic JSON write: sorted keys, no NaN, trailing newline."""
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str)
    atomic_write_bytes(path, (text + "\n").encode("utf-8"))


# ----------------------------------------------------------------- manifests


def _require_mapping(payload: Any, source: str) -> Mapping[str, Any]:
    """A manifest must be a JSON object.

    ``read_json`` only guarantees valid JSON, so a structurally corrupt file
    like ``[]`` would otherwise reach ``.get`` and raise ``AttributeError`` --
    escaping the CLI's ``QuantArkException`` handler and losing the
    machine-readable failure the agent contract promises.
    """
    if not isinstance(payload, Mapping):
        raise ValidationError(
            f"{source}: manifest must be a JSON object, got "
            f"{type(payload).__name__}"
        )
    return payload


def _records_by_date(payload: Mapping[str, Any], source: str) -> Dict[str, dict]:
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValidationError(f"{source}: 'records' must be a list")
    out: Dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict) or "date" not in record:
            raise ValidationError(f"{source}: every record needs a 'date'")
        out[str(record["date"])] = dict(record)
    return out


def load_surface_manifest(layout: StoreLayout) -> Tuple[dict, Dict[str, dict]]:
    """Full manifest payload plus its records keyed by date tag."""
    payload = _require_mapping(
        read_json(
            layout.surface_manifest,
            default={"schema_version": SURFACE_MANIFEST_SCHEMA_VERSION, "records": []},
        ),
        str(layout.surface_manifest),
    )
    if payload.get("schema_version") != SURFACE_MANIFEST_SCHEMA_VERSION:
        raise ValidationError(
            f"{layout.surface_manifest}: unsupported schema_version "
            f"{payload.get('schema_version')!r}"
        )
    return payload, _records_by_date(payload, str(layout.surface_manifest))


def save_surface_manifest(
    layout: StoreLayout,
    records: Mapping[str, Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    window: Mapping[str, str],
    price_field: str,
    source_class: Optional[str],
) -> None:
    """Rewrite the surface manifest, preserving foreign top-level blocks.

    ``exclude_thin_surfaces.py`` writes a ``study_admission`` block at top
    level; overwriting the file wholesale would drop the record of why two
    dates are excluded.  Everything the builder does not own is carried
    forward untouched.
    """
    previous = read_json(layout.surface_manifest, default={})
    payload = dict(previous) if isinstance(previous, dict) else {}
    payload.update(
        {
            "schema_version": SURFACE_MANIFEST_SCHEMA_VERSION,
            "source": source_class,
            "price_field": price_field,
            "generated_at": _iso_utc(),
            "window": dict(window),
            "gap_policy": GAP_POLICY,
            "config": dict(config),
            "records": [dict(records[tag]) for tag in sorted(records)],
        }
    )
    atomic_write_json(layout.surface_manifest, payload)


def load_calibration_manifest(layout: StoreLayout) -> Tuple[dict, Dict[str, dict]]:
    """Full calibration-manifest payload plus its records keyed by date tag."""
    payload = _require_mapping(
        read_json(
            layout.calibration_manifest,
            default={
                "schema_version": CALIBRATION_MANIFEST_SCHEMA_VERSION,
                "records": [],
                "bootstrap_policy": BOOTSTRAP_POLICY,
            },
        ),
        str(layout.calibration_manifest),
    )
    if payload.get("schema_version") != CALIBRATION_MANIFEST_SCHEMA_VERSION:
        raise ValidationError(
            f"{layout.calibration_manifest}: unsupported schema_version "
            f"{payload.get('schema_version')!r}"
        )
    return payload, _records_by_date(payload, str(layout.calibration_manifest))


def save_calibration_manifest(
    layout: StoreLayout,
    base_payload: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> None:
    """Rewrite the calibration manifest, keeping its policy metadata."""
    payload = dict(base_payload)
    payload.update(
        {
            "schema_version": CALIBRATION_MANIFEST_SCHEMA_VERSION,
            "generated_at": _iso_utc(),
            "config": dict(config),
            "records": [dict(records[tag]) for tag in sorted(records)],
        }
    )
    payload.setdefault("bootstrap_policy", BOOTSTRAP_POLICY)
    atomic_write_json(layout.calibration_manifest, payload)
