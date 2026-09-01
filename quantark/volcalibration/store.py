"""On-disk contracts: artifact serialization and surface-manifest records.

Artifact bytes are frozen (spec 5.3): their sha256 feeds the calibration cache
key, so provenance lives in manifest records and never in the artifact body.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

# Bump whenever a change to normalization, smoothing or admission CODE alters
# builder output.  Config changes are covered by the fingerprint; code changes
# are not, so this constant is the only mechanism that invalidates artifacts
# after a builder upgrade.  Same obligation as _CACHE_SCHEMA_VERSION in
# quantark/volcalibration/calibrate.py.
BUILDER_SCHEMA_VERSION = 1

PROVENANCE_VERIFIED = "verified"
PROVENANCE_GRANDFATHERED = "grandfathered"


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
    price_field: Optional[str] = None,
    fingerprint: Optional[str] = None,
    provenance: str = PROVENANCE_VERIFIED,
) -> Dict[str, Any]:
    """Build one surface-manifest record."""
    return {
        "date": str(trade_date),
        "status": str(status),
        "reason": reason,
        "detail": detail,
        "n_expiries": int(n_expiries),
        "artifact_sha256": artifact_sha256,
        "snapshot_sha256": snapshot_sha256,
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
