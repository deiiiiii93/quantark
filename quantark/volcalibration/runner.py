"""Orchestration: per-date surface build and per-date calibration.

The runner is a loop, not a framework.  It owns three things the stage
functions deliberately do not: an advisory lock so two invocations cannot
interleave writes, a *checkable* resume rule so "already built" is verified
rather than inferred from a filename's existence, and the freshness status an
agent reads to decide what to do next.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from quantark.util.exceptions import QuantArkException, ValidationError
from quantark.volcalibration.admission import AdmissionError, AdmissionReason
from quantark.volcalibration.config import RunConfig
from quantark.volcalibration.snapshot import (
    CONVENTION_FX_DELTA,
    CONVENTION_LISTED,
    PRICE_FIELD_SETTLEMENT,
    QuoteSnapshot,
)
from quantark.volcalibration.store import (
    BUILDER_SCHEMA_VERSION,
    PROVENANCE_GRANDFATHERED,
    StoreLayout,
    atomic_write_bytes,
    builder_fingerprint,
    load_surface_manifest,
    save_surface_manifest,
    serialize_artifact,
    surface_record,
)
from quantark.volcalibration.surface import build_artifact

Logger = Callable[[str], None]


def _null_log(message: str) -> None:
    return None


# --------------------------------------------------------------- normalizers


def normalizer_for(convention: str, price_field: str):
    """The one pluggable stage: quote convention -> normalizer.

    Listed-live and listed-settlement are separate normalizers rather than one
    with a flag: they disagree on the price field, the liquidity rule, the
    maturity derivation, the expiry-date check and the IV-inversion units, so a
    parameterized version could select combinations no real venue produces.
    """
    if convention == CONVENTION_LISTED:
        if price_field == PRICE_FIELD_SETTLEMENT:
            from quantark.volcalibration.normalize.settlement import (
                SettlementNormalizer,
            )

            return SettlementNormalizer()
        from quantark.volcalibration.normalize.listed import ListedNormalizer

        return ListedNormalizer()
    if convention == CONVENTION_FX_DELTA:
        from quantark.volcalibration.normalize.fxdelta import FxDeltaNormalizer

        return FxDeltaNormalizer()
    raise ValidationError(f"no normalizer for convention {convention!r}")


# ------------------------------------------------------------- the resume rule


def _reason_is_builder_owned(reason: Optional[str]) -> bool:
    """Was this exclusion written by the builder, or by something else?

    ``example/mo_volmodels/exclude_thin_surfaces.py`` marks two dates excluded
    for a study-level reason the builder has no vocabulary for, and keeps their
    artifacts on disk.  Rebuilding such a record would silently re-admit a
    surface a human decided to exclude, so the discriminator is the reason
    itself -- data, not a hardcoded date list.
    """
    if reason is None:
        return True
    try:
        AdmissionReason(reason)
    except ValueError:
        return False
    return True


def surface_record_is_current(
    record: Mapping[str, Any],
    *,
    snapshot_sha: Optional[str],
    price_field: str,
    fingerprint: str,
) -> bool:
    """Is this recorded date still valid for the current inputs and config?"""
    if not record:
        return False
    if record.get("provenance") == PROVENANCE_GRANDFATHERED:
        # Trusted as-is: the artifact bytes are the pinned object, and a
        # rebuild would destroy them to prove a property nobody doubts.
        return True
    if record.get("status") == "excluded" and not _reason_is_builder_owned(
        record.get("reason")
    ):
        return True
    return bool(
        record.get("snapshot_sha256") == snapshot_sha
        and record.get("price_field") == price_field
        and record.get("builder_fingerprint") == fingerprint
        and int(record.get("builder_schema_version", -1)) == BUILDER_SCHEMA_VERSION
    )


def _snapshot_field(layout: StoreLayout, tag: str, *keys: str) -> Optional[Any]:
    """Read one nested field out of a snapshot on disk; None if unreadable."""
    try:
        payload: Any = json.loads(
            layout.snapshot_path(tag).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    for key in keys:
        if not isinstance(payload, Mapping):
            return None
        payload = payload.get(key)
    return payload


def _snapshot_sha_on_disk(layout: StoreLayout, tag: str) -> Optional[str]:
    sha = _snapshot_field(layout, tag, "source", "sha256")
    return str(sha) if sha else None


def plan_surface_dates(
    layout: StoreLayout,
    records: Mapping[str, Mapping[str, Any]],
    config: RunConfig,
    *,
    tags: Optional[Sequence[str]] = None,
    force: bool = False,
) -> List[str]:
    """Which dates a surface run would build. Writes nothing."""
    candidates = list(tags) if tags is not None else layout.available_snapshot_dates()
    fingerprint = builder_fingerprint(config.surface.fingerprint_payload())
    pending: List[str] = []
    for tag in sorted(candidates):
        record = records.get(tag, {})
        if force:
            # --force is the documented way to demand a verified rebuild, and
            # it overrides both bounded exceptions above.
            pending.append(tag)
            continue
        if surface_record_is_current(
            record,
            snapshot_sha=_snapshot_sha_on_disk(layout, tag),
            price_field=config.underlying.price_field,
            fingerprint=fingerprint,
        ):
            continue
        pending.append(tag)
    return pending


# ----------------------------------------------------------- the build worker


def _remove_stale_artifact(artifact_dir: Path, tag: str) -> None:
    """A re-excluded date must not leave a stale artifact for globbing consumers."""
    try:
        (artifact_dir / f"mo_iv_surface_{tag}.json").unlink()
    except FileNotFoundError:
        pass


def build_one_surface(task) -> Dict[str, Any]:
    """Worker: build and atomically write one date's artifact.

    Module-level and tuple-argued so ``ProcessPoolExecutor`` can pickle it.
    A per-date failure returns an ``excluded`` record rather than raising: one
    inadmissible date must not abort a backfill of eight hundred.
    """
    (
        tag,
        snapshot_path,
        artifact_dir,
        convention,
        price_field,
        sabr_beta,
        fingerprint,
    ) = task
    record = surface_record(
        tag, status="excluded", price_field=price_field, fingerprint=fingerprint
    )
    try:
        payload = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
        snapshot = QuoteSnapshot.from_payload(payload)
        record["snapshot_sha256"] = snapshot.sha256
        quotes = normalizer_for(convention, price_field).normalize(snapshot)
        artifact = build_artifact(quotes, snapshot, sabr_beta=float(sabr_beta))
        data = serialize_artifact(artifact)
    except AdmissionError as exc:
        _remove_stale_artifact(Path(artifact_dir), tag)
        record["reason"] = exc.reason.value
        record["detail"] = exc.detail
        return record
    except (OSError, json.JSONDecodeError) as exc:
        _remove_stale_artifact(Path(artifact_dir), tag)
        record["reason"] = AdmissionReason.MISSING_SOURCE.value
        record["detail"] = f"{type(exc).__name__}: {exc}"
        return record
    except QuantArkException as exc:
        _remove_stale_artifact(Path(artifact_dir), tag)
        record["reason"] = AdmissionReason.UNEXPECTED_ERROR.value
        record["detail"] = f"{type(exc).__name__}: {exc}"
        return record

    atomic_write_bytes(Path(artifact_dir) / f"mo_iv_surface_{tag}.json", data)
    record.update(
        status="ok",
        reason=None,
        detail=None,
        n_expiries=len(artifact["maturities"]),
        artifact_sha256=hashlib.sha256(data).hexdigest(),
    )
    return record


# -------------------------------------------------------------- the surface loop


def run_surface_stage(
    layout: StoreLayout,
    config: RunConfig,
    *,
    tags: Optional[Sequence[str]] = None,
    force: bool = False,
    log: Logger = _null_log,
) -> Dict[str, Dict[str, Any]]:
    """Build every pending date, persist the manifest, return all records."""
    _payload, stored = load_surface_manifest(layout)
    records: Dict[str, Dict[str, Any]] = {
        tag: dict(record) for tag, record in stored.items()
    }
    pending = plan_surface_dates(layout, records, config, tags=tags, force=force)
    if not pending:
        log("surfaces: nothing to build")
        return records

    fingerprint = builder_fingerprint(config.surface.fingerprint_payload())
    layout.surface_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            tag,
            str(layout.snapshot_path(tag)),
            str(layout.surface_dir),
            config.underlying.convention,
            config.underlying.price_field,
            float(config.surface.sabr_beta),
            fingerprint,
        )
        for tag in pending
    ]

    results: List[Dict[str, Any]] = []
    if len(tasks) > 1 and int(config.workers) > 1:
        with ProcessPoolExecutor(max_workers=int(config.workers)) as pool:
            for record in pool.map(build_one_surface, tasks):
                results.append(record)
                log(_surface_line(record))
    else:
        for task in tasks:
            record = build_one_surface(task)
            results.append(record)
            log(_surface_line(record))

    for record in results:
        records[record["date"]] = record

    present = sorted(records)
    save_surface_manifest(
        layout,
        records,
        config=config.surface.fingerprint_payload(),
        window={"start": present[0], "end": present[-1]},
        price_field=config.underlying.price_field,
        source_class=_source_class(layout, pending),
    )
    return records


def _surface_line(record: Mapping[str, Any]) -> str:
    suffix = f" ({record['reason']})" if record.get("reason") else ""
    return f"{record['date']}: {record['status']}{suffix}"


def _source_class(layout: StoreLayout, tags: Sequence[str]) -> str:
    """The vendor label the built snapshots declare.

    Read from the snapshot rather than configured: the manifest's ``source``
    field is provenance, and provenance is what the data says about itself.
    """
    for tag in tags:
        vendor = _snapshot_field(layout, tag, "source", "vendor")
        if vendor:
            return str(vendor)
    raise ValidationError(
        f"no snapshot in {layout.snapshots_dir} declares source.vendor; "
        "the manifest's provenance field cannot be written"
    )
