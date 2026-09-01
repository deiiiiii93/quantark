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
import math
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
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
    load_calibration_manifest,
    load_surface_manifest,
    save_calibration_manifest,
    save_surface_manifest,
    serialize_artifact,
    surface_record,
)
from quantark.volcalibration.surface import build_artifact

Logger = Callable[[str], None]


def _null_log(message: str) -> None:
    return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


# ------------------------------------------------- Heston temporal smoothing

HESTON_PARAMETER_NAMES = ("v0", "kappa", "theta", "sigma", "rho")
HESTON_STRUCTURAL_PARAMETER_NAMES = ("kappa", "theta", "sigma", "rho")


def heston_params_payload(record: Mapping[str, Any]) -> Optional[Dict[str, float]]:
    """Extract a complete Heston vector from a calibration record."""
    try:
        payload = {name: float(record[name]) for name in HESTON_PARAMETER_NAMES}
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in payload.values()):
        return None
    return payload


def heston_vector(payload: Mapping[str, float]):
    return tuple(float(payload[name]) for name in HESTON_PARAMETER_NAMES)


def heston_feller_diagnostics(payload: Mapping[str, float]) -> Dict[str, Any]:
    """2*kappa*theta vs sigma^2 -- the ratio and the verdict."""
    numerator = 2.0 * float(payload["kappa"]) * float(payload["theta"])
    denominator = float(payload["sigma"]) ** 2
    return {
        "feller_ratio": numerator / denominator,
        "feller_satisfied": bool(numerator >= denominator),
    }


def raw_heston_from_calibration_record(
    record: Mapping[str, Any],
) -> Optional[Dict[str, float]]:
    """The independent daily Heston fit stored for one date.

    Temporal records carry the raw fit explicitly.  Legacy independent records
    use the governed Heston variant itself, which lets an opt-in transition
    seed its EWMA from already-audited history.
    """
    temporal = record.get("temporal_scheme", {})
    if isinstance(temporal, Mapping):
        raw = temporal.get("raw_heston")
        if isinstance(raw, Mapping):
            extracted = heston_params_payload(raw)
            if extracted is not None:
                return extracted
    variants = record.get("variants", {})
    if not isinstance(variants, Mapping):
        return None
    heston = variants.get("heston", {})
    if not isinstance(heston, Mapping) or heston.get("status") != "ok":
        return None
    governed = heston.get("record", {})
    if not isinstance(governed, Mapping):
        return None
    return heston_params_payload(governed)


def structural_ewma_before(
    calibration_records: Mapping[str, Mapping[str, Any]],
    *,
    before_date: str,
    span: int,
) -> Optional[Dict[str, Any]]:
    """Recursive span-N EWMA of prior admitted raw structural Heston fits."""
    alpha = 2.0 / (float(span) + 1.0)
    state: Optional[Dict[str, float]] = None
    source_dates: List[str] = []
    for trade_date, record in sorted(calibration_records.items()):
        if trade_date >= before_date or record.get("status") != "ok":
            continue
        raw = raw_heston_from_calibration_record(record)
        if raw is None:
            continue
        if not heston_feller_diagnostics(raw)["feller_satisfied"]:
            raise ValidationError(
                f"{trade_date}: historical raw Heston seed violates the "
                "Feller constraint"
            )
        if state is None:
            state = {name: raw[name] for name in HESTON_STRUCTURAL_PARAMETER_NAMES}
        else:
            state = {
                name: alpha * raw[name] + (1.0 - alpha) * state[name]
                for name in HESTON_STRUCTURAL_PARAMETER_NAMES
            }
        source_dates.append(trade_date)
    if state is None:
        return None
    return {
        "parameters": state,
        "span": int(span),
        "alpha": alpha,
        "observation_count": len(source_dates),
        "first_source_date": source_dates[0],
        "last_source_date": source_dates[-1],
    }


def update_structural_ewma(
    prior: Optional[Mapping[str, Any]],
    raw_heston: Mapping[str, float],
    *,
    trade_date: str,
    span: int,
) -> Dict[str, Any]:
    """Add today's raw structural fit to a prior EWMA state."""
    alpha = 2.0 / (float(span) + 1.0)
    if prior is None:
        parameters = {
            name: float(raw_heston[name])
            for name in HESTON_STRUCTURAL_PARAMETER_NAMES
        }
        count = 1
        first_source_date = trade_date
    else:
        prior_parameters = prior["parameters"]
        parameters = {
            name: alpha * float(raw_heston[name])
            + (1.0 - alpha) * float(prior_parameters[name])
            for name in HESTON_STRUCTURAL_PARAMETER_NAMES
        }
        count = int(prior["observation_count"]) + 1
        first_source_date = str(prior["first_source_date"])
    return {
        "parameters": parameters,
        "span": int(span),
        "alpha": alpha,
        "observation_count": count,
        "first_source_date": first_source_date,
        "last_source_date": trade_date,
    }


def combine_daily_v0_and_structure(
    raw_heston: Mapping[str, float],
    structural_state: Mapping[str, Any],
) -> Dict[str, float]:
    """The reference vector: today's unaveraged v0 over smoothed structure."""
    return {
        "v0": float(raw_heston["v0"]),
        **{
            name: float(structural_state["parameters"][name])
            for name in HESTON_STRUCTURAL_PARAMETER_NAMES
        },
    }


# ----------------------------------------------------------- date selection


def calibration_record_is_current(
    record: Mapping[str, Any],
    surface_record_: Mapping[str, Any],
    config: Mapping[str, Any],
    variants: Sequence[str],
) -> bool:
    """Does this record already cover today's surface, config and variants?"""
    return bool(
        record.get("status") == "ok"
        and record.get("surface_sha") == surface_record_.get("artifact_sha256")
        and record.get("config") == config
        and all(
            record.get("variants", {}).get(variant, {}).get("status") == "ok"
            for variant in variants
        )
    )


def select_calibration_dates(
    surface_records: Mapping[str, Mapping[str, Any]],
    calibration_records: Mapping[str, Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    backfill: bool,
    max_dates: Optional[int],
    baseline_date: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    variants: Sequence[str],
) -> List[str]:
    """Resumable calibration work, without an accidental multi-year bootstrap."""
    eligible = [
        trade_date
        for trade_date, record in sorted(surface_records.items())
        if (
            record.get("status") == "ok"
            and (start_date is None or trade_date >= start_date)
            and (end_date is None or trade_date <= end_date)
        )
    ]
    if not eligible:
        return []
    if baseline_date is not None and not backfill:
        eligible = [d for d in eligible if d >= baseline_date]

    stale = [
        trade_date
        for trade_date in eligible
        if not calibration_record_is_current(
            calibration_records.get(trade_date, {}),
            surface_records[trade_date],
            config,
            variants,
        )
    ]
    if not backfill:
        latest_existing = max(calibration_records) if calibration_records else None
        if (
            latest_existing is not None
            and calibration_records[latest_existing].get("config") == config
        ):
            # Incremental operation for an established configuration starts at
            # its latest record.  Older surfaces need an explicit backfill.
            stale = [d for d in stale if d >= latest_existing]
        else:
            # First use of a new opt-in scheme bootstraps only the latest
            # admitted surface: a scheduler config change must not turn into
            # an accidental multi-year SLV backfill.
            stale = stale[-1:]
    if max_dates is not None:
        stale = stale[:max_dates]
    return stale


# ------------------------------------------------------- the calibration loop


def calibrate_one_surface(
    trade_date: str,
    *,
    layout: StoreLayout,
    config: RunConfig,
    calibration_records: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Calibrate every configured variant for one admitted surface."""
    from quantark.param.vol.surface_history import IvSurfaceArtifact
    from quantark.volcalibration.calibrate import VolModelCalibrator

    calibration = config.calibration
    variants = tuple(calibration.variants)
    artifact = IvSurfaceArtifact.from_file(layout.artifact_path(trade_date))
    base_config = calibration.calibrator_config(layout.calibration_cache)
    started = _utc_now()
    variant_records: Dict[str, Dict[str, Any]] = {}
    temporal_audit: Optional[Dict[str, Any]] = None

    if calibration.temporal_smoothing:
        calibrator, temporal_audit = _temporal_calibrator(
            trade_date,
            artifact=artifact,
            calibration=calibration,
            base_config=base_config,
            calibration_records=calibration_records or {},
        )
    else:
        calibrator = VolModelCalibrator(base_config)

    for variant in variants:
        variant_started = time.perf_counter()
        if (
            temporal_audit is not None
            and temporal_audit.get("status") == "failed"
            and variant != "localvol"
        ):
            # Local Vol stays independently diagnosable; the two
            # Heston-dependent variants fail closed on the raw-fit dependency.
            variant_records[variant] = {
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - variant_started,
                "error_type": "DependencyError",
                "error": (
                    "raw daily Heston calibration failed: "
                    f"{temporal_audit['raw_heston']['error']}"
                ),
            }
            continue
        try:
            model = calibrator.calibrate(variant, artifact)
        except Exception as exc:  # fail-closed, but keep every attempted record
            variant_records[variant] = {
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - variant_started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        else:
            variant_records[variant] = {
                "status": "ok",
                "elapsed_seconds": time.perf_counter() - variant_started,
                "record": dict(model.record),
            }

    completed = _utc_now()
    status = (
        "ok"
        if all(item.get("status") == "ok" for item in variant_records.values())
        else "failed"
    )
    record = {
        "date": trade_date,
        "surface_sha": artifact.sha256,
        "surface_path": str(artifact.path),
        "status": status,
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "config": calibration.manifest_payload(),
        "variants": variant_records,
    }
    if temporal_audit is not None:
        record["temporal_scheme"] = temporal_audit
    return record


def _temporal_calibrator(
    trade_date: str,
    *,
    artifact,
    calibration,
    base_config,
    calibration_records: Mapping[str, Mapping[str, Any]],
):
    """Build the EWMA-regularized calibrator and its audit block."""
    from quantark.volcalibration.calibrate import VolModelCalibrator
    from quantark.volcalibration.config import TEMPORAL_SCHEME

    raw_calibrator = VolModelCalibrator(base_config)
    raw_started = time.perf_counter()
    try:
        raw_model = raw_calibrator.calibrate("heston", artifact)
    except Exception as exc:
        audit = {
            "name": TEMPORAL_SCHEME,
            "status": "failed",
            "raw_heston": {
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - raw_started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        }
        return raw_calibrator, audit

    raw_record = dict(raw_model.record)
    raw_heston = heston_params_payload(raw_record)
    if raw_heston is None:
        raise ValidationError(
            f"{trade_date}: raw Heston record has no complete parameter vector"
        )
    if not heston_feller_diagnostics(raw_heston)["feller_satisfied"]:
        raise ValidationError(
            f"{trade_date}: raw Heston calibration violates the Feller constraint"
        )

    span = int(calibration.structural_ewma_span)
    prior = structural_ewma_before(
        calibration_records, before_date=trade_date, span=span
    )
    updated = update_structural_ewma(
        prior, raw_heston, trade_date=trade_date, span=span
    )
    # Pure Heston is regularized toward the prior structural state; on the
    # first observation, bootstrap against today's raw fit.
    heston_reference = combine_daily_v0_and_structure(raw_heston, prior or updated)
    # SLV consumes the state *after* incorporating today's raw fit.
    slv_heston = combine_daily_v0_and_structure(raw_heston, updated)
    slv_feller = heston_feller_diagnostics(slv_heston)
    if not slv_feller["feller_satisfied"]:
        raise ValidationError(
            f"{trade_date}: structural EWMA produced a Feller-violating "
            "SLV Heston vector"
        )

    from dataclasses import replace as _replace

    temporal_config = _replace(
        base_config,
        heston_temporal_reference=heston_vector(heston_reference),
        heston_temporal_regularization=float(
            calibration.heston_temporal_regularization
        ),
        slv_heston_override=heston_vector(slv_heston),
    )
    audit = {
        "name": TEMPORAL_SCHEME,
        "status": "ok",
        "daily_parameters": ["v0"],
        "structural_parameters": list(HESTON_STRUCTURAL_PARAMETER_NAMES),
        "raw_heston": raw_record,
        "prior_structural_ewma": prior,
        "updated_structural_ewma": updated,
        "heston_temporal_reference": heston_reference,
        "heston_temporal_regularization": float(
            calibration.heston_temporal_regularization
        ),
        "slv_heston": slv_heston,
        "slv_heston_feller_ratio": slv_feller["feller_ratio"],
        "slv_heston_feller_satisfied": slv_feller["feller_satisfied"],
    }
    return VolModelCalibrator(temporal_config), audit


def run_calibration_stage(
    layout: StoreLayout,
    config: RunConfig,
    *,
    surface_records: Mapping[str, Mapping[str, Any]],
    backfill: bool = False,
    max_dates: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    log: Logger = _null_log,
) -> Dict[str, Dict[str, Any]]:
    """Calibrate every selected admitted surface, persisting after each date."""
    base_payload, stored = load_calibration_manifest(layout)
    records: Dict[str, Dict[str, Any]] = {
        tag: dict(record) for tag, record in stored.items()
    }
    payload = config.calibration.manifest_payload()
    selected = select_calibration_dates(
        surface_records,
        records,
        config=payload,
        backfill=backfill,
        max_dates=max_dates,
        baseline_date=base_payload.get("baseline_date")
        or (min(records) if records else None),
        start_date=start_date,
        end_date=end_date,
        variants=tuple(config.calibration.variants),
    )
    if (
        selected
        and not records
        and not backfill
        and base_payload.get("baseline_date") is None
    ):
        base_payload["baseline_date"] = selected[0]

    if not selected:
        log("calibration: nothing to do")
        return records

    for tag in selected:
        log(f"{tag}: calibrating {', '.join(config.calibration.variants)}")
        record = calibrate_one_surface(
            tag, layout=layout, config=config, calibration_records=records
        )
        records[tag] = record
        # Persist after every date: an interrupted backfill resumes from here.
        save_calibration_manifest(layout, base_payload, records, config=payload)
        log(f"{tag}: {record['status']} [{record['elapsed_seconds']:.2f}s]")
    return records
