"""Operational daily MO settlement -> surface -> Heston/SLV calibration pipeline.

This command turns the staged MO examples into one resumable EOD workflow:

1. refresh CSI 1000 spot and IM futures caches through the requested as-of date;
2. fetch every newly observed CFFEX settlement CSV;
3. build arbitrage-checked SABR IV surfaces for new settlement dates;
4. calibrate Local Vol, hard-Feller Heston, and Heston-SLV once per admitted
   surface, using the canonical SHA/config-keyed persistent cache;
5. publish an atomic calibration manifest and freshness/status artifact.

Pass ``--temporal-smoothing`` to opt into daily ``v0`` plus a five-day
structural EWMA for SLV, and structural temporal regularization for pure
Heston.  Independent daily calibration remains the default.

The first run intentionally bootstraps at the latest admitted surface instead
of attempting a multi-year SLV backfill.  Later runs calibrate every newly
admitted date.  Pass ``--backfill-calibrations`` for an explicit backfill,
optionally bounded by ``--calibration-start`` and ``--calibration-end``.

Examples::

    .venv/bin/python example/mo_volmodels/14_daily_calibration_pipeline.py run
    .venv/bin/python example/mo_volmodels/14_daily_calibration_pipeline.py status

Exit codes for ``run`` and ``status``:

* 0: current (calibration date equals the latest refreshed trading date);
* 2: non-current but fail-closed (source pending, surface excluded, or stale);
* 1: pipeline/calibration failure;
* 75: another pipeline process owns the lock.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_HISTORY_DIR = HERE / "data" / "history"
DEFAULT_RUNTIME_DIR = PROJECT_ROOT / "output" / "mo_daily_calibration"
DEFAULT_STAGE01_PYTHON = Path("/opt/anaconda3/bin/python")
DEFAULT_STAGE_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

REFRESH_SCRIPT = HERE / "01_refresh_market_cache.py"
SETTLEMENT_SCRIPT = HERE / "01_bulk_fetch_settlement_history.py"
SURFACE_SCRIPT = HERE / "03_build_iv_surface_history.py"

SPOT_FILENAME = "csi1000_spot.csv"
SETTLEMENT_MANIFEST_FILENAME = "settlement_manifest.json"
SURFACE_MANIFEST_FILENAME = "surface_manifest.json"
CALIBRATION_MANIFEST_FILENAME = "calibration_manifest.json"
STATUS_FILENAME = "status.json"
LOCK_FILENAME = "pipeline.lock"
CALIBRATION_CACHE_DIRNAME = "calibration_cache"

SHANGHAI = ZoneInfo("Asia/Shanghai")
SCHEMA_VERSION = 1
VARIANTS = ("localvol", "heston", "heston_slv")
HESTON_PARAMETER_NAMES = ("v0", "kappa", "theta", "sigma", "rho")
HESTON_STRUCTURAL_PARAMETER_NAMES = ("kappa", "theta", "sigma", "rho")
TEMPORAL_SCHEME = "daily_v0_structural_ewma"
EXIT_CURRENT = 0
EXIT_FAILED = 1
EXIT_NON_CURRENT = 2
EXIT_LOCKED = 75
TAIL_CHARS = 8_000
STATUS_CACHE_MAX_AGE_DAYS = 4


class PipelineError(RuntimeError):
    """A pipeline stage failed and the run must stop fail-closed."""


class LockBusy(PipelineError):
    """Another daily pipeline invocation owns the runtime lock."""


@dataclass(frozen=True)
class PipelinePaths:
    history_dir: Path
    runtime_dir: Path

    @property
    def spot_csv(self) -> Path:
        return self.history_dir / SPOT_FILENAME

    @property
    def raw_dir(self) -> Path:
        return self.history_dir / "settlement_csv"

    @property
    def settlement_manifest(self) -> Path:
        return self.history_dir / SETTLEMENT_MANIFEST_FILENAME

    @property
    def surface_dir(self) -> Path:
        return self.history_dir / "iv_surface"

    @property
    def surface_manifest(self) -> Path:
        return self.history_dir / SURFACE_MANIFEST_FILENAME

    @property
    def calibration_manifest(self) -> Path:
        return self.runtime_dir / CALIBRATION_MANIFEST_FILENAME

    @property
    def calibration_cache(self) -> Path:
        return self.runtime_dir / CALIBRATION_CACHE_DIRNAME

    @property
    def status(self) -> Path:
        return self.runtime_dir / STATUS_FILENAME

    @property
    def lock(self) -> Path:
        return self.runtime_dir / LOCK_FILENAME


@dataclass
class StageResult:
    name: str
    command: list[str]
    started_at: str
    completed_at: str
    elapsed_seconds: float
    returncode: int
    stdout_tail: str
    stderr_tail: str


StageRunner = Callable[[str, Sequence[str]], StageResult]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime | None = None) -> str:
    return (value or utc_now()).astimezone(timezone.utc).isoformat()


def parse_as_of(value: str | None) -> date:
    if value is None:
        return datetime.now(SHANGHAI).date()
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid --as-of {value!r}; expected YYYY-MM-DD"
        ) from exc


def parse_date_tag(value: str) -> str:
    """Normalize an ISO or compact calendar date to YYYYMMDD."""
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"invalid date {value!r}; expected YYYY-MM-DD or YYYYMMDD"
    )


def compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def read_json(path: Path, *, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"cannot read JSON artifact {path}: {exc}") from exc


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
                default=str,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@contextmanager
def acquire_lock(path: Path) -> Iterator[None]:
    """Acquire a non-blocking advisory lock for the whole daily transaction."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            owner = handle.read().strip()
            detail = f" ({owner})" if owner else ""
            raise LockBusy(f"MO daily pipeline already running{detail}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            compact_json(
                {
                    "pid": os.getpid(),
                    "started_at": iso_utc(),
                    "cwd": str(PROJECT_ROOT),
                }
            )
        )
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            handle.seek(0)
            handle.truncate()
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def default_stage_runner(name: str, command: Sequence[str]) -> StageResult:
    started = utc_now()
    t0 = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    ended = utc_now()
    result = StageResult(
        name=name,
        command=[str(item) for item in command],
        started_at=iso_utc(started),
        completed_at=iso_utc(ended),
        elapsed_seconds=time.perf_counter() - t0,
        returncode=int(completed.returncode),
        stdout_tail=(completed.stdout or "")[-TAIL_CHARS:],
        stderr_tail=(completed.stderr or "")[-TAIL_CHARS:],
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(
            completed.stderr,
            end="" if completed.stderr.endswith("\n") else "\n",
            file=sys.stderr,
        )
    if completed.returncode != 0:
        raise PipelineError(
            f"{name} failed with exit code {completed.returncode}: "
            f"{result.stderr_tail or result.stdout_tail}"
        )
    return result


def validate_python(path: Path, label: str) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise PipelineError(f"{label} interpreter is unavailable or not executable: {path}")


def load_trading_dates(spot_csv: Path, *, as_of: date | None = None) -> list[str]:
    if not spot_csv.is_file():
        raise PipelineError(f"spot trading-calendar cache is missing: {spot_csv}")
    dates: list[str] = []
    try:
        with spot_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                raw = str(row.get("date", "")).strip()
                if not raw:
                    continue
                try:
                    parsed = date.fromisoformat(raw)
                except ValueError:
                    continue
                if as_of is None or parsed <= as_of:
                    dates.append(parsed.strftime("%Y%m%d"))
    except OSError as exc:
        raise PipelineError(f"cannot read spot trading calendar {spot_csv}: {exc}") from exc
    unique = sorted(set(dates))
    if not unique:
        raise PipelineError(f"spot trading calendar {spot_csv} has no usable dates")
    return unique


def records_by_date(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise PipelineError("manifest records must be a list")
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or "date" not in record:
            raise PipelineError("manifest contains a malformed record")
        out[str(record["date"])] = dict(record)
    return out


def latest_record_date(
    records: Mapping[str, Mapping[str, Any]],
    *,
    status: str | None = None,
    require_file: Callable[[str], bool] | None = None,
) -> str | None:
    candidates = []
    for trade_date, record in records.items():
        if status is not None and record.get("status") != status:
            continue
        if require_file is not None and not require_file(trade_date):
            continue
        candidates.append(trade_date)
    return max(candidates) if candidates else None


def day_after(tag: str) -> str:
    parsed = datetime.strptime(tag, "%Y%m%d").date()
    return (parsed + timedelta(days=1)).strftime("%Y%m%d")


def surface_artifact_path(paths: PipelinePaths, trade_date: str) -> Path:
    return paths.surface_dir / f"mo_iv_surface_{trade_date}.json"


def calibration_config_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "variants": list(VARIANTS),
        "heston_preset": "mo_frozen",
        "heston_max_nfev": int(args.heston_max_nfev),
        "slv_eta": float(args.slv_eta),
        "slv_n_steps": int(args.slv_n_steps),
        "slv_n_x": int(args.slv_n_x),
        "slv_n_z": int(args.slv_n_z),
    }
    if bool(getattr(args, "temporal_smoothing", False)):
        payload["temporal_scheme"] = {
            "name": TEMPORAL_SCHEME,
            "structural_ewma_span": int(args.structural_ewma_span),
            "structural_ewma_alpha": float(
                2.0 / (int(args.structural_ewma_span) + 1.0)
            ),
            "heston_temporal_regularization": float(
                args.heston_temporal_regularization
            ),
            "daily_parameters": ["v0"],
            "structural_parameters": list(
                HESTON_STRUCTURAL_PARAMETER_NAMES
            ),
        }
    return payload


def heston_params_payload(record: Mapping[str, Any]) -> dict[str, float] | None:
    """Extract a complete Heston vector from a calibration record."""
    try:
        payload = {
            name: float(record[name]) for name in HESTON_PARAMETER_NAMES
        }
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in payload.values()):
        return None
    return payload


def raw_heston_from_calibration_record(
    record: Mapping[str, Any],
) -> dict[str, float] | None:
    """Return the independent daily Heston fit stored for one date.

    Temporal records carry the raw fit explicitly.  Legacy independent
    records use the governed Heston variant itself, which makes an opt-in
    transition able to seed its EWMA from already-audited history.
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
    governed_record = heston.get("record", {})
    if not isinstance(governed_record, Mapping):
        return None
    return heston_params_payload(governed_record)


def structural_ewma_before(
    calibration_records: Mapping[str, Mapping[str, Any]],
    *,
    before_date: str,
    span: int,
) -> dict[str, Any] | None:
    """Recursive span-N EWMA of prior admitted raw structural Heston fits."""
    alpha = 2.0 / (float(span) + 1.0)
    state: dict[str, float] | None = None
    source_dates: list[str] = []
    for trade_date, record in sorted(calibration_records.items()):
        if trade_date >= before_date or record.get("status") != "ok":
            continue
        raw = raw_heston_from_calibration_record(record)
        if raw is None:
            continue
        if not heston_feller_diagnostics(raw)["feller_satisfied"]:
            raise PipelineError(
                f"{trade_date}: historical raw Heston seed violates "
                "the Feller constraint"
            )
        if state is None:
            state = {
                name: raw[name] for name in HESTON_STRUCTURAL_PARAMETER_NAMES
            }
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
    prior: Mapping[str, Any] | None,
    raw_heston: Mapping[str, float],
    *,
    trade_date: str,
    span: int,
) -> dict[str, Any]:
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
) -> dict[str, float]:
    """Build the SLV/Heston reference vector with today's unaveraged v0."""
    return {
        "v0": float(raw_heston["v0"]),
        **{
            name: float(structural_state["parameters"][name])
            for name in HESTON_STRUCTURAL_PARAMETER_NAMES
        },
    }


def heston_vector(payload: Mapping[str, float]) -> tuple[float, ...]:
    return tuple(float(payload[name]) for name in HESTON_PARAMETER_NAMES)


def heston_feller_diagnostics(
    payload: Mapping[str, float],
) -> dict[str, Any]:
    numerator = (
        2.0 * float(payload["kappa"]) * float(payload["theta"])
    )
    denominator = float(payload["sigma"]) ** 2
    return {
        "feller_ratio": numerator / denominator,
        "feller_satisfied": bool(numerator >= denominator),
    }


def load_calibration_records(path: Path) -> tuple[dict[str, Any], dict[str, dict]]:
    payload = read_json(
        path,
        default={
            "schema_version": SCHEMA_VERSION,
            "records": [],
            "bootstrap_policy": "latest_admitted_surface_only",
        },
    )
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PipelineError(
            f"unsupported calibration manifest schema in {path}: "
            f"{payload.get('schema_version')!r}"
        )
    return payload, records_by_date(payload)


def calibration_record_is_current(
    record: Mapping[str, Any],
    surface_record: Mapping[str, Any],
    config: Mapping[str, Any],
) -> bool:
    return bool(
        record.get("status") == "ok"
        and record.get("surface_sha") == surface_record.get("artifact_sha256")
        and record.get("config") == config
        and all(
            record.get("variants", {}).get(variant, {}).get("status") == "ok"
            for variant in VARIANTS
        )
    )


def select_calibration_dates(
    surface_records: Mapping[str, Mapping[str, Any]],
    calibration_records: Mapping[str, Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    backfill: bool,
    max_dates: int | None,
    baseline_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[str]:
    """Select resumable calibration work without accidental multi-year bootstrap."""
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
        eligible = [trade_date for trade_date in eligible if trade_date >= baseline_date]

    stale = [
        trade_date
        for trade_date in eligible
        if not calibration_record_is_current(
            calibration_records.get(trade_date, {}),
            surface_records[trade_date],
            config,
        )
    ]
    if not backfill:
        latest_existing = (
            max(calibration_records) if calibration_records else None
        )
        if (
            latest_existing is not None
            and calibration_records[latest_existing].get("config") == config
        ):
            # Incremental operation for an established configuration starts at
            # its latest record.  Older surfaces need an explicit backfill,
            # even when another calibration scheme occupies those dates.
            stale = [
                trade_date for trade_date in stale
                if trade_date >= latest_existing
            ]
        elif latest_existing is not None:
            # First use of a new opt-in scheme bootstraps only the latest
            # admitted surface; do not turn a scheduler config change into an
            # accidental multi-year SLV backfill.
            stale = stale[-1:]
        else:
            stale = stale[-1:]
    if max_dates is not None:
        stale = stale[:max_dates]
    return stale


def calibrate_one_surface(
    trade_date: str,
    *,
    paths: PipelinePaths,
    args: argparse.Namespace,
    calibration_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Calibrate all governed variants for one admitted surface."""
    from quantark.backtest.replay.config import VolModelCalibrationConfig
    from quantark.param.vol.surface_history import IvSurfaceArtifact
    from quantark.volmodels.calibration import VolModelCalibrator

    artifact = IvSurfaceArtifact.from_file(surface_artifact_path(paths, trade_date))
    base_config_kwargs = dict(
        cache_dir=str(paths.calibration_cache),
        heston_preset="mo_frozen",
        heston_max_nfev=int(args.heston_max_nfev),
        slv_eta=float(args.slv_eta),
        slv_n_steps=int(args.slv_n_steps),
        slv_n_x=int(args.slv_n_x),
        slv_n_z=int(args.slv_n_z),
    )
    started = utc_now()
    variants: dict[str, dict[str, Any]] = {}
    temporal_audit: dict[str, Any] | None = None

    if bool(getattr(args, "temporal_smoothing", False)):
        raw_calibrator = VolModelCalibrator(
            VolModelCalibrationConfig(**base_config_kwargs)
        )
        raw_started = time.perf_counter()
        try:
            raw_model = raw_calibrator.calibrate("heston", artifact)
        except Exception as exc:
            raw_failure = {
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - raw_started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            temporal_audit = {
                "name": TEMPORAL_SCHEME,
                "status": "failed",
                "raw_heston": raw_failure,
            }
            # Local Vol remains independently diagnosable, while the two
            # Heston-dependent variants fail closed on the raw-fit dependency.
            calibrator = raw_calibrator
        else:
            raw_record = dict(raw_model.record)
            raw_heston = heston_params_payload(raw_record)
            if raw_heston is None:
                raise PipelineError(
                    f"{trade_date}: raw Heston record has no complete parameter vector"
                )
            if not heston_feller_diagnostics(raw_heston)[
                "feller_satisfied"
            ]:
                raise PipelineError(
                    f"{trade_date}: raw Heston calibration violates "
                    "the Feller constraint"
                )
            span = int(args.structural_ewma_span)
            prior = structural_ewma_before(
                calibration_records or {},
                before_date=trade_date,
                span=span,
            )
            updated = update_structural_ewma(
                prior,
                raw_heston,
                trade_date=trade_date,
                span=span,
            )
            # Pure Heston is regularized toward the prior structural state.
            # On the first observation, bootstrap against today's raw fit.
            heston_reference_state = prior or updated
            heston_reference = combine_daily_v0_and_structure(
                raw_heston, heston_reference_state
            )
            # SLV consumes the state after incorporating today's raw fit.
            slv_heston = combine_daily_v0_and_structure(
                raw_heston, updated
            )
            slv_feller = heston_feller_diagnostics(slv_heston)
            if not slv_feller["feller_satisfied"]:
                raise PipelineError(
                    f"{trade_date}: structural EWMA produced a "
                    "Feller-violating SLV Heston vector"
                )
            temporal_config = VolModelCalibrationConfig(
                **base_config_kwargs,
                heston_temporal_reference=heston_vector(heston_reference),
                heston_temporal_regularization=float(
                    args.heston_temporal_regularization
                ),
                slv_heston_override=heston_vector(slv_heston),
            )
            calibrator = VolModelCalibrator(temporal_config)
            temporal_audit = {
                "name": TEMPORAL_SCHEME,
                "status": "ok",
                "daily_parameters": ["v0"],
                "structural_parameters": list(
                    HESTON_STRUCTURAL_PARAMETER_NAMES
                ),
                "raw_heston": raw_record,
                "prior_structural_ewma": prior,
                "updated_structural_ewma": updated,
                "heston_temporal_reference": heston_reference,
                "heston_temporal_regularization": float(
                    args.heston_temporal_regularization
                ),
                "slv_heston": slv_heston,
                "slv_heston_feller_ratio": slv_feller["feller_ratio"],
                "slv_heston_feller_satisfied": slv_feller[
                    "feller_satisfied"
                ],
            }
    else:
        calibrator = VolModelCalibrator(
            VolModelCalibrationConfig(**base_config_kwargs)
        )

    for variant in VARIANTS:
        variant_started = time.perf_counter()
        if (
            temporal_audit is not None
            and temporal_audit.get("status") == "failed"
            and variant != "localvol"
        ):
            raw_failure = temporal_audit["raw_heston"]
            variants[variant] = {
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - variant_started,
                "error_type": "DependencyError",
                "error": (
                    "raw daily Heston calibration failed: "
                    f"{raw_failure['error']}"
                ),
            }
            continue
        try:
            model = calibrator.calibrate(variant, artifact)
        except Exception as exc:  # fail-closed, but preserve every attempted record
            variants[variant] = {
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - variant_started,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        else:
            variants[variant] = {
                "status": "ok",
                "elapsed_seconds": time.perf_counter() - variant_started,
                "record": dict(model.record),
            }
    completed = utc_now()
    status = (
        "ok"
        if all(item.get("status") == "ok" for item in variants.values())
        else "failed"
    )
    record = {
        "date": trade_date,
        "surface_sha": artifact.sha256,
        "surface_path": str(artifact.path),
        "status": status,
        "started_at": iso_utc(started),
        "completed_at": iso_utc(completed),
        "elapsed_seconds": (completed - started).total_seconds(),
        "config": calibration_config_payload(args),
        "variants": variants,
    }
    if temporal_audit is not None:
        record["temporal_scheme"] = temporal_audit
    return record


def persist_calibration_manifest(
    path: Path,
    *,
    base_payload: MutableMapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    payload = dict(base_payload)
    payload.update(
        schema_version=SCHEMA_VERSION,
        generated_at=iso_utc(),
        config=dict(config),
        records=[dict(records[tag]) for tag in sorted(records)],
    )
    atomic_write_json(path, payload)


def trading_day_lag(
    trading_dates: Sequence[str], latest_date: str | None, expected_date: str
) -> int | None:
    if latest_date is None:
        return None
    index = {value: position for position, value in enumerate(trading_dates)}
    if latest_date not in index or expected_date not in index:
        return None
    return max(0, index[expected_date] - index[latest_date])


def build_freshness_status(
    *,
    paths: PipelinePaths,
    as_of: date,
    stages: Sequence[StageResult] = (),
    run_id: str | None = None,
    last_error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    trading_dates = load_trading_dates(paths.spot_csv, as_of=as_of)
    expected = trading_dates[-1]

    settlement_payload = read_json(paths.settlement_manifest, default={"records": []})
    settlement_records = records_by_date(settlement_payload)
    settlement_latest = latest_record_date(
        settlement_records,
        status="ok",
        require_file=lambda tag: (
            paths.raw_dir / f"{tag}_1.csv"
        ).is_file(),
    )

    surface_payload = read_json(paths.surface_manifest, default={"records": []})
    surface_records = records_by_date(surface_payload)
    surface_latest = latest_record_date(
        surface_records,
        status="ok",
        require_file=lambda tag: surface_artifact_path(paths, tag).is_file(),
    )

    _calibration_payload, calibration_records = load_calibration_records(
        paths.calibration_manifest
    )
    calibration_latest = latest_record_date(calibration_records, status="ok")

    expected_settlement = settlement_records.get(expected, {})
    expected_surface = surface_records.get(expected, {})
    expected_calibration = calibration_records.get(expected, {})
    cache_age_days = (
        as_of - datetime.strptime(trading_dates[-1], "%Y%m%d").date()
    ).days
    market_refresh_completed = any(
        stage.name == "refresh_market_cache" and stage.returncode == 0
        for stage in stages
    )

    if last_error is not None:
        overall = "failed"
    elif cache_age_days > STATUS_CACHE_MAX_AGE_DAYS and not market_refresh_completed:
        overall = "market_cache_stale"
    elif expected_settlement.get("status") != "ok":
        overall = "source_pending"
    elif expected_surface.get("status") == "excluded":
        overall = "surface_excluded"
    elif expected_surface.get("status") != "ok":
        overall = "surface_pending"
    elif expected_calibration.get("status") == "failed":
        overall = "calibration_failed"
    elif expected_calibration.get("status") != "ok":
        overall = "calibration_pending"
    else:
        overall = "current"

    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline": "mo_daily_calibration",
        "overall_status": overall,
        "generated_at": iso_utc(),
        "timezone": "Asia/Shanghai",
        "as_of_date": as_of.isoformat(),
        "run_id": run_id,
        "expected_trade_date": expected,
        "freshness": {
            "spot_cache_latest": trading_dates[-1],
            "spot_cache_calendar_age_days": cache_age_days,
            "spot_cache_max_age_days": STATUS_CACHE_MAX_AGE_DAYS,
            "market_refresh_completed_this_run": market_refresh_completed,
            "settlement_latest": settlement_latest,
            "settlement_lag_trading_days": trading_day_lag(
                trading_dates, settlement_latest, expected
            ),
            "surface_latest": surface_latest,
            "surface_lag_trading_days": trading_day_lag(
                trading_dates, surface_latest, expected
            ),
            "calibration_latest": calibration_latest,
            "calibration_lag_trading_days": trading_day_lag(
                trading_dates, calibration_latest, expected
            ),
        },
        "expected_date_records": {
            "settlement": expected_settlement or None,
            "surface": expected_surface or None,
            "calibration": expected_calibration or None,
        },
        "stages": [asdict(stage) for stage in stages],
        "last_error": dict(last_error) if last_error is not None else None,
    }


def status_exit_code(status: Mapping[str, Any]) -> int:
    overall = status.get("overall_status")
    if overall == "current":
        return EXIT_CURRENT
    if overall == "failed" or overall == "calibration_failed":
        return EXIT_FAILED
    return EXIT_NON_CURRENT


def print_status(status: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(status, indent=2, sort_keys=True, default=str))
        return
    freshness = status.get("freshness", {})
    print(
        f"MO daily calibration: {status.get('overall_status')} | "
        f"expected={status.get('expected_trade_date')} "
        f"settlement={freshness.get('settlement_latest')} "
        f"surface={freshness.get('surface_latest')} "
        f"calibration={freshness.get('calibration_latest')}"
    )
    if status.get("last_error"):
        error = status["last_error"]
        print(f"error: {error.get('error_type')}: {error.get('message')}")


def run_pipeline(
    args: argparse.Namespace,
    *,
    stage_runner: StageRunner = default_stage_runner,
) -> tuple[int, dict[str, Any]]:
    paths = PipelinePaths(args.history_dir.resolve(), args.runtime_dir.resolve())
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    # QuantArk imports plotting modules through package surfaces even though
    # this command does not render charts.  Give Matplotlib/fontconfig stable,
    # writable caches so launchd and sandboxed/manual runs do not rebuild font
    # caches or emit home-directory permission warnings.
    mpl_cache = paths.runtime_dir / "matplotlib"
    xdg_cache = paths.runtime_dir / "cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    run_id = datetime.now(SHANGHAI).strftime("%Y%m%dT%H%M%S%z")
    stages: list[StageResult] = []
    running = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": "mo_daily_calibration",
        "overall_status": "running",
        "generated_at": iso_utc(),
        "run_id": run_id,
        "as_of_date": args.as_of.isoformat(),
        "pid": os.getpid(),
    }

    try:
        with acquire_lock(paths.lock):
            atomic_write_json(paths.status, running)
            if not args.skip_market_refresh:
                validate_python(args.stage01_python, "stage-01")
                stages.append(
                    stage_runner(
                        "refresh_market_cache",
                        [
                            str(args.stage01_python),
                            str(REFRESH_SCRIPT),
                            "--end",
                            args.as_of.isoformat(),
                            "--output-dir",
                            str(paths.history_dir),
                        ],
                    )
                )

            trading_dates = load_trading_dates(paths.spot_csv, as_of=args.as_of)
            expected = trading_dates[-1]

            if not args.skip_settlement_fetch:
                validate_python(args.stage01_python, "stage-01")
                settlement_payload = read_json(
                    paths.settlement_manifest, default={"records": []}
                )
                settlement_records = records_by_date(settlement_payload)
                last_settlement = latest_record_date(
                    settlement_records,
                    status="ok",
                    require_file=lambda tag: (
                        paths.raw_dir / f"{tag}_1.csv"
                    ).is_file(),
                )
                fetch_start = last_settlement or trading_dates[0]
                stages.append(
                    stage_runner(
                        "fetch_settlements",
                        [
                            str(args.stage01_python),
                            str(SETTLEMENT_SCRIPT),
                            "--spot-csv",
                            str(paths.spot_csv),
                            "--raw-dir",
                            str(paths.raw_dir),
                            "--manifest",
                            str(paths.settlement_manifest),
                            "--start",
                            fetch_start,
                            "--end",
                            expected,
                            "--delay",
                            str(args.fetch_delay),
                        ],
                    )
                )

            if not args.skip_surface_build:
                validate_python(args.stage_python, "QuantArk")
                surface_payload = read_json(
                    paths.surface_manifest, default={"records": []}
                )
                surface_records = records_by_date(surface_payload)
                last_surface_record = latest_record_date(surface_records)
                build_start = (
                    day_after(last_surface_record)
                    if last_surface_record is not None
                    else trading_dates[0]
                )
                settlement_payload = read_json(
                    paths.settlement_manifest, default={"records": []}
                )
                settlement_records = records_by_date(settlement_payload)
                pending_raw_dates = [
                    trade_date
                    for trade_date, record in sorted(settlement_records.items())
                    if (
                        build_start <= trade_date <= expected
                        and record.get("status") == "ok"
                        and (paths.raw_dir / f"{trade_date}_1.csv").is_file()
                    )
                ]
                if pending_raw_dates:
                    stages.append(
                        stage_runner(
                            "build_surfaces",
                            [
                                str(args.stage_python),
                                str(SURFACE_SCRIPT),
                                "--csv-dir",
                                str(paths.raw_dir),
                                "--output-dir",
                                str(paths.surface_dir),
                                "--manifest",
                                str(paths.surface_manifest),
                                "--spot-csv",
                                str(paths.spot_csv),
                                "--start",
                                pending_raw_dates[0],
                                "--end",
                                pending_raw_dates[-1],
                                "--workers",
                                str(args.surface_workers),
                            ],
                        )
                    )

            surface_payload = read_json(
                paths.surface_manifest, default={"records": []}
            )
            surface_records = records_by_date(surface_payload)
            manifest_payload, calibration_records = load_calibration_records(
                paths.calibration_manifest
            )
            config = calibration_config_payload(args)
            selected = select_calibration_dates(
                surface_records,
                calibration_records,
                config=config,
                backfill=bool(args.backfill_calibrations),
                max_dates=args.max_calibration_dates,
                baseline_date=(
                    manifest_payload.get("baseline_date")
                    or (min(calibration_records) if calibration_records else None)
                ),
                start_date=args.calibration_start,
                end_date=args.calibration_end,
            )
            if (
                selected
                and not calibration_records
                and not args.backfill_calibrations
                and manifest_payload.get("baseline_date") is None
            ):
                manifest_payload["baseline_date"] = selected[0]
            for trade_date in selected:
                print(f"{trade_date}: calibrating {', '.join(VARIANTS)}")
                record = calibrate_one_surface(
                    trade_date,
                    paths=paths,
                    args=args,
                    calibration_records=calibration_records,
                )
                calibration_records[trade_date] = record
                persist_calibration_manifest(
                    paths.calibration_manifest,
                    base_payload=manifest_payload,
                    records=calibration_records,
                    config=config,
                )
                print(
                    f"{trade_date}: {record['status']} "
                    f"[{record['elapsed_seconds']:.2f}s]"
                )

            status = build_freshness_status(
                paths=paths,
                as_of=args.as_of,
                stages=stages,
                run_id=run_id,
            )
            atomic_write_json(paths.status, status)
            return status_exit_code(status), status
    except LockBusy:
        raise
    except Exception as exc:
        error = {
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            status = build_freshness_status(
                paths=paths,
                as_of=args.as_of,
                stages=stages,
                run_id=run_id,
                last_error=error,
            )
        except Exception as status_exc:  # preserve the primary failure
            status = {
                **running,
                "overall_status": "failed",
                "last_error": error,
                "status_build_error": {
                    "error_type": type(status_exc).__name__,
                    "message": str(status_exc),
                },
                "stages": [asdict(stage) for stage in stages],
            }
        atomic_write_json(paths.status, status)
        return EXIT_FAILED, status


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--as-of",
        type=parse_as_of,
        default=parse_as_of(None),
        help="Shanghai business-date horizon, YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=DEFAULT_HISTORY_DIR,
        help=f"MO history root (default: {DEFAULT_HISTORY_DIR})",
    )
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=DEFAULT_RUNTIME_DIR,
        help=f"runtime status/cache root (default: {DEFAULT_RUNTIME_DIR})",
    )
    parser.add_argument("--json", action="store_true", help="print full JSON status")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="execute the incremental EOD pipeline")
    add_shared_args(run)
    run.add_argument(
        "--stage01-python", type=Path, default=DEFAULT_STAGE01_PYTHON
    )
    run.add_argument("--stage-python", type=Path, default=DEFAULT_STAGE_PYTHON)
    run.add_argument(
        "--surface-workers",
        type=int,
        default=1,
        help="surface-build workers; production-safe default is 1",
    )
    run.add_argument("--fetch-delay", type=float, default=0.5)
    run.add_argument("--heston-max-nfev", type=int, default=200)
    run.add_argument(
        "--temporal-smoothing",
        action="store_true",
        help=(
            "opt in to daily v0 + structural EWMA for SLV and "
            "temporally regularized pure Heston"
        ),
    )
    run.add_argument(
        "--structural-ewma-span",
        type=int,
        default=5,
        help="structural Heston EWMA span (default: 5 admitted dates)",
    )
    run.add_argument(
        "--heston-temporal-regularization",
        type=float,
        default=0.1,
        help=(
            "bound-normalized structural Heston temporal penalty "
            "(default: 0.1; active only with --temporal-smoothing)"
        ),
    )
    run.add_argument("--slv-eta", type=float, default=1.0)
    run.add_argument("--slv-n-steps", type=int, default=40)
    run.add_argument("--slv-n-x", type=int, default=161)
    run.add_argument("--slv-n-z", type=int, default=81)
    run.add_argument(
        "--max-calibration-dates",
        type=int,
        default=None,
        help="optional cap for controlled catch-up runs",
    )
    run.add_argument(
        "--backfill-calibrations",
        action="store_true",
        help="explicitly calibrate all uncached historical surfaces",
    )
    run.add_argument(
        "--calibration-start",
        type=parse_date_tag,
        default=None,
        help="inclusive backfill lower bound, YYYY-MM-DD or YYYYMMDD",
    )
    run.add_argument(
        "--calibration-end",
        type=parse_date_tag,
        default=None,
        help="inclusive backfill upper bound, YYYY-MM-DD or YYYYMMDD",
    )
    run.add_argument("--skip-market-refresh", action="store_true")
    run.add_argument("--skip-settlement-fetch", action="store_true")
    run.add_argument("--skip-surface-build", action="store_true")

    status = subparsers.add_parser("status", help="inspect current artifact freshness")
    add_shared_args(status)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command != "run":
        return
    if args.surface_workers < 1:
        raise SystemExit("--surface-workers must be >= 1")
    if args.fetch_delay < 0.0:
        raise SystemExit("--fetch-delay must be non-negative")
    if args.heston_max_nfev < 1:
        raise SystemExit("--heston-max-nfev must be >= 1")
    if args.structural_ewma_span < 1:
        raise SystemExit("--structural-ewma-span must be >= 1")
    if not math.isfinite(args.heston_temporal_regularization) or (
        args.heston_temporal_regularization < 0.0
    ):
        raise SystemExit(
            "--heston-temporal-regularization must be finite and non-negative"
        )
    if args.slv_eta < 0.0:
        raise SystemExit("--slv-eta must be non-negative")
    for name in ("slv_n_steps", "slv_n_x", "slv_n_z"):
        if int(getattr(args, name)) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1")
    if args.max_calibration_dates is not None and args.max_calibration_dates < 1:
        raise SystemExit("--max-calibration-dates must be >= 1")
    if (args.calibration_start or args.calibration_end) and not (
        args.backfill_calibrations
    ):
        raise SystemExit(
            "--calibration-start/--calibration-end require --backfill-calibrations"
        )
    if (
        args.calibration_start
        and args.calibration_end
        and args.calibration_start > args.calibration_end
    ):
        raise SystemExit("--calibration-start must be <= --calibration-end")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args)
    paths = PipelinePaths(args.history_dir.resolve(), args.runtime_dir.resolve())

    if args.command == "status":
        try:
            status = build_freshness_status(paths=paths, as_of=args.as_of)
        except Exception as exc:
            status = {
                "schema_version": SCHEMA_VERSION,
                "pipeline": "mo_daily_calibration",
                "overall_status": "failed",
                "generated_at": iso_utc(),
                "as_of_date": args.as_of.isoformat(),
                "last_error": {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            }
            print_status(status, as_json=args.json)
            return EXIT_FAILED
        print_status(status, as_json=args.json)
        return status_exit_code(status)

    try:
        code, status = run_pipeline(args)
    except LockBusy as exc:
        status = {
            "schema_version": SCHEMA_VERSION,
            "pipeline": "mo_daily_calibration",
            "overall_status": "locked",
            "generated_at": iso_utc(),
            "message": str(exc),
        }
        print_status(status, as_json=args.json)
        return EXIT_LOCKED
    print_status(status, as_json=args.json)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
