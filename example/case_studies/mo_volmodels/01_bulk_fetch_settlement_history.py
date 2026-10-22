"""Bulk-download official CFFEX daily-statistics CSVs for every trading day.

This is the raw acquisition stage of the MO settlement history build.  It
walks the trading calendar implied by the refreshed market cache
(``data/history/csi1000_spot.csv`` — the repo convention is "the trading
calendar is the market-data index") and saves the raw official CSV bytes for
each trading date to ``data/history/settlement_csv/{YYYYMMDD}_1.csv`` so the
single-date freezer ``01_fetch_mo_settlement_history.py`` can later run fully
offline via its ``--input-dir`` replay path.

Fetching reuses ``official_url`` / ``_download`` from
``01_fetch_mo_settlement_history.py`` (loaded via importlib because the
filename starts with digits).  The module is import-safe: no network call and
no third-party import happens at module level, so tests can load it under any
interpreter.  Run it under the stage-01 interpreter::

    /opt/anaconda3/bin/python example/mo_volmodels/01_bulk_fetch_settlement_history.py \
        --max-dates 5

Resume / fail-closed policy:

- A date whose ``{YYYYMMDD}_1.csv`` already exists, is non-empty, AND has a
  manifest record is skipped (never re-downloaded), so interrupted runs
  resume cheaply.  A CSV present WITHOUT a manifest record has no provenance
  (it may be a torn write from a crashed run): it is treated as NOT frozen
  and re-fetched.  This reconciliation runs on every startup.
- If the manifest file itself is missing but frozen CSVs exist, the manifest
  is rebuilt from those CSVs (status ``ok``, sha256/bytes recomputed,
  ``attempts`` null, reason ``"rebuilt from csv"``).  If the manifest exists
  but is not valid JSON, the run aborts with a clear recovery message.
- Every attempted date is recorded in ``settlement_manifest.json`` with
  ``{date, status, sha256, bytes, attempts, reason}``.  ``missing`` means the
  server answered HTTP 404 (the evidence is kept in ``reason``); ``error``
  means the fetch still failed after all retry attempts.
- Payloads are sniffed before freezing: anything that does not GB18030-decode
  to a first line containing ``合约代码`` is treated as an error (retried,
  then recorded) and never written.  Placeholder or empty CSVs are never
  written either: a date that fails simply has no CSV and a manifest entry
  explaining why.
- All writes (raw CSV, manifest JSON) are atomic: temp file, fsync,
  ``os.replace``.  Within one date the CSV is replaced BEFORE the manifest is
  updated, so the manifest never vouches for a file that is not in place.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import tempfile
import time
import urllib.error
from pathlib import Path
from typing import Callable


HERE = Path(__file__).resolve().parent
DEFAULT_SPOT_CSV = HERE / "data" / "history" / "csi1000_spot.csv"
DEFAULT_RAW_DIR = HERE / "data" / "history" / "settlement_csv"
DEFAULT_MANIFEST = HERE / "data" / "history" / "settlement_manifest.json"
DEFAULT_DELAY_SECONDS = 0.5
MAX_ATTEMPTS = 3
MANIFEST_SCHEMA_VERSION = 1

_FETCHER_PATH = HERE / "01_fetch_mo_settlement_history.py"
_spec = importlib.util.spec_from_file_location("mo_settlement_fetcher", _FETCHER_PATH)
assert _spec is not None and _spec.loader is not None
_fetcher = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fetcher)

official_url = _fetcher.official_url
_parse_trade_date = _fetcher._parse_trade_date

DownloadFn = Callable[[str], "tuple[bytes, str]"]
SleepFn = Callable[[float], None]


class ManifestError(Exception):
    """Raised when the settlement manifest exists but cannot be parsed."""


def download_one(trade_date: str) -> tuple[bytes, str]:
    """Fetch one official CSV via the stage-01 fetcher (30s per-request timeout)."""
    return _fetcher._download(trade_date)


def is_missing_error(exc: BaseException) -> bool:
    """Return True when the server answered that no file exists for the date."""
    return isinstance(exc, urllib.error.HTTPError) and exc.code == 404


def load_trading_dates(spot_csv: Path) -> list[str]:
    """Read the trading calendar (YYYYMMDD tags) from the spot cache index."""
    if not spot_csv.is_file():
        raise FileNotFoundError(
            f"trading calendar {spot_csv} is missing; run "
            "01_refresh_market_cache.py first"
        )
    dates: list[str] = []
    with spot_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tag = str(row["date"]).strip().replace("-", "")
            _parse_trade_date(tag)
            dates.append(tag)
    if not dates:
        raise ValueError(f"trading calendar {spot_csv} contains no dates")
    return sorted(set(dates))


def csv_path(raw_dir: Path, trade_date: str) -> Path:
    return raw_dir / f"{trade_date}_1.csv"


def has_frozen_csv(raw_dir: Path, trade_date: str) -> bool:
    path = csv_path(raw_dir, trade_date)
    return path.is_file() and path.stat().st_size > 0


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically: temp file, fsync, os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_manifest(manifest_path: Path) -> dict[str, dict]:
    """Load existing manifest records keyed by date; tolerate a missing file.

    A manifest that exists but is not valid JSON raises :class:`ManifestError`
    with recovery instructions rather than being silently discarded (which
    would re-download hundreds of dates or, worse, mix provenance).
    """
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(
            f"settlement manifest {manifest_path} is corrupt ({exc}); "
            "delete the file and rerun to rebuild records from the frozen CSVs"
        ) from exc
    records = payload.get("records", [])
    return {str(record["date"]): record for record in records}


def rebuild_manifest_from_csvs(raw_dir: Path) -> dict[str, dict]:
    """Reconstruct ok records from frozen CSVs when the manifest was lost.

    Rebuilt records carry recomputed sha256/bytes, ``attempts`` null, and
    reason ``"rebuilt from csv"`` so they are distinguishable from records
    written by an actual download.  Empty files are not provenance and are
    ignored (their dates stay pending and will be re-fetched).
    """
    records: dict[str, dict] = {}
    if not raw_dir.is_dir():
        return records
    for path in sorted(raw_dir.glob("*_1.csv")):
        tag = path.name[: -len("_1.csv")]
        try:
            _parse_trade_date(tag)
        except ValueError:
            continue
        payload = path.read_bytes()
        if not payload:
            continue
        records[tag] = {
            "date": tag,
            "status": "ok",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "attempts": None,
            "reason": "rebuilt from csv",
        }
    return records


def save_manifest(manifest_path: Path, records: dict[str, dict]) -> None:
    """Rewrite the manifest deterministically (one record per attempted date)."""
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "records": [records[tag] for tag in sorted(records)],
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _atomic_write_bytes(manifest_path, serialized.encode("utf-8"))


def _error_record(trade_date: str, attempts: int, reason: str) -> dict:
    return {
        "date": trade_date,
        "status": "error",
        "sha256": None,
        "bytes": None,
        "attempts": attempts,
        "reason": reason,
    }


def _payload_problem(payload: bytes) -> str | None:
    """Return None iff ``payload`` looks like a genuine CFFEX statistics CSV.

    The exchange can answer 200 with an HTML error page or a truncated body;
    such bytes must never be frozen.  The sniff mirrors the strict checks the
    stage-01 parser applies (GB18030 text whose header names 合约代码).
    """
    if not payload:
        return "empty response body"
    try:
        text = payload.decode("gb18030")
    except UnicodeDecodeError:
        return "payload is not valid GB18030 text"
    stripped = text.lstrip("\ufeff").lstrip()
    first_line = stripped.splitlines()[0] if stripped else ""
    if "合约代码" not in first_line:
        return "payload first line is not a CFFEX CSV header (missing 合约代码)"
    return None


def fetch_with_retries(
    trade_date: str,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    delay: float = DEFAULT_DELAY_SECONDS,
    sleep: SleepFn = time.sleep,
    download: DownloadFn | None = None,
) -> tuple[dict, bytes | None]:
    """Fetch one date with exponential backoff; never fabricate a payload.

    Returns ``(manifest_record, payload)`` where ``payload`` is ``None`` for
    any non-ok status.  HTTP 404 short-circuits as ``missing`` (retrying a
    date the server says it does not have is pointless); every other failure
    — including a payload that fails the CFFEX sniff — is retried up to
    ``max_attempts`` with ``delay * 2**n`` backoff.  ``download`` defaults to
    :func:`download_one`, resolved at call time so tests can monkeypatch it.
    """
    if download is None:
        download = download_one
    attempts = 0
    while True:
        attempts += 1
        try:
            payload, _url = download(trade_date)
        except Exception as exc:  # noqa: BLE001 - recorded verbatim in the manifest
            if is_missing_error(exc):
                evidence = (
                    f"HTTP {exc.code} {exc.reason}: "
                    f"{getattr(exc, 'url', None) or official_url(trade_date)}"
                )
                return (
                    {
                        "date": trade_date,
                        "status": "missing",
                        "sha256": None,
                        "bytes": None,
                        "attempts": attempts,
                        "reason": evidence,
                    },
                    None,
                )
            if attempts >= max_attempts:
                return (
                    _error_record(
                        trade_date, attempts, f"{type(exc).__name__}: {exc}"
                    ),
                    None,
                )
            sleep(delay * (2 ** (attempts - 1)))
            continue
        problem = _payload_problem(payload)
        if problem is None:
            return (
                {
                    "date": trade_date,
                    "status": "ok",
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "attempts": attempts,
                    "reason": None,
                },
                payload,
            )
        if attempts >= max_attempts:
            return _error_record(trade_date, attempts, problem), None
        sleep(delay * (2 ** (attempts - 1)))


def process_dates(
    dates: list[str],
    *,
    raw_dir: Path,
    manifest_path: Path,
    delay: float = DEFAULT_DELAY_SECONDS,
    max_attempts: int = MAX_ATTEMPTS,
    max_dates: int | None = None,
    sleep: SleepFn = time.sleep,
    download: DownloadFn | None = None,
    verbose: bool = True,
) -> dict[str, dict]:
    """Download every pending date and refresh the manifest.

    A date is frozen only when its CSV exists, is non-empty, AND has a
    manifest record; anything else is (re-)fetched.  When the manifest file
    is missing, records are first rebuilt from the frozen CSVs.  The CSV for
    a date is replaced BEFORE the manifest is saved, and both writes are
    atomic, so a crash can lose at most the current date's progress and can
    never leave a file the manifest vouches for but that is not in place.
    """
    if download is None:
        download = download_one
    raw_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        records = load_manifest(manifest_path)
    else:
        records = rebuild_manifest_from_csvs(raw_dir)
        if records:
            save_manifest(manifest_path, records)
            if verbose:
                print(f"rebuilt {len(records)} manifest records from frozen CSVs")
    frozen = {
        tag for tag in dates if tag in records and has_frozen_csv(raw_dir, tag)
    }
    pending = [tag for tag in dates if tag not in frozen]
    if max_dates is not None:
        pending = pending[:max_dates]
    if verbose:
        print(
            f"{len(dates)} trading dates requested: {len(frozen)} already frozen, "
            f"{len(pending)} to download"
        )
    for index, trade_date in enumerate(pending):
        record, payload = fetch_with_retries(
            trade_date,
            max_attempts=max_attempts,
            delay=delay,
            sleep=sleep,
            download=download,
        )
        # CSV first, manifest second: the manifest must never vouch for a
        # file that is not safely in place.
        if payload is not None:
            _atomic_write_bytes(csv_path(raw_dir, trade_date), payload)
        records[trade_date] = record
        save_manifest(manifest_path, records)
        if verbose:
            detail = (
                f"{record['bytes']} bytes sha256={record['sha256'][:12]}…"
                if record["status"] == "ok"
                else record["reason"]
            )
            print(
                f"{trade_date}: {record['status']} "
                f"(attempts={record['attempts']}) {detail}"
            )
        if index < len(pending) - 1:
            sleep(delay)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spot-csv",
        type=Path,
        default=DEFAULT_SPOT_CSV,
        help=f"trading-calendar source (default: {DEFAULT_SPOT_CSV})",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"raw CSV output directory (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"manifest path (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help=f"seconds between requests (default: {DEFAULT_DELAY_SECONDS})",
    )
    parser.add_argument("--start", help="first trade date, YYYYMMDD (inclusive)")
    parser.add_argument("--end", help="last trade date, YYYYMMDD (inclusive)")
    parser.add_argument(
        "--max-dates",
        type=int,
        help="process only the first N pending dates (smoke tests)",
    )
    args = parser.parse_args()

    if args.delay < 0:
        raise SystemExit("--delay must be non-negative")
    if args.max_dates is not None and args.max_dates <= 0:
        raise SystemExit("--max-dates must be positive")

    dates = load_trading_dates(args.spot_csv)
    if args.start is not None:
        _parse_trade_date(args.start)
        dates = [tag for tag in dates if tag >= args.start]
    if args.end is not None:
        _parse_trade_date(args.end)
        dates = [tag for tag in dates if tag <= args.end]
    if not dates:
        raise SystemExit("no trading dates in the requested window")

    try:
        records = process_dates(
            dates,
            raw_dir=args.raw_dir,
            manifest_path=args.manifest,
            delay=args.delay,
            max_dates=args.max_dates,
        )
    except ManifestError as exc:
        raise SystemExit(f"[error] {exc}") from exc

    window = [tag for tag in dates if tag in records]
    counts = {"ok": 0, "missing": 0, "error": 0}
    for tag in window:
        status = records[tag]["status"]
        counts[status] = counts.get(status, 0) + 1
    print(
        f"summary: {counts['ok']} ok, {counts['missing']} missing, "
        f"{counts['error']} error across {len(window)} window dates with manifest "
        f"records; manifest -> {args.manifest}"
    )
    if counts["error"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
