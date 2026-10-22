"""Export settlement CSVs as canonical volcalibration snapshots.

``quantark.volcalibration`` reads canonical ``QuoteSnapshot`` envelopes from
``<root>/snapshots/{YYYYMMDD}.json`` and never parses a vendor format -- that
boundary is what keeps the library network-free and deterministic under test.
This script is the one-way bridge for the existing CFFEX settlement CSV
history, and it lives in ``example/`` because the CSV parser does.

Each exported snapshot carries the source CSV's own ``source_sha256`` verbatim,
so an exported date agrees with the sha already recorded in its migrated
manifest record and a resumed run rebuilds nothing.

Idempotent: a date whose snapshot already exists is skipped unless ``--force``.

Run:  .venv/bin/python example/mo_volmodels/export_snapshots.py
      .venv/bin/python example/mo_volmodels/export_snapshots.py --start 20260701
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quantark.volcalibration.snapshot import QuoteSnapshot  # noqa: E402
from quantark.volcalibration.store import atomic_write_json  # noqa: E402

DEFAULT_HISTORY_DIR = HERE / "data" / "history"


def _load_stage01():
    """Import the numbered fetch module for its CFFEX CSV parser."""
    path = HERE / "01_fetch_mo_settlement_history.py"
    spec = importlib.util.spec_from_file_location("mo_stage01", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_spot_map(spot_csv: Path) -> dict[str, float]:
    """CSI 1000 spot cache: ISO date -> close."""
    spots: dict[str, float] = {}
    with spot_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                value = float(row["spot"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value) and value > 0.0:
                spots[str(row["date"])] = value
    if not spots:
        raise SystemExit(f"spot cache {spot_csv} contains no usable rows")
    return spots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--start", default=None, help="first trade date, YYYYMMDD")
    parser.add_argument("--end", default=None, help="last trade date, YYYYMMDD")
    parser.add_argument("--symbol", default="000852.SH")
    parser.add_argument(
        "--force", action="store_true", help="rewrite snapshots that already exist"
    )
    args = parser.parse_args()

    stage01 = _load_stage01()
    raw_dir = args.history_dir / "settlement_csv"
    out_dir = args.history_dir / "snapshots"
    spots = load_spot_map(args.history_dir / "csi1000_spot.csv")

    written = skipped = failed = 0
    for path in sorted(raw_dir.glob("*_1.csv")):
        tag = path.name[: -len("_1.csv")]
        try:
            datetime.strptime(tag, "%Y%m%d")
        except ValueError:
            continue
        if args.start and tag < args.start:
            continue
        if args.end and tag > args.end:
            continue

        target = out_dir / f"{tag}.json"
        if target.is_file() and not args.force:
            skipped += 1
            continue

        iso = f"{tag[:4]}-{tag[4:6]}-{tag[6:]}"
        spot = spots.get(iso)
        if spot is None:
            print(f"{tag}: no spot in the cache -- skipped", file=sys.stderr)
            failed += 1
            continue

        payload = path.read_bytes()
        try:
            legacy = stage01.parse_cffex_csv(payload, tag)
        except ValueError as exc:
            print(f"{tag}: parse failed ({exc}) -- skipped", file=sys.stderr)
            failed += 1
            continue

        # The stored artifacts key off the sha of the CSV bytes; recompute it
        # here only when the parser did not already record one, so an exported
        # snapshot never disagrees with a manifest record.
        legacy.setdefault("source_sha256", hashlib.sha256(payload).hexdigest())
        snapshot = QuoteSnapshot.from_legacy_settlement(
            legacy, trade_date=iso, spot=spot, symbol=args.symbol
        )
        atomic_write_json(target, snapshot.to_payload())
        written += 1

    print(
        f"written={written} skipped={skipped} failed={failed} -> {out_dir}",
        file=sys.stderr,
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
