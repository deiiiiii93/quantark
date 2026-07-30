"""Exclude IV surfaces too thin in maturity for the vol-model study (Gate G1 top-up).

The Phase-1 surface builder admits a surface with ``min_expiries = 2``.  That is
enough for a flat-ATM or term-structure read, but NOT for Dupire local
volatility, which needs at least three maturity pillars to form dC/dT --
``build_dupire_local_vol`` fails closed below that.  Two admitted dates in the
2023-05..2026-07 window carry only two expiries:

    2024-09-30  16 strikes, 2 maturities
    2025-04-08  23 strikes, 2 maturities

Left admitted, they break 20 of 27 ``localvol`` runs (and would break
``heston_slv``, which consumes an LV surface for its leverage calibration)
partway through the replay.

This script re-marks those records ``status="excluded"`` with a logged reason,
so consumers fall back to the manifest's documented carry-forward policy
("consumers carry forward previous admitted surface").  No surface is
invented and no artifact is deleted -- the file stays on disk and the record
keeps its sha, it simply stops being admitted.

Idempotent: re-running changes nothing once the records are excluded.

Run:  .venv/bin/python example/mo_volmodels/exclude_thin_surfaces.py [--check]
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_DIR = PROJECT_ROOT / "example/mo_volmodels/data/history"

# Dupire's dC/dT needs an interior maturity node.
MIN_EXPIRIES_FOR_DUPIRE = 3
EXCLUSION_REASON = "insufficient_expiries_for_dupire"


def thin_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Admitted records whose expiry count cannot support Dupire."""
    out = []
    for rec in records:
        if rec.get("status") != "ok":
            continue
        n = rec.get("n_expiries")
        if n is not None and int(n) < MIN_EXPIRIES_FOR_DUPIRE:
            out.append(rec)
    return out


def apply_exclusions(manifest: Dict[str, Any]) -> List[str]:
    """Mark thin admitted records excluded. Returns the dates changed."""
    changed = []
    for rec in thin_records(manifest.get("records", [])):
        rec["status"] = "excluded"
        rec["reason"] = EXCLUSION_REASON
        rec["detail"] = (
            f"{rec.get('n_expiries')} expiries (need >= {MIN_EXPIRIES_FOR_DUPIRE} "
            "for Dupire dC/dT); artifact retained on disk but not admitted, "
            "consumers carry forward the previous admitted surface"
        )
        changed.append(str(rec.get("date")))
    if changed:
        manifest.setdefault("study_admission", {})[
            "vol_model_backtest"
        ] = {
            "min_expiries": MIN_EXPIRIES_FOR_DUPIRE,
            "rationale": (
                "Dupire local volatility (and Heston-SLV's leverage "
                "calibration, which consumes an LV surface) needs at least "
                "three maturity pillars; the builder's own min_expiries=2 "
                "admits surfaces those models cannot price."
            ),
            "excluded_dates": changed,
        }
    return changed


def _atomic_write_json(path: Path, payload: Any) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--history-dir", default=str(DEFAULT_HISTORY_DIR))
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would be excluded without writing",
    )
    args = parser.parse_args(argv)

    path = Path(args.history_dir) / "surface_manifest.json"
    manifest = json.loads(path.read_text())
    before_ok = sum(1 for r in manifest["records"] if r.get("status") == "ok")

    thin = thin_records(manifest["records"])
    if args.check:
        print(f"admitted before: {before_ok}")
        print(f"too thin for Dupire (< {MIN_EXPIRIES_FOR_DUPIRE} expiries): {len(thin)}")
        for rec in thin:
            print(f"   {rec['date']}  n_expiries={rec.get('n_expiries')}")
        return 1 if thin else 0

    changed = apply_exclusions(manifest)
    if not changed:
        print(f"no change: all {before_ok} admitted surfaces already have "
              f">= {MIN_EXPIRIES_FOR_DUPIRE} expiries")
        return 0
    _atomic_write_json(path, manifest)
    after_ok = sum(1 for r in manifest["records"] if r.get("status") == "ok")
    print(f"excluded {len(changed)} thin surfaces: {', '.join(changed)}")
    print(f"admitted: {before_ok} -> {after_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
