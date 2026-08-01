"""Gate G1: verify the pinned surface cohort against the artifacts on disk.

This is a verifier, not a builder.  It answers one question per admitted date:
"is there an artifact the fleet can actually price against?"

Two things it deliberately does NOT do:

* It does not read admission out of an artifact.  An artifact's ``admission``
  block records the *criteria* the builder used (``min_expiries: 2``,
  ``sabr_beta``, ...) and carries no per-surface verdict.  The verdict is in
  ``surface_manifest.json``, which ``cohort.admitted_dates`` reads.
* It does not walk ``iv_surface/``.  That directory holds 768 files while 766
  are admitted: the two thin surfaces (2024-09-30, 2025-04-08) were excluded in
  the manifest, not deleted.  Globbing would fail them on the 3-expiry rule and
  halt the study on a false alarm.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IV_DIR = PROJECT_ROOT / "example/mo_volmodels/data/history/iv_surface"

# Dupire local vol needs at least three expiries to form dw/dT; the Phase-1
# builder admits two, so G1 re-checks rather than trusting the manifest alone.
MIN_EXPIRIES = 3


def _load_cohort():
    """Import the sibling cohort module (the stages are not a package)."""
    path = Path(__file__).resolve().parent / "cohort.py"
    spec = importlib.util.spec_from_file_location("mo_cohort", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mo_cohort"] = module
    spec.loader.exec_module(module)
    return module


cohort = _load_cohort()


def artifact_path(day: date, iv_dir: Path) -> Path:
    return Path(iv_dir) / f"mo_iv_surface_{day.strftime('%Y%m%d')}.json"


def verify_surface(day: date, iv_dir: Path) -> Tuple[bool, str]:
    """Return (ok, reason) for one admitted date.  Empty reason iff ok."""
    path = artifact_path(day, iv_dir)
    if not path.is_file():
        return False, f"no artifact at {path.name}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"artifact unreadable: {exc}"
    n_expiries = len(payload.get("maturities") or [])
    if n_expiries < MIN_EXPIRIES:
        return False, (
            f"{n_expiries} expiries < {MIN_EXPIRIES} required by Dupire"
        )
    return True, ""


def scan_cohort(
    iv_dir: Optional[Path] = None,
    history_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Verify every admitted date at or before the pin; JSON-safe summary."""
    iv_dir = Path(iv_dir or DEFAULT_IV_DIR)
    admitted = cohort.admitted_dates(history_dir)

    failures = []
    min_expiries_seen = None
    for day in admitted:
        ok, reason = verify_surface(day, iv_dir)
        if not ok:
            failures.append({"date": day.isoformat(), "reason": reason})
            continue
        payload = json.loads(artifact_path(day, iv_dir).read_text(encoding="utf-8"))
        n_exp = len(payload.get("maturities") or [])
        min_expiries_seen = (
            n_exp if min_expiries_seen is None else min(min_expiries_seen, n_exp)
        )
    return {
        "gate": "G1",
        "asof": cohort.COHORT_ASOF.isoformat(),
        "iv_dir": str(iv_dir),
        "n_admitted": len(admitted),
        "n_verified": len(admitted) - len(failures),
        "failures": failures,
        "min_expiries_seen": min_expiries_seen,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iv-dir", default=str(DEFAULT_IV_DIR))
    parser.add_argument("--history-dir", default=None)
    parser.add_argument("--out", default="output/gate_g1_admission.json")
    args = parser.parse_args(argv)

    summary = scan_cohort(
        Path(args.iv_dir),
        Path(args.history_dir) if args.history_dir else None,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))

    print(f"[G1] asof {summary['asof']}, "
          f"admitted {summary['n_admitted']}, "
          f"verified {summary['n_verified']}, "
          f"min expiries {summary['min_expiries_seen']}")
    for f in summary["failures"][:20]:
        print(f"  FAIL {f['date']}: {f['reason']}")
    if summary["failures"]:
        print(f"[G1] FAILED — {len(summary['failures'])} surface(s) unusable")
        return 1
    print("[G1] PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
