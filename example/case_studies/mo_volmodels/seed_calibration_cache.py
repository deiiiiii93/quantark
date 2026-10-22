"""Seed a fleet calibration cache from the daily pipeline's cache.

The cache key is ``sha256(surface_sha | variant | config_fingerprint)`` and the
filename is ``{variant}-{key}.json``, so an entry written under one config is
reusable by any run that computes the same fingerprint — verified 240/240 per
variant against stage 12's full-quality config.  Copying is therefore sound and
needs no re-validation here; the calibrator re-checks schema version and
surface sha on read and treats a mismatch as a miss.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = PROJECT_ROOT / "output/mo_daily_calibration/calibration_cache"


def seed(src: Path, dst: Path, *, dry_run: bool = False) -> Dict[str, Any]:
    """Copy every cache entry from ``src`` into ``dst`` without overwriting."""
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        raise FileNotFoundError(f"source cache directory not found: {src}")
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    by_variant: Counter = Counter()
    fingerprints = set()
    n_source = n_copied = n_skipped = 0
    for path in sorted(src.glob("*.json")):
        n_source += 1
        payload = json.loads(path.read_text(encoding="utf-8"))
        fingerprints.add(payload.get("config_fingerprint"))
        target = dst / path.name
        if target.exists():
            n_skipped += 1
            continue
        n_copied += 1
        by_variant[str(payload.get("variant"))] += 1
        if not dry_run:
            shutil.copy2(path, target)
    return {
        "n_source": n_source,
        "n_copied": n_copied,
        "n_skipped_existing": n_skipped,
        "by_variant": dict(by_variant),
        "fingerprints": sorted(f for f in fingerprints if f is not None),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", default=str(DEFAULT_SRC))
    parser.add_argument("--dst", required=True, help="<out-dir>/calibration_cache")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    summary = seed(Path(args.src), Path(args.dst), dry_run=args.dry_run)
    print(
        f"[seed] source {summary['n_source']}, copied {summary['n_copied']}, "
        f"skipped {summary['n_skipped_existing']} existing"
    )
    print(f"[seed] by variant: {summary['by_variant']}")
    print(f"[seed] distinct fingerprints: {len(summary['fingerprints'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
