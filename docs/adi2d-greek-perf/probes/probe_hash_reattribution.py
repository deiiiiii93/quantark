"""Does a validation-only edit change any certified number? Measure, don't argue.

The 14-cell gate-driven fleet ran from source ``350d323`` and banked every cell
under ``implementation=0a93c4940b1edb98``. Publishing then failed, because
``validate_payload`` asserted ``batches_used == expected_batches`` -- an equality
gate-driven stopping breaks by design. Fixing that validator edited
``16_adi_greek_certification.py``, which is inside ``IMPLEMENTATION_INPUTS``, so
the implementation digest moved to ``c471537734a253da`` and ``--resume`` refused
the banked checkpoints. The guard is behaving correctly: it enforces
ATTRIBUTION ("these numbers came from this exact source"), which is recorded,
not testable.

REPRODUCIBILITY is testable, and it is the question a re-stamp decision turns on:
if the new source recomputes a banked cell bitwise, then the edit provably moved
no number and the digest change is pure attribution. This probe re-runs one cheap
cell under the new source and compares its checkpoint evidence field by field
against the banked one.

Fields that are wall-clock or environment observations cannot be expected to
match and are reported separately rather than silently excluded -- a probe that
hides what it ignored is how a "bitwise" claim gets made about a filtered subset.

Usage:
    PYTHONPATH=$PWD python docs/adi2d-greek-perf/probes/probe_hash_reattribution.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
BANKED = ROOT / "output" / "p14_sequential" / "checkpoints"
REPLAY = ROOT / "output" / "hash_reattribution" / "checkpoints"
OUTPUT_DIR = ROOT / "output" / "hash_reattribution"

# Observations of the run, not results of it. Timings and backend probes are
# allowed to differ; every other leaf must be identical.
WALL_CLOCK_KEYS = frozenset(
    {
        "seconds",
        "wall_seconds",
        "elapsed_seconds",
        "duration_seconds",
        "runtime_seconds",
        "priced_at",
        "timestamp",
        "generated_at",
    }
)


def _is_wall_clock(path: str) -> bool:
    return any(part in WALL_CLOCK_KEYS for part in path.split("."))


def diff_leaves(
    left: Any, right: Any, path: str = ""
) -> tuple[list[tuple[str, Any, Any]], list[str], int]:
    """Recursive leaf comparison.

    Returns (numeric_or_structural_differences, wall_clock_differences, leaves).
    Floats compare by exact bit pattern: the claim under test is that no number
    moved at all, so an approximate comparison would not test it.
    """
    differences: list[tuple[str, Any, Any]] = []
    wall_clock: list[str] = []
    leaves = 0

    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                differences.append(
                    (child, "<missing>" if key not in left else "present",
                     "<missing>" if key not in right else "present")
                )
                continue
            sub, sub_wall, sub_leaves = diff_leaves(left[key], right[key], child)
            differences.extend(sub)
            wall_clock.extend(sub_wall)
            leaves += sub_leaves
        return differences, wall_clock, leaves

    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            differences.append((f"{path}[len]", len(left), len(right)))
            return differences, wall_clock, leaves
        for index, (one, two) in enumerate(zip(left, right)):
            sub, sub_wall, sub_leaves = diff_leaves(one, two, f"{path}[{index}]")
            differences.extend(sub)
            wall_clock.extend(sub_wall)
            leaves += sub_leaves
        return differences, wall_clock, leaves

    leaves += 1
    identical = _leaf_identical(left, right)
    if not identical:
        if _is_wall_clock(path):
            wall_clock.append(path)
        else:
            differences.append((path, left, right))
    return differences, wall_clock, leaves


def _leaf_identical(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        try:
            one, two = float(left), float(right)
        except (TypeError, ValueError):
            return False
        if math.isnan(one) and math.isnan(two):
            return True
        # Exact bit equality, so a 1e-16 drift is a failure and not a rounding.
        return one.hex() == two.hex()
    return left == right


def load_evidence(path: Path) -> dict:
    record = json.loads(path.read_text())
    if record.get("kind") != "cell":
        raise ValueError(f"{path}: expected a cell checkpoint, got {record.get('kind')}")
    return record


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", default="heston__near_ko")
    args = parser.parse_args(argv)

    banked_path = BANKED / f"{args.cell}.json"
    replay_path = REPLAY / f"{args.cell}.json"
    for path in (banked_path, replay_path):
        if not path.exists():
            print(f"missing checkpoint: {path}")
            return 2

    banked = load_evidence(banked_path)
    replay = load_evidence(replay_path)

    print(f"cell {args.cell}")
    print(f"  banked  config {banked['run_configuration_sha256'][:16]}  {banked_path}")
    print(f"  replay  config {replay['run_configuration_sha256'][:16]}  {replay_path}")
    print(
        "  the configuration digests DIFFER by construction: the replay declares a "
        "one-cell subset.\n  what is under test is the evidence, not the digest.\n"
    )

    differences, wall_clock, leaves = diff_leaves(banked["evidence"], replay["evidence"])

    print(f"  leaves compared          {leaves}")
    print(f"  wall-clock leaves differing {len(wall_clock)}")
    print(f"  substantive differences  {len(differences)}")
    if wall_clock:
        print("\n  wall-clock/environment leaves (expected to differ):")
        for path in wall_clock[:12]:
            print(f"    {path}")
        if len(wall_clock) > 12:
            print(f"    ... and {len(wall_clock) - 12} more")
    if differences:
        print("\n  SUBSTANTIVE DIFFERENCES:")
        for path, left, right in differences[:40]:
            print(f"    {path}\n      banked={left!r}\n      replay={right!r}")
        if len(differences) > 40:
            print(f"    ... and {len(differences) - 40} more")

    reproduced = not differences
    verdict = (
        "the validation-only edit recomputes the banked cell BITWISE -- the digest "
        "change is pure attribution"
        if reproduced
        else "THE REPLAY DIFFERS -- the edit is not validation-only, or the run is "
        "not reproducible"
    )
    print(f"\nVERDICT: {verdict}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / f"reattribution_{args.cell}.json"
    destination.write_text(
        json.dumps(
            {
                "cell": args.cell,
                "banked_configuration_sha256": banked["run_configuration_sha256"],
                "replay_configuration_sha256": replay["run_configuration_sha256"],
                "leaves_compared": leaves,
                "wall_clock_differences": sorted(wall_clock),
                "substantive_differences": [
                    {"path": path, "banked": repr(left), "replay": repr(right)}
                    for path, left, right in differences
                ],
                "bitwise_reproduced": bool(reproduced),
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {destination}")
    return 0 if reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())
