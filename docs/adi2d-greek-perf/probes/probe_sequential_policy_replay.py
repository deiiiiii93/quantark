"""Replay archived batch streams through the SHIPPED sequential policy.

Two jobs.

1. **Validate the module against real data.** ``quantark.validation`` now
   carries the anytime-valid rule; running the archived pilot streams through it
   checks it reproduces the P1 probe's qualitative picture (smooth cells decide
   at the floor, ``near_ki`` gamma is the slow one) with production semantics.

2. **Correct the headline.** P1 reported a 13.8x fleet saving by letting cells
   stop as early as 16 batches. But a downstream aggregate gate reads a *common
   scramble prefix* across cells, and ``cohort_contribution`` requires exactly
   ``AMENDMENT_AGGREGATE_BATCHES`` batches from every row -- it raises otherwise.
   A cell that stopped at 16 therefore cannot contribute to the aggregate at
   all, so that saving is not reachable while the aggregate is in play.

   This sweeps the floor to price the constraint instead of assuming it away:
     floor 0/16 -- per-cell gates only (P1's regime)
     floor 128  -- the cohort path's common-scramble requirement

Usage:
    PYTHONPATH=$PWD python docs/adi2d-greek-perf/probes/probe_sequential_policy_replay.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from quantark.validation import (
    SequentialAdmissionPolicy,
    SequentialAdmissionStatus,
    scan_admission_stream,
)

ROOT = Path(__file__).resolve().parents[3]
PILOT = ROOT / "output" / "allocation_pilot" / "checkpoints"
DEEP = ROOT / "output" / "near_ki_deep_pilot" / "checkpoints"
OUTPUT_DIR = ROOT / "output" / "sequential_policy_replay"

FAMILY_ALPHA = 0.05
GREEK_COLUMNS = {"delta": 3, "gamma": 4}

# The frozen allocation each cell is measured against.
FROZEN_BATCHES = {
    ("heston", "ordinary_full"): 1024,
    ("heston", "ordinary_decayed"): 1024,
    ("heston", "near_ko"): 1024,
    ("heston", "near_ki"): 2048,
    ("heston", "low_feller"): 1024,
    ("heston", "sigma_collapse"): 1024,
    ("heston", "near_expiry"): 1024,
    ("heston_slv", "ordinary_full"): 128,
    ("heston_slv", "ordinary_decayed"): 128,
    ("heston_slv", "near_ko"): 128,
    ("heston_slv", "near_ki"): 256,
    ("heston_slv", "low_feller"): 512,
    ("heston_slv", "sigma_collapse"): 128,
    ("heston_slv", "near_expiry"): 128,
}


def _load_streams() -> list[dict]:
    """One record per (cell, greek), in contracts units."""
    paths = {os.path.basename(p): p for p in glob.glob(str(PILOT / "*__*.json"))}
    for path in glob.glob(str(DEEP / "*__*.json")):
        paths[os.path.basename(path)] = path  # deep pilot supersedes: more batches
    streams: list[dict] = []
    for name, path in sorted(paths.items()):
        variant, case = name.replace(".json", "").split("__")
        with open(path) as handle:
            evidence = json.load(handle)["evidence"]
        batch_estimates = np.asarray(
            evidence["reference"]["fine"]["batch_estimates"], dtype=float
        )
        for greek, column in GREEK_COLUMNS.items():
            certification = evidence["certifications"][greek]
            verdict = certification["verdict"]
            raw = batch_estimates[:, column]
            raw_mean = float(np.mean(raw))
            # Scale the raw column to contracts via the recorded contracts mean.
            scale = (
                float(certification["reference"]) / raw_mean
                if abs(raw_mean) > 1e-300
                else 1.0
            )
            substep = certification.get("reference_substep_batch_contracts")
            streams.append(
                {
                    "variant": variant,
                    "case": case,
                    "greek": greek,
                    "series": raw * scale,
                    "substep": (
                        np.asarray(substep, dtype=float) if substep else None
                    ),
                    "pde": float(certification["pde"]),
                    "pde_envelope": float(verdict["pde_discretization_envelope"]),
                    "frozen_bias_envelope": float(verdict["reference_bias_envelope"]),
                    "bound": float(verdict["economic_bound"]),
                    "frozen_status": str(verdict.get("status", "?")),
                }
            )
    return streams


def _replay(streams: Sequence[dict], *, floor: int) -> dict:
    tests = len(streams)
    rows = []
    for stream in streams:
        key = (stream["variant"], stream["case"])
        frozen = FROZEN_BATCHES[key]
        available = int(stream["series"].size)
        # The policy caps at what the archive can actually show.
        cap = min(frozen, available)
        effective_floor = max(int(floor), 2)
        if cap < effective_floor:
            rows.append(
                {
                    **{k: stream[k] for k in ("variant", "case", "greek")},
                    "status": "ARCHIVE_TOO_SHORT",
                    "batches_used": None,
                    "frozen_batches": frozen,
                    "archive_batches": available,
                }
            )
            continue
        policy = SequentialAdmissionPolicy(
            family_alpha=FAMILY_ALPHA,
            tests=tests,
            min_batches=max(effective_floor, 2),
            aggregate_floor_batches=int(floor),
            planned_batches=cap,
            max_batches=cap,
        )
        decision = scan_admission_stream(
            policy=policy,
            pde_value=stream["pde"],
            greek_series=stream["series"],
            substep_series=stream["substep"],
            pde_discretization_envelope=stream["pde_envelope"],
            economic_bound=stream["bound"],
            frozen_bias_envelope=stream["frozen_bias_envelope"],
        )
        rows.append(
            {
                **{k: stream[k] for k in ("variant", "case", "greek")},
                "status": decision.status.value,
                "batches_used": decision.batches_used,
                "frozen_status": stream["frozen_status"],
                "frozen_batches": frozen,
                "archive_batches": available,
                "gap": round(decision.reference_gap, 5),
                "greek_half_width": round(decision.greek_half_width, 5),
                "bias_envelope": round(decision.bias_envelope, 5),
                "policy_sha256": policy.sha256()[:12],
            }
        )

    # A cell costs the max over its two greeks: both must be decided.
    per_cell: dict = {}
    for row in rows:
        key = (row["variant"], row["case"])
        entry = per_cell.setdefault(
            key, {"frozen": row["frozen_batches"], "used": 0, "undecided": False}
        )
        if row["status"] == SequentialAdmissionStatus.ADMIT.value:
            entry["used"] = max(entry["used"], int(row["batches_used"]))
        else:
            # Undecided within the archive: it must spend its frozen allocation.
            entry["undecided"] = True
    for entry in per_cell.values():
        if entry["undecided"]:
            entry["used"] = entry["frozen"]
        entry["used"] = max(entry["used"], int(floor)) if floor else entry["used"]

    fleet_frozen = sum(e["frozen"] for e in per_cell.values())
    fleet_used = sum(e["used"] for e in per_cell.values())
    return {
        "floor": int(floor),
        "tests": tests,
        "rows": rows,
        "fleet_frozen_batches": fleet_frozen,
        "fleet_sequential_batches": fleet_used,
        "fleet_speedup": round(fleet_frozen / max(fleet_used, 1), 2),
        "cells_undecided": sorted(
            f"{v}/{c}" for (v, c), e in per_cell.items() if e["undecided"]
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floors", type=int, nargs="+", default=(0, 16, 128))
    args = parser.parse_args(argv)

    streams = _load_streams()
    print(f"{len(streams)} (cell, greek) streams from the archive\n")

    payload = {"family_alpha": FAMILY_ALPHA, "sweeps": []}
    for floor in args.floors:
        result = _replay(streams, floor=floor)
        payload["sweeps"].append(result)
        print(f"=== aggregate floor {floor} ===")
        print(
            f"{'cell':<30} {'greek':<6} {'frozen':>7} {'archive':>8} "
            f"{'T_stop':>7} {'status':>10}"
        )
        for row in result["rows"]:
            label = f"{row['variant']}/{row['case']}"
            print(
                f"{label:<30} {row['greek']:<6} {row['frozen_batches']:>7} "
                f"{row['archive_batches']:>8} "
                f"{str(row['batches_used'] or '-'):>7} {row['status']:>10}"
            )
        print(
            f"fleet: {result['fleet_frozen_batches']} -> "
            f"{result['fleet_sequential_batches']} batch-cells = "
            f"{result['fleet_speedup']}x"
        )
        if result["cells_undecided"]:
            print(f"undecided within archive: {', '.join(result['cells_undecided'])}")
        print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / "sequential_policy_replay.json"
    destination.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    print(f"wrote {destination}")
    print(
        "caveat: in-sample replay of data the pilot also saw, so the alpha "
        "guarantee is prospective, not demonstrated here."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
