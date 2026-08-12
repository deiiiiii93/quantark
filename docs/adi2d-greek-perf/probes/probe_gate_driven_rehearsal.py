"""End-to-end rehearsal: one real cell certified through the gate-driven loop.

Everything so far tests the pieces. ``probe_gate_driven_cell`` proved the policy
closes ``low_feller`` on real streams, but it reimplements the scan; the unit
tests drive ``gate_driven_reference_levels`` with deterministic stand-ins. This
runs the real ``certify_case`` -- PDE ladders, contracts scaling, the actual
certification arithmetic -- with a policy attached, so the wiring is exercised as
production will exercise it.

What it checks:
  * the cell stops early rather than spending its cap,
  * the banked reference has exactly the batches the decision rested on,
  * the resulting verdicts are PASS,
  * the evidence carries the policy that licensed the stop.

Usage:
    PYTHONPATH=$PWD python docs/adi2d-greek-perf/probes/probe_gate_driven_rehearsal.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
STAGE16 = ROOT / "example" / "mo_volmodels" / "16_adi_greek_certification.py"
OUTPUT_DIR = ROOT / "output" / "gate_driven_rehearsal"

# Matches the production matrix the real run will declare.
PRODUCTION_TESTS = 28
FAMILY_ALPHA = 0.05
COHORT_FLOOR = 128
MARGIN_FRACTION = 0.05


def load_stage16():
    spec = importlib.util.spec_from_file_location("stage16_cert", STAGE16)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage16_cert"] = module
    spec.loader.exec_module(module)
    return module


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="heston")
    parser.add_argument("--case", default="low_feller")
    parser.add_argument("--cap", type=int, default=512)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--margin", type=float, default=MARGIN_FRACTION)
    args = parser.parse_args(argv)

    s16 = load_stage16()
    case = {c.name: c for c in s16.certification_cases(quick=False)}[args.case]
    bridge = s16.HESTON_SPOT_BRIDGE_PROFILE_BY_CASE[case.name]

    policy = s16.SequentialAdmissionPolicy(
        family_alpha=FAMILY_ALPHA,
        tests=PRODUCTION_TESTS,
        min_batches=s16.MIN_PRODUCTION_RQMC_BATCHES,
        aggregate_floor_batches=COHORT_FLOOR,
        planned_batches=min(256, args.cap),
        max_batches=args.cap,
        margin_fraction=args.margin,
    )
    print(
        f"rehearsal: {args.variant}/{case.name}  cap {args.cap}  chunk {args.chunk}"
        f"  margin {args.margin:.0%}\n  policy {policy.sha256()[:12]}",
        flush=True,
    )

    started = time.perf_counter()
    cell = s16.certify_case(
        args.variant,
        case,
        quick=False,
        paths_per_batch=s16.PRODUCTION_HESTON_PATHS_PER_BATCH,
        batches=args.cap,
        seed=s16.HESTON_REFERENCE_SEED,
        hedge_inception_spot=s16.DEFAULT_HEDGE_INCEPTION_SPOT,
        sequential_policy=policy,
        sequential_chunk_batches=args.chunk,
        heston_spot_bridge_strata=bridge["strata"],
        heston_spot_bridge_dimensions=bridge["dimensions"],
        rqmc_batch_workers=(
            s16.PRODUCTION_RQMC_BATCH_WORKERS_BY_VARIANT_CASE[args.variant][case.name]
        ),
    )
    wall = time.perf_counter() - started

    record = cell.get("sequential_stopping")
    print(f"\nwall {wall/60:.1f} min")
    if record is None:
        print("FAIL: the cell reports no sequential_stopping record")
        return 1
    print(f"batches banked: {record['batches_banked']} of cap {args.cap}")
    for greek, decision in sorted(record["decisions"].items()):
        print(
            f"  {greek:<6} {decision['status']:<9} at {decision['batches_used']:>4}"
            f"  gap {decision['reference_gap']:.4f}"
            f"  total {decision['reference_gap'] + decision['total_uncertainty']:.4f}"
            f" / {decision['economic_bound']:.2f}"
        )
    for greek, certification in sorted(cell["certifications"].items()):
        verdict = certification["verdict"]
        print(
            f"  {greek:<6} frozen-form verdict {verdict['status']:<12}"
            f" total_uncertainty {verdict['total_uncertainty']:.4f}"
        )
    print(f"cell status: {cell['status']}")

    banked = int(record["batches_banked"])
    checks = {
        "stopped_before_the_cap": banked < args.cap,
        "banked_is_a_chunk_multiple": banked % args.chunk == 0,
        "policy_recorded": record["policy_sha256"] == policy.sha256(),
        "all_greeks_admitted": all(
            d["status"] == "ADMIT" for d in record["decisions"].values()
        ),
        "cell_passes": cell["status"] == "PASS",
    }
    print("\nchecks:")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / f"{args.variant}__{case.name}.json"
    destination.write_text(
        json.dumps(
            {
                "wall_seconds": wall,
                "cap": args.cap,
                "chunk": args.chunk,
                "checks": checks,
                "sequential_stopping": record,
                "status": cell["status"],
                "certifications": {
                    greek: certification["verdict"]
                    for greek, certification in cell["certifications"].items()
                },
            },
            indent=1,
            sort_keys=True,
            default=str,
        )
    )
    print(f"\nwrote {destination}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
