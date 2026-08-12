"""Prerequisite for gate-driven stopping: is the batch stream chunk-invariant?

Gate-driven stopping runs a cell in chunks -- price some batches, evaluate the
gate, continue if undecided. That is only sound if batch *k* is the SAME batch
however the run was segmented, because otherwise extending a run silently
rewrites the evidence already banked and the accumulated mean is not the mean of
a fixed point set.

The property tested is prefix invariance: the batch estimates from a B-batch run
must be byte-identical to the first B entries of any larger run. ``run_rqmc`` and
``run_paired_rqmc_greeks`` both iterate ``for batch_id in range(...)`` and key
each scramble off ``batch_id``, so this SHOULD hold structurally -- but P2 proved
*worker-count* invariance, which is a different property, and this program has
already been burned once by assuming a neighbouring result transfers.

Bitwise, not approximate: a 1e-16 drift would mean the point set moved.

Usage:
    PYTHONPATH=$PWD python docs/adi2d-greek-perf/probes/probe_chunk_invariance.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STAGE16 = ROOT / "example" / "mo_volmodels" / "16_adi_greek_certification.py"
OUTPUT_DIR = ROOT / "output" / "chunk_invariance"

LADDER = (4, 8, 16)


def load_stage16():
    spec = importlib.util.spec_from_file_location("stage16_cert", STAGE16)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage16_cert"] = module
    spec.loader.exec_module(module)
    return module


def heston_batches(s16, case, *, batches: int) -> np.ndarray:
    bridge = s16.HESTON_SPOT_BRIDGE_PROFILE_BY_CASE[case.name]
    substeps = s16.PRODUCTION_QE_SUBSTEPS_BY_VARIANT_CASE["heston"][case.name]
    evidence = s16.build_heston_high_control_evidence(
        case,
        paths_per_batch=s16.PRODUCTION_HESTON_PATHS_PER_BATCH,
        batches=int(batches),
        seed=s16.HESTON_REFERENCE_SEED,
        target_substeps=substeps["target"],
        fine_substeps=substeps["fine"],
        heston_spot_bridge_strata=bridge["strata"],
        heston_spot_bridge_dimensions=bridge["dimensions"],
        rqmc_batch_workers=s16.PRODUCTION_RQMC_BATCH_WORKERS_BY_VARIANT_CASE["heston"][
            case.name
        ],
    )
    return np.asarray(
        evidence["reference"]["fine"]["batch_estimates"], dtype=float
    )


def paired_batches(
    s16, variant: str, case, *, batches: int, first_batch: int = 0
) -> np.ndarray:
    """Per-batch delta/gamma rows from the generic paired reference."""
    product = s16.make_snowball(case, dense_ki=True)
    env = s16.make_environment(
        case.spot, float(np.sqrt(max(case.params.v0, case.params.theta)))
    )
    leverage = (
        s16.make_leverage_surface(case.maturity) if variant == "heston_slv" else None
    )
    substeps = s16.PRODUCTION_QE_SUBSTEPS_BY_VARIANT_CASE[variant][case.name]
    if variant == "heston":
        bridge = s16.HESTON_SPOT_BRIDGE_PROFILE_BY_CASE[case.name]
        extra = {
            "heston_spot_bridge_strata": bridge["strata"],
            "heston_spot_bridge_dimensions": bridge["dimensions"],
        }
    else:
        bridge = s16.SLV_SPOT_BRIDGE_PROFILE_BY_CASE[case.name]
        extra = {
            "slv_spot_bridge_strata": bridge["strata"],
            "slv_spot_bridge_dimensions": bridge["dimensions"],
        }
    result = s16.paired_mc_reference(
        variant,
        case,
        product,
        env,
        leverage,
        paths_per_batch=(
            s16.PRODUCTION_HESTON_PATHS_PER_BATCH
            if variant == "heston"
            else s16.PRODUCTION_SLV_PATHS_PER_BATCH
        ),
        batches=int(batches),
        first_batch=int(first_batch),
        seed=(
            s16.HESTON_REFERENCE_SEED if variant == "heston" else s16.SLV_PRIMARY_SEED
        ),
        substeps=substeps["target"],
        bump=s16.SPOT_BUMP,
        rqmc_batch_workers=s16.PRODUCTION_RQMC_BATCH_WORKERS_BY_VARIANT_CASE[variant][
            case.name
        ],
        **extra,
    )
    return np.column_stack(
        [
            np.asarray(result.batch_delta, dtype=float),
            np.asarray(result.batch_gamma, dtype=float),
        ]
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="ordinary_full")
    parser.add_argument("--ladder", type=int, nargs="+", default=LADDER)
    parser.add_argument(
        "--first-batch-chunks",
        action="store_true",
        help="assemble a run from offset chunks and compare against the whole",
    )
    parser.add_argument(
        "--variant",
        default="heston",
        choices=("heston", "heston_slv"),
        help="heston uses the high-control builder; heston_slv the paired reference",
    )
    args = parser.parse_args(argv)

    s16 = load_stage16()
    case = {c.name: c for c in s16.certification_cases(quick=False)}[args.case]
    print(
        f"chunk invariance on {args.variant}/{case.name}, ladder {list(args.ladder)}\n"
    )
    def batches_for(count: int) -> np.ndarray:
        if args.variant == "heston":
            return heston_batches(s16, case, batches=count)
        return paired_batches(s16, args.variant, case, batches=count)

    if args.first_batch_chunks:
        # The stronger claim the gate-driven loop actually relies on: a run
        # ASSEMBLED from offset chunks equals the whole run. This also tests that
        # widening the engine spec's max_batches (needed so the later ids exist)
        # leaves the draws alone -- a real-engine check the synthetic driver test
        # cannot make.
        total = max(args.ladder)
        half = total // 2
        whole = paired_batches(s16, args.variant, case, batches=total)
        first = paired_batches(
            s16, args.variant, case, batches=half, first_batch=0
        )
        second = paired_batches(
            s16, args.variant, case, batches=total - half, first_batch=half
        )
        joined = np.concatenate([first, second], axis=0)
        identical = joined.tobytes() == whole.tobytes()
        print(
            f"  assembled [0,{half}) + [{half},{total}) vs whole B={total}: "
            f"bitwise={identical}  "
            f"max_abs_diff={float(np.max(np.abs(joined - whole))):.3e}"
        )
        verdict = (
            "offset chunks assemble into the whole run"
            if identical
            else "OFFSET CHUNKS DIVERGE -- gate-driven stopping is unsound"
        )
        print(f"\nVERDICT: {verdict}")
        return 0 if identical else 1

    runs: dict[int, np.ndarray] = {}
    for batches in sorted(args.ladder):
        started = time.perf_counter()
        runs[batches] = batches_for(batches)
        print(
            f"  B={batches:<4} shape={runs[batches].shape}  "
            f"{time.perf_counter() - started:6.1f}s",
            flush=True,
        )

    print()
    largest = max(runs)
    reference = runs[largest]
    results = []
    ok = True
    for batches in sorted(runs):
        if batches == largest:
            continue
        candidate = runs[batches]
        prefix = reference[: candidate.shape[0]]
        identical = (
            candidate.shape == prefix.shape
            and candidate.tobytes() == prefix.tobytes()
        )
        max_diff = (
            float(np.max(np.abs(candidate - prefix)))
            if candidate.shape == prefix.shape
            else float("nan")
        )
        ok &= identical
        results.append(
            {
                "batches": batches,
                "against": largest,
                "bitwise": bool(identical),
                "max_abs_diff": max_diff,
            }
        )
        print(
            f"  B={batches} vs first {batches} of B={largest}: "
            f"bitwise={identical}  max_abs_diff={max_diff:.3e}"
        )

    verdict = (
        "prefix-invariant -- gate-driven chunking is sound"
        if ok
        else "NOT prefix-invariant -- chunked stopping would rewrite banked evidence"
    )
    print(f"\nVERDICT: {verdict}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / "chunk_invariance.json"
    destination.write_text(
        json.dumps(
            {"case": case.name, "ladder": sorted(runs), "checks": results,
             "prefix_invariant": bool(ok)},
            indent=1,
            sort_keys=True,
        )
    )
    print(f"wrote {destination}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
