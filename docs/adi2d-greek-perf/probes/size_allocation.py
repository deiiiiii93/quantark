"""How many RQMC batches does the certification actually need, post-upgrade?

The frozen allocation (8192x1024 Heston, 1024x{128..512} SLV) was sized on
variance measured *before* the bridge8 treatment, QE-M, and the v-axis scheme
fix. Those upgrades changed two things that both cut the batch requirement:

1. Less variance per batch on the treated cells (2.14x/2.62x/1.49x in SE^2*sec).
2. A smaller aggregate bias estimate, because `v_drift_scheme="auto"` removes
   the sigma_collapse first-order error (+0.1149 heston, +0.1119 SLV). Both gates
   read |estimate| + uncertainty <= bound, so shrinking the estimate hands the
   whole difference back to the uncertainty budget.

Running the frozen counts anyway costs ~42 h of wall clock. This reproduces the
certification's own arithmetic on measured pilot evidence and solves for the
batch count each gate actually requires.

The gates, mirroring `quantark/validation/greek_certification.py` and the
aggregate assembly in stage 16:

    per-cell (bound 0.5, per greek):
        |diff_j| + c(n)*SD_j/sqrt(n) + pde_j + (|sub_j| + c(n)*SDsub_j/sqrt(n))

    aggregate delta bias (bound 0.1, per variant):
        |mean(B)| + c(n)*SD(B)/sqrt(n) + sum_axis |mean_j signed_j[axis]|
                  + (|mean(S)| + c(n)*SD(S)/sqrt(n))

with c(n) = t.ppf(0.5 + 0.5*0.975, n-1), B the cell-averaged difference series on
common scrambles, and S the cell-averaged substep series. Three details here are
easy to get wrong and all three change the answer:

* c(n) is the TWO-sided quantile at 0.975, i.e. t.ppf(0.9875, .) ~ 2.24, not
  t.ppf(0.975, .) ~ 1.96. Using the one-sided value understates every half-width
  by 14%.
* The aggregate PDE envelope sums |mean across cells| PER AXIS, so signed
  refinements CANCEL across cells. Averaging per-cell envelopes instead
  overstates the deterministic floor and oversizes the allocation.
* The substep bias envelope is not deterministic: its half-width shrinks with n
  too, so treating it as a fixed floor also oversizes.

Usage:
    PYTHONPATH=$PWD python docs/adi2d-greek-perf/probes/size_allocation.py \
        --evidence output/allocation_pilot/adi_greek_certification.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[3]

# All four are the certification's own constants, not ours to choose:
# CONFIDENCE 0.95 -> STOCHASTIC_COMPONENT_CONFIDENCE = 1 - (1-0.95)/2.
CONFIDENCE = 0.975
CELL_BOUND = 0.5
AGGREGATE_DELTA_BOUND = 0.1
MIN_BATCHES = 16
AXES = ("n_x", "n_v", "n_t")
# A pilot SD carries ~1/sqrt(2(n-1)) relative error -- ~13% at n=32. Size against
# an inflated SD so a lucky pilot cannot under-allocate the real run.
SD_SAFETY = 1.25


def _critical(n: int) -> float:
    """Two-sided Student-t critical value, exactly as certify_equivalence uses it."""
    return float(student_t.ppf(0.5 + 0.5 * CONFIDENCE, max(n - 1, 1)))


def _series(values, common: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr if common is None else arr[:common]


def load_cells(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    cells = []
    for cell in payload["cells"]:
        entry = {
            "variant": cell["variant"],
            "case": cell["case"]["name"],
            "scheme": cell["variance_operator"]["scheme"],
            "greeks": {},
        }
        for greek in ("delta", "gamma"):
            certification = cell["certifications"][greek]
            entry["greeks"][greek] = {
                "difference": float(certification["difference_economic_contracts"]),
                "differences": _series(cell["batch_difference_contracts"][greek]),
                "substeps": _series(
                    certification["reference_substep_batch_contracts"]
                ),
                "pde_envelope": float(
                    certification["verdict"]["pde_discretization_envelope"]
                ),
                "pde_signed": {
                    axis: float(
                        certification["pde_signed_refinement_contracts"][axis]
                    )
                    for axis in AXES
                },
                "status": certification["verdict"]["status"],
            }
        cells.append(entry)
    return cells


def smallest_n(
    *,
    fixed: float,
    scaling_sds: Sequence[float],
    bound: float,
    cap: int = 1 << 20,
) -> Optional[int]:
    """Smallest power-of-two n >= 16 with fixed + c(n)*sum(sd/sqrt(n)) <= bound.

    ``fixed`` is the part no allocation can move. When it already exceeds the
    bound the honest answer is None: returning the cap would dress an
    unreachable gate up as an expensive one.
    """
    if fixed >= bound:
        return None
    n = MIN_BATCHES
    while n <= cap:
        stochastic = _critical(n) * sum(scaling_sds) / math.sqrt(n)
        if fixed + stochastic <= bound:
            return n
        n *= 2
    return None


def size_cells(cells: Sequence[dict]) -> list[dict]:
    rows = []
    for cell in cells:
        for greek, data in cell["greeks"].items():
            sd_difference = float(np.std(data["differences"], ddof=1)) * SD_SAFETY
            sd_substep = float(np.std(data["substeps"], ddof=1)) * SD_SAFETY
            fixed = (
                abs(data["difference"])
                + data["pde_envelope"]
                + abs(float(np.mean(data["substeps"])))
            )
            rows.append(
                {
                    "variant": cell["variant"],
                    "case": cell["case"],
                    "greek": greek,
                    "pilot_batches": int(data["differences"].size),
                    "sd_difference": sd_difference,
                    "sd_substep": sd_substep,
                    "fixed": fixed,
                    "required_batches": smallest_n(
                        fixed=fixed,
                        scaling_sds=(sd_difference, sd_substep),
                        bound=CELL_BOUND,
                    ),
                    "status_at_pilot": data["status"],
                }
            )
    return rows


def size_aggregate(cells: Sequence[dict], variant: str) -> Optional[dict]:
    """Reproduce stage 16's per-variant mean-signed-delta-bias gate."""
    variant_cells = [cell for cell in cells if cell["variant"] == variant]
    if len(variant_cells) < 2:
        return None
    common = min(
        cell["greeks"]["delta"]["differences"].size for cell in variant_cells
    )

    differences = np.vstack(
        [cell["greeks"]["delta"]["differences"][:common] for cell in variant_cells]
    )
    averaged = differences.mean(axis=0)

    substeps = np.vstack(
        [cell["greeks"]["delta"]["substeps"][:common] for cell in variant_cells]
    )
    averaged_substeps = substeps.mean(axis=0)

    # Signed per axis, THEN absolute -- cancellation across cells is the point.
    signed_axes = {
        axis: float(
            np.mean(
                [cell["greeks"]["delta"]["pde_signed"][axis] for cell in variant_cells]
            )
        )
        for axis in AXES
    }
    pde_envelope = float(sum(abs(value) for value in signed_axes.values()))
    mean_of_per_cell_envelopes = float(
        np.mean([cell["greeks"]["delta"]["pde_envelope"] for cell in variant_cells])
    )

    sd_averaged = float(np.std(averaged, ddof=1)) * SD_SAFETY
    sd_substep = float(np.std(averaged_substeps, ddof=1)) * SD_SAFETY
    estimate = float(np.mean(averaged))
    fixed = estimate.__abs__() + pde_envelope + abs(float(np.mean(averaged_substeps)))

    return {
        "variant": variant,
        "cases": [cell["case"] for cell in variant_cells],
        "common_scrambles": common,
        "estimate": estimate,
        "sd_averaged_series": sd_averaged,
        "sd_if_assumed_independent": float(
            np.sqrt(np.sum(np.var(differences, axis=1, ddof=1)))
            / differences.shape[0]
        )
        * SD_SAFETY,
        "sd_substep": sd_substep,
        "pde_signed_axes": signed_axes,
        "pde_envelope": pde_envelope,
        "mean_of_per_cell_envelopes": mean_of_per_cell_envelopes,
        "fixed": fixed,
        "bound": AGGREGATE_DELTA_BOUND,
        "required_batches": smallest_n(
            fixed=fixed,
            scaling_sds=(sd_averaged, sd_substep),
            bound=AGGREGATE_DELTA_BOUND,
        ),
    }


def render(
    cell_rows: Sequence[dict], aggregate: Optional[dict], frozen: dict[str, int]
) -> str:
    variant = aggregate["variant"] if aggregate else cell_rows[0]["variant"]
    lines = [f"\n=== {variant} ===", "", f"per-cell gate (bound {CELL_BOUND}):"]
    per_cell_max = 0
    for row in sorted(cell_rows, key=lambda r: (r["case"], r["greek"])):
        need = row["required_batches"]
        per_cell_max = max(per_cell_max, need or 1 << 20)
        lines.append(
            f"  {row['case']:18s} {row['greek']:5s} "
            f"sd={row['sd_difference']:7.4f} fixed={row['fixed']:7.4f} "
            f"need={'INFEASIBLE' if need is None else need:>11} "
            f"({row['status_at_pilot']} at {row['pilot_batches']})"
        )
    if aggregate is None:
        lines.append("\n  (single cell -- no aggregate gate)")
        return "\n".join(lines)

    need = aggregate["required_batches"]
    lines += [
        "",
        f"aggregate mean signed delta bias (bound {AGGREGATE_DELTA_BOUND}):",
        f"  estimate            {aggregate['estimate']:+.5f}",
        f"  sd(cell-averaged)   {aggregate['sd_averaged_series']:.5f}"
        f"   [assuming independence would say "
        f"{aggregate['sd_if_assumed_independent']:.5f}]",
        f"  pde envelope        {aggregate['pde_envelope']:.5f}"
        f"   [mean of per-cell envelopes would say "
        f"{aggregate['mean_of_per_cell_envelopes']:.5f}]",
        f"  signed axes         "
        + ", ".join(f"{k}={v:+.5f}" for k, v in aggregate['pde_signed_axes'].items()),
        f"  fixed floor         {aggregate['fixed']:.5f}",
        f"  need                {'INFEASIBLE' if need is None else need}",
    ]
    # The aggregate consumes min(batches) across cells, so its requirement is a
    # FLOOR under every case; a case whose own gate needs more just gets more.
    # That is cheaper than levelling every case up to the worst one.
    lines += ["", "derived per-case allocation (max of own gate and aggregate floor):"]
    floor = need or 1 << 20
    per_case: dict[str, int] = {}
    for row in cell_rows:
        want = row["required_batches"] or 1 << 20
        per_case[row["case"]] = max(per_case.get(row["case"], 0), want, floor)
    for case, count in sorted(per_case.items(), key=lambda kv: -kv[1]):
        driver = "own gate" if count > floor else "aggregate floor"
        lines.append(f"  {case:18s} {count:>6}   ({driver})")

    total = sum(per_case.values())
    frozen_total = sum(frozen.get(case, 0) for case in per_case)
    lines += [
        "",
        f"  aggregate floor     {floor} batches",
        f"  derived total       {total} batch-cells",
        f"  frozen total        {frozen_total} batch-cells",
        f"  reduction           {frozen_total / max(total, 1):.1f}x",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output" / "allocation_pilot" / "sizing.json",
    )
    args = parser.parse_args(argv)

    cells = load_cells(args.evidence)
    rows = size_cells(cells)
    # The frozen counts we are comparing against, per case, from stage 16's
    # PRODUCTION_HESTON_BATCHES_BY_CASE / PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE.
    frozen = {
        "heston": {
            "ordinary_full": 1024, "ordinary_decayed": 1024, "near_ko": 1024,
            "near_ki": 2048, "low_feller": 1024, "sigma_collapse": 1024,
            "near_expiry": 1024,
        },
        "heston_slv": {
            "ordinary_full": 128, "ordinary_decayed": 128, "near_ko": 128,
            "near_ki": 256, "low_feller": 512, "sigma_collapse": 128,
            "near_expiry": 128,
        },
    }
    report = []
    for variant in ("heston", "heston_slv"):
        variant_rows = [row for row in rows if row["variant"] == variant]
        if not variant_rows:
            continue
        aggregate = size_aggregate(cells, variant)
        print(render(variant_rows, aggregate, frozen[variant]))
        report.append(
            {"variant": variant, "cells": variant_rows, "aggregate": aggregate}
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
