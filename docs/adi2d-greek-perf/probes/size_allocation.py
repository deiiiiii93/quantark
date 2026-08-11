"""How many RQMC batches does the certification actually need, post-upgrade?

The frozen allocation (8192x1024 Heston, 1024x{128..512} SLV) was sized on
variance measured *before* the bridge8 treatment, QE-M, and the v-axis scheme
fix. Those upgrades changed two things that both cut the batch requirement:

1. Less variance per batch on the treated cells (2.14x/2.62x/1.49x in SE^2*sec).
2. A smaller aggregate bias estimate, because `v_drift_scheme="auto"` removes
   the sigma_collapse first-order error (+0.1149 heston, +0.1119 SLV). The
   aggregate gate is |estimate| + uncertainty <= bound, so shrinking the
   estimate hands the whole difference back to the uncertainty budget -- roughly
   0.016 of the 0.10 bound for Heston.

Running the frozen counts anyway costs ~42 h of wall clock. This script derives
the counts the gates actually require, from measured per-cell evidence, using
the certification's own formulas rather than a reconstruction of them.

The two gates, from `quantark/validation/greek_certification.py`:

    per-cell:   |diff_j| + t(conf, n-1)*SD_j/sqrt(n) + pde_j + bias_j <= 0.5
    aggregate:  |mean_j diff_j| + t(conf, n-1)*SD_agg/sqrt(n)
                                + pde_agg + bias_agg              <= 0.10

`SD_agg` is the SD of the *cell-averaged* series on common scrambles, so it
retains the cross-cell covariance the shared scrambles induce -- it is not
sqrt(sum SD_j^2)/J, and computing it that way would misstate the requirement.

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
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]

# Bounds and confidence are the certification's, not ours to choose.
CELL_BOUND = 0.5
AGGREGATE_BOUND = 0.10
CONFIDENCE = 0.975
# A pilot SD carries ~1/sqrt(2(n-1)) relative error, so size against an inflated
# SD rather than the point estimate. At n=32 that is ~13%; 1.25 covers it with
# room, and costs 1.56x in batches rather than a re-pilot.
SD_SAFETY = 1.25
MIN_BATCHES = 16


def _t(confidence: float, n: int) -> float:
    return float(stats.t.ppf(confidence, max(n - 1, 1)))


def load_cells(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    rows = []
    for cell in payload["cells"]:
        for greek in ("delta", "gamma"):
            certification = cell["certifications"][greek]
            series = np.asarray(
                cell["batch_difference_contracts"][greek], dtype=float
            )
            rows.append(
                {
                    "variant": cell["variant"],
                    "case": cell["case"]["name"],
                    "greek": greek,
                    "series": series,
                    "difference": float(
                        certification["difference_economic_contracts"]
                    ),
                    "pde": float(
                        certification["verdict"]["pde_discretization_envelope"]
                    ),
                    "bias": float(
                        certification["verdict"]["reference_bias_envelope"]
                    ),
                    "scheme": cell["variance_operator"]["scheme"],
                }
            )
    return rows


def batches_for_bound(
    *, sd: float, deterministic: float, bound: float, cap: int = 1 << 20
) -> Optional[int]:
    """Smallest n with |det| + t(n-1)*sd/sqrt(n) <= bound, or None if infeasible.

    The deterministic part does not shrink with n, so when it already exceeds
    the bound no allocation can rescue the cell -- that is a real answer, and
    silently returning the cap would hide it.
    """
    if deterministic >= bound:
        return None
    n = MIN_BATCHES
    while n <= cap:
        if deterministic + _t(CONFIDENCE, n) * sd / math.sqrt(n) <= bound:
            return n
        n *= 2
    return None


def size_variant(rows: Sequence[dict], variant: str) -> dict:
    """Per-cell and aggregate requirements for one variant."""
    variant_rows = [row for row in rows if row["variant"] == variant]
    report: dict = {"variant": variant, "cells": [], "aggregate": {}}

    for row in variant_rows:
        sd = float(np.std(row["series"], ddof=1)) * SD_SAFETY
        deterministic = abs(row["difference"]) + row["pde"] + row["bias"]
        report["cells"].append(
            {
                "case": row["case"],
                "greek": row["greek"],
                "sd": sd,
                "deterministic": deterministic,
                "pilot_batches": int(row["series"].size),
                "required_batches": batches_for_bound(
                    sd=sd, deterministic=deterministic, bound=CELL_BOUND
                ),
            }
        )

    # Aggregate: average across cells within each scramble, so the shared
    # randomization's covariance survives into the standard error.
    for greek in ("delta", "gamma"):
        greek_rows = [row for row in variant_rows if row["greek"] == greek]
        if not greek_rows:
            continue
        common = min(row["series"].size for row in greek_rows)
        stacked = np.vstack([row["series"][:common] for row in greek_rows])
        averaged = stacked.mean(axis=0)
        sd_aggregate = float(np.std(averaged, ddof=1)) * SD_SAFETY
        estimate = float(np.mean(averaged))
        pde_aggregate = float(np.mean([row["pde"] for row in greek_rows]))
        bias_aggregate = float(np.mean([row["bias"] for row in greek_rows]))
        deterministic = abs(estimate) + pde_aggregate + bias_aggregate
        report["aggregate"][greek] = {
            "estimate": estimate,
            "sd_averaged_series": sd_aggregate,
            "sd_if_wrongly_assumed_independent": float(
                np.sqrt(np.sum([np.var(r["series"][:common], ddof=1) for r in greek_rows]))
                / len(greek_rows)
            ) * SD_SAFETY,
            "pde_envelope": pde_aggregate,
            "bias_envelope": bias_aggregate,
            "deterministic_floor": deterministic,
            "common_scrambles": common,
            "required_batches": batches_for_bound(
                sd=sd_aggregate, deterministic=deterministic, bound=AGGREGATE_BOUND
            ),
        }
    return report


def render(report: dict, frozen: dict) -> str:
    lines = [f"\n=== {report['variant']} ===", "", "per-cell gate (bound 0.5):"]
    worst: dict[str, int] = {}
    for cell in report["cells"]:
        need = cell["required_batches"]
        worst[cell["case"]] = max(worst.get(cell["case"], 0), need or 1 << 20)
        lines.append(
            f"  {cell['case']:18s} {cell['greek']:5s} sd={cell['sd']:7.4f} "
            f"det={cell['deterministic']:7.4f} "
            f"need={'INFEASIBLE' if need is None else need}"
        )
    lines += ["", "aggregate gate (bound 0.10):"]
    for greek, agg in report["aggregate"].items():
        lines.append(
            f"  {greek:5s} est={agg['estimate']:+8.5f} "
            f"sd_avg={agg['sd_averaged_series']:7.4f} "
            f"(independent-assumption would say {agg['sd_if_wrongly_assumed_independent']:.4f}) "
            f"floor={agg['deterministic_floor']:7.4f} "
            f"need={'INFEASIBLE' if agg['required_batches'] is None else agg['required_batches']}"
        )
    binding = max(
        [a["required_batches"] or 1 << 20 for a in report["aggregate"].values()]
        + [max(worst.values())]
    )
    frozen_total = frozen.get(report["variant"], 0)
    lines += [
        "",
        f"  binding requirement: {binding} batches",
        f"  frozen allocation:   {frozen_total} batches",
        f"  reduction:           {frozen_total / max(binding, 1):.1f}x",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "output" / "allocation_pilot" / "sizing.json"
    )
    args = parser.parse_args(argv)

    rows = load_cells(args.evidence)
    frozen = {"heston": 1024, "heston_slv": 128}
    reports = []
    for variant in ("heston", "heston_slv"):
        if not any(row["variant"] == variant for row in rows):
            continue
        report = size_variant(rows, variant)
        reports.append(report)
        print(render(report, frozen))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2, sort_keys=True, default=str))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
