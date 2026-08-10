"""Does the measured V1 treatment still reach the overnight precision target?

Reads the completed demo logs, rebuilds the aggregate statistical budget from
the recovered schema-11 variance shares, and solves the cost-weighted Neyman
allocation for the spec's 0.02-contract half-width inside the 12 h cap.

Everything here is arithmetic on measured inputs; no new MC is run.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from quantark.validation import (
    CellPrecision,
    neyman_allocation,
    projected_aggregate_halfwidth,
)

LOGS = Path(__file__).resolve().parent / "logs"

# Recovered schema-11 statistical budget (2026-08-07 analysis): the aggregate
# half-width and each cell's share of the aggregate MC variance at 128 batches.
SCHEMA11_HALFWIDTH = 0.0545
SCHEMA11_BATCHES = 128
VARIANCE_SHARE = {
    "ordinary_decayed": 0.375,
    "ordinary_full": 0.364,
    "sigma_collapse": 0.158,
    "low_feller": 0.049,
    "near_ko": 0.026,
    "near_expiry": 0.017,
    "near_ki": 0.011,
}
# Cells with no demo keep their untreated cost; measured cells override it.
DEFAULT_SECONDS_PER_BATCH = 25.0
TARGET_HALFWIDTH = 0.02
BUDGET_HOURS = 12.0


def load_rows() -> dict:
    rows = {}
    for path in sorted(LOGS.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if "summary" in record:
                continue
            rows.setdefault(record["cell"], {})[record["label"]] = record
    return rows


def main() -> None:
    rows = load_rows()

    # Per-cell batch SD in contracts, implied by the schema-11 share split, so
    # the treated and untreated cells sit on one consistent scale.
    total_variance_at_128 = (SCHEMA11_HALFWIDTH / 1.96) ** 2
    print(f"{'cell':>18} {'var share':>10} {'sd_base':>9} {'sd_treat':>9} {'var gain':>9} {'sec/b':>7}")
    cells_baseline, cells_treated = [], []
    for cell, share in VARIANCE_SHARE.items():
        # SE_j^2 contribution = share * total; with k cells and n batches,
        # contribution = sd_j^2 / (k^2 n)  =>  sd_j = sqrt(share*total*k^2*n)
        k = len(VARIANCE_SHARE)
        sd_base = math.sqrt(share * total_variance_at_128 * k * k * SCHEMA11_BATCHES)
        demo = rows.get(cell, {})
        if "baseline" in demo and "bridge8" in demo:
            gain = (
                demo["baseline"]["batch_sd_contracts"]
                / demo["bridge8"]["batch_sd_contracts"]
            ) ** 2
            seconds = demo["bridge8"]["seconds_per_batch"]
        else:
            gain = 1.0
            seconds = DEFAULT_SECONDS_PER_BATCH
        sd_treated = sd_base / math.sqrt(gain)
        print(
            f"{cell:>18} {share:>10.3f} {sd_base:>9.3f} {sd_treated:>9.3f} "
            f"{gain:>9.2f} {seconds:>7.1f}"
        )
        cells_baseline.append(
            CellPrecision(cell, SCHEMA11_BATCHES, sd_base, seconds)
        )
        cells_treated.append(
            CellPrecision(cell, SCHEMA11_BATCHES, sd_treated, seconds)
        )

    print(f"\nreconstructed schema-11 half-width @128: "
          f"{projected_aggregate_halfwidth(cells_baseline):.4f} (recorded {SCHEMA11_HALFWIDTH})")
    print(f"treated half-width @128 batches:        "
          f"{projected_aggregate_halfwidth(cells_treated):.4f}")

    budget_seconds = BUDGET_HOURS * 3600.0
    allocation = neyman_allocation(cells_treated, budget_seconds, min_batches=32)
    allocated = [
        CellPrecision(c.name, allocation[c.name], c.batch_sd, c.seconds_per_batch)
        for c in cells_treated
    ]
    spent = sum(allocation[c.name] * c.seconds_per_batch for c in cells_treated)
    achieved = projected_aggregate_halfwidth(allocated)

    print(f"\nNeyman allocation inside {BUDGET_HOURS} h (single stream):")
    for c in cells_treated:
        print(f"  {c.name:>18}: {allocation[c.name]:>6} batches")
    print(f"  serial cost: {spent/3600:.1f} h; with 6-way cell parallelism: {spent/3600/6:.1f} h")
    print(f"  achieved half-width: {achieved:.4f} (target {TARGET_HALFWIDTH})")
    print(f"  VERDICT: {'target met' if achieved <= TARGET_HALFWIDTH else 'target MISSED at this budget'}")

    # How much serial budget would the target actually need?
    scale = (achieved / TARGET_HALFWIDTH) ** 2
    print(f"\n  serial-budget multiple needed for the target: {scale:.2f}x"
          f"  => {BUDGET_HOURS*scale:.1f} h serial")

    # The cap is WALL-CLOCK, and the host runs 7 cells x 2 workers on 14 cores.
    # Under a wall-clock constraint the streams do not compete for one budget,
    # so each cell simply fills its own stream: n_j = wall / cost_j. Neyman
    # weighting is the answer to a shared-budget question that no longer binds.
    print("\nWall-clock model (7 cells concurrently, 2 workers each):")
    for contention in (1.0, 1.5, 2.0):
        filled = [
            CellPrecision(
                c.name,
                max(32, int(BUDGET_HOURS * 3600.0 / (c.seconds_per_batch * contention))),
                c.batch_sd,
                c.seconds_per_batch * contention,
            )
            for c in cells_treated
        ]
        halfwidth = projected_aggregate_halfwidth(filled)
        # Hours actually needed to hit the target under this contention factor.
        hours_needed = BUDGET_HOURS * (halfwidth / TARGET_HALFWIDTH) ** 2
        verdict = "MET" if halfwidth <= TARGET_HALFWIDTH else "missed"
        print(
            f"  contention x{contention:.1f}: half-width {halfwidth:.4f} in "
            f"{BUDGET_HOURS:.0f} h [{verdict}]; target needs {hours_needed:.1f} h"
        )
    print(
        "\n  Contention x1.0 assumes the measured 3-cell rates hold at 7 cells;\n"
        "  x2.0 is the pessimistic case where doubling the worker count halves\n"
        "  per-worker throughput. Peak RSS measured 4.7 GiB per 2-worker cell,\n"
        "  so 7 concurrent cells is about 33 GiB of the host's 48 GiB."
    )


if __name__ == "__main__":
    main()
