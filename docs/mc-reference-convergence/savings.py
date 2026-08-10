"""What does the new work save on the next stage-17 aggregate run?

Compares the frozen 4096/256 design against the treated + adaptively allocated
run, at equal achieved precision, using measured per-batch costs. Pure
arithmetic on measured inputs; no Monte Carlo is run.

Cost model: one "batch" is a paired (down, base, up) reference batch at
PRODUCTION_PRIMARY_PATHS_PER_BATCH = 1024 paths, which is exactly what the
2026-08-10 demos timed. Costs are stated as single-stream seconds; the
wall-clock lines then divide by the concurrency the host actually runs.
"""

from __future__ import annotations

import math

from quantark.validation import CellPrecision, neyman_allocation

# Per-cell batch SD in contracts, reconstructed from the recorded schema-11
# variance shares (docs/mc-reference-convergence/feasibility.py validates the
# reconstruction: 0.0546 vs the recorded 0.0545 aggregate half-width).
BATCH_SD = {
    "ordinary_decayed": 1.349,
    "ordinary_full": 1.329,
    "sigma_collapse": 0.875,
    "low_feller": 0.487,
    "near_ko": 0.355,
    "near_expiry": 0.287,
    "near_ki": 0.231,
}
# Variance gain from the shipped bridge8 treatment (measured 2026-08-10).
VARIANCE_GAIN = {
    "ordinary_decayed": 2.67,
    "ordinary_full": 2.10,
    "sigma_collapse": 1.46,
}
# Measured seconds per paired batch (2 workers, 3 concurrent cells). Cells the
# demos did not time inherit the mean of those that were.
SECONDS_PER_BATCH = {
    "ordinary_decayed": 17.4,
    "ordinary_full": 33.4,
    "sigma_collapse": 33.3,
}
DEFAULT_SECONDS = 28.0

# The frozen design: 4096 primary batches per refreshed case, and its own
# recorded target (the guarded interval it was sized to deliver).
FROZEN_BATCHES = 4096
FROZEN_CASES = (
    "ordinary_full",
    "ordinary_decayed",
    "near_ko",
    "sigma_collapse",
    "near_expiry",
)
FROZEN_GUARDED_INTERVAL = (-0.09927214226746098, -0.038637498083171816)
CARRIED_BATCHES = 128  # near_ki and low_feller ride carried schema-11 cohorts
CONFIDENCE_T = 1.96
OUR_TARGET = 0.02
CONCURRENCY = 6  # 6-7 cells x 2 workers fits the 14-core host


def seconds(cell: str) -> float:
    return SECONDS_PER_BATCH.get(cell, DEFAULT_SECONDS)


def halfwidth(batches: dict, treated: bool) -> float:
    k = len(BATCH_SD)
    total = 0.0
    for cell, sd in BATCH_SD.items():
        gain = VARIANCE_GAIN.get(cell, 1.0) if treated else 1.0
        total += (sd * sd / gain) / batches[cell]
    return CONFIDENCE_T * math.sqrt(total / (k * k))


def cost_hours(batches: dict) -> float:
    return sum(batches[c] * seconds(c) for c in BATCH_SD) / 3600.0


def batches_for(target: float, treated: bool, adaptive: bool) -> dict:
    """Smallest batch counts reaching `target`, either flat or Neyman-shaped."""
    if not adaptive:
        # Flat scaling of the frozen shape until the target is met.
        n = 16
        while True:
            trial = {c: (n if c in FROZEN_CASES else CARRIED_BATCHES) for c in BATCH_SD}
            if halfwidth(trial, treated) <= target or n > 2_000_000:
                return trial
            n = int(n * 1.05) + 1
    cells = [
        CellPrecision(
            name=c,
            n_batches=32,
            batch_sd=BATCH_SD[c] / math.sqrt(VARIANCE_GAIN.get(c, 1.0) if treated else 1.0),
            seconds_per_batch=seconds(c),
        )
        for c in BATCH_SD
    ]
    budget = 600.0
    while budget < 4_000_000.0:
        allocation = neyman_allocation(cells, budget_seconds=budget, min_batches=32)
        if halfwidth(allocation, treated) <= target:
            return allocation
        budget *= 1.05
    raise RuntimeError("target unreachable")


def show(label: str, batches: dict, treated: bool) -> tuple[float, float]:
    hours = cost_hours(batches)
    hw = halfwidth(batches, treated)
    print(
        f"{label:<44} half-width {hw:.4f}  "
        f"{hours:8.1f} h serial  {hours / CONCURRENCY:6.1f} h wall"
    )
    return hours, hw


def main() -> None:
    frozen = {c: (FROZEN_BATCHES if c in FROZEN_CASES else CARRIED_BATCHES) for c in BATCH_SD}
    frozen_target = (FROZEN_GUARDED_INTERVAL[1] - FROZEN_GUARDED_INTERVAL[0]) / 2.0
    print(f"frozen design's own recorded target half-width: {frozen_target:.4f}\n")

    print("=== the run as designed before this work ===")
    base_hours, base_hw = show("frozen 4096/256, untreated estimators", frozen, False)

    print("\n=== same precision, with the new work ===")
    a = batches_for(base_hw, treated=True, adaptive=False)
    treat_hours, _ = show("+ bridge8 treatment (flat shape)", a, True)
    b = batches_for(base_hw, treated=True, adaptive=True)
    both_hours, _ = show("+ treatment + Neyman allocation", b, True)

    print("\n=== our tighter 0.02 target, with the new work ===")
    c = batches_for(OUR_TARGET, treated=True, adaptive=True)
    tight_hours, _ = show("+ treatment + Neyman @ 0.02", c, True)
    d = batches_for(OUR_TARGET, treated=False, adaptive=False)
    old_tight_hours, _ = show("old machinery @ 0.02 (for scale)", d, False)

    print("\n=== savings at equal precision ===")
    print(f"  treatment alone      {base_hours / treat_hours:5.2f}x  "
          f"({base_hours - treat_hours:7.1f} h serial saved, "
          f"{(base_hours - treat_hours) / CONCURRENCY:5.1f} h wall)")
    print(f"  treatment + Neyman   {base_hours / both_hours:5.2f}x  "
          f"({base_hours - both_hours:7.1f} h serial saved, "
          f"{(base_hours - both_hours) / CONCURRENCY:5.1f} h wall)")
    print(f"\n  reaching the tighter 0.02 target costs {tight_hours / 3600.0 * 3600.0:.1f} h serial "
          f"({tight_hours / CONCURRENCY:.1f} h wall), versus {old_tight_hours:.1f} h serial "
          f"({old_tight_hours / CONCURRENCY:.1f} h wall) on the old machinery "
          f"-- a {old_tight_hours / tight_hours:.2f}x reduction.")


if __name__ == "__main__":
    main()
