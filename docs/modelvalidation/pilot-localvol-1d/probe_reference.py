"""Pilot controls 1 and 2 for the snowball-localvol-1d certification.

(1) CONVERGENCE, DEMONSTRATED NOT INHERITED. FINDING-2026-08-26 section 5
demonstrated substeps=8 for a THREE-year trade at mo-study scale. This is a
one-year trade at a different notional; it does not inherit that result. Walk
the ladder 4 -> 8 -> 16 and confirm the estimate has stopped moving.

Read the LADDER, not a single level: the FINDING's ladder CROSSES zero rather
than decaying to it, so a one-sided reading at any single level mis-signs the
error. That is exactly how substeps=1 made the PDE look 1.27 contracts wrong.

(2) ESTIMATOR CHOICE, MEASURED NOT DERIVED. Per-quantity standard error for
`plain` versus `one_step_survival`. Whichever meets the 0.25 x cell budget wins.
TIEBREAK: if `plain` meets the budget on every quantity, `plain` is adopted --
one estimator, fewer moving parts, and the same shape as the flat-BSM study.

The substeps=8 rung of the ladder IS the `plain` arm of control 2, so it is
computed once and reused.

Run (long -- expect ~90 minutes):
  PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
    docs/modelvalidation/pilot-localvol-1d/probe_reference.py
"""

import math
import statistics
import time

from quantark.modelvalidation.builders.equity_snowball_localvol import (
    build_localvol_mc_reference,
    build_localvol_pde_candidate,
)
from quantark.modelvalidation.study import CaseSpec, HedgeContractScale, SamplingPolicy

CRASH = "example/modelvalidation/data/iv_surface_20240208.json"
ENV = {"surface": CRASH, "rate": 0.02}
PRODUCT = {
    "strike_moneyness": 1.0,
    "ko_barrier_moneyness": 1.03,
    "ki_barrier_moneyness": 0.85,
    "ko_rate": 0.15,
    "rebate_rate": 0.15,
    "months": 12,
    "maturity": 1.0,
}
QUANTITIES = ("pv", "delta", "gamma")

SCALE = HedgeContractScale(
    hedge_multiplier=200.0, hedge_inception_spot=4993.105, notional=998621.0
)
CELL_BOUND = 0.5
SE_BUDGET = 0.25 * CELL_BOUND          # 0.125 contracts

# The two hardest cells: at inception on the steepest surface, and sitting just
# above the KI barrier where the barrier-local coefficients bind.
CELLS = {
    "ordinary": CaseSpec(name="ordinary"),
    "near_ki": CaseSpec(name="near_ki", environment_params={"spot_moneyness": 0.86}),
}

BATCHES = 6
PATHS = 65536
LADDER = (4, 8, 16)


def pde_values(case):
    candidate = build_localvol_pde_candidate(
        environment_params=ENV,
        product_params=PRODUCT,
        quantities=QUANTITIES,
        params={"accuracy": "standard"},
    )
    return candidate.evaluate(case).values


def sample(case, substeps, estimator, batches=BATCHES, paths=PATHS):
    policy = SamplingPolicy(
        paths_per_batch=paths,
        min_batches=batches,
        max_batches=batches,
        seed=20260828,
        bump=0.01,
    )
    ref = build_localvol_mc_reference(
        environment_params=ENV,
        product_params=PRODUCT,
        sampling=policy,
        quantities=QUANTITIES,
        params={
            "substeps_per_interval": substeps,
            "lv_time_sampling": "integrated",
            "estimator": estimator,
        },
    )
    started = time.time()
    rows = [ref.run_batch(case, i).values for i in range(batches)]
    elapsed = time.time() - started
    out = {}
    for q in QUANTITIES:
        xs = [r[q] for r in rows]
        se = statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("inf")
        out[q] = (statistics.fmean(xs), se)
    return out, elapsed


print("=" * 82)
print("CONTROL 1 -- reference convergence ladder (estimator=plain)")
print("=" * 82)

plain_at_8 = {}
for label, case in CELLS.items():
    pde = pde_values(case)
    pde_delta_c = SCALE.to_economic("delta", pde["delta"])
    print(f"\n--- {label}   PDE delta = {pde['delta']:.6f} "
          f"({pde_delta_c:.4f} contracts)   PDE pv = {pde['pv']:.4f}")
    print(f"{'substeps':>9} {'gap (contracts)':>18} {'+/- SE':>10} "
          f"{'sigma':>7} {'secs':>8}")
    for substeps in LADDER:
        stats, secs = sample(case, substeps, "plain")
        if substeps == 8:
            plain_at_8[label] = (stats, secs)
        mean, se = stats["delta"]
        gap = SCALE.to_economic("delta", mean - pde["delta"])
        gap_se = abs(SCALE.to_economic("delta", se))
        sigma = abs(gap) / gap_se if gap_se else float("inf")
        print(f"{substeps:9d} {gap:18.4f} {gap_se:10.4f} {sigma:7.2f} {secs:8.0f}")
        # flush so a long run is readable while it is still going
        print("", end="", flush=True)
    print("  READ THE LADDER: adjacent levels agreeing within ~1 sigma means it")
    print("  has stopped moving. A ladder that CROSSES zero mis-signs any single")
    print("  level read on its own.")

print()
print("=" * 82)
print(f"CONTROL 2 -- estimator choice (SE budget = {SE_BUDGET} contracts)")
print("=" * 82)

case = CELLS["near_ki"]
results = {}
stats, secs = plain_at_8["near_ki"]
results["plain"] = (stats, secs)
try:
    results["one_step_survival"] = sample(case, 8, "one_step_survival")
except Exception as exc:                                        # noqa: BLE001
    print(f"\none_step_survival UNAVAILABLE: {type(exc).__name__}: {exc}")

for estimator, (stats, secs) in results.items():
    print(f"\n{estimator}  ({secs:.0f}s for {BATCHES} batches at substeps=8)")
    for q in QUANTITIES:
        mean, se = stats[q]
        se_c = abs(SCALE.to_economic(q, se))
        verdict = "MEETS" if se_c <= SE_BUDGET else "OVER BUDGET"
        print(f"  {q:6s} mean = {mean:14.6f}   SE = {se_c:9.5f} contracts   {verdict}")

print()
print("DECISION RULE: if `plain` meets the budget on every quantity, adopt")
print("`plain`. Introduce `one_step_survival` only for a quantity `plain`")
print("cannot resolve. The DISCRETIZATION never splits, only the estimator.")
