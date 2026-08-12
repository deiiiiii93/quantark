# Chunk invariance: verified, and it dissolves the SLV near-KI blocker

**Date** 2026-08-12 · Probe `probes/probe_chunk_invariance.py` · Raw
`output/chunk_invariance/` · Prerequisite for gate-driven stopping.

## The property

Gate-driven stopping prices a cell in chunks: run some batches, evaluate the
gate, continue if undecided. That is sound only if batch *k* is the same batch
however the run was segmented — otherwise extending a run silently rewrites
evidence already banked, and the accumulated mean is not the mean of a fixed
point set. Tested as **prefix invariance**: a B-batch run must be byte-identical
to the first B rows of any longer run.

Bitwise, not approximate. A 1e-16 drift would mean the point set moved.

## Result

| stream | estimator | ladder | verdict |
|---|---|---|---|
| `heston/ordinary_full` | high-control builder | 4 / 8 / 16 | **prefix-invariant**, max_abs_diff 0.0 |
| `heston_slv/ordinary_full` | paired reference (direct) | 4 / 8 / 16 | **prefix-invariant**, max_abs_diff 0.0 |
| `heston_slv/near_ki` | multilevel telescoping | — | **structurally excluded** (see below) |

Wall-clock is linear in batches — Heston 32.4 / 62.8 / 124.8 s and SLV
59.2 / 128.0 / 262.9 s for 4 / 8 / 16 — so per-batch cost is constant and
chunking carries no overhead penalty.

This is a different property from the worker-count invariance P2 proved and this
program re-confirmed; it was measured rather than inferred from it.

**Derived, not directly measured:** the bias envelope consumes the substep
difference series, which is the difference of two paired references at target and
fine substeps. Each is prefix-invariant by the above, so their difference is too.

## The multilevel cell is fixed at 256 by design

`build_slv_multilevel_reference` guards:

    primary_target.batches_used != PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE[case]
    or primary_fine.batches_used != primary_target.batches_used
    or primary_target.batches_used % PRODUCTION_SLV_BATCHES != 0
        -> raise "SLV primary batches do not match the multilevel profile"

Exact equality with the declared profile, so the cell accepts **only 256**. The
telescoping estimator's level weights and mid-level coupling are calibrated to
that profile, so this is a real constraint, not an incidental check.

**Consequence, and it is good news:** `heston_slv/near_ki` is not sized, it is
*declared*. The missing 256-batch sizing run that blocked the allocation — and
therefore blocked P1.4 — was never a decision to make. Under gate-driven stopping
the cell runs at its declared 256 and the other thirteen stop when their gates
close. Task #26 has no remaining content.

## What this clears

Gate-driven stopping is sound on both estimators that support it, the one cell
that cannot participate has a declared count, and the sizing exercise that has
been blocking the run is moot. The remaining work is wiring: replace the pinned
`rqmc_target_std=1e-12, min=max=batches` with the policy loop from
`quantark.validation.sequential_admission`, floored at the cohort's 128 and
capped at the frozen allocation so a run can never cost more than today's.
