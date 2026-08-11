# Sequential admission: shipped policy, and the honest fleet number

**Date** 2026-08-11 · Module `quantark/validation/sequential_admission.py` ·
Tests `test/test_sequential_admission.py` (25) · Probe
`probes/probe_sequential_policy_replay.py` · Raw `output/sequential_policy_replay/`

Adopts the cert-cost-reduction P1 finding. The rule itself is P1's; what is new
here is the declared policy, the safety property, and a corrected fleet figure.

## What makes early stopping legitimate

The frozen gate spends a declared `B` and judges once, which is what licenses
the Student-t quantile. Watching that same interval as batches arrive and
stopping at the first favourable look is invalid — the interval is valid at one
pre-chosen `B` only. The module replaces the t-quantile with the asymptotic
confidence sequence of Waudby-Smith et al. (*Ann. Statist.* 2024), which covers
the mean simultaneously at every `t`.

**The safety property, pinned by test:** the confidence-sequence half-width is
strictly wider than the fixed-`B` half-width at every batch count (verified at
16 / 32 / 128 / 512 / 2048). So a sequential ADMIT at `t` implies the frozen
gate would also have passed had it been judged at `t`. Early stopping trades
wall-clock for width, never evidence for optimism.

Two components shrink, not one: the Greek interval and the substep bias
envelope are estimated from the same batches, so the rule tracks both and splits
alpha between them. Freezing the bias envelope mis-attributes why a cell needs
batches — on `heston/near_ki` it is the bias interval, not the Greek interval,
that the large allocation buys.

Everything is declared in advance and hashed (`policy.sha256()`): family alpha,
the test count K it is spread over, floors, cap, and the `rho` that fixes the
sequence's shape. `rho` is tuned at a *declared* horizon, never from the data.
A policy chosen after seeing the data is not a policy.

## Cross-validation against P1

Replaying the same archived streams through the shipped module reproduces P1's
per-cell picture from an independent implementation:

| stream | P1 | shipped module |
|---|---|---|
| smooth Heston cells | floor | floor (16) |
| `heston/near_ki` delta | 22 | 23 |
| `heston/near_ki` **gamma** | **160** | **160** |
| SLV delta cells | floor–28 | floor–28 |
| SLV gamma, 4 fat-envelope cells | UNDECIDED | EXHAUSTED |

## The fleet number is 7.9×, not 13.8×

| aggregate floor | fleet batch-cells | speedup |
|---|---|---|
| 2 (per-cell gates only) | 9344 → 1091 | 8.56× |
| **16 (production minimum)** | **9344 → 1186** | **7.88×** |
| 128 (cohort path) | not measurable from a 32-batch archive | ~4.8× projected |

Two separate corrections to 13.8×:

1. **Undecided cells were credited a projection.** P1's fleet total substitutes
   a Mode-B *projected* stop for the four SLV gamma cells that never decided
   inside the archive. This replay charges them their full frozen allocation
   instead, because a projection is not evidence. That alone moves 13.8× → 8.6×.

2. **The aggregate floor is not modelled at all.** `cohort_contribution`
   requires *exactly* `AMENDMENT_AGGREGATE_BATCHES = 128` batches from every row
   and raises otherwise, because the aggregate reads a common scramble prefix
   across cells. A cell that stopped at 16 cannot contribute to the aggregate at
   all. With a 128 floor the six smooth Heston cells sit at 128 (they admit far
   earlier, but may not stop there), `near_ki` at 160, and the fleet lands near
   4.8× — subject to the four SLV gamma cells deciding by 128, which a 32-batch
   archive cannot show.

The floor is therefore a **policy parameter**, and which value binds depends on
what consumes the evidence: stage 16's own per-variant aggregate needs only
`common_batches >= MIN_PRODUCTION_RQMC_BATCHES` (16), while the amendment cohort
path needs 128.

## The design tension worth naming

The existing adaptive machinery in `17_adi_slv_aggregate_certification.py` is
deliberately **precision-blind**: `"stopping_rule": "precision_blind"`, the
allocation is frozen and hashed *before* the run so "the recorded decision
cannot drift with the data that follows it", and `adaptive_batches_by_case`
takes `max(allocation, default)` so "an adaptive run never *reduces* evidence
below what the fixed allocation would have banked."

Sequential admission is estimate-dependent stopping and does reduce below the
fixed allocation. It is not unsound — the anytime-valid width is precisely what
buys that right — but it consciously replaces the previous stance. Adopting it
means accepting *anytime-valid inference with pre-declared parameters* in place
of *freeze-before-run plus never-reduce*. That is the policy decision the
evidence doc flagged as open, and it should be made explicitly rather than
inherited from a module landing.

## What is not done

The module is library code with tests and a validated replay; it is **not yet
wired into the stage-16 run loop**. Wiring it requires one more prerequisite
that is not yet verified: driving batches incrementally needs the batch stream
to be **chunk-invariant** (batch *k* identical whether produced in one call or
several). P2 proved *worker-count* invariance and this program re-confirmed it
bitwise, but that is a different property from chunk-boundary invariance, and it
must be measured before the loop is restructured.

## Two hash-coverage gaps found while landing this

`IMPLEMENTATION_INPUTS` names files, not directories, and had outgrown the
`quantark/validation` package: **`adaptive_allocation.py`** — which freezes the
adaptive allocation and holds `precision_stop` — was outside the digest, as was
the new module. Both added, and the coverage test now enumerates the package
from disk so the next module is covered by construction rather than by memory.
