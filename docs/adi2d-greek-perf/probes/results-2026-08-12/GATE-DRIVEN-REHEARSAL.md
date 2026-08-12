# Gate-driven rehearsal: one real cell, end to end, PASS

**Date** 2026-08-12 · `probes/probe_gate_driven_rehearsal.py` ·
`output/gate_driven_rehearsal/`

`heston/low_feller` through the real `certify_case` with a policy attached
(K=28, floor 128, chunk 128, cap 512, margin 0), 29.2 min:

| greek | decision | gap | sequential total | frozen-form total | verdict |
|---|---|---|---|---|---|
| delta | ADMIT @ 256 | 0.082 | 0.423 / 0.50 | 0.189 | PASS |
| gamma | ADMIT @ 256 | 0.059 | 0.249 / 0.50 | 0.116 | PASS |

Cell status PASS. All five checks pass: stopped before the cap, banked a chunk
multiple, policy digest recorded, both greeks admitted, cell passes.

**256 of a 1024 frozen allocation — 4x on this cell.**

## The conservatism property, visible in production numbers

Sequential total uncertainty 0.423 against the frozen form's 0.189 on the same
banked batches. The anytime-valid width is ~2.2x the fixed-B width here, and
both pass: exactly the claim the unit test pins abstractly, now observed on a
real cell. Early stopping bought wall-clock and paid in width.

## Why it stopped at 256 and the standalone probe said 273

The standalone `probe_gate_driven_cell` did not pass a QE draw provider, so its
target and fine levels were independently seeded. Production builds them through
`coupled_qe_providers`, which shares randomness across the two substep levels --
that is the whole point of the coupling -- so the variance of the target-minus-fine
difference is far smaller. Bias envelope at 256 batches: ~0.15 coupled versus
0.255 uncoupled.

**The probe's low_feller numbers were pessimistic**, and the earlier prediction
that this cell needs ~273 batches was too high for the same reason. Coupling is
worth roughly a 40% reduction in the bias envelope on this cell.

## Chunk granularity

Stops land on chunk multiples, so a cell whose gate closes at 200 pays 256. At
chunk 128 the overshoot is bounded by 127 batches; at chunk 64 it halves for the
cost of extra gate evaluations, which are free next to the MC. Worth tuning once
the fleet's stop distribution is known -- not before.

## Bug this rehearsal caught

Running it end to end is what exposed the threaded-branch defect (`bbd452e`):
`first_batch` was honoured serially and ignored under `batch_workers > 1`, so
every production cell would have recomputed batches [0, n) while reporting them
as [first_batch, first_batch + n). Standard errors would have shrunk on evidence
that did not exist. No unit test on the serial path could have found it.
