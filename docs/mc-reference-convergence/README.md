# MC Reference-Stack Convergence — Demos & Decision Matrix

Spec: `docs/superpowers/specs/2026-08-10-mc-reference-convergence-design.md`
Plan: `docs/superpowers/plans/2026-08-10-mc-reference-convergence.md`

Rows are measured by `demo_cell.py` using the stage-16 harness's own case,
product, environment, engine, and paired-RQMC estimator (loaded by path, so
the treatment under test is the production estimator with one knob changed).
Raw outputs live in `logs/`. **Numbers here are copied only from completed
runs** — no projections.

## What the rows mean

- `baseline` — the cell's current production profile (bridge dimensions 1),
  the SD and cost anchor.
- `bridge8` — the V1 treatment candidate: 8 leading residual bridge
  coordinates, the profile `near_ki` already uses.
- `unbias` — `bridge8` again on an independent seed, giving the V1-G1
  agreement check against `baseline`.

Gates (spec §4): **V1-G1** |baseline − unbias| ≤ 2σ combined; **V1-G2** SD
factor ≥ 4×; **V1-G3** cost recorded so the decision uses SE²·seconds, not SD
alone.

> **V1-G2 was mis-specified.** It reads "SD-reduction factor ≥ 4×", but the
> 33× cited for `near_ki` in the recovered analysis is a *variance* ratio (SD
> 1.08 → 0.19, an SD factor of 5.7), and it came from the full multilevel +
> 8×8 bridge + 16-substep stack — not from bridge dimensions alone. As
> written the gate demanded 16× variance from one knob. The decision-relevant
> quantity is SE²·seconds, which every row below reports.

## Measured — 32 batches/row, `heston_slv`, 2026-08-10

| Cell | Row | batch SD (c) | sec/batch | peak RSS | SD factor | variance | SE²·sec | V1-G1 |
|---|---|---|---|---|---|---|---|---|
| `ordinary_full` | baseline | 1.0673 | 34.1 | 4.65 | — | — | — | — |
| | bridge8 | 0.7364 | 33.4 | 4.67 | 1.45× | 2.10× | **2.14×** | — |
| | unbias | 0.7683 | 29.4 | 4.68 | — | — | — | 0.64σ ✅ |
| `ordinary_decayed` | baseline | 1.0043 | 17.0 | 2.51 | — | — | — | — |
| | bridge8 | 0.6145 | 17.4 | 2.53 | 1.63× | 2.67× | **2.62×** | — |
| | unbias | 0.5981 | 17.4 | 2.54 | — | — | — | 0.65σ ✅ |
| `sigma_collapse` | baseline | 0.7221 | 34.1 | 4.69 | — | — | — | — |
| | bridge8 | 0.5978 | 33.3 | 4.70 | 1.21× | 1.46× | **1.49×** | — |
| | unbias | 0.6053 | 29.7 | 4.70 | — | — | — | 0.28σ ✅ |

**Fidelity check.** The baseline SDs reproduce the schema-11 recorded per-cell
batch SDs (1.07 / 1.08 / 0.70) to within pilot noise, and `feasibility.py`
reconstructs the recorded aggregate half-width as 0.0546 against 0.0545. The
demos measure the production estimator, not a lookalike.

**Reading.** Every treatment is unbiased and every one is free (cost flat or
slightly lower), but the gains are 1.5–2.6× in SE²·seconds, not the 8×
the spec used for planning. `sigma_collapse` gains least — consistent with its
error being dominated by the PDE-side v-axis truncation the ADI probe pinned
down, not by MC noise.

## Feasibility at the measured gains (`logs/feasibility.log`)

Treatment alone takes the aggregate half-width from 0.0546 to 0.0395 at 128
batches/cell. Reaching the 0.02 target then needs, with all 7 cells running
concurrently (2 workers each, ~33 GiB of 48 GiB):

| Contention assumption | half-width in 12 h | hours to reach 0.02 |
|---|---|---|
| ×1.0 (measured 3-cell rates hold) | 0.0113 | **3.8 h** |
| ×1.5 | 0.0138 | 5.7 h |
| ×2.0 (pessimistic) | 0.0159 | 7.6 h |

**Spec goal G1 holds** even under the pessimistic assumption.

Note on WS-S: with 7 cells and 14 cores the streams do not compete for a
shared budget, so Neyman weighting is close to inert here — each cell fills
its own stream. Neyman earns its place in the memory-constrained and serial
cases; the overnight win comes from precision *stopping* (finish near 4 h
instead of burning 12).

## V2 — cross-fitted control weights: DESIGN INVALIDATED, NOT RUN

The demo paired two separate runs by seed, but stage-17's control is a
three-level telescoping estimator whose control rows are evaluated on the same
paths inside one run. See `demo_v2_weights.py` for the corrected design; it is
post-processing over references the production run already builds, so it costs
no extra Monte Carlo and is deferred to after Phase-1 regeneration.

## User decisions (Task 7 checkpoint, 2026-08-10)

- [x] `ordinary_full` treatment: **bridge8 ships** (2.14× SE²·sec, 0.64σ)
- [x] `ordinary_decayed` treatment: **bridge8 ships** (2.62× SE²·sec, 0.65σ)
- [x] `sigma_collapse` treatment: **bridge8 ships** (1.49× SE²·sec, 0.28σ)
- [x] V1-G2 gate: **replaced** by `SE²·sec ≥ 1.25×` plus the unchanged V1-G1
      unbiasedness check. Rationale: gate the quantity the decision actually
      turns on; the old SD-unit floor was unachievable by any single knob.
- [x] Schema numbering: stage-16 `SCHEMA_VERSION` 11 → **13**, stage-17 stays
      at 12, stage-17 `PARENT_SCHEMA_VERSION` 11 → 13. Smallest blast radius;
      the frozen stage-17 pins and cohort labels are untouched.
- [x] V2 cross-fitted weights: **deferred** to post-regeneration
      post-processing (design invalidated as standalone; corrected design in
      `demo_v2_weights.py` costs no extra Monte Carlo once references exist).

Applied in stage-16 as `SLV_SPOT_BRIDGE_PROFILE_BY_CASE[cell]["dimensions"] = 8`
for the three cells above. `low_feller` is deliberately untouched: its direct
estimator was selected on measured 2026-08-05/06 evidence recorded in the
harness, and `near_ki` already runs the full treatment.

Cells left at `baseline` are not failures — WS-S allocation absorbs them by
sending more batches their way.

## Reproduction

```bash
cd .worktrees/adi-greek-certification
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
    docs/mc-reference-convergence/demo_cell.py --cell ordinary_full
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
    docs/mc-reference-convergence/demo_v2_weights.py
```

`PYTHONPATH` is required: the editable install's compat `.pth` binds
`quantark` at interpreter startup, before any in-script `sys.path` edit.
