# low_feller closes under the gate-driven loop — the "hard bias" was a 32-batch artifact

**Date** 2026-08-12 · Probe `probes/probe_gate_driven_cell.py` · Raw
`output/gate_driven_cell/`

## Result

`heston/low_feller`, scanned with the shipped anytime-valid policy
(K = 28 tests, family alpha 0.05, floor 128, cap 512):

| greek | stop | gap | w_greek | pde env | bias env | total / bound |
|---|---|---|---|---|---|---|
| delta | **273 batches** | 0.060 | 0.172 | 0.012 | 0.255 | **0.499 / 0.50** |
| gamma | **128 batches** (the floor) | 0.012 | 0.121 | 0.015 | 0.187 | 0.334 / 0.50 |

Both close well inside the frozen 1024. No scoping out, no documented exception.

## The belief this corrects

`low_feller` has been carried through this program as the cell whose bias no
scheme work can fix: it has zero non-monotone variance rows, so
`v_drift_scheme="auto"` correctly leaves it on `adaptive_upwind`, and its
recorded −0.107 / −0.159 gap was treated as a hard floor. On that basis the plan
was to expect an INCONCLUSIVE and decide in advance whether to scope it out.

Measured, the cell is **noise-limited, not bias-limited**:

| term | delta @ 256 | reading |
|---|---|---|
| \|gap\| | 0.076 (was 0.157 at 32 batches) | mostly MC noise in a 32-batch gap |
| \|substep mean\| | **0.015** | the only irreducible term — 3% of the bound |
| w_substep | 0.240 | a *half-width*; shrinks |
| w_greek | 0.178 | shrinks |
| **irreducible floor** | **0.103** | vs 0.397 of room |

What looked like a fat 0.255 bias envelope is 94% half-width. The gap itself fell
by half between 32 and 256 batches. Almost nothing about this cell was actually
bias.

## Projection quality, for calibration

The 32-batch projection said delta needs ~84 batches; it measured 273. The
factor of 3.3 decomposes cleanly: anytime-valid widths are ~1.3x wider than
fixed-B (1.3² = 1.7x the batches, by construction — that conservatism is what
licenses stopping at a data-dependent time), and the two-component rule charges
the substep half-width as well. The projection was right about *feasibility* and
wrong about *cost*, which is the expected failure mode and the reason this was
measured rather than shipped on the projection.

Prefix invariance was exercised end to end: the 512-run's scan reproduces the
256-run's non-closure at t = 256 before closing at 273.

## One property of stopping rules worth naming

Delta closes at 0.499 against a 0.50 bound — a margin of 0.001. That is not a
warning sign, it is the definition of stopping at the first crossing, and the
anytime-valid guarantee already covers optional stopping. But a published
certificate reading "PASS with 0.2% margin" invites questions the statistics do
not, so the policy should probably carry a **margin fraction** — stop when
`total <= (1 - m) * bound` — trading a little compute for a legible buffer. One
parameter, declared and hashed with the rest.

## Still to measure

This is one cell. `heston_slv/low_feller` and the three SLV gamma cells with
genuinely larger substep means (0.145 - 0.183) have not been run, and SLV is
roughly 10x slower per batch. Given how the 32-batch projections have performed
today, those should come out of the real gate-driven run rather than another
standalone projection.
