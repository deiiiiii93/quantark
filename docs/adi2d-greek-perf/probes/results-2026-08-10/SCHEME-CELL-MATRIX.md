# Which v_drift_scheme should P1.4 certify? — measured 2026-08-10

Run: `probe_scheme_cell_matrix.py` (42 solves, target grids, backends `c`/`numba`)
and `probe_variance_operator_fallback.py`. Raw: `output/scheme_cell_matrix/`.

## The question

The plan of record contains a circular dependency. D-0 says fix the v-axis scheme
*before* regenerating MC references, because certifying a knowingly first-order
PDE would land INCONCLUSIVE again. C-G6 defers the scheme default change to
*after* P1.4. And `test_adi_greek_certification.py` pins stage-16's controls dict
equal to stage-11's and stage-12's — so there is no way to certify one scheme and
ship another. The three statements cannot all hold; this probe breaks the tie with
data instead of sequencing.

## Result: exactly one cell disagrees, and it is the one we predicted

Delta gap versus `adaptive_upwind`, in futures contracts. The aggregate signed
bias bound is ±0.10; the per-cell bound is ±0.5.

| cell | upwind delta | SL − upwind | centered − upwind | SL cost |
|---|---|---|---|---|
| heston/near_expiry | −20.0799 | −0.0012 | 0.0000 | 1.25× |
| heston/near_ko | +5.6578 | −0.0026 | 0.0000 | 1.23× |
| heston/near_ki | +113.3878 | +0.0008 | 0.0000 | 1.17× |
| heston/ordinary_decayed | +32.2921 | +0.0011 | 0.0000 | 1.25× |
| heston/ordinary_full | +11.1365 | −0.0009 | 0.0000 | 1.24× |
| heston/low_feller | −0.7035 | −0.0029 | 0.0000 | 1.21× |
| **heston/sigma_collapse** | +17.7984 | **+0.1149** | **+0.1183** | 1.19× |
| heston_slv/near_expiry | −19.7960 | −0.0012 | 0.0000 | 1.24× |
| heston_slv/near_ko | +6.1818 | −0.0026 | 0.0000 | 1.29× |
| heston_slv/near_ki | +110.1028 | +0.0008 | 0.0000 | 1.21× |
| heston_slv/ordinary_decayed | +32.8645 | +0.0011 | 0.0000 | 1.25× |
| heston_slv/ordinary_full | +11.5158 | −0.0009 | 0.0000 | 1.24× |
| heston_slv/low_feller | −0.1436 | −0.0030 | 0.0000 | 1.31× |
| **heston_slv/sigma_collapse** | +18.6280 | **+0.1119** | **+0.1152** | 1.30× |

`centered` is a diagnostic-only yardstick, not a candidate: `adi_core.py:207`
retains it precisely because it loses monotonicity where drift dominates.

## Why: the fallback engages in one regime only

| cell | interior nodes | donor-cell fallback nodes | max local Péclet |
|---|---|---|---|
| sigma_collapse | 133 | **132** | **10 875.5** |
| low_feller | 133 | **0** | 0.543 |
| all five others | 133 | 2 | 5.028 |

`adaptive_upwind` keeps the second-order centered row wherever it stays an
M-matrix. In five regimes that is 131 of 133 nodes, so upwind *is* centered and
the two agree to every printed digit — there is no upwind bias to remove there.
In `low_feller` it is all 133 nodes. Only `sigma_collapse`, at a local Péclet of
ten thousand, falls back almost everywhere, and that is where the first-order
error lives.

## Two independent measurements agree to 3%

The recorded schema-11 bias for `heston/sigma_collapse` was **−0.112 ± 0.010**
contracts against the banked MC reference. Semi-Lagrangian transport moves that
cell by **+0.1149** — measured here with no MC in the loop at all, and 97.1% of
the way to the centered yardstick (+0.1183). The residual after the fix is inside
the reference's own standard error.

This is the strongest evidence in the program that the σ-collapse bias is a PDE
v-axis artifact and not a reference problem, because the prediction and the
measurement come from disjoint machinery.

## What it does not fix

`low_feller` has **zero** fallback nodes, so no v-axis scheme can touch it, and SL
in fact moves it −0.0029/−0.0030 (marginally worse). Its recorded −0.107/−0.159
therefore remains attributable to the v=0 boundary treatment in a strongly
Feller-violated regime, or to the MC reference side. Consistent with the
2026-08-10 movement probe, where `centered` moved it exactly 0.00000. This stays
a quality item (P1.1b), not an admission blocker.

## Aggregate arithmetic

SLV aggregate estimate was −0.0527, interval [−0.1394, +0.0340]. Applying the
per-cell SL movements above (+0.1119 on sigma_collapse, −0.0058 summed across the
other six, each entering the 7-cell mean at 1/7):

    -0.0527 + 0.1119/7 - 0.0058/7 = -0.0375

At the treated half-width of 0.0395 (feasibility.log, 128 batches) that is
[−0.077, +0.002]; at the 0.02 target it is [−0.058, −0.018]. Both sit inside
±0.10.

Staying at upwind projects to −0.0527 ± 0.0395 = [−0.092, −0.013], which also
sits inside ±0.10. **So the noise treatment alone would probably pass the
aggregate gate at either scheme.** The case for SL is not that it is needed to
pass; it is that admitting upwind means admitting an engine with a measured 11σ
structural error in a production regime, diluted below the bound by a 7-cell
average. The margin difference is 63% versus 47% of bound.

## Operational readiness

Semi-Lagrangian completed all 14 cells at production target grids with no
`NumericalError`, no negativity failure, and no grid refusal. That was the
rehearsal P1.4 needed before hours of MC compute were staked on the scheme.

Cost is 1.17–1.31× on the PDE march (mean 1.24×), against the 3.7× the perf
program just delivered.

## Blast radius of a flip

Three constants, which the pinned invariant requires be changed together:

- `example/mo_volmodels/11_pde_convergence_gate.py:178`
- `example/mo_volmodels/12_snowball_volmodel_backtest.py:122`
- `example/mo_volmodels/16_adi_greek_certification.py:355`

Tests reference the dicts symbolically and follow automatically; no replay golden
references `ADI_2D_PRODUCTION_ENGINE_CONTROLS`. Stage-11 gate and stage-12
backtest *outputs* move — the Phase-3 re-baseline consumes them anyway.
