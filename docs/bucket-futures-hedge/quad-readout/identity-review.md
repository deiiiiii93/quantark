# Gate D review: the spot bumps were not refined together

Implementation follow-up: [matched audit and historical validation](identity-fix.md).

Measured on 2026-09-10 in `worktree-bucket-futures-hedge`, commit `9af4f47f`.
The dominant 1% residual is finite-difference truncation. The experiment
previously described as refuting truncation changed only the audit spot
bump, leaving the product delta at 1%. It therefore did not test the
predicted quadratic convergence of a consistently refined identity.

The production pricer, hedge settings, saved audit statuses and tolerances
were not changed. These are diagnostic repricings of the same product.

## Exact reproduction and controlled experiment

The historical coupon is held at `0.045276487940179744`; it is not solved
again when changing readout or resolution. The original inception spot is
`6733.969`. The study calendar, time shift, lifecycle state and eligible
futures quotes are rebuilt using the same helpers as the replay.

For 2024-01-19, legacy pricing reproduces the recorded short-book PV of
`8965571.517393284` CNY within `3e-8` CNY, and the residual
`0.548283264208967` hands within `8e-11` hands. The actual grid is **683**
nodes, with log spacing `0.003711601968116531`, and KI at `5050.47675`.

Here both spot differences use the bump in the first column; futures
buckets still use their original +/-1 index point:

| Relative spot bump | Legacy residual, hands | Transition residual, hands |
|---:|---:|---:|
| 0.01 | +0.548283264 | +0.560638806 |
| 0.005 | +0.119140657 | +0.138641085 |
| 0.0025 | +0.018561785 | +0.034530597 |
| 0.001 | -0.005966028 | +0.005479909 |
| 0.0001 | -0.000107683 | +0.000006701 |
| 0.00005 | -0.000063299 | -0.000034757 |

Transition gives essentially a factor of four per halving until the fixed
quote-bump error becomes visible. Its +/-1 quote bucket sum is
`-99.129059570045` hands; at +/-0.5 it is `-99.129096001817`, and at +/-0.1
it is `-99.129107660595`. The quote differences converge too.

Richardson extrapolation of D and DF from spot bumps 0.0001/0.00005, and of
the bucket sum from quote bumps 1/0.5, gives the following decomposition on
the default 683-node transition surface:

| Estimator | At audit bump | Extrapolated local derivative | Difference, hands |
|---|---:|---:|---:|
| D | -91.411717290114 | -91.592292124621 | +0.180574834507 |
| DF | +7.156703473690 | +7.536816021246 | -0.380112547556 |
| Sum of scaled buckets | -99.129059570045 | -99.129108145742 | +0.000048575697 |

Thus `R = 0.180574834507 - (-0.380112547556) - 0.000048575697
= 0.560638806240` hands. The extrapolated chain identity closes to
`1.3e-10` hands. This is a limit of this numerical surface, not an external
validation of its Greeks.

## All live dates, with the original readout and buckets

This sweep reuses the archived +/-1-point bucket samples, recomputes both
spot derivatives with the same smaller bump, and holds the original
coupon, market inputs, lifecycle state and legacy readout fixed. It covers
241 live valuation dates with finite archived residuals. A further live
date, 2024-02-05, was already inconclusive because its 1% pinned spot
scenario exceeded the curve's yield range. The terminal date was not
measured. The original 242 measured statuses therefore included one
inconclusive date, not 242 finite residuals.

| Both spot bumps | Mean absolute residual | Maximum absolute residual | Failures above 0.01 |
|---:|---:|---:|---:|
| 0.01, archived baseline | 0.042500493 | 0.548283264 | 175 / 241 |
| 0.001, repriced | 0.000941613 | 0.011049610 | 1 / 241 |
| 0.00025, repriced | 0.000102608 | 0.002195980 | 0 / 241 |

No tolerance relaxation and no readout change are needed to make this
*matched derivative diagnostic* close on these dates. This does not turn
the saved production audits into passes. It also does not recommend using
a sub-cell legacy delta as the economically accurate hedge delta: the
readout errors in D and DF can cancel in their difference.

Three other pre-KI states (2023-05-04, 2023-09-15, 2023-12-01) show the same
quadratic pattern under transition. The post-KI 2024-03-15 control falls
from `0.001875129` at 1% to `0.000468137` at 0.5%.

## Why the earlier bump experiment was misleading

Write errors relative to the local derivatives of one numerical pricing
function as `e_D(h_D)`, `e_DF(h_F)` and `e_i(b_i)`. Then, in hands,

```
R = [e_D(h_D) - e_DF(h_F) - sum_i (F_i/S) e_i(b_i)] / 200.
```

Only when the relevant steps are refined together is an overall
quadratic reduction expected. In the earlier experiment `h_D` stayed at
1%. On the worst date under transition, reducing only `h_F` makes the
residual tend to approximately **0.18053 hands**, the remaining 1% D
error, after accounting for the quote-bump error. Under the legacy
readout the changing staircase error adds oscillations and can increase
the cross-date mean. Growth of that mixed-step metric is not proof of
price noise divided by a small step.

## Curve and grid mechanism

`carry_context.py:with_spot` pins the quotes and rebuilds implied zero
yields as `q_i = r_i - log(F_i/S)/T_i`. The interpolation knots and quote
universe remain fixed. At the worst date the front nodal yield is
24.193137%; spot +1% moves it to 37.164104%, and spot -1% moves it to
11.091806%. The maturity pricing yield moves from 14.780118% to 19.075698%
and 10.441366%, respectively. These are large carry perturbations for a
test of a local differential identity.

The grid count and width depend on maturity, volatility and KI spacing,
not directly on q. All worst-date scenarios use the same 683 nodes and
spacing. D and DF at the same bumped spot have the same grid geometry;
quote-only bumps leave that geometry fixed. The absolute KI node remains
5050.47675. Spot moves reindex the grid window, but q does not introduce an
extra regridding effect here. No KO is filtered in any worst-date scenario.

There is a conditional q-dependent path through KO reachability filtering
and then alignment selection; it should be instrumented in general. One
December control changes a remote KO's filtering status. The coherent
ladder still exhibits quadratic convergence on that date.

A futures quote bump is a drift/curve perturbation at fixed spot, not a
move of the readout across the spatial lattice. Describing +/-1 quote
point as 0.05 spatial cells does not establish that the bucket estimate
is under-resolved; its own bump ladder is the relevant check.

An actual resolution ladder at the worst date gives:

| Actual nodes | Legacy R at 1% | Transition R at 1% |
|---:|---:|---:|
| 683 | 0.548283264 | 0.560638806 |
| 1365 | 0.556758251 | 0.559531695 |
| 2729 | 0.558768835 | 0.559250521 |

The persistent ~0.56 is a finite-step effect. Spatial refinement does not
make a differential identity exact for a 1% secant.

## Acceptance budget and the separate surface error

Keep numerical identity closure, the desk's choice of a 1% hedge secant,
and external accuracy of the product delta as separate checks. A 0.01-hand
closure target is attainable in this experiment. It was not justified as
an absolute bound for the original mixed finite differences.

For deterministic error bounds in currency delta units, an identity error
budget is `(E_D + E_DF + sum_i abs(F_i/S)*E_Bi)/200`. Estimate each component
with its own bump ladder and genuine spatial refinement. For a central
difference of step a, bounded scenario PV errors propagate as
`(eta_plus + eta_minus)/(2*a)`; the smooth truncation term is
`a*a*abs(directional third derivative)/6`. Do not assume independent
random errors and combine them by root-sum-square without evidence.

An economic hand limit should come from permitted hedge PnL error L over
a specified index move X: `L/(200*abs(X))`. At the worst-date 1% spot move,
0.01 hands is approximately 106 CNY, or 0.0212 bp of this notional. A
fixture tolerance, a rounding increment and a relative delta percentage
are not substitutes for this choice.

A smooth biased value function still obeys the chain rule for its own
derivatives. A near-barrier surface defect can change the finite-step
coefficients, but neither a signed delta lobe nor the pre/post-KI pattern
identifies it as the cause of a differential inconsistency. The tests
above resolve the dominant audit mechanism without resolving the surface
accuracy question.

There is also a unit issue in the repository's lobe probe.
`near_barrier_vs_pde.py` uses inception spot **100** and scales delta by
`50_000_000/100/200`. This is hands for a 100-point index. To compare with
CSI1000/IM at inception 6733.969 and multiplier 200, multiply those results
by `100/6733.969`. Repricing the same fixture with all price levels scaled
to 6733.969 and multiplier `50_000_000/6733.969` gives QUAD minus PDE gaps
of **+0.183622, -0.372982, -0.023649 hands** at -1.5%, +0.84%, +5% from KI.
The lobe remains real, but the quoted 12.4/25.1/1.6 figures are not in
comparable IM-hand units. This correction applies to the repository
fixture; any separate external reference needs its own unit check.

**Subsequent reference correction:** commit `fd1efc03` records that this
branch's PDE predates its near-barrier delta repair `25d4f7d6` (the latter
is not an ancestor). The conversion above is valid, but the corrected-unit
QUAD-minus-PDE figures compare against that pre-fix PDE and cannot establish
QUAD's error against truth. A fresh reference comparison is required. None
of the QUAD-versus-QUAD identity measurements above uses the PDE.

## Reproduce

From the named worktree (Python environment is in the main checkout):

```sh
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/gate-d-mpl \
  /Users/fuxinyao/quant-ark/.venv/bin/python \
  docs/bucket-futures-hedge/quad-readout/identity_review.py \
  --history-dir /Users/fuxinyao/quant-ark/example/mo_volmodels/data/history \
  --grids 683 1365 2729
```

The script emits signed bumped PVs, actual grid metadata, both spot
derivatives, separate quote-bump ladders, and the result of keeping D at 1%.
`identity_review_results.jsonl` contains the measured worst-date grid ladder.

For the all-date legacy experiment:

```sh
PYTHONPATH=. MPLCONFIGDIR=/private/tmp/gate-d-mpl \
  /Users/fuxinyao/quant-ark/.venv/bin/python \
  docs/bucket-futures-hedge/quad-readout/identity_review.py \
  --history-dir /Users/fuxinyao/quant-ark/example/mo_volmodels/data/history \
  --readouts legacy_linear --reuse-recorded-buckets --all-live-dates \
  --spot-bumps .001 .00025
```

`identity_review_live_dates.jsonl` contains the per-date evidence. This
mode also verifies that current base PV reproduces the archived book to
0.01 CNY before using its saved buckets. All files in this directory are
locally excluded by the repository's `/docs/` Git exclusion.
