# Bucket futures hedge — acceptance gate record

Implementation of the revised design
(`docs/superpowers/specs/2026-09-09-bucket-futures-hedge-design-revised.md`),
Tasks 1–16 of the implementation plan. This file records what was run, what
it produced, and what it did not establish.

Branch: `worktree-bucket-futures-hedge`, based on `feat/simulated-path-backtest`.

All commands assume the repository root and the shadowing form used in a
worktree: `PYTHONPATH=. <repo>/.venv/bin/python`.

---

## Gate A — risk math

Independent reprices, all three residual formulas, node/parallel/spot
ladders, quantity/sign/multiplier cases.

```sh
.venv/bin/python -m pytest -n0 -q test/test_futures_risk.py \
  test/test_carry_curve_context.py test/test_futures_carry_risk.py \
  test/test_futures_carry_audit.py test/test_futures_bucket_strategy.py \
  test/test_futures_carry_stress.py test/test_futures_delta_strategy.py
```

**Result: 253 passed.**

What it establishes:

- both supported builders satisfy the two invariances of design section 2.2
  on the first interval, an interior interval, the nodes, the tail and the
  one-node limit, checked against closed forms in `_bucket_oracles.py` that
  import nothing from `quantark`;
- the three policies produce the design's exact holdings on the section 3.3
  fixture, and their residuals are confirmed by INDEPENDENT repricing rather
  than by the algebra that sized them;
- a bucket perturbed by 0.05 still cancels to zero in `held_book_risk` and
  is caught by both the direct nodal measurement and the identity residual;
- re-sizing inside a scenario manufactures a pass for a book holding
  nothing, which is why the audit holds quantities fixed;
- the tail stress moves a 1Y claim by −49.8752 bp with exactly zero hedge
  P&L, and the interpolation-shape stress moves a price while every quote it
  is built from is unchanged.

Two numerical findings recorded rather than smoothed over:

- the far-tail bucket misses its analytic value by 1.3e-7 relative. That is
  the O(h²) truncation of a one-point central difference against a log
  elasticity of 3.6, predicted exactly; a ladder test pins the error
  quartering as the bump halves.
- a one-index-point bump on a contract three days from expiry implies a
  121% yield, which the dividend curve refuses rather than clipping. This is
  why the production minimum tenor is seven days.

## Gate B — accounting and compatibility

Exact local ledger comparison, old goldens, atomic failure and lifecycle
close.

```sh
.venv/bin/python -m pytest -n0 -q test/test_replay_futures_buckets.py \
  test/test_futures_hedge_book.py test/test_replay_goldens.py \
  test/test_replay_engine_unified.py test/test_replay_termination.py \
  test/test_pnlexplain_backtest_replay.py test/test_replay_greeks_failclosed.py
```

**Result: 122 passed, 4 skipped.**

**Post-merge 2026-09-11: 3 failed, 119 passed, 4 skipped.** The three are the
inherited replay goldens; see "Post-merge gate status" under Gate D. The
bullet below saying the historical goldens pass unchanged was true when
written and is FALSE after the merge, because the base's PDE barrier-readout
repair moves the very deltas those goldens freeze.

- a nine-step sequence of non-integer trades recorded against
  `FuturesHedgePosition` BEFORE the accounting transition was extracted is
  reproduced with exact float equality plus two `float.hex()` bit patterns;
- the historical goldens pass unchanged, and no golden tolerance was
  touched or regenerated;
- a cost model that raises on the second leg leaves no first-leg trade, no
  cost and no ledger change;
- a retired leg closes inside a band wide enough to suppress every ordinary
  rebalance; the automatic correction pair moves with eligibility.

One regression the goldens caught: `current_contracts` in the rebalance
frame has always recorded the POST-trade quantity while the decision uses
the pre-trade one. The quirk is preserved and now documented.

## Gate C — end-to-end risk output

Both result APIs, genuine direct audits, finite stress residuals.

```sh
.venv/bin/python -m pytest -n0 -q test/test_futures_carry_recorder.py \
  test/test_replay_futures_buckets.py test/test_futures_bucket_results.py \
  test/test_replay_schema.py test/test_replay_termination.py \
  test/test_replay_goldens.py
```

**Result: 123 passed, 4 skipped.**

**Post-merge 2026-09-11: 3 failed, 125 passed, 4 skipped.** The same three
inherited goldens. Separately, the 0.0094 reference hands recorded in the
second finding below was measured on the PRE-repair PDE and has not been
re-measured since `25d4f7d6`; treat the number as dated, though the point it
makes, resolution rather than tolerance, is unaffected.

- every eligible node gets a leg row even at zero holdings, so a
  single-contract control cannot hide the rhoq at the tenors it does not
  hedge;
- unmeasured direct fields are NaN with a status, never a zero;
- a node-set change withholds the linear decomposition with a reason and
  keeps the day's actual P&L and costs;
- a perturbed bucket in a replay that completed with finite P&L still fails
  the audit.

Two findings:

- the legacy roll window (five days) is looser than the curve's minimum
  tenor (seven), so a single-contract control can hold a contract that is
  not a risk coordinate. Its scenario P&L would be dropped from both sides
  of the audit, so such a day reports `inconclusive`; the leg's own delta
  and carry sensitivity still appear in the net.
- the audit is sharp enough to see the pricing grid. On the standard PDE
  grid an engine's own delta and a central difference of its own price
  differ by about 0.0094 reference hands. The gap does NOT shrink as the
  audit bump halves — so it is not truncation — and roughly halves on the
  finer grid. Resolution, not tolerance.

## Gate D — subset and runtime

### Required regression

```sh
.venv/bin/python -m pytest -n0 -q test/test_futures_risk.py \
  test/test_carry_curve_context.py test/test_futures_carry_risk.py \
  test/test_futures_carry_audit.py test/test_futures_carry_stress.py \
  test/test_futures_bucket_strategy.py test/test_futures_hedge_book.py \
  test/test_futures_bucket_config.py test/test_futures_bucket_results.py \
  test/test_futures_carry_recorder.py test/test_replay_futures_buckets.py \
  test/test_bucket_hedge_study.py test/test_bucket_hedge_analysis.py \
  test/test_bucket_hedge_validation_cli.py test/test_futures_ledger.py \
  test/test_futures_delta_strategy.py test/test_replay_dividend_source.py \
  test/test_replay_env_helpers.py test/test_replay_greeks_failclosed.py \
  test/test_replay_schema.py test/test_replay_goldens.py \
  test/test_replay_engine_unified.py test/test_replay_termination.py \
  test/test_pnlexplain_backtest_replay.py \
  test/test_snowball_q_term_structure_study.py test/simulation/test_config.py
```

**Result: 636 passed, 4 skipped, 131 s.** No golden was updated.

Re-run on 2026-09-10 with the matched identity, adding
`test/test_futures_carry_identity_quad.py` to the list above:
**658 passed, 4 skipped, 178 s.** Still no golden updated. The two warnings
are the pre-existing `quantark.backtest.otc` import aliases.

### Post-merge gate status

After merging the base on 2026-09-11, all four recorded suites were re-run.
The ONLY change anywhere is the three inherited replay goldens:

| Suite | before the merge | after |
|---|---|---|
| Gate A | 253 passed, then 262 with the new tests | **262 passed** |
| Gate B | 122 passed, 4 skipped | 3 failed, **119 passed**, 4 skipped |
| Gate C | 123 passed, 4 skipped | 3 failed, **125 passed**, 4 skipped |
| Gate D | 658 passed, 4 skipped | 3 failed, **655 passed**, 4 skipped |

655 plus 3 is 658. Every test that passed before still passes; exactly the
three goldens moved, and they are the same three in all four suites because
all four include `test_replay_goldens.py`. Gate A does not, and is green.

Both validations still pass post-merge: **7/7 synthetic** and **7/7 on the
real 2025-03-03 chain**, 0 failed and 0 inconclusive in each.

The subset runs predate the merge and are NOT invalidated by it: the merge
changed no QUAD, carry, futures or replay source file, and the frozen
identity regression still reproduces 0.548283.

### Synthetic validation

```sh
.venv/bin/python example/snowball_q_term_structure/05_bucket_hedge_validation.py \
  --synthetic --out-dir OUT
```

**Result: 7/7 required cases passed, exit 0, about 2 s, no vendor history.**
Re-run under the matched identity on 2026-09-10: **7/7, 0 failed, 0
inconclusive, exit 0.**

### Historical validation

```sh
.venv/bin/python example/snowball_q_term_structure/05_bucket_hedge_validation.py \
  --historical-dates 2025-03-03 \
  --out-dir example/snowball_q_term_structure/data/bucket_hedge_v2/validation
```

**Result: 7/7 required cases passed on the real chain.** Re-run under the
matched identity on 2026-09-10: **7/7, 0 failed, 0 inconclusive**, on the
same 2025-03-03 chain.

Spot 6273.67 with four listed IM contracts at 0.049 / 0.126 / 0.299 / 0.548
years, in steep backwardation. Under both conventions:

| Objective | Direct net delta (hands) | Audit error (hands) | Parallel error (bp) |
|---|---:|---:|---:|
| `nodes` | 39.85 | 5e-12 | 6e-10 |
| `spot_far` | 0 | 5e-12 | 2e-09 |
| `spot_parallel` | 0 | 5e-12 | 3e-15 |

The `nodes` residual of 39.85 reference hands IS `D_F`, measured on a real
curve: direct evidence that the pinned-futures spot delta is not a tail
artefact and not negligible.

Tail stress −54.64 bp of PV with exactly zero hedge P&L; shape stress
+100.50 bp on a claim at the peak of the displacement, also with zero hedge
P&L, while every listed quote was unchanged.

Artifacts: `validation_manifest.json`, `input_snapshots.json`,
`price_ladder.csv`, `greek_ladder.csv`, `policy_holdings.csv`,
`direct_audits.csv`, `stress_results.csv`, `validation_summary.md`.

### Subset benchmark

```sh
.venv/bin/python example/snowball_q_term_structure/02_backtest_fleet.py \
  --study-grid buckets --max-inceptions 1 --workers 2 \
  --carry-audit-mode daily --record-carry-exposure \
  --out-dir example/snowball_q_term_structure/data/bucket_hedge_v2/subset --resume
```

**Result: 14 cells ran, 14 ok, 0 failed, 6059 s wall clock on 2 workers.**

Inception 2023-05-04, 243 trading days, every cell terminating `ki_maturity`.

| Cell type | Seconds per cell |
|---|---:|
| single-contract control, recording on | 743–775 |
| bucket policy, daily audits | 853–1093 |

Extrapolation to the full primary grid: 14 cells x 29 eligible inceptions is
406 cells at roughly 800 s, about 90 CPU-hours, near 23 hours of wall clock
on four workers. Per-cell storage is about 2 MB of frames, so the full grid
is under a gigabyte.

Measured cost by stage, from `carry_cost()` on a seven-day synthetic run
with three nodes:

| Stage | Price calls |
|---|---:|
| bucket sampling | 1 base + 2 per node, per day |
| direct audit | 14 per day on three nodes (12 on two) |
| finite stresses | 2 per scenario per holdings kind |

This confirms the design's section 10.3 estimate: four nodes add about 20
pricing calls per day at one audit level, counting the eight bucket prices.

#### Audit completeness: the audits did NOT pass

*As of 2026-09-10 these 175 failures are resolved and re-attributed to
1%-secant truncation. This subsection and the three after it are the record
of how they were investigated, including two wrong conclusions. Skip to "The
truncation hypothesis was right" for the outcome.*

Every cell recorded 242 measured audit dates out of 243, and **175 of them
report `fail`**. This is the same figure in all fourteen cells, including
the plain single-contract controls, which is what identifies it as
systematic rather than a property of any hedge policy.

Only ONE of the four checks breaches:

| Check | Dates over the 0.01 budget |
|---|---:|
| net delta audit error | 0 |
| parallel rhoq audit error | 0 |
| nodal rhoq audit error | 0 (of 655 leg rows) |
| **identity residual** | **175** |

The identity residual is `(D - D_F_direct - sum_i (F_i/S) B_i) / m_ref`. It
is the one quantity that mixes the ENGINE'S OWN Greek `D` with repriced
quantities; every check that compares two repriced numbers agrees to about
1e-13. So the disagreement is between the QUAD engine's delta and a central
difference of the QUAD engine's own price.

`audit_failure_attribution.py` now splits the four checks over all 27
completed cells, both readouts: every failure is the identity ALONE, and the
three holdings-dependent checks breach on zero dates out of roughly 6,500
date-checks each. The identity is also the only check with no holdings in
it, which is why it is byte-identical across all seven policies of a
q-model — 0.042500 hands for every `term_flat_q` cell, 0.041171 for every
`term_flat_fwd` cell, and 0.042613 / 0.041276 respectively under
`transition`. So the audit's failure is a statement about the engine that
the bucket hedge merely reports, and a single cell is fully representative
of it. Every check that does test the hedge passes on every date.

It is strongly state-dependent:

| State | pass | fail |
|---|---:|---:|
| before knock-in | 3 | 175 |
| after knock-in | 63 | 0 |

and the worst dates are those approaching the KI barrier from above
(2024-01-19: spot 5306.99 against a 5050.48 barrier, residual 0.548 hands;
passing dates average 0.0004 hands, three orders of magnitude smaller).

`direct_net_delta` agrees with `mapped_net_delta` to 1e-13 on every date.
That agreement looks like independent confirmation of the engine's Greek. It
is NOT, and reading it that way was a mistake made once in this document's
history: at the default settings the audit's spot bump is RESOLVED FROM the
effective pricing bump, so both sides of that comparison use the same bump
and it is one estimator computed twice. Decoupling them, below, shows they
do not agree.

#### Convergence analysis

The first hypothesis was quadrature resolution around the discretely
monitored barrier. Tripling the grid changed nothing:

| `--quad-grid` | fail | pass | mean abs identity | max abs identity |
|---:|---:|---:|---:|---:|
| 401 | 175 | 66 | 0.042500 | 0.548283 |
| 801 | 175 | 66 | 0.042463 | 0.559124 |
| 1201 | 175 | 66 | 0.042456 | 0.556560 |

**That ladder measured nothing, and reading it as a refutation was a
mistake.** `SnowballQuadEngine._resolve_grid_points` returns
`max(requested, required)`, where `required` comes from
`min_diffusion_stddev_cells` (default 2.5) and the daily KI spacing. On the
long-dated states — which are exactly the failing ones — the adaptive floor
dominates and all three requests resolve to the SAME grid:

| ttm | grid actually used at `--quad-grid` 401 / 801 / 1201 |
|---:|---|
| 1.00 | 1581 / 1581 / 1581 |
| 0.75 | 2593 / 2593 / 2593 |
| 0.60 | 1117 / 1117 / **1201** |
| 0.25 | 635 / **801** / **1201** |

The flat rows are flat because nothing moved. `quad-readout/probe_grid.py`
reproduces the table. The grid hypothesis was never tested; it turns out to
be beside the point anyway, for the reason recorded below.

The remaining candidate is the audit's own SPOT BUMP. The identity
`D = D_F + sum_i (F_i/S) B_i` is a first-order statement, and the three
directions in it carry different second-order terms. The audit inherits its
spot bump from the effective pricing bump, which defaults to 1% — about 53
index points — and it is taken beside a knock-in barrier where gamma is
large. That would explain both the clustering before knock-in and the
indifference to the grid.

Quantitatively: each direction's central difference carries an error of
`(h^2/6) * d3V/dS3` in its own direction, so the identity residual should be
about `(h_S^2 / 6) * [d3V/dS3 frozen - d3V/dS3 pinned]`. That predicts the
residual scaling with `h_S^2`: a 4x smaller bump should shrink it about 16x,
a 10x smaller bump about 100x.

**This hypothesis was refuted too** — and the refutation is now itself
retracted. See "The truncation hypothesis was right" below: the ladder
shrank only ONE of the two steps, so it never tested the prediction it is
about to reject. Read the table as a record of the mistake.

Shrinking the bump makes it WORSE:

| audit spot bump | fail | pass | mean abs identity | max abs identity | net delta audit error, mean |
|---:|---:|---:|---:|---:|---:|
| 0.01 (default) | 175 | 66 | 0.042500 | 0.548283 | 4.2e-14 |
| 0.0025 | 155 | 86 | 0.051449 | 0.477425 | 5.0e-02 |
| 0.001 | 213 | 29 | 0.084353 | 1.337295 | 8.4e-02 |

Predicted 16x and 100x reductions; observed 0.83x and 0.61x, i.e. increases.
I concluded that a residual which GROWS as the step shrinks is noise divided
by a small step, not truncation. That inference is invalid: the quadratic
prediction applies to the residual of a CONSISTENTLY refined identity, and
this ladder moved `h_F` while `h_D` stayed at 1%. Unmatching the two steps
exposes a one-sided readout error that the matched difference cancels, which
is what grows here.

The last column is the decisive measurement. At the default the audit's spot
bump is resolved from the pricing bump, so both are 1% and the "agreement"
is trivial. The moment they are decoupled, the QUAD engine's own delta and a
central difference of the QUAD engine's own price differ by 0.05 to 0.08
reference hands. `delta_f_direct` moves with the bump too — 5.0842, 5.0867,
5.1209 — while `delta_f_derived` is pinned at 5.0968 by the engine's Greek.

#### Root cause: the price is a linear interpolant on a lattice that does not move with spot

Full evidence and the candidate comparison are in `quad-readout/`.

`SnowballQuadEngine._price_once` returns
`math_utils.interpolate(value_surface, x=0.0)`
(`snowball_quad_engine.py:463`), and `QuadratureMath.interpolate`
(`quad_math.py:243`) is `np.interp` — linear.

That would cost nothing if the grid moved with spot. It does not. Barrier
alignment snaps a node onto the closest barrier, which translates the
log-moneyness lattice with spot in exactly the way that leaves the
ABSOLUTE-PRICE lattice standing still: node `k` sits at `B * exp(k*h)`
whatever the spot is. Three predictions follow, all measured at the worst
date:

| Prediction | Measured |
|---|---|
| the absolute price nodes do not move with spot | max drift 9.1e-13 index points over a 1.8-point spot move |
| the kinks in `V(S)` sit on the nodes | largest kink at 5319.8700, nearest node 5319.8741 — 0.0002 cells |
| the price IS the linear interpolant of the node values | max relative difference 2.0e-14 over 97 probes |

The third is not a resemblance. At machine precision `V(S)` is piecewise
linear in `log S`, so **delta is a staircase**: flat across a cell, stepping
at every node. At the worst date the cell is 19.7 index points wide (0.371%
of spot) and the risers are about 0.25 reference hands.

A finite difference narrower than a cell returns the chord slope, which is
the true delta at the cell MIDPOINT; its error is `gamma * (S_mid - S)`, a
sawtooth that vanishes at midpoints and is worst at the nodes, with a
peak-to-peak size of `h * |S*Gamma + Delta| / m_ref` set in the LOG
coordinate the interpolation actually uses. Swept across two cells at a 0.05% bump it runs from
-0.087 to +0.158 hands, mean absolute 0.054 — the scale the audit reports.

This accounts for all three observations at once: no improvement from more
quadrature nodes (the grid was never the variable), worse as the bump
shrinks (a narrower bump is likelier to sit inside one cell and lose the
curvature entirely), and confined to the pre-knock-in dates (after knock-in
the KI barrier is gone and the surface near the money is far smoother).

#### It does not reach the hedge

The replay sizes the hedge from `calculate_greeks`, a 1% bump-and-reprice.
That is 5.4 cells wide, and it averages the staircase away:

| | contracts |
|---|---:|
| position | 90.79 |
| sawtooth in the 1% delta, peak to peak | 0.0275 |
| worst deviation from the grid delta | 0.1446 |
| rounding granularity of a hedge trade | 1.0000 |

The worst deviation is seven times smaller than the smallest tradeable
increment, and hedge quantities round to whole contracts. The audit saw the
staircase only because decoupling its spot bump took it below one cell —
finer than the engine's own price grid.

**Gate D was NOT met at this point**, and the tolerance was not touched. The
failure was attributed rather than open: on the reading current here, a
property of the engine's readout, measured, and shown not to affect the
hedge. That attribution was wrong, and the section below replaces it.

---

## The truncation hypothesis was right, and the audit now refines both steps

Everything above this line is the record as it stood before 2026-09-10. The
identity failure is resolved; the retracted sections are kept because the
readings in them were mine and every one was reached from real measurements.

The residual is
`R = [e_D(h_D) - e_DF(h_F) - sum_i (F_i/S) e_i(b_i)] / m_ref`, where each `e`
is that estimator's error against the local derivative of the SAME pricing
function. A quadratic collapse is predicted only when the steps that appear
in it are refined TOGETHER. They never were:

- `delta_q` is the engine's own delta Greek at the engine's own `BumpConfig`
  bump (`equity/engine/base_engine.py:243`), effectively 1%, and is exactly invariant to
  every audit setting;
- only `delta_f_direct` followed `audit_spot_bump_rel`.

At the 1% default the two coincided, so the check looked matched and the
sawtooth cancelled. Every ladder that tried to refine it moved one side
alone. That is why refinement made the residual worse, and why the growth
was misread twice: first as price noise, then as a knock-in surface defect.

Repricing BOTH spot directions at a matched, refined step closes it. On the
worst date, 2024-01-19, with buckets held at their original +/-1 point:

| Matched spot bump | Legacy R, hands | Transition R, hands | Transition ratio |
|---:|---:|---:|---:|
| 0.01 | +0.548283264 | +0.560638806 | — |
| 0.005 | +0.119140657 | +0.138641085 | 4.04 |
| 0.0025 | +0.018561785 | +0.034530597 | 4.02 |
| 0.001 | -0.005966028 | +0.005479909 | 6.30 |
| 0.0001 | -0.000107683 | +0.000006701 | — |
| 0.00005 | -0.000063298 | -0.000034756 | — |

The ratio column is each row against the one above it, and it is `h^2` to
two digits under `transition`. The two halvings predict 4 and give 4.04 and
4.02. The 0.0025 to 0.001 step is a factor of 2.5, so it predicts 6.25 and
gives 6.30. That is the convergence the earlier ladder was built to test and
never did. Under `legacy_linear` it converges too but not monotonically,
because the staircase adds a non-smooth component of its own.

Below about `0.0001` both readouts stall against a floor. It is the FIXED
+/-1 point quote bump in the bucket sum, not the spot steps: Richardson the
two spot derivatives alone and the residual settles at -4.850e-05 hands
under legacy and -4.858e-05 under transition, agreeing to three digits
precisely because it is a quote-bump error and the readout has nothing to do
with it. Extrapolate the bucket sum as well, from +/-1 and +/-0.5 point
quote bumps, and the chain identity closes:

| Fully extrapolated chain residual | hands |
|---|---:|
| legacy_linear | +1.6e-10 |
| transition | +2.0e-10 |

So the chain rule, the bucket vector and the engine's own delta are mutually
consistent, and the 0.548 hands was the truncation of a 1% secant taken
where the third derivative is large. The production ladder,
`(0.001, 0.0005, 0.00025)`, sits above that quote-bump floor by design.

### What the pre/post-knock-in split did and did not show

The 71x collapse at knock-in is real and reproduces. It localises the
residual, and I read that as identifying the live KI barrier as the cause.
It does not: a live daily-monitored knock-in barrier is also exactly where
`d3V/dS3` is largest, so a 1% secant truncates most there. The split is
consistent with BOTH hypotheses and discriminates between neither. The
matched ladder does discriminate, because truncation is the only one of the
two that obeys `h^2`.

The corollary matters for the near-barrier work: **the identity residual was
never evidence for the delta lobe.** They are not one defect after all. The
lobe stands or falls on its own bump-free comparison against a validated
reference, which this branch still cannot run.

### The implementation

`audit_held_book` now recomputes both spot derivatives at each of
`identity_spot_bumps_rel = (0.001, 0.0005, 0.00025)`, four price calls per
level, twelve per audited date. The supplied bucket vector is RETAINED, so a
wrong bucket still fails rather than being replaced by a freshly sampled
correct one. Let `R` be the finest matched residual and `E` the absolute
change from the previous level. The identity passes only when
`abs(R) + E <= 0.01`, fails when `abs(R) - E > 0.01`, and is otherwise
inconclusive. `E` is an observed refinement allowance, not an error bound.

The bumps are configurable through `CarryRiskSettings` and
`--identity-spot-bumps-rel`, and they enter the study fingerprint.

Nothing about pricing, hedge sizing, the 1% pricing bump or the tolerance
changed. Three diagnostics are added and reported, none of them gated:
`finite_bump_identity_residual_hands` reproduces the original quantity,
`pricing_delta_local_gap_hands` is the reported hedge delta minus the finest
local delta, and `identity_ladder` carries every level.

### Historical validation against the archived books

`verify_identity_fix.py` runs the production `audit_held_book` over every
archived live state of both front controls, using the actual held quantities
and the saved pricing Greeks. Product, coupon and market state are rebuilt
first, and every base PV matches the archive before any saved Greek is
reused. No hedge is replayed or changed.

| Curve | Live dates | Overall passes | Max abs R | Max abs(R)+E | Max base PV error |
|---|---:|---:|---:|---:|---:|
| Flat zero q | 242 | 242 | 0.002196 | 0.004340 | 1.6e-6 CNY |
| Flat forward carry | 242 | 242 | 0.001994 | 0.003947 | 1.5e-6 CNY |

Net-delta reproduction stayed below 4.9e-11 hands and every nodal and
parallel rho-q error below 0.000097 bp. On 2024-02-05 the flat-q local audit
now passes while its 1% pinned scenario still records `inconclusive:
dividend yield magnitude must be <= 1.0`, which accounts for the one
archived date that had no finite residual.

The check keeps its teeth on that same real state rather than only on the
linear fixture. Perturbing the front bucket by 0.02 hands, twice the budget,
fails the identity; a reported hedge delta wrong by 0.05 hands fails the
separate net-delta reproduction check.

Evidence: `quad-readout/identity-fix.md`, `identity-review.md`,
`identity_fix_flat_q.csv`, `identity_fix_flat_forward.csv`, and the frozen
regression `test/test_futures_carry_identity_quad.py`, which needs no market
cache and reproduces the original 0.548283-hand residual under both readouts.

### The subset, re-run

```sh
.venv/bin/python example/snowball_q_term_structure/02_backtest_fleet.py \
  --study-grid buckets --max-inceptions 1 --workers 6 \
  --carry-audit-mode daily --record-carry-exposure \
  --out-dir example/snowball_q_term_structure/data/bucket_hedge_v2/subset_matched \
  --resume
```

A NEW output directory on purpose. `identity_spot_bumps_rel` is in
`RISK_FINGERPRINT_KEYS`, so no banked cell resumes and `--resume` into the
original `subset` would have re-run all fourteen in place — destroying the
archive that `verify_identity_fix.py` reads. The pre-fix run stays where it
is.

**Result: 14 cells ran, 14 ok, 0 failed.** Every cell now reports
`all_measured_passed: true`, against `false` in every pre-fix cell.

| | pass | fail | not measured | inconclusive |
|---|---:|---:|---:|---:|
| pre-fix, `term_flat_q` cells | 66 | 175 | 1 | 1 |
| pre-fix, `term_flat_fwd` cells | 66 | 176 | 1 | 0 |
| post-fix, every cell | 242 | 0 | 1 | 0 |

The one remaining `not_measured` date is the terminal date, as before.

Leg rows follow, and their totals differ between policies because a bucket
book carries more eligible node rows than a single-contract control. What is
uniform is the verdict: **no leg row fails or is inconclusive in any cell**,
against 655 to 664 failures per cell before, and four inconclusive rows in
seven of the fourteen.

The identity numbers reproduce the offline replay to six decimals, which is
worth stating because those are separate code paths — the recorder inside
the replay loop, and `verify_identity_fix.py` rebuilding archived states:

| Over 242 dates | max abs R | mean abs R | max abs(R)+E |
|---|---:|---:|---:|
| every `term_flat_q` cell | 0.002196 | 0.000102 | 0.004340 |
| every `term_flat_fwd` cell | 0.001994 | 0.000090 | 0.003947 |

Both agree with `identity_fix_flat_q.csv` and `identity_fix_flat_forward.csv`
to every digit printed here.

It is byte-identical across all seven policies of a q-model, as it was
before. That has to hold: the identity is the only check with no holdings in
it.

#### The re-run changed nothing that is priced or traded

Diffed cell by cell against the pre-fix archive. `states.csv`, `trades.csv`,
`actions.csv`, `rebalances.csv`, `greeks.csv` and `hedge_stresses.csv` are
BYTE identical in every cell, and `final_total_pnl` is bit-identical. Only
three kinds of value moved:

- the audit verdicts;
- `identity_residual_hands`, which is now the matched-ladder residual rather
  than the 1% secant. That is the fix;
- five fields on ONE date, 2024-02-05, under the flat-q curve only, which
  were NaN because the old 1% pinned scenario aborted on the dividend yield
  range and are now genuinely measured. Their nodal rho-q audit errors land
  between 1e-10 and 1e-7 bp against a 0.01 bp tolerance.

That last one is coverage going UP. Four leg rows and one attribution row
per flat-q cell stop being unmeasurable. The flat-forward cells never had
the abort and recover nothing.

#### Cost, measured in price calls rather than seconds

**Per-cell wall clock is NOT comparable between these two runs.** The
original subset ran on two workers and this one on six, so contention
differs and the per-cell seconds cannot be divided. Two flat-forward cells
even came out FASTER post-fix, which is contention, not a speedup. Ratios
taken from those columns would be meaningless.

Price calls are deterministic and are the right measure. From `carry_cost()`
on `term_flat_q__front`, 242 audited dates:

| Stage | pre-fix calls | post-fix calls |
|---|---:|---:|
| bucket sampling | 1824 | 1824 |
| direct audit | 3744 | 6662 |
| finite stresses | 968 | 968 |
| **total** | **6536** | **9454** |

The bucket and stress stages are IDENTICAL, which is its own confirmation
that the change is confined to the audit. The audit stage costs 1.78x, the
whole run 1.45x, and the difference works out at 12.06 calls per audited
date — the twelve of the ladder, plus the node-count variation the audit
already had.

Extrapolating the full primary grid at the run-level 1.45x, 406 cells cost
roughly 130 CPU-hours rather than 90.

For the record and not as a ratio: this run took 3242 s of wall clock on six
workers; the original took 6059 s on two.

**Gate D is met**, on this one inception, for the audit criteria it states.
The tolerance was never touched and no golden was regenerated.

### The transition subset, re-run — and the readout turns out to matter after all

`subset_transition` was run pre-fix, so its figures described a check that no
longer exists. Re-run into `subset_transition_matched`, same config plus
`--quad-readout transition`, six workers, 3309 s:

**14 cells ran, 14 ok, 0 failed.** All fourteen report 242 pass, 0 fail, 1
not measured, 0 inconclusive, and `all_measured_passed: true`, against 0 of
14 before. Diffed against the pre-fix transition run, the six priced and
traded frames are byte-identical and `final_total_pnl` is bit-identical in
every cell, so this run is audit-only too.

Now the part that is new. The old record concluded that the readout
"changes nothing at the matched bump", from the 1% default where the mean
residual moved only from 0.042500 to 0.042613. That is true AT 1%, and false
at the production ladder:

| `__front`, 242 dates | mean abs R | max abs R | max abs(R)+E |
|---|---:|---:|---:|
| `term_flat_q`, legacy | 0.000102 | 0.002196 | 0.004340 |
| `term_flat_q`, transition | 0.000017 | 0.000311 | 0.001333 |
| `term_flat_fwd`, legacy | 0.000090 | 0.001994 | 0.003947 |
| `term_flat_fwd`, transition | 0.000022 | 0.000307 | 0.001274 |

**The transition readout shrinks the residual about sevenfold and the noise
floor about threefold.** The mechanism is the one already established here:
at 0.00025 relative the bump is 1.3 index points inside a 19.7-point cell,
so under `legacy_linear` both derivatives are chord slopes and what survives
the subtraction is the UNCANCELLED part of the staircase, roughly
`(Gamma_frozen - Gamma_pinned) * (S_mid - S)`. The transition readout
removes the staircase, leaving only smooth truncation, which is why its
ladder converges at a clean `h^2` while legacy's oscillates.

So the readout earns something here it was not built for. It was adopted to
fix a sub-cell delta probe; it also tightens this audit's floor, and the
evidence for that did not exist while the audit was pinned to a 1% secant.

One caveat on the comparison: the two runs solve slightly different coupons,
4.5276% against 4.5272%, because the readout changes prices. That is far too
small to account for a sevenfold gap, but the cells are not the same product
to the last digit.

### The 0.01-hand tolerance, derived at last — and a budget that had to be rejected

The figure was inherited from a deterministic fixture and never sized for a
quadrature-priced snowball beside a discretely monitored barrier. It is now
derived. No repricing was needed: the production run records both spot
derivatives at all three levels, so each component's error follows from its
own ladder. For an `h^2` estimator `D(2h) - D(h) = 3 C h^2`, so the error at
the finest bump is `abs(D(2h) - D(h)) / 3`.

**The obvious budget is wrong, and the measurement says so.** Summing the
component magnitudes, `(E_D + E_DF + sum_i abs(F_i/S) E_Bi)`, as the earlier
draft proposed, gives this over the 242 flat-q dates:

| hands | mean | p95 | max |
|---|---:|---:|---:|
| `E_D`, frozen-curve delta | 1.02e-02 | 5.88e-02 | 1.16e-01 |
| `E_DF`, pinned-futures delta | 1.02e-02 | 5.85e-02 | 1.16e-01 |
| summed budget | 2.04e-02 | 1.17e-01 | 2.32e-01 |
| observed abs(R) | 1.02e-04 | 3.92e-04 | 2.20e-03 |

The two components are each about a hundred times the residual they produce.
They CANCEL, systematically, because both are derivatives of the same surface
at the same spot and share its readout and grid error. A sum-of-magnitudes
budget ignores exactly the cancellation the check is built on. Adopting it
would set the tolerance at 0.232 hands, 23x looser than today, and a bucket
wrong by 0.2 hands would pass a check that currently catches 0.02.

That is the same trap as the sawtooth budget recorded above, in mirror
image. There, a tolerance sized to an error the check cannot SEE would not
have admitted one extra date. Here, a tolerance sized to an error the check
CANCELS would admit errors it is supposed to catch. Component budgets are
for a probe that measures one derivative. This check measures a difference.

**The right derivation is two-sided.** The tolerance has to clear the
residual a CORRECT book produces, and stay under the smallest error worth
catching.

| Bound | hands | where it comes from |
|---|---:|---|
| noise floor, `legacy_linear` | 0.00434 | max abs(R)+E over 484 correct-book date-curves |
| noise floor, `transition` | 0.00133 | the same 484, under the other readout |
| one IM contract | 1.0 | `reference_multiplier` is 200, the index multiplier, so one hand IS one contract, and hedges round to whole contracts |

Any tolerance in that window works, and the window spans a factor of 230, so
this is not a delicately poised number. **0.01 sits near the conservative
end**: 2.3x above the noise floor under the default readout, 7.5x under
`transition`, and 100x below a one-contract error either way. It maximises
detection and accepts a modest false-positive margin. Keep it.

The floor is readout-dependent, so the binding number is the legacy one,
since `legacy_linear` is still the default and nothing about this tolerance
should assume the opt-in mode is on.

Economically, at the worst date's spot of 5306.99 and a 50m notional:

| tolerance, hands | CNY per 1% index move | bp of notional |
|---:|---:|---:|
| 0.01 | 106.14 | 0.0212 |
| 0.001 | 10.61 | 0.0021 |
| one contract, 1.0 | 10614 | 2.1229 |

**The one caveat is sample size.** That 2.3x margin is measured on a single
inception. If the worst abs(R)+E grows past about 0.005 on other inceptions
at Gate E, the margin is gone and the tolerance needs revisiting. The
components to watch are in `identity_ladder`, which every run now records.

Method note: on 34 flat-forward dates abs(R) slightly exceeds this summed
budget. Those are quiet dates where the spot truncation has vanished and the
residual sits on the quote-bump floor, which I held fixed at the 4.86e-05
measured on the worst date. The floor is date-dependent. It does not change
the conclusion, and it is another reason not to build a tolerance out of
per-date component estimates.

Reproduce with `quad-readout/tolerance_budget.py`, which reads the recorded
ladders and prices nothing.

---

## Open item — SUPERSEDED 2026-09-10

**Superseded by the matched-identity section above, and kept for the two
readings it retracts.** Its premise, that the audit measures the spot
derivative below one grid cell and therefore cannot certify anything, does
not survive: at a matched step BOTH derivatives sit inside the same cell and
the staircase cancels between them.

No tested setting certifies this product on this engine, and none will while
the audit measures the spot derivative below one grid cell. Every subset
cell's `audit_summary.json` records `all_measured_passed: false`, so none of
them can be read as a validity pass by accident.

The cause is no longer open. The QUAD price is the LINEAR interpolant of a
node array on a lattice pinned to the barrier rather than to spot, so delta
is a staircase with 19.7-point treads and 0.25-hand risers, and every
sub-cell finite difference inherits an `h * |S*Gamma + Delta|` sawtooth. See
`quad-readout/` for the proof and the candidate readouts.

What is NOT in question:

- the hedge sizing. The bucket vector and the frozen-carry delta feeding it
  are consistent to 1e-13, the three policies produce the design's exact
  holdings, and the replay's P&L is untouched. The 1% pricing bump spans 5.4
  cells and averages the staircase to 0.028 contracts peak to peak against a
  1-contract rounding granularity;
- the audit machinery. It caught a deliberately perturbed bucket in Gate A,
  it refuses to be satisfied by re-sizing inside a scenario, and here it has
  surfaced a genuine engine property rather than a defect of its own.

**The staircase has since been fixed**, as an opt-in mode:
`QuadParams.readout` selects `"legacy_linear"` (the default, so nothing
moved) or `"transition"`, which evaluates the final backward transition at
the spot instead of interpolating its output. On the worst state above, the
detrended sub-cell delta spread falls from 4.6297 to 0.0168 reference hands
within two cells of the barrier.

**That is not the whole near-barrier story.** A separate, larger delta error
survives it: a smooth signed lobe centred on the barrier, present
identically under both readouts and therefore in the surface rather than the
readout. The detrended metric above cannot see it, because detrending
removes an error shared by every cell by construction.

It is QUAD-specific, on a peer session's bump-free comparison against an
independently validated reference: PDE at 1601 points sits 0.38 and 0.58
study hands from that reference near the barrier, while QUAD sits 16.73
hands out at the study's own 401 points and still 13.03 at 3201.

My own QUAD-minus-PDE reproduction of it is **retracted as a magnitude**. It
ran on this worktree's PDE, which predates `25d4f7d6` and therefore carried
three near-barrier readout defects of its own, so it measured two errors at
once. The lobe is real and QUAD-specific on the peer's evidence; my numbers
for it are not to be quoted, and re-measuring needs this branch merged up to
its base. See `quad-readout/README.md`. Unresolved.

**UNBLOCKED 2026-09-11.** The branch is merged up to its base, so `25d4f7d6`
is now an ancestor and the corrected PDE is available here. The reconcile
was smaller than it looked: the base carries this branch's readout work as
`0e4afd67`, the code half of `806187fc`, so all six shared engine files
merged with no change. The only real divergence was
`backtest/simulation/config.py`, where this branch's multi-leg bucket-hedge
guard and the base's `barrier_offsets` were independent additions from the
merge base, and both survive.

The merge touched no QUAD, carry, futures or replay source file, and the
frozen identity regression still reproduces 0.548283, so none of the Gate D
evidence above moved. Re-measuring the lobe is now work, not a blocker.

**The merge also inherits 24 failing tests, and they are the base's, not
this branch's.** The full suite after merging is 24 failed, 8073 passed,
120 skipped. All 24 sit in three files, and every one of them fails on the
base commit `d926e1f5` with nothing from this branch present:

| Where | Count | Why |
|---|---:|---|
| `test_variant_case_builders.py::test_banked_cells_keep_their_identity` | 10 | `QuadParams.readout` arriving on the base moves every QUAD candidate identity hash |
| `test_banked_certificates.py::test_banked_certification_still_describes_its_engines` | 11 | the same moved hashes |
| `test_replay_goldens.py::test_frame_matches_golden` | 3 | the PDE barrier-readout repair moves PDE deltas |

Checked out at `d926e1f5` alone, those three files give 24 failed and 99
passed, the same count as the merged suite, and `test_replay_goldens.py`
alone gives the same three failures by name. Before the merge this branch
passed `test_replay_goldens.py`, because it did not yet carry `25d4f7d6`.

The replay golden is not float churn. On `scalar_bsm/greeks` the delta moves
from 18.296472 to 19.509487, 6.6% relative, which is the barrier readout
repair doing exactly what its message says it does.

`d926e1f5` predicted half of this itself: "What did move the QUAD identities
is the readout field arriving on QuadParams, which is a numerics-relevant
knob and is meant to." Neither the certificates nor the goldens were
re-banked to match.

**Nothing here was re-banked either, and that is deliberate.** Re-banking a
replay golden hides a real pricing change, and re-banking a QUAD certificate
means RE-PRICING its cells, not re-hashing them, because a certificate
covers only the configuration it names. This is the base branch's debt and
it should be settled there, on purpose, not absorbed silently by a merge
whose own subject is something else.

What remains open, with the measured evidence in `quad-readout/README.md`:

- **the budget, but NOT the one proposed here.** Earlier drafts of this
  section argued for deriving the audit tolerance from the sawtooth
  amplitude `h * |S*Gamma + Delta| / m_ref`, and called it worth doing
  whatever readout is in use. That is now the wrong budget for this check.
  The audit's two spot derivatives use matched steps, so the sawtooth enters
  both and cancels; a tolerance sized to an error the check cannot see would
  not admit a single extra date. The amplitude formula remains the right
  budget for a SUB-CELL delta probe, which is a different instrument. The
  0.01-hand figure is still an initial deterministic-fixture tolerance never
  derived for a quadrature-priced snowball beside a discretely monitored
  barrier, so the budget question is open — it just has to be sized against
  the knock-in barrier error below, which is what the check actually
  resolves.

  **CLOSED 2026-09-11.** The instinct here was right and its final clause
  was wrong. A component budget is the wrong instrument, for the same
  cancellation reason given above, and the knock-in barrier error is not
  what the check resolves — that reading is retracted with the rest. The
  tolerance is now derived two-sided, between a measured noise floor and one
  contract. See "The 0.01-hand tolerance, derived at last".
- **the remaining floor**, which is that knock-in barrier error. See below.
  **RETRACTED**: the floor was 1%-secant truncation, not the barrier.

Disabling alignment is not among the options: measured, it makes the
sawtooth 70x worse and moves prices 14.9 bp, because the barrier's own
projection error is what alignment exists to prevent.

### The audit cannot see the staircase at its own bump, and the ladder above misled me

*This section is the one that got closest, and it holds up with two
amendments. It establishes correctly that `delta_q` is invariant to the
audit bump. It missed that a MATCHED pair still carries truncation, because
the two directions have different third derivatives, so it stopped one step
short and went looking for a surface defect instead. And its "the staircase
cancels" is only approximate: the uncancelled remainder is invisible at a 1%
bump but is the DOMINANT term at the refined ladder, where it makes the
legacy readout's floor 3.3x the transition readout's. See "The transition
subset, re-run". The `subset_transition` figures quoted below are pre-fix
and stale for the same reason as the primary subset.*

The subset has now been re-run entirely under `readout="transition"`
(`bucket_hedge_v2/subset_transition`). Every cell's verdict is unchanged. On
`term_flat_q__front` the two runs agree to the leg row: 66 pass, 175 fail,
1 not measured, 1 inconclusive under both readouts, and the mean identity
residual moves only from 0.042500 to 0.042613 reference hands.

That is not the readout failing. It is the audit being blind to the
staircase at its own settings, by construction.

The residual is
`(delta_q - delta_f_direct - sum_i (F_i/S) B_i) / m_ref`. Both `delta_q` and
`delta_f_direct` are central differences of the same price surface at the
same spot. At the default the two steps are the same size, because
`audit_spot_bump_rel` resolves from the pricing bump. So both sides inherit
the same sawtooth and it cancels in the subtraction. Measured over the same
243 dates, as a mean absolute change in reference hands:

| what changes | `delta_f_derived` | `delta_f_direct` | identity residual |
|---|---:|---:|---:|
| readout, at the matched default bump | 0.006814 | 0.006503 | 0.000787 |
| bump 0.01 to 0.001, readout fixed | 0.000000 | 0.014838 | 0.014838 |

The first row is the cancellation: the readout moves both sides by nearly
the same amount, so the residual barely moves and no verdict flips.

The second row says why the ladder ever showed anything. `delta_f_derived`
is EXACTLY invariant to the audit bump, to every digit, because `delta_q` is
the engine's own delta Greek at the engine's own `BumpConfig` bump
(`equity/engine/base_engine.py:243`) and never the audit's. Only `delta_f_direct` follows
`audit_spot_bump_rel`. Shrinking the audit bump therefore unmatches the two
steps and exposes the staircase on one side alone.

The ladder under both readouts:

| audit spot bump | legacy | transition |
|---:|---:|---:|
| 0.01 (default, matched) | 0.042500 | 0.042613 |
| 0.0025 | 0.051449 | 0.043244 |
| 0.001 | 0.084353 | 0.046811 |

The readout does exactly what it was built to do. It removes the
bump-dependence: the mismatched-bump residual falls by 44% and the passing
dates rise from 29 to 65. It changes nothing at the matched bump.

**So the ladder must be read the other way round.** I built it to diagnose
why the audit fails and read its growth as evidence about that failure. The
growth was real, and following it did find a real engine defect which is now
fixed. But the growth was an artefact of the probe unmatching the two bumps.
The audit, at its own settings, was never failing on the staircase.

### What is left is the live knock-in barrier, and nothing else — RETRACTED

The split below is correct and reproduces. The conclusion drawn from it is
not: a live daily-monitored knock-in barrier is also where `d3V/dS3` is
largest, so 1%-secant truncation predicts exactly the same localisation.
The measurement never separated the two. See "What the pre/post-knock-in
split did and did not show" above.

Splitting the residual on the lifecycle flag the run already records
(`quad-readout/residual_by_ki_state.py`) localizes it completely:

| | dates | mean abs | median | max |
|---|---:|---:|---:|---:|
| knock-in barrier live | 178 | 0.057257 | 0.044636 | 0.548283 |
| after knock-in | 63 | 0.000808 | 0.000578 | 0.002909 |

A 71x collapse, on the same product, engine, curve, tolerances and bumps.
Under `transition` the same split gives 0.057410 and 0.000808; the
post-knock-in figures agree to six decimals.

Once the barrier is extinguished the identity holds to 8e-4 hands, an order
of magnitude inside the 0.01 budget, on 63 consecutive dates. So on this
evidence the chain rule, the audit machinery, the bucket deltas and the
engine's delta Greek are all sound. The monthly knock-OUT barriers are live
throughout both legs, so this is not barriers in general: it is specifically
the daily-monitored knock-in barrier.

That is the same object as the near-barrier delta lobe — a knock-in-barrier
property, identical under both readouts, resistant to refinement — so there
is one open defect here rather than two.

**Both sentences above are withdrawn.** "Resistant to refinement" was only
ever true of the mismatched ladder; the matched one collapses quadratically.
And with the residual explained as truncation, nothing ties it to the lobe.
There is one open defect here, the lobe, and the identity is not evidence
for it.

Two hypotheses died, and both were mine:

- **the uncovered tail.** The identity sums over LISTED contracts, and this
  1Y product outlives a strip running about eight months, so the
  extrapolated tail looked structurally unattributed. Refuted
  (`quad-readout/tail_hypothesis.py`): the correlation between uncovered
  span and residual is +0.16, dates where the curve spans the product carry
  a HIGHER mean residual than dates where it does not, and
  `has_product_tail` is false throughout.
- **the bucket bump.** `B_i` is taken at a fixed 1 index point regardless of
  the spot bump, which would reproduce the observed bump-invariance. The
  knock-in split refutes it: a fixed bucket bump cannot switch itself off
  when the product knocks in. The confirming 10-point run never completed —
  it raises `dividend yield magnitude must be <= 1.0`, because a 10-point
  move on a short-tenor contract implies an absurd yield. So this is
  refuted by the split, not by its own run.

A third attempt failed differently and is worth recording as a method note.
`quad-readout/identity_vs_pde.py` strips the identity to three price bumps
through a single flat yield, to compare QUAD against PDE with no study, no
curve and no replay. Its own bump control condemns it: the residual there
scales ~100x between a 1% and a 0.1% bump, which is O(h^2) truncation in the
central differences, while the study's residual is bump-invariant over the
same three bumps. Different error structure means a different quantity, so
its cross-engine numbers say nothing about the floor. Routing the chain rule
through one analytic yield is what breaks it; the study measures the bucket
deltas directly. The script is kept WITH the control that condemns it. The
one claim that survives is that QUAD legacy and QUAD transition agree
closely in every cell, which is the readout being irrelevant again.

## What these gates do NOT establish

- **No economic conclusion.** Gate E (the full paired study over every
  eligible inception) has not been run. Nothing here says the bucket hedge
  performs better, or worse, than a single contract.
- **A one-inception subset cannot establish long-history behaviour**, and
  the timing it gives is one inception's, not the grid's.
- **Certificates cover only what they name.** These gates cover the QUAD
  engine on the flat-vol scalar channel with the two supported futures
  conventions. Vol-model runs, other engines and other carry sources are
  outside them.
- **Unquoted carry risk stays unhedged.** The tail and shape stresses are
  reported, not mitigated: no listed futures position responds to them at
  all. That remains true however neutral the nodal column looks.
- **The readout finding is not a fix.** `quad-readout/` measures the cause
  and the candidates; nothing in the engine was changed, and no golden was
  regenerated. Whether the staircase matters elsewhere in the library — the
  KI-probability readouts at `snowball_quad_engine.py:1227` go through the
  same `interpolate` — was not investigated.
- **The identity does not certify the delta.** Its `D` term is now a
  repriced matched-step derivative, so the engine's own Greek left the gated
  equation. What still constrains that Greek is the net-delta reproduction
  check, which confirms it IS a 1% secant of this pricer and nothing more.
  Closure is internal consistency of one numerical surface, not accuracy.
- **The 1% secant's own gap is measured but NOT gated.**
  `pricing_delta_local_gap_hands` is the reported hedge delta minus the
  finest local delta, over the 242 archived flat-q dates:

  | | hands |
  |---|---:|
  | mean | 0.176 |
  | max | 0.887 |
  | dates above the 0.01 budget | 229 |

  Gating it would fail 95% of dates. It is left ungated because a 1% secant
  is the desk's hedge convention rather than an approximation to the local
  slope, and because 0.18 contracts sits inside the 1-contract trading
  increment against a 91-contract position — the same argument used for the
  staircase, now with the quantity itself on the record. If that convention
  is ever revisited, this column is where the cost of it is already written
  down.
