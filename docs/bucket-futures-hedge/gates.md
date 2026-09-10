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

### Synthetic validation

```sh
.venv/bin/python example/snowball_q_term_structure/05_bucket_hedge_validation.py \
  --synthetic --out-dir OUT
```

**Result: 7/7 required cases passed, exit 0, about 2 s, no vendor history.**

### Historical validation

```sh
.venv/bin/python example/snowball_q_term_structure/05_bucket_hedge_validation.py \
  --historical-dates 2025-03-03 \
  --out-dir example/snowball_q_term_structure/data/bucket_hedge_v2/validation
```

**Result: 7/7 required cases passed on the real chain.**

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

**This hypothesis was refuted too.** Shrinking the bump makes it WORSE:

| audit spot bump | fail | pass | mean abs identity | max abs identity | net delta audit error, mean |
|---:|---:|---:|---:|---:|---:|
| 0.01 (default) | 175 | 66 | 0.042500 | 0.548283 | 4.2e-14 |
| 0.0025 | 155 | 86 | 0.051449 | 0.477425 | 5.0e-02 |
| 0.001 | 213 | 29 | 0.084353 | 1.337295 | 8.4e-02 |

Predicted 16x and 100x reductions; observed 0.83x and 0.61x, i.e. increases.
A residual that GROWS as the step shrinks is noise divided by a small step,
not truncation.

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

**Gate D is still NOT met**, and the tolerance was not touched. But the
failure is now attributed rather than open: it is a property of the engine's
readout, measured, and shown not to affect the hedge.

---

## Open item

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
survives it: a smooth signed lobe centred on the barrier, of order 10 to 25
study hands, present identically under both readouts and therefore in the
surface rather than the readout. It was found by a peer session against an
independently validated reference and reproduced here against the PDE
engine. The detrended metric above cannot see it, because detrending removes
an error shared by every cell by construction. See `quad-readout/README.md`.
Unresolved.

What remains open, with the measured evidence in `quad-readout/README.md`:

- **the budget.** The sawtooth amplitude is not noise, it is
  `h * |S*Gamma + Delta| / m_ref`, a quantity the engine knows. An audit
  budget derived from it would say what this engine can actually resolve.
  The 0.01-hand figure is an initial deterministic-fixture tolerance and was
  never derived for a quadrature-priced snowball beside a discretely
  monitored barrier. This is still worth doing whatever readout is in use.
- **re-running the subset under `readout="transition"`.** A run entirely
  under the new mode would validate the NEW mode. It would say nothing about
  whether the shipped default certifies, and must not be presented as the
  old path passing. The replay does not yet expose the setting.

Disabling alignment is not among the options: measured, it makes the
sawtooth 70x worse and moves prices 14.9 bp, because the barrier's own
projection error is what alignment exists to prevent.

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
