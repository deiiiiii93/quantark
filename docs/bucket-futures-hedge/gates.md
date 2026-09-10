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

That is the signature of quadrature resolution around a discretely monitored
knock-in barrier, not of a hedge defect: the hedge's own sizing inputs, the
buckets and nodal yields, are internally consistent to 1e-13 throughout.

**Gate D is therefore NOT met at `--quad-grid 401`.** A convergence analysis
at finer grids is recorded below. The tolerance was not touched.

---

## Open item

The identity residual above is unresolved at the study's default quadrature
grid. Until a grid is identified at which it settles inside the 0.01-hand
budget, no run at that grid can claim `numerical_validity`, and the
`audit_summary.json` of every subset cell already records
`all_measured_passed: false`, which keeps them out of a validity pass
automatically.

Note what this does and does not affect. The hedge is sized from the bucket
vector and the frozen-carry delta; those are consistent to 1e-13 and the
replay's P&L is unaffected. What is unresolved is whether the ENGINE's delta
and its own repriced delta agree closely enough for the identity check to
certify the run.

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
