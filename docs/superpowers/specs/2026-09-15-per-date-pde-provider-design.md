# Per-date PDE solve for the simulated-path snowball study (Design B)

Status: approved in brainstorming 2026-09-15, sections 1–3.
Branch: `feat/simulated-path-backtest`.
Supersedes the study's use of the whole-life surface (Design A) as its fleet
provider. The library keeps Design A.

## 1. Decision and why

The study (`example/snowball_simulated_paths`) compares carry models
(`flat_from_hedge`, `term_flat_q`, `term_opt_tail`) × hedge contracts
(`front`, `far`) on bootstrapped paths. Two thirds of the fleet are
term-carry cells.

**Design A cannot run the term cells.** `LifeSurfacePricer` solves one
backward PDE over the whole product life per `(vol, q, rate)` bucket and
reads a column per day. A term dividend curve has to be collapsed to the
scalar `q_T` to fit that key, and the flattening is worth 107.34 bp of the
108.73 bp worst gate gap on `term_flat_q` (README, measured 2026-09-14).

**A curve-aware key does not rescue Design A.** The whole-life surface
reuses a solve because a *flat* market is the same market on every day.
A term curve is re-snapshotted each simulated day, relative to that day
(bootstrap carry levels are `history.carry[row]` for a resampled history
row), so a surface solved for one day's curve is the right market on that
day only. Counted on the study's 2,000 × 275 batch (seed 1, carry as
levels):

| key | 40 paths | 2,000 paths |
|---|---|---|
| `(vol, q, rate)` buckets, no day (q proxied by the nearest carry pillar) | 1.6 states/solve | 22.7 states/solve |
| `(day, vol, rate, carry row)`, the most permissive exact key | 1.00 | 1.05 (3.3% of groups shared) |

**Design B: solve each date separately.** On each date, V(S) is solved from
maturity back to that date with that date's real dividend object; the
result is one column. One provider serves all six cells. **No hybrid:**
assigning one provider to the flat arms and another to the term arms would
make provider choice perfectly correlated with the treatment the study
tests.

Prerequisite, done: the aged-contract semantics audit (schedules against
the term sheet over all 275 days) and the four aging defects it found
(`8cb0ec67`, `f8987344`).

## 2. Goals and non-goals

Goals
- All six cells priced by one provider that hands the engine the real term
  dividend object on every state.
- One PDE solve per state, not two.
- A 2,000-path fleet from the 2026-09-09 start state, with a cross-engine
  check against exact QUAD and oracle spot checks.

Non-goals
- Changing `LifeSurfacePricer` or the ladder (they remain valid for flat
  carry books).
- The grid-binder mesh memo. Measured today, a repeat solve of the same
  state on the pinned mesh costs the same as the first, so the sweep, not
  the mesh build, looks like the cost.
- The last-trading-day near-barrier delta oscillation (a stated limit).
- Approximate grouping of states (see 3.1).

## 3. Design

### 3.1 B is exact repricing on the PDE engine

No new provider class. `RepricingPricer` in exact mode, driven by the
study's PDE engine config, already prices each distinct state with one
engine solve from maturity back to the state's date, on the tracker's aged
product, with that date's `TermStructureDividendYield` and implied basis.
That is the per-date solve.

Decided: **one solve per state**, keyed on the existing exact `StateKey`
(product fingerprint, day, knock-in flag, spot, vol, `env_key` over rate,
spot and carry row, exact engine fingerprint). Grouping states on
`(date, vol, rate, carry)` with spot as a query coordinate was rejected:
it saves about 5% of solves at 2,000 paths and gives up bitwise parity
(the dividend object and basis are spot-dependent at the last ulp, and the
PDE mesh is built around the solve's spot).

Consequences carried over unchanged from exact mode:
- the gate report is zero by construction (nothing approximated);
- the `StateCache` memory and disk tiers, keyed exactly;
- bitwise parity with `ReplayBacktestEngine`, established by plan 2's
  single-path oracle comparisons through one shared engine.

### 3.2 The one library change: one solve instead of two

`RepricingPricer._price_env` calls `engine.price(product, env)` and then
`engine.calculate_greeks(product, env)`. On `SnowballPDESolver` each call
runs its own `_solve`; there is no memo. Both read the price identically
from the solve result (`readout_override`, else interpolation of
`readout_vec` / `solution_vec`), so one solve can in principle serve all
three numbers.

Measured on path 0 of the banked batch, `term_flat_q`, the pinned mesh,
seconds on a shared machine (6 of 14 cores busy):

| day | `price()` | `calculate_greeks()` | `price()` repeated, same state | exact QUAD price+greeks |
|---|---|---|---|---|
| 1 | 0.67 | 0.63 | 0.80 | 1.11 |
| 60 | 0.51 | 0.47 | 0.58 | 0.92 |
| 130 | 0.31 | 0.29 | 0.35 | 0.24 |
| 200 | 0.11 | 0.10 | 0.11 | 0.15 |
| 250 | 0.02 | 0.01 | 0.02 | 0.01 |

**Phase 0 is a standalone runtime-patched demo, and the user chooses from
its decision matrix before any engine code changes.** Candidates:

(a) **Solve memo in `SnowballPDESolver`.** One entry: a `calculate_greeks`
(or `price`) call whose product and environment match the last solve reuses
its `PDESolutionResult`. The key is content, never `id()`:
`time_shift` writes `valuation_date` into the environment it is handed, and
a solver instance is shared across paths. The key must cover everything
`_solve` reads — the product token (strict), spot, vol surface, rate
curve, dividend object, basis, valuation date and the solver params — and
any difference is a miss. Replay backtests and pnlexplain, which make the
same two calls, get the saving too.

(b) **Provider-level.** `RepricingPricer` takes the mark from
`calculate_greeks()["price"]` and skips `price()`. No engine change; the
simulation's call pattern then differs from the replay's, so oracle parity
has to be re-proven under warm solver caches (plan 2 measured that cache
state moves results at 5e-10).

Evidence per candidate, in the matrix:
1. PV, delta and gamma bitwise against today's two-call outputs over every
   alive state of one study path, warm caches included, both KI branches
   where reachable;
2. replay-oracle parity at zero tolerance on the plan-2 fixtures, discrete
   and continuous knock-in;
3. (a) only: replay goldens byte-identical, and the memo misses on each
   input changed in isolation;
4. time per state on the study product;
5. the complete list of callers whose behaviour or cost changes.

If neither candidate is bitwise-safe, the study runs with two solves per
state and the stage-2 budget doubles (section 6); no approximation is
introduced to avoid it.

**Decision (2026-09-15, Phase 0).** The user chose candidate b
from `output/snowball_simulated_paths/phase0/decision_matrix.md`: solves per
state [baseline 2.000, chosen 1.000], study path bitwise [true],
conformance [pass], replay goldens [n.a.].  Seconds per state on path 0
(261 states, back-to-back): baseline 0.574 and 0.677 on its repeat, (a)
0.299, (b) 0.303.  Candidate (a) also passed every row, but only after its
key was corrected: the first key, `repricing._canonical` with lifecycle
attributes kept, renders a dataclass by its declared fields, and the
lifecycle tracker `setattr`s `_otc_lifecycle_knocked_in` onto the dataclass
product, so an alive and a knocked-in state keyed alike and the knock-in
probe reused the alive solve.  (b) has no key to keep complete and gives the
study the same saving.  Found on the way, not fixed here:
`product_fingerprint` does not render `contract_multiplier` or
`settlement_convention`, the product's other attributes outside its
dataclass fields, so two products differing only in those share state-cache
shards; the study prices one product.

### 3.3 The fair coupon

Unchanged: solved once under `term_flat_q` with QUAD on a 401-point grid
at the start state. This keeps the contract identical to the 2026-09-09
quick run, which stage 1 bridges to. Because cells now mark on PDE, each
cell's day-0 mark is the PDE-vs-QUAD gap rather than exactly zero —
measured about 0.5 bp of notional on day 1 (PDE 1,048,796 against QUAD
1,046,457 on 50M). Every bootstrap run records its day-0 mark in its
manifest, so the offset is visible, not buried in terminal P&L. Stage 1
also checks that the re-solved coupon still equals the banked 37.8254%
(the aging fixes do not act at inception).

## 4. Study changes

### `01_build_paths.py`
- `--history-end DATE` (inclusive): cuts the spot and futures frames before
  the vol channel and `PathHistory` are built; recorded in
  `paths/manifest.json` (absent key = no cut).
- Verified 2026-09-15: today's cache (819 days to 2026-09-14) cut at
  2026-09-09 reproduces the banked history fingerprint, bootstrap
  fingerprint and stress fingerprint exactly, and its 816-day prefix is
  identical to the cut history.
- `stationary_block_indices` draws `(n_paths, n_days)` in one call, so the
  first 40 paths of a 2,000-path batch are NOT the banked 40. Same history
  and start state, different draws.

### `02_ensemble_fleet.py`
- `PROVIDERS = ("per_date", "exact", "life_surface", "ladder")`, default
  `per_date`. `per_date` = exact repricing on the PDE config
  (`C.engine_config(model, "pde", s0=..., spot_range=C.SURFACE_SPOT_RANGE)`:
  1601 points, 16 steps/day, `max_steps` 8000, `max_points` 2000, bounds
  0.40–1.60 S0). `exact` keeps meaning exact QUAD so banked `run.json`
  records keep their meaning.
- `cell_config` engine selection: `pde` for `per_date` and `life_surface`,
  `quad` for `exact` and `ladder`. `_pricing("per_date")` is the exact
  repricing config (no gate samples).
- Defaults: `--exact-paths 40` (the check run `<cell>__exact_quad`),
  `--check-paths 0` (ladder check off; its 2026-09-09 failure was the
  flat-node carry gap, structural). `--quick`: two cells (baseline front,
  `term_flat_q` front), `--exact-paths` capped at 8, 1 oracle path.
- `--paths-dir DIR` (default `<out-dir>/paths`): a run reads a batch from
  another directory, so a new output directory never overwrites the banked
  2026-09-09 cells and needs no copied batch.
- Oracle: `--oracle-paths` (default 3) single paths per bootstrap run, and
  the first of them on the `__exact_quad` check, through the replay engine
  at the exact tolerances (0.0) the exact mode already uses. Today the
  check runs receive the whole oracle list; they get one path.
- Resume fingerprints already include `repr(engine_config)` and the
  provider; a `per_date` run can never resume from an `exact` one.
- Every bootstrap run's manifest records its day-0 book mark (3.3).

### `03_report.py`
`aggregate` already pairs `<cell>__exact_quad` against the bootstrap cell
over the check run's paths (`bootstrap.take(range(n))`, then `_paired`).
Changes: relabel (the variable and headings say "surface"; the pair is
`per_date − exact_quad`), state the check as reported, not gated, and show
each cell's day-0 mark. Distributions, carry-model pairs, stress, gate
table and historical location are unchanged.

### `README.md`
- "What a cell measures under each provider" is rewritten around Design B:
  the flattening measurement, the re-snapshot argument with the count, per
  date = exact repricing, the Phase 0 outcome.
- The life-surface sections shrink to a dated history note (what was
  measured and why it is not the fleet's provider).
- "Running it" gets the stage-1 and stage-2 commands; "Results" is replaced
  when stage 2 lands. No absolute machine path in any shipped table.

## 5. Checks

| check | on | pass rule |
|---|---|---|
| gate | every `per_date` run | trivially zero (exact mode) |
| oracle | 3 bootstrap paths + 1 check path per cell | lifecycle columns, hedge contract and trades exact; PV/delta at 0.0 |
| engine check | first 40 paths of each cell, `per_date − exact_quad` paired per hedge measure | reported, not gated |
| coupon | stage 1 | re-solved coupon equals banked 37.8254% |
| aging | done | `8cb0ec67`, `f8987344` |

## 6. Run plan and budget

| phase | work | exit |
|---|---|---|
| 0 | single-solve demo (3.2) | user picks from the matrix |
| 1 | library change for the chosen candidate, test-first | full suite green; replay goldens byte-identical |
| 2 | study changes (section 4) with tests | study tests and CLI end-to-end green |
| 3 | stage 1 on the banked 40 paths | measured cost per state and worker scaling; bridge table against the banked 2026-09-09 exact-QUAD numbers; go/no-go for stage 2 with a measured wall estimate |
| 4 | stage 2: 2,000 × 275 batch, 6 cells | all runs banked, oracles pass |
| 5 | report, README results, commits by pathspec | tables in `data/`, README updated |

Stage 1:

```bash
.venv/bin/python example/snowball_simulated_paths/02_ensemble_fleet.py \
    --out-dir output/snowball_simulated_paths/per_date_40 \
    --paths-dir output/snowball_simulated_paths/paths \
    --workers 6 --batch-paths 7 --resume
```

Stage 2 (after the peer's fleet has exited):

```bash
.venv/bin/python example/snowball_simulated_paths/01_build_paths.py \
    --history-end 2026-09-09 --out-dir output/snowball_simulated_paths/per_date_2000
nohup caffeinate -i -m -s .venv/bin/python example/snowball_simulated_paths/02_ensemble_fleet.py \
    --out-dir output/snowball_simulated_paths/per_date_2000 \
    --workers 12 --batch-paths 170 --disk-cache --resume \
    > output/snowball_simulated_paths/per_date_2000/fleet.log 2>&1 &
```

`--batch-paths` must be set: without it the batch is one range and
`--workers` is inert.

Budget, extrapolated (about 0.37 s per state for one solve averaged over
remaining life, 162 alive states per path from the banked run); stage 1
replaces these with measurements:

| run | CPU | wall |
|---|---|---|
| stage 1 | ≈ 10 h | ≈ 2 h at 6 workers |
| stage 2, one solve per state | ≈ 205 h | ≈ 17 h at 12 workers |
| stage 2, two solves per state | ≈ 405 h | ≈ 34 h at 12 workers |

Machine: 14 cores, 48 GB, shared. The peer session's `bucket_hedge_v3`
fleet holds 6 workers until about 17:00 on 2026-09-15. Stage 1 runs beside
it; stage 2 starts only after it exits (checked by process group). Two
cores stay free. Processes are killed by process group, never `pkill -f`,
and verified by CPU against wall time.

Output layout: `output/snowball_simulated_paths/` keeps the banked
2026-09-09 run untouched; `per_date_40/` and `per_date_2000/` are new.
Stage-1 tables go to a data directory under `per_date_40/`; only stage-2
tables go to the tracked `example/snowball_simulated_paths/data/`.

## 7. Testing

Library (Phase 1, depends on the Phase 0 choice):
- (a) memo: two-call and one-call outputs bitwise equal; a miss on each
  `_solve` input changed alone, `valuation_date` included; replay goldens
  byte-identical; one solve counted per `price` + `calculate_greeks` pair.
- (b) provider: `RepricingPricer` makes one engine solve per state; oracle
  parity at zero tolerance on the plan-2 fixtures.

Study (`test/test_snowball_simulated_paths_study.py`, one-month fixture):
- `--history-end` cuts the frames and is recorded; no cut leaves the
  manifest key absent;
- `per_date` builds the PDE config and exact pricing; `exact` still builds
  QUAD;
- `--paths-dir` reads a batch from another directory and writes nothing
  into it;
- the default check run is `__exact_quad` over `--exact-paths` paths;
- `aggregate` pairs `per_date − exact_quad` and reports the day-0 mark;
- the CLI end-to-end test on the `per_date` default;
- resume refuses to reuse an `exact` run for a `per_date` config.

## 8. Risks

- **Phase 0 finds neither candidate bitwise-safe.** Two solves per state,
  budget doubles; no approximation.
- **A stale memo (candidate a).** The "correct at inception, wrong
  afterwards" shape the aging defects had; mitigated by the content key and
  the one-input-at-a-time miss tests.
- **Worker scaling below linear** (spawn pools, per-worker caches).
  Measured in stage 1 before stage 2 is committed.
- **Extrapolated budget.** The per-state average assumes solve time linear
  in remaining life; stage 1 measures it.
- **Engine difference is not zero.** Day 200 on path 0: PDE 206,219 vs
  QUAD 191,841 (2.9 bp of notional). The engine check reports this on hedge
  measures; the carry-model comparison is unaffected because every cell uses
  the same engine.
