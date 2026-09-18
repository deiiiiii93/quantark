# Snowball hedging on simulated paths

## What it asks

The q term-structure study (`example/snowball_q_term_structure`) answered
on 29 realised inceptions of one product: does pricing a CSI 1000 snowball
off the IM chain's carry term structure hedge better than pricing off one
flat yield implied from the hedge contract?  Twenty-nine overlapping
histories are a thin sample of a distribution.  This study runs the same
product, the same carry models and the same two hedge contracts over
2,000 bootstrapped futures paths and five designed stresses from one start
state, and reports distributions of the hedge measures, paired differences
on matched paths, and where the realised runs sit inside the simulated
distribution.

## Data

The same local, untracked caches as the q study, under
`example/mo_volmodels/data/history`: the CSI 1000 spot, the IM chain and
the admitted IV surfaces (the vol channel is the ATM implied vol at the
product tenor).  The history's fingerprint is recorded in
`paths/manifest.json`, so a batch can always be traced to the days it was
resampled from.

## Paths

Stage 01 builds the joint daily history (log spot, vol, rate, carry curve
on the default tenor grid) and resamples its daily changes with a
stationary block bootstrap: mean block 20 days, returns demeaned, annual
drift 0, vol floor 0.08, carry resampled as levels.  Paths start from the
history's last day and run 275 weekdays forward: a 12-month product
matures about 261 weekdays after inception and the engine refuses a
calendar that ends before the book settles (the spec's 260 was short).

Carry is resampled as levels, not as daily changes, because the curve holds
the total log basis per tenor and a year of resampled changes is a random
walk: by day 274 the cross-path dispersion of the 1.5Y pillar was twice the
history's (0.119 against 0.056) and the 1M pillar reached −9.8% against a
historical floor of −5%, which a contract 25 days from expiry turns into an
annualised yield above 100% that the term dividend curve refuses.  Levels
keep every pillar inside the history's range.  Vol has no such switch: it
is a random walk of daily changes above the floor, and on 40 paths it
reaches 0.56 against a historical maximum of 0.33 — a caveat, and a
generator follow-up (a levels mode for vol).

The stress set on the same calendar: a 30% crash over 20 days into the
knock-in barrier; a V shape 28% down over 20 days and back over 40; a 15
point vol spike decaying over 40 days; a 5 point basis blow-out over 10
days; a 0.2% per day grind over 60 days into the knock-out barrier.

## Product and cells

The q study's 1Y standard snowball at the start spot: KO 103% on monthly
observations after a 3-month lockout, KI 75% on every trading day, seller
short one unit sized to 50 million.  The fair coupon is solved once, under
`term_flat_q` with QUAD on a 401-point grid at the start state
(`coupon.json`), and every cell starts from the traded price (initial
price 0), so paired terminal P&L differences are pure hedge P&L, as in the
historical study.

Cells are the baseline flat-q model (the engine's historical default, named
by the q study), `term_flat_q` and `term_opt_tail`, each hedged with the
front-month IM contract and with the longest listed one; 1 bp per side.
A q-study model whose carry comes from a contract other than the hedge's
is refused: the simulation inverts the active contract only.

## How a cell is priced

Every cell is priced by one provider, `per_date`: exact repricing on the PDE
engine.  Each state is one backward solve from maturity to that state's date,
on the tracker's aged contract, with that date's own term dividend object.
Nothing is collapsed to a scalar yield, so the carry models under test reach
the engine intact.  One provider for all six cells is deliberate: a flat-arm
provider different from the term-arm provider would make the provider choice
move with the treatment the study measures.

**Why not the whole-life surface.**  The surface solved one PDE over the
product's life per `(vol, q, rate)` bucket and read a column per day.  A term
curve has to be flattened to `q_T` to fit that key, and on 2026-09-14 that
flattening was 107.34 of the 108.73 bp worst gate gap on `term_flat_q`.  A key
that holds the curve does not rescue it: the surface's reuse comes from a flat
market being the same market on every day, while a term curve is
re-snapshotted each day relative to that day.  Counted on the 2,000 × 275
batch, the flat key shares 22.7 states per solve and a (day, vol, rate, carry)
key 1.05.  So the per-date solve costs about what exact repricing costs, and it
is exact.

**One solve per state.**  Exact repricing used to call `price()` and then
`calculate_greeks()`, and each ran the full solve.  Phase 0
(`phase0_single_solve.py`) compared removing the second solve with a solver
memo or in the provider; the decision and its evidence are in the design note
`docs/superpowers/specs/2026-09-15-per-date-pde-provider-design.md`.

The mesh is pinned (1601 points, 16 steps a day, step cap 8000, bounds
0.40–1.60 of the initial spot); the day-zero convergence evidence for it is
in "Earlier provider" below.

**Engine check.**  Each cell's first 40 bootstrap paths are also repriced on
exact QUAD and paired with the cell on the same paths.  It is reported, not
gated.  The fair coupon is solved under `term_flat_q` on QUAD, so a cell's
day-0 mark is its own carry model's price of that contract on the PDE: the
carry-model gap (zero only for `term_flat_q`) plus the engine gap.  The
report's day-0 table separates them against the exact-QUAD check, which
starts from the same state.

## Checks

- `per_date` is exact: its gate report is zero by construction.
- The oracle spot check: 3 single paths per bootstrap run through the replay
  engine at zero tolerance, and the first of them on each exact-QUAD check.
- The engine check: exact QUAD on each cell's first 40 paths, paired with the
  cell (reported, not gated).

## Earlier provider: the whole-life surface (2026-09-09 to 2026-09-15)

The study's first fleet provider, kept in the library for flat-carry books.
It ran the flat arm within its 25 bp gate after the readout fix and the
pinned mesh below, and could not run the term arms (see "How a cell is
priced").  The measurements that shaped it stay here because the mesh they
pinned is the one `per_date` uses.

### What a cell measured under the surface

The bootstrap and stress batches run on the PDE life surface: one solve per
(vol, q, rate) bucket, read along the path.  A surface is solved at a flat
`q`, the bucket centre of the zero yield at the remaining maturity, and a
ladder node's environment carries a flat yield too.  So under the surface
and the ladder the term dividend models enter only through that scalar
`q_T` along the path; only exact repricing hands the engine the term
object.  A cell therefore measures the term model's `q_T` path under the
surface and the ladder, and the term object itself only in the optional
exact-QUAD subset (`--exact-paths`).  The gate reports the gap between the
two on every cell.

**Measured on 2026-09-14, this is decisive, not a caveat.**  One bootstrap
path, 65 sampled states, each priced three ways: on the surface, exactly
with the term object, and exactly with the surface's own flat `q`.  That
splits every gate gap into the carry SHAPE, which no mesh or bucket step
can remove, and everything else.

| carry model | worst gate gap | of which carry shape | the surface itself |
|---|---|---|---|
| `flat_from_hedge` | 14.72 bp | 0.00 | 14.72 |
| `term_flat_q` | 108.73 bp | 107.34 | 1.39 |

On the flat model the carry term is identically zero on all 65 states and
the surface holds its 25 bp budget.  On the term model the surface is just
as accurate — worst residual 15.28 bp over the path — and the gate failure
at 38.24 bp is entirely the flat-versus-term dividend object.  So the life
surface can run the flat-carry arm of this study and cannot run the term
arms: flattening the carry there would compare a flattened term model
against a flat model, which measures nothing.

The root cause is the cache key, not the solver.  `solve_life_surface`
accepts a `TermStructureDividendYield` and prices it correctly — the
surface's day-0 column equals `PDEEngine.price` to the digit under both a
flat and a term object, and at day 0 the term object behaves like a flat
14.6603% against the scalar's 14.6554%, worth 0.34 bp.  What the surface
cannot do is LOOK ONE UP: `LifeSurfacePricer._surface` keys its store on
`(vol, q, rate)`, three floats, so a curve has to be collapsed to one
number first.  The number is `q_T`, the zero yield at the product's
remaining maturity, which reproduces the terminal forward exactly and
carries nothing about the shape in between — and a snowball prices off the
intermediate forwards, because that is what its monthly knock-out and
daily knock-in observations see.

On path 0 that shape is violent.  The one-month zero yield falls to 0.05%
on day 120 and stands at 26.5% on day 121 while `q_T` reads 11.36% and
17.02%, a curve spread of 11.4 and 9.7 percentage points; the gate gaps on
those two days are −108.73 and +98.41 bp.  At about 70 bp of notional per
1% of carry (measured: 352,000 on a 50M book) a shape mismatch of a
fraction of a percent in effective yield is worth tens of basis points.

So the fix is a key a term curve can occupy.  The obvious objection is
that reuse is the whole point of the provider and a seven-tenor curve
would give one surface per state — but measured, there is almost no reuse
to lose.  At 40 paths the study's steps give 3,037 distinct (vol, q, rate)
buckets over 11,000 path-days, and the run performed 6,568 solves for
6,472 priced states: more than one full solve per state.  The 2 GB surface
store holds 99 surfaces of 20 MB against those 3,037 buckets, so it evicts
6,172 times and re-solves what it dropped.

That has a consequence beyond the term-carry question.  At 40 paths the
life surface is not cheaper than exact repricing: 2,654 s on four workers
against the published exact-QUAD cell's 2,542 s on one, for the same
paths.  Distinct buckets grow as `paths**0.745`, so reuse rises from 3.6
states per surface at 40 paths to about 8 at 2,000, which is where the
provider starts to earn its keep — by roughly threefold in CPU, not the
fiftyfold the 35-hour exact-QUAD figure suggests.  Surface capacity, not
mesh and not bucket width, is the lever on that number.

The surface's spot domain is set explicitly to 0.40–1.60 of the initial
spot (`SURFACE_SPOT_RANGE`).  A surface is solved once at the start spot
and read along the whole path, so its domain has to be a path envelope,
not the pricer's default vol-scaled pricing envelope, which a 30% crash on
a low-vol day would leave (the readout then fails closed).  The upside is
capped by the knock-out.  The choice is also a cost: the PDE's concentrated
mesh is built per distinct spot, and the gate's exact leg and the oracle pay
it once per state — measured on the 1Y product, 1.07 s on the default
domain, 0.15 s on 0.6–1.6, 1.08 s on 0.4–2.5.

### Bucket steps

A surface, or a ladder node, is solved at its vol and q bucket centres, so
half a bucket times the sensitivity is a PV gap the gate sees.  Measured on
the 1Y product at the 2026-09-09 start state: vega about 80 bp of notional
per vol point at the start spot and about 150 bp per point near the
knock-in barrier; yield sensitivity about 70 bp per 1% of q.  At the
plan's 0.01 vol step the first quick run failed the surface gate at 68 bp
(0.48 points off a centre), so the study uses `VOL_STEP = 0.002` and
`Q_STEP = 0.001`; at (0.0025, 0.000625) on 8 real paths the worst of 64
sampled states was 12.7 bp and 0.40 hands.  On real paths vol and carry
both move daily, so nearly every state is its own bucket until the path
count is in the thousands: 8 paths needed 942 surface solves.

### Surface checks

The surface gate repriced 64 reservoir-sampled states exactly against 25 bp
of notional and 2 hands of the hedge; the ladder gate 64 states, 10 bp,
2 hands.  A cell that missed its budget produced nothing, and the engine
check paired each surface cell's first 200 paths with the QUAD spot ladder.

- A near-barrier readout defect, since fixed.  A discrete knock-in
  observation writes a value JUMP onto the grid, and the surface used to read
  and differentiate the event-projected column, which within a cell of the
  barrier is the value of neither branch.  Measured on the test fixture at
  0.02% above a 90% barrier five days from maturity: 86 bp and 140 hands
  against the exact engine, and worsening under refinement rather than
  improving.  The solver now returns the continuation branches, and the same
  fixture state costs 2.24 bp and 0.81 hands, inside the study's own budget.
  The ladder's 12 bp at that state is interpolation and is unchanged.
- The mesh, since pinned.  The surface ran on the engine's "standard"
  accuracy profile: 400 points, 4 steps a day, a mesh concentrated around
  the critical prices.  It now runs a fixed 1601 points at 16 steps a day,
  which against the Gaussian-transition quadrature reference converges at
  second order and lands the study's designed states within 0.14 hands away
  from expiry.  Sixteen steps a day asks for 5060 intervals and the
  engine's shipped cap of 5000 used to deliver 4799 without saying so,
  which is why the old refinement ladder looked divergent; the cap is
  stated at 8000 and the surface now refuses a scaled fill rather than
  pricing on it.
- What the mesh costs: nothing, here.  A two-path cell does 509 solves in
  592 s on the profile and 521 in 599 s on this mesh, because the grid
  binder rebuilds its layout for every market bucket and that rebuild, not
  the sweep, is the solve.  Doubling again to 3201 by 32 does cost, about
  4.4 times per solve, and buys a smaller number against a reference that
  cannot support it, so it was not taken.  A surface here is four times the
  size of a profile one, so the 2 GB surface cache holds 99 of them against
  397; that budget is what to raise at fleet path counts.
- Known cost: the PDE grid binder keys its layout cache on the whole market
  snapshot, so every vol or q bucket rebuilds the same mesh under the
  study's explicit bounds; a mesh memo on the mesh's own inputs is the
  library follow-up that would remove the rebuild.

## Running it

```bash
# stage 1: the banked 40-path batch of 2026-09-09, all six cells, in its own directory
.venv/bin/python example/snowball_simulated_paths/02_ensemble_fleet.py \
    --out-dir output/snowball_simulated_paths/per_date_40 \
    --paths-dir output/snowball_simulated_paths/paths --workers 6 --batch-paths 7 --resume
.venv/bin/python example/snowball_simulated_paths/03_report.py \
    --out-dir output/snowball_simulated_paths/per_date_40 \
    --data-dir output/snowball_simulated_paths/per_date_40/data

# stage 2: 2,000 paths from the same history cut
.venv/bin/python example/snowball_simulated_paths/01_build_paths.py \
    --history-end 2026-09-09 --out-dir output/snowball_simulated_paths/per_date_2000
nohup caffeinate -i -m -s .venv/bin/python example/snowball_simulated_paths/02_ensemble_fleet.py \
    --out-dir output/snowball_simulated_paths/per_date_2000 \
    --workers 12 --batch-paths 50 --resume \
    > output/snowball_simulated_paths/per_date_2000/fleet.log 2>&1 &
.venv/bin/python example/snowball_simulated_paths/03_report.py \
    --out-dir output/snowball_simulated_paths/per_date_2000
```

Paths, cells (`cells/<cell>[__stress|__exact_quad|__ladder_quad]/`, each with
`config.json` and `run.json`), `coupon.json` and `fleet_manifest.json` go to
the `--out-dir`; `--paths-dir` reads a batch from elsewhere without copying.
`--resume` skips a run whose `config.json` fingerprint matches, a recorded
failure included, and a run interrupted in its oracle reuses its results.
`--history-end` cuts the history at a day: at 2026-09-09 it reproduces the
banked batch's history, bootstrap and stress fingerprints from the longer
cache.  `--batch-paths` must be set for `--workers` to take effect; each run
is batched for its own path count.  Keep it well below `n_paths / workers`:
`batch_for` caps the batch at one per worker, and at that size a single slow
batch idles every other worker for the rest of the run — at 2,000 paths and
12 workers, 170 left 11 workers idle for 80 minutes while 50 held them
within 4% of each other.  `--disk-cache` is omitted deliberately (see
Caveats).  `--provider exact|life_surface|ladder` remain available.

## Results

Six cells on 2,000 bootstrap paths and the five stresses, from the
2026-09-09 start state (history 2023-05-04 to 2026-09-09, 816 days),
every cell priced by **per-date PDE repricing** — one backward solve from
maturity to each state's own date, carrying that date's real term
dividend object.  Fair coupon 37.8254% under `term_flat_q`.  All 18 runs
passed their gate at 0.00 bp and 0.00 hands, and all 24 replay oracles
matched at zero tolerance.  Run 2026-09-16 to 2026-09-17.

**Lifecycle** (path-determined, so identical across cells): KO 69.1%,
KI 25.2%, reaching maturity 30.9%.

**Hedge-cost distributions, 2,000 paths** (bp of notional; ES = mean of
the 5% loss tail for P&L, of the 5% upper tail for cost-like measures):

| measure | cell | mean | q05 | q50 | q95 | ES |
|---|---|---|---|---|---|---|
| terminal P&L | flat_from_hedge front | −324 | −1576 | −252 | 789 | −2332 |
| | flat_from_hedge far | −66 | −1414 | −45 | 1142 | −2165 |
| | term_flat_q front | −202 | −1400 | −170 | 884 | −2103 |
| | term_flat_q far | −41 | −1325 | −25 | 1133 | −2108 |
| | term_opt_tail front | −190 | −1394 | −166 | 929 | −2095 |
| | term_opt_tail far | −30 | −1336 | −20 | 1151 | −2104 |
| daily P&L std | flat_from_hedge front | 218 | 93 | 197 | 428 | 548 |
| | flat_from_hedge far | 89 | 31 | 84 | 164 | 215 |
| | term_flat_q front | 91 | 40 | 89 | 146 | 178 |
| | term_flat_q far | 71 | 32 | 68 | 124 | 154 |
| | term_opt_tail front | 87 | 37 | 85 | 140 | 171 |
| | term_opt_tail far | 70 | 31 | 66 | 123 | 153 |
| variance reduction R² | flat_from_hedge front | 0.40 | 0.08 | 0.41 | 0.68 | |
| | term_flat_q front | 0.69 | 0.28 | 0.75 | 0.90 | |
| | term_opt_tail far | 0.80 | 0.56 | 0.83 | 0.95 | |
| cost | flat_from_hedge front | 30 | 7 | 24 | 75 | 91 |
| | term_opt_tail far | 15 | 3 | 10 | 42 | 54 |

At 1 bp per side, cost in bp equals turnover by construction.

**Paired against the baseline `flat_from_hedge__front`, same 2,000 paths:**

| variant | terminal mean bp | share > 0 | t | daily std mean bp | share > 0 | t |
|---|---|---|---|---|---|---|
| flat_from_hedge far | +258 | 79% | 28.7 | −129 | 0.1% | −65.6 |
| term_flat_q front | +122 | 66% | 17.2 | −127 | 0.1% | −63.9 |
| term_flat_q far | +283 | 80% | 29.7 | −146 | 0.1% | −70.7 |
| term_opt_tail front | +134 | 67% | 17.7 | −131 | 0.1% | −61.8 |
| term_opt_tail far | +294 | 79% | 29.4 | −148 | 0.2% | −68.5 |

**Two levers, and they compose.**  Holding the carry model fixed, moving
the hedge from the front contract to the far one is worth +161 bp of
terminal P&L and −19.6 bp of daily hedge error under `term_flat_q`
(t 26.5 and −64.6), +160 and −17.0 under `term_opt_tail`.  Holding the
hedge fixed, the term models cut daily hedge error by 127–131 bp against
the flat baseline.  Doing both is the best cell in the grid: `term_opt_tail`
on the far contract cuts daily hedge error from 218 bp to 70 and lifts
terminal P&L by 294 bp on 79% of paths.  The daily-std improvement is
near-universal — on 1,998 of 2,000 paths — while the terminal-P&L gain,
though large in the mean, fails on a fifth to a third of paths.

**A cell's day-0 mark** is its carry model's price of a contract whose
coupon is fair under `term_flat_q` on QUAD, plus the engine gap:
`flat_from_hedge` +69.18 bp (front) and −28.43 (far), `term_flat_q`
−0.29 both, `term_opt_tail` −6.05 both.  The engine gap alone is −0.29 bp
in all six cells.

**Engine check** (per-date PDE minus exact QUAD on each cell's first 40
paths; `data/engine_check.csv`):

| cell | terminal P&L bp mean / t | daily std bp mean / t |
|---|---|---|
| flat_from_hedge front | +0.14 / 0.08 | −0.01 / −0.36 |
| flat_from_hedge far | +1.08 / 0.52 | +0.04 / 1.52 |
| term_flat_q front | −1.74 / −0.80 | +0.00 / 0.03 |
| term_flat_q far | −8.33 / −2.76 | +0.23 / 2.77 |
| term_opt_tail front | −1.53 / −0.89 | −0.01 / −0.49 |
| term_opt_tail far | −0.40 / −0.22 | −0.01 / −0.40 |

Every entry is small against the 122–294 bp effects the study measures —
at most 8.3 bp, and under 2 bp in five of six cells.  The QUAD reference
is not itself clean, but re-running it with a forced lattice alignment
moves these numbers by less than 0.71 bp: see the alignment caveat below.

**Stress** (terminal P&L bp / daily std bp; the designed paths move the
product, not the hedge choice, so cells agree within tens of bp):

| scenario | flat front | flat far | term_flat_q far | termination |
|---|---|---|---|---|
| crash 30% over 20 days into KI | +381 / 2.5 | +486 / 4.2 | +486 / 2.9 | maturity, knocked in |
| V shape 28% down, back over 40 days | +2382 / 27.6 | +2231 / 28.3 | +2226 / 27.9 | maturity, knocked in |
| vol spike +15 pts decaying over 40 days | −3225 / 12.4 | −3301 / 12.4 | −3302 / 12.0 | maturity |
| basis blow-out −5 pts over 10 days | −2763 / 13.6 | −3057 / 12.4 | — | maturity |
| grind +0.2%/day into KO | −296 / 5.3 | −386 / 4.1 | — | knock out, day 66 |

**Historical runs inside the simulated distribution** (29 inceptions per
cell, 2023-05 to 2025-09; `data/historical_location.csv`):

| cell | realised terminal bp | pct | realised daily std bp | pct |
|---|---|---|---|---|
| flat_from_hedge front | 489 | 84 | 384 | 84 |
| flat_from_hedge far | 426 | 70 | 86 | 46 |
| term_flat_q front | 546 | 80 | 57 | 20 |
| term_flat_q far | 442 | 69 | 53 | 31 |
| term_opt_tail front | 567 | 78 | 56 | 21 |

Realised inceptions earned more than most simulated paths from this one
38%-coupon start state, so read the terminal percentile as indicative.
The hedge-error percentiles carry more: on realised history the flat
front-contract model was worse than the bootstrap suggests (84th
percentile of its own distribution) and every term model better (20th to
31st).

Tables: `data/fleet_cells.json`, `data/fleet_paired.csv`,
`data/engine_check.csv`, `data/stress_table.csv`,
`data/historical_location.csv`, `data/fleet_summary.json` (gates and
oracles).  Report: `data/simulated_paths_report.html`.

### Superseded: the 2026-09-09 quick run

**Recorded before Design B, on exact QUAD, before the ageing fixes of
2026-09-15 (`8cb0ec67`, `f8987344`).  The section above replaces it.**
What follows is the quick run of 2026-09-09: 40 bootstrap
paths and the five stresses from the 2026-09-09 start state (history
2023-05-04 to 2026-09-09, 816 days; spot 7659.6, ATM vol 26.1%, front IM
carry 14.0%), two cells (`flat_from_hedge__front`, `term_flat_q__front`),
fair coupon 37.83% under `term_flat_q`, the bootstrap and stress batches on
**exact QUAD repricing** (`--provider exact`, so every number below is the
engine's own, no gate question), the 8-path ladder check with the study's
gates, one oracle path per run.  Wall clock on a shared machine.

| run | seconds | engine calls | gate | oracle |
|---|---|---|---|---|
| flat_from_hedge__front (40 paths) | 2542 | 6472 | exact | pass, exact columns match |
| flat_from_hedge__front__stress | 355 | | exact | – |
| flat_from_hedge__front__ladder_quad (8 paths) | 922 | | **failed**: 14.24 bp / 0.27 hands vs 10 bp / 2 hands | – |
| term_flat_q__front (40 paths) | 205 (oracle only; results resumed from a run interrupted in its oracle) | 6472 | exact | pass, exact columns match |
| term_flat_q__front__stress | 436 | | exact | – |
| term_flat_q__front__ladder_quad (8 paths) | 1051 | | **failed**: 70.42 bp / 1.66 hands | – |

The two ladder failures are the study's providers measured, not noise.
The baseline's 14 bp is the spot-ladder and bucket interpolation on a
product whose vega is 80–150 bp per vol point.  `term_flat_q`'s 70 bp is
the caveat stated above made concrete: a ladder node carries a flat q at
the bucket centre while the exact leg hands the engine the term object,
and on the 2026-09 chain that difference is worth 70 bp of notional.  The
life surface, run earlier on the plan's 0.01 vol step, failed the same
product's gate at 68 bp (vol bucketing) and, at the finer steps, at 2.84
hands (its stencil delta against the engine's bump delta near a barrier).
Both approximate providers therefore produced no numbers here; the
distributions below are exact repricing, one engine call per state.

Re-measured on 2026-09-11, after the readout fix and on the pinned mesh,
the surface's day-zero gate on this product is 5.61 bp and 7.15 hands,
against 16.95 bp and 13.04 hands on the accuracy profile it replaced, and
every designed state away from expiry is 0.14 hands or better.  **The
surface now passes its gate.**  The 7.15 hands is one state, the last
trading day a fraction of a percent above the knock-in barrier, where the
surface and the fresh solve the gate scores it against both oscillate as
the barrier moves within a cell: 7.15 hands at this mesh, 2.42 at 3201
points and 32 steps, 3.06 at 6401 and 64.  The reference itself moves by
about half a percent between meshes, so a budget tighter than that would
measure the mesh rather than the provider.  The gate therefore judges that
state on its own delta scale, 1% of 814 hands against 2 hands for a
typical 40-hand state, and it passes at 7.15 against an 8.14 allowance.
The two ladder failures are unchanged; the ladder is not the fleet's
provider.

**Hedge-cost distributions, 40 paths** (bp of notional; ES = mean of the
5% loss tail for P&L, of the 5% upper tail for cost-like measures):

| measure | cell | mean | q05 | q50 | q95 | ES | share > 0 |
|---|---|---|---|---|---|---|---|
| terminal P&L | flat_from_hedge | −343 | −1924 | −329 | 948 | −2626 | 17% |
| terminal P&L | term_flat_q | −176 | −1886 | −66 | 1039 | −2384 | 35% |
| daily P&L std | flat_from_hedge | 229 | 100 | 216 | 466 | 636 | |
| daily P&L std | term_flat_q | 98 | 49 | 98 | 136 | 154 | |
| variance reduction R² | flat_from_hedge | 0.43 | 0.21 | 0.42 | 0.68 | | |
| variance reduction R² | term_flat_q | 0.69 | 0.33 | 0.76 | 0.88 | | |
| cost | flat_from_hedge | 33 | 7 | 27 | 76 | 99 | |
| cost | term_flat_q | 30 | 5 | 25 | 73 | 91 | |
| roll-day MTM jump | flat_from_hedge | 240 | 56 | 199 | 452 | 1137 | |
| roll-day MTM jump | term_flat_q | 162 | 59 | 128 | 374 | 585 | |
| delta churn (hands/day) | flat_from_hedge | 4.0 | 1.4 | 3.8 | 6.9 | 10.7 | |
| delta churn (hands/day) | term_flat_q | 3.3 | 1.0 | 3.0 | 5.9 | 9.7 | |

KO 57%, KI 40%, maturity 42% of paths (the lifecycle depends on the path
alone, so both cells agree).  At 1 bp per side, cost in bp equals turnover
by construction.

**Paired, term_flat_q − flat_from_hedge on the same 40 paths:**

| measure | mean | median | share > 0 | t |
|---|---|---|---|---|
| terminal P&L bp | +166 | +90 | 82% | 4.2 |
| daily P&L std bp | −131 | −104 | 0% | −7.2 |
| variance reduction R² | +0.26 | +0.28 | 93% | 10.9 |
| max drawdown bp | −859 | −545 | 0% | −5.8 |
| cost bp | −3.0 | −2.7 | 0% | −9.3 |
| roll-day MTM jump bp | −78 | −36 | 12% | −2.9 |
| other-day MTM jump bp | −56 | −49 | 0% | −11.0 |
| delta churn | −0.75 | −0.67 | 0% | −10.2 |

The historical study's direction holds on simulated paths: the term model
halves the daily hedge error on every path, removes a quarter more of the
product's daily variance, trades less, and its terminal P&L is better on
82% of paths by 166 bp on average.

**Stress** (terminal P&L bp / daily std bp, both cells within a few bp of
each other):

| scenario | flat_from_hedge | term_flat_q | termination |
|---|---|---|---|
| crash 30% over 20 days into KI | +385 / 2.5 | +376 / 1.1 | maturity, knocked in |
| V shape 28% down, back over 40 days | +2381 / 27.8 | +2342 / 27.9 | maturity, knocked in |
| vol spike +15 pts decaying over 40 days | −3225 / 12.4 | −3259 / 12.0 | maturity |
| basis blow-out −5 pts over 10 days | −2763 / 13.6 | −2810 / 13.0 | maturity |
| grind +0.2%/day into KO | −294 / 5.3 | −336 / 4.0 | knock out, day 66 |

The designed paths move the product, not the hedge choice: a vol spike or a
basis blow-out on a seller's book costs 28–33% of the coupon-sized notional
either way, and the two carry models differ by tens of bp on them.

**Historical runs inside the simulated distribution** (29 inceptions per
cell, 2023-05 to 2025-09, each measured with the same library function;
`data/historical_location.csv`): the realised terminal P&L sits at the
84th percentile (flat) and 77th (term) of the simulated distribution on
average — the realised inceptions earned more than most simulated paths
from one 38%-coupon start state, so read the percentile as indicative.
The realised daily hedge error of the flat model sits at the 83rd
percentile of its simulated distribution (realised 384 bp mean against 229
simulated), the term model's at the 14th (57 against 98): on realised
history the flat model was worse, and the term model better, than the
bootstrap suggests.

Tables: `data/fleet_cells.json`, `data/fleet_paired.csv`,
`data/stress_table.csv`, `data/historical_location.csv`,
`data/fleet_summary.json` (gates and oracles included);
`data/engine_check.csv` is empty because both ladder checks failed their
gates.  Report: `data/simulated_paths_report.html`.

## Caveats

- One start state, the history's last day, for every simulated path; a
  historical inception's percentile is indicative.
- The fair coupon is fair under `term_flat_q` on QUAD while cells price
  their own carry model on the PDE, so a cell's day-0 mark holds a
  carry-model gap and an engine gap (report, day-0 table).
- Paired t-statistics treat the simulated paths as independent draws,
  unlike the historical study's overlapping inceptions.
- **The exact-QUAD reference in the engine check is not clean, though it
  costs the study at most 0.71 bp.**  With
  `QuadParams.align_priority="auto"` the lattice pins whichever barrier is
  nearest spot in log space, so the alignment target changes at
  `sqrt(KI·KO)` = 0.8789 of the inception spot.  A bumped delta evaluates
  the base, up and down states separately, so within one 1% bump of that
  level the three evaluations do not share an alignment and the delta
  carries the grid change as well as the market change.  Measured on this
  product by forcing the priority: outside the window `auto` reproduces a
  forced branch exactly, and inside 0.8709–0.8869 it differs from both
  forced branches by 17.5–20.3 index-delta units (0.09–0.10 of an IM
  contract).  5.85% of the check cells' priced states sit within 1% of
  that level.  The two forced branches differ by only about 2 units
  here, so the damage does not come from the branches
  disagreeing — it comes from mixing them inside one finite difference,
  which means a product whose branches nearly coincide is no safer.
  Re-running all six check cells with `--quad-align ko`, which makes every
  evaluation share one alignment, moves the engine check by −0.71 to
  +0.65 bp of terminal P&L and at most 0.09 bp of daily std
  (`data/engine_check_align_ko.csv`).  Two things attenuate it, and the
  second matters more.  The cube's `delta` is in index units, 200 to an IM
  contract, so 19 units is 0.095 contracts against a book averaging 35 —
  but that does *not* mean it rounds away: comparing the forced and
  default runs state by state, the rounded hedge differs on 15.3% of the
  states inside the window (307 of 2,010), and on 1.2% of all states.
  What keeps those from mattering is that the hedge rebalances daily, so
  each one is a one-contract difference for one day that the next
  rebalance corrects; the error appears and disappears as a path crosses
  the window and never compounds.  Note the largest engine gap,
  `term_flat_q__far` at −8.33 bp,
  **survives** forcing (−8.81 bp, t −3.14): that cell's PDE-QUAD
  difference is something else, not the alignment.  `align_cell_stretch`
  (unmerged elsewhere) is the better fix than forcing, since it puts both
  barriers on nodes instead of pinning around the inconsistency.  The
  per-date PDE cells the study reports are unaffected either way: they are
  compared against each other, same engine both sides.
- Do not run the fleet with `--disk-cache` across cells that differ in
  hedge.  `StateKey` deliberately excludes the hedge, but in this study
  the hedge selects the active futures contract and therefore the priced
  dividend, while `env_key` is built from `(rate, spot, carry row)` only —
  so a `far` cell reads a `front` cell's prices.  The gate does not catch
  it (it re-prices through the same provider); the oracle does.
- The stress paths are designed, not sampled.
- The bootstrap's vol is a random walk of daily changes; its dispersion
  over a year exceeds the history's.
