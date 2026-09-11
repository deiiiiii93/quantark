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

## What a cell measures under each provider

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

The surface's spot domain is set explicitly to 0.40–1.60 of the initial
spot (`SURFACE_SPOT_RANGE`).  A surface is solved once at the start spot
and read along the whole path, so its domain has to be a path envelope,
not the pricer's default vol-scaled pricing envelope, which a 30% crash on
a low-vol day would leave (the readout then fails closed).  The upside is
capped by the knock-out.  The choice is also a cost: the PDE's concentrated
mesh is built per distinct spot, and the gate's exact leg and the oracle pay
it once per state — measured on the 1Y product, 1.07 s on the default
domain, 0.15 s on 0.6–1.6, 1.08 s on 0.4–2.5.

## Bucket steps

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

## Gates and checks

- The surface gate: 64 reservoir-sampled states repriced exactly, 25 bp of
  notional and 2 hands of the hedge.  The ladder gate: 64 states, 10 bp,
  2 hands.  A cell that misses its budget produces nothing.
- The oracle spot check: 3 single paths per run through the replay engine
  with the gate's tolerances; the lifecycle columns and the hedge contract
  must match exactly, trades within the tolerance are netted per day.
- The engine check: the first 200 bootstrap paths of every cell on the QUAD
  spot ladder (0.25% nodes), paired against the same paths on the surface.
- A near-barrier readout defect, since fixed.  A discrete knock-in
  observation writes a value JUMP onto the grid, and the surface used to read
  and differentiate the event-projected column, which within a cell of the
  barrier is the value of neither branch.  Measured on the test fixture at
  0.02% above a 90% barrier five days from maturity: 86 bp and 140 hands
  against the exact engine, and worsening under refinement rather than
  improving.  The solver now returns the continuation branches, and the same
  fixture state costs 2.24 bp and 0.81 hands, inside the study's own budget.
  The ladder's 12 bp at that state is interpolation and is unchanged.
- Known cost: the PDE grid binder keys its layout cache on the whole market
  snapshot, so every vol or q bucket rebuilds an identical concentrated mesh
  under the study's explicit bounds, about a second each; a mesh memo on the
  mesh's own inputs is the library follow-up that would make the surface
  solves cheap.

## Running it

```bash
.venv/bin/python example/snowball_simulated_paths/01_build_paths.py            # 2,000 x 275 (--quick: 40 paths)
.venv/bin/python example/snowball_simulated_paths/02_ensemble_fleet.py --quick  # two cells, 8 check paths, 1 oracle path
nohup caffeinate -i -m -s .venv/bin/python example/snowball_simulated_paths/02_ensemble_fleet.py \
    --workers 4 --batch-paths 250 --disk-cache --resume > output/snowball_simulated_paths/fleet.log 2>&1 &
.venv/bin/python example/snowball_simulated_paths/03_report.py
```

Measured on the quick run: an exact 40-path cell is 40–50 minutes (6,472
engine calls, one per state), a 5-path stress set 6–7 minutes, an 8-path
ladder check 15–18 minutes, one oracle path 3–4 minutes.  The 2,000-path
fleet on exact QUAD would be about 35 hours per cell on one worker; the
life surface is the provider meant for it, once its two gate failures
recorded below are addressed.

Paths, cells (`cells/<cell>[__stress|__ladder_quad|__exact_quad]/` written
by `EnsembleResults.to_dir`, with `config.json` and `run.json`),
`coupon.json` and `fleet_manifest.json` go to
`output/snowball_simulated_paths/`; the tables and the HTML report go to
this directory's `data/`.  `--resume` skips a run whose `config.json`
fingerprint matches, a recorded failure included, and a run interrupted in
its oracle (results on disk, no `run.json`) reuses its results and runs
only the oracle.  A run that misses its gate produces no results; it is
recorded in its `run.json` with `failed` set, the fleet carries on, and the
report lists it as a failed gate.  `--provider exact` runs the bootstrap
and stress batches on exact QUAD repricing instead of the surface: no gate
question, one engine call per state, the term dividend object handed to
the engine on every call.

## Results

**The 2,000-path fleet has not been run yet; its numbers will replace this
section.**  What follows is the quick run of 2026-09-09: 40 bootstrap
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
- Under the surface and the ladder the term models enter through `q_T`
  only; `--exact-paths` is where the engine receives the term object.
- Paired t-statistics treat the simulated paths as independent draws,
  unlike the historical study's overlapping inceptions.
- The stress paths are designed, not sampled.
- The bootstrap's vol is a random walk of daily changes; its dispersion
  over a year exceeds the history's.
