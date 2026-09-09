# `quantark.backtest.simulation` — simulated-path backtests

## What it is

The replay backtest answers "how did this hedge do on the one path history
handed us?".  This package supplies the other kind of path: a batch of
simulated joint market futures a book can be replayed over, so the answer
becomes a distribution instead of a single number.  A path-dependent product
such as a snowball has almost all of its risk in the shape of the path, and
one realised history samples that shape once.

Plan 1 of the design (`docs/superpowers/specs/2026-09-08-simulated-path-backtest-design.md`)
ships the data layer: the batch state type, the carry-to-chain arithmetic,
the shared dividend rule, the history builder and the three generators.
Plan 2 ships the first usable engine on top of it: an exact repricing
provider with a state cache, the vectorised lifecycle and futures ledger,
the daily ensemble loop with the replay engine's accounting, and a
conformance oracle that reproduces the replay engine bit for bit on a
single path.  Plan 3 makes it fast enough for thousands of paths without
giving up that reference: a spot-ladder mode for the repricing provider,
a PDE life-surface provider that solves once per bucket and reads along
the path, an on-disk cache tier, the sampling gate that measures every
approximation against direct repricing, and path batching over a spawn
pool.  Plan 4 turns a run into answers: a per-path summary in the
historical study's own hedge measures, distributions with expected
shortfall, paired comparisons on matched paths, persistence, and the worked
study in `example/snowball_simulated_paths`.

## `MarketPath`

A `MarketPath` is one batch of paths on one trading calendar.  Every day of
every path is a complete market snapshot, so a pricer never has to guess at
a missing channel.

| Field | Shape | Meaning |
|---|---|---|
| `dates` | `(n_days,)` | the shared trading calendar |
| `spot` | `(n_paths, n_days)` | index level |
| `atm_vol` | `(n_paths, n_days)` | the scalar ATM vol the pricer receives |
| `rate` | `(n_paths, n_days)` | flat continuously-compounded rate |
| `carry` | `(n_paths, n_days, n_tenors)` | the constant-maturity carry curve |
| `tenor_grid` | `(n_tenors,)` | year fractions the curve is quoted on |
| `meta` | dict | generator, parameters, seed, fingerprints |

The carry channel is a *constant-maturity* curve: `carry[i, d, k]` is
`B(T_k) = ln(F(T_k) / S)` for a contract maturing `tenor_grid[k]` years
after day `d`, not for a fixed listed contract.  `B(0) = 0` is implicit and
the curve interpolates piecewise-linearly in `B`, continuing the last
segment's slope beyond the last node — the `ForwardCarryCurve` conventions,
vectorised by `carry_at`.  Quoting carry this way is what lets a listed
contract's basis converge to zero as it approaches expiry without anyone
modelling the convergence: the chain builder simply reads the curve at the
contract's shrinking tenor.

Every batch is validated on construction (shapes, finiteness, positive spot
and vol, increasing dates and tenors) and can be fingerprinted, written to
an npz and read back unchanged.

## Generators

All three take their parameters as explicit keyword arguments — nothing has
a hidden default — and record every one of them, plus the seed, in
`meta`.  The same seed always reproduces the same batch.

**`StationaryBlockBootstrap`** resamples the realised joint daily state
vector `[Δln S, Δvol, Δrate, ΔB_1 … ΔB_K]` in Politis–Romano geometric
blocks, so a resampled path keeps the co-movement of spot, vol and basis
that a marginal-by-marginal resample would destroy.  `mean_block_days` sets
the expected block length, `demean_returns` and `annual_drift` replace the
historical drift with a stated one, `vol_floor` bounds the simulated vol
(day 0 is the declared start state and is never floored;
`meta['vol_floor_hits']` counts the generated days it bound on), and
`carry_mode` chooses between resampling curve *changes* and curve *levels*.

**`GBMPaths`** is the transparent control: lognormal spot with a real-world
drift `mu` and volatility `sigma`, a vol rule (`ConstantVol`, or
`StickyRealisedVol` where the quoted vol tracks trailing realised vol), and a
carry curve that is either held at the start state or driven by an explicit
`carry_schedule`.

**Designed paths** turn a `DayPath` from `quantark.dynamicscenario` into a
single `MarketPath`: each `DayStep`'s `ParameterChange`s apply to the
previous day's state.  Alongside `spot`, `volatility` and `rate`, a `basis`
change shifts the annualised carry in parallel.  `SnowballStressLibrary`
names the paths a snowball desk actually worries about — a crash into the
knock-in barrier, a V-shaped round trip, a vol spike that decays, a basis
blowout, a grind up into the knock-out — and `stress_set` stacks equal-length
ones into a single batch.

## From curve to chain to q(T)

`listed_im_contracts` reproduces the CFFEX listing cycle: the current month,
the next month and the next two quarterlies, each expiring on the third
Friday (moved forward if that day does not trade).  `day_chain` prices all
four for every path at once as `F_i = S · exp(B(T_i))`, reading each
contract's own remaining tenor off the path's curve, and `DayChain.frame`
hands one path's chain back in the replay engine's futures-frame layout.

`dividend_yield_for_day` then turns that chain into the object the pricer
receives, under the same `dividend_source` setting the replay engine takes.
The term-structure branches go through `term_dividend_yield` in
`quantark.backtest.replay.dividend_source` — the engine and the q
term-structure study call the identical function — so a model name means the
same thing on a real path and a simulated one.  For the
`surface_forward_carry` convention, where the engine reads an option surface
past the last listed contract, the simulated path supplies its own curve as
the tail through `CurveTailPillars`.

## Using a path with the replay engine

`to_market_dataset(paths, i)` builds the `AutocallableMarketDataSet` for one
path of a batch, so any existing `ReplayBacktestEngine` configuration runs on
a simulated path with no changes.  That is also how later plans check the
fast vectorised loop: the replay engine on the same path is the oracle.

```python
from datetime import date
from quantark.backtest.simulation import (DEFAULT_TENOR_GRID, PathHistory, StationaryBlockBootstrap,
                                          trading_calendar, to_market_dataset)
history = PathHistory.from_frames(spot=spot_df, vol=vol_df, futures=futures_df, rate=0.02, tenor_grid=DEFAULT_TENOR_GRID)
gen = StationaryBlockBootstrap(history, mean_block_days=20, demean_returns=True, annual_drift=0.0, vol_floor=0.08,
                               carry_mode="changes", start=history.snapshot(), calendar=trading_calendar(date(2026, 9, 8), 260))
paths = gen.generate(2000, 260, seed=1)
dataset = to_market_dataset(paths, 17)   # one path for ReplayBacktestEngine
```

## Running a backtest

`EnsembleConfig` is one cell of a simulated run: a `products` list of
`ReplayProduct`s (standard snowballs with a lifecycle; the sign of
`quantity` is the side), the `engine_config` the replay engine takes
(`AutocallableEngineConfig`, scalar vol only, `dividend_source` of `None`,
`active_contract` or `futures_curve`), a futures `hedge` (`HedgeSpec`) with
its roll policy, the delta-hedge `strategy`, a `transaction_cost_model`, and
a `pricing` block naming the provider (`repricing` or `life_surface`), its
bucket steps, its cache budget and its gate tolerances.  None of those has
a hidden default, and a setting this package cannot honour is rejected
rather than accepted and ignored: `vol_step` without a `spot_step` in
repricing mode, a `spot_step` on the life surface, a life surface on a
non-PDE engine.  Optional: `underlying`, the greek bump sizes,
`allow_data_end` (let paths end before the book settles), `workers` and
`batch_paths` (see "Batching") and free-form `metadata`.

`pricing.mode` names what actually runs: `exact` (the default: no
`spot_step`), `ladder` (a `spot_step` on the repricing provider) or
`life_surface`.  Exact mode reproduces the replay engine bit for bit; the
other two are measured by the gate.

`EnsembleBacktestEngine(config).run(paths)` walks the calendar once and
returns an `EnsembleResults` whose `cube` holds every replay state column
with a per-path meaning as a `(n_paths, n_days)` array — `portfolio_value`,
`product_mtm`, `hedge_mtm`, `cash`, `total_pnl`, `delta`, `gamma`,
`futures_contracts`, the lifecycle flags and so on — plus the trade log,
the event log and a manifest (path and engine fingerprints, engine calls,
cache counters, wall time).  `path_states(i)` hands one path back as a frame
in the replay engine's schema, cut at the day that path settled.

```python
from quantark.backtest.replay import AutocallableEngineConfig, HedgeSpec, ReplayProduct
from quantark.backtest.simulation import (CacheConfig, EnsembleBacktestEngine, EnsembleConfig,
                                          GateConfig, PricingProviderConfig)
from quantark.backtest.strategy.futures_delta_strategy import AutocallableDeltaHedgeStrategy
from quantark.backtest.transaction_costs import ZeroCostModel

config = EnsembleConfig(
    products=[ReplayProduct(product=snowball, quantity=-1000.0, position_id=1, has_lifecycle=True)],
    engine_config=AutocallableEngineConfig(),                     # PDE, scalar vol, active-contract q
    hedge=HedgeSpec(kind="futures", multiplier=200.0),
    strategy=AutocallableDeltaHedgeStrategy(delta_threshold=0.0),
    transaction_cost_model=ZeroCostModel(),
    pricing=PricingProviderConfig(provider="repricing", cache=CacheConfig(memory_bytes=200_000_000),
                                  gate=GateConfig(sample_states=0, pv_tolerance_bp=0.0, delta_tolerance_hands=0.0)),
    underlying="CSI1000",
)
results = EnsembleBacktestEngine(config).run(paths)
results.cube.total_pnl[:, -1]        # terminal P&L per path
results.path_states(0)               # one path, replay schema
```

## How a state gets priced

The only scalar work in the loop is pricing.  Each day the engine gathers
the alive states of every product — spot, vol, rate, the dividend object,
the basis, the knock-in flag — into a `DayStates` and hands it to the
provider.  The `repricing` provider is exact mode: it deduplicates the
states by `StateKey`, prices each distinct key once through the real
engine, and scatters the result back.  The key is the product's terms, the
day, the knock-in flag, the float64 bits of spot and vol, an environment
key over (rate, spot, carry curve) and the engine settings; nothing about
the hedge, the cost model or the strategy is in it, so cells that differ
only there price the same states once between them.  Keys and seeds are
`blake2b` digests, never Python's salted `hash`, so a run reproduces across
processes and machines.  An MC engine is seeded per state from the key, so
a recomputed state equals its cached value bit for bit.

The dividend object and the basis are built once per distinct (rate, spot,
carry row) rather than once per path — every path on day 0 shares one
build.  Spot is in that key on purpose: `F / S` is spot-free in exact
arithmetic, but the replay's basis goes through `(F − S) / S`, and a one-ulp
difference in `q` moves a PDE mark by about `1e-11`, which the oracle sees.

The aged product (the tracker's time-decayed copy for the day) reads no
market data, so one copy per day and knock-in flag serves every path, as the
replay's single per-day copy serves the whole book.  Marks come from
`engine.price` and greeks from `engine.calculate_greeks`, two separate
calls, because that is how the replay makes them.

**Ladder mode** (`spot_step`, in log-spot) prices on a fixed ladder of
spots instead of at each path's own spot: with `S_ref` the product's
initial price, node `j` is `S_ref · exp(j · spot_step)`, the same node on
every day and in every cell.  A state at `x = ln S` needs the two nodes
that bracket it, each priced once through the real engine and cached, and
its PV, delta and gamma are linear in `x` between them (a state exactly on
a node takes that node's value).  `vol_step` and `q_step` bucket the vol
and the flat dividend yield to their centres, `np.round(v / step) · step`,
half-to-even on every machine; a step of 0 keeps that input exact.  A node's
environment carries the path's exact rate, a flat vol and a flat
`SignedDividendYield` and no basis, because nothing under `asset/equity`
reads one.  The ladder's keys are `(node index, vol bucket, rate and q
centre)`, so two cells that differ in their hedge, or two runs on different
path batches, share every node they both visit.  The gap the ladder opens
against exact repricing shrinks with the step, second order in PV, first in
delta; on the test fixture six days from expiry the PV gap is
1.8e-3 / 4.2e-4 / 5.5e-5 relative at steps 0.01 / 0.005 / 0.002.

**The disk tier** (`CacheConfig.disk_dir`) keeps priced states across runs
and processes.  Each `(product, engine, library version)` has one npz
shard named by the three fingerprints; a shard is read lazily on the first
miss for its pair, a memory miss that the shard answers is promoted to
memory, and `flush()` at the end of a run re-reads the shard under an
`fcntl` lock, merges and replaces it atomically, so batch workers flushing
the same shard lose nothing.  A shard whose headers disagree with its name
(another library version, another engine) is a miss and is never written
to.  The gate's exact repricings are keyed under the exact-mode fingerprint
and go through the same cache, so a second run of an approximate cell makes
no engine call at all, and an exact-mode run of the same book finds them.

## The life surface

The `life_surface` provider (PDE engines only) solves each product once per
`(vol, q, rate)` bucket instead of once per state.  The solve is the
ordinary snowball two-surface solve at the start date, for the whole life,
with an extra time node on every simulation day inside `(0, maturity)`; the
engine hands back both value slabs, alive and knocked-in, on every node
(`SnowballPDESolver.solve_life_surface`).  A day's readout picks the node
at the tracker's elapsed time `(date − start).days / 365`, the knocked-in
slab for a knocked-in path, else the alive one, and interpolates PV, delta
and gamma linearly in `ln S`; delta and gamma per node come from the
solver's own non-uniform three-point stencil applied to every interior node
of the column.  A spot outside the solve's grid raises rather than
extrapolates.  Surfaces live in an LRU by bytes (`surface_cache_bytes`).

Two choices to know about.  The rate is bucketed with `q_step` (both are
yields, and a surface needs one flat rate for its life), and the surface's
`q` is the flat bucket centre at the start: under a term dividend source
the readout is a flat-`q` reading of a term-`q` engine, and the gate is
what says whether that was good enough.  On the test fixture the surface
is within 2.5 bp of unit notional and 0.04 of unit delta of exact
repricing.

A surface keeps only the columns a run can read: column 0, the terminal
column and the node of every calendar day up to maturity
(`LifeSurface.select`, applied by the pricer's `compact=True` default).
Column data is copied, not recomputed, so a readout of a kept column is
bit-identical to the readout of the full surface; on the plan-3 layout that
is about four times less memory per surface, so the byte budget holds about
four times more buckets.  The PDE mesh, not the solve, is the cost of an
exact repricing at a NEW spot: the certified concentrated mesh is built per
distinct grid (the spot is a critical price), and on a wide domain it costs
about a second (measured: default vol-scaled domain of a 1Y product 1.07 s,
`bounds` at 0.6–1.6 of spot 0.15 s, at 0.4–2.5 of spot 1.08 s).  The gate's
exact leg and the oracle's replay engine pay it once per state.

## The gate

Every approximation is measured.  An approximate provider keeps a
reservoir (Algorithm R, `gate.sample_states` slots, seeded from the
product and engine fingerprints so the same run samples the same states on
every machine) of the states it priced, and `verify` reprices each sample
exactly through the engine at the state's own spot, vol and dividend object
and reports the largest PV gap in bp of unit notional and the largest delta
gap in hands of the hedge.  The engine runs it twice: on day 0's states
before pricing anything, and on the reservoir after the loop.  A gap over
`pv_tolerance_bp` or `delta_tolerance_hands` raises `GateFailure` carrying
the report, and the cell returns nothing.  Exact mode never samples and
reports zero.  The manifest records the combined report.

## Batching

`run_ensemble(config, paths)` splits the batch into consecutive ranges of
`batch_paths` paths (`None` means one range) and runs each through the
engine, in process when `workers == 1` and otherwise on a `spawn` process
pool, then concatenates the cubes, logs and manifests (counters summed,
gate reports combined, one manifest per batch kept under `batches`).  The
split is bit-inert: a path's numbers do not depend on which other paths
share its engine, and the test pins that with `array_equal` on the
accounting columns.  Batches keep their own memory caches; with a
`disk_dir` they share states through the shards.  A worker failure aborts
the run with the worker's error.

## Results

`EnsembleResults.summary` is one row per path: `termination_reason` (the
path's last `knock_out` or `maturity` event, else `data_end`),
`knocked_in`, `ko_observation_index`, `terminal_pnl` in currency, and the
hedge measures of `measures.py` (`path_measures`) in bp of the book's unit
notional — `manifest["book_notional"]`, `sum |quantity| * initial_price *
contract_multiplier` — which are the q term-structure study's
`hedge_measures` moved into the library so the historical replay and the
simulated ensemble report one implementation (the study's function
delegates).  The summary is memoised on the results object.

`distribution(measure, es_level=..., tail="lower")` reduces a summary
column to `n`, mean, std, the quantiles `q01 … q99`, expected shortfall
(the mean of the values at or below the `es_level` quantile; `tail="upper"`
takes the values at or above the `1 − es_level` quantile, for a cost-like
measure), the share of positive values, and KO, KI, maturity and data-end
frequencies over every path.  NaN values (a two-day path has no daily std)
are dropped and `n` counts what remains.

`variant.paired(base)` is `variant − base` per path for `terminal_pnl` and
every measure, plus `same_termination`; matched paths are proved by the
data — same count, same calendar, and bit-identical spot, vol and rate
columns up to each path's last day — not by a label, and unmatched runs
raise.  `PairedComparison.describe(measure)` gives mean, median, std, share
positive and the paired t-statistic.  `take(indices)` is the sub-run of the
chosen paths with renumbered logs and `manifest["path_indices"]`, which is
how a 2,000-path surface run is paired with a 200-path QUAD check run.

`to_dir(path)` writes `cube.npz` (every column, the dates as nanoseconds
with their own resolution, `last_day`, `initial_book_value`), `trades.csv`,
`events.csv`, `summary.csv` and `manifest.json` (`results_format` 1 plus the
manifest through `jsonable`); `from_dir` reads them back — the summary is
recomputed from the cube, so it is exactly the original's — and refuses a
missing file or another format.  The manifest carries `library_version`.

## Why you can trust it

`run_oracle(config, paths, i)` runs the ensemble engine over the batch and
the replay engine over path `i` alone (through `to_market_dataset`) with the
same settings, and compares them.  In exact mode the two make the same engine
calls on the same aged product in the same environment, so the target is
equality: the lifecycle flags, the active contract, the hedge size and every
trade (day, type, contract, size, price) must match exactly, and
`product_mtm`, `delta` and `total_pnl` to a stated tolerance whose default
is zero.  The suite runs it on GBM paths through knock-out, knock-in and
maturity, with a term dividend source, a traded initial price and a
two-product book, and the reported gaps are all exactly `0.0`.  For an
approximate provider the same oracle takes `pv_tolerance`, `delta_tolerance`
and `contracts_tolerance` (hands: an approximate delta can round to a
different hand, which is a size difference in a trade, not a missing
trade); the lifecycle flags and the contract must still agree exactly, and
the suite checks the ladder and the life surface within their gates.  A
path can be checked from the shell:

```
python -m quantark.backtest.simulation.conformance --path 17 --builder mypkg.cells:desk_a
```

where the builder is any callable returning `(EnsembleConfig, MarketPath)`.

## Notes on the design

Two interfaces differ from the spec's draft, deliberately.  `DayStates`
carries the whole pricing environment — the dividend *object*, the basis
and an environment key — not only a scalar `q_T`: under a term dividend
source the replay hands the engine a `q(T)` curve, and a pricer given only
the yield at the remaining maturity could not be bit-identical with it;
`q_T` is kept as the scalar the cube records in `pricing_q`.  And
`EnsembleConfig` has no `rate_schedule`: the rate rides on the `MarketPath`,
per path per day.

Delayed settlement follows the tracker's ledger on both of its clocks.
A terminal cashflow whose payment is not after its determination is paid
and settled on the event day.  A delayed one is parked as a receivable
until its settlement day; on the date clock it is paid when that day
arrives, on the numeric clock (year-fraction schedules) the ledger pays it
as soon as the valuation point reaches `determination_time + delay`, and
the receivable is discounted through `payment_time − point.time`, refusing
a remaining time at or below the settlement resolver's own tolerance.  The
`CalendarSchedule` carries the clock and the delays; the lifecycle tests
walk the tracker itself as the oracle, because the PDE engine refuses
settlement conventions and no pricing oracle can cover it.

Three things live outside this package.
`AutocallableLifecycleTracker.resolve_calendar_schedule(dates, env)` maps a
product's observations onto a run calendar using the tracker's own records
and due rule, so the vectorised lifecycle fires on exactly the days the
tracker fires on; a second copy of that rule here would be two rules to keep
in step by hand.  `GridRequest.extra_times` (the PDE grid layer) declares
time nodes that are exact grid nodes but carry no event and no damping: an
event node triggers Rannacher restarts, and daily damping would degrade
Crank–Nicolson over the whole life, so the daily readout nodes are a
separate field whose default leaves every existing request and cache key
unchanged.  And `SnowballPDESolver.solve_life_surface` runs one ordinary
solve with those nodes and hands back both slabs, restoring the solver so
a later `price` is byte-identical to a fresh one.

## The worked study

`example/snowball_simulated_paths/README.md` runs the q term-structure
study's snowball, carry models and hedge contracts over 2,000 bootstrapped
paths and five designed stresses, checks the life surface against QUAD, and
locates the historical study's realised runs inside the simulated
distribution.
