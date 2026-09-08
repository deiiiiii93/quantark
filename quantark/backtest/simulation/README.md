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
single path.  The spot ladder, the disk cache, the accuracy gate, the
results distributions and the worked study follow in later plans.

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
a `pricing` block naming the provider, its cache budget and its gate
tolerances.  None of those has a hidden default, and a setting this package
cannot honour yet is absent rather than accepted and ignored.  Optional:
`underlying`, the greek bump sizes, `allow_data_end` (let paths end before
the book settles) and free-form `metadata`.

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
two-product book, and the reported gaps are all exactly `0.0`.  A path can
be checked from the shell:

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

One method lives outside this package.
`AutocallableLifecycleTracker.resolve_calendar_schedule(dates, env)` maps a
product's observations onto a run calendar using the tracker's own records
and due rule, so the vectorised lifecycle fires on exactly the days the
tracker fires on; a second copy of that rule here would be two rules to keep
in step by hand.

## What comes next

- **Plan 3** — the spot-ladder pricing mode with vol and q bucketing, the on-disk cache tier, the sampling accuracy gate, the PDE life-surface pricer, and process-pool batching.
- **Plan 4** — ensemble results: summaries, distributions, paired comparisons and persistence, plus the worked snowball study in `example/`.
