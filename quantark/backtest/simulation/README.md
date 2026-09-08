# `quantark.backtest.simulation` — simulated-path backtests

## What it is

The replay backtest answers "how did this hedge do on the one path history
handed us?".  This package supplies the other kind of path: a batch of
simulated joint market futures a book can be replayed over, so the answer
becomes a distribution instead of a single number.  A path-dependent product
such as a snowball has almost all of its risk in the shape of the path, and
one realised history samples that shape once.

Plan 1 of the design (`docs/superpowers/specs/2026-09-08-simulated-path-backtest-design.md`)
ships the data layer only: the batch state type, the carry-to-chain
arithmetic, the shared dividend rule, the history builder and the three
generators.  The pricing loop, the cache and the results layer follow in
later plans.

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

## What comes next

- **Plan 2** — the repricing pricer and its state cache, the vectorised lifecycle, hedge and ensemble loop, and the bit-for-bit conformance check against the replay engine.
- **Plan 3** — the spot-ladder pricing mode, the on-disk cache tier, the accuracy gate, and the PDE life-surface pricer.
- **Plan 4** — ensemble results, distributions and persistence, plus the worked snowball study in `example/`.
