# Simulated-path backtest for autocallable books — design

Date: 2026-09-08
Status: draft for review
Builds on: `quantark.backtest.replay` (historical product replay),
`example/snowball_q_term_structure` (the historical q term-structure study),
`SnowballPDESolver` (retains the full time-space solution after one solve).

## 1. Purpose

The replay engine backtests a product on the one realised path. For a
path-dependent product such as a snowball, one path is one draw: it says
what happened, not what the hedge costs. This feature runs the same book,
the same hedge and the same accounting over an ensemble of simulated paths
and reports distributions.

Priority of use, as decided:

1. **Hedge-cost distribution** — the distribution of hedged P&L of a product
   over many paths, for pricing, reserves and risk limits (mean, quantiles,
   expected shortfall of the hedged book).
2. **Model-risk comparison** — the same paths hedged under different pricing
   choices (term q vs flat q, front vs far contract, engine families) and
   compared pairwise on matched paths.
3. **Stress and tail scenarios** — designed adverse paths (crash into KI,
   V-shape, vol spike, basis blow-out) run through the identical loop.

Robustness of historical results (bootstrapping the realised history) falls
out of the block-bootstrap generator and is not a separate deliverable.

## 2. Scope and non-goals

In scope for the first version:

- Autocallable books priced and hedged with the replay engine's semantics:
  standard snowball products (`SnowballOption`), quantity-signed positions,
  the futures delta hedge with the front-month or far-contract roll policy,
  proportional or zero transaction costs.
- Joint simulation of **spot, ATM implied vol and the futures carry term
  structure**; rates held at a deterministic schedule (flat by default).
- Three path families: stationary block bootstrap of the realised history,
  GBM with a stated real-world drift and a vol rule, and designed paths
  from the `dynamicscenario` `PathBuilder`.
- Two pricing providers behind one interface: a PDE life surface (one solve
  per bucket, read along the path) and engine-agnostic repricing with a
  state cache (QUAD, PDE, MC, analytical, vol-model engines).
- A vectorised ensemble engine that steps every path at once, plus a
  conformance oracle that runs any single path through the real
  `ReplayBacktestEngine` and asserts equality.
- Library module `quantark.backtest.simulation` and an example study
  `example/snowball_simulated_paths/`.

Not in scope (explicitly):

- Phoenix and KO-reset lifecycle branches in the vectorised state machine.
  They ship only if a conformance test covers them; the first version
  states its product scope as the standard snowball and refuses others.
- Listed-option hedges (vega or gamma hedging) and a simulated option
  surface. Only the ATM vol scalar is simulated.
- Stochastic rates.
- Vol-model recalibration along a simulated path (the `surface_history`
  channel). Vol-model engines can be used as repricing engines with a
  frozen calibration only.
- Any pricing shortcut other than the two providers (no analytic
  approximations of the snowball, no coarser "fast" engine substitutes
  that are not themselves QuantArk engines).

## 3. Architecture

```
quantark/backtest/simulation/
├── __init__.py       public API
├── config.py         EnsembleConfig, PricingProviderConfig, CacheConfig,
│                     GateConfig (validation, required fields)
├── paths/
│   ├── market_path.py  MarketPath (arrays), TenorGrid, trading calendar helpers
│   ├── bootstrap.py    StationaryBlockBootstrap
│   ├── gbm.py          GBMPaths (+ vol rules)
│   └── designed.py     DayPath -> MarketPath, SnowballStressLibrary
├── carry.py          simulated carry curve -> listed IM chain per day;
│                     q(T) per dividend-source rule
├── pricing/
│   ├── base.py         PathPricer protocol, DayStates, StateKey
│   ├── surface.py      LifeSurface, LifeSurfacePricer
│   ├── repricing.py    RepricingPricer (engine-agnostic, ladder or exact)
│   └── cache.py        StateCache: in-memory LRU + optional on-disk store
├── lifecycle.py      vectorised snowball state machine
├── hedge.py          vectorised futures ledger, roll policy, delta strategy
├── engine.py         EnsembleBacktestEngine (the daily loop over arrays)
├── results.py        EnsembleResults, PathSummary, distributions, manifest
└── conformance.py    replay-oracle comparison + CLI entry point
```

Everything public is importable from `quantark.backtest.simulation`.
Nothing in `quantark.backtest.replay` changes behaviour; the oracle
consumes `BookBacktestResults` frames as they are. The one change outside
the package is in the PDE grid layer (`asset/equity/engine/pde/grid`): a
`GridRequest` option to include extra time nodes (the simulation dates) so
the life surface has a node on every day. Existing callers pass none and
are unaffected; a test pins that the default grid is byte-identical.

## 4. Data model

### 4.1 `MarketPath`

A batch of paths on one trading calendar:

| field | shape | meaning |
|-------|-------|---------|
| `dates` | `(n_days,)` | trading days, `pd.DatetimeIndex`, strictly increasing |
| `spot` | `(n_paths, n_days)` | index close |
| `atm_vol` | `(n_paths, n_days)` | the scalar vol the pricer receives (ATM, tenor stated in `meta`) |
| `rate` | `(n_paths, n_days)` | flat continuously-compounded rate |
| `carry` | `(n_paths, n_days, n_tenors)` | cumulative log carry `B(T_k) = ln(F(T_k)/S)` on `TenorGrid` |
| `tenor_grid` | `(n_tenors,)` | tenors in years, e.g. `[1/12, 2/12, 3/12, 6/12, 9/12, 1, 1.5]` |
| `meta` | dict | generator name, parameters, seed, source history fingerprint |

Invariants, validated on construction: finite arrays, positive spot and vol,
`tenor_grid` strictly increasing and positive, `dates` a business-day
calendar (the history's calendar when bootstrapped, a generated
Mon–Fri calendar with an explicit holiday list otherwise).

`carry[i, d, :]` is the **constant-maturity carry curve** of path `i` on
day `d`: `B(T_k)` at tenors measured from day `d` itself, not to fixed
calendar dates. It is the futures-market analogue of a constant-maturity
yield curve. `B(0) = 0` is an implicit first node. A listed contract's
carry on that day is the curve read at the contract's *actual* remaining
tenor, which shrinks daily, so a contract's basis converges to zero at its
expiry automatically; the generators only model how the constant-maturity
curve moves. The history side builds the same object: `PathHistory`
derives each historical day's curve from that day's listed chain with the
`ForwardCarryCurve` rules (log-forward interpolation between contracts,
flat forward carry beyond the last), so a bootstrap resamples changes of a
consistently defined curve rather than of contracts whose tenors drift.

Every day of a path is a full market snapshot, so the ensemble engine
never needs to look at another path or another day to price.

### 4.2 Lifecycle and hedge state (per path, arrays of length `n_paths`)

`alive`, `knocked_in`, `knocked_out`, `matured`, `settled`,
`observed_ko_mask (n_paths, n_ko)`, `pending_cashflow`, `pending_settlement_day`,
`realized_cashflows`, hedge `contract_id`, `quantity`, `avg_price`,
`multiplier`, `realized_pnl`, `transaction_costs`.

### 4.3 State cube

Float and bool arrays of shape `(n_paths, n_days)` for the replay state
columns that have a per-path meaning: `portfolio_value`, `product_mtm`,
`hedge_mtm`, `cash`, `cashflows`, `transaction_costs`, `product_pnl`,
`hedge_pnl`, `total_pnl`, `spot`, `volatility`, `rate`, `pricing_q`,
`futures_price`, `futures_contracts`, `alive`, `knocked_in`,
`knocked_out`, `matured`, `pending_receivable_pv`, plus the Greeks
`delta`, `gamma`, `pre_hedge_contracts`, `post_hedge_contracts`.
`active_contract` is per day (common to all paths) and stored once.
`EnsembleResults.path_states(i)` assembles a frame in the replay schema.

## 5. Path generators

All generators share `PathGenerator.generate(n_paths, n_days, *, seed) -> MarketPath`
and record every parameter in `MarketPath.meta`. Random draws come from
`numpy.random.default_rng(seed)`; a run with the same seed and parameters is
bit-reproducible. No generator parameter has a hidden default: block
length, drift, vol, tenor grid and calendar are explicit.

### 5.1 Stationary block bootstrap (primary)

Input: a `PathHistory` built from the study's history frames — per day the
spot, the ATM vol at the stated tenor, the flat rate and the carry curve
`B(T_k)` sampled at the tenor grid from the listed chain (log-forward
interpolation, flat forward carry beyond the last contract — the
`ForwardCarryCurve` rules). The state vector of daily changes is

```
u_d = [ ln(S_d/S_{d-1}),  σ_d − σ_{d-1},  r_d − r_{d-1},  B_d(T_k) − B_{d-1}(T_k) for k ]
```

Politis–Romano stationary bootstrap: blocks start at a uniformly random
day and have geometric length with mean `mean_block_days` (required).
Consecutive blocks are concatenated until `n_days` changes are drawn; the
path is integrated from the inception snapshot (`start_state`: spot, vol,
rate, carry curve — either the history's last day or a user-supplied
snapshot). Options, all explicit:

- `demean_returns: bool` and `annual_drift: float` — the return series is
  demeaned and the stated drift added (`annual_drift / 252` per day).
- `vol_floor: float` — vol changes are integrated and floored (a floor
  hit is counted and reported in `meta`, never silent).
- `carry_mode: "changes" | "levels"` — resample the daily *changes* of
  `B(T_k)` (what the example study uses) or the *levels* (a stationary
  basis regime). Required, no default.

The joint dependence of spot, vol and basis is kept because the whole
state vector is resampled by the same block indices.

### 5.2 GBM

`ln S` follows `(μ − σ²/2) dt + σ √dt Z` on the trading calendar with
`dt = 1/252`; `μ`, `σ` required. Vol rule, required:

- `constant`: `atm_vol = vol_level`;
- `sticky_realised(window)`: `atm_vol = a + b · realised_vol(window)`,
  with `a`, `b`, `window` required.

Carry: `carry_schedule` — either a constant curve (the inception curve
held) or a per-day deterministic schedule. Rate: constant `rate`.

### 5.3 Designed paths

`DayPath` from `quantark.dynamicscenario.PathBuilder` (spot, vol, rate
levels or trends by day, `basis_values` for a parallel carry move) converts
to a one-path `MarketPath` on a stated calendar and inception snapshot.
`SnowballStressLibrary` ships named paths: `crash_into_ki(depth, days)`,
`v_shape(depth, days_down, days_up)`, `vol_spike(vol_up, decay_days)`,
`basis_blowout(carry_shift, days)`, `grind_up_to_ko(pct_per_day)`. A
designed set is a list of paths run as one ensemble (n_paths = number of
scenarios) so stress results share every downstream table with the
statistical ensembles.

## 6. Carry curve to listed chain and to `q(T)`

`carry.py` turns each day's simulated curve into the IM chain the hedge
rolls on and the pricer's dividend input, with the rules the replay
engine already applies to real data:

- **Contract calendar**: CFFEX IM listing — the current month, the next
  month and the next two months of the March/June/September/December
  cycle; a contract expires on the third Friday of its month (moved to the
  next trading day if that Friday is a holiday on the path calendar). A
  new contract is listed the trading day after one expires. The calendar
  is a pure function of the date, so it is common to all paths.
- **Prices**: `F_i = S · exp(B(T_i))`, `T_i` ACT/365 from the day to the
  contract expiry, `B(T_i)` read from the path's constant-maturity curve
  for that day (`carry[i, d, :]`, section 4.1) by piecewise-linear
  interpolation of `B` in `T` between the grid tenors, with `B(0) = 0` as
  the first node and the last segment's slope continued beyond the last
  tenor (constant forward carry between grid tenors, i.e. log-forward
  interpolation). `multiplier = 200`.
- **`q(T)` for the pricer**: built from the day's chain by the same rule
  as `ProductReplay._term_dividend` for the configured `dividend_source`
  (`None`/`active_contract`: floored simple-compounded yield from the
  active hedge contract; `futures_curve` with `flat_q`,
  `flat_forward_carry` or `surface_forward_carry` and
  `futures_curve_min_tenor_days`). For `surface_forward_carry` the tail
  source is the simulated curve's nodes beyond the last listed contract
  (the simulated curve plays the option surface's role; no artifact
  exists on a simulated path). The function is shared with the study's
  `dividend_for` so a model name means the same thing on real and
  simulated paths.

The chain builder is exported so the conformance oracle can turn a
`MarketPath` into an `AutocallableMarketDataSet` (spot, vol, rate,
futures frames) for the real replay engine.

## 7. Pricing providers

### 7.1 Interface

```python
class DayStates(NamedTuple):
    day_index: int              # position in MarketPath.dates
    path_index: np.ndarray      # alive paths, shape (m,)
    spot: np.ndarray            # (m,)
    vol: np.ndarray             # (m,)
    q_T: np.ndarray             # (m,) zero yield to remaining maturity
    rate: np.ndarray            # (m,)
    knocked_in: np.ndarray      # (m,) bool

class PathPricer(Protocol):
    def price_day(self, states: DayStates) -> tuple[np.ndarray, np.ndarray]:
        """(pv, delta) per alive path, per unit product."""
    def verify(self, states: DayStates, sample: int, rng) -> GateReport: ...
    def fingerprint(self) -> str: ...
```

Gamma: in repricing mode it is the engine's own (`calculate_greeks`); in
surface mode it is the grid second difference in `x`, and the manifest
labels the column's origin.

### 7.2 `LifeSurfacePricer` (PDE only)

- One `SnowballPDESolver` solve per `(product_fingerprint, vol_bucket,
  q_bucket)`. The solve is requested with a time grid that contains every
  simulation date of the product's life as a node (the grid layer's event
  schedule gains "extra time nodes"; daily nodes for a 1Y product are about
  250, well within the solver's usual `time_steps`). After the solve the
  provider copies `V0(t, x)`, `V1(t, x)` and the grid vectors, and computes
  central first differences in `x` for delta (converted to spot delta by
  `1/S`), the same formula as `_calculate_delta_gamma`.
- Readout for a day: the slice at that date's node, state `V1` where
  `knocked_in` else `V0`, linear interpolation in `x = ln S` at each path's
  spot, vectorised. A spot outside the grid is an error (the grid is built
  wide enough from the vol bucket; the run reports and stops).
- Buckets: `vol_step` and `q_step` required; the bucket centre is the
  solve's vol and `q`. The `q` used for the solve is the day's `q(T)` at
  the product's remaining maturity, flat. This is an approximation of the
  replay's term-structured curve and is measured by the gate.
- Cache: `LifeSurface` objects in an LRU with a byte budget
  (`surface_cache_bytes`, required); eviction is by recency; a hit or
  miss count is reported.

### 7.3 `RepricingPricer` (any engine)

- Engine from `create_pricing_engine(product, AutocallableEngineConfig)`,
  exactly as the replay engine builds its per-product engine; bump sizes
  honoured.
- The product is aged to the day with the tracker's rule (`deepcopy`,
  `maturity −= elapsed`, `barrier_config.time_shift`), once per day (not
  per path), and priced in a `PricingEnvironment` with a flat vol, flat
  rate and the day's dividend object.
- Modes, by `spot_step`:
  - `spot_step=None` (**exact**): every distinct `(spot, vol, q, knocked_in)`
    state prices through `engine.calculate_greeks`; the result is identical
    to what the replay loop would compute for that path on that day.
  - `spot_step>0` (**ladder**): the day's alive spots are covered by a
    ladder `S_j = S_ref · exp(j · spot_step)` spanning the alive range with
    one node beyond each end; each `(ladder node, vol bucket, q bucket,
    knocked_in)` is priced once; PV and delta are linear interpolations
    between the two neighbouring nodes. Vol and `q` are bucketed by
    `vol_step`/`q_step` (may be zero, meaning exact).
- MC engines: the seed of every priced state is
  `hash(state_key) mod 2**32`, so a recomputed state matches its cached
  value bit for bit and the run manifest records the policy.

### 7.4 `StateCache`

Key: `(product_fingerprint, day_index, knocked_in, spot_key, vol_key,
q_key, engine_fingerprint)` where the `*_key` are integer bucket indices
(or the float bits when the step is `None`/zero). Value: `(pv, delta,
gamma)` as float64. Two tiers:

- in-memory LRU per process with `memory_bytes` (required);
- optional on-disk store (`disk_dir`): one append-only `npz` shard per
  `(product_fingerprint, engine_fingerprint)`, keyed by the remaining
  tuple, loaded lazily. A shard whose header library version or engine
  fingerprint differs is ignored (a miss), never reinterpreted.

Reuse rules: the key holds nothing about the hedge, cost or strategy, so
cells that differ only there share entries; the runner orders work as
`for path batch: for cell:` so the batch's states are priced once for all
cells. Deduplication within a day is by unique keys.

### 7.5 Gate

`GateConfig(sample_states: int, pv_tolerance_bp: float,
delta_tolerance_hands: float)` — all required. Before a cell runs, and
again on a random sample of states visited during the run, the provider
reprices the sampled states directly with the engine at the exact spot,
vol and `q` (a fresh aged solve for the surface provider) and reports the
max gap in PV (bp of notional) and delta (hands). A failed gate aborts the
cell with the report; no fallback to another mode. In exact repricing
mode the gate is identically zero and recorded as such.

## 8. The vectorised daily loop

`EnsembleBacktestEngine.run(paths: MarketPath) -> EnsembleResults`. For
each day `d`:

1. **Chain and roll.** The day's chain (common) from `carry.py`; the roll
   policy (`front`: nearest contract with more than `roll_days_before_expiry`
   days left unless the current one still qualifies; `far`: longest listed,
   same stickiness) picks the active contract. Paths holding the old
   contract close at its price and open the new one: two trade legs, both
   costed, average-cost realised P&L booked. Prices differ per path (they
   come from the path's curve), so the roll is vectorised over paths with a
   common contract id.
2. **Dividend input.** `q(T)` per path from the chain per the configured
   rule; `pricing_q` recorded at the remaining maturity.
3. **Initial book value** on day 0, before lifecycle: `initial_price` if
   the `ReplayProduct` sets it, else the provider's PV at the day-0 state,
   times quantity.
4. **Lifecycle.** For alive paths: KO on a due observation where the close
   is at or above the barrier (`disable_ko_after_ki` honoured; observation
   index marked observed once whether hit or not); KI on every day for
   continuous monitoring or on due KI observations for discrete, where the
   close is at or below the barrier (strict/inclusive comparison taken from
   `_barrier_hit`); maturity payoff via `product.get_payoff(spot, env,
   knocked_in)` when the terminal timing is due. A KO or maturity cashflow
   is `quantity · payoff` with the settlement date from the schedule
   record; until it lands it is carried as `pending_receivable_pv`
   (discounted at the path's rate); on the settlement day it moves to
   `realized_cashflows` and the path is `settled`. Observation "due" dates
   are resolved once from the product schedule on the path calendar with
   the tracker's own resolver, so date rounding matches the replay.
5. **Pricing.** Alive paths through the provider; dead paths carry
   `pv = delta = 0`.
6. **Hedge.** `target = round(hedge_ratio · (target_delta − delta · quantity)
   / multiplier)`, the `AutocallableDeltaHedgeStrategy.target_contracts`
   formula (rounding off when `round_contracts` is false); trade when `|target − current| > delta_threshold`; when the
   product is dead, close to zero with `trade_type = hedge_close`. Trades
   at the active contract's path price; proportional or zero cost model;
   ledger identical to `FuturesHedgePosition`.
7. **Accounting.** `hedge_mtm = realized_pnl + quantity · (F − avg_price) ·
   multiplier`; `product_pnl = product_mtm + receivable_pv + cashflows −
   initial_book_value`; `total_pnl = product_pnl + hedge_mtm − costs`;
   `cash = cashflows − costs`; `portfolio_value = product_mtm + hedge_mtm +
   cash + receivable_pv`. The identity `total_pnl == portfolio_value −
   initial_book_value` is asserted every day in tests.
8. **Record** the state cube row for the day; trades and lifecycle events
   are appended to compact event lists `(path, day, type, contract,
   quantity, price, cost)` and `(path, day, event, index, spot, barrier,
   cashflow)`.

A settled path stops updating (its columns repeat the terminal values,
matching the replay's `terminate_on_lifecycle_end` semantics per path).
The run ends when every path is settled or the calendar is exhausted; a
path that ends with an open receivable is `data_end`, never relabelled.

## 9. Results

`EnsembleResults`:

- `summary: DataFrame`, one row per path: `terminal_pnl`, `terminal_pnl_bp`,
  `daily_pnl_std_bp`, `variance_reduction_r2`, `max_drawdown_bp`,
  `turnover`, `cost_bp`, `roll_day_mtm_jump_bp`, `other_day_mtm_jump_bp`,
  `delta_churn`, `termination_reason`, `days`, `knocked_in`,
  `ko_observation_index`. Definitions are those of the study's
  `hedge_measures`, moved into the library so both use one implementation.
- `distribution(measure) -> dict`: mean, std, quantiles 1/5/25/50/75/95/99,
  expected shortfall at `es_level` (required), share positive, KO and KI
  frequencies.
- `paired(other: EnsembleResults)`: matched-path differences (same
  `MarketPath` fingerprint required) with mean, share positive and t-stat.
- `path_states(i) -> DataFrame` in the replay state schema;
  `path_trades(i)`, `path_events(i)`.
- `manifest`: generator meta and seed, product fingerprints, engine and
  provider fingerprints, cache statistics, gate reports, library version,
  timing.
- `to_dir(path)` / `from_dir(path)` persist the cube (`npz`), summary and
  manifest.

## 10. Conformance oracle

`conformance.run_oracle(config, market_path, path_index) -> OracleReport`:

1. Builds an `AutocallableMarketDataSet` from the single path (spot, vol,
   rate frames and the chain frame from `carry.py`).
2. Runs the real `ReplayBacktestEngine` with the same products, hedge
   spec, strategy, cost model and `AutocallableEngineConfig`, with the
   configured engine (PDE for a surface cell, the cell's engine otherwise).
3. Compares day by day: lifecycle flags, `active_contract`,
   `futures_contracts` and every trade must match exactly; `product_mtm`,
   `delta`, `total_pnl` must match within the gate tolerances (and to
   float rounding in exact repricing mode).

The oracle runs in the test suite on synthetic paths (short products,
PDE engine) and is a CLI (`python -m quantark.backtest.simulation.conformance
--run-dir ... --path 17`) for spot checks of real runs.

## 11. Configuration

```python
@dataclass
class EnsembleConfig:
    products: list[ReplayProduct]           # standard snowballs only (validated)
    engine_config: AutocallableEngineConfig # engine type/params, dividend source
    hedge: HedgeSpec                        # kind="futures", roll policy, multiplier
    strategy: AutocallableDeltaHedgeStrategy
    transaction_cost_model: TransactionCostModel
    pricing: PricingProviderConfig          # provider + steps + cache + gate
    rate_schedule: float | Callable[[pd.Timestamp], float]
    es_level: float                         # e.g. 0.05
    workers: int                            # process pool for path batches
    batch_paths: int                        # paths per batch
    metadata: dict

@dataclass
class PricingProviderConfig:
    provider: Literal["life_surface", "repricing"]
    vol_step: float | None
    q_step: float | None
    spot_step: float | None                 # repricing ladder; None = exact
    cache: CacheConfig                      # memory_bytes, disk_dir | None
    gate: GateConfig                        # sample_states, tolerances
```

Validation (all `ValidationError`, fail closed): a non-snowball product;
`life_surface` with a non-PDE engine type; `spot_step` given with
`life_surface`; a term dividend source with `fixed_dividend_yield`; a
missing tolerance; `workers < 1`; a `MarketPath` whose calendar does not
cover the product's life plus the longest settlement lag (reported as
`data_end` risk, allowed only with `allow_data_end=True`).

## 12. Error handling

Fail closed everywhere: an engine error on any state aborts the batch
with the state key in the message; a gate failure aborts the cell; a cache
shard with a foreign fingerprint is a miss; a spot outside a surface grid
aborts; a designed path shorter than the product's life is rejected
unless `allow_data_end`. No silent substitution of providers, engines,
seeds or tolerances.

## 13. Testing

TDD throughout; each item is a failing test first.

- **Generators**: seed reproducibility; block structure (a mean block
  length of 1 reproduces iid resampling, a large one reproduces the
  history in order); demean + drift give the stated mean; vol floor counts;
  GBM moments; designed path conversion.
- **Carry to chain**: contract calendar on known dates (third Fridays,
  quarterly cycle, listing after expiry); chain prices reproduce the curve;
  `q(T)` equals the study's `dividend_for` for each model on a synthetic
  day.
- **Life surface**: on a flat market the readout at every date equals a
  fresh aged solve within the gate; KI state switches the surface; delta
  matches the engine's; grid coverage error.
- **Repricing**: exact mode equals `engine.calculate_greeks` per state;
  ladder interpolation within tolerance; cache hit on a repeated state;
  disk round trip; MC seed determinism.
- **Lifecycle**: KO at the observation (and not a day earlier), KI then
  maturity with the KI payoff, settlement lag booking and receivable PV,
  `disable_ko_after_ki`; each case cross-checked against
  `AutocallableLifecycleTracker` on the same hand-built path.
- **Hedge**: random trade sequence against `FuturesHedgePosition`; roll
  legs and costs; rounding and threshold; hedge close on death.
- **Accounting identity** every day of every test run.
- **Conformance**: an end-to-end synthetic ensemble (PDE engine, 6-day
  product like the replay tests) where every path matches the replay
  engine in exact repricing mode bit for bit, and the surface mode within
  the gate.
- **Results**: measures equal the study's `hedge_measures` on a known
  path; distribution quantiles; paired differences on matched paths.

## 14. Example study — `example/snowball_simulated_paths/`

Reuses the term-structure study's product terms, fair-coupon solver and
model catalogue (`example/snowball_q_term_structure/_common.py`):

1. `01_build_paths.py`: bootstrap 2,000 paths of 260 trading days from the
   history (mean block 20 days, demeaned, stated drift 0), plus the stress
   library; persist the `MarketPath` files and their fingerprints.
2. `02_ensemble_fleet.py`: cells = {flat_active, term_flat_q, term_opt_tail}
   × {front, far} hedges, surface provider with PDE, repricing provider
   with QUAD on a 200-path subset as the engine-substitute check, gates
   recorded; conformance spot check on three paths per cell.
3. `03_report.py`: hedge-cost distributions per cell (mean, quantiles,
   expected shortfall), paired differences on matched paths, stress table,
   the historical study's realised path located inside the simulated
   distribution, and an HTML report.

## 15. Performance expectations

For a 1Y snowball, 2,000 paths, daily hedging:

- surface provider: a few hundred PDE solves (about 1–3 s each on the
  study's grid) plus vectorised daily work of a few milliseconds per day
  per cell — minutes per cell;
- repricing provider, QUAD, ladder `spot_step = 0.25%`: a few dozen unique
  states per day per bucket, roughly 100–300 engine calls a day, tens of
  minutes per cell on four workers; exact mode scales with alive paths
  (2,000 calls a day, hours) and is meant for validation subsets.

These are estimates to be measured in the example, not commitments.

## 16. Delivery order

1. `MarketPath`, generators, carry-to-chain, tests.
2. `RepricingPricer` exact mode + `StateCache` memory tier; lifecycle and
   hedge vectors; `EnsembleBacktestEngine`; the conformance oracle in exact
   mode (bit-for-bit target). This is the first usable engine.
3. Ladder mode, disk cache tier, gate.
4. `LifeSurfacePricer` (extra time nodes in the grid layer, readout,
   surface cache) and its gate.
5. Results distributions, paired analysis, persistence.
6. Example study and report; README in the package (`README.md`, since
   `CLAUDE.md` files do not ship).

## 17. Decisions taken during design

- Purpose priority: hedge-cost distribution, then model-risk comparison,
  then stress. Robustness of historical results is served by the
  bootstrap generator.
- Simulated factors: spot, ATM vol and the futures carry term structure;
  rates deterministic.
- Families: block bootstrap, GBM with real-world drift, designed paths;
  parametric stochastic vol deferred.
- Scale: exact pricing with a cache is the reference procedure; the PDE
  life surface is the accelerator; both are gated against direct
  repricing. Engine substitutes (QUAD, MC, analytical, vol-model engines)
  must be supported through the repricing provider even though they
  cannot offer a life surface.
- Scope: autocallable book through the replay engine's semantics, futures
  delta hedge; no option hedges.
- Placement: library module `quantark.backtest.simulation` plus an example
  study.
- Approach: vectorised ensemble engine (B), with the replay engine as the
  conformance oracle rather than as the runner.
