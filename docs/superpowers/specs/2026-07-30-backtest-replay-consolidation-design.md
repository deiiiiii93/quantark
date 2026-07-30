# Design: Backtest Module Consolidation — Product Replay as a First-Class Package

Status: requirements locked with user on 2026-07-30.

Runs **before** the snowball vol-model study fleet
(`docs/superpowers/specs/2026-07-30-snowball-volmodel-backtest-040-rebaseline-design.md`);
its §10 step 0 (commit the study framework scope) is a precondition of this work.

---

## 1. Problem

`quantark/backtest/otc/` grew into a ~5,000-line module that duplicates
infrastructure and violates the project's own engineering rules:

- Two parallel ~500-line daily loops (`AutocallableBacktestEngine`,
  `BookAutocallableBacktestEngine`) whose equivalence is a docstring claim
  ("a book of a single product is byte-identical"), not a construction.
- Silent fallbacks: `_replay.calculate_greeks` returns `delta=0.0` on any
  exception; `_calculate_event_stats` silently swaps in a hardcoded
  5000-path MC engine. Both contradict the fail-closed / exact-semantics
  conventions this repo enforces elsewhere.
- A mutable engine swap (`self.pricing_engine` reassigned per day, manually
  re-synced onto the replay, with a "do not rely on it" comment).
- ~70 lines of hand-rolled bump-greeks duplicating the engine layer's
  `BumpConfig` machinery.
- Model-calibration infrastructure (`vol_calibrators.py`, 687 lines;
  `vol_history.py`, 468 lines) trapped behind a backtest import path.
- Futures position accounting duplicated against
  `equity/multi_hedge_executor.py`.
- Implicit record schemas spread across three files; no shared performance
  metrics; replay continues pricing dead contracts to calendar end.

A full merge into the root strategy-simulator engines was considered and
rejected: the root engines vary *strategies* over a fixed book; the replay
engine varies *pricing models* over a fixed strategy. They share protocols
(`BaseBacktestEngine`/`Results`, `get_backtest_engine`), transaction costs,
and the lifecycle core — that seam is correct and stays.

## 2. Decisions locked (owner, 2026-07-30)

1. **Scope: full package** — all consolidation, correctness, and efficiency
   items below land before the study fleet runs (the 0.4.0 re-baseline
   discards prior results and re-runs every gate anyway).
2. **Compatibility: shims + deprecations.** Public names keep working;
   moved modules leave `DeprecationWarning` re-export shims (pattern:
   `quantark/_compat.py`). Shims are dropped at 0.5.0.
3. **Relocation split:** `vol_calibrators.py` → `quantark/volmodels/`;
   `vol_history.py` → `quantark/param/vol/`.
4. **Structure: approach B** — a new `quantark/backtest/replay/` subpackage
   is the canonical home; `quantark/backtest/otc/` becomes a pure shim
   package.

## 3. Package layout

```
quantark/backtest/
├── futures_ledger.py        # FuturesHedgePosition + FuturesRollPolicy (shared)
├── metrics.py               # PerformanceMetrics over the BaseBacktestResults protocol
├── replay/
│   ├── engine.py            # ReplayBacktestEngine — THE daily loop (multi-product book)
│   ├── config.py            # ReplayBacktestConfig, ReplayProduct (was BookProduct),
│   │                        #   HedgeSpec, AutocallableEngineConfig,
│   │                        #   VolModelCalibrationConfig, SurfaceGridConfig
│   ├── single.py            # AutocallableBacktestEngine/Config — book-of-one wrapper
│   ├── product_replay.py    # ProductReplay (was otc/_replay.py)
│   ├── market.py            # AutocallableMarketDataSet, basis helpers, AKShare adapter
│   ├── engine_factory.py    # unchanged role
│   ├── results.py           # ReplayBacktestResults + single-product view; .metrics
│   ├── schema.py            # TypedDict row schemas — single source of truth
│   └── dashboard.py
├── strategy/
│   └── futures_delta_strategy.py  # AutocallableDeltaHedgeStrategy, subclassing BaseStrategy
└── otc/                     # shim package only; dropped at 0.5.0

quantark/volmodels/calibration.py      # from otc/vol_calibrators.py
quantark/param/vol/surface_history.py  # from otc/vol_history.py
quantark/util/io.py                    # _atomic_write_json (currently a private
                                       #   cross-import from vol_calibrators)
```

- Shim modules (verified importers: `backtest/base.py`, 3 examples, 8 test
  files, 2 stage tests): `otc/__init__.py` plus per-module shims for
  `config`, `engine`, `book_engine`, `market`, `engine_factory`, `state`,
  `results`, `dashboard`, `vol_calibrators`, `vol_history`, `_replay`.
- Deprecated aliases: `BookAutocallableBacktestEngine`,
  `BookAutocallableBacktestConfig`, `BookProduct`, `BookBacktestResults`.
- In-repo consumers (stages 11/12/13, examples, tests) move to canonical
  paths in the same change; one dedicated compat test keeps importing the
  old paths and asserts the `DeprecationWarning` until 0.5.0.
- `get_backtest_engine` gains `ReplayBacktestConfig` dispatch;
  `AutocallableBacktestConfig` dispatch is preserved.
- Calibration cache safety: keys are
  `sha256(surface_sha|variant|config_fingerprint)` with a canonical-JSON
  fingerprint and `_CACHE_SCHEMA_VERSION` — no module paths — so the
  existing 554-entry on-disk cache remains valid across the relocation.
  A test asserts key invariance.

## 4. Engine unification

`ReplayBacktestEngine` (evolved from the book engine) becomes the only
daily loop. It absorbs the three capabilities only the single engine has
today:

1. **Per-day vol-model calibration.** `_calibrate_day` moves into the
   unified loop: one calibration per day per variant (the surface artifact
   is shared across products) producing one `CalibratedVolModel`; each
   product's day-engine is then constructed from the frozen model via
   `create_vol_model_engine`. Calibration is the cached expensive step;
   engine construction is cheap.
2. **`initial_product_price`** — maps onto `ReplayProduct.initial_price`.
3. **Calibration records** (incl. `pricing_seconds`) — carried into
   `ReplayBacktestResults`.

`AutocallableBacktestEngine` (`replay/single.py`) becomes: build a
`ReplayBacktestConfig` with one `ReplayProduct`, run the unified engine,
present single-product results.

**Frozen constraint:** a single-product run must produce **byte-identical
output** to today's single engine. Today's single-engine row schemas are
canonical; the unified engine emits them (book-level rows are the same
schema aggregated). Reconciliation details are plan-level; the gate is
byte-identity on the §9 goldens.

## 5. Day loop: no mutable engine state

The engine-of-the-day flows as an explicit argument. `ProductReplay`
methods that price (`calculate_greeks`, `record_surfaces`, initial pricing)
take `engine` as a parameter; the mutable `pricing_engine` attribute and
its reassign-and-resync pattern are deleted. Exactly one place per day
resolves the engine (classic BSM → factory engine; vol-model → the day's
calibrated engine).

## 6. Error handling: fail-closed, opt-in fallbacks

- **Greeks:** the `except Exception → delta=0.0` path is deleted. A pricing
  failure raises; the fleet runner already records per-run tracebacks.
- **Event stats:** new config field
  `event_stats_fallback: Literal["none", "mc"] = "none"` on
  `AutocallableEngineConfig`. `"none"` → failure raises. `"mc"` → the MC
  fallback runs, is logged, and affected rows record provenance in a new
  `event_stats_engine` column.
- **Surface recording:** NaN rows are kept (visible failure) but every
  failure is logged with its exception; no bare `except: pass`.

## 7. Greeks path

The replay-side manual bump code is deleted. `delta_bump_size` /
`gamma_bump_size` overrides translate into the engine layer's `BumpConfig`
at factory time; the replay always calls `engine.calculate_greeks`, and the
engine decides native-grid vs bump as it does everywhere else in quantark.
If `BumpConfig` cannot express asymmetric delta-vs-gamma bumps, the plan
resolves it engine-side — never by resurrecting replay-side bumping.

## 8. Records, termination, metrics, efficiency

**Schemas.** `replay/schema.py` defines `TypedDict`s for every row type
(`StateRow`, `GreekRow`, `TradeRow`, `RebalanceRow`, `ActionRow`,
`SurfaceRow`, `DailyEventRow`, `EventProbRow`, `CalibrationRecord`) with
column order as the single source of truth, documented in the module
docstring. No runtime validation layer.

**KO termination** (module-level implementation of study-spec §6). New
config flag `terminate_on_lifecycle_end: bool = True`. This is **not** a
bare loop stop: the current tracker posts the whole KO payoff to
`realized_cashflows` on the observation date and does not retain
`settlement_time`, so delayed or `CouponPayType.EXPIRY` settlement cannot
be expressed by truncation alone. The design introduces an explicit
**pending-settlement state**, separating four moments:

1. **Economic termination** — the terminal observation date: product dead,
   pricing/greeks/event-stats stop (as today), hedge closes here.
2. **Receivable valuation** — between observation and settlement the
   terminal cashflow is carried in portfolio value as a discounted
   receivable, `cashflow × df(t_settle)` off the day's rate curve.
3. **Cash posting** — `realized_cashflows` is credited on the resolved
   settlement date, not the observation date. The tracker is extended to
   surface each terminal event's `settlement_time` (the product's KO
   records already carry it, `snowball_option.py:1041`).
4. **Replay stop** — the run ends at
   `date_resolver(max(observation_date, settlement_date))`; if that date
   lies beyond the market data, the run ends at data end with
   `termination_reason="data_end"` and the receivable still open in the
   summary.

Under the study's current term sheet settlement resolves to T+0, so moments
1–4 coincide and behavior degenerates to "stop on the KO date with cash
posted" — but the rule is stated generally so a settlement lag or
EXPIRY-paid KO does not silently change the ledger. Book semantics:
terminate when **all** products are settled. KI never terminates.
`get_summary()` gains
`termination_reason ∈ {ko, ki_maturity, maturity, data_end}`,
`days_replayed`, `days_in_contract`. Default ON; the flag exists for golden
comparison and diagnostics. Tests cover T+0 (degenerate), a synthetic T+5
lag, and an EXPIRY-paid KO (dead-but-unsettled run to maturity).

**Metrics.** The existing `PerformanceMetrics` is **not** protocol-clean:
it also reads `num_hedges`, `state_tracker`, `get_delta_series()`,
`total_transaction_costs`, `initial_value`/`final_value`, and
`config.strategy.target_delta`. It is therefore **split**, not moved
wholesale:

- `backtest/metrics.py` gains `CorePerformanceMetrics`, consuming only the
  `BaseBacktestResults` protocol (`get_pnl_series`/`get_value_series`/
  `get_hedge_trades`): Sharpe, drawdown (+duration), VaR/CVaR, volatility,
  win rate, profit factor, skew/kurtosis.
- Equity's `PerformanceMetrics` stays in `equity/metrics.py`, now extending
  `CorePerformanceMetrics` with the equity-only hedge/delta metrics — its
  public API is unchanged, so no shim and no equity breakage.
- `ReplayBacktestResults.metrics` returns `CorePerformanceMetrics`.
- Contract tests exercise every public metric on both equity and replay
  results. FI-specific metrics stay in `fi/`.

**Efficiency.**

- KO termination removes 61% of study replay-*days* (measured, study spec
  §7.2) — but post-KO days are already cheap because pricing stops at
  `alive=False`; the saving is loop iteration, env construction, futures
  selection, and record rows, not pricing. The primary benefit is
  record correctness, and the cost claim is stated accordingly.
- `_env_with_spot` shallow-copy helper (curves/surfaces shared — they are
  immutable by project convention) replaces `deepcopy(env)` in the
  surface-grid recorder; the greeks path inherits engine-side copying by
  construction (§7).
- Futures chain pre-grouped by date once in `AutocallableMarketDataSet`.

**Out of scope, recorded so they are not re-proposed:** reading the spot
ladder off the PDE grid (measured ~zero benefit on the PDE route due to
layout caching); intra-day parallelism (fleet-level process parallelism is
the right axis); migrating `equity/multi_hedge_executor.py` onto
`futures_ledger.py` (a later package — the ledger is placed at
backtest-common level now so that migration needs no further moves).

## 9. Testing and golden gates

1. **Golden capture precedes any refactor**, three goldens:
   - the pinned stage-12 quick run (`--quick --max-inceptions 1`,
     `flat_bsm` + `ts_bsm`, `terminate_on_lifecycle_end=False`);
   - **a calibrated-variant golden** — `localvol` on the PDE route (fully
     deterministic), one inception, short window. The bsm-only quick run
     never touches `VolModelCalibrator`/`create_vol_model_engine`, so
     without this golden the riskiest ported capability
     (calibrate-before-any-pricing ordering, per-day engine swap,
     calibration records) would go unexercised. The golden additionally
     asserts calibration call count and surface selection per day;
   - the existing `test_otc_autocallable_backtest` / `test_book_backtest`
     fixtures.
2. **Refactor gate — canonical comparison contract.** "Byte-identical"
   means: for **every** deterministic result surface — states, greeks,
   trades, rebalances, actions, surfaces, daily-event summary, event
   probabilities, calibration records, and `get_summary()` —
   `assert_frame_equal(check_exact=True)` plus explicit column-name and
   column-order equality (dict/scalar outputs compared by `==` after
   ordering). Wall-clock fields (`pricing_seconds`,
   `calibration_seconds`) are excluded by name; `cache_hit` is kept but the
   golden runs pin a fresh in-memory cache so its sequence is
   deterministic. Book-of-one ≡ single becomes a permanent test under the
   same contract, not a docstring claim.
3. **New behavior tests:** greeks failure propagates;
   `event_stats_fallback` opt-in + provenance column; termination
   semantics per §8 (T+0 degenerate, synthetic T+5, EXPIRY-paid KO;
   reason/day counts; receivable and cash-posting dates); schema
   column-order stability; calibration cache-key invariance across
   relocation.
4. **Behavioral legacy compatibility**, not import-only: the old book API
   exposes *callable* accessors (`results.trades_df()`) where the single
   results use *properties*, and `products_meta` in its constructor —
   legacy adapters must preserve constructor signatures and
   method-vs-property shapes exactly. The compat test constructs configs
   through the old paths, runs both engines end-to-end, touches **every**
   public result accessor, verifies root `quantark.backtest` exports, and
   asserts the `DeprecationWarning`s.

## 10. Sequencing

1. **Precondition** — study spec §10 step 0: commit the uncommitted study
   framework scope, and only that scope, on a feature branch. This work
   branches from that commit in its own worktree.
2. Capture goldens → structural moves (shims, relocations) → engine
   unification → behavior changes (fallbacks, termination). Byte-identity
   gates bracket the risky middle; behavior changes land last so their
   effects are isolated in the diff.
3. After merge: study gates G4/G1/G2 re-run on the consolidated module
   (they re-run on 0.4.0 regardless); `CHANGELOG`, `backtest/CLAUDE.md`,
   and the root `CLAUDE.md` layout table are updated; the study spec's §6
   termination requirement is satisfied by the module feature.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Byte-identity fails between unified and legacy single engine | Goldens captured first; unification lands as its own commit against them; any diff is a bug to fix, not a tolerance to widen |
| Book engine lacks calibrator support today; porting it introduces subtle ordering differences (calibrate-before-any-pricing invariant) | The invariant is stated in §4; the dedicated **calibrated-variant golden** (§9.1, `localvol`/PDE) exercises exactly this path — the bsm-only quick run does not |
| Pending-settlement state (§8) is new machinery in the terminal path | Degenerate T+0 case must reproduce legacy cash timing exactly (golden-gated); T+N and EXPIRY paths are new-behavior tests, unreachable under the study term sheet |
| Relocation invalidates the on-disk calibration cache | Keys verified path-independent; explicit invariance test (§9.3) |
| Another session's WIP in the shared tree | §10 step 0 commits a strict file scope first; this work proceeds in an isolated worktree |
| Behavior changes (termination, fail-closed) alter study numbers | Intended — they land before gates G4/G2 re-run, so gates certify the final configuration |
| Shim removal at 0.5.0 breaks external users | Deprecation warnings live for the whole 0.4.x line; changelog documents the mapping |
