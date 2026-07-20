# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).
During 0.x the public API may still change between minor versions.

## [0.3.0] - 2026-07-20

First release of the `quantark.execution` framework kernel and the
`quantark.execution.greeks` scenario layer. The `test_snowball_quad_flat_identity_golden`
failure predates this program and remains quarantined (reproduces on
unmodified main). The spec §20 controlled-host performance gates in
`docs/execution/README.md` are documented as release-preparation evidence
and were not re-measured on the release host.

### Added
- **`quantark.execution.greeks` — greek bumps as scenario cells** (spec
  2026-07-20): `TradeState` per-trade base, same-type `greek-bump/v1`
  transformer with real spot/vol/rate/div/time mutation attribution,
  `greek-value/v1` float runner, and assemblers mirroring
  `EquityPosition.get_trade_risk` / `GreeksCalculator.
  calculate_numerical_greeks` operation-for-operation (bitwise unit gate).
- **`PricingSession.run_scenario_plans`** — many `(base, specs)` plans
  packed through ONE bounded-window process pool (per-cell worker-spec
  payloads; the portfolio × bumps shape), with a per-plan error boundary
  under `collect_errors=True` (base-resolution/planning failures become
  aligned typed `PricingFailure`s while other plans execute).
- **`quantark.execution` — composable execution kernel** (framework contract
  v1; spec `docs/superpowers/specs/2026-07-15-mc-pde-performance-generalization-design.md`).
  `PricingSession` wraps every exported MC/PDE engine without changing direct
  legacy behavior: adapter registry, resource budgets/leases, prepared-artifact
  and session draw caches, reproducibility manifests and immutable diagnostics.
  Capabilities: fixed-batch MC backends (serial/threads, bit-identical
  reductions in canonical order), adaptive RQMC session mode (bitwise vs
  direct across the 12 autocallable vol engines), PDE preparation artifacts
  (grids, step coefficients, factorization packs; one-solve PV/event/grid
  outputs across 18 prepared engines + FX LV), typed scenario execution with
  verified mutation footprints, spawn-safe `WorkerSpec` process workers, a
  Dask backend over the same plans, `price_many` grouping, and a
  complete-payload scenario validator. Published artifacts: generated
  [capability matrix](docs/execution/capability-matrix.md) (CI-fresh), policy
  guide, legacy-internals rationale, reproducibility JSON Schemas (validated
  against live payloads), and runnable migration examples. Adoption is
  inventory-driven and honest: `temporary_legacy` rows remain, and no
  universal framework support is claimed until the controlled-host gates pass.
- **DCN (Digital Coupon Note)**: product + observation-schedule generator,
  curve-aware MC engine with leg decomposition and event stats, two-surface
  PDE engine with MC↔PDE cross-validation gates, LocalVol/Heston/QE/SLV DCN
  MC engines sharing one payoff kernel, MLMC-style coupled timestep-ladder
  Heston pair, SVI/cleaned-quote calibration layers with no-arbitrage and
  repricing-residual reports, node-bump vega buckets and carry-invariant rate
  bumps.
- **Implied futures carry**: `IndexFuturesCurve` implied q(T), futures-tenor
  delta/rhoq buckets, delta-one futures rhoq, portfolio aggregation and hedge
  integration.
- **Engine term-structure upgrade**: MC, PDE, and quadrature autocallable
  engines consume full r/q/vol term structures with a cross-family term
  agreement gate.
- **Vol-model barrier exotics**: LV/Heston/SLV barrier MC pricers on a shared
  monitoring core, LV 1D barrier PDE, Heston/SLV 2D barrier ADI solvers.
- **Vol-model numerics program (phases 2–5)**: vectorized Lewis Heston
  calibration (~21x), unified Heston/SLV ADI core, opt-in Krylov and TR-BDF2
  Fokker–Planck marches, opt-in QE-M martingale correction, concentrated
  (x,v) grids, degenerate v=0 Feller boundary, LV Rannacher start-up
  (default on), opt-in QMC sampling for Heston/SLV/LV kernels.
- **PDE event-stats API** on Snowball/Phoenix vol solvers with event
  distributions surfaced through session outputs.

### Changed
- DCN MC engine performance: draw cache, batched threads, LV build hoist.
- The legacy autocallable Dask batch loop (Snowball vanilla/KO-reset,
  Phoenix) is consolidated into one shared reducer
  (`autocallable_dask_batch`) with byte-preserved behavior, gated by frozen
  bitwise goldens. The legacy `use_dask` route itself is unchanged and
  preserved (see `docs/execution/internals-and-legacy.md` for the §17.3
  removal preconditions).
- Dev extras now include `dask[distributed]` and `jsonschema`.

### Fixed
- `PhoenixMCEngine` silently disabled its Dask parallel path on modern dask
  (`from dask.compute import compute` no longer resolves); availability now
  matches `SnowballMCEngine`, with a regression test.
- Processes scenario backend: a parent whose `__main__` cannot be
  re-imported by spawn children (stdin/heredoc scripts record
  `__file__ = '<stdin>'`; deleted script files) now fails closed with a
  typed `CapabilityError` before any worker is spawned, instead of every
  child dying at bootstrap and surfacing as an opaque
  `BrokenProcessPool` with zero completed cells.
- Craig–Sneyd ADI corrector order restored (base `Y0` + implicit `-rU`),
  QE sampler sign bug, and the 2026-07 PDE/QUAD audit fixes (13 findings)
  are included via the programs above.

## [0.2.5] - 2026-07-03

### Fixed
- 2026-07 PDE/QUAD audit: 13 numerical findings fixed (discrete-monitoring
  boundary handling, Rannacher terminal off-by-one, BGK state resolution,
  smoothed KI for Phoenix/KO-reset, shared theta damping schedule) with
  solver-family consolidation and regression tests.

## [0.2.4] - 2026-07-02

### Added
- Opt-in BGK continuous-KI mode for Snowball and Phoenix PDE solvers
  (direction-aware shifted barrier, per-product flag).
- `EquityPosition.get_trade_risk` (product + total, frozen one-loop) and
  `AutocallableCashLeg.time_shift`; event streams selected by leg
  requirements.

### Fixed
- Portfolio vega/rho/dividend-rho aligned with the canonical greek
  convention; bump contexts freeze PDE critical points for grid steadiness.

## [0.2.3] - 2026-07-01

### Added
- Cross-engine-consistent `ki_ever` / `ki_survive` event-stat fields.

### Fixed
- Theta advances by business day; quadrature event-stats recursion uses
  smoothed barrier indicators.

## [0.2.2] - 2026-07-01

### Fixed
- PDE numerical Greeks: freeze the base spatial domain for finite-difference
  bump repricing so rho and dividend rho measure market sensitivity without
  contamination from auto-grid/domain movement under bumped rate, dividend,
  volatility, or time inputs.

## [0.2.1] - 2026-06-30

### Fixed
- `PhoenixPDESolver`: apply a KO observation scheduled exactly at maturity.
  The inherited grid builder stores the maturity KO in `_ko_terminal_record`
  (intentionally kept out of `_ko_observation_indices`), but the Phoenix
  `_solve` override looked the terminal KO up in `_ko_observation_indices`
  and therefore dropped it — mispricing products with a terminal KO date by
  several percent versus the quadrature and Monte Carlo engines. Terminal KO
  is now applied after the terminal coupon/KI jumps (matching
  `SnowballPDESolver._solve`), routed through `_apply_ko_jump_vector` so the
  same-date coupon-at-KO payoff is preserved.

## [0.1.2] - 2026-06-13

### Added
- Credit dual-measure framework: a recovery convention layer
  (`quantark.asset.credit.conventions`, `STANDARD_RECOVERY=0.40`) that
  separates the canonical shared-curve **hazard01** factor from the
  recovery-converted **CS01** used by products and SIMM. Curve shocks stay
  in hazard space; spread stresses convert through the recovery convention.
- Single-name CDS **roll-down / as-of pricing** via effective and maturity
  dates (seasoned and forward-start), with `schedule_asof` and a
  total-return coupon cash ledger threaded through the dynamic-scenario and
  backtest engines. SIMM buckets the remaining tenor. (Basket as-of is
  deferred.)

## [0.1.1] - 2026-06-11

### Added
- `SnowballQuadEngine`: explicit `ki_monitoring_mode` on `QuadParams`
  (`KnockInMonitoringMode`). `EXACT_DISCRETE` (default) prices every KI
  observation date exactly with adaptive spatial-grid refinement.
  `BGK_APPROXIMATION` is an opt-in performance mode that replaces a dense
  discrete KI schedule with continuous monitoring at a
  Broadie-Glasserman-Kou shifted barrier; the engine validates approximately
  regular spacing (median-band dispersion test), a constant resolved
  barrier, full-horizon coverage, stable volatility, and a minimum schedule
  density (`bgk_min_ki_observations`), raising `ValidationError` otherwise.
  Converted pricing matches the equivalent shifted-continuous product
  exactly (grid-aligned to the shifted barrier) while the valuation-time KI
  state keeps contractual discrete semantics. A first-order residual bias
  remains (a few bp of PV at daily spacing, growing with observation
  spacing and drift).

### Fixed
- `SnowballQuadEngine`: dense discrete KI schedules now retain their explicit
  observation dates instead of being delegated to continuous monitoring. The
  engine adaptively refines its internal spatial grid to resolve short
  diffusion intervals (accuracy-oriented default of 2.5 cells per interval
  diffusion stddev; lower to 1.25 or opt into BGK for speed), avoiding the
  material fair-KO-rate bias caused by the previous continuous-monitoring
  approximation.

## [0.1.0] - 2026-06-11

### Added
- First public release.
- Equity derivatives: European/American/Asian vanilla options, barrier,
  one-touch, digital, sharkfin, and autocallable products (snowball,
  phoenix, KO-reset snowball, range accrual) with analytical, Monte
  Carlo, PDE, quadrature, and tree engines.
- Fixed income: fixed bonds, FRNs, bond options, bond forwards/futures,
  convertible bonds, interest rate swaps.
- Market data layer (`quantark.param`, `quantark.priceenv`), Greeks
  calculators, portfolio VaR (parametric/historical/Monte Carlo),
  ISDA SIMM v2.6, stress testing, multi-day scenario simulation, and a
  hedging backtest framework.
- Legacy flat-import compatibility shim (`asset`, `util`, …) with
  `DeprecationWarning`; slated for removal in 1.0.
