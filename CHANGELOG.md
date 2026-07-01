# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).
During 0.x the public API may still change between minor versions.

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
