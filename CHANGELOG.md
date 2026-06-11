# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).
During 0.x the public API may still change between minor versions.

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

### Fixed
- `SnowballQuadEngine`: the dense-discrete-KI-to-continuous bridge
  approximation now applies a Broadie-Glasserman-Kou barrier shift to
  emulate discrete monitoring. Previously the conversion monitored the
  unshifted barrier continuously, overstating knock-in probability (fair
  KO-rate bias of up to ~0.9 coupon points vs PDE/MC on daily-KI
  snowballs). A first-order residual can remain for strongly drifted
  (deep-carry) underlyings; set
  `QuadParams.dense_discrete_ki_as_continuous_threshold=0` with a
  sufficiently fine grid for exact discrete pricing.
