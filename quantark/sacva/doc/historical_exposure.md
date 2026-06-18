# Historical Exposure Engine (non-regulatory)

`quantark/sacva/exposure/historical/` provides a **real-world** expected-exposure
(EE) and **potential future exposure** (PFE) backend for the SA-CVA module. It is
deliberately **not** part of the regulatory capital path.

## Purpose & consumers

- **PFE / counterparty-credit limits** — quantiles of future exposure.
- **Exposure-model backtesting** — predicted quantiles vs realized outcomes
  (Kupiec coverage test in `pfe.py`).
- **What-if analysis.**

It is the historical-vs-MC twin of the regulatory `MonteCarloExposureEngine`,
mirroring VaR's historical-vs-MC duality — but the two engines target **different
measures and different consumers**.

## The hard regulatory boundary (MAR50.34(1))

> *"Drifts of risk factors must be consistent with a risk-neutral probability
> measure. Historical calibration of drifts is not allowed."*

Regulatory CVA is a risk-neutral price. This engine uses **historical (real-world)
drifts**, so every `ExposureProfile` it emits carries `measure = REAL_WORLD` and
`regulatory_eligible = False`. The eligibility **guard** lives in the capital-path
consumer: `RegulatoryCVAEngine.compute` **raises** on any non-eligible / non
risk-neutral profile, so a historical exposure can never feed the SA-CVA capital
number. The two-measure
PFE convention is followed: **real-world path generation, risk-neutral valuation**
at each horizon (reused value-surface layer).

## Path generation

`HistoricalPathGenerator` produces a real-world state tensor
`states[n_paths, n_grid, n_factors]`, consumed unchanged by the (provisional)
repricer:

- **`REPLAY_RAW`** — each historical window → one forward trajectory by compounding
  that window's *actual* multivariate log-returns (drift untouched).
- **`REPLAY_DRIFT_ADJUSTED`** — same, but re-centred per `drift_mode`
  (disclosed in metadata as *not* pure replay).
- **`BOOTSTRAP`** — resample multivariate return/residual vectors and compound
  forward for long horizons.

Levels are built by **compounding daily log-returns** over each grid interval —
never sqrt-t scaling (a VaR vol approximation invalid for path construction).

## Resampling schemes (`resampling.py`)

All resample whole same-date vectors by a **common time index** (cross-factor
co-movement is empirical; no Cholesky recolouring):

- **`IID_RAW`** — i.i.d. vectors; destroys autocorrelation/clustering.
- **`BLOCK_FHS`** — fixed-length block bootstrap of **EWMA-standardized** residual
  vectors, with conditional vol **recursively evolved along the simulated path**
  (FHS): `r_k = μ* + σ_k z_k`, `σ²_{k+1} = λσ²_k + (1−λ)(r_k−μ*)²`, seeded from
  today's EWMA vol.
- **`STATIONARY_BLOCK`** — Politis–Romano geometric block length
  (`p = 1/expected_block_length`), circular indexing.

`path_mode` and the bootstrap scheme/lengths are **required explicitly** — no
silent production default (repo no-fallback rule).

## PFE / outputs (`pfe.py`)

`ExposureProfile` (additive over the MC contract) exposes:

- `ee_undiscounted[t]` — mean positive exposure (primary for limits/PFE).
- `pfe[confidence_bps][t]` — empirical quantiles keyed by integer bps
  (`9500`, `9900`); default Hyndman–Fan type 7 (`"linear"`), conservative
  `"inverted_cdf"` (type 1) optional. Tail adequacy `n·(1−conf) ≥ m_tail_min`.
- `epe` — scalar time-weighted EE (trapezoid).

The regulatory `epe_discounted` field is **always `None`** on historical profiles;
`RegulatoryCVAEngine` reads only `epe_discounted`, never `discounted_ee_nonreg`
(field-identity audit).

## Canonical contract (reconciled)

The engine consumes the **canonical** MC-owned contract directly: it reprices
canonical `CVATrade(product, engine, env)` via `AnalyticValueSurface` on
`ExposureGrid` (the same repricing the MC engine uses), and returns the shared
`ExposureProfile` — extended additively with `ee_undiscounted` / `pfe` / `epe` /
`metadata`, with `epe_discounted = None` on real-world profiles. The provisional
contract has been removed; `test_merge_gate_no_provisional_import` enforces that no
`historical/*` module imports it.

**Deferred follow-ups:** stateful (snowball) historical exposure (raises in v1),
and a calendar/event-driven exposure grid (the engine uses a uniform year-fraction
grid to the longest maturity, mirroring the MC vanilla path).

## Scope (v1, mirrors MC v1)

Supported: equity spot, reporting-vs-foreign FX spot; deterministic rates; single
reporting currency; 1D-spot-per-trade. Rejected (raises): stochastic-rate /
commodity / inflation / cross-pair-FX / multi-factor / foreign-underlying / quanto
/ continuous-barrier trades. GARCH FHS and antithetic historical variance
reduction are deferred.
