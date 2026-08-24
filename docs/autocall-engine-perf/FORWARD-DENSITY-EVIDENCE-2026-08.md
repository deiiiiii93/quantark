# Forward-Density Event Stats — Validation Evidence (2026-08)

Spec: `docs/superpowers/specs/2026-08-24-quad-forward-density-event-stats-design.md`
Plan: `docs/superpowers/plans/2026-08-24-quad-forward-density-event-stats.md`
Battery: `test/test_quad_forward_density_stats.py`
Pilot driver: session scratchpad `fwd_tolerance_pilot.py` (grids 1001/2001/4001,
one machine, one window). All numbers measured 2026-08-24 on the ARM64 dev
machine, numpy 2.4.6.

## Analytic identity gates (battery gate c)

Free march, flat r=3%/q=5%/σ=20%, T=1, grid 2001, 50 steps:

- mass − 1: < 1e-6 (measured ~1e-14 at every grid in the event pilot)
- mean / variance vs m·T, σ²T: within 1e-4 relative
- undiscounted BS call (K=105): rel. err **1.36e-5 @2001 → 4.7e-6 @4001 →
  2.9e-7 @8001**; the residual is IDENTICAL when the kinked payoff is
  integrated against the exact analytic Gaussian with the same Simpson rule —
  i.e. quadrature-on-kink error, not a marching defect.
- first passage (continuous KI 80, unreachable KO, T=1):
  |ki_ever − closed form| = **4.45e-5 @1001 → 1.11e-5 @2001 → 2.78e-6 @4001**.

## Pilot convergence (forward vs stacked, same grid)

max |forward − stacked| per field; `ed_*` columns are max relative errors
(scale floor 1e-4). Cases: 1.9y 23-obs snowball family + 96-date discrete KI
+ reverse + disable_ko_after_ki + NODAL projection + Phoenix (memory and
non-memory), flat env spot 100 / vol 20% / r 3% / q 5%.

| field | g1001 | g2001 | g4001 | notes |
|---|---|---|---|---|
| ko_prob (max over cases) | 9.1e-4 | 2.9e-4 | 7.8e-5 | contracts ~4x/level |
| survival | 9.1e-4 | 2.9e-4 | 7.8e-5 | = cumulative of ko_prob |
| ed_ko_cf (rel) | 4.6e-3 | 1.3e-3 | 3.3e-4 | worst case: reverse |
| ki_prob (no-KO scalar) | 9.6e-3 | 4.9e-3 | 2.5e-3 | O(h): stacked's hard-mask KO absorption vs forward's weighted absorption; both converge to the same limit |
| ki_ever | 4.0e-3 | 5.2e-4 | 1.5e-3 | non-monotone ONLY for disc_ki — see below |
| coupon_prob | 1.0e-3 | 3.3e-4 | 9.1e-5 | phoenix |
| ed_coup_cf (rel) | 1.5e-3 | 4.8e-4 | 1.3e-4 | non-memory phoenix |
| \|1 − mass diag\| | 6.2e-4* | ~3e-15 | ~2e-15 | *disc_ki @1001 only; all other cases ~1e-14 at every grid |

Phoenix `ed_ko_cf` is identically 0 in both modes (standard phoenix has
`ko_rate=0`; the KO stream carries no cashflow).

### disc_ki ki_ever non-contraction is the STACKED side

Anchored per mode (96-date discrete KI):

| grid | stacked ki_ever | forward ki_ever |
|---|---|---|
| 1001 | 0.364084 | 0.360079 |
| 2001 | 0.359640 | 0.359117 |
| 4001 | 0.357667 | 0.359193 |
| 8001 | 0.358709 | 0.359151 |
| MC 500k QMC daily | 0.359518 | — |

The forward value is stable to 4 decimals from grid 2001; the stacked value
oscillates ±1.5e-3 (hard 0/1 KI masks → O(h) node-alignment noise). The
|forward − stacked| column crosses zero — a difference-metric artifact, not a
forward defect.

## MC cross-check (battery gate d)

1.9y continuous-KI snowball, forward @2001 vs SnowballMCEngine 500k QMC paths
(479 steps, seed 7): max |Δko_prob| < 4e-3 (3-sigma binomial band),
|Δki_ever| = 4.0e-4 (bound 6e-3).

## Banked tolerances (2x the grid-2001 pilot deltas)

| constant | value | measured @2001 |
|---|---|---|
| KO_PROB_ATOL | 6e-4 | 2.9e-4 |
| KI_PROB_ATOL | 1e-2 | 4.9e-3 |
| CF_RTOL | 3e-3 | 1.3e-3 |
| FWD_VALUE_RTOL | 3e-5 | 1.36e-5 |
| MASS_TOL (free march) | 1e-6 | ~1e-14 |
| MOMENT_RTOL | 1e-4 | passes |
| first-passage bound | 3e-5 | 1.11e-5 |
| mass-diagnostic bound (cont-KI @1001) | 1e-8 | 1.9e-14 |
| MC ko / ki_ever | 4e-3 / 6e-3 | <4e-3 / 4.0e-4 |

## npv invariance (battery gate a)

`pv` and `price_with_events().npv` are hex-string-equal between modes in the
battery (same-run comparisons; the forward mode never computes its own npv —
it is always the backward `price()`).

## Full-book adapter A/B (battery gate e)

_To be appended by Task 11._
