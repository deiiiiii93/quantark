# Intraday Pricing — Baseline Record

Companion to `2026-09-15-intraday-pricing-design.md` and the three implementation
plans (`plan1-clock-context-result`, `plan2-engines-reference`,
`plan3-greeks-roll-batch`). Worktree branch `worktree-intraday-plan1`, based on
`64b2832d` (feat/simulated-path-backtest, which carries QUAD V2).

## 2026-09-17 follow-up

The historical runs below are preserved. The [third review and correction record](../reviews/intraday-pricing-2026-09-16/RESOLUTION.md)
supersedes the open PDE-gamma diagnosis and the 29-day Greek limit: a query-centred cubic readout removes the eight
original above-budget refinement failures, and 308 additional cells extend the matching QUAD/digital Greek windows
to 90 days. Seven of the eight remeasured default PDE gamma values remain unqualified; there is no PDE point-gamma
certificate. The original failure set also included a 29-day KI cell at +10 bp.

Analytical barrier point theta now reports a per-request truncation/cancellation estimate against the frozen budget.
Numerical Greek certificates additionally match conditional contract/market economics and the demonstrated spot
envelope. The resolution record contains the new independent evidence, regression results, and resource controls;
the earlier timing and curve-qualification counts below have not been rerun as performance claims.

## Inventory (2026-09-15)

- `PricingEnvironment` (`quantark/priceenv/pricing_environment.py:44-52`): fields `rate_curve, valuation_date: datetime, spot_quote, vol_surface, div_yield, basis_yield, day_count_convention=CALENDAR_DAYS, bus_days_in_year=252, calendar`. No tz validation, `.date()` only in `__repr__`. `__post_init__` calls `validate_trading_clock_configuration` only under `BUSINESS_DAYS`.
- `SpotQuote` (`quantark/param/quote/spot_quote.py:13-35`) already has `timestamp: Optional[datetime] = None` — never read for pricing today. The provisional assumption record needs it.
- `calculate_year_fraction` (`quantark/util/calendar/day_counter.py:44-100`) truncates to `timedelta.days` and raises on `end <= start`. `Calendar._normalize_date` (`business_calendar.py:80-83`) strips time-of-day. **We do not route intraday time through either.**
- `BusinessTimeMap` (`quantark/util/calendar/trading_clock.py:35-131`): `__init__(clock, anchor_date, horizon_date, extend_weekdays=False)`, `to_trading(tau_cal)`, `to_calendar(tau_td)`, `initial_slope()`, `re_anchored(anchor)`, attributes `clock, anchor_date, horizon_date`. Plateau contract: stored knot value returned inside a zero-slope segment (`:101-105`).
- `TradingClockVolSurface` (`quantark/param/vol/trading_clock_surface.py:25-91`): frozen dataclass `(inner, time_map)`, `exposes_exact_total_variance = True`, `total_variance(strike, t, spot) = σ_inner(K, u)²·u` with `u = time_map.to_trading(t)`, `get_vol` uses `time_map.initial_slope()` at `t<=0`, `parallel_shifted`, `with_time_map`. It only calls `to_trading`, `initial_slope` on the map → any object with that interface works.
- Consumers of the exact-variance protocol: `TermCoefficients.from_env` (`quantark/priceenv/term_sampling.py:70-100`, PDE+MC) and `interval_moments` (`quantark/asset/equity/engine/quad/v2/context.py:86-123`). Zero-variance step handling already exists: QUAD V2 deterministic shift (`context.py:160-171`), PDE upwind θ=1 (`base_pde_solver.py:1188-1215, 1326-1331`), MC drift-only.
- QUAD V2 (`quantark/asset/equity/engine/quad/v2/engine.py`): `price(product, env, *, lifecycle_state=None, event_phase="before")` → float; `price_components(...)` → `{price, ko, coupon, terminal, pending, reconciliation_error}`; `calculate_point_greeks` → `{price, delta, gamma}`; `_validate_market` accepts `TradingClockVolSurface` (recurses into `.inner`, `:124-131`); `event_phase="after"` deletes `t == 0` events, `"before"` prices them deterministically at spot (`adapters.py:160-164, 417-420`); `type(product) is product_type` exact check (`:46-49`); `supports_lifecycle_state = True`, reads `lifecycle_state.knocked_in`, `.coupon_memory_count`, ledger via `pending_receivable_pv` (`settlement_support.py:408`).
- Snowball payoff composition (`snowball_option.py:1030-1088`): `accrual_factor = accrual_config.accrual_factors[source_idx]` when supplied (positional in the KO schedule), `rate = record.return_rate or get_ko_rate_at(idx)`, `payoff = principal + initial_price·contract_multiplier·rate·accrual_factor`. Terminal rebate uses `get_contract_tenor` → `get_tenor` → returns `self.tenor` when set (`base_equity_option.py:357-359`). Past filter is strict `observation_date < valuation_date` (`:987-1001`).
- Lifecycle: `AutocallableLifecycleState` (`lifecycle/state.py:31-55`: `alive, knocked_in, knocked_out, matured, ki_date, ko_date, coupon_memory_count, valuation_point, ledger, observed_*_indices, ...`), `BarrierLifecycleState` (`:252-264`), `ValuationPoint(date|time)` exactly one (`cashflows.py:19-47`), `RealizedCashflow` time pair all-or-nothing, negative `determination_time` legal, `payment_time >= determination_time` (`cashflows.py:100-142`), `LifecycleCashflowLedger._is_pending` time-based = `payment_time > point.time` (`:250-268`). `AutocallableLifecycleTracker(product=, quantity=, lifecycle=, start_date=)` with `observe(date, product, env, spot) -> List[LifecycleEvent]` (`lifecycle/autocallable.py:53-68, 143`) and `settle_maturity_if_due(date, product, env, spot)` (`:307`); `BarrierLifecycleTracker(product=, quantity=, start_date=)` with `observe(date, env, spot)` (`lifecycle/barrier.py:84-98, 243`) — fresh `state`, assignable.
- Execution: `PricingSession.execute(engine, PricingRequest) -> PricingOutcome(value, normalized_economics, diagnostics, manifest)` (`execution/api.py:125`); `PricingRequest(product, pricing_env, operation, outputs, operation_options, request_id, lifecycle_state)` (`contracts.py:52-60`); `CapabilityError` (`execution/errors.py:19`); `fingerprint`/`canonical_tree` (`execution/cache/fingerprint.py:21-66`, tz-aware datetimes canonicalise via `isoformat()` including offset).
- `ObservationRecord` (`observation_schedule.py:33-64`) fields are all Optional; `ObservationSchedule.resolve(...)` builds `ResolvedObservationRecord(observation_time, barrier, payoff, settlement_time, observation_date, settlement_date)`; `SettlementConvention(lag, lag_unit=SettlementLagUnit.YEAR_FRACTION)` yields `payment_time = determination_time + lag` (`settlement.py:383-384`).
- `DigitalOptionAnalyticalEngine.price(product, env, *, lifecycle_state=None)` (`analytical/digital_option_engine.py:67`) reads `T, r, q, σ` at maturity, rejects `σ <= 0` BEFORE its `T < 1e-10` intrinsic branch, discounts with `resolve_terminal_timing(...).payment_df`.

## Daily regression gate (must stay green, never regenerated)
test/test_trading_clock_map.py test/test_trading_clock_surface.py test/test_trading_clock_parity.py
test/test_trading_clock_axis_equivalence.py test/test_term_sampling_total_variance.py
test/test_quad_v2_engine.py test/test_quad_v2_terms_lifecycle.py test/test_quad_v2_coupon_event_stats.py
test/test_observation_schedule.py test/test_equity_settlement_resolver.py test/test_lifecycle_cashflow_ledger.py
test/test_equity_lifecycle_trackers.py test/test_snowball_pde_date_schedule.py test/test_digital_option_mc_engine.py
test/execution/test_session_parity.py test/execution/test_registry.py

## Baseline run
2026-09-15, worktree at `64b2832d`, `-n auto`: the 16 gate files + `test/intraday/test_daily_regression_gate.py`
→ **178 passed, 0 failed** (114 s).

Runner note: `quantark_compat.pth` imports `quantark` during site init, so worktree
source only wins when `PYTHONPATH` points at the worktree before the interpreter
starts (a `python -m` cwd insertion comes too late). Tests import shared fixtures as
`from intraday.conftest import ...` (pytest prepends `test/`; there is no
`test/__init__.py`, so `test.` would resolve to the stdlib package).

## Plan 1 exit
2026-09-15, full suite `-n auto` after Task 16 (machine shared with another session's long job):
**7829 passed, 142 skipped, 0 failed** (1096 s) — 7732 legacy (same count as the post-Task-6 run) + 97 intraday
(Gate A 6, Gate B 5). Demo: `example/intraday_snowball_fixing_demo.py` (four rows at 103.5 and 102.0).

Plan deviations worth knowing (details in the commit messages):
- instant arithmetic through UTC (same-tzinfo datetime arithmetic is wall-clock across DST);
- checkpoint-covered events are marked observed before tracker replay; each replay step must observe exactly its instant;
- every numerical twin re-resolves its own cash and must match the contract; Phoenix coupons ride `accrual_factors` only
  when they agree with the KO accrual; a terminated contract has no twin;
- continuous-KI autocallables disclose a `ContinuousHistoryAssumption` already in Plan 1;
- a date-only payment is deemed at the calendar's payment time but never before its determination;
- Gate A: the price is NOT monotone towards a KO fixing above the barrier (6.86 a day out, 6.05 a second out — the
  unknocked contract is worth more than the KO cash); the gap to the discounted KO cash is.

## Acceptance budgets
Fixed by Plan 2 Task 1 (independent reference) BEFORE any route is tuned; frozen in
`test/intraday/reference/budgets.py`:
`PRICE_ABS_PER_NOTIONAL = 1e-6` (x initial_price x multiplier), `DELTA_ABS, DELTA_REL = 1e-5, 1e-4` on
d* = Delta S/N, `GAMMA_ABS, GAMMA_REL = 1e-4, 1e-3` on g* = Gamma S^2/N,
`REFERENCE_UNCERTAINTY_MULTIPLIER = 3.0`: a route passes iff |route - ref| <= budget + 3 ref_uncertainty.

Reference (`test/intraday/reference/gaussian_reference.py`): each state is a piecewise-linear function of ln S with
exact jumps on barrier knots; its Gaussian expectation and first two derivatives are closed form (erf/R(a)); the
continuation at the first remaining event is re-evaluated exactly on a local grid scaled to the remaining standard
deviation, so the final (down to one-second) propagation is an exact pointwise expectation. Error = O(h^2)
interpolation of the smooth continuation, estimated by Richardson |p2 - p1|/3. Continuous KI and Phoenix coupons
are outside the reference (raise).

Evidence (`python -m intraday.reference.evidence`): fixture snowball, sixth fixing, 5 confirmed fixings; profiles
uniform / desk / sessions_only; spots B(1 +- 1e-4), B(1 +- 1e-3), B exp(+-k sqrt W) k in {0.5, 1, 2} for B in
{103 (KO), 75 (KI)}; 480 cells.

At points (2001, 4001, 8001): worst 3*unc/budget price 0.785, delta 0.521, gamma 1.188 (6h; a cell where gamma crosses
zero and the absolute floor binds). Doubling the grid cut that uncertainty by exactly 4x (7.14e-7 -> 1.79e-7), so it
is resolution, not a limit: **Gate C uses `GATE_C_POINTS = (4001, 8001, 16001)`**.

**Correction (same day).** The first evidence run shared the reference's global-sweep cache across horizons and the
cache returned the time to the second remaining event measured from the FIRST valuation instant that populated it, so
every later horizon priced its final interval over the wrong span (found when the reference disagreed with QUAD V2 by
2.2e-3 at 15 minutes; an independent quadrature over QUAD V2's own continuation sided with QUAD V2). Richardson
uncertainties were unaffected (every level shared the stale time), prices were not. Fixed in the reference; a
call-order regression test now guards it. The re-run (624 s, machine shared) confirms the conclusion:

| horizon | max unc price | max unc delta | max unc gamma | worst 3*unc/budget (price, delta, gamma) |
|---|---|---|---|---|
| 1 day | 6.53e-06 | 3.10e-06 | 1.83e-06 | 0.196, 0.130, 0.051 |
| 6h | 6.52e-06 | 5.34e-06 | 6.60e-06 | 0.196, 0.130, 0.323 |
| 1h | 4.93e-06 | 1.21e-05 | 3.71e-05 | 0.148, 0.079, 0.015 |
| 15m | 4.23e-06 | 2.34e-05 | 1.45e-04 | 0.127, 0.060, 0.045 |
| 5m | 3.94e-06 | 3.99e-05 | 4.29e-04 | 0.118, 0.052, 0.005 |
| 1m | 3.76e-06 | 8.53e-05 | 2.13e-03 | 0.113, 0.059, 0.004 |
| 10s | 3.77e-06 | 1.92e-04 | 1.27e-02 | 0.113, 0.099, 0.039 |
| 1s | 3.77e-06 | 6.04e-04 | 1.27e-01 | 0.113, 0.099, 0.039 |

**No horizon is reference-limited** at Gate C resolution (the plan expected 10s/1s gamma to be; the local final grid
removes that limit). `budgets.REFERENCE_LIMITED` stays empty for the snowball family.

## Gate C results
Packaged evidence: `quantark/intraday/evidence/gate_c_results.json` (3432 cells; git 510c3da0 — the first 1207
cells ran at 9110eb62, whose classification is identical: 510c3da0 only releases a PDE memo between solves). Four
xdist workers under a process-group RSS guard (20 GiB); the guard stopped the run once when heavy PDE cells
coincided, and the resume finished the last 48 cells on two workers. About 27 minutes of wall time.

| product | engine | passed | unqualified | failed |
|---|---|---|---|---|
| snowball (discrete KI) | QUAD V2 | 528 | 0 | 0 |
| snowball (discrete KI) | PDE | 3 | 525 | 0 |
| snowball (discrete KI) | MC (RQMC) | 5 | 523 | 0 |
| cash-or-nothing digital | analytical | 264 | 0 | 0 |
| cash-or-nothing digital | MC (RQMC) | 8 | 256 | 0 |
| up-and-out call, zero carry | analytical | 264 | 0 | 0 |
| up-and-out call, zero carry | PDE | 186 | 78 | 0 |
| up-and-out call, zero carry | MC (RQMC, bridge) | 148 | 116 | 0 |
| up one-touch, zero carry | analytical | 264 | 0 | 0 |
| up one-touch, zero carry | PDE | 144 | 120 | 0 |

**Qualified rows (every cell at the horizon and above passed, all three profiles):** `SnowballQuadEngineV2`
discrete, `DigitalOptionAnalyticalEngine` terminal, `BarrierAnalyticalEngine` and `OneTouchAnalyticalEngine`
continuous (on zero-carry contracts) — all to **1 second** before the fixing. PDE and MC rows stay `supported`: at
default settings their errors exceed the 1e-6-of-notional price budget while their refinement ladders (or phase
envelopes) converge, or their sampling error exceeds the budget — honest `unqualified`, never a defect. Barrier PDE
passes at 1 day / 6 h on individual profiles but not on all three at one horizon.

Defects Gate C found and fixed before this run (each with a regression test):
- **MC, zero-variance steps** (4230ff99): the Brownian-bridge crossing probability rejected σ = 0 on
  sessions-only lunch/overnight steps; a zero-variance step cannot cross between same-side endpoints (probability 0).
- **PDE resolution ignored time** (0ea58a89): a sessions-only barrier one day before expiry priced 0.895 against the
  reference 0.515 while reporting `resolved` — four steps per day put a whole session's variance into one
  Crank–Nicolson step. `resolved` now also needs >= 16 steps across the layer variance and e^-8 damping of the
  grid-scale mode (theta-scheme factor with the solver's own Rannacher steps); the route refines steps per day on a clone.
- **One-touch / discrete barriers on PDE** (0ea58a89): `OneTouchPDESolver` overwrites the nodes beyond the barrier, so
  the effective barrier is the next node (first order, alignment-dependent: +1 point moved the error from 0.016 to
  0.057). Legacy daily behaviour, not changed; the intraday route reports `unqualified` unless a node sits on the barrier.
- **Harness ladder** (fb449eeb): PDE levels scaled the default request and stayed pinned at the route's layer floor
  (time-only refinement); levels now double the grid actually solved. A single-phase ladder that is not monotone is
  judged on the envelope over +1..+3 points per level (the snowball's discrete-KI error depends on the KI barrier's
  position in its cell: envelope 1.1e-2, 2.0e-3, 4.7e-4, 1.3e-4 per doubling).
- **Memory** (9110eb62, 510c3da0): the two-surface PDE solvers keep both value surfaces over the whole time grid and
  a per-market coefficient memo; an uncapped level-2 ladder on a 10-second grid exceeded 12 GiB per worker (an
  earlier 12-worker run exhausted the 48 GB machine). The route's refinement is bounded by
  `INTRADAY_PDE_MAX_GRID_CELLS` (5e7, ~1.2 GiB measured) and releases its surfaces; ladders stop at 1e8 cells and a
  capped ladder is `unqualified`.

## Gate C greeks
Packaged evidence: `quantark/intraday/evidence/gate_c_greeks.json` (schema `intraday-gate-c-greeks/2`, git 959346b5, the 2026-09-16
re-review sweep; 792 groups = snowball and digital x desk/sessions-only x 12 horizons (1 s to 29 days) x 11 offsets x
barriers; 2112 route results). Four workers under a 40 GiB guard, 2 h 22 min, peak ~12 GiB, on a machine shared
with two other sessions' process pools. Every row records the engine's `accuracy_settings`; a certificate is a
window of consecutive swept horizons at which every cell of (product, route, measure, monitoring, profile, engine
settings, measure settings) passed. Budgets frozen before the run: point/desk vega, rho, dividend rho m* = move /
N and theta t* = |theta per hour| / N, both absolute 1e-6, relative 1e-4.

| product | route | point delta | point gamma | desk delta | desk gamma | point vega / rho / div rho | desk vega / rho / div rho | desk theta | point theta |
|---|---|---|---|---|---|---|---|---|---|
| snowball | QUAD V2 | 528 passed | 528 passed | 528 passed | 528 passed | 528 passed each | 528 passed each | 528 passed | 528 passed |
| snowball | PDE | 20 passed, 508 unqualified | 5 passed, 8 failed, 515 unqualified | 179 passed, 349 unqualified | 72 passed, 456 unqualified | not run | not run | not run | not run |
| snowball | MC (RQMC) | 528 unqualified | 528 unqualified | 528 unqualified | 528 unqualified | not run | not run | not run | not run |
| digital | analytical | 264 passed | 264 passed | 264 passed | 264 passed | 264 passed each | 264 passed each | 264 passed | 264 passed |
| digital | MC | 264 unsupported | 264 unsupported | — | — | — | — | — | — |

**Certified, 1 s to 29 days, both profiles:** QUAD V2 snowball and analytical digital, every measure. PDE desk delta
only within 1-300 s (desk) and 10-60 s (sessions-only) of a fixing. Nothing on MC.

What changed from the first sweep (git 9eb643df, 8 horizons to 1 day):
- The ladder read the route's GATED point status, and the gate reads this evidence: MC point Greeks could never be
  measured (they were "unqualified by construction"). Routes now take `certify=False` for the ladder.
- The desk classifier tested a pass before the sampling error, so a standard error far above the budget widened the
  tolerance into a pass: MC desk delta was "352 passed" with standard errors ~12x its budget. The sampling check now
  comes first, and MC desk moves are unqualified.
- A desk move's reference uncertainty is the Richardson estimate of the MOVE's own levels (its prices share each
  grid), not the sum of its prices' uncertainties; the sum made the 14- and 29-day QUAD desk cells inconclusive.
- The reference evaluated an event decided exactly at the spot on its right-continuous side; every autocallable
  barrier is inclusive, so a spot on a standard KI level now takes the knocked-in side (a one-hour desk theta roll
  landing on the fixing with the spot on the barrier).
- An MC point ladder stops after level 0 when both measures are already beyond their noise there (the 4x-path level
  cannot change that status).
- The 8 failed cells are PDE point gamma at 3 to 29 days, one or two standard deviations above the KO barrier: the
  refinement ladder and its phase envelope oscillate at errors of 1e-5 to 4e-4 without approaching the reference.
  PDE point gamma carries no certificate anywhere; the oscillation is a PDE diagnostic still to explain.

Found while building this gate: the reference's global-sweep cache keyed a bumped context on its REQUEST's market, so
rate and dividend bumps reused the unbumped sweep (reference rho -0.348 and dividend rho +0.240, where QUAD V2, a
fine PDE and a CRN MC agreed on -0.27 and -0.06); fixed in 14128502 (content key for bumped markets) with a regression
test. Price evidence never used bumped contexts.

## Gate D results
`example/intraday_benchmark/run_benchmark.py`, one child process per engine x workload, single-threaded BLAS
(`OPENBLAS/OMP/VECLIB_MAXIMUM_THREADS=1`), medians of 5 repeats (3 for PDE/MC curves). Machine: Apple Silicon
(Darwin-arm64), 48 GB, shared with another session's 6-core simulation fleet for the QUAD V2 / analytical rows
(load ~2-7) and quiet for the PDE / MC rows (load ~1.6-2.7). Code: QUAD V2 / analytical at 3bb79729; PDE single /
curve and MC at b74e4d52; PDE batch at b901e9bf (PDE budget in bytes). There is no SLA; "peak traced" is the
`tracemalloc` peak of one extra pass (NumPy buffers included). Accuracy column: the Gate C price status of the
matching cell (spot one standard deviation below KO for snowballs, above the strike for digitals).

| engine | state | cold (ms) | warm (ms) | peak traced (MiB) | Gate C price status |
|---|---|---|---|---|---|
| QUAD V2 | snowball 1 h, point (price + delta, gamma, vega, rho, theta) | 82.9 | 82.9 | 2.2 | passed |
| QUAD V2 | snowball 1 h, desk_bump | 61.1 | 60.0 | 2.4 | passed |
| QUAD V2 | snowball 1 s, point | 81.1 | 81.9 | 2.2 | passed |
| QUAD V2 | snowball 1 s, desk_bump | 60.7 | 59.3 | 2.4 | passed |
| analytical | digital 1 h, point | 2.5 | 2.4 | 0.1 | passed |
| analytical | digital 1 h, desk_bump | 2.4 | 2.3 | 0.3 | passed |
| analytical | digital 1 s, point | 2.5 | 2.3 | 0.1 | passed |
| analytical | digital 1 s, desk_bump | 2.4 | 2.4 | 0.2 | passed |
| PDE | snowball 1 h, point | 700.8 | 700.6 | 102.3 | unqualified |
| PDE | snowball 1 h, desk_bump | 1442.6 | 1441.3 | 110.9 | unqualified |
| PDE | snowball 1 s, point | 1042.9 | 1060.9 | 531.2 | unqualified |
| PDE | snowball 1 s, desk_bump | 2173.0 | 2161.5 | 534.6 | unqualified |
| MC (RQMC, 2^14 paths) | snowball 1 h, point | 578.0 | 581.1 | 4.2 | unqualified |
| MC (RQMC) | snowball 1 h, desk_bump | 826.0 | 828.9 | 4.4 | unqualified |
| MC (RQMC) | snowball 1 s, point | 574.2 | 570.8 | 4.2 | unqualified |
| MC (RQMC) | snowball 1 s, desk_bump | 754.6 | 749.7 | 4.4 | unqualified |

| engine | 101-spot curve (95..106, across the KO barrier) | first curve (s) | warm curve (s) | peak traced (MiB) | points with point greeks |
|---|---|---|---|---|---|
| QUAD V2 | snowball 1 h before a fixing | 0.029 | 0.028 | 1.4 | 101 |
| QUAD V2 | snowball 1 s before a fixing | 0.025 | 0.024 | 1.4 | 101 |
| analytical | digital 1 h before expiry | 0.040 | 0.039 | 0.1 | 0 (per-spot pricing) |
| analytical | digital 1 s before expiry | 0.039 | 0.039 | 0.1 | 0 |
| PDE | snowball 1 h before a fixing | 23.4 | 23.5 | 87.3 | 0 |
| PDE | snowball 1 s before a fixing | 35.3 | 35.9 | 380.4 | 0 |
| MC (RQMC) | snowball 1 h before a fixing | 11.7 | 11.7 | 4.4 | 0 |
| MC (RQMC) | snowball 1 s before a fixing | 10.4 | 10.4 | 4.4 | 0 |

| engine | batch: 100 mixed items (median per kind) | total (s) | per-item median (ms) | failures | peak traced (MiB) |
|---|---|---|---|---|---|
| QUAD V2 (+ closed forms for digital / barrier) | snowball 9.7 ms, phoenix 17.7 ms, digital 0.4 ms, barrier 0.7 ms | 0.75 | 5.2 | 0 | 2.2 |
| analytical | 50 items: digital 0.4 ms, barrier 0.6 ms (no autocallable route) | 0.02 | 0.5 | 0 | 0.2 |
| PDE (+ closed form for digital) | snowball 600 ms, phoenix 3833 ms, digital 0.5 ms, barrier 7.1 ms | 106.5 | 90.5 | 0 | 886.5 |
| MC (RQMC) | snowball 139 ms, phoenix 1670 ms, digital 155 ms, barrier 199 ms | 55.8 | 198.8 | 0 | 80.8 |

Found while measuring (fixed, each with a regression test): a 101-spot PDE curve kept every spot's copied market —
and its PDE coefficient memo — alive and grew past 8 GiB (3bb79729: contexts one spot at a time, now 0.73 GiB); the
PDE route left the legacy coefficient memo on the request's rate curve, so a batch holding its requests kept every
route's coefficient sets (b74e4d52); the grid budget assumed a snowball's two surfaces while a memory Phoenix keeps
2(k+1)+2 (b901e9bf: budget in bytes per the solver's surfaces; PDE batch 209 s / 3.5 GiB traced -> 106 s / 0.9 GiB).

## Open questions (need a desk decision; intraday fails closed meanwhile)
- **Phoenix realized coupon amount.** `AutocallableLifecycleTracker.observe` books a coupon as
  `get_coupon_payoff(idx)` = principal·coupon_rate·1.0, while every Phoenix engine (QUAD V2 adapter,
  QUAD, PDE) pays principal·coupon_rate·period_year_fraction, so a replayed coupon had two amounts.
  RESOLVED 2026-09-16: the tracker books each period's own fraction (commit 4a613401) and a memory
  coupon now names the arrears periods it released in its ledger metadata, so the intraday layer
  reconciles the entry against the sum of the contract's own per-period amounts instead of the
  current period alone (review 2026-09-16 finding 10). Outstanding arrears reach the stateless
  engines on the twin (`CouponBarrierConfig.initial_coupon_arrears`) and QUAD V2 through the
  lifecycle state's count; both still require equal periods.

## Known day-resolution hazards that intraday mode must bypass (not fix)
- calculate_year_fraction → timedelta.days; Calendar._normalize_date → midnight; lifecycle trackers → pd.Timestamp.normalize().
- SnowballMCEngine._build_time_grid keeps t=0 nodes → GBMPathGenerator raises on dt=0 (Plan 2 fixes).
- SnowballPDESolver._time_stepping_two_surface does not force θ=1 on zero-diffusion sets (Plan 2 fixes).
