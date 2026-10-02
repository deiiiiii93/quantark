# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).
During 0.x the public API may still change between minor versions.

## [0.8.0] - 2026-10-03

PDE 1D numerical policy (spec: `docs/superpowers/specs/2026-10-01-pde-1d-numerical-policy-design.md`).
Seven deliberate numerics changes, each its own commit checked against the 445-row PDE byte
book with its declared movers: the grid day is the environment's business days, the step
cap raises instead of scaling, the barrier family projects discrete observations, Snowball
`NODAL` means nodal, Phoenix takes θ = 1 on zero-diffusion steps, half-step damping where
monitoring is discrete, and forward far-field rows. No flip touches the damping of a
product with continuous knock-in: on a 252-day environment, inside 0.7.0's 5,000-step cap
and with the default projection, it keeps its 0.7.0 numbers bit for bit (one exception,
below); the grid day (a) and the cap (b) move it like any product. The flat Snowball, Phoenix and
KO-reset and the local-vol Snowball certificates are amended and re-banked (2026-10-02) with
their PDE candidates re-admitted; the intraday daily-KI certificate is re-banked (2026-10-03)
with QUAD V2 admitted and its PDE candidate rejected, as in 0.7.0.

### Added
- `CrankNicolson(damping="full_step" | "half_step")`: the damping form; `None` is the
  route's. Half-step splits each damped interval at its midpoint; both halves take
  `event_theta` below an event and 1.0 at the terminal node.
- `event_projection` is honoured by `BarrierPDESolver`, `DoubleBarrierPDESolver`,
  `OneTouchPDESolver` and `DoubleOneTouchPDESolver`.
- `MarketSnapshot.bus_days_in_year`, `GridRequest.day_count`, `GridRequest.refinement`,
  `TimeRefinement`, `time_fill`, `EventRecord`, `DiscreteEvent`, `apply_discrete_event`
  (`quantark.asset.equity.engine.pde.grid`).
- `DAMPING_DEFAULTS["core-continuous-ki"]`, `["2D"]`, `["local-vol"]` (`pde.grid.config`).
- `grid.mesh_kernels.backend()`: `"numba"` with the `accel` extra, else `"python"`.
- `pde_declared.environment_day_count` / `refuse_case_day_count` (model validation): the
  environment's business days recorded on a PDE candidate's `grid` record.

### Changed (defaults)
- **Core damping:** half-step, one interval at the terminal node and at each event,
  `(half_step, 1, 1)`, for every product without continuous knock-in; a product with
  continuous knock-in keeps full-step `(1, 2)`. The class is read from the contract, never
  from the spot or the knocked-in state.
- **Base-family discrete observations:** dual-cell projection (`event_projection=None` is
  `CELL_AVERAGE` on every core route). `"nodal"` keeps the 0.7.0 masks.
- **`max_steps`:** 12,000; a grid over it raises `ValidationError` naming the count it
  needs, `GridConfig(max_steps=...)` and `steps_per_day`. A grid is never scaled.
- **Snowball `NODAL`:** nodal on interior KO and discrete-KI events too.
- **Phoenix:** θ = 1 on zero-diffusion steps, as every value group.
- **Grid day:** `steps_per_day` counts the pricing environment's `bus_days_in_year`.
- **Far-field rows:** European and Barrier puts' lower row is `K·DF(t,T) − S_min·CF(t,T)`
  (floored at 0); the Barrier call's upper-row carry factor is the forward one
  `exp(−(q(T)T − q(t)t))` (it moves only on a term-dependent carry curve).
- `adapter.stepper(params)` becomes `adapter.stepper(params, product)`.
- `GridRequest(day_count=...)` and `MarketSnapshot(bus_days_in_year=...)` are required: code
  that constructs either directly gets a `TypeError` until it passes them.
- Event statistics (KO/KI probabilities) march the same layout, so they move with the half
  nodes at (f), as the prices do; their conventions are unchanged.
- The step cap holds a little under six years at `high` and twelve at `standard` on a 252-day
  environment; a six-year monthly Snowball at `high` now raises where 0.7.0 scaled it.
- The certificates' `grid` record: `max_steps` 12000, `day_count` from the environment,
  and `discrete_monitoring_damping: ["half_step", 1, 1]`. A full-step `(1, 2)`
  configuration with `max_steps=5000` on a 252-day environment still serializes to the
  0.7.0 record.
- Prepared PDE sessions: artefact builder `pde-prep/v3` (the request carries `refinement`
  and `day_count`); no 0.7.0 artefact is reused.

**Clean break (count meaning).** A caller who sets a count and no form, for example
`CrankNicolson(event_damping_steps=2)`, gets two halved intervals after flip (f) on every
solve that resolves against `"core"`, where 0.7.0 gave two whole steps; on a product with
continuous knock-in the count keeps its 0.7.0 meaning. Nothing can detect the intent, so no
error is possible. `CrankNicolson(damping="full_step", terminal_damping_steps=1,
event_damping_steps=2)` restores 0.7.0 everywhere; `damping="full_step"` alone takes the
route's counts, `(full_step, 1, 1)` on the core. A `CrankNicolson` **subclass** is a custom
stepper, used as given: one that sets counts and no `damping` priced in 0.7.0 and now raises
`ValidationError` naming the fields to set.

### Removed
| Removed | Replacement |
|---|---|
| `GridConfig(day_count=...)` (raises `ValidationError`) | `PricingEnvironment(bus_days_in_year=...)` |
| `TimeLayout.fill_scaled`, `TimeLayout.cap_exceeded`; the same keys in `explain` and the intraday numerical record; `LifeSurfaceSolution.fill_scaled` | none: a grid over `max_steps` raises |
| `groups.VALUE_NO_ZERO_DIFFUSION_OVERRIDE` | `groups.VALUE` |

`GridConfig` has eight fields. `TimeLayout` and `LifeSurfaceSolution` are results the
library constructs, so reading a removed field is a plain `AttributeError`.

### Fixed
- **Put far-field row.** The European and Barrier put's lower row was its `S → 0` limit
  `K·DF` at a grid edge four standard deviations below spot, ~39 too high there. Error
  against Black-Scholes at 400 points (pv / delta / gamma), study environment:

  | European put, strike 100 | 0.7.0 | 0.8.0 | the call, for scale |
  |---|---|---|---|
  | vol 22%, 1y, spot 100 | +2.5e-3 / −4.3e-4 / +9.1e-5 | +1.1e-4 / +4.1e-5 / +2.1e-6 | +2.1e-4 / +4.2e-5 / +2.2e-6 |
  | vol 22%, 1y, spot 80 | +1.9e-3 / −5.4e-4 / +1.1e-4 | −1.4e-5 / −6.6e-5 / +1.7e-6 | +2.6e-5 / −6.6e-5 / +1.7e-6 |
  | vol 22%, 3m, spot 100 | +4.2e-3 / −1.4e-3 / +5.9e-4 | +1.9e-4 / +8.4e-5 / +2.2e-5 | +2.5e-4 / +8.4e-5 / +2.2e-5 |
  | vol 22%, 3y, spot 100 | +1.1e-3 / −1.0e-4 / +1.5e-5 | −7.6e-5 / +2.3e-5 / +1.9e-7 | +2.0e-5 / +2.7e-5 / +2.2e-7 |
  | vol 12%, 1y, spot 100 | +3.9e-3 / −1.2e-3 / +4.6e-4 | +2.0e-4 / +6.2e-5 / +1.6e-5 | +2.8e-4 / +6.2e-5 / +1.6e-5 |
  | vol 40%, 1y, spot 100 | +1.0e-3 / −9.3e-5 / +1.3e-5 | −9.6e-5 / +2.4e-5 / +1.2e-7 | +1.4e-5 / +2.9e-5 / +1.6e-7 |

  The down-and-in put, dominated by that row, now meets its exact reference. The American
  put's lower row is unchanged (spec §9).
- **Snowball `event_projection="nodal"`** projected interior events anyway; it now masks
  them (spec §4.6 cases, NODAL pv 0.7.0 → 0.8.0: ordinary +3.5e-2, spot 102.5 +2.6e-2,
  monthly discrete KI +2.6e-2).
- **Phoenix** took θ = 0.5 on zero-diffusion (trading-clock night and weekend) steps where
  every other family took 1 (trading-clock Phoenix: +1.6e-7).
- **Continuous knock-in, alive boundary:** the knocked-in boundary rows are written before
  the alive rows that read them. Bitwise on every certified cell but one:
  `snowball-localvol-1d/calm_near_expiry` moves by 1.1e-13 (pv) to 3.5e-11 (delta) relative;
  the sparse-LU fallback Snowball (two interior nodes) moves +1.457.

### Performance
- **Mesh build.** The concentrated mesh's RK4 integration (the cold-solve cost, spec §4.9)
  runs on a Python-float kernel, or on a Numba kernel with the `accel` extra
  (`pip install "quantark[accel]"`), both bit-identical to the 0.7.0 NumPy reference on 549
  layouts. A value a kernel does not handle (non-finite, zero denominator, more than eight
  critical points) re-runs that call on the NumPy reference, so every exception and fallback
  is 0.7.0's. On 49 production layouts: 43.1 s (NumPy), 4.06 s (Python kernel), 0.30 s (Numba).
  CI requires the Python kernel on its four Python jobs and Numba in the `accel` job
  (`QUANTARK_REQUIRE_MESH_KERNEL`, a test switch). The
  Numba runtime adds ~30 MB to a process's peak memory.
- **Barrier step events** fold in place on the grid column (no per-step copy), and the
  terminal settlement timing is resolved once per march, not once per step (bitwise).

Timing gate (`test/pde_bitwise/timing.py`, 0.7.0 and 0.8.0 interleaved, two rounds of five
repetitions, ARM64 macOS with the `accel` extra; seconds, better round):

| Family (timing book, `standard`) | Cold 0.7.0 | Cold 0.8.0 | Speed-up | Warm 0.7.0 | Warm 0.8.0 |
|---|---|---|---|---|---|
| European | 0.774 | 0.022 | 35× | 0.0185 | 0.0150 |
| American | 0.765 | 0.026 | 29× | 0.0207 | 0.0207 |
| Snowball | 0.916 | 0.050 | 18× | 0.0434 | 0.0432 |
| Local-vol Snowball | 1.002 | 0.126 | 8.0× | 0.119 | 0.119 |
| Event statistics | 1.04 | 0.165 | 6.3× | 0.158 | 0.158 |
| Phoenix (memory) | 1.542 | 0.668 | 2.3× | 0.654 | 0.658 |
| Barrier (continuous) | 0.030 | 0.021 | 1.4× | 0.0282 | 0.0189 |
| KO-reset | 0.026 | 0.026 | 1.0× | 0.0254 | 0.0255 |

A cold solve is now dominated by its march: the Phoenix's mesh share fell from 0.89 s to
0.01 s, and the barrier and KO-reset cases build no concentrated mesh. Event statistics
from 15-repetition interleaved runs. Peak memory rises ~30 MB with Numba loaded.

### Per-flip deltas (PDE byte book, largest price moves)
- (a) grid day: 244-day environments only — Phoenix memory −2.7e-4, continuous-KI Snowball
  −1.6e-4, European −5.2e-8, intraday Snowball +6.4e-8. The intraday route's grid moved
  because its numerical environment has 244 days (F1).
- (b) cap: five-year standard Snowball −4.9e-7 (5,040 steps, was 4,980), three-year high
  −2.1e-6 (6,048, was 4,968).
- (c) projection: discrete barrier calls up to ±3.8e-2, double knock-out −2.8e-2, the
  down-and-in composite −8.0e-3, touches ~1e-3.
- (d) Snowball NODAL: +1.2e-2 to +1.8e-2 on the book's NODAL rows.
- (e) Phoenix θ: +1.6e-7 on the trading-clock Phoenix only.
- (f) half-step damping: local-vol Phoenix daily KI −3.6e-3, Snowball daily KI −3.3e-3,
  KO-reset discrete −1.6e-3, Snowball monthly −1.3e-4, discrete Snowball/Phoenix ≤ 9e-5,
  barrier/touch ≤ 3.6e-5, European ~1e-6. No continuous-KI row moved.
- (g) far field: puts (above); flat-curve barrier calls bitwise unchanged.

Goldens: replay frames (scalar_bsm, localvol) moved at (f) by ~1.6e-5 of notional;
execution goldens at (d) and (f) (`test/execution/goldens/PROVENANCE.md`).

## [0.7.0] - 2026-10-01

PDE 1D one core (spec: `docs/superpowers/specs/2026-09-30-pde-1d-one-core-design.md`;
architecture: `docs/pde-core.md`). Every shared-grid 1D PDE family (European, American,
the barrier family, Snowball, Phoenix, KO-reset, and the local-vol Snowball and Phoenix)
prices through one `prepare → march → read` path; the four per-family time-stepping loops
and the separate statistics sweep are gone. Numbers are unchanged: a 229-row `float.hex`
byte compare of the 1D book against 0.6.0 shows zero differences, and every banked
certificate keeps its identity. The scheme's settings move onto a stepper component, a
clean break with the old names raising `ValidationError` naming their replacement.

### Added
- `PDEParams.stepper`, `PDEParams.operator`, `PDEParams.readout`: component slots whose
  built-ins are `CrankNicolson()`, `CentralDifference()` and `StandardReadout()`
  (`quantark.asset.equity.engine.pde.core.components`). `CrankNicolson` carries the
  scheme's four settings: `theta`, `event_theta`, `terminal_damping_steps`,
  `event_damping_steps` (`None` = the route's defaults). A custom component runs on the
  core routes; the core validates its results and raises `NumericalError` naming it.
  `from_config` accepts `stepper: {theta: 1.0, ...}` and rejects `operator` / `readout`.
- `explain(product, pricing_env)` on `PDEEngine` and every core solver: the resolved plan
  (route, settings with their source and applicability, space, time, events, scheme,
  outputs, identities, per-operation terminal outcomes) from the same preparation `price`
  runs, without marching.
- `resolved_grid_evidence(params)` (`pde.grid.config`): the certificate `grid` record,
  the nine resolved grid fields plus the stepper's two damping counts under their 0.6.0
  keys. `PDEParams.evidence()` for `engine_config`.
- `OutputRequest`: what one call wants from a preparation (price, Greeks, event
  statistics, grid, life surface).
- `Calendar.identity()`: a value identity (name, sorted holidays, weekend days) for the
  settlement-calendar part of the schedule caches' keys.

### Removed
| Removed name | Replacement |
|---|---|
| `PDEParams(theta=...)` | `stepper=CrankNicolson(theta=...)`; 0.5 is Crank-Nicolson, 1.0 backward Euler |
| `PDEParams(event_theta=...)` | `stepper=CrankNicolson(event_theta=...)` |
| `GridConfig(terminal_damping_steps=...)` | `PDEParams(stepper=CrankNicolson(terminal_damping_steps=N))`; 0 disables |
| `GridConfig(event_damping_steps=...)` | `PDEParams(stepper=CrankNicolson(event_damping_steps=N))`; 0 disables |

`GridConfig` has nine fields and raises `ValidationError` (not `TypeError`) for the two
removed ones. The 0.6.0 entries that pointed at `GridConfig` damping counts
(`use_rannacher`, `rannacher_steps`, `rannacher_at_events`, `event_rannacher_steps`,
`DCNPDEEngine(rannacher_steps=)`) now point at the stepper. `TimeLayout` no longer carries
the damping index sets; `CrankNicolson.schedule` computes them.

### Changed
- The factorization caches key on the exact θ. Two θ agreeing to 6 (LU) or 12 (banded)
  decimals no longer share a matrix. `dt` stays rounded to 12 places, as every banked
  number was computed.
- Event statistics are a column group of the price march. `calculate_event_stats`
  no longer re-prices, and `price_with_events` marches once.
- Prepared sessions store the bound `Layout` (artefact builder `pde-prep/v2`). Bump
  contexts are fresh instances holding a frozen layout; the transient-attribute copy
  machinery is gone (the 2D Heston/SLV Greeks keep a legacy clone).
- Market memos are keyed by value fingerprints, so mutating a market object after a solve
  is no longer silently stale; an unfingerprintable input is sampled fresh.
- Certificate candidates are built from their declared params (`PDEParams.from_config`),
  so a declared setting is the one executed; a key the engine rejects fails the study at
  load.
- The DCN, local-vol vanilla/barrier and 2D Heston/SLV solvers read their damping counts
  and weights from `params.stepper`, and reject a custom component ("no core seam until
  sub-project 3"). `LocalVolPDESolver` checks route applicability on construction.

### Fixed
- Event statistics on a reused solver no longer reuse another market's factorizations
  (measured: a 0.041 KO-probability error on the second market).
- American Greeks on a fresh solver, for exercise with a settlement lag, no longer raise.
- PDE solvers deep-copy and pickle after pricing.
- A session no longer stores a factorization for a stepper without a value fingerprint
  (two such steppers could have shared one).
- Knock-in barriers (single and double) priced through a `PricingSession` now equal the
  direct price bit for bit. 0.6.0 marched the knock-out leg on the knock-in product's own
  spatial grid, masking the barrier instead of using the hard-edged domain (measured:
  up-and-in call −0.9%, double knock-in −2.2%). A session's layout is now refused, not
  reassembled, when the hard bounds differ. The session path was not in the byte book;
  it now has an execution test.
- A session with `cache_enabled=False` no longer warms a factorization pack.

Unchanged and still raising: bump-context Greeks on a knock-in barrier that has not
knocked in (`PricingError`: the frozen layout has the knock-in's bounds, the knock-out
leg needs the barrier's). It is a declared failure in the byte book; freezing the
knock-out leg's layout is sub-project 3 work.

### Sub-project 3
Still planned: the environment's business day as the grid day (full re-certification);
half-step damping as a `CrankNicolson` option; coefficient-break nodes; a budget default
for path-independent routes; failing instead of scaling at the caps; DCN and the
local-vol vanilla/barrier space axis onto the core; native KO-reset statistics. Added by
this release, each its own re-certified commit: statistics Level 2 (the statistics group
on the value policy with a marched maturity leg, memory-state Phoenix coupons); the
zero-diffusion θ override for Phoenix; Snowball `NODAL` on interior schedule events (or
withdrawing `NODAL` from the shared-grid table); native 2D Heston/SLV statistics.

## [0.6.0] - 2026-09-29

PDE 1D surface cleanup (spec: `docs/superpowers/specs/2026-09-28-pde-1d-surface-cleanup-design.md`).
Every 1D PDE route — the shared-grid solvers behind `PDEEngine`, the local-vol vanilla
and barrier solvers, and DCN — is configured by one `PDEParams` and nothing else, and
nothing is silently ignored: a field a route cannot honour raises. This is a clean break:
the removed names raise `ValidationError` naming their replacement. Numbers are unchanged
on every route except the local-vol discretely monitored barrier (see Changed), proven by
a 112-row `float.hex` byte compare of the whole 1D book against 0.5.0.

### Added
- `GridConfig.time_steps`: a total time-step budget over the horizon, spread over the
  mandatory intervals by length (at least one step each), and the alternative to
  `steps_per_day`. Exactly one time basis is in force after resolution; an overlay that
  sets one basis clears the other, and setting both raises.
- `GridConfig.evidence()` (the certificate serialization: unset optionals omitted, so no
  banked identity moves), `GridConfig.with_steps_per_day()` and
  `GridConfig.with_time_steps()`.
- `PDEParams.resolved_grid()`: the requested grid (the `accuracy` preset under the `grid`
  overlay), before the `max_steps` cap that binds once a maturity is known.
- The route applicability table (`quantark.asset.equity.engine.pde.route_applicability`,
  `ROUTE_RULES`): per route, the fields it cannot honour. An inapplicable field accepts its
  preset default and rejects an explicit override; a rejected field rejects any explicit
  value. The error names the field, the route, and sub-project 3.
- `PDEParams.from_config(config_or_path, **overrides)` over field names, with `grid` a
  nested `GridConfig` mapping (a `bounds` list becomes a tuple).

### Removed
`PDEParams`, `PDEEngine` and the DCN constructors take keyword arguments only; each
removed name raises `ValidationError` with its replacement:

| Removed | Replacement |
|---|---|
| `PDEParams(grid_size=...)` | grid=GridConfig(points=...) on every route. |
| `PDEParams(time_steps=...)` | grid=GridConfig(time_steps=N): the same meaning, a total step budget over the horizon. steps_per_day= is the per-day alternative. |
| `PDEParams(s_max=...)` | grid=GridConfig(bounds=(None, hi)) on the shared-grid solvers. On the two local-vol solvers an explicit bound is rejected until sub-project 3: the kernel sets its own domain and aligns the top, so there is no hard edge to honour. |
| `PDEParams(s_min=...)` | removed; it was read by nothing. |
| `PDEParams(use_rannacher=...)` | grid=GridConfig(terminal_damping_steps=N); 0 disables. use_rannacher=False also switched off event damping on the 2D Heston/SLV solvers; add event_damping_steps=0 to keep that |
| `PDEParams(rannacher_steps=...)` | grid=GridConfig(terminal_damping_steps=N); 0 disables |
| `PDEParams(rannacher_at_events=...)` | grid=GridConfig(event_damping_steps=N); 0 disables |
| `PDEParams(event_rannacher_steps=...)` | grid=GridConfig(event_damping_steps=N); 0 disables |
| `PDEParams(boundary_mode=...)` | removed; asymptotic boundaries are the only treatment |
| `PDEParams(continuous_ki_correction=...)` | removed; the first-passage correction always applies to continuously monitored KI |
| `PDEParams(ki_monitoring_mode=...)` | removed from the PDE; each barrier is monitored as the product declares it (discrete dates, continuous, or maturity-only). To price a dense discrete KI as continuous, declare it continuous on the product. QuadParams.ki_monitoring_mode is unchanged. |
| `PDEParams(cache_strategy=...)` | cache_enabled=True/False is the only cache control |
| `PDEParams(grid_cache_max_entries=...)` | cache_enabled=True/False is the only cache control |
| `PDEParams(banded_cache_max_entries=...)` | cache_enabled=True/False is the only cache control |
| `PDEParams(use_banded_solver=...)` | removed; the linear solver is chosen by the solver family |

| Removed | Replacement |
|---|---|
| `PDEEngine(method=...)` | Crank-Nicolson is the only built-in scheme; PDEEngine(params=PDEParams(...)). theta=1.0 gives backward Euler. |

| Removed | Replacement |
|---|---|
| `DCNPDEEngine / LocalVolDCNPDEEngine(num_space_nodes=...)` | DCNPDEEngine(params=PDEParams(grid=GridConfig(points=...))) |
| `DCNPDEEngine / LocalVolDCNPDEEngine(rannacher_steps=...)` | grid=GridConfig(terminal_damping_steps=N, event_damping_steps=N) (DCN applies both in its half-step form) |
| `DCNPDEEngine / LocalVolDCNPDEEngine(s_min_mult=...)` | removed; never read |
| `DCNPDEEngine / LocalVolDCNPDEEngine(s_max_mult=...)` | removed; never read |
| `DCNPDEEngine / LocalVolDCNPDEEngine(concentration=...)` | removed; never read |

Also removed: `BackwardOperator` (`backward_operator.py`), `ContinuousKICorrection`,
`make_pde_params`, `PDE_PARAM_PRESETS`, the `"pde"` branch of `make_engine_params` /
`list_param_profiles` (PDE has no profile tier: use `PDEParams(accuracy=...)`), the
class-level cache setters (`set_cache_enabled`, `set_cache_strategy`,
`set_grid_cache_max_entries`, `clear_grid_cache`), and `reject_legacy_resolution_knobs`.

### Changed
- **Local-vol vanilla and barrier solvers are configured by `PDEParams`.** Previously
  `accuracy` and `grid` were ignored there; they are now honoured or rejected per the
  route table. Their own preset table keeps today's plan at `standard` (400 points, a
  200-step budget); `fast`/`high` are new resolution choices. `theta` reaches both
  kernels; the vanilla's terminal damping is 0 or 1 (an explicit count of 2 or more is
  rejected). Bitwise at the defaults and for continuous and maturity-only monitoring.
- **The local-vol discretely monitored barrier observes on exact event nodes** (the
  declared number change). Each interior observation date is a node of the time layout,
  the step budget spread over the intervals between them, instead of the nearest node of
  a uniform grid. Byte gate: the two `lv_barrier_discrete` rows move by -7.9e-4 and
  -1.0e-3. Against a 1600 x 2400 reference PDE (every date on a node), flat 20% vol:

  | product | grid | base PDE | branch PDE | abs(base-ref) | abs(branch-ref) |
  |---|---|---|---|---|---|
  | UO call B130 monthly | 400x200 | 3.92225 | 3.92154 | 2.47e-03 | 1.75e-03 |
  | UO call B130 monthly | 600x240 | 3.92049 | 3.92049 | 7.03e-04 | 7.03e-04 |
  | UO call B130 monthly | 200x200 | 3.92822 | 3.92752 | 8.44e-03 | 7.74e-03 |
  | UO call B130 monthly | 800x400 | 3.91984 | 3.92013 | 5.49e-05 | 3.44e-04 |
  | UO call B130 monthly | 400x150 | 3.92460 | 3.92149 | 4.81e-03 | 1.70e-03 |
  | UO call B120 monthly | 400x200 | 1.76249 | 1.76185 | 3.05e-03 | 2.41e-03 |
  | UO call B120 monthly | 600x240 | 1.76040 | 1.76040 | 9.57e-04 | 9.57e-04 |
  | UO call B120 monthly | 200x200 | 1.77063 | 1.77001 | 1.12e-02 | 1.06e-02 |
  | UO call B120 monthly | 800x400 | 1.75965 | 1.75991 | 2.13e-04 | 4.74e-04 |
  | UO call B120 monthly | 400x150 | 1.76428 | 1.76179 | 4.84e-03 | 2.35e-03 |
  | UO call B120 quarterly | 400x200 | 2.16190 | 2.16190 | 1.04e-03 | 1.04e-03 |
  | UO call B120 quarterly | 600x240 | 2.16128 | 2.16128 | 4.18e-04 | 4.18e-04 |
  | UO call B120 quarterly | 200x200 | 2.16532 | 2.16532 | 4.46e-03 | 4.46e-03 |
  | UO call B120 quarterly | 800x400 | 2.16107 | 2.16107 | 2.06e-04 | 2.06e-04 |
  | UO call B120 quarterly | 400x150 | 2.16442 | 2.16189 | 3.55e-03 | 1.03e-03 |
  | DO put B85 monthly | 400x200 | 1.20049 | 1.20002 | 4.22e-03 | 3.75e-03 |
  | DO put B85 monthly | 600x240 | 1.19755 | 1.19755 | 1.27e-03 | 1.27e-03 |
  | DO put B85 monthly | 200x200 | 1.21140 | 1.21096 | 1.51e-02 | 1.47e-02 |
  | DO put B85 monthly | 800x400 | 1.19682 | 1.19701 | 5.46e-04 | 7.37e-04 |
  | DO put B85 monthly | 400x150 | 1.20189 | 1.19997 | 5.61e-03 | 3.69e-03 |
  | UI call B130 monthly | 400x200 | 4.90751 | 4.90822 | 1.83e-04 | 5.31e-04 |
  | UI call B130 monthly | 600x240 | 4.90791 | 4.90791 | 2.23e-04 | 2.23e-04 |
  | UI call B130 monthly | 200x200 | 4.90897 | 4.90967 | 1.28e-03 | 1.98e-03 |
  | UI call B130 monthly | 800x400 | 4.90809 | 4.90780 | 4.00e-04 | 1.11e-04 |
  | UI call B130 monthly | 400x150 | 4.90516 | 4.90827 | 2.53e-03 | 5.81e-04 |

  Closer to the reference in 16 of 25 cells; where the old grid misplaced a date by up to
  half a step (400x150) the error falls 2-3x, and elsewhere both placements sit inside
  the grid's own error band (every cell moves by <= 7e-4).
- **DCN is configured by `PDEParams`** with its own `accuracy` presets, which scale
  spatial resolution only (401 / 801 / 1601 points; `fast` and `high` have no banked
  evidence). The observation schedule is the time resolution: an explicit
  `steps_per_day` or `time_steps` is rejected, as is an explicit CELL_AVERAGE. The damping
  counts drive DCN's half-step form; `theta` and `event_theta` reach the loop. Bitwise at
  the defaults.
- **`ki_monitoring_mode` is removed from the PDE.** Each barrier is monitored as the
  product declares it. The two honest alternatives: `QuadParams.ki_monitoring_mode`
  (BGK stays in QUAD), or declare the KI continuous on the product.
- **`event_projection` defaults to `None`**, the route's own treatment (CELL_AVERAGE on
  the shared-grid autocallable solvers, as before).
- **`PDEEngine(params)` is the whole constructor**; Crank-Nicolson is the only built-in
  scheme. `theta=1.0` is backward Euler: first order in time and more diffusive, not
  "monotone" (monotonicity also needs the spatial operator's sign condition).
- The first-passage continuous-KI correction and the asymptotic boundaries are
  unconditional (their opt-outs are removed).
- **2D Heston/SLV damping:** event damping no longer follows the terminal switch
  (0.5's `use_rannacher=False` turned both off; set `event_damping_steps=0` too to keep
  that), and the 2D solvers read `terminal_damping_steps` as on/off (`> 0`).
- A budget-basis `GridConfig` resolves `day_count` to `None` (the day basis is not in
  force), so a resolved grid (`PDEParams.resolved_grid()`) can be fed back as an overlay.
- The grid "day" remains 1/252 of a solver year (`GridConfig.day_count`); it becomes the
  environment's business day in sub-project 3.

## [0.5.0] - 2026-09-28

Intraday pricing: `quantark.intraday` values a contract at a timezone-aware timestamp
instead of a date (spec: `docs/superpowers/specs/2026-09-15-intraday-pricing-design.md`),
and its certification lives in `quantark.modelvalidation`, not in the runtime
(spec: `docs/superpowers/specs/2026-09-18-intraday-modelvalidation-certification-design.md`,
reference revision: `docs/superpowers/specs/2026-09-18-intraday-deterministic-reference-revision.md`).
A certificate constrains the release, never the code. The autocallable engines agree on
the maturity close, on an observation at the valuation instant, on the knock-out-reset
snowball's contract rules and on a knock-out and knock-in at one instant (see Fixed).

The release also carries what `main` gained since 0.4.7: the simulated-path backtest,
futures bucket hedging in the replay engine, the opt-in QUAD V2 engines, PnL explain,
vol-model calibration and the Greeks facade (see "Also in this release").

### Added
- `quantark.intraday`: `value_intraday(request)` and `PricingSession.value_intraday`
  return an `IntradayValuationResult` with PV components, cashflows, a status per
  requested output and the provenance of every assumption. Three clocks: seconds-exact
  calendar time for carry and discounting, a versioned `VarianceProfile` (`desk`,
  `uniform`, `sessions_only`) mapped by `IntradayTimeMap` into `TradingClockVolSurface`
  for variance, and the contract's own day count for accruals. `TradingSessionCalendar`
  declares sessions, breaks, early closes and the payment time; a DST gap or fold is an
  error, never a guess.
- Fixings: confirmed `Fixing`s replay through the lifecycle trackers; a fixing that is due
  but not published is replaced by the latest spot and recorded as an `AssumedFixing`,
  and the flows it can change carry that dependency. `ObservationRecord.observation_timestamp`
  gives a timed observation (intraday only).
- Routes: QUAD V2 (snowball, Phoenix, knock-out-reset snowball), PDE (with an
  `under_resolved` diagnostic when the diffusion layer is not resolved), Monte Carlo,
  analytical digital (exact zero-variance limit) and analytical barrier/one-touch (gated
  by an exact admissibility test), behind an explicit capability matrix with fail-closed
  lookup (`docs/execution/intraday-capability-matrix.md`).
- Greeks: desk-bump Greeks on the frozen float-time twin, point Greeks per route
  (`undefined` at a payoff kink), local theta with a declared step and unit,
  `roll_context` and `roll_through_events`.
- Batch: `value_intraday_many`, prepared `spot_curve`s and provenance-preserving
  `aggregate_intraday`.
- Lifecycle: trackers declare coupon arrears and a settled terminal claim.
- `quantark.modelvalidation` schema 2: a quantity catalogue with per-quantity budgets,
  a `context` block with per-case overrides, semantic `expect:` cells, per-case random
  substreams, convergence axes with at least three levels, three-way verdicts on cells
  and aggregates, and a `contract` block an amendment may not change. Schema-1 studies,
  evidence and wire format are unchanged.
- A deterministic reference kind (`reference_kind = "deterministic"`): a typed record
  with a radius per quantity that declares itself `analytical` or a `calibrated_estimate`,
  consumed whole by the gate (`bounds.radius_budget_fraction`, no interval multiplier) and
  added linearly in the aggregate. `reference_qualification` names a stochastic arm that
  must agree with it case by case; an unqualified case is not decision-eligible, and the
  arm's sampling is frozen in the contract.
- Anchors of quantities that difference prices: a candidate may declare
  `anchor_noise_weights(case)`, the L1 weight of each quantity's price stencil, and off the
  banking machine such an anchor is compared at the prices' tolerance carried through that
  stencil (`modelvalidation.anchors.anchor_tolerance`). The intraday snowball candidates
  declare the runtime's desk bump and theta step. A one-second desk theta had moved 2.2e-8
  relative on x86_64 while the prices it differences agreed within 1e-9; the new tolerance is
  at most 4e-8 of a spot Greek's column scale and 2e-6 of desk theta's. Same-machine
  comparison stays exact.
- `python -m quantark.modelvalidation bank <run dir>`: validates, refuses quick runs,
  extracts anchors and never overwrites a banked directory.
- Study `snowball_intraday_daily_ki_bsm.yaml`: the daily-KI snowball on the intraday
  clock, 23 cases, QUAD V2 and PDE intraday routes against the engine-independent
  Gaussian-transition reference, qualified by paired RQMC. Banked 2026-09-19
  (`docs/modelvalidation/certificates/snowball-intraday-daily-ki-bsm/2026-09-19`, digest
  `a0569e97687cc01c...`): `SnowballQuadEngineV2` ADMITTED, `SnowballPDESolver` REJECTED.

### Removed (breaking)
The import shims deprecated in 0.4.x are removed, as announced:
- `quantark.backtest.otc` and its twelve modules. Import from `quantark.backtest.replay`
  (`config`, `single`, `market`, `engine_factory`, `strategy_state`, `results`,
  `product_replay`, `dashboard`), `quantark.param.vol.surface_history` (was
  `otc.vol_history`) and `quantark.volcalibration.calibrate` (was `otc.vol_calibrators`).
- `quantark.volmodels.calibration`: use `quantark.volcalibration.calibrate` (the same module;
  a monkeypatch on its kernels reaches the calibrator as before).
- `VolModelCalibrationConfig` and `HESTON_PRESETS` are no longer re-exported from
  `quantark.backtest.replay` or `quantark.backtest.replay.config`: import them from
  `quantark.volcalibration.config`.

### Changed
- PDE delta and gamma are read off a local cubic through the four log-spot nodes around
  the spot (`BasePDESolver._calculate_delta_gamma`; a three-node grid keeps the quadratic).
  The nearest-node quadratic carried that node's constant curvature to the spot: an O(dx)
  gamma error whose sign flipped as refinement moved the nearest node across the spot.
  Every PDE engine's delta and gamma move; PV does not. The replay goldens moved by at most
  1.2e-4 of scale in delta and 3.6% in gamma. The four studies that anchor PDE Greeks were
  re-certified and banked 2026-09-28 with every decision unchanged (ADMITTED):
  `snowball-flat-bsm` (`ccba1c32d4055333...`), `phoenix-flat-bsm` (`8cd5584888009924...`),
  `ko-reset-flat-bsm` (`722b7726fadc4d41...`) and `snowball-localvol-1d`
  (`c0d62f9f4c0986a9...`); their predecessors are retired with `superseded_by`. On those
  cells the readout is accuracy-neutral (worst cells move by at most 0.004 of a
  0.5-contract bound).
- `quantark.intraday` returns what it computes. Every requested output carries its own
  status (`ok`, `undefined`, `failed`, `not_requested`) with diagnostics; the `unqualified`
  status, the `certify=` flags, `GreekValue.error_budget` and the packaged Gate C evidence
  (`quantark/intraday/evidence`) are removed. The PDE resolution verdict is `under_resolved`.
  The runtime reads no certificate and does not import `quantark.modelvalidation`.
- The Gate C evidence is archived under `docs/modelvalidation/legacy/intraday-gate-c/2026-09-18/`
  with an inventory of what each fixture's successor study is.
- A knock-out and a knock-in observed at the same instant: the knock-out wins, on every
  autocallable and every engine (desk ruling, 2026-09-21). The knock-in observed with it neither
  disables that knock-out under `disable_ko_after_ki` nor, on a knock-out-reset snowball,
  replaces the first schedule. The rule is for a discretely observed knock-in; a continuously
  monitored level breached at an observation was touched before it and keeps its precedence.
  One spot breaches both only when the knock-out level is at or below the knock-in level, so the
  usual contract is unchanged: 240 prices (three products, Monte Carlo, PDE, QUAD V1 in both
  event projections and QUAD V2, discrete and continuous knock-in, flag on and off) are
  byte-identical before and after. On a tie contract the engines used to disagree (spot 70,
  knock-out at 60 under a knock-in at 75, monthly observations): the snowball under
  `disable_ko_after_ki` was -22.4 on Monte Carlo, PDE and QUAD V1 against 14.0 on QUAD V2; the
  Phoenix 82.1 on Monte Carlo and both QUAD engines against 100.2 on PDE; the knock-out-reset
  snowball -21.6 against 14.0 on QUAD V2, with or without the flag. All four now agree with a
  semi-analytic reference (the snowball under the flag with two observations: 3.0803, where the
  knock-in winning gives -18.1045).
  Changed: `SnowballMCEngine` (both products) and `PhoenixMCEngine` (a knock-out at the knock-in
  instant is valid), `SnowballPDESolver` and `KOResetSnowballPDESolver` (the knock-out jump on
  the not-knocked-in surface goes after the knock-in jump where the knocked-in surface does not
  carry it, at interior observations and in the terminal condition), `SnowballQuadEngine`,
  `KOResetSnowballQuadEngine` and `PhoenixQuadEngine` (the same, in both event projections),
  and QUAD V2's Phoenix, which let the knock-in win under the flag on purpose. `PhoenixPDESolver`
  already let the knock-out win. The event-statistics sweeps of the PDE and QUAD V1 engines are
  not reordered, and the two-dimensional vol-model solvers were not examined.

### Fixed
- The maturity close under `BEFORE`, where every remaining event is decided on the known
  spot. The QUAD V2 intraday route raised there; the Monte Carlo route priced a claim that
  knocks in or out at that instant as if it had survived (12 where the lifecycle pays -30 at
  spot 70), and the PDE route paid a knock-out decided there as a survival (the rebate, not
  the knock-out coupon; its desk theta one hour earlier was wrong by 7 per hour). All three
  routes now resolve the decided claim through the runtime's own event resolution.
- The same defect at its source: the zero-maturity shortcut of `SnowballMCEngine`,
  `SnowballPDESolver` and `SnowballQuadEngine` returned the terminal payoff of the carried
  knock-in state and never applied the observations still pending at that instant, although
  each engine decides such observations correctly whenever more of the claim lies ahead.
  They now share `SnowballOption.decide_observations_at_valuation`. Day-level dated products
  cannot reach this state (valuation on the exercise date is refused); a time-based product
  at zero maturity can.
- The same defect in the other two autocallables. At zero maturity `PhoenixMCEngine`,
  `PhoenixPDESolver` and `PhoenixQuadEngine` returned the redemption alone: the last coupon,
  the memory arrears it releases, a knock-out and a knock-in at the last observation were
  dropped (Monte Carlo and QUAD V1 also dropped a knock-in carried from an earlier day). The
  knock-out-reset snowball's shortcut in `SnowballMCEngine`, `KOResetSnowballPDESolver` and
  `KOResetSnowballQuadEngine` ignored the pending knock-out of the schedule in force (pre-KI
  before a knock-in, post-KI after one), and QUAD V1 the carried knock-in. They now call
  `PhoenixOption.decide_observations_at_valuation` and
  `KnockOutResetSnowballOption.decide_observations_at_valuation`. The Monte Carlo and PDE
  intraday routes resolve the maturity close for all three autocallables; the knock-out-reset
  snowball, which the runtime cannot replay, is decided from its float-time twin, and its QUAD
  V2 route no longer raises there.
- The knock-out-reset snowball's two contract rules, on every engine: a knock-in REPLACES the
  first knock-out schedule by the second and stays to the end, and the second schedule only
  becomes effective through a knock-in DURING the first, so a contract not knocked in when the
  first schedule ends has matured there (`KnockOutResetSnowballOption.first_schedule_ended`,
  `require_alive`). `SnowballMCEngine` kept testing discrete knock-in records after the first
  schedule had ended (3.41 against 5.94 from PDE and QUAD on a contract with monthly knock-in
  records for a year and a six-month first schedule), read a spot-dependent rebate off the end
  of the time grid instead of the end of the first schedule, and ignored a knock-in carried
  into the valuation. `KOResetSnowballQuadEngine` always priced the not-knocked-in state and
  refused a contract whose first schedule had ended; it now reads the knocked-in surface, on
  a lattice pinned to the second schedule's level. Every engine, and the intraday twin, priced
  or refused the wrong way round past the first schedule: a knocked-in contract is now valued
  there on all of them, and one not knocked in is refused as matured instead of being priced
  with the first schedule extended to the final maturity.
- `SnowballMCEngine`, knock-out-reset contracts: the second schedule is in force from the
  observation that knocks the contract in, that observation included, as in the PDE and QUAD
  engines (and in this engine's own REBASED branch). Monte Carlo counted it strictly after the
  knock-in, which only shows when the second schedule's level is at or below the knock-in
  level: 4.60 against 4.75 there, now 4.74 +/- 0.01. The usual contract is unchanged.
- `PhoenixQuadEngine` never read `CouponBarrierConfig.initial_coupon_arrears`: memory coupons
  owed at the pricing date were worth nothing. They are released by the first period that pays,
  as the Monte Carlo and PDE engines release them.
- `AutocallableLifecycleTracker.settle_maturity_if_due` raised `TypeError` for every Phoenix
  alive at maturity: it called `get_payoff` with the snowball's signature. This reached the
  portfolio lifecycle manager and the backtest replay, not only intraday valuation.
- `PhoenixPDESolver`: a knock-out decided at the valuation instant (any instant, not only
  the last) left out the memory coupons in arrears, which Monte Carlo, QUAD V2 and the
  lifecycle tracker release with that period's coupon (101 against 106 with five owed).
- An observation at the valuation instant with more of the claim ahead. `PhoenixQuadEngine` and
  `KOResetSnowballQuadEngine` skipped it (their time grid drops time zero): no knock-out, no
  coupon, no missed-coupon memory and no knock-in from it, so a Phoenix observed now at 104 was
  94.54 against 101 on PDE and QUAD V2, and one at 90 exactly one coupon short.
  `SnowballMCEngine` refused a knock-out-reset contract observed now (`dt_array must have
  strictly positive entries`, in pricing and in the event statistics): its grid kept time zero as
  a simulation node where the plain snowball's grid reads such a record off the known spot. All
  three now decide the observation on the known spot, and each engine satisfies the identities
  that need no second engine: a coupon observed now raises the price by exactly the coupon, a
  memory coupon missed now equals the contract carrying it as arrears, and a knock-in observed
  now equals the contract carrying the knock-in (to 1e-8, on all four engines).
- A Phoenix coupon pays principal x rate x the period's own fraction. The daily lifecycle
  tracker passed no fraction to `get_coupon_payoff` and got a silent 1.0, so a monthly
  product paid about 12x more through the tracker than through any engine. The quotation
  is now declared, `AccrualConfig.is_annualized_coupon` (default annualized, as every engine
  already priced: engine PVs are unchanged), and `get_coupon_payoff` resolves the contract's
  own fraction or fails closed. The tracker also kept missed memory coupons only as a count
  and returned before the coupon loop on a knock-out; both now match the engines.

### Also in this release (merged to `main` since 0.4.7)
- `quantark.backtest.simulation`: simulated-path ensemble backtest for autocallable books.
  `MarketPath` batches from a stationary block bootstrap of the joint spot/vol/carry
  history, GBM or designed stress paths; the CFFEX IM chain read off each path's carry
  curve; a vectorised lifecycle and futures ledger; exact, spot-ladder and PDE
  life-surface repricing providers with in-memory and on-disk state caches; a sampling
  accuracy gate; spawn-pool path batching; a conformance oracle against the replay
  engine; distributions with expected shortfall, paired comparisons on matched paths and
  run persistence. Worked study: `example/snowball_simulated_paths`.
- Replay engine, futures carry buckets: immutable carry scenarios, signed bucket
  aggregation, joint delta and parallel carry hedge strategies, a multi-leg futures ledger
  with atomic execution, carry audits by independent repricing, stress of unquoted carry,
  and an opt-in term-structure dividend source.
- QUAD: opt-in QUAD V2 Gaussian autocallable engines (`quantark.asset.equity.engine.quad.v2`:
  `SnowballQuadEngineV2`, `PhoenixQuadEngineV2`, `KOResetSnowballQuadEngineV2`,
  `AutocallableQuadEngineV2`). `QuadParams.readout` selects `"legacy_linear"` (default) or
  `"transition"`, which evaluates the final transition at the spot instead of interpolating
  between nodes (no delta staircase); `QuadParams.align_cell_stretch` (opt-in, `None` by
  default) puts every barrier on a node at once. Defaults leave every price unchanged.
- `quantark.pnlexplain`: waterfall and Taylor attribution of PV changes, lifecycle
  transitions, bucketed factor coordinates, trading-clock theta, and portfolio, position
  and quoted-leg explain with a replay recorder.
- `quantark.volcalibration`: market quotes to IV surface to LV/Heston/SLV calibration,
  with quote normalization, admission checks, a run store and a CLI
  (`python -m quantark.volcalibration`).
- Risk measures: the `GreeksCalculator` facade over a `GreekDef` registry, higher-order
  Greeks (speed, zomma, dividend volga, analytical closed forms) and a dual-clock theta
  suite (charm, color, vega/gamma theta, 1D/1TD, exact mode).
- Changed: scalar rho parallel-shifts the rate curve instead of replacing it with
  `FlatRateCurve(r(T) + bump)`, which measured a curve reshaping on any term structure
  (98x the parallel-shift rho on a 1%/3%/5% curve); flat curves are bitwise unchanged. On
  a trading clock the vol bump unit is one point of the trading-quoted sigma.


## [0.4.7] - 2026-08-25

Settlement-date-aware payoff discounting across the equity option stack
(spec: `docs/superpowers/specs/2026-07-29-equity-option-settlement-date-support-design.md`,
plan: `docs/plans/2026-07-29-equity-option-settlement-date-support.md`).
Determination and payment are now two separate clocks: underlying dynamics,
monitoring and optimal exercise stop at determination; every cashflow
discounts at its own payment time with curve-exact discount factors.

Alongside it, autocallable batch pricing gets its performance pass — the PDE
solve loop no longer re-serializes the product on every time step, and the
QUAD event-stats recursion gains a fused KI walk plus an opt-in forward
transition-density mode (`docs/autocall-engine-perf/`).

### Added
- `quantark.asset.equity.settlement`: shared settlement timing kernel —
  `SettlementConvention` (business-day / calendar-day / year-fraction lag
  with business-day adjustment and calendar), `SettlementRequest`,
  `ResolvedPaymentTiming` (determination/payment times + curve-exact
  `determination_df`/`payment_df`/`delay_df`), `SettlementResolver`
  (`resolve_contingent` / `resolve_pending`). No fabricated dates: a
  date-based convention on a time-only determination raises.
- Product contract: `BaseEquityOption.settlement_convention`; the existing
  terminal `settlement_date` is now honored by engines as the
  terminal-payment override; per-record `ObservationRecord.settlement_date` /
  `settlement_time` drive event-cashflow payment (KO redemptions, coupons,
  hit-paid rebates) across analytical, MC, PDE and QUAD engines for the
  BSM / Local Vol / Heston / SLV families.
- American exercise obstacle reflects delayed cash receipt in PDE and LSMC
  (node-specific `delay_df`); analytical American approximations raise a
  capability error for settlement forms they cannot represent.
- Engines declare settlement/timing capabilities and raise precise
  capability errors before numerical work rather than silently pricing at
  determination.
- Lifecycle: pending-cashflow ledger
  (`quantark.asset.equity.lifecycle.cashflows`) carries
  determined-but-unpaid receivables at `amount * DF(valuation, payment)`
  through payment; `price(..., lifecycle_state=...)` propagates through
  engines, positions, portfolio valuation, Greeks and the consolidated
  replay backtest (settlement decomposition; KO termination waits for
  terminal cash to land).
- Payment-aware event statistics: determination/payment time arrays with
  undiscounted and discounted expected cashflows that reconcile to engine
  PV.
- QUAD autocallable engines: opt-in `QuadParams.event_stats_mode="forward_density"`
  computes the event distribution from a forward transition-density march
  (2–3 surfaces instead of one indicator row per KO observation). `npv` is
  unchanged (always the backward value solve); distribution fields differ
  within the banked validation tolerances
  (`docs/autocall-engine-perf/FORWARD-DENSITY-EVIDENCE-2026-08.md`).
  KO-reset ignores the flag. Default remains `"stacked"`.

### Changed
- Snowball/Phoenix PDE: the product cache token is now memoized per solve
  (`SnowballPDESolver._product_token_memo`, cleared at every solve entry).
  The boundary-condition path previously re-serialized the whole product —
  every KO/KI schedule record — once per time step, >90% of a snowball
  solve; bit-identical output, ~7× faster per solve
  (`docs/autocall-engine-perf/FINDINGS-2026-08-24.md`).
- QUAD stacked event stats: the KI-probability recursion now rides the main
  stacked recursion as fused surfaces (bit-identical output, one fewer full
  time loop per `price_with_events`); identically-zero indicator rows are
  skipped in the diffusion/bridge steps and the row-independent bridge
  kernel stack is cached per engine (64-entry bound). Bit-identical output.
- QUAD `price_with_events` now honors `streams` pruning [§11.1], mirroring
  the PDE contract: KI-probability recursion and Phoenix coupon rows are
  skipped when no requested stream reads them. Callers that pass `streams`
  (the execution framework's greek-bump runner does) get `0`/empty for
  pruned distribution fields where QUAD previously returned computed values;
  `ko_probability`/`survival`/`pv` and all requested fields are unchanged.
  `streams=None` keeps the old full computation.
- **The flat dividend/carry sanity bound moves from 20% to 100%**
  (`ContinuousDividendYield`, the BSM process gate, and the market-data
  point model), matching `TermStructureDividendYield`'s own magnitude
  bound. OTC borrow/lending carry (融券率) legitimately exceeds 20% on
  some trades; the 100% ceiling still rejects unit errors (entering
  `26.89` for `0.2689`). Values between 20% and 100% now price instead of
  raising `ValidationError`.

### Fixed
- Repository tooling only — not shipped in the wheel, which contains
  `quantark/` alone. The `mo_volmodels` dashboard decided whether a results
  aggregate was stale by comparing two ISO-8601 stamps as strings, which
  orders correctly only when both carry the same UTC offset. Read from a
  zone other than the one that wrote the cells, an aggregate *newer* than
  the cells it summarises was reported stale. It now compares instants.
- Repository tooling only. The same dashboard read git commit dates with
  `datetime.fromisoformat`, which rejects an RFC 3339 `Z` suffix on Python
  3.10 (3.11 was the release that learned it). `git log --format=%cI`
  renders a UTC commit date as `...Z` on some git versions, so reading a
  repository committed under UTC raised `Invalid isoformat string` on the
  oldest supported Python. All three readers now share one parser that
  accepts both spellings.

### Compatibility
- Zero-lag identity: with no settlement terms supplied, payment equals
  determination (`delay_df == 1`) and all prices, Greeks, event statistics
  and exercise boundaries are unchanged — verified by the existing suite
  plus the dedicated settlement test files.

### Also in this release
- Constructor normalization forwarding the shared lifecycle fields across
  the concrete equity option products (wip/equity-option-contract).
- Heston temporal regularization in the ADI core
  (wip/heston-temporal-regularization).
- PDE regression coverage ports, per-module reference docs, example suites
  (FX vol models, snowball backtest terms, bucketed-greeks scenarios), and
  modelvalidation certification archival.

## [0.4.6] - 2026-08-20

0.4.5's test suite passed on the ARM64 machine its goldens were frozen on and
failed 28 tests on x86_64 Linux. Chasing that down found one genuine pricing
defect and a set of comparisons that were architecture-dependent by accident.
The published 0.4.5 package is unaffected except as noted below.

### Fixed
- **Parallel-batch pricing of continuously monitored knock-in under Local Vol /
  Heston / SLV Monte Carlo.** The per-step variance the Brownian-bridge crossing
  estimate needs was held in a single engine attribute, written while simulating
  and read while computing payoffs. The dask path (`use_dask=True`) submits every
  batch through `delayed`/`compute` on the threaded scheduler, so batches sat
  between their own generation and their payoffs concurrently on one engine and
  overwrote each other — 20,000 paths over 3 batches is 6667/6667/6666. Now
  scoped per thread, which is where a batch begins and ends. A shape guard made
  this raise rather than price one batch against another's variance, so no wrong
  number could have been returned; serial pricing was never affected.
- `quantark.modelvalidation.anchors`: comparing a banked certificate off the
  banking machine now floors the tolerance at the suite's cross-architecture
  standard (`DEFAULT_REL_TOL` 1e-12 → 1e-9, `DEFAULT_ABS_TOL` 1e-15 → 1e-12).
  Measured across 326 anchors, x86_64-vs-ARM64 drift spans 1e-14 to 2.2e-11
  relative (median 2.0e-12, both signs, no outlier of a different order): last-ULP
  differences in FMA contraction, libm transcendentals and BLAS kernels, amplified
  by an autocallable PDE/QUAD solve marching hundreds of steps through barrier
  events. The bound keeps ~44x margin over that and stays six orders under any
  real numerics change. Applied as a floor on each banked file's own tolerances,
  so already-banked evidence is corrected without its anchored *values* being
  touched. Same-machine comparison remains exact.

### Changed (tests)
- The replay golden gate compares floats against each column's scale rather than
  bitwise. These frames carry bump derivatives, so the rounding noise is set by
  the magnitude of the price being differenced and is near-constant in absolute
  terms down a column: gamma drifts 4.6e-10 against a column max of 43, and the
  same quantity in cash drifts 3.8e-8 against 4,301 — one ratio, ~1e-11. Column
  names and order, row count, and integer, string and date columns still compare
  exactly. New tests pin the tolerance itself: it admits the measured drift,
  rejects 1e-6 of column scale, and every real golden frame is checked against a
  simulated drift locally, because bitwise-identical frames short-circuit before
  any tolerance is consulted and made earlier calibration bugs invisible here.
- Two cross-architecture anchor tests hardcoded the fingerprint
  `{"machine": "x86_64", "system": "Linux"}` — which is the CI machine, so there
  they took the same-machine *exact* branch, the opposite of what they verify.
  They now derive a fingerprint that cannot be the current machine.
- `test_event_stats_pruning` pinned a PV of ~9.9e5 with a `1e-6` absolute
  epsilon, i.e. 1e-12 relative — tighter than the measured drift, passing on
  margin rather than by design. Now on the documented cross-arch tolerance.
- A settlement fixture placed a strike exactly on the parity-implied forward.
  Stage 10 splits OTM puts from calls with `strike < forward` against an
  OLS-regressed forward, so that strike changed side with the last ULP of the
  regression. The grid now keeps every strike clear of the forward.

## [0.4.5] - 2026-08-19

Two programs land together: **model-release certification** — the first
module that decides whether an engine is fit to ship, and the numerics
fixes that decision surfaced — and the **backtest replay consolidation**.

### Model-release certification

#### Added
- `quantark.modelvalidation`: engine-release certification. A study YAML
  declares the product, the market environment, the candidate engines and
  the economic scale; the runner prices every (candidate, case) cell against
  a paired-RQMC benchmark, applies per-cell and aggregate-bias gates, and
  emits a signed certificate, a Markdown/HTML report, and an *anchor file*.
  Anchors are the cheap residue of an expensive run — the deterministic
  engines' own outputs at the certified configurations — so CI can re-check
  in seconds that a released engine still produces what its certificate
  claims (exact on the banking machine, tolerance-based elsewhere).
  `amend` extends a certification's scope while carrying unaffected cells
  forward on identity hashes; scope may grow but never shrink.
  Procedure: `docs/modelvalidation/RELEASE_PROCEDURE.md`.
- Banked flat-BSM certifications for snowball (23 cases), phoenix (14) and
  KO-reset snowball (14), PDE and QUAD both ADMITTED, under
  `docs/modelvalidation/certificates/`, covering the full product feature
  surface: discrete / European / stepping knock-in, step-down and parachute
  knock-out, reverse, airbag, protection types, participation, call rebate,
  `disable_ko_after_ki`, and EXPIRY coupons.

#### Fixed
- **Continuous knock-in was monitored at the time-step width** by the
  local-vol, Heston and SLV solvers, which were gated off the FIRST_PASSAGE
  correction. The correction is barrier-local, so it never needed globally
  constant coefficients — only the dynamics at the barrier. Each solver now
  reports its own: `sigma_loc(barrier, t)` for local vol, one
  `(mu, sigma^2)` per variance column for Heston, and `L(barrier,t)^2 * v`
  for SLV. On a flat surface the local-vol solver had been bitwise identical
  to the *uncorrected* flat-BSM solver (+0.099% of PV). Applies to the
  snowball and phoenix families alike; the phoenix 2-D solvers had never
  built the state at all.
- **The Monte Carlo continuous-KI Brownian bridge used the implied vol at
  the strike** as the variance of every path and every step. The vol-model
  engines now supply the log-variance their own scheme accumulated. Against
  the closed-form first-passage probability the error falls from -10.0 / -8.5
  / -9.2 standard errors (Heston / local vol / SLV) to inside 4.
- The first-passage correction estimated its slope from a node that could sit
  a fraction of a cell above the barrier, where the knock-in mask's own step
  — not the slope — dominates. Sampling is now floored at half a cell.
- `PhoenixOption` never built its knock-in observation schedule, so a
  discretely monitored phoenix raised instead of pricing.
- `disable_ko_after_ki` was ignored by the snowball and phoenix PDE solvers,
  which wrote the knock-out payoff to the knocked-in surface as well.
- Phoenix `coupon_pay_type=EXPIRY` coupons roll up and are paid at
  termination — the knock-out settlement when it knocks out, maturity
  otherwise — and are never forfeited. All three engine families agree;
  PDE and QUAD gained a termination-value surface to price it.

#### Changed
- Continuous knock-in prices move in the local-vol, Heston and SLV engines
  (PDE and Monte Carlo). Discretely monitored, European and no-knock-in
  products are bitwise unchanged, verified across 54 engine/product/
  monitoring combinations; the sole exception is the opt-in
  `KnockInMonitoringMode.BGK_APPROXIMATION`, which replaces a discrete
  schedule with continuous monitoring by design.

### Backtest replay consolidation

(spec: `docs/superpowers/specs/2026-07-30-backtest-replay-consolidation-design.md`).

#### Added
- `quantark.backtest.replay`: canonical home of the product-replay backtest —
  ONE multi-product daily loop (`ReplayBacktestEngine`/`ReplayBacktestConfig`/
  `ReplayProduct`/`ReplayBacktestResults`); the single-product
  `AutocallableBacktestEngine` is now a book-of-one wrapper with byte-identical
  output (golden-gated). Book runs gain per-day vol-model calibration.
- `quantark.backtest.futures_ledger`: shared `FuturesHedgePosition` +
  `FuturesRollPolicy`.
- `quantark.backtest.metrics.CorePerformanceMetrics`: cross-asset metrics over
  the `BaseBacktestResults` protocol; replay results expose `.metrics` and the
  full protocol; equity `PerformanceMetrics` now extends the core.
- `quantark.backtest.replay.schema`: typed row schemas — the single source of
  truth for every record column.
- Pending-settlement KO termination: `terminate_on_lifecycle_end=True`
  (default) ends a replay when terminal cash lands (observation vs settlement
  are separate moments; discounted receivable carried in portfolio value);
  summaries gain `termination_reason` / `days_replayed` / `days_in_contract`.
- `AutocallableEngineConfig.event_stats_fallback` (`"none"` default): the
  silent 5000-path MC event-stats fallback is now opt-in, logged, and recorded
  per row in the new `event_stats_engine` column.
- `BumpConfig.gamma_spot_bump`: engine-side asymmetric gamma bumping;
  replay-side manual bumping removed (greeks are always
  `engine.calculate_greeks`, failures raise — no silent zero-delta days).
- `AutocallableDeltaHedgeStrategy` joined the `BaseStrategy` hierarchy
  (`quantark.backtest.strategy`).

#### Changed
- Relocations (old paths remain as `DeprecationWarning` shims until 0.5.0):
  `quantark.backtest.otc.*` -> `quantark.backtest.replay.*`;
  `otc.vol_calibrators` -> `quantark.volmodels.calibration`;
  `otc.vol_history` -> `quantark.param.vol.surface_history`;
  `_atomic_write_json` -> `quantark.util.io.atomic_write_json`.
- Replay books with `vol_model != "bsm"` reject non-Snowball products at
  config construction (the vol-model engine factory is snowball-only).
- **Heston/Heston-SLV Snowball and Phoenix 2D PDE solvers default
  `v0_boundary="degenerate_pde"`** (was the ADI core's `"neumann"`), and accept
  it as a constructor argument. Smile-calibrated Heston sits near the Feller
  boundary, where the Neumann treatment of the `v = 0` edge mis-prices by
  −0.540% of notional against a QE-M MC reference while the degenerate
  treatment holds +0.156%. `"neumann"` remains selectable for cross-checks.
  **Prices change** for these four solvers at default settings.
- **Heston/Heston-SLV Snowball concentrated PDE grids default to a power-graded
  variance axis** (`v_grid_power=2.5`) to resolve the near-zero layer around the
  Feller boundary. `v_grid_power=0.0` retains the legacy concentrated mesh for
  explicit cross-checks; uniform grids remain ungraded. **Snowball Heston/SLV
  prices and finite-difference Greeks change** at default settings.
- **The `mo_frozen` Heston preset enforces the Feller condition**
  (`enforce_feller=True`; the soft `regularize_feller` penalty is retained).
  Deliberate divergence from the `mo_volmodels` diagnostics scripts, which keep
  the soft-only policy. Feasibility is bought with smile accuracy — measured
  across 762 CFFEX MO settlement surfaces: median 8.4 bp, p90 29.3 bp, max
  217 bp of IV. On 6.6% of dates the constraint is met by driving vol-of-vol to
  its lower bound, which degenerates Heston to deterministic variance; consumers
  should screen on the new `feller_ratio`. Heston and Heston-SLV calibration
  cache entries are invalidated automatically (the fingerprint embeds the
  preset); `localvol` entries are unaffected.

#### Fixed
- `BaseEngine.calculate_greeks` and Gate G2's deterministic delta helper bypassed
  `create_bump_context`, so Heston/SLV central spot bumps rebuilt their S grids even
  though the solvers already provided a frozen-layout context. Base, up, down, and
  separate gamma reprices now all use the resolved base-market context.
- `Heston2DAutocallableSessionAdapter` now preserves the Snowball solver's
  `v_grid_power`; an explicit grading or legacy opt-out no longer disappears when
  the execution session clones the engine. Gate metadata records the selected
  value and Stage 12 forwards it into the production engine options.
- `calibrate_heston(enforce_feller=True)` failed closed on any problem where the
  constraint was active at the optimum and `ftol` was loose. SLSQP satisfies
  constraints only to about its own accuracy tolerance, but the feasibility
  buffer was a constant `1e-8` — measured slack is ~1e-9 at `ftol=1e-8` but
  ~1.6e-7 at `ftol=1e-6`, so the strict post-check rejected every fit. The
  buffer now scales as `max(1e-8, 10 * ftol)`; the post-check is unchanged and
  still fails closed.
- `VolModelCalibrator` Heston records carry `feller_ratio` (`2*kappa*theta /
  sigma**2`) alongside `feller_margin`. The margin is a difference and so is not
  comparable across dates; the ratio is.
- `Heston2DAutocallableSessionAdapter._clone_engine` rebuilds the engine from an
  explicit kwargs list and dropped `v0_boundary`, so a session or prepared-adapter
  run silently priced with the default boundary while the direct call honoured the
  constructor. Any 2D autocallable constructor argument omitted from that list
  fails this way — silently, with no error.

## [0.4.0] - 2026-07-27

Clean rewrite of the PDE grid construction and event application layer
(spec: `docs/superpowers/specs/2026-07-27-pde-grid-redesign-design.md`).

### Added
- `quantark.asset.equity.engine.pde.grid`: declarative grid layer —
  `GridRequest` (frozen geometry), `GridConfig` + accuracy profiles
  (`fast`/`standard`/`high`), ONE time builder (event-aligned, extras-scaled
  cap, damping schedules), ONE spatial builder (multi-point sinh, pinned
  auto-bounds, no barrier snapping), four-stage `EventSchedule`
  (terminal/interior/continuous/valuation-readout, pure block transforms),
  engine-owned `GridBinder` (LRU, `bind_shared` for same-underlying books,
  `rebind_time` for calendar rolls) and `GridLayerMixin` for
  BaseEngine-derived adopters.
- `PDEParams(accuracy=..., grid=GridConfig(...))` tiered front door.
- Frozen-`Layout` bump contexts: spot/vol/rate/div bumps reuse the base
  layout by object identity; calendar rolls rebind time on the same spatial
  object.
- `HestonSLVADICore(x_nodes=...)`: 2D solvers take their S-axis from the
  shared spatial builder.
- Tier-2 anchor certification (`test/pde_grid/test_anchor_certification.py`):
  MC/QUAD/closed-form/smoothness anchors gate every accuracy profile.

### Removed (breaking)
- Legacy grid modules `time_grid.py`, `spatial_grid.py`,
  `event_projection.py` (projection math moved verbatim to `grid/events.py`).
- `PDEParams` knobs: `auto_grid`, `time_grid_type`, `grade_exponent`,
  `event_min_steps_per_interval`, `log_dx_target`,
  `include_spot_in_critical_points`, `frozen_critical_points`,
  `barrier_refine_log_width`, `barrier_refine_levels`,
  `barrier_domain_expand`, `adaptive_grid`, `event_steps_per_day`,
  `max_time_steps`, `max_grid_size` (the last three had no remaining
  consumers — `GridConfig.steps_per_day` / `max_steps` / `max_points` own
  those roles).
- The class-level shared grid cache (`clear_grid_cache` is now a no-op).

### Changed
- All PDE prices reprice slightly (re-certified against MC/QUAD/closed-form
  anchors, goldens re-frozen): the layer builds concentration-based grids
  without barrier snapping — cell-average event projection is placement-
  independent — and event-aligned time grids emit one exact dt per interval
  (operator caches are arithmetic-neutral: cache-on == cache-off bitwise).
- Grid-layer solvers REJECT non-default `grid_size`/`time_steps`/`s_min`/
  `s_max` with a `ValidationError` naming the `accuracy`/`GridConfig`
  replacement (previously the values would have been silently inert); the
  knobs remain live inputs for the standalone vol-model solvers only.
- `make_pde_params` profiles (`fast`/`accurate`/`barrier_sensitive`/
  `reverse_sensitive`) now emit `accuracy`/`grid=GridConfig(...)` instead of
  the legacy knobs (which also fixes a crash: the product-hint path injected
  deleted `barrier_refine_*` keys).
- Bump contexts (`create_bump_context`) clone via `deepcopy`, preserving
  full constructor state (Heston `model_params`, SLV leverage surfaces, ADI
  dimensions); `PDEEngine.create_bump_context` carries the frozen solver
  instance. Frozen layouts fail closed when reused across different hard
  (absorbing-barrier) bounds, and calendar-roll rebinds coverage-validate.
- 2D Heston/SLV snowball/phoenix bump contexts freeze the ADI S-axis at the
  solve configuration (`n_x`, `num_std=8`) and reuse it by identity under
  market bumps (`_layer_x_nodes` routes through `resolve_bound_layout`);
  previously the S-axis was rebuilt per bump, mixing grid movement into 2D
  numerical greeks.
- The scheme knobs are LIVE on the grid layer: `use_rannacher`/
  `rannacher_steps` drive terminal damping and `rannacher_at_events`/
  `event_rannacher_steps` drive event damping (the binder derives the
  damping schedule from them; explicit `GridConfig` fields still win). At
  default knob values the schedule is unchanged.
- Prepared/session PDE pricing resolves solve state
  (`_prepare_for_request`) before fingerprinting grid geometry — BGK
  products no longer trip `DeterminismViolation` against their own
  preparation when an artifact cache is active.

### Fixed
- Autocallable PDE grids ignore out-of-reach critical-price markers (such as
  disabled 100× KO observations) for bounds and concentration while retaining
  the observation date and any coupon event. Dense critical sets now use
  bounded cumulative inversion of the Tavella-Randall monitor, removing the
  overflow-prone shooting path and its non-monotonic construction-time cliff.
- Snowball/Phoenix QUAD event handling now defaults to cell-average projection
  and phase-stable trapezoidal integration. Unreachable KO markers are
  suppressed without deleting their dates, and the legacy nodal/Simpson
  behavior remains available explicitly.
- QUAD can optionally refine nested odd grids until consecutive PV estimates
  satisfy configured absolute/relative tolerances. Fixed
  `grid_points=1001` remains a speed-oriented estimate rather than a universal
  convergence guarantee.

### Deferred (recorded follow-ups)
- Event-aligned 2D ADI time axis (uniform march retained).
- Standalone vol engines (`LocalVolPDESolver`, `HestonPDESolver`,
  `HestonSLVPDESolver`, LV/Heston/SLV barrier engines, `HestonDCNPDESolver`)
  keep bespoke kernels and their `grid_size`/`time_steps`/`s_min`/`s_max`
  inputs; `event_projection` + Rannacher scheme knobs survive for the live
  characterization paths.
- Stencil-delta accuracy on concentrated grids (~2e-3 absolute) —
  provisional 5e-3 anchor band.

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
