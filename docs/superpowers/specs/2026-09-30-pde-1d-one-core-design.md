# PDE 1D One Core — Design Spec (sub-project 2 of 3)

**Date:** 2026-09-30
**Status:** approved design, revised 2026-09-30 after three rounds of external design review (section 14); awaiting spec review
**Release:** 0.7.0 (clean break, matching the 0.4.0 / 0.6.0 convention: removed, not deprecated)
**Prerequisite:** sub-project 1 merged. Its implementation is complete on
`worktree-pde-1d-surface-cleanup` (18 commits ending at `302fa2ed`, the 0.6.0 release
note) and unmerged at the time of writing. The revised spec 1 text (section 12 of
`2026-09-28-pde-1d-surface-cleanup-design.md`) is uncommitted in the main checkout and
must land with that merge. This spec is written against the worktree's code.
**Scope:** every shared-grid 1D equity PDE route: the base family (European, American,
Barrier, DoubleBarrier, OneTouch, DoubleOneTouch), Snowball, Phoenix, KO-reset, and the
local-vol Snowball and Phoenix, plus the event-statistics sweep they share and every
entry point that prepares them (direct pricing, prepared sessions, bump contexts, intraday,
life surfaces, and the 2D solvers' 1D statistics). DCN, the local-vol vanilla and barrier
kernels, and the 2D Heston/SLV cores are touched only where they read the four scheme
settings (section 7).

This is the second of the three sub-projects of the "common core and numerical policy"
plan (spec 1, section 10). It is **bitwise**: no number moves on any route, proven by the
byte book of spec 1 section 8.1 extended with the rows this work newly exercises (section
9.1). It moves **no banked identity** (decision 1). Everything that would move a number is
named in section 8 and section 12 and scheduled for sub-project 3.

---

## 1. Problem

Sub-project 1 gave every 1D route one parameter surface. Behind it the implementation
is still six hand-written copies of the same backward march, each with its own
preparation, and the surface itself still splits the built-in scheme's four settings
across two objects. Measured in the worktree at `302fa2ed`:

- **Six θ-step loops inside the package**, seven counting the local-vol kernel outside
  it: the base sparse sweep (`BasePDESolver._time_stepping`), the snowball two-surface
  sweep (`_time_stepping_two_surface`, also used by KO-reset and the local-vol snowball),
  the snowball/phoenix event-statistics sweep (`_compute_event_stats`), the phoenix vector
  sweep (`_time_stepping_vector_surface` / `_step_grids`), DCN's `_theta_step`, and the
  local-vol kernel's own. Every loop iterates raw node indices; none iterates the
  `TimeLayout`. They differ in backend (SuperLU always; banded or LU; banded only;
  full-size banded without a cache), in how many columns one solve advances (one; two in
  one call; one at a time; 2·n_cols in one call), in boundary convention (set before the
  step and injected; carried forward then zero-slope after events; identity rows then
  linear extrapolation), in whether the zero-diffusion steps force θ=1 (base and
  snowball yes; phoenix and statistics no), in where the coefficients come from
  (session-aware `_step_coefficients_for_solve` for base and snowball; a direct
  `_build_step_coefficients` for phoenix, KO-reset, and statistics, which therefore
  ignore a prepared session's coefficients), and in the cache key (`round(theta, 6)` on
  the LU path, `round(theta, 12)` on the banded path, first build wins).
- **Three coefficient builders** (`_calculate_coefficients`, `_build_step_coefficients`,
  `_flat_exact_step_coefficients`) chained differently at seven call sites, with the
  local-vol classes overriding two of them.
- **Six preparation paths** (direct, session, bump, intraday, life surface, local-vol)
  with nineteen recorded divergences. Among them: the snowball builds its observation
  maps twice (once from `layout.step_at`, then cleared and rebuilt by a nearest-match
  scan); the bump path runs `_prepare_for_request` while every other path runs
  `_prepare_solve_state`; the session execute path builds a synthetic `SpatialLayout`
  with `achieved_eps = nan`; the statistics sweep re-derives the solve state inline; no
  session artefact fingerprint includes `theta` or `event_theta`; `_params_cache_key`
  and `_get_cached_critical_points` have no production caller.
- **Event statistics are a second march** with thirteen conventions of their own
  (section 6), and the maturity leg is a residual with `reconciliation_error` hard-coded
  to zero.
- **No `explain`.** The resolved plan is readable only through private attributes; the
  intraday route reads `_active_layout` and `grid_binder.config` and calls `build_time`
  itself to probe fills.
- **The built-in scheme's four settings sit two-plus-two** across `PDEParams` (`theta`,
  `event_theta`) and `GridConfig` (`terminal_damping_steps`, `event_damping_steps`),
  accepted in sub-project 1 because regrouping them would have moved the certificate
  identity (spec 1 decision 14, risk row).

## 2. Goals

1. One implementation of the march, one of coefficient assembly, one preparation path,
   one readout, with product families as adapters. The old loops are deleted, not wrapped.
2. Bit-identical numbers on every route in scope, proven by the extended byte book;
   anchors and the identity guard unchanged; no banked hash moves.
3. `engine.explain(product, env)` reports what a request will actually do, from the same
   preparation the march uses.
4. The four scheme settings live together in the built-in stepper's configuration; the
   three component seams (operator, stepper, readout) are public, with the built-ins
   behind them and a fingerprint rule that keeps certification honest.
5. Every cache is keyed on a complete numerical identity.

### Non-goals (deferred, section 12)

- Any change to the statistics conventions (Level 2), the Phoenix zero-diffusion
  override, the snowball's NODAL treatment on its schedule path, or any other number.
- DCN and the local-vol vanilla/barrier kernels onto the core; the 2D cores.
- The environment's business day as the grid day (spec 1 decision 6): sub-project 3.
- A second built-in stepper, operator, or readout. Sub-project 3's half-step damping is
  the first.

## 3. Decision log (settled with the user, 2026-09-30)

| # | Decision | Rationale |
|---|---|---|
| 1 | Relocate `theta`, `event_theta`, `terminal_damping_steps`, `event_damping_steps` into the built-in stepper's config **and keep every banked identity**: the certificate builders serialize a composed evidence record that carries the two damping counts under their historical `grid.*` keys (section 4.6). Spec 1 decision 14 is restated as "no banked identity moves in a bitwise sub-project" rather than "no `GridConfig` field changes". | Dropping the two `GridConfig` fields moves the hash of every banked PDE cell; an `amend` whose every cell moved re-runs every cell, a full certification for zero number change. The evidence record is the numerical fact, not a mirror of the public dataclass layout. |
| 2 | Event statistics join the price march as a **column group carrying its legacy policy** (Level 1, bitwise). Unifying that policy with the value policy and marching the maturity leg (Level 2) is sub-project 3's first statistics commit. | Thirteen conventions separate the two sweeps (section 6); a true fold moves every KO and KI probability. Per-group solve calls make the structural fold bitwise: the value columns keep their own solve with today's right-hand side. |
| 3 | Shape: one march with product adapters and column groups, over two private arithmetic backends. No new grid or event framework; `TimeLayout`, `SpatialLayout`, `GridRequest`, and `EventSchedule` are reused. | A thin extraction leaves five orchestrations and gives `explain` nothing to report from; a new framework reopens the grid layer that is the certificate identity. |
| 4 | Three public component slots on `PDEParams`: `stepper`, `operator`, `readout`, with built-ins `CrankNicolson`, `CentralDifference`, `StandardReadout`. | Spec 1 decision 7: constructor arguments beside `PDEParams` are a second configuration surface. An ordinary user never sets them. |
| 5 | The damping schedule is the stepper's. `TimeLayout` is geometry only; `CrankNicolson.schedule(layout)` computes the per-step θ exactly as `_theta_schedule_from_layout` does today. | The counts are scheme settings; sub-project 3's half-step damping is a stepper policy. |
| 6 | The factorization cache keys on the exact **θ** value and keeps today's `round(dt, 12)`. Today's per-backend θ rounding is a defect (two θ values agreeing to six decimals share a matrix, first build wins) fixed in its own commit before the core lands. The `dt` rounding is characterized and kept: a monthly snowball's twelve intervals produce five distinct `dt` floats within 1.3e-18 of each other that share one factorization today; exact `dt` keys would build five and move its price at the 1e-13 level (measured, section 14). | No book row or study has a near-equal θ pair, so the θ fix is bitwise on the book; the `dt` behaviour is what every banked number was computed with. |
| 7 | A column group is exactly what one solve call advances today. | The grouping, the backend, and the operation order inside a step are the three things that decide bit-identity. |
| 8 | Components fingerprint by value through `quantark.execution.cache.fingerprint`, and that one function, `try_fingerprint(component)`, is the component's identity everywhere: artefact keys and the problem identity. `evidence()` is the human-readable certificate record and is never a key. A component without a reproducible fingerprint prices directly, gets no session artefacts, and cannot be certified. | Spec 1 section 10: no registry, no versioning system. Two components with different classes or settings can return the same `evidence()` (the built-in operator and readout both return `{}`), so evidence cannot key anything. |
| 9 | The backend stays fixed per family: sparse LU for the base family and American, banded for the two-surface family, banded per column or LU for Phoenix, banded only for statistics. | Spec 1 section 10; SuperLU and LAPACK banded solves are different algorithms. |
| 10 | `explain` is `prepare` without the march: one function, one report type. | The core-first review's finding that `resolved_grid()` cannot show the cap; anything derived separately would drift. |
| 11 | Release 0.7.0, a clean break. Removed names raise `ValidationError` naming the replacement; no study YAML names any of them. | 0.4.0 / 0.6.0 convention. |
| 13 | Two identities, not one: the **artefact keys** (grid, coefficients, factorizations) stay narrow and product-economics-free so a session can reuse them across products with the same geometry; the **problem identity** adds the complete economic state (section 5.9). | Two snowballs with identical grid requests and different KO coupons must never share a problem identity (measured: 97.74 versus 100.57 with the same `GridRequest`). |
| 14 | `prepare` is the only place the environment is read; the march and readout consume sampled arrays. The two environment memos key on the market objects' value fingerprints, computed once per `prepare`, never on object identity. | `FlatRateCurve`, `ContinuousDividendYield`, `FlatVolSurface`, `GridVolSurface` are plain mutable dataclasses; the immutability in `CLAUDE.md` is a convention. Today, mutating a rate after a solve returns the memoized coefficients and discount factors (measured). |
| 15 | One parameter-resolution path for execution and evidence: a certificate candidate builds its engine from `PDEParams.from_config(declared)` and records the same object. Certification accepts only what `from_config` can name, so only built-in components are certifiable in this release. | Today the builders record the declared mapping and construct from `accuracy` alone: a declared `stepper: {theta: 1.0}` is banked and not executed (measured). A value fingerprint covers a component's configuration, not its code. |
| 16 | `prepare` resolves terminal outcomes (expired, knocked out at valuation) before any grid **per operation**, exactly as each entry point short-circuits today (section 5.2 step 0): `price`, `calculate_greeks`, and `price_with_events` return their short-circuit values without a layout; `calculate_event_stats` never short-circuits and marches; a life surface raises. | `GridRequest` requires a positive maturity; the short-circuits are part of the contract (`test_price_with_events_single_pass.py`). Measured: on a product knocked out at valuation, `price_with_events` returns the single-pass trivial distribution while direct `calculate_event_stats` marches and reports KO probability one at the valuation node. One terminal rule for every operation would change the direct statistics. |
| 17 | The delivered `Layout` has its own fingerprint over the arrays the march consumes (`x`, `dx`, `t`, `dt`), the event-node and extra-node index sets, and the config key; the coefficient key and the problem identity use it, never the request. The factorization identity adds the effective per-group θ schedule. Measured: a three-step budget with no interior node, with a readout node at 2/3, and with an event node at 2/3 share `t` exactly, differ in `dt` by an ULP, and carry θ schedules `[0.5, 0.5, 1]` against `[1, 1, 1]`. The grid **artefact** key stays request-based because it answers a different question: which layout a fresh bind of this request produces. | A bump clone frozen at spot 100 and a fresh solver both priced at spot 101 build the same request key, different spatial grids, and different prices (measured: `98.25175716148267` versus `98.25326843586967`). |
| 18 | The coefficient key is the fingerprint of every input the sampler passed: the curves, the vol surface **or the local-vol surface actually used** (prebuilt or built), the spot, the reference strike, and the layout fingerprint. A bump instance is constructed with the same constructor arguments as its parent, so a prebuilt surface travels with it. | Two local-vol solvers with prebuilt 15% and 35% surfaces share the environment's implied-vol fingerprint and price at `100.03` and `94.51` (measured). `TermCoefficients.from_env` passes the spot into every `get_vol`. |
| 19 | Every memo's key is the fingerprint of every input the memoized function reads, enumerated per memo (section 5.9 table). The observation resolvers read the **effective settlement calendar**, `product.settlement_convention.calendar` when supplied and the environment's otherwise (`settlement.py`), so their keys and the economic identity carry that calendar's `identity()` (name, sorted holidays, weekend days; new, because `Calendar` is a plain class over sets, its `repr` is a name and a holiday count, and the product's `cache_key()` names a calendar by name alone). An input without a fingerprint disables that memo. | Changing the environment's calendar after a solve returned the cached settlement date, and so did adding a holiday to a product-supplied settlement calendar (measured: January 12 cached, January 13 fresh, with the product key unchanged). |
| 20 | The byte gate is a manifest, not a diff: every row in the manifest present on both sides, every numerical row a finite hex on both sides, a row may raise only when the manifest declares it with its exception type, and a row may be `None` only when the manifest declares it (the expired statistics shortcut). | `compare.py` exits 0 when both books carry the same `__error__` for a row (verified by reading it), so zero diffs alone proves nothing about preservation. |
| 21 | Knock-in barriers (single and double) are a **composite** of two prepared problems, the vanilla on its own automatic domain and the knock-out on the barrier's hard-edged domain, each prepared and marched independently with its own geometry and arithmetic, composed at readout as vanilla minus knock-out for the price and for each Greek, as today. Inside a bump context the knock-out layout is frozen and the vanilla rebinds fresh on every solve, as today. | `BarrierPDESolver.price` and `calculate_greeks` (and the double-barrier solver) price a knock-in as `_price_vanilla` through a fresh `EuropeanPDESolver` minus the knock-out; measured: two marches, upper bounds `222.554…` (automatic) and `120` (the barrier). One column on one layout would be a different pricer. |
| 12 | The solver class names stay as facades (`SnowballPDESolver`, `PDEEngine`, …); a solver holds its params, its adapter, and its caches, and `price` is prepare, march, readout. | The execution capability matrix, the replay engine factory, and the certificates name these classes. |

## 4. Public surface after sub-project 2

### 4.1 `PDEParams`

```python
@dataclass(kw_only=True)
class PDEParams(EngineParams):
    accuracy: str = "standard"                     # fast | standard | high
    grid: Optional[GridConfig] = None              # expert overlay, nine fields (4.3)
    stepper: TimeStepper = CrankNicolson()         # built-in; carries the four scheme settings
    operator: OperatorFactory = CentralDifference()
    readout: Readout = StandardReadout()
    event_projection: Optional[EventProjectionMode] = None
    cache_enabled: bool = True
    # inherited from EngineParams: bump_config, bump_size (deprecated), bus_days_in_year
```

| Field | When to change it | What happens |
|---|---|---|
| `accuracy`, `grid`, `event_projection`, `cache_enabled`, `bump_config` | as in spec 1 section 4.1 | unchanged |
| `stepper` | You want a different time weighting or damping schedule (`CrankNicolson(theta=1.0)`, `CrankNicolson(terminal_damping_steps=0)`), or you are writing a stepper. An unset damping count is the route's default (section 4.2). | The built-in advances each column group with the θ schedule of section 4.2 over the family's backend. A custom stepper receives the assembled coefficients and the group and returns the advanced columns; the core validates shape and finiteness. |
| `operator` | You are writing an operator (an exponentially fitted stencil, say). | The built-in assembles the central-difference tridiagonal with first-order upwind rows on zero-diffusion steps, exactly today's `_build_step_coefficients` plus `_flat_exact_step_coefficients`. |
| `readout` | You are writing a readout. | The built-in interpolates the price linearly between the two nodes straddling the spot and takes delta and gamma from the local cubic, today's `_interpolate_price` and `_calculate_delta_gamma`. |

Keyword-only construction and the removal map wrapper are unchanged from spec 1
section 4.4. `theta` and `event_theta` are removed names (section 4.4). Validation of the
four settings moves into `CrankNicolson.__post_init__` with the same ranges (`theta` in
[0, 1], `event_theta` in (0, 1], counts non-negative integers).

#### Route applicability

Spec 1's table (section 4.1) is re-keyed: every `grid.terminal_damping_steps` /
`grid.event_damping_steps` row becomes `stepper.terminal_damping_steps` /
`stepper.event_damping_steps` with the same status, and `theta` / `event_theta` rows read
`stepper.theta` / `stepper.event_theta`. `ROUTE_RULES._explicit` learns the `stepper.`
prefix: explicit means a value away from the `CrankNicolson` default. Two new rows for
the routes outside the core:

| Field | Shared-grid solvers | LV vanilla | LV barrier | DCN |
|---|---|---|---|---|
| `stepper` not a `CrankNicolson` | honoured | rejected: no core seam until sub-project 3 | rejected | rejected |
| `operator`, `readout` non-default | honoured | rejected | rejected | rejected |

### 4.2 `CrankNicolson` and the schedule

```python
@dataclass(frozen=True, kw_only=True)
class CrankNicolson:
    theta: float = 0.5                          # weight away from discontinuities
    event_theta: float = 1.0                    # weight on the event-damped steps
    terminal_damping_steps: Optional[int] = None  # None = the route's default (1; DCN 2)
    event_damping_steps: Optional[int] = None     # None = the route's default (2; DCN 2)

    def schedule(self, layout: Layout) -> np.ndarray: ...
    def step(self, group, coefficients, dt, theta, boundaries) -> None: ...
    def evidence(self) -> dict: ...
```

The two counts follow spec 1 decision 15's pattern: `None` is the route's default, 1 and
2 on every core route and 2 and 2 on DCN, so `CrankNicolson()` means today's plan on
every route and an explicit count is always distinguishable from the default. The core
resolves the counts once in `prepare` (the route's `damping_defaults`); the evidence and
the plan carry the resolved values; `CrankNicolson.evidence()` omits an unset count, as
`GridConfig.evidence()` does.

`schedule` returns the per-step θ that `_theta_schedule_from_layout` returns today:
`theta` everywhere; `event_theta` on the `event_damping_steps` steps before each
interior **event** node (readout nodes from `extra_times` get none); `1.0` on the
`terminal_damping_steps` steps off the payoff; terminal wins on overlap; indices below
zero dropped; a count larger than an interval's fill spills into the earlier interval.
Spec 1 section 4.2 precedence row 4 moves here verbatim. The two damping index sets
leave `TimeLayout` (`build_time` no longer reads damping counts; `TimeLayout` loses the
two frozensets); the stepper computes them from `layout.time.step_of` and the request's
event times. DCN, the 2D solvers, and the intraday route read the counts from
`params.stepper` (section 7).

Resolved defaults equal today's, so the schedule is bit-identical and the built-in
stepper at its defaults is what every banked certificate ran.

### 4.3 `GridConfig` (nine fields)

`points`, `steps_per_day`, `time_steps`, `eps_crit`, `num_std`, `bounds`, `max_points`,
`max_steps`, `day_count`. `GridConfig.key` has nine entries; `evidence()`,
`with_steps_per_day`, `with_time_steps`, `resolve_config`, the route tables, and the
precedence tables of spec 1 section 4.2 are otherwise unchanged. `terminal_damping_steps`
and `event_damping_steps` are removed names (section 4.4); the shared defaults table
loses them; `DCN_ACCURACY_PROFILES` loses its two damping entries; DCN's defaults of 2 and 2
are its route's `damping_defaults` (section 4.2, section 7).

### 4.4 Removed names (additions to the map)

| Removed name | Replacement text |
|---|---|
| `PDEParams(theta=...)` | `stepper=CrankNicolson(theta=...)`; 0.5 is Crank-Nicolson, 1.0 backward Euler |
| `PDEParams(event_theta=...)` | `stepper=CrankNicolson(event_theta=...)` |
| `GridConfig(terminal_damping_steps=...)` | `PDEParams(stepper=CrankNicolson(terminal_damping_steps=N))`; 0 disables |
| `GridConfig(event_damping_steps=...)` | `PDEParams(stepper=CrankNicolson(event_damping_steps=N))`; 0 disables |

The 0.6.0 entries that name `grid=GridConfig(terminal_damping_steps=N)` as a replacement
(`use_rannacher`, `rannacher_steps`, `rannacher_at_events`, `event_rannacher_steps`,
`DCNPDEEngine(rannacher_steps=)`) are re-pointed at the stepper. `GridConfig` gains the
same keyword-check wrapper `PDEParams` has, so its two removed names raise
`ValidationError` with the replacement rather than the dataclass's `TypeError`.

### 4.5 `from_config`

`stepper` is a nested mapping of `CrankNicolson` fields: `stepper: {theta: 1.0,
terminal_damping_steps: 0}`. `operator` and `readout` keys are rejected in a config
file (a custom component cannot be named from YAML; the built-ins have no settings in
this release). `theta` / `event_theta` at the top level and damping counts under `grid`
go through the removal map.

### 4.6 Evidence record (identity compatibility)

Certificates hash `{declared params, engine, grid: <resolved record>}` where the record
is built by the candidate builders (`equity_snowball.py`, `equity_phoenix.py`,
`equity_ko_reset.py`, `equity_snowball_localvol.py`, `intraday_snowball.py`), today as
`engine_config(resolve_config(accuracy, None))`. They now build the engine they execute from the mapping they record (decision 15):

```python
params = PDEParams.from_config(declared)            # the YAML's candidate params, unchanged
engine = SnowballPDESolver(params=params)           # what runs
record = {**declared, "engine": ..., "grid": resolved_grid_evidence(params)}
```

`resolved_grid_evidence(params)` in `grid/config.py` returns
`resolve_config(params.accuracy, params.grid).evidence()` plus the resolved
`terminal_damping_steps` and `event_damping_steps` of `params.stepper`: the historical
ten-key `grid` record, byte-identical for every banked configuration (every banked PDE
candidate declares `{accuracy: standard}` and nothing else, so `from_config` reproduces
today's engine and the declared dict is untouched). A declared key that `from_config`
rejects fails the study at load, the way the MC reference arm already refuses knobs it
cannot honour. Today the builders construct from `accuracy` alone while recording the
whole declared mapping, so a declared `stepper: {theta: 1.0}` would be banked and run at
0.5 (measured, section 14). `canonical_json`
sorts keys, so field order is irrelevant. `test_banked_cells_keep_their_identity` and the
anchors are the proof; a new test asserts the record equals the banked `grid` dict of the
three guarded studies. The docs say what the record is: the numerical grid plus the
built-in scheme's damping counts under their historical keys, not the serialization of a
`GridConfig`.

`PDEParams.evidence()` (new, consumed by `engine_config(params)`) emits `accuracy`,
`grid` (the overlay's `evidence()` as today), `stepper` (the built-in's four settings as
values; a custom component as its fingerprint, or `"uncertifiable"`), `operator` and
`readout` only when non-default (fingerprint or `"uncertifiable"`), `event_projection`,
`cache_enabled`, and the inherited fields as today. No banked certificate serializes a
`PDEParams` object, so this shape is new evidence, not a moved identity. Certification
covers built-in components only: a study names its candidate through `from_config`, which
cannot name a custom component, and a candidate constructed any other way is refused at
load. The reason is stated in the docs: a value fingerprint covers a component's
configuration and class name, not its code (reassigning a method leaves it unchanged,
measured), and the built-ins are covered by the release's own source.

### 4.7 `explain`

```python
plan = engine.explain(product, pricing_env, *, lifecycle_state=None)   # PDEEngine and every core solver
plan.to_dict(); str(plan)
```

`explain` runs `prepare` (section 5.2) and skips the march. `PDEPlan` carries:

| Part | Contents |
|---|---|
| route | engine class, adapter, model (BSM / local vol), lifecycle facts the adapter resolved (knocked in, KI regime, reverse, schedule in force); for a knock-in barrier, both subproblems and the composition |
| settings | per `PDEParams` field and per `GridConfig` field: requested value, effective value, source (`preset`, `overlay`, `route`), applicability (`honoured`, `vacuous`, `inapplicable`) |
| space | points, bounds with the rule that set each side (hard edge, `bounds`, envelope), `achieved_eps`, active and ignored critical prices |
| time | basis in force and its value, `day_count` when the density is in force, requested and actual steps, `fill_scaled`, `cap_exceeded`, the mandatory nodes |
| events | per barrier: monitoring as declared, its nodes, projection treatment, order within a step; coupon and settlement schedule summary |
| scheme | stepper evidence, the number of terminal- and event-damped steps, the distinct θ values and how many steps carry each, zero-diffusion steps |
| outputs | the column groups the request creates, with column counts and backend; retention (rolling / full / branch capture); readout mode |
| identity | the delivered layout's fingerprint, the artefact keys, the problem identity (section 5.9), and whether every component is fingerprintable |

Large node lists live in `to_dict()`, not the text view. A preparation failure raises the
same error `price` would; `explain` never returns a plausible plan for a request that
cannot price. The intraday route's `_layout_numbers` and `_time_status` read the plan.
`resolved_grid()` stays as the cheap preset expansion with its 0.6.0 docstring. The
routes outside the core (DCN, the local-vol vanilla and barrier solvers) have no
`explain` until sub-project 3 moves them onto it; their classes do not define the method.

## 5. The core

Package: `quantark/asset/equity/engine/pde/core/` with `prepare.py`, `march.py`,
`readout.py`, `groups.py`, `backends.py`, `components.py` (protocols and built-ins),
`plan.py`, and `adapters/` (one module per family).

### 5.1 Stages and types

```
engine.price / explain / session / bump / intraday / life surface
    -> prepare(product, env, params, request, frozen=None, session=None) -> PreparedProblem
    -> march(prepared) -> MarchResult
    -> readout(prepared, result) -> price / greeks / event stats / surfaces
```

- `OutputRequest` (frozen): `price`, `greeks`, `event_stats`, `streams`, `grid`,
  `life_surface`, `external_pv`. Replaces `_session_outputs(want_events, want_grid,
  streams)` and the ad hoc flags on the solvers.
- `CompositeProblem` (frozen): an ordered tuple of `PreparedProblem`s and a composition
  readout; what `prepare` returns for a knock-in barrier (decision 21).
- `PreparedProblem` (frozen): adapter, solve state, market snapshot, `Layout`,
  observation maps, ordered events per step, coefficients (`StepCoefficients`),
  θ schedule, column groups, captures to take, retention, fingerprint.
- `ColumnGroup` (frozen): name, surface (alive / knocked-in / memory state k /
  termination), column count, backend, `GroupPolicy`, terminal fill.
- `GroupPolicy` (frozen): boundary treatment, θ source, event set, after-event fixup.
- `MarchResult`: the retained columns per group (rolling or full), the captures.

### 5.2 `prepare`

In this order, for every entry point:

0. **Terminal outcomes, per operation** (decision 16). The adapter is asked whether the
   request resolves without a march for **this operation**, exactly as each entry point
   short-circuits today:

   | Operation | Expired | Knocked out at valuation |
   |---|---|---|
   | `price` | the terminal value at valuation | the immediate KO payoff |
   | `calculate_greeks` | today's short-circuit values | KO payoff, delta 0, gamma 0 |
   | `price_with_events` | the single-pass terminal distribution | the single-pass trivial distribution |
   | `calculate_event_stats` | returns `None` before any grid, as `_compute_event_stats` does today | no short-circuit: marches; KO probability one at the valuation node |
   | `solve_life_surface` | raises `ValidationError` | raises `ValidationError` |

   When an operation is terminal, `PreparedProblem.terminal` carries the outcome, there
   is no layout, `march` is skipped, and `readout` returns the outcome. `explain` reports,
   per operation, whether the request is terminal and why. The two different answers the
   statistics give for a product knocked out at valuation are a characterized oddity
   (section 8), preserved here.
1. Route applicability (`check_route_applicability`), before any grid.
2. The adapter's **solve state** (today's `_prepare_solve_state`: KI regime, knocked-in
   at valuation, reverse, KI barrier, terminal delay factor, exercise dates). This is the
   only preamble; the bump path's `_prepare_for_request` and the statistics sweep's inline
   copy go.
3. Market snapshot (`market_snapshot`, `representative_vol` unchanged).
4. Grid request from the adapter (`grid_request`; the life-surface request adds
   `extra_times`).
5. Layout: `resolve_bound_layout(binder, frozen, request, market)` as today, or the
   session's stored `Layout` object (the session stores the `Layout`, not five arrays;
   the synthetic `SpatialLayout` with `achieved_eps = nan` goes).
   `validate_external_layout` also compares `extra_times`, so a frozen price layout is
   not reused for a life-surface request.
6. Observation maps from `layout.step_at` once (the snowball's nearest-match rebuild
   goes; KO-reset's pre/post maps come from the same pass).
7. The adapter's ordered **events**: a table `step -> [transform, ...]` over named
   state blocks, built from today's `EventSchedule` (snowball interior events) and the
   inline paths (valuation date, Phoenix, KO-reset, the barrier family's masks), in the
   order each takes today. The first-passage continuous-KI correction is prepared here.
8. Market samples, then coefficients. The route's model owns the **sampling**
   (`MarketSamples`): the Black-Scholes routes sample forward rates, carries, and a scalar
   vol per step from `TermCoefficients`; the local-vol routes sample a vol vector per step
   at the interval midpoint from the Dupire surface, as `_local_vol_coefficients` does
   today. The operator factory owns the **stencil**: the session's injected
   `StepCoefficients` when present, else `params.operator.assemble(samples, layout)`, on
   **every** family (Phoenix and KO-reset stop bypassing injection; the values are
   identical, section 8). The first-passage continuous-KI correction takes its
   barrier-local vol from the same `MarketSamples`, so the correction and the operator see
   one set of dynamics.
9. θ schedule: `params.stepper.schedule(layout)`.
10. Column groups from the adapter for this request (section 5.4); the statistics group
    only when `request.event_stats`.
11. Captures and retention from the request (valuation-date pre-event columns; branch
    columns for a life surface; full surfaces when Greeks or a life surface are wanted).
12. The problem fingerprint (section 5.9).

`explain` returns the `PDEPlan` built from steps 1 to 12.

### 5.3 `march`

One loop over the layout's steps, last to first. Per step `j`:

1. For each column group in declared order: the group's boundary policy supplies the
   two boundary values for step `j` (adapter Dirichlet, computed before the solve as
   today, or the carried-forward previous row); `stepper.step(group, coefficients_j, dt_j,
   theta_j(group), boundaries)` advances the group's columns in one solve against the
   backend's factorization for `(set_j, dt_j, theta)`; then the group's post-step hook
   (the American obstacle projection where eligible).
2. Valuation-date pre-event captures when `j == 0` and the adapter asks.
3. The events of step `j`, in the adapter's order, over the named state blocks; then
   each group's after-event fixup (the statistics group's zero-slope copy).
4. Branch captures for a life surface.

Retention follows the request: full `(num_x, num_t)` surfaces where a Greek readout or a
life surface needs them, rolling columns otherwise. The loop has no product branches; the
adapter's tables are the only product knowledge.

### 5.4 Column groups (decision 7)

| Family | Value groups | Backend and solve |
|---|---|---|
| Base family (European, knock-out barriers, touches), American | one column | sparse LU, `M2_lu.solve(rhs)` |
| Knock-in barriers, single and double | two prepared problems (decision 21): the vanilla, one column on its automatic domain; the knock-out, one column on the hard-edged domain; already knocked in: the vanilla alone | sparse LU each, marched independently, composed at readout |
| Snowball, KO-reset, LV snowball | alive and knocked-in as **one two-column group** | banded, `(n_int, 2)` right-hand side in one `solve_banded` call; when the interior has two nodes or fewer, LU with alive then knocked-in as two one-column groups, sequential |
| Phoenix, LV phoenix | one group per (memory state k, surface), plus the two termination surfaces under EXPIRY coupons, each **one column** | banded per column (1D right-hand side), or LU per column |
| Event statistics | one group of `2·n_cols` indicator columns (both surfaces) | banded only, `(n_int, 2·n_cols)` right-hand side in one call; raises otherwise, as today |

The right-hand-side construction stays per backend: the sparse `M1 @ v` for LU, the
manual `main1 · v` with the lower and upper products for banded. Boundary injection keeps
its formula and its timing (`dt · ((1 − θ) · l · v_next[0] + θ · l · v_cur[0])`).

### 5.5 Group policies

| Policy | Boundary | θ | Events | After events |
|---|---|---|---|---|
| value (every value group) | adapter Dirichlet, set before the solve; the alive column's lower boundary under continuous KI is **zero** (today it reads the knocked-in column before it is set) and the coupling overwrites it | the schedule; θ=1 on zero-diffusion steps for the base family, American, Snowball, KO-reset; **no override for Phoenix** (section 8) | the adapter's ordered table | none |
| statistics-legacy | carry the previous row into the right-hand side; after events set `v[0] = v[1]`, `v[-1] = v[-2]` | the schedule, no zero-diffusion override | KO absorption (KI-ever column exempt), Phoenix coupon columns, KI copy with the first-passage correction, in that fixed order, no `ko_last`; projection per `_event_uses_projection` | zero-slope copy |

Level 2 (sub-project 3) is the statistics group taking the value policy plus a maturity-leg
column; nothing in this release changes a policy.

### 5.6 Adapters

`EuropeanAdapter`, `AmericanAdapter`, `BarrierAdapter` (single, double, one-touch, double
one-touch, by product type), `SnowballAdapter`, `PhoenixAdapter`, `KOResetAdapter`. For a
knock-in product `BarrierAdapter.prepare` returns a `CompositeProblem` of two prepared
problems, the vanilla (through `EuropeanAdapter`, the participation rate applied at
composition) and the knock-out, with the composition readout `vanilla − knock-out` for the
price and for each Greek (decision 21); the vanilla subproblem takes no frozen layout,
because today it is a fresh solver on every call. Each adapter supplies: solve state; grid request; terminal fill per column; the exercise obstacle
(American); boundary values per column and step; the ordered events per step; captures;
the column-group declaration per request; the readout branch (today's
`_compose_t0_readout`: pre-event valuation-date column, knocked-in surface when knocked
in, overrides). The local-vol Snowball and Phoenix are the same adapters with a
local-vol `MarketSamples` (one vol vector per step at the interval midpoint); the stencil
comes from the same `CentralDifference` (section 5.8). There is no separate local-vol
operator.
The barrier-local first-passage coefficients of `vol_continuous_ki.py` stay a mixin on the
adapter. KO-reset keeps its QUAD delegation for statistics (section 12); its value groups
are the snowball's with the pre/post schedule events.

### 5.7 Backends and the factorization cache

`SparseLU` (`splu` of `I − θ·dt·A`, `M1 = I + (1 − θ)·dt·A`) and `Banded`
(`main = 1 − θ·dt·c`, the explicit triple, `solve_banded((1, 1), …, check_finite=False)`),
private, chosen by the group's declaration. One `FactorizationCache` per prepared problem,
keyed on `(coefficient set index, round(dt, 12), theta)`: `dt` rounded as today, θ exact
(decision 6). The `dt` rounding is deliberate and documented where the key is built: the
time builder constructs one `dt` per interval, intervals of nearly equal length differ by
ULPs, and the rounded key lets them share a factorization; every banked number was
computed that way. The cache is bounded by `BANDED_CACHE_MAX_ENTRIES` /
`GRID_CACHE_MAX_ENTRIES` as today, cleared at the start of each march, bypassed when
`cache_enabled` is false. A prepared session hands the cache
object across (section 5.10); the pack enumeration of `_session_factorization_packs`
becomes the cache warmed in march order, capped at 64 entries as today.

### 5.8 Protocols and built-ins

```python
class OperatorFactory(Protocol):
    def assemble(self, samples: MarketSamples, layout: Layout) -> StepCoefficients: ...
    def evidence(self) -> dict: ...          # built-in: {}

class TimeStepper(Protocol):
    def schedule(self, layout: Layout) -> np.ndarray: ...                  # per-step theta
    def step(self, group: GroupState, coefficients: StepSet, dt: float,
             theta: float, boundaries: tuple[np.ndarray, np.ndarray]) -> None: ...
    def evidence(self) -> dict: ...

class Readout(Protocol):
    def price(self, column: np.ndarray, spot: float, layout: Layout) -> float: ...
    def delta_gamma(self, column: np.ndarray, spot: float, layout: Layout) -> tuple[float, float]: ...
    def evidence(self) -> dict: ...          # built-in: {}
```

`MarketSamples` is what `prepare` sampled from the environment (section 5.2 step 8),
in one of two kinds: `scalar` (forward rate, carry, vol per step) or `nodal` (forward rate
and carry per step, a vol vector per step). `CentralDifference.assemble` accepts both,
each on today's arithmetic: the scalar path rounds the triples, deduplicates them, builds
the stencil vectorized, writes first-order upwind rows on zero-diffusion sets, and applies
the flat-exact replacement when one set remains; the nodal path builds one set per step
with no deduplication and no zero-diffusion sets. A custom operator declares
`supports: frozenset[SampleKind]`; a route whose samples it does not support is rejected
by the applicability check, before preparation.

The core validates every component result and raises `NumericalError` naming the
component; nothing falls back:

- schedule: length `actual_steps`, finite, every weight in [0, 1];
- `StepCoefficients`: the arrays with the shapes `_build_step_coefficients` produces
  today, finite; `set_index` an integer array of length `actual_steps` with every value in
  `[0, n_unique)`; `zero_diffusion_sets` a subset of `range(n_unique)`;
- advanced columns: the group's shape, finite, written only into the group's buffer;
- readout values: finite.

Ownership: the core hands components read-only views and copies any array it retains
from them; a component that needs scratch space allocates its own. A component's fingerprint is `try_fingerprint(component)`; a frozen dataclass
of plain values fingerprints itself, an arbitrary object returns `None`. A `None`
fingerprint makes the prepared problem unfingerprintable: `price` works, no session
artefact is built or reused, `explain` says so. The fingerprint covers configuration, not
code; that is enough for session artefacts, which live in one process's memory
(`PreparedArtifactCache` is an in-memory store, so a component's code cannot change under
a live artefact short of monkeypatching), and not enough for certification, which is why
only built-in components are certifiable (decision 15).

### 5.9 Caches and identities

| Cache | Fate |
|---|---|
| `GridBinder` layout cache | stays; key unchanged (`GridConfig.key` has nine entries) |
| matrix / LU cache, banded cache, session packs | become the `FactorizationCache` (section 5.7) |
| KO-records, KI-profile, KO-reset pre/post caches | stay on the adapter; keys gain the environment's schedule identity (decision 19, memo table below) |
| `_ENV_STEP_COEFF_MEMO`, `_ENV_DF_MEMO` | keyed by the fingerprint of every input they read, computed once per `prepare` (memo table below); an unfingerprintable input gets no memo and is sampled fresh; the intraday route's explicit memo pop goes (decisions 14, 19) |
| `_product_token_memo` | cleared at every `prepare` (today it is not cleared on `calculate_greeks`) |
| `_critical_points_cache`, `_get_cached_critical_points`, `_params_cache_key`, `_bump_transient_attrs`, `_bump_clone` | deleted (no production caller, or replaced by section 5.10) |
| `PDEEngine._solver_cache` | stays |

**Two identities** (decisions 13, 17, 18). The **artefact keys** are narrow and carry no
product economics, so a session reuses them across products with the same geometry:

| Key | Contents |
|---|---|
| grid artefact key | `(class, GridRequest, MarketSnapshot, config.key)`, unchanged: it names the layout a **fresh bind** of this request produces, and is used only to look one up |
| `Layout.fingerprint` | sha256 over the arrays the march consumes, `spatial.x`, `spatial.dx`, `time.t`, `time.dt` (the constructed per-interval constants, not `diff(t)`), the event-node and extra-node index sets, and `config_key`: the layout actually delivered, frozen or rebound or fresh (decision 17) |
| coefficient key | `Layout.fingerprint` plus the fingerprint of every input the sampler passed: rate curve, dividend yield, the vol surface or the local-vol surface actually used (prebuilt or built), spot, reference strike, and `try_fingerprint(operator)` (decision 18) |
| factorization key | the coefficient key, `try_fingerprint(stepper)`, the backend, and the effective per-group θ schedules as arrays (a node reclassified from readout to event changes the schedule and nothing else, decision 17) |

`_BUILDER_VERSION` becomes `pde-prep/v2`. The **problem identity** is the fingerprint of
the factorization key plus `try_fingerprint(readout)` (the readout changes prices and
Greeks) plus the complete economic state: the product's cache token (today's "standard"
keying: the product's frozen fields, or its `cache_key()`), the resolved solve state (KI
regime, knocked in, reverse, schedule in force, exercise dates), the event table's
evidence (per step: kind, barrier, cash amount, settlement time, projection), the
environment's schedule identity (valuation date, day-count convention, `bus_days_in_year`)
and the **effective settlement calendar's** identity (the product's when supplied, else the
environment's; decision 19), `event_projection`, and the output request. Two requests with the
same artefact keys and different coupons, settlement lags, lifecycle states, or readouts
have different problem identities; a frozen bump solve and a fresh solve at the same spot
have different layout fingerprints and therefore different coefficient keys and
identities. `explain` reports the layout fingerprint, the artefact keys, and the problem
identity.

**Memo keys** (decision 19): each memo's key is the fingerprint of every input the
memoized function reads. An input without a fingerprint disables that memo for that call.

| Memo | Inputs it reads | Key |
|---|---|---|
| step coefficients (`_ENV_STEP_COEFF_MEMO`) | rate curve, dividend yield, vol surface or local-vol surface, spot, reference strike, the time and space arrays | the coefficient key above, computed once per `prepare` (the spot is in it: `TermCoefficients.from_env` passes it to every `get_vol`; no shipped surface reads it in a value-changing way, so on the book this is bitwise) |
| discount factors (`_ENV_DF_MEMO`) | rate curve, or dividend yield, and the two times | curve fingerprint per `prepare`; `(t, T)` inner key as today |
| KO records, KI profile, KO-reset pre/post | product, valuation date, the effective settlement calendar (product's or environment's), day-count convention, `bus_days_in_year` | product token plus the environment's schedule identity plus the effective settlement calendar's `Calendar.identity()` (name, sorted holidays, weekend days; new, because `Calendar` is a plain class over sets and the product token names a calendar by name alone) |
| layout cache (`GridBinder`) | request, market snapshot, config | unchanged |

### 5.10 Entry points

| Entry | Before | After |
|---|---|---|
| direct `price`, `calculate_greeks`, `calculate_event_stats`, `price_with_events` | each solver's `_solve` plus a second march for statistics; `calculate_event_stats` without a price re-prices | one `prepare` with the request that names the outputs; one march; readout. `calculate_event_stats` without a price marches the value and statistics groups together |
| prepared session (`pde_session_prep.py`, `pde_execution_adapters.py`) | fresh clone; grid, coefficient, and pack artefacts; synthetic spatial layout on execute | `prepare(session=None)` builds the artefacts (the `Layout`, the `StepCoefficients`, the warmed `FactorizationCache`); `prepare(session=artefacts)` executes; the three artefact kinds keep their names |
| bump context | deep copy with a transient-attribute list; `_prepare_for_request` | a fresh instance built with the parent's constructor arguments (params and, on the local-vol solvers, the prebuilt surface) holding the frozen `Layout`; bumped solves call `prepare(frozen=layout)`; `resolve_bound_layout` unchanged; the frozen layout's fingerprint keys everything downstream (decision 17) |
| intraday `PDERoute` | prices, then reads `_active_layout` and `grid_binder.config`, calls `build_time` to probe | prices; reads `explain` for resolution numbers; the refined request is built as today |
| life surface (`solve_life_surface`, `backtest/simulation/pricing/surface.py`) | `_extra_time_nodes`, `_capture_branch_columns` flags on the solver | `OutputRequest(life_surface=True)`; `LifeSurfaceSolution` fields unchanged; the pricer's node Greeks unchanged |
| 2D Heston/SLV statistics | inherit the 1D `calculate_event_stats(npv=2D pv)` | `OutputRequest(event_stats=True, price=False, external_pv=pv)`: a statistics-only march on the 1D core with the 2D price as residual |

## 6. Event statistics on the core (Level 1)

The statistics group reproduces `_compute_event_stats` column for column. Per surface
(alive, knocked-in): one column per KO date, one per Phoenix coupon date, a KI column
and a KI-ever column; `streams` prunes the KI pair as today. Terminal fill: zeros except
the knocked-in surface's two KI columns at 1. At a KO step: every column zeroed in the KO
region and the KO column set to the discount factor from observation to settlement
(KI-ever exempt; both surfaces, or alive only when KO does not survive KI). Phoenix
coupon columns set to their pay-date discount factor on the pay mask after the KO jump.
KI step: alive takes knocked-in below the barrier on every column, plus the first-passage
correction. Readout: per column at the spot on the pre-event valuation-date capture;
probabilities by dividing by the discount factor to settlement (KO, coupon) or to
maturity (KI); `expected_discounted_maturity_cashflow = pv − Σ KO − Σ coupon`;
`reconciliation_error = 0.0`; already-knocked-in short-circuits as today. Result types
(`AutocallableEventStats`, `PhoenixEventStats`) unchanged; consumers (cash legs, replay,
dashboard, risk report) see the same values.

The thirteen conventions, recorded so nobody "fixes" one inside this release. Two are
structural and are replaced now with identical values (the direct coefficient build, the
nearest-match indices; section 8); the other eleven are Level 2's: carry-forward boundaries with the zero-slope copy; KI-ever exempt from
absorption; KO absorption to a discount factor; the terminal fill without the terminal
delay factor; the fixed KO, coupon, KI order without `ko_last`; projection gated by
`_event_uses_projection` while the snowball value sweep's interior events always
cell-average; no zero-diffusion θ override; a direct coefficient build (now the shared
assembly, section 8); banded only; nearest-match KO indices (now `step_at`, section 8);
Phoenix coupon columns without the memory dimension and EXPIRY coupons discounted to the
terminal date; the residual maturity leg; KI hard-coded to one when already knocked in.

## 7. Routes outside the core

- **DCN** (`DCNPDEEngine`, `LocalVolDCNPDEEngine`): `rann = int(config.terminal_damping_steps)`
  and the re-arm after events read `params.stepper.terminal_damping_steps` /
  `event_damping_steps` resolved against DCN's `damping_defaults` of 2 and 2 (an
  unset count is DCN's default; an explicit count is honoured in DCN's half-step form,
  as in 0.6.0). A stepper that is not a `CrankNicolson`, or a
  non-default `operator` / `readout`, is rejected (section 4.1). Loop unchanged.
- **Local-vol vanilla and barrier**: `terminal_damping_steps` 0 or 1 on the vanilla, the
  inapplicable rows on the barrier, `theta` honoured, all re-keyed to `stepper.*`; the
  three component rejections. Kernels unchanged.
- **2D Heston/SLV** (`snowball_vol_pde_solvers.py`, `phoenix_vol_pde_solvers.py`,
  `dcn_vol_pde_solvers.py`): `event_damped_step_keys` takes the stepper's counts and
  `event_theta`; `rannacher=… terminal_damping_steps > 0` reads the stepper; their 1D
  statistics use the statistics-only march (section 5.10). ADI cores unchanged.
- **Intraday**: `rannacher = resolved().terminal_damping_steps > 0` reads the stepper.
- **`GridLayerMixin`**: its docstring's claim that the LV/Heston/SLV engines adopt it is
  corrected (only DCN does).

## 8. Preserved on purpose, and structural changes proven by rows

**Characterized oddities the core reproduces bit for bit**, each with where it is
decided later:

| Oddity | Where it lives in the core | Decided in |
|---|---|---|
| Snowball interior discrete events cell-average through the schedule even under `event_projection=NODAL`; terminal, valuation-date, and statistics events go nodal | `SnowballAdapter` events; the NODAL byte row | sub-project 3 |
| The thirteen statistics conventions (section 6) | statistics-legacy policy | sub-project 3, Level 2 |
| Phoenix value sweep has no zero-diffusion θ override; base, American, Snowball, KO-reset do | value policy flag per family | sub-project 3 (number change on trading-clock environments) |
| Alive column's lower boundary under continuous KI is zero before the coupling | `SnowballAdapter` boundary | sub-project 3 |
| Heston/SLV statistics march in 1D on implied vol with the 2D price as residual | statistics-only march | sub-project 3 candidate (native 2D) |
| KO-reset statistics from QUAD with the price difference in the maturity leg | `KOResetAdapter.event_stats` | sub-project 3 item |
| The life-surface pricer's node Greeks use its own three-point stencil | outside the readout protocol | unchanged |
| On a product knocked out at valuation, `price_with_events` returns the single-pass trivial distribution while direct `calculate_event_stats` marches and reports KO probability one at the valuation node | terminal table, section 5.2 step 0 | sub-project 3 |
| A knock-in barrier's bump context freezes the knock-out layout but rebinds the vanilla subproblem on every bumped solve | `CompositeProblem`, decision 21 | sub-project 3 |

**Structural changes that are bitwise because the values are identical**, proven by
book rows rather than by reasoning: observation maps from `layout.step_at` alone (the
rebuild's nearest match lands on the same exact node); Phoenix and KO-reset honouring
session-injected coefficients (the same builder on the same layout; the new session rows
prove it); the session's stored `Layout` replacing the synthetic spatial layout (its
diagnostic fields never enter the arithmetic); the product-token memo cleared at every
`prepare` (tokens key caches only); `validate_external_layout` comparing `extra_times`
(raises or rebinds where it would have reused a layout lacking the readout nodes).

**Defect fixed in its own commit, bitwise on the book:** the exact θ in the factorization
key (decision 6). The `dt` rounding stays (section 5.7).

## 9. Verification

### 9.1 Byte book

`test/pde_bitwise/capture_book.py` runs on the merged sub-project 1 commit (base, detached
worktree, `PYTHONPATH=$PWD <main venv>/bin/python`) and on the branch, same machine, one
window. The gate is a manifest, not a diff (decision 20): `test/pde_bitwise/manifest.json`
lists every row id and, for each output, whether it is numerical, a declared `None` (the
expired statistics shortcut), or a declared expected failure with its exception type; `compare.py --manifest` fails when a manifest row is
missing from either book, when a numerical output is not a finite hex on both sides, when
an undeclared row raised on either side, or when a declared failure raised a different
type; and it fails on any diff with no `--allow`. Today it exits 0 when both books carry
the same `__error__` for a row. Rows whose construction differs between base and branch
detect the API per row (the budget row takes its step count from `explain` on the branch
and from the private layout on base). Rows added:

- `calculate_event_stats` and `price_with_events` on Snowball (continuous KI; monthly and
  daily discrete KI; knocked-in), Phoenix (memory and non-memory), KO-reset, on flat and
  term environments, at the three accuracies; `calculate_event_stats` without a price.
- Statistics with `streams` pruning; statistics under `event_projection=NODAL`.
- Phoenix and KO-reset through a prepared session (PV and statistics).
- A Snowball life surface (`solve_life_surface`) and a bump-context Greek on Phoenix.
- Heston and SLV Snowball and Phoenix statistics (the statistics-only march).
- Scheme rows through the new home: `stepper=CrankNicolson(theta=1.0)`,
  `CrankNicolson(terminal_damping_steps=0, event_damping_steps=1)`, on the European and the
  Snowball, matching the 0.6.0 rows bit for bit.
- Terminal outcomes: an expired Snowball, Phoenix, and Barrier; a Snowball knocked out at
  valuation; price, Greeks, `price_with_events`, and direct `calculate_event_stats` for
  each (the direct statistics on the knocked-out product march and report the KO at the
  valuation node); the life surface on the knocked-out product as a declared failure.
- A frozen bump solve and a fresh solve at the bumped spot, both recorded (they differ,
  and both must stay what they are). A local-vol Snowball with a prebuilt surface.
- Knock-in barriers, before the existing orchestration is deleted: up-and-in call,
  down-and-in put, double knock-in, price, solver Greeks, and bump-context Greeks; the
  same products already knocked in. Delayed settlement: a
  Snowball and a Phoenix with a positive settlement lag, price, Greeks, and statistics.
- The 244-day environment row, the intraday row, the DCN rows, the local-vol vanilla and
  barrier rows, and the 2D rows of 0.6.0, unchanged.

### 9.2 Anchors, identities, evidence

`assert_anchors` for every banked PDE certificate and `test_banked_cells_keep_their_identity`
pass unchanged (snowball `c5b6a27b…`, phoenix `6a9aae02…`, ko-reset `f2c9bd52…`). New:
`resolved_grid_evidence(accuracy)` equals the banked `grid` dict of the three guarded
studies for `standard`; the intraday schema-2 certificate's implementation digest moves
(by design, as in 0.6.0) and its anchors pass.

### 9.3 New tests

- `test/pde_core/test_groups.py`: each family's group declaration equals its historical
  solve calls (a literal table: columns per group, backend, order).
- `test/pde_core/test_march_step.py`: a synthetic two-column banded group advanced by
  `CrankNicolson.step` equals, at the `test/golden_compare.py` tolerance, a reference step
  computed in the test from the legacy formulas (no frozen array: a byte-pinned array
  would be a cross-architecture golden); same for the LU single column and the statistics
  `(n, 2·n_cols)` group. The exact gate is the same-machine byte book (section 9.1).
- `test/pde_core/test_components.py`: toy `BackwardEuler` stepper, `ShiftedOperator`,
  `NearestNodeReadout` conform; a stepper returning a NaN or a wrong shape raises
  `NumericalError` naming it; a component that is a plain class has no fingerprint,
  prices, gets no session artefact, and is refused by the modelvalidation builder;
  `CrankNicolson(theta=1.0)` and the toy `BackwardEuler` agree to `Tolerance` on a European.
- `test/pde_core/test_prepare.py`: an expired product's `calculate_event_stats` returns
  `None` and builds no grid; a knock-in barrier prepares two subproblems whose upper
  bounds are the automatic envelope and the barrier, composed as vanilla minus knock-out
  for price and each Greek; the six entry points produce the same problem
  identity for the same request; two Snowballs differing only in KO coupon share every
  artefact key and differ in problem identity, likewise a knocked-in versus alive
  lifecycle and a settlement lag; `explain` on a five-year `standard` Snowball reports
  5,040 requested, 5,000 delivered, `fill_scaled`; on a three-year `high`, 6,048 and 5,000;
  `explain` raises where `price` would; an expired product and a product knocked out at
  valuation give a terminal plan and the short-circuit values.
- `test/pde_core/test_market_memos.py`: after a solve, mutating the rate curve's `rate`
  and preparing again re-samples (the coefficients and discount factors follow the live
  value); an unfingerprintable market object is sampled fresh every time; changing the
  environment's calendar re-resolves the settlement dates; a spot change with a
  spot-reading vol surface (a test double) re-samples the coefficients.
- `test/pde_core/test_identity.py`: a three-step budget with no interior node, with a
  readout node, and with an event node at the same time have different layout fingerprints
  and different factorization identities; a holiday added to a product-supplied settlement
  calendar changes the observation key and the problem identity; a frozen bump solve and
  a fresh solve at the same spot have different layout fingerprints, coefficient keys, and
  problem identities; two
  local-vol solvers with prebuilt 15% and 35% surfaces have different coefficient keys; two
  operators with different settings and identical `evidence()` have different fingerprints
  and different keys; two readouts give different problem identities and the same
  factorization key; a bump instance of a local-vol solver carries the prebuilt surface.
- `test/modelvalidation/test_candidate_wiring.py`: a candidate declaring
  `stepper: {theta: 1.0}` executes θ=1.0 and records it; a declared key `from_config`
  rejects fails the study at load; the three guarded studies' `grid` records are
  byte-identical to the banked dicts.
- `test/pde_core/test_factorization_cache.py`: two θ values agreeing to six decimals get
  two factorizations; `dt` values within an ULP share one (the monthly snowball's five
  distinct `dt` floats resolve to one key).
- `test/pde_grid/test_time_builder.py`: `TimeLayout` has no damping sets; the stepper's
  schedule equals the frozen 0.6.0 schedule on the existing fixtures (terminal wins,
  spill backward, readout nodes undamped).
- `test/test_pde_params_surface.py`: the four removed names raise with the replacement,
  on `PDEParams` and on `GridConfig`; `from_config` round-trips `stepper`; `operator` in
  a config raises; `GridConfig.key` has nine entries.
- Session: fingerprints move when the stepper config moves; the stored `Layout` is what
  execute binds; `pde-prep/v2`.

### 9.4 Test changes

Tests that drive deleted private seams are rewritten against the core's public seams
(`prepare`, `march`, `explain`, the component protocols) with the same assertions. The
AST scan in the plan lists them; known today: `test/execution/test_pde_artifact_reuse.py`,
`test_pde_prepared_adapters.py`, `test_pde_session_seams.py`, `test_kernel_prepare.py`;
`test/helpers_pde_zero_diffusion.py`, `test/intraday/test_pde_zero_diffusion_two_surface.py`,
`test/intraday/test_route_pde.py`; `test/pde_grid/test_frozen_bump_layouts.py`,
`test_solver_seam.py`; `test_event_stats_pruning.py`, `test_pde_event_projection.py`,
`test_pde_fixed_bump_grid.py`, `test_pde_quad_audit_regressions.py`,
`test_pde_term_coefficients.py`, `test_pde_term_structure_solvers.py`,
`test_trade_stream_wiring.py`; and every test that passes `theta=`, `event_theta=`, or a
damping count on `GridConfig`. `test/execution/goldens/pde_phase4_goldens.json` keeps its
numbers; if its structure names artefact kinds or `pde-prep/v1`, it is re-frozen once for
structure, declared in the commit.

### 9.5 Performance

Cold and warm timings per family on the book, both arms in one window, five repetitions
after the same warm-up; no median regression above ten percent per family; peak memory
per family measured in isolation, same bound. An unstable measurement is inconclusive,
not a pass.

## 10. Migration order

Each step is its own commit (or a few), the legacy loop deleted in the commit that moves
its family, the byte gate run after each:

0. The exact θ in the factorization key (decision 6).
1. The core package with the base family and American on it, the knock-in composition
   as a `CompositeProblem` with its byte rows landed first. `_time_stepping` and
   `_get_matrices` are deleted; the three coefficient builders' bodies move into
   `CentralDifference`, and the legacy methods delegate to it until their last caller
   moves.
2. Snowball on the core, with the life surface and the local-vol snowball;
   `_time_stepping_two_surface` deleted, `event_schedule` folded into `SnowballAdapter`,
   `_local_vol_coefficients` into `CentralDifference`'s nodal path.
3. Phoenix and the local-vol phoenix; `_time_stepping_vector_surface` / `_step_grids`
   deleted.
4. KO-reset.
5. The statistics group; `_compute_event_stats`, `_get_banded_system`, and the legacy
   coefficient delegates deleted (no caller remains); the 2D solvers' statistics-only
   march.
6. One preparation path for sessions, bumps, intraday, and the life-surface pricer;
   `pde_session_prep.py` and the adapters re-pointed; the transient-attribute machinery
   deleted.
7. The scheme relocation, the evidence record, `explain`, docs, changelog, version.

The plan sequences the commits so that main stays green after each.

## 11. Docs and release

- `docs/engine_param_guide.md`: the component slots, the `CrankNicolson` table with the
  "when to change it" lines of section 4.1, an `explain` example with its text view, the
  removal additions.
- `docs/pde-core.md` (new): the three stages, `PreparedProblem`, column groups and
  policies, adapters, the protocols with a worked custom component, the fingerprint rule,
  the evidence record.
- `quantark/asset/equity/CLAUDE.md` ("Engine Parameters"), `README.md` snippet,
  `docs/execution/capability-matrix.md` (PDE session adapters through the core),
  `docs/modelvalidation/RELEASE_PROCEDURE.md` (a sentence on the composed `grid` record).
- `CHANGELOG.md` `## [0.7.0]`: Added (`stepper` / `operator` / `readout`, `explain`,
  `resolved_grid_evidence`, `OutputRequest`), Removed (section 4.4 table), Changed (the
  exact θ in the factorization key; statistics computed in the price march;
  `calculate_event_stats` no longer re-prices; sessions store the layout; market memos
  keyed by value, so mutating a market object after a solve is no longer silently stale;
  certificate candidates built from their declared params), and the sub-project 3 list of section
  12.
- `pyproject.toml` version 0.7.0.

## 12. Sub-project 3 handoff (spec 1 section 10, updated)

Unchanged items: the environment's business day as the grid day (`day_count` removed,
every identity moves, full re-certification); half-step damping as a `CrankNicolson`
option; coefficient-break nodes; a budget default for path-independent routes; fail
instead of scale at the caps; DCN onto the core; the local-vol vanilla/barrier space axis
onto the shared grid; native KO-reset statistics. Added by this spec, each its own
attributable commit under the full re-certification:

| Item | Effect |
|---|---|
| Statistics Level 2: the statistics group takes the value policy (adapter Dirichlet boundaries per column, the θ override, the adapter's event order with `ko_last`, the value projection) and a maturity-leg column is marched, so `reconciliation_error` is a real round-off check; Phoenix coupon columns follow the memory states, so memory coupons get a cashflow decomposition | every KO, KI, and coupon probability moves by a discretization-sized amount; replay event goldens and the legacy Dask goldens re-frozen; qualified against MC event statistics and the reconciliation identity |
| Phoenix value groups take the zero-diffusion θ override | Phoenix moves on trading-clock environments |
| Snowball NODAL applies to interior schedule events, or NODAL is withdrawn from the shared-grid table | NODAL rows move |
| Native 2D Heston/SLV statistics | 2D statistics move; a candidate, not yet scheduled |

A merged single solve call for value and statistics columns is not planned: per-group
calls are what keep the value columns bitwise, and the saving would be loop overhead only.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| A slip inside the 3,400-line snowball migration changes a number. | Family-by-family gate; the book's continuous-KI, knocked-in, daily-KI, NODAL, term, and life-surface rows; the frozen-step-array test of section 9.3. |
| The statistics group interleaves with value events. | Groups are independent within a step (no value column reads a statistics column); the statistics rows on every autocallable, flat and term. |
| The execution goldens encode artefact structure. | Numbers unchanged; structure re-frozen once, declared in the commit. |
| A 2D or DCN read of the old damping fields is missed. | Grep gate for `terminal_damping_steps` / `event_damping_steps` / `params.theta` / `params.event_theta` outside `components.py` and the removal map must be empty; the 2D and DCN book rows. |
| Abstraction overhead in the step loop. | Section 9.5 timing gate; the per-step work is the same solves, and the adapter tables are built once in `prepare`. |
| `explain` drifts from what `price` does. | They are one function with the march skipped; `test_prepare` asserts the same fingerprint from every entry point. |
| A custom component silently degrades results. | Schedule, coefficient, index, and finiteness validation (section 5.8); failures propagate; no fingerprint means no artefacts; no custom component is certifiable. |
| An identity-keyed cache serves a mutated market object. | Decision 14: the environment is read only in `prepare`, memos key on value fingerprints; `test_market_memos.py`. |
| A session artefact is reused for an economically different product. | Artefact keys carry geometry and market only by design; the problem identity carries the economics; `test_prepare.py` asserts the split. |
| A key describes the request while a frozen layout delivered something else. | Decision 17: the layout fingerprint keys coefficients and identity; `test_identity.py`. |
| The byte gate passes on a shared failure. | Decision 20: manifest, finiteness, declared failures and declared `None`s. |
| The knock-in composition is flattened onto one layout. | Decision 21: two prepared problems with their own domains; the knock-in byte rows land before the orchestration is deleted. |
| The evidence record hides a real change. | It composes only the nine grid fields and the stepper's two counts; a non-default stepper, operator, or readout enters the evidence; the identity guard runs after the relocation. |
| Sub-project 1 merge slips and this spec's base moves. | The byte book base is the merged commit; the plan starts from `main` after the merge. |

## 14. Review record (2026-09-30)

An external design review (seven findings, with a read-only probe run against the
worktree) was verified in the repository before adoption. All seven were confirmed.

| Finding | Verified | Change here |
|---|---|---|
| Exact factorization keys break the bitwise gate: the key rounds `dt` as well as θ. | A monthly snowball's 12 intervals give 5 distinct `dt` floats (spread 1.3e-18) and 1 rounded key; the probe's exact-key patch moved the price from `97.74119755943117` to `97.74119755943177` and moved delta and gamma on the term environment. | Decision 6 narrowed to θ; `dt` rounding kept and documented (section 5.7). |
| The prepared fingerprint omitted economic inputs. | Same `GridRequest`, KO coupon 15% versus 30%: `97.74119755943117` versus `100.56613635265587`. | Decision 13: artefact keys versus problem identity (section 5.9); tests. |
| The environment memos key on object identity. | `FlatRateCurve` and the other market params are plain `@dataclass`, mutable; after `rate_curve.rate = 0.07` the coefficient builder returned the memoized object and the discount memo returned `0.9880717128619306` for a live `0.9723883668012469`. | Decision 14: value-fingerprint keys computed once per `prepare`; `prepare` the only environment reader; test. |
| Evidence not tied to executed components; value fingerprints do not cover code. | `SnowballPDECandidate._greeks` constructs `PDEParams(accuracy=...)` while `params()` records the declared mapping: a declared `stepper: {theta: 1.0}` executed θ=0.5. Reassigning a method on a frozen-dataclass component left `try_fingerprint` unchanged. | Decision 15: engines built from `from_config(declared)`; certification limited to built-in components; the declared dicts of every banked PDE candidate are `{accuracy: standard}`, so no identity moves. |
| Local-vol operator ownership conflicted between preparation and adapters. | Design defect in the 2026-09-30 text. | Sampling (route's model) separated from the stencil (operator factory); `CentralDifference` takes scalar and nodal samples on today's arithmetic; a custom operator declares the kinds it supports; the continuous-KI correction samples from the same object (sections 5.2, 5.6, 5.8). |
| Preparation lacked terminal and immediate-KO outcomes. | `_price_with_solution` returns before any grid for expired and knocked-out-at-valuation products; `GridRequest` requires a positive maturity. | Decision 16: `prepare` step 0; byte rows for expired, immediate KO, and delayed settlement. |
| Component validation allowed malformed but finite inputs. | Design gap in the 2026-09-30 text. | Section 5.8: schedule length and range, integer index bounds, zero-diffusion subset, ownership. |

**Second round (2026-09-30), six findings, all confirmed:**

| Finding | Verified | Change here |
|---|---|---|
| The grid key describes the request, not the delivered layout. | A bump clone frozen at 100 and a fresh solver, both at 101: same request key, different `spatial.x`, prices `98.25175716148267` versus `98.25326843586967`. | Decision 17: `Layout.fingerprint`; coefficient key and identity use it. |
| The coefficient identity omitted a prebuilt local-vol surface. | `LocalVolSnowballPDESolver(local_vol_surface=...)` at 15% and 35%: same curve fingerprint, prices `100.0318` and `94.5103`. The session adapter already keys on the Dupire surface; the spec's key did not. | Decision 18: samples provenance in the coefficient key; bump instances keep constructor arguments. |
| Cache keys incomplete: calendar; spot. | `_observation_cache_key` carries valuation date, day count, and `bus_days_in_year` but no calendar: a holiday added after a solve left the cached settlement on January 12 against a fresh January 13. `TermCoefficients.from_env` passes the spot into every `get_vol`, and the memo key omits it; no shipped surface reads the spot in a value-changing way (flat and grid ignore it, sticky moneyness is a new view object per shock), so the key fix is bitwise on the book. | Decision 19: per-memo input tables, `Calendar.identity()`, spot in the coefficient key. |
| Terminal handling conflated output contracts. | `price_with_events` short-circuits in `_price_with_solution`; `calculate_event_stats` has no short-circuit and marches with the KO at step 0; `calculate_greeks` returns the KO payoff with zero delta and gamma; `solve_life_surface` raises. | Decision 16 restated per operation; the direct statistics row added; the divergence listed in section 8. |
| Component identity inconsistent; readout missing from the problem identity. | Section 5.8 keyed on `try_fingerprint`, section 5.9 on `evidence()`; the built-in operator and readout both return `{}`. | Decision 8 restated: one identity function; readout in the problem identity. |
| `compare.py` passes on a shared failure. | It compares row dicts for equality; a row that raised the same error on both sides is equal. | Decision 20: manifest, finiteness, declared expected failures. |

**Third round (2026-09-30), four findings, all confirmed:**

| Finding | Verified | Change here |
|---|---|---|
| The layout fingerprint omitted `dt` and the θ schedule. | Three binds of a three-step budget (no interior node; a readout node at 2/3; an event node at 2/3) share `t` bit for bit, differ in the constructed `dt` by an ULP, and carry schedules `[0.5, 0.5, 1]` against `[1, 1, 1]`. | Decision 17 amended: fingerprint over `x`, `dx`, `t`, `dt`, and the node classes; the effective per-group schedule in the factorization identity. |
| Observation caches missed the product's settlement calendar. | `SettlementConvention.calendar` exists; `settlement.py` resolves `convention.calendar or pricing_env.calendar`; `Calendar.__repr__` is a name and a holiday count; the product's `cache_key()` names a calendar by name. Adding a holiday to a product calendar left the product key and the cached records unchanged (January 12 against a fresh January 13). | Decision 19 amended: the effective settlement calendar's identity in the observation key and the economic identity. |
| Expired statistics were specified to march. | `_compute_event_stats` returns `None` when `tau <= 0`, before any grid; the revised table would have reached `GridRequest`, which rejects a non-positive maturity. | Terminal table corrected; manifest gains a declared `None` kind. |
| Knock-in composition was missing. | `BarrierPDESolver` and `DoubleBarrierPDESolver` price a knock-in as `_price_vanilla` (a fresh `EuropeanPDESolver`) minus the knock-out, for price and Greeks; two marches with upper bounds `222.554…` and `120`. Inside a bump context the knock-out layout is frozen and the vanilla rebinds each solve. | Decision 21: `CompositeProblem`; the split preserved; knock-in byte rows before the orchestration is deleted; listed in section 8. |
