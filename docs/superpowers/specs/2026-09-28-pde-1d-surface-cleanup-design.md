# PDE 1D Surface Cleanup — Design Spec (sub-project 1 of 3)

**Date:** 2026-09-28
**Status:** approved design, awaiting spec review
**Release:** 0.6.0 (clean break, matching the 0.4.0 convention: removed, not deprecated)
**Scope:** the 1D equity PDE pricers — every solver dispatched by `PDEEngine`, the
1D local-vol solvers, and `DCNPDEEngine`. The 2D Heston/SLV solvers are touched only
where they read a removed `PDEParams` field.

This is the first of three sub-projects that together implement the "common core
and numerical policy" plan (section 10). Sub-project 1 is a **bitwise** cleanup of
the parameter surface and the legacy code paths behind it. It changes no number on
any solver that has a golden or a certificate.

---

## 1. Problem

A user pricing an equity product on the 1D PDE path faces about 45 controls spread
over `PDEEngine(method=...)`, 23 `PDEParams` fields, 10 `GridConfig` fields, a
five-name profile tier (`make_pde_params`), and three class-level cache setters.
Measured against the code that consumes them:

- **Dead.** `PDEEngine.method` writes `params.scheme`, which `PDEParams` does not
  declare and no equity solver reads. `GridConfig.day_count` has zero callers.
  `BackwardOperator` (144 lines, 7 tests) has zero callers. Three `DCNPDEEngine`
  constructor arguments are stored and never read.
- **Duplicated.** `use_rannacher`/`rannacher_steps`/`rannacher_at_events`/
  `event_rannacher_steps` and `GridConfig.terminal_damping_steps`/
  `event_damping_steps` describe one damping schedule, bridged by
  `scheme_config_overlay`. `cache_enabled`, `cache_strategy="disable"`, and
  `BasePDESolver.set_cache_enabled` are three ways to switch one cache off. The
  profile names `fast`/`balanced`/`accurate` alias `accuracy`; `barrier_sensitive`
  and `reverse_sensitive` are identical for PDE.
- **Legacy paths.** `event_projection=NODAL`, `continuous_ki_correction=NONE`, and
  `boundary_mode="default"` keep pre-fix code alive; their only callers are the
  tests of the fixes themselves.
- **Internals exposed.** `use_banded_solver`, `grid_cache_max_entries`,
  `banded_cache_max_entries` are implementation details with zero or one caller.
- **Legacy resolution knobs.** `grid_size`/`time_steps`/`s_min`/`s_max` are read
  only by `LocalVolPDESolver` and `LocalVolBarrierPDESolver`; every other solver
  raises on a non-default value.
- **Stale docs.** `docs/engine_param_guide.md` documents six knobs deleted in 0.4.0
  and uses the deprecated flat import path.

The user's stated pain: "a lot of params which makes it hard to choose and hard to
understand what will result in changing them."

## 2. Goals

1. Every surviving control has a one-line answer to "when would I change this, and
   what happens".
2. Default numbers are bit-identical before and after on every solver with a golden
   or a certificate. Proven by byte-compare, not asserted.
3. Removed names fail loudly with the replacement named.
4. Certification evidence stays valid: the release-procedure "anchors only" path.

### Non-goals (deferred to sub-projects 2 and 3, section 10)

- Unifying the stepping kernels (five hand-written copies of the θ-step).
- Any numerical policy change (half-step damping, coefficient-break nodes, fail
  instead of scale, DCN or local-vol vanilla/barrier onto the shared grid, native
  KO-reset event statistics).
- Expert extension protocols (operator factory, time stepper, readout).
- Renaming `steps_per_day` to `steps_per_year`: dropped, not deferred (decision 6 and
  the note at the end of section 10).
- `EngineParams.bump_size` (deprecated) and `EngineParams.bus_days_in_year`: shared
  with MC, Quad, and the analytical engines, out of scope. The PDE path stops reading
  the params field; the grid's day convention comes from the pricing environment
  (decision 6). That two objects carry a `bus_days_in_year` is a cross-engine
  inconsistency to resolve in its own change, not here.

## 3. Decision log (settled with the user)

| # | Decision | Rationale |
|---|---|---|
| 1 | Delete, do not deprecate. Removed names raise `ValidationError` naming the replacement. | 0.4.0 convention; a silent no-op knob is the failure mode we are removing. |
| 2 | Delete the legacy modes `continuous_ki_correction=NONE` and `boundary_mode="default"` and the code behind them. | Their only callers are the tests of the fixes. |
| 3 | **Keep** `event_projection` (CELL_AVERAGE default, NODAL expert choice). | Reverses an earlier "delete" decision: the pasted core plan lists nodal projection as a supported expert alternative, and it is cheap to keep once the checks are consolidated. |
| 4 | Keep the built-in scheme's four settings user-facing: `theta` and `event_theta` on `PDEParams`, `terminal_damping_steps` and `event_damping_steps` on `GridConfig` (today's placement). Delete only the four duplicate Rannacher knobs. | An expert who wants fully implicit stepping or heavier damping sets a field; requiring a stepper component for that would lock experts out (user, 2026-09-29). Under the core plan the built-in Crank-Nicolson stepper is a component with its own configuration, and these four settings are it; sub-project 2 moves them together into that component's config object. Fixing them as constants was rejected. |
| 5 | Keep exactly one cache control, `cache_enabled`. | Same numbers either way; the other controls were keying heuristics and sizes. |
| 6 | Keep the name `steps_per_day`; remove `GridConfig.day_count`; a "day" is one business day of the **pricing environment's** convention (`PricingEnvironment.bus_days_in_year`: 244 on the SSE/CFFEX desks, 252 on CFETS, per `docs/trading-clock.md`). No constant anywhere. | The desk convention is market state and is already declared once on the environment, where it also drives the vol clock and per-trading-day theta; the grid follows it. A second field or a hardcoded 252 would be a second source of truth. The name survives because certification study YAMLs name `steps_per_day` (the anchor test rebuilds studies from the YAML text embedded in each anchor file) and because, with the day defined this way, the name is exact. |
| 7 | Migrate the two 1D local-vol solvers to `GridConfig` now by field mapping, kernel unchanged. | Deletes the four legacy knobs from the surface today. These two solvers have no goldens or certificates; their tests compare against Black-Scholes and analytic barriers. Sub-project 3 moves them fully onto the shared grid. |
| 8 | `DCNPDEEngine` keeps a bespoke two-argument constructor. | One step per observation day is part of its certified event semantics, so `accuracy` cannot apply to it until it joins the core in sub-project 3. |
| 9 | `PDEParams.from_config` stays; the PDE profile tier goes. | Batch/YAML users keep a loader over the real field names; the profile names were aliases. |
| 10 | Add `PDEParams.resolved_grid()`. | Directly serves "understand what changing it does": shows the fully resolved `GridConfig` before pricing. |

## 4. Parameter surface

### 4.1 `PDEParams` (final)

```python
@dataclass
class PDEParams(EngineParams):
    accuracy: str = "standard"                     # fast | standard | high
    grid: Optional[GridConfig] = None              # expert overlay, section 4.2
    theta: float = 0.5                             # scheme weight away from discontinuities (0.5 CN, 1.0 fully implicit)
    event_theta: float = 1.0                       # scheme weight on the event-damped steps
    event_projection: EventProjectionMode = CELL_AVERAGE   # NODAL is the expert alternative
    ki_monitoring_mode: KnockInMonitoringMode = EXACT_DISCRETE  # BGK_APPROXIMATION opt-in
    cache_enabled: bool = True
    # inherited from EngineParams: bump_config, bus_days_in_year (not read by PDE), bump_size (deprecated)
```

One-line contract per field, which is also the docs table:

| Field | When to change it | What happens |
|---|---|---|
| `accuracy` | You want a faster preview or a finer answer. | Selects the `GridConfig` preset: fast (200 pts, 4/day), standard (400, 4/day), high (800, 8/day). Resolution settings, not accuracy guarantees. |
| `grid` | You need to override one preset field. | Each non-`None` field replaces that preset value; the rest inherit. |
| `theta` | Ringing at a discontinuity that damping does not cure, or you want a monotone scheme. | 0.5 is Crank-Nicolson (second order in time); 1.0 is fully implicit (first order, monotone, more diffusive). Validation stays [0, 1] as today. |
| `event_theta` | You are studying damping strength after events. | The weight on the `event_damping_steps` after each interior event; 1.0 is backward Euler (Rannacher restart). Terminal damping steps always use 1.0. |
| `event_projection` | You are validating the event treatment. | CELL_AVERAGE splits the trigger cell by its exact overlap (removes the half-cell bias); NODAL assigns whole nodes (legacy, biased by up to half a cell). |
| `ki_monitoring_mode` | A dense daily KI schedule is too slow. | BGK replaces the discrete schedule with continuous monitoring at a shifted barrier; inert unless the KI is discretely monitored. First-order residual bias. |
| `cache_enabled` | Memory pressure, or you want a fresh solve every call. | Same numbers either way; off means grids and factorizations are rebuilt per solve. |
| `bump_config` | You want different bump sizes for numerical Greeks. | Unchanged. |

Enum coercion from lower-case strings stays for `event_projection` and
`ki_monitoring_mode`.

### 4.2 `GridConfig` (final)

```python
@dataclass(frozen=True)
class GridConfig:
    points: Optional[int] = None          # spatial nodes
    steps_per_day: Optional[float] = None # fill per business day of the environment's convention (bus_days_in_year)
    eps_crit: Optional[float] = None      # target log-spacing at critical prices
    num_std: Optional[float] = None       # half-width of the auto domain in vol·sqrt(T) units
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None  # per-side hard override
    max_points: Optional[int] = None      # resource cap on points
    max_steps: Optional[int] = None       # resource cap on time steps (still scales the fill in this release)
    terminal_damping_steps: Optional[int] = None  # backward-Euler steps off the terminal payoff (0 disables)
    event_damping_steps: Optional[int] = None     # backward-Euler steps after each interior event (0 disables)
```

`ACCURACY_PROFILES` keep points/steps_per_day/eps_crit as today; `_SHARED_DEFAULTS`
lose only `day_count` (damping defaults stay 1 and 2). `GridConfig.key` therefore
has nine entries. `resolve_config(accuracy, override)` is unchanged in semantics.

The two damping counts are the built-in scheme's restart controls (section 4.3):
more terminal steps damp the payoff kink harder at first-order cost on those steps;
two event steps is the field-tested minimum for smooth convergence with projected
event data (Pooley-Vetzal-Forsyth 2003). Zero disables that damping kind, which is
what the deleted `use_rannacher` / `rannacher_at_events` booleans did.

The day convention travels with the market, not the config:

- `MarketSnapshot` gains `bus_days_in_year: int` (required, positive), set by each
  solver's `market_snapshot(product, env)` from `pricing_env.bus_days_in_year`.
- `GridRequest` gains `day_count: int` (required, positive), set by the three library
  request builders (`_generic_grid_request`, the snowball family's, DCN's) from
  `market.bus_days_in_year`. It is part of the frozen geometry, so it is in the
  layout cache key and survives `rebind_time` on calendar bumps.
- `build_time(request, config)` reads `request.day_count`. Callers that already hold
  a request (`quantark/intraday/engines/pde.py`) need no change.

Under `DayCountConvention.BUSINESS_DAYS` the product `Δt · day_count` is the exact
number of business days in an interval, so `steps_per_day` is literally steps per
business day. Under `CALENDAR_DAYS` it is the trading-day equivalent at the desk's
convention, the same D the desk quotes vol in.

### 4.3 The built-in scheme and its four settings

The built-in scheme is Crank-Nicolson with backward-Euler damping. Its
configuration is exactly four settings, all user-facing (decision 4): `theta` and
`event_theta` on `PDEParams`, `terminal_damping_steps` and `event_damping_steps` on
`GridConfig`. Their defaults (0.5, 1.0, 1, 2) equal today's, which is what makes
the change bitwise. The per-step schedule is what `_theta_schedule_from_layout`
computes today: `theta` everywhere, `event_theta` on the `event_damping_steps`
after each interior event node, 1.0 on the `terminal_damping_steps` off the payoff,
terminal winning on overlap.

There is no policy module and no day-length constant: the day is the environment's
(section 4.2), and the scheme's defaults are the fields' defaults. The two cache
sizes become module constants in `base_pde_solver.py` (`GRID_CACHE_MAX_ENTRIES = 128`,
`BANDED_CACHE_MAX_ENTRIES = 512`, today's values). Sub-project 2 moves the four
scheme settings together into the built-in stepper's configuration object;
sub-project 3 adds the half-step damping mode there.

### 4.4 Loud failure on removed names

`PDEParams`, `GridConfig`, and `PDEEngine.__init__` reject unknown keyword names
with `ValidationError`, not `TypeError`. One removal map (module-level, also rendered
into the docs and the changelog) supplies the replacement text:

| Removed name | Replacement text |
|---|---|
| `PDEEngine(method=...)` | Crank-Nicolson is the only built-in scheme; `PDEEngine(params=PDEParams(...))`. |
| `grid_size` | `grid=GridConfig(points=...)` |
| `time_steps` | `grid=GridConfig(steps_per_day=...)` |
| `s_min`, `s_max` | `grid=GridConfig(bounds=(lo, hi))` |
| `use_rannacher`, `rannacher_steps` | `grid=GridConfig(terminal_damping_steps=N)`; 0 disables |
| `rannacher_at_events`, `event_rannacher_steps` | `grid=GridConfig(event_damping_steps=N)`; 0 disables |
| `boundary_mode` | removed; asymptotic boundaries are the only treatment |
| `continuous_ki_correction` | removed; the first-passage correction always applies to continuously monitored KI |
| `cache_strategy`, `grid_cache_max_entries`, `banded_cache_max_entries` | `cache_enabled=True/False` is the only cache control |
| `use_banded_solver` | removed; the linear solver is chosen by the solver family |
| `GridConfig.day_count` | removed; the day is the pricing environment's `bus_days_in_year` |

Names not in the map raise the same `ValidationError` without a replacement line.
Implementation: capture the dataclass-generated `__init__`, wrap it with a keyword
check; positional use is unchanged. `reject_legacy_resolution_knobs`,
`_LEGACY_KNOB_REPLACEMENTS`, and `scheme_config_overlay` are deleted.

### 4.5 `from_config` and the profile tier

- `PDEParams.from_config(mapping_or_path, **overrides)` loads a dict, YAML, or JSON
  whose keys are `PDEParams` field names; `grid` is a nested mapping of `GridConfig`
  fields. `profile`, `reverse`, and `engine != "pde"` keys are rejected. Unknown keys
  go through the removal map.
- Deleted: `PDE_PARAM_PRESETS`, `make_pde_params`, the `"pde"` branch of
  `make_engine_params` and `list_param_profiles` (both raise `ValidationError`
  saying PDE has no profile tier). Quad profiles are untouched.
- `quantark/asset/equity/param/__init__.py` stops exporting `make_pde_params`.

### 4.6 `PDEParams.resolved_grid()`

Returns `resolve_config(self.accuracy, self.grid)`. Pure convenience; the solvers
keep calling `resolve_config` through the binder.

## 5. Solver changes on the 1D Black-Scholes path

All edits below preserve the arithmetic and its order on the default path. The
byte-compare gate (section 8) is the proof.

### 5.1 `PDEEngine`

- `__init__(self, params=None, **removed)`; `method`, `DEFAULT_METHOD`,
  `_apply_method_to_params`, and the `params.scheme` write go. `__repr__` shows
  `PDEEngine(accuracy='standard')`.
- `create_bump_context` clones with `type(self)(params=self.params)`.

### 5.2 `BasePDESolver`

- Constructor reads `cache_enabled` only. `_cache_strategy` and the class attributes
  `_global_cache_enabled`, `_global_cache_strategy`, and the class methods
  `set_cache_enabled`, `set_cache_strategy`, `set_grid_cache_max_entries`,
  `clear_grid_cache` (a no-op since 0.4.0) are deleted. `clear_grid_cache` has five
  external call sites (`test/test_snowball_pde.py`, `test/execution/benchmark_phase4.py`
  ×3, `test/execution/test_pde_artifact_reuse.py`, `example/snowball_pde_perf_demo.py`);
  the calls are removed. `_resolve_cache_strategy` is replaced by
  `_is_cache_enabled()` returning the instance flag; product cache tokens use
  today's "standard" keying unconditionally.
- The collapse guard's `getattr(self.params, "strict_smile", False)` stays as is.
  `strict_smile` is not a `PDEParams` field today, nothing sets it, and the MC
  engines share the read; it is outside this scope and unaffected by the keyword
  check, which applies to constructor arguments only.
- `_theta_schedule_from_layout` is unchanged: `params.theta` everywhere,
  `params.event_theta` on event-damped steps, `1.0` on terminal-damped steps.
- `_params_cache_key` becomes `(self.grid_binder.config.key, params.event_projection,
  params.theta, params.event_theta)`.
- `_pack_uses_banded` stays a per-family class decision (base: sparse LU; the
  two-surface family: banded when the interior has more than two nodes).
- Grid cache and critical-points cache sizes read the module constant
  `GRID_CACHE_MAX_ENTRIES = 128`.
- `params.bus_days_in_year` is no longer read. `market_snapshot` passes
  `pricing_env.bus_days_in_year` into the snapshot (section 4.2); the two existing
  reads of `pricing_env.bus_days_in_year` in cache keys stay.

### 5.3 Time builder

`build_time` reads `request.day_count` instead of `config.day_count`; the damping
counts still come from the resolved config. The fill formula is otherwise
unchanged: `fill = max(1, round(max(1, Δt · day_count) · steps_per_day))`. On a
252 environment this is bit-identical to today. On a 244 environment the fill
follows the desk's own business days, about 3% fewer steps than today's fixed
252; no golden or certificate uses a non-252 environment (checked: all seventeen
244 sites are tests and one demo), so this is a behaviour change only where the
old behaviour was wrong. `max_steps` keeps today's scale-with-warning behaviour in
this release (changing it is a sub-project 3 item).

### 5.4 Snowball, Phoenix, KO-reset

- `_use_cell_average_events` reads `self.params.event_projection is CELL_AVERAGE`
  (no `getattr` default). The projection helpers keep both strategies; the checks
  stay where they are. This is the one legacy branch that survives, by decision 3.
- `_prepare_continuous_ki_correction` loses its `continuous_ki_correction` guard;
  the correction is prepared whenever the KI is continuously monitored. Same in
  `vol_continuous_ki.py` for the vol-model mixins.
- The two `boundary_mode == "asymptotic"` conditions in the snowball boundary
  methods become unconditional.
- Banded selection: `use_banded = (num_x - 2) > 2` in the two-surface sweep and in
  the phoenix vector sweep; the event-stats sweep raises on `(num_x - 2) <= 2` as it
  does today for the non-banded case.
- `_banded_cache_max_entries` reads the module constant `BANDED_CACHE_MAX_ENTRIES = 512`;
  the KO-records and KI-profile caches read `GRID_CACHE_MAX_ENTRIES`.
- `SnowballPDESolver(enable_profiling=...)` is unchanged (one caller; diagnostic,
  not numerical).

### 5.5 `BackwardOperator`

Deleted with `backward_operator.py` and `test/test_backward_operator.py`. Its
`theta_by_step` had no callers; the layout-derived schedule is the only one.

### 5.6 Session preparation and execution adapters

- `pde_session_prep.py`: `engine._resolve_cache_strategy() == "disable"` becomes
  `not engine._is_cache_enabled()`; `banded_policy` comes from
  `engine._pack_uses_banded(num_x)` rather than the params field.
- `pde_execution_adapters.py`: the four `_session_matrix_pack` references are
  unchanged in behaviour; only the cache-flag read moves.
- `quantark/intraday/engines/pde.py` uses `replace(grid, steps_per_day=..., max_steps=...)`
  and `replace(grid, points=..., max_points=...)` — unchanged.

### 5.7 Enums

`ContinuousKICorrection` is deleted from `engine_enums.py` (only the PDE path used
it). `PDEMethod` stays (the convertible-bond engines use it). `EventProjectionMode`
and `KnockInMonitoringMode` are unchanged.

## 6. Local-vol 1D solvers and DCN

### 6.1 `LocalVolPDESolver` and `LocalVolBarrierPDESolver`

Kernels in `quantark/volmodels/localvol/pde_kernel.py` are unchanged (the FX
local-vol solver shares them). The two equity solvers derive kernel arguments from
`resolve_config(params.accuracy, params.grid)`:

| Kernel argument | Source |
|---|---|
| `n_s` | `points` |
| `step_dt`, `t_grid` | `build_time(GridRequest(tau=T, bound_anchors=(spot,), critical_prices=(spot, strike), hard_lower=None, hard_upper=None, event_times=<interior observation dates>, day_count=env.bus_days_in_year), config)`; `dt = layout.dt`, `t = layout.t` |
| `observe_steps` (barrier, discrete) | `layout.step_at(d)` for interior dates; valuation-date and maturity observations keep today's index-0 / last-index handling |
| `s_max` | `bounds[1]` if set, else `0.0` (kernel default: 4 × max(spot, strike)) |
| `theta` (both kernels), `rannacher` (vanilla kernel only) | `params.theta`, `config.terminal_damping_steps > 0` |

Fields the uniform kernel cannot honour raise `ValidationError` when set explicitly
in the overlay: `eps_crit`, `num_std`, `max_points`, and a lower bound. The error
names sub-project 3 as where these solvers join the shared grid. `max_steps` applies
through `build_time`.

The Greeks path keeps pinning `s_max` across spot bumps (`_greeks_smax`), now seeded
from `bounds[1]` or the kernel default.

**Numbers change** for these two solvers at default parameters: 200 uniform total
steps become four per trading day, and discrete observation dates become exact
nodes. Their tests compare against Black-Scholes and analytic barriers with
tolerances; the implementation reports the before/after deltas alongside the
byte-compare report.

### 6.2 `DCNPDEEngine`

`DCNPDEEngine(points: int = 801, damping_steps: int = 2)`; `num_space_nodes` is
renamed, `s_min_mult`/`s_max_mult`/`concentration` are deleted (never read). The
`points >= 201` check, the `steps_per_day=0.1` default grid config, and the stepping
loop are unchanged. `LocalVolDCNPDEEngine(local_vol_surface=None, **kwargs)` forwards
as today.

## 7. The 2D solvers

`_Heston2DSnowballPDEBase`, `_Heston2DPhoenixPDEBase`, and `event_damped_step_keys`
keep reading `params.theta` and `params.event_theta`, and replace their reads of
`params.use_rannacher` and `params.event_rannacher_steps` with the resolved
`GridConfig`'s `terminal_damping_steps > 0` and `event_damping_steps`. Same values
as today's defaults, so bitwise. `snowball_vol_pde_solvers.py` keeps reading
`bus_days_in_year` at its one site (`EngineParams` field, out of scope). Nothing else
in 2D changes.

## 8. Verification

### 8.1 Byte-compare gate (the proof of bitwise)

`test/pde_bitwise/capture_book.py` prices a fixed book and writes every output as
`float.hex()` into JSON. It is run twice on the same machine, back to back: once on
the base commit in a detached worktree (`PYTHONPATH=$PWD <main venv>/bin/python`),
once on the branch. The diff must be empty for every row except the two local-vol
vanilla/barrier solvers, whose rows are reported as deltas against their oracles.
The base JSON is not committed (cross-architecture goldens are fragile); the diff
report is attached to the merge commit message.

Book (all at `accuracy` fast, standard, and high unless noted):

- Facade products via `PDEEngine`: European, American (with and without exercise
  dates), Barrier (continuous and discrete), DoubleBarrier, OneTouch, DoubleOneTouch,
  Snowball (continuous KI, discrete KI, knocked-in), KOResetSnowball, Phoenix (with
  memory), each on a flat environment and a term-structure environment.
- Outputs per product: `price`, `calculate_greeks` (price/delta/gamma),
  `GreeksCalculator` bump Greeks through `create_bump_context`, and
  `calculate_event_stats` where supported (snowball, phoenix, KO-reset).
- `event_projection=NODAL` and `ki_monitoring_mode=BGK` rows for the snowball.
- Scheme-setting rows for the snowball and the European: `theta=1.0`, and
  `grid=GridConfig(terminal_damping_steps=0, event_damping_steps=1)`. The second
  proves the damping counts alone reproduce what `scheme_config_overlay` produced
  from the deleted booleans.
- Local-vol snowball and phoenix on a `GridVolSurface` fixture.
- Heston snowball, SLV snowball, Heston phoenix at their defaults (2D re-point).
- `DCNPDEEngine` and `LocalVolDCNPDEEngine` at defaults.
- One snowball through a prepared `PricingSession` (execution adapters) and one
  through the intraday PDE route.
- One snowball and one European on a `bus_days_in_year=244`, `BUSINESS_DAYS`
  environment with a calendar. These rows are **expected to change** (section 5.3)
  and are reported as deltas, not required to match.

### 8.2 Anchors

Every banked PDE certificate's anchor test (`assert_anchors`) must pass unchanged.
This is the release procedure's "anchors only" path for a bitwise refactor. No
study YAML is edited; none names a removed field.

### 8.3 Test changes

| Test | Change |
|---|---|
| `test_pde_engine_dispatcher.py` | Drop the `method` cases; add a case that `method=` raises with the replacement text. |
| `test_pde_engine.py` | Unchanged; `theta=` stays a field. |
| `test_pde_event_projection.py` | Unchanged in intent (NODAL stays); `event_rannacher_steps=1` becomes `grid=GridConfig(event_damping_steps=1)`, so every row keeps exactly the schedule it had. |
| `test_pde_continuous_ki_first_passage.py`, `test_vol_pde_continuous_ki.py` | Drop the `"none"` arms and the "can be disabled" test; keep the price-moves-down and convergence assertions against the MC/analytic references they already use. |
| `test/pde_grid/test_snowball_transform_oracle.py` | Deleted (self-described scaffolding for the legacy jump code). |
| `test_backward_operator.py` | Deleted with the module. |
| `test/pde_grid/test_solver_seam.py`, `test_pde_quad_audit_regressions.py`, `test_pde_term_structure_solvers.py`, `test_pde_fixed_bump_grid.py` | Replace `use_rannacher`/`rannacher_steps`/`rannacher_at_events`/`use_banded_solver`/`cache_strategy` arguments; a "no damping" schedule is `GridConfig(terminal_damping_steps=0, event_damping_steps=0)`. |
| `test/pde_grid/test_grid_config.py`, `test_time_builder.py` | Drop `day_count` from `GridConfig`; damping assertions unchanged. Replace the `make_pde_params` test with a `from_config` test. Add a time-builder case showing a 244-day request fills fewer steps than a 252-day request over the same interval. |
| `test/pde_grid/*` direct `GridRequest(...)` constructions (twelve sites) and direct `MarketSnapshot(...)` constructions | Add `day_count=252` / `bus_days_in_year=252` explicitly; the fields are required, so a test states its convention. |
| `test_snowball_pde_date_schedule.py`, `test_greeks_theta_schedule.py`, `test/intraday/test_context.py` | Run on 244 environments with PDE solvers; their numbers move by the fill change. Assertions are on schedules and structure and are expected to hold; any that pin a value is re-derived and the delta reported. |
| `test_engine_param_profiles.py` | Quad-only for profiles; PDE `from_config` round-trip; `make_engine_params("pde")` raises. |
| `test_local_vol_equity_engines.py`, `test_barrier_vol_pde_engines.py`, `test_vol_model_term_structure_compatibility.py` | `grid_size`/`time_steps` become `grid=GridConfig(points=..., steps_per_day=...)`. |
| `test_dcn_pde_solver.py`, `test_dcn_vol_pde_solvers.py` | `num_space_nodes` becomes `points`. |
| New `test/test_pde_params_surface.py` | Every removed name raises with its replacement text (parametrised over the removal map); `resolved_grid()`; `from_config` rejects `profile`; `GridConfig.key` has nine entries. |
| New in `test_local_vol_equity_engines.py` | `points`/`steps_per_day`/`bounds` reach the kernel; `eps_crit`/`num_std`/lower bound raise. |
| New in `test_heston_pde_engines.py` | 2D damping keys follow the resolved `GridConfig` counts: default equals today's set, 0 disables, and a non-default count changes the damped key set. |

Examples: `example/pde_engine_demo.py` (three `method=` sites),
`example/engine_param_profiles_demo.py`, `example/snowball_quad_mc_pde_demo.py`
(five `make_pde_params` sites) move to `PDEParams(accuracy=...)` /
`from_config`. `example/mo_volmodels/16_adi_greek_certification.py` and the
modelvalidation builders pass only `accuracy`, `cache_enabled`, `points`,
`steps_per_day` and need no change.

## 9. Docs and release

- `docs/engine_param_guide.md`: PDE half rewritten as the section 4.1 table plus
  the `GridConfig` field table and the removal map; canonical imports.
- `README.md` (the `make_pde_params` snippet) and `quantark/asset/equity/CLAUDE.md`
  ("Engine Parameters") updated to the final surface.
- `CHANGELOG.md`: `## [0.6.0]` with a "Removed" table copied from the removal map,
  a "Changed" note for the two local-vol solvers, and a pointer to this spec.
- `pyproject.toml` version 0.6.0.

## 10. Roadmap: sub-projects 2 and 3

Recorded here so the order and the gates are settled; each gets its own spec and
plan after the previous gate passes.

### Sub-project 2 — one core (bitwise)

Scope: one core that assembles operators, marches value columns, applies boundaries,
and retains requested outputs; product adapters supply terminal payoffs, exercise
obstacles, lifecycle/regime/memory/stage columns, and ordered event
transformations. Concretely:

- A single θ-step kernel replaces the five copies (base sparse sweep, snowball
  two-surface sweep, snowball event-stats sweep, phoenix vector sweep, and the
  already-deleted `BackwardOperator`). It advances any number of columns and takes
  boundary values as inputs, so the value sweeps' "set before step" and the
  indicator sweep's "carry forward" conventions both fit. Backend stays fixed per
  family (sparse LU for the simple solvers, banded for the two-surface family) so
  the gate stays bitwise.
- One coefficient assembler for scalar and spatially varying volatility, preserving
  term curves, the trading clock, and zero-diffusion steps; replaces
  `_calculate_coefficients` / `_build_step_coefficients` /
  `_flat_exact_step_coefficients`.
- One preparation path for direct pricing, sessions, bumps, intraday, and life
  surfaces; frozen spatial geometry across bumps; distinct propagation and readout
  columns.
- Caches keyed on complete numerical identities; the session matrix pack becomes
  the kernel's cache handed across.
- Event probabilities and discounted cashflow components as indicator columns in
  the same march (folds `_compute_event_stats`).
- Public protocols — operator factory, time stepper, readout — with the built-ins
  behind them; extension outputs validated for shape, finiteness, and declared
  capabilities; failures propagate. Each component exposes a fingerprint (no
  registry, no versioning system). Components without a reproducible fingerprint
  price directly but cannot reuse banked certification or prepared artefacts. The
  built-in Crank-Nicolson stepper's configuration object carries the four scheme
  settings of section 4.3, which leave `PDEParams` and `GridConfig` at that point;
  a custom stepper receives its own configuration, not these.

Gate: byte-compare of the section 8.1 book plus anchors. Prerequisite: sub-project 1
merged.

### Sub-project 3 — numerical policy upgrade (certification-gated)

Each item is its own commit on the clean core so number movements are attributable;
then one golden regeneration and one full re-certification pass.

| Item | Effect |
|---|---|
| Two implicit half-steps on each of the first two intervals after every terminal or event discontinuity; coefficients and boundaries recomputed on the actual sub-intervals. | Every 1D price and Greek moves. |
| Coefficient-break times (curve knots, vol term pillars, trading-clock breaks) inserted as exact nodes. | Term-structure environments move; flat unchanged. |
| Fail with diagnostics when requested resolution exceeds `max_steps`/`max_points`; raise the default `max_steps` so the presets never bind below five years. | Cells scaled today (5y standard, 3y+ high) move. |
| `DCNPDEEngine` onto the common core (Dirichlet boundaries, shared damping). | DCN moves; re-certify against DCN MC. |
| Local-vol vanilla and barrier onto the shared concentrated grid. | Oracle-tested only. |
| KO-reset native PDE event statistics; the QUAD delegation and residual adjustment go. | Event stats move; PV unchanged. |

A `steps_per_year` rename was considered and dropped: with the day defined as the
environment's business day (decision 6), `steps_per_day` is the exact name, and a
per-year density would have to be re-expressed per desk convention anyway.

Gate: full certification per the release procedure.

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| A legacy-branch deletion in the 3400-line snowball solver changes the default path. | The byte gate covers snowball on flat and term environments, both KI types, knocked-in, NODAL, and BGK rows. |
| The 2D constant re-point misses a read. | Heston/SLV rows in the byte gate; grep for every removed field name must return zero hits outside the removal map. |
| External consumers (the OTC price adapter) pass a removed name. | They get a `ValidationError` with the replacement; the changelog table is the migration note. |
| Local-vol vanilla/barrier default resolution changes surprise a user. | Reported deltas against Black-Scholes and analytic barriers in the merge message; documented under "Changed". |
| The four scheme settings sit two-plus-two across `PDEParams` and `GridConfig` until sub-project 2. | Accepted: regrouping them now would change the `grid.*` fields the certificates record; sub-project 2 groups them into the built-in stepper's config. |
| Anchor tests fail because a study YAML names a removed field. | Verified: study YAMLs name only `accuracy`, `points`, `steps_per_day`, `cache_enabled`. |
| A 244-convention desk sees its PDE numbers move. | Intended: the fill now follows the desk's business days. Called out under "Changed" in the changelog with the ~3% fill figure; no golden or certificate is on a 244 environment. |
