# PDE Grid & Event Layer Redesign — Design Spec

**Date:** 2026-07-27 (rev 3 — post adversarial review rounds 1 & 2)
**Status:** Approved design, pre-implementation
**Background:** `quantark/asset/equity/engine/docs/pde_auto_grid_investigation.md` (the auto-grid
root-cause program). That program produced a *correct* grid/event stack; this program replaces it
with a *clean* one. Numerical semantics certified there are carried over verbatim; the code that
hosts them is rewritten.

---

## 1. Problem

The current grid construction stack is the accretion of successive fix rounds. Each fix added a
new path beside the legacy one instead of replacing it, because each fix had to prove it did not
change legacy behavior. The result:

- `time_grid.py` (384 lines): five overlapping builders (`build_uniform`, `build_graded`,
  `build_event_clustered`, `build_event_aligned`, `build_mandatory`) plus string-dispatch
  `build()`. Only `build_mandatory` semantics are current; the rest are legacy.
- `spatial_grid.py` (750 lines): three builders and two beta-search algorithms
  (`use_heuristic_beta` flag).
- `base_pde_solver.py`: ~300 lines of `_resolve_*` methods with interacting flags
  (`auto_grid` × `time_grid_type` string × legacy heuristic fallbacks).
- `snowball_pde_solver.py`: grid overrides whose correctness depends on two call sites reading
  the same predicate (`_ki_nodes_in_grid`), enforced by comments, not structure. The 2D path
  duplicates the 1D damping gating ("mirrors … exactly").
- `PDEParams`: ~20 grid-related knobs a new user must face.

## 2. Goals

1. **Understandable by a new user**: one obvious way to get a grid; a front-door param surface of
   1–2 knobs; grid logic readable in one package.
2. **Bump-stable greeks by construction**: the spatial grid used for bumped re-solves is the
   base-market grid, structurally — not via a knob.
3. **Cross-position spatial sharing**: positions on the same underlying can share the spatial
   layout, with per-product time layouts (`bind_shared`).
4. **Invariants by construction, not by comment**: grid geometry and event application consume
   one declaration; they cannot disagree.
5. **No legacy layer at the end**: old builders, flags, and modes are deleted, not deprecated.

### Non-goals (out of scope)

- The batched book-pricing function ("PDERun orchestrator"): explicitly rejected as a framework.
  Batching later arrives as a plain function consuming this layer. The only provisions made now
  are the pure-transform contract (§4.2) and `bind_shared`.
- **Operator/factorization reuse across positions and book partitioning** belong to that later
  batch program, not this rewrite — they have no consumer in this migration. This rewrite only
  guarantees the precondition: same `SpatialLayout` ⇒ same discretized operator for the same
  market inputs.
- MC and QUAD engines; the QUAD `ki_probability` definitional gap (separately tracked).
- GPU / Numba stepping.

## 3. Decision log (settled with the user)

| Decision | Choice |
|---|---|
| Scope | Grid **construction + event application** (full clean-room of both layers) |
| Compatibility | **Clean break**: old knobs deleted; goldens re-baselined vs MC/analytic anchors; next release 0.4.0 |
| Solver coverage | **All PDE solvers, phased** (full manifest in §6): autocallables → simple 1D → LV/2D |
| Param surface | **Tiered**: `accuracy` profile front door + expert `GridConfig` |
| Architecture | **A′**: declarative `GridRequest` + engine-owned `GridBinder`; shareable `SpatialLayout` per underlying; per-product `TimeLayout` + `EventSchedule`. No stepping orchestrator — solvers keep their certified loops |
| Batch performance | Deferred; later a plain multi-RHS function over this layer |
| Review gates | Codex adversarial review, 2 iterations per gate (spec / plan / code) |

## 4. Architecture

### 4.1 New package

```
quantark/asset/equity/engine/pde/grid/
├── __init__.py      # public API: GridRequest, GridConfig, MarketSnapshot, GridBinder,
│                    #             SpatialLayout, TimeLayout, Layout, EventSchedule
├── request.py       # GridRequest — frozen, hashable geometry declaration
├── config.py        # GridConfig + accuracy profiles (fast / standard / high)
├── binder.py        # GridBinder — engine-owned constructor + cache (bind / bind_shared /
│                    #             rebind_time)
├── space.py         # ONE spatial builder → SpatialLayout
├── time.py          # ONE time builder → TimeLayout (incl. damping schedules)
└── events.py        # EventSchedule + cell-average projection operators
```

`backward_operator.py` (the shared 1D operator assembly) is retained; it consumes
`SpatialLayout` instead of raw vectors. The variance-axis builder for 2D (concentrated-v,
degenerate v=0) stays in `quantark/volmodels/` (model-specific) but follows the same layout
idiom (`VarianceLayout`, configured by a `VarianceConfig` pinned at Phase 3).

### 4.2 Core contracts

Load-bearing separation: **geometry** (hashable, cacheable, freezable) vs **event semantics**
(per-solve, needs market data).

```python
@dataclass(frozen=True)
class GridRequest:                          # GEOMETRY ONLY — per product
    tau: float
    bound_anchors: tuple[float, ...]        # centers of the ±h auto-bounds envelope
                                            #   (spot, strike)
    critical_prices: tuple[float, ...]      # concentration targets: barriers, strike, spot
    hard_lower: float | None                # ABSORBING domain edges (continuous KO / touch);
    hard_upper: float | None                #   each side independently optional
    event_times: tuple[float, ...]          # ALL interior event-bearing dates (KO, coupon,
                                            #   discrete KI) — every one an exact node,
                                            #   indexed in step_of, and damped

@dataclass(frozen=True)
class MarketSnapshot:                       # the ONLY market inputs the builder sees
    spot: float
    sigma_ref: float                        # representative vol — see solver hook §4.6
    r_ref: float                            # term-average rate (bounds drift)
    q_ref: float                            # term-average carry

@dataclass(frozen=True, eq=False)           # eq=False: layouts compare by IDENTITY
class SpatialLayout:                        # per underlying — shareable, bump-frozen
    s: np.ndarray; x: np.ndarray; dx: np.ndarray   # ndarray.flags.writeable = False
    bounds: tuple[float, float]
    achieved_eps: float                     # worst-case spacing at critical prices (§4.4)

@dataclass(frozen=True, eq=False)
class TimeLayout:                           # per product
    t: np.ndarray; dt: np.ndarray           # writeable = False
    step_of: Mapping[float, int]            # MappingProxyType over a private copy;
                                            #   covers EVERY event_times entry
    event_damping_steps: frozenset[int]     # steps after event nodes → event_theta
    terminal_damping_steps: frozenset[int]  # first backward steps → theta = 1.0
    requested_steps: int                    # diagnostics (cap behavior observable)
    actual_steps: int
    cap_exceeded: bool                      # True iff actual_steps > max_steps
                                            #   (mandatory-overflow case only)

@dataclass(frozen=True, eq=False)
class Layout:
    spatial: SpatialLayout
    time: TimeLayout
    request: GridRequest                    # provenance — validated on external supply
    config_key: tuple                       # resolved GridConfig fingerprint

class GridBinder:                           # ENGINE-OWNED: one per engine instance,
    def __init__(self, config, *,           # constructed from PDEParams (accuracy/grid/
                 cache_enabled, cache_max_entries): ...   # cache settings)
    def bind(self, request, market) -> Layout
    def bind_shared(self, requests, market) -> list[Layout]
    def rebind_time(self, layout, new_request) -> Layout   # frozen spatial, new time (§4.8)
```

Contract details (each is a tier-1 test):

- `step_of` keys are the **verbatim floats** from the request's `event_times` — the builder
  places those exact values as nodes, so exact dict lookup is well-defined (no `is_close`
  searching at consumption time). There is no separate "monitor" collection: a date either
  bears an event (→ `event_times`) or it does not exist for the grid.
- **Immutability & equality**: `GridRequest` / `MarketSnapshot` / `GridConfig` compare and hash
  by value. Layout classes are declared `eq=False` — identity comparison, matching the cache
  contract (the binder returns the same object, observable via `is`). All arrays are stored
  with `writeable=False` and `step_of` wraps a private dict copy in `MappingProxyType`. Tier-1
  tests assert a consumer-side write attempt raises; `setflags(write=True)` re-enablement is
  acknowledged as circumventable (read-only is enforced, not tamper-proof).
- **Caching**: the binder memoizes `bind` results in an LRU keyed by
  `(request, market, config_key)`, bounded by `grid_cache_max_entries` (128), disabled by
  `cache_enabled=False`. Engine-owned, so cache scope and lifetime equal engine lifetime — this
  replaces `_grid_cache_key` / `_observation_cache_key`.
- **`bind_shared` semantics**: empty input → `ValidationError`; a single request is exactly
  equivalent to `bind`; returns complete per-product `Layout`s aligned to input order, all
  sharing one `SpatialLayout` **object** (identity-shared); value-identical requests share one
  `TimeLayout` object. One call takes one `MarketSnapshot` — same-underlying consistency is
  structural, not checked. Any request with a hard bound on either side → `ValidationError`
  (continuous-barrier products get private layouts via plain `bind`).
- **Bounds precedence** (highest first): product `hard_lower`/`hard_upper` (correctness) →
  expert `GridConfig.bounds` per side (conflict with a hard bound on the same side →
  `ValidationError`) → auto bounds (§4.4).

Event semantics live in a second, per-solve object with **four explicit stages** (matching the
certified endpoint treatment — interior `event_times` cover only `0 < t < tau`; the endpoints
have their own stages):

```python
class EventSchedule:
    def terminal(self, states) -> states
        # t = tau stage: applied ONCE, after terminal-payoff construction and before the
        # first backward step (today's terminal-KO handling).
    def apply(self, step: int, states) -> states
        # interior stage: applied when the loop lands on an event node (cell-average
        # projected transforms).
    def continuous(self, step: int, states) -> states
        # every-step stage: continuous-monitoring regime coupling (continuous KI, BGK
        # shifted-barrier KI: alive[s ≤ B] ← ki[s ≤ B], nodal, full domain — NOT a domain
        # edge). Identity for products without continuous monitoring.
    def valuation_readout(self, spot: float, states) -> float
        # t = 0 stage: pointwise inclusive trigger semantics at spot, smooth-branch
        # interpolation (today's certified t0 readout).
# states: Mapping[str, np.ndarray] — named blocks, S always axis 0
```

- **State blocks are named, S is always axis 0.** Each value in `states` is an array whose
  axis 0 is the S-grid; trailing axes (variance, …) broadcast untouched. A 1D snowball passes
  `{"alive": (n_s,), "ki": (n_s,)}`; a Heston snowball `{"alive": (n_s, n_v), "ki": (n_s, n_v)}`;
  Phoenix adds its coupon-memory surfaces. No transform ever needs to know whether it runs
  under 1D, LV, or 2D.
- **Semantic split of "continuous barrier"**: *absorbing* continuous KO / one-touch barriers are
  domain edges (`hard_lower`/`hard_upper` — Dirichlet at the barrier, as today's 1D solvers).
  *Continuous KI* is per-step regime coupling on the full domain via `continuous()` (as today).
  The 2D barrier solvers' per-step-injection KO treatment is **preserved as-is** and expressed
  through `continuous()` at Phase 3 — this spec does not force 2D KO onto domain truncation.
- Ordering per backward step: PDE step → `apply` (if the landed node is an event node) →
  `continuous`. At maturity: payoff → `terminal`. At valuation: `valuation_readout` (never a
  cell average — exact inclusive triggers at spot).
- Cell-average projection is precomputed at schedule-build time as a sparse operator on the
  S-axis; interior events are affine maps `P @ V + c` per named block; inter-block coupling is
  a pure function of the block dict. **Nodal mode is deleted** for interior discrete events;
  `continuous()` transforms are nodal by design (unchanged from today).
- The projection *math* is carried verbatim from the certified implementation
  (`event_projection.py`, incl. straddling-cell complete-function averaging, piecewise
  coincident coupon+KO cells).

### 4.3 Time builder — the one algorithm

Current `build_mandatory` semantics, kept verbatim and now pinned exactly:

- every `event_times` entry becomes a node **exactly** (all are inviolable);
- fill per interval: `interval_days = max(1.0, interval_years × day_count)`;
  `fill = max(1, round(interval_days × steps_per_day))` with NumPy half-to-even rounding —
  **no per-interval floor** beyond the single-step minimum (the ≥10-step floor caused the
  historical ~10× grid inflation);
- **cap policy** (`max_steps` counts total steps, `len(dt)`): with `n_int` = number of
  intervals (interior event nodes + 1), each interval's baseline of 1 step is inviolable. If
  `n_int > max_steps`: keep all nodes, `fill = 1` everywhere, `cap_exceeded = True`, log.
  Otherwise scale **only the extras** into the remaining budget: with
  `extras_i = fill_i − 1` and `budget = max_steps − n_int`,
  `fill_i = 1 + floor(extras_i × budget / Σ extras)` when `Σ extras > budget` — the total is
  then ≤ `max_steps` **by construction** (baselines + scaled extras cannot exceed baselines +
  budget), deterministic, no remainder redistribution. `cap_exceeded` is True **iff**
  `actual_steps > max_steps` (i.e. only the mandatory-overflow case). Diagnostics
  (`requested_steps`, `actual_steps`, `cap_exceeded`) are `TimeLayout` fields, not log lines;
- **damping schedules** (two, distinct): `terminal_damping_steps` = the first
  `GridConfig.terminal_damping_steps` backward steps from maturity; `event_damping_steps` = the
  `GridConfig.event_damping_steps` steps immediately after (backward-time) **each event node**
  — KO, coupon, and discrete-KI dates alike, preserving today's union damping [§11.2].
  Computed here once, consumed identically by the 1D θ-loop and the 2D ADI stepper;
- no events ⇒ the same algorithm degenerates to uniform fill sized by `steps_per_day`.

**Deleted concepts:** graded (power-law) time grids, `event_clustered`, `event_aligned`,
string-dispatch `TimeGrid.build`, `time_grid_type`, `grade_exponent`, per-interval minimums,
`time_steps` as a target-total-step count (resolution is `steps_per_day`, nothing else), and
the align-vs-monitor field split (all event-bearing dates are one collection).

### 4.4 Spatial builder — the one algorithm

Multi-point sinh (Tavella–Randall ODE) as the single algorithm. The auto-bounds formula is
pinned here. **This is an intentional, recertified numerical change** — it replaces (not
reproduces) the current `calculate_auto_bounds` heuristics (0.5×/2× strike, 0.8×/1.2× barrier
expansions); tier-2 anchors certify the new formula:

- work in log-space; `h = num_std·σ_ref·√τ + |r_ref − q_ref − ½σ_ref²|·τ`
  (τ = the request's `tau`; for `bind_shared`, the max over requests);
- base interval = envelope of `[ln(a) − h, ln(a) + h]` over the request's `bound_anchors`
  (spot and strike — a **separate field** from `critical_prices`, so the builder needs no
  classification heuristics);
- widen so **every** critical price sits strictly interior with a margin of
  `5 × ln(1 + eps_crit)` (five cells at target resolution);
- degenerate floor: `h ≥ ln(1.10)` (±10 % domain) — covers near-expiry and near-zero vol;
- hard bounds override their side verbatim (they are the domain edge — the absorbing barrier
  condition applies exactly there); critical prices equal to a hard edge are domain-boundary
  prices, not interior targets (see coverage validation, §4.6);
- concentration at the interior `critical_prices` targeting `eps_crit` relative spacing.
  **One** beta search, defined as an inequality on the worst critical point: local spacing at
  a critical price = min of its two adjacent `dx`; `achieved_eps` = the **max** of that local
  spacing over all interior critical prices (worst case). If the uniform grid already
  satisfies `achieved_eps ≤ ln(1 + eps_crit)`, clamp to (near-)uniform (`beta_hi`) — no
  bracket search. Otherwise bisect beta on `[1e-12, 1e3·(x_max − x_min)]`; if the target is
  unreachable at maximum concentration, take `beta_lo` (best achievable). The
  `use_heuristic_beta` variant dies;
- zero interior critical points ⇒ uniform-in-log; uniform stops being a separate mode;
- **infeasibility policy**: a critical price outside `[bounds]` (possible only under hard/expert
  bounds) is excluded from concentration and logged; `achieved_eps > 2 × eps_crit` logs a
  warning (accuracy degradation, not an error); explicit `points > max_points` →
  `ValidationError` at config construction;
- **barrier snapping is dropped.** Cell-average projection keeps second-order accuracy wherever
  a barrier falls relative to nodes, so node-on-barrier is no longer required. Not snapping is
  what makes layouts shareable across products and stable across bumps.

### 4.5 Scheme & damping contract

The full damping semantics carried from today, made explicit (this replaces four current knobs —
`use_rannacher`, `rannacher_steps`, `rannacher_at_events`, `event_rannacher_steps` — with two
counts and one theta):

- per-step θ: `theta = 1.0` at steps in `terminal_damping_steps`; `theta =
  PDEParams.event_theta` (default 1.0) at steps in `event_damping_steps`; `PDEParams.theta`
  (default 0.5, Crank–Nicolson) elsewhere. On overlap, terminal damping wins (θ = 1.0).
- counts live on `GridConfig`: `terminal_damping_steps = 1` (today's `rannacher_steps`),
  `event_damping_steps = 2` (today's field-tested minimum, Pooley–Vetzal–Forsyth). A count of 0
  disables that damping kind (replaces the `use_rannacher` / `rannacher_at_events` booleans).
- the 2D ADI stepper consumes the same two frozensets to select damped (Douglas, θ-style) vs
  undamped (Craig–Sneyd corrected) steps — one derivation, two consumers.

### 4.6 Solver consumption

A solver's entire grid involvement is three small hooks:

```python
class SnowballPDESolver(BasePDESolver):
    def grid_request(self, product, market) -> GridRequest:
        # ~30 lines. ALL gating decided here, once:
        #  - BGK active → shift KI barrier (critical price), drop discrete-KI event_times
        #    (KI coupling moves to the continuous() stage at the shifted barrier)
        #  - already knocked in → drop KI event_times + KI critical price
        #  - KO/coupon/discrete-KI dates → event_times
    def representative_vol(self, product, pricing_env) -> float:
        # sigma_ref for MarketSnapshot. Constant-vol solvers: strike-selected vol
        # (today's behavior). LV/Heston/SLV solvers: the representative measure
        # each already uses for bounds today; exact 2D protocol pinned at Phase 3.
    def event_schedule(self, product, env, layout) -> EventSchedule:
        # terminal / interior / continuous / valuation_readout stages (§4.2)
```

The base class drives: `layout = self._binder.bind(self.grid_request(product, market), market)`
→ `self._solve(product, env, layout, schedule)`. Time-stepping loops keep their current
certified shape but read `layout.time.*damping_steps` and call the schedule stages instead of
consulting private `_ko_observation_indices` dicts.

**Externally supplied layouts are validated, not trusted.** `solve(..., layout=...)`
(frozen-bump and shared-book cases) recomputes the product's `GridRequest` and requires:

- *alignment-critical fields exactly equal*: `tau`, `event_times`, `hard_lower`, `hard_upper`
  — mismatch → `ValidationError` (a layout from another product or another schedule cannot
  silently misprice);
- *coverage*: current spot and all freshly-derived **interior** critical prices strictly inside
  `layout.spatial.bounds`; a critical price `is_close` to a hard edge is exempt (it IS the
  boundary — continuous-barrier products would otherwise always fail) — violation →
  `NumericalError`;
- *concentration drift allowed*: `critical_prices` / `bound_anchors` may differ (spot moved
  under a bump) — this affects accuracy constants only, never alignment or correctness.

Per-solver notes:

- **KO-reset snowball**: reset dates are scheduled → ordinary `event_times`; reset semantics
  are interior transforms.
- **DCN**: coupon schedule → `event_times`; coupon decisions → interior transforms.
- **American**: empty `event_times`; the per-step obstacle projection is a stepping concern,
  unchanged.
- **Barrier / one-touch (continuous observation)**: `hard_lower` and/or `hard_upper` at the
  barrier(s) — single-barrier products set exactly one side. Discretely observed variants use
  `event_times` + interior transforms.
- **Continuous-KI autocallables**: KI coupling via the `continuous()` stage (full domain),
  KI barrier as a critical price — never a hard bound.
- **LV / 2D vol solvers**: bind the same S-axis `SpatialLayout` (shareable with 1D) and
  `TimeLayout`; 2D adds `VarianceLayout` from `volmodels`; the ADI stepper consumes the same
  damping frozensets and the same named-block stages (arrays are `(n_s, n_v)`); 2D KO keeps
  its per-step injection via `continuous()`.

Public API (`engine.price(product, env)`, `calculate_greeks`) is unchanged.

### 4.7 Parameter surface

```python
PDEParams(accuracy="standard")                      # the whole front door

PDEParams(accuracy="high", grid=GridConfig(         # expert tier — every field Optional;
    points=800, steps_per_day=8.0,                  # None (default) = inherit from profile,
    eps_crit=0.002, num_std=4.0,                    # explicit value = override that field
    bounds=(50.0, None),                            # per-side hard override
    max_points=2000, max_steps=5000,
    day_count=252,
    terminal_damping_steps=1, event_damping_steps=2,
))
```

- `accuracy` ∈ {"fast", "standard", "high"} selects a `GridConfig` preset. Indicative presets —
  fast ≈ (200 pts, 4/day), standard ≈ (400, 4/day), high ≈ (800, 8/day) — are calibrated
  against the tier-2 anchor set during Phase 1. `steps_per_day` never goes below 4 in any
  preset (continuously monitored KI needs the resolution — current convergence-gate finding,
  preserved uniformly so no preset is product-dependent; "fast" economizes on the spatial
  axis).
- **Complete disposition of every current PDE param** (nothing left implicit):

| Current param | Disposition |
|---|---|
| `grid_size` | → `GridConfig.points` |
| `time_steps` | **deleted** (resolution is `steps_per_day`; no target-total concept) |
| `adaptive_grid` | **deleted** (concentration engages iff critical prices exist) |
| `auto_grid` | **deleted** (the builder is always feature-aware; explicit bounds/points are overrides, not modes) |
| `s_min` / `s_max` | → `GridConfig.bounds` (per-side, optional) |
| `time_grid_type` / `grade_exponent` | **deleted** |
| `event_steps_per_day` | → `GridConfig.steps_per_day` (float) |
| `event_min_steps_per_interval` | **deleted** (the inflation bug) |
| `max_time_steps` / `max_grid_size` | → `GridConfig.max_steps` / `max_points` |
| `log_dx_target` | **deleted** (superseded by `eps_crit` + profiles) |
| `include_spot_in_critical_points` | **deleted** (spot is always a critical price and a bound anchor) |
| `frozen_critical_points` | **deleted** (frozen `Layout` objects replace the knob, §4.8) |
| `barrier_refine_log_width` / `_levels` | **deleted** |
| `barrier_domain_expand` | **deleted** (subsumed by the pinned margin rule §4.4) |
| `event_projection` | **deleted** (cell-average is the only interior mode) |
| `use_rannacher` / `rannacher_steps` | → `GridConfig.terminal_damping_steps` (0 disables) |
| `rannacher_at_events` / `event_rannacher_steps` | → `GridConfig.event_damping_steps` (0 disables) |
| `event_theta` | **survives** on `PDEParams` (scheme knob, §4.5) |
| `theta` | **survives** on `PDEParams` |
| `boundary_mode` | **survives** on `PDEParams` ("asymptotic" default, unchanged) |
| `ki_monitoring_mode` | **survives** on `PDEParams` — product-semantics knob, not a grid knob. `EXACT_DISCRETE` default; `BGK_APPROXIMATION` opt-in engages only for discretely monitored KI (European-KI / continuous / no-KI / already-knocked-in are inert with a logged note) — current applicability rule preserved verbatim |
| `bus_days_in_year` (inherited from `EngineParams`) | **survives on `EngineParams`** for non-grid uses; the PDE grid layer stops reading it — `GridConfig.day_count` is the grid-side replacement |
| `cache_enabled` / `cache_strategy` / `grid_cache_max_entries` | **survive** (feed the `GridBinder` cache, §4.2) |
| `use_banded_solver` / `banded_cache_max_entries` | **survive** (solver internals, untouched) |
| Phase-3 solver-level discretization args (`n_x`, `n_t`, `grid_style`, `grid_focus`, `pin_critical_spots` on LV/Heston/SLV engines) | S-axis and time controls **replaced by `GridConfig`** at Phase 3 (one configuration language everywhere) |
| Phase-3 variance-axis args (`n_v`, `v_grid_power`, v-concentration controls) | → `VarianceConfig` in `volmodels`, pinned at Phase 3 (variance axis is model-specific by design) |

- Release becomes **0.4.0**; `otc-price-adapter` updates its pin and any grid params it passes.

### 4.8 Greeks and sharing behavior

- **Frozen-by-default bumps.** `GreeksCalculator` binds once at base market, then per bump type:
  - *spot / vol / rate / div bumps*: reuse the **whole** `Layout` (geometry is unchanged —
    `tau` and schedules don't move); coverage validated per §4.6.
  - *theta (calendar) bumps*: `tau` and relative event times change → `binder.rebind_time(
    layout, rolled_request)` rebuilds the `TimeLayout` deterministically while the
    `SpatialLayout` is retained by identity. `EventSchedule` is per-solve and always rebuilt.
  - A bumped input outside the frozen spatial bounds → `NumericalError` — never a silent
    rebuild.
- **Spot greeks** read delta/gamma off the single solved surface (the PDE solution does not
  depend on spot; spot only selects the readout point) — existing engine-mode greeks, now the
  documented default for spot.
- **Sharing:** `bind_shared` for same-underlying books of discrete-observation products
  (semantics in §4.2). Continuous-barrier products take private layouts.

### 4.9 Validation & error handling

Existing exception hierarchy; `quantark.util.numerical` throughout.

- `GridRequest.__post_init__`: `tau > 0`; prices positive; `event_times` within the open
  interval `(0, tau)` after `is_close` de-duplication (endpoint semantics live in the
  `terminal` / `valuation_readout` stages, §4.2 — never as interior events); hard bounds
  positive, `hard_lower < hard_upper` when both set → else `ValidationError`.
- `GridConfig` validation at construction: explicit `points > max_points`, negative counts,
  per-side `bounds` conflicts → `ValidationError`.
- `bind`: infeasible domain (e.g. hard bounds excluding spot) → `PricingError`;
  event-count-exceeds-cap → proceed + diagnostics fields + log (never drop a node).
- The `_aligned_time_index` runtime search-and-assert disappears: `step_of` is constructed by
  the builder, not re-derived by searching `t_vec`.
- External-layout mismatch → `ValidationError`; frozen-layout coverage violation →
  `NumericalError` (§4.6, §4.8).

## 5. Testing strategy (full rewrite)

Three tiers; goldens re-baselined, never compared against the old PDE stack.

**Tier 1 — builder unit tests** (pure, fast, no solvers): exact event-node alignment
(verbatim-float `step_of` lookup, every `event_times` entry indexed); fill density equals the
§4.3 formula in the uncapped case, with explicit fixtures for sub-day maturity, a single
interior event, and event-count > cap; **cap enforcement** — extras-scaling keeps
`actual_steps ≤ max_steps` including the adversarial many-small-intervals case (99 × fill-1
plus 1 × fill-901 under cap 100), and `cap_exceeded` is True only on mandatory overflow;
damping frozensets over event nodes incl. the overlap rule; concentration spacing
(`achieved_eps` worst-case definition) and the uniform-clamp / no-bracket / unreachable-target
policies; uniform degeneration; per-side hard bounds incl. critical-price-on-edge exemption;
the pinned auto-bounds formula (`bound_anchors` envelope, margin, `ln(1.10)` floor);
projection operators satisfy **constant preservation `P @ 1 = 1`** plus the certified
piecewise exactness cases, and converge at second order on synthetic discontinuities; the four
schedule stages exist and order correctly (terminal → stepping/apply/continuous → readout);
`bind_shared` degenerate cases (empty, single, duplicates, hard-bounds rejection,
spatial-identity sharing); `rebind_time` retains the spatial object by identity; request
hashing; layout `eq=False` identity semantics; immutability (`writeable=False`, proxy copy);
cache identity-reuse and `cache_enabled=False`.

**Tier 2 — certification vs anchors** (replaces PDE-vs-old-PDE goldens), with pinned
provisional acceptance criteria — tightened during Phase-1 calibration, never loosened without
user sign-off:

- European PDE vs Black–Scholes closed form: relative PV error < 5e-4 (standard profile),
  < 1e-4 (high); |delta error| < 5e-4; gamma relative error < 1e-2.
- Continuous-barrier PDE (hard bounds) vs closed form: relative PV error < 1e-3 (standard).
- Autocallables (snowball / phoenix / KO-reset / DCN) vs QMC MC (fixed seed, ≥ 2^18 paths):
  agreement within 3× MC standard error; vs QUAD: relative PV difference < 5e-4.
- Grid-refinement convergence: observed spatial order ≥ 1.7 on smooth-payoff anchors
  (points × 2, steps × 2 refinement pairs).
- Greek smoothness on the frozen layout: across a ±2 % spot bump ladder, delta is monotone
  where the payoff implies it and gamma shows no sign-flipping grid noise (replaces
  `test_bump_grid_steadiness` / `test_pde_fixed_bump_grid`).
- 2D Heston/SLV convergence re-baselined with the same structure.

**Tier 3 — new goldens**, frozen only after tier 2 passes, compared with the tolerance-based
`test/golden_compare.py` (cross-arch safe; never bitwise float `==` across architectures).

Test files to rewrite or retire (enumerated fully in the implementation plan):
`test_pde_event_projection.py`, `test_pde_fixed_bump_grid.py`, `test_bump_grid_steadiness.py`,
`test_pde_grid_convergence_gate.py`, plus the grid-touching portions of the per-solver PDE
tests.

## 6. Migration phases & consumer manifest

Gate for every phase: full suite green + tier-2 anchors green. Old code for not-yet-migrated
solvers keeps working within each phase; final deletion happens only when the last consumer
moves.

| Phase | Consumers migrated |
|---|---|
| 0 | `grid/` package + tier-1 tests; purely additive, no consumers |
| 1 | `SnowballPDESolver`, `PhoenixPDESolver`, `KOResetSnowballPDESolver`, `DCNPDEEngine` (+ `BasePDESolver` plumbing); accuracy presets calibrated; autocallable goldens re-baselined |
| 2 | `EuropeanPDESolver`, `AmericanPDESolver`, `BarrierPDESolver`, `DoubleBarrierPDESolver`, `OneTouchPDESolver`, `DoubleOneTouchPDESolver` |
| 3 | LV 1D: `LocalVolPDESolver`, `LocalVolBarrierPDESolver`, `LocalVolSnowballPDESolver`, `LocalVolPhoenixPDESolver`, `LocalVolDCNPDEEngine`; 2D: `HestonPDESolver`, `HestonSLVPDESolver`, `HestonBarrierPDESolver`, `HestonSLVBarrierPDESolver`, `HestonSnowballPDESolver`, `HestonSLVSnowballPDESolver`, `HestonPhoenixPDESolver`, `HestonSLVPhoenixPDESolver`, `HestonDCNPDESolver` + `adi_core` damping/2D-stage consumption; `VarianceLayout` / `VarianceConfig` + 2D `representative_vol` protocol pinned; solver-level `n_x`/`n_t`/`grid_style`/`grid_focus`/`pin_critical_spots` retired into `GridConfig` |
| 4 | Delete `time_grid.py`, `spatial_grid.py`, `event_projection.py`, legacy `PDEParams` knobs per §4.7 table; `backward_operator.py` consumes `SpatialLayout`; re-verify execution-framework seams (`pde_execution_adapters.py`, `pde_session_prep.py` — prepared grid artifacts hold `Layout` objects); docs + 0.4.0 release notes; update `asset/equity/CLAUDE.md` |

Every name above is an exported symbol of `quantark.asset.equity.engine.pde` (or the DCN module
pair); the table is the exhaustive consumer inventory — a Phase-4 checklist item greps for any
remaining importer of the deleted modules.

Later, separate program: the batch book-pricing function (shared factorization + multi-RHS
banded solve over this layer) and any book-partitioning helpers.

## 6b. Post-implementation amendments (recorded during Phase 3–4)

- **2D time axis deferred:** the 2D solvers take their S-axis from the shared
  spatial builder (`adi_core x_nodes`) but keep their certified uniform ADI
  time march; event-aligned 2D time is a follow-up.
- **Standalone vol engines deferred:** `LocalVolPDESolver`, `HestonPDESolver`,
  `HestonSLVPDESolver`, the LV/Heston/SLV barrier engines and
  `HestonDCNPDESolver` keep bespoke kernels (they import no legacy grid
  module, so deletion was not blocked). Consequently `grid_size`,
  `time_steps`, `s_min`, `s_max` SURVIVE on `PDEParams` as their inputs, and
  `event_projection` + the Rannacher scheme knobs (`use_rannacher`,
  `rannacher_steps`, `rannacher_at_events`, `event_rannacher_steps`,
  `event_theta`) survive for the live characterization paths — amending the
  §4.7 "deleted" rows for those entries. The 11 remaining table rows were
  deleted as specified.
- **§4.4 amendment:** the spatial builder carries the certified
  tail-coarseness guard (max/min dx ratio ≤ 100, ported from
  `check_grid_quality`) with final say over the eps_crit refinement — a
  pointwise spacing target met by a degenerate grid is treated as
  unreachable.
- **§4.2 amendment:** `TimeLayout.step_at(t)` provides exact-float lookup
  with an `is_close` fallback for cross-schedule float drift (two schedules
  resolving the same date to floats an ULP apart); construction-time dedup
  guarantees at most one match.
- **§5 amendment:** the continuous-KI PDE-vs-QUAD anchor is banded at
  1.5e-3 notional-relative (pre-existing cross-family treatment divergence,
  reproduced on the pre-rewrite stack); 5e-4 applies to discrete-KI rows.

### Stage-6 review amendments (Codex code gate, 2026-07-27)

- **§4.7 amendment — surviving-knob semantics:** grid-layer solvers REJECT
  non-default `grid_size`/`time_steps`/`s_min`/`s_max` (ValidationError
  naming the `accuracy`/`GridConfig` replacement) instead of silently
  ignoring them; the knobs remain live for the standalone vol-model solvers
  only. `event_steps_per_day`, `max_time_steps`, `max_grid_size` are deleted
  outright (no remaining consumers — `GridConfig.steps_per_day`/`max_steps`/
  `max_points` own those roles). `make_pde_params` profiles now emit
  `accuracy`/`grid=GridConfig(...)`.
- **§4.8 amendment — bump-clone construction:** `create_bump_context` clones
  the live engine via `deepcopy` (transient solve state stripped), so
  subclasses with richer constructors (Heston `model_params`, SLV leverage
  surfaces, ADI dimensions, prebuilt LV surfaces) keep full configuration;
  `PDEEngine.create_bump_context` seeds its facade clone's dispatch cache
  with the fixed solver instance (the freeze lives on the instance, not in
  params).
- **§4.8 amendment — frozen-layout reuse guards:** reuse fails closed when
  the fresh request's hard (absorbing-barrier) bounds differ from the frozen
  request's, and the calendar-roll `rebind_time` path coverage-validates the
  rebound layout (`resolve_bound_layout` is the single shared semantics for
  BasePDESolver and GridLayerMixin).

## 7. Risks & mitigations

- **Re-certification risk** (numbers move under the clean break — including the intentionally
  changed auto-bounds formula): mitigated by anchor-first testing (tier 2) before any golden is
  frozen, and by carrying the certified projection math verbatim.
- **Two-surface coupling regressions** (snowball KI): the block transforms are ported as pure
  functions with dedicated unit tests against the current implementation's outputs *before* the
  old code is deleted (scaffolding oracle, removed at Phase 4).
- **2D divergence**: shared damping frozensets remove the mirrored-gating class of bug; ADI
  internals untouched; 2D KO per-step injection preserved rather than redesigned.
- **Downstream breakage** (`otc-price-adapter`, execution framework): version-gated at 0.4.0;
  execution seams re-verified in Phase 4; adapter updates its pin on its own schedule (0.3.x
  remains on PyPI).
- **Dropped barrier snapping** shifts prices slightly relative to snapped grids: intended
  accuracy model (cell-average, second order regardless of node placement); tier-2 convergence
  tests demonstrate order, and profiles are calibrated to match or beat the current default's
  tier-2 anchor performance.
