# PDE Grid & Event Layer Redesign — Design Spec

**Date:** 2026-07-27
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
2. **Bump-stable greeks by construction**: the grid used for bumped re-solves is the base-market
   grid, structurally — not via a knob.
3. **Cross-position spatial sharing**: positions on the same underlying can share the spatial
   layout (and therefore operators/factorizations), with per-product time layouts.
4. **Invariants by construction, not by comment**: grid geometry and event application consume
   one declaration; they cannot disagree.
5. **No legacy layer at the end**: old builders, flags, and modes are deleted, not deprecated.

### Non-goals (out of scope)

- The batched book-pricing function ("PDERun orchestrator"): explicitly rejected as a framework.
  Batching later arrives as a plain function consuming this layer. The only provision made now is
  the pure-transform contract (§6).
- MC and QUAD engines; the QUAD `ki_probability` definitional gap (separately tracked).
- GPU / Numba stepping.

## 3. Decision log (settled with the user)

| Decision | Choice |
|---|---|
| Scope | Grid **construction + event application** (full clean-room of both layers) |
| Compatibility | **Clean break**: old knobs deleted; goldens re-baselined vs MC/analytic anchors; next release 0.4.0 |
| Solver coverage | **All PDE solvers, phased**: autocallables → simple 1D → 2D ADI consumers |
| Param surface | **Tiered**: `accuracy` profile front door + expert `GridConfig` |
| Architecture | **A′**: declarative `GridRequest` + single builder; shareable `SpatialLayout` per underlying; per-product `TimeLayout` + `EventSchedule`. No stepping orchestrator — solvers keep their certified loops |
| Batch performance | Deferred; achieved later via shared factorization + multi-RHS banded solve as a ~100-line function over this layer |

## 4. Architecture

### 4.1 New package

```
quantark/asset/equity/engine/pde/grid/
├── __init__.py      # public API: GridRequest, GridConfig, bind, bind_shared,
│                    #             SpatialLayout, TimeLayout, Layout, EventSchedule
├── request.py       # GridRequest — frozen, hashable geometry declaration
├── config.py        # GridConfig + accuracy profiles (fast / standard / high)
├── space.py         # ONE spatial builder → SpatialLayout
├── time.py          # ONE time builder → TimeLayout (incl. damping schedule)
└── events.py        # EventSchedule + cell-average projection operators
```

The variance-axis builder for 2D (concentrated-v, degenerate v=0) stays in
`quantark/volmodels/` (model-specific) but follows the same layout idiom (`VarianceLayout`).

### 4.2 Core contracts

Load-bearing separation: **geometry** (hashable, cacheable, freezable) vs **event semantics**
(per-solve, needs market data).

```python
@dataclass(frozen=True)
class GridRequest:                          # GEOMETRY ONLY — per product
    tau: float
    critical_prices: tuple[float, ...]      # concentration targets: spot, strike, barriers
    hard_bounds: tuple[float, float] | None # continuous-KO domain truncation; None = auto
    mandatory_times: tuple[float, ...]      # KO/coupon dates — become nodes exactly
    monitor_times: tuple[float, ...]        # daily-KI dates — drive resolution only

@dataclass(frozen=True)
class SpatialLayout:                        # per underlying — shareable, bump-frozen
    s: np.ndarray; x: np.ndarray; dx: np.ndarray
    bounds: tuple[float, float]

@dataclass(frozen=True)
class TimeLayout:                           # per product
    t: np.ndarray; dt: np.ndarray
    step_of: Mapping[float, int]            # mandatory time → exact node index
    damping_steps: frozenset[int]           # event-adjacent steps needing damped scheme
    # step_of keys are the verbatim floats from the request's mandatory_times —
    # the builder places those exact values as nodes, so exact dict lookup is
    # well-defined (no is_close searching at consumption time).

@dataclass(frozen=True)
class Layout:
    spatial: SpatialLayout
    time: TimeLayout

def bind(request, market, config) -> Layout
def bind_shared(requests, market, config) -> tuple[SpatialLayout, list[TimeLayout]]
```

- `market` is a small frozen snapshot `(spot, sigma, r, q)` resolved exactly as solvers resolve
  scalar inputs today (strike-selected vol).
- `bind_shared` unions critical prices and takes the widest bounds (bounds width uses the max
  `tau` across requests). Any request with `hard_bounds` in a shared bind raises
  `ValidationError`; a `partition_shareable(requests)` helper splits a book into shareable and
  private groups first.
- `bind` results are memoized on `(request, market, config)` — this replaces
  `_grid_cache_key` / `_observation_cache_key`.

Event semantics live in a second, per-solve object:

```python
class EventSchedule:
    # built by the solver per solve: {step_index: transform}
    # transform: PURE (n_s, k_surfaces) -> (n_s, k_surfaces); no solver instance state
    def apply(self, step: int, surfaces: np.ndarray) -> np.ndarray
```

Cell-average projection is precomputed at schedule-build time as a sparse **linear operator on
the S-axis**. Applying an event is `P @ V` (affine `P @ V + c` for events with cash legs), which
broadcasts unchanged over 1D vectors, two-surface stacks `(n_s, 2)`, and 2D `(n_s, n_v)`
surfaces. **Nodal mode is deleted** — cell-average is the only event representation. The
projection *math* is carried verbatim from the certified implementation
(`event_projection.py`, incl. straddling-cell complete-function averaging, piecewise coincident
coupon+KO cells, and the exact inclusive t=0 pointwise readout); only its packaging changes.

### 4.3 Time builder — the one algorithm

Current `build_mandatory` semantics, kept verbatim (this is the certified fix):

- every `mandatory_times` and `monitor_times` entry becomes a node **exactly**;
- fill between adjacent nodes = `max(1, round(interval_days × steps_per_day))`; **no
  per-interval floor** (the floor caused the historical ~10× grid inflation);
- `max_steps` caps fill only; mandatory nodes are inviolable — if mandatory count alone exceeds
  the cap, the cap is exceeded and logged; a node is never moved or dropped;
- `damping_steps` = the `GridConfig.event_damping_steps` steps following each mandatory node —
  computed here once, consumed identically by the 1D θ-loop and the 2D ADI stepper (removes the
  hand-mirrored gating);
- no events ⇒ the same algorithm degenerates to uniform fill sized by `steps_per_day`.

**Deleted concepts:** graded (power-law) time grids, `event_clustered`, `event_aligned`,
string-dispatch `TimeGrid.build`, `time_grid_type`, `grade_exponent`, per-interval minimums.

### 4.4 Spatial builder — the one algorithm

Multi-point sinh (Tavella–Randall ODE) as the single algorithm:

- bounds: `hard_bounds` if declared, else ±`num_std`·σ√τ around spot/strike, widened so every
  critical price sits strictly interior with margin (the exact margin rule is pinned in the
  implementation plan and validated against the tier-2 anchor set — a deliberate deferral, since
  the right constant is an empirical accuracy/width trade-off);
- concentration at `critical_prices` targeting `eps_crit` relative spacing; **one** beta search
  (bisection); the `use_heuristic_beta` variant dies;
- zero critical points ⇒ degenerates to uniform-in-log; uniform stops being a separate mode;
- **barrier snapping is dropped.** Cell-average projection keeps second-order accuracy wherever
  a barrier falls relative to nodes, so node-on-barrier is no longer required. Not snapping is
  what makes layouts shareable across products and stable across bumps. (Continuous-observation
  KO products still get exact barrier boundaries — via `hard_bounds`, where the barrier IS the
  domain edge, not a snapped interior node.)

### 4.5 Solver consumption

A solver's entire grid involvement is two hooks:

```python
class SnowballPDESolver(BasePDESolver):
    def grid_request(self, product, market) -> GridRequest:
        # ~30 lines. ALL gating decided here, once:
        #  - BGK active → shift KI barrier (critical price), drop monitor_times
        #  - already knocked in → drop KI monitor times + KI critical price
        #  - KO/coupon dates → mandatory_times
    def event_schedule(self, product, env, layout) -> EventSchedule:
        # KO redemption, coupon payments, KI surface-coupling transforms
        # (two-surface: V0[s ≤ KI] ← V1[s ≤ KI] as a pure block transform)
```

The base class drives: `layout = bind(self.grid_request(product, market), market, config)` →
`self._solve(product, env, layout, schedule)`. Time-stepping loops keep their current certified
shape but read `layout.time.damping_steps` and call `schedule.apply(k, V)` instead of consulting
private `_ko_observation_indices` dicts. `solve(..., layout=...)` accepts an externally supplied
layout (frozen-bump and shared-book cases).

Per-solver notes:

- **KO-reset snowball**: reset dates are scheduled, so they are ordinary `mandatory_times`;
  reset semantics are `EventSchedule` transforms.
- **DCN**: coupon schedule → `mandatory_times`; coupon decisions → transforms.
- **American**: empty `mandatory_times`/`monitor_times`; the per-step obstacle projection is a
  stepping concern, unchanged.
- **Barrier / one-touch (continuous observation)**: `hard_bounds` at the barrier(s); no discrete
  events. Discretely observed variants use `mandatory_times` + transforms like autocallables.
- **2D vol solvers (Heston/SLV)**: bind the same S-axis `SpatialLayout` (shareable with 1D) and
  `TimeLayout`; add `VarianceLayout` from `volmodels`; the ADI stepper consumes the same
  `damping_steps` and applies the same `P @ V` transforms across variance columns.

Public API (`engine.price(product, env)`, `calculate_greeks`) is unchanged.

### 4.6 Parameter surface

```python
PDEParams(accuracy="standard")                      # the whole front door

PDEParams(accuracy="high", grid=GridConfig(         # expert tier
    points=800, steps_per_day=8.0,
    eps_crit=0.002, num_std=4.0,
    bounds=(50.0, 200.0),                            # optional hard override
    max_points=2000, max_steps=5000,
    day_count=252,                                   # business-day convention
    event_damping_steps=2,                           # damped steps after each event node
))
```

- `accuracy` ∈ {"fast", "standard", "high"} selects a `GridConfig` preset. Indicative presets —
  fast ≈ (200 pts, 2/day), standard ≈ (400, 4), high ≈ (800, 8) — are **calibrated against
  anchors during implementation** (Phase 1 gate), not guessed.
- Explicit `grid=GridConfig(...)` overrides the profile field-by-field.
- Scheme knobs surviving on `PDEParams`: `theta`, `use_banded_solver`, `cache_enabled`.
  (`event_damping_steps` lives on `GridConfig`, not `PDEParams`, because the time builder — which
  only sees `GridConfig` — derives `damping_steps` from it.)
- **Deleted knobs:** `auto_grid`, `time_grid_type`, `grade_exponent`,
  `steps_per_interval` / `min_steps_per_interval` / `min_steps_total`,
  `include_spot_in_critical_points`, `barrier_refine_log_width`, `barrier_refine_levels`,
  `adaptive_grid`, `log_dx_target`, `frozen_critical_points`, `event_projection`,
  `s_min`, `s_max`, `barrier_domain_expand`, `event_steps_per_day` (→ `GridConfig.steps_per_day`),
  `max_time_steps` / `max_grid_size` (→ `GridConfig.max_steps` / `max_points`),
  `bus_days_in_year` for grid purposes (→ `GridConfig.day_count`).
- Release becomes **0.4.0**; `otc-price-adapter` updates its pin and any grid params it passes.

### 4.7 Greeks and sharing behavior

- **Frozen-by-default bumps:** `GreeksCalculator` binds the layout once at base market and
  passes the same `Layout` into every bumped re-solve. No knob. Auto-bounds carry ±4σ margin, so
  standard 1% bumps cannot escape the domain; if a bumped input falls outside the frozen bounds,
  raise `NumericalError` — never silently rebuild.
- **Spot greeks** read delta/gamma off the single solved surface (the PDE solution does not
  depend on spot; spot only selects the readout point) — existing engine-mode greeks, now the
  documented default for spot.
- **Sharing:** `bind_shared` for same-underlying books. Discrete-observation products share;
  `hard_bounds` products get private layouts via `partition_shareable`. Shared spatial layout ⇒
  shared discretized operator ⇒ factorization reuse keyed by `(dt, t-dependent coefficients)`.

### 4.8 Validation & error handling

Existing exception hierarchy; `quantark.util.numerical` throughout.

- `GridRequest.__post_init__`: `tau > 0`; prices positive; times within `(0, tau)` after
  `is_close` de-duplication; `hard_bounds` ordered and positive → else `ValidationError`.
- `bind`: infeasible bounds (e.g. `hard_bounds` excluding all critical prices) → `PricingError`;
  mandatory-count-exceeds-cap → proceed + log (never drop a node).
- The `_aligned_time_index` runtime search-and-assert disappears: `step_of` is constructed by
  the builder, not re-derived by searching `t_vec`.
- Frozen-layout coverage violation during bumps → `NumericalError` (§4.7).

## 5. Testing strategy (full rewrite)

Three tiers; goldens re-baselined, never compared against the old PDE stack.

1. **Builder unit tests** (pure, fast, no solvers): exact mandatory-node alignment; fill density
   `= round(interval_days × steps_per_day)`; cap behavior incl. the inviolable-nodes overflow
   path; concentration spacing hits `eps_crit` at critical prices; uniform degeneration cases;
   projection operators conserve (rows sum to cell measure) and converge at second order on
   synthetic discontinuities; `bind_shared` union/partition behavior; request hashing/caching.
2. **Certification vs anchors** (replaces PDE-vs-old-PDE goldens): European PDE vs Black–Scholes
   closed form; barrier vs closed form; snowball/phoenix/KO-reset/DCN vs MC and QUAD (existing
   cross-family agreement gate pattern); grid-refinement convergence-order checks; greek
   smoothness across bump ladders on the frozen layout (replaces `test_bump_grid_steadiness` /
   `test_pde_fixed_bump_grid`); 2D Heston/SLV convergence re-baselined.
3. **New goldens**, frozen only after tier 2 passes, compared with the tolerance-based
   `test/golden_compare.py` (cross-arch safe; never bitwise float `==` across architectures).

Test files to rewrite or retire (non-exhaustive; enumerated fully in the implementation plan):
`test_pde_event_projection.py`, `test_pde_fixed_bump_grid.py`, `test_bump_grid_steadiness.py`,
`test_pde_grid_convergence_gate.py`, plus the grid-touching portions of the per-solver PDE tests.

## 6. Migration phases

Gate for every phase: full suite green + tier-2 anchors green. Old code for not-yet-migrated
solvers keeps working within each phase; final deletion happens only when the last consumer moves.

| Phase | Content |
|---|---|
| 0 | `grid/` package + tier-1 tests; purely additive, no consumers |
| 1 | Base solver plumbing + autocallable family (snowball, phoenix, KO-reset, DCN) on the new layer; delete their grid overrides/plumbing; calibrate accuracy presets; re-baseline autocallable goldens |
| 2 | Simple 1D solvers: barrier, double-barrier, one-touch, double-one-touch, American, European |
| 3 | 2D vol solvers (snowball/phoenix/barrier/DCN vol) + `adi_core` consume shared layouts and `damping_steps`; `VarianceLayout` idiom in `volmodels` |
| 4 | Delete `time_grid.py`, `spatial_grid.py`, `event_projection.py`, legacy `PDEParams` knobs; re-verify execution-framework seams (prepared grid artifacts hold `Layout` objects; session-prep and adapter tests); docs + 0.4.0 release notes; update `asset/equity/CLAUDE.md` |

Later, separate program: the batch book-pricing function (shared factorization + multi-RHS
banded solve over this layer).

## 7. Risks & mitigations

- **Re-certification risk** (numbers move under the clean break): mitigated by anchor-first
  testing (tier 2) before any golden is frozen, and by carrying the certified projection math
  verbatim.
- **Two-surface coupling regressions** (snowball KI): the two-surface transforms are ported as
  pure functions with dedicated unit tests against the current implementation's outputs *before*
  the old code is deleted (scaffolding oracle, removed at Phase 4).
- **2D divergence**: `damping_steps` as shared data removes the mirrored-gating class of bug;
  ADI internals untouched.
- **Downstream breakage** (`otc-price-adapter`, execution framework): version-gated at 0.4.0;
  execution seams re-verified in Phase 4; adapter updates its pin on its own schedule (0.3.x
  remains on PyPI).
- **Dropped barrier snapping** shifts prices slightly relative to snapped grids: this is the
  intended accuracy model (cell-average, second order regardless of node placement); tier-2
  convergence tests demonstrate order, and profiles are calibrated to match or beat current
  default accuracy on the anchor set.
