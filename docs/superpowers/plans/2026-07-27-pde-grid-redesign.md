# PDE Grid & Event Layer Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task (feature-flow Stage 5 mandates inline execution). Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Replace the accreted PDE grid/event stack with the declarative layer specified in
`docs/superpowers/specs/2026-07-27-pde-grid-redesign-design.md` (rev 3): `GridRequest` →
`GridBinder` → `SpatialLayout`/`TimeLayout` + four-stage `EventSchedule`, across all PDE
solvers, deleting every legacy builder and knob at the end.

**Architecture:** New package `quantark/asset/equity/engine/pde/grid/` owns all node placement
and event application; solvers shrink to three hooks (`grid_request`, `representative_vol`,
`event_schedule`) and keep their certified stepping loops. Phases 0–4 migrate: package →
autocallables → simple 1D → LV/2D → legacy deletion.

**Tech Stack:** Python 3.11, NumPy, SciPy sparse, pytest (+xdist), dataclasses.

## Global Constraints

- The spec (rev 3) is the contract. Where this plan and the spec disagree, the spec wins; stop
  and flag rather than improvise.
- Always `quantark.*` imports; `quantark.util.numerical` (`is_close`, `is_zero`, `safe_log`) —
  never bare float compares or `math.log`.
- **No MC imports anywhere under `engine/pde/`** (deterministic engines stay deterministic).
- Exact semantics by default; approximations (BGK) stay opt-in; no invented fallbacks — if a
  port is unclear, `# TODO` + stop and surface it.
- Worktree testing: the editable install resolves `quantark` to the main repo — run tests as
  `PYTHONPATH=$PWD <main>/.venv/bin/python -m pytest …` from the worktree root.
- A long-running backtest fleet shares this machine: cap pytest parallelism at `-n 4` for
  full-suite runs (targeted files can use default).
- Phase gates: full suite green + the tier-2 anchor file green before the next phase starts.
- Commit per task (conventional commits, `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`).
- Frozen dataclasses: layout classes `eq=False`; arrays `writeable=False`; never mutate a
  layout after construction.
- Numbers WILL move (clean break): never "fix" a tier-2 failure by comparing against the old
  PDE stack; anchors are BS closed form / MC / QUAD only.

## File Structure (end state)

```
quantark/asset/equity/engine/pde/grid/
├── __init__.py        # exports: GridRequest, MarketSnapshot, GridConfig, GridBinder,
│                      #          SpatialLayout, TimeLayout, Layout, EventSchedule, stages
├── request.py         # GridRequest, MarketSnapshot (frozen, validated, hashable)
├── config.py          # GridConfig (all-Optional overlay) + ACCURACY_PROFILES
├── time.py            # build_time(request, config) -> TimeLayout
├── space.py           # build_space(request, market, config) -> SpatialLayout
├── events.py          # EventSchedule ABC + ProjectedTransform helpers (ports
│                      #   event_projection.py math verbatim)
└── binder.py          # GridBinder (bind / bind_shared / rebind_time, LRU)
test/pde_grid/
├── test_grid_request.py       test_grid_config.py      test_time_builder.py
├── test_space_builder.py      test_event_schedule.py   test_grid_binder.py
└── test_anchor_certification.py   (tier 2, grows per phase)
```

Deleted at Phase 4: `time_grid.py`, `spatial_grid.py`, `event_projection.py`, legacy params
(§4.7 table), `test_pde_event_projection.py`, `test_pde_fixed_bump_grid.py`,
`test_bump_grid_steadiness.py`, `test_pde_grid_convergence_gate.py` (superseded by
`test/pde_grid/*`).

---

# Phase 0 — the grid package (additive, no consumers)

### Task 1: `GridRequest` + `MarketSnapshot`

**Files:**
- Create: `quantark/asset/equity/engine/pde/grid/request.py`
- Create: `quantark/asset/equity/engine/pde/grid/__init__.py` (exports as they appear)
- Test: `test/pde_grid/test_grid_request.py`

**Interfaces:**
- Produces: `GridRequest(tau, bound_anchors, critical_prices, hard_lower, hard_upper,
  event_times)` — frozen, hashable, validated in `__post_init__`;
  `MarketSnapshot(spot, sigma_ref, r_ref, q_ref)` — frozen, hashable.

- [ ] **Step 1: Write failing tests**

```python
# test/pde_grid/test_grid_request.py
import pytest
from quantark.asset.equity.engine.pde.grid import GridRequest, MarketSnapshot
from quantark.util.exceptions import ValidationError

def make_request(**kw):
    base = dict(tau=1.0, bound_anchors=(100.0,), critical_prices=(100.0, 103.0),
                hard_lower=None, hard_upper=None, event_times=(0.25, 0.5, 0.75))
    base.update(kw)
    return GridRequest(**base)

def test_valid_request_hashable_and_equal_by_value():
    assert make_request() == make_request()
    assert hash(make_request()) == hash(make_request())

def test_tau_must_be_positive():
    with pytest.raises(ValidationError):
        make_request(tau=0.0)

def test_event_times_must_be_interior():
    with pytest.raises(ValidationError):
        make_request(event_times=(0.0, 0.5))       # t=0 belongs to valuation_readout
    with pytest.raises(ValidationError):
        make_request(event_times=(0.5, 1.0))       # t=tau belongs to terminal stage

def test_event_times_deduplicated_and_sorted():
    r = make_request(event_times=(0.5, 0.25, 0.5 + 1e-14))
    assert r.event_times == (0.25, 0.5)            # is_close dedup, sorted

def test_hard_bounds_ordering():
    with pytest.raises(ValidationError):
        make_request(hard_lower=120.0, hard_upper=80.0)
    r = make_request(hard_lower=80.0, hard_upper=None)   # single-sided OK
    assert r.hard_lower == 80.0 and r.hard_upper is None

def test_prices_positive():
    with pytest.raises(ValidationError):
        make_request(critical_prices=(100.0, -3.0))
    with pytest.raises(ValidationError):
        make_request(bound_anchors=(0.0,))

def test_market_snapshot_value_semantics():
    a = MarketSnapshot(spot=100.0, sigma_ref=0.2, r_ref=0.03, q_ref=0.01)
    b = MarketSnapshot(spot=100.0, sigma_ref=0.2, r_ref=0.03, q_ref=0.01)
    assert a == b and hash(a) == hash(b)
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest test/pde_grid/test_grid_request.py -n0 -q`
  Expected: ImportError (module does not exist).

- [ ] **Step 3: Implement**

```python
# quantark/asset/equity/engine/pde/grid/request.py
"""Declarative grid geometry — the ONLY input the builders see from a product."""
from __future__ import annotations
from dataclasses import dataclass
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close


def _dedup_sorted_interior(times, tau):
    out = []
    for t in sorted(float(t) for t in times):
        if t <= 0.0 or t >= tau or is_close(t, 0.0) or is_close(t, tau):
            raise ValidationError(
                f"event_times must lie strictly inside (0, {tau}); got {t} "
                "(endpoint events belong to the terminal/valuation_readout stages)"
            )
        if out and is_close(out[-1], t):
            continue
        out.append(t)
    return tuple(out)


@dataclass(frozen=True)
class MarketSnapshot:
    spot: float
    sigma_ref: float
    r_ref: float
    q_ref: float

    def __post_init__(self):
        if self.spot <= 0.0:
            raise ValidationError(f"spot must be positive, got {self.spot}")
        if self.sigma_ref <= 0.0:
            raise ValidationError(f"sigma_ref must be positive, got {self.sigma_ref}")


@dataclass(frozen=True)
class GridRequest:
    tau: float
    bound_anchors: tuple
    critical_prices: tuple
    hard_lower: float | None
    hard_upper: float | None
    event_times: tuple

    def __post_init__(self):
        if self.tau <= 0.0:
            raise ValidationError(f"tau must be positive, got {self.tau}")
        for name in ("bound_anchors", "critical_prices"):
            vals = tuple(float(p) for p in getattr(self, name))
            if any(p <= 0.0 for p in vals):
                raise ValidationError(f"{name} must be positive, got {vals}")
            object.__setattr__(self, name, vals)
        if not self.bound_anchors:
            raise ValidationError("bound_anchors must contain at least one price (spot)")
        for side in ("hard_lower", "hard_upper"):
            v = getattr(self, side)
            if v is not None and float(v) <= 0.0:
                raise ValidationError(f"{side} must be positive, got {v}")
        if (self.hard_lower is not None and self.hard_upper is not None
                and self.hard_lower >= self.hard_upper):
            raise ValidationError(
                f"hard_lower ({self.hard_lower}) must be < hard_upper ({self.hard_upper})"
            )
        object.__setattr__(
            self, "event_times", _dedup_sorted_interior(self.event_times, self.tau)
        )
```

```python
# quantark/asset/equity/engine/pde/grid/__init__.py
from quantark.asset.equity.engine.pde.grid.request import GridRequest, MarketSnapshot

__all__ = ["GridRequest", "MarketSnapshot"]
```

- [ ] **Step 4: Run to verify pass** — same command, expect all PASS.
- [ ] **Step 5: Commit** — `git add quantark/asset/equity/engine/pde/grid test/pde_grid && git commit -m "feat(pde-grid): GridRequest + MarketSnapshot declarative geometry"`

### Task 2: `GridConfig` + accuracy profiles

**Files:**
- Create: `quantark/asset/equity/engine/pde/grid/config.py`
- Modify: `quantark/asset/equity/engine/pde/grid/__init__.py`
- Test: `test/pde_grid/test_grid_config.py`

**Interfaces:**
- Produces: `GridConfig(points, steps_per_day, eps_crit, num_std, bounds, max_points,
  max_steps, day_count, terminal_damping_steps, event_damping_steps)` — every field
  `Optional`, `None` = inherit; `resolve_config(accuracy: str, override: GridConfig | None)
  -> GridConfig` (fully-populated); resolved configs expose `.key -> tuple` (fingerprint).
- Profile presets (provisional, calibrated in Task 15): fast=(200 pts, 4/day),
  standard=(400, 4), high=(800, 8); shared defaults: eps_crit .003 (.002 high), num_std 4.0,
  max_points 2000, max_steps 5000, day_count 252, terminal_damping_steps 1,
  event_damping_steps 2.

- [ ] **Step 1: Failing tests** — value semantics; `resolve_config("standard", None)` fully
  populated; field-by-field override (`GridConfig(steps_per_day=8.0)` on "standard" keeps
  points=400); unknown accuracy → `ValidationError`; explicit `points > max_points` →
  `ValidationError` at resolve; negative counts rejected; `bounds=(50.0, None)` per-side kept;
  `.key` equal iff resolved fields equal.

```python
# test/pde_grid/test_grid_config.py (core cases; write all listed above)
import pytest
from quantark.asset.equity.engine.pde.grid import GridConfig, resolve_config
from quantark.util.exceptions import ValidationError

def test_resolve_standard_fully_populated():
    c = resolve_config("standard", None)
    assert c.points == 400 and c.steps_per_day == 4.0 and c.day_count == 252
    assert c.terminal_damping_steps == 1 and c.event_damping_steps == 2

def test_field_by_field_override():
    c = resolve_config("standard", GridConfig(steps_per_day=8.0))
    assert c.steps_per_day == 8.0 and c.points == 400      # only that field moved

def test_points_over_max_rejected():
    with pytest.raises(ValidationError):
        resolve_config("standard", GridConfig(points=4000))  # > max_points 2000

def test_key_fingerprint():
    assert resolve_config("fast", None).key == resolve_config("fast", None).key
    assert resolve_config("fast", None).key != resolve_config("high", None).key
```

- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement** — `GridConfig` frozen dataclass, all fields `Optional=None`;
  `ACCURACY_PROFILES: dict[str, dict]` with the preset values above; `resolve_config` merges
  profile → override (non-None wins), validates (positivity, `points <= max_points`,
  `steps_per_day >= 4.0` enforced for presets by construction), returns a resolved instance
  whose `key` property is the tuple of all fields.
- [ ] **Step 4: Verify pass.**  - [ ] **Step 5: Commit.**

### Task 3: Time builder → `TimeLayout`

**Files:**
- Create: `quantark/asset/equity/engine/pde/grid/time.py`
- Modify: `grid/__init__.py`
- Test: `test/pde_grid/test_time_builder.py`

**Interfaces:**
- Consumes: `GridRequest`, resolved `GridConfig`.
- Produces: `build_time(request, config) -> TimeLayout` with fields exactly as spec §4.2:
  `t, dt` (writeable=False), `step_of` (MappingProxyType, verbatim-float keys covering every
  `event_times` entry), `event_damping_steps`, `terminal_damping_steps` (frozensets of
  **step indices**, where step k advances from node k+1 to node k in backward time — index
  convention: a step is damped if it is among the first N backward steps after the event's
  node), `requested_steps`, `actual_steps`, `cap_exceeded`.

- [ ] **Step 1: Failing tests** (all of these, exactly):

```python
# test/pde_grid/test_time_builder.py
import numpy as np, pytest, types
from quantark.asset.equity.engine.pde.grid import GridRequest, GridConfig, resolve_config
from quantark.asset.equity.engine.pde.grid.time import build_time

CFG = lambda **kw: resolve_config("standard", GridConfig(**kw) if kw else None)

def req(tau=1.0, events=(0.25, 0.5, 0.75)):
    return GridRequest(tau=tau, bound_anchors=(100.0,), critical_prices=(100.0,),
                       hard_lower=None, hard_upper=None, event_times=events)

def test_every_event_time_is_exact_node_with_verbatim_key():
    tl = build_time(req(), CFG())
    for e in (0.25, 0.5, 0.75):
        k = tl.step_of[e]                       # exact float key — no searching
        assert tl.t[k] == e                     # exact placement

def test_fill_formula_uncapped():
    # interval 0.25y * 252 d/y = 63 days * 4/day = 252 steps per interval
    tl = build_time(req(), CFG())
    assert tl.actual_steps == 4 * 252           # 4 intervals x 252
    assert not tl.cap_exceeded

def test_no_events_degenerates_to_uniform():
    tl = build_time(req(events=()), CFG())
    assert np.allclose(np.diff(tl.t), tl.dt) and len(np.unique(np.round(tl.dt, 15))) == 1

def test_sub_day_maturity_floor():
    tl = build_time(req(tau=0.5/252, events=()), CFG())
    assert tl.actual_steps == 4                 # interval_days floored at 1.0 → 4 steps

def test_cap_extras_scaling_enforces_cap():
    # adversarial case from spec §5: 99 tiny + 1 huge interval under cap 100 → ≤ 100
    events = tuple(np.linspace(1e-4, 99e-4, 99))          # 99 tiny intervals + tail
    tl = build_time(req(tau=1.0, events=events), CFG(max_steps=100))
    assert tl.actual_steps <= 100 and not tl.cap_exceeded

def test_mandatory_overflow_keeps_nodes_sets_flag():
    events = tuple(np.linspace(0.001, 0.999, 400))        # 401 intervals > cap 100
    tl = build_time(req(events=events), CFG(max_steps=100))
    for e in events:
        assert tl.t[tl.step_of[e]] == e                   # nodes inviolable
    assert tl.cap_exceeded and tl.actual_steps == 401

def test_damping_steps_after_each_event_node_and_terminal():
    tl = build_time(req(), CFG())
    for e in (0.25, 0.5, 0.75):
        k = tl.step_of[e]
        assert {k - 1, k - 2} <= tl.event_damping_steps    # 2 steps after, backward time
    n = tl.actual_steps
    assert {n - 1} <= tl.terminal_damping_steps            # first backward step

def test_immutability():
    tl = build_time(req(), CFG())
    with pytest.raises(ValueError):
        tl.t[0] = 99.0
    assert isinstance(tl.step_of, types.MappingProxyType)
```

- [ ] **Step 2: Verify failure.**
- [ ] **Step 3: Implement** — port `TimeGrid.build_mandatory` semantics
  (`time_grid.py:156-230`) with the spec §4.3 cap fix:

```python
# quantark/asset/equity/engine/pde/grid/time.py  (core algorithm)
def build_time(request, config):
    tau, events = request.tau, request.event_times
    boundaries = np.array([0.0, *events, tau])
    lengths = np.diff(boundaries)
    days = np.maximum(1.0, lengths * float(config.day_count))
    fill = np.maximum(1, np.round(days * float(config.steps_per_day)).astype(int))
    requested = int(fill.sum())
    n_int, cap = len(lengths), int(config.max_steps)
    cap_exceeded = False
    if n_int > cap:
        fill = np.ones_like(fill); cap_exceeded = True          # nodes inviolable
        logger.warning("time grid: %d event intervals exceed max_steps=%d", n_int, cap)
    elif requested > cap:
        extras = fill - 1
        budget = cap - n_int
        fill = 1 + np.floor(extras * (budget / extras.sum())).astype(int)
    points = [np.array([0.0])]
    for start, end, n in zip(boundaries[:-1], boundaries[1:], fill):
        points.append(np.linspace(start, end, int(n) + 1)[1:])
    t = np.concatenate(points); dt = np.diff(t)
    step_of = {e: int(np.searchsorted(t, e)) for e in events}   # built ONCE, here
    for e, k in step_of.items():
        assert t[k] == e                                        # exact by construction
    event_damp = frozenset(
        k - j for k in step_of.values()
        for j in range(1, config.event_damping_steps + 1) if k - j >= 0)
    n_steps = len(dt)
    term_damp = frozenset(
        n_steps - j for j in range(1, config.terminal_damping_steps + 1) if n_steps - j >= 0)
    ...  # freeze arrays (writeable=False), wrap MappingProxyType(dict(step_of)), return TimeLayout
```

  `TimeLayout` dataclass (`frozen=True, eq=False`) lives in this file.
- [ ] **Step 4: Verify pass.**  - [ ] **Step 5: Commit.**

### Task 4: Spatial builder → `SpatialLayout`

**Files:**
- Create: `quantark/asset/equity/engine/pde/grid/space.py`
- Test: `test/pde_grid/test_space_builder.py`

**Interfaces:**
- Consumes: `GridRequest`, `MarketSnapshot`, resolved `GridConfig`.
- Produces: `build_space(request, market, config, tau_max=None) -> SpatialLayout`
  (`s, x, dx` writeable=False; `bounds`; `achieved_eps`). Port the TR-ODE machinery from
  `spatial_grid.py:313-434` (`_ode_f`, `_ode_rk4_step`, `_ode_integrate`, `_ode_find_A`)
  verbatim; ONE beta search per spec §4.4 (worst-critical-point inequality, uniform clamp,
  `[1e-12, 1e3·(x_max−x_min)]` bracket); **no snapping** (`_snap_critical_points` is not
  ported).

- [ ] **Step 1: Failing tests** — auto-bounds formula
  (`h = num_std·σ√τ + |r−q−σ²/2|·τ`, envelope over `bound_anchors`, critical-price margin
  `5·ln(1+eps_crit)`, floor `h ≥ ln(1.10)`); hard bound overrides its side verbatim while the
  other side stays auto; degenerate near-expiry (tau=1e-4 → width ≥ ±10 %); concentration:
  `achieved_eps ≤ eps_crit` on a standard 400-pt request with 3 criticals, and equals the
  worst-case (max over criticals of min-adjacent-dx converted via `expm1`); uniform clamp:
  2 criticals on a tight domain with 2000 points → near-uniform grid (max/min dx < 1.05);
  zero interior criticals → exactly uniform; critical price outside hard bounds → excluded +
  grid still builds; **no node sits exactly on a barrier-critical** unless coincidentally
  (assert not required — instead assert monotone strictly-increasing `x`, `s == exp(x)`);
  immutability.

```python
# test/pde_grid/test_space_builder.py (representative — write all cases above)
import numpy as np
from quantark.asset.equity.engine.pde.grid import GridRequest, MarketSnapshot, resolve_config
from quantark.asset.equity.engine.pde.grid.space import build_space

MKT = MarketSnapshot(spot=100.0, sigma_ref=0.2, r_ref=0.03, q_ref=0.01)

def req(**kw):
    base = dict(tau=1.0, bound_anchors=(100.0,), critical_prices=(80.0, 100.0, 103.0),
                hard_lower=None, hard_upper=None, event_times=())
    base.update(kw); return GridRequest(**base)

def test_auto_bounds_formula():
    lay = build_space(req(), MKT, resolve_config("standard", None))
    h = 4.0 * 0.2 * 1.0 + abs(0.03 - 0.01 - 0.5 * 0.04) * 1.0
    lo, hi = lay.bounds
    assert lo <= 100.0 * np.exp(-h) * (1 + 1e-12) and hi >= 100.0 * np.exp(h) * (1 - 1e-12)

def test_hard_bound_is_domain_edge():
    lay = build_space(req(hard_upper=103.0), MKT, resolve_config("standard", None))
    assert lay.bounds[1] == 103.0 and lay.s[-1] == 103.0

def test_achieved_eps_meets_target():
    cfg = resolve_config("standard", None)
    lay = build_space(req(), MKT, cfg)
    assert lay.achieved_eps <= cfg.eps_crit * (1 + 1e-6)
```

- [ ] **Step 2: Verify failure.**  - [ ] **Step 3: Implement** (port + new bounds/beta per
  spec §4.4; `achieved_eps` computed post-build as the spec defines).
- [ ] **Step 4: Verify pass.**  - [ ] **Step 5: Commit.**

### Task 5: `EventSchedule` + projection operators

**Files:**
- Create: `quantark/asset/equity/engine/pde/grid/events.py`
- Test: `test/pde_grid/test_event_schedule.py`

**Interfaces:**
- Consumes: `SpatialLayout` (for operator precompute).
- Produces:
  - `breach_weights(layout, barrier, breach_up) -> np.ndarray` — port of
    `event_projection.breach_fractions` (`event_projection.py:57-79`) taking `layout.x` +
    `safe_log(barrier)`.
  - `project_between(layout, barrier, breach_up, v_breach, v_survive) -> np.ndarray` — port
    of `project_event_values` (`event_projection.py:231+`), operating on axis 0 of arrays of
    any trailing shape (1D `(n_s,)` and 2D `(n_s, n_v)` — vectorize the interpolation columns
    exactly as the current implementation already does for slice-wise use).
  - `project_piecewise(layout, cells) -> np.ndarray` — port of `project_piecewise_event`
    (`event_projection.py:169+`) for coincident coupon+KO cells.
  - `class EventSchedule` with the four stages, default-identity:
    ```python
    class EventSchedule:
        def __init__(self, interior: Mapping[int, Callable], continuous=None,
                     terminal=None, readout=None): ...
        def terminal(self, states: dict) -> dict
        def apply(self, step: int, states: dict) -> dict      # no-op if step not in interior
        def continuous(self, step: int, states: dict) -> dict
        def valuation_readout(self, spot: float, states: dict) -> float
    ```
    `states` values: arrays with S on axis 0. Transforms are pure (return new dict; inputs
    unmodified — tests assert input arrays unchanged).

- [ ] **Step 1: Failing tests** — port the invariants from `test_pde_event_projection.py`
  against the new API: `P` weights in [0,1]; constant preservation
  (`project_between(..., ones, ones) == ones`); envelope containment; affine exactness in the
  straddling cell; threshold-perturbation continuity; broadcasting: a `(n_s, 3)` block
  projects identically column-wise to three 1D calls; stage ordering test: schedule with all
  four stages records call order `[terminal, apply@k, continuous@k, readout]` on a scripted
  walk; purity test (input arrays bit-identical after apply).
- [ ] **Step 2: Verify failure.**  - [ ] **Step 3: Implement** (math ported verbatim from
  `event_projection.py`; do not re-derive).
- [ ] **Step 4: Verify pass.**  - [ ] **Step 5: Commit.**

### Task 6: `GridBinder` (+ `Layout`)

**Files:**
- Create: `quantark/asset/equity/engine/pde/grid/binder.py`
- Test: `test/pde_grid/test_grid_binder.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  ```python
  class GridBinder:
      def __init__(self, accuracy: str, override: GridConfig | None, *,
                   cache_enabled: bool = True, cache_max_entries: int = 128): ...
      def bind(self, request, market) -> Layout
      def bind_shared(self, requests, market) -> list[Layout]
      def rebind_time(self, layout, new_request) -> Layout
  ```
  `Layout(spatial, time, request, config_key)` frozen `eq=False`, defined here.
  Cache: `functools`-style LRU dict keyed `(request, market, config_key)`; identity reuse
  observable (`bind(r, m) is bind(r, m)`).

- [ ] **Step 1: Failing tests** — identity reuse with cache on; distinct objects with
  `cache_enabled=False`; LRU eviction at `cache_max_entries`; `bind_shared([])` →
  `ValidationError`; `bind_shared([r]) == [bind(r)]`-equivalent (same spatial bounds/points);
  multi-request sharing: `len({id(l.spatial) for l in layouts}) == 1`; value-identical
  requests share one `TimeLayout` object; request with any hard bound in `bind_shared` →
  `ValidationError`; shared spatial uses max-tau bounds and unioned critical prices;
  `rebind_time(layout, rolled_request)` keeps `spatial` by identity, rebuilds time;
  infeasible domain (hard bounds exclude every anchor) → `PricingError`.
- [ ] **Step 2–5:** implement (space built from a synthetic union-request in `bind_shared`),
  verify, commit.

### Task 7: Phase-0 gate

- [ ] Run `python -m pytest test/pde_grid -n0 -q` → all green.
- [ ] Run full suite `python -m pytest -n 4 -q` → green (package is additive; nothing else
  may move).
- [ ] Commit any stragglers; tag the phase in the commit message:
  `feat(pde-grid): phase 0 complete — grid package + tier-1 tests`.

---

# Phase 1 — base plumbing + autocallable family

### Task 8: Base-solver seam (`grid_request`/`representative_vol`/`event_schedule` + binder)

**Files:**
- Modify: `quantark/asset/equity/engine/pde/base_pde_solver.py`
- Test: `test/pde_grid/test_solver_seam.py`

**Interfaces:**
- Produces (on `BasePDESolver`):
  - `_uses_grid_layer() -> bool` (default `False`; migrated solvers return `True`);
  - `grid_request(product, market) -> GridRequest` (raises `NotImplementedError` unless
    migrated);
  - `representative_vol(product, pricing_env) -> float` — default: the existing
    strike-selected constant-vol resolution already used by `_solve` (reuse that code path);
  - `event_schedule(product, pricing_env, layout) -> EventSchedule` (default: empty schedule);
  - `_binder` lazy property: `GridBinder(params.accuracy, params.grid,
    cache_enabled=params.cache_enabled, cache_max_entries=params.grid_cache_max_entries)` —
    for this phase, `accuracy`/`grid` read via `getattr(params, ..., "standard"/None)` so old
    `PDEParams` still works (fields added for real in Task 9);
  - `solve(..., layout: Layout | None = None)` plumbed through `_solve`; when a layout is
    supplied AND `_uses_grid_layer()`: validate per spec §4.6 (alignment fields exact match on
    a freshly-computed request → `ValidationError`; spot + interior criticals strictly inside
    bounds with hard-edge `is_close` exemption → `NumericalError`).
  - When `_uses_grid_layer()` is False, everything behaves exactly as today (legacy path
    untouched until Phase 4).
- [ ] Tests: a minimal fake solver subclass exercising validation acceptance/rejection paths
  (5 cases: match, tau mismatch, event_times mismatch, spot outside bounds, barrier-on-edge
  exempt). Implement, verify, commit.

### Task 9: `PDEParams.accuracy` + `grid` fields

**Files:**
- Modify: `quantark/asset/equity/param/engine_params.py` (PDEParams: add
  `accuracy: str = "standard"`, `grid: Optional[GridConfig] = None`; validate accuracy ∈
  {fast, standard, high}; import from the grid package)
- Test: extend `test/pde_grid/test_grid_config.py` with a `PDEParams(accuracy="high",
  grid=GridConfig(points=600))` construction + validation-rejection test.
- [ ] Old knobs stay functional this phase (deleted in Task 24). Implement, verify, commit.

### Task 10: Snowball transform oracle (scaffolding, deleted at Phase 4)

**Files:**
- Create: `test/pde_grid/test_snowball_transform_oracle.py`

The KO/KI/coupon block transforms that Task 11 writes must reproduce the CURRENT certified
code before the old code goes. Build fixtures by calling today's private methods directly:

- [ ] Construct a standard daily-KI snowball (from `test_pde_engine.py` fixtures: initial=100,
  KO 103 monthly, KI 75 daily, coupon 15 %, tau=1) and a `SnowballPDESolver` with default
  params; run `_build_grids`, capture `(x_vec, s_vec)`.
- [ ] For synthetic surfaces `V0 = linspace ramp`, `V1 = payoff-shaped`, call today's
  `_apply_ko_jump` / `_apply_ki_jump` (`snowball_pde_solver.py:2343+`) at a KO index and a KI
  index; save inputs/outputs as `.npz` under `test/pde_grid/data/oracle_snowball.npz`
  (COMMITTED — the fixture must outlive the old code).
- [ ] Write the (initially skipped) comparison test: new `EventSchedule` stages on the same
  inputs reproduce outputs within 1e-12 on the same grid. Task 11 un-skips it. Commit.

### Task 11: Migrate `SnowballPDESolver`

**Files:**
- Modify: `quantark/asset/equity/engine/pde/snowball_pde_solver.py`
- Test: oracle from Task 10 + existing snowball tests in `test_pde_engine.py`

**Interfaces (pattern-setter for Tasks 12–14):**

- [ ] **Step 1: implement `grid_request`** (replaces `_build_grids` override +
  `_time_grid_spec` + `_ko_coupon_align_times` + `_ki_monitor_times` + `_ki_nodes_in_grid` —
  port their logic verbatim into ONE method):

```python
def _uses_grid_layer(self):
    return True

def grid_request(self, product, market):
    tau = self._current_tau                       # set by _prepare_solve_state, as today
    cfg = product.barrier_config
    ko_times = self._ko_coupon_event_times(cfg, tau)       # port of _ko_coupon_align_times
    ki_times = []
    criticals = [market.spot, product.strike, *self._ko_barrier_levels(cfg)]
    ki_barrier = self._effective_ki_barrier(product)       # BGK-shifted when active
    if product.has_ki_barrier and not self._already_knocked_in(product):
        criticals.append(ki_barrier)
        if not self._ki_continuous and not self._bgk_active:
            ki_times = self._ki_event_times(cfg, tau)      # port of _ki_monitor_times
    return GridRequest(
        tau=tau,
        bound_anchors=(market.spot, product.strike),
        critical_prices=tuple(criticals),
        hard_lower=None, hard_upper=None,                  # discrete observation
        event_times=tuple(sorted(set(ko_times) | set(ki_times))),
    )
```

- [ ] **Step 2: implement `event_schedule`** — builds interior transforms from the KO records
  (`{layout.time.step_of[rec.observation_time]: ko_transform(rec)}`), KI interior transforms
  at discrete-KI steps, `continuous` for continuous/BGK KI, `terminal` from today's
  `_apply_terminal_ko`, `valuation_readout` from today's `_compose_t0_readout` /
  `_reset_t0_readout_state` semantics. Each transform is a pure closure using
  `project_between` / `project_piecewise` with the values today's `_apply_ko_jump` /
  `_apply_ki_jump` compute (`ko_payoff`, discounting via the same
  `_get_ko_payoff_at_time`); states dict: `{"alive": grid_v0, "ki": grid_v1}`.
- [ ] **Step 3: rewire `_solve`** — `_build_grids` override deleted; base builds
  `layout = self._binder.bind(self.grid_request(product, market), market)`;
  `_time_stepping_two_surface` reads `layout.time` (damping via the two frozensets replacing
  `theta_by_step` construction; event application via `schedule.apply` / `schedule.continuous`
  replacing `_apply_step_modifications_two_surface`'s index-dict lookups). DELETE:
  `_build_grids` override, `_ko_observation_indices`/`_ki_observation_indices`/
  `_ki_barrier_by_tidx` state, `_aligned_time_index`, `_grid_cache_key`,
  `_observation_cache_key`, `_time_grid_spec`, `_get_event_times`, `_ki_nodes_in_grid`
  (absorbed into `grid_request`).
- [ ] **Step 4:** un-skip the Task-10 oracle test → green; run
  `python -m pytest test/test_pde_engine.py test/pde_grid -n0 -q` → snowball cases green
  (PV moves are expected vs old goldens — those goldens are re-baselined in Task 15, so run
  only the non-golden subset here: `-k "snowball and not golden"`).
- [ ] **Step 5: Commit.**

### Task 12: Migrate `PhoenixPDESolver`

Same pattern as Task 11 (`phoenix_pde_solver.py` extends the snowball solver — most of the
migration is inherited). Phoenix specifics: coupon-barrier decisions at coupon dates are
interior transforms on the phoenix state blocks (today's phoenix step-modification override);
coupon dates == KO dates → single `event_times` set; coincident coupon+KO uses
`project_piecewise` (today's one-pass `project_piecewise_event` call). Un-skip/extend oracle
with a phoenix fixture. Run `-k "phoenix and not golden"`. Commit.

### Task 13: Migrate `KOResetSnowballPDESolver`

Reset dates → `event_times`; reset semantics (barrier-level switch at scheduled dates —
today's override of the KO record consumption) → interior transforms keyed by reset steps.
Verify `test_ko_reset_snowball_pde.py` non-golden subset. Commit.

### Task 14: Migrate `DCNPDEEngine`

`dcn_pde_solver.py`: coupon schedule → `event_times`; coupon decision transforms; no KI
blocks (states: `{"alive": ...}`). Verify `test_dcn_pde_solver.py` non-golden subset. Commit.

### Task 15: Phase-1 gate — anchors, presets, goldens, greeks

**Files:**
- Create: `test/pde_grid/test_anchor_certification.py`
- Modify: `quantark/asset/equity/riskmeasures/greeks_calculator.py` +
  `quantark/asset/equity/engine/pde/base_pde_solver.py` (`create_bump_context` →
  frozen-`Layout`; theta bumps via `binder.rebind_time`)
- Modify: golden fixtures for autocallable PDE tests

- [ ] Write the tier-2 anchor tests exactly per spec §5 provisional tolerances (European vs
  BS is Phase 2 — here: snowball/phoenix/KO-reset/DCN vs `SnowballMCEngine`-family QMC
  (seed=42, 2^18 paths, within 3× stderr) and vs the Quad engines (rel < 5e-4); greek
  smoothness ladder (±2 % spot, 9 points: delta monotone where payoff implies, gamma no
  sign-flip)).
- [ ] Calibrate presets: run the anchor set under fast/standard/high; adjust `points` /
  `steps_per_day` upward only if an anchor fails; record final presets in `config.py` +
  spec-deviation note if changed.
- [ ] Replace `create_bump_context` (`base_pde_solver.py:360`) — bump context now carries the
  frozen `Layout` (drop `frozen_critical_points` plumbing); `greeks_calculator.py:1234`
  consumes it; theta bump path calls `rebind_time`. Greek-smoothness anchor is the test.
- [ ] Re-baseline autocallable goldens (`git grep -l golden test/ | grep -i "snowball\|phoenix\|dcn\|ko_reset"`);
  regenerate via each test's frozen-generation path, verify vs anchors FIRST, then freeze.
- [ ] Full suite `-n 4` green. Commit: `feat(pde-grid): phase 1 complete — autocallables on the grid layer`.

---

# Phase 2 — simple 1D solvers

### Task 16: `EuropeanPDESolver` + `AmericanPDESolver`

Empty `event_times`; `bound_anchors=(spot, strike)`; criticals `(spot, strike)`;
no hard bounds. American keeps its obstacle projection untouched. Add the European-vs-BS
anchor rows (spec §5 tolerances: rel PV < 5e-4 standard / < 1e-4 high, |Δdelta| < 5e-4,
gamma rel < 1e-2) and the spatial-order ≥ 1.7 refinement test to
`test_anchor_certification.py`. Delete their `_build_grids`-era plumbing. Commit.

### Task 17: `BarrierPDESolver` + `DoubleBarrierPDESolver`

Continuous KO → `hard_lower`/`hard_upper` (single-barrier sets exactly ONE side — the up/down
logic currently in `_resolve_spatial_bounds` `base_pde_solver.py:882-902` moves into each
solver's `grid_request`); discrete variants → `event_times` + interior transforms; rebate
handling stays in terminal/boundary code. Anchor: continuous barrier vs closed form
(rel < 1e-3). Commit.

### Task 18: `OneTouchPDESolver` + `DoubleOneTouchPDESolver`

Same shape as Task 17 (touch payoffs; hard bounds at barriers; anchors vs their analytical
engines). Commit.

### Task 19: Phase-2 gate

Full suite `-n 4` + anchors green; re-baseline any 1D goldens; commit
`feat(pde-grid): phase 2 complete — simple 1D solvers migrated`.

---

# Phase 3 — LV + 2D vol solvers

### Task 20: LV 1D family (`LocalVolPDESolver`, `LocalVolBarrier/Snowball/Phoenix/DCN`)

`representative_vol` override: the representative measure each engine uses for bounds today
(read each solver's current bounds code and port that exact expression). S-axis + time via
the binder; solver-level `n_x`/`n_t`/`grid_style`/`grid_focus`/`pin_critical_spots` args
retired → constructor accepts `PDEParams` accuracy/grid (keep old kwargs as hard
`ValidationError` mentions in Phase 4, functional here). LV stepping loops read damping
frozensets. Anchor: LV European vs its own MC engine. Commit per sub-family if diffs get
large (LV European+Barrier, then LV autocallables).

### Task 21: 2D Heston/SLV family + `adi_core` consumption

- `VarianceConfig` (in `quantark/volmodels/`): `n_v`, `v_grid_power`, concentration controls —
  frozen dataclass consumed by the existing variance-grid builders (`VarianceLayout` naming
  over current arrays; builders' math untouched).
- 2D solvers (`Heston*/HestonSLV*` European, Barrier, Snowball, Phoenix, DCN): S-axis
  layout + `TimeLayout` from the binder; `adi_core` gains
  `damping_steps: tuple[frozenset, frozenset]` input replacing its internal
  Rannacher/`rannacher` bookkeeping and the mirrored `theta_by_step` gating
  (`snowball_vol_pde_solvers.py` / `phoenix_vol_pde_solvers.py` pass it through);
  slice-wise event application replaced by the same `EventSchedule` stages on
  `(n_s, n_v)` blocks (2D KO stays per-step injection via `continuous`).
- 2D `representative_vol` protocol pinned here: Heston `sigma_ref = sqrt(max(v0, theta))`
  unless the current bounds code says otherwise — **port the existing expression**.
- Anchors: existing `test_heston_pde_convergence.py` re-baselined + MC agreement rows.
Commit per sub-family (European 2D, then barrier, then autocallables).

### Task 22: Phase-3 gate

Full suite `-n 4` + anchors green. Commit `feat(pde-grid): phase 3 complete — LV/2D migrated`.

---

# Phase 4 — deletion & release

### Task 23: Delete legacy grid modules

- [ ] `git grep -l "time_grid\|spatial_grid\|event_projection" quantark/` → migrate/delete
  every remaining importer (expected: only `base_pde_solver.py` legacy path + `__init__`).
- [ ] Delete `time_grid.py`, `spatial_grid.py`, `event_projection.py`, the base legacy
  `_resolve_*`/`_build_grids_uncached` path and `_uses_grid_layer` (now always true →
  remove the flag entirely).
- [ ] `backward_operator.py` consumes `SpatialLayout`.
- [ ] Full suite green. Commit.

### Task 24: Delete legacy params + finalize surface

- [ ] Remove every "deleted" row of spec §4.7 from `PDEParams`/`EngineParams` docstrings and
  fields; constructing with a removed kwarg → normal dataclass `TypeError` (add a
  `__init__` note in the class docstring listing the 0.4.0 removals).
- [ ] Retire superseded test files (`test_pde_event_projection.py`,
  `test_pde_fixed_bump_grid.py`, `test_bump_grid_steadiness.py`,
  `test_pde_grid_convergence_gate.py`) after confirming each invariant has a `test/pde_grid`
  successor (grep the class/test names; port any orphan invariant first).
- [ ] Delete the Task-10 oracle scaffolding (fixtures + test).
- [ ] Full suite green. Commit.

### Task 25: Execution-framework seams + docs + release notes

- [ ] `pde_execution_adapters.py` / `pde_session_prep.py`: prepared grid artifacts hold
  `Layout` objects; run `test/` files covering execution adapters + session prep.
- [ ] Update `quantark/asset/equity/CLAUDE.md` (grid section + params example),
  `docs/` release notes for 0.4.0 (param removals table copied from spec §4.7).
- [ ] Tier-3: regenerate remaining goldens, verify with `test/golden_compare.py` tolerances.
- [ ] Full suite `-n 4` green twice consecutively. Commit
  `feat(pde-grid): phase 4 complete — legacy layer deleted, 0.4.0 surface final`.

---

## Self-review checklist (run after writing, before execution)

- Spec §4.2 contracts ↔ Tasks 1–6; §4.3 ↔ Task 3; §4.4 ↔ Task 4; §4.5 ↔ Tasks 3/11/21;
  §4.6 ↔ Tasks 8/11–14/16–21; §4.7 ↔ Tasks 2/9/24; §4.8 ↔ Task 15; §4.9 ↔ Tasks 1/2/6/8;
  §5 ↔ Tasks 7/10/15/16/19/22/24/25; §6 ↔ phase gates. No spec row without a task.
- Names used across tasks: `build_time`, `build_space`, `project_between`,
  `project_piecewise`, `breach_weights`, `GridBinder.bind/bind_shared/rebind_time`,
  `_uses_grid_layer`, `grid_request`, `representative_vol`, `event_schedule` — consistent
  everywhere above.
