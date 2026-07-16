# Execution Framework Phase 4 — PDE Preparation and Rich Outcomes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Session-dispatched PDE engines reuse safe preparation (grids, term-structure step coefficients, factorization packs, Dupire surfaces) through `PreparedArtifactCache` descriptors, and one **value** solve populates PV + event stats + grid projections (event-stat indicator sweeps remain the engines' designed separate pass — the session performs exactly as many backward marches as the direct `price_with_events` path, never more) — all bitwise-identical to direct calls.

**Architecture:** Clone-based prepared adapters (the reviewed Phase 1 DCN pattern — no live-engine mutation, `exact=True` registrations so subclass `price()` overrides fall to legacy). Native seams on the solvers keep ONE implementation: `price()` delegates to `_price_with_solution`, `price_with_events` delegates to `_session_outputs`, `calculate_spot_greeks_curve` delegates to `_grid_projection_from_solution`. Artifact keys derive from the engine's own sanctioned reuse contract (`_grid_cache_key`) plus curve/surface value fingerprints; anything uncanonicalizable builds fresh (spec §10.1 — correctness never depends on cacheability).

**Tech Stack:** quantark.execution (contracts/kernel/registry/cache), scipy.sparse splu + solve_banded, numpy, pytest.

**Scope decisions (kickoff, 2026-07-16):** full sweep — every one of the 29 exported PDE engines (24 equity + 3 FX + 2 bond) gets an explicit `prepared_state`; prepared adapters land on the 18 where at least one artifact or rich output exists; the rest carry specific spec-§9.2-legality rationales. Full §9.2 artifact menu: grids, event-aligned time grids (inside the grid artifact), term contexts (materialized as the step-coefficients artifact), local-vol surfaces, static operator coefficients, factorization packs (1D families only — 2D ADI operators are time-dependent and cannot prove reuse safe, so they are rebuilt, which is the spec's own legality escape). Rich outputs: PV + EVENT_STATS + GRID. Gates: bitwise-vs-direct + build-count; merge to local main only, no push.

## Global Constraints

- Canonical `quantark.*` imports only; PEP 8; dataclasses + type hints; `quantark.util.numerical` for float comparisons (never hardcoded tolerances).
- **No MC inside PDE engines** (deterministic engines stay deterministic).
- **Exact semantics by default**: session results must be **bitwise equal** to direct calls — `==` on floats, `np.array_equal` on arrays, in every parity test. No tolerances.
- The kernel (`quantark/execution/kernel.py`) never statically imports asset code. Adapters live near their engine families; registry entries are lazy string paths.
- `exact=True` registration whenever the adapter clones/reconstructs the engine (Phase 3 code-gate rule): unknown subclasses must fall to the legacy adapter, never be silently driven through the clone.
- Artifact payloads are immutable once published; failure paths release handles/leases in `finally`; correctness never depends on cache admission (bypass path must stay bitwise).
- Run worktree tests as `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest …` so worktree source shadows the editable install.
- Known pre-existing failure `test_snowball_quad_flat_identity_golden` is out of scope — never "fix" it here.
- Execution tests import fixtures as `from execution.matrix_fixtures import …` (test/execution is a package on the pytest path).
- Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `quantark/asset/equity/engine/pde/base_pde_solver.py` (modify) | `PDESessionOutputs`, `_price_with_solution`, `_session_outputs` (base), `_grid_projection_from_solution`, injection seams (`_session_grids` / `_step_coefficients_for_solve` / `_session_matrix_pack`), `_session_factorization_packs` |
| `quantark/asset/equity/engine/pde/snowball_pde_solver.py` (modify) | autocallable `_price_with_solution` / `_session_outputs`; `price_with_events` refactored onto the seam; `_session_banded_pack` read path; banded pack builder override |
| `quantark/asset/equity/engine/pde/phoenix_pde_solver.py`, `ko_reset_snowball_pde_solver.py` | inherit the seams (no edits expected; verify) |
| `quantark/asset/equity/engine/pde/snowball_vol_pde_solvers.py`, `phoenix_vol_pde_solvers.py` (modify) | LV `_session_outputs` surface-wrap overrides; 2D `_session_outputs` overrides replacing the 2D `price_with_events` |
| `quantark/asset/equity/engine/pde/pde_session_prep.py` (create) | grid / step-coefficients / factorization artifact states (descriptors + builders + byte estimates) |
| `quantark/execution/prep/__init__.py`, `quantark/execution/prep/dupire.py` (create) | shared Dupire-surface artifact state, extracted from `_DCNLocalVolAdapterBase._prepare_surface` |
| `quantark/asset/equity/engine/mc/dcn_execution_adapters.py` (modify) | `_prepare_surface` delegates to `execution.prep.dupire` (behavior-preserving) |
| `quantark/asset/equity/engine/pde/pde_execution_adapters.py` (create) | `PDESessionValue`, `_EquityPDESessionBase`, 1D adapter, LV adapters, Heston/SLV 2D autocallable adapter |
| `quantark/asset/fx/engine/pde/fx_pde_execution_adapters.py` (create) | FX LV prepared adapter |
| `quantark/execution/registry.py` (modify) | 18 exact registrations + factories |
| `quantark/execution/inventory.py` (modify) | `PREPARED_STATES`, `prepared_state`/`prepared_rationale` fields, all rows |
| `test/execution/test_pde_session_seams.py` (create) | seam-refactor equivalence tests |
| `test/execution/test_pde_prepared_adapters.py` (create) | bitwise parity + rich outputs + fail-closed + subclass fallback |
| `test/execution/test_pde_artifact_reuse.py` (create) | build-count, cache-stats, bump-invalidation, bypass gates |
| `test/execution/test_pde_convergence_gate.py` (create) | production + refined-resolution bitwise gates |
| `test/execution/test_registry.py`, `test_inventory.py` (modify) | expected-set + CI gates |
| `test/execution/benchmark_phase4.py` (create) | CRN-repricing + overhead benchmark (mirrors `benchmark_phase2.py` conventions) |

Engines gaining prepared adapters (18, all `exact=True`):
1D base family (9): `EuropeanPDESolver`, `AmericanPDESolver`, `BarrierPDESolver`, `DoubleBarrierPDESolver`, `OneTouchPDESolver`, `DoubleOneTouchPDESolver`, `SnowballPDESolver`, `KOResetSnowballPDESolver`, `PhoenixPDESolver`.
LV family (4): `LocalVolPDESolver`, `LocalVolBarrierPDESolver`, `LocalVolSnowballPDESolver`, `LocalVolPhoenixPDESolver`.
Heston/SLV autocallables (4): `HestonSnowballPDESolver`, `HestonSLVSnowballPDESolver`, `HestonPhoenixPDESolver`, `HestonSLVPhoenixPDESolver`.
FX (1): `FxLocalVolPDESolver`.

Engines staying legacy with explicit rationale: `HestonPDESolver`, `HestonSLVPDESolver`, `HestonBarrierPDESolver`, `HestonSLVBarrierPDESolver` (frozen construction-owned params; time-dependent ADI operators rebuilt per §9.2), `DCNPDEEngine`, `HestonDCNPDESolver` (kernel-internal grids; DCN LV variants already prepared), FX Heston/SLV, bond convertibles + facades (env-bound, internal discretization), `PDEEngine` facade (binds at solver level), `BasePDESolver` (abstract).

---

### Task 0: Freeze pre-refactor goldens (independent oracle — plan-gate finding)

Task 1 rewrites the DIRECT methods (`price`, `price_with_events`, `calculate_spot_greeks_curve`) onto the session seams, so session-vs-direct parity alone is a common-mode comparison: a refactor bug would shift both sides. Pin the pre-refactor behavior FIRST, on the untouched worktree.

**Files:**
- Modify: `test/execution/freeze_goldens.py` (extend with a Phase 4 section, following its existing conventions — read it first)
- Create: `test/execution/goldens/pde_phase4_goldens.json`
- Create: `test/execution/test_pde_pre_refactor_goldens.py`

- [ ] **Step 1:** Before ANY source edit, extend the freeze script to record, for `EuropeanPDESolver`, `SnowballPDESolver`, `PhoenixPDESolver`, `KOResetSnowballPDESolver`, `LocalVolSnowballPDESolver`, and `HestonSnowballPDESolver` on the matrix-fixture cases: `price`, `price_with_events` npv + every event-distribution array (as exact float lists), and `calculate_spot_greeks_curve` at levels `0.9/1.0/1.1 × spot` — at production params AND at refined params (`grid_size*2, time_steps*2`) for the two 1D representatives.
- [ ] **Step 2:** Run it on the pristine worktree; commit the JSON.
- [ ] **Step 3:** Add `test_pde_pre_refactor_goldens.py` asserting the DIRECT outputs equal the frozen goldens **bitwise** (exact float equality). This test must pass before Task 1 and keep passing through every later task — it is the independent evidence behind the Task 9 convergence gate.
- [ ] **Step 4: Commit** — `test(execution): freeze pre-refactor PDE goldens (Phase 4 oracle)`

---

### Task 1: Session output seams on the 1D solvers (single implementation)

**Files:**
- Modify: `quantark/asset/equity/engine/pde/base_pde_solver.py`
- Modify: `quantark/asset/equity/engine/pde/snowball_pde_solver.py`
- Modify: `quantark/asset/equity/engine/pde/snowball_vol_pde_solvers.py`
- Modify: `quantark/asset/equity/engine/pde/phoenix_vol_pde_solvers.py`
- Test: `test/execution/test_pde_session_seams.py`

**Interfaces:**
- Produces: `PDESessionOutputs(npv: float, solution: Optional[PDESolutionResult], event_stats, event_distribution)` NamedTuple in `base_pde_solver.py`; `BasePDESolver._price_with_solution(product, env) -> (float, Optional[PDESolutionResult])`; `BasePDESolver._session_outputs(product, env, want_events=False, want_grid=False, streams=None) -> PDESessionOutputs`; `BasePDESolver._grid_projection_from_solution(result, spot_levels=None) -> list[dict]`.
- Consumed by: Task 5's adapters and by `price` / `price_with_events` / `calculate_spot_greeks_curve` themselves.

- [ ] **Step 1: Write failing tests** (`test/execution/test_pde_session_seams.py`)

```python
"""Seam-refactor equivalence: price/price_with_events/spot-curve drive the
session seams, so seam outputs and public outputs are the same object graph."""
import numpy as np
import pytest

from execution.matrix_fixtures import _pdep  # PDEParams helper


def _snowball_case():
    from execution.matrix_fixtures import _build_equity_pde  # reuse builders
    fixtures = _build_equity_pde()
    return fixtures["SnowballPDESolver"]()   # (engine, product, env)


class TestPriceWithSolution:
    def test_price_equals_seam_pv_vanilla(self):
        from quantark.asset.equity.engine.pde import EuropeanPDESolver
        from execution.matrix_fixtures import _build_equity_pde
        engine, product, env = _build_equity_pde()["EuropeanPDESolver"]()
        pv, solution = engine._price_with_solution(product, env)
        assert engine.price(product, env) == pv
        assert solution is not None
        assert np.array_equal(solution.solution_vec, engine._solve(product, env).solution_vec)

    def test_price_equals_seam_pv_snowball(self):
        engine, product, env = _snowball_case()
        pv, solution = engine._price_with_solution(product, env)
        assert engine.price(product, env) == pv
        assert solution is not None


class TestSessionOutputs:
    def test_events_match_price_with_events(self):
        engine, product, env = _snowball_case()
        direct = engine.price_with_events(product, env, emit_distribution=True)
        out = engine._session_outputs(product, env, want_events=True)
        assert out.npv == direct.npv
        assert np.array_equal(
            np.asarray(out.event_distribution.ko_probabilities),
            np.asarray(direct.event_distribution.ko_probabilities),
        )

    def test_grid_matches_spot_greeks_curve(self):
        engine, product, env = _snowball_case()
        levels = [float(env.spot) * m for m in (0.9, 1.0, 1.1)]
        direct = engine.calculate_spot_greeks_curve(product, env, levels)
        out = engine._session_outputs(product, env, want_grid=True)
        projected = engine._grid_projection_from_solution(out.solution, levels)
        assert projected == direct

    def test_heston_2d_events_match(self):
        from execution.matrix_fixtures import _build_equity_pde
        engine, product, env = _build_equity_pde()["HestonSnowballPDESolver"]()
        direct = engine.price_with_events(product, env, emit_distribution=True)
        out = engine._session_outputs(product, env, want_events=True)
        assert out.npv == direct.npv
        assert out.solution is None  # 2D never exposes a 1D grid
```

Note: adapt fixture access to `matrix_fixtures`' actual builder API at implementation time (the builders return `(engine, product, env)` thunks keyed by class name; if `HestonSnowballPDESolver` needs its `_heston_pde` sub-map, follow the file's own pattern). `event_distribution` field names must match `quantark/cashleg/event_distribution.py` — read it first and fix the attribute names in the test (the assertion structure stays).

- [ ] **Step 2: Run tests, verify they fail** (`_price_with_solution` missing)

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/execution/test_pde_session_seams.py -x -q
```

- [ ] **Step 3: Implement the seams**

In `base_pde_solver.py` (after `PDESolutionResult`):

```python
class PDESessionOutputs(NamedTuple):
    """One-solve session bundle (spec sections 8/9.3 native seam)."""

    npv: float
    solution: Optional[PDESolutionResult]
    event_stats: object
    event_distribution: object
```

Refactor `BasePDESolver.price` (pure extraction — copy the existing body verbatim into the seam):

```python
def _price_with_solution(self, product, pricing_env):
    """price() preamble + one solve. None solution = short-circuit path."""
    spot = pricing_env.spot
    tau = product.get_maturity(pricing_env)
    if tau <= 0:
        return self._calculate_intrinsic(product, spot), None
    result = self._solve(product, pricing_env)
    return (
        self._interpolate_price(result.solution_vec, result.x_vec, result.spot_log),
        result,
    )

def price(self, product, pricing_env):
    """Price the option using the PDE finite difference method."""
    return self._price_with_solution(product, pricing_env)[0]

def _session_outputs(self, product, pricing_env, want_events=False,
                     want_grid=False, streams=None):
    npv, solution = self._price_with_solution(product, pricing_env)
    return PDESessionOutputs(
        npv=float(npv),
        solution=solution if want_grid else None,
        event_stats=None,
        event_distribution=None,
    )
```

Extract `_grid_projection_from_solution` from `calculate_spot_greeks_curve` (the method keeps its own validation + expired fallback and then delegates; `spot_levels=None` means project at the grid nodes `result.s_vec`):

```python
def _grid_projection_from_solution(self, result, spot_levels=None):
    prices = np.asarray(result.solution_vec, dtype=float)
    deltas = np.gradient(prices, result.s_vec, edge_order=2)
    gammas = np.gradient(deltas, result.s_vec, edge_order=2)
    spots = np.asarray(
        result.s_vec if spot_levels is None else spot_levels, dtype=float
    )
    return [
        {
            "spot": float(spot),
            "price": float(np.interp(spot, result.s_vec, prices)),
            "delta": float(np.interp(spot, result.s_vec, deltas)),
            "gamma": float(np.interp(spot, result.s_vec, gammas)),
            "calculation_mode": "engine_grid",
        }
        for spot in spots
    ]
```

In `snowball_pde_solver.py`, extract `price()`'s body verbatim (check/validate → expired → immediate-KO → solve) into `_price_with_solution` returning `(value, None)` on the short-circuits and `(interp, result)` after `_solve`; `price()` becomes the one-liner. Override `_session_outputs`:

```python
def _session_outputs(self, product, pricing_env, want_events=False,
                     want_grid=False, streams=None):
    from quantark.cashleg.event_distribution import EventDistribution

    npv, solution = self._price_with_solution(product, pricing_env)
    stats = None
    dist = None
    if want_events and solution is not None:
        stats = self.calculate_event_stats(
            product, pricing_env, npv=float(npv), streams=streams
        )
        if stats is not None:
            dist = EventDistribution.from_autocallable_stats(stats)
    return PDESessionOutputs(
        npv=float(npv),
        solution=solution if want_grid else None,
        event_stats=stats,
        event_distribution=dist,
    )
```

Refactor `price_with_events` into the shared wrapper (this REPLACES its current body; the trivial-distribution taus are equivalent because `max(float(tau), 0.0) == float(tau)` on every non-expired short-circuit — verify by reading the current branches once more before deleting):

```python
def price_with_events(self, product, pricing_env, emit_distribution=True,
                      streams=None):
    from quantark.cashleg.event_distribution import EventDistribution, PricingResult

    out = self._session_outputs(
        product, pricing_env, want_events=emit_distribution, streams=streams
    )
    if out.event_distribution is not None:
        return PricingResult(npv=out.npv, event_distribution=out.event_distribution)
    tau = product.get_maturity(pricing_env)
    return PricingResult(
        npv=out.npv,
        event_distribution=EventDistribution.trivial(max(float(tau), 0.0)),
    )
```

In `snowball_vol_pde_solvers.py`: on `LocalVolSnowballPDESolver`, DELETE the `price_with_events` override and ADD:

```python
def _session_outputs(self, product, pricing_env, **kwargs):
    return self._with_surface(
        pricing_env,
        lambda: SnowballPDESolver._session_outputs(self, product, pricing_env, **kwargs),
    )
```

On `_Heston2DSnowballPDEBase`, DELETE the `price_with_events` override and ADD (its `calculate_event_stats` override stays):

```python
def _session_outputs(self, product, pricing_env, want_events=False,
                     want_grid=False, streams=None):
    from quantark.cashleg.event_distribution import EventDistribution
    from quantark.asset.equity.engine.pde.base_pde_solver import PDESessionOutputs

    npv = float(self.price(product, pricing_env))
    stats = None
    dist = None
    if want_events:
        stats = self.calculate_event_stats(
            product, pricing_env, npv=npv, streams=streams
        )
        if stats is not None:
            dist = EventDistribution.from_autocallable_stats(stats)
    return PDESessionOutputs(
        npv=npv, solution=None, event_stats=stats, event_distribution=dist
    )
```

Mirror both changes in `phoenix_vol_pde_solvers.py` (`LocalVolPhoenixPDESolver`, `_Heston2DPhoenixPDEBase`). `PhoenixPDESolver` and `KOResetSnowballPDESolver` inherit the snowball seams unchanged — verify neither overrides `price`/`price_with_events` (grep before assuming).

Behavior note to record in the commit: for an EXPIRED product, 2D `price_with_events` previously built `EventDistribution.trivial(negative_tau)`; the shared wrapper clamps to `0.0`, matching 1D semantics — deliberate.

- [ ] **Step 4: Run the seam tests + the touched engine suites**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/execution/test_pde_session_seams.py -q
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q -k "snowball_pde or phoenix_pde or pde_event or spot_greeks"
```
Expected: PASS (plus the pre-existing quad golden failure if that -k catches it — it does not).

- [ ] **Step 5: Commit** — `feat(pde): session output seams — price/price_with_events/grid-projection on one implementation`

---

### Task 2: Injection seams and eager factorization packs

**Files:**
- Modify: `quantark/asset/equity/engine/pde/base_pde_solver.py`
- Modify: `quantark/asset/equity/engine/pde/snowball_pde_solver.py`
- Test: append to `test/execution/test_pde_session_seams.py`

**Interfaces:**
- Produces: `BasePDESolver` attributes `_session_grids`, `_session_step_coefficients`, `_session_matrix_pack`, `_session_banded_pack` (all default `None`, initialized in `__init__`); `_step_coefficients_for_solve(pricing_env, ref_strike, t_vec, dx_vec, num_x)`; `_session_factorization_packs(product, pricing_env, grids) -> (dict, dict)` (matrix pack keyed `(coeff_key, round(dt,12), round(theta,6))`, banded pack keyed `(coeff_key, round(dt,12), round(theta,12))`).

- [ ] **Step 1: Write failing tests** (append)

```python
class TestInjectionSeams:
    def test_injected_artifacts_reproduce_solve_bitwise(self):
        engine, product, env = _snowball_case()
        baseline = engine.price(product, env)

        clone = type(engine)(params=engine.params)
        spot = env.spot
        tau = product.get_maturity(env)
        strike = product.strike
        r, q = env.get_rate(tau), env.get_div_yield(tau)
        sigma = env.get_vol(strike, tau)
        grids = clone._build_grids(product, env, spot, sigma, tau, r, q)
        coeffs = clone._build_step_coefficients(
            env, strike, grids[3], grids[2], len(grids[0])
        )
        matrix_pack, banded_pack = clone._session_factorization_packs(
            product, env, grids
        )
        clone._session_grids = grids
        clone._session_step_coefficients = coeffs
        clone._session_matrix_pack = matrix_pack
        clone._session_banded_pack = banded_pack
        assert clone.price(product, env) == baseline

    def test_pack_covers_every_step_key(self):
        """The eager pack enumerates exactly the (coeff_key, dt, theta) keys
        the backward march will request — no lazy build during the solve."""
        engine, product, env = _snowball_case()
        clone = type(engine)(params=engine.params)
        # build grids/packs as above, inject, then monkeypatch
        # scipy.sparse.linalg.splu and scipy.linalg.solve_banded wrappers:
        # after injection, _get_matrices/_get_banded_system must not build.
        # Implement by counting engine._matrix_cache/_banded_cache growth:
        ...  # see step 3 for the exact assertion helpers
```

(Write the second test concretely at implementation time: inject packs, run `clone.price`, assert `len(clone._matrix_cache) == 0` and `len(clone._banded_cache) == 0` after the solve — every hit came from the read-only pack.)

- [ ] **Step 2: Run, verify failure** (attributes missing).

- [ ] **Step 3: Implement**

`BasePDESolver.__init__` additions:

```python
# Session-injected preparation (adapter-owned clones only; spec section 9.2).
self._session_grids = None
self._session_step_coefficients = None
self._session_matrix_pack = None
```

`_build_grids` — first lines:

```python
if self._session_grids is not None:
    return self._session_grids
```

New dispatch wrapper + call-site changes (base `_solve` line ~581 and snowball `_solve` line ~252 call `_step_coefficients_for_solve(...)` instead of `_build_step_coefficients(...)`; the injected value is PRE-flat-exact, and `_flat_exact_step_coefficients` stays applied by `_solve` in both paths — it is deterministic and idempotent for `n_unique == 1`):

```python
def _step_coefficients_for_solve(self, pricing_env, ref_strike, t_vec, dx_vec, num_x):
    if self._session_step_coefficients is not None:
        return self._session_step_coefficients
    return self._build_step_coefficients(pricing_env, ref_strike, t_vec, dx_vec, num_x)
```

`_get_matrices` — consult the read-only pack before the per-solve cache (misses fall through and populate `_matrix_cache` locally, never the pack):

```python
key = (coeff_key, round(dt, 12), round(theta, 6))
pack = self._session_matrix_pack
if pack is not None:
    hit = pack.get(key)
    if hit is not None:
        return hit
```

`SnowballPDESolver.__init__`: `self._session_banded_pack = None`; `_get_banded_system` gets the same read-only-pack head with key `(coeff_key, round(dt, 12), round(theta, 12))`.

`_session_factorization_packs` on `BasePDESolver` — eagerly runs the SAME enumeration `_time_stepping` performs, through the SAME builders, into fresh dicts:

```python
def _session_factorization_packs(self, product, pricing_env, grids):
    """Eagerly build the (coeff_key, dt, theta) -> matrices maps one solve
    would lazily build (spec section 9.2 factorization legality: the key
    derives from identical coefficients, dt, theta, and scheme params)."""
    x_vec, s_vec, dx_vec, t_vec, dt_vec = grids
    spot = pricing_env.spot
    tau = product.get_maturity(pricing_env)
    strike = getattr(product, "strike", spot)
    r = pricing_env.get_rate(tau)
    q = pricing_env.get_div_yield(tau)
    sigma = pricing_env.get_vol(strike, tau)
    num_x = len(x_vec)

    sc = self._step_coefficients_for_solve(pricing_env, strike, t_vec, dx_vec, num_x)
    sc = self._flat_exact_step_coefficients(sc, r, q, sigma, dx_vec, num_x)
    step_coeffs = None if sc.n_unique == 1 else sc
    l, c, u = self._calculate_coefficients(r, q, sigma, dx_vec, num_x)
    A = self._build_operator_matrix(l, c, u, num_x)

    theta_by_step = BackwardOperator.theta_by_step(
        np.asarray(t_vec), np.asarray(dt_vec), self.params,
        self._get_event_times(product, tau),
    )
    saved_matrix, self._matrix_cache = self._matrix_cache, {}
    saved_term_A = getattr(self, "_term_A_cache", {})
    self._term_A_cache = {}
    use_banded = self._pack_uses_banded(num_x)
    banded: dict = {}
    try:
        I_int = sp.eye(num_x - 2, format="csc")
        for j in range(len(t_vec) - 2, -1, -1):
            dt = dt_vec[j]
            theta = float(theta_by_step[j])
            if step_coeffs is not None:
                k = int(step_coeffs.set_index[j])
                l_j, _c_j, u_j = step_coeffs.lcu_sets[k]
            else:
                k = 0
                l_j, _c_j, u_j = l, c, u
            if use_banded:
                self._pack_banded_entry(banded, step_coeffs, l_j, _c_j, u_j,
                                        dt, theta, k)
            else:
                A_j = (self._operator_matrix_for_set(step_coeffs, k, num_x)
                       if step_coeffs is not None else A)
                self._get_matrices(I_int, A_j, dt, theta, coeff_key=k)
        matrix_pack = dict(self._matrix_cache)
    finally:
        self._matrix_cache = saved_matrix
        self._term_A_cache = saved_term_A
    return matrix_pack, banded
```

with two small hooks so the two-surface banded branch shares the machinery:

```python
def _pack_uses_banded(self, num_x) -> bool:
    return False  # base family always uses the sparse-splu path

def _pack_banded_entry(self, banded, step_coeffs, l, c, u, dt, theta, coeff_key):
    raise NotImplementedError  # only meaningful where _pack_uses_banded is True
```

`SnowballPDESolver` overrides (mirroring `_time_stepping_two_surface`'s `use_banded and n_int > 2` condition and `_get_banded_system`'s construction — route through `_get_banded_system` itself with a swapped `_banded_cache` so there is exactly one construction implementation):

```python
def _pack_uses_banded(self, num_x) -> bool:
    return bool(self.params.use_banded_solver) and (num_x - 2) > 2

def _pack_banded_entry(self, banded, step_coeffs, l, c, u, dt, theta, coeff_key):
    saved, self._banded_cache = self._banded_cache, OrderedDict()
    try:
        entry = self._get_banded_system(l, c, u, dt, theta, coeff_key=coeff_key)
        banded[(coeff_key, round(dt, 12), round(theta, 12))] = entry
    finally:
        self._banded_cache = saved
```

Caveat found in recon: `_get_banded_system` with cache strategy "disable" returns without caching — the pack builder result is still the returned tuple, so the pack works regardless; but when the engine's strategy is "disable" the ADAPTER skips artifact preparation entirely (Task 3), so this path only matters defensively.

- [ ] **Step 4: Run**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/execution/test_pde_session_seams.py -q
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q test/execution/ -x
```

- [ ] **Step 5: Commit** — `feat(pde): session injection seams + eager factorization packs`

---

### Task 3: Artifact states (`pde_session_prep.py`)

**Files:**
- Create: `quantark/asset/equity/engine/pde/pde_session_prep.py`
- Test: `test/execution/test_pde_artifact_reuse.py` (descriptor-level tests)

**Interfaces:**
- Produces:
  - `market_scalars(engine, product, env) -> (spot, strike, tau, r, q, sigma)` (the `_solve` preamble, shared),
  - `grid_state(engine, product, env, context) -> ArtifactState`,
  - `step_coefficients_state(engine, product, env, grids, grid_fp, extra_fp, context) -> ArtifactState`,
  - `factorization_state(engine, product, env, grids, coeff_fp, context) -> ArtifactState`,
  - `ArtifactState = NamedTuple(value, descriptor: Optional[ArtifactDescriptor], handle: Optional[ArtifactHandle], fingerprint: Optional[str])`.
- Consumes: `engine._grid_cache_key` (polymorphic — Snowball overrides it), `engine._build_grids`, `engine._build_step_coefficients`, `engine._session_factorization_packs`, `try_fingerprint`, `PreparedArtifactCache.get_or_build`.

Key rules (all must be implemented exactly):
- **Capture-and-reverify (plan-gate finding):** every state function captures its dependencies ONCE up front — the env field objects `(env.rate_curve, env.div_yield, env.vol_surface, env.spot_quote)` plus the derived scalars and the polymorphic `grid_cache_key` tuple — and the builder consumes ONLY the captures. After `get_or_build` returns, re-verify: (a) each captured env field is still the SAME object (`is`), and (b) recomputing the fingerprint from live state equals the descriptor fingerprint (for the grid state this means recomputing `engine._grid_cache_key(...)` and comparing). On mismatch: close the handle, `cache.invalidate_tags(<state's tags>)`, raise `DeterminismViolation` — exactly the Dupire-state contract. Add concurrent-replacement tests (dispatch, swap `env.rate_curve` mid-prepare via a monkeypatched builder hook, assert `DeterminismViolation`) for all three artifact kinds.
- If `engine._resolve_cache_strategy() == "disable"` → every state returns `(freshly built value, None, None, None)` (no session caching; direct path also builds uncached → bitwise).
- Grid fingerprint = `try_fingerprint(("pde-grid", engine_class_path, grid_cache_key))`. `None` (uncanonicalizable) → build fresh via `engine._build_grids(...)`, no descriptor.
- Coefficient fingerprint = `try_fingerprint(("pde-step-coeffs", engine_class_path, grid_cache_key, curves_fp_or_surface_fp))` where `curves_fp = try_fingerprint((env.rate_curve, env.div_yield, env.vol_surface))`; any `None` → fresh build.
- Factorization fingerprint = `try_fingerprint(("pde-fact", engine_class_path, grid_cache_key, coeff_fp))`; requires a non-None coefficient fingerprint.
- Dependency tags: grid `{"spot","product_terms","grid","vol_surface","rate_curve","dividend_curve","valuation_date"}`; coefficients add nothing (same set); factorization adds `"solver_policy"`.
- Byte sizes: grids/coefficients measured exactly (`sum(a.nbytes …)` walking the tuples); factorization pack ESTIMATE `n_keys * (num_x * 8 * 12) + (1 << 20)` with no `measure` (SuperLU opaque), banded pack measured exactly.
- Builder for the matrix/banded packs runs `engine._session_factorization_packs(...)` and wraps both dicts in `types.MappingProxyType` before publishing (immutable payload).

- [ ] **Step 1: Failing tests** — descriptor determinism (same env/product/engine → same fingerprint; bumped rate curve → different fingerprint), disable-strategy returns no descriptor, uncacheable env (a curve object monkeypatched to a non-dataclass) returns value with `None` descriptor and still prices bitwise when injected.
- [ ] **Step 2: Run, verify failure.**
- [ ] **Step 3: Implement the module** per the interface above (single `_cached_state(kind, fp, tags, builder, estimate, measure, context)` helper wrapping `get_or_build`, with the descriptor-vs-fresh branch in one place).
- [ ] **Step 4: Run the new tests.**
- [ ] **Step 5: Commit** — `feat(pde): grid/coefficient/factorization artifact states behind PreparedArtifactCache descriptors`

---

### Task 4: Shared Dupire artifact state (`execution/prep/dupire.py`)

**Files:**
- Create: `quantark/execution/prep/__init__.py`, `quantark/execution/prep/dupire.py`
- Modify: `quantark/asset/equity/engine/mc/dcn_execution_adapters.py`
- Test: existing `test/execution/test_dcn_batch_adapter.py` + `test_kernel_prepare.py` must pass unchanged

**Interfaces:**
- Produces: `dupire_surface_state(prebuilt, capture, spot, div_fn, context, estimate_bytes, builder) -> PreparedState` — the EXACT logic currently in `_DCNLocalVolAdapterBase._prepare_surface` (capture-once inputs, fingerprint, single-flight build, post-build identity/fingerprint re-verify raising `DeterminismViolation`, tag invalidation on violation), parameterized over the captured input tuple and the surface builder so equity-env and fx-env callers share it.
- `_DCNLocalVolAdapterBase._prepare_surface` becomes a thin call into it (identical observable behavior — same descriptor kind `"dupire-local-vol"`, same tags, same builder version).

- [ ] **Step 1: Move the logic; keep `dcn_execution_adapters.py` public names/behavior identical.** (`quantark.execution.prep` importing `quantark.volmodels.localvol` follows the existing precedent of `execution.cache.draws` importing `quantark.montecarlo`; the KERNEL still never imports asset code.)
- [ ] **Step 2: Run the Phase 1/2/3 DCN adapter suites**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q test/execution/test_dcn_batch_adapter.py test/execution/test_kernel_prepare.py test/execution/test_heston_dcn_batch.py
```
Expected: PASS with zero test edits.
- [ ] **Step 3: Commit** — `refactor(execution): extract shared Dupire surface artifact state`

---

### Task 5: Equity 1D prepared adapter + rich session outputs

**Files:**
- Create: `quantark/asset/equity/engine/pde/pde_execution_adapters.py`
- Modify: `quantark/execution/registry.py`
- Modify: `test/execution/test_registry.py`
- Test: `test/execution/test_pde_prepared_adapters.py`

**Interfaces:**
- Produces:
  - `PDESessionValue` frozen dataclass `(pv: float, event_stats, event_distribution, grid: tuple)` — the outcome `value` whenever outputs beyond `{PV}` are requested (there is NO legacy value shape to preserve there: the legacy adapter rejected such requests).
  - `EquityPDE1DSessionAdapter(LegacyPriceAdapter)` with `events_capable: bool` ctor flag; adapter_id `"equity-pde-prepared"`, version `"1"`.
  - request options read from `request.operation_options` pairs: `("grid_spot_levels", tuple_of_floats)` and `("event_streams", tuple_of_stream_names)` (converted to `frozenset` for `streams=`).

Adapter contract (implement exactly):
- `capabilities()`: operations `{PRICE, EVENT_STATS}` when events_capable else `{PRICE}`; output_kinds `{PV, GRID}` ∪ (`{EVENT_STATS}` if events_capable); serial-only; `instance_reentrant=False`; `peak_memory_estimate="conservative"`.
- `validate()`: PRICE outputs must be a subset of the capability output kinds (CapabilityError otherwise — fail closed, never silently drop); EVENT_STATS operation allowed only when events_capable; `pricing_env` required.
- `prepare(engine, request, context)`:
  ```python
  clone = self._clone_engine(engine)
  if request.operation not in (PRICE, EVENT_STATS): return PreparedState(payload=_Prepared(clone), ...)
  tau = request.product.get_maturity(request.pricing_env)   # guard errors → no artifacts
  if tau <= 0 → PreparedState(clone only)                    # short-circuit path needs no grids
  grid = grid_state(clone, product, env, context)
  clone._session_grids = grid.value                          # inject IMMEDIATELY
  coeff = step_coefficients_state(clone, product, env, grid.value, grid.fingerprint, curves_fp, context)
  clone._session_step_coefficients = coeff.value             # inject BEFORE factorization —
  # _session_factorization_packs goes through _step_coefficients_for_solve, which must
  # return the injected artifact rather than re-building (plan-gate finding: ordering)
  fact  = factorization_state(...) if coeff.fingerprint is not None else fresh-build-skip
  inject: clone._session_matrix_pack / (_session_banded_pack when present)
  handles = tuple of non-None artifact handles; on ANY exception close already-acquired handles, re-raise
  fingerprint = try_fingerprint(tuple of the three artifact fingerprints)
  ```
  When the engine strategy is "disable" or a fingerprint is None, the corresponding value is built fresh and injected anyway (bitwise; just uncached).
- `execute_native(engine, request, normalized, context, prepared)`:
  - PRICE, outputs == `{PV}` → `value = clone.price(product, env)`, economics `(("pv", float(value)),)`.
  - PRICE, richer outputs → `out = clone._session_outputs(product, env, want_events=…, want_grid=…, streams=…)`; if GRID requested and `out.solution is None` → `CapabilityError("grid projection unavailable for expired/knocked-out product")` (fail closed — Phase 3 lesson); `grid = tuple(clone._grid_projection_from_solution(out.solution, levels))` (each row a dict; tuple-of-dicts for the frozen dataclass); if EVENT_STATS requested and the engine returned `event_stats is None` → CapabilityError (product type does not support event stats). `value = PDESessionValue(...)`, economics `(("pv", float(out.npv)),)`.
  - EVENT_STATS operation → `super().execute_native(clone, request, normalized, context)` (legacy call shape on the prepared clone).
- `_clone_engine(self, engine)`: `type(engine)(params=engine.params)` — uniform across the 9 (verified in recon; Snowball's `enable_profiling` defaults False on the clone; session dispatch does not populate live-engine profile stats — document in the module docstring).

Registration (in `registry.py`, all `exact=True`): the 9 base-family class paths → `_equity_pde_1d_adapter` (events_capable=False factory) for the 6 vanilla/barrier/touch solvers and `_equity_pde_autocall_adapter` (events_capable=True) for Snowball/KOReset/Phoenix.

- [ ] **Step 1: Failing tests** (`test_pde_prepared_adapters.py`) — the load-bearing ones:

```python
class TestBitwiseParity:
    # parametrized over the 9 class names using matrix_fixtures builders:
    def test_session_price_bitwise(self, name):
        engine, product, env = CASES[name]()
        direct = engine.price(product, env)
        with PricingSession() as session:
            outcome = session.execute(engine, PricingRequest(product=product, pricing_env=env))
        assert outcome.value == direct
        assert outcome.manifest.adapter_id == "equity-pde-prepared"  # not legacy

class TestRichOutputs:
    def test_pv_events_grid_one_solve(self):
        engine, product, env = CASES["SnowballPDESolver"]()
        direct_events = engine.price_with_events(product, env)
        levels = tuple(float(env.spot) * m for m in (0.9, 1.0, 1.1))
        direct_curve = engine.calculate_spot_greeks_curve(product, env, list(levels))
        calls = []
        orig = SnowballPDESolver._solve
        # count value solves through the session (monkeypatch _solve)
        ...
        request = PricingRequest(
            product=product, pricing_env=env,
            outputs=frozenset({OutputKind.PV, OutputKind.EVENT_STATS, OutputKind.GRID}),
            operation_options=(("grid_spot_levels", levels),),
        )
        outcome = session.execute(engine, request)
        assert outcome.value.pv == direct_events.npv
        assert list(outcome.value.grid) == direct_curve
        # event distribution arrays bitwise vs direct
        # TOTAL backward-march parity (plan-gate finding): count EVERY
        # backward-core invocation — SnowballPDESolver._solve AND the
        # event-stat indicator sweep entry point AND scipy solve_banded —
        # during (a) direct price_with_events + calculate_spot_greeks_curve
        # done naively (two value solves + indicator sweep) and (b) the ONE
        # session dispatch. Assert: session value-solve count == 1 (vs 2
        # naive direct), and session TOTAL march count == direct
        # price_with_events march count (the indicator sweep is the engines'
        # designed separate pass — the session adds NO extra marches).

class TestFailClosed:
    def test_grid_on_expired_raises(self): ...          # CapabilityError, not silent pv-only
    def test_unsupported_output_rejected(self): ...     # CASHFLOWS -> CapabilityError at validate
    def test_events_on_vanilla_rejected(self): ...      # EuropeanPDESolver + EVENT_STATS output

class TestSubclassFallback:
    def test_price_override_honored_via_legacy(self):
        class FlatSnowballPDE(SnowballPDESolver):
            def price(self, product, env): return 123.0
        # session dispatch -> legacy adapter -> 123.0 (exact=True regression)
```

- [ ] **Step 2: Run, verify failures.**
- [ ] **Step 3: Implement adapter + registry entries + update `test_registry.py` expected set (+9 paths).**
- [ ] **Step 4: Run**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q test/execution/test_pde_prepared_adapters.py test/execution/test_registry.py test/execution/test_matrix_parity.py test/execution/test_session_parity.py
```
The matrix-parity suite now exercises the new adapter for these engines and must stay bitwise-green.
- [ ] **Step 5: Commit** — `feat(execution): prepared session adapter + rich outputs for the nine 1D equity PDE solvers`

---

### Task 6: LV adapters (vanilla, barrier, snowball, phoenix) + build-count gates

**Files:**
- Modify: `quantark/asset/equity/engine/pde/pde_execution_adapters.py`
- Modify: `quantark/execution/registry.py`, `test/execution/test_registry.py`
- Test: `test/execution/test_pde_artifact_reuse.py`

**Interfaces:**
- `EquityLVPDESessionAdapter` (vanilla + barrier LV): prepare = Dupire state (via `execution.prep.dupire`, capture `(env.vol_surface, env.spot_quote, env.rate_curve, env.div_yield)`, builder `build_dupire_local_vol` — same as DCN) → clone with the surface:
  - `LocalVolPDESolver`: `type(engine)(params=engine.params, local_vol_surface=surface)`
  - `LocalVolBarrierPDESolver`: same signature.
  PV-only outputs.
- `EquityLVAutocallablePDESessionAdapter`: Dupire state → clone (`LocalVolSnowballPDESolver(params=…, local_vol_surface=surface, enable_profiling=False)`; `LocalVolPhoenixPDESolver` — verify its exact ctor signature in the file before writing the clone) → THEN 1D artifact states on the clone with the surface active:
  ```python
  clone._active_lv_surface = surface        # builder context for coefficients
  clone._active_s_vec = grids[1]
  coeff = step_coefficients_state(clone, product, env, grids, grid_fp,
                                  extra_fp=surface_fingerprint, context)
  clone._active_lv_surface = None; clone._active_s_vec = None   # restore; _with_surface re-arms at execute
  ```
  **Ordering (plan-gate finding — cold-cache LV failure otherwise):** the LV surface context must stay active through BOTH the coefficient build AND the factorization build, and the coefficient artifact must be injected before `factorization_state` runs, because `_session_factorization_packs` → `_step_coefficients_for_solve` falls back to `_build_step_coefficients`, which raises `PricingError` when `_active_lv_surface`/`_active_s_vec` are `None`:
  ```python
  clone._active_lv_surface = surface
  clone._active_s_vec = grids[1]
  try:
      coeff = step_coefficients_state(...)
      clone._session_grids = grids
      clone._session_step_coefficients = coeff.value
      fact = factorization_state(...)          # reads the injected coefficients
      clone._session_matrix_pack, clone._session_banded_pack = fact.value
  finally:
      clone._active_lv_surface = None          # _with_surface re-arms at execute
      clone._active_s_vec = None
  ```
  Add a COLD-CACHE regression test: a fresh session's very first LV-snowball and LV-phoenix dispatch (nothing cached anywhere) must succeed and be bitwise-equal to direct.
  The coefficient/factorization fingerprints use the SURFACE fingerprint in place of `curves_fp` (plus `curves_fp` itself — LV coefficients also sample rates/carry). If the surface state has no fingerprint (prebuilt engine surface or uncanonicalizable env) → coefficients/factorization build fresh.
  Note: LV coefficients have `n_unique == n_steps`, so the factorization pack holds one entry per step — the byte ESTIMATE must scale with `n_steps` (`n_keys` already does).
- Respect `engine._prebuilt`: when the live engine was constructed with a prebuilt surface, reuse it directly (no descriptor), exactly like the DCN adapter.

Build-count gates (the Phase 4 exit-gate evidence — in `test_pde_artifact_reuse.py`):

```python
def test_dupire_built_once_across_crn_repricings(monkeypatch):
    calls = _count_calls(monkeypatch, "quantark.volmodels.localvol", "build_dupire_local_vol")
    with PricingSession() as session:
        for _ in range(3):
            outcome = session.execute(engine, request)   # LocalVolSnowballPDESolver
    assert calls() == 1

def test_grid_and_splu_built_once_across_crn_repricings(monkeypatch):
    SnowballPDESolver.clear_grid_cache()                  # cold class cache
    grid_calls = _count_calls(monkeypatch, SpatialGrid, "build")
    splu_calls = _count_calls(monkeypatch, spla, "splu")  # patch in base_pde_solver namespace
    ... 3 session dispatches ...
    assert grid_calls() == 1
    assert splu_calls() == n_unique_keys_once             # captured from the first dispatch

def test_bumped_env_never_reuses_stale_artifacts():
    # dispatch, then REPLACE env.rate_curve with a bumped curve, dispatch again:
    # session price(bumped) == direct price(bumped) bitwise (fresh descriptors)

def test_cache_bypass_stays_bitwise():
    # PricingSession with a tiny artifact_cache_bytes budget: artifacts bypass
    # admission; session price still == direct bitwise; no bytes retained after close
```

- [ ] Steps: failing tests → implement → register 4 paths (`exact=True`) → update expected set → run the reuse + adapter + matrix suites → commit `feat(execution): LV PDE prepared adapters with Dupire/grid/coefficient/factorization reuse`.

---

### Task 7: Heston/SLV 2D autocallable adapters + FX LV adapter

**Files:**
- Modify: `quantark/asset/equity/engine/pde/pde_execution_adapters.py` (2D adapter)
- Create: `quantark/asset/fx/engine/pde/fx_pde_execution_adapters.py`
- Modify: `quantark/execution/registry.py`, `test/execution/test_registry.py`
- Test: append to `test/execution/test_pde_prepared_adapters.py`

**Interfaces:**
- `Heston2DAutocallablePDESessionAdapter`: no artifacts (time-dependent ADI operators are rebuilt — record this in the class docstring with the §9.2 citation); clone + rich outputs `{PV, EVENT_STATS}` (never GRID — validate rejects it):
  ```python
  def _clone_engine(self, engine):
      kwargs = dict(model_params=engine.model_params, params=engine.params,
                    n_x=engine.n_x, n_v=engine.n_v, n_t=engine.n_t,
                    scheme=engine.scheme, grid_style=engine.grid_style,
                    grid_focus=engine.grid_focus,
                    pin_critical_spots=engine.pin_critical_spots)
      if isinstance(engine, (HestonSLVSnowballPDESolver, HestonSLVPhoenixPDESolver)):
          kwargs.update(leverage_surface=engine.leverage_surface, eta=engine.eta)
      return type(engine)(**kwargs)
  ```
  (One adapter class; the isinstance is over the four registered-exact classes only, so it cannot misfire on unknown subclasses.)
- `FxLVPDESessionAdapter` (in the fx module): Dupire state with FX capture `(fx_env.vol_surface, fx_env.domestic_curve, fx_env.foreign_curve)` + `spot = float(fx_env.effective_spot())`, builder mirroring `build_fx_local_vol`'s internals over the CAPTURED objects (`div_fn = lambda t: float(captured_foreign.get_rate(t))`); clone `FxLocalVolPDESolver(params=engine.params, grid_size=engine.grid_size, time_steps=engine.time_steps, theta=engine.theta, local_vol_surface=surface)`. PV-only. NOTE: `effective_spot()` may depend on `spot_days`/market forward — include the effective-spot float in the fingerprint tuple.
- Bitwise tests: session PRICE == direct for the 5; session PV+EVENT_STATS == direct `price_with_events` for the 4 Heston/SLV autocallables (npv + distribution arrays via `np.array_equal`); count parity — session dispatch performs the SAME number of `HestonSLVADICore.solve` calls as direct `price_with_events` (no extra solves), asserted with a monkeypatch counter.

- [ ] Steps: failing tests → implement both adapters → register 5 paths exact → run → commit `feat(execution): Heston/SLV autocallable + FX LV PDE session adapters`.

---

### Task 8: Inventory prepared-state exit gate

**Files:**
- Modify: `quantark/execution/inventory.py`
- Modify: `test/execution/test_inventory.py`

**Interfaces:**
- `PREPARED_STATES = ("prepared_capable", "temporary_legacy", "not_applicable")`.
- `InventoryRecord` gains `prepared_state: str = "not_applicable"`, `prepared_rationale: str = _PREPARED_NON_PDE` (helpers set real values). `validation_profile` (existing field) also set for prepared rows.
- Semantics: `prepared_capable` = resolves to an adapter that reuses preparation artifacts across requests via the session cache and/or serves one-solve rich outputs. MC rows: DCN LV MC = prepared_capable (Dupire artifact, Phase 1); all other MC rows `not_applicable` with `_PREPARED_MC = "no cross-request preparation artifacts; draw reuse is DrawRepository (Phase 2), adaptive/batch states cover execution"`.
- PDE rows: the 18 adapter engines + `LocalVolDCNPDEEngine` → `prepared_capable`, profile `bitwise-vs-direct/v1`; Heston/SLV vanilla+barrier PDE, `DCNPDEEngine`, `HestonDCNPDESolver`, FX Heston/SLV → `temporary_legacy` with `_PREPARED_ADI = "frozen construction-owned model inputs; time-dependent ADI/kernel operators cannot prove factorization reuse safe and are rebuilt (spec 9.2); no session artifacts"`; bond convertibles + facades → `temporary_legacy` with env-bound rationale; abstract bases → `not_applicable`.
- CI gates in `test_inventory.py`: every row's `prepared_state` in `PREPARED_STATES`; `prepared_capable` ⟹ non-empty `validation_profile` AND the resolved adapter id ≠ `"legacy-price"`; non-capable ⟹ non-empty rationale; the 18+2 prepared names asserted as an explicit set (same style as the batch/adaptive gates).

- [ ] Steps: failing gates → implement → run `test_inventory.py` + full `test/execution/` → commit `feat(execution): prepared_state inventory exit gate for Phase 4`.

---

### Task 9: Convergence + memory gates, benchmark, full suite

**Files:**
- Create: `test/execution/test_pde_convergence_gate.py`
- Create: `test/execution/benchmark_phase4.py`
- Modify: `quantark/execution/__init__.py` (docstring "Phases 0-4"), `docs` references if any

- [ ] **Convergence gate:** for `SnowballPDESolver` and `EuropeanPDESolver`, at production params AND refined params (`grid_size*2, time_steps*2`): session == direct bitwise at BOTH resolutions, AND both equal the **Task 0 pre-refactor goldens** bitwise (the independent oracle — this is what makes the gate non-tautological; plan-gate finding). The refinement delta therefore matches the pre-refactor refinement behavior exactly.
- [ ] **Memory gate:** after `session.close()`, the owned lease manager reports zero leased artifact bytes (inspect `cache.stats()["bytes_in_use"] == 0` before close is NOT required — pinned entries release with prepared handles; assert `bytes_in_use` does not grow across repeated dispatches beyond the artifact set, and close() releases everything). Factorization-pack admission failure (tiny budget) → bypass, bitwise unchanged.
- [ ] **Benchmark** (`benchmark_phase4.py`, mirroring `benchmark_phase2.py`'s opt-in conventions — read it first and copy its skip/marker pattern): (a) 10 CRN repricings of `LocalVolSnowballPDESolver` direct vs session (expect ≥2x from Dupire+grid+factorization reuse — report, don't hard-assert wall time in CI); (b) single-dispatch overhead `EuropeanPDESolver` session vs direct (report %, spec budget ≤3%).
- [ ] **Full suite:**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q
```
Expected: green except the known pre-existing `test_snowball_quad_flat_identity_golden`.
- [ ] **Commit** — `test(execution): Phase 4 convergence/memory gates + benchmarks`

---

## Plan-Gate Findings Applied (Codex adversarial review, 1 iteration, 2026-07-16)

1. **[high] LV factorization ordering** — coefficients injected before `factorization_state`; LV surface context held under try/finally through the factorization build; cold-cache LV regression test (Task 6).
2. **[high] Capture-and-reverify** — all three PDE artifact states capture dependencies once, build only from captures, and re-verify identity + fingerprint post-build, raising `DeterminismViolation` (Task 3).
3. **[high] Common-mode convergence gate** — Task 0 freezes pre-refactor goldens as the independent oracle; the Task 9 gate asserts against them, not only session-vs-refactored-direct.
4. **[medium] One-solve contract precision** — the guarantee is one VALUE solve plus the engines' designed indicator sweep; tests count total backward marches and assert session == direct `price_with_events` parity (Task 5).

## Self-Review Notes

- Spec §21 Phase 4 coverage: "adapt all 1D/2D PDE engines" → 18 prepared + 11 explicit-rationale rows (Task 8 makes the sweep checkable in CI); "grid/event-map/term-context/LV/coefficient/factorization reuse behind artifact descriptors" → Tasks 3/4/6 (event maps ride inside the grid artifact's event-aligned time grid; per-solve schedule resolution keeps engine-owned instance caches — products lack safe canonicalizers, the spec's own uncacheable rule); "one-solve PV/event/grid projections" → Tasks 1/5/7.
- Type consistency: `PDESessionOutputs` defined once in `base_pde_solver.py`; `PDESessionValue` once in `pde_execution_adapters.py`; artifact-state NamedTuple once in `pde_session_prep.py`.
- Known risks for the implementer: `matrix_fixtures` builder access patterns must be read, not assumed; `EventDistribution` field names must be read; `LocalVolPhoenixPDESolver.__init__` signature must be read; Snowball's `_grid_cache_key` override must be used polymorphically (never re-derive key material in the prep module).
