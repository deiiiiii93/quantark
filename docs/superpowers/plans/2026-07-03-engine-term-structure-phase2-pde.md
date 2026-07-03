# Engine Term-Structure Upgrade — Phase 2 (PDE) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every BSM PDE solver consumes term-structured r/q/vol through per-step operator coefficients, with flat inputs preserving today's factorization reuse bit-for-bit and term inputs paying a bounded (≤20%) rebuild cost.

**Architecture:** `BasePDESolver` gains a step-coefficient builder that samples Phase 0 `TermCoefficients` on the solver's own time grid and dedupes per-unique `(r_k, q_k, σ_k)` triples into `(l, c, u)` sets with a step→set index. The two stepping paths — the base sparse-LU path (`_time_stepping`/`_get_matrices`) and the snowball-family banded path (`_get_banded_system`) — extend their cache keys with the coefficient-set index. Flat inputs produce exactly one set (index 0), so cache behavior and numerics are identical to today.

**Tech Stack:** numpy/scipy banded + sparse LU, Phase 0 `quantark.priceenv.TermCoefficients`, `test/term_structure_benchmarks.py`.

**Spec:** `docs/superpowers/specs/2026-07-03-engine-term-structure-upgrade-design.md` (Component 4)

## Global Constraints

- Canonical `quantark.*` imports only; no MC imports inside PDE code (standing rule).
- **Identity on flat inputs:** the full suite passes unchanged after every task; flat envs produce ONE coefficient set so factorization caches behave identically. A fixed-config snowball PDE golden price is pinned before wiring and asserted after.
- **Per-step rebuild is designed behavior for term inputs** (forward params are NOT piecewise-constant under linear zero interpolation — spec). Budget: term-structured pricing within **20%** of flat-input time on the same grid; CI carries only a loose 2× smoke guard (timing tests flake), the 20% claim is measured by the Task 7 benchmark and recorded in the commit message.
- Grid geometry (`_build_grids`, spatial bounds) keeps using the cumulative-to-maturity scalars — same grids as today for flat inputs.
- Discounting inside the operator: the `-r` term in `c` uses that step's forward rate (curve-exact per interval).
- Vol reference strike per solver stays what it is today (`get_vol(strike, tau)`).
- Test runner: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest` from the worktree.
- Commit per task with the repo's Claude co-author line.

## Verified architecture anchors

- `BasePDESolver._calculate_coefficients(r, q, sigma, dx_vec, num_x)` — `base_pde_solver.py:915` (log-space `mu = r − q − σ²/2`, `D = σ²/2`, `−r` in `c`; uniform and non-uniform dx branches).
- Path A (sparse LU): `_solve` scalar extraction `base_pde_solver.py:505-527`, loop `_time_stepping` `base_pde_solver.py:962-1007` (`self._get_matrices(I_int, A, dt, theta)`, `_inject_boundary_contributions(rhs, grid, l, u, j, dt, theta)`, `self._matrix_cache`).
- Path B (banded): `SnowballPDESolver._get_banded_system(l, c, u, dt, theta)` `snowball_pde_solver.py:1670` with `(dt, theta)`-keyed `self._banded_cache`; pricing sweep coefficients `snowball_pde_solver.py:248`, event sweep `snowball_pde_solver.py:627`; `PhoenixPDESolver(SnowballPDESolver)` coefficients `phoenix_pde_solver.py:324`; `KOResetSnowballPDESolver(SnowballPDESolver)` coefficients `ko_reset_snowball_pde_solver.py:398`.
- Solver-specific scalar extractions: `european_pde_solver.py:132-133`, `barrier_pde_solver.py:432`, snowball `:182-185`, phoenix `:226`, ko_reset `:332`; heston carry `heston_pde_solver.py:59,83`.
- `BackwardOperator.theta_by_step` is coefficient-independent (depends on grid + params only) — unchanged.

---

### Task 1: Step-coefficient builder on `BasePDESolver`

**Files:**
- Modify: `quantark/asset/equity/engine/pde/base_pde_solver.py` (add two methods near `_calculate_coefficients` at :915)
- Test: `test/test_pde_term_coefficients.py` (new)

**Interfaces:**
- Consumes: `TermCoefficients.from_env(pricing_env, t_grid, ref_strike)` (Phase 0; t_grid must be the solver's full `t_vec` including 0 — verify `t_vec[0] == 0.0` in the test).
- Produces (later tasks rely on these exact names):

```python
class StepCoefficients(NamedTuple):
    lcu_sets: list      # [(l, c, u), ...] one per unique (r_k, q_k, sigma_k)
    set_index: np.ndarray  # shape (n_steps,) int — step j uses lcu_sets[set_index[j]]
    n_unique: int

def _build_step_coefficients(
    self, pricing_env, ref_strike, t_vec, dx_vec, num_x,
) -> StepCoefficients: ...
```

Dedupe key: `(round(r_k, 15), round(q_k, 15), round(sigma_k, 15))`. Flat env ⇒ `n_unique == 1` and `lcu_sets[0]` equals `_calculate_coefficients(r, q, sigma, ...)` exactly (same code path — implement by calling `_calculate_coefficients` per unique triple).

- [ ] **Step 1: Write the failing tests**

Create `test/test_pde_term_coefficients.py`:

```python
"""Per-step PDE operator coefficients from term structures."""
import numpy as np
import pytest

from term_structure_benchmarks import make_term_env

from quantark.asset.equity.engine.pde.european_pde_solver import EuropeanPDESolver


def _grids():
    # simple uniform grids: 5 time steps to 1y, 11 space nodes
    t_vec = np.linspace(0.0, 1.0, 6)
    dx_vec = np.full(10, 0.02)
    return t_vec, dx_vec, 11


def test_flat_env_single_unique_set_matches_scalar_path():
    solver = EuropeanPDESolver()
    env = make_term_env("flat")
    t_vec, dx_vec, num_x = _grids()
    sc = solver._build_step_coefficients(env, 100.0, t_vec, dx_vec, num_x)
    assert sc.n_unique == 1
    assert np.all(sc.set_index == 0)
    l, c, u = solver._calculate_coefficients(0.03, 0.01, 0.20, dx_vec, num_x)
    l2, c2, u2 = sc.lcu_sets[0]
    assert np.array_equal(l, l2) and np.array_equal(c, c2) and np.array_equal(u, u2)


def test_term_env_many_unique_sets_and_correct_per_step_values():
    solver = EuropeanPDESolver()
    env = make_term_env("kinked")
    t_vec, dx_vec, num_x = _grids()
    sc = solver._build_step_coefficients(env, 100.0, t_vec, dx_vec, num_x)
    assert sc.n_unique > 1
    assert sc.set_index.shape == (5,)
    # step j coefficients must equal _calculate_coefficients at the forward
    # triple for [t_j, t_{j+1}]
    from quantark.priceenv import TermCoefficients

    tc = TermCoefficients.from_env(env, t_vec, ref_strike=100.0)
    for j in range(5):
        l, c, u = solver._calculate_coefficients(
            float(tc.fwd_rates[j]), float(tc.fwd_carry[j]),
            float(tc.step_vols[j]), dx_vec, num_x,
        )
        l2, c2, u2 = sc.lcu_sets[int(sc.set_index[j])]
        assert np.array_equal(l, l2) and np.array_equal(c, c2) and np.array_equal(u, u2)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_pde_term_coefficients.py -n0 -q`
Expected: FAIL — `AttributeError: _build_step_coefficients`

- [ ] **Step 3: Implement**

In `base_pde_solver.py`, next to `_calculate_coefficients` (:915), add:

```python
class StepCoefficients(NamedTuple):
    """Per-step (l, c, u) operator coefficient sets, deduped by unique triple."""

    lcu_sets: list
    set_index: np.ndarray
    n_unique: int
```

(module level, `NamedTuple` from `typing` — extend the existing import) and the method:

```python
    def _build_step_coefficients(
        self,
        pricing_env: PricingEnvironment,
        ref_strike: float,
        t_vec: np.ndarray,
        dx_vec: np.ndarray,
        num_x: int,
    ) -> StepCoefficients:
        """Sample forward (r, q, sigma) per time step and build operator sets.

        Flat market inputs produce exactly ONE set (index 0 for every step),
        preserving today's single-operator factorization reuse. Term inputs
        generally produce one set per step — the designed per-step rebuild.
        """
        from quantark.priceenv.term_sampling import TermCoefficients

        tc = TermCoefficients.from_env(
            pricing_env, np.asarray(t_vec, dtype=float), ref_strike=float(ref_strike)
        )
        lcu_sets: list = []
        keys: dict = {}
        n_steps = len(t_vec) - 1
        set_index = np.zeros(n_steps, dtype=int)
        for j in range(n_steps):
            key = (
                round(float(tc.fwd_rates[j]), 15),
                round(float(tc.fwd_carry[j]), 15),
                round(float(tc.step_vols[j]), 15),
            )
            idx = keys.get(key)
            if idx is None:
                idx = len(lcu_sets)
                keys[key] = idx
                lcu_sets.append(
                    self._calculate_coefficients(key[0], key[1], key[2], dx_vec, num_x)
                )
            set_index[j] = idx
        return StepCoefficients(lcu_sets=lcu_sets, set_index=set_index, n_unique=len(lcu_sets))
```

Note the flat-path exactness subtlety: for a flat env, `fwd_rates[j]` may differ from the scalar `r` by ~1 ulp (DF round-trip). The flat identity test above compares against `_calculate_coefficients(0.03, 0.01, 0.20, ...)` — if it fails at bit level, compare with `pytest.approx(..., rel=1e-13)` instead AND add a fast-path: if `n_unique == 1` and the unique triple is within 1e-14 of the cumulative scalars, rebuild the single set from the exact scalars `(r, q, sigma)` passed by the caller. Prefer the fast-path (bit-exact flat identity is the Phase gate); implement it in Task 2/3 call sites by passing the scalars in:

```python
    # call-site pattern (Tasks 2-4): exact flat identity
    sc = self._build_step_coefficients(pricing_env, strike, t_vec, dx_vec, num_x)
    if sc.n_unique == 1:
        sc = StepCoefficients(
            lcu_sets=[self._calculate_coefficients(r, q, sigma, dx_vec, num_x)],
            set_index=sc.set_index,
            n_unique=1,
        )
```

Wait — that would silently replace a genuinely-term-but-constant curve with cumulative scalars; those coincide in that case (constant forward ⇒ forward == cumulative), so it is exact, not an approximation. Keep the substitution but ONLY when the unique triple matches the scalars within 1e-12 (assert, else keep the term set):

```python
    def _flat_exact_step_coefficients(self, sc, r, q, sigma, dx_vec, num_x):
        """When one unique set ~= the cumulative scalars, rebuild it from the
        exact scalars so flat envs are bit-identical to the pre-term code."""
        if sc.n_unique != 1:
            return sc
        # forward of a flat curve == the flat scalar (up to DF round-trip ulp)
        lcu = self._calculate_coefficients(r, q, sigma, dx_vec, num_x)
        return StepCoefficients(lcu_sets=[lcu], set_index=sc.set_index, n_unique=1)
```

(one unique set means the curve is constant on the grid; the cumulative scalars then equal the forwards mathematically, so substituting is exact by construction — add this as a base method used by all call sites.)

- [ ] **Step 4: Run tests** — both pass. Also `python -m pytest -q` full suite (no behavior change yet).

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/pde/base_pde_solver.py test/test_pde_term_coefficients.py
git commit -m "feat(pde): per-step operator coefficient sets from term structures"
```

---

### Task 2: Path A — base sparse-LU stepping consumes step coefficients

**Files:**
- Modify: `quantark/asset/equity/engine/pde/base_pde_solver.py` (`_solve` :505-560, `_time_stepping` :962-1007, `_get_matrices`, `_inject_boundary_contributions`)
- Test: `test/test_pde_term_structure_solvers.py` (new)

**Interfaces:**
- Consumes: Task 1 `StepCoefficients`, `_flat_exact_step_coefficients`.
- Produces: `_time_stepping(..., step_coeffs: Optional[StepCoefficients] = None)`; when provided, step `j` uses `lcu_sets[set_index[j]]` and its sparse `A_k` (built lazily per unique set); `_get_matrices` cache key gains the set index. All Path-A solvers (european, american, barrier, one_touch, double_barrier, double_one_touch) become term-aware through the shared `_solve`.

- [ ] **Step 1: Write the failing tests**

Create `test/test_pde_term_structure_solvers.py`:

```python
"""Term-structure tests for PDE solvers (spec test layers 2/4)."""
from datetime import datetime

import numpy as np
import pytest

from term_structure_benchmarks import make_term_env, reference_european_call_price

from quantark.asset.equity.engine.pde.european_pde_solver import EuropeanPDESolver
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType


def _collapsed_flat_env(env_term, maturity, ref_strike=100.0):
    T = float(maturity)
    return PricingEnvironment(
        rate_curve=FlatRateCurve(env_term.get_rate(T)),
        valuation_date=datetime(2026, 7, 3),
        spot_quote=SpotQuote(100.0),
        vol_surface=FlatVolSurface(env_term.get_vol(ref_strike, T)),
        div_yield=ContinuousDividendYield(
            max(-0.20, min(0.20, env_term.get_div_yield(T)))
        ),
    )


@pytest.mark.parametrize("shape", ["up", "down", "kinked"])
def test_european_pde_matches_term_benchmark(shape):
    """Deterministic solver: tight tolerance against the exact reference.

    NOTE: a European PDE price under a TERM structure differs from the
    collapsed-scalar price only through the path of coefficients; the
    terminal-value problem depends only on cumulative quantities, so the
    term and collapsed prices AGREE for a European payoff. The benchmark
    checks correctness of the per-step discretization, not discrimination.
    """
    env = make_term_env(shape)
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.5)
    px = EuropeanPDESolver().price(option, env)
    ref = reference_european_call_price(env, 100.0, 1.5)
    assert px == pytest.approx(ref, rel=2e-3)


def test_barrier_pde_sees_term_structure():
    """Path-dependent payoff: term vs collapsed must differ."""
    from quantark.asset.equity.engine.pde.barrier_pde_solver import BarrierPDESolver
    from quantark.asset.equity.product.option import BarrierOption
    from quantark.util.enum import BarrierType, ObservationType

    def price_fn(env):
        option = BarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            barrier=120.0,
            barrier_type=BarrierType.UP_OUT,
            maturity=1.0,
            rebate=0.0,
            observation_type=ObservationType.CONTINUOUS,
        )
        return BarrierPDESolver().price(option, env)

    env_term = make_term_env("kinked")
    px_term = price_fn(env_term)
    px_collapsed = price_fn(_collapsed_flat_env(env_term, 1.0))
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)


def test_american_pde_sees_term_structure():
    from quantark.asset.equity.engine.pde.american_pde_solver import (
        AmericanPDESolver,
    )
    from quantark.asset.equity.product.option import AmericanOption

    def price_fn(env):
        option = AmericanOption(100.0, OptionType.PUT, maturity=1.0)
        return AmericanPDESolver().price(option, env)

    env_term = make_term_env("kinked")
    px_term = price_fn(env_term)
    px_collapsed = price_fn(_collapsed_flat_env(env_term, 1.0))
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)
```

(Solver class names/constructors: resolve the exact exported names from each solver file's `class` statement or an existing test — e.g. `test/test_european_option.py`-adjacent PDE tests; do not invent.)

- [ ] **Step 2: Run — expect** the barrier/american discrimination tests FAIL (equal prices) and the European benchmark to PASS or nearly pass (Europeans depend only on cumulative values — if it fails badly, the wiring below fixes residual coefficient issues).

- [ ] **Step 3: Implement**

1. `_solve` (base :505): after grids are built, add

```python
        sc = self._build_step_coefficients(pricing_env, strike, t_vec, dx_vec, len(x_vec))
        sc = self._flat_exact_step_coefficients(sc, r, q, sigma, dx_vec, len(x_vec))
```

keep the existing single `(l, c, u)`/`A` for `sc.n_unique == 1` (pass `step_coeffs=None` → zero behavior change), else pass `step_coeffs=sc` to `_time_stepping` and use `sc.lcu_sets[sc.set_index[-1]]` for anything outside the loop that needs terminal-step coefficients.

2. `_time_stepping`: add `step_coeffs: Optional[StepCoefficients] = None`; inside the loop:

```python
        for j in range(num_t - 2, -1, -1):
            dt = dt_vec[j]
            theta = float(theta_by_step[j])
            if step_coeffs is not None:
                k = int(step_coeffs.set_index[j])
                l, c, u = step_coeffs.lcu_sets[k]
                A = self._operator_matrix_for_set(step_coeffs, k, num_x)
            else:
                k = 0
            M1, M2_lu = self._get_matrices(I_int, A, dt, theta, coeff_key=k)
```

with a small lazy per-set sparse-matrix cache:

```python
    def _operator_matrix_for_set(self, step_coeffs, k, num_x):
        cache = getattr(step_coeffs, "_A_cache", None)
        if cache is None:
            cache = {}
            object.__setattr__(step_coeffs, "_A_cache", cache)  # NamedTuple: use a dict on solver instead if this fails
        A = cache.get(k)
        if A is None:
            l, c, u = step_coeffs.lcu_sets[k]
            A = self._build_operator_matrix(l, c, u, num_x)
            cache[k] = A
        return A
```

NamedTuples reject attribute assignment — hold the cache on the solver instead: `self._term_A_cache: dict` cleared at the top of `_time_stepping`. Implement it that way (the snippet above shows intent; the solver-held dict is the actual mechanism).

3. `_get_matrices(...)`: extend its cache key (whatever `self._matrix_cache` keys on today — read the method) with `coeff_key` (default 0 keeps old keys equivalent).

4. `_inject_boundary_contributions(rhs, grid, l, u, j, dt, theta)` already receives `l, u` per call — pass the step's `l, u` when `step_coeffs` is set.

- [ ] **Step 4: Run** the new tests + full suite. Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/pde/base_pde_solver.py test/test_pde_term_structure_solvers.py
git commit -m "feat(pde): base sparse-LU stepping consumes per-step term coefficients"
```

---

### Task 3: Path B — snowball banded sweeps

**Files:**
- Modify: `quantark/asset/equity/engine/pde/snowball_pde_solver.py` (`_get_banded_system` :1670 — add `coeff_key: int = 0` to the cache key; pricing sweep :248 loop; event sweep :627 loop)
- Test: extend `test/test_pde_term_structure_solvers.py`

**Interfaces:**
- Consumes: Task 1 builder + flat-exact substitution.
- Produces: both snowball sweeps step-aware; `_get_banded_system(l, c, u, dt, theta, coeff_key=0)` cache key `(coeff_key, round(dt,12), round(theta,12))`.

- [ ] **Step 1: Capture the pre-wiring golden** (run BEFORE editing the solver):

```python
# scratch: golden capture (fixed grid config)
import sys; sys.path.insert(0, "test")
from term_structure_benchmarks import make_term_env
from quantark.asset.equity.engine.pde import SnowballPDESolver
# build the standard 1y snowball exactly as in test_mc_term_structure_engines._standard_snowball()
px = SnowballPDESolver().price(product, make_term_env("flat"))
print(repr(px))
```

Paste the printed value into the golden test below.

- [ ] **Step 2: Write the failing tests**

```python
def test_snowball_pde_sees_term_structure():
    from quantark.asset.equity.engine.pde import SnowballPDESolver
    from test_mc_term_structure_engines import _standard_snowball

    env_term = make_term_env("kinked")
    px_term = SnowballPDESolver().price(_standard_snowball(), env_term)
    px_collapsed = SnowballPDESolver().price(
        _standard_snowball(), _collapsed_flat_env(env_term, 1.0)
    )
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)


def test_snowball_pde_flat_identity_golden():
    from quantark.asset.equity.engine.pde import SnowballPDESolver
    from test_mc_term_structure_engines import _standard_snowball

    GOLDEN_PRE_UPGRADE = <paste from Step 1>
    px = SnowballPDESolver().price(_standard_snowball(), make_term_env("flat"))
    assert px == GOLDEN_PRE_UPGRADE  # bit-exact via _flat_exact_step_coefficients
```

(If importing the `_standard_snowball` helper across test modules is awkward under pytest rootdir rules, copy the product construction verbatim instead.)

- [ ] **Step 3: Implement.** In the pricing sweep (:248 site) and event sweep (:627 site): build `sc` via the Task 1/Task 2 call-site pattern right after `l, c, u = self._calculate_coefficients(...)`; inside each `for j ...` loop select `l, c, u = sc.lcu_sets[int(sc.set_index[j])]` and call `self._get_banded_system(l, c, u, dt, theta, coeff_key=int(sc.set_index[j]))`. `_get_banded_system` gains the `coeff_key` parameter included in both cache lookups (`key = (coeff_key, round(dt, 12), round(theta, 12))`). The boundary-row updates in the loop that use `l[1]` / `u[-2]` etc. must use the step's set.

  **Sparse fallback (codex plan-review finding):** the snowball family has a
  supported `use_banded_solver=False` branch (`_time_stepping_two_surface`
  style) that consumes a single sparse `A` + scalar `l, u` through
  `_get_matrices(I_int, A, dt, theta)`. Wire it the same way as Task 2's
  Path A: per-step set selection, per-set sparse `A` via the solver-held
  `_term_A_cache`, `coeff_key` into `_get_matrices`, step-local boundary
  contributions. Add a test variant that prices the snowball discrimination
  case with `PDEParams(use_banded_solver=False)` (resolve the exact param
  name from `PDEParams`) and asserts the same `px_term != px_collapsed`
  outcome AND that banded and sparse paths agree on the term env within the
  solver's own cross-path tolerance (copy from any existing banded-vs-sparse
  test if one exists, else `rel=1e-6`).

- [ ] **Step 4: Run** new tests + `test/test_snowball_option.py` + event-distribution suites (`test/test_ki_probability_definitions.py test/test_quad_event_stats_smoothing.py test/test_cashleg/`) + full suite. Expected: all PASS — the event sweep and pricing sweep share the same per-step schedule, so their reconciliation tests are the real gate here.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/pde/snowball_pde_solver.py test/test_pde_term_structure_solvers.py
git commit -m "feat(pde): snowball banded sweeps consume per-step term coefficients"
```

---

### Task 4: Path B — phoenix + KO-reset sweeps

**Files:**
- Modify: `quantark/asset/equity/engine/pde/phoenix_pde_solver.py` (:324 site + its loop), `quantark/asset/equity/engine/pde/ko_reset_snowball_pde_solver.py` (:398 site + its loop)
- Test: extend `test/test_pde_term_structure_solvers.py`

**Interfaces:** same call-site pattern as Task 3 (both inherit `_get_banded_system` from `SnowballPDESolver`, already `coeff_key`-aware after Task 3).

- [ ] **Step 1: Write failing discrimination tests** — phoenix (product construction copied from `test_mc_term_structure_engines.test_phoenix_mc_sees_term_structure`) and KO-reset (product via `create_ko_reset_snowball` from `quantark.asset.equity.product.option`, construction copied from `test/conftest.py` fixtures). Same `_collapsed_flat_env` pattern, `rel=1e-5`. Phoenix has the same banded/sparse split as snowball — cover its `use_banded_solver=False` branch exactly as in Task 3.

- [ ] **Step 2: Run — expect equal-price failures.**

- [ ] **Step 3: Implement** the Task 3 call-site pattern at `phoenix_pde_solver.py:324` and `ko_reset_snowball_pde_solver.py:398` and select per-step sets inside their loops.

- [ ] **Step 4: Run** new tests + `test/test_phoenix_option.py` + KO-reset suites + full suite. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/pde/phoenix_pde_solver.py quantark/asset/equity/engine/pde/ko_reset_snowball_pde_solver.py test/test_pde_term_structure_solvers.py
git commit -m "feat(pde): phoenix + KO-reset sweeps consume per-step term coefficients"
```

---

### Task 5: Solver-specific extractions and event/settlement discounting audit

**Files:**
- Modify: any of the 9 BSM solver files whose own code discounts cashflows or rebates with `exp(-r * t)` instead of the curve, and any `tau > 0 else 0.0` scalar extraction that bypasses the shared `_solve` (`european_pde_solver.py:132`, `barrier_pde_solver.py:432` are known anchors; sweep the family)
- Test: extend `test/test_pde_term_structure_solvers.py`

- [ ] **Step 1: Sweep** — `rg -n "exp\(-r|exp\(-rate" quantark/asset/equity/engine/pde/` and `rg -n "get_rate\(|get_div_yield\(" quantark/asset/equity/engine/pde/`. For every discounting hit: replace with `pricing_env.get_discount_factor(t)` (these sites have `pricing_env` in scope — PDE methods thread it). For scalar extractions feeding only grid geometry or near-expiry shortcuts: leave.
- [ ] **Step 2: One-touch discrimination test** (pay-at-hit rebates discount at hit times — curve-exactness matters):

```python
def test_one_touch_pde_sees_term_structure():
    from quantark.asset.equity.engine.pde.one_touch_pde_solver import (
        OneTouchPDESolver,
    )
    # product construction copied from the existing one-touch PDE test file
    ...
```

(Resolve the exact product/class names from `test/` one-touch tests; same `_collapsed_flat_env` pattern.)
- [ ] **Step 3: Run** family test files + full suite. PASS.
- [ ] **Step 4: Commit**

```bash
git add quantark/asset/equity/engine/pde/ test/test_pde_term_structure_solvers.py
git commit -m "feat(pde): curve discounting + term extraction across remaining BSM solvers"
```

---

### Task 6: Heston / Heston-SLV PDE — cumulative-equivalence convention

**Scope corrected by codex plan review:** the Heston PDE kernel prices a
EUROPEAN terminal-value problem, which depends on the carry/rate curves only
through the terminal forward and discount factor. Its existing cumulative
`get_rate(T)` / `get_div_yield(T)` inputs are therefore **already
term-structure exact** — the same convention as the analytical engines. A
term-vs-collapsed discrimination test would assert a false invariant. No
kernel change.

**Files:**
- Modify: `quantark/asset/equity/engine/pde/heston_pde_solver.py` (docstring note only, at the :59/:83 extraction sites), `docs/superpowers/specs/2026-07-03-engine-term-structure-upgrade-design.md` (inventory row: heston/heston_slv PDE disposition becomes "documented cumulative convention — no change", matching the analytical-engines row)
- Test: extend `test/test_pde_term_structure_solvers.py`

- [ ] **Step 1: Equivalence test** — Heston PDE on a term env equals Heston PDE on the collapsed cumulative env within solver tolerance (`rel=1e-8` — identical kernel inputs by construction), pinning the convention:

```python
def test_heston_pde_cumulative_convention_is_term_exact_for_europeans():
    # Heston params + product copied from the existing test/test_heston_pde*.py
    env_term = make_term_env("kinked")  # attach the Heston vol/model per that file
    ...
    assert px_term == pytest.approx(px_collapsed, rel=1e-8)
```

- [ ] **Step 2: Docstring notes** at the two extraction sites + spec inventory row update.
- [ ] **Step 3: Run heston test files + full suite; PASS.**
- [ ] **Step 4: Commit**

```bash
git add quantark/asset/equity/engine/pde/heston_pde_solver.py docs/superpowers/specs/2026-07-03-engine-term-structure-upgrade-design.md test/test_pde_term_structure_solvers.py
git commit -m "docs(pde): heston PDE cumulative-inputs convention is term-exact for europeans"
```

---

### Task 7: Performance gate + Phase 2 wrap

**Files:**
- Create: `example/pde_term_structure_benchmark.py` (manual benchmark script)
- Test: extend `test/test_pde_term_structure_solvers.py` (loose smoke guard)

- [ ] **Step 1: Benchmark script** — times the standard snowball PDE (same product/params as the golden test) on: (a) flat env, (b) term env with `LinearRateCurve` + `TermStructureDividendYield` + `TermStructureVolSurface` on 4 pillars with default solver time steps (pillars ≪ steps — the worst case for cache reuse). Prints a table of median-of-5 wall times and the term/flat ratio. Record the numbers in the Task 7 commit message. **Budget: ratio ≤ 1.20.** If exceeded: profile (`solver.enable_profiling(True)`) — the expected hotspot is per-step `_build`/factorization; acceptable mitigations are vectorizing `_calculate_coefficients` over steps or memoizing `banded` construction per set; do NOT quantize coefficients (that's an approximation — forbidden by default).
- [ ] **Step 2: CI smoke guard** (loose, non-flaky):

```python
def test_term_pde_not_pathologically_slower_than_flat():
    import time
    from quantark.asset.equity.engine.pde import SnowballPDESolver
    from test_mc_term_structure_engines import _standard_snowball

    def t(env):
        best = float("inf")
        for _ in range(3):
            s = time.perf_counter()
            SnowballPDESolver().price(_standard_snowball(), env)
            best = min(best, time.perf_counter() - s)
        return best

    flat, term = t(make_term_env("flat")), t(make_term_env("kinked"))
    assert term <= 2.0 * flat + 0.05  # generous: catches pathology, not noise
```

- [ ] **Step 3: Full suite** — green. Run the benchmark script and record: flat median, term median, ratio.
- [ ] **Step 4: Commit**

```bash
git add example/pde_term_structure_benchmark.py test/test_pde_term_structure_solvers.py
git commit -m "test(pde): Phase 2 gate — perf benchmark (flat vs term) + smoke guard"
```
