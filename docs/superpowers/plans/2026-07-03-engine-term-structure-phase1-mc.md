# Engine Term-Structure Upgrade — Phase 1 (MC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every constant-vol equity MC engine consumes term-structured r/q/vol through per-step arrays in `GBMPathGenerator` and curve discount factors, so bucketed carry/vol risk and futures-implied dividends produce correct multi-tenor sensitivities.

**Architecture:** `GBMPathGenerator` (the single shared path kernel all 11 BSM MC engines build) gains scalar-or-array `vol/rrf/div`; a small helper converts `PricingEnvironment` + the generator's own time grid into those arrays via the Phase 0 `TermCoefficients`; each engine swaps its scalar extraction for the helper and its `exp(-r·t)` discounting for curve DFs. Flat inputs produce constant arrays → results identical to today (the identity gate).

**Tech Stack:** numpy, Phase 0 `quantark.priceenv.TermCoefficients`, `test/term_structure_benchmarks.py` harness.

**Spec:** `docs/superpowers/specs/2026-07-03-engine-term-structure-upgrade-design.md` (Component 3)

## Global Constraints

- Canonical `quantark.*` imports only.
- **Identity on flat inputs:** the full existing suite must pass unchanged after every task. Unit-level: constant arrays must reproduce scalar results bit-identically in the generator (same ops, same values). Engine-level: forward-rate/step-vol recomputation may differ by ~1 ulp; existing tolerances absorb this.
- Discounting always from the rate curve (`pricing_env.get_discount_factor(t)` / `TermCoefficients.step_dfs`), never a freshly composed `exp(-r*t)` — with one deliberate exception: per-path CONSTANT-rate shortcuts that are part of a variance-reduction control variate stay as-is when they only affect variance, not the estimator's mean (none identified; flag if found).
- Vol reference strike: keep each engine's existing `get_vol(<ref>, T)` reference unchanged — the helper takes it as a parameter.
- Test runner: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest` from the worktree; `-n0` for single files.
- Commit per task with the repo's Claude co-author line.

## Engine inventory (scalar-extraction anchors, verified by grep)

| Engine | q-extraction anchor | Discounting pattern |
|--------|--------------------:|---------------------|
| `euro_mc_engine.py` | :149 | `math.exp(-r*T)` at :334, :369; lower bound `safe_exp(-r*T)` :200 |
| `digital_option_mc_engine.py` | :148 | same pattern as euro |
| `asian_option_mc_engine.py` | :184 | terminal DF |
| `barrier_option_mc_engine.py` | :147 | terminal DF |
| `range_accrual_mc_engine.py` | :183 | per-observation DF |
| `single_sharkfin_option_mc_engine.py` | :87 | terminal DF |
| `double_sharkfin_option_mc_engine.py` | :105 | terminal DF |
| `accumulator_mc_engine.py` | :108 | per-settlement DF |
| `snowball_mc_engine.py` | :212, :271, :460 | per-event DF (KO/coupon/maturity), scalars threaded through `_price_*` privates |
| `phoenix_mc_engine.py` | :141, :172 | per-coupon DF |
| `american_option_mc_engine.py` | :189 | per-step `safe_exp(-r*dt)` at :284, :351 |
| `sabr_mc_engine.py` | :55 | drift only (SABR vol dynamics unchanged) |

---

### Task 1: `GBMPathGenerator` accepts per-step arrays for vol / rrf / div

**Files:**
- Modify: `quantark/asset/equity/process/bsm/qmc_path_generator.py:82-300` (`GBMPathGenerator`)
- Test: `test/test_gbm_path_generator_term.py` (new)

**Interfaces:**
- Produces: `GBMPathGenerator(vol=..., rrf=..., div=...)` where each of the three accepts `float` OR a 1-D array of length `time_steps` (entry k applies over `[times[k], times[k+1]]`). Internally normalized to arrays `self._vol_vec`, `self._drift_vec` (length `time_steps`). `generate_paths` and `generate_terminal_values_qmc` honor them. Scalar inputs remain bit-identical to today.

- [ ] **Step 1: Write the failing tests**

Create `test/test_gbm_path_generator_term.py`:

```python
"""Per-step (term-structured) coefficients in GBMPathGenerator."""
import numpy as np
import pytest

from quantark.asset.equity.process.bsm.qmc_path_generator import GBMPathGenerator
from quantark.asset.equity.process.bsm.qmc_path_generator import (
    PseudoRandomNormalGenerator,
)


def _gen(vol, rrf, div, **kw):
    return GBMPathGenerator(
        initial_value=100.0, vol=vol, rrf=rrf, div=div, maturity=1.0,
        time_steps=12, num_paths=64,
        random_stream=PseudoRandomNormalGenerator(seed=7), **kw,
    )


def test_constant_arrays_bit_identical_to_scalars():
    scal, _ = _gen(0.2, 0.03, 0.01).generate_paths()
    n = 12
    arr, _ = _gen(np.full(n, 0.2), np.full(n, 0.03), np.full(n, 0.01)).generate_paths()
    assert np.array_equal(scal, arr)


def test_terminal_qmc_constant_arrays_bit_identical():
    n = 12
    a = _gen(0.2, 0.03, 0.01).generate_terminal_values_qmc()
    b = _gen(np.full(n, 0.2), np.full(n, 0.03), np.full(n, 0.01)).generate_terminal_values_qmc()
    assert np.array_equal(a, b)


def test_per_step_drift_reproduces_forward():
    """E[S_T] must equal S0 * exp(sum((r_k - q_k) dt_k)) exactly in expectation
    of the log: check the deterministic part by using zero vol epsilon paths."""
    n = 12
    rrf = np.linspace(0.01, 0.05, n)
    div = np.linspace(0.02, -0.01, n)
    g = _gen(1e-12, rrf, div)  # vanishing vol isolates the drift
    paths, _ = g.generate_paths()
    dt = g.dt_vector
    expected_T = 100.0 * np.exp(np.sum((rrf - div) * dt))
    assert paths[:, -1] == pytest.approx(expected_T, rel=1e-9)


def test_per_step_vol_scales_increments():
    """Two-step generator with vols [a, b]: log-increment stds scale as a, b."""
    n = 2
    g = _gen(np.array([0.1, 0.4]), 0.0, 0.0, )
    # rebuild with more paths for a stable std estimate
    g = GBMPathGenerator(
        initial_value=100.0, vol=np.array([0.1, 0.4]), rrf=0.0, div=0.0,
        maturity=1.0, time_steps=2, num_paths=200_000,
        random_stream=PseudoRandomNormalGenerator(seed=11),
    )
    paths, _ = g.generate_paths()
    logs = np.diff(np.log(paths), axis=1)
    stds = logs.std(axis=0, ddof=1) / np.sqrt(g.dt_vector)
    assert stds == pytest.approx([0.1, 0.4], rel=2e-2)


def test_array_length_mismatch_rejected():
    with pytest.raises(ValueError):
        _gen(np.full(5, 0.2), 0.03, 0.01)  # 5 != time_steps=12
    with pytest.raises(ValueError):
        _gen(0.2, np.full(5, 0.03), 0.01)


def test_negative_vol_entry_rejected():
    arr = np.full(12, 0.2); arr[3] = -0.1
    with pytest.raises(ValueError):
        _gen(arr, 0.03, 0.01)
```

If `PseudoRandomNormalGenerator` is not importable from `qmc_path_generator`, import it from where that module imports it (check its import block) and fix the test imports accordingly.

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_gbm_path_generator_term.py -n0 -q`
Expected: FAIL — array inputs break `__post_init__` scalar validation (`vol < 0.0` on an array raises "truth value ambiguous") or produce wrong shapes.

- [ ] **Step 3: Implement**

In `GBMPathGenerator`:

1. `__post_init__` — replace the scalar `vol` check and add normalization after the time grid is built (arrays need `time_steps`):

```python
        if self.initial_value <= 0.0:
            raise ValueError("initial_value must be positive")
        if self.num_paths <= 0:
            raise ValueError("num_paths must be positive")
        # ... (model / random_stream blocks unchanged) ...
        # Build time grid FIRST (moved above drift setup)
        self.times, self.dt_vector = _build_time_grid(
            maturity=self.maturity,
            time_steps=self.time_steps,
            dt_array=self.dt_array,
        )
        self._vol_vec = self._as_step_vector(self.vol, "vol")
        if np.any(self._vol_vec < 0.0):
            raise ValueError("vol must be non-negative")
        self._set_drift()
```

2. Add the normalizer method:

```python
    def _as_step_vector(self, value, name: str) -> np.ndarray:
        """Normalize a scalar-or-per-step input to shape (time_steps,)."""
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(self.time_steps, float(arr))
        if arr.shape != (self.time_steps,):
            raise ValueError(
                f"{name} must be scalar or shape ({self.time_steps},), got {arr.shape}"
            )
        return arr.copy()
```

3. `_set_drift` — vectorize:

```python
    def _set_drift(self) -> None:
        rrf_vec = self._as_step_vector(self.rrf, "rrf")
        div_vec = self._as_step_vector(self.div, "div")
        if self.model == "black":
            self._drift_vec = rrf_vec
        elif self.model == "bsm":
            self._drift_vec = rrf_vec - div_vec
        elif self.model == "gbm":
            self._drift_vec = np.zeros(self.time_steps)
        else:
            raise ValueError(f"Invalid model '{self.model}'")
        # Backward-compat scalar view for external readers (drift of step 0)
        self.drift = float(self._drift_vec[0])
```

4. `generate_paths` — swap scalars for vectors (bit-identical for constant vectors):

```python
        drift_term = (self._drift_vec - 0.5 * self._vol_vec * self._vol_vec) * self.dt_vector
        drift_term = drift_term.reshape(1, -1)
        diffusion_term = self._vol_vec.reshape(1, -1) * dW
```

5. `generate_terminal_values_qmc` — integrate the vectors (exact for terminal law):

```python
        total_drift = float(np.sum((self._drift_vec - 0.5 * self._vol_vec**2) * self.dt_vector))
        total_var = float(np.sum(self._vol_vec**2 * self.dt_vector))
        return self.initial_value * np.exp(total_drift + np.sqrt(total_var) * z_1d)
```

6. Check `GBMPathGeneratorQMC` (line 301) and any other subclass/reader of `self.vol` / `self.drift` inside this module: `rg -n "self\.vol|self\.drift" quantark/asset/equity/process/bsm/qmc_path_generator.py` — route every pricing-path use through `self._vol_vec` / `self._drift_vec`. External readers (e.g. control-variate code in engines or `quantark/montecarlo/qmc_variance_reduction.py`, `quantark/sacva/exposure/paths.py`): `rg -n "\.vol\b|\.drift\b" quantark/ --glob '!**/pycache/**'` — for each hit that assumes a scalar, either it receives a scalar-configured generator (unchanged behavior, fine) or it must switch to the vectors; do not leave a silent scalar read on the pricing path.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_gbm_path_generator_term.py -n0 -q`
Expected: all PASS

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q`
Expected: full suite PASS (scalar behavior unchanged)

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/process/bsm/qmc_path_generator.py test/test_gbm_path_generator_term.py
git commit -m "feat(mc): GBMPathGenerator accepts per-step vol/rrf/div arrays"
```

---

### Task 2: Shared engine helper — env → generator arrays

**Files:**
- Create: `quantark/asset/equity/engine/mc/term_inputs.py`
- Test: `test/test_mc_term_inputs.py` (new)

**Interfaces:**
- Consumes: `TermCoefficients.from_env` (Phase 0), `_build_time_grid` from `quantark.asset.equity.process.bsm.qmc_path_generator`.
- Produces:

```python
@dataclass(frozen=True)
class McTermInputs:
    times: np.ndarray      # (time_steps + 1,) grid including t=0
    rrf: np.ndarray        # (time_steps,) forward rates
    div: np.ndarray        # (time_steps,) forward carry
    vol: np.ndarray        # (time_steps,) step vols
    node_dfs: np.ndarray   # (time_steps + 1,) DF(0, t_i)

def build_mc_term_inputs(pricing_env, ref_strike, maturity, time_steps, dt_array=None) -> McTermInputs
def df_at(inputs: McTermInputs, t: float) -> float   # DF at grid time t (nearest node, tol 1e-12)
```

- [ ] **Step 1: Write the failing tests**

Create `test/test_mc_term_inputs.py`:

```python
"""Engine-facing term-input builder on the generator time grid."""
import numpy as np
import pytest

from quantark.asset.equity.engine.mc.term_inputs import (
    build_mc_term_inputs,
    df_at,
)
from term_structure_benchmarks import make_term_env


def test_flat_env_constant_arrays():
    env = make_term_env("flat")
    ti = build_mc_term_inputs(env, ref_strike=100.0, maturity=1.0, time_steps=12)
    assert ti.rrf == pytest.approx(np.full(12, 0.03), abs=1e-12)
    assert ti.div == pytest.approx(np.full(12, 0.01), abs=1e-12)
    assert ti.vol == pytest.approx(np.full(12, 0.20), abs=1e-12)
    assert ti.times.shape == (13,)
    assert ti.node_dfs[0] == 1.0


def test_term_env_forward_consistency():
    """Compounded step quantities must reproduce cumulative-to-T values."""
    env = make_term_env("kinked")
    T = 2.0
    ti = build_mc_term_inputs(env, ref_strike=100.0, maturity=T, time_steps=24)
    dt = np.diff(ti.times)
    assert float(np.sum(ti.rrf * dt)) == pytest.approx(env.get_rate(T) * T, rel=1e-10)
    assert float(np.sum(ti.div * dt)) == pytest.approx(env.get_div_yield(T) * T, rel=1e-10)
    assert float(np.sum(ti.vol**2 * dt)) == pytest.approx(
        env.get_vol(100.0, T) ** 2 * T, rel=1e-10
    )
    assert ti.node_dfs[-1] == pytest.approx(env.get_discount_factor(T), rel=1e-12)


def test_df_at_matches_nodes_and_rejects_off_grid():
    env = make_term_env("up")
    ti = build_mc_term_inputs(env, ref_strike=100.0, maturity=1.0, time_steps=4)
    assert df_at(ti, ti.times[2]) == pytest.approx(ti.node_dfs[2])
    with pytest.raises(ValueError):
        df_at(ti, 0.123456789)  # not a grid node
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_mc_term_inputs.py -n0 -q`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `quantark/asset/equity/engine/mc/term_inputs.py`:

```python
"""Bridge PricingEnvironment term structures onto MC generator time grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from quantark.asset.equity.process.bsm.qmc_path_generator import _build_time_grid
from quantark.priceenv.term_sampling import TermCoefficients


@dataclass(frozen=True)
class McTermInputs:
    """Per-step forward coefficients on an MC engine's time grid."""

    times: np.ndarray
    rrf: np.ndarray
    div: np.ndarray
    vol: np.ndarray
    node_dfs: np.ndarray


def build_mc_term_inputs(
    pricing_env,
    ref_strike: float,
    maturity: float,
    time_steps: int,
    dt_array: Optional[np.ndarray] = None,
) -> McTermInputs:
    """Sample forward r/q/vol and node DFs on the generator's exact grid.

    NOTE: _build_time_grid returns step-terminal times only, shape
    (time_steps,), NOT including t=0 (it is np.cumsum(dt)). TermCoefficients
    needs the full node grid, so prepend 0.0 — interval arrays then have
    length time_steps and node_dfs length time_steps + 1.
    """
    times, _ = _build_time_grid(
        maturity=maturity, time_steps=time_steps, dt_array=dt_array
    )
    grid = np.concatenate(([0.0], np.asarray(times, dtype=float)))
    tc = TermCoefficients.from_env(pricing_env, grid, ref_strike=ref_strike)
    return McTermInputs(
        times=tc.t_grid,
        rrf=tc.fwd_rates,
        div=tc.fwd_carry,
        vol=tc.step_vols,
        node_dfs=tc.node_dfs,
    )


def df_at(inputs: McTermInputs, t: float) -> float:
    """Discount factor at a grid node time; rejects off-grid times."""
    idx = int(np.argmin(np.abs(inputs.times - float(t))))
    if abs(float(inputs.times[idx]) - float(t)) > 1e-12:
        raise ValueError(f"t={t} is not a node of the MC time grid")
    return float(inputs.node_dfs[idx])
```

If `_build_time_grid` returns `(times, dt_vector)` with `times` of length `time_steps + 1` (verify at its definition in `qmc_path_generator.py`), the shapes line up; adjust unpacking to its actual signature if not.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_mc_term_inputs.py -n0 -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/mc/term_inputs.py test/test_mc_term_inputs.py
git commit -m "feat(mc): McTermInputs bridge from PricingEnvironment to generator grids"
```

---

### Task 3: Terminal-payoff engines — euro + digital

**Files:**
- Modify: `quantark/asset/equity/engine/mc/euro_mc_engine.py` (q at :149; DFs at :200, :334, :369; `_create_path_generator` at :220)
- Modify: `quantark/asset/equity/engine/mc/digital_option_mc_engine.py` (q at :148; same patterns)
- Test: `test/test_mc_term_structure_engines.py` (new)

**Interfaces:**
- Consumes: `build_mc_term_inputs`, `df_at` (Task 2); array-capable `GBMPathGenerator` (Task 1); `make_term_env`, `reference_european_call_price` from `test/term_structure_benchmarks.py`.
- Produces: the wiring pattern every later engine task repeats (shown fully here):

```python
# BEFORE (scalar extraction, engine price()):
r = pricing_env.get_rate(T)
q = pricing_env.get_div_yield(T)
sigma = pricing_env.get_vol(<ref>, T)

# AFTER:
term = build_mc_term_inputs(
    pricing_env, ref_strike=<ref>, maturity=T,
    time_steps=self.params.time_steps,
)
r = pricing_env.get_rate(T)          # keep scalars ONLY where near-expiry
q = pricing_env.get_div_yield(T)     # shortcuts / validation messages need them
sigma = pricing_env.get_vol(<ref>, T)

# generator construction: pass arrays
generator = GBMPathGenerator(..., vol=term.vol, rrf=term.rrf, div=term.div, ...)

# discounting: replace math.exp(-r * T) with
discount_factor = pricing_env.get_discount_factor(T)
# per-event discounting at grid time t_i: df_at(term, t_i)
```

- [ ] **Step 1: Write the failing benchmark tests**

Create `test/test_mc_term_structure_engines.py`:

```python
"""Term-structure benchmark tests for upgraded MC engines (spec test layer 2/3)."""
import numpy as np
import pytest

from term_structure_benchmarks import make_term_env, reference_european_call_price

from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanOptionMCEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.util.enum import OptionType


@pytest.mark.parametrize("shape", ["up", "down", "kinked"])
def test_euro_mc_matches_term_benchmark(shape):
    env = make_term_env(shape)
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.5)
    engine = EuropeanOptionMCEngine()  # default params; seed fixed in MCParams
    px = engine.price(option, env)
    ref = reference_european_call_price(env, 100.0, 1.5)
    # MC error at default paths: assert within 3 standard errors ~ generous 1%
    assert px == pytest.approx(ref, rel=1e-2)


def test_euro_mc_flat_env_still_matches_reference():
    env = make_term_env("flat")
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.0)
    px = EuropeanOptionMCEngine().price(option, env)
    assert px == pytest.approx(reference_european_call_price(env, 100.0, 1.0), rel=1e-2)


@pytest.mark.parametrize("shape", ["up", "kinked"])
def test_euro_mc_forward_reproduction(shape):
    """Deep ITM call ~ forward: PV(call K→0) = DF*(F - K). Use K=1e-6 proxy via
    put-call parity instead: C - P = DF*(F - K) exactly in the model."""
    env = make_term_env(shape)
    K, T = 100.0, 2.0
    call = EuropeanVanillaOption(K, OptionType.CALL, maturity=T)
    put = EuropeanVanillaOption(K, OptionType.PUT, maturity=T)
    engine = EuropeanOptionMCEngine()
    c, p = engine.price(call, env), engine.price(put, env)
    df = env.get_discount_factor(T)
    fwd = env.spot * np.exp((env.get_rate(T) - env.get_div_yield(T)) * T)
    assert c - p == pytest.approx(df * (fwd - K), rel=2e-2, abs=0.15)
```

Adjust the engine class name/constructor if it differs — resolve from the engine file's own `__init__` docstring block at :43-:49 (e.g. presets using `MCParams`). Use the exact exported class name.

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_mc_term_structure_engines.py -n0 -q`
Expected: FAIL — term shapes misprice (engine still collapses the curve to `q(T)`, `r(T)`, flat drift): `up/down/kinked` benchmark and forward tests off by more than tolerance. (`flat` test may already pass — that is expected.)

- [ ] **Step 3: Wire the two engines**

In each engine's `price()` (and any `_price_mc_or_qmc` / `_price_rqmc` privates):
1. After the scalar extraction, add the `build_mc_term_inputs(...)` call (pattern above), importing `from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs`.
2. Thread `term` down to `_create_path_generator` and pass `vol=term.vol, rrf=term.rrf, div=term.div` to `GBMPathGenerator` (keep the scalar parameters in the private-method signatures — add `term` as an extra keyword argument — so diffs stay small).
3. Replace `math.exp(-r * T)` / `safe_exp(-r * T)` discounting (euro :334, :369; lower bound :200; digital equivalents) with `pricing_env.get_discount_factor(T)` — thread `pricing_env` if not already in scope.
4. `generate_terminal_values_qmc` callers need no change (Task 1 made it term-aware).

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_mc_term_structure_engines.py test/test_european_option.py test/test_digital_option.py -n0 -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/mc/euro_mc_engine.py quantark/asset/equity/engine/mc/digital_option_mc_engine.py test/test_mc_term_structure_engines.py
git commit -m "feat(mc): euro + digital engines consume term-structured r/q/vol"
```

---

### Task 4: Path-payoff engines — asian, barrier, range accrual, sharkfin ×2, accumulator

**Files:**
- Modify: `asian_option_mc_engine.py` (:184), `barrier_option_mc_engine.py` (:147), `range_accrual_mc_engine.py` (:183), `single_sharkfin_option_mc_engine.py` (:87), `double_sharkfin_option_mc_engine.py` (:105), `accumulator_mc_engine.py` (:108) — all under `quantark/asset/equity/engine/mc/`
- Test: extend `test/test_mc_term_structure_engines.py`

**Interfaces:**
- Consumes: exactly the Task 3 wiring pattern (repeated per engine): `build_mc_term_inputs` → array generator → curve DFs (`pricing_env.get_discount_factor(T)` for terminal payoffs; `df_at(term, t_i)` for per-observation/settlement cashflows at grid times; for cashflow dates NOT on the generator grid, use `pricing_env.get_discount_factor(t)` directly).

- [ ] **Step 1: Write the failing tests**

Append to `test/test_mc_term_structure_engines.py` — one term-vs-flat discrimination test per engine. The pattern (asian shown; repeat for each engine with its product constructor resolved from that engine's existing test file, e.g. `test/test_asian_option.py`):

```python
def _term_sensitivity_check(price_fn, maturity):
    """An upgraded engine must price 'up' and 'down' term shapes differently
    even though both have the same cumulative-to-maturity endpoint values at
    the product maturity ONLY when shapes differ at intermediates; use shapes
    whose q(T)/r(T)/vol(T) at T=2.0 coincide by construction? They don't —
    so instead assert the engine's term price differs from the price under a
    flat env matched to the cumulative-to-T scalars (the old collapse)."""
    env_term = make_term_env("kinked")
    px_term = price_fn(env_term)

    # Flat env matched to cumulative values at T: what the OLD engine computed
    from datetime import datetime
    from quantark.param import (
        ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote,
    )
    from quantark.priceenv import PricingEnvironment
    T = float(maturity)  # MUST be the product's actual pricing maturity —
    # a mismatched T lets an un-wired engine "pass" via scalar endpoint
    # differences (codex plan-review finding)
    env_flat = PricingEnvironment(
        rate_curve=FlatRateCurve(env_term.get_rate(T)),
        valuation_date=datetime(2026, 7, 3),
        spot_quote=SpotQuote(100.0),
        vol_surface=FlatVolSurface(env_term.get_vol(100.0, T)),
        div_yield=ContinuousDividendYield(
            max(-0.20, min(0.20, env_term.get_div_yield(T)))
        ),
    )
    px_collapsed = price_fn(env_flat)
    assert px_term != pytest.approx(px_collapsed, rel=1e-4)
    return px_term, px_collapsed
```

Then per engine (constructors copied from each engine's existing test file):

```python
def test_asian_mc_sees_term_structure():
    from quantark.asset.equity.engine.mc.asian_option_mc_engine import AsianOptionMCEngine
    from quantark.asset.equity.product.option import AsianOption  # resolve exact import from test_asian_option.py

    def price_fn(env):
        option = AsianOption(strike=100.0, maturity=2.0, option_type=OptionType.CALL)  # resolve exact ctor
        return AsianOptionMCEngine().price(option, env)

    _term_sensitivity_check(price_fn, maturity=2.0)  # maturity == product's
```

(One such test per engine: asian, barrier, range accrual, single sharkfin, double sharkfin, accumulator. For each, copy the product construction from its existing test file — `test/test_barrier_option.py`, `test/test_range_accrual.py`, `test/test_sharkfin*.py`, `test/test_accumulator*.py` — with maturity stretched to 2.0 where the product allows, else use its test default maturity and accept the shapes still differ at that horizon.)

- [ ] **Step 2: Run to verify failures**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_mc_term_structure_engines.py -n0 -q -k "sees_term_structure"`
Expected: each new test FAILS with `px_term == px_collapsed` (the un-wired engine literally computes the collapsed price).

- [ ] **Step 3: Wire the six engines**

Apply the Task 3 pattern at each anchor. Specifics:
- Terminal-DF engines (asian, barrier, sharkfin ×2): replace the single `exp(-r*T)` discount with `pricing_env.get_discount_factor(T)`.
- Per-observation engines (range accrual: coupon dates; accumulator: settlement dates): discount each dated cashflow with `pricing_env.get_discount_factor(t_cf)` (grid-independent, exact). Locate the discount expressions with `rg -n "exp\(-r" <engine file>` and replace every hit.
- Barrier engines may consume `aux["batch_id"]` for barrier corrections that use `sigma` — a barrier-crossing (Brownian-bridge) correction between steps must use that step's `term.vol[k]`, not the scalar; check `rg -n "sigma|vol" quantark/asset/equity/engine/mc/barrier_option_mc_engine.py` and route step-local corrections through the array.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_mc_term_structure_engines.py -n0 -q && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q`
Expected: new tests PASS; full suite PASS.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/mc/ test/test_mc_term_structure_engines.py
git commit -m "feat(mc): path-payoff engines consume term-structured r/q/vol"
```

---

### Task 5: Autocallables — snowball + phoenix

**Files:**
- Modify: `quantark/asset/equity/engine/mc/snowball_mc_engine.py` (scalar extraction :212, :271, :460; scalars threaded through `_price_snowball*`, `_price_ko_reset*` privates)
- Modify: `quantark/asset/equity/engine/mc/phoenix_mc_engine.py` (:141, :172)
- Test: extend `test/test_mc_term_structure_engines.py`

**Interfaces:**
- Consumes: Task 2/3 pattern. The autocallable payoff loops discount KO payments, coupons, and maturity legs at dated times — every such site switches to `pricing_env.get_discount_factor(t_event)`.

- [ ] **Step 1: Write the failing tests**

Append (product constructors copied from `test/test_snowball_option.py` / `test/test_phoenix_option.py`):

```python
def test_snowball_mc_sees_term_structure():
    # copy the standard 2Y snowball construction from test_snowball_option.py
    ...build product...
    def price_fn(env):
        return SnowballMCEngine().price(product, env)
    _term_sensitivity_check(price_fn)


def test_snowball_mc_flat_identity_vs_pre_upgrade_golden():
    """Flat env price with fixed seed must match the pre-upgrade value.
    Capture the golden number BEFORE wiring (run on main) and hard-code it."""
    env = make_term_env("flat")
    ...build product, fixed-seed MCParams...
    px = SnowballMCEngine(params=...).price(product, env)
    assert px == pytest.approx(GOLDEN_PRE_UPGRADE, rel=1e-12)


def test_phoenix_mc_sees_term_structure():
    ...same pattern...
```

The golden number: before Step 3, run the snowball pricing snippet on the current worktree state and paste the printed price into `GOLDEN_PRE_UPGRADE`.

- [ ] **Step 2: Run to verify failures**

`sees_term_structure` tests FAIL (equal prices); golden test PASSES already (it pins the pre-change value — it exists to prove Step 3 doesn't move flat pricing).

- [ ] **Step 3: Wire snowball + phoenix**

- Extend each private pricing method's signature with `term: McTermInputs` (keyword, after existing args) and pass `vol=term.vol, rrf=term.rrf, div=term.div` at every `GBMPathGenerator(` construction site (`rg -n "GBMPathGenerator\(" <file>`).
- Discounting: `rg -n "exp\(-r" <file>` — every KO/coupon/maturity discount `exp(-r * t)` becomes `pricing_env.get_discount_factor(t)`; `pricing_env` is already threaded through these methods (verified: signatures carry `pricing_env`).
- The near-expiry shortcut (`if T < 1e-10`) and `_validate_inputs` keep their scalars.
- KO-reset variant: same treatment inside `_price_ko_reset_*` (RQMC + parallel + serial paths).

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_mc_term_structure_engines.py test/test_snowball_option.py test/test_phoenix_option.py -n0 -q && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q`
Expected: all PASS, including the golden flat-identity pin.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/mc/snowball_mc_engine.py quantark/asset/equity/engine/mc/phoenix_mc_engine.py test/test_mc_term_structure_engines.py
git commit -m "feat(mc): snowball + phoenix autocallables consume term-structured r/q/vol"
```

---

### Task 6: American LSMC + SABR drift

**Files:**
- Modify: `quantark/asset/equity/engine/mc/american_option_mc_engine.py` (q at :189; per-step DFs `safe_exp(-r * generator.dt_vector)` at :284, :351)
- Modify: `quantark/asset/equity/engine/mc/sabr_mc_engine.py` (q at :55 — drift only)
- Test: extend `test/test_mc_term_structure_engines.py`

**Interfaces:**
- Consumes: Task 2/3 pattern. American per-step discounting uses the term step DFs: `discount_factors = term.node_dfs[1:] / term.node_dfs[:-1]` (length `time_steps`, exactly what the LSM backward loop consumes at :284/:351). SABR keeps its own vol dynamics; only the deterministic drift `(r - q)` becomes the per-step forward arrays.

- [ ] **Step 1: Write the failing tests**

```python
def test_american_mc_sees_term_structure():
    # copy product construction from test_american_option.py; put deep enough
    # ITM that early exercise matters, maturity 2.0
    ...
    _term_sensitivity_check(price_fn)


def test_american_mc_flat_matches_european_lower_bound():
    """American call on non-dividend flat env == European (no early exercise):
    cross-check against reference_european_call_price with q=0 flat env."""
    ...build flat env with div_yield=ContinuousDividendYield(0.0)...
    assert american_px == pytest.approx(reference_european_call_price(env, K, T), rel=2e-2)


def test_sabr_mc_drift_sees_term_structure():
    # SABR engine: same _term_sensitivity_check; vol dynamics stay SABR so use
    # shapes that differ in r/q only — build two envs sharing one vol surface.
    ...
```

- [ ] **Step 2: Run to verify failures** (`sees_term_structure` equal-price failures as before)

- [ ] **Step 3: Wire.** American: `build_mc_term_inputs` → array generator → `discount_factors = term.node_dfs[1:] / term.node_dfs[:-1]` replacing both `safe_exp(-r * generator.dt_vector)` sites. SABR: forward `r`/`q` arrays into its drift computation at the :55 extraction (check how the SABR path loop applies drift — `rg -n "drift|r - q|rrf" quantark/asset/equity/engine/mc/sabr_mc_engine.py` — and apply per-step values).

- [ ] **Step 4: Run** the new tests plus `test/test_american_option.py`, then the full suite. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/mc/american_option_mc_engine.py quantark/asset/equity/engine/mc/sabr_mc_engine.py test/test_mc_term_structure_engines.py
git commit -m "feat(mc): american LSMC per-step term DFs; sabr term drift"
```

---

### Task 7: Phase 1 gate — forward reproduction across engines + suite

**Files:**
- Test: extend `test/test_mc_term_structure_engines.py`

**Interfaces:**
- Consumes: everything above; `IndexFuturesCurve`-style implied carry is NOT built yet (that's the futures-carry feature) — this gate uses a hand-built `TermStructureDividendYield` implied from synthetic futures marks.

- [ ] **Step 1: Write the gate test**

```python
def test_forward_reproduction_from_synthetic_futures_marks():
    """Spec test layer 3: q(T) implied from futures marks; the engine's
    simulated forwards must reprice those marks at every node."""
    import numpy as np
    from datetime import datetime
    from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.param.div.dividend_yield import TermStructureDividendYield
    from quantark.priceenv import PricingEnvironment
    from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs
    from quantark.asset.equity.process.bsm.qmc_path_generator import GBMPathGenerator

    S, r = 100.0, 0.03
    node_times = [0.25, 0.5, 1.0, 2.0]
    marks = [100.3, 100.9, 101.5, 103.2]  # synthetic futures marks
    q_nodes = [r - np.log(f / S) / t for f, t in zip(marks, node_times)]
    env = PricingEnvironment(
        rate_curve=FlatRateCurve(r),
        valuation_date=datetime(2026, 7, 3),
        spot_quote=SpotQuote(S),
        vol_surface=FlatVolSurface(0.20),
        div_yield=TermStructureDividendYield(times=node_times, yields=q_nodes),
    )
    # grid with nodes exactly at the futures maturities
    dt_array = np.diff([0.0] + node_times)
    ti = build_mc_term_inputs(env, ref_strike=S, maturity=2.0,
                              time_steps=len(dt_array), dt_array=dt_array)
    dt = np.diff(ti.times)
    for i, (t_i, f_i) in enumerate(zip(node_times, marks)):
        model_fwd = S * np.exp(np.sum((ti.rrf[: i + 1] - ti.div[: i + 1]) * dt[: i + 1]))
        assert model_fwd == pytest.approx(f_i, rel=1e-10), f"node {t_i}"
```

- [ ] **Step 2: Run it** — should PASS immediately if Tasks 1–6 are correct (it validates the composition; a failure means a drift/carry sampling bug).

- [ ] **Step 3: Full-suite gate**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q`
Expected: full suite PASS.

- [ ] **Step 4: Commit**

```bash
git add test/test_mc_term_structure_engines.py
git commit -m "test(mc): Phase 1 gate — forward reproduction from synthetic futures marks"
```
