# Engine Term-Structure Upgrade — Phase 3 (QUAD) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every QUAD engine consumes term-structured r/q/vol as per-observation-interval forward parameters, closing the last engine family and enabling the cross-family MC/PDE/QUAD term agreement gate (spec test layer 4).

**Architecture:** `QuadratureCore` already accepts per-step `rate/div/vol` sequences (`_broadcast_param`, per-step `tau/alpha/beta`) — the kernel is term-ready. The work is engine wiring: a small helper samples Phase 0 `TermCoefficients` on each engine's observation grid; the adapter context carries arrays instead of scalars; the snowball-family's own recursion builds its per-step `tau/alpha/beta` from arrays (its consumers already index `tau[step_index]`); discounting switches to curve DFs. `european_quad` integrates a terminal density only, so its cumulative inputs are already term-exact — documented convention (the Heston-PDE argument).

**Tech Stack:** numpy/scipy, Phase 0 `TermCoefficients`, `test/term_structure_benchmarks.py`.

**Spec:** `docs/superpowers/specs/2026-07-03-engine-term-structure-upgrade-design.md` (Component 5)

## Global Constraints

- Canonical `quantark.*` imports; QUAD imports the env-facing helper from `quantark.priceenv.term_sampling` (never from `engine/mc/`).
- **Identity on flat inputs:** full suite passes unchanged after every task; snowball QUAD flat golden pinned pre-wiring, asserted post (exact-scalar substitution for single-unique-triple curves, as in Phase 2).
- KI-probability definitional gap (QUAD `ki_probability` = P(KI without prior KO)) remains **out of scope** — untouched.
- Discounting from the curve (`pricing_env.get_discount_factor(t)` / forward DFs), never freshly composed `exp(-r*t)`.
- Test runner: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest` from the worktree.
- Commit per task with the repo's Claude co-author line.

## Verified architecture anchors

- `QuadratureCore.__init__` (`quad_core.py:59-126`) accepts `rate/div/vol: float | Sequence[float]`; `_broadcast_param` (:127) normalizes to length `grid_t + 1` (0-prefixed); `tau/alpha/beta` are per-step arrays already. **Kernel needs no change.**
- `DiscreteQuadEngine.price` (`discrete_quad_engine.py:60-81`) builds the core from `QuadPricingContext(spot, maturity, rate, div, vol, contract_multiplier)` (`quad_adapters.py:43-51`, scalar fields); `_select_grid_points` already handles vector vol (`np.max`).
- Adapter scalar extractions: `quad_adapters.py:216-218` (barrier), `:513-515` (one-touch area), `:692-694`, `:993-996` — each `build_pricing_context` samples `get_rate/get_div_yield/get_vol` at maturity.
- Snowball family (own recursion on `QuadratureMath`, NOT `QuadratureCore`): `SnowballQuadEngine` extraction `snowball_quad_engine.py:119-121` (price) and `:428` area (event stats); per-step params built at `:206-211` and `:511-513` (`tau = 0.5*vol²*dt`, `alpha`, `beta` from SCALAR rate/div/vol × per-step `dt`); consumers index per step (`tau[step_index]` at :303, :601, :750). Discount sites: `math.exp(-rate * t)` :665, :843; `_ko_discount` :1201-1212 (`safe_exp(-rate * delay)`). BGK barrier shift uses vol (:934-1073, already samples `get_vol(barrier, t)` at :1068 for stability). `QuadratureMath(grid_x, spot, maturity, vol_max)` takes a max vol for grid sizing.
- `PhoenixQuadEngine(SnowballQuadEngine)` (`phoenix_quad_engine.py:26`, extraction :66-68), `KOResetSnowballQuadEngine(SnowballQuadEngine)` (`ko_reset_snowball_quad_engine.py:39`, extractions :76-78, :342-344).
- `european_quad_engine.py`: direct terminal-density integration (:124-126 extraction, :261 discount) — cumulative inputs term-exact for Europeans.

---

### Task 1: Env-facing helper reuse — `make_df_fn` in priceenv + QUAD term-param builder

**Files:**
- Modify: `quantark/priceenv/term_sampling.py` (add `make_df_fn`)
- Create: `quantark/asset/equity/engine/quad/term_inputs.py`
- Test: `test/test_quad_term_inputs.py` (new)

**Interfaces:**
- Produces:

```python
# quantark/priceenv/term_sampling.py
def make_df_fn(pricing_env):  # vectorized curve DFs: scalar->float, array->array
    ...

# quantark/asset/equity/engine/quad/term_inputs.py
@dataclass(frozen=True)
class QuadTermParams:
    rate: np.ndarray  # (n_obs,) forward rates, entry i covers (t_{i-1}, t_i]
    div: np.ndarray   # (n_obs,) forward carry
    vol: np.ndarray   # (n_obs,) step vols
    node_dfs: np.ndarray  # (n_obs + 1,) DF(0, t_i) incl. t=0

def build_quad_term_params(pricing_env, ref_strike, observation_times) -> QuadTermParams
```

Grid convention matches `QuadratureCore._broadcast_param(size == grid_t)`: arrays of length `n_obs` are 0-prefixed internally by the core, so `QuadTermParams.rate/div/vol` plug straight into the core's `rate=`/`div=`/`vol=`.

- [ ] **Step 1: Failing tests** (`test/test_quad_term_inputs.py`):

```python
"""QUAD engine term-parameter builder on observation grids."""
import numpy as np
import pytest

from term_structure_benchmarks import make_term_env

from quantark.asset.equity.engine.quad.term_inputs import build_quad_term_params
from quantark.priceenv.term_sampling import make_df_fn


def test_flat_env_constant_arrays():
    env = make_term_env("flat")
    tp = build_quad_term_params(env, 100.0, [0.25, 0.5, 0.75, 1.0])
    assert tp.rate == pytest.approx(np.full(4, 0.03), abs=1e-12)
    assert tp.div == pytest.approx(np.full(4, 0.01), abs=1e-12)
    assert tp.vol == pytest.approx(np.full(4, 0.20), abs=1e-12)
    assert tp.node_dfs.shape == (5,)


def test_term_env_reproduces_cumulative_quantities():
    env = make_term_env("kinked")
    obs = [0.25, 0.5, 0.75, 1.0]
    tp = build_quad_term_params(env, 100.0, obs)
    dt = np.diff(np.concatenate(([0.0], obs)))
    T = 1.0
    assert float(np.sum(tp.rate * dt)) == pytest.approx(env.get_rate(T) * T, rel=1e-10)
    assert float(np.sum(tp.div * dt)) == pytest.approx(
        env.get_div_yield(T) * T, rel=1e-10
    )
    assert float(np.sum(tp.vol**2 * dt)) == pytest.approx(
        env.get_vol(100.0, T) ** 2 * T, rel=1e-10
    )


def test_make_df_fn_scalar_and_array():
    env = make_term_env("up")
    df = make_df_fn(env)
    assert df(1.0) == pytest.approx(env.get_discount_factor(1.0), rel=1e-14)
    out = df(np.array([0.5, 1.0]))
    assert out.shape == (2,)
    assert out[1] == pytest.approx(env.get_discount_factor(1.0), rel=1e-14)
```

- [ ] **Step 2: Run — FAIL** (module missing).
- [ ] **Step 3: Implement.** `make_df_fn` in `priceenv/term_sampling.py` (same body as the MC one — scalar→float, array→elementwise). `build_quad_term_params`:

```python
"""Bridge PricingEnvironment term structures onto QUAD observation grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from quantark.priceenv.term_sampling import TermCoefficients


@dataclass(frozen=True)
class QuadTermParams:
    rate: np.ndarray
    div: np.ndarray
    vol: np.ndarray
    node_dfs: np.ndarray


def build_quad_term_params(
    pricing_env, ref_strike: float, observation_times: Sequence[float]
) -> QuadTermParams:
    grid = np.concatenate(([0.0], np.asarray(observation_times, dtype=float)))
    tc = TermCoefficients.from_env(pricing_env, grid, ref_strike=float(ref_strike))
    return QuadTermParams(
        rate=tc.fwd_rates, div=tc.fwd_carry, vol=tc.step_vols, node_dfs=tc.node_dfs
    )
```

Leave `engine/mc/term_inputs.make_df_fn` in place (re-point its body to the priceenv one via import alias to avoid duplication: `from quantark.priceenv.term_sampling import make_df_fn  # noqa: F401` replacing the local def — keep public name).
- [ ] **Step 4: Run new tests + `test/test_mc_term_inputs.py` + full suite region `-k quad`. PASS.**
- [ ] **Step 5: Commit** `feat(quad): term-param builder on observation grids; make_df_fn in priceenv`

---

### Task 2: DiscreteQuadEngine adapters — term context

**Files:**
- Modify: `quantark/asset/equity/engine/quad/quad_adapters.py` (`QuadPricingContext` :43-51 → `rate/div/vol: float | np.ndarray`; each adapter `build_pricing_context` at the four extraction sites; audit every scalar use of `context.rate/div/vol` — `rg -n "context\.(rate|div|vol)" quantark/asset/equity/engine/quad/` — rebate/early-price discounting switches to curve DFs via `make_df_fn(pricing_env)`)
- Modify: `quantark/asset/equity/engine/quad/discrete_quad_engine.py` (no core change; `_select_grid_points` already vector-ready — verify only)
- Test: `test/test_quad_term_structure_engines.py` (new)

**Interfaces:**
- Consumes: Task 1 `build_quad_term_params`, `make_df_fn`.
- Produces: the wiring pattern for adapter contexts:

```python
# in build_pricing_context, replacing scalar sampling:
tp = build_quad_term_params(pricing_env, ref_strike=<existing ref>, observation_times=<resolved obs times>)
context = QuadPricingContext(spot=..., maturity=..., rate=tp.rate, div=tp.div, vol=tp.vol, ...)
```

NOTE: if `build_pricing_context` runs before the observation schedule is resolved (check call order in `DiscreteQuadEngine.price` :60-66 — context is built FIRST, schedule resolved after), sample on the schedule inside `build_inputs`/`resolve_schedule` instead, or restructure so the context is enriched after `resolve_schedule`; keep scalars in the context for maturity-level uses and pass the arrays to the core construction site in `discrete_quad_engine.py:72-79` via a new context field `term_params: Optional[QuadTermParams]`. Choose the minimal-diff variant that keeps `QuadratureCore(rate=..., div=..., vol=...)` receiving arrays.

- [ ] **Step 1: Failing tests** — barrier + one-touch QUAD discrimination (products/configs copied from `test/test_one_touch_quad_engine.py:35-64` and the barrier adapter's existing test file; `_collapsed_flat_env` helper copied from `test/test_pde_term_structure_solvers.py:21-33`):

```python
"""Term-structure tests for QUAD engines (spec test layers 2/4)."""
from datetime import datetime

import numpy as np
import pytest

from term_structure_benchmarks import make_term_env, reference_european_call_price

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


def test_one_touch_quad_sees_term_structure():
    from quantark.asset.equity.engine.quad import OneTouchQuadEngine  # resolve exact export
    from quantark.asset.equity.param import QuadParams
    from quantark.asset.equity.product.option import OneTouchOption
    from quantark.util.enum import BarrierDirection, ObservationType, TouchType

    def price_fn(env):
        option = OneTouchOption(
            barrier=110.0, barrier_direction=BarrierDirection.UP, maturity=1.0,
            rebate=5.0, payment_at_hit=True, touch_type=TouchType.ONE_TOUCH,
            observation_type=ObservationType.DISCRETE,
            # discrete observation dates: copy the schedule pattern from the
            # existing one-touch QUAD test (monthly)
        )
        return OneTouchQuadEngine(params=QuadParams(grid_points=801)).price(option, env)

    env_term = make_term_env("kinked")
    assert price_fn(env_term) != pytest.approx(
        price_fn(_collapsed_flat_env(env_term, 1.0)), rel=1e-5
    )
```

(plus the barrier-adapter equivalent; resolve engine class names from `quantark/asset/equity/engine/quad/__init__.py` — do not invent.)
- [ ] **Step 2: Run — expect equal-price failures.**
- [ ] **Step 3: Wire adapters per the Interfaces note; audit `context.rate` scalar uses for rebate/settlement discounting → `make_df_fn`.**
- [ ] **Step 4: Run new tests + existing quad adapter suites + full suite. PASS.**
- [ ] **Step 5: Commit** `feat(quad): discrete-quad adapters pass per-interval term params to the core`

---

### Task 3: european_quad — documented cumulative convention

**Files:**
- Modify: `quantark/asset/equity/engine/quad/european_quad_engine.py` (:261 discount → `pricing_env.get_discount_factor(T)`; :168-169 lower bound → curve DF; docstring note at :124-126 that cumulative inputs are term-exact for the terminal-density integral)
- Test: extend `test/test_quad_term_structure_engines.py`

- [ ] **Step 1: Test** — european QUAD vs the exact term reference (cumulative convention makes it agree):

```python
@pytest.mark.parametrize("shape", ["up", "down", "kinked"])
def test_european_quad_matches_term_benchmark(shape):
    from quantark.asset.equity.engine.quad.european_quad_engine import (
        EuropeanQuadEngine,  # resolve exact name
    )
    from quantark.asset.equity.product.option import EuropeanVanillaOption

    env = make_term_env(shape)
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.5)
    px = EuropeanQuadEngine().price(option, env)
    assert px == pytest.approx(
        reference_european_call_price(env, 100.0, 1.5), rel=2e-3
    )
```

- [ ] **Step 2-4: Implement discount/docstring changes; run; commit** `feat(quad): european quad curve discounting + cumulative-convention note`

---

### Task 4: Snowball QUAD — per-step recursion parameters

**Files:**
- Modify: `quantark/asset/equity/engine/quad/snowball_quad_engine.py`:
  - extraction sites :119-121 (price) and :428 (event stats): build `tp = build_quad_term_params(pricing_env, product.strike, obs_times)` once the observation grid is known; keep scalars for validation/near-expiry.
  - per-step parameter builds :206-211 and :511-513: `tau = 0.5 * vol_vec**2 * dt`, `alpha`, `beta` from 0-prefixed per-step arrays (`np.concatenate(([x[0]], x))` to match the existing `dt` 0-prefix layout — mirror `QuadratureCore._broadcast_param`); with flat inputs these arrays are constant → exact-scalar substitution (Phase 2 pattern) keeps the golden bit-exact.
  - `QuadratureMath(vol_max=...)`: pass `float(np.max(vol_vec))`.
  - discounting :665, :843 → `pricing_env.get_discount_factor(t)`; `_ko_discount` :1201-1212 → forward curve DF between observation and settlement (`df(settle)/df(obs)` via `make_df_fn` — thread `pricing_env` or a df callable to its callers, `rg -n "_ko_discount" <file>`).
  - BGK shift: uses per-interval vol where it currently uses the scalar (`rg -n "vol" quantark/asset/equity/engine/quad/snowball_quad_engine.py` in the `_bgk_*` region :934-1073); the stability validator already samples `get_vol(barrier, t)` per node (:1068).
- Test: extend `test/test_quad_term_structure_engines.py`

- [ ] **Step 1: Capture golden** (pre-wiring, flat env, `_standard_snowball()` copied from `test/test_pde_term_structure_solvers.py:75-95`, default `QuadParams`): print and pin.
- [ ] **Step 2: Failing tests** — snowball QUAD discrimination (kinked vs collapsed, `rel=1e-5`) + golden flat identity (exact equality).
- [ ] **Step 3: Wire per the file notes.**
- [ ] **Step 4: Run** new tests + `test/test_snowball_quad*.py` + KI/event suites (`test/test_ki_probability_definitions.py test/test_quad_event_stats_smoothing.py test/test_cashleg/`) + full suite. PASS.
- [ ] **Step 5: Commit** `feat(quad): snowball recursion consumes per-step term parameters`

---

### Task 5: Phoenix + KO-reset QUAD

**Files:**
- Modify: `quantark/asset/equity/engine/quad/phoenix_quad_engine.py` (:66-68), `quantark/asset/equity/engine/quad/ko_reset_snowball_quad_engine.py` (:76-78, :342-344) — same wiring as Task 4 at their extraction sites (shared machinery inherited from `SnowballQuadEngine` is already per-step after Task 4).
- Test: extend `test/test_quad_term_structure_engines.py` — phoenix discrimination (product = `_standard_phoenix()` discrete-KI builder copied from `test/test_pde_term_structure_solvers.py:138-159`), KO-reset discrimination (product from `test/conftest.py:165-174`).

- [ ] **Steps: failing tests → wire → run (phoenix/ko-reset quad suites + full suite) → commit** `feat(quad): phoenix + KO-reset quad consume per-step term parameters`

---

### Task 6: Phase 3 gate — cross-family term agreement (spec test layer 4)

**Files:**
- Test: extend `test/test_quad_term_structure_engines.py`

- [ ] **Step 1: The flagship test** — MC vs PDE vs QUAD on the same term-structured snowball:

```python
def test_snowball_mc_pde_quad_agree_on_term_structure():
    """Spec test layer 4: all three engine families price the same
    term-structured snowball within cross-family tolerances."""
    from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
    from quantark.asset.equity.engine.pde import SnowballPDESolver
    from quantark.asset.equity.engine.quad import SnowballQuadEngine
    from quantark.asset.equity.param import MCParams

    env = make_term_env("kinked")
    product = _standard_snowball()  # discrete-KI variant if QUAD requires it —
    # reuse whatever product config the existing three-engine agreement tests
    # use (rg -n "agree" test/test_ki_probability_definitions.py)

    px_mc = SnowballMCEngine(
        params=MCParams(num_paths=300_000, time_steps=252, use_qmc=True, seed=7)
    ).price(product, env)
    px_pde = SnowballPDESolver().price(product, env)
    px_quad = SnowballQuadEngine().price(product, env)

    assert px_pde == pytest.approx(px_mc, rel=5e-3)
    assert px_quad == pytest.approx(px_pde, rel=5e-3)
```

Tolerances: copy from the existing flat-env three-engine agreement test if one exists (KI-probability suite); 5e-3 as fallback. Mind the KI monitoring convention: use a product config all three families support (discrete KO + the KI convention the existing agreement tests use).
- [ ] **Step 2: Full suite — green.**
- [ ] **Step 3: Commit** `test(quad): Phase 3 gate — cross-family MC/PDE/QUAD term agreement`
