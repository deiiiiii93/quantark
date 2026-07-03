# Engine Term-Structure Upgrade — Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared term-sampling layer (`TermCoefficients` + grid samplers), enable signed dividend/carry yields end-to-end, and create the European term-structure benchmark harness that Phases 1–3 validate against.

**Architecture:** Low-level grid samplers move from `quantark/volmodels/curves.py` to `quantark/param/term_sampling.py` (with re-exports preserved); an environment-facing `TermCoefficients` builder lives in `quantark/priceenv/term_sampling.py`. Four validation/clamp sites that forbid negative carry are relaxed. A test-helper module provides term-structured environments and an exact closed-form European reference price.

**Tech Stack:** Python 3.10+, numpy, scipy.stats (already dependencies), pytest.

**Spec:** `docs/superpowers/specs/2026-07-03-engine-term-structure-upgrade-design.md`

## Global Constraints

- Always use canonical `quantark.*` imports; never flat legacy imports.
- Use `quantark/util/numerical/` helpers (`is_close`, `safe_sqrt`, …) instead of raw float comparisons where applicable; hand-rolled tolerances only in tests via `pytest.approx`.
- Exceptions: `ValidationError` for bad inputs, `NumericalError` for numerical failure (`quantark.util.exceptions`).
- **Identity on flat inputs:** flat curves/surfaces must produce per-interval arrays where every entry equals the flat scalar; no behavior change for flat-input pricing anywhere.
- Signed carry bounds (from spec): term-structure nodes `|y| <= 1.0`; `ContinuousDividendYield` `-0.20 <= q <= 0.20`.
- Test runner: `.venv/bin/python -m pytest` (parallel by default; add `-n0` only for pdb).
- Commit after every task; conventional-commit messages; end commit body with the Claude co-author line used in this repo.

---

### Task 1: Move grid samplers to `quantark/param/term_sampling.py`

**Files:**
- Create: `quantark/param/term_sampling.py`
- Modify: `quantark/volmodels/curves.py` (becomes a re-export shim)
- Test: `test/test_param_term_sampling.py` (new)
- Unchanged but must keep passing: `test/test_volmodels_curves.py`

**Interfaces:**
- Consumes: `RateCurve.get_forward_rate(t1, t2)`, `RateCurve.get_discount_factor(T)` from `quantark/param/rrf/rate_curve.py`.
- Produces: `quantark.param.term_sampling.forward_rates_on_grid(rate_curve, t_grid) -> np.ndarray`, `forward_carry_on_grid(zero_yield: Callable[[float], float], t_grid) -> np.ndarray`, `_validate_grid(t_grid) -> np.ndarray`. Nine engine files and `quantark/asset/vol_model_risk.py` keep importing from `quantark.volmodels.curves` — the shim must preserve that.

- [ ] **Step 1: Write the failing test (new import location + identity with old location)**

Create `test/test_param_term_sampling.py`:

```python
"""Tests for quantark.param.term_sampling (moved from volmodels.curves)."""
import numpy as np
import pytest

from quantark.param.term_sampling import (
    forward_carry_on_grid,
    forward_rates_on_grid,
)
from quantark.param.rrf.rate_curve import FlatRateCurve, LinearRateCurve
from quantark.util.exceptions import ValidationError


def test_reexport_identity_with_volmodels_curves():
    from quantark.volmodels import curves as old

    assert old.forward_rates_on_grid is forward_rates_on_grid
    assert old.forward_carry_on_grid is forward_carry_on_grid


def test_forward_rates_flat_curve_is_flat():
    grid = np.array([0.0, 0.5, 1.0, 2.0])
    out = forward_rates_on_grid(FlatRateCurve(0.03), grid)
    assert out == pytest.approx([0.03, 0.03, 0.03])


def test_forward_rates_linear_curve_hand_computed():
    curve = LinearRateCurve([(1.0, 0.03), (2.0, 0.04)])
    out = forward_rates_on_grid(curve, np.array([0.0, 1.0, 2.0]))
    # zero(1)=3%; fwd over [1,2] = (0.04*2 - 0.03*1)/1 = 5%
    assert out == pytest.approx([0.03, 0.05], abs=1e-12)


def test_forward_carry_flat_yield_is_flat():
    grid = np.array([0.0, 0.5, 1.0])
    out = forward_carry_on_grid(lambda t: 0.02, grid)
    assert out == pytest.approx([0.02, 0.02])


def test_forward_carry_term_structure_hand_computed():
    def q(t):  # q(0.5)=1%, q(1.0)=2%, linear between
        return float(np.interp(t, [0.5, 1.0], [0.01, 0.02]))

    out = forward_carry_on_grid(q, np.array([0.0, 0.5, 1.0]))
    # fwd[0] = q(0.5)*0.5/0.5 = 1%; fwd[1] = (0.02*1 - 0.01*0.5)/0.5 = 3%
    assert out == pytest.approx([0.01, 0.03], abs=1e-12)


def test_grid_validation_rejects_bad_grids():
    curve = FlatRateCurve(0.03)
    with pytest.raises(ValidationError):
        forward_rates_on_grid(curve, np.array([0.5]))
    with pytest.raises(ValidationError):
        forward_rates_on_grid(curve, np.array([0.0, 1.0, 1.0]))
    with pytest.raises(ValidationError):
        forward_rates_on_grid(curve, np.array([-0.1, 1.0]))
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest test/test_param_term_sampling.py -n0 -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'quantark.param.term_sampling'`

- [ ] **Step 3: Move the code**

Create `quantark/param/term_sampling.py` with the **verbatim current contents** of `quantark/volmodels/curves.py` (module docstring, `_validate_grid`, `forward_rates_on_grid`, `forward_carry_on_grid`, and its imports — `numpy`, `Callable`, `NumericalError`, `ValidationError`). Do not edit the function bodies.

Replace the entire body of `quantark/volmodels/curves.py` with:

```python
"""Re-exports: grid samplers moved to quantark.param.term_sampling.

Kept so existing imports (engines, tests) continue to work.
"""

from quantark.param.term_sampling import (  # noqa: F401
    _validate_grid,
    forward_carry_on_grid,
    forward_rates_on_grid,
)
```

- [ ] **Step 4: Run new test, old test, and importer smoke test**

Run: `.venv/bin/python -m pytest test/test_param_term_sampling.py test/test_volmodels_curves.py -n0 -v`
Expected: all PASS

Run: `.venv/bin/python -c "import quantark.asset.equity.engine.mc.heston_mc_engine, quantark.asset.vol_model_risk"`
Expected: no output (imports resolve through the shim)

- [ ] **Step 5: Commit**

```bash
git add quantark/param/term_sampling.py quantark/volmodels/curves.py test/test_param_term_sampling.py
git commit -m "refactor(param): move grid samplers to param.term_sampling with volmodels re-export"
```

---

### Task 2: Add `discount_factors_on_grid` and `step_vols_on_grid`

**Files:**
- Modify: `quantark/param/term_sampling.py`
- Test: `test/test_param_term_sampling.py`

**Interfaces:**
- Produces:
  - `discount_factors_on_grid(rate_curve, t_grid) -> np.ndarray` — **node** discount factors, length `len(t_grid)` (entry i = DF(0, t_grid[i]); DF(0,0)=1).
  - `step_vols_on_grid(get_vol: Callable[[float, float], float], ref_strike: float, t_grid) -> np.ndarray` — per-interval vols, length `len(t_grid)-1`, from total-variance differencing at `ref_strike`; `get_vol(strike, ttm)` matches `PricingEnvironment.get_vol`.
  - Raises `NumericalError` if total variance decreases by more than `1e-12` between nodes (calendar arbitrage in the input surface); clamps sub-tolerance noise to zero. This is deliberately stricter than `PricingEnvironment.get_step_volatility` (which clamps silently and is left unchanged for compatibility).

- [ ] **Step 1: Write the failing tests**

Append to `test/test_param_term_sampling.py`:

```python
from quantark.param.term_sampling import (
    discount_factors_on_grid,
    step_vols_on_grid,
)
from quantark.util.exceptions import NumericalError


def test_discount_factors_on_grid_matches_curve():
    curve = FlatRateCurve(0.03)
    grid = np.array([0.0, 0.5, 1.0])
    out = discount_factors_on_grid(curve, grid)
    assert out == pytest.approx([1.0, np.exp(-0.015), np.exp(-0.03)], abs=1e-14)


def test_step_vols_flat_surface_is_flat():
    out = step_vols_on_grid(lambda k, t: 0.20, 100.0, np.array([0.0, 0.5, 1.0]))
    assert out == pytest.approx([0.20, 0.20], abs=1e-14)


def test_step_vols_term_structure_hand_computed():
    def vol(k, t):  # sigma(0.5)=20%, sigma(1.0)=25%
        return float(np.interp(t, [0.5, 1.0], [0.20, 0.25]))

    out = step_vols_on_grid(vol, 100.0, np.array([0.0, 0.5, 1.0]))
    # w(0.5)=0.04*0.5=0.02; w(1.0)=0.0625; step var=(0.0625-0.02)/0.5=0.085
    assert out == pytest.approx([0.20, np.sqrt(0.085)], abs=1e-12)


def test_step_vols_rejects_decreasing_total_variance():
    def vol(k, t):  # w(0.5)=0.045, w(1.0)=0.01 -> decreasing
        return 0.30 if t <= 0.5 else 0.10

    with pytest.raises(NumericalError):
        step_vols_on_grid(vol, 100.0, np.array([0.0, 0.5, 1.0]))
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest test/test_param_term_sampling.py -n0 -v -k "discount or step_vols"`
Expected: FAIL — `ImportError: cannot import name 'discount_factors_on_grid'`

- [ ] **Step 3: Implement**

Append to `quantark/param/term_sampling.py`:

```python
def discount_factors_on_grid(rate_curve, t_grid: np.ndarray) -> np.ndarray:
    """Node discount factors DF(0, t_i) for each grid node (DF at t=0 is 1)."""
    t = _validate_grid(t_grid)
    out = np.array(
        [1.0 if ti <= 0.0 else float(rate_curve.get_discount_factor(ti)) for ti in t]
    )
    if not np.all(np.isfinite(out)) or np.any(out <= 0.0):
        raise NumericalError("rate_curve produced invalid discount factors")
    return out


def step_vols_on_grid(
    get_vol: Callable[[float, float], float], ref_strike: float, t_grid: np.ndarray
) -> np.ndarray:
    """Per-interval vols from total-variance differencing at a reference strike.

    w(t) = get_vol(ref_strike, t)^2 * t;  step vol = sqrt((w1 - w0) / dt).
    Raises NumericalError if total variance decreases beyond tolerance
    (calendar arbitrage in the input surface).
    """
    t = _validate_grid(t_grid)
    w = np.empty(t.size, dtype=float)
    for i, ti in enumerate(t):
        if ti <= 0.0:
            w[i] = 0.0
        else:
            v = float(get_vol(float(ref_strike), float(ti)))
            w[i] = v * v * ti
    dw = np.diff(w)
    if np.any(dw < -1e-12):
        raise NumericalError(
            "total variance is decreasing on the grid (calendar arbitrage)"
        )
    out = np.sqrt(np.maximum(dw, 0.0) / np.diff(t))
    if not np.all(np.isfinite(out)):
        raise NumericalError("get_vol produced non-finite step vols")
    return out
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest test/test_param_term_sampling.py -n0 -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add quantark/param/term_sampling.py test/test_param_term_sampling.py
git commit -m "feat(param): add discount_factors_on_grid and step_vols_on_grid samplers"
```

---

### Task 3: `TermCoefficients` builder in `quantark/priceenv/term_sampling.py`

**Files:**
- Create: `quantark/priceenv/term_sampling.py`
- Modify: `quantark/priceenv/__init__.py` (export `TermCoefficients`)
- Test: `test/test_term_coefficients.py` (new)

**Interfaces:**
- Consumes: Task 1–2 samplers; `PricingEnvironment` (`rate_curve`, `get_div_yield`, `get_vol`).
- Produces (Phases 1–3 consume this exact API):

```python
@dataclass(frozen=True)
class TermCoefficients:
    t_grid: np.ndarray    # shape (n,)
    fwd_rates: np.ndarray # shape (n-1,)
    fwd_carry: np.ndarray # shape (n-1,)
    step_vols: np.ndarray # shape (n-1,)
    node_dfs: np.ndarray  # shape (n,)   DF(0, t_i)
    step_dfs: np.ndarray  # shape (n-1,) DF(t_i, t_{i+1}) = node_dfs[i+1]/node_dfs[i]

    @classmethod
    def from_env(cls, pricing_env, t_grid, ref_strike) -> "TermCoefficients": ...
```

- [ ] **Step 1: Write the failing tests**

Create `test/test_term_coefficients.py`:

```python
"""TermCoefficients: env -> per-interval forward coefficient arrays."""
from datetime import datetime

import numpy as np
import pytest

from quantark.param import SpotQuote
from quantark.param.div.dividend_yield import (
    ContinuousDividendYield,
    TermStructureDividendYield,
)
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.priceenv import PricingEnvironment, TermCoefficients


def make_flat_env(r=0.03, q=0.01, vol=0.20, spot=100.0):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(r),
        valuation_date=datetime(2026, 7, 3),
        spot_quote=SpotQuote(spot),
        vol_surface=FlatVolSurface(vol),
        div_yield=ContinuousDividendYield(q),
    )


def test_flat_identity_every_entry_equals_the_scalar():
    env = make_flat_env()
    grid = np.linspace(0.0, 2.0, 25)
    tc = TermCoefficients.from_env(env, grid, ref_strike=100.0)
    assert tc.fwd_rates == pytest.approx(np.full(24, 0.03), abs=1e-14)
    assert tc.fwd_carry == pytest.approx(np.full(24, 0.01), abs=1e-14)
    assert tc.step_vols == pytest.approx(np.full(24, 0.20), abs=1e-14)
    assert tc.node_dfs == pytest.approx(np.exp(-0.03 * grid), abs=1e-14)
    assert tc.step_dfs == pytest.approx(
        np.exp(-0.03 * np.diff(grid)), abs=1e-14
    )


def test_term_carry_hand_computed():
    env = make_flat_env()
    env.div_yield = TermStructureDividendYield(
        times=[0.5, 1.0], yields=[0.01, 0.02]
    )
    tc = TermCoefficients.from_env(
        env, np.array([0.0, 0.5, 1.0]), ref_strike=100.0
    )
    assert tc.fwd_carry == pytest.approx([0.01, 0.03], abs=1e-12)


def test_shapes_are_consistent():
    tc = TermCoefficients.from_env(
        make_flat_env(), np.array([0.0, 1.0, 2.0]), ref_strike=100.0
    )
    assert tc.t_grid.shape == (3,)
    assert tc.fwd_rates.shape == tc.fwd_carry.shape == tc.step_vols.shape == (2,)
    assert tc.node_dfs.shape == (3,)
    assert tc.step_dfs.shape == (2,)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest test/test_term_coefficients.py -n0 -v`
Expected: FAIL — `ImportError: cannot import name 'TermCoefficients'`

- [ ] **Step 3: Implement**

Create `quantark/priceenv/term_sampling.py`:

```python
"""Environment-facing builder for per-interval forward market coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantark.param.term_sampling import (
    _validate_grid,
    discount_factors_on_grid,
    forward_carry_on_grid,
    forward_rates_on_grid,
    step_vols_on_grid,
)


@dataclass(frozen=True)
class TermCoefficients:
    """Per-interval forward market coefficients on a time grid.

    Interval arrays have length len(t_grid) - 1; entry i covers
    [t_grid[i], t_grid[i+1]]. Node arrays have length len(t_grid).
    Flat market inputs produce arrays where every entry equals the
    flat scalar (identity property relied on for backward compatibility).
    """

    t_grid: np.ndarray
    fwd_rates: np.ndarray
    fwd_carry: np.ndarray
    step_vols: np.ndarray
    node_dfs: np.ndarray
    step_dfs: np.ndarray

    @classmethod
    def from_env(
        cls, pricing_env, t_grid: np.ndarray, ref_strike: float
    ) -> "TermCoefficients":
        t = _validate_grid(t_grid)
        node_dfs = discount_factors_on_grid(pricing_env.rate_curve, t)
        return cls(
            t_grid=t,
            fwd_rates=forward_rates_on_grid(pricing_env.rate_curve, t),
            fwd_carry=forward_carry_on_grid(pricing_env.get_div_yield, t),
            step_vols=step_vols_on_grid(pricing_env.get_vol, ref_strike, t),
            node_dfs=node_dfs,
            step_dfs=node_dfs[1:] / node_dfs[:-1],
        )
```

Add to `quantark/priceenv/__init__.py` exports: `from quantark.priceenv.term_sampling import TermCoefficients` (and append `"TermCoefficients"` to `__all__` if the module defines one).

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest test/test_term_coefficients.py -n0 -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add quantark/priceenv/term_sampling.py quantark/priceenv/__init__.py test/test_term_coefficients.py
git commit -m "feat(priceenv): TermCoefficients per-interval forward coefficient builder"
```

---

### Task 4: Signed carry — dividend yield classes

**Files:**
- Modify: `quantark/param/div/dividend_yield.py:44-49` (`ContinuousDividendYield.__post_init__`), `quantark/param/div/dividend_yield.py:87-88` (`TermStructureDividendYield.__post_init__`)
- Test: `test/test_signed_dividend_yield.py` (new)

**Interfaces:**
- Produces: `ContinuousDividendYield` accepts `-0.20 <= q <= 0.20`; `TermStructureDividendYield` accepts nodes with `|y| <= 1.0` and finite. Everything else unchanged.

- [ ] **Step 1: Write the failing tests**

Create `test/test_signed_dividend_yield.py`:

```python
"""Signed dividend/carry yields (negative implied carry from futures)."""
import math

import pytest

from quantark.param.div.dividend_yield import (
    ContinuousDividendYield,
    TermStructureDividendYield,
)
from quantark.util.exceptions import ValidationError


def test_continuous_accepts_negative_within_bound():
    assert ContinuousDividendYield(-0.05).get_yield(1.0) == -0.05


def test_continuous_rejects_beyond_symmetric_bound():
    with pytest.raises(ValidationError):
        ContinuousDividendYield(-0.25)
    with pytest.raises(ValidationError):
        ContinuousDividendYield(0.25)


def test_term_structure_accepts_negative_nodes():
    ts = TermStructureDividendYield(times=[0.1, 0.5], yields=[-0.02, 0.03])
    assert ts.get_yield(0.1) == pytest.approx(-0.02)
    assert ts.get_yield(0.05) == pytest.approx(-0.02)  # flat extrapolation


def test_term_structure_rejects_magnitude_over_one():
    with pytest.raises(ValidationError):
        TermStructureDividendYield(times=[0.1, 0.5], yields=[-1.5, 0.03])


def test_term_structure_rejects_non_finite():
    with pytest.raises(ValidationError):
        TermStructureDividendYield(times=[0.1, 0.5], yields=[math.nan, 0.03])
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest test/test_signed_dividend_yield.py -n0 -v`
Expected: FAIL — negative yields raise `ValidationError` under current validation.

- [ ] **Step 3: Implement**

In `ContinuousDividendYield.__post_init__`, replace the two checks with:

```python
    def __post_init__(self):
        """Validate dividend yield (signed carry allowed, symmetric bound)."""
        if not math.isfinite(self.div_yield):
            raise ValidationError(f"Dividend yield must be finite, got {self.div_yield}")
        if abs(self.div_yield) > 0.20:  # symmetric sanity bound
            raise ValidationError(
                f"Dividend yield magnitude seems unreasonably high: {self.div_yield}"
            )
```

(add `import math` at the top of the module).

In `TermStructureDividendYield.__post_init__`, replace

```python
        if any(y < 0 for y in self.yields):
            raise ValidationError("dividend yields must be non-negative.")
```

with

```python
        if any(not math.isfinite(y) for y in self.yields):
            raise ValidationError("dividend yields must be finite.")
        if any(abs(y) > 1.0 for y in self.yields):
            raise ValidationError("dividend yield magnitude must be <= 1.0.")
```

- [ ] **Step 4: Run new tests, then hunt for old-behavior assertions**

Run: `.venv/bin/python -m pytest test/test_signed_dividend_yield.py -n0 -v`
Expected: all PASS

Run: `grep -rn "must be non-negative\|non-negative" test/ | grep -i "div"` and
`grep -rln "ContinuousDividendYield(-\|yields=\[-" test/`
For every existing test that asserts a negative yield **raises**, update it to assert the new bound instead (raises only beyond −0.20 / |y| > 1.0). Do not delete coverage — retarget it.

Run: `.venv/bin/python -m pytest -x -q`
Expected: full suite PASS

- [ ] **Step 5: Commit**

```bash
git add quantark/param/div/dividend_yield.py test/test_signed_dividend_yield.py test/
git commit -m "feat(param): allow signed dividend/carry yields with symmetric sanity bounds"
```

---

### Task 5: Signed carry — remove clamps in bump wrappers and greeks calculator

**Files:**
- Modify: `quantark/asset/equity/report/term_structure.py:96-110` (`BucketedDividendYield.get_yield`, `ShiftedDividendYield.get_yield`)
- Modify: `quantark/asset/equity/riskmeasures/greeks_calculator.py:945-956` (one-sided fallback in `calculate_numerical_delta_q`) and `greeks_calculator.py:1171-1202` (`_build_div_bumped_env` negative raises)
- Test: `test/test_signed_dividend_yield.py` (extend)

**Interfaces:**
- Consumes: Task 4 (signed `ContinuousDividendYield` / `TermStructureDividendYield` must exist first, or `_build_div_bumped_env` would construct invalid objects).
- Produces: dividend bumps below zero flow through unchanged; `calculate_numerical_delta_q` always uses the central difference.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_signed_dividend_yield.py`:

```python
import numpy as np
from datetime import datetime
from scipy import stats

from quantark.asset.equity.report.term_structure import (
    BucketedDividendYield,
    ShiftedDividendYield,
)
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType


def _env_q0():
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.03),
        valuation_date=datetime(2026, 7, 3),
        spot_quote=SpotQuote(100.0),
        vol_surface=FlatVolSurface(0.20),
        div_yield=ContinuousDividendYield(0.0),
    )


def test_shifted_yield_goes_negative_without_clamp():
    shifted = ShiftedDividendYield(base=ContinuousDividendYield(0.0), shift=-0.01)
    assert shifted.get_yield(1.0) == pytest.approx(-0.01)


def test_bucketed_yield_goes_negative_without_clamp():
    bucketed = BucketedDividendYield(
        base=ContinuousDividendYield(0.0),
        bucket_start=0.0, bucket_end=1.0, bump=-0.01,
    )
    assert bucketed.get_yield(0.5) == pytest.approx(-0.01)


def test_central_dividend_rho_at_q_zero_matches_analytical():
    """At q=0 the old clamp broke the down bump; central FD must now match BS."""
    env = _env_q0()
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.0)
    engine = BlackScholesEngine()
    base = engine.price(option, env)

    bump = 0.0001
    pv = {}
    for sign in (+1.0, -1.0):
        e = _env_q0()
        e.div_yield = ShiftedDividendYield(
            base=ContinuousDividendYield(0.0), shift=sign * bump
        )
        pv[sign] = engine.price(option, e)
    fd_rhoq_per_1pct = (pv[1.0] - pv[-1.0]) / (2 * bump) * 0.01

    S, K, T, r, q, vol = 100.0, 100.0, 1.0, 0.03, 0.0, 0.20
    d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    analytical = -S * T * np.exp(-q * T) * stats.norm.cdf(d1) / 100.0
    assert fd_rhoq_per_1pct == pytest.approx(analytical, rel=1e-4)
    assert pv[-1.0] != pytest.approx(base, abs=1e-12)  # down bump really applied


def test_calculate_numerical_delta_q_at_q_zero_is_finite_and_central():
    env = _env_q0()
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.0)
    engine = BlackScholesEngine()
    out = GreeksCalculator().calculate_numerical_delta_q(option, env, engine)
    # dDelta/dq for a call: exp(-qT) * (-T*N(d1) - n(d1)*sqrt(T)/vol), q=0
    S, K, T, r, vol = 100.0, 100.0, 1.0, 0.03, 0.20
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    expected = -T * stats.norm.cdf(d1) - stats.norm.pdf(d1) * np.sqrt(T) / vol
    assert out == pytest.approx(expected, rel=1e-3)
```

(Import paths and constructor signatures above are verified against `test/test_european_option.py:15-21,102`.)

- [ ] **Step 2: Run to verify the clamp tests fail**

Run: `.venv/bin/python -m pytest test/test_signed_dividend_yield.py -n0 -v -k "clamp or central or delta_q"`
Expected: `test_shifted_yield_goes_negative_without_clamp` and `test_bucketed_yield_goes_negative_without_clamp` FAIL (clamped to 0.0); the FD tests FAIL or ERROR via the clamps/guards.

- [ ] **Step 3: Implement**

In `quantark/asset/equity/report/term_structure.py`:
- `BucketedDividendYield.get_yield`: replace `return max(0.0, base_yield + float(self.bump))` with `return base_yield + float(self.bump)`.
- `ShiftedDividendYield.get_yield`: replace `return max(0.0, base_yield + float(self.shift))` with `return base_yield + float(self.shift)`.

In `quantark/asset/equity/riskmeasures/greeks_calculator.py`:
- `_build_div_bumped_env`: delete the `if new_div < 0: raise ValidationError(...)` block and the `if any(y < 0 for y in new_yields): raise ValidationError(...)` block. Keep everything else.
- `calculate_numerical_delta_q`: delete the entire `if current_div - div_bump < 0:` one-sided branch (the early-return block at lines 945–956) so the central-difference path below always runs.

- [ ] **Step 4: Run tests and the full suite**

Run: `.venv/bin/python -m pytest test/test_signed_dividend_yield.py -n0 -v`
Expected: all PASS

Run: `.venv/bin/python -m pytest -q`
Expected: full suite PASS. If any existing test asserted the one-sided fallback or the negative-bump `ValidationError` (check `test/test_greeks_bump_config.py` first), retarget it to the new central-difference behavior — values may shift within bump-order tolerance; do not loosen tolerances beyond `1e-6` without noting why in the test.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/report/term_structure.py quantark/asset/equity/riskmeasures/greeks_calculator.py test/
git commit -m "fix(equity): remove zero-floor clamps on dividend bumps; delta_q always central"
```

---

### Task 6: European term-structure benchmark harness

**Files:**
- Create: `test/term_structure_benchmarks.py` (importable helper module, no `test_` prefix)
- Test: `test/test_term_structure_benchmarks.py` (new)

**Interfaces:**
- Consumes: `LinearRateCurve`, `TermStructureDividendYield`, `TermStructureVolSurface`, `FlatRateCurve`, `FlatVolSurface`, `ContinuousDividendYield`, `SpotQuote`, `PricingEnvironment`.
- Produces (Phases 1–3 import these):
  - `make_term_env(shape: str, spot: float = 100.0, valuation_date: datetime = datetime(2026, 7, 3)) -> PricingEnvironment` for `shape in {"flat", "up", "down", "kinked"}`; `"kinked"` includes a negative-carry segment.
  - `reference_european_call_price(env: PricingEnvironment, strike: float, maturity: float) -> float` — exact closed form under any term structure: `w = get_vol(K,T)^2 * T`, `DF = get_discount_factor(T)`, `F = S * exp((r(T) - q(T)) * T)`, Black formula.

- [ ] **Step 1: Write the failing test**

Create `test/test_term_structure_benchmarks.py`:

```python
"""Validate the shared term-structure benchmark harness itself."""
import numpy as np
import pytest
from scipy import stats

from test.term_structure_benchmarks import (
    make_term_env,
    reference_european_call_price,
)


def test_flat_shape_matches_hand_black_scholes():
    env = make_term_env("flat")
    S, K, T, r, q, vol = 100.0, 100.0, 1.0, 0.03, 0.01, 0.20
    d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    expected = S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(
        -r * T
    ) * stats.norm.cdf(d2)
    assert reference_european_call_price(env, 100.0, 1.0) == pytest.approx(
        expected, abs=1e-12
    )


@pytest.mark.parametrize("shape", ["up", "down", "kinked"])
def test_term_shapes_are_genuinely_non_flat(shape):
    env = make_term_env(shape)
    assert env.get_rate(0.25) != pytest.approx(env.get_rate(2.0), abs=1e-6)
    assert env.get_div_yield(0.25) != pytest.approx(
        env.get_div_yield(2.0), abs=1e-6
    )
    assert env.get_vol(100.0, 0.25) != pytest.approx(
        env.get_vol(100.0, 2.0), abs=1e-6
    )


def test_kinked_shape_has_negative_carry_segment():
    env = make_term_env("kinked")
    times = np.linspace(0.05, 2.0, 40)
    assert min(env.get_div_yield(float(t)) for t in times) < 0.0


def test_reference_price_uses_curve_df_not_flat_rate():
    env = make_term_env("up")
    px = reference_european_call_price(env, 100.0, 2.0)
    assert 0.0 < px < 100.0
    # recompute with the same formula, independently
    S = env.spot
    T = 2.0
    r, q = env.get_rate(T), env.get_div_yield(T)
    vol = env.get_vol(100.0, T)
    d1 = (np.log(S / 100.0) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    F = S * np.exp((r - q) * T)
    expected = env.get_discount_factor(T) * (
        F * stats.norm.cdf(d1) - 100.0 * stats.norm.cdf(d2)
    )
    assert px == pytest.approx(expected, abs=1e-12)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest test/test_term_structure_benchmarks.py -n0 -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'test.term_structure_benchmarks'` (if `test/` is not a package, import as a plain module per this repo's pytest rootdir behavior; match how other test helpers are imported — check `test/` for an existing `__init__.py` or conftest-based helper imports and follow that pattern).

- [ ] **Step 3: Implement**

Create `test/term_structure_benchmarks.py`:

```python
"""Shared term-structured environments + exact European reference price.

Used by the engine term-structure upgrade phases (spec
2026-07-03-engine-term-structure-upgrade-design.md, test layer 2).
"""
from datetime import datetime

import numpy as np
from scipy import stats

from quantark.param import SpotQuote
from quantark.param.div.dividend_yield import TermStructureDividendYield
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.param.vol.vol_surface import (
    FlatVolSurface,
    TermStructureVolSurface,
)
from quantark.param.div.dividend_yield import ContinuousDividendYield
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.priceenv import PricingEnvironment

_SHAPES = {
    # times,   rates,             carries,             vols
    "up": (
        [0.25, 0.5, 1.0, 2.0],
        [0.020, 0.025, 0.030, 0.038],
        [0.005, 0.010, 0.015, 0.020],
        [0.18, 0.20, 0.22, 0.25],
    ),
    "down": (
        [0.25, 0.5, 1.0, 2.0],
        [0.038, 0.030, 0.025, 0.020],
        [0.020, 0.015, 0.010, 0.005],
        [0.25, 0.22, 0.20, 0.18],
    ),
    "kinked": (
        [0.25, 0.5, 1.0, 2.0],
        [0.020, 0.035, 0.025, 0.030],
        [-0.015, 0.020, -0.005, 0.010],   # negative-carry segments
        [0.22, 0.18, 0.24, 0.20],
    ),
}


def make_term_env(shape, spot=100.0, valuation_date=datetime(2026, 7, 3)):
    """Deterministic PricingEnvironment for shape in {flat, up, down, kinked}."""
    if shape == "flat":
        return PricingEnvironment(
            rate_curve=FlatRateCurve(0.03),
            valuation_date=valuation_date,
            spot_quote=SpotQuote(spot),
            vol_surface=FlatVolSurface(0.20),
            div_yield=ContinuousDividendYield(0.01),
        )
    times, rates, carries, vols = _SHAPES[shape]
    return PricingEnvironment(
        rate_curve=LinearRateCurve(list(zip(times, rates))),
        valuation_date=valuation_date,
        spot_quote=SpotQuote(spot),
        vol_surface=TermStructureVolSurface(times=list(times), vols=list(vols)),
        div_yield=TermStructureDividendYield(
            times=list(times), yields=list(carries)
        ),
    )


def reference_european_call_price(env, strike, maturity):
    """Exact European call under term structures via cumulative-to-T inputs."""
    S = env.spot
    T = float(maturity)
    vol = env.get_vol(strike, T)
    w = vol * vol * T
    df = env.get_discount_factor(T)
    fwd = S * np.exp((env.get_rate(T) - env.get_div_yield(T)) * T)
    d1 = (np.log(fwd / strike) + 0.5 * w) / np.sqrt(w)
    d2 = d1 - np.sqrt(w)
    return float(df * (fwd * stats.norm.cdf(d1) - strike * stats.norm.cdf(d2)))
```

(Signatures verified: `TermStructureVolSurface(times=, vols=)` is a dataclass at `vol_surface.py:91` and interpolates in total variance — exactly the convention `step_vols_on_grid` assumes. `LinearRateCurve` takes a `pillars` list of `(time, rate)` tuples per `rate_curve.py:157`.)

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest test/test_term_structure_benchmarks.py -n0 -v`
Expected: all PASS

Run: `.venv/bin/python -m pytest -q`
Expected: full suite PASS (Phase 0 gate)

- [ ] **Step 5: Commit**

```bash
git add test/term_structure_benchmarks.py test/test_term_structure_benchmarks.py
git commit -m "test: shared term-structure benchmark harness (env shapes + exact European reference)"
```

---

## Phase 0 gate (before starting the Phase 1 plan)

- [ ] `.venv/bin/python -m pytest -q` — entire suite green.
- [ ] `TermCoefficients` flat-identity test green (spec principle 1 evidence).
- [ ] Negative carry constructs, prices, and bumps end-to-end (Task 4–5 tests).
- [ ] Benchmark harness importable and validated (Task 6).
