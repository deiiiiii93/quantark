# 1D MC Perf Program — Phase 1 (+ Phase-2 spike) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the two bitwise-proven perf candidates in engine code (WS-1 LV scalar-t fast path, WS-2 shared GBM path-build kernel with optional Numba backend) and run the WS-3 fused-kernel spike to a recorded verdict.

**Architecture:** WS-1 adds a scalar-time branch inside `LocalVolSurface.local_vol` (shared strike-weight helper, 1-D row gathers, unchanged arithmetic order). WS-2 extracts the `GBMPathGenerator.generate_paths` tail into `quantark/montecarlo/gbm_kernels.py` under the established optional-accelerator contract (NumPy reference = today's exact code; Numba path asserted bit-identical). WS-3 is a demo-only spike, gated, no engine changes.

**Tech Stack:** NumPy 2.x, numba 0.66 (optional, `[accel]` extra — already in `pyproject.toml`), pytest.

**Spec:** `docs/superpowers/specs/2026-08-10-mc1d-perf-program-design.md`. Proven reference implementations: `docs/mc1d-perf/demo_lv_scalar_t.py`, `docs/mc1d-perf/demo_gbm_numba_fusion.py`.

## Global Constraints

- Work in the worktree `/Users/fuxinyao/quant-ark/.claude/worktrees/mc1d-perf-migration` (branch `worktree-mc1d-perf-migration`). Run everything from there.
- Tests MUST shadow the editable install: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest …` (the editable install resolves `quantark` to the main repo otherwise).
- `njit(cache=True, fastmath=False)` only; `fastmath` is forbidden (licenses reassociation → breaks bit-identity).
- Bit-identity is asserted with `.tobytes()` / `np.array_equal`, never `allclose`.
- Never track numba cache artifacts: before any `git add` of a docs dir, check for `__pycache__/*.nbc|*.nbi` and exclude them.
- Never `git add example/` (sample-data float churn).
- No golden re-freeze anywhere in this plan; defaults must be byte-stable (Task 6 proves it).
- Commit messages: conventional style, end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 0: Tag the pre-change base for byte comparison

**Files:** none (git only).

- [ ] **Step 1: Tag the current HEAD**

```bash
cd /Users/fuxinyao/quant-ark/.claude/worktrees/mc1d-perf-migration
git tag -f mc1d-phase1-base
git rev-parse mc1d-phase1-base
```

Expected: prints the current HEAD sha (docs-only commits so far; engine code is untouched at this tag).

---

### Task 1: WS-1 — scalar-t fast path in `LocalVolSurface.local_vol`

**Files:**
- Modify: `quantark/volmodels/localvol/surface.py` (the `local_vol` method, currently lines ~57–96)
- Test (create): `test/test_local_vol_scalar_t_fastpath.py`

**Interfaces:**
- Produces: `local_vol(spot, t)` — public behavior unchanged; scalar-`t` calls take a fast branch. New private helpers `_strike_weights(s_flat)` and `_local_vol_scalar_t(s, t)`.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing test**

Create `test/test_local_vol_scalar_t_fastpath.py`:

```python
"""Scalar-time fast path in LocalVolSurface.local_vol.

MC kernels call `local_vol(spot_vector, scalar_t)` once per step; profiling
2026-08-10 put 82% of an LV European MC run inside this method (broadcast
scalar t, n_paths-long searchsorted over identical values, four 2-D fancy
gathers). The fast path hoists the time bracket out of the per-path work.
These tests pin the invariant that makes the hoist legitimate: a scalar time
and a vector of that same time must agree EXACTLY.

Mirrors test_leverage_lookup_fastpath.py (2D program, b98a8d9).
"""

import numpy as np
import pytest

from quantark.volmodels.localvol.surface import LocalVolSurface

TIME_GRID = np.array([0.0, 0.5, 1.5, 3.0])
STRIKE_GRID = np.array([50.0, 80.0, 100.0, 125.0, 200.0])
LV_GRID = np.array(
    [
        [1.05, 1.00, 0.95, 0.92, 0.90],
        [1.08, 1.02, 0.97, 0.94, 0.91],
        [1.12, 1.05, 0.99, 0.96, 0.93],
        [1.20, 1.10, 1.02, 0.98, 0.95],
    ]
)
SPOTS = np.array([40.0, 50.0, 63.0, 80.0, 100.0, 117.0, 125.0, 200.0, 260.0])


def _surface(interp="linear_s"):
    return LocalVolSurface(STRIKE_GRID, TIME_GRID, LV_GRID, interp=interp)


def _single_time_surface():
    return LocalVolSurface(STRIKE_GRID, np.array([1.0]), LV_GRID[:1])


@pytest.mark.parametrize("interp", ["linear_s", "linear_logs"])
@pytest.mark.parametrize("t", [0.0, 0.25, 0.5, 1.1, 3.0, 4.0, -1.0])
def test_scalar_time_matches_vector_of_same_time_bitwise(interp, t):
    """The invariant the fast path rests on: broadcasting t must be redundant."""
    lv = _surface(interp)
    scalar = lv.local_vol(SPOTS, t)
    vector = lv.local_vol(SPOTS, np.full(SPOTS.shape, t))
    assert np.array_equal(scalar, vector)
    assert scalar.tobytes() == vector.tobytes()


def test_single_time_grid_scalar_matches_vector_bitwise():
    lv = _single_time_surface()
    assert np.array_equal(lv.local_vol(SPOTS, 0.7),
                          lv.local_vol(SPOTS, np.full(SPOTS.shape, 0.7)))


def test_node_values_reproduced_exactly():
    lv = _surface()
    for i, t in enumerate(TIME_GRID):
        got = lv.local_vol(STRIKE_GRID, float(t))
        assert got == pytest.approx(LV_GRID[i], rel=0, abs=1e-15)


def test_time_midpoint_is_average_of_bracketing_rows():
    lv = _surface()
    got = lv.local_vol(STRIKE_GRID, 1.0)  # midway between 0.5 and 1.5
    assert got == pytest.approx(0.5 * (LV_GRID[1] + LV_GRID[2]), rel=1e-15)


def test_strike_clamps_flat_extrapolation():
    lv = _surface()
    lo = lv.local_vol(np.array([10.0, 50.0]), 0.25)
    hi = lv.local_vol(np.array([200.0, 500.0]), 0.25)
    assert lo[0] == lo[1] and hi[0] == hi[1]


def test_scalar_spot_scalar_t_returns_float():
    lv = _surface()
    out = lv.local_vol(100.0, 0.7)
    assert isinstance(out, float)
    vec = lv.local_vol(np.array([100.0]), 0.7)
    assert out == float(vec[0])


def test_large_vector_matches_bitwise():
    lv = _surface()
    rng = np.random.default_rng(7)
    spots = np.exp(rng.normal(np.log(100.0), 0.6, size=100_000))
    a = lv.local_vol(spots, 0.083333333)
    b = lv.local_vol(spots, np.full(spots.shape, 0.083333333))
    assert a.tobytes() == b.tobytes()
```

- [ ] **Step 2: Run to verify current behavior**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 -q test/test_local_vol_scalar_t_fastpath.py
```

Expected: PASSES already (the invariant holds for the shipped code — these
tests pin semantics; the fast path must keep them green). This is the one
deviation from strict test-fails-first: the contract predates the change.

- [ ] **Step 3: Implement the fast path**

In `quantark/volmodels/localvol/surface.py`, replace the body of `local_vol`
and add two helpers (same class). The general vector-`t` code moves into the
`else` branch unchanged except that strike weights come from the shared
helper:

```python
    def _strike_weights(self, s_flat: np.ndarray):
        """Bracketing strike indices and linear weights for clamped spots."""
        K = self.strike_grid
        jK = np.clip(np.searchsorted(K, s_flat, side="right"), 1, K.size - 1)
        j0, j1 = jK - 1, jK
        if self.interp == "linear_logs":
            lnK = np.log(K)
            wK = (np.log(s_flat) - lnK[j0]) / (lnK[j1] - lnK[j0])
        else:
            wK = (s_flat - K[j0]) / (K[j1] - K[j0])
        return j0, j1, wK

    def _local_vol_scalar_t(self, s: np.ndarray, tt: np.ndarray):
        """Scalar-t fast path: one time bracket, 1-D row gathers.

        Same arithmetic ORDER as the general path (strike interpolation per
        row first, then the time blend) — reordering is not bitwise.
        """
        shape = s.shape
        K = self.strike_grid
        s_flat = np.clip(s.ravel(), K[0], K[-1])
        j0, j1, wK = self._strike_weights(s_flat)

        g = self.lv_grid
        if self.time_grid.size == 1:
            row = g[0]
            vals = row[j0] * (1.0 - wK) + row[j1] * wK
        else:
            Tg = self.time_grid
            t_val = float(np.clip(tt, Tg[0], Tg[-1]))
            iT = int(np.clip(np.searchsorted(Tg, t_val, side="right"), 1, Tg.size - 1))
            i0, i1 = iT - 1, iT
            wT = (t_val - Tg[i0]) / (Tg[i1] - Tg[i0])
            g0, g1 = g[i0], g[i1]  # 1-D row views, not 2-D fancy gathers
            bottom = g0[j0] * (1.0 - wK) + g0[j1] * wK
            top = g1[j0] * (1.0 - wK) + g1[j1] * wK
            vals = bottom * (1.0 - wT) + top * wT

        result = np.asarray(vals, dtype=float).reshape(shape)
        return result if result.shape else float(result)

    def local_vol(self, spot: ArrayLike, t: ArrayLike) -> "float | np.ndarray":
        """Vectorized bilinear (time, strike) interpolation with flat extrapolation.

        Gathers only the surrounding grid nodes (no per-point Python loop), so it is
        suitable for Monte Carlo path evaluation. A scalar ``t`` (the MC per-step
        call shape) takes a fast path that computes the time bracket once.
        """
        s = np.asarray(spot, dtype=float)
        tt = np.asarray(t, dtype=float)
        if not (np.all(np.isfinite(s)) and np.all(np.isfinite(tt))):
            raise ValidationError("spot and t must be finite")
        if tt.ndim == 0:
            return self._local_vol_scalar_t(s, tt)

        s_b, t_b = np.broadcast_arrays(s, tt)
        shape = s_b.shape

        K = self.strike_grid
        s_flat = np.clip(s_b.ravel(), K[0], K[-1])
        j0, j1, wK = self._strike_weights(s_flat)

        g = self.lv_grid
        if self.time_grid.size == 1:
            row = g[0]
            vals = row[j0] * (1.0 - wK) + row[j1] * wK
        else:
            Tg = self.time_grid
            t_flat = np.clip(t_b.ravel(), Tg[0], Tg[-1])
            iT = np.clip(np.searchsorted(Tg, t_flat, side="right"), 1, Tg.size - 1)
            i0, i1 = iT - 1, iT
            wT = (t_flat - Tg[i0]) / (Tg[i1] - Tg[i0])
            bottom = g[i0, j0] * (1.0 - wK) + g[i0, j1] * wK
            top = g[i1, j0] * (1.0 - wK) + g[i1, j1] * wK
            vals = bottom * (1.0 - wT) + top * wT

        result = np.asarray(vals, dtype=float).reshape(shape)
        return result if result.shape else float(result)
```

(The general branch is today's code verbatim apart from the extracted
`_strike_weights` call — identical operations in identical order.)

- [ ] **Step 4: Run the new test file and every existing LV consumer suite**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q \
  test/test_local_vol_scalar_t_fastpath.py test/test_local_vol_mc_kernel.py \
  test/test_local_vol_equity_engines.py test/test_local_vol_fx_engines.py \
  test/test_volmodel_structured_risk.py
```

Expected: all pass.

- [ ] **Step 5: Re-run the standalone demo as a regression probe**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python docs/mc1d-perf/demo_lv_scalar_t.py
```

Expected: bitwise sweep PASS; end-to-end speedups now ≈1.0× (the demo's
"fast" patch and the shipped code are the same algorithm after this task) —
that ≈1.0× IS the confirmation the fast path landed.

- [ ] **Step 6: Commit**

```bash
git add quantark/volmodels/localvol/surface.py test/test_local_vol_scalar_t_fastpath.py
git commit -m "perf(localvol): scalar-time fast path in the local vol lookup

local_vol(spot_vector, scalar_t) — the MC per-step call shape — computed an
n_paths-long searchsorted over identical times and four 2-D fancy gathers;
82% of an LV European MC run (docs/mc1d-perf/prof_baseline.py). The scalar-t
branch computes the time bracket once and uses 1-D row views with the same
arithmetic order, so it is bitwise (measured 1.46x/1.31x end-to-end,
docs/mc1d-perf/demo_lv_scalar_t.py).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: WS-2 — `quantark/montecarlo/gbm_kernels.py`

**Files:**
- Create: `quantark/montecarlo/gbm_kernels.py`
- Test (create): `test/test_gbm_path_kernel.py`

**Interfaces:**
- Produces: `gbm_path_tail(dW, drift_dt, vol_vec, s0) -> np.ndarray`
  (shape `(n_paths, n_steps + 1)`, `paths[:, 0] == s0`),
  `gbm_path_tail_numpy(...)` (same signature, reference), and
  `gbm_backend() -> str` (`"numba" | "numpy"`). Task 3 wires
  `gbm_path_tail` into `GBMPathGenerator`.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing test**

Create `test/test_gbm_path_kernel.py`:

```python
"""Shared GBM path-build tail with an optional Numba backend.

Contract mirrors test_qe_variance_kernel.py: the NumPy reference reproduces
the legacy inline tail bit-for-bit, the Numba path (when installed) is
asserted bit-identical to the reference, and the live backend self-reports.
"""

import numpy as np
import pytest

from quantark.montecarlo import gbm_kernels


def _inputs(n_paths=257, n_steps=63, seed=11):
    rng = np.random.default_rng(seed)
    dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(1.0 / n_steps)
    k = np.arange(n_steps)
    vol = 0.18 + 0.06 * np.sin(2 * np.pi * k / n_steps) ** 2
    drift_dt = (0.03 + 0.02 * k / n_steps - 0.5 * vol * vol) * (1.0 / n_steps)
    return dW, drift_dt, vol, 100.0


def _legacy_tail(dW, drift_dt, vol_vec, s0):
    """The pre-extraction generate_paths tail, verbatim."""
    paths = np.zeros((dW.shape[0], dW.shape[1] + 1), dtype=float)
    paths[:, 0] = s0
    exp_term = np.exp(drift_dt.reshape(1, -1) + vol_vec.reshape(1, -1) * dW)
    paths[:, 1:] = s0 * np.cumprod(exp_term, axis=1)
    return paths


def test_backend_reports_a_known_value():
    assert gbm_kernels.gbm_backend() in ("numba", "numpy")


def test_numpy_reference_matches_legacy_tail_bitwise():
    dW, drift_dt, vol, s0 = _inputs()
    a = gbm_kernels.gbm_path_tail_numpy(dW, drift_dt, vol, s0)
    assert a.tobytes() == _legacy_tail(dW, drift_dt, vol, s0).tobytes()


def test_dispatcher_matches_reference_bitwise():
    dW, drift_dt, vol, s0 = _inputs()
    a = gbm_kernels.gbm_path_tail(dW, drift_dt, vol, s0)
    b = gbm_kernels.gbm_path_tail_numpy(dW, drift_dt, vol, s0)
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("n_paths,n_steps", [(1, 1), (7, 5), (1024, 252)])
def test_shapes_and_initial_column(n_paths, n_steps):
    dW, drift_dt, vol, s0 = _inputs(n_paths=n_paths, n_steps=n_steps)
    out = gbm_kernels.gbm_path_tail(dW, drift_dt, vol, s0)
    assert out.shape == (n_paths, n_steps + 1)
    assert np.all(out[:, 0] == s0)


def test_noncontiguous_input_matches_contiguous():
    dW, drift_dt, vol, s0 = _inputs(n_paths=64, n_steps=32)
    wide = np.empty((64, 64))
    wide[:, ::2] = dW
    view = wide[:, ::2]
    assert not view.flags.c_contiguous
    a = gbm_kernels.gbm_path_tail(view, drift_dt, vol, s0)
    assert a.tobytes() == gbm_kernels.gbm_path_tail(dW, drift_dt, vol, s0).tobytes()


@pytest.mark.skipif(gbm_kernels.gbm_backend() != "numba",
                    reason="numba accelerator not installed")
def test_numba_path_is_bit_identical_across_regimes():
    for seed, n_paths, n_steps in ((1, 1024, 252), (2, 8192, 63), (3, 3, 1)):
        dW, drift_dt, vol, s0 = _inputs(n_paths=n_paths, n_steps=n_steps, seed=seed)
        a = gbm_kernels._gbm_path_tail_numba(dW, drift_dt, vol, s0)
        b = gbm_kernels.gbm_path_tail_numpy(dW, drift_dt, vol, s0)
        assert a.tobytes() == b.tobytes()
```

- [ ] **Step 2: Run to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 -q test/test_gbm_path_kernel.py
```

Expected: FAIL — `ModuleNotFoundError` / `AttributeError: gbm_kernels`.

- [ ] **Step 3: Implement the module**

Create `quantark/montecarlo/gbm_kernels.py`:

```python
"""Shared GBM/BSM path-build tail, with an optional Numba backend.

Every BSM 1D engine builds paths through GBMPathGenerator, whose tail
(exp of drift+diffusion, then cumprod) was ~70% of a European MC pricing and
allocates four (n_paths, n_steps) temporaries (docs/mc1d-perf, 2026-08-10).
The fused per-path Numba loop performs the SAME operations in the SAME order:
c_k = c_{k-1} * exp(drift_dt[k] + vol[k] * dW[p, k]) is exactly np.cumprod's
left-to-right fold, and the s0 multiply is applied AFTER the fold — never
seeded into the accumulator, because FP multiplication is not associative.

Same optional-accelerator contract as qe_kernels: NumPy reference by default,
`pip install quantark[accel]` for Numba, backend self-reporting, fastmath OFF
(it licenses reassociation), bit-identity asserted in
test_gbm_path_kernel.py rather than assumed.

Measured (2026-08-10, arm64): tail 1.37-1.64x, European engine 1.33x
end-to-end (docs/mc1d-perf/demo_gbm_numba_fusion.py).
"""

from __future__ import annotations

import numpy as np


def gbm_path_tail_numpy(
    dW: np.ndarray,
    drift_dt: np.ndarray,
    vol_vec: np.ndarray,
    s0: float,
) -> np.ndarray:
    """Reference implementation: the historical generate_paths tail, verbatim."""
    paths = np.zeros((dW.shape[0], dW.shape[1] + 1), dtype=float)
    paths[:, 0] = s0
    exp_term = np.exp(drift_dt.reshape(1, -1) + vol_vec.reshape(1, -1) * dW)
    paths[:, 1:] = s0 * np.cumprod(exp_term, axis=1)
    return paths


def _build_numba_kernel():
    """Compile the fused kernel, or return None when Numba is unavailable."""
    try:
        from numba import njit
    except ImportError:  # pragma: no cover - depends on the environment
        return None

    # fastmath stays OFF: it licenses reassociation, which would break
    # bit-identity with the NumPy reference.
    @njit(cache=True, fastmath=False)
    def _kernel(dW, drift_dt, vol_vec, s0, out):  # pragma: no cover - via dispatcher
        n_paths, n_steps = dW.shape
        for p in range(n_paths):
            c = 1.0
            for k in range(n_steps):
                c = c * np.exp(drift_dt[k] + vol_vec[k] * dW[p, k])
                out[p, k + 1] = s0 * c

    return _kernel


_NUMBA_KERNEL = _build_numba_kernel()


def gbm_backend() -> str:
    """``"numba"`` when the accelerator compiled, else ``"numpy"``."""
    return "numba" if _NUMBA_KERNEL is not None else "numpy"


def _gbm_path_tail_numba(dW, drift_dt, vol_vec, s0) -> np.ndarray:
    dW = np.ascontiguousarray(dW, dtype=float)
    drift_dt = np.ascontiguousarray(drift_dt, dtype=float)
    vol_vec = np.ascontiguousarray(vol_vec, dtype=float)
    paths = np.empty((dW.shape[0], dW.shape[1] + 1), dtype=float)
    paths[:, 0] = s0
    _NUMBA_KERNEL(dW, drift_dt, vol_vec, float(s0), paths)
    return paths


def gbm_path_tail(dW, drift_dt, vol_vec, s0) -> np.ndarray:
    """Build GBM paths from increments, via Numba when available."""
    if _NUMBA_KERNEL is not None:
        return _gbm_path_tail_numba(dW, drift_dt, vol_vec, s0)
    return gbm_path_tail_numpy(
        np.asarray(dW, dtype=float),
        np.asarray(drift_dt, dtype=float),
        np.asarray(vol_vec, dtype=float),
        float(s0),
    )
```

- [ ] **Step 4: Run to verify it passes**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 -q test/test_gbm_path_kernel.py
```

Expected: all pass (numba installed in this venv, so the skipif test runs).

- [ ] **Step 5: Commit**

```bash
git add quantark/montecarlo/gbm_kernels.py test/test_gbm_path_kernel.py
git commit -m "perf(mc): shared GBM path-build tail with an optional Numba backend

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: WS-2 — wire `GBMPathGenerator.generate_paths` onto the kernel

**Files:**
- Modify: `quantark/asset/equity/process/bsm/qmc_path_generator.py` (the
  `generate_paths` tail, currently lines ~254–266)
- Test (create): `test/test_gbm_generator_kernel_wiring.py`

**Interfaces:**
- Consumes: `gbm_path_tail(dW, drift_dt, vol_vec, s0)` from Task 2.
- Produces: unchanged public `generate_paths` behavior (byte-stable outputs).

- [ ] **Step 1: Write the failing test**

Create `test/test_gbm_generator_kernel_wiring.py`:

```python
"""generate_paths must route through the shared GBM kernel byte-stably."""

import numpy as np

from quantark.asset.equity.process.bsm.qmc_path_generator import GBMPathGenerator
from quantark.asset.equity.process.bsm.qmc_sobol import (
    PseudoRandomNormalGenerator, SobolNormalGenerator,
)
from quantark.asset.equity.process.bsm.qmc_variance_reduction import (
    VarianceReductionConfig,
)
from quantark.montecarlo import gbm_kernels


def _gen(stream, is_qmc, vr=None, bridge=False, n_paths=513, n_steps=63):
    k = np.arange(n_steps)
    vol = 0.18 + 0.06 * np.sin(2 * np.pi * k / n_steps) ** 2
    return GBMPathGenerator(
        initial_value=100.0, vol=vol, rrf=0.03 + 0.02 * k / n_steps,
        div=0.01 + 0.0 * k, maturity=1.0, time_steps=n_steps,
        num_paths=n_paths, model="bsm", random_stream=stream,
        use_brownian_bridge=bridge, vr_config=vr, is_qmc=is_qmc,
    )


def _reference_paths(gen, batch_id=None):
    """Recompute what generate_paths must produce, via the NumPy reference."""
    from quantark.asset.equity.process.bsm.qmc_variance_reduction import (
        apply_variance_reduction_to_normals,
    )
    z = gen._generate_base_normals(batch_id=batch_id)
    z, _, _ = apply_variance_reduction_to_normals(
        n_paths=gen.num_paths, dim=gen.time_steps, base_normals=z,
        vr_config=gen.vr_config, is_qmc=gen.is_qmc,
    )
    dW = gen._build_brownian_increments(z)
    drift_dt = (gen._drift_vec - 0.5 * gen._vol_vec * gen._vol_vec) * gen.dt_vector
    return gbm_kernels.gbm_path_tail_numpy(dW, drift_dt, gen._vol_vec,
                                           gen.initial_value)


def test_pseudo_paths_match_reference_bitwise():
    a, _ = _gen(PseudoRandomNormalGenerator(seed=42), False).generate_paths()
    b = _reference_paths(_gen(PseudoRandomNormalGenerator(seed=42), False))
    assert a.tobytes() == b.tobytes()


def test_sobol_bridge_paths_match_reference_bitwise():
    a, _ = _gen(SobolNormalGenerator(base_seed=7), True,
                bridge=True).generate_paths(batch_id=3)
    b = _reference_paths(_gen(SobolNormalGenerator(base_seed=7), True,
                              bridge=True), batch_id=3)
    assert a.tobytes() == b.tobytes()


def test_antithetic_paths_match_reference_bitwise():
    vr = VarianceReductionConfig(antithetic=True)
    a, _ = _gen(PseudoRandomNormalGenerator(seed=5), False, vr=vr).generate_paths()
    b = _reference_paths(_gen(PseudoRandomNormalGenerator(seed=5), False, vr=vr))
    assert a.tobytes() == b.tobytes()


def test_aux_contract_unchanged():
    gen = _gen(PseudoRandomNormalGenerator(seed=1), False)
    paths, aux = gen.generate_paths(batch_id=2, return_aux=True)
    assert aux["batch_id"] == np.array(2)
```

- [ ] **Step 2: Run to verify the reference tests already bind**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 -q test/test_gbm_generator_kernel_wiring.py
```

Expected: PASS already (the reference reproduces today's tail exactly). These
tests pin byte-stability across the rewiring; they must stay green after
Step 3, where the tail's implementation changes underneath them.

- [ ] **Step 3: Rewire the tail**

In `qmc_path_generator.py`, add the import at module top with the existing
imports:

```python
from quantark.montecarlo.gbm_kernels import gbm_path_tail
```

Replace the tail of `generate_paths` (from `# Build GBM/BSM paths` through
`paths[:, 1:] = self.initial_value * np.cumprod(exp_term, axis=1)`) with:

```python
        # Build GBM/BSM paths via the shared kernel (Numba-accelerated when
        # quantark[accel] is installed; NumPy reference otherwise — bitwise
        # either way, asserted in test_gbm_path_kernel.py).
        drift_dt = (
            self._drift_vec - 0.5 * self._vol_vec * self._vol_vec
        ) * self.dt_vector
        paths = gbm_path_tail(dW, drift_dt, self._vol_vec, self.initial_value)
```

(Note `paths = np.zeros(...)`/`paths[:, 0] = ...` lines are deleted — the
kernel owns allocation.)

- [ ] **Step 4: Run wiring tests + kernel tests + all MC engine suites**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q \
  test/test_gbm_generator_kernel_wiring.py test/test_gbm_path_kernel.py \
  test/test_qmc_sampler.py test/test_snowball_mc_engine.py \
  test/test_ko_reset_snowball_mc_engine.py
```

Expected: all pass.

- [ ] **Step 5: Re-run the fusion demo as a regression probe**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python docs/mc1d-perf/demo_gbm_numba_fusion.py
```

Expected: bitwise PASS; engine end-to-end "shipped" vs "numba" now both fast
(≈1.0× between them) because shipped IS the kernel path after this task.

- [ ] **Step 6: Commit**

```bash
git add quantark/asset/equity/process/bsm/qmc_path_generator.py test/test_gbm_generator_kernel_wiring.py
git commit -m "perf(mc): route the GBM path build through the shared kernel

1.33x European MC end-to-end with numba installed; byte-stable with or
without it (docs/mc1d-perf/demo_gbm_numba_fusion.py).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: detached-worktree byte-compare (spec gate 2.1)

**Files:**
- Create: `/private/tmp/claude-501/-Users-fuxinyao-quant-ark/279c9e7f-c336-4440-be1f-3bc56a7f5e9e/scratchpad/probe_phase1.py` (outside the repo — it must run against BOTH trees)

**Interfaces:** consumes public APIs only, so it runs at the base tag too.

- [ ] **Step 1: Write the probe**

```python
"""Dump full-precision 1D MC outputs for byte comparison across trees."""
import sys, numpy as np
from datetime import datetime

out_dir = sys.argv[1]

from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.param import MCParams
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.volmodels.localvol.surface import LocalVolSurface
from quantark.volmodels.localvol.mc_kernel import price_european_lv_mc, price_barrier_lv_mc
from quantark.asset.equity.process.bsm.qmc_path_generator import GBMPathGenerator
from quantark.asset.equity.process.bsm.qmc_sobol import (
    PseudoRandomNormalGenerator, SobolNormalGenerator,
)

env = PricingEnvironment(
    spot_quote=SpotQuote(spot=100.0), vol_surface=FlatVolSurface(volatility=0.2),
    rate_curve=FlatRateCurve(rate=0.05),
    div_yield=ContinuousDividendYield(div_yield=0.02),
    valuation_date=datetime(2024, 1, 1),
)

# 1. raw paths, pseudo + sobol
for name, stream, qmc in (("pseudo", PseudoRandomNormalGenerator(seed=42), False),
                          ("sobol", SobolNormalGenerator(base_seed=42), True)):
    gen = GBMPathGenerator(initial_value=100.0, vol=0.2, rrf=0.05, div=0.02,
                           maturity=1.0, time_steps=126, num_paths=4096,
                           model="bsm", random_stream=stream,
                           use_brownian_bridge=qmc, vr_config=None, is_qmc=qmc)
    paths, _ = gen.generate_paths(batch_id=1 if qmc else None)
    open(f"{out_dir}/paths_{name}.bin", "wb").write(paths.tobytes())

# 2. euro engine price
eng = EuropeanMCEngine(params=MCParams(num_paths=100_000, time_steps=252, seed=42),
                       method=MonteCarloMethod.PSEUDO)
p = eng.price(EuropeanVanillaOption(strike=100.0, maturity=1.0,
                                    option_type=OptionType.CALL), env)
open(f"{out_dir}/euro.bin", "wb").write(np.float64(p).tobytes())

# 3. LV kernels on a smiled surface
t_grid = np.linspace(0.0, 2.0, 25)
k_grid = np.exp(np.linspace(np.log(50.0), np.log(200.0), 61))
logm = np.log(k_grid / 100.0)
grid = np.clip((0.20 + 0.15 * logm**2 - 0.05 * logm)[None, :]
               * (1.0 + 0.1 * np.sqrt(t_grid))[:, None], 0.05, 1.5)
lv = LocalVolSurface(k_grid, t_grid, grid)
n = 252
common = dict(s0=100.0, strike=100.0, is_call=True, lv_surface=lv,
              step_dt=np.full(n, 1.0 / n), r_fwd=np.full(n, 0.05),
              carry_fwd=np.full(n, 0.02), disc_factor=float(np.exp(-0.05)),
              num_paths=50_000, seed=42)
pe = price_european_lv_mc(**common)
pb = price_barrier_lv_mc(**common, barrier=130.0, is_up=True, is_out=True,
                         rebate=1.0, continuous=True)
open(f"{out_dir}/lv.bin", "wb").write(np.array([pe, pb]).tobytes())
print("probe done", out_dir)
```

- [ ] **Step 2: Run against both trees and compare**

```bash
WT=/Users/fuxinyao/quant-ark/.claude/worktrees/mc1d-perf-migration
SCRATCH=/private/tmp/claude-501/-Users-fuxinyao-quant-ark/279c9e7f-c336-4440-be1f-3bc56a7f5e9e/scratchpad
BASE_TREE=$SCRATCH/mc1d-base-tree
cd $WT && git worktree add --detach $BASE_TREE mc1d-phase1-base
mkdir -p $SCRATCH/out_new $SCRATCH/out_base
PYTHONPATH=$WT /Users/fuxinyao/quant-ark/.venv/bin/python $SCRATCH/probe_phase1.py $SCRATCH/out_new
PYTHONPATH=$BASE_TREE /Users/fuxinyao/quant-ark/.venv/bin/python $SCRATCH/probe_phase1.py $SCRATCH/out_base
for f in paths_pseudo paths_sobol euro lv; do
  cmp $SCRATCH/out_base/$f.bin $SCRATCH/out_new/$f.bin && echo "$f IDENTICAL"
done
cd $WT && git worktree remove --force $BASE_TREE
```

Expected: all four `IDENTICAL`. If any differs, STOP — do not rationalize;
find the operation-order divergence (this gate exists precisely to catch it).

- [ ] **Step 3: Record the result**

Append to `docs/mc1d-perf/DECISION-2026-08-10.md` under a new
`## Phase-1 landing (date)` heading: the four IDENTICAL lines, byte counts,
and `gbm_backend()` value during the run. Commit:

```bash
git add docs/mc1d-perf/DECISION-2026-08-10.md
git commit -m "docs(perf): record the phase-1 detached-worktree byte gate

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: WS-3 spike — fused single-pass path+scan demo (gated, no engine change)

**Files:**
- Create: `docs/mc1d-perf/demo_fused_snowball_kernel.py`
- Modify: `docs/mc1d-perf/DECISION-2026-08-10.md` (verdict row)

**Interfaces:** demo-only; nothing downstream consumes it in this plan.

- [ ] **Step 1: Write the demo**

Create `docs/mc1d-perf/demo_fused_snowball_kernel.py`. Scope: discrete-KO +
discrete-KI standard snowball primitives. The normals matrix still
materializes (bitwise constraint: same NumPy stream); the fusion removes the
paths matrix and the scan gathers. Gate: primitives bit-identical AND
stage-level (path build + KO/KI scans) speedup ≥1.5×.

```python
"""WS-3 spike: fused single-pass path build + discrete KO/KI scan (Numba).

Gate (spec 2026-08-10-mc1d-perf-program-design.md WS-3): primitives
(first_ko_idx, ki_triggered, first_ki_idx, terminal) bit-identical to the
shipped pipeline on the same draws, AND >=1.5x on the fused stage at
100k x 252. Continuous-KI stays out of scope (data-dependent rng draws).
"""

import time

import numpy as np
from numba import njit

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.montecarlo.gbm_kernels import gbm_path_tail


@njit(cache=True, fastmath=False)
def _fused_scan(dW, drift_dt, vol, s0, ko_cols, ko_barriers, ki_cols,
                ki_barriers, is_reverse, first_ko, first_ki, terminal):
    n_paths, n_steps = dW.shape
    n_ko = ko_cols.shape[0]
    n_ki = ki_cols.shape[0]
    for p in range(n_paths):
        c = 1.0
        fko = -1
        fki = -1
        iko = 0
        iki = 0
        for k in range(n_steps):
            c = c * np.exp(drift_dt[k] + vol[k] * dW[p, k])
            s = s0 * c
            if iko < n_ko and k == ko_cols[iko]:
                if fko < 0:
                    hit = (s <= ko_barriers[iko]) if is_reverse else (s >= ko_barriers[iko])
                    if hit:
                        fko = iko
                iko += 1
            if iki < n_ki and k == ki_cols[iki]:
                if fki < 0:
                    hit = (s >= ki_barriers[iki]) if is_reverse else (s <= ki_barriers[iki])
                    if hit:
                        fki = iki
                iki += 1
        first_ko[p] = fko
        first_ki[p] = fki
        terminal[p] = s0 * c


def shipped_stage(engine, dW, drift_dt, vol, s0, ko_cols, ko_barriers,
                  ki_cols, ki_barriers, is_reverse):
    # the dispatcher IS the post-Phase-1 shipped path (numba-backed here),
    # so the fused kernel is measured against the fair baseline
    paths = gbm_path_tail(dW, drift_dt, vol, s0)
    ko_trig, first_ko = engine._check_ko_barriers(paths, ko_cols, ko_barriers,
                                                  is_reverse)
    ki_trig, first_ki = engine._check_ki_barriers(paths, ki_cols, ki_barriers,
                                                   is_reverse)
    return first_ko, ki_trig, first_ki, paths[:, -1]


def fused_stage(dW, drift_dt, vol, s0, ko_cols, ko_barriers, ki_cols,
                ki_barriers, is_reverse):
    n = dW.shape[0]
    first_ko = np.empty(n, dtype=np.int64)
    first_ki = np.empty(n, dtype=np.int64)
    terminal = np.empty(n, dtype=float)
    _fused_scan(np.ascontiguousarray(dW), drift_dt, vol, s0, ko_cols,
                ko_barriers, ki_cols, ki_barriers, is_reverse,
                first_ko, first_ki, terminal)
    ki_trig = first_ki >= 0
    return first_ko, ki_trig, first_ki, terminal


def main():
    n_paths, n_steps = 100_000, 252
    rng = np.random.default_rng(42)
    dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(1.0 / n_steps)
    vol = np.full(n_steps, 0.2)
    drift_dt = (0.03 - 0.5 * vol * vol) * (1.0 / n_steps)
    s0 = 100.0
    ko_cols = (np.arange(1, 13) * 21 - 1).astype(np.int64)   # monthly on a 252 grid
    ko_barriers = np.full(12, 103.0)
    ki_cols = np.arange(n_steps, dtype=np.int64)             # daily discrete KI
    ki_barriers = np.full(n_steps, 75.0)
    engine = SnowballMCEngine(params=MCParams(num_paths=n_paths, seed=1))

    # NOTE: shipped _check_* take grid indices whose +1 offset maps to path
    # columns; ko_cols here are step indices k where node index = k+1, i.e.
    # the same convention (paths[:, idx + 1]). Verify once, loudly:
    a = shipped_stage(engine, dW, drift_dt, vol, s0, ko_cols, ko_barriers,
                      ki_cols, ki_barriers, False)
    b = fused_stage(dW, drift_dt, vol, s0, ko_cols, ko_barriers,
                    ki_cols, ki_barriers, False)
    names = ("first_ko", "ki_trig", "first_ki", "terminal")
    ok = True
    for name, x, y in zip(names, a, b):
        same = np.asarray(x).tobytes() == np.asarray(y).astype(np.asarray(x).dtype).tobytes()
        ok &= same
        print(f"  {name}: {'IDENTICAL' if same else 'MISMATCH'}")

    t_ship = t_fused = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        shipped_stage(engine, dW, drift_dt, vol, s0, ko_cols, ko_barriers,
                      ki_cols, ki_barriers, False)
        t_ship = min(t_ship, time.perf_counter() - t0)
        t0 = time.perf_counter()
        fused_stage(dW, drift_dt, vol, s0, ko_cols, ko_barriers,
                    ki_cols, ki_barriers, False)
        t_fused = min(t_fused, time.perf_counter() - t0)
    speedup = t_ship / t_fused
    print(f"  stage: shipped {t_ship*1e3:.1f} ms  fused {t_fused*1e3:.1f} ms  "
          f"speedup {speedup:.2f}x")
    print(f"\nVERDICT: bitwise={'yes' if ok else 'NO'}, "
          f"gate(>=1.5x)={'PASS' if ok and speedup >= 1.5 else 'FAIL'}")


if __name__ == "__main__":
    # warm the JIT outside the timers
    _ = fused_stage(np.zeros((4, 8)), np.zeros(8), np.full(8, 0.2), 100.0,
                    np.array([3], dtype=np.int64), np.array([103.0]),
                    np.array([1], dtype=np.int64), np.array([75.0]), False)
    main()
```

- [ ] **Step 2: Run it**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python docs/mc1d-perf/demo_fused_snowball_kernel.py
```

Expected: primitives IDENTICAL (if the index-convention note turns out wrong,
fix the demo's column mapping — the shipped `_check_*` functions are the
authority). Speedup is an open measurement: record whatever it is.

- [ ] **Step 3: Record the verdict**

Append the demo's output to `docs/mc1d-perf/DECISION-2026-08-10.md` as a new
row/section: PASS → note "wiring plan to follow, separate plan"; FAIL → mark
skip-on-evidence with the measured number (like C3). Commit (exclude
`__pycache__`):

```bash
git add docs/mc1d-perf/demo_fused_snowball_kernel.py docs/mc1d-perf/DECISION-2026-08-10.md
git commit -m "docs(perf): WS-3 fused path+scan spike verdict

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: full-suite gate

**Files:** none.

- [ ] **Step 1: Run the full test suite**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q 2>&1 | tail -15
```

Expected: no new failures vs the branch baseline (28 replay goldens included).
If `example/mo_volmodels` sample files show float churn in `git status`,
restore them: `git checkout -- example/`.

- [ ] **Step 2: Verify no stray artifacts, then final status**

```bash
git status --short   # expect: clean (no .nbc/.nbi, no example/ churn)
git log --oneline mc1d-phase1-base..HEAD
```

Expected: the Task 1–5 commits only. Phase 1 complete; report the measured
end-to-end numbers (from the demo re-runs) in the wrap-up message and leave
the WS-4 (one-step survival) plan as the next document to write after spec
review.
