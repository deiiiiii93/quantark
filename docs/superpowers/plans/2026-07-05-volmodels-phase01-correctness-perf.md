# Vol Models Phase 0+1 (Correctness + Performance) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phases 0+1 of the volmodels improvement spec — fix the two behavioral defects (Dupire w-floor, SLV leverage-clip mismatch), remove stale docs/dead config/dead fallbacks, and land the equality-gated performance work (batched tridiagonal solves, shared FD utils, MC micro-perf).

**Architecture:** Phase 0 changes semantics deliberately (widened leverage clip band, explicit w-floor raise) with new acceptance tests; Phase 1 is performance-only and gated on exact/near-exact equality with the current implementations — capture-before regression tests are written *before* each swap so the old numbers are pinned as literals.

**Tech Stack:** NumPy vectorization, `scipy.linalg.solve_banded`, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-04-volmodels-dupire-heston-slv-improvements-design.md` (workstreams WS-A1, WS-A2, WS-A3, WS-B2, WS-D2, WS-B4 only — later phases out of scope).

## Global Constraints

- Canonical `quantark.*` imports only; never flat legacy names.
- Use `quantark/util/numerical` guards (`safe_log`, `Tolerance`, …) for real-domain scalar math; raw NumPy allowed inside hot vectorized kernels where already the norm.
- Exceptions: `ValidationError` for bad inputs, `NumericalError` for numerical failure — never bare `ValueError`/`RuntimeError`.
- No MC imports inside deterministic (PDE/FFP) code paths.
- No silent fallbacks: on numerical failure, raise with a diagnostic message.
- Tests run from the worktree with the main venv but worktree source: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest …` (editable install otherwise resolves `quantark` to the main repo).
- Serial debugging: add `-n0` (suite default is `-n auto`).
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Equality-gate philosophy (spec §9): Phase 1 swaps must reproduce old outputs — ADI batched Thomas bit-identical (≤1e-13 relative on production grids), LV `solve_banded` ≤1e-12 relative, `bin_conditional` exact-equality.

---

### Task 1: WS-A3 hygiene — dead fallbacks, stale docstring, dead config

**Files:**
- Modify: `quantark/volmodels/heston/calibration.py:107-121`
- Modify: `quantark/volmodels/slv/slv_mc_kernel.py:1-12` (module docstring)
- Modify: `quantark/volmodels/slv/fokkerplanck/config.py`
- Modify: `quantark/volmodels/slv/fokkerplanck/calibration.py:28-29` (stale comment)
- Modify: `test/test_fp_solver.py`, `test/test_fp_craig_sneyd.py` (direct consumers of the removed fields)
- Test: `test/test_fp_config.py`, existing `test/test_heston_calibration_curves.py`, `test/test_slv_mc_kernel.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `FpCalibrationConfig` without `rannacher_steps`/`scheme` fields (construction with them raises `TypeError`). No other API change.

- [ ] **Step 1: Update `test/test_fp_config.py` for the removed fields (failing first)**

Read the file; find every test constructing or asserting on `rannacher_steps` or `scheme`. Replace them with:

```python
def test_removed_fields_raise_typeerror():
    import pytest
    from quantark.volmodels.slv.fokkerplanck.config import FpCalibrationConfig
    with pytest.raises(TypeError):
        FpCalibrationConfig(rannacher_steps=2)
    with pytest.raises(TypeError):
        FpCalibrationConfig(scheme=None)
```

Delete any test that asserts `ValidationError` for bad `rannacher_steps`/`scheme` values or asserts the default values of those fields.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_fp_config.py -v`
Expected: `test_removed_fields_raise_typeerror` FAILS (fields still exist, no TypeError).

- [ ] **Step 3: Remove the dead config fields**

In `quantark/volmodels/slv/fokkerplanck/config.py`:
- Delete the field lines `scheme: ADIScheme = ADIScheme.CRAIG_SNEYD` and `rannacher_steps: int = 2` (with their trailing comments).
- Delete their `__post_init__` validation blocks (`rannacher_steps must be a non-negative integer`, `only ADIScheme.CRAIG_SNEYD is specified in v1`).
- Delete the now-unused `from quantark.util.enum.engine_enums import ADIScheme` import.

In `quantark/volmodels/slv/fokkerplanck/calibration.py` delete the stale comment (lines 28-29):

```python
    # rannacher_steps need no upper bound: the loop's `n < rannacher_steps` self-clamps (extra
    # implicit start-up steps are harmless), and the deterministic branch ignores them entirely.
```

**Fix the direct test consumers** (found in the current tree, not covered by test_fp_config.py):
- `test/test_fp_solver.py` and `test/test_fp_craig_sneyd.py` construct `FpCalibrationConfig(..., rannacher_steps=2)` and/or read `cfg.rannacher_steps`. Update each site: drop the removed kwarg from the constructor call; where the value is used downstream, replace the attribute read with a local `rannacher_steps = 2` variable; delete assertions that exist only to check the removed fields' defaults/validation (they no longer reflect production FFP calibration, which is unconditionally fully-implicit).

- [ ] **Step 4: Fix the stale SLV MC module docstring**

In `quantark/volmodels/slv/slv_mc_kernel.py` replace the sentence in the module docstring:

```
The variance follows the QE scheme with vol-of-vol
eta*sigma.
```

with:

```
The variance follows full-truncation Euler (CIR) with vol-of-vol
eta*sigma; spot and variance share the same correlated Brownian, so the spot scheme is
martingale-consistent up to the O(dt) Euler bias (QE was deliberately avoided here — see
_simulate_slv's docstring for the rationale).
```

- [ ] **Step 5: Remove the dead calibration fallbacks**

In `quantark/volmodels/heston/calibration.py`, the two target-construction branches are only reached when the other quote field is provably non-None (validated earlier). Replace:

```python
            else bs_call_price(
                s0, opt.K, opt.T, opt.iv if opt.iv is not None else 0.2, rate_i, carry_i
            )
```

with:

```python
            else bs_call_price(s0, opt.K, opt.T, opt.iv, rate_i, carry_i)
```

and:

```python
            else implied_vol_call(
                s0, opt.K, opt.T, opt.price if opt.price is not None else 0.0, rate_i, carry_i
            )
```

with:

```python
            else implied_vol_call(s0, opt.K, opt.T, opt.price, rate_i, carry_i)
```

- [ ] **Step 6: Run the affected suites**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_fp_config.py test/test_fp_solver.py test/test_fp_craig_sneyd.py test/test_heston_calibration_curves.py test/test_slv_mc_kernel.py test/test_fp_acceptance.py test/test_slv_calibration_spec_method.py test/test_leverage_calibration_dispatch.py -v`
Expected: all PASS. Also run `grep -rn "rannacher_steps\|FpCalibrationConfig(.*scheme" test/ quantark/volmodels/` to confirm no consumer of the removed fields remains (bond/equity `rannacher_steps` hits are unrelated engine params — leave them).

- [ ] **Step 7: Commit**

```bash
git add -A quantark/volmodels test/
git commit -m "chore(volmodels): WS-A3 — remove dead FP config fields, stale docs, dead calibration fallbacks

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: WS-A1 — Dupire total-variance floor

**Files:**
- Modify: `quantark/volmodels/localvol/dupire.py:26,131-148`
- Test: `test/test_dupire_local_vol.py`

**Interfaces:**
- Produces: `build_dupire_local_vol` raises `NumericalError` naming (T, K) node indices when any `w = iv²·T ≤ 1e-12`. Module constant `_W_FLOOR = 1e-12`.

- [ ] **Step 1: Write the failing boundary tests**

Append to `test/test_dupire_local_vol.py`, using the constructors the file already imports and uses directly (`GridVolSurface`, `FlatRateCurve` — match the existing import block; it has no `_make_grid_surface`/`_flat_curve` helpers):

```python
def test_w_floor_below_raises():
    # front row w = (1e-5)^2 * 1e-3 = 1e-13 < 1e-12
    strikes = np.array([90.0, 100.0, 110.0])
    mats = np.array([1e-3, 0.5, 1.0])
    iv = np.array([[1e-5] * 3, [0.2] * 3, [0.2] * 3])
    surface = GridVolSurface(strikes=strikes, maturities=mats, iv_grid=iv)
    with pytest.raises(NumericalError, match="total implied variance"):
        build_dupire_local_vol(surface, spot=100.0, rate_curve=FlatRateCurve(0.02),
                               div_yield=lambda t: 0.0, validate_arbitrage=False)


def test_w_floor_above_builds():
    # front row w = (1e-4)^2 * 0.5 = 5e-9 > 1e-12 -> must NOT trip the floor
    strikes = np.array([90.0, 100.0, 110.0])
    mats = np.array([0.5, 0.75, 1.0])
    iv = np.array([[1e-4] * 3, [0.2] * 3, [0.2] * 3])
    surface = GridVolSurface(strikes=strikes, maturities=mats, iv_grid=iv)
    # non-flat in T so Dupire may reject on arbitrage grounds, but NOT with the w-floor message
    try:
        build_dupire_local_vol(surface, spot=100.0, rate_curve=FlatRateCurve(0.02),
                               div_yield=lambda t: 0.0, validate_arbitrage=False,
                               vol_floor=1e-4)
    except NumericalError as exc:
        assert "total implied variance" not in str(exc)
```

(If `GridVolSurface`'s constructor signature differs — check the existing tests in this file for the exact keyword names — adapt the two constructions, nothing else.)

- [ ] **Step 2: Run to verify the first fails**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_dupire_local_vol.py -k w_floor -v`
Expected: `test_w_floor_below_raises` FAILS (no such raise yet); `test_w_floor_above_builds` may already pass.

- [ ] **Step 3: Implement the floor**

In `quantark/volmodels/localvol/dupire.py`:

Add after the imports:

```python
# Absolute floor on total implied variance w = iv^2 * T. Deliberately NOT derived from
# Tolerance.ZERO (1e-10): legitimate short-dated low-vol rows (iv=0.03, T=1e-4 -> w=9e-8)
# must pass, while genuinely degenerate rows must raise instead of silently bypassing
# the butterfly check (safe_divide's 0-fallback pointed the failure the wrong way).
_W_FLOOR = 1e-12
```

Replace (after `w = iv ** 2 * T[:, None]` is computed) the line

```python
    inv_w = safe_divide(1.0, w)
```

with:

```python
    if np.any(w <= _W_FLOOR):
        idx = np.argwhere(w <= _W_FLOOR)
        raise NumericalError(
            f"total implied variance w <= {_W_FLOOR} at (T, K) nodes {idx.tolist()}; "
            "the input surface has a degenerate row — fix the input, do not floor"
        )
    inv_w = 1.0 / w
```

Remove `safe_divide` from the `quantark.util.numerical` import line (it is now unused in this module).

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_dupire_local_vol.py -v`
Expected: all PASS, including the pre-existing flat-surface machine-precision round-trip.

- [ ] **Step 5: Commit**

```bash
git add quantark/volmodels/localvol/dupire.py test/test_dupire_local_vol.py
git commit -m "fix(dupire): WS-A1 — explicit w-floor raise replaces safe_divide zero-fallback

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: WS-A2 — unified leverage clip (code + unit tests)

**Files:**
- Modify: `quantark/volmodels/slv/leverage.py` (add constant)
- Modify: `quantark/volmodels/slv/slv_mc_kernel.py` (thread clip; diagnostics; return shape)
- Modify: `quantark/volmodels/slv/fokkerplanck/config.py` (default from constant)
- Test: `test/test_slv_mc_kernel.py`

**Interfaces:**
- Produces: `DEFAULT_LEVERAGE_CLIP: Tuple[float, float] = (0.05, 20.0)` in `quantark.volmodels.slv.leverage`. `_simulate_slv(..., leverage_clip=DEFAULT_LEVERAGE_CLIP)` returns `(s_terminal, records, n_clip_records)` (3-tuple, was 2). `price_european_slv_mc(..., leverage_clip=DEFAULT_LEVERAGE_CLIP)`, `_calibrate_mc_binning(..., leverage_clip=DEFAULT_LEVERAGE_CLIP)`; `calibrate_leverage_surface` accepts `leverage_clip` in the MC-option slots. MC-binning `LeverageSurface.diagnostics = {"method": "mc_binning", "n_clipped": <int>}`.
- Precomputed `LeverageSurface` inputs are **never** re-clipped (unchanged behavior).

- [ ] **Step 1: Write the failing unit tests**

Append to `test/test_slv_mc_kernel.py`:

```python
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.localvol.surface import LocalVolSurface
from quantark.volmodels.slv.slv_mc_kernel import _calibrate_mc_binning
from quantark.volmodels.slv.leverage import DEFAULT_LEVERAGE_CLIP


def _flat_lv(sigma):
    return LocalVolSurface(strike_grid=np.array([1.0, 1000.0]),
                           time_grid=np.array([0.0, 5.0]),
                           lv_grid=np.full((2, 2), sigma))


def test_clip_band_default_is_ffp_band():
    assert DEFAULT_LEVERAGE_CLIP == (0.05, 20.0)


def test_recorded_leverage_reaches_four_with_new_band():
    # eta=0, kappa=0 -> variance frozen at v0=0.01 on every path; flat sigma_LV=0.4
    # -> true leverage 0.4/0.1 = 4.0 everywhere. Old band capped this at sqrt(10)=3.162.
    params = HestonParams(v0=0.01, kappa=0.0, theta=0.01, sigma=0.5, rho=0.0)
    dt = np.full(4, 0.25)
    surf = _calibrate_mc_binning(100.0, params, _flat_lv(0.4), dt,
                                 np.zeros(4), np.zeros(4), eta=0.0,
                                 num_paths=20_000, num_bins=10, seed=7)
    assert np.allclose(surf.leverage_grid, 4.0, rtol=1e-10)
    assert surf.diagnostics is not None
    assert surf.diagnostics["method"] == "mc_binning"
    assert surf.diagnostics["n_clipped"] == 0


def test_recorded_leverage_clips_at_upper_band():
    # frozen v = 0.0025 (sqrt = 0.05), sigma_LV = 1.2 -> raw L = 24 -> clipped to 20.
    params = HestonParams(v0=0.0025, kappa=0.0, theta=0.0025, sigma=0.5, rho=0.0)
    dt = np.full(2, 0.5)
    surf = _calibrate_mc_binning(100.0, params, _flat_lv(1.2), dt,
                                 np.zeros(2), np.zeros(2), eta=0.0,
                                 num_paths=20_000, num_bins=10, seed=7)
    assert np.allclose(surf.leverage_grid, 20.0, rtol=1e-12)
    assert surf.diagnostics["n_clipped"] == surf.leverage_grid.size
```

- [ ] **Step 2: Run to verify they fail**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_slv_mc_kernel.py -k "clip or four" -v`
Expected: ImportError (`DEFAULT_LEVERAGE_CLIP` doesn't exist) / value 3.162 vs 4.0.

- [ ] **Step 3: Implement**

`quantark/volmodels/slv/leverage.py` — add near the top (after imports):

```python
# Shared clip band for CALIBRATION-GENERATED leverage (both the MC-binning and the
# forward-Fokker-Planck routes). Clip applies to L (not L^2). User-supplied precomputed
# LeverageSurface artifacts are consumed as-is and never re-clipped.
DEFAULT_LEVERAGE_CLIP: "tuple[float, float]" = (0.05, 20.0)
```

`quantark/volmodels/slv/fokkerplanck/config.py` — import it and change the field default:

```python
from quantark.volmodels.slv.leverage import DEFAULT_LEVERAGE_CLIP
...
    leverage_clip: Tuple[float, float] = DEFAULT_LEVERAGE_CLIP
```

(Keep the existing `__post_init__` validation of the tuple.)

`quantark/volmodels/slv/slv_mc_kernel.py`:

1. Import: `from quantark.volmodels.slv.leverage import (BinMethod, DEFAULT_LEVERAGE_CLIP, LeverageSurface, bin_conditional, eval_binned)`.
2. `_simulate_slv` gains a trailing parameter `leverage_clip=DEFAULT_LEVERAGE_CLIP`; unpack `lo, hi = leverage_clip` before the loop; add `n_clip_records = 0` before the loop.
3. Replace the on-the-fly clip block:

```python
            sigma_lv = np.asarray(lv_surface.local_vol(S, t), dtype=float)
            sigma_hat2 = np.clip(sigma_lv * sigma_lv / econd, 1e-8, 10.0)
            sigma_hat = np.sqrt(sigma_hat2)
```

with:

```python
            sigma_lv = np.asarray(lv_surface.local_vol(S, t), dtype=float)
            sigma_hat = np.clip(sigma_lv / np.sqrt(econd), lo, hi)
            sigma_hat2 = sigma_hat * sigma_hat
```

4. Replace the record block:

```python
                sigma_hat2_nodes = np.clip(lv_nodes * lv_nodes / econd_nodes, 1e-8, 10.0)
                records.append(np.sqrt(sigma_hat2_nodes))
```

with:

```python
                raw_nodes = lv_nodes / np.sqrt(econd_nodes)
                clipped_nodes = np.clip(raw_nodes, lo, hi)
                n_clip_records += int(np.sum(clipped_nodes != raw_nodes))
                records.append(clipped_nodes)
```

(update the preceding comment: same clip as the in-simulation `sigma_hat`, now on L directly).
5. Change the return to `return np.exp(log_s), records, n_clip_records`.
6. `price_european_slv_mc`: add parameter `leverage_clip: Tuple[float, float] = DEFAULT_LEVERAGE_CLIP` (after `leverage_surface`); validate and thread it:

```python
    lo, hi = leverage_clip
    if not (np.isfinite(lo) and np.isfinite(hi) and 0.0 < lo < hi):
        raise ValidationError("leverage_clip must be a finite positive ordered (lo, hi) tuple")
```

Update its `_simulate_slv` call to unpack the 3-tuple: `s_terminal, _, _ = _simulate_slv(..., leverage_surface=leverage_surface, leverage_clip=leverage_clip)`.
7. `_calibrate_mc_binning`: same new parameter + validation; unpack `_, records, n_clip = _simulate_slv(..., record_grid=strike_grid, leverage_clip=leverage_clip)`; construct the surface with diagnostics:

```python
    return LeverageSurface(time_grid=record_times, strike_grid=strike_grid,
                           leverage_grid=leverage_grid,
                           diagnostics={"method": "mc_binning", "n_clipped": int(n_clip)})
```

8. `calibrate_leverage_surface`: add `leverage_clip=_UNSET` after `strike_span_stds` in the signature and to the `mc_opts` dict (so passing it under FFP raises the existing "MC options … not valid" error; FFP users set the band via `fp_config.leverage_clip`).

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_slv_mc_kernel.py test/test_leverage_calibration_dispatch.py test/test_fp_config.py -v`
Expected: new tests PASS. Pre-existing tests that pinned values in the old saturated band: update expected values **deliberately** and note the widened band in the assertion comment (spec §WS-A2 golden policy).

- [ ] **Step 5: Run the wider SLV suites to catch golden movement**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_heston_slv_mc_engines.py test/test_heston_slv_pde_engines.py test/test_slv_pde_kernel.py test/test_slv_calibration_spec_method.py -v`
Expected: PASS if no fixture saturates the old band (spec open-question 1). If a golden moves, verify the fixture's true leverage exceeds 3.162 (i.e. the old cap was binding), then update the golden with a comment `# widened leverage clip band (0.05, 20): see 2026-07-04 volmodels spec WS-A2`.

- [ ] **Step 6: Commit**

```bash
git add quantark/volmodels/slv test/
git commit -m "fix(slv): WS-A2 — unify leverage clip band across MC-binning and FFP routes

MC on-the-fly binning previously clipped sigma_hat^2 to [1e-8, 10] (L <= 3.162)
while FFP allows (0.05, 20); the two calibration routes were not cross-checks in
high-leverage regimes. Single DEFAULT_LEVERAGE_CLIP source of truth; clip on L;
n_clipped diagnostics for the MC route.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: WS-A2 — cross-route acceptance gate (MC-binning vs FFP)

**Files:**
- Test: `test/test_slv_calibration_spec_method.py` (append)

**Interfaces:**
- Consumes: Task 3's `_calibrate_mc_binning(..., leverage_clip=...)` and existing `calibrate_leverage_surface_fp`.

- [ ] **Step 1: Write the two gate tests**

Append to `test/test_slv_calibration_spec_method.py` (define the skewed LV fixture inline — it is a direct `LocalVolSurface`, consumed identically by both routes):

```python
def _skewed_lv():
    strikes = np.linspace(60.0, 160.0, 21)
    times = np.linspace(0.0, 1.25, 6)
    base = 0.25 - 0.0012 * (strikes - 100.0)          # mild downward skew, stays positive
    lv = np.tile(base, (times.size, 1))
    return LocalVolSurface(strike_grid=strikes, time_grid=times, lv_grid=lv)


def _cross_route_max_rel_diff(params, lv_surface, num_paths=200_000):
    dt = np.full(12, 1.0 / 12.0)
    rf = np.full(12, 0.02)
    cf = np.zeros(12)
    mc = _calibrate_mc_binning(100.0, params, lv_surface, dt, rf, cf, eta=1.0,
                               seed=42, num_paths=num_paths, num_bins=30)
    fp = calibrate_leverage_surface_fp(100.0, params, lv_surface, dt, rf, cf, eta=1.0)
    # shared evaluation grid: +/- 2 total-vol stds around s0 (spec WS-A2 acceptance 2)
    v_ref = max(params.v0, params.theta)
    width = 2.0 * np.sqrt(v_ref * 1.0)
    s_eval = 100.0 * np.exp(np.linspace(-width, width, 41))
    t_eval = np.concatenate([[0.0], np.cumsum(dt)])[:-1]
    tt, ss = np.meshgrid(t_eval, s_eval, indexing="ij")
    l_mc = mc.leverage(ss, tt)
    l_fp = fp.leverage(ss, tt)
    return float(np.max(np.abs(l_mc - l_fp) / np.abs(l_fp)))


def test_cross_route_standard_fixture():
    params = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.8, rho=-0.7)
    assert _cross_route_max_rel_diff(params, _skewed_lv()) < 0.10


def test_cross_route_high_leverage_fixture():
    # true L ~ 0.40 / sqrt(0.01) = 4 everywhere: fails by construction under the old
    # MC clip (pinned at 3.162); must pass the same 0.10 gate under the unified band.
    params = HestonParams(v0=0.01, kappa=1.5, theta=0.01, sigma=0.8, rho=-0.7)
    assert _cross_route_max_rel_diff(params, _flat_lv_040()) < 0.10


def _flat_lv_040():
    return LocalVolSurface(strike_grid=np.array([1.0, 1000.0]),
                           time_grid=np.array([0.0, 5.0]),
                           lv_grid=np.full((2, 2), 0.40))
```

Match the file's existing imports (`_calibrate_mc_binning`, `calibrate_leverage_surface_fp`, `HestonParams`, `LocalVolSurface`, `np`) — add any missing ones.

- [ ] **Step 2: Run the gates**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_slv_calibration_spec_method.py -k cross_route -v`
Expected: both PASS. Contingencies (do NOT change the fixtures):
- FFP raises negative-mass/mass-loss on the high-leverage fixture (Feller strongly violated): pass a finer `FpCalibrationConfig(n_x=301, n_z=151)` via the `config=` argument — resolution, not approximation, per repo convention.
- Threshold exceeded only in the outermost eval nodes: verify the ±2-std window is honored; if MC tail noise dominates, raise `num_paths` (spec pins 200k as the *minimum* reported configuration — record what was needed).

- [ ] **Step 3: Commit**

```bash
git add test/test_slv_calibration_spec_method.py
git commit -m "test(slv): WS-A2 cross-route gate — MC-binning vs FFP leverage agreement incl. high-leverage fixture

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: WS-D2a — shared non-uniform FD stencils

**Files:**
- Create: `quantark/util/numerical/finite_difference.py`
- Modify: `quantark/util/numerical/__init__.py` (export)
- Modify: `quantark/volmodels/localvol/dupire.py` (consume; delete local copies)
- Test: `test/test_finite_difference.py` (create)

**Interfaces:**
- Produces: `fd1_nonuniform(values: np.ndarray, x: np.ndarray) -> np.ndarray` and `fd2_nonuniform(values, x) -> np.ndarray` — derivative along the **last axis**, second-order on non-uniform grids, 3-point Lagrange one-sided at both ends, exact for quadratics. These are verbatim moves of `_fd1`/`_fd2` from `dupire.py` (same math, renamed).

- [ ] **Step 1: Write the failing tests**

Create `test/test_finite_difference.py`:

```python
import numpy as np

from quantark.util.numerical import fd1_nonuniform, fd2_nonuniform


def test_exact_for_quadratics_nonuniform():
    x = np.array([0.0, 0.1, 0.35, 0.5, 1.0, 1.7])
    y = 3.0 * x * x - 2.0 * x + 5.0
    np.testing.assert_allclose(fd1_nonuniform(y, x), 6.0 * x - 2.0, rtol=1e-12)
    np.testing.assert_allclose(fd2_nonuniform(y, x), np.full_like(x, 6.0), rtol=1e-12)


def test_batched_last_axis():
    x = np.array([0.0, 0.2, 0.7, 1.0])
    ys = np.vstack([x * x, 2.0 * x * x + x])
    d = fd1_nonuniform(ys, x)
    np.testing.assert_allclose(d[0], 2.0 * x, rtol=1e-12)
    np.testing.assert_allclose(d[1], 4.0 * x + 1.0, rtol=1e-12)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_finite_difference.py -v`
Expected: ImportError.

- [ ] **Step 3: Create the module and rewire dupire**

Create `quantark/util/numerical/finite_difference.py` with a module docstring ("Second-order finite-difference stencils on non-uniform 1D grids (last-axis batched)."), containing the two functions moved **verbatim** from `quantark/volmodels/localvol/dupire.py::_fd1/_fd2` (same bodies and per-function docstrings), renamed `fd1_nonuniform` / `fd2_nonuniform` with the first parameter renamed `values`. Import only `numpy`.

Add to `quantark/util/numerical/__init__.py`'s imports/`__all__`: `fd1_nonuniform`, `fd2_nonuniform` (follow the file's existing export style).

In `dupire.py`: delete `_fd1`/`_fd2`; add `from quantark.util.numerical import fd1_nonuniform, fd2_nonuniform`; replace the three call sites (`_fd1(w, ln_k)`, `_fd2(w, ln_k)`, `_fd1(w.T, T).T`, `_fd1(ln_fwd, T)`) with the new names.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_finite_difference.py test/test_dupire_local_vol.py -v`
Expected: all PASS (pure move — Dupire outputs bit-identical).

- [ ] **Step 5: Commit**

```bash
git add quantark/util/numerical quantark/volmodels/localvol/dupire.py test/test_finite_difference.py
git commit -m "refactor(numerical): WS-D2 — promote Dupire's non-uniform FD stencils to util/numerical

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: WS-D2b — shared tridiagonal solvers

**Files:**
- Create: `quantark/util/numerical/tridiag.py`
- Modify: `quantark/util/numerical/__init__.py` (export)
- Test: `test/test_tridiag.py` (create)

**Interfaces:**
- Produces:
  - `solve_tridiag_batch(sub, diag, sup, rhs) -> np.ndarray` — all args shape `(n_sys, N)`, `sub[:, 0]` and `sup[:, -1]` ignored (matches the ADI full-length convention). Sequential Thomas sweep over N, vectorized across systems ⇒ **bit-identical per system** to the scalar Thomas. Raises `NumericalError` if any pivot `|denom| < 1e-14` (the ADI threshold).
  - `solve_tridiag(sub, diag, sup, rhs) -> np.ndarray` — 1D single system, `sub`/`sup` length `N-1` (the localvol convention), delegates to `scipy.linalg.solve_banded((1, 1), ab, rhs, check_finite=False)`; catches `scipy.linalg.LinAlgError` → raises `NumericalError("singular tridiagonal system")`.

- [ ] **Step 1: Write the failing tests**

Create `test/test_tridiag.py`:

```python
import numpy as np
import pytest

from quantark.util.exceptions import NumericalError
from quantark.util.numerical import solve_tridiag, solve_tridiag_batch


def _thomas_reference(a, b, c, d):
    # scalar Thomas, full-length convention (a[0], c[-1] ignored) — mirrors the ADI solvers
    n = len(d)
    cp = np.zeros(n); dp = np.zeros(n); x = np.zeros(n)
    cp[0] = c[0] / b[0]; dp[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def test_batch_bit_identical_to_scalar_thomas():
    rng = np.random.default_rng(0)
    n_sys, N = 7, 40
    sub = rng.normal(size=(n_sys, N))
    sup = rng.normal(size=(n_sys, N))
    diag = 4.0 + rng.random((n_sys, N))          # diagonally dominant
    rhs = rng.normal(size=(n_sys, N))
    out = solve_tridiag_batch(sub, diag, sup, rhs)
    for k in range(n_sys):
        ref = _thomas_reference(sub[k], diag[k], sup[k], rhs[k])
        assert np.array_equal(out[k], ref)        # bit-identical, not just close


def test_batch_zero_pivot_raises():
    sub = np.zeros((1, 3)); sup = np.zeros((1, 3))
    diag = np.array([[1.0, 0.0, 1.0]])
    with pytest.raises(NumericalError):
        solve_tridiag_batch(sub, diag, sup, np.ones((1, 3)))


def test_single_matches_dense_solve():
    rng = np.random.default_rng(1)
    N = 50
    sub = rng.normal(size=N - 1); sup = rng.normal(size=N - 1)
    diag = 4.0 + rng.random(N); rhs = rng.normal(size=N)
    A = np.diag(diag) + np.diag(sub, -1) + np.diag(sup, 1)
    np.testing.assert_allclose(solve_tridiag(sub, diag, sup, rhs),
                               np.linalg.solve(A, rhs), rtol=1e-10, atol=1e-12)


def test_single_singular_raises():
    with pytest.raises(NumericalError):
        solve_tridiag(np.zeros(1), np.zeros(2), np.zeros(1), np.ones(2))
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_tridiag.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement**

Create `quantark/util/numerical/tridiag.py`:

```python
"""Tridiagonal solvers: batched Thomas (bit-identical to the scalar sweep) and a
LAPACK-banded single-system wrapper. See the 2026-07-04 volmodels spec, WS-B2."""

from __future__ import annotations

import numpy as np
from scipy.linalg import LinAlgError, solve_banded

from quantark.util.exceptions import NumericalError

_PIVOT_MIN = 1e-14


def solve_tridiag_batch(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray,
                        rhs: np.ndarray) -> np.ndarray:
    """Solve n_sys independent tridiagonal systems, all inputs shape (n_sys, N).

    Full-length convention: sub[:, 0] and sup[:, -1] are ignored. The sweep is the
    sequential Thomas recurrence over N vectorized across systems, so each system's
    result is bit-identical to a scalar Thomas solve with the same convention.
    Raises NumericalError if any pivot magnitude falls below 1e-14.
    """
    diag = np.asarray(diag, dtype=float)
    n_sys, n = diag.shape
    sub = np.asarray(sub, dtype=float)
    sup = np.asarray(sup, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    cp = np.empty((n_sys, n))
    dp = np.empty((n_sys, n))
    if np.any(np.abs(diag[:, 0]) < _PIVOT_MIN):
        raise NumericalError("zero pivot in batched tridiagonal solve (refine grid)")
    cp[:, 0] = sup[:, 0] / diag[:, 0]
    dp[:, 0] = rhs[:, 0] / diag[:, 0]
    for i in range(1, n):
        denom = diag[:, i] - sub[:, i] * cp[:, i - 1]
        if np.any(np.abs(denom) < _PIVOT_MIN):
            raise NumericalError("zero pivot in batched tridiagonal solve (refine grid)")
        cp[:, i] = sup[:, i] / denom
        dp[:, i] = (rhs[:, i] - sub[:, i] * dp[:, i - 1]) / denom
    x = np.empty((n_sys, n))
    x[:, n - 1] = dp[:, n - 1]
    for i in range(n - 2, -1, -1):
        x[:, i] = dp[:, i] - cp[:, i] * x[:, i + 1]
    return x


def solve_tridiag(sub: np.ndarray, diag: np.ndarray, sup: np.ndarray,
                  rhs: np.ndarray) -> np.ndarray:
    """Solve one tridiagonal system via LAPACK (scipy solve_banded).

    Off-diagonals have length N-1. Arithmetic order differs from the Thomas sweep, so
    results are numerically equivalent (not bit-identical) to a scalar Thomas solve.
    Raises NumericalError on a singular system.
    """
    diag = np.asarray(diag, dtype=float)
    n = diag.shape[0]
    ab = np.zeros((3, n))
    ab[0, 1:] = np.asarray(sup, dtype=float)
    ab[1, :] = diag
    ab[2, :-1] = np.asarray(sub, dtype=float)
    try:
        return solve_banded((1, 1), ab, np.asarray(rhs, dtype=float), check_finite=False)
    except LinAlgError as exc:
        raise NumericalError(f"singular tridiagonal system: {exc}") from exc
```

Export both from `quantark/util/numerical/__init__.py`.

Note on `test_batch_bit_identical_to_scalar_thomas`: the reference and the batch compute `sup[:, i] / denom` etc. with identical scalar operations per system — `np.array_equal` is the assertion, and it must hold exactly. If it does not, the implementation deviated from the sweep order; fix the implementation, never relax this test.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_tridiag.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add quantark/util/numerical test/test_tridiag.py
git commit -m "feat(numerical): WS-D2 — shared batched-Thomas and banded tridiagonal solvers

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: WS-B2a — LV PDE solve_banded swap (1e-12 gate)

**Files:**
- Modify: `quantark/volmodels/localvol/pde_kernel.py`
- Test: `test/test_local_vol_pde_kernel.py`

**Interfaces:**
- Consumes: `solve_tridiag` from Task 6.
- Produces: same public API; `_thomas_solve` deleted.

- [ ] **Step 1: Capture-before regression test (must pass BEFORE the swap)**

Append to `test/test_local_vol_pde_kernel.py`:

```python
def test_price_regression_pinned_for_solver_swap():
    # Captured from the pre-solve_banded implementation (Thomas). Gate: <= 1e-12 rel.
    surf = LocalVolSurface(strike_grid=np.array([50.0, 100.0, 200.0]),
                           time_grid=np.array([0.0, 1.0]),
                           lv_grid=np.array([[0.30, 0.22, 0.20], [0.32, 0.24, 0.21]]))
    dt = np.full(50, 0.02)
    price = price_european_lv_pde(100.0, 105.0, True, 1.0, surf, dt,
                                  np.full(50, 0.03), np.full(50, 0.01), n_s=200)
    assert np.isclose(price, PINNED_PRICE, rtol=1e-12)
```

First run the body once with the CURRENT implementation and paste the printed value as `PINNED_PRICE = <literal>` (use a quick inline `python -c` run; print with `repr()` for full precision). Verify the test passes pre-change.

- [ ] **Step 2: Swap the solver**

In `quantark/volmodels/localvol/pde_kernel.py`: delete `_thomas_solve`; add `from quantark.util.numerical import solve_tridiag`; replace the call

```python
        v[1:-1] = _thomas_solve(sub_A, diag_A, sup_A, rhs)
```

with:

```python
        v[1:-1] = solve_tridiag(sub_A, diag_A, sup_A, rhs)
```

(the existing `sub_A`/`sup_A` are already length m−1 — same convention). Remove the now-unused `NumericalError` import **only if** nothing else in the module uses it (check first).

- [ ] **Step 3: Run tests**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_local_vol_pde_kernel.py test/test_local_vol_equity_engines.py test/test_local_vol_fx_engines.py -v`
Expected: all PASS including the pinned regression (LAPACK vs Thomas differences are ≪1e-12 for these well-conditioned CN systems).

- [ ] **Step 4: Commit**

```bash
git add quantark/volmodels/localvol/pde_kernel.py test/test_local_vol_pde_kernel.py
git commit -m "perf(localvol): WS-B2 — LV CN solver uses LAPACK banded solve (1e-12-pinned regression)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: WS-B2b — Heston ADI batched solves + dense coefficient caching

**Files:**
- Modify: `quantark/volmodels/heston/pde_kernel.py`
- Test: `test/test_heston_pde_kernel.py`

**Interfaces:**
- Consumes: `solve_tridiag_batch` from Task 6.
- Produces: same public API. Internally: `_tri_S(dt, θ)` and `_tri_V(dt, θ)` return `(n_sys, N)` coefficient arrays; dense-mode cache `_S_tri_cache` / `_V_tri_cache` keyed `(float(dt_step), float(theta_loc))`.

- [ ] **Step 1: Capture-before regression tests**

Append to `test/test_heston_pde_kernel.py` — pin BOTH schemes and both call/put at full precision from the CURRENT implementation (same capture procedure as Task 7):

```python
_REG_PARAMS = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.5, rho=-0.7)

@pytest.mark.parametrize("scheme,is_call,pinned", [
    (ADIScheme.CRAIG_SNEYD, True,  PIN_CS_CALL),
    (ADIScheme.CRAIG_SNEYD, False, PIN_CS_PUT),
    (ADIScheme.DOUGLAS,     True,  PIN_DO_CALL),
])
def test_adi_price_regression_pinned_for_batch_swap(scheme, is_call, pinned):
    p = price_european_heston_pde(100.0, 100.0, is_call, 1.0, _REG_PARAMS, 0.03, 0.01,
                                  n_x=120, n_v=60, n_t=60, scheme=scheme)
    assert np.isclose(p, pinned, rtol=1e-13)
```

Verify all pass pre-change.

- [ ] **Step 2: Vectorize `_tri_S` / `_tri_V` and add dense caches**

In `_HestonADI`:

Replace `_tri_S` (currently returns a Python list of per-j tuples) with a batched builder using the **same expression forms** (bit-identity requirement):

```python
    def _tri_S(self, dt_step, theta_loc):
        # (N_V, N_S) batched tridiagonal coefficients; boundary rows are identity.
        key = (float(dt_step), float(theta_loc))
        cached = self._S_tri_cache.get(key)
        if cached is not None:
            return cached
        dx = self.dx
        V = np.maximum(self.V_grid, 1e-10)[:, None]            # (N_V, 1)
        c2 = 0.5 * V / (dx * dx)
        c1 = ((self.r - self.q) - 0.5 * V) / (2.0 * dx)
        alpha = np.zeros((self.N_V, self.N_S)); beta = np.ones((self.N_V, self.N_S))
        gamma = np.zeros((self.N_V, self.N_S))
        alpha[:, 1:-1] = -theta_loc * dt_step * (c2 - c1)
        beta[:, 1:-1] = 1.0 + theta_loc * dt_step * (2.0 * c2)
        gamma[:, 1:-1] = -theta_loc * dt_step * (c2 + c1)
        self._S_tri_cache[key] = (alpha, beta, gamma)
        return alpha, beta, gamma
```

(Note `c2`/`c1` are `(N_V, 1)` and broadcast across the interior slice — identical scalar values per row as before. `max(float(...), 1e-10)` becomes `np.maximum(..., 1e-10)`, same values.)

Initialize `self._S_tri_cache = {}` and `self._V_tri_cache = {}` in `__init__` (keep the existing `_S_lu_cache`/`_V_lu_cache` for the sparse path). `_tri_V` keeps returning `(alpha, beta, gamma)` 1D arrays but gains the same keyed cache.

**Sparse path:** `_ensure_S_lus` currently iterates `_tri_S`'s per-j list. Update it to consume the batched arrays:

```python
        alpha, beta, gamma = self._tri_S(dt_step, theta_loc)
        for j in range(self.N_V):
            A = sp.diags([alpha[j, 1:], beta[j], gamma[j, :-1]], offsets=[-1, 0, 1], format="csc")
            lus.append(spla.splu(A))
```

- [ ] **Step 3: Batch `_solve_S` and `_solve_V` (dense path)**

Replace the dense branch of `_solve_S`:

```python
        else:
            alpha, beta, gamma = self._tri_S(dt_step, theta_loc)
            rhs = (source - theta_loc * dt_step * A1U)          # (N_S, N_V)
            if self._opt_is_call:
                rhs[0, :] = 0.0
                rhs[-1, :] = max(0.0, self.S_max * np.exp(-self.q * tau)
                                 - self.K * np.exp(-self.r * tau))
            else:
                rhs[0, :] = self.K * np.exp(-self.r * tau)
                rhs[-1, :] = 0.0
            Y = solve_tridiag_batch(alpha, beta, gamma, rhs.T).T
```

(`rhs.T` is `(N_V, N_S)` = one system per V-slice; the boundary values are exactly what `_s_boundary_rhs` set per column.) Keep the sparse branch as-is. Keep the trailing `self._bc(Y, tau)`.

Replace the dense branch of `_solve_V` similarly:

```python
        else:
            alpha, beta, gamma = self._tri_V(dt_step, theta_loc)
            rhs = source - theta_loc * dt_step * A2U            # (N_S, N_V)
            rhs[:, 0] = 0.0
            rhs[:, -1] = 0.0
            sysN = self.N_S
            U_out = solve_tridiag_batch(np.broadcast_to(alpha, (sysN, self.N_V)),
                                        np.broadcast_to(beta, (sysN, self.N_V)),
                                        np.broadcast_to(gamma, (sysN, self.N_V)),
                                        rhs)
```

Delete the now-unused `_thomas` static method; add `from quantark.util.numerical import solve_tridiag_batch` to the imports. **Do not modify `source` in place** — `source - theta_loc * dt_step * A1U` already allocates a fresh array, so the boundary-row assignments above are safe.

- [ ] **Step 4: Run tests + full Heston suites**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_heston_pde_kernel.py test/test_heston_pde_engines.py -v`
Expected: all PASS including the 1e-13-pinned regressions (rounding-identical: same per-system arithmetic order). If a pinned test fails at the last ulp, inspect for an expression-form deviation (e.g. `V / (dx*dx)` vs `V / dx / dx`) and restore the original form — do not relax the tolerance.

- [ ] **Step 5: Benchmark (≥10× gate)**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python - <<'EOF'
import time
import numpy as np
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.heston.pde_kernel import price_european_heston_pde
p = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.5, rho=-0.7)
t0 = time.perf_counter()
price = price_european_heston_pde(100.0, 100.0, True, 1.0, p, 0.03, 0.01,
                                  n_x=200, n_v=100, n_t=100)
print(f"200x100x100 solve: {time.perf_counter()-t0:.2f}s  price={price:.6f}")
EOF
```

Run this on `main`'s version first (e.g. `git stash` → run → `git stash pop`) to record the before number. Gate: after/before ≥ 10×. Record both numbers in the commit message.

- [ ] **Step 6: Commit**

```bash
git add quantark/volmodels/heston/pde_kernel.py test/test_heston_pde_kernel.py
git commit -m "perf(heston): WS-B2 — batched tridiagonal ADI solves + dense coefficient caching

Bit-identical to the sequential Thomas (1e-13-pinned regressions).
Benchmark 200x100x100 CS solve: <before>s -> <after>s.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: WS-B2c — SLV ADI batched solves

**Files:**
- Modify: `quantark/volmodels/slv/slv_pde_kernel.py`
- Test: `test/test_slv_pde_kernel.py`

**Interfaces:**
- Consumes: `solve_tridiag_batch` from Task 6.
- Produces: same public API. S-coefficients rebuilt per step (L is time-dependent — no cache), built vectorized as `(N_V, N_S)` arrays.

- [ ] **Step 1: Capture-before regression test**

Append to `test/test_slv_pde_kernel.py` (pin from the CURRENT implementation, capture procedure as Task 7; build a small deterministic `LeverageSurface` inline):

```python
def test_slv_adi_price_regression_pinned_for_batch_swap():
    params = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.5, rho=-0.7)
    lev = LeverageSurface(time_grid=np.array([0.0, 0.5, 1.0]),
                          strike_grid=np.array([60.0, 100.0, 160.0]),
                          leverage_grid=np.array([[1.3, 1.0, 0.9],
                                                  [1.25, 1.0, 0.92],
                                                  [1.2, 1.0, 0.95]]))
    p = price_european_slv_pde(100.0, 100.0, True, 1.0, params, lev, 0.03, 0.01,
                               n_x=120, n_v=60, n_t=60)
    assert np.isclose(p, PINNED_SLV_PRICE, rtol=1e-13)
```

- [ ] **Step 2: Batch the solves**

In `_HestonSLVADI._solve_S`, replace the per-j loop with a vectorized coefficient build + one batch solve (same expression forms):

```python
    def _solve_S(self, source, A1U, dt_step, theta_loc, tau, t_mid):
        L2 = self._L(t_mid) ** 2                               # (N_S-2,)
        V = np.maximum(self.V_grid, 1e-10)[:, None]            # (N_V, 1)
        c2 = 0.5 * (L2[None, :] * V) / (self.dx * self.dx)     # (N_V, N_S-2)
        c1 = ((self.r - self.q) - 0.5 * (L2[None, :] * V)) / (2.0 * self.dx)
        a = np.zeros((self.N_V, self.N_S)); b = np.ones((self.N_V, self.N_S))
        c = np.zeros((self.N_V, self.N_S))
        a[:, 1:-1] = -theta_loc * dt_step * (c2 - c1)
        b[:, 1:-1] = 1.0 + theta_loc * dt_step * (2.0 * c2)
        c[:, 1:-1] = -theta_loc * dt_step * (c2 + c1)
        rhs = source - theta_loc * dt_step * A1U               # (N_S, N_V)
        if self._opt_is_call:
            rhs[0, :] = 0.0
            rhs[-1, :] = max(0.0, self.S_max * np.exp(-self.q * tau)
                             - self.K * np.exp(-self.r * tau))
        else:
            rhs[0, :] = self.K * np.exp(-self.r * tau)
            rhs[-1, :] = 0.0
        Y = solve_tridiag_batch(a, b, c, rhs.T).T
        self._bc(Y, tau)
        return Y
```

(Note the current per-j code computes `c2 = 0.5 * (L2 * vj) / (self.dx * self.dx)` — the batched `L2[None, :] * V` reproduces the same scalar products elementwise.)

`_solve_V`: identical refactor to Task 8's Heston `_solve_V` (coefficients are 1D, broadcast to `(N_S, N_V)` systems, rhs boundary zeros, batch solve, then `_bc`). Delete `_thomas`; import `solve_tridiag_batch`.

- [ ] **Step 3: Run tests**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_slv_pde_kernel.py test/test_heston_slv_pde_engines.py -v`
Expected: all PASS including the pinned regression.

- [ ] **Step 4: Commit**

```bash
git add quantark/volmodels/slv/slv_pde_kernel.py test/test_slv_pde_kernel.py
git commit -m "perf(slv): WS-B2 — batched tridiagonal solves in the backward SLV ADI

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: WS-B4a — QUADEXP-only uniform draws (seed-stability pinned)

**Files:**
- Modify: `quantark/volmodels/heston/mc_kernel.py:168-184`
- Test: `test/test_heston_mc_kernel.py`

**Interfaces:**
- Produces: same public API; `u_var` is `None` for EULER/EULERLOG (drawn only for QUADEXP). Seed-identical prices for ALL schemes (u draws happen after z draws, so removing them cannot perturb the z-stream; QUADEXP still draws the identical u-stream).

- [ ] **Step 1: Pin current seeded prices (must pass BEFORE the change)**

Append to `test/test_heston_mc_kernel.py`, capturing the three literals from the CURRENT implementation first:

```python
@pytest.mark.parametrize("scheme,pinned", [
    (HestonMCScheme.EULER,    PIN_MC_EULER),
    (HestonMCScheme.EULERLOG, PIN_MC_EULERLOG),
    (HestonMCScheme.QUADEXP,  PIN_MC_QUADEXP),
])
def test_seed_stability_pinned_across_u_var_change(scheme, pinned):
    params = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.5, rho=-0.7)
    dt = np.full(12, 1.0 / 12.0)
    p = price_european_heston_mc(100.0, 100.0, True, params, dt,
                                 np.full(12, 0.03), np.full(12, 0.01),
                                 disc_factor=np.exp(-0.03), scheme=scheme,
                                 num_paths=20_000, seed=42)
    assert p == pinned    # exact: the random streams must be unchanged
```

Capture three **separate** literals (EULER, EULERLOG, QUADEXP) from the current implementation — the whole point is pinning the EULER/EULERLOG z-streams and the QUADEXP u-stream across the draw-block change.

- [ ] **Step 2: Make u_var conditional**

In `price_european_heston_mc` replace the draw block:

```python
    need_u = scheme == HestonMCScheme.QUADEXP
    if use_antithetic:
        z_var_h = rng.standard_normal((half, M))
        z_ind_h = rng.standard_normal((half, M))
        z_var = np.concatenate([z_var_h, -z_var_h], axis=0)
        z_ind = np.concatenate([z_ind_h, -z_ind_h], axis=0)
        if need_u:
            u_var_h = rng.random((half, M))
            u_var = np.concatenate([u_var_h, 1.0 - u_var_h], axis=0)
        else:
            u_var = None
    else:
        z_var = rng.standard_normal((n_eff, M))
        z_ind = rng.standard_normal((n_eff, M))
        u_var = rng.random((n_eff, M)) if need_u else None
```

(The current code draws `z_var_h`, `z_ind_h`, `u_var_h` in that order — preserve z-before-u ordering exactly.) `_simulate_terminal_spot` needs no change: the EULER/EULERLOG branches never touch `u_var`.

- [ ] **Step 3: Run tests**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_heston_mc_kernel.py test/test_heston_mc_engines.py -v`
Expected: all PASS with `==` pinned equality on all three schemes.

- [ ] **Step 4: Commit**

```bash
git add quantark/volmodels/heston/mc_kernel.py test/test_heston_mc_kernel.py
git commit -m "perf(heston): WS-B4 — draw QE uniforms only for QUADEXP (seed-stability pinned)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: WS-B4b — `bin_conditional` split-index rewrite (exact-equality gated)

**Files:**
- Modify: `quantark/volmodels/slv/leverage.py:27-70`
- Test: `test/test_slv_mc_kernel.py` (or the file that currently tests `bin_conditional` — grep `bin_conditional` under `test/` and extend that file)

**Interfaces:**
- Produces: identical `(boundaries, bin_means)` outputs. Tie convention pinned: bin 0 = `[b_0, b_1]`, bin k>0 = `(b_k, b_{k+1}]`; splits via `np.searchsorted(s_sorted, boundaries[1:-1], side="right")`.
- **Spec deviation, documented:** the spec names `np.add.reduceat`; this task uses split indices + per-slice `np.mean` instead, because both `reduceat` and cumsum-differencing change floating-point summation order and would break the exact-equality acceptance the spec itself mandates. The perf win — eliminating the O(bins×n) boolean-mask scans — is preserved; the remaining per-bin loop is O(num_bins) slice means.

- [ ] **Step 1: Write the equality tests (against the CURRENT implementation, so: implement reference copy first)**

In the test file, snapshot the current implementation as a reference:

```python
def _bin_conditional_reference(stock_values, variance_values, num_bins, method):
    # verbatim copy of the pre-rewrite quantark implementation (masks + np.where)
    ...  # paste the current function body here before modifying the source
```

Then the gate tests:

```python
@pytest.mark.parametrize("method", [BinMethod.EQUIDISTANT, BinMethod.EQUAL_WEIGHTED])
def test_bin_conditional_exact_equality_random(method):
    rng = np.random.default_rng(3)
    for _ in range(5):
        S = rng.lognormal(4.6, 0.3, size=5000)
        v = rng.gamma(2.0, 0.02, size=5000)
        b_new, m_new = bin_conditional(S, v, 20, method)
        b_ref, m_ref = _bin_conditional_reference(S, v, 20, method)
        assert np.array_equal(b_new, b_ref)
        assert np.array_equal(m_new, m_ref)


@pytest.mark.parametrize("method", [BinMethod.EQUIDISTANT, BinMethod.EQUAL_WEIGHTED])
def test_bin_conditional_exact_equality_ties_and_empty_bins(method):
    rng = np.random.default_rng(4)
    # heavy ties: quantized spots -> many samples exactly on equal-weighted boundaries;
    # a hole in the support -> empty equidistant bins (neighbor-fill path)
    S = np.round(rng.lognormal(4.6, 0.4, size=3000), 0)
    S[S > 120] += 60.0
    v = rng.gamma(2.0, 0.02, size=3000)
    b_new, m_new = bin_conditional(S, v, 15, method)
    b_ref, m_ref = _bin_conditional_reference(S, v, 15, method)
    assert np.array_equal(b_new, b_ref)
    assert np.array_equal(m_new, m_ref)
```

Run them with `bin_conditional` still the old code: they pass trivially (new == old == reference). That's expected; they become the gate after Step 2.

- [ ] **Step 2: Rewrite `bin_conditional`**

Replace the per-bin mask loop (keep the boundary construction and the empty-bin neighbor-fill loop unchanged):

```python
    # contiguous segments on the sorted array; side="right" puts boundary-tied samples
    # in the LEFT bin — exactly the historical mask convention (bin 0 inclusive both
    # edges, bin k>0 half-open (b_k, b_{k+1}])
    splits = np.searchsorted(s_sorted, boundaries[1:-1], side="right")
    seg_starts = np.concatenate([[0], splits]).astype(int)
    seg_ends = np.concatenate([splits, [n]]).astype(int)
    bin_means = np.zeros(num_bins)
    bin_counts = seg_ends - seg_starts
    for k in range(num_bins):
        if bin_counts[k] > 0:
            bin_means[k] = float(np.mean(v_sorted[seg_starts[k]:seg_ends[k]]))
```

(`np.mean` on the identical contiguous slice contents reproduces the old `np.mean(v_sorted[idx])` bit-for-bit.) One historical subtlety: the old first-bin mask was `s_sorted >= boundaries[0]`, which is every element (boundaries[0] = s_min) — the segment formulation matches. The old interior masks `> bl & <= br` on the sorted array select exactly `[split_{k}, split_{k+1})` — verified by the tie tests.

- [ ] **Step 3: Run tests**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_slv_mc_kernel.py test/test_leverage_surface_diagnostics.py -k "bin_conditional or slv" -v`
Expected: exact-equality tests PASS. Then run the full SLV MC suite: `... -m pytest -n0 test/test_slv_mc_kernel.py test/test_slv_calibration_spec_method.py -v` — all PASS (outputs unchanged ⇒ no golden movement).

- [ ] **Step 4: Commit**

```bash
git add quantark/volmodels/slv/leverage.py test/
git commit -m "perf(slv): WS-B4 — bin_conditional via sorted split indices (exact-equality gated)

Replaces O(bins*n) boolean-mask scans with searchsorted splits + per-slice means.
Deviation from spec's reduceat suggestion documented: summation-order change would
break the exact-equality gate the spec mandates.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: Full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the entire test suite**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest`
Expected: all PASS (parallel run). Any failure outside `test/test_*` files touched above means an unexpected consumer — investigate before proceeding; do not paper over.

- [ ] **Step 2: Spot-check an SLV example end-to-end**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python example/snowball_mc_demo.py` (or the SLV/Heston demo present in `example/` — `ls example/ | grep -i -E "heston|slv|local_vol"`) and confirm it completes without error.

- [ ] **Step 3: Commit any stragglers, then hand off to code review**

```bash
git status --short   # should be clean or test-only leftovers
```

---

## Self-Review Notes

- **Spec coverage:** WS-A1 → Task 2; WS-A2 → Tasks 3–4 (all four acceptance bullets); WS-A3 → Task 1; WS-B2 → Tasks 7–9 (LV 1e-12 gate, ADI bit-identity, dense caching, ≥10× benchmark, sparse path untouched); WS-D2 → Tasks 5–6; WS-B4 → Tasks 10–11 (u_var + bin_conditional with pinned tie convention). Cross-cutting §9 equality gates realized as capture-before pinned regressions.
- **Known deviation:** Task 11 uses split-index + per-slice `np.mean` instead of `np.add.reduceat` (spec WS-B4) to satisfy the spec's own exact-equality acceptance; flagged in the task and the commit message.
- **Type consistency:** `solve_tridiag_batch(sub, diag, sup, rhs)` full-length `(n_sys, N)` convention consumed by Tasks 8–9; `solve_tridiag` length-`N−1` off-diagonals consumed by Task 7 (matches the existing localvol call shape); `_simulate_slv` 3-tuple return consumed at both call sites in Task 3.
