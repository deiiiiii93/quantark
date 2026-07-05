# Volmodels Phase 4 — FP Positivity + Krylov + TR-BDF2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **OUTCOME AMENDMENT (2026-07-05, Codex-concurred).** WS-C4's core premise (Chang–Cooper/SG fluxes cut
> negative mass ≥10×) was **empirically falsified during implementation** and is NOT shipped. Investigation
> (see `example/fp_positivity_diagnostic.py`): the x/z directional operators are **already M-matrices**
> (zero negative off-diagonals) under central differencing, so exponential fitting has nothing to fix; the
> **entire** negativity budget is the mixed (correlation) cross-derivative term (exactly zero at ρ=0), which
> SG leaves untouched. Worse, SG is first-order (vs central's second-order) and **degrades the leverage
> calibration** (6× worse flat-vanilla repricing) for zero positivity gain. A genuine fix is provably
> **infeasible on this grid** (log-variance anisotropy b/a up to 4900×, `|c|>min(a,b)` at 0–8% of nodes →
> grid-aligned M-matrix positivity impossible; the true fix is a state-dependent sheared grid, deferred as a
> separate FP redesign). **What actually shipped for WS-C4:** the Chang–Cooper `flux_scheme` was **removed**;
> the bilinear Dirac seed split is kept as an opt-in `seed_split` flag (O(h)→O(h²), decoupled from any flux
> scheme); `tol_neg` is tightened 0.5→0.05 as an **empirical operating-domain tripwire** (measured negative
> mass < 0.03 on realistic high-stress fixtures, bounded under refinement), documented as an observed bound
> not a positivity claim. Tasks 2 and 7 below are therefore superseded; Tasks 1/3/5/6 shipped as written
> (with `flux_scheme` replaced by `seed_split`, and the Krylov parity gate met at a fresh preconditioner with
> the lagged mode's real accuracy characterized honestly). WS-B3 (Krylov) and WS-C6 (TR-BDF2) shipped in full.

**Goal:** Add positivity-preserving Chang–Cooper fluxes + bilinear Dirac seed splitting (WS-C4), an opt-in lagged-factorization Krylov FP march (WS-B3), and an opt-in second-order TR-BDF2 time march (WS-C6) to the SLV forward Fokker–Planck leverage calibration — then flip the WS-C4 default (central→chang_cooper, `tol_neg` 0.5→0.05) once the ≥10× negative-mass-reduction acceptance gate holds.

**Architecture:** Three orthogonal upgrades to the forward FP calibration in `quantark/volmodels/slv/fokkerplanck/`. WS-C4 modifies the *spatial* discretization (`fp_operators.py` face fluxes + `fp_solver.py` seed), gated by a new `flux_scheme` config. WS-B3 modifies the *linear-solve backend* of the implicit march (`fp_solver.py`), gated by `linear_solver`. WS-C6 modifies the *time integrator* (`fp_solver.py` march + `calibration.py` dispatch), gated by `time_scheme`. All three defaults preserve the exact current behavior byte-for-byte until the WS-C4 gate is met; WS-B3 and WS-C6 defaults stay opt-in (their flips are explicitly deferred per the design doc).

**Tech Stack:** NumPy, SciPy sparse (`splu`, `bicgstab`, `LinearOperator`), `quantark.util.numerical`, pytest.

## Global Constraints

Copied verbatim from `docs/superpowers/specs/2026-07-04-volmodels-dupire-heston-slv-improvements-design.md`:

- **Benchmark-first, opt-in-then-flip.** A pure accuracy upgrade at unchanged cost may flip its default with a deliberate golden update; anything that changes the cost/semantics trade-off ships opt-in. WS-C4 flips *within this phase* only if the ≥10× negative-mass benchmark holds; WS-B3 and WS-C6 keep their current defaults (`direct`, `backward_euler`) — flips are separate reviewed changes.
- **Equality gates before accuracy changes.** The `flux_scheme="central"` / `linear_solver="direct"` / `time_scheme="backward_euler"` code paths must remain **bit-identical** to the current implementation (exact equality on small grids; the existing `test_fp_operators.py` / `test_fp_solver.py` / `test_fp_acceptance.py` fixtures stay green with zero pin changes until Task 7 deliberately updates goldens).
- **No stupid fallbacks / no silent degradation.** A Krylov convergence failure triggers exactly one immediate refactor + retry; a second failure raises `NumericalError`. Never clamp, never silently accept a non-converged solve. (Per `[[feedback_no_stupid_fallbacks]]`.)
- **Numerical guards only.** Use `quantark.util.numerical` (`is_zero`, `safe_*`) and guarded `np.expm1`; never raw float tolerances or unprotected division. `tol_neg` and `mass_tol` come from config, never hardcoded.
- **No MC inside the deterministic FP engine.** (Per `[[feedback_no_mc_in_pde]]`.) The FP calibration is deterministic; no RNG anywhere in this plan.
- Commit messages end with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- `docs/` is gitignored on `main` — stage plan/spec with `git add -f`.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `quantark/volmodels/slv/fokkerplanck/config.py` | Calibration config dataclass | Add `flux_scheme`, `linear_solver`, `refactor_every`, `time_scheme` fields + validation (Task 1); flip `flux_scheme` default + tighten `tol_neg` (Task 7) |
| `quantark/volmodels/slv/fokkerplanck/fp_operators.py` | Directional face-flux operators | Add Chang–Cooper δ-reweighting to `_x_operator`/`_z_operator`, threaded via a `flux_scheme` kwarg (Task 2) |
| `quantark/volmodels/slv/fokkerplanck/fp_solver.py` | Forward march solver | Bilinear seed (Task 3); wire `flux_scheme` into `step` (Task 4); Krylov backend `_solve_implicit` + `_MarchState` (Task 5); TR-BDF2 `march_step` (Task 6) |
| `quantark/volmodels/slv/fokkerplanck/calibration.py` | Calibration march loop | Dispatch march on `time_scheme`/`linear_solver`; thread `_MarchState` across steps (Tasks 5, 6) |
| `example/fp_positivity_benchmark.py` | WS-C4 ≥10× negativity benchmark | Create (Task 4) |
| `example/fp_krylov_benchmark.py` | WS-B3 direct-vs-Krylov parity + timing | Create (Task 5) |
| `example/fp_trbdf2_convergence.py` | WS-C6 second-order time slope | Create (Task 6) |
| `test/test_fp_chang_cooper.py` | WS-C4 unit tests | Create (Tasks 2, 3) |
| `test/test_fp_krylov.py` | WS-B3 parity tests | Create (Task 5) |
| `test/test_fp_trbdf2.py` | WS-C6 convergence tests | Create (Task 6) |
| `test/test_fp_config.py` | Config validation | Extend (Tasks 1, 7) |
| `test/test_fp_acceptance.py` | End-to-end acceptance | Extend: ≥10× negativity, both-modes-green, tol_neg=0.05 flip (Tasks 4, 7) |

---

## Task 1: Config fields for the three new switches

**Files:**
- Modify: `quantark/volmodels/slv/fokkerplanck/config.py`
- Test: `test/test_fp_config.py`

**Interfaces:**
- Produces: `FpCalibrationConfig` gains `flux_scheme: str = "central"` (∈ `{"central","chang_cooper"}`), `linear_solver: str = "direct"` (∈ `{"direct","krylov_lagged"}`), `refactor_every: int = 5` (≥1), `time_scheme: str = "backward_euler"` (∈ `{"backward_euler","tr_bdf2"}`). All defaults preserve current behavior.

- [ ] **Step 1: Write the failing test**

Add to `test/test_fp_config.py`:

```python
import pytest
from quantark.util.exceptions import ValidationError
from quantark.volmodels.slv.fokkerplanck.config import FpCalibrationConfig


def test_new_switch_defaults_preserve_current_behavior():
    cfg = FpCalibrationConfig()
    assert cfg.flux_scheme == "central"
    assert cfg.linear_solver == "direct"
    assert cfg.time_scheme == "backward_euler"
    assert cfg.refactor_every == 5


@pytest.mark.parametrize("field,bad", [
    ("flux_scheme", "upwind"),
    ("linear_solver", "gmres"),
    ("time_scheme", "crank_nicolson"),
])
def test_invalid_scheme_strings_raise(field, bad):
    with pytest.raises(ValidationError):
        FpCalibrationConfig(**{field: bad})


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_refactor_every_must_be_positive_integer(bad):
    with pytest.raises(ValidationError):
        FpCalibrationConfig(refactor_every=bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_fp_config.py::test_new_switch_defaults_preserve_current_behavior -v`
Expected: FAIL — `AttributeError: 'FpCalibrationConfig' object has no attribute 'flux_scheme'`.

- [ ] **Step 3: Add the fields + validation**

In `config.py`, add fields to the dataclass (after `tol_neg`, before `eps_mass`):

```python
    flux_scheme: str = "central"          # {"central","chang_cooper"} spatial face-flux scheme (WS-C4)
    linear_solver: str = "direct"         # {"direct","krylov_lagged"} implicit-solve backend (WS-B3)
    refactor_every: int = 5               # krylov_lagged: refresh splu preconditioner every N steps
    time_scheme: str = "backward_euler"   # {"backward_euler","tr_bdf2"} FP time integrator (WS-C6)
```

Add to `__post_init__` (after the existing `leverage_clip` check):

```python
        if self.flux_scheme not in ("central", "chang_cooper"):
            raise ValidationError("flux_scheme must be 'central' or 'chang_cooper'")
        if self.linear_solver not in ("direct", "krylov_lagged"):
            raise ValidationError("linear_solver must be 'direct' or 'krylov_lagged'")
        if self.time_scheme not in ("backward_euler", "tr_bdf2"):
            raise ValidationError("time_scheme must be 'backward_euler' or 'tr_bdf2'")
        if not (isinstance(self.refactor_every, Integral) and self.refactor_every >= 1):
            raise ValidationError("refactor_every must be an integer >= 1")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test/test_fp_config.py -v`
Expected: PASS (all, including the pre-existing config tests).

- [ ] **Step 5: Commit**

```bash
git add quantark/volmodels/slv/fokkerplanck/config.py test/test_fp_config.py
git commit -m "feat(fp): config switches for flux_scheme/linear_solver/time_scheme (WS-C4/B3/C6)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: Chang–Cooper δ-reweighted directional fluxes

**Files:**
- Modify: `quantark/volmodels/slv/fokkerplanck/fp_operators.py`
- Test: `test/test_fp_chang_cooper.py` (create)

**Interfaces:**
- Consumes: `flux_scheme` string from Task 1's config (threaded as a kwarg, not the config object, to keep `fp_operators` config-agnostic).
- Produces: `build_directional_operators(x, z, L, params, eta, b, *, flux_scheme="central")` and `build_forward_operator(..., *, flux_scheme="central")` gain a keyword-only `flux_scheme`. New module-private `_cc_delta(P)`. Default `"central"` keeps every current caller (incl. `test_fp_operators.py`) bit-identical.

**Discretization (why this is bit-safe at P→0):** the current central face flux is `cL = 0.5·μ_f + D_i/h`, `cR = 0.5·μ_f − D_{i+1}/h`. Chang–Cooper replaces *only* the convective weight `0.5` with `δ = 1/P − 1/expm1(P)` (and `1−δ` on the other side), where the local Péclet `P = μ_f·h/D_f` uses the face-averaged diffusion `D_f = ½(D_i + D_{i+1})`. The node-local diffusion terms `±D/h` are unchanged, so at `δ=½` (P→0) the coefficients are byte-for-byte the current scheme, and the flux stays a telescoping face flux ⇒ mass conservation is preserved. `D_f > 0` always (`D = ½L²ν`, `L>0`, `ν=e^z>0`).

- [ ] **Step 1: Write the failing test**

Create `test/test_fp_chang_cooper.py`:

```python
import numpy as np
import pytest

from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.fokkerplanck.coordinates import concentrated_grid, trapezoid_weights
from quantark.volmodels.slv.fokkerplanck.fp_operators import (
    build_forward_operator, build_directional_operators, _cc_delta,
)

_P = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)


def _grid(params=_P):
    x = concentrated_grid(np.log(60.0), np.log(160.0), np.log(100.0), 61, 0.1)
    z = concentrated_grid(np.log(0.01), np.log(0.30), np.log(params.v0), 41, 0.1)
    return x, z


def _mass_residual(A, x, z):
    w = np.outer(trapezoid_weights(x), trapezoid_weights(z)).ravel()
    Ad = A.toarray()
    return np.max(np.abs(w @ Ad)) / max(np.max(np.abs(Ad)), 1e-300)


def test_cc_delta_central_limit():
    # delta -> 1/2 as P -> 0 (guarded series), monotone-ish, finite for large |P|
    P = np.array([-50.0, -1e-3, 0.0, 1e-3, 50.0])
    d = _cc_delta(P)
    assert np.all(np.isfinite(d))
    assert abs(d[2] - 0.5) < 1e-12          # exactly 1/2 at P=0
    assert abs(d[1] - 0.5) < 1e-3 and abs(d[3] - 0.5) < 1e-3


def _smooth_density(x, z):
    X, Z = np.meshgrid(x, z, indexing="ij")
    f = np.exp(-((X - np.log(100.0)) ** 2) / 0.1 - ((Z - np.log(_P.v0)) ** 2) / 0.5).ravel()
    return f / f.sum()


def test_cc_converges_to_central_under_refinement():
    # CC differs from central by an artificial-diffusion flux with relative size P^2/12
    # (delta - 1/2 = -P/12, P = mu_f*h/D_f). The matrix ENTRIES differ by O(1) (flux-coeff O(h)
    # over quadrature weight O(h)), but the ACTION on a smooth density is an O(h^2)-relative
    # perturbation that shrinks under grid refinement. This is the defensible central-recovery
    # invariant (an entry-wise A_cc==A_c match is impossible: mu = b - D is never ~0 here).
    def gap(nx, nz):
        x = concentrated_grid(np.log(60.0), np.log(160.0), np.log(100.0), nx, 0.1)
        z = concentrated_grid(np.log(0.01), np.log(0.30), np.log(_P.v0), nz, 0.1)
        L = np.ones(x.size)
        Ac = build_forward_operator(x, z, L, params=_P, eta=1.0, b=0.05, flux_scheme="central")
        Acc = build_forward_operator(x, z, L, params=_P, eta=1.0, b=0.05, flux_scheme="chang_cooper")
        f = _smooth_density(x, z)
        return np.linalg.norm((Acc - Ac) @ f) / max(np.linalg.norm(Ac @ f), 1e-300)

    g1, g2 = gap(61, 41), gap(121, 81)
    assert g2 < g1                           # gap shrinks under refinement (CC -> central as h -> 0)
    assert g2 < 0.6 * g1                     # decreasing faster than first order (artificial diffusion ~ P^2)


def test_cc_mass_conserved_high_correlation():
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.6, rho=-0.9)  # Feller-violated, |rho|=0.9
    x, z = _grid(p)
    L = np.linspace(0.8, 1.3, x.size)
    A = build_forward_operator(x, z, L, params=p, eta=1.0, b=0.05, flux_scheme="chang_cooper")
    assert _mass_residual(A, x, z) < 1e-9    # telescoping flux => still mass-conserving


def test_cc_directional_split_sums_to_full():
    x, z = _grid()
    L = np.linspace(0.9, 1.1, x.size)
    A = build_forward_operator(x, z, L, params=_P, eta=1.0, b=0.02, flux_scheme="chang_cooper")
    Ax, Az, Axz = build_directional_operators(x, z, L, params=_P, eta=1.0, b=0.02,
                                              flux_scheme="chang_cooper")
    assert np.max(np.abs((A - (Ax + Az + Axz)).toarray())) < 1e-12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_fp_chang_cooper.py -v`
Expected: FAIL — `ImportError: cannot import name '_cc_delta'` / `build_forward_operator() got an unexpected keyword argument 'flux_scheme'`.

- [ ] **Step 3: Implement `_cc_delta` and thread `flux_scheme`**

In `fp_operators.py`, add after `_coo`:

```python
def _cc_delta(P):
    """Chang-Cooper face weight delta = 1/P - 1/expm1(P); guarded to the exact central limit 1/2 at P->0.

    Series near zero: delta = 1/2 - P/12 + O(P^2). Replaces the 0.5 convective average weight so the
    face flux is exponentially fitted to the local convection-diffusion balance (positivity-preserving).
    """
    P = np.asarray(P, float)
    out = np.empty(P.shape, float)
    small = np.abs(P) < 1e-6
    Pb = P[~small]
    out[~small] = 1.0 / Pb - 1.0 / np.expm1(Pb)
    out[small] = 0.5 - P[small] / 12.0
    return out
```

Change `_x_operator` signature to `def _x_operator(x, z, L, params, eta, b, flux_scheme):` and replace the `cL`/`cR` block:

```python
    mu_f = 0.5 * (mu[:-1, :] + mu[1:, :])                       # (nx-1, nz)
    if flux_scheme == "chang_cooper":
        D_f = 0.5 * (D[:-1, :] + D[1:, :])                      # face diffusion (nx-1, nz), > 0
        delta = _cc_delta(mu_f * h / D_f)                       # local Peclet P = mu_f*h/D_f
    else:
        delta = 0.5
    cL = delta * mu_f + D[:-1, :] / h                           # coeff on f_i  at face i+1/2
    cR = (1.0 - delta) * mu_f - D[1:, :] / h                    # coeff on f_{i+1}
```

Change `_z_operator` signature to `def _z_operator(x, z, L, params, eta, b, flux_scheme):` and replace its `cL`/`cR` block (note the z-direction shapes are `(nz-1,)`):

```python
    mu_f = 0.5 * (mu_z[:-1] + mu_z[1:])                         # (nz-1,)
    if flux_scheme == "chang_cooper":
        D_f = 0.5 * (Dz[:-1] + Dz[1:])                          # (nz-1,), > 0
        delta = _cc_delta(mu_f * h / D_f)                       # (nz-1,)
    else:
        delta = 0.5
    cL = delta * mu_f + Dz[:-1] / h                             # (nz-1,) coeff on f_j
    cR = (1.0 - delta) * mu_f - Dz[1:] / h
```

Leave `_mixed_operator` unchanged (spec: mixed operator untouched). Add `flux_scheme` to it as an ignored kwarg for a uniform call signature: `def _mixed_operator(x, z, L, params, eta, b, flux_scheme):` (body unchanged).

Update `build_directional_operators` and `build_forward_operator`:

```python
def build_directional_operators(x, z, L, params: HestonParams, eta: float, b: float,
                                *, flux_scheme: str = "central"):
    """Return (Ax, Az, Axz) sparse CSR pieces of the forward generator for the ADI split."""
    x, z, L = _check(x, z, L)
    return (_x_operator(x, z, L, params, eta, b, flux_scheme),
            _z_operator(x, z, L, params, eta, b, flux_scheme),
            _mixed_operator(x, z, L, params, eta, b, flux_scheme))


def build_forward_operator(x, z, L, params: HestonParams, eta: float, b: float,
                           *, flux_scheme: str = "central"):
    """Sparse CSR forward Fokker-Planck generator A on the (x,z) grid (SLV log-variance density).

    Layout: row-major index k = i*nz + j for (x_i, z_j). Mass-conserving by construction
    (quadrature-weighted constant is a left-null vector). Densify via ``.toarray()`` only for
    small-grid tests; the production grid is far too large to densify.
    """
    Ax, Az, Axz = build_directional_operators(x, z, L, params, eta, b, flux_scheme=flux_scheme)
    return (Ax + Az + Axz).tocsr()
```

- [ ] **Step 4: Run the new + existing operator tests**

Run: `python -m pytest test/test_fp_chang_cooper.py test/test_fp_operators.py -v`
Expected: PASS. `test_fp_operators.py` (no `flux_scheme` arg → default `"central"`) stays green bit-identically.

- [ ] **Step 5: Commit**

```bash
git add quantark/volmodels/slv/fokkerplanck/fp_operators.py test/test_fp_chang_cooper.py
git commit -m "feat(fp): Chang-Cooper exponentially-fitted face fluxes (WS-C4)

delta-reweight the convective average only; node-local diffusion unchanged so
flux_scheme='central' is bit-identical and CC stays a telescoping (mass-conserving) flux.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: Bilinear Dirac seed splitting

**Files:**
- Modify: `quantark/volmodels/slv/fokkerplanck/fp_solver.py` (`seed_dirac`)
- Test: `test/test_fp_chang_cooper.py` (extend)

**Interfaces:**
- Consumes: `self.cfg.flux_scheme` (the solver already stores `self.cfg`). Seed splitting is **bundled** under `flux_scheme` — `"central"` keeps nearest-node placement (bit-identical), `"chang_cooper"` uses the bilinear split. Rationale (flag for reviewer): one flag gives a clean bisection of old-vs-new and satisfies the acceptance requirement that both modes be exercisable during the transition; the split is a pure accuracy improvement that pairs naturally with the positivity fix.
- Produces: `seed_dirac(s0, v0)` returns a unit-mass vector; in CC mode, mass spread bilinearly over the ≤4 nodes bracketing `(ln s0, ln v0)`.

- [ ] **Step 1: Write the failing test**

Append to `test/test_fp_chang_cooper.py`:

```python
from quantark.volmodels.slv.fokkerplanck.config import FpCalibrationConfig
from quantark.volmodels.slv.fokkerplanck.fp_solver import ForwardFPADI


def _solver(flux_scheme):
    cfg = FpCalibrationConfig(n_x=81, n_z=61, flux_scheme=flux_scheme)
    return ForwardFPADI.from_config(100.0, _P, eta=1.0, b=0.0,
                                    step_dt=np.full(10, 0.1), config=cfg)


def test_seed_unit_mass_both_schemes():
    for scheme in ("central", "chang_cooper"):
        s = _solver(scheme)
        f = s.seed_dirac(100.0, _P.v0)
        assert abs(s.total_mass(f) - 1.0) < 1e-12


def test_central_seed_is_still_nearest_node():
    s = _solver("central")
    f = s.seed_dirac(100.0, _P.v0)
    assert int(np.count_nonzero(f)) == 1     # exactly one node carries mass


def test_bilinear_seed_preserves_mean_to_second_order():
    # place the seed strictly between nodes; bilinear mean matches (ln s0, ln v0) to O(h^2)
    s = _solver("chang_cooper")
    # pick a spot/vol landing between grid nodes
    xs = 0.5 * (s.x[40] + s.x[41]); zs = 0.5 * (s.z[30] + s.z[31])
    s0, v0 = np.exp(xs), np.exp(zs)
    f = s.seed_dirac(s0, v0)
    assert abs(s.total_mass(f) - 1.0) < 1e-12
    F = f.reshape(s.nx, s.nz)
    mean_x = float((s.w.reshape(s.nx, s.nz) * F).sum(axis=1) @ s.x)  # sum_k w_k f_k x_i
    mean_z = float((s.w.reshape(s.nx, s.nz) * F).sum(axis=0) @ s.z)
    hx = s.x[41] - s.x[40]; hz = s.z[31] - s.z[30]
    assert abs(mean_x - xs) < hx ** 2 + 1e-12
    assert abs(mean_z - zs) < hz ** 2 + 1e-12
    assert int(np.count_nonzero(f)) <= 4     # at most the 4 bracketing nodes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_fp_chang_cooper.py::test_bilinear_seed_preserves_mean_to_second_order -v`
Expected: FAIL — the current nearest-node seed puts all mass on one node, so `mean_x` is off by ~½hx (≫ hx²), and `count_nonzero==1`.

- [ ] **Step 3: Implement bilinear splitting in `seed_dirac`**

Replace `ForwardFPADI.seed_dirac`:

```python
    def seed_dirac(self, s0, v0):
        """Discrete unit-mass seed at (ln s0, ln v0).

        ``flux_scheme='central'``: all mass on the nearest node, f(node) = 1/w_node (legacy).
        ``flux_scheme='chang_cooper'``: bilinear split over the <=4 bracketing nodes with weights
        divided by their quadrature weights -- total mass exactly 1, seed mean location exact to O(h^2).
        """
        f = np.zeros(self.nx * self.nz)
        xs, zs = float(np.log(s0)), float(np.log(v0))
        if self.cfg.flux_scheme != "chang_cooper":
            i = int(np.argmin(np.abs(self.x - xs)))
            j = int(np.argmin(np.abs(self.z - zs)))
            k = i * self.nz + j
            f[k] = 1.0 / self.w[k]
            return f
        i = int(np.clip(np.searchsorted(self.x, xs) - 1, 0, self.nx - 2))
        j = int(np.clip(np.searchsorted(self.z, zs) - 1, 0, self.nz - 2))
        tx = np.clip((xs - self.x[i]) / (self.x[i + 1] - self.x[i]), 0.0, 1.0)
        tz = np.clip((zs - self.z[j]) / (self.z[j + 1] - self.z[j]), 0.0, 1.0)
        for di, wx in ((0, 1.0 - tx), (1, tx)):
            for dj, wz in ((0, 1.0 - tz), (1, tz)):
                k = (i + di) * self.nz + (j + dj)
                f[k] += (wx * wz) / self.w[k]
        return f
```

- [ ] **Step 4: Run the seed tests + existing solver test**

Run: `python -m pytest test/test_fp_chang_cooper.py test/test_fp_solver.py -v`
Expected: PASS. `test_fp_solver.py::test_seed_has_unit_mass` (default central) stays green bit-identically.

- [ ] **Step 5: Commit**

```bash
git add quantark/volmodels/slv/fokkerplanck/fp_solver.py test/test_fp_chang_cooper.py
git commit -m "feat(fp): bilinear Dirac seed splitting under chang_cooper (WS-C4)

Distributes unit mass over the <=4 bracketing nodes (weights / quadrature weights):
exact unit mass, seed mean exact to O(h^2). central mode keeps nearest-node placement.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 4: Wire `flux_scheme` through the march + ≥10× negativity benchmark

**Files:**
- Modify: `quantark/volmodels/slv/fokkerplanck/fp_solver.py` (`step` passes `self.cfg.flux_scheme`)
- Create: `example/fp_positivity_benchmark.py`
- Test: `test/test_fp_acceptance.py` (extend)

**Interfaces:**
- Consumes: `self.cfg.flux_scheme` inside `step`.
- Produces: `calibrate_leverage_surface_fp(...)` now honors `flux_scheme` end-to-end. Benchmark script exits non-zero if CC does not cut max negative mass ≥10× vs central on the adversarial fixture.

- [ ] **Step 1: Write the failing test**

Append to `test/test_fp_acceptance.py`:

```python
def _adversarial_fixture(n=60):
    # high sigma + |rho|=0.9 => strong mixed-term negativity; the regime where CC must pay off
    p = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.7, rho=-0.9)
    surf = _surface(skew=True)
    return p, surf, n


def _max_neg(flux_scheme, n=60):
    p, surf, n = _adversarial_fixture(n)
    cfg = FpCalibrationConfig(n_x=161, n_z=121, flux_scheme=flux_scheme, tol_neg=0.5)
    lev = _calibrate(surf, p, n=n, fp_config=cfg)
    return lev.diagnostics["max_negative_mass"], lev


def test_chang_cooper_cuts_negative_mass_at_least_10x():
    neg_central, _ = _max_neg("central")
    neg_cc, _ = _max_neg("chang_cooper")
    assert neg_central > 1e-4                    # fixture is adversarial enough to matter
    assert neg_cc <= neg_central / 10.0          # >=10x reduction (the WS-C4 acceptance gate)


def test_chang_cooper_still_reprices_flat_vanilla():
    # positivity change must not break the exact-vanilla property (moves within discretization tol)
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    cfg = FpCalibrationConfig(n_x=201, n_z=101, flux_scheme="chang_cooper")
    lev = _calibrate(_surface(skew=False), p, fp_config=cfg)
    price = price_european_slv_pde(100.0, 100.0, True, 1.0, p, lev, 0.02, 0.0,
                                   eta=1.0, n_x=180, n_v=70, n_t=70)
    bs = bs_call_price(100.0, 100.0, 1.0, 0.20, 0.02, 0.0)
    assert abs(price - bs) < 0.25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_fp_acceptance.py::test_chang_cooper_cuts_negative_mass_at_least_10x -v`
Expected: FAIL — `step` still builds central operators regardless of config, so both runs give the same negative mass (ratio ≈ 1, not ≥10).

- [ ] **Step 3: Thread `flux_scheme` into `step`**

In `fp_solver.py::step`, change the operator build line:

```python
        Ax, Az, Axz = build_directional_operators(self.x, self.z, L, self.params, self.eta, b_eff,
                                                   flux_scheme=self.cfg.flux_scheme)
```

- [ ] **Step 4: Run the acceptance tests**

Run: `python -m pytest test/test_fp_acceptance.py -v`
Expected: PASS. **If the ≥10× gate fails**, this is the "hold the gate" decision from kickoff — increase `n_z` (the mixed/convective negativity is z-driven), verify CC delivers ≥10× at the finer resolution, and pin that resolution in the test. Do NOT weaken the assertion below 10×. Record the tuned resolution in the commit message.

- [ ] **Step 5: Create the benchmark script**

Create `example/fp_positivity_benchmark.py`:

```python
"""WS-C4 acceptance benchmark: Chang-Cooper vs central max negative probability mass.

Exits non-zero unless Chang-Cooper cuts the max negative mass >= 10x on the adversarial
(high-sigma, |rho|=0.9) fixture. Run: python example/fp_positivity_benchmark.py
"""
import sys
import numpy as np

from quantark.param import FlatRateCurve, GridVolSurface
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.slv import calibrate_leverage_surface, FpCalibrationConfig

_ZERO_DIV = (lambda t: 0.0)


def _skew_surface():
    strikes = list(100.0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    mats = list(np.linspace(0.1, 2.0, 6))
    ks = np.linspace(-0.6, 0.6, 9)
    grid = np.array([[0.22 - 0.04 * ks[j] for j in range(9)] for _ in mats])
    return GridVolSurface(strikes, mats, grid)


def _run(flux_scheme, n=60):
    p = HestonParams(v0=0.04, kappa=1.5, theta=0.04, sigma=0.7, rho=-0.9)
    lv = build_dupire_local_vol(_skew_surface(), spot=100.0,
                                rate_curve=FlatRateCurve(0.02), div_yield=_ZERO_DIV)
    t_grid = np.linspace(0.0, 1.0, n + 1)
    cfg = FpCalibrationConfig(n_x=161, n_z=121, flux_scheme=flux_scheme, tol_neg=0.5)
    lev = calibrate_leverage_surface(100.0, p, lv, np.diff(t_grid), np.full(n, 0.02),
                                     np.full(n, 0.0), eta=1.0, fp_config=cfg)
    return lev.diagnostics["max_negative_mass"]


def main():
    neg_c = _run("central")
    neg_cc = _run("chang_cooper")
    ratio = neg_c / max(neg_cc, 1e-300)
    print(f"central  max negative mass = {neg_c:.3e}")
    print(f"chang_cooper max negative mass = {neg_cc:.3e}")
    print(f"reduction factor = {ratio:.1f}x  (target >= 10x)")
    if ratio < 10.0:
        print("FAIL: Chang-Cooper did not achieve the >=10x negativity reduction gate")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the benchmark**

Run: `python example/fp_positivity_benchmark.py`
Expected: prints the reduction factor and `PASS` (exit 0). If it fails, tune `n_z` per Step 4.

- [ ] **Step 7: Commit**

```bash
git add quantark/volmodels/slv/fokkerplanck/fp_solver.py example/fp_positivity_benchmark.py test/test_fp_acceptance.py
git commit -m "feat(fp): wire flux_scheme through the FP march + >=10x negativity benchmark (WS-C4)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 5: WS-B3 — lagged-factorization Krylov FP march

**Files:**
- Modify: `quantark/volmodels/slv/fokkerplanck/fp_solver.py` (add `_MarchState`, `_solve_implicit`; route `step`'s implicit solve through it)
- Modify: `quantark/volmodels/slv/fokkerplanck/calibration.py` (thread a `_MarchState` across the march loop)
- Create: `example/fp_krylov_benchmark.py`
- Test: `test/test_fp_krylov.py` (create)

**Interfaces:**
- Produces:
  - `_MarchState` — holds the lagged `splu` preconditioner, the matrix it factorized, and a step counter.
  - `ForwardFPADI._solve_implicit(self, M, rhs, state)` → solves `M x = rhs`. `direct`: `splu(M).solve(rhs)`. `krylov_lagged`: BiCGStab preconditioned by a `splu` of a lagged `M`, refreshed every `refactor_every` steps or on non-convergence (immediate refactor + one retry; second failure → `NumericalError`).
  - `ForwardFPADI.step(..., state=None)` gains an optional `state` (defaults to a throwaway direct-mode state so existing callers are unaffected).
- Consumes: `self.cfg.linear_solver`, `self.cfg.refactor_every`.

**Design note (flag for reviewer):** the preconditioner is the `splu` of a *stale* `(I − dt·A_k)`; because `A` drifts slowly with `L(t)`, BiCGStab converges in few iterations against the lagged preconditioner. Refresh cadence is `refactor_every`. On `bicgstab` `info != 0`: refactor `M` now, retry once; a second `info != 0` raises `NumericalError` (no silent degradation, per Global Constraints).

- [ ] **Step 1: Write the failing test**

Create `test/test_fp_krylov.py`:

```python
import numpy as np
import pytest

from quantark.param import FlatRateCurve, GridVolSurface
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.slv import calibrate_leverage_surface, FpCalibrationConfig

_ZERO_DIV = (lambda t: 0.0)


def _surface():
    strikes = list(100.0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    mats = list(np.linspace(0.1, 2.0, 6))
    grid = np.full((6, 9), 0.20)
    return GridVolSurface(strikes, mats, grid)


def _calibrate(linear_solver, rho=-0.5, n=30):
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=rho)
    lv = build_dupire_local_vol(_surface(), spot=100.0,
                                rate_curve=FlatRateCurve(0.02), div_yield=_ZERO_DIV)
    t_grid = np.linspace(0.0, 1.0, n + 1)
    cfg = FpCalibrationConfig(n_x=161, n_z=101, linear_solver=linear_solver, refactor_every=5)
    return calibrate_leverage_surface(100.0, p, lv, np.diff(t_grid), np.full(n, 0.02),
                                      np.full(n, 0.0), eta=1.0, fp_config=cfg)


@pytest.mark.parametrize("rho", [-0.5, -0.9])
def test_krylov_matches_direct(rho):
    a = _calibrate("direct", rho=rho)
    b = _calibrate("krylov_lagged", rho=rho)
    assert np.max(np.abs(a.leverage_grid - b.leverage_grid)) < 1e-10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_fp_krylov.py -v`
Expected: FAIL — `linear_solver="krylov_lagged"` is accepted by config but `step` ignores it (always `splu`), so either the surfaces match trivially (test still meaningful) or (if not yet routed) a code path error. The test asserts the parity that only exists once Krylov is implemented; before implementation the two calls are identical (direct) so it *passes vacuously* — therefore FIRST assert routing exists:

Add this guard test that genuinely fails pre-implementation:

```python
def test_krylov_mode_actually_uses_bicgstab(monkeypatch):
    import quantark.volmodels.slv.fokkerplanck.fp_solver as mod
    calls = {"n": 0}
    real = mod.bicgstab

    def _spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(mod, "bicgstab", _spy)
    _calibrate("krylov_lagged", n=10)
    assert calls["n"] > 0                         # the march went through BiCGStab


def _count_splu(monkeypatch, **cfg_over):
    import quantark.volmodels.slv.fokkerplanck.fp_solver as mod
    calls = {"n": 0}
    real = mod.splu

    def _spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(mod, "splu", _spy)
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    lv = build_dupire_local_vol(_surface(), spot=100.0,
                                rate_curve=FlatRateCurve(0.02), div_yield=_ZERO_DIV)
    n = 12
    t_grid = np.linspace(0.0, 1.0, n + 1)
    cfg = FpCalibrationConfig(n_x=121, n_z=81, linear_solver="krylov_lagged", **cfg_over)
    calibrate_leverage_surface(100.0, p, lv, np.diff(t_grid), np.full(n, 0.02),
                               np.full(n, 0.0), eta=1.0, fp_config=cfg)
    return calls["n"], n


def test_krylov_refactor_cadence_advances_once_per_step(monkeypatch):
    # refactor_every=1 => one splu refresh per MARCH step (n steps), even though backward-Euler
    # does one solve/step. Assumes no convergence-failure refactors on this benign fixture.
    c, n = _count_splu(monkeypatch, refactor_every=1)
    assert c == n


def test_krylov_trbdf2_reuses_matrix_within_step(monkeypatch):
    # TR-BDF2 does two solves per step against the SAME M; with refactor_every=1 the cadence must
    # still refresh only once per step (step 0 is backward-Euler start-up, steps 1..n-1 are TR-BDF2).
    c, n = _count_splu(monkeypatch, refactor_every=1, time_scheme="tr_bdf2")
    assert c == n                                 # NOT 2n-1: the second substep reuses (advance=False)
```

Expected pre-implementation: FAIL — `module 'fp_solver' has no attribute 'bicgstab'`.

- [ ] **Step 3: Implement `_MarchState` + `_solve_implicit` and route `step`**

In `fp_solver.py`, update imports:

```python
from scipy.sparse.linalg import splu, bicgstab, LinearOperator
```

Add the state class above `ForwardFPADI`:

```python
class _MarchState:
    """Carries the lagged splu preconditioner for the krylov_lagged linear solver across march steps."""
    __slots__ = ("precond", "steps_since_refactor")

    def __init__(self):
        self.precond = None
        self.steps_since_refactor = 0
```

Add a method to `ForwardFPADI`. The `advance` flag makes the refactor cadence tick **once per march step**, not once per linear solve — so TR-BDF2's two solves against the identical `M` (Task 6) don't double-count the cadence or spuriously refactor the same matrix:

```python
    def _solve_implicit(self, M, rhs, state, *, advance=True):
        """Solve M x = rhs. ``direct``: splu. ``krylov_lagged``: BiCGStab + lagged splu preconditioner.

        ``advance``: when True (default) this solve participates in the refactor cadence (may refresh the
        preconditioner at the ``refactor_every`` boundary and increments the step counter). Pass
        ``advance=False`` for TR-BDF2's second substep, which reuses the first substep's identical M.
        """
        if self.cfg.linear_solver == "direct":
            return self._splu(M).solve(rhs)
        # krylov_lagged
        if state.precond is None or (advance and state.steps_since_refactor >= self.cfg.refactor_every):
            state.precond = self._splu(M)
            state.steps_since_refactor = 0
        Mpre = LinearOperator(M.shape, matvec=state.precond.solve)
        x, info = bicgstab(M, rhs, M=Mpre, rtol=1e-12, atol=0.0, maxiter=200)
        if info != 0:                                       # one refactor + retry, then raise
            state.precond = self._splu(M)
            state.steps_since_refactor = 0
            Mpre = LinearOperator(M.shape, matvec=state.precond.solve)
            x, info = bicgstab(M, rhs, M=Mpre, rtol=1e-12, atol=0.0, maxiter=200)
            if info != 0:
                raise NumericalError(f"Krylov FP solve failed to converge (info={info}); refine grid/steps")
        if advance:
            state.steps_since_refactor += 1
        return x
```

Update `step` to accept and use `state`:

```python
    def step(self, f, L, dt, implicit=True, theta=0.5, b=None, state=None):
```

and replace the implicit-branch solve:

```python
        if implicit:
            if state is None:
                state = _MarchState()
            M = (self._I - dt * (Ax + Az + Axz)).tocsc()
            out = self._solve_implicit(M, f, state)
```

(The ADI `implicit=False` branch is unchanged — Krylov applies only to the coupled backward-Euler/TR-BDF2 solves.)

- [ ] **Step 4: Thread the state through calibration**

In `calibration.py::calibrate_leverage_surface_fp`, before the march loop add:

```python
    from quantark.volmodels.slv.fokkerplanck.fp_solver import _MarchState
    march_state = _MarchState()
```

and change the `solver.step(...)` call:

```python
        f = solver.step(f, L, dt[n], implicit=True, b=float(rf[n] - cf[n]), state=march_state)
```

- [ ] **Step 5: Run the Krylov tests**

Run: `python -m pytest test/test_fp_krylov.py test/test_fp_solver.py test/test_fp_acceptance.py -v`
Expected: PASS. Direct mode unaffected (default); Krylov matches direct < 1e-10.

- [ ] **Step 6: Create the parity+timing benchmark**

Create `example/fp_krylov_benchmark.py`:

```python
"""WS-B3 benchmark: krylov_lagged vs direct FP march -- leverage parity + wall-clock.

Run: python example/fp_krylov_benchmark.py
"""
import time
import numpy as np

from quantark.param import FlatRateCurve, GridVolSurface
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.slv import calibrate_leverage_surface, FpCalibrationConfig

_ZERO_DIV = (lambda t: 0.0)


def _run(linear_solver, n=60):
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.9)
    strikes = list(100.0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    mats = list(np.linspace(0.1, 2.0, 6))
    surf = GridVolSurface(strikes, mats, np.full((6, 9), 0.20))
    lv = build_dupire_local_vol(surf, spot=100.0, rate_curve=FlatRateCurve(0.02), div_yield=_ZERO_DIV)
    t_grid = np.linspace(0.0, 1.0, n + 1)
    cfg = FpCalibrationConfig(n_x=201, n_z=141, linear_solver=linear_solver, refactor_every=5)
    t0 = time.perf_counter()
    lev = calibrate_leverage_surface(100.0, p, lv, np.diff(t_grid), np.full(n, 0.02),
                                     np.full(n, 0.0), eta=1.0, fp_config=cfg)
    return lev, time.perf_counter() - t0


def main():
    a, ta = _run("direct")
    b, tb = _run("krylov_lagged")
    err = np.max(np.abs(a.leverage_grid - b.leverage_grid))
    print(f"direct        : {ta:.2f}s")
    print(f"krylov_lagged : {tb:.2f}s  (speedup {ta / tb:.2f}x)")
    print(f"max leverage abs diff = {err:.2e}  (target < 1e-10)")
    assert err < 1e-10, "Krylov leverage surface diverged from direct beyond 1e-10"


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the benchmark, then commit**

Run: `python example/fp_krylov_benchmark.py`
Expected: parity < 1e-10; timing recorded (speedup may be <1× on these grid sizes — Krylov is an opt-in escape hatch, not a guaranteed win; the default stays `direct`).

```bash
git add quantark/volmodels/slv/fokkerplanck/fp_solver.py quantark/volmodels/slv/fokkerplanck/calibration.py example/fp_krylov_benchmark.py test/test_fp_krylov.py
git commit -m "feat(fp): opt-in lagged-factorization Krylov FP march (WS-B3)

BiCGStab + stale-splu preconditioner refreshed every refactor_every steps; one
refactor+retry on non-convergence then NumericalError. direct stays default; parity <1e-10.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 6: WS-C6 — TR-BDF2 FP time march

**Files:**
- Modify: `quantark/volmodels/slv/fokkerplanck/fp_solver.py` (add `march_step`)
- Modify: `quantark/volmodels/slv/fokkerplanck/calibration.py` (dispatch on `time_scheme`; Rannacher start-up)
- Create: `example/fp_trbdf2_convergence.py`
- Test: `test/test_fp_trbdf2.py` (create)

**Interfaces:**
- Produces: `ForwardFPADI.march_step(self, f, L, dt, b, state, is_first)` — advances one step under `self.cfg.time_scheme`. `backward_euler`: identical to `step(implicit=True, state=state)`. `tr_bdf2`: γ=2−√2 trapezoidal substep + BDF2 substep, both against the same frozen operator `A`; `is_first=True` forces a backward-Euler step (Rannacher start-up to damp the seed).
- Consumes: `self.cfg.time_scheme`, plus the WS-B3 `state`/`linear_solver` (TR-BDF2 solves route through `_solve_implicit`, so Krylov composes).

**Coefficients (γ = 2−√2):** the two implicit substep operators coincide because `½γ = (1−γ)/(2−γ)` (the defining property of γ=2−√2), so **one** factorization/preconditioner serves both solves.
- TR: `(I − ½γ·dt·A) y_γ = (I + ½γ·dt·A) f`
- BDF2: `(I − ½γ·dt·A) y_1 = c1·y_γ − c0·f`, with `c1 = 1/(γ(2−γ))`, `c0 = (1−γ)²/(γ(2−γ))`.

- [ ] **Step 1: Write the failing test**

Create `test/test_fp_trbdf2.py`:

```python
import numpy as np
import pytest

from quantark.param import FlatRateCurve, GridVolSurface
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.slv import calibrate_leverage_surface, FpCalibrationConfig

_ZERO_DIV = (lambda t: 0.0)


def _surface():
    strikes = list(100.0 * np.exp(np.linspace(-0.6, 0.6, 9)))
    mats = list(np.linspace(0.1, 2.0, 6))
    return GridVolSurface(strikes, mats, np.full((6, 9), 0.20))


def _calibrate(time_scheme, n):
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    lv = build_dupire_local_vol(_surface(), spot=100.0,
                                rate_curve=FlatRateCurve(0.02), div_yield=_ZERO_DIV)
    t_grid = np.linspace(0.0, 1.0, n + 1)
    cfg = FpCalibrationConfig(n_x=161, n_z=101, time_scheme=time_scheme)
    return calibrate_leverage_surface(100.0, p, lv, np.diff(t_grid), np.full(n, 0.02),
                                      np.full(n, 0.0), eta=1.0, fp_config=cfg)


def _err_vs_ref(time_scheme, n, ref):
    lev = _calibrate(time_scheme, n)
    # compare on the shared coarse record grid (ref is finest); use the ATM leverage path at t=0.5
    return abs(lev.leverage(100.0, 0.5) - ref.leverage(100.0, 0.5))


def test_trbdf2_second_order_in_time():
    ref = _calibrate("tr_bdf2", 160)                  # fine-dt reference
    e = [_err_vs_ref("tr_bdf2", n, ref) for n in (20, 40, 80)]
    # successive halving of dt -> error ratio ~4 (second order). Require > 3.0 (allow spatial floor slack).
    r1 = e[0] / max(e[1], 1e-300)
    r2 = e[1] / max(e[2], 1e-300)
    assert r1 > 3.0 and r2 > 3.0


def test_trbdf2_mass_and_negativity_no_worse_than_be():
    be = _calibrate("backward_euler", 60)
    tr = _calibrate("tr_bdf2", 60)
    assert max(tr.diagnostics["mass_residual"]) < 1e-3
    assert tr.diagnostics["max_negative_mass"] <= 10.0 * be.diagnostics["max_negative_mass"] + 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_fp_trbdf2.py -v`
Expected: FAIL — `time_scheme="tr_bdf2"` is accepted by config but the march is still backward-Euler (`calibration.py` calls `step`), so `tr_bdf2` behaves as `backward_euler` (first-order) and `r1`/`r2` ≈ 2, not > 3.

- [ ] **Step 3: Implement `march_step`**

In `fp_solver.py`, add:

```python
    _TRBDF2_GAMMA = 2.0 - np.sqrt(2.0)

    def march_step(self, f, L, dt, b, state, is_first):
        """Advance one step under cfg.time_scheme. TR-BDF2 uses a backward-Euler first step (Rannacher)."""
        if self.cfg.time_scheme == "backward_euler" or is_first:
            return self.step(f, L, dt, implicit=True, b=b, state=state)
        g = self._TRBDF2_GAMMA
        L = np.asarray(L, float)
        Ax, Az, Axz = build_directional_operators(self.x, self.z, L, self.params, self.eta,
                                                  float(b), flux_scheme=self.cfg.flux_scheme)
        A = (Ax + Az + Axz).tocsc()
        M = (self._I - 0.5 * g * dt * A).tocsc()          # SAME matrix serves both substeps
        # TR substep to t + gamma*dt
        rhs_tr = f + 0.5 * g * dt * (A @ f)
        y_g = self._solve_implicit(M, rhs_tr, state)
        # BDF2 substep to t + dt
        c1 = 1.0 / (g * (2.0 - g))
        c0 = (1.0 - g) ** 2 / (g * (2.0 - g))
        rhs_bdf = c1 * y_g - c0 * f
        out = self._solve_implicit(M, rhs_bdf, state, advance=False)   # reuse first substep's M/precond
        if not np.all(np.isfinite(out)):
            raise NumericalError("TR-BDF2 FP step produced non-finite density")
        return out
```

**Cadence:** both `_solve_implicit(M, ...)` calls share the identical `M` (γ=2−√2). The first substep uses the default `advance=True` (participates in the refactor cadence, increments the counter); the second uses `advance=False` (reuses the first substep's preconditioner, no cadence tick). So the lagged-Krylov preconditioner is refreshed **once per march step**, never twice for the same matrix — pinned by `test_krylov_trbdf2_reuses_matrix_within_step`.

- [ ] **Step 4: Dispatch in calibration + Rannacher start-up**

In `calibration.py::calibrate_leverage_surface_fp`, replace the `solver.step(...)` march call with:

```python
        f = solver.march_step(f, L, dt[n], b=float(rf[n] - cf[n]), state=march_state, is_first=(n == 0))
```

(The `n == 0` step is the exact-seed row where `L` is the analytic `sigma_lv/sqrt(v0)`; forcing backward-Euler there matches the existing Rannacher-style damping and the current behavior byte-for-byte in `backward_euler` mode.)

- [ ] **Step 5: Run the TR-BDF2 tests + full FP suite**

Run: `python -m pytest test/test_fp_trbdf2.py test/test_fp_acceptance.py test/test_fp_solver.py test/test_fp_krylov.py -v`
Expected: PASS. `backward_euler` (default) unaffected and bit-identical; `tr_bdf2` shows > 3× error ratio under dt-halving.

- [ ] **Step 6: Create the convergence benchmark**

Create `example/fp_trbdf2_convergence.py`:

```python
"""WS-C6 benchmark: TR-BDF2 vs backward-Euler leverage-surface convergence in dt.

Prints the error-vs-fine-reference ladder and the observed order. Run:
python example/fp_trbdf2_convergence.py
"""
import numpy as np

from quantark.param import FlatRateCurve, GridVolSurface
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.slv import calibrate_leverage_surface, FpCalibrationConfig

_ZERO_DIV = (lambda t: 0.0)


def _calibrate(time_scheme, n):
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    surf = GridVolSurface(list(100.0 * np.exp(np.linspace(-0.6, 0.6, 9))),
                          list(np.linspace(0.1, 2.0, 6)), np.full((6, 9), 0.20))
    lv = build_dupire_local_vol(surf, spot=100.0, rate_curve=FlatRateCurve(0.02), div_yield=_ZERO_DIV)
    t_grid = np.linspace(0.0, 1.0, n + 1)
    cfg = FpCalibrationConfig(n_x=161, n_z=101, time_scheme=time_scheme)
    return calibrate_leverage_surface(100.0, p, lv, np.diff(t_grid), np.full(n, 0.02),
                                      np.full(n, 0.0), eta=1.0, fp_config=cfg)


def _order(scheme):
    ref = _calibrate(scheme, 160)
    errs = [abs(_calibrate(scheme, n).leverage(100.0, 0.5) - ref.leverage(100.0, 0.5))
            for n in (20, 40, 80)]
    print(f"{scheme}: errs={['%.2e' % e for e in errs]}  "
          f"orders={[round(np.log2(errs[i] / errs[i + 1]), 2) for i in range(len(errs) - 1)]}")


def main():
    _order("backward_euler")
    _order("tr_bdf2")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the benchmark, then commit**

Run: `python example/fp_trbdf2_convergence.py`
Expected: `tr_bdf2` orders ≈ 2; `backward_euler` orders ≈ 1.

```bash
git add quantark/volmodels/slv/fokkerplanck/fp_solver.py quantark/volmodels/slv/fokkerplanck/calibration.py example/fp_trbdf2_convergence.py test/test_fp_trbdf2.py
git commit -m "feat(fp): opt-in TR-BDF2 second-order FP time march (WS-C6)

gamma=2-sqrt2 so the TR and BDF2 substep operators coincide (one factorization); backward-Euler
first step (Rannacher) damps the seed. backward_euler stays default; composes with krylov_lagged.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 7: WS-C4 default flip — central→chang_cooper + tighten `tol_neg` 0.5→0.05

**Files:**
- Modify: `quantark/volmodels/slv/fokkerplanck/config.py` (flip `flux_scheme` default, tighten `tol_neg` default)
- Modify: `test/test_fp_config.py`, `test/test_fp_acceptance.py` (deliberate golden/default updates)

**Precondition:** Task 4's `test_chang_cooper_cuts_negative_mass_at_least_10x` is green (the ≥10× gate holds). If it is not green, STOP — do not flip the default; report and pause per the kickoff decision ("hold the gate").

- [ ] **Step 1: Write the failing test**

Add to `test/test_fp_config.py`:

```python
def test_default_flux_scheme_is_chang_cooper_after_flip():
    cfg = FpCalibrationConfig()
    assert cfg.flux_scheme == "chang_cooper"
    assert cfg.tol_neg == 0.05
```

And add a regression guard to `test/test_fp_acceptance.py` that the tighter budget holds on the standard fixtures under the new default:

```python
def test_default_config_holds_tighter_tol_neg_on_standard_fixtures():
    # under the new default (chang_cooper + tol_neg=0.05) the standard fixtures still calibrate
    p = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    lev = _calibrate(_surface(), p, n=100, fp_config=FpCalibrationConfig(n_x=161, n_z=101))
    assert np.all(np.isfinite(lev.leverage_grid)) and np.all(lev.leverage_grid > 0)
    assert lev.diagnostics["max_negative_mass"] < 0.05      # within the tightened budget
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test/test_fp_config.py::test_default_flux_scheme_is_chang_cooper_after_flip -v`
Expected: FAIL — default is still `central` / `tol_neg=0.5`.

- [ ] **Step 3: Flip the defaults**

In `config.py`, change:

```python
    flux_scheme: str = "chang_cooper"     # {"central","chang_cooper"} default flipped after WS-C4 >=10x gate
```

and:

```python
    tol_neg: float = 0.05             # max negative probability mass sum_k w_k*max(-f_k,0). Tightened from
    #                                   0.5 to 0.05 once Chang-Cooper cut central-scheme negativity >=10x.
```

- [ ] **Step 4: Update the pre-flip default test**

In `test/test_fp_config.py`, the earlier `test_new_switch_defaults_preserve_current_behavior` asserted `flux_scheme == "central"`. Update it to reflect the post-flip default:

```python
def test_new_switch_defaults():
    cfg = FpCalibrationConfig()
    assert cfg.flux_scheme == "chang_cooper"     # flipped after the WS-C4 >=10x gate
    assert cfg.linear_solver == "direct"         # WS-B3 flip deferred (opt-in)
    assert cfg.time_scheme == "backward_euler"   # WS-C6 flip deferred (opt-in)
    assert cfg.refactor_every == 5
```

- [ ] **Step 5: Run the full FP suite under the new default**

Run: `python -m pytest test/test_fp_config.py test/test_fp_acceptance.py test/test_fp_solver.py test/test_fp_operators.py test/test_fp_chang_cooper.py test/test_fp_krylov.py test/test_fp_trbdf2.py test/test_fp_bootstrap.py test/test_fp_calibration.py test/test_fp_craig_sneyd.py test/test_fp_extents.py test/test_fp_fx_parity.py -v`
Expected: PASS. Any acceptance test that constructs `FpCalibrationConfig(...)` without `flux_scheme` now runs Chang–Cooper; the loose repricing gates (< 0.25 / < 0.4) hold since CC moves results within discretization tolerance. If a test pins a *tighter* central-specific value, set `flux_scheme="central"` explicitly in that test (documenting the intent) rather than loosening the gate.

- [ ] **Step 6: Run the whole suite**

Run: `python -m pytest -q`
Expected: all pass. Investigate any downstream SLV/engine test that relied implicitly on the old default; pin `flux_scheme="central"` where a test genuinely wants the legacy discretization.

- [ ] **Step 7: Commit**

```bash
git add quantark/volmodels/slv/fokkerplanck/config.py test/test_fp_config.py test/test_fp_acceptance.py
git commit -m "feat(fp): flip flux_scheme default central->chang_cooper + tighten tol_neg 0.5->0.05 (WS-C4)

The >=10x negative-mass reduction gate holds, so the positivity-preserving scheme becomes
the default and the negative-mass tripwire tightens 10x. WS-B3/C6 defaults stay opt-in.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Final verification

- [ ] **Run the complete test suite:** `python -m pytest -q` — all green.
- [ ] **Run all three benchmarks:**
  - `python example/fp_positivity_benchmark.py` → PASS (≥10×)
  - `python example/fp_krylov_benchmark.py` → parity < 1e-10, timing recorded
  - `python example/fp_trbdf2_convergence.py` → tr_bdf2 order ≈ 2
- [ ] Proceed to the Stage 6 code-review gate, then Stage 7 merge per feature-flow.

## Self-Review (spec coverage)

- **WS-C4 (F17 Chang–Cooper + F18 seed splitting):** Tasks 2 (fluxes), 3 (seed), 4 (wiring + ≥10× benchmark), 7 (default flip + tol_neg tighten). ✓ All four acceptance clauses: (i) ≥10× negativity — Task 4; (ii) flat-LV reprices Heston within tol — Task 4 `test_chang_cooper_still_reprices_flat_vanilla`; (iii) mass conservation ≤ mass_tol — Task 2 `test_cc_mass_conserved_high_correlation`; (iv) both modes green during transition — Tasks 2–4 keep central green, Task 7 flips.
- **WS-B3 (F20 Krylov):** Task 5. Acceptance: leverage parity < 1e-10 (incl. |ρ|=0.9) — `test_krylov_matches_direct`; wall-clock recorded — `example/fp_krylov_benchmark.py`. Default stays `direct`. ✓
- **WS-C6 (F19 TR-BDF2):** Task 6. Acceptance: second-order slope — `test_trbdf2_second_order_in_time`; Rannacher first step — `march_step(is_first)`; mass/negativity no worse — `test_trbdf2_mass_and_negativity_no_worse_than_be`. Default stays `backward_euler`. ✓
- **Global constraints:** equality gates (central/direct/backward_euler bit-identical) enforced by keeping the existing `test_fp_operators.py`/`test_fp_solver.py` green with zero pin changes through Tasks 1–6; no-silent-degradation enforced by the Krylov retry-then-raise; guarded `expm1` in `_cc_delta`. ✓
