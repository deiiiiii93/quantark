# Volmodels Phase 5 — Concentrated Grids, Degenerate v=0 Boundary, QE-M, LV Rannacher, Opt-in QMC

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the five Phase-5 accuracy workstreams (WS-C2 concentrated ADI grids, WS-C3 degenerate v=0 boundary, WS-C5 QE-M martingale correction, WS-C7 LV Rannacher + strike-aware grid, WS-D7 opt-in QMC) from `docs/superpowers/specs/2026-07-04-volmodels-dupire-heston-slv-improvements-design.md`, each behind an opt-in flag except WS-C7 (default-on, deliberate LV golden update).

**Architecture:** Four of five workstreams are strictly additive opt-in paths that leave every existing golden byte-for-byte unchanged; WS-C7 is the one deliberate golden move. The ADI changes (C2/C3) branch inside `HestonSLVADICore` on `grid_style`/`v0_boundary` so the uniform/Neumann scalar-dx code path is preserved verbatim. QE-M is a one-constant patch on the existing QUADEXP update. QMC is a `sampler` parameter that, when `None`, leaves the `np.random.default_rng` path bit-identical.

**Tech Stack:** NumPy, SciPy (`scipy.sparse`, `scipy.stats.ncx2`, `scipy.special.ndtri`, `scipy.stats.qmc`), pytest.

## Global Constraints

- **quantark is PUBLIC** — GitHub `deiiiiii93/quantark` primary (fresh history), Gitee private archive, PyPI 0.1.0 tag-triggered. `docs/` is gitignored on `main`; stage plan/spec with `git add -f`.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- **Never** import or instantiate any `*MCEngine` / Monte-Carlo path from a PDE/FP/deterministic engine or solver. Deterministic engines stay deterministic.
- **Never** invent fallback semantics. If a numerical path cannot be made correct, write a `# TODO` and stop — do not ship a "documented approximation" as an auto-default.
- Approximations are **opt-in modes, never auto-defaults**. Benchmark the exact path before assuming an approximation is needed. Default flips require a recorded benchmark + deliberate golden update.
- Use `quantark/util/numerical` guards (`is_zero`, `is_close`, `safe_log`, `safe_exp`, `safe_sqrt`, `safe_divide`, `Tolerance`) — never raw float comparisons or hardcoded tolerances in new library code.
- Exception hierarchy: `ValidationError` (bad input), `NumericalError` (instability/undefined), `MarketDataError`, `PricingError`.
- **Worktree testing:** the editable install resolves `quantark` to the main repo. To test worktree source, shadow with `PYTHONPATH=$PWD <main-venv>/bin/python -m pytest`.
- Run the full FP+SLV+volmodels suites at each task boundary; the cross-family agreement gates (analytical↔PDE↔MC Heston; LV MC↔LV PDE↔input-IV; SLV(flat LV,η=1)↔Heston) must stay green at every task boundary.

---

## File Structure

| File | Responsibility | Workstreams |
|------|----------------|-------------|
| `quantark/util/enum/engine_enums.py` | `HestonMCScheme.QUADEXP_M` member | WS-C5 |
| `quantark/volmodels/heston/mc_kernel.py` | QE-M martingale-corrected drift; `sampler` param | WS-C5, WS-D7 |
| `quantark/montecarlo/qmc_sobol.py` | `.uniform(...)` dual method on both generators | WS-D7 |
| `quantark/volmodels/slv/slv_mc_kernel.py` | `sampler` param | WS-D7 |
| `quantark/volmodels/localvol/mc_kernel.py` | `sampler` param | WS-D7 |
| `quantark/volmodels/localvol/pde_kernel.py` | `rannacher` default-on + strike mid-cell grid | WS-C7 |
| `quantark/util/numerical/finite_difference.py` | `fd1_interior_coeffs`, `fd2_interior_coeffs` | WS-C2 |
| `quantark/volmodels/adi_core.py` | `grid_style`, `v0_boundary` opt-in branches | WS-C2, WS-C3 |
| `quantark/volmodels/heston/pde_kernel.py`, `slv/slv_pde_kernel.py` | thread `grid_style`/`v0_boundary` through | WS-C2, WS-C3 |
| `example/volmodels_phase5_*.py` | benchmark/validation scripts | all |
| `test/test_qem_martingale.py`, `test/test_qmc_sampler.py`, `test/test_lv_rannacher.py`, `test/test_adi_degenerate_boundary.py`, `test/test_adi_concentrated_grid.py` | acceptance tests | all |

**Task ordering rationale:** C5 and D7 are MC-only and independent — do them first (lowest risk, no PDE golden interaction). C7 is LV-PDE-only. C3 then C2 are ADI-core changes; C3 (boundary row) is sequenced before C2 (grid geometry) because the concentrated v-grid path must honor whichever v=0 boundary is selected, and C3's tridiag-row edit is a smaller, self-contained diff to land first.

---

## Task 1: WS-C5 — QE-M martingale correction

**Files:**
- Modify: `quantark/util/enum/engine_enums.py:200-202` (add enum member)
- Modify: `quantark/volmodels/heston/mc_kernel.py` (QUADEXP branch → shared QUADEXP/QUADEXP_M branch; `need_u` gate)
- Test: `test/test_qem_martingale.py` (create)

**Interfaces:**
- Consumes: existing `_simulate_terminal_spot`, `price_european_heston_mc(..., scheme=HestonMCScheme.QUADEXP_M)`.
- Produces: `HestonMCScheme.QUADEXP_M`; QUADEXP output unchanged bit-for-bit.

**Math (Andersen 2008 §4.2, Prop. 4.1).** The existing QUADEXP log-spot update
`(drift−½v̄)Δ + corr + √(v̄Δ)·ρ̂·Z` with `corr=(ρ/σ)(V_{t+Δ}−V_t−κ(θ−v̄)Δ)`
is algebraically Andersen's central-γ (γ₁=γ₂=½) form
`drift·Δ + K0 + K1·V_t + K2·V_{t+Δ} + √(K3·V_t+K4·V_{t+Δ})·Z`, with
`K0=−ρκθΔ/σ`, `K1=½Δ(κρ/σ−½)−ρ/σ`, `K2=½Δ(κρ/σ−½)+ρ/σ`, `K3=K4=½Δ(1−ρ²)`.
QE-M replaces the *approximate* constant `K0` with the exact per-path
`K0* = −ln M_V(A) − (K1+½K3)·V_t`, `A=K2+½K4`, where `M_V(x)=E[e^{x V_{t+Δ}}|V_t]`
is the branch-selected CIR-transition MGF:
- quadratic branch (ψ≤ψc), `V=a(b+Z)²`: `M=exp(A a b²/(1−2Aa))/√(1−2Aa)`, needs `1−2Aa>0`.
- exponential branch (ψ>ψc): `M=p+(1−p)β/(β−A)`, needs `β−A>0`.
Then `E[S_{t+Δ}|F_t]=S_t e^{drift·Δ}` exactly. If the finiteness guard fails on any
active-branch path, raise `NumericalError` (do not fall back — per exact-semantics).

- [ ] **Step 1: Write the failing martingale test**

Create `test/test_qem_martingale.py`:

```python
"""WS-C5: Andersen QE-M martingale correction (HestonMCScheme.QUADEXP_M)."""
import numpy as np
import pytest

from quantark.util.enum.engine_enums import HestonMCScheme
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.heston.mc_kernel import price_european_heston_mc


def _forward_and_df(s0, r, q, T):
    return s0 * np.exp((r - q) * T), np.exp(-r * T)


def _run(scheme, s0=100.0, T=1.0, r=0.03, q=0.0, steps=8, num_paths=200_000, seed=7):
    # Feller-violated, high vol-of-vol, strong negative correlation: QE bias regime.
    params = HestonParams(kappa=1.0, theta=0.09, sigma=1.0, rho=-0.9, v0=0.09)
    dt = np.full(steps, T / steps)
    rf = np.full(steps, r)
    cf = np.full(steps, q)
    fwd, df = _forward_and_df(s0, r, q, T)
    # Price a deep-ITM call so payoff ~ S_T - K*df: E[payoff]*df ~ (fwd-K)*df tracks E[S_T].
    k = 1e-6  # near-zero strike -> call payoff == S_T, discounted price == S0 under martingale
    price, stderr = price_european_heston_mc(
        s0, k, True, params, dt, rf, cf, df, scheme=scheme,
        num_paths=num_paths, seed=seed, return_stderr=True,
    )
    # discounted E[S_T]; under exact martingale this equals s0.
    return price, stderr


def test_qem_removes_martingale_bias_where_qe_shows_it():
    p_qe, se_qe = _run(HestonMCScheme.QUADEXP)
    p_qem, se_qem = _run(HestonMCScheme.QUADEXP_M)
    s0 = 100.0
    # QE-M within 3 stderr of the exact forward (discounted E[S_T] == s0).
    assert abs(p_qem - s0) <= 3.0 * se_qem
    # And QE-M is at least as unbiased as QE in this regime.
    assert abs(p_qem - s0) <= abs(p_qe - s0) + 1e-9


def test_qem_reprices_european_within_mc_error_vs_analytical():
    from quantark.volmodels.heston.analytical_kernel import heston_call_price
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.0
    params = HestonParams(kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04)
    dt = np.full(24, T / 24); rf = np.full(24, r); cf = np.full(24, q)
    df = np.exp(-r * T)
    analytic = heston_call_price(s0, k, T, params, r, q)
    price, stderr = price_european_heston_mc(
        s0, k, True, params, dt, rf, cf, df, scheme=HestonMCScheme.QUADEXP_M,
        num_paths=200_000, seed=11, return_stderr=True,
    )
    assert abs(price - analytic) <= 4.0 * stderr + 1e-3


def test_quadexp_output_unchanged_by_qem_addition():
    # Guard: QUADEXP must be byte-identical to its pre-QE-M behavior (same seed/paths).
    s0, k, T, r, q = 100.0, 105.0, 1.0, 0.02, 0.01
    params = HestonParams(kappa=1.5, theta=0.05, sigma=0.6, rho=-0.5, v0=0.05)
    dt = np.full(12, T / 12); rf = np.full(12, r); cf = np.full(12, q)
    df = np.exp(-r * T)
    price = price_european_heston_mc(
        s0, k, True, params, dt, rf, cf, df, scheme=HestonMCScheme.QUADEXP,
        num_paths=50_000, seed=42,
    )
    # Pin the exact value once observed (fill in after Step 2 run of the CURRENT code).
    assert np.isfinite(price)
```

Check `heston_call_price` exists (analytical kernel) before relying on it:

```bash
grep -n "def heston_call_price" quantark/volmodels/heston/analytical_kernel.py
```
If the symbol name differs, adjust the import in the test to the actual analytical European entrypoint.

- [ ] **Step 2: Run the test to verify it fails**

Run: `PYTHONPATH=$PWD <main-venv>/bin/python -m pytest test/test_qem_martingale.py -x -q`
Expected: FAIL — `HestonMCScheme.QUADEXP_M` does not exist (AttributeError).

- [ ] **Step 3: Add the enum member**

In `quantark/util/enum/engine_enums.py`, edit the `HestonMCScheme` body (lines 200-202):

```python
    EULER = auto()       # full-truncation Euler on variance
    EULERLOG = auto()    # Euler in log-spot
    QUADEXP = auto()     # Andersen (2008) quadratic-exponential
    QUADEXP_M = auto()   # QUADEXP + Andersen §4.2 exact martingale (K0*) correction
```

- [ ] **Step 4: Implement the QE-M drift in the kernel**

In `quantark/volmodels/heston/mc_kernel.py`:

(a) Change the QUADEXP branch guard so it accepts both schemes. Replace
`if scheme == HestonMCScheme.QUADEXP:` (line 73) with:

```python
    if scheme in (HestonMCScheme.QUADEXP, HestonMCScheme.QUADEXP_M):
        martingale = scheme == HestonMCScheme.QUADEXP_M
        psi_c = 1.5
```
(delete the now-duplicate `psi_c = 1.5` line that followed).

(b) Replace the log-spot update block (current lines 125-131, from `v_bar = ...` through `v_n = v_np`) with:

```python
            v_bar = np.maximum(0.5 * (v_np + np.maximum(v_n, 0.0)), 0.0)
            if deterministic_vol:
                corr = 0.0
            else:
                corr = (rho / sigma) * (v_np - v_n - kappa * (theta - v_bar) * dt)
            base = (drift - 0.5 * v_bar) * dt + corr
            if martingale and not deterministic_vol:
                # Andersen §4.2 Prop. 4.1: swap the approximate constant K0 = -rho*kappa*theta*dt/sigma
                # for the exact per-path K0* so E[S_{t+dt}|F_t] = S_t*e^{drift*dt} exactly.
                ros = rho / sigma
                K3 = 0.5 * (1.0 - rho * rho) * dt          # == K4 (central gamma = 1/2)
                K1 = 0.5 * dt * (kappa * ros - 0.5) - ros
                K2 = 0.5 * dt * (kappa * ros - 0.5) + ros
                A = K2 + 0.5 * K3                           # coefficient on V_{t+dt} after E_Z
                quad_mask = psi <= psi_c
                denom_q = 1.0 - 2.0 * A * a                 # quadratic-branch MGF domain
                denom_e = beta - A                          # exponential-branch MGF domain
                bad = (quad_mask & (denom_q <= 0.0)) | (~quad_mask & (denom_e <= 0.0))
                if np.any(bad):
                    from quantark.util.exceptions import NumericalError
                    raise NumericalError(
                        "QE-M martingale MGF is undefined at these parameters "
                        "(A outside the CIR-transition MGF domain); tighten dt or use QUADEXP"
                    )
                safe_q = np.where(denom_q > 0.0, denom_q, 1.0)
                safe_e = np.where(denom_e > 0.0, denom_e, 1.0)
                m_quad = np.exp(A * a * b * b / safe_q) / np.sqrt(safe_q)
                m_exp = p + (1.0 - p) * beta / safe_e
                M = np.where(quad_mask, m_quad, m_exp)
                ln_M = np.log(M)
                K0 = -ros * kappa * theta * dt
                K0_star = -ln_M - (K1 + 0.5 * K3) * v_n
                base = base - K0 + K0_star                  # replace K0 with K0*
            log_s = log_s + base + np.sqrt(v_bar) * sqrt_dt * diff_coef * z_ind[:, i]
            v_n = v_np
        return np.exp(log_s)
```

(c) The uniform stream is consumed by both QE schemes. Update `need_u` in
`price_european_heston_mc` (line 179):

```python
    need_u = scheme in (HestonMCScheme.QUADEXP, HestonMCScheme.QUADEXP_M)
```

- [ ] **Step 5: Pin the QUADEXP-unchanged value and run**

Run the QUADEXP guard test first to capture the exact value, then paste it into the
assertion (replacing `assert np.isfinite(price)`):

Run: `PYTHONPATH=$PWD <main-venv>/bin/python -m pytest test/test_qem_martingale.py::test_quadexp_output_unchanged_by_qem_addition -q -s`
Then set `assert price == pytest.approx(<observed>, abs=0, rel=0)` (bit-identical pin).

Run the full file:
Run: `PYTHONPATH=$PWD <main-venv>/bin/python -m pytest test/test_qem_martingale.py -q`
Expected: PASS (all four tests).

- [ ] **Step 6: Regression — full Heston MC + analytical cross-family**

Run: `PYTHONPATH=$PWD <main-venv>/bin/python -m pytest test/ -k "heston and (mc or analytic or cross)" -q`
Expected: PASS, no QUADEXP goldens moved.

- [ ] **Step 7: Commit**

```bash
git add quantark/util/enum/engine_enums.py quantark/volmodels/heston/mc_kernel.py test/test_qem_martingale.py
git commit -m "feat(heston-mc): opt-in QE-M exact martingale correction (WS-C5)"
```

---

## Task 2: WS-D7 — Opt-in QMC sampler on the three MC kernels

**Files:**
- Modify: `quantark/montecarlo/qmc_sobol.py` (add `.uniform(...)` to both generators)
- Modify: `quantark/volmodels/heston/mc_kernel.py`, `slv/slv_mc_kernel.py`, `localvol/mc_kernel.py` (`sampler` param)
- Test: `test/test_qmc_sampler.py` (create)

**Interfaces:**
- Consumes: `PseudoRandomNormalGenerator`, `SobolNormalGenerator`.
- Produces: `.uniform(n_paths, dim, batch_id=None) -> (n_paths, dim) in (0,1)` on both;
  `sampler=None` default on `price_european_{heston,slv,lv}_mc` keeps the pseudo path bit-identical.

**Column layout (documented):** grouped-by-stream, matching the existing array shapes —
Heston/SLV: `[z_var(M) | z_ind(M) | u_var(M)]` (u block present only for QE/QE-M);
LV: `[z(M)]`. Variance normal first per the spec's "variance draws first" note.
`sampler` and `use_antithetic` are mutually exclusive (RQMC randomization is the
variance-reduction mechanism for QMC) — raise `ValidationError` if both are set.

- [ ] **Step 1: Write the failing sampler test**

Create `test/test_qmc_sampler.py`:

```python
"""WS-D7: opt-in QMC sampler on the volmodels MC kernels."""
import numpy as np
import pytest

from quantark.montecarlo.qmc_sobol import PseudoRandomNormalGenerator, SobolNormalGenerator
from quantark.util.enum.engine_enums import HestonMCScheme
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.heston.mc_kernel import price_european_heston_mc
from quantark.volmodels.localvol.mc_kernel import price_european_lv_mc
from quantark.volmodels.localvol.surface import LocalVolSurface


def test_generators_uniform_shape_and_range():
    for gen in (PseudoRandomNormalGenerator(seed=1), SobolNormalGenerator(base_seed=1)):
        u = gen.uniform(64, 3)
        assert u.shape == (64, 3)
        assert np.all(u > 0.0) and np.all(u < 1.0)


def test_sampler_none_is_bit_identical_pseudo():
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.02, 0.0
    params = HestonParams(kappa=1.5, theta=0.04, sigma=0.5, rho=-0.6, v0=0.04)
    dt = np.full(8, T / 8); rf = np.full(8, r); cf = np.full(8, q); df = np.exp(-r * T)
    p_default = price_european_heston_mc(s0, k, True, params, dt, rf, cf, df,
                                         scheme=HestonMCScheme.QUADEXP, num_paths=8192, seed=3)
    p_explicit_none = price_european_heston_mc(s0, k, True, params, dt, rf, cf, df,
                                               scheme=HestonMCScheme.QUADEXP, num_paths=8192,
                                               seed=3, sampler=None)
    assert p_default == p_explicit_none  # exact


def test_sampler_and_antithetic_are_mutually_exclusive():
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.02, 0.0
    params = HestonParams(kappa=1.5, theta=0.04, sigma=0.5, rho=-0.6, v0=0.04)
    dt = np.full(4, T / 4); rf = np.full(4, r); cf = np.full(4, q); df = np.exp(-r * T)
    with pytest.raises(Exception):
        price_european_heston_mc(s0, k, True, params, dt, rf, cf, df, num_paths=1024,
                                 use_antithetic=True, sampler=SobolNormalGenerator(base_seed=1))


def test_qmc_rmse_beats_pseudo_on_lv_european():
    # Flat LV: QMC should reach a lower RMSE-vs-analytical than pseudo at equal small path counts.
    from quantark.volmodels.black_scholes import bs_call_price
    s0, k, T, r, q, vol = 100.0, 100.0, 1.0, 0.03, 0.0, 0.2
    surface = LocalVolSurface.flat(vol)  # if no .flat, construct a constant surface (see note)
    dt = np.full(4, T / 4); rf = np.full(4, r); cf = np.full(4, q); df = np.exp(-r * T)
    analytic = bs_call_price(s0, k, T, vol, r, q)
    n = 4096
    err_pseudo, err_qmc = [], []
    for b in range(16):
        p_ps = price_european_lv_mc(s0, k, True, T, surface, dt, rf, cf, df,
                                    num_paths=n, seed=100 + b)
        p_q = price_european_lv_mc(s0, k, True, T, surface, dt, rf, cf, df,
                                   num_paths=n, sampler=SobolNormalGenerator(base_seed=100 + b))
        err_pseudo.append((p_ps - analytic) ** 2)
        err_qmc.append((p_q - analytic) ** 2)
    rmse_pseudo = np.sqrt(np.mean(err_pseudo))
    rmse_qmc = np.sqrt(np.mean(err_qmc))
    assert rmse_qmc < rmse_pseudo
```

Confirm the exact signatures/helpers referenced (`price_european_lv_mc` positional
order, `LocalVolSurface.flat`, `bs_call_price`) before running:

```bash
sed -n '19,40p' quantark/volmodels/localvol/mc_kernel.py
grep -n "def flat\|def __init__\|classmethod" quantark/volmodels/localvol/surface.py | head
grep -n "def bs_call_price" quantark/volmodels/black_scholes.py
```
If `LocalVolSurface.flat` does not exist, build a constant surface via its constructor
in the test's `surface` line instead.

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$PWD <main-venv>/bin/python -m pytest test/test_qmc_sampler.py::test_generators_uniform_shape_and_range -x -q`
Expected: FAIL — `.uniform` not defined (AttributeError).

- [ ] **Step 3: Add `.uniform()` to both generators**

In `quantark/volmodels`... no — in `quantark/montecarlo/qmc_sobol.py`.

Add to `PseudoRandomNormalGenerator` (after its `normal` method, ~line 84):

```python
    def uniform(
        self, n_paths: int, dim: int, batch_id: Optional[int] = None
    ) -> np.ndarray:
        """Uniform (0,1) samples, the dual of ``normal`` for inverse-CDF draws.

        ``batch_id`` is accepted for API symmetry but ignored (independent batches
        come from the RNG state).
        """
        return self._rng.random(size=(n_paths, dim))
```

Add to `SobolNormalGenerator` (after its `normal` method, ~line 189):

```python
    def uniform(
        self, n_paths: int, dim: int, batch_id: Optional[int] = None
    ) -> np.ndarray:
        """Scrambled Sobol uniforms in (0,1) (pre-ndtri), the dual of ``normal``.

        Uses exactly 2**m points (m = ceil(log2 n_paths)) to preserve balance, then
        truncates to ``n_paths``; clipped off {0,1} so downstream ndtri stays finite.
        """
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._check_scipy()
        n_total = _next_power_of_two(n_paths)
        if self.strict_power_of_two and n_total != n_paths:
            raise ValueError(
                f"SobolNormalGenerator with strict_power_of_two=True requires "
                f"n_paths to be a power of two, got {n_paths}."
            )
        m = int(np.log2(n_total))
        engine = self._make_engine(dim=dim, batch_id=batch_id)
        u = engine.random_base2(m)
        eps = 1e-12
        u = np.clip(u, eps, 1.0 - eps)
        if n_paths != n_total:
            u = u[:n_paths]
        return np.asarray(u, dtype=float)
```

Add `"uniform"` is a method (not in `__all__`; `__all__` lists classes). No change to `__all__`.

- [ ] **Step 4: Thread `sampler` through the Heston MC kernel**

In `quantark/volmodels/heston/mc_kernel.py`, add `sampler=None` to
`price_european_heston_mc` signature (after `use_antithetic: bool = False`):

```python
    use_antithetic: bool = False,
    sampler=None,
    return_stderr: bool = False,
```

Replace the RNG draw block (lines 172-193, from `rng = np.random.default_rng(seed)`
through the `else:` pseudo branch) with:

```python
    half = (num_paths + 1) // 2
    n_eff = 2 * half if use_antithetic else num_paths
    need_u = scheme in (HestonMCScheme.QUADEXP, HestonMCScheme.QUADEXP_M)

    if sampler is not None:
        if use_antithetic:
            raise ValidationError("sampler and use_antithetic are mutually exclusive")
        from scipy.special import ndtri
        n_streams = 3 if need_u else 2          # [z_var | z_ind | (u_var)]
        block = np.asarray(sampler.uniform(num_paths, n_streams * M), dtype=float)
        block = np.clip(block, 1e-12, 1.0 - 1e-12)
        z_var = ndtri(block[:, 0:M])
        z_ind = ndtri(block[:, M:2 * M])
        u_var = block[:, 2 * M:3 * M] if need_u else None
    else:
        rng = np.random.default_rng(seed)
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

(The `need_u` line previously at 179 is now folded in; remove the old standalone `need_u = ...` line.)

- [ ] **Step 5: Thread `sampler` through the SLV and LV MC kernels**

**LV kernel** (`quantark/volmodels/localvol/mc_kernel.py`): add `sampler=None` to the
signature (after `use_antithetic`), then in the RNG section (around lines 62-76) branch:

```python
    n_eff = 2 * half if use_antithetic else num_paths
    if sampler is not None:
        if use_antithetic:
            raise ValidationError("sampler and use_antithetic are mutually exclusive")
        from scipy.special import ndtri
        block = np.clip(np.asarray(sampler.uniform(num_paths, M), dtype=float),
                        1e-12, 1.0 - 1e-12)
        z_all = ndtri(block)                 # (num_paths, M); one normal stream per step
        use_sampler = True
    else:
        rng = np.random.default_rng(seed)
        use_sampler = False
```
Then in the per-step loop, source `z` from `z_all[:, i]` when `use_sampler`, else keep
the existing `rng.standard_normal` draw. (Read the current loop with
`sed -n '62,95p' quantark/volmodels/localvol/mc_kernel.py` and wire `z = z_all[:, i]`
in the sampler branch, preserving the antithetic and pseudo branches verbatim.)

**SLV kernel** (`quantark/volmodels/slv/slv_mc_kernel.py`): `_simulate_slv` draws
`z_v`/`z_i` per step inside the loop (lines 92-102). Add a `sampler=None` param to
`price_european_slv_mc` and pass a pre-drawn `(num_paths, 2, M)` uniform block into
`_simulate_slv` (new optional `qmc_block=None` arg). In `_simulate_slv`, when
`qmc_block is not None`, use `dW_v = sqrt_dt * qmc_block_z[:, 0, i]`,
`dW_s = rho*dW_v + rho_bar*sqrt_dt*qmc_block_z[:, 1, i]` where
`qmc_block_z = ndtri(block)`, column 0 = variance normal, column 1 = spot-independent
normal; guard `use_antithetic` mutually exclusive. Build the block in
`price_european_slv_mc`:

```python
    if sampler is not None:
        if use_antithetic:
            raise ValidationError("sampler and use_antithetic are mutually exclusive")
        from scipy.special import ndtri
        raw = np.clip(np.asarray(sampler.uniform(num_paths, 2 * M), dtype=float),
                      1e-12, 1.0 - 1e-12)
        qmc_z = ndtri(raw).reshape(num_paths, 2, M)   # [:,0,:]=z_var, [:,1,:]=z_ind
    else:
        qmc_z = None
```
Pass `qmc_z` to `_simulate_slv(...)` and select the draw source per step there.

- [ ] **Step 6: Document the dimension layout**

Add to each kernel's docstring a "QMC dimension layout" line, e.g. for Heston:
`"sampler columns: [z_var(M) | z_ind(M) | u_var(M)]; u block present only for QE/QE-M."`

- [ ] **Step 7: Run the sampler tests + MC regressions**

Run: `PYTHONPATH=$PWD <main-venv>/bin/python -m pytest test/test_qmc_sampler.py -q`
Expected: PASS.
Run: `PYTHONPATH=$PWD <main-venv>/bin/python -m pytest test/ -k "mc" -q`
Expected: PASS — every `sampler=None` default path bit-identical, no goldens moved.

- [ ] **Step 8: Commit**

```bash
git add quantark/montecarlo/qmc_sobol.py quantark/volmodels/heston/mc_kernel.py \
        quantark/volmodels/slv/slv_mc_kernel.py quantark/volmodels/localvol/mc_kernel.py \
        test/test_qmc_sampler.py
git commit -m "feat(volmodels-mc): opt-in QMC sampler on Heston/SLV/LV kernels (WS-D7)"
```

---

## Task 3: WS-C7 — LV PDE Rannacher (default on) + strike mid-cell grid

**Files:**
- Modify: `quantark/volmodels/localvol/pde_kernel.py` (`_solve_lv_pde` + both public entrypoints)
- Test: `test/test_lv_rannacher.py` (create)

**Interfaces:**
- Consumes: `_solve_lv_pde(..., rannacher=True)`.
- Produces: `rannacher: bool = True` on `_solve_lv_pde`, `price_european_lv_pde`,
  `price_delta_gamma_european_lv_pde`; **deliberate LV golden move** (grid + first-step change).

**Design.** (1) Rannacher: replace the first backward step (m = M−1, the step from T)
with two fully-implicit half-steps (θ_loc=1) of `dt_m/2`, damping the payoff kink so
CN gamma stops oscillating. (2) Strike mid-cell: **adjust the grid spacing** (not a
whole-grid shift) so K lands halfway between two nodes while the lower node stays exactly
at `S=0` and the boundary economics are preserved — pick `j=round(K/ds−½)` and set
`ds=K/(j+½)`, `smax=ds·(N−1)`, `s_grid=linspace(0,smax,N)` (so `s_grid[0]==0`, `K` is
mid-cell between `s_grid[j]` and `s_grid[j+1]`, and the upper bound moves by `<ds`). Both
alter node placement / first-step semantics → goldens updated deliberately (spec §WS-C7).

> **Do NOT shift the whole grid by a positive offset** — that moves `s_grid[0]` off `S=0`
> while `boundaries()` still imposes the zero-spot value there, corrupting puts and
> low-spot prices (Codex plan-gate [high]). Adjusting spacing keeps both boundary nodes
> at their economic locations.

- [ ] **Step 1: Write the failing gamma-oscillation test**

Create `test/test_lv_rannacher.py`:

```python
"""WS-C7: LV Crank-Nicolson Rannacher start-up + strike mid-cell grid."""
import numpy as np

from quantark.volmodels.localvol.pde_kernel import (
    _solve_lv_pde, price_european_lv_pde, price_delta_gamma_european_lv_pde,
)
from quantark.volmodels.localvol.surface import LocalVolSurface
from quantark.volmodels.black_scholes import bs_call_price


def _flat_surface(vol=0.2):
    try:
        return LocalVolSurface.flat(vol)
    except AttributeError:  # constant surface via constructor (adjust to actual API)
        raise


def test_rannacher_default_is_on():
    import inspect
    sig = inspect.signature(price_european_lv_pde)
    assert sig.parameters["rannacher"].default is True


def test_rannacher_removes_gamma_oscillation_near_strike():
    # Short-dated ATM: CN without Rannacher ripples in gamma across the strike cells.
    s0, k, T, r, q = 100.0, 100.0, 0.1, 0.03, 0.0
    surface = _flat_surface(0.2)
    dt = np.full(20, T / 20); rf = np.full(20, r); cf = np.full(20, q)
    s_grid, v_on = _solve_lv_pde(s0, k, True, T, surface, dt, rf, cf, n_s=201,
                                 rannacher=True)
    s_grid2, v_off = _solve_lv_pde(s0, k, True, T, surface, dt, rf, cf, n_s=201,
                                   rannacher=False)
    # gamma = second difference on the near-strike window
    def max_gamma_ripple(sg, v):
        mask = (sg > 0.8 * k) & (sg < 1.2 * k)
        g = np.gradient(np.gradient(v, sg), sg)[mask]
        return np.max(np.abs(np.diff(g, 2)))  # curvature of gamma = ripple proxy
    assert max_gamma_ripple(s_grid, v_on) < max_gamma_ripple(s_grid2, v_off)


def test_price_still_matches_bs_within_tolerance():
    s0, k, T, r, q, vol = 100.0, 105.0, 1.0, 0.03, 0.01, 0.2
    surface = _flat_surface(vol)
    dt = np.full(50, T / 50); rf = np.full(50, r); cf = np.full(50, q)
    price = price_european_lv_pde(s0, k, True, T, surface, dt, rf, cf, n_s=400)
    assert abs(price - bs_call_price(s0, k, T, vol, r, q)) < 0.05


def test_strike_is_mid_cell_and_boundary_nodes_preserved():
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.0
    surface = _flat_surface(0.2)
    dt = np.full(10, T / 10); rf = np.full(10, r); cf = np.full(10, q)
    s_grid, _ = _solve_lv_pde(s0, k, True, T, surface, dt, rf, cf, n_s=101)
    ds = s_grid[1] - s_grid[0]
    # K exactly mid-cell: distance to nearest node ~ ds/2, not ~0.
    d = np.min(np.abs(s_grid - k))
    assert abs(d - ds / 2.0) < 0.02 * ds
    # Lower boundary node stays at S=0 (spacing adjusted, grid NOT shifted).
    assert s_grid[0] == 0.0
    assert np.isclose(s_grid[-1], ds * (len(s_grid) - 1))


def test_put_and_low_spot_prices_match_bs():
    # The mid-cell change must not corrupt puts or low-spot cases (boundary economics).
    T, r, q, vol = 1.0, 0.03, 0.01, 0.2
    surface = _flat_surface(vol)
    dt = np.full(50, T / 50); rf = np.full(50, r); cf = np.full(50, q)
    from quantark.volmodels.black_scholes import bs_put_price
    for s0, k in [(100.0, 100.0), (60.0, 100.0), (100.0, 140.0)]:
        p_put = price_european_lv_pde(s0, k, False, T, surface, dt, rf, cf, n_s=400)
        assert abs(p_put - bs_put_price(s0, k, T, vol, r, q)) < 0.06
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$PWD <main-venv>/bin/python -m pytest test/test_lv_rannacher.py::test_rannacher_default_is_on -x -q`
Expected: FAIL — `rannacher` param does not exist.

- [ ] **Step 3: Add `rannacher` + mid-cell grid to `_solve_lv_pde`**

In `quantark/volmodels/localvol/pde_kernel.py`, add `rannacher: bool = True` to the
`_solve_lv_pde` signature (after `theta`). Replace the grid construction (lines 69-71)
with a mid-cell-shifted grid:

```python
    # Strike mid-cell (kink-averaging): adjust the SPACING so K sits halfway between two
    # nodes while s_grid[0] stays exactly at S=0 and s_grid[-1] stays the upper boundary.
    # Deliberate golden move (WS-C7). (Never shift the whole grid off S=0.)
    ds_nom = smax / (N - 1)
    j = max(int(round(strike / ds_nom - 0.5)), 0)   # cell index whose midpoint hosts K
    ds = strike / (j + 0.5)                          # K == (j + 0.5) * ds  (mid-cell)
    smax = ds * (N - 1)                              # upper bound moves by < ds
    if smax <= s0:
        raise ValidationError("s_max (after mid-cell adjustment) must exceed spot")
    s_grid = np.linspace(0.0, smax, N)               # s_grid[0] == 0 preserved
    s_int = s_grid[1:-1]
```

Extract the per-step CN update into a local closure so the first step can call it
twice at θ=1. Replace the backward loop (lines 90-117) with:

```python
    def cn_step(v, m, dt_m, theta_loc):
        r_m, carry_m = rf[m], cf[m]
        t_mid = node_t[m] + 0.5 * dt_m
        left_next, right_next = boundaries(m + 1)
        left_curr, right_curr = boundaries(m)
        v = v.copy()
        v[0], v[-1] = left_next, right_next
        sigma = np.asarray(lv_surface.local_vol(s_int, t_mid), dtype=float)
        alpha = 0.5 * sigma * sigma * s_int * s_int / (ds * ds)
        beta = (r_m - carry_m) * s_int / (2.0 * ds)
        A = alpha - beta
        B = -2.0 * alpha - r_m
        C = alpha + beta
        sub_A = -theta_loc * dt_m * A[1:]
        diag_A = 1.0 - theta_loc * dt_m * B
        sup_A = -theta_loc * dt_m * C[:-1]
        rhs = (1.0 + (1.0 - theta_loc) * dt_m * B) * v[1:-1]
        rhs[:-1] += (1.0 - theta_loc) * dt_m * C[:-1] * v[2:-1]
        rhs[1:] += (1.0 - theta_loc) * dt_m * A[1:] * v[1:-2]
        rhs[0] += (1.0 - theta_loc) * dt_m * A[0] * left_next + theta_loc * dt_m * A[0] * left_curr
        rhs[-1] += (1.0 - theta_loc) * dt_m * C[-1] * right_next + theta_loc * dt_m * C[-1] * right_curr
        v[1:-1] = solve_tridiag(sub_A, diag_A, sup_A, rhs)
        v[0], v[-1] = left_curr, right_curr
        return v

    for m in range(M - 1, -1, -1):
        if rannacher and m == M - 1:
            # Rannacher: two fully-implicit half-steps for the first (terminal) step.
            half = 0.5 * dt[m]
            # first half-step lands at node time node_t[m]+half; reuse cn_step with a
            # temporary node by splitting: implicit Euler on [T, T-half] then [T-half, T-dt].
            v = _cn_half_implicit(v, m, half, node_t, rf, cf, boundaries, lv_surface, s_int, ds)
            v = _cn_half_implicit(v, m, half, node_t, rf, cf, boundaries, lv_surface, s_int, ds,
                                  second=True)
        else:
            v = cn_step(v, m, dt[m], theta)

    return s_grid, v
```

Add the half-step helper as a module-level function (Rannacher needs the midpoint at
`node_t[m] + 0.25*dt_m` and `+0.75*dt_m` for the two half-steps, with θ=1 and the
boundary interpolated at the intermediate node times):

```python
def _cn_half_implicit(v, m, half, node_t, rf, cf, boundaries, lv_surface, s_int, ds,
                      second=False):
    """One fully-implicit (theta=1) half-step of the LV CN march (Rannacher start-up).

    ``second=False`` covers [t_{m+1}, t_{m+1}-half]; ``second=True`` covers
    [t_{m+1}-half, t_m]. Boundaries use the step endpoints; the discount-factor
    boundary values are second-order-accurate at the half nodes via the same
    ``boundaries`` closure evaluated at the bracketing whole nodes (half-node boundary
    error is O(dt) at two rows only and Rannacher's purpose is kink damping, not
    boundary order).
    """
    import numpy as np
    from quantark.util.numerical import solve_tridiag
    dt_m = 2.0 * half
    r_m, carry_m = rf[m], cf[m]
    # local vol at the half-step midpoint
    t0 = node_t[m + 1]
    t_mid = (t0 - 0.25 * dt_m) if not second else (t0 - 0.75 * dt_m)
    left_next, right_next = boundaries(m + 1) if not second else _interp_bndry(
        boundaries, m, 0.5)
    left_curr, right_curr = (_interp_bndry(boundaries, m, 0.5) if not second
                             else boundaries(m))
    v = v.copy()
    v[0], v[-1] = left_next, right_next
    sigma = np.asarray(lv_surface.local_vol(s_int, t_mid), dtype=float)
    alpha = 0.5 * sigma * sigma * s_int * s_int / (ds * ds)
    beta = (r_m - carry_m) * s_int / (2.0 * ds)
    A = alpha - beta
    B = -2.0 * alpha - r_m
    C = alpha + beta
    sub_A = -half * A[1:]
    diag_A = 1.0 - half * B
    sup_A = -half * C[:-1]
    rhs = v[1:-1].copy()
    rhs[0] += half * A[0] * left_curr
    rhs[-1] += half * C[-1] * right_curr
    v[1:-1] = solve_tridiag(sub_A, diag_A, sup_A, rhs)
    v[0], v[-1] = left_curr, right_curr
    return v


def _interp_bndry(boundaries, m, frac):
    """Linear blend of the whole-node boundary values at fractional node m+frac."""
    l1, r1 = boundaries(m + 1)
    l0, r0 = boundaries(m)
    return (1.0 - frac) * l1 + frac * l0, (1.0 - frac) * r1 + frac * r0
```

> **Reviewer note to self:** if the `cn_step`/`_cn_half_implicit` duplication is flagged,
> collapse both into one `_theta_step(v, m, dt_m, theta_loc, t_mid, l_next, r_next,
> l_curr, r_curr)` primitive and have `cn_step` and the two half-steps call it. Keep the
> refactor within this task; do not change the uniform-CN numerics for `rannacher=False`.

- [ ] **Step 4: Propagate `rannacher` through the public entrypoints**

Add `rannacher: bool = True` to `price_european_lv_pde` and
`price_delta_gamma_european_lv_pde` signatures (after `theta`) and pass it into
`_solve_lv_pde(...)` in both.

- [ ] **Step 5: Run the WS-C7 tests**

Run: `PYTHONPATH=$PWD <main-venv>/bin/python -m pytest test/test_lv_rannacher.py -q`
Expected: PASS.

- [ ] **Step 6: Update LV PDE goldens (deliberate)**

Run the LV PDE suite; expect price goldens to move slightly (grid shift + Rannacher):
Run: `PYTHONPATH=$PWD <main-venv>/bin/python -m pytest test/ -k "lv and pde" -q`
For each moved golden, verify the new value is *closer* to the analytical/LV-MC
reference (record before/after in the benchmark script, Task 6), then update the
expected constant. Do **not** loosen tolerances to hide a move — update the pinned value.
Re-run the LV MC↔LV PDE cross-family gate and confirm green.

- [ ] **Step 7: Commit**

```bash
git add quantark/volmodels/localvol/pde_kernel.py test/test_lv_rannacher.py test/<moved-lv-goldens>
git commit -m "feat(lv-pde): Rannacher start-up + strike mid-cell grid, default on (WS-C7)"
```

---

## Task 4: WS-C3 — Degenerate v=0 boundary (opt-in)

**Files:**
- Modify: `quantark/volmodels/adi_core.py` (`v0_boundary` param; `_A2`, `_tri_V`, `_solve_V`, `_bc`)
- Modify: `quantark/volmodels/heston/pde_kernel.py`, `slv/slv_pde_kernel.py` (thread param)
- Test: `test/test_adi_degenerate_boundary.py` (create)

**Interfaces:**
- Consumes: `HestonSLVADICore(..., v0_boundary="neumann"|"degenerate_pde")`.
- Produces: `v0_boundary: str = "neumann"` on the core and both PDE entrypoints
  (`price_european_heston_pde`, `price_european_slv_pde`, and their delta/gamma twins).

**Design.** At v=0 the CIR diffusion `½σ²v·U_vv` vanishes, leaving the degenerate row
`U_τ = κθ·U_v(forward one-sided) + [x-operator + reaction]`. The x-operator and the
implicit `−rU` at the j=0 column are **already** handled by the S-sweep (`_tri_S` builds
all V-slices including v=0, where `L²v=0` collapses it to pure drift + reaction). The
only piece Neumann discards is the `κθ·U_v` convection. `v0_boundary="degenerate_pde"`
replaces the Neumann V-row 0 (`b=1, c=−1`) with the 2-point forward-convection row and
stops `_bc` from overwriting the solved v=0 column. Documented O(dV) local error at the
one boundary row (spec §WS-C3). Default `"neumann"` keeps every existing golden.

- [ ] **Step 1: Write the failing degenerate-boundary test**

Create `test/test_adi_degenerate_boundary.py`:

```python
"""WS-C3: degenerate v=0 boundary for the Heston/SLV ADI core."""
import numpy as np
import pytest

from quantark.util.enum.engine_enums import ADIScheme
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.heston.pde_kernel import price_european_heston_pde
from quantark.volmodels.heston.analytical_kernel import heston_call_price


def _feller_violated():
    # 2*kappa*theta = 2*0.5*0.04 = 0.04 << sigma^2 = 0.81  (strongly Feller-violated)
    return HestonParams(kappa=0.5, theta=0.04, sigma=0.9, rho=-0.5, v0=0.04)


def test_default_is_neumann():
    import inspect
    sig = inspect.signature(price_european_heston_pde)
    assert sig.parameters["v0_boundary"].default == "neumann"


def test_degenerate_boundary_reduces_feller_violated_error():
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.0
    params = _feller_violated()
    ref = heston_call_price(s0, k, T, params, r, q)
    common = dict(n_x=200, n_v=80, n_t=100, scheme=ADIScheme.CRAIG_SNEYD)
    p_neu = price_european_heston_pde(s0, k, True, T, params, r, q,
                                      v0_boundary="neumann", **common)
    p_deg = price_european_heston_pde(s0, k, True, T, params, r, q,
                                      v0_boundary="degenerate_pde", **common)
    assert abs(p_deg - ref) <= abs(p_neu - ref) + 1e-6  # no worse; expected better


def test_feller_satisfied_case_essentially_unchanged():
    # 2*kappa*theta = 2*3*0.04 = 0.24 > sigma^2 = 0.04 (Feller satisfied): boundary inert.
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.0
    params = HestonParams(kappa=3.0, theta=0.04, sigma=0.2, rho=-0.5, v0=0.04)
    common = dict(n_x=200, n_v=80, n_t=100, scheme=ADIScheme.CRAIG_SNEYD)
    p_neu = price_european_heston_pde(s0, k, True, T, params, r, q,
                                      v0_boundary="neumann", **common)
    p_deg = price_european_heston_pde(s0, k, True, T, params, r, q,
                                      v0_boundary="degenerate_pde", **common)
    assert abs(p_deg - p_neu) < 5e-3


def test_invalid_v0_boundary_raises():
    with pytest.raises(Exception):
        price_european_heston_pde(100.0, 100.0, True, 1.0, _feller_violated(), 0.03, 0.0,
                                  v0_boundary="bogus")
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$PWD <main-venv>/bin/python -m pytest test/test_adi_degenerate_boundary.py::test_default_is_neumann -x -q`
Expected: FAIL — `v0_boundary` param does not exist.

- [ ] **Step 3: Add `v0_boundary` to the core and branch the v=0 row**

In `quantark/volmodels/adi_core.py`:

(a) `__init__` signature — add `v0_boundary="neumann"` after `grid_spot=None`, and store
with validation:

```python
                 use_sparse=False, grid_spot=None, v0_boundary="neumann"):
```
and after the existing assignments (near line 55):

```python
        if v0_boundary not in ("neumann", "degenerate_pde"):
            raise ValidationError("v0_boundary must be 'neumann' or 'degenerate_pde'")
        self.v0_boundary = v0_boundary
        self._degenerate_v0 = v0_boundary == "degenerate_pde"
```
Add `from quantark.util.exceptions import ValidationError` to the imports.

(b) `_A2` — when degenerate, add the forward-convection contribution on the j=0 interior-x
column (insert before `return out`, after the `out[1:-1, 1:-1] = ...` assignment):

```python
        if self._degenerate_v0:
            # v=0 row: only kappa*theta*U_v survives (diffusion vanishes); 2-point forward.
            out[1:-1, 0] = self.kappa * self.theta * (U[1:-1, 1] - U[1:-1, 0]) / self.dV
        return out
```

(c) `_tri_V` — replace the Neumann row-0 assignment. Change the boundary lines
(currently `b[0] = 1.0; c[0] = -1.0; a[-1] = -1.0; b[-1] = 1.0`) to:

```python
        if self._degenerate_v0:
            conv = self.kappa * self.theta / dV
            b[0] = 1.0 + theta_loc * dt_step * conv
            c[0] = -theta_loc * dt_step * conv
        else:
            b[0] = 1.0; c[0] = -1.0
        a[-1] = -1.0; b[-1] = 1.0
```
(The upper v=V_max Neumann row is unchanged.)

(d) `_solve_V` — the v=0 rhs. In the batched (dense) branch, `rhs[:, 0] = 0.0` currently
imposes the Neumann RHS. Gate it:

```python
        rhs = source - theta_loc * dt_step * A2U
        if not self._degenerate_v0:
            rhs[:, 0] = 0.0
        rhs[:, -1] = 0.0
```
In the sparse branch, `rhs[0] = 0.0; rhs[-1] = 0.0` → gate the first:

```python
                if not self._degenerate_v0:
                    rhs[0] = 0.0
                rhs[-1] = 0.0
```
(When degenerate, the v=0 rhs is the swept `source − θ dt A2U` value, i.e. the normal
interior formula, so the degenerate PDE row is actually solved.)

(e) `_bc` — do not overwrite the solved v=0 column when degenerate. Change
`U[:, 0] = U[:, 1]` to:

```python
        if not self._degenerate_v0:
            U[:, 0] = U[:, 1]
        U[:, -1] = U[:, -2]
```

- [ ] **Step 4: Thread `v0_boundary` through the PDE wrappers**

In `quantark/volmodels/heston/pde_kernel.py`, add `v0_boundary: str = "neumann"` to
`price_european_heston_pde` and `price_delta_gamma_heston_pde` (after `grid_spot`), and
pass it into the two `HestonSLVADICore(...)` constructions:

```python
    solver = HestonSLVADICore(s0, strike, T, r, carry, params, n_x, n_v, n_t,
                              leverage=None, eta=1.0, use_sparse=use_sparse,
                              grid_spot=(grid_spot if grid_spot > 0 else None),
                              v0_boundary=v0_boundary)
```
Do the same in `quantark/volmodels/slv/slv_pde_kernel.py` for `price_european_slv_pde`
and `price_delta_gamma_slv_pde`.

- [ ] **Step 5: Run WS-C3 tests + full PDE regression**

Run: `PYTHONPATH=$PWD <main-venv>/bin/python -m pytest test/test_adi_degenerate_boundary.py -q`
Expected: PASS.
Run: `PYTHONPATH=$PWD <main-venv>/bin/python -m pytest test/ -k "pde or adi or heston or slv" -q`
Expected: PASS — default `"neumann"` path bit-identical, no goldens moved.

- [ ] **Step 6: Commit**

```bash
git add quantark/volmodels/adi_core.py quantark/volmodels/heston/pde_kernel.py \
        quantark/volmodels/slv/slv_pde_kernel.py test/test_adi_degenerate_boundary.py
git commit -m "feat(adi): opt-in degenerate v=0 boundary for Feller-violated Heston/SLV (WS-C3)"
```

---

## Task 5: WS-C2 — Concentrated (x, v) ADI grids (opt-in)

**Files:**
- Modify: `quantark/util/numerical/finite_difference.py` (add coefficient extractors)
- Modify: `quantark/util/numerical/__init__.py` (export them)
- Modify: `quantark/volmodels/adi_core.py` (`grid_style`; non-uniform operator/tridiag branch)
- Modify: `quantark/volmodels/heston/pde_kernel.py`, `slv/slv_pde_kernel.py` (thread param)
- Test: `test/test_adi_concentrated_grid.py` (create)

**Interfaces:**
- Consumes: `fd1_interior_coeffs(x)`, `fd2_interior_coeffs(x)`;
  `HestonSLVADICore(..., grid_style="uniform"|"concentrated")`.
- Produces: `grid_style: str = "uniform"` on the core and PDE entrypoints; uniform path
  bit-identical (branch, do not rewrite scalar-dx code).

**Design.** `grid_style="concentrated"` builds a sinh-concentrated X-grid around `ln K`
and a sinh-concentrated V-grid around `min(v0, θ)` with a CIR-quantile extent (reuse
`concentrated_grid` and `z_extents` from `slv/fokkerplanck/coordinates.py`). The uniform
scalar-dx operators/tridiag builders are the special case and stay verbatim behind the
`if self._uniform:` guard; the concentrated branch uses precomputed per-node stencil
coefficient arrays. **Critical:** never route the uniform grid through the general
coefficient path — the last-ULP rounding differs and would move every existing golden
(and break the WS-D1 1e-13 Heston≡SLV gate).

- [ ] **Step 1: Write coefficient-extractor tests**

Create `test/test_adi_concentrated_grid.py` (start with the FD-coeff unit tests):

```python
"""WS-C2: non-uniform stencil coefficients + concentrated-grid ADI convergence."""
import numpy as np
import pytest

from quantark.util.numerical.finite_difference import (
    fd1_interior_coeffs, fd2_interior_coeffs, fd1_nonuniform, fd2_nonuniform,
)


def test_fd1_coeffs_exact_for_quadratic_on_nonuniform_grid():
    x = np.array([0.0, 0.3, 0.7, 1.2, 2.0, 3.5])
    f = 2.0 * x ** 2 - 3.0 * x + 1.0
    wm, w0, wp = fd1_interior_coeffs(x)
    approx = wm * f[:-2] + w0 * f[1:-1] + wp * f[2:]
    exact = 4.0 * x[1:-1] - 3.0
    assert np.allclose(approx, exact, atol=1e-10)


def test_fd2_coeffs_exact_for_quadratic_on_nonuniform_grid():
    x = np.array([0.0, 0.3, 0.7, 1.2, 2.0, 3.5])
    f = 2.0 * x ** 2 - 3.0 * x + 1.0
    wm, w0, wp = fd2_interior_coeffs(x)
    approx = wm * f[:-2] + w0 * f[1:-1] + wp * f[2:]
    assert np.allclose(approx, 4.0, atol=1e-10)


def test_coeffs_match_applied_stencils():
    x = np.array([0.0, 0.3, 0.7, 1.2, 2.0, 3.5])
    f = np.sin(x)
    wm1, w01, wp1 = fd1_interior_coeffs(x)
    assert np.allclose(wm1 * f[:-2] + w01 * f[1:-1] + wp1 * f[2:],
                       fd1_nonuniform(f, x)[1:-1], atol=1e-14)
    wm2, w02, wp2 = fd2_interior_coeffs(x)
    assert np.allclose(wm2 * f[:-2] + w02 * f[1:-1] + wp2 * f[2:],
                       fd2_nonuniform(f, x)[1:-1], atol=1e-14)


def test_uniform_grid_coeffs_reduce_to_scalar_form():
    x = np.linspace(0.0, 1.0, 7)
    h = x[1] - x[0]
    wm1, w01, wp1 = fd1_interior_coeffs(x)
    assert np.allclose(wm1, -1.0 / (2 * h)) and np.allclose(w01, 0.0) and np.allclose(wp1, 1.0 / (2 * h))
    wm2, w02, wp2 = fd2_interior_coeffs(x)
    assert np.allclose(wm2, 1.0 / h ** 2) and np.allclose(w02, -2.0 / h ** 2) and np.allclose(wp2, 1.0 / h ** 2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=$PWD <main-venv>/bin/python -m pytest test/test_adi_concentrated_grid.py -x -q`
Expected: FAIL — `fd1_interior_coeffs` not defined.

- [ ] **Step 3: Add the coefficient extractors**

In `quantark/util/numerical/finite_difference.py`, append:

```python
def fd1_interior_coeffs(x: np.ndarray):
    """Interior non-uniform first-derivative stencil weights (wm, w0, wp), each (N-2,).

    ``d f/dx |_j ≈ wm[j-1] f[j-1] + w0[j-1] f[j] + wp[j-1] f[j+1]`` for j=1..N-2.
    Reduces to the uniform (-1/2h, 0, 1/2h) at equal spacing; exact for quadratics.
    """
    x = np.asarray(x, dtype=float)
    dxm = x[1:-1] - x[:-2]
    dxp = x[2:] - x[1:-1]
    wm = -dxp / (dxm * (dxm + dxp))
    w0 = (dxp - dxm) / (dxm * dxp)
    wp = dxm / (dxp * (dxm + dxp))
    return wm, w0, wp


def fd2_interior_coeffs(x: np.ndarray):
    """Interior non-uniform second-derivative stencil weights (wm, w0, wp), each (N-2,).

    Reduces to the uniform (1/h^2, -2/h^2, 1/h^2) at equal spacing; exact for quadratics.
    """
    x = np.asarray(x, dtype=float)
    dxm = x[1:-1] - x[:-2]
    dxp = x[2:] - x[1:-1]
    denom = 0.5 * dxm * dxp * (dxm + dxp)
    wm = dxp / denom
    w0 = -(dxm + dxp) / denom
    wp = dxm / denom
    return wm, w0, wp
```

Export from `quantark/util/numerical/__init__.py` (add to the finite-difference import
line and `__all__`):

```bash
grep -n "fd1_nonuniform\|fd2_nonuniform\|finite_difference" quantark/util/numerical/__init__.py
```
Add `fd1_interior_coeffs, fd2_interior_coeffs` alongside the existing
`fd1_nonuniform, fd2_nonuniform` import and `__all__` entries.

Run: `PYTHONPATH=$PWD <main-venv>/bin/python -m pytest test/test_adi_concentrated_grid.py -q`
Expected: the four FD-coeff tests PASS.

- [ ] **Step 4: Add `grid_style` + concentrated grid construction to the core**

In `quantark/volmodels/adi_core.py`:

(a) Imports:

```python
from quantark.util.numerical import (
    solve_tridiag_batch, fd1_interior_coeffs, fd2_interior_coeffs,
)
from quantark.volmodels.slv.fokkerplanck.coordinates import concentrated_grid, z_extents
```

(b) `__init__` signature: add `grid_style="uniform"` after `v0_boundary`. Validate and set
`self._uniform = grid_style == "uniform"`. Replace the grid-construction block
(lines 58-71) so the uniform path is untouched and the concentrated path builds sinh grids:

```python
        if grid_style not in ("uniform", "concentrated"):
            raise ValidationError("grid_style must be 'uniform' or 'concentrated'")
        self._uniform = grid_style == "uniform"

        var_eff = max(self.theta, self.v0, 0.25 * self.sig_eff2, 0.04)
        x_width = 8.0 * np.sqrt(var_eff * max(T, 1e-12))
        grid_center = grid_spot if grid_spot is not None else s0
        x_center = float(np.log(max(grid_center, 1e-12)))
        self.x_min, self.x_max = x_center - x_width, x_center + x_width
        self.V_max = max(5.0 * self.theta, 0.5, 2.0 * self.v0)

        if self._uniform:
            self.X_grid = np.linspace(self.x_min, self.x_max, n_x)
            self.V_grid = np.linspace(0.0, self.V_max, n_v)
            self.dx = float(self.X_grid[1] - self.X_grid[0])
            self.dV = float(self.V_grid[1] - self.V_grid[0])
        else:
            # x concentrated around ln K (payoff kink); v concentrated around min(v0, theta),
            # with a CIR-quantile-based lower extent when vol-of-vol is live.
            xk = float(np.log(max(self.K, 1e-12)))
            xk = min(max(xk, self.x_min), self.x_max)
            self.X_grid = concentrated_grid(self.x_min, self.x_max, xk, n_x,
                                            concentration=max(0.25 * x_width, 1e-6))
            v_lo = 0.0
            if self.sig_eff > 0.0:
                t_probe = np.array([0.25 * T, 0.5 * T, T])
                try:
                    q_lo, q_hi = z_extents(params, float(eta), t_probe,
                                           cir_quantile=1e-5, v_floor=1e-8)
                    self.V_max = max(self.V_max, q_hi)
                except ValidationError:
                    pass  # keep the envelope V_max; degenerate CIR handled by uniform-style extent
            v_center = min(max(self.v0, 0.0), self.theta) if self.theta > 0 else self.v0
            v_center = min(max(v_center, v_lo), self.V_max)
            self.V_grid = concentrated_grid(v_lo, self.V_max, v_center, n_v,
                                            concentration=max(0.5 * self.V_max, 1e-6))
            # per-node stencil coefficients (interior) for both directions
            self._xx = fd2_interior_coeffs(self.X_grid)   # (wm, w0, wp) each (n_x-2,)
            self._x1 = fd1_interior_coeffs(self.X_grid)
            self._vv = fd2_interior_coeffs(self.V_grid)
            self._v1 = fd1_interior_coeffs(self.V_grid)
            self.dx = None  # guard: scalar spacing is undefined on a concentrated grid
            self.dV = None

        self.S_grid = np.exp(self.X_grid)
        self.S_max = float(self.S_grid[-1])
        self.dt = float(T / max(n_t, 1))
        self._S_int = self.S_grid[1:-1]
        self._ones_int = np.ones(self.N_S - 2)
```

> **Concentration constants** (`0.25*x_width`, `0.5*V_max`) are the initial Tavella-Randall
> scales; the Task-6 benchmark tunes them against the ≥4× node-reduction acceptance and
> they may be adjusted there before the (deferred) default flip.

(c) Guard the sparse path: SuperLU tridiag caching assumes constant coefficients per
direction; concentrated grids keep per-node coefficients but the matrices are still
tridiagonal, so the existing sparse assembly works — **but** `use_sparse` combined with
`concentrated` is out of scope for this task. Add to `__init__`:

```python
        self.use_sparse = bool(use_sparse) and self._constant_leverage and self._uniform
```
(This replaces the existing `self.use_sparse = ...` line; concentrated always uses the
batched-Thomas dense path.)

- [ ] **Step 5: Add the concentrated operator + tridiag branches**

In `_A1`, `_A2`, `_A0`, `_tri_S`, `_tri_V`, branch on `self._uniform`. Keep the existing
scalar-dx body as the `if self._uniform:` branch verbatim; add the `else:` concentrated
branch using the precomputed coefficients. Example for `_A2`:

```python
    def _A2(self, U):
        out = np.zeros_like(U)
        if self.N_S < 3 or self.N_V < 3:
            return out
        v_int = self.V_grid[1:-1]
        coef_d2 = 0.5 * self.sig_eff2 * v_int
        coef_d1 = self.kappa * (self.theta - v_int)
        if self._uniform:
            U_VV = (U[1:-1, 2:] - 2.0 * U[1:-1, 1:-1] + U[1:-1, :-2]) / (self.dV * self.dV)
            U_V = (U[1:-1, 2:] - U[1:-1, :-2]) / (2.0 * self.dV)
        else:
            wm2, w02, wp2 = self._vv
            wm1, w01, wp1 = self._v1
            U_VV = U[1:-1, :-2] * wm2 + U[1:-1, 1:-1] * w02 + U[1:-1, 2:] * wp2
            U_V = U[1:-1, :-2] * wm1 + U[1:-1, 1:-1] * w01 + U[1:-1, 2:] * wp1
        out[1:-1, 1:-1] = coef_d2 * U_VV + coef_d1 * U_V
        if self._degenerate_v0:
            dV0 = self.V_grid[1] - self.V_grid[0]
            out[1:-1, 0] = self.kappa * self.theta * (U[1:-1, 1] - U[1:-1, 0]) / dV0
        return out
```
(Note the degenerate-boundary `dV` now reads `V_grid[1]-V_grid[0]` so it works on both
grids — update the Task-4 `_A2`/`_tri_V` edits to use this local `dV0` too when merging.)

Apply the analogous `if self._uniform:` / `else:` split to `_A1` (x-direction `U_xx`,
`U_x` via `self._xx`, `self._x1`), and to `_A0` (mixed term: replace
`(U[2:,2:]-U[2:,:-2]-U[:-2,2:]+U[:-2,:-2])/(4 dx dV)` with the outer product of the x and
v first-derivative interior stencils):

```python
        # concentrated mixed term: U_xv = (fd1_x) applied then (fd1_v), interior nodes
        wxm, wx0, wxp = self._x1
        wvm, wv0, wvp = self._v1
        Ux = (U[:-2, :] * wxm[:, None] + U[1:-1, :] * wx0[:, None] + U[2:, :] * wxp[:, None])
        U_xv = (Ux[:, :-2] * wvm + Ux[:, 1:-1] * wv0 + Ux[:, 2:] * wvp)
        out[1:-1, 1:-1] = self.rho * self.sig_eff * Lv * U_xv
```

For the tridiag builders, replace the scalar-spacing coefficients with per-node ones in
the concentrated branch. `_tri_V` concentrated interior:

```python
        if self._uniform:
            v = np.maximum(self.V_grid, 1e-10)
            coef_d2 = 0.5 * self.sig_eff2 * v / (dV * dV)
            coef_d1 = self.kappa * (self.theta - v) / (2.0 * dV)
            a[1:-1] = -theta_loc * dt_step * (coef_d2[1:-1] - coef_d1[1:-1])
            b[1:-1] = 1.0 + theta_loc * dt_step * (2.0 * coef_d2[1:-1])
            c[1:-1] = -theta_loc * dt_step * (coef_d2[1:-1] + coef_d1[1:-1])
        else:
            v_int = np.maximum(self.V_grid[1:-1], 1e-10)
            d2 = 0.5 * self.sig_eff2 * v_int          # operator diffusion coeff (unscaled)
            d1 = self.kappa * (self.theta - v_int)    # operator convection coeff
            wm2, w02, wp2 = self._vv
            wm1, w01, wp1 = self._v1
            # per-node operator stencil L = d2*fd2 + d1*fd1, then implicit (I - theta*dt*L):
            sub_op = d2 * wm2 + d1 * wm1
            diag_op = d2 * w02 + d1 * w01             # w02 < 0 (fd2 center weight)
            sup_op = d2 * wp2 + d1 * wp1
            a[1:-1] = -theta_loc * dt_step * sub_op
            b[1:-1] = 1.0 - theta_loc * dt_step * diag_op   # MINUS: I - theta*dt*L
            c[1:-1] = -theta_loc * dt_step * sup_op
```
**Sign convention (Codex plan-gate [high]):** the implicit matrix is `I − θ·dt·L`, so every
diagonal is `1 − θ·dt·diag_op`. The uniform branch's `b = 1 + θ·dt·2c2` is the *same*
formula with its diag operator-coeff `−2c2` already folded in; the general branch keeps
`w02 < 0` un-folded, so the outer sign MUST stay minus. Apply the identical
`sub_op/diag_op/sup_op` → `(−θdt·sub_op, 1−θdt·diag_op, −θdt·sup_op)` pattern to `_tri_S`
(x-direction, per-node `self._xx`, `self._x1`; `_tri_S` builds `(N_V, N_S)` so the
x-coefficients broadcast across V-slices; the diagonal additionally carries `+ θ·dt·r`
from the WS-C1 implicit `−rU` reaction — i.e. `b = 1 − θdt·diag_op + θdt·r`). Keep the
concentrated V-row-0 (degenerate or Neumann) and row-(N-1) Neumann assignments using
`dV0 = V_grid[1]-V_grid[0]` (degenerate) or `c[0]=-1` (Neumann) exactly as Task 4.

> **Bit-identity guard:** the `if self._uniform:` branch bodies must be copied *verbatim*
> from the current code (same parenthesization) so uniform goldens do not move by a ULP.
> The Step-7 regression is the gate for this.

- [ ] **Step 6: Add the concentrated-grid convergence test**

Append to `test/test_adi_concentrated_grid.py`:

```python
from quantark.util.enum.engine_enums import ADIScheme
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.heston.pde_kernel import price_european_heston_pde
from quantark.volmodels.heston.analytical_kernel import heston_call_price


def test_grid_style_default_is_uniform():
    import inspect
    sig = inspect.signature(price_european_heston_pde)
    assert sig.parameters["grid_style"].default == "uniform"


def test_uniform_path_bit_identical_when_grid_style_uniform():
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.0
    params = HestonParams(kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04)
    p_a = price_european_heston_pde(s0, k, True, T, params, r, q, n_x=120, n_v=60, n_t=50)
    p_b = price_european_heston_pde(s0, k, True, T, params, r, q, n_x=120, n_v=60, n_t=50,
                                    grid_style="uniform")
    assert p_a == p_b  # exact


def test_concentrated_grid_equal_node_error_reduction():
    # Same node budget: concentrated should be materially closer to the analytical price.
    s0, k, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.0
    params = HestonParams(kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04)
    ref = heston_call_price(s0, k, T, params, r, q)
    common = dict(n_x=80, n_v=40, n_t=100, scheme=ADIScheme.CRAIG_SNEYD)
    e_uni = abs(price_european_heston_pde(s0, k, True, T, params, r, q,
                                          grid_style="uniform", **common) - ref)
    e_con = abs(price_european_heston_pde(s0, k, True, T, params, r, q,
                                          grid_style="concentrated", **common) - ref)
    assert e_con < e_uni  # equal-node error reduction (>= ; benchmark quantifies the factor)


def test_concentrated_tridiag_rows_match_uniform_on_a_uniform_grid():
    # Full-row equivalence (Codex plan-gate): on a UNIFORM grid the concentrated
    # coefficient path must reproduce the uniform scalar-dx tridiagonal rows (a,b,c),
    # not just the derivative weights. Proves the concentrated math (esp. the diagonal
    # sign) is correct. Closeness, not bit-identity — the branches differ by ULPs.
    from quantark.volmodels.adi_core import HestonSLVADICore
    params = HestonParams(kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7, v0=0.04)
    kw = dict(s0=100.0, strike=100.0, T=1.0, r=0.03, carry=0.0, params=params,
              n_x=40, n_v=30, n_t=50, leverage=None, eta=1.0)
    core_u = HestonSLVADICore(**kw, grid_style="uniform")
    core_c = HestonSLVADICore(**kw, grid_style="concentrated")
    # force the concentrated core onto the SAME uniform nodes so the rows are comparable
    core_c.X_grid = core_u.X_grid.copy(); core_c.V_grid = core_u.V_grid.copy()
    from quantark.util.numerical import fd1_interior_coeffs, fd2_interior_coeffs
    core_c._xx = fd2_interior_coeffs(core_c.X_grid); core_c._x1 = fd1_interior_coeffs(core_c.X_grid)
    core_c._vv = fd2_interior_coeffs(core_c.V_grid); core_c._v1 = fd1_interior_coeffs(core_c.V_grid)
    for builder, args in [("_tri_V", (core_u.dt, 1.0)), ("_tri_S", (core_u.dt, 1.0, 0.0))]:
        au, bu, cu = getattr(core_u, builder)(*args)
        ac, bc, cc = getattr(core_c, builder)(*args)
        assert np.allclose(au, ac, atol=1e-10) and np.allclose(bu, bc, atol=1e-10) \
            and np.allclose(cu, cc, atol=1e-10), f"{builder} rows diverge (sign/coeff bug)"
```
(`_tri_S` returns `(N_V, N_S)` arrays and `_tri_V` returns `(N_V,)` — `np.allclose`
handles both shapes; the concentrated core here keeps `self.dx/self.dV` as `None` but the
`_tri_*` builders read the uniform-branch scalars only under `if self._uniform`, so the
concentrated branch under test never dereferences them.)

- [ ] **Step 7: Thread `grid_style` through wrappers + run full regression**

Add `grid_style: str = "uniform"` to `price_european_heston_pde`,
`price_delta_gamma_heston_pde`, `price_european_slv_pde`, `price_delta_gamma_slv_pde`
(after `v0_boundary`) and pass into the `HestonSLVADICore(...)` constructions.

Run: `PYTHONPATH=$PWD <main-venv>/bin/python -m pytest test/test_adi_concentrated_grid.py -q`
Expected: PASS.
Run: `PYTHONPATH=$PWD <main-venv>/bin/python -m pytest test/ -k "pde or adi or heston or slv" -q`
Expected: PASS — uniform bit-identical (the `p_a == p_b` and WS-D1 1e-13 Heston≡SLV
gate must hold).

- [ ] **Step 8: Commit**

```bash
git add quantark/util/numerical/finite_difference.py quantark/util/numerical/__init__.py \
        quantark/volmodels/adi_core.py quantark/volmodels/heston/pde_kernel.py \
        quantark/volmodels/slv/slv_pde_kernel.py test/test_adi_concentrated_grid.py
git commit -m "feat(adi): opt-in concentrated (x,v) grids with non-uniform stencils (WS-C2)"
```

---

## Task 6: Benchmarks, validation scripts, and memory

**Files:**
- Create: `example/volmodels_phase5_qem_martingale.py`, `example/volmodels_phase5_qmc_convergence.py`,
  `example/volmodels_phase5_concentrated_grid_benchmark.py`,
  `example/volmodels_phase5_degenerate_boundary.py`, `example/volmodels_phase5_lv_rannacher.py`
- Modify: memory `project_volmodels_improvement_program.md` + `MEMORY.md`

**Interfaces:** none (scripts + docs).

- [ ] **Step 1: QE-M martingale benchmark script**

Create `example/volmodels_phase5_qem_martingale.py` printing, for σ∈{0.5,1.0},
ρ∈{−0.9,−0.5}, steps∈{4,8,24}, the discounted `E[S_T]/fwd − 1` (in stderr units) for
QUADEXP vs QUADEXP_M — the acceptance table for WS-C5.

- [ ] **Step 2: QMC convergence script**

Create `example/volmodels_phase5_qmc_convergence.py` printing RMSE-vs-paths for pseudo
vs Sobol on a European LV/Heston fixture (log-log slope) — the WS-D7 acceptance artifact.

- [ ] **Step 3: Concentrated-grid equal-accuracy benchmark**

Create `example/volmodels_phase5_concentrated_grid_benchmark.py` computing, on the
Hout-Foulon four-case set + the repo Feller-violated fixture, the node count each grid
style needs to reach a fixed error vs the analytical Heston price — records the node
reduction factor (acceptance target ≥4× or equal-node error reduction ≥5×). **Tune the
concentration constants** in `adi_core.py` here if ≥4× is not met at the initial scales;
re-run Task 5 tests after any tune.

- [ ] **Step 4: Degenerate-boundary + LV-Rannacher before/after scripts**

Create `example/volmodels_phase5_degenerate_boundary.py` (Feller-violated PDE error
Neumann vs degenerate vs analytical) and `example/volmodels_phase5_lv_rannacher.py`
(near-strike gamma profile with/without Rannacher; moved-golden before/after table).

- [ ] **Step 5: Run all scripts**

Run each with `PYTHONPATH=$PWD <main-venv>/bin/python example/volmodels_phase5_*.py`;
confirm the acceptance numbers (QE-M ≤3 stderr; QMC slope improvement; concentrated
node reduction; degenerate error down on Feller-violated, unchanged on Feller-satisfied;
LV gamma ripple down). If concentrated node reduction < 4× after tuning, **do not flip
any default and do not weaken the acceptance** — record the achieved factor honestly in
the script output and note it in memory as a bounded result (per exact-semantics).

- [ ] **Step 6: Full suite**

Run: `PYTHONPATH=$PWD <main-venv>/bin/python -m pytest -q`
Expected: all green; only WS-C7 LV goldens moved (deliberately, Task 3).

- [ ] **Step 7: Update memory**

Update `project_volmodels_improvement_program.md`: change the description to
"Phases 0-5 merged" and add a Phase-5 body paragraph recording WS-C2/C3/C5/C7/D7, the
default policy (C7 on, C2/C3/D7 opt-in), the benchmark outcomes (node-reduction factor,
QMC slope, QE-M bias removal, degenerate-boundary improvement), and any concentration
tuning. Update the `MEMORY.md` one-liner. Note the still-deferred items: the
sheared/eigenvector-aligned FP grid (WS-C4 true fix) and any C2/C3 default flips.

- [ ] **Step 8: Commit**

```bash
git add -f docs/superpowers/plans/2026-07-05-volmodels-phase5-grids-boundary-qem-rannacher-qmc.md
git add example/volmodels_phase5_*.py
git commit -m "docs(volmodels): Phase 5 benchmarks + validation scripts (C2/C3/C5/C7/D7)"
```
(Memory files live outside the repo and are written with the Write tool, not committed.)

---

## Self-Review

**Spec coverage:** WS-C2 (Task 5), WS-C3 (Task 4), WS-C5 (Task 1), WS-C7 (Task 3),
WS-D7 (Task 2), benchmarks/validation (Task 6). All five Phase-5 workstreams covered.

**Default policy (kickoff: spec-exact):** WS-C7 `rannacher=True` default (deliberate LV
golden move, Task 3 Step 6). WS-C2 `grid_style="uniform"`, WS-C3 `v0_boundary="neumann"`,
WS-D7 `sampler=None` — all opt-in, zero golden churn. C2/C3 default flips deferred to a
follow-up (bisectability), per spec open-question #2.

**Bit-identity gates:** Task 1 pins QUADEXP unchanged; Task 2 asserts `sampler=None`
bit-identical; Task 5 asserts `grid_style="uniform"` bit-identical + the WS-D1 1e-13
Heston≡SLV gate. These are the guardrails against silent golden drift.

**Type consistency:** `HestonSLVADICore` gains `v0_boundary` then `grid_style` (both
keyword, defaulted) — the Task-4 and Task-5 wrapper edits pass them positionally-after
`grid_spot`; both PDE wrapper families (`price_european_*`, `price_delta_gamma_*`) get
identical param additions. `fd1_interior_coeffs`/`fd2_interior_coeffs` return
`(wm, w0, wp)` tuples consumed identically in `_A1/_A2/_A0/_tri_S/_tri_V`.

**No-fallback compliance:** QE-M raises `NumericalError` when the MGF domain is violated
(no silent fallback); concentrated node-reduction shortfalls are recorded honestly, not
hidden by weakened acceptance; the uniform path is preserved, not approximated.
