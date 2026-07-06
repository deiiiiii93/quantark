# Barrier Pricing under Local-Vol / Heston / SLV — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline). Steps use `- [ ]` checkboxes.

**Goal:** Add `quantark` engines that price single-barrier options under Dupire Local Vol, Heston, and Heston-SLV by both Monte Carlo and PDE, validated 4 ways, then wire an up-and-out-call case into the MO lecture.

**Architecture:** Reuse the existing `volmodels` simulation/ADI kernels. A single shared `volmodels/barrier.py` centralizes barrier payoff/rebate/`pay_at_hit`/knock-in logic (MC) and knock-out boundary logic (PDE), so all three MC kernels and all three PDE kernels stay DRY and consistent. Thin equity engines wrap the kernels to price the existing `BarrierOption` product.

**Tech Stack:** NumPy/SciPy, `quantark.volmodels` (localvol/heston/slv kernels, `adi_core`), `quantark.asset.equity` engines, `quantark.util.numerical`.

## Global Constraints

- Honor `BarrierOption.pay_at_hit`: KO rebate PV = `rebate` at hit (`pay_at_hit=True`) else `rebate·DF(t_hit→T)`.
- Knock-in is rebate-correct: MC prices KI directly (payoff if breached else `rebate` at expiry); PDE uses `KI = Vanilla − KO(rebate=0) + rebate·NoTouch`. `KI+KO=Vanilla` asserted only at `rebate=0`.
- All 4 `BarrierType` (UP/DOWN × IN/OUT); `ObservationType.CONTINUOUS` and `DISCRETE` (via `observation_dates`).
- `quantark.util.numerical` for tolerances/protected math. No MC inside PDE. Canonical `quantark.*` imports.
- Barrier ≤ 0 or barrier == spot exactly → `ValidationError`. Barrier already breached at t=0 handled explicitly.
- Tests under `test/`, run `.venv/bin/python -m pytest -n0 <file>` while developing.

## Kernel/product facts (verified)
- `price_european_lv_mc(s0, strike, is_call, lv_surface, step_dt, r_fwd, carry_fwd, num_paths, seed, use_antithetic, return_stderr)`.
- `price_european_heston_mc(s0, strike, is_call, params, step_dt, r_fwd, carry_fwd, scheme, num_paths, seed, ...)`.
- `price_european_slv_mc(s0, strike, is_call, params, lv_surface, step_dt, r_fwd, carry_fwd, disc_factor, eta, num_paths, num_bins, ..., leverage_surface=None)`.
- `BarrierType.{UP_IN,UP_OUT,DOWN_IN,DOWN_OUT}` with `.is_up/.is_down/.is_knock_in/.is_knock_out`. `ObservationType.{CONTINUOUS,DISCRETE}`.
- `BarrierOption(strike, maturity, option_type, barrier, barrier_type, rebate, pay_at_hit, observation_type, observation_dates)`, helpers `is_knock_out/is_knock_in/is_up_barrier/is_down_barrier/is_barrier_hit(spot)`.
- BSM anchor: `BarrierAnalyticalEngine` (prices KO both `pay_at_hit` settings + KI no-touch rebate, continuous).
- European engine wrappers thread per-step curves via `forward_rates_on_grid` / `forward_carry_on_grid` and check the product type in `_price_with_surface`; mirror that structure.

---

## PHASE 1 — Barrier Monte Carlo

### Task 1: Shared barrier core `volmodels/barrier.py`

**Files:** Create `quantark/volmodels/barrier.py`; Test `test/volmodels/test_barrier_core.py`.

**Interfaces — Produces:**
- `BarrierSpec` dataclass: `is_up: bool, is_out: bool, is_call: bool, barrier: float, strike: float, rebate: float, pay_at_hit: bool`.
- `mc_barrier_cashflows(terminal_s, breached_mask, hit_step_dt_cumT, spec, disc_T) -> np.ndarray`
  returns per-path **present value** (discounted) cashflows, honoring KO/KI + rebate + `pay_at_hit`.
- `monitored_breach(run_min, run_max, samples, spec) -> (breached: bool[np], first_hit_idx: int[np])`
  given continuous running extrema OR discrete monitoring samples.
- `validate_barrier(spec, s0)` → raises `ValidationError` on `barrier<=0` or `barrier==s0`.

- [ ] **Step 1: Failing test** — payoff logic on hand-built inputs.

```python
# test/volmodels/test_barrier_core.py
import numpy as np, pytest
from quantark.volmodels.barrier import BarrierSpec, mc_barrier_cashflows, monitored_breach, validate_barrier
from quantark.util.exceptions import ValidationError

def test_up_out_call_knocks_out_with_rebate():
    spec = BarrierSpec(is_up=True, is_out=True, is_call=True, barrier=120., strike=100., rebate=5., pay_at_hit=False)
    term = np.array([130., 110.])          # path0 breaches (up>=120), path1 does not
    breached = np.array([True, False]); hitT = np.array([0.5, 0.0])
    pv = mc_barrier_cashflows(term, breached, hitT, spec, disc_T=lambda t: np.exp(-0.02*t))
    # path0: knocked out -> rebate 5 discounted to T=1; path1: alive call payoff (110-100)=10 disc to T
    assert pv[0] == pytest.approx(5*np.exp(-0.02*1.0), rel=1e-9)
    assert pv[1] == pytest.approx(10*np.exp(-0.02*1.0), rel=1e-9)

def test_pay_at_hit_discounts_to_hit_time():
    spec = BarrierSpec(True, True, True, 120., 100., 5., pay_at_hit=True)
    pv = mc_barrier_cashflows(np.array([130.]), np.array([True]), np.array([0.5]), spec, lambda t: np.exp(-0.02*t))
    assert pv[0] == pytest.approx(5*np.exp(-0.02*0.5), rel=1e-9)

def test_validate_rejects_barrier_at_spot():
    with pytest.raises(ValidationError):
        validate_barrier(BarrierSpec(True, True, True, 100., 100., 0., False), s0=100.)
```

- [ ] **Step 2:** Run → FAIL. `.venv/bin/python -m pytest -n0 test/volmodels/test_barrier_core.py`

- [ ] **Step 3:** Implement `barrier.py`. Full cashflow logic (the load-bearing correctness core):

```python
from dataclasses import dataclass
import numpy as np
from quantark.util.exceptions import ValidationError

@dataclass
class BarrierSpec:
    is_up: bool; is_out: bool; is_call: bool
    barrier: float; strike: float; rebate: float; pay_at_hit: bool

def validate_barrier(spec: "BarrierSpec", s0: float) -> None:
    if not np.isfinite(spec.barrier) or spec.barrier <= 0:
        raise ValidationError("barrier must be positive and finite")
    if spec.barrier == s0:
        raise ValidationError("barrier equal to spot is degenerate")

def _vanilla(term, spec):
    return np.maximum(term - spec.strike, 0.0) if spec.is_call else np.maximum(spec.strike - term, 0.0)

def mc_barrier_cashflows(terminal_s, breached, hit_cumT, spec, disc_T):
    """Per-path PV. disc_T(t)->DF from 0 to t. T is the last time (=hit_cumT max horizon)."""
    T = float(np.max(hit_cumT[breached])) if np.any(breached) else None
    # infer horizon: caller passes hit_cumT for breached paths and T via the terminal disc; use maturity
    payoff = _vanilla(terminal_s, spec)
    dfT = disc_T(_MATURITY.value)  # set by caller via set_maturity; see note
    pv = np.empty(terminal_s.shape[0])
    surv = ~breached
    if spec.is_out:
        pv[surv] = payoff[surv] * dfT                       # survived KO -> option pays
        reb_df = disc_T(hit_cumT[breached]) if spec.pay_at_hit else dfT
        pv[breached] = spec.rebate * reb_df                 # knocked out -> rebate
    else:  # knock-in
        pv[breached] = payoff[breached] * dfT               # knocked in -> option pays at T
        pv[surv] = spec.rebate * dfT                        # never knocked in -> rebate at expiry
    return pv
```

Note: pass maturity + a vectorized `disc_T` from the kernel (a closure over `r_fwd`); implement `disc_T` to accept scalar or array and return `DF(0→t)`. Add `monitored_breach` computing breach from continuous `run_min/run_max` vs `barrier` (up: `run_max>=B`; down: `run_min<=B`) or from discrete `samples` (any obs beyond barrier), returning `first_hit_idx` for `pay_at_hit`.

- [ ] **Step 4:** Run → PASS. **Step 5:** Commit `feat(volmodels): shared barrier payoff/monitoring core`.

### Task 2: LV barrier MC kernel

**Files:** Modify `quantark/volmodels/localvol/mc_kernel.py`; Test `test/volmodels/test_barrier_lv_mc.py`.

**Produces:** `price_barrier_lv_mc(s0, strike, is_call, lv_surface, step_dt, r_fwd, carry_fwd, barrier, is_up, is_out, rebate, pay_at_hit, observe_idx=None, num_paths, seed, use_antithetic, return_stderr)`. `observe_idx=None` → continuous (running extremum); else discrete indices into the step grid.

- [ ] **Step 1: Failing tests** — European limit + BSM-flat analytical + in-out parity(rebate=0).

```python
# test/volmodels/test_barrier_lv_mc.py  (key cases)
import numpy as np
from quantark.volmodels.localvol.mc_kernel import price_barrier_lv_mc, price_european_lv_mc
# flat LV surface helper builds a GridVolSurface->LocalVolSurface at 20% (see conftest/util)
def test_barrier_far_reproduces_european(flat_lv, grid):
    p_bar = price_barrier_lv_mc(100.,100.,True, flat_lv, *grid, barrier=1e6, is_up=True, is_out=True,
                                rebate=0., pay_at_hit=False, num_paths=60_000, seed=1)
    p_eur = price_european_lv_mc(100.,100.,True, flat_lv, *grid, num_paths=60_000, seed=1)
    assert abs(p_bar - p_eur) < 0.05
def test_up_out_call_matches_reiner_rubinstein(flat_lv, grid, rr_uo_call):
    p = price_barrier_lv_mc(100.,100.,True, flat_lv, *grid, barrier=130., is_up=True, is_out=True,
                            rebate=0., pay_at_hit=False, num_paths=200_000, seed=7)
    assert abs(p - rr_uo_call) < 3*se  # within 3 standard errors of analytical
```

- [ ] **Step 2:** FAIL. **Step 3:** Refactor `_simulate_lv` (or add one) to optionally record running min/max and monitoring-index spots (O(1)/O(#obs) memory), returning `(terminal, run_min, run_max, samples)`; add `price_barrier_lv_mc` calling `monitored_breach` + `mc_barrier_cashflows`. European pricer keeps using terminal only. **Step 4:** PASS. **Step 5:** Commit.

### Task 3: Heston barrier MC kernel
**Files:** Modify `quantark/volmodels/heston/mc_kernel.py`; Test `test/volmodels/test_barrier_heston_mc.py`.
**Produces:** `price_barrier_heston_mc(... params, scheme, barrier, is_up, is_out, rebate, pay_at_hit, observe_idx, ...)`.
- [ ] Steps mirror Task 2: refactor `_simulate_terminal_spot`→`_simulate_heston` to emit extremum/samples; add barrier pricer. Tests: European limit, BSM-flat analytical (Heston with `v0=θ`, `σ=1e-6`, `ρ=0` → GBM), in-out parity(rebate=0). Commit.

### Task 4: SLV barrier MC kernel
**Files:** Modify `quantark/volmodels/slv/slv_mc_kernel.py`; Test `test/volmodels/test_barrier_slv_mc.py`.
**Produces:** `price_barrier_slv_mc(... params, lv_surface, ..., leverage_surface, barrier, is_up, is_out, rebate, pay_at_hit, observe_idx, ...)`.
- [ ] `_simulate_slv` already loops per step and returns `np.exp(log_s)`; add optional running-extremum + sample recording, return them; add barrier pricer. Tests: European limit (barrier far ⇒ `price_european_slv_mc`), BSM-flat (leverage≡1 + `σ=1e-6` ⇒ GBM), in-out parity(rebate=0). Commit.

### Task 5: Three equity MC barrier engines
**Files:** Create `quantark/asset/equity/engine/mc/{local_vol,heston,heston_slv}_barrier_mc_engine.py`; modify `mc/__init__.py`; Test `test/test_barrier_vol_mc_engines.py`.
**Produces:** `LocalVolBarrierMCEngine(params, local_vol_surface=None)`, `HestonBarrierMCEngine(model_params, scheme, params)`, `HestonSLVBarrierMCEngine(model_params, leverage_surface, eta, params)`; each `price(BarrierOption, env)`.

**Interfaces — Consumes:** `forward_rates_on_grid`, `forward_carry_on_grid`, the Task 2–4 kernels, `BarrierType`/`ObservationType`.

- [ ] **Step 1: Failing test** — a `LocalVolBarrierMCEngine` prices an `UP_OUT` call and matches `price_barrier_lv_mc` directly; rejects non-`BarrierOption`.

```python
def test_lv_barrier_engine_prices_up_out_call(env_flat, uo_call):
    from quantark.asset.equity.engine.mc import LocalVolBarrierMCEngine
    from quantark.asset.equity.param import MCParams
    px = LocalVolBarrierMCEngine(MCParams(num_paths=80_000, time_steps=100, seed=3)).price(uo_call, env_flat)
    assert px > 0
```

- [ ] **Step 2:** FAIL. **Step 3:** Implement wrappers. Template (LV; others mirror with their kernel + model inputs):

```python
class LocalVolBarrierMCEngine(BaseEngine):
    engine_type = EngineType.MC
    def __init__(self, params=None, local_vol_surface=None):
        super().__init__(params or MCParams()); self._prebuilt = local_vol_surface
    def price(self, product, env):
        if not isinstance(product, BarrierOption):
            raise PricingError("LocalVolBarrierMCEngine supports BarrierOption only")
        T = float(product.get_maturity(env)); n = int(self.params.time_steps)
        t_grid = np.linspace(0., T, n+1)
        r_fwd = forward_rates_on_grid(env.rate_curve, t_grid); carry_fwd = forward_carry_on_grid(env.get_div_yield, t_grid)
        lv = self._prebuilt or build_dupire_local_vol(env.vol_surface, spot=env.spot, rate_curve=env.rate_curve, div_yield=env.get_div_yield)
        observe_idx = _observation_indices(product, t_grid)   # None if continuous; else np indices
        unit = price_barrier_lv_mc(float(env.spot), float(product.strike), product.option_type==OptionType.CALL,
            lv, np.diff(t_grid), r_fwd, carry_fwd, barrier=float(product.barrier),
            is_up=product.is_up_barrier, is_out=product.is_knock_out, rebate=float(product.rebate),
            pay_at_hit=product.pay_at_hit, observe_idx=observe_idx,
            num_paths=self.params.num_paths, seed=self.params.seed, use_antithetic=self.params.use_antithetic)
        return unit * float(getattr(product, "contract_multiplier", 1.0))
```

Add a shared `_observation_indices(product, t_grid)` helper (module-level, reused by all six engines). Register the three in `mc/__init__.py`. **Step 4:** PASS. **Step 5:** Commit.

---

## PHASE 2 — Barrier PDE

### Task 6: LV 1D barrier PDE kernel + engine
**Files:** Modify `quantark/volmodels/localvol/pde_kernel.py`; Create `quantark/asset/equity/engine/pde/local_vol_barrier_pde_solver.py`; modify `pde/__init__.py`; Test `test/volmodels/test_barrier_lv_pde.py`, `test/test_barrier_vol_pde_engines.py`.
**Produces:** `price_barrier_lv_pde(s0, strike, is_call, lv_surface, step_dt, r_fwd, carry_fwd, barrier, is_up, is_out, rebate, pay_at_hit, observe_steps=None, n_s, s_max, theta)`; `LocalVolBarrierPDESolver(params, local_vol_surface=None)`.

- [ ] **Step 1: Failing tests** — European limit, BSM-flat analytical (continuous), MC↔PDE agreement.
- [ ] **Step 3:** In `_solve_lv_pde`, after each CN step (continuous) or at `observe_steps` (discrete), impose the KO Dirichlet condition: for `is_up` set `V[s_grid >= barrier] = rebate_boundary(t)`; for down, `V[s_grid <= barrier]`. `rebate_boundary(t) = rebate` if `pay_at_hit` else `rebate * DF(t→T)`. KI via `Vanilla − KO(rebate=0) + rebate·NoTouch`, NoTouch = same solve with payoff≡0, unit rebate, `pay_at_hit=False`. Snap discrete `observe_steps` to grid; validate. **Step 4/5:** PASS + commit.

### Task 7: Heston 2D barrier ADI kernel + engine
**Files:** Modify `quantark/volmodels/heston/pde_kernel.py`; Create `.../pde/heston_barrier_pde_solver.py`; Test `test/volmodels/test_barrier_heston_pde.py`.
**Produces:** `price_barrier_heston_pde(..., barrier, is_up, is_out, rebate, pay_at_hit, observe_steps, scheme, n_x, n_v, n_t)`; `HestonBarrierPDESolver(model_params, leverage_surface=None?, scheme, n_x, n_v, n_t)` (no leverage for pure Heston).

- [ ] **Step 3 (the hard part):** Reuse `adi_core` on the full `(x=lnS, v)` grid. Between ADI steps (continuous) or at `observe_steps` (discrete), inject the barrier: for `is_up`, set `V[x_index where exp(x) >= barrier, :] = rebate_boundary(t)` across **all v** (whole x-columns, never partial cells — preserves the tridiagonal structure per the spec); down analogous. Keep the existing Dirichlet far-field edges. KI via the same `Vanilla − KO(0) + rebate·NoTouch` decomposition (NoTouch = zero terminal payoff, unit rebate). Terminal condition = vanilla payoff on the grid.

- [ ] **Tests:** European limit (`barrier→∞` ⇒ `price_european_heston_pde`), BSM-flat analytical (`v0=θ,σ=1e-6,ρ=0`), MC↔PDE agreement vs Task 3 within combined tol. Commit.

### Task 8: SLV 2D barrier ADI kernel + engine
**Files:** Modify `quantark/volmodels/slv/slv_pde_kernel.py`; Create `.../pde/heston_slv_barrier_pde_solver.py`; Test `test/volmodels/test_barrier_slv_pde.py`.
**Produces:** `price_barrier_slv_pde(..., leverage_surface, eta, barrier, ..., n_x, n_v, n_t)`; `HestonSLVBarrierPDESolver(model_params, leverage_surface, eta, scheme, n_x, n_v, n_t)`.
- [ ] Same barrier-injection wrapper as Task 7 but over the SLV ADI operators (leverage-scaled diffusion). Tests: European limit vs `price_european_slv_pde`, BSM-flat (leverage≡1 + `σ=1e-6`), MC↔PDE vs Task 4. Register all PDE engines in `pde/__init__.py`. Commit.

---

## PHASE 3 — MO consumer + lecture

### Task 9: `07_barrier_exotic.py` + `_mo_common` helper
**Files:** Modify `example/mo_volmodels/_mo_common.py` (+`build_barrier_models`), Create `example/mo_volmodels/07_barrier_exotic.py`; Test `test/mo_volmodels/test_stage07_barrier.py`.
**Produces:** up-and-out call (K≈forward of a chosen mid expiry, B≈110%·spot, daily discrete, rebate 0) priced MC **and** PDE under BSM/LV/Heston/SLV; writes `data/mo_barrier_{tag}.json` (`{model: {mc, mc_se, pde}}`) + a divergence bar chart `data/plots/07_barrier_{tag}.png`.

- [ ] **Step 1: Failing test** (sample tag): runs `02–05` then `07`; asserts `mo_barrier_sample.json` has all four models with finite `mc` and `pde` (where available), BSM `mc≈pde` within 3·se+tol, LV `mc≈pde`, and Heston≠LV (non-trivial model spread).
- [ ] **Step 3:** `build_barrier_models(surface_json, calib)` returns the six configured engines from the calibrated `LocalVolSurface`/`HestonParams`/`LeverageSurface` (reuse `build_env`, `build_dupire_local_vol` with the opt-in `--vol-floor`). Stage 07: construct the `BarrierOption`, price each cell, cross-validate, emit JSON + bar chart. **Step 4/5:** PASS + commit.

### Task 10: Lecture section + smoke test
**Files:** Modify `example/mo_volmodels/06_lecture.py`, `example/mo_volmodels/README.md`, `test/mo_volmodels/test_suite_smoke.py`.
- [ ] Insert lecture section **"Exotics: where models diverge"** after the SLV section (renumber "Model comparison"→next, "Reproducing"→last): the up-and-out payoff, the MC+PDE agreement table (cross-validation), the model-divergence bar chart, and the punchline that all four fit the same vanilla smile yet disagree on the barrier because it depends on forward-vol dynamics. Data-driven from `mo_barrier_{tag}.json`. Add stage 07 to the smoke test's stage list. README: document stage 07 + the new engines. **Step 4:** full `test/mo_volmodels/` green + commit.

---

## Self-Review

**Spec coverage:** §2 rebate/`pay_at_hit` → Task 1 (`mc_barrier_cashflows`) + Tasks 6–8 (KO boundary); KI no-touch → Task 1 (MC direct) + Tasks 6–8 (decomposition); §2 MC kernels → Tasks 2–4; engines → Task 5; §2 PDE 1D/2D → Tasks 6–8; §4 four cross-checks → tests in every kernel/engine task; §7 phasing → Phases 1/2/3; MO consumer → Tasks 9–10.

**Placeholder scan:** the `_MATURITY.value` shim in Task 1 Step 3 is illustrative — implement `disc_T`/maturity as explicit kernel-passed args (documented in the Note), not a global. No other placeholders.

**Type consistency:** `BarrierSpec(is_up,is_out,is_call,barrier,strike,rebate,pay_at_hit)` used identically across MC kernels and the core; engines pass `is_up=product.is_up_barrier, is_out=product.is_knock_out`. `_observation_indices(product, t_grid)` shared by all six engines. `price_barrier_*` signatures share the `(barrier, is_up, is_out, rebate, pay_at_hit, observe_idx|observe_steps)` block.

**Risk note:** the ADI barrier injection (Tasks 7–8) is the highest-risk piece — mitigated by the European-limit + BSM-flat analytical + MC↔PDE triangulation, and by overriding only whole x-columns (never partial cells) to preserve the tridiagonal solves.
