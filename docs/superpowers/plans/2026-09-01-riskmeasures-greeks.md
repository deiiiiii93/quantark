# Riskmeasures Greeks Refactor + Higher-Order Greeks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 2,497-line `GreeksCalculator` into a facade + registry package and add charm, color, speed, zomma, vega_theta, dividend_volga, gamma_theta with dual-clock (1D/1TD) theta support — preserving every incumbent number bit-for-bit.

**Architecture:** Facade class keeps the full public API; a `GreekDef` registry becomes the single source for validation/aliases/defaults/dispatch; bucketed coordinate families move to `bucketed_coordinates/` modules as pure code motion; new greeks are bump-based (any engine) + closed-form BS (vanillas) with the analytical forms as test oracles.

**Tech Stack:** Python, scipy.stats, pytest. Test invocation shadows the editable install: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/...`

**Spec:** `docs/superpowers/specs/2026-09-01-riskmeasures-greeks-design.md`

## Global Constraints

- Branch `worktree-riskmeasures-greeks`; every task ends in a commit; docs need `git add -f`.
- Public API frozen: `GreeksCalculator` module path, class name, every public method signature (§3.1).
- Incumbent numbers bit-for-bit: default set `{price, delta, gamma, vega, theta, rho, dividend_rho, convexity_theta, r_theta, q_theta}`; auto-routing analytical set for existing names `{price, delta, gamma, vega, theta, rho}`; bare theta-family names keep estimate `/365` scaling verbatim (§3.2).
- `calculate_analytical_greeks()` default key set unchanged; extended keys only via new `greeks=` param (§3.3).
- Fingerprint gate after Tasks 1–4 and 8: same-machine byte-identical JSON (scratchpad script, never committed as golden).
- No MC engines imported anywhere in riskmeasures (deterministic module).
- Test files touched by refactor gates: `test_point_greeks.py`, `test_greeks_mode_and_engine_type.py`, `test_greeks_bump_config.py`, `test_greeks_theta_schedule.py`, `test_rate_keyrate_buckets.py`, `test_equity_futures_delta_buckets.py`, `test_cash_greeks_report.py`.

---

### Task 0: Baseline + fingerprint harness

**Files:**
- Create: `<scratchpad>/greeks_fingerprint.py` (session scratchpad, NOT committed)
- Create: `<scratchpad>/fp_baseline.json`

**Interfaces:**
- Produces: `run_fingerprint(out_path)` — dumps a dict of scenario-name → greeks dict (floats via `repr`) for byte comparison.

- [ ] **Step 1: Verify the gate suite is green on the untouched branch**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_point_greeks.py test/test_greeks_mode_and_engine_type.py test/test_greeks_bump_config.py test/test_greeks_theta_schedule.py test/test_rate_keyrate_buckets.py test/test_equity_futures_delta_buckets.py test/test_cash_greeks_report.py -q`
Expected: all pass.

- [ ] **Step 2: Write the fingerprint script**

Scenario matrix (all currently-working requests only):
- vanilla call + put (`EuropeanVanillaOption(strike=100, maturity=1.0)`, env from `test_point_greeks._build_env` pattern) × `BlackScholesEngine`: `calculate()` default, `calculate(method="analytical")`, `calculate(greeks=["vanna","volga","delta_q"])`, each public `calculate_numerical_*` method individually.
- Same product × `PDEEngine(PDEParams(spot_steps=101, time_steps=50))` with `greeks_mode=BUMP` and `ENGINE`: default set.
- Business-day env (`test_greeks_theta_schedule._business_day_pricing_env` pattern: CHINA_SSE calendar, `bus_days_in_year=244`, valuation 2026-06-26) × BS engine: theta + default set.
- Bucketed: `calculate_bucketed_greeks` with default request on a `TermStructureVolSurface` env (VOL_TENOR_VEGA + CARRY_RHOQ path).
Serialize with `repr(float)`; `json.dump(..., sort_keys=True, indent=1)`.

- [ ] **Step 3: Capture baseline**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python <scratchpad>/greeks_fingerprint.py <scratchpad>/fp_baseline.json`

- [ ] **Step 4: No commit** (scratchpad only). Record baseline SHA of greeks_calculator.py in the JSON header for provenance.

---

### Task 1 (R1a): Extract `greeks/bump_envs.py`

**Files:**
- Create: `quantark/asset/equity/riskmeasures/greeks/__init__.py`, `greeks/bump_envs.py`
- Modify: `greeks_calculator.py`

**Interfaces:**
- Produces module functions (bodies moved verbatim, `self` dropped):
  - `ensure_base_price(product, pricing_env, engine, base_price) -> float`
  - `resolve_bump_engine(product, pricing_env, engine) -> BaseEngine`
  - `calculate_sensitivity(base_price, price_up, price_down=None, bump=1.0, scale=1.0, mode="central") -> float`
  - `spot_bumped_prices(product, pricing_env, engine, bump, base_price=None, reuse=None) -> tuple[float, float, float]`
  - `build_vol_bumped_env(pricing_env, product, current_vol, vol_bump, *, direction) -> PricingEnvironment`
  - `build_div_bumped_env(pricing_env, product, current_div, div_bump, *, direction) -> PricingEnvironment`
  - `advance_theta_bump(pricing_env, time_bump_days, time_bump_mode) -> tuple[datetime, float, str]`
  - `resolve_theta_bump_mode(pricing_env, time_bump_mode) -> str`

- [ ] **Step 1:** Move the eight bodies; facade keeps same-named private methods as one-line delegates (`def _ensure_base_price(self, ...): return bump_envs.ensure_base_price(...)`) because tests call privates (`calc._advance_theta_bump` in test_greeks_theta_schedule.py:238).
- [ ] **Step 2:** Gate: pytest suite from Task 0 Step 1 + fingerprint rerun byte-identical vs baseline (`cmp fp_baseline.json fp_r1a.json`).
- [ ] **Step 3:** Commit `refactor(riskmeasures): extract bump-env helpers to greeks/bump_envs`.

---

### Task 2 (R1b): Extract `bucketed_coordinates/`

**Files:**
- Create: `bucketed_coordinates/__init__.py`, `rate_keyrate.py`, `futures_delta.py`, `carry_rhoq.py`, `vol_tenor_vega.py`, `vol_model.py`
- Modify: `greeks_calculator.py`

**Interfaces:** each module exposes `calculate_points(calc, product, pricing_env, engine, request, mode) -> List[BucketedGreekPoint]` where `calc` is the GreeksCalculator (functions call `calc._ensure_base_price`, `calc._resolve_bump_engine`, `calc._calculate_sensitivity`, `calc._bump_config` exactly as the methods did via `self`). Family-private helpers move with their family:
- rate_keyrate.py ← `_calculate_bucketed_rate_keyrate_points`
- futures_delta.py ← `_calculate_bucketed_futures_delta_points`, `_calculate_futures_delta_one_sided_points`, `_calculate_futures_delta_central_points`, and the body of public `calculate_futures_delta_buckets` as `calculate_futures_delta_buckets(calc, ...)`
- carry_rhoq.py ← `_calculate_bucketed_carry_rhoq_points`, `_calculate_futures_rhoq_one_sided_points`, `_calculate_futures_rhoq_central_points`, `_calculate_generic_carry_rhoq_points`, `_calculate_node_aligned_carry_rhoq_points`, `_bucket_label`, `_carry_rhoq_point`, `_rhoq_bucket_row`, body of `calculate_futures_rhoq_buckets`
- vol_tenor_vega.py ← `_calculate_bucketed_vol_tenor_vega_points`, `_calculate_node_aligned_vol_vega_points`, `_bump_term_vol_node`, `_vol_tenor_vega_point`
- vol_model.py ← `_calculate_bucketed_vol_model_points`, `_vol_risk_result_to_bucketed_points`

- [ ] **Step 1:** Move bodies verbatim (`self` → `calc`); facade dispatch in `calculate_bucketed_greeks` calls module functions; public `calculate_futures_delta_buckets`/`calculate_futures_rhoq_buckets` become delegates.
- [ ] **Step 2:** Gate: suite + fingerprint byte-identical.
- [ ] **Step 3:** Commit `refactor(riskmeasures): move bucketed coordinate families to bucketed_coordinates/`.

---

### Task 3 (R1c): Extract `greeks/analytical.py` + `greeks/theta_decomposition.py`

**Files:**
- Create: `greeks/analytical.py`, `greeks/theta_decomposition.py`
- Modify: `greeks_calculator.py`

**Interfaces:**
- `analytical.calculate_analytical_greeks(product, pricing_env, price=None) -> Dict[str, float]` (body moved verbatim incl. `_greeks_at_expiry` as module function `greeks_at_expiry`)
- `theta_decomposition.estimate_theta_components(theta, rho, dividend_rho, r, q, T, rate_bump=0.01, dividend_bump=0.01) -> Dict[str, float]`
- `theta_decomposition.exact_theta_components(calc, product, pricing_env, engine, base_price=None, time_bump_days=None) -> Dict[str, float]` (body of `_calculate_numerical_theta_components`)
- Facade keeps `calculate_analytical_greeks`, `estimate_theta_components`, `_calculate_numerical_theta_components`, `_greeks_at_expiry` as delegates.

- [ ] **Step 1:** Move; delegate; keep imports (`scipy.stats`, `is_zero`, day-count) in the new modules.
- [ ] **Step 2:** Gate: suite + fingerprint byte-identical.
- [ ] **Step 3:** Commit `refactor(riskmeasures): extract analytical + theta decomposition modules`.

---

### Task 4 (R2): Registry + `greeks/numerical.py`

**Files:**
- Create: `greeks/registry.py`, `greeks/numerical.py`
- Modify: `greeks_calculator.py`
- Test: `test/test_greeks_registry.py` (new)

**Interfaces:**
```python
# greeks/registry.py
@dataclass(frozen=True)
class GreekDef:
    name: str
    aliases: tuple = ()
    analytical_auto: bool = False   # in the auto-routing analytical set
    default: bool = False           # in the greeks=None numerical default set
    linear_value: float = 0.0
    supports_clock: bool = False
    requires: tuple = ()            # canonical names this derives from

REGISTRY: dict[str, GreekDef]       # canonical name -> def (Task 4 registers only the incumbent names)
ALIASES: dict[str, str]             # built from REGISTRY
DEFAULT_SET: frozenset[str]
ANALYTICAL_AUTO_SET: frozenset[str]

@dataclass(frozen=True)
class GreekRequest:
    key: str          # output key as requested, e.g. "theta_1td"
    canonical: str    # registry name, e.g. "theta"
    clock: str | None # None | "1d" | "1td"   (always None until Task 6)

def normalize_greeks(greeks: Sequence | None) -> set[GreekRequest] | None
```
Incumbent entries (values copied from current code): price, delta, gamma, vega, theta, rho, dividend_rho (aliases rhoq/div_rho/dividendrho), vanna, volga, delta_q (aliases deltaq/deltadq/d_delta_d_q/d_delta_dq), charm, color, convexity_theta, r_theta, q_theta. `default=True` for the ten default names; `analytical_auto=True` for the six; `linear_value=1.0` for delta, else 0.0; theta family `supports_clock=True` (parsing still off). charm/color have no numerical fn **yet** — Task 4 keeps their incumbent silent-miss behavior so the fingerprint stays byte-identical; Task 6 implements them (the completeness test flips from xfail to pass there).

`greeks/numerical.py`: per-greek functions with the exact bodies of the facade's `calculate_numerical_*` methods, signature `(calc, product, pricing_env, engine, ...same kwargs...)`; facade public methods delegate. `calculate_numerical_greeks` stays on the facade with its **exact current order** (delta/gamma pre-pass → vega → volga → vanna → delta_q → theta → rho → dividend_rho → theta-components block) but membership/aliases/defaults consult the registry; `_normalize_greeks` becomes a shim returning `{req.key for req in normalize_greeks(...)}`.

- [ ] **Step 1: Failing test** — `test/test_greeks_registry.py`:
```python
import pytest
from quantark.asset.equity.riskmeasures.greeks.registry import (
    ALIASES, DEFAULT_SET, REGISTRY, normalize_greeks)

def test_registry_matches_incumbent_surface():
    assert DEFAULT_SET == {"price","delta","gamma","vega","theta","rho",
        "dividend_rho","convexity_theta","r_theta","q_theta"}
    assert ALIASES["deltadq"] == "delta_q"
    assert ALIASES["rhoq"] == "dividend_rho"

def test_normalize_rejects_unknown():
    from quantark.util.exceptions import ValidationError
    with pytest.raises(ValidationError):
        normalize_greeks(["not_a_greek"])
```
- [ ] **Step 2:** Run → FAIL (module missing). Implement registry + numerical extraction + rewire. Run → PASS.
- [ ] **Step 3:** Gate: full suite + fingerprint byte-identical (this is the delicate gate — identical call order guarantees identical engine invocation sequence).
- [ ] **Step 4:** Commit `refactor(riskmeasures): GreekDef registry as single validation/dispatch source`.

---

### Task 5 (E1a): speed, zomma, dividend_volga (numerical)

**Files:**
- Modify: `greeks/numerical.py`, `greeks/registry.py`, facade, `quantark/util/enum/greeks_enums.py`
- Test: `test/test_higher_order_greeks.py` (new)

**Interfaces (facade delegates added):**
- `calculate_numerical_speed(product, pricing_env, engine, base_price=None, bump=None) -> float`
- `calculate_numerical_zomma(product, pricing_env, engine, base_price=None, vol_bump=None) -> float`
- `calculate_numerical_dividend_volga(product, pricing_env, engine, base_price=None, div_bump=None) -> float`

Implementations (in numerical.py, following existing helper style):
```python
def numerical_speed(calc, product, pricing_env, engine, base_price=None, bump=None):
    engine = calc._resolve_bump_engine(product, pricing_env, engine)
    bump = bump if bump is not None else calc._bump_config.spot_bump
    base_price, v_up, v_dn = calc._spot_bumped_prices(product, pricing_env, engine, bump, base_price=base_price)
    v_up2 = price at spot*(1+2*bump); v_dn2 = price at spot*(1-2*bump)   # deepcopy env, spot_quote.spot *= (1±2*bump)
    h = pricing_env.spot * bump
    return (v_up2 - 2.0*v_up + 2.0*v_dn - v_dn2) / (2.0 * h**3)

def numerical_zomma(calc, product, pricing_env, engine, base_price=None, vol_bump=None):
    # gamma at vol-bumped envs via calc._get_delta_gamma (honors greeks_mode);
    # central /(2*vol_bump); if current_vol - vol_bump <= 0: one-sided
    # (gamma(vol_up) - gamma(base)) / vol_bump   [mirrors vanna's guard]

def numerical_dividend_volga(calc, product, pricing_env, engine, base_price=None, div_bump=None):
    # envs via calc._build_div_bumped_env(±); calc._calculate_sensitivity(
    #   base_price, price_up, price_down, bump=div_bump, mode="second_order")
```
Registry entries: `speed` (linear 0), `zomma` (linear 0), `dividend_volga` (linear 0), all `default=False`. Enum: add `SPEED`, `ZOMMA`, `DIVIDEND_VOLGA` (VEGA_THETA/GAMMA_THETA in Task 6). Dispatch: appended to `calculate_numerical_greeks` after `dividend_rho`.

- [ ] **Step 1: Failing tests** (vanilla call, BS engine, env from `_build_env` pattern):
```python
def test_speed_matches_manual_stencil():   # explicit 4 reprices vs method
def test_zomma_matches_manual_gamma_diff():
def test_dividend_volga_matches_manual_second_diff():
def test_new_greeks_zero_for_linear_products():   # forward → 0.0 each
def test_request_via_enum_and_string():   # EquityGreek.SPEED and "speed"
```
Each manual test reprices with the same bumps by hand and asserts `pytest.approx(rel=1e-9)` (identical arithmetic path, tight tolerance).
- [ ] **Step 2:** FAIL → implement → PASS.
- [ ] **Step 3:** Gate: suite + fingerprint (old requests unchanged).
- [ ] **Step 4:** Commit `feat(riskmeasures): numerical speed, zomma, dividend_volga`.

---

### Task 6 (E1b): charm, color, vega_theta, gamma_theta + dual clock + exact theta mode

**Files:**
- Modify: `greeks/registry.py` (clock parsing), `greeks/numerical.py`, `greeks/theta_decomposition.py`, facade, enum (`VEGA_THETA`, `GAMMA_THETA`)
- Test: `test/test_higher_order_greeks.py`, `test/test_theta_suite_clocks.py` (new)

**Clock rules (spec §8, exact):**
- `normalize_greeks` strips `_1td` then `_1d` suffix; only for `supports_clock` names (`theta, r_theta, q_theta, convexity_theta, gamma_theta, charm, color, vega_theta`); otherwise `ValidationError`. `GreekRequest.key` keeps the qualified spelling.
- Advance: `1d` → `advance_theta_bump(env, 1, "calendar_days")`; `1td` → `(env, 1, "business_days")`; bare → `(env, bump_config.time_bump_days, bump_config.time_bump_mode)` (incumbent).
- Time-family numericals share one advanced scenario per clock via a per-call memo dict keyed `"time_adv:<clock>"` holding `(product_adv, env_adv, dropped, time_bump, bumped_date)`; `dropped or time_bump<=0 or T<=time_bump` → greek = 0.0 (theta's incumbent guards).

**Implementations:**
```python
def numerical_charm(...):   # delta(adv) - delta(base), inner via calc._get_delta_gamma
def numerical_color(...):   # gamma(adv) - gamma(base), shares adv scenario with charm
def numerical_vega_theta(...):  # vega(product_adv, env_adv) - vega(product, env), each one-sided vol bump
def gamma_theta(calc, product, pricing_env, engine, gamma, clock):
    T = product.get_maturity(pricing_env)
    sigma = pricing_env.get_vol(getattr(product, "strike", pricing_env.spot), T)
    days = 365.0 if clock resolves calendar else float(pricing_env.bus_days_in_year)
    return -0.5 * sigma**2 * pricing_env.spot**2 * gamma / days
```
- Qualified `r_theta_1x`/`q_theta_1x` estimate: `annual = -r/T*(rho/rate_bump)`; scale by `(bumped_date - valuation_date).days / 365.0` (for `_1d` ≡ 1/365, matching incumbent). `convexity_theta_1x = theta_1x - r_theta_1x - q_theta_1x`.
- `theta_decomposition_mode: str = "estimate"` param on `calculate()` + `calculate_numerical_greeks()`; `"exact"` routes components through `exact_theta_components` (extended with `time_bump_mode` passthrough for qualified clocks); invalid value → `ValidationError`.
- charm/color registered with their numerical fns — the silent-miss gap closes here.

- [ ] **Step 1: Failing tests** — `test_theta_suite_clocks.py` (env = CHINA_SSE business-day pattern, valuation Friday 2026-06-26):
```python
def test_theta_1d_and_1td_coexist_in_one_call():      # both keys returned, different values
def test_theta_1td_carries_weekend_calendar_gap():     # matches manual reprice at Monday
def test_gamma_theta_clock_scaling_exact_ratio():      # _1td/_1d == 365/244 exactly
def test_r_theta_1td_uses_actual_calendar_gap():       # factor 3/365 over the weekend
def test_bare_names_unchanged_vs_pre_refactor():       # bare theta == calculate_numerical_theta()
def test_clock_suffix_rejected_for_non_time_greek():   # vanna_1td → ValidationError
def test_charm_color_now_computed():                   # keys PRESENT (the old silent miss)
def test_charm_matches_manual_delta_diff(); test_color_matches_manual_gamma_diff()
def test_vega_theta_matches_manual(); def test_observation_drop_returns_zero()
def test_exact_mode_components_reconcile():            # sum ≈ theta; ValidationError on bad mode
```
`test/test_greeks_registry.py::test_every_registered_name_is_computable` — request each canonical name, alias, and qualified form individually on vanilla+BS; assert key present (the anti-drift test; unblocked from Task 4's charm/color carve-out).
- [ ] **Step 2:** FAIL → implement → PASS.
- [ ] **Step 3:** Gate: full suite + fingerprint byte-identical for old requests.
- [ ] **Step 4:** Commit `feat(riskmeasures): dual-clock theta suite (charm, color, vega_theta, gamma_theta, 1D/1TD, exact mode)`.

---

### Task 7 (E2): Analytical closed forms + auto-routing extension

**Files:**
- Modify: `greeks/analytical.py`, facade (`calculate_analytical_greeks(..., greeks=None)`, `analytical_supported` from `ANALYTICAL_AUTO_SET` ∪ new names)
- Test: `test/test_analytical_higher_order.py` (new)

Closed forms (continuous q; τ=T; all × `contract_multiplier`; φ/Φ from scipy):
```text
vanna            = -exp(-qτ)·φ(d1)·d2/σ                              (call = put)
volga            = S·exp(-qτ)·φ(d1)·√τ·d1·d2/σ                       (call = put)
speed            = -(gamma/S)·(d1/(σ√τ) + 1)                         (call = put)
zomma            = gamma·(d1·d2 - 1)/σ                               (call = put)
charm  (per day) = [q·e^{-qτ}Φ(±d1)·(±1) - e^{-qτ}φ(d1)·(2(r-q)τ - d2σ√τ)/(2τσ√τ)] / 365
                   (call: +Φ(d1); put: -Φ(-d1); sign pinned by FD oracle test)
color  (per day) = [-e^{-qτ}φ(d1)/(2Sτσ√τ)·(2qτ + 1 + d1·(2(r-q)τ - d2σ√τ)/(σ√τ))] / 365
vega_theta       = veta/365/100 where veta = -S·e^{-qτ}φ(d1)√τ·[q + (r-q)d1/(σ√τ) - (1+d1d2)/(2τ)]
                   (vega convention /100, per day /365; sign pinned by FD oracle)
gamma_theta      = -0.5·σ²·S²·gamma / 365
delta_q  (call)  = -τ·e^{-qτ}Φ(d1) - e^{-qτ}φ(d1)√τ/σ
delta_q  (put)   =  τ·e^{-qτ}Φ(-d1) - e^{-qτ}φ(d1)√τ/σ
dividend_volga (call) =  τ²·S·e^{-qτ}Φ(d1)  + τ^{3/2}·S·e^{-qτ}φ(d1)/σ
dividend_volga (put)  = -τ²·S·e^{-qτ}Φ(-d1) + τ^{3/2}·S·e^{-qτ}φ(d1)/σ
   (derivation: dV/dq = ∓τS·e^{-qτ}Φ(±d1) after the φ-identity cancellation
    S·e^{-qτ}φ(d1) = K·e^{-rτ}φ(d2); differentiate once more, ∂d1/∂q = -√τ/σ)
```
- `calculate_analytical_greeks(product, env, price=None, greeks=None)`: `greeks=None` → incumbent keys exactly; explicit list → normalize (canonical only — clock-qualified or unsupported → `ValidationError`), return requested keys. At expiry (`is_zero(T)`) extended names → 0.0.
- `analytical_supported` in `calculate()` = incumbent 6 + {charm, color, speed, zomma, vega_theta, dividend_volga, gamma_theta} (registry `analytical_auto`); clock-qualified requests carry their suffix in the key so never subset-match.

- [ ] **Step 1: Failing oracle tests** (grid: strikes {80,100,120} × maturities {0.25, 1.0} × {call, put}; BS engine numerical as reference):
```python
def test_analytical_vs_numerical_oracle():          # each new greek, tolerances scaled to bump size (rel 5e-2 for 3rd-order, 1e-2 else)
def test_gamma_theta_equals_convexity_theta_for_vanillas():   # analytical: exact; numerical: approx
def test_analytical_default_keys_unchanged()
def test_analytical_greeks_param_returns_requested_only()
def test_auto_routes_new_names_to_analytical_for_vanilla()    # value == explicit analytical call
def test_auto_keeps_vanna_numerical_for_vanilla()             # value == explicit numerical call
def test_analytical_rejects_clock_qualified()
```
- [ ] **Step 2:** FAIL → implement (fix charm/color/veta signs against the FD oracle if needed — the formulas above are the standard published ones, and the oracle is authoritative) → PASS.
- [ ] **Step 3:** Gate: full suite + fingerprint.
- [ ] **Step 4:** Commit `feat(riskmeasures): analytical closed forms for higher-order greeks`.

---

### Task 8: Full verification + review loop

- [ ] **Step 1:** Full test run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/ -q -x --ignore=test/replay_golden -p no:cacheprovider` (fall back to the greeks-related subset + SIMM/portfolio/backtest suites if wall-clock is prohibitive; record what ran).
- [ ] **Step 2:** Final fingerprint vs baseline: byte-identical.
- [ ] **Step 3:** `git checkout -- example/` if mo sample files churned; review `git status` clean.
- [ ] **Step 4:** Run `/zenmux-codex-review-loop` — 2 iterations, model `gpt-5.6-sol`, reasoning `max`, on the branch diff vs main. Apply fixes; rerun gate suite + fingerprint after each fix round.
- [ ] **Step 5:** Final commit(s); summary report.

## Self-review notes

- Spec coverage: §4→Tasks 1–4; §5→Task 4+6; §6→Tasks 5–6; §7–8→Task 6; §9→Task 7; §10→Tasks 5–6; §11–12 gates embedded per task; §3 enforced by fingerprint + key-set tests.
- Type consistency: `GreekRequest(key, canonical, clock)` used in Tasks 4/6/7; `calculate_points(calc, product, pricing_env, engine, request, mode)` uniform across Task 2 modules; per-greek numericals uniformly `(calc, product, pricing_env, engine, ...)`.
- The charm/color Task-4 carve-out (registered without numerical fn until Task 6) is deliberate to keep the R2 fingerprint byte-identical; the completeness test lands with Task 6 which removes the carve-out.
