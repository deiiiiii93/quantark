# QUAD Forward-Density Event Stats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make QUAD full-book batch pricing land in the PDE zone by (Track 1) fusing the KI-probability walk into the stacked event-stats recursion bit-identically, and (Track 2) adding an opt-in `event_stats_mode="forward_density"` that replaces the stacked KO-indicator rows with a 2–3-surface forward transition-density march.

**Architecture:** `SnowballQuadEngine._compute_event_stats` keeps its preamble, gains a shared assembly tail (`_assemble_event_stats`) consumed by both modes, and dispatches on `QuadParams.event_stats_mode`. The forward mode reuses the existing FFT/bridge machinery: the forward kernel is the backward `omega_array` with the α sign flipped and the discount term dropped from β; the forward continuous-KI bridge reuses the landed `_bridge_kernels` cache by passing `-alpha`. `npv` is always the backward `price()`.

**Tech Stack:** numpy (pocketfft), existing `QuadratureMath` (`quad_math.py`), pytest.

**Spec:** `docs/superpowers/specs/2026-08-24-quad-forward-density-event-stats-design.md`

## Global Constraints

- Default behavior (`event_stats_mode="stacked"`) must stay **bit-identical** to pre-change: Tasks 1, 2, 8 are gated by the golden capture script (Task 1 Step 1) run on this machine.
- Checked-in bitwise tests compare two paths **computed in the same test run** — never a live value against a frozen constant (cross-arch rule, spec §6).
- Use `quantark.util.numerical` (`is_close`, `Tolerance`, `safe_log`, `validate_positive`) — never raw float comparisons or hardcoded epsilons in engine code. Test tolerances live as named constants at the top of the test file, marked provisional until the Task 9 pilot bank.
- Canonical `quantark.*` imports only. PEP 8. Docstrings on public methods.
- No silent renormalization or clipping of the density — mass bookkeeping is explicit (spec §8).
- Run tests with `.venv/bin/python -m pytest`; full-suite runs deselect the history-gated test: `--deselect test/mo_volmodels/test_pde_convergence_gate.py::test_quick_end_to_end`.
- Spec deviations decided at planning (record in the evidence doc): (i) forward mode leaves the per-KI-date arrays (`ki_times`, `ki_event_probability`, `ki_survival_probability`) empty for non-latched products, matching stacked-mode schema parity (spec §4.4 said fill them — parity wins until the default flip revisits); (ii) `ki_ever` accumulates touched mass per step (leak-robust) with `1 − ∫p_notouch(T)` kept as a diagnostic identity.

---

### Task 1: Golden capture tool + extract the shared assembly tail

**Files:**
- Create: `docs/autocall-engine-perf/demos/capture_stats_goldens.py`
- Modify: `quantark/asset/equity/engine/quad/snowball_quad_engine.py` (extract `_assemble_event_stats` from the tail of `_compute_event_stats`)

**Interfaces:**
- Produces: `SnowballQuadEngine._assemble_event_stats(*, product, pricing_env, pv, ko_records, ko_times, ko_prob, survival_prob, ed_ko_cf, expected_discounted_maturity_cf, ki_probability, ki_times, ki_event_probability, ki_survival_probability, ki_ever_probability, ki_survive_knocked_in_probability, extra_fields) -> AutocallableEventStats` — Tasks 2/5/6/7 rely on this exact name and keyword list. `expected_discounted_maturity_cf` is passed **before** the coupon subtraction; the helper performs it.

- [ ] **Step 1: Write the golden capture script** (the "failing test" for Tasks 1–2 is bitwise drift against these goldens)

Write `docs/autocall-engine-perf/demos/capture_stats_goldens.py`:

```python
"""Bitwise stats goldens for the QUAD event-stats refactors (same-machine tool).

Capture BEFORE an engine change, `--check` after: every AutocallableEventStats /
PhoenixEventStats field, price(), and price_with_events npv, hex-exact, over a
product matrix. Dev-time gate only — never wire into CI (cross-arch rule).

Run:  .venv/bin/python docs/autocall-engine-perf/demos/capture_stats_goldens.py [--check]
"""
import json
import sys
from dataclasses import fields as dc_fields
from datetime import datetime

import numpy as np

from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment

GOLDEN_PATH = "docs/autocall-engine-perf/demos/.stats_goldens.local.json"  # gitignored-by-location


def env(spot=100.0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.05),
        valuation_date=datetime(2026, 6, 30),
    )


def cases():
    std = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0,
    )
    yield "snowball_cont_ki", SnowballQuadEngine, std, env()
    yield "snowball_ki_low_spot", SnowballQuadEngine, std, env(spot=74.0)  # knocked-in at valuation
    disc = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0,
        ki_observation_type=__import__("quantark.util.enum", fromlist=["ObservationType"]).ObservationType.DISCRETE,
        ki_continuous=False,
        ki_observation_dates=[(d + 1) * 1.9 / 96 for d in range(96)],
    )
    yield "snowball_discrete_ki", SnowballQuadEngine, disc, env()
    dko = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0, disable_ko_after_ki=True,
    )
    yield "snowball_disable_ko_after_ki", SnowballQuadEngine, dko, env()
    phx = create_standard_phoenix(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, coupon_barrier=85.0, coupon_rate=0.01,
        num_observations=23,
    )
    yield "phoenix", PhoenixQuadEngine, phx, env()


def stats_hex(stats):
    out = {}
    for f in dc_fields(stats):
        v = getattr(stats, f.name)
        if isinstance(v, np.ndarray):
            out[f.name] = [float(x).hex() for x in np.asarray(v, dtype=float).ravel()]
        elif isinstance(v, float):
            out[f.name] = v.hex()
        else:
            out[f.name] = repr(v)
    return out


def collect():
    data = {}
    for name, cls, product, e in cases():
        engine = cls(params=QuadParams(grid_points=1001))
        price = float(engine.price(product, e))
        engine = cls(params=QuadParams(grid_points=1001))
        npv = float(engine.price_with_events(product, e).npv)
        engine = cls(params=QuadParams(grid_points=1001))
        stats = engine.calculate_event_stats(product, e)
        data[name] = {"price": price.hex(), "npv": npv.hex(),
                      "stats": stats_hex(stats) if stats is not None else None}
    return data


def main():
    data = collect()
    if "--check" in sys.argv:
        with open(GOLDEN_PATH) as fh:
            golden = json.load(fh)
        diffs = []

        def walk(path, a, b):
            if isinstance(a, dict):
                for k in a:
                    walk(f"{path}.{k}", a[k], b.get(k, "<missing>") if isinstance(b, dict) else "<missing>")
            elif a != b:
                diffs.append((path, a, b))

        walk("", golden, data)
        if diffs:
            print(f"BITWISE FAIL: {len(diffs)} diffs")
            for p, a, b in diffs[:20]:
                print(f"  {p}: {a} -> {b}")
            sys.exit(1)
        print("BITWISE OK")
    else:
        with open(GOLDEN_PATH, "w") as fh:
            json.dump(data, fh, indent=1)
        print(f"captured -> {GOLDEN_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Capture goldens on the unmodified engine**

Run: `.venv/bin/python docs/autocall-engine-perf/demos/capture_stats_goldens.py`
Expected: `captured -> docs/autocall-engine-perf/demos/.stats_goldens.local.json`
(If `ki_observation_dates`/`ki_observation_type` kwargs are rejected by `create_standard_snowball`, build that one case directly with `BarrierConfig` + `PayoffConfig(rebate_rate=0.15, include_principal=False)` + `AccrualConfig(coupon_pay_type=CouponPayType.INSTANT, is_annualized=True)` — the pattern in `docs/autocall-engine-perf/demos/demo_pde_cache_key.py::build_booklike_case`.)

- [ ] **Step 3: Extract `_assemble_event_stats` (pure code motion)**

In `snowball_quad_engine.py`, move the tail of `_compute_event_stats` — everything from `payment_timings = resolve_structured_payment_timings(...)` through `return self._make_event_stats(...)` — into a new method placed directly after `_compute_event_stats`, and move the coupon/maturity adjustment in with it:

```python
def _assemble_event_stats(
    self, *, product, pricing_env, pv, ko_records, ko_times, ko_prob,
    survival_prob, ed_ko_cf, expected_discounted_maturity_cf,
    ki_probability, ki_times, ki_event_probability, ki_survival_probability,
    ki_ever_probability, ki_survive_knocked_in_probability, extra_fields,
):
    """Shared stats assembly for the stacked and forward event-stats modes.

    ``expected_discounted_maturity_cf`` arrives BEFORE the coupon subtraction;
    extra cashflow streams (Phoenix coupons) are removed here so
    pv = sum(ko) + sum(coupon) + maturity stays correctly classified.
    """
    n_ko = len(ko_records)
    coupon_cf = extra_fields.get("expected_discounted_coupon_cashflow")
    if coupon_cf is not None:
        expected_discounted_maturity_cf -= float(np.sum(coupon_cf))

    payment_timings = resolve_structured_payment_timings(
        product, pricing_env, ko_records,
    )
    # ... [MOVED VERBATIM: terminal/determination/payment lists, the
    #      `if "coupon_probability" in extra_fields:` block, the terminal
    #      appends, payment_aware_cashflow_fields(...)] ...
    return self._make_event_stats(
        pv=pv,
        ko_times=ko_times,
        ko_probability=ko_prob,
        survival_probability=survival_prob,
        expected_discounted_ko_cashflow=ed_ko_cf,
        ki_probability=ki_probability,
        expected_discounted_maturity_cashflow=expected_discounted_maturity_cf,
        reconciliation_error=0.0,
        ki_times=ki_times,
        ki_event_probability=ki_event_probability,
        ki_survival_probability=ki_survival_probability,
        ki_ever_probability=ki_ever_probability,
        ki_survive_knocked_in_probability=ki_survive_knocked_in_probability,
        **cashflow_fields,
        **extra_fields,
    )
```

In `_compute_event_stats`, replace the moved region (including the two moved
lines `coupon_cf = ...` / `expected_discounted_maturity_cf -= ...`) with:

```python
        return self._assemble_event_stats(
            product=product, pricing_env=pricing_env, pv=pv,
            ko_records=ko_records, ko_times=ko_times, ko_prob=ko_prob,
            survival_prob=survival_prob, ed_ko_cf=ed_ko_cf,
            expected_discounted_maturity_cf=expected_discounted_maturity_cf,
            ki_probability=ki_probability, ki_times=ki_times,
            ki_event_probability=ki_event_probability,
            ki_survival_probability=ki_survival_probability,
            ki_ever_probability=ki_ever_probability,
            ki_survive_knocked_in_probability=ki_survive_knocked_in_probability,
            extra_fields=extra_fields,
        )
```

- [ ] **Step 4: Verify bitwise + suite subset**

Run: `.venv/bin/python docs/autocall-engine-perf/demos/capture_stats_goldens.py --check`
Expected: `BITWISE OK`
Run: `.venv/bin/python -m pytest test/ -k "quad" -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -f docs/autocall-engine-perf/demos/capture_stats_goldens.py
git add quantark/asset/equity/engine/quad/snowball_quad_engine.py
git commit -m "refactor(quad): extract shared event-stats assembly tail (bitwise)"
```

---

### Task 2: Track 1 — fuse the KI-probability walk into the stacked recursion

**Files:**
- Modify: `quantark/asset/equity/engine/quad/snowball_quad_engine.py` (`_compute_event_stats` stacked body: the `n_rows` allocation, the main loop's KO/KI event blocks, and the `elif product.has_ki_barrier and want_ki:` second walk)

**Interfaces:**
- Consumes: nothing new. Produces: no API change — internal only. The row layout inside the stacked recursion becomes `[0:n_base) = KO + coupon rows, n_base = ki row, n_base+1 = ever row` where `n_base = n_ko + (extra coupon rows if want_coupon)`.

- [ ] **Step 1: Restructure the allocation**

Replace the current allocation:

```python
        n_ko = len(ko_records)
        n_base = n_ko + (self._n_extra_quad_rows(n_ko) if want_coupon else 0)
        fuse_ki = (
            product.has_ki_barrier and want_ki and not knocked_in_at_valuation
        )
        n_rows = n_base + (2 if fuse_ki else 0)
        ki_row, ever_row = n_base, n_base + 1
        v_in = np.zeros((n_rows, grid.size), dtype=float)
        v_out = np.zeros((n_rows, grid.size), dtype=float)
        if fuse_ki:
            # KI indicator pair terminal/initial condition: knocked-in
            # surface = 1, not-knocked-in = 0 (v_out rows stay zero).
            v_in[ki_row] = 1.0
            v_in[ever_row] = 1.0
```

- [ ] **Step 2: Slice the whole-stack event operations to `[:n_base]` and add the fused rows' hard-mask semantics**

In the main loop's KO block, change every whole-stack operation to a
`[:n_base]` slice (values unchanged when `fuse_ki` is False since then
`n_base == n_rows`):

```python
                if self._use_cell_average_events() and ko_is_reachable:
                    breach = np.zeros((n_base, grid.size), dtype=float)
                    breach[int(ko_index)] = float(discount_delay)
                    v_out[:n_base] = self._project_quad_event(
                        grid, ko_record.barrier, spot,
                        v_survive=v_out[:n_base], v_breach=breach,
                        breach_up=not product.is_reverse,
                    )
                    if not disable_ko_after_ki:
                        v_in[:n_base] = self._project_quad_event(
                            grid, ko_record.barrier, spot,
                            v_survive=v_in[:n_base], v_breach=breach,
                            breach_up=not product.is_reverse,
                        )
                elif ko_is_reachable:
                    v_out[:n_base] *= (1.0 - ko_w)
                    v_out[int(ko_index)] += ko_w * float(discount_delay)
                    if not disable_ko_after_ki:
                        v_in[:n_base] *= (1.0 - ko_w)
                        v_in[int(ko_index)] += ko_w * float(discount_delay)
```

Immediately after the KO block (still inside `if ko_record is not None:`,
after `_set_extra_quad_indicators`), add the fused rows' hard KO masks —
NOTE: applied whenever the record matches, with **no** reachability gate and
**no** cell-average variant, exactly as the old second walk did:

```python
                hard_ko_mask = None
                if fuse_ki:
                    hard_ko_mask = (
                        spot_grid <= ko_record.barrier
                        if product.is_reverse
                        else spot_grid >= ko_record.barrier
                    )
                    v_out[ki_row][hard_ko_mask] = 0.0
                    if not disable_ko_after_ki:
                        v_in[ki_row][hard_ko_mask] = 0.0
                    # ever rows carry NO KO absorption (pure first-passage).
```

(`hard_ko_mask = None` must also be initialized once per loop iteration
before the `if ko_record is not None:` block so the discrete-KI block below
can reference it.)

In the discrete-KI block, slice the blend and add the fused rows' hard
transfers (old-walk order: ever copy on the raw mask FIRST, then the
narrowed ki copy):

```python
                    v_out[:n_base] = self._blend_ki_transition(
                        v_out[:n_base], v_in[:n_base], grid, spot_grid,
                        ki_record.barrier, spot, smoothing_width,
                        product.is_reverse,
                        ko_weight=(ko_w if not disable_ko_after_ki else None),
                    )
                    if fuse_ki:
                        hard_ki_mask = (
                            spot_grid >= ki_record.barrier
                            if product.is_reverse
                            else spot_grid <= ki_record.barrier
                        )
                        v_out[ever_row][hard_ki_mask] = v_in[ever_row][hard_ki_mask]
                        if hard_ko_mask is not None and not disable_ko_after_ki:
                            hard_ki_mask = hard_ki_mask & ~hard_ko_mask
                        v_out[ki_row][hard_ki_mask] = v_in[ki_row][hard_ki_mask]
```

(Check the old walk's exact conditions while editing: the KO hard mask there
is built from `self._match_record(obs_time, ko_records)` — confirm it selects
the same record as the main loop's `is_close` scan; if `_match_record` uses a
different tolerance, keep the old walk's `_match_record` call for the fused
rows so behavior is identical. The Step 5 bitwise gate is the authority.)

- [ ] **Step 3: Replace the second walk with fused-row readouts**

Delete the entire `elif product.has_ki_barrier and want_ki:` walk body
(surface allocations, its own step loop, its diffusion calls) and replace
with readouts from the fused rows:

```python
        elif product.has_ki_barrier and want_ki:
            df_T = float(df_local(maturity))
            pv_ki_no_ko = float(math_utils.interpolate(v_out[ki_row], x=0.0))
            pv_ki_ever = float(math_utils.interpolate(v_out[ever_row], x=0.0))
            if df_T > 0:
                ki_probability = float(pv_ki_no_ko / df_T)
                ki_survive_knocked_in_probability = ki_probability
                ki_ever_probability = float(pv_ki_ever / df_T)
```

Also confirm the KO readout block (`ed_unit = ... for i in range(n_ko)`)
indexes only `[0:n_ko)` (it does — `range(n_ko)`), and that
`_extract_extra_quad_stats` reads rows `[n_ko : n_ko + extras)` — both
unaffected by the appended rows.

- [ ] **Step 4: Check the terminal-condition and t=0 special paths**

Grep the stacked body for any other whole-stack operation
(`np.zeros_like(v_out)`, `v_out[...] =` without a row index, terminal-KO
handling) and apply the same `[:n_base]` slicing + fused-row exceptions.
Expected: the terminal KO observation flows through the same KO block edited
in Step 2 (no separate terminal branch exists in the QUAD stats loop).

- [ ] **Step 5: Verify bitwise, suite, and the walk elimination**

Run: `.venv/bin/python docs/autocall-engine-perf/demos/capture_stats_goldens.py --check`
Expected: `BITWISE OK` (all five cases — this is the task's hard gate).
Run: `.venv/bin/python -m pytest test/ -k "quad or event_stats or phoenix or snowball" -q`
Expected: all pass.
Run: `.venv/bin/python docs/autocall-engine-perf/demos/demo_quad_event_stats.py`
Expected: baseline `pwe()` drops versus the pre-task run (record the number);
`bitwise OK` rows still hold.

- [ ] **Step 6: Commit**

```bash
git add quantark/asset/equity/engine/quad/snowball_quad_engine.py
git commit -m "perf(quad): fuse the KI-probability walk into the stacked recursion (bitwise)"
```

---

### Task 3: `QuadParams.event_stats_mode`

**Files:**
- Modify: `quantark/asset/equity/param/engine_params.py` (`QuadParams` dataclass + its `__post_init__` validation + class docstring)
- Test: `test/test_quad_forward_density_stats.py` (new file)

**Interfaces:**
- Produces: `QuadParams.event_stats_mode: str = "stacked"`, validated member of `{"stacked", "forward_density"}`. Tasks 5–8 read it via `self.params.event_stats_mode`.

- [ ] **Step 1: Write the failing tests**

```python
"""Forward-density event-stats mode (spec 2026-08-24). Battery grows task by task."""
import pytest

from quantark.asset.equity.param import QuadParams
from quantark.util.exceptions import ValidationError


def test_event_stats_mode_default_is_stacked():
    assert QuadParams().event_stats_mode == "stacked"


def test_event_stats_mode_accepts_forward_density():
    assert QuadParams(event_stats_mode="forward_density").event_stats_mode == "forward_density"


def test_event_stats_mode_rejects_unknown():
    with pytest.raises(ValidationError):
        QuadParams(event_stats_mode="fwd")
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_quad_forward_density_stats.py -q`
Expected: FAIL (`unexpected keyword argument 'event_stats_mode'`).

- [ ] **Step 3: Implement**

In `QuadParams`: add the field (grouped with the other stats-affecting
params) and validation mirroring the existing `cache_strategy` pattern:

```python
    event_stats_mode: str = "stacked"
```

In `QuadParams.__post_init__` (create one that calls `super().__post_init__()`
first if the class does not define one yet):

```python
        if self.event_stats_mode not in ("stacked", "forward_density"):
            raise ValidationError(
                "event_stats_mode must be one of stacked, forward_density, "
                f"got {self.event_stats_mode}"
            )
```

Docstring line for the attribute list: `event_stats_mode: Event-distribution
algorithm: "stacked" (per-observation indicator rows, default) or
"forward_density" (forward transition-density march; distribution values
differ at finite grid, npv identical; see spec 2026-08-24).`

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest test/test_quad_forward_density_stats.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/param/engine_params.py test/test_quad_forward_density_stats.py
git commit -m "feat(quad): QuadParams.event_stats_mode flag (stacked default)"
```

---

### Task 4: Forward primitives + density identities (battery gate c, no barriers)

**Files:**
- Modify: `quantark/asset/equity/engine/quad/snowball_quad_engine.py` (new private helpers after `_bridge_kernels`)
- Test: `test/test_quad_forward_density_stats.py`

**Interfaces:**
- Produces (Tasks 5–7 consume these exact signatures):
  - `SnowballQuadEngine._forward_kernel(math_utils, tau_step, alpha) -> (omega_fwd, prefactor_fwd)`
  - `SnowballQuadEngine._diffuse_density(values, math_utils, omega_fwd, prefactor_fwd, p_lr, p_ur, p0) -> np.ndarray` (1-D or (k,N); **no tail correction**)
  - `SnowballQuadEngine._density_integral(math_utils, values) -> float` (staticmethod)
  - `SnowballQuadEngine._forward_seed(math_utils, tau1, alpha1) -> np.ndarray`
  - `SnowballQuadEngine._forward_seed_touch_probability(math_utils, tau1, log_barrier, is_reverse) -> np.ndarray`

- [ ] **Step 1: Write the failing identity tests**

Append to `test/test_quad_forward_density_stats.py`:

```python
import math

import numpy as np
from scipy.stats import norm

from quantark.asset.equity.engine.quad.quad_math import QuadratureMath
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine

# Provisional gate-(c) tolerances; tightened + banked by the Task 9 pilot.
MASS_TOL = 1e-6
MOMENT_RTOL = 1e-4
FWD_VALUE_RTOL = 1e-5

SPOT, VOL, R, Q, T = 100.0, 0.2, 0.03, 0.05, 1.0
ALPHA = (R - Q - 0.5 * VOL * VOL) / (VOL * VOL)


def _math_utils(grid_x=2001):
    return QuadratureMath(
        grid_x=grid_x, spot=SPOT, maturity=T, vol_max=VOL,
        num_std_devs=10, align_log=None, integration_rule="simpson",
        fft_padding_factor=2, fft_filter_alpha=0.0, fft_filter_power=2,
    )


def _march_free_density(engine, mu, n_steps=50):
    dt = T / n_steps
    tau_step = 0.5 * VOL * VOL * dt
    p = engine._forward_seed(mu, tau_step, ALPHA)
    omega, pref = engine._forward_kernel(mu, tau_step, ALPHA)
    p_lr, p_ur, p0 = 0, mu.grid.size - 1, (mu.grid.size - 1) % 2
    for _ in range(n_steps - 1):
        p = engine._diffuse_density(p, mu, omega, pref, p_lr, p_ur, p0)
    return p


def test_forward_density_mass_mean_variance():
    engine = SnowballQuadEngine()
    mu = _math_utils()
    p = _march_free_density(engine, mu)
    mass = engine._density_integral(mu, p)
    mean = engine._density_integral(mu, mu.grid * p)
    var = engine._density_integral(mu, (mu.grid - mean) ** 2 * p)
    m = R - Q - 0.5 * VOL * VOL
    assert abs(mass - 1.0) < MASS_TOL
    # Sign of the drift is the kernel-orientation detector.
    assert mean * m > 0.0
    assert abs(mean - m * T) < MOMENT_RTOL * max(abs(m * T), 1e-3)
    assert abs(var - VOL * VOL * T) / (VOL * VOL * T) < MOMENT_RTOL


def test_forward_density_undiscounted_call_value():
    engine = SnowballQuadEngine()
    mu = _math_utils()
    p = _march_free_density(engine, mu)
    strike = 105.0
    payoff = np.maximum(SPOT * np.exp(mu.grid) - strike, 0.0)
    got = engine._density_integral(mu, payoff * p)
    fwd = SPOT * math.exp((R - Q) * T)
    sig = VOL * math.sqrt(T)
    d1 = (math.log(fwd / strike) + 0.5 * sig * sig) / sig
    want = fwd * norm.cdf(d1) - strike * norm.cdf(d1 - sig)
    assert abs(got - want) / want < FWD_VALUE_RTOL
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_quad_forward_density_stats.py -k forward_density -q`
Expected: FAIL (`AttributeError: _forward_seed`).
(If `QuadratureMath` rejects any constructor kwarg above, copy the exact call
used by `_compute_event_stats` — the parameter set there is authoritative.)

- [ ] **Step 3: Implement the primitives**

Add after `_bridge_kernels` in `snowball_quad_engine.py`:

```python
    # --- Forward-density primitives (event_stats_mode="forward_density") ---

    def _forward_kernel(self, math_utils, tau_step: float, alpha: float):
        """Forward transition kernel: the backward omega with the alpha sign
        flipped (correlation -> convolution) and the discount term dropped
        from beta (densities are undiscounted; discounting happens at
        readout)."""
        omega_fwd = np.exp(
            -(math_utils.z_grid ** 2) / (4.0 * tau_step)
            + alpha * math_utils.z_grid
        )
        prefactor_fwd = (
            math.exp(-(alpha * alpha) * tau_step)
            / math.sqrt(math.pi * tau_step) / 2.0
        )
        return omega_fwd, prefactor_fwd

    def _diffuse_density(
        self, values, math_utils, omega_fwd, prefactor_fwd, p_lr, p_ur, p0
    ):
        """One forward transport step. No tail correction: a density is ~0 at
        the grid edges by construction (num_std_devs-wide grid); mass
        bookkeeping is the caller's diagnostic."""
        if values.ndim == 1:
            u = math_utils.simpson_weights(values, p_lr, p_ur, p0)
            return prefactor_fwd * math_utils.convolution_fft(omega_fwd, u)
        u = math_utils.simpson_weights_many(values, p_lr, p_ur, p0)
        return prefactor_fwd * math_utils.convolution_fft_many(omega_fwd, u)

    @staticmethod
    def _density_integral(math_utils, values) -> float:
        """Simpson integral of a grid function (weights are h-scaled)."""
        return float(np.dot(math_utils.simpson_weight_vector(), values))

    def _forward_seed(self, math_utils, tau1: float, alpha1: float):
        """Exact density after the first interval (analytic first step —
        no discrete delta on the grid; spec section 4.3). var = sigma^2*dt1 =
        2*tau1; mean = m*dt1 = alpha1 * 2*tau1."""
        var = 2.0 * tau1
        mean = var * alpha1
        grid = math_utils.grid
        return np.exp(-((grid - mean) ** 2) / (2.0 * var)) / math.sqrt(
            2.0 * math.pi * var
        )

    def _forward_seed_touch_probability(
        self, math_utils, tau1: float, log_barrier: float, is_reverse: bool
    ):
        """Point-source Brownian-bridge touch probability over the first
        interval (source pinned at x=0 = spot), same formula family as the
        step kernel's p_hit."""
        grid = math_utils.grid
        denom = 2.0 * tau1
        if is_reverse:
            d0 = log_barrier - grid
            d1 = log_barrier - 0.0
        else:
            d0 = grid - log_barrier
            d1 = 0.0 - log_barrier
        safe = (d0 > 0.0) & (d1 > 0.0)
        exponent = np.clip(
            np.where(safe, -2.0 * d0 * d1 / denom, 0.0), -745.0, 0.0
        )
        return np.where(safe, np.exp(exponent), 1.0)
```

- [ ] **Step 4: Run the identity tests**

Run: `.venv/bin/python -m pytest test/test_quad_forward_density_stats.py -q`
Expected: all pass. If the drift-sign assertion fails, flip the alpha sign in
`_forward_kernel` — the test is the authority on the convolution orientation
(spec §8); do not touch anything else.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/quad/snowball_quad_engine.py test/test_quad_forward_density_stats.py
git commit -m "feat(quad): forward-density primitives with analytic identity gates"
```

---

### Task 5: Forward event quantities (KO + discrete KI + terminal split) + dispatch

**Files:**
- Modify: `quantark/asset/equity/engine/quad/snowball_quad_engine.py` (`_forward_event_quantities`, extra-readout hooks, dispatch inside `_compute_event_stats`)
- Test: `test/test_quad_forward_density_stats.py`

**Interfaces:**
- Consumes: Task 1 `_assemble_event_stats`, Task 4 primitives.
- Produces:
  - `SnowballQuadEngine._forward_event_quantities(*, product, pricing_env, math_utils, times, tau, alpha_vec, vol_vec, dt, ko_records, ki_records, ki_continuous, ki_barrier_continuous, reachable_ko, spot, spot_grid, smoothing_width, disable_ko_after_ki, knocked_in_at_valuation, want_ki, want_coupon, df_local, rate) -> dict` with keys `ko_prob, ed_ko_cf, survival_prob, ki_probability, ki_ever_probability, ki_survive_knocked_in_probability, ki_times, ki_event_probability, ki_survival_probability, extra_fields, mass_diagnostic`.
  - Hooks (Phoenix overrides in Task 7): `_forward_extra_state(n_ko) -> object|None` (returns `None` for Snowball), `_forward_extra_readouts_at_obs(state, ko_index, ko_record, p_alive, math_utils, smoothing_width, product) -> None`, `_forward_extra_fields(state, ko_records, product, pricing_env, rate, df_local) -> dict` (returns `{}` for Snowball).

- [ ] **Step 1: Write the failing parity test (discrete products first)**

Append:

```python
from datetime import datetime

from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment

# Provisional forward-vs-stacked parity tolerances at grid 2001 (Task 9 pilot
# tightens and banks these).
KO_PROB_ATOL = 2e-3
KI_PROB_ATOL = 2e-2
CF_RTOL = 5e-3


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.05),
        valuation_date=datetime(2026, 6, 30),
    )


def _stats_pair(engine_cls, product, env, grid_points=2001, **extra_params):
    stacked = engine_cls(
        params=QuadParams(grid_points=grid_points, **extra_params)
    ).calculate_event_stats(product, env)
    forward = engine_cls(
        params=QuadParams(grid_points=grid_points,
                          event_stats_mode="forward_density", **extra_params)
    ).calculate_event_stats(product, env)
    return stacked, forward


def _no_ki_snowball():
    return create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=None, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0,
    )


def test_forward_matches_stacked_no_ki_snowball():
    stacked, forward = _stats_pair(SnowballQuadEngine, _no_ki_snowball(), _env())
    assert np.max(np.abs(forward.ko_probability - stacked.ko_probability)) < KO_PROB_ATOL
    assert np.max(np.abs(forward.survival_probability - stacked.survival_probability)) < KO_PROB_ATOL
    np.testing.assert_allclose(
        forward.expected_discounted_ko_cashflow,
        stacked.expected_discounted_ko_cashflow,
        rtol=CF_RTOL, atol=1e-4,
    )
    # npv path is shared: pv must be EXACTLY the backward price in both modes.
    assert float(forward.pv).hex() == float(stacked.pv).hex()
```

(If `create_standard_snowball(ki_barrier=None)` still defaults a KI barrier
in, pass whatever the helper documents for "no KI" — check its docstring —
or build the product directly with `BarrierConfig(ko_barrier=..., ko_rate=...,
ko_observation_dates=[...], ki_barrier=None)`.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_quad_forward_density_stats.py::test_forward_matches_stacked_no_ki_snowball -q`
Expected: FAIL — forward mode not implemented (stats identical to stacked or
an `AttributeError`, depending on dispatch state; either counts).

- [ ] **Step 3: Implement `_forward_event_quantities` and the hooks**

Add to `snowball_quad_engine.py` (after `_forward_seed_touch_probability`).
The per-step vectors and indexing conventions are the stacked loop's own:
`tau[k]`/`alpha_vec[k]`/`vol_vec[k]`/`dt[k]` describe the interval **ending**
at `times[k-1]` (interval 1 runs 0 → `times[0]`).

```python
    # --- Extra forward-readout hooks (Phoenix overrides; Snowball no-ops) ---

    def _forward_extra_state(self, n_ko: int):
        return None

    def _forward_extra_readouts_at_obs(
        self, state, ko_index, ko_record, p_alive, math_utils,
        smoothing_width, product,
    ) -> None:
        return None

    def _forward_extra_fields(
        self, state, ko_records, product, pricing_env, rate, df_local
    ) -> dict:
        return {}

    def _forward_event_quantities(
        self, *, product, pricing_env, math_utils, times, tau, alpha_vec,
        vol_vec, dt, ko_records, ki_records, ki_continuous,
        ki_barrier_continuous, reachable_ko, spot, spot_grid,
        smoothing_width, disable_ko_after_ki, knocked_in_at_valuation,
        want_ki, want_coupon, df_local, rate,
    ) -> dict:
        """Forward transition-density march producing the event-distribution
        quantities (spec section 4). 2-3 surfaces: p_out / p_in (KO-absorbing,
        KI-splitting) and, when a KI stream is requested, p_notouch (bridge
        survival only, no KO absorption) for the "ever" statistic."""
        grid = math_utils.grid
        n_ko = len(ko_records)
        full_p_lr, full_p_ur, full_p0 = 0, grid.size - 1, (grid.size - 1) % 2
        absorbed = np.zeros(n_ko, dtype=float)
        ever_touched = 0.0
        track_ever = bool(
            want_ki and product.has_ki_barrier and not knocked_in_at_valuation
        )
        extra_state = self._forward_extra_state(n_ko) if want_coupon else None
        log_ki = (
            safe_log(ki_barrier_continuous / spot) if ki_continuous else None
        )

        # Analytic first step to times[0] (interval index 1).
        tau1, alpha1 = float(tau[1]), float(alpha_vec[1])
        p_free = self._forward_seed(math_utils, tau1, alpha1)
        if knocked_in_at_valuation:
            p_in = p_free
            p_out = np.zeros_like(p_free)
            p_nt = None
        elif ki_continuous and product.has_ki_barrier:
            touch0 = self._forward_seed_touch_probability(
                math_utils, tau1, log_ki, product.is_reverse
            )
            p_out = p_free * (1.0 - touch0)
            p_in = p_free * touch0
            if track_ever:
                ever_touched += self._density_integral(math_utils, p_in)
                p_nt = p_out.copy()
            else:
                p_nt = None
        else:
            p_out = p_free
            p_in = np.zeros_like(p_free)
            p_nt = p_free.copy() if track_ever else None

        for step_index in range(1, len(times) + 1):
            if step_index > 1:
                tau_step = float(tau[step_index])
                alpha = float(alpha_vec[step_index])
                vol_step = float(vol_vec[step_index])
                omega_fwd, prefactor_fwd = self._forward_kernel(
                    math_utils, tau_step, alpha
                )
                rows = [p_out, p_in] + ([p_nt] if p_nt is not None else [])
                transported = self._diffuse_density(
                    np.vstack(rows), math_utils, omega_fwd, prefactor_fwd,
                    full_p_lr, full_p_ur, full_p0,
                )
                if ki_continuous and not knocked_in_at_valuation:
                    band = self._bridge_band(tau_step, math_utils.h, grid.size)
                    denom = vol_step * vol_step * float(dt[step_index])
                    kernels = self._bridge_kernels(
                        math_utils, band, tau_step, -alpha, denom,
                        log_ki, product.is_reverse,
                    )
                    touch_rows = np.vstack(
                        [p_out] + ([p_nt] if p_nt is not None else [])
                    )
                    touched = np.zeros(
                        (touch_rows.shape[0], grid.size), dtype=float
                    )
                    for source, target, kernel in kernels:
                        touched[:, target] += (
                            prefactor_fwd * kernel * touch_rows[:, source]
                        )
                    p_out = transported[0] - touched[0]
                    p_in = transported[1] + touched[0]
                    if p_nt is not None:
                        p_nt = transported[2] - touched[1]
                        ever_touched += self._density_integral(
                            math_utils, touched[1]
                        )
                else:
                    p_out = transported[0]
                    p_in = transported[1]
                    if p_nt is not None:
                        p_nt = transported[2]

            obs_time = times[step_index - 1]
            ko_index, ko_record = None, None
            for idx, rec in enumerate(ko_records):
                if is_close(obs_time, rec.observation_time,
                            abs_tol=Tolerance.PRECISION):
                    ko_index, ko_record = idx, rec
                    break
            if ko_record is not None:
                if extra_state is not None:
                    # Coupons read the alive density BEFORE KO absorption
                    # (a coupon on a simultaneous KO is still paid).
                    self._forward_extra_readouts_at_obs(
                        extra_state, int(ko_index), ko_record, p_out + p_in,
                        math_utils, smoothing_width, product,
                    )
                if bool(reachable_ko[int(ko_index)]):
                    if self._use_cell_average_events():
                        zero = np.zeros((1, grid.size), dtype=float)
                        proj = self._project_quad_event(
                            grid, ko_record.barrier, spot,
                            v_survive=p_out[None, :], v_breach=zero,
                            breach_up=not product.is_reverse,
                        )[0]
                        absorbed_mass = self._density_integral(
                            math_utils, p_out
                        ) - self._density_integral(math_utils, proj)
                        p_out = proj
                        if not disable_ko_after_ki:
                            proj_in = self._project_quad_event(
                                grid, ko_record.barrier, spot,
                                v_survive=p_in[None, :], v_breach=zero,
                                breach_up=not product.is_reverse,
                            )[0]
                            absorbed_mass += self._density_integral(
                                math_utils, p_in
                            ) - self._density_integral(math_utils, proj_in)
                            p_in = proj_in
                    else:
                        ko_w = self._event_weight(
                            grid, ko_record.barrier, spot, smoothing_width,
                            trigger_is_down=product.is_reverse,
                        )
                        absorbed_mass = self._density_integral(
                            math_utils, ko_w * p_out
                        )
                        p_out = p_out * (1.0 - ko_w)
                        if not disable_ko_after_ki:
                            absorbed_mass += self._density_integral(
                                math_utils, ko_w * p_in
                            )
                            p_in = p_in * (1.0 - ko_w)
                    absorbed[int(ko_index)] = absorbed_mass

            if (
                (not ki_continuous)
                and ki_records
                and not knocked_in_at_valuation
            ):
                ki_record = self._match_record(obs_time, ki_records)
                if ki_record is not None:
                    ki_w = self._event_weight(
                        grid, ki_record.barrier, spot, smoothing_width,
                        trigger_is_down=not product.is_reverse,
                    )
                    transfer = ki_w * p_out
                    p_out = p_out - transfer
                    p_in = p_in + transfer
                    if p_nt is not None:
                        touched_nt = ki_w * p_nt
                        ever_touched += self._density_integral(
                            math_utils, touched_nt
                        )
                        p_nt = p_nt - touched_nt

        # --- Readouts ---
        ko_prob = absorbed.copy()
        ed_ko_cf = np.zeros(n_ko, dtype=float)
        survival_prob = np.ones(n_ko, dtype=float)
        cumulative = 0.0
        for i, rec in enumerate(ko_records):
            obs = float(rec.observation_time)
            df_total = float(df_local(obs)) * float(
                self._ko_discount(rate, obs, rec.settlement_time,
                                  df_fn=df_local)
            )
            payoff = float(rec.payoff) if rec.payoff is not None else 0.0
            ed_ko_cf[i] = ko_prob[i] * df_total * payoff
            cumulative += ko_prob[i]
            survival_prob[i] = max(0.0, 1.0 - cumulative)

        ki_probability = 0.0
        ki_ever_probability = 0.0
        ki_survive_knocked_in_probability = 0.0
        ki_times = np.array([], dtype=float)
        ki_event_probability = np.array([], dtype=float)
        ki_survival_probability = np.array([], dtype=float)
        if knocked_in_at_valuation:
            ki_probability = 1.0
            ki_ever_probability = 1.0
            ki_survive_knocked_in_probability = 1.0
            ki_times = np.array([0.0], dtype=float)
            ki_event_probability = np.array([1.0], dtype=float)
            ki_survival_probability = np.array([0.0], dtype=float)
        elif product.has_ki_barrier and want_ki:
            ki_probability = self._density_integral(math_utils, p_in)
            ki_survive_knocked_in_probability = ki_probability
            ki_ever_probability = float(ever_touched)

        extra_fields = (
            self._forward_extra_fields(
                extra_state, ko_records, product, pricing_env, rate, df_local
            )
            if extra_state is not None
            else {}
        )
        terminal_mass = self._density_integral(math_utils, p_out + p_in)
        return {
            "ko_prob": ko_prob,
            "ed_ko_cf": ed_ko_cf,
            "survival_prob": survival_prob,
            "ki_probability": ki_probability,
            "ki_ever_probability": ki_ever_probability,
            "ki_survive_knocked_in_probability": ki_survive_knocked_in_probability,
            "ki_times": ki_times,
            "ki_event_probability": ki_event_probability,
            "ki_survival_probability": ki_survival_probability,
            "extra_fields": extra_fields,
            "mass_diagnostic": float(terminal_mass + np.sum(absorbed)),
        }
```

- [ ] **Step 4: Wire the dispatch into `_compute_event_stats`**

Immediately before the stacked allocation (`n_ko = len(ko_records)`), insert:

```python
        if str(self.params.event_stats_mode) == "forward_density":
            q = self._forward_event_quantities(
                product=product, pricing_env=pricing_env,
                math_utils=math_utils, times=times, tau=tau,
                alpha_vec=alpha_vec, vol_vec=vol_vec, dt=dt,
                ko_records=ko_records, ki_records=ki_records,
                ki_continuous=ki_continuous,
                ki_barrier_continuous=ki_barrier_continuous,
                reachable_ko=reachable_ko, spot=spot, spot_grid=spot_grid,
                smoothing_width=smoothing_width,
                disable_ko_after_ki=disable_ko_after_ki,
                knocked_in_at_valuation=knocked_in_at_valuation,
                want_ki=want_ki, want_coupon=want_coupon,
                df_local=df_local, rate=rate,
            )
            pv = float(self.price(product, pricing_env))
            ko_times = np.array(
                [rec.observation_time for rec in ko_records], dtype=float
            )
            # Diagnostic only (spec section 8): terminal mass + absorbed KO
            # mass; drifts from 1.0 by edge leakage. Never used numerically.
            self._last_forward_mass_diagnostic = float(q["mass_diagnostic"])
            return self._assemble_event_stats(
                product=product, pricing_env=pricing_env, pv=pv,
                ko_records=ko_records, ko_times=ko_times,
                ko_prob=q["ko_prob"], survival_prob=q["survival_prob"],
                ed_ko_cf=q["ed_ko_cf"],
                expected_discounted_maturity_cf=float(
                    pv - float(np.sum(q["ed_ko_cf"]))
                ),
                ki_probability=q["ki_probability"],
                ki_times=q["ki_times"],
                ki_event_probability=q["ki_event_probability"],
                ki_survival_probability=q["ki_survival_probability"],
                ki_ever_probability=q["ki_ever_probability"],
                ki_survive_knocked_in_probability=q[
                    "ki_survive_knocked_in_probability"
                ],
                extra_fields=q["extra_fields"],
            )
```

While inserting, confirm every name in the call exists in the preamble at
that point (`times`, `tau`, `alpha_vec`, `vol_vec`, `dt`, `df_local`,
`ki_barrier_continuous`, `reachable_ko`, `knocked_in_at_valuation`); if any
is computed later in the stacked body, hoist its computation above the
dispatch (values unchanged, order-only — then run the Task 1 golden check to
prove the hoist is bitwise for the stacked mode).

- [ ] **Step 5: Add the discrete-KI parity test and run**

```python
def _discrete_ki_snowball():
    from quantark.util.enum import ObservationType
    return create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0,
        ki_observation_type=ObservationType.DISCRETE, ki_continuous=False,
        ki_observation_dates=[(d + 1) * 1.9 / 96 for d in range(96)],
    )


def test_forward_matches_stacked_discrete_ki():
    stacked, forward = _stats_pair(SnowballQuadEngine, _discrete_ki_snowball(), _env())
    assert np.max(np.abs(forward.ko_probability - stacked.ko_probability)) < KO_PROB_ATOL
    assert abs(forward.ki_probability - stacked.ki_probability) < KI_PROB_ATOL
    assert abs(forward.ki_ever_probability - stacked.ki_ever_probability) < KI_PROB_ATOL
```

Run: `.venv/bin/python -m pytest test/test_quad_forward_density_stats.py -q`
Expected: all pass. Then run the stacked-mode golden gate (the dispatch and
any hoist must not perturb the default): `.venv/bin/python
docs/autocall-engine-perf/demos/capture_stats_goldens.py --check` → `BITWISE OK`.

- [ ] **Step 6: Commit**

```bash
git add quantark/asset/equity/engine/quad/snowball_quad_engine.py test/test_quad_forward_density_stats.py
git commit -m "feat(quad): forward-density event quantities for discrete products"
```

---

### Task 6: Continuous-KI forward bridge + first-passage gate

**Files:**
- Modify: `quantark/asset/equity/engine/quad/snowball_quad_engine.py` (only if Step 2 reveals fixes needed — the bridge path was written in Task 5)
- Test: `test/test_quad_forward_density_stats.py`

**Interfaces:** none new — this task certifies the continuous-KI path.

- [ ] **Step 1: Write the failing analytic first-passage test**

Constant vol, continuous down-barrier, no KO interference (KI stream only):
`P(min S_t <= B, t <= T)` for GBM has the closed form
`N((ln(B/S)-mT)/(σ√T)) + (B/S)^(2m/σ²) · N((ln(B/S)+mT)/(σ√T))`, `m = r−q−σ²/2`.

```python
def test_forward_ki_ever_matches_analytic_first_passage():
    product = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.0,
        ko_barrier=1e6,  # unreachable: isolates the first-passage statistic
        ki_barrier=80.0, ko_rate=0.15, num_observations=12,
        contract_multiplier=1.0,
    )
    e = _env()
    forward = SnowballQuadEngine(
        params=QuadParams(grid_points=2001, event_stats_mode="forward_density")
    ).calculate_event_stats(product, e)
    S, B, sig = 100.0, 80.0, 0.20
    m = 0.03 - 0.05 - 0.5 * sig * sig
    T = product.get_maturity(e)
    x = math.log(B / S)
    p_touch = norm.cdf((x - m * T) / (sig * math.sqrt(T))) + (
        B / S
    ) ** (2.0 * m / (sig * sig)) * norm.cdf((x + m * T) / (sig * math.sqrt(T)))
    assert abs(forward.ki_ever_probability - p_touch) < 5e-3  # provisional


def test_forward_matches_stacked_continuous_ki():
    product = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0,
    )
    stacked, forward = _stats_pair(SnowballQuadEngine, product, _env())
    assert np.max(np.abs(forward.ko_probability - stacked.ko_probability)) < KO_PROB_ATOL
    assert abs(forward.ki_probability - stacked.ki_probability) < KI_PROB_ATOL
    assert abs(forward.ki_ever_probability - stacked.ki_ever_probability) < KI_PROB_ATOL
    assert float(forward.pv).hex() == float(stacked.pv).hex()
```

(If the unreachable-KO construction trips `filter_unreachable_barriers`
validation, set `ko_barrier=200.0` instead and keep the tolerance — KO mass
at 10σ is negligible against 5e-3.)

- [ ] **Step 2: Run, diagnose, fix**

Run: `.venv/bin/python -m pytest test/test_quad_forward_density_stats.py -k "continuous or first_passage" -q`
Expected on first run: possibly FAIL. The two known candidate defects, in
order of likelihood: (i) bridge-kernel drift orientation — the forward call
passes `-alpha` to `_bridge_kernels`; if the first-passage test overshoots
systematically, try `+alpha` (the kernel's `omega` drift factor must match
`_forward_kernel`'s orientation; the analytic test is the authority);
(ii) `touched` accumulated on target nodes must use the same `prefactor_fwd`
as transport. Fix only in `_forward_event_quantities`; re-run until green.

- [ ] **Step 3: Add the mass-conservation diagnostic test**

```python
def test_forward_mass_diagnostic_conserved():
    product = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0,
    )
    engine = SnowballQuadEngine(
        params=QuadParams(grid_points=1001, event_stats_mode="forward_density")
    )
    engine.calculate_event_stats(product, _env())
    # Genuine conservation check: terminal integral of the marched densities
    # plus the absorbed KO mass (stored by the dispatch as a diagnostic; the
    # survival/ko fields cannot test this — survival is DEFINED as
    # 1 - cumulative KO, so any identity built from them is a tautology).
    assert abs(engine._last_forward_mass_diagnostic - 1.0) < 1e-4  # provisional
```

Run: `.venv/bin/python -m pytest test/test_quad_forward_density_stats.py -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add quantark/asset/equity/engine/quad/snowball_quad_engine.py test/test_quad_forward_density_stats.py
git commit -m "feat(quad): certify forward continuous-KI bridge against analytic first passage"
```

---

### Task 7: Phoenix forward hooks

**Files:**
- Modify: `quantark/asset/equity/engine/quad/phoenix_quad_engine.py`
- Modify: `quantark/asset/equity/engine/quad/phoenix_quad_engine.py` — also extract `_coupon_cashflows_from_probability` from the deterministic tail of `_extract_extra_quad_stats` (code motion) so both modes share it
- Test: `test/test_quad_forward_density_stats.py`

**Interfaces:**
- Consumes: Task 5 hook signatures.
- Produces: `PhoenixQuadEngine._coupon_cashflows_from_probability(coupon_probability, ko_records, product, pricing_env, rate, df_fn) -> np.ndarray` (used by `_extract_extra_quad_stats` and `_forward_extra_fields`).

- [ ] **Step 1: Write the failing Phoenix parity test**

```python
def test_forward_matches_stacked_phoenix():
    from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
    from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix

    product = create_standard_phoenix(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, coupon_barrier=85.0, coupon_rate=0.01,
        num_observations=23,
    )
    stacked, forward = _stats_pair(PhoenixQuadEngine, product, _env())
    assert np.max(np.abs(forward.ko_probability - stacked.ko_probability)) < KO_PROB_ATOL
    assert np.max(np.abs(forward.coupon_probability - stacked.coupon_probability)) < KO_PROB_ATOL
    np.testing.assert_allclose(
        forward.expected_discounted_coupon_cashflow,
        stacked.expected_discounted_coupon_cashflow,
        rtol=CF_RTOL, atol=1e-4,
    )
    assert float(forward.pv).hex() == float(stacked.pv).hex()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest test/test_quad_forward_density_stats.py::test_forward_matches_stacked_phoenix -q`
Expected: FAIL (forward `coupon_probability` all zeros — Snowball no-op hooks).

- [ ] **Step 3: Extract the deterministic coupon-cashflow block (code motion)**

In `phoenix_quad_engine.py`, move the `if not product.has_memory_coupon:`
body of `_extract_extra_quad_stats` into:

```python
    def _coupon_cashflows_from_probability(
        self, coupon_probability, ko_records, product, pricing_env, rate,
        df_fn,
    ):
        """E[DF*amount*1{coupon}] = DF(0->settle) * amount * P(coupon) for
        deterministic (non-memory) coupon amounts. Shared by the stacked
        extraction and the forward readout."""
        n_ko = len(ko_records)
        ko_times = [float(rec.observation_time) for rec in ko_records]
        period_yf = product.get_coupon_period_year_fractions(ko_times)
        coupon_amounts = [
            float(product.get_coupon_payoff(i, year_fraction=period_yf[i]))
            for i in range(n_ko)
        ]
        payment_timings = resolve_structured_payment_timings(
            product, pricing_env, ko_records,
        )
        expiry = product.coupon_config.coupon_pay_type == CouponPayType.EXPIRY
        ecc = np.zeros(n_ko, dtype=float)
        for i in range(n_ko):
            settle = (
                float(payment_timings.terminal.payment_time)
                if expiry
                else float(payment_timings.event_payment_times[i])
            )
            df_settle = (
                float(df_fn(settle)) if df_fn is not None
                else math.exp(-rate * settle)
            )
            ecc[i] = float(
                df_settle * coupon_amounts[i] * coupon_probability[i]
            )
        return ecc
```

and have `_extract_extra_quad_stats` call it. Run the Task 1 golden check
(`--check`) — the phoenix case must stay `BITWISE OK`.

- [ ] **Step 4: Implement the Phoenix forward hooks**

```python
    def _forward_extra_state(self, n_ko: int):
        return np.zeros(n_ko, dtype=float)

    def _forward_extra_readouts_at_obs(
        self, state, ko_index, ko_record, p_alive, math_utils,
        smoothing_width, product,
    ) -> None:
        barrier = self._coupon_barrier_for_obs(product, ko_index)
        pay_w = self._event_weight(
            math_utils.grid, barrier, math_utils.spot, smoothing_width,
            trigger_is_down=product.is_reverse,
        )
        state[ko_index] = self._density_integral(math_utils, pay_w * p_alive)

    def _forward_extra_fields(
        self, state, ko_records, product, pricing_env, rate, df_local
    ) -> dict:
        result = {"coupon_probability": np.asarray(state, dtype=float)}
        if not product.has_memory_coupon:
            result["expected_discounted_coupon_cashflow"] = (
                self._coupon_cashflows_from_probability(
                    result["coupon_probability"], ko_records, product,
                    pricing_env, rate, df_local,
                )
            )
        return result
```

(`math_utils.spot` — confirm the attribute name on `QuadratureMath`; the
stacked `_set_extra_quad_indicators` passes `math_utils.grid, math_utils.spot`
in the same shape. If it is stored differently, thread `spot` through the
hook signature instead — update Task 5's call site accordingly.)

- [ ] **Step 5: Run tests + gates**

Run: `.venv/bin/python -m pytest test/test_quad_forward_density_stats.py -q`
Expected: all pass.
Run: `.venv/bin/python docs/autocall-engine-perf/demos/capture_stats_goldens.py --check`
Expected: `BITWISE OK`.

- [ ] **Step 6: Commit**

```bash
git add quantark/asset/equity/engine/quad/phoenix_quad_engine.py test/test_quad_forward_density_stats.py
git commit -m "feat(quad): Phoenix coupon readouts on the forward density"
```

---

### Task 8: Matrix and contract coverage

**Files:**
- Modify: `quantark/asset/equity/engine/quad/ko_reset_snowball_quad_engine.py` (docstring only)
- Test: `test/test_quad_forward_density_stats.py`

**Interfaces:** none new.

- [ ] **Step 1: Write the matrix tests (each is small — same `_stats_pair` pattern)**

```python
def test_forward_reverse_snowball_parity():
    product = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=97.0,
        ki_barrier=125.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0, is_reverse=True,
    )
    stacked, forward = _stats_pair(SnowballQuadEngine, product, _env())
    assert np.max(np.abs(forward.ko_probability - stacked.ko_probability)) < KO_PROB_ATOL
    assert abs(forward.ki_probability - stacked.ki_probability) < KI_PROB_ATOL


def test_forward_disable_ko_after_ki_parity():
    product = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0, disable_ko_after_ki=True,
    )
    stacked, forward = _stats_pair(SnowballQuadEngine, product, _env())
    assert np.max(np.abs(forward.ko_probability - stacked.ko_probability)) < KO_PROB_ATOL
    assert abs(forward.ki_probability - stacked.ki_probability) < KI_PROB_ATOL


def test_forward_knocked_in_at_valuation_latches():
    product = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0,
    )
    e = PricingEnvironment(
        spot_quote=SpotQuote(spot=74.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.05),
        valuation_date=datetime(2026, 6, 30),
    )
    stacked, forward = _stats_pair(SnowballQuadEngine, product, e)
    assert forward.ki_probability == 1.0 == stacked.ki_probability
    assert forward.ki_ever_probability == 1.0
    assert np.max(np.abs(forward.ko_probability - stacked.ko_probability)) < KO_PROB_ATOL


def test_forward_r_q_zero_bisection():
    product = _discrete_ki_snowball()
    e = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.0),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 6, 30),
    )
    stacked, forward = _stats_pair(SnowballQuadEngine, product, e)
    assert np.max(np.abs(forward.ko_probability - stacked.ko_probability)) < KO_PROB_ATOL
    assert abs(forward.ki_probability - stacked.ki_probability) < KI_PROB_ATOL


def test_forward_streams_pruning_contract():
    from quantark.cashleg.event_distribution import EventType

    product = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0,
    )
    ko_only = frozenset({EventType.KO, EventType.MATURITY_NO_KO})
    full_engine = SnowballQuadEngine(
        params=QuadParams(grid_points=1001, event_stats_mode="forward_density")
    )
    full = full_engine.calculate_event_stats(product, _env())
    pruned_engine = SnowballQuadEngine(
        params=QuadParams(grid_points=1001, event_stats_mode="forward_density")
    )
    pruned = pruned_engine.calculate_event_stats(
        product, _env(), streams=ko_only
    )
    # Same-run exact equality on the surviving fields; pruned fields zero.
    assert np.asarray(full.ko_probability).tobytes() == np.asarray(pruned.ko_probability).tobytes()
    assert float(full.pv).hex() == float(pruned.pv).hex()
    assert pruned.ki_probability == 0.0
    assert pruned.ki_ever_probability == 0.0


def test_forward_pwe_npv_bit_equal_to_stacked():
    product = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0,
    )
    e = _env()
    npv_stacked = SnowballQuadEngine(
        params=QuadParams(grid_points=1001)
    ).price_with_events(product, e).npv
    npv_forward = SnowballQuadEngine(
        params=QuadParams(grid_points=1001, event_stats_mode="forward_density")
    ).price_with_events(product, e).npv
    assert float(npv_stacked).hex() == float(npv_forward).hex()


def test_ko_reset_ignores_forward_flag():
    # KO-reset keeps its own stacked implementation; the flag is a
    # permission, not an obligation (same-run exact equality).
    from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import (
        KOResetSnowballQuadEngine,
    )
    from dataclasses import fields as dc_fields

    # Reuse whatever KO-reset product an existing test constructs:
    from test.test_ko_reset_snowball_quad import make_product  # adjust to the
    # actual fixture name found in the existing KO-reset quad test file; if a
    # factory does not exist, copy the product construction from that file.

    product = make_product()
    e = _env()
    a = KOResetSnowballQuadEngine(
        params=QuadParams(grid_points=1001)
    ).calculate_event_stats(product, e)
    b = KOResetSnowballQuadEngine(
        params=QuadParams(grid_points=1001, event_stats_mode="forward_density")
    ).calculate_event_stats(product, e)
    for f in dc_fields(a):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if isinstance(va, np.ndarray):
            assert va.tobytes() == vb.tobytes(), f.name
        else:
            assert va == vb, f.name
```

Also add a cell-average-mode parity test: construct the snowball of
`test_forward_matches_stacked_continuous_ki` with the engine-params setting
that makes `_use_cell_average_events()` return True (read that method to find
the governing `QuadParams` field — it is the event-projection knob), and
assert the same KO/KI tolerances between stacked and forward.

- [ ] **Step 2: Run, fix, iterate**

Run: `.venv/bin/python -m pytest test/test_quad_forward_density_stats.py -q`
Expected: failures localize to specific semantics (reverse-direction masks,
`disable_ko_after_ki` absorption, projection adjoint). Fix in
`_forward_event_quantities` only. The cell-average identity to verify if the
projection parity fails: `_project_quad_event` applied to
`(v_survive=p, v_breach=0)` must act per-node (diagonal) for the mass
bookkeeping to be its own adjoint — probe with unit basis vectors at the
nodes flanking the barrier; if it redistributes across nodes, build the
local transpose from those probes and apply it instead.

- [ ] **Step 3: KO-reset docstring**

Extend the existing `calculate_event_stats` comment in
`ko_reset_snowball_quad_engine.py`: `"...always computes the full
distribution, and ``event_stats_mode='forward_density'`` is likewise
ignored."`

- [ ] **Step 4: Full-suite gate**

Run: `.venv/bin/python -m pytest test/ -q --deselect test/mo_volmodels/test_pde_convergence_gate.py::test_quick_end_to_end`
Expected: no new failures vs the known-pre-existing mo_volmodels artifact set.
Run: `.venv/bin/python docs/autocall-engine-perf/demos/capture_stats_goldens.py --check`
Expected: `BITWISE OK`.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/quad/ko_reset_snowball_quad_engine.py test/test_quad_forward_density_stats.py
git commit -m "test(quad): forward-density matrix + contract coverage"
```

---

### Task 9: MC cross-check + tolerance pilot bank

**Files:**
- Test: `test/test_quad_forward_density_stats.py` (slow-marked MC test)
- Create: `docs/autocall-engine-perf/FORWARD-DENSITY-EVIDENCE-2026-08.md` (pilot numbers; extended by Task 11)

**Interfaces:** none new.

- [ ] **Step 1: Write the MC cross-check test**

```python
@pytest.mark.slow
def test_forward_matches_mc_event_stats():
    from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
    from quantark.asset.equity.param import MCParams

    product = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0,
    )
    e = _env()
    forward = SnowballQuadEngine(
        params=QuadParams(grid_points=2001, event_stats_mode="forward_density")
    ).calculate_event_stats(product, e)
    mc = SnowballMCEngine(
        params=MCParams(num_paths=500_000, time_steps=479, use_qmc=True, seed=7)
    ).calculate_event_stats(product, e)
    # 3-sigma band on a 500k-path binomial proportion is ~0.002 absolute.
    assert np.max(np.abs(forward.ko_probability - mc.ko_probability)) < 4e-3
    assert abs(forward.ki_ever_probability - mc.ki_ever_probability) < 6e-3
```

(Adjust the MC field names to what `SnowballMCEngine.calculate_event_stats`
returns — same dataclass family; run the stacked engine against MC first if
in doubt about the field conventions, since stacked-vs-MC agreement is the
established baseline.)

- [ ] **Step 2: Run and record**

Run: `.venv/bin/python -m pytest test/test_quad_forward_density_stats.py -k mc_event -q`
Expected: pass (minutes-scale; keep the `slow` marker).

- [ ] **Step 3: Tolerance pilot**

Write a scratchpad script that runs every parity test's product at
`grid_points in (1001, 2001, 4001)` and prints `max|forward − stacked|` per
field. Bank the table in
`docs/autocall-engine-perf/FORWARD-DENSITY-EVIDENCE-2026-08.md` under a
"Pilot convergence" heading, then **tighten the provisional constants**
(`KO_PROB_ATOL`, `KI_PROB_ATOL`, `CF_RTOL`, `MASS_TOL`, the first-passage and
mass-diagnostic bounds) to ~2× the measured grid-2001 deltas, and note each
final value in the evidence doc. Contraction from 1001 → 2001 → 4001 must be
visible for every field; a non-contracting field is a defect — stop and
root-cause before proceeding (systematic-debugging skill).

- [ ] **Step 4: Re-run the whole battery + commit**

Run: `.venv/bin/python -m pytest test/test_quad_forward_density_stats.py -q`
Expected: all pass with the tightened constants.

```bash
git add test/test_quad_forward_density_stats.py
git add -f docs/autocall-engine-perf/FORWARD-DENSITY-EVIDENCE-2026-08.md
git commit -m "test(quad): MC cross-check + banked forward-density tolerance pilot"
```

---

### Task 10: Documentation

**Files:**
- Modify: `CHANGELOG.md` (Unreleased section — note: the file carries
  unrelated staged-elsewhere WIP; add ONLY the new entry, do not touch other
  hunks; commit with `git add -p CHANGELOG.md` selecting only this hunk)
- Modify: `docs/autocall-engine-perf/SOLUTIONS-2026-08-24.md` (mark F2/F3 delivered, link the evidence doc)
- Modify: `quantark/asset/equity/CLAUDE.md` (one line in the Parameters section: `QuadParams(event_stats_mode="forward_density")  # opt-in forward-density event stats`)

**Interfaces:** none.

- [ ] **Step 1: Write the CHANGELOG entry**

Under `## [Unreleased]`:

```markdown
### Added
- QUAD autocallable engines: opt-in `QuadParams.event_stats_mode="forward_density"`
  computes the event distribution from a forward transition-density march
  (2–3 surfaces instead of one indicator row per KO observation). `npv` is
  unchanged (always the backward value solve); distribution fields differ
  within the banked validation tolerances
  (`docs/autocall-engine-perf/FORWARD-DENSITY-EVIDENCE-2026-08.md`).
  KO-reset ignores the flag. Default remains `"stacked"`.

### Changed
- QUAD stacked event stats: the KI-probability recursion now rides the main
  stacked recursion as fused rows (bit-identical output, one fewer full time
  loop per `price_with_events`).
```

- [ ] **Step 2: Update SOLUTIONS doc**

Append to the "Landed" section: F2 (as Track 1 fusion) and F3 (flagged
forward-density) delivered, with the Task 9/11 measured numbers and a
pointer to the spec + evidence doc.

- [ ] **Step 3: Commit (CHANGELOG hunk only + docs)**

```bash
git add -p CHANGELOG.md   # select ONLY the new entry hunk
git add -f docs/autocall-engine-perf/SOLUTIONS-2026-08-24.md quantark/asset/equity/CLAUDE.md
git commit -m "docs(quad): forward-density event-stats mode + fusion notes"
```

---

### Task 11: Full-book adapter A/B (battery gate e)

**Files:**
- Create: `docs/autocall-engine-perf/demos/run_book_forward_density.md` (protocol notes; the driver lives outside this repo's package code)
- Modify: `docs/autocall-engine-perf/FORWARD-DENSITY-EVIDENCE-2026-08.md` (results)

**Interfaces:** none — external measurement.

- [ ] **Step 1: Build the flag driver**

The adapter (`/Users/fuxinyao/otc-price-adapter/otc_quantark_pricer_v047.py`)
constructs `QuadParams` internally and knows nothing of the flag. Drive the
flag from outside via a `sitecustomize.py` placed on the `PYTHONPATH` next to
the shadowing tree (workers inherit it):

```python
# scratchpad/fwdflag/sitecustomize.py
"""Force forward_density event stats for the book A/B (measurement shim)."""
import quantark.asset.equity.param.engine_params as ep

_orig = ep.QuadParams.__post_init__

def _patched(self):
    object.__setattr__(self, "event_stats_mode", "forward_density")
    _orig(self)

ep.QuadParams.__post_init__ = _patched
```

(`QuadParams` is a plain dataclass — if it is not frozen, a simple attribute
assignment before `_orig(self)` also works; check and use the simpler form.)

- [ ] **Step 2: Run the A/B**

Protocol identical to 2026-08-24 (`SOLUTIONS-2026-08-24.md` §Full-book A/B):
one detached worktree at the branch tip; adapter venv;
`--as-of-date 2026-06-30 --workers 8`; QUAD config; arms back-to-back in one
window: (arm 1) tree without the shim = stacked, (arm 2) tree +
`PYTHONPATH=<tree>:<scratchpad>/fwdflag` = forward. Re-run any anomalous arm
in a fresh window before believing it (shared-host rule). Then compare with
the existing comparison pattern (`compare_ab.py` from the 2026-08-24 session,
or rewrite from `SOLUTIONS-2026-08-24.md`): the **npv column must be exactly
equal**; leg-sum/greek columns are quantified (max abs and bp-of-notional
per row), and wall times recorded.

- [ ] **Step 3: Bank the evidence + acceptance check**

Record in `FORWARD-DENSITY-EVIDENCE-2026-08.md`: wall times (stacked vs
forward vs the 111 s PDE / 230 s MC references), npv equality, leg/greek
diff distribution, status counts. Acceptance (spec §9): forward-mode book
wall in or credibly near 100–150 s. If it misses, profile one row
(`cProfile`, the `docs/autocall-engine-perf/demos/profile` pattern) before
any tuning — no speculative optimization.

- [ ] **Step 4: Commit**

```bash
git add -f docs/autocall-engine-perf/demos/run_book_forward_density.md docs/autocall-engine-perf/FORWARD-DENSITY-EVIDENCE-2026-08.md
git commit -m "docs(quad): bank forward-density full-book A/B evidence"
```
