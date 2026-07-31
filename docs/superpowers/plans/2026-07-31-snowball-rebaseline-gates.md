# Snowball Re-baseline Gates (G1 / G4 / G2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the snowball vol-model study's admission gates on the 0.4.0 engines with the §7A.4 fixes in place, and re-scope Gate G2 from a 2-variant PDE-vs-MC convergence check into a 6-variant engine-admission gate that tests delta as well as PV.

**Architecture:** Three phases against the existing `example/mo_volmodels/` stage scripts. Phase A (Tasks 1–2) re-establishes the study's inputs — surface admission is *verified* over the existing artifacts (never rebuilt), and the fair coupon is re-solved because 0.4.0 repriced the 1D PDE. Phase B (Tasks 3–7) rewrites Gate G2's scope inside `11_pde_convergence_gate.py`: a production-vs-reference pair table replacing the hard-coded 2-variant assumption, maturity-bucketed bias detection, delta admission expressed in IM contracts, and Feller-regime conditioning. Phase C (Task 8) runs the gate and emits a fresh `gate_decision.json` that stage 12 consumes.

**Tech Stack:** Python 3.11, numpy, pytest (`-n auto` by default; use `-n0` for serial debugging), `quantark.*` canonical imports, the project venv at `.venv/`.

## Global Constraints

- Source spec: `docs/superpowers/specs/2026-07-30-snowball-volmodel-backtest-040-rebaseline-design.md`. Every section reference below (§5, §7A.10, §9, §11) points there.
- Run everything with `.venv/bin/python` or after `source .venv/bin/activate`.
- Canonical imports only: `quantark.*`. Never `backtest.otc.*` (0.5.0 shim) or bare flat names.
- Numerics: use `quantark.util.numerical` (`is_zero`, `is_close`, `safe_divide`, …). Never a raw float `==` or a hardcoded tolerance.
- Fail closed. No silent fallbacks, no default-on approximations, no repaired data. A gate that cannot evaluate a cell records an error and counts that cell as failed.
- Gate tolerance floor: `TOL_ABS_PCT = 0.25` (% of notional); per-cell tolerance is `max(2 * mc_se_pct, 0.25)`.
- Study term sheet: `NOTIONAL = 50_000_000.0`, `FUTURES_MULTIPLIER = 200.0`, 3Y maturity, `KO_PCT = 1.03`, `KI_PCT = 0.75`, 3-month lockout.
- The gate prices a **1-unit** product (`contract_multiplier = 1.0`, `notional = s0`). The backtest prices `contract_multiplier = NOTIONAL / s0`. Any tolerance derived from hedge instrument size must convert between the two — see Task 6.
- Do **not** regenerate `example/mo_volmodels/data/history/`. The Phase-1 builder still uses `min_expiries: 2`, so a rebuild would re-admit the two thin surfaces (2024-09-30, 2025-04-08) that `exclude_thin_surfaces.py` removed.
- Commit after every task. Branch: `fix/snowball-rebaseline-7a4-engine-fixes` (or a descendant).

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `example/mo_volmodels/13_gate_g1_surface_admission.py` | G1: verify every artifact carries a passing admission record | **create** |
| `example/mo_volmodels/12_snowball_volmodel_backtest.py` | G4 coupon solve; adds `flat_bsm_quad` variant; consumes 6-variant routing | modify |
| `example/mo_volmodels/11_pde_convergence_gate.py` | G2: variant pair table, bucketed bias, delta admission, Feller conditioning | modify |
| `test/mo_volmodels/test_gate_g1_admission.py` | G1 verifier unit tests | **create** |
| `test/mo_volmodels/test_gate_scope.py` | G2 re-scope unit tests (pure functions only) | **create** |

`11_pde_convergence_gate.py` is already ~1600 lines. Do not restructure it wholesale — this plan adds pure functions near their existing neighbours and rewires `process_date`. If a task's diff exceeds ~150 lines in that file, stop and flag it rather than improvising a split.

---

## Task 1: G1 — verify surface admission over the existing artifacts

**Files:**
- Create: `example/mo_volmodels/13_gate_g1_surface_admission.py`
- Create: `test/mo_volmodels/test_gate_g1_admission.py`

**Interfaces:**
- Consumes: `quantark.param.vol.surface_history.IvSurfaceArtifact`
- Produces: `verify_admission(payload: dict) -> tuple[bool, str]` — `(ok, reason)`; `scan_history(iv_dir: Path) -> dict` with keys `n_scanned`, `n_admitted`, `failures` (list of `{date, reason}`), `min_expiries_seen`.

**Why this is a verifier and not a rebuild:** the §7A.4 fixes touched calibration and the 2D PDE, neither of which builds surfaces. G1's job here is to confirm the 762 artifacts the fleet will consume each carry a passing admission record, and that none has fewer than 3 expiries (Dupire's requirement). Rebuilding would re-admit the two thin surfaces — see Global Constraints.

- [ ] **Step 1: Write the failing test**

```python
# test/mo_volmodels/test_gate_g1_admission.py
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load():
    path = REPO / "example" / "mo_volmodels" / "13_gate_g1_surface_admission.py"
    spec = importlib.util.spec_from_file_location("g1", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["g1"] = mod          # @dataclass resolves cls.__module__ here
    spec.loader.exec_module(mod)
    return mod


def test_admitted_surface_with_three_expiries_passes():
    g1 = _load()
    ok, reason = g1.verify_admission(
        {"admission": {"admitted": True}, "maturities": [0.1, 0.3, 0.6]}
    )
    assert ok is True
    assert reason == ""


def test_rejected_surface_fails_with_its_recorded_reason():
    g1 = _load()
    ok, reason = g1.verify_admission(
        {"admission": {"admitted": False, "reason": "static_arbitrage"},
         "maturities": [0.1, 0.3, 0.6]}
    )
    assert ok is False
    assert "static_arbitrage" in reason


def test_thin_surface_fails_even_when_marked_admitted():
    """The Phase-1 builder admits 2-expiry surfaces; Dupire needs 3."""
    g1 = _load()
    ok, reason = g1.verify_admission(
        {"admission": {"admitted": True}, "maturities": [0.1, 0.3]}
    )
    assert ok is False
    assert "expiries" in reason


def test_missing_admission_record_fails_closed():
    g1 = _load()
    ok, reason = g1.verify_admission({"maturities": [0.1, 0.3, 0.6]})
    assert ok is False
    assert "admission" in reason
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_g1_admission.py -n0 -q`
Expected: FAIL — `FileNotFoundError` / `spec_from_file_location` returns None for the missing script.

- [ ] **Step 3: Write the script**

```python
# example/mo_volmodels/13_gate_g1_surface_admission.py
"""Gate G1: verify surface admission over the EXISTING IV-surface artifacts.

This is a verifier, not a builder.  The Phase-1 history builder still uses
min_expiries=2, so regenerating the history would re-admit the two thin
surfaces (2024-09-30, 2025-04-08) that exclude_thin_surfaces.py removed.
G1 therefore reads what the fleet will actually consume and fails closed on
anything that is not admissible.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IV_DIR = PROJECT_ROOT / "example/mo_volmodels/data/history/iv_surface"

# Dupire local vol needs at least three expiries to form dw/dT; the Phase-1
# builder admits two, so G1 re-checks rather than trusting the flag alone.
MIN_EXPIRIES = 3


def verify_admission(payload: Dict[str, Any]) -> Tuple[bool, str]:
    """Return (ok, reason) for one artifact payload.  Empty reason iff ok."""
    admission = payload.get("admission")
    if not isinstance(admission, dict):
        return False, "missing 'admission' record"
    if admission.get("admitted") is not True:
        return False, f"not admitted: {admission.get('reason', 'no reason recorded')}"
    maturities = payload.get("maturities") or []
    if len(maturities) < MIN_EXPIRIES:
        return False, (
            f"{len(maturities)} expiries < {MIN_EXPIRIES} required by Dupire"
        )
    return True, ""


def scan_history(iv_dir: Path) -> Dict[str, Any]:
    """Verify every artifact in ``iv_dir``; returns a JSON-safe summary."""
    failures = []
    n_scanned = 0
    min_expiries_seen = None
    for path in sorted(Path(iv_dir).glob("*.json")):
        payload = json.loads(path.read_text())
        n_scanned += 1
        n_exp = len(payload.get("maturities") or [])
        min_expiries_seen = n_exp if min_expiries_seen is None else min(min_expiries_seen, n_exp)
        ok, reason = verify_admission(payload)
        if not ok:
            failures.append({"date": payload.get("trade_date"), "reason": reason})
    return {
        "gate": "G1",
        "iv_dir": str(iv_dir),
        "n_scanned": n_scanned,
        "n_admitted": n_scanned - len(failures),
        "failures": failures,
        "min_expiries_seen": min_expiries_seen,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iv-dir", default=str(DEFAULT_IV_DIR))
    parser.add_argument("--out", default="output/gate_g1_admission.json")
    args = parser.parse_args(argv)

    summary = scan_history(Path(args.iv_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))

    print(f"[G1] scanned {summary['n_scanned']}, "
          f"admitted {summary['n_admitted']}, "
          f"min expiries {summary['min_expiries_seen']}")
    for f in summary["failures"][:20]:
        print(f"  FAIL {f['date']}: {f['reason']}")
    if summary["failures"]:
        print(f"[G1] FAILED — {len(summary['failures'])} surface(s) not admissible")
        return 1
    print("[G1] PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_g1_admission.py -n0 -q`
Expected: PASS, 4 tests.

- [ ] **Step 5: Run G1 for real**

Run: `.venv/bin/python example/mo_volmodels/13_gate_g1_surface_admission.py`
Expected: `[G1] scanned 762, admitted 762, min expiries 3` then `[G1] PASSED`, exit 0.

If any surface fails, **stop and report** — do not edit the history. A failure means the excluded-thin-surface filter has drifted or an artifact was regenerated.

- [ ] **Step 6: Commit**

```bash
git add example/mo_volmodels/13_gate_g1_surface_admission.py test/mo_volmodels/test_gate_g1_admission.py
git commit -m "feat(mo): add Gate G1 surface-admission verifier over existing artifacts"
```

---

## Task 2: G4 — re-solve the fair coupon on the 0.4.0 engines

**Files:**
- Modify: none (uses `12_snowball_volmodel_backtest.py` as-is)

**Interfaces:**
- Consumes: `solve_fair_coupon` (`12_snowball_volmodel_backtest.py:524`)
- Produces: `output/volmodel_backtest/inception_coupons.json` (or whatever the runner's prepare step writes) — the per-inception coupon table every later phase depends on.

**Why re-run:** the coupon is solved on `flat_bsm` — a **1D BSM PDE** (`12_snowball_volmodel_backtest.py:1001`). The §7A.4 fixes touch the Heston preset and the 2D `v0_boundary`, so they do **not** move these roots. 0.4.0's PDE grid rewrite does: the spec records 15.0975% → 15.0707% on the first inception. This is a spec correction — §10 implies G4 re-runs *because of* §7A.4; it re-runs because of 0.4.0, and it has not been run since.

- [ ] **Step 1: Find the coupon-only entry point**

Run: `.venv/bin/python example/mo_volmodels/12_snowball_volmodel_backtest.py --help`

Identify the flag that stops after the prepare/coupon phase. If none exists, run with `--variants flat_bsm --limit 27` and take the coupon table from the run manifest.

- [ ] **Step 2: Solve coupons for all 27 inceptions**

Run (background; ~3 PDE prices × 27 inceptions, expect 20–40 min):

```bash
.venv/bin/python example/mo_volmodels/12_snowball_volmodel_backtest.py \
  --variants flat_bsm --workers 4 2>&1 | tee output/g4_coupons.log
```

Expected per line: `[k/27] YYYY-MM-DD s0=… coupon=15.xxxx% |PV|=… (N iters, …s)`.

- [ ] **Step 3: Verify G4's fail-closed contract held**

Every inception must appear with a converged coupon. `solve_fair_coupon` raises rather than returning an unconverged or boundary value, so a completed run **is** the gate passing. Confirm explicitly:

```bash
grep -c "coupon=" output/g4_coupons.log     # expect 27
grep -E "coupon=(0\.0000|80\.0000)%" output/g4_coupons.log   # expect no matches (bounds)
```

- [ ] **Step 4: Record the coupon table in the spec**

Add a short table to §2 of the spec: inception, `s0`, solved coupon, `|PV|`. State the first inception's coupon against the recorded 0.3.0 value (15.0975%) and 0.4.0 value (15.0707%) so the re-baseline is auditable.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-07-30-snowball-volmodel-backtest-040-rebaseline-design.md
git commit -m "docs(mo): record the G4 fair-coupon re-solve on 0.4.0"
```

---

## Task 3: Add the `flat_bsm_quad` engine-control variant

**Files:**
- Modify: `example/mo_volmodels/12_snowball_volmodel_backtest.py:135-141` (`VARIANTS`), `:144-153` (`VariantSpec`), `:158-194` (`VARIANT_SPECS`), `:669-705` (`make_engine_config`)

**Interfaces:**
- Produces: `"flat_bsm_quad"` in `VARIANTS`; a new `VariantSpec.pricing_engine_type: EngineType = EngineType.PDE` field; and `VARIANT_SPECS["flat_bsm_quad"]` identical to `flat_bsm` except `pricing_engine_type=EngineType.QUADRATURE`.

**The distinguishing field does not exist yet.** `VariantSpec` (`:144`) carries only `name`, `vol_source`, `surface_vol_mode`, `vol_model`, `description` — the solver is picked by `GateRouting.solver_for(spec.vol_model)`, which routes *vol-model* engines only. `make_engine_config` hardcodes `pricing_engine_type=EngineType.PDE` for every variant (`:691`). So `flat_bsm_quad` needs a new field; it cannot be expressed with the current dataclass. `EngineType.QUADRATURE` and `AutocallableEngineConfig.quad_params` both already exist.

**Why this comes before G2:** the spec's §5.1 gates all six variants, but §10 adds `flat_bsm_quad` at step 3 — *after* G2 at step 2. That ordering is impossible: you cannot gate a variant that does not exist. This plan pulls the addition forward. Note the spec correction in the commit message.

`flat_bsm` and `flat_bsm_quad` reference each other in §5.1. That is not circular — it is one PDE-vs-QUAD comparison serving as the admission test for both routes and as the study's engine control.

- [ ] **Step 1: Write the failing test**

```python
# append to test/mo_volmodels/test_gate_scope.py
from quantark.util.enum.engine_enums import EngineType


def test_flat_bsm_quad_differs_from_flat_bsm_in_engine_only():
    """The engine control is only a control if the market data is identical."""
    s12 = _load_stage12()
    assert "flat_bsm_quad" in s12.VARIANTS
    bsm = s12.VARIANT_SPECS["flat_bsm"]
    quad = s12.VARIANT_SPECS["flat_bsm_quad"]
    assert quad.vol_source == bsm.vol_source
    assert quad.surface_vol_mode == bsm.surface_vol_mode
    assert quad.vol_model == bsm.vol_model == "bsm"
    assert bsm.pricing_engine_type == EngineType.PDE
    assert quad.pricing_engine_type == EngineType.QUADRATURE


def test_engine_config_honours_the_variant_pricing_engine_type():
    """A new VariantSpec field is inert until make_engine_config reads it."""
    s12 = _load_stage12()
    routing = s12.GateRouting("p", None, {}, {})
    cfg = s12.make_engine_config("flat_bsm_quad", routing=routing)
    assert cfg.pricing_engine_type == EngineType.QUADRATURE
    assert s12.make_engine_config(
        "flat_bsm", routing=routing
    ).pricing_engine_type == EngineType.PDE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_scope.py -n0 -q -k flat_bsm_quad`
Expected: FAIL — `"flat_bsm_quad" not in VARIANTS` (and `AttributeError` on `pricing_engine_type`).

- [ ] **Step 3: Add the field, the variant, and the wiring**

Add the field to `VariantSpec` (`:144`), defaulted so the five existing entries are unchanged:

```python
@dataclass(frozen=True)
class VariantSpec:
    """How one model variant configures the daily pricing engine."""

    name: str
    vol_source: str
    surface_vol_mode: str
    vol_model: str
    description: str
    # Engine family for the DAILY PRICING engine.  Defaults to PDE so the
    # five original variants are untouched; flat_bsm_quad overrides it to
    # make an engine control that differs from flat_bsm in engine only.
    # The surface and event-stats engines stay 1D PDE for every variant.
    pricing_engine_type: EngineType = EngineType.PDE

    def uses_calibration(self) -> bool:
        return self.vol_model != "bsm"
```

Add to `VARIANTS` (after `"flat_bsm"`) and to `VARIANT_SPECS`:

```python
    "flat_bsm_quad": VariantSpec(
        name="flat_bsm_quad",
        vol_source="surface",
        surface_vol_mode="flat_atm_remaining",
        vol_model="bsm",
        description=(
            "Engine control: flat_bsm's market data priced by FFT "
            "regime-switching quadrature instead of the 1D PDE"
        ),
        pricing_engine_type=EngineType.QUADRATURE,
    ),
```

In `make_engine_config` (`:691`), replace the hardcoded `pricing_engine_type=EngineType.PDE` with `pricing_engine_type=spec.pricing_engine_type`, and add `quad_params=QuadParams()` alongside `pde_params=PDEParams()` so the QUAD route has its settings. Import `QuadParams` from wherever `PDEParams` comes from and confirm the name with:

`.venv/bin/python -c "import dataclasses; from quantark.backtest.replay.config import AutocallableEngineConfig as C; print([(f.name, f.type) for f in dataclasses.fields(C) if 'param' in f.name])"`

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_scope.py -n0 -q -k flat_bsm_quad`
Expected: PASS.

- [ ] **Step 5: Smoke-price one inception on the new variant**

Run: `.venv/bin/python example/mo_volmodels/12_snowball_volmodel_backtest.py --variants flat_bsm_quad --limit 1`
Expected: completes without error. A QUAD failure here (unreachable barrier, dense KI) is a §9/G5 finding — record it, do not paper over it.

- [ ] **Step 6: Commit**

```bash
git add example/mo_volmodels/12_snowball_volmodel_backtest.py test/mo_volmodels/test_gate_scope.py
git commit -m "feat(mo): add flat_bsm_quad engine-control variant

Spec §5.1 gates six variants but §10 added this one after G2; a variant
cannot be gated before it exists, so the addition moves ahead of the gate."
```

---

## Task 4: Generalise G2 from 2 hard-coded variants to a production/reference pair table

**Files:**
- Modify: `example/mo_volmodels/11_pde_convergence_gate.py:88` (`VARIANTS`), `:460-494` (`_make_mc_engine` / `_make_pde_engine`), `:776` (`validate_decision_payload`), `:844` (the `for variant in VARIANTS` loop)
- Test: `test/mo_volmodels/test_gate_scope.py`

**Interfaces:**
- Produces: `GATE_PAIRS: dict[str, GatePair]` where `GatePair` is a frozen dataclass with fields `production: str`, `reference: str` (free-text labels for the record) and two builders `build_production(model, case) -> engine`, `build_reference(model, case) -> engine`. `process_date` consumes `GATE_PAIRS` instead of branching on `variant == VOL_MODEL_HESTON`.

The §5.1 table this encodes:

| variant | production | reference |
|---|---|---|
| `flat_bsm` | PDE 1D | QUAD (high `grid_points`) |
| `flat_bsm_quad` | QUAD | PDE 1D `accuracy="high"` |
| `ts_bsm` | QUAD | PDE 1D `accuracy="high"` |
| `localvol` | PDE 1D | `LocalVolSnowballMCEngine` (RQMC) |
| `heston` | gate-decided | `QESnowballMCEngine` (RQMC QE-M) |
| `heston_slv` | gate-decided | `HestonSLVQESnowballMCEngine` |

- [ ] **Step 1: Write the failing test**

```python
# test/mo_volmodels/test_gate_scope.py
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(stem):
    path = REPO / "example" / "mo_volmodels" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem.split("_")[0], path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_gate():
    return _load("11_pde_convergence_gate")


def _load_stage12():
    return _load("12_snowball_volmodel_backtest")


EXPECTED = {
    "flat_bsm", "flat_bsm_quad", "ts_bsm", "localvol", "heston", "heston_slv",
}


def test_gate_covers_every_study_variant():
    gate = _load_gate()
    assert set(gate.GATE_PAIRS) == EXPECTED
    assert set(gate.VARIANTS) == EXPECTED


def test_gate_variants_match_the_backtest_variants():
    """A variant the fleet runs but the gate never admitted is unrouted."""
    assert set(_load_gate().VARIANTS) == set(_load_stage12().VARIANTS)


def test_every_pair_uses_two_distinct_numerical_methods():
    gate = _load_gate()
    for name, pair in gate.GATE_PAIRS.items():
        assert pair.production != pair.reference, name


def test_mc_referenced_variants_are_exactly_the_ones_needing_std_error():
    """Only these three get a 2*mc_se tolerance term; the rest get the floor."""
    gate = _load_gate()
    mc_refs = {n for n, p in gate.GATE_PAIRS.items() if p.reference_is_mc}
    assert mc_refs == {"localvol", "heston", "heston_slv"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_scope.py -n0 -q`
Expected: FAIL — `AttributeError: module has no attribute 'GATE_PAIRS'`.

- [ ] **Step 3: Add the pair table**

Insert near the existing engine factories (`~:445`):

```python
@dataclass(frozen=True)
class GatePair:
    """One variant's production engine and its independent reference.

    ``production`` / ``reference`` are labels recorded in the evidence; the
    builders are what actually run.  The two must be different numerical
    methods, or the cell is a common-mode comparison and proves nothing.

    Builder signature: ``(model, grid) -> engine``.  ``model`` is the
    CalibratedVolModel for calibrated variants and ``None`` for the BSM ones;
    ``grid`` is the ladder level's knob tuple (see _production_grid below).
    ``reference_is_mc`` drives whether a std error is expected: a
    deterministic reference has ``mc_se = None`` and the flat TOL_ABS floor.
    """

    production: str
    reference: str
    build_production: Callable[..., Any]
    build_reference: Callable[..., Any]
    reference_is_mc: bool


def _bsm_pde(accuracy: str = "standard"):
    def build(model, grid):
        return SnowballPDESolver(PDEParams(accuracy=accuracy))
    return build


def _bsm_quad(grid_points: Optional[int] = None):
    def build(model, grid):
        params = QuadParams() if grid_points is None else QuadParams(grid_points=grid_points)
        return SnowballQuadEngine(params=params)
    return build


GATE_PAIRS: Dict[str, GatePair] = {
    "flat_bsm": GatePair(
        production="pde_1d", reference="quad_high",
        build_production=_bsm_pde("standard"),
        build_reference=_bsm_quad(grid_points=QUAD_REFERENCE_POINTS),
        reference_is_mc=False,
    ),
    "flat_bsm_quad": GatePair(
        production="quad", reference="pde_1d_high",
        build_production=_bsm_quad(),
        build_reference=_bsm_pde("high"),
        reference_is_mc=False,
    ),
    "ts_bsm": GatePair(
        production="quad", reference="pde_1d_high",
        build_production=_bsm_quad(),
        build_reference=_bsm_pde("high"),
        reference_is_mc=False,
    ),
    "localvol": GatePair(
        production="pde_1d_lv", reference="lv_mc_rqmc",
        build_production=lambda model, grid: LocalVolSnowballPDESolver(
            params=PDEParams(), local_vol_surface=model.local_vol_surface),
        build_reference=lambda model, grid: LocalVolSnowballMCEngine(
            local_vol_surface=model.local_vol_surface,
            params=_make_mc_params(MC_FULL, SEED),
            method=MonteCarloMethod.RANDOMIZED_QUASI),
        reference_is_mc=True,
    ),
    "heston": GatePair(
        production="pde_2d_adi", reference="qe_m_rqmc",
        build_production=lambda model, grid: _make_pde_engine(
            VOL_MODEL_HESTON, model, PDEParams(), grid),
        build_reference=lambda model, grid: _make_mc_engine(
            VOL_MODEL_HESTON, model, _make_mc_params(MC_FULL, SEED),
            MC_FULL["substeps_per_interval"]),
        reference_is_mc=True,
    ),
    "heston_slv": GatePair(
        production="pde_2d_adi_slv", reference="slv_qe_m_rqmc",
        build_production=lambda model, grid: _make_pde_engine(
            VOL_MODEL_HESTON_SLV, model, PDEParams(), grid),
        build_reference=lambda model, grid: _make_mc_engine(
            VOL_MODEL_HESTON_SLV, model, _make_mc_params(MC_FULL, SEED),
            MC_FULL["substeps_per_interval"]),
        reference_is_mc=True,
    ),
}

VARIANTS = tuple(GATE_PAIRS)
```

`QUAD_REFERENCE_POINTS` is a new module constant — set it to at least 4× the QUAD production default so the reference is genuinely finer, and record it in the decision payload's `config` block.

**Do not import stage 12 to reuse its engine factory.** Stage 12 already imports stage 11 (`stage11()`, `:196-218`) because stage 11 owns the certified term sheet; importing back would be a cycle. Stage 11 builds its own engines from `quantark.*`, exactly as it does today for PDE and MC. The cost is a drift risk between the two files' engine settings — mitigate it with the test below rather than with an import.

- [ ] **Step 3b: Guard against gate/fleet engine drift**

```python
def test_gate_prices_the_same_engine_family_the_fleet_will_run():
    """Stage 11 cannot import stage 12 (cycle), so assert the pairing instead."""
    gate, s12 = _load_gate(), _load_stage12()
    for name, spec in s12.VARIANT_SPECS.items():
        pair = gate.GATE_PAIRS[name]
        if spec.vol_model == "bsm":
            expected = "quad" if spec.pricing_engine_type.name == "QUADRATURE" else "pde_1d"
            assert pair.production.startswith(expected), name
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_scope.py -n0 -q`
Expected: PASS, 4 tests.

- [ ] **Step 5: Rewire `process_date` and the payload validator**

Replace the `variant == VOL_MODEL_HESTON` branches in `process_date` with `GATE_PAIRS[variant]` lookups. Keep the existing per-cell error handling verbatim — a reference failure must still mark every cell for that variant failed.

Two behaviours must be preserved for MC-referenced variants (`pair.reference_is_mc`) and *skipped* for deterministic ones:

- `mc_se` and `mc_se_pct` are `None` for a deterministic reference; the tolerance is then the flat `TOL_ABS_PCT`. Extend `gate_tolerance_pct` to accept `None` explicitly rather than passing `0.0`, so the record says "no MC error" instead of "zero MC error":

```python
def gate_tolerance_pct(mc_se_pct: Optional[float], tol_abs_pct: float = TOL_ABS_PCT) -> float:
    """Per-cell tolerance in % of notional: max(2 x MC SE, TOL_ABS).

    ``mc_se_pct=None`` means the reference is DETERMINISTIC (QUAD or a finer
    PDE), not that its error is zero -- the tolerance is then the flat floor.
    """
    if mc_se_pct is None:
        return float(tol_abs_pct)
    return max(MC_SE_FACTOR * float(mc_se_pct), float(tol_abs_pct))
```

  Also relax `validate_gate_payload`'s finiteness check (`:795`): it currently demands finite `mc_se` on any cell without an `error`, which every deterministic-reference cell would now trip. Require finite `mc_se` only when `reference_is_mc`.

- The refinement ladder applies to the *production* engine, and its knob differs by family. Add:

```python
# QUAD ladder over grid_points, and the 1D-PDE ladder over accuracy
# profiles.  Both mirror the 2D ladder's shape: coarse -> medium -> fine,
# with 'medium' the level the fleet would actually run.
QUAD_LADDER = {"coarse": 2048, "medium": 4096, "fine": 8192}
QUAD_REFERENCE_POINTS = 16384          # >= 4x the QUAD production default
PDE1D_LADDER = {"coarse": "fast", "medium": "standard", "fine": "high"}


def _production_grid(variant: str, level: str, T: float, quick: bool):
    """Ladder knob for the PRODUCTION engine of ``variant`` at ``level``.

    2D ADI ladders over (n_x, n_v, n_t); QUAD over grid_points; the 1D PDE
    over its accuracy profile.  Returns a JSON-safe dict stored unchanged in
    ``cell["grid"]``, so the evidence keeps ONE schema across families and a
    reader can always see which knob moved.
    """
    pair = GATE_PAIRS[variant]
    if pair.production.startswith("pde_2d"):
        entry = next(g for g in pde_ladder(T, quick) if g[0] == level)
        return {"kind": "adi_2d", "n_x": entry[1], "n_v": entry[2], "n_t": entry[3]}
    if pair.production.startswith("quad"):
        return {"kind": "quad", "grid_points": QUAD_LADDER[level]}
    return {"kind": "pde_1d", "accuracy": PDE1D_LADDER[level]}
```

  The builders in Step 3 must consume this dict rather than the bare tuple — update `_bsm_pde` / `_bsm_quad` to read `grid["accuracy"]` / `grid["grid_points"]`, and the 2D lambdas to pass `(grid["n_x"], grid["n_v"], grid["n_t"])` to `_make_pde_engine`. Verify the ladder is monotone before running anything expensive:

```python
def test_every_production_family_ladders_monotonically():
    gate = _load_gate()
    for variant in gate.VARIANTS:
        grids = [gate._production_grid(variant, lvl, 3.0, False)
                 for lvl in ("coarse", "medium", "fine")]
        kinds = {g["kind"] for g in grids}
        assert len(kinds) == 1, variant
        if grids[0]["kind"] == "quad":
            pts = [g["grid_points"] for g in grids]
            assert pts == sorted(pts) and len(set(pts)) == 3, variant
        elif grids[0]["kind"] == "adi_2d":
            assert [g["n_x"] for g in grids] == sorted(g["n_x"] for g in grids), variant
```

- [ ] **Step 6: Run the gate in quick mode**

Run: `.venv/bin/python example/mo_volmodels/11_pde_convergence_gate.py --quick --out-dir output/gate_smoke`
Expected: completes, writes `pde_convergence_gate.json` and `gate_decision.json` with six variant entries. Quick mode's verdict is explicitly non-production-valid — this step checks plumbing only.

- [ ] **Step 7: Commit**

```bash
git add example/mo_volmodels/11_pde_convergence_gate.py test/mo_volmodels/test_gate_scope.py
git commit -m "feat(mo): gate all six study variants via a production/reference pair table"
```

---

## Task 5: Evaluate the bias detector within maturity buckets

**Files:**
- Modify: `example/mo_volmodels/11_pde_convergence_gate.py:552` (`detect_systematic_bias`), `:590` (`decide_route`)
- Test: `test/mo_volmodels/test_gate_scope.py`

**Interfaces:**
- Produces: `detect_systematic_bias_bucketed(cells, tol_abs_pct=TOL_ABS_PCT) -> tuple[bool, dict]`. Input cells need `signed_diff_pct` and `case`. Returns biased-if-**any**-bucket-is-biased, with per-bucket detail under key `buckets`.

**Why:** §5.2. The PDE error changes sign with remaining maturity (positive at T=0.25, negative at T≥1). Pooled across maturities that scores a 0.75 sign fraction — under the 0.9 threshold — which is how the original G2 recorded `biased: false` at 0.533 while a systematic per-maturity bias was present. Within any single maturity the sign is unanimous.

The existing `case` field (`"full"` ≈ 3Y, `"decayed"` ≈ 1Y remaining) is the bucket key. Do not invent maturity bins.

- [ ] **Step 1: Write the failing test**

```python
def test_pooled_bias_hides_a_sign_flip_that_buckets_expose():
    """The exact failure mode from the original G2 run."""
    gate = _load_gate()
    cells = (
        [{"case": "full", "signed_diff_pct": +0.20} for _ in range(6)]
        + [{"case": "decayed", "signed_diff_pct": -0.20} for _ in range(6)]
    )
    pooled, _ = gate.detect_systematic_bias([c["signed_diff_pct"] for c in cells])
    bucketed, info = gate.detect_systematic_bias_bucketed(cells)

    assert pooled is False        # 0.5 sign fraction: reads as unbiased
    assert bucketed is True       # each bucket is unanimous
    assert set(info["buckets"]) == {"full", "decayed"}


def test_unbiased_cells_stay_unbiased_under_bucketing():
    gate = _load_gate()
    cells = [
        {"case": "full", "signed_diff_pct": v}
        for v in (+0.20, -0.18, +0.02, -0.21, +0.19, -0.05)
    ]
    biased, _ = gate.detect_systematic_bias_bucketed(cells)
    assert biased is False


def test_a_bucket_below_the_minimum_cell_count_cannot_flag_bias():
    """detect_systematic_bias needs >= 4 cells; a thin bucket must not vote."""
    gate = _load_gate()
    cells = (
        [{"case": "full", "signed_diff_pct": +0.02} for _ in range(6)]
        + [{"case": "decayed", "signed_diff_pct": +0.30} for _ in range(2)]
    )
    biased, info = gate.detect_systematic_bias_bucketed(cells)
    assert biased is False
    assert info["buckets"]["decayed"]["skipped"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_scope.py -n0 -q -k bias`
Expected: FAIL — `AttributeError: ... 'detect_systematic_bias_bucketed'`.

- [ ] **Step 3: Implement**

```python
def detect_systematic_bias_bucketed(
    cells: Sequence[Dict[str, Any]],
    tol_abs_pct: float = TOL_ABS_PCT,
) -> Tuple[bool, Dict[str, Any]]:
    """Bias detection WITHIN maturity buckets, never pooled across them.

    The 2D PDE error changes sign with remaining maturity (spec §5.2), so a
    pooled sign fraction averages the two regimes and reads as unbiased while
    each bucket is unanimous.  A variant is biased if ANY bucket with enough
    cells is biased.  Buckets below detect_systematic_bias's 4-cell minimum
    are recorded and skipped -- they cannot flag, and they cannot mask.
    """
    buckets: Dict[str, Any] = {}
    any_biased = False
    for case in CASES:
        rows = [c for c in cells if c.get("case") == case]
        diffs = [c.get("signed_diff_pct") for c in rows]
        usable = [d for d in diffs if d is not None and math.isfinite(float(d))]
        if len(usable) < 4:
            buckets[case] = {"n_cells": len(usable), "skipped": True}
            continue
        biased, info = detect_systematic_bias(usable, tol_abs_pct)
        buckets[case] = {**info, "skipped": False, "biased": bool(biased)}
        any_biased = any_biased or bool(biased)
    return any_biased, {"buckets": buckets, "pooled_not_used": True}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_scope.py -n0 -q -k bias`
Expected: PASS, 3 tests.

- [ ] **Step 5: Wire it into `decide_route`**

In `decide_route`, replace

```python
biased, bias_info = detect_systematic_bias(
    [c.get("signed_diff_pct") for c in medium_cells], tol_abs_pct
)
```

with `biased, bias_info = detect_systematic_bias_bucketed(medium_cells, tol_abs_pct)`, and update the rationale string to name the offending bucket(s).

- [ ] **Step 6: Run the gate's existing unit tests**

Run: `.venv/bin/python -m pytest test/mo_volmodels -n0 -q`
Expected: PASS. If a pre-existing test asserts the pooled behaviour, update it — the pooled path is now a helper, not the gate criterion.

- [ ] **Step 7: Commit**

```bash
git add example/mo_volmodels/11_pde_convergence_gate.py test/mo_volmodels/test_gate_scope.py
git commit -m "fix(mo): evaluate G2 bias within maturity buckets, not pooled"
```

---

## Task 6: Promote delta from secondary evidence to a gate criterion

**Files:**
- Modify: `example/mo_volmodels/11_pde_convergence_gate.py:950-995` (delta rows), `:590` (`decide_route`)
- Test: `test/mo_volmodels/test_gate_scope.py`

**Interfaces:**
- Produces:
  - `delta_quantum_per_unit(s0: float, notional: float = NOTIONAL, multiplier: float = FUTURES_MULTIPLIER) -> float` — per-unit delta moved by **one** IM futures contract.
  - `delta_cell_passed(abs_diff: float, s0: float) -> bool` — `abs_diff <= 0.5 * quantum`.
  - `detect_delta_bias(rows) -> tuple[bool, dict]` — mean **signed** delta difference, in contracts, against a 0.1-contract bound.

**Why:** §5.3. This is a hedging study; P&L is driven by model-consistent delta, not PV. The original G2 gated PV only. The tolerance is derived from what the hedge can physically express rather than picked as a percentage.

**The unit conversion is the whole trick — get it wrong and the threshold is meaningless.** The gate prices a 1-unit product, so its delta is per index unit. The backtest holds `contract_multiplier = NOTIONAL / s0` index units, and one IM contract covers `FUTURES_MULTIPLIER = 200` index points. So one contract moves

```
per-unit delta quantum = 200 / contract_multiplier = 200 * s0 / NOTIONAL
```

Worked for 2023-05-04 (`s0 = 6733.97`, `contract_multiplier = 7425.5`): `200 / 7425.5 = 0.02694` per-unit delta, i.e. 13,468 CNY per 1% spot move; half a contract is `0.01347`.

`s0` differs per inception — **4,532.52** (2024-09-02) to **6,733.97** (2023-05-04) — so the quantum spans `0.01813` to `0.02694`, a factor of 1.49. The threshold is therefore computed **per date** and never hardcoded.

- [ ] **Step 1: Write the failing test**

```python
def test_delta_quantum_matches_the_worked_example():
    """Spec §5.3, 2023-05-04: s0=6733.97, multiplier=7425.5 -> 0.02694."""
    gate = _load_gate()
    q = gate.delta_quantum_per_unit(6733.97, notional=50_000_000.0)
    assert q == pytest.approx(0.026936, abs=1e-6)


def test_delta_quantum_scales_with_inception_spot():
    """A fixed threshold would be 1.49x wrong across the 27 inceptions."""
    gate = _load_gate()
    lo = gate.delta_quantum_per_unit(4532.52, notional=50_000_000.0)
    hi = gate.delta_quantum_per_unit(6733.97, notional=50_000_000.0)
    assert hi / lo == pytest.approx(6733.97 / 4532.52, rel=1e-9)


def test_disagreement_under_half_a_contract_passes():
    gate = _load_gate()
    q = gate.delta_quantum_per_unit(6733.97)
    assert gate.delta_cell_passed(0.49 * q, 6733.97) is True
    assert gate.delta_cell_passed(0.51 * q, 6733.97) is False


def test_small_one_sided_delta_bias_is_caught_even_though_each_cell_passes():
    """Rounding absorbs a single cell; 700 rebalances accumulate the mean."""
    gate = _load_gate()
    s0 = 6733.97
    q = gate.delta_quantum_per_unit(s0)
    rows = [{"s0": s0, "signed_diff": 0.2 * q} for _ in range(8)]
    assert all(gate.delta_cell_passed(abs(r["signed_diff"]), s0) for r in rows)
    biased, info = gate.detect_delta_bias(rows)
    assert biased is True
    assert info["mean_signed_contracts"] == pytest.approx(0.2, rel=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_scope.py -n0 -q -k delta`
Expected: FAIL — `AttributeError: ... 'delta_quantum_per_unit'`.

- [ ] **Step 3: Implement**

```python
# Hedge-derived delta tolerances (spec §5.3).  The gate prices a 1-unit
# product; the backtest holds NOTIONAL/s0 index units.  One IM contract
# covers FUTURES_MULTIPLIER index points, so it moves
# FUTURES_MULTIPLIER * s0 / NOTIONAL of per-unit delta.  s0 varies 1.49x
# across the 27 inceptions, so this is computed per date, never fixed.
FUTURES_MULTIPLIER = 200.0
STUDY_NOTIONAL = 50_000_000.0
DELTA_CELL_CONTRACTS = 0.5    # rounding provably absorbs less than this
DELTA_BIAS_CONTRACTS = 0.1    # accumulates over ~700 rebalances


def delta_quantum_per_unit(
    s0: float,
    notional: float = STUDY_NOTIONAL,
    multiplier: float = FUTURES_MULTIPLIER,
) -> float:
    """Per-unit delta moved by exactly one IM futures contract."""
    return float(multiplier) * float(s0) / float(notional)


def delta_cell_passed(abs_diff: float, s0: float, **kw) -> bool:
    return abs(float(abs_diff)) <= DELTA_CELL_CONTRACTS * delta_quantum_per_unit(s0, **kw)


def detect_delta_bias(rows: Sequence[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    """Mean SIGNED delta difference in contracts, against the bias bound.

    Signed, not absolute: contract rounding kills symmetric noise but lets a
    one-sided bias accumulate over the rebalance schedule.
    """
    contracts = [
        float(r["signed_diff"]) / delta_quantum_per_unit(float(r["s0"]))
        for r in rows
        if r.get("signed_diff") is not None and r.get("s0")
    ]
    if not contracts:
        return False, {"n_rows": 0, "mean_signed_contracts": None}
    mean_signed = float(np.mean(contracts))
    return abs(mean_signed) > DELTA_BIAS_CONTRACTS, {
        "n_rows": len(contracts),
        "mean_signed_contracts": mean_signed,
        "max_abs_contracts": max(abs(c) for c in contracts),
        "bound_contracts": DELTA_BIAS_CONTRACTS,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_scope.py -n0 -q -k delta`
Expected: PASS, 4 tests.

- [ ] **Step 5: Record `signed_diff`, `s0` and the verdict on every delta row**

The existing row (`:955`) stores only `abs_diff`. `detect_delta_bias` needs the **sign**, and the tolerance needs `s0`.

First rename the row's keys `delta_pde` → `delta_production` and `delta_mc` → `delta_reference`: after Task 4 the production engine is not always a PDE, and the reference not always MC. Update every reader (`grep -n "delta_pde\|delta_mc" example/mo_volmodels/`), including the report/printing code. Then extend the row:

```python
delta_row["s0"] = float(s0_inception)
if delta_row["delta_production"] is not None and delta_row["delta_reference"] is not None:
    signed = delta_row["delta_production"] - delta_row["delta_reference"]
    delta_row["signed_diff"] = signed
    delta_row["abs_diff"] = abs(signed)
    delta_row["diff_contracts"] = signed / delta_quantum_per_unit(float(s0_inception))
    delta_row["passed"] = delta_cell_passed(abs(signed), float(s0_inception))
else:
    delta_row["signed_diff"] = None
    delta_row["diff_contracts"] = None
    delta_row["passed"] = False        # fail closed
```

The `_bumped_pde_delta` / `_bumped_mc_delta` helpers (`:895`, and its MC twin) must likewise be driven off `GATE_PAIRS[variant]` rather than assuming PDE-vs-MC. A deterministic-reference variant uses the same central bump on both engines; there is no CRN to arrange.

- [ ] **Step 6: Add delta admission to `decide_route`**

`decide_route` currently takes only PV cells. Add a `delta_rows` parameter and two more conjuncts to the route decision:

```python
delta_pass = all(r.get("passed") is True for r in delta_rows) if delta_rows else False
delta_biased, delta_info = detect_delta_bias(delta_rows)
route = "pde" if (medium_pass and not biased and fine_pass and drift_ok
                  and delta_pass and not delta_biased) else "mc"
```

Add a `reasons.append(...)` for each new failure mode naming the offending dates, and surface `delta_pass` / `delta_biased` / `delta_info` in the returned dict and in `gate_decision.json`. Empty `delta_rows` must not admit — mirror the existing "refusing to admit PDE on empty evidence" behaviour.

- [ ] **Step 7: Run the full mo test suite**

Run: `.venv/bin/python -m pytest test/mo_volmodels -n0 -q`
Expected: PASS. Update any test that constructs `decide_route` positionally.

- [ ] **Step 8: Commit**

```bash
git add example/mo_volmodels/11_pde_convergence_gate.py test/mo_volmodels/test_gate_scope.py
git commit -m "feat(mo): gate G2 on delta agreement in IM contracts, not PV alone"
```

---

## Task 7: Condition the G2 verdict on the Feller regime

**Files:**
- Modify: `example/mo_volmodels/11_pde_convergence_gate.py:1132` (`_calibration_record`), `:590` (`decide_route`)
- Test: `test/mo_volmodels/test_gate_scope.py`

**Interfaces:**
- Produces: `feller_bucket(ratio: float | None) -> str` returning `"violated"` (< 0.5), `"boundary"` (0.5 … 10), `"degenerate"` (> 10) or `"unknown"`; and per-bucket route detail in `decide_route`'s output under `feller_buckets`.

**Why, and why these thresholds:** §7A.4(3) and §7A.11. A uniform verdict would average a 0.03% regime with a 2.5% one. The cut points come from the measurements, not from taste:

- **0.5** — §7A.11's measured failure boundary. Unconstrained fits fail the gate at ratio ≤ 0.50 and pass above it; 16.4% of the cohort sits below 0.50.
- **10** — the σ-collapse marker from §7A.10(3). 6.6% of enforced fits satisfy Feller by driving σ to its lower bound (ratios up to 1.7e5), giving a deterministic-variance model that fails the gate under *both* calibration policies.

With `enforce_feller=True` shipped, **80% of dates land in `boundary` at ratio ≈ 1.0**. `violated` should be empty in the re-run; if it is not, the enforcement did not take and that is a finding.

`feller_ratio` is already in the calibration record (landed with §7A.4) and `_calibration_record` passes the record through verbatim, so no plumbing is needed to obtain it — only to key on it.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("ratio,expected", [
    (0.197, "violated"), (0.493, "violated"), (0.5, "boundary"),
    (1.0001, "boundary"), (7.1945, "boundary"), (10.0, "boundary"),
    (172224.1, "degenerate"), (None, "unknown"),
])
def test_feller_buckets_use_the_measured_cut_points(ratio, expected):
    assert _load_gate().feller_bucket(ratio) == expected


def test_route_records_a_verdict_per_feller_bucket():
    """A pooled verdict averages a 0.03% regime with a 2.5% one (§7A.3)."""
    gate = _load_gate()
    cells = (
        [{"date": "2024-01-12", "case": "full", "signed_diff_pct": +0.58,
          "passed": False, "feller_ratio": 0.197, "pde_price": 1.0, "notional": 1.0}]
        + [{"date": f"d{i}", "case": "full", "signed_diff_pct": +0.10,
            "passed": True, "feller_ratio": 1.0001, "pde_price": 1.0, "notional": 1.0}
           for i in range(6)]
    )
    out = gate.decide_route(cells, cells, delta_rows=[])
    assert out["feller_buckets"]["violated"]["n_cells"] == 1
    assert out["feller_buckets"]["violated"]["n_passed"] == 0
    assert out["feller_buckets"]["boundary"]["n_passed"] == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_scope.py -n0 -q -k feller`
Expected: FAIL — `AttributeError: ... 'feller_bucket'`.

- [ ] **Step 3: Implement**

```python
# Feller-regime buckets.  Cut points are measured, not chosen:
#   0.5  -- §7A.11's failure boundary: unconstrained fits fail the gate at
#           ratio <= 0.50 and pass above it (16.4% of the cohort below).
#   10   -- §7A.10(3)'s sigma-collapse marker: 6.6% of enforced fits satisfy
#           Feller by driving sigma to its bound, giving a
#           deterministic-variance model that fails under BOTH policies.
FELLER_VIOLATED_BELOW = 0.5
FELLER_DEGENERATE_ABOVE = 10.0


def feller_bucket(ratio: Optional[float]) -> str:
    if ratio is None or not math.isfinite(float(ratio)):
        return "unknown"
    ratio = float(ratio)
    if ratio < FELLER_VIOLATED_BELOW:
        return "violated"
    if ratio > FELLER_DEGENERATE_ABOVE:
        return "degenerate"
    return "boundary"
```

In `process_date`, copy `feller_ratio` from the Heston calibration record onto every `heston` / `heston_slv` cell and delta row. Non-Heston variants carry `None` → `"unknown"`.

In `decide_route`, group the medium cells by `feller_bucket(c.get("feller_ratio"))` and emit per-bucket `n_cells` / `n_passed` / `max_abs_diff_pct` under `feller_buckets`. **The route stays all-cells-must-pass** — bucketing reports *where* a variant fails, it does not license failure in an unpopular regime.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_gate_scope.py -n0 -q -k feller`
Expected: PASS, 9 tests.

- [ ] **Step 5: Run the full mo test suite**

Run: `.venv/bin/python -m pytest test/mo_volmodels -n0 -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add example/mo_volmodels/11_pde_convergence_gate.py test/mo_volmodels/test_gate_scope.py
git commit -m "feat(mo): report the G2 verdict per Feller regime, on measured cut points"
```

---

## Task 8: Run G2 and emit the decision

**Files:**
- Modify: `docs/superpowers/specs/2026-07-30-snowball-volmodel-backtest-040-rebaseline-design.md` (§5 results)

**Interfaces:**
- Consumes: everything above.
- Produces: `output/pde_convergence_gate/gate_decision.json` with six variant entries and a fresh `evidence_sha256`; stage 12 reads it via `load_gate_routing`.

- [ ] **Step 1: Confirm the whole suite is green before spending compute**

Run: `.venv/bin/python -m pytest -q`
Expected: the four known pre-existing failures only — `test_adi_core_tau_exactness.py` (×3) and `test_snowball_pde_knocked_in_grid.py` (×1). Both files are another session's untracked WIP and reproduce against a pristine `git archive HEAD` tree. **Any other failure blocks this task.**

- [ ] **Step 2: Update `GateRouting` for six variants**

`GateRouting.solver_for` (`12_snowball_volmodel_backtest.py:238`) short-circuits `bsm` and `localvol` to `"pde"` as "outside the 2D-ADI gate's scope". After Task 4 they are inside it. Remove the short-circuit and let every variant read its route from the decision, keeping the fail-closed raise for a missing or unusable route.

Add a test in `test/mo_volmodels/test_gate_scope.py`:

```python
def test_routing_no_longer_short_circuits_one_d_variants():
    s12 = _load_stage12()
    routing = s12.GateRouting("p", None, {"localvol": "mc"}, {})
    assert routing.solver_for("localvol") == "mc"
```

Run it (expect FAIL: returns `"pde"`), then remove the short-circuit, then re-run (expect PASS). Commit.

- [ ] **Step 3: Run the gate for real**

Background; the QE-M reference at 4 substeps quadruples the reference MC time grid, and §7A.6 measured MC at 40–50 s/case against PDE at ~9 s, so **budget several hours**:

```bash
.venv/bin/python example/mo_volmodels/11_pde_convergence_gate.py \
  --workers 4 --out-dir output/pde_convergence_gate 2>&1 | tee output/g2_run.log
```

- [ ] **Step 4: Read the decision**

```bash
.venv/bin/python -c "
import json
d = json.load(open('output/pde_convergence_gate/gate_decision.json'))
print('evidence', d['evidence_sha256'][:16])
print('mc_reference', d['mc_reference']['scheme'])
for v, row in sorted(d['variants'].items()):
    print(f\"{v:16} route={row['route']:4} {row['rationale'][:90]}\")
"
```

Check three things specifically:
- `mc_reference.scheme == "QUADEXP_M"` — confirms the §7A.4(4) upgrade is in the evidence.
- `feller_buckets.violated` is empty for `heston` / `heston_slv`. If it is not, `enforce_feller` did not take; **stop and investigate** rather than accepting the verdict.
- Whether `biased` is `false` by a *narrow* median. §7A.11 measured sign fraction 1.0 with the enforced median at 0.113% against a 0.125% threshold — a 0.012-point margin. Record the actual margin; a narrow pass is "biased but small", not "clean".

- [ ] **Step 5: Record the outcome in the spec**

Add a §5.5 with: the per-variant routes, the per-Feller-bucket table, the delta admission results in contracts, the bias margin from Step 4, and `evidence_sha256`. State plainly which variants were admitted to PDE and which fell back to MC — including whether §7A.8's recorded *prediction* (both 2D variants admitted at 200×60×`ceil(400·T)`, on delta stability more than speed) held or was falsified.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/specs/2026-07-30-snowball-volmodel-backtest-040-rebaseline-design.md
git commit -m "docs(mo): record the re-scoped G2 outcome and per-regime evidence"
```

---

## Out of scope for this plan

Deliberately deferred; each needs its own plan.

- **G5 pre-flight grid sweep** (§9) — build grids only, no solve, for every operating point before the fleet. `fdf3a70` made under-resolution a fail-closed `ValidationError`, and `test_adi_core_tau_exactness.py`'s failures show `n_x=60` at T≥2 already trips it.
- **Task 6.1 timing run** (§7.3) — the fleet total is set by measurement, never extrapolated from single solves.
- **G3 accounting sanity** (§11) on one inception.
- **Stage 13 σ-collapse handling** — §7A.10(3)'s 50 dates must be flagged or excluded, never averaged into a `heston` result.
- **Deriving `_clone_engine` from the constructor signature** (§7A.10) — the hand-transcribed kwargs list is a standing silent-mispricing hazard.
