# MC Reference-Stack Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make schema-12 MC reference regeneration + the Stage-17 aggregate certification decisive overnight (≤12 h, half-width ≤ 0.02 contracts) on the 48 GiB / 14-core host, via per-cell variance-reduction treatments and estimate-blind adaptive allocation/stopping.

**Architecture:** Two new library modules (`quantark/montecarlo/control_weights.py`, `quantark/validation/adaptive_allocation.py`) built TDD; a standalone demo package (`docs/mc-reference-convergence/`) that measures treatment candidates per cell against the recovered schema-11 anchors; a **user decision checkpoint** on the measured matrix; then stage-16/17 harness wiring behind gates. Spec: `docs/superpowers/specs/2026-08-10-mc-reference-convergence-design.md`.

**Tech Stack:** Python 3.11, NumPy/SciPy, pytest (`-n0` for anything spawning process pools), existing stage-16/17 harnesses (`example/mo_volmodels/16_adi_greek_certification.py`, `17_adi_slv_aggregate_certification.py`), `quantark/validation/greek_certification.py`, `quantark/montecarlo/qmc_rqmc_driver.py`.

## Global Constraints

- Work in `.worktrees/adi-greek-certification` on branch `codex/adi-greek-certification`. Every Bash call must `cd` there first (working directory resets between calls).
- Run everything with the worktree shadowed: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python …` (the editable install otherwise binds `quantark` to the main checkout).
- `docs/` is gitignored: new docs files need `git add -f`.
- Precision target: aggregate statistical half-width ≤ **0.02 contracts** (t₀.₉₇₅,df × SE, schema-11 convention). Budget cap: **12 h**. Pilot size: **32 batches** per cell×variant.
- Stopping logic must be **estimate-blind** (spec S-G1): no estimate value may be reachable from any stopping/allocation code path.
- Estimator changes are treatments selected per cell by the user decision matrix (Task 7 checkpoint); never enable a treatment the matrix rejected. `heston_slv/low_feller` keeps its measured direct estimator (control tried and rejected 2026-08-05/06 — see stage-16 comment at `PRODUCTION_SLV_PRIMARY_BATCHES_BY_CASE`).
- Long-running measurement steps run in background (`run_in_background`), with durations noted. No measurement value may be written into any doc before the run completes (report actual numbers only).
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## File Structure

```
quantark/montecarlo/control_weights.py          NEW  cross-fitted control-weight estimator (V2)
quantark/validation/adaptive_allocation.py      NEW  Neyman allocation + precision stopping (S1-S3)
test/test_control_weights.py                    NEW
test/test_adaptive_allocation.py                NEW
docs/mc-reference-convergence/README.md         NEW  decision matrix + measured results
docs/mc-reference-convergence/demo_common.py    NEW  stage-16 loader + measurement core
docs/mc-reference-convergence/demo_cell.py      NEW  one CLI demo, parameterized by cell
docs/mc-reference-convergence/logs/             NEW  raw demo outputs (committed)
example/mo_volmodels/16_adi_greek_certification.py   MODIFY  per-cell profiles + schema-12 treatment fields (Task 8)
example/mo_volmodels/17_adi_slv_aggregate_certification.py  MODIFY  pilot→allocation→monitor wiring (Task 9)
test/mo_volmodels/test_adi_slv_aggregate_certification.py   MODIFY  allocation/monitor/fallback gates
```

---

### Task 1: Cross-fitted control weights module (V2)

**Files:**
- Create: `quantark/montecarlo/control_weights.py`
- Test: `test/test_control_weights.py`

**Interfaces:**
- Consumes: nothing project-specific (NumPy only).
- Produces: `CrossFittedControl` dataclass and
  `cross_fitted_control(primary: np.ndarray, control: np.ndarray, control_expectation: float, folds: int = 2) -> CrossFittedControl`
  with fields `adjusted: np.ndarray` (per-batch adjusted values, same length as `primary`),
  `weights: np.ndarray` (per-fold β), `variance_ratio: float` (Var(adjusted)/Var(primary)).
  Task 5's demo rows and Task 9's wiring call exactly this.

- [ ] **Step 1: Write the failing tests**

```python
# test/test_control_weights.py
import numpy as np
import pytest

from quantark.montecarlo.control_weights import CrossFittedControl, cross_fitted_control


def _correlated_batches(rho: float, n: int = 400, seed: int = 7):
    rng = np.random.default_rng(seed)
    control = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, np.sqrt(1.0 - rho * rho), n)
    primary = 5.0 + rho * control + noise
    return primary, control


def test_unbiased_mean_preserved():
    primary, control = _correlated_batches(rho=0.9)
    result = cross_fitted_control(primary, control, control_expectation=0.0)
    # Adjusted mean must agree with the primary mean within its own SE.
    se = primary.std(ddof=1) / np.sqrt(primary.size)
    assert abs(result.adjusted.mean() - primary.mean()) < 3.0 * se


def test_variance_reduced_when_correlated():
    primary, control = _correlated_batches(rho=0.9)
    result = cross_fitted_control(primary, control, control_expectation=0.0)
    assert result.variance_ratio < 0.35  # 1 - rho^2 = 0.19 plus cross-fit slack


def test_no_gain_when_uncorrelated_but_still_unbiased():
    primary, control = _correlated_batches(rho=0.0)
    result = cross_fitted_control(primary, control, control_expectation=0.0)
    assert 0.8 < result.variance_ratio < 1.3
    se = primary.std(ddof=1) / np.sqrt(primary.size)
    assert abs(result.adjusted.mean() - primary.mean()) < 3.0 * se


def test_weights_are_out_of_fold():
    # A pathological fold-A-only outlier must not contaminate fold A's beta.
    primary, control = _correlated_batches(rho=0.9, n=100)
    primary = primary.copy()
    primary[0] += 50.0  # outlier lives in fold A
    result = cross_fitted_control(primary, control, control_expectation=0.0, folds=2)
    # Fold A's beta is fitted on fold B, which is outlier-free:
    clean_primary, clean_control = _correlated_batches(rho=0.9, n=100)
    clean = cross_fitted_control(clean_primary, clean_control, control_expectation=0.0, folds=2)
    assert result.weights[0] == pytest.approx(clean.weights[0])


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cross_fitted_control(np.ones(10), np.ones(9), control_expectation=0.0)


def test_rejects_too_few_batches_per_fold():
    with pytest.raises(ValueError):
        cross_fitted_control(np.ones(3), np.ones(3), control_expectation=0.0, folds=2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_control_weights.py -v`
Expected: FAIL with `ModuleNotFoundError: quantark.montecarlo.control_weights`

- [ ] **Step 3: Implement the module**

```python
# quantark/montecarlo/control_weights.py
"""Cross-fitted control-variate weights for batched reference estimators.

The frozen certification design uses a single global control weight. This
module estimates per-cell weights without introducing bias: batches are split
into folds, each fold's weight is fitted on the *other* folds only, and the
control's expectation is supplied externally (from the independent
high-precision control run), so E[adjusted] == E[primary] exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class CrossFittedControl:
    """Per-batch adjusted values plus the out-of-fold weights that made them."""

    adjusted: np.ndarray
    weights: np.ndarray
    variance_ratio: float

    def as_dict(self) -> dict:
        return {
            "weights": [float(w) for w in self.weights],
            "variance_ratio": float(self.variance_ratio),
            "n_batches": int(self.adjusted.size),
        }


def cross_fitted_control(
    primary: np.ndarray,
    control: np.ndarray,
    control_expectation: float,
    folds: int = 2,
) -> CrossFittedControl:
    """Adjust ``primary`` batch means by an out-of-fold-fitted control weight.

    ``adjusted[i] = primary[i] - beta_k * (control[i] - control_expectation)``
    where ``beta_k`` is the OLS weight fitted on every fold except the one
    containing batch ``i``. Unbiasedness needs only that
    ``E[control] == control_expectation`` and that folds are independent.
    """
    primary = np.asarray(primary, dtype=float)
    control = np.asarray(control, dtype=float)
    if primary.shape != control.shape or primary.ndim != 1:
        raise ValidationError("primary and control must be 1-D arrays of equal length")
    if folds < 2:
        raise ValidationError("cross-fitting requires at least 2 folds")
    if primary.size < 2 * folds:
        raise ValidationError("need at least 2 batches per fold")

    fold_ids = np.arange(primary.size) % folds
    weights = np.empty(folds, dtype=float)
    adjusted = np.empty_like(primary)
    centered = control - float(control_expectation)
    for k in range(folds):
        out = fold_ids != k
        var = np.var(centered[out], ddof=1)
        if var <= 0.0:
            weights[k] = 0.0
        else:
            cov = np.cov(primary[out], centered[out], ddof=1)[0, 1]
            weights[k] = cov / var
        mask = fold_ids == k
        adjusted[mask] = primary[mask] - weights[k] * centered[mask]

    var_primary = np.var(primary, ddof=1)
    var_adjusted = np.var(adjusted, ddof=1)
    ratio = float(var_adjusted / var_primary) if var_primary > 0.0 else 1.0
    return CrossFittedControl(adjusted=adjusted, weights=weights, variance_ratio=ratio)
```

Also add the export in `quantark/montecarlo/__init__.py` next to the existing
`conditional_snowball` exports:

```python
from quantark.montecarlo.control_weights import CrossFittedControl, cross_fitted_control
```

(and extend its `__all__` list with `"CrossFittedControl", "cross_fitted_control"`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_control_weights.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification
git add quantark/montecarlo/control_weights.py quantark/montecarlo/__init__.py test/test_control_weights.py
git commit -m "feat(mc): cross-fitted control-variate weights (V2)"
```

---

### Task 2: Adaptive allocation + precision stopping module (S1–S3 core)

**Files:**
- Create: `quantark/validation/adaptive_allocation.py`
- Test: `test/test_adaptive_allocation.py`

**Interfaces:**
- Consumes: nothing project-specific (NumPy, SciPy `student_t`).
- Produces (Task 6 demos and Task 9 wiring call exactly these):
  - `CellPrecision(name: str, n_batches: int, batch_sd: float, seconds_per_batch: float)` — deliberately has **no estimate field**.
  - `neyman_allocation(cells: Sequence[CellPrecision], budget_seconds: float, min_batches: int = 16) -> dict[str, int]`
  - `projected_aggregate_halfwidth(cells: Sequence[CellPrecision], confidence: float = 0.975) -> float`
  - `precision_stop(cells, target_halfwidth: float, elapsed_seconds: float, budget_seconds: float, confidence: float = 0.975) -> StopDecision` where `StopDecision` has `stop: bool`, `trigger: str | None` (`"target-reached" | "budget-cap" | None`), `projected_halfwidth: float`.

- [ ] **Step 1: Write the failing tests**

```python
# test/test_adaptive_allocation.py
import dataclasses

import numpy as np
import pytest
from scipy.stats import t as student_t

from quantark.validation.adaptive_allocation import (
    CellPrecision,
    StopDecision,
    neyman_allocation,
    precision_stop,
    projected_aggregate_halfwidth,
)


def _cells(sds=(1.0, 1.0, 1.0), costs=(1.0, 1.0, 1.0), n=32):
    return [
        CellPrecision(name=f"c{i}", n_batches=n, batch_sd=s, seconds_per_batch=c)
        for i, (s, c) in enumerate(zip(sds, costs))
    ]


def test_blindness_no_estimate_field():
    # S-G1: the type itself must not accept an estimate.
    with pytest.raises(TypeError):
        CellPrecision(name="x", n_batches=1, batch_sd=1.0, seconds_per_batch=1.0, estimate=0.5)
    assert "estimate" not in {f.name for f in dataclasses.fields(CellPrecision)}


def test_neyman_matches_analytic_optimum():
    # S-G2: n_j proportional to sd_j / sqrt(cost_j); equal-cost case with sds 1:2:4
    cells = _cells(sds=(1.0, 2.0, 4.0), costs=(1.0, 1.0, 1.0))
    alloc = neyman_allocation(cells, budget_seconds=7000.0, min_batches=16)
    assert alloc["c1"] / alloc["c0"] == pytest.approx(2.0, rel=0.05)
    assert alloc["c2"] / alloc["c0"] == pytest.approx(4.0, rel=0.05)
    spent = sum(alloc[c.name] * c.seconds_per_batch for c in cells)
    assert spent <= 7000.0


def test_neyman_cost_weighting():
    # doubling a cell's cost scales its share by 1/sqrt(2)
    cheap = neyman_allocation(_cells(sds=(1.0, 1.0), costs=(1.0, 1.0)), budget_seconds=2000.0)
    costly = neyman_allocation(_cells(sds=(1.0, 1.0), costs=(1.0, 4.0)), budget_seconds=2000.0)
    assert costly["c1"] / costly["c0"] == pytest.approx(0.5, rel=0.1)
    assert cheap["c1"] / cheap["c0"] == pytest.approx(1.0, rel=0.05)


def test_min_batches_floor():
    cells = _cells(sds=(0.001, 5.0), costs=(1.0, 1.0))
    alloc = neyman_allocation(cells, budget_seconds=5000.0, min_batches=16)
    assert alloc["c0"] >= 16


def test_projected_halfwidth_matches_hand_computation():
    cells = _cells(sds=(0.7, 1.1), costs=(1.0, 1.0), n=64)
    # aggregate = mean of cell means; SE^2 = (1/k^2) sum sd_j^2/n_j; df = sum(n_j-1)
    se = np.sqrt((0.7**2 / 64 + 1.1**2 / 64) / 4.0)
    df = 63 + 63
    expected = float(student_t.ppf(0.975, df)) * se
    assert projected_aggregate_halfwidth(cells) == pytest.approx(expected, rel=1e-12)


def test_precision_stop_target_reached():
    cells = _cells(sds=(0.01, 0.01), costs=(1.0, 1.0), n=64)
    decision = precision_stop(cells, target_halfwidth=0.02, elapsed_seconds=10.0, budget_seconds=100.0)
    assert decision == StopDecision(stop=True, trigger="target-reached",
                                    projected_halfwidth=decision.projected_halfwidth)


def test_precision_stop_budget_cap():
    cells = _cells(sds=(5.0, 5.0), costs=(1.0, 1.0), n=8)
    decision = precision_stop(cells, target_halfwidth=0.02, elapsed_seconds=101.0, budget_seconds=100.0)
    assert decision.stop is True and decision.trigger == "budget-cap"


def test_precision_stop_keep_going():
    cells = _cells(sds=(5.0, 5.0), costs=(1.0, 1.0), n=8)
    decision = precision_stop(cells, target_halfwidth=0.02, elapsed_seconds=10.0, budget_seconds=100.0)
    assert decision.stop is False and decision.trigger is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_adaptive_allocation.py -v`
Expected: FAIL with `ModuleNotFoundError: quantark.validation.adaptive_allocation`

- [ ] **Step 3: Implement the module**

```python
# quantark/validation/adaptive_allocation.py
"""Estimate-blind batch allocation and precision-based stopping.

Used by the certification banking loop: a pilot measures per-cell batch SD
and cost, ``neyman_allocation`` freezes where further batches go, and
``precision_stop`` halts on achieved *precision* (never the estimate), so the
final fixed-confidence verdict needs no sequential-testing correction.

The blindness is structural: ``CellPrecision`` has no field an estimate could
travel through, so the stopping path cannot read one (spec gate S-G1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from scipy.stats import t as student_t

from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class CellPrecision:
    """Precision state of one certification cell. Deliberately estimate-free."""

    name: str
    n_batches: int
    batch_sd: float
    seconds_per_batch: float

    def __post_init__(self):
        if self.n_batches < 2:
            raise ValidationError(f"{self.name}: need >= 2 batches for an SD")
        if self.batch_sd < 0.0 or not math.isfinite(self.batch_sd):
            raise ValidationError(f"{self.name}: batch_sd must be finite and >= 0")
        if self.seconds_per_batch <= 0.0:
            raise ValidationError(f"{self.name}: seconds_per_batch must be > 0")


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    trigger: Optional[str]  # "target-reached" | "budget-cap" | None
    projected_halfwidth: float

    def as_dict(self) -> dict:
        return {
            "stop": self.stop,
            "trigger": self.trigger,
            "projected_halfwidth": float(self.projected_halfwidth),
        }


def projected_aggregate_halfwidth(
    cells: Sequence[CellPrecision], confidence: float = 0.975
) -> float:
    """t x SE of the equal-weight mean of cell means (schema-11 convention)."""
    if not cells:
        raise ValidationError("need at least one cell")
    k = len(cells)
    variance = sum((c.batch_sd**2) / c.n_batches for c in cells) / (k * k)
    df = sum(c.n_batches - 1 for c in cells)
    return float(student_t.ppf(confidence, df)) * math.sqrt(variance)


def neyman_allocation(
    cells: Sequence[CellPrecision],
    budget_seconds: float,
    min_batches: int = 16,
) -> dict:
    """Cost-weighted Neyman allocation: n_j proportional to sd_j / sqrt(cost_j).

    Returns total batch counts per cell (including the pilot batches already
    banked), floored at ``min_batches`` and fitted inside ``budget_seconds``.
    """
    if budget_seconds <= 0.0:
        raise ValidationError("budget_seconds must be > 0")
    shares = {c.name: c.batch_sd / math.sqrt(c.seconds_per_batch) for c in cells}
    total_share = sum(shares.values())
    allocation = {}
    if total_share <= 0.0:
        # All cells report zero SD: nothing reduces variance; keep the floor.
        return {c.name: max(min_batches, c.n_batches) for c in cells}
    for c in cells:
        seconds_j = budget_seconds * shares[c.name] / total_share
        allocation[c.name] = max(min_batches, int(seconds_j / c.seconds_per_batch))
    # Shrink proportionally if the floors pushed us over budget.
    spent = sum(allocation[c.name] * c.seconds_per_batch for c in cells)
    if spent > budget_seconds:
        scale = budget_seconds / spent
        for c in cells:
            allocation[c.name] = max(min_batches, int(allocation[c.name] * scale))
    return allocation


def precision_stop(
    cells: Sequence[CellPrecision],
    target_halfwidth: float,
    elapsed_seconds: float,
    budget_seconds: float,
    confidence: float = 0.975,
) -> StopDecision:
    """Stop on achieved precision or exhausted budget; never on the estimate."""
    if target_halfwidth <= 0.0:
        raise ValidationError("target_halfwidth must be > 0")
    halfwidth = projected_aggregate_halfwidth(cells, confidence=confidence)
    if halfwidth <= target_halfwidth:
        return StopDecision(stop=True, trigger="target-reached", projected_halfwidth=halfwidth)
    if elapsed_seconds >= budget_seconds:
        return StopDecision(stop=True, trigger="budget-cap", projected_halfwidth=halfwidth)
    return StopDecision(stop=False, trigger=None, projected_halfwidth=halfwidth)
```

Also export from `quantark/validation/__init__.py` next to the existing
`greek_certification` exports:

```python
from quantark.validation.adaptive_allocation import (
    CellPrecision,
    StopDecision,
    neyman_allocation,
    precision_stop,
    projected_aggregate_halfwidth,
)
```

(and extend its `__all__` accordingly).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/test_adaptive_allocation.py -v`
Expected: 8 PASS

- [ ] **Step 5: Run the full fast test suite to catch import regressions**

Run: `cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_adaptive_allocation.py test/test_control_weights.py test/test_greek_certification.py test/test_conditional_snowball.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification
git add quantark/validation/adaptive_allocation.py quantark/validation/__init__.py test/test_adaptive_allocation.py
git commit -m "feat(validation): estimate-blind Neyman allocation + precision stopping (WS-S core)"
```

---

### Task 3: Demo package scaffold + measurement core

**Files:**
- Create: `docs/mc-reference-convergence/demo_common.py`
- Create: `docs/mc-reference-convergence/README.md` (skeleton; numbers land in Tasks 4–6)

**Interfaces:**
- Consumes: stage-16 module via `importlib` (same pattern as `docs/adi2d-greek-perf/probes/probe_delta_attribution.py`); `paired_mc_reference(variant, case, product, env, leverage, *, paths_per_batch, batches, seed, substeps, bump, slv_spot_bridge_dimensions=…, rqmc_batch_workers=…) -> PairedRQMCGreeksResult` (fields used: `batch_delta` → per-batch delta array); `EconomicGreekScale` from `quantark.validation`.
- Produces: `measure_row(cell: str, variant: str, label: str, *, batches: int, seed: int, bridge_dimensions: int, workers: int) -> dict` returning
  `{"label", "cell", "variant", "batches", "bridge_dimensions", "delta_mean", "delta_se", "batch_sd_contracts", "seconds_per_batch", "peak_rss_gib"}`. Tasks 4–6 call exactly this.

- [ ] **Step 1: Write the measurement core**

```python
# docs/mc-reference-convergence/demo_common.py
"""Shared loader + measurement core for the MC reference-convergence demos.

Loads the stage-16 harness by path (semantic fidelity: cases, products,
environments, engines, and the paired-RQMC reference estimator are the
harness's own), runs one treatment row, and reports the three decision-matrix
numbers: batch SD (contracts), seconds/batch, peak RSS.
"""

from __future__ import annotations

import importlib.util
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "cert16", REPO / "example" / "mo_volmodels" / "16_adi_greek_certification.py"
)
assert _spec is not None and _spec.loader is not None
cert = importlib.util.module_from_spec(_spec)
sys.modules["cert16"] = cert
_spec.loader.exec_module(cert)

HEDGE_INCEPTION_SPOT = 4532.52  # recovered schema-11 anchor, scale-verified bitwise


def _scale(case) -> "cert.EconomicGreekScale":
    return cert.EconomicGreekScale(
        model_spot=case.spot,
        hedge_inception_spot=HEDGE_INCEPTION_SPOT,
        study_notional=cert.STUDY_NOTIONAL,
        hedge_multiplier=cert.HEDGE_MULTIPLIER,
    )


def measure_row(
    cell: str,
    variant: str,
    label: str,
    *,
    batches: int,
    seed: int,
    bridge_dimensions: int,
    workers: int = 2,
) -> dict:
    case = next(c for c in cert.certification_cases(quick=False) if c.name == cell)
    product = cert.make_snowball(case, dense_ki=True)
    env = cert.make_environment(case.spot, math.sqrt(max(case.params.v0, case.params.theta)))
    leverage = cert.make_leverage_surface(case.maturity) if variant == "heston_slv" else None
    substeps = cert.PRODUCTION_QE_SUBSTEPS_BY_VARIANT_CASE[variant][cell]["target"]
    paths = (
        cert.PRODUCTION_SLV_PATHS_PER_BATCH
        if variant == "heston_slv"
        else cert.PRODUCTION_HESTON_PATHS_PER_BATCH
    )
    kwargs = dict(
        paths_per_batch=paths,
        batches=batches,
        seed=seed,
        substeps=substeps,
        bump=cert.SPOT_BUMP,
        rqmc_batch_workers=workers,
    )
    if variant == "heston_slv":
        kwargs["slv_spot_bridge_dimensions"] = bridge_dimensions
    else:
        kwargs["heston_spot_bridge_dimensions"] = bridge_dimensions
        kwargs["heston_spot_bridge_strata"] = 1 if bridge_dimensions == 1 else 4

    started = time.perf_counter()
    result = cert.paired_mc_reference(variant, case, product, env, leverage, **kwargs)
    elapsed = time.perf_counter() - started

    scale = _scale(case)
    batch_delta = np.asarray(result.batch_delta, dtype=float)
    contracts = np.array([scale.delta_contracts(d - batch_delta.mean()) for d in batch_delta])
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)
    return {
        "label": label,
        "cell": cell,
        "variant": variant,
        "batches": int(batches),
        "bridge_dimensions": int(bridge_dimensions),
        "delta_mean": float(batch_delta.mean()),
        "delta_se": float(batch_delta.std(ddof=1) / np.sqrt(batch_delta.size)),
        "batch_sd_contracts": float(np.std(contracts, ddof=1)),
        "seconds_per_batch": round(elapsed / batches, 3),
        "peak_rss_gib": round(peak_rss_gib, 2),
    }
```

- [ ] **Step 2: Write the README skeleton**

```markdown
# MC Reference-Stack Convergence — Demos & Decision Matrix

Spec: `docs/superpowers/specs/2026-08-10-mc-reference-convergence-design.md`.
Rows are measured by `demo_cell.py` (harness-faithful stage-16 estimators);
raw outputs in `logs/`. Numbers are filled ONLY from completed runs.

## Decision matrix (V1 treatments, heston_slv variant)

| Cell | Row | batch SD (c) | sec/batch | peak RSS (GiB) | SD factor vs baseline | SE²·sec factor | Unbiasedness |
|---|---|---|---|---|---|---|---|
| (filled by Tasks 4–6) |

## User decisions

- [ ] ordinary_full treatment: __
- [ ] ordinary_decayed treatment: __
- [ ] sigma_collapse treatment: __
- [ ] V2 cross-fitted weights adopted where a control ships: __
```

- [ ] **Step 3: Smoke-run the core (2 batches, cheapest cell) — ~2 min**

Run: `cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -c "
import sys; sys.path.insert(0, 'docs/mc-reference-convergence')
from demo_common import measure_row
print(measure_row('ordinary_full', 'heston_slv', 'smoke', batches=2, seed=1, bridge_dimensions=1, workers=1))"`
Expected: a dict with finite `delta_mean`, `batch_sd_contracts > 0`, no exception.

- [ ] **Step 4: Commit**

```bash
cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification
git add -f docs/mc-reference-convergence/demo_common.py docs/mc-reference-convergence/README.md
git commit -m "docs(mc): demo scaffold + harness-faithful measurement core"
```

---

### Task 4: Treatment demo — `ordinary_full` (heston_slv)

**Files:**
- Create: `docs/mc-reference-convergence/demo_cell.py`
- Create (output): `docs/mc-reference-convergence/logs/ordinary_full.jsonl`
- Modify: `docs/mc-reference-convergence/README.md` (fill measured rows)

**Interfaces:**
- Consumes: `measure_row` from Task 3.
- Produces: JSONL rows per treatment; README matrix rows with the V1-G1/G2/G3 verdicts.

- [ ] **Step 1: Write the CLI demo**

```python
# docs/mc-reference-convergence/demo_cell.py
"""Run the V1 treatment matrix for one cell and append JSONL rows.

Rows: baseline (production profile, bridge dims 1, seed A) — the SD/cost
anchor; bridge8 (dims 8, seed A) — the treatment candidate; unbias (bridge8,
independent seed B, matched budget) — V1-G1 evidence that the treated
estimator agrees with the baseline within 2x combined SE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_common import measure_row

ROWS = (
    ("baseline", 1, 20260810),
    ("bridge8", 8, 20260810),
    ("unbias", 8, 20260811),  # independent seed for the agreement check
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--variant", default="heston_slv")
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    out = Path(__file__).resolve().parent / "logs" / f"{args.cell}.jsonl"
    out.parent.mkdir(exist_ok=True)
    rows = {}
    for label, dims, seed in ROWS:
        record = measure_row(
            args.cell, args.variant, label,
            batches=args.batches, seed=seed,
            bridge_dimensions=dims, workers=args.workers,
        )
        rows[label] = record
        with out.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    base, treat, unbias = rows["baseline"], rows["bridge8"], rows["unbias"]
    sd_factor = base["batch_sd_contracts"] / max(treat["batch_sd_contracts"], 1e-12)
    eff_factor = (
        (base["batch_sd_contracts"] ** 2 * base["seconds_per_batch"])
        / max(treat["batch_sd_contracts"] ** 2 * treat["seconds_per_batch"], 1e-12)
    )
    combined_se = (base["delta_se"] ** 2 + unbias["delta_se"] ** 2) ** 0.5
    agreement = abs(base["delta_mean"] - unbias["delta_mean"]) / max(combined_se, 1e-12)
    summary = {
        "cell": args.cell,
        "sd_factor": round(sd_factor, 2),
        "se2_sec_factor": round(eff_factor, 2),
        "unbias_sigma": round(agreement, 2),
        "v1_g1_pass": bool(agreement <= 2.0),
        "v1_g2_pass": bool(sd_factor >= 4.0),
    }
    with out.open("a") as handle:
        handle.write(json.dumps({"summary": summary}) + "\n")
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run for ordinary_full in background — expect ~30–60 min (3 × 32 SLV batches)**

Run (background): `cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python docs/mc-reference-convergence/demo_cell.py --cell ordinary_full`
Expected: three JSONL rows + a summary row with `sd_factor`, `se2_sec_factor`, `unbias_sigma` all finite.

- [ ] **Step 3: Fill the README matrix row from the completed log (actual numbers only)**

Copy `batch_sd_contracts`, `seconds_per_batch`, `peak_rss_gib`, `sd_factor`, `se2_sec_factor`, and the V1-G1 verdict from `logs/ordinary_full.jsonl` into the README table. If `v1_g2_pass` is false, record the row anyway — the decision matrix, not the plan, decides shipping.

- [ ] **Step 4: Commit**

```bash
cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification
git add -f docs/mc-reference-convergence/demo_cell.py docs/mc-reference-convergence/logs/ordinary_full.jsonl docs/mc-reference-convergence/README.md
git commit -m "docs(mc): ordinary_full treatment demo — measured V1 matrix row"
```

---

### Task 5: Treatment demo — `ordinary_decayed` (heston_slv)

**Files:**
- Create (output): `docs/mc-reference-convergence/logs/ordinary_decayed.jsonl`
- Modify: `docs/mc-reference-convergence/README.md`

**Interfaces:**
- Consumes: `demo_cell.py` from Task 4 (unchanged).
- Produces: measured matrix row for `ordinary_decayed`.

- [ ] **Step 1: Run in background — ~30–60 min**

Run (background): `cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python docs/mc-reference-convergence/demo_cell.py --cell ordinary_decayed`
Expected: three JSONL rows + summary, all finite.

- [ ] **Step 2: Fill the README row from the completed log**

Same columns as Task 4 Step 3, from `logs/ordinary_decayed.jsonl`.

- [ ] **Step 3: Commit**

```bash
cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification
git add -f docs/mc-reference-convergence/logs/ordinary_decayed.jsonl docs/mc-reference-convergence/README.md
git commit -m "docs(mc): ordinary_decayed treatment demo — measured V1 matrix row"
```

---

### Task 6: Treatment demo — `sigma_collapse` (heston_slv) + V2 pilot on the winning rows

**Files:**
- Create (output): `docs/mc-reference-convergence/logs/sigma_collapse.jsonl`
- Create: `docs/mc-reference-convergence/demo_v2_weights.py`
- Create (output): `docs/mc-reference-convergence/logs/v2_weights.jsonl`
- Modify: `docs/mc-reference-convergence/README.md`

**Interfaces:**
- Consumes: `demo_cell.py`, `measure_row`, `cross_fitted_control` (Task 1).
- Produces: sigma_collapse matrix row; V2 variance-ratio evidence per treated cell.

- [ ] **Step 1: Run the sigma_collapse matrix in background — ~30–60 min**

Run (background): `cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python docs/mc-reference-convergence/demo_cell.py --cell sigma_collapse`
Expected: three JSONL rows + summary, all finite.

- [ ] **Step 2: Write the V2 demo (cross-fit on paired heston/heston_slv batch deltas)**

```python
# docs/mc-reference-convergence/demo_v2_weights.py
"""V2 evidence: cross-fitted Heston-control weights on real cell batch means.

For each cell, generate matched-seed heston_slv (primary) and heston
(control) batch deltas via the harness estimator, feed them to
cross_fitted_control with the heston run's own mean as the control
expectation (independent seed), and record the variance ratio. This is the
decision-matrix input for adopting per-cell weights over the frozen 0.7.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_common import cert  # the loaded stage-16 module

from quantark.montecarlo.control_weights import cross_fitted_control

CELLS = ("ordinary_full", "ordinary_decayed", "sigma_collapse")


def batch_deltas(cell: str, variant: str, seed: int, batches: int = 32):
    import math
    case = next(c for c in cert.certification_cases(quick=False) if c.name == cell)
    product = cert.make_snowball(case, dense_ki=True)
    env = cert.make_environment(case.spot, math.sqrt(max(case.params.v0, case.params.theta)))
    leverage = cert.make_leverage_surface(case.maturity) if variant == "heston_slv" else None
    substeps = cert.PRODUCTION_QE_SUBSTEPS_BY_VARIANT_CASE[variant][cell]["target"]
    paths = (
        cert.PRODUCTION_SLV_PATHS_PER_BATCH if variant == "heston_slv"
        else cert.PRODUCTION_HESTON_PATHS_PER_BATCH
    )
    result = cert.paired_mc_reference(
        variant, case, product, env, leverage,
        paths_per_batch=paths, batches=batches, seed=seed,
        substeps=substeps, bump=cert.SPOT_BUMP, rqmc_batch_workers=2,
    )
    return np.asarray(result.batch_delta, dtype=float)


def main() -> None:
    out = Path(__file__).resolve().parent / "logs" / "v2_weights.jsonl"
    for cell in CELLS:
        primary = batch_deltas(cell, "heston_slv", seed=20260810)
        control = batch_deltas(cell, "heston", seed=20260810)      # matched scrambles
        expectation = float(batch_deltas(cell, "heston", seed=20260812).mean())  # independent
        fitted = cross_fitted_control(primary, control, control_expectation=expectation)
        fixed_07 = primary - 0.7 * (control - expectation)
        primary_se = float(primary.std(ddof=1) / np.sqrt(primary.size))
        record = {
            "cell": cell,
            **fitted.as_dict(),
            # V2-G1: cross-fitted estimator must agree with the fixed-0.7
            # estimator within combined SE (both are unbiased for E[primary]).
            "adjusted_mean": float(fitted.adjusted.mean()),
            "fixed_07_mean": float(fixed_07.mean()),
            "primary_se": primary_se,
            "v2_g1_sigma": float(
                abs(fitted.adjusted.mean() - fixed_07.mean())
                / max(
                    (np.var(fitted.adjusted, ddof=1) / fitted.adjusted.size
                     + np.var(fixed_07, ddof=1) / fixed_07.size) ** 0.5,
                    1e-12,
                )
            ),
        }
        with out.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the V2 demo in background — ~1–2 h (3 cells × 3 runs of 32 batches)**

Run (background): `cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python docs/mc-reference-convergence/demo_v2_weights.py`
Expected: one JSONL record per cell with `weights` (2 floats) and `variance_ratio`.

- [ ] **Step 4: Fill the README rows from both completed logs; commit**

```bash
cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification
git add -f docs/mc-reference-convergence/demo_v2_weights.py docs/mc-reference-convergence/logs/ docs/mc-reference-convergence/README.md
git commit -m "docs(mc): sigma_collapse demo + V2 cross-fitted weight evidence"
```

---

### Task 7: USER CHECKPOINT — decision matrix

**Files:**
- Modify: `docs/mc-reference-convergence/README.md` (record decisions)

This is a hard stop (spec §7 D1→D2 gate; house rule standalone-demo-first).
Present the completed matrix to the user: per cell, the measured SD factor,
SE²·sec factor, peak RSS, unbiasedness sigma, and the V2 variance ratios.
The user picks, per cell: `baseline` (no change) or `bridge8`, and whether V2
weights ship where a control exists. Record the choices in the README's
"User decisions" checklist and commit. **Do not proceed to Task 8 without
recorded decisions.** A cell whose demo underdelivered stays `baseline` —
Task 9's allocation absorbs it.

```bash
cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification
git add -f docs/mc-reference-convergence/README.md
git commit -m "docs(mc): record user treatment decisions (V1/V2 matrix)"
```

---

### Task 8: Stage-16 wiring — per-cell treatment profiles + schema-12 treatment descriptors

**Files:**
- Modify: `example/mo_volmodels/16_adi_greek_certification.py` (the `SLV_SPOT_BRIDGE_PROFILE_BY_CASE` dict, currently at ~line 261, and the evidence-payload builder)
- Test: `test/mo_volmodels/test_adi_greek_certification.py` (extend)

**Interfaces:**
- Consumes: the recorded Task 7 decisions; existing `SLV_SPOT_BRIDGE_PROFILE_BY_CASE` shape `{"strata": int, "dimensions": int}`.
- Produces: schema-12 evidence field `cells[i].reference_treatment` = `{"bridge_strata": int, "bridge_dimensions": int, "control": "none" | "cross_fitted", "control_weights": [float, ...] | null}`; schema id bumped to 12 wherever the payload records it.

- [ ] **Step 1: Write the failing test**

```python
# append to test/mo_volmodels/test_adi_greek_certification.py
# NOTE: match the file's existing loader convention — these test files load the
# stage script via a module-level _load() helper, not a pytest fixture.
def test_reference_treatment_descriptor_matches_profiles():
    module = _load()
    # Whatever Task 7 decided, the profile dict and the evidence descriptor
    # must agree, cell by cell — no silent treatment drift.
    for case_name, profile in module.SLV_SPOT_BRIDGE_PROFILE_BY_CASE.items():
        descriptor = module.reference_treatment_descriptor("heston_slv", case_name)
        assert descriptor["bridge_strata"] == profile["strata"]
        assert descriptor["bridge_dimensions"] == profile["dimensions"]
        assert descriptor["control"] in ("none", "cross_fitted")
```

- [ ] **Step 2: Run to verify it fails** (`reference_treatment_descriptor` undefined)

Run: `cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/mo_volmodels/test_adi_greek_certification.py -k schema12 -v`
Expected: FAIL with AttributeError.

- [ ] **Step 3: Implement**

In `16_adi_greek_certification.py`:
1. Update `SLV_SPOT_BRIDGE_PROFILE_BY_CASE` entries for exactly the cells the
   user approved in Task 7 (e.g. `"ordinary_full": {"strata": SLV_SPOT_BRIDGE_STRATA, "dimensions": 8}`) — leave rejected cells untouched, and leave
   `low_feller` untouched regardless (its direct-estimator decision is recorded evidence).
2. Add, next to the profile dicts:

```python
def reference_treatment_descriptor(variant: str, case_name: str) -> dict:
    """Schema-12: the exact treatment this cell's reference was built with."""
    if variant == "heston_slv":
        profile = SLV_SPOT_BRIDGE_PROFILE_BY_CASE[case_name]
    else:
        profile = HESTON_SPOT_BRIDGE_PROFILE_BY_CASE[case_name]
    control = SCHEMA12_CONTROL_BY_VARIANT_CASE.get(variant, {}).get(case_name, "none")
    return {
        "bridge_strata": int(profile["strata"]),
        "bridge_dimensions": int(profile["dimensions"]),
        "control": control,
        "control_weights": None,  # populated at run time when control == "cross_fitted"
    }
```

with `SCHEMA12_CONTROL_BY_VARIANT_CASE` a module-level dict holding exactly the
Task 7 decisions (empty inner dicts if the user rejected all controls).
3. In the cell-evidence payload builder (where each cell dict gains
   `certifications`), add `"reference_treatment": reference_treatment_descriptor(variant, case.name)`.
4. Schema numbering: follow the harness's own sequence — grep stage-16 for its
   schema-id constant and the stage-17 tests for the pinned parent schema
   (`test_schema12_parent_and_development_families_are_pinned` already exists).
   The descriptor changes the payload shape, so bump the stage-16 payload id by
   exactly one from whatever it currently is and update every test that pins
   it. Do not renumber the stage-17 amendment schema.

- [ ] **Step 4: Run the stage-16 test file**

Run: `cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/mo_volmodels/test_adi_greek_certification.py -v`
Expected: all PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification
git add example/mo_volmodels/16_adi_greek_certification.py test/mo_volmodels/test_adi_greek_certification.py
git commit -m "feat(cert): schema-12 per-cell treatment profiles + descriptors (user-approved matrix)"
```

---

### Task 9: Stage-17 wiring — pilot → frozen allocation → precision monitor (with fallback)

**Files:**
- Modify: `example/mo_volmodels/17_adi_slv_aggregate_certification.py`
- Test: `test/mo_volmodels/test_adi_slv_aggregate_certification.py` (extend)

**Interfaces:**
- Consumes: `CellPrecision`, `neyman_allocation`, `precision_stop`, `StopDecision` (Task 2 exports from `quantark.validation`); existing stage-17 structures: `run_development_pilot`, `production_run_configuration`, `frozen_allocation_manifest`, the banking loop that appends batches per cell, and the evidence dict it publishes.
- Produces: CLI flags `--adaptive` (default off → S-G4 fallback = frozen 4096/256 allocation, untouched), `--precision-target 0.02`, `--budget-hours 12.0`, `--pilot-batches 32`; evidence fields
  `adaptive_run = {"pilot": {cell: {"batches": int, "batch_sd": float, "seconds_per_batch": float}}, "allocation": {cell: int}, "allocation_sha256": str, "precision_target": float, "budget_hours": float, "stopping": {"trigger": str, "projected_halfwidth": float, "checks": int}}`.

- [ ] **Step 1: Write the failing tests**

```python
# append to test/mo_volmodels/test_adi_slv_aggregate_certification.py
# NOTE: this file loads the stage script via its module-level _load() helper.

def test_adaptive_flag_default_off_preserves_frozen_allocation():
    module = _load()
    # S-G4: without --adaptive the run configuration is bit-identical to the
    # frozen manifest — the hash pin still binds.
    config = module.production_run_configuration(adaptive=False)
    manifest = module.frozen_allocation_manifest()
    assert config["allocation"]["primary_batches"] == manifest["production"]["primary_batches"]
    assert config["allocation"]["middle_batches"] == manifest["production"]["middle_batches"]


def test_adaptive_allocation_is_pilot_deterministic():
    module = _load()
    # S-G3: same pilot stats -> same frozen allocation and same sha256.
    pilot = {
        "ordinary_full": {"batches": 32, "batch_sd": 1.07, "seconds_per_batch": 40.0},
        "ordinary_decayed": {"batches": 32, "batch_sd": 1.08, "seconds_per_batch": 41.0},
        "sigma_collapse": {"batches": 32, "batch_sd": 0.70, "seconds_per_batch": 55.0},
    }
    a = module.freeze_adaptive_allocation(pilot, budget_hours=12.0)
    b = module.freeze_adaptive_allocation(pilot, budget_hours=12.0)
    assert a == b
    assert a["allocation_sha256"] == b["allocation_sha256"]


def test_monitor_hook_is_estimate_blind():
    module = _load()
    # S-G1 at the wiring level: the monitor input builder only exposes
    # precision fields, whatever the banked records contain.
    banked = {
        "ordinary_full": {
            "batch_deltas": np.array([0.1, 0.2, 0.15, 0.12]),
            "seconds_per_batch": 40.0,
        }
    }
    cells = module.monitor_cells_from_banked(banked)
    assert all(not hasattr(c, "estimate") for c in cells)
    assert cells[0].n_batches == 4 and cells[0].seconds_per_batch == 40.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/mo_volmodels/test_adi_slv_aggregate_certification.py -k "adaptive or monitor" -v`
Expected: FAIL (missing `adaptive` kwarg, `freeze_adaptive_allocation`, `monitor_cells_from_banked`).

- [ ] **Step 3: Implement in `17_adi_slv_aggregate_certification.py`**

1. `production_run_configuration(..., adaptive: bool = False)`: when False,
   return exactly the current frozen configuration (no other change); when
   True, add the `adaptive_run` sub-dict (fields above).
2. New pure functions next to `frozen_allocation_manifest`:

```python
def freeze_adaptive_allocation(pilot: dict, *, budget_hours: float) -> dict:
    """Turn pilot precision stats into a frozen, hash-pinned batch allocation."""
    cells = [
        CellPrecision(
            name=name,
            n_batches=int(stats["batches"]),
            batch_sd=float(stats["batch_sd"]),
            seconds_per_batch=float(stats["seconds_per_batch"]),
        )
        for name, stats in sorted(pilot.items())
    ]
    allocation = neyman_allocation(
        cells, budget_seconds=budget_hours * 3600.0, min_batches=MIN_PRODUCTION_RQMC_BATCHES
    )
    frozen = {"pilot": pilot, "allocation": allocation, "budget_hours": float(budget_hours)}
    frozen["allocation_sha256"] = _canonical_sha256(frozen)
    return frozen


def monitor_cells_from_banked(banked: dict) -> list:
    """Precision-only view of the banked batches (estimate-blind by type)."""
    cells = []
    for name, record in sorted(banked.items()):
        deltas = np.asarray(record["batch_deltas"], dtype=float)
        cells.append(
            CellPrecision(
                name=name,
                n_batches=int(deltas.size),
                batch_sd=float(np.std(deltas, ddof=1)),
                seconds_per_batch=float(record["seconds_per_batch"]),
            )
        )
    return cells
```

   with `from quantark.validation import CellPrecision, neyman_allocation, precision_stop`
   added to the imports, and `MIN_PRODUCTION_RQMC_BATCHES = 16` mirrored from
   stage-16 if not already present.
3. Build on the machinery that already exists rather than beside it: stage-17
   already has `--development-pilot` (`run_development_pilot`) and
   `--project-allocation --pilot-evidence` (the flow that produced the frozen
   allocation). Wire `freeze_adaptive_allocation` into the
   `--project-allocation` output so the projection now also records the
   Neyman allocation + `allocation_sha256`. Then, in the production banking
   loop under the new `--adaptive` flag: after the `--pilot-batches` pilot
   cohort completes, freeze the allocation and write it into the evidence
   *before* the main run; after each completed cohort call
   `precision_stop(monitor_cells_from_banked(banked), target_halfwidth=args.precision_target, elapsed_seconds=…, budget_seconds=args.budget_hours * 3600)`
   and halt banking when `decision.stop`, recording `decision.as_dict()` plus
   the number of monitor checks under `adaptive_run["stopping"]`.
4. Verdict path unchanged: the existing `make_aggregate_decisions` /
   `certify_signed_bias_from_independent_cohorts` calls run once, after banking
   stops, on everything banked.

- [ ] **Step 4: Run the stage-17 test file**

Run: `cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/mo_volmodels/test_adi_slv_aggregate_certification.py -v`
Expected: all PASS (existing + 3 new).

- [ ] **Step 5: Commit**

```bash
cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification
git add example/mo_volmodels/17_adi_slv_aggregate_certification.py test/mo_volmodels/test_adi_slv_aggregate_certification.py
git commit -m "feat(cert): pilot->Neyman->precision-stop adaptive banking with frozen-allocation fallback"
```

---

### Task 10: End-to-end rehearsal + plan-of-record update

**Files:**
- Modify: `docs/adi2d-greek-perf/STATE-AND-PLAN-2026-08-10.md` (P1.3/P1.4 mechanics now delegated to this program)
- Create (output): `docs/mc-reference-convergence/logs/rehearsal.log`

**Interfaces:**
- Consumes: everything above; stage-17's existing `--quick`-style development path (`run_development_pilot`).

- [ ] **Step 1: Quick-mode rehearsal in background — ~1–2 h**

Run (background): `cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python example/mo_volmodels/17_adi_slv_aggregate_certification.py --adaptive --pilot-batches 8 --precision-target 0.5 --budget-hours 0.5 --output-dir output/adaptive_rehearsal 2>&1 | tee docs/mc-reference-convergence/logs/rehearsal.log`
(Loose target/budget so the rehearsal exercises `target-reached` cheaply; then rerun with `--precision-target 0.0001 --budget-hours 0.05` to exercise `budget-cap`.)
Expected: evidence JSON contains `adaptive_run` with a frozen allocation sha256 and a stopping record; the second run stops with `trigger == "budget-cap"`; verdict machinery runs once in both.

- [ ] **Step 2: Update the plan of record**

In `STATE-AND-PLAN-2026-08-10.md` §4 Phase 1, point P1.3/P1.4 at this
program: reference regeneration runs `--adaptive` with the Task 7 treatments,
target 0.02 contracts, cap 12 h; frozen 4096/256 remains the recorded
fallback. Note the rehearsal log path.

- [ ] **Step 3: Full test sweep**

Run: `cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/test_control_weights.py test/test_adaptive_allocation.py test/mo_volmodels/ -v`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
cd /Users/fuxinyao/quant-ark/.worktrees/adi-greek-certification
git add -f docs/adi2d-greek-perf/STATE-AND-PLAN-2026-08-10.md docs/mc-reference-convergence/logs/rehearsal.log
git commit -m "docs(cert): adaptive-banking rehearsal evidence + plan-of-record update"
```

---

## Not in this plan (deliberately)

- The production schema-12 regeneration itself (runs after the WS-C scheme fix
  lands, per STATE-AND-PLAN Phase 1 — regenerate exactly once).
- Multilevel mid-control extension to new cells (only if a Task 4–6 demo
  underdelivers AND the user asks; it was measured-rejected on `low_feller`).
- Memory/dtype/streaming work (descoped by user 2026-08-10).
- Fleet/backtest MC engines (own spec later).
