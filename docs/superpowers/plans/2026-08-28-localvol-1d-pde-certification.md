# Local-Vol 1-D Snowball PDE Certification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Certify `LocalVolSnowballPDESolver` as study `snowball-localvol-1d` against a demonstrably converged local-vol Monte-Carlo reference, on two real calibrated CSI1000 Dupire surfaces, and bank the evidence.

**Architecture:** One new builder module registers three builders (environment / candidate / reference) into `quantark/modelvalidation/`'s registry, exactly as `equity_snowball.py` does for flat Black-Scholes. A YAML study names them, two committed surface artifacts supply the local-vol surfaces, and the existing `certify()` pipeline does the rest. The product builder is **reused unchanged** from the flat-BSM study so both certificates describe the same trade construction.

**Tech Stack:** Python 3.11, NumPy, pytest (with `pytest-xdist`), `quantark.modelvalidation`, `quantark.volmodels.localvol`, `quantark.param.vol.surface_history`.

**Spec:** `docs/superpowers/specs/2026-08-28-localvol-1d-pde-certification-design.md`

## Global Constraints

- **Working directory is the worktree:** `/Users/fuxinyao/quant-ark/.claude/worktrees/localvol-1d-certification`. Run every command from there. Do not `cd` to the main checkout.
- **The worktree has no `.venv`.** The editable install resolves `quantark` to the *main* repo, which would silently test the wrong source. Every Python and pytest invocation MUST shadow it:
  `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest ...`
- **Never `git stash` / `git stash pop`.** The stash stack is shared with the main checkout and other sessions.
- **Reference discretization is one value for all quantities:** `substeps_per_interval = 8`, `lv_time_sampling = "integrated"`. Never split the discretization between PV and Greeks (spec §5).
- **The local-vol surface is built once per (artifact, rate) and reused across spot bumps.** Rebuilding it at a bumped spot would fold a surface-rebuild derivative into delta.
- **Never re-smooth the stored `iv_grid`.** It is already `sabr_calendar_projected`.
- **Bounds are fixed:** `cell: 0.5`, `mean_signed_bias: 0.1`. Never widen a bound to convert `INCONCLUSIVE`/`REJECTED` into `ADMITTED`.
- **Market:** flat `rate: 0.02` (`FLAT_RATE`), carry from each artifact's own per-expiry parity pillars.
- `REFERENCE_SPOT = 4993.105` — the economic-scale basis (spec §8).
- Commit messages end with:
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`

---

## File Structure

| Path | Responsibility |
|---|---|
| `example/modelvalidation/data/iv_surface_20240208.json` | Crash-regime Dupire surface, committed (new) |
| `example/modelvalidation/data/iv_surface_20231115.json` | Calm-regime Dupire surface, committed (new) |
| `quantark/modelvalidation/builders/equity_snowball_localvol.py` | All three builders + shared spec resolution (new) |
| `quantark/modelvalidation/builders/__init__.py` | Register the new module (modify) |
| `example/modelvalidation/snowball_localvol_1d.yaml` | The study: 16 cases, bounds, sampling (new) |
| `test/modelvalidation/test_localvol_study.py` | Wiring + soundness tests (new) |
| `docs/modelvalidation/pilot-localvol-1d/` | Pilot probes and `RESULTS.md` (new) |
| `docs/modelvalidation/certificates/snowball-localvol-1d/2026-08-28/` | Banked evidence (new, Task 9) |

`test/modelvalidation/test_banked_certificates.py` needs **no change** — it already globs `certificates/*/*/anchors.json`, so banking wires the anchor guard into CI automatically.

### One refinement to spec §8

The spec pins `contract_multiplier` per surface as a literal (`1.000000`, `0.804397`). Because arms own construction (`yaml_loader.py:191` resolves the environment builder but never calls it), the arm can compute it: `contract_multiplier = REFERENCE_SPOT / artifact.s0`. Product levels are likewise declared as **moneyness** and resolved against the artifact's own `s0`. Same numbers, no transcription risk, and the eight case shapes are written once per surface instead of being hand-scaled twice.

---

### Task 1: Surface artifacts and the environment builder

**Files:**
- Create: `example/modelvalidation/data/iv_surface_20240208.json`
- Create: `example/modelvalidation/data/iv_surface_20231115.json`
- Create: `quantark/modelvalidation/builders/equity_snowball_localvol.py`
- Modify: `quantark/modelvalidation/builders/__init__.py`
- Test: `test/modelvalidation/test_localvol_study.py`

**Interfaces:**
- Consumes: `IvSurfaceArtifact.from_file`, `build_dupire_local_vol`, `build_snowball_product_spec`, `make_snowball` (all existing).
- Produces:
  - `REFERENCE_SPOT: float = 4993.105`
  - `load_surface(path: str, rate: float) -> _Surface` (cached), where `_Surface` has fields `artifact, grid, rate_curve, div_yield, local_vol`
  - `build_localvol_market_spec(params: Mapping[str, Any]) -> dict` — registered environment builder `equity.snowball.localvol_market`
  - `resolve_product_spec(environment: Mapping, product: Mapping, s0: float) -> dict` — moneyness → absolute
  - `make_localvol_environment(spec: Mapping[str, Any], spot: float | None = None) -> PricingEnvironment`

- [ ] **Step 1: Copy the two surface artifacts and verify they are committable**

```bash
mkdir -p example/modelvalidation/data
cp /Users/fuxinyao/quant-ark/example/mo_volmodels/data/history/iv_surface/mo_iv_surface_20240208.json \
   example/modelvalidation/data/iv_surface_20240208.json
cp /Users/fuxinyao/quant-ark/example/mo_volmodels/data/history/iv_surface/mo_iv_surface_20231115.json \
   example/modelvalidation/data/iv_surface_20231115.json
git check-ignore -v example/modelvalidation/data/iv_surface_20240208.json && echo "IGNORED - STOP" || echo "OK"
```

Expected: `OK` (not ignored). If `IGNORED - STOP`, the whole reproducibility argument of spec §3 fails — stop and report.

- [ ] **Step 2: Write the failing test**

Create `test/modelvalidation/test_localvol_study.py`:

```python
"""Wiring and soundness tests for the local-vol snowball certification study.

These assert SOUNDNESS, not outcome. At the tiny sampling budget used here the
benchmark cannot meet its standard-error budget, so INCONCLUSIVE is correct;
asserting ADMITTED would be asserting that noise agrees with us. The real
verdict comes from the offline run whose evidence is banked.
"""

from pathlib import Path

import pytest

from quantark.modelvalidation.builders.equity_snowball_localvol import (
    REFERENCE_SPOT,
    build_localvol_market_spec,
    load_surface,
    make_localvol_environment,
)
from quantark.util.exceptions import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "example" / "modelvalidation" / "data"

CRASH = "example/modelvalidation/data/iv_surface_20240208.json"
CALM = "example/modelvalidation/data/iv_surface_20231115.json"

# Pinned from the artifacts as committed. A change here means the surface
# bytes moved, which invalidates every certificate built on them.
CRASH_S0 = 4993.105
CALM_S0 = 6207.268
CRASH_SHA16 = "b0e63653a774b5b3"
CALM_SHA16 = "a7917303394e114f"


def test_both_surface_artifacts_are_present():
    assert (DATA / "iv_surface_20240208.json").is_file()
    assert (DATA / "iv_surface_20231115.json").is_file()


@pytest.mark.parametrize(
    "path, s0, sha16",
    [(CRASH, CRASH_S0, CRASH_SHA16), (CALM, CALM_S0, CALM_SHA16)],
)
def test_artifact_identity_is_pinned(path, s0, sha16):
    """The sha is what pins a certificate to exact surface bytes."""
    surface = load_surface(path, 0.02)
    assert surface.artifact.s0 == pytest.approx(s0, abs=1e-3)
    assert surface.artifact.sha256.startswith(sha16)


def test_surface_is_cached_not_rebuilt():
    """Rebuilding per call would re-run Dupire on every bumped price."""
    assert load_surface(CRASH, 0.02) is load_surface(CRASH, 0.02)


def test_local_vol_is_built_at_the_artifact_spot():
    """Not at a bumped spot -- otherwise delta absorbs a surface-rebuild term."""
    surface = load_surface(CRASH, 0.02)
    bumped = make_localvol_environment(
        {"surface": CRASH, "rate": 0.02}, spot=CRASH_S0 * 1.01
    )
    assert bumped.spot == pytest.approx(CRASH_S0 * 1.01)
    # The surface object handed to the engines is the same one regardless.
    assert load_surface(CRASH, 0.02).local_vol is surface.local_vol


def test_environment_carries_the_artifact_trade_date_and_carry():
    env = make_localvol_environment({"surface": CRASH, "rate": 0.02})
    assert env.valuation_date.date().isoformat() == "2024-02-08"
    assert env.spot == pytest.approx(CRASH_S0, abs=1e-3)


def test_unknown_environment_key_is_refused():
    with pytest.raises(ValidationError, match="localvol_market"):
        build_localvol_market_spec({"surface": CRASH, "rate": 0.02, "vol": 0.2})


def test_missing_surface_path_is_refused():
    with pytest.raises(ValidationError, match="surface"):
        build_localvol_market_spec({"rate": 0.02})
```

- [ ] **Step 3: Run the test to verify it fails**

Run:
```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 \
  test/modelvalidation/test_localvol_study.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'quantark.modelvalidation.builders.equity_snowball_localvol'`.

- [ ] **Step 4: Write the builder module**

Create `quantark/modelvalidation/builders/equity_snowball_localvol.py`:

```python
"""Snowball under a real calibrated Dupire local-volatility surface.

The 1-D local-vol counterpart of ``equity_snowball.py``. One deterministic
engine -- ``LocalVolSnowballPDESolver`` -- is certified against a local-vol
Monte-Carlo reference on two real CSI1000 surfaces: the 2024-02-08 crash bottom
and the calm 2023-11-15, chosen as the cohort's steepest and flattest by Dupire
local-vol slope.

Two things differ from the flat-BSM study, and both are deliberate.

**The surfaces are committed data, not generated.** ``example/mo_volmodels/data/
history`` is excluded through ``.git/info/exclude``, a per-clone file that is
never pushed, so a study reading from it would bank a certificate whose CI
anchors fail everywhere but one machine. The artifacts live under
``example/modelvalidation/data/`` and their sha256 enters the identity hash, so
the certificate pins the exact surface bytes it was certified against.

**Levels are declared as moneyness and resolved against each artifact's own
s0.** The two surfaces sit at different index levels (4993.105 and 6207.268), and
``economic_scale`` is a single study-level block. Resolving here lets one set of
case shapes serve both surfaces, and lets ``contract_multiplier`` be *computed*
as ``REFERENCE_SPOT / s0`` rather than transcribed -- uncorrected, every
calm-surface error would be overstated by 1.243x, which risks a false REJECTED
rather than a merely conservative pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (
    LocalVolSnowballMCEngine,
)
from quantark.asset.equity.engine.pde.grid.config import resolve_config
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    LocalVolSnowballPDESolver,
)
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.param import FlatRateCurve, GridVolSurface, SpotQuote
from quantark.param.div import TermStructureDividendYield
from quantark.param.vol.surface_history import IvSurfaceArtifact
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError
from quantark.volmodels.localvol import LocalVolSurface, build_dupire_local_vol

from quantark.modelvalidation.builders.equity_snowball import (
    _central_difference_greeks,
    build_snowball_product_spec,
    make_snowball,
)
from quantark.modelvalidation.candidate import CandidateResult, LadderRung
from quantark.modelvalidation.engine_config import engine_config
from quantark.modelvalidation.reference import BatchResult
from quantark.modelvalidation.registry import register_builder
from quantark.modelvalidation.study import SamplingPolicy

#: The economic-scale basis. ``notional = 200 * REFERENCE_SPOT`` makes
#: ``delta_quantum`` exactly 1.0, matching the flat-BSM study's normalization,
#: so raw delta reads directly as hedge contracts on both certificates.
REFERENCE_SPOT = 4993.105

#: Repo root, for resolving study-relative artifact paths. The study is run from
#: the repo root, but anchors replay from pytest, so this is derived from the
#: module location rather than from the process working directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]

_ENVIRONMENT_KEYS = ("surface", "rate", "spot_moneyness", "asset_name")

#: Product keys this study accepts. Levels are MONEYNESS; everything else is
#: passed through to the flat-BSM product builder unchanged.
_PRODUCT_KEYS = (
    "strike_moneyness",
    "ko_barrier_moneyness",
    "ki_barrier_moneyness",
    "ko_rate",
    "rebate_rate",
    "months",
    "maturity",
    "ki_monitoring",
    "ko_stepdown",
)

#: One profile coarser than each target, for the refinement ladder.
_COARSER_ACCURACY = {"high": "standard", "standard": "fast", "fast": "fast"}


@dataclass(frozen=True)
class _Surface:
    """A loaded artifact and everything derived from it, built once."""

    artifact: IvSurfaceArtifact
    grid: GridVolSurface
    rate_curve: FlatRateCurve
    div_yield: TermStructureDividendYield
    local_vol: LocalVolSurface


def _resolve_path(surface: str) -> Path:
    path = Path(surface)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    if not path.is_file():
        raise ValidationError(
            f"equity.snowball.localvol_market surface artifact not found: {path}. "
            "Study surfaces live under example/modelvalidation/data/ and are "
            "committed; they are NOT read from example/mo_volmodels/data/history, "
            "which is excluded per-clone and would not exist in CI."
        )
    return path


@lru_cache(maxsize=8)
def load_surface(surface: str, rate: float) -> _Surface:
    """Load an artifact and build its Dupire surface, once per (path, rate).

    The local-vol surface is built at the artifact's OWN s0 and reused across
    spot bumps. Rebuilding it at a bumped spot would make delta a derivative of
    the surface construction as well as of the price.

    The stored ``iv_grid`` is already SABR-smoothed and calendar-projected, so
    it is used as-is: Dupire differentiates total variance twice in strike, and
    smoothing a second time would certify a different surface from the one the
    artifact names.
    """
    artifact = IvSurfaceArtifact.from_file(_resolve_path(surface))
    grid = artifact.grid_vol_surface()
    rate_curve = FlatRateCurve(float(rate))
    div_yield = artifact.term_structure_dividend_yield(float(rate))
    local_vol = build_dupire_local_vol(
        grid,
        spot=float(artifact.s0),
        rate_curve=rate_curve,
        div_yield=div_yield.get_yield,
    )
    return _Surface(artifact, grid, rate_curve, div_yield, local_vol)


@register_builder("equity.snowball.localvol_market", kind="environment")
def build_localvol_market_spec(params: Mapping[str, Any]) -> dict:
    """Validate a local-vol market spec."""
    spec = dict(params)
    unknown = set(spec) - set(_ENVIRONMENT_KEYS)
    if unknown:
        raise ValidationError(
            f"Unknown equity.snowball.localvol_market keys: {sorted(unknown)}; "
            f"expected a subset of {_ENVIRONMENT_KEYS}"
        )
    for key in ("surface", "rate"):
        if key not in spec:
            raise ValidationError(
                f"equity.snowball.localvol_market is missing {key!r}"
            )
    moneyness = float(spec.get("spot_moneyness", 1.0))
    if not moneyness > 0.0:
        raise ValidationError(
            f"spot_moneyness must be positive, got {moneyness}"
        )
    return spec


def make_localvol_environment(
    spec: Mapping[str, Any], spot: float | None = None
) -> PricingEnvironment:
    """The market one case prices in, optionally at a bumped spot.

    ``valuation_date`` is the artifact's own trade date, so the study is not
    calendar-bound and the surface's maturities mean what they say.
    """
    surface = load_surface(str(spec["surface"]), float(spec["rate"]))
    resolved = (
        float(spec.get("spot_moneyness", 1.0)) * float(surface.artifact.s0)
        if spot is None
        else float(spot)
    )
    trade_date = surface.artifact.trade_date
    return PricingEnvironment(
        rate_curve=surface.rate_curve,
        valuation_date=datetime(trade_date.year, trade_date.month, trade_date.day),
        spot_quote=SpotQuote(
            resolved, asset_name=str(spec.get("asset_name", "CSI1000"))
        ),
        vol_surface=surface.grid,
        div_yield=surface.div_yield,
    )


def resolve_product_spec(
    environment: Mapping[str, Any], product: Mapping[str, Any]
) -> dict:
    """Turn moneyness-declared levels into the absolute spec make_snowball wants.

    ``contract_multiplier`` is COMPUTED, not declared: both surfaces carry the
    same economic notional expressed at their own index level, which is what
    keeps the study-level economic scale honest across two index levels.
    """
    unknown = set(product) - set(_PRODUCT_KEYS)
    if unknown:
        raise ValidationError(
            f"Unknown equity.snowball.localvol product keys: {sorted(unknown)}; "
            f"expected a subset of {_PRODUCT_KEYS}"
        )
    for key in ("strike_moneyness", "ko_barrier_moneyness", "ki_barrier_moneyness",
                "months", "maturity"):
        if key not in product:
            raise ValidationError(
                f"equity.snowball.localvol product is missing {key!r}"
            )
    s0 = float(load_surface(str(environment["surface"]),
                            float(environment["rate"])).artifact.s0)
    spec: dict[str, Any] = {
        "initial_price": s0,
        "strike": float(product["strike_moneyness"]) * s0,
        "ko_barrier": float(product["ko_barrier_moneyness"]) * s0,
        "ki_barrier": float(product["ki_barrier_moneyness"]) * s0,
        "months": int(product["months"]),
        "maturity": float(product["maturity"]),
        "contract_multiplier": REFERENCE_SPOT / s0,
    }
    for key in ("ko_rate", "rebate_rate", "ki_monitoring", "ko_stepdown"):
        if key in product:
            spec[key] = product[key]
    return build_snowball_product_spec(spec)


class _LocalVolArm:
    """Shared spec handling for every arm of this study."""

    def __init__(
        self,
        environment_params: Mapping[str, Any],
        product_params: Mapping[str, Any],
        quantities: Sequence[str],
        params: Mapping[str, Any],
    ) -> None:
        self.environment_params = dict(environment_params)
        self.product_params = dict(product_params)
        self.quantities = tuple(quantities)
        self._params = dict(params)

    def _specs(self, case) -> tuple[dict, dict]:
        environment = dict(self.environment_params)
        environment.update(case.environment_params)
        product = dict(self.product_params)
        product.update(case.product_params)
        build_localvol_market_spec(environment)
        return environment, resolve_product_spec(environment, product)

    def _surface(self, environment: Mapping[str, Any]) -> _Surface:
        return load_surface(str(environment["surface"]), float(environment["rate"]))
```

- [ ] **Step 5: Register the module**

Modify `quantark/modelvalidation/builders/__init__.py` — add the import next to its siblings and the `__all__` entry:

```python
from quantark.modelvalidation.builders import equity_snowball_localvol  # noqa: F401
```

```python
    "equity_snowball_localvol",
```

- [ ] **Step 6: Run the test to verify it passes**

Run:
```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 \
  test/modelvalidation/test_localvol_study.py -v
```
Expected: PASS, 8 tests.

If `test_artifact_identity_is_pinned` fails on the sha, the copy is not byte-identical — re-copy rather than editing the expected value.

- [ ] **Step 7: Commit**

```bash
git add example/modelvalidation/data quantark/modelvalidation/builders/equity_snowball_localvol.py quantark/modelvalidation/builders/__init__.py test/modelvalidation/test_localvol_study.py
git commit -m "feat(modelvalidation): local-vol snowball environment builder

Commits the two CSI1000 Dupire surfaces the study certifies on. They cannot
be read from example/mo_volmodels/data/history: that path is excluded through
.git/info/exclude, a per-clone file that is never pushed, so CI anchors would
fail everywhere but the machine that built them.

Levels are declared as moneyness and resolved against each artifact's own s0,
which lets contract_multiplier be computed as REFERENCE_SPOT / s0 rather than
transcribed. Uncorrected, the calm surface's errors would be overstated 1.243x.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: The PDE candidate

**Files:**
- Modify: `quantark/modelvalidation/builders/equity_snowball_localvol.py`
- Test: `test/modelvalidation/test_localvol_study.py`

**Interfaces:**
- Consumes: `_LocalVolArm`, `make_localvol_environment`, `resolve_product_spec`, `make_snowball` (Task 1).
- Produces: `LocalVolPDECandidate` with `name() -> "equity.snowball.localvol_pde"`, `params() -> Mapping`, `evaluate(case) -> CandidateResult`; registered candidate builder `equity.snowball.localvol_pde`.

- [ ] **Step 1: Write the failing test**

Append to `test/modelvalidation/test_localvol_study.py`:

```python
import math

from quantark.modelvalidation.builders.equity_snowball_localvol import (
    build_localvol_pde_candidate,
)
from quantark.modelvalidation.study import CaseSpec

ENV = {"surface": CRASH, "rate": 0.02}
PRODUCT = {
    "strike_moneyness": 1.0,
    "ko_barrier_moneyness": 1.03,
    "ki_barrier_moneyness": 0.85,
    "ko_rate": 0.15,
    "rebate_rate": 0.15,
    "months": 12,
    "maturity": 1.0,
}


def _candidate(**params):
    return build_localvol_pde_candidate(
        environment_params=ENV,
        product_params=PRODUCT,
        quantities=("pv", "delta", "gamma"),
        params=params or {"accuracy": "standard"},
    )


def test_pde_candidate_is_named_for_its_engine():
    assert _candidate().name() == "equity.snowball.localvol_pde"


def test_pde_candidate_records_its_resolved_grid():
    """A profile name is an indirection; the resolved grid is the evidence."""
    params = _candidate().params()
    assert params["engine"] == "LocalVolSnowballPDESolver"
    assert params["grid"]["points"] > 0
    assert params["grid"]["steps_per_day"] > 0


def test_pde_candidate_produces_finite_greeks_with_a_ladder():
    result = _candidate().evaluate(CaseSpec(name="ordinary"))
    assert set(result.values) == {"pv", "delta", "gamma"}
    assert all(math.isfinite(v) for v in result.values.values())
    assert [rung.level for rung in result.ladders] == ["target", "medium"]


def test_pde_candidate_delta_is_stable_across_its_own_ladder():
    """FINDING-2026-08-26: the PDE moved 0.0079 contracts across its whole
    accuracy ladder, 63x tighter than the bound it was failing. If that is no
    longer true, the engine changed and the certification premise with it."""
    result = _candidate().evaluate(CaseSpec(name="ordinary"))
    target, medium = result.ladders
    assert abs(target.values["delta"] - medium.values["delta"]) < 0.05
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 \
  test/modelvalidation/test_localvol_study.py -v -k pde
```
Expected: FAIL — `ImportError: cannot import name 'build_localvol_pde_candidate'`.

- [ ] **Step 3: Implement the candidate**

Append to `quantark/modelvalidation/builders/equity_snowball_localvol.py`:

```python
class LocalVolPDECandidate(_LocalVolArm):
    """The 1-D two-surface local-vol PDE solver at a declared accuracy profile."""

    def name(self) -> str:
        return "equity.snowball.localvol_pde"

    def params(self) -> Mapping[str, Any]:
        """Declared settings plus the grid the accuracy profile resolves to.

        Recording the resolved grid rather than the profile name is what makes
        the certificate self-describing, and what makes the identity hash move
        if a future release redefines the profile.
        """
        accuracy = str(self._params.get("accuracy", "standard"))
        return {
            **self._params,
            "engine": "LocalVolSnowballPDESolver",
            "grid": engine_config(resolve_config(accuracy, None)),
        }

    def _greeks(self, case, accuracy: str) -> dict:
        environment, product_spec = self._specs(case)
        surface = self._surface(environment)
        solver = LocalVolSnowballPDESolver(
            params=PDEParams(accuracy=accuracy),
            local_vol_surface=surface.local_vol,
        )
        result = solver.calculate_greeks(
            make_snowball(product_spec), make_localvol_environment(environment)
        )
        return {
            "pv": result["price"],
            "delta": result["delta"],
            "gamma": result["gamma"],
        }

    def evaluate(self, case) -> CandidateResult:
        accuracy = str(self._params.get("accuracy", "standard"))
        values = self._greeks(case, accuracy)
        rungs = [LadderRung(axis="accuracy", level="target", values=values)]
        coarser = _COARSER_ACCURACY[accuracy]
        if coarser != accuracy:
            rungs.append(
                LadderRung(
                    axis="accuracy", level="medium",
                    values=self._greeks(case, coarser),
                )
            )
        return CandidateResult(values=values, ladders=tuple(rungs))


@register_builder("equity.snowball.localvol_pde", kind="candidate")
def build_localvol_pde_candidate(
    environment_params: Mapping[str, Any],
    product_params: Mapping[str, Any],
    quantities: Sequence[str],
    params: Mapping[str, Any],
) -> LocalVolPDECandidate:
    return LocalVolPDECandidate(
        environment_params=environment_params,
        product_params=product_params,
        quantities=quantities,
        params=params,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run:
```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 \
  test/modelvalidation/test_localvol_study.py -v
```
Expected: PASS, 12 tests. Runtime ~10s (each `evaluate` runs two PDE solves at ~1s each).

- [ ] **Step 5: Commit**

```bash
git add quantark/modelvalidation/builders/equity_snowball_localvol.py test/modelvalidation/test_localvol_study.py
git commit -m "feat(modelvalidation): local-vol snowball PDE candidate

Records the resolved grid rather than the accuracy profile name, so the
identity hash moves if a release redefines the profile.

The ladder-stability test pins FINDING-2026-08-26's measurement that the PDE
moves under 0.05 contracts across its own accuracy ladder -- the observation
the whole certification premise rests on.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: The local-vol Monte-Carlo reference

**Files:**
- Modify: `quantark/modelvalidation/builders/equity_snowball_localvol.py`
- Test: `test/modelvalidation/test_localvol_study.py`

**Interfaces:**
- Consumes: `_LocalVolArm`, `make_localvol_environment`, `_central_difference_greeks`, `SamplingPolicy`, `BatchResult`.
- Produces: `LocalVolMCReference` with `config() -> Mapping`, `identity(case) -> Mapping`, `run_batch(case, batch_index) -> BatchResult`; registered reference builder `equity.snowball.localvol_mc`. Accepted `params` keys: `substeps_per_interval` (default 8), `lv_time_sampling` (default `"integrated"`), `estimator` (default `"plain"`).

- [ ] **Step 1: Write the failing test**

Append to `test/modelvalidation/test_localvol_study.py`:

```python
from quantark.modelvalidation.builders.equity_snowball_localvol import (
    build_localvol_mc_reference,
)
from quantark.modelvalidation.study import SamplingPolicy

TINY = SamplingPolicy(
    paths_per_batch=1024, min_batches=2, max_batches=3, seed=20260828, bump=0.01
)


def _reference(**params):
    return build_localvol_mc_reference(
        environment_params=ENV,
        product_params=PRODUCT,
        sampling=TINY,
        quantities=("pv", "delta", "gamma"),
        params=params,
    )


def test_reference_declares_the_discretization_it_runs():
    """The FINDING's root cause A was a reference whose declared substeps were
    not the ones it executed. The config must report what run_batch uses."""
    config = _reference().config()
    assert config["substeps_per_interval"] == 8
    assert config["lv_time_sampling"] == "integrated"
    assert config["estimator"] == "plain"
    assert config["engine"] == "LocalVolSnowballMCEngine"


def test_reference_honours_the_seed_contract():
    """_validate_batch requires seed == policy.seed + index; that contract is
    what makes each batch an independent Sobol scramble."""
    batch = _reference().run_batch(CaseSpec(name="ordinary"), 0)
    assert batch.index == 0
    assert batch.seed == TINY.seed


def test_reference_produces_all_three_finite_quantities():
    batch = _reference().run_batch(CaseSpec(name="ordinary"), 1)
    assert set(batch.values) == {"pv", "delta", "gamma"}
    assert all(math.isfinite(v) for v in batch.values.values())


def test_reference_batches_differ():
    """Identical batches would collapse the standard error toward zero and fire
    SE_BUDGET_MET on noise -- a false ADMITTED."""
    ref = _reference()
    a = ref.run_batch(CaseSpec(name="ordinary"), 0)
    b = ref.run_batch(CaseSpec(name="ordinary"), 1)
    assert a.values["pv"] != b.values["pv"]


def test_reference_identity_pins_the_surface_bytes():
    identity = _reference().identity(CaseSpec(name="ordinary"))
    assert identity["surface_sha256"].startswith(CRASH_SHA16)


def test_unsupported_reference_knob_is_refused():
    """A knob that is recorded but never applied would move the identity hash
    while moving no number."""
    with pytest.raises(ValidationError, match="localvol_mc"):
        _reference(martingale_correction=True)
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 \
  test/modelvalidation/test_localvol_study.py -v -k reference
```
Expected: FAIL — `ImportError: cannot import name 'build_localvol_mc_reference'`.

- [ ] **Step 3: Implement the reference**

Append to `quantark/modelvalidation/builders/equity_snowball_localvol.py`:

```python
#: Knobs the local-vol benchmark can actually honour. Anything else would be
#: banked as a benchmark setting and folded into the identity hash while moving
#: no number, so it is refused rather than ignored.
_REFERENCE_KEYS = frozenset(
    {"substeps_per_interval", "lv_time_sampling", "estimator"}
)

#: The discretization FINDING-2026-08-26 section 5 demonstrated the estimate
#: stops moving at: substeps 8 and 16 differ by 0.04 sigma. PV converged at 2,
#: but delta had not converged at 4 -- reading "PV is flat under refinement" as
#: "the reference is converged" is the inference that produced the defect.
_DEFAULT_SUBSTEPS = 8

#: Exact per-step time-averaged variance instead of the left-endpoint sigma
#: freeze. Exact on time-only surfaces; removes a measured -1.26c daily-grid
#: bias at zero per-step cost (docs/lv-mc-scheme-demos/RESULTS.md).
_DEFAULT_TIME_SAMPLING = "integrated"


class LocalVolMCReference(_LocalVolArm):
    """Paired local-vol MC benchmark: one randomization per batch, shared bumps.

    ONE discretization serves pv, delta and gamma. Running PV at one substep
    level and Greeks at another estimates P(h) and P(h/2) -- different numbers at
    finite h -- so the certified delta would not be the derivative of the
    certified price.
    """

    def __init__(self, sampling: SamplingPolicy, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sampling = sampling
        unsupported = set(self._params) - _REFERENCE_KEYS
        if unsupported:
            raise ValidationError(
                f"equity.snowball.localvol_mc does not support "
                f"{sorted(unsupported)}. Supported knobs: {sorted(_REFERENCE_KEYS)}."
            )
        self.substeps = int(
            self._params.get("substeps_per_interval", _DEFAULT_SUBSTEPS)
        )
        self.time_sampling = str(
            self._params.get("lv_time_sampling", _DEFAULT_TIME_SAMPLING)
        )
        self.estimator = str(self._params.get("estimator", "plain"))
        # one_step_survival rejects RANDOMIZED_QUASI (its control-variate
        # machinery assumes the plain estimator), so an OSS arm runs QUASI.
        # Both scramble: _qmc_normals always builds Sobol(scramble=True,
        # seed=base_seed + batch_id), so batches stay independent and the
        # batch-to-batch standard error remains valid.
        self.method = (
            MonteCarloMethod.QUASI
            if self.estimator == "one_step_survival"
            else MonteCarloMethod.RANDOMIZED_QUASI
        )

    def config(self) -> Mapping[str, Any]:
        """The benchmark's own settings -- it is half of every comparison."""
        return {
            "engine": "LocalVolSnowballMCEngine",
            "method": self.method.value,
            "substeps_per_interval": self.substeps,
            "lv_time_sampling": self.time_sampling,
            "estimator": self.estimator,
            "paths_per_batch": self.sampling.paths_per_batch,
            "greeks": "paired central difference (common random numbers)",
        }

    def identity(self, case) -> Mapping[str, Any]:
        environment, product = self._specs(case)
        return {
            "builder": "equity.snowball.localvol_mc",
            "case": case.name,
            "environment": environment,
            "product": product,
            "surface_sha256": self._surface(environment).artifact.sha256,
            "quantities": list(self.quantities),
            "params": dict(self._params),
            "config": dict(self.config()),
            "sampling": {
                "paths_per_batch": self.sampling.paths_per_batch,
                "min_batches": self.sampling.min_batches,
                "max_batches": self.sampling.max_batches,
                "seed": self.sampling.seed,
                "bump": self.sampling.bump,
            },
        }

    def run_batch(self, case, batch_index: int) -> BatchResult:
        environment, product_spec = self._specs(case)
        surface = self._surface(environment)
        product = make_snowball(product_spec)
        seed = self.sampling.seed + batch_index

        def price_at(spot: float) -> float:
            # A fresh engine per pricing call: engine instances are not safe to
            # reuse across calls, and the shared seed is what pairs the three
            # bump arms onto one set of paths. The local-vol surface is the SAME
            # object across bumps -- it is built at the artifact spot, never
            # rebuilt at a bumped one.
            engine = LocalVolSnowballMCEngine(
                local_vol_surface=surface.local_vol,
                params=MCParams(
                    seed=seed,
                    num_paths=self.sampling.paths_per_batch,
                    use_qmc=True,
                    rqmc_min_batches=1,
                    rqmc_max_batches=1,
                    rqmc_paths_mode="per_batch",
                ),
                method=self.method,
                substeps_per_interval=self.substeps,
                lv_time_sampling=self.time_sampling,
                estimator=self.estimator,
            )
            return engine.price(
                product, make_localvol_environment(environment, spot)
            )

        base_spot = float(environment.get("spot_moneyness", 1.0)) * float(
            surface.artifact.s0
        )
        values = _central_difference_greeks(
            price_at, base_spot, self.sampling.bump
        )
        return BatchResult(index=batch_index, seed=seed, values=values)


@register_builder("equity.snowball.localvol_mc", kind="reference")
def build_localvol_mc_reference(
    environment_params: Mapping[str, Any],
    product_params: Mapping[str, Any],
    sampling: SamplingPolicy,
    quantities: Sequence[str],
    params: Mapping[str, Any],
) -> LocalVolMCReference:
    return LocalVolMCReference(
        sampling=sampling,
        environment_params=environment_params,
        product_params=product_params,
        quantities=quantities,
        params=params,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run:
```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 \
  test/modelvalidation/test_localvol_study.py -v
```
Expected: PASS, 18 tests. Runtime ~60s (each `run_batch` is three 1024-path MC prices at substeps 8).

- [ ] **Step 5: Commit**

```bash
git add quantark/modelvalidation/builders/equity_snowball_localvol.py test/modelvalidation/test_localvol_study.py
git commit -m "feat(modelvalidation): local-vol snowball MC reference

One discretization (substeps=8, lv_time_sampling=integrated) serves pv, delta
and gamma, so the certified delta is the derivative of the certified price.
substeps=8 is the level FINDING-2026-08-26 section 5 demonstrated the estimate
stops moving at; PV was flat from 2 while delta was still moving at 4.

Refuses knobs it cannot honour rather than banking them into the identity hash,
and records the substeps it actually executes -- the FINDING's root cause A was
a reference whose declared resolution was not the one it ran.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: The study YAML and an end-to-end soundness test

**Files:**
- Create: `example/modelvalidation/snowball_localvol_1d.yaml`
- Test: `test/modelvalidation/test_localvol_study.py`

**Interfaces:**
- Consumes: all three builders from Tasks 1–3.
- Produces: a loadable `CertificationStudy` named `snowball-localvol-1d` with 16 cases and one candidate.

- [ ] **Step 1: Write the study file**

Create `example/modelvalidation/snowball_localvol_1d.yaml`:

```yaml
# Snowball under a real calibrated Dupire local-volatility surface.
#
# The 1-D local-vol counterpart of snowball_flat_bsm.yaml. One deterministic
# engine -- LocalVolSnowballPDESolver -- is certified against a local-vol Monte
# Carlo reference on TWO real CSI1000 surfaces:
#
#   2024-02-08  the crash bottom. The surface that put localvol on the MC route
#               in Gate G2, and the worst cell in the sample (-1.2726 contracts
#               against a converged reference, -2.88 sigma on PV).
#   2023-11-15  the cohort's flattest surface by Dupire local-vol slope
#               (dsigma/dlnS -0.020, dsigma/dt -0.017) at roughly half the vol
#               level. The contrast tests FINDING-2026-08-26's central claim
#               that the discretization bias scales with surface steepness.
#
# Both artifacts are COMMITTED under example/modelvalidation/data/. They are not
# read from example/mo_volmodels/data/history, which is excluded per-clone and
# would not exist in CI -- and assert_anchors re-runs the deterministic arm on
# every commit.
#
# Levels are declared as MONEYNESS and resolved against each artifact's own s0,
# so one set of case shapes serves two different index levels and
# contract_multiplier is computed rather than transcribed.
#
# Scope: this certifies the surface machinery -- the Dupire read on the PDE
# (S,t) mesh and the barrier-local sigma_loc(B) continuous-KI treatment. It does
# NOT cover the flat-BSM product variants (parachute, airbag, protection_*,
# reverse, call_rebate, disable_ko_after_ki, ...), which exercise payoff code
# inherited from SnowballPDESolver and already certified under flat BSM. A
# certificate covers only the configurations its YAML names.
#
# Offline study (hours, not minutes). Run it with:
#   python -m quantark.modelvalidation run example/modelvalidation/snowball_localvol_1d.yaml
# Add --quick for a wiring check that is explicitly not bankable evidence.

study: snowball-localvol-1d
schema: 1

quantities: [pv, delta, gamma]

bounds:
  cell: 0.5
  mean_signed_bias: 0.1

sampling:
  paths_per_batch: 65536
  min_batches: 4
  max_batches: 32
  seed: 20260828
  bump: 0.01

# One CSI1000 index-futures contract at multiplier 200 and spot 4993.105 is
# exactly the study notional, so delta_quantum is 1.0 and raw delta reads
# directly as hedge contracts -- the same normalization the flat-BSM study uses
# (200 * 100 / 20000 = 1.0), which makes the two certificates comparable.
economic_scale:
  builder: hedge_contracts
  params:
    hedge_multiplier: 200.0
    hedge_inception_spot: 4993.105
    notional: 998621.0

environment:
  builder: equity.snowball.localvol_market
  params:
    surface: example/modelvalidation/data/iv_surface_20240208.json
    rate: 0.02

product:
  builder: equity.snowball
  params:
    strike_moneyness: 1.0
    ko_barrier_moneyness: 1.03
    ki_barrier_moneyness: 0.85
    ko_rate: 0.15
    rebate_rate: 0.15
    months: 12
    maturity: 1.0

reference:
  builder: equity.snowball.localvol_mc
  params:
    substeps_per_interval: 8
    lv_time_sampling: integrated
    estimator: plain

candidates:
  - builder: equity.snowball.localvol_pde
    params: {accuracy: standard}

cases:
  # ---- 2024-02-08, the crash bottom -------------------------------------
  # At inception between the barriers. T=1.0 sits ~13% beyond the surface's
  # last listed expiry (0.866y), where flat_total_variance extrapolation runs.
  - {name: crash_ordinary}
  # Entirely inside the listed expiries: isolates the engine from the
  # extrapolation policy the case above necessarily includes.
  - {name: crash_inside_listed_grid, product: {months: 9, maturity: 0.75}}
  # Just under the KO barrier: the discontinuity sits inside the bump stencil,
  # which is where a deterministic gamma usually breaks first.
  - {name: crash_near_ko, environment: {spot_moneyness: 1.025}}
  # Just above the KI barrier, where the barrier-local sigma_loc(B) coefficients
  # bind. This is the cell the continuous-KI machinery exists for.
  - {name: crash_near_ki, environment: {spot_moneyness: 0.86}}
  # Discretely monitored KI on the KO dates: the benchmark drops its Brownian
  # bridge crossing correction and the PDE its per-step first-passage transfer,
  # so the continuous cells say nothing about this one.
  - {name: crash_discrete_ki, product: {ki_monitoring: discrete}}
  # European KI: the two-surface dynamic programme collapses to one terminal
  # test and the knock-in probability becomes closed-form.
  - {name: crash_european_ki, product: {ki_monitoring: european}}
  # Step-down KO: twelve distinct levels compete for grid alignment where every
  # case above had one -- now on a skewed surface rather than a flat one.
  - {name: crash_stepdown_ko, product: {ko_stepdown: 0.005}}
  # Three months, in the steepest part of the term structure (ATM 0.284 -> 0.429
  # within three weeks). Time discretization error has nowhere to average out.
  - {name: crash_near_expiry, product: {months: 3, maturity: 0.25}}

  # ---- 2023-11-15, the calm contrast ------------------------------------
  - {name: calm_ordinary,
     environment: {surface: example/modelvalidation/data/iv_surface_20231115.json}}
  - {name: calm_inside_listed_grid,
     environment: {surface: example/modelvalidation/data/iv_surface_20231115.json},
     product: {months: 9, maturity: 0.75}}
  - {name: calm_near_ko,
     environment: {surface: example/modelvalidation/data/iv_surface_20231115.json,
                   spot_moneyness: 1.025}}
  - {name: calm_near_ki,
     environment: {surface: example/modelvalidation/data/iv_surface_20231115.json,
                   spot_moneyness: 0.86}}
  - {name: calm_discrete_ki,
     environment: {surface: example/modelvalidation/data/iv_surface_20231115.json},
     product: {ki_monitoring: discrete}}
  - {name: calm_european_ki,
     environment: {surface: example/modelvalidation/data/iv_surface_20231115.json},
     product: {ki_monitoring: european}}
  - {name: calm_stepdown_ko,
     environment: {surface: example/modelvalidation/data/iv_surface_20231115.json},
     product: {ko_stepdown: 0.005}}
  - {name: calm_near_expiry,
     environment: {surface: example/modelvalidation/data/iv_surface_20231115.json},
     product: {months: 3, maturity: 0.25}}
```

**Note:** the `product.builder` name is `equity.snowball` — the flat-BSM product builder is registered under that name and the loader only *resolves* it, so the local-vol arm's `resolve_product_spec` is what actually validates the moneyness keys. If the loader ever calls product builders, this must change to a dedicated registration.

- [ ] **Step 2: Write the failing test**

Append to `test/modelvalidation/test_localvol_study.py`:

```python
import dataclasses

from quantark.modelvalidation.pipeline import certify
from quantark.modelvalidation.yaml_loader import load_study

STUDY_PATH = REPO_ROOT / "example" / "modelvalidation" / "snowball_localvol_1d.yaml"

EXPECTED_CASES = [
    "crash_ordinary", "crash_inside_listed_grid", "crash_near_ko",
    "crash_near_ki", "crash_discrete_ki", "crash_european_ki",
    "crash_stepdown_ko", "crash_near_expiry",
    "calm_ordinary", "calm_inside_listed_grid", "calm_near_ko",
    "calm_near_ki", "calm_discrete_ki", "calm_european_ki",
    "calm_stepdown_ko", "calm_near_expiry",
]


@pytest.fixture(scope="module")
def study():
    return load_study(STUDY_PATH)


def test_study_loads_with_all_sixteen_cases(study):
    assert study.name == "snowball-localvol-1d"
    assert [case.name for case in study.cases] == EXPECTED_CASES
    assert tuple(c.name() for c in study.candidates) == ("equity.snowball.localvol_pde",)


def test_bounds_are_the_desk_convention(study):
    """Never widened to make a result pass."""
    assert study.bounds.cell == 0.5
    assert study.bounds.mean_signed_bias == 0.1


def test_delta_quantum_is_exactly_one(study):
    """Matches the flat-BSM normalization, so raw delta reads as contracts."""
    assert study.scale.delta_quantum == pytest.approx(1.0, rel=1e-9)


def test_contract_multiplier_is_computed_per_surface():
    """The two-surface scale correction: uncorrected, calm-surface errors would
    be overstated by 6207.268 / 4993.105 = 1.243 and risk a false REJECTED."""
    from quantark.modelvalidation.builders.equity_snowball_localvol import (
        resolve_product_spec,
    )
    crash = resolve_product_spec({"surface": CRASH, "rate": 0.02}, PRODUCT)
    calm = resolve_product_spec({"surface": CALM, "rate": 0.02}, PRODUCT)
    assert crash["contract_multiplier"] == pytest.approx(1.0, rel=1e-9)
    assert calm["contract_multiplier"] == pytest.approx(0.804397, rel=1e-5)


def test_moneyness_resolves_against_each_surfaces_own_spot():
    from quantark.modelvalidation.builders.equity_snowball_localvol import (
        resolve_product_spec,
    )
    calm = resolve_product_spec({"surface": CALM, "rate": 0.02}, PRODUCT)
    assert calm["initial_price"] == pytest.approx(CALM_S0, abs=1e-3)
    assert calm["ko_barrier"] == pytest.approx(1.03 * CALM_S0, abs=1e-3)
    assert calm["ki_barrier"] == pytest.approx(0.85 * CALM_S0, abs=1e-3)


@pytest.mark.slow
def test_study_runs_end_to_end_and_gates_every_cell(study, tmp_path):
    """Soundness, not outcome: at this budget INCONCLUSIVE is correct."""
    small = dataclasses.replace(
        study,
        cases=(study.cases[0], study.cases[8]),   # one crash cell, one calm
        sampling=dataclasses.replace(
            study.sampling, paths_per_batch=1024, min_batches=2, max_batches=2
        ),
    )
    result = certify(small, out_dir=tmp_path)
    assert (tmp_path / "snowball-localvol-1d" / "certificate.json").is_file()
    assert (tmp_path / "snowball-localvol-1d" / "report.md").is_file()
    assert (tmp_path / "snowball-localvol-1d" / "report.html").is_file()
    assert result is not None
```

- [ ] **Step 3: Run to verify it fails**

Run:
```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 \
  test/modelvalidation/test_localvol_study.py -v -k "study or contract or moneyness"
```
Expected: FAIL — the study file does not parse, or `certify` is not reached.

- [ ] **Step 4: Fix whatever the loader rejects, then run again**

Run the same command. Expected: PASS.

Common failures and their real fixes:
- `Unknown product builder 'equity.snowball'` → the flat-BSM module was not imported; check `builders/__init__.py`.
- `equity.snowball product is missing 'initial_price'` → `resolve_product_spec` is not being reached; the arm's `_specs` must call it before `make_snowball`.
- A `ko_stepdown` validation error → the step-down schedule walks below KI. Recompute: `1.03 - 0.005*11 = 0.975 > 0.85`, so this should not fire; if it does, `months` is larger than 12 somewhere.

- [ ] **Step 5: Run the slow end-to-end case**

Run:
```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 \
  test/modelvalidation/test_localvol_study.py -v -k end_to_end --no-header
```
Expected: PASS in ~2–4 minutes. `INCONCLUSIVE` in the report is correct at this budget.

- [ ] **Step 6: Commit**

```bash
git add example/modelvalidation/snowball_localvol_1d.yaml test/modelvalidation/test_localvol_study.py
git commit -m "feat(modelvalidation): snowball-localvol-1d study, 16 cells on two real surfaces

Eight case shapes on the 2024-02-08 crash bottom and the calm 2023-11-15,
chosen as the cohort's steepest and flattest by Dupire local-vol slope so the
pair tests the bias-scales-with-steepness mechanism directly.

Scope is stated in the file: this certifies the surface machinery, not the
flat-BSM product variants, which are inherited from SnowballPDESolver and
already certified. A certificate covers only the configurations its YAML names.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Pilot — economic-scale and flat-surface controls

Spec §7 items 3 and 5. These two are grouped because both are cheap deterministic
checks that must pass before any sampling is spent.

**Files:**
- Create: `docs/modelvalidation/pilot-localvol-1d/probe_scale_and_flat.py`
- Create: `docs/modelvalidation/pilot-localvol-1d/RESULTS.md`

**Interfaces:**
- Consumes: `load_surface`, `resolve_product_spec`, `make_localvol_environment`, `LocalVolPDECandidate` (Tasks 1–2).
- Produces: a `RESULTS.md` section recording both control outcomes. No code other tasks import.

- [ ] **Step 1: Write the probe**

Create `docs/modelvalidation/pilot-localvol-1d/probe_scale_and_flat.py`:

```python
"""Pilot controls 3 and 5 for the snowball-localvol-1d certification.

(5) ECONOMIC SCALE. A known raw delta must convert to the intended contract
count on BOTH surfaces. Uncorrected, the calm surface's errors would be
overstated by 6207.268 / 4993.105 = 1.243 -- which inflates a measured error and
so risks a false REJECTED, not a merely conservative pass.

(3) FLAT-SURFACE CONTROL. Flatten a surface to a constant vol and the local-vol
PDE must collapse onto the flat-BSM PDE. This separates "the input is wrong"
from "the formula is wrong", and it is the control that settled the original
diagnosis in FINDING-2026-08-26.

Run:
  PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
    docs/modelvalidation/pilot-localvol-1d/probe_scale_and_flat.py
"""

import numpy as np

from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    LocalVolSnowballPDESolver,
)
from quantark.asset.equity.param import PDEParams
from quantark.modelvalidation.builders.equity_snowball import make_snowball
from quantark.modelvalidation.builders.equity_snowball_localvol import (
    REFERENCE_SPOT,
    load_surface,
    make_localvol_environment,
    resolve_product_spec,
)
from quantark.modelvalidation.study import HedgeContractScale
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface
from quantark.param import GridVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.volmodels.localvol import build_dupire_local_vol

CRASH = "example/modelvalidation/data/iv_surface_20240208.json"
CALM = "example/modelvalidation/data/iv_surface_20231115.json"
RATE = 0.02

PRODUCT = {
    "strike_moneyness": 1.0, "ko_barrier_moneyness": 1.03,
    "ki_barrier_moneyness": 0.85, "ko_rate": 0.15, "rebate_rate": 0.15,
    "months": 12, "maturity": 1.0,
}

SCALE = HedgeContractScale(
    hedge_multiplier=200.0, hedge_inception_spot=REFERENCE_SPOT, notional=998621.0
)

print("=" * 74)
print("CONTROL 5 -- economic scale on both surfaces")
print("=" * 74)
print(f"delta_quantum = {SCALE.delta_quantum:.9f}   (must be 1.0)")

for tag, path in (("crash 2024-02-08", CRASH), ("calm  2023-11-15", CALM)):
    surface = load_surface(path, RATE)
    s0 = float(surface.artifact.s0)
    spec = resolve_product_spec({"surface": path, "rate": RATE}, PRODUCT)
    cm = spec["contract_multiplier"]

    # A desk holding `notional` at this index level holds notional/s0 units,
    # and one futures contract covers 200*s0 of exposure. A raw per-unit delta
    # of 1.0 is therefore this many contracts:
    true_contracts = (998621.0 / s0) / (200.0 * s0) * s0
    reported = SCALE.to_economic("delta", 1.0 * cm)
    print(f"{tag}: s0={s0:9.3f}  contract_multiplier={cm:.6f}  "
          f"reported={reported:.6f}  true={true_contracts:.6f}  "
          f"ratio={reported / true_contracts:.6f}")

print()
print("=" * 74)
print("CONTROL 3 -- flat-surface collapse (LV PDE must equal flat-BSM PDE)")
print("=" * 74)

surface = load_surface(CRASH, RATE)
s0 = float(surface.artifact.s0)
flat_vol = float(max(surface.artifact.atm_pillars, key=lambda p: p["T"])["atm_vol"])

flat_grid = GridVolSurface(
    list(surface.artifact.strikes),
    list(surface.artifact.maturities),
    np.full((len(surface.artifact.maturities), len(surface.artifact.strikes)), flat_vol),
)
flat_env = PricingEnvironment(
    rate_curve=FlatRateCurve(RATE),
    valuation_date=make_localvol_environment({"surface": CRASH, "rate": RATE}).valuation_date,
    spot_quote=SpotQuote(s0),
    vol_surface=flat_grid,
    div_yield=ContinuousDividendYield(0.0),
)
lv_flat = build_dupire_local_vol(
    flat_grid, spot=s0, rate_curve=flat_env.rate_curve,
    div_yield=lambda t: 0.0,
)
spec = resolve_product_spec({"surface": CRASH, "rate": RATE}, PRODUCT)
product = make_snowball(spec)

lv_solver = LocalVolSnowballPDESolver(
    params=PDEParams(accuracy="standard"), local_vol_surface=lv_flat
)
lv_greeks = lv_solver.calculate_greeks(product, flat_env)

bsm_env = PricingEnvironment(
    rate_curve=FlatRateCurve(RATE),
    valuation_date=flat_env.valuation_date,
    spot_quote=SpotQuote(s0),
    vol_surface=FlatVolSurface(flat_vol),
    div_yield=ContinuousDividendYield(0.0),
)
bsm_greeks = SnowballPDESolver(
    params=PDEParams(accuracy="standard")
).calculate_greeks(product, bsm_env)

print(f"flat vol = {flat_vol:.6f}")
for key in ("price", "delta", "gamma"):
    lv_v, bsm_v = lv_greeks[key], bsm_greeks[key]
    denom = max(abs(bsm_v), 1e-12)
    print(f"{key:6s}  LV={lv_v: .8f}  BSM={bsm_v: .8f}  "
          f"rel={abs(lv_v - bsm_v) / denom:.3e}")
print()
print("PASS CRITERIA")
print("  control 5: ratio == 1.000000 on BOTH surfaces")
print("  control 3: delta rel < 1e-3 (grid alignment differs; the number must "
      "agree, not the discretization)")
```

- [ ] **Step 2: Run the probe**

Run:
```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  docs/modelvalidation/pilot-localvol-1d/probe_scale_and_flat.py
```
Expected: `ratio = 1.000000` on both surfaces, and `delta rel < 1e-3` on the flat control.

**If control 5's ratio is not 1.0 on the calm surface:** the `contract_multiplier`
derivation in `resolve_product_spec` is wrong. Do NOT adjust the expected value —
fix the derivation, because the whole calm arm is measured through it.

**If control 3's relative delta exceeds 1e-3:** stop. A local-vol PDE that does
not collapse onto flat BSM on a flat surface has a defect in the surface read
itself, which no amount of reference tuning can absolve. Report before continuing.

- [ ] **Step 3: Record the results**

Create `docs/modelvalidation/pilot-localvol-1d/RESULTS.md` with a heading, the
date, the exact command, and the probe's printed table pasted verbatim under a
`## Controls 3 and 5` section, plus a one-line verdict for each.

- [ ] **Step 4: Commit**

```bash
git add docs/modelvalidation/pilot-localvol-1d
git commit -m "test(modelvalidation): pilot controls for scale and flat-surface collapse

Control 5 verifies the two-surface contract_multiplier correction numerically
rather than trusting the algebra: an uncorrected calm surface would overstate
every error by 1.243x and risk a false REJECTED.

Control 3 is the flat-surface collapse that settled FINDING-2026-08-26 -- it
separates a wrong input from a wrong formula.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Pilot — reference convergence ladder and estimator choice

Spec §7 items 1 and 2. This task decides the reference configuration the real
run uses, so nothing after it can start until it reports.

**Files:**
- Create: `docs/modelvalidation/pilot-localvol-1d/probe_reference.py`
- Modify: `docs/modelvalidation/pilot-localvol-1d/RESULTS.md`

**Interfaces:**
- Consumes: `LocalVolMCReference` via `build_localvol_mc_reference` (Task 3), `LocalVolPDECandidate` (Task 2).
- Produces: two decisions recorded in `RESULTS.md` — the adopted `substeps_per_interval`, and the adopted `estimator` per quantity.

- [ ] **Step 1: Write the probe**

Create `docs/modelvalidation/pilot-localvol-1d/probe_reference.py`:

```python
"""Pilot controls 1 and 2 for the snowball-localvol-1d certification.

(1) CONVERGENCE, DEMONSTRATED NOT INHERITED. FINDING-2026-08-26 section 5
demonstrated substeps=8 for a THREE-year trade at mo-study scale. This is a
one-year trade at a different notional; it does not inherit that result. Walk
the ladder 4 -> 8 -> 16 and confirm the estimate has stopped moving.

Read the LADDER, not a single level: the FINDING's ladder CROSSES zero rather
than decaying to it, so a one-sided reading at any single level mis-signs the
error. That is exactly how substeps=1 made the PDE look 1.27 contracts wrong.

(2) ESTIMATOR CHOICE, MEASURED NOT DERIVED. Per-quantity standard error for
`plain` versus `one_step_survival`. Whichever meets the 0.25 x cell budget wins.
TIEBREAK: if `plain` meets the budget on every quantity, `plain` is adopted.

Run (long -- expect ~30-60 minutes):
  PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
    docs/modelvalidation/pilot-localvol-1d/probe_reference.py
"""

import math
import statistics
import time

from quantark.modelvalidation.builders.equity_snowball_localvol import (
    build_localvol_mc_reference,
    build_localvol_pde_candidate,
)
from quantark.modelvalidation.study import CaseSpec, HedgeContractScale, SamplingPolicy

CRASH = "example/modelvalidation/data/iv_surface_20240208.json"
ENV = {"surface": CRASH, "rate": 0.02}
PRODUCT = {
    "strike_moneyness": 1.0, "ko_barrier_moneyness": 1.03,
    "ki_barrier_moneyness": 0.85, "ko_rate": 0.15, "rebate_rate": 0.15,
    "months": 12, "maturity": 1.0,
}
QUANTITIES = ("pv", "delta", "gamma")
SCALE = HedgeContractScale(
    hedge_multiplier=200.0, hedge_inception_spot=4993.105, notional=998621.0
)
CELL_BOUND = 0.5
SE_BUDGET = 0.25 * CELL_BOUND      # 0.125 contracts

# The two hardest cells: at inception on the steepest surface, and sitting on
# the KI barrier where the barrier-local coefficients bind.
CELLS = {
    "ordinary": (CaseSpec(name="ordinary"), {}),
    "near_ki": (CaseSpec(name="near_ki", environment_params={"spot_moneyness": 0.86}), {}),
}
BATCHES = 6
PATHS = 65536


def pde_values(case):
    candidate = build_localvol_pde_candidate(
        environment_params=ENV, product_params=PRODUCT,
        quantities=QUANTITIES, params={"accuracy": "standard"},
    )
    return candidate.evaluate(case).values


def sample(case, substeps, estimator, batches=BATCHES, paths=PATHS):
    policy = SamplingPolicy(
        paths_per_batch=paths, min_batches=batches, max_batches=batches,
        seed=20260828, bump=0.01,
    )
    ref = build_localvol_mc_reference(
        environment_params=ENV, product_params=PRODUCT, sampling=policy,
        quantities=QUANTITIES,
        params={"substeps_per_interval": substeps,
                "lv_time_sampling": "integrated", "estimator": estimator},
    )
    started = time.time()
    rows = [ref.run_batch(case, i).values for i in range(batches)]
    elapsed = time.time() - started
    out = {}
    for q in QUANTITIES:
        xs = [r[q] for r in rows]
        out[q] = (
            statistics.fmean(xs),
            statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("inf"),
        )
    return out, elapsed


print("=" * 78)
print("CONTROL 1 -- reference convergence ladder (estimator=plain)")
print("=" * 78)
for label, (case, _) in CELLS.items():
    pde = pde_values(case)
    print(f"\n--- {label}   PDE delta = {pde['delta']:.6f} "
          f"({SCALE.to_economic('delta', pde['delta']):.4f} contracts)")
    print(f"{'substeps':>9} {'gap (contracts)':>18} {'+/- SE':>10} {'sigma':>7} {'secs':>7}")
    for substeps in (4, 8, 16):
        stats, secs = sample(case, substeps, "plain")
        mean, se = stats["delta"]
        gap = SCALE.to_economic("delta", mean - pde["delta"])
        gap_se = SCALE.to_economic("delta", se)
        sigma = abs(gap) / gap_se if gap_se else float("inf")
        print(f"{substeps:9d} {gap:18.4f} {gap_se:10.4f} {sigma:7.2f} {secs:7.0f}")
    print("  READ THE LADDER: adjacent levels agreeing within ~1 sigma means it "
          "has stopped moving. A ladder that CROSSES zero mis-signs any single "
          "level read on its own.")

print()
print("=" * 78)
print(f"CONTROL 2 -- estimator choice (SE budget = {SE_BUDGET} contracts)")
print("=" * 78)
case, _ = CELLS["near_ki"]
for estimator in ("plain", "one_step_survival"):
    try:
        stats, secs = sample(case, 8, estimator)
    except Exception as exc:                       # noqa: BLE001
        print(f"{estimator:20s} UNAVAILABLE: {type(exc).__name__}: {exc}")
        continue
    print(f"\n{estimator}  ({secs:.0f}s for {BATCHES} batches)")
    for q in QUANTITIES:
        mean, se = stats[q]
        se_c = abs(SCALE.to_economic(q, se))
        verdict = "MEETS" if se_c <= SE_BUDGET else "over budget"
        print(f"  {q:6s} SE = {se_c:9.5f} contracts   {verdict}")

print()
print("DECISION RULE: if `plain` meets the budget on every quantity, adopt "
      "`plain` -- one estimator, fewer moving parts. Introduce "
      "`one_step_survival` only for a quantity `plain` cannot resolve.")
```

- [ ] **Step 2: Run the probe**

Run:
```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  docs/modelvalidation/pilot-localvol-1d/probe_reference.py 2>&1 | tee \
  docs/modelvalidation/pilot-localvol-1d/out_reference.txt
```
Expected: ~30–60 minutes. Two ladder tables and one estimator table.

- [ ] **Step 3: Read the result and decide**

Apply these rules, and write the decision into `RESULTS.md` with the numbers:

- **Substeps.** Adopt the lowest level whose gap agrees with the next level up within ~1σ. If 8 and 16 disagree by more than 2σ, the reference has *not* converged: raise the ladder to 24 (the hard ceiling is `substeps ≤ 28`, since `_qmc_normals` requests dimension `732 × substeps` and SciPy's Sobol caps at 21201). If it still has not converged, stop and report — this RQMC construction cannot reach it by refinement alone.
- **Estimator.** If `plain` meets the budget on pv, delta and gamma, adopt `plain` and leave the YAML unchanged. Otherwise adopt `one_step_survival` for the failing quantity only, and open a follow-up task to split the reference (the spec permits a split of the *estimator*, never of the discretization).

- [ ] **Step 4: Update the YAML only if the pilot says so**

If the decision differs from the defaults, edit
`example/modelvalidation/snowball_localvol_1d.yaml`'s `reference.params` to
match, and add a comment naming the pilot measurement that chose it.

- [ ] **Step 5: Commit**

```bash
git add docs/modelvalidation/pilot-localvol-1d example/modelvalidation/snowball_localvol_1d.yaml
git commit -m "test(modelvalidation): demonstrate the local-vol reference's own convergence

Walks the substeps ladder 4/8/16 on the two hardest cells and measures the
per-quantity standard error for both estimators. FINDING-2026-08-26 demonstrated
substeps=8 for a three-year trade; a one-year trade does not inherit it.

The whole defect this certification traces back to was a reference trusted
without its own convergence being shown, so this is recorded before any
evidence is banked.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Reconcile the surface-steepness discrepancy

Spec §10. This is the open item; it gates banking because the calm-surface
contrast is selected on the metric in question.

**Files:**
- Create: `docs/modelvalidation/pilot-localvol-1d/probe_steepness.py`
- Modify: `docs/modelvalidation/pilot-localvol-1d/RESULTS.md`
- Modify (only if reconciled): `docs/modelvalidation/FINDING-2026-08-26-localvol-1d-pde.md`

**Interfaces:** none consumed or produced by other tasks; this produces a written finding.

- [ ] **Step 1: Write the probe**

Create `docs/modelvalidation/pilot-localvol-1d/probe_steepness.py` measuring
`dσ/dlnS` and `dσ/dt` for 2024-02-08 across a grid of definitions, printing a
table with one row per definition:

```python
"""Reconcile FINDING-2026-08-26's reported surface slopes.

The FINDING records dsigma/dlnS = -0.371, dsigma/dt = -0.082 for 2024-02-08.
Direct measurement gives -0.056 / -0.269 on the Dupire surface and
-0.031 / -0.081 on the implied surface. The implied TERM slope reproduces almost
exactly while the skew is off by an order of magnitude, which points to a
definitional mismatch -- which surface, which slice, which moneyness window --
rather than a data problem.

This does not change the case list: 2024-02-08 is unambiguously the worst cell
empirically (-1.2726 contracts, -2.88 sigma). But "bias scales with surface
steepness" is the mechanism the calm-surface contrast is selected on, so the
definition has to be pinned down or explicitly recorded as unreconciled.

Run:
  PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
    docs/modelvalidation/pilot-localvol-1d/probe_steepness.py
"""

import numpy as np

from quantark.modelvalidation.builders.equity_snowball_localvol import load_surface

TARGET_SKEW, TARGET_TERM = -0.371, -0.082
PATH = "example/modelvalidation/data/iv_surface_20240208.json"

surface = load_surface(PATH, 0.02)
art, grid, lv = surface.artifact, surface.grid, surface.local_vol
s0 = float(art.s0)
K = np.array(art.strikes, float)
T = np.array(art.maturities, float)
G = np.array(art.iv_grid, float)

rows = []
for which in ("implied", "dupire"):
    for slice_t in (0.05, 0.25, 0.5, float(art.max_listed_T)):
        for width in (0.05, 0.10, 0.15, 0.25, 0.35):
            x = np.linspace(-width, width, 21)
            if which == "dupire":
                sig = np.array([lv.local_vol(s0 * np.exp(xi), slice_t) for xi in x])
            else:
                j = int(np.argmin(np.abs(T - slice_t)))
                xs = np.log(K / s0)
                m = np.abs(xs) <= width
                if m.sum() < 3:
                    continue
                sig = np.interp(x, xs[m], G[j, m])
            skew = np.polyfit(x, sig, 1)[0]
            rows.append((which, slice_t, width, skew, abs(skew - TARGET_SKEW)))

rows.sort(key=lambda r: r[4])
print(f"{'surface':>9} {'slice_t':>8} {'width':>6} {'dsig/dlnS':>10} {'|err|':>8}")
print("-" * 46)
for which, slice_t, width, skew, err in rows[:15]:
    print(f"{which:>9} {slice_t:8.3f} {width:6.2f} {skew:10.4f} {err:8.4f}")
print(f"\nFINDING target: {TARGET_SKEW}")
print("If no definition lands within ~0.02, record as UNRECONCILED in RESULTS.md")
print("and add a correction note to the FINDING rather than silently adopting "
      "either number.")
```

- [ ] **Step 2: Run it**

Run:
```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python \
  docs/modelvalidation/pilot-localvol-1d/probe_steepness.py
```
Expected: a ranked table. Either one definition reproduces `−0.371`, or none does.

- [ ] **Step 3: Record the outcome**

- **If reconciled:** add a `## Surface steepness` section to `RESULTS.md` naming the exact definition, and append a short clarifying note to the FINDING stating which definition its numbers use. Do not alter the FINDING's conclusions.
- **If not reconciled:** record it in `RESULTS.md` as UNRECONCILED with the closest candidates, and state plainly that the calm/crash contrast is justified by the *empirical* per-cell gaps (`−1.2726` vs `+0.2614`) rather than by a slope metric. Do not silently adopt either number.

- [ ] **Step 4: Commit**

```bash
git add docs/modelvalidation/pilot-localvol-1d docs/modelvalidation/FINDING-2026-08-26-localvol-1d-pde.md
git commit -m "docs(modelvalidation): reconcile the reported surface-steepness slopes

FINDING-2026-08-26 reports dsigma/dlnS = -0.371 for 2024-02-08; direct
measurement gives -0.056 (Dupire) and -0.031 (implied) while the implied TERM
slope reproduces almost exactly. Sweeps the definition space to find which
convention the FINDING used.

Bias-scales-with-steepness is the mechanism the calm-surface contrast is chosen
on, so the metric is pinned down or recorded as unreconciled -- not assumed.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Wiring check and the full certification run

**Files:**
- Create: `output/modelvalidation/snowball-localvol-1d/` (not committed)

**Interfaces:** consumes the study and all builders; produces `certificate.json`, `report.md`, `report.html`.

- [ ] **Step 1: Quick wiring check**

Run:
```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m quantark.modelvalidation \
  run example/modelvalidation/snowball_localvol_1d.yaml --quick --out output/modelvalidation
```
Expected: completes in minutes. `INCONCLUSIVE` is **correct** here — quick mode shrinks sampling so the standard error misses its budget. This proves plumbing, nothing more.

- [ ] **Step 2: Confirm all 16 cells were exercised**

Run:
```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -c "
import json
d = json.load(open('output/modelvalidation/snowball-localvol-1d/certificate.json'))
cells = d['cells'] if 'cells' in d else d
print(json.dumps(list(d.keys()), indent=1)[:400])
"
```
Then read `output/modelvalidation/snowball-localvol-1d/report.md` and confirm all 16 case names appear and none errored. An **errored** cell is a bug to fix; an **unresolved** cell is expected in quick mode.

- [ ] **Step 3: The real run**

Run (long — 2–14 hours; use a background-safe invocation and do not interrupt):
```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m quantark.modelvalidation \
  run example/modelvalidation/snowball_localvol_1d.yaml --out output/modelvalidation
```

If interrupted, resume with `--resume`, which reuses every checkpoint whose configuration still matches.

- [ ] **Step 4: Read the decision**

Open `output/modelvalidation/snowball-localvol-1d/report.html` in a browser and read the **budget-consumed** column first: a study full of passes at 90%+ is one small change away from failing.

- `ADMITTED` → proceed to Task 9.
- `INCONCLUSIVE` → more sampling (raise `max_batches`) or a fixed engine. **Never** a loosened bound.
- `REJECTED` → stop and report. Do not bank. A confident disagreement against a converged reference is a real finding and needs its own investigation, exactly as the original one did.

- [ ] **Step 5: Commit nothing yet**

The run writes only to `output/`, which is not banked evidence. Banking is Task 9.

---

### Task 9: Bank the evidence and document it

**Files:**
- Create: `docs/modelvalidation/certificates/snowball-localvol-1d/2026-08-28/{certificate.json,report.md,report.html,anchors.json,README.md}`

**Interfaces:** produces the banked certificate the CI anchor guard picks up automatically.

- [ ] **Step 1: Copy the evidence (never `checkpoints/`)**

```bash
mkdir -p docs/modelvalidation/certificates/snowball-localvol-1d/2026-08-28
cp output/modelvalidation/snowball-localvol-1d/certificate.json \
   output/modelvalidation/snowball-localvol-1d/report.md \
   output/modelvalidation/snowball-localvol-1d/report.html \
   docs/modelvalidation/certificates/snowball-localvol-1d/2026-08-28/
```

- [ ] **Step 2: Extract the anchors**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m quantark.modelvalidation \
  anchors docs/modelvalidation/certificates/snowball-localvol-1d/2026-08-28/certificate.json
```

- [ ] **Step 3: Verify CI picks it up automatically**

Run:
```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 \
  test/modelvalidation/test_banked_certificates.py -v
```
Expected: a new parametrized case `snowball-localvol-1d/2026-08-28` appears and PASSES. `test_banked_certificates.py` globs `*/*/anchors.json`, so no test edit is needed — if the new case does **not** appear, the anchors file was not written where the glob looks.

- [ ] **Step 4: Write the certificate README**

Create `docs/modelvalidation/certificates/snowball-localvol-1d/2026-08-28/README.md` recording, each in one or two lines:

- the engine certified and the decision;
- the evidence digest from `report.md`;
- the two surface artifacts and their sha256, and that they are committed under `example/modelvalidation/data/`;
- the adopted reference configuration **and the pilot measurement that chose it** (Task 6), including the substeps ladder;
- the pilot control outcomes (Task 5) and the steepness reconciliation (Task 7);
- the scope statement from spec §2, in full — what is covered and what must not be inferred;
- the two items FINDING §7 leaves outstanding: the fleet's existing localvol numbers were produced under the biased `substeps=1` route and must be discarded, and the 40.6 CPU-hour cost projection needs re-measuring;
- that stage 11 may now delegate `delta_authority` for `localvol` to this evidence, as `heston` / `heston_slv` do.

- [ ] **Step 5: Commit**

```bash
git add docs/modelvalidation/certificates/snowball-localvol-1d
git commit -m "cert(modelvalidation): bank the snowball-localvol-1d certification

Certifies LocalVolSnowballPDESolver on two real CSI1000 Dupire surfaces, 16
cells, against a reference whose own convergence is demonstrated rather than
assumed -- the defect the whole request traced back to.

Anchors are picked up automatically by test_banked_certificates.py's
certificates/*/*/anchors.json glob, so the deterministic arm is now guarded on
every commit at roughly one second per cell.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Close the loop on the request

**Files:**
- Modify: `docs/modelvalidation/REQUEST-2026-08-26-localvol-1d-pde.md`
- Modify: `docs/modelvalidation/RELEASE_PROCEDURE.md` (only if the local-vol arm needs a mention)

- [ ] **Step 1: Append a resolution note to the REQUEST**

Add a short closing section stating that the request's first branch was taken
(FINDING-2026-08-26 — the reference was the guilty party), and that certification
was nevertheless carried out for banked evidence, `delta_authority` delegation
and CI anchors, pointing at the banked certificate directory. Do not rewrite the
request's original text.

- [ ] **Step 2: Run the whole modelvalidation suite**

Run:
```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest \
  test/modelvalidation/ -v
```
Expected: all pass, including the new banked-certificate parametrization.

- [ ] **Step 3: Run the broader suite for regressions**

Run:
```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest \
  test/modelvalidation/ test/mo_volmodels/ -q
```
Expected: all pass. If `test/mo_volmodels/` rewrites tracked sample files, `git checkout --` them; never `git add example/`.

- [ ] **Step 4: Commit**

```bash
git add docs/modelvalidation
git commit -m "docs(modelvalidation): close REQUEST-2026-08-26 with the banked certificate

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §1 motivation | Task 9 step 4 (README), Task 10 step 1 |
| §2 scope statement | Task 4 (YAML header), Task 9 step 4 |
| §3 committed surfaces | Task 1 steps 1–2 |
| §4 surfaces and extrapolation | Task 4 (`inside_listed_grid` / `ordinary` pair) |
| §5 architecture, three builders | Tasks 1, 2, 3 |
| §5 one discretization | Task 3 (`_DEFAULT_SUBSTEPS`, single value in `config()`) |
| §5 estimator split rules, QUASI scrambling | Task 3 (`self.method`), Task 6 step 3 |
| §6 sixteen cases | Task 4 |
| §7.1 convergence ladder | Task 6 |
| §7.2 estimator measured | Task 6 |
| §7.3 flat-surface control | Task 5 |
| §7.4 `--quick` | Task 8 step 1 |
| §7.5 economic-scale verification | Task 5 |
| §8 bounds/scale/sampling | Task 4 (YAML + tests) |
| §9 run, bank, anchors | Tasks 8, 9 |
| §10 open item | Task 7 |

No gaps.

**Placeholder scan:** no `TBD`/`TODO`/"handle edge cases"/"similar to Task N". Task 6 step 3 and Task 7 step 3 are *decision* steps, but each states its rule and both branches explicitly rather than deferring.

**Type consistency:** `load_surface(surface: str, rate: float) -> _Surface` is used with the same signature in Tasks 1, 2, 3, 5, 7. `resolve_product_spec(environment, product)` takes two arguments everywhere (an earlier draft had a third `s0` parameter — corrected to derive `s0` internally). `_Surface` field names `artifact/grid/rate_curve/div_yield/local_vol` are consistent across the module and all probes. `build_localvol_pde_candidate` / `build_localvol_mc_reference` keyword arguments match the registry's calling convention in `yaml_loader.py:195` and `:216`.
