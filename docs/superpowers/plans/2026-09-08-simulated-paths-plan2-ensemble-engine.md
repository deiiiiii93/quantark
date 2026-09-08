# Simulated-Path Backtest — Plan 2 of 4: Ensemble Engine, Exact Repricing and the Conformance Oracle

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the plan-1 data layer into the first usable simulated-path backtest: an exact repricing provider with a memory state cache, a vectorised snowball lifecycle and futures hedge ledger, the daily ensemble loop with the replay engine's accounting, and a conformance oracle that reproduces `ReplayBacktestEngine` bit for bit on a single path.

**Architecture:** `EnsembleBacktestEngine.run(MarketPath)` walks the calendar once, holding every per-path quantity as a numpy array of shape `(n_paths,)` and every per-product-per-path quantity as `(n_products, n_paths)`. Pricing is the only scalar work: a provider is handed the day's alive states, deduplicates them by key, prices each distinct state once through the real engine and scatters the results back. Lifecycle, hedging and accounting are pure array arithmetic whose rules are lifted from `AutocallableLifecycleTracker`, `FuturesHedgePosition` and `ReplayBacktestEngine._record_day`. The observation schedule is resolved once, by the tracker itself, through a new public method, so the two code paths cannot drift.

**Tech Stack:** Python 3.10–3.13, numpy, pandas, pytest (`-n0` while iterating; the parallel default for the final run).

**Spec:** `docs/superpowers/specs/2026-09-08-simulated-path-backtest-design.md`, sections 4.2, 4.3, 7.1, 7.3 (exact mode only), 7.4 (memory tier only), 7.5 (zero-gate case), 8, 10, 11, 12 and the plan-2 line of 16. Plan 1 (`docs/superpowers/plans/2026-09-08-simulated-paths-plan1-market-paths.md`) is merged into the branch and supplies everything under `quantark.backtest.simulation` today.

## Global Constraints

- Work on branch `feat/simulated-path-backtest` (already exists, holds the spec, plan 1 and its implementation). Verify `git log --oneline -1` shows `4cd28de5` or later before starting. Run tests as `/Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 <files>` (add `PYTHONPATH=$PWD` in a worktree so its source shadows the editable install).
- Commits: every message ends with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Never `git add example/` wholesale (two `example/mo_volmodels/data/` sample files churn under test runs; `git checkout --` them before staging). Files under `docs/` need `git add -f`. `CLAUDE.md` files are never added.
- **No behaviour change in `quantark.backtest.replay` or `quantark.asset.equity.lifecycle`.** Task 5 adds one new public method to `AutocallableLifecycleTracker`; it is additive and calls the tracker's existing private helpers. `test/replay_golden`, `test/test_replay_dividend_source.py`, `test/test_snowball_q_term_structure_study.py` and `test/test_otc_lifecycle*.py` must stay green with no test edits.
- Fail closed (spec §12): raise `quantark.util.exceptions.ValidationError`; never substitute a provider, engine, seed or tolerance, and never fall back to an approximation.
- No hidden defaults on the new config objects: cache budget, gate tolerances and the hedge strategy are required arguments; a missing one is a `TypeError`, a bad one a `ValidationError`.
- Determinism must not depend on `PYTHONHASHSEED`. Every state key and every derived seed goes through `hashlib.blake2b`, never Python's builtin `hash()` on a string or tuple.
- Use `quantark.util.numerical` helpers (`is_zero`, `is_close`) instead of raw float comparisons in library code, EXCEPT where this plan reproduces a replay threshold verbatim (the `1e-12` position guards in `FuturesHedgePosition` and `_rebalance`); those are copied exactly, with a comment, because the goal is bit-identical behaviour.
- Style: PEP 8, dataclasses with type hints, docstrings on all public APIs, `from __future__ import annotations` at the top of every new module.

---

## Scope of this plan

**In:** exact repricing (one engine call per distinct state), the in-memory state cache, the vectorised lifecycle and hedge, the daily loop and its accounting, a minimal `EnsembleResults` holding the state cube and per-path frames, and the bit-for-bit oracle.

**Out, by design, with the plan that carries them:**

| Deferred | Plan |
|---|---|
| Spot ladder (`spot_step`), vol/`q` bucketing (`vol_step`, `q_step`) | 3 |
| On-disk cache tier (`disk_dir`) | 3 |
| The sampling accuracy gate (in exact mode the gate is identically zero, spec §7.5) | 3 |
| `LifeSurfacePricer` and the grid layer's extra time nodes | 3 |
| Process-pool batching (`workers`, `batch_paths`) | 3 |
| `EnsembleResults.summary` / `distribution` / `paired` / `to_dir` | 4 |
| The example study | 4 |

Deferred settings are **absent from the config objects**, not present-and-ignored. `PricingProviderConfig.provider` is `Literal["repricing"]` in this plan and gains `"life_surface"` when that provider lands; `spot_step`, `vol_step`, `q_step`, `disk_dir`, `workers` and `batch_paths` are added by the plan that implements them. A config that cannot be honoured is never accepted.

Two deliberate departures from the spec's draft interfaces, both recorded in the module README in Task 11:

1. **`DayStates` carries the whole pricing environment, not only a scalar `q_T`.** Under a term dividend source the replay engine hands the pricing engine a whole `q(T)` curve, and it also sets a basis yield on the environment; a pricer given only the scalar yield at the remaining maturity could not be bit-identical with it. `DayStates` therefore carries `div_yield` (the objects), `basis_yield`, and `env_key` (the identity of the rate and carry curve those two are built from, which is what the cache keys on), and keeps `q_T` as the scalar the state cube records in `pricing_q`, exactly what `ProductReplay.recorded_pricing_q` records.
2. **`EnsembleConfig` has no `rate_schedule`.** The spec listed one before `MarketPath` carried a `rate` channel; the rate now comes from the path, per path per day.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `quantark/backtest/simulation/config.py` (create) | `GateConfig`, `CacheConfig`, `PricingProviderConfig`, `EnsembleConfig` and their validation | 1 |
| `quantark/backtest/simulation/pricing/__init__.py` (create) | subpackage exports | 2 |
| `quantark/backtest/simulation/pricing/base.py` (create) | `DayStates`, `StateKey`, `row_keys`, `GateReport`, `PathPricer` protocol | 2 |
| `quantark/backtest/simulation/pricing/cache.py` (create) | `StateCache` memory tier + `CacheStats` | 3 |
| `quantark/backtest/simulation/pricing/repricing.py` (create) | `RepricingPricer` (exact mode) | 4 |
| `quantark/asset/equity/lifecycle/autocallable.py` (modify) | new public `resolve_calendar_schedule` + `CalendarSchedule` | 5 |
| `quantark/backtest/simulation/lifecycle.py` (create) | `VectorLifecycle` | 6 |
| `quantark/backtest/simulation/hedge.py` (create) | `VectorHedgeLedger`, `day_active_contract`, `target_contracts_vector` | 7, 8 |
| `quantark/backtest/simulation/engine.py` (create) | `StateCube`, `EnsembleBacktestEngine`, `EnsembleResults` (minimal) | 9 |
| `quantark/backtest/simulation/conformance.py` (create) | `OracleReport`, `run_oracle`, CLI | 10 |
| `quantark/backtest/simulation/__init__.py` (modify) | public exports | 11 |
| `quantark/backtest/simulation/README.md` (modify) | plan-2 sections | 11 |
| `test/simulation/conftest.py` (modify) | shared product / config fixtures | 1 |
| `test/simulation/test_config.py` (create) | Task 1 | 1 |
| `test/simulation/test_pricing_base.py` (create) | Task 2 | 2 |
| `test/simulation/test_cache.py` (create) | Task 3 | 3 |
| `test/simulation/test_repricing.py` (create) | Task 4 | 4 |
| `test/test_lifecycle_calendar_schedule.py` (create) | Task 5 (lives with the lifecycle tests, not the simulation ones) | 5 |
| `test/simulation/test_lifecycle.py` (create) | Task 6 | 6 |
| `test/simulation/test_hedge.py` (create) | Tasks 7, 8 | 7, 8 |
| `test/simulation/test_engine.py` (create) | Task 9 | 9 |
| `test/simulation/test_conformance.py` (create) | Task 10 | 10 |

Conventions used by every task:

- Per-path arrays have shape `(n_paths,)`; per-product-per-path arrays `(n_products, n_paths)`; the state cube `(n_days,)` or `(n_paths, n_days)`.
- `p` indexes products, `i` paths, `d` days, `j` contracts, `k` observation indices.
- A "hand" is one futures contract. `quantity` on a `ReplayProduct` is the position size and is negative for the seller.

---

### Task 1: Configuration objects

**Files:**
- Create: `quantark/backtest/simulation/config.py`
- Modify: `test/simulation/conftest.py` (append product and config fixtures)
- Create: `test/simulation/test_config.py`

**Interfaces:**
- Consumes: `ReplayProduct`, `HedgeSpec`, `AutocallableEngineConfig` (`quantark.backtest.replay`), `AutocallableDeltaHedgeStrategy` (`quantark.backtest.strategy.futures_delta_strategy`), `TransactionCostModel` (`quantark.backtest.transaction_costs`), `SnowballOption` (`quantark.asset.equity.product.option.snowball_option`).
- Produces:
  - `GateConfig(sample_states: int, pv_tolerance_bp: float, delta_tolerance_hands: float)` frozen.
  - `CacheConfig(memory_bytes: int)` frozen.
  - `PricingProviderConfig(provider: Literal["repricing"], cache: CacheConfig, gate: GateConfig)` frozen.
  - `EnsembleConfig(products, engine_config, hedge, strategy, transaction_cost_model, pricing, underlying="equity_index", delta_bump_size=None, gamma_bump_size=None, allow_data_end=False, metadata=...)`; property `quantities -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

Append to `test/simulation/conftest.py`:

```python
from quantark.asset.equity.product.option import create_standard_snowball
from quantark.backtest.futures_ledger import FuturesRollPolicy
from quantark.backtest.replay import AutocallableEngineConfig, HedgeSpec, ReplayProduct
from quantark.backtest.strategy.futures_delta_strategy import AutocallableDeltaHedgeStrategy
from quantark.backtest.transaction_costs import ZeroCostModel
from quantark.asset.equity.param import PDEParams
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import EngineType

KO_BARRIER = 1.03 * SPOT
KI_BARRIER = 0.75 * SPOT


def short_snowball(maturity_days: int = 6, *, ko_days=(2, 5), ki_days=(1, 3), continuous_ki: bool = False):
    """A snowball short enough to run a whole life inside a test."""
    return create_standard_snowball(
        initial_price=SPOT, strike=SPOT, maturity=maturity_days / 365.0, contract_multiplier=1.0,
        ko_barrier=KO_BARRIER, ki_barrier=KI_BARRIER, ko_rate=0.20, num_observations=len(ko_days),
        ko_observation_dates=[d / 365.0 for d in ko_days],
        ki_observation_type=ObservationType.CONTINUOUS if continuous_ki else ObservationType.DISCRETE,
        ki_continuous=continuous_ki,
        ki_observation_dates=None if continuous_ki else [d / 365.0 for d in ki_days],
        include_principal=True,
    )


def pde_engine_config(**overrides) -> AutocallableEngineConfig:
    """PDE with stock parameters, like the replay goldens: the QUAD recursion
    cannot age the short fixture's KO grid day by day."""
    kwargs = dict(pricing_engine_type=EngineType.PDE, pde_params=PDEParams())
    kwargs.update(overrides)
    return AutocallableEngineConfig(**kwargs)


def ensemble_config(products=None, **overrides):
    """A one-snowball seller book with a zero-cost front-month futures hedge."""
    from quantark.backtest.simulation.config import CacheConfig, EnsembleConfig, GateConfig, PricingProviderConfig

    if products is None:
        products = [ReplayProduct(product=short_snowball(), quantity=-1.0, position_id=1, has_lifecycle=True)]
    kwargs = dict(
        products=products,
        engine_config=pde_engine_config(),
        hedge=HedgeSpec(kind="futures", multiplier=200.0, roll_policy=FuturesRollPolicy()),
        strategy=AutocallableDeltaHedgeStrategy(delta_threshold=0.0, hedge_ratio=1.0, target_delta=0.0),
        transaction_cost_model=ZeroCostModel(),
        pricing=PricingProviderConfig(
            provider="repricing",
            cache=CacheConfig(memory_bytes=8_000_000),
            gate=GateConfig(sample_states=0, pv_tolerance_bp=0.0, delta_tolerance_hands=0.0),
        ),
        underlying="CSI1000",
    )
    kwargs.update(overrides)
    return EnsembleConfig(**kwargs)
```

`test/simulation/test_config.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.backtest.replay import AutocallableEngineConfig, HedgeSpec, ReplayProduct
from quantark.backtest.simulation.config import CacheConfig, EnsembleConfig, GateConfig, PricingProviderConfig
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

from .conftest import SPOT, ensemble_config, short_snowball


def test_a_valid_config_exposes_its_quantities():
    cfg = ensemble_config(products=[
        ReplayProduct(product=short_snowball(), quantity=-1.0, position_id=1, has_lifecycle=True),
        ReplayProduct(product=short_snowball(), quantity=2.0, position_id=2, has_lifecycle=True),
    ])
    assert cfg.quantities == pytest.approx(np.array([-1.0, 2.0]))
    assert cfg.pricing.provider == "repricing"


def test_gate_and_cache_tolerances_are_required_and_validated():
    with pytest.raises(TypeError):
        GateConfig(sample_states=5)                       # tolerances have no default
    with pytest.raises(ValidationError):
        GateConfig(sample_states=-1, pv_tolerance_bp=1.0, delta_tolerance_hands=0.1)
    with pytest.raises(ValidationError):
        GateConfig(sample_states=5, pv_tolerance_bp=-1.0, delta_tolerance_hands=0.1)
    with pytest.raises(ValidationError):
        CacheConfig(memory_bytes=0)


def test_only_snowballs_and_only_a_futures_hedge_are_accepted():
    vanilla = EuropeanVanillaOption(strike=SPOT, option_type=OptionType.CALL, maturity=0.5)
    with pytest.raises(ValidationError):
        ensemble_config(products=[ReplayProduct(product=vanilla, quantity=1.0, position_id=1, has_lifecycle=True)])
    with pytest.raises(ValidationError):
        ensemble_config(hedge=HedgeSpec(kind="spot"))
    with pytest.raises(ValidationError):
        ensemble_config(products=[])


def test_unsupported_dividend_sources_and_surface_vol_fail_closed():
    with pytest.raises(ValidationError):
        ensemble_config(engine_config=AutocallableEngineConfig(dividend_source="surface_forwards"))
    with pytest.raises(ValidationError):
        ensemble_config(engine_config=AutocallableEngineConfig(vol_source="surface"))


def test_provider_literal_admits_only_what_is_implemented():
    with pytest.raises(ValidationError):
        PricingProviderConfig(provider="life_surface",
                              cache=CacheConfig(memory_bytes=1024),
                              gate=GateConfig(sample_states=0, pv_tolerance_bp=0.0, delta_tolerance_hands=0.0))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_config.py -q`
Expected: `ModuleNotFoundError: No module named 'quantark.backtest.simulation.config'`

- [ ] **Step 3: Write the implementation**

`quantark/backtest/simulation/config.py`:

```python
"""Configuration for the simulated-path ensemble backtest (spec 11).

Nothing here has a hidden default: a cache budget, gate tolerances and a
hedge strategy are required arguments.  A setting this package cannot yet
honour is absent from these objects rather than accepted and ignored, so a
config that is constructed is a config that runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.backtest.replay import AutocallableEngineConfig, HedgeSpec, ReplayProduct
from quantark.backtest.transaction_costs import TransactionCostModel
from quantark.util.exceptions import ValidationError

PROVIDERS = ("repricing",)  # "life_surface" joins this when that provider lands


@dataclass(frozen=True)
class GateConfig:
    """Accuracy budget for the pricing provider (spec 7.5).

    In exact repricing mode the gap is identically zero and the report says
    so; the sampling gate that consumes ``sample_states`` arrives with the
    approximate providers.
    """

    sample_states: int
    pv_tolerance_bp: float
    delta_tolerance_hands: float

    def __post_init__(self) -> None:
        if int(self.sample_states) < 0:
            raise ValidationError("GateConfig.sample_states must be non-negative")
        if float(self.pv_tolerance_bp) < 0.0 or float(self.delta_tolerance_hands) < 0.0:
            raise ValidationError("GateConfig tolerances must be non-negative")


@dataclass(frozen=True)
class CacheConfig:
    """State-cache budget (spec 7.4); the on-disk tier arrives with the ladder."""

    memory_bytes: int

    def __post_init__(self) -> None:
        if int(self.memory_bytes) <= 0:
            raise ValidationError("CacheConfig.memory_bytes must be positive")


@dataclass(frozen=True)
class PricingProviderConfig:
    """Which pricer runs and how much it may spend."""

    provider: Literal["repricing"]
    cache: CacheConfig
    gate: GateConfig

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise ValidationError(
                f"provider must be one of {PROVIDERS}, got {self.provider!r}"
            )


@dataclass
class EnsembleConfig:
    """A book, an engine, a hedge and a pricer: one cell of a simulated run.

    The rate comes from the ``MarketPath`` (per path, per day), so there is
    no rate schedule here.  Batch parallelism is not part of this
    configuration yet; the engine runs the whole batch in process.
    """

    products: List[ReplayProduct]
    engine_config: AutocallableEngineConfig
    hedge: HedgeSpec
    strategy: Any
    transaction_cost_model: TransactionCostModel
    pricing: PricingProviderConfig
    underlying: str = "equity_index"
    delta_bump_size: Optional[float] = None
    gamma_bump_size: Optional[float] = None
    allow_data_end: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.products:
            raise ValidationError("EnsembleConfig needs at least one product")
        for bp in self.products:
            if not isinstance(bp.product, SnowballOption):
                raise ValidationError(
                    "the simulated-path engine prices standard snowballs only, got "
                    f"{type(bp.product).__name__} for position {bp.position_id}"
                )
            if not bp.has_lifecycle:
                raise ValidationError(
                    f"position {bp.position_id} must carry a lifecycle in a simulated run"
                )
        if self.hedge.kind != "futures":
            raise ValidationError(
                f"the simulated-path engine hedges with futures only, got {self.hedge.kind!r}"
            )
        if self.strategy is None:
            raise ValidationError("EnsembleConfig.strategy is required")
        source = getattr(self.engine_config, "dividend_source", None)
        if source not in (None, "active_contract", "futures_curve"):
            raise ValidationError(
                f"dividend_source {source!r} has no meaning on a simulated path; "
                "use None, 'active_contract' or 'futures_curve'"
            )
        if getattr(self.engine_config, "vol_source", "scalar") != "scalar":
            raise ValidationError(
                "a simulated path carries one ATM vol per day: vol_source must be 'scalar'"
            )

    @property
    def quantities(self) -> np.ndarray:
        return np.array([float(bp.quantity) for bp in self.products])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_config.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/config.py test/simulation/conftest.py test/simulation/test_config.py
git commit -m "feat(backtest/simulation): ensemble configuration objects with fail-closed validation

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: State keys and the pricer interface

**Files:**
- Create: `quantark/backtest/simulation/pricing/__init__.py`, `quantark/backtest/simulation/pricing/base.py`
- Create: `test/simulation/test_pricing_base.py`

**Interfaces:**
- Produces:
  - `row_keys(rows: np.ndarray) -> np.ndarray` — an int64 identity per row of a 2-D float64 array, from `blake2b` of the row's bytes. Stable across processes and runs (`PYTHONHASHSEED` never enters), equal for equal bytes, and `-0.0` is normalised to `0.0` so it keys with `0.0`.
  - `DayStates(day_index, date, path_index, spot, vol, rate, q_T, div_yield, basis_yield, env_key, knocked_in)` NamedTuple; `date` is the calendar day (`day_index` is its position), `m` = number of alive paths; `len(states)` is `m`; `states.empty` is `m == 0`.
  - `StateKey(product_fingerprint: str, day_index: int, knocked_in: bool, spot_key: int, vol_key: int, env_key: int, engine_fingerprint: str)` frozen, hashable; `digest() -> bytes` (blake2b of a canonical byte encoding) and `seed() -> int` (`int.from_bytes(digest()[:4])`, the MC seed policy of spec §7.3).
  - `float_key(values: np.ndarray) -> np.ndarray` — the raw float64 bits as int64, `-0.0` normalised; the exact-mode bucket key.
  - `GateReport(mode: str, sampled: int, max_pv_gap_bp: float, max_delta_gap_hands: float, passed: bool)` frozen, `as_dict()`.
  - `PathPricer` Protocol: `price_day(states) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning `(pv, delta, gamma)` per alive path per unit product; `verify(states, gate) -> GateReport`; `fingerprint() -> str`; `stats() -> dict`.

- [ ] **Step 1: Write the failing tests**

`test/simulation/test_pricing_base.py`:

```python
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.simulation.pricing.base import (
    DayStates,
    GateReport,
    StateKey,
    float_key,
    row_keys,
)


def test_row_keys_are_equal_for_equal_bytes_and_differ_otherwise():
    rows = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.000000001]])
    keys = row_keys(rows)
    assert keys.dtype == np.int64 and keys.shape == (3,)
    assert keys[0] == keys[1] != keys[2]


def test_row_keys_normalise_negative_zero():
    assert row_keys(np.array([[0.0, 1.0]]))[0] == row_keys(np.array([[-0.0, 1.0]]))[0]


def test_row_keys_do_not_depend_on_the_hash_seed():
    code = (
        "import numpy as np;"
        "from quantark.backtest.simulation.pricing.base import row_keys;"
        "print(int(row_keys(np.array([[6000.0, 0.22, -0.03]]))[0]))"
    )
    outs = {
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"}, check=True).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert len(outs) == 1


def test_float_key_round_trips_a_value_and_separates_neighbours():
    values = np.array([6000.0, 6000.0, np.nextafter(6000.0, 7000.0), -0.0, 0.0])
    keys = float_key(values)
    assert keys[0] == keys[1] != keys[2]
    assert keys[3] == keys[4]


def test_state_key_digest_and_seed_are_deterministic_and_specific():
    a = StateKey("prod", 3, False, 11, 22, 33, "eng")
    b = StateKey("prod", 3, False, 11, 22, 33, "eng")
    c = StateKey("prod", 3, True, 11, 22, 33, "eng")   # differs only in the KI flag
    assert a == b and hash(a) == hash(b)
    assert a.digest() == b.digest() != c.digest()
    assert a.seed() == b.seed() != c.seed()
    assert 0 <= a.seed() < 2**32


def test_day_states_length_and_emptiness():
    day = pd.Timestamp("2024-01-02")
    empty = DayStates(day_index=0, date=day, path_index=np.array([], dtype=int), spot=np.array([]),
                      vol=np.array([]), rate=np.array([]), q_T=np.array([]), div_yield=(),
                      basis_yield=np.array([]), env_key=np.array([], dtype=np.int64),
                      knocked_in=np.array([], dtype=bool))
    assert len(empty) == 0 and empty.empty
    one = DayStates(day_index=1, date=day, path_index=np.array([4]), spot=np.array([6000.0]),
                    vol=np.array([0.2]), rate=np.array([0.02]), q_T=np.array([0.05]), div_yield=(None,),
                    basis_yield=np.array([-0.03]), env_key=np.array([7], dtype=np.int64),
                    knocked_in=np.array([False]))
    assert len(one) == 1 and not one.empty
    assert one.date == day


def test_gate_report_records_the_exact_mode_zero():
    report = GateReport(mode="exact", sampled=0, max_pv_gap_bp=0.0, max_delta_gap_hands=0.0, passed=True)
    assert report.as_dict()["mode"] == "exact" and report.as_dict()["max_pv_gap_bp"] == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_pricing_base.py -q`
Expected: `ModuleNotFoundError: No module named 'quantark.backtest.simulation.pricing'`

- [ ] **Step 3: Write the implementation**

`quantark/backtest/simulation/pricing/__init__.py`:

```python
"""Pricing providers for the simulated-path ensemble (spec 7)."""
from __future__ import annotations

from .base import DayStates, GateReport, PathPricer, StateKey, float_key, row_keys

__all__ = ["DayStates", "GateReport", "PathPricer", "StateKey", "float_key", "row_keys"]
```

`quantark/backtest/simulation/pricing/base.py`:

```python
"""The pricer interface and the state identity it caches on (spec 7.1, 7.4).

Every key here is derived with ``blake2b`` rather than Python's builtin
``hash``: string and tuple hashing is salted per process, and a cache or an
MC seed that moved with ``PYTHONHASHSEED`` would make a run unreproducible
across machines.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Protocol, Sequence, Tuple

import numpy as np


def _normalised(values: np.ndarray) -> np.ndarray:
    """float64 with ``-0.0`` folded onto ``0.0`` so the two key together."""
    out = np.ascontiguousarray(values, dtype=np.float64).copy()
    out[out == 0.0] = 0.0
    return out


def float_key(values: np.ndarray) -> np.ndarray:
    """The float64 bit pattern of each value as an int64 key (exact bucketing)."""
    return _normalised(np.asarray(values)).view(np.int64)


def row_keys(rows: np.ndarray) -> np.ndarray:
    """A stable int64 identity per row of a 2-D float64 array.

    Used for state channels that are defined by several numbers at once --
    a whole dividend curve, say -- where one float cannot stand for the
    state.  Equal bytes give equal keys, on any machine and in any process.
    """
    data = _normalised(np.atleast_2d(np.asarray(rows, dtype=np.float64)))
    out = np.empty(data.shape[0], dtype=np.int64)
    for i in range(data.shape[0]):
        digest = hashlib.blake2b(data[i].tobytes(), digest_size=8).digest()
        out[i] = int.from_bytes(digest, "big", signed=True)
    return out


class DayStates(NamedTuple):
    """The alive states of one day handed to a pricer (spec 7.1).

    ``div_yield`` holds the dividend OBJECT each path's engine call
    receives -- under a term source that is a whole ``q(T)`` curve, and a
    scalar could not stand in for it -- while ``q_T`` is the scalar zero
    yield at the remaining maturity that the state cube records as
    ``pricing_q``, exactly what ``ProductReplay.recorded_pricing_q``
    records.

    ``env_key`` identifies everything the pricing environment holds beyond
    spot and vol: the rate and the day's carry curve, which between them
    fix the dividend object AND the basis yield.  Keying on those inputs
    rather than on the objects keeps the cache exact without asking a
    dividend curve to hash itself.
    """

    day_index: int
    date: Any                   # pd.Timestamp: the calendar day itself
    path_index: np.ndarray      # (m,) positions in the batch
    spot: np.ndarray            # (m,)
    vol: np.ndarray             # (m,)
    rate: np.ndarray            # (m,)
    q_T: np.ndarray             # (m,)
    div_yield: Tuple[Any, ...]  # (m,)
    basis_yield: np.ndarray     # (m,)
    env_key: np.ndarray         # (m,) int64
    knocked_in: np.ndarray      # (m,) bool

    def __len__(self) -> int:
        return int(self.path_index.size)

    @property
    def empty(self) -> bool:
        return len(self) == 0


@dataclass(frozen=True)
class StateKey:
    """What a priced state is identified by (spec 7.4).

    Nothing about the hedge, the cost model or the strategy is in the key,
    so cells that differ only there share cache entries.
    """

    product_fingerprint: str
    day_index: int
    knocked_in: bool
    spot_key: int
    vol_key: int
    env_key: int
    engine_fingerprint: str

    def digest(self) -> bytes:
        h = hashlib.blake2b(digest_size=16)
        h.update(self.product_fingerprint.encode())
        h.update(b"\x00")
        h.update(self.engine_fingerprint.encode())
        h.update(
            np.array(
                [self.day_index, int(self.knocked_in), self.spot_key, self.vol_key, self.env_key],
                dtype=np.int64,
            ).tobytes()
        )
        return h.digest()

    def seed(self) -> int:
        """The MC seed this state prices with (spec 7.3): stable, state-specific."""
        return int.from_bytes(self.digest()[:4], "big")


@dataclass(frozen=True)
class GateReport:
    """What the provider claims about its own accuracy (spec 7.5)."""

    mode: str
    sampled: int
    max_pv_gap_bp: float
    max_delta_gap_hands: float
    passed: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode, "sampled": int(self.sampled),
            "max_pv_gap_bp": float(self.max_pv_gap_bp),
            "max_delta_gap_hands": float(self.max_delta_gap_hands),
            "passed": bool(self.passed),
        }


class PathPricer(Protocol):
    """Prices one day's alive states, per unit product."""

    def price_day(self, states: DayStates) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(pv, delta, gamma)``, each shape ``(len(states),)``, per unit product."""

    def verify(self, states: DayStates, gate: Any) -> GateReport:
        """Measure this provider's gap against direct repricing."""

    def fingerprint(self) -> str:
        """Identity of the engine and settings behind the prices."""

    def stats(self) -> Dict[str, Any]:
        """Counters worth putting in the run manifest."""
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_pricing_base.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/pricing test/simulation/test_pricing_base.py
git commit -m "feat(backtest/simulation): pricer interface and process-stable state keys

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `StateCache` memory tier

**Files:**
- Create: `quantark/backtest/simulation/pricing/cache.py`
- Create: `test/simulation/test_cache.py`

**Interfaces:**
- Consumes: `StateKey` (Task 2), `CacheConfig` (Task 1).
- Produces:
  - `CacheStats(hits: int, misses: int, evictions: int, entries: int, bytes_used: int)` frozen, `as_dict()`.
  - `StateCache(config: CacheConfig)`: `get(key) -> tuple[float, float, float] | None`, `put(key, pv, delta, gamma) -> None`, `get_many(keys) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` returning `(hit_mask, pv, delta, gamma)`, `stats() -> CacheStats`, `clear()`. Recency-ordered eviction once `bytes_used` would exceed the budget; `ENTRY_BYTES = 88` (three float64 values plus the key digest and dict overhead, counted as a fixed cost so the budget is a predictable entry count).

- [ ] **Step 1: Write the failing tests**

`test/simulation/test_cache.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from quantark.backtest.simulation.config import CacheConfig
from quantark.backtest.simulation.pricing.base import StateKey
from quantark.backtest.simulation.pricing.cache import ENTRY_BYTES, StateCache


def _key(i: int) -> StateKey:
    return StateKey("prod", 0, False, i, 0, 0, "eng")


def test_a_stored_state_comes_back_and_counts_as_a_hit():
    cache = StateCache(CacheConfig(memory_bytes=10 * ENTRY_BYTES))
    assert cache.get(_key(1)) is None
    cache.put(_key(1), 12.5, -0.4, 0.001)
    assert cache.get(_key(1)) == (12.5, -0.4, 0.001)
    stats = cache.stats()
    assert (stats.hits, stats.misses, stats.entries) == (1, 1, 1)


def test_the_least_recently_used_entry_is_evicted_at_the_budget():
    cache = StateCache(CacheConfig(memory_bytes=3 * ENTRY_BYTES))
    for i in range(3):
        cache.put(_key(i), float(i), 0.0, 0.0)
    cache.get(_key(0))                      # 0 becomes the most recent, 1 the oldest
    cache.put(_key(3), 3.0, 0.0, 0.0)
    assert cache.get(_key(1)) is None
    assert cache.get(_key(0)) == (0.0, 0.0, 0.0)
    assert cache.stats().evictions == 1
    assert cache.stats().entries == 3


def test_get_many_returns_a_hit_mask_and_the_stored_values():
    cache = StateCache(CacheConfig(memory_bytes=10 * ENTRY_BYTES))
    cache.put(_key(1), 1.0, 0.1, 0.01)
    cache.put(_key(3), 3.0, 0.3, 0.03)
    hit, pv, delta, gamma = cache.get_many([_key(0), _key(1), _key(2), _key(3)])
    assert list(hit) == [False, True, False, True]
    assert pv[1] == pytest.approx(1.0) and gamma[3] == pytest.approx(0.03)
    assert np.isnan(pv[0]) and np.isnan(pv[2])


def test_clear_empties_the_cache_and_the_counters():
    cache = StateCache(CacheConfig(memory_bytes=10 * ENTRY_BYTES))
    cache.put(_key(1), 1.0, 0.0, 0.0)
    cache.get(_key(1))
    cache.clear()
    assert cache.stats().entries == 0 and cache.stats().hits == 0
    assert cache.get(_key(1)) is None


def test_a_budget_below_one_entry_is_rejected():
    with pytest.raises(Exception):
        StateCache(CacheConfig(memory_bytes=ENTRY_BYTES - 1))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_cache.py -q`
Expected: `ModuleNotFoundError: No module named 'quantark.backtest.simulation.pricing.cache'`

- [ ] **Step 3: Write the implementation**

`quantark/backtest/simulation/pricing/cache.py`:

```python
"""In-memory state cache for priced states (spec 7.4, memory tier).

The key holds nothing about the hedge, the cost model or the strategy, so
two cells that differ only there price the same states once between them.
The on-disk tier arrives with the approximate providers.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from quantark.util.exceptions import ValidationError

from ..config import CacheConfig
from .base import StateKey

#: Charged per entry: three float64 values plus the key and dict overhead.
#: A fixed cost makes the budget a predictable entry count rather than a
#: guess at CPython's per-object footprint.
ENTRY_BYTES = 88


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    evictions: int
    entries: int
    bytes_used: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits, "misses": self.misses, "evictions": self.evictions,
            "entries": self.entries, "bytes_used": self.bytes_used,
        }


class StateCache:
    """Recency-ordered cache of ``(pv, delta, gamma)`` by ``StateKey``."""

    def __init__(self, config: CacheConfig) -> None:
        self.capacity = int(config.memory_bytes) // ENTRY_BYTES
        if self.capacity < 1:
            raise ValidationError(
                f"cache memory_bytes must hold at least one entry ({ENTRY_BYTES} bytes)"
            )
        self._store: "OrderedDict[StateKey, Tuple[float, float, float]]" = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: StateKey) -> Optional[Tuple[float, float, float]]:
        value = self._store.get(key)
        if value is None:
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        return value

    def put(self, key: StateKey, pv: float, delta: float, gamma: float) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (float(pv), float(delta), float(gamma))
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)
            self._evictions += 1

    def get_many(
        self, keys: Iterable[StateKey]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(hit_mask, pv, delta, gamma)``; misses carry NaN."""
        keys = list(keys)
        hit = np.zeros(len(keys), dtype=bool)
        pv = np.full(len(keys), np.nan)
        delta = np.full(len(keys), np.nan)
        gamma = np.full(len(keys), np.nan)
        for n, key in enumerate(keys):
            value = self.get(key)
            if value is not None:
                hit[n] = True
                pv[n], delta[n], gamma[n] = value
        return hit, pv, delta, gamma

    def stats(self) -> CacheStats:
        return CacheStats(
            hits=self._hits, misses=self._misses, evictions=self._evictions,
            entries=len(self._store), bytes_used=len(self._store) * ENTRY_BYTES,
        )

    def clear(self) -> None:
        self._store.clear()
        self._hits = self._misses = self._evictions = 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_cache.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/pricing/cache.py test/simulation/test_cache.py
git commit -m "feat(backtest/simulation): in-memory state cache with recency eviction

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `RepricingPricer` — exact mode

**Files:**
- Create: `quantark/backtest/simulation/pricing/repricing.py`
- Create: `test/simulation/test_repricing.py`

**Interfaces:**
- Consumes: `DayStates`, `StateKey`, `float_key`, `GateReport` (Task 2), `StateCache` (Task 3), `create_pricing_engine` (`quantark.backtest.replay.engine_factory`), `AutocallableLifecycleTracker` (`quantark.asset.equity.lifecycle`), `PricingEnvironment`, `SpotQuote`, `FlatVolSurface`, `FlatRateCurve`, `ImpliedBasisYield`.
- Produces:
  - `RepricingPricer(product, *, engine_config, start_date, underlying, cache, delta_bump_size=None, gamma_bump_size=None)`; implements `PathPricer`.
  - `aged_product(date, *, knocked_in)` — the tracker's `product_for_pricing` copy for that day and KI flag, memoised per `(day, knocked_in)`.
  - `price_day(states) -> (pv, delta, gamma)`, `verify(states, gate) -> GateReport` (exact mode: a zero report), `fingerprint()`, `stats()`.
  - `engine_fingerprint(engine_config, delta_bump_size, gamma_bump_size) -> str` (module-level; blake2b of the repr of the engine settings that change a price).

**Two facts this task depends on, each pinned by its own test:**

1. **The aged product does not depend on the market.** `AutocallableLifecycleTracker.product_for_pricing` deep-copies, subtracts elapsed years from `maturity` and calls `barrier_config.time_shift(elapsed, date, env)`. `time_shift` only ever *writes* `env.valuation_date`; it reads no spot, vol, rate or dividend. So one aged copy per `(day, knocked_in)` serves every path, exactly as the replay's single per-day copy serves the whole book.
2. **The replay prices twice.** `ReplayBacktestEngine.run` calls `engine.price(product, env)` for the mark and `engine.calculate_greeks(product, env)` for delta and gamma. Those are separate calls and need not agree for every engine, so `price_day` makes both and takes the PV from `price`, not from the greeks dict.

- [ ] **Step 1: Write the failing tests**

`test/simulation/test_repricing.py`:

```python
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.replay.engine_factory import create_pricing_engine
from quantark.backtest.replay.market import SignedDividendYield
from quantark.backtest.simulation.config import CacheConfig
from quantark.backtest.simulation.pricing.base import DayStates, float_key
from quantark.backtest.simulation.pricing.cache import StateCache
from quantark.backtest.simulation.pricing.repricing import RepricingPricer
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.backtest.replay.market import ImpliedBasisYield
from quantark.priceenv import PricingEnvironment

from .conftest import RATE, SPOT, pde_engine_config, short_snowball

START = pd.Timestamp("2024-01-02")


def _pricer(cache_bytes: int = 8_000_000) -> RepricingPricer:
    return RepricingPricer(
        short_snowball(), engine_config=pde_engine_config(), start_date=START,
        underlying="CSI1000", cache=StateCache(CacheConfig(memory_bytes=cache_bytes)),
    )


def _states(day_index: int, spots, *, knocked_in=None, vol: float = 0.22) -> DayStates:
    spots = np.asarray(spots, dtype=float)
    m = spots.size
    q = np.full(m, 0.05)
    return DayStates(
        day_index=day_index, date=START + pd.Timedelta(days=day_index),
        path_index=np.arange(m), spot=spots, vol=np.full(m, vol),
        rate=np.full(m, RATE), q_T=q, div_yield=tuple(SignedDividendYield(float(x)) for x in q),
        basis_yield=np.full(m, -0.03),
        env_key=float_key(np.full(m, RATE)),
        knocked_in=np.zeros(m, dtype=bool) if knocked_in is None else np.asarray(knocked_in, dtype=bool),
    )


def test_the_aged_product_does_not_depend_on_the_market():
    pricer = _pricer()
    day = START + pd.Timedelta(days=3)
    a = pricer.aged_product(day, knocked_in=False)
    pricer.reset_aging_memo()
    b = pricer.aged_product(day, knocked_in=False)
    assert a.maturity == pytest.approx(b.maturity)
    assert a.barrier_config.ko_observation_dates == pytest.approx(b.barrier_config.ko_observation_dates)
    assert a.barrier_config.ki_observation_dates == pytest.approx(b.barrier_config.ki_observation_dates)


def test_exact_mode_equals_a_direct_engine_call_state_by_state():
    pricer = _pricer()
    day_index, day = 3, START + pd.Timedelta(days=3)
    states = _states(day_index, [SPOT * 0.9, SPOT, SPOT * 1.02])
    pv, delta, gamma = pricer.price_day(states)
    product = pricer.aged_product(day, knocked_in=False)
    engine = create_pricing_engine(product, pde_engine_config())
    for n in range(len(states)):
        env = PricingEnvironment(
            spot_quote=SpotQuote(spot=float(states.spot[n]), asset_name="CSI1000"),
            vol_surface=FlatVolSurface(volatility=float(states.vol[n])),
            rate_curve=FlatRateCurve(rate=float(states.rate[n])),
            div_yield=states.div_yield[n],
            basis_yield=ImpliedBasisYield(float(states.basis_yield[n])),
            valuation_date=day.to_pydatetime(),
        )
        assert pv[n] == float(engine.price(product, env))
        greeks = engine.calculate_greeks(product, env)
        assert delta[n] == float(greeks["delta"])
        assert gamma[n] == float(greeks["gamma"])


def test_the_ki_flag_selects_a_different_product_and_price():
    pricer = _pricer()
    states = _states(3, [SPOT * 0.7, SPOT * 0.7], knocked_in=[False, True])
    pv, _, _ = pricer.price_day(states)
    assert pv[0] != pv[1]
    assert pricer.aged_product(START + pd.Timedelta(days=3), knocked_in=True)._otc_lifecycle_knocked_in is True


def test_a_repeated_state_is_a_cache_hit_and_not_a_second_engine_call():
    pricer = _pricer()
    states = _states(3, [SPOT, SPOT, SPOT * 1.01])
    pv, delta, _ = pricer.price_day(states)
    assert pv[0] == pv[1] and delta[0] == delta[1]
    assert pricer.stats()["engine_calls"] == 2            # deduplicated within the day
    pricer.price_day(states)
    assert pricer.stats()["engine_calls"] == 2            # served from the cache
    assert pricer.stats()["cache"]["hits"] >= 3


def test_an_empty_day_prices_nothing():
    pricer = _pricer()
    pv, delta, gamma = pricer.price_day(_states(3, []))
    assert pv.shape == delta.shape == gamma.shape == (0,)
    assert pricer.stats()["engine_calls"] == 0


def test_the_exact_gate_reports_a_zero_gap():
    pricer = _pricer()
    from quantark.backtest.simulation.config import GateConfig

    report = pricer.verify(_states(3, [SPOT]), GateConfig(sample_states=5, pv_tolerance_bp=1.0,
                                                          delta_tolerance_hands=0.1))
    assert report.mode == "exact" and report.passed
    assert report.max_pv_gap_bp == 0.0 and report.max_delta_gap_hands == 0.0


def test_the_fingerprint_moves_with_the_engine_settings():
    a = _pricer().fingerprint()
    b = RepricingPricer(short_snowball(), engine_config=pde_engine_config(), start_date=START,
                        underlying="CSI1000",
                        cache=StateCache(CacheConfig(memory_bytes=1024)), delta_bump_size=0.02).fingerprint()
    assert a != b and len(a) == 32
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_repricing.py -q`
Expected: `ModuleNotFoundError: No module named 'quantark.backtest.simulation.pricing.repricing'`

- [ ] **Step 3: Write the implementation**

`quantark/backtest/simulation/pricing/repricing.py`:

```python
"""Exact repricing provider: one engine call per distinct state (spec 7.3).

This is the reference procedure the approximate providers are gated
against, and the mode the conformance oracle runs in: for a given day and
path it makes exactly the calls ``ReplayBacktestEngine`` would make, with
the same aged product and the same environment, so the two agree to the
last bit.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.backtest.replay.engine_factory import create_pricing_engine
from quantark.backtest.replay.market import ImpliedBasisYield
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError

from ..config import GateConfig
from .base import DayStates, GateReport, StateKey, float_key
from .cache import StateCache


def engine_fingerprint(
    engine_config: Any, delta_bump_size: Optional[float], gamma_bump_size: Optional[float]
) -> str:
    """Identity of everything that can change a price for a given state."""
    h = hashlib.blake2b(digest_size=16)
    for name in (
        "pricing_engine_type", "method", "vol_model_mc_method", "pde_params", "mc_params",
        "quad_params", "vol_source", "surface_vol_mode", "vol_model", "vol_model_solver",
        "vol_model_engine_options", "dividend_source", "futures_curve_extrapolation",
        "futures_curve_min_tenor_days",
    ):
        h.update(repr(getattr(engine_config, name, None)).encode())
        h.update(b"\x1f")
    h.update(repr((delta_bump_size, gamma_bump_size)).encode())
    return h.hexdigest()


def product_fingerprint(product: Any) -> str:
    """Identity of the contract as priced (terms only, no lifecycle state)."""
    return hashlib.blake2b(repr(product).encode(), digest_size=16).hexdigest()


class RepricingPricer:
    """Prices each distinct ``(spot, vol, env, knocked_in)`` state once, exactly."""

    def __init__(
        self,
        product: Any,
        *,
        engine_config: Any,
        start_date: pd.Timestamp,
        underlying: str,
        cache: StateCache,
        delta_bump_size: Optional[float] = None,
        gamma_bump_size: Optional[float] = None,
    ) -> None:
        self.product = product
        self.engine_config = engine_config
        self.start_date = pd.Timestamp(start_date).normalize()
        self.underlying = underlying
        self.cache = cache
        self.delta_bump_size = delta_bump_size
        self.gamma_bump_size = gamma_bump_size
        self._engine_fp = engine_fingerprint(engine_config, delta_bump_size, gamma_bump_size)
        self._product_fp = product_fingerprint(product)
        self._aged: Dict[Tuple[pd.Timestamp, bool], Any] = {}
        self._engines: Dict[Tuple[pd.Timestamp, bool], Any] = {}
        self._engine_calls = 0

    # -- aging ---------------------------------------------------------

    def reset_aging_memo(self) -> None:
        """Drop the per-day product and engine memo (tests and long runs)."""
        self._aged.clear()
        self._engines.clear()

    def aged_product(self, date: pd.Timestamp, *, knocked_in: bool) -> Any:
        """The tracker's time-decayed copy for ``date`` and the KI flag.

        One copy serves every path: ``product_for_pricing`` reads no market
        data, it only subtracts elapsed time and shifts the observation
        schedule (``barrier_config.time_shift`` writes ``valuation_date``
        on the environment it is handed and reads nothing from it), so the
        environment below is a throwaway.
        """
        key = (pd.Timestamp(date).normalize(), bool(knocked_in))
        product = self._aged.get(key)
        if product is None:
            tracker = AutocallableLifecycleTracker(
                product=self.product, quantity=1.0, has_lifecycle=True,
                lifecycle=AutocallableLifecycleState(knocked_in=bool(knocked_in)),
                start_date=self.start_date,
            )
            product = tracker.product_for_pricing(key[0], self._aging_env(key[0]))
            self._aged[key] = product
        return product

    def _aging_env(self, date: pd.Timestamp) -> PricingEnvironment:
        """A throwaway environment for schedule shifting; never used to price."""
        return PricingEnvironment(
            spot_quote=SpotQuote(spot=1.0, asset_name=self.underlying),
            vol_surface=FlatVolSurface(volatility=0.2),
            rate_curve=FlatRateCurve(rate=0.0),
            valuation_date=pd.Timestamp(date).to_pydatetime(),
        )

    def _engine(self, date: pd.Timestamp, knocked_in: bool):
        key = (pd.Timestamp(date).normalize(), bool(knocked_in))
        engine = self._engines.get(key)
        if engine is None:
            engine = create_pricing_engine(
                self.aged_product(key[0], knocked_in=key[1]),
                self.engine_config,
                delta_bump_size=self.delta_bump_size,
                gamma_bump_size=self.gamma_bump_size,
            )
            self._engines[key] = engine
        return engine

    # -- pricing -------------------------------------------------------

    def price_day(self, states: DayStates) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(pv, delta, gamma)`` per alive state, per unit product."""
        date = pd.Timestamp(states.date).normalize()
        m = len(states)
        pv = np.empty(m)
        delta = np.empty(m)
        gamma = np.empty(m)
        if m == 0:
            return pv, delta, gamma
        keys = self.state_keys(states)
        hit, c_pv, c_delta, c_gamma = self.cache.get_many(keys)
        pv[hit], delta[hit], gamma[hit] = c_pv[hit], c_delta[hit], c_gamma[hit]

        # One engine call per DISTINCT missing key, scattered back to every
        # state that shares it.
        pending: Dict[StateKey, List[int]] = {}
        for n in np.flatnonzero(~hit):
            pending.setdefault(keys[int(n)], []).append(int(n))
        for key, positions in pending.items():
            n = positions[0]
            values = self._price_one(states, n, date, key)
            self.cache.put(key, *values)
            for pos in positions:
                pv[pos], delta[pos], gamma[pos] = values
        return pv, delta, gamma

    def _price_one(
        self, states: DayStates, n: int, date: pd.Timestamp, key: StateKey
    ) -> Tuple[float, float, float]:
        product = self.aged_product(date, knocked_in=bool(states.knocked_in[n]))
        engine = self._engine(date, bool(states.knocked_in[n]))
        self._seed_engine(engine, key)
        env = PricingEnvironment(
            spot_quote=SpotQuote(spot=float(states.spot[n]), asset_name=self.underlying),
            vol_surface=FlatVolSurface(volatility=float(states.vol[n])),
            rate_curve=FlatRateCurve(rate=float(states.rate[n])),
            div_yield=states.div_yield[n],
            basis_yield=ImpliedBasisYield(float(states.basis_yield[n])),
            valuation_date=pd.Timestamp(date).to_pydatetime(),
        )
        try:
            price = float(engine.price(product, env))
            greeks = engine.calculate_greeks(product, env)
        except Exception as exc:  # fail closed with the state in the message
            raise ValidationError(
                f"pricing failed on day {states.day_index} at spot={states.spot[n]!r}, "
                f"vol={states.vol[n]!r}, knocked_in={bool(states.knocked_in[n])}: {exc}"
            ) from exc
        self._engine_calls += 1
        # The replay marks with price() and takes greeks from
        # calculate_greeks(); the two are separate calls there, so they are
        # separate calls here.
        return price, float(greeks["delta"]), float(greeks["gamma"])

    def _seed_engine(self, engine: Any, key: StateKey) -> None:
        """Give an MC engine this state's own seed (spec 7.3).

        A recomputed state then matches its cached value bit for bit, and
        the seed is a function of the state, not of iteration order.
        """
        params = getattr(engine, "params", None)
        if params is not None and hasattr(params, "random_seed"):
            params.random_seed = key.seed()

    def state_keys(self, states: DayStates) -> List[StateKey]:
        spot_keys = float_key(states.spot)
        vol_keys = float_key(states.vol)
        return [
            StateKey(
                product_fingerprint=self._product_fp, day_index=int(states.day_index),
                knocked_in=bool(states.knocked_in[n]), spot_key=int(spot_keys[n]),
                vol_key=int(vol_keys[n]), env_key=int(states.env_key[n]),
                engine_fingerprint=self._engine_fp,
            )
            for n in range(len(states))
        ]

    # -- reporting -----------------------------------------------------

    def verify(self, states: DayStates, gate: GateConfig) -> GateReport:
        """Exact mode prices the states themselves: the gap is zero by construction."""
        return GateReport(mode="exact", sampled=0, max_pv_gap_bp=0.0,
                          max_delta_gap_hands=0.0, passed=True)

    def fingerprint(self) -> str:
        return self._engine_fp

    def stats(self) -> Dict[str, Any]:
        return {
            "provider": "repricing", "mode": "exact", "engine_calls": self._engine_calls,
            "cache": self.cache.stats().as_dict(),
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_repricing.py test/simulation/test_pricing_base.py -q`
Expected: 7 + 7 passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/pricing/repricing.py test/simulation/test_repricing.py
git commit -m "feat(backtest/simulation): exact repricing provider with per-state caching

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `resolve_calendar_schedule` on the lifecycle tracker

**Files:**
- Modify: `quantark/asset/equity/lifecycle/autocallable.py` (append `CalendarSchedule` and one public method; no existing line changes)
- Create: `test/test_lifecycle_calendar_schedule.py`

**Why it lives in the tracker.** The vectorised lifecycle has to fire on exactly the days `AutocallableLifecycleTracker.observe` fires on. The due rule is `_record_is_due`, which mixes a date branch and a rounded elapsed-day branch, and the records come from `_scheduled_records`, which resolves the schedule from the contract's issue date and derives calendar dates for numeric schedules. Reimplementing either in the simulation package would be two rules that must be kept in step by hand. Putting one public resolver on the tracker means the simulation reads the tracker's own answer, and the test below walks a real tracker day by day to prove the answer is the same.

**Interfaces:**
- Produces (in `quantark.asset.equity.lifecycle.autocallable`, re-exported from `quantark.asset.equity.lifecycle`):
  - `CalendarSchedule` frozen dataclass: `ko_due_day, ko_barrier, ko_payoff, ko_settlement_day` (`(n_ko,)` arrays; `due_day`/`settlement_day` are int positions in the calendar, `-1` when the day falls outside it), `ki_due_day, ki_barrier` (`(n_ki,)`), `ki_continuous: bool`, `ki_continuous_barrier: float | None`, `terminal_due_day: int`, `terminal_settlement_day: int`, `is_reverse: bool`, `disable_ko_after_ki: bool`, `dates: pd.DatetimeIndex`.
  - `AutocallableLifecycleTracker.resolve_calendar_schedule(dates: pd.DatetimeIndex, env: PricingEnvironment) -> CalendarSchedule`.

- [ ] **Step 1: Write the failing test**

`test/test_lifecycle_calendar_schedule.py`:

```python
"""The calendar schedule must fire on exactly the tracker's own days."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker, CalendarSchedule
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.asset.equity.product.option import create_standard_snowball
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType

SPOT = 100.0
DATES = pd.DatetimeIndex(pd.date_range("2024-01-02", periods=8, freq="D"))


def _product():
    return create_standard_snowball(
        initial_price=SPOT, strike=SPOT, maturity=6.0 / 365.0, contract_multiplier=1.0,
        ko_barrier=103.0, ki_barrier=97.0, ko_rate=0.02, num_observations=2,
        ko_observation_dates=[2.0 / 365.0, 5.0 / 365.0],
        ki_observation_type=ObservationType.DISCRETE, ki_continuous=False,
        ki_observation_dates=[1.0 / 365.0, 3.0 / 365.0], include_principal=True,
    )


def _env(day: pd.Timestamp) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=SPOT, asset_name="X"), vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.02), valuation_date=pd.Timestamp(day).to_pydatetime(),
    )


def _tracker():
    return AutocallableLifecycleTracker(
        product=_product(), quantity=-1.0, has_lifecycle=True,
        lifecycle=AutocallableLifecycleState(), start_date=DATES[0],
    )


def test_the_schedule_fires_on_the_days_the_tracker_observes():
    schedule = _tracker().resolve_calendar_schedule(DATES, _env(DATES[0]))
    assert isinstance(schedule, CalendarSchedule)

    # Walk a real tracker day by day on a spot that never touches a barrier
    # and record which observation index each day marks as observed.
    tracker = _tracker()
    product = tracker.product_for_lifecycle()
    observed_ko, observed_ki = {}, {}
    for d, day in enumerate(DATES):
        before_ko = set(tracker.lifecycle.observed_ko_indices)
        before_ki = set(tracker.lifecycle.observed_ki_indices)
        tracker.observe(day, product, _env(day), SPOT)      # 100: between 97 and 103
        for idx in sorted(set(tracker.lifecycle.observed_ko_indices) - before_ko):
            observed_ko[idx] = d
        for idx in sorted(set(tracker.lifecycle.observed_ki_indices) - before_ki):
            observed_ki[idx] = d

    assert {int(k): int(v) for k, v in enumerate(schedule.ko_due_day)} == observed_ko
    assert {int(k): int(v) for k, v in enumerate(schedule.ki_due_day)} == observed_ki


def test_the_terminal_day_is_the_day_the_tracker_settles_maturity():
    schedule = _tracker().resolve_calendar_schedule(DATES, _env(DATES[0]))
    tracker = _tracker()
    product = tracker.product_for_lifecycle()
    settled_on = None
    for d, day in enumerate(DATES):
        tracker.observe(day, product, _env(day), SPOT)
        if tracker.settle_maturity_if_due(day, product, _env(day), SPOT) is not None:
            settled_on = d
            break
    assert schedule.terminal_due_day == settled_on
    assert schedule.terminal_settlement_day >= schedule.terminal_due_day


def test_barriers_payoffs_and_flags_come_off_the_resolved_records():
    schedule = _tracker().resolve_calendar_schedule(DATES, _env(DATES[0]))
    assert schedule.ko_barrier == pytest.approx([103.0, 103.0])
    assert np.all(schedule.ko_payoff > 0.0)
    assert schedule.ki_barrier == pytest.approx([97.0, 97.0])
    assert schedule.ki_continuous is False and schedule.is_reverse is False


def test_a_continuous_ki_product_reports_its_single_barrier():
    product = create_standard_snowball(
        initial_price=SPOT, strike=SPOT, maturity=6.0 / 365.0, contract_multiplier=1.0,
        ko_barrier=103.0, ki_barrier=97.0, ko_rate=0.02, num_observations=2,
        ko_observation_dates=[2.0 / 365.0, 5.0 / 365.0],
        ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True, include_principal=True,
    )
    tracker = AutocallableLifecycleTracker(product=product, quantity=-1.0, has_lifecycle=True,
                                           lifecycle=AutocallableLifecycleState(), start_date=DATES[0])
    schedule = tracker.resolve_calendar_schedule(DATES, _env(DATES[0]))
    assert schedule.ki_continuous is True
    assert schedule.ki_continuous_barrier == pytest.approx(97.0)
    assert schedule.ki_due_day.size == 0


def test_resolution_does_not_depend_on_the_day_the_environment_carries():
    early = _tracker().resolve_calendar_schedule(DATES, _env(DATES[0]))
    late = _tracker().resolve_calendar_schedule(DATES, _env(DATES[-1]))
    assert list(early.ko_due_day) == list(late.ko_due_day)
    assert list(early.ki_due_day) == list(late.ki_due_day)
    assert early.terminal_due_day == late.terminal_due_day


def test_an_observation_beyond_the_calendar_is_marked_minus_one():
    short = DATES[:2]
    schedule = _tracker().resolve_calendar_schedule(short, _env(short[0]))
    assert schedule.ko_due_day[-1] == -1
    assert schedule.terminal_due_day == -1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest -n0 test/test_lifecycle_calendar_schedule.py -q`
Expected: `ImportError: cannot import name 'CalendarSchedule'`

- [ ] **Step 3: Write the implementation**

Append to `quantark/asset/equity/lifecycle/autocallable.py` (module level, after the tracker class), and add the method inside `AutocallableLifecycleTracker`:

```python
@dataclass(frozen=True)
class CalendarSchedule:
    """When a product's observations fall on a given run calendar.

    Every ``*_day`` is a position in ``dates``, or ``-1`` when the day lies
    beyond the calendar's end.  Built by
    :meth:`AutocallableLifecycleTracker.resolve_calendar_schedule` from the
    same records and due rule ``observe`` uses, so a caller that fires on
    these days fires exactly when the tracker does.
    """

    dates: pd.DatetimeIndex
    ko_due_day: np.ndarray
    ko_barrier: np.ndarray
    ko_payoff: np.ndarray
    ko_settlement_day: np.ndarray
    ki_due_day: np.ndarray
    ki_barrier: np.ndarray
    ki_continuous: bool
    ki_continuous_barrier: Optional[float]
    terminal_due_day: int
    terminal_settlement_day: int
    is_reverse: bool
    disable_ko_after_ki: bool
```

Method on `AutocallableLifecycleTracker`:

```python
    def resolve_calendar_schedule(
        self, dates: pd.DatetimeIndex, env: PricingEnvironment
    ) -> "CalendarSchedule":
        """Map this product's observations onto ``dates`` (one resolution).

        A vectorised replay needs to know, once, which day each observation
        first becomes due on; walking ``observe`` day by day would answer
        the same question one path at a time.  The records and the due rule
        here are the ones ``observe`` and ``settle_maturity_if_due`` use.
        """
        dates = pd.DatetimeIndex(dates)
        product = self.product

        def first_due(is_due) -> int:
            for d, day in enumerate(dates):
                if is_due(day, self._valuation_point(day, product)):
                    return d
            return -1

        def settlement_position(settlement_date, due_day: int) -> int:
            if due_day < 0:
                return -1
            if settlement_date is None:
                return due_day
            stamp = pd.Timestamp(settlement_date).normalize()
            later = np.flatnonzero(dates >= stamp)
            return int(later[0]) if later.size else -1

        ko_records = self._scheduled_records(product, env, "ko")
        ko_due, ko_barrier, ko_payoff, ko_settle = [], [], [], []
        for rec in ko_records:
            due = first_due(lambda day, vp, rec=rec: self._record_is_due(day, vp, rec))
            ko_due.append(due)
            ko_barrier.append(float(rec["barrier"]) if rec["barrier"] is not None else np.nan)
            ko_payoff.append(float(rec["payoff"]))
            ko_settle.append(settlement_position(rec["settlement_date"], due))

        ki_observation_type = getattr(product.barrier_config, "ki_observation_type", None)
        ki_continuous = bool(
            getattr(product, "has_ki_barrier", False)
            and (
                product.barrier_config.ki_continuous
                or getattr(ki_observation_type, "name", None) == "CONTINUOUS"
            )
        )
        ki_due, ki_barrier = [], []
        continuous_barrier: Optional[float] = None
        if ki_continuous:
            barrier = product.barrier_config.ki_barrier
            if isinstance(barrier, list):
                barrier = barrier[0]
            continuous_barrier = float(barrier)
        else:
            for rec in self._scheduled_records(product, env, "ki"):
                ki_due.append(first_due(lambda day, vp, rec=rec: self._record_is_due(day, vp, rec)))
                ki_barrier.append(float(rec["barrier"]) if rec["barrier"] is not None else np.nan)

        timing = resolve_terminal_timing(product, self._schedule_resolution_env(product, env))
        terminal_due = first_due(
            lambda day, vp: self._timing_is_due(day, vp, timing)
        )
        terminal_settle = settlement_position(
            getattr(timing, "settlement_date", None), terminal_due
        )
        return CalendarSchedule(
            dates=dates,
            ko_due_day=np.array(ko_due, dtype=np.int64),
            ko_barrier=np.array(ko_barrier, dtype=float),
            ko_payoff=np.array(ko_payoff, dtype=float),
            ko_settlement_day=np.array(ko_settle, dtype=np.int64),
            ki_due_day=np.array(ki_due, dtype=np.int64),
            ki_barrier=np.array(ki_barrier, dtype=float),
            ki_continuous=ki_continuous,
            ki_continuous_barrier=continuous_barrier,
            terminal_due_day=int(terminal_due),
            terminal_settlement_day=int(terminal_settle),
            is_reverse=bool(product.is_reverse),
            disable_ko_after_ki=bool(
                getattr(product.barrier_config, "disable_ko_after_ki", False)
            ),
        )
```

Add `import numpy as np` and `from dataclasses import dataclass` to the module's imports if they are not already there (check with `grep -n "^import\|^from" quantark/asset/equity/lifecycle/autocallable.py`), and add `CalendarSchedule` to `quantark/asset/equity/lifecycle/__init__.py`'s imports and `__all__`.

**Terminal settlement.** `resolve_terminal_timing` returns a `ResolvedPaymentTiming` (`quantark.asset.equity.settlement`) carrying both `payment_date: datetime | None` and `payment_time: float`. Use the date when it is there; otherwise derive one exactly the way `_scheduled_records` does for numeric schedules, so the two agree:

```python
        def terminal_payment_date(timing) -> Optional[pd.Timestamp]:
            """The terminal payment day, on the same clock ``_scheduled_records`` uses."""
            if getattr(timing, "payment_date", None) is not None:
                return pd.Timestamp(timing.payment_date).normalize()
            base = getattr(product, "initial_date", None) or self.start_date
            if base is None:
                return None
            payment_time = float(timing.payment_time)
            if payment_time <= float(timing.determination_time):
                return None          # pays at determination
            return self._date_resolver(
                (pd.Timestamp(base) + timedelta(days=int(round(payment_time * 365)))).normalize()
            )
```

and call `settlement_position(terminal_payment_date(timing), terminal_due)`. A `None` payment date means the flow pays on its determination day, which is what `mark_maturity`'s default entry does.

- [ ] **Step 4: Run the test and the lifecycle regression**

Run: `.venv/bin/python -m pytest -n0 test/test_lifecycle_calendar_schedule.py test/replay_golden test/test_replay_dividend_source.py -q`
Expected: all pass, with no edits to the existing tests.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/lifecycle/autocallable.py quantark/asset/equity/lifecycle/__init__.py \
        test/test_lifecycle_calendar_schedule.py
git commit -m "feat(lifecycle): resolve a product's observations onto a run calendar once

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: `VectorLifecycle` — the snowball state machine over paths

**Files:**
- Create: `quantark/backtest/simulation/lifecycle.py`
- Create: `test/simulation/test_lifecycle.py`

**The rules being vectorised**, read off `AutocallableLifecycleTracker.observe`, `settle_maturity_if_due` and `ProductReplay.settle_pending_if_due`, in the order the replay applies them each day:

1. KO is skipped entirely for a path that is knocked in when `disable_ko_after_ki` is set. Otherwise every KO index whose due day is today is **marked observed whether or not it hits** — that is what stops it firing later — and the first index that hits terminates the path. `observe` returns immediately on a KO hit, so no KI is evaluated for that path today.
2. KI: for continuous monitoring, every alive path is tested against the single barrier every day; for discrete, only on an index's due day. `mark_ki` is a no-op for a path already knocked in or out.
3. Maturity runs after KO/KI and only for paths still alive. The payoff is `product.get_payoff(spot, env, knocked_in=...)`, per path.
4. Settlement: a terminal cashflow counts in `realized_cashflows` from the first day on or after its payment date, and sits in `pending_settlement_cashflow` before that. `realized_cashflows` is the ledger's paid total at the day's valuation point, so this is the whole rule.
5. `_barrier_hit`: KO is `spot >= barrier` (`<=` when reverse), KI is `spot <= barrier` (`>=` when reverse). Both inclusive.

**Interfaces:**
- Consumes: `CalendarSchedule` (Task 5), `PricingEnvironment`.
- Produces:
  - `VectorLifecycle(schedules: Sequence[CalendarSchedule], quantities: np.ndarray, n_paths: int)`; arrays of shape `(n_products, n_paths)`: `alive`, `knocked_in`, `knocked_out`, `matured`, `settled`, `pending`, `realized`, `terminal_day` (int, `-1`), `settlement_day` (int, `-1`), `ko_index` (int, `-1`); `observed_ko` of shape `(n_products, n_paths, n_ko)`.
  - `step(day_index, spot, payoff_fn) -> list[LifecycleRecord]` where `payoff_fn(product_index, path_indices, knocked_in) -> np.ndarray` supplies maturity payoffs per unit product; the returned records are `(product, path, day, event, index, spot, barrier, cashflow)` tuples.
  - `book_flags() -> dict` with the replay's aggregation: `alive = any product alive`, `knocked_in = any`, `knocked_out = all`, `matured = all` (per path).

- [ ] **Step 1: Write the failing tests**

`test/simulation/test_lifecycle.py`:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.backtest.simulation.lifecycle import VectorLifecycle
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment

from .conftest import SPOT, short_snowball

DATES = pd.DatetimeIndex(pd.date_range("2024-01-02", periods=8, freq="D"))
KO = 1.03 * SPOT
KI = 0.75 * SPOT


def _env(day) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=SPOT, asset_name="X"), vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.02), valuation_date=pd.Timestamp(day).to_pydatetime(),
    )


def _tracker(product, quantity=-1.0):
    return AutocallableLifecycleTracker(product=product, quantity=quantity, has_lifecycle=True,
                                        lifecycle=AutocallableLifecycleState(), start_date=DATES[0])


def _vector(product, quantity=-1.0, n_paths=1):
    schedule = _tracker(product, quantity).resolve_calendar_schedule(DATES, _env(DATES[0]))
    return VectorLifecycle([schedule], np.array([quantity]), n_paths=n_paths)


def _walk_tracker(product, spots, quantity=-1.0):
    """The tracker's own answer on one hand-built path."""
    tracker = _tracker(product, quantity)
    lifecycle_product = tracker.product_for_lifecycle()
    out = []
    for day, spot in zip(DATES, spots):
        tracker.observe(day, lifecycle_product, _env(day), float(spot))
        tracker.settle_maturity_if_due(day, lifecycle_product, _env(day), float(spot))
        if not tracker.lifecycle.settled and tracker.lifecycle.settlement_date is not None:
            if pd.Timestamp(day).normalize() >= pd.Timestamp(tracker.lifecycle.settlement_date).normalize():
                tracker.lifecycle.settle()
        out.append({
            "alive": tracker.lifecycle.alive, "knocked_in": tracker.lifecycle.knocked_in,
            "knocked_out": tracker.lifecycle.knocked_out, "matured": tracker.lifecycle.matured,
            "settled": tracker.lifecycle.settled,
            "realized": tracker.lifecycle.realized_cashflows,
            "pending": tracker.lifecycle.pending_settlement_cashflow,
        })
    return out


def _walk_vector(product, spots, quantity=-1.0):
    vec = _vector(product, quantity)
    out = []
    for d, spot in enumerate(spots):
        vec.step(d, np.array([float(spot)]),
                 lambda p, paths, ki, spot=spot: np.array([
                     float(product.get_payoff(float(spot), _env(DATES[d]), knocked_in=bool(k))) for k in ki
                 ]))
        out.append({
            "alive": bool(vec.alive[0, 0]), "knocked_in": bool(vec.knocked_in[0, 0]),
            "knocked_out": bool(vec.knocked_out[0, 0]), "matured": bool(vec.matured[0, 0]),
            "settled": bool(vec.settled[0, 0]), "realized": float(vec.realized[0, 0]),
            "pending": float(vec.pending[0, 0]),
        })
    return out


@pytest.mark.parametrize("name, spots", [
    ("never_touches", [SPOT] * 8),
    ("ko_on_the_first_observation", [SPOT, SPOT, KO + 1.0, SPOT, SPOT, SPOT, SPOT, SPOT]),
    ("ko_only_on_a_due_day", [KO + 1.0, KO + 1.0, SPOT, SPOT, SPOT, SPOT, SPOT, SPOT]),
    ("ki_then_maturity", [SPOT, KI - 1.0, SPOT, SPOT, SPOT, SPOT, SPOT * 0.8, SPOT * 0.8]),
    ("ki_and_ko_on_the_same_day", [SPOT, KI - 1.0, KO + 1.0, SPOT, SPOT, SPOT, SPOT, SPOT]),
    ("exactly_on_the_barrier", [SPOT, KI, KO, SPOT, SPOT, SPOT, SPOT, SPOT]),
])
def test_the_vector_lifecycle_matches_the_tracker_day_by_day(name, spots):
    product = short_snowball()
    expected = _walk_tracker(product, spots)
    actual = _walk_vector(product, spots)
    for d, (e, a) in enumerate(zip(expected, actual)):
        assert a == pytest.approx(e), f"{name} day {d}: {a} != {e}"


def test_continuous_ki_fires_on_any_day():
    product = short_snowball(continuous_ki=True)
    spots = [SPOT, SPOT, KI - 1.0, SPOT, SPOT, SPOT, SPOT, SPOT]
    assert _walk_vector(product, spots) == _walk_tracker(product, spots)


def test_disable_ko_after_ki_stops_marking_ko_observations():
    product = short_snowball()
    product.barrier_config.disable_ko_after_ki = True
    spots = [SPOT, KI - 1.0, KO + 1.0, SPOT, SPOT, SPOT, SPOT, SPOT]
    assert _walk_vector(product, spots) == _walk_tracker(product, spots)


def test_paths_are_independent_within_one_batch():
    product = short_snowball()
    vec = _vector(product, n_paths=3)
    lanes = [[SPOT] * 8, [SPOT, SPOT, KO + 1.0] + [SPOT] * 5, [SPOT, KI - 1.0] + [SPOT] * 6]
    for d in range(8):
        spot = np.array([lane[d] for lane in lanes])
        vec.step(d, spot, lambda p, paths, ki: np.zeros(paths.size))
    assert list(vec.knocked_out[0]) == [False, True, False]
    assert list(vec.knocked_in[0]) == [False, False, True]


def test_book_flags_use_the_replay_aggregation():
    product = short_snowball()
    schedules = [_tracker(product).resolve_calendar_schedule(DATES, _env(DATES[0]))] * 2
    vec = VectorLifecycle(schedules, np.array([-1.0, -1.0]), n_paths=1)
    vec.knocked_out[0, 0] = True
    vec.alive[0, 0] = False
    flags = vec.book_flags()
    assert bool(flags["alive"][0])              # ANY product alive: the second one is
    assert not bool(flags["knocked_out"][0])    # ALL products out: only the first one is


def test_records_report_each_event_once():
    product = short_snowball()
    vec = _vector(product, n_paths=1)
    spots = [SPOT, KI - 1.0, KO + 1.0, SPOT, SPOT, SPOT, SPOT, SPOT]
    events = []
    for d, spot in enumerate(spots):
        events += vec.step(d, np.array([spot]), lambda p, paths, ki: np.zeros(paths.size))
    kinds = [e.event for e in events]
    assert kinds.count("knock_in") == 1 and kinds.count("knock_out") == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_lifecycle.py -q`
Expected: `ModuleNotFoundError: No module named 'quantark.backtest.simulation.lifecycle'`

- [ ] **Step 3: Write the implementation**

`quantark/backtest/simulation/lifecycle.py`:

```python
"""The snowball lifecycle over a batch of paths (spec 4.2, 8.4).

Every rule here is ``AutocallableLifecycleTracker``'s, applied to arrays
instead of one product at a time, and the days each observation fires on
come from the tracker itself (``resolve_calendar_schedule``).  Where the
tracker returns early -- a knock-out ends the day's observation before any
knock-in is considered -- this keeps a mask to the same effect.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from quantark.asset.equity.lifecycle.autocallable import CalendarSchedule
from quantark.util.exceptions import ValidationError

PayoffFn = Callable[[int, np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class LifecycleRecord:
    """One event, for the run's event log."""

    product: int
    path: int
    day: int
    event: str            # "knock_in" | "knock_out" | "maturity" | "settlement"
    index: int            # observation index, -1 where there is none
    spot: float
    barrier: float
    cashflow: float


def _barrier_hit(spot: np.ndarray, barrier: float, is_reverse: bool, *, is_ko: bool) -> np.ndarray:
    """``AutocallableLifecycleTracker._barrier_hit``, vectorised (inclusive both ways)."""
    if not np.isfinite(barrier):
        return np.zeros(spot.shape, dtype=bool)
    if is_ko:
        return spot <= barrier if is_reverse else spot >= barrier
    return spot >= barrier if is_reverse else spot <= barrier


class VectorLifecycle:
    """Per-product, per-path lifecycle state advanced one day at a time."""

    def __init__(
        self, schedules: Sequence[CalendarSchedule], quantities: np.ndarray, n_paths: int
    ) -> None:
        if len(schedules) != len(quantities):
            raise ValidationError("one schedule per product is required")
        if n_paths < 1:
            raise ValidationError("n_paths must be positive")
        self.schedules = list(schedules)
        self.quantities = np.asarray(quantities, dtype=float)
        self.n_products = len(self.schedules)
        self.n_paths = int(n_paths)
        shape = (self.n_products, self.n_paths)
        self.alive = np.ones(shape, dtype=bool)
        self.knocked_in = np.zeros(shape, dtype=bool)
        self.knocked_out = np.zeros(shape, dtype=bool)
        self.matured = np.zeros(shape, dtype=bool)
        self.settled = np.zeros(shape, dtype=bool)
        self.pending = np.zeros(shape)
        self.realized = np.zeros(shape)
        self.terminal_day = np.full(shape, -1, dtype=np.int64)
        self.settlement_day = np.full(shape, -1, dtype=np.int64)
        self.ko_index = np.full(shape, -1, dtype=np.int64)
        self.observed_ko = [
            np.zeros((self.n_paths, s.ko_due_day.size), dtype=bool) for s in self.schedules
        ]
        self.observed_ki = [
            np.zeros((self.n_paths, s.ki_due_day.size), dtype=bool) for s in self.schedules
        ]

    def step(self, day_index: int, spot: np.ndarray, payoff_fn: PayoffFn) -> List[LifecycleRecord]:
        """Advance one day; returns the events that fired."""
        spot = np.asarray(spot, dtype=float)
        if spot.shape != (self.n_paths,):
            raise ValidationError(f"spot must have shape {(self.n_paths,)}, got {spot.shape}")
        records: List[LifecycleRecord] = []
        for p, schedule in enumerate(self.schedules):
            # ``today_ko`` is per product: one product knocking out must not
            # silence another product's knock-in on the same path.
            ko_records, today_ko = self._knock_out(p, schedule, day_index, spot)
            records += ko_records
            records += self._knock_in(p, schedule, day_index, spot, today_ko)
            records += self._maturity(p, schedule, day_index, spot, payoff_fn)
            records += self._settle(p, day_index)
        return records

    # -- the day's three blocks, in the tracker's order -----------------

    def _knock_out(self, p, schedule, d, spot) -> Tuple[List[LifecycleRecord], np.ndarray]:
        records: List[LifecycleRecord] = []
        today_ko = np.zeros(self.n_paths, dtype=bool)
        # A knocked-in path skips the KO block entirely under this flag, so
        # its indices are not marked observed either.
        eligible_block = self.alive[p] & ~(
            self.knocked_in[p] if schedule.disable_ko_after_ki else np.zeros(self.n_paths, bool)
        )
        for idx in np.flatnonzero(schedule.ko_due_day == d):
            idx = int(idx)
            due = eligible_block & self.alive[p] & ~self.observed_ko[p][:, idx]
            self.observed_ko[p][due, idx] = True     # observed whether or not it hits
            hit = due & _barrier_hit(spot, float(schedule.ko_barrier[idx]),
                                     schedule.is_reverse, is_ko=True)
            if not hit.any():
                continue
            cashflow = float(self.quantities[p]) * float(schedule.ko_payoff[idx])
            self.knocked_out[p][hit] = True
            self.alive[p][hit] = False
            self.ko_index[p][hit] = idx
            self.terminal_day[p][hit] = d
            settle_day = int(schedule.ko_settlement_day[idx])
            self.settlement_day[p][hit] = settle_day
            self.pending[p][hit] = cashflow
            today_ko |= hit
            for i in np.flatnonzero(hit):
                records.append(LifecycleRecord(p, int(i), d, "knock_out", idx, float(spot[i]),
                                               float(schedule.ko_barrier[idx]), cashflow))
        return records, today_ko

    def _knock_in(self, p, schedule, d, spot, today_ko: np.ndarray) -> List[LifecycleRecord]:
        records: List[LifecycleRecord] = []
        # ``observe`` returns as soon as a knock-out fires, so a path that
        # knocked out today never reaches its knock-in test.
        candidates = self.alive[p] & ~self.knocked_in[p] & ~today_ko
        if schedule.ki_continuous:
            barrier = float(schedule.ki_continuous_barrier)
            hit = candidates & _barrier_hit(spot, barrier, schedule.is_reverse, is_ko=False)
            self.knocked_in[p][hit] = True
            for i in np.flatnonzero(hit):
                records.append(LifecycleRecord(p, int(i), d, "knock_in", -1, float(spot[i]),
                                               barrier, 0.0))
            return records
        for idx in np.flatnonzero(schedule.ki_due_day == d):
            idx = int(idx)
            due = self.alive[p] & ~today_ko & ~self.observed_ki[p][:, idx]
            self.observed_ki[p][due, idx] = True
            barrier = float(schedule.ki_barrier[idx])
            hit = due & ~self.knocked_in[p] & _barrier_hit(spot, barrier, schedule.is_reverse, is_ko=False)
            self.knocked_in[p][hit] = True
            for i in np.flatnonzero(hit):
                records.append(LifecycleRecord(p, int(i), d, "knock_in", idx, float(spot[i]),
                                               barrier, 0.0))
        return records

    def _maturity(self, p, schedule, d, spot, payoff_fn) -> List[LifecycleRecord]:
        if schedule.terminal_due_day != d:
            return []
        due = np.flatnonzero(self.alive[p])
        if due.size == 0:
            return []
        payoffs = np.asarray(payoff_fn(p, due, self.knocked_in[p][due]), dtype=float)
        cashflows = float(self.quantities[p]) * payoffs
        self.matured[p][due] = True
        self.alive[p][due] = False
        self.terminal_day[p][due] = d
        self.settlement_day[p][due] = int(schedule.terminal_settlement_day)
        self.pending[p][due] = cashflows
        return [
            LifecycleRecord(p, int(i), d, "maturity", -1, float(spot[i]), float("nan"), float(c))
            for i, c in zip(due, cashflows)
        ]

    def _settle(self, p, d) -> List[LifecycleRecord]:
        landing = (
            ~self.settled[p] & (self.settlement_day[p] >= 0) & (self.settlement_day[p] <= d)
            & (self.terminal_day[p] >= 0)
        )
        if not landing.any():
            return []
        amounts = self.pending[p][landing]
        self.realized[p][landing] += amounts
        self.pending[p][landing] = 0.0
        self.settled[p][landing] = True
        return [
            LifecycleRecord(p, int(i), d, "settlement", -1, float("nan"), float("nan"), float(a))
            for i, a in zip(np.flatnonzero(landing), amounts)
        ]

    # -- aggregation ---------------------------------------------------

    def book_flags(self) -> Dict[str, np.ndarray]:
        """The replay's book-level reduction over products, per path."""
        return {
            "alive": self.alive.any(axis=0),
            "knocked_in": self.knocked_in.any(axis=0),
            "knocked_out": self.knocked_out.all(axis=0),
            "matured": self.matured.all(axis=0),
            "settled": self.settled.all(axis=0),
        }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_lifecycle.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/lifecycle.py test/simulation/test_lifecycle.py
git commit -m "feat(backtest/simulation): vectorised snowball lifecycle pinned to the tracker

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: `VectorHedgeLedger` — average-cost futures accounting over paths

**Files:**
- Create: `quantark/backtest/simulation/hedge.py`
- Create: `test/simulation/test_hedge.py`

**The rule being vectorised** is `FuturesHedgePosition.trade`, branch for branch: a trade below `1e-12` in size is ignored; opening from flat sets the average price; adding in the same direction takes the weighted average of absolute sizes; trading against the position realises `direction · min(|qty|, |delta|) · (price − avg) · multiplier`, and a flip resets the average to the trade price. `mark_to_market` returns `realized_pnl` alone when the position is flat. Prices differ per path (they come from each path's own curve) while the contract is common, which is exactly the shape the ledger needs.

**Interfaces:**
- Produces:
  - `VectorHedgeLedger(n_paths: int)` with arrays `quantity`, `avg_price`, `realized_pnl` (all `(n_paths,)`), scalars `contract: str | None`, `multiplier: float`.
  - `trade(quantity_delta: np.ndarray, price: np.ndarray, contract: str, multiplier: float) -> None`.
  - `mark_to_market(price: np.ndarray) -> np.ndarray`.
  - `close_all(price, contract, multiplier) -> np.ndarray` returning the traded sizes.

- [ ] **Step 1: Write the failing tests**

`test/simulation/test_hedge.py` (Task 8 appends to this file):

```python
from __future__ import annotations

import numpy as np
import pytest

from quantark.backtest.futures_ledger import FuturesHedgePosition
from quantark.backtest.simulation.hedge import VectorHedgeLedger
from quantark.util.exceptions import ValidationError

MULT = 200.0


def _scalar_walk(trades):
    """The same trade sequence through the scalar ledger."""
    pos = FuturesHedgePosition(multiplier=MULT)
    for qty, price, contract in trades:
        pos.trade(qty, price, contract, MULT)
    return pos


def test_a_random_trade_sequence_matches_the_scalar_ledger():
    rng = np.random.default_rng(7)
    n_paths = 6
    ledger = VectorHedgeLedger(n_paths)
    scalars = [FuturesHedgePosition(multiplier=MULT) for _ in range(n_paths)]
    for _ in range(40):
        qty = np.round(rng.normal(0.0, 4.0, n_paths))
        price = 6000.0 * np.exp(rng.normal(0.0, 0.01, n_paths))
        ledger.trade(qty, price, "IM2403", MULT)
        for i, pos in enumerate(scalars):
            pos.trade(float(qty[i]), float(price[i]), "IM2403", MULT)
    assert ledger.quantity == pytest.approx([p.quantity for p in scalars])
    assert ledger.avg_price == pytest.approx([p.avg_price for p in scalars])
    assert ledger.realized_pnl == pytest.approx([p.realized_pnl for p in scalars])
    mark = 6100.0
    assert ledger.mark_to_market(np.full(n_paths, mark)) == pytest.approx(
        [p.mark_to_market(mark) for p in scalars]
    )


@pytest.mark.parametrize("sequence", [
    [(3.0, 100.0), (2.0, 110.0)],                      # add in the same direction
    [(3.0, 100.0), (-1.0, 110.0)],                     # partial close
    [(3.0, 100.0), (-3.0, 110.0)],                     # full close, back to flat
    [(3.0, 100.0), (-5.0, 110.0)],                     # flip through zero
    [(-4.0, 100.0), (-2.0, 90.0), (6.0, 95.0)],        # short, add, close
    [(0.0, 100.0), (2.0, 100.0)],                      # a zero trade is ignored
])
def test_each_ledger_branch_matches(sequence):
    ledger = VectorHedgeLedger(1)
    pos = FuturesHedgePosition(multiplier=MULT)
    for qty, price in sequence:
        ledger.trade(np.array([qty]), np.array([price]), "IM2403", MULT)
        pos.trade(qty, price, "IM2403", MULT)
    assert ledger.quantity[0] == pytest.approx(pos.quantity)
    assert ledger.avg_price[0] == pytest.approx(pos.avg_price)
    assert ledger.realized_pnl[0] == pytest.approx(pos.realized_pnl)
    assert ledger.contract == pos.contract


def test_a_flat_position_marks_at_its_realised_pnl_only():
    ledger = VectorHedgeLedger(2)
    ledger.trade(np.array([2.0, 0.0]), np.array([100.0, 100.0]), "IM2403", MULT)
    ledger.trade(np.array([-2.0, 0.0]), np.array([110.0, 110.0]), "IM2403", MULT)
    assert ledger.mark_to_market(np.array([999.0, 999.0])) == pytest.approx([2 * 10 * MULT, 0.0])


def test_paths_can_hold_different_sizes_of_the_same_contract():
    ledger = VectorHedgeLedger(3)
    ledger.trade(np.array([1.0, -2.0, 0.0]), np.array([100.0, 101.0, 102.0]), "IM2403", MULT)
    assert list(ledger.quantity) == [1.0, -2.0, 0.0]
    assert ledger.avg_price == pytest.approx([100.0, 101.0, 0.0])


def test_close_all_returns_the_trades_it_made():
    ledger = VectorHedgeLedger(2)
    ledger.trade(np.array([3.0, -1.0]), np.array([100.0, 100.0]), "IM2403", MULT)
    traded = ledger.close_all(np.array([110.0, 110.0]), "IM2403", MULT)
    assert traded == pytest.approx([-3.0, 1.0])
    assert list(ledger.quantity) == [0.0, 0.0]


def test_trading_a_different_contract_without_a_roll_fails_closed():
    ledger = VectorHedgeLedger(1)
    ledger.trade(np.array([1.0]), np.array([100.0]), "IM2403", MULT)
    with pytest.raises(ValidationError):
        ledger.trade(np.array([1.0]), np.array([100.0]), "IM2406", MULT)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_hedge.py -q`
Expected: `ModuleNotFoundError: No module named 'quantark.backtest.simulation.hedge'`

- [ ] **Step 3: Write the implementation**

`quantark/backtest/simulation/hedge.py`:

```python
"""The futures hedge over a batch of paths (spec 4.2, 8.1, 8.6).

``FuturesHedgePosition``'s average-cost rules, applied to arrays: every
path holds the same contract (the roll policy reads only expiry dates,
which are common) in its own size, at its own price.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from quantark.util.exceptions import ValidationError

#: The replay's own dead-zone for "no position" and "no trade".  Copied
#: verbatim from ``FuturesHedgePosition`` and ``ReplayBacktestEngine._rebalance``
#: rather than routed through ``util.numerical``: the goal is a ledger that
#: agrees with the replay bit for bit, so it must share the exact threshold.
POSITION_EPS = 1e-12


class VectorHedgeLedger:
    """Average-cost futures ledger, one lane per path."""

    def __init__(self, n_paths: int) -> None:
        if n_paths < 1:
            raise ValidationError("n_paths must be positive")
        self.n_paths = int(n_paths)
        self.quantity = np.zeros(self.n_paths)
        self.avg_price = np.zeros(self.n_paths)
        self.realized_pnl = np.zeros(self.n_paths)
        self.contract: Optional[str] = None
        self.multiplier = 1.0

    def trade(
        self, quantity_delta: np.ndarray, price: np.ndarray, contract: str, multiplier: float
    ) -> None:
        """Book a per-path trade in a common contract."""
        delta = np.asarray(quantity_delta, dtype=float)
        px = np.asarray(price, dtype=float)
        if delta.shape != (self.n_paths,) or px.shape != (self.n_paths,):
            raise ValidationError(f"trade arrays must have shape {(self.n_paths,)}")
        active = np.abs(delta) >= POSITION_EPS
        if not active.any():
            return
        flat = np.abs(self.quantity) < POSITION_EPS
        if self.contract is not None and contract != self.contract and (~flat & active).any():
            raise ValidationError("Cannot trade a different contract without rolling")

        opening = active & flat
        self.quantity[opening] = delta[opening]
        self.avg_price[opening] = px[opening]

        held = active & ~flat
        same = held & (self.quantity * delta > 0.0)
        new_qty = self.quantity[same] + delta[same]
        self.avg_price[same] = (
            self.avg_price[same] * np.abs(self.quantity[same]) + px[same] * np.abs(delta[same])
        ) / np.abs(new_qty)
        self.quantity[same] = new_qty

        against = held & ~same
        if against.any():
            close_qty = np.minimum(np.abs(self.quantity[against]), np.abs(delta[against]))
            direction = np.where(self.quantity[against] > 0.0, 1.0, -1.0)
            self.realized_pnl[against] += (
                direction * close_qty * (px[against] - self.avg_price[against]) * multiplier
            )
            after = self.quantity[against] + delta[against]
            flipped = self.quantity[against] * after < 0.0
            closed = np.abs(after) < POSITION_EPS
            avg = self.avg_price[against]
            avg = np.where(flipped, px[against], avg)
            avg = np.where(closed, 0.0, avg)
            self.avg_price[against] = avg
            self.quantity[against] = np.where(closed, 0.0, after)

        self.multiplier = float(multiplier)
        # The scalar ledger drops the contract when a lane goes flat; with
        # one contract shared by every lane, the batch keeps it until every
        # lane is flat, which is the same statement for the book.
        self.contract = None if np.all(np.abs(self.quantity) < POSITION_EPS) else contract

    def mark_to_market(self, price: np.ndarray) -> np.ndarray:
        """Realised P&L plus the open position's mark, per path."""
        px = np.asarray(price, dtype=float)
        open_ = np.abs(self.quantity) >= POSITION_EPS
        return self.realized_pnl + np.where(
            open_, self.quantity * (px - self.avg_price) * self.multiplier, 0.0
        )

    def close_all(self, price: np.ndarray, contract: str, multiplier: float) -> np.ndarray:
        """Flatten every lane; returns the sizes traded."""
        traded = -self.quantity.copy()
        self.trade(traded, price, contract, multiplier)
        return traded
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_hedge.py -q`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/hedge.py test/simulation/test_hedge.py
git commit -m "feat(backtest/simulation): vectorised average-cost futures ledger

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Roll selection and the delta target, vectorised

**Files:**
- Modify: `quantark/backtest/simulation/hedge.py` (append)
- Modify: `test/simulation/test_hedge.py` (append)

**Why the contract is common to every path.** `FuturesRollPolicy.select_contract` filters on `expiry_date` and sorts on `("expiry_date", "contract")`; it never looks at `futures_price`. Every path in a batch shares the listing calendar, so the selection is one contract per day for the whole batch, and only the price it trades at differs. The first test below pins that by running the real policy on two paths' frames.

**Interfaces:**
- Consumes: `DayChain`, `day_chain` (plan 1), `FuturesRollPolicy`, `AutocallableDeltaHedgeStrategy`, `VectorHedgeLedger` (Task 7).
- Produces:
  - `day_active_contract(chain: DayChain, roll_policy, current_contract: str | None) -> tuple[str, int]` — the active contract code and its column in the chain.
  - `target_contracts_vector(strategy, net_delta: np.ndarray, multiplier: float) -> np.ndarray` — `AutocallableDeltaHedgeStrategy.target_contracts` applied to an array, including `round_contracts`.
  - `should_rebalance_vector(strategy, current: np.ndarray, target: np.ndarray) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

Append to `test/simulation/test_hedge.py`:

```python
from datetime import date

import pandas as pd

from quantark.backtest.futures_ledger import FuturesRollPolicy
from quantark.backtest.simulation.carry import day_chain
from quantark.backtest.simulation.hedge import (
    day_active_contract,
    should_rebalance_vector,
    target_contracts_vector,
)
from quantark.backtest.strategy.futures_delta_strategy import AutocallableDeltaHedgeStrategy

from .conftest import make_market_path


def test_the_roll_policy_picks_the_same_contract_for_every_path():
    mp = make_market_path(n_paths=3, n_days=30, start=date(2024, 1, 2))
    policy = FuturesRollPolicy()
    for d in range(mp.n_days):
        chain = day_chain(mp, d)
        picked = {str(policy.select_contract(chain.frame(i), chain.date, None)["contract"])
                  for i in range(mp.n_paths)}
        assert len(picked) == 1
        code, column = day_active_contract(chain, policy, None)
        assert code == picked.pop() and chain.contracts[column] == code


def test_the_active_contract_is_sticky_until_the_roll_window():
    mp = make_market_path(n_paths=1, n_days=30, start=date(2024, 1, 2))
    policy = FuturesRollPolicy(roll_days_before_expiry=5)
    chain = day_chain(mp, 0)
    assert day_active_contract(chain, policy, None)[0] == "IM2401"
    # inside the window the front contract is dropped
    late = day_chain(mp, list(mp.dates).index(pd.Timestamp("2024-01-16")))
    assert day_active_contract(late, policy, "IM2401")[0] == "IM2402"


@pytest.mark.parametrize("round_contracts", [True, False])
@pytest.mark.parametrize("hedge_ratio, target_delta", [(1.0, 0.0), (0.8, 3.0)])
def test_the_target_matches_the_scalar_strategy(round_contracts, hedge_ratio, target_delta):
    strategy = AutocallableDeltaHedgeStrategy(delta_threshold=0.5, hedge_ratio=hedge_ratio,
                                              target_delta=target_delta, round_contracts=round_contracts)
    net_delta = np.array([-1234.5, 0.0, 987.6, -1.0])
    got = target_contracts_vector(strategy, net_delta, MULT)
    expected = [strategy.target_contracts(product_delta=float(x), product_quantity=1.0,
                                          futures_multiplier=MULT) for x in net_delta]
    assert got == pytest.approx(expected)


def test_the_threshold_matches_the_scalar_strategy():
    strategy = AutocallableDeltaHedgeStrategy(delta_threshold=2.0, hedge_ratio=1.0, target_delta=0.0)
    current = np.array([0.0, 5.0, 5.0])
    target = np.array([1.0, 8.0, 5.5])
    got = should_rebalance_vector(strategy, current, target)
    expected = [strategy.should_rebalance(float(c), float(t)) for c, t in zip(current, target)]
    assert list(got) == expected
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_hedge.py -q -k "roll or target or threshold"`
Expected: `ImportError: cannot import name 'day_active_contract'`

- [ ] **Step 3: Write the implementation**

Append to `quantark/backtest/simulation/hedge.py`:

```python
def day_active_contract(chain, roll_policy, current_contract: Optional[str]):
    """The day's hedge contract and its column in ``chain``.

    ``FuturesRollPolicy.select_contract`` reads only ``expiry_date`` and the
    contract code, never a price, and every path shares the listing
    calendar -- so one selection serves the whole batch and only the
    execution price differs per path.  Path 0's frame is therefore
    representative, and the roll test pins that.
    """
    selected = roll_policy.select_contract(chain.frame(0), chain.date, current_contract)
    code = str(selected["contract"])
    return code, chain.contracts.index(code)


def target_contracts_vector(strategy, net_delta: np.ndarray, multiplier: float) -> np.ndarray:
    """``AutocallableDeltaHedgeStrategy.target_contracts`` over an array."""
    if multiplier <= 0:
        raise ValidationError("futures_multiplier must be positive")
    target = -((np.asarray(net_delta, dtype=float) - float(strategy.target_delta)) / float(multiplier))
    target = target * float(strategy.hedge_ratio)
    if getattr(strategy, "round_contracts", True):
        # np.round is half-to-even, like Python's round(), which the scalar
        # strategy uses; a half-up rule would differ on exact halves.
        return np.round(target)
    return target


def should_rebalance_vector(strategy, current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """``strategy.should_rebalance`` over arrays."""
    return np.abs(np.asarray(target, dtype=float) - np.asarray(current, dtype=float)) > float(
        strategy.delta_threshold
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_hedge.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/hedge.py test/simulation/test_hedge.py
git commit -m "feat(backtest/simulation): vectorised roll selection and delta target

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: `EnsembleBacktestEngine` — the daily loop and the state cube

**Files:**
- Create: `quantark/backtest/simulation/engine.py`
- Create: `test/simulation/test_engine.py`

**One fact this task turns into a dedup, pinned by a test:** the day's dividend object and basis yield do **not** depend on spot. The chain is `F_j = S · exp(B_j)`, and every consumer takes a ratio: the term curve inverts `q_j = r − ln(F_j/S)/T_j = r − B_j/T_j`, the legacy channel takes `max(0, r − (F/S − 1)/T)`, and the basis is `(F/S − 1)/T`. All three see only `(rate, carry row)`. That tuple is `env_key` from Task 2, so the engine builds one dividend object and one basis per **distinct `env_key`** per day, not one per path. Under a generator that holds the carry curve fixed, that is one object a day for the whole batch.

**The order inside a day**, copied from `ReplayBacktestEngine.run`:

1. Day's chain; roll policy picks the active contract; if it changed, close the old lane and open the new one at each path's own price, both legs costed.
2. Dividends, basis and `q_T` per distinct `env_key`.
3. **Day 0 only, before lifecycle:** the initial book value per path (`initial_price` when the `ReplayProduct` sets one, else the day-0 PV), and it is priced with `knocked_in=False` because no event has fired yet.
4. Lifecycle: knock-out, knock-in, maturity, settlement.
5. Price the paths still alive; dead paths carry `pv = delta = gamma = 0`.
6. Rebalance: target from the strategy when any product is alive on that path, otherwise close to zero as `hedge_close`.
7. Record the day's cube row.

**Interfaces:**
- Consumes: everything from Tasks 1–8, `day_chain`/`DayChain` and `dividend_yield_for_day` (plan 1), `MarketPath`.
- Produces:
  - `StateCube(n_paths, n_days)` — float arrays `portfolio_value, product_mtm, hedge_mtm, cash, cashflows, transaction_costs, product_pnl, hedge_pnl, total_pnl, spot, volatility, rate, pricing_q, implied_q, basis_yield, futures_price, futures_contracts, pre_hedge_contracts, delta, gamma, pending_receivable_pv`; bool arrays `alive, knocked_in, knocked_out, matured, settled`; per-day `active_contract: list[str]`, `dates`; `frame(i, last_day) -> pd.DataFrame`.
  - `EnsembleResults(cube, trades, events, manifest, last_day)`; `path_states(i)`, `path_trades(i)`, `path_events(i)`, `n_paths`.
  - `EnsembleBacktestEngine(config: EnsembleConfig)`; `run(paths: MarketPath) -> EnsembleResults`.

- [ ] **Step 1: Write the failing tests**

`test/simulation/test_engine.py`:

```python
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.replay import ReplayProduct
from quantark.backtest.simulation.engine import EnsembleBacktestEngine, EnsembleResults
from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID, StartState, trading_calendar
from quantark.backtest.simulation.paths.gbm import ConstantVol, GBMPaths
from quantark.util.exceptions import ValidationError

from .conftest import RATE, SPOT, VOL, ensemble_config, flat_carry, short_snowball

START = date(2024, 1, 2)


def _paths(n_paths: int = 4, n_days: int = 8, sigma: float = 0.25):
    start = StartState(spot=SPOT, atm_vol=VOL, rate=RATE, carry=flat_carry(DEFAULT_TENOR_GRID))
    gen = GBMPaths(start=start, calendar=trading_calendar(START, n_days), mu=0.0, sigma=sigma,
                   vol_rule=ConstantVol(VOL), carry_schedule=None, rate=RATE,
                   tenor_grid=DEFAULT_TENOR_GRID)
    return gen.generate(n_paths, n_days, seed=3)


def test_the_run_produces_a_cube_of_the_right_shape():
    paths = _paths()
    results = EnsembleBacktestEngine(ensemble_config()).run(paths)
    assert isinstance(results, EnsembleResults)
    assert results.n_paths == paths.n_paths
    assert results.cube.total_pnl.shape == (paths.n_paths, paths.n_days)
    assert len(results.cube.active_contract) == paths.n_days
    frame = results.path_states(0)
    assert {"date", "portfolio_value", "total_pnl", "futures_contracts", "alive"} <= set(frame.columns)


def test_the_accounting_identity_holds_on_every_day_of_every_path():
    results = EnsembleBacktestEngine(ensemble_config()).run(_paths())
    cube = results.cube
    identity = cube.portfolio_value - results.initial_book_value[:, None]
    assert cube.total_pnl == pytest.approx(identity, abs=1e-9)


def test_portfolio_value_is_its_own_parts():
    results = EnsembleBacktestEngine(ensemble_config()).run(_paths())
    cube = results.cube
    assert cube.portfolio_value == pytest.approx(
        cube.product_mtm + cube.hedge_mtm + cube.cash + cube.pending_receivable_pv, abs=1e-9
    )
    assert cube.cash == pytest.approx(cube.cashflows - cube.transaction_costs, abs=1e-9)


def test_the_dividend_object_is_built_once_per_distinct_env_key():
    # a held carry curve and a flat rate: every path shares one env_key a day,
    # so the build count is the number of days the loop actually executed
    results = EnsembleBacktestEngine(ensemble_config()).run(_paths(n_paths=6))
    assert results.manifest["dividend_builds"] == results.manifest["days_run"]
    assert results.manifest["days_run"] >= 1


def test_a_dead_path_carries_no_product_mark_and_no_hedge():
    # A deep crash knocks in but never knocks out; a strong rally knocks out.
    paths = _paths(n_paths=8, sigma=1.2)
    results = EnsembleBacktestEngine(ensemble_config()).run(paths)
    dead = ~results.cube.alive
    assert results.cube.product_mtm[dead] == pytest.approx(0.0)
    assert results.cube.delta[dead] == pytest.approx(0.0)
    assert results.cube.futures_contracts[dead] == pytest.approx(0.0)


def test_the_hedge_targets_the_book_delta_each_day():
    cfg = ensemble_config()
    results = EnsembleBacktestEngine(cfg).run(_paths())
    cube = results.cube
    alive = cube.alive
    expected = np.round(-cube.delta / cfg.hedge.multiplier)
    assert cube.futures_contracts[alive] == pytest.approx(expected[alive])


def test_a_path_stops_updating_once_it_settles():
    paths = _paths(n_paths=6, sigma=1.2)
    results = EnsembleBacktestEngine(ensemble_config()).run(paths)
    for i in range(results.n_paths):
        last = int(results.last_day[i])
        if last < paths.n_days - 1:
            assert results.cube.total_pnl[i, last:] == pytest.approx(results.cube.total_pnl[i, last])
        assert len(results.path_states(i)) == last + 1


def test_an_initial_price_on_the_product_replaces_the_day_zero_mark():
    cfg = ensemble_config(products=[
        ReplayProduct(product=short_snowball(), quantity=-1.0, position_id=1,
                      has_lifecycle=True, initial_price=0.0)
    ])
    results = EnsembleBacktestEngine(cfg).run(_paths())
    assert results.initial_book_value == pytest.approx(0.0)
    assert results.cube.total_pnl[:, 0] == pytest.approx(results.cube.portfolio_value[:, 0], abs=1e-9)


def test_a_calendar_too_short_for_the_product_fails_closed():
    with pytest.raises(ValidationError):
        EnsembleBacktestEngine(ensemble_config()).run(_paths(n_days=3))


def test_a_short_calendar_is_allowed_when_data_end_is_declared():
    results = EnsembleBacktestEngine(ensemble_config(allow_data_end=True)).run(_paths(n_days=3))
    assert results.manifest["data_end_paths"] == results.n_paths


def test_the_manifest_records_the_run():
    results = EnsembleBacktestEngine(ensemble_config()).run(_paths())
    manifest = results.manifest
    assert manifest["path_fingerprint"] and manifest["engine_fingerprint"]
    assert manifest["gate"]["mode"] == "exact"
    assert manifest["cache"]["hits"] >= 0 and manifest["engine_calls"] > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_engine.py -q`
Expected: `ModuleNotFoundError: No module named 'quantark.backtest.simulation.engine'`

- [ ] **Step 3: Write the module skeleton and the run loop**

`quantark/backtest/simulation/engine.py`:

```python
"""The vectorised daily loop (spec 4.3, 8).

One pass over the calendar, every per-path quantity an array.  The order of
operations inside a day, the accounting identity and the hedge rule are
``ReplayBacktestEngine``'s; the conformance oracle checks that claim on a
real path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quantark.asset.equity.lifecycle import AutocallableLifecycleTracker
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.backtest.replay.market import ImpliedBasisYield
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError

from .carry import day_chain
from .config import EnsembleConfig
from .dividends import dividend_yield_for_day
from .hedge import (
    POSITION_EPS,
    VectorHedgeLedger,
    day_active_contract,
    should_rebalance_vector,
    target_contracts_vector,
)
from .lifecycle import LifecycleRecord, VectorLifecycle
from .paths.market_path import MarketPath
from .pricing.base import DayStates, row_keys
from .pricing.cache import StateCache
from .pricing.repricing import RepricingPricer

FLOAT_COLUMNS = (
    "portfolio_value", "product_mtm", "hedge_mtm", "cash", "cashflows", "transaction_costs",
    "product_pnl", "hedge_pnl", "total_pnl", "spot", "volatility", "rate", "pricing_q",
    "implied_q", "basis_yield", "futures_price", "futures_contracts", "pre_hedge_contracts",
    "delta", "gamma", "pending_receivable_pv",
)
BOOL_COLUMNS = ("alive", "knocked_in", "knocked_out", "matured", "settled")

TRADE_COLUMNS = [
    "path", "day", "date", "trade_type", "contract", "quantity", "price",
    "multiplier", "notional", "transaction_cost", "reason",
]


def empty_states(day_index: int, date) -> DayStates:
    """A ``DayStates`` with no paths, for reports that need the shape only."""
    nothing = np.array([])
    return DayStates(
        day_index=day_index, date=date, path_index=np.array([], dtype=int), spot=nothing,
        vol=nothing, rate=nothing, q_T=nothing, div_yield=(), basis_yield=nothing,
        env_key=np.array([], dtype=np.int64), knocked_in=np.array([], dtype=bool),
    )


class StateCube:
    """The replay's state columns that have a per-path meaning (spec 4.3)."""

    def __init__(self, dates: pd.DatetimeIndex, n_paths: int) -> None:
        self.dates = pd.DatetimeIndex(dates)
        self.n_paths = int(n_paths)
        self.active_contract: List[str] = []
        shape = (self.n_paths, len(self.dates))
        for name in FLOAT_COLUMNS:
            setattr(self, name, np.zeros(shape))
        for name in BOOL_COLUMNS:
            setattr(self, name, np.zeros(shape, dtype=bool))

    def frame(self, i: int, last_day: int) -> pd.DataFrame:
        """One path's state rows up to and including ``last_day``."""
        end = int(last_day) + 1
        data: Dict[str, Any] = {
            "date": self.dates[:end],
            "active_contract": self.active_contract[:end],
        }
        for name in FLOAT_COLUMNS + BOOL_COLUMNS:
            data[name] = getattr(self, name)[i, :end]
        return pd.DataFrame(data)

    def freeze_from(self, day: int) -> None:
        """Repeat day ``day``'s values over the rest of the calendar.

        The loop stops once every path has settled; a settled path's columns
        repeat its terminal values, which is the per-path reading of the
        replay's ``terminate_on_lifecycle_end``.
        """
        for name in FLOAT_COLUMNS + BOOL_COLUMNS:
            column = getattr(self, name)
            column[:, day + 1:] = column[:, day: day + 1]
        if self.active_contract:
            self.active_contract += [self.active_contract[-1]] * (
                len(self.dates) - len(self.active_contract)
            )


@dataclass
class EnsembleResults:
    """The run's cube, event logs and manifest (distributions land in plan 4)."""

    cube: StateCube
    trades: List[Dict[str, Any]]
    events: List[LifecycleRecord]
    manifest: Dict[str, Any]
    last_day: np.ndarray
    initial_book_value: np.ndarray

    @property
    def n_paths(self) -> int:
        return int(self.cube.n_paths)

    def path_states(self, i: int) -> pd.DataFrame:
        return self.cube.frame(i, int(self.last_day[i]))

    def path_trades(self, i: int) -> pd.DataFrame:
        rows = [t for t in self.trades if t["path"] == i]
        return pd.DataFrame(rows, columns=TRADE_COLUMNS)

    def path_events(self, i: int) -> pd.DataFrame:
        rows = [e.__dict__ for e in self.events if e.path == i]
        return pd.DataFrame(
            rows,
            columns=["product", "path", "day", "event", "index", "spot", "barrier", "cashflow"],
        )


class EnsembleBacktestEngine:
    """Runs a book of snowballs over a batch of simulated paths."""

    def __init__(self, config: EnsembleConfig) -> None:
        self.config = config
        self.quantities = config.quantities

    def run(self, paths: MarketPath) -> EnsembleResults:
        cfg = self.config
        started = time.perf_counter()
        n_paths, n_days = paths.n_paths, paths.n_days
        dates = paths.dates
        multiplier = float(cfg.hedge.multiplier)

        cache = StateCache(cfg.pricing.cache)
        pricers = [
            RepricingPricer(
                bp.product, engine_config=cfg.engine_config, start_date=dates[0],
                underlying=cfg.underlying, cache=cache,
                delta_bump_size=cfg.delta_bump_size, gamma_bump_size=cfg.gamma_bump_size,
            )
            for bp in cfg.products
        ]
        schedule_env = self._schedule_env(dates[0])
        schedules = [
            self._tracker(bp, dates[0]).resolve_calendar_schedule(dates, schedule_env)
            for bp in cfg.products
        ]
        self._check_calendar(schedules, n_days)

        lifecycle = VectorLifecycle(schedules, self.quantities, n_paths=n_paths)
        ledger = VectorHedgeLedger(n_paths)
        cube = StateCube(dates, n_paths)
        trades: List[Dict[str, Any]] = []
        events: List[LifecycleRecord] = []
        costs = np.zeros(n_paths)
        initial_book_value: Optional[np.ndarray] = None
        settled_day = np.full(n_paths, -1, dtype=np.int64)
        current_contract: Optional[str] = None
        dividend_builds = 0
        last_executed = n_days - 1

        for d in range(n_days):
            day = pd.Timestamp(dates[d])
            spot, vol, rate = paths.spot[:, d], paths.atm_vol[:, d], paths.rate[:, d]

            chain = day_chain(paths, d, multiplier=multiplier)
            code, column = day_active_contract(chain, cfg.hedge.roll_policy, current_contract)
            futures_price = chain.prices[:, column]
            if current_contract is not None and code != current_contract:
                self._roll(ledger, chain, code, column, current_contract, d, day, trades, costs)
            current_contract = code
            cube.active_contract.append(code)

            env_key, div_yield, basis, implied_q, builds = self._day_market(
                paths, d, chain, column, code, rate
            )
            dividend_builds += builds
            q_T = self._pricing_q(pricers[0], day, div_yield, implied_q)

            if initial_book_value is None:
                initial_book_value = self._initial_book_value(
                    pricers, day, d, spot, vol, rate, q_T, div_yield, basis, env_key
                )

            events += lifecycle.step(
                d, spot,
                lambda p, idx, ki, day=day, spot=spot, vol=vol, rate=rate,
                       div_yield=div_yield, basis=basis: self._payoffs(
                    pricers[p], day, idx, ki, spot, vol, rate, div_yield, basis
                ),
            )

            product_mtm, net_delta, net_gamma = self._price_book(
                pricers, lifecycle, day, d, spot, vol, rate, q_T, div_yield, basis, env_key
            )
            receivable = self._receivable_pv(lifecycle, rate, d, schedules)
            cashflows = lifecycle.realized.sum(axis=0)
            flags = lifecycle.book_flags()

            pre_hedge = ledger.quantity.copy()
            self._rebalance(ledger, net_delta, flags["alive"], futures_price, code,
                            multiplier, d, day, trades, costs)

            hedge_mtm = ledger.mark_to_market(futures_price)
            product_pnl = product_mtm + receivable + cashflows - initial_book_value
            total_pnl = product_pnl + hedge_mtm - costs
            cash = cashflows - costs
            portfolio_value = product_mtm + hedge_mtm + cash + receivable

            self._record(
                cube, d, spot=spot, vol=vol, rate=rate, q_T=q_T, implied_q=implied_q, basis=basis,
                futures_price=futures_price, pre_hedge=pre_hedge, contracts=ledger.quantity,
                product_mtm=product_mtm, receivable=receivable, cashflows=cashflows, costs=costs,
                product_pnl=product_pnl, hedge_mtm=hedge_mtm, total_pnl=total_pnl, cash=cash,
                portfolio_value=portfolio_value, delta=net_delta, gamma=net_gamma, flags=flags,
            )

            newly = flags["settled"] & (settled_day < 0)
            settled_day[newly] = d
            last_executed = d
            if flags["settled"].all():
                cube.freeze_from(d)
                break

        last_day = np.where(settled_day >= 0, settled_day, last_executed)
        manifest = self._manifest(paths, pricers, cache, dividend_builds, lifecycle, started,
                                  days_run=last_executed + 1)
        return EnsembleResults(cube=cube, trades=trades, events=events, manifest=manifest,
                               last_day=last_day, initial_book_value=initial_book_value)
```

- [ ] **Step 4: Write the setup and market helpers**

Append to the class:

```python
    # -- setup ---------------------------------------------------------

    def _tracker(self, bp, start_date) -> AutocallableLifecycleTracker:
        return AutocallableLifecycleTracker(
            product=bp.product, quantity=float(bp.quantity), has_lifecycle=True,
            lifecycle=AutocallableLifecycleState(), start_date=start_date,
        )

    def _schedule_env(self, day) -> PricingEnvironment:
        """A throwaway environment carrying only the date, for schedule work."""
        return PricingEnvironment(
            spot_quote=SpotQuote(spot=1.0, asset_name=self.config.underlying),
            vol_surface=FlatVolSurface(volatility=0.2), rate_curve=FlatRateCurve(rate=0.0),
            valuation_date=pd.Timestamp(day).to_pydatetime(),
        )

    def _check_calendar(self, schedules, n_days: int) -> None:
        short = [s for s in schedules if s.terminal_settlement_day < 0]
        if short and not self.config.allow_data_end:
            raise ValidationError(
                f"the path calendar of {n_days} days ends before {len(short)} product(s) "
                "settle; extend the paths or set allow_data_end=True"
            )

    # -- the day's market ----------------------------------------------

    def _day_market(self, paths, d, chain, column, code, rate):
        """Dividend object, basis and legacy scalar yield per distinct state.

        None of the three depends on spot.  The chain is ``F_j = S exp(B_j)``
        and every consumer takes the ratio ``F_j / S``: the term curve
        inverts ``q_j = r - B_j / T_j``, the legacy channel takes
        ``max(0, r - (F/S - 1)/T)`` and the basis is ``(F/S - 1)/T``.  So
        ``(rate, carry row)`` -- the ``env_key`` the cache keys on -- fixes
        all of them, and one build serves every path that shares it.
        """
        n_paths = paths.n_paths
        env_key = row_keys(np.column_stack([rate, paths.carry[:, d, :]]))
        div_yield: List[Any] = [None] * n_paths
        basis = np.zeros(n_paths)
        implied_q = np.zeros(n_paths)
        tenor = float(chain.tenors[column])
        if tenor <= 0.0:
            # The roll policy only ever selects a contract with expiry after
            # the day, so this cannot happen; fail closed rather than divide.
            raise ValidationError(
                f"active contract {code} has no time to expiry on {chain.date.date()}"
            )
        builds = 0
        for first in np.unique(env_key, return_index=True)[1]:
            i = int(first)
            same = env_key == env_key[i]
            div = dividend_yield_for_day(
                chain, i, spot=float(paths.spot[i, d]), rate=float(rate[i]),
                engine_config=self.config.engine_config, active_contract=code,
                curve_tenors=paths.tenor_grid, curve_carry=paths.carry[i, d, :],
            )
            for j in np.flatnonzero(same):
                div_yield[int(j)] = div
            b = (float(chain.prices[i, column]) / float(paths.spot[i, d]) - 1.0) / tenor
            basis[same] = b
            implied_q[same] = max(0.0, float(rate[i]) - b)
            builds += 1
        return env_key, tuple(div_yield), basis, implied_q, builds

    def _pricing_q(self, pricer, day, div_yield, implied_q) -> np.ndarray:
        """The scalar the cube records, per ``ProductReplay.recorded_pricing_q``.

        A legacy channel records the flat yield the pricer received; a term
        source has no single number, so the row carries the zero yield at
        the first product's remaining maturity.
        """
        if getattr(self.config.engine_config, "dividend_source", None) is None:
            return implied_q.copy()
        probe = self._schedule_env(day)
        remaining = float(pricer.aged_product(day, knocked_in=False).get_maturity(probe))
        return np.array([float(div.get_yield(remaining)) for div in div_yield])
```

- [ ] **Step 5: Write the pricing and lifecycle helpers**

```python
    # -- pricing -------------------------------------------------------

    def _states_for(self, day, d, idx, knocked_in, spot, vol, rate, q_T, div_yield, basis, env_key):
        return DayStates(
            day_index=d, date=day, path_index=idx, spot=spot[idx], vol=vol[idx], rate=rate[idx],
            q_T=q_T[idx], div_yield=tuple(div_yield[int(i)] for i in idx),
            basis_yield=basis[idx], env_key=env_key[idx], knocked_in=knocked_in,
        )

    def _initial_book_value(self, pricers, day, d, spot, vol, rate, q_T, div_yield, basis, env_key):
        """Day 0's book value, priced BEFORE any lifecycle event, like the replay.

        A ``ReplayProduct`` that states an ``initial_price`` uses it (that is
        how a paired study books the traded price); otherwise the day-0 PV
        stands in, with no knock-in, because nothing has happened yet.
        """
        total = np.zeros(spot.size)
        everyone = np.arange(spot.size)
        for bp, quantity, pricer in zip(self.config.products, self.quantities, pricers):
            if bp.initial_price is not None:
                total += float(quantity) * float(bp.initial_price)
                continue
            states = self._states_for(day, d, everyone, np.zeros(spot.size, dtype=bool),
                                      spot, vol, rate, q_T, div_yield, basis, env_key)
            pv, _, _ = pricer.price_day(states)
            total += float(quantity) * pv
        return total

    def _price_book(self, pricers, lifecycle, day, d, spot, vol, rate, q_T, div_yield, basis, env_key):
        """Mark every alive product-path; a dead one contributes nothing."""
        n_paths = spot.size
        product_mtm = np.zeros(n_paths)
        net_delta = np.zeros(n_paths)
        net_gamma = np.zeros(n_paths)
        for p, pricer in enumerate(pricers):
            idx = np.flatnonzero(lifecycle.alive[p])
            if idx.size == 0:
                continue
            states = self._states_for(day, d, idx, lifecycle.knocked_in[p][idx], spot, vol, rate,
                                      q_T, div_yield, basis, env_key)
            pv, delta, gamma = pricer.price_day(states)
            quantity = float(self.quantities[p])
            product_mtm[idx] += quantity * pv
            net_delta[idx] += quantity * delta
            net_gamma[idx] += quantity * gamma
        return product_mtm, net_delta, net_gamma

    def _payoffs(self, pricer, day, idx, knocked_in, spot, vol, rate, div_yield, basis):
        """``product.get_payoff`` per path, each in its own environment."""
        product = pricer.aged_product(day, knocked_in=False)
        out = np.empty(idx.size)
        for n, i in enumerate(idx):
            i = int(i)
            env = PricingEnvironment(
                spot_quote=SpotQuote(spot=float(spot[i]), asset_name=self.config.underlying),
                vol_surface=FlatVolSurface(volatility=float(vol[i])),
                rate_curve=FlatRateCurve(rate=float(rate[i])),
                div_yield=div_yield[i],
                basis_yield=ImpliedBasisYield(float(basis[i])),
                valuation_date=pd.Timestamp(day).to_pydatetime(),
            )
            out[n] = float(product.get_payoff(float(spot[i]), env, knocked_in=bool(knocked_in[n])))
        return out

    def _receivable_pv(self, lifecycle, rate, d, schedules) -> np.ndarray:
        """Determined-but-unpaid cash, discounted the replay's way.

        ``(settlement - date).days / 365`` through the day's flat rate curve
        is the arithmetic ``ProductReplay.pending_receivable_pv`` uses, so
        the two agree to the bit.
        """
        out = np.zeros(rate.size)
        for p, schedule in enumerate(schedules):
            live = np.flatnonzero(lifecycle.pending[p] != 0.0)
            for i in live:
                i = int(i)
                settle_day = int(lifecycle.settlement_day[p][i])
                if settle_day < 0:
                    continue
                tau = max((schedule.dates[settle_day] - schedule.dates[d]).days / 365.0, 0.0)
                out[i] += float(lifecycle.pending[p][i]) * float(
                    FlatRateCurve(rate=float(rate[i])).get_discount_factor(tau)
                )
        return out
```

- [ ] **Step 6: Write the hedge and recording helpers**

These follow `ReplayBacktestEngine._rebalance`, `_roll_contract`, `_execute_hedge_trade` and `_record_day` line for line.

```python
    # -- hedge ---------------------------------------------------------

    def _rebalance(self, ledger, net_delta, any_alive, futures_price, code, multiplier,
                   d, day, trades, costs) -> None:
        """Target the book delta while alive; close out once it is dead."""
        strategy = self.config.strategy
        target = np.where(any_alive, target_contracts_vector(strategy, net_delta, multiplier), 0.0)
        should = np.where(
            any_alive,
            should_rebalance_vector(strategy, ledger.quantity, target),
            np.abs(ledger.quantity) > POSITION_EPS,
        )
        trade_size = np.where(should, target - ledger.quantity, 0.0)
        trade_size[np.abs(trade_size) <= POSITION_EPS] = 0.0
        if not np.any(trade_size):
            return
        trade_type = np.where(any_alive, "hedge_rebalance", "hedge_close")
        reason = np.where(any_alive, "delta_rebalance", "product_terminated")
        self._execute(ledger, trade_size, futures_price, code, multiplier, d, day,
                      trade_type, reason, trades, costs)

    def _roll(self, ledger, chain, code, column, old_contract, d, day, trades, costs) -> None:
        """Close the old lane and reopen in the new contract, both legs costed."""
        qty = ledger.quantity.copy()
        if not np.any(np.abs(qty) > POSITION_EPS):
            return
        if old_contract in chain.contracts:
            old_price = chain.prices[:, chain.contracts.index(old_contract)]
            reason = "futures_roll"
        else:
            # The replay closes at the NEW contract's price when the old one
            # is no longer listed, and records that in the reason.
            old_price = chain.prices[:, column]
            reason = "futures_roll_missing_old_contract"
        reasons = np.full(qty.size, reason)
        self._execute(ledger, -qty, old_price, old_contract, chain.multiplier, d, day,
                      np.full(qty.size, "roll_close"), reasons, trades, costs)
        self._execute(ledger, qty, chain.prices[:, column], code, chain.multiplier, d, day,
                      np.full(qty.size, "roll_open"), reasons, trades, costs)

    def _execute(self, ledger, quantity_delta, price, contract, multiplier, d, day,
                 trade_type, reason, trades, costs) -> None:
        """Book a per-path trade and its cost, one log row per trading path."""
        for i in np.flatnonzero(np.abs(quantity_delta) > POSITION_EPS):
            i = int(i)
            qty, px = float(quantity_delta[i]), float(price[i])
            notional = abs(qty * px * float(multiplier))
            cost = float(self.config.transaction_cost_model.calculate_cost(
                quantity=qty, price=px, notional=notional,
                instrument_type="futures", trade_type=str(trade_type[i]),
            ))
            costs[i] += cost
            trades.append({
                "path": i, "day": d, "date": day, "trade_type": str(trade_type[i]),
                "contract": contract, "quantity": qty, "price": px,
                "multiplier": float(multiplier), "notional": notional,
                "transaction_cost": cost, "reason": str(reason[i]),
            })
        ledger.trade(quantity_delta, price, contract, multiplier)

    # -- recording -----------------------------------------------------

    def _record(self, cube, d, *, spot, vol, rate, q_T, implied_q, basis, futures_price,
                pre_hedge, contracts, product_mtm, receivable, cashflows, costs, product_pnl,
                hedge_mtm, total_pnl, cash, portfolio_value, delta, gamma, flags) -> None:
        """Write the day's column of the state cube."""
        columns = {
            "portfolio_value": portfolio_value, "product_mtm": product_mtm, "hedge_mtm": hedge_mtm,
            "cash": cash, "cashflows": cashflows, "transaction_costs": costs,
            "product_pnl": product_pnl, "hedge_pnl": hedge_mtm, "total_pnl": total_pnl,
            "spot": spot, "volatility": vol, "rate": rate, "pricing_q": q_T,
            "implied_q": implied_q, "basis_yield": basis, "futures_price": futures_price,
            "futures_contracts": contracts, "pre_hedge_contracts": pre_hedge,
            "delta": delta, "gamma": gamma, "pending_receivable_pv": receivable,
        }
        for name, values in columns.items():
            getattr(cube, name)[:, d] = values
        for name in BOOL_COLUMNS:
            getattr(cube, name)[:, d] = flags[name]

    def _manifest(self, paths, pricers, cache, dividend_builds, lifecycle, started, days_run):
        return {
            "path_fingerprint": paths.fingerprint(),
            "days_run": int(days_run),
            "path_meta": dict(paths.meta),
            "engine_fingerprint": pricers[0].fingerprint(),
            "product_fingerprints": [p.product_fingerprint for p in pricers],
            "provider": "repricing",
            "mode": "exact",
            "gate": pricers[0].verify(
                empty_states(0, paths.dates[0]), self.config.pricing.gate
            ).as_dict(),
            "engine_calls": sum(p.stats()["engine_calls"] for p in pricers),
            "cache": cache.stats().as_dict(),
            "dividend_builds": dividend_builds,
            "data_end_paths": int(np.count_nonzero(~lifecycle.settled.all(axis=0))),
            "underlying": self.config.underlying,
            "seconds": time.perf_counter() - started,
            "metadata": dict(self.config.metadata),
        }
```

**One edit back in Task 4:** the manifest reads `pricer.product_fingerprint`. Expose it there as a read-only property beside `fingerprint()`:

```python
    @property
    def product_fingerprint(self) -> str:
        """Identity of the contract this pricer prices."""
        return self._product_fp
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_engine.py -q`
Expected: 11 passed.

- [ ] **Step 8: Commit**

```bash
git add quantark/backtest/simulation/engine.py quantark/backtest/simulation/pricing/repricing.py \
        test/simulation/test_engine.py
git commit -m "feat(backtest/simulation): the vectorised daily loop and its state cube

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 10: The conformance oracle

**Files:**
- Create: `quantark/backtest/simulation/conformance.py`
- Create: `test/simulation/test_conformance.py`

**What it proves.** Plan 1 gave the simulation a way to hand one path to the real `ReplayBacktestEngine` (`to_market_dataset`). This task runs both engines on the same path with the same settings and compares them. In exact repricing mode the two make the same engine calls on the same aged product in the same environment, so the claim is equality, not closeness: lifecycle flags, the active contract, hedge size and every trade match exactly, and `product_mtm`, `delta` and `total_pnl` match to float rounding. That is the guarantee that makes a simulated result mean the same thing as a historical one.

**Interfaces:**
- Consumes: `EnsembleConfig`, `EnsembleBacktestEngine` (Task 9), `to_market_dataset` (plan 1), `ReplayBacktestEngine`, `ReplayBacktestConfig`, `BookBacktestResults`.
- Produces:
  - `OracleReport(path_index, days, exact_columns_match, max_pv_gap, max_delta_gap, max_total_pnl_gap, trade_mismatches, first_mismatch, passed)` frozen; `as_dict()`; `summary() -> str`.
  - `run_oracle(config: EnsembleConfig, paths: MarketPath, path_index: int, *, pv_tolerance: float = 0.0, delta_tolerance: float = 0.0) -> OracleReport`.
  - `main(argv=None) -> int` and a `__main__` guard, so `python -m quantark.backtest.simulation.conformance --run-dir DIR --path 17` works once plan 4 adds `to_dir`; in this plan the CLI takes `--path` plus the arguments needed to rebuild a config from a Python entry point, and prints `report.summary()`.

- [ ] **Step 1: Write the failing tests**

`test/simulation/test_conformance.py`:

```python
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantark.backtest.replay import ReplayProduct
from quantark.backtest.simulation.conformance import OracleReport, run_oracle
from quantark.backtest.simulation.engine import EnsembleBacktestEngine
from quantark.backtest.simulation.paths.gbm import ConstantVol, GBMPaths
from quantark.backtest.simulation.paths.market_path import DEFAULT_TENOR_GRID, StartState, trading_calendar

from .conftest import RATE, SPOT, VOL, ensemble_config, flat_carry, pde_engine_config, short_snowball

START = date(2024, 1, 2)


def _paths(n_paths: int = 5, n_days: int = 8, sigma: float = 0.6):
    start = StartState(spot=SPOT, atm_vol=VOL, rate=RATE, carry=flat_carry(DEFAULT_TENOR_GRID))
    gen = GBMPaths(start=start, calendar=trading_calendar(START, n_days), mu=0.0, sigma=sigma,
                   vol_rule=ConstantVol(VOL), carry_schedule=None, rate=RATE,
                   tenor_grid=DEFAULT_TENOR_GRID)
    return gen.generate(n_paths, n_days, seed=11)


@pytest.mark.parametrize("path_index", [0, 1, 2, 3, 4])
def test_every_path_matches_the_replay_engine_exactly(path_index):
    report = run_oracle(ensemble_config(), _paths(), path_index)
    assert isinstance(report, OracleReport)
    assert report.passed, report.summary()
    assert report.exact_columns_match
    assert report.trade_mismatches == 0
    assert report.max_pv_gap == 0.0
    assert report.max_delta_gap == 0.0
    assert report.max_total_pnl_gap == 0.0


def test_a_term_dividend_source_also_matches():
    cfg = ensemble_config(engine_config=pde_engine_config(
        dividend_source="futures_curve", futures_curve_extrapolation="flat_q",
        futures_curve_min_tenor_days=1,
    ))
    report = run_oracle(cfg, _paths(n_paths=2), 1)
    assert report.passed, report.summary()


def test_a_traded_initial_price_matches():
    cfg = ensemble_config(products=[
        ReplayProduct(product=short_snowball(), quantity=-1.0, position_id=1,
                      has_lifecycle=True, initial_price=0.0)
    ])
    report = run_oracle(cfg, _paths(n_paths=2), 0)
    assert report.passed, report.summary()


def test_a_two_product_book_matches():
    cfg = ensemble_config(products=[
        ReplayProduct(product=short_snowball(), quantity=-1.0, position_id=1, has_lifecycle=True),
        ReplayProduct(product=short_snowball(ko_days=(3, 5)), quantity=-2.0, position_id=2,
                      has_lifecycle=True),
    ])
    report = run_oracle(cfg, _paths(n_paths=2), 0)
    assert report.passed, report.summary()


def test_a_deliberate_disagreement_is_reported_not_hidden():
    # A different hedge ratio in the simulation must show up as a mismatch.
    from quantark.backtest.strategy.futures_delta_strategy import AutocallableDeltaHedgeStrategy

    cfg = ensemble_config(strategy=AutocallableDeltaHedgeStrategy(
        delta_threshold=0.0, hedge_ratio=0.5, target_delta=0.0))
    report = run_oracle(cfg, _paths(n_paths=1), 0, _replay_strategy=AutocallableDeltaHedgeStrategy(
        delta_threshold=0.0, hedge_ratio=1.0, target_delta=0.0))
    assert not report.passed
    assert report.first_mismatch is not None


def test_the_report_summary_names_the_path_and_the_day_count():
    report = run_oracle(ensemble_config(), _paths(n_paths=1), 0)
    text = report.summary()
    assert "path 0" in text and str(report.days) in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_conformance.py -q`
Expected: `ModuleNotFoundError: No module named 'quantark.backtest.simulation.conformance'`

- [ ] **Step 3: Write the implementation**

`quantark/backtest/simulation/conformance.py`:

```python
"""The replay engine as the simulation's oracle (spec 10).

One path, both engines, the same settings.  In exact repricing mode the
two make the same engine calls on the same aged product in the same
environment, so the target is equality: the lifecycle flags, the active
contract, the hedge size and every trade must match exactly, and the
marked columns to float rounding.  A gap is reported, never absorbed.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from quantark.backtest.replay import ReplayBacktestConfig, ReplayBacktestEngine
from quantark.util.exceptions import ValidationError

from .config import EnsembleConfig
from .dataset import to_market_dataset
from .engine import EnsembleBacktestEngine
from .paths.market_path import MarketPath

#: Columns that must agree exactly, whatever the tolerances.
EXACT_COLUMNS = ("alive", "knocked_in", "knocked_out", "matured", "active_contract",
                 "futures_contracts")
#: Columns compared against the tolerances.
NUMERIC_COLUMNS = ("product_mtm", "total_pnl")


@dataclass(frozen=True)
class OracleReport:
    """What the two engines agreed and disagreed about on one path."""

    path_index: int
    days: int
    exact_columns_match: bool
    max_pv_gap: float
    max_delta_gap: float
    max_total_pnl_gap: float
    trade_mismatches: int
    first_mismatch: Optional[str]
    passed: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path_index": self.path_index, "days": self.days,
            "exact_columns_match": self.exact_columns_match, "max_pv_gap": self.max_pv_gap,
            "max_delta_gap": self.max_delta_gap, "max_total_pnl_gap": self.max_total_pnl_gap,
            "trade_mismatches": self.trade_mismatches, "first_mismatch": self.first_mismatch,
            "passed": self.passed,
        }

    def summary(self) -> str:
        verdict = "matches" if self.passed else "DIFFERS FROM"
        lines = [
            f"simulated path {self.path_index} {verdict} the replay engine over {self.days} days",
            f"  max product mark gap : {self.max_pv_gap:.6g}",
            f"  max delta gap        : {self.max_delta_gap:.6g}",
            f"  max total P&L gap    : {self.max_total_pnl_gap:.6g}",
            f"  trade mismatches     : {self.trade_mismatches}",
        ]
        if self.first_mismatch:
            lines.append(f"  first mismatch       : {self.first_mismatch}")
        return "\n".join(lines)


def run_oracle(
    config: EnsembleConfig,
    paths: MarketPath,
    path_index: int,
    *,
    pv_tolerance: float = 0.0,
    delta_tolerance: float = 0.0,
    _replay_strategy: Any = None,
) -> OracleReport:
    """Compare the ensemble engine with the replay engine on one path.

    ``_replay_strategy`` overrides the strategy on the replay side only; it
    exists so a test can introduce a known disagreement and check that this
    function reports it.
    """
    if not 0 <= path_index < paths.n_paths:
        raise ValidationError(f"path index {path_index} out of range for {paths.n_paths} paths")

    ensemble = EnsembleBacktestEngine(config).run(paths)
    simulated = ensemble.path_states(path_index)
    sim_trades = ensemble.path_trades(path_index)

    replay = ReplayBacktestEngine(
        ReplayBacktestConfig(
            products=list(config.products),
            market_data=to_market_dataset(paths, path_index, multiplier=config.hedge.multiplier),
            hedge=config.hedge,
            engine_config=config.engine_config,
            strategy=_replay_strategy or config.strategy,
            transaction_cost_model=config.transaction_cost_model,
            underlying=config.underlying,
            delta_bump_size=config.delta_bump_size,
            gamma_bump_size=config.gamma_bump_size,
            calculate_surfaces=False,
            calculate_event_probabilities=False,
            terminate_on_lifecycle_end=True,
        )
    ).run()
    expected = pd.DataFrame(replay.states)
    expected_greeks = pd.DataFrame(replay.greeks)

    days = min(len(simulated), len(expected))
    mismatches: List[str] = []
    exact_ok = True
    for name in EXACT_COLUMNS:
        left = simulated[name].to_numpy()[:days]
        right = expected[name].to_numpy()[:days]
        bad = np.flatnonzero(left != right)
        if bad.size:
            exact_ok = False
            d = int(bad[0])
            mismatches.append(f"day {d}: {name} {left[d]!r} vs {right[d]!r}")
    if len(simulated) != len(expected):
        exact_ok = False
        mismatches.append(f"day count {len(simulated)} vs {len(expected)}")

    def gap(name: str, frame: pd.DataFrame) -> float:
        return float(
            np.max(np.abs(simulated[name].to_numpy()[:days] - frame[name].to_numpy()[:days]))
        ) if days else 0.0

    max_pv_gap = gap("product_mtm", expected)
    max_total_pnl_gap = gap("total_pnl", expected)
    max_delta_gap = float(
        np.max(np.abs(simulated["delta"].to_numpy()[:days] - expected_greeks["delta"].to_numpy()[:days]))
    ) if days else 0.0

    trade_mismatches = _compare_trades(sim_trades, pd.DataFrame(replay.trades), days, mismatches)

    passed = (
        exact_ok
        and trade_mismatches == 0
        and max_pv_gap <= pv_tolerance
        and max_delta_gap <= delta_tolerance
        and max_total_pnl_gap <= pv_tolerance
    )
    return OracleReport(
        path_index=int(path_index), days=days, exact_columns_match=exact_ok,
        max_pv_gap=max_pv_gap, max_delta_gap=max_delta_gap, max_total_pnl_gap=max_total_pnl_gap,
        trade_mismatches=trade_mismatches,
        first_mismatch=mismatches[0] if mismatches else None, passed=passed,
    )


def _compare_trades(simulated: pd.DataFrame, expected: pd.DataFrame, days: int,
                    mismatches: List[str]) -> int:
    """Trade for trade: same day, type, contract, size and price."""
    fields = ("trade_type", "contract", "quantity", "price")
    left = [
        (pd.Timestamp(r["date"]).normalize(), *(r[f] for f in fields))
        for _, r in simulated.iterrows()
    ]
    right = [
        (pd.Timestamp(r["date"]).normalize(), *(r[f] for f in fields))
        for _, r in expected.iterrows()
    ]
    bad = 0
    for n, (a, b) in enumerate(zip(left, right)):
        if a != b:
            bad += 1
            if len(mismatches) < 5:
                mismatches.append(f"trade {n}: {a} vs {b}")
    if len(left) != len(right):
        bad += abs(len(left) - len(right))
        mismatches.append(f"trade count {len(left)} vs {len(right)}")
    return bad


def main(argv: Optional[List[str]] = None) -> int:
    """CLI: check one path of a run built by an entry point module."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", type=int, required=True, help="path index to check")
    parser.add_argument(
        "--builder", required=True,
        help="import path of a callable returning (EnsembleConfig, MarketPath), "
             "e.g. mypkg.cells:desk_a",
    )
    args = parser.parse_args(argv)
    module_name, _, attribute = args.builder.partition(":")
    if not attribute:
        raise ValidationError("--builder must look like 'module:callable'")
    module = __import__(module_name, fromlist=[attribute])
    config, paths = getattr(module, attribute)()
    report = run_oracle(config, paths, args.path)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_conformance.py -q`
Expected: 10 passed. This is the task most likely to surface a real discrepancy; when a comparison fails, the report's `first_mismatch` names the day and the column, and the fix belongs in whichever of Tasks 6–9 owns that column, not in a loosened tolerance.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation/conformance.py test/simulation/test_conformance.py
git commit -m "feat(backtest/simulation): conformance oracle against the replay engine

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 11: Public exports, module guide, full regression

**Files:**
- Modify: `quantark/backtest/simulation/__init__.py`
- Modify: `quantark/backtest/simulation/README.md`
- Modify: `test/simulation/test_market_path.py` (extend the export test)

**Interfaces:**
- Produces: `from quantark.backtest.simulation import EnsembleConfig, PricingProviderConfig, CacheConfig, GateConfig, EnsembleBacktestEngine, EnsembleResults, StateCube, RepricingPricer, StateCache, DayStates, StateKey, GateReport, VectorLifecycle, VectorHedgeLedger, OracleReport, run_oracle` alongside everything plan 1 exports.

- [ ] **Step 1: Extend the failing export test**

Replace the name list in `test_public_api_is_exported` in `test/simulation/test_market_path.py` with the plan-1 names plus:

```python
                 "EnsembleConfig", "PricingProviderConfig", "CacheConfig", "GateConfig",
                 "EnsembleBacktestEngine", "EnsembleResults", "StateCube", "RepricingPricer",
                 "StateCache", "DayStates", "StateKey", "GateReport", "VectorLifecycle",
                 "VectorHedgeLedger", "OracleReport", "run_oracle",
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest -n0 test/simulation/test_market_path.py -k public_api -q`
Expected: `AssertionError: EnsembleConfig`

- [ ] **Step 3: Export and document**

Add to `quantark/backtest/simulation/__init__.py`, keeping the plan-1 block:

```python
from .config import CacheConfig, EnsembleConfig, GateConfig, PricingProviderConfig
from .conformance import OracleReport, run_oracle
from .engine import EnsembleBacktestEngine, EnsembleResults, StateCube
from .hedge import VectorHedgeLedger
from .lifecycle import VectorLifecycle
from .pricing import DayStates, GateReport, StateKey
from .pricing.cache import StateCache
from .pricing.repricing import RepricingPricer
```

and extend `__all__` with the same names.

Then add three sections to `quantark/backtest/simulation/README.md`, each two to six sentences, and replace the "What comes next" list with the remaining plans:

- **Running a backtest** — `EnsembleConfig` and what each field means; the `EnsembleBacktestEngine.run(paths)` call and the shape of the cube; a runnable snippet building a config and reading `path_states(0)`.
- **How a state gets priced** — the exact provider, the state key and why nothing about the hedge is in it, the per-day dedup of the dividend object, and the MC seed rule.
- **Why you can trust it** — the conformance oracle: what matches exactly and what matches to rounding, and the one-line CLI to check a path.

Also record the two departures from the spec's draft interfaces (`DayStates` carrying the environment, no `rate_schedule`) in a short "Notes on the design" paragraph, so a reader of the spec is not surprised.

- [ ] **Step 4: Run the full regression**

Run: `.venv/bin/python -m pytest test/simulation test/test_lifecycle_calendar_schedule.py test/test_replay_dividend_source.py test/test_snowball_q_term_structure_study.py test/replay_golden test/test_forward_carry_curve.py -q`
Then the whole suite: `.venv/bin/python -m pytest -q`
Expected: green apart from the two pre-existing `test/mo_volmodels/test_dashboard.py` failures, which pin on-disk run-dir counts and predate this work. Before staging, `git checkout -- example/mo_volmodels/data/` (a suite run rewrites two tracked sample files).

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/simulation test/simulation
git commit -m "feat(backtest/simulation): public API and module guide for the ensemble engine

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review against the spec

- **§4.2 lifecycle and hedge state** → Tasks 6, 7. Every array the spec lists is present, with `observed_ko`/`observed_ki` per product and the terminal/settlement day positions that carry the pending cashflow.
- **§4.3 state cube** → Task 9's `StateCube`; `path_states(i)` returns a frame in the replay's schema, truncated at the path's own settlement day.
- **§7.1 interface** → Task 2, with the documented change that `DayStates` carries the dividend object, the basis and an `env_key` rather than a lone `q_T`.
- **§7.3 exact mode** → Task 4, including the MC seed rule and the two-call (price + greeks) discipline the replay uses.
- **§7.4 cache** → Task 3 (memory tier); the disk tier and its shard-fingerprint rule are plan 3.
- **§7.5 gate** → Task 4's `verify` returns the zero report the spec prescribes for exact mode; the sampling gate is plan 3.
- **§8 the loop** → Task 9, step for step, including the day-0 pre-lifecycle initial book value, the roll's two costed legs, the hedge-close on death and the accounting identity.
- **§10 oracle** → Task 10, with equality on flags, contract, hedge size and trades, and rounding tolerance on the marked columns.
- **§11 configuration** → Task 1, minus the fields whose features are not in this plan, which are added by the plan that implements them.
- **§12 error handling** → the validation tests in Tasks 1, 3, 6, 7, 9 and the fail-closed `ValidationError` wrapping in Task 4's `_price_one`.
- **§13 testing** → generators (plan 1), carry to chain (plan 1), repricing (Task 4), lifecycle cross-checked against the tracker (Task 6), hedge against `FuturesHedgePosition` (Task 7), the accounting identity (Task 9), conformance (Task 10). Life-surface tests are plan 3; results tests are plan 4.
- **Type consistency:** `DayStates` field names are identical in Tasks 2, 4 and 9; `StateKey` uses `env_key` throughout; `VectorLifecycle` exposes `alive/knocked_in/pending/realized/settlement_day` under those names in Tasks 6 and 9; `VectorHedgeLedger.quantity` and `mark_to_market` are used as defined in Tasks 7 and 9; `RepricingPricer.price_day(states)` takes one argument everywhere; `product_fingerprint` is added as a property in Task 9's Step 6 note and consumed by the manifest in the same task.
- **Not in this plan, by design:** the ladder, bucketing, the disk cache, the sampling gate, the life surface, process pools, results distributions and the example study, each named in the Scope table with its plan.
