# Execution Framework Phase 0 — Contracts and Inventory — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the `quantark.execution` package — contracts, errors, policy
resolution, run context, adapter registry, serial `LegacyPriceAdapter`,
manifest/diagnostics skeletons, and the checked-in engine inventory — so that
every exported MC/PDE engine is inventoried and session-reachable in serial,
with zero direct public behavior change.

**Architecture:** New top-level package `quantark/execution/` per spec §4.1
(subset needed for Phase 0). The kernel never statically imports asset code;
adapter resolution matches engine classes by MRO against *string* class paths.
`BaseEngine.execute` is added with a lazy import. All framework dataclasses are
frozen. Tests live in `test/execution/`.

**Tech Stack:** Python 3.10+ stdlib (`dataclasses`, `enum`, `uuid`, `time`,
`importlib`, `platform`), NumPy/SciPy only for version stamping. pytest.

**Spec:** `docs/superpowers/specs/2026-07-15-mc-pde-performance-generalization-design.md`
(contract v1, Phase 0 scope from §21).

## Global Constraints

- Python env: `.venv/` — run tests as `.venv/bin/python -m pytest` (serial
  debugging: `-n0`).
- Canonical `quantark.*` imports only; never flat legacy imports.
- `quantark/execution/` modules must have **no static import** of
  `quantark.asset.*`, `quantark.volmodels.*`, `quantark.montecarlo.*`, or any
  product code (spec §4.1). Dynamic per-call imports inside inventory
  discovery functions are allowed.
- All framework exceptions derive from `quantark.util.exceptions.QuantArkException`
  (spec §15).
- All new public dataclasses are `@dataclass(frozen=True)` (spec §3.3: no
  mutable run state).
- No existing constructor, method signature, default, warning, or exception
  behavior changes (spec §17.1). The ONLY touch to existing code is adding the
  non-abstract `BaseEngine.execute` method (spec §5.4).
- Phase 0 backends: serial only. Any other backend request raises
  `CapabilityError` — no silent fallback (spec §3.3).
- Inventory `temporary_legacy` rows require `owner` and `milestone`
  (spec §18).
- Commit style: `feat(execution): ...` / `test(execution): ...`; end commit
  messages with the Claude co-author trailer.

## File Structure

```
quantark/execution/
    __init__.py          # public exports (spec §5)
    errors.py            # 6 framework exceptions
    diagnostics.py       # RunDiagnostics + InMemoryDiagnosticsSink
    manifest.py          # ReproducibilityManifest skeleton
    contracts.py         # requests, outcomes, capabilities, scenario contracts
    policy.py            # ExecutionPolicy/DeterminismPolicy/ResourceBudget + env resolution
    context.py           # PricingRunContext
    registry.py          # AdapterRegistry (string-keyed MRO resolution)
    legacy_adapter.py    # LegacyPriceAdapter (serial compatibility adapter)
    kernel.py            # ExecutionKernel.dispatch (serial lifecycle)
    api.py               # PricingSession
    inventory.py         # InventoryRecord + checked-in ENGINE_INVENTORY + discovery
test/execution/
    __init__.py
    test_errors.py
    test_contracts.py
    test_policy.py
    test_context.py
    test_registry.py
    test_session_parity.py
    test_inventory.py
    freeze_goldens.py    # golden-fixture freeze script (run manually once)
    goldens/phase0_goldens.json
quantark/asset/equity/engine/base_engine.py   # + execute() method only
```

---

### Task 1: Framework errors

**Files:**
- Create: `quantark/execution/__init__.py` (placeholder docstring only, filled in Task 7)
- Create: `quantark/execution/errors.py`
- Test: `test/execution/test_errors.py` (+ empty `test/execution/__init__.py`)

**Interfaces:**
- Produces: `CapabilityError`, `ResourceBudgetExceeded`, `PreparationError`,
  `TaskExecutionError`, `DeterminismViolation`, `ValidationGateError` — all
  subclasses of `QuantArkException`, importable from
  `quantark.execution.errors`.

- [ ] **Step 1: Write the failing test**

```python
# test/execution/test_errors.py
"""Framework exception hierarchy (spec section 15)."""
import pytest

from quantark.util.exceptions import QuantArkException


FRAMEWORK_ERRORS = [
    "CapabilityError",
    "ResourceBudgetExceeded",
    "PreparationError",
    "TaskExecutionError",
    "DeterminismViolation",
    "ValidationGateError",
]


@pytest.mark.parametrize("name", FRAMEWORK_ERRORS)
def test_framework_error_derives_quantark_root(name):
    import quantark.execution.errors as errors

    exc_type = getattr(errors, name)
    assert issubclass(exc_type, QuantArkException)
    with pytest.raises(QuantArkException):
        raise exc_type("boom")


def test_errors_module_all_is_exact():
    import quantark.execution.errors as errors

    assert sorted(errors.__all__) == sorted(FRAMEWORK_ERRORS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest -n0 test/execution/test_errors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'quantark.execution'`

- [ ] **Step 3: Write minimal implementation**

`quantark/execution/__init__.py`:

```python
"""QuantArk composable execution kernel (framework contract v1, Phase 0)."""
```

`test/execution/__init__.py`: empty file.

`quantark/execution/errors.py`:

```python
"""Typed framework exceptions (spec section 15).

All framework errors derive from the existing ``QuantArkException`` root.
Direct legacy engine methods re-raise their historical exceptions without
framework wrapping; these types appear only on explicit framework APIs.
"""
from quantark.util.exceptions import QuantArkException

__all__ = [
    "CapabilityError",
    "ResourceBudgetExceeded",
    "PreparationError",
    "TaskExecutionError",
    "DeterminismViolation",
    "ValidationGateError",
]


class CapabilityError(QuantArkException):
    """An engine/adapter does not support the requested operation, output,
    or backend. Explicit requests never silently fall back."""


class ResourceBudgetExceeded(QuantArkException):
    """A resource lease could not be acquired within the admitted budget."""


class PreparationError(QuantArkException):
    """Immutable prepared-state construction failed."""


class TaskExecutionError(QuantArkException):
    """A submitted execution task failed."""


class DeterminismViolation(QuantArkException):
    """Input mutated during execution, or a reproducibility check failed."""


class ValidationGateError(QuantArkException):
    """A declared numerical or scenario validation gate failed."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest -n0 test/execution/test_errors.py -v`
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
git add quantark/execution/__init__.py quantark/execution/errors.py test/execution/
git commit -m "feat(execution): framework exception types deriving QuantArkException"
```

---

### Task 2: Diagnostics and manifest skeletons

**Files:**
- Create: `quantark/execution/diagnostics.py`
- Create: `quantark/execution/manifest.py`
- Test: `test/execution/test_contracts.py` (first half; extended in Task 3)

**Interfaces:**
- Produces: `RunDiagnostics(adapter_id: str, timings: tuple[tuple[str, float], ...],
  policy_sources: tuple[tuple[str, str], ...], records: tuple[str, ...])` — frozen.
- Produces: `InMemoryDiagnosticsSink` with `emit(diagnostics) -> None` and
  `.entries: list[RunDiagnostics]`.
- Produces: `ReproducibilityManifest(schema_version: str, request_fingerprint: str | None,
  plan_fingerprint: str | None, adapter_id: str, adapter_version: str,
  engine_class_path: str, versions: tuple[tuple[str, str], ...], platform: str,
  resolved_policy: tuple[tuple[str, str], ...])` — frozen.
- Produces: `build_versions() -> tuple[tuple[str, str], ...]` stamping
  python/quantark/numpy/scipy versions.

- [ ] **Step 1: Write the failing test**

```python
# test/execution/test_contracts.py
"""Frozen framework value objects (spec sections 5, 14.3, 16)."""
import dataclasses

import pytest


class TestDiagnosticsAndManifest:
    def test_run_diagnostics_is_frozen(self):
        from quantark.execution.diagnostics import RunDiagnostics

        diag = RunDiagnostics(
            adapter_id="legacy-price",
            timings=(("execute_seconds", 0.5),),
            policy_sources=(("batch.backend", "default"),),
            records=(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            diag.adapter_id = "x"

    def test_in_memory_sink_collects(self):
        from quantark.execution.diagnostics import (
            InMemoryDiagnosticsSink,
            RunDiagnostics,
        )

        sink = InMemoryDiagnosticsSink()
        diag = RunDiagnostics(
            adapter_id="a", timings=(), policy_sources=(), records=()
        )
        sink.emit(diag)
        assert sink.entries == [diag]

    def test_manifest_versions_stamped(self):
        from quantark.execution.manifest import build_versions

        versions = dict(build_versions())
        assert set(versions) == {"python", "quantark", "numpy", "scipy"}
        assert all(isinstance(v, str) and v for v in versions.values())

    def test_manifest_is_frozen_with_schema_version(self):
        from quantark.execution.manifest import (
            MANIFEST_SCHEMA_VERSION,
            ReproducibilityManifest,
            build_versions,
        )

        manifest = ReproducibilityManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            request_fingerprint=None,
            plan_fingerprint=None,
            adapter_id="legacy-price",
            adapter_version="0",
            engine_class_path="x.Y",
            versions=build_versions(),
            platform="test",
            resolved_policy=(),
        )
        assert manifest.schema_version == "execution-manifest/0"
        with pytest.raises(dataclasses.FrozenInstanceError):
            manifest.platform = "other"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest -n0 test/execution/test_contracts.py -v`
Expected: FAIL with `ModuleNotFoundError` on `quantark.execution.diagnostics`

- [ ] **Step 3: Write minimal implementation**

`quantark/execution/diagnostics.py`:

```python
"""Operational diagnostics records and sinks (spec section 16).

Diagnostics are immutable after outcome construction and never participate in
economic equality or plan fingerprints.
"""
from dataclasses import dataclass, field

__all__ = ["RunDiagnostics", "InMemoryDiagnosticsSink"]


@dataclass(frozen=True)
class RunDiagnostics:
    """Immutable per-run operational record."""

    adapter_id: str
    timings: tuple = ()          # (("execute_seconds", 0.12), ...)
    policy_sources: tuple = ()   # (("batch.backend", "default"), ...)
    records: tuple = ()          # free-form operational note strings


class InMemoryDiagnosticsSink:
    """Default library sink: appends records to a list (spec section 16)."""

    def __init__(self):
        self.entries: list = []

    def emit(self, diagnostics: RunDiagnostics) -> None:
        self.entries.append(diagnostics)
```

`quantark/execution/manifest.py`:

```python
"""Reproducibility manifest skeleton (spec section 14.3).

Phase 0 stamps identity and dependency versions. Request and plan
fingerprints are populated from Phase 1 onward; ``None`` means
"fingerprint unavailable" (an uncacheable, legacy-adapted request).
"""
import platform as _platform
from dataclasses import dataclass

__all__ = ["MANIFEST_SCHEMA_VERSION", "ReproducibilityManifest", "build_versions"]

MANIFEST_SCHEMA_VERSION = "execution-manifest/0"


@dataclass(frozen=True)
class ReproducibilityManifest:
    schema_version: str
    request_fingerprint: str | None
    plan_fingerprint: str | None
    adapter_id: str
    adapter_version: str
    engine_class_path: str
    versions: tuple           # (("python", "3.12.1"), ...)
    platform: str
    resolved_policy: tuple    # (("batch.backend", "serial"), ...)


def build_versions() -> tuple:
    """Stamp interpreter and numerical dependency versions."""
    import numpy
    import scipy

    import quantark

    return (
        ("python", _platform.python_version()),
        ("quantark", getattr(quantark, "__version__", "unknown")),
        ("numpy", numpy.__version__),
        ("scipy", scipy.__version__),
    )


def platform_tag() -> str:
    return f"{_platform.system()}-{_platform.machine()}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest -n0 test/execution/test_contracts.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add quantark/execution/diagnostics.py quantark/execution/manifest.py test/execution/test_contracts.py
git commit -m "feat(execution): diagnostics records and reproducibility manifest skeleton"
```

---

### Task 3: Request/outcome/capability contracts

**Files:**
- Create: `quantark/execution/contracts.py`
- Modify: `test/execution/test_contracts.py` (append second test class)

**Interfaces:**
- Produces (all frozen dataclasses / enums, importable from
  `quantark.execution.contracts`):
  - `PricingOperation` enum: `PRICE`, `PRICE_DETAILED`, `EVENT_STATS`
  - `OutputKind` enum: `PV`, `ERROR_ESTIMATE`, `EVENT_STATS`, `CASHFLOWS`, `GRID`
  - `DEFAULT_OUTPUTS = frozenset({OutputKind.PV})`
  - `PricingRequest(product, pricing_env=None, operation=PRICE, outputs=DEFAULT_OUTPUTS, operation_options=(), request_id=None)`
    — `operation_options` is a tuple of `(key, value)` pairs (immutable mapping form).
  - `NormalizedPricingRequest(engine_class_path: str, operation, outputs, operation_options: tuple, product_ref, pricing_env_ref, snapshot_complete: bool, fingerprint: str | None)`
  - `FrameworkErrorInfo(error_type: str, message: str)`
  - `PricingOutcome(value, normalized_economics: tuple, diagnostics, manifest)`
  - `PricingFailure(item_id: str, error: FrameworkErrorInfo, diagnostics, manifest=None)`
  - `EngineCapabilities(operations: frozenset, output_kinds: frozenset, supported_backends: frozenset, fixed_planning: bool | None, prepared_state_thread_safe: bool, instance_reentrant: bool, process_reconstructable: bool, deterministic_reduction: bool, peak_memory_estimate: str, adapter_id: str, adapter_version: str)`
  - `ScenarioSpec(scenario_id, transformer_id, parameters: tuple, mutation_tags: frozenset, required_capabilities=frozenset(), validation_policy=None)`
  - `ScenarioOutcome(scenario_id, value, normalized_economics: tuple, diagnostics, manifest_fingerprint: str | None)`
- Note: `normalized_economics` is stored as a tuple of `(key, value)` pairs in
  Phase 0 (hashable, immutable); helpers `economics_mapping(outcome) -> dict`
  provided for reading.

- [ ] **Step 1: Write the failing test** (append to `test/execution/test_contracts.py`)

```python
class TestRequestAndOutcomeContracts:
    def test_pricing_request_defaults(self):
        from quantark.execution.contracts import (
            DEFAULT_OUTPUTS,
            OutputKind,
            PricingOperation,
            PricingRequest,
        )

        req = PricingRequest(product="P", pricing_env="E")
        assert req.operation is PricingOperation.PRICE
        assert req.outputs == DEFAULT_OUTPUTS == frozenset({OutputKind.PV})
        assert req.operation_options == ()
        assert req.request_id is None

    def test_pricing_request_is_frozen(self):
        import dataclasses

        from quantark.execution.contracts import PricingRequest

        req = PricingRequest(product="P")
        with pytest.raises(dataclasses.FrozenInstanceError):
            req.product = "Q"

    def test_env_bound_request_allows_missing_env(self):
        from quantark.execution.contracts import PricingRequest

        req = PricingRequest(product="bond")
        assert req.pricing_env is None

    def test_outcome_and_failure_are_frozen(self):
        import dataclasses

        from quantark.execution.contracts import (
            FrameworkErrorInfo,
            PricingFailure,
            PricingOutcome,
            economics_mapping,
        )
        from quantark.execution.diagnostics import RunDiagnostics

        diag = RunDiagnostics(adapter_id="a")
        outcome = PricingOutcome(
            value=1.25,
            normalized_economics=(("pv", 1.25),),
            diagnostics=diag,
            manifest=None,
        )
        assert economics_mapping(outcome) == {"pv": 1.25}
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.value = 2.0

        failure = PricingFailure(
            item_id="0",
            error=FrameworkErrorInfo("ValueError", "bad"),
            diagnostics=diag,
        )
        assert failure.manifest is None

    def test_engine_capabilities_and_scenario_contracts_exist(self):
        from quantark.execution.contracts import (
            EngineCapabilities,
            OutputKind,
            PricingOperation,
            ScenarioOutcome,
            ScenarioSpec,
        )

        caps = EngineCapabilities(
            operations=frozenset({PricingOperation.PRICE}),
            output_kinds=frozenset({OutputKind.PV}),
            supported_backends=frozenset({"serial"}),
            fixed_planning=None,
            prepared_state_thread_safe=False,
            instance_reentrant=False,
            process_reconstructable=False,
            deterministic_reduction=True,
            peak_memory_estimate="unavailable",
            adapter_id="legacy-price",
            adapter_version="0",
        )
        assert "serial" in caps.supported_backends
        spec = ScenarioSpec(
            scenario_id="s1",
            transformer_id="t",
            parameters=(("shift", 0.01),),
            mutation_tags=frozenset({"spot"}),
        )
        assert spec.required_capabilities == frozenset()
        assert ScenarioOutcome is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest -n0 test/execution/test_contracts.py -v`
Expected: new tests FAIL with `ModuleNotFoundError` on `quantark.execution.contracts`

- [ ] **Step 3: Write minimal implementation**

`quantark/execution/contracts.py`:

```python
"""Framework request, outcome, capability, and scenario contracts (spec section 5).

``PricingRequest`` is a shallow frozen envelope around potentially mutable
legacy objects. ``NormalizedPricingRequest`` is the immutable snapshot every
post-normalization capability method receives (spec section 6); in Phase 0 the
snapshot is shallow (``snapshot_complete=False``), so requests are uncacheable
and no fingerprint is claimed.

Immutable mappings are represented as sorted tuples of ``(key, value)`` pairs.
"""
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "PricingOperation",
    "OutputKind",
    "DEFAULT_OUTPUTS",
    "PricingRequest",
    "NormalizedPricingRequest",
    "FrameworkErrorInfo",
    "PricingOutcome",
    "PricingFailure",
    "EngineCapabilities",
    "ScenarioSpec",
    "ScenarioOutcome",
    "economics_mapping",
]


class PricingOperation(Enum):
    PRICE = "price"
    PRICE_DETAILED = "price_detailed"
    EVENT_STATS = "event_stats"


class OutputKind(Enum):
    PV = "pv"
    ERROR_ESTIMATE = "error_estimate"
    EVENT_STATS = "event_stats"
    CASHFLOWS = "cashflows"
    GRID = "grid"


DEFAULT_OUTPUTS = frozenset({OutputKind.PV})


@dataclass(frozen=True)
class PricingRequest:
    product: object
    pricing_env: object | None = None
    operation: PricingOperation = PricingOperation.PRICE
    outputs: frozenset = DEFAULT_OUTPUTS
    operation_options: tuple = ()
    request_id: str | None = None


@dataclass(frozen=True)
class NormalizedPricingRequest:
    engine_class_path: str
    operation: PricingOperation
    outputs: frozenset
    operation_options: tuple
    product_ref: object
    pricing_env_ref: object | None
    snapshot_complete: bool
    fingerprint: str | None


@dataclass(frozen=True)
class FrameworkErrorInfo:
    error_type: str
    message: str


@dataclass(frozen=True)
class PricingOutcome:
    value: object
    normalized_economics: tuple
    diagnostics: object
    manifest: object


@dataclass(frozen=True)
class PricingFailure:
    item_id: str
    error: FrameworkErrorInfo
    diagnostics: object
    manifest: object | None = None


@dataclass(frozen=True)
class EngineCapabilities:
    operations: frozenset
    output_kinds: frozenset
    supported_backends: frozenset
    fixed_planning: bool | None
    prepared_state_thread_safe: bool
    instance_reentrant: bool
    process_reconstructable: bool
    deterministic_reduction: bool
    peak_memory_estimate: str  # "exact" | "conservative" | "unavailable"
    adapter_id: str
    adapter_version: str


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    transformer_id: str
    parameters: tuple
    mutation_tags: frozenset
    required_capabilities: frozenset = frozenset()
    validation_policy: object | None = None


@dataclass(frozen=True)
class ScenarioOutcome:
    scenario_id: str
    value: object
    normalized_economics: tuple
    diagnostics: object
    manifest_fingerprint: str | None


def economics_mapping(outcome) -> dict:
    """Read a normalized-economics tuple back as a dict."""
    return dict(outcome.normalized_economics)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest -n0 test/execution/test_contracts.py -v`
Expected: all PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add quantark/execution/contracts.py test/execution/test_contracts.py
git commit -m "feat(execution): frozen request/outcome/capability/scenario contracts"
```

---

### Task 4: Policies, resource budget, env-alias resolution

**Files:**
- Create: `quantark/execution/policy.py`
- Test: `test/execution/test_policy.py`

**Interfaces:**
- Produces: `ExecutorSelection(backend="serial", workers=1, max_in_flight=None, may_shrink=False, fallback_order=())`
- Produces: `ExecutionPolicy(batch=ExecutorSelection(), scenario=ExecutorSelection(), nested_execution=False, fail_fast=True, retries=0)`
- Produces: `DeterminismPolicy(require_manifest=True, changed_plan_profile="reject", mismatch_raises=True)`
- Produces: `ResourceBudget(max_processes=1, max_threads=1, total_memory_bytes=None, draw_cache_bytes=None, artifact_cache_bytes=None, max_in_flight=1)`
- Produces: `resolve_execution_policy(explicit=None, environ=None) -> tuple[ExecutionPolicy, tuple[tuple[str, str], ...]]`
  — field-by-field precedence explicit > `QUANTARK_EXEC_*` env > default
  (spec §17.2), returning the policy plus a `(field, source)` map where source
  is one of `"explicit" | "env" | "default" | "env_invalid_default"`.
- Env aliases handled: `QUANTARK_EXEC_BATCH_BACKEND`, `QUANTARK_EXEC_BATCH_WORKERS`,
  `QUANTARK_EXEC_SCENARIO_BACKEND`, `QUANTARK_EXEC_SCENARIO_WORKERS`,
  `QUANTARK_EXEC_MEMORY_MB`, `QUANTARK_EXEC_CACHE_MB`, `QUANTARK_EXEC_MAX_IN_FLIGHT`.
  (`MEMORY_MB`/`CACHE_MB`/`MAX_IN_FLIGHT` resolve into
  `resolve_resource_budget(explicit=None, environ=None)` with the same
  precedence and source map.)
- Invalid env text (non-integer workers, unknown backend) resolves to the
  historical default and records `"env_invalid_default"` — matching the legacy
  tolerance for invalid `QUANTARK_DCN_MC_WORKERS` text (spec §17.1).

- [ ] **Step 1: Write the failing test**

```python
# test/execution/test_policy.py
"""Policy objects and field-by-field precedence resolution (spec section 17.2)."""
import dataclasses

import pytest

from quantark.execution.policy import (
    DeterminismPolicy,
    ExecutionPolicy,
    ExecutorSelection,
    ResourceBudget,
    resolve_execution_policy,
    resolve_resource_budget,
)


def test_defaults_are_serial_one_worker_fail_fast():
    policy = ExecutionPolicy()
    assert policy.batch.backend == "serial"
    assert policy.batch.workers == 1
    assert policy.scenario.backend == "serial"
    assert policy.nested_execution is False
    assert policy.fail_fast is True
    assert policy.retries == 0
    assert policy.batch.fallback_order == ()


def test_policy_objects_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ExecutionPolicy().fail_fast = False
    with pytest.raises(dataclasses.FrozenInstanceError):
        DeterminismPolicy().require_manifest = False
    with pytest.raises(dataclasses.FrozenInstanceError):
        ResourceBudget().max_threads = 4


def test_env_alias_resolution():
    env = {
        "QUANTARK_EXEC_BATCH_BACKEND": "threads",
        "QUANTARK_EXEC_BATCH_WORKERS": "4",
    }
    policy, sources = resolve_execution_policy(explicit=None, environ=env)
    assert policy.batch.backend == "threads"
    assert policy.batch.workers == 4
    src = dict(sources)
    assert src["batch.backend"] == "env"
    assert src["batch.workers"] == "env"
    assert src["scenario.backend"] == "default"


def test_explicit_wins_over_env_field_by_field():
    env = {"QUANTARK_EXEC_BATCH_WORKERS": "4"}
    explicit = ExecutionPolicy(batch=ExecutorSelection(backend="serial", workers=2))
    policy, sources = resolve_execution_policy(explicit=explicit, environ=env)
    # Explicit object wins wholesale for fields it sets.
    assert policy.batch.workers == 2
    assert dict(sources)["batch.workers"] == "explicit"


def test_invalid_env_text_falls_back_to_default():
    env = {
        "QUANTARK_EXEC_BATCH_WORKERS": "not-a-number",
        "QUANTARK_EXEC_BATCH_BACKEND": "quantum",
    }
    policy, sources = resolve_execution_policy(explicit=None, environ=env)
    assert policy.batch.workers == 1
    assert policy.batch.backend == "serial"
    src = dict(sources)
    assert src["batch.workers"] == "env_invalid_default"
    assert src["batch.backend"] == "env_invalid_default"


def test_resource_budget_env_resolution():
    env = {"QUANTARK_EXEC_MEMORY_MB": "1024", "QUANTARK_EXEC_MAX_IN_FLIGHT": "2"}
    budget, sources = resolve_resource_budget(explicit=None, environ=env)
    assert budget.total_memory_bytes == 1024 * 2**20  # 1024 MiB in bytes
    assert budget.max_in_flight == 2
    assert dict(sources)["total_memory_bytes"] == "env"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest -n0 test/execution/test_policy.py -v`
Expected: FAIL with `ModuleNotFoundError` on `quantark.execution.policy`

- [ ] **Step 3: Write minimal implementation**

`quantark/execution/policy.py`:

```python
"""Execution, determinism, and resource policies with precedence resolution.

Resolution is field-by-field, highest precedence first (spec section 17.2):
explicit setting > generic ``QUANTARK_EXEC_*`` environment alias > historical
default. Resolution happens once (at session construction); children receive
resolved values and never re-resolve the host environment.

Phase 0 note: resolution accepts any known backend string, but the serial
compatibility adapter only advertises the ``serial`` backend, so a non-serial
resolved backend fails capability validation at dispatch (no silent fallback).
"""
import os
from dataclasses import dataclass

__all__ = [
    "KNOWN_BACKENDS",
    "ExecutorSelection",
    "ExecutionPolicy",
    "DeterminismPolicy",
    "ResourceBudget",
    "resolve_execution_policy",
    "resolve_resource_budget",
]

KNOWN_BACKENDS = ("serial", "threads", "processes", "dask")


@dataclass(frozen=True)
class ExecutorSelection:
    backend: str = "serial"
    workers: int = 1
    max_in_flight: int | None = None
    may_shrink: bool = False
    fallback_order: tuple = ()


@dataclass(frozen=True)
class ExecutionPolicy:
    batch: ExecutorSelection = ExecutorSelection()
    scenario: ExecutorSelection = ExecutorSelection()
    nested_execution: bool = False
    fail_fast: bool = True
    retries: int = 0


@dataclass(frozen=True)
class DeterminismPolicy:
    require_manifest: bool = True
    changed_plan_profile: str = "reject"
    mismatch_raises: bool = True


@dataclass(frozen=True)
class ResourceBudget:
    max_processes: int = 1
    max_threads: int = 1
    total_memory_bytes: int | None = None
    draw_cache_bytes: int | None = None
    artifact_cache_bytes: int | None = None
    max_in_flight: int = 1


def _env_int(environ, key, default, field, sources):
    raw = environ.get(key)
    if raw is None:
        sources.append((field, "default"))
        return default
    try:
        value = int(raw)
    except ValueError:
        sources.append((field, "env_invalid_default"))
        return default
    sources.append((field, "env"))
    return value


def _env_backend(environ, key, default, field, sources):
    raw = environ.get(key)
    if raw is None:
        sources.append((field, "default"))
        return default
    if raw not in KNOWN_BACKENDS:
        sources.append((field, "env_invalid_default"))
        return default
    sources.append((field, "env"))
    return raw


def resolve_execution_policy(explicit=None, environ=None):
    """Resolve an ExecutionPolicy once; returns (policy, source map)."""
    if explicit is not None:
        # An explicit policy object is a complete, highest-precedence setting.
        fields = [
            "batch.backend", "batch.workers", "scenario.backend",
            "scenario.workers", "nested_execution", "fail_fast", "retries",
        ]
        return explicit, tuple((f, "explicit") for f in fields)

    environ = os.environ if environ is None else environ
    sources: list = []
    batch = ExecutorSelection(
        backend=_env_backend(environ, "QUANTARK_EXEC_BATCH_BACKEND",
                             "serial", "batch.backend", sources),
        workers=_env_int(environ, "QUANTARK_EXEC_BATCH_WORKERS",
                         1, "batch.workers", sources),
    )
    scenario = ExecutorSelection(
        backend=_env_backend(environ, "QUANTARK_EXEC_SCENARIO_BACKEND",
                             "serial", "scenario.backend", sources),
        workers=_env_int(environ, "QUANTARK_EXEC_SCENARIO_WORKERS",
                         1, "scenario.workers", sources),
    )
    sources.extend(
        [("nested_execution", "default"), ("fail_fast", "default"),
         ("retries", "default")]
    )
    return ExecutionPolicy(batch=batch, scenario=scenario), tuple(sources)


def resolve_resource_budget(explicit=None, environ=None):
    """Resolve a ResourceBudget once; returns (budget, source map)."""
    if explicit is not None:
        fields = ["total_memory_bytes", "artifact_cache_bytes", "max_in_flight"]
        return explicit, tuple((f, "explicit") for f in fields)

    environ = os.environ if environ is None else environ
    sources: list = []
    memory_mb = _env_int(environ, "QUANTARK_EXEC_MEMORY_MB",
                         None, "total_memory_bytes", sources)
    cache_mb = _env_int(environ, "QUANTARK_EXEC_CACHE_MB",
                        None, "artifact_cache_bytes", sources)
    max_in_flight = _env_int(environ, "QUANTARK_EXEC_MAX_IN_FLIGHT",
                             1, "max_in_flight", sources)
    return (
        ResourceBudget(
            total_memory_bytes=None if memory_mb is None else memory_mb * 2**20,
            artifact_cache_bytes=None if cache_mb is None else cache_mb * 2**20,
            max_in_flight=max_in_flight,
        ),
        tuple(sources),
    )
```

Note: `_env_int` with `default=None` must keep returning `None` when the key
is absent — the code above already handles this because `default` passes
through unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest -n0 test/execution/test_policy.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add quantark/execution/policy.py test/execution/test_policy.py
git commit -m "feat(execution): policy objects and QUANTARK_EXEC_* precedence resolution"
```

---

### Task 5: Run context

**Files:**
- Create: `quantark/execution/context.py`
- Test: `test/execution/test_context.py`

**Interfaces:**
- Consumes: `ExecutionPolicy`, `DeterminismPolicy`, `ResourceBudget` from Task 4;
  `InMemoryDiagnosticsSink` from Task 2.
- Produces: `PricingRunContext(execution_policy, resource_budget, determinism_policy,
  diagnostics_sink, adapter_registry=None, cancellation_token=None, run_id=<uuid str>,
  parent_run_id=None, config_snapshot=())` — frozen; `default_context()` factory;
  `context.child() -> PricingRunContext` with `parent_run_id` set and shared
  service handles.

- [ ] **Step 1: Write the failing test**

```python
# test/execution/test_context.py
"""Immutable run context (spec section 5.3)."""
import dataclasses

import pytest

from quantark.execution.context import PricingRunContext, default_context


def test_default_context_is_serial_and_frozen():
    ctx = default_context()
    assert ctx.execution_policy.batch.backend == "serial"
    assert ctx.parent_run_id is None
    assert isinstance(ctx.run_id, str) and ctx.run_id
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.run_id = "x"


def test_child_shares_services_and_links_parent():
    ctx = default_context()
    child = ctx.child()
    assert child.parent_run_id == ctx.run_id
    assert child.run_id != ctx.run_id
    assert child.diagnostics_sink is ctx.diagnostics_sink
    assert child.execution_policy is ctx.execution_policy
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest -n0 test/execution/test_context.py -v`
Expected: FAIL with `ModuleNotFoundError` on `quantark.execution.context`

- [ ] **Step 3: Write minimal implementation**

`quantark/execution/context.py`:

```python
"""Immutable pricing run context (spec section 5.3).

The context itself never mutates; repositories, sinks, and registries are
mutable services behind stable handles. ``child`` returns a new context in a
child scope. No active run context is ever stored globally or thread-locally
(spec section 3.3).
"""
import uuid
from dataclasses import dataclass, field, replace

from quantark.execution.diagnostics import InMemoryDiagnosticsSink
from quantark.execution.policy import (
    DeterminismPolicy,
    ExecutionPolicy,
    ResourceBudget,
    resolve_execution_policy,
    resolve_resource_budget,
)

__all__ = ["PricingRunContext", "default_context"]


def _new_run_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class PricingRunContext:
    execution_policy: ExecutionPolicy
    resource_budget: ResourceBudget
    determinism_policy: DeterminismPolicy
    diagnostics_sink: object
    adapter_registry: object | None = None
    cancellation_token: object | None = None
    run_id: str = field(default_factory=_new_run_id)
    parent_run_id: str | None = None
    config_snapshot: tuple = ()

    def child(self) -> "PricingRunContext":
        """New context in a child scope sharing service handles."""
        return replace(
            self, run_id=_new_run_id(), parent_run_id=self.run_id
        )


def default_context(environ=None) -> PricingRunContext:
    """Serial default context; resolves policy and budget exactly once."""
    policy, policy_sources = resolve_execution_policy(environ=environ)
    budget, budget_sources = resolve_resource_budget(environ=environ)
    return PricingRunContext(
        execution_policy=policy,
        resource_budget=budget,
        determinism_policy=DeterminismPolicy(),
        diagnostics_sink=InMemoryDiagnosticsSink(),
        config_snapshot=tuple(policy_sources) + tuple(budget_sources),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest -n0 test/execution/test_context.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add quantark/execution/context.py test/execution/test_context.py
git commit -m "feat(execution): immutable PricingRunContext with child scoping"
```

---

### Task 6: Adapter registry and LegacyPriceAdapter

**Files:**
- Create: `quantark/execution/registry.py`
- Create: `quantark/execution/legacy_adapter.py`
- Test: `test/execution/test_registry.py`

**Interfaces:**
- Produces: `AdapterRegistry` with
  `register(engine_class_path: str, adapter_factory: Callable[[], object]) -> None`
  (raises `ValidationError` on duplicate path or after freeze),
  `freeze() -> None`, `resolve(engine) -> object` (walks
  `type(engine).__mro__` matching `f"{cls.__module__}.{cls.__qualname__}"`;
  raises `CapabilityError` when nothing matches).
- Produces: `build_default_registry() -> AdapterRegistry` — registers the
  `LegacyPriceAdapter` for the known engine family base classes **by string
  path** (no asset imports):
  - `quantark.asset.equity.engine.base_engine.BaseEngine` → `product_env`
  - `quantark.asset.fx.engine.base_fx_engine.BaseFxEngine` → `product_env`
  - `quantark.asset.credit.engine.base_credit_engine.BaseCreditEngine` → `product_env`
  - `quantark.asset.bond.engine.pde.convertible.jump_diffusion_engine.ConvertibleBondJumpDiffusionEngine` → `env_bound`
  - `quantark.asset.bond.engine.pde.convertible.tf_engine.ConvertibleBondTFEngine` → `env_bound`
  - `quantark.asset.bond.engine.convertible.convertible_bond_engine.ConvertibleBondEngine` → `env_bound`
- Produces: `LegacyPriceAdapter(call_shape: str)` with
  `capabilities() -> EngineCapabilities`,
  `normalize(engine, request) -> NormalizedPricingRequest`,
  `execute_native(engine, request, normalized, context) -> tuple[object, tuple]`
  (native value + normalized-economics pairs). Constants
  `ADAPTER_ID = "legacy-price"`, `ADAPTER_VERSION = "0"`.
- Operation dispatch in `execute_native`:
  - `PRICE`: `engine.price(product, env)` (product_env) or `engine.price(product)` (env_bound)
  - `PRICE_DETAILED`: `engine.price_detailed(product, env)` if the attribute
    exists; env_bound engines use `engine.price_with_details(product)`;
    otherwise raise `CapabilityError`.
  - `EVENT_STATS`: `engine.calculate_event_stats(product, env)`; a `None`
    return or missing attribute raises `CapabilityError` (the legacy default
    "unsupported" contract, surfaced as an explicit-request failure).
- Native legacy exceptions propagate unwrapped (spec §15: direct legacy
  errors keep their types; the adapter adds no wrapping in Phase 0).

- [ ] **Step 1: Write the failing test**

```python
# test/execution/test_registry.py
"""Adapter registry resolution and the serial compatibility adapter."""
import pytest

from quantark.execution.contracts import (
    OutputKind,
    PricingOperation,
    PricingRequest,
)
from quantark.execution.errors import CapabilityError
from quantark.execution.legacy_adapter import ADAPTER_ID, LegacyPriceAdapter
from quantark.execution.registry import AdapterRegistry, build_default_registry
from quantark.util.exceptions import ValidationError


class _FakeProductEnvEngine:
    """Stands in for an equity-style engine: price(product, env)."""

    def price(self, product, env):
        return 42.0


class _FakeEnvBoundEngine:
    """Stands in for a convertible-bond-style engine: price(product)."""

    def price(self, product):
        return 7.0


def _register_fakes(registry):
    registry.register(
        f"{_FakeProductEnvEngine.__module__}.{_FakeProductEnvEngine.__qualname__}",
        lambda: LegacyPriceAdapter(call_shape="product_env"),
    )
    registry.register(
        f"{_FakeEnvBoundEngine.__module__}.{_FakeEnvBoundEngine.__qualname__}",
        lambda: LegacyPriceAdapter(call_shape="env_bound"),
    )


def test_resolution_matches_exact_class_then_mro():
    registry = AdapterRegistry()
    _register_fakes(registry)
    registry.freeze()

    adapter = registry.resolve(_FakeProductEnvEngine())
    assert adapter.capabilities().adapter_id == ADAPTER_ID

    class Sub(_FakeProductEnvEngine):
        pass

    assert registry.resolve(Sub()) is not None  # nearest registered base


def test_unregistered_engine_raises_capability_error():
    registry = AdapterRegistry()
    registry.freeze()
    with pytest.raises(CapabilityError):
        registry.resolve(object())


def test_duplicate_registration_and_frozen_registration_fail():
    registry = AdapterRegistry()
    registry.register("a.B", lambda: None)
    with pytest.raises(ValidationError):
        registry.register("a.B", lambda: None)
    registry.freeze()
    with pytest.raises(ValidationError):
        registry.register("c.D", lambda: None)


def test_default_registry_covers_engine_family_roots():
    registry = build_default_registry()
    expected = {
        "quantark.asset.equity.engine.base_engine.BaseEngine",
        "quantark.asset.fx.engine.base_fx_engine.BaseFxEngine",
        "quantark.asset.credit.engine.base_credit_engine.BaseCreditEngine",
        "quantark.asset.bond.engine.pde.convertible."
        "jump_diffusion_engine.ConvertibleBondJumpDiffusionEngine",
        "quantark.asset.bond.engine.pde.convertible.tf_engine.ConvertibleBondTFEngine",
        "quantark.asset.bond.engine.convertible."
        "convertible_bond_engine.ConvertibleBondEngine",
    }
    assert set(registry.registered_paths()) == expected


def test_legacy_adapter_price_dispatch_both_shapes():
    ctx = None  # execute_native does not need the context in Phase 0
    adapter_pe = LegacyPriceAdapter(call_shape="product_env")
    req = PricingRequest(product="P", pricing_env="E")
    norm = adapter_pe.normalize(_FakeProductEnvEngine(), req)
    assert norm.snapshot_complete is False and norm.fingerprint is None
    value, economics = adapter_pe.execute_native(
        _FakeProductEnvEngine(), req, norm, ctx
    )
    assert value == 42.0
    assert dict(economics)["pv"] == 42.0

    adapter_eb = LegacyPriceAdapter(call_shape="env_bound")
    req_eb = PricingRequest(product="bond")
    norm_eb = adapter_eb.normalize(_FakeEnvBoundEngine(), req_eb)
    value_eb, economics_eb = adapter_eb.execute_native(
        _FakeEnvBoundEngine(), req_eb, norm_eb, ctx
    )
    assert value_eb == 7.0


def test_legacy_adapter_rejects_unsupported_operation_and_output():
    adapter = LegacyPriceAdapter(call_shape="product_env")
    engine = _FakeProductEnvEngine()

    req_detailed = PricingRequest(
        product="P", pricing_env="E", operation=PricingOperation.PRICE_DETAILED
    )
    norm = adapter.normalize(engine, req_detailed)
    with pytest.raises(CapabilityError):
        adapter.execute_native(engine, req_detailed, norm, None)

    req_grid = PricingRequest(
        product="P", pricing_env="E",
        outputs=frozenset({OutputKind.PV, OutputKind.GRID}),
    )
    with pytest.raises(CapabilityError):
        adapter.validate(engine, req_grid)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest -n0 test/execution/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError` on `quantark.execution.registry`

- [ ] **Step 3: Write minimal implementation**

`quantark/execution/registry.py`:

```python
"""Adapter registry with string-keyed MRO resolution (spec section 6.2).

Engines are matched by walking ``type(engine).__mro__`` against registered
``"module.qualname"`` strings, so this module never imports asset, product,
or engine code. Registries freeze for the lifetime of a session; duplicate
or post-freeze registration is a validation error.

Resolution order (spec section 6.2): exact engine class first (MRO position
0), then the nearest registered base class in MRO order. Python's MRO is a
total order, so "nearest" is deterministic; ambiguity is prevented at
registration time by rejecting duplicate paths.
"""
from quantark.execution.errors import CapabilityError
from quantark.util.exceptions import ValidationError

__all__ = ["AdapterRegistry", "build_default_registry"]


class AdapterRegistry:
    def __init__(self):
        self._factories: dict = {}
        self._frozen = False

    def register(self, engine_class_path: str, adapter_factory) -> None:
        if self._frozen:
            raise ValidationError(
                "AdapterRegistry is frozen; register adapters before "
                "session construction"
            )
        if engine_class_path in self._factories:
            raise ValidationError(
                f"duplicate adapter registration for {engine_class_path}"
            )
        self._factories[engine_class_path] = adapter_factory

    def freeze(self) -> None:
        self._frozen = True

    def registered_paths(self) -> tuple:
        return tuple(self._factories)

    def resolve(self, engine):
        for cls in type(engine).__mro__:
            key = f"{cls.__module__}.{cls.__qualname__}"
            factory = self._factories.get(key)
            if factory is not None:
                return factory()
        raise CapabilityError(
            f"no execution adapter registered for engine type "
            f"{type(engine).__module__}.{type(engine).__qualname__}"
        )


_DEFAULT_REGISTRATIONS = (
    ("quantark.asset.equity.engine.base_engine.BaseEngine", "product_env"),
    ("quantark.asset.fx.engine.base_fx_engine.BaseFxEngine", "product_env"),
    (
        "quantark.asset.credit.engine.base_credit_engine.BaseCreditEngine",
        "product_env",
    ),
    (
        "quantark.asset.bond.engine.pde.convertible.jump_diffusion_engine."
        "ConvertibleBondJumpDiffusionEngine",
        "env_bound",
    ),
    (
        "quantark.asset.bond.engine.pde.convertible.tf_engine."
        "ConvertibleBondTFEngine",
        "env_bound",
    ),
    (
        "quantark.asset.bond.engine.convertible.convertible_bond_engine."
        "ConvertibleBondEngine",
        "env_bound",
    ),
)


def build_default_registry() -> AdapterRegistry:
    """Fresh registry with the serial compatibility adapter registered for
    every known engine-family root. Callers freeze it at session construction."""
    from quantark.execution.legacy_adapter import LegacyPriceAdapter

    registry = AdapterRegistry()
    for path, shape in _DEFAULT_REGISTRATIONS:
        registry.register(
            path,
            (lambda s: (lambda: LegacyPriceAdapter(call_shape=s)))(shape),
        )
    return registry
```

`quantark/execution/legacy_adapter.py`:

```python
"""Serial compatibility adapter: routes framework requests to legacy methods.

This is the terminal fallback of adapter resolution (spec section 6.2). It
performs no caching, no batching, and no parallel submission, so its shallow
normalized snapshot (``snapshot_complete=False``, no fingerprint) is safe:
the raw objects are used exactly as a direct legacy call would use them, on
the calling thread. Native legacy exceptions propagate unwrapped.

Serial-boundary note (spec sections 12.4, 17.1): "serial" here means the
FRAMEWORK submits no parallel work. Engine-internal legacy parallelism —
Snowball/Phoenix ``use_dask=True``, ``QUANTARK_DCN_MC_WORKERS`` threads —
is engine-owned preserved behavior and passes through unchanged, exactly as
a direct call would run it. It is outside the framework's (Phase-0-empty)
budget claims; later phases route it through budgeted plans. A session
therefore never alters, rejects, or silently disables an engine's own
configured execution.
"""
from quantark.execution.contracts import (
    DEFAULT_OUTPUTS,
    EngineCapabilities,
    NormalizedPricingRequest,
    OutputKind,
    PricingOperation,
    PricingRequest,
)
from quantark.execution.errors import CapabilityError

__all__ = ["ADAPTER_ID", "ADAPTER_VERSION", "LegacyPriceAdapter"]

ADAPTER_ID = "legacy-price"
ADAPTER_VERSION = "0"

_OP_OUTPUTS = {
    PricingOperation.PRICE: frozenset({OutputKind.PV}),
    PricingOperation.PRICE_DETAILED: frozenset(
        {OutputKind.PV, OutputKind.ERROR_ESTIMATE, OutputKind.CASHFLOWS}
    ),
    PricingOperation.EVENT_STATS: frozenset(
        {OutputKind.PV, OutputKind.EVENT_STATS, OutputKind.CASHFLOWS}
    ),
}


class LegacyPriceAdapter:
    def __init__(self, call_shape: str):
        if call_shape not in ("product_env", "env_bound"):
            raise CapabilityError(f"unknown call shape {call_shape!r}")
        self.call_shape = call_shape

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            operations=frozenset(_OP_OUTPUTS),
            output_kinds=frozenset(
                {OutputKind.PV, OutputKind.ERROR_ESTIMATE,
                 OutputKind.EVENT_STATS, OutputKind.CASHFLOWS}
            ),
            supported_backends=frozenset({"serial"}),
            fixed_planning=None,
            prepared_state_thread_safe=False,
            instance_reentrant=False,
            process_reconstructable=False,
            deterministic_reduction=True,
            peak_memory_estimate="unavailable",
            adapter_id=ADAPTER_ID,
            adapter_version=ADAPTER_VERSION,
        )

    def validate(self, engine, request: PricingRequest) -> None:
        allowed = _OP_OUTPUTS.get(request.operation)
        if allowed is None:
            raise CapabilityError(
                f"operation {request.operation} unsupported by {ADAPTER_ID}"
            )
        extra = request.outputs - allowed
        if extra:
            raise CapabilityError(
                f"outputs {sorted(k.value for k in extra)} unsupported for "
                f"operation {request.operation.value} via {ADAPTER_ID}"
            )
        if self.call_shape == "product_env" and request.pricing_env is None:
            raise CapabilityError(
                "pricing_env is required for product_env engines"
            )

    def normalize(self, engine, request: PricingRequest) -> NormalizedPricingRequest:
        cls = type(engine)
        return NormalizedPricingRequest(
            engine_class_path=f"{cls.__module__}.{cls.__qualname__}",
            operation=request.operation,
            outputs=request.outputs,
            operation_options=tuple(sorted(request.operation_options)),
            product_ref=request.product,
            pricing_env_ref=request.pricing_env,
            snapshot_complete=False,
            fingerprint=None,
        )

    def execute_native(self, engine, request, normalized, context):
        op = request.operation
        if op is PricingOperation.PRICE:
            value = self._call_price(engine, request)
            return value, (("pv", float(value)),)
        if op is PricingOperation.PRICE_DETAILED:
            value = self._call_detailed(engine, request)
            return value, (("pv", _extract_pv(value)),)
        if op is PricingOperation.EVENT_STATS:
            value = self._call_event_stats(engine, request)
            return value, (("pv", _extract_pv(value)),)
        raise CapabilityError(f"operation {op} unsupported by {ADAPTER_ID}")

    def _call_price(self, engine, request):
        if self.call_shape == "env_bound":
            return engine.price(request.product)
        return engine.price(request.product, request.pricing_env)

    def _call_detailed(self, engine, request):
        if self.call_shape == "env_bound":
            method = getattr(engine, "price_with_details", None)
            if method is None:
                raise CapabilityError(
                    f"{type(engine).__qualname__} has no price_with_details"
                )
            return method(request.product)
        method = getattr(engine, "price_detailed", None)
        if method is None:
            raise CapabilityError(
                f"{type(engine).__qualname__} has no price_detailed"
            )
        return method(request.product, request.pricing_env)

    def _call_event_stats(self, engine, request):
        method = getattr(engine, "calculate_event_stats", None)
        if method is None:
            raise CapabilityError(
                f"{type(engine).__qualname__} has no calculate_event_stats"
            )
        stats = method(request.product, request.pricing_env)
        if stats is None:
            raise CapabilityError(
                f"{type(engine).__qualname__} does not support event stats"
            )
        return stats


def _extract_pv(value):
    for attr in ("pv", "price", "npv"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, (int, float)):
            return float(candidate)
    if isinstance(value, (int, float)):
        return float(value)
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest -n0 test/execution/test_registry.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add quantark/execution/registry.py quantark/execution/legacy_adapter.py test/execution/test_registry.py
git commit -m "feat(execution): string-keyed adapter registry and LegacyPriceAdapter"
```

---

### Task 7: Kernel, session, `BaseEngine.execute`, public exports

**Files:**
- Create: `quantark/execution/kernel.py`
- Create: `quantark/execution/api.py`
- Modify: `quantark/execution/__init__.py` (public exports)
- Modify: `quantark/asset/equity/engine/base_engine.py` (append `execute` method
  after `create_bump_context`, around line 66 — no other change)
- Test: `test/execution/test_session_parity.py` (first tests; parity matrix
  extended in Task 9)

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces: `ExecutionKernel.dispatch(engine, request, context) -> PricingOutcome`
  — resolves the adapter from `context.adapter_registry` (falling back to a
  lazily-built, frozen module-default registry), validates capability/backend,
  normalizes, executes, assembles manifest + diagnostics, emits to the sink.
- Produces: `PricingSession(context=None)` with `execute(engine, request)`,
  `price(engine, product, pricing_env=None)`, `price_many(items, collect_errors=False)`
  (`items` = sequence of `(engine, PricingRequest)` pairs; order preserved),
  `run_scenarios(...)` raising `CapabilityError` (arrives Phase 5), context
  manager + idempotent `close()`.
- Produces: `BaseEngine.execute(request, context) -> PricingOutcome` (lazy
  import of the kernel; spec §5.4).
- `quantark.execution.__init__` exports the spec §5 surface:
  `DeterminismPolicy, EngineCapabilities, ExecutionPolicy, PricingFailure,
  PricingOutcome, PricingRequest, PricingRunContext, PricingSession,
  ResourceBudget, ScenarioOutcome, ScenarioSpec` plus `PricingOperation,
  OutputKind` and the six errors.

- [ ] **Step 1: Write the failing test**

```python
# test/execution/test_session_parity.py
"""Session-vs-direct parity and the BaseEngine.execute seam (spec sections 5.4, 5.5)."""
import pytest

from quantark.asset.equity.engine.mc import EuropeanMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.execution import (
    OutputKind,
    PricingOperation,
    PricingRequest,
    PricingSession,
)
from quantark.execution.contracts import economics_mapping
from quantark.execution.errors import CapabilityError
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType


@pytest.fixture()
def equity_env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
    )


@pytest.fixture()
def european_option():
    return EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )


def _mc_engine():
    return EuropeanMCEngine(params=MCParams(num_paths=2000, seed=42))


def test_session_price_equals_direct_price(equity_env, european_option):
    direct = _mc_engine().price(european_option, equity_env)
    with PricingSession() as session:
        via_session = session.price(_mc_engine(), european_option, equity_env)
    assert via_session == direct  # exact: same numerical plan, same code path


def test_execute_returns_outcome_with_manifest(equity_env, european_option):
    with PricingSession() as session:
        outcome = session.execute(
            _mc_engine(),
            PricingRequest(product=european_option, pricing_env=equity_env),
        )
    assert isinstance(outcome.value, float)
    assert economics_mapping(outcome)["pv"] == outcome.value
    assert outcome.manifest.adapter_id == "legacy-price"
    assert outcome.manifest.request_fingerprint is None
    assert dict(outcome.manifest.versions)["numpy"]
    assert outcome.diagnostics.adapter_id == "legacy-price"


def test_base_engine_execute_method(equity_env, european_option):
    from quantark.execution.context import default_context

    engine = _mc_engine()
    outcome = engine.execute(
        PricingRequest(product=european_option, pricing_env=equity_env),
        default_context(),
    )
    assert outcome.value == engine.price(european_option, equity_env)


def test_price_many_preserves_order_and_types(equity_env, european_option):
    put = EuropeanVanillaOption(
        strike=110.0, option_type=OptionType.PUT, maturity=1.0
    )
    items = [
        (_mc_engine(), PricingRequest(product=european_option, pricing_env=equity_env)),
        (_mc_engine(), PricingRequest(product=put, pricing_env=equity_env)),
    ]
    with PricingSession() as session:
        values = session.price_many(items)
    assert len(values) == 2 and all(isinstance(v, float) for v in values)
    assert values[0] == _mc_engine().price(european_option, equity_env)


def test_price_many_collect_errors(equity_env, european_option):
    from quantark.execution import PricingFailure

    class _Boom:
        def price(self, product, env):
            raise ValueError("boom")

    # _Boom is not a registered engine family -> CapabilityError, collected.
    items = [
        (_mc_engine(), PricingRequest(product=european_option, pricing_env=equity_env)),
        (_Boom(), PricingRequest(product=european_option, pricing_env=equity_env)),
    ]
    with PricingSession() as session:
        results = session.price_many(items, collect_errors=True)
    assert isinstance(results[0], float)
    assert isinstance(results[1], PricingFailure)
    assert results[1].error.error_type == "CapabilityError"


def test_non_serial_backend_raises_capability_error(equity_env, european_option):
    from quantark.execution import ExecutionPolicy
    from quantark.execution.context import default_context
    from quantark.execution.policy import ExecutorSelection
    import dataclasses

    ctx = dataclasses.replace(
        default_context(),
        execution_policy=ExecutionPolicy(
            batch=ExecutorSelection(backend="threads", workers=4)
        ),
    )
    with PricingSession(ctx) as session:
        with pytest.raises(CapabilityError):
            session.price(_mc_engine(), european_option, equity_env)


def test_run_scenarios_is_phase5(equity_env, european_option):
    with PricingSession() as session:
        with pytest.raises(CapabilityError):
            session.run_scenarios(
                PricingRequest(product=european_option, pricing_env=equity_env),
                scenario_specs=(),
                engine_factory=_mc_engine,
            )


def test_unsupported_output_raises(equity_env, european_option):
    req = PricingRequest(
        product=european_option,
        pricing_env=equity_env,
        outputs=frozenset({OutputKind.PV, OutputKind.GRID}),
    )
    with PricingSession() as session:
        with pytest.raises(CapabilityError):
            session.execute(_mc_engine(), req)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest -n0 test/execution/test_session_parity.py -v`
Expected: FAIL with `ImportError` (no `PricingSession` in `quantark.execution`)

- [ ] **Step 3: Write minimal implementation**

`quantark/execution/kernel.py`:

```python
"""Canonical execution lifecycle, serial subset (spec section 7).

Phase 0 implements lifecycle steps: validate -> normalize -> execute ->
normalize output -> emit diagnostics/manifest. Resource leases, plan
fingerprints, and parallel backends arrive in later phases. The kernel never
statically imports asset code; engines reach it via ``BaseEngine.execute``'s
lazy import or through ``PricingSession``.
"""
import time

from quantark.execution.contracts import PricingOutcome
from quantark.execution.diagnostics import RunDiagnostics
from quantark.execution.errors import CapabilityError
from quantark.execution.manifest import (
    MANIFEST_SCHEMA_VERSION,
    ReproducibilityManifest,
    build_versions,
    platform_tag,
)
from quantark.execution.registry import build_default_registry

__all__ = ["ExecutionKernel"]

_module_default_registry = None


def _default_registry():
    global _module_default_registry
    if _module_default_registry is None:
        registry = build_default_registry()
        registry.freeze()
        _module_default_registry = registry
    return _module_default_registry


class ExecutionKernel:
    @staticmethod
    def dispatch(engine, request, context) -> PricingOutcome:
        registry = context.adapter_registry or _default_registry()
        adapter = registry.resolve(engine)
        caps = adapter.capabilities()

        backend = context.execution_policy.batch.backend
        if backend not in caps.supported_backends:
            raise CapabilityError(
                f"backend {backend!r} not supported by adapter "
                f"{caps.adapter_id!r}; supported: "
                f"{sorted(caps.supported_backends)}"
            )
        adapter.validate(engine, request)

        normalized = adapter.normalize(engine, request)
        start = time.perf_counter()
        value, economics = adapter.execute_native(
            engine, request, normalized, context
        )
        elapsed = time.perf_counter() - start

        manifest = ReproducibilityManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            request_fingerprint=normalized.fingerprint,
            plan_fingerprint=None,
            adapter_id=caps.adapter_id,
            adapter_version=caps.adapter_version,
            engine_class_path=normalized.engine_class_path,
            versions=build_versions(),
            platform=platform_tag(),
            resolved_policy=context.config_snapshot,
        )
        diagnostics = RunDiagnostics(
            adapter_id=caps.adapter_id,
            timings=(("execute_seconds", elapsed),),
            policy_sources=context.config_snapshot,
            records=(),
        )
        outcome = PricingOutcome(
            value=value,
            normalized_economics=economics,
            diagnostics=diagnostics,
            manifest=manifest,
        )
        sink = context.diagnostics_sink
        if sink is not None:
            sink.emit(diagnostics)
        return outcome
```

`quantark/execution/api.py`:

```python
"""PricingSession: the explicit framework entry point (spec section 5.5)."""
from quantark.execution.contracts import (
    FrameworkErrorInfo,
    PricingFailure,
    PricingRequest,
)
from quantark.execution.context import PricingRunContext, default_context
from quantark.execution.errors import CapabilityError
from quantark.execution.kernel import ExecutionKernel
from quantark.execution.registry import build_default_registry

__all__ = ["PricingSession"]


class PricingSession:
    """Serial pricing session. Owns only services it creates; idempotently
    closable. ``PricingSession()`` with no context resolves a safe serial
    default exactly once at construction (spec section 11.1)."""

    def __init__(self, context: PricingRunContext | None = None):
        if context is None:
            context = default_context()
        if context.adapter_registry is None:
            import dataclasses

            registry = build_default_registry()
            registry.freeze()
            context = dataclasses.replace(context, adapter_registry=registry)
        self._context = context
        self._closed = False

    @property
    def context(self) -> PricingRunContext:
        return self._context

    def execute(self, engine, request: PricingRequest):
        self._ensure_open()
        return ExecutionKernel.dispatch(engine, request, self._context)

    def price(self, engine, product, pricing_env=None):
        outcome = self.execute(
            engine, PricingRequest(product=product, pricing_env=pricing_env)
        )
        return outcome.value

    def price_many(self, items, *, collect_errors: bool = False) -> list:
        """Serial, caller-ordered pricing of (engine, PricingRequest) pairs.

        Fail-fast by default; ``collect_errors=True`` returns a
        ``PricingFailure`` in place of each failed item (spec section 15).
        """
        self._ensure_open()
        results: list = []
        for index, (engine, request) in enumerate(items):
            try:
                results.append(self.execute(engine, request).value)
            except Exception as exc:  # noqa: BLE001 - typed into the failure record
                if not collect_errors:
                    raise
                item_id = request.request_id or str(index)
                from quantark.execution.diagnostics import RunDiagnostics

                results.append(
                    PricingFailure(
                        item_id=item_id,
                        error=FrameworkErrorInfo(
                            error_type=type(exc).__name__, message=str(exc)
                        ),
                        diagnostics=RunDiagnostics(adapter_id="unresolved"),
                    )
                )
        return results

    def run_scenarios(self, base_request, scenario_specs, engine_factory):
        raise CapabilityError(
            "scenario execution is not available in framework Phase 0; "
            "it arrives with the scenario planner in Phase 5"
        )

    def close(self) -> None:
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise CapabilityError("PricingSession is closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
```

`quantark/execution/__init__.py` (replace placeholder):

```python
"""QuantArk composable execution kernel (framework contract v1, Phase 0).

Additive public surface (spec section 5). Direct legacy engine calls are
unchanged; this package is reached only through explicit sessions or
``BaseEngine.execute``.
"""
from quantark.execution.api import PricingSession
from quantark.execution.context import PricingRunContext, default_context
from quantark.execution.contracts import (
    DEFAULT_OUTPUTS,
    EngineCapabilities,
    FrameworkErrorInfo,
    NormalizedPricingRequest,
    OutputKind,
    PricingFailure,
    PricingOperation,
    PricingOutcome,
    PricingRequest,
    ScenarioOutcome,
    ScenarioSpec,
)
from quantark.execution.errors import (
    CapabilityError,
    DeterminismViolation,
    PreparationError,
    ResourceBudgetExceeded,
    TaskExecutionError,
    ValidationGateError,
)
from quantark.execution.policy import (
    DeterminismPolicy,
    ExecutionPolicy,
    ExecutorSelection,
    ResourceBudget,
)

__all__ = [
    "DEFAULT_OUTPUTS",
    "CapabilityError",
    "DeterminismPolicy",
    "DeterminismViolation",
    "EngineCapabilities",
    "ExecutionPolicy",
    "ExecutorSelection",
    "FrameworkErrorInfo",
    "NormalizedPricingRequest",
    "OutputKind",
    "PreparationError",
    "PricingFailure",
    "PricingOperation",
    "PricingOutcome",
    "PricingRequest",
    "PricingRunContext",
    "PricingSession",
    "ResourceBudget",
    "ResourceBudgetExceeded",
    "ScenarioOutcome",
    "ScenarioSpec",
    "TaskExecutionError",
    "ValidationGateError",
    "default_context",
]
```

`quantark/asset/equity/engine/base_engine.py` — insert directly after the
`create_bump_context` method (after its `return self`, before
`price_with_events`):

```python
    def execute(self, request, context):
        """Route a framework ``PricingRequest`` through the execution kernel.

        Non-abstract compatibility entry point (execution-framework spec
        section 5.4). Existing subclasses need no change; the kernel resolves
        a capability adapter for this engine and falls back to the serial
        LegacyPriceAdapter. Direct ``price``/``price_detailed`` calls are
        unaffected.
        """
        from quantark.execution.kernel import ExecutionKernel

        return ExecutionKernel.dispatch(self, request, context)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest -n0 test/execution/test_session_parity.py -v`
Expected: 9 PASS

- [ ] **Step 5: Run the full execution test suite plus equity engine smoke tests**

Run: `.venv/bin/python -m pytest test/execution/ test/test_greeks_mode_and_engine_type.py -q`
Expected: all PASS (proves the `BaseEngine` touch changed no legacy behavior)

- [ ] **Step 6: Commit**

```bash
git add quantark/execution/ quantark/asset/equity/engine/base_engine.py test/execution/test_session_parity.py
git commit -m "feat(execution): serial kernel, PricingSession, and BaseEngine.execute seam"
```

---

### Task 8: Checked-in engine inventory and discovery

**Files:**
- Create: `quantark/execution/inventory.py`
- Test: `test/execution/test_inventory.py`

**Interfaces:**
- Produces: `InventoryRecord(name, import_path, engine_type, asset_family,
  model_family, product_family, planning, call_shape, role, backends,
  adoption_state, owner=None, milestone=None, reason=None)` — frozen.
  - `engine_type`: `"mc" | "pde"`; `role`: `"engine" | "facade" | "abstract"`;
    `planning`: `"fixed" | "both"`; `adoption_state`:
    `"supported" | "not_applicable" | "temporary_legacy"`.
- Produces: `ENGINE_INVENTORY: tuple[InventoryRecord, ...]` (74 rows, below).
- Produces: `SUPPORTING_EXPORTS: dict[str, tuple[str, ...]]` — non-engine
  names per surface (results, grids, params, helpers) so discovery can prove
  every exported name is classified.
- Produces: `DISCOVERY_SURFACES: tuple[str, ...]` — module paths whose
  `__all__` is unioned, plus `EXPLICIT_FACADES` for `PDEEngine` and
  `ConvertibleBondEngine`.
- Produces: `discover_exported_engine_names() -> dict[str, tuple[str, ...]]`
  (dynamic import at call time only) and
  `inventory_by_name() -> dict[str, InventoryRecord]`.

- [ ] **Step 1: Write the failing test**

```python
# test/execution/test_inventory.py
"""Checked-in engine inventory and CI discovery gate (spec section 18)."""
import importlib

import pytest

from quantark.execution.inventory import (
    DISCOVERY_SURFACES,
    ENGINE_INVENTORY,
    EXPLICIT_FACADES,
    SUPPORTING_EXPORTS,
    discover_exported_engine_names,
    inventory_by_name,
)
from quantark.execution.registry import build_default_registry


def test_every_public_export_is_classified():
    """CI gate: a new public MC/PDE export must be inventoried or classified
    as supporting; otherwise this test fails (spec section 18)."""
    discovered = discover_exported_engine_names()
    inventoried = set(inventory_by_name())
    for surface, names in discovered.items():
        supporting = set(SUPPORTING_EXPORTS.get(surface, ()))
        for name in names:
            assert name in inventoried or name in supporting, (
                f"public export {surface}:{name} is neither inventoried "
                f"nor classified as supporting"
            )


def test_inventory_names_exist_and_import():
    for record in ENGINE_INVENTORY:
        module_path, _, class_name = record.import_path.rpartition(".")
        module = importlib.import_module(module_path)
        assert hasattr(module, class_name), record.import_path


def test_inventory_counts_match_spec_snapshot():
    by_family = {}
    for r in ENGINE_INVENTORY:
        by_family.setdefault((r.asset_family, r.engine_type, r.role), []).append(r)
    assert len(by_family[("equity", "mc", "engine")]) == 33
    assert len(by_family[("equity", "pde", "engine")]) == 24
    assert len(by_family[("equity", "pde", "abstract")]) == 1  # BasePDESolver
    assert len(by_family[("equity", "pde", "facade")]) == 1  # PDEEngine
    assert len(by_family[("fx", "mc", "engine")]) == 8
    assert len(by_family[("fx", "pde", "engine")]) == 3
    assert len(by_family[("credit", "mc", "engine")]) == 1
    assert len(by_family[("bond", "pde", "engine")]) == 2
    assert len(by_family[("bond", "pde", "facade")]) == 1


def test_temporary_legacy_rows_have_owner_and_milestone():
    for record in ENGINE_INVENTORY:
        if record.adoption_state == "temporary_legacy":
            assert record.owner and record.milestone, record.name
        if record.adoption_state == "not_applicable":
            assert record.reason, record.name


def test_every_concrete_engine_is_session_reachable():
    """Phase 0 exit gate: for every concrete inventoried engine class, the
    default registry resolves a serial adapter AND the class exposes a
    ``price`` callable whose arity matches the declared call shape.

    Full direct-versus-session numerical parity for every row is Phase 1's
    exit gate (spec section 21, Phase 1: "direct versus session parity
    across the matrix"); Phase 0 proves reachability plus the representative
    per-family parity matrix in test_session_parity.py.
    """
    import inspect

    registry = build_default_registry()
    registry.freeze()
    for record in ENGINE_INVENTORY:
        if record.role == "abstract":
            continue
        module_path, _, class_name = record.import_path.rpartition(".")
        cls = getattr(importlib.import_module(module_path), class_name)
        fake = cls.__new__(cls)  # resolution is type-based; no construction
        adapter = registry.resolve(fake)
        assert adapter.capabilities().adapter_id == "legacy-price", record.name
        assert adapter.call_shape == record.call_shape, record.name
        price = getattr(cls, "price", None)
        assert callable(price), f"{record.name} has no price method"
        params = [
            p for p in inspect.signature(price).parameters.values()
            if p.name != "self" and p.kind is not p.VAR_KEYWORD
        ]
        required = [p for p in params if p.default is p.empty]
        expected_args = 1 if record.call_shape == "env_bound" else 2
        assert len(required) <= expected_args <= len(params), (
            f"{record.name}: price arity {len(params)} does not fit "
            f"declared call shape {record.call_shape}"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest -n0 test/execution/test_inventory.py -v`
Expected: FAIL with `ModuleNotFoundError` on `quantark.execution.inventory`

- [ ] **Step 3: Write the inventory module**

`quantark/execution/inventory.py` — the checked-in source of truth. Static
data only; the discovery function performs dynamic imports at call time.

```python
"""Checked-in exported-engine capability inventory (spec section 18).

This module is the release-gate source of truth. It contains no static
imports of asset code; ``discover_exported_engine_names`` imports the public
surfaces dynamically when called (from tests/CI only).

Phase 0: every row is ``temporary_legacy`` (session-reachable through the
serial LegacyPriceAdapter) except abstract bases, which are
``not_applicable``. Later phases upgrade rows to ``supported`` as native
adapters land.
"""
import importlib
from dataclasses import dataclass

__all__ = [
    "InventoryRecord",
    "ENGINE_INVENTORY",
    "SUPPORTING_EXPORTS",
    "DISCOVERY_SURFACES",
    "EXPLICIT_FACADES",
    "discover_exported_engine_names",
    "inventory_by_name",
]

_OWNER = "execution-framework"
_MILESTONE = "phase-1+"


@dataclass(frozen=True)
class InventoryRecord:
    name: str
    import_path: str
    engine_type: str      # "mc" | "pde"
    asset_family: str     # "equity" | "fx" | "credit" | "bond"
    model_family: str     # "bsm" | "lv" | "heston" | "slv" | "sabr" | "copula" | "jump_diffusion" | "dispatch"
    product_family: str
    planning: str         # "fixed" | "both" (fixed + adaptive RQMC mode)
    call_shape: str       # "product_env" | "env_bound"
    role: str = "engine"  # "engine" | "facade" | "abstract"
    backends: tuple = ("serial",)
    adoption_state: str = "temporary_legacy"
    owner: str | None = _OWNER
    milestone: str | None = _MILESTONE
    reason: str | None = None


def _eq_mc(name, model, product, planning="fixed"):
    return InventoryRecord(
        name=name,
        import_path=f"quantark.asset.equity.engine.mc.{name}",
        engine_type="mc", asset_family="equity", model_family=model,
        product_family=product, planning=planning, call_shape="product_env",
    )


def _eq_pde(name, model, product, role="engine"):
    return InventoryRecord(
        name=name,
        import_path=f"quantark.asset.equity.engine.pde.{name}",
        engine_type="pde", asset_family="equity", model_family=model,
        product_family=product, planning="fixed", call_shape="product_env",
        role=role,
        adoption_state="not_applicable" if role == "abstract" else "temporary_legacy",
        owner=None if role == "abstract" else _OWNER,
        milestone=None if role == "abstract" else _MILESTONE,
        reason="abstract base class" if role == "abstract" else None,
    )


def _fx(name, engine_type, model, product):
    sub = "mc" if engine_type == "mc" else "pde"
    return InventoryRecord(
        name=name,
        import_path=f"quantark.asset.fx.engine.{sub}.{name}",
        engine_type=engine_type, asset_family="fx", model_family=model,
        product_family=product, planning="fixed", call_shape="product_env",
    )


ENGINE_INVENTORY = (
    # --- Equity MC (33 engines) ---
    _eq_mc("EuropeanMCEngine", "bsm", "vanilla"),
    _eq_mc("LocalVolMCEngine", "lv", "vanilla"),
    _eq_mc("HestonMCEngine", "heston", "vanilla"),
    _eq_mc("HestonSLVMCEngine", "slv", "vanilla"),
    _eq_mc("SABRMCEngine", "sabr", "vanilla"),
    _eq_mc("AmericanOptionMCEngine", "bsm", "american"),
    _eq_mc("AsianOptionMCEngine", "bsm", "asian"),
    _eq_mc("DigitalOptionMCEngine", "bsm", "digital"),
    _eq_mc("BarrierOptionMCEngine", "bsm", "barrier"),
    _eq_mc("LocalVolBarrierMCEngine", "lv", "barrier"),
    _eq_mc("HestonBarrierMCEngine", "heston", "barrier"),
    _eq_mc("HestonSLVBarrierMCEngine", "slv", "barrier"),
    _eq_mc("SingleSharkfinOptionMCEngine", "bsm", "sharkfin"),
    _eq_mc("DoubleSharkfinOptionMCEngine", "bsm", "sharkfin"),
    _eq_mc("RangeAccrualMCEngine", "bsm", "range_accrual"),
    _eq_mc("AccumulatorMCEngine", "bsm", "accumulator"),
    _eq_mc("SnowballMCEngine", "bsm", "snowball", planning="both"),
    _eq_mc("LocalVolSnowballMCEngine", "lv", "snowball", planning="both"),
    _eq_mc("HestonSnowballMCEngine", "heston", "snowball", planning="both"),
    _eq_mc("QESnowballMCEngine", "heston", "snowball", planning="both"),
    _eq_mc("HestonSLVSnowballMCEngine", "slv", "snowball", planning="both"),
    _eq_mc("HestonSLVQESnowballMCEngine", "slv", "snowball", planning="both"),
    _eq_mc("PhoenixMCEngine", "bsm", "phoenix", planning="both"),
    _eq_mc("LocalVolPhoenixMCEngine", "lv", "phoenix", planning="both"),
    _eq_mc("HestonPhoenixMCEngine", "heston", "phoenix", planning="both"),
    _eq_mc("QEPhoenixMCEngine", "heston", "phoenix", planning="both"),
    _eq_mc("HestonSLVPhoenixMCEngine", "slv", "phoenix", planning="both"),
    _eq_mc("HestonSLVQEPhoenixMCEngine", "slv", "phoenix", planning="both"),
    _eq_mc("DCNMCEngine", "bsm", "dcn", planning="both"),
    _eq_mc("LocalVolDCNMCEngine", "lv", "dcn", planning="both"),
    _eq_mc("HestonDCNMCEngine", "heston", "dcn", planning="both"),
    _eq_mc("QEDCNMCEngine", "heston", "dcn", planning="both"),
    _eq_mc("CoupledCoarseHestonDCNMCEngine", "heston", "dcn", planning="both"),
    # --- Equity PDE (24 concrete + abstract base) ---
    _eq_pde("BasePDESolver", "dispatch", "any", role="abstract"),
    _eq_pde("EuropeanPDESolver", "bsm", "vanilla"),
    _eq_pde("AmericanPDESolver", "bsm", "american"),
    _eq_pde("BarrierPDESolver", "bsm", "barrier"),
    _eq_pde("DoubleBarrierPDESolver", "bsm", "double_barrier"),
    _eq_pde("OneTouchPDESolver", "bsm", "one_touch"),
    _eq_pde("DoubleOneTouchPDESolver", "bsm", "one_touch"),
    _eq_pde("SnowballPDESolver", "bsm", "snowball"),
    _eq_pde("KOResetSnowballPDESolver", "bsm", "ko_reset_snowball"),
    _eq_pde("PhoenixPDESolver", "bsm", "phoenix"),
    _eq_pde("LocalVolPDESolver", "lv", "vanilla"),
    _eq_pde("HestonPDESolver", "heston", "vanilla"),
    _eq_pde("HestonSLVPDESolver", "slv", "vanilla"),
    _eq_pde("LocalVolBarrierPDESolver", "lv", "barrier"),
    _eq_pde("HestonBarrierPDESolver", "heston", "barrier"),
    _eq_pde("HestonSLVBarrierPDESolver", "slv", "barrier"),
    _eq_pde("LocalVolSnowballPDESolver", "lv", "snowball"),
    _eq_pde("HestonSnowballPDESolver", "heston", "snowball"),
    _eq_pde("HestonSLVSnowballPDESolver", "slv", "snowball"),
    _eq_pde("LocalVolPhoenixPDESolver", "lv", "phoenix"),
    _eq_pde("HestonPhoenixPDESolver", "heston", "phoenix"),
    _eq_pde("HestonSLVPhoenixPDESolver", "slv", "phoenix"),
    _eq_pde("DCNPDEEngine", "bsm", "dcn"),
    _eq_pde("LocalVolDCNPDEEngine", "lv", "dcn"),
    _eq_pde("HestonDCNPDESolver", "heston", "dcn"),
    # --- Equity facade ---
    InventoryRecord(
        name="PDEEngine",
        import_path="quantark.asset.equity.engine.PDEEngine",
        engine_type="pde", asset_family="equity", model_family="dispatch",
        product_family="any", planning="fixed", call_shape="product_env",
        role="facade",
    ),
    # --- FX MC (8) ---
    _fx("FxLocalVolMCEngine", "mc", "lv", "vanilla"),
    _fx("FxHestonMCEngine", "mc", "heston", "vanilla"),
    _fx("FxHestonSLVMCEngine", "mc", "slv", "vanilla"),
    _fx("FxRangeAccrualMCEngine", "mc", "bsm", "range_accrual"),
    _fx("FxBarrierMCEngine", "mc", "bsm", "barrier"),
    _fx("FxSharkfinMCEngine", "mc", "bsm", "sharkfin"),
    _fx("FxTarnForwardMCEngine", "mc", "bsm", "tarn"),
    _fx("FxTargetRedemptionNoteMCEngine", "mc", "bsm", "tarn"),
    # --- FX PDE (3) ---
    _fx("FxLocalVolPDESolver", "pde", "lv", "vanilla"),
    _fx("FxHestonPDESolver", "pde", "heston", "vanilla"),
    _fx("FxHestonSLVPDESolver", "pde", "slv", "vanilla"),
    # --- Credit MC (1) ---
    InventoryRecord(
        name="BasketCDSEngine",
        import_path="quantark.asset.credit.engine.mc.BasketCDSEngine",
        engine_type="mc", asset_family="credit", model_family="copula",
        product_family="basket_cds", planning="fixed",
        call_shape="product_env",
    ),
    # --- Bond PDE (2 + facade) ---
    InventoryRecord(
        name="ConvertibleBondJumpDiffusionEngine",
        import_path=(
            "quantark.asset.bond.engine.pde.ConvertibleBondJumpDiffusionEngine"
        ),
        engine_type="pde", asset_family="bond",
        model_family="jump_diffusion", product_family="convertible",
        planning="fixed", call_shape="env_bound",
    ),
    InventoryRecord(
        name="ConvertibleBondTFEngine",
        import_path="quantark.asset.bond.engine.pde.ConvertibleBondTFEngine",
        engine_type="pde", asset_family="bond", model_family="bsm",
        product_family="convertible", planning="fixed",
        call_shape="env_bound",
    ),
    InventoryRecord(
        name="ConvertibleBondEngine",
        import_path="quantark.asset.bond.engine.ConvertibleBondEngine",
        engine_type="pde", asset_family="bond", model_family="dispatch",
        product_family="convertible", planning="fixed",
        call_shape="env_bound", role="facade",
    ),
)


DISCOVERY_SURFACES = (
    "quantark.asset.equity.engine.mc",
    "quantark.asset.equity.engine.pde",
    "quantark.asset.fx.engine.mc",
    "quantark.asset.fx.engine.pde",
    "quantark.asset.credit.engine.mc",
    "quantark.asset.bond.engine.pde",
)

# Facades exported from wider (mixed analytical/quad) surfaces are named
# explicitly rather than discovered by __all__ union.
EXPLICIT_FACADES = (
    ("quantark.asset.equity.engine", "PDEEngine"),
    ("quantark.asset.bond.engine", "ConvertibleBondEngine"),
)

SUPPORTING_EXPORTS = {
    "quantark.asset.equity.engine.mc": (
        "AmericanMCResult", "DCNMCResult", "PhoenixMCResult", "AsianMCResult",
        "RangeAccrualMCResult", "coupled_heston_ladder_pair",
    ),
    "quantark.asset.equity.engine.pde": (
        "TimeGrid", "SpatialGrid", "DCNPDEResult",
    ),
    "quantark.asset.fx.engine.mc": (
        "FxRangeAccrualMCResult", "FxBarrierMCResult", "FxSharkfinMCResult",
        "FxTarnMCResult",
    ),
    "quantark.asset.fx.engine.pde": (),
    "quantark.asset.credit.engine.mc": (),
    "quantark.asset.bond.engine.pde": ("ConvertibleBondPDEParams",),
}


def discover_exported_engine_names() -> dict:
    """Union the public ``__all__`` of every discovery surface (dynamic
    import at call time; used by tests/CI, never at library import time)."""
    discovered = {}
    for surface in DISCOVERY_SURFACES:
        module = importlib.import_module(surface)
        discovered[surface] = tuple(module.__all__)
    for module_path, facade_name in EXPLICIT_FACADES:
        module = importlib.import_module(module_path)
        assert facade_name in module.__all__, (module_path, facade_name)
        discovered.setdefault(module_path, ())
        discovered[module_path] = discovered[module_path] + (facade_name,)
    return discovered


def inventory_by_name() -> dict:
    return {record.name: record for record in ENGINE_INVENTORY}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest -n0 test/execution/test_inventory.py -v`
Expected: 5 PASS. If `test_every_public_export_is_classified` fails, the
failure message names the exact export to classify — add it to the correct
list rather than loosening the test.

- [ ] **Step 5: Commit**

```bash
git add quantark/execution/inventory.py test/execution/test_inventory.py
git commit -m "feat(execution): checked-in engine inventory with CI discovery gate"
```

---

### Task 9: Cross-family parity matrix and golden fixtures

**Files:**
- Create: `test/execution/freeze_goldens.py`
- Create: `test/execution/goldens/phase0_goldens.json` (generated by the script, checked in)
- Modify: `test/execution/test_session_parity.py` (append cross-family tests)

**Interfaces:**
- Consumes: `PricingSession` (Task 7); representative engines across all four
  asset families.
- Produces: `build_representative_cases() -> dict[str, tuple]` in
  `freeze_goldens.py`, mapping case name to
  `(engine, product, env_or_None, call_shape)`; reused by both the freeze
  script and the parity test.

- [ ] **Step 1: Write the shared fixture module and freeze script**

```python
# test/execution/freeze_goldens.py
"""Representative cross-family fixtures + golden freeze script (Phase 0).

Run once to (re)generate the checked-in goldens:
    .venv/bin/python -m test.execution.freeze_goldens
Goldens protect later phases against silent serial-path changes; they are
same-machine, version-stamped references, not cross-platform bit claims.
"""
import json
import pathlib
from datetime import datetime

GOLDEN_PATH = pathlib.Path(__file__).parent / "goldens" / "phase0_goldens.json"


def build_representative_cases() -> dict:
    import numpy as np

    from quantark.asset.bond.engine.pde import (
        ConvertibleBondPDEParams,
        ConvertibleBondTFEngine,
    )
    from quantark.asset.bond.product import ConvertibleBond
    from quantark.asset.credit.engine.mc import BasketCDSEngine
    from quantark.asset.credit.product import BasketCDS, BasketType
    from quantark.asset.equity.engine.mc import EuropeanMCEngine
    from quantark.asset.equity.engine.pde import EuropeanPDESolver
    from quantark.asset.equity.param import MCParams, PDEParams
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.asset.fx.engine.mc import FxBarrierMCEngine
    from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
    from quantark.asset.fx.product.option import FxBarrierOption
    from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.param.credit import FlatHazardCurve
    from quantark.priceenv import (
        BasketCreditPricingEnvironment,
        FxPricingEnvironment,
        PricingEnvironment,
    )
    from quantark.util.enum import FxBarrierType, OptionType
    from quantark.util.currency import CurrencyPair

    eq_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
    )
    eq_option = EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )

    fx_env = FxPricingEnvironment(
        valuation_date=datetime(2026, 6, 15),
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=FlatVolSurface(volatility=0.10),
    )
    fx_option = FxBarrierOption(
        strike=1.20, barrier=1.35, is_up=True,
        knock_type=FxBarrierType.KNOCK_OUT, option_type=OptionType.CALL,
        currency_pair=CurrencyPair("EUR", "USD"), maturity=1.0,
    )

    n_names = 5
    corr = np.full((n_names, n_names), 0.3)
    np.fill_diagonal(corr, 1.0)
    cds_product = BasketCDS(
        notional=10_000_000.0, maturity=5.0,
        recovery_rates=[0.4] * n_names, basket_type=BasketType.FTD,
        n_to_default=1, correlation_matrix=corr,
    )
    cds_env = BasketCreditPricingEnvironment(
        valuation_date=datetime(2026, 6, 13),
        discount_curve=FlatRateCurve(rate=0.03),
        hazard_curves=[FlatHazardCurve(hazard_rate=0.02)] * n_names,
    )

    cb_env = PricingEnvironment(
        valuation_date=datetime(2024, 6, 1),
        spot_quote=SpotQuote(spot=12.0),
        vol_surface=FlatVolSurface(volatility=0.30),
        rate_curve=FlatRateCurve(rate=0.05),
    )
    cb = ConvertibleBond(
        issue_date=datetime(2024, 1, 1), maturity_date=datetime(2029, 1, 1),
        face_value=100.0, coupon_rate=0.02, conversion_ratio=10.0,
        credit_spread=0.02, hazard_rate=0.01, recovery_rate=0.4,
    )

    return {
        "equity_mc_european": (
            EuropeanMCEngine(params=MCParams(num_paths=2000, seed=42)),
            eq_option, eq_env, "product_env",
        ),
        "equity_pde_european": (
            EuropeanPDESolver(PDEParams(grid_size=200, time_steps=100)),
            eq_option, eq_env, "product_env",
        ),
        "fx_mc_barrier": (
            FxBarrierMCEngine(
                params=FxMCParams(num_paths=20_000, time_steps=60, seed=3)
            ),
            fx_option, fx_env, "product_env",
        ),
        "credit_mc_basket_cds": (
            BasketCDSEngine(n_simulations=10_000, seed=7),
            cds_product, cds_env, "product_env",
        ),
        "bond_pde_convertible_tf": (
            ConvertibleBondTFEngine(
                cb_env,
                ConvertibleBondPDEParams(num_space_steps=50, num_time_steps=100),
            ),
            cb, None, "env_bound",
        ),
    }


def main() -> None:
    import numpy
    import scipy

    values = {}
    for name, (engine, product, env, call_shape) in build_representative_cases().items():
        if call_shape == "env_bound":
            values[name] = float(engine.price(product))
        else:
            values[name] = float(engine.price(product, env))
    payload = {
        "values": values,
        "versions": {"numpy": numpy.__version__, "scipy": scipy.__version__},
    }
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN_PATH} with {len(values)} goldens")


if __name__ == "__main__":
    main()
```

Note: import paths in `build_representative_cases` (e.g. `CurrencyPair`,
`BasketType`, `ConvertibleBond`) must be verified against the real modules
when implementing — `test/test_fx_barrier_mc.py`, `test/test_basket_cds.py`,
and `test/test_convertible_bond_engines.py` contain the authoritative import
lines; copy them exactly if they differ from the above.

- [ ] **Step 2: Append the cross-family parity tests**

Append to `test/execution/test_session_parity.py`:

```python
class TestCrossFamilyParity:
    """Phase 0 exit evidence: session == direct for one engine per family."""

    @pytest.fixture(scope="class")
    def cases(self):
        from test.execution.freeze_goldens import build_representative_cases

        return build_representative_cases()

    @pytest.mark.parametrize(
        "case_name",
        [
            "equity_mc_european",
            "equity_pde_european",
            "fx_mc_barrier",
            "credit_mc_basket_cds",
            "bond_pde_convertible_tf",
        ],
    )
    def test_session_price_equals_direct(self, cases, case_name):
        engine, product, env, call_shape = cases[case_name]
        if call_shape == "env_bound":
            direct = engine.price(product)
        else:
            direct = engine.price(product, env)
        with PricingSession() as session:
            via_session = session.price(engine, product, env)
        assert via_session == direct

    def test_legacy_internal_parallelism_preserved(self, cases):
        """Spec sections 12.4/17.1: engine-owned parallel settings (Snowball
        use_dask, DCN workers) behave identically through the session,
        including the missing-Dask UserWarning fallback."""
        import warnings

        from quantark.asset.equity.engine.mc import SnowballMCEngine
        from quantark.asset.equity.param import MCParams
        from quantark.asset.equity.product.option import SnowballOption
        from quantark.asset.equity.product.option.snowball_config import (
            BarrierConfig,
        )

        _, _, eq_env, _ = cases["equity_mc_european"]
        snowball = SnowballOption(
            initial_price=100.0, strike=100.0,
            barrier_config=BarrierConfig(
                ko_barrier=103.0, ko_rate=0.15,
                ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
                ki_barrier=75.0, ki_continuous=True,
            ),
            contract_multiplier=10_000.0, maturity=1.0,
        )

        def _build():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # missing-Dask UserWarning ok
                return SnowballMCEngine(
                    params=MCParams(num_paths=4000, seed=11),
                    use_dask=True, num_batches=4,
                )

        direct = _build().price(snowball, eq_env)
        with PricingSession() as session:
            via_session = session.price(_build(), snowball, eq_env)
        assert via_session == direct

    def test_goldens_match_current_serial_results(self, cases):
        import json
        import pathlib

        golden_path = (
            pathlib.Path(__file__).parent / "goldens" / "phase0_goldens.json"
        )
        goldens = json.loads(golden_path.read_text())["values"]
        for name, (engine, product, env, call_shape) in cases.items():
            if call_shape == "env_bound":
                value = float(engine.price(product))
            else:
                value = float(engine.price(product, env))
            assert value == pytest.approx(goldens[name], abs=1e-10), name
```

- [ ] **Step 3: Generate the goldens and run**

```bash
.venv/bin/python -m test.execution.freeze_goldens
.venv/bin/python -m pytest -n0 test/execution/test_session_parity.py -v
```

Expected: `wrote .../phase0_goldens.json with 5 goldens`, then all parity
tests PASS. If a fixture import fails, correct it from the authoritative
test files listed in the Step 1 note (fixture bug, not a framework bug).

- [ ] **Step 4: Run the full test suite**

Run: `.venv/bin/python -m pytest -q`
Expected: no new failures versus the base ref (pre-existing quarantined
failures excluded). The execution suite adds ~35 tests, all green.

- [ ] **Step 5: Commit**

```bash
git add test/execution/freeze_goldens.py test/execution/goldens/ test/execution/test_session_parity.py
git commit -m "test(execution): cross-family session parity matrix and phase-0 goldens"
```

---

## Phase 0 exit checklist (spec §21)

- [ ] Every name in every discovery surface is inventoried or classified
  (`test_every_public_export_is_classified`).
- [ ] Every concrete inventoried engine resolves a serial adapter
  (`test_every_concrete_engine_is_session_reachable`).
- [ ] Session == direct parity for one representative engine per asset family
  (`TestCrossFamilyParity`).
- [ ] No direct public behavior change: full suite green; the only modified
  legacy file is `base_engine.py` (one added method).
- [ ] Golden fixtures frozen and checked in.

## Self-Review Notes

- Spec §5 public surface: all exported (Task 7 `__init__`); `PricingRunContext`
  exported via `context` import in `__init__` — included in the export list.
- §5.4: `execute` is non-abstract, kernel never recursively calls
  `engine.execute` (kernel invokes adapters directly). ✓
- §6.2 resolution order: exact class = MRO[0]; structural capability detection
  is a Phase 1 concern (no specialized adapters exist yet) — documented in
  `registry.py` docstring. ✓
- §17.1: no `MCParams`/`PDEParams` fields added; no legacy signature changed;
  legacy env vars untouched (the new `QUANTARK_EXEC_*` aliases are additive). ✓
- §18: `temporary_legacy` rows all carry owner/milestone; `not_applicable`
  rows carry a reason. ✓
- Known Phase 0 simplifications (deliberate, spec-consistent): shallow
  normalization (`snapshot_complete=False`, no fingerprints), no resource
  leases (serial only), scenario contracts are data-only. These are Phase 1/5
  work, not gaps.
- Review dispositions (Codex plan gate): budget-test arithmetic fixed;
  engine-internal legacy parallelism (Snowball Dask, DCN workers) is
  documented passthrough per spec §12.4/§17.1 with a dedicated parity test —
  NOT rejected, since rejecting would change preserved direct behavior;
  full per-row numerical parity is Phase 1's exit gate by spec §21, so the
  Phase 0 reachability gate checks adapter resolution + price-method arity
  per row plus representative per-family parity.
