# Execution Framework Phase 2 — Fixed-Batch MC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement spec Phase 2 (`docs/superpowers/specs/2026-07-15-mc-pde-performance-generalization-design.md` §8, §11, §12.1–12.2, §21): `BatchPlan` / compact `BatchOutcome` contracts, bounded serial+thread backends with incremental canonical reduction, a session-owned `DrawRepository`, and DCN as the first migrated fixed-batch MC family — with **bit-identical** direct-vs-session parity in every mode.

**Architecture:** The kernel gains a batch dispatch path: when the resolved adapter implements `plan_batches`/`execute_batch`/`reduce_batches`, the kernel builds an immutable `BatchPlan`, runs its tasks through a backend (serial loop or bounded ThreadPoolExecutor), and feeds outcomes to the adapter's reducer as a **lazily ordered iterator** (buffered only until the next canonical batch index is available). The DCN batch adapters produce compact per-batch sufficient statistics whose merge in batch-index order is arithmetically identical to the legacy accumulator loop, so parity stays exact. `DrawRepository` wraps the Phase-1 `PreparedArtifactCache` machinery (parametrized lease pool `draw_cache`) and serves read-only Sobol masters to batch tasks through a thread-safe pinning provider that the kernel's existing `finally` releases.

**Tech Stack:** Python stdlib (`concurrent.futures`, `threading`, `dataclasses`), NumPy, existing `quantark.execution` Phase 0/1 contracts, `quantark.montecarlo.qmc_sobol.SobolNormalGenerator`.

## Global Constraints

- Kickoff decisions (2026-07-16): migrate **DCN only** (base GBM + LocalVol); all other MC engines get an honest batch-capability audit entry with rationale. Speed gates run via a **standalone benchmark script** (results recorded in docs); CI gets **deterministic structural gates only** (no wall-clock assertions). `DrawRepository` is a **session-owned service consumed via the adapter**; `quantark/montecarlo` and the legacy direct path are untouched. **No release prep**; merge to local main only.
- Spec invariants (§3.3, §17): no change to any direct legacy call path's behavior or performance; the framework never silently substitutes a backend/output/plan; correctness never depends on caching; engine-internal legacy parallelism passes through unchanged on the *direct* path (on the *session batch path* the framework owns threading and engine clones run `num_workers=1` — nested execution off, §12.5).
- Bit-identical exactness: for the migrated DCN engines, `session(serial)`, `session(threads, any worker count)`, and the direct call must agree **bitwise** on every `DCNMCResult` field except `elapsed_seconds`.
- Reviewed decisions that must NOT be "fixed": `LegacyPriceAdapter` guarantees only PV outputs; Snowball `use_dask` / `QUANTARK_DCN_MC_WORKERS` passthrough on the legacy path.
- Repo conventions: `quantark.*` imports only; exceptions derive `QuantArkException`; `quantark.util.numerical` for float comparisons in tests where tolerances appear; commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Test invocation (worktree): `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest` so worktree source shadows the editable install. Execution tests live in `test/execution/` and import sibling fixtures as top-level modules (e.g. `from execution.matrix_fixtures import ...` is NOT used; they use `sys.path` conftest conventions already in place — follow existing `test/execution/test_*.py` import style).
- The kernel must not statically import asset code (spec §4.1). DCN adapters live in `quantark/asset/equity/engine/mc/dcn_execution_adapters.py` and are reached via lazy registry factories.
- Known pre-existing failure `test_snowball_quad_flat_identity_golden` is out of scope (fails on unmodified main).

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `quantark/execution/contracts.py` | Modify | Add `BatchTask`, `BatchPlan`, `BatchOutcome` frozen dataclasses |
| `quantark/execution/policy.py` | Modify | Resolve `max_threads` / `draw_cache_bytes` from env; keep `policy_values` accurate |
| `quantark/execution/leases.py` | Modify | Add `draw_cache` and `task_scratch` pool capacities |
| `quantark/execution/cache/artifacts.py` | Modify | Parametrize lease pool name |
| `quantark/execution/cache/draws.py` | Create | `DrawDescriptor`, `DrawRepository` (wraps a pool-parametrized `PreparedArtifactCache`) |
| `quantark/execution/backends/__init__.py` | Create | Package init, shared exports |
| `quantark/execution/backends/serial.py` | Create | Ordered serial batch iterator |
| `quantark/execution/backends/threads.py` | Create | Bounded-window threaded batch iterator with canonical re-ordering |
| `quantark/execution/kernel.py` | Modify | Batch dispatch path, plan fingerprint, thread-clamp diagnostics |
| `quantark/execution/api.py` | Modify | Session-owned/validated `DrawRepository` |
| `quantark/execution/context.py` | Modify | `draw_repository` field |
| `quantark/execution/inventory.py` | Modify | `batch_state` + `batch_rationale` audit on every MC record |
| `quantark/execution/__init__.py` | Modify | Export new public names |
| `quantark/asset/equity/engine/mc/dcn_mc_engine.py` | Modify | Extract shared finalization; `_draw_provider` hook (pure refactor, direct path bit-identical) |
| `quantark/asset/equity/engine/mc/dcn_execution_adapters.py` | Modify | Batch adapters for `DCNMCEngine` + `LocalVolDCNMCEngine`; compact stats; provider |
| `quantark/execution/registry.py` | Modify | Batch adapter factories; exact-match pins for Heston DCN subclasses |
| `test/execution/test_batch_contracts.py` | Create | Contract + fingerprint tests |
| `test/execution/test_backends.py` | Create | Ordering, bounded window, failure propagation |
| `test/execution/test_draw_repository.py` | Create | Value parity, read-only, single-flight, budget, descriptor completeness |
| `test/execution/test_dcn_batch_adapter.py` | Create | Bit-identical parity (serial/threads/all stderr modes), compactness, draw reuse |
| `test/execution/test_inventory.py` | Modify | Batch-capability audit gate |
| `test/execution/benchmark_phase2.py` | Create | §20 gates 3+4 benchmark script (not pytest-collected) |
| `docs/superpowers/benchmarks/2026-07-16-execution-phase2-benchmark.md` | Create | Recorded benchmark results (git add -f; docs/ is in info/exclude) |

---

### Task 1: Batch contracts and policy/budget resolution

**Files:**
- Modify: `quantark/execution/contracts.py`
- Modify: `quantark/execution/policy.py`
- Test: `test/execution/test_batch_contracts.py`

**Interfaces:**
- Produces: `BatchTask(plan_id, batch_index, batch_id, n_paths)`, `BatchPlan(...)` (fields below), `BatchOutcome(batch_index, n_paths, payload)`; `ResourceBudget.max_threads` resolved from `QUANTARK_EXEC_MAX_THREADS` (owned default: `os.cpu_count()`), `draw_cache_bytes` from `QUANTARK_EXEC_DRAW_CACHE_MB`.

- [ ] **Step 1: Write the failing tests**

```python
# test/execution/test_batch_contracts.py
"""BatchPlan/BatchTask/BatchOutcome contracts and plan fingerprints."""
import dataclasses

import pytest

from quantark.execution.cache.fingerprint import try_fingerprint
from quantark.execution.contracts import BatchOutcome, BatchPlan, BatchTask
from quantark.execution.policy import resolve_resource_budget


def make_plan(**overrides):
    tasks = tuple(
        BatchTask(plan_id="p1", batch_index=i, batch_id=i, n_paths=256)
        for i in range(4)
    )
    base = dict(
        plan_id="p1",
        engine_class_path="quantark.asset.equity.engine.mc.dcn_mc_engine.DCNMCEngine",
        num_batches=4, paths_per_batch=256, total_paths=1024,
        seed=42, stream_kind="sobol", stream_layout="batch-shifted-sobol/v1",
        time_steps=252, dimension=252, dtype="float64",
        scheme="gbm-term/v1", stderr_mode="scramble_means",
        reduction_order="batch_index/v1", tasks=tasks,
        est_task_peak_bytes=1_000_000, est_outcome_bytes=4_096,
        implementation_fingerprint=None,
    )
    base.update(overrides)
    return BatchPlan(**base)


def test_plan_is_frozen():
    plan = make_plan()
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.seed = 43


def test_plan_fingerprint_stable_and_sensitive():
    fp1 = try_fingerprint(make_plan())
    fp2 = try_fingerprint(make_plan())
    fp3 = try_fingerprint(make_plan(seed=43))
    assert fp1 is not None and fp1 == fp2 and fp1 != fp3


def test_outcome_carries_index_and_payload():
    out = BatchOutcome(batch_index=2, n_paths=256, payload=("stats",))
    assert out.batch_index == 2 and out.n_paths == 256


def test_budget_resolves_threads_and_draw_cache(monkeypatch):
    budget, sources = resolve_resource_budget(environ={
        "QUANTARK_EXEC_MAX_THREADS": "8",
        "QUANTARK_EXEC_DRAW_CACHE_MB": "64",
    })
    assert budget.max_threads == 8
    assert budget.draw_cache_bytes == 64 * 2**20
    fields = dict(sources)
    assert fields["max_threads"] == "env"
    assert fields["draw_cache_bytes"] == "env"


def test_budget_default_threads_is_one_without_env():
    budget, _ = resolve_resource_budget(environ={})
    assert budget.max_threads == 1  # session upgrades to cpu_count for OWNED budgets
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/execution/test_batch_contracts.py -v`
Expected: FAIL — `ImportError: cannot import name 'BatchPlan'`.

- [ ] **Step 3: Implement contracts**

Append to `quantark/execution/contracts.py` (and extend `__all__`):

```python
@dataclass(frozen=True)
class BatchTask:
    """One unit of MC work; ``batch_id`` is the engine stream-shift id
    (None means the engine's single-batch stream)."""

    plan_id: str
    batch_index: int
    batch_id: int | None
    n_paths: int


@dataclass(frozen=True)
class BatchPlan:
    """Immutable fixed-batch MC plan (spec section 8.1). The plan, not
    executor completion order, determines economics."""

    plan_id: str
    engine_class_path: str
    num_batches: int
    paths_per_batch: int
    total_paths: int
    seed: int
    stream_kind: str        # "sobol" | "pseudorandom" | "antithetic"
    stream_layout: str
    time_steps: int
    dimension: int
    dtype: str
    scheme: str
    stderr_mode: str        # "scramble_means" | "pathwise_iid"
    reduction_order: str    # "batch_index/v1"
    tasks: tuple
    est_task_peak_bytes: int | None
    est_outcome_bytes: int | None
    implementation_fingerprint: str | None


@dataclass(frozen=True)
class BatchOutcome:
    """Compact per-batch sufficient statistics (spec section 8.2)."""

    batch_index: int
    n_paths: int
    payload: object
```

- [ ] **Step 4: Implement budget resolution**

In `quantark/execution/policy.py`, extend `resolve_resource_budget` (explicit path lists the new fields too):

```python
    max_threads = _env_int(environ, "QUANTARK_EXEC_MAX_THREADS",
                           1, "max_threads", sources)
    draw_mb = _env_int(environ, "QUANTARK_EXEC_DRAW_CACHE_MB",
                       None, "draw_cache_bytes", sources)
    return (
        ResourceBudget(
            max_threads=max_threads,
            total_memory_bytes=None if memory_mb is None else memory_mb * 2**20,
            draw_cache_bytes=None if draw_mb is None else draw_mb * 2**20,
            artifact_cache_bytes=None if cache_mb is None else cache_mb * 2**20,
            max_in_flight=max_in_flight,
        ),
        tuple(sources),
    )
```

(`policy_values` already prints `budget.max_threads` and `budget.draw_cache_bytes` — no change needed.)

- [ ] **Step 5: Run tests to verify pass**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/execution/test_batch_contracts.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add quantark/execution/contracts.py quantark/execution/policy.py test/execution/test_batch_contracts.py
git commit -m "feat(execution): BatchPlan/BatchTask/BatchOutcome contracts + thread/draw budget resolution"
```

---

### Task 2: Lease pools and pool-parametrized artifact cache

**Files:**
- Modify: `quantark/execution/leases.py`
- Modify: `quantark/execution/cache/artifacts.py`
- Test: `test/execution/test_leases.py` (extend), `test/execution/test_artifact_cache.py` (extend)

**Interfaces:**
- Consumes: `ResourceBudget.draw_cache_bytes`, `total_memory_bytes` (Task 1).
- Produces: `ResourceLeaseManager` capacities `{"artifact_cache", "draw_cache", "task_scratch"}`; `PreparedArtifactCache(lease_manager, pool="artifact_cache")` — default unchanged.

- [ ] **Step 1: Write the failing tests** (append to the two existing test files)

```python
# append to test/execution/test_leases.py
def test_draw_cache_and_task_scratch_pools_enforced():
    budget = ResourceBudget(
        draw_cache_bytes=100, total_memory_bytes=200, artifact_cache_bytes=50
    )
    mgr = ResourceLeaseManager(budget)
    mgr.lease_bytes(90, "draw_cache")
    with pytest.raises(ResourceBudgetExceeded):
        mgr.lease_bytes(20, "draw_cache")
    mgr.lease_bytes(200, "task_scratch")
    with pytest.raises(ResourceBudgetExceeded):
        mgr.lease_bytes(1, "task_scratch")


def test_task_scratch_unlimited_when_total_memory_unset():
    mgr = ResourceLeaseManager(ResourceBudget())
    mgr.lease_bytes(10**12, "task_scratch")  # no capacity -> no limit
```

```python
# append to test/execution/test_artifact_cache.py
def test_cache_pool_is_parametrizable():
    budget = ResourceBudget(draw_cache_bytes=1024, artifact_cache_bytes=0)
    mgr = ResourceLeaseManager(budget)
    cache = PreparedArtifactCache(mgr, pool="draw_cache")
    desc = ArtifactDescriptor(
        kind="k", fingerprint="f", dependency_tags=frozenset(), builder_version="1"
    )
    handle = cache.get_or_build(desc, lambda: b"x" * 10, size_bytes=10)
    assert mgr.pool_bytes("draw_cache") == 10
    assert mgr.pool_bytes("artifact_cache") == 0
    handle.close()
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/execution/test_leases.py test/execution/test_artifact_cache.py -v`
Expected: new tests FAIL (`task_scratch` capacity missing / unexpected `pool` kwarg).

- [ ] **Step 3: Implement**

`leases.py` — in `ResourceLeaseManager.__init__`:

```python
        self._capacities = {
            "artifact_cache": budget.artifact_cache_bytes,
            "draw_cache": budget.draw_cache_bytes,
            "task_scratch": budget.total_memory_bytes,
        }
```

`cache/artifacts.py` — replace the class-level `_POOL` constant with an instance field:

```python
    def __init__(self, lease_manager, pool: str = "artifact_cache"):
        self._leases = lease_manager
        self.lease_manager = lease_manager  # public: pairing identity check
        self._pool = pool
        ...
```

and replace every `self._POOL` use with `self._pool`.

- [ ] **Step 4: Run the full execution suite** (regression: Phase 0/1 tests must stay green)

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/execution/ -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add quantark/execution/leases.py quantark/execution/cache/artifacts.py test/execution/test_leases.py test/execution/test_artifact_cache.py
git commit -m "feat(execution): draw_cache/task_scratch lease pools; pool-parametrized artifact cache"
```

---

### Task 3: DrawRepository

**Files:**
- Create: `quantark/execution/cache/draws.py`
- Test: `test/execution/test_draw_repository.py`

**Interfaces:**
- Consumes: `PreparedArtifactCache(mgr, pool="draw_cache")` (Task 2), `try_fingerprint`, `quantark.montecarlo.qmc_sobol.SobolNormalGenerator`, `quantark.execution.manifest.build_versions`.
- Produces: `DrawDescriptor` (frozen, complete per spec §8.3), `DrawRepository(lease_manager)` with `.normals_handle(seed, n_paths, dim, batch_id, writable=False) -> ArtifactHandle`, `.stats() -> dict`, `.close()`, `.lease_manager` property. Masters are read-only; `writable=True` returns a private copy via a handle whose close is a no-op release.

- [ ] **Step 1: Write the failing tests**

```python
# test/execution/test_draw_repository.py
"""Session-owned Sobol DrawRepository (spec section 8.3)."""
import numpy as np
import pytest

from quantark.asset.equity.engine.mc.qmc_draws import qmc_normals
from quantark.execution.cache.draws import DrawDescriptor, DrawRepository
from quantark.execution.leases import ResourceLeaseManager
from quantark.execution.policy import ResourceBudget


def make_repo(draw_bytes=64 * 2**20):
    mgr = ResourceLeaseManager(ResourceBudget(draw_cache_bytes=draw_bytes))
    return DrawRepository(mgr), mgr


def test_values_identical_to_legacy_generator():
    repo, _ = make_repo()
    with repo.normals_handle(seed=42, n_paths=128, dim=16, batch_id=3) as h:
        legacy = qmc_normals(42, 128, 16, batch_id=3)
        np.testing.assert_array_equal(h.value, legacy)


def test_master_is_read_only_and_cached_once():
    repo, mgr = make_repo()
    with repo.normals_handle(seed=1, n_paths=64, dim=8, batch_id=0) as h1:
        assert not h1.value.flags.writeable
        with repo.normals_handle(seed=1, n_paths=64, dim=8, batch_id=0) as h2:
            assert h2.value is h1.value
    assert repo.stats()["hits"] == 1


def test_writable_copy_is_private():
    repo, _ = make_repo()
    with repo.normals_handle(seed=1, n_paths=64, dim=8, batch_id=0) as h:
        master = h.value
    with repo.normals_handle(
        seed=1, n_paths=64, dim=8, batch_id=0, writable=True
    ) as w:
        assert w.value.flags.writeable
        w.value[:] = 0.0
    with repo.normals_handle(seed=1, n_paths=64, dim=8, batch_id=0) as h2:
        np.testing.assert_array_equal(h2.value, master)


def test_descriptor_completeness_every_field_changes_key():
    base = dict(seed=1, n_paths=64, dim=8, batch_id=0)
    repo, _ = make_repo()
    keys = set()
    for override in ({}, {"seed": 2}, {"n_paths": 128}, {"dim": 16},
                     {"batch_id": 1}, {"batch_id": None}):
        d = repo.descriptor(**{**base, **override})
        keys.add(d.fingerprint)
    assert len(keys) == 6


def test_zero_budget_bypasses_but_stays_correct():
    repo, mgr = make_repo(draw_bytes=0)
    with repo.normals_handle(seed=42, n_paths=64, dim=8, batch_id=0) as h:
        np.testing.assert_array_equal(h.value, qmc_normals(42, 64, 8, batch_id=0))
    assert mgr.pool_bytes("draw_cache") == 0


def test_budget_accounting_exact_bytes():
    repo, mgr = make_repo()
    with repo.normals_handle(seed=1, n_paths=64, dim=8, batch_id=0):
        assert mgr.pool_bytes("draw_cache") == 64 * 8 * 8
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/execution/test_draw_repository.py -v`
Expected: FAIL — module `quantark.execution.cache.draws` does not exist.

- [ ] **Step 3: Implement `quantark/execution/cache/draws.py`**

```python
"""Session-owned Sobol draw repository (spec section 8.3).

Reuses the hardened ``PreparedArtifactCache`` single-flight/lease machinery
on the dedicated ``draw_cache`` pool. Masters are published READ-ONLY;
``writable=True`` hands out a private copy (the only surface on which
in-place transforms are permitted). Only fully-descriptor-identified
scrambled-Sobol blocks are cacheable; stateful pseudorandom streams never
reach this repository (engines keep their legacy ``_draws`` for those).
"""
from dataclasses import dataclass

import numpy as np

from quantark.execution.cache.artifacts import (
    ArtifactDescriptor,
    ArtifactHandle,
    PreparedArtifactCache,
)
from quantark.execution.cache.fingerprint import fingerprint
from quantark.execution.manifest import build_versions

__all__ = ["DrawDescriptor", "DrawRepository"]

_IMPL_ID = "quantark.montecarlo.qmc_sobol.SobolNormalGenerator"
_IMPL_VERSION = "1"
_LAYOUT = "batch-shifted-sobol/v1"


@dataclass(frozen=True)
class DrawDescriptor:
    """Complete identification of a generated draw block (spec section 8.3).
    A block not fully identified by its descriptor is not cacheable."""

    generator_family: str
    implementation_id: str
    implementation_version: str
    distribution: str
    stream_layout: str
    seed: int
    batch_id: int | None
    n_paths: int
    dim: int
    shape: tuple
    memory_order: str
    dtype: str
    antithetic: bool
    transform_pipeline: tuple
    numpy_version: str
    scipy_version: str

    @property
    def fingerprint(self) -> str:
        return fingerprint(self)


class DrawRepository:
    def __init__(self, lease_manager):
        self._cache = PreparedArtifactCache(lease_manager, pool="draw_cache")
        self.lease_manager = lease_manager  # public: pairing identity check

    def descriptor(self, *, seed, n_paths, dim, batch_id) -> DrawDescriptor:
        versions = dict(build_versions())
        return DrawDescriptor(
            generator_family="sobol-scrambled",
            implementation_id=_IMPL_ID,
            implementation_version=_IMPL_VERSION,
            distribution="normal",
            stream_layout=_LAYOUT,
            seed=int(seed),
            batch_id=None if batch_id is None else int(batch_id),
            n_paths=int(n_paths),
            dim=int(dim),
            shape=(int(n_paths), int(dim)),
            memory_order="C",
            dtype="float64",
            antithetic=False,
            transform_pipeline=("ndtri/v1",),
            numpy_version=versions.get("numpy", "unknown"),
            scipy_version=versions.get("scipy", "unknown"),
        )

    def normals_handle(
        self, *, seed, n_paths, dim, batch_id, writable=False
    ) -> ArtifactHandle:
        desc = self.descriptor(
            seed=seed, n_paths=n_paths, dim=dim, batch_id=batch_id
        )
        art = ArtifactDescriptor(
            kind="sobol-normal-block",
            fingerprint=desc.fingerprint,
            dependency_tags=frozenset({"draws"}),
            builder_version=_IMPL_VERSION,
        )

        def builder():
            from quantark.montecarlo.qmc_sobol import SobolNormalGenerator

            block = np.ascontiguousarray(
                SobolNormalGenerator(base_seed=int(seed)).normal(
                    int(n_paths), int(dim), batch_id=batch_id
                )
            )
            block.flags.writeable = False
            return block

        handle = self._cache.get_or_build(
            art, builder,
            size_bytes=int(n_paths) * int(dim) * 8,
            measure=lambda block: block.nbytes,
        )
        if not writable:
            return handle
        copy = handle.value.copy()
        handle.close()  # writable scratch is task-owned, not pinned
        return ArtifactHandle(copy, lambda: None)

    def stats(self) -> dict:
        return self._cache.stats()

    def close(self) -> None:
        self._cache.close()
```

- [ ] **Step 4: Run tests to verify pass**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/execution/test_draw_repository.py -v`
Expected: PASS. (Value parity holds because both paths run `SobolNormalGenerator(base_seed=seed).normal(n, d, batch_id)` — the repository builds directly, deliberately bypassing the process-global `QMCDrawCache` to avoid double-charging bytes.)

- [ ] **Step 5: Commit**

```bash
git add quantark/execution/cache/draws.py test/execution/test_draw_repository.py
git commit -m "feat(execution): session-owned Sobol DrawRepository on the draw_cache lease pool"
```

---

### Task 4: Serial and threaded batch backends

**Files:**
- Create: `quantark/execution/backends/__init__.py`, `quantark/execution/backends/serial.py`, `quantark/execution/backends/threads.py`
- Test: `test/execution/test_backends.py`

**Interfaces:**
- Consumes: `BatchPlan`/`BatchTask` (Task 1); `ResourceLeaseManager.lease_bytes/release_bytes` on `task_scratch` and `task_slot()` per executing batch.
- Produces: `serial.iter_ordered(plan, execute, lease_manager=None)` and `threads.iter_ordered(plan, execute, workers, window, lease_manager=None, observer=None)` — both yield `(batch_index, outcome)` in strictly increasing `batch_index` order; `observer` (test hook) is called as `observer(in_flight, buffered)` after each state change.

Admission-control contract (Codex plan-gate findings, 2026-07-16):

1. **Per-batch task slots.** Each executing batch task holds a `lease_manager.task_slot()` for its execution duration (both backends). The kernel does NOT hold its dispatch-wide slot around a batch plan (Task 7) — batch tasks ARE the admitted tasks, so `ResourceBudget.max_in_flight` genuinely bounds concurrent batch execution. The kernel additionally clamps `window <= budget.max_in_flight`, so within one dispatch slot acquisition never over-asks; cross-session contention on a shared lease manager still raises `ResourceBudgetExceeded` (admission control working as specified, `may_shrink` is Phase-3+).
2. **Buffered outcomes stay charged.** A task's `est_task_peak_bytes` lease is held while it executes; on completion it is swapped for an `est_outcome_bytes` lease that is released only when the ordered iterator YIELDS that outcome to the reducer. Submission gates on `len(pending) + len(buffered) < window`, so total retained work is ≤ `window` items and a stalled batch 0 cannot accumulate unbounded completed outcomes (pathwise-IID outcomes carry per-path totals, so this bound is load-bearing). Progress is deadlock-free: tasks are submitted in index order, so whenever buffered outcomes exist, `next_index` is already pending.

- [ ] **Step 1: Write the failing tests**

```python
# test/execution/test_backends.py
"""Ordered, bounded batch execution backends (spec sections 8.2, 12.1-12.2)."""
import threading
import time

import pytest

from quantark.execution.backends import serial, threads
from quantark.execution.contracts import BatchTask
from quantark.execution.errors import ResourceBudgetExceeded
from quantark.execution.leases import ResourceLeaseManager
from quantark.execution.policy import ResourceBudget


class FakePlan:
    def __init__(self, n, est_task_peak_bytes=None, est_outcome_bytes=None):
        self.tasks = tuple(
            BatchTask(plan_id="p", batch_index=i, batch_id=i, n_paths=8)
            for i in range(n)
        )
        self.est_task_peak_bytes = est_task_peak_bytes
        self.est_outcome_bytes = est_outcome_bytes


def test_serial_yields_in_order():
    plan = FakePlan(5)
    out = list(serial.iter_ordered(plan, lambda t: t.batch_index * 10))
    assert out == [(i, i * 10) for i in range(5)]


def test_threads_yield_in_canonical_order_despite_reversed_completion():
    plan = FakePlan(8)

    def execute(task):
        time.sleep(0.02 * (8 - task.batch_index))  # later batches finish first
        return task.batch_index * 10

    out = list(threads.iter_ordered(plan, execute, workers=8, window=8))
    assert out == [(i, i * 10) for i in range(8)]


def test_threads_bounded_window_and_buffering():
    plan = FakePlan(12)
    seen = {"max_total": 0}
    lock = threading.Lock()

    def observer(in_flight, buffered):
        with lock:
            seen["max_total"] = max(seen["max_total"], in_flight + buffered)

    def execute(task):
        time.sleep(0.01)
        return task.batch_index

    out = list(threads.iter_ordered(
        plan, execute, workers=4, window=4, observer=observer
    ))
    assert [i for i, _ in out] == list(range(12))
    assert seen["max_total"] <= 4  # pending + buffered never exceeds window


def test_threads_stalled_first_batch_cannot_accumulate_unbounded_outcomes():
    # Batch 0 stalls; later batches finish fast. Submission is gated on
    # pending + buffered < window, so completed outcomes cannot pile up.
    plan = FakePlan(12)
    release = threading.Event()
    seen = {"max_buffered": 0}
    lock = threading.Lock()

    def observer(in_flight, buffered):
        with lock:
            seen["max_buffered"] = max(seen["max_buffered"], buffered)

    def execute(task):
        if task.batch_index == 0:
            release.wait(timeout=10)
        return task.batch_index

    def unblock():
        time.sleep(0.2)
        release.set()

    threading.Thread(target=unblock).start()
    out = list(threads.iter_ordered(
        plan, execute, workers=4, window=4, observer=observer
    ))
    assert [i for i, _ in out] == list(range(12))
    assert seen["max_buffered"] <= 3  # window - the stalled pending task


def test_threads_respect_max_in_flight_slots():
    # budget.max_in_flight=1 with workers=4: per-batch task slots serialize
    # execution (Codex plan-gate finding: admission control must bind).
    mgr = ResourceLeaseManager(ResourceBudget(max_in_flight=1))
    active = {"now": 0, "max": 0}
    lock = threading.Lock()

    def execute(task):
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.01)
        with lock:
            active["now"] -= 1
        return task.batch_index

    plan = FakePlan(6)
    out = list(threads.iter_ordered(
        plan, execute, workers=4, window=1, lease_manager=mgr
    ))
    assert [i for i, _ in out] == list(range(6))
    assert active["max"] == 1


def test_buffered_outcome_bytes_stay_leased_until_yield():
    mgr = ResourceLeaseManager(ResourceBudget(total_memory_bytes=10_000))
    plan = FakePlan(4, est_task_peak_bytes=100, est_outcome_bytes=40)
    observed = []

    def execute(task):
        if task.batch_index == 0:
            time.sleep(0.1)  # others complete first and sit buffered
        return task.batch_index

    for index, _ in threads.iter_ordered(
        plan, execute, workers=4, window=4, lease_manager=mgr
    ):
        observed.append((index, mgr.pool_bytes("task_scratch")))
    # after the final yield every task and outcome lease is back
    assert mgr.pool_bytes("task_scratch") == 0
    assert observed[0][0] == 0


def test_threads_propagates_failure_and_stops():
    plan = FakePlan(6)
    started = []

    def execute(task):
        started.append(task.batch_index)
        if task.batch_index == 1:
            raise ValueError("boom")
        time.sleep(0.01)
        return task.batch_index

    with pytest.raises(ValueError, match="boom"):
        list(threads.iter_ordered(plan, execute, workers=2, window=2))
    assert len(started) < 6  # fail-fast: pending tasks were not all submitted


def test_threads_leases_task_scratch_per_in_flight():
    mgr = ResourceLeaseManager(ResourceBudget(total_memory_bytes=250))
    plan = FakePlan(4, est_task_peak_bytes=100)  # window 2 => 200 <= 250 ok
    out = list(threads.iter_ordered(
        plan, lambda t: t.batch_index, workers=2, window=2, lease_manager=mgr
    ))
    assert [i for i, _ in out] == list(range(4))
    assert mgr.pool_bytes("task_scratch") == 0  # all released


def test_single_task_exceeding_budget_fails_before_execution():
    mgr = ResourceLeaseManager(ResourceBudget(total_memory_bytes=50))
    plan = FakePlan(2, est_task_peak_bytes=100)
    executed = []
    with pytest.raises(ResourceBudgetExceeded):
        list(threads.iter_ordered(
            plan, lambda t: executed.append(t), workers=1, window=1,
            lease_manager=mgr,
        ))
    assert executed == []
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/execution/test_backends.py -v`
Expected: FAIL — backends package missing.

- [ ] **Step 3: Implement**

`quantark/execution/backends/__init__.py`:

```python
"""Batch execution backends (spec section 12). All backends consume the same
immutable BatchPlan and yield outcomes in canonical batch-index order; they
own scheduling only, never numerical meaning."""
from quantark.execution.backends import serial, threads

__all__ = ["serial", "threads"]
```

`quantark/execution/backends/serial.py`:

```python
"""Serial backend (spec section 12.1): the compatibility reference.

Each batch still holds one task slot while executing, so a shared lease
manager sees the same admission accounting as the threaded backend.
"""
import contextlib

__all__ = ["iter_ordered"]


def iter_ordered(plan, execute, lease_manager=None):
    for task in plan.tasks:
        slot = (lease_manager.task_slot() if lease_manager is not None
                else contextlib.nullcontext())
        with slot:
            outcome = execute(task)
        yield task.batch_index, outcome
```

`quantark/execution/backends/threads.py`:

```python
"""Bounded threaded backend (spec sections 8.2, 11, 12.2).

Admission contract (hardened at the 2026-07-16 plan gate):

- Every executing batch holds a per-task slot from the lease manager, so
  ``ResourceBudget.max_in_flight`` bounds CONCURRENT BATCH EXECUTION, not
  merely dispatches. The kernel clamps ``window <= max_in_flight``.
- ``est_task_peak_bytes`` is leased while a task executes; on completion it
  is swapped for an ``est_outcome_bytes`` lease held until the ordered
  iterator yields that outcome to the reducer — buffered outcomes stay
  charged (pathwise-IID outcomes carry per-path totals).
- Submission gates on ``len(pending) + len(buffered) < window``: total
  retained work is at most ``window`` items. Deadlock-free because tasks
  are submitted in index order, so whenever anything is buffered the next
  canonical index is already pending.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

__all__ = ["iter_ordered"]

_POOL = "task_scratch"


class _Leases:
    def __init__(self, lease_manager, plan):
        self._mgr = lease_manager
        self._task = plan.est_task_peak_bytes
        self._out = plan.est_outcome_bytes

    def _move(self, n, sign):
        if self._mgr is not None and n is not None:
            if sign > 0:
                self._mgr.lease_bytes(n, _POOL)
            else:
                self._mgr.release_bytes(n, _POOL)

    def start_task(self):
        self._move(self._task, +1)
        if self._mgr is not None:
            slot = self._mgr.task_slot()
            slot.__enter__()
            return slot
        return None

    def finish_task(self, slot, *, to_outcome):
        if slot is not None:
            slot.__exit__(None, None, None)
        self._move(self._task, -1)
        if to_outcome:
            self._move(self._out, +1)

    def yield_outcome(self):
        self._move(self._out, -1)


def iter_ordered(plan, execute, workers, window, lease_manager=None,
                 observer=None):
    tasks = list(plan.tasks)
    leases = _Leases(lease_manager, plan)

    def run(task):
        slot = leases.start_task()
        try:
            outcome = execute(task)
        except BaseException:
            leases.finish_task(slot, to_outcome=False)
            raise
        leases.finish_task(slot, to_outcome=True)
        return outcome

    buffered: dict = {}
    next_index = 0
    submitted = 0
    pending: dict = {}

    def submit_up_to_window(pool):
        nonlocal submitted
        while (submitted < len(tasks)
               and len(pending) + len(buffered) < window):
            future = pool.submit(run, tasks[submitted])
            pending[future] = tasks[submitted].batch_index
            submitted += 1
            if observer is not None:
                observer(len(pending), len(buffered))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            submit_up_to_window(pool)
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    buffered[index] = future.result()  # raises on failure
                    if observer is not None:
                        observer(len(pending), len(buffered))
                while next_index in buffered:
                    outcome = buffered.pop(next_index)
                    leases.yield_outcome()
                    yield next_index, outcome
                    next_index += 1
                submit_up_to_window(pool)
        except BaseException:
            for future in pending:
                future.cancel()
            # Running tasks cannot be interrupted (pool shutdown waits on
            # them regardless); wait so their outcome leases are visible,
            # then sweep every charged-but-unconsumed outcome.
            done, _ = wait(pending)
            for future in done:
                if not future.cancelled() and future.exception() is None:
                    leases.yield_outcome()
            for _ in range(len(buffered)):
                leases.yield_outcome()
            raise
```

Lease accounting on every path: an executing task's slot+bytes are released inside `run` (worker thread, success or failure); a completed outcome's bytes are released at yield, in the failure sweep for outcomes still buffered, or in the failure sweep for futures that completed after the fault. Futures cancelled before starting never leased anything (leasing happens inside `run`). The arbiter tests are `pool_bytes("task_scratch") == 0` after both success and failure, and `max_in_flight` slot compliance under `workers > max_in_flight`.

Note the slot-acquisition semantics: `task_slot()` RAISES `ResourceBudgetExceeded` when exhausted rather than blocking. Within a single dispatch this cannot fire (the kernel clamps `window <= max_in_flight`, and submission never exceeds the window); if two sessions share one lease manager, a raise is admission control refusing over-commitment, which is the specified §11 terminal response (`may_shrink`/serial-fallback negotiation is Phase-3+).

- [ ] **Step 4: Run tests to verify pass**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/execution/test_backends.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add quantark/execution/backends/ test/execution/test_backends.py
git commit -m "feat(execution): ordered bounded serial/thread batch backends with task-scratch leasing"
```

---

### Task 5: DCN engine refactor — shared finalization and draw-provider hook

**Files:**
- Modify: `quantark/asset/equity/engine/mc/dcn_mc_engine.py`
- Test: existing `test/test_dcn_*.py` suite + `test/execution/goldens` (unchanged behavior is the test)

This is a pure extraction: direct-path results must be bit-identical before/after.

- [ ] **Step 1: Record the before state**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q -k "dcn" | tail -3`
Expected: current pass count noted (all green).

- [ ] **Step 2: Extract finalization**

In `dcn_mc_engine.py`, move lines 235–299 of `price_detailed` (from `sign = product.direction_sign` through `result = DCNMCResult(...)`) into a module function, with stderr computed by the caller:

```python
def _finalize_dcn_result(acc, product, seed, stderr, t0) -> DCNMCResult:
    """Assemble a DCNMCResult from a fully-populated accumulator.

    Shared verbatim by the direct path and the execution-framework batch
    reducer so the two can never drift arithmetically.
    """
    sign = product.direction_sign
    n = acc.n
    # legs first; pv is DEFINED as their sum (exact invariant)
    pv_fixed = float(sign * acc.fixed_sum / n)
    ...  # existing lines verbatim, ending with `return result`
```

`price_detailed` then reads:

```python
        sign = product.direction_sign
        n = acc.n
        if self.use_sobol and self.num_batches >= 2:
            batch_means = np.array(
                [float(sign * t.mean()) for t in acc.totals]
            )
            stderr = float(
                batch_means.std(ddof=1) / np.sqrt(batch_means.size)
            )
        else:
            totals = sign * np.concatenate(acc.totals)
            stderr = float(totals.std(ddof=1) / np.sqrt(n))
        result = _finalize_dcn_result(acc, product, self.seed, stderr, t0)
        self._last_result = result
        return result
```

(Keep the existing RQMC/IID stderr comments with the moved branches.)

- [ ] **Step 3: Add the draw-provider hook**

```python
class DCNMCEngine(BaseEngine):
    ...
    _draw_provider = None  # session-path hook; None on every direct-path engine

    def _draws(self, n_dims, n_paths, batch_id):
        if self._draw_provider is not None:
            block = self._draw_provider(n_dims, n_paths, batch_id)
            if block is not None:
                return block
        if self.use_sobol:
            ...  # existing body unchanged
```

The provider is only ever set on adapter-owned clones (Task 6); a provider returning `None` (mode it does not serve) falls through to the legacy path.

- [ ] **Step 4: Verify bit-identical direct behavior**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q -k "dcn" && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q test/execution/test_session_parity.py test/execution/test_regression_gates.py`
Expected: PASS with the same counts as Step 1; parity goldens untouched.

- [ ] **Step 5: Commit**

```bash
git add quantark/asset/equity/engine/mc/dcn_mc_engine.py
git commit -m "refactor(dcn-mc): extract shared result finalization + draw-provider hook (bit-identical)"
```

---

### Task 6: DCN batch adapters, compact stats, registry pins

**Files:**
- Modify: `quantark/asset/equity/engine/mc/dcn_execution_adapters.py`
- Modify: `quantark/execution/registry.py`
- Test: `test/execution/test_dcn_batch_adapter.py` (kernel integration lands in Task 7; this task unit-tests the adapter surface directly)

**Interfaces:**
- Consumes: `_LegAccumulator`, `_finalize_dcn_result`, `compute_dcn_cashflows`, `build_dcn_grid_context`, `build_mc_term_inputs`, `make_df_fn` (all existing); `DrawRepository.normals_handle` (Task 3); `BatchPlan/BatchTask/BatchOutcome` (Task 1); Phase-1 Dupire prepare.
- Produces:
  - `DCNBatchStats` frozen dataclass: `n, fixed_sum, ko_sum, loss_sum, fixed_by_period, ko_by_period, coupon_paid, ko_timing, ki_count, survive_no_ki_count, survive_ki_count, life_sum, batch_mean, totals` (`totals: np.ndarray | None`, kept only in `pathwise_iid` mode).
  - `DCNBatchMCAdapter` (for `DCNMCEngine`) and `DCNLocalVolMCAdapter` (extended) implementing `prepare` / `plan_batches(engine, request, state, context)` / `execute_batch(task, state, context)` / `reduce_batches(outcomes, plan, state, context) -> (value, economics)`.
  - `_DCNSimContext` frozen payload: `engine` (clone, `num_workers=1`), `product`, `pricing_env`, `ctx`, `term`, `df`, `spot0`, `dt_array`, `obs_times`, `n_obs`, `t0`.
  - Registry: lazy factories map `DCNMCEngine -> DCNBatchMCAdapter`; exact pins `HestonDCNMCEngine`/`CoupledCoarseHestonDCNMCEngine` -> plain `LegacyPriceAdapter(call_shape="product_env")` so MRO never leaks batch capability to the un-migrated Heston family (`QEDCNMCEngine` resolves to the `HestonDCNMCEngine` pin as nearest base).

Key mechanics:

1. `prepare` (base GBM adapter): builds the sim context once (grid, term inputs, df, clone). Clone is `type(engine)(num_paths=..., seed=..., use_sobol=..., use_antithetic=..., num_batches=..., num_workers=1)`. If `engine.use_sobol` and `context.draw_repository is not None`, attach a thread-safe pinning provider to the clone and include it in `PreparedState.handles` (the kernel's existing `finally` closes it):

```python
class _PinnedDrawProvider:
    """Thread-safe draw fetcher; pins masters until the kernel closes it."""

    def __init__(self, repository, seed):
        self._repo = repository
        self._seed = seed
        self._lock = threading.Lock()
        self._handles = []

    def __call__(self, n_dims, n_paths, batch_id):
        handle = self._repo.normals_handle(
            seed=self._seed, n_paths=n_paths, dim=n_dims, batch_id=batch_id
        )
        with self._lock:
            self._handles.append(handle)
        return handle.value

    def close(self):
        with self._lock:
            handles, self._handles = self._handles, []
        for handle in handles:
            handle.close()
```

2. The LV adapter's `prepare` composes: Phase-1 Dupire surface fetch (unchanged logic) → clone via `_clone_with_surface` (now passing `num_workers=1`) → same provider attachment → `PreparedState(payload=sim_context, handles=(surface_handle, provider))`.

3. `plan_batches`: `batch_id = None if engine.num_batches == 1 else index`; `stderr_mode = "scramble_means" if (engine.use_sobol and engine.num_batches >= 2) else "pathwise_iid"`; `stream_kind` from `use_sobol`/`use_antithetic`; `est_task_peak_bytes = 8 * batch_size * (2 * time_steps + 1 + 6 * n_obs) + (1 << 20)` (draws + nodes + cashflow arrays, conservative); `est_outcome_bytes = 8 * (6 * n_obs + 16) + (8 * batch_size if pathwise else 0)`.

4. `execute_batch`: run the clone's `_simulate` + `compute_dcn_cashflows`, then reduce that one batch through a fresh `_LegAccumulator` (identical arithmetic to the legacy per-batch `add`) and extract `DCNBatchStats`. `batch_mean = float(cf.total_pv.mean())`; `totals = cf.total_pv` only when `plan.stderr_mode == "pathwise_iid"`.

5. `reduce_batches`: merge stats into one `_LegAccumulator` in the yielded (canonical) order — each field merge is the same `+=` the legacy loop performs; compute stderr per `plan.stderr_mode` (scramble means: `np.array([float(sign * m) for m in means])`, identical arithmetic since `float(sign * t.mean()) == float(sign * batch_mean)`; pathwise: append `totals` arrays to `acc.totals` and run the legacy formula); call `_finalize_dcn_result`. Project by operation: `PRICE -> result.pv`, `PRICE_DETAILED -> result`, plus economics `(("pv", result.pv), ("std_error", result.std_error))`.

6. `capabilities()` override: `output_kinds = {PV, ERROR_ESTIMATE, EVENT_STATS}`, `operations = {PRICE, PRICE_DETAILED}`, `supported_backends = {"serial", "threads"}`, `prepared_state_thread_safe=True`, `instance_reentrant=False`, `fixed_planning=True`, `peak_memory_estimate="conservative"`, `adapter_id="dcn-batch-mc"`, `adapter_version="1"`. `validate()` override allows requested outputs within that set.

- [ ] **Step 1: Write the failing tests**

```python
# test/execution/test_dcn_batch_adapter.py
"""DCN batch adapter unit tests: plans, compact outcomes, exact reduction."""
import numpy as np
import pytest

from quantark.asset.equity.engine.mc.dcn_execution_adapters import (
    DCNBatchMCAdapter,
)
from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.execution.backends import serial
from quantark.execution.context import default_context
from quantark.execution.contracts import PricingRequest

from execution_fixture_helpers import make_dcn_product_env  # see Step 2 note


def price_via_adapter(engine, product, env, context=None):
    adapter = DCNBatchMCAdapter()
    request = PricingRequest(product=product, pricing_env=env)
    context = context or default_context()
    state = adapter.prepare(engine, request, context)
    try:
        plan = adapter.plan_batches(engine, request, state, context)
        outcomes = serial.iter_ordered(
            plan, lambda t: adapter.execute_batch(t, state, context)
        )
        value, economics = adapter.reduce_batches(
            outcomes, plan, state, context
        )
        return value, economics, plan, state
    finally:
        for handle in state.handles:
            handle.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_paths=2048, num_batches=4, use_sobol=True),      # scramble_means
        dict(num_paths=2048, num_batches=1, use_sobol=True),      # pathwise, 1 batch
        dict(num_paths=2048, num_batches=4, use_sobol=False),     # pathwise IID
        dict(num_paths=2048, num_batches=4, use_sobol=False,
             use_antithetic=True),                                # antithetic
    ],
)
def test_bitwise_identical_to_direct(kwargs):
    product, env = make_dcn_product_env()
    engine = DCNMCEngine(seed=7, **kwargs)
    direct = engine.price_detailed(product, env)
    value, economics, plan, _ = price_via_adapter(
        DCNMCEngine(seed=7, **kwargs), product, env
    )
    for field in type(direct).__dataclass_fields__:
        if field in ("elapsed_seconds", "event_stats"):
            continue
        assert getattr(value, field) == getattr(direct, field), field
    assert dict(economics)["pv"] == direct.pv


def test_outcome_payload_is_compact():
    product, env = make_dcn_product_env()
    engine = DCNMCEngine(seed=7, num_paths=2048, num_batches=4, use_sobol=True)
    adapter = DCNBatchMCAdapter()
    request = PricingRequest(product=product, pricing_env=env)
    context = default_context()
    state = adapter.prepare(engine, request, context)
    plan = adapter.plan_batches(engine, request, state, context)
    outcome = adapter.execute_batch(plan.tasks[0], state, context)
    stats = outcome.payload
    assert stats.totals is None  # scramble_means: no per-path retention
    for name in ("fixed_by_period", "ko_by_period", "coupon_paid", "ko_timing"):
        assert getattr(stats, name).size == state.payload.n_obs


def test_plan_reflects_engine_configuration():
    product, env = make_dcn_product_env()
    engine = DCNMCEngine(seed=7, num_paths=4096, num_batches=8, use_sobol=True)
    adapter = DCNBatchMCAdapter()
    request = PricingRequest(product=product, pricing_env=env)
    context = default_context()
    state = adapter.prepare(engine, request, context)
    plan = adapter.plan_batches(engine, request, state, context)
    assert plan.num_batches == 8 and plan.paths_per_batch == 512
    assert plan.stderr_mode == "scramble_means"
    assert plan.tasks[3].batch_id == 3
    assert plan.est_task_peak_bytes > 0


def test_clone_never_mutates_borrowed_engine():
    product, env = make_dcn_product_env()
    engine = DCNMCEngine(seed=7, num_paths=1024, num_batches=2, num_workers=4)
    price_via_adapter(engine, product, env)
    assert engine._last_result is None
    assert engine._draw_provider is None
    assert engine.num_workers == 4
```

- [ ] **Step 2: Fixture helper**

`make_dcn_product_env` — reuse the DCN fixture already present in `test/execution/matrix_fixtures.py` (the `DCNMCEngine` row). Extract the product/env construction into a small shared helper module `test/execution/execution_fixture_helpers.py` importing from `matrix_fixtures` (or add the helper to `matrix_fixtures.py` directly and import it from both — follow the existing import style in `test/execution/`).

- [ ] **Step 3: Run to verify failure**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/execution/test_dcn_batch_adapter.py -v`
Expected: FAIL — `DCNBatchMCAdapter` missing.

- [ ] **Step 4: Implement adapters + registry pins** (per Interfaces above)

Registry (`quantark/execution/registry.py`) — extend the lazy DCN factories:

```python
def _dcn_batch_mc_adapter():
    from quantark.asset.equity.engine.mc.dcn_execution_adapters import (
        DCNBatchMCAdapter,
    )
    return DCNBatchMCAdapter()


def _legacy_product_env_adapter():
    from quantark.execution.legacy_adapter import LegacyPriceAdapter

    return LegacyPriceAdapter(call_shape="product_env")
```

registered for `quantark...dcn_mc_engine.DCNMCEngine` -> `_dcn_batch_mc_adapter`, and exact pins for `...dcn_vol_mc_engines.HestonDCNMCEngine` and `...dcn_vol_mc_engines.CoupledCoarseHestonDCNMCEngine` -> `_legacy_product_env_adapter` (MRO containment; QE resolves via the Heston pin). `LocalVolDCNMCEngine`'s existing factory now returns the batch-capable LV adapter.

- [ ] **Step 5: Run tests to verify pass**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/execution/test_dcn_batch_adapter.py test/execution/test_registry.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add quantark/asset/equity/engine/mc/dcn_execution_adapters.py quantark/execution/registry.py test/execution/test_dcn_batch_adapter.py test/execution/execution_fixture_helpers.py
git commit -m "feat(execution): DCN batch adapters with compact bit-exact outcomes; Heston DCN MRO pins"
```

---

### Task 7: Kernel batch dispatch path

**Files:**
- Modify: `quantark/execution/kernel.py`
- Modify: `quantark/execution/context.py` (add `draw_repository: object | None = None`)
- Test: extend `test/execution/test_dcn_batch_adapter.py` with kernel/session-level tests (session wiring itself is Task 8; here use hand-built contexts)

Kernel changes inside `dispatch`, after `prepare`:

```python
            if hasattr(adapter, "plan_batches"):
                plan = adapter.plan_batches(engine, request, prepared, context)
                plan_fingerprint = try_fingerprint(plan)
                value, economics = _run_batch_plan(
                    adapter, plan, prepared, context, caps, clamp_records
                )
            else:
                plan_fingerprint = None
                value, economics = adapter.execute_native(
                    engine, request, normalized, context, prepared=prepared
                )
```

with:

```python
def _run_batch_plan(adapter, plan, prepared, context, caps, clamp_records):
    backend = context.execution_policy.batch.backend
    if backend == "threads":
        if not caps.prepared_state_thread_safe:
            raise CapabilityError(
                f"adapter {caps.adapter_id!r} prepared state is not "
                "thread-safe; threads backend rejected"
            )
        budget = context.resource_budget
        requested = context.execution_policy.batch.workers
        workers = max(1, min(requested, budget.max_threads, plan.num_batches))
        if workers < requested:
            clamp_records.append(
                f"clamp:batch.workers={requested}->{workers}"
            )
        # window is bounded by max_in_flight: batch tasks hold per-task
        # slots, so admission control binds on CONCURRENT BATCH EXECUTION
        # (Codex plan-gate finding, 2026-07-16).
        requested_window = (
            context.execution_policy.batch.max_in_flight or workers
        )
        window = max(1, min(requested_window, budget.max_in_flight))
        if window < requested_window:
            clamp_records.append(
                f"clamp:batch.window={requested_window}->{window}"
            )
        outcomes = threads.iter_ordered(
            plan, lambda t: adapter.execute_batch(t, prepared, context),
            workers=workers, window=window,
            lease_manager=context.lease_manager,
        )
    else:
        outcomes = serial.iter_ordered(
            plan, lambda t: adapter.execute_batch(t, prepared, context),
            lease_manager=context.lease_manager,
        )
    return adapter.reduce_batches(outcomes, plan, prepared, context)
```

Backend imports at module top (`from quantark.execution.backends import serial, threads` — framework-internal, no asset imports). `plan_fingerprint` goes into the manifest's existing `plan_fingerprint` field; clamp records join the diagnostics `records` tuple; diagnostics timings gain `("batch_count", float(plan.num_batches))` only when a plan ran. Backend validation change: batch-capable adapters advertise `{"serial", "threads"}`, so the existing top-of-dispatch backend check keeps rejecting `threads` for non-batch adapters (unchanged code path).

**Dispatch-slot narrowing:** when the adapter has `plan_batches`, the kernel does NOT wrap execution in its dispatch-wide `task_slot()` — the backends acquire one slot per executing batch instead (Task 4's admission contract), so `max_in_flight=1` genuinely serializes batch execution rather than merely serializing dispatches while a wide window runs many batches inside one slot. Non-batch dispatches keep the Phase-1 dispatch-wide slot unchanged. Add a kernel-level test: `max_in_flight=1`, threads backend, `workers=4` on an 8-batch DCN plan → result still bitwise equal and a concurrency probe (adapter subclass counting concurrent `execute_batch` entries) observes max 1.

- [ ] **Step 1: Write the failing tests** (append to `test_dcn_batch_adapter.py`)

```python
def make_batch_context(workers=1, backend="serial", **budget_kw):
    import dataclasses

    from quantark.execution.cache.artifacts import PreparedArtifactCache
    from quantark.execution.cache.draws import DrawRepository
    from quantark.execution.leases import ResourceLeaseManager
    from quantark.execution.policy import (
        ExecutionPolicy, ExecutorSelection, ResourceBudget,
    )

    budget_kw.setdefault("max_in_flight", 8)  # batch tasks hold per-task slots
    budget = ResourceBudget(
        max_threads=8, artifact_cache_bytes=64 * 2**20,
        draw_cache_bytes=64 * 2**20, **budget_kw,
    )
    leases = ResourceLeaseManager(budget)
    context = default_context()
    return dataclasses.replace(
        context,
        execution_policy=ExecutionPolicy(
            batch=ExecutorSelection(backend=backend, workers=workers)
        ),
        resource_budget=budget,
        lease_manager=leases,
        artifact_cache=PreparedArtifactCache(leases),
        draw_repository=DrawRepository(leases),
    )


def test_kernel_batch_path_serial_and_threads_bitwise_equal():
    from quantark.execution.kernel import ExecutionKernel
    from quantark.execution.contracts import PricingOperation

    product, env = make_dcn_product_env()
    direct = DCNMCEngine(
        seed=7, num_paths=2048, num_batches=8
    ).price_detailed(product, env)

    def run(backend, workers):
        engine = DCNMCEngine(seed=7, num_paths=2048, num_batches=8)
        request = PricingRequest(
            product=product, pricing_env=env,
            operation=PricingOperation.PRICE_DETAILED,
        )
        ctx = make_batch_context(workers=workers, backend=backend)
        return ExecutionKernel.dispatch(engine, request, ctx)

    serial_out = run("serial", 1)
    threads_out = run("threads", 4)
    for field in type(direct).__dataclass_fields__:
        if field in ("elapsed_seconds", "event_stats"):
            continue
        assert getattr(serial_out.value, field) == getattr(direct, field)
        assert getattr(threads_out.value, field) == getattr(direct, field)
    assert serial_out.manifest.plan_fingerprint is not None
    assert (serial_out.manifest.plan_fingerprint
            == threads_out.manifest.plan_fingerprint)


def test_kernel_records_worker_clamp():
    from quantark.execution.kernel import ExecutionKernel

    product, env = make_dcn_product_env()
    engine = DCNMCEngine(seed=7, num_paths=1024, num_batches=4)
    ctx = make_batch_context(workers=64, backend="threads")
    outcome = ExecutionKernel.dispatch(
        engine, PricingRequest(product=product, pricing_env=env), ctx
    )
    assert any("clamp:batch.workers=64->" in r
               for r in outcome.diagnostics.records)


def test_draws_pinned_bytes_released_after_dispatch():
    from quantark.execution.kernel import ExecutionKernel

    product, env = make_dcn_product_env()
    engine = DCNMCEngine(seed=7, num_paths=1024, num_batches=4)
    ctx = make_batch_context(workers=2, backend="threads")
    ExecutionKernel.dispatch(
        engine, PricingRequest(product=product, pricing_env=env), ctx
    )
    stats = ctx.draw_repository.stats()
    assert stats["misses"] == 4          # one Sobol block per batch
    assert stats["bytes_in_use"] > 0     # masters retained for CRN reuse
    ExecutionKernel.dispatch(            # identical repricing: all hits
        engine, PricingRequest(product=product, pricing_env=env), ctx
    )
    assert ctx.draw_repository.stats()["misses"] == 4
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/execution/test_dcn_batch_adapter.py -v`
Expected: new tests FAIL (`draw_repository` context field / batch path missing).

- [ ] **Step 3: Implement** kernel + context changes per the sketch above.

- [ ] **Step 4: Run the execution suite**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/execution/ -q`
Expected: PASS (matrix parity in particular — DCN rows now take the batch path and must remain exactly equal).

- [ ] **Step 5: Commit**

```bash
git add quantark/execution/kernel.py quantark/execution/context.py test/execution/test_dcn_batch_adapter.py
git commit -m "feat(execution): kernel batch dispatch with plan fingerprints and thread clamps"
```

---

### Task 8: Session wiring for DrawRepository

**Files:**
- Modify: `quantark/execution/api.py`
- Modify: `quantark/execution/__init__.py` (export `BatchPlan`, `BatchTask`, `BatchOutcome`, `DrawRepository`)
- Test: extend `test/execution/test_kernel_prepare.py` (where Phase-1 pairing tests live)

Rules (mirror the Phase-1 pair contract):
- If the caller supplies `draw_repository`, it must be backed by the same lease manager (`ValidationError` otherwise) and the session never closes it.
- If absent, the session creates one from the (owned or borrowed) lease manager and owns/closes it.
- When the session constructs an OWNED budget (no context supplied): default `draw_cache_bytes` 512 MiB, `max_threads = os.cpu_count() or 1`, and `max_in_flight = os.cpu_count() or 1` (batch tasks now hold per-task slots, so the Phase-1 default of 1 would serialize every threaded plan) — all recorded via the manifest's `policy_values` (already value-bearing). Env-resolved and explicit budgets are never upgraded — only the owned auto-budget (spec §11.1 "safe auto budget").

- [ ] **Step 1: Write the failing tests** (append to `test_kernel_prepare.py`)

```python
def test_session_owns_draw_repository_and_closes_it():
    session = PricingSession()
    repo = session.context.draw_repository
    assert repo is not None
    assert repo.lease_manager is session.context.lease_manager
    session.close()
    with pytest.raises(PreparationError):
        repo.normals_handle(seed=1, n_paths=8, dim=2, batch_id=0)


def test_supplied_draw_repository_must_share_lease_manager():
    budget = ResourceBudget(artifact_cache_bytes=1024, draw_cache_bytes=1024)
    leases = ResourceLeaseManager(budget)
    other = ResourceLeaseManager(budget)
    cache = PreparedArtifactCache(leases)
    foreign_repo = DrawRepository(other)
    context = dataclasses.replace(
        default_context(), lease_manager=leases, artifact_cache=cache,
        draw_repository=foreign_repo,
    )
    with pytest.raises(ValidationError):
        PricingSession(context)


def test_borrowed_draw_repository_survives_session_close():
    budget = ResourceBudget(artifact_cache_bytes=1024, draw_cache_bytes=2**20)
    leases = ResourceLeaseManager(budget)
    cache = PreparedArtifactCache(leases)
    repo = DrawRepository(leases)
    context = dataclasses.replace(
        default_context(), lease_manager=leases, artifact_cache=cache,
        draw_repository=repo,
    )
    session = PricingSession(context)
    session.close()
    with repo.normals_handle(seed=1, n_paths=8, dim=2, batch_id=0) as h:
        assert h.value.shape == (8, 2)


def test_owned_default_budget_has_cpu_threads_and_in_flight():
    import os

    session = PricingSession()
    budget = session.context.resource_budget
    assert budget.max_threads == (os.cpu_count() or 1)
    assert budget.max_in_flight == (os.cpu_count() or 1)
    session.close()
```

- [ ] **Step 2: Run to verify failure**, implement in `api.py` (extend the existing pair-validation block; owned repo tracked in `self._owned_draw_repo` and closed in `close()`), update `__init__.py` exports.

- [ ] **Step 3: Run the execution suite**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/execution/ -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add quantark/execution/api.py quantark/execution/__init__.py test/execution/test_kernel_prepare.py
git commit -m "feat(execution): session-owned DrawRepository with paired-injection validation"
```

---

### Task 9: Inventory batch-capability audit

**Files:**
- Modify: `quantark/execution/inventory.py`
- Test: extend `test/execution/test_inventory.py`

**Interfaces:**
- Produces: `InventoryRecord.batch_state: str = "temporary_legacy"` and `batch_rationale: str = ""`; `BATCH_STATES = ("batch_capable", "temporary_legacy", "not_applicable")`.

Audit assignments (MC rows only; PDE rows get `not_applicable` / rationale `"PDE solve has no batch axis; Phase 4 covers PDE preparation"`):

| Engines | batch_state | rationale |
|---|---|---|
| `DCNMCEngine`, `LocalVolDCNMCEngine` | `batch_capable` | (empty — capability advertised by the registered batch adapter) |
| `HestonDCNMCEngine`, `QEDCNMCEngine`, `CoupledCoarseHestonDCNMCEngine` | `temporary_legacy` | "Heston draw pipeline (paired normal+uniform streams) not yet routed through DrawRepository; Phase 3 model families" |
| Snowball/Phoenix MC ×12 (`planning="both"`) | `temporary_legacy` | "adaptive RQMC compatibility stopping is sequential by contract (spec 8.4); parallel-wave is a Phase 3 opt-in plan" |
| `AmericanOptionMCEngine` | `not_applicable` | "LSM regression couples all paths cross-sectionally; no independent batch decomposition exists" |
| `EuropeanMCEngine`, `AsianOptionMCEngine`, `DigitalOptionMCEngine`, `BarrierOptionMCEngine`, `SingleSharkfinOptionMCEngine`, `DoubleSharkfinOptionMCEngine`, `RangeAccrualMCEngine`, `AccumulatorMCEngine`, `SABRMCEngine`, `LocalVolMCEngine`, `HestonMCEngine`, `HestonSLVMCEngine`, `LocalVolBarrierMCEngine`, `HestonBarrierMCEngine`, `HestonSLVBarrierMCEngine` | `temporary_legacy` | "single-solve engine without a batch axis; introducing one is a changed numerical plan — re-scope on Phase 2 benchmark evidence (spec 21)" |
| FX MC rows (8) | `temporary_legacy` | "FX MC engines are single-solve on the shared montecarlo layer; batch decomposition re-scoped on Phase 2 benchmark evidence" |

(Audit the FX rows against their actual engine code during implementation; if any has a genuine internal batch loop, say so in its rationale instead.)

- [ ] **Step 1: Write the failing test** (append to `test_inventory.py`)

```python
def test_every_mc_row_has_audited_batch_state():
    from quantark.execution.inventory import BATCH_STATES, ENGINE_INVENTORY

    for record in ENGINE_INVENTORY:
        assert record.batch_state in BATCH_STATES, record.name
        if record.engine_type == "mc" and record.batch_state != "batch_capable":
            assert record.batch_rationale.strip(), (
                f"{record.name}: non-capable MC rows need a specific rationale"
            )


def test_batch_capable_rows_resolve_to_batch_adapters():
    from quantark.execution.inventory import ENGINE_INVENTORY
    from quantark.execution.registry import build_default_registry

    registry = build_default_registry()
    for record in ENGINE_INVENTORY:
        if record.batch_state != "batch_capable":
            continue
        module_path, _, cls_name = record.import_path.rpartition(".")
        module = __import__(module_path, fromlist=[cls_name])
        engine_cls = getattr(module, cls_name)
        adapter = registry.resolve_class(engine_cls)
        assert hasattr(adapter, "plan_batches"), record.name
```

(If `registry.resolve_class` does not exist, add it as a thin classmethod-style variant of `resolve` that takes the class instead of an instance — same MRO walk.)

- [ ] **Step 2: Run to verify failure, implement, re-run**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/execution/test_inventory.py -v`
Expected: PASS after implementation (including the pre-existing discovery gate).

- [ ] **Step 3: Commit**

```bash
git add quantark/execution/inventory.py quantark/execution/registry.py test/execution/test_inventory.py
git commit -m "feat(execution): batch-capability audit on the engine inventory (Phase 2 exit gate)"
```

---

### Task 10: Structural CI gates and full-matrix regression

**Files:**
- Modify: `test/execution/test_regression_gates.py`
- Test: itself

Deterministic structural gates standing in for the §20 wall-clock gates (kickoff decision):

- [ ] **Step 1: Add gates**

```python
def test_threads_any_worker_count_bitwise_equal_serial():
    # workers in {1, 2, 3, 8} over 8 batches: canonical reduction makes every
    # worker count bit-identical (gate: reduction order, not timing)
    ...


def test_bounded_outcome_buffering_no_unbounded_retention():
    # instrument threads.iter_ordered observer on a DCN plan: buffered <= window
    # and outcome payloads contain no path arrays in scramble_means mode
    ...


def test_crn_repricing_reuses_draws_and_surface():
    # one session, 10 spot-bumped repricings, same seed: draw misses stay at
    # num_batches; Dupire builds stay at 1 per distinct vol surface
    ...


def test_disabled_caches_bitwise_exact():
    # draw_cache_bytes=0 and artifact_cache_bytes=0: results still bitwise
    # equal direct (correctness never depends on caching)
    ...
```

Write these fully (they follow the patterns from Tasks 6–7's tests; the CRN gate uses `LocalVolDCNMCEngine` via `PricingSession` and asserts on `draw_repository.stats()` / Dupire build-count spy exactly as Phase 1's `test_kernel_prepare.py` does).

- [ ] **Step 2: Run the full execution suite + DCN suite**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/execution/ -q && PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q -k "dcn"`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add test/execution/test_regression_gates.py
git commit -m "test(execution): Phase 2 structural exit gates (order, buffering, CRN reuse, cache-off exactness)"
```

---

### Task 11: Benchmark script and recorded results

**Files:**
- Create: `test/execution/benchmark_phase2.py` (no `test_` prefix — not collected)
- Create: `docs/superpowers/benchmarks/2026-07-16-execution-phase2-benchmark.md` (needs `git add -f`)

- [ ] **Step 1: Write the script**

`benchmark_phase2.py` measures, each with ≥5 post-warm-up repetitions reporting median + IQR (spec §20 protocol):

1. **Gate 3 (thread scaling):** `LocalVolDCNMCEngine` (2^17 paths, 16 batches) through `PricingSession` with `threads` backend at workers ∈ {1, 2, 4, 8}; report speedup vs workers=1. Cold vs warm draw cache reported separately (mechanism attribution).
2. **Gate 4 (CRN reuse):** 10 spot-bumped repricings (same seed) in one session — cached (default budgets) vs uncached (`draw_cache_bytes=0`, `artifact_cache_bytes=0`); report ratio.
3. **Serial overhead check:** direct `price_detailed` vs session-serial median (supports the standing ≤3% gate 1).

Output: a markdown table printed to stdout and written to the docs path. Uses `time.perf_counter`, `statistics.median`; honest environment header (`platform_tag()`, CPU count, path count).

- [ ] **Step 2: Run it and record**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python test/execution/benchmark_phase2.py`
Expected: gate 3 ≥1.5x@4 / ≥2.5x@8 (16 batches = 2+ per worker at 8), gate 4 ≥2x, serial overhead ≤3%. Paste actual numbers into the markdown file. If a gate misses, STOP and investigate before merging — do not record a failing number as passing.

- [ ] **Step 3: Commit**

```bash
git add test/execution/benchmark_phase2.py
git add -f docs/superpowers/benchmarks/2026-07-16-execution-phase2-benchmark.md
git commit -m "bench(execution): Phase 2 speed-gate benchmark script + recorded results"
```

---

### Task 12: Full-suite verification

- [ ] **Step 1:** `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q`
Expected: everything green except the known pre-existing `test_snowball_quad_flat_identity_golden`.

- [ ] **Step 2:** Verify the direct path is untouched where it must be: `git diff main --stat` shows changes only in `quantark/execution/`, the two DCN mc files, `test/execution/`, and docs.

---

## Self-Review Notes

- **Spec coverage:** §8.1 (BatchPlan fields) → Task 1; §8.2 (compact outcomes, bounded buffering, canonical order, legacy arithmetic preservation) → Tasks 4+6; §8.3 (DrawRepository, complete descriptors, read-only masters, writable copies, single-flight, lease-backed pinning) → Tasks 3+6; §8.4 fixed-batch serial/threads → Tasks 4+7; §8.5 single-solve bundle → Task 6 (one traversal, operation projection); §11 (task-scratch admission, single-oversized-task failure, clamp diagnostics) → Tasks 2+4+7; §12.1–12.2 → Tasks 4+7; §12.5 (clones run `num_workers=1`) → Task 6; §21 exit gate (audit) → Task 9; §20 gates → Tasks 10 (structural) + 11 (measured).
- **Bit-identity argument** (the load-bearing claim): legacy reduces per-batch via `acc.add` in batch order; per-batch stats are exactly the increments `add` produces from a single-batch accumulator; merging them in batch-index order performs the same float additions in the same order; stderr branches reuse identical expressions; `_finalize_dcn_result` is shared code. Threads change scheduling, never reduction order.
- **Not fixed by design:** Heston DCN engines stay legacy (exact-match registry pins prevent MRO leakage); the process-global `QMCDrawCache` on the direct path is untouched; `LegacyPriceAdapter` PV-only guarantee unchanged.
- **Plan-gate findings applied (Codex, 2026-07-16, 1 iteration):** (1) `max_in_flight` now binds concurrent batch execution — per-task slots in both backends, kernel window clamp, dispatch-slot narrowing for batch plans, owned auto-budget upgrade; (2) buffered outcomes stay byte-charged until the reducer consumes them, with submission gated on `pending + buffered < window` (stalled-batch-0 pathwise test added).
