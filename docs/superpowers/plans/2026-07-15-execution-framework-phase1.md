# Execution Framework Phase 1 — Serial Kernel and Preparation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement spec Phase 1: canonical value fingerprints, a resource
lease manager, the session-owned `PreparedArtifactCache` (single-flight,
LRU-by-bytes, pinned handles), the upgraded kernel lifecycle
(reserve → prepare → execute → release), the DCN local-vol preparation
adapter (factory-clone, no mutable engine state on the adapter path), and the
**full-matrix** direct-vs-session parity gate over every concrete inventory
row, plus disabled-cache and serial-overhead regression gates.

**Architecture:** New `quantark/execution/cache/` (fingerprint + artifacts)
and `quantark/execution/leases.py`. The kernel gains an optional
prepare step driven structurally (`hasattr(adapter, "prepare")`). The DCN
adapter lives asset-side (`quantark/asset/equity/engine/mc/dcn_execution_adapters.py`,
spec §4.1) and serves cache-fetched Dupire surfaces through **prebuilt-surface
factory clones** — the engines' `_prepare_simulation`/`_resolve_surface` hooks
and `_active_surface` state are untouched on the direct path (spec §17.1).

**Tech Stack:** stdlib (`dataclasses`, `hashlib`, `threading`), NumPy. pytest.

**Spec:** `docs/superpowers/specs/2026-07-15-mc-pde-performance-generalization-design.md`
§7, §10, §11, §21 Phase 1. **Baseline:** Phase 0 merged (`d6f4241..c0c786e`).

## Global Constraints

- Run tests with `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest`
  in the worktree (editable install shadows otherwise).
- `quantark/execution/` still never statically imports asset code; the DCN
  adapter module lives under `quantark/asset/equity/` and imports execution
  (allowed direction). `registry.py` registers it by string path with a lazy
  factory import.
- Frozen dataclasses; no mutable global/thread-local run state (spec §3.3).
- Direct legacy behavior unchanged. The mini-project hooks
  (`_prepare_simulation`/`_resolve_surface`) keep their names, semantics, and
  mutable pattern on the direct path (spec §17.1). NO edits to
  `dcn_mc_engine.py`, `dcn_vol_mc_engines.py`, `dcn_pde_solver.py`,
  `dcn_vol_pde_solvers.py`.
- Cached values immutable: cached `LocalVolSurface` objects are handed to
  clones read-only; clones never mutate them (`_prebuilt` path is read-only by
  construction).
- Legacy exceptions propagate unwrapped on adapter execution.
- Commit per task: `feat(execution): ...` / `test(execution): ...` with the
  Claude co-author trailer.
- Fixture parameters: cheapest observed in the source tests (cited per
  family); the matrix must stay under ~90s wall.

## File Structure

```
quantark/execution/
    leases.py                # ResourceLeaseManager
    cache/__init__.py
    cache/fingerprint.py     # canonical_tree / fingerprint / try_fingerprint
    cache/artifacts.py       # ArtifactDescriptor, ArtifactHandle, PreparedArtifactCache
    contracts.py             # + PreparedState (additive)
    manifest.py              # + preparation_fingerprint field (additive, defaulted)
    context.py               # + artifact_cache, lease_manager handles (additive)
    kernel.py                # lifecycle: reserve -> prepare -> execute -> release
    api.py                   # session owns cache + lease manager; close() releases
    registry.py              # + DCN adapter registrations (lazy factory import)
    inventory.py             # DCN LV rows -> adoption_state "supported"
quantark/asset/equity/engine/mc/
    dcn_execution_adapters.py  # DCNLocalVolMCAdapter, DCNLocalVolPDEAdapter
test/execution/
    test_fingerprint.py
    test_leases.py
    test_artifact_cache.py
    test_kernel_prepare.py    # lifecycle + DCN adapter build-count/parity
    matrix_fixtures.py        # 73 fixture builders (from mined test recipes)
    test_matrix_parity.py     # full-matrix direct-vs-session gate
    test_regression_gates.py  # disabled-cache parity + serial-overhead smoke
```

---

### Task 1: Canonical value fingerprints

**Files:**
- Create: `quantark/execution/cache/__init__.py`, `quantark/execution/cache/fingerprint.py`
- Test: `test/execution/test_fingerprint.py`

**Interfaces:**
- Produces: `Uncanonicalizable(Exception)`; `canonical_tree(obj) -> tuple`;
  `fingerprint(obj) -> str` (sha256 hex); `try_fingerprint(obj) -> str | None`.
- Safe leaves: `None`/`bool`/`int`/`str`; `float` via `.hex()`; `datetime`/`date`
  ISO; `Enum` by qualname+name; `np.ndarray` by dtype/shape/content-sha;
  `tuple`/`list`; `dict` with str keys (sorted); dataclass instances recurse
  fields with class qualname. Anything else raises `Uncanonicalizable`
  (spec §10.1: no safe canonicalizer → uncacheable).

- [ ] **Step 1: Write the failing test**

```python
# test/execution/test_fingerprint.py
"""Canonical value fingerprints (spec section 10.1)."""
from datetime import datetime

import numpy as np
import pytest

from quantark.execution.cache.fingerprint import (
    Uncanonicalizable,
    fingerprint,
    try_fingerprint,
)


def test_equal_valued_dataclasses_share_fingerprint():
    from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote

    assert fingerprint(FlatRateCurve(rate=0.05)) == fingerprint(
        FlatRateCurve(rate=0.05)
    )
    assert fingerprint(FlatRateCurve(rate=0.05)) != fingerprint(
        FlatRateCurve(rate=0.051)
    )
    assert fingerprint(SpotQuote(spot=100.0)) != fingerprint(
        FlatVolSurface(volatility=100.0)
    )  # class identity participates


def test_grid_surface_and_composites():
    from quantark.param import GridVolSurface

    def grid():
        return GridVolSurface(
            strikes=[90.0, 100.0, 110.0],
            maturities=[0.5, 1.0],
            iv_grid=np.full((2, 3), 0.2),
        )

    assert fingerprint(grid()) == fingerprint(grid())
    changed = grid()
    changed.iv_grid = np.full((2, 3), 0.21)
    assert fingerprint(grid()) != fingerprint(changed)
    assert fingerprint((grid(), datetime(2026, 1, 1))) == fingerprint(
        (grid(), datetime(2026, 1, 1))
    )


def test_unregistered_type_is_uncanonicalizable():
    class Opaque:
        pass

    with pytest.raises(Uncanonicalizable):
        fingerprint(Opaque())
    assert try_fingerprint(Opaque()) is None
    assert try_fingerprint({"k": Opaque()}) is None


def test_float_precision_and_nan_distinct():
    assert fingerprint(0.1) != fingerprint(0.1 + 1e-17) or (0.1 == 0.1 + 1e-17)
    assert fingerprint(float("nan")) == fingerprint(float("nan"))
    assert fingerprint(1.0) != fingerprint(1)  # float vs int tagged apart
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -n0 test/execution/test_fingerprint.py -q`
Expected: FAIL `ModuleNotFoundError: quantark.execution.cache`

- [ ] **Step 3: Write minimal implementation**

`quantark/execution/cache/__init__.py`: `"""Draw/artifact caches and canonical fingerprints (spec section 10)."""`

`quantark/execution/cache/fingerprint.py`:

```python
"""Canonical value fingerprints (spec section 10.1).

Keys derive from VALUES, never Python object identity. A type without a safe
canonicalizer raises ``Uncanonicalizable``; callers treat that artifact as
uncacheable and build fresh — correctness never depends on cacheability.
"""
import dataclasses
import hashlib
from datetime import date, datetime
from enum import Enum

import numpy as np

__all__ = ["Uncanonicalizable", "canonical_tree", "fingerprint", "try_fingerprint"]


class Uncanonicalizable(Exception):
    """No safe canonicalizer exists for this value."""


def canonical_tree(obj):
    if obj is None or isinstance(obj, (bool, int, str)):
        # bool before int is irrelevant here: tag carries the concrete type.
        return ("atom", type(obj).__name__, obj)
    if isinstance(obj, float):
        if obj != obj:  # NaN: hex() differs across signs/payloads
            return ("float", "nan")
        return ("float", obj.hex())
    if isinstance(obj, (datetime, date)):
        return ("dt", obj.isoformat())
    if isinstance(obj, Enum):
        return ("enum", f"{type(obj).__module__}.{type(obj).__qualname__}", obj.name)
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return (
            "nd", str(arr.dtype), tuple(arr.shape),
            hashlib.sha256(arr.tobytes()).hexdigest(),
        )
    if isinstance(obj, np.generic):
        return canonical_tree(obj.item())
    if isinstance(obj, (tuple, list)):
        return ("seq", tuple(canonical_tree(x) for x in obj))
    if isinstance(obj, dict):
        if not all(isinstance(k, str) for k in obj):
            raise Uncanonicalizable("dict keys must be strings")
        return (
            "map",
            tuple((k, canonical_tree(v)) for k, v in sorted(obj.items())),
        )
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = type(obj)
        return (
            "dc", f"{cls.__module__}.{cls.__qualname__}",
            tuple(
                (f.name, canonical_tree(getattr(obj, f.name)))
                for f in dataclasses.fields(cls)
            ),
        )
    raise Uncanonicalizable(
        f"no canonicalizer for {type(obj).__module__}.{type(obj).__qualname__}"
    )


def fingerprint(obj) -> str:
    tree = canonical_tree(obj)
    return hashlib.sha256(repr(tree).encode()).hexdigest()


def try_fingerprint(obj):
    try:
        return fingerprint(obj)
    except Uncanonicalizable:
        return None
```

- [ ] **Step 4: Run test to verify it passes** — same command, expected 4 PASS.
- [ ] **Step 5: Commit** — `feat(execution): canonical value fingerprints`

---

### Task 2: Resource lease manager

**Files:**
- Create: `quantark/execution/leases.py`
- Test: `test/execution/test_leases.py`

**Interfaces:**
- Produces: `ResourceLeaseManager(budget: ResourceBudget)` with
  `task_slot()` (context manager enforcing `max_in_flight`; raises
  `ResourceBudgetExceeded` when exhausted rather than blocking — Phase 1 is
  serial so contention means a bug), `lease_bytes(n, pool: str)` /
  `release_bytes(n, pool)` for the `"artifact_cache"` pool bounded by
  `budget.artifact_cache_bytes` (None = unlimited), `pool_bytes(pool) -> int`,
  and `close()`. Thread-safe via one `threading.Lock`.

- [ ] **Step 1: Write the failing test**

```python
# test/execution/test_leases.py
"""Resource lease accounting (spec section 11, Phase 1 subset)."""
import pytest

from quantark.execution.errors import ResourceBudgetExceeded
from quantark.execution.leases import ResourceLeaseManager
from quantark.execution.policy import ResourceBudget


def test_task_slot_enforces_max_in_flight():
    mgr = ResourceLeaseManager(ResourceBudget(max_in_flight=1))
    with mgr.task_slot():
        with pytest.raises(ResourceBudgetExceeded):
            with mgr.task_slot():
                pass
    with mgr.task_slot():  # released correctly
        pass


def test_byte_leases_enforce_pool_capacity():
    mgr = ResourceLeaseManager(ResourceBudget(artifact_cache_bytes=100))
    mgr.lease_bytes(60, "artifact_cache")
    with pytest.raises(ResourceBudgetExceeded):
        mgr.lease_bytes(50, "artifact_cache")
    mgr.release_bytes(60, "artifact_cache")
    mgr.lease_bytes(100, "artifact_cache")
    assert mgr.pool_bytes("artifact_cache") == 100


def test_unlimited_pool_when_budget_none():
    mgr = ResourceLeaseManager(ResourceBudget(artifact_cache_bytes=None))
    mgr.lease_bytes(10**12, "artifact_cache")  # no limit configured
```

- [ ] **Step 2: Run to verify FAIL** (`ModuleNotFoundError`).
- [ ] **Step 3: Implementation**

```python
# quantark/execution/leases.py
"""Resource lease accounting (spec section 11 — Phase 1 serial subset).

Admission control bounds ADMITTED estimates, not OS memory (spec section 11
as amended). Phase 1 enforces the in-flight task slot and the artifact-cache
byte pool; task-scratch and worker pools arrive with parallel backends in
Phase 2.
"""
import threading

from quantark.execution.errors import ResourceBudgetExceeded
from quantark.execution.policy import ResourceBudget

__all__ = ["ResourceLeaseManager"]


class ResourceLeaseManager:
    def __init__(self, budget: ResourceBudget):
        self._budget = budget
        self._lock = threading.Lock()
        self._in_flight = 0
        self._pools: dict = {}
        self._capacities = {"artifact_cache": budget.artifact_cache_bytes}
        self._closed = False

    def task_slot(self):
        return _TaskSlot(self)

    def lease_bytes(self, n: int, pool: str) -> None:
        with self._lock:
            if self._closed:
                raise ResourceBudgetExceeded(
                    "ResourceLeaseManager is closed; no post-close acquisitions"
                )
            capacity = self._capacities.get(pool)
            used = self._pools.get(pool, 0)
            if capacity is not None and used + n > capacity:
                raise ResourceBudgetExceeded(
                    f"pool {pool!r}: lease of {n} bytes exceeds capacity "
                    f"{capacity} (in use: {used})"
                )
            self._pools[pool] = used + n

    def release_bytes(self, n: int, pool: str) -> None:
        with self._lock:
            self._pools[pool] = max(0, self._pools.get(pool, 0) - n)

    def pool_bytes(self, pool: str) -> int:
        with self._lock:
            return self._pools.get(pool, 0)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._pools.clear()
            self._in_flight = 0

    def _acquire_slot(self) -> None:
        with self._lock:
            if self._closed:
                raise ResourceBudgetExceeded(
                    "ResourceLeaseManager is closed; no post-close acquisitions"
                )
            if self._in_flight >= self._budget.max_in_flight:
                raise ResourceBudgetExceeded(
                    f"max_in_flight={self._budget.max_in_flight} exhausted"
                )
            self._in_flight += 1

    def _release_slot(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)


class _TaskSlot:
    def __init__(self, mgr):
        self._mgr = mgr

    def __enter__(self):
        self._mgr._acquire_slot()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._mgr._release_slot()
        return False
```

- [ ] **Step 4: Run to PASS (3 tests).**
- [ ] **Step 5: Commit** — `feat(execution): resource lease manager (serial subset)`

---

### Task 3: PreparedArtifactCache

**Files:**
- Create: `quantark/execution/cache/artifacts.py`
- Test: `test/execution/test_artifact_cache.py`

**Interfaces:**
- Produces: `ArtifactDescriptor(kind: str, fingerprint: str, dependency_tags: frozenset, builder_version: str)` — frozen.
- Produces: `ArtifactHandle` — context manager; `.value` (immutable payload);
  exit unpins.
- Produces: `PreparedArtifactCache(lease_manager)` with
  `get_or_build(descriptor, builder, size_bytes) -> ArtifactHandle`,
  `stats() -> dict` (hits, misses, evictions, bytes_in_use,
  single_flight_waits), `invalidate_tags(tags)`, `close()`.
- Semantics (spec §10.1 as hardened): single-flight per key with leader
  election — leader builds under its reservation; waiters block on the
  in-flight entry honoring no cancellation in Phase 1; leader failure
  publishes the exception to CURRENT waiters, removes the in-flight entry and
  reservation atomically, next requester re-elects; hit verifies full
  descriptor equality; eviction is LRU by bytes over UNPINNED entries only;
  pinned bytes stay leased until the last handle closes; `close()` fails
  in-flight builds' waiters and releases everything.

- [ ] **Step 1: Write the failing test**

```python
# test/execution/test_artifact_cache.py
"""Session-owned prepared-artifact cache (spec section 10.1)."""
import threading

import pytest

from quantark.execution.cache.artifacts import (
    ArtifactDescriptor,
    PreparedArtifactCache,
)
from quantark.execution.errors import PreparationError
from quantark.execution.leases import ResourceLeaseManager
from quantark.execution.policy import ResourceBudget


def _desc(fp="f1", kind="k"):
    return ArtifactDescriptor(
        kind=kind, fingerprint=fp,
        dependency_tags=frozenset({"vol_surface"}), builder_version="1",
    )


def _cache(capacity=1000):
    return PreparedArtifactCache(
        ResourceLeaseManager(ResourceBudget(artifact_cache_bytes=capacity))
    )


def test_hit_returns_same_payload_and_counts():
    cache = _cache()
    builds = []

    def build():
        builds.append(1)
        return {"surface": 42}

    with cache.get_or_build(_desc(), build, size_bytes=100) as h1:
        v1 = h1.value
    with cache.get_or_build(_desc(), build, size_bytes=100) as h2:
        assert h2.value is v1
    assert len(builds) == 1
    s = cache.stats()
    assert s["hits"] == 1 and s["misses"] == 1


def test_lru_eviction_respects_pins():
    cache = _cache(capacity=250)
    with cache.get_or_build(_desc("a"), lambda: "A", size_bytes=100):
        # "a" pinned; adding two more forces eviction of the UNPINNED lru
        with cache.get_or_build(_desc("b"), lambda: "B", size_bytes=100):
            pass  # b unpinned after exit
        with cache.get_or_build(_desc("c"), lambda: "C", size_bytes=100):
            pass
    s = cache.stats()
    assert s["evictions"] >= 1
    assert s["bytes_in_use"] <= 250
    builds = []
    with cache.get_or_build(_desc("a"), lambda: builds.append(1) or "A2",
                            size_bytes=100) as h:
        assert h.value == "A"  # pinned entry survived
    assert not builds


def test_oversize_artifact_bypasses_cache():
    cache = _cache(capacity=50)
    with cache.get_or_build(_desc("big"), lambda: "BIG", size_bytes=100) as h:
        assert h.value == "BIG"  # built and returned, not admitted
    assert cache.stats()["bytes_in_use"] == 0


def test_single_flight_leader_failure_releases_waiters():
    cache = _cache()
    started = threading.Event()
    release = threading.Event()
    results = []

    def failing_build():
        started.set()
        release.wait(timeout=5)
        raise ValueError("leader boom")

    def leader():
        try:
            with cache.get_or_build(_desc("x"), failing_build, size_bytes=10):
                pass
        except Exception as exc:
            results.append(("leader", type(exc).__name__))

    def waiter():
        started.wait(timeout=5)
        try:
            with cache.get_or_build(_desc("x"), lambda: "OK", size_bytes=10) as h:
                results.append(("waiter", h.value))
        except Exception as exc:
            results.append(("waiter_exc", type(exc).__name__))

    t1 = threading.Thread(target=leader)
    t2 = threading.Thread(target=waiter)
    t1.start(); started.wait(timeout=5); t2.start()
    release.set(); t1.join(timeout=10); t2.join(timeout=10)
    assert ("leader", "ValueError") in results
    # Waiter either saw the published failure or re-elected and built OK —
    # both are legal; it must NOT hang or see a poisoned key forever.
    assert any(r[0] in ("waiter", "waiter_exc") for r in results)
    with cache.get_or_build(_desc("x"), lambda: "OK", size_bytes=10) as h:
        assert h.value == "OK"  # key not poisoned


def test_invalidate_tags():
    cache = _cache()
    with cache.get_or_build(_desc("a"), lambda: "A", size_bytes=10):
        pass
    cache.invalidate_tags({"vol_surface"})
    builds = []
    with cache.get_or_build(_desc("a"), lambda: builds.append(1) or "A2",
                            size_bytes=10) as h:
        assert h.value == "A2"
    assert builds


def test_close_during_build_fails_leader_and_leaks_nothing():
    """Codex plan-gate finding 3: a leader finishing after close() must not
    publish or retain bytes."""
    import threading

    mgr = ResourceLeaseManager(ResourceBudget(artifact_cache_bytes=1000))
    cache = PreparedArtifactCache(mgr)
    started = threading.Event()
    release = threading.Event()
    errors = []

    def slow_build():
        started.set()
        release.wait(timeout=5)
        return "LATE"

    def leader():
        try:
            with cache.get_or_build(_desc("z"), slow_build, size_bytes=100):
                errors.append("published")
        except Exception as exc:
            errors.append(type(exc).__name__)

    t = threading.Thread(target=leader)
    t.start(); started.wait(timeout=5)
    cache.close()
    release.set(); t.join(timeout=10)
    assert errors == ["PreparationError"]
    assert mgr.pool_bytes("artifact_cache") == 0


def test_reservation_precedes_build():
    """Codex plan-gate finding 2: the byte lease is held BEFORE builder()."""
    mgr = ResourceLeaseManager(ResourceBudget(artifact_cache_bytes=1000))
    cache = PreparedArtifactCache(mgr)
    seen = []

    def builder():
        seen.append(mgr.pool_bytes("artifact_cache"))
        return "V"

    with cache.get_or_build(_desc("r"), builder, size_bytes=100):
        pass
    assert seen == [100]  # lease already charged while building


def test_lease_manager_rejects_after_close():
    mgr = ResourceLeaseManager(ResourceBudget(artifact_cache_bytes=1000))
    mgr.close()
    with pytest.raises(Exception):
        mgr.lease_bytes(1, "artifact_cache")
    with pytest.raises(Exception):
        with mgr.task_slot():
            pass


def test_close_is_idempotent_and_releases_bytes():
    mgr = ResourceLeaseManager(ResourceBudget(artifact_cache_bytes=1000))
    cache = PreparedArtifactCache(mgr)
    with cache.get_or_build(_desc("a"), lambda: "A", size_bytes=100):
        pass
    cache.close(); cache.close()
    assert mgr.pool_bytes("artifact_cache") == 0
    with pytest.raises(PreparationError):
        cache.get_or_build(_desc("a"), lambda: "A", size_bytes=10)
```

- [ ] **Step 2: Run to verify FAIL.**
- [ ] **Step 3: Implementation**

```python
# quantark/execution/cache/artifacts.py
"""Session-owned prepared-artifact cache (spec section 10.1).

Keys are canonical value fingerprints inside a full ``ArtifactDescriptor``;
a hit verifies complete descriptor equality (hash-collision defense). Miss
construction is single-flight with the leader-election contract from the
hardened spec: leader failure publishes to current waiters, cleans up
atomically, and the next requester re-elects. Eviction is LRU by bytes over
unpinned entries; pinned bytes stay leased until the last handle closes.
An artifact whose size exceeds remaining admittable capacity is built and
returned WITHOUT cache admission (correctness never depends on caching).
"""
import threading
from collections import OrderedDict
from dataclasses import dataclass

from quantark.execution.errors import PreparationError, ResourceBudgetExceeded

__all__ = ["ArtifactDescriptor", "ArtifactHandle", "PreparedArtifactCache"]


@dataclass(frozen=True)
class ArtifactDescriptor:
    kind: str
    fingerprint: str
    dependency_tags: frozenset
    builder_version: str


class ArtifactHandle:
    def __init__(self, value, release):
        self.value = value
        self._release = release

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def close(self):
        release, self._release = self._release, None
        if release is not None:
            release()


class _Entry:
    __slots__ = ("descriptor", "value", "size", "pins")

    def __init__(self, descriptor, value, size):
        self.descriptor = descriptor
        self.value = value
        self.size = size
        self.pins = 0


class _InFlight:
    __slots__ = ("event", "error")

    def __init__(self):
        self.event = threading.Event()
        self.error = None


class PreparedArtifactCache:
    _POOL = "artifact_cache"

    def __init__(self, lease_manager):
        self._leases = lease_manager
        self.lease_manager = lease_manager  # public: pairing identity check
        self._lock = threading.Lock()
        self._entries: "OrderedDict[ArtifactDescriptor, _Entry]" = OrderedDict()
        self._in_flight: dict = {}
        self._closed = False
        self._stats = {
            "hits": 0, "misses": 0, "evictions": 0, "single_flight_waits": 0,
        }

    # -- public ----------------------------------------------------------
    def get_or_build(self, descriptor, builder, size_bytes) -> ArtifactHandle:
        while True:
            with self._lock:
                if self._closed:
                    raise PreparationError("PreparedArtifactCache is closed")
                entry = self._entries.get(descriptor)
                if entry is not None:
                    # full descriptor equality is the dict key itself
                    self._entries.move_to_end(descriptor)
                    entry.pins += 1
                    self._stats["hits"] += 1
                    return ArtifactHandle(
                        entry.value, lambda e=entry: self._unpin(e)
                    )
                flight = self._in_flight.get(descriptor)
                if flight is None:
                    self._in_flight[descriptor] = flight = _InFlight()
                    leader = True
                else:
                    leader = False
                    self._stats["single_flight_waits"] += 1
            if leader:
                return self._build_as_leader(descriptor, builder, size_bytes)
            flight.event.wait()
            if flight.error is not None:
                raise flight.error
            # success published (or cleanup after failure raced): loop re-reads

    def stats(self) -> dict:
        with self._lock:
            out = dict(self._stats)
            out["bytes_in_use"] = sum(e.size for e in self._entries.values())
            return out

    def invalidate_tags(self, tags) -> None:
        tags = frozenset(tags)
        with self._lock:
            stale = [
                d for d, e in self._entries.items()
                if e.pins == 0 and d.dependency_tags & tags
            ]
            for d in stale:
                self._drop(d)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for flight in self._in_flight.values():
                flight.error = PreparationError("cache closed during build")
                flight.event.set()
            self._in_flight.clear()
            for d in list(self._entries):
                self._drop(d, count_eviction=False)

    # -- internals ---------------------------------------------------------
    def _build_as_leader(self, descriptor, builder, size_bytes):
        # RESERVE the estimated bytes BEFORE building (Codex plan-gate
        # finding 2): the expensive allocation happens under a held lease,
        # so concurrent distinct-key builds cannot collectively exceed the
        # budget. If no lease is obtainable even after evicting unpinned
        # entries, the build proceeds WITHOUT cache admission (bypass) —
        # correctness never depends on caching, but the bypass is a
        # deliberate, diagnosed state, not an accounting hole.
        with self._lock:
            reserved = self._reserve(size_bytes)
        try:
            value = builder()
        except BaseException as exc:
            with self._lock:
                if reserved:
                    self._leases.release_bytes(size_bytes, self._POOL)
                flight = self._in_flight.pop(descriptor, None)
            if flight is not None:
                flight.error = exc
                flight.event.set()
            raise
        with self._lock:
            if self._closed:
                # close() raced the build (Codex plan-gate finding 3):
                # never publish or retain bytes past session lifetime.
                if reserved:
                    self._leases.release_bytes(size_bytes, self._POOL)
                flight = self._in_flight.pop(descriptor, None)
                error = PreparationError("cache closed during build")
                if flight is not None:
                    flight.error = error
                    flight.event.set()
                raise error
            self._stats["misses"] += 1
            flight = self._in_flight.pop(descriptor, None)
            if reserved:
                entry = _Entry(descriptor, value, size_bytes)
                entry.pins = 1
                self._entries[descriptor] = entry
            else:
                entry = None
        if flight is not None:
            flight.event.set()
        if entry is not None:
            return ArtifactHandle(value, lambda e=entry: self._unpin(e))
        return ArtifactHandle(value, lambda: None)  # bypass: not cached

    def _reserve(self, size_bytes) -> bool:
        """Caller holds the lock. Lease bytes, evicting unpinned LRU entries
        as needed; False means bypass (build uncached)."""
        while True:
            try:
                self._leases.lease_bytes(size_bytes, self._POOL)
                return True
            except ResourceBudgetExceeded:
                victim = next(
                    (d for d, e in self._entries.items() if e.pins == 0), None
                )
                if victim is None:
                    return False
                self._drop(victim)

    def _drop(self, descriptor, count_eviction=True):
        entry = self._entries.pop(descriptor)
        self._leases.release_bytes(entry.size, self._POOL)
        if count_eviction:
            self._stats["evictions"] += 1

    def _unpin(self, entry):
        with self._lock:
            entry.pins = max(0, entry.pins - 1)
```

Note the deliberate simplification versus the full spec: pinned entries stay
in the lookup index (they are shareable); eviction simply skips them. The
"removed from index while pinned" state only arises via `invalidate_tags`,
which skips pinned entries in Phase 1 — document this as a Phase 4 tightening
when PDE artifacts land.

- [ ] **Step 4: Run to PASS (6 tests).**
- [ ] **Step 5: Commit** — `feat(execution): PreparedArtifactCache with single-flight and pinned LRU`

---

### Task 4: Kernel lifecycle, PreparedState, session-owned services

**Files:**
- Modify: `quantark/execution/contracts.py` (append `PreparedState`)
- Modify: `quantark/execution/manifest.py` (append defaulted `preparation_fingerprint` field)
- Modify: `quantark/execution/context.py` (append `artifact_cache`, `lease_manager` handles)
- Modify: `quantark/execution/kernel.py` (reserve → prepare → execute → release)
- Modify: `quantark/execution/api.py` (session owns cache + lease manager)
- Test: `test/execution/test_kernel_prepare.py` (first half)

**Interfaces:**
- `PreparedState(payload: object, descriptors: tuple, fingerprint: str | None, byte_estimate: int | None, handles: tuple = ())` — frozen; `handles` are ArtifactHandles the kernel closes in `finally`.
- `ReproducibilityManifest` gains `preparation_fingerprint: str | None = None` (last field, defaulted — additive).
- `PricingRunContext` gains `artifact_cache: object | None = None`, `lease_manager: object | None = None` (defaulted — additive).
- Kernel dispatch order: resolve → validate → normalize → **task_slot lease**
  → **adapter.prepare(engine, normalized_or_request, context) if the adapter
  defines it** → execute (adapter `execute_native` gains optional
  `prepared=None` keyword) → manifest/diagnostics → **close prepared handles
  and release slot in `finally`**.
- `PricingSession.__init__` creates (and owns) a `ResourceLeaseManager` and
  `PreparedArtifactCache` when the context has none; default cache capacity =
  `budget.artifact_cache_bytes` or 512 MiB when None (recorded in
  diagnostics-facing `config_snapshot` as `("artifact_cache_default", "536870912")`).
  `close()` closes owned cache then lease manager, idempotently.
- `LegacyPriceAdapter.execute_native` signature grows `prepared=None`
  keyword (ignored) so the kernel can pass uniformly.

- [ ] **Step 1: Write the failing test**

```python
# test/execution/test_kernel_prepare.py
"""Kernel prepare lifecycle and session-owned services (spec section 7)."""
import dataclasses

import pytest

from quantark.execution import PricingRequest, PricingSession
from quantark.execution.contracts import PreparedState


class _PrepEngine:
    """price(product, env) engine whose adapter prepares state."""

    def price(self, product, env):
        return 1.0


class _PrepAdapter:
    """Minimal specialized adapter with a prepare step."""

    def __init__(self):
        from quantark.execution.legacy_adapter import LegacyPriceAdapter

        self._legacy = LegacyPriceAdapter(call_shape="product_env")
        self.prepared_seen = []

    def capabilities(self):
        return self._legacy.capabilities()

    def validate(self, engine, request):
        return self._legacy.validate(engine, request)

    def normalize(self, engine, request):
        return self._legacy.normalize(engine, request)

    def prepare(self, engine, request, context):
        return PreparedState(
            payload={"k": 1}, descriptors=(), fingerprint="prep-fp",
            byte_estimate=8,
        )

    def execute_native(self, engine, request, normalized, context, prepared=None):
        self.prepared_seen.append(prepared)
        return 1.0, (("pv", 1.0),)


def _session_with(adapter):
    from quantark.execution.context import default_context
    from quantark.execution.registry import AdapterRegistry

    registry = AdapterRegistry()
    registry.register(
        f"{_PrepEngine.__module__}.{_PrepEngine.__qualname__}", lambda: adapter
    )
    ctx = dataclasses.replace(default_context(), adapter_registry=registry)
    return PricingSession(ctx)


def test_kernel_calls_prepare_and_stamps_manifest():
    adapter = _PrepAdapter()
    with _session_with(adapter) as session:
        outcome = session.execute(
            _PrepEngine(), PricingRequest(product="P", pricing_env="E")
        )
    assert adapter.prepared_seen and adapter.prepared_seen[0].fingerprint == "prep-fp"
    assert outcome.manifest.preparation_fingerprint == "prep-fp"


def test_session_owns_cache_and_lease_manager():
    with PricingSession() as session:
        ctx = session.context
        assert ctx.artifact_cache is not None
        assert ctx.lease_manager is not None
        cache = ctx.artifact_cache
    # closed with the session: further use raises
    from quantark.execution.errors import PreparationError

    with pytest.raises(PreparationError):
        cache.get_or_build(None, lambda: 1, size_bytes=1)


def test_legacy_engines_still_work_without_prepare(monkeypatch):
    from datetime import datetime

    from quantark.asset.equity.engine.mc import EuropeanMCEngine
    from quantark.asset.equity.param import MCParams
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.priceenv import PricingEnvironment
    from quantark.util.enum import OptionType

    env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        valuation_date=datetime(2024, 1, 1),
    )
    opt = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    engine = EuropeanMCEngine(params=MCParams(num_paths=64, seed=1))
    direct = engine.price(opt, env)
    with PricingSession() as session:
        outcome = session.execute(engine, PricingRequest(product=opt, pricing_env=env))
    assert outcome.value == direct
    assert outcome.manifest.preparation_fingerprint is None
```

- [ ] **Step 2: Run to verify FAIL** (PreparedState import error).
- [ ] **Step 3: Implement.** Additions:

`contracts.py` (append; add `"PreparedState"` to `__all__`):

```python
@dataclass(frozen=True)
class PreparedState:
    payload: object
    descriptors: tuple
    fingerprint: str | None
    byte_estimate: int | None
    handles: tuple = ()
```

`manifest.py`: add `preparation_fingerprint: str | None = None` as the LAST
field of `ReproducibilityManifest`.

`context.py`: add fields after `cancellation_token`:
`artifact_cache: object | None = None` and `lease_manager: object | None = None`.

`kernel.py` — replace the body of `dispatch` from normalization onward:

```python
        normalized = adapter.normalize(engine, request)
        lease_manager = context.lease_manager
        slot = lease_manager.task_slot() if lease_manager is not None else None
        prepared = None
        start = time.perf_counter()
        prep_seconds = 0.0
        try:
            if slot is not None:
                slot.__enter__()
            if hasattr(adapter, "prepare"):
                t_prep = time.perf_counter()
                prepared = adapter.prepare(engine, request, context)
                prep_seconds = time.perf_counter() - t_prep
            value, economics = adapter.execute_native(
                engine, request, normalized, context, prepared=prepared
            )
        finally:
            if prepared is not None:
                for handle in prepared.handles:
                    handle.close()
            if slot is not None:
                slot.__exit__(None, None, None)
        elapsed = time.perf_counter() - start
```

and stamp `preparation_fingerprint=prepared.fingerprint if prepared else None`
into the manifest; add `("prepare_seconds", prep_seconds)` to diagnostics
timings; append cache stats to diagnostics `records` as
`f"cache:{k}={v}"` lines when `context.artifact_cache` is not None.

`legacy_adapter.py`: change `execute_native(self, engine, request, normalized, context)`
to `execute_native(self, engine, request, normalized, context, prepared=None)`.

`api.py` `__init__` — after registry handling:

```python
        # Codex plan-gate finding 4: cache and lease manager form ONE budget
        # domain. They are supplied as a validated pair or created as a pair;
        # partial injection is rejected, and a supplied pair must be linked.
        supplied_cache = context.artifact_cache
        supplied_leases = context.lease_manager
        self._owned_cache = None
        self._owned_leases = None
        if (supplied_cache is None) != (supplied_leases is None):
            from quantark.util.exceptions import ValidationError

            raise ValidationError(
                "artifact_cache and lease_manager must be supplied together "
                "(one shared budget domain) or both omitted"
            )
        if supplied_cache is not None:
            if supplied_cache.lease_manager is not supplied_leases:
                from quantark.util.exceptions import ValidationError

                raise ValidationError(
                    "supplied artifact_cache is not backed by the supplied "
                    "lease_manager; budget domains would split"
                )
        else:
            import dataclasses

            from quantark.execution.cache.artifacts import PreparedArtifactCache
            from quantark.execution.leases import ResourceLeaseManager

            budget = context.resource_budget
            if budget.artifact_cache_bytes is None:
                budget = dataclasses.replace(
                    budget, artifact_cache_bytes=512 * 2**20
                )
            leases = ResourceLeaseManager(budget)
            cache = PreparedArtifactCache(leases)
            self._owned_leases = leases
            self._owned_cache = cache
            context = dataclasses.replace(
                context, resource_budget=budget,
                lease_manager=leases, artifact_cache=cache,
            )
```

and in `close()`: close `self._owned_cache` then `self._owned_leases` (both
guarded, idempotent) before setting `_closed`.

- [ ] **Step 4: Run to PASS** — `test_kernel_prepare.py` (3) plus the whole
  existing `test/execution/` suite (no regressions; the Phase 0 parity tests
  now run through the lease-slot path).
- [ ] **Step 5: Commit** — `feat(execution): kernel prepare lifecycle and session-owned cache/leases`

---

### Task 5: DCN local-vol preparation adapter

**Files:**
- Create: `quantark/asset/equity/engine/mc/dcn_execution_adapters.py`
- Modify: `quantark/execution/registry.py` (register by string path, lazy import)
- Modify: `quantark/execution/inventory.py` (DCN LV rows → `adoption_state="supported"`)
- Test: `test/execution/test_kernel_prepare.py` (second half)

**Interfaces:**
- Produces: `DCNLocalVolMCAdapter` / `DCNLocalVolPDEAdapter` (both subclass
  `LegacyPriceAdapter`, `call_shape="product_env"`), registered for exact
  classes `LocalVolDCNMCEngine` / `LocalVolDCNPDEEngine`.
- `prepare`: if `engine._prebuilt is not None` → return
  `PreparedState(payload=engine._prebuilt, descriptors=(), fingerprint=None, byte_estimate=None)`
  (prebuilt bypass preserved). Otherwise fingerprint the env components the
  Dupire build consumes: `(vol_surface, spot_quote, rate_curve, div_yield)`
  via `try_fingerprint`. Fingerprint unavailable or no cache on context →
  build fresh, uncached. Else
  `cache.get_or_build(ArtifactDescriptor(kind="dupire-local-vol", fingerprint=fp, dependency_tags=frozenset({"vol_surface","spot","rate_curve","dividend_curve"}), builder_version="1"), builder, size_bytes=_surface_nbytes(...))`
  — handle goes into `PreparedState.handles` so the kernel releases the pin.
- `execute_native(..., prepared=None)`: clone the engine with the prepared
  surface and delegate to the legacy dispatch on the clone:
  - MC clone: `type(engine)(local_vol_surface=surface, num_paths=engine.num_paths, seed=engine.seed, use_sobol=engine.use_sobol, use_antithetic=engine.use_antithetic, num_batches=engine.num_batches, num_workers=engine.num_workers)`
  - PDE clone: `type(engine)(local_vol_surface=surface, num_space_nodes=engine.n, s_min_mult=engine.s_min_mult, s_max_mult=engine.s_max_mult, rannacher_steps=engine.rannacher_steps, concentration=engine.concentration)`
  - `prepared is None` (defensive) → fall back to `super().execute_native`.
- Builder replicates the engine's own build exactly:
  `build_dupire_local_vol(env.vol_surface, spot=env.spot, rate_curve=env.rate_curve, div_yield=env.get_div_yield)`
  (the same call as `LocalVolDCNMCEngine._build_surface`); a non-`GridVolSurface`
  env raises the same `PricingError` — build it OUTSIDE the cache builder's
  single-flight? No: raise happens inside `builder`, which the single-flight
  contract already publishes correctly. Keep it inside.

- [ ] **Step 1: Write the failing test** (append to `test_kernel_prepare.py`)

```python
class TestDCNLocalVolAdapter:
    @pytest.fixture()
    def dcn_case(self):
        import numpy as np

        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        from dcn_fixtures import DCN_A, FLAT, flat_env, make_dcn
        from quantark.param import GridVolSurface

        def grid_env():
            env = flat_env(**FLAT)
            env.vol_surface = GridVolSurface(
                strikes=[3000.0, 4500.0, 6000.0, 7500.0, 9000.0],
                maturities=[0.25, 0.5, 1.0, 1.5, 2.0, 2.5],
                iv_grid=np.full((6, 5), FLAT["sigma"]),
            )
            return env

        return make_dcn(DCN_A), grid_env

    def test_session_parity_and_single_build(self, dcn_case, monkeypatch):
        import quantark.asset.equity.engine.mc.dcn_execution_adapters as mod
        from quantark.asset.equity.engine.mc import LocalVolDCNMCEngine

        product, grid_env = dcn_case
        engine = LocalVolDCNMCEngine(num_paths=2**9, seed=42)
        direct = engine.price(product, grid_env())

        builds = []
        real_build = mod.build_dupire_local_vol

        def counting_build(*args, **kwargs):
            builds.append(1)
            return real_build(*args, **kwargs)

        monkeypatch.setattr(mod, "build_dupire_local_vol", counting_build)
        with PricingSession() as session:
            v1 = session.price(LocalVolDCNMCEngine(num_paths=2**9, seed=42),
                               product, grid_env())
            v2 = session.price(LocalVolDCNMCEngine(num_paths=2**9, seed=42),
                               product, grid_env())  # equal-VALUED new env
        assert v1 == direct and v2 == direct   # bit-identical
        assert len(builds) == 1                # second call hit the cache

    def test_changed_vol_rebuilds(self, dcn_case, monkeypatch):
        import numpy as np

        import quantark.asset.equity.engine.mc.dcn_execution_adapters as mod
        from quantark.asset.equity.engine.mc import LocalVolDCNMCEngine
        from quantark.param import GridVolSurface

        product, grid_env = dcn_case
        builds = []
        real_build = mod.build_dupire_local_vol
        monkeypatch.setattr(
            mod, "build_dupire_local_vol",
            lambda *a, **k: builds.append(1) or real_build(*a, **k),
        )
        env2 = grid_env()
        env2.vol_surface = GridVolSurface(
            strikes=env2.vol_surface.strikes,
            maturities=env2.vol_surface.maturities,
            iv_grid=np.full((6, 5), 0.30),
        )
        with PricingSession() as session:
            session.price(LocalVolDCNMCEngine(num_paths=2**8), product, grid_env())
            session.price(LocalVolDCNMCEngine(num_paths=2**8), product, env2)
        assert len(builds) == 2

    def test_disabled_cache_still_exact(self, dcn_case):
        import dataclasses

        from quantark.asset.equity.engine.mc import LocalVolDCNMCEngine
        from quantark.execution import ResourceBudget
        from quantark.execution.context import default_context

        product, grid_env = dcn_case
        engine = LocalVolDCNMCEngine(num_paths=2**8, seed=3)
        direct = engine.price(product, grid_env())
        ctx = dataclasses.replace(
            default_context(),
            resource_budget=ResourceBudget(artifact_cache_bytes=0),
        )
        with PricingSession(ctx) as session:
            assert session.price(engine, product, grid_env()) == direct

    def test_pde_adapter_parity(self, dcn_case):
        from quantark.asset.equity.engine.pde import LocalVolDCNPDEEngine

        product, grid_env = dcn_case
        engine = LocalVolDCNPDEEngine(num_space_nodes=301)
        direct = engine.price(product, grid_env())
        with PricingSession() as session:
            assert session.price(
                LocalVolDCNPDEEngine(num_space_nodes=301), product, grid_env()
            ) == direct

    def test_mutation_during_prepare_raises_determinism_violation(
        self, dcn_case, monkeypatch
    ):
        """Codex plan-gate finding 1: env mutated mid-build -> loud failure,
        never a cached surface that mismatches its key."""
        import numpy as np

        import quantark.asset.equity.engine.mc.dcn_execution_adapters as mod
        from quantark.asset.equity.engine.mc import LocalVolDCNMCEngine
        from quantark.execution.errors import DeterminismViolation

        product, grid_env = dcn_case
        env = grid_env()
        real_build = mod.build_dupire_local_vol

        def mutating_build(*args, **kwargs):
            surface = real_build(*args, **kwargs)
            env.vol_surface.iv_grid = env.vol_surface.iv_grid + 0.01
            return surface

        monkeypatch.setattr(mod, "build_dupire_local_vol", mutating_build)
        with PricingSession() as session:
            with pytest.raises(DeterminismViolation):
                session.price(
                    LocalVolDCNMCEngine(num_paths=2**8), product, env
                )

    def test_partial_service_injection_rejected(self):
        """Codex plan-gate finding 4: cache/lease-manager come as a pair."""
        import dataclasses

        from quantark.execution.cache.artifacts import PreparedArtifactCache
        from quantark.execution.context import default_context
        from quantark.execution.leases import ResourceLeaseManager
        from quantark.execution.policy import ResourceBudget
        from quantark.util.exceptions import ValidationError

        leases = ResourceLeaseManager(ResourceBudget(artifact_cache_bytes=100))
        cache = PreparedArtifactCache(leases)
        base = default_context()
        with pytest.raises(ValidationError):
            PricingSession(dataclasses.replace(base, artifact_cache=cache))
        with pytest.raises(ValidationError):
            PricingSession(dataclasses.replace(base, lease_manager=leases))
        other = ResourceLeaseManager(ResourceBudget(artifact_cache_bytes=100))
        with pytest.raises(ValidationError):
            PricingSession(dataclasses.replace(
                base, artifact_cache=cache, lease_manager=other,
            ))
        with PricingSession(dataclasses.replace(
            base, artifact_cache=cache, lease_manager=leases,
        )) as session:  # matched pair accepted; borrowed, not closed
            assert session.context.artifact_cache is cache

    def test_prebuilt_surface_bypasses_cache(self, dcn_case):
        from quantark.asset.equity.engine.mc import LocalVolDCNMCEngine
        from quantark.volmodels.localvol import build_dupire_local_vol

        product, grid_env = dcn_case
        env = grid_env()
        surface = build_dupire_local_vol(
            env.vol_surface, spot=env.spot,
            rate_curve=env.rate_curve, div_yield=env.get_div_yield,
        )
        engine = LocalVolDCNMCEngine(
            local_vol_surface=surface, num_paths=2**8, seed=5
        )
        direct = engine.price(product, env)
        with PricingSession() as session:
            outcome = session.execute(
                engine, PricingRequest(product=product, pricing_env=env)
            )
        assert outcome.value == direct
        assert outcome.manifest.preparation_fingerprint is None
```

- [ ] **Step 2: Run to verify FAIL** (module not found).
- [ ] **Step 3: Implementation**

```python
# quantark/asset/equity/engine/mc/dcn_execution_adapters.py
"""Execution-framework adapters for the DCN local-vol engines (spec Phase 1).

The adapter path serves cache-fetched Dupire surfaces through PREBUILT-SURFACE
FACTORY CLONES: a clone of the target engine is constructed with
``local_vol_surface=<cached surface>``, so the original engine's
``_prepare_simulation``/``_resolve_surface`` hooks and ``_active_surface``
state are never touched (spec sections 6.3 + 17.1: mutable-state removal
applies to the adapter path only; the direct path and its subclass hooks are
preserved verbatim).
"""
from quantark.execution.cache.artifacts import ArtifactDescriptor
from quantark.execution.cache.fingerprint import try_fingerprint
from quantark.execution.contracts import PreparedState
from quantark.execution.legacy_adapter import LegacyPriceAdapter
from quantark.volmodels.localvol import build_dupire_local_vol

__all__ = ["DCNLocalVolMCAdapter", "DCNLocalVolPDEAdapter"]

_DUPIRE_TAGS = frozenset({"vol_surface", "spot", "rate_curve", "dividend_curve"})
_BUILDER_VERSION = "1"


def _surface_nbytes(surface) -> int:
    total = 0
    for value in vars(surface).values():
        nbytes = getattr(value, "nbytes", None)
        if isinstance(nbytes, int):
            total += nbytes
    return total or (1 << 20)  # conservative floor: 1 MiB


class _DCNLocalVolAdapterBase(LegacyPriceAdapter):
    def __init__(self):
        super().__init__(call_shape="product_env")

    def prepare(self, engine, request, context) -> PreparedState:
        if engine._prebuilt is not None:
            return PreparedState(
                payload=engine._prebuilt, descriptors=(),
                fingerprint=None, byte_estimate=None,
            )
        env = request.pricing_env
        # Codex plan-gate finding 1: BIND the preparation inputs once, so the
        # cache key, the build, and the verification all see the same
        # objects; verify the fingerprint after the build and raise
        # DeterminismViolation on concurrent mutation (spec section 5.1).
        inputs = (env.vol_surface, env.spot_quote, env.rate_curve, env.div_yield)
        spot, div_fn = env.spot, env.get_div_yield
        fp = try_fingerprint(inputs)
        cache = context.artifact_cache

        def builder():
            return build_dupire_local_vol(
                inputs[0], spot=spot, rate_curve=inputs[2], div_yield=div_fn,
            )

        if fp is None or cache is None:
            surface = builder()  # uncacheable: fresh build, still correct
            return PreparedState(
                payload=surface, descriptors=(),
                fingerprint=None, byte_estimate=_surface_nbytes(surface),
            )
        descriptor = ArtifactDescriptor(
            kind="dupire-local-vol", fingerprint=fp,
            dependency_tags=_DUPIRE_TAGS, builder_version=_BUILDER_VERSION,
        )
        handle = cache.get_or_build(
            descriptor, builder, size_bytes=_ESTIMATED_SURFACE_BYTES,
        )
        if try_fingerprint(inputs) != fp:
            handle.close()
            cache.invalidate_tags(_DUPIRE_TAGS)
            from quantark.execution.errors import DeterminismViolation

            raise DeterminismViolation(
                "pricing environment mutated during preparation; the cached "
                "Dupire surface no longer matches its key"
            )
        return PreparedState(
            payload=handle.value, descriptors=(descriptor,),
            fingerprint=fp, byte_estimate=None, handles=(handle,),
        )

    def execute_native(self, engine, request, normalized, context, prepared=None):
        if prepared is None:
            return super().execute_native(
                engine, request, normalized, context
            )
        clone = self._clone_with_surface(engine, prepared.payload)
        return super().execute_native(clone, request, normalized, context)


# Size is only known after the first build; admission uses a conservative
# estimate (Phase 1: estimates are admission accounting, spec section 11).
# 8 MiB covers realistic Dupire grids.
_ESTIMATED_SURFACE_BYTES = 8 << 20


class DCNLocalVolMCAdapter(_DCNLocalVolAdapterBase):
    def _clone_with_surface(self, engine, surface):
        return type(engine)(
            local_vol_surface=surface,
            num_paths=engine.num_paths, seed=engine.seed,
            use_sobol=engine.use_sobol, use_antithetic=engine.use_antithetic,
            num_batches=engine.num_batches, num_workers=engine.num_workers,
        )


class DCNLocalVolPDEAdapter(_DCNLocalVolAdapterBase):
    def _clone_with_surface(self, engine, surface):
        return type(engine)(
            local_vol_surface=surface,
            num_space_nodes=engine.n,
            s_min_mult=engine.s_min_mult, s_max_mult=engine.s_max_mult,
            rannacher_steps=engine.rannacher_steps,
            concentration=engine.concentration,
        )
```

`registry.py` — extend `_DEFAULT_REGISTRATIONS` mechanism: exact-class
specialized registrations use lazy factories importing the adapter module at
resolve time (registry itself still never imports asset code statically):

```python
def _dcn_mc_adapter():
    from quantark.asset.equity.engine.mc.dcn_execution_adapters import (
        DCNLocalVolMCAdapter,
    )

    return DCNLocalVolMCAdapter()


def _dcn_pde_adapter():
    from quantark.asset.equity.engine.mc.dcn_execution_adapters import (
        DCNLocalVolPDEAdapter,
    )

    return DCNLocalVolPDEAdapter()
```

and in `build_default_registry()` after the loop:

```python
    registry.register(
        "quantark.asset.equity.engine.mc.dcn_vol_mc_engines.LocalVolDCNMCEngine",
        _dcn_mc_adapter,
    )
    registry.register(
        "quantark.asset.equity.engine.pde.dcn_vol_pde_solvers.LocalVolDCNPDEEngine",
        _dcn_pde_adapter,
    )
```

Update `test_default_registry_covers_engine_family_roots` expected set with
these two paths.

`inventory.py`: change the `LocalVolDCNMCEngine` and `LocalVolDCNPDEEngine`
rows to `adoption_state="supported"` (keep owner/milestone as-is — allowed for
supported rows; the gate only REQUIRES them for temporary_legacy).
`_eq_mc`/`_eq_pde` don't expose adoption_state — give them an optional
`adoption_state="temporary_legacy"` parameter and pass `"supported"` for
those two rows.

- [ ] **Step 4: Run to PASS** — `test_kernel_prepare.py` complete +
  `test_registry.py` + `test_inventory.py`.
- [ ] **Step 5: Commit** — `feat(execution): DCN local-vol preparation adapter with cached Dupire surfaces`

---

### Task 6: Full-matrix fixtures

**Files:**
- Create: `test/execution/matrix_fixtures.py`
- Test: `test/execution/test_matrix_parity.py`

**Interfaces:**
- Produces: `FIXTURE_BUILDERS: dict[str, Callable[[], tuple]]` mapping EVERY
  concrete `ENGINE_INVENTORY` name → `(engine, product, env_or_None, call_shape)`.
  Builders are lazy (import inside), share family helpers, use the cheapest
  parameters cited from the source tests. `test/dcn_fixtures.py` is imported
  via a `sys.path` shim (it is a test-local module).

The module content is the mined fixture report converted to code. Structure
(complete code — family helpers then builders; parameters and constructor
shapes are verbatim from the cited test files):

```python
# test/execution/matrix_fixtures.py
"""Executable fixtures for every concrete inventoried engine (Phase 1 gate).

Recipes are lifted from the authoritative test files (cited per family in
docs/superpowers/plans/2026-07-15-execution-framework-phase1.md). Parameters
are the cheapest observed that still price successfully.
"""
import pathlib
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # test/


# ---------------------------------------------------------------- equity core
def _eq_flat_env():
    from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
    from quantark.param.div import ContinuousDividendYield
    from quantark.priceenv import PricingEnvironment

    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.02),
        valuation_date=datetime(2024, 1, 1),
    )


def _eq_grid_env():
    from quantark.param import FlatRateCurve, GridVolSurface, SpotQuote
    from quantark.param.div import ContinuousDividendYield
    from quantark.priceenv import PricingEnvironment

    s0 = 100.0
    strikes = list(s0 * np.exp(np.linspace(-0.5, 0.5, 9)))
    maturities = list(np.linspace(0.25, 1.0, 4))
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.03), valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=s0),
        vol_surface=GridVolSurface(
            strikes, maturities,
            np.full((len(maturities), len(strikes)), 0.20),
        ),
        div_yield=ContinuousDividendYield(0.01),
    )


def _euro():
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.util.enum import OptionType

    return EuropeanVanillaOption(
        strike=100.0, option_type=OptionType.CALL, maturity=1.0
    )


def _hp():
    from quantark.volmodels.heston import HestonParams

    return HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)


def _unit_leverage(s0=100.0):
    from quantark.volmodels.slv.leverage import LeverageSurface

    ks = np.array(list(s0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    return LeverageSurface(
        time_grid=np.linspace(0.0, 1.0, 4), strike_grid=ks,
        leverage_grid=np.ones((4, ks.size)),
    )


def _mcp(**kw):
    from quantark.asset.equity.param import MCParams

    return MCParams(**kw)


def _pdep(**kw):
    from quantark.asset.equity.param import PDEParams

    return PDEParams(**kw)


# ------------------------------------------------------------ family products
def _snowball():
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig
    from quantark.asset.equity.product.option.snowball_option import SnowballOption
    from quantark.util.enum import ObservationType

    return SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0,
        contract_multiplier=10_000.0, is_reverse=False,
        barrier_config=BarrierConfig(
            ko_barrier=105.0, ko_rate=0.12,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0,
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        ),
    )


def _phoenix():
    from quantark.asset.equity.product.option.phoenix_config import (
        CouponBarrierConfig,
    )
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
    from quantark.asset.equity.product.option.snowball_config import (
        BarrierConfig,
        PayoffConfig,
    )
    from quantark.util.calendar.day_counter import DayCountConvention
    from quantark.util.enum import CouponPayType, ObservationType

    return PhoenixOption(
        initial_price=100.0, strike=100.0, maturity=1.0,
        contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(
            ko_barrier=105.0, ko_rate=0.12,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0,
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        ),
        coupon_config=CouponBarrierConfig(
            coupon_barrier=90.0, coupon_rate=0.02,
            coupon_pay_type=CouponPayType.INSTANT,
            day_count_convention=DayCountConvention.ACT_365,
            memory_coupon=False,
        ),
        payoff_config=PayoffConfig(include_principal=True),
    )


def _barrier(maturity=1.0):
    from quantark.asset.equity.product.option.barrier_option import BarrierOption
    from quantark.util.enum import BarrierType, ObservationType, OptionType

    return BarrierOption(
        strike=100.0, option_type=OptionType.CALL, barrier=130.0,
        barrier_type=BarrierType.UP_OUT, maturity=maturity,
        observation_type=ObservationType.CONTINUOUS,
    )


def _dcn():
    from dcn_fixtures import DCN_A, make_dcn

    return make_dcn(DCN_A)


def _dcn_flat_env():
    from dcn_fixtures import FLAT, flat_env

    return flat_env(**FLAT)


def _dcn_grid_env():
    from dcn_fixtures import FLAT, flat_env
    from quantark.param import GridVolSurface

    env = flat_env(**FLAT)
    env.vol_surface = GridVolSurface(
        strikes=[3000.0, 4500.0, 6000.0, 7500.0, 9000.0],
        maturities=[0.25, 0.5, 1.0, 1.5, 2.0, 2.5],
        iv_grid=np.full((6, 5), FLAT["sigma"]),
    )
    return env


def _dcn_hp():
    from quantark.volmodels.heston import HestonParams

    return HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)


# ------------------------------------------------------------------ FX shared
def _fx_env(surface):
    from quantark.param import FlatRateCurve, SpotQuote
    from quantark.priceenv import FxPricingEnvironment

    return FxPricingEnvironment(
        valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=1.20),
        domestic_curve=FlatRateCurve(rate=0.05),
        foreign_curve=FlatRateCurve(rate=0.03),
        vol_surface=surface,
    )


def _fx_flat_env(vol=0.10):
    from quantark.param import FlatVolSurface

    return _fx_env(FlatVolSurface(volatility=vol))


def _fx_grid_env():
    from quantark.param import GridVolSurface

    strikes = list(1.20 * np.exp(np.linspace(-0.3, 0.3, 7)))
    maturities = [0.25, 0.5, 1.0, 1.5]
    return _fx_env(
        GridVolSurface(strikes, maturities, np.full((4, 7), 0.10))
    )


def _fx_vanilla():
    from quantark.asset.fx.product.option.fx_vanilla_option import FxVanillaOption
    from quantark.util.enum import OptionType

    return FxVanillaOption(
        strike=1.20, option_type=OptionType.CALL, maturity=1.0,
        notional_foreign=1_000_000.0,
    )


def _fx_unit_leverage():
    from quantark.volmodels.slv.leverage import LeverageSurface

    ks = np.array(list(1.20 * np.exp(np.linspace(-0.4, 0.4, 9))))
    return LeverageSurface(
        time_grid=np.linspace(0.0, 1.0, 4), strike_grid=ks,
        leverage_grid=np.ones((4, ks.size)),
    )


def _pair():
    from quantark.asset.fx.product import CurrencyPair

    return CurrencyPair("EUR", "USD")


# ------------------------------------------------------------------- builders
def _build_equity_mc():
    out = {}
    mc = dict(num_paths=1_024, time_steps=24, seed=19)

    def european():
        from quantark.asset.equity.engine.mc import EuropeanMCEngine

        return (
            EuropeanMCEngine(params=_mcp(num_paths=64, time_steps=4, seed=42)),
            _euro(), _eq_flat_env(), "product_env",
        )

    out["EuropeanMCEngine"] = european

    def local_vol():
        from quantark.asset.equity.engine.mc import LocalVolMCEngine

        return (
            LocalVolMCEngine(params=_mcp(num_paths=2_048, time_steps=24, seed=11)),
            _euro(), _eq_grid_env(), "product_env",
        )

    out["LocalVolMCEngine"] = local_vol

    def heston():
        from quantark.asset.equity.engine.mc import HestonMCEngine
        from quantark.util.enum.engine_enums import HestonMCScheme

        return (
            HestonMCEngine(_hp(), scheme=HestonMCScheme.QUADEXP,
                           params=_mcp(num_paths=2_048, time_steps=24, seed=1)),
            _euro(), _eq_flat_env(), "product_env",
        )

    out["HestonMCEngine"] = heston

    def heston_slv():
        from quantark.asset.equity.engine.mc import HestonSLVMCEngine

        return (
            HestonSLVMCEngine(_hp(), eta=1.0,
                              params=_mcp(num_paths=2_048, time_steps=24, seed=1),
                              leverage_surface=_unit_leverage()),
            _euro(), _eq_grid_env(), "product_env",
        )

    out["HestonSLVMCEngine"] = heston_slv

    def sabr():
        from quantark.asset.equity.engine.mc import SABRMCEngine
        from quantark.param import FlatRateCurve, SpotQuote
        from quantark.param.vol import SABRVolSurface
        from quantark.priceenv import PricingEnvironment

        env = PricingEnvironment(
            rate_curve=FlatRateCurve(rate=0.0),
            valuation_date=datetime(2026, 6, 24),
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=SABRVolSurface.from_params(
                alpha=0.2, beta=1.0, rho=-0.4, nu=0.5, maturity=1.0
            ),
        )
        return (
            SABRMCEngine(params=_mcp(num_paths=4_096, time_steps=8, seed=5)),
            _euro(), env, "product_env",
        )

    out["SABRMCEngine"] = sabr

    def american():
        from quantark.asset.equity.product.option import AmericanOption
        from quantark.asset.equity.engine.mc import AmericanOptionMCEngine
        from quantark.util.enum import OptionType
        from quantark.util.enum.engine_enums import MonteCarloMethod

        return (
            AmericanOptionMCEngine(
                params=_mcp(num_paths=2_000, time_steps=50, seed=42),
                method=MonteCarloMethod.QUASI,
            ),
            AmericanOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0),
            _eq_flat_env(), "product_env",
        )

    out["AmericanOptionMCEngine"] = american

    def asian():
        from quantark.asset.equity.engine.mc import AsianOptionMCEngine
        from quantark.asset.equity.product.option import AsianOption
        from quantark.asset.equity.product.option.asian_option import (
            AsianObservationRecord,
        )
        from quantark.util.enum import (
            AsianStrikeType, AveragingType, OptionType,
        )

        product = AsianOption(
            strike=100.0, option_type=OptionType.CALL,
            asian_strike_type=AsianStrikeType.FIXED,
            averaging_type=AveragingType.ARITHMETIC, maturity=1.0,
            observation_records=[
                AsianObservationRecord(observation_time=t)
                for t in [0.25, 0.5, 0.75, 1.0]
            ],
        )
        return (
            AsianOptionMCEngine(params=_mcp(num_paths=1_000, seed=3)),
            product, _eq_flat_env(), "product_env",
        )

    out["AsianOptionMCEngine"] = asian

    def digital():
        from quantark.asset.equity.engine.mc import DigitalOptionMCEngine
        from quantark.asset.equity.product.option.digital_option import (
            CashOrNothingDigitalOption,
        )
        from quantark.util.enum import OptionType

        return (
            DigitalOptionMCEngine(params=_mcp(num_paths=1_000, seed=3)),
            CashOrNothingDigitalOption(
                strike=100.0, option_type=OptionType.CALL,
                maturity=1.0, cash_payout=10.0,
            ),
            _eq_flat_env(), "product_env",
        )

    out["DigitalOptionMCEngine"] = digital

    def barrier():
        from quantark.asset.equity.engine.mc import BarrierOptionMCEngine

        return (
            BarrierOptionMCEngine(params=_mcp(num_paths=2_000, time_steps=50, seed=123)),
            _barrier(), _eq_flat_env(), "product_env",
        )

    out["BarrierOptionMCEngine"] = barrier

    for name, cls_name in [
        ("LocalVolBarrierMCEngine", "lv"),
        ("HestonBarrierMCEngine", "heston"),
        ("HestonSLVBarrierMCEngine", "slv"),
    ]:
        def barrier_vol(kind=cls_name):
            from quantark.asset.equity.engine.mc import (
                HestonBarrierMCEngine,
                HestonSLVBarrierMCEngine,
                LocalVolBarrierMCEngine,
            )

            mcp = _mcp(num_paths=2_048, time_steps=24, seed=2)
            if kind == "lv":
                return (LocalVolBarrierMCEngine(mcp), _barrier(),
                        _eq_grid_env(), "product_env")
            if kind == "heston":
                return (HestonBarrierMCEngine(_hp(), mcp), _barrier(),
                        _eq_grid_env(), "product_env")
            return (HestonSLVBarrierMCEngine(_hp(), _unit_leverage(), mcp),
                    _barrier(), _eq_grid_env(), "product_env")

        out[name] = barrier_vol

    def sharkfin_single():
        from quantark.asset.equity.engine.mc import SingleSharkfinOptionMCEngine
        from quantark.asset.equity.product.option import SingleSharkfinOption
        from quantark.util.enum import ObservationType, OptionType

        return (
            SingleSharkfinOptionMCEngine(params=_mcp(num_paths=1_024, time_steps=16, seed=5)),
            SingleSharkfinOption(
                strike=95.0, option_type=OptionType.CALL, barrier=120.0,
                maturity=1.0, participation_rate=0.7, knock_out_rebate=2.0,
                no_hit_rebate=0.5, observation_type=ObservationType.EXPIRY,
            ),
            _eq_flat_env(), "product_env",
        )

    out["SingleSharkfinOptionMCEngine"] = sharkfin_single

    def sharkfin_double():
        from quantark.asset.equity.engine.mc import DoubleSharkfinOptionMCEngine
        from quantark.asset.equity.product.option import DoubleSharkfinOption
        from quantark.util.enum import ObservationType, OptionType

        return (
            DoubleSharkfinOptionMCEngine(params=_mcp(num_paths=1_024, time_steps=16, seed=123)),
            DoubleSharkfinOption(
                strike=100.0, option_type=OptionType.CALL, lower_barrier=70.0,
                upper_barrier=130.0, maturity=1.0, participation_rate=0.8,
                knock_out_rebate=2.0, no_hit_rebate=0.5,
                observation_type=ObservationType.EXPIRY,
            ),
            _eq_flat_env(), "product_env",
        )

    out["DoubleSharkfinOptionMCEngine"] = sharkfin_double

    def range_accrual():
        from quantark.asset.equity.engine.mc import RangeAccrualMCEngine
        from quantark.asset.equity.product.option.range_accrual_config import (
            RangeAccrualConfig,
        )
        from quantark.asset.equity.product.option.range_accrual_option import (
            RangeAccrualOption,
        )

        return (
            RangeAccrualMCEngine(params=_mcp(num_paths=2_000, seed=42)),
            RangeAccrualOption(
                initial_price=100.0,
                range_config=RangeAccrualConfig(
                    upper_barrier=110.0, lower_barrier=90.0,
                    accrual_rate=0.05, is_rate_annualized=True,
                ),
                observation_times=[0.25, 0.5, 0.75, 1.0],
                maturity=1.0, contract_multiplier=10_000.0,
            ),
            _eq_flat_env(), "product_env",
        )

    out["RangeAccrualMCEngine"] = range_accrual

    def accumulator():
        from quantark.asset.equity.engine.mc import AccumulatorMCEngine
        from quantark.asset.equity.product.option import AccumulatorOption
        from quantark.util.enum import AccumulatorKnockOutType, OptionType

        obs = [round(m / 12.0, 6) for m in range(1, 13)]
        return (
            AccumulatorMCEngine(_mcp(num_paths=2_000, seed=7)),
            AccumulatorOption(
                strike=96.0, knock_out_barrier=1.0e6,
                option_type=OptionType.CALL, maturity=1.0,
                daily_share_accumulation=1.0, gearing=2.0,
                knock_out_type=AccumulatorKnockOutType.TERMINATION,
                observation_dates=obs,
            ),
            _eq_flat_env(), "product_env",
        )

    out["AccumulatorMCEngine"] = accumulator

    def snowball_base():
        from quantark.asset.equity.engine.mc import SnowballMCEngine

        return (
            SnowballMCEngine(params=_mcp(num_paths=2_000, time_steps=64, seed=7)),
            _snowball(), _eq_flat_env(), "product_env",
        )

    out["SnowballMCEngine"] = snowball_base

    def phoenix_base():
        from quantark.asset.equity.engine.mc import PhoenixMCEngine

        return (
            PhoenixMCEngine(params=_mcp(num_paths=2_000, seed=7)),
            _phoenix(), _eq_flat_env(), "product_env",
        )

    out["PhoenixMCEngine"] = phoenix_base

    # Snowball/Phoenix vol-model MC (test_snowball_vol_model_engines.py:119-123,
    # test_phoenix_vol_model_engines.py:110-114)
    def _autocall_vol(name):
        def build():
            from quantark.asset.equity.engine.mc import (
                HestonPhoenixMCEngine, HestonSLVPhoenixMCEngine,
                HestonSLVQEPhoenixMCEngine, HestonSLVQESnowballMCEngine,
                HestonSLVSnowballMCEngine, HestonSnowballMCEngine,
                LocalVolPhoenixMCEngine, LocalVolSnowballMCEngine,
                QEPhoenixMCEngine, QESnowballMCEngine,
            )

            mcp = _mcp(num_paths=1_024, time_steps=24, seed=19)
            product = _snowball() if "Snowball" in name else _phoenix()
            cls = {
                "LocalVolSnowballMCEngine": lambda: LocalVolSnowballMCEngine(mcp),
                "HestonSnowballMCEngine": lambda: HestonSnowballMCEngine(_hp(), mcp),
                "QESnowballMCEngine": lambda: QESnowballMCEngine(_hp(), mcp),
                "HestonSLVSnowballMCEngine": lambda: HestonSLVSnowballMCEngine(
                    _hp(), params=mcp, leverage_surface=_unit_leverage()),
                "HestonSLVQESnowballMCEngine": lambda: HestonSLVQESnowballMCEngine(
                    _hp(), params=mcp, leverage_surface=_unit_leverage()),
                "LocalVolPhoenixMCEngine": lambda: LocalVolPhoenixMCEngine(mcp),
                "HestonPhoenixMCEngine": lambda: HestonPhoenixMCEngine(_hp(), mcp),
                "QEPhoenixMCEngine": lambda: QEPhoenixMCEngine(_hp(), mcp),
                "HestonSLVPhoenixMCEngine": lambda: HestonSLVPhoenixMCEngine(
                    _hp(), params=mcp, leverage_surface=_unit_leverage()),
                "HestonSLVQEPhoenixMCEngine": lambda: HestonSLVQEPhoenixMCEngine(
                    _hp(), params=mcp, leverage_surface=_unit_leverage()),
            }[name]
            return cls(), product, _eq_grid_env(), "product_env"

        return build

    for name in [
        "LocalVolSnowballMCEngine", "HestonSnowballMCEngine",
        "QESnowballMCEngine", "HestonSLVSnowballMCEngine",
        "HestonSLVQESnowballMCEngine", "LocalVolPhoenixMCEngine",
        "HestonPhoenixMCEngine", "QEPhoenixMCEngine",
        "HestonSLVPhoenixMCEngine", "HestonSLVQEPhoenixMCEngine",
    ]:
        out[name] = _autocall_vol(name)

    # DCN MC (test_dcn_mc_engine.py:13, test_dcn_vol_mc_engines.py:39-83,
    # test_dcn_coupled_ladder.py:24-34)
    def dcn():
        from quantark.asset.equity.engine.mc import DCNMCEngine

        return (DCNMCEngine(num_paths=2**9, seed=42), _dcn(),
                _dcn_flat_env(), "product_env")

    out["DCNMCEngine"] = dcn

    def dcn_lv():
        from quantark.asset.equity.engine.mc import LocalVolDCNMCEngine

        return (LocalVolDCNMCEngine(num_paths=2**9, seed=42), _dcn(),
                _dcn_grid_env(), "product_env")

    out["LocalVolDCNMCEngine"] = dcn_lv

    def dcn_heston():
        from quantark.asset.equity.engine.mc import HestonDCNMCEngine

        return (HestonDCNMCEngine(model_params=_dcn_hp(), num_paths=2**9, seed=42),
                _dcn(), _dcn_flat_env(), "product_env")

    out["HestonDCNMCEngine"] = dcn_heston

    def dcn_qe():
        from quantark.asset.equity.engine.mc import QEDCNMCEngine

        return (QEDCNMCEngine(_dcn_hp(), num_paths=2**9, seed=7), _dcn(),
                _dcn_flat_env(), "product_env")

    out["QEDCNMCEngine"] = dcn_qe

    def dcn_coupled():
        from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import (
            coupled_heston_ladder_pair,
        )
        from quantark.util.enum.engine_enums import HestonMCScheme

        coarse, _fine = coupled_heston_ladder_pair(
            _dcn_hp(), 2, HestonMCScheme.QUADEXP_M,
            num_paths=2**9, seed=42, use_sobol=True, num_batches=1,
        )
        return coarse, _dcn(), _dcn_flat_env(), "product_env"

    out["CoupledCoarseHestonDCNMCEngine"] = dcn_coupled
    return out


def _build_equity_pde():
    out = {}
    pde = dict(grid_size=90, time_steps=48)

    simple = {
        "EuropeanPDESolver": ("EuropeanPDESolver", _euro),
        "LocalVolPDESolver": ("LocalVolPDESolver", _euro),
        "SnowballPDESolver": ("SnowballPDESolver", _snowball),
        "LocalVolSnowballPDESolver": ("LocalVolSnowballPDESolver", _snowball),
        "PhoenixPDESolver": ("PhoenixPDESolver", _phoenix),
        "LocalVolPhoenixPDESolver": ("LocalVolPhoenixPDESolver", _phoenix),
        "BarrierPDESolver": ("BarrierPDESolver", lambda: _barrier(0.5)),
        "LocalVolBarrierPDESolver": ("LocalVolBarrierPDESolver", _barrier),
    }

    def _simple(cls_name, product_fn, grid_env):
        def build():
            import quantark.asset.equity.engine.pde as pde_mod

            cls = getattr(pde_mod, cls_name)
            engine = cls(_pdep(grid_size=90, time_steps=48, auto_grid=False))
            env = _eq_grid_env() if grid_env else _eq_flat_env()
            return engine, product_fn(), env, "product_env"

        return build

    for name, (cls_name, product_fn) in simple.items():
        out[name] = _simple(cls_name, product_fn, grid_env="LocalVol" in name)

    def american_pde():
        from quantark.asset.equity.engine.pde import AmericanPDESolver
        from quantark.asset.equity.product.option import AmericanOption
        from quantark.util.enum import OptionType

        return (
            AmericanPDESolver(_pdep(grid_size=90, time_steps=48, auto_grid=False)),
            AmericanOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0),
            _eq_flat_env(), "product_env",
        )

    out["AmericanPDESolver"] = american_pde

    def double_barrier():
        from quantark.asset.equity.engine.pde import DoubleBarrierPDESolver
        from quantark.asset.equity.product.option import DoubleBarrierOption
        from quantark.util.enum import DoubleBarrierType, OptionType

        return (
            DoubleBarrierPDESolver(_pdep(grid_size=90, time_steps=48, auto_grid=False)),
            DoubleBarrierOption(
                strike=100.0, option_type=OptionType.CALL, upper_barrier=120.0,
                lower_barrier=80.0, barrier_type=DoubleBarrierType.KNOCK_OUT,
                maturity=0.5,
            ),
            _eq_flat_env(), "product_env",
        )

    out["DoubleBarrierPDESolver"] = double_barrier

    def one_touch():
        from quantark.asset.equity.engine.pde import OneTouchPDESolver
        from quantark.asset.equity.product.option import OneTouchOption
        from quantark.util.enum import BarrierDirection, TouchType

        return (
            OneTouchPDESolver(_pdep(grid_size=90, time_steps=48, auto_grid=False)),
            OneTouchOption(
                barrier=110.0, barrier_direction=BarrierDirection.UP,
                maturity=1.0, rebate=100.0, payment_at_hit=True,
                touch_type=TouchType.ONE_TOUCH,
            ),
            _eq_flat_env(), "product_env",
        )

    out["OneTouchPDESolver"] = one_touch

    def double_one_touch():
        from quantark.asset.equity.engine.pde import DoubleOneTouchPDESolver
        from quantark.asset.equity.product.option import DoubleOneTouchOption
        from quantark.util.enum import TouchType

        return (
            DoubleOneTouchPDESolver(_pdep(grid_size=90, time_steps=48, auto_grid=False)),
            DoubleOneTouchOption(
                upper_barrier=110.0, lower_barrier=90.0, maturity=1.0,
                rebate=100.0, payment_at_hit=True,
                touch_type=TouchType.DOUBLE_ONE_TOUCH,
            ),
            _eq_flat_env(), "product_env",
        )

    out["DoubleOneTouchPDESolver"] = double_one_touch

    def ko_reset():
        from quantark.asset.equity.engine.pde import KOResetSnowballPDESolver
        from quantark.asset.equity.product.option.ko_reset_snowball_option import (
            create_ko_reset_snowball,
        )
        from quantark.util.enum import PostKOScheduleMode

        return (
            KOResetSnowballPDESolver(_pdep(grid_size=80, time_steps=40, auto_grid=False)),
            create_ko_reset_snowball(
                initial_price=100.0, strike=100.0, maturity_pre=1.0,
                maturity_post=2.0, post_ko_mode=PostKOScheduleMode.ABSOLUTE,
                ki_continuous=False,
            ),
            _eq_flat_env(), "product_env",
        )

    out["KOResetSnowballPDESolver"] = ko_reset

    def _heston_pde(name):
        def build():
            from quantark.asset.equity.engine.pde import (
                HestonBarrierPDESolver, HestonPDESolver,
                HestonPhoenixPDESolver, HestonSLVBarrierPDESolver,
                HestonSLVPDESolver, HestonSLVPhoenixPDESolver,
                HestonSLVSnowballPDESolver, HestonSnowballPDESolver,
            )

            grid = dict(n_x=48, n_v=18, n_t=16)
            table = {
                "HestonPDESolver": lambda: (HestonPDESolver(_hp(), **grid), _euro(), _eq_flat_env()),
                "HestonSLVPDESolver": lambda: (HestonSLVPDESolver(_hp(), _unit_leverage(), eta=1.0, **grid), _euro(), _eq_grid_env()),
                "HestonBarrierPDESolver": lambda: (HestonBarrierPDESolver(_hp(), **grid), _barrier(), _eq_grid_env()),
                "HestonSLVBarrierPDESolver": lambda: (HestonSLVBarrierPDESolver(_hp(), _unit_leverage(), **grid), _barrier(), _eq_grid_env()),
                "HestonSnowballPDESolver": lambda: (HestonSnowballPDESolver(_hp(), **grid), _snowball(), _eq_grid_env()),
                "HestonSLVSnowballPDESolver": lambda: (HestonSLVSnowballPDESolver(_hp(), _unit_leverage(), **grid), _snowball(), _eq_grid_env()),
                "HestonPhoenixPDESolver": lambda: (HestonPhoenixPDESolver(_hp(), grid_style="uniform", **grid), _phoenix(), _eq_grid_env()),
                "HestonSLVPhoenixPDESolver": lambda: (HestonSLVPhoenixPDESolver(_hp(), _unit_leverage(), grid_style="uniform", **grid), _phoenix(), _eq_grid_env()),
            }
            engine, product, env = table[name]()
            return engine, product, env, "product_env"

        return build

    for name in [
        "HestonPDESolver", "HestonSLVPDESolver", "HestonBarrierPDESolver",
        "HestonSLVBarrierPDESolver", "HestonSnowballPDESolver",
        "HestonSLVSnowballPDESolver", "HestonPhoenixPDESolver",
        "HestonSLVPhoenixPDESolver",
    ]:
        out[name] = _heston_pde(name)

    def dcn_pde():
        from quantark.asset.equity.engine.pde import DCNPDEEngine

        return (DCNPDEEngine(num_space_nodes=301), _dcn(),
                _dcn_flat_env(), "product_env")

    out["DCNPDEEngine"] = dcn_pde

    def dcn_lv_pde():
        from quantark.asset.equity.engine.pde import LocalVolDCNPDEEngine

        return (LocalVolDCNPDEEngine(num_space_nodes=301), _dcn(),
                _dcn_grid_env(), "product_env")

    out["LocalVolDCNPDEEngine"] = dcn_lv_pde

    def dcn_heston_pde():
        from quantark.asset.equity.engine.pde import HestonDCNPDESolver

        return (HestonDCNPDESolver(_dcn_hp(), n_x=151, n_v=41), _dcn(),
                _dcn_flat_env(), "product_env")

    out["HestonDCNPDESolver"] = dcn_heston_pde

    def pde_facade():
        from quantark.asset.equity.engine.pde_engine import PDEEngine

        return (
            PDEEngine(_pdep(grid_size=80, time_steps=40, auto_grid=False)),
            _euro(), _eq_flat_env(), "product_env",
        )

    out["PDEEngine"] = pde_facade
    return out


def _build_fx():
    out = {}

    def fx_lv_mc():
        from quantark.asset.fx.engine.mc import FxLocalVolMCEngine

        return (FxLocalVolMCEngine(num_paths=4_000, time_steps=24, seed=5),
                _fx_vanilla(), _fx_grid_env(), "product_env")

    out["FxLocalVolMCEngine"] = fx_lv_mc

    def fx_heston_mc():
        from quantark.asset.fx.engine.mc import FxHestonMCEngine

        return (FxHestonMCEngine(_hp(), num_paths=4_000, time_steps=24, seed=9),
                _fx_vanilla(), _fx_flat_env(), "product_env")

    out["FxHestonMCEngine"] = fx_heston_mc

    def fx_slv_mc():
        from quantark.asset.fx.engine.mc import FxHestonSLVMCEngine

        return (
            FxHestonSLVMCEngine(_hp(), eta=1.0, num_paths=4_000,
                                time_steps=24, seed=9),
            _fx_vanilla(), _fx_grid_env(), "product_env",
        )

    out["FxHestonSLVMCEngine"] = fx_slv_mc

    def fx_lv_pde():
        from quantark.asset.fx.engine.pde import FxLocalVolPDESolver

        return (FxLocalVolPDESolver(grid_size=120, time_steps=48),
                _fx_vanilla(), _fx_grid_env(), "product_env")

    out["FxLocalVolPDESolver"] = fx_lv_pde

    def fx_heston_pde():
        from quantark.asset.fx.engine.pde import FxHestonPDESolver

        return (FxHestonPDESolver(_hp(), n_x=48, n_v=18, n_t=16),
                _fx_vanilla(), _fx_flat_env(), "product_env")

    out["FxHestonPDESolver"] = fx_heston_pde

    def fx_slv_pde():
        from quantark.asset.fx.engine.pde import FxHestonSLVPDESolver

        return (
            FxHestonSLVPDESolver(_hp(), _fx_unit_leverage(), eta=1.0,
                                 n_x=48, n_v=18, n_t=16),
            _fx_vanilla(), _fx_grid_env(), "product_env",
        )

    out["FxHestonSLVPDESolver"] = fx_slv_pde

    def _fx_params(**kw):
        from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams

        return FxMCParams(**kw)

    def fx_range_accrual():
        from quantark.asset.fx.engine.mc import FxRangeAccrualMCEngine
        from quantark.asset.fx.product.option.fx_range_accrual_option import (
            FxRangeAccrualConfig, FxRangeAccrualOption,
        )

        return (
            FxRangeAccrualMCEngine(params=_fx_params(num_paths=2_000, seed=1)),
            FxRangeAccrualOption(
                notional=1_000_000.0,
                range_config=FxRangeAccrualConfig(
                    upper_barrier=1.30, lower_barrier=1.10, accrual_rate=0.04
                ),
                currency_pair=_pair(), maturity=1.0, num_observations=12,
            ),
            _fx_flat_env(), "product_env",
        )

    out["FxRangeAccrualMCEngine"] = fx_range_accrual

    def fx_barrier():
        from quantark.asset.fx.engine.mc import FxBarrierMCEngine
        from quantark.asset.fx.product.option import FxBarrierOption
        from quantark.util.enum import FxBarrierType, OptionType

        return (
            FxBarrierMCEngine(params=_fx_params(num_paths=4_000, time_steps=24, seed=3)),
            FxBarrierOption(
                strike=1.20, barrier=1.35, is_up=True,
                knock_type=FxBarrierType.KNOCK_OUT,
                option_type=OptionType.CALL,
                currency_pair=_pair(), maturity=1.0,
            ),
            _fx_flat_env(), "product_env",
        )

    out["FxBarrierMCEngine"] = fx_barrier

    def fx_sharkfin():
        from quantark.asset.fx.engine.mc import FxSharkfinMCEngine
        from quantark.asset.fx.product.option import FxSharkfinOption
        from quantark.util.enum import OptionType

        return (
            FxSharkfinMCEngine(params=_fx_params(num_paths=4_000, time_steps=24, seed=3)),
            FxSharkfinOption(
                strike=1.20, barrier=1.35, is_up=True,
                option_type=OptionType.CALL,
                currency_pair=_pair(), maturity=1.0,
            ),
            _fx_flat_env(), "product_env",
        )

    out["FxSharkfinMCEngine"] = fx_sharkfin

    def fx_tarf():
        from quantark.asset.fx.engine.mc import FxTarnForwardMCEngine
        from quantark.asset.fx.product import FxTargetRedemptionForward

        return (
            FxTarnForwardMCEngine(params=_fx_params(num_paths=2_000, seed=7)),
            FxTargetRedemptionForward(
                strike=1.20, fixing_times=[0.25, 0.5, 0.75, 1.0],
                currency_pair=_pair(),
            ),
            _fx_flat_env(), "product_env",
        )

    out["FxTarnForwardMCEngine"] = fx_tarf

    def fx_tarn():
        from quantark.asset.fx.engine.mc import FxTargetRedemptionNoteMCEngine
        from quantark.asset.fx.product import FxTargetRedemptionNote

        return (
            FxTargetRedemptionNoteMCEngine(params=_fx_params(num_paths=2_000, seed=11)),
            FxTargetRedemptionNote(
                fixing_times=[0.25, 0.5, 0.75, 1.0], coupon_rate=0.08,
                notional=1.0, strike=1.20, currency_pair=_pair(),
            ),
            _fx_flat_env(), "product_env",
        )

    out["FxTargetRedemptionNoteMCEngine"] = fx_tarn
    return out


def _build_credit_bond():
    out = {}

    def basket_cds():
        from quantark.asset.credit.engine.mc import BasketCDSEngine
        from quantark.asset.credit.product import BasketCDS, BasketType
        from quantark.param import FlatRateCurve
        from quantark.param.credit import FlatHazardCurve
        from quantark.priceenv import BasketCreditPricingEnvironment

        n = 5
        corr = np.full((n, n), 0.3)
        np.fill_diagonal(corr, 1.0)
        return (
            BasketCDSEngine(n_simulations=5_000, seed=11),
            BasketCDS(
                notional=10_000_000.0, maturity=5.0,
                recovery_rates=[0.4] * n, basket_type=BasketType.FTD,
                n_to_default=1, correlation_matrix=corr,
            ),
            BasketCreditPricingEnvironment(
                valuation_date=datetime(2026, 6, 13),
                discount_curve=FlatRateCurve(rate=0.03),
                hazard_curves=[FlatHazardCurve(hazard_rate=0.02)] * n,
            ),
            "product_env",
        )

    out["BasketCDSEngine"] = basket_cds

    def _cb_fixture():
        from quantark.asset.bond.product.convertible.convertible_bond import (
            ConvertibleBond,
        )
        from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
        from quantark.priceenv import PricingEnvironment

        cb = ConvertibleBond(
            issue_date=datetime(2024, 1, 1), maturity_date=datetime(2029, 1, 1),
            face_value=100.0, coupon_rate=0.02, conversion_ratio=10.0,
            credit_spread=0.02, hazard_rate=0.01, recovery_rate=0.4,
        )
        env = PricingEnvironment(
            valuation_date=datetime(2024, 6, 1),
            spot_quote=SpotQuote(spot=12.0),
            vol_surface=FlatVolSurface(volatility=0.30),
            rate_curve=FlatRateCurve(rate=0.05),
        )
        return cb, env

    def cb_jump():
        from quantark.asset.bond.engine.pde import (
            ConvertibleBondJumpDiffusionEngine, ConvertibleBondPDEParams,
        )

        cb, env = _cb_fixture()
        return (
            ConvertibleBondJumpDiffusionEngine(
                env, ConvertibleBondPDEParams(num_space_steps=40, num_time_steps=80)
            ),
            cb, None, "env_bound",
        )

    out["ConvertibleBondJumpDiffusionEngine"] = cb_jump

    def cb_tf():
        from quantark.asset.bond.engine.pde import (
            ConvertibleBondPDEParams, ConvertibleBondTFEngine,
        )

        cb, env = _cb_fixture()
        return (
            ConvertibleBondTFEngine(
                env, ConvertibleBondPDEParams(num_space_steps=40, num_time_steps=80)
            ),
            cb, None, "env_bound",
        )

    out["ConvertibleBondTFEngine"] = cb_tf

    def cb_facade():
        from quantark.asset.bond.engine.convertible import ConvertibleBondEngine
        from quantark.util.enum.engine_enums import (
            ConvertibleBondMethod, EngineType,
        )

        cb, env = _cb_fixture()
        return (
            ConvertibleBondEngine(
                env, method=EngineType.PDE(ConvertibleBondMethod.TF)
            ),
            cb, None, "env_bound",
        )

    out["ConvertibleBondEngine"] = cb_facade
    return out


FIXTURE_BUILDERS = {
    **_build_equity_mc(),
    **_build_equity_pde(),
    **_build_fx(),
    **_build_credit_bond(),
}
```

- [ ] **Step 1: Write `matrix_fixtures.py`** (above) and the parity test:

```python
# test/execution/test_matrix_parity.py
"""Phase 1 exit gate: direct-vs-session parity for EVERY concrete inventory
row (spec section 21 Phase 1)."""
import pytest

from quantark.execution import PricingRequest, PricingSession
from quantark.execution.inventory import ENGINE_INVENTORY

from execution.matrix_fixtures import FIXTURE_BUILDERS

CONCRETE = [r for r in ENGINE_INVENTORY if r.role != "abstract"]


def test_every_concrete_row_has_a_fixture():
    missing = [r.name for r in CONCRETE if r.name not in FIXTURE_BUILDERS]
    assert not missing, f"inventory rows without executable fixtures: {missing}"


@pytest.mark.parametrize("record", CONCRETE, ids=lambda r: r.name)
def test_direct_equals_session(record):
    engine, product, env, call_shape = FIXTURE_BUILDERS[record.name]()
    assert call_shape == record.call_shape
    if call_shape == "env_bound":
        direct = engine.price(product)
    else:
        direct = engine.price(product, env)
    with PricingSession() as session:
        outcome = session.execute(
            engine if call_shape == "env_bound"
            else engine,
            PricingRequest(product=product, pricing_env=env),
        )
    assert outcome.value == direct, record.name
    assert type(outcome.value) is type(direct), record.name
```

Engines with stochastic-but-seeded results must compare EXACTLY equal — the
session path runs the same code. Any mismatch is a framework bug, not
tolerance noise. If a specific engine mutates internal state so that the
second pricing differs from the first even directly (path-dependent RNG
state), rebuild a fresh fixture for each side inside the test — do this ONLY
for engines where `engine.price(p, e) != engine.price(p, e)` reproducibly,
and document each in the test with a comment. (The DCN LV adapter clone is
covered separately in `test_kernel_prepare.py`.)

- [ ] **Step 2: Run** `PYTHONPATH=$PWD ... -m pytest -n0 test/execution/test_matrix_parity.py -q`
  Expected first run: collection succeeds, individual engines may fail on
  fixture details (constructor drift). Fix each failing fixture against its
  cited source test until green. This step is expected to be iterative.
- [ ] **Step 3: Timebox check** — the whole file must run < ~90s serial
  (`-n0`). If an engine dominates, shrink its parameters (respect ctor
  minimums, e.g. `num_space_nodes >= 201`; Heston PDE `n_x=48,n_v=18,n_t=16`
  is the tested floor).
- [ ] **Step 4: Commit** — `test(execution): full-matrix direct-vs-session parity fixtures (73 engines)`

---

### Task 7: Regression gates and full suite

**Files:**
- Create: `test/execution/test_regression_gates.py`

- [ ] **Step 1: Write the gates**

```python
# test/execution/test_regression_gates.py
"""Phase 1 regression gates (spec section 20 gates 1-2, CI smoke form)."""
import dataclasses
import time

from quantark.execution import PricingRequest, PricingSession, ResourceBudget
from quantark.execution.context import default_context

from execution.matrix_fixtures import FIXTURE_BUILDERS


def test_disabled_cache_matches_enabled_cache():
    """Gate: caches off == caches on, exactly (spec section 20 gate 2)."""
    engine, product, env, _ = FIXTURE_BUILDERS["LocalVolDCNMCEngine"]()
    with PricingSession() as s_on:
        v_on = s_on.price(engine, product, env)
    ctx = dataclasses.replace(
        default_context(),
        resource_budget=ResourceBudget(artifact_cache_bytes=0),
    )
    with PricingSession(ctx) as s_off:
        v_off = s_off.price(engine, product, env)
    assert v_on == v_off


def test_serial_session_overhead_smoke():
    """CI smoke for spec section 20 gate 1 (the 3% gate runs on a controlled
    host; this catches gross regressions only)."""
    engine, product, env, _ = FIXTURE_BUILDERS["EuropeanMCEngine"]()

    def median_seconds(fn, n=15):
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return sorted(times)[n // 2]

    engine.price(product, env)  # warm-up
    direct = median_seconds(lambda: engine.price(product, env))
    with PricingSession() as session:
        session.price(engine, product, env)  # warm-up
        via = median_seconds(lambda: session.price(engine, product, env))
    # generous CI bound: framework layers must stay in the noise, not 2x
    assert via <= direct * 2.0 + 0.005
```

- [ ] **Step 2: Run the whole execution suite + full suite**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/execution/ -q
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest -q
```

Expected: execution suite fully green; full suite has no NEW failures versus
main (the pre-existing `test_snowball_quad_flat_identity_golden` failure is
known and out of scope).

- [ ] **Step 3: Commit** — `test(execution): phase-1 disabled-cache and serial-overhead gates`

---

## Phase 1 exit checklist (spec §21)

- [ ] Direct-versus-session parity for EVERY concrete inventory row
  (`test_matrix_parity.py`, 73 rows).
- [ ] DCN preparation adapted: cached Dupire build with build-count proof;
  mutable `_active_surface` never touched on the adapter path; prebuilt and
  disabled-cache paths exact.
- [ ] Serial regression gate (smoke) and disabled-cache gate pass.
- [ ] Artifact cache: single-flight, LRU-by-bytes, pin-respecting eviction,
  leader-failure recovery, idempotent close — all tested.
- [ ] Full suite: no new failures.

## Self-Review Notes

- Spec §7 lifecycle: normalize/resolve/validate (Phase 0) + reserve/prepare/
  release (this phase); plan/reduce steps arrive with BatchPlan in Phase 2.
- §10.1: descriptor-equality hit verification is the dict key; single-flight
  state machine matches the hardened contract (leader failure → publish,
  atomic cleanup, re-election). `invalidate_tags` skips pinned entries —
  documented Phase 4 tightening.
- §17.1: DCN engine files untouched; adapter clones via public constructors.
- §11: byte budget is admission accounting; oversize artifacts bypass the
  cache rather than failing the request (correctness first).
- Fixture constructor shapes come from cited tests; Step 2 of Task 6
  explicitly budgets an iterative fix-up pass for drift.
- Codex plan-gate dispositions (all four applied): (1) DCN prepare binds its
  inputs once and re-verifies the fingerprint after build, raising
  DeterminismViolation on mutation; (2) the cache reserves bytes BEFORE
  invoking the builder, converting the reservation to the entry on publish;
  (3) close() is synchronized with leader publication and the lease manager
  rejects post-close acquisitions; (4) sessions reject partial cache/lease
  injection and verify pair identity via cache.lease_manager. Borrowed
  (supplied) services are never closed by the session.
