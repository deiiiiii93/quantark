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
    """A leader finishing after close() must not publish or retain bytes."""
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
    """The byte lease is held BEFORE builder() runs."""
    mgr = ResourceLeaseManager(ResourceBudget(artifact_cache_bytes=1000))
    cache = PreparedArtifactCache(mgr)
    seen = []

    def builder():
        seen.append(mgr.pool_bytes("artifact_cache"))
        return "V"

    with cache.get_or_build(_desc("r"), builder, size_bytes=100):
        pass
    assert seen == [100]  # lease already charged while building


def test_close_is_idempotent_and_releases_bytes():
    mgr = ResourceLeaseManager(ResourceBudget(artifact_cache_bytes=1000))
    cache = PreparedArtifactCache(mgr)
    with cache.get_or_build(_desc("a"), lambda: "A", size_bytes=100):
        pass
    cache.close(); cache.close()
    assert mgr.pool_bytes("artifact_cache") == 0
    with pytest.raises(PreparationError):
        cache.get_or_build(_desc("a"), lambda: "A", size_bytes=10)


def test_cache_pool_is_parametrizable():
    budget = ResourceBudget(draw_cache_bytes=1024, artifact_cache_bytes=0)
    mgr = ResourceLeaseManager(budget)
    cache = PreparedArtifactCache(mgr, pool="draw_cache")
    desc = _desc()
    handle = cache.get_or_build(desc, lambda: b"x" * 10, size_bytes=10)
    assert mgr.pool_bytes("draw_cache") == 10
    assert mgr.pool_bytes("artifact_cache") == 0
    handle.close()
