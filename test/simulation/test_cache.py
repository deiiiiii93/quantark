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


from quantark.backtest.simulation.pricing.cache import DiskTier, shard_name  # noqa: E402


def test_the_disk_tier_round_trips_between_two_caches(tmp_path):
    first = StateCache(CacheConfig(memory_bytes=10 * ENTRY_BYTES, disk_dir=str(tmp_path)))
    first.put(_key(1), 1.5, -0.2, 0.01)
    first.put(_key(2), 2.5, -0.3, 0.02)
    assert first.flush() == 2
    second = StateCache(CacheConfig(memory_bytes=10 * ENTRY_BYTES, disk_dir=str(tmp_path)))
    assert second.get(_key(1)) == (1.5, -0.2, 0.01)
    assert second.get(_key(3)) is None
    stats = second.stats()
    assert stats.disk["disk_hits"] == 1 and stats.disk["shards_loaded"] == 1
    assert stats.entries == 1                       # the hit was promoted to memory


def test_flush_merges_with_what_another_process_wrote(tmp_path):
    a = StateCache(CacheConfig(memory_bytes=10 * ENTRY_BYTES, disk_dir=str(tmp_path)))
    b = StateCache(CacheConfig(memory_bytes=10 * ENTRY_BYTES, disk_dir=str(tmp_path)))
    a.put(_key(1), 1.0, 0.0, 0.0)
    b.put(_key(2), 2.0, 0.0, 0.0)
    a.flush()
    b.flush()
    c = StateCache(CacheConfig(memory_bytes=10 * ENTRY_BYTES, disk_dir=str(tmp_path)))
    assert c.get(_key(1)) == (1.0, 0.0, 0.0) and c.get(_key(2)) == (2.0, 0.0, 0.0)


def test_a_foreign_shard_is_a_miss_and_is_left_alone(tmp_path):
    path = tmp_path / shard_name("prod", "eng")
    np.savez(path, keys=np.zeros((1, 5), dtype=np.int64), values=np.ones((1, 3)),
             library_version=np.array("0.0.0"), engine_fingerprint=np.array("eng"),
             product_fingerprint=np.array("prod"))
    before = path.read_bytes()
    cache = StateCache(CacheConfig(memory_bytes=10 * ENTRY_BYTES, disk_dir=str(tmp_path)))
    assert cache.get(StateKey("prod", 0, False, 0, 0, 0, "eng")) is None
    cache.put(StateKey("prod", 0, False, 0, 0, 0, "eng"), 5.0, 0.0, 0.0)
    cache.flush()
    assert path.read_bytes() == before
    assert cache.stats().disk["foreign_shards"] == 1


def test_a_memory_only_cache_has_no_disk_stats():
    cache = StateCache(CacheConfig(memory_bytes=10 * ENTRY_BYTES))
    assert cache.stats().disk is None
    assert cache.flush() == 0
