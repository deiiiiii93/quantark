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
