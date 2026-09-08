"""In-memory state cache for priced states (spec 7.4, memory tier).

The key holds nothing about the hedge, the cost model or the strategy, so
two cells that differ only there price the same states once between them.
The on-disk tier arrives with the approximate providers.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

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
    """Counters for the manifest."""

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
        """The stored triple, marking the entry most recent; ``None`` on a miss."""
        value = self._store.get(key)
        if value is None:
            self._misses += 1
            return None
        self._store.move_to_end(key)
        self._hits += 1
        return value

    def put(self, key: StateKey, pv: float, delta: float, gamma: float) -> None:
        """Store a triple, evicting the least recently used past the budget."""
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
        """Drop every entry and reset the counters."""
        self._store.clear()
        self._hits = self._misses = self._evictions = 0
