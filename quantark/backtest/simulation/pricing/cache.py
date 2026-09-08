"""State cache for priced states: a memory tier and an on-disk tier (spec 7.4).

The key holds nothing about the hedge, the cost model or the strategy, so
two cells that differ only there price the same states once between them.
The disk tier is what lets a second cell -- or a second process -- reuse
the first one's states.
"""
from __future__ import annotations

import fcntl
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

import quantark
from quantark.util.exceptions import ValidationError

from ..config import CacheConfig
from .base import StateKey

#: Charged per entry: three float64 values plus the key and dict overhead.
#: A fixed cost makes the budget a predictable entry count rather than a
#: guess at CPython's per-object footprint.
ENTRY_BYTES = 88

LIBRARY_VERSION = str(quantark.__version__)
_KEY_FIELDS = ("day_index", "knocked_in", "spot_key", "vol_key", "env_key")


def shard_name(product_fp: str, engine_fp: str) -> str:
    """The shard file for one product and engine at this library version."""
    return f"{product_fp}-{engine_fp}-{LIBRARY_VERSION.replace('.', '_')}.npz"


def _key_row(key: StateKey) -> Tuple[int, int, int, int, int]:
    return (int(key.day_index), int(key.knocked_in), int(key.spot_key), int(key.vol_key), int(key.env_key))


@dataclass(frozen=True)
class CacheStats:
    """Counters for the manifest."""

    hits: int
    misses: int
    evictions: int
    entries: int
    bytes_used: int
    disk: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits, "misses": self.misses, "evictions": self.evictions,
            "entries": self.entries, "bytes_used": self.bytes_used, "disk": self.disk,
        }


Table = Dict[tuple, Tuple[float, float, float]]


class DiskTier:
    """Append-only npz shards, one per (product, engine, library version).

    A shard is read once, lazily, on the first miss for its pair.  Its
    headers must match its name; otherwise it is foreign -- a miss, never
    reinterpreted, never overwritten.  ``flush`` merges under an advisory
    lock so concurrent writers (batch workers) cannot lose each other's
    entries, and replaces the file atomically.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._loaded: Dict[Tuple[str, str], Table] = {}
        self._foreign: set = set()
        self._pending: Dict[Tuple[str, str], Table] = {}
        self._hits = self._misses = self._loads = self._flushed = 0

    @staticmethod
    def _pair(key: StateKey) -> Tuple[str, str]:
        return (key.product_fingerprint, key.engine_fingerprint)

    def _path(self, pair: Tuple[str, str]) -> Path:
        return self.directory / shard_name(*pair)

    def _read(self, pair: Tuple[str, str]) -> Table:
        path = self._path(pair)
        if not path.exists():
            return {}
        with np.load(path, allow_pickle=False) as z:
            headers = (str(z["library_version"]), str(z["product_fingerprint"]), str(z["engine_fingerprint"]))
            if headers != (LIBRARY_VERSION, pair[0], pair[1]):
                self._foreign.add(pair)
                return {}
            keys = np.asarray(z["keys"], dtype=np.int64)
            values = np.asarray(z["values"], dtype=np.float64)
        return {tuple(int(x) for x in k): (float(v[0]), float(v[1]), float(v[2])) for k, v in zip(keys, values)}

    def _table(self, pair: Tuple[str, str]) -> Table:
        table = self._loaded.get(pair)
        if table is None:
            table = self._read(pair)
            self._loaded[pair] = table
            self._loads += 1
        return table

    def get(self, key: StateKey) -> Optional[Tuple[float, float, float]]:
        """The stored triple, or ``None``; loads the shard on first touch."""
        value = self._table(self._pair(key)).get(_key_row(key))
        if value is None:
            self._misses += 1
        else:
            self._hits += 1
        return value

    def record(self, key: StateKey, pv: float, delta: float, gamma: float) -> None:
        """Queue an entry for the next ``flush`` (and serve it from memory meanwhile)."""
        pair = self._pair(key)
        triple = (float(pv), float(delta), float(gamma))
        self._pending.setdefault(pair, {})[_key_row(key)] = triple
        self._table(pair)[_key_row(key)] = triple

    def flush(self) -> int:
        """Merge the pending entries into their shards; returns the count written."""
        written = 0
        for pair, entries in list(self._pending.items()):
            if not entries or pair in self._foreign:
                continue                      # never overwrite a shard we could not read
            path = self._path(pair)
            lock = path.with_suffix(".lock")
            with open(lock, "w") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    merged = self._read(pair)
                    if pair in self._foreign:
                        continue
                    merged.update(entries)
                    keys = np.array(list(merged.keys()), dtype=np.int64).reshape(-1, 5)
                    values = np.array(list(merged.values()), dtype=np.float64).reshape(-1, 3)
                    fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".npz.tmp")
                    with os.fdopen(fd, "wb") as fh:       # a file handle: savez writes exactly here
                        np.savez(fh, keys=keys, values=values,
                                 library_version=np.array(LIBRARY_VERSION),
                                 product_fingerprint=np.array(pair[0]), engine_fingerprint=np.array(pair[1]))
                    os.replace(tmp, path)
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
            written += len(entries)
            self._pending[pair] = {}
        self._flushed += written
        return written

    def stats(self) -> Dict[str, Any]:
        return {
            "disk_hits": self._hits, "disk_misses": self._misses, "shards_loaded": self._loads,
            "foreign_shards": len(self._foreign), "flushed": self._flushed,
            "pending": sum(len(v) for v in self._pending.values()),
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
        self._disk = DiskTier(Path(config.disk_dir)) if config.disk_dir is not None else None

    def get(self, key: StateKey) -> Optional[Tuple[float, float, float]]:
        """The stored triple, marking the entry most recent; ``None`` on a miss.

        A memory miss consults the disk tier and promotes what it finds.
        """
        value = self._store.get(key)
        if value is None:
            if self._disk is not None:
                value = self._disk.get(key)
                if value is not None:
                    self._store[key] = value
                    self._evict()
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
        if self._disk is not None:
            self._disk.record(key, pv, delta, gamma)
        self._evict()

    def _evict(self) -> None:
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)
            self._evictions += 1

    def flush(self) -> int:
        """Write pending entries to the disk tier; 0 without one."""
        return 0 if self._disk is None else self._disk.flush()

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
            disk=None if self._disk is None else self._disk.stats(),
        )

    def clear(self) -> None:
        """Drop every entry and reset the counters."""
        self._store.clear()
        self._hits = self._misses = self._evictions = 0
