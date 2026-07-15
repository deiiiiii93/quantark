"""Session-owned prepared-artifact cache (spec section 10.1).

Keys are canonical value fingerprints inside a full ``ArtifactDescriptor``;
a hit verifies complete descriptor equality (hash-collision defense — the
descriptor IS the dict key). Miss construction is single-flight with the
leader-election contract from the hardened spec: leader failure publishes to
current waiters, cleans up atomically, and the next requester re-elects.

Byte accounting: the leader RESERVES the estimated bytes before invoking the
builder, evicting unpinned LRU entries as needed; on publish the reservation
becomes the entry's charge. If no reservation is obtainable the artifact is
built and returned WITHOUT cache admission (bypass) — correctness never
depends on caching. A close() that races a running build discards the value:
the leader fails with ``PreparationError`` and no bytes outlive the session.
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
                # close() raced the build: never publish or retain bytes
                # past session lifetime.
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
