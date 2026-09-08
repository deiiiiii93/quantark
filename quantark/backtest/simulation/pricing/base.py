"""The pricer interface and the state identity it caches on (spec 7.1, 7.4).

Every key here is derived with ``blake2b`` rather than Python's builtin
``hash``: string and tuple hashing is salted per process, and a cache or an
MC seed that moved with ``PYTHONHASHSEED`` would make a run unreproducible
across machines.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional, Protocol, Tuple

import numpy as np


def _normalised(values: np.ndarray) -> np.ndarray:
    """float64 with ``-0.0`` folded onto ``0.0`` so the two key together."""
    out = np.ascontiguousarray(values, dtype=np.float64).copy()
    out[out == 0.0] = 0.0
    return out


def float_key(values: np.ndarray) -> np.ndarray:
    """The float64 bit pattern of each value as an int64 key (exact bucketing)."""
    return _normalised(np.asarray(values)).view(np.int64)


def bucket_key(values: np.ndarray, step: Optional[float]) -> np.ndarray:
    """Integer bucket per value: ``round(v / step)``, or the float bits when the step is None/0.

    ``np.round`` is half-to-even on every platform, so a value on a bucket
    edge lands in the same bucket on every machine.
    """
    arr = np.asarray(values, dtype=np.float64)
    if step is None or float(step) == 0.0:
        return float_key(arr)
    return np.round(arr / float(step)).astype(np.int64)


def bucket_centre(values: np.ndarray, step: Optional[float]) -> np.ndarray:
    """The bucket centre each value prices at; the values themselves when the step is None/0."""
    if step is None or float(step) == 0.0:
        return values
    return np.round(np.asarray(values, dtype=np.float64) / float(step)) * float(step)


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
    vol: the rate, the spot and the day's carry curve, which between them
    fix the dividend object AND the basis yield (spot is in it because the
    replay's basis arithmetic is not spot-free at the last ulp).  Keying on
    those inputs rather than on the objects keeps the cache exact without
    asking a dividend curve to hash itself.
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
        """blake2b of a canonical byte encoding of the key."""
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
        ...

    def verify(self, states: DayStates, gate: Any) -> GateReport:
        """Measure this provider's gap against direct repricing."""
        ...

    def fingerprint(self) -> str:
        """Identity of the engine and settings behind the prices."""
        ...

    def stats(self) -> Dict[str, Any]:
        """Counters worth putting in the run manifest."""
        ...
