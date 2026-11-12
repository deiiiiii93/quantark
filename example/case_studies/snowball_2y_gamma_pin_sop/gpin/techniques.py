"""The technique interface and the two baselines (spec C6, D9). Seller view, hedge terms.

A technique reads a ``DayContext`` (only data dated on or before the decision day) and returns a ``Target``: the
futures hedge it wants for the note (``note_delta``, CNY per index point, positive = long IM) and, for the overlay
families, the option legs it wants. It never subtracts the overlay's delta and never sizes IM lots: the ledger does
both (``gpin.ledger``, spec C7). Techniques never call the engine; they read the recorded ladders.

This module holds the interface and ``base`` / ``base_noclip``. The other techniques and the registry are added
to it later and need no change to anything defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

from quantark.util.exceptions import ValidationError

from . import LADDER, load_prereg
from .board import DayBoard

#: ``(series, "C" | "P", strike)``: one listed MO option.
LegKey = Tuple[str, str, float]

#: Model id of the unmodified note in a day record's ``states``.
TRUE = "true"


@dataclass(frozen=True)
class Target:
    """What a technique wants on one day.

    ``note_delta``: the futures hedge for the note if no overlay is held (CNY per index point, positive = long IM).
    ``legs``: desired option positions ``{(series, cp, strike): lots}``, signed lots per 10 mio of notional, positive =
    long the option. ``note_delta_if_filled``: the note-side hedge for the filled fraction of the overlay (default
    ``note_delta``). The ledger blends the two with the fill ratio and subtracts the held legs' delta.
    """

    note_delta: float
    legs: Mapping[LegKey, float] = field(default_factory=dict)
    note_delta_if_filled: Optional[float] = None


def _bump_offset(bump: float) -> int:
    step = LADDER[1] - LADDER[0]
    k = int(round(bump / step))
    centre = len(LADDER) // 2
    if k < 1 or centre - k < 0 or centre + k >= len(LADDER) or abs(k * step - bump) > 1e-9 * max(1.0, bump):
        raise ValidationError(f"bump {bump} is not a whole number of ladder steps inside the ladder")
    return k


def hedge_delta(state: Mapping[str, Any], spot: float, bump: Optional[float] = None) -> float:
    """Central bump delta at the ladder centre of one recorded model ladder, as a hedge: CNY per index point of the
    note's engine price (positive = long IM). ``bump`` defaults to ``models.bump`` of the pre-registration (1%)."""
    bump = float(load_prereg()["models"]["bump"]) if bump is None else float(bump)
    k = _bump_offset(bump)
    c = len(LADDER) // 2
    price = state["engine_price"]
    up, dn = price[c + k], price[c - k]
    if up is None or dn is None:
        raise ValidationError("a ladder price needed for the bump delta is missing")
    return float((up - dn) / (spot * (LADDER[c + k] - LADDER[c - k])))


# ---------------------------------------------------------------------------
# What a technique may see
# ---------------------------------------------------------------------------


def freeze(obj: Any) -> Any:
    """A read-only copy of a JSON-shaped value: dicts become mapping proxies, lists tuples (idempotent)."""
    if isinstance(obj, MappingProxyType):
        return obj
    if isinstance(obj, dict):
        return MappingProxyType({k: freeze(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(freeze(v) for v in obj)
    return obj


@dataclass(frozen=True)
class NoteInfo:
    """Static term information of the note (known at booking, never changes). ``s0`` is the booking-day spot;
    ``strike``, ``ko_level`` and ``ki_level`` are the book's levels in index points as the pricing pass wrote them to
    the state file (``ki_pct`` / ``ko_pct`` are multiples of the STRIKE, which is the booking spot only for the 2Y
    product: a seasoned note's strike is not its booking spot)."""

    book: str
    inception: date
    coupon: float
    ki_pct: float
    ko_pct: float
    s0: float
    strike: float
    ko_level: float
    ki_level: float


@dataclass(frozen=True)
class DayContext:
    """Everything a technique may read on decision day ``day``: the day's recorded state of every model, its board
    and flat vol, the static terms, the positions held at the start of the decision (IM lots and option legs in lots
    per 10 mio) and the previous day's context. Nothing dated after ``day`` is reachable from here: ``build`` checks
    that the record, the board and the previous context are the day's and earlier, and all mappings are read-only."""

    day: date
    day_index: int
    spot: float
    vol: float
    record: Mapping[str, Any]
    board: DayBoard
    note: NoteInfo
    held_im: Mapping[str, float]
    held_legs: Mapping[LegKey, float]
    prev: Optional["DayContext"] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.record["date"] != self.day.isoformat():
            raise ValidationError(f"record {self.record['date']} is not the decision day {self.day}")
        if self.board.day != self.day:
            raise ValidationError(f"board {self.board.day} is not the decision day {self.day}")
        if self.prev is not None and not self.prev.day < self.day:
            raise ValidationError(f"previous context {self.prev.day} is not before {self.day}")

    @classmethod
    def build(cls, day: date, day_index: int, spot: float, vol: float, record: Mapping[str, Any], board: DayBoard,
              note: NoteInfo, held_im: Mapping[str, float], held_legs: Mapping[LegKey, float],
              prev: Optional["DayContext"]) -> "DayContext":
        """The context with read-only copies of the record and the positions."""
        return cls(day, day_index, float(spot), float(vol), freeze(record), board, note,
                   MappingProxyType(dict(held_im)), MappingProxyType(dict(held_legs)), prev)

    @property
    def alive(self) -> bool:
        return bool(self.record["alive"])

    @property
    def knocked_in(self) -> bool:
        return bool(self.record["knocked_in"])

    @property
    def gap_ko(self) -> Optional[float]:
        return self.record["gap_ko"]

    @property
    def ko_dir(self) -> Optional[int]:
        return self.record["ko_dir"]

    @property
    def days_to_ko(self) -> Optional[int]:
        return self.record["days_to_ko"]

    @property
    def next_ko(self) -> Optional[date]:
        raw = self.record["next_ko"]
        return None if raw is None else date.fromisoformat(raw)

    @property
    def states(self) -> Mapping[str, Any]:
        """``{model id: {engine_price, engine_delta, engine_gamma}}`` of the day (models outside their window absent)."""
        return self.record["states"]

    def state(self, model: str) -> Optional[Mapping[str, Any]]:
        """One model's recorded ladder, ``None`` when the model has no state on this day."""
        return self.record["states"].get(model)


# ---------------------------------------------------------------------------
# Techniques
# ---------------------------------------------------------------------------


class Technique:
    """A technique: ``id``, ``family``, ``params`` (its grid point) and ``target(ctx)``. ``clip``: whether the ledger
    applies the D9 clip on the hedge ratio (only the ``base_noclip`` control switches it off)."""

    clip: bool = True

    def __init__(self, id: str, family: str, params: Optional[Mapping[str, Any]] = None) -> None:
        self.id = id
        self.family = family
        self.params: Dict[str, Any] = dict(params or {})

    def target(self, ctx: DayContext) -> Target:  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.id!r}, params={self.params!r})"


class Base(Technique):
    """The baseline: the central 1% bump delta of the ``true`` ladder every day."""

    def __init__(self) -> None:
        super().__init__("base", "baseline")

    def target(self, ctx: DayContext) -> Target:
        state = ctx.state(TRUE)
        if state is None:
            raise ValidationError(f"{ctx.day}: no recorded state of the true model")
        return Target(hedge_delta(state, ctx.spot))


class BaseNoClip(Base):
    """The control: ``base`` with the ledger's D9 clip switched off."""

    clip = False

    def __init__(self) -> None:
        Technique.__init__(self, "base_noclip", "control")
