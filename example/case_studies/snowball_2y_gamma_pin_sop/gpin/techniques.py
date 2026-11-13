"""The techniques (spec C6, "Technique registry"). Seller view, hedge terms.

A technique reads a ``DayContext`` (only data dated on or before the decision day) and returns a ``Target``: the
futures hedge it wants for the note (``note_delta``, CNY per index point, positive = long IM) and, for the overlay
families, the option legs it wants. It never subtracts the overlay's delta and never sizes IM lots: the ledger does
both (``gpin.ledger``, spec C7). Techniques never call the engine; they read the recorded ladders.

``REGISTRY`` holds one technique per pre-registered grid point (``results/preregistration.json``, ids ``row@values``
such as ``ko_shift@0.01`` or ``mo_ko@1,scenario``) and, once ``results/selection.json`` names their members, the two
composites. Rules common to every technique:

* outside its window or trigger, or when a model state it needs is absent that day, a technique returns the baseline
  target (there is no KI-model state on a knock-in day; ``ko_dir`` 0 or null means no shift model for the observation);
* windows are the pricing pass's (``gpin.pricing``), read here from the day record and the note's levels: the KO window
  is ``days_to_ko <= ko_days`` and ``|S / B - 1| <= ko_band``, the KI window is not knocked in and ``S / K - 1 <=
  ki_band``; the KO trigger is the KO window and ``|z| <= trigger_z`` with ``z = ln(S / B) / (vol / sqrt(244) x
  sqrt(max(n, 1)))``;
* "do not trade IM" (``freeze``, ``band``) is expressed as the hedge currently carried: the IM lots held at the
  board's closes plus the delta of the legs held (``carried_delta``), which the ledger sizes back to the lots held;
* an overlay that cannot be built whole that day (no series, a strike not listed or a barrier more than one listed
  step beyond the listing, a leg without an implied vol, a leg that would have to be opened or increased and fails
  the D8 screen) is refused: the target keeps the legs held
  and carries a flag (``overlay_no_series``, ``overlay_no_strike``, ``overlay_no_vol``, ``overlay_not_openable``)
  that the ledger writes to the row.

Ledger-side code: no engine, no market data of its own; the only file read is the selection of the composites.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close, is_zero, safe_log, safe_sqrt

from s2ts import ACT, IM_MULT, MO_MULT, NOTIONAL
from s2ts.chain import sse_trading_days

from . import LADDER, RESULTS_DIR, StudyDataError, load_prereg
from . import black
from .board import DayBoard, OptionQuote, SeriesInfo

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
    ``note_delta``). The ledger blends the two with the fill ratio and subtracts the held legs' delta. ``flags``: what
    the technique refused that day (an overlay it could not build); the ledger writes them to the day's row.
    """

    note_delta: float
    legs: Mapping[LegKey, float] = field(default_factory=dict)
    note_delta_if_filled: Optional[float] = None
    flags: Tuple[str, ...] = ()


def _bump_offset(bump: float) -> int:
    step = LADDER[1] - LADDER[0]
    k = int(round(bump / step))
    centre = len(LADDER) // 2
    if k < 1 or centre - k < 0 or centre + k >= len(LADDER) or not is_close(k * step, bump):
        raise ValidationError(f"bump {bump} is not a whole number of ladder steps inside the ladder")
    return k


@lru_cache(maxsize=1)
def _default_bump() -> float:
    """``models.bump`` of the pre-registration (read once: the ledger asks for it on every day of every run)."""
    return float(load_prereg()["models"]["bump"])


def hedge_delta(state: Mapping[str, Any], spot: float, bump: Optional[float] = None) -> float:
    """Central bump delta at the ladder centre of one recorded model ladder, as a hedge: CNY per index point of the
    note's engine price (positive = long IM). ``bump`` defaults to ``models.bump`` of the pre-registration (1%)."""
    bump = _default_bump() if bump is None else float(bump)
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
    """A technique: ``id``, ``row`` (its row of the pre-registration), ``family``, ``params`` (its grid point) and
    ``target(ctx)``. ``clip``: whether the ledger applies the D9 clip on the hedge ratio (only the ``base_noclip``
    control switches it off). ``models``: the hedge models whose recorded states it reads beside ``true``."""

    clip: bool = True
    models: Tuple[str, ...] = ()

    def __init__(self, id: str, family: str, params: Optional[Mapping[str, Any]] = None,
                 row: Optional[str] = None) -> None:
        self.id = id
        self.family = family
        self.params: Dict[str, Any] = dict(params or {})
        self.row = id.split("@")[0] if row is None else row

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


# ---------------------------------------------------------------------------
# Pre-registered numbers, windows and triggers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """The pre-registered numbers the techniques use (``windows``, ``models``, ``overlay``, ``tradability``)."""

    ko_days: int
    ko_band: float
    ki_band: float
    trigger_z: float
    days_per_year: int
    overlay_ko_days: int
    overlay_ki_min_days_left: int
    bump: float
    scenario_cap: float
    min_volume: int
    min_open_interest: int
    ko_shift_sizes: Tuple[float, ...]
    ko_wrong_size: float
    ki_shifts: Tuple[float, ...]
    ki_wrong_shift: float

    @classmethod
    def from_prereg(cls, prereg: Mapping[str, Any]) -> "Settings":
        """Read from a loaded pre-registration."""
        w, m, t = prereg["windows"], prereg["models"], prereg["tradability"]
        return cls(int(w["ko_days"]), float(w["ko_band"]), float(w["ki_band"]), float(w["trigger_z"]),
                   int(w["days_per_year"]), int(w["overlay_ko_days"]), int(w["overlay_ki_min_days_left"]),
                   float(m["bump"]), float(prereg["overlay"]["scenario_cap_x_payoff"]), int(t["min_volume"]),
                   int(t["min_open_interest"]), tuple(float(x) for x in m["ko_shift_sizes"]),
                   float(m["ko_wrong_size"]), tuple(float(x) for x in m["ki_shifts"]), float(m["ki_wrong_shift"]))


def _at_most(value: float, bound: float) -> bool:
    """``value <= bound`` with the repo's tolerance (the pricing pass's comparison for its windows)."""
    return value < bound or is_close(value, bound)


def in_ko_window(ctx: DayContext, s: Settings) -> bool:
    """The pricing pass's KO window: the next KO observation is at most ``ko_days`` trading days ahead and ``|S / B -
    1| <= ko_band``, ``B`` the note's KO level."""
    n = ctx.days_to_ko
    if n is None or n > s.ko_days:
        return False
    return _at_most(abs(ctx.spot / ctx.note.ko_level - 1.0), s.ko_band)


def in_ki_window(ctx: DayContext, s: Settings) -> bool:
    """The pricing pass's KI window: not knocked in and ``S / K - 1 <= ki_band``, ``K`` the note's KI level."""
    return (not ctx.knocked_in) and _at_most(ctx.spot / ctx.note.ki_level - 1.0, s.ki_band)


def daily_sd(ctx: DayContext, s: Settings) -> float:
    """One daily standard deviation: the day's flat vol over ``sqrt(days_per_year)``; fails closed without a vol."""
    if not ctx.vol > 0.0 or is_zero(ctx.vol):
        raise ValidationError(f"{ctx.day}: the day's flat vol {ctx.vol!r} is not positive")
    return float(ctx.vol / safe_sqrt(float(s.days_per_year)))


def ko_z(ctx: DayContext, s: Settings) -> float:
    """``z = ln(S / B) / (sd x sqrt(max(n, 1)))``: the distance to the KO barrier in standard deviations of the move
    left until the observation."""
    n = ctx.days_to_ko
    if n is None:
        raise ValidationError(f"{ctx.day}: no KO observation ahead, z is undefined")
    return float(safe_log(ctx.spot / ctx.note.ko_level) / (daily_sd(ctx, s) * safe_sqrt(float(max(n, 1)))))


def ko_trigger(ctx: DayContext, s: Settings) -> bool:
    """The KO trigger: inside the KO window and ``|z| <= trigger_z``."""
    return in_ko_window(ctx, s) and _at_most(abs(ko_z(ctx, s)), s.trigger_z)


def base_delta(ctx: DayContext, s: Settings) -> float:
    """``d(true)``: the baseline hedge of the day."""
    state = ctx.state(TRUE)
    if state is None:
        raise ValidationError(f"{ctx.day}: no recorded state of the true model")
    return hedge_delta(state, ctx.spot, s.bump)


def model_delta(ctx: DayContext, model: str, s: Settings) -> Optional[float]:
    """``d(model)``, ``None`` when the model has no state that day."""
    state = ctx.state(model)
    return None if state is None else hedge_delta(state, ctx.spot, s.bump)


def _leg_str(key: LegKey) -> str:
    return f"{key[0]}:{key[1]}:{key[2]:g}"


def carried_delta(ctx: DayContext) -> float:
    """The hedge the seller carries into the decision, as a note-side delta (CNY per index point): the IM lots held at
    the board's closes, ``lots x 200 x close / S``, plus the Black-76 index delta of the option legs held (which the
    ledger subtracts again). A technique that returns it as ``note_delta`` makes the ledger trade no IM while the legs
    do not change. Fails closed when the board cannot mark a position held."""
    total = 0.0
    for code, lots in ctx.held_im.items():
        bar = ctx.board.im.get(code)
        if bar is None or bar.close is None:
            raise StudyDataError(f"{ctx.day}: no IM close on the board for held contract {code}")
        total += lots * IM_MULT * float(bar.close)
    total /= ctx.spot
    for key, lots in ctx.held_legs.items():
        quote = ctx.board.mo.get(key[0], {}).get((key[1], key[2]))
        info = ctx.board.series.get(key[0])
        greeks = None if quote is None or info is None else black.greeks_at_settle(
            quote.settle, key[1], key[2], info, ctx.spot)
        if greeks is None:
            raise StudyDataError(f"{ctx.day}: held leg {_leg_str(key)} has no settlement or no delta on the board")
        total += lots * MO_MULT * greeks[0]
    return float(total)


# ---------------------------------------------------------------------------
# The trading calendar (known in advance: no look-ahead)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _year_days(year: int) -> Tuple[date, ...]:
    return tuple(d.date() for d in sse_trading_days(year, year))


def trading_days_between(day: date, until: date) -> int:
    """SSE trading days in ``(day, until]``."""
    return sum(bisect_right(_year_days(year), until) - bisect_right(_year_days(year), day)
               for year in range(day.year, until.year + 1))


def next_trading_day(day: date) -> date:
    """The first SSE trading day after ``day``."""
    for year in (day.year, day.year + 1):
        for d in _year_days(year):
            if d > day:
                return d
    raise StudyDataError(f"no trading day after {day} on the calendar")


# ---------------------------------------------------------------------------
# Futures-side rules: one delta, or the baseline
# ---------------------------------------------------------------------------


def grid_id(row: str, point: Mapping[str, Any]) -> str:
    """The registry id of one grid point: the row id, then ``@`` and the grid values in the grid's key order
    (``ko_bend@0.015,3``); the bare row id for a row without parameters."""
    if not point:
        return row
    return f"{row}@{','.join(str(v) for v in point.values())}"


def _model_tag(value: float) -> str:
    """Basis points of ``value`` as a model id suffix (``gpin.models``' rule): 0.005 -> ``050``, 0.03 -> ``300``."""
    return f"{int(round(abs(value) * 1.0e4)):03d}"


def _priced(value: float, priced: Sequence[float], what: str) -> float:
    for x in priced:
        if is_close(float(x), float(value)):
            return float(x)
    raise ValidationError(f"{what}: no hedge model is priced for {value} (priced: {list(priced)})")


class _Rule(Technique):
    """A futures-side rule: ``note_delta`` is ``rule(ctx, base)`` or, when that is ``None``, the baseline."""

    def __init__(self, row: Mapping[str, Any], params: Mapping[str, Any], settings: Settings) -> None:
        super().__init__(grid_id(row["id"], params), str(row["family"]), params, row=str(row["id"]))
        self.s = settings

    def rule(self, ctx: DayContext, base: float) -> Optional[float]:  # pragma: no cover - abstract
        raise NotImplementedError

    def target(self, ctx: DayContext) -> Target:
        base = base_delta(ctx, self.s)
        delta = self.rule(ctx, base)
        return Target(base if delta is None else float(delta))


class KoModels(_Rule):
    """``ko_shift``, ``ko_spread``, ``ko_bend``, ``ko_wrong``: the mean of ``d`` over ``models`` in the KO window (one
    shift model; or the four equally spaced shifts 0 .. e of a spread, shift 0 being ``true``), with ``n_b``: only
    on the last ``n_b`` closes before the observation and under the KO trigger. No shift model exists when the
    observation's ``ko_dir`` is 0 or null."""

    def __init__(self, row: Mapping[str, Any], params: Mapping[str, Any], settings: Settings, models: Sequence[str],
                 n_b: Optional[int] = None) -> None:
        super().__init__(row, params, settings)
        self.models = tuple(models)
        self.n_b = n_b

    def rule(self, ctx: DayContext, base: float) -> Optional[float]:
        if not in_ko_window(ctx, self.s) or ctx.ko_dir not in (-1, 1):
            return None
        if self.n_b is not None and not (ctx.days_to_ko <= self.n_b and ko_trigger(ctx, self.s)):
            return None
        deltas = [base if m == TRUE else model_delta(ctx, m, self.s) for m in self.models]
        if any(d is None for d in deltas):
            return None
        return sum(deltas) / len(deltas)


class KiModel(_Rule):
    """``ki_shift``, ``ki_wrong``: ``d(model)`` in the KI window."""

    def __init__(self, row: Mapping[str, Any], params: Mapping[str, Any], settings: Settings, model: str) -> None:
        super().__init__(row, params, settings)
        self.models = (model,)

    def rule(self, ctx: DayContext, base: float) -> Optional[float]:
        return model_delta(ctx, self.models[0], self.s) if in_ki_window(ctx, self.s) else None


class Secant(_Rule):
    """``secant``: ``(V(S(1+h)) - V(S(1-h))) / (2hS)`` on the ``true`` ladder inside the KO or the KI window."""

    def __init__(self, row: Mapping[str, Any], params: Mapping[str, Any], settings: Settings) -> None:
        super().__init__(row, params, settings)
        self.h = float(params["h"])
        _bump_offset(self.h)                                 # h must be a ladder point: fails closed at registration

    def rule(self, ctx: DayContext, base: float) -> Optional[float]:
        if not (in_ko_window(ctx, self.s) or in_ki_window(ctx, self.s)):
            return None
        return hedge_delta(ctx.state(TRUE), ctx.spot, self.h)


class KiCap(_Rule):
    """``ki_cap``: ``min(d(true), c x d(kin))`` in the KI window."""

    models = ("kin",)

    def rule(self, ctx: DayContext, base: float) -> Optional[float]:
        if not in_ki_window(ctx, self.s):
            return None
        kin = model_delta(ctx, "kin", self.s)
        return None if kin is None else min(base, float(self.params["c"]) * kin)


class SurvHold(_Rule):
    """``surv_hold``: ``d(surv)`` on the last ``n_s`` closes before a KO observation under the KO trigger."""

    models = ("surv",)

    def rule(self, ctx: DayContext, base: float) -> Optional[float]:
        n = ctx.days_to_ko
        if n is None or n > int(self.params["n_s"]) or not ko_trigger(ctx, self.s):
            return None
        return model_delta(ctx, "surv", self.s)


class Derisk(_Rule):
    """``derisk``: a linear blend from ``d(true)`` at ``n = n_d`` to ``d(surv)`` at ``n = 1`` under the KO trigger."""

    models = ("surv",)

    def __init__(self, row: Mapping[str, Any], params: Mapping[str, Any], settings: Settings) -> None:
        super().__init__(row, params, settings)
        if int(params["n_d"]) < 2:
            raise ValidationError(f"{self.id}: n_d must be at least 2 for a blend down to n = 1")

    def rule(self, ctx: DayContext, base: float) -> Optional[float]:
        n, n_d = ctx.days_to_ko, int(self.params["n_d"])
        if n is None or n > n_d or not ko_trigger(ctx, self.s):
            return None
        surv = model_delta(ctx, "surv", self.s)
        if surv is None:
            return None
        weight = (n_d - max(n, 1)) / (n_d - 1.0)             # on the survived branch: 0 at n_d, 1 at n = 1
        return (1.0 - weight) * base + weight * surv


class Freeze(_Rule):
    """``freeze``: no IM trade on the last ``n_f`` closes before a KO observation under the KO trigger: the target is
    the hedge carried (``carried_delta``)."""

    def rule(self, ctx: DayContext, base: float) -> Optional[float]:
        n = ctx.days_to_ko
        if n is None or n > int(self.params["n_f"]) or not ko_trigger(ctx, self.s):
            return None
        return carried_delta(ctx)


class Band(_Rule):
    """``band`` (cost lever): outside the KO and KI windows IM is traded only when the baseline is more than ``w x``
    the notional delta (``NOTIONAL / S``) away from the hedge carried; inside a window there is no band."""

    def rule(self, ctx: DayContext, base: float) -> Optional[float]:
        if in_ko_window(ctx, self.s) or in_ki_window(ctx, self.s):
            return None
        carried = carried_delta(ctx)
        if _at_most(abs(base - carried), float(self.params["w"]) * NOTIONAL / ctx.spot):
            return carried
        return None


# ---------------------------------------------------------------------------
# MO overlays (spec "Overlay construction")
# ---------------------------------------------------------------------------

#: Flags of a refused overlay (``Target.flags``; the ledger writes them to the row). Every refusal flag starts with
#: ``REFUSAL_PREFIX``: the ledger reads a target carrying one as "the overlay was not sized today".
REFUSAL_PREFIX = "overlay_"
NO_SERIES, NO_STRIKE, NO_VOL, NOT_OPENABLE = ("overlay_no_series", "overlay_no_strike", "overlay_no_vol",
                                              "overlay_not_openable")


def is_refusal(target: Target) -> bool:
    """True when the target carries an overlay refusal flag."""
    return any(str(name).startswith(REFUSAL_PREFIX) for name in target.flags)


@dataclass(frozen=True)
class OverlayPlan:
    """What an overlay hedges on one day: the signed ``gap`` (CNY per note; positive = the seller is worse off above
    ``level`` than below it), the barrier ``level``, the ``series`` to trade (``None``: none qualifies) and the date
    ``value_date`` at which the scenario sizing values the spread (the barrier's next observation)."""

    gap: float
    level: float
    series: Optional[str]
    value_date: date


def spread_strikes(gap: float, level: float, spot: float, strikes: Sequence[float],
                   steps: int) -> Optional[Tuple[str, float, float, float]]:
    """The spread across ``level``: ``(cp, K1, K2, sign)`` with ``sign`` lots of the K1 option and ``-sign`` of the K2
    option per lot of spread, or ``None`` when the listing has no strike to start from.

    The strike step is the listed spacing at the level: the distance between the highest listed strike at or below it
    and the lowest listed strike above it. A level beyond the listing takes the spacing of the listed interval nearest
    to it (the last one below a level above every strike, the first one above a level below every strike), and only
    when it is at most ONE such step beyond the listing: further out there is no strike near the barrier and the result
    is ``None`` (a spread struck several steps away from the barrier is not the barrier's hedge).

    ``gap > 0`` (payoff rising across the level): K2 = the highest listed strike at or below the level, K1 = K2 minus
    ``steps`` steps, long K1 / short K2 in calls when spot is at or below K2, in puts when both strikes are below spot.
    ``gap < 0`` (payoff falling): K1 = the lowest listed strike at or above the level, K2 = K1 plus ``steps`` steps,
    short K1 / long K2 in calls when both strikes are at or above spot, in puts otherwise. K1 and K2 are returned as
    computed; whether both are listed is the caller's check."""
    listed = sorted(float(k) for k in strikes)
    below = [k for k in listed if _at_most(k, level)]
    above = [k for k in listed if not _at_most(k, level)]
    if below and above:
        step = above[0] - below[-1]
    elif len(below) >= 2:
        step = below[-1] - below[-2]
        if not _at_most(level - below[-1], step):            # more than one listed step above the listing
            return None
    elif len(above) >= 2:
        step = above[1] - above[0]
        if not _at_most(above[0] - level, step):             # more than one listed step below the listing
            return None
    else:
        return None
    if gap > 0.0:
        if not below:
            return None
        k2 = below[-1]
        k1 = k2 - steps * step
        return ("C" if _at_most(spot, k2) else "P"), k1, k2, 1.0
    if below and is_close(below[-1], level):
        k1 = below[-1]
    elif above:
        k1 = above[0]
    else:
        return None
    k2 = k1 + steps * step
    return ("C" if _at_most(spot, k1) else "P"), k1, k2, -1.0


class _Overlay(Technique):
    """An overlay row: ``note_delta = note_delta_if_filled = d(true)``; the legs of ``overlay(ctx)``."""

    def __init__(self, row: Mapping[str, Any], params: Mapping[str, Any], settings: Settings) -> None:
        super().__init__(grid_id(row["id"], params), str(row["family"]), params, row=str(row["id"]))
        self.s = settings
        self.steps = int(params["steps"])
        self.sizing = str(params["sizing"])
        if self.steps < 1 or self.sizing not in ("payoff", "scenario"):
            raise ValidationError(f"{self.id}: steps must be at least 1 and sizing 'payoff' or 'scenario'")

    def plan(self, ctx: DayContext) -> Optional[OverlayPlan]:  # pragma: no cover - abstract
        """The day's plan, ``None`` outside the trigger (the target is flat)."""
        raise NotImplementedError

    def overlay(self, ctx: DayContext) -> Tuple[Optional[Dict[LegKey, float]], Tuple[str, ...]]:
        """``(legs, flags)``: the legs wanted (``{}`` = flat), or ``None`` with the reason when the overlay is refused
        that day (the caller keeps the legs held)."""
        plan = self.plan(ctx)
        if plan is None or is_zero(plan.gap):
            return {}, ()
        if plan.series is None:
            return None, (NO_SERIES,)
        quotes: Mapping[Tuple[str, float], OptionQuote] = ctx.board.mo.get(plan.series, {})
        info: SeriesInfo = ctx.board.series[plan.series]
        strikes = sorted({k for _, k in quotes})
        found = spread_strikes(plan.gap, plan.level, ctx.spot, strikes, self.steps)
        if found is None:
            return None, (NO_STRIKE,)
        cp, k1, k2, sign = found
        listed = [next((k for k in strikes if is_close(k, target)), None) for target in (k1, k2)]
        if any(k is None or (cp, k) not in quotes for k in listed):
            return None, (NO_STRIKE,)
        k1, k2 = listed
        vols = [black.implied_vol(quotes[(cp, k)].settle, cp, k, info) for k in (k1, k2)]
        if any(v is None for v in vols):
            return None, (NO_VOL,)
        payoff_lots = abs(plan.gap) / ((k2 - k1) * MO_MULT)
        lots = payoff_lots
        if self.sizing == "scenario":
            lots = self._scenario_lots(ctx, plan, info, cp, (k1, k2), vols, sign, payoff_lots)
        legs = {(plan.series, cp, k1): sign * lots, (plan.series, cp, k2): -sign * lots}
        for key, want in legs.items():
            held = ctx.held_legs.get(key, 0.0)
            reduces = held * want > 0.0 and _at_most(abs(want), abs(held))
            quote = quotes[(key[1], key[2])]
            if not reduces and (quote.volume < self.s.min_volume or quote.open_interest is None
                                or quote.open_interest < self.s.min_open_interest):
                return None, (NOT_OPENABLE,)
        return legs, ()

    def _scenario_lots(self, ctx: DayContext, plan: OverlayPlan, info: SeriesInfo, cp: str,
                       strikes: Tuple[float, float], vols: Sequence[float], sign: float, payoff_lots: float) -> float:
        """Lots for which the spread's Black-76 value on ``plan.value_date`` (the series' time left then, each leg's
        implied vol of today, the forward moved in proportion to spot) differs by ``|gap|`` between one daily sigma
        above and below the level; at most ``scenario_cap`` times the payoff lots."""
        left = max((info.last_trading_day - plan.value_date).days, 0) / ACT
        sd = daily_sd(ctx, self.s)

        def value(spot_scenario: float) -> float:
            return sum(w * black.value_at(vol, cp, k, info, ctx.spot, spot_scenario, left)
                       for w, k, vol in zip((sign, -sign), strikes, vols))

        move = abs(value(plan.level * (1.0 + sd)) - value(plan.level * (1.0 - sd)))
        cap = self.s.scenario_cap * payoff_lots
        if move * MO_MULT * cap <= abs(plan.gap):            # the uncapped count is at or beyond the cap
            return cap
        return abs(plan.gap) / (move * MO_MULT)

    def target(self, ctx: DayContext) -> Target:
        base = base_delta(ctx, self.s)
        legs, flags = self.overlay(ctx)
        if legs is None:
            legs = dict(ctx.held_legs)
        return Target(base, legs, base if legs else None, flags)


def _series_by_end(board: DayBoard) -> List[Tuple[str, SeriesInfo]]:
    """The board's MO series, the one ending first first."""
    return sorted(board.series.items(), key=lambda kv: (kv[1].last_trading_day, kv[0]))


class MoKo(_Overlay):
    """``mo_ko``: the spread across the KO barrier sized to the day's ``gap_ko``, on the last ``overlay_ko_days``
    closes before the observation under the KO trigger; the series is the first whose last trading day is on or after
    the observation. Flat from the observation day's decision on (the next observation is then weeks away)."""

    def plan(self, ctx: DayContext) -> Optional[OverlayPlan]:
        n, gap, obs = ctx.days_to_ko, ctx.gap_ko, ctx.next_ko
        if n is None or obs is None or gap is None or n > self.s.overlay_ko_days or not ko_trigger(ctx, self.s):
            return None
        series = [name for name, info in _series_by_end(ctx.board) if info.last_trading_day >= obs]
        return OverlayPlan(float(gap), ctx.note.ko_level, series[0] if series else None, obs)


def ki_gap(ctx: DayContext) -> Optional[float]:
    """The knock-in gap in engine terms (CNY per note): the ``ki_m200`` engine price minus the ``kin`` engine price,
    each interpolated linearly on its ladder at the KI level; what the seller gains when the note knocks in. ``None``
    when a state is missing; fails closed when the KI level is off the ladder."""
    upper, kin = ctx.state("ki_m200"), ctx.state("kin")
    if upper is None or kin is None:
        return None
    spots = ctx.spot * (1.0 + np.asarray(LADDER, dtype=float))
    level = ctx.note.ki_level
    if not (spots[0] <= level <= spots[-1]):
        raise StudyDataError(f"{ctx.day}: the KI level {level} is off the ladder {spots[0]}..{spots[-1]}")
    return float(np.interp(level, spots, np.asarray(upper["engine_price"], dtype=float))
                 - np.interp(level, spots, np.asarray(kin["engine_price"], dtype=float)))


class MoKi(_Overlay):
    """``mo_ki``: the same spread across the KI barrier, sized to ``ki_gap``, on every KI-window day; the front series
    with at least ``overlay_ki_min_days_left`` trading days left (rolled when it has fewer). The scenario sizing
    values the spread at the next daily knock-in observation, the next trading day. Flat on a knock-in and on
    leaving the window."""

    models = ("ki_m200", "kin")

    def plan(self, ctx: DayContext) -> Optional[OverlayPlan]:
        if not in_ki_window(ctx, self.s):
            return None
        gap = ki_gap(ctx)
        if gap is None:
            return None
        series = [name for name, info in _series_by_end(ctx.board)
                  if trading_days_between(ctx.day, info.last_trading_day) >= self.s.overlay_ki_min_days_left]
        return OverlayPlan(gap, ctx.note.ki_level, series[0] if series else None, next_trading_day(ctx.day))


# ---------------------------------------------------------------------------
# Composites (members named by the training selection)
# ---------------------------------------------------------------------------


class SopFutures(Technique):
    """``sop_futures``: the KO member's target inside the KO window, the KI member's inside the KI window, the
    baseline elsewhere; when both windows are active the KO rule applies."""

    def __init__(self, row: Mapping[str, Any], ko: Technique, ki: Technique, settings: Settings) -> None:
        super().__init__(str(row["id"]), str(row["family"]), {"ko": ko.id, "ki": ki.id})
        self.ko, self.ki, self.s = ko, ki, settings
        self.models = tuple(dict.fromkeys(ko.models + ki.models))

    def target(self, ctx: DayContext) -> Target:
        if in_ko_window(ctx, self.s):
            return self.ko.target(ctx)
        if in_ki_window(ctx, self.s):
            return self.ki.target(ctx)
        return Target(base_delta(ctx, self.s))


class SopOverlay(Technique):
    """``sop_overlay``: ``note_delta`` is ``sop_futures``' delta, the legs are the two overlay members' and
    ``note_delta_if_filled = d(true)``: the covered fraction of the digital is delta-neutral against the overlay, the
    uncovered fraction follows the futures rule. With no leg wanted nothing is covered and the futures rule is the
    whole target. When a member refuses its overlay nothing new is traded that day and every leg held stays."""

    def __init__(self, row: Mapping[str, Any], futures: Technique, ko: _Overlay, ki: _Overlay,
                 settings: Settings) -> None:
        super().__init__(str(row["id"]), str(row["family"]), {"futures": futures.id, "ko": ko.id, "ki": ki.id})
        self.futures, self.ko, self.ki, self.s = futures, ko, ki, settings
        self.models = tuple(dict.fromkeys(futures.models + ko.models + ki.models))

    def target(self, ctx: DayContext) -> Target:
        futures = self.futures.target(ctx)
        base = base_delta(ctx, self.s)
        parts = [self.ko.overlay(ctx), self.ki.overlay(ctx)]
        flags = tuple(futures.flags) + tuple(f for _, fl in parts for f in fl)
        if any(legs is None for legs, _ in parts):
            legs: Dict[LegKey, float] = dict(ctx.held_legs)
        else:
            legs = {}
            for part, _ in parts:
                for key, lots in part.items():
                    legs[key] = legs.get(key, 0.0) + lots
            legs = {k: v for k, v in legs.items() if not is_zero(v)}
        return Target(futures.note_delta, legs, base if legs else None, flags)


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

#: The training selection (written by ``04_ledger.py --slice train``); its ``composites`` block names the members.
SELECTION_PATH = RESULTS_DIR / "selection.json"


def _ko_model(size: float, s: Settings, what: str) -> str:
    return f"ko_s{_model_tag(_priced(size, s.ko_shift_sizes, what))}"


def _spread_models(e: float, s: Settings, what: str) -> List[str]:
    """``true`` and the three shift models of the four equally spaced shifts 0, e/3, 2e/3, e."""
    return [TRUE] + [_ko_model(e * k / 3.0, s, what) for k in (1, 2, 3)]


def _row_factories() -> Dict[str, Callable[[Mapping[str, Any], Dict[str, Any], Settings], Technique]]:
    def what(row, p):
        return grid_id(row["id"], p)

    return {
        "base": lambda row, p, s: Base(),
        "base_noclip": lambda row, p, s: BaseNoClip(),
        "ko_shift": lambda row, p, s: KoModels(row, p, s, [_ko_model(p["size"], s, what(row, p))]),
        "ko_spread": lambda row, p, s: KoModels(row, p, s, _spread_models(p["e"], s, what(row, p))),
        "ko_bend": lambda row, p, s: KoModels(row, p, s, _spread_models(p["e"], s, what(row, p)), n_b=int(p["n_b"])),
        "ko_wrong": lambda row, p, s: KoModels(row, p, s, [f"ko_w{_model_tag(s.ko_wrong_size)}"]),
        "ki_shift": lambda row, p, s: KiModel(
            row, p, s, f"ki_m{_model_tag(_priced(p['x'], s.ki_shifts, what(row, p)))}"),
        "ki_wrong": lambda row, p, s: KiModel(row, p, s, f"ki_p{_model_tag(s.ki_wrong_shift)}"),
        "secant": Secant,
        "ki_cap": KiCap,
        "surv_hold": SurvHold,
        "derisk": Derisk,
        "freeze": Freeze,
        "mo_ko": MoKo,
        "mo_ki": MoKi,
        "band": Band,
    }


#: Composite rows and the slots a selection must fill for each.
_COMPOSITE_SLOTS: Dict[str, Tuple[str, ...]] = {"sop_futures": ("ko", "ki"), "sop_overlay": ("ko", "ki")}


def _member(reg: Mapping[str, Technique], row: Mapping[str, Any], slot: str, picks: Mapping[str, Any]) -> Technique:
    """The registered technique a selection names for ``slot`` of composite ``row``; fails closed."""
    tid = picks.get(slot)
    allowed = row["members"][slot]
    if tid not in reg or reg[tid].row not in allowed:
        raise ValidationError(f"{row['id']}: slot {slot!r} must name a registered grid point of {allowed}, got {tid!r}")
    return reg[tid]


def build_registry(prereg: Mapping[str, Any],
                   selection: Optional[Mapping[str, Mapping[str, str]]] = None) -> Dict[str, Technique]:
    """``{id: Technique}``: one technique per grid point of every row of ``prereg["techniques"]``, in the
    pre-registration's order; then, when ``selection`` (``{"sop_futures": {"ko": id, "ki": id}, "sop_overlay": {"ko":
    id, "ki": id}}``) is given, the composites built from the members it names. Without a selection the composites
    are not defined. Fails closed on a row without an implementation, on a grid point no priced model covers and on
    a selection that does not name one registered grid point of an allowed row for every slot."""
    settings = Settings.from_prereg(prereg)
    factories = _row_factories()
    reg: Dict[str, Technique] = {}
    composites: List[Mapping[str, Any]] = []
    for row in prereg["techniques"]:
        rid = row["id"]
        if rid in _COMPOSITE_SLOTS:
            composites.append(row)
            continue
        if rid not in factories:
            raise ValidationError(f"technique row {rid!r} of the pre-registration has no implementation")
        for point in row["grid"]:
            tech = factories[rid](row, dict(point), settings)
            if tech.id in reg:
                raise ValidationError(f"technique id {tech.id!r} is registered twice")
            reg[tech.id] = tech
    if selection is None:
        return reg
    for row in composites:
        rid = row["id"]
        picks = selection.get(rid)
        if not isinstance(picks, Mapping) or set(picks) != set(_COMPOSITE_SLOTS[rid]):
            raise ValidationError(
                f"the selection must give {rid} exactly the slots {_COMPOSITE_SLOTS[rid]}, got {picks!r}")
        ko, ki = _member(reg, row, "ko", picks), _member(reg, row, "ki", picks)
        if rid == "sop_futures":
            reg[rid] = SopFutures(row, ko, ki, settings)
        else:
            (futures_id,) = row["members"]["futures"]
            if futures_id not in reg:
                raise ValidationError(f"{rid}: its futures member {futures_id!r} is not registered")
            reg[rid] = SopOverlay(row, reg[futures_id], ko, ki, settings)
    return reg


def composite_selection(path: Optional[Path] = None) -> Optional[Dict[str, Dict[str, str]]]:
    """The composites' members from the selection file (its ``composites`` block), ``None`` when the file does not
    exist yet (no training selection has been made); a file without the block fails closed."""
    path = SELECTION_PATH if path is None else Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    block = data.get("composites") if isinstance(data, dict) else None
    if not isinstance(block, dict) or not all(isinstance(v, dict) for v in block.values()):
        raise StudyDataError(f"{path}: no 'composites' block naming the members of the composites")
    return {str(k): {str(slot): str(tid) for slot, tid in v.items()} for k, v in block.items()}


#: Every registered technique: the pre-registered grid and, once the training selection exists, the composites.
REGISTRY: Dict[str, Technique] = build_registry(load_prereg(), composite_selection())
