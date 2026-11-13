"""The ledger pass: replay one technique on one note's recorded states and the real IM / MO marks (spec C7).

Seller view (D13). The recorded states are raw engine values for the LONG note; here ``seller_pv = -engine_price``
at the ladder centre, ``position_delta = -engine bump delta``, ``position_gamma = -engine_gamma`` and
``note_cash = -engine_note_cash`` (negative when the seller pays a coupon). The delta hedge is long IM when the
technique's ``note_delta`` is positive. Prices and cash are CNY per note (10 mio), deltas CNY per index point.

Order of one day ``t`` (C7): mark the positions held from ``t-1``; read the note's day (cash, alive); legs whose
series ends today exit; a fixing-day decision of the previous close fills (``next_open``); get the technique's
target; fill the option legs under D8, giving the legs held and the fill ratio ``phi``; IM target = ``clip(phi x
note_delta_if_filled + (1 - phi) x note_delta - delta of the legs held)``; trade IM under the clock. A dead note
targets flat and rows continue until every position is flat (a leg whose close was cut stays on the book and its
delta stays in the IM target). On the last available day (the note's last record while alive, the data end, or
``until``) everything is marked flat at that day's marks and the row is flagged.

Accounting: ``im_vm`` / ``mo_mtm`` = held x (mark_t - mark_{t-1}) x multiplier + traded x (mark_t - fill_t) x
multiplier (the second part also in ``fill_pnl``); ``accrual = cash_{t-1} x r x dt``; ``hedged = note_pnl + im_vm +
mo_mtm + accrual - cost``; ``cash_t = cash_{t-1} + accrual + note_cash + im_vm - premium paid - cost``, ``cash_{-1} =
0``. Day 0 is the inception close and ``note_pnl = seller_pv_0`` (the note trades at 0). Identity: the sum of
``hedged`` equals ``cash_end + seller_pv_end + option value_end``.

Choices the contract leaves open (each is also in a test): ``phi`` is the covered share of the desired lots,
``sum over desired legs of min(|desired|, held on the desired side) / sum |desired|``; option fills are at the
settlement under every clock; a leg exits uncut on its last trading day (at the close if it traded, else
cash-settled at the settlement, which costs nothing) and is flattened uncut on the last day; a held leg without a
mark, an implied vol or a series fails closed; the day's volume cap is shared by every trade of a contract that day;
what a technique refused that day (``Target.flags``, an overlay it could not build) is written to the row's flags;
on such a refusal day with legs held the target asks for exactly the legs held, so ``phi`` is not recomputed (it would
read 1): the day uses the ``phi`` of the last day on which an overlay was sized (legs desired, no refusal flag), the
legs held being unchanged since but for expiries, and the row's ``phi`` carries the value used; with no such earlier
day the recomputed value stands; the flatten on the last available day (data end, ``until``, the note's last record)
is a mark, not a trade, and costs nothing (a dead note's own flat target on that day is a trade and does cost).

Ledger-side code: no engine, no market data of its own, no file access.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close, is_zero

from s2ts import ACT, IM_MULT, MO_MULT, NOTIONAL, RATE
from s2ts.chain import DayChain
from s2ts.note import KO_PCT

from . import LADDER, StudyDataError, load_prereg
from . import black
from .board import DayBoard, OptionQuote
from .techniques import TRUE, DayContext, LegKey, NoteInfo, Target, Technique, freeze, hedge_delta, is_refusal

#: The three execution clocks (D7).
CLOCKS: Tuple[str, ...] = ("close", "twap", "next_open")

#: Ladder index of the spot itself.
CENTRE = len(LADDER) // 2


def leg_key_str(key: LegKey) -> str:
    """``"MO2403:C:5000"``: the JSON form of a leg key."""
    return f"{key[0]}:{key[1]}:{key[2]:g}"


@dataclass
class Run:
    """One technique on one note under one clock: the daily rows (spec C7's keys) and what identifies the run."""

    book: str
    inception: date
    coupon: float
    technique: str
    family: str
    params: Dict[str, Any]
    mode: str
    mo_cost_pt: float
    im_cost_rate: float
    desk_notional: float
    until: Optional[date]
    rows: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def clip_count(self) -> int:
        """Days on which the D9 clip bound."""
        return sum(1 for r in self.rows if r["clipped"])

    def to_json(self) -> Dict[str, Any]:
        """JSON-ready (no NaN is ever produced)."""
        return {"book": self.book, "inception": self.inception.isoformat(), "coupon": self.coupon,
                "technique": self.technique, "family": self.family, "params": dict(self.params), "mode": self.mode,
                "mo_cost_pt": self.mo_cost_pt, "im_cost_rate": self.im_cost_rate, "desk_notional": self.desk_notional,
                "until": None if self.until is None else self.until.isoformat(), "clip_count": self.clip_count,
                "rows": self.rows}


@dataclass(frozen=True)
class _LegRules:
    """D8 at desk scale: ``scale = desk_notional / NOTIONAL`` desk lots per ledger lot."""

    scale: float
    min_volume: int
    min_open_interest: int
    max_volume_share: float
    series_limit: float
    cost_pt: float


@dataclass
class _Positions:
    """What the desk holds between two closes, and the marks it was last marked at."""

    im: Dict[str, float] = field(default_factory=dict)
    im_mark: Dict[str, float] = field(default_factory=dict)
    legs: Dict[LegKey, float] = field(default_factory=dict)
    leg_mark: Dict[LegKey, float] = field(default_factory=dict)

    def held_im(self) -> Dict[str, float]:
        return {k: v for k, v in self.im.items() if not is_zero(v)}

    def held_legs(self) -> Dict[LegKey, float]:
        return {k: v for k, v in self.legs.items() if not is_zero(v)}

    def flat(self) -> bool:
        return not self.held_im() and not self.held_legs()


@dataclass
class _Acc:
    """One day's accumulators."""

    im_vm: float = 0.0
    mo_mtm: float = 0.0
    fill_pnl: float = 0.0
    cost: float = 0.0
    premium: float = 0.0
    clipped: bool = False
    phi: float = 1.0
    unfilled: Dict[LegKey, float] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)

    def flag(self, name: str) -> None:
        if name not in self.flags:
            self.flags.append(name)


@dataclass(frozen=True)
class _Pending:
    """A fixing-day decision waiting for the next open (``next_open``): the target and what it was sized with."""

    target: Target
    spot: float
    far_code: str
    far_close: float
    clip: bool


def _states_dict(states: Any) -> Dict[str, Any]:
    """A ``NoteStates`` or its JSON form as one dict."""
    return states.to_json() if hasattr(states, "to_json") else dict(states)


# ---------------------------------------------------------------------------
# IM
# ---------------------------------------------------------------------------


def _im_close(code: str, dc: DayChain, board: DayBoard) -> float:
    """The day's IM close: the chain's quote when it lists the contract, else the board's; fails closed."""
    for q in dc.im:
        if q.code == code:
            return float(q.close)
    bar = board.im.get(code)
    if bar is None or bar.close is None:
        raise StudyDataError(f"{dc.day}: no IM close for held contract {code}")
    return float(bar.close)


def _im_fill(code: str, kind: str, dc: DayChain, board: DayBoard, close: float) -> float:
    """Fill price of an IM trade: ``close``; ``vwap`` (the day VWAP, the close when nothing traded); ``open``."""
    if kind == "close":
        return close
    bar = board.im.get(code)
    if bar is None:
        raise StudyDataError(f"{dc.day}: no IM bar for {code} on the board")
    if kind == "vwap":
        return close if (not bar.volume or bar.vwap is None) else float(bar.vwap)
    if bar.open is None:
        raise StudyDataError(f"{dc.day}: no open for {code}: the next-open fill cannot be made")
    return float(bar.open)


def _im_trade(acc: _Acc, pos: _Positions, new_im: Mapping[str, float], kind: str, dc: DayChain, board: DayBoard,
              rate: float) -> None:
    """Trade the IM book to ``new_im`` at the ``kind`` fill."""
    for code in sorted(set(pos.im) | set(new_im)):
        dq = new_im.get(code, 0.0) - pos.im.get(code, 0.0)
        if is_zero(dq):
            continue
        close = _im_close(code, dc, board)
        fill = _im_fill(code, kind, dc, board, close)
        acc.cost += rate * abs(dq) * fill * IM_MULT
        gain = dq * (close - fill) * IM_MULT
        acc.im_vm += gain
        acc.fill_pnl += gain
    pos.im = {c: v for c, v in new_im.items() if not is_zero(v)}


def _im_lots(im_delta: float, spot: float, far_code: str, far_close: float, clip: bool,
             bounds: Tuple[float, float]) -> Tuple[Dict[str, float], bool]:
    """IM lots for ``im_delta`` sized at ``spot`` / ``far_close``, with the D9 clip on ``h = im_delta x spot /
    NOTIONAL``."""
    clipped = False
    if clip:
        h = im_delta * spot / NOTIONAL
        h_clipped = min(max(h, bounds[0]), bounds[1])
        if not is_close(h, h_clipped):
            clipped, im_delta = True, h_clipped * NOTIONAL / spot
    lots = im_delta * spot / (IM_MULT * far_close)
    return ({} if is_zero(lots) else {far_code: lots}), clipped


# ---------------------------------------------------------------------------
# Option legs (D8)
# ---------------------------------------------------------------------------


def _quote(board: DayBoard, key: LegKey) -> Optional[OptionQuote]:
    return board.mo.get(key[0], {}).get((key[1], key[2]))


def _held_quote(board: DayBoard, key: LegKey, day: date) -> OptionQuote:
    q = _quote(board, key)
    if q is None:
        raise StudyDataError(f"{day}: no settlement for held leg {leg_key_str(key)}")
    return q


def _side(cp: str, lots: float) -> int:
    """CFFEX position side: long calls and short puts are one side (+1), short calls and long puts the other (-1)."""
    return 1 if (cp == "C") == (lots > 0) else -1


def _trade_leg(acc: _Acc, pos: _Positions, key: LegKey, dq: float, price: float, rules: _LegRules,
               *, cost: bool = True) -> None:
    pos.legs[key] = pos.legs.get(key, 0.0) + dq
    if is_zero(pos.legs[key]):
        del pos.legs[key]
    acc.premium += dq * price * MO_MULT
    if cost:
        acc.cost += rules.cost_pt * MO_MULT * abs(dq)


def _expire_legs(acc: _Acc, pos: _Positions, board: DayBoard, dc: DayChain, rules: _LegRules) -> None:
    """Legs whose series ends today exit, uncut: at the close if the option traded (and pays the cost), else
    cash-settled at the day's settlement (free)."""
    for key in sorted(pos.held_legs()):
        info = board.series.get(key[0])
        if info is None:
            raise StudyDataError(f"{dc.day}: no series info for held leg {leg_key_str(key)}")
        if info.last_trading_day < dc.day:
            raise StudyDataError(f"{dc.day}: leg {leg_key_str(key)} is held past its last trading day")
        if info.last_trading_day > dc.day:
            continue
        q = _held_quote(board, key, dc.day)
        lots = pos.legs[key]
        traded = bool(q.volume) and q.volume > 0 and q.close is not None
        price = float(q.close) if traded else float(q.settle)
        gain = -lots * (q.settle - price) * MO_MULT
        acc.mo_mtm += gain
        acc.fill_pnl += gain
        _trade_leg(acc, pos, key, -lots, price, rules, cost=traded)
        acc.flag("expiry_exit" if traded else "expiry_settled")


def _fill_legs(acc: _Acc, pos: _Positions, desired: Mapping[LegKey, float], board: DayBoard, dc: DayChain,
               rules: _LegRules, used: Dict[LegKey, float]) -> None:
    """Move the leg book towards ``desired`` (ledger lots, signed, positive = long) under D8; sets ``acc.phi`` and
    ``acc.unfilled`` (desired minus held, per leg). Reductions need a settlement only and go first; openings need
    a settlement, an implied vol, volume and open interest; every trade is cut to the volume share left in the
    contract that day, an opening also to the series' side limit."""
    desired = {k: float(v) for k, v in desired.items() if not is_zero(v)}
    plan: Dict[LegKey, Tuple[float, float]] = {}
    for key in sorted(set(desired) | set(pos.held_legs())):
        held = pos.legs.get(key, 0.0)
        dq = desired.get(key, 0.0) - held
        if is_zero(dq):
            continue
        reduce = math.copysign(min(abs(dq), abs(held)), dq) if (not is_zero(held) and dq * held < 0) else 0.0
        plan[key] = (reduce, dq - reduce)

    def room(key: LegKey, q: OptionQuote) -> float:
        return max(0.0, rules.max_volume_share * q.volume / rules.scale - used.get(key, 0.0))

    for key, (reduce, _) in plan.items():
        if is_zero(reduce):
            continue
        q = _held_quote(board, key, dc.day)
        take = min(abs(reduce), room(key, q))
        if take < abs(reduce) and not is_close(take, abs(reduce)):
            acc.flag("volume_cut")
        if not is_zero(take):
            used[key] = used.get(key, 0.0) + take
            _trade_leg(acc, pos, key, math.copysign(take, reduce), float(q.settle), rules)
    for key, (reduce, opening) in plan.items():
        if is_zero(opening):
            continue
        if not is_zero(reduce) and not is_zero(pos.legs.get(key, 0.0)) and pos.legs[key] * opening < 0:
            continue                                     # the old side is not closed yet: nothing opens the other
        q = _quote(board, key)
        info = board.series.get(key[0])
        if (q is None or info is None or q.volume < rules.min_volume or q.open_interest is None
                or q.open_interest < rules.min_open_interest or black.implied_vol(q.settle, key[1], key[2], info) is None):
            acc.flag("refused_open")
            continue
        take = min(abs(opening), room(key, q))
        if take < abs(opening) and not is_close(take, abs(opening)):
            acc.flag("volume_cut")
        side = _side(key[1], opening)
        on_side = sum(abs(v) for k, v in pos.held_legs().items() if k[0] == key[0] and _side(k[1], v) == side)
        headroom = max(0.0, rules.series_limit / rules.scale - on_side)
        if take > headroom and not is_close(take, headroom):
            take = headroom
            acc.flag("series_limit")
        if not is_zero(take):
            used[key] = used.get(key, 0.0) + take
            _trade_leg(acc, pos, key, math.copysign(take, opening), float(q.settle), rules)
    held = pos.held_legs()
    acc.unfilled = {k: desired.get(k, 0.0) - held.get(k, 0.0) for k in set(desired) | set(held)
                    if not is_zero(desired.get(k, 0.0) - held.get(k, 0.0))}
    total = sum(abs(v) for v in desired.values())
    covered = sum(min(abs(v), max(0.0, held.get(k, 0.0) * math.copysign(1.0, v))) for k, v in desired.items())
    acc.phi = 1.0 if is_zero(total) else covered / total


def _legs_delta(pos: _Positions, board: DayBoard, dc: DayChain) -> float:
    """CNY per index point of the legs held: Black-76 at each leg's implied vol of the day, and its zero-vol limit
    for a settlement at or below the intrinsic bound (``black.greeks_at_settle``); any other leg without an implied
    vol fails closed."""
    total = 0.0
    for key, lots in pos.held_legs().items():
        q = _held_quote(board, key, dc.day)
        info = board.series.get(key[0])
        greeks = None if info is None else black.greeks_at_settle(q.settle, key[1], key[2], info, dc.spot)
        if greeks is None:
            raise StudyDataError(f"{dc.day}: held leg {leg_key_str(key)} has no implied vol: its delta is unknown")
        total += lots * greeks[0] * MO_MULT
    return total


def _flatten(acc: _Acc, pos: _Positions, board: DayBoard, dc: DayChain, rules: _LegRules) -> None:
    """Everything marked flat at the day's marks (IM close, option settlement), uncut and free of cost: the book ends
    where the data or the slice ends, which is not a trade."""
    _im_trade(acc, pos, {}, "close", dc, board, 0.0)
    for key in sorted(pos.held_legs()):
        q = _held_quote(board, key, dc.day)
        _trade_leg(acc, pos, key, -pos.legs[key], float(q.settle), rules, cost=False)


def _execute(acc: _Acc, pos: _Positions, target: Target, spot: float, far_code: str, far_close: float, clip: bool,
             im_kind: str, dc: DayChain, board: DayBoard, rules: _LegRules, used: Dict[LegKey, float],
             bounds: Tuple[float, float], im_rate: float, sized_phi: Optional[float] = None) -> Optional[float]:
    """Fill the target's legs, size the IM hedge net of the legs held and trade it (``im_kind`` fill).

    ``sized_phi``: the fill ratio of the last day on which an overlay was sized. A target that carries an overlay
    refusal flag while legs are held asks for exactly the legs held, which says nothing about how much of the digital
    they cover: that day uses ``sized_phi`` (when there is one) instead of the ratio of held to "desired". Returns the
    fill ratio to carry forward: today's when an overlay was sized today (legs desired, no refusal), else
    ``sized_phi`` unchanged."""
    _fill_legs(acc, pos, target.legs, board, dc, rules, used)
    refused = is_refusal(target)
    if refused and pos.held_legs() and sized_phi is not None:
        acc.phi = sized_phi
    elif not refused and any(not is_zero(v) for v in target.legs.values()):
        sized_phi = acc.phi
    filled = target.note_delta if target.note_delta_if_filled is None else target.note_delta_if_filled
    im_delta = acc.phi * filled + (1.0 - acc.phi) * target.note_delta - _legs_delta(pos, board, dc)
    new_im, clipped = _im_lots(im_delta, spot, far_code, far_close, clip, bounds)
    acc.clipped = acc.clipped or clipped
    held = pos.held_im()
    if held and new_im and set(held) != set(new_im):
        acc.flag("roll")
    _im_trade(acc, pos, new_im, im_kind, dc, board, im_rate)
    return sized_phi


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run_technique(states: Any, technique: Technique, boards: Mapping[date, DayBoard], chains: Sequence[DayChain], *,
                  mode: str, mo_cost_pt: float, desk_notional: float, until: Optional[date] = None) -> Run:
    """Replay ``technique`` on the recorded ``states`` of one note (module docstring). ``boards`` by day, ``chains``
    in date order covering at least the note's life (and the days after it until the book is flat)."""
    if mode not in CLOCKS:
        raise ValidationError(f"mode must be one of {CLOCKS}, got {mode!r}")
    prereg = load_prereg()
    trade = prereg["tradability"]
    im_rate = float(trade["im_cost_rate"])
    bounds = (float(prereg["im_clip"][0]), float(prereg["im_clip"][1]))
    rules = _LegRules(float(desk_notional) / NOTIONAL, int(trade["min_volume"]), int(trade["min_open_interest"]),
                      float(trade["max_volume_share"]), float(trade["series_limit_lots"]), float(mo_cost_pt))
    data = _states_dict(states)
    records: List[Dict[str, Any]] = list(data["days"])
    if not records:
        raise StudyDataError("a note with no recorded day")
    frozen = [freeze(r) for r in records]
    chain_days = [c.day for c in chains]
    inception = date.fromisoformat(records[0]["date"])
    if inception not in chain_days:
        raise StudyDataError(f"inception {inception} is not a chain day")
    start = chain_days.index(inception)
    for k, rec in enumerate(records):
        if start + k >= len(chains) or chain_days[start + k].isoformat() != rec["date"]:
            raise StudyDataError(f"record {rec['date']} is not the chain's day {start + k}")
    missing = [k for k in ("book", "ki_pct", "strike", "ko_level", "ki_level") if data.get(k) is None]
    if missing:
        raise StudyDataError(f"the state file carries no {', '.join(missing)}: the note's book and levels are the "
                             "pricing pass's (a seasoned note is not struck at its booking spot) and are never "
                             "defaulted or derived here")
    note = NoteInfo(str(data["book"]), inception, float(data["coupon"]), float(data["ki_pct"]),
                    KO_PCT, float(records[0]["spot"]), float(data["strike"]), float(data["ko_level"]),
                    float(data["ki_level"]))
    run = Run(note.book, inception, note.coupon, technique.id, technique.family, dict(technique.params), mode,
              float(mo_cost_pt), im_rate, float(desk_notional), until)
    im_kind = "vwap" if mode == "twap" else "close"

    pos = _Positions()
    pending: Optional[_Pending] = None
    sized_phi: Optional[float] = None                        # the fill ratio of the last day an overlay was sized
    cash = 0.0
    prev_pv = 0.0
    prev_day: Optional[date] = None
    prev_ctx: Optional[DayContext] = None
    for i, dc in enumerate(chains[start:]):
        day = dc.day
        if until is not None and day > until:
            break
        board = boards.get(day)
        if board is None:
            raise StudyDataError(f"{day}: no board")
        rec = records[i] if i < len(records) else None
        alive = rec is not None and bool(rec["alive"])
        next_idx = start + i + 1
        has_next = next_idx < len(chains) and (until is None or chains[next_idx].day <= until)
        final = (alive and i == len(records) - 1) or not has_next
        cut_by_until = until is not None and next_idx < len(chains) and chains[next_idx].day > until
        is_fixing = rec is not None and i > 0 and (
            records[i - 1]["next_ko"] == rec["date"] or (rec["knocked_in"] and not records[i - 1]["knocked_in"]))
        acc = _Acc()
        used: Dict[LegKey, float] = {}

        # 1. mark the positions held from the previous close
        for code, lots in pos.held_im().items():
            acc.im_vm += lots * (_im_close(code, dc, board) - pos.im_mark[code]) * IM_MULT
        for key, lots in pos.held_legs().items():
            acc.mo_mtm += lots * (_held_quote(board, key, day).settle - pos.leg_mark[key]) * MO_MULT
        # 2. the note's day
        if alive:
            true = rec["states"].get(TRUE)
            if true is None:
                raise StudyDataError(f"{day}: the note is alive but has no recorded state of the true model")
            seller_pv = -float(true["engine_price"][CENTRE])
            position_delta = -hedge_delta(true, dc.spot)
            gamma = true["engine_gamma"][CENTRE]
            position_gamma = None if gamma is None else -float(gamma)
        else:
            seller_pv, position_delta, position_gamma = 0.0, 0.0, 0.0
        note_cash = 0.0 if rec is None else -float(rec["engine_note_cash"])
        note_pnl = seller_pv - prev_pv + note_cash

        # 3. legs whose series ends today exit; a fixing-day decision of the previous close fills (next_open)
        _expire_legs(acc, pos, board, dc, rules)
        if pending is not None:
            sized_phi = _execute(acc, pos, pending.target, pending.spot, pending.far_code, pending.far_close,
                                 pending.clip, "open", dc, board, rules, used, bounds, im_rate, sized_phi)
            acc.flag("deferred_fill")
            pending = None

        # 4. the technique's target for today's close, and the fills
        far = dc.im[-1]
        if final:
            # an alive note is flattened at once; a dead note's flat target is executed (cuts apply) and only what
            # that leaves is flattened
            if not alive:
                _execute(acc, pos, Target(0.0, {}), dc.spot, far.code, float(far.close), False, "close", dc, board,
                         rules, used, bounds, im_rate)
            if alive or not pos.flat():
                _flatten(acc, pos, board, dc, rules)
                acc.flag("until" if cut_by_until else "data_end")
            if mode == "next_open" and is_fixing:
                acc.flag("last_day_fixing")
        else:
            if alive:
                ctx = DayContext.build(day, i, dc.spot, dc.mo.vol, frozen[i], board, note, pos.held_im(),
                                       pos.held_legs(), prev_ctx)
                target = technique.target(ctx)
                if not isinstance(target, Target):
                    raise ValidationError(f"{technique.id}: target() must return a Target, got {type(target).__name__}")
                for name in target.flags:                    # what the technique refused today (an overlay)
                    acc.flag(name)
                prev_ctx = ctx
            else:
                target = Target(0.0, {})
            clip = technique.clip and alive
            if mode == "next_open" and is_fixing:
                pending = _Pending(target, dc.spot, far.code, float(far.close), clip)
                acc.flag("fixing_deferred")
            else:
                sized_phi = _execute(acc, pos, target, dc.spot, far.code, float(far.close), clip, im_kind, dc, board,
                                     rules, used, bounds, im_rate, sized_phi)
        pos.im_mark = {code: _im_close(code, dc, board) for code in pos.held_im()}
        pos.leg_mark = {key: float(_held_quote(board, key, day).settle) for key in pos.held_legs()}
        im_lots = sum(pos.held_im().values())
        hedge_ratio = sum(lots * pos.im_mark[code] for code, lots in pos.held_im().items()) * IM_MULT / NOTIONAL

        # 5. accounting
        accrual = 0.0 if prev_day is None else cash * RATE * (day - prev_day).days / ACT
        hedged = note_pnl + acc.im_vm + acc.mo_mtm + accrual - acc.cost
        cash = cash + accrual + note_cash + acc.im_vm - acc.premium - acc.cost
        run.rows.append({
            "date": day.isoformat(), "seller_pv": seller_pv, "position_delta": position_delta,
            "position_gamma": position_gamma, "note_cash": note_cash, "note_pnl": note_pnl, "im_vm": acc.im_vm,
            "mo_mtm": acc.mo_mtm, "fill_pnl": acc.fill_pnl, "accrual": accrual, "cost": acc.cost, "hedged": hedged,
            "cash": cash, "hedge_ratio": hedge_ratio, "im_lots": im_lots,
            "legs": {leg_key_str(k): v for k, v in sorted(pos.held_legs().items())}, "phi": acc.phi,
            "clipped": acc.clipped,
            "unfilled": {leg_key_str(k): v for k, v in sorted(acc.unfilled.items())}, "flags": acc.flags,
        })
        prev_pv, prev_day = seller_pv, day
        if final or (i >= len(records) - 1 and pos.flat() and pending is None):
            break
    return run
