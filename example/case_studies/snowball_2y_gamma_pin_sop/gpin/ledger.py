"""The ledger pass: replay one technique on one note's recorded states and the real IM / MO marks (spec C7).

Seller view (D13). The recorded states are raw engine values for the LONG note; here ``seller_pv = -engine_price``
at the ladder centre, ``position_delta = -engine bump delta``, ``position_gamma = -engine_gamma`` and
``note_cash = -engine_note_cash`` (negative when the seller pays a coupon). The delta hedge is long IM when the
technique's ``note_delta`` is positive. Prices and cash are CNY per note (10 mio), deltas CNY per index point.

Order of one day ``t`` (C7): mark the positions held from ``t-1``; read the note's day (cash, alive); get the
technique's target; fill the option legs under D8; IM target = ``clip(phi x note_delta_if_filled + (1 - phi) x
note_delta - delta of the legs held)``; trade IM under the clock. A dead note targets flat and rows continue until
every position is flat. On the last available day (the note's last record while alive, the data end, or ``until``)
everything is marked flat at that day's marks and the row is flagged.

Accounting: ``im_vm`` / ``mo_mtm`` = held x (mark_t - mark_{t-1}) x multiplier + traded x (mark_t - fill_t) x
multiplier (the second part also in ``fill_pnl``); ``accrual = cash_{t-1} x r x dt``; ``hedged = note_pnl + im_vm +
mo_mtm + accrual - cost``; ``cash_t = cash_{t-1} + accrual + note_cash + im_vm - premium paid - cost``, ``cash_{-1} =
0``. Day 0 is the inception close and ``note_pnl = seller_pv_0`` (the note trades at 0). Identity: the sum of
``hedged`` equals ``cash_end + seller_pv_end + option value_end``.

Ledger-side code: no engine, no market data of its own, no file access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close, is_zero, safe_divide

from s2ts import ACT, IM_MULT, NOTIONAL, RATE
from s2ts.chain import DayChain
from s2ts.note import KO_PCT

from . import LADDER, StudyDataError, load_prereg
from .board import DayBoard
from .techniques import TRUE, DayContext, NoteInfo, Target, Technique, freeze, hedge_delta

#: The three execution clocks (D7).
CLOCKS: Tuple[str, ...] = ("close", "twap", "next_open")

#: Ladder index of the spot itself.
CENTRE = len(LADDER) // 2


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


@dataclass
class _Positions:
    """What the desk holds between two closes, and the marks it was last marked at."""

    im: Dict[str, float] = field(default_factory=dict)
    im_mark: Dict[str, float] = field(default_factory=dict)

    def held_im(self) -> Dict[str, float]:
        return {k: v for k, v in self.im.items() if not is_zero(v)}


def _states_dict(states: Any) -> Dict[str, Any]:
    """A ``NoteStates`` or its JSON form as one dict."""
    return states.to_json() if hasattr(states, "to_json") else dict(states)


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


def _im_trade(pos: _Positions, new_im: Mapping[str, float], kind: str, dc: DayChain, board: DayBoard,
              rate: float) -> Tuple[float, float]:
    """Trade the IM book to ``new_im`` at the ``kind`` fill; returns ``(cost, fill part of the day's P&L)``."""
    cost = gain = 0.0
    for code in sorted(set(pos.im) | set(new_im)):
        dq = new_im.get(code, 0.0) - pos.im.get(code, 0.0)
        if is_zero(dq):
            continue
        close = _im_close(code, dc, board)
        fill = _im_fill(code, kind, dc, board, close)
        cost += rate * abs(dq) * fill * IM_MULT
        gain += dq * (close - fill) * IM_MULT
    pos.im = {c: v for c, v in new_im.items() if not is_zero(v)}
    return cost, gain


def _im_target(target: Target, spot: float, far_code: str, far_close: float, clip: bool,
               bounds: Tuple[float, float]) -> Tuple[Dict[str, float], bool]:
    """IM lots for a target sized at ``spot`` / ``far_close``, with the D9 clip on ``h = im_delta x spot / NOTIONAL``."""
    im_delta = float(target.note_delta)
    clipped = False
    if clip:
        h = im_delta * spot / NOTIONAL
        h_clipped = min(max(h, bounds[0]), bounds[1])
        if not is_close(h, h_clipped):
            clipped, im_delta = True, h_clipped * NOTIONAL / spot
    lots = im_delta * spot / (IM_MULT * far_close)
    return ({} if is_zero(lots) else {far_code: lots}), clipped


@dataclass(frozen=True)
class _Pending:
    """A fixing-day decision waiting for the next open (``next_open``): the target and what it was sized with."""

    target: Target
    spot: float
    far_code: str
    far_close: float
    clip: bool


def run_technique(states: Any, technique: Technique, boards: Mapping[date, DayBoard], chains: Sequence[DayChain], *,
                  mode: str, mo_cost_pt: float, desk_notional: float, until: Optional[date] = None) -> Run:
    """Replay ``technique`` on the recorded ``states`` of one note (module docstring). ``boards`` by day, ``chains``
    in date order covering at least the note's life (and the days after it until the book is flat)."""
    if mode not in CLOCKS:
        raise ValidationError(f"mode must be one of {CLOCKS}, got {mode!r}")
    prereg = load_prereg()
    trade = prereg["tradability"]
    im_cost_rate = float(trade["im_cost_rate"])
    bounds = (float(prereg["im_clip"][0]), float(prereg["im_clip"][1]))
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
    note = NoteInfo(str(data.get("book", "ki75")), inception, float(data["coupon"]), float(data.get("ki_pct", 0.75)),
                    KO_PCT, float(records[0]["spot"]))
    run = Run(note.book, inception, note.coupon, technique.id, technique.family, dict(technique.params), mode,
              float(mo_cost_pt), im_cost_rate, float(desk_notional), until)
    im_kind = "vwap" if mode == "twap" else "close"

    pos = _Positions()
    pending: Optional[_Pending] = None
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
        is_fixing = rec is not None and i > 0 and (
            records[i - 1]["next_ko"] == rec["date"] or (rec["knocked_in"] and not records[i - 1]["knocked_in"]))
        flags: List[str] = []
        cost = fill_pnl = 0.0
        clipped = False

        # 1. mark the IM positions held from the previous close
        im_vm = 0.0
        for code, lots in pos.held_im().items():
            im_vm += lots * (_im_close(code, dc, board) - pos.im_mark[code]) * IM_MULT
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

        # 3. a fixing-day decision of the previous close fills at today's open (next_open)
        if pending is not None:
            new_im, clipped_a = _im_target(pending.target, pending.spot, pending.far_code, pending.far_close,
                                           pending.clip, bounds)
            c, g = _im_trade(pos, new_im, "open", dc, board, im_cost_rate)
            cost, fill_pnl, im_vm = cost + c, fill_pnl + g, im_vm + g
            clipped = clipped or clipped_a
            flags.append("deferred_fill")
            pending = None

        # 4. the technique's target for today's close
        held_now = pos.held_im()
        if alive and not final:
            ctx = DayContext.build(day, i, dc.spot, dc.mo.vol, frozen[i], board, note, held_now, {}, prev_ctx)
            target = technique.target(ctx)
            if not isinstance(target, Target):
                raise ValidationError(f"{technique.id}: target() must return a Target, got {type(target).__name__}")
            if target.legs:
                raise ValidationError("option legs are not built yet")
            prev_ctx = ctx
        else:
            target = Target(0.0, {})
        far = dc.im[-1]
        fixing_clock = mode == "next_open" and is_fixing
        if fixing_clock and not final:
            pending = _Pending(target, dc.spot, far.code, float(far.close), technique.clip and alive)
            flags.append("fixing_deferred")
        else:
            if fixing_clock:
                flags.append("last_day_fixing")
            new_im, clipped_b = _im_target(target, dc.spot, far.code, float(far.close),
                                           technique.clip and alive and not final, bounds)
            if held_now and new_im and set(held_now) != set(new_im):
                flags.append("roll")
            c, g = _im_trade(pos, new_im, "close" if final else im_kind, dc, board, im_cost_rate)
            cost, fill_pnl, im_vm = cost + c, fill_pnl + g, im_vm + g
            clipped = clipped or clipped_b
        if final and alive:
            flags.append("data_end")
        pos.im_mark = {code: _im_close(code, dc, board) for code in pos.held_im()}
        im_lots = sum(pos.held_im().values())
        hedge_ratio = sum(lots * pos.im_mark[code] for code, lots in pos.held_im().items()) * IM_MULT / NOTIONAL

        # 5. accounting
        accrual = 0.0 if prev_day is None else cash * RATE * (day - prev_day).days / ACT
        hedged = note_pnl + im_vm + accrual - cost
        cash = cash + accrual + note_cash + im_vm - cost
        run.rows.append({
            "date": day.isoformat(), "seller_pv": seller_pv, "position_delta": position_delta,
            "position_gamma": position_gamma, "note_cash": note_cash, "note_pnl": note_pnl, "im_vm": im_vm,
            "mo_mtm": 0.0, "fill_pnl": fill_pnl, "accrual": accrual, "cost": cost, "hedged": hedged, "cash": cash,
            "hedge_ratio": hedge_ratio, "im_lots": im_lots, "legs": {}, "phi": 1.0, "clipped": clipped,
            "unfilled": {}, "flags": flags,
        })
        prev_pv, prev_day = seller_pv, day
        if final or (i >= len(records) - 1 and not pos.held_im() and pending is None):
            break
    return run
