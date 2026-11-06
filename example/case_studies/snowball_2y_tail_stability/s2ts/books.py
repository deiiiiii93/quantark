"""Hedge books (design §4.4): pure ledgers — targets, trades, marks, overnight P&L. No pricing.

The desk is SHORT the note and ``DayGreeks`` are holder-side, so every target is
``+greek / multiplier`` (the hedge replicates the holder's exposure).

- **Book A** — one IM contract, the longest listed (``dc.im[-1]``; ``dc.im``
  already excludes < 7 days): ``+spot_delta·S/(200·F_far)`` lots. It rolls on
  the first day a longer contract appears (the old key leaves the target).
- **Book B** — the curve's pillars: ``+δ_i/200`` per IM pillar, plus the
  pinned-forward delta ``+pinned·S/(200·F_far)`` on the far IM, and
  ``+δ_MO/(100·DF(T_n))`` C−P pairs of the longest MO series. The pair is keyed
  ``MOyymm@K`` and held at the strike it was opened at; on an MO roll the old
  key is closed and the new pillar opened at that day's ATM strike.

Instrument keys: ``IMyymm`` (marked at the CFFEX close, 200 CNY/pt/lot) and
``MOyymm@K`` (marked at the settlement ``C − P`` of strike K, 100 CNY/pt/pair).
A position the day's chain cannot mark — an IM pillar that fell under 7 days,
the old MO series on a roll day — needs ``extra`` marks for that day
(``offchain_marks`` over the day's parsed CFFEX rows); without them every call
fails closed.  IM futures carry no value (variation margin settles daily); an
MO pair's value is ``qty·(C−P)·100``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_zero, safe_exp

from . import IM_MULT, MO_MULT, RATE, StudyDataError
from .chain import DayChain
from .note import DayGreeks

#: IM cost: 1 bp of traded notional.
IM_COST_RATE = 1e-4
#: MO cost: 0.2 index points per option leg per side; a C−P pair is two legs.
MO_LEG_COST_PT = 0.2
MO_PAIR_COST = 2.0 * MO_LEG_COST_PT * MO_MULT   # 40 CNY per pair traded

Marks = Mapping[str, float]


@dataclass(frozen=True)
class Trade:
    """``qty`` signed lots (IM) or pairs (MO) at ``price`` (IM close / MO ``C−P``); ``cost`` in CNY."""

    instrument: str
    qty: float
    price: float
    cost: float


def is_mo(key: str) -> bool:
    return "@" in key


def mo_key(series: str, strike: float) -> str:
    return f"{series}@{strike:g}"


def split_mo_key(key: str) -> Tuple[str, float]:
    series, k = key.split("@")
    return series, float(k)


def multiplier(key: str) -> float:
    return MO_MULT if is_mo(key) else IM_MULT


def trade_cost(key: str, qty: float, price: float) -> float:
    if is_mo(key):
        return MO_PAIR_COST * abs(qty)
    return IM_COST_RATE * abs(qty) * price * IM_MULT


def turnover(trades: Iterable[Trade]) -> float:
    """``Σ |qty|`` (MO pairs counted as lots)."""
    return float(sum(abs(t.qty) for t in trades))


def total_cost(trades: Iterable[Trade]) -> float:
    return float(sum(t.cost for t in trades))


def offchain_marks(rows: Mapping[str, Mapping[str, float]], keys: Iterable[str]) -> Dict[str, float]:
    """Marks for ``keys`` from one day's parsed CFFEX rows (``chain.load_day_rows``):
    IM close; MO settlement ``C − P`` at the key's strike."""
    out: Dict[str, float] = {}
    for key in keys:
        try:
            if is_mo(key):
                series, k = split_mo_key(key)
                out[key] = float(rows[f"{series}-C-{k:g}"]["settle"]) - float(rows[f"{series}-P-{k:g}"]["settle"])
            else:
                out[key] = float(rows[key]["close"])
        except KeyError as err:
            raise StudyDataError(f"CFFEX rows carry no mark for {key} ({err})") from err
    return out


class Book:
    """Shared ledger: positions by instrument key, the marks they were last set at, trades and P&L."""

    name = "book"

    def __init__(self) -> None:
        #: key -> qty for every instrument of the latest target (zero quantities kept).
        self.positions: Dict[str, float] = {}
        self._marks: Dict[str, float] = {}
        self._last_day: Optional[date] = None
        #: True when the last rebalance changed the book's instrument set (a roll).
        self.rolled = False

    # ----------------------------------------------------------------- marks

    @staticmethod
    def _chain_price(dc: DayChain, key: str) -> Optional[float]:
        if is_mo(key):
            series, k = split_mo_key(key)
            if series == dc.mo.series and k in dc.mo_all:
                c, p = dc.mo_all[k]
                return float(c - p)
            return None
        for q in dc.im:
            if q.code == key:
                return float(q.close)
        return None

    def _price(self, dc: DayChain, key: str, extra: Optional[Marks]) -> float:
        price = self._chain_price(dc, key)
        if price is not None:
            return price
        if extra is not None and key in extra:
            return float(extra[key])
        raise StudyDataError(f"{dc.day}: the chain has no mark for {key}; pass it in extra (offchain_marks)")

    def unmarkable(self, dc: DayChain) -> List[str]:
        """Held keys (non-zero) the day's chain cannot mark: they need ``extra``."""
        return sorted(k for k, q in self.positions.items() if not is_zero(q) and self._chain_price(dc, k) is None)

    def mark(self, dc: DayChain, extra: Optional[Marks] = None) -> Dict[str, float]:
        """Today's price of every held key."""
        return {k: self._price(dc, k, extra) for k, q in self.positions.items()
                if not is_zero(q) or self._chain_price(dc, k) is not None}

    def value(self, dc: DayChain, extra: Optional[Marks] = None) -> float:
        """Mark-to-market value of the MO pairs (IM futures carry none)."""
        return float(sum(q * self._price(dc, k, extra) * MO_MULT
                         for k, q in self.positions.items() if is_mo(k) and not is_zero(q)))

    # ----------------------------------------------------------------- P&L

    def pnl_split(self, prev: DayChain, dc: DayChain, extra: Optional[Marks] = None) -> Tuple[float, float]:
        """Overnight P&L of the positions held from ``prev``'s close: (IM variation margin, MO pair MTM)."""
        if not self.positions:
            return 0.0, 0.0
        if prev.day != self._last_day:
            raise ValidationError(f"{self.name}: last rebalanced on {self._last_day}, P&L asked from {prev.day}")
        if not dc.day > prev.day:
            raise ValidationError(f"{self.name}: {dc.day} is not after {prev.day}")
        im = mo = 0.0
        for key, qty in self.positions.items():
            if is_zero(qty):
                continue
            change = qty * (self._price(dc, key, extra) - self._marks[key]) * multiplier(key)
            if is_mo(key):
                mo += change
            else:
                im += change
        return im, mo

    def pnl(self, prev: DayChain, dc: DayChain, extra: Optional[Marks] = None) -> float:
        """Positions held overnight × price change × multiplier (IM 200, MO 100)."""
        im, mo = self.pnl_split(prev, dc, extra)
        return im + mo

    @staticmethod
    def premium(trades: Iterable[Trade]) -> float:
        """MO premium paid for ``trades`` (``Σ qty·(C−P)·100``, signed: > 0 is an outlay)."""
        return float(sum(t.qty * t.price * MO_MULT for t in trades if is_mo(t.instrument)))

    # ----------------------------------------------------------------- trading

    def rebalance(self, dc: DayChain, target: Mapping[str, float], extra: Optional[Marks] = None) -> List[Trade]:
        """Trade to ``target`` at today's marks; keys absent from ``target`` are closed.

        Zero-quantity changes (``is_zero``) are not traded. ``rolled`` records
        whether the instrument set changed (never on the first fill).
        """
        before = dict(self.positions)
        trades: List[Trade] = []
        marks: Dict[str, float] = {}
        for key in sorted(set(before) | set(target)):
            old = before.get(key, 0.0)
            new = float(target.get(key, 0.0))
            if key not in target and is_zero(old):
                continue
            price = self._price(dc, key, extra)
            dq = new - old
            if not is_zero(dq):
                trades.append(Trade(key, dq, price, trade_cost(key, dq, price)))
            if key in target:
                marks[key] = price
        self.positions = {k: float(q) for k, q in target.items()}
        self._marks = marks
        self._last_day = dc.day
        self.rolled = bool(before) and set(before) != set(self.positions)
        return trades

    def unwind(self, dc: DayChain, extra: Optional[Marks] = None) -> List[Trade]:
        """Close everything at today's marks (note ended or last data day)."""
        trades = self.rebalance(dc, {}, extra)
        self.rolled = False
        return trades


class BookA(Book):
    """Single far IM: the longest listed contract, spot-neutral ``S/F`` sizing."""

    name = "A"

    def target(self, dc: DayChain, g: DayGreeks) -> Dict[str, float]:
        far = dc.im[-1]
        return {far.code: g.spot_delta * dc.spot / (IM_MULT * far.close)}


class BookB(Book):
    """Pillar lots on every curve node plus the pinned-forward delta on the far IM."""

    name = "B"

    def target(self, dc: DayChain, g: DayGreeks) -> Dict[str, float]:
        n_im = len(dc.im)
        if len(g.pillar_delta) != n_im + 1:
            raise ValidationError(
                f"{dc.day}: {len(g.pillar_delta)} pillar deltas for {n_im} IM + 1 MO pillars")
        if not dc.mo.tenor > dc.im[-1].tenor:
            raise ValidationError(f"{dc.day}: MO pillar is not the last curve node; pillar order breaks")
        out: Dict[str, float] = {q.code: g.pillar_delta[i] / IM_MULT for i, q in enumerate(dc.im)}
        far = dc.im[-1]
        out[far.code] += g.pinned_delta * dc.spot / (IM_MULT * far.close)
        held = [k for k in self.positions if is_mo(k) and split_mo_key(k)[0] == dc.mo.series]
        if len(held) > 1:
            raise ValidationError(f"{dc.day}: several {dc.mo.series} pairs held: {held}")
        key = held[0] if held else mo_key(dc.mo.series, dc.mo.strike)
        out[key] = g.pillar_delta[n_im] / (MO_MULT * float(safe_exp(-RATE * dc.mo.tenor)))
        return out
