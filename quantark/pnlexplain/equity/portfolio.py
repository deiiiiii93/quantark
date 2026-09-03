"""Position and portfolio layers (spec §9)."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from quantark.pnlexplain.base import (
    ExplainMethod, ExplainRow, Factor, PnLExplainResult, RowKind, component_sum, make_total_row,
    rows_to_frame,
)
from quantark.pnlexplain.config import PnLExplainConfig
from quantark.pnlexplain.equity.explain import explain
from quantark.pnlexplain.equity.lifecycle import LifecycleTransition
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot, is_terminal, value
from quantark.pnlexplain.equity.trades import ExplainTrade
from quantark.util.exceptions import NumericalError, ValidationError
from quantark.util.numerical import is_close

_EMPTY: Mapping = MappingProxyType({})


@dataclass(frozen=True)
class PositionSnapshot:
    position_id: str
    underlying: str
    snapshot: ValuationSnapshot
    tombstone: bool = False


def _finite_number(value: Any, what: str) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{what} must be a number, got {value!r}") from None
    if not math.isfinite(f):
        raise ValidationError(f"{what} must be finite, got {value!r}")
    return f


@dataclass(frozen=True)
class QuotedLegSnapshot:
    """A hedge leg valued at a quoted price (units x price); no engine.

    ``units`` may be zero only for a tombstone (a leg opened and closed within
    the step); ``spot`` must be positive because spot returns divide by it.
    """
    position_id: str
    underlying: str
    units: float
    price: float
    spot: float
    date: datetime
    tombstone: bool = False

    def __post_init__(self) -> None:
        if not self.position_id:
            raise ValidationError("quoted leg requires a position_id")
        units = _finite_number(self.units, "quoted leg units")
        price = _finite_number(self.price, "quoted leg price")
        spot = _finite_number(self.spot, "quoted leg spot")
        if spot <= 0.0:
            raise ValidationError(f"quoted leg spot must be positive, got {self.spot!r}")
        if units == 0.0 and not self.tombstone:
            raise ValidationError("a live quoted leg must hold a non-zero number of units")
        if not math.isfinite(units * price):
            raise NumericalError(f"quoted leg value overflows: {units} x {price}")
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "spot", spot)

    @property
    def total(self) -> float:
        return self.units * self.price


@dataclass(frozen=True)
class BookSnapshot:
    date: datetime
    positions: Mapping[str, PositionSnapshot]
    environments: Mapping[str, Any]
    quoted_legs: Mapping[str, QuotedLegSnapshot] = field(default_factory=lambda: _EMPTY)
    currency: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "positions", MappingProxyType(dict(self.positions)))
        object.__setattr__(self, "environments", MappingProxyType(dict(self.environments)))
        object.__setattr__(self, "quoted_legs", MappingProxyType(dict(self.quoted_legs)))
        labels = set()
        for pid, ps in self.positions.items():
            if ps.position_id != pid:
                raise ValidationError(f"position key {pid!r} != position_id {ps.position_id!r}")
            if ps.snapshot.date != self.date:
                raise ValidationError(f"position {pid} snapshot date differs from the book date")
            if ps.snapshot.currency is not None:
                labels.add(ps.snapshot.currency)
        # A book is single-currency (spec §9): every label present must agree, with or
        # without a book-level label; one label present and no book label infers it.
        if self.currency is not None:
            labels.add(self.currency)
        if len(labels) > 1:
            raise ValidationError(f"mixed currencies in one book: {sorted(labels)}")
        if self.currency is None and labels:
            object.__setattr__(self, "currency", next(iter(labels)))
        for pid, leg in self.quoted_legs.items():
            if leg.position_id != pid:
                raise ValidationError(f"quoted leg key {pid!r} != position_id {leg.position_id!r}")
            if leg.date != self.date:
                raise ValidationError(f"quoted leg {pid} date differs from the book date")
        if set(self.positions) & set(self.quoted_legs):
            raise ValidationError("a position id cannot be both a priced position and a quoted leg")

    @classmethod
    def from_portfolio(cls, portfolio, date: datetime, *,
                       tombstones: Optional[Mapping[str, PositionSnapshot]] = None,
                       currency: Optional[str] = None) -> "BookSnapshot":
        """Snapshot the portfolio as it stands on `date` (post-event products; the alive
        contracts travel in transitions). Environments must already be dated `date`."""
        from copy import deepcopy
        envs = {u: deepcopy(env) for u, env in portfolio.pricing_environments.items()}
        for u, env in envs.items():
            if env.valuation_date != date:
                raise ValidationError(f"environment for {u} is dated {env.valuation_date}, book date is {date}")
        positions: Dict[str, PositionSnapshot] = {}
        for pid, pos in portfolio.positions.items():
            state = deepcopy(getattr(pos, "lifecycle_state", None))
            snap = ValuationSnapshot(product=pos.product, engine=pos.engine, pricing_env=envs[pos.underlying],
                                     date=date, quantity=float(pos.quantity), lifecycle_state=state,
                                     currency=currency, label=pid)
            positions[pid] = PositionSnapshot(position_id=pid, underlying=pos.underlying, snapshot=snap)
        for pid, tomb in (tombstones or {}).items():
            if pid in positions:
                raise ValidationError(f"tombstone {pid} collides with a live position")
            positions[pid] = tomb
        return cls(date=date, positions=positions, environments=envs, currency=currency)


@dataclass(frozen=True)
class PositionExplainResult:
    position_id: str
    underlying: str
    instrument: Optional[PnLExplainResult]
    trade_rows: Tuple[ExplainRow, ...]
    total_pnl: float
    rows: Tuple[ExplainRow, ...]
    coordinate: Optional[Tuple[Optional[float], Optional[float]]] = None

    def reconcile(self, method: ExplainMethod) -> float:
        return self.total_pnl - component_sum(self.rows, method)


@dataclass(frozen=True)
class PortfolioExplainResult:
    date_t0: datetime
    date_t1: datetime
    positions: Mapping[str, PositionExplainResult]
    rows: Tuple[ExplainRow, ...]
    total_pnl: float
    metadata: Mapping[str, Any]
    _by_underlying: Mapping[str, Tuple[ExplainRow, ...]] = field(default_factory=lambda: _EMPTY, repr=False)

    def reconcile(self, method: ExplainMethod) -> float:
        return self.total_pnl - component_sum(self.rows, method)

    def by_underlying(self) -> Mapping[str, Tuple[ExplainRow, ...]]:
        return self._by_underlying

    def to_frame(self) -> pd.DataFrame:
        frames = []
        for pid in self.positions:
            pr = self.positions[pid]
            frames.append(rows_to_frame(pr.rows, date=self.date_t1, position_id=pid, underlying=pr.underlying))
        for underlying, rows in self._by_underlying.items():
            frames.append(rows_to_frame(rows, date=self.date_t1, position_id="portfolio", underlying=underlying))
        frames.append(rows_to_frame(self.rows, date=self.date_t1, position_id="portfolio", underlying="*"))
        return pd.concat(frames, ignore_index=True) if frames else rows_to_frame([], date=self.date_t1)


def _validate_trades(position_id: str, trades: Sequence[ExplainTrade], t0: Optional[datetime],
                     t1: datetime) -> None:
    """Trades belong to the position and, when dated, fall in (t0, t1]; t0=None means an
    opening step with no lower bound (no naive sentinel that a tz-aware stamp cannot compare to)."""
    for t in trades:
        if t.position_id != position_id:
            raise ValidationError(f"trade for {t.position_id} passed to position {position_id}")
        if t.timestamp is None:
            continue
        try:
            inside = t.timestamp <= t1 and (t0 is None or t0 < t.timestamp)
        except TypeError:
            raise ValidationError(
                f"trade timestamp {t.timestamp!r} and the snapshot dates disagree on timezone-awareness"
            ) from None
        if not inside:
            raise ValidationError(f"trade timestamp {t.timestamp} outside ({t0}, {t1}]")


def _check_sum(trades: Sequence[ExplainTrade], expected: float, what: str) -> None:
    traded = sum(t.quantity for t in trades)
    if not is_close(traded, expected, rel_tol=0.0, abs_tol=1e-9 * max(1.0, abs(expected), abs(traded))):
        raise ValidationError(f"trades sum to {traded}, expected {expected} ({what})")


def _trade_rows(trades: Sequence[ExplainTrade], unit_pv_t1: float) -> Tuple[ExplainRow, ...]:
    rows = []
    for t in trades:
        rows.append(ExplainRow(
            factor=Factor.TRADE, term=f"trade:{t.kind}", method=ExplainMethod.SHARED, kind=RowKind.COMPONENT,
            level="position", pnl=t.quantity * (unit_pv_t1 - t.price),
            metadata={"kind": t.kind, "quantity": t.quantity, "price": t.price, "unit_pv_t1": unit_pv_t1,
                      "timestamp": t.timestamp.isoformat() if t.timestamp else None},
        ))
    return tuple(rows)


def _promote(instrument: PnLExplainResult) -> List[ExplainRow]:
    return [r.relabel("position") for r in instrument.rows if r.kind is not RowKind.SUMMARY]


def explain_position(
    pos_t0: Optional[PositionSnapshot],
    pos_t1: PositionSnapshot,
    *,
    trades: Sequence[ExplainTrade] = (),
    transition: Optional[LifecycleTransition] = None,
    config: Optional[PnLExplainConfig] = None,
) -> PositionExplainResult:
    trades = tuple(trades)
    pid, underlying = pos_t1.position_id, pos_t1.underlying
    s1 = pos_t1.snapshot
    if pos_t0 is None:
        if pos_t1.tombstone:
            raise ValidationError(f"position {pid}: a tombstone needs a t0 side")
        if not trades:
            raise ValidationError(f"position {pid} is new at t1 but no trades were supplied")
        _validate_trades(pid, trades, None, s1.date)
        _check_sum(trades, s1.quantity, "opened today")
        v1 = value(s1).total
        unit = v1 / s1.quantity
        trade_rows = _trade_rows(trades, unit)
        total = v1 + sum(t.cash for t in trades)
        rows = list(trade_rows) + [make_total_row("position", total)]
        return PositionExplainResult(pid, underlying, None, trade_rows, total, tuple(rows))

    s0 = pos_t0.snapshot
    if pos_t0.position_id != pid:
        raise ValidationError("pos_t0 and pos_t1 must share a position_id")
    if pos_t0.underlying != underlying:
        raise ValidationError(
            f"position {pid} switches underlying {pos_t0.underlying!r} -> {underlying!r}; "
            "a replacement is a close and an open under distinct ids"
        )
    _validate_trades(pid, trades, s0.date, s1.date)
    q0, q1 = s0.quantity, s1.quantity
    has_lifecycle = s0.lifecycle_state is not None or s1.lifecycle_state is not None
    terminal_t1 = is_terminal(s1.lifecycle_state)
    if has_lifecycle and trades:
        raise ValidationError(f"position {pid} is lifecycle-tracked; trades are not allowed")
    if has_lifecycle and q0 != q1:
        raise ValidationError(f"position {pid} is lifecycle-tracked; its quantity cannot change")
    if pos_t1.tombstone and not terminal_t1 and not trades:
        raise ValidationError(f"position {pid} disappeared without closing trades or a terminal state")
    if pos_t1.tombstone and terminal_t1 and trades:
        raise ValidationError(f"position {pid} terminated by lifecycle; trades are not allowed")

    if pos_t1.tombstone and not terminal_t1:
        # closed by trading: the trades must consume the WHOLE old position, and the
        # tombstone is priced at q0 whatever quantity it was written with
        _check_sum(trades, -q0, "closed by trading")
        s1_at_q0 = s1 if q1 == q0 else dataclasses.replace(s1, quantity=q0)
    elif q1 != q0:
        _check_sum(trades, q1 - q0, "quantity change")
        s1_at_q0 = dataclasses.replace(s1, quantity=q0)
    else:
        if trades:
            _check_sum(trades, 0.0, "unchanged quantity")
        s1_at_q0 = s1
    instrument = explain(s0, s1_at_q0, config=config, transition=transition)
    unit = instrument.pv_t1.total / q0
    trade_rows = _trade_rows(trades, unit) if trades else ()
    if pos_t1.tombstone and not terminal_t1:
        v1 = 0.0
    elif q1 != q0:
        v1 = instrument.pv_t1.total * (q1 / q0)
    else:
        v1 = instrument.pv_t1.total
    total = v1 - instrument.pv_t0.total + sum(t.cash for t in trades)
    rows = _promote(instrument) + list(trade_rows) + [make_total_row("position", total)]
    coord = instrument.metadata.get("coordinate")
    return PositionExplainResult(pid, underlying, instrument, trade_rows, total, tuple(rows), coord)


def explain_quoted_leg(leg_t0: Optional[QuotedLegSnapshot], leg_t1: QuotedLegSnapshot, *,
                       trades: Sequence[ExplainTrade] = ()) -> PositionExplainResult:
    trades = tuple(trades)
    pid = leg_t1.position_id
    rows: List[ExplainRow] = []

    def shared(factor, term, pnl, moves=None):
        return ExplainRow(factor=factor, term=term, method=ExplainMethod.SHARED, kind=RowKind.COMPONENT,
                          level="position", pnl=pnl, moves=moves or {})

    if leg_t0 is None:
        if not trades:
            raise ValidationError(f"quoted leg {pid} is new at t1 but no trades were supplied")
        _validate_trades(pid, trades, None, leg_t1.date)
        if leg_t1.tombstone:
            # opened AND closed within the step: nothing is held at t1, the round trip is the PnL
            if leg_t1.units != 0.0:
                raise ValidationError(
                    f"quoted leg {pid} is new at t1 and a tombstone: it must carry zero units"
                )
            _check_sum(trades, 0.0, "leg opened and closed within the step")
        else:
            _check_sum(trades, leg_t1.units, "leg opened today")
        trade_rows = _trade_rows(trades, leg_t1.price)
        v1 = 0.0 if leg_t1.tombstone else leg_t1.total
        total = v1 + sum(t.cash for t in trades)
        rows = list(trade_rows) + [make_total_row("position", total)]
        return PositionExplainResult(pid, leg_t1.underlying, None, trade_rows, total, tuple(rows))
    if leg_t0.position_id != pid:
        raise ValidationError("quoted legs must share a position_id (rolls use contract-specific ids)")
    if leg_t0.underlying != leg_t1.underlying:
        raise ValidationError(
            f"quoted leg {pid} switches underlying {leg_t0.underlying!r} -> {leg_t1.underlying!r}"
        )
    if not leg_t1.date > leg_t0.date:
        raise ValidationError(f"quoted leg {pid}: t1 date {leg_t1.date} must be after t0 date {leg_t0.date}")
    _validate_trades(pid, trades, leg_t0.date, leg_t1.date)
    if leg_t1.tombstone:
        _check_sum(trades, -leg_t0.units, "leg closed")
    else:
        _check_sum(trades, leg_t1.units - leg_t0.units, "leg quantity change")
    d_spot = leg_t1.spot - leg_t0.spot
    d_price = leg_t1.price - leg_t0.price
    rows.append(shared(Factor.SPOT, "spot", leg_t0.units * d_spot,
                       {"spot_return": d_spot / leg_t0.spot}))
    rows.append(shared(Factor.BASIS, "basis", leg_t0.units * (d_price - d_spot)))
    trade_rows = _trade_rows(trades, leg_t1.price)
    rows.extend(trade_rows)
    v1 = 0.0 if leg_t1.tombstone else leg_t1.total
    total = v1 - leg_t0.total + sum(t.cash for t in trades)
    rows.append(make_total_row("position", total))
    return PositionExplainResult(pid, leg_t1.underlying, None, trade_rows, total, tuple(rows))


def _aggregate(results: Sequence[PositionExplainResult], level_underlying: Optional[str]
               ) -> Tuple[ExplainRow, ...]:
    """Aggregate COMPONENT/INFORMATIONAL rows by (method, kind, factor, term); display
    columns survive only when every constituent shares one coordinate and step."""
    groups: Dict[Tuple, List[Tuple[ExplainRow, PositionExplainResult]]] = {}
    order: List[Tuple] = []
    for pr in results:
        for r in pr.rows:
            if r.kind is RowKind.SUMMARY:
                continue
            key = (r.method, r.kind, r.factor, r.term)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append((r, pr))
    out = []
    for key in order:
        items = groups[key]
        method, kind, factor, term = key
        pnl = sum(r.pnl for r, _ in items)
        coords = {(pr.underlying, pr.coordinate) for _, pr in items}
        steps = {r.step for r, _ in items}
        same = len(coords) == 1 and len(steps) == 1     # one underlying + one coordinate + one step
        greek = cash = None
        moves: Dict[str, float] = {}
        if same:
            if all(r.greek is not None for r, _ in items):
                greek = sum(r.greek for r, _ in items)
            if all(r.cash_greek is not None for r, _ in items):
                cash = sum(r.cash_greek for r, _ in items)
            moves = dict(items[0][0].moves)
        out.append(ExplainRow(
            factor=factor, term=term, method=method, kind=kind, level="portfolio", pnl=pnl, moves=moves,
            greek=greek, cash_greek=cash, step=steps.pop() if same else None,
            metadata={"positions": tuple(pr.position_id for _, pr in items),
                      "underlying": level_underlying or "*"},
        ))
    return tuple(out)


def explain_portfolio(
    book_t0: BookSnapshot,
    book_t1: BookSnapshot,
    *,
    trades: Sequence[ExplainTrade] = (),
    transaction_costs: float = 0.0,
    transitions: Mapping[str, LifecycleTransition] = _EMPTY,
    config: Optional[PnLExplainConfig] = None,
) -> PortfolioExplainResult:
    if book_t1.date <= book_t0.date:
        raise ValidationError("book_t1 must be dated after book_t0")
    if book_t0.currency is not None and book_t1.currency is not None and book_t0.currency != book_t1.currency:
        raise ValidationError("book currencies differ")
    by_pid: Dict[str, List[ExplainTrade]] = {}
    for t in trades:
        # every dated trade belongs to this step, whichever position it opens or adjusts
        if t.timestamp is not None and not (book_t0.date < t.timestamp <= book_t1.date):
            raise ValidationError(
                f"trade for {t.position_id} at {t.timestamp} is outside ({book_t0.date}, {book_t1.date}]"
            )
        by_pid.setdefault(t.position_id, []).append(t)
    results: Dict[str, PositionExplainResult] = {}
    ids = sorted(set(book_t0.positions) | set(book_t1.positions))
    for pid in ids:
        p0, p1 = book_t0.positions.get(pid), book_t1.positions.get(pid)
        if p1 is None:
            raise ValidationError(f"position {pid} is absent at t1 with no tombstone")
        results[pid] = explain_position(p0, p1, trades=by_pid.pop(pid, ()),
                                        transition=transitions.get(pid), config=config)
    leg_ids = sorted(set(book_t0.quoted_legs) | set(book_t1.quoted_legs))
    for pid in leg_ids:
        l0, l1 = book_t0.quoted_legs.get(pid), book_t1.quoted_legs.get(pid)
        if l1 is None:
            raise ValidationError(f"quoted leg {pid} is absent at t1 with no tombstone")
        results[pid] = explain_quoted_leg(l0, l1, trades=by_pid.pop(pid, ()))
    if by_pid:
        raise ValidationError(f"trades for unknown position ids: {sorted(by_pid)}")

    standalone_cost = _finite_number(transaction_costs, "transaction_costs")
    if standalone_cost < 0.0:
        raise ValidationError(f"transaction_costs must be >= 0, got {transaction_costs!r}")
    cost_total = math.fsum([t.transaction_cost for t in trades] + [standalone_cost])
    if not math.isfinite(cost_total):
        raise NumericalError("transaction cost total overflows")
    ordered = [results[pid] for pid in sorted(results)]
    per_underlying: Dict[str, Tuple[ExplainRow, ...]] = {}
    for u in sorted({pr.underlying for pr in ordered}):
        per_underlying[u] = _aggregate([pr for pr in ordered if pr.underlying == u], u)
    book_rows = list(_aggregate(ordered, None))
    book_rows.append(ExplainRow(factor=Factor.TRANSACTION_COST, term="transaction_cost", method=ExplainMethod.SHARED,
                                kind=RowKind.COMPONENT, level="portfolio", pnl=-cost_total))
    total = sum(pr.total_pnl for pr in ordered) - cost_total
    book_rows.append(make_total_row("portfolio", total))
    recon: Dict[str, Dict[str, Any]] = {}
    methods = (config or PnLExplainConfig()).methods
    for m in methods:
        explained = component_sum(book_rows, m)
        gap = total - explained
        recon[m.value] = {"expected": total, "explained": explained, "gap": gap,
                          "ok": bool(is_close(gap, 0.0, rel_tol=0.0, abs_tol=1e-8 * max(1.0, abs(total))))}
    return PortfolioExplainResult(
        date_t0=book_t0.date, date_t1=book_t1.date, positions=MappingProxyType(results),
        rows=tuple(book_rows), total_pnl=total,
        metadata={"reconciliation": recon, "transaction_costs": cost_total},
        _by_underlying=MappingProxyType(per_underlying),
    )
