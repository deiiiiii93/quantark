"""Backtest recorders: one explain row set per day (spec §10)."""
from __future__ import annotations

import math
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from quantark.asset.equity.lifecycle.cashflows import ValuationPoint
from quantark.pnlexplain.base import ExplainMethod, component_sum, rows_to_frame
from quantark.pnlexplain.config import PnLExplainConfig
from quantark.pnlexplain.equity.lifecycle import LifecycleTransition
from quantark.pnlexplain.equity.portfolio import (
    BookSnapshot, PortfolioExplainResult, PositionSnapshot, explain_portfolio,
)
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot, is_terminal
from quantark.pnlexplain.equity.trades import ExplainTrade
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close, is_zero

RECON_COLUMNS = ["date", "method", "level", "position_id", "expected", "explained", "gap", "ok",
                 "expected_states", "gap_states"]
# every trade type the two executors and the replay engine emit; anything else is an error
_KIND_MAP = {"open": "open", "adjust": "adjust", "close": "close", "roll_close": "roll_close",
             "roll_open": "roll_open", "hedge": "adjust", "hedge_rebalance": "adjust", "hedge_close": "close"}


def trade_kind(trade_type: str) -> str:
    try:
        return _KIND_MAP[trade_type]
    except KeyError:
        raise ValidationError(
            f"unknown backtest trade type {trade_type!r}; known {sorted(_KIND_MAP)}"
        ) from None


def _midnight(ts) -> datetime:
    return pd.Timestamp(ts).normalize().to_pydatetime()


def _finite(value: Any, what: str) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"trade record {what} is not a number: {value!r}") from None
    if not math.isfinite(f):
        raise ValidationError(f"trade record {what} is not finite: {value!r}")
    return f


def _is_unrolled_float_contract(product_alive: Any, product_held: Any) -> bool:
    """True when the engine hands back the SAME float-maturity contract, unrolled."""
    if product_alive is not product_held:
        return False                      # a tracker rolled it (a new object)
    if getattr(product_held, "maturity", None) is None:
        return False
    return getattr(product_held, "exercise_date", None) is None \
        and getattr(product_held, "maturity_date", None) is None


def _state_point(state: Any, ts: datetime, base_date: Optional[datetime]) -> Optional[ValuationPoint]:
    """The ledger valuation point of a lifecycle state on ``ts``.

    Float-schedule products keep their ledger in contract time; the manager
    values them at ``(date - base_date) / 365`` and so does this recorder.
    Date-based (or unobserved) states read at the snapshot date (None).
    """
    vp = getattr(state, "valuation_point", None)
    if vp is None or vp.time is None:
        return None
    if base_date is None:
        raise ValidationError("a contract-time lifecycle state needs the manager's base_date")
    elapsed = (pd.Timestamp(ts).normalize() - pd.Timestamp(base_date).normalize()).days
    return ValuationPoint(time=max(0.0, elapsed / 365.0))


class _RowSink:
    def __init__(self) -> None:
        self.frames: List[pd.DataFrame] = []
        self.recon: List[Dict[str, Any]] = []

    def add(self, date: datetime, result: PortfolioExplainResult, methods: Sequence[ExplainMethod],
            expected_states: float) -> None:
        self.frames.append(result.to_frame())
        for m in methods:
            explained = component_sum(result.rows, m)
            gap = result.total_pnl - explained
            self.recon.append({
                "date": pd.Timestamp(date), "method": m.value, "level": "portfolio", "position_id": "portfolio",
                "expected": result.total_pnl, "explained": explained, "gap": gap,
                "ok": bool(is_close(gap, 0.0, rel_tol=0.0, abs_tol=1e-8 * max(1.0, abs(result.total_pnl)))),
                "expected_states": expected_states, "gap_states": expected_states - explained,
            })
            for pid, pr in result.positions.items():
                pexp = component_sum(pr.rows, m)
                pgap = pr.total_pnl - pexp
                self.recon.append({
                    "date": pd.Timestamp(date), "method": m.value, "level": "position", "position_id": pid,
                    "expected": pr.total_pnl, "explained": pexp, "gap": pgap,
                    "ok": bool(is_close(pgap, 0.0, rel_tol=0.0, abs_tol=1e-8 * max(1.0, abs(pr.total_pnl)))),
                    "expected_states": float("nan"), "gap_states": float("nan"),
                })

    def result_frames(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.frames:
            explain_df = pd.concat(self.frames, ignore_index=True)
        else:
            explain_df = rows_to_frame([], date=datetime(1970, 1, 1)).iloc[0:0]
        recon = pd.DataFrame(self.recon, columns=RECON_COLUMNS)
        if not self.recon:
            recon = recon.astype({"expected": float, "explained": float, "gap": float,
                                  "expected_states": float, "gap_states": float, "ok": bool})
        return explain_df, recon


def _trade_from_record(rec: Any) -> Optional[ExplainTrade]:
    """Normalise one backtest TradeRecord; fails closed on anything malformed.

    Only an explicit zero-quantity "no trade" record is skipped. Every non-zero
    record must carry a position id, a finite price and a finite quantity.
    """
    quantity = _finite(getattr(rec, "quantity", None), "quantity")
    if quantity == 0.0:
        return None                       # the executors' zero-quantity "no trade" records
    position_id = getattr(rec, "position_id", None)
    if not position_id:
        raise ValidationError(
            f"trade record of type {getattr(rec, 'trade_type', None)!r} with quantity {quantity} "
            "carries no position_id"
        )
    price = _finite(getattr(rec, "price", None), "price")
    cost = _finite(getattr(rec, "transaction_cost", 0.0), "transaction_cost")
    return ExplainTrade(position_id=str(position_id), quantity=quantity, price=price,
                        transaction_cost=cost, timestamp=_midnight(rec.timestamp),
                        kind=trade_kind(getattr(rec, "trade_type", None)),
                        instrument_type=str(getattr(rec, "instrument_type", "") or ""))


class PnLExplainRecorder:
    """Hook for quantark.backtest.equity.BacktestEngine."""

    def __init__(self, config: PnLExplainConfig) -> None:
        self.config = config
        self._sink = _RowSink()
        self._prev_book: Optional[BookSnapshot] = None
        self._prev_costs = 0.0
        self._prev_state_pnl = 0.0
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._carried: Dict[str, PositionSnapshot] = {}

    @staticmethod
    def _base_date(engine: Any) -> Optional[datetime]:
        manager = getattr(engine, "lifecycle_manager", None)
        return None if manager is None else manager.base_date

    # -- day boundaries -------------------------------------------------
    def begin_day(self, engine: Any, timestamp: datetime) -> None:
        """After _update_pricing_environment, BEFORE lifecycle processing."""
        self._pending = {}
        if self._prev_book is None:
            return
        portfolio = engine.portfolio
        manager = engine.lifecycle_manager
        products = (manager.pricing_products(portfolio, timestamp) if manager is not None
                    else {pid: p.product for pid, p in portfolio.positions.items()})
        for pid, pos in portfolio.positions.items():
            self._pending[pid] = {
                "product_alive": products[pid], "engine_alive": pos.engine, "underlying": pos.underlying,
                "quantity": float(pos.quantity), "state_ref": getattr(pos, "lifecycle_state", None),
                "state_before": deepcopy(getattr(pos, "lifecycle_state", None)),
                # The equity BacktestEngine reprices an untracked float-maturity product with a
                # CONSTANT time to maturity (it never rolls it). Declare that explicitly so the
                # time row explains what was booked instead of guessing a roll (spec §8).
                "roll_days": 0 if _is_unrolled_float_contract(products[pid], pos.product) else None,
            }

    def end_day(self, engine: Any, timestamp: datetime, trade_records: Sequence[Any],
                cumulative_costs: float, lifecycle_events: Sequence[Any], state_pnl: float) -> None:
        """After _record_state."""
        ts = _midnight(timestamp)
        base = self._base_date(engine)
        portfolio = engine.portfolio
        envs = {u: deepcopy(env) for u, env in portfolio.pricing_environments.items()}
        live: Dict[str, PositionSnapshot] = {}
        for pid, pos in portfolio.positions.items():
            state = deepcopy(getattr(pos, "lifecycle_state", None))
            snap = ValuationSnapshot(product=pos.product, engine=pos.engine, pricing_env=envs[pos.underlying],
                                     date=ts, quantity=float(pos.quantity), lifecycle_state=state,
                                     valuation_point=_state_point(state, ts, base), label=pid)
            live[pid] = PositionSnapshot(pid, pos.underlying, snap)
        if self._prev_book is None:
            self._prev_book = BookSnapshot(date=ts, positions=live, environments=envs)
            self._prev_costs, self._prev_state_pnl = float(cumulative_costs), float(state_pnl)
            return

        trades = [t for t in (_trade_from_record(r) for r in trade_records) if t is not None]
        traded_ids = {t.position_id for t in trades}
        events_by_pid: Dict[str, List[Any]] = {}
        for item in lifecycle_events:
            events_by_pid.setdefault(item.position_id, []).append(item.event)

        transitions: Dict[str, LifecycleTransition] = {}
        tombstones: Dict[str, PositionSnapshot] = {}
        for pid, pend in self._pending.items():
            state_after = deepcopy(pend["state_ref"]) if pend["state_ref"] is not None else None
            transitions[pid] = LifecycleTransition(
                product_alive_t1=pend["product_alive"], engine_alive_t1=pend["engine_alive"],
                state_before=pend["state_before"], state_after=state_after,
                events=tuple(events_by_pid.get(pid, ())), contract_roll_days=pend["roll_days"])
            if pid in live:
                continue
            env = envs[pend["underlying"]]
            if is_terminal(state_after):
                snap = ValuationSnapshot(pend["product_alive"], pend["engine_alive"], env, date=ts,
                                         quantity=pend["quantity"], lifecycle_state=state_after,
                                         valuation_point=_state_point(state_after, ts, base), label=pid)
            elif pid in traded_ids:
                snap = ValuationSnapshot(pend["product_alive"], pend["engine_alive"], env, date=ts,
                                         quantity=pend["quantity"], lifecycle_state=None, label=pid)
            else:
                raise ValidationError(f"position {pid} left the book without a terminal state or closing trades")
            tombstones[pid] = PositionSnapshot(pid, pend["underlying"], snap, tombstone=True)
        for pid, old in self._carried.items():          # terminal tombstones still awaiting cash
            if pid in tombstones or pid in live:
                continue
            snap = old.snapshot
            state = deepcopy(snap.lifecycle_state)
            tombstones[pid] = PositionSnapshot(pid, old.underlying, ValuationSnapshot(
                snap.product, snap.engine, envs[old.underlying], date=ts, quantity=snap.quantity,
                lifecycle_state=state, valuation_point=_state_point(state, ts, base), label=pid),
                tombstone=True)

        book_t1 = BookSnapshot(date=ts, positions={**live, **tombstones}, environments=envs)
        day_costs = float(cumulative_costs) - self._prev_costs
        extra_costs = day_costs - sum(t.transaction_cost for t in trades)
        if is_zero(extra_costs):
            extra_costs = 0.0
        result = explain_portfolio(self._prev_book, book_t1, trades=trades, transaction_costs=extra_costs,
                                   transitions=transitions, config=self.config)
        self._sink.add(ts, result, self.config.methods, float(state_pnl) - self._prev_state_pnl)

        # tomorrow's t0: live positions + terminal tombstones with pending cash
        carried: Dict[str, PositionSnapshot] = {}
        for pid, tomb in tombstones.items():
            st = tomb.snapshot.lifecycle_state
            if st is not None and is_terminal(st) and st.ledger.pending(tomb.snapshot.point):
                carried[pid] = tomb
        self._carried = carried
        self._prev_book = BookSnapshot(date=ts, positions={**live, **carried}, environments=envs)
        self._prev_costs, self._prev_state_pnl = float(cumulative_costs), float(state_pnl)

    def frames(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return self._sink.result_frames()
