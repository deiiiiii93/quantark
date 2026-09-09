"""What a run returns: the state cube, the logs, the manifest, and the answers (spec 9).

``summary`` is one row per path in the historical study's measures;
``distribution`` and ``paired`` reduce it; ``to_dir`` / ``from_dir``
persist a run.  Nothing here prices anything.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from quantark.util.exceptions import ValidationError

from .lifecycle import LifecycleRecord
from .measures import MEASURE_COLUMNS, path_measures

FLOAT_COLUMNS = (
    "portfolio_value", "product_mtm", "hedge_mtm", "cash", "cashflows", "transaction_costs",
    "product_pnl", "hedge_pnl", "total_pnl", "spot", "volatility", "rate", "pricing_q",
    "implied_q", "basis_yield", "futures_price", "futures_contracts", "pre_hedge_contracts",
    "delta", "gamma", "pending_receivable_pv",
)
BOOL_COLUMNS = ("alive", "knocked_in", "knocked_out", "matured", "settled")

TRADE_COLUMNS = [
    "path", "day", "date", "trade_type", "contract", "quantity", "price",
    "multiplier", "notional", "transaction_cost", "reason",
]
EVENT_COLUMNS = ("product", "path", "day", "event", "index", "spot", "barrier", "cashflow")
#: ``days`` is the first measure, so it is not repeated in the head.
SUMMARY_COLUMNS: Tuple[str, ...] = (
    "path", "termination_reason", "knocked_in", "ko_observation_index", "terminal_pnl",
) + MEASURE_COLUMNS


class StateCube:
    """The replay's state columns that have a per-path meaning (spec 4.3)."""

    def __init__(self, dates: pd.DatetimeIndex, n_paths: int) -> None:
        self.dates = pd.DatetimeIndex(dates)
        self.n_paths = int(n_paths)
        self.active_contract: List[str] = []
        shape = (self.n_paths, len(self.dates))
        for name in FLOAT_COLUMNS:
            setattr(self, name, np.zeros(shape))
        for name in BOOL_COLUMNS:
            setattr(self, name, np.zeros(shape, dtype=bool))

    def frame(self, i: int, last_day: int) -> pd.DataFrame:
        """One path's state rows up to and including ``last_day``."""
        end = int(last_day) + 1
        data: Dict[str, Any] = {
            "date": self.dates[:end],
            "active_contract": self.active_contract[:end],
        }
        for name in FLOAT_COLUMNS + BOOL_COLUMNS:
            data[name] = getattr(self, name)[i, :end]
        return pd.DataFrame(data)

    def freeze_from(self, day: int) -> None:
        """Repeat day ``day``'s values over the rest of the calendar.

        The loop stops once every path has settled; a settled path's columns
        repeat its terminal values, which is the per-path reading of the
        replay's ``terminate_on_lifecycle_end``.
        """
        for name in FLOAT_COLUMNS + BOOL_COLUMNS:
            column = getattr(self, name)
            column[:, day + 1:] = column[:, day: day + 1]
        if self.active_contract:
            self.active_contract += [self.active_contract[-1]] * (
                len(self.dates) - len(self.active_contract)
            )


@dataclass
class EnsembleResults:
    """The run's cube, event logs and manifest, and the answers built from them."""

    cube: StateCube
    trades: List[Dict[str, Any]]
    events: List[LifecycleRecord]
    manifest: Dict[str, Any]
    last_day: np.ndarray
    initial_book_value: np.ndarray
    _summary: Optional[pd.DataFrame] = field(default=None, init=False, repr=False, compare=False)

    @property
    def n_paths(self) -> int:
        return int(self.cube.n_paths)

    def path_states(self, i: int) -> pd.DataFrame:
        """One path's daily state rows, in the replay's schema, to its last day."""
        return self.cube.frame(i, int(self.last_day[i]))

    def path_trades(self, i: int) -> pd.DataFrame:
        rows = [t for t in self.trades if t["path"] == i]
        return pd.DataFrame(rows, columns=TRADE_COLUMNS)

    def path_events(self, i: int) -> pd.DataFrame:
        rows = [e.__dict__ for e in self.events if e.path == i]
        return pd.DataFrame(rows, columns=list(EVENT_COLUMNS))

    # -- the summary ---------------------------------------------------

    @property
    def notional(self) -> float:
        """The book's unit notional, ``sum |quantity| * initial_price * contract_multiplier``."""
        value = self.manifest.get("book_notional")
        if value is None:
            raise ValidationError("the manifest carries no book_notional; these results predate plan 4")
        return float(value)

    @property
    def summary(self) -> pd.DataFrame:
        """One row per path: termination, flags, terminal P&L and the hedge measures (memoised)."""
        if self._summary is None:
            self._summary = self._build_summary()
        return self._summary

    def _build_summary(self) -> pd.DataFrame:
        notional = self.notional
        terminal: Dict[int, Tuple[str, int]] = {}
        for e in self.events:                      # day order: the LAST terminal event wins
            if e.event in ("knock_out", "maturity"):
                terminal[int(e.path)] = (e.event, int(e.index))
        rows = []
        for i in range(self.n_paths):
            last = int(self.last_day[i])
            reason, ko_index = terminal.get(i, ("data_end", -1))
            row = {
                "path": i, "termination_reason": reason,
                "knocked_in": bool(self.cube.knocked_in[i, last]),
                "ko_observation_index": int(ko_index) if reason == "knock_out" else -1,
                "terminal_pnl": float(self.cube.total_pnl[i, last]),
            }
            row.update(path_measures(self.path_states(i), self.path_trades(i), notional=notional))
            rows.append(row)
        return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))
