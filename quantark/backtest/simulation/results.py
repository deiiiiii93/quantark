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
QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
DISTRIBUTION_MEASURES = ("terminal_pnl",) + MEASURE_COLUMNS


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


@dataclass(frozen=True)
class PairedComparison:
    """``variant - base`` per matched path, for ``terminal_pnl`` and every measure."""

    differences: pd.DataFrame
    n_paths: int
    path_fingerprint: Optional[str]

    def describe(self, measure: str) -> Dict[str, Any]:
        """Mean, median, std, share positive and paired t-statistic of one measure's differences."""
        if measure not in self.differences.columns or measure in ("path", "same_termination"):
            raise ValidationError(f"unknown paired measure {measure!r}")
        values = self.differences[measure].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {"measure": measure, "n": 0, "mean": None, "median": None, "std": None,
                    "share_positive": None, "t_stat": None}
        std = float(finite.std(ddof=1)) if finite.size > 1 else None
        mean = float(finite.mean())
        t_stat = mean / (std / math.sqrt(finite.size)) if std not in (None, 0.0) else None
        return {
            "measure": measure, "n": int(finite.size), "mean": mean, "median": float(np.median(finite)),
            "std": std, "share_positive": float((finite > 0.0).mean()), "t_stat": t_stat,
        }


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

    # -- reductions ----------------------------------------------------

    def distribution(self, measure: str, *, es_level: float, tail: str = "lower") -> Dict[str, Any]:
        """Moments, quantiles, expected shortfall and event frequencies of one measure over the paths.

        Expected shortfall is the mean of the ``es_level`` tail: the values
        at or below that quantile (``lower``, the loss tail of a P&L measure)
        or at or above the ``1 - es_level`` quantile (``upper``, for a
        cost-like measure).  NaN values are dropped and ``n`` counts what
        remains; the frequencies are over every path.
        """
        if measure not in DISTRIBUTION_MEASURES:
            raise ValidationError(f"unknown measure {measure!r}; one of {DISTRIBUTION_MEASURES}")
        if not 0.0 < float(es_level) < 1.0:
            raise ValidationError("es_level must lie strictly between 0 and 1")
        if tail not in ("lower", "upper"):
            raise ValidationError("tail must be 'lower' or 'upper'")
        summary = self.summary
        values = summary[measure].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        reasons = summary["termination_reason"]
        out: Dict[str, Any] = {
            "measure": measure, "n": int(finite.size), "n_paths": int(self.n_paths),
            "es_level": float(es_level), "tail": tail,
            "ko_frequency": float((reasons == "knock_out").mean()),
            "ki_frequency": float(summary["knocked_in"].astype(bool).mean()),
            "maturity_frequency": float((reasons == "maturity").mean()),
            "data_end_frequency": float((reasons == "data_end").mean()),
        }
        if finite.size == 0:
            out.update(mean=None, std=None, quantiles={}, expected_shortfall=None, share_positive=None)
            return out
        if tail == "lower":
            cut = float(np.quantile(finite, float(es_level)))
            tail_values = finite[finite <= cut]
        else:
            cut = float(np.quantile(finite, 1.0 - float(es_level)))
            tail_values = finite[finite >= cut]
        out.update(
            mean=float(finite.mean()),
            std=float(finite.std(ddof=1)) if finite.size > 1 else None,
            quantiles={f"q{int(round(p * 100)):02d}": float(v)
                       for p, v in zip(QUANTILES, np.quantile(finite, QUANTILES))},
            expected_shortfall=float(tail_values.mean()),
            share_positive=float((finite > 0.0).mean()),
        )
        return out

    # -- matched paths -------------------------------------------------

    def _matches(self, other: "EnsembleResults") -> bool:
        """Same paths, proved by the data: the market columns are bit-identical up to each path's last day.

        Beyond a path's last day the cube repeats its final values, and a
        batch that settled early froze earlier than the whole run did, so
        only the days a path actually ran are compared; the last days
        themselves must agree (the lifecycle depends on the path alone).
        """
        if self.n_paths != other.n_paths or not self.cube.dates.equals(other.cube.dates):
            return False
        if not np.array_equal(self.last_day, other.last_day):
            return False
        for name in ("spot", "volatility", "rate"):
            mine, theirs = getattr(self.cube, name), getattr(other.cube, name)
            for i in range(self.n_paths):
                end = int(self.last_day[i]) + 1
                if not np.array_equal(mine[i, :end], theirs[i, :end]):
                    return False
        return True

    def paired(self, base: "EnsembleResults") -> PairedComparison:
        """``self - base`` per path; the two runs must be on the same paths."""
        if not self._matches(base):
            raise ValidationError(
                "paired comparison needs the same paths on both sides: same count, calendar and market columns"
            )
        columns = ["terminal_pnl", *MEASURE_COLUMNS]
        diff = self.summary[columns].to_numpy(dtype=float) - base.summary[columns].to_numpy(dtype=float)
        frame = pd.DataFrame(diff, columns=columns)
        frame.insert(0, "path", self.summary["path"].to_numpy())
        frame["same_termination"] = (
            self.summary["termination_reason"].to_numpy() == base.summary["termination_reason"].to_numpy()
        )
        mine, theirs = self.manifest.get("path_fingerprint"), base.manifest.get("path_fingerprint")
        return PairedComparison(differences=frame, n_paths=int(self.n_paths),
                                path_fingerprint=mine if mine == theirs else None)

    def take(self, indices: Sequence[int]) -> "EnsembleResults":
        """The sub-run of the given paths, in the given order; logs renumbered, manifest annotated."""
        idx = [int(i) for i in indices]
        if not idx or any(not 0 <= i < self.n_paths for i in idx):
            raise ValidationError(f"path indices {idx} out of range for {self.n_paths} paths")
        position = {old: new for new, old in enumerate(idx)}
        cube = StateCube(self.cube.dates, len(idx))
        cube.active_contract = list(self.cube.active_contract)
        for name in FLOAT_COLUMNS + BOOL_COLUMNS:
            setattr(cube, name, np.array(getattr(self.cube, name)[idx]))
        trades = [{**t, "path": position[int(t["path"])]} for t in self.trades if int(t["path"]) in position]
        events = [LifecycleRecord(**{**e.__dict__, "path": position[e.path]})
                  for e in self.events if e.path in position]
        manifest = {**self.manifest, "path_indices": idx}
        return EnsembleResults(cube=cube, trades=trades, events=events, manifest=manifest,
                               last_day=np.array(self.last_day[idx]),
                               initial_book_value=np.array(self.initial_book_value[idx]))
