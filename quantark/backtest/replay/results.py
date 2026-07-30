"""
Result container for OTC autocallable backtests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from quantark.util.io import atomic_write_json


class AutocallableBacktestResults:
    """DataFrame-oriented backtest result container."""

    def __init__(
        self,
        *,
        config: Any,
        states: list[dict[str, Any]],
        greeks: list[dict[str, Any]],
        rebalances: list[dict[str, Any]],
        trades: list[dict[str, Any]],
        actions: list[dict[str, Any]],
        surfaces: list[dict[str, Any]],
        daily_event_summary: list[dict[str, Any]],
        event_probabilities: list[dict[str, Any]],
        calibration_records: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self.config = config
        self._states = states
        self._greeks = greeks
        self._rebalances = rebalances
        self._trades = trades
        self._actions = actions
        self._surfaces = surfaces
        self._daily_event_summary = daily_event_summary
        self._event_probabilities = event_probabilities
        self._calibration_records = [dict(r) for r in (calibration_records or [])]

    @staticmethod
    def _frame(rows: list[dict[str, Any]], index: str | None = None) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        if index is not None and index in df.columns:
            df[index] = pd.to_datetime(df[index])
            df = df.set_index(index)
        return df

    @property
    def states_df(self) -> pd.DataFrame:
        return self._frame(self._states, "date")

    @property
    def greeks_df(self) -> pd.DataFrame:
        return self._frame(self._greeks, "date")

    @property
    def rebalance_df(self) -> pd.DataFrame:
        return self._frame(self._rebalances, "date")

    @property
    def trades_df(self) -> pd.DataFrame:
        return self._frame(self._trades, "date")

    @property
    def actions_df(self) -> pd.DataFrame:
        return self._frame(self._actions, "date")

    @property
    def surfaces_df(self) -> pd.DataFrame:
        return self._frame(self._surfaces, "date")

    @property
    def daily_event_summary_df(self) -> pd.DataFrame:
        return self._frame(self._daily_event_summary, "date")

    @property
    def event_probability_df(self) -> pd.DataFrame:
        return self._frame(self._event_probabilities, "date")

    @property
    def calibration_records(self) -> list[dict[str, Any]]:
        """Per-day vol-model calibration records (empty for BSM runs).

        One entry per priced day: date, variant, surface_date, surface_sha,
        cache_hit, calibration_seconds, pricing_seconds, plus the
        variant-specific calibration record (LV stats / Heston params +
        fit metrics / SLV leverage stats).
        """
        return [dict(r) for r in self._calibration_records]

    # ------------------------------------------------------------------
    # BaseBacktestResults interface (thin adapters over the *_df frames),
    # so OTC results are interchangeable with equity/FI results.
    # ------------------------------------------------------------------

    def get_total_pnl(self) -> float:
        """Total P&L (product + hedge mark-to-market, net of costs)."""
        states = self.states_df
        if states.empty or "total_pnl" not in states.columns:
            return 0.0
        return float(states["total_pnl"].iloc[-1])

    def get_total_return(self) -> float:
        """Total P&L as a fraction of the initial portfolio value."""
        states = self.states_df
        if states.empty or "portfolio_value" not in states.columns:
            return 0.0
        initial = float(states["portfolio_value"].iloc[0])
        if initial == 0.0:
            return 0.0
        return self.get_total_pnl() / initial

    def get_pnl_series(self) -> pd.Series:
        """P&L time series, indexed by date."""
        states = self.states_df
        if states.empty or "total_pnl" not in states.columns:
            return pd.Series(dtype=float)
        return states["total_pnl"]

    def get_value_series(self) -> pd.Series:
        """Portfolio value time series, indexed by date."""
        states = self.states_df
        if states.empty or "portfolio_value" not in states.columns:
            return pd.Series(dtype=float)
        return states["portfolio_value"]

    def get_hedge_trades(self) -> pd.DataFrame:
        """Executed futures hedge trades."""
        return self.trades_df

    def get_lifecycle_events(self) -> pd.DataFrame:
        """Realized lifecycle events (KO/KI/coupon/maturity), one row each.

        Adapts the OTC action log to the shared lifecycle-events schema by
        exposing the event kind under an ``event_type`` column (matching the
        equity engine's ``get_lifecycle_events``).
        """
        actions = self.actions_df
        if actions.empty:
            return actions
        if "action_type" in actions.columns:
            actions = actions.rename(columns={"action_type": "event_type"})
        return actions

    def get_summary(self) -> dict[str, Any]:
        states = self.states_df
        if states.empty:
            return {"num_days": 0, "total_pnl": 0.0}
        return {
            "num_days": int(len(states)),
            "start_date": states.index.min(),
            "end_date": states.index.max(),
            "initial_portfolio_value": float(states["portfolio_value"].iloc[0]),
            "final_portfolio_value": float(states["portfolio_value"].iloc[-1]),
            "total_pnl": float(states["total_pnl"].iloc[-1]),
            "num_trades": int(len(self._trades)),
            "num_actions": int(len(self._actions)),
        }

    def export_to_excel(self, filepath: str) -> None:
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            self.states_df.to_excel(writer, sheet_name="States")
            self.greeks_df.to_excel(writer, sheet_name="Greeks")
            self.rebalance_df.to_excel(writer, sheet_name="Rebalances")
            self.trades_df.to_excel(writer, sheet_name="Trades")
            self.actions_df.to_excel(writer, sheet_name="Actions")
            self.daily_event_summary_df.to_excel(writer, sheet_name="DailyEvents")
            self.event_probability_df.to_excel(writer, sheet_name="EventProb")
            if not self.surfaces_df.empty:
                self.surfaces_df.to_excel(writer, sheet_name="Surfaces")

    def export_surfaces_to_parquet(self, filepath: str) -> None:
        self.surfaces_df.to_parquet(filepath)

    def export_calibration_records(self, filepath: str) -> None:
        """Persist per-day calibration records as one JSON file per run.

        The payload is the bare list of per-day records (date, variant,
        surface_sha, timings, ...).  Written atomically (tmp + os.replace).
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, self.calibration_records)


class BookBacktestResults:
    def __init__(self, *, config, states, greeks, rebalances, trades, actions,
                 daily_event_summary, event_probabilities, surfaces, products_meta):
        self.config = config
        self._states = states
        self._greeks = greeks
        self._rebalances = rebalances
        self._trades = trades
        self._actions = actions
        self._daily_event_summary = daily_event_summary
        self._event_probabilities = event_probabilities
        self._surfaces = surfaces
        self._products_meta = products_meta

    @staticmethod
    def _frame(rows, index=None):
        df = pd.DataFrame(rows)
        if index and not df.empty:
            df = df.set_index(index)
        return df

    def states_df(self): return self._frame(self._states)
    def greeks_df(self): return self._frame(self._greeks)
    def rebalances_df(self): return self._frame(self._rebalances)
    def trades_df(self): return self._frame(self._trades)
    def actions_df(self): return self._frame(self._actions)
    def daily_event_summary_df(self): return self._frame(self._daily_event_summary)
    def event_probability_df(self): return self._frame(self._event_probabilities)
    def surfaces_df(self): return self._frame(self._surfaces)

    def get_summary(self):
        states = self.states_df()
        if states.empty:
            return {"num_days": 0, "num_trades": len(self._trades), "total_pnl": 0.0,
                    "num_products": len(self._products_meta), "num_lifecycle_events": len(self._actions)}
        return {
            "num_days": int(len(states)),
            "start_date": str(states["date"].iloc[0]),
            "end_date": str(states["date"].iloc[-1]),
            "initial_portfolio_value": float(states["portfolio_value"].iloc[0]),
            "final_portfolio_value": float(states["portfolio_value"].iloc[-1]),
            "total_pnl": float(states["total_pnl"].iloc[-1]),
            "product_pnl": float(states["product_pnl"].iloc[-1]),
            "hedge_pnl": float(states["hedge_pnl"].iloc[-1]),
            "transaction_costs": float(states["transaction_costs"].iloc[-1]),
            "num_trades": int(len(self._trades)),
            "num_products": len(self._products_meta),
            "num_lifecycle_events": len(self._actions),
        }
