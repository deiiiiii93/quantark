"""
Result container for OTC autocallable backtests.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


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

