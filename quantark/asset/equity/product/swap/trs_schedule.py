"""Schedule manager and margin account for Total Return Swaps."""

from typing import Any, Dict

import pandas as pd

from quantark.util.numerical import safe_divide
from quantark.asset.equity.product.swap.trs_params import TRSParams, SwapState


class MarginAccount:
    """Tracks margin balances and the cashflow history of a TRS."""

    def __init__(self, margin_params):
        self.initial_margin = margin_params.initial_margin
        self.outstanding_margin = margin_params.outstanding_margin
        self.margin_events = margin_params.margin_events
        self.settle_type = margin_params.settle_type
        self.history = pd.DataFrame(columns=["date", "amount", "type", "balance"])

        if self.initial_margin > 0:
            # Record the opening balance as the running outstanding margin so the
            # ledger's balance column reconciles with subsequent apply_cashflow
            # calls (which start from self.outstanding_margin).
            self._add_history_record(
                None, self.initial_margin, "INITIAL", self.outstanding_margin
            )

    def _add_history_record(self, date, amount, event_type, balance):
        """Append a row to the margin cashflow history."""
        self.history = pd.concat(
            [
                self.history,
                pd.DataFrame(
                    [
                        {
                            "date": date,
                            "amount": amount,
                            "type": event_type,
                            "balance": balance,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    def apply_cashflow(self, date, amount, event_type):
        """Apply a cashflow to the outstanding margin and record it."""
        self.outstanding_margin += amount
        self._add_history_record(date, amount, event_type, self.outstanding_margin)

    def get_current_state(self) -> Dict[str, Any]:
        """Return a snapshot of the margin account."""
        return {
            "initial_margin": self.initial_margin,
            "outstanding_margin": self.outstanding_margin,
            "history": self.history,
        }


class ScheduleManager:
    """Builds and aligns the notional/quantity schedules of a TRS."""

    def __init__(self, trs_params: TRSParams):
        self.params = trs_params
        self.notional_schedule = pd.DataFrame()
        self.asset_quantity_schedule = pd.Series(dtype=float)
        self.asset_initial_price_schedule = pd.Series(dtype=float)
        self.redeem_quantity_schedule = pd.Series(dtype=float)
        self.redeem_int_schedule = pd.Series(dtype=float)
        self.outstanding_margin_schedule = pd.Series(dtype=float)
        self.merged_schedule = pd.DataFrame()
        self.state = SwapState.INITIATED

    def create_notional_schedule(self, engine) -> pd.DataFrame:
        """Build the per-date notional schedule using the pricing engine."""
        notional_list = engine.create_notional_schedule(self.params)

        self.notional_schedule = pd.DataFrame(notional_list)

        dates = [item["date"] for item in notional_list]
        self.asset_quantity_schedule = pd.Series(
            [item["asset_quantity"] for item in notional_list], index=dates
        )
        self.asset_initial_price_schedule = pd.Series(
            [item["asset_initial_price"] for item in notional_list], index=dates
        )
        self.redeem_quantity_schedule = pd.Series(
            [
                safe_divide(item["redeem_notional"], item["asset_initial_price"])
                for item in notional_list
            ],
            index=dates,
        )
        self.redeem_int_schedule = pd.Series(
            [item["fix_interest_realized"] for item in notional_list], index=dates
        )
        self.outstanding_margin_schedule = pd.Series(
            [item["outstanding_margin"] for item in notional_list], index=dates
        )
        return self.notional_schedule

    def align_schedules(self) -> pd.DataFrame:
        """Merge the individual schedules onto a common date index."""
        self.merged_schedule = pd.DataFrame(
            {
                "asset_quantity": self.asset_quantity_schedule,
                "asset_initial_price": self.asset_initial_price_schedule,
                "redeem_quantity": self.redeem_quantity_schedule,
                "redeem_int": self.redeem_int_schedule,
                "outstanding_margin": self.outstanding_margin_schedule,
            }
        )
        self.merged_schedule = self.merged_schedule.ffill()
        return self.merged_schedule

    def get_schedule_by_date(self, date) -> Dict[str, Any]:
        """Return the schedule row for ``date`` (nearest if absent)."""
        if date in self.merged_schedule.index:
            return self.merged_schedule.loc[date].to_dict()
        nearest_date = self.merged_schedule.index[
            self.merged_schedule.index.get_indexer([date], method="nearest")[0]
        ]
        return self.merged_schedule.loc[nearest_date].to_dict()

    def update_state(self, new_state: SwapState) -> None:
        """Set the lifecycle state."""
        self.state = new_state
