"""Single-asset Total Return Swap product."""

import pandas as pd

from quantark.asset.equity.product.swap.base_swap import BaseSwap
from quantark.asset.equity.product.swap.trs_params import (
    TRSParams,
    EventParams,
    SwapState,
)
from quantark.asset.equity.product.swap.trs_schedule import (
    ScheduleManager,
    MarginAccount,
)
from quantark.asset.equity.product.swap.trs_event_handler import TRSEventHandler
from quantark.asset.equity.engine.cashflow.total_return_swap_engine import (
    TotalReturnSwapEngine,
)


class OneAssetTotalReturnSwap(BaseSwap):
    """A single-asset Total Return Swap priced from an observed price path."""

    def __init__(self, params: TRSParams):
        self.engine = TotalReturnSwapEngine()
        self.params = params

        self._init_managers()
        self._process_events(params.events)
        self._set_state()

    def _init_managers(self) -> None:
        """Create the schedule manager, margin account and event handler."""
        self.scheduler = ScheduleManager(self.params)
        self.margin_account = MarginAccount(self.params.margin)
        self.event_handler = TRSEventHandler(self.scheduler, self.margin_account)
        self.scheduler.create_notional_schedule(self.engine)

    def _process_events(self, events: EventParams) -> None:
        """Align schedules, then record contract events on the margin ledger.

        Alignment must happen first: the event handler queries the merged
        schedule by date, so it has to be populated before events are processed.
        """
        self.scheduler.align_schedules()
        if events and events.events:
            self.event_handler.process_events(events.events)

    def _set_state(self) -> None:
        """Set ACTIVE/MATURED based on the contract end vs valuation date."""
        contract_end_date = max(
            self.params.fix_leg.end_date, self.params.float_leg.end_date
        )
        if contract_end_date < self.params.pricing.valuation_date:
            self.scheduler.update_state(SwapState.MATURED)
        else:
            self.scheduler.update_state(SwapState.ACTIVE)

    @property
    def state(self) -> SwapState:
        """Current lifecycle state of the contract."""
        return self.scheduler.state

    def price(self, precision: int = 2) -> pd.DataFrame:
        """Compute the swap's realized cashflow table."""
        return self.engine.price(self.params, precision)
