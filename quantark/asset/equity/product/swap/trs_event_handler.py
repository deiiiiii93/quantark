"""Lifecycle event handler for Total Return Swaps.

This handler records contract events (dividends, redemptions, fees, margin
movements) against the :class:`MarginAccount` history. The authoritative
cashflow arithmetic lives in :class:`TotalReturnSwapEngine`; this handler tracks
the margin ledger entries.
"""

from typing import Dict

from quantark.asset.equity.product.swap.trs_schedule import (
    ScheduleManager,
    MarginAccount,
)


class TRSEventHandler:
    """Applies TRS lifecycle events to the schedule and margin account."""

    def __init__(self, scheduler: ScheduleManager, margin_account: MarginAccount):
        self.scheduler = scheduler
        self.margin_account = margin_account

    def handle_dividend(self, event: Dict) -> None:
        """Record a cash-dividend cashflow on the margin account."""
        date = event.get("date")
        cash_div_per_share = event.get("cash_div_per_share", 0)
        deliver_ratio = event.get("deliver_ratio", 1.0)

        schedule = self.scheduler.get_schedule_by_date(date)
        asset_quantity = schedule.get("asset_quantity", 0)

        dividend_amount = cash_div_per_share * asset_quantity * deliver_ratio
        self.margin_account.apply_cashflow(date, dividend_amount, "CASH_DIV")

    def handle_share_dividend(self, event: Dict) -> None:
        """Record a share-dividend event (quantity adjustment handled in engine)."""
        # Quantity/initial-price adjustment is performed by the engine when it
        # builds the notional schedule; nothing to settle on the margin account.

    def handle_redemption(self, event: Dict) -> None:
        """Record the net cashflow of a redemption on the margin account."""
        date = event.get("date")
        redeem_notional = event.get("redeem_notional", 0)
        redeem_price = event.get("redeem_price", 0)
        redeem_fee_rate = event.get("redeem_fee_rate", 0)
        redeem_settle_option = event.get("redeem_settle_option", [])

        schedule = self.scheduler.get_schedule_by_date(date)
        asset_initial_price = schedule.get("asset_initial_price", 0)

        redeem_quantity = (
            redeem_notional / asset_initial_price if asset_initial_price else 0
        )
        redeem_fee = redeem_fee_rate * redeem_price * redeem_quantity

        asset_interest = 0
        if "asset" in redeem_settle_option:
            asset_interest = (redeem_price - asset_initial_price) * redeem_quantity

        total_cashflow = asset_interest - redeem_fee
        self.margin_account.apply_cashflow(date, total_cashflow, "REDM")

    def handle_margin_event(self, event: Dict) -> None:
        """Record an explicit margin in/out movement."""
        date = event.get("date")
        amount = event.get("amount", 0)
        event_type = event.get("type", "").lower()
        direction = 1 if event_type == "in" else -1
        self.margin_account.apply_cashflow(
            date, amount * direction, f"MARGIN_{event_type.upper()}"
        )

    def process_events(self, events: Dict) -> None:
        """Dispatch each grouped event list to its handler."""
        if not events:
            return

        for event in events.get("div_cash", []):
            self.handle_dividend(event)
        for event in events.get("div_share", []):
            self.handle_share_dividend(event)
        for event in events.get("redm", []):
            self.handle_redemption(event)
        for event in events.get("margin_event", []):
            self.handle_margin_event(event)
