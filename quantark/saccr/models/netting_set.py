"""Netting set data model for SA-CCR calculations.

Reference: Basel Committee SA-CCR document, paragraphs 134-145.

Sign conventions (all amounts in the single reporting currency):
- ``market_value`` V: positive = in-the-money to the bank.
- ``variation_margin`` (VM) and ``independent_collateral_received`` increase C.
- ``independent_collateral_posted`` (unsegregated) reduces C and NICA.
- ``independent_collateral_posted_segregated`` is excluded from C (para 143).
- ``NICA = ICA_received - ICA_posted(unsegregated)``.
- Collateral is assumed already haircut-adjusted (cash).
"""

from dataclasses import dataclass, field
from typing import List, Optional

from quantark.saccr.models.trade import SACCRTrade
from quantark.util.exceptions import ValidationError

#: Business days per year used for the MPOR year-fraction conversion.
BUSINESS_DAYS_PER_YEAR = 250


@dataclass
class SACCRNettingSet:
    """A netting set of trades for SA-CCR calculation (paragraphs 134-145)."""

    netting_set_id: str
    trades: List[SACCRTrade] = field(default_factory=list)

    # Margin agreement
    is_margined: bool = False
    threshold: float = 0.0  # TH
    minimum_transfer_amount: float = 0.0  # MTA

    # Collateral components
    variation_margin: float = 0.0  # VM
    independent_collateral_received: float = 0.0  # ICA received from counterparty
    independent_collateral_posted: float = 0.0  # ICA posted (unsegregated)
    independent_collateral_posted_segregated: float = 0.0  # ICA posted (segregated)

    # Margin period of risk (business days)
    mpor_days: int = 10

    # Optional pre-calculated net collateral (overrides the component formula)
    net_collateral: Optional[float] = None

    def __post_init__(self):
        if not self.trades:
            raise ValidationError("Netting set must contain at least one trade")

    @property
    def market_value(self) -> float:
        """Total market value of all trades (V) — paragraph 136."""
        return sum(trade.market_value for trade in self.trades)

    @property
    def collateral(self) -> float:
        """Haircut value of net collateral held (C) — paragraphs 136, 143.

        Segregated collateral posted by the bank is NOT subtracted from C.
        """
        if self.net_collateral is not None:
            return self.net_collateral
        return (
            self.variation_margin
            + self.independent_collateral_received
            - self.independent_collateral_posted
        )

    @property
    def nica(self) -> float:
        """Net Independent Collateral Amount (paragraph 143).

        ``NICA = ICA_received - ICA_posted(unsegregated)``.
        """
        return self.independent_collateral_received - self.independent_collateral_posted

    @property
    def mpor_years(self) -> float:
        """Margin period of risk in years (``mpor_days / 250``)."""
        return self.mpor_days / BUSINESS_DAYS_PER_YEAR
