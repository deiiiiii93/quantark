"""Top-level position input: counterparties + eligible CVA hedges (spec §3.1)."""

from dataclasses import dataclass
from typing import List

from quantark.sacva.portfolio.counterparty import Counterparty
from quantark.sacva.portfolio.trade import CVAHedge
from quantark.util.exceptions import ValidationError


@dataclass
class CVATradePortfolio:
    counterparties: List[Counterparty]
    hedges: List[CVAHedge]
    reporting_currency: str = "USD"

    def __post_init__(self) -> None:
        if not self.counterparties:
            raise ValidationError("CVATradePortfolio requires >=1 counterparty")
        if not all(isinstance(c, Counterparty) for c in self.counterparties):
            raise ValidationError("counterparties must be Counterparty instances")
        if not all(isinstance(h, CVAHedge) for h in self.hedges):
            raise ValidationError("hedges must be CVAHedge instances")
        if not self.reporting_currency:
            raise ValidationError("reporting_currency required")
        self.reporting_currency = self.reporting_currency.upper()
