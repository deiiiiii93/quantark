"""Base class for asset-class add-on calculators.

Reference: Basel Committee SA-CCR document, paragraphs 150-165.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from quantark.saccr.engines.maths import (
    maturity_factor_unmargined,
    maturity_factor_margined,
)
from quantark.saccr.models.netting_set import SACCRNettingSet
from quantark.saccr.models.trade import SACCRTrade


class BaseAddOn(ABC):
    """Abstract base class for asset-class add-on calculators.

    The primary contract is :meth:`calculate_with_breakdown`, which returns the
    total add-on for the asset class plus a per-hedging-set breakdown used to
    populate :class:`~quantark.saccr.results.result.SACCRResult`. :meth:`calculate`
    is a convenience returning just the total.
    """

    @abstractmethod
    def calculate_with_breakdown(
        self,
        trades: List[SACCRTrade],
        netting_set: SACCRNettingSet,
    ) -> Tuple[float, Dict[str, float]]:
        """Return ``(total_addon, {hedging_set_label: addon})`` for this asset class."""

    def calculate(
        self,
        trades: List[SACCRTrade],
        netting_set: SACCRNettingSet,
    ) -> float:
        """Return the total add-on for this asset class."""
        total, _ = self.calculate_with_breakdown(trades, netting_set)
        return total

    def get_maturity_factor(
        self,
        trade: SACCRTrade,
        netting_set: SACCRNettingSet,
    ) -> float:
        """Maturity factor for a trade (paragraph 164).

        Margined netting sets use the MPOR-based factor; unmargined sets use the
        trade-maturity factor.
        """
        if netting_set.is_margined:
            return maturity_factor_margined(netting_set.mpor_years)
        return maturity_factor_unmargined(trade.maturity)
