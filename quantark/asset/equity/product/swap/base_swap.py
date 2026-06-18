"""Base interface for realized-cashflow swap products.

Unlike :class:`quantark.asset.equity.product.base_equity_product.BaseEquityProduct`,
a swap is not a payoff-on-spot instrument. It is priced by walking an observed
asset price path over a calendar and accruing realized cashflows. This base class
captures only what every swap exposes: a lifecycle ``state`` and a ``price`` method
returning a per-period cashflow table.
"""

from abc import ABC, abstractmethod

import pandas as pd

from quantark.asset.equity.product.swap.trs_params import SwapState


class BaseSwap(ABC):
    """Abstract base for realized-cashflow swap products."""

    state: SwapState

    @abstractmethod
    def price(self, precision: int = 2) -> pd.DataFrame:
        """
        Compute the swap's realized cashflows.

        Args:
            precision: Number of decimal places for rounding outputs.

        Returns:
            A DataFrame of per-period cashflows (or a single summary row when
            the pricing parameters request ``spot`` output).
        """
        raise NotImplementedError
