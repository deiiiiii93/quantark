"""Equity Total Return Swap (TRS) products.

A TRS is a realized-cashflow instrument: given an observed asset price series it
accrues fixed-leg interest, float-leg mark-to-market, dividends, redemptions,
fees, and a margin ledger over a trading calendar. These products are standalone
cashflow products (see :class:`base_swap.BaseSwap`); they do not implement the
risk-neutral payoff interface of :class:`BaseEquityProduct`.
"""

from quantark.asset.equity.product.swap.trs_params import (
    SwapState,
    AccrualType,
    AccrualSide,
    SettleType,
    AssetParams,
    FixLegParams,
    FloatLegParams,
    EventParams,
    MarginParams,
    PricingParams,
    TRSParams,
)

__all__ = [
    "SwapState",
    "AccrualType",
    "AccrualSide",
    "SettleType",
    "AssetParams",
    "FixLegParams",
    "FloatLegParams",
    "EventParams",
    "MarginParams",
    "PricingParams",
    "TRSParams",
]
