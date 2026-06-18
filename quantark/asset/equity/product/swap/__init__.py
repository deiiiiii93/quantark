"""Equity Total Return Swap (TRS) products.

A TRS is a realized-cashflow instrument: given an observed asset price series it
accrues fixed-leg interest, float-leg mark-to-market, dividends, redemptions,
fees, and a margin ledger over a trading calendar. These products are standalone
cashflow products (see :class:`base_swap.BaseSwap`); they do not implement the
risk-neutral payoff interface of :class:`BaseEquityProduct`.

The product classes (which depend on the cashflow engine) are exposed lazily via
:pep:`562` ``__getattr__`` so that importing the engine package - which only needs
the lightweight enums from :mod:`trs_params` - does not trigger a
``product.swap`` <-> ``engine.cashflow`` circular import.
"""

import importlib

# Eager, engine-free exports (params, enums, base interface).
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
from quantark.asset.equity.product.swap.base_swap import BaseSwap

# Lazily-loaded product classes -> their defining submodule.
_LAZY_EXPORTS = {
    "OneAssetTotalReturnSwap": "one_asset_trs",
    "MultiAssetTRS": "multi_asset_trs",
    "MultiAssetInfo": "multi_asset_trs",
    "OneAssetTotalReturnSwapDualCcy": "one_asset_trs_dual_ccy",
}

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
    "BaseSwap",
    "OneAssetTotalReturnSwap",
    "MultiAssetTRS",
    "MultiAssetInfo",
    "OneAssetTotalReturnSwapDualCcy",
]


def __getattr__(name):
    """Lazily import engine-dependent product classes (PEP 562)."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(module, name)
