"""Parameter objects for the Total Return Swap product family."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from quantark.util.exceptions import ValidationError


class SwapState(Enum):
    """Lifecycle state of a swap contract."""

    INITIATED = "initiated"
    ACTIVE = "active"
    MATURED = "matured"
    DEFAULTED = "defaulted"


class AccrualType(Enum):
    """Basis on which fixed-leg interest accrues."""

    NOTIONAL = "notional"
    MARKET_VALUE = "marketvalue"
    LAST_MARKET_VALUE = "last_marketvalue"


class AccrualSide(Enum):
    """Endpoint inclusion convention for interest accrual."""

    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"
    NEITHER = "neither"


class SettleType(Enum):
    """Settlement type for the contract."""

    CASH = "cash"
    MARGIN = "margin"


def _coerce_enum(value: Any, enum_cls, field_name: str):
    """Coerce a string to ``enum_cls`` (by value, case-insensitive)."""
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value.lower())
        except ValueError as exc:
            raise ValidationError(
                f"Invalid {field_name}: {value!r}"
            ) from exc
    raise ValidationError(f"Invalid {field_name}: {value!r}")


@dataclass
class AssetParams:
    """Underlying asset parameters and its observed price series."""

    asset_id: str
    asset_initial_price: float
    asset_prices: pd.Series
    asset_price_precision: int = 2

    def __post_init__(self):
        if not isinstance(self.asset_prices, pd.Series):
            raise ValidationError("asset_prices must be a pandas Series")
        self.asset_prices = self.asset_prices.round(self.asset_price_precision)


@dataclass
class FixLegParams:
    """Fixed-leg (financing) parameters."""

    rate: float
    tenor: str = "A"
    day_count: str = "A365F"
    notional: float = 1.0
    initial_notional: float = 1.0
    start_date: str = "2020-03-01"
    end_date: str = "2020-04-01"
    payment_lag: int = 0
    payment_calendar: Any = None
    accrual_type: AccrualType = AccrualType.NOTIONAL
    accrual_side: AccrualSide = AccrualSide.LEFT
    direction: int = -1

    def __post_init__(self):
        self.accrual_type = _coerce_enum(
            self.accrual_type, AccrualType, "accrual_type"
        )
        self.accrual_side = _coerce_enum(
            self.accrual_side, AccrualSide, "accrual_side"
        )


@dataclass
class FloatLegParams:
    """Floating-leg (asset total-return) parameters."""

    rate: float = 0.0
    tenor: str = "A"
    day_count: str = "A365F"
    notional: float = 1.0
    initial_notional: float = 1.0
    start_date: str = "2020-03-01"
    end_date: str = "2020-04-01"
    payment_lag: int = 0
    payment_calendar: Any = None
    direction: int = 1


@dataclass
class EventParams:
    """Lifecycle events (dividends, redemptions, fees) grouped by type."""

    dividend_events: List[Dict] = field(default_factory=list)
    redemption_events: List[Dict] = field(default_factory=list)
    upfront_fee_events: List[Dict] = field(default_factory=list)
    unwind_fee_events: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        self.events: Dict[str, List[Dict]] = {}
        if self.dividend_events:
            self.events["div_cash"] = self.dividend_events
        if self.redemption_events:
            self.events["redm"] = self.redemption_events
        if self.upfront_fee_events:
            self.events["upfront_fee"] = self.upfront_fee_events
        if self.unwind_fee_events:
            self.events["unwind_fee"] = self.unwind_fee_events


@dataclass
class MarginParams:
    """Margin account parameters and movements."""

    initial_margin: float = 0.0
    outstanding_margin: float = 0.0
    margin_events: List[Dict] = field(default_factory=list)
    settle_type: SettleType = SettleType.MARGIN

    def __post_init__(self):
        self.settle_type = _coerce_enum(
            self.settle_type, SettleType, "settle_type"
        )


@dataclass
class PricingParams:
    """Valuation parameters."""

    valuation_date: str
    output_mode: str = "full"
    interest_basis: str = "act/365"

    def __post_init__(self):
        if self.output_mode.lower() not in ("spot", "full"):
            raise ValidationError(
                f"output_mode must be 'spot' or 'full', got {self.output_mode!r}"
            )


@dataclass
class TRSParams:
    """Aggregate parameters for a single-asset Total Return Swap."""

    contract_id: str
    asset: AssetParams
    fix_leg: FixLegParams
    float_leg: FloatLegParams
    events: EventParams = field(default_factory=EventParams)
    margin: MarginParams = field(default_factory=MarginParams)
    pricing: Optional[PricingParams] = None

    def __post_init__(self):
        if self.pricing is None:
            raise ValidationError("pricing parameters must be provided")

        min_prices_required = (
            2 if self.fix_leg.accrual_type == AccrualType.LAST_MARKET_VALUE else 1
        )
        if len(self.asset.asset_prices) < min_prices_required:
            raise ValidationError(
                f"insufficient asset_prices given, at least "
                f"{min_prices_required} is needed"
            )
