"""
Data models for RFQ quote solving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close


class RFQInputMode(Enum):
    """Supported RFQ request input modes."""

    OBJECT = "object"
    TERMSHEET = "termsheet"


class RFQTargetLabel(Enum):
    """Target labels supported by the RFQ solver."""

    PRICE = "price"
    PREMIUM = "premium"
    REOFFER = "reoffer"


class RFQQuoteStatus(Enum):
    """RFQ quote lifecycle status for v1."""

    SUCCESS = "success"


@dataclass(frozen=True)
class RFQUnknownSpec:
    """Describe the single unknown value to solve for."""

    field_path: str
    lower_bound: float
    upper_bound: float
    initial_guess: Optional[float] = None
    display_label: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.field_path:
            raise ValidationError("field_path is required")
        if self.lower_bound >= self.upper_bound:
            raise ValidationError(
                "lower_bound must be strictly less than upper_bound"
            )
        if self.initial_guess is not None:
            if not (self.lower_bound <= self.initial_guess <= self.upper_bound):
                raise ValidationError(
                    "initial_guess must lie within [lower_bound, upper_bound]"
                )


@dataclass(frozen=True)
class RFQTarget:
    """Explicit target price objective."""

    label: RFQTargetLabel
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, RFQTargetLabel):
            raise ValidationError(f"Invalid RFQ target label: {self.label}")


@dataclass(frozen=True)
class RFQObjectInput:
    """Object-based RFQ input path."""

    product: Any
    pricing_env: PricingEnvironment
    engine: BaseEngine

    def __post_init__(self) -> None:
        if self.product is None:
            raise ValidationError("product is required for object RFQ input")
        if self.pricing_env is None:
            raise ValidationError("pricing_env is required for object RFQ input")
        if self.engine is None:
            raise ValidationError("engine is required for object RFQ input")


@dataclass(frozen=True)
class RFQEngineSpec:
    """Term-sheet engine selection."""

    engine_name: str
    params_type: Optional[str] = None
    params_kwargs: Dict[str, Any] = field(default_factory=dict)
    method: Optional[Any] = None
    engine_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.engine_name:
            raise ValidationError("engine_name is required in engine_spec")


@dataclass(frozen=True)
class RFQTermsheetInput:
    """Term-sheet RFQ input path."""

    product_type: str
    product_kwargs: Dict[str, Any]
    market_kwargs: Dict[str, Any]
    engine_spec: RFQEngineSpec

    def __post_init__(self) -> None:
        if not self.product_type:
            raise ValidationError("product_type is required for term-sheet RFQ input")
        if self.product_kwargs is None:
            raise ValidationError("product_kwargs is required for term-sheet RFQ input")
        if self.market_kwargs is None:
            raise ValidationError("market_kwargs is required for term-sheet RFQ input")
        if self.engine_spec is None:
            raise ValidationError("engine_spec is required for term-sheet RFQ input")


@dataclass(frozen=True)
class RFQRequest:
    """Canonical RFQ request."""

    input_mode: RFQInputMode
    unknown: RFQUnknownSpec
    target: RFQTarget
    object_input: Optional[RFQObjectInput] = None
    termsheet_input: Optional[RFQTermsheetInput] = None
    valid_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.input_mode, RFQInputMode):
            raise ValidationError(f"Invalid RFQ input mode: {self.input_mode}")
        if self.unknown is None:
            raise ValidationError("unknown is required")
        if self.target is None:
            raise ValidationError("target is required")

        if self.input_mode == RFQInputMode.OBJECT:
            if self.object_input is None:
                raise ValidationError("object_input is required when input_mode=OBJECT")
            if self.termsheet_input is not None:
                raise ValidationError(
                    "termsheet_input must not be supplied when input_mode=OBJECT"
                )
        elif self.input_mode == RFQInputMode.TERMSHEET:
            if self.termsheet_input is None:
                raise ValidationError(
                    "termsheet_input is required when input_mode=TERMSHEET"
                )
            if self.object_input is not None:
                raise ValidationError(
                    "object_input must not be supplied when input_mode=TERMSHEET"
                )


@dataclass(frozen=True)
class RFQQuote:
    """Successful RFQ quote response."""

    quote_id: str
    quoted_at: datetime
    status: RFQQuoteStatus
    field_path: str
    field_label: str
    solved_value: float
    target_label: RFQTargetLabel
    target_value: float
    achieved_price: float
    residual: float
    engine_summary: Dict[str, Any]
    request_summary: Dict[str, Any]
    valid_until: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not self.quote_id:
            raise ValidationError("quote_id is required")
        if not isinstance(self.status, RFQQuoteStatus):
            raise ValidationError(f"Invalid RFQ quote status: {self.status}")
        if not isinstance(self.target_label, RFQTargetLabel):
            raise ValidationError(f"Invalid RFQ target label: {self.target_label}")

    @property
    def converged(self) -> bool:
        """Whether the solved price matches the target within tight tolerance."""
        return is_close(self.achieved_price, self.target_value, abs_tol=1e-8)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize quote payload for reporting and tests."""
        return {
            "quote_id": self.quote_id,
            "quoted_at": self.quoted_at.isoformat(),
            "status": self.status.value,
            "field_path": self.field_path,
            "field_label": self.field_label,
            "solved_value": self.solved_value,
            "target_label": self.target_label.value,
            "target_value": self.target_value,
            "achieved_price": self.achieved_price,
            "residual": self.residual,
            "engine_summary": dict(self.engine_summary),
            "request_summary": dict(self.request_summary),
            "valid_until": (
                self.valid_until.isoformat() if self.valid_until is not None else None
            ),
        }
