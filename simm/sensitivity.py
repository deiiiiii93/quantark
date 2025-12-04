"""
SIMM Sensitivity Module.

This module defines sensitivity data models and protocols for ISDA SIMM.
Includes base protocols, concrete sensitivity dataclasses, and sensitivity collections.
"""
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

from .taxonomy import (
    IRSubCurve,
    MarginType,
    RiskClass,
    SensitivityType,
)


@runtime_checkable
class Sensitivity(Protocol):
    """Base protocol for all SIMM sensitivities.
    
    All sensitivity types must implement this protocol to be used
    in SIMM calculations.
    """
    
    @property
    @abstractmethod
    def risk_class(self) -> RiskClass:
        """The risk class this sensitivity belongs to."""
        ...
    
    @property
    @abstractmethod
    def margin_type(self) -> MarginType:
        """Delta, Vega, Curvature, or BaseCorr."""
        ...
    
    @property
    @abstractmethod
    def amount(self) -> float:
        """The sensitivity value in calculation currency."""
        ...
    
    @property
    @abstractmethod
    def bucket(self) -> Any:
        """The bucket this sensitivity is assigned to."""
        ...
    
    @property
    @abstractmethod
    def qualifier(self) -> str:
        """The qualifier identifying the risk factor (currency, issuer, etc.)."""
        ...


@dataclass
class BaseSensitivity:
    """Base class for all sensitivity dataclasses.
    
    Provides common fields and validation.
    
    Attributes:
        trade_id: Identifier for the trade this sensitivity belongs to.
        amount: The sensitivity value in calculation currency.
        amount_currency: Currency of the sensitivity amount.
    """
    trade_id: str
    amount: float
    amount_currency: str = "USD"


@dataclass
class IRDeltaSensitivity(BaseSensitivity):
    """Interest rate delta sensitivity.
    
    Represents the sensitivity of a position to changes in interest rates
    at a specific tenor point on a specific curve.
    
    Attributes:
        currency: The currency of the rate curve (also serves as bucket).
        tenor: Tenor in years (must match one of IR_TENORS).
        sub_curve: The specific yield curve (OIS, Libor, etc.).
    """
    currency: str = "USD"
    tenor: float = 1.0
    sub_curve: IRSubCurve = IRSubCurve.OIS
    
    @property
    def risk_class(self) -> RiskClass:
        return RiskClass.INTEREST_RATE
    
    @property
    def margin_type(self) -> MarginType:
        return MarginType.DELTA
    
    @property
    def bucket(self) -> str:
        return self.currency
    
    @property
    def qualifier(self) -> str:
        return self.currency


@dataclass
class IRVegaSensitivity(BaseSensitivity):
    """Interest rate vega sensitivity.
    
    Represents the sensitivity to changes in interest rate volatility.
    
    Attributes:
        currency: The currency of the rate curve.
        option_tenor: Option expiry tenor in years.
        underlying_tenor: Underlying swap tenor in years.
    """
    currency: str = "USD"
    option_tenor: float = 1.0
    underlying_tenor: float = 1.0
    
    @property
    def risk_class(self) -> RiskClass:
        return RiskClass.INTEREST_RATE
    
    @property
    def margin_type(self) -> MarginType:
        return MarginType.VEGA
    
    @property
    def bucket(self) -> str:
        return self.currency
    
    @property
    def qualifier(self) -> str:
        return self.currency


@dataclass
class CreditDeltaSensitivity(BaseSensitivity):
    """Credit delta sensitivity.
    
    Represents the sensitivity to changes in credit spreads.
    
    Attributes:
        issuer: Issuer identifier (typically legal entity ID or name).
        bucket_number: SIMM bucket number (1-12 for CQ, 1-2 for CNQ).
        tenor: Tenor in years (must match one of CREDIT_TENORS).
        is_qualifying: True for Credit Qualifying, False for Non-Qualifying.
    """
    issuer: str = ""
    bucket_number: int = 1
    tenor: float = 5.0
    is_qualifying: bool = True
    
    @property
    def risk_class(self) -> RiskClass:
        if self.is_qualifying:
            return RiskClass.CREDIT_QUALIFYING
        else:
            return RiskClass.CREDIT_NON_QUALIFYING
    
    @property
    def margin_type(self) -> MarginType:
        return MarginType.DELTA
    
    @property
    def bucket(self) -> int:
        return self.bucket_number
    
    @property
    def qualifier(self) -> str:
        return self.issuer


@dataclass
class CreditVegaSensitivity(BaseSensitivity):
    """Credit vega sensitivity.
    
    Represents the sensitivity to changes in credit spread volatility.
    
    Attributes:
        issuer: Issuer identifier.
        bucket_number: SIMM bucket number.
        option_tenor: Option expiry tenor in years.
        is_qualifying: True for Credit Qualifying, False for Non-Qualifying.
    """
    issuer: str = ""
    bucket_number: int = 1
    option_tenor: float = 1.0
    is_qualifying: bool = True
    
    @property
    def risk_class(self) -> RiskClass:
        if self.is_qualifying:
            return RiskClass.CREDIT_QUALIFYING
        else:
            return RiskClass.CREDIT_NON_QUALIFYING
    
    @property
    def margin_type(self) -> MarginType:
        return MarginType.VEGA
    
    @property
    def bucket(self) -> int:
        return self.bucket_number
    
    @property
    def qualifier(self) -> str:
        return self.issuer


@dataclass
class BaseCorrSensitivity(BaseSensitivity):
    """Base correlation sensitivity (Credit Qualifying only).
    
    Represents sensitivity to changes in credit index base correlation.
    Only applicable to tranched credit products.
    
    Attributes:
        index_name: Credit index identifier.
    """
    index_name: str = ""
    
    @property
    def risk_class(self) -> RiskClass:
        return RiskClass.CREDIT_QUALIFYING
    
    @property
    def margin_type(self) -> MarginType:
        return MarginType.BASE_CORR
    
    @property
    def bucket(self) -> str:
        return "BaseCorr"
    
    @property
    def qualifier(self) -> str:
        return self.index_name


@dataclass
class EquityDeltaSensitivity(BaseSensitivity):
    """Equity delta sensitivity.
    
    Represents sensitivity to equity spot price changes.
    
    Attributes:
        issuer: Equity issuer identifier (ISIN, ticker, etc.).
        bucket_number: SIMM bucket number (1-12).
    """
    issuer: str = ""
    bucket_number: int = 1
    
    @property
    def risk_class(self) -> RiskClass:
        return RiskClass.EQUITY
    
    @property
    def margin_type(self) -> MarginType:
        return MarginType.DELTA
    
    @property
    def bucket(self) -> int:
        return self.bucket_number
    
    @property
    def qualifier(self) -> str:
        return self.issuer


@dataclass
class EquityVegaSensitivity(BaseSensitivity):
    """Equity vega sensitivity.
    
    Represents sensitivity to equity volatility changes.
    
    Attributes:
        issuer: Equity issuer identifier.
        bucket_number: SIMM bucket number.
        option_tenor: Option expiry tenor in years.
    """
    issuer: str = ""
    bucket_number: int = 1
    option_tenor: float = 1.0
    
    @property
    def risk_class(self) -> RiskClass:
        return RiskClass.EQUITY
    
    @property
    def margin_type(self) -> MarginType:
        return MarginType.VEGA
    
    @property
    def bucket(self) -> int:
        return self.bucket_number
    
    @property
    def qualifier(self) -> str:
        return self.issuer


@dataclass
class CommodityDeltaSensitivity(BaseSensitivity):
    """Commodity delta sensitivity.
    
    Represents sensitivity to commodity price changes.
    
    Attributes:
        commodity_name: Commodity identifier.
        bucket_number: SIMM bucket number (1-17).
        delivery_tenor: Delivery/forward tenor in years.
    """
    commodity_name: str = ""
    bucket_number: int = 1
    delivery_tenor: float = 0.0
    
    @property
    def risk_class(self) -> RiskClass:
        return RiskClass.COMMODITY
    
    @property
    def margin_type(self) -> MarginType:
        return MarginType.DELTA
    
    @property
    def bucket(self) -> int:
        return self.bucket_number
    
    @property
    def qualifier(self) -> str:
        return self.commodity_name


@dataclass
class CommodityVegaSensitivity(BaseSensitivity):
    """Commodity vega sensitivity.
    
    Represents sensitivity to commodity volatility changes.
    
    Attributes:
        commodity_name: Commodity identifier.
        bucket_number: SIMM bucket number.
        option_tenor: Option expiry tenor in years.
    """
    commodity_name: str = ""
    bucket_number: int = 1
    option_tenor: float = 1.0
    
    @property
    def risk_class(self) -> RiskClass:
        return RiskClass.COMMODITY
    
    @property
    def margin_type(self) -> MarginType:
        return MarginType.VEGA
    
    @property
    def bucket(self) -> int:
        return self.bucket_number
    
    @property
    def qualifier(self) -> str:
        return self.commodity_name


@dataclass
class FXDeltaSensitivity(BaseSensitivity):
    """FX delta sensitivity.
    
    Represents sensitivity to FX spot rate changes.
    
    Attributes:
        currency_pair: Currency pair (e.g., "EURUSD").
    """
    currency_pair: str = ""
    
    @property
    def risk_class(self) -> RiskClass:
        return RiskClass.FX
    
    @property
    def margin_type(self) -> MarginType:
        return MarginType.DELTA
    
    @property
    def bucket(self) -> int:
        return 1  # FX has single bucket
    
    @property
    def qualifier(self) -> str:
        return self.currency_pair


@dataclass
class FXVegaSensitivity(BaseSensitivity):
    """FX vega sensitivity.
    
    Represents sensitivity to FX volatility changes.
    
    Attributes:
        currency_pair: Currency pair.
        option_tenor: Option expiry tenor in years.
    """
    currency_pair: str = ""
    option_tenor: float = 1.0
    
    @property
    def risk_class(self) -> RiskClass:
        return RiskClass.FX
    
    @property
    def margin_type(self) -> MarginType:
        return MarginType.VEGA
    
    @property
    def bucket(self) -> int:
        return 1  # FX has single bucket
    
    @property
    def qualifier(self) -> str:
        return self.currency_pair


@dataclass
class CurvatureSensitivity(BaseSensitivity):
    """Curvature sensitivity (gamma/convexity).
    
    Generic curvature sensitivity that can apply to any risk class.
    Curvature margin captures the second-order risk not covered by delta/vega.
    
    Attributes:
        risk_class_value: The risk class this curvature applies to.
        qualifier_value: Risk factor identifier.
        bucket_value: Bucket assignment.
        cvr_up: Scenario up P&L.
        cvr_down: Scenario down P&L.
    """
    risk_class_value: RiskClass = RiskClass.INTEREST_RATE
    qualifier_value: str = ""
    bucket_value: Union[str, int] = ""
    cvr_up: float = 0.0
    cvr_down: float = 0.0
    
    @property
    def risk_class(self) -> RiskClass:
        return self.risk_class_value
    
    @property
    def margin_type(self) -> MarginType:
        return MarginType.CURVATURE
    
    @property
    def bucket(self) -> Union[str, int]:
        return self.bucket_value
    
    @property
    def qualifier(self) -> str:
        return self.qualifier_value


# Type alias for any sensitivity type
AnySensitivity = Union[
    IRDeltaSensitivity,
    IRVegaSensitivity,
    CreditDeltaSensitivity,
    CreditVegaSensitivity,
    BaseCorrSensitivity,
    EquityDeltaSensitivity,
    EquityVegaSensitivity,
    CommodityDeltaSensitivity,
    CommodityVegaSensitivity,
    FXDeltaSensitivity,
    FXVegaSensitivity,
    CurvatureSensitivity,
]


@dataclass
class SensitivityCollection:
    """Collection of sensitivities grouped by risk class and margin type.
    
    Provides methods for adding, grouping, and iterating over sensitivities.
    
    Attributes:
        sensitivities: List of all sensitivities in the collection.
    """
    sensitivities: List[AnySensitivity] = field(default_factory=list)
    
    def add(self, sensitivity: AnySensitivity) -> None:
        """Add a sensitivity to the collection."""
        self.sensitivities.append(sensitivity)
    
    def add_many(self, sensitivities: List[AnySensitivity]) -> None:
        """Add multiple sensitivities to the collection."""
        self.sensitivities.extend(sensitivities)
    
    def by_risk_class(self, risk_class: RiskClass) -> List[AnySensitivity]:
        """Get all sensitivities for a specific risk class."""
        return [s for s in self.sensitivities if s.risk_class == risk_class]
    
    def by_margin_type(self, margin_type: MarginType) -> List[AnySensitivity]:
        """Get all sensitivities for a specific margin type."""
        return [s for s in self.sensitivities if s.margin_type == margin_type]
    
    def by_risk_class_and_margin_type(
        self, risk_class: RiskClass, margin_type: MarginType
    ) -> List[AnySensitivity]:
        """Get sensitivities for a specific risk class and margin type combination."""
        return [
            s for s in self.sensitivities 
            if s.risk_class == risk_class and s.margin_type == margin_type
        ]
    
    def group_by_bucket(
        self, risk_class: RiskClass, margin_type: MarginType
    ) -> Dict[Any, List[AnySensitivity]]:
        """Group sensitivities by bucket for a risk class and margin type."""
        filtered = self.by_risk_class_and_margin_type(risk_class, margin_type)
        result: Dict[Any, List[AnySensitivity]] = {}
        for s in filtered:
            bucket = s.bucket
            if bucket not in result:
                result[bucket] = []
            result[bucket].append(s)
        return result
    
    def __len__(self) -> int:
        return len(self.sensitivities)
    
    def __iter__(self):
        return iter(self.sensitivities)
