"""
SIMM Taxonomy Module.

This module defines all ISDA SIMM v2.6 taxonomy elements including:
- Risk classes (IR, Credit, Equity, Commodity, FX)
- Product classes (RatesFX, Credit, Equity, Commodity)
- Margin types (Delta, Vega, Curvature, BaseCorr)
- Sensitivity types for CRIF classification
- Tenor definitions for IR and Credit risk
- Currency volatility classifications
- Bucket definitions for each risk class
- IR sub-curve definitions
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Tuple


class RiskClass(Enum):
    """SIMM risk class taxonomy.
    
    Six risk classes as defined by ISDA SIMM specification.
    Each risk class has its own bucketing scheme and risk weights.
    """
    INTEREST_RATE = "IR"
    CREDIT_QUALIFYING = "CreditQ"
    CREDIT_NON_QUALIFYING = "CreditNQ"
    EQUITY = "Equity"
    COMMODITY = "Commodity"
    FX = "FX"
    
    def __str__(self) -> str:
        return self.value


class ProductClass(Enum):
    """SIMM product class taxonomy.
    
    Four product classes for margin aggregation.
    Final SIMM is calculated per product class then summed.
    """
    RATES_FX = "RatesFX"
    CREDIT = "Credit"
    EQUITY = "Equity"
    COMMODITY = "Commodity"
    
    def __str__(self) -> str:
        return self.value


class MarginType(Enum):
    """SIMM margin type (sensitivity category).
    
    Four margin types corresponding to different risk measures:
    - Delta: First-order price sensitivity
    - Vega: Volatility sensitivity
    - Curvature: Second-order gamma/convexity
    - BaseCorr: Credit base correlation (Credit Qualifying only)
    """
    DELTA = "Delta"
    VEGA = "Vega"
    CURVATURE = "Curvature"
    BASE_CORR = "BaseCorr"
    
    def __str__(self) -> str:
        return self.value


class SensitivityType(Enum):
    """CRIF risk_type values for SIMM classification.
    
    These map to specific sensitivity calculations and bucket assignments.
    """
    # Interest Rate
    RISK_IR_CURVE = "Risk_IRCurve"
    RISK_IR_VOL = "Risk_IRVol"
    RISK_INFLATION = "Risk_Inflation"
    RISK_INFLATION_VOL = "Risk_InflationVol"
    RISK_XCCY_BASIS = "Risk_XCcyBasis"
    
    # Credit
    RISK_CREDIT_Q = "Risk_CreditQ"
    RISK_CREDIT_VOL = "Risk_CreditVol"
    RISK_CREDIT_NQ = "Risk_CreditNonQ"
    RISK_CREDIT_NQ_VOL = "Risk_CreditVolNonQ"
    RISK_BASE_CORR = "Risk_BaseCorr"
    
    # Equity
    RISK_EQUITY = "Risk_Equity"
    RISK_EQUITY_VOL = "Risk_EquityVol"
    
    # Commodity
    RISK_COMMODITY = "Risk_Commodity"
    RISK_COMMODITY_VOL = "Risk_CommodityVol"
    
    # FX
    RISK_FX = "Risk_FX"
    RISK_FX_VOL = "Risk_FXVol"
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def risk_class(self) -> RiskClass:
        """Return the risk class for this sensitivity type."""
        mapping = {
            SensitivityType.RISK_IR_CURVE: RiskClass.INTEREST_RATE,
            SensitivityType.RISK_IR_VOL: RiskClass.INTEREST_RATE,
            SensitivityType.RISK_INFLATION: RiskClass.INTEREST_RATE,
            SensitivityType.RISK_INFLATION_VOL: RiskClass.INTEREST_RATE,
            SensitivityType.RISK_XCCY_BASIS: RiskClass.INTEREST_RATE,
            SensitivityType.RISK_CREDIT_Q: RiskClass.CREDIT_QUALIFYING,
            SensitivityType.RISK_CREDIT_VOL: RiskClass.CREDIT_QUALIFYING,
            SensitivityType.RISK_BASE_CORR: RiskClass.CREDIT_QUALIFYING,
            SensitivityType.RISK_CREDIT_NQ: RiskClass.CREDIT_NON_QUALIFYING,
            SensitivityType.RISK_CREDIT_NQ_VOL: RiskClass.CREDIT_NON_QUALIFYING,
            SensitivityType.RISK_EQUITY: RiskClass.EQUITY,
            SensitivityType.RISK_EQUITY_VOL: RiskClass.EQUITY,
            SensitivityType.RISK_COMMODITY: RiskClass.COMMODITY,
            SensitivityType.RISK_COMMODITY_VOL: RiskClass.COMMODITY,
            SensitivityType.RISK_FX: RiskClass.FX,
            SensitivityType.RISK_FX_VOL: RiskClass.FX,
        }
        return mapping[self]
    
    @property
    def margin_type(self) -> MarginType:
        """Return the margin type for this sensitivity type."""
        vega_types = {
            SensitivityType.RISK_IR_VOL,
            SensitivityType.RISK_INFLATION_VOL,
            SensitivityType.RISK_CREDIT_VOL,
            SensitivityType.RISK_CREDIT_NQ_VOL,
            SensitivityType.RISK_EQUITY_VOL,
            SensitivityType.RISK_COMMODITY_VOL,
            SensitivityType.RISK_FX_VOL,
        }
        if self == SensitivityType.RISK_BASE_CORR:
            return MarginType.BASE_CORR
        elif self in vega_types:
            return MarginType.VEGA
        else:
            return MarginType.DELTA


class IRSubCurve(Enum):
    """Interest rate sub-curve definitions.
    
    SIMM distinguishes different yield curves within each currency.
    """
    OIS = "OIS"
    LIBOR_1M = "Libor1m"
    LIBOR_3M = "Libor3m"
    LIBOR_6M = "Libor6m"
    LIBOR_12M = "Libor12m"
    PRIME = "Prime"
    MUNICIPAL = "Municipal"
    
    def __str__(self) -> str:
        return self.value


# -----------------------------------------------------------------------------
# Tenor Definitions
# -----------------------------------------------------------------------------

# IR tenor vertices (in years, matching SIMM v2.6 spec)
IR_TENORS: Tuple[float, ...] = (
    0.0384,  # 2 weeks = 14/365
    0.0833,  # 1 month
    0.25,    # 3 months
    0.5,     # 6 months
    1.0,     # 1 year
    2.0,     # 2 years
    3.0,     # 3 years
    5.0,     # 5 years
    10.0,    # 10 years
    15.0,    # 15 years
    20.0,    # 20 years
    30.0,    # 30 years
)

# IR tenor labels for display
IR_TENOR_LABELS: Tuple[str, ...] = (
    "2w", "1m", "3m", "6m", "1y", "2y", "3y", "5y", "10y", "15y", "20y", "30y"
)

# Credit tenor vertices (in years)
CREDIT_TENORS: Tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 10.0)

# Credit tenor labels for display
CREDIT_TENOR_LABELS: Tuple[str, ...] = ("1y", "2y", "3y", "5y", "10y")

# Vega tenor labels (used for all risk classes)
VEGA_TENORS: Tuple[float, ...] = (0.5, 1.0, 3.0, 5.0, 10.0)
VEGA_TENOR_LABELS: Tuple[str, ...] = ("6m", "1y", "3y", "5y", "10y")


# -----------------------------------------------------------------------------
# Currency Volatility Classifications
# -----------------------------------------------------------------------------

class CurrencyVolatility(Enum):
    """Currency volatility classification for IR risk weights.
    
    SIMM classifies currencies into three volatility buckets
    with different risk weight scaling.
    """
    REGULAR = "Regular"
    LOW = "Low"
    HIGH = "High"
    
    def __str__(self) -> str:
        return self.value


# Low volatility currencies (major reserve currencies)
LOW_VOL_CURRENCIES: Tuple[str, ...] = ("EUR", "USD", "GBP", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK", "DKK", "HKD", "SGD", "TWD")

# High volatility currencies (emerging markets, high inflation)
HIGH_VOL_CURRENCIES: Tuple[str, ...] = (
    "ARS", "BRL", "CLP", "COP", "IDR", "INR", "MXN", "MYR", "PEN", "PHP", 
    "RUB", "THB", "TRY", "ZAR", "KRW", "HUF", "PLN", "CZK", "ILS", "RON"
)


def get_currency_volatility(currency: str) -> CurrencyVolatility:
    """Get the volatility classification for a currency.
    
    Args:
        currency: Three-letter ISO currency code.
        
    Returns:
        CurrencyVolatility classification.
    """
    currency = currency.upper()
    if currency in LOW_VOL_CURRENCIES:
        return CurrencyVolatility.LOW
    elif currency in HIGH_VOL_CURRENCIES:
        return CurrencyVolatility.HIGH
    else:
        return CurrencyVolatility.REGULAR


# -----------------------------------------------------------------------------
# Bucket Definitions
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class IRBucket:
    """Interest rate bucket definition.
    
    For IR risk, buckets are defined by currency.
    Each currency is a separate bucket with its own risk weights.
    
    Attributes:
        currency: Three-letter ISO currency code.
        volatility: Currency volatility classification.
    """
    currency: str
    volatility: CurrencyVolatility
    
    @classmethod
    def from_currency(cls, currency: str) -> "IRBucket":
        """Create an IR bucket from a currency code."""
        return cls(
            currency=currency.upper(),
            volatility=get_currency_volatility(currency)
        )


@dataclass(frozen=True)
class CreditQualifyingBucket:
    """Credit Qualifying bucket definition.
    
    12 buckets plus residual for Credit Qualifying risk.
    Buckets are defined by credit quality (IG/HY) and sector.
    
    Attributes:
        bucket_number: Bucket identifier (1-12, or "Residual").
        credit_quality: "IG" (Investment Grade) or "HY/NR" (High Yield/Not Rated).
        sector: Sector description.
    """
    bucket_number: int
    credit_quality: str
    sector: str


# Credit Qualifying bucket definitions (SIMM v2.6)
CREDIT_QUALIFYING_BUCKETS: Dict[int, CreditQualifyingBucket] = {
    1: CreditQualifyingBucket(1, "IG", "Sovereigns including central banks"),
    2: CreditQualifyingBucket(2, "IG", "Financials including government-backed"),
    3: CreditQualifyingBucket(3, "IG", "Basic materials, energy, industrials"),
    4: CreditQualifyingBucket(4, "IG", "Consumer"),
    5: CreditQualifyingBucket(5, "IG", "Technology, telecommunications"),
    6: CreditQualifyingBucket(6, "IG", "Health care, utilities, local government"),
    7: CreditQualifyingBucket(7, "HY/NR", "Sovereigns including central banks"),
    8: CreditQualifyingBucket(8, "HY/NR", "Financials including government-backed"),
    9: CreditQualifyingBucket(9, "HY/NR", "Basic materials, energy, industrials"),
    10: CreditQualifyingBucket(10, "HY/NR", "Consumer"),
    11: CreditQualifyingBucket(11, "HY/NR", "Technology, telecommunications"),
    12: CreditQualifyingBucket(12, "HY/NR", "Health care, utilities, local government"),
}
CREDIT_QUALIFYING_RESIDUAL_BUCKET = CreditQualifyingBucket(-1, "Residual", "Residual")


@dataclass(frozen=True)
class CreditNonQualifyingBucket:
    """Credit Non-Qualifying bucket definition.
    
    2 buckets plus residual for Credit Non-Qualifying risk.
    
    Attributes:
        bucket_number: Bucket identifier (1-2, or "Residual").
        credit_quality: "IG" or "HY/NR".
    """
    bucket_number: int
    credit_quality: str


# Credit Non-Qualifying bucket definitions
CREDIT_NON_QUALIFYING_BUCKETS: Dict[int, CreditNonQualifyingBucket] = {
    1: CreditNonQualifyingBucket(1, "IG"),
    2: CreditNonQualifyingBucket(2, "HY/NR"),
}
CREDIT_NON_QUALIFYING_RESIDUAL_BUCKET = CreditNonQualifyingBucket(-1, "Residual")


@dataclass(frozen=True)
class EquityBucket:
    """Equity bucket definition.
    
    12 buckets plus residual for Equity risk.
    Buckets are defined by market cap size, region, and sector.
    
    Attributes:
        bucket_number: Bucket identifier (1-12, or "Residual").
        size: "Large", "Small", or "All".
        region: "Emerging", "Developed", or "All".
        sector: Sector description.
    """
    bucket_number: int
    size: str
    region: str
    sector: str


# Equity bucket definitions (SIMM v2.6)
EQUITY_BUCKETS: Dict[int, EquityBucket] = {
    1: EquityBucket(1, "Large", "Emerging", "Consumer goods and services, transportation, admin, agriculture"),
    2: EquityBucket(2, "Large", "Emerging", "Telecommunications, industrials, utilities"),
    3: EquityBucket(3, "Large", "Emerging", "Basic materials, energy, agriculture"),
    4: EquityBucket(4, "Large", "Emerging", "Financials, tech, health care, real estate"),
    5: EquityBucket(5, "Large", "Developed", "Consumer goods and services, transportation, admin"),
    6: EquityBucket(6, "Large", "Developed", "Telecommunications, industrials"),
    7: EquityBucket(7, "Large", "Developed", "Basic materials, energy, utilities"),
    8: EquityBucket(8, "Large", "Developed", "Financials, tech, health care, real estate"),
    9: EquityBucket(9, "Small", "Emerging", "All sectors"),
    10: EquityBucket(10, "Small", "Developed", "All sectors"),
    11: EquityBucket(11, "All", "All", "Indices, funds, ETFs"),
    12: EquityBucket(12, "All", "All", "Volatility indices"),
}
EQUITY_RESIDUAL_BUCKET = EquityBucket(-1, "All", "All", "Residual")


@dataclass(frozen=True)
class CommodityBucket:
    """Commodity bucket definition.
    
    17 buckets for Commodity risk, defined by commodity type.
    
    Attributes:
        bucket_number: Bucket identifier (1-17).
        commodity_type: Commodity type description.
    """
    bucket_number: int
    commodity_type: str


# Commodity bucket definitions (SIMM v2.6)
COMMODITY_BUCKETS: Dict[int, CommodityBucket] = {
    1: CommodityBucket(1, "Coal"),
    2: CommodityBucket(2, "Crude oil"),
    3: CommodityBucket(3, "Light ends"),
    4: CommodityBucket(4, "Middle distillates"),
    5: CommodityBucket(5, "Heavy distillates"),
    6: CommodityBucket(6, "North American natural gas"),
    7: CommodityBucket(7, "European natural gas"),
    8: CommodityBucket(8, "North American power"),
    9: CommodityBucket(9, "European power"),
    10: CommodityBucket(10, "Freight"),
    11: CommodityBucket(11, "Base metals"),
    12: CommodityBucket(12, "Precious metals"),
    13: CommodityBucket(13, "Grains"),
    14: CommodityBucket(14, "Softs"),
    15: CommodityBucket(15, "Livestock"),
    16: CommodityBucket(16, "Other"),
    17: CommodityBucket(17, "Indices"),
}


@dataclass(frozen=True)
class FXBucket:
    """FX bucket definition.
    
    FX has a single bucket containing all currency pairs.
    Sensitivities are distinguished by the currency pair qualifier.
    
    Attributes:
        bucket_number: Always 1 for FX.
    """
    bucket_number: int = 1


# Single FX bucket
FX_BUCKET = FXBucket()
