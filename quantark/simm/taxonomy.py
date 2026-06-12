"""
SIMM Taxonomy Module.

This module defines all ISDA SIMM v2.6 taxonomy elements including:
- Risk classes (IR, Credit Qualifying/Non-Qualifying, Equity, Commodity, FX)
- Product classes (RatesFX, Credit, Equity, Commodity)
- Margin types (Delta, Vega, Curvature, BaseCorr)
- Sensitivity types for CRIF classification
- Tenor vertex definitions for IR and Credit risk
- Currency group classifications (IR risk-weight groups per paragraph 33,
  IR concentration groups per paragraph 75, FX volatility groups per
  paragraphs 67-68, FX concentration categories per paragraph 80)
- Bucket definitions for each risk class

All paragraph references are to the ISDA SIMM Methodology, version 2.6
(see quantark/simm/doc/isda_simm.md).
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, Union


class RiskClass(Enum):
    """SIMM risk class taxonomy (paragraph 5).

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
    """SIMM product class taxonomy (paragraph 6).

    Every trade is assigned to a single product class and SIMM is
    considered separately for each product class. The total SIMM is the
    sum of the four product class SIMM values.
    """
    RATES_FX = "RatesFX"
    CREDIT = "Credit"
    EQUITY = "Equity"
    COMMODITY = "Commodity"

    def __str__(self) -> str:
        return self.value


class MarginType(Enum):
    """SIMM margin type (paragraph 5).

    IM_X = DeltaMargin_X + VegaMargin_X + CurvatureMargin_X + BaseCorrMargin_X,
    where BaseCorrMargin only exists in the Credit Qualifying risk class.
    """
    DELTA = "Delta"
    VEGA = "Vega"
    CURVATURE = "Curvature"
    BASE_CORR = "BaseCorr"

    def __str__(self) -> str:
        return self.value


# Default product class for sensitivities that do not carry an explicit
# product class assignment. Note this is a *fallback*: paragraph 6 requires
# the product class to follow the trade, e.g. the IR delta of an equity
# derivative belongs to the Equity product class.
DEFAULT_PRODUCT_CLASS: Dict[RiskClass, ProductClass] = {
    RiskClass.INTEREST_RATE: ProductClass.RATES_FX,
    RiskClass.FX: ProductClass.RATES_FX,
    RiskClass.CREDIT_QUALIFYING: ProductClass.CREDIT,
    RiskClass.CREDIT_NON_QUALIFYING: ProductClass.CREDIT,
    RiskClass.EQUITY: ProductClass.EQUITY,
    RiskClass.COMMODITY: ProductClass.COMMODITY,
}


class SensitivityType(Enum):
    """CRIF RiskType values for SIMM classification."""
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
    """Interest rate sub yield curve names (paragraph 14)."""
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
# Tenor Vertex Definitions
# -----------------------------------------------------------------------------

# IR tenor vertices in years (paragraph 14). The 2-week vertex uses the
# calendar-day convention of paragraph 11(a): 12m = 365 days, pro-rata
# scaling for other tenors.
IR_TENORS: Tuple[float, ...] = (
    14.0 / 365.0,  # 2 weeks
    1.0 / 12.0,    # 1 month
    0.25,          # 3 months
    0.5,           # 6 months
    1.0,           # 1 year
    2.0,           # 2 years
    3.0,           # 3 years
    5.0,           # 5 years
    10.0,          # 10 years
    15.0,          # 15 years
    20.0,          # 20 years
    30.0,          # 30 years
)

# IR tenor vertex labels. These are also the vol-tenor (option expiry)
# buckets used for vega and curvature risk in all risk classes
# (paragraph 10(b)).
IR_TENOR_LABELS: Tuple[str, ...] = (
    "2w", "1m", "3m", "6m", "1y", "2y", "3y", "5y", "10y", "15y", "20y", "30y"
)

# Vega / curvature expiry buckets are the same twelve vertices.
VEGA_TENORS: Tuple[float, ...] = IR_TENORS
VEGA_TENOR_LABELS: Tuple[str, ...] = IR_TENOR_LABELS

# Credit tenor vertices in years (paragraphs 15-16).
CREDIT_TENORS: Tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 10.0)
CREDIT_TENOR_LABELS: Tuple[str, ...] = ("1y", "2y", "3y", "5y", "10y")

# Expiry time in calendar days for each tenor label, using the convention
# of paragraph 11(a): "12m" equals 365 calendar days with pro-rata scaling
# (1m = 365/12 days, 5y = 365*5 days).
TENOR_LABEL_DAYS: Dict[str, float] = {
    "2w": 14.0,
    "1m": 365.0 / 12.0,
    "3m": 365.0 / 4.0,
    "6m": 365.0 / 2.0,
    "1y": 365.0,
    "2y": 365.0 * 2.0,
    "3y": 365.0 * 3.0,
    "5y": 365.0 * 5.0,
    "10y": 365.0 * 10.0,
    "15y": 365.0 * 15.0,
    "20y": 365.0 * 20.0,
    "30y": 365.0 * 30.0,
}

# Alternative spellings accepted when normalising tenor labels.
_TENOR_LABEL_ALIASES: Dict[str, str] = {
    "2w": "2w", "14d": "2w",
    "1m": "1m",
    "3m": "3m",
    "6m": "6m",
    "12m": "1y", "1y": "1y", "1yr": "1y",
    "2y": "2y", "2yr": "2y",
    "3y": "3y", "3yr": "3y",
    "5y": "5y", "5yr": "5y",
    "10y": "10y", "10yr": "10y",
    "15y": "15y", "15yr": "15y",
    "20y": "20y", "20yr": "20y",
    "30y": "30y", "30yr": "30y",
}


def normalize_tenor_label(label: str) -> str:
    """Normalise a tenor label to the canonical SIMM vertex spelling.

    Args:
        label: A tenor label such as "5y", "5yr" or "12m".

    Returns:
        The canonical vertex label (one of IR_TENOR_LABELS).

    Raises:
        KeyError: If the label is not a recognised SIMM vertex.
    """
    return _TENOR_LABEL_ALIASES[label.strip().lower()]


def tenor_to_vertex_label(tenor_years: float) -> str:
    """Snap a tenor in years to the nearest of the twelve SIMM vertices.

    Args:
        tenor_years: Tenor in years.

    Returns:
        The label of the nearest vertex (one of IR_TENOR_LABELS).
    """
    idx = min(range(len(IR_TENORS)), key=lambda i: abs(IR_TENORS[i] - tenor_years))
    return IR_TENOR_LABELS[idx]


def credit_tenor_to_vertex_label(tenor_years: float) -> str:
    """Snap a tenor in years to the nearest of the five credit vertices."""
    idx = min(
        range(len(CREDIT_TENORS)), key=lambda i: abs(CREDIT_TENORS[i] - tenor_years)
    )
    return CREDIT_TENOR_LABELS[idx]


# -----------------------------------------------------------------------------
# Currency Group Classifications
# -----------------------------------------------------------------------------

class CurrencyVolatility(Enum):
    """IR risk-weight currency volatility group (paragraph 33).

    (1) Regular volatility: USD, EUR, GBP, CHF, AUD, NZD, CAD, SEK, NOK,
        DKK, HKD, KRW, SGD, TWD.
    (2) Low volatility: JPY only.
    (3) High volatility: all other currencies.
    """
    REGULAR = "Regular"
    LOW = "Low"
    HIGH = "High"

    def __str__(self) -> str:
        return self.value


# Regular volatility currencies (paragraph 33(1)).
REGULAR_VOL_CURRENCIES: Tuple[str, ...] = (
    "USD", "EUR", "GBP", "CHF", "AUD", "NZD", "CAD",
    "SEK", "NOK", "DKK", "HKD", "KRW", "SGD", "TWD",
)

# Low volatility currencies (paragraph 33(2)).
LOW_VOL_CURRENCIES: Tuple[str, ...] = ("JPY",)


def get_currency_volatility(currency: str) -> CurrencyVolatility:
    """Get the IR risk-weight volatility group for a currency (paragraph 33).

    Args:
        currency: Three-letter ISO currency code.

    Returns:
        CurrencyVolatility classification.
    """
    currency = currency.upper()
    if currency in REGULAR_VOL_CURRENCIES:
        return CurrencyVolatility.REGULAR
    elif currency in LOW_VOL_CURRENCIES:
        return CurrencyVolatility.LOW
    else:
        return CurrencyVolatility.HIGH


class IRConcentrationGroup(Enum):
    """IR concentration-threshold currency risk group (paragraph 75)."""
    HIGH_VOLATILITY = "High volatility"
    REGULAR_WELL_TRADED = "Regular volatility, well-traded"
    REGULAR_LESS_WELL_TRADED = "Regular volatility, less well-traded"
    LOW_VOLATILITY = "Low volatility"


# Regular volatility, well-traded currencies (paragraph 75(2)).
IR_WELL_TRADED_CURRENCIES: Tuple[str, ...] = ("USD", "EUR", "GBP")


def get_ir_concentration_group(currency: str) -> IRConcentrationGroup:
    """Get the IR concentration-threshold group for a currency (paragraph 75)."""
    currency = currency.upper()
    if currency in IR_WELL_TRADED_CURRENCIES:
        return IRConcentrationGroup.REGULAR_WELL_TRADED
    vol = get_currency_volatility(currency)
    if vol == CurrencyVolatility.REGULAR:
        return IRConcentrationGroup.REGULAR_LESS_WELL_TRADED
    elif vol == CurrencyVolatility.LOW:
        return IRConcentrationGroup.LOW_VOLATILITY
    else:
        return IRConcentrationGroup.HIGH_VOLATILITY


class FXVolatilityGroup(Enum):
    """FX volatility group (paragraphs 67-68).

    High FX volatility currencies: BRL, RUB, TRY. All others are regular.
    """
    REGULAR = "Regular"
    HIGH = "High"

    def __str__(self) -> str:
        return self.value


# High FX volatility currencies (paragraph 67).
FX_HIGH_VOL_CURRENCIES: Tuple[str, ...] = ("BRL", "RUB", "TRY")


def get_fx_volatility_group(currency: str) -> FXVolatilityGroup:
    """Get the FX volatility group for a currency (paragraphs 67-68)."""
    if currency.upper() in FX_HIGH_VOL_CURRENCIES:
        return FXVolatilityGroup.HIGH
    return FXVolatilityGroup.REGULAR


# FX concentration categories (paragraph 80).
FX_CATEGORY_1_CURRENCIES: Tuple[str, ...] = (
    "USD", "EUR", "JPY", "GBP", "AUD", "CHF", "CAD",
)
FX_CATEGORY_2_CURRENCIES: Tuple[str, ...] = (
    "BRL", "CNY", "HKD", "INR", "KRW", "MXN", "NOK",
    "NZD", "RUB", "SEK", "SGD", "TRY", "ZAR",
)


def get_fx_concentration_category(currency: str) -> int:
    """Get the FX concentration category for a currency (paragraph 80).

    Returns:
        1 for significantly material, 2 for frequently traded, 3 for others.
    """
    currency = currency.upper()
    if currency in FX_CATEGORY_1_CURRENCIES:
        return 1
    elif currency in FX_CATEGORY_2_CURRENCIES:
        return 2
    else:
        return 3


# -----------------------------------------------------------------------------
# Bucket Definitions
# -----------------------------------------------------------------------------

# Sentinel identifier for the residual bucket of a risk class.
RESIDUAL_BUCKET = "Residual"


def is_residual_bucket(bucket: Union[str, int]) -> bool:
    """Check whether a bucket identifier denotes the residual bucket."""
    if isinstance(bucket, str):
        return bucket.strip().lower() in ("residual", "res", "-1")
    return bucket == -1


@dataclass(frozen=True)
class IRBucket:
    """Interest rate bucket definition (paragraph 32).

    The set of risk-free yield curves within each currency is a separate
    bucket.

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
            volatility=get_currency_volatility(currency),
        )


@dataclass(frozen=True)
class CreditQualifyingBucket:
    """Credit Qualifying bucket definition (paragraph 38).

    Attributes:
        bucket_number: Bucket identifier (1-12; -1 denotes residual).
        credit_quality: "IG" or "HY/NR".
        sector: Sector description.
    """
    bucket_number: int
    credit_quality: str
    sector: str


CREDIT_QUALIFYING_BUCKETS: Dict[int, CreditQualifyingBucket] = {
    1: CreditQualifyingBucket(1, "IG", "Sovereigns including central banks"),
    2: CreditQualifyingBucket(2, "IG", "Financials including government-backed financials"),
    3: CreditQualifyingBucket(3, "IG", "Basic materials, energy, industrials"),
    4: CreditQualifyingBucket(4, "IG", "Consumer"),
    5: CreditQualifyingBucket(5, "IG", "Technology, telecommunications"),
    6: CreditQualifyingBucket(6, "IG", "Health care, utilities, local government, government-backed corporates (non-financial)"),
    7: CreditQualifyingBucket(7, "HY/NR", "Sovereigns including central banks"),
    8: CreditQualifyingBucket(8, "HY/NR", "Financials including government-backed financials"),
    9: CreditQualifyingBucket(9, "HY/NR", "Basic materials, energy, industrials"),
    10: CreditQualifyingBucket(10, "HY/NR", "Consumer"),
    11: CreditQualifyingBucket(11, "HY/NR", "Technology, telecommunications"),
    12: CreditQualifyingBucket(12, "HY/NR", "Health care, utilities, local government, government-backed corporates (non-financial)"),
}
CREDIT_QUALIFYING_RESIDUAL_BUCKET = CreditQualifyingBucket(-1, "Residual", "Residual")


@dataclass(frozen=True)
class CreditNonQualifyingBucket:
    """Credit Non-Qualifying bucket definition (paragraph 45).

    Attributes:
        bucket_number: Bucket identifier (1-2; -1 denotes residual).
        credit_quality: "IG" or "HY/NR".
    """
    bucket_number: int
    credit_quality: str


CREDIT_NON_QUALIFYING_BUCKETS: Dict[int, CreditNonQualifyingBucket] = {
    1: CreditNonQualifyingBucket(1, "IG"),
    2: CreditNonQualifyingBucket(2, "HY/NR"),
}
CREDIT_NON_QUALIFYING_RESIDUAL_BUCKET = CreditNonQualifyingBucket(-1, "Residual")


@dataclass(frozen=True)
class EquityBucket:
    """Equity bucket definition (paragraph 50).

    Attributes:
        bucket_number: Bucket identifier (1-12; -1 denotes residual).
        size: "Large", "Small", or "All".
        region: "Emerging", "Developed", or "All".
        sector: Sector description.
    """
    bucket_number: int
    size: str
    region: str
    sector: str


EQUITY_BUCKETS: Dict[int, EquityBucket] = {
    1: EquityBucket(1, "Large", "Emerging", "Consumer goods and services, transportation and storage, administrative and support service activities, healthcare, utilities"),
    2: EquityBucket(2, "Large", "Emerging", "Telecommunications, industrials"),
    3: EquityBucket(3, "Large", "Emerging", "Basic materials, energy, agriculture, manufacturing, mining and quarrying"),
    4: EquityBucket(4, "Large", "Emerging", "Financials including gov't-backed financials, real estate activities, technology"),
    5: EquityBucket(5, "Large", "Developed", "Consumer goods and services, transportation and storage, administrative and support service activities, healthcare, utilities"),
    6: EquityBucket(6, "Large", "Developed", "Telecommunications, industrials"),
    7: EquityBucket(7, "Large", "Developed", "Basic materials, energy, agriculture, manufacturing, mining and quarrying"),
    8: EquityBucket(8, "Large", "Developed", "Financials including gov't-backed financials, real estate activities, technology"),
    9: EquityBucket(9, "Small", "Emerging", "All sectors"),
    10: EquityBucket(10, "Small", "Developed", "All sectors"),
    11: EquityBucket(11, "All", "All", "Indexes, Funds, ETFs"),
    12: EquityBucket(12, "All", "All", "Volatility Indexes"),
}
EQUITY_RESIDUAL_BUCKET = EquityBucket(-1, "All", "All", "Residual")

# Bucket 12 (Volatility Indexes) has zero curvature exposure (paragraph 11(b)).
EQUITY_VOLATILITY_INDEX_BUCKET = 12


@dataclass(frozen=True)
class CommodityBucket:
    """Commodity bucket definition (paragraph 61).

    Attributes:
        bucket_number: Bucket identifier (1-17).
        commodity_type: Commodity type description.
    """
    bucket_number: int
    commodity_type: str


COMMODITY_BUCKETS: Dict[int, CommodityBucket] = {
    1: CommodityBucket(1, "Coal"),
    2: CommodityBucket(2, "Crude"),
    3: CommodityBucket(3, "Light Ends"),
    4: CommodityBucket(4, "Middle Distillates"),
    5: CommodityBucket(5, "Heavy Distillates"),
    6: CommodityBucket(6, "North America Natural Gas"),
    7: CommodityBucket(7, "European Natural Gas"),
    8: CommodityBucket(8, "North American Power"),
    9: CommodityBucket(9, "European Power and Carbon"),
    10: CommodityBucket(10, "Freight"),
    11: CommodityBucket(11, "Base Metals"),
    12: CommodityBucket(12, "Precious Metals"),
    13: CommodityBucket(13, "Grains and Oilseed"),
    14: CommodityBucket(14, "Softs and Other Agriculturals"),
    15: CommodityBucket(15, "Livestock and Dairy"),
    16: CommodityBucket(16, "Other"),
    17: CommodityBucket(17, "Indexes"),
}


@dataclass(frozen=True)
class FXBucket:
    """FX bucket definition (paragraph 66).

    All FX sensitivities are within a single bucket. Note the cross-bucket
    curvature calculations of paragraph 11(d) are still required on the
    single bucket.

    Attributes:
        bucket_number: Always 1 for FX.
    """
    bucket_number: int = 1


FX_BUCKET = FXBucket()
