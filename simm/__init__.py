"""
SIMM Module - ISDA Standard Initial Margin Model.

This module provides the foundational data structures and utilities
for ISDA SIMM v2.6 calculations, including:

- Risk class taxonomy (IR, Credit, Equity, Commodity, FX)
- Product class taxonomy (RatesFX, Credit, Equity, Commodity)
- Margin type taxonomy (Delta, Vega, Curvature, BaseCorr)
- Sensitivity data models and protocols
- CRIF (Common Risk Interchange Format) parsing and export
- SIMM configuration

This is the foundation module. Additional modules provide:
- Calibration data (risk weights, correlations)
- Sensitivity calculation engines
- Margin aggregation logic
- Reporting and attribution
"""
from .config import SIMMConfig, SIMMVersion
from .crif import (
    CRIFHeader,
    CRIFRecord,
    CRIFValidationError,
    crif_to_sensitivities,
    parse_crif_csv,
    sensitivities_to_crif,
    write_crif_csv,
)
from .sensitivity import (
    AnySensitivity,
    BaseCorrSensitivity,
    BaseSensitivity,
    CommodityDeltaSensitivity,
    CommodityVegaSensitivity,
    CreditDeltaSensitivity,
    CreditVegaSensitivity,
    CurvatureSensitivity,
    EquityDeltaSensitivity,
    EquityVegaSensitivity,
    FXDeltaSensitivity,
    FXVegaSensitivity,
    IRDeltaSensitivity,
    IRVegaSensitivity,
    Sensitivity,
    SensitivityCollection,
)
from .taxonomy import (
    # Enums
    CurrencyVolatility,
    IRSubCurve,
    MarginType,
    ProductClass,
    RiskClass,
    SensitivityType,
    # Tenors
    CREDIT_TENOR_LABELS,
    CREDIT_TENORS,
    IR_TENOR_LABELS,
    IR_TENORS,
    VEGA_TENOR_LABELS,
    VEGA_TENORS,
    # Currency classifications
    HIGH_VOL_CURRENCIES,
    LOW_VOL_CURRENCIES,
    get_currency_volatility,
    # Bucket definitions
    COMMODITY_BUCKETS,
    CREDIT_NON_QUALIFYING_BUCKETS,
    CREDIT_NON_QUALIFYING_RESIDUAL_BUCKET,
    CREDIT_QUALIFYING_BUCKETS,
    CREDIT_QUALIFYING_RESIDUAL_BUCKET,
    EQUITY_BUCKETS,
    EQUITY_RESIDUAL_BUCKET,
    FX_BUCKET,
    CommodityBucket,
    CreditNonQualifyingBucket,
    CreditQualifyingBucket,
    EquityBucket,
    FXBucket,
    IRBucket,
)

__all__ = [
    # Config
    "SIMMConfig",
    "SIMMVersion",
    # CRIF
    "CRIFHeader",
    "CRIFRecord",
    "CRIFValidationError",
    "crif_to_sensitivities",
    "parse_crif_csv",
    "sensitivities_to_crif",
    "write_crif_csv",
    # Sensitivity
    "AnySensitivity",
    "BaseCorrSensitivity",
    "BaseSensitivity",
    "CommodityDeltaSensitivity",
    "CommodityVegaSensitivity",
    "CreditDeltaSensitivity",
    "CreditVegaSensitivity",
    "CurvatureSensitivity",
    "EquityDeltaSensitivity",
    "EquityVegaSensitivity",
    "FXDeltaSensitivity",
    "FXVegaSensitivity",
    "IRDeltaSensitivity",
    "IRVegaSensitivity",
    "Sensitivity",
    "SensitivityCollection",
    # Taxonomy - Enums
    "CurrencyVolatility",
    "IRSubCurve",
    "MarginType",
    "ProductClass",
    "RiskClass",
    "SensitivityType",
    # Taxonomy - Tenors
    "CREDIT_TENOR_LABELS",
    "CREDIT_TENORS",
    "IR_TENOR_LABELS",
    "IR_TENORS",
    "VEGA_TENOR_LABELS",
    "VEGA_TENORS",
    # Taxonomy - Currency classifications
    "HIGH_VOL_CURRENCIES",
    "LOW_VOL_CURRENCIES",
    "get_currency_volatility",
    # Taxonomy - Bucket definitions
    "COMMODITY_BUCKETS",
    "CREDIT_NON_QUALIFYING_BUCKETS",
    "CREDIT_NON_QUALIFYING_RESIDUAL_BUCKET",
    "CREDIT_QUALIFYING_BUCKETS",
    "CREDIT_QUALIFYING_RESIDUAL_BUCKET",
    "EQUITY_BUCKETS",
    "EQUITY_RESIDUAL_BUCKET",
    "FX_BUCKET",
    "CommodityBucket",
    "CreditNonQualifyingBucket",
    "CreditQualifyingBucket",
    "EquityBucket",
    "FXBucket",
    "IRBucket",
]
