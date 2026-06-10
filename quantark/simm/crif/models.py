"""
CRIF Data Models.

This module defines the CRIF (Common Risk Interchange Format) data structures
for ISDA SIMM sensitivity exchange.
"""
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from ..taxonomy import MarginType, ProductClass, RiskClass, SensitivityType


@dataclass
class CRIFHeader:
    """CRIF file header metadata.
    
    Contains information about the CRIF file including valuation date,
    reporting entity, and file format version.
    
    Attributes:
        valuation_date: Date of valuation for the sensitivities.
        reporting_entity: Legal entity identifier of the reporter.
        counterparty: Counterparty legal entity identifier.
        crif_version: CRIF format version (e.g., "2.0").
        im_model: Initial margin model identifier (typically "SIMM").
        base_currency: Base currency for amounts.
    """
    valuation_date: date
    reporting_entity: str = ""
    counterparty: str = ""
    crif_version: str = "2.0"
    im_model: str = "SIMM"
    base_currency: str = "USD"


@dataclass
class CRIFRecord:
    """Single CRIF record representing one sensitivity.
    
    This dataclass follows the ISDA CRIF v2.x specification for
    representing sensitivities in a standardized interchange format.
    
    Attributes:
        trade_id: Unique identifier for the trade.
        valuation_date: Date of the valuation.
        
        risk_type: SIMM risk type (e.g., "Risk_IRCurve", "Risk_FX").
        qualifier: Primary risk factor identifier (currency for IR, issuer for Credit).
        bucket: SIMM bucket assignment.
        label1: First label (tenor for IR/Credit, empty for others).
        label2: Second label (sub-curve for IR, empty for others).
        
        amount: Sensitivity amount.
        amount_currency: Currency of the amount.
        amount_usd: Amount converted to USD (optional).
        
        product_class: SIMM product class (optional, can be inferred).
        risk_class: SIMM risk class (optional, can be inferred).
        
        im_model: Initial margin model (typically "SIMM").
        post_regulations: Posting regulations (e.g., "CFTC", "EMIR").
        collect_regulations: Collection regulations.
        
        call_put: "C" or "P" for options (optional).
        notional: Trade notional (optional, for reference).
        notional_currency: Currency of notional (optional).
        
    Examples:
        IR Delta sensitivity:
        >>> record = CRIFRecord(
        ...     trade_id="TRADE001",
        ...     valuation_date=date(2024, 1, 15),
        ...     risk_type="Risk_IRCurve",
        ...     qualifier="USD",
        ...     bucket="1",
        ...     label1="5y",
        ...     label2="OIS",
        ...     amount=150000.0,
        ...     amount_currency="USD"
        ... )
        
        FX Delta sensitivity:
        >>> record = CRIFRecord(
        ...     trade_id="TRADE002",
        ...     valuation_date=date(2024, 1, 15),
        ...     risk_type="Risk_FX",
        ...     qualifier="EURUSD",
        ...     bucket="",
        ...     label1="",
        ...     label2="",
        ...     amount=50000.0,
        ...     amount_currency="USD"
        ... )
    """
    # Required identification fields
    trade_id: str
    valuation_date: date
    
    # SIMM classification
    risk_type: str
    qualifier: str
    bucket: str
    label1: str = ""
    label2: str = ""
    
    # Sensitivity value
    amount: float = 0.0
    amount_currency: str = "USD"
    amount_usd: Optional[float] = None
    
    # SIMM classification (optional, can be inferred)
    product_class: Optional[str] = None
    risk_class: Optional[str] = None
    
    # Model and regulatory info
    im_model: str = "SIMM"
    post_regulations: Optional[str] = None
    collect_regulations: Optional[str] = None
    
    # Option-specific fields
    call_put: Optional[str] = None
    
    # Reference fields
    notional: Optional[float] = None
    notional_currency: Optional[str] = None
    
    def get_sensitivity_type(self) -> Optional[SensitivityType]:
        """Get the SensitivityType enum from the risk_type string.
        
        Returns:
            SensitivityType if valid, None otherwise.
        """
        for st in SensitivityType:
            if st.value == self.risk_type:
                return st
        return None
    
    def get_risk_class(self) -> Optional[RiskClass]:
        """Infer the RiskClass from the risk_type.
        
        Returns:
            RiskClass if determinable, None otherwise.
        """
        sensitivity_type = self.get_sensitivity_type()
        if sensitivity_type:
            return sensitivity_type.risk_class
        return None
    
    def get_margin_type(self) -> Optional[MarginType]:
        """Infer the MarginType from the risk_type.
        
        Returns:
            MarginType if determinable, None otherwise.
        """
        sensitivity_type = self.get_sensitivity_type()
        if sensitivity_type:
            return sensitivity_type.margin_type
        return None
    
    def get_product_class(self) -> Optional[ProductClass]:
        """Infer or return the ProductClass.
        
        Returns:
            ProductClass based on risk_class mapping.
        """
        if self.product_class:
            for pc in ProductClass:
                if pc.value == self.product_class:
                    return pc
        
        # Infer from risk class
        risk_class = self.get_risk_class()
        if risk_class:
            mapping = {
                RiskClass.INTEREST_RATE: ProductClass.RATES_FX,
                RiskClass.FX: ProductClass.RATES_FX,
                RiskClass.CREDIT_QUALIFYING: ProductClass.CREDIT,
                RiskClass.CREDIT_NON_QUALIFYING: ProductClass.CREDIT,
                RiskClass.EQUITY: ProductClass.EQUITY,
                RiskClass.COMMODITY: ProductClass.COMMODITY,
            }
            return mapping.get(risk_class)
        return None


# Standard CRIF column names
CRIF_COLUMNS = (
    "TradeID",
    "ValuationDate",
    "IMModel",
    "ProductClass",
    "RiskType",
    "Qualifier",
    "Bucket",
    "Label1",
    "Label2",
    "Amount",
    "AmountCurrency",
    "AmountUSD",
    "PostRegulations",
    "CollectRegulations",
    "Notional",
    "NotionalCurrency",
    "CallPut",
)

# Mapping from CRIF column names to CRIFRecord field names
CRIF_COLUMN_MAPPING = {
    "TradeID": "trade_id",
    "ValuationDate": "valuation_date",
    "IMModel": "im_model",
    "ProductClass": "product_class",
    "RiskType": "risk_type",
    "Qualifier": "qualifier",
    "Bucket": "bucket",
    "Label1": "label1",
    "Label2": "label2",
    "Amount": "amount",
    "AmountCurrency": "amount_currency",
    "AmountUSD": "amount_usd",
    "PostRegulations": "post_regulations",
    "CollectRegulations": "collect_regulations",
    "Notional": "notional",
    "NotionalCurrency": "notional_currency",
    "CallPut": "call_put",
}

# Required CRIF columns
CRIF_REQUIRED_COLUMNS = ("TradeID", "ValuationDate", "RiskType", "Qualifier", "Amount", "AmountCurrency")
