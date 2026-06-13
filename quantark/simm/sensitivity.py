"""
SIMM Sensitivity Module.

Defines the sensitivity data model for ISDA SIMM v2.6. Each sensitivity
record identifies one risk factor (Section C.1) and carries the net
sensitivity amount in calculation-currency units, following the shift
conventions of Sections C.2 and C.3:

- Interest Rate / Credit delta: s = V(x + 1bp) - V(x)  (PV01 / CS01 per 1bp)
- Equity / Commodity / FX delta: s = V(x + 1%.x) - V(x)  (per 1% relative)
- Vega: amount = sigma_kj * dV/dsigma, i.e. the *vol-weighted* vega
  VR contribution of paragraph 10(c) before HVR, VRW and concentration.
  For Equity, FX and Commodity, sigma_kj is derived from the delta risk
  weight (paragraph 10(b)); use the ``vol_weighted_vega`` helpers. For
  Interest Rate and Credit, sigma_kj is the implied ATM volatility quoted
  per paragraph 10(a).

Each record exposes:
- ``risk_class``, ``margin_type``, ``bucket``, ``qualifier`` (CRIF-style)
- ``risk_factor``: hashable key identifying the netting unit within its
  bucket per Section C.1 (e.g. (vertex, sub-curve) for IR delta)
- ``product_class``: the product class of the trade (paragraph 6); if not
  provided, the conventional default for the risk class is used.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Protocol, Tuple, Union, runtime_checkable

from quantark.simm.calibration.commodity import COMMODITY_RISK_WEIGHTS
from quantark.simm.calibration.equity import EQUITY_RISK_WEIGHTS
from quantark.simm.calibration.fx import FX_RISK_WEIGHTS
from quantark.simm.calibration.accessors import VOL_SCALE
from quantark.simm.taxonomy import (
    DEFAULT_PRODUCT_CLASS,
    IRSubCurve,
    MarginType,
    ProductClass,
    RiskClass,
    credit_tenor_to_vertex_label,
    get_fx_volatility_group,
    is_residual_bucket,
    tenor_to_vertex_label,
)


@runtime_checkable
class Sensitivity(Protocol):
    """Base protocol for all SIMM sensitivities."""

    @property
    def risk_class(self) -> RiskClass:
        """The risk class this sensitivity belongs to."""
        ...

    @property
    def margin_type(self) -> MarginType:
        """Delta, Vega, Curvature, or BaseCorr."""
        ...

    @property
    def amount(self) -> float:
        """The sensitivity value in calculation currency."""
        ...

    @property
    def bucket(self) -> Any:
        """The bucket this sensitivity is assigned to."""
        ...

    @property
    def qualifier(self) -> str:
        """The qualifier identifying the risk factor (currency, issuer, etc.)."""
        ...

    @property
    def risk_factor(self) -> Hashable:
        """Hashable key identifying the netting unit within the bucket."""
        ...


@dataclass
class BaseSensitivity:
    """Base class for all sensitivity dataclasses.

    Attributes:
        trade_id: Identifier for the trade this sensitivity belongs to.
        amount: The sensitivity value in calculation currency.
        amount_currency: Currency of the sensitivity amount.
        product_class: Product class of the trade (paragraph 6). When None,
            the conventional default for the risk class is used (e.g. the
            IR delta of an equity derivative should be tagged
            ProductClass.EQUITY explicitly).
    """
    trade_id: str
    amount: float
    amount_currency: str = "USD"
    product_class: Optional[ProductClass] = None

    @property
    def effective_product_class(self) -> ProductClass:
        """Product class used for aggregation (explicit or default)."""
        if self.product_class is not None:
            return self.product_class
        return DEFAULT_PRODUCT_CLASS[self.risk_class]  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# Interest Rate
# -----------------------------------------------------------------------------

@dataclass
class IRDeltaSensitivity(BaseSensitivity):
    """Interest rate delta sensitivity (PV01 per 1bp, paragraph 22).

    Attributes:
        currency: The currency of the rate curve (also serves as bucket).
        tenor: Tenor in years; snapped to the nearest of the 12 vertices.
        sub_curve: The sub yield curve (paragraph 14).
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
        return self.currency.upper()

    @property
    def qualifier(self) -> str:
        return self.currency.upper()

    @property
    def vertex(self) -> str:
        """The SIMM vertex label for this tenor."""
        return tenor_to_vertex_label(self.tenor)

    @property
    def risk_factor(self) -> Hashable:
        return ("Yield", self.vertex, self.sub_curve.value)


@dataclass
class IRInflationDeltaSensitivity(BaseSensitivity):
    """Flat inflation rate delta sensitivity for a currency (paragraph 14).

    All sensitivities to inflation rates for the same currency are fully
    offset, so there is a single inflation risk factor per currency.
    """
    currency: str = "USD"

    @property
    def risk_class(self) -> RiskClass:
        return RiskClass.INTEREST_RATE

    @property
    def margin_type(self) -> MarginType:
        return MarginType.DELTA

    @property
    def bucket(self) -> str:
        return self.currency.upper()

    @property
    def qualifier(self) -> str:
        return self.currency.upper()

    @property
    def risk_factor(self) -> Hashable:
        return ("Inflation",)


@dataclass
class IRXCcyBasisSensitivity(BaseSensitivity):
    """Flat cross-currency basis swap spread sensitivity (paragraph 14).

    Cross-currency basis swap sensitivities are not scaled by the
    concentration risk factor and are excluded from its computation
    (paragraph 7(b)).
    """
    currency: str = "USD"

    @property
    def risk_class(self) -> RiskClass:
        return RiskClass.INTEREST_RATE

    @property
    def margin_type(self) -> MarginType:
        return MarginType.DELTA

    @property
    def bucket(self) -> str:
        return self.currency.upper()

    @property
    def qualifier(self) -> str:
        return self.currency.upper()

    @property
    def risk_factor(self) -> Hashable:
        return ("XCcyBasis",)


@dataclass
class IRVegaSensitivity(BaseSensitivity):
    """Interest rate vega sensitivity (vol-weighted, paragraph 10(a)/(c)).

    The amount is sigma_kj * dV/dsigma for an option with expiry equal to
    ``option_tenor``. The risk factor is the option expiry vertex; vegas
    with the same expiry but different underlying swap maturities net.

    Attributes:
        currency: The currency of the rate curve.
        option_tenor: Option expiry in years; snapped to the 12 vertices.
        is_inflation: True for inflation swaption vega (paragraph 10(a)).
    """
    currency: str = "USD"
    option_tenor: float = 1.0
    is_inflation: bool = False

    @property
    def risk_class(self) -> RiskClass:
        return RiskClass.INTEREST_RATE

    @property
    def margin_type(self) -> MarginType:
        return MarginType.VEGA

    @property
    def bucket(self) -> str:
        return self.currency.upper()

    @property
    def qualifier(self) -> str:
        return self.currency.upper()

    @property
    def vertex(self) -> str:
        """The SIMM expiry vertex label."""
        return tenor_to_vertex_label(self.option_tenor)

    @property
    def risk_factor(self) -> Hashable:
        kind = "InflationVol" if self.is_inflation else "Vol"
        return (kind, self.vertex)


# -----------------------------------------------------------------------------
# Credit
# -----------------------------------------------------------------------------

@dataclass
class CreditDeltaSensitivity(BaseSensitivity):
    """Credit delta sensitivity (CS01 per 1bp, paragraphs 23-24).

    Attributes:
        issuer: Issuer/seniority identifier (Qualifying, paragraph 15) or
            issuer/tranche identifier (Non-Qualifying, paragraph 16).
        bucket_number: SIMM bucket (1-12 for CQ, 1-2 for CNQ, -1 residual).
        tenor: Tenor in years; snapped to the five credit vertices.
        is_qualifying: True for Credit Qualifying.
        payment_currency: Payment currency of the trade. Sensitivities to
            different payment currencies (Quanto CDS) are distinct risk
            factors of the same issuer/seniority (paragraph 15).
        group: Group name for Non-Qualifying intra-bucket correlation
            (such as CMBX or ABX, paragraph 48). Defaults to the issuer.
    """
    issuer: str = ""
    bucket_number: Union[int, str] = 1
    tenor: float = 5.0
    is_qualifying: bool = True
    payment_currency: str = ""
    group: str = ""

    @property
    def risk_class(self) -> RiskClass:
        if self.is_qualifying:
            return RiskClass.CREDIT_QUALIFYING
        return RiskClass.CREDIT_NON_QUALIFYING

    @property
    def margin_type(self) -> MarginType:
        return MarginType.DELTA

    @property
    def bucket(self) -> Union[int, str]:
        return self.bucket_number

    @property
    def qualifier(self) -> str:
        return self.issuer

    @property
    def vertex(self) -> str:
        """The SIMM credit vertex label for this tenor."""
        return credit_tenor_to_vertex_label(self.tenor)

    @property
    def group_name(self) -> str:
        """Group name used for Non-Qualifying correlations."""
        return self.group or self.issuer

    @property
    def risk_factor(self) -> Hashable:
        return (self.issuer, self.vertex, self.payment_currency)


@dataclass
class CreditVegaSensitivity(BaseSensitivity):
    """Credit vega sensitivity (vol-weighted, paragraph 10(a)/(c)).

    Attributes:
        issuer: Issuer/seniority (CQ) or issuer/tranche (CNQ) identifier.
            Index vega need not be allocated to underlying issuers
            (paragraphs 15-16); pass the index name and its bucket.
        bucket_number: SIMM bucket number (-1 / "Residual" for residual).
        option_tenor: Option expiry in years; snapped to credit vertices.
        is_qualifying: True for Credit Qualifying.
        group: Group name for Non-Qualifying correlations.
    """
    issuer: str = ""
    bucket_number: Union[int, str] = 1
    option_tenor: float = 1.0
    is_qualifying: bool = True
    group: str = ""

    @property
    def risk_class(self) -> RiskClass:
        if self.is_qualifying:
            return RiskClass.CREDIT_QUALIFYING
        return RiskClass.CREDIT_NON_QUALIFYING

    @property
    def margin_type(self) -> MarginType:
        return MarginType.VEGA

    @property
    def bucket(self) -> Union[int, str]:
        return self.bucket_number

    @property
    def qualifier(self) -> str:
        return self.issuer

    @property
    def vertex(self) -> str:
        """The SIMM credit expiry vertex label."""
        return credit_tenor_to_vertex_label(self.option_tenor)

    @property
    def group_name(self) -> str:
        """Group name used for Non-Qualifying correlations."""
        return self.group or self.issuer

    @property
    def risk_factor(self) -> Hashable:
        return (self.issuer, self.vertex)


@dataclass
class BaseCorrSensitivity(BaseSensitivity):
    """Base correlation sensitivity BC01 (paragraphs 13 and 25).

    One flat risk factor per index family (such as CDX IG or iTraxx Main);
    risks to the same family fully offset (paragraph 15).
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

    @property
    def risk_factor(self) -> Hashable:
        return (self.index_name,)


# -----------------------------------------------------------------------------
# Equity
# -----------------------------------------------------------------------------

@dataclass
class EquityDeltaSensitivity(BaseSensitivity):
    """Equity delta sensitivity (per 1% relative move, paragraph 26).

    Attributes:
        issuer: Equity identifier (ISIN, ticker, index name, etc.).
        bucket_number: SIMM bucket (1-12, -1 / "Residual" for residual).
    """
    issuer: str = ""
    bucket_number: Union[int, str] = 1

    @property
    def risk_class(self) -> RiskClass:
        return RiskClass.EQUITY

    @property
    def margin_type(self) -> MarginType:
        return MarginType.DELTA

    @property
    def bucket(self) -> Union[int, str]:
        return self.bucket_number

    @property
    def qualifier(self) -> str:
        return self.issuer

    @property
    def risk_factor(self) -> Hashable:
        return (self.issuer,)


@dataclass
class EquityVegaSensitivity(BaseSensitivity):
    """Equity vega sensitivity (vol-weighted, paragraph 10(b)/(c)).

    The amount is sigma_kj * dV/dsigma where sigma_kj is derived from the
    delta risk weight (use ``vol_weighted_vega_equity``). HVR is applied
    by the engine.

    Attributes:
        issuer: Equity identifier.
        bucket_number: SIMM bucket number.
        option_tenor: Option expiry in years (drives the curvature SF).
    """
    issuer: str = ""
    bucket_number: Union[int, str] = 1
    option_tenor: float = 1.0

    @property
    def risk_class(self) -> RiskClass:
        return RiskClass.EQUITY

    @property
    def margin_type(self) -> MarginType:
        return MarginType.VEGA

    @property
    def bucket(self) -> Union[int, str]:
        return self.bucket_number

    @property
    def qualifier(self) -> str:
        return self.issuer

    @property
    def vertex(self) -> str:
        """The SIMM expiry vertex label."""
        return tenor_to_vertex_label(self.option_tenor)

    @property
    def risk_factor(self) -> Hashable:
        return (self.issuer,)


# -----------------------------------------------------------------------------
# Commodity
# -----------------------------------------------------------------------------

@dataclass
class CommodityDeltaSensitivity(BaseSensitivity):
    """Commodity delta sensitivity (per 1% relative move, paragraph 27).

    Risks to forward prices are allocated back to spot price risks
    (paragraph 18).

    Attributes:
        commodity_name: Commodity identifier (e.g. "Precious Metals Gold").
        bucket_number: SIMM bucket number (1-17).
    """
    commodity_name: str = ""
    bucket_number: int = 1

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

    @property
    def risk_factor(self) -> Hashable:
        return (self.commodity_name,)


@dataclass
class CommodityVegaSensitivity(BaseSensitivity):
    """Commodity vega sensitivity (vol-weighted, paragraph 10(b)/(c)).

    Attributes:
        commodity_name: Commodity identifier.
        bucket_number: SIMM bucket number.
        option_tenor: Option expiry in years (drives the curvature SF).
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

    @property
    def vertex(self) -> str:
        """The SIMM expiry vertex label."""
        return tenor_to_vertex_label(self.option_tenor)

    @property
    def risk_factor(self) -> Hashable:
        return (self.commodity_name,)


# -----------------------------------------------------------------------------
# FX
# -----------------------------------------------------------------------------

@dataclass
class FXDeltaSensitivity(BaseSensitivity):
    """FX delta sensitivity (per 1% relative move, paragraph 28).

    The risk factor is the exchange rate between ``currency`` and the
    calculation currency (paragraph 19). There is no FX risk factor for
    the calculation currency itself.

    Attributes:
        currency: The (non-calculation) currency of the FX risk factor.
    """
    currency: str = ""

    @property
    def risk_class(self) -> RiskClass:
        return RiskClass.FX

    @property
    def margin_type(self) -> MarginType:
        return MarginType.DELTA

    @property
    def bucket(self) -> int:
        return 1  # FX has a single bucket (paragraph 66)

    @property
    def qualifier(self) -> str:
        return self.currency.upper()

    @property
    def risk_factor(self) -> Hashable:
        return (self.currency.upper(),)


@dataclass
class FXVegaSensitivity(BaseSensitivity):
    """FX vega sensitivity (vol-weighted, paragraph 10(b)/(c)).

    The vega risk factors are currency pairs (paragraph 19). The pair is
    unordered: "USDJPY" and "JPYUSD" identify the same risk factor.

    Attributes:
        currency_pair: Six-character currency pair (e.g. "EURUSD").
        option_tenor: Option expiry in years (drives the curvature SF).
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
        return 1  # FX has a single bucket (paragraph 66)

    @property
    def qualifier(self) -> str:
        return self.currency_pair.upper()

    @property
    def currencies(self) -> Tuple[str, str]:
        """The two currencies of the pair."""
        pair = self.currency_pair.upper()
        return pair[:3], pair[3:6]

    @property
    def vertex(self) -> str:
        """The SIMM expiry vertex label."""
        return tenor_to_vertex_label(self.option_tenor)

    @property
    def risk_factor(self) -> Hashable:
        return frozenset(self.currencies)


# -----------------------------------------------------------------------------
# Explicit curvature input
# -----------------------------------------------------------------------------

@dataclass
class CurvatureSensitivity(BaseSensitivity):
    """Explicit curvature exposure CVR_ik (paragraph 11(a)).

    By default the engine derives curvature exposures from vega
    sensitivities (CVR = SF(t) * vol-weighted vega). Use this class to
    supply pre-computed CVR values directly instead; the amount must
    already include the SF(t) scaling.

    Attributes:
        risk_class_value: The risk class this curvature applies to.
        qualifier_value: Risk factor identifier.
        bucket_value: Bucket assignment.
    """
    risk_class_value: RiskClass = RiskClass.INTEREST_RATE
    qualifier_value: str = ""
    bucket_value: Union[str, int] = ""
    label1: str = ""
    label2: str = ""
    risk_factor_value: Optional[Hashable] = None
    group: str = ""

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

    @property
    def risk_factor(self) -> Hashable:
        if self.risk_factor_value is not None:
            return self.risk_factor_value
        if self.risk_class_value == RiskClass.INTEREST_RATE and self.label1:
            return ("Yield", self.label1, self.label2 or IRSubCurve.OIS.value)
        if self.risk_class_value in (
            RiskClass.CREDIT_QUALIFYING,
            RiskClass.CREDIT_NON_QUALIFYING,
        ):
            return (self.qualifier_value, self.label1, self.label2)
        return (self.qualifier_value,)

    @property
    def group_name(self) -> str:
        return self.group


# Type alias for any sensitivity type
AnySensitivity = Union[
    IRDeltaSensitivity,
    IRInflationDeltaSensitivity,
    IRXCcyBasisSensitivity,
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


# -----------------------------------------------------------------------------
# Vol-weighted vega helpers (paragraph 10(b))
# -----------------------------------------------------------------------------

def equity_vega_sigma(bucket: Union[int, str]) -> float:
    """Implied volatility sigma_kj for an equity risk factor (paragraph 10(b)).

    sigma_kj = RW_k * sqrt(365/14) / PHI_INV(99%), with RW_k the delta risk
    weight of the bucket.
    """
    key = "Residual" if is_residual_bucket(bucket) else bucket
    return EQUITY_RISK_WEIGHTS[key] * VOL_SCALE


def commodity_vega_sigma(bucket: int) -> float:
    """Implied volatility sigma_kj for a commodity risk factor
    (paragraph 10(b)). Commodity index volatilities use the "Indexes"
    bucket risk weight.
    """
    return COMMODITY_RISK_WEIGHTS[bucket] * VOL_SCALE


def fx_vega_sigma(currency_1: str, currency_2: str) -> float:
    """Implied volatility sigma_kj for an FX volatility risk factor
    (paragraph 10(b)).

    The risk weight is the FX delta risk weight entry whose row is the FX
    volatility group of the first currency and whose column is the group
    of the second currency.
    """
    g1 = get_fx_volatility_group(currency_1)
    g2 = get_fx_volatility_group(currency_2)
    return FX_RISK_WEIGHTS[(g1, g2)] * VOL_SCALE


def vol_weighted_vega_equity(raw_vega: float, bucket: Union[int, str]) -> float:
    """Convert a raw equity vega dV/dsigma (per 1 vol point) into the
    vol-weighted amount sigma_kj * dV/dsigma expected by
    EquityVegaSensitivity."""
    return equity_vega_sigma(bucket) * raw_vega


def vol_weighted_vega_commodity(raw_vega: float, bucket: int) -> float:
    """Convert a raw commodity vega into the vol-weighted amount expected
    by CommodityVegaSensitivity."""
    return commodity_vega_sigma(bucket) * raw_vega


def vol_weighted_vega_fx(raw_vega: float, currency_pair: str) -> float:
    """Convert a raw FX vega into the vol-weighted amount expected by
    FXVegaSensitivity."""
    pair = currency_pair.upper()
    return fx_vega_sigma(pair[:3], pair[3:6]) * raw_vega


# -----------------------------------------------------------------------------
# Collection
# -----------------------------------------------------------------------------

@dataclass
class SensitivityCollection:
    """Collection of sensitivities with grouping helpers.

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

    def by_product_class(self, product_class: ProductClass) -> List[AnySensitivity]:
        """Get all sensitivities for a specific product class."""
        return [
            s for s in self.sensitivities
            if s.effective_product_class == product_class
        ]

    def product_classes(self) -> List[ProductClass]:
        """Product classes present in the collection, in enum order."""
        present = {s.effective_product_class for s in self.sensitivities}
        return [pc for pc in ProductClass if pc in present]

    def by_risk_class_and_margin_type(
        self, risk_class: RiskClass, margin_type: MarginType
    ) -> List[AnySensitivity]:
        """Get sensitivities for a risk class and margin type combination."""
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
            result.setdefault(s.bucket, []).append(s)
        return result

    def __len__(self) -> int:
        return len(self.sensitivities)

    def __iter__(self):
        return iter(self.sensitivities)
