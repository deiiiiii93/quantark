"""
Risk-factor netting and weighted sensitivity calculation for SIMM.

Implements:
- Netting of sensitivities to each risk factor (paragraphs 7(a), 8(a), 10(d))
- Delta risk weight lookup (Sections D-I)
- Vega risk exposure construction VR_k = VRW * (sum_i VR_ik) * VCR
  (paragraph 10(c)-(d)), where the historical volatility ratio HVR is
  applied to Equity, FX and Commodity vegas
- The weighted sensitivity WS_k = RW_k * s_k * CR_k (paragraphs 7(b), 8(b))

Input sensitivity amounts follow the conventions documented in
quantark.simm.sensitivity: PV01/CS01 per 1bp for IR/Credit delta, per-1%
relative for Equity/Commodity/FX delta, and vol-weighted vega
(sigma_kj * dV/dsigma) for all vega records.
"""

from dataclasses import dataclass
from typing import Dict, Hashable, List, Sequence, Union

from quantark.simm.taxonomy import (
    MarginType,
    RiskClass,
    get_currency_volatility,
    get_fx_volatility_group,
    is_residual_bucket,
)
from quantark.simm.sensitivity import AnySensitivity
from quantark.simm.calibration.ir import (
    IR_DELTA_RISK_WEIGHTS,
    IR_INFLATION_RISK_WEIGHT,
    IR_XCCY_BASIS_RISK_WEIGHT,
    IR_VRW,
)
from quantark.simm.calibration.credit_qualifying import (
    CREDIT_QUALIFYING_RISK_WEIGHTS,
    CREDIT_QUALIFYING_VRW,
)
from quantark.simm.calibration.credit_non_qualifying import (
    CREDIT_NON_QUALIFYING_RISK_WEIGHTS,
    CREDIT_NON_QUALIFYING_VRW,
)
from quantark.simm.calibration.equity import (
    EQUITY_RISK_WEIGHTS,
    EQUITY_HVR,
    get_equity_vrw,
)
from quantark.simm.calibration.commodity import (
    COMMODITY_RISK_WEIGHTS,
    COMMODITY_HVR,
    COMMODITY_VRW,
)
from quantark.simm.calibration.fx import (
    FX_RISK_WEIGHTS,
    FX_HVR,
    FX_VRW,
)
from quantark.util.exceptions import ValidationError


@dataclass
class NettedSensitivity:
    """Net sensitivity to a single risk factor within a bucket.

    Attributes:
        risk_factor: Risk factor key (see quantark.simm.sensitivity).
        amount: Net sensitivity s_k (for vega: net vol-weighted vega with
            HVR applied, i.e. sum_i VR_ik of paragraph 10(c)).
        qualifier: CRIF-style qualifier (for reporting).
        group: Group name for Credit Non-Qualifying correlations.
    """
    risk_factor: Hashable
    amount: float
    qualifier: str = ""
    group: str = ""


@dataclass
class WeightedSensitivity:
    """A netted risk factor with its weighted value.

    Attributes:
        risk_factor: Risk factor key.
        bucket: The bucket.
        net_sensitivity: The net input sensitivity s_k (post-HVR for vega).
        risk_weight: The applied risk weight (RW_k or VRW).
        concentration_factor: The CR_k / VCR_k factor applied.
        weighted_value: WS_k = RW * s * CR (or VR_k = VRW * s * VCR).
        qualifier: CRIF-style qualifier (for reporting).
        group: Group name for Credit Non-Qualifying correlations.
    """
    risk_factor: Hashable
    bucket: Union[str, int]
    net_sensitivity: float
    risk_weight: float
    concentration_factor: float
    weighted_value: float
    qualifier: str = ""
    group: str = ""


def hvr_for_vega(risk_class: RiskClass) -> float:
    """Historical volatility ratio HVR_c applied to vega (paragraph 10(c)).

    HVR applies to Equity, Commodity and FX vega only; for Interest Rate
    and Credit the ratio is 1 at this stage (the IR HVR enters the
    curvature margin scale factor instead, paragraph 11(d)).
    """
    if risk_class == RiskClass.EQUITY:
        return EQUITY_HVR
    elif risk_class == RiskClass.COMMODITY:
        return COMMODITY_HVR
    elif risk_class == RiskClass.FX:
        return FX_HVR
    return 1.0


def net_by_risk_factor(
    sensitivities: Sequence[AnySensitivity],
    margin_type: MarginType,
) -> List[NettedSensitivity]:
    """Net sensitivities across instruments to each risk factor.

    For vega, the historical volatility ratio is applied so the result is
    sum_i VR_ik of paragraph 10(c).

    Args:
        sensitivities: Sensitivities of one (risk class, margin type,
            bucket) cell.
        margin_type: DELTA or VEGA.

    Returns:
        One NettedSensitivity per distinct risk factor.
    """
    netted: Dict[Hashable, NettedSensitivity] = {}
    for sens in sensitivities:
        hvr = hvr_for_vega(sens.risk_class) if margin_type == MarginType.VEGA else 1.0
        rf = sens.risk_factor
        entry = netted.get(rf)
        if entry is None:
            netted[rf] = NettedSensitivity(
                risk_factor=rf,
                amount=hvr * sens.amount,
                qualifier=sens.qualifier,
                group=getattr(sens, "group_name", ""),
            )
        else:
            entry.amount += hvr * sens.amount
    return list(netted.values())


def delta_risk_weight(
    risk_class: RiskClass,
    bucket: Union[str, int],
    risk_factor: Hashable,
    calculation_currency: str = "USD",
) -> float:
    """Delta risk weight RW_k for a risk factor (Sections D-I).

    Args:
        risk_class: The risk class.
        bucket: Bucket identifier (currency for IR).
        risk_factor: Risk factor key.
        calculation_currency: SIMM calculation currency. The FX risk
            weight is zero for the calculation currency itself, and the
            FX table is keyed by both volatility groups (paragraph 69).

    Returns:
        The risk weight RW_k.
    """
    if risk_class == RiskClass.INTEREST_RATE:
        kind = risk_factor[0]  # type: ignore[index]
        if kind == "Inflation":
            return IR_INFLATION_RISK_WEIGHT
        elif kind == "XCcyBasis":
            return IR_XCCY_BASIS_RISK_WEIGHT
        elif kind == "Yield":
            group = get_currency_volatility(str(bucket))
            vertex = risk_factor[1]  # type: ignore[index]
            try:
                return IR_DELTA_RISK_WEIGHTS[group][vertex]
            except KeyError as exc:
                raise ValidationError(f"Unknown IR vertex: {vertex}") from exc
        raise ValidationError(f"Unknown IR risk factor kind: {kind}")

    elif risk_class == RiskClass.CREDIT_QUALIFYING:
        key = "Residual" if is_residual_bucket(bucket) else int(bucket)
        return CREDIT_QUALIFYING_RISK_WEIGHTS[key]

    elif risk_class == RiskClass.CREDIT_NON_QUALIFYING:
        key = "Residual" if is_residual_bucket(bucket) else int(bucket)
        return CREDIT_NON_QUALIFYING_RISK_WEIGHTS[key]

    elif risk_class == RiskClass.EQUITY:
        key = "Residual" if is_residual_bucket(bucket) else int(bucket)
        return EQUITY_RISK_WEIGHTS[key]

    elif risk_class == RiskClass.COMMODITY:
        return COMMODITY_RISK_WEIGHTS[int(bucket)]

    elif risk_class == RiskClass.FX:
        currency = str(risk_factor[0])  # type: ignore[index]
        calc = calculation_currency.upper()
        if currency == calc:
            # No FX risk factor for the calculation currency (paragraph 69).
            return 0.0
        g_ccy = get_fx_volatility_group(currency)
        g_calc = get_fx_volatility_group(calc)
        return FX_RISK_WEIGHTS[(g_ccy, g_calc)]

    raise ValidationError(f"Unknown risk class: {risk_class}")


def vega_risk_weight(risk_class: RiskClass, bucket: Union[str, int]) -> float:
    """Vega risk weight VRW for a risk class (Sections D-I)."""
    if risk_class == RiskClass.INTEREST_RATE:
        return IR_VRW
    elif risk_class == RiskClass.CREDIT_QUALIFYING:
        return CREDIT_QUALIFYING_VRW
    elif risk_class == RiskClass.CREDIT_NON_QUALIFYING:
        return CREDIT_NON_QUALIFYING_VRW
    elif risk_class == RiskClass.EQUITY:
        if is_residual_bucket(bucket):
            return get_equity_vrw("Residual")
        return get_equity_vrw(int(bucket))
    elif risk_class == RiskClass.COMMODITY:
        return COMMODITY_VRW
    elif risk_class == RiskClass.FX:
        return FX_VRW
    raise ValidationError(f"Unknown risk class: {risk_class}")


class WeightedSensitivityCalculator:
    """Builds weighted sensitivities WS_k = RW_k * s_k * CR_k.

    For vega, builds VR_k = VRW * (sum_i VR_ik) * VCR_k (paragraph 10(d));
    the netted input amounts already include sigma_kj and HVR.
    """

    def __init__(self, calculation_currency: str = "USD"):
        self.calculation_currency = calculation_currency.upper()

    def calculate(
        self,
        netted: List[NettedSensitivity],
        risk_class: RiskClass,
        margin_type: MarginType,
        bucket: Union[str, int],
        cr_values: Dict[Hashable, float],
    ) -> List[WeightedSensitivity]:
        """Weight netted sensitivities by risk weight and concentration.

        Args:
            netted: Netted sensitivities per risk factor.
            risk_class: The risk class.
            margin_type: DELTA or VEGA.
            bucket: The bucket identifier.
            cr_values: Concentration risk factor per risk factor key.

        Returns:
            List of WeightedSensitivity objects.
        """
        results = []
        for ns in netted:
            if margin_type == MarginType.VEGA:
                rw = vega_risk_weight(risk_class, bucket)
            else:
                rw = delta_risk_weight(
                    risk_class, bucket, ns.risk_factor, self.calculation_currency
                )
            cr = cr_values.get(ns.risk_factor, 1.0)
            results.append(WeightedSensitivity(
                risk_factor=ns.risk_factor,
                bucket=bucket,
                net_sensitivity=ns.amount,
                risk_weight=rw,
                concentration_factor=cr,
                weighted_value=rw * ns.amount * cr,
                qualifier=ns.qualifier,
                group=ns.group,
            ))
        return results
