"""
Concentration risk factors for SIMM.

Implements the delta concentration risk factors CR (paragraphs 7(b) and
8(b)) and the vega concentration risk factors VCR (paragraph 10(d)) using
the thresholds of Section J:

- Interest Rate: one CR_b per currency over the net of all yield and
  inflation sensitivities; cross-currency basis swap sensitivities are
  excluded and are not scaled by CR.
- Credit (Qualifying and Non-Qualifying): one CR_k per issuer/seniority,
  summing over all tenors and payment currencies of that issuer.
- Equity, Commodity: CR_k per individual risk factor.
- FX: CR_k per currency (delta, threshold by paragraph-80 category) or
  per currency pair (vega, threshold by category pair).
- Base correlation sensitivities take CR = 1 (paragraph 8(b)).

Concentration thresholds are denominated in USD; sensitivity amounts are
assumed to be expressed in USD (or converted upstream).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Union

from quantark.simm.taxonomy import (
    MarginType,
    RiskClass,
    get_fx_concentration_category,
    get_ir_concentration_group,
    is_residual_bucket,
)
from quantark.simm.engines.aggregation.weighted_sensitivity import NettedSensitivity
from quantark.simm.calibration.ir import (
    IR_DELTA_CONCENTRATION_THRESHOLDS,
    IR_VEGA_CONCENTRATION_THRESHOLDS,
)
from quantark.simm.calibration.credit_qualifying import (
    CREDIT_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS,
    CREDIT_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD,
)
from quantark.simm.calibration.credit_non_qualifying import (
    CREDIT_NON_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS,
    CREDIT_NON_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD,
)
from quantark.simm.calibration.equity import (
    EQUITY_DELTA_CONCENTRATION_THRESHOLDS,
    EQUITY_VEGA_CONCENTRATION_THRESHOLDS,
)
from quantark.simm.calibration.commodity import (
    COMMODITY_DELTA_CONCENTRATION_THRESHOLDS,
    COMMODITY_VEGA_CONCENTRATION_THRESHOLDS,
)
from quantark.simm.calibration.fx import (
    FX_DELTA_CONCENTRATION_THRESHOLDS,
    get_fx_vega_concentration_threshold,
)
from quantark.util.exceptions import ValidationError

# Thresholds are quoted in USD millions.
_MM = 1_000_000.0


@dataclass
class ConcentrationResult:
    """Concentration risk factors for the risk factors of one bucket.

    Attributes:
        cr_values: CR_k (or VCR_k) per risk factor key.
        bucket_cr: The bucket-level CR_b / VCR_b (Interest Rate only;
            1.0 elsewhere). Used for the g_bc factor (paragraphs 7(d)
            and 10(f)).
    """
    cr_values: Dict[Hashable, float] = field(default_factory=dict)
    bucket_cr: float = 1.0


def _cr(net_amount: float, threshold_mm: float) -> float:
    """CR = max(1, sqrt(|net| / T)) with T converted from USD millions."""
    if threshold_mm <= 0:
        raise ValidationError(f"Concentration threshold must be positive, got {threshold_mm}")
    return max(1.0, math.sqrt(abs(net_amount) / (threshold_mm * _MM)))


class ConcentrationCalculator:
    """Calculator for concentration risk factors per bucket."""

    def calculate(
        self,
        netted: List[NettedSensitivity],
        risk_class: RiskClass,
        margin_type: MarginType,
        bucket: Union[str, int],
    ) -> ConcentrationResult:
        """Calculate concentration risk factors for one bucket.

        Args:
            netted: Netted sensitivities per risk factor (post-HVR for
                vega, per paragraph 10(c)).
            risk_class: The risk class.
            margin_type: DELTA or VEGA.
            bucket: The bucket identifier (currency for IR and FX).

        Returns:
            ConcentrationResult with CR per risk factor.
        """
        if not netted:
            return ConcentrationResult()

        if risk_class == RiskClass.INTEREST_RATE:
            return self._ir(netted, margin_type, str(bucket))
        elif risk_class in (RiskClass.CREDIT_QUALIFYING, RiskClass.CREDIT_NON_QUALIFYING):
            return self._credit(netted, risk_class, margin_type, bucket)
        elif risk_class in (RiskClass.EQUITY, RiskClass.COMMODITY):
            return self._per_factor(netted, risk_class, margin_type, bucket)
        elif risk_class == RiskClass.FX:
            return self._fx(netted, margin_type)
        raise ValidationError(f"Unknown risk class: {risk_class}")

    def _ir(
        self,
        netted: List[NettedSensitivity],
        margin_type: MarginType,
        currency: str,
    ) -> ConcentrationResult:
        """IR concentration (paragraphs 7(b) and 10(d)).

        CR_b = max(1, sqrt(|sum s| / T_b)) per currency. Inflation
        sensitivities are included in the sum; cross-currency basis swap
        sensitivities are excluded and take CR = 1.
        """
        group = get_ir_concentration_group(currency)
        if margin_type == MarginType.DELTA:
            threshold = IR_DELTA_CONCENTRATION_THRESHOLDS[group]
        else:
            threshold = IR_VEGA_CONCENTRATION_THRESHOLDS[group]

        net = sum(
            ns.amount for ns in netted
            if ns.risk_factor[0] != "XCcyBasis"  # type: ignore[index]
        )
        bucket_cr = _cr(net, threshold)

        cr_values = {
            ns.risk_factor: (
                1.0 if ns.risk_factor[0] == "XCcyBasis" else bucket_cr  # type: ignore[index]
            )
            for ns in netted
        }
        return ConcentrationResult(cr_values=cr_values, bucket_cr=bucket_cr)

    def _credit(
        self,
        netted: List[NettedSensitivity],
        risk_class: RiskClass,
        margin_type: MarginType,
        bucket: Union[str, int],
    ) -> ConcentrationResult:
        """Credit concentration (paragraphs 8(b) and 10(d)).

        CR_k = max(1, sqrt(|sum_j s_j| / T_b)) with the sum over all risk
        factors with the same issuer/seniority, irrespective of tenor or
        payment currency.
        """
        if risk_class == RiskClass.CREDIT_QUALIFYING:
            if margin_type == MarginType.DELTA:
                key = "Residual" if is_residual_bucket(bucket) else int(bucket)
                threshold = CREDIT_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS[key]
            else:
                threshold = CREDIT_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD
        else:
            if margin_type == MarginType.DELTA:
                key = "Residual" if is_residual_bucket(bucket) else int(bucket)
                threshold = CREDIT_NON_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS[key]
            else:
                threshold = CREDIT_NON_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD

        issuer_sums: Dict[str, float] = {}
        for ns in netted:
            issuer = str(ns.risk_factor[0])  # type: ignore[index]
            issuer_sums[issuer] = issuer_sums.get(issuer, 0.0) + ns.amount

        issuer_cr = {issuer: _cr(net, threshold) for issuer, net in issuer_sums.items()}
        cr_values = {
            ns.risk_factor: issuer_cr[str(ns.risk_factor[0])]  # type: ignore[index]
            for ns in netted
        }
        return ConcentrationResult(cr_values=cr_values)

    def _per_factor(
        self,
        netted: List[NettedSensitivity],
        risk_class: RiskClass,
        margin_type: MarginType,
        bucket: Union[str, int],
    ) -> ConcentrationResult:
        """Equity / Commodity concentration: CR_k per risk factor
        (paragraphs 8(b) and 10(d))."""
        if risk_class == RiskClass.EQUITY:
            key = "Residual" if is_residual_bucket(bucket) else int(bucket)
            if margin_type == MarginType.DELTA:
                threshold = EQUITY_DELTA_CONCENTRATION_THRESHOLDS[key]
            else:
                threshold = EQUITY_VEGA_CONCENTRATION_THRESHOLDS[key]
        else:
            if margin_type == MarginType.DELTA:
                threshold = COMMODITY_DELTA_CONCENTRATION_THRESHOLDS[int(bucket)]
            else:
                threshold = COMMODITY_VEGA_CONCENTRATION_THRESHOLDS[int(bucket)]

        cr_values = {ns.risk_factor: _cr(ns.amount, threshold) for ns in netted}
        return ConcentrationResult(cr_values=cr_values)

    def _fx(
        self,
        netted: List[NettedSensitivity],
        margin_type: MarginType,
    ) -> ConcentrationResult:
        """FX concentration (paragraphs 8(b) and 10(d), Sections J.5/J.10).

        Delta thresholds depend on the category of the currency; vega
        thresholds depend on the category pair of the two currencies.
        """
        cr_values: Dict[Hashable, float] = {}
        for ns in netted:
            if margin_type == MarginType.DELTA:
                currency = str(ns.risk_factor[0])  # type: ignore[index]
                category = get_fx_concentration_category(currency)
                threshold = FX_DELTA_CONCENTRATION_THRESHOLDS[category]
            else:
                currencies = list(ns.risk_factor)  # type: ignore[arg-type]
                cats = [get_fx_concentration_category(str(c)) for c in currencies]
                if len(cats) == 1:
                    cats = cats * 2
                threshold = get_fx_vega_concentration_threshold(cats[0], cats[1])
            cr_values[ns.risk_factor] = _cr(ns.amount, threshold)
        return ConcentrationResult(cr_values=cr_values)

    @staticmethod
    def g_factor(cr_b: float, cr_c: float) -> float:
        """g_bc = min(CR_b, CR_c) / max(CR_b, CR_c) (paragraphs 7(d), 10(f))."""
        if cr_b <= 0 or cr_c <= 0:
            return 1.0
        return min(cr_b, cr_c) / max(cr_b, cr_c)

    @staticmethod
    def f_factor(cr_k: float, cr_l: float) -> float:
        """f_kl = min(CR_k, CR_l) / max(CR_k, CR_l) (paragraphs 8(c), 10(e))."""
        if cr_k <= 0 or cr_l <= 0:
            return 1.0
        return min(cr_k, cr_l) / max(cr_k, cr_l)
