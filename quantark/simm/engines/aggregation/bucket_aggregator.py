"""
Within-bucket aggregation for SIMM.

Implements the K_b formula of paragraphs 7(c), 8(c) and 10(e):

    K_b = sqrt( sum_k WS_k^2 + sum_k sum_{l != k} rho_kl f_kl WS_k WS_l )

where rho_kl is the intra-bucket correlation (Sections D-I) and

    f_kl = min(CR_k, CR_l) / max(CR_k, CR_l)

is the concentration adjustment. f_kl is identically 1 in the
Interest Rate risk class for vega (paragraph 10(e)); for IR delta all
risk factors share the same bucket-level CR so the ratio is 1 as well.

The residual bucket uses the same formula with its own correlation
parameters; its K is then added outside the cross-bucket aggregation
(paragraphs 8(d) and 10(f)).
"""

import math
from dataclasses import dataclass, field
from typing import List, Union

from quantark.simm.taxonomy import MarginType, RiskClass, is_residual_bucket
from quantark.simm.engines.aggregation.weighted_sensitivity import WeightedSensitivity
from quantark.simm.engines.aggregation.concentration import ConcentrationCalculator
from quantark.simm.engines.aggregation.correlations import intra_bucket_correlation


@dataclass
class BucketResult:
    """Result of bucket-level aggregation.

    Attributes:
        risk_class: The risk class.
        margin_type: Delta or Vega.
        bucket: The bucket identifier.
        k_b: The aggregated bucket margin K_b.
        ws_sum: Sum of weighted sensitivities (input to S_b).
        bucket_cr: Bucket-level concentration factor (IR only; for g_bc).
        weighted_sensitivities: The weighted sensitivities in this bucket.
        is_residual: True if this is the residual bucket.
    """
    risk_class: RiskClass
    margin_type: MarginType
    bucket: Union[str, int]
    k_b: float
    ws_sum: float
    bucket_cr: float = 1.0
    weighted_sensitivities: List[WeightedSensitivity] = field(default_factory=list)
    is_residual: bool = False

    @property
    def s_b(self) -> float:
        """Capped bucket sum S_b = max(min(ws_sum, K_b), -K_b)."""
        return max(min(self.ws_sum, self.k_b), -self.k_b)


class BucketAggregator:
    """Aggregator for within-bucket margin calculation."""

    def __init__(self, calculation_currency: str = "USD"):
        self.calculation_currency = calculation_currency.upper()

    def aggregate(
        self,
        weighted_sensitivities: List[WeightedSensitivity],
        risk_class: RiskClass,
        margin_type: MarginType,
        bucket: Union[str, int],
        bucket_cr: float = 1.0,
    ) -> BucketResult:
        """Aggregate weighted sensitivities within a bucket.

        Args:
            weighted_sensitivities: One entry per netted risk factor.
            risk_class: The risk class.
            margin_type: DELTA or VEGA.
            bucket: The bucket identifier.
            bucket_cr: Bucket-level CR (IR only; carried for g_bc).

        Returns:
            BucketResult with K_b and the (uncapped) WS sum.
        """
        residual = is_residual_bucket(bucket)

        if not weighted_sensitivities:
            return BucketResult(
                risk_class=risk_class,
                margin_type=margin_type,
                bucket=bucket,
                k_b=0.0,
                ws_sum=0.0,
                bucket_cr=bucket_cr,
                is_residual=residual,
            )

        # f_kl is identically 1 in the Interest Rate risk class
        # (paragraph 10(e); for delta all IR factors share CR_b).
        use_f = risk_class != RiskClass.INTEREST_RATE

        n = len(weighted_sensitivities)
        total = 0.0
        for i in range(n):
            ws_i = weighted_sensitivities[i]
            total += ws_i.weighted_value ** 2
            for j in range(i + 1, n):
                ws_j = weighted_sensitivities[j]
                rho = intra_bucket_correlation(
                    risk_class,
                    margin_type,
                    bucket,
                    ws_i.risk_factor,
                    ws_j.risk_factor,
                    group_1=ws_i.group,
                    group_2=ws_j.group,
                    calculation_currency=self.calculation_currency,
                )
                f = 1.0
                if use_f:
                    f = ConcentrationCalculator.f_factor(
                        ws_i.concentration_factor, ws_j.concentration_factor
                    )
                total += 2.0 * rho * f * ws_i.weighted_value * ws_j.weighted_value

        k_b = math.sqrt(max(0.0, total))
        ws_sum = sum(ws.weighted_value for ws in weighted_sensitivities)

        return BucketResult(
            risk_class=risk_class,
            margin_type=margin_type,
            bucket=bucket,
            k_b=k_b,
            ws_sum=ws_sum,
            bucket_cr=bucket_cr,
            weighted_sensitivities=weighted_sensitivities,
            is_residual=residual,
        )
