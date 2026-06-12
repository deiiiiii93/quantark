"""
Risk-class level aggregation for SIMM.

Implements:
- Cross-bucket delta/vega aggregation (paragraphs 7(d), 8(d), 10(f)):

      Margin = sqrt( sum_b K_b^2
                     + sum_b sum_{c != b} gamma_bc [g_bc] S_b S_c )
               + K_residual

  with S_b = max(min(sum_k WS_k, K_b), -K_b). The g_bc factor applies in
  the Interest Rate risk class only.

- Curvature margin (paragraph 11) with the theta / lambda factors,
  squared correlations, separate residual treatment and the HVR_IR^-2
  scale factor for the interest-rate risk class.

- Base Correlation margin (paragraph 13).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Union

from quantark.simm.taxonomy import MarginType, RiskClass, is_residual_bucket
from quantark.simm.engines.aggregation.bucket_aggregator import BucketResult
from quantark.simm.engines.aggregation.concentration import ConcentrationCalculator
from quantark.simm.engines.aggregation.correlations import (
    inter_bucket_correlation,
    intra_bucket_correlation,
)
from quantark.simm.calibration.ir import IR_HVR
from quantark.simm.calibration.accessors import PHI_INV_995
from quantark.simm.calibration.credit_qualifying import (
    CREDIT_QUALIFYING_BASE_CORRELATION_INTER_INDEX_CORRELATION,
)


@dataclass
class RiskClassResult:
    """Result of risk class aggregation for one margin type.

    Attributes:
        risk_class: The risk class.
        margin_type: Delta, Vega, Curvature, or BaseCorr.
        margin: The aggregated margin.
        bucket_results: Bucket-level results (delta/vega only).
        residual_margin: The residual bucket contribution.
    """
    risk_class: RiskClass
    margin_type: MarginType
    margin: float
    bucket_results: Dict[Union[str, int], BucketResult] = field(default_factory=dict)
    residual_margin: float = 0.0


@dataclass
class CurvatureExposure:
    """Net curvature exposure CVR_k for one risk factor (paragraph 11(a)-(b)).

    Attributes:
        bucket: The bucket identifier.
        risk_factor: Risk factor key (vega-style).
        amount: Net CVR (scaling function already applied).
        group: Group name for Credit Non-Qualifying correlations.
    """
    bucket: Union[str, int]
    risk_factor: Hashable
    amount: float
    group: str = ""


class RiskClassAggregator:
    """Aggregator across buckets within a risk class."""

    def __init__(self, calculation_currency: str = "USD"):
        self.calculation_currency = calculation_currency.upper()

    # ------------------------------------------------------------------
    # Delta / Vega
    # ------------------------------------------------------------------

    def aggregate_delta_vega(
        self,
        bucket_results: List[BucketResult],
        risk_class: RiskClass,
        margin_type: MarginType,
    ) -> RiskClassResult:
        """Aggregate delta or vega margin across buckets.

        Args:
            bucket_results: One BucketResult per bucket (residual included).
            risk_class: The risk class.
            margin_type: DELTA or VEGA.

        Returns:
            RiskClassResult with the aggregated margin.
        """
        if not bucket_results:
            return RiskClassResult(risk_class=risk_class, margin_type=margin_type, margin=0.0)

        residual = [b for b in bucket_results if b.is_residual]
        non_residual = [b for b in bucket_results if not b.is_residual]

        # K_residual is added without diversification (paragraph 8(d)).
        residual_margin = sum(b.k_b for b in residual)

        total = 0.0
        n = len(non_residual)
        for i in range(n):
            b_i = non_residual[i]
            total += b_i.k_b ** 2
            for j in range(i + 1, n):
                b_j = non_residual[j]
                gamma = inter_bucket_correlation(risk_class, b_i.bucket, b_j.bucket)
                # g_bc applies in the Interest Rate risk class only
                # (paragraphs 7(d) and 10(f)).
                if risk_class == RiskClass.INTEREST_RATE:
                    gamma *= ConcentrationCalculator.g_factor(b_i.bucket_cr, b_j.bucket_cr)
                total += 2.0 * gamma * b_i.s_b * b_j.s_b

        margin = math.sqrt(max(0.0, total)) + residual_margin

        return RiskClassResult(
            risk_class=risk_class,
            margin_type=margin_type,
            margin=margin,
            bucket_results={b.bucket: b for b in bucket_results},
            residual_margin=residual_margin,
        )

    # ------------------------------------------------------------------
    # Curvature
    # ------------------------------------------------------------------

    def aggregate_curvature(
        self,
        exposures: List[CurvatureExposure],
        risk_class: RiskClass,
    ) -> RiskClassResult:
        """Aggregate curvature margin for a risk class (paragraph 11).

        Args:
            exposures: Net CVR exposures per (bucket, risk factor).
            risk_class: The risk class.

        Returns:
            RiskClassResult with the curvature margin.
        """
        if not exposures:
            return RiskClassResult(
                risk_class=risk_class, margin_type=MarginType.CURVATURE, margin=0.0
            )

        non_residual = [e for e in exposures if not is_residual_bucket(e.bucket)]
        residual = [e for e in exposures if is_residual_bucket(e.bucket)]

        margin_non_res = self._curvature_non_residual(non_residual, risk_class)
        margin_res = self._curvature_residual(residual, risk_class)

        margin = margin_non_res + margin_res

        # For the interest-rate risk class only, the CurvatureMargin is
        # multiplied by HVR_IR^-2 (paragraph 11(d)).
        if risk_class == RiskClass.INTEREST_RATE:
            margin *= IR_HVR ** (-2)

        return RiskClassResult(
            risk_class=risk_class,
            margin_type=MarginType.CURVATURE,
            margin=margin,
            residual_margin=margin_res,
        )

    def _curvature_bucket_k(
        self,
        exposures: List[CurvatureExposure],
        risk_class: RiskClass,
        bucket: Union[str, int],
    ) -> float:
        """K_b = sqrt(sum CVR^2 + sum_{k != l} rho_kl^2 CVR_k CVR_l)
        (paragraph 11(c))."""
        n = len(exposures)
        total = 0.0
        for i in range(n):
            e_i = exposures[i]
            total += e_i.amount ** 2
            for j in range(i + 1, n):
                e_j = exposures[j]
                rho = intra_bucket_correlation(
                    risk_class,
                    MarginType.CURVATURE,
                    bucket,
                    e_i.risk_factor,
                    e_j.risk_factor,
                    group_1=e_i.group,
                    group_2=e_j.group,
                    calculation_currency=self.calculation_currency,
                )
                total += 2.0 * (rho ** 2) * e_i.amount * e_j.amount
        return math.sqrt(max(0.0, total))

    @staticmethod
    def _lambda(theta: float) -> float:
        """lambda = (PHI_INV(99.5%)^2 - 1)(1 + theta) - theta (paragraph 11(d))."""
        return (PHI_INV_995 ** 2 - 1.0) * (1.0 + theta) - theta

    def _curvature_non_residual(
        self,
        exposures: List[CurvatureExposure],
        risk_class: RiskClass,
    ) -> float:
        """Non-residual curvature margin (paragraph 11(d))."""
        if not exposures:
            return 0.0

        by_bucket: Dict[Union[str, int], List[CurvatureExposure]] = {}
        for e in exposures:
            by_bucket.setdefault(e.bucket, []).append(e)

        bucket_k = {
            bucket: self._curvature_bucket_k(items, risk_class, bucket)
            for bucket, items in by_bucket.items()
        }
        bucket_sum = {
            bucket: sum(e.amount for e in items)
            for bucket, items in by_bucket.items()
        }
        s_values = {
            bucket: max(min(bucket_sum[bucket], bucket_k[bucket]), -bucket_k[bucket])
            for bucket in by_bucket
        }

        total_cvr = sum(e.amount for e in exposures)
        total_abs_cvr = sum(abs(e.amount) for e in exposures)

        theta = min(total_cvr / total_abs_cvr, 0.0) if total_abs_cvr > 0 else 0.0
        lam = self._lambda(theta)

        buckets = list(by_bucket.keys())
        n = len(buckets)
        total = 0.0
        for i in range(n):
            total += bucket_k[buckets[i]] ** 2
            for j in range(i + 1, n):
                gamma = inter_bucket_correlation(risk_class, buckets[i], buckets[j])
                total += 2.0 * (gamma ** 2) * s_values[buckets[i]] * s_values[buckets[j]]

        return max(total_cvr + lam * math.sqrt(max(0.0, total)), 0.0)

    def _curvature_residual(
        self,
        exposures: List[CurvatureExposure],
        risk_class: RiskClass,
    ) -> float:
        """Residual curvature margin (paragraph 11(d))."""
        if not exposures:
            return 0.0

        total_cvr = sum(e.amount for e in exposures)
        total_abs_cvr = sum(abs(e.amount) for e in exposures)

        theta = min(total_cvr / total_abs_cvr, 0.0) if total_abs_cvr > 0 else 0.0
        lam = self._lambda(theta)

        # All residual exposures share the residual bucket.
        bucket = exposures[0].bucket
        k_res = self._curvature_bucket_k(exposures, risk_class, bucket)

        return max(total_cvr + lam * k_res, 0.0)

    # ------------------------------------------------------------------
    # Base Correlation
    # ------------------------------------------------------------------

    def aggregate_base_corr(
        self,
        weighted_by_index: Dict[str, float],
    ) -> RiskClassResult:
        """Base Correlation margin (paragraph 13, Credit Qualifying only).

        BaseCorrMargin = sqrt(sum WS_k^2 + sum_{k != l} rho_kl WS_k WS_l)
        with rho_kl = 29% across different index families.

        Args:
            weighted_by_index: WS_k = RW * s_k per index family.

        Returns:
            RiskClassResult with the base correlation margin.
        """
        if not weighted_by_index:
            return RiskClassResult(
                risk_class=RiskClass.CREDIT_QUALIFYING,
                margin_type=MarginType.BASE_CORR,
                margin=0.0,
            )

        rho = CREDIT_QUALIFYING_BASE_CORRELATION_INTER_INDEX_CORRELATION
        values = list(weighted_by_index.values())
        n = len(values)
        total = 0.0
        for i in range(n):
            total += values[i] ** 2
            for j in range(i + 1, n):
                total += 2.0 * rho * values[i] * values[j]

        return RiskClassResult(
            risk_class=RiskClass.CREDIT_QUALIFYING,
            margin_type=MarginType.BASE_CORR,
            margin=math.sqrt(max(0.0, total)),
        )
