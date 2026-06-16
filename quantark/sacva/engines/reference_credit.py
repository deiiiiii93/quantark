"""Reference credit spread engine for SA-CVA (MAR50.66-50.69).

Single factor per bucket (all 17 buckets). Delta + vega.
"""

from typing import Dict, List

from quantark.sacva.engines.base import BaseRiskClassEngine
from quantark.sacva.models.enums import RiskType
from quantark.sacva.models.sensitivity import CVASensitivity
from quantark.sacva.parameters.supervisory import SupervisoryParameters as SP


class ReferenceCreditEngine(BaseRiskClassEngine):
    """Reference credit spread delta and vega risk (MAR50.66-50.69)."""

    def supports(self, risk_type: RiskType) -> bool:
        return True

    def is_single_factor(self) -> bool:
        return True

    def factor_identity(self, s: CVASensitivity):
        return s.bucket

    def risk_weight(self, s: CVASensitivity, risk_type: RiskType) -> float:
        return 1.00 if risk_type == RiskType.VEGA else SP.refcredit_rw(s.bucket)

    def group_buckets(self, sens: List[CVASensitivity], reporting_currency: str) -> Dict:
        out: Dict[int, List[CVASensitivity]] = {}
        for s in sens:
            out.setdefault(s.bucket, []).append(s)
        return out

    def intra_correlation(self, a, b) -> float:
        return 0.0

    def cross_bucket_correlation(self, b, c) -> float:
        return SP.refcredit_gamma(b, c)
