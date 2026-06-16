"""Equity risk engine for SA-CVA (MAR50.70-50.73). Single factor per bucket."""

from typing import Dict, List

from quantark.sacva.engines.base import BaseRiskClassEngine
from quantark.sacva.models.enums import RiskType
from quantark.sacva.models.sensitivity import CVASensitivity
from quantark.sacva.parameters.supervisory import SupervisoryParameters as SP


class EquityEngine(BaseRiskClassEngine):
    """Equity delta and vega risk (MAR50.70-50.73)."""

    def supports(self, risk_type: RiskType) -> bool:
        return True

    def is_single_factor(self) -> bool:
        return True

    def factor_identity(self, s: CVASensitivity):
        return s.bucket

    def risk_weight(self, s: CVASensitivity, risk_type: RiskType) -> float:
        return SP.equity_vega_rw(s.bucket) if risk_type == RiskType.VEGA \
            else SP.equity_delta_rw(s.bucket)

    def group_buckets(self, sens: List[CVASensitivity], reporting_currency: str) -> Dict:
        out: Dict[int, List[CVASensitivity]] = {}
        for s in sens:
            out.setdefault(s.bucket, []).append(s)
        return out

    def intra_correlation(self, a, b) -> float:
        return 0.0

    def cross_bucket_correlation(self, b, c) -> float:
        return SP.equity_gamma(b, c)
