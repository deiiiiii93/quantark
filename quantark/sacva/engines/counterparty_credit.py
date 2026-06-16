"""Counterparty credit spread risk engine for SA-CVA (MAR50.63-50.65).

Delta only (no vega). Multi-factor buckets keyed by (name, tenor[, index_series]).
"""

from typing import Dict, List

from quantark.sacva.engines.base import BaseRiskClassEngine
from quantark.sacva.engines.correlations import counterparty_rho
from quantark.sacva.models.enums import RiskType
from quantark.sacva.models.sensitivity import CVASensitivity
from quantark.sacva.parameters.supervisory import SupervisoryParameters as SP


class CounterpartyCreditEngine(BaseRiskClassEngine):
    """Counterparty credit spread delta risk (MAR50.63-50.65)."""

    def supports(self, risk_type: RiskType) -> bool:
        return risk_type == RiskType.DELTA

    def is_single_factor(self) -> bool:
        return False

    def factor_identity(self, s: CVASensitivity):
        return (s.bucket, s.name, s.tenor, s.index_series)

    def risk_weight(self, s: CVASensitivity, risk_type: RiskType) -> float:
        return SP.cpty_rw(s.bucket, s.credit_quality, s.sub_bucket)

    def group_buckets(self, sens: List[CVASensitivity], reporting_currency: str) -> Dict:
        out: Dict[int, List[CVASensitivity]] = {}
        for s in sens:
            out.setdefault(s.bucket, []).append(s)
        return out

    def intra_correlation(self, a: CVASensitivity, b: CVASensitivity) -> float:
        return counterparty_rho(a, b)

    def cross_bucket_correlation(self, b, c) -> float:
        return SP.cpty_gamma(b, c)
