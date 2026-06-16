"""FX risk engine for SA-CVA (MAR50.59-50.62). Single factor per currency bucket."""

from typing import Dict, List

from quantark.sacva.engines.base import BaseRiskClassEngine
from quantark.sacva.models.enums import RiskType
from quantark.sacva.models.sensitivity import CVASensitivity
from quantark.sacva.parameters.supervisory import SupervisoryParameters as SP


class FXEngine(BaseRiskClassEngine):
    """FX delta and vega risk (MAR50.59-50.62)."""

    def supports(self, risk_type: RiskType) -> bool:
        return True

    def is_single_factor(self) -> bool:
        return True

    def factor_identity(self, s: CVASensitivity):
        return s.currency

    def risk_weight(self, s: CVASensitivity, risk_type: RiskType) -> float:
        return SP.FX_VEGA_RW if risk_type == RiskType.VEGA else SP.FX_DELTA_RW

    def group_buckets(self, sens: List[CVASensitivity], reporting_currency: str) -> Dict:
        out: Dict[str, List[CVASensitivity]] = {}
        for s in sens:
            out.setdefault(s.currency, []).append(s)
        return out

    def intra_correlation(self, a, b) -> float:
        return 0.0

    def cross_bucket_correlation(self, b, c) -> float:
        return SP.GAMMA_FX
