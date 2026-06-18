"""Interest-rate risk engine for SA-CVA (MAR50.54-50.58).

Buckets = currencies; cross-bucket gamma=0.5. Specified currencies use 5 tenors +
inflation (Table 4 correlations); other currencies use a parallel-yield factor +
inflation (rho=0.40). Vega uses rate-vol + inflation-vol factors (rho=0.40, RW=1.0).
"""

from typing import Dict, List

from quantark.sacva.engines.base import BaseRiskClassEngine
from quantark.sacva.engines.correlations import ir_specified_rho
from quantark.sacva.models.enums import RiskType
from quantark.sacva.models.sensitivity import CVASensitivity
from quantark.sacva.parameters.supervisory import SupervisoryParameters as SP


class InterestRateEngine(BaseRiskClassEngine):
    """Interest-rate delta and vega risk (MAR50.54-50.58)."""

    def __init__(self):
        self._specified_cache: Dict[str, bool] = {}

    def supports(self, risk_type: RiskType) -> bool:
        return True

    def is_single_factor(self) -> bool:
        return False

    def factor_identity(self, s: CVASensitivity):
        # One regulatory factor per: (vega) rate-vol & inflation-vol;
        # (specified delta) each tenor + inflation; (other delta) one
        # parallel-yield factor + one inflation factor. Splitting a single
        # regulatory factor would wrongly inflate the per-factor hedge
        # disallowance R*sum(WS_hdg^2). _specified_cache is populated by
        # group_buckets() before netting runs.
        if s.risk_type == RiskType.VEGA:
            return (s.currency, "infl_vol" if s.is_inflation else "rate_vol")
        if s.is_inflation:
            return (s.currency, "inflation")
        if self._specified_cache.get(s.currency, False):
            return (s.currency, s.tenor)   # one factor per tenor
        return (s.currency, "yield")        # other ccy: one parallel-yield factor

    def _is_specified(self, currency: str, reporting_currency: str) -> bool:
        return currency == reporting_currency or currency in SP.SPECIFIED_IR_BASE

    def risk_weight(self, s: CVASensitivity, risk_type: RiskType) -> float:
        if risk_type == RiskType.VEGA:
            return SP.IR_VEGA_RW
        if self._specified_cache.get(s.currency, False):
            return SP.ir_specified_rw(tenor=s.tenor, inflation=s.is_inflation)
        return SP.IR_OTHER_RW

    def group_buckets(self, sens: List[CVASensitivity], reporting_currency: str) -> Dict:
        out: Dict[str, List[CVASensitivity]] = {}
        for s in sens:
            self._specified_cache[s.currency] = self._is_specified(
                s.currency, reporting_currency)
            out.setdefault(s.currency, []).append(s)
        return out

    def intra_correlation(self, a: CVASensitivity, b: CVASensitivity) -> float:
        if a.risk_type == RiskType.VEGA:
            return 1.0 if a.is_inflation == b.is_inflation else SP.IR_VEGA_CORR
        if self._specified_cache.get(a.currency, False):
            return ir_specified_rho(a, b)
        # other currency: parallel-yield vs inflation
        return 1.0 if a.is_inflation == b.is_inflation else SP.IR_OTHER_CORR

    def cross_bucket_correlation(self, b, c) -> float:
        return SP.GAMMA_IR
