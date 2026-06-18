"""SA-CVA capital calculator (MAR50.42-50.53).

Aggregates supplied CVA + hedge sensitivities into delta + vega capital. The
module consumes sensitivities; it does not compute CVA itself (MAR50.29).
See ``quantark/sacva/doc/sacva_basel.md`` and the design spec.
"""

from typing import Dict, List

from quantark.sacva.engines.commodity import CommodityEngine
from quantark.sacva.engines.counterparty_credit import CounterpartyCreditEngine
from quantark.sacva.engines.equity import EquityEngine
from quantark.sacva.engines.fx import FXEngine
from quantark.sacva.engines.interest_rate import InterestRateEngine
from quantark.sacva.engines.reference_credit import ReferenceCreditEngine
from quantark.sacva.models.enums import RiskClass, RiskType
from quantark.sacva.models.portfolio import CVAPortfolio
from quantark.sacva.models.sensitivity import CVASensitivity
from quantark.sacva.parameters.supervisory import SACVA_VERSION, SupervisoryParameters
from quantark.sacva.results.result import SACVAResult
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_valid_number


class SACVACalculator:
    """Standardised approach for CVA risk capital."""

    def __init__(self, m_cva: float = SupervisoryParameters.M_CVA_DEFAULT):
        if (isinstance(m_cva, bool) or not isinstance(m_cva, (int, float))
                or not is_valid_number(m_cva)):
            raise ValidationError(f"m_cva must be a finite number, got {m_cva!r}")
        if m_cva < 1.0:
            raise ValidationError(f"m_cva must be >= 1.0 (MAR50.41), got {m_cva}")
        self.m_cva = float(m_cva)
        self.engines = {
            RiskClass.INTEREST_RATE: InterestRateEngine(),
            RiskClass.FX: FXEngine(),
            RiskClass.COUNTERPARTY_CREDIT: CounterpartyCreditEngine(),
            RiskClass.REFERENCE_CREDIT: ReferenceCreditEngine(),
            RiskClass.EQUITY: EquityEngine(),
            RiskClass.COMMODITY: CommodityEngine(),
        }

    def calculate(self, portfolio: CVAPortfolio) -> SACVAResult:
        by_class: Dict[RiskClass, List[CVASensitivity]] = {}
        for s in portfolio.sensitivities:
            by_class.setdefault(s.risk_class, []).append(s)

        delta_total = 0.0
        vega_total = 0.0
        result = SACVAResult(total_capital=0.0, delta_capital=0.0, vega_capital=0.0,
                             m_cva=self.m_cva, version=SACVA_VERSION)

        for risk_class, sens in by_class.items():
            engine = self.engines[risk_class]
            for risk_type in (RiskType.DELTA, RiskType.VEGA):
                if not engine.supports(risk_type):
                    continue
                rc_res = engine.calculate(sens, risk_type, portfolio.reporting_currency)
                if not rc_res.by_bucket:
                    continue
                k = self.m_cva * rc_res.k
                label = f"{risk_class.name}:{risk_type.name}"
                result.by_risk_class[label] = k  # m_cva-scaled per-class capital
                # Decomposition building blocks are stored RAW (unscaled).
                for bkey, kb in rc_res.by_bucket.items():
                    bl = f"{label}:{bkey}"
                    result.by_bucket[bl] = kb
                    result.bucket_s_b[bl] = rc_res.bucket_s_b[bkey]
                    result.bucket_sum_ws[bl] = rc_res.bucket_sum_ws[bkey]
                    result.hedge_disallowance[bl] = rc_res.hedge_disallowance[bkey]
                if risk_type == RiskType.DELTA:
                    delta_total += k
                else:
                    vega_total += k

        result.delta_capital = delta_total
        result.vega_capital = vega_total
        result.total_capital = delta_total + vega_total
        return result
