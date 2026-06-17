"""SA-CVA end-to-end façade: real portfolio -> regulatory capital (spec §3.5).

Wires the pieces "like SIMM does": for each counterparty, run the risk-neutral MC
exposure engine to an EE profile, derive bump-and-revalue SA-CVA sensitivities, then
feed the existing sensitivity-based-aggregation calculator (MAR50.42-50.53) to get
capital. The aggregation engine itself is unchanged — this façade only produces its
``CVASensitivity`` inputs from priced trades.

v1 emits counterparty credit-spread delta (the irreducible CVA risk class). Equity/IR
market delta+vega and stateful (snowball) exposure are scoped extensions; unsupported
inputs raise in the underlying engines rather than being silently dropped.
"""

from quantark.sacva.calculator import SACVACalculator
from quantark.sacva.exposure.simulator import MonteCarloExposureEngine
from quantark.sacva.models.portfolio import CVAPortfolio
from quantark.sacva.sensitivities.engine import CVASensitivityEngine
from quantark.util.exceptions import ValidationError


class SACVAEngine:
    def __init__(self, exposure_engine=None, sensitivity_engine=None, calculator=None):
        self.exposure_engine = exposure_engine or MonteCarloExposureEngine()
        self.sensitivity_engine = sensitivity_engine or CVASensitivityEngine()
        self.calculator = calculator or SACVACalculator()

    def compute(self, portfolio):
        """Compute SA-CVA capital from a ``CVATradePortfolio``.

        Returns the ``SACVAResult`` from the SBA calculator. ``profiles`` and
        per-counterparty base CVA are attached for inspection/audit.
        """
        sensitivities = []
        profiles = {}
        base_cva = {}
        for cp in portfolio.counterparties:
            profile = self.exposure_engine.compute(cp)
            profiles[cp.name] = profile
            base_cva[cp.name] = self.sensitivity_engine.cva_engine.compute(
                cp.credit_curve, profile)
            sensitivities.extend(
                self.sensitivity_engine.counterparty_spread_deltas(cp, profile))
        if not sensitivities:
            raise ValidationError("no SA-CVA sensitivities produced from portfolio")

        cva_portfolio = CVAPortfolio(sensitivities=sensitivities,
                                     reporting_currency=portfolio.reporting_currency)
        result = self.calculator.calculate(cva_portfolio)
        # audit attachments (non-regulatory, for inspection)
        result.exposure_profiles = profiles
        result.counterparty_cva = base_cva
        return result
