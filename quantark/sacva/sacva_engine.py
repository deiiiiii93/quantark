"""SA-CVA end-to-end façade: real portfolio -> regulatory capital (spec §3.5).

Wires the pieces "like SIMM does": for each counterparty, run the risk-neutral MC
exposure engine to an EE profile, derive bump-and-revalue SA-CVA sensitivities, then
feed the existing sensitivity-based-aggregation calculator (MAR50.42-50.53) to get
capital. The aggregation engine itself is unchanged — this façade only produces its
``CVASensitivity`` inputs from priced trades.

Sensitivities emitted (v1):
- counterparty credit-spread delta (per entity x tenor, MAR50.63) — exposure is
  invariant to the counterparty hazard, so only the CVA integral re-runs;
- equity delta + vega (single factor per bucket, MAR50.70) — these MOVE the
  exposure, so each is a portfolio-wide bump (re-run MC for every affected
  counterparty with common random numbers) summed across counterparties.

Stateful (snowball/phoenix) grid exposure and IR/FX market sensitivities raise in
the underlying engines rather than being silently dropped.
"""

from dataclasses import replace

from quantark.sacva.calculator import SACVACalculator
from quantark.sacva.exposure.simulator import MonteCarloExposureEngine
from quantark.sacva.models.enums import RiskClass, RiskType
from quantark.sacva.models.portfolio import CVAPortfolio
from quantark.sacva.models.sensitivity import CVASensitivity
from quantark.sacva.sensitivities.engine import CVASensitivityEngine
from quantark.sacva.sensitivities.shifts import (
    EQUITY_SPOT_SHIFT,
    EQUITY_VOL_SHIFT,
    bump_spot_env,
    bump_vol_env,
)
from quantark.util.exceptions import ValidationError

_EQUITY_INDEX_BUCKETS = {12, 13}


class SACVAEngine:
    def __init__(self, exposure_engine=None, sensitivity_engine=None, calculator=None):
        self.exposure_engine = exposure_engine or MonteCarloExposureEngine()
        self.sensitivity_engine = sensitivity_engine or CVASensitivityEngine()
        self.calculator = calculator or SACVACalculator()

    # -- CVA helpers ------------------------------------------------------------
    def _counterparty_cva(self, cp):
        profile = self.exposure_engine.compute(cp)
        cva = self.sensitivity_engine.cva_engine.compute(cp.credit_curve, profile)
        return profile, cva

    @staticmethod
    def _bump_counterparty(cp, bucket, kind):
        """Clone ``cp`` with bucket-``bucket`` trade envs bumped (kind: spot/vol)."""
        bump = bump_spot_env if kind == "spot" else bump_vol_env
        factor = 1.0 + (EQUITY_SPOT_SHIFT if kind == "spot" else EQUITY_VOL_SHIFT)
        new_sets = []
        for ns in cp.netting_sets:
            trades = [replace(t, env=bump(t.env, factor)) if t.equity_bucket == bucket
                      else t for t in ns.trades]
            new_sets.append(replace(ns, trades=trades))
        return replace(cp, netting_sets=new_sets)

    # -- market-sensitivity scope ----------------------------------------------
    @staticmethod
    def _equity_buckets(portfolio):
        trades = [t for cp in portfolio.counterparties
                  for ns in cp.netting_sets for t in ns.trades]
        tagged = [t for t in trades if t.equity_bucket is not None]
        if not tagged:
            return []                          # credit-only run
        if len(tagged) != len(trades):
            raise ValidationError(
                "market sensitivities require every trade to declare an equity_bucket "
                "(all-or-none); mixed tagging is ambiguous")
        return sorted({t.equity_bucket for t in tagged})

    def _equity_sensitivities(self, portfolio, base_cva):
        sens = []
        buckets = self._equity_buckets(portfolio)
        for b in buckets:
            affected = [cp for cp in portfolio.counterparties
                        if any(t.equity_bucket == b
                               for ns in cp.netting_sets for t in ns.trades)]
            for kind, risk_type, shift in (
                ("spot", RiskType.DELTA, EQUITY_SPOT_SHIFT),
                ("vol", RiskType.VEGA, EQUITY_VOL_SHIFT),
            ):
                d_cva = 0.0
                for cp in affected:
                    _, cva_up = self._counterparty_cva(
                        self._bump_counterparty(cp, b, kind))
                    d_cva += cva_up - base_cva[cp.name]
                sens.append(CVASensitivity(
                    risk_class=RiskClass.EQUITY, risk_type=risk_type, bucket=b,
                    risk_factor=f"EQ:{b}:{kind}", s_cva=d_cva / shift,
                    is_index=b in _EQUITY_INDEX_BUCKETS))
        return sens

    # -- entry point ------------------------------------------------------------
    def compute(self, portfolio):
        """Compute SA-CVA capital from a ``CVATradePortfolio``.

        Returns the SBA ``SACVAResult``; per-counterparty EE profiles and base CVA
        are attached for inspection/audit.
        """
        sensitivities = []
        profiles = {}
        base_cva = {}
        for cp in portfolio.counterparties:
            profile, cva = self._counterparty_cva(cp)
            profiles[cp.name] = profile
            base_cva[cp.name] = cva
            sensitivities.extend(
                self.sensitivity_engine.counterparty_spread_deltas(cp, profile))

        sensitivities.extend(self._equity_sensitivities(portfolio, base_cva))

        if not sensitivities:
            raise ValidationError("no SA-CVA sensitivities produced from portfolio")

        cva_portfolio = CVAPortfolio(sensitivities=sensitivities,
                                     reporting_currency=portfolio.reporting_currency)
        result = self.calculator.calculate(cva_portfolio)
        result.exposure_profiles = profiles
        result.counterparty_cva = base_cva
        return result
