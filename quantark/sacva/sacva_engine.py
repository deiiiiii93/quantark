"""SA-CVA end-to-end façade: real portfolio -> regulatory capital (spec §3.5).

Wires the pieces "like SIMM does": for each counterparty, run the risk-neutral MC
exposure engine to an EE profile, derive bump-and-revalue SA-CVA sensitivities, then
feed the existing sensitivity-based-aggregation calculator (MAR50.42-50.53) to get
capital. The aggregation engine itself is unchanged — this façade only produces its
``CVASensitivity`` inputs from priced trades.

Sensitivities emitted (v1):
- counterparty credit-spread delta (per entity x tenor, MAR50.63) — exposure is
  invariant to the counterparty hazard, so only the CVA integral re-runs;
- equity / FX spot delta + vega (single factor per bucket / currency, MAR50.59,
  MAR50.70) — these MOVE the exposure, so each is a portfolio-wide bump (re-run MC
  for every affected counterparty with common random numbers) summed across
  counterparties;
- eligible-hedge market-value sensitivities (S_k^Hdg, MAR50.29) — each hedge's MV
  is bumped on the SAME factor and emitted with s_cva=0 so the SBA risk-factor
  netting forms WS = RW*(s_cva - s_hdg).

IR market sensitivities need a key-rate-bumpable term curve (the v1 flat-rate envs
cannot produce regulatory per-tenor factors) and stateful Phoenix/KO-reset exposure
raise in the underlying engines rather than being silently approximated.
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
    bump_spot_env,
    bump_vol_env,
)
from quantark.util.exceptions import ValidationError

_EQUITY_INDEX_BUCKETS = {12, 13}
_MARKET_SHIFT = EQUITY_SPOT_SHIFT  # 1% relative for spot & vol (MAR50.63), divisor 1e-2


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

    # -- market factors (equity bucket / FX currency) --------------------------
    @staticmethod
    def _trade_matches(trade, risk_class, key):
        if risk_class is RiskClass.EQUITY:
            return trade.equity_bucket == key
        return trade.fx_currency == key

    @staticmethod
    def _trade_factor(trade):
        """The (risk_class, key) market factor a trade/hedge declares, or None."""
        if trade.equity_bucket is not None:
            return (RiskClass.EQUITY, trade.equity_bucket)
        if trade.fx_currency is not None:
            return (RiskClass.FX, trade.fx_currency)
        return None

    @classmethod
    def _market_factors(cls, portfolio):
        """Distinct (risk_class, key) factors; raise on partial/invalid tagging."""
        trades = [t for cp in portfolio.counterparties
                  for ns in cp.netting_sets for t in ns.trades]
        tagged = [t for t in trades if cls._trade_factor(t) is not None]
        if not tagged:
            return []                          # credit-only run
        if len(tagged) != len(trades):
            raise ValidationError(
                "market sensitivities require every trade to declare a market factor "
                "(equity_bucket or fx_currency); mixed tagging is ambiguous")
        reporting = portfolio.reporting_currency.upper()
        factors = {}
        for t in tagged:
            rc, key = cls._trade_factor(t)
            if rc is RiskClass.FX and key == reporting:
                raise ValidationError(
                    f"{t.trade_id}: fx_currency equals the reporting currency "
                    f"{reporting}; there is no FX risk to the reporting currency")
            factors[(rc, key)] = None
        return sorted(factors, key=lambda f: (f[0].name, str(f[1])))

    def _bump_counterparty(self, cp, risk_class, key, kind):
        """Clone ``cp`` with the factor's matching trade envs bumped (kind: spot/vol)."""
        bump = bump_spot_env if kind == "spot" else bump_vol_env
        factor = 1.0 + _MARKET_SHIFT
        new_sets = []
        for ns in cp.netting_sets:
            trades = [replace(t, env=bump(t.env, factor))
                      if self._trade_matches(t, risk_class, key) else t
                      for t in ns.trades]
            new_sets.append(replace(ns, trades=trades))
        return replace(cp, netting_sets=new_sets)

    @staticmethod
    def _market_record(risk_class, key, kind, risk_type, s_cva=0.0, s_hdg=0.0):
        if risk_class is RiskClass.EQUITY:
            return CVASensitivity(
                risk_class=risk_class, risk_type=risk_type, bucket=key,
                risk_factor=f"EQ:{key}:{kind}", s_cva=s_cva, s_hdg=s_hdg,
                is_index=key in _EQUITY_INDEX_BUCKETS)
        return CVASensitivity(
            risk_class=risk_class, risk_type=risk_type, bucket=0, currency=key,
            risk_factor=f"FX:{key}:{kind}", s_cva=s_cva, s_hdg=s_hdg)

    def _market_sensitivities(self, portfolio, base_cva):
        sens = []
        for risk_class, key in self._market_factors(portfolio):
            affected = [cp for cp in portfolio.counterparties
                        if any(self._trade_matches(t, risk_class, key)
                               for ns in cp.netting_sets for t in ns.trades)]
            for kind, risk_type in (("spot", RiskType.DELTA), ("vol", RiskType.VEGA)):
                d_cva = 0.0
                for cp in affected:
                    _, cva_up = self._counterparty_cva(
                        self._bump_counterparty(cp, risk_class, key, kind))
                    d_cva += cva_up - base_cva[cp.name]
                sens.append(self._market_record(
                    risk_class, key, kind, risk_type, s_cva=d_cva / _MARKET_SHIFT))
        return sens

    def _hedge_sensitivities(self, portfolio):
        """S_k^Hdg from each eligible hedge's own market-value bump (MAR50.29)."""
        sens = []
        reporting = portfolio.reporting_currency.upper()
        for h in portfolio.hedges:
            factor = self._trade_factor(h)
            if factor is None:
                raise ValidationError(
                    f"{h.trade_id}: hedge must declare a market factor (equity_bucket "
                    "or fx_currency) to attribute its market-value sensitivity")
            risk_class, key = factor
            if risk_class is RiskClass.FX and key == reporting:
                raise ValidationError(
                    f"{h.trade_id}: fx_currency equals the reporting currency {reporting}")
            base_mv = float(h.engine.price(h.product, h.env)) * float(h.quantity)
            for kind, risk_type, bump in (
                ("spot", RiskType.DELTA, bump_spot_env),
                ("vol", RiskType.VEGA, bump_vol_env),
            ):
                up = float(h.engine.price(
                    h.product, bump(h.env, 1.0 + _MARKET_SHIFT))) * float(h.quantity)
                s_hdg = (up - base_mv) / _MARKET_SHIFT
                sens.append(self._market_record(
                    risk_class, key, kind, risk_type, s_hdg=s_hdg))
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

        sensitivities.extend(self._market_sensitivities(portfolio, base_cva))
        sensitivities.extend(self._hedge_sensitivities(portfolio))

        if not sensitivities:
            raise ValidationError("no SA-CVA sensitivities produced from portfolio")

        cva_portfolio = CVAPortfolio(sensitivities=sensitivities,
                                     reporting_currency=portfolio.reporting_currency)
        result = self.calculator.calculate(cva_portfolio)
        result.exposure_profiles = profiles
        result.counterparty_cva = base_cva
        return result
