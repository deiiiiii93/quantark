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

- interest-rate delta (per currency x tenor, MAR50.54-50.58) — for a term-structure
  reporting curve, each SA-CVA vertex pillar is bumped 1bp and the CVA re-run (the curve
  moves drift AND discounting). Exact for vanilla analytic trades (term-structure forward
  drift + forward-curve roll-down); a flat curve yields no per-tenor factors, and stateful
  (snowball/Phoenix) or FX trades raise rather than being silently approximated.
"""

from dataclasses import replace

from quantark.sacva.calculator import SACVACalculator
from quantark.sacva.exposure.simulator import MonteCarloExposureEngine
from quantark.sacva.models.enums import RiskClass, RiskType
from quantark.sacva.models.portfolio import CVAPortfolio
from quantark.sacva.models.sensitivity import CVASensitivity
from quantark.sacva.sensitivities.engine import CVASensitivityEngine
from quantark.sacva.exposure.curves import IR_DELTA_TENORS, key_rate_bumped_curve
from quantark.sacva.sensitivities.shifts import (
    EQUITY_SPOT_SHIFT,
    bump_spot_env,
    bump_vol_env,
)
from quantark.param.rrf.rate_curve import InterpolatedRateCurve
from quantark.util.exceptions import ValidationError

_EQUITY_INDEX_BUCKETS = {12, 13}
# 1% relative shift for equity/FX spot & vega (MAR50.59 FX, MAR50.70 equity); divisor 1e-2
_MARKET_SHIFT = EQUITY_SPOT_SHIFT
# 1bp absolute key-rate shift for IR delta; sensitivity is dCVA/dr (divide by the shift)
_IR_SHIFT = 1e-4


class SACVAEngine:
    def __init__(self, exposure_engine=None, sensitivity_engine=None, calculator=None,
                 include_ir_delta=True):
        self.exposure_engine = exposure_engine or MonteCarloExposureEngine()
        self.sensitivity_engine = sensitivity_engine or CVASensitivityEngine()
        self.calculator = calculator or SACVACalculator()
        # IR delta is ON by default (a regulatory SA-CVA run with a vertex-pillared
        # reporting curve produces it; a flat curve then RAISES rather than silently
        # omitting IR risk). Set False to explicitly run WITHOUT IR delta (e.g. a
        # portfolio intentionally modelled on a flat discount curve).
        if not isinstance(include_ir_delta, bool):  # avoid accidental 0/None/"" toggling
            raise ValidationError("include_ir_delta must be a bool")
        self.include_ir_delta = include_ir_delta

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
        # a tagged HEDGE also requests market mode: otherwise an untagged trade's own
        # market CVA would be silently dropped while the hedge's s_hdg is capitalised.
        hedges_tagged = any(cls._trade_factor(h) is not None for h in portfolio.hedges)
        if not tagged and not hedges_tagged:
            return []                          # credit-only run
        if trades and len(tagged) != len(trades):
            raise ValidationError(
                "market sensitivities require every trade to declare a market factor "
                "(equity_bucket or fx_currency) when any trade or hedge does; mixed "
                "tagging is ambiguous")
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

    @staticmethod
    def _validate_fx_identity(portfolio):
        """FX is one factor per foreign currency, backed by exactly one GBM underlying.

        The SBA groups FX sensitivities by ``fx_currency`` while the exposure engine
        keys paths by ``env.spot_quote.asset_name``; enforce a 1:1 currency<->underlying
        map so a single regulatory FX factor cannot secretly span several simulated
        FX processes (equity buckets, by contrast, legitimately hold many names).
        """
        items = [t for cp in portfolio.counterparties
                 for ns in cp.netting_sets for t in ns.trades]
        items += list(portfolio.hedges)
        ccy_to_asset, asset_to_ccy = {}, {}
        for t in items:
            if t.fx_currency is None:
                continue
            sq = getattr(t.env, "spot_quote", None)
            asset = getattr(sq, "asset_name", None) if sq is not None else None
            if not asset:
                raise ValidationError(
                    f"{t.trade_id}: FX trade requires env.spot_quote.asset_name")
            if ccy_to_asset.setdefault(t.fx_currency, asset) != asset:
                raise ValidationError(
                    f"FX currency {t.fx_currency} maps to multiple underlyings "
                    f"({ccy_to_asset[t.fx_currency]}, {asset}); one currency = one factor")
            if asset_to_ccy.setdefault(asset, t.fx_currency) != t.fx_currency:
                raise ValidationError(
                    f"underlying {asset} maps to multiple FX currencies "
                    f"({asset_to_ccy[asset]}, {t.fx_currency})")

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

    # -- interest-rate delta (key-rate, MAR50.54-50.58) -------------------------
    def _ir_sensitivities(self, portfolio, base_cva):
        """Per-tenor IR delta on the single shared reporting-currency discount curve.

        For each SA-CVA vertex the curve pillar is shifted by 1bp and every counterparty's
        CVA re-run (the reporting curve moves both drift and discounting); the emitted
        s_cva is the regulatory 1bp finite-difference delta ``(CVA_up - CVA)/1bp`` (units:
        reporting currency per unit rate, matching the calculator's IR risk weights) — the
        standard bump-and-revalue sensitivity, not an analytic derivative.

        EXACT only for vanilla analytic trades. Preconditions, all enforced by raising
        rather than approximating: every eligible trade shares ONE term-structure
        (InterpolatedRateCurve) reporting curve with pillars at every SA-CVA vertex; no
        stateful (single-rate QUAD) or FX trade is present; and a flat reporting curve
        (no exact per-tenor IR degrees of freedom) RAISES rather than silently omitting IR
        risk. This method is only reached when IR delta is enabled (the default); callers
        that intentionally model a flat curve pass ``include_ir_delta=False`` to opt out.
        """
        trades = [t for cp in portfolio.counterparties
                  for ns in cp.netting_sets for t in ns.trades]
        if not trades:
            return []
        curves = [getattr(t.env, "rate_curve", None) for t in trades]
        term = [c for c in curves if isinstance(c, InterpolatedRateCurve)]
        if not term:
            # IR delta is enabled but no per-tenor curve is supplied. A flat curve has no
            # exact per-tenor IR degrees of freedom -> raise (do NOT silently omit IR risk;
            # CVA still depends on the level r). Opt out with include_ir_delta=False.
            raise ValidationError(
                "SA-CVA IR delta is enabled but the reporting discount curve is flat "
                "(no exact per-tenor IR degrees of freedom); supply a curve pillared "
                f"exactly at the SA-CVA vertices {list(IR_DELTA_TENORS)}, or construct "
                "SACVAEngine(include_ir_delta=False) to run without IR delta")
        if len(term) != len(trades):
            raise ValidationError(
                "IR delta requires ALL trades to share one term-structure reporting curve;"
                " some trades use a flat or absent rate curve (mixed curves are ambiguous)")
        identity = {(type(c).__name__,
                     tuple((float(t), float(r)) for t, r in c.pillars)) for c in term}
        if len(identity) != 1:
            raise ValidationError(
                "IR delta requires a single shared reporting curve; trades carry differing "
                "curve types/pillars")
        for t in trades:
            if getattr(t.engine, "supports_spot_greeks_grid", False):
                raise ValidationError(
                    f"{t.trade_id}: IR delta is deferred for stateful (snowball/Phoenix) "
                    "trades — their QUAD engines price under a single rate")
            if t.fx_currency is not None:
                raise ValidationError(
                    f"{t.trade_id}: IR delta is deferred for FX trades (the foreign curve "
                    "is a separate currency factor)")
        curve = term[0]
        reporting = portfolio.reporting_currency.upper()
        # The reporting curve must be parameterized EXACTLY by the SA-CVA vertices: every
        # vertex present (no silent zero for a missing tenor) AND no extra independent
        # pillar (whose CVA dependence would otherwise be silently held fixed/unreported).
        pillar_tenors = {float(tt) for tt, _ in curve.pillars}
        missing = [v for v in IR_DELTA_TENORS if v not in pillar_tenors]
        if missing:
            raise ValidationError(
                f"IR delta requires exact reporting-curve pillars at the SA-CVA vertices "
                f"{list(IR_DELTA_TENORS)}; missing {missing}")
        extra = sorted(pillar_tenors - set(IR_DELTA_TENORS))
        if extra:
            raise ValidationError(
                f"IR delta reporting curve must be parameterized exactly by the SA-CVA "
                f"vertices {list(IR_DELTA_TENORS)}; unsupported extra pillars {extra}")
        sens = []
        for tenor in IR_DELTA_TENORS:
            bumped = key_rate_bumped_curve(curve, tenor, _IR_SHIFT)
            d_cva = 0.0
            for cp in portfolio.counterparties:
                _, cva_up = self._counterparty_cva(self._bump_ir_curve(cp, bumped))
                d_cva += cva_up - base_cva[cp.name]
            sens.append(CVASensitivity(
                risk_class=RiskClass.INTEREST_RATE, risk_type=RiskType.DELTA, bucket=0,
                currency=reporting, tenor=tenor, risk_factor=f"IR:{reporting}:{tenor:g}",
                s_cva=d_cva / _IR_SHIFT))
        return sens

    @staticmethod
    def _bump_ir_curve(cp, bumped_curve):
        """Clone ``cp`` with every trade's discount curve replaced by ``bumped_curve``
        (the single shared reporting curve, key-rate bumped)."""
        new_sets = []
        for ns in cp.netting_sets:
            trades = [replace(t, env=replace(t.env, rate_curve=bumped_curve))
                      for t in ns.trades]
            new_sets.append(replace(ns, trades=trades))
        return replace(cp, netting_sets=new_sets)

    # -- entry point ------------------------------------------------------------
    def compute(self, portfolio):
        """Compute SA-CVA capital from a ``CVATradePortfolio``.

        Returns the SBA ``SACVAResult``; per-counterparty EE profiles and base CVA
        are attached for inspection/audit.
        """
        self._validate_fx_identity(portfolio)
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
        if self.include_ir_delta:
            sensitivities.extend(self._ir_sensitivities(portfolio, base_cva))
        sensitivities.extend(self._hedge_sensitivities(portfolio))

        if not sensitivities:
            raise ValidationError("no SA-CVA sensitivities produced from portfolio")

        cva_portfolio = CVAPortfolio(sensitivities=sensitivities,
                                     reporting_currency=portfolio.reporting_currency)
        result = self.calculator.calculate(cva_portfolio)
        result.exposure_profiles = profiles
        result.counterparty_cva = base_cva
        result.sensitivities = sensitivities   # produced CVASensitivity inputs (audit)
        # record the IR-delta mode so downstream can tell "IR delta exactly zero" from
        # "IR delta intentionally not computed"
        result.ir_delta_included = self.include_ir_delta
        return result
