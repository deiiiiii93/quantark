"""Monte-Carlo regulatory exposure engine (spec §3.2).

Risk-neutral MC: vectorized GBM spot paths (one constant-vol factor per underlying)
+ deterministic value-surface repricing at each exposure node, aggregated to a
discounted EPE profile per MAR50.35 (netting within enforceable sets, summed across
sets). The profile is RISK_NEUTRAL / regulatory-eligible and feeds RegulatoryCVAEngine.

v1 scope: equity (and reporting-vs-foreign FX) spot underlyings, deterministic
rates, single reporting currency, uncollateralized. Vanilla (single-state) trades
are priced via the analytic value surface (full vol surface re-evaluated, terminal
payoff exact). Stateful trades whose engine advertises ``supports_spot_greeks_grid``
need the grid value-surface + barrier state machine wiring and currently raise with
a clear message (next integration step). Unsupported products raise, never approximate.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from quantark.sacva.exposure.asof import equity_asof_env
from quantark.sacva.exposure.correlation import CorrelationModel
from quantark.sacva.exposure.engine import (
    ExposureEngine,
    ExposureProfile,
    Measure,
    aggregate_epe,
)
from quantark.sacva.exposure.grid import ExposureGrid
from quantark.sacva.exposure.paths import StatePathGenerator
from quantark.sacva.exposure.value_surface import AnalyticValueSurface
from quantark.util.exceptions import ValidationError

# tau below this (in years, ~1 calendar day) is treated as the terminal node and
# valued by the contractual payoff, sidestepping the engine's exercise-date guard
_TAU_FLOOR = 1.0 / 365.0


@dataclass
class MonteCarloExposureConfig:
    num_paths: int = 20000
    n_steps: int = 24
    seed: int = 12345
    correlation: Optional[CorrelationModel] = None  # required iff >1 underlying


class MonteCarloExposureEngine(ExposureEngine):
    """Risk-neutral MC exposure for one counterparty -> discounted EPE profile."""

    measure = Measure.RISK_NEUTRAL
    regulatory_eligible = True

    def __init__(self, config: Optional[MonteCarloExposureConfig] = None):
        self.config = config or MonteCarloExposureConfig()

    # -- risk-factor extraction -------------------------------------------------
    def _underlying_key(self, trade):
        sq = getattr(trade.env, "spot_quote", None)
        key = getattr(sq, "asset_name", None) if sq is not None else None
        if not key:
            raise ValidationError(
                f"{trade.trade_id}: env.spot_quote.asset_name is required to identify "
                "the underlying risk factor")
        return key

    def _trade_maturity(self, trade):
        T = float(trade.product.get_maturity(trade.env))
        if not (T > 0):
            raise ValidationError(f"{trade.trade_id}: non-positive maturity {T}")
        return T

    def compute(self, counterparty) -> ExposureProfile:
        trades = [t for ns in counterparty.netting_sets for t in ns.trades]
        if not trades:
            raise ValidationError(f"{counterparty.name}: no trades")

        # stateful (grid) trades are deferred within this engine version
        for t in trades:
            if getattr(t.engine, "supports_spot_greeks_grid", False):
                raise ValidationError(
                    f"{t.trade_id}: grid/stateful (snowball/phoenix) exposure needs the "
                    "GridValueSurface + BarrierStateMachine wiring (next integration step)")

        # per-underlying market (constant-vol GBM factor), taken from the first trade
        # on that key; horizon = max maturity across the counterparty
        horizon = max(self._trade_maturity(t) for t in trades)
        keys, spots, vols, rates, divs = [], [], [], [], []
        market_env = {}
        for t in trades:
            k = self._underlying_key(t)
            if k in market_env:
                if abs(market_env[k].spot - t.env.spot) > 1e-9:
                    raise ValidationError(
                        f"inconsistent spot for underlying {k} across trades")
                continue
            market_env[k] = t.env
            spot = float(t.env.spot)
            keys.append(k)
            spots.append(spot)
            vols.append(float(t.env.get_vol(spot, horizon)))   # ATM term vol
            rates.append(float(t.env.get_rate(horizon)))
            divs.append(float(t.env.get_div_yield(horizon)))

        if len(keys) == 1:
            corr = [[1.0]]
        else:
            if self.config.correlation is None:
                raise ValidationError(
                    "multi-underlying counterparty requires a correlation matrix in "
                    "MonteCarloExposureConfig (independence is not assumed)")
            cm = self.config.correlation
            if list(cm.keys) != keys:
                raise ValidationError(
                    f"correlation keys {list(cm.keys)} must match underlyings {keys}")
            corr = cm.matrix

        grid = ExposureGrid.build(horizon=horizon, n_steps=self.config.n_steps,
                                  event_times=[])
        times = grid.times
        gen = StatePathGenerator(keys=keys, spots=spots, vols=vols, rates=rates,
                                 divs=divs, corr=corr, grid_times=times,
                                 num_paths=self.config.num_paths, seed=self.config.seed)
        paths = gen.generate()

        # discount factors on the reporting curve (single currency v1)
        ref_curve = trades[0].env
        df = np.array([float(ref_curve.get_discount_factor(float(ti))) for ti in times])

        # pathwise undiscounted values per trade
        trade_values = {id(t): self._trade_value_array(t, paths, times) for t in trades}

        # counterparty EPE = sum over sets of set-level discounted EPE (MAR50.35)
        epe = np.zeros(len(times))
        for ns in counterparty.netting_sets:
            arrays = [trade_values[id(t)] for t in ns.trades]
            epe = epe + aggregate_epe(arrays, enforceable=ns.netting_enforceable, df=df)

        return ExposureProfile(times=times, epe_discounted=epe,
                               measure=Measure.RISK_NEUTRAL, regulatory_eligible=True)

    def _trade_value_array(self, trade, paths, times):
        """Pathwise UNDISCOUNTED reporting-currency value, shape (num_paths, n_t)."""
        key = self._underlying_key(trade)
        spots = paths[key]                       # (num_paths, n_t)
        T = self._trade_maturity(trade)
        surface = AnalyticValueSurface(
            engine=trade.engine, product=trade.product, base_env=trade.env,
            as_of_env=equity_asof_env, currency=trade.trade_currency)
        out = np.zeros_like(spots)
        for j, tj in enumerate(times):
            tau = T - float(tj)
            col = spots[:, j]
            if tau >= _TAU_FLOOR:
                out[:, j] = surface.value_at(col, float(tj), None)
            elif tau >= 0.0:                     # terminal node: contractual payoff
                if not hasattr(trade.product, "get_payoff"):
                    raise ValidationError(
                        f"{trade.trade_id}: product lacks get_payoff for terminal node")
                out[:, j] = np.array([trade.product.get_payoff(float(s)) for s in col])
            # tau < 0: matured/settled -> 0
        return out * float(trade.quantity)
