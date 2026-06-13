"""
FX Dynamic Scenario Engine.

Walks an FX portfolio day-by-day through a :class:`DayPath`, applying spot /
vol / domestic-rate / foreign-rate changes to each pair's FxPricingEnvironment
and recording value, P&L and FX greeks at every step. Reuses the generic
``DayResult`` / ``DynamicScenarioResults`` containers.
"""
from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

from quantark.asset.fx.riskmeasures.fx_greeks_calculator import FxGreeksCalculator
from quantark.dynamicscenario.base import BaseDynamicScenarioEngine, RiskMetricsAdapter
from quantark.dynamicscenario.fx.config import FXDynamicScenarioConfig
from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
from quantark.dynamicscenario.results.dynamic_results import (
    DayResult,
    DynamicScenarioResults,
    MarketState,
)
from quantark.portfolio.fx import FXPortfolio
from quantark.stresstest.stress.stress_types import StressLevel
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import pnl_pct_of_abs_baseline
from quantark.var.fx.revaluation import bump_env

_CORE_GREEKS = ("delta", "gamma", "vega", "theta", "rho_dom", "rho_for")


class FXRiskMetricsAdapter(RiskMetricsAdapter):
    """Computes FX greeks for a portfolio (dynamic-scenario protocol)."""

    def __init__(self) -> None:
        self._calc = FxGreeksCalculator()

    def compute_metrics(self, portfolio: Any, pricing_environments: Dict[str, Any]) -> Dict[str, float]:
        return portfolio.get_portfolio_greeks(self._calc)

    def get_metric_names(self) -> List[str]:
        return list(_CORE_GREEKS)


class FXDynamicScenarioEngine(BaseDynamicScenarioEngine):
    """Engine for FX multi-day scenario analysis."""

    def __init__(self, config: Optional[FXDynamicScenarioConfig] = None):
        self.config = config or FXDynamicScenarioConfig()
        self._calc = FxGreeksCalculator()

    def supports_portfolio(self, portfolio: Any) -> bool:
        return isinstance(portfolio, FXPortfolio)

    def get_asset_class(self) -> str:
        return "fx"

    def run(
        self,
        portfolio: FXPortfolio,
        day_path: DayPath,
        hedge_strategy: Optional[Any] = None,
        transaction_cost_model: Optional[Any] = None,
    ) -> DynamicScenarioResults:
        if not self.supports_portfolio(portfolio):
            raise ValidationError("FXDynamicScenarioEngine requires an FXPortfolio")
        if len(portfolio) == 0:
            raise ValidationError("Portfolio must contain at least one position")
        if not day_path or day_path.num_days == 0:
            raise ValidationError("Day path must have at least one day")
        if hedge_strategy is not None:
            raise NotImplementedError(
                "Hedging within FX dynamic scenarios is handled by the FX backtest "
                "module (quantark.backtest.fx); run unhedged scenarios here."
            )

        start_time = time.time()
        working = self._clone_portfolio(portfolio)
        baseline_value = working.get_portfolio_value()
        previous_value = baseline_value
        day_results: List[DayResult] = []

        for day_step in day_path:
            day_date = day_path.get_date_for_day(day_step.day_index)
            self._apply_day_changes(working, day_step)
            if day_date is not None:
                for env in working.pricing_environments.values():
                    env.valuation_date = day_date

            value = working.get_portfolio_value()
            daily_pnl = value - previous_value
            cumulative_pnl = value - baseline_value

            greeks: Dict[str, float] = {}
            if self.config.calculate_greeks:
                greeks = working.get_portfolio_greeks(self._calc)

            day_results.append(DayResult(
                day_index=day_step.day_index,
                date=day_date,
                label=day_step.label,
                portfolio_value=value,
                daily_pnl=daily_pnl,
                cumulative_pnl=cumulative_pnl,
                net_pnl=cumulative_pnl,
                greeks=greeks,
                market_state=self._market_state(working),
            ))
            previous_value = value

        final_value = working.get_portfolio_value()
        return DynamicScenarioResults(
            path_name=day_path.name,
            baseline_value=baseline_value,
            final_value=final_value,
            day_results=day_results,
            total_pnl=final_value - baseline_value,
            total_pnl_pct=pnl_pct_of_abs_baseline(final_value - baseline_value, baseline_value),
            total_execution_time=time.time() - start_time,
            config_summary=self.config.get_summary(),
            metadata={"path_description": day_path.description, "asset_class": "fx"},
        )

    # ------------------------------------------------------------------ #
    def _clone_portfolio(self, portfolio: FXPortfolio) -> FXPortfolio:
        cloned = FXPortfolio(
            portfolio_name=portfolio.portfolio_name + "_simulation",
            pricing_environments={
                pair: deepcopy(env) for pair, env in portfolio.pricing_environments.items()
            },
            creation_date=portfolio.creation_date,
        )
        cloned.positions = deepcopy(portfolio.positions)
        return cloned

    def _apply_day_changes(self, portfolio: FXPortfolio, day_step: DayStep) -> None:
        for change in day_step.changes:
            if change.level == StressLevel.PORTFOLIO:
                for pair in portfolio.pricing_environments:
                    self._apply_change(portfolio, pair, change)
            elif change.level == StressLevel.UNDERLYING:
                if change.target in portfolio.pricing_environments:
                    self._apply_change(portfolio, change.target, change)

    def _apply_change(self, portfolio: FXPortfolio, pair: str, change: ParameterChange) -> None:
        env = portfolio.pricing_environments[pair]
        param = change.parameter
        if param == "spot":
            new = change.apply(env.spot_quote.spot)
            portfolio.pricing_environments[pair] = bump_env(
                env, spot_return=new / env.spot_quote.spot - 1.0)
        elif param in ("vol", "volatility"):
            if env.vol_surface is None:
                return
            new = change.apply(env.vol_surface.volatility)
            portfolio.pricing_environments[pair] = bump_env(
                env, vol_change=new - env.vol_surface.volatility)
        elif param in ("domestic_rate", "rate_dom", "rate"):
            new = change.apply(env.domestic_curve.get_rate(1.0))
            portfolio.pricing_environments[pair] = bump_env(
                env, dom_shift=new - env.domestic_curve.get_rate(1.0))
        elif param in ("foreign_rate", "rate_for"):
            new = change.apply(env.foreign_curve.get_rate(1.0))
            portfolio.pricing_environments[pair] = bump_env(
                env, for_shift=new - env.foreign_curve.get_rate(1.0))
        else:
            raise ValidationError(f"Unsupported FX dynamic parameter '{param}'")

    @staticmethod
    def _market_state(portfolio: FXPortfolio) -> MarketState:
        spot = {p: e.spot_quote.spot for p, e in portfolio.pricing_environments.items()}
        vol = {
            p: (e.vol_surface.volatility if e.vol_surface else 0.0)
            for p, e in portfolio.pricing_environments.items()
        }
        first = next(iter(portfolio.pricing_environments.values()))
        return MarketState(spot=spot, volatility=vol, rate=first.domestic_curve.get_rate(1.0))

    def __repr__(self) -> str:
        return f"FXDynamicScenarioEngine(config={self.config})"
