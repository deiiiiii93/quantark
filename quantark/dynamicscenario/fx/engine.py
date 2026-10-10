"""
FX Dynamic Scenario Engine.

Walks an FX portfolio day-by-day through a :class:`DayPath`, applying spot /
vol / domestic-rate / foreign-rate changes to each pair's FxPricingEnvironment
and recording value, P&L and FX greeks at every step. Reuses the generic
``DayResult`` / ``DynamicScenarioResults`` containers.

TARN / TARF positions age with a dated run on the run-fixings recorder: each banked fixing is reported as one
``FIXING`` lifecycle snapshot, determined cash moves into ``paid_cash`` / ``realized_cash`` once its pay date is past
(on the next path instant), each day carries the pending cash's PV (``pending_receivable_pv``) and ``portfolio_value``
includes cumulative paid cash. An undated path cannot age them and is refused (spec 2026-10-09 fx-tarn 12 Q8 a).
"""
from __future__ import annotations

import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

from quantark.asset.fx.product.option.target_redemption_base import TargetRedemptionProduct
from quantark.asset.fx.product.option.target_redemption_state import (
    TargetRedemptionEvent,
    TargetRedemptionEventType,
    booked_cash,
)
from quantark.asset.fx.riskmeasures.fx_greeks_calculator import FxGreeksCalculator
from quantark.dynamicscenario.base import BaseDynamicScenarioEngine, RiskMetricsAdapter
from quantark.dynamicscenario.fx.config import FXDynamicScenarioConfig
from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
from quantark.dynamicscenario.results.dynamic_results import (
    DayResult,
    DynamicScenarioResults,
    LifecycleEventSnapshot,
    MarketState,
)
from quantark.portfolio.fx import FXPortfolio
from quantark.stresstest.stress.stress_types import StressLevel
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import pnl_pct_of_abs_baseline
from quantark.var.fx.revaluation import bump_env
from quantark.util.run_fixings import RunFixingRecorder

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
        self.run_fixings: RunFixingRecorder = RunFixingRecorder()   # the last run's (spec 2026-10-09 run fixings)

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
        if day_path.start_date is None and any(isinstance(p.product, TargetRedemptionProduct)
                                               for p in portfolio.positions.values()):
            raise ValidationError(  # TARN spec 12 Q8 (a): no clock is invented (an undated run never advances, W6 R7)
                "the book holds an FX TARN / TARF, which ages by date: set DayPath.start_date (an undated path "
                "cannot age it)")
        working = self._clone_portfolio(portfolio)
        # Run fixings (spec 2026-10-09 run fixings §3.5): registered on the WORKING copy at the first valuation (the
        # input portfolio is never touched); a fixing already missing is filled once at the baseline spot (R5).
        self.run_fixings = RunFixingRecorder()
        self.run_fixings.register(working)
        baseline_value = working.get_portfolio_value()
        previous_value = baseline_value
        cum_paid = 0.0   # TARN / TARF cash paid inside the run (spec 2026-10-09 fx-tarn 7.3): quantity-scaled
        day_results: List[DayResult] = []

        for day_step in day_path:
            day_date = day_path.get_date_for_day(day_step.day_index)
            self._apply_day_changes(working, day_step)
            snapshots: List[LifecycleEventSnapshot] = []
            pending_pv = 0.0
            if day_date is not None:
                for env in working.pricing_environments.values():
                    env.valuation_date = day_date
                # the clock moved: record the fixings it passed at the previous instant's spot (R3/R4); an undated
                # path never moves the date, so nothing passes and nothing is advanced (R7)
                day = self.run_fixings.advance(working)
                cum_paid += booked_cash(day.product_events)
                pending_pv = day.pending_pv
                snapshots = self._fixing_snapshots(working, day, day_date)

            value = working.get_portfolio_value() + cum_paid
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
                lifecycle_events=snapshots,
                pending_receivable_pv=pending_pv,
                paid_cash=cum_paid,
                realized_cash=cum_paid,
            ))
            previous_value = value

        final_value = working.get_portfolio_value() + cum_paid
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
    def _fixing_snapshots(portfolio: FXPortfolio, day: Any, day_date: datetime) -> List[LifecycleEventSnapshot]:
        """One FIXING snapshot per TARN / TARF fixing banked in the step (``RunFixingDay.product_events``);
        ``cashflow`` = quantity x that fixing's cash paid in the same step (an immediate PAYMENT carries its FIXING's
        label - Ruling R6 of the W4b plan)."""
        fixing, payment = TargetRedemptionEventType.FIXING, TargetRedemptionEventType.PAYMENT
        out: List[LifecycleEventSnapshot] = []
        for pid, quantity, ev in day.product_events:
            if not (isinstance(ev, TargetRedemptionEvent) and ev.event_type is fixing):
                continue
            paid = 0.0
            for other_pid, _q, other in day.product_events:
                if (other_pid == pid and isinstance(other, TargetRedemptionEvent) and other.event_type is payment
                        and other.label == ev.label):
                    paid += other.amount
            position = portfolio.positions[pid]
            out.append(LifecycleEventSnapshot(
                position_id=pid, underlying=position.underlying, product_type=type(position.product).__name__,
                event_type="FIXING", date=day_date, observation_index=None, spot=ev.fixing, barrier=None,
                payoff=ev.amount, cashflow=quantity * paid, terminates_position=ev.redeemed,
            ))
        return out

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
