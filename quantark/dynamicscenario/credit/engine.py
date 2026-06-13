"""
Credit Dynamic Scenario Engine.

Walks a credit portfolio day-by-day through a DayPath, applying spread (hazard)
and rate changes to each entity's CreditPricingEnvironment and recording value,
P&L and credit greeks at every step. Reuses the generic DayResult /
DynamicScenarioResults containers.
"""
from __future__ import annotations

import time
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional

from quantark.asset.credit.engine.schedule import contractual_coupon_dates
from quantark.asset.credit.product.cds import CDS
from quantark.asset.credit.riskmeasures import CreditGreeksCalculator
from quantark.dynamicscenario.base import BaseDynamicScenarioEngine, RiskMetricsAdapter
from quantark.dynamicscenario.credit.config import CreditDynamicScenarioConfig
from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
from quantark.dynamicscenario.results.dynamic_results import (
    DayResult,
    DynamicScenarioResults,
    MarketState,
)
from quantark.portfolio.credit import CreditPortfolio
from quantark.stresstest.stress.stress_types import StressLevel
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import pnl_pct_of_abs_baseline
from quantark.var.credit.revaluation import bump_env

_CORE_GREEKS = ("hazard01", "cs01", "ir01", "rec01")


class CreditRiskMetricsAdapter(RiskMetricsAdapter):
    """Computes credit greeks for a portfolio (dynamic-scenario protocol)."""

    def __init__(self) -> None:
        self._calc = CreditGreeksCalculator()

    def compute_metrics(
        self, portfolio: Any, pricing_environments: Dict[str, Any]
    ) -> Dict[str, float]:
        return portfolio.get_portfolio_greeks(self._calc)

    def get_metric_names(self) -> List[str]:
        return list(_CORE_GREEKS)


class CreditDynamicScenarioEngine(BaseDynamicScenarioEngine):
    """Engine for credit multi-day scenario analysis."""

    def __init__(self, config: Optional[CreditDynamicScenarioConfig] = None):
        self.config = config or CreditDynamicScenarioConfig()
        self._calc = CreditGreeksCalculator()

    def supports_portfolio(self, portfolio: Any) -> bool:
        return isinstance(portfolio, CreditPortfolio)

    def get_asset_class(self) -> str:
        return "credit"

    def run(
        self,
        portfolio: CreditPortfolio,
        day_path: DayPath,
        hedge_strategy: Optional[Any] = None,
        transaction_cost_model: Optional[Any] = None,
    ) -> DynamicScenarioResults:
        if not self.supports_portfolio(portfolio):
            raise ValidationError("CreditDynamicScenarioEngine requires a CreditPortfolio")
        if len(portfolio) == 0:
            raise ValidationError("Portfolio must contain at least one position")
        if not day_path or day_path.num_days == 0:
            raise ValidationError("Day path must have at least one day")
        if hedge_strategy is not None:
            raise NotImplementedError(
                "Hedging within credit dynamic scenarios is handled by the credit "
                "backtest module (quantark.backtest.credit); run unhedged here."
            )

        start_time = time.time()
        working = self._clone_portfolio(portfolio)
        baseline_value = working.get_portfolio_value()
        previous_value = baseline_value
        day_results: List[DayResult] = []

        # As the valuation date advances the reduced-form engine reprices each
        # seasoned CDS (roll-down). Coupons that settle along the way leave the
        # PV, so they are booked into a realized-cash ledger to report a
        # carry-preserving *total return* rather than a mark-to-market that dips
        # on every coupon date. Un-dated CDS produce no coupon dates and so leave
        # the ledger at zero (legacy price-return behaviour is unchanged).
        previous_date = self._portfolio_valuation_date(working)
        realized_cash = 0.0

        for day_step in day_path:
            day_date = day_path.get_date_for_day(day_step.day_index)
            self._apply_day_changes(working, day_step)
            if day_date is not None:
                for env in working.pricing_environments.values():
                    env.valuation_date = day_date
                if previous_date is not None:
                    for pos in working.positions.values():
                        realized_cash += self._realized_coupon_cash(
                            pos, previous_date, day_date
                        )
                previous_date = day_date

            # Total-return NAV = remaining mark-to-market + cash booked so far.
            value = working.get_portfolio_value() + realized_cash
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

        final_value = working.get_portfolio_value() + realized_cash
        return DynamicScenarioResults(
            path_name=day_path.name,
            baseline_value=baseline_value,
            final_value=final_value,
            day_results=day_results,
            total_pnl=final_value - baseline_value,
            total_pnl_pct=pnl_pct_of_abs_baseline(
                final_value - baseline_value, baseline_value
            ),
            total_execution_time=time.time() - start_time,
            config_summary=self.config.get_summary(),
            metadata={"path_description": day_path.description, "asset_class": "credit"},
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _portfolio_valuation_date(portfolio: CreditPortfolio):
        """Common valuation date of the book (cloned envs share one date)."""
        for env in portfolio.pricing_environments.values():
            return env.valuation_date
        return None

    @staticmethod
    def _realized_coupon_cash(position, start, end) -> float:
        """
        Premium cash settled in ``(start, end]`` for one seasoned CDS position.

        Protection buyers pay the running coupon (cash outflow), sellers receive
        it. In a no-default scenario walk the name is assumed to survive, so each
        contractual coupon is booked in full at its face amount when its payment
        date is crossed. Un-dated CDS (no contractual calendar) book nothing.
        """
        product = position.product
        if not isinstance(product, CDS) or not product.is_dated:
            return 0.0
        cash = 0.0
        for pay_date, accrual in contractual_coupon_dates(
            product.effective_date, product.maturity_date, product.payment_freq
        ):
            if start < pay_date <= end:
                coupon = product.notional * product.coupon_spread * accrual
                cash += -product.side_sign * position.quantity * coupon
        return cash

    def _clone_portfolio(self, portfolio: CreditPortfolio) -> CreditPortfolio:
        cloned = CreditPortfolio(
            portfolio_name=portfolio.portfolio_name + "_simulation",
            pricing_environments={
                entity: deepcopy(env)
                for entity, env in portfolio.pricing_environments.items()
            },
            creation_date=portfolio.creation_date,
        )
        cloned.positions = deepcopy(portfolio.positions)
        return cloned

    def _apply_day_changes(self, portfolio: CreditPortfolio, day_step: DayStep) -> None:
        for change in day_step.changes:
            if change.level == StressLevel.PORTFOLIO:
                for entity in portfolio.pricing_environments:
                    self._apply_change(portfolio, entity, change)
            elif change.level == StressLevel.UNDERLYING:
                if change.target in portfolio.pricing_environments:
                    self._apply_change(portfolio, change.target, change)

    def _apply_change(
        self, portfolio: CreditPortfolio, entity: str, change: ParameterChange
    ) -> None:
        env = portfolio.pricing_environments[entity]
        param = change.parameter
        if param in ("hazard", "spread", "credit_spread"):
            if param != "hazard":
                warnings.warn(
                    f"Credit dynamic-scenario parameter '{param}' is applied as a "
                    "hazard-intensity shock; use the canonical 'hazard'. The "
                    "'spread'/'credit_spread' aliases are deprecated.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            current = env.hazard_curve.get_hazard_rate(1.0)
            new = change.apply(current)
            portfolio.pricing_environments[entity] = bump_env(
                env, hazard_change=new - current
            )
        elif param == "rate":
            current = env.discount_curve.get_rate(1.0)
            new = change.apply(current)
            portfolio.pricing_environments[entity] = bump_env(
                env, rate_shift=new - current
            )
        else:
            raise ValidationError(f"Unsupported credit dynamic parameter '{param}'")

    @staticmethod
    def _market_state(portfolio: CreditPortfolio) -> MarketState:
        # Display the issuer hazard intensities under the generic 'spot' map.
        hazards = {
            e: env.hazard_curve.get_hazard_rate(1.0)
            for e, env in portfolio.pricing_environments.items()
        }
        first = next(iter(portfolio.pricing_environments.values()))
        return MarketState(spot=hazards, volatility={}, rate=first.discount_curve.get_rate(1.0))

    def __repr__(self) -> str:
        return f"CreditDynamicScenarioEngine(config={self.config})"
