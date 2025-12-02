"""
Historical VaR engine using full portfolio revaluation.
"""

import time
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from portfolio.equity.portfolio import EquityPortfolio
from portfolio.fi.portfolio import FIPortfolio
from priceenv import PricingEnvironment
from util.exceptions import ValidationError, MarketDataError
from var.base import VaRResult
from var.config import VaRConfig, VaRMethod


class HistoricalVaREngine:
    """
    Historical VaR engine using full portfolio revaluation.

    Computes VaR by repricing the portfolio under historical scenarios
    without making distributional assumptions.
    """

    def __init__(self, config: Optional[VaRConfig] = None):
        """
        Initialize historical VaR engine.

        Args:
            config: VaR configuration
        """
        self.config = config if config is not None else VaRConfig()

        if self.config.var_method != VaRMethod.HISTORICAL:
            self.config.var_method = VaRMethod.HISTORICAL

    def supports_portfolio(self, portfolio: any) -> bool:
        """Check if engine supports the portfolio type."""
        return isinstance(portfolio, (EquityPortfolio, FIPortfolio))

    def calculate_var(
        self,
        portfolio: Union[EquityPortfolio, FIPortfolio],
        historical_data: Union[any, pd.DataFrame],
    ) -> VaRResult:
        """
        Calculate historical VaR using full revaluation.

        Args:
            portfolio: Portfolio object
            historical_data: Historical market data

        Returns:
            VaRResult with VaR metrics
        """
        start_time = time.time()

        if len(portfolio.positions) == 0:
            raise ValidationError("Cannot calculate VaR for empty portfolio")

        if isinstance(historical_data, pd.DataFrame):
            scenarios = self._scenarios_from_dataframe(historical_data)
        else:
            scenarios = self._scenarios_from_market_data(historical_data)

        if len(scenarios) < self.config.lookback_days:
            raise MarketDataError(
                f"Insufficient historical data: {len(scenarios)} days available, "
                f"{self.config.lookback_days} required"
            )

        scenarios = scenarios.tail(self.config.lookback_days)

        base_value = portfolio.get_portfolio_value()

        pnl_scenarios = self._compute_scenario_pnl(portfolio, scenarios, base_value)

        var_percentile = (1 - self.config.confidence_level) * 100
        var = -np.percentile(pnl_scenarios, var_percentile)

        tail_losses = pnl_scenarios[pnl_scenarios <= -var]
        cvar = -tail_losses.mean() if len(tail_losses) > 0 else var

        worst_idx = pnl_scenarios.argsort()[:10]
        worst_scenarios = [
            {"scenario_idx": int(idx), "pnl": float(pnl_scenarios[idx])}
            for idx in worst_idx
        ]

        result = VaRResult(
            var=var,
            cvar=cvar,
            confidence_level=self.config.confidence_level,
            holding_period=self.config.holding_period,
            method=VaRMethod.HISTORICAL,
            portfolio_value=base_value,
            var_as_pct=var / base_value if base_value != 0 else 0.0,
            scenarios=pd.DataFrame({"pnl": pnl_scenarios}),
            worst_scenarios=worst_scenarios,
        )

        result.execution_time_seconds = time.time() - start_time
        result.config_summary = {
            "confidence_level": self.config.confidence_level,
            "holding_period": self.config.holding_period,
            "lookback_days": self.config.lookback_days,
            "method": str(self.config.var_method),
            "num_scenarios": len(scenarios),
        }

        return result

    def _scenarios_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract scenarios from DataFrame."""
        return df

    def _scenarios_from_market_data(self, market_data: any) -> pd.DataFrame:
        """Extract scenarios from MarketDataSet."""
        raise NotImplementedError("MarketDataSet extraction not yet implemented")

    def _compute_scenario_pnl(
        self,
        portfolio: Union[EquityPortfolio, FIPortfolio],
        scenarios: pd.DataFrame,
        base_value: float,
    ) -> np.ndarray:
        """
        Compute P&L for each historical scenario.

        Args:
            portfolio: Portfolio to revalue
            scenarios: DataFrame of historical scenarios
            base_value: Base portfolio value

        Returns:
            Array of P&L values
        """
        pnl_list = []

        for idx, scenario in scenarios.iterrows():
            stressed_value = self._revalue_portfolio_under_scenario(portfolio, scenario)
            pnl = stressed_value - base_value
            pnl_list.append(pnl)

        return np.array(pnl_list)

    def _revalue_portfolio_under_scenario(
        self,
        portfolio: Union[EquityPortfolio, FIPortfolio],
        scenario: pd.Series,
    ) -> float:
        """
        Reprice portfolio under a stressed scenario.

        Args:
            portfolio: Portfolio to reprice
            scenario: Scenario with risk factor shocks

        Returns:
            Stressed portfolio value
        """
        total_value = 0.0

        for position in portfolio.positions.values():
            base_env = portfolio.pricing_environments[position.underlying]

            stressed_env = self._create_stressed_environment(base_env, scenario)

            stressed_price = position.engine.price(position.product, stressed_env)

            total_value += stressed_price * position.quantity

        return total_value

    def _create_stressed_environment(
        self,
        base_env: PricingEnvironment,
        scenario: pd.Series,
    ) -> PricingEnvironment:
        """
        Create stressed pricing environment from scenario.

        Args:
            base_env: Base pricing environment
            scenario: Risk factor shocks

        Returns:
            Stressed pricing environment
        """
        from copy import deepcopy

        stressed_env = deepcopy(base_env)

        if "spot_return" in scenario.index:
            stressed_env.spot_quote.spot = base_env.spot * (1 + scenario["spot_return"])

        if "vol_change" in scenario.index:
            pass

        if "rate_shift" in scenario.index:
            pass

        return stressed_env
