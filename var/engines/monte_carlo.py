"""
Monte Carlo VaR engine using simulated scenarios.
"""

import time
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from portfolio.equity.portfolio import EquityPortfolio
from portfolio.fi.portfolio import FIPortfolio
from priceenv import PricingEnvironment
from util.exceptions import ValidationError, MarketDataError
from var.base import VaRResult
from var.config import VaRConfig, VaRMethod


class MonteCarloVaREngine:
    """
    Monte Carlo VaR engine using simulated scenarios.

    Fits multivariate distribution to historical data and generates
    correlated scenarios for full portfolio revaluation.
    """

    def __init__(self, config: Optional[VaRConfig] = None):
        """
        Initialize Monte Carlo VaR engine.

        Args:
            config: VaR configuration
        """
        self.config = config if config is not None else VaRConfig()

        if self.config.var_method != VaRMethod.MONTE_CARLO:
            self.config.var_method = VaRMethod.MONTE_CARLO

    def supports_portfolio(self, portfolio: any) -> bool:
        """Check if engine supports the portfolio type."""
        return isinstance(portfolio, (EquityPortfolio, FIPortfolio))

    def calculate_var(
        self,
        portfolio: Union[EquityPortfolio, FIPortfolio],
        historical_data: Union[any, pd.DataFrame],
    ) -> VaRResult:
        """
        Calculate Monte Carlo VaR.

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
            historical_returns = historical_data
        else:
            raise NotImplementedError("MarketDataSet not yet supported")

        if len(historical_returns) < self.config.lookback_days:
            raise MarketDataError(
                f"Insufficient historical data: {len(historical_returns)} days available"
            )

        historical_returns = historical_returns.tail(self.config.lookback_days)

        simulated_scenarios = self._generate_scenarios(historical_returns)

        base_value = portfolio.get_portfolio_value()

        pnl_scenarios = self._compute_scenario_pnl(
            portfolio, simulated_scenarios, base_value
        )

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
            method=VaRMethod.MONTE_CARLO,
            portfolio_value=base_value,
            var_as_pct=var / base_value if base_value != 0 else 0.0,
            scenarios=pd.DataFrame({"pnl": pnl_scenarios}),
            worst_scenarios=worst_scenarios,
        )

        result.execution_time_seconds = time.time() - start_time
        result.config_summary = {
            "confidence_level": self.config.confidence_level,
            "holding_period": self.config.holding_period,
            "mc_num_simulations": self.config.mc_num_simulations,
            "mc_seed": self.config.mc_seed,
            "method": str(self.config.var_method),
        }

        return result

    def _generate_scenarios(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate correlated scenarios using Cholesky decomposition.

        Args:
            historical_data: Historical returns/changes

        Returns:
            DataFrame of simulated scenarios
        """
        if self.config.mc_seed is not None:
            np.random.seed(self.config.mc_seed)

        means = historical_data.mean().values
        cov_matrix = historical_data.cov().values

        try:
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            cov_matrix += np.eye(len(cov_matrix)) * 1e-8
            L = np.linalg.cholesky(cov_matrix)

        num_factors = len(means)
        Z = np.random.standard_normal((self.config.mc_num_simulations, num_factors))

        scenarios = means + (L @ Z.T).T

        return pd.DataFrame(scenarios, columns=historical_data.columns)

    def _compute_scenario_pnl(
        self,
        portfolio: Union[EquityPortfolio, FIPortfolio],
        scenarios: pd.DataFrame,
        base_value: float,
    ) -> np.ndarray:
        """Compute P&L for each simulated scenario."""
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
        """Reprice portfolio under a scenario."""
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
        """Create stressed pricing environment."""
        from copy import deepcopy

        stressed_env = deepcopy(base_env)

        if "spot_return" in scenario.index:
            stressed_env.spot_quote.spot = base_env.spot * (1 + scenario["spot_return"])

        return stressed_env
