"""
Monte Carlo VaR engine using simulated scenarios.
"""

import time
from datetime import datetime
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from portfolio.equity.portfolio import EquityPortfolio
from portfolio.fi.portfolio import FIPortfolio
from priceenv import PricingEnvironment
from util.exceptions import ValidationError, MarketDataError
from util.marketdata.models import MarketDataSet
from var.results import IncrementalVaRResult, VaRResult
from var.config import VaRConfig, VaRMethod


class MonteCarloVaREngine:
    """
    Monte Carlo Value-at-Risk engine using simulated scenarios.

    The Monte Carlo VaR engine calculates VaR by fitting a multivariate
    distribution to historical market data and generating correlated scenarios
    through simulation. It then reprices the portfolio under each simulated
    scenario for accurate VaR estimation.

    Key Features:
    - Fits multivariate distribution to historical data
    - Generates correlated risk factor scenarios
    - Full portfolio revaluation under each scenario
    - Handles path-dependent and complex derivatives
    - Supports both DataFrame and MarketDataSet inputs
    - Works with equity and fixed income portfolios
    - Supports Component, Marginal, Factor, Incremental, and Stressed VaR
    - Configurable number of simulations and random seed

    Methodology:
    1. Extract and align historical risk factor time series
    2. Fit multivariate distribution (typically Gaussian or t-distribution)
    3. Estimate correlation matrix of risk factors
    4. Generate N correlated scenarios via simulation
    5. Revalue portfolio under each scenario
    6. Calculate VaR from simulated P&L distribution

    Advantages:
    - Flexible: Can model complex dependencies and non-linearities
    - Handles path-dependent derivatives (Asian, barrier, lookback options)
    - Can incorporate stochastic volatility and jumps
    - More accurate than parametric for complex portfolios
    - No distributional assumptions at portfolio level
    - Supports stress testing scenarios

    Disadvantages:
    - Slowest calculation method (requires many simulations)
    - Model risk (choice of distribution, correlation structure)
    - Computational intensity for large portfolios
    - Requires careful calibration of simulation parameters
    - May need large number of simulations for tail events

    Performance:
    - Calculation time: O(n * s * p) where n = scenarios, s = simulations, p = positions
    - Memory usage: O(n * s) for storing all simulated scenarios
    - Typical: 10,000 - 100,000 simulations for accurate results
    - Best for portfolios with < 1,000 positions

    Use Cases:
    - Complex derivatives portfolios (Asian, barrier, path-dependent)
    - Options with early exercise features (American, Bermudan)
    - When historical data is limited or incomplete
    - Stress testing with custom scenario generation
    - Portfolios with stochastic volatility or jumps
    - Risk factor modeling with complex dependencies

    Simulation Parameters:
    - mc_num_simulations: Number of scenarios to generate (default: 10,000)
      * More simulations = higher accuracy = slower calculation
      * Typical range: 10,000 - 100,000
    - mc_seed: Random seed for reproducibility (optional)
      * Set to integer for reproducible results
      * Leave as None for system randomness

    Examples:
        Basic Monte Carlo VaR:
        >>> from var import VaRConfig, MonteCarloVaREngine
        >>> config = VaRConfig(
        ...     confidence_level=0.99,
        ...     mc_num_simulations=50000,
        ...     mc_seed=42
        ... )
        >>> engine = MonteCarloVaREngine(config=config)
        >>> result = engine.calculate_var(portfolio, historical_data)

        With attribution:
        >>> config = VaRConfig(
        ...     confidence_level=0.99,
        ...     mc_num_simulations=25000,
        ...     calculate_component_var=True,
        ...     calculate_marginal_var=True
        ... )
        >>> engine = MonteCarloVaREngine(config=config)
        >>> result = engine.calculate_var(portfolio, data)

        For path-dependent options:
        >>> config = VaRConfig(
        ...     confidence_level=0.99,
        ...     mc_num_simulations=100000,  # More simulations needed
        ...     mc_seed=123
        ... )
        >>> engine = MonteCarloVaREngine(config=config)
        >>> result = engine.calculate_var(path_dependent_portfolio, data)

    References:
        - Glasserman, P. "Monte Carlo Methods in Financial Engineering"
        - Boyle, P. "A Lattice Approach to Option Pricing"
        - Glasserman, P., Heidelberger, P., Shahabuddin, P. "Portfolio Specific Value-at-Risk"
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
            historical_returns = self._scenarios_from_dataframe(historical_data)
        else:
            historical_returns = self._scenarios_from_market_data(historical_data)

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

        # Create scenarios DataFrame with both risk factors and P&L
        simulated_scenarios_with_pnl = simulated_scenarios.copy()
        simulated_scenarios_with_pnl['pnl'] = pnl_scenarios

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

        # Calculate factor attribution from simulated scenarios
        if self.config.calculate_factor_var:
            result.factor_var = self._calculate_factor_attribution(simulated_scenarios_with_pnl)

        # Calculate component VaR if enabled
        if self.config.calculate_component_var:
            result.component_var = self._calculate_component_var(
                portfolio, simulated_scenarios, base_value
            )

        # Calculate marginal VaR if enabled
        if self.config.calculate_marginal_var:
            result.marginal_var = self._calculate_marginal_var(
                portfolio, simulated_scenarios, base_value
            )

        # Calculate Stressed VaR if enabled
        if self.config.calculate_stressed_var:
            stressed_period = self._detect_stressed_period(
                simulated_scenarios, self.config.stressed_lookback_days
            )
            result.stressed_period = stressed_period

            # Filter scenarios to stressed period
            stressed_start = stressed_period["start_date"]
            stressed_end = stressed_period["end_date"]
            stressed_scenarios = simulated_scenarios[
                (simulated_scenarios.index >= stressed_start)
                & (simulated_scenarios.index <= stressed_end)
            ]

            if len(stressed_scenarios) > 0:
                # Calculate VaR using stressed period scenarios
                pnl_stressed = self._compute_scenario_pnl(
                    portfolio, stressed_scenarios, base_value
                )

                var_percentile = (1 - self.config.confidence_level) * 100
                stressed_var = -np.percentile(pnl_stressed, var_percentile)

                tail_losses_stressed = pnl_stressed[pnl_stressed <= -stressed_var]
                stressed_cvar = (
                    -tail_losses_stressed.mean()
                    if len(tail_losses_stressed) > 0
                    else stressed_var
                )

                result.stressed_var = abs(stressed_var)
                result.stressed_cvar = abs(stressed_cvar)

        # Calculate Incremental VaR if enabled
        if self.config.calculate_incremental_var:
            result.incremental_var = self._calculate_incremental_var(
                portfolio, simulated_scenarios, base_value, var
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

    def _scenarios_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract scenarios from DataFrame."""
        return df

    def _scenarios_from_market_data(self, market_data: MarketDataSet) -> pd.DataFrame:
        """
        Extract scenarios from MarketDataSet.

        Converts MarketDataSet containing spot, vol, rate, and dividend time series
        into a DataFrame with returns and changes.

        Args:
            market_data: MarketDataSet with historical time series

        Returns:
            DataFrame with columns: spot_return, vol_change, rate_shift, div_yield_shift
        """
        # Align all time series to common date range
        aligned_data = market_data.align_dates()

        # Convert to DataFrames
        spot_df = aligned_data.spot_data.to_dataframe()
        vol_df = aligned_data.vol_data.to_dataframe()
        rate_df = aligned_data.rate_data.to_dataframe()

        # Calculate spot returns (percentage change)
        spot_returns = spot_df["spot"].pct_change().dropna()

        # Calculate volatility changes (absolute change)
        vol_changes = vol_df["volatility"].diff().dropna()

        # Calculate rate shifts (absolute change)
        rate_shifts = rate_df["rate"].diff().dropna()

        # Calculate dividend yield shifts if available
        div_yield_shifts = pd.Series(dtype=float, index=spot_returns.index)
        if aligned_data.div_yield_data is not None:
            div_df = aligned_data.div_yield_data.to_dataframe()
            div_yield_shifts = div_df["div_yield"].diff().dropna()

        # Align all series to common index
        common_index = spot_returns.index.intersection(vol_changes.index)
        common_index = common_index.intersection(rate_shifts.index)
        if len(div_yield_shifts) > 0:
            common_index = common_index.intersection(div_yield_shifts.index)

        if len(common_index) == 0:
            raise MarketDataError("No common dates across all risk factor series")

        # Create scenarios DataFrame
        scenarios = pd.DataFrame(index=common_index)
        scenarios["spot_return"] = spot_returns[common_index]
        scenarios["vol_change"] = vol_changes[common_index]
        scenarios["rate_shift"] = rate_shifts[common_index]

        if len(div_yield_shifts) > 0:
            scenarios["div_yield_shift"] = div_yield_shifts[common_index]
        else:
            scenarios["div_yield_shift"] = 0.0

        # Drop any remaining NaN values
        scenarios = scenarios.dropna()

        if len(scenarios) == 0:
            raise MarketDataError(
                "No valid scenarios after processing. Check data quality."
            )

        return scenarios

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
        """
        Create stressed pricing environment from scenario.

        Applies all risk factor shocks from the scenario to create a stressed
        pricing environment that can be used to reprice portfolio positions.

        Args:
            base_env: Base pricing environment
            scenario: Series with risk factor shocks:
                - spot_return: Percentage change in spot price
                - vol_change: Absolute change in volatility
                - rate_shift: Absolute change in interest rate
                - div_yield_shift: Absolute change in dividend yield

        Returns:
            Stressed pricing environment with all shocks applied
        """
        from copy import deepcopy

        stressed_env = deepcopy(base_env)

        # Apply spot price shock
        if "spot_return" in scenario.index and not pd.isna(scenario["spot_return"]):
            spot_return = scenario["spot_return"]
            stressed_env.spot_quote.spot = base_env.spot_quote.spot * (
                1.0 + spot_return
            )

        # Apply volatility shock
        if "vol_change" in scenario.index and not pd.isna(scenario["vol_change"]):
            vol_change = scenario["vol_change"]
            # Convert vol surface if it exists
            if hasattr(base_env.vol_surface, "volatility"):
                # FlatVolSurface case
                stressed_env.vol_surface.volatility = max(
                    0.0001, base_env.vol_surface.volatility + vol_change
                )

        # Apply interest rate shock
        if "rate_shift" in scenario.index and not pd.isna(scenario["rate_shift"]):
            rate_shift = scenario["rate_shift"]
            if hasattr(base_env.rate_curve, "rate"):
                # FlatRateCurve case
                stressed_env.rate_curve.rate = base_env.rate_curve.rate + rate_shift

        # Apply dividend yield shock
        if "div_yield_shift" in scenario.index and not pd.isna(
            scenario["div_yield_shift"]
        ):
            div_yield_shift = scenario["div_yield_shift"]
            if base_env.div_yield is not None:
                if hasattr(base_env.div_yield, "div_yield"):
                    # ContinuousDividendYield case
                    stressed_env.div_yield.div_yield = max(
                        0.0, base_env.div_yield.div_yield + div_yield_shift
                    )

        return stressed_env

    def _calculate_factor_attribution(
        self, scenarios: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate VaR attribution by risk factor from simulated scenarios.

        Uses correlation-based approach:
        Factor VaR_i = |Correlation(Factor_i, P&L)| × Portfolio VaR

        This approach is more stable than covariance-based decomposition
        when factors are in different units (returns vs dollars).

        Args:
            scenarios: DataFrame of simulated scenarios with 'pnl' column

        Returns:
            Dictionary mapping factor name to VaR contribution
        """
        if 'pnl' not in scenarios.columns:
            # P&L not available in scenarios
            return {}

        pnl = scenarios['pnl'].values

        # Calculate VaR
        var_percentile = (1 - self.config.confidence_level) * 100
        portfolio_var_result = -np.percentile(pnl, var_percentile)

        # Calculate VaR contribution for each risk factor
        factor_var = {}

        # Only consider risk factor columns (skip P&L column)
        for factor in scenarios.columns:
            if factor == 'pnl' or factor == 'scenario_idx':
                continue

            factor_values = scenarios[factor].values

            # Calculate correlation between factor and P&L
            if len(pnl) > 1:
                # Use correlation (unitless) which is more stable
                correlation = np.corrcoef(factor_values, pnl)[0, 1]

                # Factor VaR = |correlation| × Portfolio VaR
                # Take absolute value because VaR is a loss measure
                factor_var_result = abs(correlation) * portfolio_var_result
            else:
                factor_var_result = 0.0

            factor_var[factor] = factor_var_result

        return factor_var

    def _calculate_component_var(
        self,
        portfolio: Union[EquityPortfolio, FIPortfolio],
        scenarios: pd.DataFrame,
        base_value: float,
    ) -> Dict[str, float]:
        """
        Calculate component VaR for each position.

        For Monte Carlo VaR, we estimate component VaR by measuring the
        standalone VaR of each position using simulated scenarios.

        Args:
            portfolio: Portfolio to analyze
            scenarios: Simulated scenarios DataFrame
            base_value: Base portfolio value

        Returns:
            Dictionary mapping position ID to component VaR
        """
        position_component_var = {}

        for pos_id, position in portfolio.positions.items():
            # Revalue this position under all scenarios
            position_pnls = []
            base_env = portfolio.pricing_environments[position.underlying]

            # Get current (un-stressed) price of the position
            current_price = position.engine.price(position.product, base_env)

            for idx, scenario in scenarios.iterrows():
                stressed_env = self._create_stressed_environment(base_env, scenario)
                stressed_price = position.engine.price(position.product, stressed_env)
                pnl = stressed_price * position.quantity - (
                    position.quantity * current_price
                )
                position_pnls.append(pnl)

            position_pnls = np.array(position_pnls)

            # Calculate VaR for this position
            var_percentile = (1 - self.config.confidence_level) * 100
            pos_var = -np.percentile(position_pnls, var_percentile)

            position_component_var[pos_id] = abs(pos_var)

        return position_component_var

    def _calculate_marginal_var(
        self,
        portfolio: Union[EquityPortfolio, FIPortfolio],
        scenarios: pd.DataFrame,
        base_value: float,
    ) -> Dict[str, float]:
        """
        Calculate marginal VaR for each position.

        For Monte Carlo VaR, we estimate marginal VaR by measuring the
        change in portfolio VaR when the position is excluded.

        Args:
            portfolio: Portfolio to analyze
            scenarios: Simulated scenarios DataFrame
            base_value: Base portfolio value

        Returns:
            Dictionary mapping position ID to marginal VaR
        """
        # Calculate VaR without each position
        position_marginal_var = {}

        for pos_id, position in portfolio.positions.items():
            # Create portfolio without this position
            portfolio_without = self._create_portfolio_without_position(
                portfolio, pos_id
            )

            if len(portfolio_without.positions) == 0:
                # Position is the entire portfolio
                position_marginal_var[pos_id] = 0.0
                continue

            # Calculate VaR for portfolio without this position
            base_value_without = portfolio_without.get_portfolio_value()

            if base_value_without > 0:
                pnl_without = []
                for idx, scenario in scenarios.iterrows():
                    stressed_value = self._revalue_portfolio_under_scenario(
                        portfolio_without, scenario
                    )
                    pnl = stressed_value - base_value_without
                    pnl_without.append(pnl)

                pnl_without = np.array(pnl_without)

                var_percentile = (1 - self.config.confidence_level) * 100
                var_without = -np.percentile(pnl_without, var_percentile)

                # Calculate full portfolio VaR
                pnl_full = []
                for idx, scenario in scenarios.iterrows():
                    stressed_value = self._revalue_portfolio_under_scenario(
                        portfolio, scenario
                    )
                    pnl = stressed_value - base_value
                    pnl_full.append(pnl)

                pnl_full = np.array(pnl_full)
                var_full = -np.percentile(pnl_full, var_percentile)

                # Marginal VaR is the difference
                position_marginal_var[pos_id] = abs(var_full - var_without)
            else:
                position_marginal_var[pos_id] = 0.0

        return position_marginal_var

    def _create_portfolio_without_position(
        self, portfolio: Union[EquityPortfolio, FIPortfolio], exclude_pos_id: str
    ) -> Union[EquityPortfolio, FIPortfolio]:
        """
        Create a copy of portfolio without a specific position.

        Args:
            portfolio: Original portfolio
            exclude_pos_id: Position ID to exclude

        Returns:
            New portfolio without the position
        """
        import copy

        # Simplified implementation
        new_portfolio = copy.deepcopy(portfolio)
        if exclude_pos_id in new_portfolio.positions:
            del new_portfolio.positions[exclude_pos_id]

        return new_portfolio

    def _detect_stressed_period(
        self, scenarios: pd.DataFrame, window_size: int = 252
    ) -> Dict[str, datetime]:
        """
        Detect the highest volatility period in the scenarios.

        Uses rolling volatility to identify the most stressful 12-month
        (or specified window) period in the historical data.

        Args:
            scenarios: DataFrame of scenarios with risk factor returns
            window_size: Size of rolling window in days (default 252)

        Returns:
            Dictionary with 'start_date' and 'end_date' of stressed period
        """
        if len(scenarios) < window_size:
            # Not enough data, return entire period
            return {
                "start_date": scenarios.index.min(),
                "end_date": scenarios.index.max(),
            }

        # Calculate portfolio-level volatility
        # For single column, use that column directly
        # For multiple columns, use equal-weighted combination
        if len(scenarios.columns) == 1:
            # Single risk factor - use it directly
            scenario_volatility = scenarios.iloc[:, 0]
        else:
            # Multiple risk factors - use equal-weighted std
            # Calculate std across columns for each row, then average
            scenario_volatility = scenarios.std(axis=1)

        # Calculate rolling volatility
        rolling_vol = scenario_volatility.rolling(window=window_size).std()

        # Drop NaN values before finding max
        rolling_vol_clean = rolling_vol.dropna()

        if len(rolling_vol_clean) == 0:
            # All values are NaN, return entire period
            return {
                "start_date": scenarios.index.min(),
                "end_date": scenarios.index.max(),
            }

        # Find the window with maximum volatility
        max_vol_idx = rolling_vol_clean.idxmax()
        max_vol_date = pd.Timestamp(max_vol_idx)

        # Calculate start and end dates of the stressed period
        end_date = max_vol_date
        start_date = end_date - pd.Timedelta(days=window_size - 1)

        # Ensure dates are within the scenario range
        min_date = scenarios.index.min()
        max_date = scenarios.index.max()

        if start_date < min_date:
            start_date = min_date
        if end_date > max_date:
            end_date = max_date

        return {"start_date": start_date, "end_date": end_date}

    def _calculate_incremental_var(
        self,
        portfolio: Union[EquityPortfolio, FIPortfolio],
        scenarios: pd.DataFrame,
        base_value: float,
        full_portfolio_var: float,
    ) -> Dict[str, float]:
        """
        Calculate Incremental VaR for each position.

        Incremental VaR = VaR(full portfolio) - VaR(portfolio without position)
        Measures the marginal contribution of each position to portfolio VaR.

        Args:
            portfolio: Portfolio to analyze
            scenarios: Simulated scenarios DataFrame
            base_value: Base portfolio value
            full_portfolio_var: VaR of full portfolio

        Returns:
            Dictionary mapping position ID to Incremental VaR
        """
        position_ivar = {}

        for pos_id, position in portfolio.positions.items():
            # Create portfolio without this position
            portfolio_without = self._create_portfolio_without_position(
                portfolio, pos_id
            )

            if len(portfolio_without.positions) == 0:
                # Position is the entire portfolio
                position_ivar[pos_id] = 0.0
                continue

            # Calculate VaR for portfolio without this position
            base_value_without = portfolio_without.get_portfolio_value()

            if base_value_without > 0:
                pnl_without = []
                for idx, scenario in scenarios.iterrows():
                    stressed_value = self._revalue_portfolio_under_scenario(
                        portfolio_without, scenario
                    )
                    pnl = stressed_value - base_value_without
                    pnl_without.append(pnl)

                pnl_without = np.array(pnl_without)

                var_percentile = (1 - self.config.confidence_level) * 100
                var_without = -np.percentile(pnl_without, var_percentile)

                # Incremental VaR is the difference
                position_ivar[pos_id] = abs(full_portfolio_var - var_without)
            else:
                position_ivar[pos_id] = 0.0

        return position_ivar

    def calculate_incremental_var(
        self,
        portfolio: Union[EquityPortfolio, FIPortfolio],
        historical_data: Union[any, pd.DataFrame],
    ) -> "IncrementalVaRResult":
        """
        Calculate Incremental VaR for the portfolio.

        Calculates the contribution of each position to total portfolio VaR
        by measuring the change in VaR when each position is excluded.

        Args:
            portfolio: Portfolio object
            historical_data: Historical market data

        Returns:
            IncrementalVaRResult with position-level IVaR analysis

        Raises:
            ValidationError: If portfolio is empty
            MarketDataError: If insufficient historical data
        """
        from var.results.incremental_var_result import IncrementalVaRResult

        # Calculate full portfolio VaR first
        full_var_result = self.calculate_var(portfolio, historical_data)
        full_var = full_var_result.var

        # Get scenarios
        if isinstance(historical_data, pd.DataFrame):
            scenarios = self._scenarios_from_dataframe(historical_data)
        else:
            scenarios = self._scenarios_from_market_data(historical_data)

        scenarios = scenarios.tail(self.config.lookback_days)
        base_value = portfolio.get_portfolio_value()

        # Generate simulated scenarios
        simulated_scenarios = self._generate_scenarios(scenarios)

        # Calculate Incremental VaR for each position
        position_ivar = self._calculate_incremental_var(
            portfolio, simulated_scenarios, base_value, full_var
        )

        # Calculate VaR without each position for reporting
        var_without_dict = {}
        for pos_id in portfolio.positions.keys():
            portfolio_without = self._create_portfolio_without_position(
                portfolio, pos_id
            )

            if len(portfolio_without.positions) > 0:
                base_value_without = portfolio_without.get_portfolio_value()

                if base_value_without > 0:
                    pnl_without = []
                    for idx, scenario in simulated_scenarios.iterrows():
                        stressed_value = self._revalue_portfolio_under_scenario(
                            portfolio_without, scenario
                        )
                        pnl = stressed_value - base_value_without
                        pnl_without.append(pnl)

                    pnl_without = np.array(pnl_without)
                    var_percentile = (1 - self.config.confidence_level) * 100
                    var_without = -np.percentile(pnl_without, var_percentile)
                    var_without_dict[pos_id] = var_without

        # Calculate diversification benefit
        total_individual_var = sum(position_ivar.values())
        diversification_benefit = total_individual_var - full_var

        # Create result
        result = IncrementalVaRResult(
            portfolio_var=full_var,
            position_ivari=position_ivar,
            diversification_benefit=diversification_benefit,
            portfolio_var_without_position=var_without_dict,
            ivari_method="Monte Carlo",
            config=self.config.__dict__,
        )

        return result
