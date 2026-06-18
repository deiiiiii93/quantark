"""
Historical VaR engine using full portfolio revaluation.
"""

import time
from datetime import datetime
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from quantark.portfolio.equity.portfolio import EquityPortfolio
from quantark.portfolio.fi.portfolio import FIPortfolio
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError, MarketDataError
from quantark.util.marketdata.models import MarketDataSet
from quantark.var.results import IncrementalVaRResult, VaRResult
from quantark.var.config import VaRConfig, VaRMethod


class HistoricalVaREngine:
    """
    Historical Value-at-Risk engine using full portfolio revaluation.

    The Historical VaR engine calculates VaR by repricing the portfolio under
    actual historical market scenarios, without making distributional assumptions
    about returns. This method is considered the most accurate as it captures
    the full non-linear behavior of portfolios, including options and derivatives.

    Key Features:
    - Full portfolio revaluation under each historical scenario
    - Captures non-linear effects (gamma, vega, convexity)
    - No distributional assumptions about returns
    - Supports both DataFrame and MarketDataSet inputs
    - Works with equity and fixed income portfolios
    - Supports Component, Marginal, Factor, Incremental, and Stressed VaR

    Advantages:
    - Most accurate method (uses actual historical data)
    - Handles complex derivatives correctly
    - Captures fat tails and skewness naturally
    - No model risk (e.g., normality assumptions)

    Disadvantages:
    - Requires high-quality historical data
    - Slower than parametric method
    - Limited by historical data length
    - May not reflect current market conditions

    Performance:
    - Calculation time: O(n * p) where n = scenarios, p = positions
    - Memory usage: O(n) for scenario storage
    - Suitable for portfolios up to ~10,000 positions

    Examples:
        Basic historical VaR:
        >>> from var import VaRConfig, HistoricalVaREngine
        >>> config = VaRConfig(confidence_level=0.99)
        >>> engine = HistoricalVaREngine(config=config)
        >>> result = engine.calculate_var(portfolio, historical_data)

        With MarketDataSet:
        >>> result = engine.calculate_var(portfolio, market_data_set)

        With attribution:
        >>> config = VaRConfig(
        ...     confidence_level=0.99,
        ...     calculate_component_var=True,
        ...     calculate_marginal_var=True
        ... )
        >>> engine = HistoricalVaREngine(config=config)
        >>> result = engine.calculate_var(portfolio, data)

        Incremental VaR:
        >>> ivar_result = engine.calculate_incremental_var(portfolio, data)
        >>> print(f"Diversification Benefit: ${ivar_result.diversification_benefit:,.2f}")

    References:
        - Jorion, P. "Value at Risk: The New Benchmark for Managing Financial Risk"
        - Basel Committee. "Basel III: Framework for the measurement and monitoring of VaR"
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

        # For multi-day VaR, apply overlapping returns if configured
        if self.config.holding_period > 1:
            if self.config.scaling_method == "overlapping":
                scenarios = self._generate_overlapping_returns(
                    scenarios, self.config.holding_period
                )
            else:
                # sqrt_t scaling - use tail of scenarios as before
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

        # Create scenarios DataFrame with both market factors and P&L
        scenarios_with_pnl = scenarios.copy()
        scenarios_with_pnl['pnl'] = pnl_scenarios

        result = VaRResult(
            var=var,
            cvar=cvar,
            confidence_level=self.config.confidence_level,
            holding_period=self.config.holding_period,
            method=VaRMethod.HISTORICAL,
            portfolio_value=base_value,
            var_as_pct=var / base_value if base_value != 0 else 0.0,
            scenarios=scenarios_with_pnl,
            worst_scenarios=worst_scenarios,
        )

        # Calculate factor attribution from scenarios (with P&L)
        if self.config.calculate_factor_var:
            result.factor_var = self._calculate_factor_attribution(scenarios_with_pnl)

        # Calculate component VaR if enabled
        if self.config.calculate_component_var:
            result.component_var = self._calculate_component_var(
                portfolio, scenarios, base_value
            )

        # Calculate marginal VaR if enabled
        if self.config.calculate_marginal_var:
            result.marginal_var = self._calculate_marginal_var(
                portfolio, scenarios, base_value
            )

        # Calculate Stressed VaR if enabled
        if self.config.calculate_stressed_var:
            stressed_period = self._detect_stressed_period(
                scenarios, self.config.stressed_lookback_days
            )
            result.stressed_period = stressed_period

            # Filter scenarios to stressed period
            stressed_start = stressed_period["start_date"]
            stressed_end = stressed_period["end_date"]
            stressed_scenarios = scenarios[
                (scenarios.index >= stressed_start) & (scenarios.index <= stressed_end)
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
                portfolio, scenarios, base_value, var
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

    def _scenarios_from_market_data(self, market_data: MarketDataSet) -> pd.DataFrame:
        """
        Extract scenarios from MarketDataSet.

        Converts MarketDataSet containing spot, vol, rate, and dividend time series
        into a DataFrame with returns and changes that can be used as scenarios.

        Args:
            market_data: MarketDataSet with historical time series

        Returns:
            DataFrame with columns: spot_return, vol_change, rate_shift, div_yield_shift

        Raises:
            MarketDataError: If market data is invalid or insufficient
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

    def _generate_overlapping_returns(
        self, scenarios: pd.DataFrame, holding_period: int
    ) -> pd.DataFrame:
        """
        Generate overlapping return scenarios for multi-day VaR.

        Creates overlapping windows of returns to generate more scenarios
        for multi-day VaR calculations, improving accuracy without
        requiring as much historical data.

        Args:
            scenarios: Single-day return scenarios DataFrame
            holding_period: Number of days for VaR calculation

        Returns:
            DataFrame with aggregated returns over the holding period
        """
        overlapping_scenarios = []

        # Generate overlapping windows
        for i in range(len(scenarios) - holding_period + 1):
            window = scenarios.iloc[i : i + holding_period]

            # Aggregate returns over the window
            aggregated = self._aggregate_returns(window)
            aggregated.name = f"window_{i}"
            overlapping_scenarios.append(aggregated)

        if len(overlapping_scenarios) == 0:
            # Fallback to single-day if not enough data
            return scenarios

        return pd.DataFrame(overlapping_scenarios)

    def _aggregate_returns(self, window: pd.DataFrame) -> pd.Series:
        """
        Aggregate returns over a window period.

        Args:
            window: DataFrame of returns over holding period

        Returns:
            Series with aggregated returns
        """
        # For spot returns: compound the returns
        spot_returns = window["spot_return"].values
        compounded_return = np.prod(1.0 + spot_returns) - 1.0

        # For vol changes: sum the changes
        vol_changes = window["vol_change"].values
        total_vol_change = np.sum(vol_changes)

        # For rate shifts: sum the changes
        rate_shifts = window["rate_shift"].values
        total_rate_shift = np.sum(rate_shifts)

        # For div yield shifts: sum the changes
        div_yield_shifts = window["div_yield_shift"].values
        total_div_shift = np.sum(div_yield_shifts)

        # Create aggregated scenario
        aggregated = pd.Series(
            {
                "spot_return": compounded_return,
                "vol_change": total_vol_change,
                "rate_shift": total_rate_shift,
                "div_yield_shift": total_div_shift,
            }
        )

        return aggregated

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

            # Revalue through the position's own BasePosition interface so the
            # engine stays agnostic to how a position prices (option engine vs
            # TRS cashflow re-run). Equivalent to engine.price * quantity for
            # payoff-on-spot positions.
            total_value += position.get_market_value(stressed_env)

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
                - spot_return: Percentage change in spot price (e.g., 0.02 for +2%)
                - vol_change: Absolute change in volatility (e.g., 0.05 for +5%)
                - rate_shift: Absolute change in interest rate (e.g., 0.01 for +1%)
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
        Calculate VaR attribution by risk factor from scenarios.

        Uses correlation-based approach:
        Factor VaR_i = |Correlation(Factor_i, P&L)| × Portfolio VaR

        This approach is more stable than covariance-based decomposition
        when factors are in different units (returns vs dollars).

        Args:
            scenarios: DataFrame of historical scenarios with 'pnl' column

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
        Calculate component VaR for each position using Euler decomposition.

        Component VaR represents each position's marginal contribution to portfolio VaR.
        Uses the P&L scenarios to compute proper Euler allocation.

        Formula: Component VaR_i = Cov(P&L_i, P&L_portfolio) / Var(P&L_portfolio) * Portfolio VaR

        Args:
            portfolio: Portfolio to analyze
            scenarios: Historical scenarios DataFrame
            base_value: Base portfolio value

        Returns:
            Dictionary mapping position ID to component VaR
        """
        # Calculate P&L for each position under all scenarios
        position_pnls = {}
        for pos_id, position in portfolio.positions.items():
            base_env = portfolio.pricing_environments[position.underlying]

            # Get current (un-stressed) price of the position (per unit) via the
            # BasePosition interface so swaps and options are handled uniformly.
            current_price = position.get_current_price(base_env)

            pnls = []
            for idx, scenario in scenarios.iterrows():
                stressed_env = self._create_stressed_environment(base_env, scenario)
                stressed_price = position.get_current_price(stressed_env)
                pnl = stressed_price * position.quantity - (
                    position.quantity * current_price
                )
                pnls.append(pnl)

            position_pnls[pos_id] = np.array(pnls)

        # Get portfolio P&L scenarios (already calculated)
        # Recalculate to ensure we have the array
        pnl_scenarios = self._compute_scenario_pnl(portfolio, scenarios, base_value)

        # Calculate variance of portfolio P&L
        portfolio_var = np.var(pnl_scenarios, ddof=1) if len(pnl_scenarios) > 1 else 1.0

        # Calculate Component VaR using Euler decomposition
        position_component_var = {}
        for pos_id, pos_pnls in position_pnls.items():
            # Calculate covariance between position P&L and portfolio P&L
            covariance = np.cov(pos_pnls, pnl_scenarios, ddof=1)[0, 1] if len(pnl_scenarios) > 1 else 0.0

            # Component VaR = (Covariance / Portfolio Variance) * Portfolio VaR
            # Get portfolio VaR
            var_percentile = (1 - self.config.confidence_level) * 100
            portfolio_var_result = -np.percentile(pnl_scenarios, var_percentile)

            # Beta-like measure of position's contribution to portfolio risk
            if portfolio_var > 0:
                contribution_ratio = covariance / portfolio_var
            else:
                contribution_ratio = 0.0

            # Component VaR can be negative (risk reducer) or positive (risk adder)
            component_var = contribution_ratio * portfolio_var_result

            position_component_var[pos_id] = abs(component_var)

        return position_component_var

    def _calculate_marginal_var(
        self,
        portfolio: Union[EquityPortfolio, FIPortfolio],
        scenarios: pd.DataFrame,
        base_value: float,
    ) -> Dict[str, float]:
        """
        Calculate marginal VaR for each position.

        For Historical VaR, we estimate marginal VaR by measuring the
        change in portfolio VaR when the position is excluded.

        Args:
            portfolio: Portfolio to analyze
            scenarios: Historical scenarios DataFrame
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

        # This is a simplified implementation
        # In practice, would need to properly copy the portfolio
        # For now, return empty dict of positions
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
            scenarios: Historical scenarios DataFrame
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
        from quantark.var.results.incremental_var_result import IncrementalVaRResult

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

        # Calculate Incremental VaR for each position
        position_ivar = self._calculate_incremental_var(
            portfolio, scenarios, base_value, full_var
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
                    for idx, scenario in scenarios.iterrows():
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
            ivari_method="Historical",
            config=self.config.__dict__,
        )

        return result
