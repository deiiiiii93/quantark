"""
Parametric VaR engine using variance-covariance approach.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from quantark.portfolio.equity.portfolio import EquityPortfolio
from quantark.portfolio.fi.portfolio import FIPortfolio
from quantark.util.exceptions import ValidationError, MarketDataError
from quantark.util.numerical import is_zero
from quantark.var.results import IncrementalVaRResult, VaRResult
from quantark.var.config import VaRConfig, VaRMethod, EquityRiskFactorConfig
from quantark.var.risk_factors import (
    SpotReturnFactor,
    VolChangeFactor,
    RateShiftFactor,
    DivYieldShiftFactor,
)


class ParametricVaREngine:
    """
    Parametric Value-at-Risk engine using variance-covariance approach.

    The Parametric VaR engine calculates VaR using portfolio sensitivities
    (Greeks for equity, DV01 for fixed income) and the historical covariance
    matrix of risk factors. This is also known as the variance-covariance method
    or the sensitivity-based method.

    Key Features:
    - Uses portfolio sensitivities (delta, gamma, vega, rho, DV01)
    - Leverages historical covariance matrix of risk factors
    - Fastest calculation method (closed-form solutions)
    - Supports both DataFrame and MarketDataSet inputs
    - Works with equity and fixed income portfolios
    - Supports Component, Marginal, Factor, Incremental, and Stressed VaR
    - Supports Fixed Income risk factors (parallel shift, key rates)

    Mathematical Foundation:
    VaR = z_score * sqrt(s^T * Σ * s)
    where:
    - s = sensitivity vector (Greeks/DV01)
    - Σ = covariance matrix of risk factors
    - z_score = inverse CDF of normal distribution at confidence level

    Advantages:
    - Fastest calculation (scalable to very large portfolios)
    - Closed-form Greeks support
    - Well-suited for linear portfolios
    - Real-time risk monitoring
    - Efficient for backtesting
    - Industry standard for equity and FI trading

    Disadvantages:
    - Assumes linear relationship (or approximations for non-linear)
    - Distributional assumptions (normally distributed returns)
    - Limited accuracy for options and derivatives
    - Requires reliable Greeks calculations
    - May not capture fat tails

    Performance:
    - Calculation time: O(f^3) for covariance matrix inversion, O(p*f) for sensitivities
    - Memory usage: O(f^2) for covariance matrix storage
    - Suitable for portfolios with 100,000+ positions
    - Excellent for real-time risk monitoring

    Use Cases:
    - Large equity portfolios (delta, gamma, vega monitoring)
    - Fixed income portfolios (DV01, convexity monitoring)
    - Real-time P&L attribution
    - Stress testing with sensitivity shocks
    - Regulatory reporting (sensitivity-based)
    - Risk decomposition and attribution

    Examples:
        Basic parametric VaR:
        >>> from var import VaRConfig, ParametricVaREngine
        >>> config = VaRConfig(confidence_level=0.99)
        >>> engine = ParametricVaREngine(config=config)
        >>> result = engine.calculate_var(portfolio, risk_factors)

        Equity with options:
        >>> from var.config import EquityRiskFactorConfig
        >>> config = VaRConfig(
        ...     confidence_level=0.99,
        ...     equity_factors=EquityRiskFactorConfig(
        ...         include_spot=True,
        ...         include_vol=True,
        ...         include_rate=True
        ...     )
        ... )
        >>> engine = ParametricVaREngine(config=config)
        >>> result = engine.calculate_var(equity_portfolio, data)

        Fixed Income with key rates:
        >>> from var.config import FIRiskFactorConfig
        >>> config = VaRConfig(
        ...     confidence_level=0.99,
        ...     fi_factors=FIRiskFactorConfig(
        ...         include_parallel_shift=True,
        ...         include_key_rates=True,
        ...         key_rate_tenors=[2.0, 5.0, 10.0, 30.0]
        ...     )
        ... )
        >>> engine = ParametricVaREngine(config=config)
        >>> result = engine.calculate_var(fi_portfolio, fi_data)

        With attribution:
        >>> config = VaRConfig(
        ...     confidence_level=0.99,
        ...     calculate_component_var=True,
        ...     calculate_marginal_var=True,
        ...     calculate_incremental_var=True
        ... )
        >>> engine = ParametricVaREngine(config=config)
        >>> result = engine.calculate_var(portfolio, data)

    References:
        - RiskMetrics Group. "RiskMetrics™ Technical Document"
        - Basel Committee. "The Internal Ratings-Based Approach"
        - Jorion, P. "Value at Risk: The New Benchmark for Managing Financial Risk"
    """

    def __init__(self, config: Optional[VaRConfig] = None):
        """
        Initialize parametric VaR engine.

        Args:
            config: VaR configuration (defaults to VaRConfig())
        """
        self.config = config if config is not None else VaRConfig()

        if self.config.var_method != VaRMethod.PARAMETRIC:
            self.config.var_method = VaRMethod.PARAMETRIC

    def supports_portfolio(self, portfolio: any) -> bool:
        """Check if engine supports the portfolio type."""
        return isinstance(portfolio, (EquityPortfolio, FIPortfolio))

    def calculate_var(
        self,
        portfolio: Union[EquityPortfolio, FIPortfolio],
        historical_data: Union[any, pd.DataFrame],
    ) -> VaRResult:
        """
        Calculate parametric VaR for the portfolio.

        Args:
            portfolio: Portfolio object
            historical_data: Historical market data (MarketDataSet or DataFrame)

        Returns:
            VaRResult with VaR metrics

        Raises:
            ValidationError: If portfolio is empty or inputs invalid
            MarketDataError: If historical data insufficient
        """
        start_time = time.time()

        if len(portfolio.positions) == 0:
            raise ValidationError("Cannot calculate VaR for empty portfolio")

        if isinstance(portfolio, EquityPortfolio):
            result = self._calculate_equity_var(portfolio, historical_data)
        elif isinstance(portfolio, FIPortfolio):
            result = self._calculate_fi_var(portfolio, historical_data)
        else:
            raise ValidationError(
                f"Unsupported portfolio type: {type(portfolio).__name__}"
            )

        result.execution_time_seconds = time.time() - start_time
        result.config_summary = {
            "confidence_level": self.config.confidence_level,
            "holding_period": self.config.holding_period,
            "lookback_days": self.config.lookback_days,
            "method": str(self.config.var_method),
        }

        return result

    def _calculate_equity_var(
        self,
        portfolio: EquityPortfolio,
        historical_data: Union[any, pd.DataFrame],
    ) -> VaRResult:
        """Calculate parametric VaR for equity portfolio."""
        if isinstance(historical_data, pd.DataFrame):
            risk_factors_df = self._extract_risk_factors_from_dataframe(
                historical_data, is_equity=True
            )
        else:
            risk_factors_df = self._extract_risk_factors_from_market_data(
                historical_data, is_equity=True
            )

        if len(risk_factors_df) < self.config.lookback_days:
            raise MarketDataError(
                f"Insufficient historical data: {len(risk_factors_df)} days "
                f"available, {self.config.lookback_days} required"
            )

        risk_factors_df = risk_factors_df.tail(self.config.lookback_days)

        sensitivities = self._compute_equity_sensitivities(portfolio)

        cov_matrix = risk_factors_df.cov().values

        sensitivity_vector = np.array(list(sensitivities.values()))

        portfolio_variance = sensitivity_vector @ cov_matrix @ sensitivity_vector
        portfolio_std = np.sqrt(portfolio_variance)

        if self.config.holding_period > 1:
            if self.config.scaling_method == "sqrt_t":
                portfolio_std *= np.sqrt(self.config.holding_period)

        z_score = stats.norm.ppf(self.config.confidence_level)
        var = z_score * portfolio_std

        cvar = (
            portfolio_std * stats.norm.pdf(z_score) / (1 - self.config.confidence_level)
        )

        portfolio_value = portfolio.get_portfolio_value()

        result = VaRResult(
            var=abs(var),
            cvar=abs(cvar),
            confidence_level=self.config.confidence_level,
            holding_period=self.config.holding_period,
            method=VaRMethod.PARAMETRIC,
            portfolio_value=portfolio_value,
            var_as_pct=abs(var) / portfolio_value if portfolio_value != 0 else 0.0,
        )

        if self.config.calculate_factor_var:
            result.factor_var = self._compute_factor_var(
                sensitivity_vector,
                cov_matrix,
                list(sensitivities.keys()),
                portfolio_std,
            )

        # Calculate component VaR if enabled
        if self.config.calculate_component_var:
            result.component_var = self._calculate_component_var(
                portfolio, risk_factors_df, sensitivities, cov_matrix
            )

        # Calculate marginal VaR if enabled
        if self.config.calculate_marginal_var:
            result.marginal_var = self._calculate_marginal_var(
                portfolio, risk_factors_df, sensitivities, cov_matrix
            )

        # Calculate Stressed VaR if enabled
        if self.config.calculate_stressed_var:
            # For parametric VaR, we use stressed volatility multipliers
            # Simplified: apply stress factor to covariance matrix
            stress_multiplier = 1.5  # 50% increase in volatility as stress

            stressed_cov_matrix = cov_matrix * stress_multiplier

            # Recalculate portfolio variance with stressed covariance
            stressed_variance = (
                sensitivity_vector @ stressed_cov_matrix @ sensitivity_vector
            )
            stressed_std = np.sqrt(stressed_variance)

            if self.config.holding_period > 1:
                if self.config.scaling_method == "sqrt_t":
                    stressed_std *= np.sqrt(self.config.holding_period)

            z_score = stats.norm.ppf(self.config.confidence_level)
            stressed_var = z_score * stressed_std
            stressed_cvar = (
                stressed_std
                * stats.norm.pdf(z_score)
                / (1 - self.config.confidence_level)
            )

            # Store stressed VaR
            result.stressed_var = abs(stressed_var)
            result.stressed_cvar = abs(stressed_cvar)

            # Store stressed period (using entire lookback as stressed)
            result.stressed_period = {
                "start_date": risk_factors_df.index.min(),
                "end_date": risk_factors_df.index.max(),
            }

        # Calculate Incremental VaR if enabled
        if self.config.calculate_incremental_var:
            result.incremental_var = self._calculate_incremental_var(
                portfolio, risk_factors_df, cov_matrix, portfolio_std
            )

        return result

    def _calculate_fi_var(
        self,
        portfolio: FIPortfolio,
        historical_data: Union[any, pd.DataFrame],
    ) -> VaRResult:
        """Calculate parametric VaR for FI portfolio."""
        if isinstance(historical_data, pd.DataFrame):
            risk_factors_df = self._extract_risk_factors_from_dataframe(
                historical_data, is_equity=False
            )
        else:
            risk_factors_df = self._extract_risk_factors_from_market_data(
                historical_data, is_equity=False
            )

        if len(risk_factors_df) < self.config.lookback_days:
            raise MarketDataError(
                f"Insufficient historical data: {len(risk_factors_df)} days "
                f"available, {self.config.lookback_days} required"
            )

        risk_factors_df = risk_factors_df.tail(self.config.lookback_days)

        sensitivities = self._compute_fi_sensitivities(portfolio)

        cov_matrix = risk_factors_df.cov().values

        sensitivity_vector = np.array(list(sensitivities.values()))

        portfolio_variance = sensitivity_vector @ cov_matrix @ sensitivity_vector
        portfolio_std = np.sqrt(portfolio_variance)

        if self.config.holding_period > 1:
            if self.config.scaling_method == "sqrt_t":
                portfolio_std *= np.sqrt(self.config.holding_period)

        z_score = stats.norm.ppf(self.config.confidence_level)
        var = z_score * portfolio_std

        cvar = (
            portfolio_std * stats.norm.pdf(z_score) / (1 - self.config.confidence_level)
        )

        portfolio_value = portfolio.get_portfolio_value()

        result = VaRResult(
            var=abs(var),
            cvar=abs(cvar),
            confidence_level=self.config.confidence_level,
            holding_period=self.config.holding_period,
            method=VaRMethod.PARAMETRIC,
            portfolio_value=portfolio_value,
            var_as_pct=abs(var) / portfolio_value if portfolio_value != 0 else 0.0,
        )

        if self.config.calculate_factor_var:
            result.factor_var = self._compute_factor_var(
                sensitivity_vector,
                cov_matrix,
                list(sensitivities.keys()),
                portfolio_std,
            )

        # Calculate component VaR if enabled
        if self.config.calculate_component_var:
            result.component_var = self._calculate_component_var(
                portfolio, risk_factors_df, sensitivities, cov_matrix
            )

        # Calculate marginal VaR if enabled
        if self.config.calculate_marginal_var:
            result.marginal_var = self._calculate_marginal_var(
                portfolio, risk_factors_df, sensitivities, cov_matrix
            )

        # Calculate Stressed VaR if enabled
        if self.config.calculate_stressed_var:
            # For parametric VaR, we use stressed volatility multipliers
            # Simplified: apply stress factor to covariance matrix
            stress_multiplier = 1.5  # 50% increase in volatility as stress

            stressed_cov_matrix = cov_matrix * stress_multiplier

            # Recalculate portfolio variance with stressed covariance
            stressed_variance = (
                sensitivity_vector @ stressed_cov_matrix @ sensitivity_vector
            )
            stressed_std = np.sqrt(stressed_variance)

            if self.config.holding_period > 1:
                if self.config.scaling_method == "sqrt_t":
                    stressed_std *= np.sqrt(self.config.holding_period)

            z_score = stats.norm.ppf(self.config.confidence_level)
            stressed_var = z_score * stressed_std
            stressed_cvar = (
                stressed_std
                * stats.norm.pdf(z_score)
                / (1 - self.config.confidence_level)
            )

            # Store stressed VaR
            result.stressed_var = abs(stressed_var)
            result.stressed_cvar = abs(stressed_cvar)

            # Store stressed period (using entire lookback as stressed)
            result.stressed_period = {
                "start_date": risk_factors_df.index.min(),
                "end_date": risk_factors_df.index.max(),
            }

        # Calculate Incremental VaR if enabled
        if self.config.calculate_incremental_var:
            result.incremental_var = self._calculate_incremental_var(
                portfolio, risk_factors_df, cov_matrix, portfolio_std
            )

        return result

    def _extract_risk_factors_from_dataframe(
        self, df: pd.DataFrame, is_equity: bool
    ) -> pd.DataFrame:
        """Extract risk factors from DataFrame."""
        if is_equity:
            factors_config = self.config.equity_factors or EquityRiskFactorConfig()

            risk_factors = {}

            if factors_config.include_spot:
                factor = SpotReturnFactor()
                risk_factors["spot_return"] = factor.extract_from_dataframe(df)

            if factors_config.include_vol:
                factor = VolChangeFactor()
                risk_factors["vol_change"] = factor.extract_from_dataframe(df)

            if factors_config.include_rate:
                factor = RateShiftFactor()
                risk_factors["rate_shift"] = factor.extract_from_dataframe(df)

            if factors_config.include_div_yield:
                factor = DivYieldShiftFactor()
                risk_factors["div_yield_shift"] = factor.extract_from_dataframe(df)

            return pd.DataFrame(risk_factors)
        else:
            # Fixed Income risk factors
            from quantark.var.config import FIRiskFactorConfig
            from quantark.var.risk_factors.fi_factors import (
                ParallelShiftFactor,
                KeyRateShiftFactor,
            )

            factors_config = self.config.fi_factors or FIRiskFactorConfig()

            risk_factors = {}

            # Parallel shift factor (most important for FI)
            if factors_config.include_parallel_shift:
                factor = ParallelShiftFactor()
                try:
                    risk_factors["parallel_shift"] = factor.extract_from_dataframe(df)
                except ValueError as e:
                    # If no parallel_shift column, try rate column
                    if "rate" in df.columns:
                        risk_factors["parallel_shift"] = df["rate"].diff().dropna()

            # Key rate factors (optional, more sophisticated)
            if factors_config.include_key_rates:
                key_rate_factor = KeyRateShiftFactor(
                    tenors=factors_config.key_rate_tenors
                )
                try:
                    key_rate_shifts = key_rate_factor.extract_from_dataframe(df)
                    # Add key rate shifts to risk factors
                    for col in key_rate_shifts.columns:
                        risk_factors[col] = key_rate_shifts[col]
                except ValueError as e:
                    # If key rate columns don't exist, skip
                    pass

            # Return risk factors DataFrame
            if not risk_factors:
                raise ValueError(
                    "No valid FI risk factors found. Check that DataFrame contains "
                    "required columns: 'parallel_shift' or 'rate' for parallel shifts, "
                    "and 'rate_Xy' for key rate shifts."
                )

            return pd.DataFrame(risk_factors)

    def _extract_risk_factors_from_market_data(
        self, market_data: any, is_equity: bool
    ) -> pd.DataFrame:
        """
        Extract risk factors from MarketDataSet.

        Converts MarketDataSet containing spot, vol, rate, and dividend time series
        into a DataFrame with returns and changes suitable for covariance calculation
        in parametric VaR.

        Args:
            market_data: MarketDataSet with historical time series
            is_equity: Whether this is for equity portfolio (vs fixed income)

        Returns:
            DataFrame with columns: spot_return, vol_change, rate_shift, div_yield_shift

        Raises:
            MarketDataError: If market data is invalid or insufficient
        """
        from quantark.util.exceptions import MarketDataError

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

        # Create risk factors DataFrame
        risk_factors = pd.DataFrame(index=common_index)
        risk_factors["spot_return"] = spot_returns[common_index]
        risk_factors["vol_change"] = vol_changes[common_index]
        risk_factors["rate_shift"] = rate_shifts[common_index]

        if len(div_yield_shifts) > 0:
            risk_factors["div_yield_shift"] = div_yield_shifts[common_index]
        else:
            risk_factors["div_yield_shift"] = 0.0

        # For parametric VaR, we need sufficient data points
        # Filter to only include lookback_days
        risk_factors = risk_factors.tail(self.config.lookback_days)

        # Drop any remaining NaN values
        risk_factors = risk_factors.dropna()

        if len(risk_factors) == 0:
            raise MarketDataError(
                "No valid risk factors after processing. Check data quality."
            )

        if len(risk_factors) < 30:
            raise MarketDataError(
                f"Insufficient data for parametric VaR: {len(risk_factors)} days, "
                f"minimum 30 days required for stable covariance estimation"
            )

        return risk_factors

    def _compute_equity_sensitivities(
        self, portfolio: EquityPortfolio
    ) -> Dict[str, float]:
        """Compute portfolio-level sensitivities for equity.

        Greeks are sourced from each position's own ``get_risk_measures`` (already
        quantity-scaled), keeping the engine agnostic to how a position prices:
        ``EquityPosition`` returns analytical option greeks, ``EquitySwapPosition``
        returns the TRS delta-one / funding-rate sensitivities. The dollar
        spot-return sensitivity multiplies the scaled delta by the underlying spot.
        """
        from quantark.portfolio.equity.swap_position import EquitySwapPosition

        factors_config = self.config.equity_factors or EquityRiskFactorConfig()

        sensitivities: Dict[str, float] = {}
        per_position_greeks = {
            position.position_id: position.get_risk_measures(
                portfolio.pricing_environments[position.underlying]
            )
            for position in portfolio.positions.values()
        }

        if factors_config.include_spot:
            total_delta = 0.0
            for position in portfolio.positions.values():
                pricing_env = portfolio.pricing_environments[position.underlying]
                greeks = per_position_greeks[position.position_id]
                total_delta += greeks["delta"] * pricing_env.spot
            sensitivities["spot_return"] = total_delta

        if factors_config.include_vol:
            sensitivities["vol_change"] = sum(
                per_position_greeks[p.position_id].get("vega", 0.0)
                for p in portfolio.positions.values()
            )

        if factors_config.include_rate:
            sensitivities["rate_shift"] = sum(
                per_position_greeks[p.position_id].get("rho", 0.0)
                for p in portfolio.positions.values()
            )

        if factors_config.include_div_yield:
            total_psi = 0.0
            for position in portfolio.positions.values():
                # A realized-cashflow TRS has no risk-neutral dividend-yield
                # sensitivity in this model; only payoff-on-spot products do.
                if isinstance(position, EquitySwapPosition):
                    continue
                pricing_env = portfolio.pricing_environments[position.underlying]
                psi = self._calculate_div_yield_sensitivity(
                    position.product, pricing_env
                )
                total_psi += psi * position.quantity
            sensitivities["div_yield_shift"] = total_psi

        return sensitivities

    def _calculate_div_yield_sensitivity(
        self, product: any, pricing_env: any, bump_size: float = 0.0001
    ) -> float:
        """
        Calculate dividend yield sensitivity (psi) using finite difference.

        Args:
            product: Option product
            pricing_env: Pricing environment
            bump_size: Bump size for dividend yield (default: 1 bp = 0.0001)

        Returns:
            Psi: $ change per 1bp change in dividend yield
        """
        from quantark.asset.equity.engine.analytical import BlackScholesEngine
        from quantark.param.div_yield import ContinuousDividendYield
        from quantark.priceenv.pricing_environment import PricingEnvironment

        engine = BlackScholesEngine()
        base_price = engine.price(product, pricing_env)

        # Bump dividend yield up
        original_div = pricing_env.div_yield
        T = product.get_maturity(pricing_env)
        base_div_yield = pricing_env.get_div_yield(T)

        bumped_div_yield = ContinuousDividendYield(base_div_yield + bump_size)
        bumped_env = PricingEnvironment(
            rate_curve=pricing_env.rate_curve,
            valuation_date=pricing_env.valuation_date,
            spot_quote=pricing_env.spot_quote,
            vol_surface=pricing_env.vol_surface,
            div_yield=bumped_div_yield,
        )

        bumped_price = engine.price(product, bumped_env)

        # Sensitivity per bump_size change
        psi = (bumped_price - base_price) / bump_size

        return psi

    def _compute_fi_sensitivities(self, portfolio: FIPortfolio) -> Dict[str, float]:
        """Compute portfolio-level sensitivities for FI."""
        sensitivities = {}

        total_dv01 = portfolio.get_portfolio_dv01()
        sensitivities["parallel_shift"] = total_dv01

        return sensitivities

    def _compute_factor_var(
        self,
        sensitivity_vector: np.ndarray,
        cov_matrix: np.ndarray,
        factor_names: List[str],
        portfolio_std: float,
    ) -> Dict[str, float]:
        """Compute VaR attribution by risk factor using correlation-based approach."""
        # Factor VaR = |Correlation(Factor, Portfolio)| × Portfolio VaR
        # This uses the actual covariance to calculate meaningful factor contributions

        z_score = stats.norm.ppf(self.config.confidence_level)
        portfolio_var_result = z_score * portfolio_std

        factor_var = {}

        # Calculate portfolio P&L as weighted sum of factor returns
        # For each factor, calculate its standalone variance and correlation with portfolio
        for i, factor_name in enumerate(factor_names):
            # Factor variance
            factor_var_i = cov_matrix[i, i]

            # Factor's standalone VaR (in return units)
            factor_std = np.sqrt(factor_var_i)
            factor_var_return = z_score * factor_std

            # Portfolio return from this factor only: sensitivity × factor_return
            # Correlation between this factor and portfolio:
            # corr = Cov(factor_return, portfolio_return) / (std(factor) × std(portfolio))
            # Cov(factor_return, portfolio_return) = sensitivity_factor × factor_var
            # std(portfolio) = portfolio_std (already calculated)

            if factor_std > 0 and portfolio_std > 0 and sensitivity_vector[i] != 0:
                # Calculate correlation between factor and portfolio
                covariance_factor_portfolio = sensitivity_vector[i] * factor_var_i
                correlation = covariance_factor_portfolio / (factor_std * portfolio_std)
                correlation = abs(correlation)  # VaR is always positive
            else:
                correlation = 0

            # Factor VaR contribution
            factor_var_result = correlation * portfolio_var_result
            factor_var[factor_name] = factor_var_result

        return factor_var

    def _calculate_component_var(
        self,
        portfolio: Union[EquityPortfolio, FIPortfolio],
        risk_factors_df: pd.DataFrame,
        factor_sensitivities: Dict[str, float],
        cov_matrix: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate component VaR for each position using Euler decomposition.

        Uses the formula: Component VaR_i = Cov(P&L_i, P&L_portfolio) / Var(P&L_portfolio) * Portfolio VaR

        This implementation converts dollar sensitivities to return sensitivities
        to match the covariance matrix units (returns).

        Args:
            portfolio: Portfolio to analyze
            risk_factors_df: Historical risk factor data
            factor_sensitivities: Portfolio-level factor sensitivities (dollar units)
            cov_matrix: Covariance matrix of risk factors (return units)

        Returns:
            Dictionary mapping position ID to component VaR
        """
        from scipy import stats

        # Calculate portfolio value to convert dollar sensitivities to return sensitivities
        portfolio_value = portfolio.get_portfolio_value()

        # Convert dollar sensitivities to return sensitivities by dividing by portfolio value
        factor_names = list(factor_sensitivities.keys())
        return_sensitivities = {
            factor: factor_sensitivities[factor] / portfolio_value
            for factor in factor_names
        }

        # Calculate portfolio P&L as return × portfolio value
        factor_returns = risk_factors_df[factor_names].values
        return_vector = np.array(
            [return_sensitivities[factor] for factor in factor_names]
        )
        portfolio_return = factor_returns @ return_vector
        portfolio_pnl = portfolio_return * portfolio_value

        # Calculate position P&L for each scenario
        position_pnls = {}
        if isinstance(portfolio, EquityPortfolio):
            for pos_id, position in portfolio.positions.items():
                pricing_env = portfolio.pricing_environments[position.underlying]
                # Dollar P&L per scenario = spot_return × (delta_$ × spot), the
                # quantity-scaled dollar delta sensitivity — consistent with
                # _compute_equity_sensitivities and the dollar-sensitivities loop
                # below, and dimensionally correct for both options and TRS. Uses
                # the scaled delta directly, so no division by quantity is needed.
                scaled = position.get_risk_measures(pricing_env)
                pos_pnl = factor_returns[:, 0] * scaled["delta"] * pricing_env.spot
                position_pnls[pos_id] = pos_pnl

        # Calculate portfolio variance and VaR
        portfolio_var = np.var(portfolio_pnl, ddof=1)
        portfolio_std = np.sqrt(portfolio_var)
        z_score = stats.norm.ppf(self.config.confidence_level)
        portfolio_var_result = z_score * portfolio_std

        # Calculate Component VaR using Euler decomposition
        # For parametric VaR: Component VaR_i = (sensitivity_i / total_sensitivity) × Portfolio VaR
        # where sensitivity_i is the dollar sensitivity (delta × spot × quantity)
        component_var = {}
        if portfolio_var > 0:
            # Get total portfolio sensitivity (dollar units)
            total_sensitivity = sum(
                [factor_sensitivities[factor] for factor in factor_names]
            )

            # Calculate position-level dollar sensitivities
            position_dollar_sensitivities = {}
            if isinstance(portfolio, EquityPortfolio):
                for pos_id, position in portfolio.positions.items():
                    pricing_env = portfolio.pricing_environments[position.underlying]
                    # Quantity-scaled delta × spot = delta × spot × quantity.
                    scaled = position.get_risk_measures(pricing_env)
                    pos_dollar_sensitivity = scaled["delta"] * pricing_env.spot
                    position_dollar_sensitivities[pos_id] = pos_dollar_sensitivity

            # Calculate Component VaR
            if abs(total_sensitivity) > 0:
                for pos_id in position_pnls.keys():
                    pos_sensitivity = position_dollar_sensitivities.get(pos_id, 0.0)
                    # Component VaR = (sensitivity_i / total_sensitivity) × Portfolio VaR
                    # This gives signed Component VaR (can be negative for short positions)
                    component_var_result = (
                        pos_sensitivity / total_sensitivity
                    ) * portfolio_var_result
                    component_var[pos_id] = component_var_result
            else:
                # No sensitivity, distribute equally
                num_positions = len(position_pnls)
                for pos_id in position_pnls.keys():
                    component_var[pos_id] = portfolio_var_result / num_positions
        else:
            # No variance, assign zero
            for pos_id in position_pnls.keys():
                component_var[pos_id] = 0.0

        return component_var

    def _calculate_marginal_var(
        self,
        portfolio: Union[EquityPortfolio, FIPortfolio],
        risk_factors_df: pd.DataFrame,
        sensitivities: Dict[str, float],
        cov_matrix: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate marginal VaR for each position.

        Args:
            portfolio: Portfolio to analyze
            risk_factors_df: Historical risk factor data
            sensitivities: Portfolio-level sensitivities
            cov_matrix: Covariance matrix of risk factors

        Returns:
            Dictionary mapping position ID to marginal VaR
        """
        # Calculate portfolio volatility
        sensitivity_vector = np.array(list(sensitivities.values()))
        portfolio_variance = sensitivity_vector @ cov_matrix @ sensitivity_vector
        portfolio_std = np.sqrt(portfolio_variance)

        # Calculate marginal VaR for each position
        marginal_var = {}
        for pos_id, position in portfolio.positions.items():
            # A flat position contributes no marginal risk (and must not divide
            # by a zero quantity when recovering the per-unit delta below).
            if is_zero(position.quantity):
                marginal_var[pos_id] = 0.0
                continue

            pricing_env = portfolio.pricing_environments[position.underlying]

            # Per-unit delta × spot, recovered from the quantity-scaled risk
            # measures so options and swaps are handled uniformly.
            scaled = position.get_risk_measures(pricing_env)
            pos_sensitivity = (scaled["delta"] / position.quantity) * pricing_env.spot

            # Marginal VaR ≈ Component VaR for parametric method
            # Use the same formula as Component VaR
            total_sensitivity = sum(sensitivities.values())
            if abs(total_sensitivity) > 0:
                marg_var = abs(pos_sensitivity / total_sensitivity) * (
                    portfolio_std * stats.norm.ppf(self.config.confidence_level)
                )
            else:
                marg_var = 0.0

            marginal_var[pos_id] = marg_var

        return marginal_var

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
        risk_factors_df: pd.DataFrame,
        cov_matrix: np.ndarray,
        portfolio_std: float,
    ) -> Dict[str, float]:
        """
        Calculate Incremental VaR for each position using parametric approach.

        For parametric VaR, Incremental VaR is calculated using the formula:
        IVaR_i = (∂VaR/∂x_i) = (cov_matrix @ sensitivity_vector / portfolio_std)[i] * z_score

        This is derived from the Euler decomposition of the risk measure.

        Args:
            portfolio: Portfolio to analyze
            risk_factors_df: Historical risk factor data
            cov_matrix: Covariance matrix of risk factors
            portfolio_std: Portfolio standard deviation

        Returns:
            Dictionary mapping position ID to Incremental VaR
        """
        position_ivar = {}

        # Calculate marginal contributions for each position
        z_score = stats.norm.ppf(self.config.confidence_level)

        # Calculate full portfolio sensitivity vector
        factor_sensitivities = self._compute_equity_sensitivities(portfolio)
        sensitivity_vector = np.array(list(factor_sensitivities.values()))

        # Pad sensitivity vector to match covariance matrix dimensions
        if len(sensitivity_vector) < cov_matrix.shape[0]:
            # Pad with zeros for uncalculated factors
            padded_vector = np.zeros(cov_matrix.shape[0])
            padded_vector[: len(sensitivity_vector)] = sensitivity_vector
            sensitivity_vector = padded_vector

        # Calculate marginal VaR components
        if portfolio_std > 0:
            marginal_contrib = (
                (cov_matrix @ sensitivity_vector) / portfolio_std * z_score
            )
        else:
            marginal_contrib = np.zeros_like(sensitivity_vector)

        # For parametric VaR, Incremental VaR ≈ Marginal VaR
        # Distribute based on position weights
        total_position_value = sum(
            [
                abs(
                    position.get_portfolio_value()
                    if hasattr(position, "get_portfolio_value")
                    else position.quantity * 100
                )  # Fallback to quantity × spot
                for position in portfolio.positions.values()
            ]
        )

        for i, pos_id in enumerate(portfolio.positions.keys()):
            position = portfolio.positions[pos_id]
            pos_value = abs(
                position.get_portfolio_value()
                if hasattr(position, "get_portfolio_value")
                else position.quantity * 100
            )
            pos_weight = (
                pos_value / total_position_value if total_position_value > 0 else 0
            )

            # Allocate portfolio-level marginal contribution to this position
            position_ivar[pos_id] = abs(sum(marginal_contrib) * pos_weight)

        return position_ivar

    def calculate_incremental_var(
        self,
        portfolio: Union[EquityPortfolio, FIPortfolio],
        historical_data: Union[any, pd.DataFrame],
    ) -> "IncrementalVaRResult":
        """
        Calculate Incremental VaR for the portfolio using parametric approach.

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

        # Get risk factors
        if isinstance(historical_data, pd.DataFrame):
            if isinstance(portfolio, EquityPortfolio):
                risk_factors_df = self._extract_risk_factors_from_dataframe(
                    historical_data, is_equity=True
                )
            else:
                risk_factors_df = self._extract_risk_factors_from_dataframe(
                    historical_data, is_equity=False
                )
        else:
            if isinstance(portfolio, EquityPortfolio):
                risk_factors_df = self._extract_risk_factors_from_market_data(
                    historical_data, is_equity=True
                )
            else:
                risk_factors_df = self._extract_risk_factors_from_market_data(
                    historical_data, is_equity=False
                )

        risk_factors_df = risk_factors_df.tail(self.config.lookback_days)
        cov_matrix = risk_factors_df.cov().values

        # Calculate portfolio standard deviation
        if isinstance(portfolio, EquityPortfolio):
            sensitivities = self._compute_equity_sensitivities(portfolio)
        else:
            sensitivities = self._compute_fi_sensitivities(portfolio)

        sensitivity_vector = np.array(list(sensitivities.values()))
        portfolio_variance = sensitivity_vector @ cov_matrix @ sensitivity_vector
        portfolio_std = np.sqrt(portfolio_variance)

        # Calculate Incremental VaR for each position
        position_ivar = self._calculate_incremental_var(
            portfolio, risk_factors_df, cov_matrix, portfolio_std
        )

        # Calculate VaR without each position for reporting
        var_without_dict = {}
        for pos_id in portfolio.positions.keys():
            # Create portfolio without this position
            portfolio_without = self._create_portfolio_without_position(
                portfolio, pos_id
            )

            if len(portfolio_without.positions) > 0:
                # Calculate VaR without this position
                if isinstance(portfolio_without, EquityPortfolio):
                    sensitivities_without = self._compute_equity_sensitivities(
                        portfolio_without
                    )
                else:
                    sensitivities_without = self._compute_fi_sensitivities(
                        portfolio_without
                    )

                if len(sensitivities_without) > 0:
                    sensitivity_vector_without = np.array(
                        list(sensitivities_without.values())
                    )
                    variance_without = (
                        sensitivity_vector_without
                        @ cov_matrix
                        @ sensitivity_vector_without
                    )
                    std_without = np.sqrt(variance_without)

                    if self.config.holding_period > 1:
                        if self.config.scaling_method == "sqrt_t":
                            std_without *= np.sqrt(self.config.holding_period)

                    z_score = stats.norm.ppf(self.config.confidence_level)
                    var_without = z_score * std_without
                    var_without_dict[pos_id] = abs(var_without)

        # Calculate diversification benefit
        total_individual_var = sum(position_ivar.values())
        diversification_benefit = total_individual_var - full_var

        # Create result
        result = IncrementalVaRResult(
            portfolio_var=full_var,
            position_ivari=position_ivar,
            diversification_benefit=diversification_benefit,
            portfolio_var_without_position=var_without_dict,
            ivari_method="Parametric",
            config=self.config.__dict__,
        )

        return result

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

        new_portfolio = copy.deepcopy(portfolio)
        if exclude_pos_id in new_portfolio.positions:
            del new_portfolio.positions[exclude_pos_id]

        return new_portfolio
