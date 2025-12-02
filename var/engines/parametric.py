"""
Parametric VaR engine using variance-covariance approach.
"""

import time
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from portfolio.equity.portfolio import EquityPortfolio
from portfolio.fi.portfolio import FIPortfolio
from util.exceptions import ValidationError, MarketDataError, NumericalError
from var.base import VaRResult
from var.config import VaRConfig, VaRMethod, EquityRiskFactorConfig
from var.risk_factors import (
    SpotReturnFactor,
    VolChangeFactor,
    RateShiftFactor,
    DivYieldShiftFactor,
)


class ParametricVaREngine:
    """
    Parametric VaR engine using variance-covariance approach.
    
    Computes VaR using portfolio sensitivities (Greeks for equity, DV01 for FI)
    and historical covariance matrix of risk factors.
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
            raise ValidationError(f"Unsupported portfolio type: {type(portfolio).__name__}")
        
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
        
        cvar = portfolio_std * stats.norm.pdf(z_score) / (1 - self.config.confidence_level)
        
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
                sensitivity_vector, cov_matrix, list(sensitivities.keys()), portfolio_std
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
        
        cvar = portfolio_std * stats.norm.pdf(z_score) / (1 - self.config.confidence_level)
        
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
                sensitivity_vector, cov_matrix, list(sensitivities.keys()), portfolio_std
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
                risk_factors['spot_return'] = factor.extract_from_dataframe(df)
            
            if factors_config.include_vol:
                factor = VolChangeFactor()
                risk_factors['vol_change'] = factor.extract_from_dataframe(df)
            
            if factors_config.include_rate:
                factor = RateShiftFactor()
                risk_factors['rate_shift'] = factor.extract_from_dataframe(df)
            
            if factors_config.include_div_yield:
                factor = DivYieldShiftFactor()
                risk_factors['div_yield_shift'] = factor.extract_from_dataframe(df)
            
            return pd.DataFrame(risk_factors)
        else:
            raise NotImplementedError("FI risk factors from DataFrame not yet implemented")
    
    def _extract_risk_factors_from_market_data(
        self, market_data: any, is_equity: bool
    ) -> pd.DataFrame:
        """Extract risk factors from MarketDataSet."""
        raise NotImplementedError("MarketDataSet extraction not yet implemented")
    
    def _compute_equity_sensitivities(self, portfolio: EquityPortfolio) -> Dict[str, float]:
        """Compute portfolio-level sensitivities for equity."""
        from asset.equity.riskmeasures import GreeksCalculator
        
        calculator = GreeksCalculator()
        factors_config = self.config.equity_factors or EquityRiskFactorConfig()
        
        sensitivities = {}
        
        if factors_config.include_spot:
            total_delta = 0.0
            for position in portfolio.positions.values():
                pricing_env = portfolio.pricing_environments[position.underlying]
                greeks = calculator.calculate_analytical_greeks(
                    position.product, pricing_env
                )
                total_delta += greeks['delta'] * position.quantity * pricing_env.spot
            sensitivities['spot_return'] = total_delta
        
        if factors_config.include_vol:
            total_vega = 0.0
            for position in portfolio.positions.values():
                pricing_env = portfolio.pricing_environments[position.underlying]
                greeks = calculator.calculate_analytical_greeks(
                    position.product, pricing_env
                )
                total_vega += greeks['vega'] * position.quantity
            sensitivities['vol_change'] = total_vega
        
        if factors_config.include_rate:
            total_rho = 0.0
            for position in portfolio.positions.values():
                pricing_env = portfolio.pricing_environments[position.underlying]
                greeks = calculator.calculate_analytical_greeks(
                    position.product, pricing_env
                )
                total_rho += greeks['rho'] * position.quantity
            sensitivities['rate_shift'] = total_rho
        
        if factors_config.include_div_yield:
            sensitivities['div_yield_shift'] = 0.0
        
        return sensitivities
    
    def _compute_fi_sensitivities(self, portfolio: FIPortfolio) -> Dict[str, float]:
        """Compute portfolio-level sensitivities for FI."""
        sensitivities = {}
        
        total_dv01 = portfolio.get_portfolio_dv01()
        sensitivities['parallel_shift'] = total_dv01
        
        return sensitivities
    
    def _compute_factor_var(
        self,
        sensitivity_vector: np.ndarray,
        cov_matrix: np.ndarray,
        factor_names: List[str],
        portfolio_std: float,
    ) -> Dict[str, float]:
        """Compute VaR attribution by risk factor."""
        z_score = stats.norm.ppf(self.config.confidence_level)
        
        marginal_var = (cov_matrix @ sensitivity_vector) / portfolio_std if portfolio_std > 0 else np.zeros_like(sensitivity_vector)
        
        component_var = sensitivity_vector * marginal_var * z_score
        
        factor_var = {
            factor_names[i]: abs(component_var[i])
            for i in range(len(factor_names))
        }
        
        return factor_var
