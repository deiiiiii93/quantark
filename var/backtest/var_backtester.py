"""
VaR backtesting module for model validation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from portfolio.equity.portfolio import EquityPortfolio
from portfolio.fi.portfolio import FIPortfolio
from util.exceptions import ValidationError
from var.base import VaREngine


@dataclass
class VaRBacktestResult:
    """Results from VaR backtesting."""
    
    num_observations: int
    num_exceptions: int
    exception_rate: float
    expected_exceptions: float
    
    kupiec_pof_statistic: float
    kupiec_pof_pvalue: float
    kupiec_pof_pass: bool
    
    christoffersen_statistic: float
    christoffersen_pvalue: float
    christoffersen_pass: bool
    
    basel_zone: str
    
    exceptions_dates: List[datetime] = field(default_factory=list)
    exception_details: List[Dict] = field(default_factory=list)
    
    confidence_level: float = 0.99
    holding_period: int = 1
    
    backtest_start_date: Optional[datetime] = None
    backtest_end_date: Optional[datetime] = None


class VaRBacktester:
    """
    VaR backtesting engine for model validation.
    
    Performs statistical tests to validate VaR model accuracy:
    - Kupiec POF (Proportion of Failures) test
    - Christoffersen conditional coverage test
    - Basel traffic light zone classification
    """
    
    def __init__(self, confidence_level: float = 0.99, holding_period: int = 1):
        """
        Initialize VaR backtester.
        
        Args:
            confidence_level: VaR confidence level
            holding_period: VaR holding period in days
        """
        self.confidence_level = confidence_level
        self.holding_period = holding_period
    
    def run_backtest(
        self,
        portfolio: Union[EquityPortfolio, FIPortfolio],
        historical_data: pd.DataFrame,
        var_engine: VaREngine,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> VaRBacktestResult:
        """
        Run VaR backtest over historical period.
        
        Args:
            portfolio: Portfolio to backtest
            historical_data: Historical market data with actual P&L
            var_engine: VaR engine to use
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            VaRBacktestResult with test statistics
        """
        if len(portfolio.positions) == 0:
            raise ValidationError("Cannot backtest empty portfolio")
        
        if 'pnl' not in historical_data.columns:
            raise ValidationError("historical_data must contain 'pnl' column")
        
        if start_date is not None:
            historical_data = historical_data[historical_data.index >= start_date]
        if end_date is not None:
            historical_data = historical_data[historical_data.index <= end_date]
        
        var_predictions = []
        actual_pnl = []
        exception_flags = []
        exception_dates = []
        exception_details = []
        
        for date in historical_data.index:
            lookback_data = historical_data[historical_data.index < date]
            
            if len(lookback_data) < 30:
                continue
            
            var_result = var_engine.calculate_var(portfolio, lookback_data)
            predicted_var = var_result.var
            
            actual_loss = historical_data.loc[date, 'pnl']
            
            is_exception = actual_loss < -predicted_var
            
            var_predictions.append(predicted_var)
            actual_pnl.append(actual_loss)
            exception_flags.append(is_exception)
            
            if is_exception:
                exception_dates.append(date)
                exception_details.append({
                    'date': date,
                    'predicted_var': predicted_var,
                    'actual_pnl': actual_loss,
                    'breach_amount': abs(actual_loss) - predicted_var,
                })
        
        num_observations = len(exception_flags)
        num_exceptions = sum(exception_flags)
        exception_rate = num_exceptions / num_observations if num_observations > 0 else 0.0
        expected_exceptions = (1 - self.confidence_level) * num_observations
        
        kupiec_stat, kupiec_pval = self._kupiec_test(
            num_exceptions, num_observations, self.confidence_level
        )
        kupiec_pass = kupiec_pval > 0.05
        
        christ_stat, christ_pval = self._christoffersen_test(
            exception_flags, self.confidence_level
        )
        christ_pass = christ_pval > 0.05
        
        basel_zone = self._basel_traffic_light(
            num_exceptions, num_observations, self.confidence_level
        )
        
        return VaRBacktestResult(
            num_observations=num_observations,
            num_exceptions=num_exceptions,
            exception_rate=exception_rate,
            expected_exceptions=expected_exceptions,
            kupiec_pof_statistic=kupiec_stat,
            kupiec_pof_pvalue=kupiec_pval,
            kupiec_pof_pass=kupiec_pass,
            christoffersen_statistic=christ_stat,
            christoffersen_pvalue=christ_pval,
            christoffersen_pass=christ_pass,
            basel_zone=basel_zone,
            exceptions_dates=exception_dates,
            exception_details=exception_details,
            confidence_level=self.confidence_level,
            holding_period=self.holding_period,
            backtest_start_date=historical_data.index.min() if len(historical_data) > 0 else None,
            backtest_end_date=historical_data.index.max() if len(historical_data) > 0 else None,
        )
    
    def _kupiec_test(
        self, num_exceptions: int, num_observations: int, confidence_level: float
    ) -> tuple:
        """
        Kupiec Proportion of Failures (POF) test.
        
        Tests whether exception frequency matches expected rate.
        
        Args:
            num_exceptions: Number of VaR breaches
            num_observations: Total observations
            confidence_level: VaR confidence level
            
        Returns:
            (test_statistic, p_value)
        """
        if num_observations == 0:
            return 0.0, 1.0
        
        p_expected = 1 - confidence_level
        p_observed = num_exceptions / num_observations
        
        if num_exceptions == 0 or num_exceptions == num_observations:
            lr_stat = 0.0
        else:
            likelihood_null = (
                p_expected ** num_exceptions *
                (1 - p_expected) ** (num_observations - num_exceptions)
            )
            likelihood_alt = (
                p_observed ** num_exceptions *
                (1 - p_observed) ** (num_observations - num_exceptions)
            )
            
            lr_stat = -2 * np.log(likelihood_null / likelihood_alt)
        
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)
        
        return lr_stat, p_value
    
    def _christoffersen_test(
        self, exception_flags: List[bool], confidence_level: float
    ) -> tuple:
        """
        Christoffersen conditional coverage test.
        
        Tests both coverage (correct frequency) and independence (no clustering).
        
        Args:
            exception_flags: List of exception indicators
            confidence_level: VaR confidence level
            
        Returns:
            (test_statistic, p_value)
        """
        if len(exception_flags) < 2:
            return 0.0, 1.0
        
        n00 = n01 = n10 = n11 = 0
        
        for i in range(len(exception_flags) - 1):
            if not exception_flags[i] and not exception_flags[i + 1]:
                n00 += 1
            elif not exception_flags[i] and exception_flags[i + 1]:
                n01 += 1
            elif exception_flags[i] and not exception_flags[i + 1]:
                n10 += 1
            else:
                n11 += 1
        
        n0 = n00 + n01
        n1 = n10 + n11
        n = n0 + n1
        
        if n == 0:
            return 0.0, 1.0
        
        p = (n01 + n11) / n if n > 0 else 0.0
        p01 = n01 / n0 if n0 > 0 else 0.0
        p11 = n11 / n1 if n1 > 0 else 0.0
        
        if p == 0 or p == 1 or p01 == 0 or p01 == 1 or p11 == 0 or p11 == 1:
            lr_ind = 0.0
        else:
            likelihood_ind = (
                (1 - p) ** (n00 + n10) * p ** (n01 + n11)
            )
            likelihood_dep = (
                (1 - p01) ** n00 * p01 ** n01 *
                (1 - p11) ** n10 * p11 ** n11
            )
            
            lr_ind = -2 * np.log(likelihood_ind / likelihood_dep)
        
        num_exceptions = sum(exception_flags)
        kupiec_stat, _ = self._kupiec_test(num_exceptions, len(exception_flags), confidence_level)
        
        lr_cc = kupiec_stat + lr_ind
        
        p_value = 1 - stats.chi2.cdf(lr_cc, df=2)
        
        return lr_cc, p_value
    
    def _basel_traffic_light(
        self, num_exceptions: int, num_observations: int, confidence_level: float
    ) -> str:
        """
        Basel traffic light zone classification.
        
        For 99% VaR with 250 observations:
        - Green: 0-4 exceptions
        - Yellow: 5-9 exceptions
        - Red: 10+ exceptions
        
        Args:
            num_exceptions: Number of VaR breaches
            num_observations: Total observations
            confidence_level: VaR confidence level
            
        Returns:
            "green", "yellow", or "red"
        """
        if confidence_level == 0.99 and num_observations >= 250:
            if num_exceptions <= 4:
                return "green"
            elif num_exceptions <= 9:
                return "yellow"
            else:
                return "red"
        
        expected_exceptions = (1 - confidence_level) * num_observations
        
        if num_exceptions <= expected_exceptions * 1.5:
            return "green"
        elif num_exceptions <= expected_exceptions * 2.5:
            return "yellow"
        else:
            return "red"
