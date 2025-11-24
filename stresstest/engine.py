"""
Core stress test execution engine.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import time
from copy import deepcopy

from portfolio import Portfolio
from asset.equity.riskmeasures import GreeksCalculator
from stresstest.config import StressTestConfig
from stresstest.scenario.scenario import Scenario
from stresstest.stress.stress_applicator import StressApplicator
from stresstest.results.stress_results import StressTestResults, ScenarioResult
from util.exceptions import ValidationError


class StressTestEngine:
    """
    Engine for executing stress tests on portfolios.
    
    This is the main entry point for running stress test scenarios. It handles:
    - Evaluating portfolio under multiple scenarios
    - Calculating P&L and Greeks for each scenario
    - Aggregating results
    - Supporting future dynamic scenario analysis
    
    Example:
        >>> config = StressTestConfig(calculate_greeks=True)
        >>> engine = StressTestEngine(config)
        >>> results = engine.run_static_scenarios(portfolio, scenarios)
        >>> print(results.get_summary())
    """
    
    def __init__(self, config: Optional[StressTestConfig] = None):
        """
        Initialize stress test engine.
        
        Args:
            config: Configuration for stress test execution
        """
        self.config = config or StressTestConfig()
        self.greeks_calculator = GreeksCalculator() if self.config.calculate_greeks else None
    
    def run_static_scenarios(
        self,
        portfolio: Portfolio,
        scenarios: List[Scenario],
        baseline_label: str = "Current Market"
    ) -> StressTestResults:
        """
        Run static stress test scenarios on a portfolio.
        
        Evaluates the portfolio under each scenario and returns comprehensive results
        including P&L and risk metrics.
        
        Args:
            portfolio: Portfolio to stress test
            scenarios: List of scenarios to evaluate
            baseline_label: Label for baseline case
            
        Returns:
            StressTestResults containing all scenario evaluations
            
        Raises:
            ValidationError: If portfolio or scenarios are invalid
        """
        if not portfolio or len(portfolio) == 0:
            raise ValidationError("Portfolio must contain at least one position")
        
        if not scenarios:
            raise ValidationError("At least one scenario is required")
        
        start_time = time.time()
        
        # Evaluate baseline (unstressed) portfolio
        print(f"Evaluating baseline portfolio...")
        baseline_value = portfolio.get_portfolio_value()
        baseline_greeks = None
        
        if self.config.calculate_greeks and self.greeks_calculator:
            use_analytical = (self.config.greeks_method == 'analytical')
            baseline_greeks = portfolio.get_portfolio_greeks(
                self.greeks_calculator,
                use_analytical=use_analytical
            )
        
        # Run each scenario
        scenario_results = []
        for i, scenario in enumerate(scenarios, 1):
            print(f"Running scenario {i}/{len(scenarios)}: {scenario.name}")
            result = self._evaluate_scenario(portfolio, scenario, baseline_value)
            scenario_results.append(result)
        
        total_time = time.time() - start_time
        
        # Build results
        results = StressTestResults(
            baseline_value=baseline_value,
            baseline_greeks=baseline_greeks,
            scenario_results=scenario_results,
            total_execution_time=total_time,
            config_summary=self.config.get_summary(),
            metadata={'baseline_label': baseline_label}
        )
        
        print(f"\nStress test completed in {total_time:.2f} seconds")
        return results
    
    def run_dynamic_scenarios(
        self,
        portfolio: Portfolio,
        scenarios: List[Scenario],
        time_steps: List[datetime],
        hedge_strategy: Optional[Any] = None
    ) -> StressTestResults:
        """
        Run dynamic stress test scenarios with time dimension.
        
        This is a placeholder for future dynamic scenario analysis that will
        include time evolution and hedging strategies.
        
        Args:
            portfolio: Portfolio to stress test
            scenarios: List of scenarios to evaluate over time
            time_steps: Time points for scenario evolution
            hedge_strategy: Optional hedging strategy to apply
            
        Returns:
            StressTestResults with time-series results
            
        Raises:
            NotImplementedError: This feature is not yet implemented
        """
        raise NotImplementedError(
            "Dynamic scenario analysis is not yet implemented. "
            "This API is reserved for future development."
        )
    
    def _evaluate_scenario(
        self,
        portfolio: Portfolio,
        scenario: Scenario,
        baseline_value: float
    ) -> ScenarioResult:
        """
        Evaluate portfolio under a single scenario.
        
        Args:
            portfolio: Portfolio to evaluate
            scenario: Scenario to apply
            baseline_value: Baseline portfolio value for P&L calculation
            
        Returns:
            ScenarioResult with evaluation details
        """
        scenario_start = time.time()
        
        # Apply stresses to get modified pricing environments
        stressed_envs = StressApplicator.apply_scenario_to_portfolio(portfolio, scenario)
        
        # Create a temporary portfolio with stressed environments
        stressed_portfolio = self._create_stressed_portfolio(portfolio, stressed_envs)
        
        # Calculate stressed portfolio value
        stressed_value = stressed_portfolio.get_portfolio_value()
        portfolio_pnl = stressed_value - baseline_value
        portfolio_pnl_pct = (portfolio_pnl / baseline_value * 100) if baseline_value != 0 else 0.0
        
        # Calculate Greeks if requested
        greeks = None
        if self.config.calculate_greeks and self.greeks_calculator:
            use_analytical = (self.config.greeks_method == 'analytical')
            greeks = stressed_portfolio.get_portfolio_greeks(
                self.greeks_calculator,
                use_analytical=use_analytical
            )
        
        # Position-level results if requested
        position_results = []
        if self.config.save_detailed_results:
            position_results = self._calculate_position_results(
                portfolio,
                stressed_portfolio,
                scenario
            )
        
        # Underlying-level aggregation
        underlying_results = self._calculate_underlying_results(
            stressed_portfolio,
            scenario
        )
        
        execution_time = time.time() - scenario_start
        
        return ScenarioResult(
            scenario=scenario,
            portfolio_value=stressed_value,
            portfolio_pnl=portfolio_pnl,
            portfolio_pnl_pct=portfolio_pnl_pct,
            greeks=greeks,
            position_results=position_results,
            underlying_results=underlying_results,
            execution_time=execution_time,
        )
    
    def _create_stressed_portfolio(
        self,
        original_portfolio: Portfolio,
        stressed_envs: Dict[str, Any]
    ) -> Portfolio:
        """
        Create a portfolio copy with stressed pricing environments.
        
        Args:
            original_portfolio: Original portfolio
            stressed_envs: Stressed pricing environments
            
        Returns:
            New portfolio with stressed environments
        """
        # Create new portfolio with stressed environments
        stressed_portfolio = Portfolio(
            portfolio_name=original_portfolio.portfolio_name + "_stressed",
            pricing_environments=stressed_envs,
            creation_date=original_portfolio.creation_date,
        )
        
        # Copy positions (they reference the stressed environments)
        stressed_portfolio.positions = deepcopy(original_portfolio.positions)
        
        return stressed_portfolio
    
    def _calculate_position_results(
        self,
        original_portfolio: Portfolio,
        stressed_portfolio: Portfolio,
        scenario: Scenario
    ) -> List[Dict[str, Any]]:
        """
        Calculate position-level results.
        
        Args:
            original_portfolio: Original portfolio
            stressed_portfolio: Stressed portfolio
            scenario: Scenario being evaluated
            
        Returns:
            List of position results
        """
        results = []
        
        for position_id, position in stressed_portfolio.positions.items():
            original_position = original_portfolio.positions[position_id]
            
            original_env = original_portfolio.pricing_environments[position.underlying]
            stressed_env = stressed_portfolio.pricing_environments[position.underlying]
            
            original_value = original_position.get_market_value(original_env)
            stressed_value = position.get_market_value(stressed_env)
            
            position_pnl = stressed_value - original_value
            position_pnl_pct = (position_pnl / original_value * 100) if original_value != 0 else 0.0
            
            result = {
                'position_id': position_id,
                'underlying': position.underlying,
                'product_type': position.product.__class__.__name__,
                'quantity': position.quantity,
                'original_value': original_value,
                'stressed_value': stressed_value,
                'pnl': position_pnl,
                'pnl_pct': position_pnl_pct,
            }
            
            # Add Greeks if calculated
            if self.config.calculate_greeks and self.greeks_calculator:
                use_analytical = (self.config.greeks_method == 'analytical')
                position_greeks = position.get_greeks(
                    stressed_env,
                    self.greeks_calculator,
                    use_analytical=use_analytical
                )
                result['greeks'] = position_greeks
            
            results.append(result)
        
        return results
    
    def _calculate_underlying_results(
        self,
        stressed_portfolio: Portfolio,
        scenario: Scenario
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate results aggregated by underlying.
        
        Args:
            stressed_portfolio: Stressed portfolio
            scenario: Scenario being evaluated
            
        Returns:
            Dictionary of results by underlying
        """
        results = {}
        
        for underlying in stressed_portfolio.pricing_environments.keys():
            positions = stressed_portfolio.get_positions_by_underlying(underlying)
            
            if not positions:
                continue
            
            # Aggregate value
            total_value = sum(
                pos.get_market_value(stressed_portfolio.pricing_environments[underlying])
                for pos in positions
            )
            
            # Aggregate Greeks if calculated
            greeks = None
            if self.config.calculate_greeks and self.greeks_calculator:
                use_analytical = (self.config.greeks_method == 'analytical')
                greeks = stressed_portfolio.get_greeks_by_underlying(
                    underlying,
                    self.greeks_calculator,
                    use_analytical=use_analytical
                )
            
            results[underlying] = {
                'num_positions': len(positions),
                'total_value': total_value,
                'greeks': greeks,
            }
        
        return results
    
    def __repr__(self) -> str:
        return f"StressTestEngine(config={self.config})"

