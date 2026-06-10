"""
Result aggregation and comparison utilities.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from quantark.stresstest.results.stress_results import StressTestResults, ScenarioResult


class ResultAggregator:
    """
    Aggregates and compares stress test results.
    
    Provides utilities for analyzing results across scenarios,
    comparing results, and identifying key risk factors.
    """
    
    @staticmethod
    def compare_scenarios(
        results: StressTestResults,
        metric: str = "portfolio_pnl"
    ) -> pd.DataFrame:
        """
        Compare scenarios by a specific metric.
        
        Args:
            results: Stress test results
            metric: Metric to compare ('portfolio_pnl', 'portfolio_pnl_pct', 'portfolio_value')
            
        Returns:
            DataFrame sorted by metric
        """
        rows = []
        for result in results.scenario_results:
            row = {
                'scenario': result.scenario.name,
                'portfolio_value': result.portfolio_value,
                'portfolio_pnl': result.portfolio_pnl,
                'portfolio_pnl_pct': result.portfolio_pnl_pct,
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        return df.sort_values(metric)
    
    @staticmethod
    def get_risk_summary(results: StressTestResults) -> Dict[str, Any]:
        """
        Generate risk summary statistics.
        
        Args:
            results: Stress test results
            
        Returns:
            Dictionary with risk metrics
        """
        pnls = [r.portfolio_pnl for r in results.scenario_results]
        pnl_pcts = [r.portfolio_pnl_pct for r in results.scenario_results]
        
        worst = results.get_worst_scenario()
        best = results.get_best_scenario()
        
        summary = {
            'baseline_value': results.baseline_value,
            'num_scenarios': len(results.scenario_results),
            'worst_pnl': worst.portfolio_pnl if worst else None,
            'worst_pnl_pct': worst.portfolio_pnl_pct if worst else None,
            'worst_scenario': worst.scenario.name if worst else None,
            'best_pnl': best.portfolio_pnl if best else None,
            'best_pnl_pct': best.portfolio_pnl_pct if best else None,
            'best_scenario': best.scenario.name if best else None,
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'median_pnl': np.median(pnls) if pnls else 0,
            'std_pnl': np.std(pnls) if pnls else 0,
            'avg_pnl_pct': np.mean(pnl_pcts) if pnl_pcts else 0,
            'median_pnl_pct': np.median(pnl_pcts) if pnl_pcts else 0,
            'max_drawdown_pct': min(pnl_pcts) if pnl_pcts else 0,
            'max_upside_pct': max(pnl_pcts) if pnl_pcts else 0,
        }
        
        return summary
    
    @staticmethod
    def get_greeks_comparison(results: StressTestResults) -> Optional[pd.DataFrame]:
        """
        Compare Greeks across scenarios.
        
        Args:
            results: Stress test results
            
        Returns:
            DataFrame with Greeks comparison, or None if Greeks not calculated
        """
        if not results.baseline_greeks:
            return None
        
        rows = []
        
        # Baseline
        baseline_row = {'scenario': 'Baseline'}
        for key, value in results.baseline_greeks.items():
            if key != 'market_value':
                baseline_row[key] = value
        rows.append(baseline_row)
        
        # Scenarios
        for result in results.scenario_results:
            if not result.greeks:
                continue
            row = {'scenario': result.scenario.name}
            for key, value in result.greeks.items():
                if key != 'market_value':
                    row[key] = value
            rows.append(row)
        
        if not rows:
            return None
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def get_pnl_distribution(
        results: StressTestResults,
        bins: int = 10
    ) -> Dict[str, Any]:
        """
        Get P&L distribution statistics.
        
        Args:
            results: Stress test results
            bins: Number of bins for histogram
            
        Returns:
            Dictionary with distribution statistics
        """
        pnls = [r.portfolio_pnl for r in results.scenario_results]
        pnl_pcts = [r.portfolio_pnl_pct for r in results.scenario_results]
        
        if not pnls:
            return {}
        
        # Calculate histogram
        hist, bin_edges = np.histogram(pnl_pcts, bins=bins)
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pnl_percentiles = {
            f"p{p}": np.percentile(pnl_pcts, p) for p in percentiles
        }
        
        return {
            'histogram': {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
            },
            'percentiles': pnl_percentiles,
            'mean': np.mean(pnl_pcts),
            'median': np.median(pnl_pcts),
            'std': np.std(pnl_pcts),
            'min': np.min(pnl_pcts),
            'max': np.max(pnl_pcts),
            'skewness': float(pd.Series(pnl_pcts).skew()),
            'kurtosis': float(pd.Series(pnl_pcts).kurtosis()),
        }
    
    @staticmethod
    def identify_key_risks(
        results: StressTestResults,
        threshold_pct: float = -5.0
    ) -> List[Dict[str, Any]]:
        """
        Identify scenarios with significant losses.
        
        Args:
            results: Stress test results
            threshold_pct: P&L percentage threshold for risk identification
            
        Returns:
            List of risky scenarios with details
        """
        risky_scenarios = []
        
        for result in results.scenario_results:
            if result.portfolio_pnl_pct < threshold_pct:
                risky_scenarios.append({
                    'scenario': result.scenario.name,
                    'description': result.scenario.description,
                    'pnl': result.portfolio_pnl,
                    'pnl_pct': result.portfolio_pnl_pct,
                    'num_stresses': len(result.scenario.stresses),
                    'severity': result.scenario.metadata.get('severity', 'unknown'),
                })
        
        # Sort by P&L (worst first)
        risky_scenarios.sort(key=lambda x: x['pnl'])
        
        return risky_scenarios
    
    @staticmethod
    def get_underlying_breakdown(
        results: StressTestResults,
        scenario_name: str
    ) -> Optional[pd.DataFrame]:
        """
        Get breakdown by underlying for a specific scenario.
        
        Args:
            results: Stress test results
            scenario_name: Name of scenario
            
        Returns:
            DataFrame with underlying breakdown, or None if not available
        """
        result = results.get_scenario_result(scenario_name)
        if not result or not result.underlying_results:
            return None
        
        rows = []
        for underlying, data in result.underlying_results.items():
            row = {
                'underlying': underlying,
                'num_positions': data['num_positions'],
                'total_value': data['total_value'],
            }
            
            if data.get('greeks'):
                for key, value in data['greeks'].items():
                    if key != 'market_value':
                        row[f'greek_{key}'] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    @staticmethod
    def calculate_var_cvar(
        results: StressTestResults,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR).
        
        Note: This is scenario-based VaR, not traditional parametric VaR.
        
        Args:
            results: Stress test results
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary with VaR and CVaR metrics
        """
        pnls = sorted([r.portfolio_pnl for r in results.scenario_results])
        
        if not pnls:
            return {'var': 0.0, 'cvar': 0.0}
        
        # VaR: worst loss at confidence level
        var_index = int(len(pnls) * (1 - confidence_level))
        var = pnls[var_index] if var_index < len(pnls) else pnls[0]
        
        # CVaR: average of losses worse than VaR
        tail_losses = pnls[:var_index + 1] if var_index > 0 else [pnls[0]]
        cvar = np.mean(tail_losses)
        
        return {
            'var': var,
            'cvar': cvar,
            'confidence_level': confidence_level,
            'num_scenarios': len(pnls),
            'num_tail_scenarios': len(tail_losses),
        }
    
    @staticmethod
    def export_comparison_table(
        results: StressTestResults,
        include_greeks: bool = True
    ) -> pd.DataFrame:
        """
        Export comprehensive comparison table.
        
        Args:
            results: Stress test results
            include_greeks: Whether to include Greeks
            
        Returns:
            Comprehensive comparison DataFrame
        """
        rows = []
        
        # Baseline
        baseline_row = {
            'scenario': 'Baseline',
            'description': 'Current market conditions',
            'portfolio_value': results.baseline_value,
            'pnl': 0.0,
            'pnl_pct': 0.0,
        }
        
        if include_greeks and results.baseline_greeks:
            for key, value in results.baseline_greeks.items():
                if key != 'market_value':
                    baseline_row[f'greek_{key}'] = value
        
        rows.append(baseline_row)
        
        # Scenarios
        for result in results.scenario_results:
            row = {
                'scenario': result.scenario.name,
                'description': result.scenario.description or '',
                'portfolio_value': result.portfolio_value,
                'pnl': result.portfolio_pnl,
                'pnl_pct': result.portfolio_pnl_pct,
            }
            
            if include_greeks and result.greeks:
                for key, value in result.greeks.items():
                    if key != 'market_value':
                        row[f'greek_{key}'] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)

