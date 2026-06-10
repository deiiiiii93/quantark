"""
Export stress test results to various formats.
"""

from pathlib import Path
from typing import Union, List, Optional
import pandas as pd
import json
from datetime import datetime

from stresstest.results.stress_results import StressTestResults
from stresstest.results.result_aggregator import ResultAggregator
from util.exceptions import ValidationError


class ResultExporter:
    """
    Exports stress test results to various file formats.
    
    Supports exporting to:
    - Parquet (efficient binary format)
    - CSV (human-readable)
    - JSON (structured data)
    """
    
    @staticmethod
    def export_to_parquet(
        results: StressTestResults,
        filepath: Union[str, Path],
        include_positions: bool = True
    ) -> None:
        """
        Export results to Parquet format.
        
        Creates multiple parquet files:
        - {name}_summary.parquet: Scenario summary
        - {name}_positions_{scenario}.parquet: Position details per scenario (if enabled)
        
        Args:
            results: Stress test results
            filepath: Output file path (without extension) or directory
            include_positions: Whether to export position-level details
            
        Example:
            >>> ResultExporter.export_to_parquet(results, "stress_results")
            # Creates:
            # - stress_results_summary.parquet
            # - stress_results_positions_MarketCrash.parquet
            # - ...
        """
        filepath = Path(filepath)
        
        # Create output directory if filepath is a directory
        if filepath.suffix == '':
            output_dir = filepath
            base_name = "stress_results"
        else:
            output_dir = filepath.parent
            base_name = filepath.stem
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export summary
        summary_df = results.to_summary_dataframe()
        summary_path = output_dir / f"{base_name}_summary.parquet"
        summary_df.to_parquet(summary_path, index=False)
        print(f"Exported summary to: {summary_path}")

        # Export FI metrics if available
        fi_metrics_df = getattr(results, "get_dv01_series", None)
        if callable(fi_metrics_df):
            dv01_df = fi_metrics_df()
            if dv01_df is not None and not dv01_df.empty:
                dv01_path = output_dir / f"{base_name}_fi_metrics.parquet"
                dv01_df.to_parquet(dv01_path, index=False)
                print(f"Exported FI metrics to: {dv01_path}")
        
        # Export position details for each scenario
        if include_positions:
            for result in results.scenario_results:
                position_df = results.to_position_dataframe(result.scenario.name)
                if position_df is not None and not position_df.empty:
                    # Sanitize scenario name for filename
                    safe_name = "".join(c if c.isalnum() else "_" for c in result.scenario.name)
                    position_path = output_dir / f"{base_name}_positions_{safe_name}.parquet"
                    position_df.to_parquet(position_path, index=False)
                    print(f"Exported positions for {result.scenario.name} to: {position_path}")
        
        # Export greeks comparison if available
        greeks_df = ResultAggregator.get_greeks_comparison(results)
        if greeks_df is not None:
            greeks_path = output_dir / f"{base_name}_greeks.parquet"
            greeks_df.to_parquet(greeks_path, index=False)
            print(f"Exported Greeks to: {greeks_path}")
    
    @staticmethod
    def export_to_csv(
        results: StressTestResults,
        filepath: Union[str, Path],
        include_positions: bool = True
    ) -> None:
        """
        Export results to CSV format.
        
        Creates multiple CSV files similar to parquet export.
        
        Args:
            results: Stress test results
            filepath: Output file path (without extension) or directory
            include_positions: Whether to export position-level details
        """
        filepath = Path(filepath)
        
        # Create output directory if filepath is a directory
        if filepath.suffix == '':
            output_dir = filepath
            base_name = "stress_results"
        else:
            output_dir = filepath.parent
            base_name = filepath.stem
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export summary
        summary_df = results.to_summary_dataframe()
        summary_path = output_dir / f"{base_name}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Exported summary to: {summary_path}")

        fi_metrics_df = getattr(results, "get_dv01_series", None)
        if callable(fi_metrics_df):
            dv01_df = fi_metrics_df()
            if dv01_df is not None and not dv01_df.empty:
                dv01_path = output_dir / f"{base_name}_fi_metrics.csv"
                dv01_df.to_csv(dv01_path, index=False)
                print(f"Exported FI metrics to: {dv01_path}")
        
        # Export position details for each scenario
        if include_positions:
            for result in results.scenario_results:
                position_df = results.to_position_dataframe(result.scenario.name)
                if position_df is not None and not position_df.empty:
                    # Sanitize scenario name for filename
                    safe_name = "".join(c if c.isalnum() else "_" for c in result.scenario.name)
                    position_path = output_dir / f"{base_name}_positions_{safe_name}.csv"
                    position_df.to_csv(position_path, index=False)
                    print(f"Exported positions for {result.scenario.name} to: {position_path}")
        
        # Export greeks comparison if available
        greeks_df = ResultAggregator.get_greeks_comparison(results)
        if greeks_df is not None:
            greeks_path = output_dir / f"{base_name}_greeks.csv"
            greeks_df.to_csv(greeks_path, index=False)
            print(f"Exported Greeks to: {greeks_path}")
    
    @staticmethod
    def export_to_json(
        results: StressTestResults,
        filepath: Union[str, Path],
        pretty: bool = True
    ) -> None:
        """
        Export results to JSON format.
        
        Creates a single JSON file with all results.
        
        Args:
            results: Stress test results
            filepath: Output file path
            pretty: Whether to use pretty-printing (indent)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Build JSON structure
        baseline_fi_metrics = {}
        if hasattr(results, "extra_metrics"):
            baseline_fi_metrics = results.extra_metrics.get("fi", {})

        data = {
            'metadata': {
                'execution_timestamp': results.execution_timestamp.isoformat(),
                'total_execution_time': results.total_execution_time,
                'num_scenarios': len(results.scenario_results),
            },
            'config': results.config_summary,
            'baseline': {
                'portfolio_value': results.baseline_value,
                'greeks': results.baseline_greeks,
                'fi_metrics': baseline_fi_metrics,
            },
            'scenarios': []
        }
        
        # Add scenario results
        for result in results.scenario_results:
            fi_metrics = getattr(result, "get_fi_metrics", None)
            scenario_data = {
                'name': result.scenario.name,
                'description': result.scenario.description,
                'portfolio_value': result.portfolio_value,
                'portfolio_pnl': result.portfolio_pnl,
                'portfolio_pnl_pct': result.portfolio_pnl_pct,
                'greeks': result.greeks,
                'execution_time': result.execution_time,
                'stresses': [s.to_dict() for s in result.scenario.stresses],
                'position_results': result.position_results,
                'underlying_results': result.underlying_results,
                'fi_metrics': fi_metrics() if callable(fi_metrics) else {},
            }
            data['scenarios'].append(scenario_data)
        
        # Add summary statistics
        data['risk_summary'] = ResultAggregator.get_risk_summary(results)
        
        # Write JSON
        with open(filepath, 'w') as f:
            if pretty:
                json.dump(data, f, indent=2)
            else:
                json.dump(data, f)
        
        print(f"Exported to JSON: {filepath}")
    
    @staticmethod
    def export(
        results: StressTestResults,
        output_dir: Union[str, Path],
        formats: Optional[List[str]] = None,
        base_name: Optional[str] = None
    ) -> None:
        """
        Export results to multiple formats.
        
        Args:
            results: Stress test results
            output_dir: Output directory
            formats: List of formats to export ('parquet', 'csv', 'json')
            base_name: Base name for output files (default: timestamp-based)
        """
        if formats is None:
            formats = ['parquet', 'csv']
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if base_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"stress_results_{timestamp}"
        
        for fmt in formats:
            if fmt == 'parquet':
                ResultExporter.export_to_parquet(results, output_dir / base_name)
            elif fmt == 'csv':
                ResultExporter.export_to_csv(results, output_dir / base_name)
            elif fmt == 'json':
                json_path = output_dir / f"{base_name}.json"
                ResultExporter.export_to_json(results, json_path)
            else:
                raise ValidationError(f"Unknown export format: {fmt}")
    
    @staticmethod
    def export_risk_metrics(
        results: StressTestResults,
        filepath: Union[str, Path]
    ) -> None:
        """
        Export risk metrics summary to CSV.
        
        Args:
            results: Stress test results
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Get risk summary
        risk_summary = ResultAggregator.get_risk_summary(results)
        
        # Convert to DataFrame
        df = pd.DataFrame([risk_summary])
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        print(f"Exported risk metrics to: {filepath}")

