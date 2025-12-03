"""
Export functionality for dynamic scenario results.
"""

from pathlib import Path
from typing import Union, Optional, List
import json
from datetime import datetime

from dynamicscenario.results.dynamic_results import DynamicScenarioResults


class DynamicResultExporter:
    """
    Exports dynamic scenario results to various formats.

    Supports:
    - CSV: Tabular data for spreadsheet analysis
    - Parquet: Efficient columnar format for large datasets
    - JSON: Full hierarchical data for programmatic access

    Example:
        >>> exporter = DynamicResultExporter(results)
        >>> exporter.export_all("./output", formats=['csv', 'json'])
    """

    def __init__(self, results: DynamicScenarioResults):
        """
        Initialize exporter with results.

        Args:
            results: Dynamic scenario results to export
        """
        self.results = results

    def export_all(
        self,
        output_dir: Union[str, Path],
        formats: Optional[List[str]] = None,
        prefix: Optional[str] = None,
    ) -> List[str]:
        """
        Export results to all specified formats.

        Args:
            output_dir: Output directory
            formats: List of formats ('csv', 'parquet', 'json')
            prefix: Optional filename prefix

        Returns:
            List of created file paths
        """
        if formats is None:
            formats = ["csv", "parquet", "json"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if prefix is None:
            # Create prefix from path name and timestamp
            safe_name = self.results.path_name.replace(" ", "_").replace("/", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = f"dynamic_{safe_name}_{timestamp}"

        created_files = []

        for fmt in formats:
            if fmt.lower() == "csv":
                files = self.export_csv(output_dir, prefix)
                created_files.extend(files)
            elif fmt.lower() == "parquet":
                files = self.export_parquet(output_dir, prefix)
                created_files.extend(files)
            elif fmt.lower() == "json":
                file = self.export_json(output_dir, prefix)
                created_files.append(file)

        return created_files

    def export_csv(self, output_dir: Union[str, Path], prefix: str) -> List[str]:
        """
        Export results to CSV files.

        Creates multiple CSV files:
        - {prefix}_summary.csv: Day-by-day summary
        - {prefix}_pnl.csv: P&L evolution
        - {prefix}_greeks.csv: Greeks evolution
        - {prefix}_positions.csv: Position evolution
        - {prefix}_trades.csv: All trades
        - {prefix}_market.csv: Market state evolution

        Args:
            output_dir: Output directory
            prefix: Filename prefix

        Returns:
            List of created file paths
        """
        output_dir = Path(output_dir)
        created_files = []

        # Summary DataFrame
        summary_df = self.results.to_summary_dataframe()
        summary_path = output_dir / f"{prefix}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        created_files.append(str(summary_path))
        print(f"Exported: {summary_path}")

        # P&L evolution
        pnl_df = self.results.get_pnl_evolution()
        pnl_path = output_dir / f"{prefix}_pnl.csv"
        pnl_df.to_csv(pnl_path, index=False)
        created_files.append(str(pnl_path))
        print(f"Exported: {pnl_path}")

        # Greeks evolution
        greeks_df = self.results.get_greeks_evolution()
        if not greeks_df.empty:
            greeks_path = output_dir / f"{prefix}_greeks.csv"
            greeks_df.to_csv(greeks_path, index=False)
            created_files.append(str(greeks_path))
            print(f"Exported: {greeks_path}")

        # Position evolution
        positions_df = self.results.get_position_evolution()
        if not positions_df.empty:
            positions_path = output_dir / f"{prefix}_positions.csv"
            positions_df.to_csv(positions_path, index=False)
            created_files.append(str(positions_path))
            print(f"Exported: {positions_path}")

        # Trades
        trades_df = self.results.get_trades_dataframe()
        if not trades_df.empty:
            trades_path = output_dir / f"{prefix}_trades.csv"
            trades_df.to_csv(trades_path, index=False)
            created_files.append(str(trades_path))
            print(f"Exported: {trades_path}")

        # Market evolution
        market_df = self.results.get_market_evolution()
        if not market_df.empty:
            market_path = output_dir / f"{prefix}_market.csv"
            market_df.to_csv(market_path, index=False)
            created_files.append(str(market_path))
            print(f"Exported: {market_path}")

        return created_files

    def export_parquet(self, output_dir: Union[str, Path], prefix: str) -> List[str]:
        """
        Export results to Parquet files.

        Same structure as CSV but in efficient Parquet format.

        Args:
            output_dir: Output directory
            prefix: Filename prefix

        Returns:
            List of created file paths
        """
        output_dir = Path(output_dir)
        created_files = []

        try:
            # Summary DataFrame
            summary_df = self.results.to_summary_dataframe()
            summary_path = output_dir / f"{prefix}_summary.parquet"
            summary_df.to_parquet(summary_path, index=False)
            created_files.append(str(summary_path))
            print(f"Exported: {summary_path}")

            # P&L evolution
            pnl_df = self.results.get_pnl_evolution()
            pnl_path = output_dir / f"{prefix}_pnl.parquet"
            pnl_df.to_parquet(pnl_path, index=False)
            created_files.append(str(pnl_path))
            print(f"Exported: {pnl_path}")

            # Greeks evolution
            greeks_df = self.results.get_greeks_evolution()
            if not greeks_df.empty:
                greeks_path = output_dir / f"{prefix}_greeks.parquet"
                greeks_df.to_parquet(greeks_path, index=False)
                created_files.append(str(greeks_path))
                print(f"Exported: {greeks_path}")

            # Position evolution
            positions_df = self.results.get_position_evolution()
            if not positions_df.empty:
                positions_path = output_dir / f"{prefix}_positions.parquet"
                positions_df.to_parquet(positions_path, index=False)
                created_files.append(str(positions_path))
                print(f"Exported: {positions_path}")

            # Trades
            trades_df = self.results.get_trades_dataframe()
            if not trades_df.empty:
                trades_path = output_dir / f"{prefix}_trades.parquet"
                trades_df.to_parquet(trades_path, index=False)
                created_files.append(str(trades_path))
                print(f"Exported: {trades_path}")

        except ImportError:
            print("Warning: pyarrow not installed, skipping Parquet export")

        return created_files

    def export_json(self, output_dir: Union[str, Path], prefix: str) -> str:
        """
        Export complete results to JSON.

        Args:
            output_dir: Output directory
            prefix: Filename prefix

        Returns:
            Path to created JSON file
        """
        output_dir = Path(output_dir)
        json_path = output_dir / f"{prefix}_results.json"

        # Convert to dict
        data = self.results.to_dict()

        # Write JSON
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Exported: {json_path}")
        return str(json_path)

    def export_summary_text(self, output_dir: Union[str, Path], prefix: str) -> str:
        """
        Export summary as text file.

        Args:
            output_dir: Output directory
            prefix: Filename prefix

        Returns:
            Path to created text file
        """
        output_dir = Path(output_dir)
        text_path = output_dir / f"{prefix}_summary.txt"

        with open(text_path, "w") as f:
            f.write(self.results.get_summary())

        print(f"Exported: {text_path}")
        return str(text_path)


class FIResultExporter:
    """
    Exports FI dynamic scenario results to various formats.

    Extends standard export with FI-specific metrics like DV01, duration,
    convexity evolution, and rate tracking.

    Example:
        >>> exporter = FIResultExporter(fi_results)
        >>> exporter.export_all("./output", formats=['csv', 'json'])
    """

    def __init__(self, results):
        """
        Initialize exporter with FI results.

        Args:
            results: FI Dynamic scenario results to export
        """
        self.results = results

    def export_all(
        self,
        output_dir: Union[str, Path],
        formats: Optional[List[str]] = None,
        prefix: Optional[str] = None,
    ) -> List[str]:
        """
        Export FI results to all specified formats.

        Args:
            output_dir: Output directory
            formats: List of formats ('csv', 'parquet', 'json')
            prefix: Optional filename prefix

        Returns:
            List of created file paths
        """
        if formats is None:
            formats = ["csv", "parquet", "json"]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if prefix is None:
            safe_name = self.results.path_name.replace(" ", "_").replace("/", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = f"fi_dynamic_{safe_name}_{timestamp}"

        created_files = []

        for fmt in formats:
            if fmt.lower() == "csv":
                files = self.export_csv(output_dir, prefix)
                created_files.extend(files)
            elif fmt.lower() == "parquet":
                files = self.export_parquet(output_dir, prefix)
                created_files.extend(files)
            elif fmt.lower() == "json":
                file = self.export_json(output_dir, prefix)
                created_files.append(file)

        return created_files

    def export_csv(self, output_dir: Union[str, Path], prefix: str) -> List[str]:
        """
        Export FI results to CSV files.

        Creates FI-specific CSV files:
        - {prefix}_summary.csv: Day-by-day summary with FI metrics
        - {prefix}_dv01.csv: DV01 evolution
        - {prefix}_duration.csv: Duration evolution
        - {prefix}_rate.csv: Rate evolution
        - {prefix}_trades.csv: Hedge trades

        Args:
            output_dir: Output directory
            prefix: Filename prefix

        Returns:
            List of created file paths
        """
        output_dir = Path(output_dir)
        created_files = []

        # Summary DataFrame
        summary_df = self.results.to_summary_dataframe()
        summary_path = output_dir / f"{prefix}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        created_files.append(str(summary_path))
        print(f"Exported: {summary_path}")

        # DV01 evolution
        dv01_df = self.results.get_dv01_evolution()
        dv01_path = output_dir / f"{prefix}_dv01.csv"
        dv01_df.to_csv(dv01_path, index=False)
        created_files.append(str(dv01_path))
        print(f"Exported: {dv01_path}")

        # Duration evolution
        duration_df = self.results.get_duration_evolution()
        duration_path = output_dir / f"{prefix}_duration.csv"
        duration_df.to_csv(duration_path, index=False)
        created_files.append(str(duration_path))
        print(f"Exported: {duration_path}")

        # Rate evolution
        rate_df = self.results.get_rate_evolution()
        rate_path = output_dir / f"{prefix}_rate.csv"
        rate_df.to_csv(rate_path, index=False)
        created_files.append(str(rate_path))
        print(f"Exported: {rate_path}")

        # Hedge trades
        trades_df = self.results.get_hedge_trades()
        if not trades_df.empty:
            trades_path = output_dir / f"{prefix}_trades.csv"
            trades_df.to_csv(trades_path, index=False)
            created_files.append(str(trades_path))
            print(f"Exported: {trades_path}")

        # Key-rate DV01 if available
        kr_dv01_df = self.results.get_key_rate_dv01_evolution()
        if not kr_dv01_df.empty and len(kr_dv01_df.columns) > 2:
            kr_path = output_dir / f"{prefix}_key_rate_dv01.csv"
            kr_dv01_df.to_csv(kr_path, index=False)
            created_files.append(str(kr_path))
            print(f"Exported: {kr_path}")

        return created_files

    def export_parquet(self, output_dir: Union[str, Path], prefix: str) -> List[str]:
        """
        Export FI results to Parquet files.

        Args:
            output_dir: Output directory
            prefix: Filename prefix

        Returns:
            List of created file paths
        """
        output_dir = Path(output_dir)
        created_files = []

        try:
            summary_df = self.results.to_summary_dataframe()
            summary_path = output_dir / f"{prefix}_summary.parquet"
            summary_df.to_parquet(summary_path, index=False)
            created_files.append(str(summary_path))
            print(f"Exported: {summary_path}")

            dv01_df = self.results.get_dv01_evolution()
            dv01_path = output_dir / f"{prefix}_dv01.parquet"
            dv01_df.to_parquet(dv01_path, index=False)
            created_files.append(str(dv01_path))
            print(f"Exported: {dv01_path}")

        except ImportError:
            print("Warning: pyarrow not installed, skipping Parquet export")

        return created_files

    def export_json(self, output_dir: Union[str, Path], prefix: str) -> str:
        """
        Export complete FI results to JSON.

        Args:
            output_dir: Output directory
            prefix: Filename prefix

        Returns:
            Path to created JSON file
        """
        output_dir = Path(output_dir)
        json_path = output_dir / f"{prefix}_results.json"

        data = self.results.to_dict()

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Exported: {json_path}")
        return str(json_path)
