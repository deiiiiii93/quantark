"""
Portfolio storage and export functionality.
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
from .equity.portfolio import Portfolio
from .portfolio_snapshot import PortfolioSnapshot
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.util.exceptions import ValidationError


class PortfolioExporter:
    """
    Export and persistence manager for portfolios and snapshots.
    
    Provides functionality to export portfolios to various formats:
    - Parquet: Efficient columnar storage
    - Excel: Multi-sheet workbooks with positions, summary, and Greeks
    
    Also supports loading portfolios from stored files.
    """
    
    def __init__(self, base_path: str = "portfolio/data"):
        """
        Initialize exporter.
        
        Args:
            base_path: Base directory for data storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def export_to_parquet(
        self,
        portfolio: Portfolio,
        filepath: Optional[str] = None,
        include_greeks: bool = True,
        greeks_calculator: Optional[GreeksCalculator] = None
    ) -> Path:
        """
        Export portfolio positions to Parquet file.
        
        Args:
            portfolio: Portfolio to export
            filepath: Output file path (optional, auto-generated if not provided)
            include_greeks: Whether to include Greeks in export
            greeks_calculator: Greeks calculator (required if include_greeks=True)
            
        Returns:
            Path where data was saved
            
        Raises:
            ValidationError: If include_greeks=True but no calculator provided
        """
        if include_greeks and greeks_calculator is None:
            raise ValidationError("greeks_calculator required when include_greeks=True")
        
        # Get DataFrame
        df = portfolio.to_dataframe()
        
        if df.empty:
            raise ValidationError("Cannot export empty portfolio")
        
        # Add Greeks if requested
        if include_greeks:
            greeks_data = []
            for position in portfolio.positions.values():
                pricing_env = portfolio.pricing_environments[position.underlying]
                try:
                    greeks = position.get_greeks(pricing_env, greeks_calculator)
                    greeks_data.append(greeks)
                except Exception as e:
                    # Add empty Greeks on error
                    greeks_data.append({
                        'market_value': 0.0,
                        'delta': 0.0,
                        'gamma': 0.0,
                        'vega': 0.0,
                        'theta': 0.0,
                        'rho': 0.0,
                        'error': str(e)
                    })
            
            greeks_df = pd.DataFrame(greeks_data)
            # Merge Greeks into main DataFrame
            for col in greeks_df.columns:
                df[col] = greeks_df[col].values
        
        # Generate filepath if not provided
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.base_path / f"{portfolio.portfolio_name}_{timestamp}.parquet"
        else:
            filepath = Path(filepath)
        
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare metadata
        summary = portfolio.get_summary()
        metadata = {
            'portfolio_name': portfolio.portfolio_name,
            'creation_date': portfolio.creation_date.isoformat(),
            'export_timestamp': datetime.now().isoformat(),
            'num_positions': len(portfolio.positions),
            'total_value': summary['total_value'],
            'total_pnl': summary['total_pnl'],
            'include_greeks': include_greeks
        }
        
        # Convert to Arrow table with metadata
        table = pa.Table.from_pandas(df)
        metadata_json = json.dumps(metadata)
        existing_metadata = table.schema.metadata or {}
        merged_metadata = {**existing_metadata, b'portfolio_metadata': metadata_json.encode()}
        table = table.replace_schema_metadata(merged_metadata)
        
        # Write to Parquet
        pq.write_table(table, filepath, compression='snappy')
        
        return filepath
    
    def export_to_excel(
        self,
        portfolio: Portfolio,
        filepath: Optional[str] = None,
        greeks_calculator: Optional[GreeksCalculator] = None
    ) -> Path:
        """
        Export portfolio to Excel file with multiple sheets.
        
        Sheets:
        1. Positions: All position details with current values
        2. Summary: Portfolio summary statistics
        3. Greeks_by_Position: Position-level Greeks
        4. Greeks_by_Underlying: Aggregated Greeks per underlying
        
        Args:
            portfolio: Portfolio to export
            filepath: Output file path (optional, auto-generated if not provided)
            greeks_calculator: Greeks calculator for risk metrics
            
        Returns:
            Path where file was saved
        """
        # Generate filepath if not provided
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.base_path / f"{portfolio.portfolio_name}_{timestamp}.xlsx"
        else:
            filepath = Path(filepath)
        
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Sheet 1: Positions
            positions_df = portfolio.to_dataframe()
            if not positions_df.empty:
                positions_df.to_excel(writer, sheet_name='Positions', index=False)
            
            # Sheet 2: Summary
            summary = portfolio.get_summary()
            summary_df = pd.DataFrame([summary])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 3 & 4: Greeks (if calculator provided)
            if greeks_calculator is not None:
                # Greeks by Position
                greeks_by_position = []
                for pos_id, position in portfolio.positions.items():
                    pricing_env = portfolio.pricing_environments[position.underlying]
                    try:
                        greeks = position.get_greeks(pricing_env, greeks_calculator)
                        greeks['position_id'] = pos_id
                        greeks['underlying'] = position.underlying
                        greeks['product_type'] = position.product.__class__.__name__
                        greeks_by_position.append(greeks)
                    except Exception as e:
                        greeks_by_position.append({
                            'position_id': pos_id,
                            'underlying': position.underlying,
                            'product_type': position.product.__class__.__name__,
                            'error': str(e)
                        })
                
                if greeks_by_position:
                    greeks_pos_df = pd.DataFrame(greeks_by_position)
                    greeks_pos_df.to_excel(writer, sheet_name='Greeks_by_Position', index=False)
                
                # Greeks by Underlying
                underlyings = set(pos.underlying for pos in portfolio.positions.values())
                greeks_by_underlying = []
                for underlying in underlyings:
                    try:
                        greeks = portfolio.get_greeks_by_underlying(
                            underlying,
                            greeks_calculator
                        )
                        greeks['underlying'] = underlying
                        num_positions = len(portfolio.get_positions_by_underlying(underlying))
                        greeks['num_positions'] = num_positions
                        greeks_by_underlying.append(greeks)
                    except Exception as e:
                        greeks_by_underlying.append({
                            'underlying': underlying,
                            'error': str(e)
                        })
                
                if greeks_by_underlying:
                    greeks_und_df = pd.DataFrame(greeks_by_underlying)
                    greeks_und_df.to_excel(writer, sheet_name='Greeks_by_Underlying', index=False)
        
        return filepath
    
    def export_snapshot_to_parquet(
        self,
        snapshot: PortfolioSnapshot,
        filepath: Optional[str] = None
    ) -> Path:
        """
        Export portfolio snapshot to Parquet file.
        
        Args:
            snapshot: PortfolioSnapshot to export
            filepath: Output file path (optional, auto-generated if not provided)
            
        Returns:
            Path where data was saved
        """
        if not snapshot.positions_data:
            raise ValidationError("Cannot export empty snapshot")
        
        # Convert positions to DataFrame
        df = pd.DataFrame(snapshot.positions_data)
        
        # Generate filepath if not provided
        if filepath is None:
            timestamp = snapshot.timestamp.strftime('%Y%m%d_%H%M%S')
            filepath = self.base_path / f"{snapshot.portfolio_name}_snapshot_{timestamp}.parquet"
        else:
            filepath = Path(filepath)
        
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare metadata
        metadata = {
            'snapshot_timestamp': snapshot.timestamp.isoformat(),
            'portfolio_name': snapshot.portfolio_name,
            'total_value': snapshot.total_value,
            'total_pnl': snapshot.total_pnl,
            'aggregated_greeks': snapshot.aggregated_greeks,
            'num_positions': len(snapshot.positions_data),
            **snapshot.metadata
        }
        
        # Convert to Arrow table with metadata
        table = pa.Table.from_pandas(df)
        metadata_json = json.dumps(metadata, default=str)
        existing_metadata = table.schema.metadata or {}
        merged_metadata = {**existing_metadata, b'snapshot_metadata': metadata_json.encode()}
        table = table.replace_schema_metadata(merged_metadata)
        
        # Write to Parquet
        pq.write_table(table, filepath, compression='snappy')
        
        return filepath
    
    def export_history_to_parquet(
        self,
        snapshots: List[PortfolioSnapshot],
        filepath: Optional[str] = None
    ) -> Path:
        """
        Export a series of portfolio snapshots (history) to Parquet file.
        
        Useful for tracking portfolio performance over time.
        
        Args:
            snapshots: List of PortfolioSnapshot objects
            filepath: Output file path (optional, auto-generated if not provided)
            
        Returns:
            Path where data was saved
            
        Raises:
            ValidationError: If snapshots list is empty
        """
        if not snapshots:
            raise ValidationError("Cannot export empty snapshot history")
        
        # Collect all position data with snapshot timestamps
        all_positions = []
        for snapshot in snapshots:
            for pos in snapshot.positions_data:
                pos_copy = pos.copy()
                pos_copy['snapshot_timestamp'] = snapshot.timestamp.isoformat()
                pos_copy['snapshot_total_value'] = snapshot.total_value
                pos_copy['snapshot_total_pnl'] = snapshot.total_pnl
                all_positions.append(pos_copy)
        
        df = pd.DataFrame(all_positions)
        
        # Generate filepath if not provided
        if filepath is None:
            portfolio_name = snapshots[0].portfolio_name
            start_time = min(s.timestamp for s in snapshots).strftime('%Y%m%d')
            end_time = max(s.timestamp for s in snapshots).strftime('%Y%m%d')
            filepath = self.base_path / f"{portfolio_name}_history_{start_time}_{end_time}.parquet"
        else:
            filepath = Path(filepath)
        
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare metadata
        metadata = {
            'portfolio_name': snapshots[0].portfolio_name,
            'num_snapshots': len(snapshots),
            'start_timestamp': min(s.timestamp for s in snapshots).isoformat(),
            'end_timestamp': max(s.timestamp for s in snapshots).isoformat(),
            'total_positions': len(all_positions),
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Convert to Arrow table with metadata
        table = pa.Table.from_pandas(df)
        metadata_json = json.dumps(metadata)
        existing_metadata = table.schema.metadata or {}
        merged_metadata = {**existing_metadata, b'history_metadata': metadata_json.encode()}
        table = table.replace_schema_metadata(merged_metadata)
        
        # Write to Parquet
        pq.write_table(table, filepath, compression='snappy')
        
        return filepath
    
    def load_from_parquet(self, filepath: str) -> pd.DataFrame:
        """
        Load portfolio data from Parquet file.
        
        Note: This loads the position data as a DataFrame. To reconstruct
        a Portfolio object, you would need to recreate products and engines.
        
        Args:
            filepath: Path to Parquet file
            
        Returns:
            DataFrame with position data and metadata
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Read Parquet file
        table = pq.read_table(filepath)
        
        # Extract metadata
        metadata = {}
        for key in [b'portfolio_metadata', b'snapshot_metadata', b'history_metadata']:
            if table.schema.metadata and key in table.schema.metadata:
                metadata_json = table.schema.metadata[key].decode()
                metadata = json.loads(metadata_json)
                break
        
        # Convert to DataFrame
        df = table.to_pandas()
        
        # Attach metadata as attribute
        df.attrs['metadata'] = metadata
        
        return df
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about stored portfolio files.
        
        Returns:
            Dictionary with storage statistics
        """
        if not self.base_path.exists():
            return {
                'base_path': str(self.base_path),
                'num_files': 0,
                'total_size_bytes': 0,
                'total_size_mb': 0.0
            }
        
        files = list(self.base_path.glob('*.parquet')) + list(self.base_path.glob('*.xlsx'))
        total_size = sum(f.stat().st_size for f in files)
        
        return {
            'base_path': str(self.base_path),
            'num_files': len(files),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'files': [str(f.name) for f in files]
        }
    
    def __repr__(self) -> str:
        info = self.get_storage_info()
        return (
            f"PortfolioExporter(base_path={info['base_path']}, "
            f"files={info['num_files']}, "
            f"size={info['total_size_mb']:.2f}MB)"
        )

