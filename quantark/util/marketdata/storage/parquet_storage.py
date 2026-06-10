"""
Parquet-based storage for market data time series.

Provides efficient columnar storage with compression and metadata preservation.
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import sys

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from quantark.util.marketdata.models import TimeSeriesData, MarketDataSet
from quantark.util.exceptions import ValidationError


class ParquetStorage:
    """
    Parquet-based storage system for market data.
    
    Features:
    - Efficient columnar storage with compression
    - Metadata preservation
    - Organized directory structure
    - Fast read/write operations
    """
    
    def __init__(self, base_path: str = "util/marketdata/data"):
        """
        Initialize storage.
        
        Args:
            base_path: Base directory for data storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_file_path(self, asset_name: str, data_type: str,
                      start_date: datetime, end_date: datetime) -> Path:
        """
        Generate file path for a dataset.
        
        Path structure: {base_path}/{asset_name}/{data_type}_{start}_{end}.parquet
        
        Args:
            asset_name: Asset identifier
            data_type: Type of data (spot, vol, rate, etc.)
            start_date: Start date
            end_date: End date
            
        Returns:
            Path object
        """
        asset_dir = self.base_path / asset_name
        asset_dir.mkdir(parents=True, exist_ok=True)
        
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        filename = f"{data_type}_{start_str}_{end_str}.parquet"
        
        return asset_dir / filename
    
    def save_time_series(self, data: TimeSeriesData, 
                        overwrite: bool = True) -> Path:
        """
        Save TimeSeriesData to Parquet file.
        
        Args:
            data: TimeSeriesData to save
            overwrite: Whether to overwrite existing file
            
        Returns:
            Path where data was saved
        """
        if len(data) == 0:
            raise ValidationError("Cannot save empty time series")
        
        # Get date range
        start_date = data.data.index.min()
        end_date = data.data.index.max()
        
        # Determine asset name and data type
        asset_name = data.asset_name or "unknown"
        data_type = data.data_type or "data"
        
        # Get file path
        file_path = self._get_file_path(asset_name, data_type, start_date, end_date)
        
        if file_path.exists() and not overwrite:
            raise FileExistsError(f"File {file_path} already exists. Set overwrite=True to replace.")
        
        # Prepare metadata
        metadata = {
            'asset_name': asset_name,
            'data_type': data_type,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'num_points': len(data),
            'columns': list(data.data.columns),
            'saved_at': datetime.now().isoformat(),
            **data.metadata
        }
        
        # Convert to Arrow table with metadata
        table = pa.Table.from_pandas(data.data, preserve_index=True)
        
        # Store metadata as custom metadata
        metadata_json = json.dumps(metadata)
        existing_metadata = table.schema.metadata or {}
        merged_metadata = {**existing_metadata, b'quantark_metadata': metadata_json.encode()}
        table = table.replace_schema_metadata(merged_metadata)
        
        # Write to Parquet with compression
        pq.write_table(table, file_path, compression='snappy')
        
        return file_path
    
    def load_time_series(self, asset_name: str, data_type: str,
                        start_date: datetime, end_date: datetime) -> TimeSeriesData:
        """
        Load TimeSeriesData from Parquet file.
        
        Args:
            asset_name: Asset identifier
            data_type: Type of data
            start_date: Start date (used to locate file)
            end_date: End date (used to locate file)
            
        Returns:
            TimeSeriesData object
        """
        file_path = self._get_file_path(asset_name, data_type, start_date, end_date)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Read Parquet file
        table = pq.read_table(file_path)
        
        # Extract metadata
        metadata = {}
        if table.schema.metadata and b'quantark_metadata' in table.schema.metadata:
            metadata_json = table.schema.metadata[b'quantark_metadata'].decode()
            metadata = json.loads(metadata_json)
        
        # Convert to DataFrame
        df = table.to_pandas()
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
        
        return TimeSeriesData(df, asset_name, data_type, metadata)
    
    def save_market_data_set(self, dataset: MarketDataSet,
                            overwrite: bool = True) -> List[Path]:
        """
        Save complete MarketDataSet (all time series).
        
        Args:
            dataset: MarketDataSet to save
            overwrite: Whether to overwrite existing files
            
        Returns:
            List of paths where data was saved
        """
        saved_paths = []
        
        # Save spot data
        saved_paths.append(self.save_time_series(dataset.spot_data, overwrite))
        
        # Save vol data
        saved_paths.append(self.save_time_series(dataset.vol_data, overwrite))
        
        # Save rate data
        saved_paths.append(self.save_time_series(dataset.rate_data, overwrite))
        
        # Save div yield data if present
        if dataset.div_yield_data is not None:
            saved_paths.append(self.save_time_series(dataset.div_yield_data, overwrite))
        
        # Save option data if present
        if dataset.option_data is not None:
            saved_paths.append(self.save_time_series(dataset.option_data, overwrite))
        
        return saved_paths
    
    def load_market_data_set(self, asset_name: str,
                            start_date: datetime,
                            end_date: datetime,
                            load_div_yield: bool = True,
                            load_option_data: bool = False) -> MarketDataSet:
        """
        Load complete MarketDataSet.
        
        Args:
            asset_name: Asset identifier
            start_date: Start date
            end_date: End date
            load_div_yield: Whether to load dividend yield data
            load_option_data: Whether to load option data
            
        Returns:
            MarketDataSet object
        """
        # Load required data
        spot_data = self.load_time_series(asset_name, 'spot', start_date, end_date)
        vol_data = self.load_time_series(asset_name, 'volatility', start_date, end_date)
        rate_data = self.load_time_series(asset_name, 'rate', start_date, end_date)
        
        # Load optional data
        div_yield_data = None
        if load_div_yield:
            try:
                div_yield_data = self.load_time_series(asset_name, 'div_yield', start_date, end_date)
            except FileNotFoundError:
                pass  # Optional, so OK if missing
        
        option_data = None
        if load_option_data:
            try:
                option_data = self.load_time_series(asset_name, 'option', start_date, end_date)
            except FileNotFoundError:
                pass
        
        return MarketDataSet(
            spot_data=spot_data,
            vol_data=vol_data,
            rate_data=rate_data,
            div_yield_data=div_yield_data,
            option_data=option_data,
            asset_name=asset_name
        )
    
    def list_assets(self) -> List[str]:
        """
        List all assets with stored data.
        
        Returns:
            List of asset names
        """
        if not self.base_path.exists():
            return []
        
        assets = []
        for item in self.base_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                assets.append(item.name)
        
        return sorted(assets)
    
    def list_files(self, asset_name: str) -> List[Dict[str, Any]]:
        """
        List all data files for an asset.
        
        Args:
            asset_name: Asset identifier
            
        Returns:
            List of file info dictionaries
        """
        asset_dir = self.base_path / asset_name
        if not asset_dir.exists():
            return []
        
        files = []
        for file_path in asset_dir.glob('*.parquet'):
            # Parse filename
            name_parts = file_path.stem.split('_')
            if len(name_parts) >= 3:
                data_type = name_parts[0]
                start_date = name_parts[1]
                end_date = name_parts[2]
                
                files.append({
                    'path': str(file_path),
                    'data_type': data_type,
                    'start_date': start_date,
                    'end_date': end_date,
                    'size_bytes': file_path.stat().st_size
                })
        
        return files
    
    def delete_file(self, asset_name: str, data_type: str,
                   start_date: datetime, end_date: datetime) -> bool:
        """
        Delete a data file.
        
        Args:
            asset_name: Asset identifier
            data_type: Type of data
            start_date: Start date
            end_date: End date
            
        Returns:
            True if file was deleted, False if it didn't exist
        """
        file_path = self._get_file_path(asset_name, data_type, start_date, end_date)
        
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get information about stored data.
        
        Returns:
            Dictionary with storage statistics
        """
        assets = self.list_assets()
        total_files = 0
        total_size = 0
        
        for asset in assets:
            files = self.list_files(asset)
            total_files += len(files)
            total_size += sum(f['size_bytes'] for f in files)
        
        return {
            'base_path': str(self.base_path),
            'num_assets': len(assets),
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'assets': assets
        }
    
    def __repr__(self) -> str:
        info = self.get_storage_info()
        return (f"ParquetStorage(base_path={info['base_path']}, "
                f"assets={info['num_assets']}, files={info['total_files']}, "
                f"size={info['total_size_mb']:.2f}MB)")

