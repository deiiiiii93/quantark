"""
CRIF Export Module.

This module provides CRIF (Common Risk Interchange Format) export functionality
for SIMM sensitivities.
"""
from typing import List, Dict, Any, Optional, Union
from datetime import date
import csv
import io

from ..crif import CRIFRecord, sensitivities_to_crif
from ..sensitivity import SensitivityCollection, Sensitivity


def export_sensitivities_to_crif(
    sensitivities: SensitivityCollection,
    trade_id: Optional[str] = None,
    valuation_date: Optional[date] = None,
    output_path: Optional[str] = None,
) -> Union[List[CRIFRecord], str]:
    """Export calculated sensitivities to CRIF format.

    Enables round-trip: Portfolio → Sensitivities → CRIF → External System

    Args:
        sensitivities: SensitivityCollection to export.
        trade_id: Trade ID to use for all sensitivities (optional).
        valuation_date: Valuation date (defaults to today).
        output_path: Optional output file path. If provided, writes CSV to file.
                     If None, returns list of CRIFRecord objects.

    Returns:
        If output_path is None: List of CRIFRecord objects.
        If output_path is provided: Output file path string.
    """
    if valuation_date is None:
        valuation_date = date.today()

    # Convert sensitivities to CRIF format
    crif_records = sensitivities_to_crif(sensitivities, valuation_date)

    # If output path provided, write to CSV
    if output_path:
        write_crif_csv(crif_records, output_path, valuation_date)
        return output_path

    return crif_records


def write_crif_csv(
    crif_records: List[CRIFRecord],
    output_path: str,
    valuation_date: date,
) -> None:
    """Write CRIF records to CSV file.

    Args:
        crif_records: List of CRIFRecord objects.
        output_path: Output CSV file path.
        valuation_date: Valuation date for the file.
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'Valuation_Date',
            'Trade_ID',
            'Risk_Type',
            'Bucket',
            'Sensitivity',
            'Qualifier',
            'Currency',
            'Tenor'
        ])

        # Write records
        for record in crif_records:
            writer.writerow([
                valuation_date.strftime('%Y-%m-%d'),
                record.trade_id,
                record.risk_type.value if hasattr(record.risk_type, 'value') else str(record.risk_type),
                record.bucket,
                record.amount,
                record.qualifier or '',
                record.amount_currency,
                record.label1 or ''
            ])


def export_result_sensitivities_to_dataframe(
    result: Any,
    include_zero_sensitivities: bool = False,
) -> Any:
    """Export sensitivities from a SIMMResult to a pandas DataFrame.

    Args:
        result: SIMMResult with sensitivities.
        include_zero_sensitivities: Whether to include zero sensitivities.

    Returns:
        DataFrame with sensitivity data.

    Raises:
        ImportError: If pandas not installed.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for DataFrame export")

    # Extract sensitivities from result
    # In practice, you'd need to store or retrieve the original sensitivities
    # This is a placeholder implementation

    data = []
    # Iterate through buckets to extract sensitivity info
    for pc, rc_dict in result.risk_class_margin.items():
        for rc, rc_margin in rc_dict.items():
            for bucket, bucket_detail in rc_margin.bucket_detail.items():
                for sens_contrib in bucket_detail.sensitivities:
                    data.append({
                        'Product_Class': pc.value,
                        'Risk_Class': rc.value,
                        'Bucket': bucket,
                        'Position_ID': sens_contrib.position_id,
                        'Sensitivity_ID': sens_contrib.sensitivity_id,
                        'WS_Value': sens_contrib.ws_value,
                        'Pct_of_Bucket': sens_contrib.pct_of_bucket,
                        'Bucket_K': bucket_detail.k_value,
                    })

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def create_crif_template(output_path: str) -> str:
    """Create a CRIF template CSV file.

    Args:
        output_path: Output file path.

    Returns:
        Output file path.
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'Valuation_Date',
            'Trade_ID',
            'Risk_Type',
            'Bucket',
            'Sensitivity',
            'Qualifier',
            'Currency',
            'Tenor'
        ])

        # Write example rows
        examples = [
            ['2024-01-15', 'TRADE_001', 'Risk_IRCurve', 'USD', '1000000', 'USD', 'USD', ''],
            ['2024-01-15', 'TRADE_001', 'Risk_IRVol', 'USD', '-50000', 'USD', 'USD', '1y'],
            ['2024-01-15', 'TRADE_002', 'Risk_Equity', '5', '100000', 'AAPL', 'USD', ''],
        ]

        for example in examples:
            writer.writerow(example)

    return output_path


def validate_crif_file(file_path: str) -> Dict[str, Any]:
    """Validate a CRIF CSV file.

    Args:
        file_path: Path to CRIF CSV file.

    Returns:
        Dictionary with validation results.
    """
    from ..crif import parse_crif_csv, CRIFValidationError

    try:
        records = parse_crif_csv(file_path)

        # Basic validation
        risk_types = set()
        currencies = set()
        trade_ids = set()

        for record in records:
            risk_types.add(str(record.risk_type))
            currencies.add(record.currency)
            trade_ids.add(record.trade_id)

        return {
            'valid': True,
            'record_count': len(records),
            'unique_risk_types': len(risk_types),
            'unique_currencies': len(currencies),
            'unique_trades': len(trade_ids),
            'risk_types': sorted(list(risk_types)),
            'currencies': sorted(list(currencies)),
        }

    except CRIFValidationError as e:
        return {
            'valid': False,
            'error': str(e),
        }
    except Exception as e:
        return {
            'valid': False,
            'error': f'Unexpected error: {str(e)}',
        }


def export_sensitivities_csv(
    sensitivities: SensitivityCollection,
    output_path: str,
    include_metadata: bool = True,
) -> str:
    """Export sensitivities to a simplified CSV format.

    Args:
        sensitivities: SensitivityCollection to export.
        output_path: Output CSV file path.
        include_metadata: Whether to include metadata columns.

    Returns:
        Output file path.
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header
        headers = [
            'Position_ID',
            'Trade_ID',
            'Risk_Class',
            'Margin_Type',
            'Sensitivity_Type',
            'Bucket',
            'Value',
            'Currency',
            'Qualifier',
        ]

        if include_metadata:
            headers.extend([
                'Timestamp',
                'Label',
            ])

        writer.writerow(headers)

        # Write sensitivities
        for sens in sensitivities.sensitivities:
            row = [
                getattr(sens, 'position_id', ''),
                getattr(sens, 'trade_id', ''),
                str(getattr(sens, 'risk_class', '')),
                str(getattr(sens, 'margin_type', '')),
                str(getattr(sens, 'sensitivity_type', '')),
                str(getattr(sens, 'bucket', '')),
                sens.value,
                getattr(sens, 'currency', ''),
                getattr(sens, 'qualifier', ''),
            ]

            if include_metadata:
                row.extend([
                    getattr(sens, 'timestamp', ''),
                    getattr(sens, 'label', ''),
                ])

            writer.writerow(row)

    return output_path
