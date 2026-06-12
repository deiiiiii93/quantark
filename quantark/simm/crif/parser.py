"""
CRIF Parser Module.

This module provides CSV parsing, validation, and conversion functions
for CRIF (Common Risk Interchange Format) data.
"""
import csv
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from quantark.util.exceptions import ValidationError

from .models import (
    CRIF_COLUMN_MAPPING,
    CRIF_REQUIRED_COLUMNS,
    CRIFHeader,
    CRIFRecord,
)
from ..sensitivity import (
    AnySensitivity,
    BaseCorrSensitivity,
    CommodityDeltaSensitivity,
    CommodityVegaSensitivity,
    CreditDeltaSensitivity,
    CreditVegaSensitivity,
    EquityDeltaSensitivity,
    EquityVegaSensitivity,
    FXDeltaSensitivity,
    FXVegaSensitivity,
    IRDeltaSensitivity,
    IRInflationDeltaSensitivity,
    IRVegaSensitivity,
    IRXCcyBasisSensitivity,
    SensitivityCollection,
)
from ..taxonomy import (
    CREDIT_TENOR_LABELS,
    IR_TENOR_LABELS,
    IR_TENORS,
    CREDIT_TENORS,
    IRSubCurve,
    MarginType,
    RiskClass,
    SensitivityType,
)


class CRIFValidationError(ValidationError):
    """Error raised when CRIF validation fails."""
    
    def __init__(self, message: str, row_number: Optional[int] = None, field: Optional[str] = None):
        self.row_number = row_number
        self.field = field
        if row_number is not None:
            message = f"Row {row_number}: {message}"
        if field is not None:
            message = f"{message} (field: {field})"
        super().__init__(message)


def parse_crif_csv(
    source: Union[str, Path, StringIO],
    validate: bool = True,
    strict: bool = False,
) -> Tuple[List[CRIFRecord], List[str]]:
    """Parse a CRIF CSV file or string into CRIFRecord objects.
    
    Args:
        source: File path, Path object, or StringIO containing CSV data.
        validate: Whether to validate records (default True).
        strict: If True, raise on any validation error. If False, collect warnings.
        
    Returns:
        Tuple of (list of CRIFRecords, list of warning messages).
        
    Raises:
        CRIFValidationError: If strict=True and validation fails.
        FileNotFoundError: If source is a path that doesn't exist.
    """
    records: List[CRIFRecord] = []
    warnings: List[str] = []
    
    # Open the source
    if isinstance(source, StringIO):
        reader = csv.DictReader(source)
        rows = list(reader)
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"CRIF file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    else:
        raise TypeError(f"source must be str, Path, or StringIO, got {type(source)}")
    
    # Check for required columns
    if rows:
        available_columns = set(rows[0].keys())
        missing_columns = set(CRIF_REQUIRED_COLUMNS) - available_columns
        if missing_columns:
            msg = f"Missing required columns: {missing_columns}"
            if strict:
                raise CRIFValidationError(msg)
            warnings.append(msg)
    
    # Parse each row
    for row_num, row in enumerate(rows, start=2):  # Start at 2 (1 is header)
        try:
            record = _parse_crif_row(row, row_num)
            if validate:
                row_warnings = _validate_crif_record(record, row_num, strict)
                warnings.extend(row_warnings)
            records.append(record)
        except CRIFValidationError:
            if strict:
                raise
            warnings.append(f"Row {row_num}: Failed to parse record")
        except Exception as e:
            if strict:
                raise CRIFValidationError(str(e), row_number=row_num)
            warnings.append(f"Row {row_num}: {str(e)}")
    
    return records, warnings


def _parse_crif_row(row: Dict[str, Any], row_num: int) -> CRIFRecord:
    """Parse a single CSV row into a CRIFRecord."""
    # Map column names to field names
    data: Dict[str, Any] = {}
    
    for col_name, field_name in CRIF_COLUMN_MAPPING.items():
        if col_name in row:
            value = row[col_name]
            if value is not None and value != "":
                data[field_name] = value
    
    # Parse special fields
    # ValuationDate
    if "valuation_date" in data:
        val_date = data["valuation_date"]
        if isinstance(val_date, str):
            # Try common date formats
            for fmt in ("%Y-%m-%d", "%Y%m%d", "%d/%m/%Y", "%m/%d/%Y"):
                try:
                    data["valuation_date"] = datetime.strptime(val_date, fmt).date()
                    break
                except ValueError:
                    continue
            else:
                raise CRIFValidationError(f"Cannot parse date: {val_date}", row_number=row_num)
    else:
        # Default to today if not provided
        data["valuation_date"] = date.today()
    
    # Parse numeric fields
    for field in ("amount", "amount_usd", "notional"):
        if field in data:
            try:
                data[field] = float(data[field])
            except (ValueError, TypeError):
                data[field] = 0.0
    
    # Ensure required fields have defaults
    data.setdefault("trade_id", f"UNKNOWN_{row_num}")
    data.setdefault("risk_type", "")
    data.setdefault("qualifier", "")
    data.setdefault("bucket", "")
    data.setdefault("amount", 0.0)
    data.setdefault("amount_currency", "USD")
    
    return CRIFRecord(**data)


def _validate_crif_record(
    record: CRIFRecord, row_num: int, strict: bool
) -> List[str]:
    """Validate a CRIF record and return warnings."""
    warnings: List[str] = []
    
    # Validate risk_type
    sensitivity_type = record.get_sensitivity_type()
    if sensitivity_type is None:
        msg = f"Unknown risk_type: {record.risk_type}"
        if strict:
            raise CRIFValidationError(msg, row_number=row_num, field="RiskType")
        warnings.append(f"Row {row_num}: {msg}")
    
    # Validate qualifier is not empty
    if not record.qualifier:
        msg = "Qualifier is empty"
        if strict:
            raise CRIFValidationError(msg, row_number=row_num, field="Qualifier")
        warnings.append(f"Row {row_num}: {msg}")
    
    # Validate amount_currency
    if len(record.amount_currency) != 3:
        msg = f"Invalid currency code: {record.amount_currency}"
        if strict:
            raise CRIFValidationError(msg, row_number=row_num, field="AmountCurrency")
        warnings.append(f"Row {row_num}: {msg}")
    
    return warnings


def crif_to_sensitivities(
    records: List[CRIFRecord],
) -> SensitivityCollection:
    """Convert CRIF records to Sensitivity objects.
    
    Args:
        records: List of CRIFRecord objects to convert.
        
    Returns:
        SensitivityCollection containing converted sensitivities.
    """
    collection = SensitivityCollection()
    
    for record in records:
        sensitivity = _convert_crif_record(record)
        if sensitivity:
            collection.add(sensitivity)
    
    return collection


def crif_records_to_sensitivities(
    records: List[Dict[str, Any]],
) -> SensitivityCollection:
    """Convert CRIF record dictionaries (CRIF column names or CRIFRecord
    field names) to a SensitivityCollection."""
    parsed: List[CRIFRecord] = []
    for i, rec in enumerate(records, start=1):
        if isinstance(rec, CRIFRecord):
            parsed.append(rec)
        else:
            row = dict(rec)
            # Accept field-name keys directly as well as CRIF column names.
            for col, fld in CRIF_COLUMN_MAPPING.items():
                if fld in row and col not in row:
                    row[col] = row.pop(fld)
            parsed.append(_parse_crif_row(row, i))
    return crif_to_sensitivities(parsed)


def _convert_crif_record(record: CRIFRecord) -> Optional[AnySensitivity]:
    """Convert a single CRIF record to a Sensitivity object."""
    sensitivity_type = record.get_sensitivity_type()
    if sensitivity_type is None:
        return None

    risk_class = sensitivity_type.risk_class
    margin_type = sensitivity_type.margin_type
    product_class = record.get_product_class()

    common: Dict[str, Any] = dict(
        trade_id=record.trade_id,
        amount=record.amount,
        amount_currency=record.amount_currency,
        product_class=product_class,
    )

    # Parse tenor from label1 if present
    tenor = _parse_tenor(record.label1, risk_class)

    # Interest Rate
    if risk_class == RiskClass.INTEREST_RATE:
        if sensitivity_type == SensitivityType.RISK_INFLATION:
            return IRInflationDeltaSensitivity(currency=record.qualifier, **common)
        elif sensitivity_type == SensitivityType.RISK_XCCY_BASIS:
            return IRXCcyBasisSensitivity(currency=record.qualifier, **common)
        elif sensitivity_type == SensitivityType.RISK_INFLATION_VOL:
            return IRVegaSensitivity(
                currency=record.qualifier,
                option_tenor=tenor,
                is_inflation=True,
                **common,
            )
        elif margin_type == MarginType.DELTA:
            sub_curve = _parse_sub_curve(record.label2)
            return IRDeltaSensitivity(
                currency=record.qualifier,
                tenor=tenor,
                sub_curve=sub_curve,
                **common,
            )
        elif margin_type == MarginType.VEGA:
            return IRVegaSensitivity(
                currency=record.qualifier,
                option_tenor=tenor,
                **common,
            )

    # Credit
    elif risk_class in (RiskClass.CREDIT_QUALIFYING, RiskClass.CREDIT_NON_QUALIFYING):
        is_qualifying = risk_class == RiskClass.CREDIT_QUALIFYING
        bucket_number = _parse_bucket_number(record.bucket)

        if sensitivity_type == SensitivityType.RISK_BASE_CORR:
            return BaseCorrSensitivity(index_name=record.qualifier, **common)
        elif margin_type == MarginType.DELTA:
            return CreditDeltaSensitivity(
                issuer=record.qualifier,
                bucket_number=bucket_number,
                tenor=tenor,
                is_qualifying=is_qualifying,
                **common,
            )
        elif margin_type == MarginType.VEGA:
            return CreditVegaSensitivity(
                issuer=record.qualifier,
                bucket_number=bucket_number,
                option_tenor=tenor,
                is_qualifying=is_qualifying,
                **common,
            )

    # Equity
    elif risk_class == RiskClass.EQUITY:
        bucket_number = _parse_bucket_number(record.bucket)
        if margin_type == MarginType.DELTA:
            return EquityDeltaSensitivity(
                issuer=record.qualifier,
                bucket_number=bucket_number,
                **common,
            )
        elif margin_type == MarginType.VEGA:
            return EquityVegaSensitivity(
                issuer=record.qualifier,
                bucket_number=bucket_number,
                option_tenor=tenor,
                **common,
            )

    # Commodity
    elif risk_class == RiskClass.COMMODITY:
        bucket_number = _parse_bucket_number(record.bucket)
        if margin_type == MarginType.DELTA:
            return CommodityDeltaSensitivity(
                commodity_name=record.qualifier,
                bucket_number=bucket_number,
                **common,
            )
        elif margin_type == MarginType.VEGA:
            return CommodityVegaSensitivity(
                commodity_name=record.qualifier,
                bucket_number=bucket_number,
                option_tenor=tenor,
                **common,
            )

    # FX
    elif risk_class == RiskClass.FX:
        if margin_type == MarginType.DELTA:
            return FXDeltaSensitivity(currency=record.qualifier, **common)
        elif margin_type == MarginType.VEGA:
            return FXVegaSensitivity(
                currency_pair=record.qualifier,
                option_tenor=tenor,
                **common,
            )

    return None


def _parse_tenor(label: str, risk_class: RiskClass) -> float:
    """Parse a tenor label to a numeric value in years."""
    if not label:
        return 1.0  # Default tenor
    
    label = label.lower().strip()
    
    # Check IR tenor labels
    if risk_class == RiskClass.INTEREST_RATE:
        if label in IR_TENOR_LABELS:
            idx = IR_TENOR_LABELS.index(label)
            return IR_TENORS[idx]
    
    # Check Credit tenor labels
    if risk_class in (RiskClass.CREDIT_QUALIFYING, RiskClass.CREDIT_NON_QUALIFYING):
        if label in CREDIT_TENOR_LABELS:
            idx = CREDIT_TENOR_LABELS.index(label)
            return CREDIT_TENORS[idx]
    
    # Try to parse numeric value
    try:
        # Handle formats like "5y", "6m", "2w"
        if label.endswith("y"):
            return float(label[:-1])
        elif label.endswith("m"):
            return float(label[:-1]) / 12.0
        elif label.endswith("w"):
            return float(label[:-1]) / 52.0
        elif label.endswith("d"):
            return float(label[:-1]) / 365.0
        else:
            return float(label)
    except ValueError:
        return 1.0  # Default if parsing fails


def _parse_sub_curve(label: str) -> IRSubCurve:
    """Parse a sub-curve label to IRSubCurve enum."""
    if not label:
        return IRSubCurve.OIS
    
    label_upper = label.upper().strip()
    
    for sub_curve in IRSubCurve:
        if sub_curve.value.upper() == label_upper:
            return sub_curve
    
    # Common aliases
    aliases = {
        "LIBOR1M": IRSubCurve.LIBOR_1M,
        "LIBOR3M": IRSubCurve.LIBOR_3M,
        "LIBOR6M": IRSubCurve.LIBOR_6M,
        "LIBOR12M": IRSubCurve.LIBOR_12M,
        "LIBOR": IRSubCurve.LIBOR_3M,  # Default LIBOR to 3M
        "SOFR": IRSubCurve.OIS,
        "ESTR": IRSubCurve.OIS,
        "SONIA": IRSubCurve.OIS,
    }
    
    return aliases.get(label_upper, IRSubCurve.OIS)


def _parse_bucket_number(bucket: str) -> int:
    """Parse a bucket string to an integer."""
    if not bucket:
        return 1
    
    try:
        return int(bucket)
    except ValueError:
        # Handle "Residual" or similar
        if bucket.lower() == "residual":
            return -1
        return 1


def sensitivities_to_crif(
    collection: SensitivityCollection,
    valuation_date: date,
) -> List[CRIFRecord]:
    """Convert Sensitivity objects to CRIF records.
    
    Args:
        collection: SensitivityCollection to convert.
        valuation_date: Valuation date for the CRIF records.
        
    Returns:
        List of CRIFRecord objects.
    """
    records: List[CRIFRecord] = []
    
    for sensitivity in collection:
        record = _convert_sensitivity_to_crif(sensitivity, valuation_date)
        if record:
            records.append(record)
    
    return records


def _convert_sensitivity_to_crif(
    sensitivity: AnySensitivity,
    valuation_date: date,
) -> Optional[CRIFRecord]:
    """Convert a single Sensitivity to a CRIF record."""
    # Determine risk_type
    risk_type = _get_risk_type(sensitivity)
    if not risk_type:
        return None
    
    # Build record
    record = CRIFRecord(
        trade_id=sensitivity.trade_id,
        valuation_date=valuation_date,
        risk_type=risk_type,
        qualifier=sensitivity.qualifier,
        bucket=str(sensitivity.bucket),
        amount=sensitivity.amount,
        amount_currency=sensitivity.amount_currency,
        product_class=sensitivity.effective_product_class.value,
    )

    # Tenor / vertex labels
    if isinstance(sensitivity, IRDeltaSensitivity):
        record.label1 = sensitivity.vertex
        record.label2 = sensitivity.sub_curve.value
    elif isinstance(sensitivity, IRVegaSensitivity):
        record.label1 = sensitivity.vertex
    elif isinstance(sensitivity, (CreditDeltaSensitivity, CreditVegaSensitivity)):
        record.label1 = sensitivity.vertex
    elif hasattr(sensitivity, "vertex"):
        record.label1 = getattr(sensitivity, "vertex")

    return record


def _get_risk_type(sensitivity: AnySensitivity) -> str:
    """Get the CRIF risk_type for a sensitivity."""
    risk_class = sensitivity.risk_class
    margin_type = sensitivity.margin_type

    if isinstance(sensitivity, IRInflationDeltaSensitivity):
        return SensitivityType.RISK_INFLATION.value
    if isinstance(sensitivity, IRXCcyBasisSensitivity):
        return SensitivityType.RISK_XCCY_BASIS.value
    if isinstance(sensitivity, IRVegaSensitivity) and sensitivity.is_inflation:
        return SensitivityType.RISK_INFLATION_VOL.value

    if risk_class == RiskClass.INTEREST_RATE:
        if margin_type == MarginType.DELTA:
            return SensitivityType.RISK_IR_CURVE.value
        elif margin_type == MarginType.VEGA:
            return SensitivityType.RISK_IR_VOL.value
    elif risk_class == RiskClass.CREDIT_QUALIFYING:
        if margin_type == MarginType.DELTA:
            return SensitivityType.RISK_CREDIT_Q.value
        elif margin_type == MarginType.VEGA:
            return SensitivityType.RISK_CREDIT_VOL.value
        elif margin_type == MarginType.BASE_CORR:
            return SensitivityType.RISK_BASE_CORR.value
    elif risk_class == RiskClass.CREDIT_NON_QUALIFYING:
        if margin_type == MarginType.DELTA:
            return SensitivityType.RISK_CREDIT_NQ.value
        elif margin_type == MarginType.VEGA:
            return SensitivityType.RISK_CREDIT_NQ_VOL.value
    elif risk_class == RiskClass.EQUITY:
        if margin_type == MarginType.DELTA:
            return SensitivityType.RISK_EQUITY.value
        elif margin_type == MarginType.VEGA:
            return SensitivityType.RISK_EQUITY_VOL.value
    elif risk_class == RiskClass.COMMODITY:
        if margin_type == MarginType.DELTA:
            return SensitivityType.RISK_COMMODITY.value
        elif margin_type == MarginType.VEGA:
            return SensitivityType.RISK_COMMODITY_VOL.value
    elif risk_class == RiskClass.FX:
        if margin_type == MarginType.DELTA:
            return SensitivityType.RISK_FX.value
        elif margin_type == MarginType.VEGA:
            return SensitivityType.RISK_FX_VOL.value
    
    return ""


def _tenor_to_label(tenor: float) -> str:
    """Convert a numeric tenor to a label string."""
    if tenor < 1/12:  # Less than 1 month
        weeks = int(tenor * 52)
        return f"{weeks}w"
    elif tenor < 1:  # Less than 1 year
        months = int(tenor * 12)
        return f"{months}m"
    else:
        years = int(tenor)
        return f"{years}y"


def write_crif_csv(
    records: List[CRIFRecord],
    output: Union[str, Path],
    columns: Optional[List[str]] = None,
) -> None:
    """Write CRIF records to a CSV file.
    
    Args:
        records: List of CRIFRecord objects to write.
        output: Output file path.
        columns: Optional list of columns to include. Defaults to standard CRIF columns.
    """
    if columns is None:
        columns = list(CRIF_COLUMN_MAPPING.keys())
    
    # Reverse mapping for field to column
    field_to_column = {v: k for k, v in CRIF_COLUMN_MAPPING.items()}
    
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for record in records:
            row = {}
            for col in columns:
                field = CRIF_COLUMN_MAPPING.get(col)
                if field:
                    value = getattr(record, field, None)
                    if value is not None:
                        if isinstance(value, date):
                            value = value.strftime("%Y-%m-%d")
                        row[col] = value
                    else:
                        row[col] = ""
            writer.writerow(row)
