"""
Tests for SIMM CRIF Module.

Tests for CRIF parsing, validation, and conversion.
"""
from datetime import date
from io import StringIO

import pytest

from quantark.simm.crif import (
    CRIFHeader,
    CRIFRecord,
    CRIFValidationError,
    crif_to_sensitivities,
    parse_crif_csv,
    sensitivities_to_crif,
)
from quantark.simm.sensitivity import (
    IRDeltaSensitivity,
    FXDeltaSensitivity,
    CreditDeltaSensitivity,
    EquityDeltaSensitivity,
    SensitivityCollection,
)
from quantark.simm.taxonomy import (
    IRSubCurve,
    MarginType,
    ProductClass,
    RiskClass,
    SensitivityType,
)


class TestCRIFRecord:
    """Tests for CRIFRecord dataclass."""
    
    def test_basic_creation(self):
        """Basic CRIF record creation."""
        record = CRIFRecord(
            trade_id="TRADE001",
            valuation_date=date(2024, 1, 15),
            risk_type="Risk_IRCurve",
            qualifier="USD",
            bucket="1",
            label1="5y",
            label2="OIS",
            amount=150000.0,
            amount_currency="USD"
        )
        
        assert record.trade_id == "TRADE001"
        assert record.valuation_date == date(2024, 1, 15)
        assert record.risk_type == "Risk_IRCurve"
        assert record.qualifier == "USD"
        assert record.amount == 150000.0
    
    def test_get_sensitivity_type(self):
        """get_sensitivity_type should return correct enum."""
        record = CRIFRecord(
            trade_id="T1",
            valuation_date=date(2024, 1, 1),
            risk_type="Risk_IRCurve",
            qualifier="USD",
            bucket="1",
            amount=100.0,
        )
        assert record.get_sensitivity_type() == SensitivityType.RISK_IR_CURVE
    
    def test_get_sensitivity_type_unknown(self):
        """Unknown risk_type should return None."""
        record = CRIFRecord(
            trade_id="T1",
            valuation_date=date(2024, 1, 1),
            risk_type="Unknown_Type",
            qualifier="USD",
            bucket="1",
            amount=100.0,
        )
        assert record.get_sensitivity_type() is None
    
    def test_get_risk_class(self):
        """get_risk_class should infer from risk_type."""
        ir_record = CRIFRecord(
            trade_id="T1",
            valuation_date=date(2024, 1, 1),
            risk_type="Risk_IRCurve",
            qualifier="USD",
            bucket="1",
            amount=100.0,
        )
        assert ir_record.get_risk_class() == RiskClass.INTEREST_RATE
        
        fx_record = CRIFRecord(
            trade_id="T2",
            valuation_date=date(2024, 1, 1),
            risk_type="Risk_FX",
            qualifier="EURUSD",
            bucket="",
            amount=100.0,
        )
        assert fx_record.get_risk_class() == RiskClass.FX
    
    def test_get_margin_type(self):
        """get_margin_type should infer from risk_type."""
        delta_record = CRIFRecord(
            trade_id="T1",
            valuation_date=date(2024, 1, 1),
            risk_type="Risk_IRCurve",
            qualifier="USD",
            bucket="1",
            amount=100.0,
        )
        assert delta_record.get_margin_type() == MarginType.DELTA
        
        vega_record = CRIFRecord(
            trade_id="T2",
            valuation_date=date(2024, 1, 1),
            risk_type="Risk_IRVol",
            qualifier="USD",
            bucket="1",
            amount=100.0,
        )
        assert vega_record.get_margin_type() == MarginType.VEGA
    
    def test_get_product_class(self):
        """get_product_class should infer from risk_class."""
        ir_record = CRIFRecord(
            trade_id="T1",
            valuation_date=date(2024, 1, 1),
            risk_type="Risk_IRCurve",
            qualifier="USD",
            bucket="1",
            amount=100.0,
        )
        assert ir_record.get_product_class() == ProductClass.RATES_FX
        
        equity_record = CRIFRecord(
            trade_id="T2",
            valuation_date=date(2024, 1, 1),
            risk_type="Risk_Equity",
            qualifier="AAPL",
            bucket="8",
            amount=100.0,
        )
        assert equity_record.get_product_class() == ProductClass.EQUITY


class TestCRIFHeader:
    """Tests for CRIFHeader dataclass."""
    
    def test_basic_creation(self):
        """Basic CRIF header creation."""
        header = CRIFHeader(
            valuation_date=date(2024, 1, 15),
            reporting_entity="BANK001",
            counterparty="CP001"
        )
        
        assert header.valuation_date == date(2024, 1, 15)
        assert header.reporting_entity == "BANK001"
        assert header.im_model == "SIMM"
        assert header.crif_version == "2.0"


class TestParseCRIFCSV:
    """Tests for parse_crif_csv function."""
    
    def test_parse_basic_csv(self):
        """Parse a basic CRIF CSV."""
        csv_data = """TradeID,ValuationDate,RiskType,Qualifier,Bucket,Label1,Label2,Amount,AmountCurrency
TRADE001,2024-01-15,Risk_IRCurve,USD,1,5y,OIS,150000,USD
TRADE002,2024-01-15,Risk_FX,EURUSD,,,,50000,USD"""
        
        records, warnings = parse_crif_csv(StringIO(csv_data))
        
        assert len(records) == 2
        assert len(warnings) == 0
        
        assert records[0].trade_id == "TRADE001"
        assert records[0].risk_type == "Risk_IRCurve"
        assert records[0].amount == 150000.0
        
        assert records[1].trade_id == "TRADE002"
        assert records[1].risk_type == "Risk_FX"
    
    def test_parse_with_date_formats(self):
        """Parse CSV with different date formats."""
        # YYYY-MM-DD format
        csv1 = """TradeID,ValuationDate,RiskType,Qualifier,Bucket,Amount,AmountCurrency
T1,2024-01-15,Risk_FX,EURUSD,,100,USD"""
        records1, _ = parse_crif_csv(StringIO(csv1))
        assert records1[0].valuation_date == date(2024, 1, 15)
        
        # YYYYMMDD format
        csv2 = """TradeID,ValuationDate,RiskType,Qualifier,Bucket,Amount,AmountCurrency
T1,20240115,Risk_FX,EURUSD,,100,USD"""
        records2, _ = parse_crif_csv(StringIO(csv2))
        assert records2[0].valuation_date == date(2024, 1, 15)
    
    def test_parse_with_missing_columns(self):
        """Parse CSV with missing optional columns."""
        csv_data = """TradeID,ValuationDate,RiskType,Qualifier,Amount,AmountCurrency
T1,2024-01-15,Risk_FX,EURUSD,100,USD"""
        
        records, warnings = parse_crif_csv(StringIO(csv_data))
        assert len(records) == 1
        assert records[0].bucket == ""  # Default value
    
    def test_parse_strict_mode(self):
        """Strict mode should raise on invalid risk_type."""
        csv_data = """TradeID,ValuationDate,RiskType,Qualifier,Bucket,Amount,AmountCurrency
T1,2024-01-15,Invalid_Type,USD,1,100,USD"""
        
        with pytest.raises(CRIFValidationError):
            parse_crif_csv(StringIO(csv_data), strict=True)
    
    def test_parse_non_strict_collects_warnings(self):
        """Non-strict mode should collect warnings."""
        csv_data = """TradeID,ValuationDate,RiskType,Qualifier,Bucket,Amount,AmountCurrency
T1,2024-01-15,Invalid_Type,USD,1,100,USD"""
        
        records, warnings = parse_crif_csv(StringIO(csv_data), strict=False)
        assert len(records) == 1
        assert len(warnings) > 0


class TestCRIFToSensitivities:
    """Tests for crif_to_sensitivities function."""
    
    def test_convert_ir_delta(self):
        """Convert IR Delta CRIF record to sensitivity."""
        records = [
            CRIFRecord(
                trade_id="T1",
                valuation_date=date(2024, 1, 1),
                risk_type="Risk_IRCurve",
                qualifier="USD",
                bucket="1",
                label1="5y",
                label2="OIS",
                amount=100000.0,
                amount_currency="USD",
            )
        ]
        
        collection = crif_to_sensitivities(records)
        
        assert len(collection) == 1
        sens = collection.sensitivities[0]
        assert isinstance(sens, IRDeltaSensitivity)
        assert sens.currency == "USD"
        assert sens.tenor == 5.0
        assert sens.sub_curve == IRSubCurve.OIS
        assert sens.amount == 100000.0
    
    def test_convert_fx_delta(self):
        """Convert FX Delta CRIF record to sensitivity."""
        records = [
            CRIFRecord(
                trade_id="T1",
                valuation_date=date(2024, 1, 1),
                risk_type="Risk_FX",
                qualifier="EUR",
                bucket="",
                amount=50000.0,
                amount_currency="USD",
            )
        ]
        
        collection = crif_to_sensitivities(records)
        
        assert len(collection) == 1
        sens = collection.sensitivities[0]
        assert isinstance(sens, FXDeltaSensitivity)
        assert sens.currency == "EUR"
    
    def test_convert_credit_delta(self):
        """Convert Credit Delta CRIF record to sensitivity."""
        records = [
            CRIFRecord(
                trade_id="T1",
                valuation_date=date(2024, 1, 1),
                risk_type="Risk_CreditQ",
                qualifier="ISSUER001",
                bucket="2",
                label1="5y",
                amount=75000.0,
                amount_currency="USD",
            )
        ]
        
        collection = crif_to_sensitivities(records)
        
        assert len(collection) == 1
        sens = collection.sensitivities[0]
        assert isinstance(sens, CreditDeltaSensitivity)
        assert sens.issuer == "ISSUER001"
        assert sens.bucket_number == 2
        assert sens.is_qualifying is True
    
    def test_convert_equity_delta(self):
        """Convert Equity Delta CRIF record to sensitivity."""
        records = [
            CRIFRecord(
                trade_id="T1",
                valuation_date=date(2024, 1, 1),
                risk_type="Risk_Equity",
                qualifier="AAPL",
                bucket="8",
                amount=25000.0,
                amount_currency="USD",
            )
        ]
        
        collection = crif_to_sensitivities(records)
        
        assert len(collection) == 1
        sens = collection.sensitivities[0]
        assert isinstance(sens, EquityDeltaSensitivity)
        assert sens.issuer == "AAPL"
        assert sens.bucket_number == 8


class TestSensitivitiesToCRIF:
    """Tests for sensitivities_to_crif function."""
    
    def test_convert_ir_delta(self):
        """Convert IR Delta sensitivity to CRIF record."""
        collection = SensitivityCollection()
        collection.add(IRDeltaSensitivity(
            trade_id="T1",
            amount=100000.0,
            amount_currency="USD",
            currency="USD",
            tenor=5.0,
            sub_curve=IRSubCurve.OIS,
        ))
        
        records = sensitivities_to_crif(collection, date(2024, 1, 1))
        
        assert len(records) == 1
        record = records[0]
        assert record.trade_id == "T1"
        assert record.risk_type == "Risk_IRCurve"
        assert record.qualifier == "USD"
        assert record.amount == 100000.0
    
    def test_convert_fx_delta(self):
        """Convert FX Delta sensitivity to CRIF record."""
        collection = SensitivityCollection()
        collection.add(FXDeltaSensitivity(
            trade_id="T1",
            amount=50000.0,
            amount_currency="USD",
            currency="EUR",
        ))
        
        records = sensitivities_to_crif(collection, date(2024, 1, 1))
        
        assert len(records) == 1
        record = records[0]
        assert record.risk_type == "Risk_FX"
        assert record.qualifier == "EUR"


class TestRoundTrip:
    """Tests for CRIF round-trip conversion."""
    
    def test_round_trip_ir_delta(self):
        """IR Delta should survive round-trip conversion."""
        original = IRDeltaSensitivity(
            trade_id="T1",
            amount=100000.0,
            amount_currency="USD",
            currency="EUR",
            tenor=10.0,
            sub_curve=IRSubCurve.LIBOR_3M,
        )
        
        # Convert to CRIF
        collection = SensitivityCollection()
        collection.add(original)
        crif_records = sensitivities_to_crif(collection, date(2024, 1, 1))
        
        # Convert back
        result_collection = crif_to_sensitivities(crif_records)
        
        assert len(result_collection) == 1
        result = result_collection.sensitivities[0]
        assert isinstance(result, IRDeltaSensitivity)
        assert result.trade_id == original.trade_id
        assert result.amount == original.amount
        assert result.currency == original.currency
