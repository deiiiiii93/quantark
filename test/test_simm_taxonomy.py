"""
Tests for SIMM Taxonomy Module.

Tests for enums, tenors, currency classifications, and bucket definitions.
"""
import pytest

from quantark.simm.taxonomy import (
    # Enums
    CurrencyVolatility,
    IRSubCurve,
    MarginType,
    ProductClass,
    RiskClass,
    SensitivityType,
    # Tenors
    CREDIT_TENOR_LABELS,
    CREDIT_TENORS,
    IR_TENOR_LABELS,
    IR_TENORS,
    VEGA_TENOR_LABELS,
    VEGA_TENORS,
    # Currency classifications
    HIGH_VOL_CURRENCIES,
    LOW_VOL_CURRENCIES,
    get_currency_volatility,
    # Bucket definitions
    COMMODITY_BUCKETS,
    CREDIT_NON_QUALIFYING_BUCKETS,
    CREDIT_NON_QUALIFYING_RESIDUAL_BUCKET,
    CREDIT_QUALIFYING_BUCKETS,
    CREDIT_QUALIFYING_RESIDUAL_BUCKET,
    EQUITY_BUCKETS,
    EQUITY_RESIDUAL_BUCKET,
    FX_BUCKET,
    CommodityBucket,
    CreditNonQualifyingBucket,
    CreditQualifyingBucket,
    EquityBucket,
    FXBucket,
    IRBucket,
)


class TestRiskClass:
    """Tests for RiskClass enum."""
    
    def test_all_risk_classes_defined(self):
        """All six SIMM risk classes should be defined."""
        assert len(RiskClass) == 6
        assert RiskClass.INTEREST_RATE.value == "IR"
        assert RiskClass.CREDIT_QUALIFYING.value == "CreditQ"
        assert RiskClass.CREDIT_NON_QUALIFYING.value == "CreditNQ"
        assert RiskClass.EQUITY.value == "Equity"
        assert RiskClass.COMMODITY.value == "Commodity"
        assert RiskClass.FX.value == "FX"
    
    def test_string_representation(self):
        """String representation should return the value."""
        assert str(RiskClass.INTEREST_RATE) == "IR"
        assert str(RiskClass.FX) == "FX"


class TestProductClass:
    """Tests for ProductClass enum."""
    
    def test_all_product_classes_defined(self):
        """All four SIMM product classes should be defined."""
        assert len(ProductClass) == 4
        assert ProductClass.RATES_FX.value == "RatesFX"
        assert ProductClass.CREDIT.value == "Credit"
        assert ProductClass.EQUITY.value == "Equity"
        assert ProductClass.COMMODITY.value == "Commodity"


class TestMarginType:
    """Tests for MarginType enum."""
    
    def test_all_margin_types_defined(self):
        """All four margin types should be defined."""
        assert len(MarginType) == 4
        assert MarginType.DELTA.value == "Delta"
        assert MarginType.VEGA.value == "Vega"
        assert MarginType.CURVATURE.value == "Curvature"
        assert MarginType.BASE_CORR.value == "BaseCorr"


class TestSensitivityType:
    """Tests for SensitivityType enum."""
    
    def test_risk_class_mapping(self):
        """SensitivityType should correctly map to RiskClass."""
        assert SensitivityType.RISK_IR_CURVE.risk_class == RiskClass.INTEREST_RATE
        assert SensitivityType.RISK_IR_VOL.risk_class == RiskClass.INTEREST_RATE
        assert SensitivityType.RISK_CREDIT_Q.risk_class == RiskClass.CREDIT_QUALIFYING
        assert SensitivityType.RISK_CREDIT_NQ.risk_class == RiskClass.CREDIT_NON_QUALIFYING
        assert SensitivityType.RISK_EQUITY.risk_class == RiskClass.EQUITY
        assert SensitivityType.RISK_COMMODITY.risk_class == RiskClass.COMMODITY
        assert SensitivityType.RISK_FX.risk_class == RiskClass.FX
    
    def test_margin_type_mapping(self):
        """SensitivityType should correctly map to MarginType."""
        # Delta types
        assert SensitivityType.RISK_IR_CURVE.margin_type == MarginType.DELTA
        assert SensitivityType.RISK_CREDIT_Q.margin_type == MarginType.DELTA
        assert SensitivityType.RISK_FX.margin_type == MarginType.DELTA
        
        # Vega types
        assert SensitivityType.RISK_IR_VOL.margin_type == MarginType.VEGA
        assert SensitivityType.RISK_CREDIT_VOL.margin_type == MarginType.VEGA
        assert SensitivityType.RISK_FX_VOL.margin_type == MarginType.VEGA
        
        # BaseCorr
        assert SensitivityType.RISK_BASE_CORR.margin_type == MarginType.BASE_CORR


class TestIRSubCurve:
    """Tests for IRSubCurve enum."""
    
    def test_all_sub_curves_defined(self):
        """All IR sub-curves should be defined."""
        assert IRSubCurve.OIS.value == "OIS"
        assert IRSubCurve.LIBOR_1M.value == "Libor1m"
        assert IRSubCurve.LIBOR_3M.value == "Libor3m"
        assert IRSubCurve.LIBOR_6M.value == "Libor6m"
        assert IRSubCurve.LIBOR_12M.value == "Libor12m"
        assert IRSubCurve.PRIME.value == "Prime"
        assert IRSubCurve.MUNICIPAL.value == "Municipal"


class TestTenors:
    """Tests for tenor definitions."""
    
    def test_ir_tenors_count(self):
        """IR tenors should have 12 points."""
        assert len(IR_TENORS) == 12
        assert len(IR_TENOR_LABELS) == 12
    
    def test_ir_tenor_values(self):
        """IR tenor values should be correct."""
        assert IR_TENORS[0] == pytest.approx(0.0384, rel=0.01)  # 2w
        assert IR_TENORS[4] == 1.0  # 1y
        assert IR_TENORS[-1] == 30.0  # 30y
    
    def test_ir_tenor_labels(self):
        """IR tenor labels should be correct."""
        assert IR_TENOR_LABELS[0] == "2w"
        assert IR_TENOR_LABELS[4] == "1y"
        assert IR_TENOR_LABELS[-1] == "30y"
    
    def test_credit_tenors_count(self):
        """Credit tenors should have 5 points."""
        assert len(CREDIT_TENORS) == 5
        assert len(CREDIT_TENOR_LABELS) == 5
    
    def test_credit_tenor_values(self):
        """Credit tenor values should be correct."""
        assert CREDIT_TENORS == (1.0, 2.0, 3.0, 5.0, 10.0)
    
    def test_vega_tenors(self):
        """Vega tenors should be defined."""
        assert len(VEGA_TENORS) == 5
        assert len(VEGA_TENOR_LABELS) == 5


class TestCurrencyClassification:
    """Tests for currency volatility classifications."""
    
    def test_low_vol_currencies(self):
        """Major currencies should be low volatility."""
        assert "USD" in LOW_VOL_CURRENCIES
        assert "EUR" in LOW_VOL_CURRENCIES
        assert "GBP" in LOW_VOL_CURRENCIES
        assert "JPY" not in LOW_VOL_CURRENCIES  # JPY is regular vol
    
    def test_high_vol_currencies(self):
        """Emerging market currencies should be high volatility."""
        assert "BRL" in HIGH_VOL_CURRENCIES
        assert "TRY" in HIGH_VOL_CURRENCIES
        assert "ZAR" in HIGH_VOL_CURRENCIES
    
    def test_get_currency_volatility(self):
        """get_currency_volatility should return correct classification."""
        assert get_currency_volatility("USD") == CurrencyVolatility.LOW
        assert get_currency_volatility("BRL") == CurrencyVolatility.HIGH
        assert get_currency_volatility("JPY") == CurrencyVolatility.REGULAR
        
        # Case insensitive
        assert get_currency_volatility("usd") == CurrencyVolatility.LOW


class TestIRBucket:
    """Tests for IRBucket dataclass."""
    
    def test_from_currency(self):
        """IRBucket.from_currency should create bucket with correct volatility."""
        usd_bucket = IRBucket.from_currency("USD")
        assert usd_bucket.currency == "USD"
        assert usd_bucket.volatility == CurrencyVolatility.LOW
        
        brl_bucket = IRBucket.from_currency("brl")
        assert brl_bucket.currency == "BRL"
        assert brl_bucket.volatility == CurrencyVolatility.HIGH
    
    def test_frozen(self):
        """IRBucket should be immutable."""
        bucket = IRBucket.from_currency("USD")
        with pytest.raises(AttributeError):
            bucket.currency = "EUR"


class TestCreditQualifyingBuckets:
    """Tests for Credit Qualifying bucket definitions."""
    
    def test_bucket_count(self):
        """Should have 12 CQ buckets."""
        assert len(CREDIT_QUALIFYING_BUCKETS) == 12
    
    def test_ig_buckets(self):
        """Buckets 1-6 should be Investment Grade."""
        for i in range(1, 7):
            assert CREDIT_QUALIFYING_BUCKETS[i].credit_quality == "IG"
    
    def test_hy_buckets(self):
        """Buckets 7-12 should be High Yield."""
        for i in range(7, 13):
            assert CREDIT_QUALIFYING_BUCKETS[i].credit_quality == "HY/NR"
    
    def test_residual_bucket(self):
        """Residual bucket should be defined."""
        assert CREDIT_QUALIFYING_RESIDUAL_BUCKET.bucket_number == -1
        assert CREDIT_QUALIFYING_RESIDUAL_BUCKET.credit_quality == "Residual"


class TestCreditNonQualifyingBuckets:
    """Tests for Credit Non-Qualifying bucket definitions."""
    
    def test_bucket_count(self):
        """Should have 2 CNQ buckets."""
        assert len(CREDIT_NON_QUALIFYING_BUCKETS) == 2
    
    def test_buckets(self):
        """CNQ buckets should have correct credit quality."""
        assert CREDIT_NON_QUALIFYING_BUCKETS[1].credit_quality == "IG"
        assert CREDIT_NON_QUALIFYING_BUCKETS[2].credit_quality == "HY/NR"
    
    def test_residual_bucket(self):
        """Residual bucket should be defined."""
        assert CREDIT_NON_QUALIFYING_RESIDUAL_BUCKET.bucket_number == -1


class TestEquityBuckets:
    """Tests for Equity bucket definitions."""
    
    def test_bucket_count(self):
        """Should have 12 Equity buckets."""
        assert len(EQUITY_BUCKETS) == 12
    
    def test_emerging_large_cap(self):
        """Buckets 1-4 should be Large Emerging."""
        for i in range(1, 5):
            assert EQUITY_BUCKETS[i].size == "Large"
            assert EQUITY_BUCKETS[i].region == "Emerging"
    
    def test_developed_large_cap(self):
        """Buckets 5-8 should be Large Developed."""
        for i in range(5, 9):
            assert EQUITY_BUCKETS[i].size == "Large"
            assert EQUITY_BUCKETS[i].region == "Developed"
    
    def test_small_cap(self):
        """Buckets 9-10 should be Small cap."""
        assert EQUITY_BUCKETS[9].size == "Small"
        assert EQUITY_BUCKETS[10].size == "Small"
    
    def test_residual_bucket(self):
        """Residual bucket should be defined."""
        assert EQUITY_RESIDUAL_BUCKET.bucket_number == -1


class TestCommodityBuckets:
    """Tests for Commodity bucket definitions."""
    
    def test_bucket_count(self):
        """Should have 17 Commodity buckets."""
        assert len(COMMODITY_BUCKETS) == 17
    
    def test_specific_buckets(self):
        """Specific commodity buckets should be correct."""
        assert COMMODITY_BUCKETS[1].commodity_type == "Coal"
        assert COMMODITY_BUCKETS[2].commodity_type == "Crude oil"
        assert COMMODITY_BUCKETS[11].commodity_type == "Base metals"
        assert COMMODITY_BUCKETS[12].commodity_type == "Precious metals"


class TestFXBucket:
    """Tests for FX bucket definition."""
    
    def test_single_bucket(self):
        """FX should have single bucket."""
        assert FX_BUCKET.bucket_number == 1
