"""Tests for SIMM taxonomy (ISDA SIMM v2.6)."""
import pytest

from quantark.simm.taxonomy import (
    CurrencyVolatility,
    FXVolatilityGroup,
    IRConcentrationGroup,
    IRSubCurve,
    MarginType,
    ProductClass,
    RiskClass,
    SensitivityType,
    CREDIT_TENOR_LABELS,
    CREDIT_TENORS,
    DEFAULT_PRODUCT_CLASS,
    IR_TENOR_LABELS,
    IR_TENORS,
    TENOR_LABEL_DAYS,
    credit_tenor_to_vertex_label,
    get_currency_volatility,
    get_fx_concentration_category,
    get_fx_volatility_group,
    get_ir_concentration_group,
    is_residual_bucket,
    normalize_tenor_label,
    tenor_to_vertex_label,
    CREDIT_QUALIFYING_BUCKETS,
    CREDIT_NON_QUALIFYING_BUCKETS,
    EQUITY_BUCKETS,
    COMMODITY_BUCKETS,
)


class TestEnums:
    def test_six_risk_classes(self):
        assert len(RiskClass) == 6

    def test_four_product_classes(self):
        assert len(ProductClass) == 4

    def test_four_margin_types(self):
        assert len(MarginType) == 4

    def test_sensitivity_type_risk_class(self):
        assert SensitivityType.RISK_IR_CURVE.risk_class == RiskClass.INTEREST_RATE
        assert SensitivityType.RISK_INFLATION.risk_class == RiskClass.INTEREST_RATE
        assert SensitivityType.RISK_XCCY_BASIS.risk_class == RiskClass.INTEREST_RATE
        assert SensitivityType.RISK_BASE_CORR.risk_class == RiskClass.CREDIT_QUALIFYING
        assert SensitivityType.RISK_FX_VOL.risk_class == RiskClass.FX

    def test_sensitivity_type_margin_type(self):
        assert SensitivityType.RISK_IR_CURVE.margin_type == MarginType.DELTA
        assert SensitivityType.RISK_IR_VOL.margin_type == MarginType.VEGA
        assert SensitivityType.RISK_BASE_CORR.margin_type == MarginType.BASE_CORR

    def test_default_product_class(self):
        assert DEFAULT_PRODUCT_CLASS[RiskClass.INTEREST_RATE] == ProductClass.RATES_FX
        assert DEFAULT_PRODUCT_CLASS[RiskClass.FX] == ProductClass.RATES_FX
        assert DEFAULT_PRODUCT_CLASS[RiskClass.CREDIT_QUALIFYING] == ProductClass.CREDIT
        assert DEFAULT_PRODUCT_CLASS[RiskClass.EQUITY] == ProductClass.EQUITY
        assert DEFAULT_PRODUCT_CLASS[RiskClass.COMMODITY] == ProductClass.COMMODITY


class TestTenors:
    def test_twelve_ir_vertices(self):
        assert len(IR_TENORS) == 12
        assert len(IR_TENOR_LABELS) == 12

    def test_five_credit_vertices(self):
        assert CREDIT_TENORS == (1.0, 2.0, 3.0, 5.0, 10.0)
        assert CREDIT_TENOR_LABELS == ("1y", "2y", "3y", "5y", "10y")

    def test_tenor_to_vertex_label(self):
        assert tenor_to_vertex_label(5.0) == "5y"
        assert tenor_to_vertex_label(14.0 / 365.0) == "2w"
        assert tenor_to_vertex_label(0.26) == "3m"
        assert tenor_to_vertex_label(50.0) == "30y"

    def test_credit_tenor_to_vertex_label(self):
        assert credit_tenor_to_vertex_label(5.0) == "5y"
        assert credit_tenor_to_vertex_label(0.5) == "1y"
        assert credit_tenor_to_vertex_label(20.0) == "10y"

    def test_normalize_tenor_label(self):
        assert normalize_tenor_label("5yr") == "5y"
        assert normalize_tenor_label("12m") == "1y"
        assert normalize_tenor_label("2W") == "2w"
        with pytest.raises(KeyError):
            normalize_tenor_label("7y")

    def test_tenor_label_days_convention(self):
        # Paragraph 11(a): 12m = 365 days, pro-rata scaling.
        assert TENOR_LABEL_DAYS["2w"] == 14.0
        assert TENOR_LABEL_DAYS["1m"] == pytest.approx(365.0 / 12.0)
        assert TENOR_LABEL_DAYS["1y"] == 365.0
        assert TENOR_LABEL_DAYS["5y"] == 365.0 * 5


class TestCurrencyGroups:
    def test_regular_volatility_currencies(self):
        # Paragraph 33(1).
        for ccy in ("USD", "EUR", "GBP", "CHF", "AUD", "NZD", "CAD", "SEK",
                    "NOK", "DKK", "HKD", "KRW", "SGD", "TWD"):
            assert get_currency_volatility(ccy) == CurrencyVolatility.REGULAR

    def test_low_volatility_is_jpy_only(self):
        # Paragraph 33(2).
        assert get_currency_volatility("JPY") == CurrencyVolatility.LOW

    def test_high_volatility_others(self):
        # Paragraph 33(3).
        for ccy in ("BRL", "MXN", "TRY", "CNY", "INR", "ZAR"):
            assert get_currency_volatility(ccy) == CurrencyVolatility.HIGH

    def test_ir_concentration_groups(self):
        # Paragraph 75.
        assert get_ir_concentration_group("USD") == IRConcentrationGroup.REGULAR_WELL_TRADED
        assert get_ir_concentration_group("EUR") == IRConcentrationGroup.REGULAR_WELL_TRADED
        assert get_ir_concentration_group("GBP") == IRConcentrationGroup.REGULAR_WELL_TRADED
        assert get_ir_concentration_group("AUD") == IRConcentrationGroup.REGULAR_LESS_WELL_TRADED
        assert get_ir_concentration_group("KRW") == IRConcentrationGroup.REGULAR_LESS_WELL_TRADED
        assert get_ir_concentration_group("JPY") == IRConcentrationGroup.LOW_VOLATILITY
        assert get_ir_concentration_group("BRL") == IRConcentrationGroup.HIGH_VOLATILITY

    def test_fx_volatility_groups(self):
        # Paragraphs 67-68: high = BRL, RUB, TRY only.
        for ccy in ("BRL", "RUB", "TRY"):
            assert get_fx_volatility_group(ccy) == FXVolatilityGroup.HIGH
        for ccy in ("USD", "JPY", "MXN", "ZAR"):
            assert get_fx_volatility_group(ccy) == FXVolatilityGroup.REGULAR

    def test_fx_concentration_categories(self):
        # Paragraph 80.
        for ccy in ("USD", "EUR", "JPY", "GBP", "AUD", "CHF", "CAD"):
            assert get_fx_concentration_category(ccy) == 1
        for ccy in ("BRL", "CNY", "HKD", "KRW", "TRY", "ZAR"):
            assert get_fx_concentration_category(ccy) == 2
        for ccy in ("THB", "PLN", "ILS"):
            assert get_fx_concentration_category(ccy) == 3


class TestBuckets:
    def test_credit_qualifying_buckets(self):
        assert len(CREDIT_QUALIFYING_BUCKETS) == 12
        assert CREDIT_QUALIFYING_BUCKETS[1].credit_quality == "IG"
        assert CREDIT_QUALIFYING_BUCKETS[7].credit_quality == "HY/NR"

    def test_credit_non_qualifying_buckets(self):
        assert len(CREDIT_NON_QUALIFYING_BUCKETS) == 2

    def test_equity_buckets(self):
        assert len(EQUITY_BUCKETS) == 12
        assert EQUITY_BUCKETS[11].sector == "Indexes, Funds, ETFs"
        assert EQUITY_BUCKETS[12].sector == "Volatility Indexes"

    def test_commodity_buckets(self):
        assert len(COMMODITY_BUCKETS) == 17
        assert COMMODITY_BUCKETS[2].commodity_type == "Crude"
        assert COMMODITY_BUCKETS[17].commodity_type == "Indexes"

    def test_is_residual_bucket(self):
        assert is_residual_bucket("Residual")
        assert is_residual_bucket("residual")
        assert is_residual_bucket(-1)
        assert not is_residual_bucket(1)
        assert not is_residual_bucket("5")


class TestIRSubCurve:
    def test_sub_curves(self):
        values = {sc.value for sc in IRSubCurve}
        assert values == {
            "OIS", "Libor1m", "Libor3m", "Libor6m", "Libor12m", "Prime", "Municipal"
        }
