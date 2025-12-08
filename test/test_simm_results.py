"""
Tests for SIMM result dataclasses.
"""
from datetime import date
from typing import Dict

import pytest

from simm.taxonomy import ProductClass, RiskClass, MarginType
from simm.results.simm_result import (
    SIMMResult,
    RiskClassMargin,
    BucketDetail,
    SensitivityContribution,
    AddonBreakdown,
)


class TestSIMMResult:
    """Test SIMMResult dataclass."""

    def test_create_simm_result(self):
        """Test creating a basic SIMMResult."""
        result = SIMMResult(
            total_simm=1000.0,
            calculation_currency="USD",
            calculation_date=date(2024, 1, 15),
            simm_version="2.6",
            product_class_simm={
                ProductClass.RATES_FX: 500.0,
                ProductClass.EQUITY: 300.0,
                ProductClass.CREDIT: 150.0,
                ProductClass.COMMODITY: 50.0,
            },
            risk_class_margin={},
            addon_amount=0.0,
        )

        assert result.total_simm == 1000.0
        assert result.calculation_currency == "USD"
        assert result.calculation_date == date(2024, 1, 15)
        assert result.simm_version == "2.6"
        assert result.addon_amount == 0.0

    def test_get_margin_by_risk_class(self):
        """Test getting margin aggregated by risk class."""
        result = SIMMResult(
            total_simm=1000.0,
            calculation_currency="USD",
            calculation_date=date(2024, 1, 15),
            simm_version="2.6",
            product_class_simm={
                ProductClass.RATES_FX: 500.0,
                ProductClass.EQUITY: 300.0,
                ProductClass.CREDIT: 150.0,
                ProductClass.COMMODITY: 50.0,
            },
            risk_class_margin={
                ProductClass.RATES_FX: {
                    RiskClass.INTEREST_RATE: RiskClassMargin(
                        risk_class=RiskClass.INTEREST_RATE,
                        product_class=ProductClass.RATES_FX,
                        delta_margin=300.0,
                        vega_margin=200.0,
                        total_margin=500.0,
                    )
                },
                ProductClass.EQUITY: {
                    RiskClass.EQUITY: RiskClassMargin(
                        risk_class=RiskClass.EQUITY,
                        product_class=ProductClass.EQUITY,
                        delta_margin=200.0,
                        vega_margin=100.0,
                        total_margin=300.0,
                    )
                },
            },
            addon_amount=0.0,
        )

        margin_by_risk = result.get_margin_by_risk_class()

        assert margin_by_risk[RiskClass.INTEREST_RATE] == 500.0
        assert margin_by_risk[RiskClass.EQUITY] == 300.0
        assert margin_by_risk[RiskClass.CREDIT_QUALIFYING] == 0.0
        assert margin_by_risk[RiskClass.CREDIT_NON_QUALIFYING] == 0.0
        assert margin_by_risk[RiskClass.COMMODITY] == 0.0
        assert margin_by_risk[RiskClass.FX] == 0.0

    def test_get_margin_by_margin_type(self):
        """Test getting margin aggregated by margin type."""
        result = SIMMResult(
            total_simm=1000.0,
            calculation_currency="USD",
            calculation_date=date(2024, 1, 15),
            simm_version="2.6",
            product_class_simm={pc: 250.0 for pc in ProductClass},
            risk_class_margin={
                ProductClass.RATES_FX: {
                    RiskClass.INTEREST_RATE: RiskClassMargin(
                        risk_class=RiskClass.INTEREST_RATE,
                        product_class=ProductClass.RATES_FX,
                        delta_margin=150.0,
                        vega_margin=100.0,
                        curvature_margin=50.0,
                        base_corr_margin=0.0,
                        total_margin=300.0,
                    )
                },
            },
            addon_amount=0.0,
        )

        margin_by_type = result.get_margin_by_margin_type()

        assert margin_by_type[MarginType.DELTA] == 150.0
        assert margin_by_type[MarginType.VEGA] == 100.0
        assert margin_by_type[MarginType.CURVATURE] == 50.0
        assert margin_by_type[MarginType.BASE_CORR] == 0.0

    def test_get_top_buckets(self):
        """Test getting top buckets by margin."""
        result = SIMMResult(
            total_simm=1000.0,
            calculation_currency="USD",
            calculation_date=date(2024, 1, 15),
            simm_version="2.6",
            product_class_simm={pc: 250.0 for pc in ProductClass},
            risk_class_margin={
                ProductClass.RATES_FX: {
                    RiskClass.INTEREST_RATE: RiskClassMargin(
                        risk_class=RiskClass.INTEREST_RATE,
                        product_class=ProductClass.RATES_FX,
                        delta_margin=500.0,
                        total_margin=500.0,
                        bucket_detail={
                            1: BucketDetail(
                                bucket=1,
                                k_value=300.0,
                                s_value=300.0,
                                ws_sum=300.0,
                                concentration_factor=1.0,
                            ),
                            2: BucketDetail(
                                bucket=2,
                                k_value=200.0,
                                s_value=200.0,
                                ws_sum=200.0,
                                concentration_factor=1.0,
                            ),
                        },
                    )
                },
            },
            addon_amount=0.0,
        )

        top_buckets = result.get_top_buckets(5)

        assert len(top_buckets) == 2
        assert top_buckets[0] == (RiskClass.INTEREST_RATE, 1, 300.0)
        assert top_buckets[1] == (RiskClass.INTEREST_RATE, 2, 200.0)

    def test_to_dict_and_from_dict(self):
        """Test serialization to and from dictionary."""
        result = SIMMResult(
            total_simm=1000.0,
            calculation_currency="USD",
            calculation_date=date(2024, 1, 15),
            simm_version="2.6",
            product_class_simm={pc: 250.0 for pc in ProductClass},
            risk_class_margin={},
            addon_amount=0.0,
            warnings=["Test warning"],
        )

        # Convert to dict
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict['total_simm'] == 1000.0
        assert result_dict['calculation_currency'] == "USD"

        # Convert back from dict
        result_restored = SIMMResult.from_dict(result_dict)
        assert result_restored.total_simm == result.total_simm
        assert result_restored.calculation_currency == result.calculation_currency
        assert result_restored.calculation_date == result.calculation_date

    def test_to_json(self):
        """Test JSON serialization."""
        result = SIMMResult(
            total_simm=1000.0,
            calculation_currency="USD",
            calculation_date=date(2024, 1, 15),
            simm_version="2.6",
            product_class_simm={pc: 250.0 for pc in ProductClass},
            risk_class_margin={},
            addon_amount=0.0,
        )

        json_str = result.to_json()
        assert isinstance(json_str, str)
        assert '"total_simm": 1000.0' in json_str

    def test_validate(self):
        """Test result validation."""
        # Valid result
        result = SIMMResult(
            total_simm=1000.0,
            calculation_currency="USD",
            calculation_date=date(2024, 1, 15),
            simm_version="2.6",
            product_class_simm={
                ProductClass.RATES_FX: 500.0,
                ProductClass.EQUITY: 500.0,
            },
            risk_class_margin={
                ProductClass.RATES_FX: {
                    RiskClass.INTEREST_RATE: RiskClassMargin(
                        risk_class=RiskClass.INTEREST_RATE,
                        product_class=ProductClass.RATES_FX,
                        delta_margin=500.0,
                        total_margin=500.0,
                    )
                },
                ProductClass.EQUITY: {
                    RiskClass.EQUITY: RiskClassMargin(
                        risk_class=RiskClass.EQUITY,
                        product_class=ProductClass.EQUITY,
                        delta_margin=500.0,
                        total_margin=500.0,
                    )
                },
            },
            addon_amount=0.0,
        )

        assert result.validate() is True


class TestRiskClassMargin:
    """Test RiskClassMargin dataclass."""

    def test_create_risk_class_margin(self):
        """Test creating a RiskClassMargin."""
        rc_margin = RiskClassMargin(
            risk_class=RiskClass.INTEREST_RATE,
            product_class=ProductClass.RATES_FX,
            delta_margin=300.0,
            vega_margin=200.0,
            curvature_margin=100.0,
            base_corr_margin=50.0,
            total_margin=650.0,
        )

        assert rc_margin.risk_class == RiskClass.INTEREST_RATE
        assert rc_margin.product_class == ProductClass.RATES_FX
        assert rc_margin.delta_margin == 300.0
        assert rc_margin.vega_margin == 200.0
        assert rc_margin.curvature_margin == 100.0
        assert rc_margin.base_corr_margin == 50.0
        assert rc_margin.total_margin == 650.0

    def test_auto_compute_total(self):
        """Test auto-computation of total margin."""
        rc_margin = RiskClassMargin(
            risk_class=RiskClass.INTEREST_RATE,
            product_class=ProductClass.RATES_FX,
            delta_margin=300.0,
            vega_margin=200.0,
            curvature_margin=100.0,
            base_corr_margin=50.0,
            # total_margin not specified
        )

        assert rc_margin.total_margin == 650.0


class TestBucketDetail:
    """Test BucketDetail dataclass."""

    def test_create_bucket_detail(self):
        """Test creating a BucketDetail."""
        bucket = BucketDetail(
            bucket=1,
            k_value=500.0,
            s_value=500.0,
            ws_sum=600.0,
            concentration_factor=1.2,
        )

        assert bucket.bucket == 1
        assert bucket.k_value == 500.0
        assert bucket.s_value == 500.0
        assert bucket.ws_sum == 600.0
        assert bucket.concentration_factor == 1.2

    def test_net_sensitivity(self):
        """Test net sensitivity calculation."""
        bucket = BucketDetail(
            bucket=1,
            k_value=500.0,
            s_value=500.0,
            ws_sum=600.0,
            concentration_factor=1.2,
        )

        assert bucket.net_sensitivity == 500.0


class TestSensitivityContribution:
    """Test SensitivityContribution dataclass."""

    def test_create_sensitivity_contribution(self):
        """Test creating a SensitivityContribution."""
        contrib = SensitivityContribution(
            sensitivity_id="SENS_001",
            position_id="POS_001",
            ws_value=100.0,
            pct_of_bucket=0.25,
        )

        assert contrib.sensitivity_id == "SENS_001"
        assert contrib.position_id == "POS_001"
        assert contrib.ws_value == 100.0
        assert contrib.pct_of_bucket == 0.25


class TestAddonBreakdown:
    """Test AddonBreakdown dataclass."""

    def test_create_addon_breakdown(self):
        """Test creating an AddonBreakdown."""
        addon = AddonBreakdown(
            supervision_addon=100.0,
            viral_addon=50.0,
            total=150.0,
        )

        assert addon.supervision_addon == 100.0
        assert addon.viral_addon == 50.0
        assert addon.total == 150.0


if __name__ == "__main__":
    pytest.main([__file__])
