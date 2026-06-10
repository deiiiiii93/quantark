"""
Tests for SIMM Interest Rate Calibration Parameters

Tests verify that all IR calibration parameters match the ISDA SIMM v2.6
specification.
"""

import pytest
import numpy as np
from quantark.simm.calibration import (
    IR_RISK_WEIGHTS,
    IR_TENOR_CORRELATIONS,
    IR_SUB_CURVE_CORRELATION,
    IR_INFLATION_CORRELATION,
    IR_CROSS_CURRENCY_BASIS_CORRELATION,
    IR_INTER_CURRENCY_CORRELATION,
    IR_INFLATION_RISK_WEIGHT,
    IR_CROSS_CURRENCY_BASIS_RISK_WEIGHT,
    IR_HVR,
    IR_VRW,
    IR_DELTA_CONCENTRATION_THRESHOLDS,
    IR_VEGA_CONCENTRATION_THRESHOLDS,
    IR_TENOR_LABELS,
)
from quantark.simm.calibration.accessors import (
    get_risk_weight,
    get_inter_bucket_correlation,
    get_concentration_threshold,
    get_hvr,
    get_vrw,
    RiskClass,
    MarginType,
)


class TestIRRiskWeights:
    """Test IR risk weights by tenor and currency group."""

    def test_regular_volatility_risk_weights(self):
        """Test risk weights for regular volatility currencies (USD, EUR, GBP)."""
        assert IR_RISK_WEIGHTS[("2w", "regular")] == 109
        assert IR_RISK_WEIGHTS[("1m", "regular")] == 105
        assert IR_RISK_WEIGHTS[("1yr", "regular")] == 66
        assert IR_RISK_WEIGHTS[("10yr", "regular")] == 60
        assert IR_RISK_WEIGHTS[("30yr", "regular")] == 67

    def test_low_volatility_risk_weights(self):
        """Test risk weights for low volatility currencies (JPY)."""
        assert IR_RISK_WEIGHTS[("2w", "low")] == 15
        assert IR_RISK_WEIGHTS[("1m", "low")] == 18
        assert IR_RISK_WEIGHTS[("1yr", "low")] == 13
        assert IR_RISK_WEIGHTS[("10yr", "low")] == 23
        assert IR_RISK_WEIGHTS[("30yr", "low")] == 23

    def test_high_volatility_risk_weights(self):
        """Test risk weights for high volatility currencies."""
        assert IR_RISK_WEIGHTS[("2w", "high")] == 163
        assert IR_RISK_WEIGHTS[("1m", "high")] == 109
        assert IR_RISK_WEIGHTS[("1yr", "high")] == 102
        assert IR_RISK_WEIGHTS[("10yr", "high")] == 97
        assert IR_RISK_WEIGHTS[("30yr", "high")] == 101

    def test_accessor_get_risk_weight(self):
        """Test unified accessor for IR risk weights."""
        assert get_risk_weight(RiskClass.IR, bucket=None, tenor="1yr", currency_group="regular") == 66
        assert get_risk_weight(RiskClass.IR, bucket=None, tenor="1yr", currency_group="low") == 13
        assert get_risk_weight(RiskClass.IR, bucket=None, tenor="1yr", currency_group="high") == 102


class TestIRCorrelations:
    """Test IR correlation parameters."""

    def test_tenor_correlation_matrix_dimensions(self):
        """Test that tenor correlation matrix is 12x12."""
        assert IR_TENOR_CORRELATIONS.shape == (12, 12)

    def test_tenor_correlation_matrix_symmetry(self):
        """Test that tenor correlation matrix is symmetric."""
        assert np.allclose(IR_TENOR_CORRELATIONS, IR_TENOR_CORRELATIONS.T)

    def test_tenor_correlation_diagonal(self):
        """Test that diagonal elements of correlation matrix are 1.0."""
        assert np.allclose(np.diag(IR_TENOR_CORRELATIONS), 1.0)

    def test_tenor_correlation_positive_semi_definite(self):
        """Test that correlation matrix is positive semi-definite."""
        eigenvalues = np.linalg.eigvals(IR_TENOR_CORRELATIONS)
        assert np.all(eigenvalues >= -1e-10), "Correlation matrix should be positive semi-definite"

    def test_specific_tenor_correlations(self):
        """Test specific correlation values from ISDA specification."""
        # Adjacent tenors should have high correlation
        assert IR_TENOR_CORRELATIONS[0, 1] == 0.77  # 2w vs 1m
        assert IR_TENOR_CORRELATIONS[1, 2] == 0.84  # 1m vs 3m

        # Distant tenors should have lower correlation
        assert IR_TENOR_CORRELATIONS[0, 11] == 0.20  # 2w vs 30yr

    def test_sub_curve_correlation(self):
        """Test IR sub-curve correlation (99.3%)."""
        assert IR_SUB_CURVE_CORRELATION == 0.993

    def test_inflation_correlation(self):
        """Test IR inflation correlation (24%)."""
        assert IR_INFLATION_CORRELATION == 0.24

    def test_cross_currency_basis_correlation(self):
        """Test IR cross-currency basis correlation (4%)."""
        assert IR_CROSS_CURRENCY_BASIS_CORRELATION == 0.04

    def test_inter_currency_correlation(self):
        """Test IR inter-currency correlation (32%)."""
        assert IR_INTER_CURRENCY_CORRELATION == 0.32

    def test_accessor_get_inter_bucket_correlation(self):
        """Test unified accessor for IR inter-bucket (tenor) correlations."""
        assert get_inter_bucket_correlation(RiskClass.IR, "1yr", "2yr") == pytest.approx(0.79)
        assert get_inter_bucket_correlation(RiskClass.IR, "2w", "30yr") == pytest.approx(0.20)


class TestIRAdditionalRiskWeights:
    """Test additional IR risk weights."""

    def test_inflation_risk_weight(self):
        """Test IR inflation risk weight."""
        assert IR_INFLATION_RISK_WEIGHT == 61

    def test_cross_currency_basis_risk_weight(self):
        """Test IR cross-currency basis risk weight."""
        assert IR_CROSS_CURRENCY_BASIS_RISK_WEIGHT == 21


class TestIRHVRVRW:
    """Test IR Historical Volatility Ratio and Vega Risk Weight."""

    def test_hvr(self):
        """Test IR Historical Volatility Ratio."""
        assert IR_HVR == 0.47

    def test_vrw(self):
        """Test IR Vega Risk Weight."""
        assert IR_VRW == 0.23

    def test_accessor_get_hvr(self):
        """Test unified accessor for HVR."""
        assert get_hvr(RiskClass.IR) == 0.47

    def test_accessor_get_vrw(self):
        """Test unified accessor for VRW."""
        assert get_vrw(RiskClass.IR) == 0.23


class TestIRConcentrationThresholds:
    """Test IR concentration thresholds."""

    def test_delta_concentration_thresholds(self):
        """Test IR delta concentration thresholds."""
        assert IR_DELTA_CONCENTRATION_THRESHOLDS["high"] == 30
        assert IR_DELTA_CONCENTRATION_THRESHOLDS["regular_well_traded"] == 330
        assert IR_DELTA_CONCENTRATION_THRESHOLDS["regular_less_traded"] == 130
        assert IR_DELTA_CONCENTRATION_THRESHOLDS["low"] == 61

    def test_vega_concentration_thresholds(self):
        """Test IR vega concentration thresholds."""
        assert IR_VEGA_CONCENTRATION_THRESHOLDS["high"] == 30
        assert IR_VEGA_CONCENTRATION_THRESHOLDS["regular_well_traded"] == 330
        assert IR_VEGA_CONCENTRATION_THRESHOLDS["regular_less_traded"] == 130
        assert IR_VEGA_CONCENTRATION_THRESHOLDS["low"] == 61

    def test_accessor_get_concentration_threshold(self):
        """Test unified accessor for concentration thresholds."""
        delta_threshold = get_concentration_threshold(
            RiskClass.IR, "regular_well_traded", MarginType.DELTA
        )
        assert delta_threshold == 330

        vega_threshold = get_concentration_threshold(
            RiskClass.IR, "low", MarginType.VEGA
        )
        assert vega_threshold == 61


class TestIRTenorLabels:
    """Test IR tenor labels."""

    def test_tenor_labels_complete(self):
        """Test that all 12 tenor labels are present."""
        expected_labels = [
            "2w", "1m", "3m", "6m", "1yr", "2yr",
            "3yr", "5yr", "10yr", "15yr", "20yr", "30yr"
        ]
        assert IR_TENOR_LABELS == expected_labels

    def test_tenor_labels_count(self):
        """Test that there are exactly 12 tenor labels."""
        assert len(IR_TENOR_LABELS) == 12
