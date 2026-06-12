"""Tests for IR calibration parameters (ISDA SIMM v2.6, Section D, J.1, J.6)."""
import numpy as np
import pytest

from quantark.simm.taxonomy import CurrencyVolatility, IRConcentrationGroup
from quantark.simm.calibration.ir import (
    IR_DELTA_RISK_WEIGHTS,
    IR_INFLATION_RISK_WEIGHT,
    IR_XCCY_BASIS_RISK_WEIGHT,
    IR_HVR,
    IR_VRW,
    IR_TENOR_CORRELATIONS,
    IR_TENOR_INDEX,
    IR_SUB_CURVE_CORRELATION,
    IR_INFLATION_CORRELATION,
    IR_XCCY_BASIS_CORRELATION,
    IR_INTER_CURRENCY_CORRELATION,
    IR_DELTA_CONCENTRATION_THRESHOLDS,
    IR_VEGA_CONCENTRATION_THRESHOLDS,
)


class TestIRRiskWeights:
    def test_regular_currency_weights(self):
        # Paragraph 33, Table 1.
        rw = IR_DELTA_RISK_WEIGHTS[CurrencyVolatility.REGULAR]
        expected = dict(zip(
            ("2w", "1m", "3m", "6m", "1y", "2y", "3y", "5y", "10y", "15y", "20y", "30y"),
            (109, 105, 90, 71, 66, 66, 64, 60, 60, 61, 61, 67),
        ))
        assert rw == expected

    def test_low_volatility_weights(self):
        # Paragraph 33, Table 2 (JPY).
        rw = IR_DELTA_RISK_WEIGHTS[CurrencyVolatility.LOW]
        assert rw["2w"] == 15
        assert rw["3m"] == 9
        assert rw["5y"] == 23
        assert rw["30y"] == 23

    def test_high_volatility_weights(self):
        # Paragraph 33, Table 3.
        rw = IR_DELTA_RISK_WEIGHTS[CurrencyVolatility.HIGH]
        assert rw["2w"] == 163
        assert rw["1y"] == 102
        assert rw["20y"] == 106

    def test_inflation_and_xccy_weights(self):
        # Paragraph 33.
        assert IR_INFLATION_RISK_WEIGHT == 61
        assert IR_XCCY_BASIS_RISK_WEIGHT == 21

    def test_hvr_and_vrw(self):
        # Paragraphs 34-35.
        assert IR_HVR == 0.47
        assert IR_VRW == 0.23


class TestIRCorrelations:
    def test_matrix_is_symmetric_with_unit_diagonal(self):
        assert IR_TENOR_CORRELATIONS.shape == (12, 12)
        assert np.allclose(IR_TENOR_CORRELATIONS, IR_TENOR_CORRELATIONS.T)
        assert np.allclose(np.diag(IR_TENOR_CORRELATIONS), 1.0)

    def test_spot_values_from_doc(self):
        # Paragraph 36 spot checks.
        idx = IR_TENOR_INDEX
        assert IR_TENOR_CORRELATIONS[idx["2w"], idx["1m"]] == pytest.approx(0.77)
        assert IR_TENOR_CORRELATIONS[idx["1y"], idx["2y"]] == pytest.approx(0.94)
        assert IR_TENOR_CORRELATIONS[idx["3y"], idx["5y"]] == pytest.approx(0.97)
        assert IR_TENOR_CORRELATIONS[idx["5y"], idx["10y"]] == pytest.approx(0.95)
        assert IR_TENOR_CORRELATIONS[idx["20y"], idx["30y"]] == pytest.approx(0.99)
        assert IR_TENOR_CORRELATIONS[idx["2w"], idx["30y"]] == pytest.approx(0.20)

    def test_special_correlations(self):
        # Paragraph 36.
        assert IR_SUB_CURVE_CORRELATION == 0.993
        assert IR_INFLATION_CORRELATION == 0.24
        assert IR_XCCY_BASIS_CORRELATION == 0.04

    def test_inter_currency_gamma(self):
        # Paragraph 37.
        assert IR_INTER_CURRENCY_CORRELATION == 0.32


class TestIRConcentrationThresholds:
    def test_delta_thresholds(self):
        # Paragraph 74 (USD mm/bp).
        t = IR_DELTA_CONCENTRATION_THRESHOLDS
        assert t[IRConcentrationGroup.HIGH_VOLATILITY] == 30
        assert t[IRConcentrationGroup.REGULAR_WELL_TRADED] == 330
        assert t[IRConcentrationGroup.REGULAR_LESS_WELL_TRADED] == 130
        assert t[IRConcentrationGroup.LOW_VOLATILITY] == 61

    def test_vega_thresholds(self):
        # Paragraph 81 (USD mm).
        t = IR_VEGA_CONCENTRATION_THRESHOLDS
        assert t[IRConcentrationGroup.HIGH_VOLATILITY] == 74
        assert t[IRConcentrationGroup.REGULAR_WELL_TRADED] == 4900
        assert t[IRConcentrationGroup.REGULAR_LESS_WELL_TRADED] == 520
        assert t[IRConcentrationGroup.LOW_VOLATILITY] == 970
