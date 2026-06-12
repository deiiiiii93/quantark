"""Tests for Credit / Equity / Commodity / FX calibration (ISDA SIMM v2.6)."""
import numpy as np
import pytest

from quantark.simm.taxonomy import FXVolatilityGroup, RiskClass
from quantark.simm.calibration.credit_qualifying import (
    CREDIT_QUALIFYING_RISK_WEIGHTS,
    CREDIT_QUALIFYING_VRW,
    CREDIT_QUALIFYING_BASE_CORRELATION_RISK_WEIGHT,
    CREDIT_QUALIFYING_BASE_CORRELATION_INTER_INDEX_CORRELATION,
    CREDIT_QUALIFYING_SAME_ISSUER_CORRELATION,
    CREDIT_QUALIFYING_DIFFERENT_ISSUER_CORRELATION,
    CREDIT_QUALIFYING_INTER_BUCKET_CORRELATIONS,
    CREDIT_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS,
    CREDIT_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD,
)
from quantark.simm.calibration.credit_non_qualifying import (
    CREDIT_NON_QUALIFYING_RISK_WEIGHTS,
    CREDIT_NON_QUALIFYING_VRW,
    CREDIT_NON_QUALIFYING_SAME_GROUP_CORRELATION,
    CREDIT_NON_QUALIFYING_DIFFERENT_GROUP_CORRELATION,
    CREDIT_NON_QUALIFYING_INTER_BUCKET_CORRELATION,
    CREDIT_NON_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS,
    CREDIT_NON_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD,
)
from quantark.simm.calibration.equity import (
    EQUITY_RISK_WEIGHTS,
    EQUITY_HVR,
    get_equity_vrw,
    EQUITY_INTRA_BUCKET_CORRELATIONS,
    EQUITY_INTER_BUCKET_CORRELATIONS,
    EQUITY_DELTA_CONCENTRATION_THRESHOLDS,
    EQUITY_VEGA_CONCENTRATION_THRESHOLDS,
)
from quantark.simm.calibration.commodity import (
    COMMODITY_RISK_WEIGHTS,
    COMMODITY_HVR,
    COMMODITY_VRW,
    COMMODITY_INTRA_BUCKET_CORRELATIONS,
    COMMODITY_INTER_BUCKET_CORRELATIONS,
    COMMODITY_DELTA_CONCENTRATION_THRESHOLDS,
    COMMODITY_VEGA_CONCENTRATION_THRESHOLDS,
)
from quantark.simm.calibration.fx import (
    FX_RISK_WEIGHTS,
    FX_HVR,
    FX_VRW,
    FX_DELTA_CORRELATIONS,
    FX_VEGA_CORRELATION,
    FX_DELTA_CONCENTRATION_THRESHOLDS,
    get_fx_vega_concentration_threshold,
)
from quantark.simm.calibration.accessors import get_inter_risk_class_correlation


class TestCreditQualifying:
    def test_risk_weights(self):
        # Paragraph 39.
        expected = {1: 75, 2: 90, 3: 84, 4: 54, 5: 62, 6: 48, 7: 185,
                    8: 343, 9: 255, 10: 250, 11: 214, 12: 173, "Residual": 343}
        for k, v in expected.items():
            assert CREDIT_QUALIFYING_RISK_WEIGHTS[k] == v

    def test_vrw_and_base_corr(self):
        assert CREDIT_QUALIFYING_VRW == 0.76  # Paragraph 40
        assert CREDIT_QUALIFYING_BASE_CORRELATION_RISK_WEIGHT == 10  # Paragraph 41
        assert CREDIT_QUALIFYING_BASE_CORRELATION_INTER_INDEX_CORRELATION == 0.29  # Paragraph 42

    def test_intra_bucket_correlations(self):
        # Paragraph 42.
        assert CREDIT_QUALIFYING_SAME_ISSUER_CORRELATION == 0.93
        assert CREDIT_QUALIFYING_DIFFERENT_ISSUER_CORRELATION == 0.46

    def test_inter_bucket_matrix(self):
        # Paragraph 43 spot checks (1-indexed buckets).
        m = CREDIT_QUALIFYING_INTER_BUCKET_CORRELATIONS
        assert m.shape == (12, 12)
        assert np.allclose(m, m.T)
        assert m[0, 1] == pytest.approx(0.38)
        assert m[0, 6] == pytest.approx(0.42)
        assert m[2, 4] == pytest.approx(0.51)
        assert m[10, 11] == pytest.approx(0.40)

    def test_concentration_thresholds(self):
        # Paragraph 76: sovereigns (1, 7) = 1.0; corporates = 0.17.
        t = CREDIT_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS
        assert t[1] == 1.0
        assert t[7] == 1.0
        assert t[2] == 0.17
        assert t["Residual"] == 0.17
        assert CREDIT_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD == 360  # Paragraph 83


class TestCreditNonQualifying:
    def test_risk_weights(self):
        # Paragraph 46.
        assert CREDIT_NON_QUALIFYING_RISK_WEIGHTS[1] == 280
        assert CREDIT_NON_QUALIFYING_RISK_WEIGHTS[2] == 1300
        assert CREDIT_NON_QUALIFYING_RISK_WEIGHTS["Residual"] == 1300

    def test_correlations(self):
        # Paragraphs 47-49.
        assert CREDIT_NON_QUALIFYING_VRW == 0.76
        assert CREDIT_NON_QUALIFYING_SAME_GROUP_CORRELATION == 0.83
        assert CREDIT_NON_QUALIFYING_DIFFERENT_GROUP_CORRELATION == 0.32
        assert CREDIT_NON_QUALIFYING_INTER_BUCKET_CORRELATION == 0.43

    def test_concentration_thresholds(self):
        # Paragraph 76.
        assert CREDIT_NON_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS[1] == 9.5
        assert CREDIT_NON_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS[2] == 0.5
        assert CREDIT_NON_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD == 70  # Paragraph 83


class TestEquity:
    def test_risk_weights(self):
        # Paragraph 56.
        expected = {1: 30, 2: 33, 3: 36, 4: 29, 5: 26, 6: 25, 7: 34, 8: 28,
                    9: 36, 10: 50, 11: 19, 12: 19, "Residual": 50}
        for k, v in expected.items():
            assert EQUITY_RISK_WEIGHTS[k] == v

    def test_hvr_and_vrw(self):
        # Paragraphs 57-58.
        assert EQUITY_HVR == 0.60
        assert get_equity_vrw(1) == 0.45
        assert get_equity_vrw(11) == 0.45
        assert get_equity_vrw(12) == 0.96

    def test_intra_bucket_correlations(self):
        # Paragraph 59.
        expected = {1: 0.18, 2: 0.20, 3: 0.28, 4: 0.24, 5: 0.25, 6: 0.36,
                    7: 0.35, 8: 0.37, 9: 0.23, 10: 0.27, 11: 0.45, 12: 0.45,
                    "Residual": 0.0}
        for k, v in expected.items():
            assert EQUITY_INTRA_BUCKET_CORRELATIONS[k] == pytest.approx(v)

    def test_inter_bucket_matrix(self):
        # Paragraph 60 spot checks.
        m = EQUITY_INTER_BUCKET_CORRELATIONS
        assert m.shape == (12, 12)
        assert np.allclose(m, m.T)
        assert m[0, 1] == pytest.approx(0.18)
        assert m[4, 5] == pytest.approx(0.29)
        assert m[10, 11] == pytest.approx(0.45)
        assert m[7, 10] == pytest.approx(0.40)
        assert m[0, 9] == pytest.approx(0.12)

    def test_concentration_thresholds(self):
        # Paragraphs 77 and 84.
        assert EQUITY_DELTA_CONCENTRATION_THRESHOLDS[1] == 3
        assert EQUITY_DELTA_CONCENTRATION_THRESHOLDS[5] == 12
        assert EQUITY_DELTA_CONCENTRATION_THRESHOLDS[9] == 0.64
        assert EQUITY_DELTA_CONCENTRATION_THRESHOLDS[10] == 0.37
        assert EQUITY_DELTA_CONCENTRATION_THRESHOLDS[11] == 810
        assert EQUITY_VEGA_CONCENTRATION_THRESHOLDS[1] == 210
        assert EQUITY_VEGA_CONCENTRATION_THRESHOLDS[5] == 1300
        assert EQUITY_VEGA_CONCENTRATION_THRESHOLDS[9] == 39
        assert EQUITY_VEGA_CONCENTRATION_THRESHOLDS[10] == 190
        assert EQUITY_VEGA_CONCENTRATION_THRESHOLDS[12] == 6400


class TestCommodity:
    def test_risk_weights(self):
        # Paragraph 61.
        expected = {1: 48, 2: 29, 3: 33, 4: 25, 5: 35, 6: 30, 7: 60, 8: 52,
                    9: 68, 10: 63, 11: 21, 12: 21, 13: 15, 14: 16, 15: 13,
                    16: 68, 17: 17}
        for k, v in expected.items():
            assert COMMODITY_RISK_WEIGHTS[k] == v

    def test_hvr_and_vrw(self):
        # Paragraphs 62-63.
        assert COMMODITY_HVR == 0.74
        assert COMMODITY_VRW == 0.55

    def test_intra_bucket_correlations(self):
        # Paragraph 64 spot checks.
        assert COMMODITY_INTRA_BUCKET_CORRELATIONS[2] == 0.97
        assert COMMODITY_INTRA_BUCKET_CORRELATIONS[8] == 0.49
        assert COMMODITY_INTRA_BUCKET_CORRELATIONS[16] == 0.0
        assert COMMODITY_INTRA_BUCKET_CORRELATIONS[17] == 0.38

    def test_inter_bucket_matrix(self):
        # Paragraph 65 spot checks (note negative values).
        m = COMMODITY_INTER_BUCKET_CORRELATIONS
        assert m.shape == (17, 17)
        assert np.allclose(m, m.T)
        assert m[1, 2] == pytest.approx(0.92)
        assert m[6, 8] == pytest.approx(0.79)
        assert m[6, 11] == pytest.approx(-0.08)
        assert m[15, 0] == pytest.approx(0.0)
        assert m[1, 16] == pytest.approx(0.64)

    def test_concentration_thresholds(self):
        # Paragraphs 78 and 85.
        assert COMMODITY_DELTA_CONCENTRATION_THRESHOLDS[2] == 2100
        assert COMMODITY_DELTA_CONCENTRATION_THRESHOLDS[10] == 52
        assert COMMODITY_DELTA_CONCENTRATION_THRESHOLDS[17] == 4000
        assert COMMODITY_VEGA_CONCENTRATION_THRESHOLDS[2] == 2900
        assert COMMODITY_VEGA_CONCENTRATION_THRESHOLDS[6] == 6300
        assert COMMODITY_VEGA_CONCENTRATION_THRESHOLDS[17] == 69


class TestFX:
    def test_risk_weights(self):
        # Paragraph 69.
        R, H = FXVolatilityGroup.REGULAR, FXVolatilityGroup.HIGH
        assert FX_RISK_WEIGHTS[(R, R)] == 7.4
        assert FX_RISK_WEIGHTS[(R, H)] == 14.7
        assert FX_RISK_WEIGHTS[(H, R)] == 14.7
        assert FX_RISK_WEIGHTS[(H, H)] == 21.4

    def test_hvr_and_vrw(self):
        # Paragraphs 70-71.
        assert FX_HVR == 0.57
        assert FX_VRW == 0.48

    def test_delta_correlations(self):
        # Paragraph 72: two tables keyed by the calculation currency group.
        R, H = FXVolatilityGroup.REGULAR, FXVolatilityGroup.HIGH
        reg = FX_DELTA_CORRELATIONS[R]
        assert reg[(R, R)] == 0.50
        assert reg[(R, H)] == 0.25
        assert reg[(H, H)] == -0.05
        high = FX_DELTA_CORRELATIONS[H]
        assert high[(R, R)] == 0.88
        assert high[(R, H)] == 0.72
        assert high[(H, H)] == 0.50

    def test_vega_correlation(self):
        # Paragraph 73.
        assert FX_VEGA_CORRELATION == 0.50

    def test_concentration_thresholds(self):
        # Paragraph 79.
        assert FX_DELTA_CONCENTRATION_THRESHOLDS[1] == 3300
        assert FX_DELTA_CONCENTRATION_THRESHOLDS[2] == 880
        assert FX_DELTA_CONCENTRATION_THRESHOLDS[3] == 170

    def test_vega_concentration_thresholds(self):
        # Paragraph 86.
        assert get_fx_vega_concentration_threshold(1, 1) == 2800
        assert get_fx_vega_concentration_threshold(1, 2) == 1400
        assert get_fx_vega_concentration_threshold(2, 1) == 1400
        assert get_fx_vega_concentration_threshold(1, 3) == 590
        assert get_fx_vega_concentration_threshold(2, 2) == 520
        assert get_fx_vega_concentration_threshold(2, 3) == 340
        assert get_fx_vega_concentration_threshold(3, 3) == 210


class TestCrossRiskClass:
    def test_psi_matrix(self):
        # Paragraph 88 spot checks.
        assert get_inter_risk_class_correlation(
            RiskClass.INTEREST_RATE, RiskClass.CREDIT_QUALIFYING) == pytest.approx(0.04)
        assert get_inter_risk_class_correlation(
            RiskClass.INTEREST_RATE, RiskClass.COMMODITY) == pytest.approx(0.37)
        assert get_inter_risk_class_correlation(
            RiskClass.CREDIT_QUALIFYING, RiskClass.EQUITY) == pytest.approx(0.70)
        assert get_inter_risk_class_correlation(
            RiskClass.EQUITY, RiskClass.FX) == pytest.approx(0.39)
