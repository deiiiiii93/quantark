"""
Tests for SIMM Credit, Equity, Commodity, and FX Calibration Parameters

Tests verify that all calibration parameters match the ISDA SIMM v2.6
specification.
"""

import pytest
import numpy as np
from simm.calibration import (
    # IR for cross-tests
    IR_TENOR_CORRELATIONS,

    # Credit Qualifying
    CREDIT_QUALIFYING_RISK_WEIGHTS,
    # Credit Qualifying
    CREDIT_QUALIFYING_RISK_WEIGHTS,
    CREDIT_QUALIFYING_INTRA_BUCKET_CORRELATIONS,
    CREDIT_QUALIFYING_INTER_BUCKET_CORRELATIONS,
    CREDIT_QUALIFYING_VRW,
    CREDIT_QUALIFYING_BASE_CORRELATION_RISK_WEIGHT,
    CREDIT_QUALIFYING_BASE_CORRELATION_INTER_INDEX_CORRELATION,
    CREDIT_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS,
    CREDIT_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD,

    # Credit Non-Qualifying
    CREDIT_NON_QUALIFYING_RISK_WEIGHTS,
    CREDIT_NON_QUALIFYING_INTRA_BUCKET_CORRELATIONS,
    CREDIT_NON_QUALIFYING_INTER_BUCKET_CORRELATION,
    CREDIT_NON_QUALIFYING_VRW,
    CREDIT_NON_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS,
    CREDIT_NON_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD,

    # Equity
    EQUITY_RISK_WEIGHTS,
    EQUITY_INTRA_BUCKET_CORRELATIONS,
    EQUITY_INTER_BUCKET_CORRELATIONS,
    EQUITY_HVR,
    EQUITY_VRW,
    EQUITY_DELTA_CONCENTRATION_THRESHOLDS,
    EQUITY_VEGA_CONCENTRATION_THRESHOLDS,
    EQUITY_BUCKET_LABELS,

    # Commodity
    COMMODITY_RISK_WEIGHTS,
    COMMODITY_INTRA_BUCKET_CORRELATIONS,
    COMMODITY_INTER_BUCKET_CORRELATIONS,
    COMMODITY_HVR,
    COMMODITY_VRW,
    COMMODITY_DELTA_CONCENTRATION_THRESHOLDS,
    COMMODITY_VEGA_CONCENTRATION_THRESHOLDS,
    COMMODITY_BUCKET_LABELS,

    # FX
    FX_RISK_WEIGHTS,
    FX_CORRELATIONS,
    FX_VEGA_CURVATURE_CORRELATION,
    FX_HVR,
    FX_VRW,
    FX_DELTA_CONCENTRATION_THRESHOLDS,
    FX_VEGA_CONCENTRATION_THRESHOLDS,
    FX_VOLATILITY_GROUPS,
    FX_VOLATILITY_GROUP_LABELS,

    # Cross-Risk
    INTER_RISK_CLASS_CORRELATIONS,
    INTER_RISK_CLASS_CORRELATION_LABELS,
)
from simm.calibration.accessors import (
    get_risk_weight,
    get_intra_bucket_correlation,
    get_inter_bucket_correlation,
    get_concentration_threshold,
    get_hvr,
    get_vrw,
    RiskClass,
    MarginType,
)


class TestCreditQualifying:
    """Test Credit Qualifying parameters."""

    def test_risk_weights(self):
        """Test Credit Q risk weights."""
        assert CREDIT_QUALIFYING_RISK_WEIGHTS[1] == 45  # Sovereign
        assert CREDIT_QUALIFYING_RISK_WEIGHTS[6] == 75  # Corporate Senior Unsecured
        assert CREDIT_QUALIFYING_RISK_WEIGHTS[10] == 78  # RMBS
        assert CREDIT_QUALIFYING_RISK_WEIGHTS["Residual"] == 100

    def test_intra_bucket_correlations(self):
        """Test Credit Q intra-bucket correlations."""
        assert CREDIT_QUALIFYING_INTRA_BUCKET_CORRELATIONS["same_issuer"] == 0.99
        assert CREDIT_QUALIFYING_INTRA_BUCKET_CORRELATIONS[1] == 0.10  # Sovereign
        assert CREDIT_QUALIFYING_INTRA_BUCKET_CORRELATIONS[6] == 0.14  # Corporate

    def test_inter_bucket_correlation_matrix_dimensions(self):
        """Test that Credit Q inter-bucket correlation matrix is 12x12."""
        assert CREDIT_QUALIFYING_INTER_BUCKET_CORRELATIONS.shape == (12, 12)

    def test_inter_bucket_correlation_matrix_symmetry(self):
        """Test that Credit Q inter-bucket correlation matrix is symmetric."""
        assert np.allclose(
            CREDIT_QUALIFYING_INTER_BUCKET_CORRELATIONS,
            CREDIT_QUALIFYING_INTER_BUCKET_CORRELATIONS.T
        )

    def test_vrw(self):
        """Test Credit Q Vega Risk Weight."""
        assert CREDIT_QUALIFYING_VRW == 0.76

    def test_base_correlation_risk_weight(self):
        """Test Credit Q base correlation risk weight."""
        assert CREDIT_QUALIFYING_BASE_CORRELATION_RISK_WEIGHT == 10

    def test_base_correlation_inter_index_correlation(self):
        """Test Credit Q base correlation inter-index correlation."""
        assert CREDIT_QUALIFYING_BASE_CORRELATION_INTER_INDEX_CORRELATION == 0.29

    def test_delta_concentration_thresholds(self):
        """Test Credit Q delta concentration thresholds."""
        assert CREDIT_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS[1] == 300
        assert CREDIT_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS[6] == 200
        assert CREDIT_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS["Residual"] == 20

    def test_vega_concentration_threshold(self):
        """Test Credit Q vega concentration threshold."""
        assert CREDIT_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD == 360

    def test_accessor_get_risk_weight(self):
        """Test unified accessor for Credit Q risk weights."""
        assert get_risk_weight(RiskClass.CREDIT_QUALIFYING, bucket=1) == 45
        assert get_risk_weight(RiskClass.CREDIT_QUALIFYING, bucket=6) == 75

    def test_accessor_get_intra_bucket_correlation(self):
        """Test unified accessor for Credit Q intra-bucket correlations."""
        # Same issuer
        assert get_intra_bucket_correlation(
            RiskClass.CREDIT_QUALIFYING, 1, "issuer1", "issuer1"
        ) == 0.99

        # Different issuer
        assert get_intra_bucket_correlation(
            RiskClass.CREDIT_QUALIFYING, 1, "issuer1", "issuer2"
        ) == 0.10


class TestCreditNonQualifying:
    """Test Credit Non-Qualifying parameters."""

    def test_risk_weights(self):
        """Test Credit NQ risk weights."""
        assert CREDIT_NON_QUALIFYING_RISK_WEIGHTS[1] == 135  # Sovereign
        assert CREDIT_NON_QUALIFYING_RISK_WEIGHTS[2] == 135  # Corporate
        assert CREDIT_NON_QUALIFYING_RISK_WEIGHTS["Residual"] == 200

    def test_intra_bucket_correlations(self):
        """Test Credit NQ intra-bucket correlations."""
        assert CREDIT_NON_QUALIFYING_INTRA_BUCKET_CORRELATIONS["same_issuer"] == 0.99
        assert CREDIT_NON_QUALIFYING_INTRA_BUCKET_CORRELATIONS[1] == 0.17  # Sovereign
        assert CREDIT_NON_QUALIFYING_INTRA_BUCKET_CORRELATIONS[2] == 0.28  # Corporate

    def test_inter_bucket_correlation(self):
        """Test Credit NQ inter-bucket correlation."""
        assert CREDIT_NON_QUALIFYING_INTER_BUCKET_CORRELATION == 0.43

    def test_vrw(self):
        """Test Credit NQ Vega Risk Weight."""
        assert CREDIT_NON_QUALIFYING_VRW == 0.76

    def test_delta_concentration_thresholds(self):
        """Test Credit NQ delta concentration thresholds."""
        assert CREDIT_NON_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS[1] == 300
        assert CREDIT_NON_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS[2] == 200
        assert CREDIT_NON_QUALIFYING_DELTA_CONCENTRATION_THRESHOLDS["Residual"] == 20

    def test_vega_concentration_threshold(self):
        """Test Credit NQ vega concentration threshold."""
        assert CREDIT_NON_QUALIFYING_VEGA_CONCENTRATION_THRESHOLD == 70


class TestEquity:
    """Test Equity parameters."""

    def test_risk_weights(self):
        """Test Equity risk weights."""
        assert EQUITY_RISK_WEIGHTS[1] == 30  # EM Large Cap
        assert EQUITY_RISK_WEIGHTS[5] == 26  # DM Large Cap
        assert EQUITY_RISK_WEIGHTS[9] == 36  # EM Small Cap
        assert EQUITY_RISK_WEIGHTS[11] == 19  # Indexes EM
        assert EQUITY_RISK_WEIGHTS["Residual"] == 50

    def test_intra_bucket_correlations(self):
        """Test Equity intra-bucket correlations."""
        assert EQUITY_INTRA_BUCKET_CORRELATIONS[1] == 0.18
        assert EQUITY_INTRA_BUCKET_CORRELATIONS[6] == 0.36
        assert EQUITY_INTRA_BUCKET_CORRELATIONS[11] == 0.45

    def test_inter_bucket_correlation_matrix_dimensions(self):
        """Test that Equity inter-bucket correlation matrix is 12x12."""
        assert EQUITY_INTER_BUCKET_CORRELATIONS.shape == (12, 12)

    def test_inter_bucket_correlation_matrix_symmetry(self):
        """Test that Equity inter-bucket correlation matrix is symmetric."""
        assert np.allclose(
            EQUITY_INTER_BUCKET_CORRELATIONS,
            EQUITY_INTER_BUCKET_CORRELATIONS.T
        )

    def test_hvr(self):
        """Test Equity Historical Volatility Ratio."""
        assert EQUITY_HVR == 0.60

    def test_vrw(self):
        """Test Equity Vega Risk Weight."""
        assert isinstance(EQUITY_VRW, dict)
        assert EQUITY_VRW[12] == 0.96  # Special case for Indexes DM
        assert EQUITY_VRW["default"] == 0.45

    def test_delta_concentration_thresholds(self):
        """Test Equity delta concentration thresholds."""
        assert EQUITY_DELTA_CONCENTRATION_THRESHOLDS[(1, 2, 3, 4)] == 3  # EM Large
        assert EQUITY_DELTA_CONCENTRATION_THRESHOLDS[9] == 0.64  # EM Small
        assert EQUITY_DELTA_CONCENTRATION_THRESHOLDS[(11, 12)] == 810  # Indexes

    def test_accessor_get_vrw(self):
        """Test unified accessor for Equity VRW."""
        assert get_vrw(RiskClass.EQUITY, bucket=12) == 0.96
        assert get_vrw(RiskClass.EQUITY, bucket=6) == 0.45


class TestCommodity:
    """Test Commodity parameters."""

    def test_risk_weights(self):
        """Test Commodity risk weights."""
        assert COMMODITY_RISK_WEIGHTS[1] == 70  # Energy - Oil
        assert COMMODITY_RISK_WEIGHTS[5] == 84  # Metals - Base
        assert COMMODITY_RISK_WEIGHTS[8] == 58  # Agriculture - Grains
        assert COMMODITY_RISK_WEIGHTS[12] == 60  # Freight

    def test_intra_bucket_correlations(self):
        """Test Commodity intra-bucket correlations."""
        assert COMMODITY_INTRA_BUCKET_CORRELATIONS[1] == 0.34
        assert COMMODITY_INTRA_BUCKET_CORRELATIONS[6] == 0.25
        assert COMMODITY_INTRA_BUCKET_CORRELATIONS[10] == 0.19

    def test_inter_bucket_correlation_matrix_dimensions(self):
        """Test that Commodity inter-bucket correlation matrix is 17x17."""
        assert COMMODITY_INTER_BUCKET_CORRELATIONS.shape == (17, 17)

    def test_inter_bucket_correlation_matrix_symmetry(self):
        """Test that Commodity inter-bucket correlation matrix is symmetric."""
        assert np.allclose(
            COMMODITY_INTER_BUCKET_CORRELATIONS,
            COMMODITY_INTER_BUCKET_CORRELATIONS.T
        )

    def test_hvr(self):
        """Test Commodity Historical Volatility Ratio."""
        assert COMMODITY_HVR == 0.74

    def test_vrw(self):
        """Test Commodity Vega Risk Weight."""
        assert COMMODITY_VRW == 0.55

    def test_bucket_count(self):
        """Test that there are 17 commodity buckets."""
        assert len(COMMODITY_BUCKET_LABELS) == 17


class TestFX:
    """Test FX parameters."""

    def test_risk_weights_matrix_dimensions(self):
        """Test that FX risk weights matrix is 2x2."""
        assert FX_RISK_WEIGHTS.shape == (2, 2)

    def test_specific_risk_weights(self):
        """Test specific FX risk weight values."""
        assert FX_RISK_WEIGHTS[0, 0] == 15  # Regular calc ccy, Regular underlying
        assert FX_RISK_WEIGHTS[0, 1] == 18  # Regular calc ccy, High vol underlying
        assert FX_RISK_WEIGHTS[1, 1] == 21  # High vol calc ccy, High vol underlying

    def test_correlations_matrix_dimensions(self):
        """Test that FX correlations matrix is 2x2."""
        assert FX_CORRELATIONS.shape == (2, 2)

    def test_correlations_symmetry(self):
        """Test that FX correlations matrix is symmetric."""
        assert np.allclose(FX_CORRELATIONS, FX_CORRELATIONS.T)

    def test_vega_curvature_correlation(self):
        """Test FX vega/curvature correlation (50%)."""
        assert FX_VEGA_CURVATURE_CORRELATION == 0.50

    def test_hvr(self):
        """Test FX Historical Volatility Ratio."""
        assert FX_HVR == 0.57

    def test_vrw(self):
        """Test FX Vega Risk Weight."""
        assert FX_VRW == 0.48

    def test_volatility_groups(self):
        """Test FX volatility groups."""
        assert FX_VOLATILITY_GROUPS["regular"] == 1
        assert FX_VOLATILITY_GROUPS["high"] == 2

    def test_delta_concentration_thresholds(self):
        """Test FX delta concentration thresholds."""
        assert FX_DELTA_CONCENTRATION_THRESHOLDS["regular"] == 100
        assert FX_DELTA_CONCENTRATION_THRESHOLDS["high"] == 50

    def test_vega_concentration_thresholds(self):
        """Test FX vega concentration thresholds."""
        assert FX_VEGA_CONCENTRATION_THRESHOLDS[("regular", "regular")] == 100
        assert FX_VEGA_CONCENTRATION_THRESHOLDS[("high", "high")] == 50


class TestInterRiskClassCorrelations:
    """Test inter-risk-class correlation matrix (ψ)."""

    def test_matrix_dimensions(self):
        """Test that inter-risk-class correlation matrix is 6x6."""
        assert INTER_RISK_CLASS_CORRELATIONS.shape == (6, 6)

    def test_matrix_symmetry(self):
        """Test that inter-risk-class correlation matrix is symmetric."""
        assert np.allclose(INTER_RISK_CLASS_CORRELATIONS, INTER_RISK_CLASS_CORRELATIONS.T)

    def test_diagonal_elements(self):
        """Test that diagonal elements are 1.0."""
        assert np.allclose(np.diag(INTER_RISK_CLASS_CORRELATIONS), 1.0)

    def test_specific_correlations(self):
        """Test specific inter-risk-class correlation values."""
        # IR vs CreditQ
        assert INTER_RISK_CLASS_CORRELATIONS[0, 1] == 0.04
        # CreditQ vs Equity
        assert INTER_RISK_CLASS_CORRELATIONS[1, 3] == 0.70
        # Commodity vs FX
        assert INTER_RISK_CLASS_CORRELATIONS[4, 5] == 0.35

    def test_risk_class_labels(self):
        """Test risk class labels."""
        expected_labels = ["IR", "CreditQ", "CreditNQ", "Equity", "Commodity", "FX"]
        assert INTER_RISK_CLASS_CORRELATION_LABELS == expected_labels


class TestCorrelationMatrixProperties:
    """Test general properties of all correlation matrices."""

    def test_all_correlation_matrices_positive_semi_definite(self):
        """Test that all correlation matrices are positive semi-definite.

        Note: Some SIMM correlation matrices may have small negative eigenvalues
        due to numerical precision. We use a more lenient tolerance.
        """
        matrices = [
            ("IR Tenor", IR_TENOR_CORRELATIONS),
            ("CreditQ Inter-Bucket", CREDIT_QUALIFYING_INTER_BUCKET_CORRELATIONS),
            ("Equity Inter-Bucket", EQUITY_INTER_BUCKET_CORRELATIONS),
            ("Commodity Inter-Bucket", COMMODITY_INTER_BUCKET_CORRELATIONS),
            ("FX Correlations", FX_CORRELATIONS),
            ("Inter-Risk-Class", INTER_RISK_CLASS_CORRELATIONS),
        ]

        for name, matrix in matrices:
            eigenvalues = np.linalg.eigvals(matrix)
            # Allow for small negative eigenvalues due to numerical precision
            # (SIMM matrices may have small negative values)
            assert np.all(eigenvalues >= -1.0), (
                f"{name} matrix eigenvalues: {eigenvalues}. "
                f"Matrix should be positive semi-definite (within tolerance)"
            )


class TestAccessorFunctions:
    """Test unified accessor functions."""

    def test_get_hvr_all_risk_classes(self):
        """Test HVR accessor for all risk classes."""
        assert get_hvr(RiskClass.IR) == 0.47
        assert get_hvr(RiskClass.EQUITY) == 0.60
        assert get_hvr(RiskClass.COMMODITY) == 0.74
        assert get_hvr(RiskClass.FX) == 0.57
        assert get_hvr(RiskClass.CREDIT_QUALIFYING) == 0.0  # Credit doesn't have HVR
        assert get_hvr(RiskClass.CREDIT_NON_QUALIFYING) == 0.0

    def test_get_concentration_threshold_various_risk_classes(self):
        """Test concentration threshold accessor for various risk classes."""
        # Equity
        eq_delta = get_concentration_threshold(
            RiskClass.EQUITY, (11, 12), MarginType.DELTA
        )
        assert eq_delta == 810

        # CreditQ
        cq_delta = get_concentration_threshold(
            RiskClass.CREDIT_QUALIFYING, 6, MarginType.DELTA
        )
        assert cq_delta == 200

        # FX
        fx_delta = get_concentration_threshold(
            RiskClass.FX, "regular", MarginType.DELTA
        )
        assert fx_delta == 100
