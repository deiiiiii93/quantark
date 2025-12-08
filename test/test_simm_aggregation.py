"""
Tests for SIMM Aggregation Engine.

This module tests the full SIMM calculation pipeline including:
- Concentration risk calculation
- Weighted sensitivity calculation
- Bucket aggregation
- Risk class aggregation
- Product class aggregation
- Main SIMMCalculator
"""

import pytest
import math
from typing import List

from simm.config import SIMMConfig
from simm.taxonomy import RiskClass, ProductClass, MarginType
from simm.sensitivity import (
    SensitivityCollection,
    IRDeltaSensitivity,
    IRVegaSensitivity,
    EquityDeltaSensitivity,
    EquityVegaSensitivity,
    CreditDeltaSensitivity,
    FXDeltaSensitivity,
    CurvatureSensitivity,
)
from simm.engines.aggregation import (
    SIMMCalculator,
    ConcentrationCalculator,
    WeightedSensitivityCalculator,
    BucketAggregator,
    RiskClassAggregator,
    ProductClassAggregator,
)


class TestConcentrationCalculator:
    """Tests for ConcentrationCalculator."""
    
    def test_ir_concentration_small_position(self):
        """Small IR position should have CR = 1."""
        calc = ConcentrationCalculator()
        
        sens = [
            IRDeltaSensitivity(
                trade_id="t1",
                amount=1_000_000,  # 1MM USD
                currency="USD",
                tenor=1.0,
            )
        ]
        
        result = calc.calculate(sens, RiskClass.INTEREST_RATE, MarginType.DELTA, "USD")
        
        # Small position relative to threshold should give CR = 1
        assert result.bucket_cr == 1.0
        assert result.cr_values.get("USD", 1.0) == 1.0
    
    def test_ir_concentration_large_position(self):
        """Large IR position should have CR > 1."""
        calc = ConcentrationCalculator()
        
        # Very large position
        sens = [
            IRDeltaSensitivity(
                trade_id="t1",
                amount=100_000_000_000,  # 100B USD
                currency="USD",
                tenor=1.0,
            )
        ]
        
        result = calc.calculate(sens, RiskClass.INTEREST_RATE, MarginType.DELTA, "USD")
        
        # Large position should give CR > 1
        assert result.bucket_cr > 1.0
    
    def test_equity_concentration_per_factor(self):
        """Equity CR should be per risk factor."""
        calc = ConcentrationCalculator()
        
        sens = [
            EquityDeltaSensitivity(
                trade_id="t1",
                amount=1_000_000,
                issuer="AAPL",
                bucket_number=8,
            ),
            EquityDeltaSensitivity(
                trade_id="t2",
                amount=2_000_000,
                issuer="GOOGL",
                bucket_number=8,
            ),
        ]
        
        result = calc.calculate(sens, RiskClass.EQUITY, MarginType.DELTA, 8)
        
        # Each issuer should have its own CR
        assert "AAPL" in result.cr_values
        assert "GOOGL" in result.cr_values
    
    def test_g_bc_calculation(self):
        """Test g_bc factor calculation."""
        # Equal CR values
        g = ConcentrationCalculator.calculate_g_bc(1.5, 1.5)
        assert g == 1.0
        
        # Different CR values
        g = ConcentrationCalculator.calculate_g_bc(1.0, 2.0)
        assert g == 0.5
        
        # Order shouldn't matter
        g1 = ConcentrationCalculator.calculate_g_bc(1.0, 2.0)
        g2 = ConcentrationCalculator.calculate_g_bc(2.0, 1.0)
        assert g1 == g2


class TestWeightedSensitivityCalculator:
    """Tests for WeightedSensitivityCalculator."""
    
    def test_weighted_sensitivity_basic(self):
        """Test basic WS = RW × s × CR calculation."""
        calc = WeightedSensitivityCalculator()
        
        sens = [
            EquityDeltaSensitivity(
                trade_id="t1",
                amount=1_000_000,
                issuer="AAPL",
                bucket_number=8,
            )
        ]
        
        cr_values = {"AAPL": 1.0}
        
        result = calc.calculate(sens, RiskClass.EQUITY, MarginType.DELTA, 8, cr_values)
        
        assert len(result) == 1
        ws = result[0]
        
        # WS should be RW × amount × CR
        assert ws.weighted_value == ws.risk_weight * 1_000_000 * 1.0
        assert ws.concentration_factor == 1.0
    
    def test_weighted_sensitivity_with_cr(self):
        """Test WS with concentration risk > 1."""
        calc = WeightedSensitivityCalculator()
        
        sens = [
            EquityDeltaSensitivity(
                trade_id="t1",
                amount=1_000_000,
                issuer="AAPL",
                bucket_number=8,
            )
        ]
        
        cr_values = {"AAPL": 2.0}  # CR = 2
        
        result = calc.calculate(sens, RiskClass.EQUITY, MarginType.DELTA, 8, cr_values)
        
        ws = result[0]
        
        # WS should be doubled due to CR = 2
        assert ws.concentration_factor == 2.0


class TestBucketAggregator:
    """Tests for BucketAggregator."""
    
    def test_single_sensitivity(self):
        """Single sensitivity K_b = |WS|."""
        agg = BucketAggregator()
        ws_calc = WeightedSensitivityCalculator()
        
        sens = [
            EquityDeltaSensitivity(
                trade_id="t1",
                amount=1_000_000,
                issuer="AAPL",
                bucket_number=8,
            )
        ]
        
        cr_values = {"AAPL": 1.0}
        ws_list = ws_calc.calculate(sens, RiskClass.EQUITY, MarginType.DELTA, 8, cr_values)
        
        result = agg.aggregate(ws_list, RiskClass.EQUITY, MarginType.DELTA, 8, cr_values)
        
        # Single sensitivity: K_b = |WS|
        expected_k = abs(ws_list[0].weighted_value)
        assert abs(result.k_b - expected_k) < 1e-6
    
    def test_two_uncorrelated_sensitivities(self):
        """Two uncorrelated sensitivities: K_b = sqrt(WS1² + WS2²)."""
        agg = BucketAggregator()
        
        # Create weighted sensitivities directly for controlled test
        from simm.engines.aggregation.weighted_sensitivity import WeightedSensitivity
        
        ws_list = [
            WeightedSensitivity(
                original=None,
                qualifier="AAPL",
                bucket=8,
                risk_weight=1.0,
                concentration_factor=1.0,
                weighted_value=3_000_000,
            ),
            WeightedSensitivity(
                original=None,
                qualifier="GOOGL",
                bucket=8,
                risk_weight=1.0,
                concentration_factor=1.0,
                weighted_value=4_000_000,
            ),
        ]
        
        cr_values = {"AAPL": 1.0, "GOOGL": 1.0}
        
        result = agg.aggregate(ws_list, RiskClass.EQUITY, MarginType.DELTA, 8, cr_values)
        
        # With correlation ρ, K_b = sqrt(WS1² + WS2² + 2*ρ*WS1*WS2)
        # The exact value depends on the correlation
        assert result.k_b > 0
        assert result.ws_sum == 7_000_000
    
    def test_residual_bucket(self):
        """Residual bucket has no diversification."""
        agg = BucketAggregator()
        
        from simm.engines.aggregation.weighted_sensitivity import WeightedSensitivity
        
        ws_list = [
            WeightedSensitivity(
                original=None,
                qualifier="A",
                bucket="Residual",
                risk_weight=1.0,
                concentration_factor=1.0,
                weighted_value=1_000_000,
            ),
            WeightedSensitivity(
                original=None,
                qualifier="B",
                bucket="Residual",
                risk_weight=1.0,
                concentration_factor=1.0,
                weighted_value=-500_000,
            ),
        ]
        
        cr_values = {}
        
        result = agg.aggregate(ws_list, RiskClass.EQUITY, MarginType.DELTA, "Residual", cr_values)
        
        # Residual: K_b = sum of absolute values
        assert result.is_residual
        assert result.k_b == 1_500_000  # |1M| + |-0.5M|


class TestRiskClassAggregator:
    """Tests for RiskClassAggregator."""
    
    def test_single_bucket(self):
        """Single bucket margin = K_b."""
        from simm.engines.aggregation.bucket_aggregator import BucketResult
        
        agg = RiskClassAggregator()
        
        bucket_results = [
            BucketResult(
                risk_class=RiskClass.EQUITY,
                margin_type=MarginType.DELTA,
                bucket=8,
                k_b=5_000_000,
                ws_sum=5_000_000,
                is_residual=False,
            )
        ]
        
        result = agg.aggregate_delta(bucket_results, RiskClass.EQUITY)
        
        # Single bucket: margin = K_b
        assert result.margin == 5_000_000
    
    def test_with_residual_bucket(self):
        """Residual bucket is added without diversification."""
        from simm.engines.aggregation.bucket_aggregator import BucketResult
        
        agg = RiskClassAggregator()
        
        bucket_results = [
            BucketResult(
                risk_class=RiskClass.EQUITY,
                margin_type=MarginType.DELTA,
                bucket=8,
                k_b=5_000_000,
                ws_sum=5_000_000,
                is_residual=False,
            ),
            BucketResult(
                risk_class=RiskClass.EQUITY,
                margin_type=MarginType.DELTA,
                bucket="Residual",
                k_b=1_000_000,
                ws_sum=1_000_000,
                is_residual=True,
            ),
        ]
        
        result = agg.aggregate_delta(bucket_results, RiskClass.EQUITY)
        
        # Margin = aggregated non-residual + residual
        assert result.margin == 6_000_000
        assert result.residual_margin == 1_000_000


class TestProductClassAggregator:
    """Tests for ProductClassAggregator."""
    
    def test_single_risk_class(self):
        """Single risk class: SIMM = IM."""
        from simm.engines.aggregation.risk_class_aggregator import RiskClassResult
        
        agg = ProductClassAggregator()
        
        risk_class_results = {
            RiskClass.EQUITY: {
                MarginType.DELTA: RiskClassResult(
                    risk_class=RiskClass.EQUITY,
                    margin_type=MarginType.DELTA,
                    margin=10_000_000,
                )
            }
        }
        
        result = agg.aggregate(risk_class_results, ProductClass.EQUITY)
        
        # Single risk class: SIMM = IM
        assert result.margin == 10_000_000


class TestSIMMCalculator:
    """Tests for the main SIMMCalculator."""
    
    def test_empty_sensitivities(self):
        """Empty sensitivities should return zero margin."""
        calc = SIMMCalculator()
        collection = SensitivityCollection()
        
        result = calc.calculate(collection)
        
        assert result.total_margin == 0.0
    
    def test_single_equity_delta(self):
        """Single equity delta sensitivity."""
        config = SIMMConfig(
            calculate_delta=True,
            calculate_vega=False,
            calculate_curvature=False,
            calculate_base_corr=False,
        )
        calc = SIMMCalculator(config)
        
        collection = SensitivityCollection()
        collection.add(EquityDeltaSensitivity(
            trade_id="t1",
            amount=1_000_000,
            issuer="AAPL",
            bucket_number=8,
        ))
        
        result = calc.calculate(collection)
        
        # Should have non-zero equity margin
        assert result.total_margin > 0
        assert result.by_risk_class.get(RiskClass.EQUITY, 0) > 0
        assert result.by_product_class.get(ProductClass.EQUITY, 0) > 0
    
    def test_multiple_risk_classes(self):
        """Multiple risk classes should aggregate correctly."""
        config = SIMMConfig(
            calculate_delta=True,
            calculate_vega=False,
            calculate_curvature=False,
            calculate_base_corr=False,
        )
        calc = SIMMCalculator(config)
        
        collection = SensitivityCollection()
        
        # Add equity sensitivity
        collection.add(EquityDeltaSensitivity(
            trade_id="t1",
            amount=1_000_000,
            issuer="AAPL",
            bucket_number=8,
        ))
        
        # Add IR sensitivity
        collection.add(IRDeltaSensitivity(
            trade_id="t2",
            amount=2_000_000,
            currency="USD",
            tenor=1.0,
        ))
        
        # Add FX sensitivity
        collection.add(FXDeltaSensitivity(
            trade_id="t3",
            amount=500_000,
            currency_pair="EURUSD",
        ))
        
        result = calc.calculate(collection)
        
        # Should have non-zero total
        assert result.total_margin > 0
        
        # Each risk class should contribute
        assert result.by_risk_class.get(RiskClass.EQUITY, 0) > 0
        assert result.by_risk_class.get(RiskClass.INTEREST_RATE, 0) > 0
        assert result.by_risk_class.get(RiskClass.FX, 0) > 0
        
        # Product classes
        assert result.by_product_class.get(ProductClass.EQUITY, 0) > 0
        assert result.by_product_class.get(ProductClass.RATES_FX, 0) > 0
    
    def test_with_addon(self):
        """Test with fixed add-on."""
        config = SIMMConfig(
            calculate_delta=True,
            calculate_vega=False,
            calculate_curvature=False,
            calculate_base_corr=False,
            addon_fixed=100_000,
        )
        calc = SIMMCalculator(config)
        
        collection = SensitivityCollection()
        collection.add(EquityDeltaSensitivity(
            trade_id="t1",
            amount=1_000_000,
            issuer="AAPL",
            bucket_number=8,
        ))
        
        result = calc.calculate(collection)
        
        # Total should include add-on
        assert result.addon is not None
        assert result.addon.fixed_addon == 100_000
        assert result.addon.total_addon == 100_000
    
    def test_with_multiplier(self):
        """Test with product class multiplier."""
        config = SIMMConfig(
            calculate_delta=True,
            calculate_vega=False,
            calculate_curvature=False,
            calculate_base_corr=False,
            ms_equity=1.5,  # 50% extra for equity
        )
        calc = SIMMCalculator(config)
        
        collection = SensitivityCollection()
        collection.add(EquityDeltaSensitivity(
            trade_id="t1",
            amount=1_000_000,
            issuer="AAPL",
            bucket_number=8,
        ))
        
        result = calc.calculate(collection)
        
        # Equity margin should have 1.5x multiplier applied
        base_calc = SIMMCalculator(SIMMConfig(
            calculate_delta=True,
            calculate_vega=False,
            calculate_curvature=False,
            calculate_base_corr=False,
        ))
        base_result = base_calc.calculate(collection)
        
        expected_equity = base_result.by_product_class.get(ProductClass.EQUITY, 0) * 1.5
        assert abs(result.by_product_class.get(ProductClass.EQUITY, 0) - expected_equity) < 1e-6
    
    def test_result_to_dict(self):
        """Test result serialization."""
        config = SIMMConfig(
            calculate_delta=True,
            calculate_vega=False,
            calculate_curvature=False,
            calculate_base_corr=False,
        )
        calc = SIMMCalculator(config)
        
        collection = SensitivityCollection()
        collection.add(EquityDeltaSensitivity(
            trade_id="t1",
            amount=1_000_000,
            issuer="AAPL",
            bucket_number=8,
        ))
        
        result = calc.calculate(collection)
        d = result.to_dict()
        
        assert "total_margin" in d
        assert "by_product_class" in d
        assert "by_risk_class" in d
        assert "calculation_currency" in d


class TestNumericalPrecision:
    """Tests for numerical precision and edge cases."""
    
    def test_zero_sensitivity(self):
        """Zero sensitivity should give zero margin."""
        calc = SIMMCalculator()
        
        collection = SensitivityCollection()
        collection.add(EquityDeltaSensitivity(
            trade_id="t1",
            amount=0.0,
            issuer="AAPL",
            bucket_number=8,
        ))
        
        result = calc.calculate(collection)
        
        assert result.by_risk_class.get(RiskClass.EQUITY, 0) == 0.0
    
    def test_offsetting_sensitivities(self):
        """Offsetting sensitivities should net out."""
        config = SIMMConfig(
            calculate_delta=True,
            calculate_vega=False,
            calculate_curvature=False,
            calculate_base_corr=False,
        )
        calc = SIMMCalculator(config)
        
        collection = SensitivityCollection()
        collection.add(EquityDeltaSensitivity(
            trade_id="t1",
            amount=1_000_000,
            issuer="AAPL",
            bucket_number=8,
        ))
        collection.add(EquityDeltaSensitivity(
            trade_id="t2",
            amount=-1_000_000,  # Offsetting
            issuer="AAPL",
            bucket_number=8,
        ))
        
        result = calc.calculate(collection)
        
        # Same issuer, same bucket - should net to zero
        assert result.by_risk_class.get(RiskClass.EQUITY, 0) == 0.0
    
    def test_very_small_sensitivity(self):
        """Very small sensitivities should be handled without numerical issues."""
        calc = SIMMCalculator()
        
        collection = SensitivityCollection()
        collection.add(EquityDeltaSensitivity(
            trade_id="t1",
            amount=1e-10,  # Very small
            issuer="AAPL",
            bucket_number=8,
        ))
        
        result = calc.calculate(collection)
        
        # Should not raise any errors
        assert result.total_margin >= 0


class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_simple_portfolio(self):
        """Test a simple multi-asset portfolio."""
        config = SIMMConfig()
        calc = SIMMCalculator(config)
        
        collection = SensitivityCollection()
        
        # Equity positions
        collection.add(EquityDeltaSensitivity(
            trade_id="eq1",
            amount=5_000_000,
            issuer="AAPL",
            bucket_number=8,
        ))
        collection.add(EquityDeltaSensitivity(
            trade_id="eq2",
            amount=3_000_000,
            issuer="GOOGL",
            bucket_number=8,
        ))
        
        # IR positions
        collection.add(IRDeltaSensitivity(
            trade_id="ir1",
            amount=10_000_000,
            currency="USD",
            tenor=5.0,
        ))
        collection.add(IRDeltaSensitivity(
            trade_id="ir2",
            amount=-5_000_000,
            currency="EUR",
            tenor=10.0,
        ))
        
        # Credit position
        collection.add(CreditDeltaSensitivity(
            trade_id="cr1",
            amount=2_000_000,
            issuer="IBM",
            bucket_number=3,
            tenor=5.0,
        ))
        
        result = calc.calculate(collection)
        
        # Verify structure
        assert result.total_margin > 0
        assert result.by_product_class[ProductClass.EQUITY] > 0
        assert result.by_product_class[ProductClass.RATES_FX] > 0
        assert result.by_product_class[ProductClass.CREDIT] > 0
        
        # Verify attribution
        assert RiskClass.EQUITY in result.by_margin_type
        assert MarginType.DELTA in result.by_margin_type[RiskClass.EQUITY]
        
        # Total should be sum of product classes (plus any add-ons)
        pc_sum = sum(result.by_product_class.values())
        assert abs(result.total_margin - pc_sum) < 1e-6  # No add-ons in this test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
