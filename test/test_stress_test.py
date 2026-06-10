"""
Unit tests for the Stress Test module.
"""

import unittest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

try:
    import yaml  # noqa: F401
except ImportError:
    yaml = None

from quantark.portfolio import Portfolio
from quantark.portfolio.fi import FIPortfolio
from quantark.priceenv import PricingEnvironment
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.asset.equity.product import EuropeanVanillaOption
from quantark.asset.equity.engine import BlackScholesEngine
from quantark.asset.bond.product.couponbond.fixed_bond import FixedBond
from quantark.asset.bond.engine.discount.bond_discount_engine import BondDiscountEngine
from quantark.util.enum import OptionType, PaymentFrequency
from quantark.util.calendar import DayCountConvention
from quantark.param.rrf.rate_curve import LinearRateCurve

from quantark.stresstest import (
    StressTestEngine,
    StressTestConfig,
    ScenarioBuilder,
    StressType,
    StressLevel,
    Scenario,
    Stress,
)
from quantark.stresstest.fi import FIStressConfig, FIStressEngine
from quantark.stresstest.scenario.scenario_library import ScenarioLibrary
from quantark.stresstest.scenario.scenario_storage import ScenarioStorage
from quantark.stresstest.stress.stress_applicator import StressApplicator
from quantark.stresstest.results.result_aggregator import ResultAggregator
from quantark.stresstest.results.result_exporter import ResultExporter


class TestScenarioDefinition(unittest.TestCase):
    """Test scenario and stress definitions."""
    
    def test_stress_creation(self):
        """Test creating a Stress object."""
        stress = Stress(
            parameter="spot",
            stress_type=StressType.PERCENTAGE,
            stress_value=-0.20,
            level=StressLevel.PORTFOLIO
        )
        
        self.assertEqual(stress.parameter, "spot")
        self.assertEqual(stress.stress_type, StressType.PERCENTAGE)
        self.assertEqual(stress.stress_value, -0.20)
        self.assertEqual(stress.level, StressLevel.PORTFOLIO)
        self.assertIsNone(stress.target)
    
    def test_stress_with_target(self):
        """Test creating a Stress with target."""
        stress = Stress(
            parameter="volatility",
            stress_type=StressType.PERCENTAGE,
            stress_value=0.50,
            level=StressLevel.UNDERLYING,
            target="AAPL"
        )
        
        self.assertEqual(stress.target, "AAPL")
    
    def test_scenario_creation(self):
        """Test creating a Scenario."""
        stresses = [
            Stress("spot", StressType.PERCENTAGE, -0.20, StressLevel.PORTFOLIO),
            Stress("volatility", StressType.PERCENTAGE, 0.50, StressLevel.PORTFOLIO),
        ]
        
        scenario = Scenario(
            name="Test Scenario",
            stresses=stresses,
            description="Test description"
        )
        
        self.assertEqual(scenario.name, "Test Scenario")
        self.assertEqual(len(scenario.stresses), 2)
        self.assertEqual(scenario.description, "Test description")
    
    def test_scenario_serialization(self):
        """Test scenario to/from dict."""
        stresses = [
            Stress("spot", StressType.PERCENTAGE, -0.20, StressLevel.PORTFOLIO),
        ]
        scenario = Scenario(name="Test", stresses=stresses)
        
        # To dict
        data = scenario.to_dict()
        self.assertIn("name", data)
        self.assertIn("stresses", data)
        
        # From dict
        loaded = Scenario.from_dict(data)
        self.assertEqual(loaded.name, scenario.name)
        self.assertEqual(len(loaded.stresses), len(scenario.stresses))


class TestScenarioBuilder(unittest.TestCase):
    """Test ScenarioBuilder."""
    
    def test_builder_basic(self):
        """Test basic builder usage."""
        scenario = (ScenarioBuilder()
            .name("Test")
            .description("Test scenario")
            .spot_stress(-0.10)
            .build()
        )
        
        self.assertEqual(scenario.name, "Test")
        self.assertEqual(len(scenario.stresses), 1)
        self.assertEqual(scenario.stresses[0].parameter, "spot")
    
    def test_builder_multiple_stresses(self):
        """Test builder with multiple stresses."""
        scenario = (ScenarioBuilder()
            .name("Multi")
            .spot_stress(-0.15)
            .vol_stress(0.40)
            .rate_stress(0.01, stress_type=StressType.ABSOLUTE)
            .build()
        )
        
        self.assertEqual(len(scenario.stresses), 3)
    
    def test_builder_with_underlying(self):
        """Test builder with underlying target."""
        scenario = (ScenarioBuilder()
            .name("AAPL Stress")
            .spot_stress(-0.20, underlying="AAPL")
            .build()
        )
        
        self.assertEqual(scenario.stresses[0].level, StressLevel.UNDERLYING)
        self.assertEqual(scenario.stresses[0].target, "AAPL")

    def test_key_rate_helper(self):
        """Ensure key-rate helper annotates metadata."""
        scenario = (
            ScenarioBuilder()
            .name("Key Rate")
            .key_rate_stress(0.01, tenor_bucket="5Y", curve="UST")
            .build()
        )
        stress = scenario.stresses[0]
        self.assertEqual(stress.parameter, "key_rate")
        self.assertEqual(stress.metadata.get("tenor_bucket"), "5Y")
        self.assertEqual(stress.metadata.get("curve"), "UST")

    def test_spread_helper(self):
        """Ensure spread helper encodes metadata."""
        scenario = (
            ScenarioBuilder()
            .name("Spread Shock")
            .spread_stress(0.0025, spread_curve="IG")
            .build()
        )
        stress = scenario.stresses[0]
        self.assertEqual(stress.parameter, "spread")
        self.assertEqual(stress.metadata.get("spread_curve"), "IG")


class TestScenarioLibrary(unittest.TestCase):
    """Test predefined scenarios."""
    
    def test_market_crash(self):
        """Test market crash scenario."""
        scenario = ScenarioLibrary.market_crash()
        
        self.assertEqual(scenario.name, "Market Crash")
        self.assertGreater(len(scenario.stresses), 0)
    
    def test_get_all_predefined(self):
        """Test getting all predefined scenarios."""
        scenarios = ScenarioLibrary.get_all_predefined()
        
        self.assertGreater(len(scenarios), 0)
        self.assertIsInstance(scenarios[0], Scenario)
    
    def test_historical_scenarios(self):
        """Test historical scenarios."""
        scenarios = ScenarioLibrary.get_historical_scenarios()
        
        self.assertGreater(len(scenarios), 0)
        # Check for specific historical events
        names = [s.name for s in scenarios]
        self.assertIn("Black Monday 1987", names)

    def test_fi_parallel_shift(self):
        """FI scenario should emit key-rate stresses."""
        scenario = ScenarioLibrary.fi_parallel_shift()
        params = {stress.parameter for stress in scenario.stresses}
        self.assertIn("key_rate", params)

    def test_fi_spread_shock(self):
        """FI spread scenario should encode metadata."""
        scenario = ScenarioLibrary.fi_spread_shock(spread_bps=0.003)
        self.assertEqual(scenario.stresses[0].parameter, "spread")
        self.assertEqual(scenario.stresses[0].metadata.get("spread_curve"), "IG")


class TestScenarioStorage(unittest.TestCase):
    """Test scenario storage."""
    
    def setUp(self):
        """Create temporary directory for tests."""
        if yaml is None:
            self.skipTest("PyYAML not installed")
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_load_yaml(self):
        """Test saving and loading YAML."""
        scenarios = [
            ScenarioBuilder().name("Test1").spot_stress(-0.10).build(),
            ScenarioBuilder().name("Test2").vol_stress(0.20).build(),
        ]
        
        filepath = self.temp_dir / "scenarios.yaml"
        ScenarioStorage.save_scenarios(scenarios, filepath)
        
        self.assertTrue(filepath.exists())
        
        loaded = ScenarioStorage.load_scenarios(filepath)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].name, "Test1")
    
    def test_save_load_json(self):
        """Test saving and loading JSON."""
        scenarios = [
            ScenarioBuilder().name("Test1").spot_stress(-0.10).build(),
        ]
        
        filepath = self.temp_dir / "scenarios.json"
        ScenarioStorage.save_scenarios(scenarios, filepath, format='json')
        
        self.assertTrue(filepath.exists())
        
        loaded = ScenarioStorage.load_scenarios(filepath, format='json')
        self.assertEqual(len(loaded), 1)


class TestStressTypes(unittest.TestCase):
    """Test stress type applications."""
    
    def test_percentage_stress(self):
        """Test percentage stress."""
        result = StressType.PERCENTAGE.apply(100.0, 0.10)
        self.assertAlmostEqual(result, 110.0)
        
        result = StressType.PERCENTAGE.apply(100.0, -0.20)
        self.assertAlmostEqual(result, 80.0)
    
    def test_absolute_stress(self):
        """Test absolute stress."""
        result = StressType.ABSOLUTE.apply(0.05, 0.02)
        self.assertAlmostEqual(result, 0.07)
        
        result = StressType.ABSOLUTE.apply(0.05, -0.01)
        self.assertAlmostEqual(result, 0.04)
    
    def test_value_stress(self):
        """Test value stress."""
        result = StressType.VALUE.apply(100.0, 50.0)
        self.assertAlmostEqual(result, 50.0)


class TestStressTestEngine(unittest.TestCase):
    """Test stress test engine."""
    
    def setUp(self):
        """Create test portfolio."""
        self.valuation_date = datetime(2024, 1, 15)
        self.maturity_date = datetime(2024, 7, 15)
        
        # Create pricing environment
        env = PricingEnvironment(
            spot_quote=SpotQuote(spot=100.0, asset_name="TEST"),
            vol_surface=FlatVolSurface(volatility=0.25),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.01),
            valuation_date=self.valuation_date
        )
        
        # Create portfolio
        self.portfolio = Portfolio(
            portfolio_name="Test Portfolio",
            pricing_environments={"TEST": env},
            creation_date=self.valuation_date
        )
        
        # Add a position
        time_to_maturity = (self.maturity_date - self.valuation_date).days / 365.0
        option = EuropeanVanillaOption(
            strike=100.0,
            maturity=time_to_maturity,
            option_type=OptionType.CALL
        )
        engine = BlackScholesEngine()
        
        self.portfolio.add_position(
            product=option,
            quantity=10,
            entry_price=engine.price(option, env),
            underlying="TEST",
            engine=engine,
            entry_timestamp=self.valuation_date
        )
    
    def test_engine_creation(self):
        """Test creating engine."""
        config = StressTestConfig()
        engine = StressTestEngine(config)
        
        self.assertIsNotNone(engine)
        self.assertEqual(engine.config, config)
    
    def test_run_single_scenario(self):
        """Test running a single scenario."""
        scenario = ScenarioBuilder().name("Test").spot_stress(-0.10).build()
        
        config = StressTestConfig(calculate_greeks=False)
        engine = StressTestEngine(config)
        
        results = engine.run_static_scenarios(self.portfolio, [scenario])
        
        self.assertEqual(len(results.scenario_results), 1)
        self.assertIsNotNone(results.baseline_value)
        self.assertGreater(results.baseline_value, 0)
    
    def test_run_multiple_scenarios(self):
        """Test running multiple scenarios."""
        scenarios = [
            ScenarioBuilder().name("Down10").spot_stress(-0.10).build(),
            ScenarioBuilder().name("Down20").spot_stress(-0.20).build(),
            ScenarioBuilder().name("Up10").spot_stress(0.10).build(),
        ]
        
        config = StressTestConfig(calculate_greeks=False)
        engine = StressTestEngine(config)
        
        results = engine.run_static_scenarios(self.portfolio, scenarios)
        
        self.assertEqual(len(results.scenario_results), 3)
        
        # Worst should be Down20
        worst = results.get_worst_scenario()
        self.assertEqual(worst.scenario.name, "Down20")
        
        # Best should be Up10
        best = results.get_best_scenario()
        self.assertEqual(best.scenario.name, "Up10")


class TestFIStressEngine(unittest.TestCase):
    """Test FI-specific stress engine."""

    def setUp(self):
        valuation_date = datetime(2025, 1, 2)
        self.portfolio = FIPortfolio(
            portfolio_name="FI Test",
            pricing_environments={
                "UST_10Y": PricingEnvironment(
                    rate_curve=LinearRateCurve([(2, 0.03), (5, 0.032), (10, 0.035), (30, 0.04)]),
                    valuation_date=valuation_date,
                ),
                "UST_30Y": PricingEnvironment(
                    rate_curve=LinearRateCurve([(2, 0.03), (5, 0.032), (10, 0.035), (30, 0.04)]),
                    valuation_date=valuation_date,
                ),
            },
            creation_date=valuation_date,
        )
        engine = BondDiscountEngine(self.portfolio.pricing_environments["UST_10Y"])
        ten_year = FixedBond(
            issue_date=datetime(2019, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            denominator=100.0,
            coupon_rate=0.04,
            payment_frequency=PaymentFrequency.SEMI_ANNUAL,
            day_count_convention=DayCountConvention.ACT_365,
        )
        self.portfolio.add_position(
            product=ten_year,
            quantity=25,
            entry_price=100.0,
            underlying="UST_10Y",
            engine=engine,
            entry_timestamp=valuation_date,
        )

    def test_fi_engine_results(self):
        engine = FIStressEngine(FIStressConfig(save_detailed_results=False))
        scenario = ScenarioLibrary.fi_parallel_shift(0.01)
        results = engine.run_static_scenarios(self.portfolio, [scenario])

        self.assertTrue(hasattr(results, "get_dv01_series"))
        dv01_df = results.get_dv01_series()
        self.assertFalse(dv01_df.empty)
        self.assertIn("fi", results.extra_metrics)
        self.assertNotAlmostEqual(
            results.scenario_results[0].portfolio_value,
            results.baseline_value,
        )


class TestResultAggregator(unittest.TestCase):
    """Test result aggregation."""
    
    def setUp(self):
        """Create test results."""
        # Create simple test results (minimal setup)
        from quantark.stresstest.results.stress_results import StressTestResults, ScenarioResult
        
        scenarios = [
            ScenarioBuilder().name("S1").spot_stress(-0.10).build(),
            ScenarioBuilder().name("S2").spot_stress(-0.20).build(),
        ]
        
        scenario_results = [
            ScenarioResult(
                scenario=scenarios[0],
                portfolio_value=90000,
                portfolio_pnl=-10000,
                portfolio_pnl_pct=-10.0
            ),
            ScenarioResult(
                scenario=scenarios[1],
                portfolio_value=80000,
                portfolio_pnl=-20000,
                portfolio_pnl_pct=-20.0
            ),
        ]
        
        self.results = StressTestResults(
            baseline_value=100000,
            baseline_greeks=None,
            scenario_results=scenario_results
        )
    
    def test_get_risk_summary(self):
        """Test risk summary calculation."""
        summary = ResultAggregator.get_risk_summary(self.results)
        
        self.assertIn('baseline_value', summary)
        self.assertIn('worst_pnl', summary)
        self.assertIn('best_pnl', summary)
        self.assertEqual(summary['baseline_value'], 100000)
    
    def test_compare_scenarios(self):
        """Test scenario comparison."""
        df = ResultAggregator.compare_scenarios(self.results)
        
        self.assertEqual(len(df), 2)
        self.assertIn('portfolio_pnl', df.columns)
    
    def test_calculate_var_cvar(self):
        """Test VaR/CVaR calculation."""
        metrics = ResultAggregator.calculate_var_cvar(self.results, confidence_level=0.95)
        
        self.assertIn('var', metrics)
        self.assertIn('cvar', metrics)


class TestConfig(unittest.TestCase):
    """Test configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = StressTestConfig()
        
        self.assertTrue(config.calculate_greeks)
        self.assertEqual(config.greeks_method, 'analytical')
        self.assertIn('parquet', config.export_formats)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = StressTestConfig(
            calculate_greeks=False,
            greeks_method='numerical',
            export_formats=['csv'],
            output_dir='./custom_output'
        )
        
        self.assertFalse(config.calculate_greeks)
        self.assertEqual(config.greeks_method, 'numerical')
        self.assertEqual(config.export_formats, ['csv'])
        self.assertEqual(config.output_dir, './custom_output')


if __name__ == '__main__':
    unittest.main()
