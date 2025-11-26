"""
Comprehensive tests for European bond option implementation.
"""
import unittest
from datetime import datetime
import math

from asset.bond.product.couponbond.fixed_bond import FixedBond, create_simple_fixed_bond
from asset.bond.product.option.euro_short_term_bond_option import (
    EuroShortTermBondOption,
    create_bond_option,
)
from asset.bond.engine.analytical.black_engine import BlackBondOptionEngine
from asset.bond.riskmeasures.bond_greeks_calculator import BondGreeksCalculator
from param.rrf.rate_curve import FlatRateCurve
from param.vol import FlatVolSurface
from priceenv import PricingEnvironment
from util.enum import OptionType, PaymentFrequency
from util.exceptions import ValidationError, PricingError


class TestBondOptionCreation(unittest.TestCase):
    """Test bond option product creation and validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.underlying = create_simple_fixed_bond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            notional=1000.0,
            coupon_rate=0.05,
            payment_frequency=PaymentFrequency.SEMI_ANNUAL
        )
    
    def test_create_call_option(self):
        """Test creating a call option."""
        option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=1000.0,
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.CALL
        )
        
        self.assertTrue(option.is_call())
        self.assertFalse(option.is_put())
        self.assertEqual(option.strike, 1000.0)
        self.assertEqual(option.notional, 1.0)
    
    def test_create_put_option(self):
        """Test creating a put option."""
        option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=950.0,
            expiry_date=datetime(2025, 6, 1),
            option_type=OptionType.PUT
        )
        
        self.assertFalse(option.is_call())
        self.assertTrue(option.is_put())
        self.assertEqual(option.strike, 950.0)
    
    def test_create_with_factory_function(self):
        """Test creating option with factory function."""
        option = create_bond_option(
            underlying=self.underlying,
            strike=1050.0,
            expiry_date=datetime(2025, 3, 1),
            option_type=OptionType.CALL,
            notional=10.0
        )
        
        self.assertEqual(option.notional, 10.0)
        self.assertTrue(option.strike_is_clean)
    
    def test_invalid_strike(self):
        """Test that invalid strike raises error."""
        with self.assertRaises(ValidationError):
            EuroShortTermBondOption(
                underlying=self.underlying,
                strike=-100.0,  # Invalid
                expiry_date=datetime(2025, 1, 1),
                option_type=OptionType.CALL
            )
    
    def test_expiry_after_maturity(self):
        """Test that expiry after bond maturity raises error."""
        with self.assertRaises(ValidationError):
            EuroShortTermBondOption(
                underlying=self.underlying,
                strike=1000.0,
                expiry_date=datetime(2030, 1, 1),  # After maturity
                option_type=OptionType.CALL
            )
    
    def test_time_to_expiry(self):
        """Test time to expiry calculation."""
        option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=1000.0,
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.CALL
        )
        
        valuation_date = datetime(2024, 1, 1)
        ttm = option.get_time_to_expiry(valuation_date)
        
        # Should be close to 1 year
        self.assertAlmostEqual(ttm, 1.0, delta=0.01)
    
    def test_is_expired(self):
        """Test expiry detection."""
        option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=1000.0,
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.CALL
        )
        
        self.assertFalse(option.is_expired(datetime(2024, 6, 1)))
        self.assertTrue(option.is_expired(datetime(2025, 1, 1)))
        self.assertTrue(option.is_expired(datetime(2025, 6, 1)))


class TestBondOptionPayoff(unittest.TestCase):
    """Test option payoff calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.underlying = create_simple_fixed_bond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            notional=1000.0,
            coupon_rate=0.05
        )
    
    def test_call_itm_payoff(self):
        """Test ITM call payoff."""
        option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=1000.0,
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.CALL,
            notional=1.0
        )
        
        # Bond at 1050, strike at 1000 -> payoff = 50
        payoff = option.get_payoff(1050.0)
        self.assertEqual(payoff, 50.0)
    
    def test_call_otm_payoff(self):
        """Test OTM call payoff."""
        option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=1000.0,
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.CALL
        )
        
        # Bond at 950, strike at 1000 -> payoff = 0
        payoff = option.get_payoff(950.0)
        self.assertEqual(payoff, 0.0)
    
    def test_put_itm_payoff(self):
        """Test ITM put payoff."""
        option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=1000.0,
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.PUT
        )
        
        # Bond at 950, strike at 1000 -> payoff = 50
        payoff = option.get_payoff(950.0)
        self.assertEqual(payoff, 50.0)
    
    def test_put_otm_payoff(self):
        """Test OTM put payoff."""
        option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=1000.0,
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.PUT
        )
        
        # Bond at 1050, strike at 1000 -> payoff = 0
        payoff = option.get_payoff(1050.0)
        self.assertEqual(payoff, 0.0)
    
    def test_payoff_with_notional(self):
        """Test payoff scales with notional."""
        option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=1000.0,
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.CALL,
            notional=10.0
        )
        
        payoff = option.get_payoff(1050.0)
        self.assertEqual(payoff, 500.0)  # 50 * 10


class TestBlackBondOptionEngine(unittest.TestCase):
    """Test Black model pricing engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.underlying = create_simple_fixed_bond(
            issue_date=datetime(2023, 1, 1),
            maturity_date=datetime(2028, 1, 1),
            notional=1000.0,
            coupon_rate=0.05,
            payment_frequency=PaymentFrequency.SEMI_ANNUAL
        )
        
        self.valuation_date = datetime(2024, 1, 1)
        self.rate_curve = FlatRateCurve(rate=0.04)
        self.vol_surface = FlatVolSurface(volatility=0.10)
        
        self.pricing_env = PricingEnvironment(
            rate_curve=self.rate_curve,
            valuation_date=self.valuation_date,
            vol_surface=self.vol_surface
        )
        
        self.engine = BlackBondOptionEngine(self.pricing_env)
    
    def test_price_call_option(self):
        """Test pricing a call option."""
        option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=1000.0,
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.CALL
        )
        
        price = self.engine.price(option, volatility=0.10)
        
        # Price should be positive
        self.assertGreater(price, 0)
        # Price should be less than bond value
        self.assertLess(price, 1100)
    
    def test_price_put_option(self):
        """Test pricing a put option."""
        option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=1000.0,
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.PUT
        )
        
        price = self.engine.price(option, volatility=0.10)
        
        # Price should be positive
        self.assertGreater(price, 0)
    
    def test_put_call_parity(self):
        """Test put-call parity relationship."""
        expiry_date = datetime(2025, 1, 1)
        strike = 1000.0
        vol = 0.10
        
        call = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=strike,
            expiry_date=expiry_date,
            option_type=OptionType.CALL
        )
        
        put = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=strike,
            expiry_date=expiry_date,
            option_type=OptionType.PUT
        )
        
        call_price = self.engine.price(call, volatility=vol)
        put_price = self.engine.price(put, volatility=vol)
        
        # Get forward price and discount factor
        results = self.engine.price_with_details(call, volatility=vol)
        F = results.forward_bond_price
        D = results.discount_factor
        
        # Put-Call Parity: C - P = D * (F - K)
        lhs = call_price - put_price
        rhs = D * (F - strike)
        
        self.assertAlmostEqual(lhs, rhs, delta=0.01)
    
    def test_atm_option_value(self):
        """Test ATM option has positive value."""
        # Get forward price to set ATM strike
        call = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=1000.0,  # Approximate ATM
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.CALL
        )
        
        results = self.engine.price_with_details(call, volatility=0.10)
        
        # Create ATM option
        atm_option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=results.forward_bond_price,
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.CALL
        )
        
        atm_price = self.engine.price(atm_option, volatility=0.10)
        
        # ATM should have positive time value
        self.assertGreater(atm_price, 0)
    
    def test_vol_increases_price(self):
        """Test that higher volatility increases option price."""
        option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=1000.0,
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.CALL
        )
        
        price_low_vol = self.engine.price(option, volatility=0.05)
        price_high_vol = self.engine.price(option, volatility=0.15)
        
        self.assertGreater(price_high_vol, price_low_vol)
    
    def test_expired_option_raises_error(self):
        """Test that pricing expired option raises error."""
        option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=1000.0,
            expiry_date=datetime(2023, 6, 1),  # Before valuation
            option_type=OptionType.CALL
        )
        
        with self.assertRaises(PricingError):
            self.engine.price(option, volatility=0.10)
    
    def test_price_with_details(self):
        """Test price_with_details returns complete results."""
        option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=1000.0,
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.CALL
        )
        
        results = self.engine.price_with_details(option, volatility=0.10)
        
        self.assertGreater(results.price, 0)
        self.assertGreater(results.forward_bond_price, 0)
        self.assertGreater(results.discount_factor, 0)
        self.assertLess(results.discount_factor, 1)
        self.assertGreater(results.time_to_expiry, 0)


class TestBondGreeksCalculator(unittest.TestCase):
    """Test Greeks calculation for bond options."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.underlying = create_simple_fixed_bond(
            issue_date=datetime(2023, 1, 1),
            maturity_date=datetime(2028, 1, 1),
            notional=1000.0,
            coupon_rate=0.05,
            payment_frequency=PaymentFrequency.SEMI_ANNUAL
        )
        
        self.valuation_date = datetime(2024, 1, 1)
        self.rate_curve = FlatRateCurve(rate=0.04)
        self.vol_surface = FlatVolSurface(volatility=0.10)
        
        self.pricing_env = PricingEnvironment(
            rate_curve=self.rate_curve,
            valuation_date=self.valuation_date,
            vol_surface=self.vol_surface
        )
        
        self.option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=1000.0,
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.CALL
        )
        
        self.calculator = BondGreeksCalculator()
    
    def test_analytical_greeks(self):
        """Test analytical Greeks calculation."""
        greeks = self.calculator.calculate_analytical_greeks(
            self.option,
            self.pricing_env,
            volatility=0.10
        )
        
        # Check all Greeks are present
        self.assertIn("price", greeks)
        self.assertIn("delta", greeks)
        self.assertIn("gamma", greeks)
        self.assertIn("vega", greeks)
        self.assertIn("theta", greeks)
        self.assertIn("rho", greeks)
        
        # Price should be positive
        self.assertGreater(greeks["price"], 0)
        
        # Delta for call should be positive and less than 1
        self.assertGreater(greeks["delta"], 0)
        self.assertLess(greeks["delta"], 1.5)
        
        # Gamma should be positive
        self.assertGreater(greeks["gamma"], 0)
        
        # Vega should be positive
        self.assertGreater(greeks["vega"], 0)
        
        # Theta for long option should be negative (time decay)
        self.assertLess(greeks["theta"], 0)
    
    def test_numerical_greeks(self):
        """Test numerical Greeks calculation."""
        greeks = self.calculator.calculate_numerical_greeks(
            self.option,
            self.pricing_env,
            volatility=0.10
        )
        
        # Check all Greeks are present
        self.assertIn("price", greeks)
        self.assertIn("delta", greeks)
        self.assertIn("gamma", greeks)
        self.assertIn("vega", greeks)
        self.assertIn("theta", greeks)
        self.assertIn("rho", greeks)
        self.assertIn("dv01", greeks)
        
        # Price should match engine price
        engine = BlackBondOptionEngine(self.pricing_env)
        expected_price = engine.price(self.option, volatility=0.10)
        self.assertAlmostEqual(greeks["price"], expected_price, delta=0.01)
    
    def test_analytical_vs_numerical_greeks(self):
        """Test that analytical and numerical Greeks are reasonably close."""
        analytical = self.calculator.calculate_analytical_greeks(
            self.option,
            self.pricing_env,
            volatility=0.10
        )
        
        numerical = self.calculator.calculate_numerical_greeks(
            self.option,
            self.pricing_env,
            volatility=0.10
        )
        
        comparison = self.calculator.compare_greeks(analytical, numerical)
        
        # Prices should match
        self.assertAlmostEqual(
            comparison["analytical"]["price"],
            comparison["numerical"]["price"],
            delta=0.01
        )
        
        # Vega should be close
        self.assertAlmostEqual(
            comparison["analytical"]["vega"],
            comparison["numerical"]["vega"],
            delta=1.0
        )
    
    def test_put_greeks(self):
        """Test Greeks for put option."""
        put_option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=1000.0,
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.PUT
        )
        
        greeks = self.calculator.calculate_analytical_greeks(
            put_option,
            self.pricing_env,
            volatility=0.10
        )
        
        # Delta for put should be negative
        self.assertLess(greeks["delta"], 0)
        
        # Gamma should still be positive
        self.assertGreater(greeks["gamma"], 0)
    
    def test_bond_sensitivities(self):
        """Test bond-specific sensitivities."""
        sensitivities = self.calculator.calculate_bond_sensitivities(
            self.option,
            self.pricing_env,
            volatility=0.10
        )
        
        self.assertIn("option_price", sensitivities)
        self.assertIn("underlying_price", sensitivities)
        self.assertIn("option_dv01", sensitivities)
        self.assertIn("option_duration", sensitivities)
        self.assertIn("underlying_dv01", sensitivities)
        self.assertIn("underlying_duration", sensitivities)
        
        # Underlying DV01 should be positive
        self.assertGreater(sensitivities["underlying_dv01"], 0)
        
        # Underlying duration should be positive
        self.assertGreater(sensitivities["underlying_duration"], 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.underlying = create_simple_fixed_bond(
            issue_date=datetime(2024, 1, 1),
            maturity_date=datetime(2029, 1, 1),
            notional=1000.0,
            coupon_rate=0.05
        )
        
        self.valuation_date = datetime(2024, 6, 1)
        self.rate_curve = FlatRateCurve(rate=0.04)
        
        self.pricing_env = PricingEnvironment(
            rate_curve=self.rate_curve,
            valuation_date=self.valuation_date
        )
    
    def test_deep_itm_call(self):
        """Test deep ITM call pricing."""
        option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=800.0,  # Deep ITM
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.CALL
        )
        
        engine = BlackBondOptionEngine(self.pricing_env)
        price = engine.price(option, volatility=0.10)
        
        # Deep ITM should have significant value
        self.assertGreater(price, 100)
    
    def test_deep_otm_put(self):
        """Test deep OTM put pricing."""
        option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=800.0,  # Deep OTM for put
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.PUT
        )
        
        engine = BlackBondOptionEngine(self.pricing_env)
        price = engine.price(option, volatility=0.10)
        
        # Deep OTM should have small but positive value
        self.assertGreater(price, 0)
        self.assertLess(price, 50)
    
    def test_short_expiry(self):
        """Test option with short time to expiry."""
        option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=1000.0,
            expiry_date=datetime(2024, 6, 15),  # 2 weeks
            option_type=OptionType.CALL
        )
        
        engine = BlackBondOptionEngine(self.pricing_env)
        price = engine.price(option, volatility=0.10)
        
        # Should still price correctly
        self.assertGreater(price, 0)
    
    def test_high_volatility(self):
        """Test with high volatility."""
        option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=1000.0,
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.CALL
        )
        
        engine = BlackBondOptionEngine(self.pricing_env)
        price = engine.price(option, volatility=0.30)  # 30% vol
        
        self.assertGreater(price, 0)
    
    def test_low_volatility(self):
        """Test with very low volatility."""
        option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=1000.0,
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.CALL
        )
        
        engine = BlackBondOptionEngine(self.pricing_env)
        price = engine.price(option, volatility=0.01)  # 1% vol
        
        self.assertGreater(price, 0)


class TestImpliedVolatility(unittest.TestCase):
    """Test implied volatility calculation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.underlying = create_simple_fixed_bond(
            issue_date=datetime(2023, 1, 1),
            maturity_date=datetime(2028, 1, 1),
            notional=1000.0,
            coupon_rate=0.05
        )
        
        self.valuation_date = datetime(2024, 1, 1)
        self.rate_curve = FlatRateCurve(rate=0.04)
        
        self.pricing_env = PricingEnvironment(
            rate_curve=self.rate_curve,
            valuation_date=self.valuation_date
        )
        
        self.option = EuroShortTermBondOption(
            underlying=self.underlying,
            strike=1000.0,
            expiry_date=datetime(2025, 1, 1),
            option_type=OptionType.CALL
        )
        
        self.engine = BlackBondOptionEngine(self.pricing_env)
    
    def test_implied_vol_recovery(self):
        """Test that we can recover implied vol from market price."""
        # Price at known vol
        true_vol = 0.12
        market_price = self.engine.price(self.option, volatility=true_vol)
        
        # Calculate implied vol
        implied_vol = self.engine.implied_volatility(
            self.option,
            market_price,
            initial_guess=0.10
        )
        
        self.assertAlmostEqual(implied_vol, true_vol, delta=0.001)
    
    def test_implied_vol_different_strikes(self):
        """Test implied vol at different strikes."""
        vol = 0.10
        
        for strike in [900, 950, 1000, 1050, 1100]:
            option = EuroShortTermBondOption(
                underlying=self.underlying,
                strike=float(strike),
                expiry_date=datetime(2025, 1, 1),
                option_type=OptionType.CALL
            )
            
            price = self.engine.price(option, volatility=vol)
            implied = self.engine.implied_volatility(option, price)
            
            self.assertAlmostEqual(implied, vol, delta=0.005)


if __name__ == '__main__':
    unittest.main()

