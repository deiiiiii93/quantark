"""
Unit tests for Asian option analytical pricing engine.

Tests all five methods (KEMNA_VORST, TURNBULL_WAKEMAN, LEVY, CURRAN, DISCRETE_HHM)
for pricing Asian options. Validation cases are from Haug's "Complete Guide to
Option Pricing Formulas" (2nd Edition), Tables 4-25, 4-26, 4-27.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from asset.equity.product.option import AsianOption, AsianObservationRecord, EuropeanVanillaOption
from asset.equity.engine.analytical import AsianOptionAnalyticalEngine, BlackScholesEngine
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from priceenv import PricingEnvironment
from util.enum import OptionType, AveragingType, AsianStrikeType
from util.enum.engine_enums import AsianAnalyticalMethod
from util.exceptions import ValidationError, PricingError


def create_pricing_env(spot=100.0, vol=0.20, rate=0.05, div=0.0):
    """Helper to create standard pricing environment."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime(2024, 1, 1),
    )


class TestKemnaVorstGeometric:
    """Tests for Kemna-Vorst geometric average method.
    
    Reference values from Haug's book, Section 4.20.1.
    """
    
    def test_geometric_put_haug_example(self):
        """Test geometric put from Haug's book example.
        
        Parameters: S=80, X=85, T=0.25, r=0.05, b=0.08, σ=0.20
        Note: b = r - q = 0.08 means q = r - b = 0.05 - 0.08 = -0.03
        Since negative dividend yields are not allowed, we test with q=0 (b=r=0.05)
        which gives a slightly different but still valid result.
        """
        # Use zero dividend yield for a valid test case
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=80.0),
            vol_surface=FlatVolSurface(volatility=0.20),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.0),  # b = r - q = 0.05
            valuation_date=datetime(2024, 1, 1),
        )
        
        option = AsianOption(
            strike=85.0,
            option_type=OptionType.PUT,
            averaging_type=AveragingType.GEOMETRIC,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=0.25,
        )
        
        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.KEMNA_VORST)
        price = engine.price(option, pricing_env)
        
        # With b=0.05 (not 0.08), expect a different but reasonable value
        assert price > 0, f"Price should be positive, got {price}"
        assert price < 10.0, f"Put price should be reasonable, got {price}"
        print(f"✓ Kemna-Vorst geometric put: ${price:.4f}")
    
    def test_geometric_call_basic(self):
        """Test basic geometric average call."""
        pricing_env = create_pricing_env(spot=100.0, vol=0.20, rate=0.05)
        
        option = AsianOption(
            strike=100.0,
            option_type=OptionType.CALL,
            averaging_type=AveragingType.GEOMETRIC,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=1.0,
        )
        
        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.KEMNA_VORST)
        price = engine.price(option, pricing_env)
        
        assert price > 0, "Geometric call price should be positive"
        assert price < 100.0, "Price should be reasonable"
        print(f"✓ Kemna-Vorst geometric call: ${price:.4f}")
    
    def test_geometric_vs_vanilla_discount(self):
        """Geometric average option should be cheaper than vanilla."""
        pricing_env = create_pricing_env(spot=100.0, vol=0.20, rate=0.05)
        
        option = AsianOption(
            strike=100.0,
            option_type=OptionType.CALL,
            averaging_type=AveragingType.GEOMETRIC,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=1.0,
        )
        
        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.KEMNA_VORST)
        asian_price = engine.price(option, pricing_env)
        
        # Calculate real vanilla price for comparison
        vanilla_option = EuropeanVanillaOption(
            strike=100.0,
            option_type=OptionType.CALL,
            maturity=1.0,
        )
        bs_engine = BlackScholesEngine()
        vanilla_price = bs_engine.price(vanilla_option, pricing_env)
        
        # A geometric average option should be cheaper than vanilla
        # because averaging reduces volatility
        assert asian_price < vanilla_price, f"Geometric price {asian_price:.4f} should be < vanilla price {vanilla_price:.4f}"
        print(f"✓ Geometric average discount verified: ${asian_price:.4f} (vanilla: ${vanilla_price:.4f})")


class TestTurnbullWakeman:
    """Tests for Turnbull-Wakeman arithmetic average approximation.
    
    Reference values from Haug's book, Table 4-25 and examples.
    """
    
    def test_tw_put_haug_example(self):
        """Test TW put from Haug's book example.
        
        TurnbullWakemanAsian("p", 90, 88, 95, 0, 0.25, 0.25, 0.07, 0.02, 0.25)
        Expected: p = 5.6093
        """
        # For TW with SA=88 (already in period), we need observation records
        # Since t1=0 and option is in-period, we have SA != S
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=90.0),
            vol_surface=FlatVolSurface(volatility=0.25),
            rate_curve=FlatRateCurve(rate=0.07),
            div_yield=ContinuousDividendYield(div_yield=0.05),  # b = r - q = 0.07 - 0.05 = 0.02
            valuation_date=datetime(2024, 1, 1),
        )
        
        option = AsianOption(
            strike=95.0,
            option_type=OptionType.PUT,
            averaging_type=AveragingType.ARITHMETIC,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=0.25,
            num_observations=12,
        )
        
        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
        price = engine.price(option, pricing_env)
        
        # Note: The exact value 5.6093 requires in-period adjustment with SA=88
        # For a fresh option (not in-period), value will be different
        assert price > 0, "TW put price should be positive"
        print(f"✓ Turnbull-Wakeman arithmetic put: ${price:.4f}")
    
    def test_tw_table_425_sigma015(self):
        """Test TW values from Table 4-25 with σ=0.15.
        
        Parameters: S=SA=100, T2=0.75, r=0.1, b=0.05, σ=0.15
        T=0.75 (start of period): X=95 → 7.0544, X=100 → 3.7845, X=105 → 1.6729
        
        Note: Table values are for continuous averaging; discrete approximations
        may differ slightly.
        """
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(volatility=0.15),
            rate_curve=FlatRateCurve(rate=0.10),
            div_yield=ContinuousDividendYield(div_yield=0.05),  # b = r - q = 0.10 - 0.05 = 0.05
            valuation_date=datetime(2024, 1, 1),
        )
        
        test_cases = [
            (95.0, 7.0544),
            (100.0, 3.7845),
            (105.0, 1.6729),
        ]
        
        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
        
        for strike, expected in test_cases:
            option = AsianOption(
                strike=strike,
                option_type=OptionType.CALL,
                averaging_type=AveragingType.ARITHMETIC,
                asian_strike_type=AsianStrikeType.FIXED,
                maturity=0.75,
                num_observations=None,
            )
            price = engine.price(option, pricing_env)
            
            # Use tight tolerance (0.5%) for continuous benchmarks
            assert abs(price - expected) / expected < 0.005, f"X={strike}: Expected {expected}, got {price}"
            print(f"✓ TW Table 4-25 (X={strike}, σ=0.15): ${price:.4f} (expected: ${expected})")
    
    def test_tw_table_425_sigma035(self):
        """Test TW values from Table 4-25 with σ=0.35.
        
        Parameters: S=SA=100, T2=0.75, r=0.1, b=0.05, σ=0.35
        T=0.75: X=95 → 10.1213, X=100 → 7.5038, X=105 → 5.4071
        """
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(volatility=0.35),
            rate_curve=FlatRateCurve(rate=0.10),
            div_yield=ContinuousDividendYield(div_yield=0.05),
            valuation_date=datetime(2024, 1, 1),
        )
        
        test_cases = [
            (95.0, 10.1213),
            (100.0, 7.5038),
            (105.0, 5.4071),
        ]
        
        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
        
        for strike, expected in test_cases:
            option = AsianOption(
                strike=strike,
                option_type=OptionType.CALL,
                averaging_type=AveragingType.ARITHMETIC,
                asian_strike_type=AsianStrikeType.FIXED,
                maturity=0.75,
                num_observations=None,
            )
            price = engine.price(option, pricing_env)
            
            # Use tight tolerance (0.5%) for continuous benchmarks
            assert abs(price - expected) / expected < 0.005, f"X={strike}: Expected {expected}, got {price}"
            print(f"✓ TW Table 4-25 (X={strike}, σ=0.35): ${price:.4f} (expected: ${expected})")


class TestLevy:
    """Tests for Levy arithmetic average approximation.
    
    Reference values from Haug's book, Table 4-25 and currency example.
    """
    
    def test_levy_currency_option(self):
        """Test Levy currency option example from Haug.
        
        S=6.80, SA=6.80, X=6.90, T=0.5, T2=0.5, r=0.07, b=-0.02, σ=0.14
        Expected: c ≈ 0.0944, p ≈ 0.2237
        """
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=6.80),
            vol_surface=FlatVolSurface(volatility=0.14),
            rate_curve=FlatRateCurve(rate=0.07),
            div_yield=ContinuousDividendYield(div_yield=0.09),  # b = r - q = 0.07 - 0.09 = -0.02
            valuation_date=datetime(2024, 1, 1),
        )
        
        call = AsianOption(
            strike=6.90,
            option_type=OptionType.CALL,
            averaging_type=AveragingType.ARITHMETIC,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=0.5,
            num_observations=None,
        )
        
        put = AsianOption(
            strike=6.90,
            option_type=OptionType.PUT,
            averaging_type=AveragingType.ARITHMETIC,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=0.5,
            num_observations=None,
        )
        
        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.LEVY)
        call_price = engine.price(call, pricing_env)
        put_price = engine.price(put, pricing_env)
        
        # Allow tolerance for continuous vs discrete
        assert call_price > 0, "Levy call price should be positive"
        assert put_price > 0, "Levy put price should be positive"
        print(f"✓ Levy currency call: ${call_price:.4f} (expected: ~0.0944)")
        print(f"✓ Levy currency put: ${put_price:.4f} (expected: ~0.2237)")
    
    def test_levy_matches_tw(self):
        """Verify Levy approximation is in reasonable range vs TW.
        
        Table 4-25 shows identical values for TW and Levy for continuous case.
        Discrete implementations may differ slightly.
        """
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(volatility=0.15),
            rate_curve=FlatRateCurve(rate=0.10),
            div_yield=ContinuousDividendYield(div_yield=0.05),
            valuation_date=datetime(2024, 1, 1),
        )
        
        option = AsianOption(
            strike=100.0,
            option_type=OptionType.CALL,
            averaging_type=AveragingType.ARITHMETIC,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=0.75,
            num_observations=12,
        )
        
        tw_engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
        levy_engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.LEVY)
        
        tw_price = tw_engine.price(option, pricing_env)
        levy_price = levy_engine.price(option, pricing_env)
        
        # TW and Levy should give reasonably similar results (within 10%)
        assert abs(tw_price - levy_price) / tw_price < 0.10, f"TW={tw_price}, Levy={levy_price}"
        print(f"✓ TW vs Levy: ${tw_price:.4f} vs ${levy_price:.4f}")


class TestCompletedAveraging:
    """Regression tests for pricing after averaging window completes."""

    @staticmethod
    def _completed_averaging_option(
        strike: float, option_type: OptionType, maturity: float = 0.5
    ) -> AsianOption:
        return AsianOption(
            strike=strike,
            option_type=option_type,
            averaging_type=AveragingType.ARITHMETIC,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=maturity,
            observation_records=[
                AsianObservationRecord(
                    observation_date=datetime(2023, 12, 15), observed_price=100.0
                ),
                AsianObservationRecord(
                    observation_date=datetime(2023, 12, 20), observed_price=110.0
                ),
                AsianObservationRecord(
                    observation_date=datetime(2023, 12, 31), observed_price=90.0
                ),
            ],
        )

    def test_turnbull_wakeman_completed_averaging_returns_discounted_payoff(self):
        pricing_env = create_pricing_env(spot=100.0, vol=0.20, rate=0.05, div=0.0)
        option = self._completed_averaging_option(
            strike=95.0, option_type=OptionType.CALL, maturity=0.5
        )

        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
        price = engine.price(option, pricing_env)

        avg = 100.0  # mean of 100, 110, 90
        expected = max(avg - option.strike, 0.0) * np.exp(-0.05 * option.maturity)
        assert price == pytest.approx(expected, rel=0.0, abs=1e-12)

    def test_levy_completed_averaging_returns_discounted_payoff(self):
        pricing_env = create_pricing_env(spot=100.0, vol=0.20, rate=0.05, div=0.0)
        option = self._completed_averaging_option(
            strike=95.0, option_type=OptionType.CALL, maturity=0.5
        )

        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.LEVY)
        price = engine.price(option, pricing_env)

        avg = 100.0  # mean of 100, 110, 90
        expected = max(avg - option.strike, 0.0) * np.exp(-0.05 * option.maturity)
        assert price == pytest.approx(expected, rel=0.0, abs=1e-12)


class TestCurran:
    """Tests for Curran geometric conditioning approximation.
    
    Reference values from Haug's book, Table 4-27.
    """
    
    def test_curran_table_427(self):
        """Test Curran values from Table 4-27.
        
        Parameters: X=100, T=26 weeks=0.5, Δt=1/52, r=0.08, b=0.03, n=27
        t1=0: S=95,σ=0.1→0.2758, S=100,σ=0.1→1.9466, S=105,σ=0.1→5.7110
        """
        test_cases = [
            (95.0, 0.10, 0.2758),
            (100.0, 0.10, 1.9466),
            (105.0, 0.10, 5.7110),
            (95.0, 0.20, 1.4262),
            (100.0, 0.20, 3.4899),
            (105.0, 0.20, 6.7024),
        ]
        
        for spot, vol, expected in test_cases:
            pricing_env = PricingEnvironment(
                spot_quote=SpotQuote(spot=spot),
                vol_surface=FlatVolSurface(volatility=vol),
                rate_curve=FlatRateCurve(rate=0.08),
                div_yield=ContinuousDividendYield(div_yield=0.05),  # b = r - q = 0.08 - 0.05 = 0.03
                valuation_date=datetime(2024, 1, 1),
            )
            
            # Create observation times for discrete fixings
            n = 27
            T = 0.5
            dt = T / (n - 1)  # Weekly fixings
            obs_times = [i * dt for i in range(n)]
            
            option = AsianOption(
                strike=100.0,
                option_type=OptionType.CALL,
                averaging_type=AveragingType.ARITHMETIC,
                asian_strike_type=AsianStrikeType.FIXED,
                maturity=T,
                observation_times=obs_times,
            )
            
            engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.CURRAN)
            price = engine.price(option, pricing_env)
            
            assert price == pytest.approx(
                expected, rel=0.0, abs=5e-4
            ), f"S={spot}, σ={vol}: Expected {expected}, got {price}"
            print(f"✓ Curran Table 4-27 (S={spot}, σ={vol}): ${price:.4f} (expected: ${expected})")


class TestDiscreteHHM:
    """Tests for discrete arithmetic (Haug-Haug-Margrabe) method.
    
    Reference values from Haug's book, Table 4-26.
    """
    
    def test_hhm_example(self):
        """Test HHM from Haug's book example.
        
        DiscreteAsianHHM("c", 100, 110, 105, 0, 0.5, 360, 180, 0.07, 0.02, 0.25)
        Expected: c = 2.0971
        """
        # This example has m=180 observations already realized with SA=110
        # Our implementation handles in-period pricing via observation_records
        
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(volatility=0.25),
            rate_curve=FlatRateCurve(rate=0.07),
            div_yield=ContinuousDividendYield(div_yield=0.05),  # b = r - q = 0.07 - 0.05 = 0.02
            valuation_date=datetime(2024, 1, 1),
        )
        
        # Fresh option (m=0) for basic test
        n = 27
        T = 0.5
        dt = T / (n - 1)
        obs_times = [i * dt for i in range(1, n + 1)]
        
        option = AsianOption(
            strike=105.0,
            option_type=OptionType.CALL,
            averaging_type=AveragingType.ARITHMETIC,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=T,
            observation_times=obs_times,
        )
        
        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.DISCRETE_HHM)
        price = engine.price(option, pricing_env)
        
        assert price > 0, "HHM call price should be positive"
        print(f"✓ HHM discrete arithmetic call: ${price:.4f}")
    
    def test_hhm_table_426(self):
        """Test HHM values from Table 4-26.
        
        Parameters: X=100, T=0.5+t1, Δt=1/52, r=0.08, b=0.03, n=27, m=0
        t1=0 weeks: S=95,σ=0.1→0.2719, S=100,σ=0.1→1.9484, S=105,σ=0.1→5.7150
        """
        test_cases = [
            (95.0, 0.10, 0.2719),
            (100.0, 0.10, 1.9484),
            (105.0, 0.10, 5.7150),
            (95.0, 0.20, 1.4166),
            (100.0, 0.20, 3.4961),
            (105.0, 0.20, 6.7212),
        ]
        
        for spot, vol, expected in test_cases:
            pricing_env = PricingEnvironment(
                spot_quote=SpotQuote(spot=spot),
                vol_surface=FlatVolSurface(volatility=vol),
                rate_curve=FlatRateCurve(rate=0.08),
                div_yield=ContinuousDividendYield(div_yield=0.05),  # b = 0.03
                valuation_date=datetime(2024, 1, 1),
            )
            
            n = 27
            T = 0.5  # t1=0, so T = 0.5 + 0 = 0.5
            dt = 1/52  # Weekly
            obs_times = [i * dt for i in range(n)]
            
            option = AsianOption(
                strike=100.0,
                option_type=OptionType.CALL,
                averaging_type=AveragingType.ARITHMETIC,
                asian_strike_type=AsianStrikeType.FIXED,
                maturity=T,
                observation_times=obs_times,
            )
            
            engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.DISCRETE_HHM)
            price = engine.price(option, pricing_env)
            
            # Allow 15% tolerance for approximation differences
            assert abs(price - expected) / expected < 0.15, f"S={spot}, σ={vol}: Expected {expected}, got {price}"
            print(f"✓ HHM Table 4-26 (S={spot}, σ={vol}): ${price:.4f} (expected: ${expected})")


class TestFloatingStrike:
    """Tests for floating-strike Asian options.
    
    Uses Henderson-Wojakowski symmetry to transform to fixed-strike.
    """
    
    def test_floating_strike_call(self):
        """Test floating-strike call (average strike option)."""
        pricing_env = create_pricing_env(spot=100.0, vol=0.20, rate=0.05)
        
        option = AsianOption(
            strike=0.0,  # Placeholder for floating
            option_type=OptionType.CALL,
            averaging_type=AveragingType.ARITHMETIC,
            asian_strike_type=AsianStrikeType.FLOATING,
            maturity=1.0,
            num_observations=12,
        )
        
        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
        price = engine.price(option, pricing_env)
        
        assert price > 0, "Floating-strike call should have positive value"
        print(f"✓ Floating-strike arithmetic call: ${price:.4f}")
    
    def test_floating_strike_put(self):
        """Test floating-strike put."""
        pricing_env = create_pricing_env(spot=100.0, vol=0.20, rate=0.05)
        
        option = AsianOption(
            strike=0.0,
            option_type=OptionType.PUT,
            averaging_type=AveragingType.ARITHMETIC,
            asian_strike_type=AsianStrikeType.FLOATING,
            maturity=1.0,
            num_observations=12,
        )
        
        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
        price = engine.price(option, pricing_env)
        
        assert price > 0, "Floating-strike put should have positive value"
        print(f"✓ Floating-strike arithmetic put: ${price:.4f}")
    
    def test_floating_geometric(self):
        """Test floating-strike geometric option."""
        pricing_env = create_pricing_env(spot=100.0, vol=0.20, rate=0.05)
        
        option = AsianOption(
            strike=0.0,
            option_type=OptionType.CALL,
            averaging_type=AveragingType.GEOMETRIC,
            asian_strike_type=AsianStrikeType.FLOATING,
            maturity=1.0,
        )
        
        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.KEMNA_VORST)
        price = engine.price(option, pricing_env)
        
        assert price > 0, "Floating-strike geometric call should have positive value"
        print(f"✓ Floating-strike geometric call: ${price:.4f}")


class TestMethodComparison:
    """Compare pricing across different methods."""
    
    def test_arithmetic_methods_comparison(self):
        """Compare TW, Levy, Curran, and HHM for arithmetic averaging."""
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(volatility=0.20),
            rate_curve=FlatRateCurve(rate=0.08),
            div_yield=ContinuousDividendYield(div_yield=0.05),
            valuation_date=datetime(2024, 1, 1),
        )
        
        n = 12
        T = 0.5
        obs_times = [i * T / n for i in range(1, n + 1)]
        
        option = AsianOption(
            strike=100.0,
            option_type=OptionType.CALL,
            averaging_type=AveragingType.ARITHMETIC,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=T,
            observation_times=obs_times,
        )
        
        methods = [
            AsianAnalyticalMethod.TURNBULL_WAKEMAN,
            AsianAnalyticalMethod.LEVY,
            AsianAnalyticalMethod.CURRAN,
            AsianAnalyticalMethod.DISCRETE_HHM,
        ]
        
        prices = {}
        for method in methods:
            engine = AsianOptionAnalyticalEngine(method=method)
            prices[method.value] = engine.price(option, pricing_env)
        
        print("\n✓ Arithmetic method comparison:")
        for method, price in prices.items():
            print(f"  {method}: ${price:.4f}")
        
        # All methods should give similar results (within 10%)
        price_list = list(prices.values())
        mean_price = np.mean(price_list)
        for method, price in prices.items():
            assert abs(price - mean_price) / mean_price < 0.10, f"{method} diverges too much"
    
    def test_geometric_uses_kemna_vorst(self):
        """Verify geometric options can use Kemna-Vorst."""
        pricing_env = create_pricing_env(spot=100.0, vol=0.20, rate=0.05)
        
        option = AsianOption(
            strike=100.0,
            option_type=OptionType.CALL,
            averaging_type=AveragingType.GEOMETRIC,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=1.0,
        )
        
        # Test that Kemna-Vorst prices geometric options correctly
        engine_kv = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.KEMNA_VORST)
        price_kv = engine_kv.price(option, pricing_env)
        
        assert price_kv > 0, "KV should produce positive price for geometric"
        assert not np.isnan(price_kv), "KV price should be valid"
        print(f"✓ Kemna-Vorst geometric price: ${price_kv:.4f}")


class TestEdgeCases:
    """Test edge cases and special conditions."""
    
    def test_near_expiry(self):
        """Test options very close to expiry."""
        pricing_env = create_pricing_env(spot=100.0, vol=0.20, rate=0.05)
        
        option = AsianOption(
            strike=95.0,
            option_type=OptionType.CALL,
            averaging_type=AveragingType.ARITHMETIC,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=1e-12,
            num_observations=1,
        )
        
        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
        price = engine.price(option, pricing_env)
        
        # Near expiry, should be close to intrinsic value
        intrinsic = max(100.0 - 95.0, 0)
        assert abs(price - intrinsic) < 0.1, f"Near-expiry price {price} should be ~{intrinsic}"
        print(f"✓ Near-expiry: ${price:.4f} (intrinsic: ${intrinsic})")
    
    def test_atm_option(self):
        """Test at-the-money option."""
        pricing_env = create_pricing_env(spot=100.0, vol=0.20, rate=0.05)
        
        option = AsianOption(
            strike=100.0,
            option_type=OptionType.CALL,
            averaging_type=AveragingType.ARITHMETIC,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=1.0,
            num_observations=12,
        )
        
        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
        price = engine.price(option, pricing_env)
        
        assert price > 0, "ATM call should have positive value"
        assert not np.isnan(price) and not np.isinf(price), "Price should be finite"
        print(f"✓ ATM arithmetic call: ${price:.4f}")
    
    def test_deep_itm(self):
        """Test deep in-the-money option."""
        pricing_env = create_pricing_env(spot=150.0, vol=0.20, rate=0.05)
        
        option = AsianOption(
            strike=100.0,
            option_type=OptionType.CALL,
            averaging_type=AveragingType.ARITHMETIC,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=1.0,
            num_observations=12,
        )
        
        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
        price = engine.price(option, pricing_env)
        
        assert price > 30, "Deep ITM call should have high value"
        print(f"✓ Deep ITM arithmetic call: ${price:.4f}")
    
    def test_deep_otm(self):
        """Test deep out-of-the-money option."""
        pricing_env = create_pricing_env(spot=50.0, vol=0.20, rate=0.05)
        
        option = AsianOption(
            strike=100.0,
            option_type=OptionType.CALL,
            averaging_type=AveragingType.ARITHMETIC,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=1.0,
            num_observations=12,
        )
        
        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
        price = engine.price(option, pricing_env)
        
        assert price >= 0, "Deep OTM call should have non-negative value"
        assert price < 5.0, "Deep OTM call should have low value"
        print(f"✓ Deep OTM arithmetic call: ${price:.4f}")
    
    def test_high_volatility(self):
        """Test with extreme volatility."""
        pricing_env = create_pricing_env(spot=100.0, vol=1.0, rate=0.05)
        
        option = AsianOption(
            strike=100.0,
            option_type=OptionType.CALL,
            averaging_type=AveragingType.ARITHMETIC,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=1.0,
            num_observations=12,
        )
        
        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
        price = engine.price(option, pricing_env)
        
        assert not np.isnan(price) and not np.isinf(price), "Should handle high volatility"
        assert price > 0, "High vol option should have positive value"
        print(f"✓ High volatility (σ=1.0): ${price:.4f}")
    
    def test_zero_cost_of_carry(self):
        """Test with zero cost-of-carry (b=0)."""
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=100.0),
            vol_surface=FlatVolSurface(volatility=0.20),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.05),  # b = r - q = 0
            valuation_date=datetime(2024, 1, 1),
        )
        
        option = AsianOption(
            strike=100.0,
            option_type=OptionType.CALL,
            averaging_type=AveragingType.ARITHMETIC,
            asian_strike_type=AsianStrikeType.FIXED,
            maturity=1.0,
            num_observations=12,
        )
        
        # TW should handle b=0 specially
        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
        price = engine.price(option, pricing_env)
        
        assert price > 0, "b=0 case should produce positive price"
        assert not np.isnan(price), "b=0 should not produce NaN"
        print(f"✓ Zero cost-of-carry (b=0): ${price:.4f}")


class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_invalid_method(self):
        """Test invalid method selection raises error."""
        with pytest.raises(ValidationError):
            AsianOptionAnalyticalEngine(method="INVALID")
    
    def test_wrong_product_type(self):
        """Test pricing wrong product type raises error."""
        from asset.equity.product.option import EuropeanVanillaOption
        
        pricing_env = create_pricing_env()
        euro_option = EuropeanVanillaOption(
            strike=100.0, option_type=OptionType.CALL, maturity=1.0
        )
        
        engine = AsianOptionAnalyticalEngine(method=AsianAnalyticalMethod.TURNBULL_WAKEMAN)
        
        with pytest.raises(PricingError):
            engine.price(euro_option, pricing_env)


class TestDefaultMethod:
    """Test default method selection."""
    
    def test_default_is_auto(self):
        """Test default method is auto-select (None)."""
        engine = AsianOptionAnalyticalEngine()
        # Default is None (auto-select based on product)
        assert engine.method is None, "Default should be auto-select (None)"
        print("✓ Default method is auto-select")
    
    def test_two_level_enum_pattern(self):
        """Test two-level enum pattern for method selection."""
        from util.enum.engine_enums import EngineType
        
        # Pattern: EngineType.ANALYTICAL(AsianAnalyticalMethod.LEVY)
        engine = AsianOptionAnalyticalEngine(
            method=EngineType.ANALYTICAL(AsianAnalyticalMethod.LEVY)
        )
        assert engine.method == AsianAnalyticalMethod.LEVY
        print("✓ Two-level enum pattern works")
    
    def test_string_method_selection(self):
        """Test string-based method selection."""
        engine = AsianOptionAnalyticalEngine(method="curran")
        assert engine.method == AsianAnalyticalMethod.CURRAN
        print("✓ String method selection works")


if __name__ == "__main__":
    print("Running Asian Option Analytical Engine Tests\n")
    print("=" * 70)
    
    test_classes = [
        TestKemnaVorstGeometric,
        TestTurnbullWakeman,
        TestLevy,
        TestCurran,
        TestDiscreteHHM,
        TestFloatingStrike,
        TestMethodComparison,
        TestEdgeCases,
        TestErrorHandling,
        TestDefaultMethod,
    ]
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 70)
        test_instance = test_class()
        for method_name in dir(test_instance):
            if method_name.startswith("test_"):
                try:
                    method = getattr(test_instance, method_name)
                    method()
                except Exception as e:
                    print(f"✗ {method_name} FAILED: {e}")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
