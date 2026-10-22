"""
Comparison of American option pricing: Analytical vs PDE methods.

This script demonstrates:
1. Pricing consistency between analytical (BS93, BS02, BAW) and PDE methods
2. Accuracy comparison across different moneyness and volatility scenarios
3. Performance comparison between methods
"""

import sys
from pathlib import Path
from datetime import datetime
import time


from quantark.asset.equity.product.option import AmericanOption
from quantark.asset.equity.engine.analytical import AmericanOptionAnalyticalEngine
from quantark.asset.equity.engine.pde import AmericanPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import AmericanAnalyticalMethod
from quantark.asset.equity.engine.pde import GridConfig


def print_section(title: str):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def compare_american_call():
    print_section("American Call: Analytical vs PDE")
    
    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.25)
    rate = FlatRateCurve(rate=0.05)
    div = ContinuousDividendYield(div_yield=0.03)
    
    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )
    
    print("\nMarket Data:")
    print(f"Spot Price (S):        ${pricing_env.spot:.2f}")
    print(f"Volatility (σ):        {pricing_env.get_vol(100, 1.0):.2%}")
    print(f"Risk-Free Rate (r):    {pricing_env.get_rate(1.0):.2%}")
    print(f"Dividend Yield (q):    {pricing_env.get_div_yield(1.0):.2%}")
    
    print("\nOption Specification:")
    call = AmericanOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    print(f"Type:                  American Call")
    print(f"Strike (K):            ${call.strike:.2f}")
    print(f"Time to Maturity (T):  {call.maturity:.2f} years")
    
    print("\n" + "-" * 80)
    print(f"{'Method':<30} {'Price':>12} {'Time (ms)':>12}")
    print("-" * 80)
    
    t0 = time.time()
    engine_bs93 = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93)
    price_bs93 = engine_bs93.price(call, pricing_env)
    time_bs93 = (time.time() - t0) * 1000
    print(f"{'Analytical (BS93)':<30} ${price_bs93:>10.6f} {time_bs93:>11.3f}")
    
    t0 = time.time()
    engine_bs02 = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS02)
    price_bs02 = engine_bs02.price(call, pricing_env)
    time_bs02 = (time.time() - t0) * 1000
    print(f"{'Analytical (BS02)':<30} ${price_bs02:>10.6f} {time_bs02:>11.3f}")
    
    t0 = time.time()
    engine_baw = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BAW)
    price_baw = engine_baw.price(call, pricing_env)
    time_baw = (time.time() - t0) * 1000
    print(f"{'Analytical (BAW)':<30} ${price_baw:>10.6f} {time_baw:>11.3f}")
    
    t0 = time.time()
    pde_params = PDEParams(grid=GridConfig(points=500))
    pde_solver = AmericanPDESolver(params=pde_params)
    price_pde = pde_solver.price(call, pricing_env)
    time_pde = (time.time() - t0) * 1000
    print(f"{'PDE (500x500 grid)':<30} ${price_pde:>10.6f} {time_pde:>11.3f}")
    
    print("\nDifference from PDE (reference):")
    print(f"BS93 - PDE:            ${price_bs93 - price_pde:>10.6f} ({abs((price_bs93 - price_pde) / price_pde * 100):.4f}%)")
    print(f"BS02 - PDE:            ${price_bs02 - price_pde:>10.6f} ({abs((price_bs02 - price_pde) / price_pde * 100):.4f}%)")
    print(f"BAW  - PDE:            ${price_baw - price_pde:>10.6f} ({abs((price_baw - price_pde) / price_pde * 100):.4f}%)")


def compare_american_put():
    print_section("American Put: Analytical vs PDE")
    
    spot = SpotQuote(spot=100.0)
    vol = FlatVolSurface(volatility=0.25)
    rate = FlatRateCurve(rate=0.05)
    div = ContinuousDividendYield(div_yield=0.03)
    
    pricing_env = PricingEnvironment(
        spot_quote=spot,
        vol_surface=vol,
        rate_curve=rate,
        div_yield=div,
        valuation_date=datetime(2024, 1, 1),
    )
    
    print("\nMarket Data:")
    print(f"Spot Price (S):        ${pricing_env.spot:.2f}")
    print(f"Volatility (σ):        {pricing_env.get_vol(100, 1.0):.2%}")
    print(f"Risk-Free Rate (r):    {pricing_env.get_rate(1.0):.2%}")
    print(f"Dividend Yield (q):    {pricing_env.get_div_yield(1.0):.2%}")
    
    print("\nOption Specification:")
    put = AmericanOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0)
    print(f"Type:                  American Put")
    print(f"Strike (K):            ${put.strike:.2f}")
    print(f"Time to Maturity (T):  {put.maturity:.2f} years")
    
    print("\n" + "-" * 80)
    print(f"{'Method':<30} {'Price':>12} {'Time (ms)':>12}")
    print("-" * 80)
    
    t0 = time.time()
    engine_bs93 = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93)
    price_bs93 = engine_bs93.price(put, pricing_env)
    time_bs93 = (time.time() - t0) * 1000
    print(f"{'Analytical (BS93)':<30} ${price_bs93:>10.6f} {time_bs93:>11.3f}")
    
    t0 = time.time()
    engine_bs02 = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS02)
    price_bs02 = engine_bs02.price(put, pricing_env)
    time_bs02 = (time.time() - t0) * 1000
    print(f"{'Analytical (BS02)':<30} ${price_bs02:>10.6f} {time_bs02:>11.3f}")
    
    t0 = time.time()
    engine_baw = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BAW)
    price_baw = engine_baw.price(put, pricing_env)
    time_baw = (time.time() - t0) * 1000
    print(f"{'Analytical (BAW)':<30} ${price_baw:>10.6f} {time_baw:>11.3f}")
    
    t0 = time.time()
    pde_params = PDEParams(grid=GridConfig(points=500))
    pde_solver = AmericanPDESolver(params=pde_params)
    price_pde = pde_solver.price(put, pricing_env)
    time_pde = (time.time() - t0) * 1000
    print(f"{'PDE (500x500 grid)':<30} ${price_pde:>10.6f} {time_pde:>11.3f}")
    
    print("\nDifference from PDE (reference):")
    print(f"BS93 - PDE:            ${price_bs93 - price_pde:>10.6f} ({abs((price_bs93 - price_pde) / price_pde * 100):.4f}%)")
    print(f"BS02 - PDE:            ${price_bs02 - price_pde:>10.6f} ({abs((price_bs02 - price_pde) / price_pde * 100):.4f}%)")
    print(f"BAW  - PDE:            ${price_baw - price_pde:>10.6f} ({abs((price_baw - price_pde) / price_pde * 100):.4f}%)")


def compare_across_scenarios():
    print_section("Consistency Across Different Scenarios")
    
    scenarios = [
        {"name": "ATM, normal vol", "spot": 100.0, "strike": 100.0, "vol": 0.20, "maturity": 1.0},
        {"name": "ITM Call", "spot": 110.0, "strike": 100.0, "vol": 0.20, "maturity": 1.0},
        {"name": "OTM Call", "spot": 90.0, "strike": 100.0, "vol": 0.20, "maturity": 1.0},
        {"name": "High volatility", "spot": 100.0, "strike": 100.0, "vol": 0.50, "maturity": 1.0},
        {"name": "Low volatility", "spot": 100.0, "strike": 100.0, "vol": 0.10, "maturity": 1.0},
        {"name": "Short maturity", "spot": 100.0, "strike": 100.0, "vol": 0.25, "maturity": 0.25},
        {"name": "Long maturity", "spot": 100.0, "strike": 100.0, "vol": 0.25, "maturity": 2.0},
    ]
    
    print("\nAmerican Call Pricing:")
    print("-" * 100)
    print(f"{'Scenario':<20} {'BS93':>12} {'BS02':>12} {'BAW':>12} {'PDE':>12} {'Max|Diff|':>12}")
    print("-" * 100)
    
    for scenario in scenarios:
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=scenario["spot"]),
            vol_surface=FlatVolSurface(volatility=scenario["vol"]),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.02),
            valuation_date=datetime(2024, 1, 1),
        )
        
        call = AmericanOption(strike=scenario["strike"], option_type=OptionType.CALL, maturity=scenario["maturity"])
        
        p_bs93 = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93).price(call, pricing_env)
        p_bs02 = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS02).price(call, pricing_env)
        p_baw = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BAW).price(call, pricing_env)
        
        pde_params = PDEParams(grid=GridConfig(points=500))
        p_pde = AmericanPDESolver(params=pde_params).price(call, pricing_env)
        
        max_diff = max(abs(p_bs93 - p_pde), abs(p_bs02 - p_pde), abs(p_baw - p_pde))
        
        print(f"{scenario['name']:<20} ${p_bs93:>10.4f} ${p_bs02:>10.4f} ${p_baw:>10.4f} ${p_pde:>10.4f} ${max_diff:>10.4f}")
    
    print("\nAmerican Put Pricing:")
    print("-" * 100)
    print(f"{'Scenario':<20} {'BS93':>12} {'BS02':>12} {'BAW':>12} {'PDE':>12} {'Max|Diff|':>12}")
    print("-" * 100)
    
    for scenario in scenarios:
        pricing_env = PricingEnvironment(
            spot_quote=SpotQuote(spot=scenario["spot"]),
            vol_surface=FlatVolSurface(volatility=scenario["vol"]),
            rate_curve=FlatRateCurve(rate=0.05),
            div_yield=ContinuousDividendYield(div_yield=0.02),
            valuation_date=datetime(2024, 1, 1),
        )
        
        put = AmericanOption(strike=scenario["strike"], option_type=OptionType.PUT, maturity=scenario["maturity"])
        
        p_bs93 = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93).price(put, pricing_env)
        p_bs02 = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS02).price(put, pricing_env)
        p_baw = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BAW).price(put, pricing_env)
        
        pde_params = PDEParams(grid=GridConfig(points=500))
        p_pde = AmericanPDESolver(params=pde_params).price(put, pricing_env)
        
        max_diff = max(abs(p_bs93 - p_pde), abs(p_bs02 - p_pde), abs(p_baw - p_pde))
        
        print(f"{scenario['name']:<20} ${p_bs93:>10.4f} ${p_bs02:>10.4f} ${p_baw:>10.4f} ${p_pde:>10.4f} ${max_diff:>10.4f}")


def pde_grid_convergence():
    print_section("PDE Grid Convergence Analysis")
    
    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.25),
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.03),
        valuation_date=datetime(2024, 1, 1),
    )
    
    call = AmericanOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    
    price_bs93 = AmericanOptionAnalyticalEngine(method=AmericanAnalyticalMethod.BS93).price(call, pricing_env)
    
    print("\nAmerican Call (K=100, S=100, T=1.0, σ=25%, r=5%, q=3%)")
    print(f"BS93 Reference Price: ${price_bs93:.6f}")
    print("\n" + "-" * 80)
    print(f"{'Grid Size':<20} {'PDE Price':>15} {'Diff from BS93':>15} {'Time (ms)':>15}")
    print("-" * 80)
    
    grids = [50, 100, 200, 500, 1000]
    
    for grid_size in grids:
        t0 = time.time()
        pde_params = PDEParams(grid=GridConfig(points=grid_size))
        pde_solver = AmericanPDESolver(params=pde_params)
        price_pde = pde_solver.price(call, pricing_env)
        elapsed_ms = (time.time() - t0) * 1000
        
        diff = price_pde - price_bs93
        
        print(f"{grid_size:>4}x{grid_size:<12} ${price_pde:>13.6f} ${diff:>13.6f} {elapsed_ms:>14.2f}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  American Option Pricing: Analytical vs PDE Comparison")
    print("=" * 80)
    
    compare_american_call()
    compare_american_put()
    compare_across_scenarios()
    pde_grid_convergence()
    
    print("\n" + "=" * 80)
    print("  Comparison Complete")
    print("=" * 80)
    print("\nKey Findings:")
    print("1. Analytical methods (BS93, BS02, BAW) are significantly faster than PDE")
    print("2. PDE provides accurate reference prices with sufficient grid resolution")
    print("3. All methods converge to similar prices for standard scenarios")
    print("4. BS02 generally has better accuracy than BS93 for longer maturities")
    print("5. PDE accuracy improves with finer grids but at computational cost")
    print()