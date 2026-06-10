"""
PDE Pricing Engine Demo

This script demonstrates the PDE pricing engine for various equity derivatives:
1. European options - compare with Black-Scholes analytical solution
2. American options - demonstrate early exercise premium
3. Barrier options - up-and-out call, down-and-in put
4. Double barrier options - knock-out corridor option
5. One-touch options - digital barrier option
6. Double one-touch options - range digital

Usage:
    python example/pde_pricing_demo.py
"""

import numpy as np
from datetime import datetime
from pathlib import Path
import sys


# Import products
from quantark.asset.equity.product.option import (
    EuropeanVanillaOption,
    AmericanOption,
    BarrierOption,
    DoubleBarrierOption,
    OneTouchOption,
    DoubleOneTouchOption,
)

# Import engines
from quantark.asset.equity.engine import (
    BlackScholesEngine,
    EuropeanPDESolver,
    AmericanPDESolver,
    BarrierPDESolver,
    DoubleBarrierPDESolver,
    OneTouchPDESolver,
    DoubleOneTouchPDESolver,
)

# Import parameters and market data
from quantark.asset.equity.param import PDEParams
from quantark.priceenv import PricingEnvironment
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield

# Import enums
from quantark.util.enum import (
    OptionType,
    BarrierType,
    DoubleBarrierType,
    BarrierDirection,
    TouchType,
    ObservationType,
)


def create_pricing_environment(spot=100.0, vol=0.20, rate=0.05, div=0.02):
    """Create a pricing environment with given market parameters."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot),
        vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate),
        div_yield=ContinuousDividendYield(div_yield=div),
        valuation_date=datetime.now(),
    )


def demo_european_option():
    """Demo: European option - PDE vs Black-Scholes."""
    print("\n" + "=" * 60)
    print("1. EUROPEAN OPTION - PDE vs Black-Scholes")
    print("=" * 60)

    # Market parameters
    spot = 100.0
    strike = 100.0
    maturity = 1.0
    vol = 0.20
    rate = 0.05
    div = 0.02

    pricing_env = create_pricing_environment(spot, vol, rate, div)

    # Create options
    call = EuropeanVanillaOption(
        strike=strike, option_type=OptionType.CALL, maturity=maturity
    )
    put = EuropeanVanillaOption(
        strike=strike, option_type=OptionType.PUT, maturity=maturity
    )

    # Price with analytical engine
    bs_engine = BlackScholesEngine()

    # Price with PDE engine (different resolutions)
    for grid_size in [100, 200, 400]:
        pde_params = PDEParams(grid_size=grid_size, time_steps=grid_size // 2)
        pde_engine = EuropeanPDESolver(pde_params)

        bs_call = bs_engine.price(call, pricing_env)
        bs_put = bs_engine.price(put, pricing_env)

        pde_call = pde_engine.price(call, pricing_env)
        pde_put = pde_engine.price(put, pricing_env)

        print(f"\nGrid size: {grid_size}")
        print(
            f"  Call: BS = {bs_call:.6f}, PDE = {pde_call:.6f}, Error = {abs(pde_call - bs_call):.6f}"
        )
        print(
            f"  Put:  BS = {bs_put:.6f}, PDE = {pde_put:.6f}, Error = {abs(pde_put - bs_put):.6f}"
        )

    # Greeks comparison
    print("\nGreeks comparison (grid_size=400):")
    pde_params = PDEParams(grid_size=400, time_steps=200)
    pde_engine = EuropeanPDESolver(pde_params)

    pde_greeks = pde_engine.calculate_greeks(call, pricing_env)
    bs_greeks = bs_engine.calculate_greeks(call, pricing_env)

    print(f"  Delta: BS = {bs_greeks['delta']:.6f}, PDE = {pde_greeks['delta']:.6f}")
    print(f"  Gamma: BS = {bs_greeks['gamma']:.6f}, PDE = {pde_greeks['gamma']:.6f}")


def demo_american_option():
    """Demo: American option - Early exercise premium."""
    print("\n" + "=" * 60)
    print("2. AMERICAN OPTION - Early Exercise Premium")
    print("=" * 60)

    spot = 100.0
    strike = 100.0
    maturity = 1.0
    vol = 0.20
    rate = 0.05
    div = 0.0  # No dividends for clear comparison

    pricing_env = create_pricing_environment(spot, vol, rate, div)

    # Create options
    euro_put = EuropeanVanillaOption(
        strike=strike, option_type=OptionType.PUT, maturity=maturity
    )
    amer_put = AmericanOption(
        strike=strike, option_type=OptionType.PUT, maturity=maturity
    )

    # Price
    pde_params = PDEParams(grid_size=400, time_steps=200)
    euro_solver = EuropeanPDESolver(pde_params)
    amer_solver = AmericanPDESolver(pde_params)

    euro_price = euro_solver.price(euro_put, pricing_env)
    amer_price = amer_solver.price(amer_put, pricing_env)

    print(f"\nATM Put (S=K={strike}, T={maturity}y, vol={vol*100}%, r={rate*100}%):")
    print(f"  European put: {euro_price:.6f}")
    print(f"  American put: {amer_price:.6f}")
    print(f"  Early exercise premium: {amer_price - euro_price:.6f}")

    # ITM put (where early exercise is more valuable)
    strike_itm = 110.0
    euro_put_itm = EuropeanVanillaOption(
        strike=strike_itm, option_type=OptionType.PUT, maturity=maturity
    )
    amer_put_itm = AmericanOption(
        strike=strike_itm, option_type=OptionType.PUT, maturity=maturity
    )

    euro_price_itm = euro_solver.price(euro_put_itm, pricing_env)
    amer_price_itm = amer_solver.price(amer_put_itm, pricing_env)

    print(f"\nITM Put (S={spot}, K={strike_itm}):")
    print(f"  European put: {euro_price_itm:.6f}")
    print(f"  American put: {amer_price_itm:.6f}")
    print(f"  Early exercise premium: {amer_price_itm - euro_price_itm:.6f}")


def demo_barrier_option():
    """Demo: Barrier options."""
    print("\n" + "=" * 60)
    print("3. BARRIER OPTIONS")
    print("=" * 60)

    spot = 100.0
    strike = 100.0
    maturity = 0.5
    vol = 0.25
    rate = 0.05
    div = 0.0

    pricing_env = create_pricing_environment(spot, vol, rate, div)
    pde_params = PDEParams(grid_size=400, time_steps=200)

    # Up-and-out call
    barrier = 120.0
    up_out_call = BarrierOption(
        strike=strike,
        option_type=OptionType.CALL,
        barrier=barrier,
        barrier_type=BarrierType.UP_OUT,
        maturity=maturity,
        rebate=0.0,
    )

    barrier_solver = BarrierPDESolver(pde_params)
    up_out_price = barrier_solver.price(up_out_call, pricing_env)

    # Compare with vanilla
    euro_call = EuropeanVanillaOption(
        strike=strike, option_type=OptionType.CALL, maturity=maturity
    )
    euro_solver = EuropeanPDESolver(pde_params)
    euro_price = euro_solver.price(euro_call, pricing_env)

    print(f"\nUp-and-Out Call (S={spot}, K={strike}, B={barrier}, T={maturity}y):")
    print(f"  Vanilla call: {euro_price:.6f}")
    print(f"  Up-out call:  {up_out_price:.6f}")
    print(f"  Knock-out discount: {(1 - up_out_price/euro_price)*100:.2f}%")

    # Down-and-in put
    barrier_down = 80.0
    down_in_put = BarrierOption(
        strike=strike,
        option_type=OptionType.PUT,
        barrier=barrier_down,
        barrier_type=BarrierType.DOWN_IN,
        maturity=maturity,
        rebate=0.0,
    )

    down_in_price = barrier_solver.price(down_in_put, pricing_env)

    euro_put = EuropeanVanillaOption(
        strike=strike, option_type=OptionType.PUT, maturity=maturity
    )
    euro_put_price = euro_solver.price(euro_put, pricing_env)

    print(f"\nDown-and-In Put (S={spot}, K={strike}, B={barrier_down}, T={maturity}y):")
    print(f"  Vanilla put: {euro_put_price:.6f}")
    print(f"  Down-in put: {down_in_price:.6f}")


def demo_double_barrier_option():
    """Demo: Double barrier (corridor) options."""
    print("\n" + "=" * 60)
    print("4. DOUBLE BARRIER (CORRIDOR) OPTION")
    print("=" * 60)

    spot = 100.0
    strike = 100.0
    maturity = 0.5
    vol = 0.25
    rate = 0.05
    div = 0.0

    pricing_env = create_pricing_environment(spot, vol, rate, div)
    pde_params = PDEParams(grid_size=400, time_steps=200)

    # Double knock-out call
    upper_barrier = 120.0
    lower_barrier = 80.0

    double_ko = DoubleBarrierOption(
        strike=strike,
        option_type=OptionType.CALL,
        upper_barrier=upper_barrier,
        lower_barrier=lower_barrier,
        barrier_type=DoubleBarrierType.KNOCK_OUT,
        maturity=maturity,
        rebate=0.0,
    )

    double_barrier_solver = DoubleBarrierPDESolver(pde_params)
    double_ko_price = double_barrier_solver.price(double_ko, pricing_env)

    # Compare with vanilla
    euro_call = EuropeanVanillaOption(
        strike=strike, option_type=OptionType.CALL, maturity=maturity
    )
    euro_solver = EuropeanPDESolver(pde_params)
    euro_price = euro_solver.price(euro_call, pricing_env)

    print(
        f"\nDouble Knock-Out Call (S={spot}, K={strike}, L={lower_barrier}, U={upper_barrier}):"
    )
    print(f"  Vanilla call:       {euro_price:.6f}")
    print(f"  Double KO call:     {double_ko_price:.6f}")
    print(f"  Corridor discount:  {(1 - double_ko_price/euro_price)*100:.2f}%")


def demo_one_touch_option():
    """Demo: One-touch digital options."""
    print("\n" + "=" * 60)
    print("5. ONE-TOUCH DIGITAL OPTION")
    print("=" * 60)

    spot = 100.0
    maturity = 0.5
    vol = 0.25
    rate = 0.05
    div = 0.0

    pricing_env = create_pricing_environment(spot, vol, rate, div)
    pde_params = PDEParams(grid_size=400, time_steps=200)

    # Up one-touch
    barrier = 120.0
    rebate = 100.0

    up_one_touch = OneTouchOption(
        barrier=barrier,
        barrier_direction=BarrierDirection.UP,
        maturity=maturity,
        rebate=rebate,
        payment_at_hit=False,  # Pay at expiry
        touch_type=TouchType.ONE_TOUCH,
    )

    one_touch_solver = OneTouchPDESolver(pde_params)
    up_touch_price = one_touch_solver.price(up_one_touch, pricing_env)

    print(f"\nUp One-Touch (S={spot}, B={barrier}, rebate={rebate}, pay at expiry):")
    print(f"  Price: {up_touch_price:.6f}")
    print(
        f"  Implied probability of touch: {up_touch_price / (rebate * np.exp(-rate * maturity)) * 100:.2f}%"
    )

    # No-touch (opposite)
    no_touch = OneTouchOption(
        barrier=barrier,
        barrier_direction=BarrierDirection.UP,
        maturity=maturity,
        rebate=rebate,
        payment_at_hit=False,
        touch_type=TouchType.NO_TOUCH,
    )

    no_touch_price = one_touch_solver.price(no_touch, pricing_env)

    print(f"\nUp No-Touch (same parameters):")
    print(f"  Price: {no_touch_price:.6f}")
    print(f"  One-touch + No-touch = {up_touch_price + no_touch_price:.6f}")
    print(f"  Discounted rebate = {rebate * np.exp(-rate * maturity):.6f}")


def demo_double_one_touch_option():
    """Demo: Double one-touch (range) options."""
    print("\n" + "=" * 60)
    print("6. DOUBLE ONE-TOUCH (RANGE) OPTION")
    print("=" * 60)

    spot = 100.0
    maturity = 0.5
    vol = 0.25
    rate = 0.05
    div = 0.0

    pricing_env = create_pricing_environment(spot, vol, rate, div)
    pde_params = PDEParams(grid_size=400, time_steps=200)

    # Double one-touch
    upper_barrier = 120.0
    lower_barrier = 80.0
    rebate = 100.0

    double_one_touch = DoubleOneTouchOption(
        upper_barrier=upper_barrier,
        lower_barrier=lower_barrier,
        maturity=maturity,
        rebate=rebate,
        payment_at_hit=False,
        touch_type=TouchType.DOUBLE_ONE_TOUCH,
    )

    double_touch_solver = DoubleOneTouchPDESolver(pde_params)
    double_touch_price = double_touch_solver.price(double_one_touch, pricing_env)

    print(
        f"\nDouble One-Touch (S={spot}, L={lower_barrier}, U={upper_barrier}, rebate={rebate}):"
    )
    print(f"  Price: {double_touch_price:.6f}")

    # Double no-touch
    double_no_touch = DoubleOneTouchOption(
        upper_barrier=upper_barrier,
        lower_barrier=lower_barrier,
        maturity=maturity,
        rebate=rebate,
        payment_at_hit=False,
        touch_type=TouchType.DOUBLE_NO_TOUCH,
    )

    double_no_touch_price = double_touch_solver.price(double_no_touch, pricing_env)

    print(f"\nDouble No-Touch (same parameters):")
    print(f"  Price: {double_no_touch_price:.6f}")
    print(
        f"  Double one-touch + Double no-touch = {double_touch_price + double_no_touch_price:.6f}"
    )
    print(f"  Discounted rebate = {rebate * np.exp(-rate * maturity):.6f}")


def demo_grid_convergence():
    """Demo: Grid convergence analysis."""
    print("\n" + "=" * 60)
    print("7. GRID CONVERGENCE ANALYSIS")
    print("=" * 60)

    spot = 100.0
    strike = 100.0
    maturity = 1.0
    vol = 0.20
    rate = 0.05
    div = 0.0

    pricing_env = create_pricing_environment(spot, vol, rate, div)

    # Get analytical price
    euro_call = EuropeanVanillaOption(
        strike=strike, option_type=OptionType.CALL, maturity=maturity
    )
    bs_engine = BlackScholesEngine()
    analytical_price = bs_engine.price(euro_call, pricing_env)

    print(f"\nEuropean Call (S={spot}, K={strike}, T={maturity}y)")
    print(f"Analytical (Black-Scholes): {analytical_price:.8f}")
    print()
    print(
        f"{'Grid Size':<12} {'Time Steps':<12} {'PDE Price':<14} {'Error':<12} {'Ratio'}"
    )
    print("-" * 65)

    grid_sizes = [50, 100, 200, 400, 800]
    errors = []

    for grid_size in grid_sizes:
        time_steps = grid_size // 2
        pde_params = PDEParams(grid_size=grid_size, time_steps=time_steps)
        pde_engine = EuropeanPDESolver(pde_params)

        pde_price = pde_engine.price(euro_call, pricing_env)
        error = abs(pde_price - analytical_price)
        errors.append(error)

        if len(errors) > 1:
            ratio = errors[-2] / errors[-1]
            print(
                f"{grid_size:<12} {time_steps:<12} {pde_price:<14.8f} {error:<12.2e} {ratio:.2f}"
            )
        else:
            print(
                f"{grid_size:<12} {time_steps:<12} {pde_price:<14.8f} {error:<12.2e} -"
            )

    print()
    print("Expected ratio ~4 for O(h^2) convergence (doubling grid size)")


def main():
    """Run all demos."""
    print("=" * 60)
    print("PDE PRICING ENGINE DEMONSTRATION")
    print("=" * 60)

    demo_european_option()
    demo_american_option()
    demo_barrier_option()
    demo_double_barrier_option()
    demo_one_touch_option()
    demo_double_one_touch_option()
    demo_grid_convergence()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
