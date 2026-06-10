"""
Delta one products demonstration.

This script demonstrates:
1. Creating different delta one products (Stock, Index, ETF, Futures)
2. Pricing with forward pricing and mark-to-market
3. Computing Greeks for delta one products
4. Adding delta one products to a portfolio
5. Portfolio-level hedging calculations
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta


from quantark.portfolio import Portfolio
from quantark.asset.equity.product.deltaone import SpotInstrument, Futures
from quantark.asset.equity.engine.analytical import DeltaOneEngine
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import DeltaOneType


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_product_details(product, engine, pricing_env):
    """Print product details and pricing."""
    print(f"\nProduct: {product}")
    price = engine.price(product, pricing_env)
    print(f"Price: ${price:.2f}")
    
    # Calculate Greeks
    greeks = engine.calculate_greeks(product, pricing_env)
    print(f"Greeks:")
    print(f"  Delta:  {greeks['delta']:10.4f}")
    print(f"  Gamma:  {greeks['gamma']:10.4f}")
    print(f"  Vega:   {greeks['vega']:10.4f}")
    print(f"  Theta:  {greeks['theta']:10.4f}")
    print(f"  Rho:    {greeks['rho']:10.4f}")


def main():
    """Run delta one products demonstration."""
    
    print_section("Delta One Products Demonstration")
    
    # =========================================================================
    # Setup: Create pricing environment
    # =========================================================================
    print_section("1. Setup Pricing Environment")
    
    valuation_date = datetime(2024, 1, 15)
    
    # Market data for SPX
    spot_spx = SpotQuote(spot=4500.0, timestamp=valuation_date)
    vol_surface_spx = FlatVolSurface(volatility=0.18)
    rate_curve = FlatRateCurve(rate=0.045)  # 4.5% risk-free rate
    div_yield = ContinuousDividendYield(div_yield=0.015)  # 1.5% dividend yield
    
    pricing_env_spx = PricingEnvironment(
        spot_quote=spot_spx,
        vol_surface=vol_surface_spx,
        rate_curve=rate_curve,
        div_yield=div_yield,
        valuation_date=valuation_date
    )
    
    print(f"Valuation Date: {valuation_date.date()}")
    print(f"SPX Spot: ${pricing_env_spx.spot:,.2f}")
    print(f"Risk-free Rate: {pricing_env_spx.get_rate(1.0):.2%}")
    print(f"Dividend Yield: {pricing_env_spx.get_div_yield(1.0):.2%}")
    print(f"Volatility: {pricing_env_spx.get_vol(pricing_env_spx.spot, 1.0):.2%}")
    
    # =========================================================================
    # Example 1: Stock Position
    # =========================================================================
    print_section("2. Stock Position (AAPL)")
    
    # Market data for AAPL
    spot_aapl = SpotQuote(spot=185.50, timestamp=valuation_date)
    vol_surface_aapl = FlatVolSurface(volatility=0.25)
    div_yield_aapl = ContinuousDividendYield(div_yield=0.005)  # 0.5% dividend yield
    
    pricing_env_aapl = PricingEnvironment(
        spot_quote=spot_aapl,
        vol_surface=vol_surface_aapl,
        rate_curve=rate_curve,
        div_yield=div_yield_aapl,
        valuation_date=valuation_date
    )
    
    # Create stock
    stock = SpotInstrument(
        underlying="AAPL",
        deltaone_type=DeltaOneType.STOCK
    )
    
    # Create engine
    engine_theoretical = DeltaOneEngine(use_market_price=False)
    
    print_product_details(stock, engine_theoretical, pricing_env_aapl)
    
    # Calculate forward price
    print(f"\nForward Pricing:")
    for months in [1, 3, 6, 12]:
        T = months / 12.0
        r = pricing_env_aapl.get_rate(T)
        q = pricing_env_aapl.get_div_yield(T)
        forward = stock.get_forward_price(pricing_env_aapl.spot, r, q, T)
        print(f"  {months:2d} months: ${forward:.2f}")
    
    # =========================================================================
    # Example 2: ETF Position
    # =========================================================================
    print_section("3. ETF Position (SPY)")
    
    # SPY tracks SPX closely
    spot_spy = SpotQuote(spot=450.0, timestamp=valuation_date)
    
    pricing_env_spy = PricingEnvironment(
        spot_quote=spot_spy,
        vol_surface=vol_surface_spx,
        rate_curve=rate_curve,
        div_yield=div_yield,
        valuation_date=valuation_date
    )
    
    # Create ETF
    etf = SpotInstrument(
        underlying="SPY",
        deltaone_type=DeltaOneType.ETF
    )
    
    print_product_details(etf, engine_theoretical, pricing_env_spy)
    
    # =========================================================================
    # Example 3: Index Position
    # =========================================================================
    print_section("4. Index Position (SPX)")
    
    # Create index
    index = SpotInstrument(
        underlying="SPX",
        deltaone_type=DeltaOneType.INDEX
    )
    
    print_product_details(index, engine_theoretical, pricing_env_spx)
    
    # =========================================================================
    # Example 4: Futures Contract - Theoretical Pricing
    # =========================================================================
    print_section("5. Futures Contract - Theoretical Pricing (ES)")
    
    # E-mini S&P 500 futures expiring in 3 months
    maturity_date = valuation_date + timedelta(days=90)
    
    futures = Futures(
        underlying="ES",
        multiplier=50.0,
        maturity_date=maturity_date,
        basis=2.5,  # Futures trading slightly above fair value
        basis_decay_rate=2.0
    )
    
    print(f"Futures Contract: {futures}")
    print(f"Maturity: {maturity_date.date()}")
    print(f"Multiplier: {futures.multiplier:.0f}")
    print(f"Basis: {futures.basis:.2f}")
    
    # Price using theoretical model
    price_theoretical = engine_theoretical.price(futures, pricing_env_spx)
    print(f"\nTheoretical Price: ${price_theoretical:.2f}")
    print(f"Contract Value: ${price_theoretical * futures.multiplier:,.2f}")
    
    # Calculate Greeks
    greeks = engine_theoretical.calculate_greeks(futures, pricing_env_spx)
    print(f"\nGreeks:")
    print(f"  Delta:  {greeks['delta']:10.4f}")
    print(f"  Gamma:  {greeks['gamma']:10.4f}")
    print(f"  Theta:  {greeks['theta']:10.4f}")
    print(f"  Rho:    {greeks['rho']:10.4f}")
    
    # =========================================================================
    # Example 5: Futures Contract - Mark-to-Market Pricing
    # =========================================================================
    print_section("6. Futures Contract - Mark-to-Market Pricing")
    
    # Update futures with observed market price
    futures.update_market_price(4515.25)
    print(f"Market Price: ${futures.market_price:.2f}")
    
    # Create MTM engine
    engine_mtm = DeltaOneEngine(use_market_price=True)
    
    price_mtm = engine_mtm.price(futures, pricing_env_spx)
    print(f"Mark-to-Market Price: ${price_mtm:.2f}")
    print(f"Contract Value (MTM): ${price_mtm * futures.multiplier:,.2f}")
    
    # Compare theoretical vs MTM
    price_diff = price_mtm - price_theoretical
    print(f"\nPrice Difference (MTM - Theoretical): ${price_diff:.2f}")
    print(f"Contract Value Difference: ${price_diff * futures.multiplier:,.2f}")
    
    # =========================================================================
    # Example 6: Portfolio with Delta One Products
    # =========================================================================
    print_section("7. Portfolio with Delta One Products")
    
    # Create portfolio
    portfolio = Portfolio(
        portfolio_name="Delta One Hedging Portfolio",
        pricing_environments={"AAPL": pricing_env_aapl, "SPY": pricing_env_spy, "ES": pricing_env_spx}
    )
    
    # Add long stock position
    portfolio.add_position(
        product=stock,
        quantity=1000,  # 1000 shares
        entry_price=185.50,
        underlying="AAPL",
        engine=engine_theoretical,
        entry_timestamp=valuation_date
    )
    
    # Add long ETF position
    portfolio.add_position(
        product=etf,
        quantity=500,  # 500 shares
        entry_price=450.0,
        underlying="SPY",
        engine=engine_theoretical,
        entry_timestamp=valuation_date
    )
    
    # Add short futures hedge
    portfolio.add_position(
        product=futures,
        quantity=-2,  # Short 2 contracts
        entry_price=4510.0,
        underlying="ES",
        engine=engine_mtm,  # Use mark-to-market pricing
        entry_timestamp=valuation_date
    )
    
    # Print portfolio summary
    print(f"\nPortfolio: {portfolio.portfolio_name}")
    print(f"Total Positions: {len(portfolio.positions)}")
    
    summary = portfolio.get_summary()
    print(f"\nPortfolio Summary:")
    print(f"  Total Value: ${summary['total_value']:,.2f}")
    print(f"  Total P&L: ${summary['total_pnl']:,.2f}")
    
    # Calculate portfolio Greeks
    # For delta one products, we calculate Greeks directly from engines
    print(f"\nPortfolio Greeks:")
    
    total_delta = 0.0
    total_gamma = 0.0
    total_vega = 0.0
    total_theta = 0.0
    
    for position in portfolio.positions.values():
        pricing_env = portfolio.pricing_environments[position.underlying]
        greeks = position.engine.calculate_greeks(position.product, pricing_env)
        
        # Scale by quantity
        total_delta += greeks['delta'] * position.quantity
        total_gamma += greeks['gamma'] * position.quantity
        total_vega += greeks['vega'] * position.quantity
        total_theta += greeks['theta'] * position.quantity
    
    print(f"  Total Delta: {total_delta:,.2f}")
    print(f"  Total Gamma: {total_gamma:,.2f}")
    print(f"  Total Vega:  {total_vega:,.2f}")
    print(f"  Total Theta: {total_theta:,.2f}")
    
    # Analyze hedging effectiveness
    print(f"\nHedging Analysis:")
    print(f"  Delta: {total_delta:,.2f}")
    if abs(total_delta) < 100:
        print(f"  Status: Well hedged (delta near zero)")
    else:
        print(f"  Status: Under-hedged (significant delta exposure)")
    
    # =========================================================================
    # Example 7: Forward Pricing Term Structure
    # =========================================================================
    print_section("8. Forward Pricing Term Structure")
    
    print("\nForward Prices for ES Futures:")
    print(f"{'Months':>10} {'Time':>8} {'Forward':>12} {'Basis Decay':>12}")
    print("-" * 45)
    
    for months in [1, 2, 3, 6, 9, 12]:
        T = months / 12.0
        r = pricing_env_spx.get_rate(T)
        q = pricing_env_spx.get_div_yield(T)
        forward = futures.get_forward_price(pricing_env_spx.spot, r, q, T)
        
        # Calculate basis decay
        import math
        basis_decay = futures.basis * math.exp(-futures.basis_decay_rate * T)
        
        print(f"{months:10d} {T:8.4f} {forward:12.2f} {basis_decay:12.4f}")
    
    print_section("Demo Complete!")
    print("\nKey Takeaways:")
    print("1. Delta one products (stocks, ETFs, indices, futures) have delta ≈ 1.0")
    print("2. Futures support both theoretical pricing (with basis) and mark-to-market")
    print("3. Forward pricing uses cost-of-carry: F = S * exp((r - q) * T)")
    print("4. Delta one products are ideal for hedging option portfolios")
    print("5. Portfolio Greeks can be aggregated to assess overall risk exposure")


if __name__ == "__main__":
    main()

