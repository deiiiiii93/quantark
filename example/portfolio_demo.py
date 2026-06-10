"""
Portfolio management demonstration.

This script demonstrates:
1. Creating a portfolio with multiple positions
2. Adding positions with different products and engines
3. Calculating portfolio value and P&L
4. Computing aggregated Greeks across positions
5. Taking portfolio snapshots
6. Exporting to Excel and Parquet formats
"""

import sys
from pathlib import Path
from datetime import datetime


from quantark.portfolio import Portfolio, PortfolioSnapshot, PortfolioExporter
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_portfolio_summary(portfolio: Portfolio):
    """Print portfolio summary."""
    summary = portfolio.get_summary()
    print(f"\nPortfolio: {summary['portfolio_name']}")
    print(f"Created: {summary['creation_date']}")
    print(f"Positions: {summary['num_positions']} ({summary['long_positions']} long, {summary['short_positions']} short)")
    print(f"Underlyings: {summary['num_underlyings']} - {summary['underlyings']}")
    print(f"Total Value: ${summary['total_value']:,.2f}")
    print(f"Total P&L: ${summary['total_pnl']:,.2f} ({summary['pnl_percentage']:.2f}%)")


def print_greeks(greeks: dict, title: str = "Portfolio Greeks"):
    """Print Greeks with nice formatting."""
    print(f"\n{title}:")
    print(f"  Market Value: ${greeks.get('market_value', 0):12,.2f}")
    print(f"  Delta:        {greeks.get('delta', 0):12,.6f}")
    print(f"  Gamma:        {greeks.get('gamma', 0):12,.6f}")
    print(f"  Vega:         {greeks.get('vega', 0):12,.6f}")
    print(f"  Theta:        {greeks.get('theta', 0):12,.6f} (per day)")
    print(f"  Rho:          {greeks.get('rho', 0):12,.6f}")


def main():
    """Run portfolio management demonstration."""
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  QUANTARK - Portfolio Management Demonstration".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    
    # =========================================================================
    # STEP 1: Set up pricing environments for multiple underlyings
    # =========================================================================
    print_section("STEP 1: Setup Pricing Environments")
    
    valuation_date = datetime(2024, 1, 1)
    
    # Pricing environment for AAPL
    aapl_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=180.0, asset_name="AAPL"),
        vol_surface=FlatVolSurface(volatility=0.25),  # 25% vol
        rate_curve=FlatRateCurve(rate=0.05),  # 5% rate
        div_yield=ContinuousDividendYield(div_yield=0.01),  # 1% div yield
        valuation_date=valuation_date
    )
    print(f"AAPL: Spot=${aapl_env.spot:.2f}, Vol={aapl_env.get_vol(180, 1.0):.2%}, Rate={aapl_env.get_rate(1.0):.2%}")
    
    # Pricing environment for MSFT
    msft_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=370.0, asset_name="MSFT"),
        vol_surface=FlatVolSurface(volatility=0.22),  # 22% vol
        rate_curve=FlatRateCurve(rate=0.05),
        div_yield=ContinuousDividendYield(div_yield=0.008),  # 0.8% div yield
        valuation_date=valuation_date
    )
    print(f"MSFT: Spot=${msft_env.spot:.2f}, Vol={msft_env.get_vol(370, 1.0):.2%}, Rate={msft_env.get_rate(1.0):.2%}")
    
    # =========================================================================
    # STEP 2: Create portfolio with pricing environments
    # =========================================================================
    print_section("STEP 2: Create Portfolio")
    
    portfolio = Portfolio(
        portfolio_name="Tech Options Portfolio",
        pricing_environments={
            'AAPL': aapl_env,
            'MSFT': msft_env
        },
        creation_date=valuation_date
    )
    print(f"Created portfolio: {portfolio}")
    
    # =========================================================================
    # STEP 3: Add positions to portfolio
    # =========================================================================
    print_section("STEP 3: Add Positions")
    
    # Create engine (will be reused for options priced with Black-Scholes)
    bs_engine = BlackScholesEngine()
    
    # Position 1: Long 10 AAPL call options (ATM, 1 year)
    aapl_call = EuropeanVanillaOption(
        strike=180.0,
        option_type=OptionType.CALL,
        maturity=1.0
    )
    entry_price_1 = bs_engine.price(aapl_call, aapl_env)
    pos1 = portfolio.add_position(
        product=aapl_call,
        quantity=10,
        entry_price=entry_price_1,
        underlying='AAPL',
        engine=bs_engine,
        entry_timestamp=valuation_date
    )
    print(f"Position 1: LONG 10 x AAPL Call (K=180, T=1y) @ ${entry_price_1:.2f}")
    
    # Position 2: Short 5 AAPL put options (OTM, 1 year)
    aapl_put = EuropeanVanillaOption(
        strike=170.0,
        option_type=OptionType.PUT,
        maturity=1.0
    )
    entry_price_2 = bs_engine.price(aapl_put, aapl_env)
    pos2 = portfolio.add_position(
        product=aapl_put,
        quantity=-5,  # Negative = short
        entry_price=entry_price_2,
        underlying='AAPL',
        engine=bs_engine,
        entry_timestamp=valuation_date
    )
    print(f"Position 2: SHORT 5 x AAPL Put (K=170, T=1y) @ ${entry_price_2:.2f}")
    
    # Position 3: Long 8 MSFT call options (slightly ITM, 6 months)
    msft_call = EuropeanVanillaOption(
        strike=360.0,
        option_type=OptionType.CALL,
        maturity=0.5
    )
    entry_price_3 = bs_engine.price(msft_call, msft_env)
    pos3 = portfolio.add_position(
        product=msft_call,
        quantity=8,
        entry_price=entry_price_3,
        underlying='MSFT',
        engine=bs_engine,
        entry_timestamp=valuation_date
    )
    print(f"Position 3: LONG 8 x MSFT Call (K=360, T=0.5y) @ ${entry_price_3:.2f}")
    
    # Position 4: Long 3 MSFT put options (OTM, 6 months)
    msft_put = EuropeanVanillaOption(
        strike=350.0,
        option_type=OptionType.PUT,
        maturity=0.5
    )
    entry_price_4 = bs_engine.price(msft_put, msft_env)
    pos4 = portfolio.add_position(
        product=msft_put,
        quantity=3,
        entry_price=entry_price_4,
        underlying='MSFT',
        engine=bs_engine,
        entry_timestamp=valuation_date
    )
    print(f"Position 4: LONG 3 x MSFT Put (K=350, T=0.5y) @ ${entry_price_4:.2f}")
    
    print(f"\nTotal positions in portfolio: {len(portfolio)}")
    
    # =========================================================================
    # STEP 4: Portfolio valuation and P&L
    # =========================================================================
    print_section("STEP 4: Portfolio Valuation")
    
    print_portfolio_summary(portfolio)
    
    # Show positions DataFrame
    print("\nPosition Details:")
    df = portfolio.to_dataframe()
    print(df[['position_id', 'underlying', 'direction', 'quantity', 'entry_price', 
              'current_price', 'market_value', 'pnl']].to_string(index=False))
    
    # =========================================================================
    # STEP 5: Calculate portfolio Greeks
    # =========================================================================
    print_section("STEP 5: Portfolio Greeks")
    
    greeks_calc = GreeksCalculator()
    
    # Portfolio-wide Greeks
    portfolio_greeks = portfolio.get_portfolio_greeks(greeks_calc)
    print_greeks(portfolio_greeks, "Aggregated Portfolio Greeks")
    
    # Greeks by underlying
    print("\n" + "-" * 80)
    aapl_greeks = portfolio.get_greeks_by_underlying('AAPL', greeks_calc)
    print_greeks(aapl_greeks, "AAPL Positions Greeks")
    
    print("\n" + "-" * 80)
    msft_greeks = portfolio.get_greeks_by_underlying('MSFT', greeks_calc)
    print_greeks(msft_greeks, "MSFT Positions Greeks")
    
    # =========================================================================
    # STEP 6: Position management
    # =========================================================================
    print_section("STEP 6: Position Management")
    
    print(f"\nOriginal AAPL call position: {pos1}")
    print(f"  Quantity: {pos1.quantity}")
    print(f"  Entry Price: ${pos1.entry_price:.2f}")
    
    # Update position (e.g., increase quantity)
    portfolio.update_position(pos1.position_id, quantity=15)
    print(f"\nAfter updating quantity to 15:")
    print(f"  Quantity: {pos1.quantity}")
    
    # Get updated valuation
    new_total_value = portfolio.get_portfolio_value()
    print(f"\nUpdated portfolio value: ${new_total_value:,.2f}")
    
    # Remove a position
    print(f"\nRemoving position: {pos4.position_id[:8]}... (MSFT Put)")
    removed = portfolio.remove_position(pos4.position_id)
    print(f"Removed: {removed}")
    print(f"Positions remaining: {len(portfolio)}")
    
    # =========================================================================
    # STEP 7: Portfolio snapshots
    # =========================================================================
    print_section("STEP 7: Portfolio Snapshots")
    
    snapshot = PortfolioSnapshot.from_portfolio(
        portfolio,
        greeks_calc,
        metadata={'note': 'Post-position updates'}
    )
    print(f"\nCreated snapshot: {snapshot}")
    
    snapshot_summary = snapshot.get_summary()
    print(f"\nSnapshot Summary:")
    print(f"  Timestamp: {snapshot_summary['timestamp']}")
    print(f"  Total Value: ${snapshot_summary['total_value']:,.2f}")
    print(f"  Total P&L: ${snapshot_summary['total_pnl']:,.2f}")
    print(f"  Positions: {snapshot_summary['num_positions']}")
    
    # =========================================================================
    # STEP 8: Export portfolio
    # =========================================================================
    print_section("STEP 8: Export Portfolio")
    
    exporter = PortfolioExporter(base_path="portfolio/data/demo")
    
    # Export to Excel
    print("\nExporting to Excel...")
    excel_path = exporter.export_to_excel(portfolio, greeks_calculator=greeks_calc)
    print(f"Exported to: {excel_path}")
    print(f"  Sheets: Positions, Summary, Greeks_by_Position, Greeks_by_Underlying")
    
    # Export to Parquet
    print("\nExporting to Parquet...")
    parquet_path = exporter.export_to_parquet(
        portfolio,
        include_greeks=True,
        greeks_calculator=greeks_calc
    )
    print(f"Exported to: {parquet_path}")
    
    # Export snapshot
    print("\nExporting snapshot...")
    snapshot_path = exporter.export_snapshot_to_parquet(snapshot)
    print(f"Snapshot saved to: {snapshot_path}")
    
    # Show storage info
    print("\nStorage Information:")
    storage_info = exporter.get_storage_info()
    print(f"  Base path: {storage_info['base_path']}")
    print(f"  Files: {storage_info['num_files']}")
    print(f"  Total size: {storage_info['total_size_mb']:.2f} MB")
    
    # =========================================================================
    # STEP 9: Load portfolio data
    # =========================================================================
    print_section("STEP 9: Load Portfolio Data")
    
    print(f"\nLoading portfolio from: {parquet_path}")
    loaded_df = exporter.load_from_parquet(parquet_path)
    print(f"\nLoaded {len(loaded_df)} positions")
    print("\nMetadata:")
    for key, value in loaded_df.attrs['metadata'].items():
        print(f"  {key}: {value}")
    
    # =========================================================================
    # COMPLETION
    # =========================================================================
    print_section("DEMONSTRATION COMPLETED")
    
    print("\nKey capabilities demonstrated:")
    print("  ✓ Multi-underlying portfolio management")
    print("  ✓ Position tracking with entry details")
    print("  ✓ Each position has its own pricing engine")
    print("  ✓ Portfolio valuation and P&L calculation")
    print("  ✓ Aggregated Greeks across positions")
    print("  ✓ Position management (add/update/remove)")
    print("  ✓ Portfolio snapshots for point-in-time state")
    print("  ✓ Export to Excel and Parquet formats")
    print("  ✓ Load portfolio data from storage")
    
    print(f"\nFinal portfolio state:")
    print_portfolio_summary(portfolio)
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

