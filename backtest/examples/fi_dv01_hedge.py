"""
Fixed Income DV01-neutral hedging backtest example.

This script demonstrates FI hedging:
- Create a bond portfolio position
- Set up a DV01-neutral hedging strategy using bond futures
- Run backtest with mock FI market data
- Compare hedged vs unhedged performance
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backtest.fi import (
    FIBacktestEngine,
    FIBacktestConfig,
)
from backtest.strategy import DV01NeutralStrategy
from backtest.transaction_costs import ZeroCostModel, ProportionalCostModel
from backtest.visualizer import StaticVisualizer
from portfolio.fi import FIPosition
from asset.bond.product.couponbond.fixed_bond import FixedBond
from asset.bond.engine.discount.bond_discount_engine import BondDiscountEngine
from priceenv import PricingEnvironment
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from util.marketdata import MockMarketDataAdapter
from util.calendar import DayCountConvention
from util.enum import PaymentFrequency


def create_initial_bond_position(underlying: str, start_date: datetime) -> FIPosition:
    """Create an initial bond position."""
    # Create issue and maturity dates (simple dates to avoid schedule generation issues)
    # Issue: 2023-01-01, Maturity: 2033-01-01 (10 years)
    issue_date = datetime(2023, 1, 1)
    maturity_date = datetime(2033, 1, 1)
    
    # Create a 10-year fixed rate bond
    bond = FixedBond(
        issue_date=issue_date,
        maturity_date=maturity_date,
        denominator=1000000.0,  # $1M denominator per bond
        coupon_rate=0.05,    # 5% coupon
        payment_frequency=PaymentFrequency.SEMI_ANNUAL,
        day_count_convention=DayCountConvention.THIRTY_360_US,
    )
    
    # Create pricing environment for the bond engine
    pricing_env = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0, asset_name=underlying),
        vol_surface=FlatVolSurface(volatility=0.01),
        rate_curve=FlatRateCurve(rate=0.045),  # 4.5% flat rate
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=start_date
    )
    
    # Create pricing engine
    engine = BondDiscountEngine(pricing_env=pricing_env)
    
    # Calculate initial price (use 100 as clean price approximation)
    initial_price = 100.0  # Par
    
    # Create position
    position = FIPosition(
        product=bond,
        quantity=10,  # 10 bonds = $10M notional
        entry_price=initial_price,
        underlying=underlying,
        engine=engine,
        entry_timestamp=start_date,
        notional_per_unit=1000000.0  # $1M per bond
    )
    
    return position


def run_backtest(hedged: bool = True):
    """
    Run FI backtest with or without hedging.
    
    Args:
        hedged: If True, run with DV01 hedging. If False, unhedged.
    """
    # =========================================================================
    # STEP 1: Configuration
    # =========================================================================
    print(f"\n{'='*80}")
    mode = "HEDGED" if hedged else "UNHEDGED"
    print(f"FI BACKTEST: {mode}".center(80))
    print(f"{'='*80}")
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)  # 1 year
    underlying = "UST_10Y"
    
    print(f"\nConfiguration:")
    print(f"  Underlying: {underlying}")
    print(f"  Period: {start_date.date()} to {end_date.date()}")
    print(f"  Mode: {mode}")
    
    # =========================================================================
    # STEP 2: Create initial bond position
    # =========================================================================
    print(f"\nCreating initial position...")
    
    initial_position = create_initial_bond_position(underlying, start_date)
    
    print(f"  Bond: {initial_position.product}")
    print(f"  Quantity: {initial_position.quantity} bonds")
    print(f"  Notional: ${initial_position.quantity * initial_position.notional_per_unit:,.0f}")
    print(f"  Entry Price: ${initial_position.entry_price:.2f}")
    
    # =========================================================================
    # STEP 3: Configure strategy
    # =========================================================================
    if hedged:
        strategy = DV01NeutralStrategy(
            name="DV01_Neutral_Daily",
            dv01_threshold=50000.0,  # Hedge when |DV01| > $50,000
            rebalance_frequency="daily",
            hedge_instrument="bond_futures",
            hedge_ratio=1.0,
            target_dv01=0.0,
            futures_dv01=1000.0,  # $1,000 DV01 per futures contract
        )
        print(f"\nStrategy: {strategy.name}")
        print(f"  DV01 Threshold: ${strategy.dv01_threshold:,.0f}")
        print(f"  Target DV01: ${strategy.target_dv01:,.0f}")
        print(f"  Futures DV01: ${strategy.futures_dv01:,.0f} per contract")
    else:
        # Create a "never hedge" strategy with very high threshold
        strategy = DV01NeutralStrategy(
            name="Unhedged",
            dv01_threshold=float('inf'),  # Never triggers
            rebalance_frequency="daily",
            futures_dv01=1000.0,
        )
        print(f"\nStrategy: {strategy.name} (no hedging)")
    
    # =========================================================================
    # STEP 4: Set up market data adapter
    # =========================================================================
    print(f"\nSetting up market data...")
    
    adapter = MockMarketDataAdapter(seed=42)
    # Configure for fixed income
    adapter.set_asset_config(underlying, {
        'initial_spot': 100.0,
        'initial_vol': 0.01,
        'initial_rate': 0.045,  # 4.5% starting rate
        'initial_div_yield': 0.0,
        'drift': 0.0,
        'vol_of_vol': 0.20,     # Rate volatility
        'jump_intensity': 0.5,
    })
    
    # =========================================================================
    # STEP 5: Configure backtest
    # =========================================================================
    config = FIBacktestConfig(
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        underlying=underlying,
        initial_positions=[initial_position],
        market_data_adapter=adapter,
        transaction_cost_model=ProportionalCostModel(commission_rate=0.0001),  # 1bp
        frequency='D',
        logging_level='INFO',
        calculate_risk_measures=True,
    )
    
    # =========================================================================
    # STEP 6: Run backtest
    # =========================================================================
    print(f"\nRunning backtest...")
    
    engine = FIBacktestEngine(config)
    results = engine.run()
    
    # =========================================================================
    # STEP 7: Display results
    # =========================================================================
    print(f"\n{'='*80}")
    print("RESULTS".center(80))
    print(f"{'='*80}")
    
    summary = results.get_summary()
    
    print(f"\nPerformance:")
    print(f"  Initial Value: ${summary['initial_value']:,.2f}")
    print(f"  Final Value: ${summary['final_value']:,.2f}")
    print(f"  Total P&L: ${summary['total_pnl']:,.2f}")
    print(f"  Total Return: {summary['total_return_pct']:.2f}%")
    
    print(f"\nHedging:")
    print(f"  Number of Hedges: {summary['num_hedges']}")
    print(f"  Total Transaction Costs: ${summary['total_transaction_costs']:,.2f}")
    
    if 'final_dv01' in summary:
        print(f"  Final DV01: ${summary['final_dv01']:,.2f}")
    
    # Calculate metrics
    metrics = results.metrics.calculate_all_metrics()
    
    print(f"\nRisk Metrics:")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Volatility: {metrics['volatility']*100:.2f}%")
    
    if hedged:
        print(f"\nDV01 Hedging Metrics:")
        print(f"  DV01 Tracking Error: ${metrics['dv01_tracking_error']:,.2f}")
        print(f"  Avg Absolute DV01: ${metrics['avg_abs_dv01']:,.2f}")
        print(f"  Hedge Effectiveness: {metrics['dv01_hedge_effectiveness']*100:.1f}%")
    
    return results


def main():
    """Run hedged and unhedged backtests and compare."""
    print("=" * 80)
    print("FIXED INCOME DV01-NEUTRAL HEDGING BACKTEST".center(80))
    print("=" * 80)
    
    # Run both hedged and unhedged
    results_unhedged = run_backtest(hedged=False)
    results_hedged = run_backtest(hedged=True)
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON: HEDGED vs UNHEDGED".center(80))
    print("=" * 80)
    
    h_summary = results_hedged.get_summary()
    u_summary = results_unhedged.get_summary()
    
    print(f"\n{'Metric':<30} {'Unhedged':>20} {'Hedged':>20}")
    print("-" * 70)
    print(f"{'Total P&L':<30} ${u_summary['total_pnl']:>18,.2f} ${h_summary['total_pnl']:>18,.2f}")
    print(f"{'Total Return':<30} {u_summary['total_return_pct']:>19.2f}% {h_summary['total_return_pct']:>19.2f}%")
    print(f"{'Number of Hedges':<30} {u_summary['num_hedges']:>20} {h_summary['num_hedges']:>20}")
    print(f"{'Transaction Costs':<30} ${u_summary['total_transaction_costs']:>18,.2f} ${h_summary['total_transaction_costs']:>18,.2f}")
    
    h_metrics = results_hedged.metrics.calculate_all_metrics()
    u_metrics = results_unhedged.metrics.calculate_all_metrics()
    
    print(f"\n{'Risk Metric':<30} {'Unhedged':>20} {'Hedged':>20}")
    print("-" * 70)
    print(f"{'Sharpe Ratio':<30} {u_metrics['sharpe_ratio']:>20.3f} {h_metrics['sharpe_ratio']:>20.3f}")
    print(f"{'Max Drawdown':<30} {u_metrics['max_drawdown_pct']:>19.2f}% {h_metrics['max_drawdown_pct']:>19.2f}%")
    print(f"{'Volatility':<30} {u_metrics['volatility']*100:>19.2f}% {h_metrics['volatility']*100:>19.2f}%")
    print(f"{'DV01 Tracking Error':<30} ${u_metrics['dv01_tracking_error']:>18,.0f} ${h_metrics['dv01_tracking_error']:>18,.0f}")
    
    print("\n" + "=" * 80)
    print("Backtest complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
