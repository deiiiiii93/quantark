"""
Benchmark script for SnowballMCEngine to test Dask parallelization speedup.
"""

import time
import datetime
import numpy as np
from asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from asset.equity.product.option.snowball_option import SnowballOption
from asset.equity.product.option.snowball_config import (
    BarrierConfig,
    PayoffConfig,
    AccrualConfig,
)
from asset.equity.param import MCParams
from priceenv import PricingEnvironment
from param import (
    SpotQuote,
    FlatVolSurface,
    FlatRateCurve,
    ContinuousDividendYield,
)
from util.enum import ObservationType, CouponPayType, EngineType
from util.enum.engine_enums import MonteCarloMethod

def run_benchmark():
    # 1. Setup Market Data
    valuation_date = datetime.date(2023, 1, 1)
    spot_price = 100.0
    risk_free_rate = 0.03
    volatility = 0.20
    dividend_yield = 0.0

    pricing_env = PricingEnvironment(
        valuation_date=datetime.datetime.combine(valuation_date, datetime.time.min),
        rate_curve=FlatRateCurve(risk_free_rate),
        spot_quote=SpotQuote(spot_price, valuation_date),
        vol_surface=FlatVolSurface(volatility),
        div_yield=ContinuousDividendYield(dividend_yield),
    )

    # 2. Setup Snowball Product
    # 2-year product, monthly observations (24 steps)
    maturity = 2.0
    num_obs = 24
    obs_dates = [maturity * (i + 1) / num_obs for i in range(num_obs)]
    
    # 100/100/75 structure
    ko_barriers = [100.0] * num_obs  # 100% KO
    ko_rates = [0.15] * num_obs      # 15% annualized coupon
    ki_barrier = 75.0                # 75% KI
    
    barrier_config = BarrierConfig(
        ko_barrier=ko_barriers,
        ko_rate=ko_rates,
        ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=obs_dates,
        ki_barrier=ki_barrier,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=obs_dates, # KI monitored on same dates
        ki_continuous=False
    )
    
    payoff_config = PayoffConfig(
        participation_rate=1.0,
        include_principal=True
    )
    
    accrual_config = AccrualConfig(
        coupon_pay_type=CouponPayType.INSTANT,
        is_annualized=True
    )

    product = SnowballOption(
        initial_price=spot_price,
        strike=spot_price, # ATM strike
        notional=1000000.0, # 1M notional
        barrier_config=barrier_config,
        payoff_config=payoff_config,
        accrual_config=accrual_config,
        maturity=maturity
    )

    # 3. Setup Monte Carlo Parameters
    # Use a large number of paths to make parallelization worthwhile
    num_paths = 500000 
    params = MCParams(
        num_paths=num_paths,
        seed=42,
        time_steps=252 # Ignored by Snowball engine but required for validation
    )

    print(f"Benchmarking Snowball Pricing (Paths: {num_paths:,})")
    print("-" * 60)

    # 4. Benchmark: No Dask (Single Threaded)
    print("Running Single-Threaded (NumPy)...")
    engine_no_dask = SnowballMCEngine(
        params=params,
        method=EngineType.MONTE_CARLO(MonteCarloMethod.PSEUDO),
        use_dask=False
    )
    
    start_time = time.perf_counter()
    price_no_dask = engine_no_dask.price(product, pricing_env)
    end_time = time.perf_counter()
    time_no_dask = end_time - start_time
    
    print(f"Price: {price_no_dask:,.2f}")
    print(f"Time:  {time_no_dask:.4f} seconds")
    print("-" * 60)

    # 5. Benchmark: With Dask (Parallel)
    # Note: Dask overhead might dominate for small workloads. 
    # For 500k paths, it should show improvement on multi-core machines.
    try:
        import dask
        print("Running Parallel (Dask)...")
        num_batches = 8 # Adjust based on CPU cores
        
        engine_dask = SnowballMCEngine(
            params=params,
            method=EngineType.MONTE_CARLO(MonteCarloMethod.PSEUDO),
            use_dask=True,
            num_batches=num_batches
        )
        
        start_time = time.perf_counter()
        price_dask = engine_dask.price(product, pricing_env)
        end_time = time.perf_counter()
        time_dask = end_time - start_time
        
        print(f"Price: {price_dask:,.2f}")
        print(f"Time:  {time_dask:.4f} seconds")
        print(f"Batches: {num_batches}")
        
        # Calculate speedup
        speedup = time_no_dask / time_dask
        print(f"Speedup: {speedup:.2f}x")
        
        # Verify prices match (within MC error tolerance, though same seed should yield exact match if implemented correctly)
        # Note: Dask splitting might change random stream consumption order if not carefully handled.
        # In this implementation, each batch gets a seed offset: seed + batch_id * 1000
        # The single threaded run uses seed + 0 * 1000.
        # So the random streams ARE different. Prices will not be identical but statistically consistent.
        diff_pct = abs(price_dask - price_no_dask) / price_no_dask
        print(f"Price Difference: {diff_pct:.6%}")

    except ImportError:
        print("Dask not installed. Skipping parallel benchmark.")

if __name__ == "__main__":
    run_benchmark()
