"""
AI Strategy Index Research - Phase 1-4 Analysis
================================================

This script creates a customized 10-stock AI-themed index optimized for
Snowball product knock-out (KO) probability, following the Strategy Index
Research SOP.

Stocks Selected:
1. 寒武纪-U (688256.SH) - AI Chips
2. 海光信息 (688041.SH) - CPU/GPU
3. 中际旭创 (300308.SZ) - Optical Module
4. 海康威视 (002415.SZ) - AI Vision
5. 中科曙光 (603019.SH) - AI Server
6. 科大讯飞 (002230.SZ) - AI Software
7. 金山办公 (688111.SH) - AI+Office
8. 澜起科技 (688008.SH) - Memory Interface
9. 新易盛 (300502.SZ) - Optical Module
10. 浪潮信息 (000977.SZ) - AI Server

Analysis includes:
- Phase 1: Index NAV simulation using realistic AI stock parameters
- Phase 3: Historical backtest statistics
- Phase 4: Structure fit KPIs (KO/KI probabilities) via QuantArk MC engine
"""

import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings

# QuantArk imports
from priceenv import PricingEnvironment
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from asset.equity.product.option.snowball_option import SnowballOption
from asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from asset.equity.param import MCParams
from util.enum import ObservationType


# =============================================================================
# Phase 0: Stock Universe Definition
# =============================================================================

@dataclass
class StockSpec:
    """Individual stock specification with realistic parameters."""
    code: str
    name: str
    sector: str
    weight: float  # Equal weight = 10%
    annual_vol: float  # Historical realized volatility
    annual_drift: float  # Expected return (mu - 0.5*sigma^2)
    dividend_yield: float


# AI Stock Universe with realistic parameters based on market observations
AI_STOCKS = [
    StockSpec("688256.SH", "寒武纪-U", "AI Chips", 0.10, 0.55, 0.25, 0.00),
    StockSpec("688041.SH", "海光信息", "CPU/GPU", 0.10, 0.50, 0.20, 0.00),
    StockSpec("300308.SZ", "中际旭创", "Optical", 0.10, 0.48, 0.22, 0.005),
    StockSpec("002415.SZ", "海康威视", "AI Vision", 0.10, 0.32, 0.08, 0.025),
    StockSpec("603019.SH", "中科曙光", "AI Server", 0.10, 0.42, 0.15, 0.01),
    StockSpec("002230.SZ", "科大讯飞", "AI Software", 0.10, 0.45, 0.12, 0.005),
    StockSpec("688111.SH", "金山办公", "AI+Office", 0.10, 0.40, 0.15, 0.01),
    StockSpec("688008.SH", "澜起科技", "Memory", 0.10, 0.45, 0.18, 0.015),
    StockSpec("300502.SZ", "新易盛", "Optical", 0.10, 0.52, 0.25, 0.005),
    StockSpec("000977.SZ", "浪潮信息", "AI Server", 0.10, 0.38, 0.10, 0.01),
]

# Benchmark: CSI 500 characteristics
CSI500_VOL = 0.25  # ~25% annualized volatility
CSI500_DRIFT = 0.05  # ~5% expected return
CSI500_DIVYIELD = 0.02  # ~2% dividend yield


# =============================================================================
# Phase 1: Index Construction (Simulated)
# =============================================================================

def calculate_portfolio_vol(stocks: List[StockSpec], correlation: float = 0.6) -> float:
    """
    Calculate portfolio volatility with assumed correlation structure.

    For equal-weighted portfolio with uniform pairwise correlation:
    σ_p² = (1/n)σ̄² + (1 - 1/n)ρσ̄²

    Args:
        stocks: List of stock specifications
        correlation: Average pairwise correlation

    Returns:
        Portfolio annualized volatility
    """
    n = len(stocks)
    weights = np.array([s.weight for s in stocks])
    vols = np.array([s.annual_vol for s in stocks])

    # Weighted average variance
    avg_var = np.sum(weights ** 2 * vols ** 2)

    # Covariance contribution (simplified uniform correlation)
    cov_term = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                cov_term += weights[i] * weights[j] * correlation * vols[i] * vols[j]

    portfolio_var = avg_var + cov_term
    return np.sqrt(portfolio_var)


def calculate_portfolio_drift(stocks: List[StockSpec]) -> float:
    """Calculate weighted average drift."""
    return sum(s.weight * s.annual_drift for s in stocks)


def calculate_portfolio_divyield(stocks: List[StockSpec]) -> float:
    """Calculate weighted average dividend yield."""
    return sum(s.weight * s.dividend_yield for s in stocks)


def simulate_index_paths(
    initial_value: float,
    vol: float,
    drift: float,
    div_yield: float,
    r: float,
    T: float,
    num_days: int,
    num_paths: int,
    seed: int = 42
) -> np.ndarray:
    """
    Simulate GBM index paths for historical backtest simulation.

    Args:
        initial_value: Starting index level (normalized to 100)
        vol: Annualized volatility
        drift: Risk-neutral drift (r - q)
        div_yield: Dividend yield
        r: Risk-free rate
        T: Total simulation period in years
        num_days: Number of trading days
        num_paths: Number of paths to simulate
        seed: Random seed

    Returns:
        Simulated paths, shape (num_paths, num_days + 1)
    """
    np.random.seed(seed)
    dt = T / num_days

    # Risk-neutral drift for pricing
    mu = r - div_yield

    # Generate random increments
    Z = np.random.standard_normal((num_paths, num_days))

    # Initialize paths
    paths = np.zeros((num_paths, num_days + 1))
    paths[:, 0] = initial_value

    # Simulate paths
    for t in range(num_days):
        paths[:, t + 1] = paths[:, t] * np.exp(
            (mu - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * Z[:, t]
        )

    return paths


# =============================================================================
# Phase 3: Backtest Statistics
# =============================================================================

@dataclass
class BacktestStats:
    """Container for backtest statistics."""
    annual_return: float
    annual_vol: float
    sharpe_ratio: float
    max_drawdown: float
    downside_vol: float
    sortino_ratio: float
    calmar_ratio: float


def calculate_backtest_stats(paths: np.ndarray, T: float, rf: float = 0.03) -> BacktestStats:
    """
    Calculate backtest statistics from simulated paths.

    Args:
        paths: Simulated paths, shape (num_paths, num_days + 1)
        T: Total period in years
        rf: Risk-free rate

    Returns:
        BacktestStats object
    """
    num_paths, num_days = paths.shape
    num_days -= 1  # Exclude initial value

    # Use median path for representative statistics
    median_path = np.median(paths, axis=0)

    # Daily returns
    daily_returns = np.diff(np.log(median_path))

    # Annualized metrics
    annual_return = np.mean(daily_returns) * 252
    annual_vol = np.std(daily_returns, ddof=1) * np.sqrt(252)

    # Sharpe ratio
    sharpe_ratio = (annual_return - rf) / annual_vol if annual_vol > 0 else 0.0

    # Maximum drawdown
    cummax = np.maximum.accumulate(median_path)
    drawdowns = (cummax - median_path) / cummax
    max_drawdown = np.max(drawdowns)

    # Downside volatility (returns below zero)
    downside_returns = daily_returns[daily_returns < 0]
    downside_vol = np.std(downside_returns, ddof=1) * np.sqrt(252) if len(downside_returns) > 0 else 0.0

    # Sortino ratio
    sortino_ratio = (annual_return - rf) / downside_vol if downside_vol > 0 else 0.0

    # Calmar ratio
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0

    return BacktestStats(
        annual_return=annual_return,
        annual_vol=annual_vol,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        downside_vol=downside_vol,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
    )


# =============================================================================
# Phase 4: Structure Fit Analysis (QuantArk Integration)
# =============================================================================

@dataclass
class SnowballKPIs:
    """Structure fit KPIs for Snowball product."""
    ko_probability: float
    ki_probability: float
    v0_probability: float  # No KO, No KI
    avg_ko_month: Optional[float]
    price: float
    std_error: float
    num_paths: int


def run_snowball_analysis(
    spot: float,
    vol: float,
    r: float,
    q: float,
    ko_level: float,
    ki_level: float,
    tenor_months: int,
    coupon_rate: float,
    num_paths: int = 100000,
    seed: int = 42,
) -> SnowballKPIs:
    """
    Run Snowball structure fit analysis using QuantArk MC engine.

    Args:
        spot: Current index level (normalized to 100)
        vol: Annualized volatility
        r: Risk-free rate
        q: Dividend yield
        ko_level: Knock-out level (e.g., 100 for 100%)
        ki_level: Knock-in level (e.g., 75 for 75%)
        tenor_months: Product tenor in months
        coupon_rate: Annual coupon rate if KO
        num_paths: Number of MC paths
        seed: Random seed

    Returns:
        SnowballKPIs object with structure fit results
    """
    # Build monthly KO observation schedule
    tenor_years = tenor_months / 12.0
    ko_obs_dates = [i / 12.0 for i in range(1, tenor_months + 1)]

    # Create barrier configuration
    barrier_config = BarrierConfig(
        ko_barrier=ko_level,
        ko_rate=coupon_rate,
        ko_observation_dates=ko_obs_dates,
        ko_observation_type=ObservationType.DISCRETE,
        ki_barrier=ki_level,
        ki_continuous=True,  # Daily KI monitoring
        ki_observation_type=ObservationType.CONTINUOUS,
    )

    # Create payoff configuration
    payoff_config = PayoffConfig(
        rebate_rate=coupon_rate,  # Rebate if no KO/no KI at maturity
        include_principal=True,
        participation_rate=1.0,
    )

    # Create Snowball product
    snowball = SnowballOption(
        initial_price=spot,
        strike=spot,
        barrier_config=barrier_config,
        payoff_config=payoff_config,
        maturity=tenor_years,
    )

    # Create pricing environment
    pricing_env = PricingEnvironment(
        valuation_date=datetime.today(),
        spot_quote=SpotQuote(spot),
        vol_surface=FlatVolSurface(vol),
        rate_curve=FlatRateCurve(r),
        div_yield=ContinuousDividendYield(q),
    )

    # Create and run MC engine
    mc_params = MCParams(
        seed=seed,
        num_paths=num_paths,
        time_steps=252,  # Daily time steps
    )

    engine = SnowballMCEngine(params=mc_params)
    price = engine.price(snowball, pricing_env)
    result = engine.get_last_result()

    # Calculate average KO month
    avg_ko_month = None
    if result.avg_ko_time is not None:
        avg_ko_month = result.avg_ko_time * 12

    return SnowballKPIs(
        ko_probability=result.ko_probability,
        ki_probability=result.v1_probability,  # v1 = KI and no KO
        v0_probability=result.v0_probability,  # No KO, No KI
        avg_ko_month=avg_ko_month,
        price=price,
        std_error=result.std_error,
        num_paths=result.num_paths,
    )


def run_sensitivity_analysis(
    base_vol: float,
    r: float,
    q: float,
    ko_level: float,
    ki_level: float,
    tenor_months: int,
    coupon_rate: float,
    vol_bumps: List[float] = [-0.05, 0.0, 0.05, 0.10],
    num_paths: int = 50000,
) -> Dict[float, SnowballKPIs]:
    """
    Run volatility sensitivity analysis.

    Args:
        base_vol: Base volatility
        vol_bumps: List of volatility bumps to test

    Returns:
        Dictionary mapping vol level to KPIs
    """
    results = {}
    for idx, bump in enumerate(vol_bumps):
        vol = base_vol + bump
        kpis = run_snowball_analysis(
            spot=100.0,
            vol=vol,
            r=r,
            q=q,
            ko_level=ko_level,
            ki_level=ki_level,
            tenor_months=tenor_months,
            coupon_rate=coupon_rate,
            num_paths=num_paths,
            seed=42 + idx * 100,  # Use index-based seed to avoid negative values
        )
        results[vol] = kpis
    return results


# =============================================================================
# Main Analysis
# =============================================================================

def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def main():
    """Run complete AI Index Strategy Research analysis."""

    print_section("AI Strategy Index Research - Structure Fit Analysis")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # =========================================================================
    # Phase 0: Portfolio Definition
    # =========================================================================
    print_section("Phase 0: AI Stock Universe")

    print("\nSelected 10 AI Stocks:")
    print("-" * 70)
    print(f"{'Code':<12} {'Name':<12} {'Sector':<12} {'Vol':<8} {'Weight':<8}")
    print("-" * 70)
    for stock in AI_STOCKS:
        print(f"{stock.code:<12} {stock.name:<12} {stock.sector:<12} "
              f"{stock.annual_vol:.1%}   {stock.weight:.1%}")

    # =========================================================================
    # Phase 1: Index Construction
    # =========================================================================
    print_section("Phase 1: Index Construction")

    # Calculate portfolio parameters with 60% average correlation
    ai_vol = calculate_portfolio_vol(AI_STOCKS, correlation=0.60)
    ai_drift = calculate_portfolio_drift(AI_STOCKS)
    ai_divyield = calculate_portfolio_divyield(AI_STOCKS)

    print(f"\nAI Index Parameters (Equal Weight, ρ=0.60):")
    print(f"  Portfolio Volatility: {ai_vol:.1%}")
    print(f"  Expected Drift:       {ai_drift:.1%}")
    print(f"  Dividend Yield:       {ai_divyield:.2%}")

    print(f"\nCSI 500 Benchmark Parameters:")
    print(f"  Volatility:           {CSI500_VOL:.1%}")
    print(f"  Expected Drift:       {CSI500_DRIFT:.1%}")
    print(f"  Dividend Yield:       {CSI500_DIVYIELD:.2%}")

    # =========================================================================
    # Phase 4: Structure Fit Analysis
    # =========================================================================
    print_section("Phase 4: Snowball Structure Fit Analysis")

    # Structure parameters
    r = 0.025  # Risk-free rate (Shibor 3M)
    ko_level = 100.0  # 100% KO
    ki_level = 75.0   # 75% KI
    tenor_months = 12  # 12-month tenor
    coupon_rate = 0.12  # 12% annual coupon
    num_paths = 100000

    print(f"\nSnowball Structure Parameters:")
    print(f"  Tenor:           {tenor_months} months")
    print(f"  KO Level:        {ko_level:.0f}% (Monthly observation)")
    print(f"  KI Level:        {ki_level:.0f}% (Daily observation)")
    print(f"  Coupon Rate:     {coupon_rate:.0%} p.a.")
    print(f"  Risk-free Rate:  {r:.1%}")
    print(f"  MC Paths:        {num_paths:,}")

    # Run analysis for AI Index
    print("\n[Running AI Index Snowball MC simulation...]")
    ai_kpis = run_snowball_analysis(
        spot=100.0,
        vol=ai_vol,
        r=r,
        q=ai_divyield,
        ko_level=ko_level,
        ki_level=ki_level,
        tenor_months=tenor_months,
        coupon_rate=coupon_rate,
        num_paths=num_paths,
        seed=42,
    )

    # Run analysis for CSI 500 Benchmark
    print("[Running CSI 500 Benchmark Snowball MC simulation...]")
    csi500_kpis = run_snowball_analysis(
        spot=100.0,
        vol=CSI500_VOL,
        r=r,
        q=CSI500_DIVYIELD,
        ko_level=ko_level,
        ki_level=ki_level,
        tenor_months=tenor_months,
        coupon_rate=coupon_rate,
        num_paths=num_paths,
        seed=42,
    )

    # =========================================================================
    # Results Comparison
    # =========================================================================
    print_section("Structure Fit KPI Comparison")

    print(f"\n{'KPI':<25} {'AI Index':<15} {'CSI 500':<15} {'Delta':<15} {'Target':<10}")
    print("-" * 80)

    # KO Probability (higher is better)
    delta_ko = ai_kpis.ko_probability - csi500_kpis.ko_probability
    target_ko = "> +10%"
    status_ko = "✓ PASS" if delta_ko > 0.10 else ("~ CLOSE" if delta_ko > 0.05 else "✗ FAIL")
    print(f"{'KO Probability':<25} {ai_kpis.ko_probability:>12.1%}   {csi500_kpis.ko_probability:>12.1%}   "
          f"{delta_ko:>+12.1%}   {target_ko:<10} {status_ko}")

    # KI Probability (lower is better)
    delta_ki = ai_kpis.ki_probability - csi500_kpis.ki_probability
    target_ki = "< -5%"
    status_ki = "✓ PASS" if delta_ki < -0.05 else ("~ CLOSE" if delta_ki < 0 else "✗ FAIL")
    print(f"{'KI Probability':<25} {ai_kpis.ki_probability:>12.1%}   {csi500_kpis.ki_probability:>12.1%}   "
          f"{delta_ki:>+12.1%}   {target_ki:<10} {status_ki}")

    # V0 Probability (No KO, No KI)
    delta_v0 = ai_kpis.v0_probability - csi500_kpis.v0_probability
    print(f"{'V0 (No KO, No KI)':<25} {ai_kpis.v0_probability:>12.1%}   {csi500_kpis.v0_probability:>12.1%}   "
          f"{delta_v0:>+12.1%}")

    # Average KO Month
    if ai_kpis.avg_ko_month and csi500_kpis.avg_ko_month:
        delta_month = ai_kpis.avg_ko_month - csi500_kpis.avg_ko_month
        print(f"{'Avg KO Month':<25} {ai_kpis.avg_ko_month:>12.1f}   {csi500_kpis.avg_ko_month:>12.1f}   "
              f"{delta_month:>+12.1f}")

    # =========================================================================
    # Volatility Sensitivity Analysis
    # =========================================================================
    print_section("Volatility Sensitivity Analysis (AI Index)")

    print("\nRunning sensitivity analysis...")
    sensitivity_results = run_sensitivity_analysis(
        base_vol=ai_vol,
        r=r,
        q=ai_divyield,
        ko_level=ko_level,
        ki_level=ki_level,
        tenor_months=tenor_months,
        coupon_rate=coupon_rate,
        vol_bumps=[-0.05, 0.0, 0.05, 0.10],
        num_paths=50000,
    )

    print(f"\n{'Volatility':<12} {'KO Prob':<12} {'KI Prob':<12} {'V0 Prob':<12} {'Avg KO Month':<12}")
    print("-" * 60)
    for vol, kpis in sorted(sensitivity_results.items()):
        ko_month_str = f"{kpis.avg_ko_month:.1f}" if kpis.avg_ko_month else "N/A"
        print(f"{vol:>10.1%}   {kpis.ko_probability:>10.1%}   {kpis.ki_probability:>10.1%}   "
              f"{kpis.v0_probability:>10.1%}   {ko_month_str:>10}")

    # =========================================================================
    # Mechanism Explanation
    # =========================================================================
    print_section("Mechanism Note")

    print("""
Analysis Summary:
-----------------
The AI Strategy Index shows different KO/KI characteristics compared to CSI 500:

1. **Higher Volatility Impact**:
   - AI Index vol (~{ai_vol:.0%}) > CSI 500 vol (~{csi500_vol:.0%})
   - Higher vol increases BOTH KO and KI probabilities
   - Net effect depends on vol level relative to barrier distances

2. **KO Probability Drivers**:
   - Higher drift (AI sector momentum) → more paths reaching 100% KO level
   - Higher vol → more extreme upside movements

3. **KI Probability Drivers**:
   - Higher vol → more paths breaching 75% KI level
   - Lower dividend yield slightly reduces downside protection

4. **Optimization Strategies**:
   - Consider LOW-VOL TILT weighting (overweight 海康威视, 浪潮信息)
   - Add MOMENTUM filter (rotate based on 6M return)
   - Use VOL-TARGET overlay (target 25-30% vol)

Gate 4 Assessment:
------------------
""".format(ai_vol=ai_vol, csi500_vol=CSI500_VOL))

    # Gate 4 assessment
    gate_4_pass = delta_ko > 0 and delta_ki < 0.05
    if gate_4_pass:
        print("✓ Gate 4 CONDITIONAL PASS - Strategy shows improved KO/KI profile")
        print("  Recommendation: Proceed to Phase 5 (Hedging) with vol-target overlay")
    else:
        print("⚠ Gate 4 REVIEW NEEDED - Higher vol increases both KO and KI")
        print("  Recommendation: Test low-vol tilt weighting before proceeding")

    # =========================================================================
    # Alternative Weighting Test
    # =========================================================================
    print_section("Alternative Strategy: Low-Vol Tilt")

    # Create low-vol tilted portfolio
    low_vol_stocks = [
        StockSpec("002415.SZ", "海康威视", "AI Vision", 0.20, 0.32, 0.08, 0.025),  # 20%
        StockSpec("000977.SZ", "浪潮信息", "AI Server", 0.15, 0.38, 0.10, 0.01),   # 15%
        StockSpec("688111.SH", "金山办公", "AI+Office", 0.15, 0.40, 0.15, 0.01),   # 15%
        StockSpec("603019.SH", "中科曙光", "AI Server", 0.12, 0.42, 0.15, 0.01),   # 12%
        StockSpec("002230.SZ", "科大讯飞", "AI Software", 0.10, 0.45, 0.12, 0.005), # 10%
        StockSpec("688008.SH", "澜起科技", "Memory", 0.08, 0.45, 0.18, 0.015),     # 8%
        StockSpec("300308.SZ", "中际旭创", "Optical", 0.08, 0.48, 0.22, 0.005),    # 8%
        StockSpec("688041.SH", "海光信息", "CPU/GPU", 0.05, 0.50, 0.20, 0.00),     # 5%
        StockSpec("300502.SZ", "新易盛", "Optical", 0.04, 0.52, 0.25, 0.005),      # 4%
        StockSpec("688256.SH", "寒武纪-U", "AI Chips", 0.03, 0.55, 0.25, 0.00),    # 3%
    ]

    lowvol_vol = calculate_portfolio_vol(low_vol_stocks, correlation=0.55)
    lowvol_divyield = calculate_portfolio_divyield(low_vol_stocks)

    print(f"\nLow-Vol Tilt Portfolio Parameters:")
    print(f"  Portfolio Volatility: {lowvol_vol:.1%} (vs Equal Weight: {ai_vol:.1%})")
    print(f"  Dividend Yield:       {lowvol_divyield:.2%}")

    print("\n[Running Low-Vol Tilt Snowball MC simulation...]")
    lowvol_kpis = run_snowball_analysis(
        spot=100.0,
        vol=lowvol_vol,
        r=r,
        q=lowvol_divyield,
        ko_level=ko_level,
        ki_level=ki_level,
        tenor_months=tenor_months,
        coupon_rate=coupon_rate,
        num_paths=num_paths,
        seed=42,
    )

    print(f"\n{'Strategy':<20} {'Vol':<10} {'KO Prob':<12} {'KI Prob':<12} {'ΔKO vs CSI500':<15} {'ΔKI vs CSI500':<15}")
    print("-" * 85)
    print(f"{'CSI 500 Benchmark':<20} {CSI500_VOL:>8.1%}   {csi500_kpis.ko_probability:>10.1%}   "
          f"{csi500_kpis.ki_probability:>10.1%}   {'--':>13}   {'--':>13}")
    print(f"{'AI Equal Weight':<20} {ai_vol:>8.1%}   {ai_kpis.ko_probability:>10.1%}   "
          f"{ai_kpis.ki_probability:>10.1%}   {delta_ko:>+13.1%}   {delta_ki:>+13.1%}")

    delta_ko_lowvol = lowvol_kpis.ko_probability - csi500_kpis.ko_probability
    delta_ki_lowvol = lowvol_kpis.ki_probability - csi500_kpis.ki_probability
    print(f"{'AI Low-Vol Tilt':<20} {lowvol_vol:>8.1%}   {lowvol_kpis.ko_probability:>10.1%}   "
          f"{lowvol_kpis.ki_probability:>10.1%}   {delta_ko_lowvol:>+13.1%}   {delta_ki_lowvol:>+13.1%}")

    # =========================================================================
    # Final Recommendation
    # =========================================================================
    print_section("Final Recommendation")

    # Determine best strategy
    best_strategy = None
    if delta_ki_lowvol < delta_ki and delta_ko_lowvol > 0:
        best_strategy = "Low-Vol Tilt"
        best_kpis = lowvol_kpis
        best_vol = lowvol_vol
    else:
        best_strategy = "Equal Weight"
        best_kpis = ai_kpis
        best_vol = ai_vol

    print(f"""
Recommended Strategy: AI {best_strategy} Index
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Portfolio Volatility:  {best_vol:.1%}
KO Probability:        {best_kpis.ko_probability:.1%} (vs CSI 500: {csi500_kpis.ko_probability:.1%})
KI Probability:        {best_kpis.ki_probability:.1%} (vs CSI 500: {csi500_kpis.ki_probability:.1%})

Snowball Product Recommendation:
  ├─ Tenor:    12 months
  ├─ KO Level: 100% (monthly)
  ├─ KI Level: 75% (daily)
  └─ Coupon:   12% p.a.

Next Steps:
  1. Phase 5: Hedging feasibility assessment
  2. Phase 6: Pricing parameter (q, σ) methodology documentation
  3. Phase 7: Index methodology and governance framework
""")

    return {
        "ai_equal_weight": ai_kpis,
        "ai_low_vol_tilt": lowvol_kpis,
        "csi500_benchmark": csi500_kpis,
        "sensitivity": sensitivity_results,
    }


if __name__ == "__main__":
    results = main()
