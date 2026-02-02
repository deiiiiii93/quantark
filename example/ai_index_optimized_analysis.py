"""
AI Strategy Index - Optimized Structure Analysis (103% KO / 70% KI)
===================================================================
Testing whether higher KO barrier better differentiates AI index benefits.
"""

import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

# QuantArk imports
from priceenv import PricingEnvironment
from param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from asset.equity.product.option.snowball_option import SnowballOption
from asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from asset.equity.param import MCParams
from util.enum import ObservationType


@dataclass
class StockSpec:
    code: str
    name: str
    sector: str
    weight: float
    annual_vol: float
    annual_drift: float
    dividend_yield: float


# Stock definitions
AI_STOCKS_EQUAL = [
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

AI_STOCKS_LOWVOL = [
    StockSpec("002415.SZ", "海康威视", "AI Vision", 0.20, 0.32, 0.08, 0.025),
    StockSpec("000977.SZ", "浪潮信息", "AI Server", 0.15, 0.38, 0.10, 0.01),
    StockSpec("688111.SH", "金山办公", "AI+Office", 0.15, 0.40, 0.15, 0.01),
    StockSpec("603019.SH", "中科曙光", "AI Server", 0.12, 0.42, 0.15, 0.01),
    StockSpec("002230.SZ", "科大讯飞", "AI Software", 0.10, 0.45, 0.12, 0.005),
    StockSpec("688008.SH", "澜起科技", "Memory", 0.08, 0.45, 0.18, 0.015),
    StockSpec("300308.SZ", "中际旭创", "Optical", 0.08, 0.48, 0.22, 0.005),
    StockSpec("688041.SH", "海光信息", "CPU/GPU", 0.05, 0.50, 0.20, 0.00),
    StockSpec("300502.SZ", "新易盛", "Optical", 0.04, 0.52, 0.25, 0.005),
    StockSpec("688256.SH", "寒武纪-U", "AI Chips", 0.03, 0.55, 0.25, 0.00),
]

CSI500_VOL = 0.25
CSI500_DIVYIELD = 0.02


def calc_portfolio_vol(stocks: List[StockSpec], corr: float = 0.6) -> float:
    n = len(stocks)
    weights = np.array([s.weight for s in stocks])
    vols = np.array([s.annual_vol for s in stocks])
    avg_var = np.sum(weights ** 2 * vols ** 2)
    cov_term = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                cov_term += weights[i] * weights[j] * corr * vols[i] * vols[j]
    return np.sqrt(avg_var + cov_term)


def calc_portfolio_divyield(stocks: List[StockSpec]) -> float:
    return sum(s.weight * s.dividend_yield for s in stocks)


@dataclass
class SnowballKPIs:
    ko_probability: float
    ki_probability: float
    v0_probability: float
    avg_ko_month: Optional[float]
    price: float


def run_snowball(
    vol: float, q: float, r: float,
    ko_level: float, ki_level: float,
    tenor_months: int, coupon_rate: float,
    num_paths: int = 100000, seed: int = 42
) -> SnowballKPIs:
    """Run Snowball MC simulation."""
    tenor_years = tenor_months / 12.0
    ko_obs_dates = [i / 12.0 for i in range(1, tenor_months + 1)]

    barrier_config = BarrierConfig(
        ko_barrier=ko_level,
        ko_rate=coupon_rate,
        ko_observation_dates=ko_obs_dates,
        ko_observation_type=ObservationType.DISCRETE,
        ki_barrier=ki_level,
        ki_continuous=True,
        ki_observation_type=ObservationType.CONTINUOUS,
    )

    payoff_config = PayoffConfig(
        rebate_rate=coupon_rate,
        include_principal=True,
        participation_rate=1.0,
    )

    snowball = SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=barrier_config,
        payoff_config=payoff_config,
        maturity=tenor_years,
    )

    pricing_env = PricingEnvironment(
        valuation_date=datetime.today(),
        spot_quote=SpotQuote(100.0),
        vol_surface=FlatVolSurface(vol),
        rate_curve=FlatRateCurve(r),
        div_yield=ContinuousDividendYield(q),
    )

    engine = SnowballMCEngine(params=MCParams(seed=seed, num_paths=num_paths, time_steps=252))
    price = engine.price(snowball, pricing_env)
    result = engine.get_last_result()

    avg_ko_month = result.avg_ko_time * 12 if result.avg_ko_time else None

    return SnowballKPIs(
        ko_probability=result.ko_probability,
        ki_probability=result.v1_probability,
        v0_probability=result.v0_probability,
        avg_ko_month=avg_ko_month,
        price=price,
    )


def print_section(title: str):
    print("\n" + "=" * 75)
    print(f" {title}")
    print("=" * 75)


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():
    print_section("AI Strategy Index - Optimized Structure Analysis")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Portfolio parameters
    ai_eq_vol = calc_portfolio_vol(AI_STOCKS_EQUAL, corr=0.60)
    ai_eq_q = calc_portfolio_divyield(AI_STOCKS_EQUAL)
    ai_lv_vol = calc_portfolio_vol(AI_STOCKS_LOWVOL, corr=0.55)
    ai_lv_q = calc_portfolio_divyield(AI_STOCKS_LOWVOL)

    print(f"\nPortfolio Parameters:")
    print(f"  AI Equal Weight:  Vol={ai_eq_vol:.1%}, q={ai_eq_q:.2%}")
    print(f"  AI Low-Vol Tilt:  Vol={ai_lv_vol:.1%}, q={ai_lv_q:.2%}")
    print(f"  CSI 500 Benchmark: Vol={CSI500_VOL:.1%}, q={CSI500_DIVYIELD:.2%}")

    # Common parameters
    r = 0.025
    num_paths = 100000

    # =========================================================================
    # TEST 1: Original Structure (100% KO / 75% KI)
    # =========================================================================
    print_section("Test 1: Original Structure (100% KO / 75% KI)")

    configs = [
        ("CSI 500", CSI500_VOL, CSI500_DIVYIELD),
        ("AI Equal Weight", ai_eq_vol, ai_eq_q),
        ("AI Low-Vol Tilt", ai_lv_vol, ai_lv_q),
    ]

    results_orig = {}
    for name, vol, q in configs:
        print(f"  Running {name}...")
        results_orig[name] = run_snowball(vol, q, r, ko_level=100.0, ki_level=75.0,
                                          tenor_months=12, coupon_rate=0.12, seed=42)

    print(f"\n{'Index':<20} {'Vol':>8} {'KO Prob':>10} {'KI Prob':>10} {'ΔKO':>10} {'ΔKI':>10}")
    print("-" * 70)
    bench = results_orig["CSI 500"]
    for name, vol, q in configs:
        kpi = results_orig[name]
        dko = kpi.ko_probability - bench.ko_probability
        dki = kpi.ki_probability - bench.ki_probability
        dko_str = f"{dko:+.1%}" if name != "CSI 500" else "--"
        dki_str = f"{dki:+.1%}" if name != "CSI 500" else "--"
        print(f"{name:<20} {vol:>7.1%} {kpi.ko_probability:>10.1%} {kpi.ki_probability:>10.1%} {dko_str:>10} {dki_str:>10}")

    # =========================================================================
    # TEST 2: Optimized Structure (103% KO / 70% KI)
    # =========================================================================
    print_section("Test 2: Optimized Structure (103% KO / 70% KI)")

    results_opt = {}
    for name, vol, q in configs:
        print(f"  Running {name}...")
        results_opt[name] = run_snowball(vol, q, r, ko_level=103.0, ki_level=70.0,
                                         tenor_months=12, coupon_rate=0.12, seed=42)

    print(f"\n{'Index':<20} {'Vol':>8} {'KO Prob':>10} {'KI Prob':>10} {'ΔKO':>10} {'ΔKI':>10}")
    print("-" * 70)
    bench_opt = results_opt["CSI 500"]
    for name, vol, q in configs:
        kpi = results_opt[name]
        dko = kpi.ko_probability - bench_opt.ko_probability
        dki = kpi.ki_probability - bench_opt.ki_probability
        dko_str = f"{dko:+.1%}" if name != "CSI 500" else "--"
        dki_str = f"{dki:+.1%}" if name != "CSI 500" else "--"
        print(f"{name:<20} {vol:>7.1%} {kpi.ko_probability:>10.1%} {kpi.ki_probability:>10.1%} {dko_str:>10} {dki_str:>10}")

    # =========================================================================
    # TEST 3: Multiple KO/KI Combinations
    # =========================================================================
    print_section("Test 3: Structure Parameter Grid Search")

    structures = [
        (100.0, 75.0, "Standard"),
        (103.0, 75.0, "103/75"),
        (103.0, 70.0, "103/70 (Opt)"),
        (105.0, 70.0, "105/70"),
        (100.0, 70.0, "100/70"),
    ]

    print(f"\n{'Structure':<12} | {'CSI 500':^22} | {'AI Equal Wt':^22} | {'AI Low-Vol':^22}")
    print(f"{'':12} | {'KO':>10} {'KI':>10} | {'KO':>10} {'KI':>10} | {'KO':>10} {'KI':>10}")
    print("-" * 80)

    for ko, ki, label in structures:
        csi = run_snowball(CSI500_VOL, CSI500_DIVYIELD, r, ko, ki, 12, 0.12, num_paths=50000, seed=42)
        ai_eq = run_snowball(ai_eq_vol, ai_eq_q, r, ko, ki, 12, 0.12, num_paths=50000, seed=42)
        ai_lv = run_snowball(ai_lv_vol, ai_lv_q, r, ko, ki, 12, 0.12, num_paths=50000, seed=42)

        print(f"{label:<12} | {csi.ko_probability:>9.1%} {csi.ki_probability:>10.1%} | "
              f"{ai_eq.ko_probability:>9.1%} {ai_eq.ki_probability:>10.1%} | "
              f"{ai_lv.ko_probability:>9.1%} {ai_lv.ki_probability:>10.1%}")

    # =========================================================================
    # TEST 4: Tenor Sensitivity
    # =========================================================================
    print_section("Test 4: Tenor Sensitivity (103% KO / 70% KI)")

    tenors = [12, 18, 24]

    print(f"\n{'Tenor':<8} | {'CSI 500':^22} | {'AI Equal Wt':^22} | {'Delta AI vs CSI':^22}")
    print(f"{'':8} | {'KO':>10} {'KI':>10} | {'KO':>10} {'KI':>10} | {'ΔKO':>10} {'ΔKI':>10}")
    print("-" * 85)

    for tenor in tenors:
        csi = run_snowball(CSI500_VOL, CSI500_DIVYIELD, r, 103.0, 70.0, tenor, 0.12, num_paths=50000, seed=42)
        ai_eq = run_snowball(ai_eq_vol, ai_eq_q, r, 103.0, 70.0, tenor, 0.12, num_paths=50000, seed=42)

        dko = ai_eq.ko_probability - csi.ko_probability
        dki = ai_eq.ki_probability - csi.ki_probability

        print(f"{tenor:>4}M    | {csi.ko_probability:>9.1%} {csi.ki_probability:>10.1%} | "
              f"{ai_eq.ko_probability:>9.1%} {ai_eq.ki_probability:>10.1%} | "
              f"{dko:>+9.1%} {dki:>+10.1%}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_section("FINAL SUMMARY & RECOMMENDATION")

    print("""
KEY FINDINGS:
─────────────
1. With STANDARD structure (100% KO / 75% KI):
   - AI Index shows SIMILAR KO probability but HIGHER KI probability
   - Higher vol doesn't help when KO barrier is at-the-money

2. With OPTIMIZED structure (103% KO / 70% KI):
   - AI Index shows HIGHER KO probability (vol helps reach 103%)
   - KI probability still higher but gap is smaller with 70% KI

3. BEST CONFIGURATION for AI Index:
   - KO Level: 103% (captures high-vol upside benefit)
   - KI Level: 70% (reduces KI risk from high vol)
   - Tenor: 12-18M

STRUCTURE RECOMMENDATION:
────────────────────────""")

    # Calculate best case
    best_csi = run_snowball(CSI500_VOL, CSI500_DIVYIELD, r, 103.0, 70.0, 12, 0.12, num_paths=100000, seed=42)
    best_ai = run_snowball(ai_lv_vol, ai_lv_q, r, 103.0, 70.0, 12, 0.12, num_paths=100000, seed=42)

    dko_best = best_ai.ko_probability - best_csi.ko_probability
    dki_best = best_ai.ki_probability - best_csi.ki_probability

    gate_status = "✓ PASS" if dko_best > 0 else "⚠ REVIEW"

    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  RECOMMENDED: AI Low-Vol Tilt Index + 103% KO / 70% KI Structure    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Index Parameters:                                                  │
│    • Weighting: Low-Vol Tilt (海康威视 20%, 浪潮信息 15%, etc.)         │
│    • Portfolio Vol: {ai_lv_vol:.1%}                                        │
│    • Dividend Yield: {ai_lv_q:.2%}                                        │
│                                                                     │
│  Structure Parameters:                                              │
│    • Tenor: 12 months                                               │
│    • KO Level: 103% (Monthly)                                       │
│    • KI Level: 70% (Daily)                                          │
│    • Coupon: 12% p.a.                                               │
│                                                                     │
│  Expected KPIs:                                                     │
│    • KO Probability: {best_ai.ko_probability:.1%} (CSI 500: {best_csi.ko_probability:.1%}, Δ={dko_best:+.1%})         │
│    • KI Probability: {best_ai.ki_probability:.1%} (CSI 500: {best_csi.ki_probability:.1%}, Δ={dki_best:+.1%})          │
│    • Avg KO Month: {best_ai.avg_ko_month:.1f}                                           │
│                                                                     │
│  Gate 4 Status: {gate_status} (KO improved by {dko_best:+.1%})                          │
└─────────────────────────────────────────────────────────────────────┘
""")

    print("Analysis complete.")


if __name__ == "__main__":
    main()
