"""
Credit portfolio risk capstone demo.

Books a small single-name CDS portfolio and runs it end-to-end through the full
risk stack that the credit migration wired up:

    stress test  ->  VaR  ->  dynamic scenario  ->  backtest  ->  SIMM margin

Run:  python example/credit_portfolio_risk_demo.py
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from quantark.asset.credit.engine.analytical import CDSReducedFormEngine
from quantark.asset.credit.product import CDS, ProtectionSide
from quantark.backtest import CreditBacktestConfig, CreditBacktestEngine
from quantark.backtest.strategy import CreditSpreadNeutralStrategy
from quantark.dynamicscenario import CreditDynamicScenarioEngine, CreditPathLibrary
from quantark.param import FlatRateCurve
from quantark.param.credit import FlatHazardCurve
from quantark.portfolio import CreditPortfolio
from quantark.priceenv import CreditPricingEnvironment
from quantark.simm import SIMMConfig, SIMMVersion
from quantark.simm.engines.aggregation import SIMMCalculator
from quantark.simm.engines.portfolio_adapter import SIMMPortfolioAdapter
from quantark.stresstest import CreditStressEngine
from quantark.stresstest.scenario.scenario import Scenario, Stress
from quantark.stresstest.stress.stress_types import StressLevel, StressType
from quantark.var import CreditParametricVaREngine
from quantark.var.config import VaRConfig


def build_book() -> CreditPortfolio:
    envs = {
        "JPMORGAN": CreditPricingEnvironment(
            valuation_date=datetime(2026, 6, 13),
            discount_curve=FlatRateCurve(rate=0.03),
            hazard_curve=FlatHazardCurve(hazard_rate=0.015),
        ),
        "WALMART": CreditPricingEnvironment(
            valuation_date=datetime(2026, 6, 13),
            discount_curve=FlatRateCurve(rate=0.03),
            hazard_curve=FlatHazardCurve(hazard_rate=0.025),
        ),
    }
    pf = CreditPortfolio(portfolio_name="CreditBook", pricing_environments=envs)
    eng = CDSReducedFormEngine()
    pf.add_position(product=CDS(notional=10_000_000, maturity=5.0, recovery_rate=0.4,
                                coupon_spread=0.01, side=ProtectionSide.BUY),
                    quantity=1.0, entry_price=0.0, reference_entity="JPMORGAN", engine=eng)
    pf.add_position(product=CDS(notional=5_000_000, maturity=3.0, recovery_rate=0.4,
                                coupon_spread=0.012, side=ProtectionSide.SELL),
                    quantity=1.0, entry_price=0.0, reference_entity="WALMART", engine=eng)
    return pf


def section(title: str) -> None:
    print("\n" + "=" * 70 + f"\n  {title}\n" + "=" * 70)


def main() -> int:
    print("\n" + "#" * 70)
    print("#  QUANTARK - Credit Portfolio Risk Capstone")
    print("#" * 70)

    pf = build_book()
    print(f"\nBook value: ${pf.get_portfolio_value():,.2f}")
    greeks = pf.get_portfolio_greeks()
    print(f"Portfolio CS01: ${greeks['cs01']:,.2f}/bp   IR01: ${greeks['ir01']:,.2f}/bp")

    # 1. STRESS ------------------------------------------------------------
    section("1. STRESS TEST - spreads +100%")
    scenario = Scenario(name="Spread doubling", stresses=[
        Stress(parameter="spread", stress_type=StressType.PERCENTAGE,
               stress_value=1.0, level=StressLevel.PORTFOLIO)])
    sres = CreditStressEngine().run_static_scenarios(pf, [scenario])
    r = sres.scenario_results[0]
    print(f"  P&L under spread doubling: ${r.portfolio_pnl:,.2f}  ({r.portfolio_pnl_pct:+.2f}%)")

    # 2. VAR ---------------------------------------------------------------
    section("2. PARAMETRIC VaR (99%, 1-day)")
    rng = np.random.default_rng(7)
    n = 300
    hist = pd.DataFrame({
        "JPMORGAN_hazard": np.abs(0.015 + np.cumsum(rng.normal(0, 0.0004, n))),
        "JPMORGAN_rate": 0.03 + np.cumsum(rng.normal(0, 0.0002, n)),
        "WALMART_hazard": np.abs(0.025 + np.cumsum(rng.normal(0, 0.0006, n))),
        "WALMART_rate": 0.03 + np.cumsum(rng.normal(0, 0.0002, n)),
    })
    var_res = CreditParametricVaREngine(
        VaRConfig(confidence_level=0.99, lookback_days=250, calculate_factor_var=True)
    ).calculate_var(pf, hist)
    print(f"  VaR: ${var_res.var:,.2f}   CVaR: ${var_res.cvar:,.2f}")
    if var_res.factor_var:
        top = max(var_res.factor_var.items(), key=lambda kv: abs(kv[1]))
        print(f"  Largest factor contribution: {top[0]} = ${top[1]:,.2f}")

    # 3. DYNAMIC SCENARIO --------------------------------------------------
    section("3. DYNAMIC SCENARIO - credit crisis path")
    dyn = CreditDynamicScenarioEngine().run(pf, CreditPathLibrary.credit_crisis(days=6))
    print(f"  Path '{dyn.path_name}': total P&L ${dyn.total_pnl:,.2f} over {dyn.num_days} days")
    worst = dyn.get_worst_day()
    if worst is not None:
        print(f"  Worst day: {worst.label} daily P&L ${worst.daily_pnl:,.2f}")

    # 4. BACKTEST ----------------------------------------------------------
    section("4. BACKTEST - CS01-neutral hedging")
    idx = pd.date_range("2026-06-15", periods=60, freq="B")
    path = pd.DataFrame({
        "JPMORGAN_hazard": np.abs(0.015 + np.cumsum(rng.normal(0, 0.0005, 60))),
        "JPMORGAN_rate": 0.03 + np.cumsum(rng.normal(0, 0.0002, 60)),
        "WALMART_hazard": np.abs(0.025 + np.cumsum(rng.normal(0, 0.0007, 60))),
        "WALMART_rate": 0.03 + np.cumsum(rng.normal(0, 0.0002, 60)),
    }, index=idx)
    bt = CreditBacktestEngine(CreditBacktestConfig(
        portfolio=pf, market_path=path,
        strategy=CreditSpreadNeutralStrategy(cs01_threshold=2_000.0))).run()
    eff = bt.get_hedge_effectiveness()
    print(f"  Hedges: {bt.num_hedges}   Net P&L: ${bt.total_net_pnl:,.2f}   "
          f"P&L vol reduction: {eff['vol_reduction_pct']:.1f}%")

    # 5. SIMM --------------------------------------------------------------
    section("5. SIMM v2.6 INITIAL MARGIN (CreditQ)")
    config = SIMMConfig(version=SIMMVersion.V2_6, calculation_currency="USD",
                        calculate_delta=True)
    sens = SIMMPortfolioAdapter(config).portfolio_to_sensitivities(pf)
    margin = SIMMCalculator(config).calculate(sens)
    print(f"  Credit sensitivities generated: {len(sens.sensitivities)}")
    print(f"  Total SIMM initial margin: ${margin.total_margin:,.2f}")

    print("\n" + "#" * 70)
    print("#  Credit risk stack: stress -> VaR -> dynamic -> backtest -> SIMM  OK")
    print("#" * 70 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
