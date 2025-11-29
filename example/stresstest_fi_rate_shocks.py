"""
Fixed Income stress testing example using FIStressEngine.
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from asset.bond.engine.discount.bond_discount_engine import BondDiscountEngine
from asset.bond.product.couponbond.fixed_bond import FixedBond
from param.rrf.rate_curve import LinearRateCurve
from portfolio.fi import FIPortfolio
from priceenv import PricingEnvironment
from stresstest.fi import FIStressConfig, FIStressEngine
from stresstest.results.result_exporter import ResultExporter
from stresstest.scenario.scenario_library import ScenarioLibrary
from util.enum import PaymentFrequency
from util.calendar import DayCountConvention


def build_demo_portfolio() -> FIPortfolio:
    valuation_date = datetime(2025, 1, 2)
    curve_pillars = [(2.0, 0.03), (5.0, 0.032), (10.0, 0.035), (30.0, 0.04)]
    envs = {
        "UST_10Y": PricingEnvironment(
            rate_curve=LinearRateCurve(curve_pillars),
            valuation_date=valuation_date,
        ),
        "UST_30Y": PricingEnvironment(
            rate_curve=LinearRateCurve(curve_pillars),
            valuation_date=valuation_date,
        ),
    }

    portfolio = FIPortfolio("Demo FI Book", pricing_environments=envs)

    ten_year = FixedBond(
        issue_date=datetime(2020, 1, 1),
        maturity_date=datetime(2030, 1, 1),
        notional=100.0,
        coupon_rate=0.04,
        payment_frequency=PaymentFrequency.SEMI_ANNUAL,
        day_count_convention=DayCountConvention.ACT_365,
    )
    portfolio.add_position(
        product=ten_year,
        quantity=50,
        entry_price=99.25,
        underlying="UST_10Y",
        engine=BondDiscountEngine(envs["UST_10Y"]),
        entry_timestamp=valuation_date,
    )

    thirty_year = FixedBond(
        issue_date=datetime(2015, 1, 1),
        maturity_date=datetime(2045, 1, 1),
        notional=100.0,
        coupon_rate=0.05,
        payment_frequency=PaymentFrequency.SEMI_ANNUAL,
        day_count_convention=DayCountConvention.ACT_365,
    )
    portfolio.add_position(
        product=thirty_year,
        quantity=30,
        entry_price=101.10,
        underlying="UST_30Y",
        engine=BondDiscountEngine(envs["UST_30Y"]),
        entry_timestamp=valuation_date,
    )

    return portfolio


def main() -> None:
    portfolio = build_demo_portfolio()
    scenarios = [
        ScenarioLibrary.fi_parallel_shift(0.01),
        ScenarioLibrary.fi_steepener(front_end_bps=0.015, long_end_bps=-0.005),
        ScenarioLibrary.fi_spread_shock(spread_bps=0.0025, curve_name="IG"),
    ]

    config = FIStressConfig(save_detailed_results=True)
    engine = FIStressEngine(config)
    results = engine.run_static_scenarios(portfolio, scenarios)

    print(results.get_summary())
    print("\nDV01 Series:")
    print(results.get_dv01_series())

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    ResultExporter.export_to_csv(
        results,
        output_dir / "fi_rate_shocks",
        include_positions=False,
    )


if __name__ == "__main__":
    main()
