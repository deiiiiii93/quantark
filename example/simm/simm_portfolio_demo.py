"""
SIMM portfolio demo: equity + fixed income portfolios → ISDA SIMM v2.6.

Demonstrates the full quant-ark portfolio-to-SIMM workflow:
  1. Build equity portfolios (stocks and options across multiple buckets)
  2. Build fixed income portfolios (bonds across multiple currencies)
  3. Convert portfolios to SIMM sensitivities via SIMMPortfolioAdapter
  4. Compute SIMM initial margin via SIMMCalculator
  5. Render an interactive dashboard

Usage:
    python example/simm_portfolio_demo.py
    # Generates simm_portfolio_dashboard.html in example/
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantark.param import SpotQuote, FlatVolSurface, FlatRateCurve, ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.asset.bond.product.couponbond.fixed_bond import FixedBond
from quantark.asset.bond.engine.discount import BondDiscountEngine
from quantark.portfolio import Portfolio as EquityPortfolio
from quantark.portfolio.fi import FIPortfolio
from quantark.util.enum import OptionType, PaymentFrequency
from quantark.util.calendar import (
    BusinessDayConvention,
    CalendarType,
    DayCountConvention,
    create_calendar,
)
from quantark.simm.config import SIMMConfig, SIMMVersion
from quantark.simm.engines.portfolio_adapter import SIMMPortfolioAdapter
from quantark.simm.engines.aggregation import SIMMCalculator
from quantark.simm.taxonomy import ProductClass, RiskClass, MarginType

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
VALUATION_DATE = datetime(2024, 6, 30)
RATE_USD = 0.0525  # 5.25%
RATE_EUR = 0.0375  # 3.75%
RATE_GBP = 0.0500  # 5.00%
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DASHBOARD = OUTPUT_DIR / "simm_portfolio_dashboard.html"


def print_section(title: str) -> None:
    print(f"\n{'─' * 76}")
    print(f"  {title}")
    print(f"{'─' * 76}")


# ---------------------------------------------------------------------------
# 1. BUILD EQUITY PORTFOLIO
# ---------------------------------------------------------------------------
def build_equity_portfolio() -> EquityPortfolio:
    """Build an equity portfolio with stocks and options across SIMM buckets.

    Bucket assignments (via BucketMapper built-in mappings):
        5 → AAPL / MSFT / GOOGL  (Technology)
        7 → JPMORGAN             (Financials)
        8 → WALMART              (Consumer)
        11 → SPY                 (Indices / ETFs)
    """
    bs_engine = BlackScholesEngine()
    greeks_calc = GreeksCalculator()

    # --- pricing environments per underlying ---
    envs: dict[str, PricingEnvironment] = {}
    spots = {
        "AAPL": 210.0, "MSFT": 445.0, "GOOGL": 180.0,
        "JPMORGAN": 202.0, "WALMART": 68.0, "SPY": 545.0,
    }
    vols = {"AAPL": 0.27, "MSFT": 0.25, "GOOGL": 0.28,
            "JPMORGAN": 0.22, "WALMART": 0.18, "SPY": 0.16}
    divs = {"AAPL": 0.005, "MSFT": 0.008, "GOOGL": 0.0,
            "JPMORGAN": 0.025, "WALMART": 0.015, "SPY": 0.014}

    for name in spots:
        envs[name] = PricingEnvironment(
            spot_quote=SpotQuote(spot=spots[name], asset_name=name),
            vol_surface=FlatVolSurface(volatility=vols[name]),
            rate_curve=FlatRateCurve(rate=RATE_USD),
            div_yield=ContinuousDividendYield(div_yield=divs[name]),
            valuation_date=VALUATION_DATE,
        )

    port = EquityPortfolio(
        portfolio_name="Equity Options Portfolio",
        pricing_environments=envs,
        creation_date=VALUATION_DATE,
    )

    # --- positions ---
    def add_opt(underlying: str, strike: float, op_type: OptionType,
                maturity: float, quantity: float) -> None:
        product = EuropeanVanillaOption(strike=strike, option_type=op_type, maturity=maturity)
        env = envs[underlying]
        entry = bs_engine.price(product, env)
        port.add_position(
            product=product, quantity=quantity, entry_price=entry,
            underlying=underlying, engine=bs_engine, entry_timestamp=VALUATION_DATE,
        )

    # Technology (bucket 5) — directional and hedged positions
    add_opt("AAPL", 210.0, OptionType.CALL, 0.50, 20)   # long ATM call
    add_opt("AAPL", 195.0, OptionType.PUT, 0.50, -5)    # short OTM put
    add_opt("MSFT", 445.0, OptionType.CALL, 0.25, 15)   # long ATM call
    add_opt("MSFT", 460.0, OptionType.CALL, 0.75, 10)   # long OTM call (more vega)
    add_opt("GOOGL", 180.0, OptionType.PUT, 0.50, 8)    # long ATM put

    # Financials (bucket 7)
    add_opt("JPMORGAN", 202.0, OptionType.CALL, 1.00, 12)  # long ATM call

    # Consumer (bucket 8)
    add_opt("WALMART", 68.0, OptionType.CALL, 0.50, -10)   # short ATM call

    # Indices / ETFs (bucket 11) — short index hedge
    add_opt("SPY", 545.0, OptionType.PUT, 0.25, -8)   # short index put (hedge)

    return port


# ---------------------------------------------------------------------------
# 2. BUILD FIXED INCOME PORTFOLIO
# ---------------------------------------------------------------------------
def build_fi_portfolio() -> FIPortfolio:
    """Build an FI portfolio of government bonds across currencies.

    IR bucket = currency (USD, EUR, GBP).  Currency volatility groups
    per SIMM v2.6 paragraph 33:
        Low-vol → USD, EUR, GBP
    """
    # --- pricing environments (rate curves only needed) ---
    fi_envs = {
        "USD": PricingEnvironment(
            rate_curve=FlatRateCurve(rate=RATE_USD), valuation_date=VALUATION_DATE,
        ),
        "EUR": PricingEnvironment(
            rate_curve=FlatRateCurve(rate=RATE_EUR), valuation_date=VALUATION_DATE,
        ),
        "GBP": PricingEnvironment(
            rate_curve=FlatRateCurve(rate=RATE_GBP), valuation_date=VALUATION_DATE,
        ),
    }

    port = FIPortfolio(
        portfolio_name="Fixed Income Portfolio",
        pricing_environments=fi_envs,
    )

    def add_bond(ccy: str, maturity_date: datetime, coupon: float,
                 quantity: float, face: float = 100.0) -> None:
        bond = FixedBond(
            issue_date=datetime(2021, 6, 30),
            maturity_date=maturity_date,
            denominator=face,
            coupon_rate=coupon,
            payment_frequency=PaymentFrequency.SEMI_ANNUAL,
            day_count_convention=DayCountConvention.ACT_ACT_ISDA,
            calendar=create_calendar(CalendarType.NONE),
            business_day_convention=BusinessDayConvention.UNADJUSTED,
            settlement_days=0,
        )
        engine = BondDiscountEngine(pricing_env=fi_envs[ccy])
        entry = engine.clean_price(bond, VALUATION_DATE, VALUATION_DATE)
        port.add_position(
            product=bond, quantity=quantity, entry_price=entry,
            underlying=ccy, engine=engine, notional_per_unit=face,
            entry_timestamp=VALUATION_DATE,
        )

    # USD bonds — receiver (DV01 positive → long rates down)
    add_bond("USD", datetime(2026, 12, 31), 0.0400, 500)   # 2.5y 4% bond
    add_bond("USD", datetime(2028, 12, 31), 0.0375, 300)   # 4.5y 3.75% bond
    add_bond("USD", datetime(2034,  6, 30), 0.0425, 200)   # 10y 4.25% bond

    # EUR bonds — payer (DV01 negative → short rates down)
    add_bond("EUR", datetime(2027,  6, 30), 0.0300, -400)  # 3y 3% bond
    add_bond("EUR", datetime(2031,  6, 30), 0.0325, -250)  # 7y 3.25% bond

    # GBP bonds — receiver
    add_bond("GBP", datetime(2029,  6, 30), 0.0475, 150)   # 5y 4.75% bond

    return port


# ---------------------------------------------------------------------------
# 3. CONVERT PORTFOLIOS → SIMM SENSITIVITIES
# ---------------------------------------------------------------------------
def convert_to_sensitivities(
    eq_port: EquityPortfolio, fi_port: FIPortfolio, config: SIMMConfig,
) -> tuple:
    """Run SIMMPortfolioAdapter on each portfolio and return collections."""
    adapter_eq = SIMMPortfolioAdapter(config)
    adapter_fi = SIMMPortfolioAdapter(config)

    print("  Converting equity portfolio …")
    eq_sens = adapter_eq.portfolio_to_sensitivities(eq_port)
    print(f"    ✓ {len(eq_sens.sensitivities)} equity sensitivities")

    print("  Converting fixed-income portfolio …")
    fi_sens = adapter_fi.portfolio_to_sensitivities(fi_port)
    print(f"    ✓ {len(fi_sens.sensitivities)} FI sensitivities")

    return eq_sens, fi_sens


# ---------------------------------------------------------------------------
# 4. RUN SIMM
# ---------------------------------------------------------------------------
def run_simm(
    eq_sens, fi_sens, config: SIMMConfig,
) -> tuple:
    """Compute SIMM on the combined sensitivity collection."""
    from quantark.simm.sensitivity import SensitivityCollection

    combined = SensitivityCollection()
    combined.add_many(eq_sens.sensitivities)
    combined.add_many(fi_sens.sensitivities)
    print(f"  Combined: {len(combined.sensitivities)} total sensitivities")

    calc = SIMMCalculator(config)
    result = calc.calculate(combined)

    print(f"\n  Total SIMM: ${result.total_margin:,.2f}  {config.calculation_currency}")
    for pc in ProductClass:
        m = result.by_product_class.get(pc, 0.0)
        if m:
            print(f"    {pc.value:>12s}  ${m:>14,.2f}")

    return result


# ---------------------------------------------------------------------------
# 5. DASHBOARD
# ---------------------------------------------------------------------------
def render_dashboard(result, eq_port, fi_port) -> None:
    """Generate an interactive SIMM dashboard HTML report."""
    from quantark.simm.dashboard import SIMMDashboard

    db = SIMMDashboard(
        result=result,
        equity_portfolio=eq_port,
        fi_portfolio=fi_port,
    )
    db.generate(str(OUTPUT_DASHBOARD))
    print(f"\n  Dashboard written to: {OUTPUT_DASHBOARD}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> int:
    print("\n" + "█" * 78)
    print("█  QUANTARK — SIMM Portfolio Demo (ISDA SIMM v2.6)")
    print("█" * 78)

    # --- configuration ---
    config = SIMMConfig(
        version=SIMMVersion.V2_6,
        calculation_currency="USD",
        calculate_delta=True,
        calculate_vega=True,
        calculate_curvature=True,
        derive_curvature_from_vega=True,
        include_bucket_detail=True,
    )

    # --- build portfolios ---
    print_section("1. BUILD EQUITY PORTFOLIO")
    eq_port = build_equity_portfolio()
    print(f"  Positions: {len(eq_port.positions)} | "
          f"Value ${eq_port.get_portfolio_value():,.2f}")

    print_section("2. BUILD FIXED INCOME PORTFOLIO")
    fi_port = build_fi_portfolio()
    print(f"  Positions: {len(fi_port.positions)} | "
          f"Value ${fi_port.get_portfolio_value():,.2f}")

    # --- sensitivity conversion ---
    print_section("3. PORTFOLIOS → SIMM SENSITIVITIES")
    eq_sens, fi_sens = convert_to_sensitivities(eq_port, fi_port, config)

    # --- profile sensitivities ---
    from quantark.simm.sensitivity import EquityDeltaSensitivity, IRDeltaSensitivity
    from quantark.simm.sensitivity import EquityVegaSensitivity

    print("\n  Sensitivity breakdown:")
    n_eqd = sum(1 for s in eq_sens if isinstance(s, EquityDeltaSensitivity))
    n_eqv = sum(1 for s in eq_sens if isinstance(s, EquityVegaSensitivity))
    n_ird = sum(1 for s in fi_sens if isinstance(s, IRDeltaSensitivity))
    print(f"    Equity Delta: {n_eqd}  |  Vega: {n_eqv}")
    print(f"    IR     Delta: {n_ird}")

    # --- SIMM ---
    print_section("4. RUN SIMM")
    result = run_simm(eq_sens, fi_sens, config)

    print_section("5. DASHBOARD")
    render_dashboard(result, eq_port, fi_port)

    print_section("COMPLETE")
    print(f"\n  Open {OUTPUT_DASHBOARD} in a browser to explore the results.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
