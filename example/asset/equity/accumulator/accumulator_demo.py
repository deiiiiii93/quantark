"""
Demonstration of accumulator option pricing.

An accumulator ("accumulator forward") is a call-only structured forward: on each
observation date the buyer accumulates shares at a strike set below spot, with a
geared loss leg below the strike and an upper knock-out barrier. This script:

- prices a TERMINATION accumulator (whole contract knocks out) with both the
  analytical decomposition engine and the exact Monte Carlo benchmark;
- prices a SINGLE_DAY accumulator (only that day's accrual is cancelled), whose
  analytical price is exact (no barrier-shift approximation);
- shows the effect of the knock-out rebate and the geared loss leg.
"""

from datetime import datetime

from quantark.asset.equity.engine.analytical import AccumulatorAnalyticalEngine
from quantark.asset.equity.engine.mc import AccumulatorMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import AccumulatorOption
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import AccumulatorKnockOutType, MonteCarloMethod, OptionType


def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def make_pricing_env() -> PricingEnvironment:
    """Flat market: spot 100, 22% vol, 3% rate, 1% dividend."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0, asset_name="XYZ"),
        vol_surface=FlatVolSurface(volatility=0.22),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2024, 1, 1),
    )


def monthly_observations(n: int = 12) -> list:
    """Twelve monthly observation dates over a one-year tenor."""
    return [round((i + 1) / 12.0, 10) for i in range(n)]


def make_accumulator(knock_out_type, rebate_rate=0.0) -> AccumulatorOption:
    """A one-year accumulator: strike 96, barrier 107, 2x gearing, monthly fixings."""
    return AccumulatorOption(
        strike=96.0,
        knock_out_barrier=107.0,
        option_type=OptionType.CALL,
        maturity=1.0,
        initial_price=100.0,
        notional=100.0,  # 1 share per observation
        gearing=2.0,
        knock_out_type=knock_out_type,
        knock_out_rebate_rate=rebate_rate,
        observation_dates=monthly_observations(),
    )


def price_both(product, env, qmc_paths=200_000):
    """Price with the analytical and Monte Carlo engines."""
    analytic = AccumulatorAnalyticalEngine().price(product, env)
    mc_engine = AccumulatorMCEngine(
        MCParams(num_paths=qmc_paths, seed=7), method=MonteCarloMethod.QUASI
    )
    mc = mc_engine.price(product, env)
    return analytic, mc, mc_engine.get_last_std_error()


def main() -> None:
    env = make_pricing_env()
    print(f"Market: spot={env.spot:.1f}, vol=22%, rate=3%, div=1%")
    print("Accumulator: K=96, KO=107, gearing=2x, 12 monthly fixings, 1 share/fixing")

    print_section("TERMINATION knock-out (whole contract terminates on a breach)")
    term = make_accumulator(AccumulatorKnockOutType.TERMINATION)
    analytic, mc, se = price_both(term, env)
    print(f"  Analytical (BGK barrier shift) : {analytic:12.4f}")
    print(f"  Monte Carlo (exact discrete)   : {mc:12.4f}  (std err {se:.4f})")
    print(f"  difference                     : {analytic - mc:12.4f}")
    print("  Note: the gap is the analytical BGK discrete-monitoring approximation.")

    print_section("TERMINATION with a 1% knock-out rebate")
    term_rebate = make_accumulator(
        AccumulatorKnockOutType.TERMINATION, rebate_rate=0.01
    )
    analytic_r, mc_r, se_r = price_both(term_rebate, env)
    print(f"  Analytical : {analytic_r:12.4f}")
    print(f"  Monte Carlo: {mc_r:12.4f}  (std err {se_r:.4f})")
    print(f"  rebate adds: {analytic_r - analytic:12.4f} (analytical)")

    print_section("SINGLE_DAY knock-out (only the breached day's accrual is cancelled)")
    single = make_accumulator(AccumulatorKnockOutType.SINGLE_DAY)
    analytic_s, mc_s, se_s = price_both(single, env)
    print(f"  Analytical (exact, no shift)   : {analytic_s:12.4f}")
    print(f"  Monte Carlo (exact discrete)   : {mc_s:12.4f}  (std err {se_s:.4f})")
    print(f"  difference                     : {analytic_s - mc_s:12.4f}")
    print("  Note: SINGLE_DAY analytical is exact, so the two agree within MC noise.")

    print_section("Interpretation")
    print("  The accumulator value is negative here: with strike below spot the")
    print("  buyer is effectively short 2x downside puts (the geared loss leg),")
    print("  which dominates the linear gain leg. The upper knock-out caps the")
    print("  accumulation; a rebate softens the knock-out for the buyer.")


if __name__ == "__main__":
    main()
