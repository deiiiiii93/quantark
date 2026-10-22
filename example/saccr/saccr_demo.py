"""SA-CCR demo: Exposure at Default for a Basel Annex 4a Example-1 netting set.

Run:
    python example/saccr_demo.py
"""

from quantark.saccr import (
    SACCRCalculator,
    SACCRTrade,
    SACCRNettingSet,
    AssetClass,
    Position,
    OptionType,
)
from quantark.util.numerical import format_currency


def build_netting_set() -> SACCRNettingSet:
    """Basel Annex 4a, Example 1: three interest-rate derivatives (unmargined)."""
    trades = [
        SACCRTrade(
            "IR_SWAP_10Y", AssetClass.INTEREST_RATE, notional=10_000, market_value=30,
            currency="USD", start_date=0, end_date=10, maturity=10,
            position=Position.LONG,
        ),
        SACCRTrade(
            "IR_SWAP_4Y", AssetClass.INTEREST_RATE, notional=10_000, market_value=-20,
            currency="USD", start_date=0, end_date=4, maturity=4,
            position=Position.SHORT,
        ),
        SACCRTrade(
            "EUR_SWAPTION_1Y10Y", AssetClass.INTEREST_RATE, notional=5_000,
            market_value=50, currency="EUR", start_date=1, end_date=11, maturity=5.5,
            position=Position.LONG, is_option=True, option_type=OptionType.PUT,
            underlying_price=0.06, strike_price=0.05, exercise_date=1.0,
        ),
    ]
    return SACCRNettingSet("DEMO_NS", trades, is_margined=False)


def main() -> None:
    netting_set = build_netting_set()
    result = SACCRCalculator().calculate(netting_set)

    print("=== SA-CCR EAD (Basel Annex 4a, Example 1) ===")
    print(f"V (market value)   : {format_currency(result.v)}")
    print(f"C (collateral)     : {format_currency(result.c)}")
    print(f"RC                 : {format_currency(result.rc)}")
    print(f"Multiplier         : {result.multiplier:.4f}")
    print(f"AddOn (aggregate)  : {format_currency(result.addon_aggregate)}")
    print(f"PFE                : {format_currency(result.pfe)}")
    print(f"alpha              : {result.alpha}")
    print(f"EAD                : {format_currency(result.ead)}")
    print()
    print("AddOn by asset class:")
    for name, value in result.addon_by_asset_class.items():
        print(f"  {name:<14}: {format_currency(value)}")
    print("AddOn by hedging set:")
    for label, value in result.addon_by_hedging_set.items():
        print(f"  {label:<14}: {format_currency(value)}")


if __name__ == "__main__":
    main()
