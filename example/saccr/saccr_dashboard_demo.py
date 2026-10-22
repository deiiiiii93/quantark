"""SA-CCR dashboard demo: render an interactive HTML EAD report.

Builds a small multi-asset-class netting set, runs the SA-CCR calculation, and
writes an interactive Plotly dashboard alongside this script.

Run:
    python example/saccr_dashboard_demo.py
"""

from pathlib import Path

from quantark.saccr import (
    SACCRCalculator,
    SACCRDashboard,
    SACCRTrade,
    SACCRNettingSet,
    AssetClass,
    Position,
    OptionType,
    CommodityType,
)


def build_netting_set() -> SACCRNettingSet:
    """A mixed netting set spanning IR, FX, equity and commodity trades."""
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
        SACCRTrade(
            "FX_FWD_EURUSD", AssetClass.FX, notional=8_000, market_value=-15,
            currency_pair="EUR/USD", maturity=1.5, position=Position.LONG,
        ),
        SACCRTrade(
            "EQ_FWD_AAPL", AssetClass.EQUITY, notional=3_000, market_value=40,
            reference_entity="AAPL", maturity=2.0, position=Position.LONG,
        ),
        SACCRTrade(
            "COMM_OIL_FWD", AssetClass.COMMODITY, notional=6_000, market_value=10,
            commodity_type=CommodityType.CRUDE_OIL, maturity=1.0, position=Position.LONG,
        ),
    ]
    return SACCRNettingSet("MIXED_NS", trades, is_margined=False)


def main() -> None:
    netting_set = build_netting_set()
    result = SACCRCalculator().calculate(netting_set)

    out = Path(__file__).with_name("saccr_dashboard.html")
    SACCRDashboard(result, netting_set=netting_set, currency="USD").generate(out)

    print(f"EAD = {result.ead:,.2f}  (RC={result.rc:,.2f}, PFE={result.pfe:,.2f})")
    print(f"Dashboard written to: {out}")


if __name__ == "__main__":
    main()
