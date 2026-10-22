"""
SIMM xlsx input template demo.

Demonstrates the pricing/stress-result workflow for ISDA SIMM v2.6:

1. Generate the input template workbook (PricingResults + StressResults).
2. Fill it with pricing results: a base PV per trade and one shifted PV
   per risk-factor shock (the Section C.2/C.3 shift conventions).
3. Load the workbook and calculate SIMM. No Greeks are required - the
   sensitivities are simply s = ShiftedPV - BasePV.

Run:
    python example/simm_xlsx_template_demo.py
"""

from pathlib import Path

import openpyxl

from quantark.simm.template import (
    PRICING_SHEET,
    STRESS_SHEET,
    calculate_simm_from_xlsx,
    create_simm_input_template,
)


def main() -> None:
    path = Path("simm_input_demo.xlsx")

    # ------------------------------------------------------------------
    # 1. Generate the template
    # ------------------------------------------------------------------
    create_simm_input_template(path)
    print(f"Template created: {path}")

    # ------------------------------------------------------------------
    # 2. Fill it with pricing / stress results
    # ------------------------------------------------------------------
    # Trade 1: a USD interest-rate swap (RatesFX product class).
    #   Base PV 1,000,000. Re-priced with the OIS curve shifted up 1bp at
    #   the 5y and 10y vertices.
    # Trade 2: an equity option on AAPL (Equity product class).
    #   Base PV 500,000. Re-priced with the spot up 1% and the implied
    #   vol up 1 vol point (1y expiry).
    pricing_rows = [
        ("SWAP1", "RatesFX", 1_000_000.0),
        ("OPT1", "Equity", 500_000.0),
    ]
    stress_rows = [
        # TradeId, RiskType, Qualifier, Bucket, Label1, Label2, ShiftedPV, ImpliedVol
        ("SWAP1", "Risk_IRCurve", "USD", "", "5y", "OIS", 1_012_500.0, None),
        ("SWAP1", "Risk_IRCurve", "USD", "", "10y", "OIS", 1_018_000.0, None),
        ("OPT1", "Risk_Equity", "AAPL", "5", "", "", 503_500.0, None),
        ("OPT1", "Risk_EquityVol", "AAPL", "5", "1y", "", 500_900.0, None),
        # The equity option also carries IR discounting risk - it stays in
        # the Equity product class (paragraph 6).
        ("OPT1", "Risk_IRCurve", "USD", "", "1y", "OIS", 500_150.0, None),
    ]

    wb = openpyxl.load_workbook(path)
    for row in pricing_rows:
        wb[PRICING_SHEET].append(row)
    for row in stress_rows:
        wb[STRESS_SHEET].append(row)
    wb.save(path)
    print(f"Filled {len(pricing_rows)} trades / {len(stress_rows)} stress results")

    # ------------------------------------------------------------------
    # 3. Calculate SIMM
    # ------------------------------------------------------------------
    template_input, result = calculate_simm_from_xlsx(path)

    print(f"\nParsed {len(template_input.sensitivities)} sensitivities")
    print()
    print(result)


if __name__ == "__main__":
    main()
