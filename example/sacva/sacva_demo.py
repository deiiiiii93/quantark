"""SA-CVA demo: multi-class CVA portfolio -> capital + dashboard.

Run:
    python example/sacva_demo.py
"""

from quantark.sacva import (SACVACalculator, CVAPortfolio, CVASensitivity,
                            RiskClass, RiskType, CreditQuality)


def main():
    sens = [
        # Counterparty credit spread: CVA exposure to counterparty A (5y), IG bucket 3
        CVASensitivity(RiskClass.COUNTERPARTY_CREDIT, RiskType.DELTA, bucket=3,
                       risk_factor="A:5y", s_cva=25_000.0, s_hdg=0.0,
                       name="A", tenor=5.0, credit_quality=CreditQuality.IG),
        # Single-name CDS hedge on A (reduces the CVA credit-spread sensitivity)
        CVASensitivity(RiskClass.COUNTERPARTY_CREDIT, RiskType.DELTA, bucket=3,
                       risk_factor="A:5y:hdg", s_cva=0.0, s_hdg=20_000.0,
                       name="A", tenor=5.0, credit_quality=CreditQuality.IG),
        # IR delta (USD 5y) on the exposure component
        CVASensitivity(RiskClass.INTEREST_RATE, RiskType.DELTA, bucket=0,
                       currency="USD", risk_factor="USD:5y", s_cva=12_000.0, tenor=5.0),
        # IR vega
        CVASensitivity(RiskClass.INTEREST_RATE, RiskType.VEGA, bucket=0,
                       currency="USD", risk_factor="USD:vol", s_cva=3_000.0),
    ]
    pf = CVAPortfolio(sens, reporting_currency="USD")
    res = SACVACalculator().calculate(pf)

    print(f"SA-CVA total capital : {res.total_capital:,.2f}")
    print(f"  delta : {res.delta_capital:,.2f}")
    print(f"  vega  : {res.vega_capital:,.2f}")
    print("  by risk class x type:")
    for k, v in res.by_risk_class.items():
        print(f"    {k:32s} {v:,.2f}")

    try:
        from quantark.sacva import SACVADashboard
        SACVADashboard(res, portfolio=pf).generate("example/sacva_dashboard.html")
        print("dashboard -> example/sacva_dashboard.html")
    except ImportError:
        print("(plotly not installed; skipping dashboard)")


if __name__ == "__main__":
    main()
