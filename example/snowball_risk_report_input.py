"""
Input module for `autocallable_risk_report`.

Usage:
  python -m asset.equity.report.autocallable_risk_report --input example/snowball_risk_report_input.py --out /tmp/snowball_report
"""

from datetime import datetime

from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.param import SpotQuote, TermStructureVolSurface
from quantark.param.div import TermStructureDividendYield
from quantark.param.rrf import FlatRateCurve
from quantark.priceenv import PricingEnvironment


def build_product():
    return create_standard_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        ko_barrier=103.0,
        ko_rate=0.15,
        ki_barrier=75.0,
        num_observations=12,
        is_reverse=False,
        include_principal=False,
    )


def build_pricing_env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=TermStructureVolSurface(
            times=[1.0 / 12.0, 0.25, 0.5, 1.0],
            vols=[0.23, 0.22, 0.215, 0.21],
        ),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=TermStructureDividendYield(
            times=[1.0 / 12.0, 0.25, 0.5, 1.0],
            yields=[0.031, 0.030, 0.029, 0.028],
        ),
        valuation_date=datetime.now(),
    )


# Optional historical inputs for shock-based analysis.
# Replace with real history (same length arrays) for production use.
historical_spot = [100.0, 101.0, 98.0, 99.5, 102.0, 101.5, 103.0, 100.5]
historical_q = [0.0300, 0.0301, 0.0298, 0.0302, 0.0305, 0.0303, 0.0301, 0.0300]


# Optional: Advanced Volatility Risk (Skew/Smile)
# This simulates the impact of a rotational shift in the vol surface.
# skew: linear slope change (dVol/dLogMoneyness)
# smile: curvature change (d^2Vol/dLogMoneyness^2)
skew_smile_shock = {"skew": -0.10, "smile": 0.05}


# Optional: Custom Stress Scenarios
# Define specific market shocks to test portfolio resilience.
stress_scenarios = [
    {
        "name": "Black Swan",
        "description": "Extreme market crash with vol spike and liquidity dry-up",
        "stresses": [
            {"parameter": "spot", "stress_value": -0.30, "stress_type": "percentage"},
            {"parameter": "vol", "stress_value": 0.50, "stress_type": "percentage"},  # +50% relative
            {"parameter": "div_yield", "stress_value": 0.01, "stress_type": "absolute"},  # +100bps
        ],
    },
    {
        "name": "Slow Bear",
        "spot_shock": -0.10,
        "vol_shock": 0.05,
        "q_shift": 0.0,
    },
]
