"""Result-dict key order is part of the GreeksCalculator contract.

Downstream code (portfolio storage, DataFrame builders) copies the dict
order into column order, so it must be deterministic across processes and
must match the incumbent ordering. These tests run the calculator in
subprocesses under several PYTHONHASHSEED values: an order that depends on
set iteration shows up as disagreement between seeds.
"""

import json
import os
import subprocess
import sys
from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine, DeltaOneEngine
from quantark.asset.equity.product.deltaone import SpotInstrument
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import DeltaOneType, OptionType

_SNIPPET = r"""
import json
from datetime import datetime
from quantark.asset.equity.engine.analytical import BlackScholesEngine, DeltaOneEngine
from quantark.asset.equity.product.deltaone import SpotInstrument
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import DeltaOneType, OptionType

env = PricingEnvironment(
    spot_quote=SpotQuote(spot=100.0),
    vol_surface=FlatVolSurface(volatility=0.2),
    rate_curve=FlatRateCurve(rate=0.02),
    div_yield=ContinuousDividendYield(div_yield=0.01),
    valuation_date=datetime(2024, 1, 1),
)
vanilla = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
linear = SpotInstrument(underlying="X", deltaone_type=DeltaOneType.STOCK)
calc = GreeksCalculator()
engine = BlackScholesEngine()
out = {
    "analytical_subset": list(calc.calculate(vanilla, env, engine, greeks=["theta", "gamma", "price", "delta"])),
    "analytical_extended": list(calc.calculate(vanilla, env, engine, greeks=["zomma", "charm", "vanna"])),
    "linear_default": list(calc.calculate(linear, env, DeltaOneEngine())),
    "linear_extras": list(calc.calculate(linear, env, DeltaOneEngine(), greeks=["volga", "vanna", "delta"])),
}
print(json.dumps(out))
"""

_EXPECTED = {
    # Incumbent analytical dict order, filtered.
    "analytical_subset": ["price", "delta", "gamma", "theta"],
    # Closed-form definition order: vanna ... zomma ... charm.
    "analytical_extended": ["vanna", "zomma", "charm"],
    # Incumbent literal order of the delta-one dict.
    "linear_default": [
        "price", "delta", "gamma", "vega", "theta",
        "convexity_theta", "r_theta", "q_theta", "rho", "dividend_rho",
    ],
    "linear_extras": ["delta", "vanna", "volga"],
}


def _run_under_seed(seed: int) -> dict:
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    env.setdefault("PYTHONPATH", os.getcwd())
    proc = subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        env=env, capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_key_order_is_hash_seed_independent(seed):
    assert _run_under_seed(seed) == _EXPECTED


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2024, 1, 1),
    )


def test_analytical_subset_keeps_incumbent_dict_order():
    calc = GreeksCalculator()
    vanilla = EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)
    out = calc.calculate(vanilla, _env(), BlackScholesEngine(), greeks=["theta", "gamma", "price", "delta"])
    assert list(out) == _EXPECTED["analytical_subset"]


def test_linear_default_keeps_incumbent_literal_order():
    calc = GreeksCalculator()
    linear = SpotInstrument(underlying="X", deltaone_type=DeltaOneType.STOCK)
    out = calc.calculate(linear, _env(), DeltaOneEngine())
    assert list(out) == _EXPECTED["linear_default"]
