"""Cross greeks requested together share engine pricings.

The spec promises that speed reuses the delta/gamma spot legs and that the
time family prices the advanced-date scenario once per clock. These tests
count engine.price calls and distinct (spot, vol, date) scenarios: every
duplicate is a wasted full solve on the PDE/QUAD autocallables the feature
targets.
"""

from datetime import datetime

import pytest

from quantark.asset.equity.engine.analytical import BlackScholesEngine
from quantark.asset.equity.engine.pde_engine import PDEEngine
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import GreeksCalculationMode


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.2),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=datetime(2024, 1, 1),
    )


def _vanilla():
    return EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0)


def _count_pricings(monkeypatch, engine_cls, request, **calc_kw):
    calls = []
    original = engine_cls.price

    def spy(self, product, pricing_env, *args, **kwargs):
        calls.append(
            (
                round(pricing_env.spot_quote.spot, 9),
                round(pricing_env.vol_surface.volatility, 9),
                pricing_env.valuation_date,
            )
        )
        return original(self, product, pricing_env, *args, **kwargs)

    monkeypatch.setattr(engine_cls, "price", spy)
    GreeksCalculator(**calc_kw).calculate(
        _vanilla(), _env(), engine_cls(), method="numerical", greeks=request
    )
    return len(calls), len(set(calls))


@pytest.mark.parametrize(
    "request_, minimal",
    [
        (["delta", "gamma", "speed"], 5),                       # base, S±h, S±2h
        (["theta", "charm", "color"], 6),                       # base, S±h, adv, adv S±h
        (["delta", "gamma", "theta", "charm", "color"], 6),
        (["delta", "gamma", "vega", "theta", "charm", "color", "vega_theta"], 8),
    ],
)
def test_bump_mode_shares_pricings_across_cross_greeks(monkeypatch, request_, minimal):
    total, distinct = _count_pricings(monkeypatch, BlackScholesEngine, request_)
    assert distinct == minimal
    assert total == distinct, f"{total - distinct} duplicate pricings for {request_}"


def test_engine_mode_base_greeks_are_not_recomputed_for_charm(monkeypatch):
    calls = []
    original = PDEEngine.calculate_greeks

    def spy(self, product, pricing_env, *args, **kwargs):
        calls.append(pricing_env.valuation_date)
        return original(self, product, pricing_env, *args, **kwargs)

    monkeypatch.setattr(PDEEngine, "calculate_greeks", spy)
    calc = GreeksCalculator(greeks_mode=GreeksCalculationMode.ENGINE)
    calc.calculate(
        _vanilla(), _env(), PDEEngine(params=PDEParams(accuracy="fast")),
        method="numerical", greeks=["delta", "gamma", "charm"],
    )
    # One grid solve at the base date, one at the advanced date.
    assert len(calls) == 2
