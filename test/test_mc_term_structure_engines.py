"""Term-structure benchmark tests for upgraded MC engines (spec test layers 2/3)."""
from datetime import datetime

import numpy as np
import pytest

from term_structure_benchmarks import make_term_env, reference_european_call_price

from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType


def _collapsed_flat_env(env_term, maturity, ref_strike=100.0):
    """Flat env matched to the term env's cumulative-to-maturity scalars —
    exactly what a pre-upgrade engine computed from the term env."""
    T = float(maturity)  # MUST be the product's actual pricing maturity
    return PricingEnvironment(
        rate_curve=FlatRateCurve(env_term.get_rate(T)),
        valuation_date=datetime(2026, 7, 3),
        spot_quote=SpotQuote(100.0),
        vol_surface=FlatVolSurface(env_term.get_vol(ref_strike, T)),
        div_yield=ContinuousDividendYield(
            max(-0.20, min(0.20, env_term.get_div_yield(T)))
        ),
    )


def _term_sensitivity_check(price_fn, maturity):
    """An upgraded engine must price the term env differently from the
    collapsed flat env (the old scalar behavior made them equal)."""
    env_term = make_term_env("kinked")
    px_term = price_fn(env_term)
    px_collapsed = price_fn(_collapsed_flat_env(env_term, maturity))
    assert px_term != pytest.approx(px_collapsed, rel=1e-4)
    return px_term, px_collapsed


@pytest.mark.parametrize("shape", ["up", "down", "kinked"])
def test_euro_mc_matches_term_benchmark(shape):
    env = make_term_env(shape)
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.5)
    engine = EuropeanMCEngine(params=MCParams(num_paths=200_000, seed=42))
    px = engine.price(option, env)
    ref = reference_european_call_price(env, 100.0, 1.5)
    assert px == pytest.approx(ref, rel=1e-2)


def test_euro_mc_flat_env_still_matches_reference():
    env = make_term_env("flat")
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.0)
    px = EuropeanMCEngine(params=MCParams(num_paths=200_000, seed=42)).price(option, env)
    assert px == pytest.approx(
        reference_european_call_price(env, 100.0, 1.0), rel=1e-2
    )


@pytest.mark.parametrize("shape", ["up", "kinked"])
def test_euro_mc_forward_reproduction_via_parity(shape):
    """C - P = DF*(F - K) exactly in the model; MC noise partially cancels."""
    env = make_term_env(shape)
    K, T = 100.0, 2.0
    call = EuropeanVanillaOption(K, OptionType.CALL, maturity=T)
    put = EuropeanVanillaOption(K, OptionType.PUT, maturity=T)
    engine = EuropeanMCEngine(params=MCParams(num_paths=200_000, seed=42))
    c, p = engine.price(call, env), engine.price(put, env)
    df = env.get_discount_factor(T)
    fwd = env.spot * np.exp((env.get_rate(T) - env.get_div_yield(T)) * T)
    assert c - p == pytest.approx(df * (fwd - K), rel=2e-2, abs=0.15)


def test_digital_mc_sees_term_structure():
    from quantark.asset.equity.engine.mc.digital_option_mc_engine import (
        DigitalOptionMCEngine,
    )
    from quantark.asset.equity.product.option.digital_option import (
        CashOrNothingDigitalOption,
    )

    def price_fn(env):
        option = CashOrNothingDigitalOption(
            strike=100.0, payout=10.0, option_type=OptionType.CALL, maturity=2.0
        )
        return DigitalOptionMCEngine(params=MCParams(num_paths=100_000, seed=42)).price(option, env)

    _term_sensitivity_check(price_fn, maturity=2.0)
