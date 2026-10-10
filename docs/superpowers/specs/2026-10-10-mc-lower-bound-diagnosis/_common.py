"""Shared helpers for the W13 diagnosis scripts (runtime patches only; no library change)."""
from __future__ import annotations

import math
import warnings
from datetime import datetime

from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.enum.engine_enums import MonteCarloMethod

VALUATION = datetime(2026, 1, 5)

#: an upward-sloping zero curve (time, zero rate), linear in the zero rate
TERM_PILLARS = [(0.0, 0.01), (0.01, 0.02), (0.25, 0.03), (1.0, 0.045), (2.0, 0.05)]


def make_env(spot=90.0, rate=0.03, div=0.01, vol=0.20, term_curve=False):
    curve = LinearRateCurve(TERM_PILLARS) if term_curve else FlatRateCurve(rate=rate)
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=vol),
                              rate_curve=curve, div_yield=ContinuousDividendYield(div), valuation_date=VALUATION)


def make_option(T=1 / 365, option_type=OptionType.PUT, strike=100.0, lag_days=0):
    kw = {}
    if lag_days:
        kw["settlement_convention"] = SettlementConvention(lag=lag_days / 365.0,
                                                           lag_unit=SettlementLagUnit.YEAR_FRACTION)
    return EuropeanVanillaOption(strike=strike, option_type=option_type, maturity=T, **kw)


METHODS = {
    "plain": (MonteCarloMethod.PSEUDO, False),
    "antithetic": (MonteCarloMethod.PSEUDO, True),
    "qmc": (MonteCarloMethod.QUASI, False),
    "qmc+antithetic": (MonteCarloMethod.QUASI, True),   # the engine drops antithetic under QMC
    "rqmc": (MonteCarloMethod.RANDOMIZED_QUASI, False),
}


def make_engine(method="plain", num_paths=256, seed=7, time_steps=100, **extra):
    mc_method, anti = METHODS[method]
    params = MCParams(seed=seed, num_paths=num_paths, time_steps=time_steps, use_antithetic=anti, **extra)
    return EuropeanMCEngine(params=params, method=mc_method)


def mc_raw(product, env, *, method="plain", num_paths=256, seed=7, time_steps=100, **extra):
    """The engine's own estimate with the lower-bound check disabled on THIS instance only.

    Returns (price, std_error, bound, engine): the bound the engine would have compared against is captured.
    """
    engine = make_engine(method, num_paths, seed, time_steps, **extra)
    captured = {}
    original = engine._european_lower_bound

    def capture(*a, **k):
        captured["bound"] = original(*a, **k)
        return -math.inf

    engine._european_lower_bound = capture  # instance-level runtime patch
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        price = float(engine.price(product, env))
    return price, float(engine.get_last_std_error()), captured["bound"], engine


def mc_checked(product, env, **kw):
    """The unpatched engine: returns the price or the refusal text."""
    engine = make_engine(**kw)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(engine.price(product, env)), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def bs_price(product, env):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(BlackScholesEngine().price(product, env))


def timing_of(product, env):
    return resolve_terminal_timing(product, env)
