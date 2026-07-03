"""Term-structure tests for PDE solvers (spec test layers 2/4)."""
from datetime import datetime

import numpy as np
import pytest

from term_structure_benchmarks import make_term_env, reference_european_call_price

from quantark.asset.equity.engine.pde.european_pde_solver import EuropeanPDESolver
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
    exactly what a pre-upgrade solver computed from the term env."""
    T = float(maturity)
    return PricingEnvironment(
        rate_curve=FlatRateCurve(env_term.get_rate(T)),
        valuation_date=datetime(2026, 7, 3),
        spot_quote=SpotQuote(100.0),
        vol_surface=FlatVolSurface(env_term.get_vol(ref_strike, T)),
        div_yield=ContinuousDividendYield(
            max(-0.20, min(0.20, env_term.get_div_yield(T)))
        ),
    )


@pytest.mark.parametrize("shape", ["up", "down", "kinked"])
def test_european_pde_matches_term_benchmark(shape):
    """Deterministic solver against the exact closed-form reference.

    A European terminal-value problem depends on the curves only through
    cumulative-to-T quantities, so this checks the correctness of the
    per-step discretization (drift/discount path), not discrimination.
    """
    env = make_term_env(shape)
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.5)
    px = EuropeanPDESolver().price(option, env)
    ref = reference_european_call_price(env, 100.0, 1.5)
    assert px == pytest.approx(ref, rel=2e-3)


def test_barrier_pde_sees_term_structure():
    """Path-dependent payoff: term vs collapsed must differ."""
    from quantark.asset.equity.engine.pde.barrier_pde_solver import BarrierPDESolver
    from quantark.asset.equity.product.option import BarrierOption
    from quantark.util.enum import BarrierType, ObservationType

    def price_fn(env):
        option = BarrierOption(
            strike=100.0,
            option_type=OptionType.CALL,
            barrier=120.0,
            barrier_type=BarrierType.UP_OUT,
            maturity=1.0,
            rebate=0.0,
            observation_type=ObservationType.CONTINUOUS,
        )
        return BarrierPDESolver().price(option, env)

    env_term = make_term_env("kinked")
    px_term = price_fn(env_term)
    px_collapsed = price_fn(_collapsed_flat_env(env_term, 1.0))
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)


def test_american_pde_sees_term_structure():
    from quantark.asset.equity.engine.pde.american_pde_solver import (
        AmericanPDESolver,
    )
    from quantark.asset.equity.product.option import AmericanOption

    def price_fn(env):
        option = AmericanOption(100.0, OptionType.PUT, maturity=1.0)
        return AmericanPDESolver().price(option, env)

    env_term = make_term_env("kinked")
    px_term = price_fn(env_term)
    px_collapsed = price_fn(_collapsed_flat_env(env_term, 1.0))
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)
