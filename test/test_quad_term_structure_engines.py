"""Term-structure tests for QUAD engines (spec test layers 2/4)."""
from datetime import datetime

import numpy as np
import pytest

from term_structure_benchmarks import make_term_env, reference_european_call_price

from quantark.asset.equity.param import QuadParams
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


_MONTHLY = [round(i / 12.0, 8) for i in range(1, 13)]


def test_one_touch_quad_sees_term_structure():
    from quantark.asset.equity.engine.quad import OneTouchQuadEngine
    from quantark.asset.equity.product.option import OneTouchOption
    from quantark.util.enum import BarrierDirection, ObservationType, TouchType

    def price_fn(env):
        option = OneTouchOption(
            barrier=110.0,
            barrier_direction=BarrierDirection.UP,
            maturity=1.0,
            rebate=5.0,
            payment_at_hit=True,
            touch_type=TouchType.ONE_TOUCH,
            observation_type=ObservationType.DISCRETE,
            observation_dates=_MONTHLY,
        )
        return OneTouchQuadEngine(params=QuadParams(grid_points=801)).price(
            option, env
        )

    env_term = make_term_env("kinked")
    assert price_fn(env_term) != pytest.approx(
        price_fn(_collapsed_flat_env(env_term, 1.0)), rel=1e-5
    )


def test_barrier_quad_sees_term_structure():
    from quantark.asset.equity.engine.quad import BarrierQuadEngine
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
            observation_type=ObservationType.DISCRETE,
            observation_dates=_MONTHLY,
        )
        return BarrierQuadEngine(params=QuadParams(grid_points=801)).price(
            option, env
        )

    env_term = make_term_env("kinked")
    assert price_fn(env_term) != pytest.approx(
        price_fn(_collapsed_flat_env(env_term, 1.0)), rel=1e-5
    )


def test_expiry_paid_rebate_discounting_is_curve_exact():
    """payment_at_hit=False rebates discount from observation to maturity via
    df(T)/df(t_obs) — the discounting leg is the term-sensitive piece
    (codex plan-review required test)."""
    from quantark.asset.equity.engine.quad import OneTouchQuadEngine
    from quantark.asset.equity.product.option import OneTouchOption
    from quantark.util.enum import BarrierDirection, ObservationType, TouchType

    def price_fn(env):
        option = OneTouchOption(
            barrier=110.0,
            barrier_direction=BarrierDirection.UP,
            maturity=1.0,
            rebate=5.0,
            payment_at_hit=False,
            touch_type=TouchType.ONE_TOUCH,
            observation_type=ObservationType.DISCRETE,
            observation_dates=_MONTHLY,
        )
        return OneTouchQuadEngine(params=QuadParams(grid_points=801)).price(
            option, env
        )

    env_term = make_term_env("kinked")
    assert price_fn(env_term) != pytest.approx(
        price_fn(_collapsed_flat_env(env_term, 1.0)), rel=1e-6
    )


@pytest.mark.parametrize("shape", ["up", "down", "kinked"])
def test_european_quad_matches_term_benchmark(shape):
    """European terminal-density integral: cumulative inputs are term-exact."""
    from quantark.asset.equity.engine.quad import EuropeanQuadEngine
    from quantark.asset.equity.product.option import EuropeanVanillaOption

    env = make_term_env(shape)
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.5)
    px = EuropeanQuadEngine().price(option, env)
    assert px == pytest.approx(
        reference_european_call_price(env, 100.0, 1.5), rel=2e-3
    )
