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


def _standard_snowball():
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig
    from quantark.asset.equity.product.option.snowball_option import SnowballOption
    from quantark.util.enum import ObservationType

    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        barrier_config=BarrierConfig(
            ko_barrier=1.03,
            ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=0.75,
            ki_observation_type=ObservationType.CONTINUOUS,
        ),
        payoff_config=None,
        contract_multiplier=1.0,
        maturity=1.0,
        is_reverse=False,
    )


def test_snowball_pde_sees_term_structure():
    from quantark.asset.equity.engine.pde import SnowballPDESolver

    env_term = make_term_env("kinked")
    px_term = SnowballPDESolver().price(_standard_snowball(), env_term)
    px_collapsed = SnowballPDESolver().price(
        _standard_snowball(), _collapsed_flat_env(env_term, 1.0)
    )
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)


def test_snowball_pde_flat_identity_golden():
    """Flat-input identity: price captured on this worktree BEFORE the
    snowball sweeps were wired for term structures (bit-exact via
    _flat_exact_step_coefficients)."""
    from quantark.asset.equity.engine.pde import SnowballPDESolver

    GOLDEN_PDE_PRE = 102.97478573304076
    px = SnowballPDESolver().price(_standard_snowball(), make_term_env("flat"))
    assert px == GOLDEN_PDE_PRE


def test_snowball_pde_sparse_fallback_sees_term_structure():
    """use_banded_solver=False path (codex plan-review finding): the sparse
    branch must consume per-step coefficients too, and agree with banded."""
    from quantark.asset.equity.engine.pde import SnowballPDESolver
    from quantark.asset.equity.param import PDEParams

    params = PDEParams(use_banded_solver=False)
    env_term = make_term_env("kinked")
    px_term = SnowballPDESolver(params=params).price(_standard_snowball(), env_term)
    px_collapsed = SnowballPDESolver(params=params).price(
        _standard_snowball(), _collapsed_flat_env(env_term, 1.0)
    )
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)

    px_banded = SnowballPDESolver().price(_standard_snowball(), env_term)
    assert px_term == pytest.approx(px_banded, rel=1e-6)


def _standard_phoenix():
    """Discrete monthly-KI Phoenix (the PDE path supports discrete KI)."""
    from quantark.asset.equity.product.option import create_standard_phoenix
    from quantark.asset.equity.product.option.observation_schedule import (
        ObservationRecord,
        ObservationSchedule,
    )
    from quantark.util.enum import ObservationType

    sched = ObservationSchedule(
        records=[
            ObservationRecord(observation_time=i / 12, barrier=75.0)
            for i in range(1, 13)
        ]
    )
    return create_standard_phoenix(
        initial_price=100.0, strike=100.0, maturity=1.0,
        ko_barrier=103.0, ki_barrier=75.0, coupon_barrier=85.0,
        coupon_rate=0.02, num_observations=12, memory_coupon=False,
        ki_continuous=False, ki_observation_type=ObservationType.DISCRETE,
        ki_observation_schedule=sched,
    )


def test_phoenix_pde_sees_term_structure():
    from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver

    env_term = make_term_env("kinked")
    px_term = PhoenixPDESolver().price(_standard_phoenix(), env_term)
    px_collapsed = PhoenixPDESolver().price(
        _standard_phoenix(), _collapsed_flat_env(env_term, 1.0)
    )
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)


def test_phoenix_pde_sparse_fallback_sees_term_structure():
    from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
    from quantark.asset.equity.param import PDEParams

    params = PDEParams(use_banded_solver=False)
    env_term = make_term_env("kinked")
    px_term = PhoenixPDESolver(params=params).price(_standard_phoenix(), env_term)
    px_collapsed = PhoenixPDESolver(params=params).price(
        _standard_phoenix(), _collapsed_flat_env(env_term, 1.0)
    )
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)

    px_banded = PhoenixPDESolver().price(_standard_phoenix(), env_term)
    assert px_term == pytest.approx(px_banded, rel=1e-6)


def test_ko_reset_pde_sees_term_structure():
    from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import (
        KOResetSnowballPDESolver,
    )
    from quantark.asset.equity.param import PDEParams
    from quantark.asset.equity.product.option import create_ko_reset_snowball
    from quantark.util.enum import PostKOScheduleMode

    product = create_ko_reset_snowball(
        initial_price=100.0,
        strike=100.0,
        maturity_pre=1.0,
        maturity_post=2.0,
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
        ki_continuous=False,
    )
    solver_params = PDEParams(grid_size=80, time_steps=40)
    env_term = make_term_env("kinked")
    px_term = KOResetSnowballPDESolver(solver_params).price(product, env_term)
    px_collapsed = KOResetSnowballPDESolver(solver_params).price(
        product, _collapsed_flat_env(env_term, 2.0)
    )
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)


def test_one_touch_pde_sees_term_structure():
    """Pay-at-hit rebate discounts at hit times — curve-exactness matters."""
    from quantark.asset.equity.engine.pde.one_touch_pde_solver import (
        OneTouchPDESolver,
    )
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
            observation_type=ObservationType.CONTINUOUS,
        )
        return OneTouchPDESolver().price(option, env)

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
