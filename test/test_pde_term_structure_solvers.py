"""Term-structure tests for PDE solvers (spec test layers 2/4)."""
from datetime import datetime

import numpy as np
import pytest

from term_structure_benchmarks import make_term_env, reference_european_call_price
from golden_compare import GOLDEN_REL_TOL

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
    # Same-machine golden; x86_64 CI drifts from the ARM64 freeze host by the
    # last ULP, so compare with cross-arch tolerance (see golden_compare).
    assert px == pytest.approx(GOLDEN_PDE_PRE, rel=GOLDEN_REL_TOL)


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


def test_heston_pde_cumulative_convention_is_term_exact_for_europeans():
    """Heston PDE prices a European terminal-value problem: cumulative
    r(T)/q(T) inputs are term-structure exact, so a term env and its
    collapsed cumulative env must agree (spec: documented convention)."""
    from quantark.asset.equity.engine.pde import HestonPDESolver
    from quantark.volmodels.heston import HestonParams

    P = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    option = EuropeanVanillaOption(100.0, OptionType.CALL, maturity=1.0)
    env_term = make_term_env("kinked")
    px_term = HestonPDESolver(P, n_x=120, n_v=50, n_t=60).price(option, env_term)
    px_collapsed = HestonPDESolver(P, n_x=120, n_v=50, n_t=60).price(
        option, _collapsed_flat_env(env_term, 1.0)
    )
    assert px_term == pytest.approx(px_collapsed, rel=1e-8)


def test_term_pde_not_pathologically_slower_than_flat():
    """Loose CI smoke guard (2x) — the formal 20% budget is measured by
    example/pde_term_structure_benchmark.py (see Phase 2 commit messages)."""
    import time

    from quantark.asset.equity.engine.pde import SnowballPDESolver

    def best_time(env):
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            SnowballPDESolver().price(_standard_snowball(), env)
            best = min(best, time.perf_counter() - start)
        return best

    flat = best_time(make_term_env("flat"))
    term = best_time(make_term_env("kinked"))
    assert term <= 2.0 * flat + 0.05  # catches pathology, not noise


def test_df_memo_invalidates_when_market_data_replaced():
    """Codex code-review regression: replacing curves on a long-lived env
    must never reuse memoized DFs/coefficients from the old market data."""
    from quantark.asset.equity.engine.pde.european_pde_solver import (
        EuropeanPDESolver,
    )

    solver = EuropeanPDESolver()
    env = make_term_env("flat")

    df_before = solver._df_between_times(env, 0.5, 1.0)
    carry_before = solver._carry_df_between_times(env, 0.5, 1.0)

    env.rate_curve = FlatRateCurve(0.10)          # attribute replacement
    env.div_yield = ContinuousDividendYield(0.05)

    df_after = solver._df_between_times(env, 0.5, 1.0)
    carry_after = solver._carry_df_between_times(env, 0.5, 1.0)
    assert df_after == pytest.approx(np.exp(-0.10 * 0.5), rel=1e-12)
    assert carry_after == pytest.approx(np.exp(-0.05 * 0.5), rel=1e-12)
    assert df_after != pytest.approx(df_before, rel=1e-6)
    assert carry_after != pytest.approx(carry_before, rel=1e-6)


def test_step_coefficients_invalidate_when_div_or_vol_replaced():
    from quantark.asset.equity.engine.pde.european_pde_solver import (
        EuropeanPDESolver,
    )
    from quantark.param.div.dividend_yield import TermStructureDividendYield

    solver = EuropeanPDESolver()
    env = make_term_env("kinked")
    t_vec = np.linspace(0.0, 1.0, 6)
    dx_vec = np.full(10, 0.02)

    sc1 = solver._build_step_coefficients(env, 100.0, t_vec, dx_vec, 11)
    sc_cached = solver._build_step_coefficients(env, 100.0, t_vec, dx_vec, 11)
    assert sc_cached is sc1  # warm hit while market objects unchanged

    env.div_yield = TermStructureDividendYield(
        times=[0.5, 1.0], yields=[0.05, 0.08]
    )
    sc2 = solver._build_step_coefficients(env, 100.0, t_vec, dx_vec, 11)
    assert sc2 is not sc1
    l1, c1, u1 = sc1.lcu_sets[int(sc1.set_index[0])]
    l2, c2, u2 = sc2.lcu_sets[int(sc2.set_index[0])]
    assert not np.array_equal(l1, l2)  # coefficients reflect the new carry


def test_pde_price_reacts_to_in_place_env_curve_replacement():
    """End-to-end: same env object, curves replaced between prices."""
    from quantark.asset.equity.engine.pde.barrier_pde_solver import BarrierPDESolver
    from quantark.asset.equity.product.option import BarrierOption
    from quantark.util.enum import BarrierType, ObservationType

    option = BarrierOption(
        strike=100.0, option_type=OptionType.CALL, barrier=120.0,
        barrier_type=BarrierType.UP_OUT, maturity=1.0, rebate=2.0,
        observation_type=ObservationType.CONTINUOUS,
    )
    env = make_term_env("kinked")
    solver = BarrierPDESolver()
    px1 = solver.price(option, env)

    env.rate_curve = FlatRateCurve(0.08)
    px2 = solver.price(option, env)
    assert px2 != pytest.approx(px1, rel=1e-6)

    fresh = make_term_env("kinked")
    fresh.rate_curve = FlatRateCurve(0.08)
    px_fresh = BarrierPDESolver().price(option, fresh)
    assert px2 == pytest.approx(px_fresh, rel=1e-12)
