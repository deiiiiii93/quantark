"""Term-structure tests for QUAD engines (spec test layers 2/4)."""
from datetime import datetime

import numpy as np
import pytest

from term_structure_benchmarks import make_term_env, reference_european_call_price
from golden_compare import GOLDEN_ABS_TOL, GOLDEN_REL_TOL

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


def test_snowball_quad_sees_term_structure():
    from quantark.asset.equity.engine.quad import SnowballQuadEngine

    env_term = make_term_env("kinked")
    px_term = SnowballQuadEngine().price(_standard_snowball(), env_term)
    px_collapsed = SnowballQuadEngine().price(
        _standard_snowball(), _collapsed_flat_env(env_term, 1.0)
    )
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)


def test_snowball_quad_flat_identity_golden():
    """Flat-input identity: price captured on this worktree BEFORE the
    snowball QUAD recursion was wired for term structures (bit-exact via
    the exact-scalar collapse in _term_step_params)."""
    from quantark.asset.equity.engine.quad import SnowballQuadEngine

    GOLDEN_QUAD_PRE = 102.97478568748562
    px = SnowballQuadEngine().price(_standard_snowball(), make_term_env("flat"))
    # Same-machine golden; x86_64 CI drifts from the ARM64 freeze host by the
    # last ULP, so compare with cross-arch tolerance (see golden_compare).
    assert px == pytest.approx(GOLDEN_QUAD_PRE, rel=GOLDEN_REL_TOL)


def _standard_phoenix():
    """Discrete monthly-KI Phoenix (matches the PDE test builder)."""
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


def test_phoenix_quad_sees_term_structure():
    from quantark.asset.equity.engine.quad import PhoenixQuadEngine

    env_term = make_term_env("kinked")
    px_term = PhoenixQuadEngine().price(_standard_phoenix(), env_term)
    px_collapsed = PhoenixQuadEngine().price(
        _standard_phoenix(), _collapsed_flat_env(env_term, 1.0)
    )
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)


def test_phoenix_quad_coupon_stats_see_term_structure():
    """Coupon stream stats (probabilities + discounted cashflows) must be
    computed with curve DFs (codex plan-review required test)."""
    from quantark.asset.equity.engine.quad import PhoenixQuadEngine
    from quantark.asset.equity.param import QuadParams

    env_term = make_term_env("kinked")
    env_coll = _collapsed_flat_env(env_term, 1.0)
    engine = PhoenixQuadEngine(params=QuadParams(grid_points=801))
    stats_term = engine.calculate_event_stats(_standard_phoenix(), env_term)
    stats_coll = engine.calculate_event_stats(_standard_phoenix(), env_coll)
    ecc_term = np.asarray(stats_term.expected_discounted_coupon_cashflow)
    ecc_coll = np.asarray(stats_coll.expected_discounted_coupon_cashflow)
    assert float(np.sum(ecc_term)) != pytest.approx(
        float(np.sum(ecc_coll)), rel=1e-6
    )


def test_ko_reset_quad_sees_term_structure():
    from quantark.asset.equity.engine.quad import KOResetSnowballQuadEngine
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
    env_term = make_term_env("kinked")
    px_term = KOResetSnowballQuadEngine().price(product, env_term)
    px_collapsed = KOResetSnowballQuadEngine().price(
        product, _collapsed_flat_env(env_term, 2.0)
    )
    assert px_term != pytest.approx(px_collapsed, rel=1e-5)


def test_ko_reset_quad_event_stats_see_term_structure():
    from quantark.asset.equity.engine.quad import KOResetSnowballQuadEngine
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
    env_term = make_term_env("kinked")
    env_coll = _collapsed_flat_env(env_term, 2.0)
    engine = KOResetSnowballQuadEngine()
    st_term = engine.calculate_event_stats(product, env_term)
    st_coll = engine.calculate_event_stats(product, env_coll)
    cf_term = float(np.sum(np.asarray(st_term.expected_discounted_ko_cashflow)))
    cf_coll = float(np.sum(np.asarray(st_coll.expected_discounted_ko_cashflow)))
    assert cf_term != pytest.approx(cf_coll, rel=1e-6)


def test_phoenix_mc_pde_quad_agree_on_term_structure():
    """Spec test layer 4 (flagship): all three engine families price the same
    term-structured autocallable within cross-family tolerances. Product is
    the discrete-KI phoenix used by the existing three-engine agreement
    tests (test_ki_probability_definitions.py)."""
    from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
    from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
    from quantark.asset.equity.engine.quad import PhoenixQuadEngine
    from quantark.asset.equity.param import MCParams, PDEParams, QuadParams

    env = make_term_env("kinked")
    product = _standard_phoenix()

    px_quad = PhoenixQuadEngine(params=QuadParams(grid_points=801)).price(
        product, env
    )
    px_pde = PhoenixPDESolver(
        params=PDEParams(grid_size=500, time_steps=500)
    ).price(product, env)
    px_mc = PhoenixMCEngine(
        params=MCParams(num_paths=300_000, time_steps=252, use_qmc=True, seed=7)
    ).price(product, env)

    # deterministic engines: tight (observed term gap ~0.12%)
    assert px_pde == pytest.approx(px_quad, rel=5e-3)
    # MC: the net value (~ -2.5) nets large offsetting legs, so the QMC
    # sampling error is material in RELATIVE terms; bound it absolutely
    # at the observed few-cents scale.
    assert px_mc == pytest.approx(px_quad, rel=5e-3, abs=0.05)


def test_ko_discount_forward_df_on_term_curve():
    """Codex code-review regression: delayed-settlement discounting must be
    the forward curve DF df(settle)/df(obs), not exp(-rate * delay)."""
    from quantark.asset.equity.engine.quad import SnowballQuadEngine
    from quantark.priceenv.term_sampling import make_df_fn

    env = make_term_env("kinked")
    engine = SnowballQuadEngine()
    df = make_df_fn(env)
    rate = env.get_rate(1.0)  # cumulative scalar the engine threads around
    obs, settle = 0.5, 0.75

    got = engine._ko_discount(rate, obs, settle, df_fn=df)
    expected = env.get_discount_factor(settle) / env.get_discount_factor(obs)
    assert got == pytest.approx(expected, rel=1e-14)
    # and it must differ from the flat-rate delay discount on a kinked curve
    assert got != pytest.approx(np.exp(-rate * (settle - obs)), rel=1e-8)


def test_snowball_continuous_ki_bridge_uses_term_vol():
    """Codex code-review regression: the continuous-KI Brownian-bridge legs
    must run with per-step vol. The PRICE path is the meaningful check —
    QUAD's continuous-KI event probabilities carry a pre-existing
    definitional gap that is explicitly out of scope (see memory/spec), so
    the event-stats path is exercised for execution + bounds only."""
    from quantark.asset.equity.engine.quad import SnowballQuadEngine
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig
    from quantark.asset.equity.product.option.snowball_option import SnowballOption
    from quantark.util.enum import ObservationType

    def _product():
        return SnowballOption(
            initial_price=100.0,
            strike=100.0,
            barrier_config=BarrierConfig(
                ko_barrier=1.03,
                ko_rate=0.15,
                ko_observation_type=ObservationType.DISCRETE,
                ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
                ki_barrier=0.90,  # close enough to bite
                ki_observation_type=ObservationType.CONTINUOUS,
            ),
            payoff_config=None,
            contract_multiplier=1.0,
            maturity=1.0,
            is_reverse=False,
        )

    env_term = make_term_env("kinked")
    env_coll = _collapsed_flat_env(env_term, 1.0)
    quad = SnowballQuadEngine()

    # price path: bridge diffusion with per-step vol must discriminate
    px_term = quad.price(_product(), env_term)
    px_coll = quad.price(_product(), env_coll)
    assert px_term != pytest.approx(px_coll, rel=1e-5)

    # event-stats path: the fixed KI-ever bridge calls execute with the
    # per-step vol binding and produce bounded probabilities
    st = quad.calculate_event_stats(_product(), env_term)
    # Bounded probabilities. The KI-ever bridge subtracts near-equal survival
    # terms, so on some architectures the result underflows to roundoff-scale
    # negative noise (~-1e-46) rather than exactly 0 -> allow a tiny margin.
    assert -GOLDEN_ABS_TOL <= float(st.ki_ever_probability) <= 1.0
    assert -GOLDEN_ABS_TOL <= float(st.ki_probability) <= 1.0
