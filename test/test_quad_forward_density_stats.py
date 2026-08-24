"""Forward-density event-stats mode (spec 2026-08-24). Battery grows task by task."""
import math

import numpy as np
import pytest
from scipy.stats import norm

from quantark.asset.equity.engine.quad.quad_math import QuadratureMath
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.util.exceptions import ValidationError

# Provisional gate-(c) tolerances; tightened + banked by the Task 9 pilot.
MASS_TOL = 1e-6
MOMENT_RTOL = 1e-4
# Measured floor at grid 2001 is 1.36e-5 and is IDENTICAL when the kinked
# call payoff is integrated against the exact analytic density (i.e. it is
# Simpson-on-kink quadrature error, not a marching defect; contracts to
# 2.9e-7 at grid 8001).
FWD_VALUE_RTOL = 3e-5

SPOT, VOL, R, Q, T = 100.0, 0.2, 0.03, 0.05, 1.0
ALPHA = (R - Q - 0.5 * VOL * VOL) / (VOL * VOL)


def test_event_stats_mode_default_is_stacked():
    assert QuadParams().event_stats_mode == "stacked"


def test_event_stats_mode_accepts_forward_density():
    assert QuadParams(event_stats_mode="forward_density").event_stats_mode == "forward_density"


def test_event_stats_mode_rejects_unknown():
    with pytest.raises(ValidationError):
        QuadParams(event_stats_mode="fwd")


def _math_utils(grid_x=2001):
    return QuadratureMath(
        grid_x=grid_x, spot=SPOT, maturity=T, vol_max=VOL,
        num_std_devs=10, align_log=None, integration_rule="simpson",
        fft_padding_factor=2, fft_filter_alpha=0.0, fft_filter_power=2,
    )


def _march_free_density(engine, mu, n_steps=50):
    dt = T / n_steps
    tau_step = 0.5 * VOL * VOL * dt
    p = engine._forward_seed(mu, tau_step, ALPHA)
    omega, pref = engine._forward_kernel(mu, tau_step, ALPHA)
    p_lr, p_ur, p0 = 0, mu.grid.size - 1, (mu.grid.size - 1) % 2
    for _ in range(n_steps - 1):
        p = engine._diffuse_density(p, mu, omega, pref, p_lr, p_ur, p0)
    return p


def test_forward_density_mass_mean_variance():
    engine = SnowballQuadEngine()
    mu = _math_utils()
    p = _march_free_density(engine, mu)
    mass = engine._density_integral(mu, p)
    mean = engine._density_integral(mu, mu.grid * p)
    var = engine._density_integral(mu, (mu.grid - mean) ** 2 * p)
    m = R - Q - 0.5 * VOL * VOL
    assert abs(mass - 1.0) < MASS_TOL
    # Sign of the drift is the kernel-orientation detector.
    assert mean * m > 0.0
    assert abs(mean - m * T) < MOMENT_RTOL * max(abs(m * T), 1e-3)
    assert abs(var - VOL * VOL * T) / (VOL * VOL * T) < MOMENT_RTOL


def test_forward_density_undiscounted_call_value():
    engine = SnowballQuadEngine()
    mu = _math_utils()
    p = _march_free_density(engine, mu)
    strike = 105.0
    payoff = np.maximum(SPOT * np.exp(mu.grid) - strike, 0.0)
    got = engine._density_integral(mu, payoff * p)
    fwd = SPOT * math.exp((R - Q) * T)
    sig = VOL * math.sqrt(T)
    d1 = (math.log(fwd / strike) + 0.5 * sig * sig) / sig
    want = fwd * norm.cdf(d1) - strike * norm.cdf(d1 - sig)
    assert abs(got - want) / want < FWD_VALUE_RTOL


# --- Forward-vs-stacked parity (Task 5+) ---

from datetime import datetime  # noqa: E402

from quantark.asset.equity.product.option.snowball_config import (  # noqa: E402
    AccrualConfig,
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.product.option.snowball_helpers import (  # noqa: E402
    create_standard_snowball,
)
from quantark.asset.equity.product.option.snowball_option import SnowballOption  # noqa: E402
from quantark.param import (  # noqa: E402
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment  # noqa: E402
from quantark.util.enum import CouponPayType, ObservationType  # noqa: E402

# Forward-vs-stacked parity tolerances at grid 2001: 2x the measured pilot
# deltas, banked in docs/autocall-engine-perf/FORWARD-DENSITY-EVIDENCE-2026-08.md.
# ki deltas are dominated by the STACKED side's hard-mask O(h) noise (the
# forward values are grid-stable and MC-confirmed — see the evidence doc).
KO_PROB_ATOL = 6e-4
KI_PROB_ATOL = 1e-2
CF_RTOL = 3e-3


def _env():
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.05),
        valuation_date=datetime(2026, 6, 30),
    )


def _stats_pair(engine_cls, product, env, grid_points=2001, **extra_params):
    stacked = engine_cls(
        params=QuadParams(grid_points=grid_points, **extra_params)
    ).calculate_event_stats(product, env)
    forward = engine_cls(
        params=QuadParams(grid_points=grid_points,
                          event_stats_mode="forward_density", **extra_params)
    ).calculate_event_stats(product, env)
    return stacked, forward


def _no_ki_snowball():
    # create_standard_snowball defaults a KI barrier in when ki_barrier=None,
    # so the KI-free contract is built directly from the configs.
    n_obs = 23
    return SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.9,
        contract_multiplier=10_000.0,
        barrier_config=BarrierConfig(
            ko_barrier=103.0, ko_rate=0.15,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[(i + 1) * 1.9 / n_obs for i in range(n_obs)],
            ki_barrier=None,
        ),
        payoff_config=PayoffConfig(rebate_rate=0.15, include_principal=False),
        accrual_config=AccrualConfig(
            coupon_pay_type=CouponPayType.INSTANT, is_annualized=True
        ),
    )


def test_forward_matches_stacked_no_ki_snowball():
    stacked, forward = _stats_pair(SnowballQuadEngine, _no_ki_snowball(), _env())
    assert np.max(np.abs(forward.ko_probability - stacked.ko_probability)) < KO_PROB_ATOL
    assert np.max(np.abs(forward.survival_probability - stacked.survival_probability)) < KO_PROB_ATOL
    np.testing.assert_allclose(
        forward.expected_discounted_ko_cashflow,
        stacked.expected_discounted_ko_cashflow,
        rtol=CF_RTOL, atol=1e-4,
    )
    # npv path is shared: pv must be EXACTLY the backward price in both modes.
    assert float(forward.pv).hex() == float(stacked.pv).hex()


def _discrete_ki_snowball():
    return create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0,
        ki_observation_type=ObservationType.DISCRETE, ki_continuous=False,
        ki_observation_dates=[(d + 1) * 1.9 / 96 for d in range(96)],
    )


def test_forward_matches_stacked_discrete_ki():
    stacked, forward = _stats_pair(SnowballQuadEngine, _discrete_ki_snowball(), _env())
    assert np.max(np.abs(forward.ko_probability - stacked.ko_probability)) < KO_PROB_ATOL
    assert abs(forward.ki_probability - stacked.ki_probability) < KI_PROB_ATOL
    assert abs(forward.ki_ever_probability - stacked.ki_ever_probability) < KI_PROB_ATOL


def test_forward_ki_ever_matches_analytic_first_passage():
    product = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.0,
        ko_barrier=1e6,  # unreachable: isolates the first-passage statistic
        ki_barrier=80.0, ko_rate=0.15, num_observations=12,
        contract_multiplier=1.0,
    )
    e = _env()
    forward = SnowballQuadEngine(
        params=QuadParams(grid_points=2001, event_stats_mode="forward_density")
    ).calculate_event_stats(product, e)
    S, B, sig = 100.0, 80.0, 0.20
    m = 0.03 - 0.05 - 0.5 * sig * sig
    T_ = product.get_maturity(e)
    x = math.log(B / S)
    p_touch = norm.cdf((x - m * T_) / (sig * math.sqrt(T_))) + (
        B / S
    ) ** (2.0 * m / (sig * sig)) * norm.cdf((x + m * T_) / (sig * math.sqrt(T_)))
    # Banked: measured 1.11e-5 at grid 2001 (4.45e-5 @1001, 2.78e-6 @4001).
    assert abs(forward.ki_ever_probability - p_touch) < 3e-5


def test_forward_matches_stacked_continuous_ki():
    product = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0,
    )
    stacked, forward = _stats_pair(SnowballQuadEngine, product, _env())
    assert np.max(np.abs(forward.ko_probability - stacked.ko_probability)) < KO_PROB_ATOL
    assert abs(forward.ki_probability - stacked.ki_probability) < KI_PROB_ATOL
    assert abs(forward.ki_ever_probability - stacked.ki_ever_probability) < KI_PROB_ATOL
    assert float(forward.pv).hex() == float(stacked.pv).hex()


def test_forward_matches_stacked_phoenix():
    from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
    from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix

    # memory_coupon=False: the helper's default (memory) phoenix carries NO
    # expected_discounted_coupon_cashflow (path-dependent amount), which would
    # make the allclose below vacuous (empty vs empty).
    product = create_standard_phoenix(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, coupon_barrier=85.0, coupon_rate=0.01,
        num_observations=23, memory_coupon=False,
    )
    stacked, forward = _stats_pair(PhoenixQuadEngine, product, _env())
    assert np.max(np.abs(forward.ko_probability - stacked.ko_probability)) < KO_PROB_ATOL
    assert np.max(np.abs(forward.coupon_probability - stacked.coupon_probability)) < KO_PROB_ATOL
    assert np.asarray(forward.expected_discounted_coupon_cashflow).size == 23
    np.testing.assert_allclose(
        forward.expected_discounted_coupon_cashflow,
        stacked.expected_discounted_coupon_cashflow,
        rtol=CF_RTOL, atol=1e-4,
    )
    assert float(forward.pv).hex() == float(stacked.pv).hex()


def test_forward_matches_stacked_phoenix_memory_coupon_probability():
    from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
    from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix

    # The memory phoenix still exposes coupon_probability; only the cashflow
    # conversion is suppressed (amount is path-dependent under memory).
    product = create_standard_phoenix(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, coupon_barrier=85.0, coupon_rate=0.01,
        num_observations=23,
    )
    stacked, forward = _stats_pair(PhoenixQuadEngine, product, _env())
    assert np.max(np.abs(forward.coupon_probability - stacked.coupon_probability)) < KO_PROB_ATOL
    assert np.asarray(forward.expected_discounted_coupon_cashflow).size == 0
    assert float(forward.pv).hex() == float(stacked.pv).hex()


def test_forward_mass_diagnostic_conserved():
    product = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0,
    )
    engine = SnowballQuadEngine(
        params=QuadParams(grid_points=1001, event_stats_mode="forward_density")
    )
    engine.calculate_event_stats(product, _env())
    # Genuine conservation check: terminal integral of the marched densities
    # plus the absorbed KO mass (stored by the dispatch as a diagnostic; the
    # survival/ko fields cannot test this — survival is DEFINED as
    # 1 - cumulative KO, so any identity built from them is a tautology).
    # Banked: measured ~1.9e-14 for this case (the pilot's worst mass defect,
    # 6.2e-4, is the 96-date discrete-KI case at grid 1001 only).
    assert abs(engine._last_forward_mass_diagnostic - 1.0) < 1e-8


# --- Matrix and contract coverage (Task 8) ---


def test_forward_reverse_snowball_parity():
    product = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=97.0,
        ki_barrier=125.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0, is_reverse=True,
    )
    stacked, forward = _stats_pair(SnowballQuadEngine, product, _env())
    assert np.max(np.abs(forward.ko_probability - stacked.ko_probability)) < KO_PROB_ATOL
    assert abs(forward.ki_probability - stacked.ki_probability) < KI_PROB_ATOL


def test_forward_disable_ko_after_ki_parity():
    product = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0, disable_ko_after_ki=True,
    )
    stacked, forward = _stats_pair(SnowballQuadEngine, product, _env())
    assert np.max(np.abs(forward.ko_probability - stacked.ko_probability)) < KO_PROB_ATOL
    assert abs(forward.ki_probability - stacked.ki_probability) < KI_PROB_ATOL


def test_forward_knocked_in_at_valuation_latches():
    product = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0,
    )
    e = PricingEnvironment(
        spot_quote=SpotQuote(spot=74.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.03),
        div_yield=ContinuousDividendYield(div_yield=0.05),
        valuation_date=datetime(2026, 6, 30),
    )
    stacked, forward = _stats_pair(SnowballQuadEngine, product, e)
    assert forward.ki_probability == 1.0 == stacked.ki_probability
    assert forward.ki_ever_probability == 1.0
    assert np.max(np.abs(forward.ko_probability - stacked.ko_probability)) < KO_PROB_ATOL


def test_forward_r_q_zero_bisection():
    product = _discrete_ki_snowball()
    e = PricingEnvironment(
        spot_quote=SpotQuote(spot=100.0),
        vol_surface=FlatVolSurface(volatility=0.20),
        rate_curve=FlatRateCurve(rate=0.0),
        div_yield=ContinuousDividendYield(div_yield=0.0),
        valuation_date=datetime(2026, 6, 30),
    )
    stacked, forward = _stats_pair(SnowballQuadEngine, product, e)
    assert np.max(np.abs(forward.ko_probability - stacked.ko_probability)) < KO_PROB_ATOL
    assert abs(forward.ki_probability - stacked.ki_probability) < KI_PROB_ATOL


def test_forward_nodal_projection_parity():
    # CELL_AVERAGE is the default every other parity test runs under; this
    # covers the legacy NODAL branch of the forward KO absorption.
    product = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0,
    )
    stacked, forward = _stats_pair(
        SnowballQuadEngine, product, _env(), event_projection="nodal"
    )
    assert np.max(np.abs(forward.ko_probability - stacked.ko_probability)) < KO_PROB_ATOL
    assert abs(forward.ki_probability - stacked.ki_probability) < KI_PROB_ATOL


def test_forward_streams_pruning_contract():
    from quantark.cashleg.event_distribution import EventType

    product = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0,
    )
    ko_only = frozenset({EventType.KO, EventType.MATURITY_NO_KO})
    full_engine = SnowballQuadEngine(
        params=QuadParams(grid_points=1001, event_stats_mode="forward_density")
    )
    full = full_engine.calculate_event_stats(product, _env())
    pruned_engine = SnowballQuadEngine(
        params=QuadParams(grid_points=1001, event_stats_mode="forward_density")
    )
    pruned = pruned_engine.calculate_event_stats(
        product, _env(), streams=ko_only
    )
    # Same-run exact equality on the surviving fields; pruned fields zero.
    assert np.asarray(full.ko_probability).tobytes() == np.asarray(pruned.ko_probability).tobytes()
    assert float(full.pv).hex() == float(pruned.pv).hex()
    assert pruned.ki_probability == 0.0
    assert pruned.ki_ever_probability == 0.0


def test_forward_pwe_npv_bit_equal_to_stacked():
    product = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0,
    )
    e = _env()
    npv_stacked = SnowballQuadEngine(
        params=QuadParams(grid_points=1001)
    ).price_with_events(product, e).npv
    npv_forward = SnowballQuadEngine(
        params=QuadParams(grid_points=1001, event_stats_mode="forward_density")
    ).price_with_events(product, e).npv
    assert float(npv_stacked).hex() == float(npv_forward).hex()


@pytest.mark.slow
def test_forward_matches_mc_event_stats():
    from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
    from quantark.asset.equity.param import MCParams

    product = create_standard_snowball(
        initial_price=100.0, strike=100.0, maturity=1.9, ko_barrier=103.0,
        ki_barrier=75.0, ko_rate=0.15, num_observations=23,
        contract_multiplier=10_000.0,
    )
    e = _env()
    forward = SnowballQuadEngine(
        params=QuadParams(grid_points=2001, event_stats_mode="forward_density")
    ).calculate_event_stats(product, e)
    mc = SnowballMCEngine(
        params=MCParams(num_paths=500_000, time_steps=479, use_qmc=True, seed=7)
    ).calculate_event_stats(product, e)
    # 3-sigma band on a 500k-path binomial proportion is ~0.002 absolute.
    assert np.max(np.abs(forward.ko_probability - mc.ko_probability)) < 4e-3
    assert abs(forward.ki_ever_probability - mc.ki_ever_probability) < 6e-3


def test_ko_reset_ignores_forward_flag():
    # KO-reset keeps its own stacked implementation; the flag is a
    # permission, not an obligation (same-run exact equality).
    from dataclasses import fields as dc_fields

    from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import (
        KOResetSnowballQuadEngine,
    )
    from quantark.asset.equity.product.option import create_ko_reset_snowball
    from quantark.util.enum import PostKOScheduleMode

    product = create_ko_reset_snowball(
        initial_price=100.0, strike=100.0, maturity_pre=1.0,
        maturity_post=2.0, post_ko_mode=PostKOScheduleMode.ABSOLUTE,
        ki_continuous=True,
    )
    e = _env()
    a = KOResetSnowballQuadEngine(
        params=QuadParams(grid_points=1001)
    ).calculate_event_stats(product, e)
    b = KOResetSnowballQuadEngine(
        params=QuadParams(grid_points=1001, event_stats_mode="forward_density")
    ).calculate_event_stats(product, e)
    for f in dc_fields(a):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        if isinstance(va, np.ndarray):
            assert va.tobytes() == vb.tobytes(), f.name
        else:
            assert va == vb, f.name
