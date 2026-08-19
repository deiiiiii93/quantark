"""Continuously monitored KI under the Local Vol / Heston / SLV Snowball PDEs.

The two-surface solvers apply the KI regime jump once per time step, which is
discrete monitoring at the step width: crossings inside a step are invisible,
the engine knocks in too rarely and the live surface is biased high by
O(sqrt(dt)).  The flat/term BSM solvers repair this with the closed-form
FIRST_PASSAGE correction; the vol-model solvers were gated off it because the
correction was parameterised by GLOBALLY sampled GBM coefficients, which their
crossing dynamics are not.

The correction is barrier-local, though -- it touches a band of a few step
widths either side of the barrier -- so what it actually needs is the dynamics
AT THE BARRIER over one step.  These tests pin that reading with invariants
rather than benchmarks:

* a flat vol surface makes the local-vol dynamics EXACTLY the sampled GBM, so
  the local-vol solver must reproduce the corrected flat-BSM solver;
* zero vol-of-vol with v0 = theta makes Heston the same GBM, so its residual
  must stop scaling like sqrt(dt);
* unit leverage makes SLV Heston.
"""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.pde import (
    HestonSLVSnowballPDESolver,
    HestonSnowballPDESolver,
    LocalVolSnowballPDESolver,
    SnowballPDESolver,
)
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import FlatRateCurve, GridVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface

FLAT_VOL = 0.20


def _flat_env(vol=FLAT_VOL, s0=100.0, r=0.03, q=0.01):
    """A FLAT surface: Dupire local vol is then the constant `vol` everywhere,
    so the local-vol solver's crossing dynamics ARE the sampled GBM."""
    strikes = list(s0 * np.exp(np.linspace(-0.5, 0.5, 9)))
    maturities = list(np.linspace(0.25, 1.0, 4))
    return PricingEnvironment(
        rate_curve=FlatRateCurve(r),
        valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=s0),
        vol_surface=GridVolSurface(
            strikes, maturities, np.full((len(maturities), len(strikes)), vol)
        ),
        div_yield=ContinuousDividendYield(q),
    )


def _continuous_ki_snowball():
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=10_000.0,
        barrier_config=BarrierConfig(
            ko_barrier=105.0,
            ko_rate=0.12,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0,
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        ),
    )


def _degenerate_heston(vol=FLAT_VOL):
    """Zero vol-of-vol with v0 = theta: variance is pinned at vol**2, so the
    spot dynamics are GBM at `vol` and every variance column carries the same
    crossing coefficients."""
    return HestonParams(v0=vol * vol, kappa=2.0, theta=vol * vol, sigma=1e-9, rho=0.0)


def _unit_leverage(s0=100.0):
    strikes = np.array(list(s0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    return LeverageSurface(
        time_grid=np.linspace(0.0, 1.0, 4),
        strike_grid=strikes,
        leverage_grid=np.ones((4, strikes.size)),
    )


def test_local_vol_continuous_ki_matches_the_corrected_flat_bsm_solver():
    """Under a flat surface the two solvers integrate the SAME dynamics, so
    they must agree to solver precision -- they differ only in how the step
    coefficients are assembled (~1e-14 relative).  Anything at the 1e-3 level
    is the missing intra-step crossing mass, not discretisation."""
    product, env = _continuous_ki_snowball(), _flat_env()
    params = PDEParams()

    bsm = SnowballPDESolver(params).price(product, env)
    lv = LocalVolSnowballPDESolver(params).price(product, env)

    assert lv == pytest.approx(bsm, rel=1e-9)


def test_local_vol_continuous_ki_correction_can_be_disabled():
    """NONE is the legacy opt-out on the vol solvers too, and it must agree
    with the uncorrected flat-BSM solver for the same reason."""
    product, env = _continuous_ki_snowball(), _flat_env()
    off = PDEParams(continuous_ki_correction="none")

    bsm = SnowballPDESolver(off).price(product, env)
    lv = LocalVolSnowballPDESolver(off).price(product, env)

    assert lv == pytest.approx(bsm, rel=1e-9)


def test_local_vol_continuous_ki_correction_moves_the_price_down():
    """Sanity on the sign and scale: monitoring the barrier continuously can
    only knock in MORE often than monitoring it at step boundaries, so the
    live surface -- and the snowball built on it -- must be worth less."""
    product, env = _continuous_ki_snowball(), _flat_env()

    corrected = LocalVolSnowballPDESolver(PDEParams()).price(product, env)
    uncorrected = LocalVolSnowballPDESolver(
        PDEParams(continuous_ki_correction="none")
    ).price(product, env)

    assert corrected < uncorrected
    assert (uncorrected - corrected) / uncorrected == pytest.approx(1e-3, rel=0.5)


def test_degenerate_heston_continuous_ki_stops_scaling_like_sqrt_dt():
    """A sqrt(dt) term halves under a 4x refinement; an O(dt) term quarters.
    Uncorrected, refining 100 -> 400 steps moves this price by ~908 (0.09% of
    PV).  With the intra-step crossing mass restored, what is left is the
    scheme's own time error, which is far smaller."""
    product, env = _continuous_ki_snowball(), _flat_env()
    hp = _degenerate_heston()

    coarse = HestonSnowballPDESolver(hp, n_x=200, n_v=40, n_t=100).price(product, env)
    fine = HestonSnowballPDESolver(hp, n_x=200, n_v=40, n_t=400).price(product, env)

    assert abs(coarse - fine) < 300.0


def test_slv_with_unit_leverage_reproduces_heston():
    """Leverage 1 makes SLV Heston exactly, correction included."""
    product, env = _continuous_ki_snowball(), _flat_env()
    hp = _degenerate_heston()

    heston = HestonSnowballPDESolver(hp, n_x=200, n_v=40, n_t=100).price(product, env)
    slv = HestonSLVSnowballPDESolver(
        hp, _unit_leverage(), n_x=200, n_v=40, n_t=100
    ).price(product, env)

    assert slv == pytest.approx(heston, rel=1e-12)


def test_the_crossing_slope_is_not_sampled_inside_the_masked_step():
    """A regression test for the root cause of the 2-D divergence.

    The correction models ``V1 - V0`` as ``lambda * (y - barrier)`` near the
    barrier and reads ``lambda`` off one node.  But the nodal mask has just
    set that difference to zero on the breached side, so ON THE GRID it steps
    from 0 below the barrier to its full size above.  A node sitting a
    fraction of a cell above the barrier therefore reports the STEP HEIGHT
    over a near-zero distance, and ``lambda`` diverges -- which is exactly
    what the ADI core's concentrated grid produced, since (unlike every 1-D
    layout) it does not pin the KI barrier to a node.

    The correction moves the live surface toward the knocked-in one by the
    probability of touching and returning, so it can never move it further
    than the difference between them.
    """
    from quantark.asset.equity.engine.pde.snowball_pde_solver import (
        _ContinuousKIFirstPassage,
    )

    barrier, dx, dt = 75.0, 0.00686, 0.01
    sig2 = 0.00726 ** 2  # a low-variance column: one step moves far less than a cell
    jump = -2.0e5  # V1 < V0 for a snowball, and the mask leaves the full jump

    def correction(offset_in_cells):
        a = offset_in_cells * dx + dx * np.arange(-5, 20, dtype=float)
        s_vec = barrier * np.exp(a)
        d = np.where(a > 0.0, jump, 0.0)
        fp = _ContinuousKIFirstPassage(
            dt=np.array([dt]),
            mu=np.array([0.02 - 0.5 * sig2]),
            sig2=np.array([sig2]),
            is_reverse=False,
        )
        return fp.step_correction(0, s_vec, barrier, d)

    hugging = correction(0.01)
    assert np.max(np.abs(hugging)) <= abs(jump)

    # A barrier-aligned grid is the 1-D case and must be unaffected by the floor.
    aligned = correction(1.0)
    assert np.max(np.abs(aligned)) <= abs(jump)


def _skewed_env(s0=100.0, r=0.03, q=0.01):
    """A real downside skew, so local vol at the KI barrier is well above the
    term vol at the strike -- the only regime that can tell the two apart."""
    strikes = list(s0 * np.exp(np.linspace(-0.6, 0.4, 11)))
    maturities = list(np.linspace(0.25, 1.0, 4))
    grid = np.array(
        [[0.34 - 0.22 * (np.log(k / s0) + 0.6) for k in strikes] for _ in maturities]
    )
    return PricingEnvironment(
        rate_curve=FlatRateCurve(r),
        valuation_date=datetime(2026, 1, 1),
        spot_quote=SpotQuote(spot=s0),
        vol_surface=GridVolSurface(strikes, maturities, grid),
        div_yield=ContinuousDividendYield(q),
    )


def test_local_vol_crossing_coefficients_are_read_at_the_barrier():
    """The crossing happens AT the barrier, so that is where the diffusion
    must be sampled -- not at the reference strike the flat solvers use."""
    from quantark.volmodels.localvol import build_dupire_local_vol

    env = _skewed_env()
    product = _continuous_ki_snowball()
    surface = build_dupire_local_vol(
        env.vol_surface, spot=env.spot, rate_curve=env.rate_curve,
        div_yield=env.get_div_yield,
    )
    solver = LocalVolSnowballPDESolver(PDEParams(), local_vol_surface=surface)
    solver._active_lv_surface = surface

    barrier = 75.0
    t_vec = np.linspace(0.0, 1.0, 13)
    _, sig2 = solver._first_passage_step_coefficients(product, env, t_vec, barrier)

    t_mid = 0.5 * (t_vec[:-1] + t_vec[1:])
    expected = np.asarray(surface.local_vol(np.full(t_mid.shape, barrier), t_mid))
    np.testing.assert_allclose(sig2, expected ** 2, rtol=0, atol=0)

    # and it is genuinely a different number from the strike's local vol
    at_strike = np.asarray(surface.local_vol(np.full(t_mid.shape, 100.0), t_mid))
    assert np.all(expected > at_strike * 1.05)


def test_local_vol_continuous_ki_is_grid_invariant_under_a_skew():
    """The correction exists to remove an O(sqrt(dt)) bias, so once it is in
    place the price must stop depending on how fine the grid is.  Uncorrected,
    this payoff moves ~255 across the accuracy profiles."""
    product, env = _continuous_ki_snowball(), _skewed_env()

    def spread(mode):
        prices = [
            LocalVolSnowballPDESolver(
                PDEParams(accuracy=acc, continuous_ki_correction=mode)
            ).price(product, env)
            for acc in ("fast", "standard", "high")
        ]
        return max(prices) - min(prices)

    corrected, uncorrected = spread("first_passage"), spread("none")
    assert corrected < 50.0
    assert corrected < uncorrected / 4.0


def test_the_two_dimensional_correction_reads_leverage_at_the_barrier():
    """The ADI operator's log-spot diffusion is ``L(S,t)**2 * v``, so a column's
    crossing variance carries the leverage AT THE BARRIER.  Constant leverage 2
    must therefore quadruple every column's crossing variance relative to
    Heston -- a solver that ignored the surface would build Heston's."""
    product, env = _continuous_ki_snowball(), _flat_env()
    hp = _degenerate_heston()
    kw = dict(n_x=200, n_v=40, n_t=100)
    lev = LeverageSurface(
        time_grid=np.linspace(0.0, 1.0, 4),
        strike_grid=np.array(list(100.0 * np.exp(np.linspace(-0.8, 0.8, 11)))),
        leverage_grid=np.full((4, 11), 2.0),
    )

    heston = HestonSnowballPDESolver(hp, **kw)
    heston.price(product, env)
    slv = HestonSLVSnowballPDESolver(hp, lev, **kw)
    slv.price(product, env)

    np.testing.assert_allclose(
        slv._ki_fp._sig2, 4.0 * heston._ki_fp._sig2, rtol=1e-12, atol=0
    )


# ---------------------------------------------------------------------------
# Phoenix runs the SAME two-surface knock-in dynamic programming on a
# different payoff, and its vol-model solvers were never gated at all: the
# local-vol one applied the correction with the flat solvers' coefficients
# (the term vol at the strike, not the barrier), and the Heston/SLV ones
# never built the state, so it was silently inert.
# ---------------------------------------------------------------------------


def _continuous_ki_phoenix():
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
    from quantark.asset.equity.product.option.phoenix_config import (
        CouponBarrierConfig,
    )
    from quantark.asset.equity.product.option.snowball_config import PayoffConfig
    from quantark.util.calendar.day_counter import DayCountConvention
    from quantark.util.enum import CouponPayType

    return PhoenixOption(
        initial_price=100.0, strike=100.0, maturity=1.0,
        contract_multiplier=10_000.0, is_reverse=False,
        barrier_config=BarrierConfig(
            ko_barrier=105.0, ko_rate=0.12,
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.25, 0.5, 0.75, 1.0],
            ki_barrier=75.0,
            ki_observation_type=ObservationType.CONTINUOUS,
            ki_continuous=True,
        ),
        coupon_config=CouponBarrierConfig(
            coupon_barrier=90.0, coupon_rate=0.02,
            coupon_pay_type=CouponPayType.INSTANT,
            day_count_convention=DayCountConvention.ACT_365,
            memory_coupon=False,
        ),
        payoff_config=PayoffConfig(include_principal=True),
    )


def test_local_vol_phoenix_continuous_ki_is_grid_invariant_under_a_skew():
    """Same property as the snowball: with the crossing read at the barrier
    the price stops depending on the grid.  Sampling the term vol at the
    strike -- which is what this solver did -- leaves most of the bias."""
    from quantark.asset.equity.engine.pde import LocalVolPhoenixPDESolver

    product, env = _continuous_ki_phoenix(), _skewed_env()

    def spread(mode):
        prices = [
            LocalVolPhoenixPDESolver(
                PDEParams(accuracy=acc, continuous_ki_correction=mode)
            ).price(product, env)
            for acc in ("fast", "standard", "high")
        ]
        return max(prices) - min(prices)

    corrected, uncorrected = spread("first_passage"), spread("none")
    assert corrected < uncorrected / 4.0


def test_heston_phoenix_continuous_ki_stops_scaling_like_sqrt_dt():
    """Uncorrected, this solver monitors the KI barrier at the step width."""
    from quantark.asset.equity.engine.pde import HestonPhoenixPDESolver

    product, env = _continuous_ki_phoenix(), _flat_env()
    hp = _degenerate_heston()

    coarse = HestonPhoenixPDESolver(hp, n_x=200, n_v=40, n_t=100).price(product, env)
    fine = HestonPhoenixPDESolver(hp, n_x=200, n_v=40, n_t=400).price(product, env)

    assert abs(coarse - fine) < 300.0


def test_the_two_dimensional_phoenix_correction_reads_leverage_at_the_barrier():
    from quantark.asset.equity.engine.pde import (
        HestonPhoenixPDESolver,
        HestonSLVPhoenixPDESolver,
    )

    product, env = _continuous_ki_phoenix(), _flat_env()
    hp = _degenerate_heston()
    kw = dict(n_x=200, n_v=40, n_t=100)
    lev = LeverageSurface(
        time_grid=np.linspace(0.0, 1.0, 4),
        strike_grid=np.array(list(100.0 * np.exp(np.linspace(-0.8, 0.8, 11)))),
        leverage_grid=np.full((4, 11), 2.0),
    )

    heston = HestonPhoenixPDESolver(hp, **kw)
    heston.price(product, env)
    slv = HestonSLVPhoenixPDESolver(hp, lev, **kw)
    slv.price(product, env)

    assert heston._ki_fp is not None, "the 2-D phoenix never built the crossing state"
    np.testing.assert_allclose(
        slv._ki_fp._sig2, 4.0 * heston._ki_fp._sig2, rtol=1e-12, atol=0
    )
