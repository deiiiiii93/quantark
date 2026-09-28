"""Zero-variance QUAD interval == deterministic shift (spec §4.5, §7.6).

A European call is encoded on the core directly: the strike is the final
k_minus (kink), payoff a_terminal*S + b_terminal = S - K above it, zero
below; barriers inert elsewhere.
"""
import numpy as np
import pytest
from golden_compare import GOLDEN_REL_TOL

from quantark.asset.equity.engine.quad.quad_core import QuadCoreInputs, QuadratureCore

SPOT, K = 100.0, 100.0


def _price(times, vols, r=0.02, q=0.01, grid_x=2001):
    core = QuadratureCore(
        grid_x=grid_x, spot=SPOT, observation_times=list(times),
        rate=r, div=q, vol=list(vols),
    )
    n = len(times)
    inputs = QuadCoreInputs(
        observation_times=list(times),
        k_minus=[0.0] * (n - 1) + [K],
        k_plus=[np.inf] * n,
        a_minus=[0.0] * n,
        b_minus=[0.0] * n,
        a_plus=[0.0] * n,
        b_plus=[0.0] * n,
        a_terminal=1.0,
        b_terminal=-K,
    )
    return core.price(inputs)


def test_zero_vol_middle_interval_equals_collapsed_grid():
    """[0.3, 0.35, 0.7] with vol=(0.2, 0.0, 0.2): the middle interval carries
    only drift/discount, so the price equals the two-interval control with the
    same per-interval variances and the same total carry."""
    p_split = _price(times=(0.3, 0.35, 0.7), vols=(0.2, 0.0, 0.2))
    vol_23 = np.sqrt((0.2**2 * 0.35) / 0.4)
    p_ctrl = _price(times=(0.3, 0.7), vols=(0.2, vol_23))
    assert p_split == pytest.approx(p_ctrl, rel=5e-5)


def test_zero_vol_first_interval_equals_collapsed_grid():
    """Valuation inside a holiday block: vol=0 on the first interval."""
    p_split = _price(times=(0.05, 0.7), vols=(0.0, 0.2))
    vol_2 = np.sqrt((0.2**2 * 0.65) / 0.7)
    p_ctrl = _price(times=(0.7,), vols=(vol_2,))
    assert p_split == pytest.approx(p_ctrl, rel=5e-5)


def test_all_positive_vols_unchanged():
    """Regression: with no zero step the refactor must not change the price.

    The frozen value was produced by this exact call on the pre-change tree
    (commit with Tasks 1-3 only; quad_core untouched), on the banking machine:
    it is bitwise there and within GOLDEN_REL_TOL on x86_64 CI, where the
    convolution lands 2 ULP away."""
    a = _price(times=(0.3, 0.7), vols=(0.2, 0.21))
    assert a == pytest.approx(FROZEN_ALL_POSITIVE, rel=GOLDEN_REL_TOL)


FROZEN_ALL_POSITIVE = 7.140157199617561  # pre-change tree (Tasks 1-3 HEAD)


def test_alpha_beta_bitwise_match_original_expressions():
    """The positive-vol slots must evaluate the pre-trading-clock formulas
    VERBATIM: pow(v, 4) and (v*v)**2 differ by an ulp on ~half of realistic
    vols, so any algebraic 'simplification' here breaks the bitwise-frozen
    all-positive path (found via the flat-BSM impact review)."""
    vols = (0.083, 0.2336, 0.2857, 0.31, 1.17)
    times = tuple(0.1 * (i + 1) for i in range(len(vols)))
    core = QuadratureCore(
        grid_x=101, spot=100.0, observation_times=list(times),
        rate=0.02, div=0.01, vol=list(vols),
    )
    v, r, q = core.vol, core.r, core.q
    alpha_ref = (r - q - 0.5 * v * v) / (v * v)
    beta_ref = (r - q - 0.5 * v * v) ** 2 / v**4 + 2.0 * r / v**2
    assert np.array_equal(core.alpha, alpha_ref)
    assert np.array_equal(core.beta, beta_ref)
