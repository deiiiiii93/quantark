"""Zero-variance QUAD interval == deterministic shift (spec §4.5, §7.6).

A European call is encoded on the core directly: the strike is the final
k_minus (kink), payoff a_terminal*S + b_terminal = S - K above it, zero
below; barriers inert elsewhere.
"""
import numpy as np
import pytest

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


def test_all_positive_vols_bitwise_unchanged():
    """Regression: with no zero step the refactor must not change a single bit.

    The frozen value was produced by this exact call on the pre-change tree
    (commit with Tasks 1-3 only; quad_core untouched)."""
    a = _price(times=(0.3, 0.7), vols=(0.2, 0.21))
    assert a == FROZEN_ALL_POSITIVE


FROZEN_ALL_POSITIVE = 7.140157199617561  # pre-change tree (Tasks 1-3 HEAD)
