"""TermCoefficients: env -> per-interval forward coefficient arrays."""
from datetime import datetime

import numpy as np
import pytest

from quantark.param import SpotQuote
from quantark.param.div.dividend_yield import (
    ContinuousDividendYield,
    TermStructureDividendYield,
)
from quantark.param.rrf.rate_curve import FlatRateCurve
from quantark.param.vol.vol_surface import FlatVolSurface
from quantark.priceenv import PricingEnvironment, TermCoefficients


def make_flat_env(r=0.03, q=0.01, vol=0.20, spot=100.0):
    return PricingEnvironment(
        rate_curve=FlatRateCurve(r),
        valuation_date=datetime(2026, 7, 3),
        spot_quote=SpotQuote(spot),
        vol_surface=FlatVolSurface(vol),
        div_yield=ContinuousDividendYield(q),
    )


def test_flat_identity_every_entry_equals_the_scalar():
    env = make_flat_env()
    grid = np.linspace(0.0, 2.0, 25)
    tc = TermCoefficients.from_env(env, grid, ref_strike=100.0)
    assert tc.fwd_rates == pytest.approx(np.full(24, 0.03), abs=1e-14)
    assert tc.fwd_carry == pytest.approx(np.full(24, 0.01), abs=1e-14)
    assert tc.step_vols == pytest.approx(np.full(24, 0.20), abs=1e-14)
    assert tc.node_dfs == pytest.approx(np.exp(-0.03 * grid), abs=1e-14)
    assert tc.step_dfs == pytest.approx(
        np.exp(-0.03 * np.diff(grid)), abs=1e-14
    )


def test_term_carry_hand_computed():
    env = make_flat_env()
    env.div_yield = TermStructureDividendYield(
        times=[0.5, 1.0], yields=[0.01, 0.02]
    )
    tc = TermCoefficients.from_env(
        env, np.array([0.0, 0.5, 1.0]), ref_strike=100.0
    )
    assert tc.fwd_carry == pytest.approx([0.01, 0.03], abs=1e-12)


def test_shapes_are_consistent():
    tc = TermCoefficients.from_env(
        make_flat_env(), np.array([0.0, 1.0, 2.0]), ref_strike=100.0
    )
    assert tc.t_grid.shape == (3,)
    assert tc.fwd_rates.shape == tc.fwd_carry.shape == tc.step_vols.shape == (2,)
    assert tc.node_dfs.shape == (3,)
    assert tc.step_dfs.shape == (2,)


def test_arrays_are_defensive_copies_and_read_only():
    grid = np.array([0.0, 1.0, 2.0])
    tc = TermCoefficients.from_env(make_flat_env(), grid, ref_strike=100.0)
    grid[1] = 99.0  # mutate caller's grid after construction
    assert tc.t_grid[1] == pytest.approx(1.0)  # no aliasing
    with pytest.raises(ValueError):
        tc.fwd_rates[0] = 0.99  # read-only
    with pytest.raises(ValueError):
        tc.t_grid[0] = -1.0


def test_mismatched_shapes_rejected():
    from quantark.util.exceptions import ValidationError

    with pytest.raises(ValidationError):
        TermCoefficients(
            t_grid=np.array([0.0, 1.0, 2.0]),
            fwd_rates=np.array([0.03]),  # wrong length: expect 2
            fwd_carry=np.array([0.01, 0.01]),
            step_vols=np.array([0.2, 0.2]),
            node_dfs=np.array([1.0, 0.97, 0.94]),
            step_dfs=np.array([0.97, 0.97 / 1.0]),
        )


def _valid_kwargs():
    node_dfs = np.array([1.0, 0.97, 0.94])
    return dict(
        t_grid=np.array([0.0, 1.0, 2.0]),
        fwd_rates=np.array([0.03, 0.03]),
        fwd_carry=np.array([0.01, 0.01]),
        step_vols=np.array([0.2, 0.2]),
        node_dfs=node_dfs,
        step_dfs=node_dfs[1:] / node_dfs[:-1],
    )


def test_public_constructor_enforces_grid_and_df_invariants():
    from quantark.util.exceptions import ValidationError

    kw = _valid_kwargs()
    TermCoefficients(**kw)  # valid baseline constructs

    bad = _valid_kwargs()
    bad["t_grid"] = np.array([0.0, 2.0, 1.0])  # not increasing
    with pytest.raises(ValidationError):
        TermCoefficients(**bad)

    bad = _valid_kwargs()
    bad["fwd_carry"] = np.array([0.01, np.nan])  # non-finite
    with pytest.raises(ValidationError):
        TermCoefficients(**bad)

    bad = _valid_kwargs()
    bad["node_dfs"] = np.array([1.0, -0.97, 0.94])  # non-positive DF
    with pytest.raises(ValidationError):
        TermCoefficients(**bad)

    bad = _valid_kwargs()
    bad["step_dfs"] = np.array([0.5, 0.5])  # inconsistent with node_dfs
    with pytest.raises(ValidationError):
        TermCoefficients(**bad)
