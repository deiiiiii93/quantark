from datetime import datetime

import numpy as np
import pytest

from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield, TermStructureDividendYield
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.priceenv import PricingEnvironment, TermMarketContext
from quantark.util.exceptions import ValidationError


def _env(r_curve=None, q=None, vol=0.20, spot=100.0):
    return PricingEnvironment(
        rate_curve=r_curve or FlatRateCurve(0.03),
        valuation_date=datetime(2026, 7, 7),
        spot_quote=SpotQuote(spot),
        vol_surface=FlatVolSurface(vol),
        div_yield=q or ContinuousDividendYield(0.01),
    )


def test_flat_context_identity_and_carry_node_dfs():
    grid = np.array([0.0, 0.5, 1.0])
    ctx = TermMarketContext.from_env(_env(), grid, ref_strike=100.0)

    assert ctx.t_grid == pytest.approx(grid, abs=0.0)
    assert ctx.fwd_rates == pytest.approx([0.03, 0.03], abs=1e-14)
    assert ctx.fwd_carry == pytest.approx([0.01, 0.01], abs=1e-14)
    assert ctx.step_vols == pytest.approx([0.20, 0.20], abs=1e-14)
    assert ctx.node_dfs == pytest.approx(np.exp(-0.03 * grid), abs=1e-14)
    assert ctx.carry_node_dfs == pytest.approx(np.exp(-0.01 * grid), abs=1e-14)
    assert ctx.step_dfs == pytest.approx(ctx.node_dfs[1:] / ctx.node_dfs[:-1])


def test_df_between_uses_grid_node_ratios():
    grid = np.array([0.0, 0.5, 1.0])
    ctx = TermMarketContext.from_env(
        _env(r_curve=LinearRateCurve([(0.5, 0.02), (1.0, 0.04)])),
        grid,
        ref_strike=100.0,
    )

    assert ctx.df_between(0.0, 1.0) == pytest.approx(ctx.node_dfs[2])
    assert ctx.df_between(0.5, 1.0) == pytest.approx(ctx.node_dfs[2] / ctx.node_dfs[1])


def test_carry_df_between_uses_grid_node_ratios():
    grid = np.array([0.0, 0.5, 1.0])
    env = _env(
        q=TermStructureDividendYield(times=[0.5, 1.0], yields=[0.01, 0.03])
    )
    ctx = TermMarketContext.from_env(env, grid, ref_strike=100.0)

    assert ctx.fwd_carry == pytest.approx([0.01, 0.05], abs=1e-12)
    assert ctx.carry_df_between(0.5, 1.0) == pytest.approx(np.exp(-0.05 * 0.5))


def test_off_grid_df_lookup_is_rejected():
    ctx = TermMarketContext.from_env(_env(), np.array([0.0, 0.5, 1.0]), 100.0)

    with pytest.raises(ValidationError, match="grid node"):
        ctx.df_between(0.25, 1.0)
    with pytest.raises(ValidationError, match="grid node"):
        ctx.carry_df_between(0.0, 0.75)


def test_rebuilding_after_env_replacement_uses_new_curve_objects():
    env = _env()
    grid = np.array([0.0, 1.0])
    base = TermMarketContext.from_env(env, grid, ref_strike=100.0)

    env.rate_curve = FlatRateCurve(0.07)
    env.div_yield = ContinuousDividendYield(0.04)
    bumped = TermMarketContext.from_env(env, grid, ref_strike=100.0)

    assert base.fwd_rates[0] == pytest.approx(0.03)
    assert bumped.fwd_rates[0] == pytest.approx(0.07)
    assert base.fwd_carry[0] == pytest.approx(0.01)
    assert bumped.fwd_carry[0] == pytest.approx(0.04)


def test_rate_carry_only_context_does_not_sample_vol_surface():
    env = _env()

    def _raise_if_sampled(strike, maturity):
        raise AssertionError("vol surface should not be sampled")

    env.get_vol = _raise_if_sampled
    ctx = TermMarketContext.from_env(env, np.array([0.0, 0.5, 1.0]), ref_strike=None)

    assert ctx.step_vols == pytest.approx([0.0, 0.0], abs=0.0)
