from datetime import datetime

import numpy as np
import pytest

from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield, TermStructureDividendYield
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.priceenv import PricingEnvironment, TermMarketContext
from quantark.volmodels.adi_core import HestonSLVADICore
from quantark.volmodels.heston import HestonParams


def _params():
    return HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)


def _env(rate_curve, div_yield):
    return PricingEnvironment(
        rate_curve=rate_curve,
        valuation_date=datetime(2026, 7, 7),
        spot_quote=SpotQuote(100.0),
        vol_surface=FlatVolSurface(0.20),
        div_yield=div_yield,
    )


def _term_context():
    env = _env(
        LinearRateCurve([(0.5, 0.01), (1.0, 0.04)]),
        TermStructureDividendYield(times=[0.5, 1.0], yields=[0.00, 0.03]),
    )
    return TermMarketContext.from_env(env, np.array([0.0, 0.5, 1.0]), 100.0)


def test_flat_market_context_matches_scalar_a1_reaction():
    grid = np.array([0.0, 0.5, 1.0])
    env = _env(FlatRateCurve(0.03), ContinuousDividendYield(0.01))
    ctx = TermMarketContext.from_env(env, grid, 100.0)

    scalar = HestonSLVADICore(100.0, 100.0, 1.0, 0.03, 0.01, _params(), 21, 11, 2)
    term = HestonSLVADICore(
        100.0,
        100.0,
        1.0,
        0.03,
        0.01,
        _params(),
        21,
        11,
        2,
        market_context=ctx,
    )
    U = np.ones((21, 11))

    assert term._A1(U, 0.25) == pytest.approx(scalar._A1(U, 0.25))
    assert term._A1(U, 0.75) == pytest.approx(scalar._A1(U, 0.75))


def test_non_flat_market_context_changes_a1_by_step():
    ctx = _term_context()
    core = HestonSLVADICore(
        100.0,
        100.0,
        1.0,
        0.04,
        0.03,
        _params(),
        21,
        11,
        2,
        market_context=ctx,
    )
    U = np.ones((21, 11))

    early = core._A1(U, 0.25)
    late = core._A1(U, 0.75)

    assert early[10, 5] != pytest.approx(late[10, 5], rel=1e-12, abs=1e-12)


def test_df_to_maturity_uses_market_context_node_ratios():
    ctx = _term_context()
    core = HestonSLVADICore(
        100.0,
        100.0,
        1.0,
        0.04,
        0.03,
        _params(),
        21,
        11,
        2,
        market_context=ctx,
    )

    assert core.df_to_maturity(1.0) == pytest.approx(ctx.df_between(0.0, 1.0))
    assert core.df_to_maturity(0.5) == pytest.approx(ctx.df_between(0.5, 1.0))
    assert core.carry_df_to_maturity(0.5) == pytest.approx(
        ctx.carry_df_between(0.5, 1.0)
    )
