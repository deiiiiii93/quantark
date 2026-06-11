"""Slow smoke checks for Snowball RFQ KO-rate consistency across engines."""

from __future__ import annotations

import pytest

from example.snowball_rfq_ko_rate_engine_compare_demo import (
    DEFAULT_SMOKE_CASES,
    build_engines,
    run_case,
)
from example.snowball_rfq_ko_rate_demo_workflow import solve_fair_ko_rate_with_engine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.util.numerical import almost_equal, is_non_positive


DETERMINISTIC_QUOTE_TOL = 0.008
MC_TO_MID_TOL = 0.010
MAX_SPREAD_TOL = 0.012


@pytest.fixture(scope="module")
def engines():
    """Requested KO-rate smoke configuration.

    50k MC paths leave ~±0.005 sampling error on the solved KO rate, which
    makes the spread tolerance a coin flip; 200k paths bring the MC quote
    within ~±0.001 of its converged value (measured across seeds) so the
    cross-engine checks test the engines, not the noise.
    """
    return build_engines(
        mc_paths=200_000,
        quad_grid=1001,
        pde_grid=400,
        pde_steps=400,
    )


@pytest.mark.slow
@pytest.mark.parametrize("case", DEFAULT_SMOKE_CASES)
def test_snowball_rfq_ko_rate_consistency(case, engines) -> None:
    """Quad/PDE/MC fair KO-rate solves should stay within smoke-check tolerances."""
    row = run_case(case, engines)

    quad_quote = row["quad"]["quoted_ko_rate"]
    pde_quote = row["pde"]["quoted_ko_rate"]
    mc_quote = row["mc"]["quoted_ko_rate"]
    deterministic_mid = 0.5 * (quad_quote + pde_quote)
    max_spread = max(quad_quote, pde_quote, mc_quote) - min(
        quad_quote, pde_quote, mc_quote
    )

    assert almost_equal(quad_quote, pde_quote, tol=DETERMINISTIC_QUOTE_TOL)
    assert almost_equal(mc_quote, deterministic_mid, tol=MC_TO_MID_TOL)
    assert is_non_positive(max_spread - MAX_SPREAD_TOL)


@pytest.mark.slow
@pytest.mark.parametrize(
    "label, case_kwargs",
    [
        ("carry", dict(rate=0.02, div_yield=0.14, vol=0.18, tenor=2.0)),
        ("stress", dict(rate=0.05, div_yield=0.06, vol=0.30, tenor=2.0)),
        ("long", dict(rate=0.04, div_yield=0.09, vol=0.25, tenor=3.0)),
    ],
)
def test_exact_discrete_fair_ko_rate_grid_convergence(label, case_kwargs) -> None:
    """Exact-discrete quad fair KO rates must converge under grid refinement.

    Measured ladders (2026-06-11): the deep-carry case is the slowest to
    converge — at the default 1001-point grid its fair KO rate sits ~38bp
    above the refined cluster (0.631800 vs ~0.6281 at 4001+), because low
    volatility concentrates barrier sensitivity below the spatial resolution
    that the stability-oriented adaptive refinement guarantees. Stress
    (+2.1bp) and long-maturity (+5.3bp) are comfortable at the default.
    The bounds below lock in those measured accuracies; deep-carry RFQ
    work should request grid_points >= 2001 explicitly.
    """
    fair = {}
    for grid in (1001, 2001, 4001):
        engine = SnowballQuadEngine(params=QuadParams(grid_points=grid))
        result = solve_fair_ko_rate_with_engine(
            engine, variant="standard", **case_kwargs
        )
        fair[grid] = result["quoted_ko_rate"]

    assert abs(fair[2001] - fair[4001]) < 2e-3
    assert abs(fair[1001] - fair[4001]) < 5e-3
