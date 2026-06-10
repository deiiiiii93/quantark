"""Slow smoke checks for Snowball RFQ KO-rate consistency across engines."""

from __future__ import annotations

import pytest

from example.snowball_rfq_ko_rate_engine_compare_demo import (
    DEFAULT_SMOKE_CASES,
    build_engines,
    run_case,
)
from quantark.util.numerical import almost_equal, is_non_positive


DETERMINISTIC_QUOTE_TOL = 0.008
MC_TO_MID_TOL = 0.010
MAX_SPREAD_TOL = 0.012


@pytest.fixture(scope="module")
def engines():
    """Requested KO-rate smoke configuration."""
    return build_engines(
        mc_paths=50_000,
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
