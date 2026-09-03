"""PnLExplainConfig validation and stencil resolution (spec §5.5, §7.3)."""
import pytest

from quantark.pnlexplain.base import ExplainMethod, Factor, MARKET_FACTORS
from quantark.pnlexplain.config import (
    PnLExplainConfig, STENCILS, TERM_FACTOR, resolve_stencil, resolved_subrows,
)
from quantark.util.exceptions import ValidationError


def test_defaults():
    cfg = PnLExplainConfig()
    assert cfg.methods == (ExplainMethod.WATERFALL, ExplainMethod.TAYLOR)
    assert cfg.waterfall_order == MARKET_FACTORS
    assert resolve_stencil(cfg) == STENCILS["standard"]
    assert resolved_subrows(cfg) == ("r_theta", "q_theta", "convexity_theta")


@pytest.mark.parametrize("kw", [
    dict(methods=()),
    dict(methods=(ExplainMethod.SHARED,)),
    dict(methods=(ExplainMethod.TAYLOR, ExplainMethod.TAYLOR)),
    dict(waterfall_order=(Factor.TIME, Factor.SPOT)),
    dict(waterfall_order=MARKET_FACTORS + (Factor.TIME,)),
    dict(waterfall_order=(Factor.LIFECYCLE_EVENT,) + MARKET_FACTORS[1:]),
    dict(waterfall_order=tuple(f.value for f in MARKET_FACTORS)),          # strings, not members
    dict(waterfall_order=(ExplainMethod.WATERFALL,) + MARKET_FACTORS[1:]),  # another enum's member
    dict(bucketed="false"),
    dict(bucketed=1),
    dict(interaction="random"),
    dict(time_term="gap"),
    dict(theta_decomposition_mode="approx"),
    dict(greeks_method="closed_form"),
    dict(clock="1td"),                      # clock with exact_gap
    dict(time_term="per_step", clock="2d"),
    dict(stencil="huge"),
    dict(stencil=["delta", "delta"]),
    dict(stencil=["delta", "theta_1td"]),   # clock qualifier rejected
    dict(stencil=["r_theta"]),              # sub-row without theta
    dict(stencil=["bogus"]),
])
def test_invalid_configs_raise(kw):
    with pytest.raises(ValidationError):
        PnLExplainConfig(**kw)


def test_explicit_stencil_resolves_aliases_and_keeps_order():
    cfg = PnLExplainConfig(stencil=["veta", "delta", "rhoq", "theta", "gamma_theta"])
    assert resolve_stencil(cfg) == ("vega_theta", "delta", "dividend_rho", "theta")
    assert resolved_subrows(cfg) == ("gamma_theta",)
    assert PnLExplainConfig(waterfall_order=tuple(reversed(MARKET_FACTORS))).waterfall_order[0] is Factor.MODEL


def test_term_factor_table_covers_every_stencil_term():
    for name in STENCILS["extended"]:
        assert name in TERM_FACTOR
    assert TERM_FACTOR["vanna"] is Factor.VOL and TERM_FACTOR["charm"] is Factor.TIME
