"""CashGreeksReport tests (spec WP2.1): conventions and CRN policy."""
import pytest

from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.riskmeasures.greek_conventions_report import (
    build_cash_greeks_report,
)

from dcn_fixtures import DCN_A, FLAT, SSE, flat_env, make_dcn

PATHS = 2 ** 14


def test_crn_zero_bump_reproduces_base_pv():
    # CRN sanity: repricing with the same engine/seed reproduces PV bitwise
    p = make_dcn(DCN_A)
    env = flat_env(**FLAT)
    e = DCNMCEngine(num_paths=PATHS, seed=42)
    assert e.price(p, env) == e.price(p, env)


def test_report_fields_and_metadata():
    p = make_dcn(DCN_A)
    env = flat_env(**FLAT)
    e = DCNMCEngine(num_paths=PATHS, seed=42)
    rep = build_cash_greeks_report(p, env, e, calendar=SSE)
    d = rep.to_dict()
    for key in ("pv", "delta_cash", "gamma_cash_1pct", "vega_1pct",
                "theta_1d", "rho_1pct", "rhoq_1pct", "bump_metadata"):
        assert key in d
    assert d["bump_metadata"]["spot"]["style"] == "central"
    assert d["bump_metadata"]["spot"]["seed_policy"] == "common_random_numbers"
    assert d["bump_metadata"]["theta"]["contract_dates"] == "fixed"


def test_delta_sign_positive_for_buyer_dcn():
    # With q >> r the KI loss leg dominates: higher spot -> smaller expected
    # loss, so buyer delta must be positive. Directional sanity, not a level.
    p = make_dcn(DCN_A)
    env = flat_env(**FLAT)
    rep = build_cash_greeks_report(
        p, env, DCNMCEngine(num_paths=PATHS, seed=42), calendar=SSE
    )
    assert rep.delta_cash > 0.0


def test_rhoq_metadata_holds_r_fixed():
    p = make_dcn(DCN_A)
    rep = build_cash_greeks_report(
        p, flat_env(**FLAT), DCNMCEngine(num_paths=PATHS, seed=42), calendar=SSE
    )
    assert rep.bump_metadata["rhoq"]["holds_fixed"] == "r"
    # q up => forward down => more KI loss => buyer PV falls: rhoq < 0
    assert rep.rhoq_1pct < 0.0


def test_theta_requires_calendar_and_spec():
    from quantark.util.exceptions import ValidationError
    p = make_dcn(DCN_A)
    with pytest.raises(ValidationError):
        build_cash_greeks_report(
            p, flat_env(**FLAT), DCNMCEngine(num_paths=PATHS, seed=42),
            calendar=None,
        )


def test_unified_greeks_entry_point_accepts_dcn():
    # review regression: DCNOption implements the BaseEquityProduct contract
    # so the unified GreeksCalculator entry point must work
    from quantark.asset.equity.riskmeasures.greeks_calculator import (
        GreeksCalculator,
    )

    p = make_dcn(DCN_A)
    env = flat_env(**FLAT)
    engine = DCNMCEngine(num_paths=2 ** 12, seed=42)
    assert p.is_linear is False
    assert p.get_maturity(env) == pytest.approx(731 / 365.0)
    out = GreeksCalculator().calculate(
        p, env, engine, greeks=["price", "delta"]
    )
    assert "delta" in out and "price" in out


def test_vega_presentation_scales_nondefault_bump_to_one_vol_point():
    class LinearVolEngine:
        def price(self, product, env):
            return 1_000_000.0 * env.vol_surface.get_vol(
                product.initial_price, 1.0, env.spot
            )

    product = make_dcn(DCN_A)
    env = flat_env(**FLAT)
    default = build_cash_greeks_report(
        product, env, LinearVolEngine(), calendar=SSE, vol_abs_bump=0.01
    )
    half_bump = build_cash_greeks_report(
        product, env, LinearVolEngine(), calendar=SSE, vol_abs_bump=0.005
    )
    assert default.vega_1pct == pytest.approx(10_000.0)
    assert half_bump.vega_1pct == pytest.approx(default.vega_1pct)
