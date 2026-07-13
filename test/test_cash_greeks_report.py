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
