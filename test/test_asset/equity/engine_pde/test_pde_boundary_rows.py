"""Far-field rows from forward discount and carry factors over [t, T] (spec 3A §5.10, decision 13)."""
import json
from math import exp
from pathlib import Path

import pytest

from golden_compare import assert_close
from quantark.asset.equity.engine.pde import AmericanPDESolver, BarrierPDESolver, EuropeanPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option import AmericanOption, BarrierOption, EuropeanVanillaOption
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.util.enum import BarrierType, ObservationType, OptionType

from term_structure_benchmarks import make_term_env

REF = json.loads((Path(__file__).parent / "pde_3a" / "fixtures" / "term_curve_reference.json").read_text())
OBS = [k / 12.0 for k in range(1, 13)]


def _lag(lag):
    return dict(settlement_convention=SettlementConvention(lag=lag, lag_unit=SettlementLagUnit.YEAR_FRACTION)) if lag else {}


def _product(name, lag):
    if name == "european_put":
        return EuropeanVanillaOption(strike=100.0, option_type=OptionType.PUT, maturity=1.0, **_lag(lag))
    return BarrierOption(strike=100.0, option_type=OptionType.PUT, barrier=130.0, barrier_type=BarrierType.UP_OUT,
                         observation_type=ObservationType.DISCRETE, observation_dates=OBS, maturity=1.0, **_lag(lag))


@pytest.mark.parametrize("row", [r for r in REF if r["product"] in ("european_put", "up_and_out_put_130")],
                         ids=lambda r: f"{r['product']}|{r['shape']}|{r['spot']}|{r['lag']}")
def test_puts_meet_their_references(row):
    cls = EuropeanPDESolver if row["product"] == "european_put" else BarrierPDESolver
    g = cls(params=PDEParams()).calculate_greeks(_product(row["product"], row["lag"]),
                                                 make_term_env(row["shape"], spot=row["spot"]))
    pv_tol, delta_tol = (3e-4, 2e-5) if row["product"] == "european_put" else (2e-4, None)
    assert abs(g["price"] - row["reference"][0]) <= pv_tol
    if delta_tol is not None:
        assert abs(g["delta"] - row["reference"][1]) <= delta_tol


def _rows_at(cls, product, env, edge):
    """{t: (row value, edge price)} for every step of a solve: the boundary row as the solver wrote it."""
    seen = {}
    orig = cls.set_boundary_conditions

    def spy(self, grid, x, s, t_idx, tau, prod, e):
        orig(self, grid, x, s, t_idx, tau, prod, e)
        seen[1.0 - tau] = (float(grid[edge, t_idx]), float(s[edge]))

    cls.set_boundary_conditions = spy
    try:
        cls(params=PDEParams()).price(product, env)
    finally:
        cls.set_boundary_conditions = orig
    return seen


def _factors(env, t):
    df = env.get_discount_factor(1.0) / env.get_discount_factor(t)
    cf = exp(-(env.get_div_yield(1.0) * 1.0 - env.get_div_yield(t) * t))
    remaining = exp(-env.get_div_yield(1.0 - t) * (1.0 - t))       # the remaining-time reading
    return df, cf, remaining


@pytest.mark.parametrize("case", ["european_put_lower", "barrier_call_upper"])
def test_the_rows_are_the_forward_formulas(case):
    env = make_term_env("kinked")
    if case == "european_put_lower":
        cls, product, edge = EuropeanPDESolver, _product("european_put", 0.0), 0
        formula = lambda df, cf, s_edge: max(100.0 * df - s_edge * cf, 0.0)          # noqa: E731
    else:
        cls, edge = BarrierPDESolver, -1
        product = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=85.0,
                                barrier_type=BarrierType.DOWN_OUT, observation_type=ObservationType.DISCRETE,
                                observation_dates=OBS, maturity=1.0)
        formula = lambda df, cf, s_edge: max(s_edge * cf - 100.0 * df, 0.0)          # noqa: E731
    seen = _rows_at(cls, product, env, edge)
    t = min(seen, key=lambda u: abs(u - 0.5))                                         # an interior time near 0.5
    row, s_edge = seen[t]
    df, cf, remaining = _factors(env, t)
    assert row == pytest.approx(formula(df, cf, s_edge), rel=1e-14, abs=1e-12)
    assert abs(remaining - cf) > 1e-6                     # the kinked curve separates the two readings
    assert abs(row - formula(df, remaining, s_edge)) > 1e-8


BEFORE_G = json.loads((Path(__file__).parent / "pde_3a" / "fixtures" / "far_field_before_g.json").read_text())


def _unmoved_cases(shape):
    env = make_term_env(shape)
    return env, {
        "european_call": (EuropeanPDESolver, EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL,
                                                                   maturity=1.0)),
        "american_put": (AmericanPDESolver, AmericanOption(100.0, OptionType.PUT, maturity=1.0)),
    }


@pytest.mark.parametrize("shape", ["flat", "up", "down", "kinked"])
def test_calls_and_the_american_put_do_not_move(shape):
    import platform
    env, cases = _unmoved_cases(shape)
    for name, (cls, product) in cases.items():
        got = cls(PDEParams()).price(product, env)
        banked = float.fromhex(BEFORE_G["values"][shape][name])
        assert_close(got, banked)
        if platform.machine() == BEFORE_G["machine"]:
            assert got == banked                           # same machine: exact (F5)


def test_a_flat_barrier_call_moves_by_rounding_only():
    product = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=85.0, barrier_type=BarrierType.DOWN_OUT,
                            observation_type=ObservationType.DISCRETE, observation_dates=OBS, maturity=1.0)
    got = BarrierPDESolver(PDEParams()).price(product, make_term_env("flat"))
    assert_close(got, float.fromhex(BEFORE_G["values"]["flat"]["barrier_call"]))            # F2
