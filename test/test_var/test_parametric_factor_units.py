"""Parametric VaR sensitivities are in the units of the factors they multiply.

The ``vol_change``, ``rate_shift`` and ``div_yield_shift`` factors are absolute changes (0.01 = one vol point / 1 %),
the same shocks historical VaR applies through ``_create_stressed_environment``. An ``EquityPosition``'s Greeks from
the default ``GreeksCalculator`` are per one vol point (vega) and per 1 % (rho, dividend_rho), so parametric VaR
rescales them by 1 / 0.01; a TRS reports rho per unit of r already. Both portfolio modes, product mode's dividend-yield
sensitivity included (it used to run a Black–Scholes proxy that could not import, nor price a Snowball).
"""
import copy

import pandas as pd
import pytest

from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.var import HistoricalVaREngine, VaRConfig
from quantark.var.config import EquityRiskFactorConfig
from test_portfolio.test_equity_portfolio_trade_value import _book
from test_stresstest.test_trs_stress import _make_trs
from test_var.test_var_trade_value import _parametric

ALL_FACTORS = EquityRiskFactorConfig(include_div_yield=True)
H = 1e-5


def _shocked_value(portfolio, factor, shock):
    """The book's value with one factor moved by ``shock``, exactly as historical VaR moves it."""
    shocker = HistoricalVaREngine(VaRConfig())
    scenario = pd.Series({factor: shock})
    pf = copy.deepcopy(portfolio)
    pf.pricing_environments = {u: shocker._create_stressed_environment(env, scenario)
                               for u, env in pf.pricing_environments.items()}
    return pf.get_portfolio_value()


def _slope(portfolio, factor, position_ids):
    pf = copy.deepcopy(portfolio)
    pf.positions = {i: pf.positions[i] for i in position_ids}
    return (_shocked_value(pf, factor, H) - _shocked_value(pf, factor, -H)) / (2.0 * H)


@pytest.mark.parametrize("trade_value", [False, True])
@pytest.mark.parametrize("factor", ["vol_change", "rate_shift", "div_yield_shift"])
def test_a_vanillas_sensitivity_is_its_value_slope_in_the_factor(trade_value, factor):
    pf = _book(trade_value=trade_value)
    pf.positions = {"vanilla": pf.positions["vanilla"]}
    got = _parametric(ALL_FACTORS)._compute_equity_sensitivities(copy.deepcopy(pf))[factor]
    # analytical (product mode) or one-sided bump (trade mode) Greeks against a central slope
    assert got == pytest.approx(_slope(pf, factor, ["vanilla"]), rel=1e-6 if not trade_value else 2e-2)


@pytest.mark.parametrize("trade_value", [False, True])
def test_a_trs_rho_is_already_per_unit_of_rate(trade_value):
    # financing that follows the curve one for one, so a rate_shift moves it (the book's TRS has no flow-through)
    pf = _book(trade_value=trade_value)
    pf.positions = {}
    swap = pf.add_swap_position(product=_make_trs(), quantity=1.0,
                                funding_rate_ref=pf.pricing_environments["IDX"].rate_curve.rate)
    pf.positions = {"trs": pf.positions.pop(swap.position_id)}
    got = _parametric(ALL_FACTORS)._compute_equity_sensitivities(copy.deepcopy(pf))
    assert got["rate_shift"] == pytest.approx(_slope(pf, "rate_shift", ["trs"]), rel=1e-6)
    assert got["div_yield_shift"] == 0.0


@pytest.mark.parametrize("trade_value", [False, True])
def test_the_books_sensitivities_are_the_greeks_in_factor_units(trade_value):
    pf = _book(trade_value=trade_value)
    ref = copy.deepcopy(pf)
    got = _parametric(ALL_FACTORS)._compute_equity_sensitivities(pf)
    gc = GreeksCalculator()
    greeks = {i: ref.position_greeks(p, gc) for i, p in ref.positions.items()}
    options = ("vanilla", "note")
    assert got["vol_change"] == sum(greeks[i]["vega"] / 0.01 for i in options) + greeks["trs"]["vega"]
    assert got["rate_shift"] == sum(greeks[i]["rho"] / 0.01 for i in options) + greeks["trs"]["rho"]
    assert got["div_yield_shift"] == sum(greeks[i]["dividend_rho"] / 0.01 for i in options)


@pytest.mark.parametrize("trade_value", [False, True])
def test_the_notes_sensitivities_track_its_value_slope(trade_value):
    """A bump-Greek Snowball: the one-sided vol-point / 1 bp Greeks agree with the central slope to a few %,
    far inside the 100x the old units were off by."""
    pf = _book(trade_value=trade_value)
    pf.positions = {"note": pf.positions["note"]}
    got = _parametric(ALL_FACTORS)._compute_equity_sensitivities(copy.deepcopy(pf))
    for factor in ("vol_change", "rate_shift", "div_yield_shift"):
        assert got[factor] == pytest.approx(_slope(pf, factor, ["note"]), rel=0.1), factor
