"""Greek cells with cash legs folded mirror EquityPosition.get_trade_risk bitwise (ship spec 2026-10-21 AC11, D11d)."""
from datetime import datetime

import pytest

from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.cashleg import FoldMode
from quantark.execution.greeks import (
    TradeState,
    assemble_trade_greeks,
    greek_bump_cells,
    greek_bump_transform,
    run_greek_bump,
)
from quantark.portfolio.equity.position import EquityPosition

from test_cashleg._autocallable_helpers import make_env, make_snowball
from test_cashleg.test_folding import _accrual, _autocallable_leg

GREEKS7 = ["delta", "gamma", "vega", "theta", "rho", "dividend_rho"]


def _legs():
    return [_accrual(), _autocallable_leg((0.5, 1.0))]


def _cells(state, gc):
    values = {}
    for cell in greek_bump_cells(GREEKS7):
        bumped = greek_bump_transform(state, {"bump_id": cell.bump_id})
        values[cell.bump_id] = run_greek_bump(cell.bump_id, state, bumped, gc)
    return values


@pytest.mark.parametrize("mode", [FoldMode.TOTAL, FoldMode.DRILLDOWN])
@pytest.mark.parametrize("quantity", [1.0, -1.0])
def test_folded_cells_are_bitwise_the_folded_position(mode, quantity):
    gc, legs = GreeksCalculator(), _legs()
    pos = EquityPosition(product=make_snowball(), quantity=quantity, entry_price=100.0, underlying="T",
                         engine=SnowballPDESolver(PDEParams()), entry_timestamp=datetime(2024, 1, 1),
                         cash_legs=legs, cash_leg_folding=mode)
    expected = pos.get_trade_risk(make_env(), gc, GREEKS7)
    state = TradeState(product=make_snowball(), pricing_env=make_env(), cash_legs=tuple(legs),
                       engine=SnowballPDESolver(PDEParams()), streams=pos._required_streams(),
                       quantity=quantity, greeks_params=gc.params, fold_mode=mode)
    values = _cells(state, gc)
    got = assemble_trade_greeks(values["base"], values, quantity=quantity, spot=make_env().spot,
                                bump_config=gc._bump_config, requested=set(GREEKS7), fold_mode=mode)
    assert got["total"].keys() == expected.total.keys()
    for key, value in expected.total.items():
        assert got["total"][key] == value, (key, got["total"][key].hex(), value.hex())
    if mode is FoldMode.TOTAL:
        assert got["product"] == {} and got["leg_pvs"] == () and got["leg_greeks"] == ()
        return
    for key, value in expected.product.items():
        assert got["product"][key] == value, (key, got["product"][key].hex(), value.hex())
    assert [pv for (_n, _d, pv) in got["leg_pvs"]] == [lp.pv for lp in expected.leg_pvs.values()]
    for (name, _d, greeks), leg in zip(got["leg_greeks"], legs):
        assert name == str(leg.name)
        assert greeks == expected.leg_greeks[leg.leg_id]


def test_without_a_fold_mode_the_cells_are_unchanged():
    state = TradeState(product=make_snowball(), pricing_env=make_env(), cash_legs=tuple(_legs()),
                       engine=SnowballPDESolver(PDEParams()), streams=None, quantity=1.0,
                       greeks_params=GreeksCalculator().params)
    assert state.fold_mode is None
    out = run_greek_bump("base", state, state, GreeksCalculator())
    assert "total" not in out
