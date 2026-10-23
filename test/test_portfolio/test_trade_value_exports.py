"""Final-review fixes (ship spec 2026-10-23 portfolio-trade-value D1/D2, C6).

Snapshots and exports of a trade-value portfolio report every position on the portfolio's basis, so the rows
add up to the totals beside them; a fraction-based accrual on a product whose notional floats with spot is
refused in a multi-day run.
"""
import copy
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from quantark.asset.equity.engine import BlackScholesEngine
from quantark.asset.equity.lifecycle import PortfolioLifecycleManager
from quantark.asset.equity.product import EuropeanVanillaOption
from quantark.asset.equity.riskmeasures import GreeksCalculator
from quantark.cashleg import AccrualLeg, BaseAmount, BaseAmountMode, LegDirection, LegSchedule
from quantark.execution.errors import CapabilityError
from quantark.portfolio import EquityPosition
from quantark.portfolio.portfolio_snapshot import PortfolioSnapshot
from quantark.portfolio.portfolio_storage import PortfolioExporter
from quantark.util.enum import OptionType
from test_portfolio.test_equity_portfolio_trade_value import _book


def _non_swap(pf):
    return {pid: p for pid, p in pf.positions.items() if pid != "trs"}


@pytest.mark.parametrize("flag", [True, False])
def test_snapshot_rows_add_up_to_the_totals(flag):
    pf = _book(trade_value=flag)
    gc = GreeksCalculator()
    snap = PortfolioSnapshot.from_portfolio(pf, gc)
    assert sum(row["market_value"] for row in snap.positions_data) == pytest.approx(snap.total_value, rel=1e-14)
    assert sum(row["pnl"] for row in snap.positions_data) == pytest.approx(snap.total_pnl, rel=1e-14)
    ref = copy.deepcopy(pf)
    for row in snap.positions_data:
        position = ref.positions[row["position_id"]]
        assert row["market_value"] == ref.position_value(position)
        assert row["greeks"] == ref.position_greeks(position, gc, True)


def test_parquet_export_greeks_follow_the_mode(tmp_path):
    pf = _book(trade_value=True)
    gc = GreeksCalculator()
    path = PortfolioExporter(base_path=str(tmp_path)).export_to_parquet(
        pf, filepath=str(tmp_path / "book.parquet"), include_greeks=True, greeks_calculator=gc)
    frame = pd.read_parquet(path).set_index("position_id")
    ref = copy.deepcopy(pf)
    want = ref.position_greeks(ref.positions["note"], gc, True)
    assert frame.loc["note", "delta"] == pytest.approx(want["delta"], rel=1e-14)
    assert frame.loc["note", "dividend_rho"] == pytest.approx(want["dividend_rho"], rel=1e-14)


def test_fraction_based_accrual_on_a_spot_notional_product_is_refused():
    vanilla = EuropeanVanillaOption(strike=100.0, maturity=0.5, option_type=OptionType.CALL)   # no initial_price
    accrual = AccrualLeg(direction=LegDirection.BUYER_RECEIVES, rate=0.04,
                         base=BaseAmount(value=0.5, mode=BaseAmountMode.NOTIONAL_FRACTION),
                         schedule=LegSchedule(period_starts=np.array([0.0]), period_ends=np.array([0.25]),
                                              payment_times=np.array([0.25])))
    pf = _book(trade_value=True)
    pf.positions = {"v": EquityPosition(product=vanilla, quantity=1.0, entry_price=0.0, underlying="X",
                                        engine=BlackScholesEngine(), entry_timestamp=datetime(2026, 3, 16),
                                        position_id="v", cash_legs=[accrual], entry_trade_value=0.0)}
    manager = PortfolioLifecycleManager(base_date=datetime(2026, 3, 16), trade_value=True)
    with pytest.raises(CapabilityError, match="notional"):
        manager.register_leg_books(pf)
