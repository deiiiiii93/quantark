from __future__ import annotations

import numpy as np
import pytest

from quantark.backtest.futures_ledger import FuturesHedgePosition
from quantark.backtest.simulation.hedge import VectorHedgeLedger
from quantark.util.exceptions import ValidationError

MULT = 200.0


def test_a_random_trade_sequence_matches_the_scalar_ledger():
    rng = np.random.default_rng(7)
    n_paths = 6
    ledger = VectorHedgeLedger(n_paths)
    scalars = [FuturesHedgePosition(multiplier=MULT) for _ in range(n_paths)]
    for _ in range(40):
        qty = np.round(rng.normal(0.0, 4.0, n_paths))
        price = 6000.0 * np.exp(rng.normal(0.0, 0.01, n_paths))
        ledger.trade(qty, price, "IM2403", MULT)
        for i, pos in enumerate(scalars):
            pos.trade(float(qty[i]), float(price[i]), "IM2403", MULT)
    assert ledger.quantity == pytest.approx([p.quantity for p in scalars])
    assert ledger.avg_price == pytest.approx([p.avg_price for p in scalars])
    assert ledger.realized_pnl == pytest.approx([p.realized_pnl for p in scalars])
    mark = 6100.0
    assert ledger.mark_to_market(np.full(n_paths, mark)) == pytest.approx(
        [p.mark_to_market(mark) for p in scalars]
    )


@pytest.mark.parametrize("sequence", [
    [(3.0, 100.0), (2.0, 110.0)],                      # add in the same direction
    [(3.0, 100.0), (-1.0, 110.0)],                     # partial close
    [(3.0, 100.0), (-3.0, 110.0)],                     # full close, back to flat
    [(3.0, 100.0), (-5.0, 110.0)],                     # flip through zero
    [(-4.0, 100.0), (-2.0, 90.0), (6.0, 95.0)],        # short, add, close
    [(0.0, 100.0), (2.0, 100.0)],                      # a zero trade is ignored
])
def test_each_ledger_branch_matches(sequence):
    ledger = VectorHedgeLedger(1)
    pos = FuturesHedgePosition(multiplier=MULT)
    for qty, price in sequence:
        ledger.trade(np.array([qty]), np.array([price]), "IM2403", MULT)
        pos.trade(qty, price, "IM2403", MULT)
    assert ledger.quantity[0] == pytest.approx(pos.quantity)
    assert ledger.avg_price[0] == pytest.approx(pos.avg_price)
    assert ledger.realized_pnl[0] == pytest.approx(pos.realized_pnl)
    assert ledger.contract == pos.contract


def test_a_flat_position_marks_at_its_realised_pnl_only():
    ledger = VectorHedgeLedger(2)
    ledger.trade(np.array([2.0, 0.0]), np.array([100.0, 100.0]), "IM2403", MULT)
    ledger.trade(np.array([-2.0, 0.0]), np.array([110.0, 110.0]), "IM2403", MULT)
    assert ledger.mark_to_market(np.array([999.0, 999.0])) == pytest.approx([2 * 10 * MULT, 0.0])


def test_paths_can_hold_different_sizes_of_the_same_contract():
    ledger = VectorHedgeLedger(3)
    ledger.trade(np.array([1.0, -2.0, 0.0]), np.array([100.0, 101.0, 102.0]), "IM2403", MULT)
    assert list(ledger.quantity) == [1.0, -2.0, 0.0]
    assert ledger.avg_price == pytest.approx([100.0, 101.0, 0.0])


def test_close_all_returns_the_trades_it_made():
    ledger = VectorHedgeLedger(2)
    ledger.trade(np.array([3.0, -1.0]), np.array([100.0, 100.0]), "IM2403", MULT)
    traded = ledger.close_all(np.array([110.0, 110.0]), "IM2403", MULT)
    assert traded == pytest.approx([-3.0, 1.0])
    assert list(ledger.quantity) == [0.0, 0.0]


def test_trading_a_different_contract_without_a_roll_fails_closed():
    ledger = VectorHedgeLedger(1)
    ledger.trade(np.array([1.0]), np.array([100.0]), "IM2403", MULT)
    with pytest.raises(ValidationError):
        ledger.trade(np.array([1.0]), np.array([100.0]), "IM2406", MULT)


# --- Task 8: roll selection and the delta target over arrays ---------------

from datetime import date  # noqa: E402

import pandas as pd  # noqa: E402

from quantark.backtest.futures_ledger import FuturesRollPolicy  # noqa: E402
from quantark.backtest.simulation.carry import day_chain  # noqa: E402
from quantark.backtest.simulation.hedge import (  # noqa: E402
    day_active_contract,
    should_rebalance_vector,
    target_contracts_vector,
)
from quantark.backtest.strategy.futures_delta_strategy import AutocallableDeltaHedgeStrategy  # noqa: E402

from .conftest import make_market_path  # noqa: E402


def test_the_roll_policy_picks_the_same_contract_for_every_path():
    mp = make_market_path(n_paths=3, n_days=30, start=date(2024, 1, 2))
    policy = FuturesRollPolicy()
    for d in range(mp.n_days):
        chain = day_chain(mp, d)
        picked = {str(policy.select_contract(chain.frame(i), chain.date, None)["contract"])
                  for i in range(mp.n_paths)}
        assert len(picked) == 1
        code, column = day_active_contract(chain, policy, None)
        assert code == picked.pop() and chain.contracts[column] == code


def test_the_active_contract_is_sticky_until_the_roll_window():
    mp = make_market_path(n_paths=1, n_days=30, start=date(2024, 1, 2))
    policy = FuturesRollPolicy(roll_days_before_expiry=5)
    chain = day_chain(mp, 0)
    assert day_active_contract(chain, policy, None)[0] == "IM2401"
    # inside the window the front contract is dropped
    late = day_chain(mp, list(mp.dates).index(pd.Timestamp("2024-01-16")))
    assert day_active_contract(late, policy, "IM2401")[0] == "IM2402"


@pytest.mark.parametrize("round_contracts", [True, False])
@pytest.mark.parametrize("hedge_ratio, target_delta", [(1.0, 0.0), (0.8, 3.0)])
def test_the_target_matches_the_scalar_strategy(round_contracts, hedge_ratio, target_delta):
    strategy = AutocallableDeltaHedgeStrategy(delta_threshold=0.5, hedge_ratio=hedge_ratio,
                                              target_delta=target_delta, round_contracts=round_contracts)
    net_delta = np.array([-1234.5, 0.0, 987.6, -1.0])
    got = target_contracts_vector(strategy, net_delta, MULT)
    expected = [strategy.target_contracts(product_delta=float(x), product_quantity=1.0,
                                          futures_multiplier=MULT) for x in net_delta]
    assert got == pytest.approx(expected)


def test_the_threshold_matches_the_scalar_strategy():
    strategy = AutocallableDeltaHedgeStrategy(delta_threshold=2.0, hedge_ratio=1.0, target_delta=0.0)
    current = np.array([0.0, 5.0, 5.0])
    target = np.array([1.0, 8.0, 5.5])
    got = should_rebalance_vector(strategy, current, target)
    expected = [strategy.should_rebalance(float(c), float(t)) for c, t in zip(current, target)]
    assert list(got) == expected
