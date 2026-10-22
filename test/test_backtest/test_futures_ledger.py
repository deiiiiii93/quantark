"""Tests for the shared futures ledger (plan Task 5)."""
from __future__ import annotations

import pandas as pd
import pytest

from quantark.backtest.futures_ledger import FuturesHedgePosition, FuturesRollPolicy
from quantark.util.exceptions import ValidationError


class TestFuturesHedgePosition:
    def test_average_cost_accumulation(self):
        pos = FuturesHedgePosition()
        pos.trade(2.0, 100.0, "IF2401", 300.0)
        pos.trade(1.0, 106.0, "IF2401", 300.0)
        assert pos.quantity == 3.0
        assert pos.avg_price == pytest.approx(102.0)
        assert pos.realized_pnl == 0.0

    def test_partial_close_realizes_pnl(self):
        pos = FuturesHedgePosition()
        pos.trade(2.0, 100.0, "IF2401", 300.0)
        pos.trade(-1.0, 110.0, "IF2401", 300.0)
        assert pos.quantity == 1.0
        assert pos.realized_pnl == pytest.approx(1.0 * 10.0 * 300.0)
        assert pos.avg_price == pytest.approx(100.0)

    def test_flip_resets_avg_to_trade_price(self):
        pos = FuturesHedgePosition()
        pos.trade(1.0, 100.0, "IF2401", 300.0)
        pos.trade(-3.0, 110.0, "IF2401", 300.0)
        assert pos.quantity == -2.0
        assert pos.avg_price == pytest.approx(110.0)
        assert pos.realized_pnl == pytest.approx(10.0 * 300.0)

    def test_full_close_clears_contract(self):
        pos = FuturesHedgePosition()
        pos.trade(2.0, 100.0, "IF2401", 300.0)
        pos.trade(-2.0, 95.0, "IF2401", 300.0)
        assert pos.quantity == 0.0
        assert pos.contract is None
        assert pos.realized_pnl == pytest.approx(-2.0 * 5.0 * 300.0)

    def test_cross_contract_trade_raises(self):
        pos = FuturesHedgePosition()
        pos.trade(1.0, 100.0, "IF2401", 300.0)
        with pytest.raises(ValidationError):
            pos.trade(1.0, 100.0, "IF2402", 300.0)

    def test_mark_to_market_includes_realized(self):
        pos = FuturesHedgePosition()
        pos.trade(2.0, 100.0, "IF2401", 300.0)
        pos.trade(-1.0, 110.0, "IF2401", 300.0)
        assert pos.mark_to_market(105.0) == pytest.approx(
            10.0 * 300.0 + 1.0 * 5.0 * 300.0
        )


def _slice(day: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "contract": "IF2401",
                "futures_price": 100.0,
                "expiry_date": pd.Timestamp("2024-01-19"),
                "multiplier": 300.0,
            },
            {
                "contract": "IF2402",
                "futures_price": 101.0,
                "expiry_date": pd.Timestamp("2024-02-16"),
                "multiplier": 300.0,
            },
        ]
    ).assign(date=pd.Timestamp(day))


class TestFuturesRollPolicy:
    def test_keeps_current_contract_far_from_expiry(self):
        policy = FuturesRollPolicy(roll_days_before_expiry=5)
        row = policy.select_contract(
            _slice("2024-01-05"), pd.Timestamp("2024-01-05"), "IF2401"
        )
        assert row["contract"] == "IF2401"

    def test_rolls_near_expiry(self):
        policy = FuturesRollPolicy(roll_days_before_expiry=5)
        row = policy.select_contract(
            _slice("2024-01-16"), pd.Timestamp("2024-01-16"), "IF2401"
        )
        assert row["contract"] == "IF2402"

    def test_negative_roll_days_rejected(self):
        with pytest.raises(ValidationError):
            FuturesRollPolicy(roll_days_before_expiry=-1)
