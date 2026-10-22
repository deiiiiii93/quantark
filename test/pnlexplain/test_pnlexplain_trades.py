"""ExplainTrade schema (spec §9)."""
import pytest

from quantark.pnlexplain.equity.trades import TRADE_KINDS, ExplainTrade
from quantark.util.exceptions import ValidationError


def test_trade_cash_and_from_contracts():
    t = ExplainTrade("h", 2.0, 10.5, transaction_cost=0.25, kind="adjust")
    assert t.cash == pytest.approx(-21.0)
    close = ExplainTrade.from_contracts("hedge:IF2401", contracts=-1.0, price=103.3, multiplier=300.0,
                                        kind="roll_close")
    assert close.quantity == -300.0 and close.cash == pytest.approx(300.0 * 103.3)
    assert close.kind == "roll_close" and "roll_open" in TRADE_KINDS


@pytest.mark.parametrize("kw", [
    dict(quantity=0.0), dict(quantity=float("nan")), dict(price=float("inf")),
    dict(transaction_cost=-1.0), dict(kind="swap"), dict(position_id=""),
])
def test_invalid_trades_raise(kw):
    base = dict(position_id="a", quantity=1.0, price=1.0)
    base.update(kw)
    with pytest.raises(ValidationError):
        ExplainTrade(**base)


def test_from_contracts_rejects_bad_multiplier():
    with pytest.raises(ValidationError):
        ExplainTrade.from_contracts("a", 1.0, 1.0, multiplier=0.0)


def test_trade_numeric_contracts():
    from quantark.util.exceptions import NumericalError
    with pytest.raises(ValidationError):
        ExplainTrade("a", "ten", 1.0)
    with pytest.raises(ValidationError):
        ExplainTrade("a", 1.0, None)
    with pytest.raises(NumericalError):
        ExplainTrade("a", 1e200, 1e200)                      # finite inputs, overflowing cash
    with pytest.raises(NumericalError):
        ExplainTrade.from_contracts("a", 1e200, 1.0, multiplier=1e200)
    with pytest.raises(ValidationError):
        ExplainTrade.from_contracts("a", "x", 1.0, multiplier=300.0)
