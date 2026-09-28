"""Multi-leg futures ledger, pinned against a frozen single-leg reference.

``STEPS_EXPECTED`` below was recorded by running ``STEPS`` against
``FuturesHedgePosition`` BEFORE the accounting transition was extracted, so
it is an independent record of the historical arithmetic rather than a
re-derivation from the shared kernel.  The prices and quantities are
deliberately non-integer: averaging, closing and flipping are all sensitive
to operation order, and integers would hide a reordered expression.
"""

from __future__ import annotations

import math

import pytest

from quantark.backtest.futures_ledger import FuturesHedgeBook, FuturesHedgePosition
from quantark.util.exceptions import ValidationError

MARK = 4700.55

STEPS = (
    ("open", 2.5, 4703.37, "IF2503", 200.0),
    ("increase", 1.25, 4711.91, "IF2503", 200.0),
    ("partial", -1.75, 4698.13, "IF2503", 200.0),
    ("flip", -3.5, 4721.07, "IF2503", 200.0),
    ("add_short", -0.75, 4717.44, "IF2503", 200.0),
    ("flip_back", 2.75, 4690.29, "IF2503", 200.0),
    ("full_close", -0.5, 4702.61, "IF2503", 200.0),
    ("roll_open", 1.5, 4655.08, "IF2506", 300.0),
    ("roll_add", 0.25, 4661.44, "IF2506", 300.0),
)

# (label, contract, quantity, avg_price, multiplier, realized_pnl, mark)
STEPS_EXPECTED = (
    ("open", "IF2503", 2.5, 4703.37, 200.0, 0.0, -1409.9999999998545),
    ("increase", "IF2503", 3.75, 4706.216666666666, 200.0, 0.0, -4249.999999999545),
    (
        "partial",
        "IF2503",
        2.0,
        4706.216666666666,
        200.0,
        -2830.3333333331466,
        -5096.999999999571,
    ),
    ("flip", "IF2503", -1.5, 4721.07, 200.0, 3111.00000000024, 9267.000000000098),
    (
        "add_short",
        "IF2503",
        -2.25,
        4719.86,
        200.0,
        3111.00000000024,
        11800.500000000011,
    ),
    (
        "flip_back",
        "IF2503",
        0.5,
        4690.29,
        200.0,
        16417.50000000011,
        17443.50000000013,
    ),
    ("full_close", None, 0.0, 0.0, 200.0, 17649.50000000008, 17649.50000000008),
    (
        "roll_open",
        "IF2506",
        1.5,
        4655.08,
        300.0,
        17649.50000000008,
        38111.0000000002,
    ),
    (
        "roll_add",
        "IF2506",
        1.75,
        4655.988571428571,
        300.0,
        17649.50000000008,
        41044.25000000038,
    ),
)

# The two most order-sensitive values, as bit patterns.
REALIZED_AT_FULL_CLOSE_HEX = "0x1.13c6000000016p+14"
MARK_AT_FULL_CLOSE_HEX = "0x1.13c6000000016p+14"


# ---------------------------------------------------------------------------
# The frozen reference still holds for the legacy position
# ---------------------------------------------------------------------------


def test_the_single_leg_sequence_is_bit_for_bit_unchanged():
    position = FuturesHedgePosition()
    for step, expected in zip(STEPS, STEPS_EXPECTED):
        _, delta, price, contract, multiplier = step
        position.trade(delta, price, contract, multiplier)
        label, e_contract, e_qty, e_avg, e_mult, e_realized, e_mark = expected
        assert position.contract == e_contract, label
        assert position.quantity == e_qty, label
        assert position.avg_price == e_avg, label
        assert position.multiplier == e_mult, label
        assert position.realized_pnl == e_realized, label
        assert position.mark_to_market(MARK) == e_mark, label


def test_the_most_order_sensitive_values_keep_their_bit_pattern():
    position = FuturesHedgePosition()
    for _, delta, price, contract, multiplier in STEPS[:7]:
        position.trade(delta, price, contract, multiplier)
    assert float(position.realized_pnl).hex() == REALIZED_AT_FULL_CLOSE_HEX
    assert float(position.mark_to_market(MARK)).hex() == MARK_AT_FULL_CLOSE_HEX
    # A fully closed leg keeps its multiplier: a fresh default position would
    # report 1.0 and silently rescale the next mark.
    assert position.contract is None
    assert position.quantity == 0.0
    assert position.multiplier == 200.0


def test_a_negligible_trade_changes_nothing():
    position = FuturesHedgePosition()
    position.trade(2.0, 4700.0, "IF2503", 200.0)
    before = (
        position.contract,
        position.quantity,
        position.avg_price,
        position.multiplier,
        position.realized_pnl,
    )
    position.trade(1e-13, 9999.0, "IF2503", 999.0)
    assert (
        position.contract,
        position.quantity,
        position.avg_price,
        position.multiplier,
        position.realized_pnl,
    ) == before


def test_a_cross_contract_trade_still_raises_on_the_legacy_position():
    position = FuturesHedgePosition()
    position.trade(1.0, 4700.0, "IF2503", 200.0)
    with pytest.raises(ValidationError):
        position.trade(1.0, 4700.0, "IF2506", 200.0)


# ---------------------------------------------------------------------------
# The book reproduces that arithmetic exactly
# ---------------------------------------------------------------------------


def test_the_book_reproduces_the_single_leg_sequence_exactly():
    book = FuturesHedgeBook()
    for step, expected in zip(STEPS, STEPS_EXPECTED):
        _, delta, price, contract, multiplier = step
        book.trade(contract, delta, price, multiplier)
        label, e_contract, e_qty, e_avg, e_mult, e_realized, e_mark = expected
        view = book.single_leg
        assert view.contract == e_contract, label
        assert view.quantity == e_qty, label
        assert view.avg_price == e_avg, label
        assert view.multiplier == e_mult, label
        assert view.realized_pnl == e_realized, label
        marks = {c: MARK for c in book.contracts()} or {"IF2503": MARK}
        assert book.mark_to_market(marks) == e_mark, label


def test_the_books_realized_accumulator_keeps_the_same_bit_pattern():
    book = FuturesHedgeBook()
    for _, delta, price, contract, multiplier in STEPS[:7]:
        book.trade(contract, delta, price, multiplier)
    assert float(book.realized_pnl).hex() == REALIZED_AT_FULL_CLOSE_HEX
    assert float(book.mark_to_market({})).hex() == MARK_AT_FULL_CLOSE_HEX
    assert book.contracts() == ()
    assert book.single_leg.multiplier == 200.0


# ---------------------------------------------------------------------------
# Several legs at once
# ---------------------------------------------------------------------------


def two_leg_book() -> FuturesHedgeBook:
    book = FuturesHedgeBook()
    book.trade("IF2503", -3.0, 4680.0, 200.0)
    book.trade("IF2506", 1.5, 4655.0, 200.0)
    return book


def test_legs_are_independent_and_realized_is_counted_once():
    book = two_leg_book()
    # Close part of each leg, in trade order.
    book.trade("IF2503", 1.0, 4670.0, 200.0)
    first_realized = -1.0 * 1.0 * (4670.0 - 4680.0) * 200.0
    assert book.realized_pnl == pytest.approx(first_realized)
    book.trade("IF2506", -0.5, 4665.0, 200.0)
    second_realized = 1.0 * 0.5 * (4665.0 - 4655.0) * 200.0
    assert book.realized_pnl == pytest.approx(first_realized + second_realized)

    marks = {"IF2503": 4690.0, "IF2506": 4640.0}
    unrealized = (
        -2.0 * (4690.0 - 4680.0) * 200.0 + 1.0 * (4640.0 - 4655.0) * 200.0
    )
    assert book.mark_to_market(marks) == pytest.approx(
        book.realized_pnl + unrealized
    )


def test_interleaved_closes_accumulate_in_trade_order():
    ordered = FuturesHedgeBook()
    for contract, delta, price in (
        ("A", 2.0, 100.0),
        ("B", -1.0, 200.0),
        ("A", -1.0, 110.0),
        ("B", 1.0, 190.0),
    ):
        ordered.trade(contract, delta, price, 10.0)
    assert ordered.realized_pnl == pytest.approx(
        1.0 * (110.0 - 100.0) * 10.0 + 1.0 * (200.0 - 190.0) * 10.0
    )
    assert ordered.contracts() == ("A",)
    assert ordered.quantity("A") == 1.0
    assert ordered.quantity("B") == 0.0


def test_the_book_exposes_gross_and_hedge_only_spot_delta():
    book = two_leg_book()
    prices = {"IF2503": 4680.0, "IF2506": 4655.0}
    assert book.gross() == pytest.approx(4.5)
    assert book.gross_notional(prices) == pytest.approx(
        3.0 * 200.0 * 4680.0 + 1.5 * 200.0 * 4655.0
    )
    spot = 4700.0
    assert book.spot_delta(prices, spot) == pytest.approx(
        (-3.0 * 200.0 * 4680.0 + 1.5 * 200.0 * 4655.0) / spot
    )


def test_the_single_leg_view_refuses_a_multi_leg_book():
    book = two_leg_book()
    with pytest.raises(ValidationError, match="multi-leg"):
        _ = book.single_leg


def test_the_single_leg_view_is_a_snapshot_not_a_handle():
    book = FuturesHedgeBook()
    book.trade("IF2503", 2.0, 4700.0, 200.0)
    view = book.single_leg
    view.quantity = 99.0
    view.realized_pnl = -12345.0
    assert book.quantity("IF2503") == 2.0
    assert book.realized_pnl == 0.0
    assert book.single_leg.quantity == 2.0


# ---------------------------------------------------------------------------
# Fail-closed
# ---------------------------------------------------------------------------


def test_a_missing_mark_on_an_open_leg_raises_before_a_partial_total():
    book = two_leg_book()
    with pytest.raises(ValidationError, match="IF2506"):
        book.mark_to_market({"IF2503": 4680.0})
    with pytest.raises(ValidationError):
        book.mark_to_market({"IF2503": 4680.0, "IF2506": float("nan")})
    with pytest.raises(ValidationError):
        book.gross_notional({"IF2503": 4680.0})
    with pytest.raises(ValidationError):
        book.spot_delta({"IF2503": 4680.0, "IF2506": 4655.0}, 0.0)


def test_a_closed_leg_needs_no_mark():
    book = FuturesHedgeBook()
    book.trade("IF2503", 2.0, 4700.0, 200.0)
    book.trade("IF2503", -2.0, 4710.0, 200.0)
    assert book.mark_to_market({}) == pytest.approx(2.0 * 10.0 * 200.0)
    assert book.contracts() == ()


def test_a_conflicting_multiplier_on_an_open_leg_is_rejected():
    book = FuturesHedgeBook()
    book.trade("IF2503", 2.0, 4700.0, 200.0)
    with pytest.raises(ValidationError, match="multiplier"):
        book.trade("IF2503", 1.0, 4700.0, 300.0)
    # A closed leg may legitimately be reopened at a new contract size.
    book.trade("IF2503", -2.0, 4700.0, 200.0)
    book.trade("IF2503", 1.0, 4700.0, 300.0)
    assert book.single_leg.multiplier == 300.0


def test_invalid_trades_are_rejected_before_any_state_changes():
    book = FuturesHedgeBook()
    book.trade("IF2503", 2.0, 4700.0, 200.0)
    snapshot = (book.quantity("IF2503"), book.realized_pnl)
    for bad in (
        ("", 1.0, 4700.0, 200.0),
        ("IF2506", float("nan"), 4700.0, 200.0),
        ("IF2506", 1.0, float("inf"), 200.0),
        ("IF2506", 1.0, 0.0, 200.0),
        ("IF2506", 1.0, 4700.0, 0.0),
        ("IF2506", 1.0, 4700.0, -200.0),
    ):
        with pytest.raises(ValidationError):
            book.trade(*bad)
    assert (book.quantity("IF2503"), book.realized_pnl) == snapshot
    assert book.contracts() == ("IF2503",)


def test_a_zero_trade_is_a_no_op_on_the_book_too():
    book = FuturesHedgeBook()
    book.trade("IF2503", 2.0, 4700.0, 200.0)
    book.trade("IF2503", 1e-13, 9999.0, 200.0)
    assert book.quantity("IF2503") == 2.0
    assert book.single_leg.avg_price == 4700.0
    # A zero trade never creates a leg either.
    book.trade("IF2506", 0.0, 4655.0, 200.0)
    assert book.contracts() == ("IF2503",)


# ---------------------------------------------------------------------------
# Copies
# ---------------------------------------------------------------------------


def test_a_copy_is_independent_in_both_directions():
    book = two_leg_book()
    trial = book.copy()
    trial.trade("IF2503", 3.0, 4690.0, 200.0)
    trial.trade("IF2509", 2.0, 4600.0, 200.0)
    assert book.quantity("IF2503") == -3.0
    assert book.contracts() == ("IF2503", "IF2506")
    assert trial.contracts() == ("IF2506", "IF2509")
    assert book.realized_pnl == 0.0
    assert trial.realized_pnl != 0.0

    book.trade("IF2506", 1.0, 4660.0, 200.0)
    assert trial.quantity("IF2506") == 1.5


def test_contracts_are_reported_in_a_deterministic_order():
    book = FuturesHedgeBook()
    for contract in ("IF2512", "IF2503", "IF2509", "IF2506"):
        book.trade(contract, 1.0, 4700.0, 200.0)
    assert book.contracts() == ("IF2503", "IF2506", "IF2509", "IF2512")
    # Summation order follows that same sequence, so a mark is reproducible.
    marks = {c: 4700.0 + index for index, c in enumerate(book.contracts())}
    assert book.mark_to_market(marks) == pytest.approx(
        sum((marks[c] - 4700.0) * 200.0 for c in book.contracts())
    )


def test_gross_measures_of_an_empty_book_are_zero():
    book = FuturesHedgeBook()
    assert book.gross() == 0.0
    assert book.gross_notional({}) == 0.0
    assert book.spot_delta({}, 4700.0) == 0.0
    assert book.mark_to_market({}) == 0.0
    assert math.isfinite(book.realized_pnl)
    view = book.single_leg
    assert view.contract is None
    assert view.quantity == 0.0
