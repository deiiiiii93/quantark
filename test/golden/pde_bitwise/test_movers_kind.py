"""movers.py's Phoenix kinds (spec 3B stage 2 (b)): the plain rows are the non-memory INSTANT Phoenix movers,
read off the previous capture's ledger."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import movers  # noqa: E402


def h(x):
    return float(x).hex()


def _row(coupon_cash, coupon_pay, terminal):
    n = len(coupon_pay)
    ko = [h(0.1 * (i + 1)) for i in range(n)]
    return {"stats_ko_times": ko, "stats_expected_discounted_coupon_cashflow": [h(c) for c in coupon_cash],
            "stats_payment_times": ko + [h(t) for t in coupon_pay] + [h(terminal)]}


def test_plain_is_instant_with_coupon_cash():
    capture = {
        "phoenix_plain|a": _row([0.1, 0.1], [0.1, 0.2], 1.0),
        "phoenix_memory|a": _row([], [0.1, 0.2], 1.0),                 # legacy memory: no coupon cash
        "phoenix_expiry|plain": _row([0.1, 0.1], [1.0, 1.0], 1.0),     # EXPIRY: every coupon at the terminal
        "settlement_lag|phoenix": _row([0.1, 0.1], [0.6, 1.1], 1.1),   # the last coupon pays with the redemption
    }
    assert movers.phoenix_kind(capture["phoenix_plain|a"]) == "plain"
    assert movers.phoenix_kind(capture["phoenix_memory|a"]) == "memory"
    assert movers.phoenix_kind(capture["phoenix_expiry|plain"]) == "expiry"
    assert movers.phoenix_kind(capture["settlement_lag|phoenix"]) == "plain"
    assert movers.phoenix_kind({"price": h(1.0)}) is None                # a pwe row carries no statistics
