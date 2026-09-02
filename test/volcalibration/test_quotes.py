from datetime import date

import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.quotes import ExpiryQuotes, IvNode, QuoteSet


def _expiry(T, label="2026-07-31", expiry_date="2026-07-31"):
    return ExpiryQuotes(
        expiry_label=label,
        expiry_date=expiry_date,
        T=T,
        forward=6000.0,
        discount_factor=0.99,
        r=0.02,
        q=0.01,
        nodes=(IvNode(5800.0, 0.21, 1.0), IvNode(6000.0, 0.20, 1.0)),
        diagnostics={"n_pairs": 7.0},
    )


def test_quoteset_orders_and_validates():
    qs = QuoteSet(
        trade_date=date(2026, 7, 6),
        spot=6000.0,
        convention="listed_strike",
        expiries=(_expiry(0.1), _expiry(0.35)),
    )
    assert [e.T for e in qs.expiries] == [0.1, 0.35]
    assert qs.expiries[0].nodes[0].strike == 5800.0


def test_quoteset_rejects_non_increasing_maturities():
    with pytest.raises(ValidationError, match="strictly increasing"):
        QuoteSet(
            trade_date=date(2026, 7, 6),
            spot=6000.0,
            convention="listed_strike",
            expiries=(_expiry(0.35), _expiry(0.1)),
        )


def test_expiry_date_is_optional_but_label_is_not():
    e = ExpiryQuotes(
        expiry_label="3M",
        expiry_date=None,
        T=0.25,
        forward=7.1,
        discount_factor=0.995,
        r=0.02,
        q=0.03,
        nodes=(IvNode(7.0, 0.08, 1.0),),
        diagnostics={},
    )
    assert e.expiry_label == "3M"
    assert e.expiry_date is None
    with pytest.raises(ValidationError, match="expiry_label"):
        ExpiryQuotes(
            expiry_label="",
            expiry_date=None,
            T=0.25,
            forward=7.1,
            discount_factor=0.995,
            r=0.02,
            q=0.03,
            nodes=(IvNode(7.0, 0.08, 1.0),),
            diagnostics={},
        )


def test_expiry_rejects_non_positive_iv():
    with pytest.raises(ValidationError, match="iv"):
        ExpiryQuotes(
            expiry_label="x",
            expiry_date=None,
            T=0.25,
            forward=7.1,
            discount_factor=0.995,
            r=0.02,
            q=0.03,
            nodes=(IvNode(7.0, 0.0, 1.0),),
            diagnostics={},
        )
