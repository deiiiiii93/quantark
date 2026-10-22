"""Quote cleaning pipeline tests (spec WP4.1)."""
import json

import numpy as np
import pytest

from quantark.param.vol.marketquotes import (
    QuoteCleaningConfig,
    black_price,
    clean_and_imply,
    implied_vol_black,
)

from dcn_fixtures import synthetic_quote_book


@pytest.fixture(scope="module")
def cleaned():
    quotes, valuation_date, spot, rate_curve, carry_curve = (
        synthetic_quote_book()
    )
    return clean_and_imply(quotes, valuation_date, spot, rate_curve, carry_curve)


def test_clean_quotes_recover_input_vol(cleaned):
    used = cleaned.all_quotes
    assert len(used) == 4  # the four clean OTM quotes
    for q in used:
        # mids sit 1 cent off the exact Black price (finite spread), so the
        # recovered vol is near-exact, not bitwise
        assert q.iv == pytest.approx(0.20, abs=5e-4)


def test_each_bad_quote_excluded_with_reason(cleaned):
    reasons = {(q.strike, q.call_put.lower()): r for q, r in cleaned.exclusions}
    assert reasons[(5200.0, "c")] == "itm_side"
    assert reasons[(6200.0, "c")] == "crossed_or_zero"
    assert reasons[(6300.0, "c")] == "crossed_or_zero"
    assert reasons[(6800.0, "c")] == "wide_spread"
    assert reasons[(5000.0, "p")] == "illiquid"
    assert reasons[(5600.0, "p")] == "above_upper_bound"
    assert reasons[(6100.0, "c")] == "no_price"
    assert len(cleaned.exclusions) == 7


def test_log_moneyness_and_market_data(cleaned):
    (t,) = cleaned.slices.keys()
    f = cleaned.forwards[t]
    assert f == pytest.approx(5800.0)
    assert cleaned.dfs[t] == pytest.approx(np.exp(-0.0356 * t))
    for q in cleaned.slices[t]:
        assert q.log_moneyness == pytest.approx(np.log(q.strike / f))
        assert q.weight > 0.0


def test_call_put_parity_unification():
    # a call and a put at the same strike, both priced from the same sigma:
    # OTM selection keeps exactly one and its IV matches the input sigma
    quotes, valuation_date, spot, rate_curve, carry_curve = (
        synthetic_quote_book()
    )
    cleaned = clean_and_imply(
        quotes, valuation_date, spot, rate_curve, carry_curve
    )
    (t,) = cleaned.slices.keys()
    at_5200 = [q for q in cleaned.slices[t] if q.strike == 5200.0]
    assert len(at_5200) == 1
    assert at_5200[0].source.call_put.lower() in ("p", "put")
    assert at_5200[0].iv == pytest.approx(0.20, abs=5e-4)


def test_allow_last_price_config():
    quotes, valuation_date, spot, rate_curve, carry_curve = (
        synthetic_quote_book()
    )
    cleaned = clean_and_imply(
        quotes, valuation_date, spot, rate_curve, carry_curve,
        config=QuoteCleaningConfig(allow_last_price=True),
    )
    reasons = {q.strike: r for q, r in cleaned.exclusions}
    assert 6100.0 not in reasons  # last-price quote now survives cleaning


def test_iv_round_trip_exact():
    price = black_price(5800.0, 6000.0, 0.25, 0.2, 0.99, True)
    assert implied_vol_black(
        price, 5800.0, 6000.0, 0.25, 0.99, True
    ) == pytest.approx(0.2, abs=1e-10)


def test_to_dict_json_safe(cleaned):
    payload = json.dumps(cleaned.to_dict())
    assert "excluded" in payload and "itm_side" in payload
