"""IndexFuturesCurve: implied carry from futures marks (spec tests 1, 2, 9)."""
import math

import pytest

from quantark.util.enum import EquityDividendInputMode, FuturesCarryRiskMode
from quantark.util.exceptions import ValidationError


def test_futures_carry_risk_mode_values():
    assert FuturesCarryRiskMode.MARKET_PRICE.value == "market_price"
    assert FuturesCarryRiskMode.THEORETICAL_CARRY.value == "theoretical_carry"
    assert FuturesCarryRiskMode.IMPLIED_FUTURES_CARRY.value == "implied_futures_carry"


def test_equity_dividend_input_mode_values():
    assert EquityDividendInputMode.FLAT_DIVIDEND.value == "flat_dividend"
    assert EquityDividendInputMode.TERM_DIVIDEND.value == "term_dividend"
