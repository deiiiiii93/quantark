import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from cashleg.base_amount import BaseAmount, BaseAmountMode
from util.exceptions import ValidationError


def test_absolute_amount():
    b = BaseAmount(value=1_000_000.0, mode=BaseAmountMode.ABSOLUTE)
    assert b.resolve(position_notional=999.0) == 1_000_000.0


def test_notional_fraction():
    b = BaseAmount(value=0.5, mode=BaseAmountMode.NOTIONAL_FRACTION)
    assert b.resolve(position_notional=2_000_000.0) == 1_000_000.0


def test_margin_fraction():
    b = BaseAmount(
        value=1.0, mode=BaseAmountMode.MARGIN_FRACTION, margin_rate=0.25
    )
    assert b.resolve(position_notional=4_000_000.0) == 1_000_000.0


def test_negative_value_rejected_for_fractions():
    with pytest.raises(ValidationError):
        BaseAmount(value=-0.5, mode=BaseAmountMode.NOTIONAL_FRACTION)


def test_fraction_above_one_rejected():
    with pytest.raises(ValidationError):
        BaseAmount(value=1.5, mode=BaseAmountMode.NOTIONAL_FRACTION)


def test_margin_rate_required_for_margin_mode():
    with pytest.raises(ValidationError):
        BaseAmount(
            value=1.0, mode=BaseAmountMode.MARGIN_FRACTION, margin_rate=0.0
        )
