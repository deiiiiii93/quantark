"""Follow-ups to the accrual-time final review (M1-M5) and its two "declined to judge" questions."""
from datetime import timedelta

import pytest

from quantark.asset.equity.product.option.snowball_config import AccrualConfig
from quantark.util.enum import PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

from .conftest import T0, dates, ko_reset


# -- M2: a REBASED knocked-in tenor hangs from the knock-in too -------------------------------------------------------

def test_a_rebased_annualized_knock_in_tenor_needs_dated_knock_ins_to_have_an_initial_date():
    """Annualized KI on a REBASED contract reads its knock-in's accrual (spec §4.4): refused at construction (R5),
    not at V1 pricing. settlement_date lets the legacy _validate_maturity_parameters rule through."""
    with pytest.raises(ValidationError, match="R5"):
        ko_reset(M.REBASED, pre=dates(91, 182), ki=dates(91, 182, level=75.0),
                 accrual=AccrualConfig(is_annualized=False, is_annualized_ki=True), maturity=(182 + 270) / 365,
                 settlement_date=T0 + timedelta(days=452))
