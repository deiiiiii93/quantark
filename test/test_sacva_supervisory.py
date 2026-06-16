"""SA-CVA supervisory parameter tests (values hand-checked vs MAR50.54-50.77)."""

import pytest
from quantark.sacva.models.enums import CreditQuality
from quantark.sacva.parameters.supervisory import SupervisoryParameters as SP, SACVA_VERSION
from quantark.util.exceptions import ValidationError


def test_constants():
    assert SP.M_CVA_DEFAULT == 1.0
    assert SP.R_HEDGE_DISALLOWANCE == 0.01
    assert "SA-CVA" in SACVA_VERSION


def test_ir_specified_rw_and_corr():
    assert SP.ir_specified_rw(tenor=5.0) == 0.0074
    assert SP.ir_specified_rw(inflation=True) == 0.0111
    assert SP.ir_specified_corr(1.0, 2.0) == 0.91
    assert SP.ir_specified_corr(5.0, 5.0) == 1.0
    assert SP.ir_specified_corr(2.0, inflation_b=True) == 0.40
    assert SP.GAMMA_IR == 0.5


def test_ir_specified_corr_rejects_unknown_equal_tenor():
    with pytest.raises(ValidationError):
        SP.ir_specified_corr(3.0, 3.0)


def test_ir_other_and_fx():
    assert SP.IR_OTHER_RW == 0.0158
    assert SP.IR_OTHER_CORR == 0.40
    assert SP.FX_DELTA_RW == 0.11
    assert SP.GAMMA_FX == 0.6


def test_counterparty_rw_and_gamma():
    assert SP.cpty_rw(1, CreditQuality.IG, sub_bucket="a") == 0.005
    assert SP.cpty_rw(1, CreditQuality.HY_NR, sub_bucket="b") == 0.040
    assert SP.cpty_rw(2, CreditQuality.IG) == 0.050
    assert SP.cpty_rw(8, CreditQuality.IG) == 0.015
    assert SP.cpty_gamma(1, 4) == 0.25
    assert SP.cpty_gamma(7, 3) == 0.0
    assert SP.cpty_gamma(3, 8) == 0.45
    assert SP.cpty_gamma(5, 5) == 1.0


def test_counterparty_rw_strict():
    with pytest.raises(ValidationError):
        SP.cpty_rw(1, CreditQuality.IG)        # missing sub_bucket
    with pytest.raises(ValidationError):
        SP.cpty_rw(99, CreditQuality.IG)       # unknown bucket
    with pytest.raises(ValidationError):
        SP.cpty_rw(2, "IG")                     # unknown quality (not enum)


def test_reference_credit_rw_and_gamma():
    assert SP.refcredit_rw(1) == 0.005
    assert SP.refcredit_rw(8) == 0.020
    assert SP.refcredit_rw(15) == 0.120
    assert SP.refcredit_rw(16) == 0.015
    assert SP.refcredit_rw(17) == 0.050
    assert SP.refcredit_gamma(1, 2) == 0.75            # same quality, sectors 1,2
    assert SP.refcredit_gamma(1, 10) == pytest.approx(0.05)  # diff quality (1,3)=0.10 /2
    assert SP.refcredit_gamma(16, 17) == 0.75
    assert SP.refcredit_gamma(16, 3) == 0.45
    assert SP.refcredit_gamma(15, 3) == 0.0


def test_equity_rw_and_gamma():
    assert SP.equity_delta_rw(1) == 0.55
    assert SP.equity_delta_rw(9) == 0.70
    assert SP.equity_delta_rw(12) == 0.15
    assert SP.equity_vega_rw(1) == 0.78
    assert SP.equity_vega_rw(9) == 1.00
    assert SP.equity_vega_rw(12) == 0.78
    assert SP.equity_vega_rw(13) == 1.00
    assert SP.equity_gamma(1, 2) == 0.15
    assert SP.equity_gamma(12, 13) == 0.75
    assert SP.equity_gamma(12, 3) == 0.45
    assert SP.equity_gamma(11, 3) == 0.0


def test_commodity_rw_and_gamma():
    assert SP.commodity_delta_rw(4) == 0.80
    assert SP.commodity_delta_rw(11) == 0.50
    assert SP.commodity_vega_rw(4) == 1.00
    assert SP.commodity_gamma(1, 2) == 0.20
    assert SP.commodity_gamma(11, 3) == 0.0


def test_unknown_keys_raise():
    with pytest.raises(ValidationError):
        SP.equity_delta_rw(99)
    with pytest.raises(ValidationError):
        SP.commodity_delta_rw(0)
    with pytest.raises(ValidationError):
        SP.refcredit_rw(18)
