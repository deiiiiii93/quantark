"""SA-CVA position-level portfolio model tests (spec §3.1)."""

import pytest

from quantark.sacva.models.enums import CreditQuality
from quantark.sacva.portfolio.counterparty import Counterparty
from quantark.sacva.portfolio.netting import NettingSet
from quantark.sacva.portfolio.trade import CVAHedge, CVATrade
from quantark.sacva.portfolio.trade_portfolio import CVATradePortfolio
from quantark.util.exceptions import ValidationError


class _Stub:  # stand-in product/engine/env (allowed only in the portfolio-model test)
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _trade(**kw):
    base = dict(trade_id="t1", product=_Stub(), engine=_Stub(), env=_Stub(),
                quantity=1.0, trade_currency="usd")
    base.update(kw)
    return CVATrade(**base)


def test_trade_requires_engine_and_env():
    with pytest.raises(ValidationError):
        _trade(engine=None)
    with pytest.raises(ValidationError):
        _trade(env=None)


def test_trade_zero_quantity_rejected():
    with pytest.raises(ValidationError):
        _trade(quantity=0.0)


def test_trade_nonfinite_quantity_rejected():
    with pytest.raises(ValidationError):
        _trade(quantity=float("nan"))
    with pytest.raises(ValidationError):
        _trade(quantity=float("inf"))


def test_netting_enforceable_must_be_bool():
    with pytest.raises(ValidationError):
        NettingSet(set_id="n1", trades=[_trade()], netting_enforceable="yes")


def test_trade_hedge_flag_must_be_bool():
    with pytest.raises(ValidationError):
        _trade(hedge="yes")


def test_portfolio_hedges_must_be_cvahedges():
    cp = Counterparty(name="A", netting_sets=[NettingSet("n1", [_trade()])],
                      credit_curve=_Stub(), bucket=2, credit_quality=CreditQuality.IG)
    with pytest.raises(ValidationError):
        CVATradePortfolio(counterparties=[cp], hedges=[_trade()])   # plain trade, not a hedge


def test_trade_currency_normalized():
    assert _trade().trade_currency == "USD"


def test_hedge_sets_flag():
    h = CVAHedge(trade_id="h1", product=_Stub(), engine=_Stub(), env=_Stub())
    assert h.hedge is True


def test_netting_set_defaults_enforceable():
    ns = NettingSet(set_id="n1", trades=[_trade()])
    assert ns.netting_enforceable is True
    assert ns.collateralized is False


def test_collateralized_set_rejected_v1():
    with pytest.raises(ValidationError):
        NettingSet(set_id="n1", trades=[_trade()], collateralized=True)


def test_empty_netting_set_rejected():
    with pytest.raises(ValidationError):
        NettingSet(set_id="n1", trades=[])


def test_netting_set_rejects_duplicate_trade_ids():
    with pytest.raises(ValidationError):
        NettingSet(set_id="n1", trades=[_trade(trade_id="dup"), _trade(trade_id="dup")])


def test_counterparty_rejects_duplicate_netting_set_ids():
    with pytest.raises(ValidationError):
        Counterparty(name="A",
                     netting_sets=[NettingSet("n1", [_trade()]),
                                   NettingSet("n1", [_trade(trade_id="t2")])],
                     credit_curve=_Stub(), bucket=2, credit_quality=CreditQuality.IG)


def test_counterparty_requires_credit_curve_and_bucket():
    with pytest.raises(ValidationError):
        Counterparty(name="A", netting_sets=[NettingSet("n1", [_trade()])],
                     credit_curve=None, bucket=2, credit_quality=CreditQuality.IG)
    with pytest.raises(ValidationError):
        Counterparty(name="A", netting_sets=[NettingSet("n1", [_trade()])],
                     credit_curve=_Stub(), bucket="2", credit_quality=CreditQuality.IG)


def test_portfolio_requires_counterparty_and_currency():
    cp = Counterparty(name="A", netting_sets=[NettingSet("n1", [_trade()])],
                      credit_curve=_Stub(), bucket=2, credit_quality=CreditQuality.IG)
    p = CVATradePortfolio(counterparties=[cp], hedges=[], reporting_currency="usd")
    assert p.reporting_currency == "USD"
    with pytest.raises(ValidationError):
        CVATradePortfolio(counterparties=[], hedges=[], reporting_currency="USD")
