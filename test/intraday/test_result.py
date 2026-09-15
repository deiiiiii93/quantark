import json
from datetime import datetime

import pytest

from quantark.intraday.events import EventPhase
from quantark.intraday.result import CashflowComponent, GreekValue, IntradayValuationResult
from quantark.util.exceptions import ValidationError
from intraday.conftest import SHANGHAI


def _res(**kw):
    base = dict(price=1.5, contingent_pv=1.0, pending_receivable_pv=0.5, paid_cash=0.2, units="price_per_contract",
                valuation_timestamp=datetime(2026, 9, 15, 15, 0, tzinfo=SHANGHAI), phase=EventPhase.AFTER, provisional=True,
                assumptions=(), continuous_assumption=None, lifecycle={"alive": True}, cashflows=(), greeks=(),
                profile_identity=("desk", "1"), session_identity=("SSE",), market_snapshot_id="m", context_identity="c",
                engine="X", method="y", numerical={}, records=())
    base.update(kw)
    return IntradayValuationResult(**base)


def test_greek_value_invariants():
    GreekValue("delta", 0.4, "per unit spot", "point")
    with pytest.raises(ValidationError):
        GreekValue("delta", None, "per unit spot", "point")                     # ok needs a value
    with pytest.raises(ValidationError):
        GreekValue("delta", float("nan"), "per unit spot", "point")
    u = GreekValue("delta", None, "per unit spot", "point", status="undefined",
                   reason="payoff discontinuity at the unfixed KO barrier")
    assert u.value is None and u.status == "undefined"
    with pytest.raises(ValidationError):
        GreekValue("delta", None, "x", "point", status="undefined")            # reason required


def test_price_components_must_reconcile_and_total_value_is_explicit():
    r = _res()
    assert r.total_value(include_paid_cash=False) == 1.5 and r.total_value(include_paid_cash=True) == pytest.approx(1.7)
    with pytest.raises(ValidationError, match="reconcile"):
        _res(price=1.4)
    with pytest.raises(ValidationError):
        r.greek("vega")


def test_to_dict_is_json_serialisable():
    r = _res(greeks=(GreekValue("delta", 0.4, "per unit spot", "point"),),
             cashflows=(CashflowComponent("paid", 0.2, "confirmed", cashflow_id="ko"),))
    s = json.dumps(r.to_dict())
    assert '"2026-09-15T15:00:00+08:00"' in s and '"after"' in s
