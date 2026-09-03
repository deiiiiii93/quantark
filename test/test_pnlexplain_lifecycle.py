"""LifecycleTransition resolution, validation rules and the event row (spec §8)."""
from copy import deepcopy
from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.lifecycle.events import LifecycleEvent, LifecycleEventType
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.pnlexplain.base import ExplainMethod, Factor, RowKind, ValueBreakdown
from quantark.pnlexplain.equity.lifecycle import (
    LifecycleTransition, event_row, event_summary, resolve_transition,
)
from quantark.pnlexplain.equity.snapshot import ValuationSnapshot
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError

FRI = datetime(2026, 6, 26)
MON = datetime(2026, 6, 29)
ENG = BlackScholesEngine()


def _env(date, spot=100.0):
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(0.2),
        rate_curve=FlatRateCurve(rate=0.03), div_yield=ContinuousDividendYield(div_yield=0.01),
        valuation_date=date)


def _call(m, strike=100.0):
    return EuropeanVanillaOption(strike=strike, option_type=OptionType.CALL, maturity=m)


def _snap(product, date, state=None, engine=ENG):
    return ValuationSnapshot(product, engine, _env(date), date=date, lifecycle_state=state)


def _event(kind, date, **kw):
    return LifecycleEvent(event_type=kind, date=date, spot=100.0, **kw)


def test_changed_is_fingerprint_based_and_ignores_bookkeeping():
    s = AutocallableLifecycleState()
    assert LifecycleTransition(_call(1.0), ENG, s, deepcopy(s)).changed is False
    s_ki = deepcopy(s)
    s_ki.mark_ki(MON)
    assert LifecycleTransition(_call(1.0), ENG, s, s_ki).changed is True
    s_obs = deepcopy(s)
    s_obs.observed_ko_indices.add(4)            # observation bookkeeping is pricing-neutral
    assert LifecycleTransition(_call(1.0), ENG, s, s_obs).changed is False


def test_none_transition_requires_unchanged_state_and_rolled_contract():
    s0, s1 = _snap(_call(1.0), FRI), _snap(_call(1.0 - 3 / 365), MON)
    t = resolve_transition(s0, s1, None, calendar_days=3)
    assert t.product_alive_t1 is s1.product and t.engine_alive_t1 is ENG
    assert t.events == () and t.changed is False
    with pytest.raises(ValidationError):                       # unrolled float maturity
        resolve_transition(s0, _snap(_call(1.0), MON), None, calendar_days=3)
    st0 = AutocallableLifecycleState()
    st1 = deepcopy(st0)
    st1.mark_ki(MON)
    with pytest.raises(ValidationError, match="supply a LifecycleTransition"):
        resolve_transition(_snap(_call(1.0), FRI, st0), _snap(_call(1.0 - 3 / 365), MON, st1),
                           None, calendar_days=3)


def test_supplied_transition_with_changed_state_is_validated():
    st0 = AutocallableLifecycleState()
    st1 = deepcopy(st0)
    st1.mark_ki(MON)
    s0, s1 = _snap(_call(1.0), FRI, st0), _snap(_call(1.0 - 3 / 365), MON, st1)
    ki = _event(LifecycleEventType.KNOCK_IN, MON)
    ok = LifecycleTransition(_call(1.0 - 3 / 365), ENG, st0, st1, events=(ki,))
    assert resolve_transition(s0, s1, ok, calendar_days=3) is ok
    with pytest.raises(ValidationError, match="state_before"):
        resolve_transition(s0, s1, LifecycleTransition(_call(1.0 - 3 / 365), ENG, st1, st1, events=(ki,)),
                           calendar_days=3)
    with pytest.raises(ValidationError, match="state_after"):
        resolve_transition(s0, s1, LifecycleTransition(_call(1.0 - 3 / 365), ENG, st0, st0, events=(ki,)),
                           calendar_days=3)
    with pytest.raises(ValidationError, match="no events"):
        resolve_transition(s0, s1, LifecycleTransition(_call(1.0 - 3 / 365), ENG, st0, st1), calendar_days=3)
    late = _event(LifecycleEventType.KNOCK_IN, MON + timedelta(days=1))
    with pytest.raises(ValidationError, match="outside"):
        resolve_transition(s0, s1, LifecycleTransition(_call(1.0 - 3 / 365), ENG, st0, st1, events=(late,)),
                           calendar_days=3)
    early = _event(LifecycleEventType.COUPON, FRI + timedelta(days=1))
    with pytest.raises(ValidationError, match="chronological"):
        resolve_transition(s0, s1, LifecycleTransition(_call(1.0 - 3 / 365), ENG, st0, st1, events=(ki, early)),
                           calendar_days=3)
    with pytest.raises(ValidationError, match="contract replacement"):
        resolve_transition(s0, s1, LifecycleTransition(_call(1.0 - 3 / 365, strike=90.0), ENG, st0, st1,
                                                       events=(ki,)), calendar_days=3)


def test_supplied_transition_with_unchanged_state_is_validated():
    st = AutocallableLifecycleState()
    s0, s1 = _snap(_call(1.0), FRI, st), _snap(_call(1.0 - 3 / 365), MON, deepcopy(st))
    ok = LifecycleTransition(_call(1.0 - 3 / 365), ENG, st, st)
    assert resolve_transition(s0, s1, ok, calendar_days=3) is ok
    coupon = _event(LifecycleEventType.COUPON, MON)
    with pytest.raises(ValidationError, match="unchanged"):
        resolve_transition(s0, s1, LifecycleTransition(_call(1.0 - 3 / 365), ENG, st, st, events=(coupon,)),
                           calendar_days=3)
    with pytest.raises(ValidationError, match="MODEL change"):
        resolve_transition(s0, s1, LifecycleTransition(_call(1.0 - 3 / 365), BlackScholesEngine(), st, st),
                           calendar_days=3)
    with pytest.raises(ValidationError):                       # t1 must hold the alive contract itself
        resolve_transition(s0, _snap(_call(1.0 - 4 / 365), MON, deepcopy(st)), ok, calendar_days=3)


def test_terminal_at_t0_skips_the_roll_check():
    st = AutocallableLifecycleState()
    st.mark_ko(FRI - timedelta(days=1), cashflow=10.0, settlement_date=MON)
    s0, s1 = _snap(_call(1.0), FRI, st), _snap(_call(1.0), MON, deepcopy(st))   # unrolled: nothing is alive
    t = resolve_transition(s0, s1, None, calendar_days=3)
    assert t.product_alive_t1 is s1.product and t.changed is False


def test_event_row_reads_both_endpoints_from_the_cache():
    class FakeCache:
        def value_t1(self):
            return ValueBreakdown(0.0, 9.5, 0.0)

        def all_market(self):
            return ValueBreakdown(12.0, 0.0, 0.0)

    st0 = AutocallableLifecycleState()
    st1 = deepcopy(st0)
    st1.mark_ko(MON, cashflow=9.5, settlement_date=MON + timedelta(days=2))
    ko = _event(LifecycleEventType.KNOCK_OUT, MON, payoff=9.5, cashflow=9.5, terminates_position=True)
    t = LifecycleTransition(_call(1.0 - 3 / 365), ENG, st0, st1, events=(ko,))
    row = event_row(FakeCache(), t)
    assert row.factor is Factor.LIFECYCLE_EVENT and row.method is ExplainMethod.SHARED
    assert row.kind is RowKind.COMPONENT and row.level == "instrument"
    assert row.pnl == pytest.approx(9.5 - 12.0)
    assert row.metadata["changed"] is True
    assert event_summary(t) == ({
        "event_type": LifecycleEventType.KNOCK_OUT.value, "date": MON.isoformat(),
        "payoff": 9.5, "cashflow": 9.5, "terminates_position": True,
    },)


def test_transition_validates_roll_days_and_the_event_protocol():
    import numpy as np
    from quantark.util.exceptions import NumericalError
    s = AutocallableLifecycleState()
    ok = LifecycleTransition(_call(1.0), ENG, s, s, contract_roll_days=np.int64(2))
    assert ok.contract_roll_days == 2 and type(ok.contract_roll_days) is int
    for bad in (1.9, True, -1, "2", 0.0):
        with pytest.raises(ValidationError, match="contract_roll_days"):
            LifecycleTransition(_call(1.0), ENG, s, s, contract_roll_days=bad)
    s_ki = deepcopy(s)
    s_ki.mark_ki(MON)
    with pytest.raises(ValidationError, match="LifecycleEvent instances"):
        LifecycleTransition(_call(1.0), ENG, s, s_ki, events=("knock_in",))
    with pytest.raises(ValidationError, match="event_type"):
        LifecycleTransition(_call(1.0), ENG, s, s_ki, events=(_event("KNOCK_IN", MON),))
    with pytest.raises(ValidationError, match="no date"):
        LifecycleTransition(_call(1.0), ENG, s, s_ki, events=(_event(LifecycleEventType.KNOCK_IN, None),))
    with pytest.raises(ValidationError, match="not a date"):
        LifecycleTransition(_call(1.0), ENG, s, s_ki, events=(_event(LifecycleEventType.KNOCK_IN, "someday"),))
    with pytest.raises(NumericalError, match="non-finite payoff"):
        LifecycleTransition(_call(1.0), ENG, s, s_ki,
                            events=(_event(LifecycleEventType.KNOCK_IN, MON, payoff=float("nan")),))
    with pytest.raises(ValidationError, match="cashflow must be a number"):
        LifecycleTransition(_call(1.0), ENG, s, s_ki, events=(_event(LifecycleEventType.KNOCK_IN, MON, cashflow="x"),))
    with pytest.raises(ValidationError, match="terminates_position"):
        LifecycleTransition(_call(1.0), ENG, s, s_ki,
                            events=(_event(LifecycleEventType.KNOCK_IN, MON, terminates_position=1),))
