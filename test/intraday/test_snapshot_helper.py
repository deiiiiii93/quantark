"""The no-mutation helper must see nested changes, including ledger entries and non-field attributes."""
from datetime import datetime

from quantark.asset.equity.lifecycle import AutocallableLifecycleState, LifecycleEventType, RealizedCashflow
from intraday.conftest import dated_snowball, snapshot


def test_snapshot_is_stable_and_sees_nested_mutations(sse_calendar):
    state = AutocallableLifecycleState()
    before = snapshot(state)
    assert snapshot(state) == before
    state.ledger.register(RealizedCashflow(cashflow_id="x", event_type=LifecycleEventType.COUPON, amount=1.0,
                                           determination_date=datetime(2026, 9, 1), payment_date=datetime(2026, 9, 1)))
    assert snapshot(state) != before
    state2 = AutocallableLifecycleState()
    before2 = snapshot(state2)
    state2.observed_ko_indices.add(3)
    assert snapshot(state2) != before2

    prod = dated_snowball(sse_calendar, datetime(2026, 3, 16))
    before3 = snapshot(prod)
    prod.barrier_config.ko_observation_schedule.records[2].barrier = 104.0
    assert snapshot(prod) != before3
    prod4 = dated_snowball(sse_calendar, datetime(2026, 3, 16))
    before4 = snapshot(prod4)
    setattr(prod4, "_otc_lifecycle_knocked_in", True)
    assert snapshot(prod4) != before4
