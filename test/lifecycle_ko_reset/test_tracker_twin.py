"""The tracker's pricing copy agrees with the intraday twin at a matching checkpoint (spec 2026-10-08 section 7):
the same schedule records are left on each schedule, the same knock-in state carried."""
from datetime import datetime

import pandas as pd
import pytest

from quantark.asset.equity.lifecycle import AutocallableLifecycleState, KOResetLifecycleTracker, ValuationPoint
from quantark.intraday.context import resolve_context
from quantark.intraday.profile import VarianceProfile
from quantark.intraday.request import IntradayValuationRequest

from intraday.conftest import SHANGHAI, dated_ko_reset, flat_env, sse_calendar, sse_sessions  # noqa: F401

T0 = datetime(2026, 3, 16)
CHECKPOINT = datetime(2026, 9, 14)
TS = datetime(2026, 9, 15, 14, 0, tzinfo=SHANGHAI)


@pytest.mark.parametrize("knocked_in", [False, True], ids=["alive", "knocked_in"])
def test_the_pricing_copy_leaves_what_the_twin_leaves(sse_calendar, sse_sessions, knocked_in):  # noqa: F811
    product = dated_ko_reset(sse_calendar, T0)
    state = AutocallableLifecycleState(knocked_in=knocked_in, valuation_point=ValuationPoint(date=CHECKPOINT))
    request = IntradayValuationRequest(
        product=product, pricing_env=flat_env(TS), session_calendar=sse_sessions,
        variance_profile=VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,)), lifecycle_state=state)
    twin = resolve_context(request).numerical.product

    tracker = KOResetLifecycleTracker(product=product, quantity=1.0, start_date=pd.Timestamp(T0),
                                      lifecycle=AutocallableLifecycleState(knocked_in=knocked_in))
    copy = tracker.product_for_pricing(pd.Timestamp(CHECKPOINT), flat_env(CHECKPOINT))

    for name in ("barrier_config", "post_barrier_config"):
        assert (len(getattr(copy, name).ko_observation_schedule.records)
                == len(getattr(twin, name).ko_observation_schedule.records)), name
    assert copy._otc_lifecycle_knocked_in is twin._otc_lifecycle_knocked_in is knocked_in
