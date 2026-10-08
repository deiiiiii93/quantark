"""The knock-in anchor is resolved once, when the knock-in happens, and persisted (accrual time spec §5.1)."""
import pickle
from copy import deepcopy
from datetime import timedelta

import pandas as pd

from quantark.asset.equity.lifecycle import AutocallableLifecycleState
from quantark.asset.equity.lifecycle.autocallable import AutocallableLifecycleTracker
from quantark.asset.equity.product.option.accrual import KnockInAnchor
from quantark.pnlexplain.equity.fingerprints import lifecycle_fingerprint

from .conftest import T0, env, snowball, times

D = 1 / 365.0


def _knocked_in_on_day_95():
    product = snowball(ko=times(200 * D, 300 * D, level=1e6), ki=times(91 * D, 93 * D, level=75.0), maturity=300 * D)
    tracker = AutocallableLifecycleTracker(product=product, quantity=1.0, start_date=pd.Timestamp(T0))
    day = pd.Timestamp(T0 + timedelta(days=95))
    tracker.observe(day, tracker.product_for_lifecycle(), env(day.to_pydatetime(), spot=70.0), 70.0)
    return tracker, day


def test_the_anchor_is_the_record_whose_test_fired_at_its_contractual_fixing():
    tracker, day = _knocked_in_on_day_95()
    state = tracker.lifecycle
    assert state.knocked_in and state.ki_date == day.to_pydatetime()
    anchor = state.knock_in_anchor
    assert isinstance(anchor, KnockInAnchor)
    assert (anchor.source_index, anchor.fixing, anchor.origin) == (0, 91 * D, 0.0)


def test_the_anchor_survives_ageing_checkpoints_and_the_pricing_copy():
    tracker, day = _knocked_in_on_day_95()
    anchor = tracker.lifecycle.knock_in_anchor
    later = day + pd.Timedelta(days=20)
    copy = tracker.product_for_pricing(later, env(later.to_pydatetime()))
    assert copy.barrier_config.ki_observation_schedule.records == []
    assert copy._otc_lifecycle_knock_in_anchor == anchor
    assert deepcopy(tracker.lifecycle).knock_in_anchor == anchor
    # a checkpoint round trip of this test's own in-process object (no untrusted data is loaded)
    assert pickle.loads(pickle.dumps(tracker.lifecycle)).knock_in_anchor == anchor


def test_the_anchor_is_state_not_bookkeeping():
    tracker, _ = _knocked_in_on_day_95()
    plain = AutocallableLifecycleState(knocked_in=True, ki_date=tracker.lifecycle.ki_date)
    assert lifecycle_fingerprint(plain) != lifecycle_fingerprint(
        AutocallableLifecycleState(knocked_in=True, ki_date=tracker.lifecycle.ki_date,
                                   knock_in_anchor=tracker.lifecycle.knock_in_anchor))
