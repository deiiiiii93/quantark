"""Replay seeds a flagged Snowball / Phoenix knocked in (0.16.0 run 6, D7; final review finding 1).

The replay engine used to hand every Snowball / Phoenix a fresh ``AutocallableLifecycleState()``, so the tracker
overwrote ``knocked_in_at_valuation``; it now passes no state and the tracker seeds from the flag (an unflagged
product gets the same not-knocked-in state as before).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from golden.replay_golden import fixtures  # noqa: E402

from quantark.backtest.replay import HedgeSpec, ReplayBacktestConfig, ReplayBacktestEngine, ReplayProduct  # noqa: E402
from quantark.backtest.transaction_costs import ZeroCostModel  # noqa: E402


def _config(product):
    return ReplayBacktestConfig(
        products=[ReplayProduct(product=product, quantity=-1.0, position_id=1, has_lifecycle=True)],
        market_data=fixtures._market_data(),
        hedge=HedgeSpec(kind="futures"),
        engine_config=fixtures.make_scalar_bsm_config().engine_config,
        transaction_cost_model=ZeroCostModel(),
        calculate_surfaces=False,
        calculate_event_probabilities=False,
    )


@pytest.mark.parametrize("make", [fixtures._snowball_product, fixtures._phoenix_product], ids=["snowball", "phoenix"])
@pytest.mark.parametrize("flag", [True, False])
def test_replay_seeds_the_tracker_from_the_flag(make, flag):
    product = make()
    product.knocked_in_at_valuation = flag
    engine = ReplayBacktestEngine(_config(product))
    assert engine._replays[0]._tracker.lifecycle.knocked_in is flag


def test_a_flagged_snowball_replays_knocked_in():
    flagged, plain = fixtures._snowball_product(), fixtures._snowball_product()
    flagged.knocked_in_at_valuation = True
    a = ReplayBacktestEngine(_config(flagged)).run().states_df()
    b = ReplayBacktestEngine(_config(plain)).run().states_df()
    assert len(a) == len(b) and a["knocked_in"].all() and not b["knocked_in"].iloc[0]
    assert a["product_mtm"].iloc[0] != b["product_mtm"].iloc[0]   # day 1: the flag alone decides
