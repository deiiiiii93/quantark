"""KO-reset snowballs on simulated paths (spec 2026-10-08 section 6): exact mode equals the replay to the last bit;
surface mode prices an ABSOLUTE contract within its gate and refuses a REBASED one."""
from __future__ import annotations

import numpy as np
import pytest

from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.backtest.replay import AutocallableEngineConfig, ReplayProduct
from quantark.backtest.simulation.conformance import run_oracle
from quantark.backtest.simulation.engine import EnsembleBacktestEngine
from quantark.util.enum import ObservationType
from quantark.util.enum.engine_enums import EngineType
from quantark.util.enum.option_enums import PostKOScheduleMode as M
from quantark.util.exceptions import ValidationError

from simulation.conftest import (INCEPTION, KI_BARRIER, KO_BARRIER, SPOT, ensemble_config, make_market_path,
                                 pde_engine_config, surface_pricing)

D = 1 / 365.0


def _schedule(days, level):
    return ObservationSchedule(records=[ObservationRecord(observation_time=d * D, barrier=level) for d in days])


def short_ko_reset(mode, *, pre=(2, 5), ki=(1, 3), post=(4, 7)):
    """Calendar days from 2024-01-02 (a Tuesday): schedule 1 (103%) on `pre`, KI (75%) on `ki`, schedule 2 (100%)
    on `post`, absolute days or offsets from the knock-in close (REBASED)."""
    maturity = (max(ki) + max(post) if mode == M.REBASED else max(post)) * D
    return KnockOutResetSnowballOption(
        initial_price=SPOT, strike=SPOT, maturity=maturity, contract_multiplier=1.0, initial_date=INCEPTION,
        barrier_config=BarrierConfig(ko_barrier=KO_BARRIER, ko_rate=0.20, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_schedule(pre, KO_BARRIER), ki_barrier=KI_BARRIER,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=_schedule(ki, KI_BARRIER)),
        post_barrier_config=BarrierConfig(ko_barrier=SPOT, ko_rate=0.05, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_schedule(post, SPOT)),
        payoff_config=PayoffConfig(rebate_rate=0.03, include_principal=True), post_ko_mode=mode)


def mc_engine_config():
    return AutocallableEngineConfig(pricing_engine_type=EngineType.MONTE_CARLO,
                                    mc_params=MCParams(seed=3, num_paths=2048))


#: spot multiples by trading-day index (2024-01-02 Tue ... ; index 4 = Mon 2024-01-08)
LEVELS = {"ko1": {2: 1.05}, "v0": {}, "ki_ko2": {1: 0.70, 4: 1.00}, "ki_v1": {1: 0.70}}


def _paths(kind, n_days=12):
    path = make_market_path(n_paths=1, n_days=n_days)
    spot = np.full(n_days, SPOT * 0.98)
    for d, k in LEVELS[kind].items():
        spot[d] = SPOT * k
    path.spot[0, :] = spot
    return path


def _book(product):
    return [ReplayProduct(product=product, quantity=-1000.0, position_id=1, has_lifecycle=True)]


@pytest.mark.parametrize("kind", list(LEVELS))
@pytest.mark.parametrize("mode", [M.ABSOLUTE, M.REBASED], ids=["absolute", "rebased"])
def test_exact_mode_matches_the_replay_to_the_last_bit(mode, kind):
    engine_config = mc_engine_config() if mode == M.REBASED else pde_engine_config()
    cfg = ensemble_config(products=_book(short_ko_reset(mode)), engine_config=engine_config)
    report = run_oracle(cfg, _paths(kind), 0)
    assert report.passed, report.summary()
    assert report.max_pv_gap == 0.0 and report.max_total_pnl_gap == 0.0


@pytest.mark.parametrize("kind, events", [
    ("ko1", ["knock_out"]), ("v0", ["maturity"]), ("ki_ko2", ["knock_in", "knock_out"]),
    ("ki_v1", ["knock_in", "maturity"]),
])
def test_each_path_books_its_events(kind, events):
    cfg = ensemble_config(products=_book(short_ko_reset(M.ABSOLUTE)))
    results = EnsembleBacktestEngine(cfg).run(_paths(kind))
    assert [e.event for e in results.events if e.event != "settlement"] == events


def test_surface_mode_prices_an_absolute_ko_reset_within_the_gate():
    cfg = ensemble_config(products=_book(short_ko_reset(M.ABSOLUTE)), pricing=surface_pricing())
    results = EnsembleBacktestEngine(cfg).run(make_market_path(n_paths=3, n_days=12))
    assert results.manifest["gate"]["passed"]


def test_surface_mode_refuses_a_rebased_ko_reset_at_configuration():
    with pytest.raises(ValidationError, match="REBASED"):
        ensemble_config(products=_book(short_ko_reset(M.REBASED)), pricing=surface_pricing())


def test_an_unknocked_rebased_ko_reset_needs_mc():
    with pytest.raises(ValidationError, match="REBASED"):
        ensemble_config(products=_book(short_ko_reset(M.REBASED)))


def test_rebased_knock_in_past_calendar_is_data_end():
    """Review focus 4: knocked in on day 1, schedule 2 ends 30 days later, past an 8-day calendar."""
    cfg = ensemble_config(products=_book(short_ko_reset(M.REBASED, post=(4, 30))), engine_config=mc_engine_config(),
                          allow_data_end=True)
    results = EnsembleBacktestEngine(cfg).run(_paths("ki_v1", n_days=8))
    assert results.manifest["data_end_paths"] == 1


# -- the knock-in anchor on simulated paths (accrual time spec §5.1) ----------------------------------------------

def _gap_close_path():
    """Knock-in records on calendar days 4 and 5 (Sat 2024-01-06, Sun 01-07) both fall due on Mon 01-08 (trading
    index 4, calendar day 6), where the path knocks in; schedule 2 (offset 4) knocks out on day 10 (index 8)."""
    path = make_market_path(n_paths=1, n_days=10)
    spot = np.full(10, SPOT * 0.98)
    spot[4], spot[8] = SPOT * 0.70, SPOT * 1.00
    path.spot[0, :] = spot
    return path


def test_a_gap_close_knock_in_accrues_from_the_record_that_fired():
    from quantark.asset.equity.lifecycle import KOResetLifecycleTracker
    import pandas as pd
    from .conftest import env

    product = short_ko_reset(M.REBASED, pre=(2, 9), ki=(4, 5), post=(4, 7))
    path = _gap_close_path()
    cfg = ensemble_config(products=_book(product), engine_config=mc_engine_config())
    report = run_oracle(cfg, path, 0)                       # simulation == replay, cash included, to the bit
    assert report.passed, report.summary()
    tracker = KOResetLifecycleTracker(product=product, quantity=1.0, start_date=pd.Timestamp(INCEPTION))
    dates = pd.DatetimeIndex(path.dates)
    anchor = tracker.knock_in_anchor_on(dates, 4, env(date=dates[0]))
    assert (anchor.source_index, anchor.fixing) == (0, 4 * D)          # the fixing, not the close (day 6)
    placement = tracker.resolve_calendar_schedule(dates, env(date=dates[0])).placement(4)
    assert placement.ko_payoff[0] == pytest.approx(SPOT + SPOT * 0.05 * (4 * D + 4 * D), rel=1e-15)


def test_knock_in_records_of_different_levels_on_one_close_are_refused():
    from quantark.asset.equity.lifecycle import KOResetLifecycleTracker
    import pandas as pd
    from .conftest import env

    product = short_ko_reset(M.REBASED, pre=(2, 9), ki=(4, 5), post=(4, 7))
    product.barrier_config.ki_observation_schedule.records[1].barrier = 0.70 * SPOT
    tracker = KOResetLifecycleTracker(product=product, quantity=1.0, start_date=pd.Timestamp(INCEPTION))
    dates = pd.DatetimeIndex(_gap_close_path().dates)
    with pytest.raises(ValidationError, match="different levels"):
        tracker.resolve_calendar_schedule(dates, env(date=dates[0]))
