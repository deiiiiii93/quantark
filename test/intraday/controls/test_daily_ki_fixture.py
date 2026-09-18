"""The daily-KI control fixture: KI at every SSE close, fixed on the 2026-09-10 close (design 2026-09-17)."""
from datetime import datetime, timedelta

from intraday.conftest import SHANGHAI, dated_snowball
from intraday.controls import fixtures as C
from intraday.controls.fixtures import build_context


def test_daily_ki_fixture_observes_ki_at_every_close_and_keeps_the_monthly_terms():
    daily, monthly = C.product("snowball_daily_ki"), dated_snowball(C.sse().calendar, C.T0)
    records = daily.barrier_config.ki_observation_schedule.records
    dates = [r.observation_date for r in records]
    assert len(records) == 243 and {r.barrier for r in records} == {75.0}
    assert dates[0] == datetime(2026, 3, 17) and dates[-1] == daily.exercise_date == datetime(2027, 3, 16)
    assert dates == sorted(set(dates)) and all(C.sse().calendar.is_business_day(d) for d in dates)
    ko_daily = [(r.observation_date, r.barrier) for r in daily.barrier_config.ko_observation_schedule.records]
    ko_monthly = [(r.observation_date, r.barrier) for r in monthly.barrier_config.ko_observation_schedule.records]
    assert ko_daily == ko_monthly and len(ko_daily) == 12
    assert (C.BARRIERS["snowball_daily_ki"], C.MONITORING["snowball_daily_ki"], C.notional("snowball_daily_ki")) == \
        (("ko", "ki"), "discrete", 100.0)


def test_daily_ki_fixing_is_the_2026_09_10_close_after_122_confirmed_closes():
    fixing, history = C.fixing_and_history("snowball_daily_ki")
    assert fixing == datetime(2026, 9, 10, 15, 0, tzinfo=SHANGHAI)
    assert len(history) == 122 and {f.value for f in history} == {100.0}
    assert all(f.timestamp < fixing for f in history)


def test_every_daily_ki_horizon_resolves_the_same_alive_claim_inside_one_gap():
    assert max(C.DAILY_KI_HORIZONS) < timedelta(days=1)      # a one-day rung would sit on the previous close
    remaining = set()
    for horizon in C.DAILY_KI_HORIZONS:
        ctx, _ = build_context(C.Cell("snowball_daily_ki", "quad_v2", "desk", horizon, "eq", "ki"))
        assert not ctx.numerical.knocked_in and not ctx.provisional
        remaining.add(len(ctx.numerical.remaining_events))
    assert remaining == {129}
