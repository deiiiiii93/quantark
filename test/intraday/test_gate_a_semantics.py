"""Gate A: clock, event and cashflow semantics, end to end through value_intraday."""
from datetime import datetime, timedelta, timezone

import pytest

from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
from quantark.intraday import Fixing, value_intraday
from quantark.intraday.context import resolve_context
from quantark.intraday.events import EventKind
from quantark.intraday.profile import VarianceProfile
from quantark.intraday.request import IntradayValuationRequest
from intraday.conftest import SHANGHAI, dated_snowball, flat_env

T0 = datetime(2026, 3, 16)
ENGINE = SnowballQuadEngineV2()


def _req(sse_calendar, sse_sessions, profile, ts, **kw):
    spot = kw.pop("spot", 100.0)
    return IntradayValuationRequest(product=dated_snowball(sse_calendar, T0), pricing_env=flat_env(ts, spot=spot),
                                    session_calendar=sse_sessions, variance_profile=profile, **kw)


def _kos(sse_calendar, sse_sessions, profile):
    ctx = resolve_context(_req(sse_calendar, sse_sessions, profile, datetime(2026, 9, 15, tzinfo=SHANGHAI)))
    return [e for e in ctx.timeline.events if e.kind is EventKind.KO]


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


def test_ladder_towards_the_fixing_converges_to_the_ko_cash_for_a_spot_above_the_barrier(sse_calendar, sse_sessions, desk):
    kos = _kos(sse_calendar, sse_sessions, desk)
    fix = kos[5].timestamp
    fixings = tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])
    prices = []
    for delta in (timedelta(days=1), timedelta(hours=6), timedelta(hours=1), timedelta(minutes=15), timedelta(minutes=5),
                  timedelta(minutes=1), timedelta(seconds=10), timedelta(seconds=1)):
        res = value_intraday(ENGINE, _req(sse_calendar, sse_sessions, desk, fix - delta, spot=103.5, fixings=fixings))
        assert not res.provisional
        prices.append(res.price)
    assert abs(prices[-1] - kos[5].cash) <= 1e-6 * kos[5].cash
    # The price itself is NOT monotone towards the fixing: the unknocked contract is worth more than
    # this KO cash (6.86 a day out, 6.05 a second out). What must shrink is the distance to the cash.
    gaps = [abs(p - kos[5].cash) for p in prices]
    assert all(b <= a + 1e-12 for a, b in zip(gaps, gaps[1:])), gaps


def test_exact_barrier_equality_uses_the_contracts_inclusive_rule(sse_calendar, sse_sessions, desk):
    kos = _kos(sse_calendar, sse_sessions, desk)
    fixings = tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])
    res = value_intraday(ENGINE, _req(sse_calendar, sse_sessions, desk, kos[5].timestamp, spot=103.0, fixings=fixings,
                                      event_phase="before"))
    assert res.price == pytest.approx(kos[5].cash, abs=1e-9)          # spot >= barrier knocks out (inclusive)


def test_missed_fixings_below_ki_knock_in_and_stay_alive(sse_calendar, sse_sessions, desk):
    kos = _kos(sse_calendar, sse_sessions, desk)
    fix = kos[5].timestamp
    res = value_intraday(ENGINE, _req(sse_calendar, sse_sessions, desk, fix + timedelta(seconds=30), spot=70.0, event_phase="after"))
    assert res.lifecycle["knocked_in"] and res.lifecycle["alive"] and res.provisional and len(res.assumptions) == 6


def test_original_cash_amounts_do_not_move_with_valuation_time(sse_calendar, sse_sessions, desk):
    kos = _kos(sse_calendar, sse_sessions, desk)
    a = resolve_context(_req(sse_calendar, sse_sessions, desk, kos[5].timestamp - timedelta(days=3)))
    b = resolve_context(_req(sse_calendar, sse_sessions, desk, kos[5].timestamp - timedelta(seconds=3)))
    assert [(e.cash, e.timestamp) for e in a.timeline.events] == [(e.cash, e.timestamp) for e in b.timeline.events]


def test_zero_variance_interval_still_carries_and_discounts(sse_calendar, sse_sessions):
    p = VarianceProfile.sessions_only(sse_sessions, 244)
    a = value_intraday(ENGINE, _req(sse_calendar, sse_sessions, p, datetime(2026, 9, 15, 11, 30, tzinfo=SHANGHAI)))
    b = value_intraday(ENGINE, _req(sse_calendar, sse_sessions, p, datetime(2026, 9, 15, 12, 59, tzinfo=SHANGHAI)))
    assert a.price != b.price and a.price == pytest.approx(b.price, rel=2e-5)


def test_timezone_conversion_does_not_change_the_instant(sse_calendar, sse_sessions, desk):
    ts_sh = datetime(2026, 9, 15, 14, 59, 59, tzinfo=SHANGHAI)
    ts_utc = ts_sh.astimezone(timezone.utc)
    a = value_intraday(ENGINE, _req(sse_calendar, sse_sessions, desk, ts_sh))
    b = value_intraday(ENGINE, _req(sse_calendar, sse_sessions, desk, ts_utc))
    assert a.price == b.price and a.context_identity == b.context_identity
