from datetime import datetime, timedelta

import pytest

from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
from quantark.asset.equity.param import EngineParams
from quantark.intraday import Fixing, value_intraday
from quantark.intraday.context import resolve_context
from quantark.intraday.events import EventKind
from quantark.intraday.greeks import desk_bump_cells
from quantark.intraday.profile import VarianceProfile
from quantark.intraday.request import IntradayValuationRequest
from intraday.conftest import SHANGHAI, dated_snowball, flat_env

T0 = datetime(2026, 3, 16)
E = SnowballQuadEngineV2()


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


def _req(sse_calendar, sse_sessions, profile, ts, prod=None, **kw):
    spot = kw.pop("spot", 100.0)
    return IntradayValuationRequest(product=prod or dated_snowball(sse_calendar, T0), pricing_env=flat_env(ts, spot=spot),
                                    session_calendar=sse_sessions, variance_profile=profile, **kw)


def _kos(sse_calendar, sse_sessions, profile, prod=None):
    ctx = resolve_context(_req(sse_calendar, sse_sessions, profile, datetime(2026, 9, 15, tzinfo=SHANGHAI), prod=prod))
    return [e for e in ctx.timeline.events if e.kind is EventKind.KO]


def test_cells_share_the_numerical_twin_and_match_the_daily_conventions(sse_calendar, sse_sessions, desk):
    ts = datetime(2026, 9, 15, 14, 0, tzinfo=SHANGHAI)
    ctx = resolve_context(_req(sse_calendar, sse_sessions, desk, ts))
    cells = desk_bump_cells(ctx, E, ("delta", "gamma", "vega", "rho", "dividend_rho"))
    assert set(cells) == {"base", "spot_up", "spot_down", "vol_up", "rate_up", "div_up"}
    assert all(c.ctx.numerical is ctx.numerical and c.ctx.reconstruction is ctx.reconstruction for c in cells.values())
    bc = EngineParams().get_effective_bump_config()
    assert cells["spot_up"].ctx.pricing_env.spot == pytest.approx(100.0 * (1 + bc.spot_bump))
    assert ctx.pricing_env.spot == 100.0                                  # the base context is untouched
    res = value_intraday(E, _req(sse_calendar, sse_sessions, desk, ts, greeks=("delta", "gamma", "vega", "rho"),
                                 greek_convention="desk_bump"))
    d = res.greek("delta")
    assert d.convention == "desk_bump" and d.status == "ok" and d.bump == pytest.approx(100.0 * bc.spot_bump)
    assert d.value == pytest.approx((cells["spot_up"].price - cells["spot_down"].price) / (2 * 100.0 * bc.spot_bump), rel=1e-12)
    g = res.greek("gamma")
    assert g.value == pytest.approx((cells["spot_up"].price - 2 * cells["base"].price + cells["spot_down"].price)
                                    / (100.0 * bc.spot_bump) ** 2, rel=1e-10)
    assert res.greek("vega").unit.startswith("PnL per +") and res.greek("rho").unit == "PnL per +1% rate"
    assert res.greek("vega").value == pytest.approx(cells["vol_up"].price - cells["base"].price, rel=1e-12)


def test_assumed_ko_has_zero_conditional_delta_and_negative_rho_on_a_delayed_payment(sse_calendar, sse_sessions, desk):
    prod = dated_snowball(sse_calendar, T0)
    for r in prod.barrier_config.ko_observation_schedule.records:
        r.settlement_date = r.observation_date + timedelta(days=5)
    kos = _kos(sse_calendar, sse_sessions, desk, prod=prod)
    ts = kos[5].timestamp + timedelta(seconds=30)
    fixings = tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])      # alive until the assumed sixth fixing
    res = value_intraday(E, _req(sse_calendar, sse_sessions, desk, ts, prod=prod, spot=104.0, event_phase="after", fixings=fixings,
                                 greeks=("delta", "rho"), greek_convention="desk_bump"))
    assert res.provisional and res.lifecycle["knocked_out"]
    assert res.greek("delta").value == 0.0                             # conditional on the assumed KO: no equity exposure
    assert res.greek("rho").value < 0.0                                # the receivable is discounted for five days
    assert res.pending_receivable_pv > 0.0 and res.contingent_pv == 0.0


def test_bumps_do_not_rebuild_assumptions_from_bumped_spots(sse_calendar, sse_sessions, desk):
    # spot 102.99 assumed alive at the sixth fixing; +1% would knock out if assumptions were rebuilt — they must not be
    kos = _kos(sse_calendar, sse_sessions, desk)
    ts = kos[5].timestamp + timedelta(seconds=30)
    fixings = tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])
    ctx = resolve_context(_req(sse_calendar, sse_sessions, desk, ts, spot=102.99, event_phase="after", fixings=fixings))
    cells = desk_bump_cells(ctx, E, ("delta",))
    # the +1% cell (104.02) prices the SAME alive contract: its sixth fixing stays the assumed 102.99
    assert all(c.ctx.numerical.lifecycle_state.alive and c.ctx.reconstruction is ctx.reconstruction for c in cells.values())
    delta = (cells["spot_up"].price - cells["spot_down"].price) / (2.0 * 102.99 * 0.01)
    assert abs(delta) < 5.0


def test_unknown_greeks_fail_closed(sse_calendar, sse_sessions, desk):
    from quantark.execution.errors import CapabilityError
    ts = datetime(2026, 9, 15, 14, 0, tzinfo=SHANGHAI)
    with pytest.raises(CapabilityError):
        value_intraday(E, _req(sse_calendar, sse_sessions, desk, ts, greeks=("vanna",), greek_convention="desk_bump"))
