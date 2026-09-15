"""Batch valuation, prepared spot curves and provenance-preserving aggregation."""
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantark.asset.equity.engine.analytical import DigitalOptionAnalyticalEngine
from quantark.asset.equity.engine.pde import SnowballPDESolver
from quantark.asset.equity.engine.quad import SnowballQuadEngine
from quantark.asset.equity.engine.quad.v2 import SnowballQuadEngineV2
from quantark.asset.equity.param import PDEParams
from quantark.execution import PricingSession
from quantark.execution.errors import CapabilityError
from quantark.intraday import Fixing, VarianceProfile, resolve_context, value_intraday
from quantark.intraday.batch import IntradayFailure, aggregate_intraday, spot_curve, value_intraday_many
from quantark.intraday.engines import route_for
from quantark.intraday.events import EventKind
from quantark.intraday.request import IntradayValuationRequest
from intraday.conftest import SHANGHAI, dated_snowball, digital, flat_env

T0 = datetime(2026, 3, 16)
QUAD = SnowballQuadEngineV2()


@pytest.fixture
def desk():
    return VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,))


def _kos(sse_calendar, sse_sessions, profile):
    probe = resolve_context(IntradayValuationRequest(product=dated_snowball(sse_calendar, T0),
                                                     pricing_env=flat_env(datetime(2026, 9, 1, tzinfo=SHANGHAI)),
                                                     session_calendar=sse_sessions, variance_profile=profile))
    return [e for e in probe.timeline.events if e.kind is EventKind.KO]


def _req(sse_calendar, sse_sessions, profile, when, spot=100.0, confirmed=True, **kw):
    kos = _kos(sse_calendar, sse_sessions, profile)
    if confirmed:
        kw.setdefault("fixings", tuple(Fixing(k.timestamp, 100.0) for k in kos[:5]))
    return IntradayValuationRequest(product=dated_snowball(sse_calendar, T0), pricing_env=flat_env(when(kos), spot=spot),
                                    session_calendar=sse_sessions, variance_profile=profile, **kw)


def test_batch_keeps_caller_order_and_types_every_failure(sse_calendar, sse_sessions, desk):
    snow = _req(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp - timedelta(hours=1), request_id="snow")
    unsupported = _req(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp - timedelta(hours=2), request_id="v1")
    dig = IntradayValuationRequest(product=digital(datetime(2026, 9, 16)), pricing_env=flat_env(datetime(2026, 9, 15, 14, 0, tzinfo=SHANGHAI)),
                                   session_calendar=sse_sessions, variance_profile=desk)
    items = [(QUAD, snow), (SnowballQuadEngine(), unsupported), (DigitalOptionAnalyticalEngine(), dig)]
    with pytest.raises(CapabilityError):
        value_intraday_many(items)
    out = value_intraday_many(items, collect_errors=True)
    assert out[0].price == value_intraday(QUAD, snow).price
    assert isinstance(out[1], IntradayFailure) and out[1].item_id == "v1" and out[1].error_type == "CapabilityError"
    assert "intraday" in out[1].message
    assert isinstance(out[2], IntradayFailure) is False and out[2].price == value_intraday(DigitalOptionAnalyticalEngine(), dig).price
    with PricingSession() as session:
        through_session = session.value_intraday_many(items, collect_errors=True)
    assert [type(r).__name__ for r in through_session] == [type(r).__name__ for r in out]
    assert out[2].__class__.__name__ == "IntradayValuationResult"


def test_quad_spot_curve_is_the_prepared_operator_of_one_context(sse_calendar, sse_sessions, desk):
    when = lambda kos: kos[5].timestamp - timedelta(hours=1)                                      # noqa: E731
    req = _req(sse_calendar, sse_sessions, desk, when)
    spots = list(np.linspace(95.0, 106.0, 101))
    curve = spot_curve(QUAD, req, spots)
    assert [p.spot for p in curve] == spots
    for p in curve[::10]:
        direct = value_intraday(QUAD, _req(sse_calendar, sse_sessions, desk, when, spot=p.spot)).price
        assert p.price == pytest.approx(direct, abs=1e-10)
    mid = curve[50]
    mid_ctx = resolve_context(_req(sse_calendar, sse_sessions, desk, when, spot=mid.spot))
    pg = route_for(mid_ctx, QUAD).point_greeks(mid_ctx, QUAD)
    assert mid.status == "ok" and mid.delta == pytest.approx(pg.delta, rel=1e-10) and mid.gamma == pytest.approx(pg.gamma, rel=1e-8)


def test_a_curve_never_rebuilds_assumptions_per_point(sse_calendar, sse_sessions, desk):
    req = _req(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp - timedelta(hours=1), confirmed=False)
    base = resolve_context(req)
    assert base.provisional and base.reconstruction.assumptions
    curve = spot_curve(QUAD, req, [98.0, 103.5, 110.0])            # 103.5 would assume a KO if rebuilt from that spot
    assert all(p.assumptions == base.reconstruction.assumptions for p in curve)


def test_other_routes_price_per_spot_without_greeks_and_flag_barrier_points(sse_calendar, sse_sessions, desk):
    at_fixing = lambda kos: kos[5].timestamp                                                      # noqa: E731
    req = _req(sse_calendar, sse_sessions, desk, at_fixing, event_phase="before")
    pde = SnowballPDESolver(PDEParams())
    curve = spot_curve(pde, req, [101.0, 103.0])
    assert curve[0].delta is None and curve[0].status == "not_requested"
    assert curve[1].status == "undefined" and "discontinuity" in curve[1].reason
    quad_curve = spot_curve(QUAD, req, [101.0, 103.0])
    assert quad_curve[1].status == "undefined" and quad_curve[1].delta is None
    assert quad_curve[1].price == pytest.approx(value_intraday(QUAD, _req(sse_calendar, sse_sessions, desk, at_fixing, spot=103.0,
                                                                           event_phase="before")).price, abs=1e-10)


def test_aggregation_preserves_provenance_and_sums_only_qualified_greeks(sse_calendar, sse_sessions, desk):
    when = lambda kos: kos[5].timestamp - timedelta(hours=1)                                      # noqa: E731
    confirmed = value_intraday(QUAD, _req(sse_calendar, sse_sessions, desk, when, greeks=("delta",), greek_convention="point"))
    provisional = value_intraday(QUAD, _req(sse_calendar, sse_sessions, desk, when, confirmed=False, greeks=("delta",),
                                            greek_convention="point"))
    at_barrier = value_intraday(QUAD, _req(sse_calendar, sse_sessions, desk, lambda kos: kos[5].timestamp, spot=103.0,
                                           event_phase="before", greeks=("delta",), greek_convention="point"))
    book = aggregate_intraday([("A", 2.0, confirmed), ("B", -1.0, provisional)])
    assert book.provisional and book.provisional_positions == ("B",)
    assert book.total_price == pytest.approx(2.0 * confirmed.price - provisional.price, rel=1e-15)
    assert book.greeks["delta"] == pytest.approx(2.0 * confirmed.greek("delta").value - provisional.greek("delta").value)
    assert book.greek_status["delta"] == "ok"
    mixed = aggregate_intraday([("A", 1.0, confirmed), ("C", 1.0, at_barrier)])
    assert mixed.greeks["delta"] is None and "C" in mixed.greek_status["delta"] and "undefined" in mixed.greek_status["delta"]
    assert not mixed.provisional and mixed.total_paid_cash == confirmed.paid_cash + at_barrier.paid_cash
