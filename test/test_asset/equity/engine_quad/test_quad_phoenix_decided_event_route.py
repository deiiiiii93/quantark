"""QUAD v1 Phoenix: a knock-out decided at valuation keeps the coupon paid with it on the event route.

The immediate knock-out statistics were the Snowball ones, so ``price_with_events`` reported the knock-out
redemption alone while ``price`` paid redemption + the coupon decided at the same instant.
"""
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.cashleg import EventType
from quantark.util.calendar import CalendarType, create_calendar

from intraday.conftest import dated_phoenix, dated_snowball, flat_env

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
T0 = datetime(2026, 3, 16)


def _decided(product, k=2):
    dates = [r.observation_date for r in product.barrier_config.ko_observation_schedule.records]
    return flat_env(dates[k], spot=110.0)


@pytest.mark.parametrize("k", [0, 2])
def test_event_route_npv_is_the_decided_price(k):
    product = dated_phoenix(CAL, T0)
    env = _decided(product, k)
    engine = PhoenixQuadEngine()
    result = engine.price_with_events(product, env)
    assert result.npv == engine.price(product, env)
    dist = result.event_distribution
    assert float(np.sum(dist.probabilities[EventType.KO])) == pytest.approx(1.0, abs=1e-12)


def test_the_decided_coupon_is_its_own_ledger_entry():
    product = dated_phoenix(CAL, T0)
    env = _decided(product)
    engine = PhoenixQuadEngine()
    stats = engine.calculate_event_stats(product, env)
    decision = product.decide_observations_at_valuation(env.spot, env, knocked_in=False)
    assert decision.knocked_out and decision.coupon > 0.0
    i = int(np.argmax(stats.observations.probability))
    df = env.get_discount_factor(decision.ko_record.settlement_time) \
        if decision.ko_record.settlement_time and decision.ko_record.settlement_time > 0.0 else 1.0
    assert stats.coupons.probability[i] == 1.0
    assert stats.coupons.discounted_cash[i] == pytest.approx(decision.coupon * df, rel=1e-14)
    assert stats.pv == pytest.approx(float(np.sum(stats.ledger.discounted_cash)), rel=1e-14)
    assert stats.pv == engine.price(product, env)


def test_snowball_decided_route_is_unchanged():
    product = dated_snowball(CAL, T0)
    env = _decided(product)
    engine = SnowballQuadEngine()
    assert engine.price_with_events(product, env).npv == engine.price(product, env)
