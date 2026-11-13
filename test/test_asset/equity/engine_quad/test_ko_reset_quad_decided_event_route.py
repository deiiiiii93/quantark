"""QUAD v1 knock-out-reset: a knock-out decided at valuation is a certain knock-out on the event route.

``price`` decides an observation at the valuation instant on the known spot, on the schedule the knock-in state puts
in force (``decide_observations_at_valuation``), but the event statistics skipped it: the recursion starts after time
zero, so the decided record got no probability and the knock-out's value was booked as the maturity cash flow. Cash
legs on a knock-out-reset were then valued wrong on an observation day (a backtest or replay starting on one).
QUAD v2 and the PDE report the certain knock-out; v1 now does too.
"""
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.v2 import KOResetSnowballQuadEngineV2
from quantark.cashleg import EventType, LegDirection
from quantark.cashleg.fixed_payoff_leg import FixedPayoffLeg, PaymentTrigger
from quantark.portfolio import EquityPosition
from quantark.util.calendar import CalendarType, create_calendar

from intraday.conftest import dated_ko_reset, flat_env

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
T0 = datetime(2026, 3, 16)

# (pre-schedule observation k, spot, carried knock-in): which knock-out is decided now
CASES = {
    "first schedule": (2, 110.0, False),
    "first schedule's last date": (5, 110.0, False),
    "second schedule, knock-in carried": (2, 100.0, True),
}


def _decided(name):
    k, spot, knocked_in = CASES[name]
    product = dated_ko_reset(CAL, T0)
    if knocked_in:
        product._otc_lifecycle_knocked_in = True
    dates = [r.observation_date for r in product.barrier_config.ko_observation_schedule.records]
    return product, flat_env(dates[k], spot=spot)


def _decision(product, env):
    return product.decide_observations_at_valuation(
        env.spot, env, knocked_in=getattr(product, "_otc_lifecycle_knocked_in", False))


@pytest.mark.parametrize("name", CASES)
def test_the_decided_knock_out_is_certain_as_on_v2(name):
    product, env = _decided(name)
    assert _decision(product, env).knocked_out
    v1 = KOResetSnowballQuadEngine().price_with_events(product, env).event_distribution
    v2 = KOResetSnowballQuadEngineV2().price_with_events(product, env).event_distribution
    assert np.array_equal(v1.stream_times[EventType.KO], v2.stream_times[EventType.KO])
    assert np.array_equal(v1.probabilities[EventType.KO], v2.probabilities[EventType.KO])
    assert float(np.sum(v1.probabilities[EventType.KO])) == 1.0
    for maturity in (EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI):
        assert v1.probabilities[maturity] == 0.0


@pytest.mark.parametrize("name", CASES)
def test_the_decided_ledger_is_the_price(name):
    product, env = _decided(name)
    engine = KOResetSnowballQuadEngine()
    stats = engine.calculate_event_stats(product, env)
    assert stats.pv == engine.price(product, env)
    assert stats.maturity.discounted_cash == 0.0 and stats.ki_settlement.discounted_cash == 0.0
    assert stats.pv == pytest.approx(float(np.sum(stats.ledger.discounted_cash)), rel=1e-14)


@pytest.mark.parametrize("name", CASES)
def test_a_knock_out_leg_pays_on_the_decided_knock_out(name):
    product, env = _decided(name)
    leg = FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=1_000.0, trigger=PaymentTrigger.AT_KO,
                         leg_id="ko")

    def legs_pv(engine):
        position = EquityPosition(product=product, quantity=1.0, entry_price=0.0, underlying="X", engine=engine,
                                  entry_timestamp=env.valuation_date, cash_legs=[leg])
        return position.get_trade_value(env) - position.get_market_value(env)

    v1 = legs_pv(KOResetSnowballQuadEngine())
    assert v1 == pytest.approx(legs_pv(KOResetSnowballQuadEngineV2()), rel=1e-12)
    settlement = _decision(product, env).ko_record.settlement_time
    df = env.get_discount_factor(settlement) if settlement and settlement > 0.0 else 1.0
    assert v1 == pytest.approx(1_000.0 * df, rel=1e-12)
