"""QUAD v1 knock-out-reset: the event route keeps working once a knock-out schedule has ended.

Both ends are legal states: past its first schedule only a knocked-in contract is alive, and its second schedule
alone can knock it out; an aged-out second schedule leaves a knock-in no knock-out before maturity. The event
statistics returned nothing when either schedule had no record left, so ``price_with_events`` fell back to a
maturity-only distribution: a knocked-in note past its first schedule had P(maturity, no knock-out) = 1, its
knock-out legs raised and its no-knock-out legs were paid in full, every day for the rest of its life. QUAD v2 is
the reference; the two lattices differ by their discretisation.
"""
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.v2 import KOResetSnowballQuadEngineV2
from quantark.asset.equity.product.option.observation_schedule import ObservationSchedule
from quantark.cashleg import EventType, LegDirection
from quantark.cashleg.fixed_payoff_leg import FixedPayoffLeg, PaymentTrigger
from quantark.portfolio import EquityPosition
from quantark.util.calendar import CalendarType, create_calendar

from intraday.conftest import dated_ko_reset, flat_env

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
T0 = datetime(2026, 3, 16)
#: QUAD v1 against v2 on a probability (default lattices)
PAIR = 2e-3


def _continuous_ki(product):
    product.barrier_config = replace(product.barrier_config, ki_continuous=True, ki_observation_schedule=None)
    product.validate()
    return product


def _first_ended(ki="discrete"):
    """Knocked in, ten days past the first schedule's last date: the second schedule alone is in force."""
    product = dated_ko_reset(CAL, T0)
    if ki == "continuous":
        product = _continuous_ki(product)
    product._otc_lifecycle_knocked_in = True
    last = product.barrier_config.ko_observation_schedule.records[-1].observation_date
    return product, flat_env(last + timedelta(days=10), spot=90.0)


def _second_ended(knocked_in):
    """The second schedule's two dates are behind the valuation date; the first schedule is still live."""
    product = dated_ko_reset(CAL, T0)
    post = product.post_barrier_config
    early = post.ko_observation_schedule.records[:2]
    product.post_barrier_config = replace(post, ko_observation_schedule=ObservationSchedule(records=list(early)))
    product.validate()
    if knocked_in:
        product._otc_lifecycle_knocked_in = True
    return product, flat_env(early[-1].observation_date + timedelta(days=5), spot=90.0)


CASES = {
    "first ended, discrete knock-in": lambda: _first_ended(),
    "first ended, continuous knock-in": lambda: _first_ended("continuous"),
    "second ended, not knocked in": lambda: _second_ended(False),
    "second ended, knocked in": lambda: _second_ended(True),
}


def _distributions(name):
    product, env = CASES[name]()
    v1 = KOResetSnowballQuadEngine().price_with_events(product, env).event_distribution
    v2 = KOResetSnowballQuadEngineV2().price_with_events(product, env).event_distribution
    return v1, v2


@pytest.mark.parametrize("name", CASES)
def test_the_distribution_is_v2s(name):
    v1, v2 = _distributions(name)
    assert np.array_equal(v1.stream_times[EventType.KO], v2.stream_times[EventType.KO])
    np.testing.assert_allclose(v1.probabilities[EventType.KO], v2.probabilities[EventType.KO], rtol=0, atol=PAIR)
    for maturity in (EventType.MATURITY_NO_KO, EventType.MATURITY_WITH_KI):
        assert v1.probabilities[maturity] == pytest.approx(v2.probabilities[maturity], abs=PAIR), maturity.name


def test_a_knocked_in_note_past_its_first_schedule_cannot_mature_without_a_knock_in():
    v1, _ = _distributions("first ended, discrete knock-in")
    assert v1.probabilities[EventType.MATURITY_NO_KO] == 0.0
    assert float(np.sum(v1.probabilities[EventType.KO])) > 0.5


@pytest.mark.parametrize("name", CASES)
def test_the_ledger_is_the_price(name):
    product, env = CASES[name]()
    engine = KOResetSnowballQuadEngine()
    stats = engine.calculate_event_stats(product, env)
    assert stats.pv == engine.price(product, env)
    assert stats.pv == pytest.approx(float(np.sum(stats.ledger.discounted_cash)), rel=1e-12)


@pytest.mark.parametrize("trigger", [PaymentTrigger.AT_KO, PaymentTrigger.AT_MATURITY_NO_KO,
                                     PaymentTrigger.AT_MATURITY_WITH_KI])
def test_legs_on_a_knocked_in_note_past_its_first_schedule_value_as_on_v2(trigger):
    product, env = _first_ended()
    leg = FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=1_000.0, trigger=trigger, leg_id="leg")

    def legs_pv(engine):
        position = EquityPosition(product=product, quantity=1.0, entry_price=0.0, underlying="X", engine=engine,
                                  entry_timestamp=env.valuation_date, cash_legs=[leg])
        return position.get_trade_value(env) - position.get_market_value(env)

    assert legs_pv(KOResetSnowballQuadEngine()) == pytest.approx(legs_pv(KOResetSnowballQuadEngineV2()),
                                                                 abs=1_000.0 * PAIR)
