"""The analytic barrier engines' observation rules (3C handoff, section 12).

1. With no observation due, a DISCRETELY monitored barrier is valued through the next observation
   (``first_observation``) whichever side of the level the spot is on - exact to that observation, the engine's
   own barrier-shifted formula after it; a spot beyond the level is not a hit. Checked beyond the level against
   the exact recursion of ``test/pde_routes/barrier_reference``.
2. At maturity only what is observed now decides: a discrete schedule's records at the valuation instant, each at
   its own level and cash; an untouched knock-in pays its rebate.
"""
from __future__ import annotations

import pytest

from pde_routes.barrier_reference import barrier_reference
from pde_routes.cases import CONST, env_const

from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
from quantark.asset.equity.product.option import BarrierOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.util.enum import BarrierType, ObservationAggregation, ObservationType, OptionType
from quantark.util.exceptions import PricingError

VOL, R, Q = CONST["vol"], CONST["rate"], CONST["div"]
BT = {"UO": BarrierType.UP_OUT, "UI": BarrierType.UP_IN, "DO": BarrierType.DOWN_OUT, "DI": BarrierType.DOWN_IN}
MONTHLY = [k / 12 for k in range(1, 13)]
AGED = [0.04 + k / 12 for k in range(12)]


def barrier(kind, call, level, obs, rebate=0.0, at_hit=False, multiplier=1.0):
    return BarrierOption(strike=100.0, option_type=OptionType.CALL if call else OptionType.PUT, barrier=level,
                         barrier_type=BT[kind], maturity=float(obs[-1]), rebate=rebate, pay_at_hit=at_hit,
                         observation_type=ObservationType.DISCRETE, observation_dates=list(obs),
                         contract_multiplier=multiplier)


def price(product, spot):
    return float(BarrierAnalyticalEngine().price(product, env_const(spot)))


BEYOND = [
    ("UO", True, 120.0, 121.0), ("UO", True, 120.0, 130.0), ("DO", False, 85.0, 84.0),
    ("UI", True, 120.0, 121.0), ("DI", False, 85.0, 80.0), ("UO", False, 120.0, 121.0), ("DO", True, 85.0, 84.0),
]


@pytest.mark.parametrize("obs", [MONTHLY, AGED], ids=["monthly", "aged"])
@pytest.mark.parametrize("rebate,at_hit", [(0.0, False), (5.0, False), (5.0, True)])
@pytest.mark.parametrize("kind,call,level,spot", BEYOND)
def test_beyond_a_discrete_barrier_is_valued_through_the_next_observation(kind, call, level, spot, rebate, at_hit, obs):
    if kind[1] == "I" and at_hit:
        pytest.skip("a knock-in's rebate is paid at maturity")
    exact = barrier_reference(kind=kind, call=call, strike=100.0, level=level, rebate=rebate, at_hit=at_hit,
                              obs=obs, spots=[spot], vol=VOL, r=R, q=Q)[spot][0]
    # measured 4.4e-2 at worst on 74 cells (today: up to 1.41); the shift's own error inside is up to 0.24
    assert price(barrier(kind, call, level, obs, rebate, at_hit), spot) == pytest.approx(exact, abs=0.05)


def test_the_value_is_continuous_across_the_barrier_within_the_shift_error():
    """Both sides are the expectation through the next observation (since 2026-10-15; before, the shifted formula
    just inside): no step at the level beyond the bound kept from that switch."""
    product = barrier("UO", True, 120.0, MONTHLY, 5.0)
    inside, beyond = price(product, 119.999), price(product, 120.001)
    assert abs(beyond - inside) < 0.1


def test_the_multiplier_scales_the_value_beyond_the_barrier():
    one = price(barrier("DI", False, 85.0, MONTHLY, 5.0), 80.0)
    assert price(barrier("DI", False, 85.0, MONTHLY, 5.0, multiplier=10.0), 80.0) == pytest.approx(10.0 * one, rel=1e-12)


def test_an_observation_at_the_valuation_instant_still_decides_on_the_spot():
    product = barrier("UO", True, 120.0, [0.0] + MONTHLY, 5.0)
    assert price(product, 125.0) == pytest.approx(5.0 * env_const(125.0).get_discount_factor(1.0), rel=1e-12)


def test_accumulate_beyond_the_barrier_is_refused():
    sch = ObservationSchedule.from_legacy(observation_dates=MONTHLY, default_barrier=120.0, default_payoff=5.0,
                                          aggregation_mode=ObservationAggregation.ACCUMULATE)
    product = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BT["UO"],
                            maturity=1.0, rebate=5.0, observation_type=ObservationType.DISCRETE,
                            observation_schedule=sch)
    with pytest.raises(PricingError, match="STOP_FIRST_HIT"):
        price(product, 125.0)


# --- at maturity (3C evidence z5 / z5b: strike 100, up barrier 130, rebate 5) -------------------------------

def at_maturity(bt, record=True, **per_record):
    """A tenor product at T = 0; ``record``: its schedule observes at the valuation instant."""
    p = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=bt, maturity=1.0,
                      rebate=5.0, observation_type=ObservationType.DISCRETE,
                      observation_schedule=ObservationSchedule(records=[ObservationRecord(observation_time=0.5),
                                                                        ObservationRecord(observation_time=1.0)]))
    fields = dict(dict(barrier=130.0, payoff=5.0), **per_record)   # a constructed product's records carry them
    records = [ObservationRecord(observation_time=0.0 if record else -0.5, **fields)]
    p.observation_schedule = ObservationSchedule(records=records)
    p.observation_dates = [r.observation_time for r in records]
    p.maturity = 0.0
    return p


@pytest.mark.parametrize("factory,spot,truth", [
    (lambda: at_maturity(BarrierType.UP_OUT), 135.0, 5.0),
    (lambda: at_maturity(BarrierType.UP_OUT), 125.0, 25.0),
    (lambda: at_maturity(BarrierType.UP_OUT, record=False), 135.0, 35.0),
    (lambda: at_maturity(BarrierType.UP_OUT, barrier=120.0, payoff=7.0), 125.0, 7.0),
    (lambda: at_maturity(BarrierType.UP_OUT, barrier=120.0, payoff=7.0), 115.0, 15.0),
    (lambda: at_maturity(BarrierType.UP_IN), 135.0, 35.0),
    (lambda: at_maturity(BarrierType.UP_IN), 125.0, 5.0),
    (lambda: at_maturity(BarrierType.UP_IN, record=False), 135.0, 5.0),
    (lambda: at_maturity(BarrierType.UP_IN, record=False), 125.0, 5.0),
], ids=["ko-hit", "ko-alive", "ko-no-record", "ko-record-level-hit", "ko-record-level-alive", "ki-hit", "ki-rebate",
        "ki-no-record-beyond", "ki-no-record-inside"])
def test_discrete_at_maturity_observes_only_the_records_now(factory, spot, truth):
    assert price(factory(), spot) == pytest.approx(truth, abs=1e-12)


@pytest.mark.parametrize("obs_type", [ObservationType.CONTINUOUS, ObservationType.EXPIRY])
@pytest.mark.parametrize("spot,truth", [(135.0, 35.0), (125.0, 5.0)])
def test_an_untouched_knock_in_pays_its_rebate_at_maturity(obs_type, spot, truth):
    p = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, barrier_type=BarrierType.UP_IN,
                      maturity=1.0, rebate=5.0, observation_type=obs_type)
    p.maturity = 0.0
    assert price(p, spot) == pytest.approx(truth, abs=1e-12)


# --- one-touch / no-touch beyond a discrete barrier --------------------------------------------------------

def touch(one_touch, up, level, obs, rebate=5.0, at_hit=False):
    from quantark.asset.equity.product.option import OneTouchOption
    from quantark.util.enum import BarrierDirection, TouchType

    return OneTouchOption(barrier=level, barrier_direction=BarrierDirection.UP if up else BarrierDirection.DOWN,
                          maturity=float(obs[-1]), rebate=rebate, payment_at_hit=at_hit,
                          touch_type=TouchType.ONE_TOUCH if one_touch else TouchType.NO_TOUCH,
                          observation_type=ObservationType.DISCRETE, observation_dates=list(obs))


def touch_exact(one_touch, up, level, obs, spot, rebate=5.0, at_hit=False):
    from pde_routes.barrier_reference import knock_out

    edges = dict(lower=None if up else level, upper=level if up else None)
    common = dict(strike=100.0, obs=obs, spots=[spot], vol=VOL, r=R, q=Q, **edges)
    if one_touch:
        return knock_out(payoff="zero", cash=rebate, at_hit=at_hit, **common)[0][spot][0]
    return rebate * knock_out(payoff="one", cash=0.0, at_hit=False, **common)[0][spot][0]


@pytest.mark.parametrize("obs", [MONTHLY, AGED], ids=["monthly", "aged"])
@pytest.mark.parametrize("one_touch,at_hit", [(True, False), (True, True), (False, False)])
@pytest.mark.parametrize("up,level,spot", [(True, 120.0, 121.0), (True, 120.0, 128.0), (False, 85.0, 84.0)])
def test_touch_beyond_a_discrete_barrier_is_valued_through_the_next_observation(up, level, spot, one_touch, at_hit, obs):
    from quantark.asset.equity.engine.analytical.one_touch_analytical_engine import OneTouchAnalyticalEngine

    exact = touch_exact(one_touch, up, level, obs, spot, at_hit=at_hit)
    value = float(OneTouchAnalyticalEngine().price(touch(one_touch, up, level, obs, at_hit=at_hit), env_const(spot)))
    # measured 2.2e-2 at worst on these 18 cells (today: up to 0.74)
    assert value == pytest.approx(exact, abs=0.05)


# --- double barrier outside a discrete corridor, and at maturity --------------------------------------------

def double(knock_out, call, obs, rebate=5.0, lower=80.0, upper=130.0):
    from quantark.asset.equity.product.option import DoubleBarrierOption
    from quantark.util.enum import DoubleBarrierType

    return DoubleBarrierOption(strike=100.0, option_type=OptionType.CALL if call else OptionType.PUT,
                               upper_barrier=upper, lower_barrier=lower,
                               barrier_type=DoubleBarrierType.KNOCK_OUT if knock_out else DoubleBarrierType.KNOCK_IN,
                               maturity=float(obs[-1]), rebate=rebate, observation_type=ObservationType.DISCRETE,
                               observation_dates=list(obs))


def double_exact(knock_out, call, obs, spot, rebate=5.0, lower=80.0, upper=130.0):
    from pde_routes.barrier_reference import _bs, knock_out as ko

    common = dict(strike=100.0, obs=obs, spots=[spot], vol=VOL, r=R, q=Q, lower=lower, upper=upper)
    payoff = "call" if call else "put"
    if knock_out:
        return ko(payoff=payoff, cash=rebate, at_hit=False, **common)[0][spot][0]
    van = float(_bs(call, spot, 100.0, float(obs[-1]), VOL, R, Q)[0])
    return (van - ko(payoff=payoff, cash=0.0, at_hit=False, **common)[0][spot][0]
            + rebate * ko(payoff="one", cash=0.0, at_hit=False, **common)[0][spot][0])


def double_price(product, spot):
    from quantark.asset.equity.engine.analytical.double_barrier_option_engine import DoubleBarrierOptionAnalyticalEngine

    return float(DoubleBarrierOptionAnalyticalEngine().price(product, env_const(spot)))


@pytest.mark.parametrize("obs", [MONTHLY, AGED], ids=["monthly", "aged"])
@pytest.mark.parametrize("knock_out", [True, False], ids=["ko", "ki"])
@pytest.mark.parametrize("call", [True, False], ids=["call", "put"])
@pytest.mark.parametrize("spot", [131.0, 135.0, 79.0, 75.0])
def test_double_outside_a_discrete_corridor_is_valued_through_the_next_observation(spot, call, knock_out, obs):
    exact = double_exact(knock_out, call, obs, spot)
    # measured 2.0e-2 at worst on these 32 cells (today: up to 0.44); the shift's own error inside is up to 0.096
    assert double_price(double(knock_out, call, obs), spot) == pytest.approx(exact, abs=0.05)


def double_at_maturity(knock_out, record=True, **per_record):
    p = double(knock_out, True, [0.5, 1.0])
    fields = dict(dict(upper_barrier=130.0, lower_barrier=80.0, payoff=5.0), **per_record)
    p.observation_schedule = ObservationSchedule(records=[ObservationRecord(observation_time=0.0 if record else -0.5, **fields)])
    p.observation_dates = [p.observation_schedule.records[0].observation_time]
    p.maturity = 0.0
    return p


@pytest.mark.parametrize("factory,spot,truth", [
    (lambda: double_at_maturity(True), 135.0, 5.0),
    (lambda: double_at_maturity(True), 125.0, 25.0),
    (lambda: double_at_maturity(True, record=False), 135.0, 35.0),
    (lambda: double_at_maturity(True, upper_barrier=120.0), 125.0, 5.0),
    (lambda: double_at_maturity(False), 135.0, 35.0),
    (lambda: double_at_maturity(False), 125.0, 5.0),
    (lambda: double_at_maturity(False, record=False), 135.0, 5.0),
], ids=["ko-hit", "ko-alive", "ko-no-record", "ko-record-level", "ki-hit", "ki-rebate", "ki-no-record"])
def test_double_discrete_at_maturity_observes_only_the_records_now(factory, spot, truth):
    assert double_price(factory(), spot) == pytest.approx(truth, abs=1e-12)


# --- double sharkfin outside a discrete corridor ------------------------------------------------------------

def sharkfin(call, obs, at_hit, ko_cash=5.0, no_hit=3.0, participation=0.8):
    from quantark.asset.equity.product.option import DoubleSharkfinOption

    return DoubleSharkfinOption(strike=100.0, option_type=OptionType.CALL if call else OptionType.PUT,
                                lower_barrier=80.0, upper_barrier=130.0, maturity=float(obs[-1]),
                                participation_rate=participation, knock_out_rebate=ko_cash, no_hit_rebate=no_hit,
                                pay_at_hit=at_hit, observation_type=ObservationType.DISCRETE,
                                observation_dates=list(obs))


def sharkfin_exact(call, obs, at_hit, spot, ko_cash=5.0, no_hit=3.0, participation=0.8):
    from pde_routes.barrier_reference import knock_out as ko

    common = dict(strike=100.0, obs=obs, spots=[spot], vol=VOL, r=R, q=Q, lower=80.0, upper=130.0)
    return (participation * ko(payoff="call" if call else "put", cash=0.0, at_hit=False, **common)[0][spot][0]
            + ko(payoff="zero", cash=ko_cash, at_hit=at_hit, **common)[0][spot][0]
            + no_hit * ko(payoff="one", cash=0.0, at_hit=False, **common)[0][spot][0])


@pytest.mark.parametrize("obs", [MONTHLY, AGED], ids=["monthly", "aged"])
@pytest.mark.parametrize("at_hit", [False, True], ids=["at-maturity", "at-hit"])
@pytest.mark.parametrize("call", [True, False], ids=["call", "put"])
@pytest.mark.parametrize("spot", [131.0, 79.0])
def test_sharkfin_outside_a_discrete_corridor_is_valued_through_the_next_observation(spot, call, at_hit, obs):
    from quantark.asset.equity.engine.analytical.double_sharkfin_option_analytical_engine import (
        DoubleSharkfinOptionAnalyticalEngine,
    )

    exact = sharkfin_exact(call, obs, at_hit, spot)
    value = float(DoubleSharkfinOptionAnalyticalEngine().price(sharkfin(call, obs, at_hit), env_const(spot)))
    assert value == pytest.approx(exact, abs=0.05)


def _sharkfin_at_maturity(single, record, level_override=None):
    """Call sharkfin, strike 100, upper 130 (lower 80), ko cash 5, no-hit cash 3, participation 0.8, at T = 0."""
    if single:
        from quantark.asset.equity.product.option import SingleSharkfinOption

        p = SingleSharkfinOption(strike=100.0, option_type=OptionType.CALL, barrier=130.0, maturity=1.0,
                                 participation_rate=0.8, knock_out_rebate=5.0, no_hit_rebate=3.0,
                                 observation_type=ObservationType.DISCRETE, observation_dates=[0.5, 1.0])
        fields = dict(barrier=level_override or 130.0, payoff=5.0)
    else:
        p = sharkfin(True, [0.5, 1.0], False)
        fields = dict(upper_barrier=level_override or 130.0, lower_barrier=80.0, payoff=5.0)
    p.observation_schedule = ObservationSchedule(records=[ObservationRecord(observation_time=0.0 if record else -0.5, **fields)])
    p.observation_dates = [p.observation_schedule.records[0].observation_time]
    p.maturity = 0.0
    return p


@pytest.mark.parametrize("single", [True, False], ids=["single", "double"])
@pytest.mark.parametrize("record,level,spot,truth", [
    (True, None, 135.0, 5.0),                 # observed now and breached: the knock-out cash
    (True, None, 125.0, 3.0 + 0.8 * 25.0),    # observed now, inside
    (False, None, 135.0, 3.0 + 0.8 * 30.0),   # nothing observed now: the capped no-hit payoff
    (True, 120.0, 125.0, 5.0),                # the record's own level decides
], ids=["hit", "alive", "no-record", "record-level"])
def test_sharkfin_discrete_at_maturity_observes_only_the_records_now(single, record, level, spot, truth):
    from quantark.asset.equity.engine.analytical.double_sharkfin_option_analytical_engine import (
        DoubleSharkfinOptionAnalyticalEngine,
    )
    from quantark.asset.equity.engine.analytical.single_sharkfin_option_analytical_engine import (
        SingleSharkfinOptionAnalyticalEngine,
    )

    engine = SingleSharkfinOptionAnalyticalEngine() if single else DoubleSharkfinOptionAnalyticalEngine()
    assert float(engine.price(_sharkfin_at_maturity(single, record, level), env_const(spot))) == pytest.approx(truth, abs=1e-12)
