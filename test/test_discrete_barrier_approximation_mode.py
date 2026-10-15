"""The discrete-barrier approximation mode of the analytical barrier-family engines (user decision 2026-10-15).

``DiscreteBarrierApproximation`` switches how a DISCRETE schedule with no record on the valuation instant is valued:

* ``BGK`` (default, "BGK all through"): the Broadie-Glasserman-Kou shifted closed form at today's spot, whatever
  side of the contractual level (or of the shifted level) the spot lies. Nothing is decided between observations
  (A15): the observation type decides, and only a record on the valuation instant decides now.
* ``FIRST_OBSERVATION_BGK`` ("first projection plus BGK"): always the exact Gaussian step through the first
  future observation (``first_observation``), the shifted closed form only from it on -- inside the level too.

Both modes leave CONTINUOUS and EXPIRY monitoring, a record on the valuation instant, T = 0 and the refusals as
they were. The pins below were captured on main 1e32c27c before the switch existed (the hybrid: BGK inside the
level, the first-observation projection beyond it); inside the level the BGK default is bitwise that hybrid.

Contracts as ``test_discrete_barrier_between_observations``: maturity 1.0, twelve monthly records, 20 % vol, r 3 %,
q 1 %; "between" is aged 0.30 (no record today), "record day" aged 0.25 onto the 3-month record.
"""
from __future__ import annotations

import math
import warnings

import pytest

import test_discrete_barrier_between_observations as T
from quantark.asset.equity.engine.analytical import (
    barrier_analytical_engine,
    double_barrier_option_engine,
    double_sharkfin_option_analytical_engine,
    one_touch_analytical_engine,
)
from quantark.asset.equity.engine.analytical.accumulator_analytical_engine import AccumulatorAnalyticalEngine
from quantark.asset.equity.engine.analytical.barrier_analytical_engine import BarrierAnalyticalEngine
from quantark.asset.equity.engine.analytical.double_barrier_option_engine import DoubleBarrierOptionAnalyticalEngine
from quantark.asset.equity.engine.analytical.double_sharkfin_option_analytical_engine import (
    DoubleSharkfinOptionAnalyticalEngine,
)
from quantark.asset.equity.engine.analytical.one_touch_analytical_engine import OneTouchAnalyticalEngine
from quantark.asset.equity.engine.analytical.single_sharkfin_option_analytical_engine import (
    SingleSharkfinOptionAnalyticalEngine,
)
from quantark.asset.equity.product.option import (
    BarrierOption,
    DoubleBarrierOption,
    DoubleSharkfinOption,
    OneTouchOption,
    SingleSharkfinOption,
)
from quantark.util.barrier_shift import apply_barrier_shift
from quantark.util.enum import (
    BarrierDirection,
    BarrierType,
    DoubleBarrierType,
    ObservationType,
    OptionType,
    TouchType,
)
from quantark.util.enum.engine_enums import DiscreteBarrierApproximation
from quantark.util.exceptions import ValidationError

BGK = DiscreteBarrierApproximation.BGK
FIRST = DiscreteBarrierApproximation.FIRST_OBSERVATION_BGK
MODES = [BGK, FIRST]

ENGINES = {
    "barrier": BarrierAnalyticalEngine,
    "touch": OneTouchAnalyticalEngine,
    "double": DoubleBarrierOptionAnalyticalEngine,
    "single_sharkfin": SingleSharkfinOptionAnalyticalEngine,
    "double_sharkfin": DoubleSharkfinOptionAnalyticalEngine,
}
FIVE = list(ENGINES.values())
#: the engine modules that hold a first-observation projection
PROJECTING = [barrier_analytical_engine, one_touch_analytical_engine, double_barrier_option_engine,
              double_sharkfin_option_analytical_engine]
#: the shifted level's factor at 20 % vol on a monthly grid
SHIFT = math.exp(0.5825971579 * 0.20 * math.sqrt(1.0 / 12.0))
#: continuity across the level, spot level -/+ 0.01 (the QUAD / PDE bound of the A15 test; the hybrid stepped by up
#: to 0.152 here)
STEP = 0.01
CONTINUITY_ABS = 0.04


# ======================================================================================================================
# contracts
# ======================================================================================================================

def _disc(obs):
    return dict(maturity=1.0, observation_type=obs)


def barrier(bt, ot, level, rebate=0.0, at_hit=False, obs=ObservationType.DISCRETE):
    def build():
        kw = _disc(obs)
        if obs == ObservationType.DISCRETE:
            kw["observation_schedule"] = T._single_schedule(level)
        return BarrierOption(strike=100.0, option_type=ot, barrier=level, barrier_type=bt, rebate=rebate,
                             pay_at_hit=at_hit, **kw)
    return build


def touch(tt, up, level, rebate=5.0, at_hit=False, obs=ObservationType.DISCRETE):
    def build():
        kw = _disc(obs)
        if obs == ObservationType.DISCRETE:
            kw["observation_schedule"] = T._single_schedule(level)
        return OneTouchOption(barrier=level, barrier_direction=BarrierDirection.UP if up else BarrierDirection.DOWN,
                              rebate=rebate, payment_at_hit=at_hit, touch_type=tt, **kw)
    return build


def double(bt, ot, rebate=0.0, obs=ObservationType.DISCRETE):
    def build():
        kw = _disc(obs)
        if obs == ObservationType.DISCRETE:
            kw["observation_schedule"] = T._double_schedule(130.0, 75.0)
        return DoubleBarrierOption(strike=100.0, option_type=ot, upper_barrier=130.0, lower_barrier=75.0,
                                   barrier_type=bt, rebate=rebate, **kw)
    return build


def ssf(ko=0.5, no_hit=0.0, at_hit=False, obs=ObservationType.DISCRETE):
    def build():
        kw = _disc(obs)
        if obs == ObservationType.DISCRETE:
            kw["observation_schedule"] = T._single_schedule(120.0)
        return SingleSharkfinOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, knock_out_rebate=ko,
                                    no_hit_rebate=no_hit, pay_at_hit=at_hit, **kw)
    return build


def dsf(ko=0.5, no_hit=0.0, at_hit=False, obs=ObservationType.DISCRETE):
    def build():
        kw = _disc(obs)
        if obs == ObservationType.DISCRETE:
            kw["observation_schedule"] = T._double_schedule(130.0, 75.0)
        return DoubleSharkfinOption(strike=100.0, option_type=OptionType.CALL, upper_barrier=130.0, lower_barrier=75.0,
                                    knock_out_rebate=ko, no_hit_rebate=no_hit, pay_at_hit=at_hit, **kw)
    return build


#: name -> (engine, builder, monitored level, side of "beyond")
KINDS = {
    "UOC": ("barrier", barrier(BarrierType.UP_OUT, OptionType.CALL, 120.0), 120.0, +1),
    "UOC_r5": ("barrier", barrier(BarrierType.UP_OUT, OptionType.CALL, 120.0, 5.0), 120.0, +1),
    "UOC_r5_hit": ("barrier", barrier(BarrierType.UP_OUT, OptionType.CALL, 120.0, 5.0, True), 120.0, +1),
    "UIC_r5": ("barrier", barrier(BarrierType.UP_IN, OptionType.CALL, 120.0, 5.0), 120.0, +1),
    "DIP_r5": ("barrier", barrier(BarrierType.DOWN_IN, OptionType.PUT, 85.0, 5.0), 85.0, -1),
    "OT_up": ("touch", touch(TouchType.ONE_TOUCH, True, 120.0), 120.0, +1),
    "OT_up_hit": ("touch", touch(TouchType.ONE_TOUCH, True, 120.0, at_hit=True), 120.0, +1),
    "NT_up": ("touch", touch(TouchType.NO_TOUCH, True, 120.0), 120.0, +1),
    "NT_dn": ("touch", touch(TouchType.NO_TOUCH, False, 85.0), 85.0, -1),
    "DKO_C": ("double", double(DoubleBarrierType.KNOCK_OUT, OptionType.CALL), 130.0, +1),
    "DKO_C_r5": ("double", double(DoubleBarrierType.KNOCK_OUT, OptionType.CALL, 5.0), 130.0, +1),
    "DKI_C": ("double", double(DoubleBarrierType.KNOCK_IN, OptionType.CALL), 130.0, +1),
    "DKI_C_r5": ("double", double(DoubleBarrierType.KNOCK_IN, OptionType.CALL, 5.0), 130.0, +1),
    "SSF": ("single_sharkfin", ssf(), 120.0, +1),
    "SSF_nh_hit": ("single_sharkfin", ssf(0.5, 3.0, True), 120.0, +1),
    "DSF": ("double_sharkfin", dsf(), 130.0, +1),
    "DSF_nh_hit": ("double_sharkfin", dsf(0.5, 3.0, True), 130.0, +1),
}

#: (kind, spot) -> the hybrid's value before the switch, spot inside the level, no record today (aged 0.30)
INSIDE_PINS = {
    ("UOC", 105.6): 2.5484088007979757, ("UOC", 119.5): 0.8708291941804163, ("UOC", 119.99): 0.7800808279621387,
    ("UOC_r5", 105.6): 4.186456206677764, ("UOC_r5", 119.99): 4.894256325686052,
    ("UOC_r5_hit", 119.5): 4.962972659516022, ("UOC_r5_hit", 119.99): 4.968663218445459,
    ("UIC_r5", 105.6): 11.497973441749087, ("UIC_r5", 119.99): 22.302473044972846,
    ("DIP_r5", 95.2): 9.567133121506908, ("DIP_r5", 85.01): 15.305265717574448,
    ("OT_up", 105.6): 1.6380474058797883, ("OT_up", 119.99): 4.114175497723913,
    ("OT_up_hit", 119.5): 4.092143465335606, ("OT_up_hit", 119.99): 4.188582390483321,
    ("NT_up", 119.5): 0.8752919289749146, ("NT_dn", 85.01): 0.7827040553784916,
    ("DKO_C", 114.4): 5.887925975448908, ("DKO_C", 129.99): 1.730166095630075,
    ("DKO_C_r5", 129.5): 5.947177139623101, ("DKO_C_r5", 129.99): 5.846384032292689,
    ("DKI_C", 129.5): 29.164679543355827, ("DKI_C_r5", 129.99): 30.597241454298665,
    ("SSF", 105.6): 2.7122135413859545, ("SSF", 119.99): 1.19149837773453,
    ("SSF_nh_hit", 119.5): 1.8052186980989255, ("SSF_nh_hit", 119.99): 1.6680906620845015,
    ("DSF", 114.4): 6.0548848285737975, ("DSF", 129.99): 2.1417878892963365,
    ("DSF_nh_hit", 129.5): 2.846128862775142, ("DSF_nh_hit", 129.99): 2.6167111243377152,
}

#: (kind, spot) -> the value before the switch, aged onto a record day (0.25): decided now beyond the level
RECORD_DAY_PINS = {
    ("UOC", 119.0): 0.8802717513822058, ("UOC", 121.0): 0.0,
    ("UOC_r5_hit", 119.0): 4.906470913595625, ("UOC_r5_hit", 121.0): 5.0,
    ("UIC_r5", 119.0): 21.68934818446092, ("UIC_r5", 121.0): 23.399089989613955,
    ("DIP_r5", 86.0): 14.720652735785304, ("DIP_r5", 84.0): 15.913008758992675,
    ("OT_up_hit", 119.0): 4.026199162213419, ("OT_up_hit", 121.0): 5.0,
    ("NT_dn", 86.0): 1.0093689592370563, ("NT_dn", 84.0): 0.0,
    ("DKO_C_r5", 129.0): 5.912991873865053, ("DKO_C_r5", 131.0): 4.888756185966682,
    ("DKI_C", 129.0): 28.814749076735502, ("DKI_C", 131.0): 32.65220451947923,
    ("SSF_nh_hit", 119.0): 1.8449542424176417, ("SSF_nh_hit", 121.0): 0.5,
    ("DSF_nh_hit", 129.0): 2.8985310494621315, ("DSF_nh_hit", 131.0): 0.5,
}

#: (engine, builder, spot) -> the value before the switch: CONTINUOUS and EXPIRY monitoring, fresh contracts
UNCHANGED_PINS = [
    ("barrier", barrier(BarrierType.UP_OUT, OptionType.CALL, 120.0, 5.0, obs=ObservationType.CONTINUOUS), 119.0,
     4.7663441024100335),
    ("barrier", barrier(BarrierType.UP_OUT, OptionType.CALL, 120.0, 5.0, obs=ObservationType.CONTINUOUS), 131.0,
     4.85222766774254),
    ("barrier", barrier(BarrierType.UP_IN, OptionType.CALL, 120.0, 5.0, obs=ObservationType.EXPIRY), 121.0,
     23.545768675142014),
    ("barrier", barrier(BarrierType.DOWN_IN, OptionType.PUT, 85.0, 5.0, obs=ObservationType.CONTINUOUS), 90.0,
     12.806054211413004),
    ("touch", touch(TouchType.ONE_TOUCH, True, 120.0, at_hit=True, obs=ObservationType.CONTINUOUS), 119.0,
     4.828401635262244),
    ("touch", touch(TouchType.NO_TOUCH, True, 120.0, obs=ObservationType.EXPIRY), 121.0, 2.3458144754668666),
    ("double", double(DoubleBarrierType.KNOCK_OUT, OptionType.CALL, 5.0, obs=ObservationType.CONTINUOUS), 119.0,
     5.293929763257003),
    ("double", double(DoubleBarrierType.KNOCK_IN, OptionType.CALL, 5.0, obs=ObservationType.EXPIRY), 131.0,
     29.39711777257039),
    ("single_sharkfin", ssf(0.5, 3.0, True, obs=ObservationType.CONTINUOUS), 119.0, 0.6560635964488636),
    ("single_sharkfin", ssf(0.5, 3.0, obs=ObservationType.EXPIRY), 121.0, 4.826620073249491),
    ("double_sharkfin", dsf(0.5, 3.0, True, obs=ObservationType.CONTINUOUS), 119.0, 3.2773006231450017),
    ("double_sharkfin", dsf(0.5, 3.0, obs=ObservationType.EXPIRY), 131.0, 7.982160607945336),
]


def _between(name):
    engine, build, _, _ = KINDS[name]
    product = build()
    product.time_shift(T.BETWEEN, T.VALUATION, T.env(100.0), keep_landing=False)
    return engine, product


def _record_day(name):
    engine, build, _, _ = KINDS[name]
    product = build()
    product.time_shift(T.RECORD_DAY, T.VALUATION, T.env(100.0), keep_landing=True)
    return engine, product


def _price(engine_name, product, spot, mode=None):
    kwargs = {} if mode is None else {"discrete_barrier_approximation": mode}
    engine = ENGINES[engine_name](**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(engine.price(product, T.env(spot)))


class _Spy:
    """Counts the first-observation projections (``through_first_observation``) of every engine module."""

    def __init__(self, monkeypatch):
        self.calls = 0
        for module in PROJECTING:
            original = module.through_first_observation

            def spy(*args, _original=original, **kwargs):
                self.calls += 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(module, "through_first_observation", spy)


# ======================================================================================================================
# the switch
# ======================================================================================================================

def test_the_enum_has_the_two_modes():
    assert {m.name for m in DiscreteBarrierApproximation} == {"BGK", "FIRST_OBSERVATION_BGK"}
    assert str(BGK) == BGK.value


@pytest.mark.parametrize("engine_cls", FIVE + [AccumulatorAnalyticalEngine])
def test_the_default_is_bgk_all_through(engine_cls):
    assert engine_cls().discrete_barrier_approximation is BGK


@pytest.mark.parametrize("engine_cls", FIVE + [AccumulatorAnalyticalEngine])
@pytest.mark.parametrize("bad", ["bgk", "BGK", 0, None, ObservationType.DISCRETE])
def test_a_non_member_is_refused(engine_cls, bad):
    with pytest.raises(ValidationError, match="DiscreteBarrierApproximation"):
        engine_cls(discrete_barrier_approximation=bad)


@pytest.mark.parametrize("mode", MODES)
def test_composite_engines_hand_the_mode_to_their_legs(mode):
    barrier_engine = BarrierAnalyticalEngine(discrete_barrier_approximation=mode)
    single = SingleSharkfinOptionAnalyticalEngine(discrete_barrier_approximation=mode)
    double_sharkfin = DoubleSharkfinOptionAnalyticalEngine(discrete_barrier_approximation=mode)
    accumulator = AccumulatorAnalyticalEngine(discrete_barrier_approximation=mode)
    legs = [barrier_engine._one_touch_engine, single._barrier_engine, single._one_touch_engine,
            single._barrier_engine._one_touch_engine, double_sharkfin._double_barrier_engine,
            accumulator._barrier_engine, accumulator._one_touch_engine]
    assert all(leg.discrete_barrier_approximation is mode for leg in legs)


@pytest.mark.parametrize("engine_cls", FIVE + [AccumulatorAnalyticalEngine])
def test_the_mode_is_part_of_the_model_fingerprint(engine_cls):
    bgk, first = engine_cls(), engine_cls(discrete_barrier_approximation=FIRST)
    assert bgk.model_fingerprint() == engine_cls().model_fingerprint()
    assert bgk.model_fingerprint() != first.model_fingerprint()


# ======================================================================================================================
# BGK all through (the default)
# ======================================================================================================================

@pytest.mark.parametrize("name,spot", list(INSIDE_PINS), ids=[f"{n}-{s}" for n, s in INSIDE_PINS])
def test_bgk_inside_the_level_is_bitwise_the_value_before_the_switch(name, spot, monkeypatch):
    engine, product = _between(name)
    spy = _Spy(monkeypatch)
    assert _price(engine, product, spot) == INSIDE_PINS[(name, spot)]
    assert spy.calls == 0


def _beyond_spots(name):
    _, _, level, side = KINDS[name]
    shifted = level * SHIFT ** side
    return [level + side * STEP, level + side * 1.0, shifted - side * 0.5]


BEYOND_CASES = [(name, spot) for name in KINDS for spot in _beyond_spots(name)]


@pytest.mark.parametrize("name,spot", BEYOND_CASES, ids=[f"{n}-{s:.2f}" for n, s in BEYOND_CASES])
def test_bgk_beyond_the_level_never_projects_and_decides_nothing(name, spot, monkeypatch):
    """Beyond the contractual level with no record today: the shifted closed form at today's spot (no projection
    through the first observation); just beyond the level, never the value of a decision taken now."""
    engine, product = _between(name)
    spy = _Spy(monkeypatch)
    value = _price(engine, product, spot)
    assert spy.calls == 0
    assert math.isfinite(value)
    _, _, level, side = KINDS[name]
    if spot == level + side * STEP and "_r5" not in name:
        # (a rebate of 5 is close to the forward value next to the level: no decision step to see there)
        _, now = _record_day(name)
        decided = _price(engine, now, spot)
        assert abs(value - decided) >= T.DECISION_STEP, (value, decided)


def _shifted(product, level, up):
    schedule = product.observation_schedule
    return apply_barrier_shift(barrier=level, is_up_barrier=up, volatility=0.20,
                               observation_interval=schedule.ensure_regular_frequency(schedule.times))


@pytest.mark.parametrize("spot", [120.01, 121.0, 123.5, 126.0])
def test_bgk_beyond_is_the_shifted_closed_form_at_todays_spot_barrier(spot):
    _, product = _between("UOC")
    environment = T.env(spot)
    engine = BarrierAnalyticalEngine()
    maturity = product.get_maturity(environment)
    expected = max(engine._price_knock_out_closed_form(
        spot=spot, strike=100.0, barrier=_shifted(product, 120.0, True), maturity=maturity,
        rate=environment.get_rate(maturity), carry=environment.get_rate(maturity) - environment.get_div_yield(maturity),
        vol=0.20, is_call=True, is_up=True), 0.0)
    assert _price("barrier", product, spot) == expected


@pytest.mark.parametrize("spot", [120.01, 121.0, 123.5, 126.0])
def test_bgk_beyond_is_the_shifted_closed_form_at_todays_spot_one_touch(spot):
    _, product = _between("OT_up_hit")
    environment = T.env(spot)
    engine = OneTouchAnalyticalEngine()
    maturity = product.get_maturity(environment)
    expected = engine._one_touch_price(
        spot=spot, barrier=_shifted(product, 120.0, True), maturity=maturity, rate=environment.get_rate(maturity),
        div=environment.get_div_yield(maturity), vol=0.20, rebate=5.0, pay_at_hit=True, is_up=True)
    assert _price("touch", product, spot) == expected


@pytest.mark.parametrize("spot", [130.01, 131.0, 134.0, 140.0])
def test_bgk_beyond_is_the_shifted_closed_form_at_todays_spot_double(spot):
    _, product = _between("DKO_C")
    environment = T.env(spot)
    engine = DoubleBarrierOptionAnalyticalEngine()
    maturity = product.get_maturity(environment)
    expected = max(engine._price_knock_out_ikeda_kuintomo(
        S=spot, K=100.0, L=_shifted(product, 75.0, False), U=_shifted(product, 130.0, True), T=maturity,
        r=environment.get_rate(maturity), b=environment.get_rate(maturity) - environment.get_div_yield(maturity),
        sigma=0.20, is_call=True), 0.0)
    assert _price("double", product, spot) == expected


@pytest.mark.parametrize("name", ["UOC", "UOC_r5_hit", "UIC_r5", "DIP_r5", "OT_up_hit", "NT_dn", "DKO_C_r5",
                                  "DKI_C", "SSF_nh_hit", "DSF_nh_hit"])
@pytest.mark.parametrize("mode", MODES)
def test_both_modes_are_continuous_across_the_level(name, mode):
    """Spot level -/+ 0.01, no record today: the step is the price's slope x 0.02, never the hybrid's switch."""
    engine, product = _between(name)
    _, _, level, side = KINDS[name]
    inside = _price(engine, product, level - side * STEP, mode)
    _, product = _between(name)
    beyond = _price(engine, product, level + side * STEP, mode)
    assert abs(beyond - inside) <= CONTINUITY_ABS, (inside, beyond)


def test_bgk_is_continuous_across_the_shifted_level():
    for name in ("UOC_r5_hit", "OT_up_hit", "NT_dn", "UIC_r5"):
        engine, _, level, side = KINDS[name]
        shifted = level * SHIFT ** side
        values = []
        for spot in (shifted - side * STEP, shifted + side * STEP):
            _, product = _between(name)
            values.append(_price(engine, product, spot))
        assert abs(values[1] - values[0]) <= CONTINUITY_ABS, (name, values)


@pytest.mark.parametrize("name", ["DKO_C_r5", "DKI_C_r5", "DSF", "DSF_nh_hit"])
def test_bgk_beyond_the_shifted_corridor_with_cash_is_a_todo(name):
    """The double no-touch probability is defined (and error-bounded) only inside its corridor; beyond the shifted
    corridor its "outside: 0" branch would decide the knock-out now. Refused, not substituted."""
    engine, product = _between(name)
    with pytest.raises(ValidationError, match=r"TODO\(bgk-beyond\)"):
        _price(engine, product, 130.0 * SHIFT + 1.0)


@pytest.mark.parametrize("name", ["DKO_C", "DKI_C"])
def test_bgk_beyond_the_shifted_corridor_without_cash_evaluates_the_series(name):
    engine, product = _between(name)
    value = _price(engine, product, 130.0 * SHIFT + 1.0)
    assert math.isfinite(value)


# ======================================================================================================================
# first projection plus BGK (opt-in)
# ======================================================================================================================

FIRST_INSIDE = [("UOC", 119.99), ("UOC_r5_hit", 119.5), ("UIC_r5", 105.6), ("DIP_r5", 85.01), ("OT_up", 119.99),
                ("NT_dn", 85.01), ("DKO_C_r5", 129.99), ("DKI_C", 129.5), ("SSF_nh_hit", 119.99),
                ("DSF_nh_hit", 129.5)]


@pytest.mark.parametrize("name,spot", FIRST_INSIDE, ids=[f"{n}-{s}" for n, s in FIRST_INSIDE])
def test_first_observation_mode_projects_inside_the_level(name, spot, monkeypatch):
    """Inside the level too: the value through the first observation, so it differs from the BGK closed form."""
    engine, product = _between(name)
    spy = _Spy(monkeypatch)
    value = _price(engine, product, spot, FIRST)
    assert spy.calls >= 1
    assert value != INSIDE_PINS.get((name, spot), _price(engine, _between(name)[1], spot, BGK))


@pytest.mark.parametrize("name", ["UOC", "OT_up_hit", "DKO_C_r5", "DSF_nh_hit"])
def test_first_observation_mode_beyond_the_level_is_the_hybrid_projection(name):
    """Beyond the level the opt-in mode is the hybrid's projection, bitwise."""
    pins = {"UOC": (121.0, 0.5696210500807283), "OT_up_hit": (121.0, 4.406089984566929),
            "DKO_C_r5": (131.0, 5.603763010485712), "DSF_nh_hit": (131.0, 2.0756207036907224)}
    spot, before = pins[name]
    engine, product = _between(name)
    assert _price(engine, product, spot, FIRST) == before


def test_first_observation_mode_single_sharkfin_legs_project(monkeypatch):
    engine, product = _between("SSF_nh_hit")
    spy = _Spy(monkeypatch)
    _price(engine, product, 110.0, FIRST)
    assert spy.calls >= 2                      # the knock-out leg and the touch legs


# ======================================================================================================================
# unchanged in both modes
# ======================================================================================================================

@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name,spot", list(RECORD_DAY_PINS), ids=[f"{n}-{s}" for n, s in RECORD_DAY_PINS])
def test_a_record_on_the_valuation_instant_decides_as_before(name, spot, mode):
    engine, product = _record_day(name)
    before = RECORD_DAY_PINS[(name, spot)]
    _, _, level, side = KINDS[name]
    if (spot - level) * side < 0 and mode is FIRST:
        # inside on a record day the record is not crossed: the opt-in mode projects through the NEXT record
        assert math.isfinite(_price(engine, product, spot, mode))
        return
    assert _price(engine, product, spot, mode) == before


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", range(len(UNCHANGED_PINS)))
def test_continuous_and_expiry_monitoring_are_unchanged(case, mode, monkeypatch):
    engine, build, spot, before = UNCHANGED_PINS[case]
    spy = _Spy(monkeypatch)
    assert _price(engine, build(), spot, mode) == before
    assert spy.calls == 0


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("engine_name,build", [
    ("barrier", barrier(BarrierType.UP_OUT, OptionType.CALL, 120.0, 5.0)),
    ("touch", touch(TouchType.ONE_TOUCH, True, 120.0)),
    ("double", double(DoubleBarrierType.KNOCK_OUT, OptionType.CALL, 5.0)),
    ("single_sharkfin", ssf(0.5, 3.0)),
    ("double_sharkfin", dsf(0.5, 3.0)),
])
def test_maturity_settlement_is_unchanged(engine_name, build, mode):
    product = build()
    product.time_shift(1.0, T.VALUATION, T.env(100.0), keep_landing=True)
    assert _price(engine_name, product, 125.0, mode) == _price(engine_name, product, 125.0)
