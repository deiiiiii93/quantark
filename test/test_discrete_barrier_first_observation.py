"""A DISCRETE barrier between observations: through the first observation, then BGK (one method; user decision
2026-10-15).

The analytical barrier-family engines (barrier, one-touch, double barrier, single and double sharkfin, and the
accumulator's barrier legs) value a DISCRETE schedule that has no record deciding now and a future observation
left in one way, whichever side of the level the spot is on: the exact Gaussian step to the first future
observation ``t1`` and the barrier test there (``first_observation``), with the Broadie-Glasserman-Kou shifted
closed form only from ``t1`` on. Before this change (main 1e32c27c) they were a hybrid: the shifted closed form at
today's spot inside the level, the projection only beyond it. Beyond the level the value is therefore bitwise the
main value. Inside it moves from the BGK closed form to the projection (closer to QUAD). The step at the level, up
to 0.161 on these contracts, goes away.

Unchanged: a record on the valuation instant that the spot breaches decides at once (W18c / A15), as do
CONTINUOUS and EXPIRY monitoring, T = 0, a carried knock-in and the refusals. With no future observation left the
closed form is kept.

Contracts as ``test_discrete_barrier_between_observations``: maturity 1.0, twelve monthly records, 20 % vol,
r 3 %, q 1 %. "Between" is aged 0.30 (no record today), "record day" aged 0.25 onto the 3-month record. The pins
were taken on main 1e32c27c.
"""
from __future__ import annotations

import math
import warnings

import pytest

import test_discrete_barrier_between_observations as T
from golden_compare import assert_close
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
from quantark.asset.equity.engine.quad import DiscreteQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option import (
    AccumulatorOption,
    BarrierOption,
    DoubleBarrierOption,
    DoubleSharkfinOption,
    OneTouchOption,
    SingleSharkfinOption,
)
from quantark.util.enum import (
    AccumulatorKnockOutType,
    BarrierDirection,
    BarrierType,
    DoubleBarrierType,
    ObservationAggregation,
    ObservationType,
    OptionType,
    TouchType,
)
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.util.exceptions import PricingError

ENGINES = {
    "barrier": BarrierAnalyticalEngine,
    "touch": OneTouchAnalyticalEngine,
    "double": DoubleBarrierOptionAnalyticalEngine,
    "single_sharkfin": SingleSharkfinOptionAnalyticalEngine,
    "double_sharkfin": DoubleSharkfinOptionAnalyticalEngine,
    "accumulator": AccumulatorAnalyticalEngine,
}
#: the engine modules that hold a first-observation projection
PROJECTING = [barrier_analytical_engine, one_touch_analytical_engine, double_barrier_option_engine,
              double_sharkfin_option_analytical_engine]
#: the BGK shift factor at 20 % vol on a monthly grid
SHIFT = math.exp(0.5825971579 * 0.20 * math.sqrt(1.0 / 12.0))
#: continuity across the level, spot level -/+ 0.01
STEP = 0.01
CONTINUITY_ABS = 0.04
#: the engine's step at the level vs QUAD's (measured <= 0.0015 on these contracts)
STEP_VS_QUAD_ABS = 0.005


# ======================================================================================================================
# contracts
# ======================================================================================================================

def _kw(obs):
    return dict(maturity=1.0, observation_type=obs)


def barrier(bt, ot, level, rebate=0.0, at_hit=False, obs=ObservationType.DISCRETE):
    def build():
        kw = _kw(obs)
        if obs == ObservationType.DISCRETE:
            kw["observation_schedule"] = T._single_schedule(level)
        return BarrierOption(strike=100.0, option_type=ot, barrier=level, barrier_type=bt, rebate=rebate,
                             pay_at_hit=at_hit, **kw)
    return build


def touch(tt, up, level, rebate=5.0, at_hit=False, obs=ObservationType.DISCRETE):
    def build():
        kw = _kw(obs)
        if obs == ObservationType.DISCRETE:
            kw["observation_schedule"] = T._single_schedule(level)
        return OneTouchOption(barrier=level, barrier_direction=BarrierDirection.UP if up else BarrierDirection.DOWN,
                              rebate=rebate, payment_at_hit=at_hit, touch_type=tt, **kw)
    return build


def double(bt, ot, rebate=0.0, obs=ObservationType.DISCRETE):
    def build():
        kw = _kw(obs)
        if obs == ObservationType.DISCRETE:
            kw["observation_schedule"] = T._double_schedule(130.0, 75.0)
        return DoubleBarrierOption(strike=100.0, option_type=ot, upper_barrier=130.0, lower_barrier=75.0,
                                   barrier_type=bt, rebate=rebate, **kw)
    return build


def ssf(ko=0.5, no_hit=0.0, at_hit=False, obs=ObservationType.DISCRETE):
    def build():
        kw = _kw(obs)
        if obs == ObservationType.DISCRETE:
            kw["observation_schedule"] = T._single_schedule(120.0)
        return SingleSharkfinOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, knock_out_rebate=ko,
                                    no_hit_rebate=no_hit, pay_at_hit=at_hit, **kw)
    return build


def dsf(ko=0.5, no_hit=0.0, at_hit=False, obs=ObservationType.DISCRETE, ot=OptionType.CALL):
    def build():
        kw = _kw(obs)
        if obs == ObservationType.DISCRETE:
            kw["observation_schedule"] = T._double_schedule(130.0, 75.0)
        return DoubleSharkfinOption(strike=100.0, option_type=ot, upper_barrier=130.0, lower_barrier=75.0,
                                    knock_out_rebate=ko, no_hit_rebate=no_hit, pay_at_hit=at_hit, **kw)
    return build


def accumulator():
    return AccumulatorOption(strike=95.0, knock_out_barrier=110.0, option_type=OptionType.CALL, maturity=1.0,
                             daily_share_accumulation=1.0, gearing=2.0,
                             knock_out_type=AccumulatorKnockOutType.TERMINATION,
                             observation_dates=[k / 12 for k in range(1, 13)])


#: name -> (engine, builder, monitored level, side of "beyond")
KINDS = {
    "UOC": ("barrier", barrier(BarrierType.UP_OUT, OptionType.CALL, 120.0), 120.0, +1),
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
    "DSF_put_lo": ("double_sharkfin", dsf(0.5, 3.0, False, ot=OptionType.PUT), 75.0, -1),
    "ACC_term": ("accumulator", accumulator, 110.0, +1),
}

#: (kind, spot) -> main 1e32c27c, aged 0.30, spot at or beyond the level: the first-observation projection there
BEYOND_PINS = {
    ("UOC", 120.01): 0.7194654095264597,
    ("UOC", 121.0): 0.5696210500807283,
    ("UOC", 122.05250148235591): 0.430262334889798,
    ("UOC", 124.60500296471182): 0.18812671865890415,
    ("UOC", 134.03340320188877): 0.0013032991155266672,
    ("UOC_r5_hit", 120.01): 4.967068013162726,
    ("UOC_r5_hit", 121.0): 4.975711034647659,
    ("UOC_r5_hit", 122.05250148235591): 4.982332424301231,
    ("UOC_r5_hit", 124.60500296471182): 4.990955121312965,
    ("UOC_r5_hit", 134.03340320188877): 4.994990730241913,
    ("UIC_r5", 120.01): 22.3231940278229,
    ("UIC_r5", 121.0): 23.204126277182045,
    ("UIC_r5", 122.05250148235591): 24.151594623696017,
    ("UIC_r5", 124.60500296471182): 26.488053032555218,
    ("UIC_r5", 134.03340320188877): 35.425950600501736,
    ("DIP_r5", 84.99): 15.305093735566208,
    ("DIP_r5", 84.0): 16.022611172007576,
    ("DIP_r5", 83.5942337389101): 16.32617216885221,
    ("DIP_r5", 81.68846747782023): 17.824401987352104,
    ("DIP_r5", 75.61339007959461): 23.179195138225055,
    ("OT_up", 120.01): 4.171857129835121,
    ("OT_up", 121.0): 4.325607738235779,
    ("OT_up", 122.05250148235591): 4.467145819558404,
    ("OT_up", 124.60500296471182): 4.710095653535253,
    ("OT_up", 134.03340320188877): 4.8948234675885285,
    ("OT_up_hit", 120.01): 4.247602603636262,
    ("OT_up_hit", 121.0): 4.406089984566929,
    ("OT_up_hit", 122.05250148235591): 4.552070089411435,
    ("OT_up_hit", 124.60500296471182): 4.802828402654055,
    ("OT_up_hit", 134.03340320188877): 4.993687431126385,
    ("NT_up", 120.01): 0.7242376930121763,
    ("NT_up", 121.0): 0.5704870846115209,
    ("NT_up", 122.05250148235591): 0.4289490032888923,
    ("NT_up", 124.60500296471182): 0.18599916931204197,
    ("NT_up", 134.03340320188877): 0.0012713552587693406,
    ("NT_dn", 84.99): 0.723560281968755,
    ("NT_dn", 84.0): 0.5099894095178865,
    ("NT_dn", 83.5942337389101): 0.43320948436155576,
    ("NT_dn", 81.68846747782023): 0.16882300616604207,
    ("NT_dn", 75.61339007959461): 0.0006936412725027053,
    ("DKO_C", 130.01): 1.59750970871203,
    ("DKO_C", 131.0): 1.2880194849039786,
    ("DKO_C", 132.22354327255223): 0.9536057587381589,
    ("DKO_C", 134.94708654510447): 0.4222552472276046,
    ("DKO_C", 145.20285346871285): 0.0028776297692666803,
    ("DKO_C_r5", 130.01): 5.771048722073472,
    ("DKO_C_r5", 131.0): 5.603763010485712,
    ("DKO_C_r5", 132.22354327255223): 5.4217406695758745,
    ("DKO_C_r5", 134.94708654510447): 5.130103583341061,
    ("DKO_C_r5", 145.20285346871285): 4.8977035278735,
    ("DKI_C", 130.01): 29.969060261712414,
    ("DKI_C", 131.0): 31.222958609957622,
    ("DKI_C", 132.22354327255223): 32.72954766989926,
    ("DKI_C", 134.94708654510447): 35.887167004360265,
    ("DKI_C", 145.20285346871285): 46.33518282831302,
    ("DKI_C_r5", 130.01): 30.691616071198276,
    ("DKI_C_r5", 131.0): 31.803309907223188,
    ("DKI_C_r5", 132.22354327255223): 33.15750758190885,
    ("DKI_C_r5", 134.94708654510447): 36.07541349109411,
    ("DKI_C_r5", 145.20285346871285): 46.33645175305609,
    ("SSF", 120.01): 1.1366511225099718,
    ("SSF", 121.0): 1.002181823904306,
    ("SSF", 122.05250148235591): 0.8769769168456386,
    ("SSF", 124.60500296471182): 0.6591362840124297,
    ("SSF", 134.03340320188877): 0.49078564587437923,
    ("SSF_nh_hit", 120.01): 1.578768285697392,
    ("SSF_nh_hit", 121.0): 1.3525222993043338,
    ("SSF_nh_hit", 122.05250148235591): 1.1428387458042772,
    ("SSF_nh_hit", 124.60500296471182): 0.7800090605115347,
    ("SSF_nh_hit", 134.03340320188877): 0.5014348553834268,
    ("DSF", 130.01): 2.014863610048174,
    ("DSF", 131.0): 1.719593837462152,
    ("DSF", 132.22354327255223): 1.4004192498219303,
    ("DSF", 134.94708654510447): 0.8930400808389505,
    ("DSF", 145.20285346871285): 0.49236021957969023,
    ("DSF_nh_hit", 130.01): 2.4557405959232557,
    ("DSF_nh_hit", 131.0): 2.0756207036907224,
    ("DSF_nh_hit", 132.22354327255223): 1.665531919012263,
    ("DSF_nh_hit", 134.94708654510447): 1.0151749267029204,
    ("DSF_nh_hit", 145.20285346871285): 0.5030072897222663,
    ("DSF_put_lo", 74.99): 2.4577107533514537,
    ("DSF_put_lo", 74.0): 1.8130509439298201,
    ("DSF_put_lo", 73.7596180049207): 1.6753361184998425,
    ("DSF_put_lo", 72.01923600984138): 0.9367219257611086,
    ("DSF_put_lo", 66.71769712905407): 0.4915343281590867,
    ("ACC_term", 110.01): 8.58219774285876,
    ("ACC_term", 111.0): 7.199198971890815,
    ("ACC_term", 111.88145969215958): 5.909693393821821,
    ("ACC_term", 114.26291938431916): 2.8768423245022183,
    ("ACC_term", 122.8639529350647): 0.02514204388106631,
}

#: (kind, spot) -> main 1e32c27c, aged 0.30, spot inside the level: the BGK closed form at today's spot (moves)
MAIN_INSIDE = {
    ("UOC", 119.5): 0.8708291941804163, ("UOC", 119.99): 0.7800808279621387,
    ("OT_up_hit", 119.5): 4.092143465335606, ("NT_dn", 85.5): 0.9137017416871772,
    ("DIP_r5", 85.5): 14.966298539500947, ("DKO_C_r5", 129.5): 5.947177139623101,
    ("DKI_C", 129.5): 29.164679543355827, ("SSF_nh_hit", 119.5): 1.8052186980989255,
    ("DSF_nh_hit", 129.5): 2.846128862775142, ("ACC_term", 109.5): 9.787817267613379,
}

#: (kind, spot) -> main 1e32c27c, aged onto a record day, spot beyond the record: decided at once
RECORD_DAY_PINS = {
    ("UOC", 121.0): 0.0, ("UOC_r5_hit", 121.0): 5.0, ("UIC_r5", 121.0): 23.399089989613955,
    ("DIP_r5", 84.0): 15.913008758992675, ("OT_up_hit", 121.0): 5.0, ("NT_dn", 84.0): 0.0,
    ("DKO_C_r5", 131.0): 4.888756185966682, ("DKI_C", 131.0): 32.65220451947923,
    ("SSF_nh_hit", 121.0): 0.5, ("DSF_nh_hit", 131.0): 0.5, ("ACC_term", 111.0): 0.0,
}

#: (engine, builder, spot) -> main 1e32c27c: CONTINUOUS and EXPIRY monitoring, fresh contracts
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


def _aged(name, years, keep_landing):
    engine, build, _, _ = KINDS[name]
    product = build()
    product.time_shift(years, T.VALUATION, T.env(100.0), keep_landing=keep_landing)
    return engine, product


def _between(name):
    return _aged(name, T.BETWEEN, False)


def _price(engine_name, product, spot):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(ENGINES[engine_name]().price(product, T.env(spot)))


def _quad(product, spot):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(DiscreteQuadEngine(params=QuadParams(grid_points=401)).price(product, T.env(spot)))


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
# the engines' identity, and what the projection needs
# ======================================================================================================================

def test_the_model_fingerprints_are_those_of_main():
    """The method carries no engine state: ``engines_equivalent`` and the pricing caches key on these attributes,
    so an engine compares equal to main's."""
    assert BarrierAnalyticalEngine.MODEL_FINGERPRINT_ATTRS == ('params', '_bs_engine', '_one_touch_engine')
    assert OneTouchAnalyticalEngine.MODEL_FINGERPRINT_ATTRS == ('params', '_digital_engine')
    assert DoubleBarrierOptionAnalyticalEngine.MODEL_FINGERPRINT_ATTRS == ('params', '_bs_engine')
    assert SingleSharkfinOptionAnalyticalEngine.MODEL_FINGERPRINT_ATTRS == (
        'params', '_barrier_engine', '_one_touch_engine')
    assert DoubleSharkfinOptionAnalyticalEngine.MODEL_FINGERPRINT_ATTRS == ('params', '_double_barrier_engine')
    assert AccumulatorAnalyticalEngine.MODEL_FINGERPRINT_ATTRS == (
        'params', '_barrier_engine', '_bs_engine', '_one_touch_engine')


@pytest.mark.parametrize("spot", [100.0, 121.0])
def test_a_schedule_that_is_not_first_hit_is_refused_inside_the_level_too(spot):
    """The projection lets the first future observation decide: an ACCUMULATE schedule has no such meaning. It is
    refused on both sides of the level (main priced the aged up-and-out call inside it, spot 100, at 2.4865 with
    the BGK closed form)."""
    schedule = ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=120.0) for t in T.TIMES],
                                   aggregation_mode=ObservationAggregation.ACCUMULATE)
    product = BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT,
                            rebate=0.0, observation_schedule=schedule, maturity=1.0,
                            observation_type=ObservationType.DISCRETE)
    product.time_shift(T.BETWEEN, T.VALUATION, T.env(100.0), keep_landing=False)
    with pytest.raises(PricingError, match="STOP_FIRST_HIT") as refused:
        BarrierAnalyticalEngine().price(product, T.env(spot))
    assert "beyond" not in str(refused.value)        # whichever side of the level the spot is on
    assert "accumulate" in str(refused.value)


# ======================================================================================================================
# inside the level: through the first observation
# ======================================================================================================================

@pytest.mark.parametrize("name,spot", list(MAIN_INSIDE), ids=[f"{n}-{s}" for n, s in MAIN_INSIDE])
def test_inside_the_level_the_engine_projects(name, spot, monkeypatch):
    """No record today, spot inside the level: priced through the first observation (the spy sees it), so the
    value leaves main's BGK closed form."""
    engine, product = _between(name)
    spy = _Spy(monkeypatch)
    value = _price(engine, product, spot)
    assert spy.calls >= 1
    assert math.isfinite(value)
    assert value != MAIN_INSIDE[(name, spot)]


@pytest.mark.parametrize("name,spot", [("UOC", 119.5), ("UOC", 119.99), ("OT_up_hit", 119.5), ("NT_dn", 85.5),
                                       ("DKO_C_r5", 129.5), ("DKI_C", 129.5)])
def test_inside_the_level_the_projection_is_closer_to_quad(name, spot):
    """QUAD (401 points) is exact discrete monitoring up to its quadrature error."""
    engine, product = _between(name)
    reference = _quad(_between(name)[1], spot)
    value = _price(engine, product, spot)
    assert abs(value - reference) < abs(MAIN_INSIDE[(name, spot)] - reference)
    assert abs(value - reference) <= 0.03


@pytest.mark.parametrize("name", ["SSF_nh_hit", "ACC_term"])
def test_composite_legs_project_inside_the_level(name, monkeypatch):
    engine, product = _between(name)
    _, _, level, side = KINDS[name]
    spy = _Spy(monkeypatch)
    _price(engine, product, level - side * 0.5)
    assert spy.calls >= 2


# ======================================================================================================================
# beyond the level: main's projection
# ======================================================================================================================

@pytest.mark.parametrize("name,spot", list(BEYOND_PINS), ids=[f"{n}-{s:.4f}" for n, s in BEYOND_PINS])
def test_beyond_the_level_the_value_is_mains(name, spot):
    """Main's values (bitwise on the ARM64 machine that took them; NumPy's SIMD exp / log / pow on x86 moves the
    last bits of the vectorised projection, so the frozen literals compare at the golden tolerance)."""
    engine, product = _between(name)
    assert_close(_price(engine, product, spot), BEYOND_PINS[(name, spot)], msg=f"{name} {spot}")


# ======================================================================================================================
# continuity across the level
# ======================================================================================================================

def _step(name, price):
    _, _, level, side = KINDS[name]
    return abs(price(_between(name), level + side * STEP) - price(_between(name), level - side * STEP))


@pytest.mark.parametrize("name", list(KINDS))
def test_the_price_is_continuous_across_the_level(name):
    """Spot level -/+ 0.01, no record today: the price's slope x 0.02, never main's switch (up to 0.161)."""
    assert _step(name, lambda case, spot: _price(case[0], case[1], spot)) <= CONTINUITY_ABS


@pytest.mark.parametrize("name", ["UOC", "UIC_r5", "DIP_r5", "OT_up_hit", "NT_dn", "DKO_C_r5", "DKI_C"])
def test_the_step_across_the_level_is_quads(name):
    ours = _step(name, lambda case, spot: _price(case[0], case[1], spot))
    quads = _step(name, lambda case, spot: _quad(case[1], spot))
    assert abs(ours - quads) <= STEP_VS_QUAD_ABS, (ours, quads)


# ======================================================================================================================
# unchanged paths
# ======================================================================================================================

@pytest.mark.parametrize("name,spot", list(RECORD_DAY_PINS), ids=[f"{n}-{s}" for n, s in RECORD_DAY_PINS])
def test_a_record_on_the_valuation_instant_decides_as_on_main(name, spot, monkeypatch):
    engine, product = _aged(name, T.RECORD_DAY, True)
    spy = _Spy(monkeypatch)
    assert_close(_price(engine, product, spot), RECORD_DAY_PINS[(name, spot)])
    assert spy.calls == 0


@pytest.mark.parametrize("case", range(len(UNCHANGED_PINS)))
def test_continuous_and_expiry_monitoring_are_unchanged(case, monkeypatch):
    engine, build, spot, before = UNCHANGED_PINS[case]
    spy = _Spy(monkeypatch)
    assert_close(_price(engine, build(), spot), before)
    assert spy.calls == 0


@pytest.mark.parametrize("engine_name,build,spot,before", [
    ("barrier", barrier(BarrierType.UP_OUT, OptionType.CALL, 120.0, 5.0), 125.0, 5.0),
    ("touch", touch(TouchType.ONE_TOUCH, True, 120.0), 125.0, 5.0),
    ("double", double(DoubleBarrierType.KNOCK_OUT, OptionType.CALL, 5.0), 125.0, 25.0),
    ("single_sharkfin", ssf(0.5, 3.0), 125.0, 0.5),
    ("double_sharkfin", dsf(0.5, 3.0), 125.0, 28.0),
])
def test_maturity_settlement_is_unchanged(engine_name, build, spot, before):
    product = build()
    product.time_shift(1.0, T.VALUATION, T.env(100.0), keep_landing=True)
    assert _price(engine_name, product, spot) == pytest.approx(before, rel=1e-12, abs=1e-12)
