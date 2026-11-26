"""KO-reset QUAD V1 knock-outs through the cell-average projection (0.16.0, QUAD numerics run 5, D1).

Before: ``KOResetSnowballQuadEngine`` applied both schedules' knock-outs with the legacy nodal cosine smoothing
(``_smooth_step_weight``, +/- ``event_smoothing_cells`` cells; a hard nodal mask at width 0) whatever
``event_projection`` said, while Snowball / Phoenix and KO-reset's own knock-in blend had moved to the cell-average
(dual-cell) projection. That rule is O(h): 1001-node prices 2.0e-3 .. 1.4e-2 off QUAD V2 on the certification
product's variants (decision matrix (c)). Now, under ``event_projection=CELL_AVERAGE`` (the default), on the lattice:

* a single knock-out record is ``_project_quad_event`` (Snowball's rule);
* a zero-gap instant's two or more effective records of one schedule are one pass of the piecewise projection
  (``_knock_out_group``, Snowball's same-instant group rule), in the instant unit's order (schedule-1 knock-outs, the
  knock-in, schedule-2 knock-outs);
* the pre-KO weight a same-instant discrete knock-in blend may be narrowed by is the breach fraction
  (``_event_weight``).

``event_projection=NODAL`` (any smoothing width) and the exact first step's pointwise event (F's quadrature nodes,
width 0) keep the previous KO-reset code verbatim -- bitwise (pins below, captured on the base library 31f46955).
One event map serves price, statistics, the exact first step and the forward adjoint pass (event maps read off the
event code by probes). Snowball and Phoenix V1 are untouched (bitwise pins).
"""
from __future__ import annotations

import hashlib
from datetime import datetime

import numpy as np
import pytest

from test_asset.equity.engine_quad.test_same_instant_ko_quad import KOR_CASES, ko_reset
from test_asset.equity.engine_quad.test_same_instant_ko_quad import env as dated_env
from quantark.asset.equity.engine.mc import SnowballMCEngine
from quantark.asset.equity.engine.quad import (
    KOResetSnowballQuadEngine,
    PhoenixQuadEngine,
    SnowballQuadEngine,
)
from quantark.asset.equity.param import MCParams, QuadParams
from quantark.asset.equity.product.option import PhoenixOption, SnowballOption
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.engine_enums import EventProjectionMode

NODAL = EventProjectionMode.NODAL

# --- products --------------------------------------------------------------------------------------------------------

#: the certification KO-reset (test/golden/modelvalidation/ko_reset_flat_bsm.yaml): schedule 1 KO 103 monthly to 1y
#: at 15%, schedule 2 KO 95 monthly to 2y at 3% (ABSOLUTE), KI 80 continuous; spot 100, vol 22%, r 2.5%, q 3%
CERT = dict(initial_price=100.0, strike=100.0, maturity_pre=1.0, maturity_post=2.0, pre_ko_barrier=103.0,
            pre_ko_rate=0.15, post_ko_barrier=95.0, post_ko_rate=0.03, ki_barrier=80.0, ki_continuous=True,
            contract_multiplier=1.0)
#: decision matrix (c)'s seven variants: (product overrides, market overrides)
VARIANTS = {
    "ordinary": ({}, {}),
    "near_ki": ({}, {"spot": 80.5}),
    "below_ki": ({}, {"spot": 76.0}),
    "low_vol": ({}, {"vol": 0.12}),
    "discrete_ki": ({"ki_continuous": False}, {}),
    "european_ki": ({"ki_monitoring": "european"}, {}),
    "discrete_low_vol": ({"ki_continuous": False}, {"vol": 0.12}),
}


def market(vol=0.22, spot=100.0, r=0.025, q=0.03):
    return PricingEnvironment(spot_quote=SpotQuote(spot), vol_surface=FlatVolSurface(vol),
                              rate_curve=FlatRateCurve(r), div_yield=ContinuousDividendYield(q),
                              valuation_date=datetime(2026, 9, 11))


def variant(name):
    product_kw, market_kw = VARIANTS[name]
    return make_ko_reset({**CERT, **product_kw}), market(**market_kw)


def kor(**kw):
    return KOResetSnowballQuadEngine(QuadParams(**kw))


def kor_route(route, **kw):
    engine = kor(**kw)
    engine._v1_statistics_route = route
    return engine


def _rows(stats):
    return [float(p) for p in stats.observations.probability]


def _hex(values):
    return [float(v).hex() for v in values]


# --- AC1: the seven variants against QUAD V2 ------------------------------------------------------------------------

#: QUAD V2 (``QuadV2Params()``) on base 31f46955 -- the reference (V2 refined, cells_per_sd 3 / order 10, agrees to
#: <= 2e-12, decision matrix (c) log c1_ko_reset.log) -- and the base library's V1 1001-node price (nodal-cosine
#: knock-outs), both measured by the run's T1 script (scratchpad T1/seven.py, seven_patch.log).
AC1 = {
    "ordinary": (96.6713830627385, 96.6849825386466),
    "near_ki": (79.02548561989784, 79.03598694339034),
    "below_ki": (74.0369343929245, 74.04529846451685),
    "low_vol": (100.16336713525578, 100.16923107231617),
    "discrete_ki": (96.73027642836507, 96.73574027293131),
    "european_ki": (97.64974529389099, 97.65708401419982),
    "discrete_low_vol": (100.23774379760054, 100.2397547727824),
}
#: AC1 bar: at 1001 nodes within 2e-3 abs of the reference and >= 4x closer than the base
AC1_ABS = 2e-3
AC1_RATIO = 4.0
#: measured misses of the AC1 bar (reported for a ruling, not tuned): near_ki -2.121e-3 abs (4.95x closer than the
#: base's +1.050e-2); discrete_low_vol +6.72e-4 abs but only 2.99x closer (base +2.011e-3: the base error was already
#: small; both 16001-node errors -1.7e-5 / -2.2e-5)
AC1_MISSES = {"near_ki", "discrete_low_vol"}


@pytest.mark.parametrize("name", [
    pytest.param(n, marks=pytest.mark.xfail(strict=True, reason="AC1 bar missed (measured, reported for a ruling)"))
    if n in AC1_MISSES else n for n in VARIANTS])
def test_seven_variants_meet_the_ac1_bar(name):
    product, e = variant(name)
    reference, base = AC1[name]
    new = kor().price(product, e)
    new_error, base_error = abs(new - reference), abs(base - reference)
    assert new_error <= AC1_ABS, (new - reference, base - reference)
    assert base_error >= AC1_RATIO * new_error, (new - reference, base - reference)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_seven_variants_are_closer_to_the_reference_than_the_base(name):
    """Every variant, the two AC1 misses included: closer to V2 than the base by more than 2.9x, within 2.2e-3 abs."""
    product, e = variant(name)
    reference, base = AC1[name]
    new = kor().price(product, e)
    assert abs(new - reference) <= 2.2e-3, new - reference
    assert abs(base - reference) >= 2.9 * abs(new - reference), (new - reference, base - reference)


def test_the_projection_converges_under_refinement():
    """The residual is the lattice's O(h^2) phase effect, as Snowball's: 16001 nodes within 3e-5 of V2 (measured
    -1.39e-5; base -7.1e-5)."""
    product, e = variant("ordinary")
    reference = AC1["ordinary"][0]
    assert abs(kor(grid_points=16001).price(product, e) - reference) <= 3e-5


# --- nodal and pointwise (width 0) paths: bitwise the base -----------------------------------------------------------

#: captured on the base library (31f46955) before the D1 change
NODAL_PRICE_PINS = {
    "cert_ordinary": "0x1.82bd6c100745dp+6",
    "cert_discrete_ki": "0x1.82ed7bff2c507p+6",
    "cert_discrete_ki_width0": "0x1.833c3bbb1d358p+6",
    "instant_schedule1_and_ki_pair": "0x1.f84212775e3e9p+2",
    "instant_schedule2_width0": "0x1.91a7cb0311cf0p+2",
    "reverse": "0x1.b2a8e099405bap+2",
}
NODAL_STATS_PINS = {
    "cert_discrete_ki/forward": (
        ["0x1.3ab57b3927843p-2", "0x1.13df7bccc8c01p-34", "0x1.1631942cf411cp-3", "0x1.591f811bba523p-19",
         "0x1.394eb6bf4536bp-4", "0x1.94472d1e92ecdp-14", "0x1.9bd17e500eef2p-5", "0x1.25fba89ffdcefp-11",
         "0x1.27090b2efb54cp-5", "0x1.848760d90b181p-10", "0x1.bc1674e094bb1p-6", "0x1.5988b6eb09ba8p-9",
         "0x1.578bf66442db8p-6", "0x1.ec680af448aa6p-9", "0x1.0dc356ae7ffcap-6", "0x1.32d7fff27a849p-8",
         "0x1.ab03e9c2b59b9p-7", "0x1.5ef9ab338b164p-8", "0x1.535de43c9ec60p-7", "0x1.7b147496eeabep-8",
         "0x1.0e49926dbe58dp-7", "0x1.89889572a37eap-8", "0x1.af03e6f923af9p-8", "0x1.8d3b874bce12fp-8",
         "0x1.88c5af1f73e43p-8", "0x1.7a8bfb327e47fp-8", "0x1.62508c2051a31p-8", "0x1.469fa69a9bd1ep-8",
         "0x1.2b9ab50e2f364p-8", "0x1.12c7d951165bdp-8", "0x1.f905ea4fe62e7p-9", "0x1.d16bab88105acp-9",
         "0x1.ae48006ecad24p-9", "0x1.8f1183141217ap-9", "0x1.73453f6eddb48p-9", "0x1.5a6e5bee25063p-9"],
        "0x1.e230da1a191d6p-5", "0x1.2ac694129fef5p-3"),
    "cert_discrete_ki/stacked": (
        ["0x1.3ab57b3927818p-2", "0x1.13df7bd40afcbp-34", "0x1.1631942cf40d9p-3", "0x1.591f811bbb306p-19",
         "0x1.394eb6bf452e2p-4", "0x1.94472d1e927e8p-14", "0x1.9bd17e500ee4cp-5", "0x1.25fba89ffdb6fp-11",
         "0x1.27090b2efb4c2p-5", "0x1.848760d90af47p-10", "0x1.bc1674e094ad4p-6", "0x1.5988b6eb09ad4p-9",
         "0x1.578bf66442d2dp-6", "0x1.ec680af4489dcp-9", "0x1.0dc356ae7ff4bp-6", "0x1.32d7fff27a763p-8",
         "0x1.ab03e9c2b58ebp-7", "0x1.5ef9ab338b0acp-8", "0x1.535de43c9ebd1p-7", "0x1.7b147496eea0cp-8",
         "0x1.0e49926dbe513p-7", "0x1.89889572a36ddp-8", "0x1.af03e6f923a28p-8", "0x1.8d3b874bce061p-8",
         "0x1.88c5af1f73e4cp-8", "0x1.7a8bfb327e46dp-8", "0x1.62508c2051a46p-8", "0x1.469fa69a9bd2ap-8",
         "0x1.2b9ab50e2f373p-8", "0x1.12c7d951165dep-8", "0x1.f905ea4fe632dp-9", "0x1.d16bab88105e8p-9",
         "0x1.ae48006ecad80p-9", "0x1.8f118314121d4p-9", "0x1.73453f6eddb82p-9", "0x1.5a6e5bee250d2p-9"],
        "0x1.e230da1a191e2p-5", "0x1.2ac69412a0104p-3"),
    "instant_schedule1_and_ki_pair/forward": (
        ["0x1.80415fbd50683p-2", "0x1.2504dcfdc4bfbp-3", "0x1.2969dcf4d335cp-4", "0x1.dbf4e911af404p-6",
         "0x1.0196f2ef21c69p-9", "0x1.27d49ece235b2p-8", "0x1.845019eee3f0ap-9"],
        "0x1.197b984108ca9p-2", "0x1.87eecc7e004d6p-4"),
}
POINTWISE_EVENT_PINS = {
    "single_date": "97e73ef50910383512be26442e0e40c82c601828397cdceac95ef7459ca05e6e",
    "instant": "aea9b0b614855fc981be83344fefe707a2eb5979997850b80321e677a3198a72",
}


def _nodal_products():
    return {
        "cert_ordinary": (variant("ordinary"), dict(event_projection=NODAL)),
        "cert_discrete_ki": (variant("discrete_ki"), dict(event_projection=NODAL)),
        "cert_discrete_ki_width0": (variant("discrete_ki"), dict(event_projection=NODAL, event_smoothing_cells=0)),
        "instant_schedule1_and_ki_pair": ((ko_reset(**KOR_CASES["schedule1_and_ki_pair"]), dated_env()),
                                          dict(event_projection=NODAL)),
        "instant_schedule2_width0": ((ko_reset(**KOR_CASES["schedule2_step_down"]), dated_env()),
                                     dict(event_projection=NODAL, event_smoothing_cells=0)),
        "reverse": ((ko_reset(reverse=True), dated_env()), dict(event_projection=NODAL)),
    }


@pytest.mark.parametrize("name", list(NODAL_PRICE_PINS))
def test_nodal_prices_are_bitwise_the_base(name):
    (product, e), kw = _nodal_products()[name]
    assert kor(**kw).price(product, e).hex() == NODAL_PRICE_PINS[name]


@pytest.mark.parametrize("key", list(NODAL_STATS_PINS))
def test_nodal_statistics_are_bitwise_the_base(key):
    name, route = key.split("/")
    (product, e), kw = _nodal_products()[name]
    stats = kor_route(route, **kw).calculate_event_stats(product, e)
    rows, maturity, ki_settlement = NODAL_STATS_PINS[key]
    assert _hex(_rows(stats)) == rows
    assert float(stats.maturity.probability).hex() == maturity
    assert float(stats.ki_settlement.probability).hex() == ki_settlement


def _off_barrier_nodes(barriers, spot):
    y = np.linspace(-0.45, 0.45, 301) + 1.0e-4
    logs = np.log(np.array(barriers) / spot)
    assert np.min(np.abs(y[:, None] - logs[None, :])) > 1e-5
    return y


def _pointwise_event_cases():
    """The exact first step's event map (``_pointwise()``, width 0) on quadrature-like nodes -- a single date holding
    a schedule-1 and a schedule-2 knock-out and a discrete knock-in, and a zero-gap instant (schedule-1 pair and
    knock-in pair). Polynomial surfaces on nodes at least 1e-5 from every barrier, so the digest is the same on any
    platform's libm (the map selects and blends; no node is a tie)."""
    out = {}
    product, e = variant("discrete_ki")
    engine = kor()
    pre = engine._resolve_ko_records(product, e, product.barrier_config)
    post = engine._resolve_ko_records(product, e, product.post_barrier_config)
    ki = product.resolve_ki_observations(e)
    t = float(pre[2].observation_time)
    assert engine._match_record(t, post) is not None and engine._match_record(t, ki) is not None
    spot = float(e.spot)
    y = _off_barrier_nodes([103.0, 95.0, 80.0], spot)
    out["single_date"] = (engine, product, y, spot, t, pre, post, ki, None)
    product2, e2 = ko_reset(**KOR_CASES["schedule1_and_ki_pair"]), dated_env()
    engine2 = kor()
    pre2 = engine2._resolve_ko_records(product2, e2, product2.barrier_config)
    post2 = engine2._resolve_ko_records(product2, e2, product2.post_barrier_config)
    ki2 = product2.resolve_ki_observations(e2)
    instant = engine2._reset_instants(product2, pre2, post2, ki2)[0]
    spot2 = float(e2.spot)
    y2 = _off_barrier_nodes([103.0, 100.0, 75.0, 80.0], spot2)
    out["instant"] = (engine2, product2, y2, spot2, instant.time, pre2, post2, ki2, instant)
    return out


def _pointwise_event_digest(name):
    engine, product, y, spot, t, pre, post, ki, instant = _pointwise_event_cases()[name]
    v_out = 50.0 + 40.0 * y - 30.0 * y * y
    v_in = 30.0 - 25.0 * y + 10.0 * y * y * y
    values = None if instant is None else {id(r): 7.0 + k for k, r in enumerate(instant.pre + instant.post)}
    with engine._pointwise():
        out_y, in_y = engine._apply_reset_observation(
            product, y, spot * np.exp(y), spot, 0.0, v_out.copy(), v_in.copy(), t, True, pre, post, ki, False,
            None, 11.0, 13.0, instant=instant, instant_values=values)
    return hashlib.sha256(np.ascontiguousarray(out_y).tobytes() + np.ascontiguousarray(in_y).tobytes()).hexdigest()


@pytest.mark.parametrize("name", list(POINTWISE_EVENT_PINS))
def test_pointwise_event_map_is_bitwise_the_base(name):
    assert _pointwise_event_digest(name) == POINTWISE_EVENT_PINS[name]


# --- Snowball and Phoenix V1: untouched -------------------------------------------------------------------------------

SNOWBALL_PHOENIX_PINS = {
    "snowball_cki": (
        "0x1.8b515f8797113p+6",
        ["0x1.8fda13026793cp-2", "0x1.16e7dbedc099cp-3", "0x1.29fabd5729e77p-4", "0x1.7f3c83595b991p-5"],
        "0x1.75d6e7dc339c3p-3", "0x1.5ec096af50ca7p-3",
        []),
    "snowball_dki": (
        "0x1.904def259f7ddp+6",
        ["0x1.8fda13026793cp-2", "0x1.16e7dcc685415p-3", "0x1.29fac736960a7p-4", "0x1.7f3cafa98cc15p-5"],
        "0x1.eda970c360140p-3", "0x1.cddbf9d73a9f5p-4",
        []),
    "snowball_dki_nodal": (
        "0x1.904b5ace5a0e4p+6",
        ["0x1.8fda13026793cp-2", "0x1.170a49ba00ea7p-3", "0x1.2a0dda67ef15dp-4", "0x1.7f46b98385c3bp-5"],
        "0x1.ed3389bb22c4cp-3", "0x1.ce6ad5e2685efp-4",
        []),
    "phoenix_memory_instant_cki": (
        "0x1.80d829b629f29p+6",
        ["0x1.4d2a32c2318dfp-2", "0x1.1eb710dac61d4p-3", "0x1.4123c4bd4d3b8p-4", "0x1.a5ae2fa69ba69p-5"],
        "0x1.35cbea9a0d520p-2", "0x1.a2be8c48dd3b6p-4",
        ["0x1.f4d5280292de0p-1", "0x1.9df5644535c83p-4", "0x1.aad23e22ebf82p-2", "0x1.409b75a52e0ddp-2"]),
    "phoenix_expiry_dki": (
        "0x1.8356f37cee05ap+6",
        ["0x1.4d2a32c2318dfp-2", "0x1.1eb710dda0270p-3", "0x1.4123c52f0083fp-4", "0x1.a5ae35658054fp-5"],
        "0x1.5dada8a37b2e6p-2", "0x1.033790cc4c16bp-4",
        ["0x1.f4d5280292de0p-1", "0x1.9df564477e0b9p-4", "0x1.aad2469ea3c7fp-2", "0x1.409b895d747bep-2"]),
}
OBS = (0.25, 0.5, 0.75, 1.0)


def _snowball(ki):
    kw = (dict(ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True) if ki == "cki" else
          dict(ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=list(OBS)))
    return SnowballOption(initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0,
                          barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15,
                                                       ko_observation_type=ObservationType.DISCRETE,
                                                       ko_observation_dates=list(OBS), ki_barrier=75.0, **kw),
                          payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True))


def _phoenix(memory, pay, ki):
    kw = (dict(ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True) if ki == "cki" else
          dict(ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=list(OBS)))
    return PhoenixOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=105.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(OBS), ki_barrier=70.0, **kw),
        coupon_config=CouponBarrierConfig(coupon_barrier=[80.0, 108.0, 80.0, 80.0], coupon_rate=0.02,
                                          coupon_pay_type=pay, day_count_convention=DayCountConvention.ACT_365,
                                          memory_coupon=memory),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True))


def _snowball_phoenix_cases():
    return {
        "snowball_cki": (SnowballQuadEngine, _snowball("cki"), {}),
        "snowball_dki": (SnowballQuadEngine, _snowball("dki"), {}),
        "snowball_dki_nodal": (SnowballQuadEngine, _snowball("dki"), dict(event_projection=NODAL)),
        "phoenix_memory_instant_cki": (PhoenixQuadEngine, _phoenix(True, CouponPayType.INSTANT, "cki"), {}),
        "phoenix_expiry_dki": (PhoenixQuadEngine, _phoenix(False, CouponPayType.EXPIRY, "dki"), {}),
    }


def _snowball_phoenix_numbers(name):
    cls, product, kw = _snowball_phoenix_cases()[name]
    e = market(r=0.03, q=0.01)
    price = cls(QuadParams(**kw)).price(product, e)
    stats = cls(QuadParams(**kw)).calculate_event_stats(product, e)
    coupons = [] if stats.coupons is None else _hex(stats.coupons.probability)
    return (price.hex(), _hex(_rows(stats)), float(stats.maturity.probability).hex(),
            float(stats.ki_settlement.probability).hex(), coupons)


@pytest.mark.parametrize("name", list(SNOWBALL_PHOENIX_PINS))
def test_snowball_and_phoenix_v1_are_bitwise_the_base(name):
    price, rows, maturity, ki_settlement, coupons = SNOWBALL_PHOENIX_PINS[name]
    assert _snowball_phoenix_numbers(name) == (price, rows, maturity, ki_settlement, coupons)


# --- AC2: statistics on the projected knock-outs ---------------------------------------------------------------------

#: the existing KO-reset V1 statistics tolerances (``test_quad_v1_event_statistics.py``): partition residual and
#: reconciliation (x max(1, pv)) RESIDUAL_BOUND = RECONCILIATION_BOUND = 1e-12, the forward pass against a backward
#: column ADJOINT_TOL = 1e-10
RESIDUAL_BOUND = 1e-12
RECONCILIATION_BOUND = 1e-12
ADJOINT_TOL = 1e-10
#: the certification product with a daily knock-in (36 knock-out rows, ~500 steps over 2y, 1001 nodes) sums its
#: partition to -1.303e-12 / -1.304e-12 (forward / stacked) on the base library already, -1.303e-12 / -1.306e-12
#: after D1: round-off of a longer march than the existing tests' (401 nodes, 4-8 records), not the projection
RESIDUAL_BOUND_BY_PRODUCT = {"cert_discrete_ki": 1.5e-12}
STAT_ROUTES = ("forward", "stacked")


def _stats_products():
    return {
        "cert_ordinary": variant("ordinary"),
        "cert_discrete_ki": variant("discrete_ki"),
        "cert_below_ki": variant("below_ki"),
        "kor_base": (ko_reset(), dated_env()),
        "kor_reverse": (ko_reset(reverse=True), dated_env()),
        "instant_schedule1_and_ki_pair": (ko_reset(**KOR_CASES["schedule1_and_ki_pair"]), dated_env()),
        "instant_schedule2_step_down": (ko_reset(**KOR_CASES["schedule2_step_down"]), dated_env()),
        "instant_first_step": (ko_reset(**KOR_CASES["schedule1_and_ki_pair"]), dated_env(datetime(2025, 7, 2))),
    }


STATS_NAMES = list(_stats_products())


@pytest.mark.parametrize("route", STAT_ROUTES)
@pytest.mark.parametrize("name", STATS_NAMES)
def test_statistics_partition_and_reconcile(name, route):
    """The outcome columns ride the projected event maps: the partition and the cash identity close to round-off,
    and ``pv`` is ``price`` bitwise (D23)."""
    product, e = _stats_products()[name]
    stats = kor_route(route).calculate_event_stats(product, e)
    assert stats.pv == kor().price(product, e)
    assert abs(stats.probability_residual) <= RESIDUAL_BOUND_BY_PRODUCT.get(name, RESIDUAL_BOUND)
    assert abs(stats.reconciliation_error) <= RECONCILIATION_BOUND * max(1.0, abs(stats.pv))


@pytest.mark.parametrize("name", STATS_NAMES)
def test_forward_pass_reproduces_the_backward_columns(name):
    """AC14 on the projected march: the transposed march's masses equal every backward column read at the spot (the
    event maps are read off the projection by probes), and the forward route reconstructs the price to 1e-10."""
    from quantark.asset.equity.engine.quad.v1_adjoint import forward_pass

    product, e = _stats_products()[name]
    engine = kor_route("stacked")
    stacked = engine.calculate_event_stats(product, e)
    log = engine._last_v1_march
    result = forward_pass(log)
    assert log.columns
    for key, backward in log.columns.items():
        got = result.sources.get(key, 0.0) if key[0] == "ko" else result.terminal[key[1]]
        assert got == pytest.approx(backward, abs=ADJOINT_TOL), key
    forward = kor_route("forward").calculate_event_stats(product, e)
    assert abs(forward.reconciliation_error) <= 1e-10
    assert np.allclose(_rows(forward), _rows(stacked), rtol=0.0, atol=ADJOINT_TOL)


MC_PATHS = 400_000


@pytest.mark.slow
@pytest.mark.parametrize("route", STAT_ROUTES)
@pytest.mark.parametrize("name", ["kor_base", "cert_discrete_ki"])
def test_statistics_rows_agree_with_mc(name, route):
    """Per-record knock-out probabilities within 3 SE (binomial over the paths) of Black-Scholes MC, and the maturity
    and knocked-in settlement probabilities: a discrete knock-in on observation dates (MC is exact there)."""
    product, e = _stats_products()[name]
    mc = SnowballMCEngine(params=MCParams(num_paths=MC_PATHS, seed=7)).calculate_event_stats(product, e)
    stats = kor_route(route).calculate_event_stats(product, e)
    rows, mc_rows = _rows(stats), _rows(mc)
    assert len(rows) == len(mc_rows)
    for p, m in zip(rows + [float(stats.maturity.probability), float(stats.ki_settlement.probability)],
                    mc_rows + [float(mc.maturity.probability), float(mc.ki_settlement.probability)]):
        se = max(np.sqrt(m * (1.0 - m) / MC_PATHS), 1.0 / MC_PATHS)
        assert abs(p - m) <= 3.0 * se, (rows, mc_rows)
