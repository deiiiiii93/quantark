"""A knocked-in KO-reset ends at schedule 2's last fixing on every grid engine (owner ruling D31, 2026-11-14; run 5 T7).

A knocked-in KO-reset is a knocked-in snowball on the second schedule, alive until that schedule ends: its knocked-in
payoff V1 is decided on the spot at schedule 2's last fixing and paid from there (Monte Carlo since D31,
``event_statistics/test_ko_reset_knocked_in_readout.py``). The grid engines (PDE 1D / LocalVol, Heston ADI, QUAD V1,
QUAD V2) decided and paid it at the product maturity (the exercise date), which differs only when the product runs on
past schedule 2's end -- the test helper ``ko_reset()``: schedule 2 ends 2025-12-03, exercise 2026-01-05
(investigation ``.ship/2026-11-27-ko-reset-mc-investigation``). Now every grid engine stages the knocked-in value at
schedule 2's last fixing (as the not-knocked-in value is staged at schedule 1's end); an EXPIRY-paid schedule-2
knock-out is paid with that note; the knocked-in note settles by the rule the never-knocked-in note follows at schedule
1's end, applied to schedule 2's last fixing (an explicit settlement date when booked, else the terminal settlement
delay after it).

- exact referee: a carried knock-in with ``disable_ko_after_ki`` is a European put on schedule 2's end
  (V1 = min(S - K, 0)): Black-Scholes put to 12-03 (PDE rel 1e-3, QUAD V1 rel 1e-4, QUAD V2 rel 1e-10); with a booked
  settlement date / a settlement lag, the put discounted on to that payment;
- Heston ADI: the same contract booked with exercise on schedule 2's end prices the same (the exercise date no longer
  matters once both notes end before it), to the grid's time-step accuracy;
- the investigation's ten cases within 3 SE of Black-Scholes MC (1M paths) on every Black-Scholes grid engine, and a
  booked settlement date / settlement lag case;
- statistics: the knocked-in settlement row sits at schedule 2's end and is paid where MC pays it; EXPIRY pay pays a
  schedule-2 knock-out there too;
- bitwise: KO-reset notes whose exercise date IS schedule 2's last fixing (price and statistics pinned before T7, every
  grid engine).
"""
from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest
from scipy.stats import norm

from test_asset.equity.engine_pde.test_same_instant_ko import HESTON, JUL3, env
from test_asset.equity.engine_quad.test_same_instant_ko_quad import KOR_CASES, KOR_PRE, ko_reset

from quantark.asset.equity.engine.mc import SnowballMCEngine
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.ko_reset_vol_pde_solvers import (
    HestonKOResetSnowballPDESolver,
    LocalVolKOResetSnowballPDESolver,
)
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import KOResetSnowballQuadEngine
from quantark.asset.equity.engine.quad.v2.engine import KOResetSnowballQuadEngineV2
from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams, QuadV2Params
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.util.enum import CouponPayType, ObservationType

D = datetime
#: schedule 2's last fixing in ``ko_reset()``; its exercise date is 2026-01-05
SCHEDULE_2_END = D(2025, 12, 3)
SETTLEMENT_DATE = D(2026, 1, 12)
SPOT, STRIKE, RATE, VOL = 80.0, 100.0, 0.02, 0.25
MC_PATHS = 1_000_000


# --- products (the investigation's builders, ``common.py``) ---------------------------------------------------------

def make(spec=None, disable=False, carried=False, pay=None, exercise=None, settlement_date=None, lag=None,
         ki_type=None):
    kw = dict(spec or {})
    if ki_type is not None:
        kw["ki_type"] = ki_type
    p = ko_reset(**kw)
    if exercise is not None:
        p.exercise_date = exercise
    p.barrier_config = replace(p.barrier_config, disable_ko_after_ki=disable)
    for key, value in (pay or {}).items():
        cfg, attr = key.split(".")
        setattr(p, cfg, replace(getattr(p, cfg), **{attr: value}))
    if settlement_date is not None:
        p.settlement_date = settlement_date
    if lag is not None:
        p.settlement_convention = SettlementConvention(lag=lag, lag_unit=SettlementLagUnit.YEAR_FRACTION)
    p.validate()
    if carried:
        p._otc_lifecycle_knocked_in = True
    return p


def carried_env(grid=False):
    return env(D(2025, 7, 7), spot=SPOT, grid=grid)


KI85 = dict(ki=[(d, 85.0) for d in KOR_PRE])
KI97 = dict(ki=[(d, 75.0) for d in KOR_PRE[:2]] + [(JUL3, 97.0)])
#: the investigation's ten cases (``t6_patches.py``): name -> (product, env)
CASES = {
    "base_ki75": (lambda: make(), env),
    "base_ki75_disable": (lambda: make(disable=True), env),
    "ki97_jul3": (lambda: make(KI97), env),
    "ki97_jul3_disable": (lambda: make(KI97, True), env),
    "ki85": (lambda: make(KI85), env),
    "ki85_disable": (lambda: make(KI85, True), env),
    "carried_jul7": (lambda: make(carried=True), carried_env),
    "carried_jul7_disable": (lambda: make(disable=True, carried=True), carried_env),
    "ki85_include_principal": (lambda: make(KI85, pay={"payoff_config.include_principal": True}), env),
    "ki85_exercise_2026_02_05": (lambda: make(KI85, exercise=D(2026, 2, 5)), env),
    # the settlement rule of the knocked-in note against MC: a booked settlement date, a settlement lag
    "ki85_settlement_date": (lambda: make(KI85, settlement_date=SETTLEMENT_DATE), env),
    "ki85_lag": (lambda: make(KI85, lag=0.02), env),
}

#: Black-Scholes grid engines: name -> (engine, env uses a flat grid surface)
ENGINES = {
    "pde": (lambda: KOResetSnowballPDESolver(PDEParams()), False),
    "lv_pde": (lambda: LocalVolKOResetSnowballPDESolver(PDEParams()), True),
    "v1": (lambda: KOResetSnowballQuadEngine(params=QuadParams(grid_points=1001)), False),
    "v2": (lambda: KOResetSnowballQuadEngineV2(QuadV2Params()), False),
}
PUT_TOLERANCE = {"pde": 1e-3, "lv_pde": 1e-3, "v1": 1e-4, "v2": 1e-10}


def _env_for(engine, make_env):
    return make_env(grid=True) if ENGINES[engine][1] else make_env()


def _bs_put(T):
    d1 = (math.log(SPOT / STRIKE) + (RATE + 0.5 * VOL * VOL) * T) / (VOL * math.sqrt(T))
    d2 = d1 - VOL * math.sqrt(T)
    return STRIKE * math.exp(-RATE * T) * norm.cdf(-d2) - SPOT * norm.cdf(-d1)


# --- the exact referee ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("settlement", ["none", "settlement_date", "lag"])
@pytest.mark.parametrize("engine", list(ENGINES))
def test_carried_knock_in_with_disable_is_the_put_to_schedule_two_end(engine, settlement):
    """Carried knock-in + ``disable_ko_after_ki``: no knock-out left, V1 = min(S - K, 0) decided on schedule 2's last
    fixing (12-03), not the exercise date (01-05: the put there differs by 4.6e-3 relative). The env is Black-Scholes
    (flat 25% vol, 2% rate on its trading clock), so the value is minus the put to 12-03, discounted on to the
    payment when a settlement date / lag pays it later (the never-knocked-in note's rule at schedule 1's end)."""
    kw = {"none": {}, "settlement_date": dict(settlement_date=SETTLEMENT_DATE), "lag": dict(lag=0.02)}[settlement]
    product = make(disable=True, carried=True, **kw)
    e = carried_env()
    T = float(product.get_post_maturity_time(e))
    # the booked settlement date is the product terminal's payment
    payment = {"none": T, "settlement_date": float(resolve_terminal_timing(product, e).payment_time),
               "lag": T + 0.02}[settlement]
    exact = -_bs_put(T) * float(e.get_discount_factor(payment)) / float(e.get_discount_factor(T))
    value = ENGINES[engine][0]().price(product, _env_for(engine, carried_env))
    assert abs(value - exact) <= PUT_TOLERANCE[engine] * abs(exact), (value, exact, (value - exact) / exact)


def test_heston_adi_knocked_in_note_ends_at_schedule_two_end():
    """Heston ADI (no closed form): the carried knock-in under ``disable_ko_after_ki`` booked with exercise 2026-01-05
    prices as the same contract booked with exercise on schedule 2's end (both notes end by 12-03), to the time
    grid's accuracy. Before T7 the 01-05 booking ran the put to 01-05: 1.0e-3 relative apart."""
    engine = lambda: HestonKOResetSnowballPDESolver(HESTON, n_x=96, n_v=16, n_t=64)  # noqa: E731
    late = engine().price(make(disable=True, carried=True), carried_env())
    on_end = engine().price(make(disable=True, carried=True, exercise=SCHEDULE_2_END), carried_env())
    assert abs(late - on_end) <= 1e-4 * abs(on_end), (late, on_end, (late - on_end) / on_end)


# --- against Monte Carlo --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mc_reference():
    cache = {}

    def get(name):
        if name not in cache:
            make_product, make_env = CASES[name]
            mc = SnowballMCEngine(params=MCParams(num_paths=MC_PATHS, seed=7))
            stats = mc.calculate_event_stats(make_product(), make_env())
            cache[name] = (float(stats.pv), float(mc.get_last_std_error()), stats)
        return cache[name]

    return get


@pytest.mark.slow
@pytest.mark.parametrize("engine", list(ENGINES))
@pytest.mark.parametrize("name", list(CASES))
def test_price_agrees_with_mc(mc_reference, name, engine):
    """Within 3 SE of Black-Scholes MC (1M paths, seed 7). Before T7 the grid engines sat at -8.6 SE (ki97 disable)
    to +13 SE (carried knock-in) of MC on these cases (investigation t1)."""
    make_product, make_env = CASES[name]
    ref, se, _ = mc_reference(name)
    value = ENGINES[engine][0]().price(make_product(), _env_for(engine, make_env))
    assert abs(value - ref) <= 3.0 * se, (value, ref, se, (value - ref) / se)


def _v1_route(route):
    engine = KOResetSnowballQuadEngine(params=QuadParams(grid_points=1001))
    engine._v1_statistics_route = route
    return engine


STATS_ENGINES = {
    "pde": lambda: KOResetSnowballPDESolver(PDEParams()),
    "heston_pde": lambda: HestonKOResetSnowballPDESolver(HESTON, n_x=64, n_v=12, n_t=48),
    "v1_forward": lambda: _v1_route("forward"),
    "v1_stacked": lambda: _v1_route("stacked"),
    "v2": lambda: KOResetSnowballQuadEngineV2(QuadV2Params()),
}


@pytest.mark.slow
@pytest.mark.parametrize("engine", list(STATS_ENGINES))
@pytest.mark.parametrize("name", ["ki85", "ki85_settlement_date", "ki85_lag", "carried_jul7"])
def test_knocked_in_settlement_row_sits_at_schedule_two_end(mc_reference, name, engine):
    """The knocked-in settlement row: determined at schedule 2's end and paid where MC pays it (the 0.9127 row of the
    investigation, t4); its probability within 3 binomial SE of MC on the Black-Scholes engines; the statistics
    reconcile with the price."""
    make_product, make_env = CASES[name]
    _, _, mc = mc_reference(name)
    stats = STATS_ENGINES[engine]().calculate_event_stats(make_product(), make_env())
    row, mc_row = stats.ki_settlement, mc.ki_settlement
    assert row.time == pytest.approx(mc_row.time, abs=1e-12), (row.time, mc_row.time)
    assert row.payment_time == pytest.approx(mc_row.payment_time, abs=1e-12), (row.payment_time, mc_row.payment_time)
    assert abs(stats.reconciliation_error) <= 1e-6 * max(1.0, abs(stats.pv))
    if engine != "heston_pde":
        sd = math.sqrt(mc_row.probability * (1.0 - mc_row.probability) / MC_PATHS)
        assert abs(row.probability - mc_row.probability) <= 3.0 * sd, (row.probability, mc_row.probability, sd)


@pytest.mark.parametrize("engine", list(STATS_ENGINES))
def test_expiry_pay_pays_a_schedule_two_knock_out_at_schedule_two_end(engine):
    """EXPIRY pay: a schedule-2 knock-out is paid with the knocked-in note, at schedule 2's end's settlement (where
    MC pays it, investigation t9), not at the exercise date's."""
    product = lambda: make(KI85, pay={"accrual_config.coupon_pay_type": CouponPayType.EXPIRY})  # noqa: E731
    mc = SnowballMCEngine(params=MCParams(num_paths=20_000, seed=7)).calculate_event_stats(product(), env())
    stats = STATS_ENGINES[engine]().calculate_event_stats(product(), env())
    post = [i for i, tag in enumerate(stats.observations.schedule) if tag == "post"]
    assert post
    np.testing.assert_allclose(stats.observations.payment_times[post], mc.observations.payment_times[post],
                               rtol=0.0, atol=1e-12)
    assert stats.ki_settlement.payment_time == pytest.approx(mc.ki_settlement.payment_time, abs=1e-12)


# --- cash-leg folding --------------------------------------------------------------------------------------------

FOLD_ENGINES = {
    **{name: make for name, (make, _) in ENGINES.items()},
    "heston_pde": lambda: HestonKOResetSnowballPDESolver(HESTON, n_x=64, n_v=12, n_t=48),
    "mc": lambda: SnowballMCEngine(params=MCParams(num_paths=20_000, seed=7)),
}


@pytest.mark.parametrize("pay", ["instant", "expiry"])
@pytest.mark.parametrize("engine", list(FOLD_ENGINES))
def test_folded_cash_lands_where_the_statistics_put_it(engine, pay):
    """The fold structure's knocked-in slot and schedule-2 EXPIRY payments sit at schedule 2's end (D31): every engine
    binds the overlay (Monte Carlo already settled the knocked-in branch there and refused the old structure's slot),
    and the folded cash is the engine's own statistics times the amounts, discounted from each payment."""
    from quantark.cashleg.folding import FoldedEventCash, FoldOverlay, _event_structure

    kw = {} if pay == "instant" else dict(pay={"accrual_config.coupon_pay_type": CouponPayType.EXPIRY})
    product = lambda: make(KI85, **kw)  # noqa: E731
    e = _env_for(engine, env) if engine in ENGINES else env()
    structure = _event_structure(product(), env())
    assert structure.ki.determination_time == pytest.approx(float(product().get_post_maturity_time(env())), abs=1e-12)
    amounts = [0.3 + 0.1 * i for i in range(len(structure.ko))]
    overlay = FoldOverlay(
        ko=tuple(FoldedEventCash(s.schedule, s.determination_time, s.payment_time, a)
                 for s, a in zip(structure.ko, amounts)),
        maturity_no_ki=FoldedEventCash("pre", structure.no_ki.determination_time, structure.no_ki.payment_time, 0.25),
        maturity_ki=FoldedEventCash("post", structure.ki.determination_time, structure.ki.payment_time, -0.5),
    )
    make_engine = FOLD_ENGINES[engine]
    folded = make_engine().price_folded(product(), e, overlay) - make_engine().price(product(), e)
    stats = make_engine().calculate_event_stats(product(), e)
    obs = stats.observations
    df = lambda t: float(e.get_discount_factor(t))  # noqa: E731
    expected = sum(a * p * df(t) for a, p, t in zip(amounts, obs.probability, obs.payment_times))
    expected += 0.25 * stats.maturity.probability * df(stats.maturity.payment_time)
    expected -= 0.5 * stats.ki_settlement.probability * df(stats.ki_settlement.payment_time)
    assert abs(folded - expected) <= 1e-10, (folded, expected)


# --- bitwise: exercise on schedule 2's end --------------------------------------------------------------------------

def _pinned(name):
    """Products whose exercise date IS schedule 2's last fixing (one maturity: unchanged by T7)."""
    on_end = dict(exercise=SCHEDULE_2_END)
    return {
        "base": (lambda: make(**on_end), env),
        "disable": (lambda: make(disable=True, **on_end), env),
        "continuous": (lambda: make(ki_type=ObservationType.CONTINUOUS, **on_end), env),
        "carried": (lambda: make(carried=True, **on_end), carried_env),
        "carried_disable": (lambda: make(disable=True, carried=True, **on_end), carried_env),
        "expiry": (lambda: make(pay={"accrual_config.coupon_pay_type": CouponPayType.EXPIRY}, **on_end), env),
        "lag": (lambda: make(lag=0.02, **on_end), env),
        "ki_pair": (lambda: make(KOR_CASES["ki_pair"], **on_end), env),
    }[name]


PIN_ENGINES = {
    "pde": (lambda: KOResetSnowballPDESolver(PDEParams()), False),
    "lv_pde": (lambda: LocalVolKOResetSnowballPDESolver(PDEParams()), True),
    "heston": (lambda: HestonKOResetSnowballPDESolver(HESTON, n_x=64, n_v=12, n_t=48), False),
    "v1": (lambda: KOResetSnowballQuadEngine(params=QuadParams(grid_points=1001)), False),
    "v2": (KOResetSnowballQuadEngineV2, False),
}


def stats_key(stats):
    """pv, the knock-out rows, the maturity and knocked-in settlement rows (probability, times, cash)."""
    out = [stats.pv, *stats.observations.probability, *stats.observations.payment_times]
    for row in (stats.maturity, stats.ki_settlement):
        out += [row.probability, row.time, row.payment_time, row.discounted_cash]
    return [float(x).hex() for x in out]


#: pinned on d83256db (before T7)
PRICE_PINS = {
    ("base", "pde"): "0x1.1ba1680f9314ap+3",
    ("base", "lv_pde"): "0x1.1ba1680f9314cp+3",
    ("base", "heston"): "0x1.10ed738d4bdccp+3",
    ("base", "v1"): "0x1.1b9804bd93d19p+3",
    ("base", "v2"): "0x1.1ba4450b66f17p+3",
    ("disable", "pde"): "0x1.1ae730abdbe94p+3",
    ("disable", "lv_pde"): "0x1.1ae730abdbe98p+3",
    ("disable", "heston"): "0x1.0fb288a3add30p+3",
    ("disable", "v1"): "0x1.1add9b44824dcp+3",
    ("disable", "v2"): "0x1.1aea1a97d6c92p+3",
    ("continuous", "pde"): "0x1.f8ca5a2a1cd9fp+2",
    ("continuous", "lv_pde"): "0x1.f8ca5a2a1cd75p+2",
    ("continuous", "heston"): "0x1.de7cdccb2c8b5p+2",
    ("continuous", "v1"): "0x1.f8bf40f2f2cc3p+2",
    ("continuous", "v2"): "0x1.f8d4c46c3c435p+2",
    ("carried", "pde"): "-0x1.2b25fed77d8d9p+4",
    ("carried", "lv_pde"): "-0x1.2b25fed77d96dp+4",
    ("carried", "heston"): "-0x1.2aa68bd1123aap+4",
    ("carried", "v1"): "-0x1.2b26b78d22f41p+4",
    ("carried", "v2"): "-0x1.2b25722c65763p+4",
    ("carried_disable", "pde"): "-0x1.3c5f2577595e1p+4",
    ("carried_disable", "lv_pde"): "-0x1.3c5f25775967cp+4",
    ("carried_disable", "heston"): "-0x1.39288531cfea8p+4",
    ("carried_disable", "v1"): "-0x1.3c5ef5e728866p+4",
    ("carried_disable", "v2"): "-0x1.3c5ef1a1d32ccp+4",
    ("expiry", "pde"): "0x1.1a2aea057a38fp+3",
    ("expiry", "lv_pde"): "0x1.1a2aea057a392p+3",
    ("expiry", "heston"): "0x1.0f64e135f3ef2p+3",
    ("expiry", "v1"): "0x1.1a217628b29b9p+3",
    ("expiry", "v2"): "0x1.1a2dc735ab379p+3",
    ("lag", "pde"): "0x1.1b845e582e45ep+3",
    ("lag", "lv_pde"): "0x1.1b845e582e465p+3",
    ("lag", "heston"): "0x1.10d18259d18adp+3",
    ("lag", "v1"): "0x1.1b7afbfc3c02ep+3",
    ("lag", "v2"): "0x1.1b873b08f7401p+3",
    ("ki_pair", "pde"): "0x1.ee104e0f4e1e3p+2",
    ("ki_pair", "lv_pde"): "0x1.ee104e0f4e1bcp+2",
    ("ki_pair", "heston"): "0x1.e8cf304f84063p+2",
    ("ki_pair", "v1"): "0x1.edf09c9e8d9f4p+2",
    ("ki_pair", "v2"): "0x1.ee130e47f5007p+2",
}
STATS_PINS = {
    ("base", "pde"): ['0x1.1ba1680f9314ap+3', '0x1.80279c3ebac48p-2', '0x1.24f2a28f84973p-3', '0x1.29622beca2c8dp-4', '0x1.bcb2d0c52b6f6p-11', '0x1.bcce8fb6a2ee4p-10', '0x1.41d5ad29ac10fp-10', '0x1.3cf3cf3cf3cf4p-3', '0x1.5555555555555p-2', '0x1.0000000000000p-1', '0x1.5555555555555p-1', '0x1.aaaaaaaaaaaabp-1', '0x1.d34d34d34d34dp-1', '0x1.64a5fd6a1c76ap-2', '0x1.0000000000000p-1', '0x1.0000000000000p-1', '0x1.b95fe65fff370p+0', '0x1.d41c6396b3c55p-5', '0x1.d34d34d34d34dp-1', '0x1.d34d34d34d34dp-1', '-0x1.b0f59d7e5e572p+0'],
    ("base", "lv_pde"): ['0x1.1ba1680f9314cp+3', '0x1.80279c3ebac5cp-2', '0x1.24f2a28f84974p-3', '0x1.29622beca2cb0p-4', '0x1.bcb2d0c52b7cap-11', '0x1.bcce8fb6a3043p-10', '0x1.41d5ad29ac298p-10', '0x1.3cf3cf3cf3cf4p-3', '0x1.5555555555555p-2', '0x1.0000000000000p-1', '0x1.5555555555555p-1', '0x1.aaaaaaaaaaaabp-1', '0x1.d34d34d34d34dp-1', '0x1.64a5fd6a1c7fep-2', '0x1.0000000000000p-1', '0x1.0000000000000p-1', '0x1.b95fe65fff431p+0', '0x1.d41c6396b3d88p-5', '0x1.d34d34d34d34dp-1', '0x1.d34d34d34d34dp-1', '-0x1.b0f59d7e5e66bp+0'],
    ("base", "v1"): ['0x1.1b9804bd93d19p+3', '0x1.80449db350e3dp-2', '0x1.24cd7065d8550p-3', '0x1.29493ccbf5ccbp-4', '0x1.bd3743ddeb43fp-11', '0x1.bd279dd5f1adap-10', '0x1.421661503d5dfp-10', '0x1.3cf3cf3cf3cf4p-3', '0x1.5555555555555p-2', '0x1.0000000000000p-1', '0x1.5555555555555p-1', '0x1.aaaaaaaaaaaabp-1', '0x1.d34d34d34d34dp-1', '0x1.6497da39f3568p-2', '0x1.0000000000000p-1', '0x1.0000000000000p-1', '0x1.b94e67677bd20p+0', '0x1.d465385de81cap-5', '0x1.d34d34d34d34dp-1', '0x1.d34d34d34d34dp-1', '-0x1.b13fdec6078bep+0'],
    ("base", "v2"): ['0x1.1ba4450b66f17p+3', '0x1.80268d2a30d98p-2', '0x1.24f61998cb59cp-3', '0x1.29640d787cedfp-4', '0x1.bc6697ca0586ep-11', '0x1.bcbb33e26a34fp-10', '0x1.41cce6cbfef2dp-10', '0x1.3cf3cf3cf3cf4p-3', '0x1.5555555555555p-2', '0x1.0000000000000p-1', '0x1.5555555555555p-1', '0x1.aaaaaaaaaaaabp-1', '0x1.d34d34d34d34dp-1', '0x1.64a75ac99f7a0p-2', '0x1.0000000000000p-1', '0x1.0000000000000p-1', '0x1.b96196bef6d21p+0', '0x1.d40a63d8bac35p-5', '0x1.d34d34d34d34dp-1', '0x1.d34d34d34d34dp-1', '-0x1.b0e472bd3c341p+0'],
    ("carried", "pde"): ['-0x1.2b25fed77d8d9p+4', '0x1.5c487f0ba9c14p-5', '0x1.559ce07fb82d7p-4', '0x1.7dfc6a6f711f6p-5', '0x1.4d34d34d34d35p-3', '0x1.5145145145145p-2', '0x1.a28a28a28a28ap-2', '0x0.0p+0', '0x1.a28a28a28a28ap-2', '0x1.a28a28a28a28ap-2', '0x0.0p+0', '0x1.a7a8155abb13ap-1', '0x1.a28a28a28a28ap-2', '0x1.a28a28a28a28ap-2', '-0x1.3361259e97948p+4'],
    ("carried", "lv_pde"): ['-0x1.2b25fed77d96dp+4', '0x1.5c487f0ba9c65p-5', '0x1.559ce07fb833ep-4', '0x1.7dfc6a6f71267p-5', '0x1.4d34d34d34d35p-3', '0x1.5145145145145p-2', '0x1.a28a28a28a28ap-2', '0x0.0p+0', '0x1.a28a28a28a28ap-2', '0x1.a28a28a28a28ap-2', '0x0.0p+0', '0x1.a7a8155abb206p-1', '0x1.a28a28a28a28ap-2', '0x1.a28a28a28a28ap-2', '-0x1.3361259e979e3p+4'],
    ("carried", "heston"): ['-0x1.2aa68bd1123aap+4', '0x1.1755c99f6cd04p-5', '0x1.2d56c61005dffp-4', '0x1.607dc042fbd00p-5', '0x1.4d34d34d34d35p-3', '0x1.5145145145145p-2', '0x1.a28a28a28a28ap-2', '0x0.0p+0', '0x1.a28a28a28a28ap-2', '0x1.a28a28a28a28ap-2', '0x0.0p+0', '0x1.b2d7ee9fd8b7cp-1', '0x1.a28a28a28a28ap-2', '0x1.a28a28a28a28ap-2', '-0x1.31d6b385c627bp+4'],
    ("carried", "v1"): ['-0x1.2b26b78d22f41p+4', '0x1.5c254b692100fp-5', '0x1.55a04afe61a3ep-4', '0x1.7dfb4341cb7c9p-5', '0x1.4d34d34d34d35p-3', '0x1.5145145145145p-2', '0x1.a28a28a28a28ap-2', '0x0.0p+0', '0x1.a28a28a28a28ap-2', '0x1.a28a28a28a28ap-2', '0x0.0p+0', '0x1.a7a9edb58503cp-1', '0x1.a28a28a28a28ap-2', '0x1.a28a28a28a28ap-2', '-0x1.3361b229cbe0fp+4'],
    ("carried", "v2"): ['-0x1.2b25722c65763p+4', '0x1.5c30b22347665p-5', '0x1.55a4f8d775045p-4', '0x1.7e0a53ae710a8p-5', '0x1.4d34d34d34d35p-3', '0x1.5145145145145p-2', '0x1.a28a28a28a28ap-2', '0x0.0p+0', '0x1.a28a28a28a28ap-2', '0x1.a28a28a28a28ap-2', '0x0.0p+0', '0x1.a7a7b087f5d82p-1', '0x1.a28a28a28a28ap-2', '0x1.a28a28a28a28ap-2', '-0x1.3360a2306a81fp+4'],
    ("expiry", "pde"): ['0x1.1a2aea057a38fp+3', '0x1.80279c3ebac47p-2', '0x1.24f2a28f84972p-3', '0x1.29622beca2c8dp-4', '0x1.bcb2d0c52b701p-11', '0x1.bcce8fb6a2eefp-10', '0x1.41d5ad29ac10fp-10', '0x1.0000000000000p-1', '0x1.0000000000000p-1', '0x1.0000000000000p-1', '0x1.d34d34d34d34dp-1', '0x1.d34d34d34d34dp-1', '0x1.d34d34d34d34dp-1', '0x1.64a5fd6a1c76ap-2', '0x1.0000000000000p-1', '0x1.0000000000000p-1', '0x1.b95fe65fff370p+0', '0x1.d41c6396b3c55p-5', '0x1.d34d34d34d34dp-1', '0x1.d34d34d34d34dp-1', '-0x1.b0f59d7e5e572p+0'],
    ("expiry", "lv_pde"): ['0x1.1a2aea057a392p+3', '0x1.80279c3ebac5fp-2', '0x1.24f2a28f84970p-3', '0x1.29622beca2cb0p-4', '0x1.bcb2d0c52b7c4p-11', '0x1.bcce8fb6a303ep-10', '0x1.41d5ad29ac298p-10', '0x1.0000000000000p-1', '0x1.0000000000000p-1', '0x1.0000000000000p-1', '0x1.d34d34d34d34dp-1', '0x1.d34d34d34d34dp-1', '0x1.d34d34d34d34dp-1', '0x1.64a5fd6a1c7fep-2', '0x1.0000000000000p-1', '0x1.0000000000000p-1', '0x1.b95fe65fff431p+0', '0x1.d41c6396b3d88p-5', '0x1.d34d34d34d34dp-1', '0x1.d34d34d34d34dp-1', '-0x1.b0f59d7e5e66bp+0'],
    ("expiry", "v1"): ['0x1.1a217628b29b9p+3', '0x1.80449db350e3dp-2', '0x1.24cd7065d8550p-3', '0x1.29493ccbf5ccbp-4', '0x1.bd3743ddeb43fp-11', '0x1.bd279dd5f1adap-10', '0x1.421661503d5dfp-10', '0x1.0000000000000p-1', '0x1.0000000000000p-1', '0x1.0000000000000p-1', '0x1.d34d34d34d34dp-1', '0x1.d34d34d34d34dp-1', '0x1.d34d34d34d34dp-1', '0x1.6497da39f3568p-2', '0x1.0000000000000p-1', '0x1.0000000000000p-1', '0x1.b94e67677bd20p+0', '0x1.d465385de81cap-5', '0x1.d34d34d34d34dp-1', '0x1.d34d34d34d34dp-1', '-0x1.b13fdec6078bep+0'],
    ("expiry", "v2"): ['0x1.1a2dc735ab379p+3', '0x1.80268d2a30d98p-2', '0x1.24f61998cb59cp-3', '0x1.29640d787cedfp-4', '0x1.bc6697ca0586ep-11', '0x1.bcbb33e26a34fp-10', '0x1.41cce6cbfef2dp-10', '0x1.0000000000000p-1', '0x1.0000000000000p-1', '0x1.0000000000000p-1', '0x1.d34d34d34d34dp-1', '0x1.d34d34d34d34dp-1', '0x1.d34d34d34d34dp-1', '0x1.64a75ac99f7a0p-2', '0x1.0000000000000p-1', '0x1.0000000000000p-1', '0x1.b96196bef6d21p+0', '0x1.d40a63d8bac35p-5', '0x1.d34d34d34d34dp-1', '0x1.d34d34d34d34dp-1', '-0x1.b0e472bd3c341p+0'],
    ("lag", "pde"): ['0x1.1b845e582e45ep+3', '0x1.80279c3ebac45p-2', '0x1.24f2a28f84973p-3', '0x1.29622beca2c8bp-4', '0x1.bcb2d0c52b6f9p-11', '0x1.bcce8fb6a2eeap-10', '0x1.41d5ad29ac10ap-10', '0x1.65e991cc4ff83p-3', '0x1.69d0369d0369dp-2', '0x1.0a3d70a3d70a4p-1', '0x1.5f92c5f92c5f9p-1', '0x1.b4e81b4e81b4fp-1', '0x1.dd8aa577243f1p-1', '0x1.64a5fd6a1c76ap-2', '0x1.0000000000000p-1', '0x1.0a3d70a3d70a4p-1', '0x1.b932b65175cb6p+0', '0x1.d41c6396b3c55p-5', '0x1.d34d34d34d34dp-1', '0x1.dd8aa577243f1p-1', '-0x1.b0c949fdd1b84p+0'],
    ("lag", "lv_pde"): ['0x1.1b845e582e465p+3', '0x1.80279c3ebac60p-2', '0x1.24f2a28f8496fp-3', '0x1.29622beca2ca5p-4', '0x1.bcb2d0c52b7d7p-11', '0x1.bcce8fb6a302bp-10', '0x1.41d5ad29ac298p-10', '0x1.65e991cc4ff83p-3', '0x1.69d0369d0369dp-2', '0x1.0a3d70a3d70a4p-1', '0x1.5f92c5f92c5f9p-1', '0x1.b4e81b4e81b4fp-1', '0x1.dd8aa577243f1p-1', '0x1.64a5fd6a1c7fep-2', '0x1.0000000000000p-1', '0x1.0a3d70a3d70a4p-1', '0x1.b932b65175d74p+0', '0x1.d41c6396b3d88p-5', '0x1.d34d34d34d34dp-1', '0x1.dd8aa577243f1p-1', '-0x1.b0c949fdd1c73p+0'],
    ("lag", "v1"): ['0x1.1b7afbfc3c02ep+3', '0x1.80449db350e3dp-2', '0x1.24cd7065d8550p-3', '0x1.29493ccbf5ccap-4', '0x1.bd3743ddeb43fp-11', '0x1.bd279dd5f1adbp-10', '0x1.421661503d5dep-10', '0x1.65e991cc4ff83p-3', '0x1.69d0369d0369dp-2', '0x1.0a3d70a3d70a4p-1', '0x1.5f92c5f92c5f9p-1', '0x1.b4e81b4e81b4fp-1', '0x1.dd8aa577243f1p-1', '0x1.6497da39f3568p-2', '0x1.0000000000000p-1', '0x1.0a3d70a3d70a4p-1', '0x1.b921392380732p+0', '0x1.d465385de81cap-5', '0x1.d34d34d34d34dp-1', '0x1.dd8aa577243f1p-1', '-0x1.b11383ab51b4fp+0'],
    ("lag", "v2"): ['0x1.1b873b08f7401p+3', '0x1.80268d2a30d98p-2', '0x1.24f61998cb59cp-3', '0x1.29640d787cedfp-4', '0x1.bc6697ca0586ep-11', '0x1.bcbb33e26a34fp-10', '0x1.41cce6cbfef2dp-10', '0x1.65e991cc4ff83p-3', '0x1.69d0369d0369dp-2', '0x1.0a3d70a3d70a4p-1', '0x1.5f92c5f92c5f9p-1', '0x1.b4e81b4e81b4fp-1', '0x1.dd8aa577243f1p-1', '0x1.64a75ac99f7a0p-2', '0x1.0000000000000p-1', '0x1.0a3d70a3d70a4p-1', '0x1.b934668429519p+0', '0x1.d40a63d8bac35p-5', '0x1.d34d34d34d34dp-1', '0x1.dd8aa577243f1p-1', '-0x1.b0b820fe9e64ap+0'],
}


@pytest.mark.parametrize("product,engine", sorted(PRICE_PINS))
def test_exercise_on_schedule_two_end_prices_bitwise(product, engine):
    make_product, make_env = _pinned(product)
    grid = PIN_ENGINES[engine][1]
    value = PIN_ENGINES[engine][0]().price(make_product(), make_env(grid=grid))
    assert float(value).hex() == PRICE_PINS[(product, engine)]


@pytest.mark.parametrize("product,engine", sorted(STATS_PINS))
def test_exercise_on_schedule_two_end_reports_bitwise(product, engine):
    make_product, make_env = _pinned(product)
    grid = PIN_ENGINES[engine][1]
    stats = PIN_ENGINES[engine][0]().calculate_event_stats(make_product(), make_env(grid=grid))
    assert stats_key(stats) == STATS_PINS[(product, engine)]
