"""The PDE 1D forward adjoint pass (unified event statistics spec T13; DESIGN §4.5 option A).

The statistics march's backward columns (T12) are read at the spot; the forward pass runs the exact transpose of the
same march -- the same Crank-Nicolson steps and Dirichlet rows, the same event stages in the family's order, the same
FIRST_PASSAGE correction -- from the readout's own weights, and reads the first knock-in mass at every KI observation
(continuous KI: every grid step, spread over trading days):

- AC14: the forward pass reproduces every backward column's readout (knock-out units, the maturity and knock-in
  settlement units, the maturity cash) to round-off, hence the termination probabilities to <= 1e-10 abs;
- AC13: Σ rows = P(KI); Σ t p / P(KI) = E[KI time] (discrete: round-off; continuous: one trading day); rows on the
  product's KI schedule (continuous: one per trading day, numeric clock 1/bus_days_in_year); ``interpolated`` where a
  day boundary is not a grid time; payment time = observation time (D8);
- AC9: no knock-in after a certain knock-out; KO wins a same-instant discrete tie;
- an independent backward number: P(KI before KO) of a note equals the knocked-in settlement of its
  ``disable_ko_after_ki`` twin (the knocked-in surface never knocks out, so every first knock-in before a knock-out
  settles knocked in), within the march's discount residual;
- the MC reference (RQMC, independent batches): P(KI), E[KI time] and the per-date table within 4 batch SEs plus the
  PDE's refinement allowance, the allowance shrinking under refinement;
- D17: a request without the KI stream never runs the pass.
"""
from __future__ import annotations

import math
import warnings
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde import GridConfig
from quantark.asset.equity.engine.pde.core.march import march
from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, market_scope, prepare
from quantark.asset.equity.engine.pde.core.statistics import adjoint
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.ko_reset_vol_pde_solvers import LocalVolKOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import LocalVolPhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import LocalVolSnowballPDESolver
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.cashleg import EventType
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.engine_enums import MonteCarloMethod

from pde_statistics.cases import trading_clock_env
from test_asset.equity.engine_mc import mc_price_path_routes as grid

warnings.filterwarnings("ignore", category=DeprecationWarning)

EXPIRY = CouponPayType.EXPIRY
#: AC14: the forward pass reproduces the backward termination probabilities to round-off
ADJOINT_BOUND = 1e-10
#: the march's discount residual on a unit column (T12 RESIDUAL_BOUND, measured)
RESIDUAL_BOUND = 1e-8
#: P(KI before KO) vs the disable_ko_after_ki twin's KI settlement: two marches' discount errors, O(dt^2) (measured
#: max 2.6e-10 at the default grid, sb_cki; x1/4 per halving on sb_cki, sb_dki, ph_arrears_dki, lv_kr_abs)
TWIN_BOUND = 2e-9

# --- dated products on the SSE calendar (the T12 byte-compare's) ------------------------------------------------

SSE = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2029))
T0 = datetime(2026, 2, 9)
AGED = datetime(2026, 6, 17)


def _monthly(t0, months):
    out = []
    for k in range(1, months + 1):
        month = (t0.month - 1 + k) % 12 + 1
        year = t0.year + (t0.month - 1 + k) // 12
        d = datetime(year, month, min(t0.day, 28))
        while not SSE.is_business_day(d):
            d += timedelta(days=1)
        out.append(d)
    return out


def dated_env(valuation=T0, spot=100.0, calendar=None):
    return PricingEnvironment(rate_curve=FlatRateCurve(0.03), valuation_date=valuation, spot_quote=SpotQuote(spot),
                              vol_surface=FlatVolSurface(0.22), div_yield=ContinuousDividendYield(0.01),
                              calendar=calendar)


def _dated_schedule(dates, level):
    return ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=level) for d in dates])


def dated_snowball(ki="discrete"):
    dates = _monthly(T0, 12)
    ki_kw = (dict(ki_barrier=75.0, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
             if ki == "continuous" else
             dict(ki_barrier=75.0, ki_observation_type=ObservationType.DISCRETE,
                  ki_observation_schedule=_dated_schedule(dates, 75.0)))
    return SnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, initial_date=T0, exercise_date=dates[-1],
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.12, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_dated_schedule(dates, 103.0), **ki_kw),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True))


def dated_phoenix(memory=True, pay=CouponPayType.INSTANT):
    dates = _monthly(T0, 12)
    return PhoenixOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, initial_date=T0, exercise_date=dates[-1],
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_dated_schedule(dates, 103.0), ki_barrier=75.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=_dated_schedule(dates, 75.0)),
        coupon_config=CouponBarrierConfig(coupon_barrier=90.0, coupon_rate=0.12, memory_coupon=memory,
                                          coupon_pay_type=pay),
        payoff_config=PayoffConfig(include_principal=True))


def dated_ko_reset():
    pre, post = _monthly(T0, 6), _monthly(T0, 12)
    return KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, initial_date=T0, exercise_date=post[-1],
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_dated_schedule(pre, 103.0), ki_barrier=75.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=_dated_schedule(post, 75.0)),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_dated_schedule(post, 95.0)),
        payoff_config=PayoffConfig(rebate_rate=0.15, include_principal=True))


# --- the route matrix -----------------------------------------------------------------------------------------

def _flat(cls, **kw):
    return lambda: cls(PDEParams(**kw))


def _lv(cls, **kw):
    return lambda: cls(params=PDEParams(**kw))


#: name -> (solver factory, product factory, env factory)
ROUTES = {
    "sb_cki": (_flat(SnowballPDESolver), grid.snowball, grid.flat_env),
    "sb_dki": (_flat(SnowballPDESolver), lambda: grid.snowball("discrete"), grid.flat_env),
    "sb_reverse": (_flat(SnowballPDESolver), lambda: grid.snowball(reverse=True), grid.flat_env),
    "sb_reverse_dki": (_flat(SnowballPDESolver), lambda: grid.snowball("discrete", reverse=True), grid.flat_env),
    "sb_no_ko_after_ki": (_flat(SnowballPDESolver), lambda: grid.snowball(disable_ko_after_ki=True), grid.flat_env),
    "sb_no_ko_after_ki_dki": (_flat(SnowballPDESolver), lambda: grid.snowball("discrete", disable_ko_after_ki=True),
                              grid.flat_env),
    "sb_expiry": (_flat(SnowballPDESolver), lambda: grid.snowball(pay=EXPIRY), grid.flat_env),
    "sb_nodal_cki": (_flat(SnowballPDESolver, event_projection="nodal"), grid.snowball, grid.flat_env),
    "sb_nodal_dki": (_flat(SnowballPDESolver, event_projection="nodal"), lambda: grid.snowball("discrete"),
                     grid.flat_env),
    "sb_fast": (_flat(SnowballPDESolver, accuracy="fast"), grid.snowball, grid.flat_env),
    "ph_mem_instant": (_flat(PhoenixPDESolver), grid.phoenix, grid.flat_env),
    "ph_nomem_instant": (_flat(PhoenixPDESolver), lambda: grid.phoenix(memory=False), grid.flat_env),
    "ph_mem_expiry": (_flat(PhoenixPDESolver), lambda: grid.phoenix(pay=EXPIRY), grid.flat_env),
    "ph_nomem_expiry": (_flat(PhoenixPDESolver), lambda: grid.phoenix(memory=False, pay=EXPIRY), grid.flat_env),
    "ph_arrears_dki": (_flat(PhoenixPDESolver), lambda: grid.phoenix(ki="discrete", arrears=0.02,
                                                                     coupon_barriers=[106.0, 106.0, 80.0, 80.0]),
                       grid.flat_env),
    "ph_nodal_dki": (_flat(PhoenixPDESolver, event_projection="nodal"), lambda: grid.phoenix(ki="discrete"),
                     grid.flat_env),
    "kr_abs_dki": (_flat(KOResetSnowballPDESolver), grid.ko_reset, grid.flat_env),
    "kr_abs_cki": (_flat(KOResetSnowballPDESolver), lambda: grid.ko_reset(ki="continuous"), grid.flat_env),
    "kr_nodal_dki": (_flat(KOResetSnowballPDESolver, event_projection="nodal"), grid.ko_reset, grid.flat_env),
    "lv_sb_cki": (_lv(LocalVolSnowballPDESolver), grid.snowball, grid.grid_env),
    "lv_sb_dki": (_lv(LocalVolSnowballPDESolver), lambda: grid.snowball("discrete"), grid.grid_env),
    "lv_ph_mem_expiry": (_lv(LocalVolPhoenixPDESolver), lambda: grid.phoenix(pay=EXPIRY), grid.grid_env),
    "lv_ph_nomem_instant": (_lv(LocalVolPhoenixPDESolver), lambda: grid.phoenix(memory=False), grid.grid_env),
    "lv_kr_abs": (_lv(LocalVolKOResetSnowballPDESolver), grid.ko_reset, grid.grid_env),
    "lv_kr_cki": (_lv(LocalVolKOResetSnowballPDESolver), lambda: grid.ko_reset(ki="continuous"), grid.grid_env),
    "dated_sb_dki_fresh": (_flat(SnowballPDESolver), dated_snowball, dated_env),
    "dated_sb_dki_aged": (_flat(SnowballPDESolver), dated_snowball, lambda: dated_env(AGED)),
    "dated_sb_cki_calendar": (_flat(SnowballPDESolver), lambda: dated_snowball("continuous"),
                              lambda: dated_env(calendar=SSE)),
    "dated_sb_cki_aged": (_flat(SnowballPDESolver), lambda: dated_snowball("continuous"), lambda: dated_env(AGED)),
    "dated_ph_mem_expiry_aged": (_flat(PhoenixPDESolver), lambda: dated_phoenix(pay=EXPIRY), lambda: dated_env(AGED)),
    "dated_ph_mem_instant_fresh": (_flat(PhoenixPDESolver), dated_phoenix, dated_env),
    "dated_kr_fresh": (_flat(KOResetSnowballPDESolver), dated_ko_reset, dated_env),
    "dated_kr_aged": (_flat(KOResetSnowballPDESolver), dated_ko_reset, lambda: dated_env(AGED)),
    "trading_clock_sb_cki": (_flat(SnowballPDESolver), grid.snowball, lambda: trading_clock_env()[0]),
}


# --- CP4: a KI schedule that ends before the note does (the price's knock-in-side row then has no event) ---------

#: a narrow spatial domain: the knock-in-side far edge absorbs measurable mass (num_std 3: ~6e-4 after the last
#: KI observation on the truncated snowball)
NARROW = GridConfig(num_std=3.0)
#: the truncated KI schedule (the notes run to 1.0; KO-reset's first schedule to 1.0 as well)
KI_ENDS = (0.25, 0.5)


def _truncated(make):
    """``make()`` with its discrete KI schedule cut to ``KI_ENDS``."""
    from dataclasses import replace

    def build():
        product = make()
        bc = product.barrier_config
        if bc.ki_observation_schedule is not None:
            level = float(bc.ki_barrier)
            bc = replace(bc, ki_observation_schedule=ObservationSchedule(records=[
                ObservationRecord(observation_time=float(t), barrier=level) for t in KI_ENDS]))
        else:
            bc = replace(bc, ki_observation_dates=list(KI_ENDS))
        product.barrier_config = bc
        product.validate()
        return product

    return build


TRUNCATED = {
    "sb_trunc_dki": (_flat(SnowballPDESolver), _truncated(lambda: grid.snowball("discrete")), grid.flat_env),
    "sb_trunc_dki_narrow": (_flat(SnowballPDESolver, grid=NARROW), _truncated(lambda: grid.snowball("discrete")),
                            grid.flat_env),
    "ph_trunc_dki_narrow": (_flat(PhoenixPDESolver, grid=NARROW), _truncated(lambda: grid.phoenix(ki="discrete")),
                            grid.flat_env),
    "lv_sb_trunc_dki": (_lv(LocalVolSnowballPDESolver), _truncated(lambda: grid.snowball("discrete")),
                        grid.grid_env),
    "lv_sb_trunc_dki_narrow": (_lv(LocalVolSnowballPDESolver, grid=NARROW),
                               _truncated(lambda: grid.snowball("discrete")), grid.grid_env),
    "kr_trunc_dki": (_flat(KOResetSnowballPDESolver), _truncated(grid.ko_reset), grid.flat_env),
    "kr_trunc_dki_narrow": (_flat(KOResetSnowballPDESolver, grid=NARROW), _truncated(grid.ko_reset), grid.flat_env),
    "lv_kr_trunc_dki": (_lv(LocalVolKOResetSnowballPDESolver), _truncated(grid.ko_reset), grid.grid_env),
    "lv_kr_trunc_dki_narrow": (_lv(LocalVolKOResetSnowballPDESolver, grid=NARROW), _truncated(grid.ko_reset),
                               grid.grid_env),
    # the full schedule on the narrow domain: the edge knocks in up to the last observation, then has no event
    "sb_dki_narrow": (_flat(SnowballPDESolver, grid=NARROW), lambda: grid.snowball("discrete"), grid.flat_env),
    "sb_cki_narrow": (_flat(SnowballPDESolver, grid=NARROW), grid.snowball, grid.flat_env),
    "kr_abs_dki_narrow": (_flat(KOResetSnowballPDESolver, grid=NARROW), grid.ko_reset, grid.flat_env),
}
ROUTES.update(TRUNCATED)
#: carried knock-in: no forward pass for the KI path statistics (definitional), but the adjoint identity holds from
#: the knocked-in surface's readout too
CARRIED = {
    "sb_carried": (_flat(SnowballPDESolver), lambda: grid.snowball(knocked_in=True), grid.flat_env),
    "ph_carried": (_flat(PhoenixPDESolver), lambda: grid.phoenix(knocked_in=True), grid.flat_env),
    "kr_carried": (_flat(KOResetSnowballPDESolver), lambda: grid.ko_reset(knocked_in=True), grid.flat_env),
}
CONTINUOUS = {name for name in ROUTES if "cki" in name or name in (
    "sb_reverse", "sb_no_ko_after_ki", "sb_expiry", "sb_fast", "ph_mem_instant", "ph_nomem_instant", "ph_mem_expiry",
    "ph_nomem_expiry", "lv_ph_mem_expiry", "lv_ph_nomem_instant")}


def _route(name, table=ROUTES):
    make_solver, make_product, make_env = table[name]
    return make_solver(), make_product(), make_env()


def _df(env, t):
    return float(env.get_discount_factor(float(t)))


def _adjoint(solver, product, env):
    """(prepared, march result, forward pass with the reconstruction, backward readout ``at``, statistics)."""
    p = prepare(solver, product, env, OutputRequest(Operation.EVENT_STATS, event_stats=True))
    # the facades build a local-vol surface around the whole run (``_with_surface``): the adapter's scope here
    with market_scope(p.solver, p.env, p.market_fps), p.adapter.scope(p.product, p.env):
        result = march(p)
        forward = adjoint.forward_pass(p.statistics, result.states, result.captures, reconstruct=True)
        at = p.statistics._reader(result.states, result.captures)
        stats = p.adapter.read(p, result)
    return p, result, forward, at, stats


def _unit_columns(statistics):
    """The termination outcomes' unit columns and the maturity cash columns of a statistics block (every column the
    generic linear part acts on: the Phoenix trigger and memory slices take their own stages)."""
    L = statistics.layout
    units = [*L.ko, *L.post_ko, *L.maturity_unit] + ([] if L.kis is None else [L.kis])
    return [int(c) for c in units], [int(c) for c in L.maturity]


# --- AC14 ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(ROUTES) + list(CARRIED))
def test_forward_pass_reproduces_every_backward_column(name):
    """The forward pass is the exact discrete adjoint: Σ_j <p_j, affine sources_j> + <p_T, V_T> equals the backward
    readout of every column (unit columns to 1e-10 abs; cash columns to 1e-12 relative)."""
    solver, product, env = _route(name, CARRIED if name in CARRIED else ROUTES)
    p, _, forward, at, _ = _adjoint(solver, product, env)
    units, cash = _unit_columns(p.statistics)
    for c in units:
        assert abs(forward.reconstructed[c] - at(c)) <= ADJOINT_BOUND, (c, forward.reconstructed[c], at(c))
    for c in cash:
        assert math.isclose(forward.reconstructed[c], at(c), rel_tol=1e-12, abs_tol=1e-12), (c,)


@pytest.mark.parametrize("name", list(ROUTES))
def test_forward_pass_reproduces_the_termination_probabilities(name):
    """AC14 as stated: Σ KO, maturity and KI settlement probabilities from the forward pass agree with the backward
    solve's to <= 1e-10 abs (each probability is its unit column over the outcome's DF)."""
    solver, product, env = _route(name)
    p, _, forward, _, stats = _adjoint(solver, product, env)
    L, st = p.statistics.layout, p.statistics
    recon = forward.reconstructed
    records = list(st.ko_records) + list(getattr(st, "post_records", []))
    ko_cols = [*L.ko, *L.post_ko]
    ko = [recon[c] / _df(env, r.settlement_time if r.settlement_time is not None else r.observation_time)
          for c, r in zip(ko_cols, records) if r.barrier is not None]
    assert abs(sum(ko) - float(np.sum(stats.observations.probability))) <= ADJOINT_BOUND
    maturity_unit = recon[L.maturity_unit[0]]
    ki_unit = recon[L.ki_unit]
    assert abs(maturity_unit / _df(env, stats.maturity.time) - stats.maturity.probability) <= ADJOINT_BOUND
    assert abs(ki_unit / _df(env, stats.ki_settlement.time) - stats.ki_settlement.probability) <= ADJOINT_BOUND


# --- AC13 ----------------------------------------------------------------------------------------------------

def _stats(name, **kw):
    solver, product, env = _route(name)
    return solver, product, env, solver.calculate_event_stats(product, env, **kw)


def _horizon(product, env):
    if isinstance(product, KnockOutResetSnowballOption):
        return float(product.get_pre_maturity_time(env))
    return float(product.get_maturity(env))


@pytest.mark.parametrize("name", list(ROUTES))
def test_knock_in_table_identities(name):
    _, product, env, stats = _stats(name)
    table, ki = stats.ki_table, stats.knock_in
    assert not set(stats.unavailable) & {"knock_in.probability", "ki_table"}, dict(stats.unavailable)
    assert ki.probability is not None and ki.probability > 0.0
    assert math.isclose(float(np.sum(table.probability)), ki.probability, rel_tol=1e-12, abs_tol=1e-15)
    np.testing.assert_array_equal(table.payment_times, table.times)                       # D8
    assert table.payment_dates == table.dates
    mean = float(np.sum(table.times * table.probability)) / ki.probability
    if name in CONTINUOUS:
        assert abs(mean - ki.expected_time) <= 1.0 / float(env.bus_days_in_year) + 1e-12, (mean, ki.expected_time)
    else:
        assert math.isclose(mean, ki.expected_time, rel_tol=1e-12, abs_tol=1e-15)
        assert not np.any(table.interpolated) and table.interpolation_scheme is None
    # P(KI before any KO) sits between the knocked-in settlement (no KO after it) and one
    assert stats.ki_settlement.probability <= ki.probability + RESIDUAL_BOUND <= 1.0 + RESIDUAL_BOUND
    # survival: not yet knocked in and not knocked out after the row, monotone
    assert np.all(np.diff(table.survival) <= 1e-12)
    assert np.all(table.survival >= -1e-12)


@pytest.mark.parametrize("name", sorted(set(ROUTES) - CONTINUOUS))
def test_discrete_rows_follow_the_product_schedule(name):
    _, product, env, stats = _stats(name)
    horizon = _horizon(product, env)
    records = [r for r in product.resolve_ki_observations(env)
               if r.observation_time is not None and -1e-10 <= r.observation_time <= horizon + 1e-10]
    np.testing.assert_allclose(stats.ki_table.times, [r.observation_time for r in records], rtol=0, atol=1e-15)
    dates = tuple(r.observation_date for r in records)
    assert stats.ki_table.dates == (dates if all(d is not None for d in dates) else None)


@pytest.mark.parametrize("name", sorted(CONTINUOUS))
def test_continuous_rows_are_trading_days(name):
    """One row per trading day of the window: numeric clock 1/bus_days_in_year (no dates); a dated contract in an
    environment with a calendar: its business days. A row is ``interpolated`` exactly when a day boundary is not a
    grid time, and then the scheme is named."""
    solver, product, env = _route(name)
    p, _, _, _, stats = _adjoint(solver, product, env)
    horizon = _horizon(product, env)
    table = stats.ki_table
    if name == "dated_sb_cki_calendar":
        assert table.dates is not None and len(table.dates) == table.size
        assert all(SSE.is_business_day(d) for d in table.dates[:-1])
        assert table.times[-1] == horizon
    else:
        bus = float(env.bus_days_in_year)
        days = [k / bus for k in range(1, int(math.ceil(horizon * bus)) + 1) if k / bus < horizon - 1e-10]
        np.testing.assert_allclose(table.times, days + [horizon], rtol=0, atol=1e-15)
        assert table.dates is None
    t = np.asarray(p.layout.time.t, dtype=float)
    starts = np.r_[0.0, table.times[:-1]]
    off = lambda b: np.min(np.abs(t[None, :] - b[:, None]), axis=1) > 1e-10  # noqa: E731
    np.testing.assert_array_equal(table.interpolated, off(table.times) | off(starts))
    assert (table.interpolation_scheme is not None) == bool(np.any(table.interpolated))
    if np.any(table.interpolated):
        assert table.interpolation_scheme == "step_overlap"


@pytest.mark.parametrize("name", ["sb_cki", "sb_dki", "ph_mem_expiry", "ph_arrears_dki", "kr_abs_dki", "kr_abs_cki",
                                  "lv_sb_cki", "lv_kr_abs", "sb_reverse", "dated_sb_dki_aged", *TRUNCATED])
def test_knock_in_before_ko_is_the_settlement_of_the_twin_that_never_knocks_out_after_ki(name):
    """An independent backward number: with ``disable_ko_after_ki`` the knocked-in surface never knocks out, so its
    knocked-in settlement IS P(KI before any KO) of the original note (the not-knocked-in surface is the same march;
    a same-instant discrete tie goes to the knock-out in both; the not-knocked-in surface's knock-in-side far edge
    knocks in exactly where the price's own row does, in both). Agreement within the two marches' discount errors
    (``TWIN_BOUND``) -- on the truncated KI schedules and the narrow domains too, where the far edge absorbs mass after
    the last KI observation that matures not knocked in."""
    from dataclasses import replace

    _, product, env, stats = _stats(name)
    twin = _route(name)[1]
    twin.barrier_config = replace(twin.barrier_config, disable_ko_after_ki=True)
    twin.validate()
    solver = _route(name)[0]
    twin_stats = solver.calculate_event_stats(twin, env)
    assert abs(stats.knock_in.probability - twin_stats.ki_settlement.probability) <= TWIN_BOUND, (
        stats.knock_in.probability, twin_stats.ki_settlement.probability)
    # and the two genuinely differ from the original's own settlement wherever a knock-out follows a knock-in
    if not product.barrier_config.disable_ko_after_ki:
        assert stats.knock_in.probability - stats.ki_settlement.probability > 100.0 * RESIDUAL_BOUND


# --- CP4: the statistics describe the price's own march (D23) ----------------------------------------------------

@pytest.mark.parametrize("name", list(ROUTES))
def test_knock_in_settlement_never_exceeds_knock_in_before_ko(name):
    """Settling knocked in needs a first knock-in before any knock-out: KIS <= P(KI before KO) (within the march's
    discount residual). A knock-in-side edge row labelled knocked in where the price's row has no event breaks it
    (the truncated narrow snowball read KIS - P = +1.6e-4)."""
    _, _, _, stats = _stats(name)
    assert stats.ki_settlement.probability <= stats.knock_in.probability + RESIDUAL_BOUND, (
        stats.ki_settlement.probability, stats.knock_in.probability)


def _edge_rows(name):
    """Per march step: (time, the price's alive / knocked-in EdgeRows, the statistics rows), and the statistics."""
    solver, product, env = _route(name)
    p = prepare(solver, product, env, OutputRequest(Operation.EVENT_STATS, event_stats=True))
    out = []
    with market_scope(p.solver, p.env, p.market_fps), p.adapter.scope(p.product, p.env):
        march(p)
        st = p.statistics
        for j in range(len(st.t) - 1):
            tau_rem = float(st.tau - st.t[j])
            alive = p.solver._boundary_rows_v0(j, tau_rem, product, env, st.s)
            ki = p.solver._boundary_rows_v1(j, tau_rem, product, env, st.s)
            out.append((j, p.solver._total_tau - tau_rem, alive, ki, st.rows(j, None)))
    return p, out


@pytest.mark.parametrize("name", ["sb_trunc_dki_narrow", "sb_dki_narrow", "sb_cki_narrow", "ph_trunc_dki_narrow",
                                  "lv_sb_trunc_dki_narrow", "kr_trunc_dki_narrow", "lv_kr_trunc_dki_narrow",
                                  "kr_abs_dki_narrow", "sb_reverse"])
def test_edge_rows_settle_where_the_price_row_does(name):
    """The not-knocked-in surface's knock-in-side row settles knocked in (KIS / U_post = DF to the determination,
    its maturity cash in the knocked-in column) exactly where the price's own row is a knock-in (``copy_from_ki`` or a
    ``*_certain_ki`` row); elsewhere it has no event (U0 / U_pre = DF, cash in the not-knocked-in column). The
    knocked-in surface's knock-out-side row pays the knocked-in maturity (no knock-out term): KIS / U_post = DF."""
    p, steps = _edge_rows(name)
    st, solver, product, env = p.statistics, p.solver, p.product, p.env
    L, w = st.layout, st.n_cols
    reverse = bool(product.is_reverse)
    kr = isinstance(product, KnockOutResetSnowballOption)
    pre_maturity = float(product.get_pre_maturity_time(env)) if kr else None
    seen = set()
    for j, current, (lo0, up0), (lo1, up1), (lower, upper) in steps:
        alive_edge, alive_row = (up0, upper[:w]) if reverse else (lo0, lower[:w])
        ko1_edge, ko1_row = (lo1, lower[w:]) if reverse else (up1, upper[w:])
        knocks_in = alive_edge.copy_from_ki or alive_edge.branch.endswith("_certain_ki")
        det = solver._df_between_times(env, current, solver._total_tau)
        if kr and current > pre_maturity + 1e-12:
            continue                                     # the not-knocked-in surface is staged there
        seen.add(knocks_in)
        if kr:
            u_pre, u_post = int(L.maturity_unit[0]), int(L.maturity_unit[1])
            det_pre = solver._df_between_times(env, current, pre_maturity)
            assert alive_row[u_post] == (det if knocks_in else 0.0), (j, alive_edge.branch)
            assert alive_row[u_pre] == (0.0 if knocks_in else det_pre), (j, alive_edge.branch)
            assert ko1_row[u_post] == det, (j, ko1_edge.branch)
        else:
            assert alive_row[L.kis] == (det if knocks_in else 0.0), (j, alive_edge.branch)
            assert alive_row[int(L.maturity_unit[0])] == (0.0 if knocks_in else det), (j, alive_edge.branch)
            assert ko1_row[L.kis] == det, (j, ko1_edge.branch)
        if not alive_edge.copy_from_ki:
            cash = sum(term.value for term in alive_edge.terms if term.column == "M")
            settled, other = (L.maturity[1], L.maturity[0]) if knocks_in else (L.maturity[0], L.maturity[1])
            assert alive_row[int(settled)] == cash and alive_row[int(other)] == 0.0, (j, alive_edge.branch)
    if "trunc" in name:
        assert seen == {True, False}, seen                # knocks in up to the last KI date, then no event
    else:
        assert seen == {True}, seen                       # a KI observation (or continuous KI) up to maturity


@pytest.mark.parametrize("name", ["sb_cki", "sb_reverse", "sb_dki", "ph_mem_instant", "kr_abs_dki"])
def test_outcomes_partition_on_a_narrow_domain(name):
    """On a domain narrow enough that the knocked-in surface's knock-out-side row carries mass (num_std 2), every
    outcome still sums to one within the march's discount residual: that row pays the knocked-in maturity, so it
    settles knocked in (it leaked +1.0e-7 on sb_cki, +7.8e-7 on sb_reverse while labelled neither)."""
    solver, product, env = _route(name)
    narrow = type(solver)(PDEParams(grid=GridConfig(num_std=2.0)))
    stats = narrow.calculate_event_stats(product, env)
    assert abs(stats.probability_residual) <= RESIDUAL_BOUND, stats.probability_residual


def test_forward_pass_reproduces_the_flat_vol_surrogate_columns():
    """AC14 on the 2D solvers' ``flat_vol_surrogate`` statistics (the 1D statistics march at the strike-implied vol,
    pv from the 2D solve), which now carry the KI table too."""
    from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import HestonSnowballPDESolver
    from quantark.volmodels.heston import HestonParams

    hp = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
    product, env = grid.snowball("discrete"), grid.flat_env()
    solver = HestonSnowballPDESolver(hp, n_x=48, n_v=18, n_t=16, statistics="flat_vol_surrogate")
    pv = float(solver.price(product, env))
    p = prepare(solver, product, env, OutputRequest(Operation.EVENT_STATS, price=False, event_stats=True,
                                                    external_pv=pv))
    with market_scope(p.solver, p.env, p.market_fps), p.adapter.scope(p.product, p.env):
        result = march(p)
        forward = adjoint.forward_pass(p.statistics, result.states, result.captures, reconstruct=True)
        at = p.statistics._reader(result.states, result.captures)
    units, cash = _unit_columns(p.statistics)
    for c in units:
        assert abs(forward.reconstructed[c] - at(c)) <= ADJOINT_BOUND, (c, forward.reconstructed[c], at(c))
    for c in cash:
        assert math.isclose(forward.reconstructed[c], at(c), rel_tol=1e-12, abs_tol=1e-12), (c,)
    stats = HestonSnowballPDESolver(hp, n_x=48, n_v=18, n_t=16, statistics="flat_vol_surrogate").calculate_event_stats(
        product, env)
    assert stats.knock_in.probability > 0.0 and "knock_in.probability" not in stats.unavailable
    assert math.isclose(float(np.sum(stats.ki_table.probability)), stats.knock_in.probability, rel_tol=1e-12)


def _two_close_ki_dates():
    """KI observed on two dates one calendar day apart, the grid 0.2 steps a day (5-day steps)."""
    return SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(grid.KO_DATES), ki_barrier=90.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_dates=[0.5, 0.5 + 1.0 / 365.0]),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True))


def test_two_knock_in_dates_inside_one_step_are_both_grid_times():
    """Two KI dates can never share a time step: every KI observation is a grid time (the event times are pinned
    into the time grid), so each date takes its own row, here with a 5-day step around two dates one day apart."""
    env = grid.flat_env()
    solver = SnowballPDESolver(PDEParams(grid=GridConfig(steps_per_day=0.2)))
    p = prepare(solver, _two_close_ki_dates(), env, OutputRequest(Operation.EVENT_STATS, event_stats=True))
    t = np.asarray(p.layout.time.t, dtype=float)
    for date in (0.5, 0.5 + 1.0 / 365.0):
        assert np.min(np.abs(t - date)) <= 1e-12, date
    stats = SnowballPDESolver(PDEParams(grid=GridConfig(steps_per_day=0.2))).calculate_event_stats(
        _two_close_ki_dates(), env)
    np.testing.assert_allclose(stats.ki_table.times, [0.5, 0.5 + 1.0 / 365.0], rtol=0, atol=1e-15)
    assert np.all(stats.ki_table.probability > 0.0)


def test_first_knock_in_mass_off_the_schedule_is_refused():
    """The invariant behind the rows: first knock-in mass only at the product's KI observations. A schedule that
    disagrees with the march (here: the table asked for one of the march's two dates) is refused, never rebinned."""
    from quantark.util.exceptions import PricingError

    env = grid.flat_env()
    product = _two_close_ki_dates()
    p = prepare(SnowballPDESolver(PDEParams()), product, env, OutputRequest(Operation.EVENT_STATS, event_stats=True))
    with market_scope(p.solver, p.env, p.market_fps), p.adapter.scope(p.product, p.env):
        result = march(p)
        full = product.resolve_ki_observations(env)
        p.statistics.product.resolve_ki_observations = lambda _env: full[:1]
        with pytest.raises(PricingError, match="off the product's KI schedule"):
            adjoint.knock_in_from_forward_pass(p.statistics, result.states, result.captures)


# --- AC9 ------------------------------------------------------------------------------------------------------

def _certain_knock_out_first(first_barrier):
    """Every path knocks out at 0.25 when ``first_barrier`` is 1; the discrete KI (99, spot 100) is observed only
    afterwards (0.5, 0.75, 1.0)."""
    return SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(ko_barrier=[first_barrier, 103.0, 103.0, 103.0], ko_rate=0.15,
                                     ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(grid.KO_DATES), ki_barrier=99.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_dates=[0.5, 0.75, 1.0]),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
    )


def test_no_knock_in_after_a_certain_knock_out():
    env = grid.flat_env()
    stats = SnowballPDESolver(PDEParams()).calculate_event_stats(_certain_knock_out_first(1.0), env)
    # zero to round-off: the far lower edge before the knock-out settles knocked in on the backward columns too
    # (1.5e-15 here, the same as the KI settlement's), and the expected time is undefined
    assert abs(stats.knock_in.probability) <= 1e-12
    assert abs(stats.knock_in.probability - stats.ki_settlement.probability) <= 1e-14
    np.testing.assert_allclose(stats.ki_table.probability, np.zeros(3), rtol=0, atol=1e-12)
    np.testing.assert_allclose(stats.ki_table.times, [0.5, 0.75, 1.0], rtol=0, atol=1e-15)
    assert stats.knock_in.expected_time is None
    assert stats.unavailable["knock_in.expected_time"].startswith("undefined:")
    assert abs(stats.knock_in.expected_loss) <= 1e-10
    reachable = SnowballPDESolver(PDEParams()).calculate_event_stats(_certain_knock_out_first(103.0), env)
    assert reachable.knock_in.probability > 0.3


def test_knock_out_wins_a_same_instant_discrete_tie():
    """Discrete KI on the KO dates with the KO level below the KI level: on a shared date every node is either
    knocked out or knocked in. The knock-out wins: the KO date's first knock-in mass comes only from the nodes the
    knock-out leaves (here none above 100 and below 101), so P(KI) is the mass below 100 at the KO-free dates."""
    env = grid.flat_env()
    tie = SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(ko_barrier=100.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=[0.5], ki_barrier=101.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_dates=[0.5, 1.0]),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
    )
    stats = SnowballPDESolver(PDEParams(event_projection="nodal")).calculate_event_stats(tie, env)
    # at 0.5 every path is knocked out (>= 100) or knocked in (< 101): the overlap [100, 101) knocks out
    first = stats.ki_table.probability[0]
    ko = stats.observations.probability[0]
    # two unit columns, each off by the march's discount error (O(dt^2); 1.5e-8 measured here)
    assert abs(first + ko - 1.0) <= 2.0 * RESIDUAL_BOUND, (first, ko)
    assert stats.ki_table.probability[1] <= 1e-12          # nothing is left alive after 0.5


# --- valuation-date rules: no forward pass ----------------------------------------------------------------------

@pytest.fixture
def no_forward_pass(monkeypatch):
    calls = []

    def refuse(*args, **kwargs):
        calls.append(args)
        raise AssertionError("the forward adjoint pass ran")

    monkeypatch.setattr(adjoint, "forward_pass", refuse)
    return calls


@pytest.mark.parametrize("case", ["ko_at_valuation", "carried", "ki_now", "no_barrier"])
def test_definitional_knock_in_runs_no_forward_pass(case, no_forward_pass):
    solver = SnowballPDESolver(PDEParams())
    if case == "ko_at_valuation":
        stats = solver.calculate_event_stats(grid.snowball(ko_dates=(0.0, *grid.KO_DATES)), grid.flat_env(110.0))
        assert stats.knock_in.probability == 0.0 and stats.ki_table.size == 0
    elif case == "carried":
        stats = solver.calculate_event_stats(grid.snowball(knocked_in=True), grid.flat_env())
        assert stats.knock_in.probability == 1.0 and stats.ki_table.times.tolist() == [0.0]
    elif case == "ki_now":
        stats = solver.calculate_event_stats(grid.snowball(), grid.flat_env(70.0))
        assert stats.knock_in.probability == 1.0 and stats.knock_in.expected_time == 0.0
    else:
        product = SnowballOption(
            initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0, is_reverse=False,
            barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                         ko_observation_dates=list(grid.KO_DATES)),
            payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True))
        stats = solver.calculate_event_stats(product, grid.flat_env())
        assert stats.knock_in.probability == 0.0 and stats.ki_table.size == 0
    assert no_forward_pass == []


# --- D17: a pruned request skips the pass -----------------------------------------------------------------------

@pytest.mark.parametrize("name", ["sb_cki", "ph_mem_expiry", "kr_abs_dki", "lv_sb_dki"])
def test_pruned_request_never_runs_the_forward_pass(name, monkeypatch):
    solver, product, env = _route(name)
    full = solver.calculate_event_stats(product, env)
    calls = []
    real = adjoint.forward_pass

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(adjoint, "forward_pass", spy)
    for streams in (frozenset({EventType.KO}), frozenset({EventType.KO, EventType.COUPON, EventType.MATURITY_NO_KO,
                                                          EventType.MATURITY_WITH_KI})):
        pruned = _route(name)[0].calculate_event_stats(product, env, streams=streams)
        assert calls == []
        assert pruned.knock_in.probability is None and pruned.ki_table is None
        assert pruned.unavailable["knock_in.probability"] == "not requested"
        np.testing.assert_array_equal(pruned.observations.probability, full.observations.probability)
        assert pruned.maturity == full.maturity and pruned.ki_settlement == full.ki_settlement
    _route(name)[0].calculate_event_stats(product, env, streams=frozenset({EventType.KO, EventType.KI}))
    assert len(calls) == 1
    _route(name)[0].price_with_events(product, env)
    assert len(calls) == 2


def test_price_with_events_npv_stays_price_bitwise():
    for name in ("sb_cki", "ph_mem_expiry", "kr_abs_dki", "lv_sb_cki"):
        solver, product, env = _route(name)
        assert _route(name)[0].price_with_events(product, env).npv == solver.price(product, env)


def test_the_knock_in_stream_reaches_the_event_distribution():
    """The KI stream on its own axis (DESIGN §4.11): the table's rows, unflagged."""
    solver, product, env = _route("sb_dki")
    stats = solver.calculate_event_stats(product, env)
    dist = _route("sb_dki")[0].price_with_events(product, env).event_distribution
    np.testing.assert_array_equal(dist.stream_times[EventType.KI], stats.ki_table.times)
    np.testing.assert_array_equal(dist.probabilities[EventType.KI], stats.ki_table.probability)
    assert EventType.KI not in dist.flagged_streams


# --- the MC reference ------------------------------------------------------------------------------------------

#: name -> (PDE solver class, MC engine class, product factory)
MC_CASES = {
    "sb_cki": (SnowballPDESolver, SnowballMCEngine, grid.snowball),
    "sb_dki": (SnowballPDESolver, SnowballMCEngine, lambda: grid.snowball("discrete")),
    "ph_mem_expiry": (PhoenixPDESolver, PhoenixMCEngine, lambda: grid.phoenix(pay=EXPIRY)),
    "ph_arrears_dki": (PhoenixPDESolver, PhoenixMCEngine,
                       lambda: grid.phoenix(ki="discrete", arrears=0.02, coupon_barriers=[106.0, 106.0, 80.0, 80.0])),
    "kr_abs_dki": (KOResetSnowballPDESolver, SnowballMCEngine, grid.ko_reset),
    "kr_abs_cki": (KOResetSnowballPDESolver, SnowballMCEngine, lambda: grid.ko_reset(ki="continuous")),
}
FINE = PDEParams(grid=GridConfig(points=1600, steps_per_day=16.0))
#: the MC reference's paths (16 independent RQMC scrambles of 4096)
MC_PATHS = 16 * 4096


def _mc(mc_cls, product, env):
    params = MCParams(num_paths=4096, seed=11, rqmc_min_batches=16, rqmc_max_batches=16, rqmc_target_std=1e-12,
                      rqmc_paths_mode="per_batch", time_steps=1008)
    return mc_cls(params=params, method=MonteCarloMethod.RANDOMIZED_QUASI).calculate_event_stats(product, env)


def _by_week(values):
    """Continuous rows summed by trading week (a day carries ~1e-3 of mass; a week enough paths for a test)."""
    values = np.asarray(values, dtype=float)
    return np.add.reduceat(values, np.arange(0, values.size, 5))


@pytest.mark.parametrize("name", list(MC_CASES))
def test_knock_in_statistics_agree_with_the_mc_reference(name):
    """P(KI before KO), E[KI time] and the per-date table within 4 MC batch SEs plus the PDE's own discretisation
    allowance, |PDE(default) - PDE(points 1600, 16 steps a day)|. Continuous rows are compared by trading week."""
    pde_cls, mc_cls, make = MC_CASES[name]
    env = grid.flat_env()
    pde = pde_cls(PDEParams()).calculate_event_stats(make(), env)
    fine = pde_cls(FINE).calculate_event_stats(make(), env)
    mc = _mc(mc_cls, make(), env)
    se = mc.standard_errors
    for path in ("knock_in.probability", "knock_in.expected_time"):
        a, b, f = pde.require(path), mc.require(path), fine.require(path)
        assert abs(a - b) <= 4.0 * se[path] + abs(a - f) + 1e-12, (path, a, b, se[path], abs(a - f))
    np.testing.assert_allclose(pde.ki_table.times, mc.ki_table.times, rtol=0, atol=1e-12)
    a, b, f, s = (pde.ki_table.probability, mc.ki_table.probability, fine.ki_table.probability,
                  np.asarray(se["ki_table.probability"], dtype=float))
    if pde.ki_table.size > 20:                        # continuous: by trading week (rows nearly uncorrelated)
        a, b, f = (_by_week(x) for x in (a, b, f))
        s = np.sqrt(_by_week(s * s))
    # a row no MC path reached has a batch SE of 0: floor the SE at the binomial one of the PDE's own probability
    s = np.maximum(s, np.sqrt(np.clip(a, 0.0, 1.0) * (1.0 - np.clip(a, 0.0, 1.0)) / MC_PATHS))
    assert np.all(np.abs(a - b) <= 4.0 * s + np.abs(a - f) + 1e-12), (name, np.max(np.abs(a - b) - 4.0 * s
                                                                                  - np.abs(a - f)))


@pytest.mark.parametrize("name", ["sb_dki", "kr_abs_dki"])
def test_knock_in_probability_converges_under_time_refinement(name):
    """The refinement allowance shrinks: on a fixed spatial grid, |P(KI)_dt - P(KI)_{dt/2}| falls by about 4 per
    halving (second order; measured at 1600 points: 4.6e-7, 1.2e-7, 2.9e-8, 7.3e-9 on sb_dki). Continuous KI moves
    by ~5e-5 without a clean order (the FIRST_PASSAGE correction's accuracy, the same in the backward columns): its
    allowance is measured against the fine grid in the MC test."""
    pde_cls, _, make = MC_CASES[name]
    env = grid.flat_env()
    values = [pde_cls(PDEParams(grid=GridConfig(points=800, steps_per_day=steps))).calculate_event_stats(
        make(), env).knock_in.probability for steps in (2.0, 4.0, 8.0, 16.0)]
    gaps = np.abs(np.diff(values))
    assert gaps[0] > gaps[1] > gaps[2] > 0.0, (values, gaps)
    for coarse, fine in zip(gaps, gaps[1:]):
        assert 3.0 < coarse / fine < 5.0, (values, gaps)
