"""PDE 1D (flat and local vol) event statistics on the unified ``EventStatistics`` (spec T12).

Snowball, Phoenix and KO-reset ABSOLUTE (REBASED KO-reset is MC-only: the PDE refuses it, DESIGN §4.9a):

- AC7: ``EventStatistics`` with nothing unavailable but the knock-in path statistics the forward adjoint pass
  produces (spec T13: ``knock_in.probability``, ``knock_in.expected_time``, ``ki_table``, each with the T13
  reason) and, where P(knock in) = 0 by the valuation-date rules, the undefined expected time;
- D23: ``pv`` is ``price()``'s value bitwise, and so is ``price_with_events(...).npv``;
- AC8 (D16): ``|reconciliation_error|`` at round-off; ``|probability_residual|`` within the banked bound
  (measured, see ``RESIDUAL_BOUND``), shrinking under refinement (the march's discount error, O(dt^2));
- AC9: a knock-in after a certain knock-out settles nothing knocked in and loses nothing;
- AC10: valuation-date rules (KO at valuation, carried knock-in, knock-in at t = 0, no KI barrier);
- AC11: a coupon paid with a knock-out is coupon cash; Snowball and KO-reset carry no coupon table;
- AC12: cash to pay is ``discounted / DF(payment)`` on deterministic rows and the directly marched coupon earned on
  path-dependent EXPIRY rows (memory: its own undiscounted column), agreeing with the INSTANT twin and with MC;
- pruning (DESIGN §4.9b): the termination outcomes are never pruned, bit for bit;
- the MC reference (RQMC, independent batches): every number within 4 batch SEs plus the refinement allowance
  measured here (|PDE(default) - PDE(refined grid)|).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from quantark.asset.equity.engine.event_stats import EventStatistics
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde import GridConfig
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.ko_reset_vol_pde_solvers import LocalVolKOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import LocalVolPhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import LocalVolSnowballPDESolver
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.cashleg import EventType
from quantark.util.enum import CouponPayType
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError

from test_asset.equity.engine_mc import mc_price_path_routes as grid

warnings.filterwarnings("ignore", category=DeprecationWarning)

EXPIRY = CouponPayType.EXPIRY

#: the knock-in path statistics the forward adjoint pass (spec T13) produces
PENDING_KEYS = {"knock_in.probability", "knock_in.expected_time", "ki_table"}
#: |reconciliation_error| / max(1, |pv|): the value and the cash columns take the same linear stages (round-off)
RECONCILIATION_BOUND = 1e-12
#: |probability_residual| at the accuracy presets (steps_per_day >= 4): the march's discount error on the unit
#: columns, O(dt^2). Measured (T12, flat + LV, this matrix): max 3.3e-9 at the default accuracy; 3.2e-8 / 8.1e-9 /
#: 2.0e-9 at 1 / 2 / 4 steps a day (factor 4 per halving, ``test_probability_residual_shrinks_under_refinement``).
RESIDUAL_BOUND = 1e-8


def _flat(cls):
    return lambda: cls(PDEParams())


def _lv(cls):
    return lambda: cls(params=PDEParams())


#: name -> (solver factory, product factory, env factory, family)
ROUTES = {
    "sb_cki": (_flat(SnowballPDESolver), grid.snowball, grid.flat_env, "snowball"),
    "sb_dki": (_flat(SnowballPDESolver), lambda: grid.snowball("discrete"), grid.flat_env, "snowball"),
    "sb_reverse": (_flat(SnowballPDESolver), lambda: grid.snowball(reverse=True), grid.flat_env, "snowball"),
    "sb_no_ko_after_ki": (_flat(SnowballPDESolver), lambda: grid.snowball(disable_ko_after_ki=True),
                          grid.flat_env, "snowball"),
    "sb_expiry": (_flat(SnowballPDESolver), lambda: grid.snowball(pay=EXPIRY), grid.flat_env, "snowball"),
    "ph_mem_instant": (_flat(PhoenixPDESolver), grid.phoenix, grid.flat_env, "phoenix"),
    "ph_nomem_instant": (_flat(PhoenixPDESolver), lambda: grid.phoenix(memory=False), grid.flat_env, "phoenix"),
    "ph_mem_expiry": (_flat(PhoenixPDESolver), lambda: grid.phoenix(pay=EXPIRY), grid.flat_env, "phoenix"),
    "ph_nomem_expiry": (_flat(PhoenixPDESolver), lambda: grid.phoenix(memory=False, pay=EXPIRY), grid.flat_env,
                        "phoenix"),
    "ph_arrears_dki": (_flat(PhoenixPDESolver), lambda: grid.phoenix(ki="discrete", arrears=0.02,
                                                                     coupon_barriers=[106.0, 106.0, 80.0, 80.0]),
                       grid.flat_env, "phoenix"),
    "kr_abs_dki": (_flat(KOResetSnowballPDESolver), grid.ko_reset, grid.flat_env, "ko_reset"),
    "kr_abs_cki": (_flat(KOResetSnowballPDESolver), lambda: grid.ko_reset(ki="continuous"), grid.flat_env,
                   "ko_reset"),
    "lv_sb_cki": (_lv(LocalVolSnowballPDESolver), grid.snowball, grid.grid_env, "snowball"),
    "lv_sb_dki": (_lv(LocalVolSnowballPDESolver), lambda: grid.snowball("discrete"), grid.grid_env, "snowball"),
    "lv_ph_mem_expiry": (_lv(LocalVolPhoenixPDESolver), lambda: grid.phoenix(pay=EXPIRY), grid.grid_env, "phoenix"),
    "lv_ph_nomem_instant": (_lv(LocalVolPhoenixPDESolver), lambda: grid.phoenix(memory=False), grid.grid_env,
                            "phoenix"),
    "lv_kr_abs": (_lv(LocalVolKOResetSnowballPDESolver), grid.ko_reset, grid.grid_env, "ko_reset"),
}
LV = [name for name in ROUTES if name.startswith("lv_")]


def _route(name):
    make_solver, make_product, make_env, _ = ROUTES[name]
    return make_solver(), make_product(), make_env()


def _stats(name, **kw):
    solver, product, env = _route(name)
    return solver, product, env, solver.calculate_event_stats(product, env, **kw)


def _df(env, t):
    return float(env.get_discount_factor(float(t)))


# --- AC7 -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(ROUTES))
def test_full_request_is_an_event_statistics_pending_only_the_forward_pass(name):
    solver, _, _, stats = _stats(name)
    assert isinstance(stats, EventStatistics)
    assert stats.product_family == ROUTES[name][3]
    assert (stats.coupons is None) == (ROUTES[name][3] != "phoenix")           # AC11
    assert set(stats.unavailable) == PENDING_KEYS, dict(stats.unavailable)
    assert all("T13" in reason and "forward adjoint pass" in reason for reason in stats.unavailable.values())
    assert stats.knock_in.probability is None and stats.knock_in.expected_time is None and stats.ki_table is None
    assert stats.knock_in.expected_loss is not None and stats.knock_in.expected_loss >= 0.0
    assert stats.ledger.cash_to_pay is not None
    assert stats.coupons is None or stats.coupons.probability is not None
    assert stats.standard_errors is None
    model = "local_vol_1d" if name in LV else "flat_vol_1d"
    assert stats.provenance.statistics_model == stats.provenance.price_model == model


# --- D23: pv is price() ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(ROUTES))
def test_pv_is_price_bitwise(name):
    solver, product, env = _route(name)
    price = solver.price(product, env)
    assert solver.calculate_event_stats(product, env).pv == price
    result = _route(name)[0].price_with_events(product, env)
    assert result.npv == price


# --- AC8 -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(ROUTES))
def test_partition_and_reconciliation_within_the_banked_bounds(name):
    _, _, _, stats = _stats(name)
    assert abs(stats.reconciliation_error) <= RECONCILIATION_BOUND * max(1.0, abs(stats.pv)), stats.reconciliation_error
    assert stats.probability_residual != 0.0                    # computed, never forced
    assert abs(stats.probability_residual) <= RESIDUAL_BOUND, stats.probability_residual
    assert stats.knock_in.probability_no_ko == stats.ki_settlement.probability
    assert math.isclose(stats.ledger.total_discounted_cash, stats.total_discounted_cash, rel_tol=1e-12)


@pytest.mark.parametrize("name", ["sb_cki", "sb_dki", "ph_mem_expiry", "kr_abs_dki", "lv_sb_cki", "lv_kr_abs"])
def test_probability_residual_shrinks_under_refinement(name):
    """D16: the residual is the march's discount error on the unit columns, second order in the time step: it
    falls by about 4 per halving of the step (and the space step with it)."""
    make_solver, make_product, make_env, _ = ROUTES[name]
    cls = type(make_solver())
    residuals = []
    for points, steps in ((400, 1.0), (800, 2.0), (1600, 4.0)):
        params = PDEParams(grid=GridConfig(points=points, steps_per_day=steps))
        solver = cls(params=params) if name in LV else cls(params)
        residuals.append(abs(solver.calculate_event_stats(make_product(), make_env()).probability_residual))
    assert residuals[0] > residuals[1] > residuals[2] > 0.0, residuals
    for coarse, fine in zip(residuals, residuals[1:]):
        assert 3.5 < coarse / fine < 4.5, residuals


# --- AC9 -------------------------------------------------------------------------------------------------------

def _certain_knock_out_first(first_barrier):
    """Every path knocks out at 0.25 when ``first_barrier`` is 1; the discrete KI (99, spot 100) is observed only
    afterwards (0.5, 0.75, 1.0)."""
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
    from quantark.asset.equity.product.option.snowball_option import SnowballOption
    from quantark.util.enum import ObservationType

    return SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(ko_barrier=[first_barrier, 103.0, 103.0, 103.0], ko_rate=0.15,
                                     ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(grid.KO_DATES), ki_barrier=99.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_dates=[0.5, 0.75, 1.0]),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
    )


def test_a_knock_in_after_a_certain_knock_out_settles_nothing_knocked_in():
    """AC9 on the backward columns: nothing settles knocked in and nothing is lost when the knock-out is certain
    before the barrier is observed, while the barrier is reachable (the same note with an ordinary first barrier
    settles knocked in on a large share of paths). P(knock in) itself is the forward pass's (T13)."""
    env = grid.flat_env()
    stats = SnowballPDESolver(PDEParams()).calculate_event_stats(_certain_knock_out_first(1.0), env)
    assert stats.observations.probability[0] > 1.0 - 1e-9
    assert abs(stats.ki_settlement.probability) <= 1e-12
    assert abs(stats.ki_settlement.discounted_cash) <= 1e-10
    assert abs(stats.knock_in.expected_loss) <= 1e-10
    assert stats.unavailable["knock_in.probability"] == stats.unavailable["ki_table"]
    reachable = SnowballPDESolver(PDEParams()).calculate_event_stats(_certain_knock_out_first(103.0), env)
    assert reachable.ki_settlement.probability > 0.2 and reachable.knock_in.expected_loss > 1.0


# --- AC10 ------------------------------------------------------------------------------------------------------

def _at_valuation(product_kw=None, *, cls=SnowballPDESolver, make=grid.snowball, spot=110.0, lv=False):
    product = make(**dict(product_kw or {}))
    env = grid.grid_env(spot) if lv else grid.flat_env(spot)
    solver = cls(params=PDEParams()) if lv else cls(PDEParams())
    return solver, product, env, solver.calculate_event_stats(product, env)


def _assert_no_knock_in(stats):
    assert stats.knock_in.probability == 0.0 and stats.knock_in.expected_loss == 0.0
    assert stats.ki_table is not None and stats.ki_table.size == 0
    assert stats.knock_in.expected_time is None
    assert stats.unavailable["knock_in.expected_time"].startswith("undefined:")
    assert set(stats.unavailable) == {"knock_in.expected_time"}


def _assert_knock_in_at_zero(stats):
    assert stats.knock_in.probability == 1.0 and stats.knock_in.expected_time == 0.0
    np.testing.assert_array_equal(stats.ki_table.times, [0.0])
    np.testing.assert_array_equal(stats.ki_table.probability, [1.0])
    assert not np.any(stats.ki_table.interpolated)
    assert dict(stats.unavailable) == {}


@pytest.mark.parametrize("family", ["snowball", "phoenix", "phoenix_expiry", "lv_snowball"])
def test_knock_out_at_valuation_has_no_knock_in(family):
    kw = dict(ko_dates=(0.0, *grid.KO_DATES))
    if family.startswith("phoenix"):
        if family == "phoenix_expiry":
            kw["pay"] = EXPIRY
        solver, product, env, stats = _at_valuation(kw, cls=PhoenixPDESolver, make=grid.phoenix)
    elif family == "lv_snowball":
        solver, product, env, stats = _at_valuation(kw, cls=LocalVolSnowballPDESolver, lv=True)
    else:
        solver, product, env, stats = _at_valuation(kw)
    assert stats.pv == solver.price(product, env)
    assert stats.observations.times[0] == 0.0 and stats.observations.probability[0] == 1.0
    assert np.all(stats.observations.probability[1:] == 0.0)
    assert stats.maturity.probability == 0.0 and stats.ki_settlement.probability == 0.0
    assert stats.expected_lifetime == 0.0
    assert stats.reconciliation_error == 0.0 and stats.probability_residual == 0.0
    _assert_no_knock_in(stats)


def test_ko_reset_knock_out_at_valuation_has_no_knock_in():
    from dataclasses import replace

    product = grid.ko_reset()
    product.barrier_config = replace(product.barrier_config, ko_observation_schedule=ObservationSchedule(
        records=[ObservationRecord(observation_time=t, barrier=103.0) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]))
    product.validate()
    solver, env = KOResetSnowballPDESolver(PDEParams()), grid.flat_env(110.0)
    stats = solver.calculate_event_stats(product, env)
    assert stats.pv == solver.price(product, env)
    assert stats.observations.schedule[0] == "pre" and stats.observations.probability[0] == 1.0
    assert stats.reconciliation_error == 0.0
    _assert_no_knock_in(stats)


@pytest.mark.parametrize("name", ["snowball", "snowball_no_ko_after_ki", "phoenix", "ko_reset", "lv_snowball"])
def test_carried_knock_in_is_a_knock_in_at_zero(name):
    if name == "phoenix":
        solver, product, env, stats = _at_valuation({"knocked_in": True}, cls=PhoenixPDESolver, make=grid.phoenix,
                                                    spot=100.0)
    elif name == "ko_reset":
        solver, product, env, stats = _at_valuation({"knocked_in": True}, cls=KOResetSnowballPDESolver,
                                                    make=grid.ko_reset, spot=100.0)
    elif name == "lv_snowball":
        solver, product, env, stats = _at_valuation({"knocked_in": True}, cls=LocalVolSnowballPDESolver, spot=100.0,
                                                    lv=True)
    else:
        kw = {"knocked_in": True, "disable_ko_after_ki": name.endswith("after_ki")}
        solver, product, env, stats = _at_valuation(kw, spot=100.0)
    _assert_knock_in_at_zero(stats)
    assert stats.maturity.probability == 0.0                     # a knocked-in note never matures not knocked in
    assert stats.ki_settlement.probability > 0.0
    assert abs(stats.probability_residual) <= RESIDUAL_BOUND


def test_carried_knock_in_knocked_out_at_valuation_keeps_its_knock_in():
    solver, product, env, stats = _at_valuation({"ko_dates": (0.0, *grid.KO_DATES), "knocked_in": True})
    assert stats.observations.probability[0] == 1.0
    _assert_knock_in_at_zero(stats)
    assert stats.ki_settlement.probability == 0.0 and stats.knock_in.expected_loss == 0.0


@pytest.mark.parametrize("ki", ["continuous", "discrete"])
def test_knock_in_observed_at_valuation_is_a_knock_in_at_zero(ki):
    """Continuous: the spot (70) is through the barrier (75); discrete: a KI observation at t = 0 breached."""
    kw = {} if ki == "continuous" else {"ko_dates": (0.0, *grid.KO_DATES)}
    _, _, _, stats = _at_valuation(dict(kw, ki=ki), spot=70.0)
    _assert_knock_in_at_zero(stats)
    assert stats.maturity.probability == 0.0


def test_no_knock_in_barrier_reports_computed_zeros():
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
    from quantark.asset.equity.product.option.snowball_option import SnowballOption
    from quantark.util.enum import ObservationType

    product = SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(grid.KO_DATES)),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
    )
    assert not product.has_ki_barrier
    stats = SnowballPDESolver(PDEParams()).calculate_event_stats(product, grid.flat_env())
    _assert_no_knock_in(stats)
    assert stats.ki_settlement.probability == 0.0 and stats.ki_settlement.discounted_cash == 0.0
    assert abs(stats.probability_residual) <= RESIDUAL_BOUND


# --- AC11 ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["ph_mem_instant", "ph_mem_expiry", "ph_arrears_dki", "lv_ph_mem_expiry"])
def test_a_coupon_paid_with_a_knock_out_is_coupon_cash(name):
    """The knock-out row pays the redemption only: its cash is P x DF(settlement) x the record's redemption
    (principal + ko_rate accrual), and the coupon decided with it sits in the coupon table."""
    _, product, env, stats = _stats(name)
    obs, coupons = stats.observations, stats.coupons
    records = product.resolve_ko_observations(env)
    redemption = np.array([float(r.payoff) for r in records])
    expected = obs.probability * np.array([_df(env, t) for t in obs.payment_times]) * redemption
    np.testing.assert_allclose(obs.discounted_cash, expected, rtol=1e-12, atol=1e-12)
    assert np.all(obs.is_coupon_obs) and np.all(obs.is_ko_obs)
    knocked_out_with_coupon = (obs.probability > 1e-3) & (coupons.discounted_cash > 1e-6)
    assert knocked_out_with_coupon.any()


@pytest.mark.parametrize("name", ["sb_cki", "sb_expiry", "kr_abs_dki", "lv_kr_abs"])
def test_snowball_and_ko_reset_carry_no_coupon_table(name):
    _, _, _, stats = _stats(name)
    assert stats.coupons is None


# --- AC12 ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(ROUTES))
def test_deterministic_payment_rows_are_df_consistent(name):
    _, _, env, stats = _stats(name)
    obs = stats.observations
    df = np.array([_df(env, t) for t in obs.payment_times])
    np.testing.assert_allclose(obs.cash_to_pay, obs.discounted_cash / df, rtol=1e-12, atol=1e-14)
    for row in (stats.maturity, stats.ki_settlement):
        assert math.isclose(row.cash_to_pay, row.discounted_cash / _df(env, row.payment_time), rel_tol=1e-12,
                            abs_tol=1e-14)
    coupons = stats.coupons
    if coupons is not None and not np.any(coupons.payment_is_path_dependent):
        df = np.array([_df(env, t) for t in coupons.payment_times])
        np.testing.assert_allclose(coupons.cash_to_pay, coupons.discounted_cash / df, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("memory", [True, False])
def test_expiry_coupon_cash_to_pay_is_the_coupon_earned(memory):
    """EXPIRY rows are path-dependent; their cash to pay is the coupon earned at the row (memory: the marched
    undiscounted carrier; without memory the amount times its probability), which does not depend on when it is
    paid: it equals the INSTANT twin's (``discounted / DF(settlement)``, a different column) to the grid's
    accuracy, while the discounted cash differs (paid later)."""
    env = grid.flat_env()
    expiry = PhoenixPDESolver(PDEParams()).calculate_event_stats(grid.phoenix(memory=memory, pay=EXPIRY), env)
    instant = PhoenixPDESolver(PDEParams()).calculate_event_stats(grid.phoenix(memory=memory), env)
    assert np.all(expiry.coupons.payment_is_path_dependent) and not np.any(instant.coupons.payment_is_path_dependent)
    np.testing.assert_allclose(expiry.coupons.cash_to_pay, instant.coupons.cash_to_pay, rtol=0.0, atol=2e-6)
    # paid later than its INSTANT twin, except the maturity row (both settle at the terminal payment)
    assert np.all(expiry.coupons.discounted_cash[:-1] < instant.coupons.discounted_cash[:-1])
    assert math.isclose(expiry.coupons.discounted_cash[-1], instant.coupons.discounted_cash[-1], rel_tol=1e-9)
    # the ledger buckets EXPIRY coupons at their actual payment events: discounted sums agree to round-off
    pieces = expiry.ledger.discounted_cash
    assert math.isclose(float(np.sum(pieces)), expiry.total_discounted_cash, rel_tol=1e-12)


def test_memory_expiry_coupon_rows_follow_the_arrears():
    """Memory arrears raise the coupon earned above amount x P(coupon) on rows after a missed coupon: the
    undiscounted carrier sees them (without memory the two are equal)."""
    env = grid.flat_env()
    make = lambda memory: grid.phoenix(memory=memory, pay=EXPIRY, coupon_barriers=[106.0, 106.0, 80.0, 80.0])
    with_memory = PhoenixPDESolver(PDEParams()).calculate_event_stats(make(True), env)
    without = PhoenixPDESolver(PDEParams()).calculate_event_stats(make(False), env)
    np.testing.assert_allclose(with_memory.coupons.probability, without.coupons.probability, rtol=0, atol=1e-12)
    assert np.all(with_memory.coupons.cash_to_pay[2:] > without.coupons.cash_to_pay[2:] + 1e-4)


# --- pruning ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["sb_cki", "ph_mem_expiry", "kr_abs_dki", "lv_ph_mem_expiry"])
def test_pruned_request_keeps_every_termination_outcome_bit_for_bit(name):
    _, _, _, full = _stats(name)
    _, _, _, pruned = _stats(name, streams=frozenset({EventType.KO}))
    for path in ("observations.probability", "observations.discounted_cash", "observations.cash_to_pay"):
        np.testing.assert_array_equal(pruned.require(path), full.require(path))
    assert pruned.maturity == full.maturity and pruned.ki_settlement == full.ki_settlement
    assert pruned.knock_in.expected_loss == full.knock_in.expected_loss
    assert pruned.knock_in.probability is None and pruned.unavailable["knock_in.probability"] == "not requested"
    if full.coupons is not None:
        assert pruned.coupons.probability is None and pruned.unavailable["coupons.probability"] == "not requested"
        np.testing.assert_array_equal(pruned.coupons.discounted_cash, full.coupons.discounted_cash)
        np.testing.assert_array_equal(pruned.coupons.cash_to_pay, full.coupons.cash_to_pay)
    np.testing.assert_array_equal(pruned.ledger.discounted_cash, full.ledger.discounted_cash)


def test_rebased_ko_reset_stays_refused():
    """DESIGN §4.9a: REBASED KO-reset is MC-only; the PDE refuses it (no statistics either)."""
    from quantark.util.enum.option_enums import PostKOScheduleMode

    product = grid.ko_reset(PostKOScheduleMode.REBASED)
    with pytest.raises(ValidationError):
        KOResetSnowballPDESolver(PDEParams()).price(product, grid.flat_env())
    with pytest.raises(ValidationError):
        KOResetSnowballPDESolver(PDEParams()).calculate_event_stats(product, grid.flat_env())


# --- the MC reference -------------------------------------------------------------------------------------------

MC_FIELDS = ("observations.probability", "observations.discounted_cash", "observations.cash_to_pay",
             "maturity.probability", "maturity.discounted_cash", "maturity.cash_to_pay",
             "ki_settlement.probability", "ki_settlement.discounted_cash", "ki_settlement.cash_to_pay",
             "knock_in.expected_loss", "expected_lifetime", "coupons.probability", "coupons.discounted_cash",
             "coupons.cash_to_pay")
#: name -> (PDE solver class, MC engine class, product factory)
MC_CASES = {
    "sb_cki": (SnowballPDESolver, SnowballMCEngine, grid.snowball),
    "ph_mem_expiry": (PhoenixPDESolver, PhoenixMCEngine, lambda: grid.phoenix(pay=EXPIRY)),
    "ph_arrears_dki": (PhoenixPDESolver, PhoenixMCEngine,
                       lambda: grid.phoenix(ki="discrete", arrears=0.02, coupon_barriers=[106.0, 106.0, 80.0, 80.0])),
    "kr_abs_dki": (KOResetSnowballPDESolver, SnowballMCEngine, grid.ko_reset),
}


@pytest.mark.parametrize("name", list(MC_CASES))
def test_pde_agrees_with_the_mc_reference(name):
    """Every PDE number within 4 MC batch SEs (RQMC, 16 independent scrambles x 4096 paths, as ``price()``
    runs them) plus the PDE's own discretisation allowance, measured here as |PDE(default grid) - PDE(points 1600,
    16 steps a day)|."""
    pde_cls, mc_cls, make = MC_CASES[name]
    env = grid.flat_env()
    pde = pde_cls(PDEParams()).calculate_event_stats(make(), env)
    fine = pde_cls(PDEParams(grid=GridConfig(points=1600, steps_per_day=16.0))).calculate_event_stats(make(), env)
    params = MCParams(num_paths=4096, seed=11, rqmc_min_batches=16, rqmc_max_batches=16, rqmc_target_std=1e-12,
                      rqmc_paths_mode="per_batch", time_steps=1008)
    mc = mc_cls(params=params, method=MonteCarloMethod.RANDOMIZED_QUASI).calculate_event_stats(make(), env)
    checked = 0
    for path in MC_FIELDS:
        a, b, f = pde.require(path), mc.require(path), fine.require(path)
        if a is None:
            continue
        se = np.atleast_1d(mc.standard_errors[path])
        a, b, f = np.atleast_1d(a), np.atleast_1d(b), np.atleast_1d(f)
        allowance = np.abs(a - f)
        assert np.all(np.abs(a - b) <= 4.0 * se + allowance + 1e-12), (path, a, b, se, allowance)
        checked += 1
    assert checked >= 11
