"""Native Heston / SLV ADI statistics on ``EventStatistics`` (unified event statistics spec T14).

The ADI march carries the PDE 1D family's statistics columns (T12) on the (log-spot, variance) mesh; its forward
adjoint pass (``pde/vol_model_adjoint.py``) is the exact transpose of the same march -- the batched ADI column step
(Douglas / Craig-Sneyd, damped and terminal half steps), the Dirichlet spot rows, the family's event stages on every
variance column and FIRST_PASSAGE with each column's own coefficients -- read from the readout's weights at
``(spot, v0)``:

- AC7: a full request fills every number (``unavailable`` empty but an undefined expected knock-in time);
- AC8: ``probability_residual`` / ``reconciliation_error`` within the ADI's measured bound, shrinking under refinement;
  the explicit ``flat_vol_surrogate`` opt-out reports its 2D - 1D price gap as its reconciliation error (DESIGN §4.9a:
  outside the identity guarantees);
- AC9: no knock-in after a knock-out (the KO-independent "KI ever" monitor is gone); KO wins a same-instant tie;
- AC10: the valuation-date rules (KO at valuation, carried KI, KI at t = 0, no KI barrier) need no forward pass;
- AC13: Σ rows = P(KI), Σ t p / P(KI) = E[KI time]; rows on the product's KI schedule or trading days;
- AC14: the forward pass reproduces every backward column (termination probabilities <= 1e-10 abs);
- D17: a request without the KI stream never runs the pass; D23: ``pv`` is ``price()`` bit for bit;
- the MC reference (Heston / SLV MC, independent RQMC batches) within 4 batch SEs plus the ADI's refinement allowance.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import replace

import numpy as np
import pytest

from quantark.asset.equity.engine.event_stats import EventStatistics, StatisticsProvenance
from quantark.asset.equity.engine.pde import (
    HestonKOResetSnowballPDESolver,
    HestonPhoenixPDESolver,
    HestonSLVKOResetSnowballPDESolver,
    HestonSLVPhoenixPDESolver,
    HestonSLVSnowballPDESolver,
    HestonSnowballPDESolver,
)
from quantark.asset.equity.engine.pde import vol_model_adjoint, vol_model_statistics
from quantark.asset.equity.engine.pde.adi_columns import step_columns
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.cashleg import EventType
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.numerical import Tolerance
from quantark.volmodels.heston import HestonParams

from pde_statistics.test_pde_forward_adjoint import AGED, dated_env, dated_ko_reset, dated_phoenix, dated_snowball
from test_asset.equity.engine_mc import mc_price_path_routes as grid

warnings.filterwarnings("ignore", category=DeprecationWarning)

EXPIRY = CouponPayType.EXPIRY
HP = grid.HESTON
#: the test mesh (the existing ADI statistics tests')
SMALL = dict(n_x=64, n_v=12, n_t=48)
#: AC14: the forward pass reproduces the backward termination probabilities to round-off
ADJOINT_BOUND = 1e-10
#: AC8 on the ADI: 1 - Σ P over the outcome partition and pv - Σ outcome cash. The ADI march discounts exactly
#: (``step_columns`` integrates -r I exponentially) and every outcome is its own linear column, so both are round-off:
#: measured max 8.7e-15 (residual) and 1.1e-13 (reconciliation, pv ~ 100, incl. D23's surface vs column readout order)
#: over ROUTES + CARRIED on ARM64 (tmp/T14-measure.txt)
RESIDUAL_BOUND = 1e-12
RECONCILIATION_BOUND = 1e-11
#: P(KI before KO) vs the disable_ko_after_ki twin's KI settlement: two marches, round-off apart (measured max 1.8e-15)
TWIN_BOUND = 1e-12
#: the contract's slack on a probability (``Tolerance.PROBABILITY``): the ADI's splitting with a mixed derivative is not
#: monotone, so a first knock-in row far from the spot at short times may read slightly negative on a coarse mesh
SLACK = Tolerance.PROBABILITY

CLASSES = {
    ("heston", "snowball"): HestonSnowballPDESolver, ("heston", "phoenix"): HestonPhoenixPDESolver,
    ("heston", "ko_reset"): HestonKOResetSnowballPDESolver,
    ("slv", "snowball"): HestonSLVSnowballPDESolver, ("slv", "phoenix"): HestonSLVPhoenixPDESolver,
    ("slv", "ko_reset"): HestonSLVKOResetSnowballPDESolver,
}


def _adi(model, family, *, leverage=None, hp=HP, params=None, **kw):
    def build():
        extra = {"leverage_surface": leverage() if leverage else grid.unit_leverage()} if model == "slv" else {}
        solver = CLASSES[(model, family)](hp, **extra, **{**SMALL, **kw})
        if params:
            solver.params = replace(solver.params, **params)
        return solver
    return build


def _family_of(product):
    from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
    if isinstance(product, KnockOutResetSnowballOption):
        return "ko_reset"
    return "phoenix" if isinstance(product, PhoenixOption) else "snowball"


#: name -> (solver factory, product factory, env factory)
ROUTES = {
    "h_sb_dki": (_adi("heston", "snowball"), lambda: grid.snowball("discrete"), grid.flat_env),
    "h_sb_cki": (_adi("heston", "snowball"), grid.snowball, grid.flat_env),
    "h_ph_dki": (_adi("heston", "phoenix"), lambda: grid.phoenix(memory=False, ki="discrete"), grid.flat_env),
    "h_ph_cki": (_adi("heston", "phoenix"), lambda: grid.phoenix(memory=False), grid.flat_env),
    "h_phm_dki": (_adi("heston", "phoenix"), lambda: grid.phoenix(ki="discrete", arrears=0.02,
                                                                    coupon_barriers=[106.0, 106.0, 80.0, 80.0]),
                  grid.flat_env),
    "h_phm_cki": (_adi("heston", "phoenix"), grid.phoenix, grid.flat_env),
    "h_ph_exp_mem": (_adi("heston", "phoenix"), lambda: grid.phoenix(pay=EXPIRY), grid.flat_env),
    "h_ph_exp_nomem_dki": (_adi("heston", "phoenix"), lambda: grid.phoenix(memory=False, pay=EXPIRY, ki="discrete"),
                           grid.flat_env),
    "h_kr_dki": (_adi("heston", "ko_reset"), grid.ko_reset, grid.flat_env),
    "h_kr_cki": (_adi("heston", "ko_reset"), lambda: grid.ko_reset(ki="continuous"), grid.flat_env),
    "s_sb_dki": (_adi("slv", "snowball"), lambda: grid.snowball("discrete"), grid.flat_env),
    "s_sb_cki": (_adi("slv", "snowball"), grid.snowball, grid.flat_env),
    "s_ph_cki": (_adi("slv", "phoenix"), grid.phoenix, grid.flat_env),
    "s_kr_dki": (_adi("slv", "ko_reset"), grid.ko_reset, grid.flat_env),
    "sl_sb_cki": (_adi("slv", "snowball", leverage=grid.skew_leverage), grid.snowball, grid.flat_env),
    "sl_sb_dki": (_adi("slv", "snowball", leverage=grid.skew_leverage), lambda: grid.snowball("discrete"),
                  grid.flat_env),
    "sl_ph_dki": (_adi("slv", "phoenix", leverage=grid.skew_leverage), lambda: grid.phoenix(ki="discrete"),
                  grid.flat_env),
    "sl_kr_cki": (_adi("slv", "ko_reset", leverage=grid.skew_leverage), lambda: grid.ko_reset(ki="continuous"),
                  grid.flat_env),
    # the reverse continuous KI reads its first knock-in rows near t = 0 at -2e-6 on the SMALL mesh (beyond the slack;
    # ``test_a_negative_knock_in_row_beyond_the_slack_is_refused``): a finer mesh (-5.4e-7)
    "h_sb_rev_cki": (_adi("heston", "snowball", n_x=128, n_v=24, n_t=96), lambda: grid.snowball(reverse=True),
                     grid.flat_env),
    "h_sb_rev_dki": (_adi("heston", "snowball"), lambda: grid.snowball("discrete", reverse=True), grid.flat_env),
    "h_sb_nokoki_dki": (_adi("heston", "snowball"), lambda: grid.snowball("discrete", disable_ko_after_ki=True),
                        grid.flat_env),
    "h_sb_nodal_dki": (_adi("heston", "snowball", params=dict(event_projection="nodal")),
                       lambda: grid.snowball("discrete"), grid.flat_env),
    "h_ph_nodal_cki": (_adi("heston", "phoenix", params=dict(event_projection="nodal")), grid.phoenix,
                       grid.flat_env),
    "h_sb_douglas_cki": (_adi("heston", "snowball", scheme="DOUGLAS"), grid.snowball, grid.flat_env),
    "h_sb_uniform_dki": (_adi("heston", "snowball", grid_style="uniform"), lambda: grid.snowball("discrete"),
                         grid.flat_env),
    "h_sb_neumann_cki": (_adi("heston", "snowball", v0_boundary="neumann"), grid.snowball, grid.flat_env),
    "h_sb_rho0_dki": (_adi("heston", "snowball", hp=HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3,
                                                                    rho=0.0)),
                      lambda: grid.snowball("discrete"), grid.flat_env),
    "h_dated_sb_dki_aged": (_adi("heston", "snowball"), dated_snowball, lambda: dated_env(AGED)),
    "h_dated_sb_cki_aged": (_adi("heston", "snowball"), lambda: dated_snowball("continuous"),
                            lambda: dated_env(AGED)),
    "h_dated_ph_exp_aged": (_adi("heston", "phoenix"), lambda: dated_phoenix(pay=EXPIRY), lambda: dated_env(AGED)),
    "h_dated_kr_fresh": (_adi("heston", "ko_reset"), dated_ko_reset, dated_env),
}
#: carried knock-in: the KI path statistics are definitional, but the adjoint identity holds from the knocked-in
#: surface's readout too
CARRIED = {
    "h_sb_carried": (_adi("heston", "snowball"), lambda: grid.snowball(knocked_in=True), grid.flat_env),
    "h_ph_carried": (_adi("heston", "phoenix"), lambda: grid.phoenix(knocked_in=True), grid.flat_env),
    "s_kr_carried": (_adi("slv", "ko_reset"), lambda: grid.ko_reset(knocked_in=True), grid.flat_env),
}
CONTINUOUS = {name for name, (_, make, _) in ROUTES.items()
              if make().barrier_config.ki_observation_type is ObservationType.CONTINUOUS
              or make().barrier_config.ki_continuous}


def _route(name, table=None):
    make_solver, make_product, make_env = (table or {**ROUTES, **CARRIED})[name]
    return make_solver(), make_product(), make_env()


def _stats(name, **kw):
    solver, product, env = _route(name)
    return solver, product, env, solver.calculate_event_stats(product, env, **kw)


def _horizon(product, env):
    if isinstance(product, KnockOutResetSnowballOption):
        return float(product.get_pre_maturity_time(env))
    return float(product.get_maturity(env))


# --- AC7 / D23 ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(ROUTES))
def test_full_request_fills_every_number(name):
    solver, product, env, stats = _stats(name)
    assert isinstance(stats, EventStatistics)
    assert stats.product_family == _family_of(product)
    assert set(stats.unavailable) <= {"knock_in.expected_time"}, dict(stats.unavailable)
    for reason in stats.unavailable.values():
        assert reason.startswith("undefined:")
    assert stats.ki_table is not None and stats.knock_in.probability is not None
    assert stats.knock_in.expected_loss is not None and stats.knock_in.expected_loss >= 0.0
    model = solver._price_model
    assert stats.provenance == StatisticsProvenance(model, model) and not stats.provenance.surrogate
    assert (stats.coupons is None) == (stats.product_family != "phoenix")


@pytest.mark.parametrize("name", list(ROUTES) + list(CARRIED))
def test_pv_is_price_bitwise(name):
    """D23: ``pv`` is the value ``price()`` reads from the same march; ``price_with_events(...).npv`` is ``price()``."""
    solver, product, env = _route(name)
    price = _route(name)[0].price(product, env)
    assert solver.calculate_event_stats(product, env).pv == price
    assert _route(name)[0].price_with_events(product, env).npv == price


# --- AC8 ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(ROUTES) + list(CARRIED))
def test_outcomes_partition_within_the_measured_bound(name):
    _, _, _, stats = _stats(name)
    assert abs(stats.probability_residual) <= RESIDUAL_BOUND, stats.probability_residual
    assert abs(stats.reconciliation_error) <= RECONCILIATION_BOUND, stats.reconciliation_error


@pytest.mark.parametrize("name", ["h_sb_dki", "h_kr_dki", "sl_ph_dki"])
def test_residuals_stay_at_round_off_under_refinement(name):
    """The ADI partition is exact on every mesh (exact discounting, one linear column per outcome): refining the
    mesh leaves both residuals at round-off -- the measured bound is no discretisation allowance."""
    make_solver, make_product, make_env = ROUTES[name]
    for scale in (1, 2):
        solver = make_solver()
        solver.n_x, solver.n_v, solver.n_t = SMALL["n_x"] * scale, SMALL["n_v"] * scale, SMALL["n_t"] * scale
        stats = solver.calculate_event_stats(make_product(), make_env())
        assert abs(stats.probability_residual) <= RESIDUAL_BOUND
        assert abs(stats.reconciliation_error) <= RECONCILIATION_BOUND


def test_flat_vol_surrogate_reports_its_price_gap_as_the_reconciliation_error():
    """DESIGN §4.9a: the explicit ``flat_vol_surrogate`` opt-out keeps the 1D flat-vol statistics beside the 2D pv,
    labelled by provenance, outside the identity guarantees and AC8's bound (the AC8 decision): its outcome cash sums
    to a flat-vol 1D price, so its reconciliation error is the 2D - 1D price gap, reported, never hidden."""
    from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver

    product, env = grid.snowball("discrete"), grid.flat_env()
    surrogate = HestonSnowballPDESolver(HP, statistics="flat_vol_surrogate", **SMALL)
    stats = surrogate.calculate_event_stats(product, env)
    assert stats.provenance.surrogate and stats.provenance.statistics_model == "flat_vol_1d"
    assert stats.pv == HestonSnowballPDESolver(HP, **SMALL).price(product, env)
    one_d = stats.pv - stats.reconciliation_error                    # Σ outcome cash: a flat-vol 1D price
    flat = SnowballPDESolver().price(product, env)
    assert abs(one_d - flat) <= 1e-2 < abs(stats.reconciliation_error)
    assert stats.reconciliation_error == pytest.approx(stats.pv - flat, abs=1e-2)


# --- AC14 -----------------------------------------------------------------------------------------------------

def _adjoint(name, table=None):
    solver, product, env = _route(name, table)
    return vol_model_statistics.native_adi(solver, product, env, event_stats=True, adjoint_check=True), env


@pytest.mark.parametrize("name", list(ROUTES) + list(CARRIED))
def test_forward_pass_reproduces_every_backward_column(name):
    """The forward pass is the exact discrete adjoint of the ADI march: Σ <p, affine sources> over every step's rows,
    every event stage on a zero block and the terminal block reproduces the backward readout of every unit column
    (<= 1e-10 abs) and every maturity cash column (1e-12 relative)."""
    check, _ = _adjoint(name)
    L, recon, at = check.statistics.layout, check.forward.reconstructed, check.at
    units = [*L.ko, *L.post_ko, *L.maturity_unit] + ([] if L.kis is None else [L.kis])
    for c in units:
        assert abs(recon[int(c)] - at(int(c))) <= ADJOINT_BOUND, (c, recon[int(c)], at(int(c)))
    for c in L.maturity:
        assert math.isclose(recon[int(c)], at(int(c)), rel_tol=1e-12, abs_tol=1e-12), (c, recon[int(c)], at(int(c)))


@pytest.mark.parametrize("name", list(ROUTES))
def test_forward_pass_reproduces_the_termination_probabilities(name):
    """AC14 as stated: Σ KO, maturity and KI settlement probabilities from the forward pass agree with the backward
    solve's to <= 1e-10 abs."""
    check, env = _adjoint(name)
    st, stats, recon = check.statistics, check.report, check.forward.reconstructed
    L = st.layout
    records = list(st.ko_records) + list(getattr(st, "post_records", []))
    ko = [recon[c] / float(env.get_discount_factor(r.settlement_time if r.settlement_time is not None
                                                   else r.observation_time))
          for c, r in zip([*L.ko, *L.post_ko], records) if r.barrier is not None]
    assert abs(sum(ko) - float(np.sum(stats.observations.probability))) <= ADJOINT_BOUND
    df = lambda t: float(env.get_discount_factor(t))  # noqa: E731
    assert abs(recon[L.maturity_unit[0]] / df(stats.maturity.time) - stats.maturity.probability) <= ADJOINT_BOUND
    assert abs(recon[L.ki_unit] / df(stats.ki_settlement.time) - stats.ki_settlement.probability) <= ADJOINT_BOUND


# --- the ADI step's transpose ---------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["h_sb_dki", "h_sb_uniform_dki", "h_sb_neumann_cki", "h_sb_rho0_dki", "sl_sb_cki"])
@pytest.mark.parametrize("douglas", [False, True])
def test_step_transpose_is_the_adjoint_of_step_columns(name, douglas):
    """<step_columns(U; rows), P> == <U, S^T P> + dlow . low + dhigh . high on random columns and rows."""
    solver, product, env = _route(name)
    core = solver._make_core(product, env, float(product.get_maturity(env)))
    rng = np.random.default_rng(7)
    n_s, n_v, n_c = core.N_S, core.N_V, 3
    dt, tau, theta = 0.02, 0.4, 0.5
    t_mid = core.T - tau + 0.5 * dt
    U, P = rng.standard_normal((n_s, n_v, n_c)), rng.standard_normal((n_s, n_v, n_c))
    low, high = rng.standard_normal(n_c), rng.standard_normal(n_c)
    out = step_columns(core, U.copy(), dt, tau, theta, t_mid, low.copy(), high.copy(), douglas)
    back, dlow, dhigh = vol_model_adjoint.StepTranspose(core, dt, theta, t_mid, douglas).apply(P)
    lhs = float(np.sum(out * P))
    rhs = float(np.sum(U * back) + dlow @ low + dhigh @ high)
    assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-10)


# --- AC13 -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(ROUTES))
def test_knock_in_table_identities(name):
    _, product, env, stats = _stats(name)
    table, ki = stats.ki_table, stats.knock_in
    assert ki.probability > 0.0
    assert math.isclose(float(np.sum(table.probability)), ki.probability, rel_tol=1e-12, abs_tol=1e-15)
    np.testing.assert_array_equal(table.payment_times, table.times)                       # D8
    assert table.payment_dates == table.dates
    mean = float(np.sum(table.times * table.probability)) / ki.probability
    if name in CONTINUOUS:
        assert abs(mean - ki.expected_time) <= 1.0 / float(env.bus_days_in_year) + 1e-12, (mean, ki.expected_time)
    else:
        assert math.isclose(mean, ki.expected_time, rel_tol=1e-12, abs_tol=1e-15)
        assert not np.any(table.interpolated) and table.interpolation_scheme is None
    # settling knocked in needs a first knock-in before any knock-out
    assert stats.ki_settlement.probability <= ki.probability + RESIDUAL_BOUND <= 1.0 + RESIDUAL_BOUND
    assert np.all(table.probability >= -SLACK)
    assert np.all(np.diff(table.survival) <= SLACK)
    assert np.all(table.survival >= -SLACK)


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
    """One row per trading day (numeric clock: ``1 / bus_days_in_year``); a row is ``interpolated`` exactly when a
    day boundary is not a node of the march (the terminal damping's half node included)."""
    check, env = _adjoint(name)
    _, product, _ = _route(name)
    stats, horizon = check.report, _horizon(product, env)
    table = stats.ki_table
    if table.dates is None:
        bus = float(env.bus_days_in_year)
        days = [k / bus for k in range(1, int(math.ceil(horizon * bus)) + 1) if k / bus < horizon - 1e-10]
        np.testing.assert_allclose(table.times, days + [horizon], rtol=0, atol=1e-15)
    else:
        assert len(table.dates) == table.size and table.times[-1] == horizon
    nodes = check.forward.times
    starts = np.r_[0.0, table.times[:-1]]
    off = lambda b: np.min(np.abs(nodes[None, :] - b[:, None]), axis=1) > 1e-10  # noqa: E731
    np.testing.assert_array_equal(table.interpolated, off(table.times) | off(starts))
    if np.any(table.interpolated):
        assert table.interpolation_scheme == "step_overlap"


@pytest.mark.parametrize("name", ["h_sb_dki", "h_sb_cki", "h_phm_dki", "h_ph_exp_mem", "h_kr_dki", "h_kr_cki",
                                  "sl_sb_cki", "sl_kr_cki", "h_sb_rev_cki", "h_dated_sb_dki_aged"])
def test_knock_in_before_ko_is_the_settlement_of_the_twin_that_never_knocks_out_after_ki(name):
    """An independent backward number: with ``disable_ko_after_ki`` the knocked-in surface never knocks out, so its
    knocked-in settlement IS P(KI before any KO) of the original note (the same not-knocked-in march), within the
    two marches' discount errors."""
    _, product, env, stats = _stats(name)
    twin = _route(name)[1]
    twin.barrier_config = replace(twin.barrier_config, disable_ko_after_ki=True)
    twin.validate()
    twin_stats = _route(name)[0].calculate_event_stats(twin, env)
    assert abs(stats.knock_in.probability - twin_stats.ki_settlement.probability) <= TWIN_BOUND, (
        stats.knock_in.probability, twin_stats.ki_settlement.probability)
    assert stats.knock_in.probability - stats.ki_settlement.probability > 100.0 * TWIN_BOUND


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


@pytest.mark.parametrize("model", ["heston", "slv"])
def test_no_knock_in_after_a_certain_knock_out(model):
    """Every grid node knocks out at 0.25 (the level 1.0 lies below the spot grid); the KI is observed only after.
    What remains is the price's own far knock-in-side edge row before 0.25: the discrete knock-in it labels certain
    (``rows.knocks_in``, D23) carries 3.6e-10 on this mesh (the high-variance columns reach the 8-std edge), and the
    knocked-in settlement reads exactly the same mass -- no knock-in after the knock-out is counted."""
    env = grid.flat_env()
    stats = _adi(model, "snowball")().calculate_event_stats(_certain_knock_out_first(1.0), env)
    assert abs(stats.knock_in.probability) <= 1e-9
    assert abs(stats.knock_in.probability - stats.ki_settlement.probability) <= 1e-14
    np.testing.assert_allclose(stats.ki_table.probability, np.zeros(3), rtol=0, atol=1e-9)
    assert abs(stats.knock_in.expected_loss) <= 1e-7
    reachable = _adi(model, "snowball")().calculate_event_stats(_certain_knock_out_first(103.0), env)
    assert reachable.knock_in.probability > 0.3


@pytest.mark.parametrize("family", ["snowball", "phoenix", "ko_reset"])
def test_knock_in_after_a_knock_out_is_not_counted(family):
    """The KO-independent "KI ever" monitor is gone: P(KI before KO) of the note is strictly below the same note's
    with an unreachable knock-out (where every first knock-in comes before a knock-out)."""
    make = {"snowball": grid.snowball, "phoenix": grid.phoenix, "ko_reset": grid.ko_reset}[family]
    env, solver = grid.flat_env(), _adi("heston", family)
    note, no_ko = make(), make()
    for config in ("barrier_config", "post_barrier_config"):
        if getattr(no_ko, config, None) is not None:
            setattr(no_ko, config, replace(getattr(no_ko, config), ko_barrier=1e9))
    for config in ("barrier_config", "post_barrier_config"):
        bc = getattr(no_ko, config, None)
        if bc is not None and bc.ko_observation_schedule is not None:
            setattr(no_ko, config, replace(bc, ko_observation_schedule=grid._schedule(
                [r.observation_time for r in bc.ko_observation_schedule.records], 1e9)))
    no_ko.validate()
    a = solver().calculate_event_stats(note, env).knock_in.probability
    b = solver().calculate_event_stats(no_ko, env).knock_in.probability
    assert a < b - 1e-3, (a, b)


def test_knock_out_wins_a_same_instant_discrete_tie():
    """Discrete KI on the KO date with the KO level below the KI level: on the shared date every node is knocked out
    or knocked in; the knock-out wins, so nothing is left alive after it."""
    env = grid.flat_env()
    tie = SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(ko_barrier=100.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=[0.5], ki_barrier=101.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_dates=[0.5, 1.0]),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
    )
    stats = _adi("heston", "snowball", params=dict(event_projection="nodal"))().calculate_event_stats(tie, env)
    first, ko = stats.ki_table.probability[0], stats.observations.probability[0]
    assert abs(first + ko - 1.0) <= 2.0 * RESIDUAL_BOUND, (first, ko)
    assert stats.ki_table.probability[1] <= 1e-12


# --- AC10: valuation-date rules, no forward pass ----------------------------------------------------------------

@pytest.fixture
def no_forward_pass(monkeypatch):
    calls = []

    def refuse(*args, **kwargs):
        calls.append(args)
        raise AssertionError("the forward adjoint pass ran")

    monkeypatch.setattr(vol_model_statistics, "adi_forward_pass", refuse)
    return calls


def _ko_at_valuation(family):
    from datetime import datetime
    from intraday.conftest import dated_phoenix as intraday_phoenix, dated_snowball as intraday_snowball, flat_env
    from quantark.util.calendar import CalendarType, create_calendar
    calendar = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    p = (intraday_snowball if family == "snowball" else intraday_phoenix)(calendar, datetime(2026, 3, 16))
    p.barrier_config = replace(p.barrier_config, ki_continuous=True)
    date = p.barrier_config.ko_observation_schedule.records[2].observation_date
    env = flat_env(date, spot=110.0)
    env.vol_surface = grid.flat_env().vol_surface
    return p, env


@pytest.mark.parametrize("model", ["heston", "slv"])
@pytest.mark.parametrize("family", ["snowball", "phoenix"])
def test_knock_out_at_valuation_reports_no_knock_in(model, family, no_forward_pass):
    product, env = _ko_at_valuation(family)
    solver = _adi(model, family)()
    stats = solver.calculate_event_stats(product, env)
    assert stats.observations.probability.sum() == 1.0
    assert stats.ki_settlement.probability == 0.0 and stats.maturity.probability == 0.0
    assert stats.knock_in.probability == 0.0 and stats.ki_table.size == 0
    assert stats.knock_in.expected_loss == 0.0
    assert stats.pv == _adi(model, family)().price(product, env)
    assert stats.provenance == StatisticsProvenance(solver._price_model, solver._price_model)
    assert no_forward_pass == []


@pytest.mark.parametrize("name", list(CARRIED))
def test_carried_knock_in_is_definitional(name, no_forward_pass):
    _, _, _, stats = _stats(name)
    assert stats.knock_in.probability == 1.0 and stats.knock_in.expected_time == 0.0
    assert stats.ki_table.times.tolist() == [0.0] and stats.ki_table.probability.tolist() == [1.0]
    assert stats.maturity.probability == 0.0
    assert stats.unavailable == {}
    assert no_forward_pass == []


def test_knock_in_at_valuation_and_no_barrier(no_forward_pass):
    now = _adi("heston", "snowball")().calculate_event_stats(grid.snowball(), grid.flat_env(70.0))
    assert now.knock_in.probability == 1.0 and now.knock_in.expected_time == 0.0
    product = SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(grid.KO_DATES)),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True))
    none = _adi("heston", "snowball")().calculate_event_stats(product, grid.flat_env())
    assert none.knock_in.probability == 0.0 and none.ki_table.size == 0
    assert none.ki_settlement.probability == 0.0 and none.knock_in.expected_loss == 0.0
    assert none.unavailable["knock_in.expected_time"].startswith("undefined:")
    assert no_forward_pass == []


# --- D17 ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["h_sb_cki", "h_ph_exp_mem", "h_kr_dki", "sl_sb_dki"])
def test_pruned_request_never_runs_the_forward_pass(name, monkeypatch):
    solver, product, env = _route(name)
    full = solver.calculate_event_stats(product, env)
    calls = []
    real = vol_model_statistics.adi_forward_pass

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(vol_model_statistics, "adi_forward_pass", spy)
    for streams in (frozenset({EventType.KO}), frozenset({EventType.KO, EventType.COUPON, EventType.MATURITY_NO_KO,
                                                          EventType.MATURITY_WITH_KI})):
        pruned = _route(name)[0].calculate_event_stats(product, env, streams=streams)
        assert calls == []
        assert pruned.knock_in.probability is None and pruned.ki_table is None
        assert pruned.unavailable["knock_in.probability"] == "not requested"
        np.testing.assert_array_equal(pruned.observations.probability, full.observations.probability)
        assert pruned.maturity == full.maturity and pruned.ki_settlement == full.ki_settlement
        assert pruned.pv == full.pv
    _route(name)[0].calculate_event_stats(product, env, streams=frozenset({EventType.KO, EventType.KI}))
    assert len(calls) == 1
    _route(name)[0].price_with_events(product, env)
    assert len(calls) == 2


def test_the_knock_in_stream_reaches_the_event_distribution():
    solver, product, env = _route("h_sb_dki")
    stats = solver.calculate_event_stats(product, env)
    dist = _route("h_sb_dki")[0].price_with_events(product, env).event_distribution
    np.testing.assert_array_equal(dist.stream_times[EventType.KI], stats.ki_table.times)
    np.testing.assert_array_equal(dist.probabilities[EventType.KI], stats.ki_table.probability)
    assert EventType.KI not in dist.flagged_streams
    assert dist.provenance == stats.provenance


# --- routes the pass does not transpose: refused with a reason ----------------------------------------------------

@pytest.mark.parametrize("kw", [dict(column_backend="scalar"), dict(v_drift_scheme="semi_lagrangian")])
def test_untransposed_sweeps_are_refused_with_a_reason(kw):
    """The scalar column backend and the semi-Lagrangian variance transport run the core's own per-claim sweeps, which
    the pass does not transpose: the knock-in path statistics are ``None`` with an ``undefined:`` reason (never an
    approximation); every termination outcome is still marched."""
    product, env = grid.snowball("discrete"), grid.flat_env()
    stats = _adi("heston", "snowball", **kw)().calculate_event_stats(product, env)
    for key in ("knock_in.probability", "knock_in.expected_time", "ki_table"):
        assert stats.unavailable[key].startswith("undefined:") and "scalar" in stats.unavailable[key]
    assert stats.knock_in.probability is None and stats.ki_table is None
    assert abs(stats.probability_residual) <= RESIDUAL_BOUND


def test_a_negative_knock_in_row_beyond_the_slack_is_refused():
    """On the SMALL mesh the reverse continuous KI's first knock-in rows near t = 0 read -2e-6 (the CS splitting with
    rho != 0 is not monotone; -4.3e-7 at the default mesh, -4.1e-9 at n_t = 400): beyond the probability slack the
    knock-in path statistics are refused with the reason, never clipped; the outcomes stay."""
    stats = _adi("heston", "snowball")().calculate_event_stats(grid.snowball(reverse=True), grid.flat_env())
    reason = stats.unavailable["ki_table"]
    assert reason.startswith("undefined:") and "negative" in reason and "refine" in reason
    assert stats.knock_in.probability is None and stats.ki_table is None
    assert abs(stats.probability_residual) <= RESIDUAL_BOUND


# --- the MC reference -------------------------------------------------------------------------------------------

MC_CASES = {
    "heston_sb_dki": ("heston", "snowball", lambda: grid.snowball("discrete")),
    "heston_sb_cki": ("heston", "snowball", grid.snowball),
    "heston_ph_dki": ("heston", "phoenix", lambda: grid.phoenix(ki="discrete")),
    "heston_kr_dki": ("heston", "ko_reset", grid.ko_reset),
    "slv_sb_cki": ("slv", "snowball", grid.snowball),
    "slv_ph_dki": ("slv", "phoenix", lambda: grid.phoenix(ki="discrete")),
}
#: the ADI mesh of the MC comparison and its refinement (the allowance is |ADI - ADI fine|)
MC_ADI = dict(n_x=120, n_v=32, n_t=200)
MC_ADI_FINE = dict(n_x=240, n_v=64, n_t=400)
MC_BATCHES, MC_BATCH_PATHS = 16, 4096
#: QE substeps per contractual interval (continuous KI already steps daily; Sobol caps the RQMC dimension at 21201)
MC_SUBSTEPS = {"discrete": 32, "continuous": 4}


def _mc(model, family, product, env, substeps):
    from quantark.asset.equity.engine import mc as m
    from quantark.asset.equity.param import MCParams
    from quantark.util.enum.engine_enums import MonteCarloMethod

    params = MCParams(num_paths=MC_BATCH_PATHS, seed=11, rqmc_min_batches=MC_BATCHES, rqmc_max_batches=MC_BATCHES,
                      rqmc_target_std=1e-12, rqmc_paths_mode="per_batch")
    # one QE step per contractual interval (the default) biases P(S_0.25 >= 103) by -2.4e-3 (-8.5 batch SEs against
    # the analytic Heston digital 0.410244): refine every interval
    kw = dict(params=params, method=MonteCarloMethod.RANDOMIZED_QUASI, substeps_per_interval=substeps)
    cls = {("heston", "snowball"): m.HestonSnowballMCEngine, ("heston", "ko_reset"): m.HestonSnowballMCEngine,
           ("heston", "phoenix"): m.HestonPhoenixMCEngine, ("slv", "snowball"): m.HestonSLVSnowballMCEngine,
           ("slv", "phoenix"): m.HestonSLVPhoenixMCEngine, ("slv", "ko_reset"): m.HestonSLVSnowballMCEngine}
    if model == "slv":
        kw["leverage_surface"] = grid.unit_leverage()
    return cls[(model, family)](HP, **kw).calculate_event_stats(product, env)


def _by_week(values):
    values = np.asarray(values, dtype=float)
    return np.add.reduceat(values, np.arange(0, values.size, 5))


@pytest.mark.slow
@pytest.mark.parametrize("name", list(MC_CASES))
def test_statistics_agree_with_the_mc_reference(name):
    """The outcome probabilities, P(KI before KO), E[KI time] and the per-date table within 4 MC batch SEs plus the
    ADI's own discretisation allowance |ADI - ADI(2x mesh)|. Continuous rows are compared by trading week."""
    model, family, make = MC_CASES[name]
    env = grid.grid_env(skew=False)          # the SLV MC needs a grid surface; the ADI reads the curves only
    adi = _adi(model, family, **MC_ADI)().calculate_event_stats(make(), env)
    fine = _adi(model, family, **MC_ADI_FINE)().calculate_event_stats(make(), env)
    continuous = make().barrier_config.ki_continuous
    mc = _mc(model, family, make(), env, MC_SUBSTEPS["continuous" if continuous else "discrete"])
    se = mc.standard_errors
    for path in ("knock_in.probability", "knock_in.expected_time", "maturity.probability",
                 "ki_settlement.probability", "knock_in.expected_loss"):
        a, b, f = adi.require(path), mc.require(path), fine.require(path)
        assert abs(a - b) <= 4.0 * se[path] + abs(a - f) + 1e-12, (path, a, b, se[path], abs(a - f))
    a, b, f = adi.observations.probability, mc.observations.probability, fine.observations.probability
    s = np.asarray(se["observations.probability"], dtype=float)
    assert np.all(np.abs(a - b) <= 4.0 * s + np.abs(a - f) + 1e-12), (a, b, s)
    np.testing.assert_allclose(adi.ki_table.times, mc.ki_table.times, rtol=0, atol=1e-12)
    a, b, f, s = (adi.ki_table.probability, mc.ki_table.probability, fine.ki_table.probability,
                  np.asarray(se["ki_table.probability"], dtype=float))
    if adi.ki_table.size > 20:
        a, b, f = (_by_week(x) for x in (a, b, f))
        s = np.sqrt(_by_week(s * s))
    s = np.maximum(s, np.sqrt(np.clip(a, 0.0, 1.0) * (1.0 - np.clip(a, 0.0, 1.0)) / (MC_BATCHES * MC_BATCH_PATHS)))
    assert np.all(np.abs(a - b) <= 4.0 * s + np.abs(a - f) + 1e-12), (name, np.max(np.abs(a - b) - 4.0 * s
                                                                                  - np.abs(a - f)))
