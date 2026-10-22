"""Cash-leg folding on the DCN engines (ship spec 2026-10-21 D9, D10, task T9).

A DCN overlay is buyer-view (``fold_cash_legs`` divides out ``direction_sign``): every engine adds its
cash to the buyer-view cash streams before it signs them, so a folded per-unit PV is
``sign · (buyer npv + Σ_e P_e · amount_e · DF(pay_e))``. Knock-out cash goes where the DCN's knock-out
coupon goes (paid at its knock-out payment), maturity cash to the two survivor branches at the loss-leg
settlement: the alive (never knocked-in) terminal column / survivors and the knocked-in branch.

The overlay carries one knock-out slot per monthly observation (the event route's axis). An observation
that is not a knock-out observation has knock-out probability exactly zero on every route, so its slot is
bound and its cash, conditional on an event that cannot happen, adds nothing (a leg-derived overlay does
carry cash there: ``v_e - v_anchor`` of a leg paying at a knock-out).

MC folds to round-off against its own event route (AC4 / AC5, same paths). DCN PDE has no event
statistics (D10): AC4 is stated on the march's own weights (unit cash per event group), the weights and
the folded leg PV are checked against the same-model MC event route within 3 standard errors (AC6), and
the knock-out coupon product twin (AC14) pins the cash against the product itself on every engine.
"""
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.mc import DCNMCEngine
from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import (
    HestonDCNMCEngine,
    LocalVolDCNMCEngine,
    QEDCNMCEngine,
    coupled_heston_ladder_pair,
)
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import HestonDCNPDESolver, LocalVolDCNPDEEngine
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
from quantark.asset.equity.product.option.dcn_option import DCNDirection
from quantark.cashleg import EventType, fold_cash_legs
from quantark.cashleg.folding import FoldedEventCash, FoldOverlay, _event_structure
from quantark.execution.errors import CapabilityError
from quantark.util.enum.engine_enums import HestonMCScheme
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams

from dcn_fixtures import DCN_A, DCN_B, make_dcn
from test_cashleg.test_folding import _leg_set
from test_asset.equity.engine_pde.test_dcn_lifecycle_state import CASH, ROW, ROW_PAY, STRADDLE, env_on, flow, state_on

HESTON = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)
START = DCN_A["valuation_date"]
SAT = datetime(2024, 6, 8)           # SSE closed: an unmonitored valuation instant (memory: any run calendar day)
OBS = datetime(2024, 12, 3)          # DCN_A month 23, a coupon + knock-out row on the valuation date
FINAL = datetime(2025, 1, 3)         # DCN_A's final observation date
IDENTITY = 1e-10                     # AC4 / AC14, relative to the PV
PARITY = 1e-9                        # AC5, relative to the trade notional
SE = 3.0                             # AC6, Monte Carlo standard errors
C = 1_000.0                          # hand-built cash per event


def _mc(**kw):
    return dict(num_paths=kw.pop("num_paths", 1 << 12), seed=kw.pop("seed", 7), **kw)


def _coupled():
    return coupled_heston_ladder_pair(HESTON, 1, HestonMCScheme.QUADEXP_M, **_mc())[0]


#: name -> (engine factory, uses a grid vol surface)
MC = {
    "gbm_sobol": (lambda: DCNMCEngine(**_mc()), False),
    "gbm_pseudo": (lambda: DCNMCEngine(**_mc(use_sobol=False)), False),
    "gbm_antithetic": (lambda: DCNMCEngine(**_mc(use_sobol=False, use_antithetic=True)), False),
    "gbm_rqmc_threads": (lambda: DCNMCEngine(**_mc(num_batches=4, num_workers=2)), False),
    "lv": (lambda: LocalVolDCNMCEngine(**_mc()), True),
    "heston": (lambda: HestonDCNMCEngine(model_params=HESTON, **_mc()), False),
    "qe": (lambda: QEDCNMCEngine(HESTON, martingale_correction=True, **_mc()), False),
    "coupled_coarse": (_coupled, False),
}
PDE = {
    "pde": (lambda: DCNPDEEngine(PDEParams(accuracy="fast")), False),
    "pde_nodal": (lambda: DCNPDEEngine(PDEParams(accuracy="fast", event_projection="nodal")), False),
    "lv_pde": (lambda: LocalVolDCNPDEEngine(params=PDEParams(accuracy="fast")), True),
    "heston_pde": (lambda: HestonDCNPDESolver(HESTON, n_x=101, n_v=31), False),
    "heston_pde_nodal": (lambda: HestonDCNPDESolver(HESTON, n_x=101, n_v=31, event_projection="nodal"), False),
}
ENGINES = {**MC, **PDE}
DIRECTIONS = [DCNDirection.BUYER, DCNDirection.SELLER]


def _case(name, contract=DCN_A, d=START, spot=6000.0, **product_kw):
    make, grid = ENGINES[name]
    return make, make_dcn(contract, valuation_date=d, **product_kw), env_on(d, spot=spot, grid=grid)


def _overlay(product, env, ko=0.0, no_ki=0.0, ki=0.0, only_ko_eligible=True):
    """A hand-built overlay on ``product``'s fold structure; knock-out cash per slot (scalar or array)."""
    s = _event_structure(product, env)
    amounts = np.broadcast_to(np.asarray(ko, dtype=float), (len(s.ko),)).copy()
    if only_ko_eligible:
        amounts[~build_dcn_grid_context(product).obs_is_ko] = 0.0
    return FoldOverlay(
        ko=tuple(FoldedEventCash(sl.schedule, sl.determination_time, sl.payment_time, float(a))
                 for sl, a in zip(s.ko, amounts)),
        maturity_no_ki=FoldedEventCash(s.no_ki.schedule, s.no_ki.determination_time, s.no_ki.payment_time,
                                       float(no_ki)),
        maturity_ki=FoldedEventCash(s.ki.schedule, s.ki.determination_time, s.ki.payment_time, float(ki)),
    )


def _per_df(product, env, cash):
    """``cash / DF(pay)`` per knock-out slot and for the maturity settlement: unit-present-value amounts."""
    s = _event_structure(product, env)
    ko = np.array([cash / env.get_discount_factor(sl.payment_time) for sl in s.ko])
    return ko, cash / env.get_discount_factor(s.ki.payment_time)


def _identity_overlay(product, env, cash=C):
    ko, mat = _per_df(product, env, cash)
    return _overlay(product, env, ko=ko, ki=mat)


def _twin_overlay(product, delta):
    """Cash of a knock-out coupon rate moved by ``delta``, per knock-out record, paid at its KO payment."""
    ctx = build_dcn_grid_context(product)
    env = env_on(product.schedule.valuation_date)
    return _overlay(product, env, ko=product.participation * delta * ctx.ko_accruals * product.notional)


def _route(engine, product, env, **kw):
    return engine.price_with_events(product, env, **kw)


def _reassemble(overlay, dist, env, unit):
    """``unit · Σ_e P_e · amount_e · DF(pay_e) + constant`` on a route's weights (``unit`` = q · sign)."""
    p_ko = np.asarray(dist.probabilities[EventType.KO], dtype=float)
    total = sum(p * e.amount * env.get_discount_factor(e.payment_time) for p, e in zip(p_ko, overlay.ko))
    for et, e in ((EventType.MATURITY_NO_KO, overlay.maturity_no_ki), (EventType.MATURITY_WITH_KI, overlay.maturity_ki)):
        total += float(dist.probabilities[et]) * e.amount * env.get_discount_factor(e.payment_time)
    return unit * total + overlay.constant_pv


# --- contract basics ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(ENGINES))
def test_every_dcn_engine_declares_folding(name):
    make, _, _ = _case(name)
    assert make().supports_cash_leg_folding is True


@pytest.mark.parametrize("name", sorted(ENGINES))
def test_an_empty_overlay_is_the_plain_price_bitwise(name):
    make, product, env = _case(name)
    assert make().price_folded(product, env, _overlay(product, env)) == make().price(product, env)


@pytest.mark.parametrize("name", sorted(ENGINES))
def test_an_overlay_of_another_dcn_is_refused(name):
    make, product, env = _case(name)
    other = make_dcn(DCN_B)
    with pytest.raises(ValidationError):
        make().price_folded(product, env, _identity_overlay(other, env))


@pytest.mark.parametrize("name", sorted(ENGINES))
def test_a_maturity_branch_at_another_settlement_is_refused(name):
    make, product, env = _case(name)
    ov = _identity_overlay(product, env)
    moved = FoldedEventCash("main", ov.maturity_ki.determination_time, ov.maturity_ki.payment_time + 0.01, C)
    with pytest.raises(ValidationError):
        make().price_folded(product, env, FoldOverlay(ov.ko, ov.maturity_no_ki, moved))


@pytest.mark.parametrize("name", sorted(ENGINES))
def test_cash_on_an_observation_that_cannot_knock_out_adds_nothing(name):
    # DCN_B's first three rows are coupon-only (ko_lock_months=6): knock-out probability zero on every route
    make, product, env = _case(name, contract=DCN_B)
    ko_eligible = build_dcn_grid_context(product).obs_is_ko
    assert not ko_eligible.all()
    plain = _overlay(product, env, ko=C, ki=C)
    junk = _overlay(product, env, ko=np.where(ko_eligible, C, 7.0 * C), ki=C, only_ko_eligible=False)
    assert make().price_folded(product, env, junk) == make().price_folded(product, env, plain)


# --- AC4: identity ------------------------------------------------------------------------------------------------

MC_SPOTS = [6000.0, 5950.0, 4550.0]          # ATM, just under the KO barrier, just above the KI barrier


@pytest.mark.parametrize("name", sorted(MC))
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("spot", MC_SPOTS)
def test_mc_identity_on_its_own_event_route(name, direction, spot):
    make, product, env = _case(name, spot=spot, direction=direction)
    result = _route(make(), product, env)
    p_no_ki = float(result.event_distribution.probabilities[EventType.MATURITY_NO_KO])
    folded = make().price_folded(product, env, _identity_overlay(product, env))
    expected = product.direction_sign * C * (1.0 - p_no_ki)
    assert abs((folded - result.npv) - expected) <= IDENTITY * abs(result.npv)


@pytest.mark.parametrize("contract", [DCN_A, DCN_B])
@pytest.mark.parametrize("d, spot", [(START, 6000.0), (SAT, 5500.0), (OBS, 5500.0), (STRADDLE, 5800.0)])
def test_mc_identity_on_other_dates_and_contracts(contract, d, spot):
    make, product, env = _case("gbm_sobol", contract=contract, d=d, spot=spot)
    result = _route(make(), product, env)
    p_no_ki = float(result.event_distribution.probabilities[EventType.MATURITY_NO_KO])
    folded = make().price_folded(product, env, _identity_overlay(product, env))
    assert abs((folded - result.npv) - C * (1.0 - p_no_ki)) <= IDENTITY * abs(result.npv)


def _march_weights(engine_factory, product, env):
    """(W_ko, W_no_ki, W_ki): the march's own present value of unit-PV cash on each event group."""
    sign = product.direction_sign
    base = engine_factory().price(product, env)
    ko, mat = _per_df(product, env, 1.0)
    w_ko = (engine_factory().price_folded(product, env, _overlay(product, env, ko=ko)) - base) / sign
    w_no_ki = (engine_factory().price_folded(product, env, _overlay(product, env, no_ki=mat)) - base) / sign
    w_ki = (engine_factory().price_folded(product, env, _overlay(product, env, ki=mat)) - base) / sign
    return base, w_ko, w_no_ki, w_ki


PDE_SPOTS = [6000.0, 5950.0, 4550.0]


@pytest.mark.parametrize("name", sorted(PDE))
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("spot", PDE_SPOTS)
def test_pde_identity_on_the_marchs_own_weights(name, direction, spot):
    make, product, env = _case(name, spot=spot, direction=direction)
    base, w_ko, w_no_ki, w_ki = _march_weights(make, product, env)
    folded = make().price_folded(product, env, _identity_overlay(product, env))
    assert abs((folded - base) - product.direction_sign * C * (w_ko + w_ki)) <= IDENTITY * abs(base)
    # the march's weights are a distribution up to its discretization
    assert w_ko + w_no_ki + w_ki == pytest.approx(1.0, abs=1e-4)
    assert min(w_ko, w_no_ki, w_ki) > -1e-6


@pytest.mark.parametrize("name", sorted(PDE))
@pytest.mark.parametrize("d, spot", [(SAT, 5500.0), (OBS, 5500.0)])
def test_pde_identity_on_a_closed_day_and_a_row_day(name, d, spot):
    make, product, env = _case(name, d=d, spot=spot)
    base, w_ko, w_no_ki, w_ki = _march_weights(make, product, env)
    folded = make().price_folded(product, env, _identity_overlay(product, env))
    assert abs((folded - base) - C * (w_ko + w_ki)) <= IDENTITY * abs(base)


# --- AC5: MC parity with every allowlisted leg type -----------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(MC))
@pytest.mark.parametrize("contract", [DCN_A, DCN_B])
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("quantity", [1.0, -2.0])
def test_mc_parity_with_the_event_route(name, contract, direction, quantity):
    make, product, env = _case(name, contract=contract, direction=direction)
    result = _route(make(), product, env)
    dist = result.event_distribution
    legs = _leg_set(dist.event_times)
    overlay = fold_cash_legs(legs, product, env, quantity=quantity, unit_notional=100.0)
    folded = quantity * make().price_folded(product, env, overlay) + overlay.constant_pv
    expected = quantity * result.npv + sum(leg.value(dist, env, 100.0) for leg in legs)
    assert abs(folded - expected) <= PARITY * product.notional * abs(quantity)


@pytest.mark.parametrize("d, spot", [(SAT, 5500.0), (OBS, 5500.0), (OBS, 6300.0), (FINAL, 6300.0),
                                     (FINAL, 4000.0), (FINAL, 5500.0)])
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_mc_parity_on_closed_row_and_final_days(d, spot, direction):
    make, product, env = _case("gbm_sobol", d=d, spot=spot, direction=direction)
    result = _route(make(), product, env)
    dist = result.event_distribution
    legs = _leg_set(dist.event_times)
    overlay = fold_cash_legs(legs, product, env, quantity=1.0, unit_notional=100.0)
    folded = make().price_folded(product, env, overlay) + overlay.constant_pv
    expected = result.npv + sum(leg.value(dist, env, 100.0) for leg in legs)
    assert abs(folded - expected) <= PARITY * product.notional


# --- the valuation-instant decisions (D13) ------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(ENGINES))
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("d, spot, outcome", [
    (OBS, 6300.0, "ko"),            # a knock-out today ends the contract
    (FINAL, 6300.0, "ko"),          # the final row knocks out
    (FINAL, 4000.0, "ki"),          # the final row is knocked in: the loss leg
    (FINAL, 5500.0, "no_ki"),       # the final row survives, never knocked in
])
def test_a_decided_valuation_row_pays_its_events_cash(name, direction, d, spot, outcome):
    make, product, env = _case(name, d=d, spot=spot, direction=direction)
    s = _event_structure(product, env)
    overlay = _overlay(product, env, ko=np.arange(1.0, len(s.ko) + 1.0) * C, no_ki=2.5 * C, ki=4.0 * C)
    paid = {"ko": (overlay.ko[0].amount, s.ko[0].payment_time),
            "no_ki": (2.5 * C, s.no_ki.payment_time), "ki": (4.0 * C, s.ki.payment_time)}[outcome]
    expected = product.direction_sign * paid[0] * env.get_discount_factor(paid[1])
    base = make().price(product, env)
    assert make().price_folded(product, env, overlay) - base == pytest.approx(expected, rel=IDENTITY)


# --- lifecycle states ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(ENGINES))
def test_a_knocked_out_state_has_no_event_to_fold_onto(name):
    make, grid = ENGINES[name]
    state = state_on(STRADDLE, flow("ko", ROW, ROW_PAY, CASH), knocked_out=True, ko_date=ROW)
    product, env = make_dcn(DCN_A, valuation_date=STRADDLE), env_on(STRADDLE, grid=grid)
    with pytest.raises(CapabilityError):
        make().price_folded(product, env, _identity_overlay(product, env), lifecycle_state=state)
    empty = _overlay(product, env)
    assert make().price_folded(product, env, empty, lifecycle_state=state) == \
        make().price(product, env, lifecycle_state=state)


@pytest.mark.parametrize("name", sorted(ENGINES))
def test_an_alive_state_adds_its_receivables_beside_the_fold(name):
    make, grid = ENGINES[name]
    state = state_on(STRADDLE, flow("coupon", ROW, ROW_PAY, CASH))
    product, env = make_dcn(DCN_A, valuation_date=STRADDLE), env_on(STRADDLE, spot=5800.0, grid=grid)
    overlay = _identity_overlay(product, env)
    with_state = make().price_folded(product, env, overlay, lifecycle_state=state) - \
        make().price(product, env, lifecycle_state=state)
    without = make().price_folded(product, env, overlay) - make().price(product, env)
    assert with_state == pytest.approx(without, rel=IDENTITY)


# --- AC6: DCN PDE vs the same-model DCN MC event route ---------------------------------------------------------------

AC6_PATHS = 1 << 16
AC6 = {
    "pde": (lambda: DCNPDEEngine(PDEParams(accuracy="standard")),
            lambda: DCNMCEngine(num_paths=AC6_PATHS, seed=11, use_sobol=False), False),
    "lv_pde": (lambda: LocalVolDCNPDEEngine(params=PDEParams(accuracy="standard")),
               lambda: LocalVolDCNMCEngine(num_paths=AC6_PATHS, seed=11, use_sobol=False), True),
    "heston_pde": (lambda: HestonDCNPDESolver(HESTON, n_x=201, n_v=61),
                   lambda: HestonDCNMCEngine(model_params=HESTON, num_paths=AC6_PATHS, seed=11, use_sobol=False),
                   False),
}


def _leg_cash_se(overlay, dist, env, unit, n):
    """Standard error of the per-path folded leg cash ``unit · amount_e · DF(pay_e)`` on ``n`` iid paths."""
    p = list(np.asarray(dist.probabilities[EventType.KO], dtype=float))
    x = [unit * e.amount * env.get_discount_factor(e.payment_time) for e in overlay.ko]
    for et, e in ((EventType.MATURITY_NO_KO, overlay.maturity_no_ki), (EventType.MATURITY_WITH_KI, overlay.maturity_ki)):
        p.append(float(dist.probabilities[et]))
        x.append(unit * e.amount * env.get_discount_factor(e.payment_time))
    p, x = np.asarray(p), np.asarray(x)
    mean = float(np.sum(p * x))
    return float(np.sqrt(max(float(np.sum(p * x * x)) - mean * mean, 0.0) / n))


@pytest.mark.parametrize("name", sorted(AC6))
@pytest.mark.parametrize("contract", [DCN_A, DCN_B])
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_pde_folded_legs_match_the_mc_event_route(name, contract, direction):
    make_pde, make_mc, grid = AC6[name]
    product, env = make_dcn(contract, direction=direction), env_on(START, grid=grid)
    dist = _route(make_mc(), product, env).event_distribution
    legs = _leg_set(dist.event_times)
    quantity = 1.0
    overlay = fold_cash_legs(legs, product, env, quantity=quantity, unit_notional=100.0)
    pde = make_pde()
    folded_legs = quantity * (pde.price_folded(product, env, overlay) - pde.price(product, env)) + overlay.constant_pv
    route_legs = sum(leg.value(dist, env, 100.0) for leg in legs)
    se = _leg_cash_se(overlay, dist, env, quantity * product.direction_sign, AC6_PATHS)
    assert abs(folded_legs - route_legs) <= SE * se


@pytest.mark.parametrize("name", sorted(AC6))
def test_pde_march_weights_match_the_mc_event_route(name):
    make_pde, make_mc, grid = AC6[name]
    product, env = make_dcn(DCN_A), env_on(START, grid=grid)
    dist = _route(make_mc(), product, env).event_distribution
    _, w_ko, w_no_ki, w_ki = _march_weights(make_pde, product, env)
    for w, p in ((w_ko, float(np.sum(dist.probabilities[EventType.KO]))),
                 (w_no_ki, float(dist.probabilities[EventType.MATURITY_NO_KO])),
                 (w_ki, float(dist.probabilities[EventType.MATURITY_WITH_KI]))):
        assert abs(w - p) <= SE * np.sqrt(p * (1.0 - p) / AC6_PATHS)


# --- AC14: the knock-out coupon twin --------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(ENGINES))
@pytest.mark.parametrize("contract", [DCN_A, DCN_B])
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_knock_out_coupon_twin(name, contract, direction):
    delta = 0.015
    make, product, env = _case(name, contract=contract, direction=direction)
    twin = make_dcn(contract, direction=direction, ko_coupon_rate=contract["ko_coupon_rate"] + delta)
    folded = make().price_folded(product, env, _twin_overlay(product, delta))
    target = make().price(twin, env)
    assert folded == pytest.approx(target, rel=IDENTITY)


@pytest.mark.parametrize("name", sorted(ENGINES))
@pytest.mark.parametrize("d, spot", [(SAT, 5500.0), (OBS, 5500.0), (OBS, 6300.0), (FINAL, 6300.0)])
def test_knock_out_coupon_twin_on_closed_row_and_final_days(name, d, spot):
    delta = 0.015
    make, product, env = _case(name, d=d, spot=spot)
    twin = make_dcn(DCN_A, valuation_date=d, ko_coupon_rate=DCN_A["ko_coupon_rate"] + delta)
    folded = make().price_folded(product, env, _twin_overlay(product, delta))
    assert folded == pytest.approx(make().price(twin, env), rel=IDENTITY)


# --- AC9: caches, D11c ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(ENGINES))
def test_folded_and_plain_solves_never_share_a_cached_answer(name):
    make, product, env = _case(name)
    a, b = _identity_overlay(product, env), _identity_overlay(product, env, cash=3.0 * C)
    engine = make()
    seq = [engine.price_folded(product, env, a), engine.price(product, env),
           engine.price_folded(product, env, b), engine.price_folded(product, env, a)]
    fresh = [make().price_folded(product, env, a), make().price(product, env),
             make().price_folded(product, env, b), make().price_folded(product, env, a)]
    assert seq == fresh
    assert len({seq[0], seq[1], seq[2]}) == 3


@pytest.mark.parametrize("name", sorted(MC))
def test_dcn_mc_drilldown_is_sound_on_every_batch_layout(name):
    # batches are seeded by batch id and reduced in batch order: the path set never depends on the payoff
    make, product, env = _case(name)
    assert make().fold_drilldown_refusal(product, env) is None


# --- the 1-D PDE injection sites, one by one ------------------------------------------------------------------------

def _prepared(engine, product, env, overlay=None):
    from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, prepare

    return prepare(engine, product, env, OutputRequest(Operation.PRICE, fold=overlay))


def _hand(product, env):
    s = _event_structure(product, env)
    return _overlay(product, env, ko=np.arange(1.0, len(s.ko) + 1.0) * C, no_ki=2.5 * C, ki=4.0 * C)


@pytest.mark.parametrize("name", ["pde", "pde_nodal", "lv_pde"])
def test_pde_edge_rows_carry_the_folded_cash(name):
    make, product, env = _case(name)
    overlay = _hand(product, env)
    plain, folded = _prepared(make(), product, env).events, _prepared(make(), product, env, overlay).events
    ctx, df = plain.ctx, plain.df
    obs = ctx.times[ctx.obs_cols]
    t = 0.5 * float(obs[3] + obs[4])
    j = int(np.argmax(obs > t))                                   # the next knock-out observation
    (a_lo, k_lo), (a_up, k_up) = plain.rows(t)
    (fa_lo, fk_lo), (fa_up, fk_up) = folded.rows(t)
    settle = df(ctx.loss_pay_time) / df(t)
    assert fa_lo - a_lo == pytest.approx(4.0 * C * settle, rel=1e-12)
    assert fk_lo - k_lo == pytest.approx(4.0 * C * settle, rel=1e-12)
    ko_cash = overlay.ko[j].amount * df(ctx.ko_pay_times[j]) / df(t)
    assert fa_up - a_up == pytest.approx(ko_cash, rel=1e-12)
    assert fk_up - k_up == pytest.approx(ko_cash, rel=1e-12)
    # no knock-out left after the horizon: each edge survives on its own branch to the loss-leg settlement
    T = plain.T
    (_, _), (a_up, k_up) = plain.rows(T)
    (_, _), (fa_up, fk_up) = folded.rows(T)
    settle = df(ctx.loss_pay_time) / df(T)
    assert (fa_up - a_up, fk_up - k_up) == pytest.approx((2.5 * C * settle, 4.0 * C * settle), rel=1e-12)


@pytest.mark.parametrize("name", ["pde", "pde_nodal", "lv_pde"])
def test_pde_terminal_columns_carry_the_folded_cash(name):
    make, product, env = _case(name)
    overlay = _hand(product, env)

    def terminal(ov):
        problem = _prepared(make(), product, env, ov)
        n_x, n_t = len(problem.layout.spatial.x), len(problem.layout.time.t)
        states = {"alive": np.zeros((n_x, n_t)), "ki": np.zeros((n_x, n_t))}
        problem.adapter.terminal(problem, states, None)
        return problem.events, states["alive"][:, -1], states["ki"][:, -1]

    ev, alive, ki = terminal(None)
    _, f_alive, f_ki = terminal(overlay)
    s, ctx, df, T = ev.s, ev.ctx, ev.df, ev.T
    settle = df(ctx.loss_pay_time) / df(T)
    ko_cash = overlay.ko[-1].amount * df(ctx.ko_pay_times[-1]) / df(T)
    below_ki = s < 0.9 * product.ki_barrier
    between = (s > 1.02 * product.ki_barrier) & (s < 0.98 * product.coupon_barrier)
    above_ko = s > 1.05 * product.ko_barrier
    assert below_ki.any() and between.any() and above_ko.any()
    np.testing.assert_allclose((f_ki - ki)[below_ki | between], 4.0 * C * settle, rtol=1e-12)
    np.testing.assert_allclose((f_alive - alive)[below_ki], 4.0 * C * settle, rtol=1e-12)
    np.testing.assert_allclose((f_alive - alive)[between], 2.5 * C * settle, rtol=1e-12)
    np.testing.assert_allclose((f_alive - alive)[above_ko], ko_cash, rtol=1e-12)
    np.testing.assert_allclose((f_ki - ki)[above_ko], ko_cash, rtol=1e-12)


def test_pde_problem_identity_carries_the_overlay():
    make, product, env = _case("pde")
    a, b = _identity_overlay(product, env), _identity_overlay(product, env, cash=3.0 * C)
    plain = _prepared(make(), product, env).identity
    assert plain == _prepared(make(), product, env).identity
    assert len({repr(plain), repr(_prepared(make(), product, env, a).identity),
                repr(_prepared(make(), product, env, b).identity)}) == 3
