"""W14a: every DCN route prices a supplied ``DCNLifecycleState`` (spec I1, C1/A5, C2, AC1-AC3, A1; T12).

The state carries what the run has already decided: realized coupon / knock-out / loss cash in its
``LifecycleCashflowLedger`` (amounts SIGNED like the PV, i.e. ``direction_sign`` applied), the knock-in flag
(which must equal ``product.knocked_in_at_valuation``) and the terminal flags. An engine prices

* ``lifecycle_state=None``: bitwise as before (AC1; pinned by the T12 capture and the existing DCN suites);
* a knocked-out or matured state: exactly ``ledger.pending_pv`` (the engine is not run);
* any other state: the engine's value + the ledger's pending receivable PV.

Double-count guard (A1): a ledger flow determined after the valuation date, or on a date that still has a monthly row
in the product's schedule, is refused. The post-close view of day d is ``schedule.without_observed_through(d)``: the
rows and daily-KI dates <= d are gone and the valuation instant is no longer a monitoring date. AC2b: pre-close value
== post-close value (engine + ledger PV).

The ledgers here are HAND-BUILT (the tracker that builds them is T13).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from dcn_fixtures import DCN_A, FLAT, make_dcn

from quantark.asset.equity.engine.mc import DCNMCEngine
from quantark.asset.equity.engine.mc.dcn_vol_mc_engines import (
    CoupledCoarseHestonDCNMCEngine,
    HestonDCNMCEngine,
    LocalVolDCNMCEngine,
    QEDCNMCEngine,
)
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import HestonDCNPDESolver, LocalVolDCNPDEEngine
from quantark.asset.equity.engine.pde.grid.config import GridConfig
from quantark.asset.equity.lifecycle import (
    AutocallableLifecycleState,
    DCNLifecycleState,
    LifecycleCashflowLedger,
    LifecycleEventType,
    RealizedCashflow,
    ValuationPoint,
)
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
from quantark.asset.equity.product.option.dcn_option import DCNDirection, DCNOption
from quantark.execution import PricingRequest, PricingSession
from quantark.execution.contracts import OutputKind, PricingOperation
from quantark.param import ContinuousDividendYield, FlatRateCurve, GridVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import DayCountConvention
from quantark.execution.errors import CapabilityError
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams

R, Q, SIGMA = FLAT["r"], FLAT["q"], FLAT["sigma"]
HESTON = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)
ROW = datetime(2024, 6, 3)          # DCN_A month 17: coupon + KO row, paid 2024-06-05
ROW_PAY = datetime(2024, 6, 5)
STRADDLE = datetime(2024, 6, 4)     # the 2024-06-03 cash is determined but not yet paid
FINAL = datetime(2025, 1, 3)
FINAL_PAY = datetime(2025, 1, 7)    # DCN_A settlement_date (the loss leg)
CASH = 10000.0                      # participation 1 x 12% x 30/360 x 1e6 (coupon and KO coupon alike)
_CTOR = (
    "notional", "initial_price", "direction", "coupon_barrier_ratio", "ko_barrier_ratio", "ki_barrier_ratio",
    "ki_put_strike_ratio", "coupon_rate", "ko_coupon_rate", "participation", "coupon_counted_days",
    "coupon_days_denom", "schedule", "settlement_date", "knocked_in_at_valuation", "settlement_convention",
)


def rebuild(product: DCNOption, **changes) -> DCNOption:
    kw = {k: getattr(product, k) for k in _CTOR}
    kw.update(changes)
    return DCNOption(**kw)


def booked(d, **overrides) -> DCNOption:
    return make_dcn(DCN_A, valuation_date=d, **overrides)


def post_close(product: DCNOption, d, knocked_in: bool) -> DCNOption:
    return rebuild(product, schedule=product.schedule.without_observed_through(d), knocked_in_at_valuation=knocked_in)


def env_on(d, spot=6000.0, grid=False):
    vol = None
    if grid:
        strikes = [3000.0, 4500.0, 6000.0, 7500.0, 9000.0]
        mats = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
        base = SIGMA + 0.05 * (6000.0 - np.array(strikes)) / 6000.0
        vol = GridVolSurface(strikes=strikes, maturities=mats, iv_grid=np.tile(base, (len(mats), 1)))
    else:
        from quantark.param import FlatVolSurface
        vol = FlatVolSurface(volatility=SIGMA)
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=vol, rate_curve=FlatRateCurve(rate=R),
        div_yield=ContinuousDividendYield(div_yield=Q), valuation_date=d,
        day_count_convention=DayCountConvention.ACT_365)


def flow(kind, determined, paid, amount):
    event = {"coupon": LifecycleEventType.COUPON, "ko": LifecycleEventType.KNOCK_OUT,
             "loss": LifecycleEventType.MATURITY}[kind]
    return RealizedCashflow(cashflow_id=f"dcn:{kind}:{determined:%Y-%m-%d}", event_type=event, amount=amount,
                            determination_date=determined, payment_date=paid)


def state_on(d, *flows, **flags) -> DCNLifecycleState:
    return DCNLifecycleState(valuation_point=ValuationPoint(date=d), ledger=LifecycleCashflowLedger(flows), **flags)


MC = dict(num_paths=4096, seed=7, use_sobol=True, num_batches=2, num_workers=1)
ROUTES = {
    "bsm_mc": (lambda: DCNMCEngine(**MC), False),
    "bsm_pde": (lambda: DCNPDEEngine(PDEParams(accuracy="fast")), False),
    "lv_mc": (lambda: LocalVolDCNMCEngine(**MC), True),
    "lv_pde": (lambda: LocalVolDCNPDEEngine(params=PDEParams(grid=GridConfig(points=301))), True),
    "heston_mc": (lambda: HestonDCNMCEngine(model_params=HESTON, num_paths=2048, seed=7, num_workers=1), False),
    "heston_pde": (lambda: HestonDCNPDESolver(HESTON, n_x=101, n_v=31), False),
}


# --- C1 / C2 -----------------------------------------------------------------------------------------------------

def test_state_fields_and_protocol():
    s = state_on(STRADDLE, flow("coupon", ROW, ROW_PAY, CASH))
    assert s.alive and not s.knocked_in and not s.knocked_out and not s.matured
    assert s.ko_date is None and s.ki_date is None and s.observed_row_indices == set()
    assert s.realized_cashflows == 0.0                      # 2024-06-05 not yet paid on 2024-06-04
    assert state_on(datetime(2024, 6, 5), flow("coupon", ROW, ROW_PAY, CASH)).realized_cashflows == CASH
    assert not DCNLifecycleState(knocked_out=True).alive
    assert not DCNLifecycleState(matured=True).alive


@pytest.mark.parametrize("cls", [DCNMCEngine, LocalVolDCNMCEngine, HestonDCNMCEngine, QEDCNMCEngine,
                                 CoupledCoarseHestonDCNMCEngine, DCNPDEEngine, LocalVolDCNPDEEngine,
                                 HestonDCNPDESolver])
def test_every_dcn_route_declares_state_support(cls):
    assert cls.supports_lifecycle_state is True


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_none_state_keyword_is_the_stateless_price(route):
    make, grid = ROUTES[route]
    p, env = booked(STRADDLE), env_on(STRADDLE, grid=grid)
    assert make().price(p, env, lifecycle_state=None) == make().price(p, env)


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_other_state_types_are_capability_errors(route):
    make, grid = ROUTES[route]
    with pytest.raises(CapabilityError):
        make().price(booked(STRADDLE), env_on(STRADDLE, grid=grid), lifecycle_state=AutocallableLifecycleState())


# --- AC2 with a hand-built state: engine + pending receivable PV, bitwise -----------------------------------------

@pytest.mark.parametrize("route", sorted(ROUTES))
def test_alive_state_adds_the_pending_receivables_bitwise(route):
    make, grid = ROUTES[route]
    p, env = booked(STRADDLE), env_on(STRADDLE, grid=grid)
    s = state_on(STRADDLE, flow("coupon", datetime(2024, 5, 6), datetime(2024, 5, 8), CASH),   # paid already
                 flow("coupon", ROW, ROW_PAY, CASH))                                         # straddles the date
    pending = s.ledger.pending_pv(s.valuation_point, env)
    assert pending == pytest.approx(CASH * np.exp(-R / 365.0), rel=1e-14)
    assert make().price(p, env, lifecycle_state=s) == make().price(p, env) + pending


@pytest.mark.parametrize("route", ["bsm_mc", "bsm_pde"])
@pytest.mark.parametrize("spot", [5500.0, 6100.0])          # a coupon today / a knock-out today
def test_alive_state_on_a_row_day_keeps_the_engines_row(route, spot):
    make, _ = ROUTES[route]
    d = datetime(2024, 7, 3)                                 # the row after ROW; ROW's cash paid 2024-06-05
    p, env = booked(d), env_on(d, spot)
    s = state_on(d, flow("coupon", ROW, ROW_PAY, CASH))
    assert s.ledger.pending_pv(s.valuation_point, env) == 0.0
    assert make().price(p, env, lifecycle_state=s) == make().price(p, env) + 0.0


def test_mc_detailed_result_reports_the_receivables():
    p, env = booked(STRADDLE), env_on(STRADDLE)
    s = state_on(STRADDLE, flow("coupon", ROW, ROW_PAY, CASH))
    base = DCNMCEngine(**MC).price_detailed(p, env)
    got = DCNMCEngine(**MC).price_detailed(p, env, lifecycle_state=s)
    pending = s.ledger.pending_pv(s.valuation_point, env)
    assert got.pv_pending_receivables == pending and base.pv_pending_receivables == 0.0
    assert got.pv == base.pv + pending
    assert (got.pv_fixed_coupons, got.pv_ko_coupons, got.pv_loss_leg, got.std_error) == (
        base.pv_fixed_coupons, base.pv_ko_coupons, base.pv_loss_leg, base.std_error)


@pytest.mark.parametrize("route", ["bsm_pde", "lv_pde"])
def test_pde_greeks_carry_the_receivables_in_price_only(route):
    make, grid = ROUTES[route]
    p, env = booked(STRADDLE), env_on(STRADDLE, grid=grid)
    s = state_on(STRADDLE, flow("coupon", ROW, ROW_PAY, CASH))
    base = make().calculate_greeks(p, env)
    got = make().calculate_greeks(p, env, lifecycle_state=s)
    assert got["price"] == base["price"] + s.ledger.pending_pv(s.valuation_point, env)
    assert (got["delta"], got["gamma"]) == (base["delta"], base["gamma"])


# --- AC3: terminal states, conflicts, the double-count guard -----------------------------------------------------

@pytest.mark.parametrize("route", sorted(ROUTES))
def test_knocked_out_state_prices_exactly_the_pending_receivable(route):
    make, grid = ROUTES[route]
    s = state_on(STRADDLE, flow("ko", ROW, ROW_PAY, CASH), knocked_out=True, ko_date=ROW)
    env = env_on(STRADDLE, grid=grid)
    assert make().price(booked(STRADDLE), env, lifecycle_state=s) == s.ledger.pending_pv(s.valuation_point, env)
    paid_day = datetime(2024, 6, 5)
    s_paid = state_on(paid_day, flow("ko", ROW, ROW_PAY, CASH), knocked_out=True, ko_date=ROW)
    assert make().price(booked(paid_day), env_on(paid_day, grid=grid), lifecycle_state=s_paid) == 0.0


def test_terminal_mc_result_runs_no_simulation():
    s = state_on(STRADDLE, flow("ko", ROW, ROW_PAY, CASH), knocked_out=True, ko_date=ROW)
    env = env_on(STRADDLE)
    r = DCNMCEngine(**MC).price_detailed(booked(STRADDLE), env, lifecycle_state=s)
    assert r.pv == r.pv_pending_receivables == s.ledger.pending_pv(s.valuation_point, env)
    assert r.num_paths == 0 and r.std_error == 0.0
    assert (r.pv_fixed_coupons, r.pv_ko_coupons, r.pv_loss_leg) == (0.0, 0.0, 0.0)


def test_terminal_pde_greeks_are_flat():
    s = state_on(STRADDLE, flow("ko", ROW, ROW_PAY, CASH), knocked_out=True, ko_date=ROW)
    env = env_on(STRADDLE)
    g = DCNPDEEngine(PDEParams(accuracy="fast")).calculate_greeks(booked(STRADDLE), env, lifecycle_state=s)
    assert g == {"price": s.ledger.pending_pv(s.valuation_point, env), "delta": 0.0, "gamma": 0.0}


@pytest.mark.parametrize("route", sorted(ROUTES))
def test_knocked_in_flag_conflict_is_refused(route):
    make, grid = ROUTES[route]
    env = env_on(STRADDLE, grid=grid)
    with pytest.raises(ValidationError, match="knocked_in"):
        make().price(booked(STRADDLE), env, lifecycle_state=state_on(STRADDLE, knocked_in=True))
    with pytest.raises(ValidationError, match="knocked_in"):
        make().price(booked(STRADDLE, knocked_in_at_valuation=True), env, lifecycle_state=state_on(STRADDLE))


@pytest.mark.parametrize("route", ["bsm_mc", "bsm_pde", "heston_pde"])
def test_double_count_guard(route):
    make, _ = ROUTES[route]
    env = env_on(ROW, 5500.0)
    pre = booked(ROW)
    today = flow("coupon", ROW, ROW_PAY, CASH)
    with pytest.raises(ValidationError, match="row"):                       # row 2024-06-03 still in the schedule
        make().price(pre, env, lifecycle_state=state_on(ROW, today))
    later = flow("coupon", datetime(2024, 7, 3), datetime(2024, 7, 5), CASH)
    with pytest.raises(ValidationError, match="after the valuation date"):
        make().price(post_close(pre, ROW, False), env, lifecycle_state=state_on(ROW, later))
    make().price(post_close(pre, ROW, False), env, lifecycle_state=state_on(ROW, today))   # post-close: accepted


def test_state_must_be_dated_on_the_valuation_date():
    env = env_on(STRADDLE)
    with pytest.raises(ValidationError, match="valuation"):
        DCNMCEngine(**MC).price(booked(STRADDLE), env, lifecycle_state=state_on(ROW))
    timed = DCNLifecycleState(valuation_point=ValuationPoint(time=0.0))
    with pytest.raises(ValidationError, match="date"):
        DCNMCEngine(**MC).price(booked(STRADDLE), env, lifecycle_state=timed)
    numeric_flow = RealizedCashflow(cashflow_id="c", event_type=LifecycleEventType.COUPON, amount=1.0,
                                    determination_time=0.1, payment_time=0.2)
    with pytest.raises(ValidationError, match="date"):
        DCNMCEngine(**MC).price(booked(STRADDLE), env, lifecycle_state=state_on(STRADDLE, numeric_flow))


# --- A1: the post-close schedule view ---------------------------------------------------------------------------

def test_post_close_schedule_drops_the_observed_day():
    pre = booked(ROW).schedule
    post = pre.without_observed_through(ROW)
    assert post.monthly == pre.monthly[1:]
    assert post.daily_ki_dates == pre.daily_ki_dates[1:]
    assert post.valuation_date == ROW and post.spec == pre.spec and post.calendar is pre.calendar
    with pytest.raises(ValidationError):
        pre.without_observed_through(STRADDLE)                    # only the schedule's own valuation date
    with pytest.raises(ValidationError, match="matured"):
        booked(FINAL).schedule.without_observed_through(FINAL)    # nothing left: price the matured state


def test_post_close_grid_starts_at_an_unmonitored_valuation_instant():
    pre_ctx = build_dcn_grid_context(booked(ROW))
    post_ctx = build_dcn_grid_context(post_close(booked(ROW), ROW, False))
    assert pre_ctx.valuation_monitored and not post_ctx.valuation_monitored
    np.testing.assert_array_equal(post_ctx.times, pre_ctx.times)        # same instants, so same MC draws
    np.testing.assert_array_equal(post_ctx.obs_cols, pre_ctx.obs_cols[1:])
    np.testing.assert_array_equal(post_ctx.coupon_pay_times, pre_ctx.coupon_pay_times[1:])


# --- AC2b: pre-close == post-close (engine + ledger) ------------------------------------------------------------

def _signed(product, cash):
    return product.direction_sign * cash


AC2B_DAYS = {                       # (day, spot)
    "coupon": (ROW, 5500.0),
    "knock_out": (ROW, 6100.0),
    "knock_in": (STRADDLE, 4300.0),
    "final_coupon": (FINAL, 5500.0),
    "final_loss": (FINAL, 4300.0),
}


def _post_close_view(case, direction):
    d, spot = AC2B_DAYS[case]
    pre = booked(d, direction=direction)
    sign = pre.direction_sign
    knocked_in = spot <= pre.ki_barrier
    flags = dict(knocked_in=knocked_in, ki_date=d if knocked_in else None)
    if case == "coupon":
        return pre, post_close(pre, d, False), state_on(d, flow("coupon", d, ROW_PAY, sign * CASH), **flags)
    if case == "knock_out":
        s = state_on(d, flow("ko", d, ROW_PAY, sign * CASH), knocked_out=True, ko_date=d, **flags)
        return pre, post_close(pre, d, False), s
    if case == "knock_in":
        return pre, post_close(pre, d, True), state_on(d, **flags)
    if case == "final_coupon":
        s = state_on(d, flow("coupon", d, datetime(2025, 1, 7), sign * CASH), matured=True, **flags)
        return pre, pre, s                 # nothing is left after the final row: the matured state prices alone
    loss = -(pre.notional / pre.initial_price) * pre.participation * (pre.k_loss - spot)
    s = state_on(d, flow("loss", d, FINAL_PAY, sign * loss), matured=True, **flags)
    # knocked in TODAY on the final row: the post-close copy carries the flag the state carries
    return pre, booked(d, direction=direction, knocked_in_at_valuation=True), s


@pytest.mark.parametrize("direction", [DCNDirection.BUYER, DCNDirection.SELLER])
@pytest.mark.parametrize("case", sorted(AC2B_DAYS))
@pytest.mark.parametrize("route", ["bsm_mc", "bsm_pde"])
def test_pre_close_equals_post_close(route, case, direction):
    make, _ = ROUTES[route]
    d, spot = AC2B_DAYS[case]
    env = env_on(d, spot)
    pre_product, post_product, s = _post_close_view(case, direction)
    pre = make().price(pre_product, env)
    post = make().price(post_product, env, lifecycle_state=s)
    assert post == pytest.approx(pre, rel=1e-10, abs=1e-8)


# --- execution sessions: the state reaches every DCN adapter ---------------------------------------------------

SESSION_ENGINES = {
    "bsm_mc": (lambda: DCNMCEngine(**MC), False),
    "lv_mc": (lambda: LocalVolDCNMCEngine(**MC), True),
    "heston_mc": (lambda: HestonDCNMCEngine(model_params=HESTON, num_paths=2048, seed=7, num_workers=1), False),
    "qe_mc": (lambda: QEDCNMCEngine(model_params=HESTON, num_paths=2048, seed=7, num_workers=1), False),
    "bsm_pde": (lambda: DCNPDEEngine(PDEParams(accuracy="fast")), False),
    "lv_pde": (lambda: LocalVolDCNPDEEngine(params=PDEParams(grid=GridConfig(points=301))), True),
}


def _session(engine, product, env, state, operation=PricingOperation.PRICE):
    with PricingSession() as s:
        return s.execute(engine, PricingRequest(product=product, pricing_env=env, operation=operation,
                                                lifecycle_state=state)).value


@pytest.mark.parametrize("route", sorted(SESSION_ENGINES))
@pytest.mark.parametrize("terminal", [False, True])
def test_session_price_equals_direct_with_a_state(route, terminal):
    make, grid = SESSION_ENGINES[route]
    p, env = booked(STRADDLE), env_on(STRADDLE, grid=grid)
    s = (state_on(STRADDLE, flow("ko", ROW, ROW_PAY, CASH), knocked_out=True, ko_date=ROW) if terminal
         else state_on(STRADDLE, flow("coupon", ROW, ROW_PAY, CASH)))
    assert _session(make(), p, env, s) == make().price(p, env, lifecycle_state=s)


@pytest.mark.parametrize("route", ["bsm_mc", "lv_mc", "heston_mc"])
@pytest.mark.parametrize("terminal", [False, True])
def test_session_detailed_equals_direct_with_a_state(route, terminal):
    make, grid = SESSION_ENGINES[route]
    p, env = booked(STRADDLE), env_on(STRADDLE, grid=grid)
    s = (state_on(STRADDLE, flow("ko", ROW, ROW_PAY, CASH), knocked_out=True, ko_date=ROW) if terminal
         else state_on(STRADDLE, flow("coupon", ROW, ROW_PAY, CASH)))
    got = _session(make(), p, env, s, PricingOperation.PRICE_DETAILED)
    want = make().price_detailed(p, env, lifecycle_state=s)
    for name in type(want).__dataclass_fields__:
        if name not in ("elapsed_seconds", "event_stats"):
            assert getattr(got, name) == getattr(want, name), name


@pytest.mark.parametrize("terminal", [False, True])
def test_session_state_runs_on_the_batch_adapter(terminal):
    """Alive: the batch plan (pending added in the reduction); terminal: answered natively, no plan."""
    s = (state_on(STRADDLE, flow("ko", ROW, ROW_PAY, CASH), knocked_out=True, ko_date=ROW) if terminal
         else state_on(STRADDLE, flow("coupon", ROW, ROW_PAY, CASH)))
    with PricingSession() as sess:
        out = sess.execute(DCNMCEngine(**MC), PricingRequest(product=booked(STRADDLE), pricing_env=env_on(STRADDLE),
                                                             lifecycle_state=s))
    assert out.manifest.adapter_id == "dcn-batch-mc"
    assert (out.manifest.plan_fingerprint is None) is terminal


def test_session_final_observation_with_a_state():
    p, env = booked(FINAL), env_on(FINAL, 5500.0)
    s = state_on(FINAL, flow("coupon", datetime(2024, 12, 3), datetime(2024, 12, 5), CASH))   # paid already
    assert _session(DCNMCEngine(**MC), p, env, s) == DCNMCEngine(**MC).price(p, env, lifecycle_state=s)


def test_session_refuses_a_bad_state():
    p, env = booked(STRADDLE), env_on(STRADDLE)
    with pytest.raises(ValidationError, match="knocked_in"):
        _session(DCNMCEngine(**MC), p, env, state_on(STRADDLE, knocked_in=True))
    with pytest.raises(ValidationError, match="knocked_in"):
        _session(DCNPDEEngine(PDEParams(accuracy="fast")), p, env, state_on(STRADDLE, knocked_in=True))


def test_session_grid_rows_carry_the_receivables():
    p, env = booked(STRADDLE), env_on(STRADDLE)
    s = state_on(STRADDLE, flow("coupon", ROW, ROW_PAY, CASH))
    levels = (("grid_spot_levels", (5000.0, 6000.0)),)

    def run(state):
        with PricingSession() as sess:
            return sess.execute(DCNPDEEngine(PDEParams(accuracy="fast")), PricingRequest(
                product=p, pricing_env=env, outputs=frozenset([OutputKind.PV, OutputKind.GRID]),
                operation_options=levels, lifecycle_state=state)).value

    base, got = run(None), run(s)
    pending = s.ledger.pending_pv(s.valuation_point, env)
    assert got.pv == base.pv + pending
    for b, g in zip(base.grid, got.grid):
        assert g["price"] == b["price"] + pending and g["delta"] == b["delta"]
