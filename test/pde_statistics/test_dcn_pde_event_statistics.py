"""DCN PDE event statistics on the unified ``EventStatistics`` (spec T15).

``DCNPDEEngine``, ``LocalVolDCNPDEEngine`` (the 1D core) and ``HestonDCNPDESolver`` (its own 2D ADI loop) produce
every outcome from the exact forward adjoint of the price's own march (``pde/dcn_statistics.py``):

- AC7: ``EventStatistics`` with a coupon table, a knock-in table and nothing unavailable but the undefined;
- AC8: the reconciliation at round-off, the probability residual within the measured bound, shrinking under
  refinement;
- AC13: the per-date knock-in table on the daily KI dates (exact dates), sum = P(KI), mean = E[KI time];
- AC14: the forward pass reproduces the backward march of every unit claim (each KO row, each coupon row, the
  maturity, the KI settlement and its loss cash) to round-off, and the backward claim march reproduces the price;
- AC10 / AC17 / AC18: valuation-date rules, receivables and terminated states, seller signs, the environment's axis
  with exact dates on ACT/360 and BUSINESS_DAYS environments (0.14.0; D12 retired), the event route equal to the fold
  route;
- ``price()``, ``calculate_event_stats(...).pv`` and ``price_with_events(...).npv`` bitwise equal (D23).
"""
from __future__ import annotations

import warnings
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.event_stats import EventStatistics
from quantark.asset.equity.engine.mc.dcn_mc_engine import DCNMCEngine
from quantark.asset.equity.engine.pde.dcn_pde_solver import DCNPDEEngine
from quantark.asset.equity.engine.pde.dcn_statistics import backward_claims_1d, backward_claims_adi
from quantark.asset.equity.engine.pde.dcn_vol_pde_solvers import HestonDCNPDESolver, LocalVolDCNPDEEngine
from quantark.asset.equity.lifecycle import DCNLifecycleState, ValuationPoint
from quantark.asset.equity.lifecycle.cashflows import LifecycleCashflowLedger, RealizedCashflow
from quantark.asset.equity.lifecycle.events import LifecycleEventType
from quantark.asset.equity.engine.pde.grid.config import GridConfig
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.dcn_grid import build_dcn_grid_context
from quantark.asset.equity.product.option.dcn_option import DCNDirection
from quantark.cashleg import EventType, FixedPayoffLeg, LegDirection, PaymentTrigger, fold_cash_legs
from quantark.param import ContinuousDividendYield, FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.priceenv.date_axis import env_times
from quantark.priceenv.term_sampling import make_df_fn
from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
from quantark.util.numerical import is_close
from quantark.volmodels.heston import HestonParams

from dcn_fixtures import DCN_A, DCN_B, FLAT, make_dcn

warnings.filterwarnings("ignore", category=DeprecationWarning)

SSE = create_calendar(CalendarType.CHINA_SSE)
SAT = datetime(2024, 6, 8)           # SSE closed: an unmonitored valuation instant
HOLIDAY = datetime(2024, 6, 10)      # SSE holiday on a Monday
DAY = datetime(2024, 5, 15)          # a trading day that is not a monthly row
OBS = datetime(2024, 12, 3)          # DCN_A month 23: a coupon + knock-out row on the valuation date
FINAL = datetime(2025, 1, 3)         # DCN_A's final observation
LOCK = datetime(2023, 3, 15)         # DCN_B inside its knock-out lock (coupon-only rows ahead)
ROW, ROW_PAY, AGED, SETTLED = datetime(2024, 6, 3), datetime(2024, 6, 5), datetime(2024, 6, 4), datetime(2024, 6, 6)
CASH = 10_000.0
HESTON = HestonParams(v0=0.04, kappa=1.0, theta=0.04, sigma=0.5, rho=-0.5)
#: the reconciliation identity: the pass sums the march's own sources (round-off relative to the cash scale)
RECONCILIATION = 1e-11
#: |probability_residual| bounds per engine, measured on every case below (the march's own discounting of a unit claim
#: against DF(t); shrinks under time refinement, ``test_residual_shrinks_under_refinement``)
RESIDUAL = {"flat": 1e-7, "lv": 1e-7, "heston": 1e-6}
#: the adjoint identity (spec AC14)
ADJOINT = 1e-10


def env_on(d, spot=6000.0, *, day_count=DayCountConvention.ACT_365, calendar=None, grid=False):
    if grid:
        from quantark.param.vol.vol_surface import GridVolSurface

        strikes = [3000.0, 4500.0, 6000.0, 7500.0, 9000.0]
        mats = [0.25, 0.5, 1.0, 1.5, 2.0, 2.5]
        base = FLAT["sigma"] + 0.05 * (6000.0 - np.array(strikes)) / 6000.0
        vol = GridVolSurface(strikes=strikes, maturities=mats, iv_grid=np.tile(base, (len(mats), 1)))
    else:
        vol = FlatVolSurface(volatility=FLAT["sigma"])
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=vol, rate_curve=FlatRateCurve(rate=FLAT["r"]),
        div_yield=ContinuousDividendYield(div_yield=FLAT["q"]), valuation_date=d, day_count_convention=day_count,
        calendar=calendar,
    )


def flow(kind, determined, paid, amount=CASH):
    event = {"coupon": LifecycleEventType.COUPON, "ko": LifecycleEventType.KNOCK_OUT,
             "loss": LifecycleEventType.MATURITY}[kind]
    return RealizedCashflow(cashflow_id=f"dcn:{kind}:{determined:%Y-%m-%d}", event_type=event, amount=amount,
                            determination_date=determined, payment_date=paid)


def state_on(d, *flows, **flags) -> DCNLifecycleState:
    return DCNLifecycleState(valuation_point=ValuationPoint(date=d), ledger=LifecycleCashflowLedger(flows), **flags)


ENGINES = {
    "flat": lambda: DCNPDEEngine(PDEParams(accuracy="fast")),
    "lv": lambda: LocalVolDCNPDEEngine(params=PDEParams(accuracy="fast")),
    "heston": lambda: HestonDCNPDESolver(HESTON, n_x=61, n_v=21),
}

#: name -> (engine key, product factory, env factory, lifecycle-state factory or None)
CASES = {
    "flat_day": ("flat", lambda: make_dcn(DCN_A, valuation_date=DAY), lambda: env_on(DAY), None),
    "lv_day": ("lv", lambda: make_dcn(DCN_A, valuation_date=DAY), lambda: env_on(DAY, grid=True), None),
    "heston_day": ("heston", lambda: make_dcn(DCN_A, valuation_date=DAY), lambda: env_on(DAY), None),
    "flat_closed_saturday": ("flat", lambda: make_dcn(DCN_A, valuation_date=SAT), lambda: env_on(SAT, 5500.0), None),
    "lv_closed_holiday": ("lv", lambda: make_dcn(DCN_A, valuation_date=HOLIDAY),
                          lambda: env_on(HOLIDAY, 5500.0, grid=True), None),
    "heston_closed_holiday": ("heston", lambda: make_dcn(DCN_A, valuation_date=HOLIDAY),
                              lambda: env_on(HOLIDAY, 5500.0), None),
    "flat_ki_at_valuation": ("flat", lambda: make_dcn(DCN_A, valuation_date=DAY), lambda: env_on(DAY, 4400.0), None),
    "heston_ki_at_valuation": ("heston", lambda: make_dcn(DCN_A, valuation_date=DAY), lambda: env_on(DAY, 4400.0),
                               None),
    "flat_carried": ("flat", lambda: make_dcn(DCN_A, valuation_date=DAY, knocked_in_at_valuation=True),
                     lambda: env_on(DAY, 5000.0), None),
    "heston_carried": ("heston", lambda: make_dcn(DCN_A, valuation_date=DAY, knocked_in_at_valuation=True),
                       lambda: env_on(DAY, 5000.0), None),
    "flat_ko_at_valuation": ("flat", lambda: make_dcn(DCN_A, valuation_date=OBS), lambda: env_on(OBS, 6300.0), None),
    "heston_ko_at_valuation": ("heston", lambda: make_dcn(DCN_A, valuation_date=OBS), lambda: env_on(OBS, 6300.0),
                               None),
    "flat_carried_ko_at_valuation": ("flat", lambda: make_dcn(DCN_A, valuation_date=OBS, knocked_in_at_valuation=True),
                                     lambda: env_on(OBS, 6300.0), None),
    "flat_coupon_today": ("flat", lambda: make_dcn(DCN_A, valuation_date=OBS), lambda: env_on(OBS, 5000.0), None),
    "heston_coupon_today": ("heston", lambda: make_dcn(DCN_A, valuation_date=OBS), lambda: env_on(OBS, 5000.0), None),
    "flat_final_survives": ("flat", lambda: make_dcn(DCN_A, valuation_date=FINAL), lambda: env_on(FINAL, 5500.0),
                            None),
    "flat_final_knocks_out": ("flat", lambda: make_dcn(DCN_A, valuation_date=FINAL), lambda: env_on(FINAL, 6300.0),
                              None),
    "flat_final_knocked_in": ("flat", lambda: make_dcn(DCN_A, valuation_date=FINAL), lambda: env_on(FINAL, 4000.0),
                              None),
    "heston_final_knocked_in": ("heston", lambda: make_dcn(DCN_A, valuation_date=FINAL),
                                lambda: env_on(FINAL, 4000.0), None),
    "flat_seller": ("flat", lambda: make_dcn(DCN_A, valuation_date=DAY, direction=DCNDirection.SELLER),
                    lambda: env_on(DAY), None),
    "heston_seller": ("heston", lambda: make_dcn(DCN_A, valuation_date=DAY, direction=DCNDirection.SELLER),
                      lambda: env_on(DAY), None),
    "flat_act_360": ("flat", lambda: make_dcn(DCN_A, valuation_date=DAY),
                     lambda: env_on(DAY, day_count=DayCountConvention.ACT_360), None),
    "flat_business_days": ("flat", lambda: make_dcn(DCN_A, valuation_date=DAY),
                           lambda: env_on(DAY, day_count=DayCountConvention.BUSINESS_DAYS, calendar=SSE), None),
    "heston_business_days": ("heston", lambda: make_dcn(DCN_A, valuation_date=DAY),
                             lambda: env_on(DAY, day_count=DayCountConvention.BUSINESS_DAYS, calendar=SSE), None),
    "flat_ko_lock": ("flat", lambda: make_dcn(DCN_B, valuation_date=LOCK), lambda: env_on(LOCK), None),
    "flat_receivable_alive": ("flat", lambda: make_dcn(DCN_A, valuation_date=AGED), lambda: env_on(AGED),
                              lambda: state_on(AGED, flow("coupon", ROW, ROW_PAY))),
    "heston_receivable_alive": ("heston", lambda: make_dcn(DCN_A, valuation_date=AGED), lambda: env_on(AGED),
                                lambda: state_on(AGED, flow("coupon", ROW, ROW_PAY))),
    "flat_receivable_alive_seller": (
        "flat", lambda: make_dcn(DCN_A, valuation_date=AGED, direction=DCNDirection.SELLER), lambda: env_on(AGED),
        lambda: state_on(AGED, flow("coupon", ROW, ROW_PAY, -CASH))),
    "flat_nodal": ("flat_nodal", lambda: make_dcn(DCN_A, valuation_date=DAY), lambda: env_on(DAY), None),
    "heston_nodal": ("heston_nodal", lambda: make_dcn(DCN_A, valuation_date=DAY), lambda: env_on(DAY), None),
}
ENGINES["flat_nodal"] = lambda: DCNPDEEngine(PDEParams(accuracy="fast", event_projection="nodal"))
# a narrow domain: the edge rows carry real mass (lower ~2% alive + 5% knocked in), so their labels are exercised
ENGINES["flat_narrow"] = lambda: DCNPDEEngine(PDEParams(accuracy="fast", grid=GridConfig(num_std=1.0)))
CASES["flat_narrow"] = ("flat_narrow", lambda: make_dcn(DCN_A, valuation_date=DAY), lambda: env_on(DAY), None)
ENGINES["heston_nodal"] = lambda: HestonDCNPDESolver(HESTON, n_x=61, n_v=21, event_projection="nodal")
# the ADI options the adjoint transposes beside the default (Craig-Sneyd, degenerate v = 0 row): Douglas splitting and
# a Neumann v = 0 boundary (CP5)
ENGINES["heston_douglas"] = lambda: HestonDCNPDESolver(HESTON, n_x=61, n_v=21, scheme="DOUGLAS")
ENGINES["heston_neumann"] = lambda: HestonDCNPDESolver(HESTON, n_x=61, n_v=21, v0_boundary="neumann")
CASES["heston_douglas"] = ("heston_douglas", lambda: make_dcn(DCN_A, valuation_date=DAY), lambda: env_on(DAY), None)
CASES["heston_neumann"] = ("heston_neumann", lambda: make_dcn(DCN_A, valuation_date=DAY), lambda: env_on(DAY), None)

TERMINATED = {
    "flat_ko_unpaid": ("flat", lambda: make_dcn(DCN_A, valuation_date=AGED), lambda: env_on(AGED),
                       lambda: state_on(AGED, flow("ko", ROW, ROW_PAY), knocked_out=True, ko_date=ROW)),
    "heston_ko_unpaid": ("heston", lambda: make_dcn(DCN_A, valuation_date=AGED), lambda: env_on(AGED),
                         lambda: state_on(AGED, flow("ko", ROW, ROW_PAY), knocked_out=True, ko_date=ROW)),
    "lv_ko_unpaid_seller": ("lv", lambda: make_dcn(DCN_A, valuation_date=AGED, direction=DCNDirection.SELLER),
                            lambda: env_on(AGED, grid=True),
                            lambda: state_on(AGED, flow("ko", ROW, ROW_PAY, -CASH), knocked_out=True, ko_date=ROW)),
    "flat_matured_loss_unpaid": (
        "flat", lambda: make_dcn(DCN_A, valuation_date=FINAL, knocked_in_at_valuation=True),
        lambda: env_on(FINAL, 4000.0),
        lambda: state_on(FINAL, flow("loss", FINAL, datetime(2025, 1, 7), -55_000.0), knocked_in=True,
                         matured=True)),
}
SETTLED_STATE = (lambda: make_dcn(DCN_A, valuation_date=SETTLED), lambda: env_on(SETTLED),
                 lambda: state_on(SETTLED, flow("ko", ROW, ROW_PAY), knocked_out=True, ko_date=ROW))

DECIDED = ["flat_ko_at_valuation", "heston_ko_at_valuation", "flat_carried_ko_at_valuation", "flat_final_survives",
           "flat_final_knocks_out", "flat_final_knocked_in", "heston_final_knocked_in"]
MARCHED = [name for name in CASES if name not in DECIDED]


def _kwargs(spec):
    state = spec[3]
    return {} if state is None else {"lifecycle_state": state()}


def _fresh(name, table=None):
    spec = (table or CASES)[name]
    return ENGINES[spec[0]](), spec[1](), spec[2]()


_STATS: dict = {}


def _stats(name) -> EventStatistics:
    if name not in _STATS:
        engine, product, env = _fresh(name)
        _STATS[name] = engine.calculate_event_stats(product, env, **_kwargs(CASES[name]))
    return _STATS[name]


def _scale(stats) -> float:
    cash = [stats.observations.discounted_cash, stats.coupons.discounted_cash, stats.receivables.discounted_cash,
            [stats.maturity.discounted_cash, stats.ki_settlement.discounted_cash, stats.pv]]
    return max(1.0, float(sum(np.sum(np.abs(c)) for c in cash)))


# --- AC7 / AC8 / D23 -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CASES))
def test_full_request_is_an_event_statistics_with_only_undefined_values_unavailable(name):
    stats = _stats(name)
    assert isinstance(stats, EventStatistics) and stats.product_family == "dcn"
    assert stats.coupons is not None and stats.ki_table is not None
    assert stats.standard_errors is None
    for key, reason in stats.unavailable.items():
        assert key == "knock_in.expected_time" and reason.startswith("undefined:"), (key, reason)
        assert stats.knock_in.probability == 0.0 or is_close(stats.knock_in.probability, 0.0)
    assert stats.knock_in.expected_loss >= 0.0
    assert stats.knock_in.probability_no_ko == stats.ki_settlement.probability


@pytest.mark.parametrize("name", sorted(CASES))
def test_partition_and_reconciliation(name):
    stats = _stats(name)
    assert abs(stats.reconciliation_error) <= RECONCILIATION * _scale(stats), stats.reconciliation_error
    assert abs(stats.probability_residual) <= RESIDUAL[CASES[name][0].split("_")[0]], stats.probability_residual


@pytest.mark.parametrize("name", sorted(CASES))
def test_pv_is_the_price_bitwise_on_every_entry_point(name):
    engine, product, env = _fresh(name)
    kwargs = _kwargs(CASES[name])
    price = engine.price(product, env, **kwargs)
    assert _stats(name).pv == price
    assert engine.price_with_events(product, env, **kwargs).npv == price
    assert engine.price_with_events(product, env, emit_distribution=False, **kwargs).npv == price


@pytest.mark.parametrize("engine_key, refine", [
    ("flat", lambda: DCNPDEEngine(PDEParams(accuracy="high"))),
    ("heston", lambda: HestonDCNPDESolver(HESTON, n_x=61, n_v=21, substeps_per_interval=2)),
])
def test_residual_shrinks_under_refinement(engine_key, refine):
    """The residual is the march's own discounting of a unit claim against DF(t): it shrinks as time refines."""
    product, env = make_dcn(DCN_A, valuation_date=DAY), env_on(DAY)
    coarse = ENGINES[engine_key]().calculate_event_stats(product, env).probability_residual
    fine = refine().calculate_event_stats(product, env).probability_residual
    assert abs(fine) < 0.5 * abs(coarse), (coarse, fine)


def test_streams_prune_nothing():
    engine, product, env = _fresh("flat_day")
    full = engine.calculate_event_stats(product, env)
    pruned = engine.calculate_event_stats(product, env, streams=frozenset({EventType.KO}))
    np.testing.assert_array_equal(pruned.observations.probability, full.observations.probability)
    np.testing.assert_array_equal(pruned.ki_table.probability, full.ki_table.probability)
    assert pruned.unavailable == full.unavailable == {}


# --- AC14: the forward pass is the adjoint of the backward march -----------------------------------------------------

def _claims_vs_stats(claims, stats, price):
    n = stats.observations.times.size
    gaps = {
        "ko_total": claims["ko_total"] - float(np.sum(stats.observations.probability)),
        "maturity": claims["maturity"] - stats.maturity.probability,
        "ki_settlement": claims["ki_settlement"] - stats.ki_settlement.probability,
    }
    if "ki_total" in claims and stats.knock_in.expected_time != 0.0:     # not decided at valuation (P = 1 by rule)
        gaps["ki_total"] = claims["ki_total"] - stats.knock_in.probability
        gaps["ki_time"] = claims["ki_time"] - float(np.sum(stats.ki_table.times * stats.ki_table.probability))
    for j in range(n):
        if f"ko_{j}" in claims:
            gaps[f"ko_{j}"] = claims[f"ko_{j}"] - stats.observations.probability[j]
            gaps[f"coupon_{j}"] = claims[f"coupon_{j}"] - stats.coupons.probability[j]
    return gaps


@pytest.mark.parametrize("name", ["flat_day", "lv_day", "flat_closed_saturday", "lv_closed_holiday",
                                  "flat_ki_at_valuation", "flat_carried", "flat_ko_lock", "flat_seller",
                                  "flat_nodal", "flat_business_days", "flat_narrow"])
def test_forward_pass_reproduces_the_backward_unit_claims_1d(name):
    engine, product, env = _fresh(name)
    stats = _stats(name)
    claims = backward_claims_1d(engine, product, env)
    sign = float(product.direction_sign)
    # the backward claim march is the price's own march: its price claim is price() bitwise
    assert sign * claims["price"] == engine.price(product, env)
    gaps = _claims_vs_stats(claims, stats, stats.pv)
    assert max(abs(g) for g in gaps.values()) <= ADJOINT, gaps
    assert abs(sign * claims["ki_settlement_cash"] - stats.ki_settlement.discounted_cash) <= RECONCILIATION * _scale(
        stats)


@pytest.mark.parametrize("name", ["heston_day", "heston_carried", "heston_closed_holiday", "heston_nodal",
                                  "heston_douglas", "heston_neumann"])
def test_forward_pass_reproduces_the_backward_unit_claims_adi(name):
    engine, product, env = _fresh(name)
    stats = _stats(name)
    names = ["price", "ko_total", "maturity", "ki_settlement", "ki_settlement_cash", "ki_total", "ki_time", "ko_0",
             "coupon_0", "ko_3", "coupon_3"]
    claims = backward_claims_adi(engine, product, env, names=names)
    sign = float(product.direction_sign)
    assert is_close(sign * claims["price"], engine.price(product, env), rel_tol=1e-13, abs_tol=1e-9)
    gaps = _claims_vs_stats(claims, stats, stats.pv)
    assert max(abs(g) for g in gaps.values()) <= ADJOINT, gaps
    assert abs(sign * claims["ki_settlement_cash"] - stats.ki_settlement.discounted_cash) <= RECONCILIATION * _scale(
        stats)


def test_edge_terms_are_the_price_rows():
    """The statistics label each edge row by the outcome it pays (D23): the labels re-sum the price's own rows."""
    from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest, prepare

    engine, product, env = _fresh("flat_day")
    problem = prepare(engine, product, env, OutputRequest(Operation.PRICE))
    ev = problem.events
    for t in np.asarray(problem.layout.time.t, dtype=float)[:-1:7]:
        (lo_a, lo_k), (up_a, up_k) = ev.rows(float(t))
        loss, (coupons, ko_row), ki_ko_row = ev.edge_terms(float(t))
        assert loss == lo_a == lo_k
        alive = sum(ev.amounts(j, float(ev.ctx.times[ev.ctx.obs_cols[j]]))[0] * ev.df(ev.ctx.times[ev.ctx.obs_cols[j]])
                    / ev.df(t) for j in coupons)
        ki = 0.0
        if ko_row is not None:
            t_obs = float(ev.ctx.times[ev.ctx.obs_cols[ko_row]])
            ko = ev.amounts(ko_row, t_obs)[1] * ev.df(t_obs) / ev.df(t)
            alive += ko
            ki = ko
        assert is_close(up_a, alive, rel_tol=1e-13, abs_tol=1e-9) and is_close(up_k, ki, rel_tol=1e-13, abs_tol=1e-9)
        assert ki_ko_row == ko_row


def test_the_narrow_domain_puts_real_mass_on_the_edge_rows(monkeypatch):
    """The AC14 case ``flat_narrow`` really exercises the edge labels (on default domains the edge mass is ~1e-30)."""
    from quantark.asset.equity.engine.pde import dcn_statistics as module

    seen = {"lower_alive": 0.0, "upper_alive": 0.0}
    lower, upper = module.DCNForwardResult.lower_edge, module.DCNForwardResult.upper_edge

    def spy_lower(self, t, alive, ki, value, df_t):
        seen["lower_alive"] += alive / df_t
        lower(self, t, alive, ki, value, df_t)

    def spy_upper(self, alive, ki, coupons, ko_row, ki_ko_row, df_t):
        seen["upper_alive"] += alive / df_t
        upper(self, alive, ki, coupons, ko_row, ki_ko_row, df_t)

    monkeypatch.setattr(module.DCNForwardResult, "lower_edge", spy_lower)
    monkeypatch.setattr(module.DCNForwardResult, "upper_edge", spy_upper)
    engine, product, env = _fresh("flat_narrow")
    engine.calculate_event_stats(product, env)
    assert seen["lower_alive"] > 1e-2 and seen["upper_alive"] > 1e-7, seen


def test_a_sparse_or_semi_lagrangian_core_is_refused():
    from quantark.asset.equity.engine.pde import dcn_statistics as module

    class Core:
        use_sparse, _semi_lagrangian_v = True, False

    with pytest.raises(module.AdjointUnavailable, match="undefined"):
        module._refuse_core(Core())


# --- AC13: the per-date knock-in table ------------------------------------------------------------------------------

@pytest.mark.parametrize("name", [n for n in MARCHED if "carried" not in n and "ki_at" not in n])
def test_knock_in_table_on_the_daily_dates(name):
    engine, product, env = _fresh(name)
    stats = _stats(name)
    ctx = build_dcn_grid_context(product, env)
    first = 0 if ctx.valuation_monitored else 1
    table = stats.ki_table
    np.testing.assert_array_equal(table.times, ctx.times[first:])
    np.testing.assert_array_equal(table.payment_times, table.times)                                 # D8
    assert table.dates == tuple(product.schedule.daily_ki_dates)
    assert not np.any(table.interpolated) and table.interpolation_scheme is None
    assert is_close(float(np.sum(table.probability)), stats.knock_in.probability, rel_tol=0.0, abs_tol=1e-15)
    assert np.all(table.probability >= -1e-15)
    if stats.knock_in.expected_time is not None:
        mean = float(np.sum(table.times * table.probability)) / stats.knock_in.probability
        assert is_close(mean, stats.knock_in.expected_time, rel_tol=1e-12, abs_tol=0.0)
    assert np.all(np.diff(table.survival) <= 1e-15)
    # a knock-in before any knock-out bounds the knocked-in settlement (up to the march's unit-claim discounting,
    # the probability residual's source: KI mass is normalised at its date, the settlement at maturity)
    assert stats.knock_in.probability >= stats.ki_settlement.probability - RESIDUAL[CASES[name][0].split("_")[0]]
    if not ctx.valuation_monitored:
        assert table.times[0] > 0.0


@pytest.mark.parametrize("name", ["flat_carried", "heston_carried", "flat_ki_at_valuation", "heston_ki_at_valuation"])
def test_a_knock_in_decided_at_valuation_is_one_row_at_zero(name):
    stats = _stats(name)
    _, product, _ = _fresh(name)
    assert stats.knock_in.probability == 1.0 and stats.knock_in.expected_time == 0.0
    np.testing.assert_array_equal(stats.ki_table.times, [0.0])
    np.testing.assert_array_equal(stats.ki_table.probability, [1.0])
    assert stats.ki_table.dates == (product.schedule.valuation_date,)


# --- AC10: valuation-date rules; decided rows equal the DCN MC's one-path settlement ---------------------------------

@pytest.mark.parametrize("name", DECIDED)
def test_decided_rows_are_the_mc_settled_row(name):
    engine, product, env = _fresh(name)
    pde = _stats(name)
    mc = DCNMCEngine(num_paths=64, seed=1).calculate_event_stats(product, env)
    for field in ("probability", "discounted_cash", "cash_to_pay", "times", "payment_times"):
        np.testing.assert_allclose(getattr(pde.observations, field), getattr(mc.observations, field), rtol=1e-14,
                                   atol=1e-12)
    for field in ("probability", "discounted_cash", "cash_to_pay"):
        np.testing.assert_allclose(getattr(pde.coupons, field), getattr(mc.coupons, field), rtol=1e-14, atol=1e-12)
    for row in ("maturity", "ki_settlement"):
        for field in ("probability", "discounted_cash", "cash_to_pay", "time", "payment_time"):
            assert is_close(getattr(getattr(pde, row), field), getattr(getattr(mc, row), field), rel_tol=1e-14,
                            abs_tol=1e-12), (row, field)
    assert pde.knock_in.probability == mc.knock_in.probability
    assert pde.knock_in.expected_time == mc.knock_in.expected_time
    np.testing.assert_array_equal(pde.ki_table.times, mc.ki_table.times)
    np.testing.assert_array_equal(pde.ki_table.probability, mc.ki_table.probability)
    assert pde.ki_table.dates == mc.ki_table.dates
    assert pde.expected_lifetime == mc.expected_lifetime


def test_knock_out_at_valuation_without_a_carried_knock_in():
    stats = _stats("flat_ko_at_valuation")
    assert stats.observations.probability[0] == 1.0 and np.all(stats.observations.probability[1:] == 0.0)
    assert stats.knock_in.probability == 0.0 and stats.ki_table.times.size == 0
    assert stats.unavailable == {"knock_in.expected_time": "undefined: P(knock in) = 0"}
    assert stats.expected_lifetime == 0.0


@pytest.mark.parametrize("name", ["flat_coupon_today", "heston_coupon_today"])
def test_a_coupon_today_is_decided_at_the_spot(name):
    stats = _stats(name)
    assert stats.coupons.probability[0] == 1.0 and stats.observations.probability[0] == 0.0
    _, product, env = _fresh(name)
    from quantark.asset.equity.engine.dcn_valuation import observe_valuation_row

    row = observe_valuation_row(product, env)
    assert is_close(stats.coupons.discounted_cash[0], product.direction_sign * row.coupon_pv, rel_tol=1e-14)


# --- AC17: receivables and terminated states -------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["flat_receivable_alive", "heston_receivable_alive", "flat_receivable_alive_seller"])
def test_receivables_are_ledger_rows_not_events(name):
    engine, product, env = _fresh(name)
    state = CASES[name][3]()
    stats = engine.calculate_event_stats(product, env, lifecycle_state=state)
    pending = state.ledger.pending(state.valuation_point)
    assert stats.receivables.cash_to_pay.tolist() == [f.amount for f in pending]
    assert stats.receivables.source == ("coupon",)
    plain = engine.calculate_event_stats(product, env)
    np.testing.assert_array_equal(stats.observations.probability, plain.observations.probability)
    dist = engine.price_with_events(product, env, lifecycle_state=state).event_distribution
    plain_dist = engine.price_with_events(product, env).event_distribution
    for stream in dist.probabilities:
        np.testing.assert_array_equal(np.asarray(dist.probabilities[stream]),
                                      np.asarray(plain_dist.probabilities[stream]))


@pytest.mark.parametrize("name", sorted(TERMINATED))
def test_a_terminated_note_with_unpaid_cash_reports_its_receivables_only(name):
    engine, product, env = _fresh(name, TERMINATED)
    state = TERMINATED[name][3]()
    stats = engine.calculate_event_stats(product, env, lifecycle_state=state)
    price = engine.price(product, env, lifecycle_state=state)
    assert stats.pv == price and stats.terminated
    assert stats.observations.times.size == 0 and stats.unavailable.get("standard_errors") is None
    assert abs(stats.reconciliation_error) <= 1e-9 * max(1.0, abs(price))
    mc = DCNMCEngine(num_paths=64, seed=1).calculate_event_stats(product, env, lifecycle_state=state)
    np.testing.assert_array_equal(stats.receivables.discounted_cash, mc.receivables.discounted_cash)
    assert stats.receivables.payment_dates == mc.receivables.payment_dates
    result = engine.price_with_events(product, env, lifecycle_state=state)
    assert result.npv == price and result.event_distribution.terminated


@pytest.mark.parametrize("engine_key", ["flat", "lv", "heston"])
def test_a_terminated_settled_note_has_no_statistics_and_no_event(engine_key):
    engine = ENGINES[engine_key]()
    product, env, state = SETTLED_STATE[0](), SETTLED_STATE[1](), SETTLED_STATE[2]()
    if engine_key == "lv":
        env = env_on(SETTLED, grid=True)
    assert engine.calculate_event_stats(product, env, lifecycle_state=state) is None
    result = engine.price_with_events(product, env, lifecycle_state=state)
    assert result.npv == engine.price(product, env, lifecycle_state=state)
    assert result.event_distribution.terminated and result.event_distribution.probabilities == {}


# --- AC18 (0.14.0: the environment's axis): axis, dates, signs, routes -----------------------------------------------

@pytest.mark.parametrize("name", ["flat_day", "flat_act_360", "flat_business_days", "heston_business_days",
                                  "flat_closed_saturday", "flat_ko_lock", "lv_closed_holiday"])
def test_times_are_the_dcn_axis_with_exact_dates_as_the_mc(name):
    engine, product, env = _fresh(name)
    pde = _stats(name)
    mc = DCNMCEngine(num_paths=256, seed=3).calculate_event_stats(product, env)
    for table in ("observations", "coupons"):
        for field in ("payment_times", "payment_dates", "is_ko_obs", "is_coupon_obs", "times", "dates"):
            if hasattr(getattr(mc, table), field):
                a, b = getattr(getattr(pde, table), field), getattr(getattr(mc, table), field)
                if isinstance(a, np.ndarray):
                    np.testing.assert_array_equal(a, b)
                else:
                    assert a == b, (table, field)
    for row in ("maturity", "ki_settlement"):
        for field in ("time", "payment_time", "date", "payment_date"):
            assert getattr(getattr(pde, row), field) == getattr(getattr(mc, row), field)
    np.testing.assert_array_equal(pde.ki_table.times, mc.ki_table.times)
    assert pde.ki_table.dates == mc.ki_table.dates
    # the environment's axis from the schedule's valuation date (0.14.0 retired D12's ACT/365F)
    v = product.schedule.valuation_date
    np.testing.assert_array_equal(pde.observations.times, env_times(env, pde.observations.dates, anchor=v))
    np.testing.assert_array_equal(pde.observations.payment_times,
                                  env_times(env, pde.observations.payment_dates, anchor=v))
    # AC12: deterministic-payment rows' cash to pay is the discounted cash over the payment DF
    df = make_df_fn(env)
    for table in (pde.observations, pde.coupons):
        dfs = np.array([df(t) for t in table.payment_times])
        np.testing.assert_allclose(table.cash_to_pay * dfs, table.discounted_cash, rtol=1e-13, atol=1e-9)
    assert is_close(pde.ki_settlement.cash_to_pay * df(pde.ki_settlement.payment_time),
                    pde.ki_settlement.discounted_cash, rel_tol=1e-13)


@pytest.mark.parametrize("engine_key", ["flat", "heston"])
def test_seller_reports_negative_cash_with_identities_intact(engine_key):
    buyer, seller = _stats(f"{engine_key}_day"), _stats(f"{engine_key}_seller")
    np.testing.assert_array_equal(seller.observations.probability, buyer.observations.probability)
    np.testing.assert_array_equal(seller.coupons.discounted_cash, -buyer.coupons.discounted_cash)
    np.testing.assert_array_equal(seller.observations.discounted_cash, -buyer.observations.discounted_cash)
    assert seller.ki_settlement.discounted_cash == -buyer.ki_settlement.discounted_cash
    assert seller.ki_settlement.discounted_cash > 0.0 and seller.knock_in.expected_loss > 0.0
    assert seller.pv == -buyer.pv or is_close(seller.pv, -buyer.pv, rel_tol=1e-13)
    assert abs(seller.reconciliation_error) <= RECONCILIATION * _scale(seller)


def _legs():
    return [
        FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=7_000.0, trigger=PaymentTrigger.AT_KO,
                       payment_offset_days=0),
        FixedPayoffLeg(direction=LegDirection.BUYER_RECEIVES, amount=3_000.0, trigger=PaymentTrigger.AT_MATURITY_ANY,
                       payment_offset_days=0),
        FixedPayoffLeg(direction=LegDirection.BUYER_PAYS, amount=4_000.0,
                       trigger=PaymentTrigger.AT_MATURITY_WITH_KI, payment_offset_days=0),
        FixedPayoffLeg(direction=LegDirection.BUYER_PAYS, amount=2_000.0,
                       trigger=PaymentTrigger.AT_MATURITY_NO_KO, payment_offset_days=0),
    ]


#: the ADI's far high-spot rows hold 0 on the folded solve, which the event route reads as maturity: the measured gap
#: (a maturity leg times the far-edge mass) on these grids
ADI_FOLD_GAP = 1e-6


@pytest.mark.parametrize("name", ["flat_day", "lv_day", "flat_act_360", "flat_business_days", "flat_seller",
                                  "flat_closed_saturday", "heston_day", "heston_business_days"])
@pytest.mark.parametrize("quantity", [1.0, -2.0])
def test_event_route_and_fold_route_agree(name, quantity):
    """AC18: the folded solve marches each leg's cash inside the price on the same grid (``bind_dcn_fold``); the event
    route values it on the statistics' axis. They agree to round-off on the 1D core, on every environment."""
    engine, product, env = _fresh(name)
    result = engine.price_with_events(product, env)
    dist = result.event_distribution
    legs = _legs()
    overlay = fold_cash_legs(legs, product, env, quantity=quantity, unit_notional=100.0)
    folded = quantity * _fresh(name)[0].price_folded(product, env, overlay) + overlay.constant_pv
    expected = quantity * result.npv + sum(leg.value(dist, env, 100.0) for leg in legs)
    tol = (ADI_FOLD_GAP if name.startswith("heston") else 1e-9) * product.notional * abs(quantity)
    assert abs(folded - expected) <= tol, (folded, expected)


@pytest.mark.parametrize("name", ["flat_act_360", "flat_business_days"])
def test_the_environment_axis_differs_there(name):
    """These environments really are on another axis than ACT/365F (so the env-axis tests are not a coincidence)."""
    from quantark.util.calendar.day_counter import calculate_year_fraction

    _, product, env = _fresh(name)
    d = product.schedule.monthly[5].observation_date
    t_env = calculate_year_fraction(env.valuation_date, d, env.day_count_convention, env.bus_days_in_year,
                                    calendar=env.calendar)
    assert abs(t_env - (d - env.valuation_date).days / 365.0) > 1e-3


def test_coupon_only_rows_carry_no_knock_out():
    stats = _stats("flat_ko_lock")
    obs = stats.observations
    lock = ~obs.is_ko_obs
    assert np.any(lock) and np.all(obs.is_coupon_obs)
    assert np.all(obs.probability[lock] == 0.0) and np.all(obs.discounted_cash[lock] == 0.0)
    assert np.any(stats.coupons.probability[lock] > 0.0)


@pytest.mark.parametrize("name", ["flat_coupon_today", "heston_coupon_today"])
def test_a_coupon_today_is_the_march_plus_the_decided_row(name):
    """The march never applies the valuation date's row (W10b): the backward claims cover every later outcome and the
    decided coupon adds probability 1 to row 0 (and its cash to the price) on top."""
    from quantark.asset.equity.engine.dcn_valuation import observe_valuation_row

    engine, product, env = _fresh(name)
    stats = _stats(name)
    names = ["price", "ko_total", "maturity", "ki_settlement", "ki_total", "ki_time", "ko_0", "coupon_0", "coupon_1"]
    claims = (backward_claims_adi(engine, product, env, names=names) if name.startswith("heston")
              else backward_claims_1d(engine, product, env))
    row = observe_valuation_row(product, env)
    sign = float(product.direction_sign)
    assert is_close(sign * (claims["price"] + row.coupon_pv), stats.pv, rel_tol=1e-13, abs_tol=1e-9)
    gaps = _claims_vs_stats({k: v for k, v in claims.items() if not k.startswith(("coupon_", "ko_"))} | {
        "ko_total": claims["ko_total"]}, stats, stats.pv)
    gaps["ko_0"] = claims["ko_0"] - stats.observations.probability[0]
    gaps["coupon_0"] = claims["coupon_0"] + 1.0 - stats.coupons.probability[0]
    gaps["coupon_1"] = claims["coupon_1"] - stats.coupons.probability[1]
    assert max(abs(g) for g in gaps.values()) <= ADJOINT, gaps
