"""KO-reset settlement precedence: an explicit settlement_date comes first (spec D33, AC36).

Owner ruling (2026-11-14): the settlement delay applies only when no ``settlement_date`` is provided. A KO-reset note
that ends with schedule 1 -- a never-knocked-in maturity, and under EXPIRY pay a schedule-1 knock-out (D30) -- is paid
on the product's explicit ``settlement_date`` when one is booked, else at schedule 1's maturity plus the settlement
delay. Monte Carlo always did this (the resolver's TERMINAL precedence at schedule 1's maturity); the grid engines paid
their never-knocked-in maturity row at ``pre_maturity + (settlement - T)``, so on a dated note whose schedule 1 ends a
year before the product they discounted that cash a year too little (T25b: PDE 99.43 vs MC ~98.0), and the MC fold
route refused the contracts (its structure assumed the grid's payment).

A numeric product cannot carry a ``settlement_date`` (clock rule), so every D33 contract here is dated.

- every engine (MC, PDE 1D, LV PDE, Heston ADI, QUAD V1, QUAD V2): with a ``settlement_date`` the never-knocked-in
  maturity row and, under EXPIRY, every schedule-1 knock-out row pay at the settlement date's year fraction
  (fresh, aged, discrete and continuous KI, a settlement lag beside the date, a far settlement, a note whose schedule
  1 ended at the valuation instant); without one at schedule 1's maturity plus the delay;
- a schedule-1 knock-out decided at valuation under EXPIRY is worth its cash x DF(settlement) on every engine;
- PDE 1D, QUAD V1 and QUAD V2 agree with an MC reference within the matrix gate on pv and the cash rows (INSTANT and
  EXPIRY);
- the MC fold route values these contracts (INSTANT and EXPIRY) and equals the MC event route on the same paths.
"""
from __future__ import annotations

import warnings
from datetime import timedelta

import numpy as np
import pytest

from event_statistics import matrix_cases as M
from test_asset.equity.engine_mc import mc_price_path_routes as routes

from quantark.asset.equity.engine.mc import SnowballMCEngine
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.ko_reset_vol_pde_solvers import HestonKOResetSnowballPDESolver
from quantark.asset.equity.engine.quad import KOResetSnowballQuadEngine, KOResetSnowballQuadEngineV2
from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.settlement import SettlementConvention, SettlementLagUnit
from quantark.util.calendar import calculate_year_fraction
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.option_enums import PostKOScheduleMode

warnings.filterwarnings("ignore", category=DeprecationWarning)

EXPIRY, INSTANT = CouponPayType.EXPIRY, CouponPayType.INSTANT
PAYS = {"expiry": EXPIRY, "instant": INSTANT}
TIME_TOL = 1e-12

# --- contracts ------------------------------------------------------------------------------------------------------

START = routes.VALUATION                      # the markets' valuation date (flat, grid and Heston groups)
#: quarterly fixings over two years: schedule 1 (KO 103, 15% p.a.; discrete KI 80) the first four, schedule 2
#: (KO 95, 3% p.a.) all eight -- the matrix KO-reset on dates
QUARTERS = [round(k * 365 / 4) for k in range(1, 9)]
SETTLE_DAYS = 20                              # the settlement date: 20 days after the exercise date


def _on(day: int):
    return START + timedelta(days=day)


def _dates(days, level):
    return ObservationSchedule(records=[ObservationRecord(observation_date=_on(d), barrier=level) for d in days])


def dated(pay, *, settle=SETTLE_DAYS, ki="discrete", lag=0.0, pre=QUARTERS[:4]) -> KnockOutResetSnowballOption:
    """The matrix KO-reset on dates from the markets' valuation date: ``settle`` days after the exercise date is the
    explicit settlement date (``None``: none booked); ``lag`` a year-fraction settlement convention beside it."""
    ki_kw = (dict(ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True) if ki == "continuous"
             else dict(ki_observation_type=ObservationType.DISCRETE, ki_observation_schedule=_dates(pre, 80.0)))
    product = KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, initial_date=START, exercise_date=_on(QUARTERS[-1]),
        settlement_date=None if settle is None else _on(QUARTERS[-1] + settle), contract_multiplier=1.0,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_dates(pre, 103.0), ki_barrier=80.0, **ki_kw),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_dates(QUARTERS, 95.0)),
        accrual_config=AccrualConfig(is_annualized=True, is_annualized_ki=True, is_annualized_rebate=True,
                                     coupon_pay_type=pay),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
        settlement_convention=(SettlementConvention(lag=lag, lag_unit=SettlementLagUnit.YEAR_FRACTION)
                               if lag else None),
    )
    product.validate()
    return product


def flat(day: int = 0, spot: float = 100.0):
    env = routes.flat_env(spot)
    env.valuation_date = _on(day)
    return env


#: name -> (product(pay), env); a settlement_date booked and schedule 1 ending a year before the product
SETTLED = {
    "fresh_dki": (lambda pay: dated(pay), lambda: flat()),
    "fresh_cki": (lambda pay: dated(pay, ki="continuous"), lambda: flat()),
    "aged": (lambda pay: dated(pay), lambda: flat(100)),
    "lag_beside_the_date": (lambda pay: dated(pay, lag=0.01), lambda: flat()),
    "far_settlement": (lambda pay: dated(pay, settle=60), lambda: flat()),
    "schedule_one_ends_now": (lambda pay: dated(pay), lambda: flat(QUARTERS[3])),
}
#: the same contracts without a settlement_date: the delay rule, unchanged
UNSETTLED = {
    "fresh_no_settlement": (lambda pay: dated(pay, settle=None), lambda: flat()),
    "lagged_no_settlement": (lambda pay: dated(pay, settle=None, lag=0.01), lambda: flat()),
}


def settlement_time(product, env) -> float:
    """The settlement date's year fraction on env's clock -- from the contract and the day count, not an engine."""
    return float(calculate_year_fraction(env.valuation_date, product.settlement_date, env.day_count_convention,
                                         env.bus_days_in_year, calendar=getattr(env, "calendar", None)))


def schedule_one_end_plus_delay(product, env) -> float:
    terminal = resolve_terminal_timing(product, env)
    pre = max(r.observation_time for r in product._resolve_ko_schedule(product.barrier_config, env)[0])
    return float(pre) + (float(terminal.payment_time) - float(terminal.determination_time))


# --- engines --------------------------------------------------------------------------------------------------------

ENGINES = {
    "mc": lambda: SnowballMCEngine(params=MCParams(num_paths=4096, seed=7)),
    "pde_1d": lambda: KOResetSnowballPDESolver(params=PDEParams(accuracy="fast")),
    "quad_v1": lambda: KOResetSnowballQuadEngine(QuadParams(grid_points=401)),
    "quad_v2": lambda: KOResetSnowballQuadEngineV2(),
}
#: the PDE family's other routes, each on its own market (the LV PDE reads a grid surface)
VOL_ENGINES = {
    "lv_pde_1d": (M.GROUPS["lv"].grid["ko_reset"]["lv_pde_1d"]["L0"], lambda day=0, spot=100.0: _grid(day, spot)),
    "heston_adi": (lambda: HestonKOResetSnowballPDESolver(M.HP, n_x=120, n_v=16, n_t=60), flat),
}


def _grid(day: int = 0, spot: float = 100.0):
    env = routes.grid_env(spot)
    env.valuation_date = _on(day)
    return env


def _assert_payments(stats, product, env, pay, expected, label):
    """The maturity row and (EXPIRY) every schedule-1 knock-out row pay at ``expected``; cash = cash to pay x DF."""
    assert stats.maturity.payment_time == pytest.approx(expected, abs=TIME_TOL), (label, stats.maturity)
    obs = stats.observations
    pre = np.asarray(obs.schedule) == "pre"
    if pay is EXPIRY and pre.any():
        np.testing.assert_allclose(obs.payment_times[pre], expected, rtol=0, atol=TIME_TOL,
                                   err_msg=f"{label}: schedule-1 knock-outs pay with the never-knocked-in maturity")
    pay_df = np.array([float(env.get_discount_factor(t)) for t in obs.payment_times])
    np.testing.assert_allclose(obs.cash_to_pay * pay_df, obs.discounted_cash, rtol=1e-12, atol=1e-12)
    if stats.maturity.probability > 0.0:
        assert stats.maturity.cash_to_pay * float(env.get_discount_factor(stats.maturity.payment_time)) == \
            pytest.approx(stats.maturity.discounted_cash, rel=1e-12, abs=1e-12), label


@pytest.mark.parametrize("pay", list(PAYS))
@pytest.mark.parametrize("engine", list(ENGINES))
@pytest.mark.parametrize("contract", list(SETTLED))
def test_a_note_ending_with_schedule_one_pays_on_the_settlement_date(contract, engine, pay):
    product_of, env_of = SETTLED[contract]
    product, env = product_of(PAYS[pay]), env_of()
    stats = ENGINES[engine]().calculate_event_stats(product, env)
    expected = settlement_time(product, env)
    _assert_payments(stats, product, env, PAYS[pay], expected, f"{engine} {contract} {pay}")
    if stats.maturity.payment_date is not None:
        assert stats.maturity.payment_date == product.settlement_date


@pytest.mark.parametrize("pay", list(PAYS))
@pytest.mark.parametrize("engine", list(ENGINES))
@pytest.mark.parametrize("contract", list(UNSETTLED))
def test_without_a_settlement_date_the_delay_applies(contract, engine, pay):
    product_of, env_of = UNSETTLED[contract]
    product, env = product_of(PAYS[pay]), env_of()
    stats = ENGINES[engine]().calculate_event_stats(product, env)
    _assert_payments(stats, product, env, PAYS[pay], schedule_one_end_plus_delay(product, env),
                     f"{engine} {contract} {pay}")


@pytest.mark.parametrize("pay", list(PAYS))
@pytest.mark.parametrize("engine", list(VOL_ENGINES))
def test_the_vol_model_pde_routes_pay_on_the_settlement_date(engine, pay):
    make, env_of = VOL_ENGINES[engine]
    product, env = dated(PAYS[pay]), env_of()
    stats = make().calculate_event_stats(product, env)
    _assert_payments(stats, product, env, PAYS[pay], settlement_time(product, env), engine)


def test_a_one_maturity_note_pays_its_maturity_on_the_settlement_date():
    """Schedule 1 runs to the product maturity: one maturity, whose payment the product terminal already made the
    settlement date (unchanged by D33)."""
    for pay in (EXPIRY, INSTANT):
        product, env = dated(pay, pre=QUARTERS), flat()
        for name, make in ENGINES.items():
            stats = make().calculate_event_stats(product, env)
            assert stats.maturity.payment_time == pytest.approx(settlement_time(product, env), abs=TIME_TOL), name


@pytest.mark.parametrize("engine", list(ENGINES) + list(VOL_ENGINES))
def test_a_schedule_one_knock_out_decided_at_valuation_pays_on_the_settlement_date(engine):
    """EXPIRY: price() of a schedule-1 knock-out decided at valuation is its contractual cash x DF(settlement)."""
    if engine in VOL_ENGINES:
        make, env_of = VOL_ENGINES[engine]
        env = env_of(QUARTERS[1], 110.0)
    else:
        make, env = ENGINES[engine], flat(QUARTERS[1], 110.0)
    product = dated(EXPIRY)
    decision = product.decide_observations_at_valuation(env.spot, env)
    assert decision.knocked_out and decision.ko_regime == "pre"
    expected = float(decision.ko_record.payoff) * float(env.get_discount_factor(settlement_time(product, env)))
    assert make().price(product, env) == pytest.approx(expected, rel=1e-12)


# --- cross-engine agreement with the MC (the matrix gate) -----------------------------------------------------------

REFERENCE_SEEDS = tuple(range(301, 309))
REFERENCE_PATHS = 16384
GRID = ("pde_1d", "quad_v1", "quad_v2")
GATED = ("pv", "total_discounted_cash", "observations.discounted_cash", "maturity.discounted_cash",
         "ki_settlement.discounted_cash")
AGREEMENT = {
    "fresh_expiry": (lambda: dated(EXPIRY), lambda: flat()),
    "fresh_instant": (lambda: dated(INSTANT), lambda: flat()),
    "aged_expiry": (lambda: dated(EXPIRY), lambda: flat(100)),
}
_REFERENCES = {}


def _reference(contract):
    if contract not in _REFERENCES:
        product_of, env_of = AGREEMENT[contract]
        runs = [M.matrix_fields_of(SnowballMCEngine(params=MCParams(num_paths=REFERENCE_PATHS, seed=seed))
                                   .calculate_event_stats(product_of(), env_of()))
                for seed in REFERENCE_SEEDS]
        _REFERENCES[contract] = M.pool_runs(runs, paths_per_run=REFERENCE_PATHS)
    return _REFERENCES[contract]


@pytest.mark.parametrize("engine", GRID)
@pytest.mark.parametrize("contract", list(AGREEMENT))
def test_grid_engines_agree_with_the_mc_on_a_settled_note(contract, engine):
    """AC36: ``|grid L0 - MC| <= K SE + |L0 - L1| + floor`` (the matrix's K, SE floors and level factories) on pv, the
    cash total and the cash rows. Before D33 the grid engines discounted the never-knocked-in maturity (and, under
    EXPIRY, the schedule-1 knock-outs) to schedule 1's maturity plus (settlement - T)."""
    from event_statistics.test_cross_engine_matrix import K, FLOOR, _floored_se

    product_of, env_of = AGREEMENT[contract]
    reference = _reference(contract)
    levels = M.GROUPS["bs"].grid["ko_reset"][engine]
    l0 = M.matrix_fields_of(levels["L0"]().calculate_event_stats(product_of(), env_of()))
    l1 = M.matrix_fields_of(levels["L1"]().calculate_event_stats(product_of(), env_of()))
    scale = max(1.0, abs(float(reference.mean["pv"][0])))
    problems = []
    for path in GATED:
        value, mc = np.asarray(l0[path], dtype=float), np.asarray(reference.mean[path], dtype=float)
        assert value.shape == mc.shape, (path, value.shape, mc.shape)
        se = _floored_se(path, value, reference, reference.se[path], l0)
        allowance = np.abs(value - np.asarray(l1[path], dtype=float))
        gap = np.abs(value - mc)
        bad = gap > K * se + allowance + FLOOR * scale
        problems += [f"{engine} {contract} {path}[{i}]: grid {value[i]!r} mc {mc[i]!r} se {se[i]:.3g} "
                     f"step {allowance[i]:.3g} z {(gap[i] - allowance[i]) / se[i]:.2f}"
                     for i in np.flatnonzero(bad)]
    assert not problems, "\n".join(problems)


# --- the MC fold route ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("pay", list(PAYS))
def test_the_mc_fold_route_values_a_settled_note_as_its_event_route(pay):
    """The fold structure pays the note that ends with schedule 1 where MC settles it, so the MC fold route binds
    (before D33 it raised a binding mismatch on every such contract) and, on the same paths, equals the event route:
    the folded leg is the leg's value on the price's own outcome distribution."""
    import importlib.util
    import os
    import sys

    from quantark.cashleg.folding import FoldMode, folded_trade_value

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "event_stats_baseline_cases", os.path.join(here, "..", "cashleg", "event_stats_baseline", "cases.py"))
    cases = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("event_stats_baseline_cases", cases)
    spec.loader.exec_module(cases)

    product, env = dated(PAYS[pay]), flat()
    engine = lambda: SnowballMCEngine(params=MCParams(num_paths=4096, seed=7))  # noqa: E731
    legs = {s.name: s.leg for s in cases.legs_for("ko_reset_absolute", product)
            if s.name in ("fixed_at_ko", "fixed_at_maturity_any")}
    assert legs
    result = engine().price_with_events(dated(PAYS[pay]), flat())
    for name, leg in legs.items():
        folded = folded_trade_value(engine(), dated(PAYS[pay]), flat(), [leg], quantity=1.0,
                                    unit_notional=cases.UNIT, mode=FoldMode.TOTAL).total
        event = float(result.npv) + float(leg.value(result.event_distribution, env, cases.UNIT))
        assert np.isfinite(folded), name
        assert folded == pytest.approx(event, rel=1e-9, abs=1e-9), (name, folded, event)


# --- theta: a schedule 1 that ends inside the bump ------------------------------------------------------------------

@pytest.mark.parametrize("settle", [SETTLE_DAYS, None])
def test_theta_pays_a_schedule_one_ending_inside_the_bump_with_the_settlement_rule(settle):
    """Theta's bumped valuation settles a note whose schedule 1 ends inside the bump un-knocked where the engines pay
    it: on the settlement date when booked, else schedule 1's maturity plus the delay."""
    from quantark.asset.equity.riskmeasures.greeks.theta_shift import _ko_reset_schedule_1_end

    product, env = dated(EXPIRY, settle=settle), flat(QUARTERS[3] - 1)
    paid = _ko_reset_schedule_1_end(product, env, 2.0 / 365.0)
    expected = settlement_time(product, env) if settle else schedule_one_end_plus_delay(product, env)
    assert paid == pytest.approx(expected, abs=TIME_TOL)
