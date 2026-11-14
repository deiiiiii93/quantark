"""KO-reset EXPIRY coupon pay: a first-schedule knock-out is paid at the first schedule's maturity (spec D30, AC32).

Owner ruling (2026-11-14): a knock-out on schedule 1 happens only on a path never knocked in, and such a note ends at
schedule 1's maturity, so EXPIRY pay means that date (plus the terminal settlement delay), not the product
(schedule-2) maturity. A schedule-2 knock-out keeps paying with the product's terminal. Monte Carlo always did this;
PDE 1D (and the LV PDE and Heston/SLV ADI routes through the same family), QUAD V1 and QUAD V2 paid at the product
maturity (T24 finding: MC 93.45 +- 0.09 vs 91.34 on the matrix contract).

- every engine: the schedule-1 knock-out rows pay with the never-knocked-in maturity row, which pays at schedule 1's
  maturity plus the terminal delay; schedule-2 rows pay with the knocked-in settlement (ABSOLUTE and REBASED, fresh and
  aged, numeric and dated, a settlement lag, a knock-out decided at valuation);
- a schedule-1 knock-out decided at valuation is worth its cash x DF(schedule 1's payment) on every engine;
- PDE 1D, QUAD V1 and QUAD V2 agree with an MC reference within the matrix gate (K SE + the |L0 - L1| refinement step)
  on pv and observations.discounted_cash (fresh discrete and continuous KI, aged dated).
"""
from __future__ import annotations

import warnings
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from event_statistics import matrix_cases as M
from test_asset.equity.engine_mc import mc_price_path_routes as routes
from test_asset.equity.engine_pde.test_dated_expiry_settles import env as dated_env

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
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.option_enums import PostKOScheduleMode

warnings.filterwarnings("ignore", category=DeprecationWarning)

EXPIRY = CouponPayType.EXPIRY
TIME_TOL = 1e-12

# --- contracts ------------------------------------------------------------------------------------------------------

INITIAL = datetime(2025, 1, 5)
#: monthly fixings on snapped contract days; schedule 1 (and the discrete KI) the first six, schedule 2 all twelve
DAYS = [round(k / 12 * 365) for k in range(1, 13)]


def _on(day: int) -> datetime:
    return INITIAL + timedelta(days=day)


def _schedule(form: str, days, level: float) -> ObservationSchedule:
    if form == "dated":
        return ObservationSchedule(records=[ObservationRecord(observation_date=_on(d), barrier=level) for d in days])
    return ObservationSchedule(records=[ObservationRecord(observation_time=d / 365.0, barrier=level) for d in days])


def monthly(form: str, day: int = 0, *, lag: float = 0.0) -> KnockOutResetSnowballOption:
    """One-year ABSOLUTE KO-reset, EXPIRY pay: schedule 1 (KO 103, 15% p.a.) and a discrete KI 80 on the first six
    monthly fixings, schedule 2 (KO 95, 3% p.a.) on all twelve; principal included. Aged to ``day`` (numeric form by
    ``time_shift``, dated form by the valuation date of its environment); ``lag`` a year-fraction settlement lag."""
    dated = form == "dated"
    product = KnockOutResetSnowballOption(
        initial_price=100.0, strike=100.0, maturity=None if dated else 1.0, initial_date=INITIAL if dated else None,
        exercise_date=_on(365) if dated else None,
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=_schedule(form, DAYS[:6], 103.0), ki_barrier=80.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_schedule=_schedule(form, DAYS[:6], 80.0)),
        post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03, ko_observation_type=ObservationType.DISCRETE,
                                          ko_observation_schedule=_schedule(form, DAYS, 95.0)),
        accrual_config=AccrualConfig(is_annualized=True, coupon_pay_type=EXPIRY),
        payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=True),
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
        settlement_convention=(SettlementConvention(lag=lag, lag_unit=SettlementLagUnit.YEAR_FRACTION)
                               if lag else None),
    )
    product.validate()
    if not dated and day:
        product.time_shift(day / 365.0, _on(day), dated_env(100.0, _on(day)), keep_landing=True)
    return product


def monthly_env(form: str, day: int = 0):
    return dated_env(100.0, _on(day))


def _ko_at_valuation():
    """The matrix KO-reset with a schedule-1 record at valuation and the spot above its level (decided now)."""
    product = routes.ko_reset(pay=EXPIRY)
    product.barrier_config = BarrierConfig(
        ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
        ko_observation_schedule=ObservationSchedule(records=[
            ObservationRecord(observation_time=t, barrier=103.0) for t in (0.0, 0.25, 0.5, 0.75, 1.0)]),
        ki_barrier=80.0, ki_observation_type=ObservationType.DISCRETE,
        ki_observation_schedule=ObservationSchedule(records=[
            ObservationRecord(observation_time=t, barrier=80.0) for t in (0.25, 0.5, 0.75, 1.0)]))
    product.validate()
    return product


#: name -> (product, env); every one is a KO-reset with EXPIRY pay whose schedule 1 ends before the product maturity
CONTRACTS = {
    "matrix_dki": (lambda: routes.ko_reset(pay=EXPIRY), routes.flat_env),
    "matrix_cki": (lambda: routes.ko_reset(ki="continuous", pay=EXPIRY), routes.flat_env),
    "numeric_fresh": (lambda: monthly("numeric"), lambda: monthly_env("numeric")),
    "numeric_aged": (lambda: monthly("numeric", 91), lambda: monthly_env("numeric", 91)),
    "dated_fresh": (lambda: monthly("dated"), lambda: monthly_env("dated")),
    "dated_aged": (lambda: monthly("dated", 91), lambda: monthly_env("dated", 91)),
    "numeric_lagged": (lambda: monthly("numeric", lag=0.01), lambda: monthly_env("numeric")),
}

# --- engines --------------------------------------------------------------------------------------------------------

ENGINES = {
    "mc": lambda: SnowballMCEngine(params=MCParams(num_paths=4096, seed=7)),
    "pde_1d": lambda: KOResetSnowballPDESolver(params=PDEParams(accuracy="fast")),
    "quad_v1": lambda: KOResetSnowballQuadEngine(QuadParams(grid_points=401)),
    "quad_v2": lambda: KOResetSnowballQuadEngineV2(),
}
#: the PDE family's other routes (they share KOResetSnowballPDESolver's records), on their own markets
VOL_ENGINES = {
    "lv_pde_1d": (M.GROUPS["lv"].grid["ko_reset"]["lv_pde_1d"]["L0"], M.GROUPS["lv"].env),
    "heston_adi": (lambda: HestonKOResetSnowballPDESolver(M.HP, n_x=120, n_v=16, n_t=60), M.GROUPS["heston"].env),
}


def _pre_maturity_payment(product, env) -> float:
    """Schedule 1's maturity plus the terminal settlement delay -- computed from the contract, not an engine."""
    terminal = resolve_terminal_timing(product, env)
    pre = max(r.observation_time for r in product._resolve_ko_schedule(product.barrier_config, env)[0])
    return float(pre) + (float(terminal.payment_time) - float(terminal.determination_time))


def _assert_pays_at_schedule_one_end(stats, product, env, label):
    obs = stats.observations
    tags = np.asarray(obs.schedule)
    pre, post = tags == "pre", tags == "post"
    assert pre.any() and post.any(), label
    expected = _pre_maturity_payment(product, env)
    terminal = resolve_terminal_timing(product, env)
    assert stats.maturity.payment_time == pytest.approx(expected, abs=TIME_TOL), (label, stats.maturity)
    np.testing.assert_allclose(obs.payment_times[pre], stats.maturity.payment_time, rtol=0, atol=TIME_TOL,
                               err_msg=f"{label}: schedule-1 knock-out rows pay with the never-knocked-in maturity")
    np.testing.assert_allclose(obs.payment_times[post], float(terminal.payment_time), rtol=0, atol=TIME_TOL,
                               err_msg=f"{label}: schedule-2 knock-out rows pay with the product terminal")
    if obs.payment_dates is not None:
        dates = np.asarray(obs.payment_dates, dtype=object)
        assert all(d == stats.maturity.payment_date for d in dates[pre]), (label, dates[pre], stats.maturity)
    # the cash to pay is the knock-out cash: discounted at the payment the row states
    pay_df = np.array([float(env.get_discount_factor(t)) for t in obs.payment_times])
    np.testing.assert_allclose(obs.cash_to_pay * pay_df, obs.discounted_cash, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("engine", list(ENGINES))
@pytest.mark.parametrize("contract", list(CONTRACTS))
def test_schedule_one_knock_outs_pay_with_the_never_knocked_in_maturity(engine, contract):
    product_of, env_of = CONTRACTS[contract]
    product, env = product_of(), env_of()
    stats = ENGINES[engine]().calculate_event_stats(product, env)
    _assert_pays_at_schedule_one_end(stats, product, env, f"{engine} {contract}")


@pytest.mark.parametrize("engine", list(VOL_ENGINES))
def test_the_vol_model_pde_routes_pay_schedule_one_at_its_maturity(engine):
    make, env_of = VOL_ENGINES[engine]
    product, env = routes.ko_reset(pay=EXPIRY), env_of()
    _assert_pays_at_schedule_one_end(make().calculate_event_stats(product, env), product, env, engine)


def test_rebased_schedule_one_knock_outs_pay_at_schedule_one_maturity():
    """REBASED (Monte Carlo only: the grid engines refuse it): the same rule, schedule 2 paying per path."""
    product, env = routes.ko_reset(PostKOScheduleMode.REBASED, pay=EXPIRY), routes.flat_env()
    stats = ENGINES["mc"]().calculate_event_stats(product, env)
    obs = stats.observations
    pre = np.asarray(obs.schedule) == "pre"
    assert pre.any()
    expected = _pre_maturity_payment(product, env)
    assert stats.maturity.payment_time == pytest.approx(expected, abs=TIME_TOL)
    np.testing.assert_allclose(obs.payment_times[pre], expected, rtol=0, atol=TIME_TOL)


@pytest.mark.parametrize("engine", list(ENGINES) + list(VOL_ENGINES))
def test_a_schedule_one_knock_out_decided_at_valuation_pays_at_schedule_one_maturity(engine):
    """price() of a schedule-1 knock-out decided at valuation: its contractual cash x DF(schedule 1's payment), on
    every engine (the decision's cash is the product's, the payment time the contract's: no engine number)."""
    if engine in VOL_ENGINES:
        make, env_of = VOL_ENGINES[engine]
        env = env_of(spot=110.0)
    else:
        make, env = ENGINES[engine], routes.flat_env(110.0)
    product = _ko_at_valuation()
    decision = product.decide_observations_at_valuation(env.spot, env)
    assert decision.knocked_out and decision.ko_regime == "pre"
    pay = _pre_maturity_payment(product, env)
    expected = float(decision.ko_record.payoff) * float(env.get_discount_factor(pay))
    assert make().price(product, env) == pytest.approx(expected, rel=1e-12)
    stats = make().calculate_event_stats(product, env)
    pre = np.asarray(stats.observations.schedule) == "pre"
    np.testing.assert_allclose(stats.observations.payment_times[pre], pay, rtol=0, atol=TIME_TOL)


# --- cross-engine agreement with the MC (the matrix gate) -----------------------------------------------------------

#: independent PSEUDO runs (PSEUDO batch seeds are seed + 1000 b: consecutive seeds do not overlap)
REFERENCE_SEEDS = tuple(range(201, 209))
REFERENCE_PATHS = 16384
GRID = ("pde_1d", "quad_v1", "quad_v2")
GATED = ("pv", "total_discounted_cash", "observations.discounted_cash", "maturity.discounted_cash")
AGREEMENT = {
    "matrix_dki": CONTRACTS["matrix_dki"],
    "matrix_cki": CONTRACTS["matrix_cki"],
    "dated_aged": CONTRACTS["dated_aged"],
}


def _mc_reference(contract):
    product_of, env_of = AGREEMENT[contract]
    runs = [M.matrix_fields_of(SnowballMCEngine(params=MCParams(num_paths=REFERENCE_PATHS, seed=seed))
                               .calculate_event_stats(product_of(), env_of()))
            for seed in REFERENCE_SEEDS]
    return M.pool_runs(runs, paths_per_run=REFERENCE_PATHS)


_REFERENCES = {}


def _reference(contract):
    if contract not in _REFERENCES:
        _REFERENCES[contract] = _mc_reference(contract)
    return _REFERENCES[contract]


@pytest.mark.parametrize("engine", GRID)
@pytest.mark.parametrize("contract", list(AGREEMENT))
def test_grid_engines_agree_with_the_mc_on_pv_and_knock_out_cash(contract, engine):
    """AC32: ``|grid L0 - MC| <= K SE + |L0 - L1| + floor`` (the matrix's K, SE floors and level factories) on pv, the
    cash total, every knock-out row's discounted cash and the maturity cash. Before D30 the grid engines paid the
    schedule-1 knock-outs at the product maturity: pv ~2.1 (24 SE) under the MC on the matrix contract."""
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
