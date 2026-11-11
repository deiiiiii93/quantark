"""Vol-model MC event statistics (spec T10): local vol, Heston (Euler / QE), SLV and SLV-QE, Snowball and Phoenix.

The route grid (``mc_price_path_routes``) covers AC7 / AC8 / AC13 / AC21 on every vol-model route (the Snowball /
KO-reset and Phoenix test modules run their identities over it). Here, on every vol-model engine of both families:

- AC9: a knock-in after a certain knock-out is no knock-in;
- AC10: knock-out at valuation, carried knock-in, knock-in observed at valuation, no KI barrier;
- AC12: deterministic payment rows are DF-consistent;
- the Heston-SLV conditional control (``rqmc_heston_conditional_control``): the price is ``native - sampled proxy +
  exact proxy`` and so is every recorded outcome. Checked by rebuilding the knock-out rows from the price's own
  native paths, proxy paths and proxy base rows (spies, plain NumPy + the normal CDF), against a large native run
  for bias, and the refusal of an estimate that breaks a probability bound (never clipped).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.special import ndtr

from quantark.asset.equity.engine.mc import snowball_mc_engine as snowball_module
from quantark.asset.equity.engine.mc.phoenix_vol_mc_engines import (
    HestonPhoenixMCEngine,
    HestonSLVPhoenixMCEngine,
    HestonSLVQEPhoenixMCEngine,
    LocalVolPhoenixMCEngine,
    QEPhoenixMCEngine,
)
from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (
    HestonSLVQESnowballMCEngine,
    HestonSLVSnowballMCEngine,
    HestonSnowballMCEngine,
    LocalVolSnowballMCEngine,
    QESnowballMCEngine,
)
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.enum.engine_enums import HestonMCScheme
from quantark.util.exceptions import PricingError
from quantark.volmodels.slv.leverage import LeverageSurface

from test_asset.equity.engine_mc import mc_price_path_routes as grid
from test_asset.equity.engine_mc import mc_same_path_reference as reference
from test_asset.equity.engine_mc.mc_price_path_routes import HESTON, ROUTES

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROUND_OFF = 1e-12


def _flat(spot):
    return grid.flat_env(spot)


def _grid(spot):
    return grid.grid_env(spot)


def _grid_flat(spot):
    return grid.grid_env(spot, skew=False)


#: name -> (engine factory per family, env factory by spot)
SNOWBALL_ENGINES = {
    "lv": (lambda: LocalVolSnowballMCEngine(grid.mc()), _grid),
    "heston_euler": (lambda: HestonSnowballMCEngine(HESTON, params=grid.mc(), scheme=HestonMCScheme.EULERLOG), _flat),
    "qe": (lambda: QESnowballMCEngine(HESTON, params=grid.mc()), _flat),
    "qe_antithetic": (lambda: QESnowballMCEngine(HESTON, params=grid.mc(use_antithetic=True)), _flat),
    "slv": (lambda: HestonSLVSnowballMCEngine(HESTON, params=grid.mc(), leverage_surface=grid.unit_leverage()),
            _grid_flat),
    "slvqe": (lambda: HestonSLVQESnowballMCEngine(HESTON, params=grid.mc(), leverage_surface=grid.skew_leverage()),
              _grid_flat),
}
PHOENIX_ENGINES = {
    "lv": (lambda: LocalVolPhoenixMCEngine(grid.mc()), _grid),
    "heston_euler": (lambda: HestonPhoenixMCEngine(HESTON, params=grid.mc(), scheme=HestonMCScheme.EULERLOG), _flat),
    "qe": (lambda: QEPhoenixMCEngine(HESTON, params=grid.mc()), _flat),
    "qe_antithetic": (lambda: QEPhoenixMCEngine(HESTON, params=grid.mc(use_antithetic=True)), _flat),
    "slv": (lambda: HestonSLVPhoenixMCEngine(HESTON, params=grid.mc(), leverage_surface=grid.unit_leverage()),
            _grid_flat),
    "slvqe": (lambda: HestonSLVQEPhoenixMCEngine(HESTON, params=grid.mc(), leverage_surface=grid.skew_leverage()),
              _grid_flat),
}
CASES = [("snowball", name) for name in SNOWBALL_ENGINES] + [("phoenix", name) for name in PHOENIX_ENGINES]


def _product(family, ko_dates=grid.KO_DATES, *, ko_barrier=105.0, ki="continuous", ki_barrier=70.0, ki_dates=None,
             knocked_in=False, has_ki=True):
    if not has_ki:
        ki_kw = {}
    elif ki == "continuous":
        ki_kw = dict(ki_barrier=ki_barrier, ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True)
    else:
        ki_kw = dict(ki_barrier=ki_barrier, ki_observation_type=ObservationType.DISCRETE,
                     ki_observation_dates=list(ki_dates if ki_dates is not None else ko_dates))
    barrier = BarrierConfig(ko_barrier=ko_barrier, ko_rate=0.15 if family == "snowball" else 0.0,
                            ko_observation_type=ObservationType.DISCRETE, ko_observation_dates=list(ko_dates), **ki_kw)
    common = dict(initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0, barrier_config=barrier,
                  payoff_config=PayoffConfig(rebate_rate=0.05 if family == "snowball" else 0.0,
                                             include_principal=True))
    if family == "snowball":
        product = SnowballOption(is_reverse=False, **common)
    else:
        product = PhoenixOption(coupon_config=CouponBarrierConfig(
            coupon_barrier=[80.0] * len(ko_dates), coupon_rate=0.02, coupon_pay_type=CouponPayType.INSTANT,
            day_count_convention=DayCountConvention.ACT_365, memory_coupon=True), **common)
    if knocked_in:
        product._otc_lifecycle_knocked_in = True
    return product


def _run(family, name, product, spot=100.0):
    engines = SNOWBALL_ENGINES if family == "snowball" else PHOENIX_ENGINES
    factory, env_of = engines[name]
    env = env_of(spot)
    return factory(), env, factory().calculate_event_stats(product, env)


def _identities(stats):
    assert abs(stats.probability_residual) <= ROUND_OFF
    assert abs(stats.reconciliation_error) <= ROUND_OFF * max(1.0, abs(stats.pv))
    assert stats.knock_in.probability_no_ko == stats.ki_settlement.probability
    assert set(stats.unavailable) <= {"knock_in.expected_time", "standard_errors",
                                      "standard_errors.knock_in.expected_time"}


# --- AC9 -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("family,name", CASES)
@pytest.mark.parametrize("ki", ["discrete", "continuous"])
def test_a_knock_in_after_a_certain_knock_out_is_no_knock_in(family, name, ki):
    if ki == "discrete":
        product = _product(family, ko_barrier=[1.0, 105.0, 105.0, 105.0], ki="discrete", ki_barrier=99.0,
                           ki_dates=(0.5, 0.75, 1.0))
        spot = 100.0
    else:
        product = _product(family, (0.0, *grid.KO_DATES), ki_barrier=108.0)
        spot = 110.0
    engine, env, stats = _run(family, name, product, spot)
    paths_of = reference.snowball_paths if family == "snowball" else reference.phoenix_paths
    ref = paths_of(engine, product, env)
    assert ref["is_ko"].all() and ref["ki_ever"].mean() > 0.2          # KI-ever would be large
    _identities(stats)
    assert stats.knock_in.probability == 0.0 and stats.knock_in.expected_loss == 0.0
    assert stats.ki_settlement.probability == 0.0
    assert np.all(stats.ki_table.probability == 0.0)
    assert stats.unavailable["knock_in.expected_time"].startswith("undefined:")


# --- AC10 ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("family,name", CASES)
def test_knock_out_at_valuation_has_no_knock_in(family, name):
    _, _, stats = _run(family, name, _product(family, (0.0, *grid.KO_DATES)), 110.0)
    _identities(stats)
    assert stats.observations.times[0] == 0.0 and stats.observations.probability[0] == 1.0
    assert stats.knock_in.probability == 0.0 and stats.ki_table.size == 0 and stats.expected_lifetime == 0.0


@pytest.mark.parametrize("family,name", CASES)
def test_carried_knock_in_is_a_knock_in_at_zero(family, name):
    _, _, stats = _run(family, name, _product(family, knocked_in=True))
    _identities(stats)
    assert stats.knock_in.probability == 1.0 and stats.knock_in.expected_time == 0.0
    np.testing.assert_array_equal(stats.ki_table.times, [0.0])
    np.testing.assert_array_equal(stats.ki_table.probability, [1.0])
    assert stats.maturity.probability == 0.0


@pytest.mark.parametrize("family,name", CASES)
@pytest.mark.parametrize("ki", ["continuous", "discrete"])
def test_knock_in_observed_at_valuation_is_a_knock_in_at_zero(family, name, ki):
    _, _, stats = _run(family, name, _product(family, ki=ki, ki_dates=(0.0, *grid.KO_DATES)), 65.0)
    _identities(stats)
    assert stats.knock_in.probability == 1.0 and stats.knock_in.expected_time == 0.0
    assert stats.ki_table.times[0] == 0.0 and stats.ki_table.probability[0] == 1.0
    assert np.all(stats.ki_table.probability[1:] == 0.0)


@pytest.mark.parametrize("family,name", CASES)
def test_no_knock_in_barrier_reports_computed_zeros(family, name):
    _, _, stats = _run(family, name, _product(family, has_ki=False))
    _identities(stats)
    assert stats.knock_in.probability == 0.0 and stats.knock_in.expected_loss == 0.0
    assert stats.ki_settlement.probability == 0.0 and stats.ki_settlement.discounted_cash == 0.0
    assert stats.ki_table.size == 0


# --- AC12 / AC13 ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("family,name", CASES)
def test_deterministic_payment_rows_and_knock_in_table(family, name):
    _, env, stats = _run(family, name, _product(family, ki="discrete"))
    _identities(stats)
    tables = [stats.observations] + ([stats.coupons] if stats.coupons is not None else [])
    for table in tables:
        dfs = np.array([env.get_discount_factor(float(t)) for t in table.payment_times])
        np.testing.assert_allclose(table.discounted_cash, table.cash_to_pay * dfs, rtol=1e-12, atol=1e-14)
    for row in (stats.maturity, stats.ki_settlement):
        assert math.isclose(row.discounted_cash, row.cash_to_pay * env.get_discount_factor(row.payment_time),
                            rel_tol=1e-12, abs_tol=1e-14)
    table = stats.ki_table
    np.testing.assert_array_equal(table.times, grid.KO_DATES)
    assert math.isclose(float(table.probability.sum()), stats.knock_in.probability, rel_tol=1e-12, abs_tol=1e-14)
    if stats.knock_in.probability > 0.0:
        mean = float(np.sum(table.times * table.probability)) / stats.knock_in.probability
        assert math.isclose(mean, stats.knock_in.expected_time, rel_tol=1e-12)


# --- the Heston-SLV conditional control (rqmc_heston_conditional_control) ------------------------------------------

CONTROL_ROUTES = ["slvqe_sb_conditional_control", "slvqe_sb_conditional_control_skew"]


@pytest.mark.parametrize("name", CONTROL_ROUTES)
def test_conditional_control_outcomes_reconcile_to_the_price(name):
    route = ROUTES[name]
    price = route.engine().price(route.product(), route.env())
    engine = route.engine()
    stats = engine.calculate_event_stats(route.product(), route.env())
    assert stats.pv == price
    assert {r["kind"] for r in engine._last_stats_records} == {"control_corrected"}
    _identities(stats)
    assert stats.standard_errors is not None and "observations.probability" in stats.standard_errors


def _first_ko(spots, barriers):
    hit = spots >= barriers
    any_hit = hit.any(axis=1)
    return any_hit, np.where(any_hit, np.argmax(hit, axis=1), -1)


def test_conditional_control_rows_are_native_minus_sampled_plus_exact(monkeypatch):
    """The knock-out rows rebuilt from the price's own arrays: the native paths and the proxy paths each kernel call
    priced, and the proxy base rows / loadings the exact control integrated. Exact first-knock-out mass per base row
    by the normal CDF on its thresholds (KO at row j: the factor above row j's threshold, below every earlier
    one). The corrected row is (native count - proxy count + g * exact mass) / native paths."""
    route = ROUTES["slvqe_sb_conditional_control_skew"]
    engine, product, env = route.engine(), route.product(), route.env()
    kernel_paths, bases = [], {}
    original_kernel = engine._compute_payoffs

    def kernel_spy(product_, env_, paths, *args, **kwargs):
        kernel_paths.append(np.array(paths, copy=True))
        return original_kernel(product_, env_, paths, *args, **kwargs)

    original_moments = snowball_module.conditional_standard_snowball_moments

    def moments_spy(**kwargs):
        bases.setdefault(id(kwargs["base_paths"]), (kwargs["base_paths"], kwargs["log_spot_factor_loadings"],
                                                    np.asarray(kwargs["ko_indices"])))
        return original_moments(**kwargs)

    monkeypatch.setattr(engine, "_compute_payoffs", kernel_spy)
    monkeypatch.setattr(snowball_module, "conditional_standard_snowball_moments", moments_spy)
    stats = engine.calculate_event_stats(product, env)
    profile = product.get_ko_observation_profile(env)
    barriers = np.asarray(profile["barriers"], dtype=float)
    rows = barriers.size
    batches = list(bases.values())
    # native, proxy per batch; then the RQMC result assembly's unrecorded rerun of batch 0
    assert len(batches) == 3 and len(kernel_paths) == 2 * len(batches) + 1
    native_n = 0
    native_total, sampled_total, exact_total = np.zeros(rows), np.zeros(rows), np.zeros(rows)
    for b, (base, loading, ko_indices) in enumerate(batches):
        native, proxy = kernel_paths[2 * b], kernel_paths[2 * b + 1]
        assert native.shape == proxy.shape
        g = native.shape[0] // base.shape[0]
        native_n += native.shape[0]
        for paths, total in ((native, native_total), (proxy, sampled_total)):
            hit, row = _first_ko(paths[:, ko_indices + 1], barriers)
            total += np.bincount(row[hit], minlength=rows)
        cols = ko_indices + 1
        thresholds = (np.log(barriers)[None, :] - np.log(base[:, cols])) / loading[:, cols]
        upper = np.minimum.accumulate(np.concatenate([np.full((base.shape[0], 1), np.inf), thresholds], axis=1),
                                      axis=1)[:, :-1]
        mass = np.maximum(ndtr(upper) - ndtr(thresholds), 0.0)
        exact_total += g * mass.sum(axis=0)
    assert np.any(native_total != sampled_total)                             # the correction is not trivially 0
    corrected = (native_total - sampled_total + exact_total) / native_n
    np.testing.assert_allclose(stats.observations.probability, corrected, rtol=1e-12, atol=1e-14)
    native_only = native_total / native_n
    assert not np.allclose(stats.observations.probability, native_only, rtol=0, atol=1e-6)
    _identities(stats)


def test_conditional_control_outcomes_are_unbiased_for_the_native_outcomes():
    """16 RQMC batches of the control estimator against 200k native PSEUDO paths of the same SLV-QE model (same
    leverage, scheme and sub-steps): every outcome within 4 standard errors (batch-mean SE on the control side,
    binomial on the native side)."""
    params = MCParams(num_paths=2048, seed=1, rqmc_min_batches=16, rqmc_max_batches=16, rqmc_target_std=1e-12,
                      rqmc_paths_mode="per_batch")
    control = HestonSLVQESnowballMCEngine(HESTON, params=params, method=grid.RQMC,
                                          leverage_surface=grid.skew_leverage(), martingale_correction=True,
                                          substeps_per_interval=2, rqmc_heston_conditional_control=True,
                                          rqmc_spot_strata=2)
    stats = control.calculate_event_stats(grid.affine_snowball(), grid.grid_env(skew=False))
    native = HestonSLVQESnowballMCEngine(HESTON, params=MCParams(num_paths=200_000, seed=101),
                                         leverage_surface=grid.skew_leverage(), martingale_correction=True,
                                         substeps_per_interval=2)
    ref = native.calculate_event_stats(grid.affine_snowball(), grid.grid_env(skew=False))
    se = stats.standard_errors
    for field in ("observations.probability", "ki_table.probability", "maturity.probability",
                  "ki_settlement.probability"):
        a = np.atleast_1d(stats.require(field))
        b = np.atleast_1d(ref.require(field))
        bound = 4.0 * np.sqrt(np.atleast_1d(se[field]) ** 2 + b * (1.0 - b) / 200_000)
        assert np.all(np.abs(a - b) <= bound), (field, a, b, bound)


def test_a_control_corrected_estimate_below_zero_is_refused_not_clipped():
    """A deep knock-in (50) on a falling leverage, 64 paths x 2 batches: one first-knock-in row's control-corrected
    estimate is below zero (measured: 1 of 32 seeds at 64 paths, 0 of 32 at 256 and 1024). The engine refuses."""
    strikes = np.array(list(100.0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    leverage = LeverageSurface(time_grid=np.linspace(0.0, 2.0, 5), strike_grid=strikes,
                               leverage_grid=np.tile(np.linspace(1.8, 0.6, strikes.size), (5, 1)))
    product = SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(ko_barrier=[105.0, 140.0, 105.0, 105.0], ko_rate=0.12,
                                     ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_dates=list(grid.KO_DATES), ki_barrier=50.0,
                                     ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_dates=list(grid.KO_DATES)))
    params = MCParams(num_paths=64, seed=0, rqmc_min_batches=2, rqmc_max_batches=2, rqmc_target_std=1e-12,
                      rqmc_paths_mode="per_batch")
    engine = HestonSLVQESnowballMCEngine(HESTON, params=params, method=grid.RQMC, leverage_surface=leverage,
                                         martingale_correction=True, substeps_per_interval=2,
                                         rqmc_heston_conditional_control=True, rqmc_spot_strata=2)
    env = grid.grid_env(skew=False)
    price = engine.price(product, env)
    with pytest.raises(PricingError, match="control-corrected outcome estimate"):
        engine.calculate_event_stats(product, env)
    assert engine.price(product, env) == price                               # the price is untouched
