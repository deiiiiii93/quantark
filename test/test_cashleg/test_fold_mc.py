"""Cash-leg folding on the Monte Carlo autocallable engines (ship spec 2026-10-21, task T7).

The overlay reaches each path's payoff where the product's own cash does: the knock-out record's
payoff (before the path's settlement discount), each maturity branch's terminal payoff, on every
route that calls the per-path payoff kernel (serial PSEUDO / QUASI, Dask batches, RQMC batches and
their finalizer statistics batch). The estimators with their own assembly fold at their own sites:
LocalVol ``one_step_survival`` (branch weights times record cash) and the Heston exact conditional
moments (interval probabilities times record cash, and the Heston-SLV control built on them).

The MC event route (``price_with_events``) runs the same paths as a serial PSEUDO / QUASI price, so
knock-out and knock-in cash fold to round-off against it (AC4 / AC5). KO-reset MC reports one
maturity outcome, so its legs are checked against the PDE 1D event route at Monte Carlo tolerance
(AC6); the product twins (AC14) pin the cash against the product itself on every route, same paths.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.base_engine import BaseEngine
from quantark.asset.equity.engine.mc import (
    HestonSLVQESnowballMCEngine,
    HestonSLVSnowballMCEngine,
    HestonSnowballMCEngine,
    LocalVolSnowballMCEngine,
    QESnowballMCEngine,
)
from quantark.asset.equity.engine.mc.autocallable_execution_adapters import AutocallableAdaptiveMCAdapter
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.phoenix_vol_mc_engines import (
    HestonPhoenixMCEngine,
    LocalVolPhoenixMCEngine,
)
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.cashleg import EventType
from quantark.cashleg.folding import FoldedEventCash, FoldOverlay, _event_structure, fold_cash_legs
from quantark.execution.contracts import OutputKind, PricingOperation
from quantark.execution.errors import CapabilityError
from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset
from quantark.montecarlo.conditional_snowball import conditional_standard_snowball_moments
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum import CouponPayType, PostKOScheduleMode
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError

from intraday.conftest import dated_ko_reset, dated_phoenix, dated_snowball, flat_env
from pde_3a.products import KO_RESET, env as ko_reset_env
from test_cashleg._autocallable_helpers import make_env, make_phoenix, make_snowball
from test_cashleg.test_fold_quad_v1 import (
    _autocallable_ko_cash,
    _branch_overlay,
    _hand_overlay,
    _knocked_in,
    _ko_cash_diff_overlay,
    _ko_dates,
    _legs,
    _p,
    _route,
)
from test_cashleg.test_folding import UNIT
from test_asset.equity.engine_pde.test_snowball_vol_model_engines import (
        _env as _vol_env,
    _heston,
    _unit_leverage,
)

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
T0, LIVE = datetime(2026, 3, 16), datetime(2026, 4, 1)
KO_DATES = (0.25, 0.5, 0.75, 1.0)
PATHS = 20_000
#: the spec's bar for routes sharing one path set with their event route (AC4 relative, AC5 per notional)
IDENTITY = 1e-10
PARITY = 1e-9
#: KO-reset MC against the PDE 1D route: Monte Carlo standard errors of the folded total
KO_RESET_SE = 3.0


def _params(**kw):
    return MCParams(num_paths=kw.pop("num_paths", PATHS), seed=kw.pop("seed", 7), **kw)


def _gbm(cls, method=MonteCarloMethod.PSEUDO, **kw):
    return lambda: cls(_params(), method=method, **kw)


def _skewed_env(spot=100.0, r=0.03, q=0.01, valuation=datetime(2026, 1, 1)):
    from quantark.param import ContinuousDividendYield, FlatRateCurve, GridVolSurface, SpotQuote
    from quantark.priceenv import PricingEnvironment

    strikes = list(100.0 * np.exp(np.linspace(-0.6, 0.4, 11)))
    maturities = list(np.linspace(0.25, 1.0, 4))
    grid = np.array([[0.34 - 0.22 * (np.log(k / 100.0) + 0.6) for k in strikes] for _ in maturities])
    return PricingEnvironment(
        rate_curve=FlatRateCurve(r), valuation_date=valuation, spot_quote=SpotQuote(spot=spot),
        vol_surface=GridVolSurface(strikes, maturities, grid), div_yield=ContinuousDividendYield(q),
    )


def _phoenix_product(**kw):
    return make_phoenix(ko_dates=KO_DATES, coupon_barrier=(80.0,) * 4, ki_barrier=70.0, **kw)


def _heston_env(spot=100.0):
    return _vol_env(s0=spot)


def _heston_snowball(ko_rate=0.12):
    """Discrete knock-out and knock-in (the exact conditional-moment estimator's domain), unit multiplier."""
    from quantark.asset.equity.product.option.snowball_config import BarrierConfig
    from quantark.asset.equity.product.option.snowball_option import SnowballOption
    from quantark.util.enum import ObservationType

    return SnowballOption(
        initial_price=100.0, strike=100.0, maturity=1.0, contract_multiplier=1.0, is_reverse=False,
        barrier_config=BarrierConfig(
            ko_barrier=105.0, ko_rate=ko_rate, ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=list(KO_DATES), ki_barrier=75.0,
            ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=list(KO_DATES),
        ),
    )


#: name -> (engine factory, product factory, env factory(spot))
FAMILIES = {
    "snowball": (_gbm(SnowballMCEngine), lambda: make_snowball(ko_dates=KO_DATES), make_env),
    "snowball_quasi": (_gbm(SnowballMCEngine, MonteCarloMethod.QUASI), lambda: make_snowball(ko_dates=KO_DATES),
                       make_env),
    "dated_snowball": (_gbm(SnowballMCEngine), lambda: dated_snowball(CAL, T0),
                       lambda spot=100.0: flat_env(LIVE, spot=spot)),
    "phoenix": (_gbm(PhoenixMCEngine), _phoenix_product, make_env),
    "phoenix_quasi": (_gbm(PhoenixMCEngine, MonteCarloMethod.QUASI), _phoenix_product, make_env),
    "phoenix_expiry": (_gbm(PhoenixMCEngine), lambda: _phoenix_product(coupon_pay=CouponPayType.EXPIRY),
                       make_env),
    "dated_phoenix": (_gbm(PhoenixMCEngine), lambda: dated_phoenix(CAL, T0),
                      lambda spot=100.0: flat_env(LIVE, spot=spot)),
    "lv_snowball": (_gbm(LocalVolSnowballMCEngine), lambda: make_snowball(ko_dates=KO_DATES), _skewed_env),
    "lv_snowball_quasi": (_gbm(LocalVolSnowballMCEngine, MonteCarloMethod.QUASI),
                          lambda: make_snowball(ko_dates=KO_DATES), _skewed_env),
    "lv_phoenix": (_gbm(LocalVolPhoenixMCEngine), _phoenix_product, _skewed_env),
    "heston_snowball": (lambda: HestonSnowballMCEngine(_heston(), params=_params()), _heston_snowball,
                        _heston_env),
    "heston_snowball_quasi": (lambda: HestonSnowballMCEngine(_heston(), params=_params(),
                                                             method=MonteCarloMethod.QUASI),
                              _heston_snowball, _heston_env),
    "slv_snowball": (lambda: HestonSLVSnowballMCEngine(_heston(), params=_params(),
                                                       leverage_surface=_unit_leverage()),
                     _heston_snowball, _heston_env),
    "heston_phoenix": (lambda: HestonPhoenixMCEngine(_heston(), params=_params()), _phoenix_product, _heston_env),
}
SNOWBALLS = [f for f in FAMILIES if "snowball" in f]
PHOENIXES = [f for f in FAMILIES if "phoenix" in f]
# ATM, just under the knock-out level, just above the knock-in level
SPOTS = {"snowball": (100.0, 102.5, 76.0), "phoenix": (100.0, 104.5, 71.0),
         "dated_snowball": (100.0, 102.5, 76.0), "dated_phoenix": (100.0, 102.5, 76.0),
         "heston": (100.0, 104.5, 76.0), "ko_reset": (100.0, 102.5, 81.0)}


def _spots(family):
    for key in ("dated_snowball", "dated_phoenix", "heston", "phoenix", "snowball"):
        if key in family:
            return SPOTS[key]
    raise KeyError(family)


def _case(family, spot=100.0):
    engine, product, env = FAMILIES[family]
    return engine, product(), env(spot)


def _ko_reset_case(spot=100.0, method=MonteCarloMethod.PSEUDO):
    return (lambda: SnowballMCEngine(_params(), method=method)), make_ko_reset(KO_RESET), ko_reset_env(spot=spot)


def _dated_ko_reset_case(spot=100.0):
    return (lambda: SnowballMCEngine(_params())), dated_ko_reset(CAL, T0), flat_env(LIVE, spot=spot)


def _weights(engine):
    """The engine's own event weights of its last (plain) price, from the same paths."""
    result = engine.get_last_result()
    return result.ko_probability, result.v0_probability, result.v1_probability


# --- contract ----------------------------------------------------------------------------------

MC_CLASSES = (SnowballMCEngine, PhoenixMCEngine, LocalVolSnowballMCEngine, HestonSnowballMCEngine,
              QESnowballMCEngine, HestonSLVSnowballMCEngine, HestonSLVQESnowballMCEngine,
              LocalVolPhoenixMCEngine, HestonPhoenixMCEngine)


def test_mc_engines_declare_folding():
    for cls in MC_CLASSES:
        assert cls.supports_cash_leg_folding is True


@pytest.mark.parametrize("family", list(FAMILIES))
def test_an_empty_overlay_is_the_plain_price_bitwise(family):
    engine, product, env = _case(family)
    overlay = _hand_overlay(product, env)
    assert overlay.is_empty
    assert engine().price_folded(product, env, overlay) == engine().price(product, env)


def test_a_lifecycle_state_is_refused_as_price_refuses_it():
    from quantark.asset.equity.lifecycle import AutocallableLifecycleState

    engine, product, env = _case("snowball")
    state = AutocallableLifecycleState()
    with pytest.raises(CapabilityError, match="lifecycle_state"):
        engine().price(product, env, lifecycle_state=state)
    with pytest.raises(CapabilityError, match="lifecycle_state"):
        engine().price_folded(product, env, _hand_overlay(product, env, ko=1.0), lifecycle_state=state)


# --- AC4: identity on the engine's own path weights ---------------------------------------------

@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("spot_index", [0, 1, 2])
def test_knock_out_and_knock_in_cash_add_c_times_one_minus_the_routes_alive_weight(family, spot_index):
    engine_f, product, env = _case(family, _spots(family)[spot_index])
    c = 2.5
    engine = engine_f()
    dist = _route(engine, product, env).event_distribution
    npv = engine_f().price(product, env)
    folded = engine_f().price_folded(product, env, _hand_overlay(product, env, ko=c, ki=c))
    expected = c * (1.0 - _p(dist, EventType.MATURITY_NO_KO))
    assert folded - npv == pytest.approx(expected, rel=IDENTITY, abs=IDENTITY * abs(npv))
    alive = engine_f().price_folded(product, env, _hand_overlay(product, env, no_ki=c))
    assert alive - npv == pytest.approx(c * _p(dist, EventType.MATURITY_NO_KO), rel=IDENTITY,
                                        abs=IDENTITY * abs(npv))


@pytest.mark.parametrize("spot_index", [0, 1, 2])
def test_ko_reset_cash_adds_c_times_the_engines_own_path_weights(spot_index):
    engine_f, product, env = _ko_reset_case(SPOTS["ko_reset"][spot_index])
    plain = engine_f()
    npv = plain.price(product, env)
    weights = dict(zip(("ko", "no_ki", "ki"), _weights(plain)))
    for kind, weight in weights.items():
        folded = engine_f().price_folded(product, env, _hand_overlay(product, env, **{kind: 1.5}))
        assert folded - npv == pytest.approx(1.5 * weight, rel=IDENTITY, abs=IDENTITY * npv)


def test_ko_reset_knock_out_weights_are_the_mc_routes_own():
    engine_f, product, env = _ko_reset_case()
    dist = _route(engine_f(), product, env).event_distribution
    npv = engine_f().price(product, env)
    folded = engine_f().price_folded(product, env, _hand_overlay(product, env, ko=2.0))
    assert folded - npv == pytest.approx(2.0 * _p(dist, EventType.KO), rel=IDENTITY)


def test_legs_reading_no_event_never_reach_a_solve():
    from quantark.cashleg import DeterministicLeg, LegDirection

    _, product, env = _case("snowball")
    leg = DeterministicLeg(direction=LegDirection.BUYER_PAYS, amount=5_000.0, payment_time=0.1)
    assert fold_cash_legs([leg], product, env, quantity=1.0, unit_notional=100.0).is_empty


# --- AC5: parity with the engine's own event route ----------------------------------------------

@pytest.mark.parametrize("quantity", [1.0, -3.0])
@pytest.mark.parametrize("family", list(FAMILIES))
def test_folded_trade_matches_the_mc_event_route(family, quantity):
    """Every allowlisted leg type, including legs paying differently on the two maturity branches."""
    engine_f, product, env = _case(family)
    route = _route(engine_f(), product, env)
    dist = route.event_distribution
    legs = _legs(dist, env, splitting=True)
    assert {type(leg).__name__ for leg in legs} >= {
        "DeterministicLeg", "AccrualLeg", "FixedPayoffLeg", "AutocallableCashLeg"}
    overlay = fold_cash_legs(legs, product, env, quantity=quantity, unit_notional=100.0)
    assert not overlay.is_empty
    folded = quantity * engine_f().price_folded(product, env, overlay) + overlay.constant_pv
    expected = quantity * route.npv + sum(leg.value(dist, env, 100.0) for leg in legs)
    notional = max(100.0 * abs(quantity), UNIT)
    assert abs(folded - expected) <= PARITY * notional


@pytest.mark.parametrize("family", ["snowball", "phoenix"])
def test_parity_holds_near_the_barriers(family):
    for spot in _spots(family)[1:]:
        engine_f, product, env = _case(family, spot)
        route = _route(engine_f(), product, env)
        legs = _legs(route.event_distribution, env, splitting=True)
        overlay = fold_cash_legs(legs, product, env, quantity=1.0, unit_notional=100.0)
        folded = engine_f().price_folded(product, env, overlay) + overlay.constant_pv
        expected = route.npv + sum(leg.value(route.event_distribution, env, 100.0) for leg in legs)
        assert abs(folded - expected) <= PARITY * UNIT


# --- AC6: KO-reset against the PDE 1D event route -----------------------------------------------

@pytest.mark.parametrize("method", [MonteCarloMethod.PSEUDO, MonteCarloMethod.QUASI])
@pytest.mark.parametrize("quantity", [1.0, -3.0])
@pytest.mark.parametrize("case", ["ko_reset", "dated_ko_reset"])
def test_folded_ko_reset_matches_the_pde_event_route(case, quantity, method):
    if case == "ko_reset":
        engine_f, product, env = _ko_reset_case(method=method)
    else:
        _, product, env = _dated_ko_reset_case()
        engine_f = lambda: SnowballMCEngine(_params(), method=method)  # noqa: E731
    route = _route(KOResetSnowballPDESolver(PDEParams()), product, env)
    dist = route.event_distribution
    legs = _legs(dist, env, splitting=True)
    assert len(legs) >= 9
    overlay = fold_cash_legs(legs, product, env, quantity=quantity, unit_notional=100.0)
    engine = engine_f()
    folded = quantity * engine.price_folded(product, env, overlay) + overlay.constant_pv
    expected = quantity * route.npv + sum(leg.value(dist, env, 100.0) for leg in legs)
    se = abs(quantity) * engine.get_last_std_error()
    assert se > 0.0
    assert abs(folded - expected) <= KO_RESET_SE * se


# --- AC14: product twins (same paths) -----------------------------------------------------------

def _mc_ko_reset_ko_cash(product, env):
    """The MC engine's own KO-reset record cash, in the structure's merged order (pre first at a tie)."""
    engine = SnowballMCEngine()
    tau = product.get_maturity(env)
    tagged = []
    for tag, profile, maturity in ((0, product.get_pre_ko_observation_profile(env), product.get_pre_maturity_time(env)),
                                   (1, product.get_post_ko_observation_profile(env), product.get_post_maturity_time(env))):
        times = np.asarray(profile["observation_times"], dtype=float)
        cash, _ = engine._compute_ko_schedule_payoffs(product, times, np.asarray(profile["rates"], dtype=float),
                                                       profile["records"], env, float(maturity))
        tagged += [(t, tag, float(c)) for t, c in zip(times, cash) if 0.0 <= t <= tau]
    return [c for _, _, c in sorted(tagged, key=lambda r: (r[0], r[1]))]


def _bumped(name, product):
    if name in ("snowball", "lv_snowball"):
        return make_snowball(ko_dates=KO_DATES, ko_rate=0.17)
    if name == "dated_snowball":
        return dated_snowball(CAL, T0, ko_rate=0.14, rebate_rate=0.12)
    if name == "heston_snowball":
        return _heston_snowball(ko_rate=0.17)
    if name == "dated_phoenix":
        bumped = deepcopy(product)
        bumped.barrier_config = replace(bumped.barrier_config, ko_rate=bumped.barrier_config.ko_rate + 0.05)
        return bumped
    if name == "ko_reset":
        return make_ko_reset(dict(KO_RESET, pre_ko_rate=KO_RESET["pre_ko_rate"] + 0.02,
                                  post_ko_rate=KO_RESET["post_ko_rate"] + 0.02))
    return dated_ko_reset(CAL, T0, post_rate=0.05, rebate_rate=0.15)


def _raised_knocked_in_redemption(product, c):
    """The knocked-in redemption raised by ``c`` (KO-reset's takes a contract tenor too)."""
    twin = deepcopy(product)
    own = twin.get_maturity_payoff_v1
    twin.get_maturity_payoff_v1 = lambda s, pricing_env=None, _own=own, **kw: _own(s, pricing_env=pricing_env, **kw) + c
    return twin


def _rqmc_params():
    return MCParams(num_paths=4096, seed=11, rqmc_min_batches=2, rqmc_max_batches=2, rqmc_target_std=1e-12,
                    rqmc_paths_mode="per_batch")


ROUTES = {
    "snowball": (lambda: SnowballMCEngine(_params()), "snowball", lambda: make_snowball(ko_dates=KO_DATES), make_env),
    "snowball_quasi": (lambda: SnowballMCEngine(_params(), method=MonteCarloMethod.QUASI), "snowball",
                       lambda: make_snowball(ko_dates=KO_DATES), make_env),
    "snowball_rqmc": (lambda: SnowballMCEngine(_rqmc_params(), method=MonteCarloMethod.RANDOMIZED_QUASI),
                      "snowball", lambda: make_snowball(ko_dates=KO_DATES), make_env),
    "snowball_dask": (lambda: SnowballMCEngine(_params(), use_dask=True, num_batches=3), "snowball",
                      lambda: make_snowball(ko_dates=KO_DATES), make_env),
    "dated_snowball": (lambda: SnowballMCEngine(_params()), "dated_snowball", lambda: dated_snowball(CAL, T0),
                       lambda: flat_env(LIVE)),
    "phoenix": (lambda: PhoenixMCEngine(_params()), "phoenix", _phoenix_product, make_env),
    "phoenix_rqmc": (lambda: PhoenixMCEngine(_rqmc_params(), method=MonteCarloMethod.RANDOMIZED_QUASI),
                     "dated_phoenix", lambda: dated_phoenix(CAL, T0), lambda: flat_env(LIVE)),
    "phoenix_dask": (lambda: PhoenixMCEngine(_params(), use_dask=True, num_batches=3), "dated_phoenix",
                     lambda: dated_phoenix(CAL, T0), lambda: flat_env(LIVE)),
    "dated_phoenix": (lambda: PhoenixMCEngine(_params()), "dated_phoenix", lambda: dated_phoenix(CAL, T0),
                      lambda: flat_env(LIVE)),
    "lv_snowball": (lambda: LocalVolSnowballMCEngine(_params()), "lv_snowball",
                    lambda: make_snowball(ko_dates=KO_DATES), _skewed_env),
    "lv_snowball_oss": (lambda: LocalVolSnowballMCEngine(_params(), estimator="one_step_survival"), "lv_snowball",
                        lambda: make_snowball(ko_dates=KO_DATES), _skewed_env),
    "heston_affine_rqmc": (lambda: QESnowballMCEngine(_heston(), params=_rqmc_params(),
                                                      method=MonteCarloMethod.RANDOMIZED_QUASI,
                                                      martingale_correction=True, substeps_per_interval=2,
                                                      rqmc_affine_spot_factor=True),
                           "heston_snowball", _heston_snowball, _heston_env),
    "slv_conditional_control": (lambda: HestonSLVQESnowballMCEngine(
        _heston(), params=_rqmc_params(), method=MonteCarloMethod.RANDOMIZED_QUASI,
        leverage_surface=_unit_leverage(), martingale_correction=True, substeps_per_interval=2,
        rqmc_heston_conditional_control=True, rqmc_spot_strata=2), "heston_snowball", _heston_snowball, _heston_env),
    "ko_reset": (lambda: SnowballMCEngine(_params()), "ko_reset", lambda: make_ko_reset(KO_RESET), ko_reset_env),
    "ko_reset_rqmc": (lambda: SnowballMCEngine(_rqmc_params(), method=MonteCarloMethod.RANDOMIZED_QUASI),
                      "ko_reset", lambda: make_ko_reset(KO_RESET), ko_reset_env),
    "ko_reset_dask": (lambda: SnowballMCEngine(_params(), use_dask=True, num_batches=3), "ko_reset",
                      lambda: make_ko_reset(KO_RESET), ko_reset_env),
    "dated_ko_reset": (lambda: SnowballMCEngine(_params()), "dated_ko_reset", lambda: dated_ko_reset(CAL, T0),
                       lambda: flat_env(LIVE)),
}


@pytest.mark.parametrize("route", [r for r in ROUTES if r != "phoenix"])   # its KO cash is principal only
def test_ko_cash_twin_equals_the_product_with_a_higher_ko_rate(route):
    engine_f, name, product_f, env_f = ROUTES[route]
    product, env = product_f(), env_f()
    bumped = _bumped(name, product)
    amounts = _mc_ko_reset_ko_cash if "ko_reset" in name else _autocallable_ko_cash
    overlay = _ko_cash_diff_overlay(product, bumped, env, amounts)
    folded = engine_f().price_folded(product, env, overlay)
    assert folded == pytest.approx(engine_f().price(bumped, env), rel=IDENTITY)


@pytest.mark.parametrize("route", list(ROUTES))
def test_knock_in_cash_twin_equals_the_product_with_a_raised_knocked_in_redemption(route):
    engine_f, _, product_f, env_f = ROUTES[route]
    product, env = product_f(), env_f()
    c = 4.0
    folded = engine_f().price_folded(product, env, _branch_overlay(product, env, ki=c))
    twin = engine_f().price(_raised_knocked_in_redemption(product, c), env)
    assert folded == pytest.approx(twin, rel=IDENTITY)
    assert folded != engine_f().price(product, env)


@pytest.mark.parametrize("route", [r for r in ROUTES if "phoenix" not in r])
def test_alive_cash_twin_equals_the_product_with_a_raised_alive_redemption(route):
    engine_f, _, product_f, env_f = ROUTES[route]
    product, env = product_f(), env_f()
    c = 3.0
    twin = deepcopy(product)
    own = twin.get_maturity_payoff_v0
    twin.get_maturity_payoff_v0 = lambda s, pricing_env=None, _own=own: _own(s, pricing_env=pricing_env) + c
    folded = engine_f().price_folded(product, env, _branch_overlay(product, env, no_ki=c))
    assert folded == pytest.approx(engine_f().price(twin, env), rel=IDENTITY)


@pytest.mark.parametrize("route", ["snowball", "ko_reset"])
def test_maturity_cash_twin_equals_the_product_with_a_higher_rebate(route):
    engine_f, _, product_f, env_f = ROUTES[route]
    product, env = product_f(), env_f()
    bumped = deepcopy(product)
    bumped.payoff_config = replace(bumped.payoff_config, rebate_rate=bumped.payoff_config.rebate_rate + 0.03)
    diffs = {bumped.get_maturity_payoff_v0(s, pricing_env=env) - product.get_maturity_payoff_v0(s, pricing_env=env)
             for s in (80.0, 100.0, 102.0)}
    assert len(diffs) == 1 and next(iter(diffs)) > 0
    folded = engine_f().price_folded(product, env, _branch_overlay(product, env, no_ki=next(iter(diffs))))
    assert folded == pytest.approx(engine_f().price(bumped, env), rel=IDENTITY)


def test_one_step_survival_identity_on_its_own_branch_weights():
    engine_f, _, product_f, env_f = ROUTES["lv_snowball_oss"]
    product, env = product_f(), env_f()
    plain = engine_f()
    npv = plain.price(product, env)
    p_ko, p_v0, p_v1 = _weights(plain)
    folded = engine_f().price_folded(product, env, _hand_overlay(product, env, ko=2.0, ki=2.0))
    assert folded - npv == pytest.approx(2.0 * (p_ko + p_v1), rel=IDENTITY)
    assert p_ko + p_v0 + p_v1 == pytest.approx(1.0, abs=1e-12)


def test_conditional_moments_fold_the_cash_where_the_record_cash_sits():
    """The affine estimator's own sites: knock-out interval mass x (record cash + folded cash)."""
    engine = QESnowballMCEngine(_heston(), params=_rqmc_params(), method=MonteCarloMethod.RANDOMIZED_QUASI,
                                martingale_correction=True, substeps_per_interval=2, rqmc_affine_spot_factor=True)
    product, env = _heston_snowball(), _heston_env()
    spec = engine.build_rqmc_session_spec(product, env)
    paths, aux = spec.path_generator.generate_paths(batch_id=0, return_aux=True)
    grid = engine._build_time_grid(product, env, product.get_maturity(env))
    kwargs = dict(product=product, pricing_env=env, base_paths=paths,
                  log_spot_factor_loadings=aux["log_spot_factor_loadings"], ko_indices=grid[2],
                  ki_indices=grid[3], maturity=product.get_maturity(env), discount_factors=engine._df)
    plain = conditional_standard_snowball_moments(**kwargs)
    n = len(grid[2])
    ko_cash = np.linspace(1.0, 2.0, n)
    folded = conditional_standard_snowball_moments(**kwargs, ko_cash=ko_cash, no_ki_cash=0.5, ki_cash=0.25)
    ko_settle = np.asarray(product.get_ko_observation_profile(env)["settlement_times"], dtype=float)
    # per-observation knock-out mass from unit cash on one record at a time
    masses = []
    for j in range(n):
        unit = np.zeros(n)
        unit[j] = 1.0
        probe = conditional_standard_snowball_moments(**kwargs, ko_cash=unit)
        masses.append((probe.discounted_payoff - plain.discounted_payoff) / engine._df(ko_settle[j]))
    masses = np.array(masses).T
    df_t = float(engine._df(np.array([product.get_maturity(env)]))[0])
    expected = (plain.discounted_payoff + masses @ (ko_cash * engine._df(ko_settle))
                + df_t * (0.5 * plain.v0_probability + 0.25 * plain.v1_probability))
    np.testing.assert_allclose(folded.discounted_payoff, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(masses.sum(axis=1), plain.ko_probability, rtol=1e-12, atol=1e-13)
    unfolded = conditional_standard_snowball_moments(**kwargs, ko_cash=None, no_ki_cash=0.0, ki_cash=0.0)
    np.testing.assert_array_equal(unfolded.discounted_payoff, plain.discounted_payoff)


# --- AC8: lifecycle -----------------------------------------------------------------------------

def _lifecycle_cases():
    sb, ph = dated_snowball(CAL, T0), dated_phoenix(CAL, T0)
    return {
        "snowball-knocked-in": (SnowballMCEngine, _knocked_in(sb), flat_env(LIVE, spot=90.0)),
        "phoenix-knocked-in": (PhoenixMCEngine, _knocked_in(ph), flat_env(LIVE, spot=90.0)),
        "snowball-ko-decided-at-valuation": (SnowballMCEngine, sb, flat_env(_ko_dates(sb)[2], spot=110.0)),
        "phoenix-ko-decided-at-valuation": (PhoenixMCEngine, ph, flat_env(_ko_dates(ph)[2], spot=110.0)),
    }


@pytest.mark.parametrize("case", list(_lifecycle_cases()))
def test_lifecycle_states_fold_like_the_event_route(case):
    engine_cls, product, env = _lifecycle_cases()[case]
    engine_f = lambda: engine_cls(_params())  # noqa: E731
    route = _route(engine_f(), product, env)
    dist = route.event_distribution
    legs = _legs(dist, env, splitting=True)
    overlay = fold_cash_legs(legs, product, env, quantity=1.0, unit_notional=100.0)
    assert not overlay.is_empty
    folded = engine_f().price_folded(product, env, overlay) + overlay.constant_pv
    expected = route.npv + sum(leg.value(dist, env, 100.0) for leg in legs)
    assert abs(folded - expected) <= PARITY * UNIT
    assert route.npv == engine_f().price(product, env)
    if "decided" in case:
        assert _p(dist, EventType.KO) == pytest.approx(1.0, abs=1e-12)
        hit = next(i for i, s in enumerate(_event_structure(product, env).ko) if s.determination_time == 0.0)
        entry = overlay.ko[hit]
        assert folded - overlay.constant_pv - engine_f().price(product, env) == pytest.approx(
            entry.amount * env.get_discount_factor(entry.payment_time), rel=1e-12)
    else:
        assert _p(dist, EventType.MATURITY_NO_KO) == pytest.approx(0.0, abs=1e-12)


def _ko_reset_lifecycle_cases():
    kr = dated_ko_reset(CAL, T0)
    return {
        "ko-reset-knocked-in": (_knocked_in(kr), flat_env(LIVE, spot=90.0)),
        "ko-reset-ko-decided-at-valuation": (kr, flat_env(_ko_dates(kr)[2], spot=110.0)),
    }


@pytest.mark.parametrize("case", list(_ko_reset_lifecycle_cases()))
def test_ko_reset_lifecycle_states_fold_like_the_pde_event_route(case):
    product, env = _ko_reset_lifecycle_cases()[case]
    route = _route(KOResetSnowballPDESolver(PDEParams()), product, env)
    dist = route.event_distribution
    legs = _legs(dist, env, splitting=True)
    overlay = fold_cash_legs(legs, product, env, quantity=1.0, unit_notional=100.0)
    engine = SnowballMCEngine(_params())
    folded = engine.price_folded(product, env, overlay) + overlay.constant_pv
    expected = route.npv + sum(leg.value(dist, env, 100.0) for leg in legs)
    if "decided" in case:
        # every path knocks out at the valuation record: the fold is exact against the route
        assert abs(folded - expected) <= PARITY * UNIT
    else:
        assert abs(folded - expected) <= KO_RESET_SE * engine.get_last_std_error()


@pytest.mark.parametrize("engine_cls, build", [(SnowballMCEngine, dated_snowball), (PhoenixMCEngine, dated_phoenix),
                                               (SnowballMCEngine, dated_ko_reset)])
def test_an_expired_product_refuses_a_non_empty_overlay(engine_cls, build):
    """Expired, the structure is trivial and leg overlays are empty; a hand-built one has no event to land on."""
    product = build(CAL, T0)
    if build is dated_ko_reset:   # alive after the first schedule only when knocked in
        product = _knocked_in(product)
    last = (product.post_barrier_config if build is dated_ko_reset else product.barrier_config)
    env = flat_env(last.ko_observation_schedule.records[-1].observation_date, spot=95.0)
    assert product.get_maturity(env) <= 0.0
    empty = _hand_overlay(product, env)
    assert empty.is_empty and engine_cls(_params()).price_folded(product, env, empty) == \
        engine_cls(_params()).price(product, env)
    with pytest.raises(CapabilityError, match="expired"):
        engine_cls(_params()).price_folded(product, env, _hand_overlay(product, env, no_ki=1.0))


# --- AC12 / fail closed -------------------------------------------------------------------------

def _rebased(product):
    product = deepcopy(product)
    product.post_ko_mode = PostKOScheduleMode.REBASED
    return product


def test_a_rebased_ko_reset_refuses_folding():
    _, product, env = _dated_ko_reset_case()
    overlay = _hand_overlay(product, env, ko=1.0)
    rebased = _rebased(product)
    with pytest.raises(CapabilityError, match="REBASED"):
        SnowballMCEngine(_params()).price_folded(rebased, env, overlay)
    with pytest.raises(CapabilityError, match="REBASED"):
        fold_cash_legs(_legs(_route(KOResetSnowballPDESolver(PDEParams()), product, env).event_distribution, env,
                             splitting=True), rebased, env, quantity=1.0, unit_notional=100.0)


@pytest.mark.parametrize("family", ["snowball", "phoenix", "lv_snowball", "heston_snowball"])
def test_an_overlay_built_for_another_schedule_is_refused(family):
    engine_f, product, env = _case(family)
    other = make_snowball(ko_dates=(0.5, 1.0))
    with pytest.raises(ValidationError):
        engine_f().price_folded(product, env, _hand_overlay(other, env, ko=1.0))


def test_one_step_survival_refuses_an_overlay_for_another_schedule():
    engine_f, _, product_f, env_f = ROUTES["lv_snowball_oss"]
    with pytest.raises(ValidationError):
        engine_f().price_folded(product_f(), env_f(), _hand_overlay(make_snowball(ko_dates=(0.5, 1.0)), env_f(),
                                                                     ko=1.0))


def test_a_ko_reset_overlay_with_its_schedules_swapped_is_refused():
    engine_f, product, env = _ko_reset_case()
    overlay = _hand_overlay(product, env, ko=1.0)
    swapped = replace(overlay, ko=tuple(replace(e, schedule="post" if e.schedule == "pre" else "pre")
                                        for e in overlay.ko))
    with pytest.raises(ValidationError):
        engine_f().price_folded(product, env, swapped)


def test_a_ko_reset_alive_entry_at_the_knocked_in_maturity_is_refused():
    engine_f, product, env = _ko_reset_case()
    overlay = _hand_overlay(product, env, no_ki=1.0)
    moved = replace(overlay, maturity_no_ki=replace(overlay.maturity_no_ki,
                                                    determination_time=overlay.maturity_ki.determination_time,
                                                    payment_time=overlay.maturity_ki.payment_time))
    with pytest.raises(ValidationError, match="no_ki"):
        engine_f().price_folded(product, env, moved)


def test_a_knock_out_paid_elsewhere_than_the_engine_settles_it_is_refused():
    engine_f, product, env = _case("snowball")
    overlay = _hand_overlay(product, env, ko=1.0)
    late = replace(overlay, ko=tuple(replace(e, payment_time=e.payment_time + 0.01) for e in overlay.ko))
    with pytest.raises(ValidationError, match="settles"):
        engine_f().price_folded(product, env, late)


# --- AC9: no folded state survives a folded solve -----------------------------------------------

@pytest.mark.parametrize("route", ["snowball", "snowball_quasi", "snowball_rqmc", "phoenix", "lv_snowball_oss",
                                   "heston_affine_rqmc", "ko_reset"])
def test_a_shared_engine_never_mixes_overlays(route):
    engine_f, _, product_f, env_f = ROUTES[route]
    product, env = product_f(), env_f()
    a = _hand_overlay(product, env, ko=1.0, ki=2.0)
    b = _hand_overlay(product, env, ko=-0.5, no_ki=0.7)
    shared = engine_f()
    seq = [shared.price_folded(product, env, a), shared.price(product, env),
           shared.price_folded(product, env, b), shared.price_folded(product, env, a)]
    fresh = [engine_f().price_folded(product, env, a), engine_f().price(product, env),
             engine_f().price_folded(product, env, b), engine_f().price_folded(product, env, a)]
    assert seq == fresh
    assert len({seq[0], seq[1], seq[2]}) == 3
    assert shared._fold is None


def test_the_event_route_after_a_folded_solve_is_a_fresh_engines():
    engine_f, product, env = _case("phoenix")
    shared = engine_f()
    shared.price_folded(product, env, _hand_overlay(product, env, ko=3.0, ki=1.0))
    got = shared.price_with_events(product, env)
    want = engine_f().price_with_events(product, env)
    assert got.npv == want.npv
    np.testing.assert_array_equal(got.event_distribution.probabilities[EventType.KO],
                                  want.event_distribution.probabilities[EventType.KO])


def test_a_failed_folded_solve_leaves_no_overlay_behind():
    engine_f, product, env = _case("snowball")
    shared = engine_f()
    with pytest.raises(ValidationError):
        shared.price_folded(product, env, _hand_overlay(make_snowball(ko_dates=(0.5, 1.0)), env, ko=1.0))
    assert shared._fold is None
    assert shared.price(product, env) == engine_f().price(product, env)


def _plan_id(engine, product, env):
    from quantark.execution.contracts import PricingRequest

    adapter = AutocallableAdaptiveMCAdapter()
    request = PricingRequest(operation=PricingOperation.PRICE, product=product, pricing_env=env,
                             outputs=frozenset({OutputKind.PV}))
    prepared = adapter.prepare(engine, request, None)
    try:
        return adapter.plan_adaptive(engine, request, prepared, None).plan_id
    finally:
        for handle in prepared.handles:
            handle.close()


@pytest.mark.parametrize("route", ["snowball_rqmc", "phoenix_rqmc", "ko_reset_rqmc"])
def test_the_adaptive_plan_id_carries_the_overlay(route):
    engine_f, _, product_f, env_f = ROUTES[route]
    product, env = product_f(), env_f()
    a = _hand_overlay(product, env, ko=1.0)
    b = _hand_overlay(product, env, ko=2.0)
    engine = engine_f()
    plain = _plan_id(engine, product, env)
    with engine._folding(a):
        folded_a = _plan_id(engine, product, env)
    with engine._folding(b):
        folded_b = _plan_id(engine, product, env)
    assert len({plain, folded_a, folded_b}) == 3
    assert _plan_id(engine, product, env) == plain == _plan_id(engine_f(), product, env)
    with engine._folding(a):
        assert _plan_id(engine, product, env) == folded_a
    assert ":fold=" not in plain


# --- D11c: DRILLDOWN refusal hook ---------------------------------------------------------------

def test_the_base_engine_never_refuses_drilldown():
    product, env = make_snowball(ko_dates=KO_DATES), make_env()
    assert BaseEngine.fold_drilldown_refusal(SnowballPDESolver(PDEParams()), product, env) is None
    assert SnowballPDESolver(PDEParams()).fold_drilldown_refusal(product, env) is None


@pytest.mark.parametrize("route", ["snowball", "snowball_quasi", "phoenix", "lv_snowball", "lv_snowball_oss",
                                   "ko_reset", "dated_phoenix"])
def test_serial_single_batch_routes_allow_drilldown(route):
    engine_f, _, product_f, env_f = ROUTES[route]
    assert engine_f().fold_drilldown_refusal(product_f(), env_f()) is None


@pytest.mark.parametrize("route, reason", [
    ("snowball_rqmc", "adaptive"), ("phoenix_rqmc", "adaptive"), ("ko_reset_rqmc", "adaptive"),
    ("heston_affine_rqmc", "adaptive"), ("slv_conditional_control", "adaptive"),
    ("snowball_dask", "Dask"), ("phoenix_dask", "Dask"), ("ko_reset_dask", "Dask"),
])
def test_payoff_dependent_path_sets_refuse_drilldown(route, reason):
    engine_f, _, product_f, env_f = ROUTES[route]
    refusal = engine_f().fold_drilldown_refusal(product_f(), env_f())
    assert isinstance(refusal, str) and reason in refusal


def test_unused_overlay_builder_is_well_formed():
    """Guard: the hand-built overlays above carry the structure's own slots."""
    _, product, env = _case("snowball")
    overlay = _hand_overlay(product, env, ko=1.0, ki=1.0)
    assert isinstance(overlay, FoldOverlay)
    assert all(isinstance(e, FoldedEventCash) for e in overlay.ko)
