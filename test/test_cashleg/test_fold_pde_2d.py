"""Cash-leg folding on the 2D Heston / Heston-SLV PDE solvers (ship spec 2026-10-21, task T6).

The 2D march folds the overlay where the product's own cash goes: each knock-out record's payoff at the
ADI step hooks (projected cell average, nodal mask and the valuation-date trigger alike; Phoenix's joint
coupon + knock-out projection included), each maturity branch's terminal surface, the 1D Dirichlet rows
the boundary hook reuses, and the valuation-instant knock-out short-circuit. The 2D event statistics
are a flat-vol 1D surrogate, so they are no reference: the fold is pinned against the 2D engine itself
(AC14 product twins, the AC4 identity on the 2D march's own weights) and against the same-model Monte
Carlo event route (AC6).
"""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.mc import (
    HestonSLVQESnowballMCEngine,
    HestonSnowballMCEngine,
)
from quantark.asset.equity.engine.mc.phoenix_vol_mc_engines import (
    HestonPhoenixMCEngine,
    HestonSLVQEPhoenixMCEngine,
)
from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import (
    HestonPhoenixPDESolver,
    HestonSLVPhoenixPDESolver,
)
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSLVSnowballPDESolver,
    HestonSnowballPDESolver,
)
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.cashleg import EventType
from quantark.cashleg.folding import _event_structure, fold_cash_legs
from quantark.execution.errors import CapabilityError
from quantark.param import ContinuousDividendYield, FlatRateCurve, GridVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum import CouponPayType, ObservationType
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface

from intraday.conftest import dated_phoenix, dated_snowball, flat_env
from test_cashleg._autocallable_helpers import make_env, make_phoenix, make_snowball
from test_cashleg.test_fold_quad_v1 import (
    _autocallable_ko_cash,
    _branch_overlay,
    _hand_overlay,
    _ko_cash_diff_overlay,
    _ko_dates,
    _legs,
)

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
T0, LIVE = datetime(2026, 3, 16), datetime(2026, 4, 1)
KO_DATES = (0.25, 0.5, 0.75, 1.0)
#: the spec's bar for a fold checked against the same engine's own march (AC4 / AC14 relative)
IDENTITY = 1e-10
HESTON = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5)
#: coarse 2D grids: every test runs in seconds; the fold is exact on any grid
GRID = dict(n_x=100, n_v=18, n_t=24)


def _leverage(value=0.9, s0=100.0):
    """A constant, non-unit leverage: the SLV march differs from the Heston one."""
    strikes = np.array(list(s0 * np.exp(np.linspace(-0.8, 0.8, 11))))
    return LeverageSurface(time_grid=np.linspace(0.0, 2.0, 5), strike_grid=strikes,
                           leverage_grid=np.full((5, strikes.size), float(value)))


def _heston_snowball():
    return HestonSnowballPDESolver(HESTON, **GRID)


def _slv_snowball():
    return HestonSLVSnowballPDESolver(HESTON, _leverage(), **GRID)


def _heston_phoenix():
    return HestonPhoenixPDESolver(HESTON, **GRID)


def _slv_phoenix():
    return HestonSLVPhoenixPDESolver(HESTON, _leverage(), **GRID)


def _discrete_ki_snowball(ko_rate=0.15):
    barrier = BarrierConfig(
        ko_barrier=103.0, ko_rate=ko_rate, ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=list(KO_DATES), ki_barrier=75.0,
        ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=list(KO_DATES),
    )
    return SnowballOption(initial_price=100.0, strike=100.0, barrier_config=barrier, contract_multiplier=1.0,
                          maturity=1.0, is_reverse=False)


def _no_ki_snowball(ko_rate=0.15):
    """No knock-in barrier: the alive surface's knock-in edge is a no-event row (principal + rebate)."""
    barrier = BarrierConfig(ko_barrier=103.0, ko_rate=ko_rate, ko_observation_type=ObservationType.DISCRETE,
                            ko_observation_dates=list(KO_DATES))
    return SnowballOption(initial_price=100.0, strike=100.0, barrier_config=barrier, contract_multiplier=1.0,
                          maturity=1.0, is_reverse=False)


def _reverse_snowball(ko_rate=0.15):
    barrier = BarrierConfig(
        ko_barrier=97.0, ko_rate=ko_rate, ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=list(KO_DATES), ki_barrier=125.0,
        ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True,
    )
    return SnowballOption(initial_price=100.0, strike=100.0, barrier_config=barrier, contract_multiplier=1.0,
                          maturity=1.0, is_reverse=True)


def _phoenix_product(ko_rate=0.0, coupon_pay=CouponPayType.INSTANT, disable_ko_after_ki=False):
    """``make_phoenix``'s Phoenix (non-memory, continuous KI 70) with an optional KO coupon."""
    base = make_phoenix(ko_dates=KO_DATES, coupon_barrier=(80.0,) * 4, ki_barrier=70.0, coupon_pay=coupon_pay,
                        disable_ko_after_ki=disable_ko_after_ki)
    barrier = BarrierConfig(
        ko_barrier=105.0, ko_rate=ko_rate, ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=list(KO_DATES), ki_barrier=70.0, ki_observation_type=ObservationType.CONTINUOUS,
        ki_continuous=True, disable_ko_after_ki=disable_ko_after_ki,
    )
    return PhoenixOption(initial_price=100.0, strike=100.0, barrier_config=barrier,
                         coupon_config=base.coupon_config, payoff_config=base.payoff_config,
                         contract_multiplier=1.0, maturity=1.0)


def _dated_env(spot=100.0):
    return flat_env(LIVE, spot=spot)


FAMILIES = {
    "heston_snowball": (_heston_snowball, lambda: make_snowball(ko_dates=KO_DATES), make_env),
    "slv_snowball": (_slv_snowball, lambda: make_snowball(ko_dates=KO_DATES), make_env),
    "discrete_ki_snowball": (_heston_snowball, _discrete_ki_snowball, make_env),
    "reverse_snowball": (_heston_snowball, _reverse_snowball, make_env),
    "no_ki_snowball": (_heston_snowball, _no_ki_snowball, make_env),
    "no_ko_after_ki_snowball": (_heston_snowball,
                                lambda: _with_barrier(make_snowball(ko_dates=KO_DATES), disable_ko_after_ki=True),
                                make_env),
    "dated_snowball": (_heston_snowball, lambda: dated_snowball(CAL, T0), _dated_env),
    "heston_phoenix": (_heston_phoenix, _phoenix_product, make_env),
    "slv_phoenix": (_slv_phoenix, _phoenix_product, make_env),
    "phoenix_expiry": (_heston_phoenix, lambda: _phoenix_product(coupon_pay=CouponPayType.EXPIRY), make_env),
    "dated_phoenix": (_heston_phoenix, lambda: dated_phoenix(CAL, T0, memory=False), _dated_env),
}
# ATM, just under the knock-out level, just above the knock-in level
SPOTS = {"heston_snowball": (100.0, 102.5, 76.0), "slv_snowball": (100.0, 102.5, 76.0),
         "discrete_ki_snowball": (100.0, 102.5, 76.0), "reverse_snowball": (100.0, 97.5, 124.0),
         "no_ki_snowball": (100.0, 102.5, 80.0),
         "no_ko_after_ki_snowball": (100.0, 102.5, 76.0), "dated_snowball": (100.0, 102.5, 76.0),
         "heston_phoenix": (100.0, 104.5, 71.0), "slv_phoenix": (100.0, 104.5, 71.0),
         "phoenix_expiry": (100.0, 104.5, 71.0), "dated_phoenix": (100.0, 102.5, 76.0)}
SNOWBALLS = [f for f in FAMILIES if "snowball" in f]
PHOENIXES = [f for f in FAMILIES if "phoenix" in f]


def _with_barrier(product, **changes):
    product = deepcopy(product)
    product.barrier_config = replace(product.barrier_config, **changes)
    return product


def _case(family, spot=100.0):
    factory, build, make = FAMILIES[family]
    return factory, build(), make(spot=spot)


# --- contract ----------------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [HestonSnowballPDESolver, HestonSLVSnowballPDESolver,
                                 HestonPhoenixPDESolver, HestonSLVPhoenixPDESolver])
def test_the_2d_solvers_declare_folding(cls):
    assert cls.supports_cash_leg_folding is True


@pytest.mark.parametrize("family", list(FAMILIES))
def test_an_empty_overlay_is_the_plain_price_bitwise(family):
    factory, product, env = _case(family)
    overlay = _hand_overlay(product, env)
    assert overlay.is_empty
    assert factory().price_folded(product, env, overlay) == factory().price(product, env)


# --- AC14: product twins on the 2D march itself ------------------------------------------------

def _bumped_ko_rate(family, product):
    """The same family with a higher KO coupon, rebuilt by its own builder (the records carry the cash)."""
    if family == "dated_snowball":
        return dated_snowball(CAL, T0, ko_rate=0.14, rebate_rate=0.12)   # the rebate stays at the base 0.12
    if family == "dated_phoenix":
        return _with_barrier(product, ko_rate=0.05)
    if family == "phoenix_expiry":
        return _phoenix_product(ko_rate=0.05, coupon_pay=CouponPayType.EXPIRY)
    if family in PHOENIXES:
        return _phoenix_product(ko_rate=0.05)
    if family == "discrete_ki_snowball":
        return _discrete_ki_snowball(ko_rate=0.17)
    if family == "reverse_snowball":
        return _reverse_snowball(ko_rate=0.17)
    if family == "no_ki_snowball":
        return _no_ki_snowball(ko_rate=0.17)
    bumped = make_snowball(ko_dates=KO_DATES, ko_rate=0.17)
    if family == "no_ko_after_ki_snowball":
        bumped = _with_barrier(bumped, disable_ko_after_ki=True)
    return bumped


@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("spot_index", [0, 1, 2])
def test_ko_cash_twin_equals_the_product_with_a_higher_ko_rate(family, spot_index):
    factory, product, env = _case(family, SPOTS[family][spot_index])
    bumped = _bumped_ko_rate(family, product)
    overlay = _ko_cash_diff_overlay(product, bumped, env, _autocallable_ko_cash)
    folded = factory().price_folded(product, env, overlay)
    assert folded == pytest.approx(factory().price(bumped, env), rel=IDENTITY)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_maturity_cash_twin_equals_the_product_with_a_higher_rebate(family):
    """The no-knock-in branch's cash lands where the product's own redemption does (terminal surface and the
    alive no-event Dirichlet rows): the twin is the product with its rebate raised."""
    factory, product, env = _case(family)
    bumped = deepcopy(product)
    bumped.payoff_config = replace(bumped.payoff_config, rebate_rate=bumped.payoff_config.rebate_rate + 0.03)
    diffs = {round(_v0_payoff(bumped, s, env) - _v0_payoff(product, s, env), 12) for s in (80.0, 100.0, 102.0)}
    assert len(diffs) == 1 and next(iter(diffs)) > 0
    assert bumped.get_maturity_payoff_v1(90.0, env) == product.get_maturity_payoff_v1(90.0, env)
    structure = _event_structure(product, env)
    # the twin's redemption is discounted by the product's own terminal delay: cash at the slot's payment
    delay = env.get_discount_factor(structure.no_ki.payment_time) / env.get_discount_factor(
        structure.no_ki.determination_time)
    cash = next(iter(diffs)) * factory()._terminal_delay_df(product, env) / delay
    folded = factory().price_folded(product, env, _branch_overlay(product, env, no_ki=cash))
    assert folded == pytest.approx(factory().price(bumped, env), rel=IDENTITY)


def _v0_payoff(product, spot, env):
    if isinstance(product, PhoenixOption):
        return product.get_maturity_payoff_v0(spot, accumulated_coupons=0.0, pricing_env=env)
    return product.get_maturity_payoff_v0(spot, pricing_env=env)


# --- AC4: the identity on the 2D march's own weights --------------------------------------------

def _rise(factory, product, env, **cash):
    return factory().price_folded(product, env, _hand_overlay(product, env, **cash)) - factory().price(product, env)


@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("spot_index", [0, 1, 2])
def test_event_cash_adds_c_times_the_2d_marchs_own_event_weights(family, spot_index):
    """``c/DF(pay)`` on every non-anchor event raises the PV by ``c·(W − P_no_ki)``: ``W`` the march's weight of a
    sure unit of cash (``c/DF`` on every event), ``P_no_ki`` its no-knock-in weight. Without native 2D statistics
    (the 2D route's are a flat-vol 1D surrogate) the weights are the 2D march's own; ``P_no_ki`` is pinned to the
    product's own redemption by the maturity twin above, the knock-out weights by the knock-out twins, and every
    weight against the same-model Monte Carlo event route (AC6)."""
    factory, product, env = _case(family, SPOTS[family][spot_index])
    c = 2.5
    npv = factory().price(product, env)
    rise = _rise(factory, product, env, ko=c, ki=c)
    w_ko, w_ki, w_no_ki = (_rise(factory, product, env, **{kind: c}) / c for kind in ("ko", "ki", "no_ki"))
    w = _rise(factory, product, env, ko=c, ki=c, no_ki=c) / c
    assert rise == pytest.approx(c * (w - w_no_ki), rel=IDENTITY, abs=IDENTITY * abs(npv))
    assert rise == pytest.approx(c * (w_ko + w_ki), rel=IDENTITY, abs=IDENTITY * abs(npv))
    assert all(0.0 <= x <= 1.0 for x in (w_ko, w_ki, w_no_ki))
    # a sure unit of cash: the march's discount of a constant (coarse time grid; dated records off the step grid)
    assert w == pytest.approx(1.0, abs=1e-4)


# --- the Dirichlet rows the boundary hook reuses -----------------------------------------------

def _recorded_rows(solver, product, env, overlay=None):
    """Every (surface, lower, upper) row value the 2D boundary hook writes during one solve, in order."""
    rows = []
    for surface in ("v0", "v1"):
        real = getattr(solver, f"_boundary_rows_{surface}")

        def spy(t_idx, tau, *args, _real=real, _surface=surface):
            lower, upper = _real(t_idx, tau, *args)
            rows.append((_surface, float(tau), lower.value, upper.value))
            return lower, upper

        setattr(solver, f"_boundary_rows_{surface}", spy)
    if overlay is None:
        solver.price(product, env)
    else:
        solver.price_folded(product, env, overlay)
    return rows


@pytest.mark.parametrize("family", ["heston_snowball", "slv_snowball", "reverse_snowball", "heston_phoenix",
                                    "no_ko_after_ki_snowball", "discrete_ki_snowball", "no_ki_snowball"])
def test_the_dirichlet_rows_carry_the_folded_cash_as_the_product_rows_carry_its_own(family):
    """``x/DF(pay)`` per event reaches each row discounted to the row's time: ``x/DF(t)``. The knock-in branch's
    cash sits on the knocked-in rows of the knock-in edge (and of the knock-out edge only when the knock-out cannot
    fire knocked in), the knock-out cash on the alive knock-out edge, the alive branch's cash on an alive edge that
    reaches neither barrier (the product's no-event redemption row), as in 1D."""
    factory, product, env = _case(family)
    x_ko, x_ki, x_no_ki = 1.5, 2.5, 0.5
    plain = _recorded_rows(factory(), product, env)
    folded = _recorded_rows(factory(), product, env, _hand_overlay(product, env, ko=x_ko, ki=x_ki, no_ki=x_no_ki))
    assert [r[:2] for r in plain] == [r[:2] for r in folded] and plain
    T = product.get_maturity(env)
    ko_edge = 0 if product.is_reverse else 1           # (lower, upper) index of the knock-out edge
    ko_fires_knocked_in = not product.barrier_config.disable_ko_after_ki
    seen = set()
    for (surface, tau, *base), (_, _, *moved) in zip(plain, folded):
        df_t = env.get_discount_factor(T - tau)
        rise = [(m - b) * df_t for b, m in zip(base, moved)]
        if surface == "v1":
            expected = [x_ki, x_ki]
            expected[ko_edge] = 0.0 if ko_fires_knocked_in else x_ki
        else:
            # the alive knock-in edge copies (continuous KI) or takes (discrete KI) the knocked-in row; without a
            # knock-in barrier it is the alive no-event row
            alive = x_ki if product.has_ki_barrier else x_no_ki
            expected = [alive, alive]
            expected[ko_edge] = x_ko
        for got, want in zip(rise, expected):
            assert got == pytest.approx(want, rel=1e-9, abs=1e-12)
        seen.add(surface)
    assert seen == {"v0", "v1"}


# --- the valuation-instant short-circuits and the knocked-in surface -------------------------------

def _decided_cases():
    sb, ph = dated_snowball(CAL, T0), dated_phoenix(CAL, T0, memory=False)
    return {
        "heston-snowball": (_heston_snowball, sb, flat_env(_ko_dates(sb)[2], spot=110.0)),
        "slv-snowball": (_slv_snowball, sb, flat_env(_ko_dates(sb)[2], spot=110.0)),
        "heston-phoenix": (_heston_phoenix, ph, flat_env(_ko_dates(ph)[2], spot=110.0)),
    }


@pytest.mark.parametrize("case", list(_decided_cases()))
def test_a_knock_out_decided_at_valuation_pays_the_deciding_records_cash(case):
    factory, product, env = _decided_cases()[case]
    structure = _event_structure(product, env)
    hit = next(i for i, s in enumerate(structure.ko) if s.determination_time == 0.0)
    overlay = _hand_overlay(product, env, ko=1.0, ki=2.0, no_ki=3.0)
    folded = factory().price_folded(product, env, overlay)
    entry = overlay.ko[hit]
    assert folded - factory().price(product, env) == pytest.approx(
        entry.amount * env.get_discount_factor(entry.payment_time), rel=1e-12)


def _knocked_in(product):
    product = deepcopy(product)
    product._otc_lifecycle_knocked_in = True
    return product


@pytest.mark.parametrize("family", ["heston_snowball", "slv_snowball", "heston_phoenix", "dated_snowball"])
def test_a_knocked_in_trade_folds_on_the_knocked_in_surface_alone(family):
    factory, product, env = _case(family, 90.0)
    product = _knocked_in(product)
    npv = factory().price(product, env)
    assert npv != _case(family, 90.0)[0]().price(_case(family, 90.0)[1], env)
    # the alive branch is gone: its cash reaches nothing
    assert _rise(factory, product, env, no_ki=3.0) == 0.0
    # the knock-out twin on the knocked-in surface
    bumped = _knocked_in(_bumped_ko_rate(family, _case(family)[1]))
    overlay = _ko_cash_diff_overlay(product, bumped, env, _autocallable_ko_cash)
    assert factory().price_folded(product, env, overlay) == pytest.approx(factory().price(bumped, env), rel=IDENTITY)


@pytest.mark.parametrize("factory, build", [(_heston_snowball, lambda: dated_snowball(CAL, T0)),
                                            (_heston_phoenix, lambda: dated_phoenix(CAL, T0, memory=False))])
def test_an_expired_product_refuses_a_non_empty_overlay(factory, build):
    product = build()
    env = flat_env(_ko_dates(product)[-1], spot=95.0)
    assert product.get_maturity(env) <= 0.0
    empty = _hand_overlay(product, env)
    assert empty.is_empty and factory().price_folded(product, env, empty) == factory().price(product, env)
    with pytest.raises(CapabilityError, match="expired"):
        factory().price_folded(product, env, _hand_overlay(product, env, no_ki=1.0))


# --- fail closed --------------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["heston_snowball", "slv_snowball", "heston_phoenix"])
def test_an_overlay_built_for_another_schedule_is_refused(family):
    factory, product, env = _case(family)
    other = make_snowball(ko_dates=(0.5, 1.0))
    with pytest.raises(ValidationError):
        factory().price_folded(product, env, _hand_overlay(other, env, ko=1.0))


def test_a_decided_knock_out_refuses_an_overlay_for_another_schedule():
    factory, product, env = _decided_cases()["heston-snowball"]
    with pytest.raises(ValidationError):
        factory().price_folded(product, env, _hand_overlay(make_snowball(), env, ko=1.0))


def test_a_ko_reset_product_does_not_fold_on_the_2d_route():
    """The 2D march fires one knock-out schedule: a KO-reset's pre/post schedules have no site (fail closed)."""
    from pde_3a.products import KO_RESET, env as ko_reset_env
    from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset

    product, env = make_ko_reset(KO_RESET), ko_reset_env()
    with pytest.raises(CapabilityError, match="KO-reset"):
        _heston_snowball().price_folded(product, env, _hand_overlay(product, env, ko=1.0))


def test_a_memory_phoenix_is_refused_as_price_refuses_it():
    product, env = dated_phoenix(CAL, T0), _dated_env()
    assert product.has_memory_coupon
    with pytest.raises(ValidationError, match="non-memory"):
        _heston_phoenix().price_folded(product, env, _hand_overlay(product, env, ko=1.0))


def test_a_lifecycle_state_is_refused_as_price_refuses_it():
    from quantark.asset.equity.lifecycle import AutocallableLifecycleState

    factory, product, env = _case("heston_snowball")
    state = AutocallableLifecycleState()
    with pytest.raises(CapabilityError, match="lifecycle_state"):
        factory().price(product, env, lifecycle_state=state)
    with pytest.raises(CapabilityError, match="lifecycle_state"):
        factory().price_folded(product, env, _hand_overlay(product, env, ko=1.0), lifecycle_state=state)


# --- AC9: per-solve state never mixes overlays --------------------------------------------------

@pytest.mark.parametrize("family", ["heston_snowball", "slv_snowball", "heston_phoenix", "slv_phoenix",
                                    "dated_snowball"])
def test_a_shared_solver_never_mixes_overlays(family):
    factory, product, env = _case(family)
    a = _hand_overlay(product, env, ko=1.0, ki=2.0)
    b = _hand_overlay(product, env, ko=-0.5, no_ki=0.7)
    shared = factory()
    seq = [shared.price_folded(product, env, a), shared.price(product, env),
           shared.price_folded(product, env, b), shared.price_folded(product, env, a)]
    fresh = [factory().price_folded(product, env, a), factory().price(product, env),
             factory().price_folded(product, env, b), factory().price_folded(product, env, a)]
    assert seq == fresh
    assert len({seq[0], seq[1], seq[2]}) == 3
    assert shared._fold is None


@pytest.mark.parametrize("family", ["heston_snowball", "heston_phoenix"])
def test_the_surrogate_route_and_greeks_after_a_folded_solve_are_a_fresh_solvers(family):
    factory, product, env = _case(family)

    def surrogate():
        return type(factory())(HESTON, statistics="flat_vol_surrogate", **GRID)

    shared = surrogate()
    shared.price_folded(product, env, _hand_overlay(product, env, ko=5.0, ki=5.0, no_ki=5.0))
    after, fresh = shared.price_with_events(product, env), surrogate().price_with_events(product, env)
    assert after.npv == fresh.npv
    for et, p in fresh.event_distribution.probabilities.items():
        np.testing.assert_array_equal(np.asarray(after.event_distribution.probabilities[et]), np.asarray(p))
    assert shared.calculate_greeks(product, env) == surrogate().calculate_greeks(product, env)
    assert shared._fold is None


@pytest.mark.parametrize("family", ["heston_snowball", "slv_phoenix"])
def test_a_folded_solve_on_the_bump_engine_folds_on_the_frozen_layout(family):
    """``get_trade_risk`` reprices on the bump engine GreeksCalculator resolves: it folds there too."""
    from quantark.asset.equity.riskmeasures import GreeksCalculator

    factory, product, env = _case(family)
    engine = factory()
    bump = GreeksCalculator()._resolve_bump_engine(product, env, engine)
    overlay = _hand_overlay(product, env, ko=2.5, ki=2.5)
    assert bump.price_folded(product, env, overlay) == engine.price_folded(product, env, overlay)
    bumped = make_env(spot=101.0)
    bumped_product = _bumped_ko_rate(family, product)
    twin = _ko_cash_diff_overlay(product, bumped_product, bumped, _autocallable_ko_cash)
    assert bump.price_folded(product, bumped, twin) == pytest.approx(bump.price(bumped_product, bumped),
                                                                     rel=IDENTITY)


# --- AC6: the folded leg PV against the same-model Monte Carlo event route ------------------------

#: a 2D grid fine enough that its discretization sits well inside the Monte Carlo error bar
AC6_GRID = dict(n_x=160, n_v=50, n_t=96)
AC6_PATHS = 40_000


def _ac6_env(spot=100.0):
    """A flat 20% grid surface: the SLV Monte Carlo builds its local-vol scaffold from it."""
    strikes = list(100.0 * np.exp(np.linspace(-0.8, 0.8, 11)))
    maturities = list(np.linspace(0.25, 1.0, 4))
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.03), valuation_date=datetime(2024, 1, 1), spot_quote=SpotQuote(spot=spot),
        vol_surface=GridVolSurface(strikes, maturities, np.full((4, 11), 0.20)),
        div_yield=ContinuousDividendYield(0.0),
    )


def _discrete_ki_phoenix():
    """Knock-in observed on the knock-out dates: both routes monitor it exactly (no continuous-monitoring bias)."""
    base = _phoenix_product()
    barrier = BarrierConfig(ko_barrier=105.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
                            ko_observation_dates=list(KO_DATES), ki_barrier=70.0,
                            ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=list(KO_DATES))
    return PhoenixOption(initial_price=100.0, strike=100.0, barrier_config=barrier, coupon_config=base.coupon_config,
                         payoff_config=base.payoff_config, contract_multiplier=1.0, maturity=1.0)


def _mc_params():
    return MCParams(num_paths=AC6_PATHS, time_steps=96, seed=2026)


AC6_CASES = {
    "heston_snowball": (lambda: HestonSnowballPDESolver(HESTON, **AC6_GRID),
                        lambda: HestonSnowballMCEngine(HESTON, _mc_params()), _discrete_ki_snowball),
    "slv_snowball": (lambda: HestonSLVSnowballPDESolver(HESTON, _leverage(), **AC6_GRID),
                     lambda: HestonSLVQESnowballMCEngine(HESTON, params=_mc_params(), leverage_surface=_leverage()),
                     _discrete_ki_snowball),
    "heston_phoenix": (lambda: HestonPhoenixPDESolver(HESTON, **AC6_GRID),
                       lambda: HestonPhoenixMCEngine(HESTON, _mc_params()), _discrete_ki_phoenix),
    "slv_phoenix": (lambda: HestonSLVPhoenixPDESolver(HESTON, _leverage(), **AC6_GRID),
                    lambda: HestonSLVQEPhoenixMCEngine(HESTON, params=_mc_params(), leverage_surface=_leverage()),
                    _discrete_ki_phoenix),
}


def _event_route_standard_error(overlay, dist, env, quantity, n_paths):
    """The Monte Carlo error of the route's leg PV: per path the leg pays one event's cash (multinomial)."""
    p = list(np.asarray(dist.probabilities[EventType.KO], dtype=float))
    x = [quantity * e.amount * env.get_discount_factor(e.payment_time) for e in overlay.ko]
    for et, entry in ((EventType.MATURITY_NO_KO, overlay.maturity_no_ki),
                      (EventType.MATURITY_WITH_KI, overlay.maturity_ki)):
        p.append(float(dist.probabilities.get(et, 0.0)))
        x.append(quantity * entry.amount * env.get_discount_factor(entry.payment_time))
    p, x = np.asarray(p), np.asarray(x)
    mean = float(p @ x)
    return float(np.sqrt(max(float(p @ (x * x)) - mean * mean, 0.0) / n_paths))


@pytest.mark.parametrize("quantity", [1.0, -3.0])
@pytest.mark.parametrize("spot", [100.0, 102.5, 80.0])
@pytest.mark.parametrize("case", list(AC6_CASES))
def test_folded_leg_pv_matches_the_same_model_mc_event_route(case, spot, quantity):
    """Every allowlisted leg type, branch-splitting legs included: the 2D fold's leg PV
    ``q·(price_folded − price) + constant_pv`` is the Monte Carlo route's ``Σ leg.value(dist)`` within three
    standard errors + 1e-3 of the event-dependent leg PV."""
    pde, mc, build = AC6_CASES[case]
    product, env = build(), _ac6_env(spot)
    route = mc().price_with_events(product, env, emit_distribution=True)
    dist = route.event_distribution
    legs = _legs(dist, env, splitting=True)
    assert {type(leg).__name__ for leg in legs} >= {
        "DeterministicLeg", "AccrualLeg", "FixedPayoffLeg", "AutocallableCashLeg"}
    overlay = fold_cash_legs(legs, product, env, quantity=quantity, unit_notional=100.0)
    folded_leg_pv = quantity * (pde().price_folded(product, env, overlay) - pde().price(product, env)) \
        + overlay.constant_pv
    route_leg_pv = sum(leg.value(dist, env, 100.0) for leg in legs)
    se = _event_route_standard_error(overlay, dist, env, quantity, AC6_PATHS)
    assert se > 0.0
    assert abs(folded_leg_pv - route_leg_pv) <= 3.0 * se + 1e-3 * abs(route_leg_pv - overlay.constant_pv)


@pytest.mark.parametrize("case", list(AC6_CASES))
def test_the_folded_event_weights_are_the_mc_routes_probabilities(case):
    """Each bucket's cash alone: the 2D march's weight is the same-model route's probability within 3 SE + 1e-3."""
    pde, mc, build = AC6_CASES[case]
    product, env = build(), _ac6_env()
    dist = mc().price_with_events(product, env, emit_distribution=True).event_distribution
    npv = pde().price(product, env)
    for kind, et in (("ko", EventType.KO), ("ki", EventType.MATURITY_WITH_KI), ("no_ki", EventType.MATURITY_NO_KO)):
        weight = pde().price_folded(product, env, _hand_overlay(product, env, **{kind: 1.0})) - npv
        p = float(np.sum(dist.probabilities[et]))
        assert abs(weight - p) <= 3.0 * np.sqrt(p * (1.0 - p) / AC6_PATHS) + 1e-3 * p, kind
