"""Cash-leg folding on the 1D PDE solvers (ship spec 2026-10-21, task T5).

The overlay reaches the two-surface march where the product's own cash does: each knock-out record's
payoff (interior events, the terminal knock-out, the valuation-date column, the knock-out edge row),
each maturity branch's terminal payoff (the knock-in branch also on the knocked-in rows, as the
event statistics' knock-in column carries it; the alive branch on the alive no-event rows, as the
product's own redemption), and the valuation-instant short-circuits. The 1D event statistics march
in the same march as the value, so knock-out and knock-in cash fold to round-off against the PDE's
own event route (AC4 / AC5); the product twins (AC14) pin the cash against the product itself.
"""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.phoenix_vol_pde_solvers import (
    HestonPhoenixPDESolver,
    LocalVolPhoenixPDESolver,
)
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSnowballPDESolver,
    LocalVolSnowballPDESolver,
)
from quantark.asset.equity.engine.pde_engine import PDEEngine
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.snowball_config import BarrierConfig
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.cashleg import EventType
from quantark.cashleg.folding import FoldedEventCash, FoldOverlay, _event_structure, fold_cash_legs
from quantark.execution.errors import CapabilityError
from quantark.modelvalidation.builders.equity_ko_reset import make_ko_reset
from quantark.param import ContinuousDividendYield, FlatRateCurve, GridVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.enum import CouponPayType, ObservationType
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

CAL = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
T0, LIVE = datetime(2026, 3, 16), datetime(2026, 4, 1)
KO_DATES = (0.25, 0.5, 0.75, 1.0)
#: the spec's bar for a route that shares the fold's discretization (AC4 relative, AC5 per notional)
IDENTITY = 1e-10
PARITY = 1e-9


def _skewed_env(spot=100.0, r=0.03, q=0.01, valuation=datetime(2026, 1, 1)):
    """A real downside skew: the local vol differs from the strike vol where the barriers are."""
    strikes = list(100.0 * np.exp(np.linspace(-0.6, 0.4, 11)))
    maturities = list(np.linspace(0.25, 1.0, 4))
    grid = np.array([[0.34 - 0.22 * (np.log(k / 100.0) + 0.6) for k in strikes] for _ in maturities])
    return PricingEnvironment(
        rate_curve=FlatRateCurve(r), valuation_date=valuation, spot_quote=SpotQuote(spot=spot),
        vol_surface=GridVolSurface(strikes, maturities, grid), div_yield=ContinuousDividendYield(q),
    )


def _reverse_snowball():
    barrier = BarrierConfig(
        ko_barrier=97.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
        ko_observation_dates=list(KO_DATES), ki_barrier=125.0,
        ki_observation_type=ObservationType.CONTINUOUS, ki_continuous=True,
    )
    return SnowballOption(initial_price=100.0, strike=100.0, barrier_config=barrier, contract_multiplier=1.0,
                          maturity=1.0, is_reverse=True)


def _phoenix_product(**kw):
    return make_phoenix(ko_dates=KO_DATES, coupon_barrier=(80.0,) * 4, ki_barrier=70.0, **kw)


def _gbm(cls):
    return lambda: cls(PDEParams())


FAMILIES = {
    "snowball": (_gbm(SnowballPDESolver), lambda: make_snowball(ko_dates=KO_DATES), make_env),
    "reverse_snowball": (_gbm(SnowballPDESolver), _reverse_snowball, make_env),
    "dated_snowball": (_gbm(SnowballPDESolver), lambda: dated_snowball(CAL, T0),
                       lambda spot=100.0: flat_env(LIVE, spot=spot)),
    "phoenix": (_gbm(PhoenixPDESolver), _phoenix_product, make_env),
    "phoenix_no_ko_after_ki": (_gbm(PhoenixPDESolver), lambda: _phoenix_product(disable_ko_after_ki=True),
                               make_env),
    "phoenix_expiry": (_gbm(PhoenixPDESolver), lambda: _phoenix_product(coupon_pay=CouponPayType.EXPIRY),
                       make_env),
    "dated_phoenix": (_gbm(PhoenixPDESolver), lambda: dated_phoenix(CAL, T0),
                      lambda spot=100.0: flat_env(LIVE, spot=spot)),
    "ko_reset": (_gbm(KOResetSnowballPDESolver), lambda: make_ko_reset(KO_RESET),
                 lambda spot=100.0: ko_reset_env(spot=spot)),
    "dated_ko_reset": (_gbm(KOResetSnowballPDESolver), lambda: dated_ko_reset(CAL, T0),
                       lambda spot=100.0: flat_env(LIVE, spot=spot)),
    "lv_snowball": (_gbm(LocalVolSnowballPDESolver), lambda: make_snowball(ko_dates=KO_DATES), _skewed_env),
    "lv_phoenix": (_gbm(LocalVolPhoenixPDESolver), _phoenix_product, _skewed_env),
    "lv_dated_phoenix": (_gbm(LocalVolPhoenixPDESolver), lambda: dated_phoenix(CAL, T0),
                         lambda spot=100.0: _skewed_env(spot=spot, valuation=LIVE)),
    "engine_snowball": (lambda: PDEEngine(PDEParams()), lambda: make_snowball(ko_dates=KO_DATES), make_env),
    "engine_phoenix": (lambda: PDEEngine(PDEParams()), _phoenix_product, make_env),
    "engine_ko_reset": (lambda: PDEEngine(PDEParams()), lambda: make_ko_reset(KO_RESET),
                        lambda spot=100.0: ko_reset_env(spot=spot)),
}
# ATM, just under the knock-out level, just above the knock-in level
SPOTS = {"snowball": (100.0, 102.5, 76.0), "reverse_snowball": (100.0, 97.5, 124.0),
         "dated_snowball": (100.0, 102.5, 76.0), "phoenix": (100.0, 104.5, 71.0),
         "phoenix_no_ko_after_ki": (100.0, 104.5, 71.0), "phoenix_expiry": (100.0, 104.5, 71.0),
         "dated_phoenix": (100.0, 102.5, 76.0), "ko_reset": (100.0, 102.5, 81.0),
         "dated_ko_reset": (100.0, 102.5, 76.0), "lv_snowball": (100.0, 102.5, 76.0),
         "lv_phoenix": (100.0, 104.5, 71.0), "lv_dated_phoenix": (100.0, 102.5, 76.0), "engine_snowball": (100.0, 102.5, 76.0),
         "engine_phoenix": (100.0, 104.5, 71.0), "engine_ko_reset": (100.0, 102.5, 81.0)}
KO_RESETS = ("ko_reset", "dated_ko_reset", "engine_ko_reset")


def _case(family, spot=100.0):
    factory, build, make = FAMILIES[family]
    return factory, build(), make(spot=spot)


def _knock_out_and_in_weight(dist):
    return _p(dist, EventType.KO) + _p(dist, EventType.MATURITY_WITH_KI)


# --- contract ----------------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [SnowballPDESolver, PhoenixPDESolver, KOResetSnowballPDESolver,
                                 LocalVolSnowballPDESolver, LocalVolPhoenixPDESolver, PDEEngine])
def test_the_1d_solvers_and_the_pde_engine_declare_folding(cls):
    assert cls.supports_cash_leg_folding is True


@pytest.mark.parametrize("cls", [HestonSnowballPDESolver, HestonPhoenixPDESolver])
def test_the_2d_solvers_do_not_inherit_the_1d_fold(cls):
    """Their price is a 2D march: the 1D fold would price the 1D surrogate. They fold on their own 2D march
    (task T6, test/test_cashleg/test_fold_pde_2d.py)."""
    assert cls.supports_cash_leg_folding is True
    assert cls.price_folded is not SnowballPDESolver.price_folded
    assert cls.price_folded is not PhoenixPDESolver.price_folded


@pytest.mark.parametrize("family", list(FAMILIES))
def test_an_empty_overlay_is_the_plain_price_bitwise(family):
    factory, product, env = _case(family)
    overlay = _hand_overlay(product, env)
    assert overlay.is_empty
    assert factory().price_folded(product, env, overlay) == factory().price(product, env)


def test_a_lifecycle_state_is_refused_as_price_refuses_it():
    from quantark.asset.equity.lifecycle import AutocallableLifecycleState

    factory, product, env = _case("snowball")
    state = AutocallableLifecycleState()
    with pytest.raises(CapabilityError, match="lifecycle_state"):
        factory().price(product, env, lifecycle_state=state)
    with pytest.raises(CapabilityError, match="lifecycle_state"):
        factory().price_folded(product, env, _hand_overlay(product, env, ko=1.0), lifecycle_state=state)


def test_the_pde_engine_refuses_a_product_whose_solver_cannot_fold():
    from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption
    from quantark.util.enum import OptionType

    option = EuropeanVanillaOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
    overlay = FoldOverlay(ko=(), maturity_no_ki=FoldedEventCash("main", 1.0, 1.0, 1.0), maturity_ki=None)
    with pytest.raises(CapabilityError):
        PDEEngine(PDEParams()).price_folded(option, make_env(), overlay)


@pytest.mark.parametrize("family", ["snowball", "phoenix", "ko_reset"])
def test_the_pde_engine_folds_on_its_dispatched_solver_bitwise(family):
    factory, product, env = _case(family)
    overlay = _hand_overlay(product, env, ko=1.5, ki=2.0, no_ki=0.5)
    direct = factory().price_folded(product, env, overlay)
    assert PDEEngine(PDEParams()).price_folded(product, env, overlay) == direct


def test_a_folded_request_never_marches_the_event_statistics():
    """The statistics columns are the event route; folded cash in them would corrupt it (stats run unfolded)."""
    from quantark.asset.equity.engine.pde.core.prepare import Operation, OutputRequest

    factory, product, env = _case("snowball")
    overlay = _hand_overlay(product, env, ko=1.0)
    with pytest.raises(CapabilityError):
        factory()._run(product, env, OutputRequest(Operation.EVENT_STATS, event_stats=True, fold=overlay))


# --- AC4: identity on the engine's own event weights -------------------------------------------

@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("spot_index", [0, 1, 2])
def test_knock_out_and_knock_in_cash_add_c_times_their_route_probabilities(family, spot_index):
    """c/DF(pay) on every knock-out record and on the knock-in branch: the folded PV rises by c·(ΣP_KO + P_KI)."""
    factory, product, env = _case(family, SPOTS[family][spot_index])
    c = 2.5
    dist = _route(factory(), product, env).event_distribution
    npv = factory().price(product, env)
    folded = factory().price_folded(product, env, _hand_overlay(product, env, ko=c, ki=c))
    assert folded - npv == pytest.approx(c * _knock_out_and_in_weight(dist), rel=IDENTITY,
                                         abs=IDENTITY * abs(npv))


@pytest.mark.parametrize("family", list(FAMILIES))
def test_knock_out_cash_alone_adds_c_times_the_knock_out_probability(family):
    factory, product, env = _case(family)
    c = 2.5
    dist = _route(factory(), product, env).event_distribution
    folded = factory().price_folded(product, env, _hand_overlay(product, env, ko=c))
    assert folded - factory().price(product, env) == pytest.approx(
        c * _p(dist, EventType.KO), rel=IDENTITY, abs=IDENTITY * abs(folded))


@pytest.mark.parametrize("family", KO_RESETS)
@pytest.mark.parametrize("spot_index", [0, 1, 2])
def test_ko_reset_no_ki_cash_lands_at_the_pre_maturity_with_its_own_probability(family, spot_index):
    """The KO-reset route reports the alive bucket explicitly (U_pre): its cash folds to round-off too."""
    factory, product, env = _case(family, SPOTS[family][spot_index])
    dist = _route(factory(), product, env).event_distribution
    c = 3.0
    folded = factory().price_folded(product, env, _hand_overlay(product, env, no_ki=c))
    npv = factory().price(product, env)
    assert folded - npv == pytest.approx(c * _p(dist, EventType.MATURITY_NO_KO), rel=IDENTITY,
                                         abs=IDENTITY * abs(npv))


# --- AC5: parity with the PDE's own event route ------------------------------------------------

@pytest.mark.parametrize("quantity", [1.0, -3.0])
@pytest.mark.parametrize("family", list(FAMILIES))
@pytest.mark.parametrize("spot_index", [0, 1])
def test_folded_trade_matches_the_pde_event_route(family, quantity, spot_index):
    """Every allowlisted leg type, branch-splitting legs included."""
    factory, product, env = _case(family, SPOTS[family][spot_index])
    route = _route(factory(), product, env)
    dist = route.event_distribution
    legs = _legs(dist, env, splitting=True)
    assert {type(leg).__name__ for leg in legs} >= {
        "DeterministicLeg", "AccrualLeg", "FixedPayoffLeg", "AutocallableCashLeg"}
    overlay = fold_cash_legs(legs, product, env, quantity=quantity, unit_notional=100.0)
    assert not overlay.is_empty
    folded = quantity * factory().price_folded(product, env, overlay) + overlay.constant_pv
    expected = quantity * route.npv + sum(leg.value(dist, env, 100.0) for leg in legs)
    notional = max(100.0 * abs(quantity), UNIT)
    assert abs(folded - expected) <= PARITY * notional


# --- AC8: lifecycle -----------------------------------------------------------------------------

def _lifecycle_cases():
    sb, ph, kr = dated_snowball(CAL, T0), dated_phoenix(CAL, T0), dated_ko_reset(CAL, T0)
    gbm = {SnowballPDESolver: _gbm(SnowballPDESolver), PhoenixPDESolver: _gbm(PhoenixPDESolver),
           KOResetSnowballPDESolver: _gbm(KOResetSnowballPDESolver)}
    return {
        "snowball-knocked-in": (gbm[SnowballPDESolver], _knocked_in(sb), flat_env(LIVE, spot=90.0)),
        "phoenix-knocked-in": (gbm[PhoenixPDESolver], _knocked_in(ph), flat_env(LIVE, spot=90.0)),
        "ko-reset-knocked-in": (gbm[KOResetSnowballPDESolver], _knocked_in(kr), flat_env(LIVE, spot=90.0)),
        "lv-snowball-knocked-in": (_gbm(LocalVolSnowballPDESolver), _knocked_in(make_snowball(ko_dates=KO_DATES)),
                                   _skewed_env(spot=90.0)),
        "snowball-ko-decided-at-valuation": (gbm[SnowballPDESolver], sb, flat_env(_ko_dates(sb)[2], spot=110.0)),
        "phoenix-ko-decided-at-valuation": (gbm[PhoenixPDESolver], ph, flat_env(_ko_dates(ph)[2], spot=110.0)),
        "ko-reset-ko-decided-at-valuation": (gbm[KOResetSnowballPDESolver], kr,
                                             flat_env(_ko_dates(kr)[2], spot=110.0)),
    }


@pytest.mark.parametrize("case", list(_lifecycle_cases()))
def test_lifecycle_states_fold_like_the_event_route(case):
    factory, product, env = _lifecycle_cases()[case]
    route = _route(factory(), product, env)
    dist = route.event_distribution
    legs = _legs(dist, env, splitting=True)
    overlay = fold_cash_legs(legs, product, env, quantity=1.0, unit_notional=100.0)
    folded = factory().price_folded(product, env, overlay) + overlay.constant_pv
    expected = route.npv + sum(leg.value(dist, env, 100.0) for leg in legs)
    assert route.npv == factory().price(product, env)
    assert abs(folded - expected) <= PARITY * UNIT
    if "decided" in case:
        assert _p(dist, EventType.KO) == pytest.approx(1.0, abs=1e-12)
        # the short-circuit pays the deciding record's cash at its settlement, exactly
        structure = _event_structure(product, env)
        hit = next(i for i, s in enumerate(structure.ko)
                   if s.determination_time == 0.0 and s.schedule in ("main", "pre"))
        entry = overlay.ko[hit]
        assert folded - overlay.constant_pv - factory().price(product, env) == pytest.approx(
            entry.amount * env.get_discount_factor(entry.payment_time), rel=1e-12)


@pytest.mark.parametrize("cls, build", [(SnowballPDESolver, dated_snowball), (PhoenixPDESolver, dated_phoenix)])
def test_an_expired_product_refuses_a_non_empty_overlay(cls, build):
    product = build(CAL, T0)
    env = flat_env(_ko_dates(product)[-1], spot=95.0)
    assert product.get_maturity(env) <= 0.0
    empty = _hand_overlay(product, env)
    assert empty.is_empty and cls(PDEParams()).price_folded(product, env, empty) == cls(PDEParams()).price(
        product, env)
    with pytest.raises(CapabilityError, match="expired"):
        cls(PDEParams()).price_folded(product, env, _hand_overlay(product, env, no_ki=1.0))


# --- fail closed --------------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["snowball", "phoenix", "lv_snowball"])
def test_an_overlay_built_for_another_schedule_is_refused(family):
    factory, product, env = _case(family)
    other = make_snowball(ko_dates=(0.5, 1.0))
    with pytest.raises(ValidationError):
        factory().price_folded(product, env, _hand_overlay(other, env, ko=1.0))


def test_a_decided_knock_out_refuses_an_overlay_for_another_schedule():
    sb = dated_snowball(CAL, T0)
    env = flat_env(_ko_dates(sb)[2], spot=110.0)
    with pytest.raises(ValidationError):
        SnowballPDESolver(PDEParams()).price_folded(sb, env, _hand_overlay(make_snowball(), env, ko=1.0))


def test_a_ko_reset_maturity_entry_at_the_wrong_branch_time_is_refused():
    factory, product, env = _case("ko_reset")
    overlay = _hand_overlay(product, env, no_ki=1.0)
    moved = replace(overlay, maturity_no_ki=replace(overlay.maturity_no_ki,
                                                    determination_time=overlay.maturity_ki.determination_time,
                                                    payment_time=overlay.maturity_ki.payment_time))
    with pytest.raises(ValidationError, match="no_ki"):
        factory().price_folded(product, env, moved)


def test_a_ko_reset_overlay_with_its_schedules_swapped_is_refused():
    factory, product, env = _case("ko_reset")
    overlay = _hand_overlay(product, env, ko=1.0)
    swapped = replace(overlay, ko=tuple(replace(e, schedule="post" if e.schedule == "pre" else "pre")
                                        for e in overlay.ko))
    with pytest.raises(ValidationError):
        factory().price_folded(product, env, swapped)


# --- AC9: caches and per-solve state never mix overlays -----------------------------------------

@pytest.mark.parametrize("family", ["snowball", "dated_phoenix", "ko_reset", "lv_snowball", "engine_snowball"])
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


@pytest.mark.parametrize("family", ["snowball", "dated_phoenix", "ko_reset"])
def test_the_event_route_and_greeks_after_a_folded_solve_are_a_fresh_solvers(family):
    factory, product, env = _case(family)
    shared = factory()
    shared.price_folded(product, env, _hand_overlay(product, env, ko=5.0, ki=5.0, no_ki=5.0))
    after, fresh = _route(shared, product, env), _route(factory(), product, env)
    assert after.npv == fresh.npv
    for et, p in fresh.event_distribution.probabilities.items():
        np.testing.assert_array_equal(np.asarray(after.event_distribution.probabilities[et]), np.asarray(p))
    assert shared.calculate_greeks(product, env) == factory().calculate_greeks(product, env)
    assert shared._fold is None


def _march_identities(monkeypatch):
    import quantark.asset.equity.engine.pde.core.prepare as prep_mod
    seen, real = [], prep_mod.march

    def spy(prepared):
        seen.append({k: prepared.identity[k] for k in ("layout", "factorization", "problem")}
                    | {"theta": np.array(prepared.theta_schedule, copy=True)})
        return real(prepared)

    monkeypatch.setattr(prep_mod, "march", spy)
    return seen


@pytest.mark.parametrize("family", ["snowball", "phoenix", "ko_reset", "lv_snowball"])
def test_folding_changes_the_problem_identity_and_nothing_of_the_grid(family, monkeypatch):
    """D8: the overlay is in the march's identity; the layout, damping and steps are the plain solve's."""
    factory, product, env = _case(family)
    seen = _march_identities(monkeypatch)
    factory().price(product, env)
    factory().price_folded(product, env, _hand_overlay(product, env, ko=1.0))
    factory().price_folded(product, env, _hand_overlay(product, env, ko=2.0))
    plain, a, b = seen[-3:]
    assert plain["layout"] == a["layout"] == b["layout"]
    assert plain["factorization"] == a["factorization"] == b["factorization"]
    np.testing.assert_array_equal(plain["theta"], a["theta"])
    assert len({plain["problem"], a["problem"], b["problem"]}) == 3


def test_a_folded_solve_on_the_bump_engine_folds_on_the_frozen_layout():
    """``get_trade_risk`` reprices on the bump engine GreeksCalculator resolves: it folds there too."""
    from quantark.asset.equity.riskmeasures import GreeksCalculator

    factory, product, env = _case("engine_snowball")
    engine = factory()
    bump = GreeksCalculator()._resolve_bump_engine(product, env, engine)
    overlay = _hand_overlay(product, env, ko=2.5, ki=2.5)
    assert bump.price_folded(product, env, overlay) == engine.price_folded(product, env, overlay)
    bumped = make_env(spot=101.0)
    dist = _route(bump, product, bumped).event_distribution
    folded = bump.price_folded(product, bumped, _hand_overlay(product, bumped, ko=2.5, ki=2.5))
    assert folded - bump.price(product, bumped) == pytest.approx(
        2.5 * _knock_out_and_in_weight(dist), rel=IDENTITY)


# --- AC14: product twins ------------------------------------------------------------------------

def _ko_reset_ko_cash(product, env):
    """The PDE solver's own record cash, in the structure's merged order (pre first at a tie)."""
    solver = KOResetSnowballPDESolver(PDEParams())
    tau = product.get_maturity(env)
    tagged = []
    for tag, config in ((0, product.barrier_config), (1, product.post_barrier_config)):
        tagged += [(r.observation_time, tag, float(r.payoff))
                   for r in solver._resolve_ko_records(product, env, config)
                   if 0.0 <= r.observation_time <= tau]
    return [cash for _, _, cash in sorted(tagged, key=lambda r: (r[0], r[1]))]


def _bumped_ko_rate(family, product):
    """The same family with a higher KO coupon, rebuilt by its own builder (records carry the cash)."""
    if family in ("snowball", "lv_snowball", "engine_snowball"):
        return make_snowball(ko_dates=KO_DATES, ko_rate=0.17)
    if family == "dated_snowball":
        return dated_snowball(CAL, T0, ko_rate=0.14, rebate_rate=0.12)   # the rebate stays at the base 0.12
    if family in ("dated_phoenix", "lv_dated_phoenix"):
        bumped = deepcopy(product)
        bumped.barrier_config = replace(bumped.barrier_config, ko_rate=0.05)
        return bumped
    if family == "ko_reset":
        return make_ko_reset(dict(KO_RESET, pre_ko_rate=KO_RESET["pre_ko_rate"] + 0.02,
                                  post_ko_rate=KO_RESET["post_ko_rate"] + 0.02))
    return dated_ko_reset(CAL, T0, post_rate=0.05, rebate_rate=0.15)   # base: post 0.03, rebate = pre 0.15


@pytest.mark.parametrize("family, amounts", [
    ("snowball", _autocallable_ko_cash),
    ("dated_snowball", _autocallable_ko_cash),
    ("dated_phoenix", _autocallable_ko_cash),
    ("lv_snowball", _autocallable_ko_cash),
    ("lv_dated_phoenix", _autocallable_ko_cash),
    ("ko_reset", _ko_reset_ko_cash),
    ("dated_ko_reset", _ko_reset_ko_cash),
])
@pytest.mark.parametrize("spot_index", [0, 1])
def test_ko_cash_twin_equals_the_product_with_a_higher_ko_rate(family, amounts, spot_index):
    factory, product, env = _case(family, SPOTS[family][spot_index])
    bumped = _bumped_ko_rate(family, product)
    overlay = _ko_cash_diff_overlay(product, bumped, env, amounts)
    folded = factory().price_folded(product, env, overlay)
    assert folded == pytest.approx(factory().price(bumped, env), rel=IDENTITY)


@pytest.mark.parametrize("family", ["snowball", "ko_reset", "lv_snowball", "dated_snowball"])
def test_maturity_cash_twin_equals_the_product_with_a_higher_rebate(family):
    factory, product, env = _case(family)
    bumped = deepcopy(product)
    bumped.payoff_config = replace(bumped.payoff_config, rebate_rate=bumped.payoff_config.rebate_rate + 0.03)
    diffs = {bumped.get_maturity_payoff_v0(s, pricing_env=env) - product.get_maturity_payoff_v0(s, pricing_env=env)
             for s in (80.0, 100.0, 102.0)}
    assert len(diffs) == 1 and next(iter(diffs)) > 0
    assert bumped.get_maturity_payoff_v1(90.0, pricing_env=env) == product.get_maturity_payoff_v1(90.0, pricing_env=env)
    structure = _event_structure(product, env)
    # the twin's redemption is discounted by the product's own terminal delay: cash at the slot's payment
    delay = env.get_discount_factor(structure.no_ki.payment_time) / env.get_discount_factor(
        structure.no_ki.determination_time)
    product_delay = env.get_discount_factor(_event_structure(product, env).ki.payment_time) / \
        env.get_discount_factor(_event_structure(product, env).ki.determination_time)
    cash = next(iter(diffs)) * product_delay / delay
    folded = factory().price_folded(product, env, _branch_overlay(product, env, no_ki=cash))
    assert folded == pytest.approx(factory().price(bumped, env), rel=IDENTITY)
