"""The Black-Scholes barrier MC decides at the valuation instant on the schedule record's own level (W18b, user
approval 2026-10-10).

Two decisions of ``BarrierOptionMCEngine`` read ``product.barrier`` while the rest of its DISCRETE path monitors each
resolved record at its own level and pays a knock-out the hit record's payoff (``_build_discrete_grid``,
``payoffs[first_idx]``):

(a) the exercise-date value, ``barrier_expiry_value`` -- also the local-vol / Heston / SLV barrier MC routes' value at
    T = 0 since W9;
(b) the "hit at valuation" decision before any path is simulated.

Both now read the first record at or after the valuation instant (the record on the valuation date when there is
one; the PDE's ``_next_observation`` rule otherwise): its level, and on a knock-out its payoff. This is the barrier
PDE's existing rule on the same contracts. CONTINUOUS / EXPIRY monitoring keep ``product.barrier``; a record without
a level takes ``product.barrier`` (the product's normalisation).
"""
from __future__ import annotations

import pytest

from test_vol_dated_schedules import BARRIER_ROUTES, DAYS, EXERCISE, INITIAL, _on, _quiet, env

from quantark.asset.equity.engine.mc import BarrierOptionMCEngine
from quantark.asset.equity.engine.pde.barrier_pde_solver import BarrierPDESolver
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.product.option import BarrierOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.util.enum import BarrierType, ObservationType, OptionType
from quantark.util.exceptions import ValidationError

#: (barrier type, option type, product.barrier, the records' level, (spot between the two, beyond both, inside both))
KINDS = {
    "up_out_call": (BarrierType.UP_OUT, OptionType.CALL, 120.0, 115.0, (117.0, 125.0, 110.0)),
    "up_in_call": (BarrierType.UP_IN, OptionType.CALL, 120.0, 115.0, (117.0, 125.0, 110.0)),
    "down_out_put": (BarrierType.DOWN_OUT, OptionType.PUT, 85.0, 90.0, (87.0, 80.0, 95.0)),
    "down_in_put": (BarrierType.DOWN_IN, OptionType.PUT, 85.0, 90.0, (87.0, 80.0, 95.0)),
}
REBATE = 5.0
TOL = dict(rel=1e-12, abs=1e-12)


def _mc():
    return MCParams(seed=11, num_paths=512, time_steps=73)


#: the Black-Scholes barrier MC and the three vol-model MC routes (the latter return its expiry value at T = 0)
MC_ROUTES = {
    "bs_mc": (lambda: BarrierOptionMCEngine(params=_mc()), False),
    **{r: BARRIER_ROUTES[r] for r in ("lv_mc", "heston_mc", "slv_mc")},
}


def _records(form: str, level, days, payoff=None):
    extra = {} if payoff is None else {"payoff": payoff}
    if form == "dated":
        return [ObservationRecord(observation_date=INITIAL + _delta(d), barrier=level, **extra) for d in days]
    return [ObservationRecord(observation_time=d / 365.0, barrier=level, **extra) for d in days]


def _delta(days: int):
    from datetime import timedelta
    return timedelta(days=days)


def _product(kind: str, form: str, barrier: float, level, days, payoff=None) -> BarrierOption:
    barrier_type, option_type, _, _, _ = KINDS[kind]
    kw = dict(strike=100.0, option_type=option_type, barrier=barrier, barrier_type=barrier_type, rebate=REBATE,
              observation_type=ObservationType.DISCRETE,
              observation_schedule=ObservationSchedule(records=_records(form, level, days, payoff)))
    if form == "dated":
        return BarrierOption(exercise_date=EXERCISE, initial_date=INITIAL, **kw)
    return BarrierOption(maturity=1.0, initial_date=INITIAL, **kw)


def _aged(product: BarrierOption, day: int, keep_landing: bool) -> BarrierOption:
    """Both forms aged to contract day ``day`` (past records dropped); ``keep_landing`` keeps a record on that day."""
    valuation = _on(day)
    product.time_shift(day / 365.0, valuation, env(100.0, valuation), keep_landing=keep_landing)
    return product


def _hand(kind: str, spot: float, level: float, ko_cash: float = REBATE) -> float:
    """The settled value on the exercise date (payment discount 1): the decision at ``level``."""
    barrier_type, option_type, _, _, _ = KINDS[kind]
    hit = spot >= level if barrier_type.is_up else spot <= level
    payoff = max(spot - 100.0, 0.0) if option_type == OptionType.CALL else max(100.0 - spot, 0.0)
    if barrier_type.is_knock_out:
        return ko_cash if hit else payoff
    return payoff if hit else REBATE


# ======================================================================================================================
# (a) the exercise date
# ======================================================================================================================

#: one record on the exercise date, or the twelve monthly records (the last on the exercise date)
SCHEDULES = {"single": (365,), "monthly": tuple(DAYS)}


def _settled(kind, form, barrier, level, schedule, payoff=None):
    product = _product(kind, form, barrier, level, SCHEDULES[schedule], payoff)
    if form == "numeric":
        _aged(product, 365, keep_landing=True)     # maturity 0.0, the landing record at 0.0
    return product


EXPIRY_CASES = [(route, kind, form, schedule, which) for route in MC_ROUTES for kind in KINDS
                for form in ("dated", "numeric") for schedule in SCHEDULES for which in range(3)]
_EXPIRY_IDS = [f"{r}-{k}-{f}-{s}-{('between', 'beyond', 'inside')[w]}" for r, k, f, s, w in EXPIRY_CASES]


@pytest.mark.parametrize("route,kind,form,schedule,which", EXPIRY_CASES, ids=_EXPIRY_IDS)
def test_the_exercise_date_value_decides_on_the_record_level(route, kind, form, schedule, which):
    """``product.barrier`` B, every record at L: the settled value decided at L, equal (bitwise) to the contract
    booked at L. Spot "between" is beyond L but short of B: the cell that moved."""
    _, _, barrier, level, spots = KINDS[kind]
    spot = spots[which]
    factory, grid = MC_ROUTES[route]
    e = env(spot, EXERCISE, grid=grid)
    priced = _quiet(lambda: factory().price(_settled(kind, form, barrier, level, schedule), e))
    booked = _quiet(lambda: factory().price(_settled(kind, form, level, level, schedule), e))
    assert priced == booked
    assert priced == pytest.approx(_hand(kind, spot, level), **TOL)


@pytest.mark.parametrize("route", list(MC_ROUTES))
@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("form", ["dated", "numeric"])
def test_the_exercise_date_value_equals_the_black_scholes_pde(route, kind, form):
    """The one-record schedule (the PDE refuses a dated schedule with records behind the valuation date), the spot
    between the levels: the MC routes equal the Black-Scholes barrier PDE's settled value."""
    _, _, barrier, level, spots = KINDS[kind]
    factory, grid = MC_ROUTES[route]
    e = env(spots[0], EXERCISE, grid=grid)
    priced = _quiet(lambda: factory().price(_settled(kind, form, barrier, level, "single"), e))
    reference = _quiet(lambda: BarrierPDESolver(PDEParams()).price(_settled(kind, form, barrier, level, "single"), e))
    assert priced == pytest.approx(reference, **TOL)


@pytest.mark.parametrize("route", list(MC_ROUTES))
@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("form", ["dated", "numeric"])
def test_the_exercise_date_knock_out_pays_the_record_payoff(route, kind, form):
    """Records paying 4.5 (``product.rebate`` 5): a knock-out hit on the exercise date pays 4.5. A knock-in has one
    rebate: a record paying anything else is refused (user ruling 2026-10-10 (a),
    ``test_knock_in_record_payoff_refused``)."""
    _, _, barrier, level, spots = KINDS[kind]
    factory, grid = MC_ROUTES[route]
    for spot in spots:
        e = env(spot, EXERCISE, grid=grid)
        if "_in_" in kind:
            with pytest.raises(ValidationError, match=r"a knock-in has one rebate \(5\.0\); a schedule record pays 4\.5"):
                _quiet(lambda: factory().price(_settled(kind, form, barrier, level, "single", payoff=4.5), e))
            continue
        priced = _quiet(lambda: factory().price(_settled(kind, form, barrier, level, "single", payoff=4.5), e))
        assert priced == pytest.approx(_hand(kind, spot, level, ko_cash=4.5), **TOL), spot


@pytest.mark.parametrize("route", list(MC_ROUTES))
@pytest.mark.parametrize("form", ["dated", "numeric"])
def test_the_exercise_date_value_of_a_record_without_a_level_reads_the_product_barrier(route, form):
    factory, grid = MC_ROUTES[route]
    for spot in (117.0, 125.0):
        e = env(spot, EXERCISE, grid=grid)
        priced = _quiet(lambda: factory().price(_settled("up_out_call", form, 120.0, None, "single"), e))
        assert priced == pytest.approx(_hand("up_out_call", spot, 120.0), **TOL)


# ======================================================================================================================
# (b) hit at valuation, mid-life (Black-Scholes barrier MC)
# ======================================================================================================================

def _engine():
    return BarrierOptionMCEngine(params=MCParams(seed=11, num_paths=20000, time_steps=73))


def _mid(kind, form, barrier, level, day, keep_landing):
    return _aged(_product(kind, form, barrier, level, DAYS), day, keep_landing)


@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("form", ["dated", "numeric"])
def test_a_valuation_date_observation_decides_on_its_level_as_the_pde_does(kind, form):
    """Day 91 holds an observation; the spot is beyond its level L but short of ``product.barrier``. Knocked out:
    the rebate, equal to the Black-Scholes barrier PDE (both pay it at the terminal). Knocked in: the vanilla, as the
    PDE (within MC error), and bitwise the contract booked at L."""
    _, _, barrier, level, spots = KINDS[kind]
    e = env(spots[0], _on(91))
    priced = _quiet(lambda: _engine().price(_mid(kind, form, barrier, level, 91, True), e))
    booked = _quiet(lambda: _engine().price(_mid(kind, form, level, level, 91, True), e))
    pde = _quiet(lambda: BarrierPDESolver(PDEParams()).price(_mid(kind, form, barrier, level, 91, True), e))
    assert priced == booked
    if "_out_" in kind:
        assert priced == pytest.approx(pde, **TOL)
    else:
        assert priced == pytest.approx(pde, rel=2e-2)


@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("form", ["dated", "numeric"])
def test_hit_at_valuation_between_observations_reads_the_next_record(kind, form):
    """Day 92, no observation today: the decision reads the next record (day 121), as the PDE's ``_next_observation``
    does -- bitwise the contract booked at its level."""
    _, _, barrier, level, spots = KINDS[kind]
    e = env(spots[0], _on(92))
    priced = _quiet(lambda: _engine().price(_mid(kind, form, barrier, level, 92, False), e))
    booked = _quiet(lambda: _engine().price(_mid(kind, form, level, level, 92, False), e))
    assert priced == booked


def test_continuous_monitoring_keeps_the_product_barrier():
    """No schedule: the hit at valuation reads ``product.barrier`` (unchanged)."""
    kw = dict(strike=100.0, option_type=OptionType.CALL, barrier=120.0, barrier_type=BarrierType.UP_OUT,
              rebate=REBATE, observation_type=ObservationType.CONTINUOUS, maturity=1.0)
    e = env(121.0, INITIAL)
    assert _quiet(lambda: _engine().price(BarrierOption(**kw), e)) == pytest.approx(
        REBATE * e.get_discount_factor(1.0), **TOL)
