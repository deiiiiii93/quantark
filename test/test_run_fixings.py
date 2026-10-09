"""Run fixings (spec docs/superpowers/specs/2026-10-09-run-fixings-design.md): product protocol and recorder."""
import copy
import warnings
from datetime import timedelta

import pytest

import run_fixings_fixtures as F
from quantark.asset.equity.product.option import RangeAccrualConfig, RangeAccrualObservationRecord, RangeAccrualOption
from quantark.asset.equity.product.option.asian_option import AsianObservationRecord, AsianOption
from quantark.util.exceptions import ValidationError
from quantark.util.fixings import MissingFixingWarning, snap_time
from quantark.util.numerical import is_close

DAY = 1.0 / 365.0


def _aged(product, days):
    for _ in range(days):
        product = product.with_time_elapsed(DAY)
    return product


# ---------------------------------------------------------------- snap_time
def test_snap_time():
    assert snap_time(4 / 365 - 4 * (1 / 365)) == 0.0 and snap_time(-DAY) == -DAY


# ---------------------------------------------------------------- equity protocol
def test_has_run_fixings():
    assert F.ra_dated().has_run_fixings and F.ra_numeric().has_run_fixings
    assert F.asian_dated().has_run_fixings and F.asian_numeric().has_run_fixings
    legacy = AsianOption(strike=100.0, maturity=1.0, num_observations=12)
    assert legacy.has_run_fixings is False


@pytest.mark.parametrize("kind", ["ra", "asian"])
def test_dated_contract_does_not_age(kind):
    prod = F.MAKERS[(kind, "dated")]()
    assert prod.with_time_elapsed(3 * DAY) is prod


@pytest.mark.parametrize("kind", ["ra", "asian"])
def test_numeric_ageing_is_a_copy_that_records_nothing(kind):
    prod = F.MAKERS[(kind, "numeric")]()
    before = copy.deepcopy(prod)
    aged = prod.with_time_elapsed(5 * DAY)
    assert aged is not prod and type(aged) is type(prod)
    assert prod.observation_records == before.observation_records and prod.maturity == before.maturity
    assert aged.maturity == 1.0 - 5 * DAY
    assert [r.observation_time for r in aged.observation_records] == [
        snap_time(r.observation_time - 5 * DAY) for r in prod.observation_records]
    outcome = "observed_in_range" if kind == "ra" else "observed_price"
    assert all(getattr(r, outcome) is None for r in aged.observation_records)
    assert prod.with_time_elapsed(0.0) is prod


def test_range_accrual_ageing_banks_the_elapsed_time():
    aged = F.ra_numeric().with_time_elapsed(5 * DAY)
    assert aged.accrued_offset == 5 * DAY
    assert aged.get_year_fraction(F.eq_env(5)) == pytest.approx(1.0, abs=1e-15)


@pytest.mark.parametrize("kind", ["ra", "asian"])
def test_a_fixing_lands_on_exactly_zero(kind):
    aged = _aged(F.MAKERS[(kind, "numeric")](), F.FIX_DAY)
    assert aged.observation_records[0].observation_time == 0.0           # not an ulp either side
    assert aged.open_fixings(F.eq_env(F.FIX_DAY))[0] == (0.0, "#0 t=0")
    dated = F.MAKERS[(kind, "dated")]()                                  # a dated contract has no times to snap
    assert _aged(dated, F.FIX_DAY) is dated


def test_num_observations_schedule_is_written_out_before_the_maturity_moves():
    prod = RangeAccrualOption(initial_price=100.0, maturity=1.0, num_observations=4,
                              range_config=RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0,
                                                              accrual_rate=0.10))
    aged = prod.with_time_elapsed(0.1)
    assert aged.num_observations is None and aged.observation_times is None
    assert [r.observation_time for r in aged.observation_records] == [0.25 - 0.1, 0.5 - 0.1, 0.75 - 0.1, 1.0 - 0.1]


def test_legacy_asian_cannot_age():
    with pytest.raises(ValidationError, match="legacy"):
        AsianOption(strike=100.0, maturity=1.0, num_observations=12).with_time_elapsed(DAY)
    with pytest.raises(ValidationError, match="legacy"):                  # the dated form refuses too
        AsianOption(strike=100.0, exercise_date=F.START + timedelta(days=365), initial_date=F.START,
                    num_observations=12).with_time_elapsed(DAY)


@pytest.mark.parametrize("kind", ["ra", "asian"])
@pytest.mark.parametrize("form", ["dated", "numeric"])
def test_open_fixings_lists_unrecorded_records_with_time_and_label(kind, form):
    env = F.eq_env(F.FIX_DAY)
    prod = F.MAKERS[(kind, form)]({0: True} if kind == "ra" else {0: 105.0})
    got = prod.open_fixings(env)
    assert 0 not in got and sorted(got) == [1, 2, 3, 4]
    rec = prod.observation_records[1]
    label = "#1 2024-01-19" if form == "dated" else f"#1 t={rec.observation_time:.6g}"
    assert got[1] == (rec.resolve_time(env), label)
    if form == "numeric":
        assert got[1][0] == rec.observation_time


@pytest.mark.parametrize("kind", ["ra", "asian"])
@pytest.mark.parametrize("day", [3, 4, 5, 10, 24])
def test_numeric_twin_aged_prices_as_the_dated_contract(kind, day):
    """R6: a numeric twin aged day by day equals its dated twin (same hand-recorded fixings), relative 1e-12."""
    engine = F.ENGINES[kind]()
    outcomes = F.hand_outcomes(kind, day - 1)
    dated = F.MAKERS[(kind, "dated")](outcomes)
    numeric = _aged(F.MAKERS[(kind, "numeric")](outcomes), day)
    env = F.eq_env(day)
    assert is_close(F.quiet_price(engine, numeric, env), F.quiet_price(engine, dated, env), rel_tol=1e-12)


def test_ageing_to_maturity_snaps_and_prices_as_settled():
    """Review Focus 2: a numeric contract aged to exactly its maturity."""
    short = [d / 365.0 for d in (4, 10)]
    numeric = _aged(F.ra_numeric({0: True, 1: True}, times=short, maturity=10 / 365.0), 10)
    assert numeric.maturity == 0.0 and numeric.observation_records[-1].observation_time == 0.0
    # A dated contract cannot be priced ON its exercise date (the engine refuses), so the numeric contract aged to
    # its maturity is checked against the settled value: both fixings in range for the whole contract length.
    assert is_close(F.quiet_price(F.ENGINES["ra"](), numeric, F.eq_env(10)), 100.0 * 0.10, rel_tol=1e-12)


# ---------------------------------------------------------------- X4: only history warns
@pytest.mark.parametrize("kind", ["ra", "asian"])
@pytest.mark.parametrize("form", ["dated", "numeric"])
def test_a_t0_fill_is_silent_and_a_past_fill_warns(kind, form):
    prod = F.MAKERS[(kind, form)]()
    if form == "numeric":
        prod = _aged(prod, F.FIX_DAY)                                     # record #0 at t = 0
    engine = F.ENGINES[kind]()
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        engine.price(prod, F.eq_env(F.FIX_DAY))                          # t = 0: live, decided silently
    past = prod.with_time_elapsed(DAY) if form == "numeric" else prod
    with pytest.warns(MissingFixingWarning, match="1 past fixing") as rec:
        engine.price(past, F.eq_env(F.FIX_DAY + 1))
    assert "#0" in str(rec[0].message)
    assert prod.missing_fixing_fills(F.eq_env(F.FIX_DAY)) == {}          # w5b regression: t = 0 is not history


# ---------------------------------------------------------------- Task 2: Asian theta records what it steps over
def test_asian_theta_step_records_the_stepped_over_fixing_at_the_held_spot():
    from quantark.asset.equity.riskmeasures.greeks.theta_shift import shift_for_theta

    prod = F.asian_dated()
    env = F.eq_env(F.FIX_DAY, spot=107.0)
    bumped = env.valuation_date + timedelta(days=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        shift = shift_for_theta(prod, env, DAY, bumped)
        theta_price = F.ENGINES["asian"]().price(shift.product, shift.env)
    assert shift.product.observation_records[0].observed_price == 107.0
    assert prod.observation_records[0].observed_price is None                 # the booked contract is untouched
    env_t1 = F.eq_env(F.FIX_DAY + 1, spot=107.0)
    assert theta_price == F.quiet_price(F.ENGINES["asian"](), prod, env_t1)   # value-neutral: pricing filled 107 too


def test_asian_greeks_on_a_fixing_day_are_silent():
    from quantark.asset.equity.riskmeasures import GreeksCalculator

    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        GreeksCalculator().calculate(F.asian_dated(), F.eq_env(F.FIX_DAY), F.ENGINES["asian"]())


def test_asian_time_shift_records_only_the_stepped_over_dated_record():
    prod = F.asian_dated()
    env = F.eq_env(F.FIX_DAY, spot=107.0)
    prod.time_shift(DAY, env.valuation_date + timedelta(days=1), env)
    prices = [r.observed_price for r in prod.observation_records]
    assert prices[0] == 107.0 and all(p is None for p in prices[1:])


def test_asian_numeric_theta_does_not_move_record_times_so_records_nothing():
    # Pre-existing, out of scope (Task 2 brief): numeric record times are not moved by the theta bump, so no numeric
    # record is stepped over and none is recorded. Pinned so a change to it is deliberate.
    prod = F.asian_numeric()
    env = F.eq_env(F.FIX_DAY, spot=107.0)
    prod.time_shift(DAY, env.valuation_date + timedelta(days=1), env)
    assert all(r.observed_price is None for r in prod.observation_records)


# ---------------------------------------------------------------- Task 3: FX protocol
FX_KINDS = ["domestic", "foreign", "quanto"]


@pytest.mark.parametrize("kind", FX_KINDS)
def test_fx_protocol_and_ageing(kind):
    dated, numeric = F.fx_ra(kind, "dated"), F.fx_ra(kind, "numeric")
    assert dated.has_run_fixings and dated.with_time_elapsed(DAY) is dated
    before = copy.deepcopy(numeric)
    aged = _aged(numeric, F.FIX_DAY)
    assert type(aged) is type(numeric) and numeric.observation_records == before.observation_records
    assert aged.observation_records[0].observation_time == 0.0
    assert is_close(aged.accrued_offset, F.FIX_DAY * DAY, rel_tol=1e-15)
    assert aged.open_fixings(F.fx_env(F.FIX_DAY))[0] == (0.0, "#0 t=0")
    assert dated.open_fixings(F.fx_env(F.FIX_DAY))[0] == (0.0, "#0 2024-01-05")


@pytest.mark.parametrize("kind", FX_KINDS)
def test_fx_ageing_moves_a_numeric_delivery(kind):
    prod = F.fx_ra(kind, "numeric")
    prod.delivery = 1.0 + 2 * DAY
    assert prod.with_time_elapsed(DAY).delivery == 1.0 + 2 * DAY - DAY


@pytest.mark.parametrize("kind", FX_KINDS)
@pytest.mark.parametrize("day", [3, 4, 5, 10, 24])
def test_fx_numeric_twin_aged_prices_as_the_dated_contract(kind, day):
    engine = F.FX_CLASSES[kind][1]()
    outcomes = F.fx_hand_outcomes(kind, day - 1)
    env = F.fx_env(day, quanto=(kind == "quanto"))
    dated = F.fx_ra(kind, "dated", outcomes)
    numeric = _aged(F.fx_ra(kind, "numeric", outcomes), day)
    assert is_close(F.quiet_price(engine, numeric, env), F.quiet_price(engine, dated, env), rel_tol=1e-12)


@pytest.mark.parametrize("kind", FX_KINDS)
@pytest.mark.parametrize("form", ["dated", "numeric"])
def test_fx_t0_fill_is_silent_and_a_past_fill_warns(kind, form):
    engine = F.FX_CLASSES[kind][1]()
    q = kind == "quanto"
    prod = F.fx_ra(kind, form)
    if form == "numeric":
        prod = _aged(prod, F.FIX_DAY)                                     # record #0 at t = 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        engine.price(prod, F.fx_env(F.FIX_DAY, quanto=q))                # t = 0: live, decided silently
    past = prod.with_time_elapsed(DAY) if form == "numeric" else prod
    with pytest.warns(MissingFixingWarning, match="1 past fixing"):
        engine.price(past, F.fx_env(F.FIX_DAY + 1, quanto=q))


# ---------------------------------------------------------------- Task 4: recorder
from quantark.util.run_fixings import (  # noqa: E402
    RecordedFixing, RunFixingRecorder, participates, run_step_dt, step_fixings,
)


def _rows(rec, pid):
    return [(f.index, f.outcome, f.spot, f.gap) for f in rec.recorded if f.position_id == pid]


def _drive(pf, rec, days, spot_of=F.spot_on):
    for k in days:
        F.move(pf, k, spot_of(k))
        rec.advance(pf)


def test_participates():
    assert participates(F.ra_dated()) and participates(F.asian_numeric()) and participates(F.fx_ra("quanto", "dated"))
    assert not participates(AsianOption(strike=100.0, maturity=1.0, num_observations=12))
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.util.enum import OptionType
    assert not participates(EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0))


def test_run_step_dt():
    env = F.eq_env(5)
    assert run_step_dt(F.START + timedelta(days=4), env) == 1 / 365.0
    assert run_step_dt(env.valuation_date, env) == 0.0
    with pytest.raises(ValidationError, match="back"):
        run_step_dt(F.START + timedelta(days=6), env)


def test_step_fixings_is_stateless_and_skips_run_fills():
    env = F.eq_env(5)
    fills = {}
    prod, new = step_fixings(F.ra_dated(), env, prev_date=F.START + timedelta(days=4), prev_spot=105.0, run_fills=fills)
    assert new == {0: True} and prod.observation_records[0].observed_in_range is True and fills == {}
    again, new2 = step_fixings(F.ra_dated(), env, prev_date=env.valuation_date, prev_spot=200.0, run_fills={0: True})
    assert new2 == {} and again.observation_records[0].observed_in_range is True


@pytest.mark.parametrize("kind", ["ra", "asian"])
def test_d2_the_clock_fixes_a_fixing_at_the_spot_of_the_day_it_was_live(kind):
    dated, numeric = F.MAKERS[(kind, "dated")](), F.MAKERS[(kind, "numeric")]()
    pf = F.eq_book(dated, numeric)
    pid_d, pid_n = list(pf.positions)
    rec = RunFixingRecorder()
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        rec.register(pf)
        _drive(pf, rec, range(0, 12))
    want_out = True if kind == "ra" else 105.0
    assert _rows(rec, pid_d) == [(0, want_out, 105.0, False)]
    assert _rows(rec, pid_d) == _rows(rec, pid_n)                         # X2 bitwise
    assert pf.positions[pid_d].product is not dated and dated.observation_records[0] == F.MAKERS[(kind, "dated")]().observation_records[0]
    day5 = next(d for d in rec.days if d.fixings)
    assert day5.date == F.START + timedelta(days=5)


def test_r4_a_gap_fixing_takes_the_previous_instants_spot():
    pf = F.eq_book(F.ra_dated(), F.ra_numeric(times=[4.5 / 365.0, 18 / 365.0, 120 / 365.0, 240 / 365.0, 1.0]))
    pid_d, pid_n = list(pf.positions)
    rec = RunFixingRecorder()
    rec.register(pf)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        _drive(pf, rec, [0, 1, 2, 3, 6], spot_of=lambda k: 100.0 + k)     # day 4 and 5 skipped
    assert _rows(rec, pid_d) == [(0, True, 103.0, True)]
    assert _rows(rec, pid_n) == [(0, True, 103.0, True)]


def test_r5_history_at_the_first_valuation_warns_once_per_position_and_the_first_day_is_live():
    dates = [F.START - timedelta(days=3), F.START] + F.fixing_dates()[1:]
    pf = F.eq_book(F.ra_dated(dates=dates, initial_date=F.START - timedelta(days=10)),
                   F.ra_dated(dates=dates, initial_date=F.START - timedelta(days=10)), env=F.eq_env(0, spot=100.0))
    rec = RunFixingRecorder()
    with pytest.warns(MissingFixingWarning) as caught:
        day0 = rec.register(pf)
    assert len([w for w in caught if issubclass(w.category, MissingFixingWarning)]) == 2 and day0.fixings == ()
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        F.move(pf, 0, 105.0)
        rec.advance(pf)                                                   # same date: refresh day 0's spot (Ruling 9)
        F.move(pf, 1, 120.0)
        rec.advance(pf)
    pid = list(pf.positions)[0]
    assert _rows(rec, pid) == [(1, True, 105.0, False)]                  # the START record, at day 0's market spot
    assert pf.positions[pid].product.observation_records[0].observed_in_range is True   # R5 fill at 100


def test_r8_a_time_of_day_instant_is_refused():
    pf = F.eq_book(F.ra_dated())
    rec = RunFixingRecorder()
    rec.register(pf)
    F.move(pf, 1, date=F.START + timedelta(days=1, hours=15))
    with pytest.raises(ValidationError, match="time of day"):
        rec.advance(pf)


def test_r8_r9_do_not_apply_without_a_participant():
    from quantark.asset.equity.product.option import EuropeanVanillaOption
    from quantark.asset.equity.engine.analytical import BlackScholesEngine
    from quantark.util.enum import OptionType
    pf = F.eq_book()
    pf.add_position(product=EuropeanVanillaOption(strike=100.0, option_type=OptionType.CALL, maturity=1.0),
                    quantity=1.0, entry_price=0.0, underlying=F.UNDERLYING, engine=BlackScholesEngine())
    rec = RunFixingRecorder()
    rec.register(pf)
    F.move(pf, 1, date=F.START + timedelta(days=1, hours=15))
    assert rec.advance(pf).fixings == ()


def test_r9_the_clock_cannot_go_back():
    pf = F.eq_book(F.ra_dated())
    rec = RunFixingRecorder()
    rec.register(pf)
    _drive(pf, rec, [2])
    F.move(pf, 1)
    with pytest.raises(ValidationError, match="back"):
        rec.advance(pf)


def test_non_participants_are_untouched():
    legacy = AsianOption(strike=100.0, maturity=1.0, num_observations=12)
    pf = F.eq_book(legacy)
    pid = list(pf.positions)[0]
    rec = RunFixingRecorder()
    rec.register(pf)
    _drive(pf, rec, range(1, 8))
    assert pf.positions[pid].product is legacy and rec.recorded == ()


def test_each_underlying_records_at_its_own_previous_spot():
    """Review Focus 1."""
    pf = F.eq_book(F.ra_dated(), underlying="IDX_A")
    pf.pricing_environments["IDX_B"] = F.eq_env(0)
    pf.add_position(product=F.ra_dated(), quantity=1.0, entry_price=0.0, underlying="IDX_B",
                    engine=F.ENGINES["ra"]())
    pid_a, pid_b = list(pf.positions)
    rec = RunFixingRecorder()
    rec.register(pf)
    for k in range(1, 6):
        F.move(pf, k, 105.0, underlying="IDX_A")
        F.move(pf, k, 120.0 if k == 4 else 105.0, underlying="IDX_B")
        rec.advance(pf)
    assert _rows(rec, pid_a) == [(0, True, 105.0, False)]
    assert _rows(rec, pid_b) == [(0, False, 120.0, False)]


def test_a_fixing_live_for_several_instants_takes_the_first_live_instants_spot():
    """Review Focus 4 / Ruling 4 (user-approved): on a business-day clock with weekend rows Friday's fixing is t = 0
    on Friday, Saturday and Sunday and is decided on Monday at FRIDAY's spot (a same-date refresh on Friday counts),
    dated and numeric twins identical."""
    from quantark.util.calendar import CalendarType, DayCountConvention, calculate_year_fraction, create_calendar
    cal = create_calendar(CalendarType.NONE, year_range=(2024, 2025))
    kw = dict(day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=252, calendar=cal)
    times = [calculate_year_fraction(F.START, d, DayCountConvention.BUSINESS_DAYS, 252, calendar=cal)
             for d in F.fixing_dates()]
    pf = F.eq_book(F.ra_dated(), F.ra_numeric(times=times), env=F.eq_env(0, spot=105.0, **kw))
    pid_d, pid_n = list(pf.positions)
    rec = RunFixingRecorder()
    rec.register(pf)
    spots = {1: 105.0, 2: 105.0, 3: 105.0, 4: 120.0, 5: 120.0, 6: 120.0, 7: 120.0}   # day 4 = Friday
    for k in range(1, 8):
        F.move(pf, k, spots[k])
        rec.advance(pf)
        if k == 4:                                          # same-date refresh: Friday's market is 105 after all
            F.move(pf, 4, 105.0)
            rec.advance(pf)
    assert _rows(rec, pid_d) == _rows(rec, pid_n) == [(0, True, 105.0, False)]   # Friday's spot, not Sunday's 120
    assert next(d for d in rec.days if d.fixings).date == F.START + timedelta(days=7)   # recorded on Monday


def test_a_position_first_seen_at_advance_is_registered_there():
    """Review Focus 3 (Ruling 8)."""
    pf = F.eq_book(F.ra_dated())
    rec = RunFixingRecorder()
    rec.register(pf)
    _drive(pf, rec, range(1, 6))
    late = F.ra_dated(dates=[F.START + timedelta(days=2)] + F.fixing_dates()[1:])
    pf.add_position(product=late, quantity=1.0, entry_price=0.0, underlying=F.UNDERLYING, engine=F.ENGINES["ra"]())
    pid = list(pf.positions)[-1]
    F.move(pf, 6)
    with pytest.warns(MissingFixingWarning):
        rec.advance(pf)                                                   # its first valuation: day 2 is history
    assert pf.positions[pid].product.observation_records[0].observed_in_range is True    # filled at 107 (in range)
    _drive(pf, rec, range(7, 20))
    assert _rows(rec, pid) == [(1, False, 119.0, False)]                 # day 18 at day 18's spot


def test_reverse_mode_is_decided_by_the_products_own_test():
    prod = F.ra_dated()
    prod.range_config = RangeAccrualConfig(upper_barrier=110.0, lower_barrier=90.0, accrual_rate=0.10, is_reverse=True)
    pf = F.eq_book(prod)
    rec = RunFixingRecorder()
    rec.register(pf)
    _drive(pf, rec, range(1, 6))
    assert _rows(rec, list(pf.positions)[0]) == [(0, False, 105.0, False)]


def test_business_day_clock_twins_age_identically():
    """X2 on BUSINESS_DAYS at recorder level (drift 3)."""
    from quantark.util.calendar import CalendarType, DayCountConvention, create_calendar
    cal = create_calendar(CalendarType.NONE, year_range=(2024, 2025))
    kw = dict(day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=252, calendar=cal)
    from quantark.util.calendar import calculate_year_fraction
    times = [calculate_year_fraction(F.START, d, DayCountConvention.BUSINESS_DAYS, 252, calendar=cal)
             for d in F.fixing_dates()]
    pf = F.eq_book(F.ra_dated(), F.ra_numeric(times=times), env=F.eq_env(0, **kw))
    pid_d, pid_n = list(pf.positions)
    rec = RunFixingRecorder()
    rec.register(pf)
    days = [k for k in range(0, 26) if (F.START + timedelta(days=k)).weekday() < 5]
    _drive(pf, rec, days)
    assert _rows(rec, pid_d) == _rows(rec, pid_n) and len(_rows(rec, pid_d)) == 2


def test_a_same_date_refresh_on_a_later_instant_keeps_the_first_live_instants_spot():
    """Ruling 4 / Ruling 9, negative side: a same-date refresh on SATURDAY refreshes only the records that went live
    on Saturday; Friday's record (live since Friday) is still recorded at Friday's spot, dated and numeric twins."""
    from quantark.util.calendar import CalendarType, DayCountConvention, calculate_year_fraction, create_calendar
    cal = create_calendar(CalendarType.NONE, year_range=(2024, 2025))
    kw = dict(day_count_convention=DayCountConvention.BUSINESS_DAYS, bus_days_in_year=252, calendar=cal)
    times = [calculate_year_fraction(F.START, d, DayCountConvention.BUSINESS_DAYS, 252, calendar=cal)
             for d in F.fixing_dates()]
    pf = F.eq_book(F.ra_dated(), F.ra_numeric(times=times), env=F.eq_env(0, spot=105.0, **kw))
    pid_d, pid_n = list(pf.positions)
    rec = RunFixingRecorder()
    rec.register(pf)
    spots = {1: 105.0, 2: 105.0, 3: 105.0, 4: 105.0, 5: 120.0, 6: 120.0, 7: 120.0}   # day 4 = Friday
    for k in range(1, 8):
        F.move(pf, k, spots[k])
        rec.advance(pf)
        if k == 5:                                          # same-date refresh on Saturday, at a moved spot
            F.move(pf, 5, 130.0)
            rec.advance(pf)
    assert _rows(rec, pid_d) == _rows(rec, pid_n) == [(0, True, 105.0, False)]   # Friday's spot, not Saturday's
    assert next(d for d in rec.days if d.fixings).date == F.START + timedelta(days=7)


def _stateless_twin(kind, form):
    days = (0, 1, 2, 3, 18, 120, 240, 365)
    if kind == "ra":
        if form == "dated":
            return F.ra_dated(dates=[F.START + timedelta(days=d) for d in days])
        return F.ra_numeric(times=[d / 365.0 for d in days]).with_time_elapsed(5 * DAY)
    if form == "dated":
        recs = [AsianObservationRecord(observation_date=F.START + timedelta(days=d)) for d in days]
        extra = dict(exercise_date=F.START + timedelta(days=365))
    else:
        recs = [AsianObservationRecord(observation_time=d / 365.0) for d in days]
        extra = dict(maturity=1.0)
    prod = AsianOption(strike=100.0, option_type=F.OptionType.CALL, asian_strike_type=F.AsianStrikeType.FIXED,
                       averaging_type=F.AveragingType.ARITHMETIC, observation_records=recs, **extra)
    return prod if form == "dated" else prod.with_time_elapsed(5 * DAY)


@pytest.mark.parametrize("kind", ["ra", "asian"])
def test_step_fixings_decides_each_live_index_at_its_own_spot_and_a_gap_at_prev_spot(kind):
    """Ruling 4, stateless: two live indices at two different first-live spots, one gap at ``prev_spot``; an index in
    ``run_fills`` is not re-decided; ``run_fills`` and the product passed in are untouched; twins identical."""
    out = (lambda s: 90.0 <= s <= 110.0) if kind == "ra" else (lambda s: s)
    outcome = "observed_in_range" if kind == "ra" else "observed_price"
    env = F.eq_env(5)
    results = []
    for form in ("dated", "numeric"):
        prod = _stateless_twin(kind, form)
        before = copy.deepcopy(prod.observation_records)
        run_fills = {0: out(100.0)}
        recorded, new = step_fixings(prod, env, prev_date=F.START + timedelta(days=4), prev_spot=95.0,
                                     run_fills=run_fills, live_spots={1: 105.0, 2: 120.0})
        assert new == {1: out(105.0), 2: out(120.0), 3: out(95.0)}
        assert [getattr(r, outcome) for r in recorded.observation_records[:5]] == [
            out(100.0), out(105.0), out(120.0), out(95.0), None]
        assert run_fills == {0: out(100.0)} and prod.observation_records == before and recorded is not prod
        results.append((new, [getattr(r, outcome) for r in recorded.observation_records]))
    assert results[0] == results[1]                                       # X2 bitwise


# ---------------------------------------------------------------- Task 5: hooks for product-held state (spec §6)
class _Stub:
    """A participating product holding run state, like W4b's TARN: events appended when a fixing is recorded."""

    has_run_fixings = True

    def __init__(self, matured=False, events=(), recorded=()):
        self.matured, self.lifecycle_events, self.recorded = matured, list(events), dict(recorded)

    def matured_before(self, env):
        return self.matured

    def missing_fixing_fills(self, env):
        return {0: float(env.spot)} if 0 not in self.recorded and env.valuation_date > F.START + timedelta(days=4) else {}

    def open_fixings(self, env):
        return {} if 0 in self.recorded else {0: (0.0 if env.valuation_date == F.START + timedelta(days=4) else 1.0, "#0")}

    def with_fixings_recorded(self, fills):
        if not fills or 0 in self.recorded:
            return self
        return _Stub(self.matured, self.lifecycle_events + [("FIXING", fills[0])], {**self.recorded, **fills})

    def with_time_elapsed(self, dt):
        return self

    def with_missing_fixings_filled(self, env, *, warn=True):
        return self

    def pending_pv(self, env):
        return 2.0 * len(self.lifecycle_events)


class _DuckBook:
    """The duck-typed portfolio the recorder needs (positions, pricing_environments, remove_position)."""

    def __init__(self, **products):
        from types import SimpleNamespace
        self.pricing_environments = {F.UNDERLYING: F.eq_env(0)}
        self.positions = {pid: SimpleNamespace(product=p, quantity=q, underlying=F.UNDERLYING)
                          for pid, (p, q) in products.items()}

    def remove_position(self, pid):
        return self.positions.pop(pid, None)


def test_hooks_matured_events_pending_pv():
    from quantark.asset.equity.lifecycle.autocallable import MaturedBeforeRunWarning
    pf = _DuckBook(keep=(_Stub(), -3.0), gone=(_Stub(matured=True), 1.0))
    rec = RunFixingRecorder()
    with pytest.warns(MaturedBeforeRunWarning):
        day0 = rec.register(pf)
    assert "gone" not in pf.positions and day0.pending_pv == 0.0
    days = []
    for k in range(1, 7):
        F.move(pf, k)
        days.append(rec.advance(pf))
    assert days[4].product_events == (("keep", -3.0, ("FIXING", 105.0)),)  # the step to day 5
    assert days[4].pending_pv == -3.0 * 2.0 and days[5].product_events == ()
