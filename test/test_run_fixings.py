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


def test_a_fixing_lands_on_exactly_zero():
    aged = _aged(F.ra_numeric(), F.FIX_DAY)
    assert aged.observation_records[0].observation_time == 0.0           # not an ulp either side
    assert aged.open_fixings(F.eq_env(F.FIX_DAY))[0] == (0.0, "#0 t=0")


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


@pytest.mark.parametrize("kind", ["ra", "asian"])
def test_open_fixings_lists_unrecorded_records_with_time_and_label(kind):
    env = F.eq_env(F.FIX_DAY)
    dated = F.MAKERS[(kind, "dated")]({0: True} if kind == "ra" else {0: 105.0})
    got = dated.open_fixings(env)
    assert 0 not in got and sorted(got) == [1, 2, 3, 4]
    assert got[1] == (dated.observation_records[1].resolve_time(env), "#1 2024-01-19")


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
