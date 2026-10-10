"""FX TARN / TARF ageing inside runs (spec 2026-10-09-fx-tarn-lifecycle-design.md section 7, tests B1-B6, Q8) on the
W6 run-fixings recorder (spec 2026-10-09-run-fixings-design.md section 6; TARN plan re-plan 2026-10-10)."""
import copy
import dataclasses
import warnings
from datetime import datetime, timedelta

import pandas as pd
import pytest

from fx_target_redemption_fixtures import (
    FIX, FIX_DATES, INCEPTION, LAG2, PAIR, TARGET_CAL, VAL, _numeric_twin, book, dated_note, dated_tarf, drive,
    engine_for, env, following, mc, note, signed_yf, tarf, twin_from_terms,
)
from quantark.asset.equity.lifecycle.autocallable import MaturedBeforeRunWarning
from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.asset.fx.engine.mc import FxTargetRedemptionNoteMCEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxVanillaOption
from quantark.asset.fx.product.option.target_redemption_state import (
    PendingPayment, TargetRedemptionEvent, TargetRedemptionEventType, TargetRedemptionState, TerminalBookingWarning,
    booked_cash,
)
from quantark.backtest import FXBacktestConfig, FXBacktestEngine
from quantark.backtest.strategy import FXDeltaNeutralStrategy
from quantark.backtest.transaction_costs import ZeroCostModel
from quantark.dynamicscenario import FXDynamicScenarioConfig, FXDynamicScenarioEngine
from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange
from quantark.param import SpotQuote
from quantark.stresstest.stress.stress_types import StressType
from quantark.util.calendar import DayCountConvention
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError
from quantark.util.fixings import MissingFixingWarning
from quantark.util.numerical import is_close
from quantark.util.run_fixings import RunFixingRecorder

FIXING, PAYMENT = TargetRedemptionEventType.FIXING, TargetRedemptionEventType.PAYMENT
BUS = dict(dc=DayCountConvention.BUSINESS_DAYS, calendar=TARGET_CAL)


def _rows(day, pid=None):
    return [(f.index, f.outcome, f.spot, f.gap) for f in day.fixings if pid is None or f.position_id == pid]


def _registered(*products, **kw):
    pf, ids = book(*products, **kw)
    rec = RunFixingRecorder()
    rec.register(pf)
    return pf, ids, rec


# ================================================================ Task 9: the recorder drives TARN / TARF


def test_register_fills_history_once_and_books_nothing():
    prod = note(fix=[-10 / 365, -5 / 365] + FIX[1:], accrual_start=-40 / 365, target=50.0)
    expected = prod.full_coupon(0) + prod.full_coupon(1)
    pf, (pid,) = book(prod)
    rec = RunFixingRecorder()
    with pytest.warns(MissingFixingWarning) as caught:
        day0 = rec.register(pf)
    msgs = [str(w.message) for w in caught if issubclass(w.category, MissingFixingWarning)]
    assert len(msgs) == 1 and "#0" in msgs[0] and "#1" in msgs[0]                 # W6 R5: once per position
    held = pf.positions[pid].product
    assert held.state.accumulated == expected
    assert [x.event_type for x in held.lifecycle_events] == [FIXING, PAYMENT, FIXING, PAYMENT]
    assert day0.product_events == () and day0.fixings == ()                       # W6 Ruling 10: nothing reported
    assert rec.advance(pf).product_events == ()                                   # same date (Ruling 9): no cash
    assert prod.state == INCEPTION and prod.lifecycle_events == ()               # the booked product is untouched


@pytest.mark.parametrize("form", ["dated", "numeric"])
def test_a_fixing_on_the_origin_date_is_live_and_records_the_next_day_silently(form):
    """Run-fixings spec 6 (W6 D1): not banked at registration; a same-date advance refreshes its spot (Ruling 9); it
    is recorded on the first later instant at that refreshed spot, without a warning."""
    if form == "dated":
        prod = dated_note(fixing_dates=[VAL] + FIX_DATES[1:], pay_dates=[VAL + timedelta(days=2)] + LAG2[1:],
                          initial_date=VAL - timedelta(days=30), target=50.0)
    else:
        prod = note(fix=[0.0] + FIX[1:], pay_times=[2 / 365] + [t + 2 / 365 for t in FIX[1:]],
                    accrual_start=-30 / 365, target=50.0)
    pf, (pid,) = book(prod)
    rec = RunFixingRecorder()
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        rec.register(pf)
        assert pf.positions[pid].product.state.accumulated == 0.0
        drive(pf, rec, VAL, 1.05)
        day1 = drive(pf, rec, VAL + timedelta(days=1), 1.30)
    assert _rows(day1) == [(0, 1.05, 1.05, False)]
    assert [(ev.event_type, ev.fixing, ev.amount) for _, _, ev in day1.product_events] == [(FIXING, 1.05, 0.0)]


@pytest.mark.parametrize("make", [
    lambda: note(fix=[-60 / 365, -30 / 365], accrual_start=-90 / 365),
    lambda: dated_note(fixing_dates=[VAL - timedelta(days=60), VAL - timedelta(days=30)],
                       initial_date=VAL - timedelta(days=90)),
])
def test_matured_before_the_run_is_removed_with_a_warning(make):
    pf, (pid,) = book(make())
    rec = RunFixingRecorder()
    with pytest.warns(MaturedBeforeRunWarning):
        day0 = rec.register(pf)
    assert pid not in pf.positions and day0.pending_pv == 0.0
    day1 = drive(pf, rec, VAL + timedelta(days=1), 1.20)
    assert (day1.product_events, day1.fixings, day1.pending_pv) == ((), (), 0.0)


def test_a_terminal_booking_is_kept_at_its_pending_pv_and_paid_after_its_day():
    with pytest.warns(TerminalBookingWarning):
        owed = note(fix=[], accrual_factors=[], target=1.5,
                    state=TargetRedemptionState(1.5, pending=(PendingPayment(100.5, pay_time=10 / 365),)))
    pf, (pid,) = book(owed)
    rec = RunFixingRecorder()
    day0 = rec.register(pf)
    e = pf.pricing_environments["EURUSD"]
    assert pid in pf.positions and day0.pending_pv == 100.5 * e.get_domestic_df(10 / 365)
    days = {k: drive(pf, rec, VAL + timedelta(days=k), 1.20) for k in range(1, 13)}
    paid = {k: [ev.amount for _, _, ev in d.product_events if ev.event_type is PAYMENT] for k, d in days.items()}
    assert days[10].pending_pv == 100.5                                           # due today: in the value (8a)
    assert paid[11] == [100.5] and sum(1 for p in paid.values() if p) == 1       # booked once its pay day is past


def test_the_run_clock_cannot_go_back():
    pf, _, rec = _registered(note(target=3.0))
    drive(pf, rec, VAL + timedelta(days=2), 1.20)
    with pytest.raises(ValidationError, match="cannot go back"):                  # W6 R9
        drive(pf, rec, VAL + timedelta(days=1), 1.20)


def test_a_time_of_day_instant_is_refused():
    pf, _, rec = _registered(dated_note(target=3.0))
    with pytest.raises(ValidationError, match="time of day"):                     # W6 R8
        drive(pf, rec, VAL + timedelta(days=1, hours=15), 1.20)


def test_other_products_are_untouched():
    vanilla = FxVanillaOption(strike=1.2, option_type=OptionType.CALL, maturity=1.0, notional_foreign=1.0,
                              currency_pair=PAIR)
    pf, _ = book(note(target=3.0))
    pid = pf.add_position(product=vanilla, quantity=1.0, entry_price=0.0, underlying="EURUSD",
                          engine=GarmanKohlhagenEngine()).position_id
    rec = RunFixingRecorder()
    rec.register(pf)
    drive(pf, rec, VAL + timedelta(days=40), 1.20)
    assert pf.positions[pid].product is vanilla and vanilla.maturity == 1.0


def test_a_calendar_gap_records_at_the_previous_instants_spot_without_a_warning():
    """W6 D5: the k/12 fixings #0 and #1 fall between two path instants: recorded at the PREVIOUS instant's spot
    (1.20, coupons), not the new day's (1.05, below the strike), gap=True, no warning; banked in order."""
    prod = note(target=50.0)
    c = [prod.full_coupon(0), prod.full_coupon(1)]
    pf, _, rec = _registered(prod)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        day = drive(pf, rec, VAL + timedelta(days=70), 1.05)
    assert _rows(day) == [(0, 1.20, 1.20, True), (1, 1.20, 1.20, True)]
    kinds = [(ev.event_type, ev.amount) for _, _, ev in day.product_events]
    assert kinds == [(FIXING, c[0]), (PAYMENT, c[0]), (FIXING, c[1]), (PAYMENT, c[1])]


def test_a_short_position_carries_its_signed_quantity():
    prod = tarf(gearing=2.0, target=0.2, pay_times=[t + 0.02 for t in FIX])
    pf, (pid,) = book(prod, quantities=[-2.0])
    rec = RunFixingRecorder()
    rec.register(pf)
    drive(pf, rec, VAL + timedelta(days=30), 1.10)                               # fixing #0 (30.4 days) still ahead
    fixing_day = drive(pf, rec, VAL + timedelta(days=31), 1.10)                  # recorded at day 30's 1.10 (gap)
    (cash,) = pf.positions[pid].product.state.pending
    assert cash.amount < 0.0                                                     # below the 1.15 strike: a liability
    assert [(q, ev.event_type) for _, q, ev in fixing_day.product_events] == [(-2.0, FIXING)]
    e = pf.pricing_environments["EURUSD"]
    assert fixing_day.pending_pv == -2.0 * (cash.amount * e.get_domestic_df(cash.pay_time))
    pay_day = drive(pf, rec, VAL + timedelta(days=38), 1.10)
    assert [(q, ev.event_type, ev.amount) for _, q, ev in pay_day.product_events] == [(-2.0, PAYMENT, cash.amount)]


def test_two_pairs_record_at_their_own_previous_spots():
    jpy_tarf = tarf(strike=145.0, currency_pair=CurrencyPair("USD", "JPY"), target=1e9)
    pf, (pid_eur,) = book(note(target=50.0))
    pf.pricing_environments["USDJPY"] = env(spot=150.0)
    pid_jpy = pf.add_position(product=jpy_tarf, quantity=1.0, entry_price=0.0, underlying="USDJPY",
                              engine=engine_for(jpy_tarf, n=2_000)).position_id
    rec = RunFixingRecorder()
    rec.register(pf)

    def move(days, eur, jpy):
        for pair, spot in (("EURUSD", eur), ("USDJPY", jpy)):
            e = pf.pricing_environments[pair]
            pf.pricing_environments[pair] = dataclasses.replace(e, valuation_date=VAL + timedelta(days=days),
                                                                spot_quote=SpotQuote(spot=spot))
        return rec.advance(pf)

    move(30, 1.05, 151.0)
    day = move(31, 1.30, 140.0)                                                  # both fixings #0 pass in the gap
    by_pid = {pid: ev for pid, _, ev in day.product_events if ev.event_type is FIXING}
    assert by_pid[pid_eur].fixing == 1.05 and by_pid[pid_eur].amount == 0.0      # below the TARN strike
    assert by_pid[pid_jpy].fixing == 151.0 and by_pid[pid_jpy].amount == 1.0 * max(151.0 - 145.0, 0.0)


def test_consecutive_daily_fixings_keep_their_own_live_spots_across_banking():
    """Review Focus 6 / Ruling R14: on day 6 fixing #0 banks (the numeric twin drops it) while #1 goes live; #1 must
    keep ITS first-live spot and gap=False, as in the dated contract."""
    dates = [VAL + timedelta(days=k) for k in (5, 6, 7)]
    dated = dated_note(fixing_dates=dates, pay_dates=[d + timedelta(days=3) for d in dates], target=50.0)
    pf, (pid_d, pid_n), rec = _registered(dated, _numeric_twin(dated, env()))
    spot = lambda k: 1.00 + 0.02 * k                                              # noqa: E731
    for k in range(1, 10):
        drive(pf, rec, VAL + timedelta(days=k), spot(k))
    rows = {pid: [(f.index, f.outcome, f.spot, f.gap) for f in rec.recorded if f.position_id == pid]
            for pid in (pid_d, pid_n)}
    assert rows[pid_d] == rows[pid_n] == [(i, spot(5 + i), spot(5 + i), False) for i in range(3)]


def test_lifecycle_events_are_append_only_in_a_run():
    pf, (pid,), rec = _registered(dated_note(target=1.5, pay_dates=LAG2))
    before = pf.positions[pid].product.lifecycle_events
    for k in range(1, 70):
        drive(pf, rec, VAL + timedelta(days=k), 1.20)
        now = pf.positions[pid].product.lifecycle_events
        assert now[:len(before)] == before
        before = now
    assert [ev.event_type for ev in before] == [FIXING, PAYMENT, FIXING, PAYMENT, PAYMENT]


def test_zero_lag_cash_of_a_live_fixing_stays_in_the_value_until_recorded():
    """OQ-1 (b), user 2026-10-10 (Task 8a): the fixing dated day 30 (pay date = fixing date) is live on day 30 and its
    cash is due that day, so day 30 holds it as pending PV (in the value, nothing booked); on day 31 the fixing is
    recorded, the cash is past and leaves the value as a PAYMENT - no dip."""
    pf, _, rec = _registered(dated_note(target=3.0))
    days = {k: drive(pf, rec, VAL + timedelta(days=k), 1.20) for k in range(1, 32)}
    fixing, payment = [ev for _, _, ev in days[31].product_events]
    assert (fixing.event_type, payment.event_type) == (FIXING, PAYMENT) and payment.amount == fixing.amount
    assert days[30].product_events == () and days[30].pending_pv == fixing.amount
    assert days[31].pending_pv == 0.0


def test_a_weekend_fixing_on_a_business_day_clock_is_live_on_friday():
    """OQ-2 (b), user 2026-10-10 (Task 8b): under BUSINESS_DAYS a Saturday fixing is t = 0 on the Friday, live in both
    forms (W6 Ruling 2); the run does not stop; on Monday it is recorded at Friday's spot, gap=False, twins identical."""
    friday, saturday, monday = datetime(2026, 8, 14), datetime(2026, 8, 15), datetime(2026, 8, 17)
    dated = dated_note(fixing_dates=[datetime(2026, 7, 15), saturday, datetime(2026, 9, 15)], target=50.0)
    pf, (id_d, id_n), rec = _registered(dated, _numeric_twin(dated, env(**BUS)), fx_env=env(**BUS))
    drive(pf, rec, friday - timedelta(days=1), 1.20)
    fri = drive(pf, rec, friday, 1.21)
    e = pf.pricing_environments["EURUSD"]
    assert pf.positions[id_d].product.open_fixings(e)[1][0] == pf.positions[id_n].product.open_fixings(e)[1][0] == 0.0
    assert fri.pending_pv > 0.0                                   # the live fixing's cash is due "today" (Task 8a)
    mon = drive(pf, rec, monday, 1.30)
    assert [r for r in _rows(mon, id_d) if r[0] == 1] == [r for r in _rows(mon, id_n) if r[0] == 1] == [
        (1, 1.21, 1.21, False)]
    assert [ev.event_type for p, _, ev in mon.product_events if p == id_d] == [FIXING, PAYMENT]


def test_oq3_friday_and_saturday_fixings_in_a_run_are_recorded_once_at_fridays_spot():
    """OQ-3 ruling in a run (BUSINESS_DAYS, weekday instants): the Friday fixing and the Saturday fixing are both live
    on the Friday (W6 Ruling 4); on Monday both are recorded at Friday's spot, gap=False, banked by identity with
    their cash; nothing is banked again later. (Pricing either form before the Friday: Task 8c.)"""
    friday, saturday, monday = datetime(2026, 8, 14), datetime(2026, 8, 15), datetime(2026, 8, 17)
    dated = dated_note(fixing_dates=[datetime(2026, 7, 15), friday, saturday, datetime(2026, 9, 15)], target=50.0)
    pf, (pid,), rec = _registered(dated, fx_env=env(**BUS))
    drive(pf, rec, friday - timedelta(days=1), 1.20)
    fri = drive(pf, rec, friday, 1.21)
    assert fri.product_events == () and fri.pending_pv > 0.0                     # both live; their cash due today
    mon = drive(pf, rec, monday, 1.30)
    assert [r for r in _rows(mon) if r[0] in (1, 2)] == [(1, 1.21, 1.21, False), (2, 1.21, 1.21, False)]
    assert [ev.event_type for _, _, ev in mon.product_events] == [FIXING, PAYMENT, FIXING, PAYMENT]
    held = pf.positions[pid].product
    assert (held.fixings_banked, held.state.as_of) == (3, monday)
    tue = drive(pf, rec, monday + timedelta(days=1), 1.30)
    assert tue.product_events == () and pf.positions[pid].product.fixings_banked == 3


def test_periods_sharing_a_fixing_date_are_recorded_and_banked_once_in_a_run():
    """OQ-4 (b), Task 8c: two periods on one fixing date are both live on it, recorded the next day at its spot (one
    RecordedFixing each, gap=False) and banked together once; each keeps its own pay date (the second pays two days
    later)."""
    dates = [FIX_DATES[0]] + FIX_DATES
    dated = dated_note(fixing_dates=dates, pay_dates=[FIX_DATES[0], LAG2[0]] + FIX_DATES[1:],
                       accrual_factors=[1 / 12, 1 / 24] + [1 / 12] * 11, target=50.0)
    pf, (pid,), rec = _registered(dated)
    days = {k: drive(pf, rec, VAL + timedelta(days=k), 1.20 + 0.001 * k) for k in range(1, 34)}
    spot30 = 1.20 + 0.001 * 30                                                   # FIX_DATES[0] is day 30
    assert _rows(days[31]) == [(0, spot30, spot30, False), (1, spot30, spot30, False)]
    assert [ev.event_type for _, _, ev in days[31].product_events] == [FIXING, PAYMENT, FIXING]
    assert pf.positions[pid].product.fixings_banked == 2
    assert days[32].product_events == ()                                         # nothing banked again
    assert [ev.event_type for _, _, ev in days[33].product_events] == [PAYMENT]  # the second period's own pay date


# ================================================================ Task 9 fix round 1: live entries are recorded first

_THU, _FRI, _SAT, _SUN, _MON, _TUE = (datetime(2026, 8, d) for d in (13, 14, 15, 16, 17, 18))


def _weekend_rows_run(fri_spot, weekend_spot, **kw):
    """BUSINESS_DAYS clock (TARGET) with weekend path rows; fixings Jul 15, Fri Aug 14, Sat Aug 15 (both live on the
    Friday, W6 Ruling 2), Sep 15; target 1.5. Booked in both forms (dated and its numeric twin)."""
    kw.setdefault("fixing_dates", [datetime(2026, 7, 15), _FRI, _SAT, datetime(2026, 9, 15)])
    dated = dated_note(target=1.5, **kw)
    pf, (id_d, id_n), rec = _registered(dated, _numeric_twin(dated, env(**BUS)), fx_env=env(**BUS))
    rows = [(datetime(2026, 7, 16), 1.20), (_THU, 1.20), (_FRI, fri_spot), (_SAT, weekend_spot),
            (_SUN, weekend_spot), (_MON, weekend_spot), (_TUE, weekend_spot)]
    days = {d: drive(pf, rec, d, s) for d, s in rows}
    return pf, (id_d, id_n), days


def _events(day, pid):
    return [(ev.event_type, ev.amount, ev.fixing, ev.redeemed) for p, _, ev in day.product_events if p == pid]


@pytest.mark.parametrize("fri, weekend, monday_rows, redeemed", [
    (1.05, 1.20, [(1, 1.05, 1.05, False), (2, 1.05, 1.05, False)], False),   # probe at 1.20 hit at #1: #2 was lost
    (1.20, 1.05, [(1, 1.20, 1.20, False)], True),                            # was KeyError: 2 (#2 dead at 1.20)
    (1.05, 1.05, [(1, 1.05, 1.05, False), (2, 1.05, 1.05, False)], False),   # control: weekend rows carry Friday's
    (1.20, 1.20, [(1, 1.20, 1.20, False)], True),                            # control
])
def test_weekend_rows_quoting_another_spot_record_the_live_fixings_at_their_first_live_spot(fri, weekend, monday_rows,
                                                                                             redeemed):
    """Review Task 9 (W6 R3 / Ruling 4): the Friday and Saturday fixings are recorded on Monday at FRIDAY's spot, in
    both forms, whatever the weekend rows quote; a target hit at the Friday fixing at Friday's spot kills Saturday's."""
    pf, (id_d, id_n), days = _weekend_rows_run(fri, weekend)
    assert _rows(days[_MON], id_d) == _rows(days[_MON], id_n) == monday_rows
    assert _rows(days[_TUE]) == []                                               # nothing is recorded again
    assert _events(days[_MON], id_d) == _events(days[_MON], id_n)
    for pid in (id_d, id_n):
        held = pf.positions[pid].product
        assert held.is_redeemed is redeemed and held.fixings_banked == (2 if redeemed else 3)
    if not redeemed:                                                              # below the strike: no coupon
        assert [x[1] for x in _events(days[_MON], id_d)] == [0.0, 0.0]


@pytest.mark.parametrize("weekend", [1.20, 1.05])                               # 1.05: control (Friday's spot)
def test_periods_sharing_a_friday_fixing_are_recorded_together_whatever_the_weekend_rows_quote(weekend):
    """Review Task 9: two periods on Friday Aug 14 (stated accrual) are both recorded on Monday at Friday's 1.05; a
    weekend row at 1.20 (a hit if the Friday fixing were probed there) must not record only the first (was
    ``ValidationError ... splits the periods fixing on 2026-08-14``)."""
    pf, (id_d, id_n), days = _weekend_rows_run(
        1.05, weekend, fixing_dates=[datetime(2026, 7, 15), _FRI, _FRI, datetime(2026, 9, 15)],
        accrual_factors=[1 / 12, 1 / 12, 1 / 24, 1 / 12])
    assert _rows(days[_MON], id_d) == _rows(days[_MON], id_n) == [(1, 1.05, 1.05, False), (2, 1.05, 1.05, False)]
    assert pf.positions[id_d].product.fixings_banked == pf.positions[id_n].product.fixings_banked == 3


@pytest.mark.parametrize("make", [lambda: note(target=3.0), lambda: dated_note(target=3.0)])
def test_product_replay_refuses_a_target_redemption_product(make):
    """Review Task 9 item 5: ``ProductReplay`` steps run fixings but never settles product-held history
    (``settle_history``): a TARN / TARF is refused there, not replayed wrongly."""
    from quantark.backtest.replay.product_replay import ProductReplay
    with pytest.raises(ValidationError, match="ProductReplay"):
        ProductReplay(product=make(), product_quantity=1.0, has_lifecycle=True, lifecycle=None, surface_engine=None,
                      event_stats_engine=None, engine_config=None, market_data=None, start_date=None,
                      underlying="EURUSD", actions_sink=[], event_prob_sink=[], daily_event_sink=[], surfaces_sink=[])


# ================================================================ Task 10: FX backtest


def _path(index, spots):
    return pd.DataFrame({"EURUSD_spot": spots, "EURUSD_vol": 0.10, "EURUSD_dom_rate": 0.05,
                         "EURUSD_for_rate": 0.03}, index=index)


def _flat_path(days, spot, start=VAL, freq="D"):
    return _path(pd.date_range(start, periods=days, freq=freq), spot)


def _run_backtest(pf, path):
    engine = FXBacktestEngine(FXBacktestConfig(
        portfolio=pf, market_path=path,
        strategy=FXDeltaNeutralStrategy(delta_threshold=1e12, rebalance_frequency="on_threshold"),
        transaction_cost_model=ZeroCostModel(), calculate_greeks=False))
    return engine, engine.run()


def _backtest(pf, path):
    return _run_backtest(pf, path)[1]


def _spy_advance(monkeypatch, keep=lambda day: True):
    """Capture ``(RunFixingDay, {position id: (product, market value)})`` after each recorder advance of the run."""
    seen = []
    original = RunFixingRecorder.advance

    def spy(self, portfolio):
        day = original(self, portfolio)
        if keep(day):
            e = portfolio.pricing_environments["EURUSD"]
            seen.append((day, {pid: (p.product, p.get_market_value(e)) for pid, p in portfolio.positions.items()}))
        return day

    monkeypatch.setattr(RunFixingRecorder, "advance", spy)
    return seen


LAG2D = [t + 2 / 365 for t in FIX]


def test_booked_cash_scales_by_the_signed_quantity_and_ignores_other_entries():
    pay = TargetRedemptionEvent(PAYMENT, -0.1, time=0.0)
    fix = TargetRedemptionEvent(FIXING, -0.1, fixing=1.10)
    assert booked_cash(()) == 0.0
    assert booked_cash((("a", -2.0, pay), ("a", -2.0, fix), ("b", 1.0, ("SETTLED", None)))) == 0.0 + -2.0 * -0.1


def test_b1_numeric_tarn_ages_in_the_backtest():
    c0 = note(target=3.0)
    pf, _ = book(c0)
    rows = _backtest(pf, _flat_path(92, 1.20)).rows
    mtm = [r["portfolio_value"] - r["paid_cash"] for r in rows]
    assert mtm[0] != mtm[91]
    tau = 91 / 365
    rebuilt = note(fix=[t - tau for t in FIX[2:]], accrual_factors=c0.accrual_factors[2:], target=3.0,
                   state=TargetRedemptionState(c0.full_coupon(0) + c0.full_coupon(1)))
    e91 = dataclasses.replace(env(), valuation_date=VAL + timedelta(days=91))
    assert is_close(mtm[91], FxTargetRedemptionNoteMCEngine(mc(n=2_000)).price(rebuilt, e91), rel_tol=1e-12)
    assert rows[91]["paid_cash"] == c0.full_coupon(0) + c0.full_coupon(1)        # gap fixings, flat path: 1.20


def test_b2_redeeming_path_books_cash_on_its_pay_days():
    c0 = note(target=1.5, pay_times=LAG2D)
    pf, _ = book(c0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", TerminalBookingWarning)
        rows = _backtest(pf, _flat_path(70, 1.20)).rows
    c_first = c0.full_coupon(0)
    c_second = min(c0.full_coupon(1), 1.5 - c_first)
    paid = [r["paid_cash"] for r in rows]
    assert paid[32] == 0.0 and paid[33] == c_first and paid[62] == c_first
    assert paid[63] == c_first + (c_second + 100.0)
    e = env()
    for d in (61, 62):  # terminal from the second fixing's recording day: value = sum DF x pending, no simulation
        t = LAG2D[1] - d / 365
        expected = 0.0 + c_second * e.get_domestic_df(t) + 100.0 * e.get_domestic_df(t)
        assert is_close(rows[d]["pending_pv"], expected, rel_tol=1e-12)
        assert is_close(rows[d]["portfolio_value"] - paid[d], rows[d]["pending_pv"], rel_tol=1e-12)
    for d in range(63, 70):
        assert rows[d]["pending_pv"] == 0.0 and rows[d]["portfolio_value"] == paid[d]


def test_b3_quiet_path_books_the_principal_and_negative_tarf_cash():
    tarn = note(target=3.0, pay_times=LAG2D)
    pf, _ = book(tarn)
    rows = _backtest(pf, _flat_path(370, 1.05)).rows
    paid = [r["paid_cash"] for r in rows]
    # the last fixing (t = 1.0, day 365) is live on day 365 and recorded on day 366 at 1.05; its principal is due on
    # day 367 (1 + 2/365 years: still in the value that day, Task 8a) and booked on day 368
    assert paid[366] == 0.0 and paid[367] == 0.0 and rows[367]["pending_pv"] == 100.0
    assert paid[368] == 100.0 and rows[-1]["portfolio_value"] == 100.0
    f = tarf(target=0.20, gearing=2.0, pay_times=LAG2D)
    pf, _ = book(f)
    rows = _backtest(pf, _flat_path(370, 1.05)).rows
    paid = [r["paid_cash"] for r in rows]
    steps = [b - a for a, b in zip([0.0] + paid[:-1], paid)]
    assert sum(1 for s in steps if s < 0.0) == 12 and all(s <= 0.0 for s in steps)
    diff = 1.0 * (1.05 - 1.15)
    cf = 0.0 - 2.0 * (1.0 * max(-diff, 0.0))
    expected = 0.0
    for _ in range(12):
        expected = expected + (0.0 + 1.0 * cf)
    assert paid[-1] == expected


def test_b4_twins_in_one_backtest(monkeypatch):
    dn = dated_note(target=1.5, pay_dates=LAG2)
    dt = dated_tarf(target=0.075, gearing=2.0, pay_dates=LAG2)
    nn, nt = _numeric_twin(dn, env()), _numeric_twin(dt, env())
    pf, (id_dn, id_nn, id_dt, id_nt) = book(dn, nn, dt, nt)
    seen = _spy_advance(monkeypatch)
    _backtest(pf, _flat_path(100, 1.20))
    assert any(day.product_events for day, _ in seen)
    for day, values in seen:
        for a, b in ((id_dn, id_nn), (id_dt, id_nt)):
            ev = {pid: [(x.event_type, x.amount, x.fixing, x.redeemed) for p, _q, x in day.product_events if p == pid]
                  for pid in (a, b)}
            assert ev[a] == ev[b]
            assert _rows(day, a) == _rows(day, b)
            assert booked_cash([x for x in day.product_events if x[0] == a]) == booked_cash(
                [x for x in day.product_events if x[0] == b])
            assert is_close(values[a][1], values[b][1], rel_tol=1e-12)


def test_b5_backtest_registration_rules():
    early = note(fix=[-10 / 365] + FIX[1:], accrual_start=-40 / 365, target=50.0)
    pf, _ = book(early)
    with pytest.warns(MissingFixingWarning) as rec:
        rows = _backtest(pf, _flat_path(5, 1.20)).rows
    assert len([w for w in rec if issubclass(w.category, MissingFixingWarning)]) == 1
    assert all(r["paid_cash"] == 0.0 for r in rows)                             # paid before the baseline
    gone = note(fix=[-60 / 365, -30 / 365], accrual_start=-90 / 365)
    pf, _ = book(gone)
    with pytest.warns(MaturedBeforeRunWarning):
        rows = _backtest(pf, _flat_path(5, 1.20)).rows
    assert all(r["portfolio_value"] == 0.0 and r["paid_cash"] == 0.0 for r in rows)


def test_b5_backwards_path_raises():
    pf, _ = book(note(target=3.0))
    idx = pd.DatetimeIndex([VAL, VAL + timedelta(days=2), VAL + timedelta(days=1)])
    with pytest.raises(ValidationError, match="cannot go back"):                  # W6 R9
        _backtest(pf, _path(idx, 1.20))


def test_business_day_path_records_a_saturday_fixing_at_fridays_spot():
    """Review Focus 2 (W6 D5): a Saturday fixing on a weekday path under a CALENDAR_DAYS environment is never live
    (t = 1/365 on Friday); it is recorded on Monday at FRIDAY's spot, gap=True, no warning, in both forms."""
    saturday = datetime(2026, 8, 15)
    dn = dated_note(fixing_dates=[datetime(2026, 7, 15), saturday, datetime(2026, 9, 15)], target=50.0)
    nn = _numeric_twin(dn, env())
    pf, (id_d, id_n) = book(dn, nn)
    index = pd.bdate_range(VAL, periods=70)
    spots = [1.20 + 0.001 * k for k in range(len(index))]
    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFixingWarning)
        engine, _ = _run_backtest(pf, _path(index, spots))
    friday_spot = spots[list(index).index(pd.Timestamp(2026, 8, 14))]
    monday = pd.Timestamp(following(saturday))
    got = [(day.date, f.position_id, f.outcome, f.spot, f.gap) for day in engine.run_fixings.days
           for f in day.fixings if f.index == 1]
    assert got == [(monday, id_d, friday_spot, friday_spot, True), (monday, id_n, friday_spot, friday_spot, True)]


def test_w8_twins_on_the_last_fixing_day_and_after_the_end(monkeypatch):
    """W8 (on main): a numeric participant aged to its maturity is exactly 0.0 and equals its dated twin. A TARN stays
    in the run after its end (Ruling R19): PV(pending) through its last pay date (the cash due that day included,
    Task 8a), then exactly 0.0."""
    dn = dated_note(target=3.0, pay_dates=LAG2)
    nn = _numeric_twin(dn, env())
    pf, (id_d, id_n) = book(dn, nn)
    last = pd.Timestamp(FIX_DATES[-1])                                            # day 365
    seen = _spy_advance(monkeypatch, keep=lambda day: day.date >= last - pd.Timedelta(days=1))
    rows = _backtest(pf, _flat_path(370, 1.05)).rows
    on = {day.date: values for day, values in seen}
    numeric_on_last, value_n = on[last][id_n]
    assert numeric_on_last.maturity == 0.0 and numeric_on_last.fixing_times == [0.0]
    assert is_close(on[last][id_d][1], value_n, rel_tol=1e-12)
    due = on[pd.Timestamp(LAG2[-1])]                                              # last pay date: cash due today
    assert due[id_d][1] == due[id_n][1] == 100.0
    after = on[pd.Timestamp(LAG2[-1]) + pd.Timedelta(days=1)]
    assert after[id_d][1] == after[id_n][1] == 0.0
    assert rows[-1]["portfolio_value"] == rows[-1]["paid_cash"] == 0.0 + 1.0 * 100.0 + 1.0 * 100.0


def test_zero_lag_fixing_day_value_keeps_the_coupon_no_dip():
    """OQ-1 (b), user 2026-10-10 (Task 8a): on its fixing day a zero-lag dated TARN keeps that day's coupon in its
    value (pending, due today); the run books it the next day, when the fixing is recorded. No dip."""
    dn = dated_note(target=3.0)                                                   # pay dates = fixing dates
    pf, _ = book(dn)
    rows = _backtest(pf, _flat_path(33, 1.20)).rows
    e30 = env(val=FIX_DATES[0])                                                   # day 30
    banked = copy.deepcopy(dn)
    banked.record_passed_fixings(e30)
    coupon = banked.state.pending[0].amount
    assert rows[30]["paid_cash"] == 0.0 and rows[30]["pending_pv"] == coupon
    assert rows[31]["paid_cash"] == coupon and rows[31]["pending_pv"] == 0.0
    assert is_close(rows[30]["portfolio_value"], engine_for(banked, n=2_000).price(banked, e30), rel_tol=1e-12)
    for d in (30, 31):                                     # flat market: carry only, never a coupon-sized step
        assert abs(rows[d]["portfolio_value"] - rows[d - 1]["portfolio_value"]) < 0.1 * coupon


def test_business_day_clock_weekend_fixing_twins_in_a_backtest(monkeypatch):
    """OQ-2 (b), user 2026-10-10 (Task 8b): on a BUSINESS_DAYS environment and a weekday path the Saturday fixing is
    live on the Friday in both forms (W6 Ruling 2), the run does not stop, Monday records it at Friday's spot with
    gap=False, and the twins agree every day."""
    saturday = datetime(2026, 8, 15)
    dn = dated_note(fixing_dates=[datetime(2026, 7, 15), saturday, datetime(2026, 9, 15)], target=50.0)
    nn = _numeric_twin(dn, env(**BUS))
    pf, (id_d, id_n) = book(dn, nn, fx_env=env(**BUS))
    seen = _spy_advance(monkeypatch)
    index = pd.bdate_range(VAL, periods=70)
    spots = [1.20 + 0.001 * k for k in range(len(index))]
    engine, _ = _run_backtest(pf, _path(index, spots))
    friday_spot = spots[list(index).index(pd.Timestamp(2026, 8, 14))]
    monday = pd.Timestamp(following(saturday))
    got = [(day.date, f.position_id, f.outcome, f.spot, f.gap) for day in engine.run_fixings.days
           for f in day.fixings if f.index == 1]
    assert got == [(monday, id_d, friday_spot, friday_spot, False), (monday, id_n, friday_spot, friday_spot, False)]
    for _day, values in seen:
        assert is_close(values[id_d][1], values[id_n][1], rel_tol=1e-12)


def test_b6_book_without_target_redemption_books_nothing():
    from quantark.portfolio.fx import FXPortfolio
    vanilla = FxVanillaOption(strike=1.2, option_type=OptionType.CALL, maturity=1.0, notional_foreign=1.0,
                              currency_pair=PAIR)
    pf = FXPortfolio(portfolio_name="b6", pricing_environments={"EURUSD": env()})
    pf.add_position(product=vanilla, quantity=1.0, entry_price=0.0, underlying="EURUSD",
                    engine=GarmanKohlhagenEngine())
    res = _backtest(pf, _flat_path(20, 1.20))
    assert all(r["paid_cash"] == 0.0 and r["pending_pv"] == 0.0 for r in res.rows)


# ================================================================ Task 11: FX dynamic scenario


def _held_path(days, spot, start=VAL):
    steps = [DayStep(day_index=i, changes=[ParameterChange("spot", StressType.VALUE, spot)]) for i in range(days)]
    return DayPath(name="held", steps=steps, start_date=start)


def _scenario(pf, path):
    return FXDynamicScenarioEngine(FXDynamicScenarioConfig(calculate_greeks=False)).run(pf, path)


def test_q8_undated_path_with_a_tarn_is_refused():
    pf, _ = book(note(target=3.0))
    with pytest.raises(ValidationError, match="DayPath.start_date"):
        _scenario(pf, _held_path(5, 1.20, start=None))


def test_start_date_before_environment_date_raises():
    pf, _ = book(note(target=3.0))
    with pytest.raises(ValidationError, match="cannot go back"):                  # W6 R9
        _scenario(pf, _held_path(5, 1.20, start=VAL - timedelta(days=1)))


def test_dynamic_tarn_redeems_and_books_cash():
    c0 = note(target=1.5, pay_times=LAG2D)
    pf, (pid,) = book(c0)
    res = _scenario(pf, _held_path(70, 1.20))
    days = res.day_results
    c_first = c0.full_coupon(0)
    c_second = min(c0.full_coupon(1), 1.5 - c_first)
    assert days[33].paid_cash == c_first and days[33].realized_cash == c_first
    assert days[63].paid_cash == c_first + (c_second + 100.0)
    fixings = [(i, s) for i, d in enumerate(days) for s in d.lifecycle_events]
    assert [i for i, _ in fixings] == [31, 61]                                    # k/12 fixings between days (D5)
    first, second = fixings[0][1], fixings[1][1]
    assert (first.event_type, first.spot, first.payoff, first.cashflow, first.terminates_position) == (
        "FIXING", 1.20, c_first, 0.0, False)  # pay lag: nothing paid in the recording step (Ruling R6)
    assert (second.payoff, second.terminates_position, second.position_id) == (c_second, True, pid)
    assert days[61].pending_receivable_pv > 0.0 and days[69].pending_receivable_pv == 0.0
    assert days[69].portfolio_value == days[69].paid_cash
    assert res.final_value == days[69].paid_cash


def test_dynamic_snapshot_cashflow_when_paid_in_the_recording_step():
    c0 = note(target=1.5)                                                         # pay time = fixing time
    pf, _ = book(c0, quantities=[3.0])
    days = _scenario(pf, _held_path(70, 1.20)).day_results
    c_first = c0.full_coupon(0)
    c_second = min(c0.full_coupon(1), 1.5 - c_first)
    (s1,), (s2,) = days[31].lifecycle_events, days[61].lifecycle_events
    assert s1.cashflow == 3.0 * c_first
    assert s2.cashflow == 3.0 * (c_second + 100.0)


def test_b4_twins_in_one_dynamic_scenario():
    """B4 on two books (Task 11 review): the dated book and its numeric-twin book give the same snapshots, value, paid
    cash and pending PV every day (one book could not separate the forms' cash)."""
    dn = dated_note(target=1.5, pay_dates=LAG2)
    dt = dated_tarf(target=0.075, gearing=2.0, pay_dates=LAG2)
    nn, nt = _numeric_twin(dn, env()), _numeric_twin(dt, env())
    pf_d, ids_d = book(dn, dt)
    pf_n, ids_n = book(nn, nt)
    days_d = _scenario(pf_d, _held_path(100, 1.20)).day_results
    days_n = _scenario(pf_n, _held_path(100, 1.20)).day_results
    assert any(d.lifecycle_events for d in days_d) and any(d.paid_cash != 0.0 for d in days_d)
    for d, n in zip(days_d, days_n):
        snaps = []
        for day, ids in ((d, ids_d), (n, ids_n)):
            snaps.append([(ids.index(s.position_id), s.payoff, s.cashflow, s.terminates_position, s.spot)
                          for s in day.lifecycle_events])
        assert snaps[0] == snaps[1]
        for field in ("portfolio_value", "paid_cash", "realized_cash", "pending_receivable_pv"):
            assert is_close(getattr(d, field), getattr(n, field), rel_tol=1e-12, abs_tol=1e-12), (d.day_index, field)


def test_business_day_clock_weekend_fixing_twins_in_a_dynamic_scenario():
    """OQ-2 (b), user 2026-10-10 (Task 8b) on a daily dated DayPath (weekend rows carry Friday's spot, user ruling)
    under BUSINESS_DAYS: the Saturday fixing is live Friday to Sunday and recorded on Monday in both forms; snapshots
    identical, and the run does not stop on the Friday."""
    saturday = datetime(2026, 8, 15)
    dn = dated_note(fixing_dates=[datetime(2026, 7, 15), saturday, datetime(2026, 9, 15)], target=50.0)
    nn = _numeric_twin(dn, env(**BUS))
    pf, (id_d, id_n) = book(dn, nn, fx_env=env(**BUS))
    days = _scenario(pf, _held_path(70, 1.20)).day_results
    snaps = {pid: [(d.date, s.payoff, s.cashflow, s.spot, s.terminates_position) for d in days
                   for s in d.lifecycle_events if s.position_id == pid] for pid in (id_d, id_n)}
    assert snaps[id_d] == snaps[id_n]
    assert [x[0] for x in snaps[id_d]] == [datetime(2026, 7, 16), datetime(2026, 8, 17)]


def test_b6_dynamic_book_without_target_redemption_books_nothing():
    from quantark.portfolio.fx import FXPortfolio
    pf = FXPortfolio(portfolio_name="b6", pricing_environments={"EURUSD": env()})
    pf.add_position(product=FxVanillaOption(strike=1.2, option_type=OptionType.CALL, maturity=1.0,
                                            notional_foreign=1.0, currency_pair=PAIR),
                    quantity=1.0, entry_price=0.0, underlying="EURUSD", engine=GarmanKohlhagenEngine())
    for start in (VAL, None):                                                    # Q8 refuses only a TARN / TARF book
        days = _scenario(pf, _held_path(10, 1.20, start=start)).day_results
        assert all(d.paid_cash == 0.0 and d.pending_receivable_pv == 0.0 and d.lifecycle_events == []
                   for d in days)


# ======================================================== Task 12: regressions from the Task 10 / 11 review probes

_GBP = CurrencyPair("GBP", "USD")


def _two_pair_run(eur_qty, gbp_qty):
    """A dated EURUSD TARN and a dated GBPUSD TARF (each held only when its quantity is given) on two different spot
    paths: the FX backtest's rows."""
    from quantark.portfolio.fx import FXPortfolio
    pf = FXPortfolio(portfolio_name="two_pairs", pricing_environments={"EURUSD": env(), "GBPUSD": env(spot=1.30)})
    for qty, pair, prod in ((eur_qty, "EURUSD", dated_note(target=3.0, pay_dates=LAG2)),
                            (gbp_qty, "GBPUSD", dated_tarf(currency_pair=_GBP, strike=1.29, gearing=2.0,
                                                           target=0.5, pay_dates=LAG2))):
        if qty is not None:
            pf.add_position(product=prod, quantity=qty, entry_price=0.0, underlying=pair,
                            engine=engine_for(prod, n=2_000))
    index = pd.date_range(VAL, periods=100, freq="D")
    path = pd.DataFrame({
        "EURUSD_spot": [1.18 + 0.0007 * k for k in range(100)],
        "GBPUSD_spot": [1.31 - 0.0005 * k + (0.02 if k % 7 == 3 else 0.0) for k in range(100)],
        "EURUSD_vol": 0.10, "EURUSD_dom_rate": 0.05, "EURUSD_for_rate": 0.03,
        "GBPUSD_vol": 0.10, "GBPUSD_dom_rate": 0.05, "GBPUSD_for_rate": 0.03,
    }, index=index)
    return _backtest(pf, path).rows


def test_short_tarn_and_long_tarf_on_two_pairs_are_the_quantity_weighted_sum_of_single_runs():
    """Task 10 review probe P1: a short EURUSD TARN (-2.5) and a long GBPUSD TARF (+3.0) on their own paths; every
    row's paid cash, value and pending PV equal -2.5 x the EUR-only run + 3.0 x the GBP-only run."""
    both = _two_pair_run(-2.5, 3.0)
    eur = _two_pair_run(1.0, None)
    gbp = _two_pair_run(None, 1.0)
    assert any(r["paid_cash"] > 0.0 for r in eur) and any(r["paid_cash"] < 0.0 for r in gbp)
    for b, e, g in zip(both, eur, gbp):
        for field in ("paid_cash", "portfolio_value", "pending_pv"):
            expected = -2.5 * e[field] + 3.0 * g[field]
            assert is_close(b[field], expected, rel_tol=1e-12, abs_tol=1e-12), (b["timestamp"], field)


def _weekend_carry_path(days=75):
    """Daily rows from VAL; weekday spots climb, Saturday and Sunday rows carry Friday's spot (user weekend ruling)."""
    index = pd.date_range(VAL, periods=days, freq="D")
    spots, last = [], None
    for k, ts in enumerate(index):
        if ts.weekday() < 5 or last is None:
            last = 1.20 + 0.001 * k
        spots.append(last)
    return _path(index, spots)


def test_business_day_backtest_weekend_rows_carry_friday_twins_and_saturday_cash_books_monday():
    """Task 10 review probe P2 on a BUSINESS_DAYS / TARGET environment: a TARN and a TARF fixing on Thursday
    2026-08-13 and paying on Saturday 08-15. The dated book equals its numeric-twin book on every row; each weekend
    row equals its Friday row; the Saturday cash stays in the value (due, zero business time) through Sunday and is
    booked on Monday."""
    thursday, saturday = datetime(2026, 8, 13), datetime(2026, 8, 15)
    fix = [datetime(2026, 7, 15), thursday, datetime(2026, 9, 15)]
    pay = [datetime(2026, 7, 17), saturday, datetime(2026, 9, 17)]
    dn = dated_note(fixing_dates=fix, pay_dates=pay, target=50.0)
    dt = dated_tarf(fixing_dates=fix, pay_dates=pay, target=50.0)
    e0 = env(**BUS)
    rows_d = _backtest(book(dn, dt, fx_env=e0)[0], _weekend_carry_path()).rows
    rows_n = _backtest(book(_numeric_twin(dn, e0), _numeric_twin(dt, e0), fx_env=e0)[0], _weekend_carry_path()).rows
    fields = ("portfolio_value", "paid_cash", "pending_pv")
    for d, n in zip(rows_d, rows_n):
        for field in fields:
            assert is_close(d[field], n[field], rel_tol=1e-12, abs_tol=1e-12), (d["timestamp"], field)
    for rows in (rows_d, rows_n):
        by_date = {r["timestamp"]: r for r in rows}
        weekends = [ts for ts in by_date if ts.weekday() >= 5 and ts - pd.Timedelta(days=ts.weekday() - 4) in by_date]
        assert weekends
        for ts in weekends:
            friday = by_date[ts - pd.Timedelta(days=ts.weekday() - 4)]
            assert all(by_date[ts][f] == friday[f] for f in fields), ts
        fri, sat, sun, mon = (by_date[pd.Timestamp(2026, 8, d)] for d in (14, 15, 16, 17))
        assert fri["paid_cash"] == sat["paid_cash"] == sun["paid_cash"] < mon["paid_cash"]
        assert sun["pending_pv"] > 0.0
        assert is_close(mon["paid_cash"] - sun["paid_cash"], sun["pending_pv"], rel_tol=1e-12)   # cash due: DF = 1


def _greeks_scenario(pf, greeks):
    return FXDynamicScenarioEngine(FXDynamicScenarioConfig(calculate_greeks=greeks)).run(pf, _held_path(40, 1.20))


def test_dynamic_scenario_with_greeks_equals_without_and_leaves_the_input_book_untouched():
    """Task 11 review probe P1: Greeks (bumped repricings of the aged products) change no value, cash, pending PV or
    snapshot, and the input book is never touched."""
    dn = dated_note(target=3.0, pay_dates=LAG2)
    nt = tarf(target=0.2, gearing=2.0, pay_times=LAG2D)
    pf, ids = book(dn, nt)
    before = {pid: (pos.product, copy.deepcopy(pos.product.state), pos.product.lifecycle_events)
              for pid, pos in pf.positions.items()}
    on = _greeks_scenario(pf, True).day_results
    off = _greeks_scenario(pf, False).day_results
    assert on[-1].greeks and not off[-1].greeks
    assert any(d.lifecycle_events for d in on) and any(d.paid_cash != 0.0 for d in on)
    for a, b in zip(on, off):
        assert (a.portfolio_value, a.paid_cash, a.realized_cash, a.pending_receivable_pv) == (
            b.portfolio_value, b.paid_cash, b.realized_cash, b.pending_receivable_pv)
        assert [dataclasses.astuple(s) for s in a.lifecycle_events] == [dataclasses.astuple(s) for s in
                                                                         b.lifecycle_events]
    for pid, (product, state, events) in before.items():
        held = pf.positions[pid].product
        assert held is product and held.state == state and held.lifecycle_events == events == ()
    assert pf.pricing_environments["EURUSD"].valuation_date == VAL


def test_dynamic_scenario_equals_the_backtest_bitwise_on_cash_and_pending_pv():
    """Task 11 review probe P3: the same book on the same dated path (the backtest fed the scenario's own spots) books
    the same paid cash and pending PV, bit for bit."""
    dn = dated_note(target=1.5, pay_dates=LAG2)
    nt = tarf(target=0.2, gearing=2.0, pay_times=LAG2D)
    spots = [1.20] + [1.20 + 0.002 * ((k % 9) - 4) for k in range(1, 70)]
    steps = [DayStep(day_index=i, changes=[ParameterChange("spot", StressType.VALUE, s)]) for i, s in enumerate(spots)]
    pf, _ = book(dn, nt, quantities=[2.0, -1.5])
    days = _scenario(pf, DayPath(name="moving", steps=steps, start_date=VAL)).day_results
    realized = [d.market_state.spot["EURUSD"] for d in days]
    rows = _backtest(book(dn, nt, quantities=[2.0, -1.5])[0],
                     _path(pd.date_range(VAL, periods=len(days), freq="D"), realized)).rows
    assert any(d.paid_cash > 0.0 for d in days) and any(d.pending_receivable_pv != 0.0 for d in days)
    assert [d.paid_cash for d in days] == [r["paid_cash"] for r in rows]
    assert [d.pending_receivable_pv for d in days] == [r["pending_pv"] for r in rows]


# A non-TARN FX backtest (GK vanilla, 30 hedges), frozen on the ARM64 dev machine: byte-identical on this branch and on
# its base main c02296cb (Task 12). Compared at GOLDEN_REL_TOL: a frozen literal crosses architectures
# (test/golden_compare.py); the same-machine byte gate is the evidence script w4b_baseline.py (B6).
_NON_TARN_BACKTEST = {
    "final_value": "0x1.3d44b9ded6c7bp-1",
    "total_hedge_pnl": "-0x1.c3c234176532ap-5",
    "last_net_pnl": "0x1.dd61d56818418p-7",
}


def test_a_non_target_redemption_backtest_value_is_pinned():
    from golden_compare import GOLDEN_REL_TOL
    from quantark.portfolio.fx import FXPortfolio
    pf = FXPortfolio(portfolio_name="pin", pricing_environments={"EURUSD": env()})
    pf.add_position(product=FxVanillaOption(strike=1.2, option_type=OptionType.CALL, maturity=1.0,
                                            notional_foreign=1.0, currency_pair=PAIR),
                    quantity=10.0, entry_price=0.0, underlying="EURUSD", engine=GarmanKohlhagenEngine())
    index = pd.date_range(VAL, periods=30, freq="D")
    engine = FXBacktestEngine(FXBacktestConfig(
        portfolio=pf, market_path=_path(index, [1.20 + 0.003 * ((k % 5) - 2) for k in range(30)]),
        strategy=FXDeltaNeutralStrategy(delta_threshold=0.05, rebalance_frequency="on_threshold"),
        transaction_cost_model=ZeroCostModel(), calculate_greeks=True))
    res = engine.run()
    live = {"final_value": res.final_value, "total_hedge_pnl": res.total_hedge_pnl,
            "last_net_pnl": res.rows[-1]["net_pnl"]}
    assert res.num_hedges > 0
    assert all(r["paid_cash"] == 0.0 and r["pending_pv"] == 0.0 for r in res.rows)
    for key, golden in _NON_TARN_BACKTEST.items():
        assert is_close(live[key], float.fromhex(golden), rel_tol=GOLDEN_REL_TOL), (key, live[key].hex())


# ================================================================ final-review Critical 1: time-of-day dates in a run

_AT10 = [d + timedelta(hours=10) for d in FIX_DATES]


def _ten_oclock(kind, lag):
    pays = [d + timedelta(days=lag) for d in _AT10]
    if kind == "tarn":
        return dated_note(fixing_dates=_AT10, pay_dates=pays, target=3.0)
    return dated_tarf(fixing_dates=_AT10, pay_dates=pays, target=0.20)


def _ten_oclock_run(product, greeks, days=36):
    pf, (pid,) = book(product)
    engine = FXBacktestEngine(FXBacktestConfig(
        portfolio=pf, market_path=_flat_path(days, 1.20),
        strategy=FXDeltaNeutralStrategy(delta_threshold=1e12, rebalance_frequency="on_threshold"),
        transaction_cost_model=ZeroCostModel(), calculate_greeks=greeks))
    return pid, engine.run().rows


@pytest.mark.parametrize("lag", [0, 2])
@pytest.mark.parametrize("kind", ["tarn", "tarf"])
def test_ten_oclock_dates_in_a_backtest_price_like_the_twin_and_book_cash_once_strictly_past(kind, lag, monkeypatch):
    """Final-review Critical 1, probe 3 in a run: fixing and pay datetimes at 10:00 on daily midnight rows (zero lag
    raised at the midnight after the fixing). Every row prices, with Greeks on or off (same rows); each row's value
    equals the numeric twin written from the terms and the held state at that row; the cash is in the value while its
    pay time resolves to >= 0 and is booked on the first row where it is strictly past."""
    seen = []
    original = RunFixingRecorder.advance

    def spy(self, portfolio):
        day = original(self, portfolio)
        e = portfolio.pricing_environments["EURUSD"]
        seen.append((e, {pid: (p.product, p.get_market_value(e)) for pid, p in portfolio.positions.items()}))
        return day

    monkeypatch.setattr(RunFixingRecorder, "advance", spy)
    pid, rows = _ten_oclock_run(_ten_oclock(kind, lag), greeks=False)
    assert len(seen) == len(rows)
    fixing, pay = _AT10[0], _AT10[0] + timedelta(days=lag)
    final = seen[-1][1][pid][0]
    (cash,) = [x.amount for x in final.lifecycle_events if x.event_type is FIXING]
    pending_rows = booked_rows = 0
    for (e, held), row in zip(seen, rows):
        product, value = held[pid]
        twin = twin_from_terms(product, e)
        assert is_close(value, engine_for(twin, n=2_000).price(twin, e), rel_tol=1e-12), e.valuation_date
        assert is_close(row["portfolio_value"] - row["paid_cash"], value, rel_tol=1e-12, abs_tol=1e-12)
        t_fix, t_pay = signed_yf(fixing, e), signed_yf(pay, e)
        if t_pay < 0.0:                                                  # strictly past: booked, once
            assert (row["paid_cash"], row["pending_pv"]) == (cash, 0.0), e.valuation_date
            booked_rows += 1
        elif t_fix <= 0.0:                                               # fixed (or live), its cash due: in value
            assert row["paid_cash"] == 0.0, e.valuation_date
            assert is_close(row["pending_pv"], cash * e.get_domestic_df(t_pay), rel_tol=1e-12), e.valuation_date
            pending_rows += 1
        else:
            assert (row["paid_cash"], row["pending_pv"]) == (0.0, 0.0), e.valuation_date
    assert pending_rows >= 2 and booked_rows >= 1                        # the midnight after the fixing is pending
    payments = [x for x in final.lifecycle_events if x.event_type is PAYMENT]
    assert [x.amount for x in payments] == [cash]
    _, with_greeks = _ten_oclock_run(_ten_oclock(kind, lag), greeks=True)
    fields = ("portfolio_value", "paid_cash", "pending_pv")
    assert [[r[f] for f in fields] for r in with_greeks] == [[r[f] for f in fields] for r in rows]
