"""FX TARN / TARF ageing inside runs (spec 2026-10-09-fx-tarn-lifecycle-design.md section 7, tests B1-B6, Q8) on the
W6 run-fixings recorder (spec 2026-10-09-run-fixings-design.md section 6; TARN plan re-plan 2026-10-10)."""
import dataclasses
import warnings
from datetime import datetime, timedelta

import pytest

from fx_target_redemption_fixtures import (
    FIX, FIX_DATES, INCEPTION, LAG2, PAIR, TARGET_CAL, VAL, _numeric_twin, book, dated_note, drive, engine_for, env,
    note, tarf,
)
from quantark.asset.equity.lifecycle.autocallable import MaturedBeforeRunWarning
from quantark.asset.fx.engine.analytical import GarmanKohlhagenEngine
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxVanillaOption
from quantark.asset.fx.product.option.target_redemption_state import (
    PendingPayment, TargetRedemptionEventType, TargetRedemptionState, TerminalBookingWarning,
)
from quantark.param import SpotQuote
from quantark.util.calendar import DayCountConvention
from quantark.util.enum import OptionType
from quantark.util.exceptions import ValidationError
from quantark.util.fixings import MissingFixingWarning
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
