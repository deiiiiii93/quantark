"""A numeric contract shows no difference from its dated twin (tracker clock decision A, 2026-10-09; clock rule
2026-10-26).

The same contract, struck N = 0, 30 and 100 days before the run's ``start_date``, is written twice. On dates:
``initial_date + d_k`` with ``d_k = round(k / 12 * 365)`` (the tracker's snapped day mapping) and an
``exercise_date``. Numerically, on the numeric clock alone (no ``initial_date``: a contract is all dates or all
floats): observation times ``k / 12`` years at the strike, aged N days to the run's start (``time_shift``), which is
how a numeric contract struck before today is booked -- records already past dropped, the elapsed time banked. Each is
run through its lifecycle tracker on the same market path (a function of the calendar date). Both must give the same
event days, the same ledger (cash, and determination/payment instants measured from the strike) and the same
pricing-copy PVs. The aged numeric form has lost the records before the run, so its observation indices (and the ids
built on them) are the dated ones less the number dropped.

Records dated before the run (N = 100 puts three of them there) are never tested (decision 2): the dated form skips
them, the numeric booking no longer carries them.

Cash that ACCRUES cannot agree across the forms, at any N: a numeric record accrues its contractual time (``k / 12``,
the accrual time spec fills it once at construction and snapping moves timing only), a dated record accrues its days
in the day count (``d_k / 365``). Record 0 of an annualized 15% snowball pays 1.25 numerically and 1.2329 on dates
at N = 0 already. So the exact cross-form parity runs on contracts whose cash does not accrue (flat KO coupon,
per-period Phoenix coupon), and the annualized contracts are held to everything else -- event days, ids, instants --
and to cash that does not depend on N within each form.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List

import pandas as pd
import pytest

from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import KOResetSnowballPDESolver
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.lifecycle.ko_reset import tracker_for
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.product.option.ko_reset_snowball_option import KnockOutResetSnowballOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_helpers import create_standard_phoenix
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import add_months
from quantark.util.enum.option_enums import ObservationType, PostKOScheduleMode as M

START = pd.Timestamp("2026-01-05")
GAPS = (0, 30, 100)
TIMES = [k / 12.0 for k in range(1, 13)]
HORIZON = 420


def env(spot: float, date) -> PricingEnvironment:
    return PricingEnvironment(spot_quote=SpotQuote(spot=spot, asset_name="IDX"),
                              vol_surface=FlatVolSurface(volatility=0.20), rate_curve=FlatRateCurve(rate=0.03),
                              valuation_date=pd.Timestamp(date).to_pydatetime())


def on(initial: pd.Timestamp, days: int):
    return (initial + pd.Timedelta(days=int(days))).to_pydatetime()


def numeric_schedule(times, level):
    return ObservationSchedule(records=[ObservationRecord(observation_time=t, barrier=level) for t in times])


def dated_schedule(initial, times, level):
    """``times`` on the tracker's snapped days: ``initial + round(t * 365)``."""
    return ObservationSchedule(records=[
        ObservationRecord(observation_date=on(initial, round(t * 365)), barrier=level) for t in times])


# -- the contracts, each in both forms -------------------------------------------------------------------------------

def _on_dates(p, initial):
    p.barrier_config = replace(p.barrier_config, ko_observation_dates=None,
                               ko_observation_schedule=dated_schedule(initial, TIMES, 103.0))
    p.maturity = None
    p.exercise_date = on(initial, 365)


def snowball(annualized: bool) -> Callable:
    """KO 103 at k/12 (15%, flat or annualized), continuous KI 75, 1 year."""
    def build(form: str, initial: pd.Timestamp):
        p = create_standard_snowball(100.0, 100.0, 1.0, ko_barrier=103.0, ki_barrier=75.0, ko_rate=0.15,
                                     is_annualized=annualized)
        if form == "dated":
            p.initial_date = initial.to_pydatetime()
            _on_dates(p, initial)
        p.validate()
        return p
    return build


def phoenix(annualized: bool, memory: bool = True) -> Callable:
    """KO 103, continuous KI 70, a (memory) coupon of 1% (per period or per annum) at barrier 80, 12 records at k/12."""
    def build(form: str, initial: pd.Timestamp):
        p = create_standard_phoenix(100.0, 100.0, 1.0, ko_barrier=103.0, ki_barrier=70.0, coupon_barrier=80.0,
                                    coupon_rate=0.01, num_observations=12, memory_coupon=memory,
                                    is_annualized=annualized)
        if form == "dated":
            p.initial_date = initial.to_pydatetime()
            _on_dates(p, initial)
        p.validate()
        return p
    return build


def _rebased_gaps(initial: pd.Timestamp) -> List[float]:
    """The dated REBASED twin's schedule 2 after a knock-in on its last knock-in record (day 182, the record the
    knock-in path fires), as offsets in years: the day gaps of ``add_months(knock-in date, k)``, k = 1..6."""
    knock_in = on(initial, round(TIMES[5] * 365))
    return [(add_months(knock_in, k) - knock_in).days / 365.0 for k in (1, 2, 3, 4, 5, 6)]


def ko_reset(mode: M) -> Callable:
    """Schedule 1 (KO 103, flat 15%) and the discrete KI (75) at k/12 for k = 1..6; schedule 2 (KO 95, flat 3%) at
    k/12 for k = 7..12 (ABSOLUTE) or k = 1..6 months after the knock-in (REBASED: a knock-in on the first record,
    day 30, plus 6 calendar months ends after schedule 1 (day 182) -- schedule 2 ends last, spec D35); 5% rebate.

    REBASED (clock rule 2026-10-26): the dated form books its offsets as whole months (k/12 years) and generates its
    schedule 2 as the knock-in date + k calendar months; the numeric form cannot vary its offsets by knock-in, so it
    books the dated twin's day gaps for the knock-in the path fires (``_rebased_gaps``) -- the same contract on every
    path this module walks."""
    pre = TIMES[:6]
    post = TIMES[6:] if mode == M.ABSOLUTE else TIMES[:6]

    def build(form: str, initial: pd.Timestamp):
        dated = form == "dated"
        sched = (lambda times, level: dated_schedule(initial, times, level)) if dated else numeric_schedule
        if mode == M.ABSOLUTE:
            post_sched = sched(post, 95.0)
            maturity, exercise = 1.0, on(initial, 365) if dated else None
        else:
            offsets = post if dated else _rebased_gaps(initial)
            post_sched = numeric_schedule(offsets, 95.0)
            maturity = pre[-1] + offsets[-1]
            exercise = add_months(on(initial, round(pre[-1] * 365)), 6) if dated else None
        p = KnockOutResetSnowballOption(
            initial_price=100.0, strike=100.0, maturity=None if exercise is not None else maturity,
            initial_date=initial.to_pydatetime() if dated else None, exercise_date=exercise,
            barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.15, ko_observation_type=ObservationType.DISCRETE,
                                         ko_observation_schedule=sched(pre, 103.0), ki_barrier=75.0,
                                         ki_observation_type=ObservationType.DISCRETE,
                                         ki_observation_schedule=sched(pre, 75.0)),
            post_barrier_config=BarrierConfig(ko_barrier=95.0, ko_rate=0.03,
                                              ko_observation_type=ObservationType.DISCRETE,
                                              ko_observation_schedule=post_sched),
            accrual_config=AccrualConfig(is_annualized=False),
            payoff_config=PayoffConfig(rebate_rate=0.05, include_principal=False), post_ko_mode=mode)
        p.validate()
        return p

    return build


#: Contracts whose cash does not accrue: the two forms are the same contract to the cent.
KINDS: Dict[str, Callable] = {
    "snowball": snowball(annualized=False),
    "phoenix": phoenix(annualized=False),
    "ko_reset_absolute": ko_reset(M.ABSOLUTE),
    "ko_reset_rebased": ko_reset(M.REBASED),
}
#: Annualized contracts: the forms accrue differently (k/12 against d_k/365) at every N, by construction.
ACCRUING: Dict[str, Callable] = {
    "snowball_annualized": snowball(annualized=True),
    "phoenix_annualized": phoenix(annualized=True),
}
BUILDERS = {**KINDS, **ACCRUING}
SOLVERS = {
    "snowball": lambda: SnowballPDESolver(PDEParams()),
    "phoenix": lambda: PhoenixPDESolver(PDEParams()),
    "ko_reset_absolute": lambda: KOResetSnowballPDESolver(PDEParams()),
    "ko_reset_rebased": lambda: KOResetSnowballPDESolver(PDEParams()),
}


# -- the market paths: spot as a function of the contract day (days since initial_date) -----------------------------

def _knock_in_path(kind: str) -> Callable[[int], float]:
    # the snowball and Phoenix knock in continuously from contract day 200; the KO-reset on its day-182 record
    low = range(180, 186) if kind.startswith("ko_reset") else range(200, 215)
    return lambda c: 70.0 if c in low else (100.0 if c < low.start else 90.0)


PATHS: Dict[str, Callable[[str], Callable[[int], float]]] = {
    "quiet": lambda kind: (lambda c: 100.0),          # no barrier event; every Phoenix coupon pays
    "knock_out": lambda kind: (lambda c: 104.0),      # knocks out on the first record tested in the run
    "knock_in": _knock_in_path,
}


def booking(kind: str, form: str, gap: int):
    """The contract struck ``gap`` days before the run, as booked on the run's start: the dated form as written, the
    numeric form aged ``gap`` days (keeping a record due on the run's first day)."""
    initial = START - pd.Timedelta(days=gap)
    product = BUILDERS[kind](form, initial)
    if form == "numeric" and gap:
        product.time_shift(gap / 365.0, START.to_pydatetime(), env(100.0, START), keep_landing=True)
    return product, initial


def dropped(kind: str, form: str, gap: int) -> int:
    """How many knock-out records the numeric booking lost to the gap (0 for the dated form)."""
    if form == "dated":
        return 0
    return BUILDERS[kind](form, START).barrier_config.ko_observation_count \
        - booking(kind, form, gap)[0].barrier_config.ko_observation_count


def tracker(kind: str, form: str, gap: int):
    product, initial = booking(kind, form, gap)
    return tracker_for(product, quantity=1.0, start_date=START), initial


def walk(kind: str, form: str, gap: int, path: str) -> dict:
    """Observe every close from START until the contract is dead; events and ledger in a form-free shape."""
    t, initial = tracker(kind, form, gap)
    offset = dropped(kind, form, gap)
    shift = gap / 365.0 if form == "numeric" else 0.0
    spot_of = PATHS[path](kind)
    events: List[tuple] = []
    for r in range(HORIZON):
        day = START + pd.Timedelta(days=r)
        spot = spot_of(r + gap)
        e = env(spot, day)
        product = t.product_for_lifecycle()
        fired = t.observe(day, product, e, spot)
        maturity = t.settle_maturity_if_due(day, product, e, spot)
        fired += [maturity] if maturity is not None else []
        events += [(r, ev.event_type.value, _index(ev.observation_index, offset),
                    _rekey(getattr(ev.realized_cashflow, "cashflow_id", None), offset), ev.cashflow) for ev in fired]
        if not t.lifecycle.alive:
            break
    ledger = sorted(_instants(cf, initial, offset, shift) for cf in t.lifecycle.ledger.cashflows)
    return {"events": events, "ledger": ledger, "alive": t.lifecycle.alive}


def _index(index, offset: int):
    """A numeric booking's record index in the contract's own numbering (the records it dropped come first)."""
    return None if index is None else index + offset


def _rekey(cashflow_id, offset: int):
    """``kind:index`` in the contract's own numbering."""
    if cashflow_id is None or ":" not in cashflow_id:
        return cashflow_id
    kind, index = cashflow_id.rsplit(":", 1)
    return f"{kind}:{int(index) + offset}"


def _instants(cf, initial: pd.Timestamp, offset: int, shift: float) -> tuple:
    """(id, amount, determination, payment), instants in years from the strike on either clock: a dated instant by its
    days from ``initial_date``, a numeric one (measured from the run's start) plus the gap ``shift``. Rounded to 12
    places: ``t - gap/365 + gap/365`` need not return ``t`` exactly."""
    def years(date, time):
        if date is not None:
            return round((pd.Timestamp(date).normalize() - initial.normalize()).days / 365.0, 12)
        return round(time + shift, 12)
    return (_rekey(cf.cashflow_id, offset), cf.amount, years(cf.determination_date, cf.determination_time),
            years(cf.payment_date, cf.payment_time))


CASES = [(k, n, p) for k in KINDS for n in GAPS for p in PATHS]


@pytest.mark.parametrize("kind, gap, path", CASES, ids=[f"{k}-N{n}-{p}" for k, n, p in CASES])
def test_a_numeric_contract_books_what_its_dated_twin_books(kind, gap, path):
    numeric = walk(kind, "numeric", gap, path)
    dated = walk(kind, "dated", gap, path)
    assert not numeric["alive"] and numeric["events"], "every path ends the contract inside the horizon"
    assert numeric["events"] == dated["events"]
    assert numeric["ledger"] == dated["ledger"]


ACCRUING_CASES = [(k, n, p) for k in ACCRUING for n in GAPS for p in PATHS]


@pytest.mark.parametrize("kind, gap, path", ACCRUING_CASES, ids=[f"{k}-N{n}-{p}" for k, n, p in ACCRUING_CASES])
def test_an_accruing_numeric_contract_is_timed_like_its_dated_twin(kind, gap, path):
    """Every event day, id and instant agrees; the cash differs only by the accrual convention, so each form's cash
    for a cashflow is the one it books with no gap at all."""
    numeric = walk(kind, "numeric", gap, path)
    dated = walk(kind, "dated", gap, path)
    drop_cash = lambda rows, at: [r[:at] + r[at + 1:] for r in rows]  # noqa: E731
    assert drop_cash(numeric["events"], 4) == drop_cash(dated["events"], 4)
    assert drop_cash(numeric["ledger"], 1) == drop_cash(dated["ledger"], 1)
    for form, got in (("numeric", numeric), ("dated", dated)):
        fresh = {row[0]: row[1] for row in walk(kind, form, 0, path)["ledger"]}
        for row in got["ledger"]:
            if row[0] in fresh:       # a gap can move the first knock-out past record 0
                assert row[1] == fresh[row[0]], f"{form} {row[0]}"


def test_the_paths_exercise_what_they_name():
    """The fixture's paths do what they claim, on the numeric form with the widest gap."""
    knock_out = walk("snowball", "numeric", 100, "knock_out")["events"]
    assert [e[1] for e in knock_out] == ["KO"] and knock_out[0][2] == 3        # records 0-2 lie before the run
    knock_in = [e[1] for e in walk("snowball", "numeric", 100, "knock_in")["events"]]
    assert knock_in == ["KI", "MATURITY"]
    rebased = walk("ko_reset_rebased", "numeric", 100, "knock_in")["events"]
    assert [e[1] for e in rebased] == ["KI", "MATURITY"]
    coupons = [e for e in walk("phoenix", "numeric", 100, "quiet")["events"] if e[1] == "COUPON"]
    assert [e[2] for e in coupons] == list(range(3, 12))                        # coupons 0-2 were paid before


def _pv_contract_days(path: str, gap: int) -> List[int]:
    """Contract days to price the copy on: the run's first valuation (before its first close) and a few later
    closes while every contract is alive (the KO-reset matures on schedule 1's last record, day 182, unless knocked
    in); around the knock-in on the knock-in path (the KO-reset knocks in on day 182, the others from day 200)."""
    return sorted({gap, gap + 15, gap + 45, 170}) if path == "quiet" else [190, 230]


def _copy_pvs(kind: str, form: str, gap: int, path: str) -> List[float]:
    """The post-close pricing copy's PV on each of ``_pv_contract_days`` (the pre-close initial copy on run day 0)."""
    t, _ = tracker(kind, form, gap)
    spot_of = PATHS[path](kind)
    solver = SOLVERS[kind]()
    days = [c - gap for c in _pv_contract_days(path, gap)]
    out = []
    for r in range(max(days) + 1):
        day = START + pd.Timedelta(days=r)
        spot = spot_of(r + gap)
        e = env(spot, day)
        if r == 0 and 0 in days:
            out.append(float(solver.price(t.product_for_pricing(day, e, initial=True), e)))
        product = t.product_for_lifecycle()
        t.observe(day, product, e, spot)
        t.settle_maturity_if_due(day, product, e, spot)
        assert t.lifecycle.alive, f"the contract died on run day {r}, before its last PV day"
        if r in days and r != 0:
            out.append(float(solver.price(t.product_for_pricing(day, e), e)))
    return out


# The PDE prices a REBASED KO-reset only once a knock-in has converted it (an ABSOLUTE contract).
PV_CASES = [(k, n, p) for k in KINDS for n in GAPS for p in ("quiet", "knock_in")
            if not (k == "ko_reset_rebased" and p == "quiet")]
#: PV per 100 notional. Measured (W1, 2026-10-09): worst |numeric - dated| 1.0e-12 over every case. The two forms
#: reach the engine as different floats (a snapped k/12 time aged by elapsed days against a day count of two dates),
#: so the PDE's time grid moves by ulps; the PVs agree to that, and each contract day prices the same at every N.
PV_ATOL = 1e-11


@pytest.mark.parametrize("kind, gap, path", PV_CASES, ids=[f"{k}-N{n}-{p}" for k, n, p in PV_CASES])
def test_a_numeric_pricing_copy_prices_like_its_dated_twin(kind, gap, path):
    numeric = _copy_pvs(kind, "numeric", gap, path)
    dated = _copy_pvs(kind, "dated", gap, path)
    assert len(numeric) == len(dated) > 0
    assert numeric == pytest.approx(dated, abs=PV_ATOL, rel=0.0)
