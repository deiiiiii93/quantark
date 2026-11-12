"""The pricing pass: per note, per live day, a spot ladder of QUAD V2 price / delta / gamma for each hedge model.

Order of one note's daily loop (``s2ts.runner.run_one``'s): the day's carry curve (``converge_hl100`` with the
63-day anchor) and environment, ``NoteLife.step`` (the tracker observes KO / KI on the close, settles maturity),
then the pricing. Day 0 is the inception close. Every model is priced with the TRUE lifecycle state except ``kin``
(a copy forced to knocked-in, ``gpin.models``); variants are rebuilt per day and aged with the tracker's
``age_copy`` (for the unmodified note that reproduces ``tracker.product_for_pricing`` exactly). One solve returns
the whole ladder (``engine.prepare(...).evaluate(spots)``): the ladder price at 0 equals the scalar price and its
1% bump delta equals ``s2ts.note.daily_greeks``' spot delta to rounding.

Windows (pre-registration ``windows``; the report's measurement of them is the ledger's, not this pass's)

* KO window: the next KO observation (the first KO date strictly after the day) is at most ``ko_days`` trading days
  ahead, counted as days in (day, T] on the SSE calendar (the chain holds every one of them; the calendar also
  covers the days past the data end), and ``|S / B - 1| <= ko_band`` with ``B`` the true KO barrier. Inside it the
  pass prices ``surv`` first and computes ``gap_ko``; with a direction it prices the KO shift models.
* KI window: the note is alive, not knocked in, and ``S / K - 1 <= ki_band`` (5%), ``K`` the book's KI level (a
  book is a KI level: ``ki75`` the product, ``ki85``, ``ki90``). Inside it: the KI shift models and ``kin``.

Books: every model, the note itself, the KI window test and the scan cross-check use the book's KI level
(``ki_pct``); the coupon is solved per (book, inception) to PV = 0 under hold-forward (``solve_book_coupon``, the
method of ``s2ts.note.solve_coupon``).

``gap_ko`` is in engine terms (the engine prices the long note), CNY per note: the KO cash paid at the next
observation, discounted at the flat rate from the observation date to today (ACT/365), minus the ``surv`` engine
price interpolated linearly on its ladder at ``B``. The KO cash is read the way the lifecycle pays it: the
``payoff`` of the first record of ``resolve_ko_observations`` of the aged note (``AutocallableLifecycleTracker.observe``
books ``quantity x rec["payoff"]``, ``_record_dicts`` copies that ``payoff``). A positive ``gap_ko`` is what the
seller loses if the note is redeemed rather than survives. ``ko_dir`` = -1 when ``gap_ko > 0``, else +1, fixed on the
first priced KO-window day of an observation and kept for it; 0 (no shift model for that observation) when that
first gap is zero. Outside the window ``gap_ko`` and ``ko_dir`` are null.

Recorded fields keep the ``engine_*`` names (raw engine values, holder side); seller conventions apply downstream.
A NaN greek at a ladder point (a contractual jump) is stored as null and never interpolated across.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from bisect import bisect_right
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import numpy as np
import pandas as pd
import scipy

from quantark.param import FlatRateCurve
from quantark.util.exceptions import ValidationError
import quantark
from quantark.util.numerical import is_close, is_valid_number, is_zero, safe_exp

from s2ts import ACT, NOTIONAL, RATE
from s2ts.chain import sse_trading_days
from s2ts.note import COUPON_PROBES, COUPON_PV_TOL_FRACTION, COUPON_TAIL, KI_PCT, KO_PCT, NoteLife, build_terms, make_env
from s2ts.runner import v2_engine, write_json_atomic
from s2ts.tails import build_curve

from . import CHAIN_PATH, LADDER, STUDY_DIR, TAIL, StudyDataError, load_prereg
from .events import BOOKS, NoteScan, note_id
from .models import MODELS, HedgeModel, book_note, next_ko_index

_WINDOWS = load_prereg()["windows"]
KO_DAYS: int = int(_WINDOWS["ko_days"])
KO_BAND: float = float(_WINDOWS["ko_band"])
KI_BAND: float = float(_WINDOWS["ki_band"])

#: The s2ts files this pass imports, hashed into the fingerprint (``books`` is imported by ``runner``).
S2TS_FILES = ("__init__.py", "books.py", "chain.py", "note.py", "runner.py", "tails.py")
#: This package's files hashed into the fingerprint; ledger-side code is deliberately not among them.
GPIN_FILES = ("__init__.py", "models.py", "pricing.py")

_STATE_KEYS = ("engine_price", "engine_delta", "engine_gamma")


# ---------------------------------------------------------------------------
# Windows and the KO direction
# ---------------------------------------------------------------------------


def _at_most(value: float, bound: float) -> bool:
    """``value <= bound`` with the repo's tolerance (a ratio of prices lands a hair off a round bound)."""
    return value < bound or is_close(value, bound)


def in_ko_window(spot: float, s0: float, days_to_ko: Optional[int]) -> bool:
    """The KO window of the module docstring (``days_to_ko`` is ``None`` when no observation is left)."""
    if days_to_ko is None or days_to_ko > KO_DAYS:
        return False
    return _at_most(abs(spot / (KO_PCT * s0) - 1.0), KO_BAND)


def in_ki_window(spot: float, s0: float, knocked_in: bool, ki_pct: float = KI_PCT) -> bool:
    """The KI window of the module docstring (the caller has checked that the note is alive)."""
    return (not knocked_in) and _at_most(spot / (ki_pct * s0) - 1.0, KI_BAND)


def book_of(ki_pct: float) -> str:
    """The book whose KI level is ``ki_pct``; ``StudyDataError`` when there is none."""
    for name, level in BOOKS.items():
        if is_close(level, ki_pct):
            return name
    raise StudyDataError(f"no book has the KI level {ki_pct}; books are {BOOKS}")


class KoDirection:
    """``ko_dir`` per KO observation, fixed by the sign of ``gap_ko`` on the first KO-window day and kept."""

    def __init__(self) -> None:
        self._dir: Dict[date, int] = {}

    def resolve(self, obs_date: date, gap_ko: float) -> int:
        """-1 when redemption costs the seller (``gap_ko > 0``), +1 otherwise, 0 when the first gap is zero."""
        if obs_date not in self._dir:
            self._dir[obs_date] = 0 if is_zero(gap_ko) else (-1 if gap_ko > 0 else 1)
        return self._dir[obs_date]


class TradingCalendar:
    """SSE trading days, for the number of trading days between two dates (also beyond the data end)."""

    def __init__(self, first_year: int, last_year: int) -> None:
        self.days: List[date] = [d.date() for d in sse_trading_days(first_year, last_year)]

    def between(self, day: date, until: date) -> int:
        """Trading days in (day, until]."""
        return bisect_right(self.days, until) - bisect_right(self.days, day)


def calendar_for(chains: Sequence) -> TradingCalendar:
    """The calendar covering the chain and the 2Y maturities of its last inceptions."""
    return TradingCalendar(chains[0].day.year, chains[-1].day.year + 3)


# ---------------------------------------------------------------------------
# Expected solves (the plan)
# ---------------------------------------------------------------------------


def expected_solves(chains: Sequence, scan: NoteScan, calendar: TradingCalendar, *,
                    price_days: Optional[Set[date]] = None, max_days: Optional[int] = None,
                    ki_pct: float = KI_PCT) -> Dict[str, int]:
    """Counts the pass will make for ``scan``'s note, from the chain and the scan record alone: priced days, days in
    each window, and the solves (1 for ``true`` + ``1 + n_ko_shift`` in the KO window + ``n_ki`` in the KI window,
    assuming every first KO gap is non-zero, i.e. every shift model is priced)."""
    by_day = {c.day: i for i, c in enumerate(chains)}
    i0 = by_day[scan.inception]
    last = by_day[scan.end_date]
    if max_days is not None:
        last = min(last, i0 + int(max_days))
    terms = build_terms(scan.inception)
    s0 = chains[i0].spot
    n_ko_models = sum(1 for m in MODELS.values() if m.window == "ko")      # surv + shifts
    n_ki_models = sum(1 for m in MODELS.values() if m.window == "ki")
    out = {"priced_days": 0, "ko_window_days": 0, "ki_window_days": 0, "solves": 0}
    for i in range(i0, last + 1):
        dc = chains[i]
        dead = i == by_day[scan.end_date] and scan.end_reason != "truncated"
        if dead or (price_days is not None and dc.day not in price_days):
            continue
        out["priced_days"] += 1
        out["solves"] += 1
        j = next_ko_index(terms, dc.day)
        n = None if j is None else calendar.between(dc.day, terms.ko_dates[j])
        if in_ko_window(dc.spot, s0, n):
            out["ko_window_days"] += 1
            out["solves"] += n_ko_models
        knocked_in = scan.ki_date is not None and dc.day >= scan.ki_date
        if in_ki_window(dc.spot, s0, knocked_in, ki_pct):
            out["ki_window_days"] += 1
            out["solves"] += n_ki_models
    return out


# ---------------------------------------------------------------------------
# Solves
# ---------------------------------------------------------------------------


def _clean(values: np.ndarray, *, price: bool, label: str) -> List[Optional[float]]:
    """A ladder array as a JSON list; a NaN greek is ``None`` (a contractual jump), a NaN price is an error."""
    out: List[Optional[float]] = []
    for v in np.asarray(values, dtype=float):
        if is_valid_number(float(v)):
            out.append(float(v))
        elif price:
            raise StudyDataError(f"{label}: the engine returned a non-finite price on the ladder")
        else:
            out.append(None)
    return out


def solve_ladder(engine, product, env, spots: np.ndarray, lifecycle_state, label: str = "") -> Dict[str, List]:
    """One solve of ``product`` (already aged) over ``spots``: ``engine_price`` / ``engine_delta`` /
    ``engine_gamma`` lists, NaN greeks as ``None``."""
    ctx = engine.prepare(product, env, spots, lifecycle_state=lifecycle_state)
    r = ctx.evaluate(spots)
    return {
        "engine_price": _clean(r["price"], price=True, label=label),
        "engine_delta": _clean(r["delta"], price=False, label=label),
        "engine_gamma": _clean(r["gamma"], price=False, label=label),
    }


def aged_model_product(life: NoteLife, model: HedgeModel, terms, s0: float, coupon: float, day: date,
                       inception: date, env, ko_dir: Optional[int], ki_pct: float = KI_PCT):
    """``model``'s product for ``day`` in the book whose KI level is ``ki_pct``, aged by the tracker exactly as
    the true note is."""
    elapsed = (day - inception).days / ACT
    product = model.product(terms, s0, coupon, day, ko_dir, ki_pct)
    aged = life.tracker.age_copy(product, elapsed, pd.Timestamp(day), env, initial=(day == inception))
    aged = _trim_ko_levels(aged, terms, elapsed)
    left = len(aged.barrier_config.ko_observation_dates or ())
    if left != sum(1 for d in terms.ko_dates if d > day):
        raise StudyDataError(f"{model.id} {day}: {left} KO observations left, the term sheet has "
                             f"{sum(1 for d in terms.ko_dates if d > day)} after the day")
    return aged


def _trim_ko_levels(product, terms, elapsed: float):
    """Drop the KO levels of the observations ageing has removed.

    ``BarrierConfig.time_shift`` shifts the observation schedule (its records carry their own barrier) and the dates,
    but leaves a per-observation ``ko_barrier`` list at its full length, so an aged variant fails validation. The
    survivors are the last ``k`` observations of the term sheet: the list is cut to them, after checking that the
    surviving record times are exactly the term-sheet times shifted by ``elapsed`` and that the records' own barriers
    equal the cut list (fail closed otherwise)."""
    cfg = product.barrier_config
    levels = cfg.ko_barrier
    if not isinstance(levels, list) or len(levels) == len(cfg.ko_observation_dates or ()):
        return product
    records = list(cfg.ko_observation_schedule.records) if cfg.ko_observation_schedule is not None else []
    k = len(records)
    if k == 0 or k > len(levels):
        raise StudyDataError(f"{k} KO records left of {len(levels)} levels")
    expected = [t - elapsed for t in terms.ko_times[-k:]]
    if not all(is_close(float(r.observation_time), t) for r, t in zip(records, expected)):
        raise StudyDataError("the aged KO times are not the last term-sheet observations shifted by the elapsed time")
    cut = list(levels[-k:])
    if not all(is_close(float(r.barrier), lv) for r, lv in zip(records, cut)):
        raise StudyDataError("the aged KO records' barriers differ from the surviving levels")
    product.barrier_config = replace(cfg, ko_barrier=cut)
    return product


def model_ladder(engine, life: NoteLife, model: HedgeModel, terms, s0: float, coupon: float, day: date,
                 inception: date, env, spots: np.ndarray, ko_dir: Optional[int],
                 ki_pct: float = KI_PCT) -> Dict[str, List]:
    """The ladder of ``model`` on ``day``: aged variant, the model's lifecycle state (true state except ``kin``)."""
    product = aged_model_product(life, model, terms, s0, coupon, day, inception, env, ko_dir, ki_pct)
    return solve_ladder(engine, product, env, spots, model.lifecycle(life.lifecycle), label=f"{model.id} {day}")


def ko_cash(product, env, obs_date: date, day: date, terms) -> float:
    """The KO cash of the next observation as the lifecycle pays it: ``payoff`` of the first record of
    ``resolve_ko_observations`` of the aged note (CNY per note, undiscounted)."""
    records = product.resolve_ko_observations(env)
    due = [d for d in terms.ko_dates if d > day]
    if len(records) != len(due):
        raise StudyDataError(f"{day}: the aged note has {len(records)} KO records, the term sheet {len(due)} left")
    first = records[0]
    expected_time = (obs_date - day).days / ACT
    if not is_close(float(first.observation_time), expected_time):
        raise StudyDataError(
            f"{day}: first KO record at t={float(first.observation_time)!r}, expected {expected_time!r} ({obs_date})")
    return float(first.payoff)


def gap_ko_engine(payoff: float, obs_date: date, day: date, spots: np.ndarray, surv_price: Sequence[float],
                  barrier: float) -> float:
    """``payoff`` discounted from ``obs_date`` at the flat rate (ACT/365) minus the ``surv`` price at ``barrier``,
    interpolated linearly on the ladder."""
    if not (spots[0] <= barrier <= spots[-1]):
        raise StudyDataError(f"{day}: the KO barrier {barrier} is off the ladder {spots[0]}..{spots[-1]}")
    discount = float(safe_exp(-RATE * (obs_date - day).days / ACT))
    return payoff * discount - float(np.interp(barrier, spots, np.asarray(surv_price, dtype=float)))


# ---------------------------------------------------------------------------
# Coupons
# ---------------------------------------------------------------------------


def solve_book_coupon(dc, engine, ki_pct: float) -> float:
    """Coupon with PV = 0 on ``dc`` under hold-forward for the note struck at ``dc.spot`` with KI level ``ki_pct``:
    the method of ``s2ts.note.solve_coupon`` (PV is affine in the coupon: two probes, one root, one re-price check,
    fails closed), so ``ki_pct = 0.75`` gives its number."""
    terms = build_terms(dc.day)
    env = make_env(dc, build_curve(dc, COUPON_TAIL, None).to_dividend_yield(FlatRateCurve(RATE)))

    def pv_at(c: float) -> float:
        return float(engine.price(book_note(c, dc.spot, terms, ki_pct), env))

    c0, c1 = COUPON_PROBES
    v0, v1 = pv_at(c0), pv_at(c1)
    annuity = (v1 - v0) / (c1 - c0)
    if not annuity > 0.0:
        raise ValidationError(f"{dc.day}: PV not increasing in the coupon (dPV/dc = {annuity})")
    coupon = c0 - v0 / annuity
    pv = pv_at(coupon)
    tol = COUPON_PV_TOL_FRACTION * NOTIONAL
    if abs(pv) > tol:
        raise ValidationError(f"{dc.day}: KI {ki_pct}: coupon {coupon:.6f} re-prices to {pv:.4f} > tol {tol:.4f}")
    return float(coupon)


def coupon_stamp_parts(*, s2ts_dir: Optional[Path] = None, data_digest: Optional[str] = None) -> Dict[str, Any]:
    """What a solved coupon depends on: the chain's content hash, the ``s2ts`` files, the engine parameters and the
    book levels (not this package's code: the solve is ``s2ts``'s method run on its note)."""
    from quantark.asset.equity.param import QuadV2Params

    s2ts_dir = (STUDY_DIR.parent / "snowball_2y_tail_stability" / "s2ts") if s2ts_dir is None else Path(s2ts_dir)
    return {
        "data": chain_digest() if data_digest is None else data_digest,
        "s2ts": {name: _file_digest(s2ts_dir / name) for name in S2TS_FILES},
        "engine": repr(QuadV2Params()),
        "books": dict(BOOKS),
    }


def coupon_stamp(*, s2ts_dir: Optional[Path] = None, data_digest: Optional[str] = None) -> str:
    """Hash of ``coupon_stamp_parts``."""
    canonical = json.dumps(coupon_stamp_parts(s2ts_dir=s2ts_dir, data_digest=data_digest), sort_keys=True,
                           separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def book_coupons(ids: Sequence[str], path: Path, stamp: str, solver) -> Dict[str, float]:
    """``{note id: coupon}`` for the note ids ``"<book>:<inception>"``, solved with ``solver(book, inception)`` and
    cached in ``path`` together with ``stamp``; a cache stamped differently (or unreadable) is discarded and every
    requested coupon is re-solved."""
    from .events import parse_note_id

    path = Path(path)
    cached: Dict[str, float] = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("stamp") == stamp and isinstance(data.get("coupons"), dict):
            cached = {str(k): float(v) for k, v in data["coupons"].items()}
    except (OSError, ValueError):
        pass
    out: Dict[str, float] = {}
    changed = False
    for nid in ids:
        if nid not in cached:
            book, inception = parse_note_id(nid)
            cached[nid] = float(solver(book, inception))
            changed = True
        out[nid] = cached[nid]
    if changed or not path.exists():
        write_json_atomic(path, {"stamp": stamp, "coupons": cached})
    return out


# ---------------------------------------------------------------------------
# One note
# ---------------------------------------------------------------------------


@dataclass
class NoteStates:
    """One note's recorded states (a note is ``book`` + ``inception``). ``days``: one record per chain day of the note's life (keys of spec C4; a day
    that was not priced under ``price_days`` has ``states == {}`` and ``unpriced: True``); ``meta``: diagnostics
    (solve count, seconds, horizon, windows)."""

    inception: date
    coupon: float
    days: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)
    book: str = "ki75"
    ki_pct: float = KI_PCT

    def to_json(self, fingerprint: Optional[str] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {"note": note_id(self.book, self.inception), "book": self.book, "ki_pct": self.ki_pct,
                               "inception": self.inception.isoformat(), "coupon": self.coupon,
                               "days": self.days, "meta": self.meta}
        if fingerprint is not None:
            out["fingerprint"] = fingerprint
        return out


def _check_against_scan(scan: NoteScan, ko_date: Optional[date], ki_date: Optional[date], end_reason: str,
                        end_date: date, horizon_end: date, natural: bool) -> None:
    """Fail closed when the tracker's lifecycle differs from the pricing-free scan (the scan's R8 cross-check). A
    run cut short by ``max_days`` is compared only inside its horizon."""
    problems = []
    for name, got, want in (("ko_date", ko_date, scan.ko_date), ("ki_date", ki_date, scan.ki_date)):
        if not natural and (want is None or want > horizon_end):
            want = None
        if got != want:
            problems.append(f"{name}: tracker {got}, scan {want}")
    if natural:
        if end_reason != scan.end_reason:
            problems.append(f"end_reason: tracker {end_reason}, scan {scan.end_reason}")
        if end_date != scan.end_date:
            problems.append(f"end_date: tracker {end_date}, scan {scan.end_date}")
    if problems:
        raise StudyDataError(f"note {scan.inception}: the tracker's lifecycle disagrees with the scan ({'; '.join(problems)})")


def price_note(inception: date, coupon: float, chains: Sequence, anchors: Mapping[date, float], scan: NoteScan, *,
               ki_pct: float = KI_PCT, engine=None, price_days: Optional[Set[date]] = None,
               max_days: Optional[int] = None) -> NoteStates:
    """The recorded states of one note of the book whose KI level is ``ki_pct`` (module docstring); ``scan`` must be
    that book's scan record. ``price_days`` / ``max_days`` are for tests and diagnostics (the lifecycle is stepped
    on every day, only those days are priced / the run stops after ``max_days`` days). Raises ``StudyDataError``
    when the realised lifecycle differs from ``scan``."""
    started = time.perf_counter()
    book = book_of(ki_pct)
    if not is_close(scan.ki_pct, ki_pct):
        raise StudyDataError(f"scan of {scan.inception} is for KI {scan.ki_pct}, not book {book} ({ki_pct})")
    engine = engine if engine is not None else v2_engine()
    by_day = {c.day: i for i, c in enumerate(chains)}
    if inception not in by_day:
        raise StudyDataError(f"inception {inception} is not a chain day")
    if scan.inception != inception:
        raise StudyDataError(f"scan record {scan.inception} is not the note's inception {inception}")
    path = list(chains[by_day[inception]:])
    if max_days is not None:
        path = path[: int(max_days) + 1]
    calendar = calendar_for(chains)
    terms = build_terms(inception)
    s0 = float(path[0].spot)
    barrier = KO_PCT * s0
    rate_curve = FlatRateCurve(RATE)
    life = NoteLife(book_note(coupon, s0, terms, ki_pct), inception)
    ko_direction = KoDirection()
    ladder = np.asarray(LADDER, dtype=float)
    ko_models = [m for m in MODELS.values() if m.window == "ko" and not m.disable_next_ko]
    ki_models = [m for m in MODELS.values() if m.window == "ki"]

    days: List[Dict[str, Any]] = []
    solves = ko_window_days = ki_window_days = 0
    ko_date: Optional[date] = None
    ki_date: Optional[date] = None
    end_reason, alive = "truncated", True
    for dc in path:
        day = dc.day
        curve = build_curve(dc, TAIL, anchors.get(day))
        env = make_env(dc, curve.to_dividend_yield(rate_curve))
        nd = life.step(dc, env)
        if nd.knocked_in and ki_date is None:
            ki_date = day
        j = next_ko_index(terms, day) if nd.alive else None
        next_ko = None if j is None else terms.ko_dates[j]
        days_to_ko = None if next_ko is None else calendar.between(day, next_ko)
        rec: Dict[str, Any] = {
            "date": day.isoformat(), "spot": float(dc.spot), "alive": bool(nd.alive),
            "knocked_in": bool(nd.knocked_in), "engine_note_cash": float(nd.cash), "end_reason": nd.end_reason,
            "next_ko": None if next_ko is None else next_ko.isoformat(), "days_to_ko": days_to_ko,
            "gap_ko": None, "ko_dir": None, "states": {},
        }
        days.append(rec)
        if not nd.alive:
            alive, end_reason = False, str(nd.end_reason)
            if nd.end_reason == "ko":
                ko_date = day
            break
        if price_days is not None and day not in price_days:
            rec["unpriced"] = True
            continue

        spots = dc.spot * (1.0 + ladder)
        states = rec["states"]
        states["true"] = solve_ladder(engine, nd.product, env, spots, life.lifecycle, label=f"true {day}")
        solves += 1
        if in_ko_window(dc.spot, s0, days_to_ko):
            ko_window_days += 1
            surv = MODELS["surv"]
            states["surv"] = model_ladder(engine, life, surv, terms, s0, coupon, day, inception, env, spots, None, ki_pct)
            solves += 1
            payoff = ko_cash(nd.product, env, next_ko, day, terms)
            gap = gap_ko_engine(payoff, next_ko, day, spots, states["surv"]["engine_price"], barrier)
            rec["gap_ko"] = gap
            rec["ko_dir"] = ko_direction.resolve(next_ko, gap)
            if rec["ko_dir"] != 0:
                for model in ko_models:
                    states[model.id] = model_ladder(engine, life, model, terms, s0, coupon, day, inception, env,
                                                    spots, rec["ko_dir"], ki_pct)
                    solves += 1
        if in_ki_window(dc.spot, s0, nd.knocked_in, ki_pct):
            ki_window_days += 1
            for model in ki_models:
                states[model.id] = model_ladder(engine, life, model, terms, s0, coupon, day, inception, env,
                                                spots, None, ki_pct)
                solves += 1

    natural = (not alive) or path[-1].day == chains[-1].day
    last_day = days[-1]["date"]
    _check_against_scan(scan, ko_date, ki_date, end_reason, date.fromisoformat(last_day), path[-1].day, natural)
    meta = {
        "book": book, "ki_pct": float(ki_pct), "coupon": float(coupon), "solves": solves, "seconds": time.perf_counter() - started,
        "ko_window_days": ko_window_days, "ki_window_days": ki_window_days, "live_days": len(days),
        "end_reason": end_reason, "end_date": last_day, "ko_date": None if ko_date is None else ko_date.isoformat(),
        "ki_date": None if ki_date is None else ki_date.isoformat(), "tail": TAIL,
        "max_days": max_days, "price_days": None if price_days is None else sorted(d.isoformat() for d in price_days),
        "ladder": list(LADDER),
    }
    return NoteStates(inception, float(coupon), days, meta, book, float(ki_pct))


# ---------------------------------------------------------------------------
# Fingerprint and the state files
# ---------------------------------------------------------------------------


def _file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def chain_digest(chain_path: Optional[Path] = None) -> str:
    """Content hash of the chain pickle (the pass's market data)."""
    path = CHAIN_PATH if chain_path is None else Path(chain_path)
    if not path.exists():
        raise StudyDataError(f"{path} missing: run 01_data.py first")
    return _file_digest(path)


def library_versions() -> Dict[str, str]:
    """python / numpy / scipy / quantark versions, part of the fingerprint."""
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "quantark": quantark.__version__}


def pricing_fingerprint(*, pkg_dir: Optional[Path] = None, s2ts_dir: Optional[Path] = None,
                        data_digest: Optional[str] = None) -> str:
    """Hash of everything the recorded states depend on: ``gpin/{__init__,models,pricing}.py``, the imported s2ts
    files, the pre-registration's ``models`` / ``windows`` / ``note`` / ``data`` / ``books`` blocks, the engine parameters,
    the python / numpy / scipy / quantark versions and the chain's content hash. Ledger-side code and git HEAD are not part of it."""
    from quantark.asset.equity.param import QuadV2Params

    pkg_dir = Path(__file__).resolve().parent if pkg_dir is None else Path(pkg_dir)
    s2ts_dir = (STUDY_DIR.parent / "snowball_2y_tail_stability" / "s2ts") if s2ts_dir is None else Path(s2ts_dir)
    prereg = load_prereg()
    parts = {
        "gpin": {name: _file_digest(pkg_dir / name) for name in GPIN_FILES},
        "s2ts": {name: _file_digest(s2ts_dir / name) for name in S2TS_FILES},
        "prereg": {k: prereg[k] for k in ("models", "windows", "note", "data", "books")},
        "engine": repr(QuadV2Params()),
        "versions": library_versions(),
        "data": chain_digest() if data_digest is None else data_digest,
    }
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def states_path(out_dir: Path, inception: date, book: str = "ki75") -> Path:
    """``out_dir/<book>__<inception>.json``."""
    return Path(out_dir) / f"{book}__{inception.isoformat()}.json"


def write_states_atomic(path: Path, payload: Any) -> None:
    """Atomic, fsynced JSON write (no NaN may be written)."""
    write_json_atomic(path, payload)


def read_current(path: Path, note: str, fingerprint: str, coupon: float) -> Optional[Dict[str, Any]]:
    """The state file at ``path`` when it parses, is note ``note``'s (``"<book>:<inception>"``), carries
    ``fingerprint`` and the current ``coupon`` and was a complete run (not written with ``max_days`` or
    ``price_days``, which diagnostic runs record in ``meta``); else ``None``."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not (isinstance(data, dict) and data.get("note") == note and data.get("fingerprint") == fingerprint
            and isinstance(data.get("days"), list) and isinstance(data.get("coupon"), (int, float))):
        return None
    meta = data.get("meta")
    if not isinstance(meta, dict) or "max_days" not in meta or "price_days" not in meta:
        return None
    if meta["max_days"] is not None or meta["price_days"] is not None:
        return None
    if not is_close(float(data["coupon"]), float(coupon)):
        return None
    return data


def price_one(inception: date, coupon: float, chains: Sequence, anchors: Mapping[date, float], scan: NoteScan,
              out_dir: Path, fingerprint: str, *, ki_pct: float = KI_PCT, engine=None,
              price_days: Optional[Set[date]] = None, max_days: Optional[int] = None) -> Dict[str, Any]:
    """Price one note into ``out_dir`` unless a complete file for this coupon stamped ``fingerprint`` is already
    there. Returns ``{"note", "inception", "reused", "seconds", "solves", "live_days"}``."""
    book = book_of(ki_pct)
    nid = note_id(book, inception)
    path = states_path(out_dir, inception, book)
    cached = read_current(path, nid, fingerprint, coupon)
    if cached is not None:
        meta = cached.get("meta", {})
        return {"note": nid, "inception": inception.isoformat(), "reused": True, "seconds": 0.0,
                "solves": int(meta.get("solves", 0)), "live_days": len(cached["days"])}
    started = time.perf_counter()
    states = price_note(inception, coupon, chains, anchors, scan, ki_pct=ki_pct, engine=engine,
                        price_days=price_days, max_days=max_days)
    write_states_atomic(path, states.to_json(fingerprint))
    return {"note": nid, "inception": inception.isoformat(), "reused": False,
            "seconds": time.perf_counter() - started, "solves": int(states.meta["solves"]),
            "live_days": len(states.days)}
