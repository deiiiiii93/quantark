"""Pricing-free lifecycle scan of every candidate inception, the pin-event sample and the train/test split.

``scan`` re-implements the note's lifecycle on closes alone; the pricing pass (T3) cross-checks every priced note
against the library's tracker and fails closed on any difference. The rules below are the library's, read from
``quantark/asset/equity/lifecycle/autocallable.py`` (``AutocallableLifecycleTracker.observe`` and
``settle_maturity_if_due``) and ``s2ts.note.NoteLife.step``:

* the inception close is the first valuation and is not observed;
* on every later close the KO schedule is checked first: a KO record is due when elapsed calendar days >= its
  due days (here: its date, every chain day being a trading day) and fires when ``spot >= barrier``
  (``_barrier_hit``, ``is_reverse = False``); a fired KO ends the note and the same close is NOT checked for KI;
* otherwise the daily KI record fires when ``spot <= barrier`` and latches (``mark_ki``); KO stays active after a
  knock-in (``disable_ko_after_ki`` is False);
* the last KO observation is the maturity date: a note alive after it has matured (``end_reason`` ``maturity``);
  a spot at or above the barrier on that day is a KO, not a maturity;
* a note alive at the last chain day is ``truncated``.

The raw ``>=`` / ``<=`` against the barrier mirror the library's operators exactly (no tolerance); the sample
thresholds (2%, 3%, 10%) are tolerant comparisons through ``quantark.util.numerical``.

Books (pre-registration ``books``): ``ki75`` is the 2Y product, one scan record per chain day; ``seas6`` is the same
75% product in its last six months, one booking per ISO week (the first anchored chain day of the week) with 18
months elapsed, its KI level 7% below the booking-day spot and the strike and KO level at KI / 0.75. A booking is
scanned by the same lifecycle rules on its own terms (``book_terms``) and levels (``book_levels``): the first KO
observation is at month ``first_ko_month`` (the booking day itself is never observed), the last is the maturity. A
seasoned book records no KO pin events (they are measured on ``ki75`` only).

Seller side (spec D13): events are described as the desk that sold the note sees them; this module only dates them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from quantark.util.numerical import is_close

from s2ts import ACT
from s2ts.chain import sse_trading_days
from s2ts.note import Terms, _is_trading, _next_trading, add_months, build_terms

from . import StudyDataError, load_prereg

_PREREG = load_prereg()
KO_PCT: float = float(_PREREG["note"]["ko_pct"])
KO_PRIOR_GAP: float = float(_PREREG["sample"]["ko_prior_gap"])
KI_NEAR: float = float(_PREREG["sample"]["ki_near"])
KI_BAND: float = float(_PREREG["windows"]["ki_band"])
KO_NOTES_PER_WEEK: int = int(_PREREG["sample"]["ko_notes_per_week"])
KI_NOTES_PER_MONTH: int = int(_PREREG["sample"]["ki_notes_per_month"])


@dataclass(frozen=True)
class BookSpec:
    """One book of the study (pre-registration ``books``). ``ki_pct``: KI level over the strike; ``first_ko_month`` /
    ``maturity_months``: the KO observations are the month anniversaries ``first_ko_month..maturity_months`` of the
    booking day, the last one the maturity; ``elapsed_months``: months the note has already run (coupon accrued over
    ``elapsed + maturity`` months); ``ki_below_spot``: the KI level is the booking-day spot over ``1 + ki_below_spot``
    (``None``: the strike is the booking-day spot); ``booking``: ``sample`` (headline and pin sets) or ``weekly`` (the
    first anchored chain day of each ISO week)."""

    id: str
    ki_pct: float
    first_ko_month: int
    maturity_months: int
    elapsed_months: int
    ki_below_spot: Optional[float]
    booking: str


def _book_spec(book_id: str, raw: Mapping) -> BookSpec:
    below = raw["ki_below_spot"]
    return BookSpec(book_id, float(raw["ki_pct"]), int(raw["first_ko_month"]), int(raw["maturity_months"]),
                    int(raw["elapsed_months"]), None if below is None else float(below), str(raw["booking"]))


#: The books of the study, read from the pre-registration; ``ki75`` is the 2Y product.
BOOKS: Dict[str, BookSpec] = {k: _book_spec(k, v) for k, v in _PREREG["books"].items()}
PRODUCT_BOOK = "ki75"
SPLIT_DATE: date = date.fromisoformat(_PREREG["split_date"])
DATA_END: date = date.fromisoformat(_PREREG["data"]["end"])


@dataclass(frozen=True)
class KoEvent:
    """A KO pin event: the note is alive going into the observation at ``obs_date`` and the prior close was within
    the pre-registered gap of the barrier. ``prior_gap`` = ``|S(T-1) / B - 1|``; ``fired``: ``S(T) >= B``."""

    inception: date
    obs_date: date
    prior_gap: float
    fired: bool


@dataclass(frozen=True)
class KiApproach:
    """A maximal run of consecutive closes with the note alive, not yet knocked in and ``S / K - 1 <= band`` that
    holds a close with ``S / K - 1 <= near`` or ends in a knock-in (that day is the run's last). Dated by ``end``."""

    inception: date
    start: date
    end: date
    knocked_in: bool


@dataclass(frozen=True)
class NoteScan:
    """One candidate inception's lifecycle on the closes. ``end_reason``: ``ko`` / ``maturity`` / ``truncated``;
    ``end_date`` the last close of the note's life in the data; ``ko_observations`` the KO observations the note
    was alive for (including the firing one); ``ko_events`` only the pin events (``ki75``); ``book``: the book the
    record belongs to."""

    inception: date
    anchored: bool
    ko_date: Optional[date]
    ki_date: Optional[date]
    end_reason: str
    end_date: date
    ko_observations: int
    ko_events: Tuple[KoEvent, ...]
    ki_approaches: Tuple[KiApproach, ...]
    book: str


Event = Union[KoEvent, KiApproach]


@dataclass(frozen=True)
class Sample:
    """``headline``: real inceptions (first chain day of a quarter, anchored); ``pin``: the pin set P;
    ``events``: every KO pin event (``ki75`` only) and KI approach of every note in ``headline`` and ``pin``;
    ``reasons``: why each pin note was selected."""

    headline: Tuple[date, ...]
    pin: Tuple[date, ...]
    events: Tuple[Event, ...]
    reasons: Dict[date, Tuple[str, ...]]


def _spec(book: str) -> BookSpec:
    try:
        return BOOKS[book]
    except KeyError:
        raise StudyDataError(f"unknown book {book!r}; books are {sorted(BOOKS)}") from None


def note_id(book: str, inception: date) -> str:
    """``"<book>:<inception ISO date>"``, the identity of a note."""
    _spec(book)
    return f"{book}:{inception.isoformat()}"


def parse_note_id(text: str) -> Tuple[str, date]:
    """Inverse of ``note_id``; ``StudyDataError`` on anything else."""
    book, sep, iso = text.partition(":")
    if not sep or book not in BOOKS:
        raise StudyDataError(f"{text!r} is not a note id '<book>:<YYYY-MM-DD>' of a known book")
    try:
        return book, date.fromisoformat(iso)
    except ValueError:
        raise StudyDataError(f"{text!r} is not a note id '<book>:<YYYY-MM-DD>' of a known book") from None


def book_terms(book: str, inception: date) -> Terms:
    """The schedule of ``book``'s note booked on ``inception`` (an SSE trading day): KO on the month anniversaries
    ``first_ko_month..maturity_months`` rolled to the next SSE trading day, the last = the maturity; KI on every SSE
    trading day after the booking day up to the maturity; ACT/365F times from the booking day. ``ki75`` is
    ``s2ts.note.build_terms`` itself."""
    spec = _spec(book)
    if book == PRODUCT_BOOK:
        return build_terms(inception)
    if not _is_trading(inception):
        raise StudyDataError(f"inception {inception} is not an SSE trading day")
    maturity = _next_trading(add_months(inception, spec.maturity_months))
    ko_dates = sorted({min(_next_trading(add_months(inception, m)), maturity)
                       for m in range(spec.first_ko_month, spec.maturity_months + 1)})
    if ko_dates[-1] != maturity:
        raise StudyDataError("last KO observation must be the maturity date")
    ki_dates: List[date] = []
    d = inception + timedelta(days=1)
    while d <= maturity:
        if _is_trading(d):
            ki_dates.append(d)
        d += timedelta(days=1)

    def years(x: date) -> float:
        return (x - inception).days / ACT

    return Terms(inception=inception, maturity_date=maturity, maturity_years=years(maturity),
                 ko_dates=tuple(ko_dates), ko_times=tuple(years(x) for x in ko_dates),
                 ki_dates=tuple(ki_dates), ki_times=tuple(years(x) for x in ki_dates))


def book_levels(book: str, spot0: float) -> Tuple[float, float, float]:
    """``(strike, ko_level, ki_level)`` in index points of ``book``'s note booked at ``spot0``: ``ki75`` the strike
    and KO level are ``spot0`` (``ko_pct`` 100%) and the KI level ``ki_pct x spot0``; a seasoned book puts the KI
    level ``ki_below_spot`` below ``spot0`` and the strike and KO level at KI / ``ki_pct``."""
    spec = _spec(book)
    if spec.ki_below_spot is None:
        strike = float(spot0)
        return strike, KO_PCT * strike, spec.ki_pct * strike
    ki = float(spot0) / (1.0 + spec.ki_below_spot)
    strike = ki / spec.ki_pct
    return strike, KO_PCT * strike, ki


def _within(value: float, bound: float) -> bool:
    """``value <= bound`` with the repo's tolerance (a ratio built from prices lands a hair off a round bound)."""
    return value < bound or is_close(value, bound)


def scan(chains: Sequence, anchors: Mapping[date, float], book: str) -> List[NoteScan]:
    """One ``NoteScan`` per candidate booking of ``book``; no pricing. Reads ``.day`` and ``.spot`` only.
    ``ki75``: every chain day that is an SSE trading day (the unrestricted counts of spec D5 need the unanchored days;
    ``anchored`` says which can be priced). ``seas6``: the first anchored chain day of each ISO week, all anchored.
    Lifecycle on ``book``'s terms and levels.

    The chain must hold every trading day between its first and last (fails closed).
    """
    spec = _spec(book)
    days = [c.day for c in chains]
    if not days:
        return []
    calendar = [d.date() for d in sse_trading_days(days[0].year, days[-1].year)]
    expected = [d for d in calendar if days[0] <= d <= days[-1]]
    if days != expected:
        bad = next((d for d, e in zip(days, expected) if d != e), None)
        raise StudyDataError(f"chain days are not the SSE trading days of {days[0]}..{days[-1]} (first difference: {bad})")
    spots = [float(c.spot) for c in chains]
    if spec.booking == "weekly":
        first: Dict[Tuple[int, int], int] = {}
        for i, d in enumerate(days):
            if d in anchors:
                first.setdefault(d.isocalendar()[:2], i)
        starts = sorted(first.values())
    else:
        starts = list(range(len(days)))
    return [_scan_one(i, days, spots, days[i] in anchors, book) for i in starts]


def _scan_one(start: int, days: Sequence[date], spots: Sequence[float], anchored: bool, book: str) -> NoteScan:
    inception, s0 = days[start], spots[start]
    terms = book_terms(book, inception)
    ko_dates = set(terms.ko_dates)
    _, ko_barrier, ki_barrier = book_levels(book, s0)
    pins = book == PRODUCT_BOOK                 # KO pin events are measured on the product's book only

    ko_date: Optional[date] = None
    ki_date: Optional[date] = None
    end_reason, end_date = "truncated", days[start]
    n_obs = 0
    ko_events: List[KoEvent] = []
    approaches: List[KiApproach] = []
    run: Optional[List] = None  # [start, last, touched_near]

    def close_run(knocked_in: bool) -> None:
        nonlocal run
        if run is not None and (run[2] or knocked_in):
            approaches.append(KiApproach(inception, run[0], run[1], knocked_in))
        run = None

    for j in range(start + 1, len(days)):
        day, spot = days[j], spots[j]
        end_date = day
        if day in ko_dates:
            n_obs += 1
            gap = abs(spots[j - 1] / ko_barrier - 1.0)
            fired = spot >= ko_barrier
            if pins and _within(gap, KO_PRIOR_GAP):
                ko_events.append(KoEvent(inception, day, gap, fired))
            if fired:
                close_run(False)
                ko_date, end_reason = day, "ko"
                break
        if ki_date is None:
            knocked = spot <= ki_barrier
            ratio = spot / ki_barrier - 1.0
            if knocked:
                ki_date = day
                if run is None:
                    run = [day, day, True]
                run[1] = day
                close_run(True)
            elif _within(ratio, KI_BAND):
                if run is None:
                    run = [day, day, False]
                run[1] = day
                run[2] = run[2] or _within(ratio, KI_NEAR)
            else:
                close_run(False)
        if day == terms.maturity_date:
            end_reason = "maturity"
            break
    close_run(False)
    return NoteScan(inception, anchored, ko_date, ki_date, end_reason, end_date, n_obs,
                    tuple(ko_events), tuple(approaches), book)


def slice_of(event_date: date) -> str:
    """``train`` before the pre-registered split date, ``test`` from it."""
    return "train" if event_date < SPLIT_DATE else "test"


def week_key(event_date: date) -> str:
    """ISO week of the event date, ``YYYY-Www`` with the ISO week-year (the bootstrap cluster)."""
    iso = event_date.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _quarter(d: date) -> Tuple[int, int]:
    return d.year, (d.month - 1) // 3


def _spaced_positions(n: int, cap: int) -> List[int]:
    """``cap`` evenly spaced positions of ``n`` items, endpoints included, rounded half up (all of them if n <= cap)."""
    if n <= cap:
        return list(range(n))
    if cap == 1:
        return [0]
    return [(2 * i * (n - 1) + (cap - 1)) // (2 * (cap - 1)) for i in range(cap)]


def select(scans: Sequence[NoteScan], book: str) -> Sample:
    """The sample of spec D5 for ``book``, deterministic and independent of the order of ``scans`` (which must be
    that book's scan); anchored inceptions only.

    ``ki75``. Headline: the first chain day of each quarter, kept when anchored. KO pin set: per ISO week of the
    observation, the ``KO_NOTES_PER_WEEK`` pin events with the smallest prior gap (ties: earlier inception). KI pin
    set: per calendar month of the approach date, ``KI_NOTES_PER_MONTH`` notes at evenly spaced positions of the
    inception-sorted list of notes with an approach that month. Events: every KO pin event and KI approach of every
    selected note.

    ``seas6`` (every booking is priced, none is picked for having a pin): every anchored weekly booking is
    ``headline``, no pin set, no reasons; events: every KI approach of every booking, no KO pin events.
    """
    spec = _spec(book)
    for s in scans:
        if s.book != book:
            raise StudyDataError(f"scan of {s.inception} is for book {s.book}, not {book}")
    ordered = sorted(scans, key=lambda s: s.inception)
    anchored = [s for s in ordered if s.anchored]
    if spec.booking == "weekly":
        events = [a for s in anchored for a in sorted(s.ki_approaches, key=lambda a: a.end)]
        return Sample(tuple(s.inception for s in anchored), (), tuple(events), {})

    first_of_quarter: Dict[Tuple[int, int], NoteScan] = {}
    for s in ordered:
        first_of_quarter.setdefault(_quarter(s.inception), s)
    headline = tuple(s.inception for s in first_of_quarter.values() if s.anchored)

    reasons: Dict[date, List[str]] = {}

    by_week: Dict[str, List[KoEvent]] = {}
    for s in anchored:
        for ev in s.ko_events:
            by_week.setdefault(week_key(ev.obs_date), []).append(ev)
    for week, events in by_week.items():
        events.sort(key=lambda e: (e.prior_gap, e.inception))
        for rank, ev in enumerate(events[:KO_NOTES_PER_WEEK], start=1):
            reasons.setdefault(ev.inception, []).append(f"ko {week} rank {rank}")

    by_month: Dict[str, List[date]] = {}
    for s in anchored:
        for month in sorted({f"{a.end:%Y-%m}" for a in s.ki_approaches}):
            by_month.setdefault(month, []).append(s.inception)
    for month, notes in by_month.items():
        for pos in _spaced_positions(len(notes), KI_NOTES_PER_MONTH):
            reasons.setdefault(notes[pos], []).append(f"ki {month} position {pos + 1} of {len(notes)}")

    pin = tuple(sorted(reasons))
    selected = set(headline) | set(pin)
    events: List[Event] = []
    for s in ordered:
        if s.inception in selected:
            events.extend(sorted(s.ko_events, key=lambda e: e.obs_date))
            events.extend(sorted(s.ki_approaches, key=lambda a: a.end))
    return Sample(headline, pin, tuple(events), {k: tuple(sorted(v)) for k, v in sorted(reasons.items())})


def event_date(event: Event) -> date:
    """The date an event is dated by: the KO observation date, the approach's last day."""
    return event.obs_date if isinstance(event, KoEvent) else event.end


def counts(scans: Sequence[NoteScan]) -> Dict[str, int]:
    """Unrestricted-style counts over ``scans``: inceptions, live KO observations, pin events, fired pin events,
    notes knocked in, KI approaches."""
    ko = [e for s in scans for e in s.ko_events]
    return {
        "inceptions": len(scans),
        "ko_observations": sum(s.ko_observations for s in scans),
        "ko_pin_events": len(ko),
        "ko_pin_fired": sum(1 for e in ko if e.fired),
        "notes_knocked_in": sum(1 for s in scans if s.ki_date is not None),
        "ki_approaches": sum(len(s.ki_approaches) for s in scans),
    }
