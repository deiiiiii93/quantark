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

Seller side (spec D13): events are described as the desk that sold the note sees them; this module only dates them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

from quantark.util.numerical import is_close

from s2ts.chain import sse_trading_days
from s2ts.note import build_terms

from . import StudyDataError, load_prereg

_PREREG = load_prereg()
KO_PCT: float = float(_PREREG["note"]["ko_pct"])
KI_PCT: float = float(_PREREG["note"]["ki_pct"])
KO_PRIOR_GAP: float = float(_PREREG["sample"]["ko_prior_gap"])
KI_NEAR: float = float(_PREREG["sample"]["ki_near"])
KI_BAND: float = float(_PREREG["windows"]["ki_band"])
KO_NOTES_PER_WEEK: int = int(_PREREG["sample"]["ko_notes_per_week"])
KI_NOTES_PER_MONTH: int = int(_PREREG["sample"]["ki_notes_per_month"])
LIFTED_KI_NOTES_PER_WEEK: int = int(_PREREG["sample"]["lifted_ki_notes_per_week"])
#: The books of the study: a book is a KI level (pre-registration ``books``); ``ki75`` is the product.
BOOKS: Dict[str, float] = {k: float(v) for k, v in _PREREG["books"].items()}
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
    was alive for (including the firing one); ``ko_events`` only the pin events."""

    inception: date
    anchored: bool
    ko_date: Optional[date]
    ki_date: Optional[date]
    end_reason: str
    end_date: date
    ko_observations: int
    ko_events: Tuple[KoEvent, ...]
    ki_approaches: Tuple[KiApproach, ...]
    ki_pct: float = KI_PCT


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


def note_id(book: str, inception: date) -> str:
    """``"<book>:<inception ISO date>"``, the identity of a note (a book is a KI level)."""
    if book not in BOOKS:
        raise StudyDataError(f"unknown book {book!r}; books are {sorted(BOOKS)}")
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


def _within(value: float, bound: float) -> bool:
    """``value <= bound`` with the repo's tolerance (a ratio built from prices lands a hair off a round bound)."""
    return value < bound or is_close(value, bound)


def scan(chains: Sequence, anchors: Mapping[date, float], ki_pct: float = KI_PCT) -> List[NoteScan]:
    """One ``NoteScan`` per chain day that is an SSE trading day; no pricing. Reads ``.day`` and ``.spot`` only.
    ``ki_pct`` is the book's KI level (the KO level is the note's); the KI window and approaches follow it.

    ``anchored``: the day has a carry anchor (it can be priced); unanchored days are kept so the unrestricted
    counts can be reproduced. The chain must hold every trading day between its first and last (fails closed).
    """
    days = [c.day for c in chains]
    if not days:
        return []
    calendar = [d.date() for d in sse_trading_days(days[0].year, days[-1].year)]
    expected = [d for d in calendar if days[0] <= d <= days[-1]]
    if days != expected:
        bad = next((d for d, e in zip(days, expected) if d != e), None)
        raise StudyDataError(f"chain days are not the SSE trading days of {days[0]}..{days[-1]} (first difference: {bad})")
    spots = [float(c.spot) for c in chains]
    return [_scan_one(i, days, spots, i_day in anchors, ki_pct) for i, i_day in enumerate(days)]


def _scan_one(start: int, days: Sequence[date], spots: Sequence[float], anchored: bool, ki_pct: float) -> NoteScan:
    inception, s0 = days[start], spots[start]
    terms = build_terms(inception)
    ko_dates = set(terms.ko_dates)
    ko_barrier, ki_barrier = KO_PCT * s0, ki_pct * s0

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
            if _within(gap, KO_PRIOR_GAP):
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
                    tuple(ko_events), tuple(approaches), float(ki_pct))


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


def select(scans: Sequence[NoteScan], book: str = PRODUCT_BOOK) -> Sample:
    """The sample of spec D5 for ``book``, deterministic and independent of the order of ``scans`` (which must be
    that book's scan); anchored inceptions only.

    Headline: the first chain day of each quarter, kept when anchored. ``ki75`` KO pin set: per ISO week of the
    observation, the ``KO_NOTES_PER_WEEK`` pin events with the smallest prior gap (ties: earlier inception). KI pin
    set: per calendar month of the approach date, ``KI_NOTES_PER_MONTH`` notes at evenly spaced positions of the
    inception-sorted list of notes with an approach that month; a lifted book samples per ISO week of the approach
    date instead (the bootstrap's cluster), ``LIFTED_KI_NOTES_PER_WEEK`` notes at evenly spaced positions of the
    inception-sorted notes with an approach dated in that week, all of them when fewer.
    Events: every KI approach of every selected note, and for ``ki75`` every KO pin event; a lifted book has no
    KO pin set and records no KO pin events.
    """
    if book not in BOOKS:
        raise StudyDataError(f"unknown book {book!r}; books are {sorted(BOOKS)}")
    for s in scans:
        if not is_close(s.ki_pct, BOOKS[book]):
            raise StudyDataError(f"scan of {s.inception} is for KI {s.ki_pct}, not book {book} ({BOOKS[book]})")
    product = book == PRODUCT_BOOK
    ki_cap = KI_NOTES_PER_MONTH if product else LIFTED_KI_NOTES_PER_WEEK
    ordered = sorted(scans, key=lambda s: s.inception)
    first_of_quarter: Dict[Tuple[int, int], NoteScan] = {}
    for s in ordered:
        first_of_quarter.setdefault(_quarter(s.inception), s)
    headline = tuple(s.inception for s in first_of_quarter.values() if s.anchored)

    anchored = [s for s in ordered if s.anchored]
    reasons: Dict[date, List[str]] = {}

    by_week: Dict[str, List[KoEvent]] = {}
    for s in anchored if product else ():
        for ev in s.ko_events:
            by_week.setdefault(week_key(ev.obs_date), []).append(ev)
    for week, events in by_week.items():
        events.sort(key=lambda e: (e.prior_gap, e.inception))
        for rank, ev in enumerate(events[:KO_NOTES_PER_WEEK], start=1):
            reasons.setdefault(ev.inception, []).append(f"ko {week} rank {rank}")

    by_month: Dict[str, List[date]] = {}
    for s in anchored:
        for month in sorted({f"{a.end:%Y-%m}" if product else week_key(a.end) for a in s.ki_approaches}):
            by_month.setdefault(month, []).append(s.inception)
    for month, notes in by_month.items():
        for pos in _spaced_positions(len(notes), ki_cap):
            reasons.setdefault(notes[pos], []).append(f"ki {month} position {pos + 1} of {len(notes)}")

    pin = tuple(sorted(reasons))
    selected = set(headline) | set(pin)
    events: List[Event] = []
    for s in ordered:
        if s.inception in selected:
            if product:
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
