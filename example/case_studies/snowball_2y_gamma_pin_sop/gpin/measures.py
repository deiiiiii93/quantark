"""Measures, training selection and the verdict (spec C8, D10, D11). Seller view: every P&L is the seller's, in bp
of notional, net of costs.

On one ledger run (``gpin.ledger.Run``): ``event_windows`` (the D10 window P&L per event), ``gamma_carry`` (what is
carried into a KO fixing), ``life_stats`` (the whole-life sufficient statistics) and ``summarise_run``, the stored form
of a run. On event P&L: ``es``, ``cluster_bootstrap``, ``tail_test``, ``life_test`` / ``whole_life`` and ``verdict``.
``select_on_train`` is the training rule of the registry. ``run_note`` runs every technique on one note and keeps the
summaries (one file of the runs directory, written by ``04_ledger.py``); ``training_selection`` and ``tables`` aggregate
such files into the selection and into everything the report prints, so the report can render from any subset of
files without running the ledger. ``stamp`` / ``check_selection`` are the three hashes and the guard of the test slice.

Conventions fixed here (each is in a test): a trading day is a day of the chain (the SSE calendar when none is
given); the note's day-0 row is in no event window and not in the whole-life standard deviation, and is in the
terminal P&L; the standard deviation is the sample one (ddof 1) of the concatenated daily rows; an event is left out
of a clock's tables when a row of its window carries ``last_day_fixing``; an event or a KO observation belongs to the
slice of its date.

Ledger-side code: no engine, no market data of its own. Every threshold is read from the pre-registration.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.util.numerical import Tolerance, is_close, is_valid_number, is_zero, safe_divide, safe_sqrt

from s2ts import MO_MULT, NOTIONAL
from s2ts.chain import sse_trading_days
from s2ts.measures import holm

from . import PREREG_PATH, StudyDataError, load_prereg, prereg_hash
from . import black
from .board import DayBoard
from .events import note_id, parse_note_id
from .ledger import Run, run_technique
from .techniques import Technique, grid_id

#: CNY per note to bp of notional.
BP = 1.0e4 / NOTIONAL


# ---------------------------------------------------------------------------
# The pre-registered numbers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rules:
    """The pre-registered numbers of the measures and the verdict (``statistics``, ``verdict``, ``windows``)."""

    es_alpha: float
    bootstrap_n: int
    seed: int
    p_max: float
    std_ratio_max: float
    terminal_slack_bp: float
    min_events: int
    min_weeks: int
    ko_days: int

    @classmethod
    def from_prereg(cls, prereg: Mapping[str, Any]) -> "Rules":
        """Read from a loaded pre-registration."""
        s, v = prereg["statistics"], prereg["verdict"]
        return cls(float(s["es_alpha"]), int(s["bootstrap_n"]), int(s["seed"]), float(s["p_max"]),
                   float(v["std_ratio_max"]), float(v["terminal_slack_bp"]), int(v["min_events"]),
                   int(v["min_weeks"]), int(prereg["windows"]["ko_days"]))


@lru_cache(maxsize=1)
def default_rules() -> Rules:
    """The rules of the committed pre-registration (read once)."""
    return Rules.from_prereg(load_prereg())


# ---------------------------------------------------------------------------
# ES10 and the week-clustered bootstrap
# ---------------------------------------------------------------------------


def _alpha_fraction(alpha: float) -> Fraction:
    """``alpha`` as an exact fraction (0.10 -> 1/10), so the tail count is integer arithmetic: ``0.1 x n`` in floating
    point can land a hair above a whole number and a ceiling would then take one event too many."""
    frac = Fraction(str(float(alpha)))
    if not 0 < frac <= 1:
        raise ValidationError(f"the ES level must be in (0, 1], got {alpha!r}")
    return frac


def tail_count(n: int, alpha: float) -> int:
    """``k = max(1, ceil(alpha x n))``: how many of ``n`` values the expected shortfall averages."""
    if n < 1:
        raise ValidationError("the expected shortfall of no value is undefined")
    frac = _alpha_fraction(alpha)
    return max(1, -(-(int(n) * frac.numerator) // frac.denominator))


def es(values: Sequence[float], alpha: Optional[float] = None) -> float:
    """Expected shortfall: the mean of the ``k = max(1, ceil(alpha x n))`` smallest values (``alpha`` defaults to the
    pre-registered 0.10). Fails closed on no value and on a value that is not a finite number."""
    alpha = default_rules().es_alpha if alpha is None else float(alpha)
    vals = sorted(float(v) for v in values)
    if any(not is_valid_number(v) for v in vals):
        raise ValidationError("the expected shortfall needs finite values")
    k = tail_count(len(vals), alpha)
    return float(sum(vals[:k]) / k)


def _es_weighted(values: np.ndarray, weights: np.ndarray, alpha: Fraction) -> np.ndarray:
    """ES of every replicate at once: ``values`` (E,), ``weights`` (R, E) how many times each event is in each
    replicate. Per replicate ``k = max(1, ceil(alpha x total weight))`` and the mean of the k smallest values, each
    event counted with its multiplicity."""
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[:, order]
    total = w.sum(axis=1)
    k = np.maximum(1, -(-(total * alpha.numerator) // alpha.denominator))
    before = np.cumsum(w, axis=1) - w
    take = np.clip(k[:, None] - before, 0, w)
    return (take * v).sum(axis=1) / k


def close_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``quantark.util.numerical.is_close`` element by element: ``|a - b| <= max(RELATIVE x max(|a|, |b|),
    ABSOLUTE)`` with the repo's constants. Symmetric in its arguments, unlike ``numpy.isclose``, whose bound is
    relative to its second argument alone."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.abs(a - b) <= np.maximum(Tolerance.RELATIVE * np.maximum(np.abs(a), np.abs(b)), Tolerance.ABSOLUTE)


def bootstrap_weights(week_by_event: Mapping[str, str], events: Sequence[str], n: int, seed: int) -> np.ndarray:
    """``(n, len(events))`` multiplicities of the cluster bootstrap: the clusters are the distinct weeks in sorted
    order (``C`` of them); replicate ``i`` takes the clusters ``default_rng(seed).integers(0, C, size=(n, C))[i]``,
    with replacement, and every event of a cluster comes with it."""
    missing = [e for e in events if e not in week_by_event]
    if missing:
        raise ValidationError(f"events without a week (the bootstrap cluster): {missing[:3]}")
    if not events:
        raise ValidationError("the bootstrap needs at least one event")
    clusters = sorted({week_by_event[e] for e in events})
    index = {c: i for i, c in enumerate(clusters)}
    draws = np.random.default_rng(int(seed)).integers(0, len(clusters), size=(int(n), len(clusters)))
    counts = np.zeros((int(n), len(clusters)), dtype=np.int64)
    np.add.at(counts, (np.arange(int(n))[:, None], draws), 1)
    return counts[:, [index[week_by_event[e]] for e in events]]


def cluster_bootstrap(tech_by_event: Mapping[str, float], base_by_event: Mapping[str, float],
                      week_by_event: Mapping[str, str], n: int = 10_000, seed: int = 20261112,
                      alpha: Optional[float] = None) -> float:
    """The pre-registered p of the tail test: the share of ``n`` week-clustered resamples on which ``ES(technique) -
    ES(base) <= 0``, both arms on the same resampled events (``bootstrap_weights``). ``n`` and ``seed`` default to the
    pre-registered 10,000 and 20261112. Two ES equal by the repo's ``is_close`` (``close_array``) are a tie and count
    as no gain
    (a technique that never acts on the events differs from the baseline by rounding only). Fails closed when the
    arms are not on the same events."""
    if set(tech_by_event) != set(base_by_event):
        raise ValidationError("the technique and the baseline must be measured on the same events")
    alpha_frac = _alpha_fraction(default_rules().es_alpha if alpha is None else alpha)
    events = sorted(tech_by_event)
    weights = bootstrap_weights(week_by_event, events, n, seed)
    tech = np.asarray([float(tech_by_event[e]) for e in events])
    base = np.asarray([float(base_by_event[e]) for e in events])
    if not (np.isfinite(tech).all() and np.isfinite(base).all()):
        raise ValidationError("the bootstrap needs finite event P&L")
    es_tech, es_base = _es_weighted(tech, weights, alpha_frac), _es_weighted(base, weights, alpha_frac)
    return float(np.mean((es_tech - es_base <= 0.0) | close_array(es_tech, es_base)))


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

KO, KI = "ko", "ki"


@dataclass(frozen=True)
class Event:
    """One measured event of one note. ``kind`` ``ko``: a KO pin event dated by its observation ``date``; ``ki``: a
    KI approach dated by its last day, ``start`` its first day. ``week`` is the bootstrap cluster (ISO week of the
    date), ``slice`` ``train`` / ``test`` by the date, ``hit`` whether the KO fired / the approach ended in a
    knock-in."""

    kind: str
    book: str
    inception: date
    date: date
    start: Optional[date]
    week: str
    slice: str
    hit: bool

    def __post_init__(self) -> None:
        if self.kind not in (KO, KI):
            raise ValidationError(f"an event is 'ko' or 'ki', got {self.kind!r}")
        if self.kind == KI and self.start is None:
            raise ValidationError("a KI approach needs its first day")

    @property
    def note(self) -> str:
        """``"<book>:<inception>"``."""
        return note_id(self.book, self.inception)

    @property
    def id(self) -> str:
        """``"<kind>|<note>|<date>"``: unique in the sample (a note has one observation, one approach end, a day)."""
        return f"{self.kind}|{self.note}|{self.date.isoformat()}"

    def to_json(self) -> Dict[str, Any]:
        """JSON-ready."""
        return {"kind": self.kind, "note": self.note, "date": self.date.isoformat(),
                "start": None if self.start is None else self.start.isoformat(), "week": self.week,
                "slice": self.slice, "hit": self.hit}


def note_events(sample: Mapping[str, Any], note: str) -> List[Event]:
    """The events of ``note`` in a loaded ``results/events.json``, by date (a KI approach before a KO on one day)."""
    book, inception = parse_note_id(note)
    block = sample["books"][book]
    iso = inception.isoformat()
    out: List[Event] = []
    for raw in block.get("ko_events", []):
        if raw["inception"] == iso:
            out.append(Event(KO, book, inception, date.fromisoformat(raw["date"]), None, str(raw["week"]),
                             str(raw["slice"]), bool(raw["fired"])))
    for raw in block.get("ki_approaches", []):
        if raw["inception"] == iso:
            out.append(Event(KI, book, inception, date.fromisoformat(raw["date"]), date.fromisoformat(raw["start"]),
                             str(raw["week"]), str(raw["slice"]), bool(raw["knocked_in"])))
    return sorted(out, key=lambda e: (e.date, e.kind))


@lru_cache(maxsize=None)
def _sse_days(first_year: int, last_year: int) -> Tuple[date, ...]:
    return tuple(d.date() for d in sse_trading_days(first_year, last_year))


def _calendar(calendar: Optional[Sequence[date]], events: Sequence[Event]) -> Sequence[date]:
    """The trading calendar: the one given (the chain's days), else the SSE calendar around the events."""
    if calendar is not None:
        return calendar
    if not events:
        return ()
    return _sse_days(min(e.inception.year for e in events) - 1, max(e.date.year for e in events) + 1)


def _position(day: date, calendar: Sequence[date]) -> int:
    i = bisect_left(calendar, day)
    if i >= len(calendar) or calendar[i] != day:
        raise ValidationError(f"{day} is not a trading day of the calendar")
    return i


def window_dates(event: Event, calendar: Sequence[date], ko_days: Optional[int] = None) -> Tuple[date, date]:
    """``(first, last)`` dates of the event window (D10) on the sorted trading ``calendar``. KO: from the first
    trading day on which the observation is at most ``ko_days`` trading days ahead through the trading day after it.
    KI: from the approach's first day through the trading day after its last day. With no trading day after the
    event on the calendar the window ends on the event's date."""
    ko_days = default_rules().ko_days if ko_days is None else int(ko_days)
    at = _position(event.date, calendar)
    last = calendar[min(at + 1, len(calendar) - 1)]
    if event.kind == KO:
        return calendar[max(at - ko_days, 0)], last
    _position(event.start, calendar)
    return event.start, last


def _own(run: Run, events: Sequence[Event]) -> None:
    for e in events:
        if (e.book, e.inception) != (run.book, run.inception):
            raise ValidationError(f"event {e.id} is not an event of note {note_id(run.book, run.inception)}")


def _window_rows(run: Run, event: Event, calendar: Sequence[date], ko_days: Optional[int]) -> List[Mapping[str, Any]]:
    """The run's rows inside the event window, the day-0 row left out (ISO dates compare as dates)."""
    first, last = (d.isoformat() for d in window_dates(event, calendar, ko_days))
    day0 = run.inception.isoformat()
    return [r for r in run.rows if first <= r["date"] <= last and r["date"] != day0]


def event_windows(run: Run, events: Sequence[Event], calendar: Optional[Sequence[date]] = None,
                  ko_days: Optional[int] = None) -> Dict[str, float]:
    """``{event id: seller P&L in bp of notional}``: the sum of ``hedged`` over the run's rows inside each event's
    window (``window_dates``). A row that does not exist counts 0 (everything is flat); the note's day-0 row is never
    in a window (it is the model gap at the traded price, the same for every technique); each row is net of the cost
    of the trades made at its close, so entry and exit costs are inside. ``calendar``: the sorted trading days (the
    chain's days; default the SSE calendar). Fails closed on an event of another note."""
    _own(run, events)
    calendar = _calendar(calendar, events)
    return {e.id: float(sum(r["hedged"] for r in _window_rows(run, e, calendar, ko_days)) * BP) for e in events}


# ---------------------------------------------------------------------------
# Whole life
# ---------------------------------------------------------------------------


def life_stats(run: Run, completed: bool) -> Dict[str, Any]:
    """One run's whole-life block: the sufficient statistics of its daily hedged P&L with the day-0 row left out
    (``n``, ``sum_bp``, ``sumsq_bp``), the terminal P&L ``terminal_bp`` (every row, day 0 included), ``completed``
    (the note ended inside the run; a truncated note's terminal P&L is not a terminal P&L), and the descriptive
    counts: ``rows``, ``cost_bp``, ``im_lots_traded`` (sum of the absolute daily changes of the IM lots held, rolls
    not counted), ``clips`` (days the D9 clip bound), ``leg_days`` (days with a leg held), ``overlay_days`` (days with
    a leg held or wanted and not filled), over which ``phi_sum`` / ``phi_min`` / ``phi_lt1_days`` describe the fill
    ratio as the rows carry it, and ``flags`` (how many days each ledger flag was raised)."""
    day0 = run.inception.isoformat()
    daily = [r["hedged"] * BP for r in run.rows if r["date"] != day0]
    overlay = [r for r in run.rows if r["legs"] or r["unfilled"]]
    flags: Counter = Counter(f for r in run.rows for f in r["flags"])
    lots = [0.0] + [float(r["im_lots"]) for r in run.rows]
    return {
        "n": len(daily), "sum_bp": float(sum(daily)), "sumsq_bp": float(sum(v * v for v in daily)),
        "terminal_bp": float(sum(r["hedged"] for r in run.rows) * BP), "completed": bool(completed),
        "rows": len(run.rows), "first": run.rows[0]["date"] if run.rows else None,
        "last": run.rows[-1]["date"] if run.rows else None,
        "cost_bp": float(sum(r["cost"] for r in run.rows) * BP),
        "im_lots_traded": float(sum(abs(b - a) for a, b in zip(lots, lots[1:]))),
        "clips": sum(1 for r in run.rows if r["clipped"]),
        "leg_days": sum(1 for r in run.rows if r["legs"]), "overlay_days": len(overlay),
        "phi_sum": float(sum(r["phi"] for r in overlay)),
        "phi_min": min((float(r["phi"]) for r in overlay), default=None),
        "phi_lt1_days": sum(1 for r in overlay if r["phi"] < 1.0 and not is_close(r["phi"], 1.0)),
        "flags": dict(flags),
    }


def _life(item: Mapping[str, Any]) -> Mapping[str, Any]:
    return item["life"] if "life" in item else item


def whole_life(runs: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """The whole-life figures of one technique over a set of notes (D10 (b)); ``runs`` are run summaries
    (``summarise_run``) or their ``life`` blocks. ``std_bp``: the sample standard deviation of the concatenated daily
    hedged P&L of every note, day-0 rows left out. ``mean_terminal_bp``: the mean terminal P&L of the completed
    notes (truncated notes left out; ``None`` when no note completed). Also ``mean_cost_bp`` and
    ``mean_im_lots_traded`` per note."""
    lives = [_life(r) for r in runs]
    if not lives:
        raise ValidationError("whole-life figures need at least one run")
    n = sum(int(x["n"]) for x in lives)
    total = sum(float(x["sum_bp"]) for x in lives)
    sumsq = sum(float(x["sumsq_bp"]) for x in lives)
    std = None if n < 2 else float(safe_sqrt(max(sumsq - total * total / n, 0.0) / (n - 1)))
    done = [float(x["terminal_bp"]) for x in lives if x["completed"]]
    return {"n_notes": len(lives), "n_days": n, "std_bp": std, "n_completed": len(done),
            "mean_terminal_bp": float(sum(done) / len(done)) if done else None,
            "mean_cost_bp": float(sum(float(x["cost_bp"]) for x in lives) / len(lives)),
            "mean_im_lots_traded": float(sum(float(x["im_lots_traded"]) for x in lives) / len(lives))}


# ---------------------------------------------------------------------------
# Gamma carry
# ---------------------------------------------------------------------------


def _parse_leg(text: str) -> Tuple[str, str, float]:
    series, cp, strike = text.split(":")
    return series, cp, float(strike)


def _legs_gamma(legs: Mapping[str, float], board: Optional[DayBoard], day: str) -> float:
    """CNY per index point squared of the legs held (seller positions): Black-76 at each leg's settlement of the
    day; a leg that cannot be marked fails closed."""
    total = 0.0
    for text, lots in legs.items():
        series, cp, strike = _parse_leg(text)
        quote = None if board is None else board.mo.get(series, {}).get((cp, strike))
        info = None if board is None else board.series.get(series)
        greeks = None if quote is None or info is None else black.greeks_at_settle(quote.settle, cp, strike, info,
                                                                                   board.spot)
        if greeks is None:
            raise StudyDataError(f"{day}: held leg {text} has no settlement or no gamma on the board")
        total += float(lots) * greeks[1] * MO_MULT
    return total


def gamma_carry(run: Run, events: Sequence[Event], boards: Mapping[date, DayBoard],
                calendar: Optional[Sequence[date]] = None, ko_days: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """Per KO event (D10, descriptive): what the seller carries into the fixing. At the close of the trading day
    before the observation (``date``): ``note_gamma`` = the seller's position gamma of the note on the true model
    (the row's ``position_gamma``, minus the engine gamma), ``legs_gamma`` = the Black-76 gamma of the option legs
    held, ``net_gamma`` their sum (CNY per index point squared), ``net_gamma_1pct_bp`` = ``net_gamma x (1% of spot)^2``
    in bp of notional, and ``hedge_into_fixing`` = the hedge ratio carried at that close. ``hedge_max``: the largest
    hedge ratio of the window's rows. Futures cannot change gamma; they change what is carried through it. A missing
    row or a null gamma gives nulls; KI approaches are not in the result."""
    _own(run, events)
    calendar = _calendar(calendar, events)
    by_date = {r["date"]: r for r in run.rows}
    out: Dict[str, Dict[str, Any]] = {}
    for e in events:
        if e.kind != KO:
            continue
        at = _position(e.date, calendar)
        t1 = calendar[at - 1] if at >= 1 else None
        row = None if t1 is None else by_date.get(t1.isoformat())
        window = _window_rows(run, e, calendar, ko_days)
        block: Dict[str, Any] = {"date": None if t1 is None else t1.isoformat(), "note_gamma": None,
                                 "legs_gamma": None, "net_gamma": None, "net_gamma_1pct_bp": None,
                                 "hedge_into_fixing": None,
                                 "hedge_max": max((float(r["hedge_ratio"]) for r in window), default=None)}
        if row is not None:
            board = boards.get(t1)
            block["hedge_into_fixing"] = float(row["hedge_ratio"])
            block["legs_gamma"] = _legs_gamma(row["legs"], board, row["date"]) if row["legs"] else 0.0
            if row["position_gamma"] is not None:
                block["note_gamma"] = float(row["position_gamma"])
                block["net_gamma"] = block["note_gamma"] + block["legs_gamma"]
                if board is not None:
                    block["net_gamma_1pct_bp"] = block["net_gamma"] * (0.01 * board.spot) ** 2 * BP
        out[e.id] = block
    return out


# ---------------------------------------------------------------------------
# The stored form of one run
# ---------------------------------------------------------------------------

#: The ledger flag of a fixing that could not wait for the next open (``next_open``): its events are left out.
LEFT_OUT_FLAG = "last_day_fixing"


def summarise_run(run: Run, events: Sequence[Event], calendar: Sequence[date], boards: Mapping[date, DayBoard], *,
                  completed: bool, ko_days: Optional[int] = None) -> Dict[str, Any]:
    """What is kept of one run (the runs directory, see ``04_ledger.py``): ``life`` (``life_stats``) and, per event
    id, ``pnl_bp`` (``event_windows``), ``cost_bp`` and ``clips`` (cost of, and days the D9 clip bound on, the
    window's rows), ``leg_days``, ``phi_min``, ``flags`` (ledger flags raised in the window, by count), ``left_out``
    (a window row carries ``last_day_fixing``: the event is left out of the ``next_open`` tables), ``days`` (the
    window's rows as ``[date, hedged bp, hedge ratio]``) and, for a KO event, ``gamma`` (``gamma_carry``)."""
    _own(run, events)
    pnl = event_windows(run, events, calendar, ko_days)
    gamma = gamma_carry(run, events, boards, calendar, ko_days)
    out: Dict[str, Any] = {}
    for e in events:
        rows = _window_rows(run, e, calendar, ko_days)
        flags: Counter = Counter(f for r in rows for f in r["flags"])
        overlay = [float(r["phi"]) for r in rows if r["legs"] or r["unfilled"]]
        block: Dict[str, Any] = {
            "pnl_bp": pnl[e.id], "cost_bp": float(sum(r["cost"] for r in rows) * BP),
            "clips": sum(1 for r in rows if r["clipped"]), "leg_days": sum(1 for r in rows if r["legs"]),
            "phi_min": min(overlay, default=None), "flags": dict(flags), "left_out": LEFT_OUT_FLAG in flags,
            "days": [[r["date"], float(r["hedged"] * BP), float(r["hedge_ratio"])] for r in rows],
        }
        if e.kind == KO:
            block["gamma"] = gamma[e.id]
        out[e.id] = block
    return {"events": out, "life": life_stats(run, completed)}


# ---------------------------------------------------------------------------
# The verdict (D10)
# ---------------------------------------------------------------------------


def _at_least(value: float, bound: float) -> bool:
    """``value >= bound`` with the repo's tolerance."""
    return value > bound or is_close(value, bound)


def _tail_measures(tech_by_event: Mapping[str, float], base_by_event: Mapping[str, float],
                   rules: Rules) -> Dict[str, Optional[float]]:
    """ES10, worst event and mean of both arms on the same events (``None`` with no event)."""
    keys = ("es10", "es10_base", "worst", "worst_base", "mean", "mean_base")
    if set(tech_by_event) != set(base_by_event):
        raise ValidationError("the technique and the baseline must be measured on the same events")
    if not tech_by_event:
        return dict.fromkeys(keys)
    tech = [float(tech_by_event[e]) for e in sorted(tech_by_event)]
    base = [float(base_by_event[e]) for e in sorted(tech_by_event)]
    return dict(zip(keys, (es(tech, rules.es_alpha), es(base, rules.es_alpha), min(tech), min(base),
                           float(sum(tech) / len(tech)), float(sum(base) / len(base)))))


def tail_test(tech_by_event: Mapping[str, float], base_by_event: Mapping[str, float],
              week_by_event: Mapping[str, str], rules: Optional[Rules] = None) -> Dict[str, Any]:
    """Criterion (a) of D10 on one set of events: ``es10`` / ``worst`` / ``mean`` of the technique's event-window
    P&L and of the baseline's (``*_base``), the bootstrap ``p``, and ``tail_pass`` = ES10 higher (``es_higher``),
    worst event not lower (``worst_not_lower``) and ``p < p_max``. With no event every measure is ``None``."""
    rules = default_rules() if rules is None else rules
    if set(tech_by_event) != set(base_by_event):
        raise ValidationError("the technique and the baseline must be measured on the same events")
    events = sorted(tech_by_event)
    out: Dict[str, Any] = {"n_events": len(events), "n_weeks": len({week_by_event[e] for e in events}),
                           "es10": None, "es10_base": None, "worst": None, "worst_base": None, "mean": None,
                           "mean_base": None, "p": None, "es_higher": None, "worst_not_lower": None,
                           "tail_pass": None}
    if not events:
        return out
    out.update(_tail_measures(tech_by_event, base_by_event, rules),
               p=cluster_bootstrap(tech_by_event, base_by_event, week_by_event, rules.bootstrap_n, rules.seed,
                                   rules.es_alpha))
    out["es_higher"] = bool(out["es10"] > out["es10_base"] and not is_close(out["es10"], out["es10_base"]))
    out["worst_not_lower"] = _at_least(out["worst"], out["worst_base"])
    out["tail_pass"] = bool(out["es_higher"] and out["worst_not_lower"] and out["p"] < rules.p_max)
    return out


def life_test(life: Mapping[str, Any], base_life: Mapping[str, Any], rules: Optional[Rules] = None) -> Dict[str, Any]:
    """Criterion (b) of D10 from two ``whole_life`` blocks (the same notes): ``std_pass`` = the technique's daily
    std is at most ``std_ratio_max`` times the baseline's; ``terminal_pass`` = its mean terminal P&L is at least the
    baseline's minus ``terminal_slack_bp``; ``life_pass`` = both. A figure that does not exist (no completed note,
    fewer than two days) leaves its criterion and ``life_pass`` ``None``."""
    rules = default_rules() if rules is None else rules
    std, std_base = life["std_bp"], base_life["std_bp"]
    term, term_base = life["mean_terminal_bp"], base_life["mean_terminal_bp"]
    out: Dict[str, Any] = {"std_bp": std, "std_base_bp": std_base, "std_ratio": None, "mean_terminal_bp": term,
                           "mean_terminal_base_bp": term_base, "terminal_diff_bp": None, "std_pass": None,
                           "terminal_pass": None, "life_pass": None, "n_notes": life["n_notes"],
                           "n_completed": life["n_completed"]}
    if std is not None and std_base is not None:
        out["std_ratio"] = None if is_zero(std_base) else float(safe_divide(std, std_base))
        out["std_pass"] = _at_least(rules.std_ratio_max * std_base, std)
    if term is not None and term_base is not None:
        out["terminal_diff_bp"] = float(term - term_base)
        out["terminal_pass"] = _at_least(term, term_base - rules.terminal_slack_bp)
    if out["std_pass"] is not None and out["terminal_pass"] is not None:
        out["life_pass"] = bool(out["std_pass"] and out["terminal_pass"])
    return out


def verdict(tech_by_event: Mapping[str, float], base_by_event: Mapping[str, float], week_by_event: Mapping[str, str],
            life: Mapping[str, Any], base_life: Mapping[str, Any], rules: Optional[Rules] = None) -> Dict[str, Any]:
    """The pre-registered verdict of one technique against the baseline (D10): ``tail_test`` on the events,
    ``life_test`` on the whole-life blocks, ``decidable`` = at least ``min_events`` events in at least ``min_weeks``
    distinct weeks, and ``win`` = (a) and (b) when decidable, ``None`` when not decidable or when a criterion could
    not be evaluated. The measures are reported either way."""
    rules = default_rules() if rules is None else rules
    out = {**tail_test(tech_by_event, base_by_event, week_by_event, rules), **life_test(life, base_life, rules)}
    out["decidable"] = bool(out["n_events"] >= rules.min_events and out["n_weeks"] >= rules.min_weeks)
    if not out["decidable"] or out["tail_pass"] is None or out["life_pass"] is None:
        out["win"] = None
    else:
        out["win"] = bool(out["tail_pass"] and out["life_pass"])
    return out


# ---------------------------------------------------------------------------
# Training selection (D6, D11, "Training picks")
# ---------------------------------------------------------------------------

BARRIERS: Tuple[str, ...] = (KO, KI)
#: Composite rows and the slots the training selection fills (``gpin.techniques`` reads the same block).
COMPOSITE_SLOTS: Dict[str, Tuple[str, ...]] = {"sop_futures": BARRIERS, "sop_overlay": BARRIERS}


def _row_ids(row: Mapping[str, Any]) -> List[str]:
    return [grid_id(row["id"], point) for point in row["grid"]]


def _better(es_a: float, cost_a: float, es_b: float, cost_b: float) -> bool:
    """Whether candidate ``a`` displaces ``b`` (registered earlier): a higher ES10; on a tie the lower cost; on a tie
    of both the earlier one stays."""
    if not is_close(es_a, es_b):
        return es_a > es_b
    return cost_a < cost_b and not is_close(cost_a, cost_b)


def select_on_train(prereg: Mapping[str, Any], windows: Mapping[str, Mapping[str, Mapping[str, float]]],
                    costs: Mapping[str, Mapping[str, float]], *, allow_empty: bool = False) -> Dict[str, Any]:
    """The training selection. ``windows[barrier][technique id]`` = ``{event: bp}``, the training event-window P&L
    under the primary clock on the barrier's book (KO: ``ki75`` KO pin events; KI: ``seas6`` approaches);
    ``costs[barrier][technique id]`` = the technique's mean event-window cost in bp on those events.

    ``picks[barrier][row]``: per candidate row of the pre-registration acting on that barrier (a ``both`` row such as
    ``secant`` is picked once per barrier), the grid point with the highest training ES10; ties go to the lower cost,
    then to the grid's order. ``beats_base`` says whether the winner beats the baseline's training ES10; a winner that
    does not is still the row's pick. ``composites``: per composite row and barrier slot, the winner across the picks
    of the rows listed in the row's ``members``, by the same rule (rows in the listed order). ``base``: the baseline's
    training ES10 per barrier. ``rows``: per barrier the picked rows in the pre-registration's order.

    Fails closed when a grid point or the baseline has no run, when arms are on different events and when a barrier
    has no training event. ``allow_empty`` (a smoke run on a partial set of states) lets such a barrier through and
    lists it in ``empty``: its rows have no pick (``id``, ``es10`` and ``cost_bp`` are null: no grid point stands in
    for a pick that was not made), and a composite, which needs a member for every barrier, is then not in
    ``composites`` at all."""
    rules = Rules.from_prereg(prereg)
    rows = {str(r["id"]): r for r in prereg["techniques"]}
    out: Dict[str, Any] = {"picks": {}, "base": {}, "composites": {}, "empty": [], "rows": {}}
    table: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {}
    for barrier in BARRIERS:
        runs = windows[barrier]
        if "base" not in runs:
            raise ValidationError(f"{barrier}: no training run of the baseline")
        events = set(runs["base"])
        if not events:
            if not allow_empty:
                raise StudyDataError(f"no training {barrier.upper()} event: the selection cannot be made")
            out["empty"].append(barrier)

        def measure(tid: str) -> Tuple[Optional[float], Optional[float]]:
            if tid not in runs or tid not in costs[barrier]:
                raise ValidationError(f"{barrier}: no training run of {tid}")
            if set(runs[tid]) != events:
                raise ValidationError(f"{barrier}: {tid} and the baseline are not measured on the same events")
            if not events:
                return None, None
            return es(list(runs[tid].values()), rules.es_alpha), float(costs[barrier][tid])

        base_es, _ = measure("base")
        out["base"][barrier] = {"es10": base_es, "n_events": len(events)}
        table[barrier] = {}
        out["picks"][barrier] = {}
        for rid, row in rows.items():
            if row["role"] != "candidate" or row["barrier"] not in (barrier, "both"):
                continue
            grid = {tid: measure(tid) for tid in _row_ids(row)}
            table[barrier].update(grid)
            best: Optional[str] = None
            for tid in grid if events else ():
                if best is None or _better(*grid[tid], *grid[best]):
                    best = tid
            best_es, best_cost = (None, None) if best is None else grid[best]
            out["picks"][barrier][rid] = {
                "id": best, "es10": best_es, "cost_bp": best_cost, "n_events": len(events),
                "beats_base": None if best_es is None else bool(best_es > base_es and not is_close(best_es, base_es)),
                "grid": {tid: {"es10": v[0], "cost_bp": v[1]} for tid, v in grid.items()}}
        out["rows"][barrier] = list(out["picks"][barrier])
    for rid, slots in COMPOSITE_SLOTS.items():
        if rid not in rows or any(barrier in out["empty"] for barrier in slots):
            continue
        out["composites"][rid] = {}
        for barrier in slots:
            members = rows[rid]["members"][barrier]
            missing = [m for m in members if m not in out["picks"][barrier]]
            if missing:
                raise ValidationError(f"{rid}: member rows {missing} have no {barrier} pick")
            cands = [out["picks"][barrier][m]["id"] for m in members]
            best = cands[0]
            for tid in cands[1:]:
                if _better(*table[barrier][tid], *table[barrier][best]):
                    best = tid
            out["composites"][rid][barrier] = best
    return out


def selection_composites(selection: Mapping[str, Any]) -> Optional[Dict[str, Dict[str, str]]]:
    """The selection's ``composites`` block in the shape ``gpin.techniques.build_registry`` takes, or ``None`` when it
    does not name every composite with a member for every slot (a partial selection with an empty barrier): the
    registry is then built without the composites, never with an invented member."""
    block = selection.get("composites") or {}
    for rid, slots in COMPOSITE_SLOTS.items():
        members = block.get(rid)
        if not isinstance(members, Mapping) or set(members) != set(slots) or any(members[s] is None for s in slots):
            return None
    return {rid: {slot: str(block[rid][slot]) for slot in COMPOSITE_SLOTS[rid]} for rid in COMPOSITE_SLOTS}


def evaluated_ids(prereg: Mapping[str, Any], selection: Mapping[str, Any]) -> List[str]:
    """The techniques of the test slice, in the pre-registration's order: the baseline, the controls and the cost
    rows (every grid point), the training pick of every candidate row (both picks of a row picked per barrier) and
    the composites. A barrier the selection lists as ``empty`` (a partial selection) has no pick: its rows are left
    out, and so are the composites, which need a member for it. Anywhere else a row without a pick, or a missing
    composite, fails closed."""
    empty = set(selection.get("empty", ()))
    composites = selection_composites(selection)
    out: List[str] = []
    for row in prereg["techniques"]:
        rid = str(row["id"])
        if row["role"] == "composite":
            if composites is None and not empty:
                raise ValidationError(f"the selection names no members for the composite {rid}")
            ids = [rid] if composites is not None else []
        elif row["role"] == "candidate":
            barriers = [b for b in BARRIERS if rid in selection["picks"][b]]
            picked = {b: selection["picks"][b][rid]["id"] for b in barriers}
            unpicked = [b for b, tid in picked.items() if tid is None and b not in empty]
            if not barriers or unpicked:
                raise ValidationError(f"the selection has no pick for row {rid}")
            ids = [tid for tid in _row_ids(row) if tid in set(picked.values())]
        else:
            ids = _row_ids(row)
        out.extend(ids)
    return out


# ---------------------------------------------------------------------------
# One note: every technique, clock and configuration, kept in the stored form
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """A ledger configuration (D8): the desk notional the tradability screen is checked at and the option cost per
    leg per side in index points."""

    desk_notional: float
    mo_cost_pt: float

    @property
    def key(self) -> str:
        """``"desk500_cost0.2"``: desk notional in mio CNY, cost in index points."""
        return f"desk{self.desk_notional / 1.0e6:g}_cost{self.mo_cost_pt:g}"


def primary_config(prereg: Mapping[str, Any]) -> Config:
    """The pre-registered configuration, the verdict's (500 mio, 0.2 points)."""
    t = prereg["tradability"]
    return Config(float(t["desk_notional"]), float(t["mo_cost_pt"]))


def sensitivity_configs(prereg: Mapping[str, Any], desk_notionals: Sequence[float],
                        mo_cost_pts: Sequence[float]) -> List[Config]:
    """The sensitivities asked for, one number changed at a time against the primary configuration (which is not in
    the result): every other desk notional at the primary cost, then every other cost at the primary desk notional."""
    primary = primary_config(prereg)
    out: List[Config] = []
    for desk in desk_notionals:
        if not is_close(float(desk), primary.desk_notional):
            out.append(Config(float(desk), primary.mo_cost_pt))
    for cost in mo_cost_pts:
        if not is_close(float(cost), primary.mo_cost_pt):
            out.append(Config(primary.desk_notional, float(cost)))
    return list(dict.fromkeys(out))


def note_completed(states: Mapping[str, Any], until: Optional[date]) -> bool:
    """Whether the note's life ends inside the run: its last record is dead (redeemed or matured) and dated on or
    before ``until``. A note alive at the data end, or cut by ``until``, is truncated."""
    last = states["days"][-1]
    return (not last["alive"]) and (until is None or date.fromisoformat(last["date"]) <= until)


def gap_observations(states: Mapping[str, Any], until: Optional[date]) -> List[Dict[str, Any]]:
    """Per KO observation the note entered a KO window for: the first day with a recorded ``gap_ko``, that gap in bp
    of notional (positive = the seller loses on redemption) and the observation's ``ko_dir``. With ``until`` (the
    training slice) only the observations dated on or before it: like an event, an observation belongs to the slice
    of its date."""
    out: Dict[str, Dict[str, Any]] = {}
    for rec in states["days"]:
        if until is not None and date.fromisoformat(rec["date"]) > until:
            break
        obs, gap = rec.get("next_ko"), rec.get("gap_ko")
        if obs is None or gap is None or obs in out or (until is not None and date.fromisoformat(obs) > until):
            continue
        out[obs] = {"obs": obs, "date": rec["date"], "gap_ko_bp": float(gap) * BP, "ko_dir": rec.get("ko_dir")}
    return list(out.values())


def measurable_events(states: Mapping[str, Any], events: Sequence[Event]) -> List[Event]:
    """The events dated on or before the note's last recorded day. A complete state file holds every event of its
    note; a short diagnostic one (``03_pricing.py --quick``) ends before some of them, and an event window without
    rows would count 0, which is not a measurement."""
    last = date.fromisoformat(states["days"][-1]["date"])
    return [e for e in events if e.date <= last]


def run_note(states: Mapping[str, Any], techniques: Sequence[Technique], boards: Mapping[date, DayBoard],
             chains: Sequence[Any], events: Sequence[Event], *, plan: Sequence[Tuple[Config, Sequence[str]]],
             until: Optional[date] = None) -> Dict[str, Any]:
    """Every technique on one note's recorded states under every (configuration, clocks) of ``plan``, each run kept
    as its summary (``summarise_run``): the content of one file of the runs directory.

    ``{"note", "book", "inception", "until", "completed", "end_reason", "last_record", "events": {id: event},
    "gap_observations": [...], "runs": {config key: {clock: {technique id: summary}}}}``. The trading calendar of the
    event windows is the chain's days. With ``until`` (the training slice) every run stops at that day; an event
    dated after it is refused and a row dated after it fails the run: no P&L of a later day is ever computed, and
    nothing recorded after it (how and when the note ends) is written: ``last_record`` is the last record seen and
    ``end_reason`` is null for a note still alive then."""
    if until is not None and any(e.date > until for e in events):
        raise ValidationError(f"an event dated after {until} cannot be measured on a run that stops there")
    data = states.to_json() if hasattr(states, "to_json") else states
    late = [e.id for e in events if e not in measurable_events(data, events)]
    if late:
        raise ValidationError(f"{late[:3]} are dated after the note's last recorded day {data['days'][-1]['date']}: "
                              "the state file does not cover them (a short diagnostic file?)")
    book = str(data["book"])
    inception = date.fromisoformat(data["days"][0]["date"])
    calendar = [c.day for c in chains]
    completed = note_completed(data, until)
    seen = [r for r in data["days"] if until is None or date.fromisoformat(r["date"]) <= until]
    if not seen:
        raise ValidationError(f"{note_id(book, inception)} has no record on or before {until}")
    last = seen[-1]
    out: Dict[str, Any] = {
        "note": note_id(book, inception), "book": book, "inception": inception.isoformat(),
        "until": None if until is None else until.isoformat(), "completed": completed,
        "end_reason": last["end_reason"] if completed else None, "last_record": last["date"],
        "events": {e.id: e.to_json() for e in events}, "gap_observations": gap_observations(data, until), "runs": {}}
    for config, modes in plan:
        cells = out["runs"].setdefault(config.key, {})
        for mode in modes:
            by_tech = cells.setdefault(mode, {})
            for tech in techniques:
                run = run_technique(data, tech, boards, chains, mode=mode, mo_cost_pt=config.mo_cost_pt,
                                    desk_notional=config.desk_notional, until=until)
                if until is not None and run.rows and run.rows[-1]["date"] > until.isoformat():
                    raise ValidationError(f"{tech.id}: a row dated after {until} on a run that stops there")
                by_tech[tech.id] = summarise_run(run, events, calendar, boards, completed=completed)
    return out


def merge_notes(first: Mapping[str, Any], second: Mapping[str, Any]) -> Dict[str, Any]:
    """One note's two ``run_note`` results as one (the grid first, then the composites the training selection
    names). Fails closed unless both are the same note, slice and events and no technique is in both."""
    for key in ("note", "until", "completed", "events", "gap_observations"):
        if first[key] != second[key]:
            raise ValidationError(f"cannot merge runs whose {key} differ")
    runs: Dict[str, Any] = {cfg: {mode: dict(cells) for mode, cells in modes.items()}
                            for cfg, modes in first["runs"].items()}
    for cfg, modes in second["runs"].items():
        for mode, cells in modes.items():
            target = runs.setdefault(cfg, {}).setdefault(mode, {})
            twice = sorted(set(target) & set(cells))
            if twice:
                raise ValidationError(f"cannot merge runs: {twice} are in both")
            target.update(cells)
    return {**first, "runs": runs}


# ---------------------------------------------------------------------------
# The three hashes and the guard of the test slice (D11)
# ---------------------------------------------------------------------------

#: The ledger-side code: an edit to any of these files stales a training selection.
CODE_FILES: Tuple[str, ...] = ("black.py", "ledger.py", "techniques.py", "measures.py")


def code_hash(directory: Optional[Path] = None) -> str:
    """sha256 over the ledger-side files (``CODE_FILES`` of the package directory): name and bytes of each."""
    directory = Path(__file__).resolve().parent if directory is None else Path(directory)
    h = hashlib.sha256()
    for name in CODE_FILES:
        h.update(name.encode("utf-8") + b"\0" + (directory / name).read_bytes() + b"\0")
    return h.hexdigest()


def file_hash(path: Path) -> str:
    """sha256 of a file's bytes; fails closed when it does not exist."""
    path = Path(path)
    if not path.exists():
        raise StudyDataError(f"{path} missing")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp(events_path: Path, prereg_path: Optional[Path] = None) -> Dict[str, str]:
    """The three hashes a training selection is made under: the pre-registration (``prereg_hash``), the events
    file and the ledger-side code."""
    return {"prereg_hash": prereg_hash(PREREG_PATH if prereg_path is None else prereg_path),
            "events_hash": file_hash(events_path), "code_hash": code_hash()}


class SelectionRefused(StudyDataError):
    """The training selection is missing, stale or partial: no test number may be computed or printed."""


_HASH_NAMES = {"prereg_hash": "the pre-registration", "events_hash": "the events file",
               "code_hash": "the ledger-side code"}


def check_selection(path: Path, current: Mapping[str, str], *, allow_partial: bool = False) -> Dict[str, Any]:
    """The training selection at ``path`` when it may be used: it exists, each of its three hashes equals
    ``current`` (``stamp``) and it was not made on a partial set of states (unless ``allow_partial``: a smoke run).
    Otherwise ``SelectionRefused`` naming every reason. The test slice and the report call this before any test
    number."""
    path = Path(path)
    if not path.exists():
        raise SelectionRefused(f"{path} does not exist: run the training slice first (04_ledger.py --slice train)")
    selection = json.loads(path.read_text(encoding="utf-8"))
    reasons = [f"{what} has changed since the training selection was made ({key})"
               for key, what in _HASH_NAMES.items() if selection.get(key) != current[key]]
    if selection.get("partial", True) and not allow_partial:
        reasons.append("it was made on a partial set of states (a smoke run)")
    if reasons:
        raise SelectionRefused(f"{path} is refused: " + "; ".join(reasons) + ". Re-run the training slice.")
    return selection


# ---------------------------------------------------------------------------
# Aggregation over stored note files: the training selection and the tables
# ---------------------------------------------------------------------------


def _cells(note: Mapping[str, Any], cfg: str, clock: str) -> Mapping[str, Any]:
    try:
        return note["runs"][cfg][clock]
    except KeyError:
        raise ValidationError(f"{note.get('note')}: no runs under {cfg} / {clock}") from None


def _common_ids(notes: Sequence[Mapping[str, Any]], cfg: str, clock: str) -> List[str]:
    """The technique ids of the runs under ``cfg`` / ``clock``; every note file must hold the same ones."""
    if not notes:
        raise ValidationError("no note file to aggregate")
    ids = list(_cells(notes[0], cfg, clock))
    for note in notes[1:]:
        if set(_cells(note, cfg, clock)) != set(ids):
            raise ValidationError(f"{note['note']} does not hold the same techniques as {notes[0]['note']} under "
                                  f"{cfg} / {clock}")
    return ids


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def training_selection(notes: Sequence[Mapping[str, Any]], prereg: Mapping[str, Any], *,
                       allow_empty: bool = False) -> Dict[str, Any]:
    """``select_on_train`` on stored training files (``run_note`` with ``until``): the event-window P&L of every grid
    point under the pre-registered selection clock and the primary configuration, KO pin events of the KO book and
    approaches of the KI book. Adds ``events``: per barrier the number of training events, their weeks and the book.
    A file holding an event that is not training-dated is refused."""
    rule = prereg["selection"]
    cfg, clock = primary_config(prereg).key, str(rule["clock"])
    books = {KO: str(rule["ko_book"]), KI: str(rule["ki_book"])}
    ids = _common_ids(notes, cfg, clock)
    for note in notes:
        late = [eid for eid, ev in note["events"].items() if ev["slice"] != rule["slice"]]
        if late:
            raise ValidationError(f"{note['note']}: {late[:3]} are not training events")
    windows: Dict[str, Dict[str, Dict[str, float]]] = {}
    costs: Dict[str, Dict[str, float]] = {}
    events: Dict[str, Dict[str, Any]] = {}
    for barrier in BARRIERS:
        windows[barrier] = {tid: {} for tid in ids}
        cost_lists: Dict[str, List[float]] = {tid: [] for tid in ids}
        weeks = set()
        for note in notes:
            if note["book"] != books[barrier]:
                continue
            cells = _cells(note, cfg, clock)
            for eid, ev in note["events"].items():
                if ev["kind"] != barrier:
                    continue
                weeks.add(ev["week"])
                for tid in ids:
                    windows[barrier][tid][eid] = float(cells[tid]["events"][eid]["pnl_bp"])
                    cost_lists[tid].append(float(cells[tid]["events"][eid]["cost_bp"]))
        costs[barrier] = {tid: (_mean(v) or 0.0) for tid, v in cost_lists.items()}
        events[barrier] = {"n": len(windows[barrier][ids[0]]), "weeks": len(weeks), "book": books[barrier]}
    out = select_on_train(prereg, windows, costs, allow_empty=allow_empty)
    out["events"] = events
    return out


def _technique_meta(prereg: Mapping[str, Any], ids: Sequence[str],
                    selection: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Per technique id its pre-registration row, family, role and barrier, and the barriers on which it is eligible
    for a verdict: a candidate row's training pick for that barrier, or a composite. The baseline, the controls, the
    cost rows and the other grid points are reported, never eligible."""
    rows = {str(r["id"]): r for r in prereg["techniques"]}
    out: Dict[str, Dict[str, Any]] = {}
    for tid in ids:
        rid = tid.split("@")[0]
        if rid not in rows:
            raise ValidationError(f"{tid} is not a technique of the pre-registration")
        row = rows[rid]
        eligible = []
        for barrier in BARRIERS:
            if row["barrier"] not in (barrier, "both"):
                continue
            pick = selection["picks"].get(barrier, {}).get(rid, {}).get("id")
            if row["role"] == "composite" or (row["role"] == "candidate" and pick == tid):
                eligible.append(barrier)
        out[tid] = {"row": rid, "family": str(row["family"]), "role": str(row["role"]),
                    "barrier": str(row["barrier"]), "eligible": eligible}
    return out


_NO_LIFE: Dict[str, Any] = {"n_notes": 0, "n_days": 0, "std_bp": None, "n_completed": 0, "mean_terminal_bp": None,
                            "mean_cost_bp": None, "mean_im_lots_traded": None}


def _whole_life(notes: Sequence[Mapping[str, Any]], cfg: str, clock: str, tid: str) -> Dict[str, Any]:
    return whole_life([_cells(n, cfg, clock)[tid] for n in notes]) if notes else dict(_NO_LIFE)


def _event_table(notes: Sequence[Mapping[str, Any]], meta: Mapping[str, Mapping[str, Any]], rules: Rules, *,
                 cfg: str, clock: str, barrier: str, book: str, slice_: str,
                 life_notes: Optional[Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    """One table: the techniques acting on ``barrier`` against the baseline on the ``slice_`` events of ``book``
    under ``clock``. With ``life_notes`` (the whole-life set) every row is a ``verdict``; without, the tail measures
    only (the transfer check)."""
    with_verdict = life_notes is not None
    book_notes = [n for n in notes if n["book"] == book]
    owner = {eid: n for n in book_notes for eid, ev in n["events"].items()
             if ev["kind"] == barrier and ev["slice"] == slice_}
    ids = [tid for tid in meta if meta[tid]["barrier"] in (barrier, "both")]
    if "base" not in ids:
        raise ValidationError("the baseline has no run: nothing can be compared")
    left_out = sorted(eid for eid, n in owner.items()
                      if any(_cells(n, cfg, clock)[tid]["events"][eid]["left_out"] for tid in ids))
    used = sorted(eid for eid in owner if eid not in left_out)
    weeks = {eid: owner[eid]["events"][eid]["week"] for eid in used}

    def cell(tid: str, eid: str) -> Mapping[str, Any]:
        return _cells(owner[eid], cfg, clock)[tid]["events"][eid]

    pnl = {tid: {eid: float(cell(tid, eid)["pnl_bp"]) for eid in used} for tid in ids}
    rows: Dict[str, Dict[str, Any]] = {}
    for tid in ids:
        eligible = barrier in meta[tid]["eligible"]
        if with_verdict:
            row = verdict(pnl[tid], pnl["base"], weeks, _whole_life(life_notes, cfg, clock, tid),
                          _whole_life(life_notes, cfg, clock, "base"), rules)
            if not eligible:
                row["win"] = None
            row["p_holm"] = None
        else:
            row = {"n_events": len(used), "n_weeks": len(set(weeks.values())),
                   **_tail_measures(pnl[tid], pnl["base"], rules)}
        row.update(row=meta[tid]["row"], role=meta[tid]["role"], eligible=eligible,
                   clips=sum(int(cell(tid, eid)["clips"]) for eid in used),
                   mean_cost_bp=_mean([float(cell(tid, eid)["cost_bp"]) for eid in used]))
        rows[tid] = row
    table: Dict[str, Any] = {"book": book, "barrier": barrier, "slice": slice_, "clock": clock, "config": cfg,
                             "verdict": with_verdict, "n_events": len(used), "n_weeks": len(set(weeks.values())),
                             "left_out": left_out, "order": list(ids), "rows": rows}
    if with_verdict:
        table["decidable"] = bool(len(used) >= rules.min_events and table["n_weeks"] >= rules.min_weeks)
        table["whole_life"] = {"set": "headline", "n_notes": len(life_notes)}
        tested = [tid for tid in ids if rows[tid]["eligible"] and rows[tid]["p"] is not None]
        for tid, adjusted in zip(tested, holm([rows[tid]["p"] for tid in tested])):
            rows[tid]["p_holm"] = float(adjusted)
        table["n_eligible"] = sum(1 for tid in ids if rows[tid]["eligible"])
        table["wins"] = [tid for tid in ids if rows[tid]["win"] is True]
    return table


def _robust(win: Optional[bool], others: Sequence[Optional[bool]]) -> Optional[bool]:
    """A winner is robust when it also passes the tail test under every other clock (D10)."""
    if win is None:
        return None
    if not win:
        return False
    if not others or any(o is None for o in others):
        return None
    return all(others)


def _gamma_table(table: Mapping[str, Any], notes: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per technique of a KO table, the means of the gamma-carry blocks over the table's events."""
    owner = {eid: n for n in notes for eid in n["events"]}
    used = [eid for eid in owner if owner[eid]["events"][eid]["kind"] == KO
            and owner[eid]["events"][eid]["slice"] == table["slice"] and owner[eid]["book"] == table["book"]
            and eid not in table["left_out"]]
    out: Dict[str, Dict[str, Any]] = {}
    for tid in table["rows"]:
        blocks = [_cells(owner[eid], table["config"], table["clock"])[tid]["events"][eid]["gamma"] for eid in used]

        def col(key: str) -> List[float]:
            return [float(b[key]) for b in blocks if b[key] is not None]

        out[tid] = {"n_events": len(blocks), "mean_note_gamma": _mean(col("note_gamma")),
                    "mean_legs_gamma": _mean(col("legs_gamma")), "mean_net_gamma": _mean(col("net_gamma")),
                    "mean_net_gamma_1pct_bp": _mean(col("net_gamma_1pct_bp")),
                    "mean_hedge_into_fixing": _mean(col("hedge_into_fixing")),
                    "mean_hedge_max": _mean(col("hedge_max")), "max_hedge_max": max(col("hedge_max"), default=None)}
    return out


def _flag_totals(counters: Iterable[Mapping[str, int]]) -> Dict[str, int]:
    total: Counter = Counter()
    for c in counters:
        total.update(c)
    return dict(sorted(total.items()))


def tables(notes: Sequence[Mapping[str, Any]], prereg: Mapping[str, Any], sample: Mapping[str, Any],
           selection: Mapping[str, Any], *, slice_: str) -> Dict[str, Any]:
    """Everything the report prints, from stored note files (``run_note``), a loaded events file and the training
    selection; seller side throughout. Works on any set of files that hold the same runs (a partial set too).

    ``order``: the technique ids in the pre-registration's order (the files are written with sorted keys; every
    table has its own ``order``). ``tables[clock]``: ``ko`` (the KO book's KO pin events, whole life on its headline
    notes), ``ki`` (the KI book's approaches, whole life on every booking) and ``ki_transfer`` (the same techniques
    on the KO book's own approaches: measures, no verdict), on the events of ``slice_`` under the primary
    configuration. A table has
    ``n_events`` / ``n_weeks`` / ``decidable`` / ``left_out`` (events with a fixing on the last day, left out under
    ``next_open``) and one row per technique acting on that barrier: the ``verdict`` fields, ``eligible`` (reported
    rows are never eligible and have no ``win``), ``p_holm`` (Holm across the eligible rows), ``clips`` and
    ``mean_cost_bp`` inside the event windows, and in the primary clock's table ``robust`` (a winner that also passes
    the tail test under every other clock). ``sensitivities[config][clock]``: the ``ko`` and ``ki`` tables of every
    other configuration. ``whole_life_pin``: the pin set's whole-life figures (decide nothing). ``gamma_carry``: per
    clock and technique the means over the KO table's events. ``extras``: ``gap_observations`` (the sign of
    ``gap_ko`` on the first KO-window day of every observation, per book), ``left_out`` counts, ``flags`` (ledger
    flags over whole lives) and ``event_flags`` (inside the event windows), ``overlay`` (leg days and fill ratio of
    the techniques that held or wanted a leg) and ``band`` (cost and IM turnover of the baseline and the cost rows
    on each book's headline notes)."""
    rules = Rules.from_prereg(prereg)
    v = prereg["verdict"]
    cfg = primary_config(prereg).key
    primary_clock = str(prereg["clocks"]["primary"])
    if not notes:
        raise ValidationError("no note file to aggregate")
    order = [primary_clock] + [str(c) for c in prereg["clocks"]["others"]]
    clocks = [c for c in order if c in notes[0]["runs"].get(cfg, {})]
    if primary_clock not in clocks:
        raise ValidationError(f"the note files hold no run under the primary clock {primary_clock}")
    ids = _common_ids(notes, cfg, primary_clock)
    for clock in clocks:
        if set(_common_ids(notes, cfg, clock)) != set(ids):
            raise ValidationError(f"the runs under {clock} are not the techniques of {primary_clock}")
    meta = _technique_meta(prereg, ids, selection)
    ko_book, ki_book = str(v["ko_book"]), str(v["ki_book"])
    transfer_books = [str(b) for b in v["ki_transfer_books"]]

    def members(book: str, key: str) -> List[Mapping[str, Any]]:
        raw = sample["books"][book][key]
        wanted = {r["inception"] if isinstance(r, Mapping) else r for r in raw}
        return [n for n in notes if n["book"] == book and n["inception"] in wanted]

    def pair(config: str, clock: str) -> Dict[str, Any]:
        return {KO: _event_table(notes, meta, rules, cfg=config, clock=clock, barrier=KO, book=ko_book,
                                 slice_=slice_, life_notes=members(ko_book, "headline")),
                KI: _event_table(notes, meta, rules, cfg=config, clock=clock, barrier=KI, book=ki_book,
                                 slice_=slice_, life_notes=members(ki_book, "headline"))}

    by_clock: Dict[str, Dict[str, Any]] = {}
    for clock in clocks:
        by_clock[clock] = pair(cfg, clock)
        for book in transfer_books:
            name = "ki_transfer" if len(transfer_books) == 1 else f"ki_transfer_{book}"
            by_clock[clock][name] = _event_table(notes, meta, rules, cfg=cfg, clock=clock, barrier=KI, book=book,
                                                 slice_=slice_, life_notes=None)
    others = [c for c in clocks if c != primary_clock]
    for barrier in BARRIERS:
        for tid, row in by_clock[primary_clock][barrier]["rows"].items():
            row["robust"] = _robust(row["win"], [by_clock[c][barrier]["rows"][tid]["tail_pass"] for c in others])

    sensitivities: Dict[str, Dict[str, Any]] = {}
    for config in notes[0]["runs"]:
        if config == cfg:
            continue
        sensitivities[config] = {}
        for clock in [c for c in order if c in notes[0]["runs"][config]]:
            if set(_common_ids(notes, config, clock)) != set(ids):
                raise ValidationError(f"the runs under {config} / {clock} are not the techniques of the primary runs")
            sensitivities[config][clock] = pair(config, clock)

    pin = members(ko_book, "pin")
    books = list(dict.fromkeys([ko_book, ki_book] + transfer_books))
    gaps: Dict[str, Dict[str, int]] = {}
    for book in books:
        values = [float(g["gap_ko_bp"]) for n in notes if n["book"] == book for g in n["gap_observations"]]
        zero = sum(1 for g in values if is_zero(g))
        positive = sum(1 for g in values if g > 0.0 and not is_zero(g))
        gaps[book] = {"n": len(values), "positive": positive, "negative": len(values) - positive - zero,
                      "zero": zero, "non_positive": len(values) - positive}
    flags: Dict[str, Dict[str, Dict[str, int]]] = {}
    event_flags: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = {}
    overlay: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for clock in clocks:
        flags[clock], overlay[clock], event_flags[clock] = {}, {}, {}
        for tid in ids:
            lives = [_cells(n, cfg, clock)[tid]["life"] for n in notes]
            flags[clock][tid] = _flag_totals(life["flags"] for life in lives)
            days = sum(int(life["overlay_days"]) for life in lives)
            if days:
                mins = [float(life["phi_min"]) for life in lives if life["phi_min"] is not None]
                overlay[clock][tid] = {
                    "leg_days": sum(int(life["leg_days"]) for life in lives), "overlay_days": days,
                    "phi_lt1_days": sum(int(life["phi_lt1_days"]) for life in lives),
                    "phi_mean": float(sum(float(life["phi_sum"]) for life in lives) / days),
                    "phi_min": min(mins, default=None), "flags": flags[clock][tid]}
        for barrier in BARRIERS:
            table = by_clock[clock][barrier]
            owner = {eid: n for n in notes if n["book"] == table["book"] for eid, ev in n["events"].items()
                     if ev["kind"] == barrier and ev["slice"] == slice_ and eid not in table["left_out"]}
            event_flags[clock][barrier] = {
                tid: _flag_totals(_cells(n, cfg, clock)[tid]["events"][eid]["flags"] for eid, n in owner.items())
                for tid in table["rows"]}
    cost_ids = [tid for tid in ids if tid == "base" or meta[tid]["role"] == "cost"]
    return {
        "view": "seller", "slice": slice_, "primary": {"clock": primary_clock, "config": cfg}, "clocks": clocks,
        "books": {KO: ko_book, KI: ki_book, "ki_transfer": transfer_books},
        "rules": {"es_alpha": rules.es_alpha, "bootstrap_n": rules.bootstrap_n, "seed": rules.seed,
                  "p_max": rules.p_max, "std_ratio_max": rules.std_ratio_max,
                  "terminal_slack_bp": rules.terminal_slack_bp, "min_events": rules.min_events,
                  "min_weeks": rules.min_weeks, "ko_days": rules.ko_days},
        "notes": {"n": len(notes), **{book: sum(1 for n in notes if n["book"] == book) for book in books}},
        "order": list(ids), "techniques": meta,
        "tables": by_clock,
        "sensitivities": sensitivities,
        "whole_life_pin": {clock: {tid: _whole_life(pin, cfg, clock, tid) for tid in ids} for clock in clocks},
        "gamma_carry": {clock: _gamma_table(by_clock[clock][KO], notes) for clock in clocks},
        "extras": {
            "gap_observations": gaps,
            "left_out": {clock: {b: len(by_clock[clock][b]["left_out"]) for b in BARRIERS} for clock in clocks},
            "flags": flags, "event_flags": event_flags, "overlay": overlay,
            "band": {book: {tid: _whole_life(members(book, "headline"), cfg, primary_clock, tid) for tid in cost_ids}
                     for book in dict.fromkeys([ko_book, ki_book])},
        },
    }
