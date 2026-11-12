"""Stage 3: the pricing pass. Per note of the sample, per live day, a spot ladder of QUAD V2 price / delta / gamma for
every hedge model whose window the day is in (``gpin.pricing``).

A note is a (book, inception): a book is a KI level (``ki75`` the product, ``ki85``, ``ki90``), identified by
``"<book>:<inception>"``. Reads ``results/events.json`` (the note list per book, ``02_events.py``) and
``OUTPUT_DIR/chain.pkl`` (``01_data.py``). Coupons: one per note, solved to PV = 0 under hold-forward for the book's
KI level (``gpin.pricing.solve_book_coupon``) and cached in ``OUTPUT_DIR/coupons.json`` with a stamp of what they
depend on (re-solved when it differs). One atomic JSON per note, ``<book>__<inception>.json``, in
``OUTPUT_DIR/states/``, resumable: a file is reused only when its stamped fingerprint matches, its stored coupon is
the current one and it was not written with ``max_days`` or ``price_days`` (see ``gpin.pricing.read_current``). A
failed note leaves ``<book>__<inception>.error.json`` and the exit status is 2; every other note completes. Runs on
a spawn pool started from this file, with the group-RSS guard and a lock file so two runs cannot share the directory.

Run (from a script file, never stdin; long runs under ``nohup caffeinate -i -m -s``):
    python example/case_studies/snowball_2y_gamma_pin_sop/03_pricing.py --plan [--workers 4] [--books ki90]
    python example/case_studies/snowball_2y_gamma_pin_sop/03_pricing.py --quick
    python example/case_studies/snowball_2y_gamma_pin_sop/03_pricing.py [--workers 4] [--books ki75,ki90] [--notes ki75:2024-02-29]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantark.util.exceptions import PricingError  # noqa: E402

from gpin import CHAIN_PATH, OUTPUT_DIR, RESULTS_DIR, STATES_DIR, StudyDataError, load_prereg  # noqa: E402
from gpin.events import BOOKS, note_id, parse_note_id, scan  # noqa: E402
from gpin.pricing import (  # noqa: E402
    book_coupons, calendar_for, coupon_stamp, expected_solves, price_one, pricing_fingerprint, solve_book_coupon,
)
from s2ts.chain import load_chain  # noqa: E402
from s2ts.runner import (  # noqa: E402
    OutDirLocked, RssGuard, RssLimitExceeded, _acquire_lock, _release_lock, v2_engine, write_json_atomic,
)
from s2ts.tails import anchor_series  # noqa: E402

EVENTS_PATH = RESULTS_DIR / "events.json"
COUPONS_PATH = OUTPUT_DIR / "coupons.json"
QUICK_DIR = OUTPUT_DIR / "states_quick"
QUICK_DAYS = 30
#: Calendar days that hold ``QUICK_DAYS`` trading days (a quick ki90 note is picked by an approach inside them).
QUICK_CALENDAR_DAYS = 42
QUICK_PROBE = date(2024, 2, 29)
#: Measured seconds per ladder solve (V2 QUAD, spec Decisions block: 0.41-0.49 s); an assumption of ``--plan``.
SECONDS_PER_SOLVE = 0.45
#: Scalar price solves per coupon: two probes and the re-price check.
COUPON_SOLVES = 3
RSS_LIMIT_BYTES = 24 * 1024 ** 3
RSS_POLL_S = 30.0


class NotesFailed(PricingError):
    """Some notes raised; each left ``<book>__<inception>.error.json`` (every other note completed)."""


# ---------------------------------------------------------------------------
# The sample and the plan
# ---------------------------------------------------------------------------


def load_sample(path: Path = EVENTS_PATH) -> Dict[str, Any]:
    """``results/events.json``; fails closed when it is missing."""
    if not Path(path).exists():
        raise StudyDataError(f"{path} missing: run 02_events.py first")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sample_notes(sample: Dict[str, Any], books: Optional[Sequence[str]] = None) -> List[str]:
    """Every note id of the sample (headline and pin of each book), books in registry order, by inception."""
    wanted = list(BOOKS) if books is None else list(books)
    unknown = [b for b in wanted if b not in BOOKS]
    if unknown:
        raise StudyDataError(f"unknown books {unknown}; books are {sorted(BOOKS)}")
    return [note_id(b, inc) for b in BOOKS if b in wanted
            for inc in sorted(date.fromisoformat(d) for d in sample["books"][b]["notes"])]


def _summarise(counts: Dict[str, int], live_days: int, notes: int, workers: int) -> Dict[str, Any]:
    return {"notes": notes, "live_note_days": live_days, **counts, "workers": workers,
            "seconds_per_solve": SECONDS_PER_SOLVE, "coupon_solves": COUPON_SOLVES * notes,
            "hours": counts["solves"] * SECONDS_PER_SOLVE / max(workers, 1) / 3600.0}


def plan(chains, scans_by_book: Dict[str, Dict[date, Any]], notes: Sequence[str], workers: int, *,
         max_days: Optional[int] = None) -> Dict[str, Any]:
    """Counts and the hour estimate per book and in total, from the chain and the scan records alone (no pricing).
    ``scans_by_book[book][inception]`` is that book's scan record."""
    calendar = calendar_for(chains)
    by_day = {c.day: i for i, c in enumerate(chains)}
    keys = ("priced_days", "ko_window_days", "ki_window_days", "solves")
    per_book: Dict[str, Dict[str, Any]] = {}
    for book in BOOKS:
        totals = {k: 0 for k in keys}
        live_days, heavy, n = 0, [], 0
        for nid in notes:
            b, inc = parse_note_id(nid)
            if b != book:
                continue
            n += 1
            rec = scans_by_book[book][inc]
            counts = expected_solves(chains, rec, calendar, max_days=max_days, ki_pct=BOOKS[book])
            for k in keys:
                totals[k] += counts[k]
            live_days += by_day[rec.end_date] - by_day[inc]
            heavy.append((counts["solves"], nid, counts))
        heavy.sort(reverse=True)
        per_book[book] = {**_summarise(totals, live_days, n, workers), "heaviest": heavy[:3]}
    total_counts = {k: sum(z[k] for z in per_book.values()) for k in keys}
    total = _summarise(total_counts, sum(z["live_note_days"] for z in per_book.values()),
                       sum(z["notes"] for z in per_book.values()), workers)
    return {"books": per_book, "total": total}


def print_plan(summary: Dict[str, Any], log: Callable[[str], None] = print) -> None:
    for book, z in summary["books"].items():
        if not z["notes"]:
            continue
        log(f"book {book} (KI {BOOKS[book]:.0%}): notes {z['notes']}; live note-days (closes after inception) "
            f"{z['live_note_days']}; priced note-days {z['priced_days']}; KO-window days {z['ko_window_days']}, "
            f"KI-window days {z['ki_window_days']}; solves {z['solves']}; {z['hours']:.2f} h wall")
        for n_solves, nid, c in z["heaviest"]:
            log(f"    heaviest: {nid} {n_solves} solves ({c['priced_days']} priced days, {c['ko_window_days']} "
                f"KO-window, {c['ki_window_days']} KI-window)")
    t = summary["total"]
    log(f"total: notes {t['notes']}; live note-days {t['live_note_days']}; priced note-days {t['priced_days']}; "
        f"KO-window days {t['ko_window_days']}, KI-window days {t['ki_window_days']}; solves {t['solves']} = priced "
        f"days + {t['ko_window_days']} x 7 (surv + 5 KO shifts + wrong-way) + {t['ki_window_days']} x 4 (3 KI "
        f"shifts + kin); assumes a non-zero first KO gap on every observation (a zero gap skips the 6 shift solves)")
    log(f"estimate: {t['solves']} solves x {t['seconds_per_solve']} s / {t['workers']} workers = {t['hours']:.2f} h wall "
        f"(assumes {t['seconds_per_solve']} s per solve, the Decisions block's 0.41-0.49 s; stepping the lifecycle "
        f"adds a little); plus {t['coupon_solves']} coupon price solves ({COUPON_SOLVES} per note) run serially "
        f"before the pool, {t['coupon_solves'] * t['seconds_per_solve'] / 3600.0:.2f} h if none is cached")


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------

_WORKER: Dict[str, Any] = {}


def init_worker(chain_path: str) -> None:
    """Spawn-worker initializer: the chain, its 63-day anchors and the lifecycle scan of every book, once per
    process."""
    chains = load_chain(Path(chain_path))
    anchors = anchor_series(chains, load_prereg()["data"]["anchor_window"])
    _WORKER["chains"] = chains
    _WORKER["anchors"] = anchors
    _WORKER["scans"] = {b: {s.inception: s for s in scan(chains, anchors, ki_pct=level)} for b, level in BOOKS.items()}


def worker_job(note: str, coupon: float, fingerprint: str, out_dir: str, max_days: Optional[int]) -> Dict[str, Any]:
    """One note in a worker (its engine is built once per process and reused)."""
    if "chains" not in _WORKER:
        init_worker(str(CHAIN_PATH))
    book, inc = parse_note_id(note)
    return price_one(inc, coupon, _WORKER["chains"], _WORKER["anchors"], _WORKER["scans"][book][inc], Path(out_dir),
                     fingerprint, ki_pct=BOOKS[book], max_days=max_days)


def _file_stem(note: str) -> str:
    book, inc = parse_note_id(note)
    return f"{book}__{inc.isoformat()}"


def error_path(out_dir: Path, note: str) -> Path:
    return Path(out_dir) / f"{_file_stem(note)}.error.json"


def _record_failure(out_dir: Path, note: str, err: BaseException, log: Callable[[str], None]) -> None:
    tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
    log(f"  FAILED {note}: {err!r}\n{tb}")
    write_json_atomic(error_path(out_dir, note), {"note": note, "error": repr(err), "traceback": tb})


def run_notes(notes: Sequence[str], coupons: Dict[str, float], out_dir: Path, fingerprint: str, *,
              workers: int = 4, max_days: Optional[int] = None,
              job: Optional[Callable[[str], Dict[str, Any]]] = None,
              rss_limit_bytes: int = RSS_LIMIT_BYTES, poll_s: float = RSS_POLL_S,
              log: Callable[[str], None] = print) -> List[Path]:
    """Price every note id into ``out_dir`` (resumable, one lock per directory); returns the state paths.

    ``workers=0`` runs in-process (``job(note)`` when given: tests). A note that raises leaves an ``.error.json``
    and the others go on; ``NotesFailed`` is raised after the last. The RSS guard terminates the workers above
    ``rss_limit_bytes``; the queue is cancelled and ``RssLimitExceeded`` is raised."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(out_dir)
    try:
        for stale in out_dir.glob("*.json.tmp"):
            stale.unlink()
        for nid in notes:
            error_path(out_dir, nid).unlink(missing_ok=True)
        started = time.perf_counter()
        failed: List[str] = []
        log(f"{len(notes)} notes requested into {out_dir}")

        def done(n: int, summary: Dict[str, Any]) -> None:
            how = "reused" if summary["reused"] else f"{summary['seconds']:.0f} s, {summary['solves']} solves"
            log(f"  [{n}/{len(notes)}] {summary['note']}: {how}, {summary['live_days']} days, "
                f"{time.perf_counter() - started:.0f} s elapsed")

        if workers == 0:
            for n, nid in enumerate(notes, start=1):
                try:
                    summary = job(nid) if job is not None else worker_job(
                        nid, coupons[nid], fingerprint, str(out_dir), max_days)
                except Exception as err:
                    _record_failure(out_dir, nid, err, log)
                    failed.append(nid)
                    continue
                done(n, summary)
        else:
            if job is not None:
                raise StudyDataError("a custom job runs in-process only (workers=0)")
            guard = RssGuard(rss_limit_bytes, poll_s)
            guard.start()
            try:
                with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                         initializer=init_worker, initargs=(str(CHAIN_PATH),)) as pool:
                    futures = {pool.submit(worker_job, nid, float(coupons[nid]), fingerprint,
                                           str(out_dir), max_days): nid for nid in notes}
                    for n, fut in enumerate(as_completed(futures), start=1):
                        nid = futures[fut]
                        try:
                            summary = fut.result()
                        except Exception as err:
                            if guard.tripped.is_set():
                                pool.shutdown(wait=False, cancel_futures=True)
                                raise RssLimitExceeded(f"group RSS above {rss_limit_bytes / 1024 ** 3:.0f} GB") from None
                            _record_failure(out_dir, nid, err, log)
                            failed.append(nid)
                            continue
                        done(n, summary)
            finally:
                guard.stop()
            if guard.tripped.is_set():
                raise RssLimitExceeded(f"group RSS above {rss_limit_bytes / 1024 ** 3:.0f} GB")
            log(f"peak group RSS {guard.peak / 1024 ** 3:.2f} GB (polled every {poll_s:.0f} s)")
        if failed:
            raise NotesFailed(f"{len(failed)} of {len(notes)} notes failed (see their .error.json): {', '.join(failed)}")
        return [out_dir / f"{_file_stem(nid)}.json" for nid in notes]
    finally:
        _release_lock(lock)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def quick_notes(sample: Dict[str, Any]) -> List[str]:
    """The ``--quick`` notes: the first ``ki75`` headline note, the ``ki75`` pin note struck 2024-02-29 (the probe)
    when it is in the sample, else the first ``ki75`` pin note, and the earliest ``ki90`` pin note with a KI approach
    that starts within ``QUICK_DAYS`` trading days of its inception (so its 30 days exercise the KI window)."""
    k75, k90 = sample["books"]["ki75"], sample["books"]["ki90"]
    pins = [date.fromisoformat(p["inception"]) for p in k75["pin"]]
    picks = [note_id("ki75", date.fromisoformat(k75["headline"][0])),
             note_id("ki75", QUICK_PROBE if QUICK_PROBE in pins else pins[0])]
    pin90 = {p["inception"] for p in k90["pin"]}
    early = sorted((r["inception"], r["start"]) for r in k90["ki_approaches"] if r["inception"] in pin90
                   and (date.fromisoformat(r["start"]) - date.fromisoformat(r["inception"])).days <= QUICK_CALENDAR_DAYS)
    if not early:
        raise StudyDataError("no ki90 pin note has a KI approach in its first 30 trading days")
    picks.append(note_id("ki90", date.fromisoformat(early[0][0])))
    return picks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", action="store_true",
                    help="notes, priced note-days, window days, solves and hours per book and in total; no pricing")
    ap.add_argument("--quick", action="store_true",
                    help="two ki75 notes and a ki90 pin note, 30 days each, into states_quick/")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--books", default="", help="comma-separated books to price (default: all of ki75,ki85,ki90)")
    ap.add_argument("--notes", default="", help="comma-separated note ids '<book>:<inception>' of the sample (default: all)")
    args = ap.parse_args()

    chains = load_chain(CHAIN_PATH)
    anchors = anchor_series(chains, load_prereg()["data"]["anchor_window"])
    sample = load_sample()
    books = [b.strip() for b in args.books.split(",") if b.strip()] or None
    notes = sample_notes(sample, books)
    if args.notes:
        wanted = {n.strip() for n in args.notes.split(",")}
        missing = wanted - set(notes)
        if missing:
            raise SystemExit(f"not in the sample: {sorted(missing)}")
        notes = [n for n in notes if n in wanted]
    per_book = {b: sum(1 for n in notes if n.startswith(f"{b}:")) for b in BOOKS}
    print(f"chain {chains[0].day} .. {chains[-1].day}; {len(notes)} notes ({per_book})")

    if args.plan:
        scans_by_book = {b: {s.inception: s for s in scan(chains, anchors, ki_pct=level)} for b, level in BOOKS.items()}
        print_plan(plan(chains, scans_by_book, notes, args.workers))
        return 0

    if args.quick:
        notes = quick_notes(sample)
        out_dir, max_days, workers = QUICK_DIR, QUICK_DAYS, min(args.workers, len(notes))
        if out_dir.exists():
            shutil.rmtree(out_dir)
    else:
        out_dir, max_days, workers = STATES_DIR, None, args.workers

    fingerprint = pricing_fingerprint()
    print(f"fingerprint {fingerprint[:16]}")
    by_day = {dc.day: dc for dc in chains}
    engine = v2_engine()
    coupons = book_coupons(notes, COUPONS_PATH, coupon_stamp(),
                           lambda book, inc: solve_book_coupon(by_day[inc], engine, BOOKS[book]))
    for nid in notes:
        print(f"  coupon {nid}: {coupons[nid]:.6f}")
    try:
        paths = run_notes(notes, coupons, out_dir, fingerprint, workers=workers, max_days=max_days)
    except RssLimitExceeded as err:
        print(f"ABORTED: {err}", file=sys.stderr)
        return 3
    except NotesFailed as err:
        print(f"FAILED NOTES (every other note completed; re-run to retry them): {err}", file=sys.stderr)
        return 2
    except OutDirLocked as err:
        print(f"REFUSED: {err}", file=sys.stderr)
        return 4
    if args.quick:
        for p in paths:
            data = json.loads(p.read_text(encoding="utf-8"))
            m = data["meta"]
            print(f"{data['note']}: {len(data['days'])} days, {m['solves']} solves in {m['seconds']:.1f} s "
                  f"({m['seconds'] / max(m['solves'], 1):.2f} s per solve), KO-window {m['ko_window_days']}, "
                  f"KI-window {m['ki_window_days']}")
    print(f"wrote {len(paths)} state files to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
