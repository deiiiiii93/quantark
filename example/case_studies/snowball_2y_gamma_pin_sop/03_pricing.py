"""Stage 3: the pricing pass. Per note of the sample, per live day, a spot ladder of QUAD V2 price / delta / gamma for
every hedge model whose window the day is in (``gpin.pricing``).

Reads ``results/events.json`` (the note list, ``02_events.py``) and ``OUTPUT_DIR/chain.pkl`` (``01_data.py``).
Coupons: one per inception, solved under hold_forward by ``s2ts.runner.coupons_for``, cached in
``OUTPUT_DIR/coupons.json``. One atomic JSON per note in ``OUTPUT_DIR/states/``, resumable (a file is reused only
when its stamped fingerprint matches: see ``gpin.pricing.pricing_fingerprint``). A failed note leaves
``<inception>.error.json`` and the exit status is 2; every other note completes. Runs on a spawn pool started from this
file, with the group-RSS guard and a lock file so two runs cannot share the directory.

Run (from a script file, never stdin; long runs under ``nohup caffeinate -i -m -s``):
    python example/case_studies/snowball_2y_gamma_pin_sop/03_pricing.py --plan [--workers 4]
    python example/case_studies/snowball_2y_gamma_pin_sop/03_pricing.py --quick
    python example/case_studies/snowball_2y_gamma_pin_sop/03_pricing.py [--workers 4] [--notes 2024-02-29,2023-10-09]
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

from gpin import CHAIN_PATH, OUTPUT_DIR, RESULTS_DIR, STATES_DIR, StudyDataError, load_prereg  # noqa: E402
from gpin.events import scan  # noqa: E402
from gpin.pricing import calendar_for, expected_solves, price_one, pricing_fingerprint  # noqa: E402
from s2ts.chain import load_chain  # noqa: E402
from s2ts.runner import (  # noqa: E402
    OutDirLocked, RssGuard, RssLimitExceeded, _acquire_lock, _release_lock, coupons_for, write_json_atomic,
)
from s2ts.tails import anchor_series  # noqa: E402

EVENTS_PATH = RESULTS_DIR / "events.json"
COUPONS_PATH = OUTPUT_DIR / "coupons.json"
QUICK_DIR = OUTPUT_DIR / "states_quick"
QUICK_DAYS = 30
#: Measured seconds per ladder solve (V2 QUAD, spec Decisions block: 0.41-0.49 s); an assumption of ``--plan``.
SECONDS_PER_SOLVE = 0.45
RSS_LIMIT_BYTES = 24 * 1024 ** 3
RSS_POLL_S = 30.0


class NotesFailed(RuntimeError):
    """Some notes raised; each left ``<inception>.error.json`` (every other note completed)."""


# ---------------------------------------------------------------------------
# The sample and the plan
# ---------------------------------------------------------------------------


def load_sample(path: Path = EVENTS_PATH) -> Dict[str, Any]:
    """``results/events.json``; fails closed when it is missing."""
    if not Path(path).exists():
        raise StudyDataError(f"{path} missing: run 02_events.py first")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sample_notes(sample: Dict[str, Any]) -> List[date]:
    """Every note of the sample (headline and pin), by inception."""
    return sorted(date.fromisoformat(d) for d in sample["notes"])


def plan(chains, scans_by_day, notes: Sequence[date], workers: int, *, max_days: Optional[int] = None) -> Dict[str, Any]:
    """Counts and the hour estimate, from the chain and the scan records alone (no pricing)."""
    calendar = calendar_for(chains)
    totals = {"priced_days": 0, "ko_window_days": 0, "ki_window_days": 0, "solves": 0}
    live_days = 0
    by_day = {c.day: i for i, c in enumerate(chains)}
    heavy = []
    for inc in notes:
        rec = scans_by_day[inc]
        counts = expected_solves(chains, rec, calendar, max_days=max_days)
        for k in totals:
            totals[k] += counts[k]
        live_days += by_day[rec.end_date] - by_day[inc]
        heavy.append((counts["solves"], inc, counts))
    heavy.sort(reverse=True)
    hours = totals["solves"] * SECONDS_PER_SOLVE / max(workers, 1) / 3600.0
    return {"notes": len(notes), "live_note_days": live_days, **totals, "workers": workers,
            "seconds_per_solve": SECONDS_PER_SOLVE, "hours": hours, "heaviest": heavy[:3]}


def print_plan(summary: Dict[str, Any]) -> None:
    print(f"notes {summary['notes']}; live note-days (closes after inception) {summary['live_note_days']}; "
          f"priced days {summary['priced_days']} (day 0 and every alive close)")
    print(f"KO-window days {summary['ko_window_days']}, KI-window days {summary['ki_window_days']} "
          f"(counted from the chain and the scan with the pass's own window rules)")
    print(f"solves {summary['solves']} = priced days + {summary['ko_window_days']} x 7 (surv + 5 KO shifts + wrong-way) "
          f"+ {summary['ki_window_days']} x 4 (3 KI shifts + kin); assumes a non-zero first KO gap on every "
          f"observation (a zero gap skips the 6 shift solves)")
    print(f"estimate: {summary['solves']} solves x {summary['seconds_per_solve']} s / {summary['workers']} workers "
          f"= {summary['hours']:.2f} h wall (assumes {summary['seconds_per_solve']} s per solve, the Decisions block's "
          f"0.41-0.49 s; stepping the lifecycle adds a little)")
    for n_solves, inc, c in summary["heaviest"]:
        print(f"  heaviest: {inc} {n_solves} solves ({c['priced_days']} priced days, {c['ko_window_days']} KO-window, "
              f"{c['ki_window_days']} KI-window)")


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------

_WORKER: Dict[str, Any] = {}


def init_worker(chain_path: str) -> None:
    """Spawn-worker initializer: the chain, its 63-day anchors and the lifecycle scan, once per process."""
    chains = load_chain(Path(chain_path))
    anchors = anchor_series(chains, load_prereg()["data"]["anchor_window"])
    _WORKER["chains"] = chains
    _WORKER["anchors"] = anchors
    _WORKER["scans"] = {s.inception: s for s in scan(chains, anchors)}


def worker_job(inception_iso: str, coupon: float, fingerprint: str, out_dir: str,
               max_days: Optional[int]) -> Dict[str, Any]:
    """One note in a worker (its engine is built once per process and reused)."""
    if "chains" not in _WORKER:
        init_worker(str(CHAIN_PATH))
    inc = date.fromisoformat(inception_iso)
    return price_one(inc, coupon, _WORKER["chains"], _WORKER["anchors"], _WORKER["scans"][inc], Path(out_dir),
                     fingerprint, max_days=max_days)


def error_path(out_dir: Path, inception: date) -> Path:
    return Path(out_dir) / f"{inception.isoformat()}.error.json"


def _record_failure(out_dir: Path, inc: date, err: BaseException, log: Callable[[str], None]) -> None:
    tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
    log(f"  FAILED {inc}: {err!r}\n{tb}")
    write_json_atomic(error_path(out_dir, inc), {"inception": inc.isoformat(), "error": repr(err), "traceback": tb})


def run_notes(notes: Sequence[date], coupons: Dict[date, float], out_dir: Path, fingerprint: str, *,
              workers: int = 4, max_days: Optional[int] = None,
              job: Optional[Callable[[date], Dict[str, Any]]] = None,
              rss_limit_bytes: int = RSS_LIMIT_BYTES, poll_s: float = RSS_POLL_S,
              log: Callable[[str], None] = print) -> List[Path]:
    """Price every note into ``out_dir`` (resumable, one lock per directory); returns the state paths.

    ``workers=0`` runs in-process (``job(inception)`` when given: tests). A note that raises leaves an
    ``.error.json`` and the others go on; ``NotesFailed`` is raised after the last. The RSS guard terminates the
    workers above ``rss_limit_bytes``; the queue is cancelled and ``RssLimitExceeded`` is raised."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(out_dir)
    try:
        for stale in out_dir.glob("*.json.tmp"):
            stale.unlink()
        for inc in notes:
            error_path(out_dir, inc).unlink(missing_ok=True)
        started = time.perf_counter()
        failed: List[str] = []
        log(f"{len(notes)} notes requested into {out_dir}")

        def done(n: int, summary: Dict[str, Any]) -> None:
            how = "reused" if summary["reused"] else f"{summary['seconds']:.0f} s, {summary['solves']} solves"
            log(f"  [{n}/{len(notes)}] {summary['inception']}: {how}, {summary['live_days']} days, "
                f"{time.perf_counter() - started:.0f} s elapsed")

        if workers == 0:
            for n, inc in enumerate(notes, start=1):
                try:
                    summary = job(inc) if job is not None else worker_job(
                        inc.isoformat(), coupons[inc], fingerprint, str(out_dir), max_days)
                except Exception as err:
                    _record_failure(out_dir, inc, err, log)
                    failed.append(inc.isoformat())
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
                    futures = {pool.submit(worker_job, inc.isoformat(), float(coupons[inc]), fingerprint,
                                           str(out_dir), max_days): inc for inc in notes}
                    for n, fut in enumerate(as_completed(futures), start=1):
                        inc = futures[fut]
                        try:
                            summary = fut.result()
                        except Exception as err:
                            if guard.tripped.is_set():
                                pool.shutdown(wait=False, cancel_futures=True)
                                raise RssLimitExceeded(f"group RSS above {rss_limit_bytes / 1024 ** 3:.0f} GB") from None
                            _record_failure(out_dir, inc, err, log)
                            failed.append(inc.isoformat())
                            continue
                        done(n, summary)
            finally:
                guard.stop()
            if guard.tripped.is_set():
                raise RssLimitExceeded(f"group RSS above {rss_limit_bytes / 1024 ** 3:.0f} GB")
            log(f"peak group RSS {guard.peak / 1024 ** 3:.2f} GB (polled every {poll_s:.0f} s)")
        if failed:
            raise NotesFailed(f"{len(failed)} of {len(notes)} notes failed (see their .error.json): {', '.join(failed)}")
        return [out_dir / f"{inc.isoformat()}.json" for inc in notes]
    finally:
        _release_lock(lock)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def quick_notes(sample: Dict[str, Any]) -> List[date]:
    """One headline note and the pin note struck 2024-02-29 (the probe) when it is in the sample, else the first pin
    note."""
    headline = [date.fromisoformat(d) for d in sample["headline"]]
    pins = [date.fromisoformat(p["inception"]) for p in sample["pin"]]
    probe = date(2024, 2, 29)
    picks = [headline[0], probe if probe in pins else pins[0]]
    return picks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", action="store_true", help="notes, live note-days, solves and the hour estimate; no pricing")
    ap.add_argument("--quick", action="store_true", help="a headline note and the probe pin note, 30 days each, into states_quick/")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--notes", default="", help="comma-separated inception dates of the sample (default: all)")
    args = ap.parse_args()

    chains = load_chain(CHAIN_PATH)
    anchors = anchor_series(chains, load_prereg()["data"]["anchor_window"])
    sample = load_sample()
    notes = sample_notes(sample)
    if args.notes:
        wanted = {date.fromisoformat(s.strip()) for s in args.notes.split(",")}
        missing = wanted - set(notes)
        if missing:
            raise SystemExit(f"not in the sample: {sorted(missing)}")
        notes = [d for d in notes if d in wanted]
    print(f"chain {chains[0].day} .. {chains[-1].day}; {len(notes)} notes "
          f"({sum(1 for d in notes if d.isoformat() in sample['headline'])} headline)")

    if args.plan:
        scans_by_day = {s.inception: s for s in scan(chains, anchors)}
        print_plan(plan(chains, scans_by_day, notes, args.workers))
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
    coupons = coupons_for(notes, chains, COUPONS_PATH)
    for inc in notes:
        print(f"  coupon {inc}: {coupons[inc]:.6f}")
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
            print(f"{data['inception']}: {len(data['days'])} days, {m['solves']} solves in {m['seconds']:.1f} s "
                  f"({m['seconds'] / max(m['solves'], 1):.2f} s per solve), KO-window {m['ko_window_days']}, "
                  f"KI-window {m['ki_window_days']}")
    print(f"wrote {len(paths)} state files to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
