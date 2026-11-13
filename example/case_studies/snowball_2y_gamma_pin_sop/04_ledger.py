"""Stage 4: the ledger pass. Replays the techniques on the recorded states and the real IM / MO marks, one slice at
a time, and keeps the two slices apart (spec C9, D11). Seller view throughout.

The stamp. Every run computes four hashes once, before anything else (``gpin.measures.stamp``): ``prereg_hash`` (the
pre-registration), ``events_hash`` (the events file), ``code_hash`` (the ledger-side code
``gpin/{black,ledger,techniques,measures}.py``) and ``inputs_hash`` (the recorded inputs: the digests of the board and
chain files and the pricing fingerprint of every state file of the sample found in the states directory). The stamp
is written into the selection, every run file, the manifest, the training tables and the payload.

``--slice train`` (blind half). Every ledger run stops at the last trading day before the pre-registered split
(``until``), so no row dated on or after the split is ever produced; only notes booked before the split are run and
only their training-dated events are measured. Those three rules are ``gpin.measures.training_scope``'s, inside the
code hash; this file only follows them. Every grid point of the pre-registration runs under the clocks asked
for; the training selection is made (``gpin.measures.training_selection``: per row the grid point with the highest
training ES10 under the primary clock); the two composites it names run too. Written: the run files, the training
tables and, last, the selection file with the picks, the ``composites`` block the registry reads and the four hashes.

``--slice test`` (unblinding). Refused with exit status 2, with no ledger run and nothing created, unless the
selection file exists and each of its four hashes equals the current files' (``gpin.measures.check_selection``; the
message names the hash that differs). A re-run of ``01_data.py`` or ``03_pricing.py`` after the selection was made
changes ``inputs_hash`` and so forces a new selection. Then the baseline, the controls, the cost rows, the pick of
every row and the composites run over the full data under the clocks asked for, plus the sensitivities asked for
under the primary clock, and the verdict tables are written to the backtest payload.

Destinations. Without ``--out-dir``: run files under ``OUTPUT_DIR/runs/<slice>/`` (untracked), the selection at
``results/selection.json`` and the payload at ``results/backtest.json`` (tracked). With ``--out-dir D``: ``D/runs/
<slice>/``, ``D/selection.json`` and ``D/backtest.json`` (``--selection`` / ``--backtest`` name them explicitly).
``--partial`` allows a states directory that holds only some notes of the sample (a smoke run, e.g. ``states_quick``):
the selection is marked partial; a barrier type without a training event is listed in ``empty``, its rows have no
pick (``id`` null) and no composite is written or run (a composite needs a member for both barriers); the test slice
of such a selection runs what was picked; a full test slice refuses it; and a partial run is never written to the
tracked results.

The runs directory (what ``05_report.py`` reads; it can render from the files of any subset of notes of one run):

* ``<book>__<inception>.json``, one per note: ``gpin.measures.run_note``'s result plus ``slice`` and ``stamp``
  (``{prereg_hash, events_hash, code_hash, inputs_hash}``). ``runs[config][clock][technique id]`` is one ledger run
  kept as ``{"events": {event id: {pnl_bp,
  cost_bp, clips, leg_days, phi_min, flags, left_out, days: [[date, hedged bp, hedge ratio], ...], gamma (KO
  events)}}, "life": {n, sum_bp, sumsq_bp, terminal_bp, completed, rows, first, last, cost_bp, im_lots_traded,
  clips, leg_days, overlay_days, phi_sum, phi_min, phi_lt1_days, flags}}`` (``gpin.measures.summarise_run``): the
  event-window P&L of every event, the sufficient statistics of the daily hedged P&L (day 0 left out) and the
  descriptive counts. ``events`` holds the note's measured events (training files: training-dated only; test files:
  all, each with its ``slice``), ``gap_observations`` the sign of ``gap_ko`` per observation. ``config`` is
  ``desk<mio>_cost<points>``; the primary one is ``desk500_cost0.2``.
* ``manifest.json``, written after the note files: slice, ``stamp`` (the four hashes), ``until``, split date, the
  notes, techniques and ``configs`` (``{config: [clocks]}``) of this run, ``partial`` and, for a partial run,
  ``events_not_measurable`` (per note, the events its short state file does not cover). Files not listed there are
  not this run's.
* ``tables.json`` (training slice only): ``gpin.measures.tables`` on the training events plus ``stamp``, the same
  shape as the backtest payload.

The selection file: ``{view, slice, split_date, until, clock, config, partial, empty, notes, events, base, rows,
picks: {barrier: {row: {id, es10, cost_bp, n_events, beats_base, grid}}}, composites, prereg_hash, events_hash,
code_hash, inputs_hash}`` (``id`` / ``es10`` / ``cost_bp`` null and ``composites`` ``{}`` for an ``empty`` barrier).

The backtest payload: ``gpin.measures.tables(..., slice_="test")`` plus ``stamp`` (the four hashes), ``selection``
(the pick per row and the composites), ``split_date``, ``data`` and ``partial``.

Before it prints a test number the report calls ``gpin.measures.check_selection`` (the selection against the current
stamp) and ``gpin.measures.check_payload`` (the stored payload and the ``runs/test`` manifest against the selection
and the current stamp); ``gpin.measures.tables`` itself refuses note files that do not carry one stamp.

A spawn pool started from this file runs the notes (``--workers``, at most 4; 1 runs in-process), with the group-RSS
guard and a lock on the runs directory. Any failed note fails the run: nothing is selected on an incomplete pass.

Run (from a script file, never stdin), with ``L`` = ``example/case_studies/snowball_2y_gamma_pin_sop/04_ledger.py``:
    python L --slice train
    python L --slice test --desk-notional 5e8 1e8 --mo-cost-pt 0.2 1.0
    python L --slice train --partial --states-dir <OUTPUT_DIR>/states_quick --out-dir <scratch>
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quantark.util.exceptions import QuantArkException  # noqa: E402

from gpin import BOARD_PATH, CHAIN_PATH, OUTPUT_DIR, RESULTS_DIR, STATES_DIR, StudyDataError, load_prereg  # noqa: E402
from gpin import measures  # noqa: E402
from gpin.board import load_boards  # noqa: E402
from gpin.ledger import CLOCKS  # noqa: E402
from gpin.techniques import build_registry  # noqa: E402
from s2ts.chain import load_chain  # noqa: E402
from s2ts.runner import (  # noqa: E402
    OutDirLocked, RssGuard, RssLimitExceeded, _acquire_lock, _release_lock, write_json_atomic,
)

EVENTS_PATH = RESULTS_DIR / "events.json"
SELECTION_NAME, BACKTEST_NAME = "selection.json", "backtest.json"
MAX_WORKERS = 4
RSS_LIMIT_BYTES = 24 * 1024 ** 3
RSS_POLL_S = 10.0


#: One run of this stage, with every path resolved (a namedtuple: this file is also loaded by path, unregistered).
Job = namedtuple("Job", ["slice", "modes", "desk_notionals", "mo_cost_pts", "states_dir", "runs_dir", "selection",
                         "backtest", "events", "board", "chain", "workers", "partial"])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slice", required=True, choices=("train", "test"))
    ap.add_argument("--mode", default="all", choices=CLOCKS + ("all",),
                    help="the clocks to run: all three, or one (the primary clock always runs: it decides)")
    ap.add_argument("--desk-notional", type=float, nargs="+", default=[],
                    help="desk notionals in CNY; the pre-registered one always runs, any other is a sensitivity")
    ap.add_argument("--mo-cost-pt", type=float, nargs="+", default=[],
                    help="option costs in index points per leg per side; any other than the pre-registered one is a "
                         "sensitivity")
    ap.add_argument("--states-dir", type=Path, default=STATES_DIR)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="scratch destination of the run files, the selection and the payload (default: the study's)")
    ap.add_argument("--selection", type=Path, default=None)
    ap.add_argument("--backtest", type=Path, default=None)
    ap.add_argument("--events", type=Path, default=EVENTS_PATH)
    ap.add_argument("--board", type=Path, default=BOARD_PATH)
    ap.add_argument("--chain", type=Path, default=CHAIN_PATH)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--partial", action="store_true",
                    help="a smoke run on a states directory that holds only some notes of the sample")
    return ap.parse_args(argv)


def resolve(args: argparse.Namespace) -> Job:
    """The run with its paths resolved. A partial run that would write a tracked result is refused."""
    prereg = load_prereg()
    primary = str(prereg["clocks"]["primary"])
    order = (primary,) + tuple(str(c) for c in prereg["clocks"]["others"])
    modes = order if args.mode == "all" else tuple(c for c in order if c in (primary, args.mode))
    out_dir = OUTPUT_DIR if args.out_dir is None else Path(args.out_dir)
    home = RESULTS_DIR if args.out_dir is None else out_dir
    selection = home / SELECTION_NAME if args.selection is None else Path(args.selection)
    backtest = home / BACKTEST_NAME if args.backtest is None else Path(args.backtest)
    if args.partial:
        tracked = [p for p in (selection, backtest) if p.resolve().parent == RESULTS_DIR.resolve()]
        if tracked or out_dir.resolve() == OUTPUT_DIR.resolve():
            raise StudyDataError("a partial run is a smoke run: give it a scratch --out-dir (and no tracked "
                                 "--selection / --backtest)")
    return Job(args.slice, modes, tuple(args.desk_notional), tuple(args.mo_cost_pt), Path(args.states_dir),
               out_dir / "runs" / args.slice, selection, backtest, Path(args.events), Path(args.board),
               Path(args.chain), max(1, min(int(args.workers), MAX_WORKERS)), bool(args.partial))


# ---------------------------------------------------------------------------
# Notes and workers
# ---------------------------------------------------------------------------


_WORKER: Dict[str, Any] = {}


def init_worker(board_path: str, chain_path: str, events_path: str) -> None:
    """Worker initializer: the boards, the chain, the events file and the pre-registration, once per process."""
    _WORKER.clear()
    _WORKER.update(boards=load_boards(Path(board_path)), chains=load_chain(Path(chain_path)),
                   sample=json.loads(Path(events_path).read_text(encoding="utf-8")), prereg=load_prereg(),
                   registries={})


def note_job(nid: str, state_path: str, technique_ids: Sequence[str], composites: Optional[Dict[str, Dict[str, str]]],
             plan: Sequence[Tuple[float, float, Sequence[str]]], until_iso: Optional[str],
             event_slice: Optional[str], partial: bool) -> Dict[str, Any]:
    """One note: every technique of ``technique_ids`` under every ``(desk notional, option cost, clocks)`` of ``plan``
    (``gpin.measures.run_note``). ``event_slice``: measure only the note's events of that slice. An event the state
    file does not cover fails the note; a ``partial`` run leaves it out and counts it (``events_not_measurable``)."""
    key = json.dumps(composites, sort_keys=True)
    if key not in _WORKER["registries"]:
        _WORKER["registries"][key] = build_registry(_WORKER["prereg"], composites)
    registry = _WORKER["registries"][key]
    states = json.loads(Path(state_path).read_text(encoding="utf-8"))
    if states.get("note") not in (None, nid):
        raise StudyDataError(f"{state_path} is the state file of {states.get('note')}, not {nid}")
    events = [e for e in measures.note_events(_WORKER["sample"], nid) if event_slice in (None, e.slice)]
    covered = measures.measurable_events(states, events) if partial else events
    note = measures.run_note(
        states, [registry[tid] for tid in technique_ids], _WORKER["boards"], _WORKER["chains"], covered,
        plan=[(measures.Config(desk, cost), tuple(modes)) for desk, cost, modes in plan],
        until=None if until_iso is None else date.fromisoformat(until_iso))
    note["events_not_measurable"] = len(events) - len(covered)
    return note


def run_notes(job: Job, files: Dict[str, Path], technique_ids: Sequence[str],
              composites: Optional[Dict[str, Dict[str, str]]], plan: Sequence[Tuple[float, float, Sequence[str]]],
              until: Optional[date], event_slice: Optional[str], *,
              log: Callable[[str], None] = print) -> Dict[str, Dict[str, Any]]:
    """``{note id: run_note result}`` for every note of ``files``, in-process (one worker) or on a spawn pool with
    the group-RSS guard. Any note that raises fails the run (the error names it)."""
    until_iso = None if until is None else until.isoformat()
    args = {nid: (nid, str(path), list(technique_ids), composites, list(plan), until_iso, event_slice, job.partial)
            for nid, path in files.items()}
    out: Dict[str, Dict[str, Any]] = {}
    if job.workers <= 1 or len(files) <= 1:
        if not _WORKER:
            init_worker(str(job.board), str(job.chain), str(job.events))
        for nid in files:
            out[nid] = note_job(*args[nid])
        return out
    guard = RssGuard(RSS_LIMIT_BYTES, RSS_POLL_S)
    guard.start()
    try:
        with ProcessPoolExecutor(max_workers=job.workers, mp_context=mp.get_context("spawn"), initializer=init_worker,
                                 initargs=(str(job.board), str(job.chain), str(job.events))) as pool:
            futures = {pool.submit(note_job, *args[nid]): nid for nid in files}
            for fut in as_completed(futures):
                nid = futures[fut]
                try:
                    out[nid] = fut.result()
                except Exception as err:
                    pool.shutdown(wait=False, cancel_futures=True)
                    if guard.tripped.is_set():
                        raise RssLimitExceeded(f"group RSS above {RSS_LIMIT_BYTES / 1024 ** 3:.0f} GB") from None
                    raise StudyDataError(f"note {nid} failed: {err!r}") from err
    finally:
        guard.stop()
    if guard.peak:
        log(f"  peak group RSS {guard.peak / 1024 ** 3:.2f} GB")
    return {nid: out[nid] for nid in files}


def write_runs(job: Job, notes: Dict[str, Dict[str, Any]], stamp: Dict[str, str], manifest: Dict[str, Any]) -> None:
    """The note files, then the manifest; note files of an earlier run that are not in this one are removed."""
    job.runs_dir.mkdir(parents=True, exist_ok=True)
    keep = set()
    for nid, note in notes.items():
        name = measures.note_file_name(nid)
        write_json_atomic(job.runs_dir / name, {**note, "slice": job.slice, "stamp": stamp})
        keep.add(name)
    for stale in job.runs_dir.iterdir():
        if measures.NOTE_FILE.match(stale.name) and stale.name not in keep:
            stale.unlink()
    write_json_atomic(job.runs_dir / "manifest.json", manifest)


# ---------------------------------------------------------------------------
# The two slices
# ---------------------------------------------------------------------------


def _plan(prereg: Dict[str, Any], job: Job) -> List[Tuple[float, float, Tuple[str, ...]]]:
    """The primary configuration under every clock asked for, then each sensitivity under the primary clock."""
    primary = measures.primary_config(prereg)
    plan = [(primary.desk_notional, primary.mo_cost_pt, job.modes)]
    for cfg in measures.sensitivity_configs(prereg, job.desk_notionals, job.mo_cost_pts):
        plan.append((cfg.desk_notional, cfg.mo_cost_pt, (str(prereg["clocks"]["primary"]),)))
    return plan


def _manifest(job: Job, stamp: Dict[str, str], notes: Dict[str, Dict[str, Any]], technique_ids: Sequence[str],
              plan: Sequence[Tuple[float, float, Sequence[str]]], until: Optional[date],
              seconds: float) -> Dict[str, Any]:
    return {"view": "seller", "slice": job.slice, "stamp": stamp, "split_date": str(load_prereg()["split_date"]),
            "until": None if until is None else until.isoformat(), "notes": list(notes),
            "techniques": list(technique_ids), "partial": job.partial,
            "events_not_measurable": {nid: n["events_not_measurable"] for nid, n in notes.items()
                                      if n["events_not_measurable"]},
            "configs": {measures.Config(desk, cost).key: list(modes) for desk, cost, modes in plan},
            "workers": job.workers, "seconds": seconds}


def run_train(job: Job, prereg: Dict[str, Any], sample: Dict[str, Any], files: Dict[str, Path],
              stamp: Dict[str, str], log: Callable[[str], None] = print) -> int:
    """The blind half: the training-slice runs, the selection, the composites, the files."""
    started = time.perf_counter()
    # the cut-off day, the notes and the event slice are measures.training_scope's (inside the code hash), not ours
    scope = measures.training_scope(load_chain(job.chain), list(files), prereg)
    until = scope.until
    train = {nid: files[nid] for nid in scope.notes}
    plan = _plan(prereg, job)
    grid_ids = list(build_registry(prereg))
    log(f"train slice: {len(train)} of {len(files)} notes booked on or before {until}; every run stops at {until}; "
        f"{len(grid_ids)} grid points; clocks {list(job.modes)}; {job.workers} worker(s)")
    notes = run_notes(job, train, grid_ids, None, plan, until, scope.event_slice, log=log)
    selection = measures.training_selection(list(notes.values()), prereg, allow_empty=job.partial)
    composites = measures.selection_composites(selection)       # None: a barrier has no pick, so no composite exists
    composite_ids = [tid for tid in build_registry(prereg, composites) if tid not in grid_ids]
    if composite_ids:
        second = run_notes(job, train, composite_ids, composites, plan, until, scope.event_slice, log=log)
        notes = {nid: measures.merge_notes(notes[nid], second[nid]) for nid in notes}
    seconds = time.perf_counter() - started
    for barrier, rows in selection["picks"].items():
        base = selection["base"][barrier]
        log(f"  {barrier.upper()}: {base['n_events']} training events, baseline ES10 {base['es10']}")
        for row, pick in rows.items():
            log(f"    {row:<10} -> {pick['id'] or 'no pick':<20} ES10 {pick['es10']}")
    log(f"  composites: {composites if composites is not None else 'none (a barrier has no training event)'}")
    technique_ids = grid_ids + composite_ids
    tables = measures.tables(list(notes.values()), prereg, sample, selection, slice_=scope.event_slice)
    lock = _acquire_lock(_ensure(job.runs_dir))
    try:
        write_runs(job, notes, stamp, _manifest(job, stamp, notes, technique_ids, plan, until, seconds))
        write_json_atomic(job.runs_dir / "tables.json", {**tables, "stamp": stamp, "until": until.isoformat(),
                                                         "split_date": scope.split_date.isoformat(),
                                                         "partial": job.partial})
    finally:
        _release_lock(lock)
    write_json_atomic(job.selection, {
        "view": "seller", "slice": scope.event_slice, "split_date": scope.split_date.isoformat(),
        "until": until.isoformat(),
        "clock": str(prereg["selection"]["clock"]), "config": measures.primary_config(prereg).key,
        "partial": job.partial, "notes": list(notes), **selection, **stamp})
    log(f"wrote {len(notes)} run files to {job.runs_dir} and the selection to {job.selection} in {seconds:.1f} s")
    return 0


def run_test(job: Job, prereg: Dict[str, Any], sample: Dict[str, Any], files: Dict[str, Path],
             stamp: Dict[str, str], log: Callable[[str], None] = print) -> int:
    """The unblinding: refused unless the selection stands; then the selected techniques over the full data."""
    selection = measures.check_selection(job.selection, stamp, allow_partial=job.partial)
    started = time.perf_counter()
    chains = load_chain(job.chain)
    composites = measures.selection_composites(selection)       # None only for a partial selection without them
    build_registry(prereg, composites)                        # fails closed on a selection the registry cannot read
    technique_ids = measures.evaluated_ids(prereg, selection)
    plan = _plan(prereg, job)
    log(f"test slice: {len(files)} notes over the full data; {len(technique_ids)} techniques; "
        f"{[measures.Config(d, c).key + ' ' + '/'.join(m) for d, c, m in plan]}; {job.workers} worker(s)")
    notes = run_notes(job, files, technique_ids, composites, plan, None, None, log=log)
    seconds = time.perf_counter() - started
    tables = measures.tables(list(notes.values()), prereg, sample, selection, slice_="test")
    payload = {**tables, "stamp": stamp, "split_date": str(prereg["split_date"]), "partial": job.partial,
               "data": {"first": chains[0].day.isoformat(), "last": chains[-1].day.isoformat()},
               "selection": {"picks": {b: {row: p["id"] for row, p in rows.items()}
                                       for b, rows in selection["picks"].items()},
                             "composites": selection["composites"]}}
    lock = _acquire_lock(_ensure(job.runs_dir))
    try:
        write_runs(job, notes, stamp, _manifest(job, stamp, notes, technique_ids, plan, None, seconds))
    finally:
        _release_lock(lock)
    write_json_atomic(job.backtest, payload)
    for barrier in measures.BARRIERS:
        table = tables["tables"][tables["primary"]["clock"]][barrier]
        log(f"  {barrier.upper()} ({table['book']}): {table['n_events']} test events in {table['n_weeks']} weeks, "
            f"decidable {table['decidable']}, wins {table['wins']}")
    log(f"wrote {len(notes)} run files to {job.runs_dir} and the payload to {job.backtest} in {seconds:.1f} s")
    return 0


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _WORKER.clear()
    try:
        job = resolve(args)
        prereg = load_prereg()
        if not job.events.exists():
            raise StudyDataError(f"{job.events} missing: run 02_events.py first")
        sample = json.loads(job.events.read_text(encoding="utf-8"))
        files = measures.state_files(job.states_dir, sample, job.partial)
        # the four hashes, once per run: the per-note jobs hash nothing
        stamp = measures.stamp(job.events, job.board, job.chain, files)
        return (run_train if job.slice == "train" else run_test)(job, prereg, sample, files, stamp)
    except RssLimitExceeded as err:
        print(f"ABORTED: {err}", file=sys.stderr)
        return 3
    except OutDirLocked as err:
        print(f"REFUSED: {err}", file=sys.stderr)
        return 4
    except QuantArkException as err:
        print(f"REFUSED: {err}", file=sys.stderr)
        return 2
    finally:
        _WORKER.clear()


if __name__ == "__main__":
    sys.exit(main())
