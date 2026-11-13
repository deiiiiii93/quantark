"""The report (spec C10, AC14, D13): aggregation and the Bloomberg x Swiss HTML, told from the snowball seller's side.

Two layers, both pure:

* ``decide`` (the unblinding guard, D11 with the four-hash ruling) and ``build_model``: from the training selection,
  the stored runs (``OUTPUT_DIR/runs/{train,test}``), the backtest payload, the recorded states and the committed
  research files to one report model (a plain dict). Nothing here runs the ledger or the engine.
* ``render``: the model to one self-contained HTML page (``docs/UI_LANGUAGE.md``: frozen function bar and numbered
  menu, light/dark switch, both colour schemes, inline SVG, no external resource), drawn with the sibling study's
  ``s2ts.html`` primitives (imported read-only).

The guard. Test numbers (the backtest payload, the ``runs/test`` files) are read and printed only when the four
hashes of the current files can be computed (``gpin.measures.stamp``: pre-registration, events file, ledger-side
code, recorded inputs), ``gpin.measures.check_selection`` accepts the training selection, the payload exists and
``gpin.measures.check_payload`` accepts it and the ``runs/test`` manifest, and every test note file carries the
current stamp. In every other case the page shows the research, the sample and the training figures that are
current, and a banner naming the guard that failed. Training runs written under other files are not shown either.
Without a valid test payload the recorded states are read only up to the split date (the reserve, the first-day
hedge step); the illustrative ladders are always taken from training-dated days.

Every missing input degrades its own section to a stated "not available"; nothing is invented.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from quantark.util.exceptions import QuantArkException
from quantark.util.numerical import Tolerance, is_close, is_valid_number, is_zero

from s2ts import NOTIONAL
from s2ts import html as h
from s2ts.chain import load_chain

from . import BOARD_PATH, CHAIN_PATH, LADDER, OUTPUT_DIR, RESULTS_DIR, STATES_DIR, load_prereg, prereg_hash
from . import measures
from .board import load_boards
from .techniques import TRUE, grid_id, hedge_delta

#: CNY per note to bp of notional.
BP = measures.BP
HASHES: Tuple[str, ...] = ("prereg_hash", "events_hash", "code_hash", "inputs_hash")
HASH_WORDS = {"prereg_hash": "the pre-registration", "events_hash": "the events file",
              "code_hash": "the ledger-side code",
              "inputs_hash": "the recorded inputs (the board, the chain or a state file's pricing fingerprint)"}
#: The model of the survived branch and of the knocked-in note: the comparison line of the illustrative ladders.
SURV, KIN = "surv", "kin"
#: The ledger flags the descriptive extras count (refusals, cuts, position limits).
EXTRA_FLAGS: Tuple[str, ...] = ("overlay_no_series", "overlay_no_strike", "overlay_no_vol", "overlay_not_openable",
                                "volume_cut", "series_limit", "refused_open")
SECTIONS: Tuple[Tuple[str, str], ...] = (
    ("design", "TESTED"), ("ko", "KO VERDICT"), ("ki", "KI VERDICT"), ("kigap", "KI GAP"), ("clocks", "CLOCKS"),
    ("sens", "SENSITIVITY"), ("gamma", "GAMMA"), ("pnl", "EVENT P&L"), ("extras", "EXTRAS"),
    ("train", "TRAINING PICKS"), ("limits", "LIMITS"), ("sources", "SOURCES"))


@dataclass(frozen=True)
class ReportPaths:
    """Where the report reads from. ``runs_dir`` holds ``train/`` and ``test/``; ``partial`` admits a smoke run (a
    partial selection and a states directory holding only some notes of the sample)."""

    runs_dir: Path = OUTPUT_DIR / "runs"
    selection: Path = RESULTS_DIR / "selection.json"
    backtest: Path = RESULTS_DIR / "backtest.json"
    states_dir: Path = STATES_DIR
    events: Path = RESULTS_DIR / "events.json"
    board: Path = BOARD_PATH
    chain: Path = CHAIN_PATH
    probe: Path = RESULTS_DIR / "ki_gap_probe.json"
    partial: bool = False


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


@dataclass
class Guard:
    """What the report may show. ``current``: the four hashes of the current files (``None`` when the recorded
    inputs cannot be hashed). ``selection``: the training selection when every computable hash matches (shown as
    training figures); ``payload``: the backtest payload, only when ``test_ok``. ``problems``: plain sentences, each
    a guard that failed."""

    hashes: Dict[str, Optional[str]]
    current: Optional[Dict[str, str]]
    selection: Optional[Dict[str, Any]] = None
    payload: Optional[Dict[str, Any]] = None
    test_ok: bool = False
    training_ok: bool = False
    train_notes: List[Dict[str, Any]] = field(default_factory=list)
    train_listed: int = 0
    train_stamp: Optional[Dict[str, str]] = None
    test_notes: List[Dict[str, Any]] = field(default_factory=list)
    state_files: Dict[str, Path] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)
    training_problems: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """The guard's verdict as the model carries it."""
        return {"hashes": dict(self.hashes), "current": self.current, "test_ok": self.test_ok,
                "training_ok": self.training_ok, "selection_ok": self.selection is not None,
                "problems": list(self.problems), "training_problems": list(self.training_problems),
                "notes": list(self.notes), "train_present": len(self.train_notes), "train_listed": self.train_listed,
                "train_stamp": self.train_stamp, "test_present": len(self.test_notes)}


def _load(path: Path) -> Optional[Dict[str, Any]]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _stale(stored: Optional[Mapping[str, Any]], hashes: Mapping[str, Optional[str]]) -> List[str]:
    """The hashes a stored stamp (or selection) does not share with the current files (only the computable ones)."""
    stored = stored or {}
    return [k for k in HASHES if hashes.get(k) is not None and stored.get(k) != hashes[k]]


def _stale_words(keys: Sequence[str]) -> str:
    return "; ".join(f"{HASH_WORDS[k]} changed ({k})" for k in keys)


def _note_files(directory: Path) -> Dict[str, Path]:
    """``{note id: run file}`` of a runs directory (the manifest and the tables are not note files)."""
    out: Dict[str, Path] = {}
    if Path(directory).is_dir():
        for path in sorted(Path(directory).iterdir()):
            m = measures.NOTE_FILE.match(path.name)
            if m:
                out[f"{m['book']}:{m['inception']}"] = path
    return out


def _hashes(paths: ReportPaths, sample: Optional[Mapping[str, Any]]) -> Tuple[Dict[str, Optional[str]], Dict[str, Path],
                                                                              Optional[str]]:
    """The current hashes (``None`` where a file is missing), the state files they cover and why the recorded
    inputs could not be hashed."""
    hashes: Dict[str, Optional[str]] = {"prereg_hash": prereg_hash(), "events_hash": None,
                                        "code_hash": measures.code_hash(), "inputs_hash": None}
    files: Dict[str, Path] = {}
    if Path(paths.events).exists():
        hashes["events_hash"] = measures.file_hash(paths.events)
    if sample is None:
        return hashes, files, "the events file is missing, so the recorded inputs cannot be listed"
    try:
        files = measures.state_files(paths.states_dir, sample, paths.partial)
        hashes["inputs_hash"] = measures.inputs_hash(paths.board, paths.chain, measures.state_fingerprints(files))
    except QuantArkException as err:
        return hashes, files, f"the recorded inputs (the board, the chain or the state files) could not be hashed: {err}"
    return hashes, files, None


def decide(paths: ReportPaths, sample: Optional[Mapping[str, Any]]) -> Guard:
    """The unblinding guard and the training-run check (see the module docstring)."""
    hashes, files, inputs_problem = _hashes(paths, sample)
    complete = all(hashes[k] is not None for k in HASHES)
    guard = Guard(hashes=hashes, current=dict(hashes) if complete else None)  # type: ignore[arg-type]
    if not files and sample is not None:
        try:
            files = measures.state_files(paths.states_dir, sample, True)
        except QuantArkException:
            files = {}
    guard.state_files = files
    if inputs_problem:
        guard.problems.append(inputs_problem + "; without the four current hashes no test number can be checked")

    # the training selection
    raw = _load(paths.selection)
    if raw is None:
        guard.problems.append(f"there is no training selection file ({Path(paths.selection).name}): run the "
                              "training slice first")
    else:
        stale = _stale(raw, hashes)
        if stale:
            guard.problems.append("the training selection was made under other files: " + _stale_words(stale))
        elif raw.get("partial", True) and not paths.partial:
            guard.problems.append("the training selection was made on a partial set of states (a smoke run)")
        else:
            guard.selection = raw
    selection_valid = False
    if guard.selection is not None and guard.current is not None:
        try:
            measures.check_selection(paths.selection, guard.current, allow_partial=paths.partial)
            selection_valid = True
        except measures.SelectionRefused as err:
            guard.problems.append(str(err))
            guard.selection = None

    # the test payload and the test runs: read only behind a valid selection
    payload_ok = False
    if selection_valid:
        payload = _load(paths.backtest)
        if payload is None:
            guard.problems.append(f"there is no test payload ({Path(paths.backtest).name}): the test slice has not "
                                  "been run")
        elif payload.get("slice") != "test":
            guard.problems.append("the test payload is not of the test slice")
        elif payload.get("partial", True) and not paths.partial:
            guard.problems.append("the test payload was written by a partial run (a smoke run)")
        else:
            try:
                measures.check_payload(payload, guard.selection, guard.current)
                payload_ok = True
                guard.payload = payload
            except measures.PayloadRefused as err:
                guard.problems.append(f"the test payload is stale: {err}")
        if payload_ok:
            payload_ok = _check_test_runs(paths, guard)
        if not payload_ok:
            guard.payload, guard.test_notes = None, []
    guard.test_ok = payload_ok

    _check_training(paths, guard)
    return guard


def _check_test_runs(paths: ReportPaths, guard: Guard) -> bool:
    """The ``runs/test`` files behind a valid payload: the manifest passes ``check_payload`` and every note file it
    lists carries the current stamp. No directory: the payload alone (the charts that need run files say so)."""
    directory = Path(paths.runs_dir) / "test"
    found = _note_files(directory)
    manifest = _load(directory / "manifest.json")
    if not found and manifest is None:
        guard.notes.append("no test run files: the charts drawn from them are not available")
        return True
    if manifest is None:
        guard.problems.append("the test run files have no manifest (an incomplete run)")
        return False
    try:
        measures.check_payload(manifest, guard.selection, guard.current)
    except measures.PayloadRefused as err:
        guard.problems.append(f"the test runs' manifest is stale: {err}")
        return False
    notes = []
    for nid in manifest.get("notes", []):
        if nid not in found:
            continue
        note = _load(found[nid])
        if note.get("stamp") != guard.current:
            guard.problems.append(f"the test run files do not carry one stamp, the current one ({nid} differs)")
            return False
        notes.append(note)
    guard.test_notes = notes
    return True


def _check_training(paths: ReportPaths, guard: Guard) -> None:
    """The training runs: shown only when their stamp is the current files' (and the selection's)."""
    directory = Path(paths.runs_dir) / "train"
    found = _note_files(directory)
    manifest = _load(directory / "manifest.json")
    listed = [nid for nid in (manifest or {}).get("notes", found)]
    guard.train_listed = len(listed)
    if not found:
        guard.training_problems.append("there is no training run file")
        return
    stamps = []
    notes = []
    for nid in listed:
        if nid in found:
            note = _load(found[nid])
            notes.append(note)
            stamps.append(note.get("stamp"))
    if manifest is not None:
        stamps.append(manifest.get("stamp"))
    if not notes or any(s != stamps[0] for s in stamps):
        guard.training_problems.append("Training figures are not shown: the training run files do not carry one stamp")
        guard.problems.extend(guard.training_problems)
        return
    stamp = stamps[0] or {}
    stale = _stale(stamp, guard.hashes)
    if not stale and guard.selection is not None:
        stale = [k for k in HASHES if guard.selection.get(k) != stamp.get(k)]
    if stale:
        guard.training_problems.append("Training figures are not shown as current: the training runs were written "
                                       "under other files: " + _stale_words(stale))
        guard.problems.extend(guard.training_problems)
        return
    if guard.hashes.get("inputs_hash") is None:
        guard.notes.append("the training runs could not be checked against the recorded inputs (no states)")
    guard.training_ok, guard.train_notes, guard.train_stamp = True, notes, dict(stamp)


# ---------------------------------------------------------------------------
# The recorded states: reserve, first-day hedge step, illustrative ladders
# ---------------------------------------------------------------------------


def _stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    a = np.asarray([v for v in values if is_valid_number(v)], dtype=float)
    if not a.size:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)), "min": float(a.min()),
            "max": float(a.max())}


def _is_ko_model(model: str) -> bool:
    return model == SURV or model.startswith("ko_")


def _ladder_profile(record: Mapping[str, Any], level: float, models: Sequence[str]) -> Dict[str, Any]:
    """The seller's position delta (% of notional) and position gamma (change of position delta, % of notional, for
    a 1% move) across the spot ladder of one recorded day: ``position = -engine``."""
    spot = float(record["spot"])
    ladder = [spot * (1.0 + x) for x in LADDER]
    out: Dict[str, Any] = {"x_pct": [100.0 * (s / level - 1.0) for s in ladder], "position_delta_pct": {},
                           "position_gamma_pct": {}}
    for m in models:
        st = record["states"][m]
        out["position_delta_pct"][m] = [None if d is None else -float(d) * s / NOTIONAL * 100.0
                                        for d, s in zip(st["engine_delta"], ladder)]
        out["position_gamma_pct"][m] = [None if g is None else -float(g) * s * s * 0.01 / NOTIONAL * 100.0
                                        for g, s in zip(st["engine_gamma"], ladder)]
    return out


def summarise_states(files: Mapping[str, Path], *, split: date, cutoff: Optional[date]) -> Dict[str, Any]:
    """One pass over the recorded state files (``{note id: path}``).

    ``windows[book][model]``: on the first day of each window of a hedge model (a KO model: the first day it is
    priced for an observation; a KI model: the first day of a run of days on which it is priced) the reserve a
    booking on that model would carry, ``reserve_bp`` = seller PV on the true model minus seller PV on the model =
    engine price of the model minus engine price of the true note at the ladder centre, in bp of notional (positive:
    a reserve held), and the hedge step the technique takes when it switches on, ``step_pct`` = ``d(model) -
    d(true)`` (1% bump deltas) x spot / notional, in % of notional (positive: buy IM). Only days before ``cutoff``
    are read (the split date unless a valid test payload exists; ``None`` reads every day).

    ``illustrative``: two training-dated days (before ``split`` whatever the cutoff), chosen by a fixed rule. KO: on
    the KO book, the T-1 close (one trading day before a KO observation) with the survived branch priced whose spot
    is nearest the KO level; KI: on the KI book, a day not knocked in with the knocked-in model priced whose spot is
    nearest the KI level. Ties: earlier date, then note id. Each carries the seller's ladders of the true model and
    of the comparison model (``surv`` / ``kin``)."""
    prereg = load_prereg()
    ko_book, ki_book = str(prereg["verdict"]["ko_book"]), str(prereg["verdict"]["ki_book"])
    acc: Dict[str, Dict[str, Dict[str, List[Any]]]] = {}
    best: Dict[str, Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]]] = {"ko": None, "ki": None}
    skipped = 0
    c = len(LADDER) // 2
    for nid, path in files.items():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        book = str(data["book"])
        ko_level, ki_level = float(data["ko_level"]), float(data["ki_level"])
        prev_models: set = set()
        prev_next = None
        for rec in data["days"]:
            day = date.fromisoformat(rec["date"])
            states = rec.get("states") or {}
            models = set(states)
            if TRUE in states and (cutoff is None or day < cutoff):
                spot = float(rec["spot"])
                true = states[TRUE]
                for m in sorted(models - {TRUE}):
                    opens = m not in prev_models or (_is_ko_model(m) and rec.get("next_ko") != prev_next)
                    if not opens:
                        continue
                    p_m, p_t = states[m]["engine_price"][c], true["engine_price"][c]
                    if p_m is None or p_t is None:            # a null centre price: no reserve that day
                        skipped += 1
                        continue
                    try:
                        step = (hedge_delta(states[m], spot) - hedge_delta(true, spot)) * spot / NOTIONAL * 100.0
                    except QuantArkException:                 # a null bump price: no hedge step that day
                        skipped += 1
                        continue
                    reserve = (float(p_m) - float(p_t)) * BP
                    cell = acc.setdefault(book, {}).setdefault(m, {"reserve": [], "step": [], "first": []})
                    cell["reserve"].append(reserve)
                    cell["step"].append(step)
                    cell["first"].append(rec["date"])
            if day < split and rec.get("alive") and TRUE in states:
                spot = float(rec["spot"])
                if book == ko_book and SURV in states and rec.get("days_to_ko") == 1:
                    key = (abs(spot / ko_level - 1.0), rec["date"], nid)
                    if best["ko"] is None or key < best["ko"][0]:
                        best["ko"] = (key, {"note": nid, "rec": rec, "level": ko_level, "comp": SURV})
                if book == ki_book and not rec.get("knocked_in") and KIN in states:
                    key = (spot / ki_level - 1.0, rec["date"], nid)
                    if best["ki"] is None or key < best["ki"][0]:
                        best["ki"] = (key, {"note": nid, "rec": rec, "level": ki_level, "comp": KIN})
            prev_models, prev_next = models, rec.get("next_ko")
    windows = {book: {m: {"n": len(cell["first"]), "first": cell["first"], "reserve_bp": _stats(cell["reserve"]),
                          "step_pct": _stats(cell["step"])} for m, cell in sorted(models_.items())}
               for book, models_ in sorted(acc.items())}
    illustrative: Dict[str, Optional[Dict[str, Any]]] = {}
    for kind, item in best.items():
        if item is None:
            illustrative[kind] = None
            continue
        info = item[1]
        rec, models_ = info["rec"], [TRUE, info["comp"]]
        illustrative[kind] = {"note": info["note"], "date": rec["date"], "spot": float(rec["spot"]),
                              "level": info["level"], "distance_pct": 100.0 * (float(rec["spot"]) / info["level"] - 1.0),
                              "days_to_ko": rec.get("days_to_ko"), "models": models_,
                              **_ladder_profile(rec, info["level"], models_)}
    return {"n_files": len(files), "cutoff": None if cutoff is None else cutoff.isoformat(), "split": split.isoformat(),
            "windows": windows, "skipped": skipped, "illustrative": illustrative}


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def _week_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return len({r["week"] for r in rows})


def sample_sizes(sample: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Per book, barrier type and slice: the events measured and their distinct ISO weeks (``events.json``)."""
    out = []
    for book, b in sample["books"].items():
        for kind, key in (("ko", "ko_events"), ("ki", "ki_approaches")):
            rows = b.get(key) or []
            if not rows:
                continue
            for slice_ in ("train", "test"):
                sel = [r for r in rows if r["slice"] == slice_]
                extra = (sum(1 for r in sel if r.get("fired")) if kind == "ko"
                         else sum(1 for r in sel if r.get("knocked_in")))
                out.append({"book": book, "kind": kind, "slice": slice_, "events": len(sel), "weeks": _week_count(sel),
                            "hits": extra, "notes": len(b.get("notes", {}))})
    return out


def _cells(note: Mapping[str, Any], cfg: str, clock: str) -> Mapping[str, Any]:
    return note.get("runs", {}).get(cfg, {}).get(clock, {})


def event_pnl(notes: Sequence[Mapping[str, Any]], *, cfg: str, clock: str, kind: str, book: str, slice_: str,
              ids: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """``{technique: {event id: window P&L (bp)}}`` over the stored note files, for one barrier type, book and slice."""
    out: Dict[str, Dict[str, float]] = {tid: {} for tid in ids}
    for note in notes:
        if note.get("book") != book:
            continue
        cells = _cells(note, cfg, clock)
        for eid, ev in note.get("events", {}).items():
            if ev["kind"] != kind or ev["slice"] != slice_:
                continue
            for tid in ids:
                cell = cells.get(tid, {}).get("events", {}).get(eid)
                if cell is not None and not cell.get("left_out"):
                    out[tid][eid] = float(cell["pnl_bp"])
    return {tid: v for tid, v in out.items() if v}


def _shown_ids(selection: Optional[Mapping[str, Any]], kind: str, available: Sequence[str]) -> List[str]:
    """The baseline, the training pick of every row of the barrier and the composites, as far as they ran."""
    ids = ["base"]
    if selection is not None:
        for row in selection.get("rows", {}).get(kind, []):
            pick = selection["picks"][kind][row]["id"]
            if pick and pick not in ids:
                ids.append(pick)
    ids += ["sop_futures", "sop_overlay"]
    return [tid for tid in ids if tid in available]


def _path(notes: Sequence[Mapping[str, Any]], *, cfg: str, clock: str, kind: str, book: str, slice_: str,
          ids: Sequence[str]) -> Optional[Dict[str, Any]]:
    """The event with the worst baseline window P&L (ties: the earlier event id) and its daily rows per technique."""
    pnl = event_pnl(notes, cfg=cfg, clock=clock, kind=kind, book=book, slice_=slice_, ids=["base"]).get("base", {})
    if not pnl:
        return None
    eid = min(pnl, key=lambda e: (pnl[e], e))
    owner = next(n for n in notes if eid in n.get("events", {}))
    cells = _cells(owner, cfg, clock)
    days = {tid: cells[tid]["events"][eid]["days"] for tid in ids if tid in cells and eid in cells[tid]["events"]}
    return {"event": eid, "date": owner["events"][eid]["date"], "pnl": {tid: cells[tid]["events"][eid]["pnl_bp"]
                                                                        for tid in days}, "days": days}


def _training_tables(paths: ReportPaths, guard: Guard, prereg: Mapping[str, Any],
                     sample: Optional[Mapping[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not guard.training_ok:
        return None, "; ".join(guard.training_problems) or "no current training run"
    if sample is None:
        return None, "the events file is missing"
    if guard.selection is not None:
        try:
            return measures.tables(guard.train_notes, prereg, sample, guard.selection, slice_="train"), None
        except QuantArkException as err:
            return None, f"the training files could not be aggregated: {err}"
    stored = _load(Path(paths.runs_dir) / "train" / "tables.json")
    if stored is not None and stored.get("stamp") == guard.train_stamp and len(guard.train_notes) == guard.train_listed:
        return stored, None
    return None, "the training tables need a current training selection (or the run's own tables file)"


def registered_order(prereg: Mapping[str, Any]) -> List[str]:
    """Every technique id in the pre-registration's order (rows, then grid points)."""
    return [grid_id(str(r["id"]), g) for r in prereg["techniques"] for g in r["grid"]]


def _reorder(result: Optional[Dict[str, Any]], order: Sequence[str]) -> Optional[Dict[str, Any]]:
    """The tables of ``measures.tables`` with every ``order`` list in the registered order (run files are written
    with sorted keys, so tables aggregated from them come back alphabetical)."""
    if result is None:
        return None
    pos = {tid: i for i, tid in enumerate(order)}

    def key(tid: str) -> Tuple[int, str]:
        return (pos.get(tid, len(pos)), tid)

    out = json.loads(json.dumps(result))
    if "order" in out:
        out["order"] = sorted(out["order"], key=key)
    blocks = list(out.get("tables", {}).values())
    blocks += [c for cfg in out.get("sensitivities", {}).values() for c in cfg.values()]
    for clock_tables in blocks:
        for t in clock_tables.values():
            if isinstance(t, dict) and "order" in t:
                t["order"] = sorted(t["order"], key=key)
    return out


#: CFFEX index futures: the daily price limit is this percentage of the previous settlement, prices move in ticks of
#: ``IM_TICK`` index points (exchange rules, not tolerances).
IM_PRICE_LIMIT_PCT = 10.0
IM_TICK = 0.2


def _within_tick(price: float, level: float) -> bool:
    gap = abs(float(price) - float(level))
    return gap < IM_TICK or is_close(gap, IM_TICK)


def limit_days(boards: Mapping[date, Any], chains: Sequence[Any]) -> List[Dict[str, Any]]:
    """The days on which the far IM contract (the hedge contract, the chain's last IM quote) closed at its daily price
    limit: the close within one tick of the previous settlement x (1 -/+ the limit). ``locked``: the settlement sits
    at the limit too. ``vwap_vs_close``: the day's VWAP over the close minus 1 (what the twap clock fills at);
    ``next_open_vs_close``: the next day's open of the same contract over the close minus 1. Board marks only."""
    days = [c.day for c in chains]
    out: List[Dict[str, Any]] = []
    for i in range(1, len(chains)):
        code_ = chains[i].im[-1].code
        bar = boards.get(days[i], None)
        prev = boards.get(days[i - 1], None)
        bar = None if bar is None else bar.im.get(code_)
        prev = None if prev is None else prev.im.get(code_)
        if bar is None or prev is None or bar.close is None:
            continue
        for side, sign in (("down", -1.0), ("up", 1.0)):
            limit = float(prev.settle) * (1.0 + sign * IM_PRICE_LIMIT_PCT / 100.0)
            if not _within_tick(bar.close, limit):
                continue
            nxt = boards.get(days[i + 1]).im.get(code_) if i + 1 < len(days) and days[i + 1] in boards else None
            out.append({"date": days[i].isoformat(), "code": code_, "side": side,
                        "locked": _within_tick(bar.settle, limit), "close": float(bar.close),
                        "settle": float(bar.settle), "prev_settle": float(prev.settle), "limit": limit,
                        "vwap_vs_close": None if bar.vwap is None else float(bar.vwap) / float(bar.close) - 1.0,
                        "next_open_vs_close": (None if nxt is None or nxt.open is None
                                               else float(nxt.open) / float(bar.close) - 1.0)})
    return out


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(sum(values) / len(values)) if values else None


def event_stats(notes: Sequence[Mapping[str, Any]], *, cfg: str, clock: str, kind: str, book: str, slice_: str,
                ids: Sequence[str], alpha: float, limit_dates: Sequence[str]) -> Dict[str, Any]:
    """What the review asked to see behind an ES10 (one barrier type, book and slice, from the run files):

    ``pnl`` (``event_pnl``); ``hits`` (event id -> the KO fired / the approach ended in a knock-in); ``tail``: the
    baseline's ES10 tail (the ``tail_count(n, alpha)`` worst events, ties by event id) grouped by event date, with how
    many of them are hits, largest group first; ``outcomes``: per technique the count and mean event P&L of the hits
    and of the others; ``touched``: per limit day, how many events' windows (the baseline's rows) include it."""
    pnl = event_pnl(notes, cfg=cfg, clock=clock, kind=kind, book=book, slice_=slice_, ids=ids)
    owner = {eid: n for n in notes for eid in n.get("events", {})}
    base = pnl.get("base", {})
    hits = {eid: bool(owner[eid]["events"][eid].get("hit")) for eid in base}
    tail: List[Dict[str, Any]] = []
    if base:
        k = measures.tail_count(len(base), alpha)
        worst = sorted(base, key=lambda e: (base[e], e))[:k]
        groups: Dict[str, List[str]] = {}
        for eid in worst:
            groups.setdefault(owner[eid]["events"][eid]["date"], []).append(eid)
        tail = sorted(({"date": d, "events": len(es), "hits": sum(hits[e] for e in es)} for d, es in groups.items()),
                      key=lambda r: (-r["events"], r["date"]))
    outcomes = {tid: {"hit": {"n": sum(1 for e in v if hits.get(e)), "mean": _mean([x for e, x in v.items() if hits.get(e)])},
                      "miss": {"n": sum(1 for e in v if not hits.get(e)),
                               "mean": _mean([x for e, x in v.items() if not hits.get(e)])}}
                for tid, v in pnl.items()}
    windows = {eid: {d for d, _, _ in _cells(owner[eid], cfg, clock)["base"]["events"][eid]["days"]} for eid in base}
    touched = {d: sum(1 for w in windows.values() if d in w) for d in limit_dates}
    return {"pnl": pnl, "hits": hits, "tail": tail, "outcomes": outcomes, "touched": touched}


def rows_after_end(notes: Sequence[Mapping[str, Any]], *, cfg: str, clock: str,
                   calendar: Sequence[date]) -> Dict[str, Dict[str, int]]:
    """Per technique, the notes whose run continues after the note's last record (option legs left on the book are
    still hedged until they are gone) and the number of such rows (trading days of ``calendar`` after the last record,
    up to the run's last row). Techniques without such a row are left out."""
    cal = [d.isoformat() for d in calendar]
    out: Dict[str, Dict[str, int]] = {}
    for note in notes:
        end = note.get("last_record")
        for tid, cell in _cells(note, cfg, clock).items():
            last = cell.get("life", {}).get("last")
            if end is None or last is None or last <= end:
                continue
            rows = sum(1 for d in cal if end < d <= last)
            if rows:
                block = out.setdefault(tid, {"notes": 0, "rows": 0})
                block["notes"] += 1
                block["rows"] += rows
    return out


def build_model(paths: ReportPaths) -> Dict[str, Any]:
    """Everything the page prints, from the files ``paths`` names; never raises on a missing input."""
    prereg = load_prereg()
    sample = _load(paths.events)
    guard = decide(paths, sample)
    split = date.fromisoformat(str(prereg["split_date"]))
    primary = measures.primary_config(prereg).key
    clock = str(prereg["clocks"]["primary"])
    books = {"ko": str(prereg["verdict"]["ko_book"]), "ki": str(prereg["verdict"]["ki_book"])}
    train_tables, train_problem = _training_tables(paths, guard, prereg, sample)
    alpha = float(prereg["statistics"]["es_alpha"])

    # board marks (any date: no P&L is attached to them here) for the limit days, and the trading calendar
    limits, limits_problem, calendar = None, None, None
    try:
        chains = load_chain(paths.chain)
        calendar = [c.day for c in chains]
        limits = limit_days(load_boards(paths.board), chains)
    except (QuantArkException, OSError) as err:
        limits_problem = f"the board or the chain could not be read: {err}"
    limit_dates = [r["date"] for r in limits or []]
    controls = {"ko": ("base_noclip", "ko_wrong"), "ki": ("base_noclip", "ki_wrong")}

    def events_block(notes: Sequence[Mapping[str, Any]], slice_: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {"slice": slice_, "dist": {}, "path": None, "stats": {}, "after_end": None}
        if not notes:
            return out
        available = list(_cells(notes[0], primary, clock))
        for kind in ("ko", "ki"):
            ids = _shown_ids(guard.selection, kind, available)
            out["dist"][kind] = event_pnl(notes, cfg=primary, clock=clock, kind=kind, book=books[kind], slice_=slice_,
                                          ids=ids)
            more = ids + [t for t in controls[kind] if t in available and t not in ids]
            out["stats"][kind] = event_stats(notes, cfg=primary, clock=clock, kind=kind, book=books[kind],
                                             slice_=slice_, ids=more, alpha=alpha, limit_dates=limit_dates)
        ids = [t for t in ("base", "sop_futures", "sop_overlay") if t in available]
        out["path"] = _path(notes, cfg=primary, clock=clock, kind="ko", book=books["ko"], slice_=slice_, ids=ids)
        if calendar is not None:
            out["after_end"] = rows_after_end(notes, cfg=primary, clock=clock, calendar=calendar)
        return out

    states, states_problem = None, None
    if guard.state_files:
        try:
            states = summarise_states(guard.state_files, split=split, cutoff=None if guard.test_ok else split)
        except (QuantArkException, KeyError, ValueError) as err:
            states_problem = f"the recorded states could not be read: {err}"
    else:
        states_problem = f"no recorded state file of the sample in {paths.states_dir}"
    probe = _load(paths.probe)
    return {
        "view": "seller",
        "prereg": {k: prereg[k] for k in ("title", "books", "note", "windows", "split_date", "clocks", "clock_notes",
                                          "im_clip", "tradability", "statistics", "verdict", "hypotheses",
                                          "sample", "techniques", "selection")},
        "primary": {"config": primary, "clock": clock}, "books": books,
        "data": (sample or {}).get("data"),
        "sample": None if sample is None else sample_sizes(sample),
        "guard": guard.as_dict(),
        "selection": guard.selection,
        "train_tables": _reorder(train_tables, registered_order(prereg)), "train_problem": train_problem,
        "test_tables": _reorder(guard.payload, registered_order(prereg)) if guard.test_ok else None,
        "train_events": events_block(guard.train_notes, "train") if guard.training_ok else None,
        "test_events": events_block(guard.test_notes, "test") if guard.test_ok and guard.test_notes else None,
        "states": states, "states_problem": states_problem,
        "probe": probe,
        "limit_days": limits, "limits_problem": limits_problem, "alpha": alpha,
        "paths": {k: str(getattr(paths, k)) for k in ("runs_dir", "selection", "backtest", "states_dir", "events",
                                                      "probe")},
    }


# ---------------------------------------------------------------------------
# Rendering: formatting and drawing helpers
# ---------------------------------------------------------------------------

num, esc, signed = h.num, h.esc, h.signed


def pct(v: Optional[float], d: int = 1) -> str:
    return "–" if v is None else num(100.0 * v, d)


def flag(v: Optional[bool]) -> str:
    if v is None:
        return '<span class="dim">n/a</span>'
    return '<b class="hit-yes">pass</b>' if v else '<span class="dim">fail</span>'


def code(s: str) -> str:
    return f"<code>{esc(s)}</code>"


def tbl(name: str, headers: Sequence[str], rows: Sequence[Sequence[str]], caption: str, note: str = "",
        row_cls: Optional[Sequence[str]] = None, text: bool = False) -> str:
    """A table with its one-line caption (what it shows, how to read it) and an optional foot note."""
    head = "".join(f"<th>{esc(x)}</th>" for x in headers)
    body = ""
    for i, r in enumerate(rows):
        cls = f' class="{row_cls[i]}"' if row_cls and row_cls[i] else ""
        body += f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    tcls = ' class="text"' if text else ""
    foot = f'<p class="note">{note}</p>' if note else ""
    return (f'<p class="tcap">{esc(caption)}</p><div class="scroll"><table data-table="{name}"{tcls}><thead><tr>'
            f"{head}</tr></thead><tbody>{body}</tbody></table></div>{foot}")


def unavailable(text: str) -> str:
    return f'<p class="note missing">NOT AVAILABLE · {esc(text)}</p>'


def chart(name: str, svg: str, title: str, caption: str) -> str:
    return f'<div data-chart="{name}">{h.figure(svg, title, caption)}</div>'


def _ticks(lo: float, hi: float, n: int = 5) -> List[float]:
    span = hi - lo
    raw = span / max(n, 1) if span > 0 else 1.0
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    out, t = [], math.floor(lo / step) * step
    while t < hi or is_close(t, hi, rel_tol=Tolerance.RELATIVE, abs_tol=step * Tolerance.RELATIVE):
        out.append(round(t, 10))
        t += step
    return out


def strip(rows: Sequence[Mapping[str, Any]], *, label: str, xlabel: str, alpha: float, width: int = 760) -> str:
    """A linear strip chart: one row per technique, one dot per event, an amber tick at its ES10."""
    left, right, top, bottom, step = 170, 24, 14, 42, 24
    vals = [v for r in rows for v in r["values"]]
    lo, hi = min(vals + [0.0]), max(vals + [0.0])
    pad = (hi - lo) * 0.05 or 1.0
    ticks = _ticks(lo - pad, hi + pad)
    xlo, xhi = min(ticks[0], lo - pad), max(ticks[-1], hi + pad)
    height = top + bottom + step * max(len(rows), 1)
    base = height - bottom

    def X(v: float) -> float:
        return left + (v - xlo) / (xhi - xlo) * (width - left - right)

    body: List[str] = []
    for t in ticks:
        body.append(f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top}" y2="{base}"/>'
                    f'<text class="tick" x="{X(t):.1f}" y="{base + 16}" text-anchor="middle">{num(t, 0)}</text>')
    body.append(f'<line class="zero" x1="{X(0.0):.1f}" x2="{X(0.0):.1f}" y1="{top}" y2="{base}"/>')
    body.append(f'<line class="axis" x1="{left}" x2="{width - right}" y1="{base}" y2="{base}"/>')
    body.append(f'<text class="axlab" x="{left}" y="{height - 4}">{esc(xlabel)}</text>')
    for i, r in enumerate(rows):
        y = top + step * (i + 0.5)
        body.append(f'<text class="dlab" x="{left - 10}" y="{y + 3.5:.1f}" text-anchor="end">{esc(r["name"])}</text>')
        for v, tip in zip(r["values"], r["tips"]):
            body.append(f'<circle class="mk" cx="{X(v):.1f}" cy="{y:.1f}" r="3.5" style="fill:{r["colour"]}"/>'
                        f'<circle class="hit" cx="{X(v):.1f}" cy="{y:.1f}" r="8" data-tip="{esc(tip)}"/>')
        if r.get("es") is not None:
            tip = f'{r["name"]} · ES10 {num(r["es"], 1)} bp'
            body.append(f'<line class="esmark" x1="{X(r["es"]):.1f}" x2="{X(r["es"]):.1f}" y1="{y - 9:.1f}" '
                        f'y2="{y + 9:.1f}"/><circle class="hit" cx="{X(r["es"]):.1f}" cy="{y:.1f}" r="9" '
                        f'data-tip="{esc(tip)}"/>')
    legend = ('<div class="legend"><span><i style="border-top:3px solid var(--s1)"></i>baseline (one dot per '
              'event)</span><span><i style="border-top:3px solid var(--s2)"></i>technique</span><span><i '
              'style="border-top:3px solid var(--amber)"></i>ES10 (mean of the worst '
              f'{num(100.0 * alpha, 0)}%)</span></div>')
    return legend + (f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label)}">'
                     + "".join(body) + "</svg>")


def _series(name: str, xs: Sequence[float], ys: Sequence[Optional[float]], colour: str,
            dash: Optional[str] = None) -> Dict[str, Any]:
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None and is_valid_number(y)]
    return {"name": name, "xs": [p[0] for p in pts], "ys": [p[1] for p in pts], "colour": colour, "dash": dash}


# ---------------------------------------------------------------------------
# Rendering: sections
# ---------------------------------------------------------------------------

ROLE_WORDS = {"baseline": "baseline", "control": "control", "cost": "cost lever"}


def _role(row: Mapping[str, Any]) -> str:
    if row.get("eligible"):
        return '<b class="hit-yes">eligible</b>' if row.get("role") != "composite" else '<b class="hit-yes">composite</b>'
    if row.get("role") == "candidate":
        return '<span class="dim">grid point</span>'
    return f'<span class="dim">{esc(ROLE_WORDS.get(str(row.get("role")), str(row.get("role"))))}</span>'


def _p(v: Optional[float]) -> str:
    """A p value at three decimals; below one in a thousand (a share of replicates of zero included) it reads
    "<0.001", so "0.000" never stands beside it."""
    if v is None:
        return "–"
    return "&lt;0.001" if v < 1.0e-3 and not is_close(v, 1.0e-3) else num(v, 3)


def _verdict_tables(t: Mapping[str, Any], *, name: str, test: bool, what: str, clock: str,
                    prereg: Mapping[str, Any]) -> str:
    """The tail table and the whole-life table of one barrier type (test: with the pass flags; training: measures
    only)."""
    lead = (f"Test slice, {clock} clock: {what}. " if test else
            f"Training figures, {clock} clock: {what}, the events the parameters were picked on; not a verdict. ")
    rows, cls, life = [], [], []
    for tid in t["order"]:
        r = t["rows"][tid]
        d_es = None if r.get("es10") is None or r.get("es10_base") is None else r["es10"] - r["es10_base"]
        cells = [code(tid), _role(r), num(r.get("es10"), 1), num(r.get("es10_base"), 1), signed(d_es, 1),
                 num(r.get("worst"), 1), num(r.get("worst_base"), 1), num(r.get("mean"), 1), _p(r.get("p")),
                 _p(r.get("p_holm"))]
        cells += [str(int(r.get("clips") or 0)), num(r.get("mean_cost_bp"), 2)]
        if test:
            cells.append(flag(r.get("tail_pass")))
        rows.append(cells)
        cls.append("vrow" if r.get("eligible") else "")
        lc = [code(tid), num(r.get("std_ratio"), 3), signed(r.get("terminal_diff_bp"), 1)]
        if test:
            lc += [flag(r.get("life_pass")), flag(r.get("win")) if r.get("eligible") else '<span class="dim">–</span>',
                   flag(r.get("robust")) if r.get("eligible") else '<span class="dim">–</span>']
        life.append(lc)
    head = ["Technique", "Role", "ES10", "Base ES10", "ES10 − base", "Worst", "Base worst", "Mean", "p", "Holm p",
            "Clip days", "Cost"]
    if test:
        head.append("Tail test")
    lhead = ["Technique", "Std ratio", "Terminal − base"]
    if test:
        lhead += ["Life test", "Win", "Robust"]
    st, v = prereg["statistics"], prereg["verdict"]
    alpha, pmax = float(st["es_alpha"]), float(st["p_max"])
    frac = Fraction(str(alpha))
    n = f"{t['n_events']} events in {t['n_weeks']} distinct weeks"
    deci = ""
    if test:
        deci = (f" Decidable (at least {v['min_events']} events in at least {v['min_weeks']} weeks)."
                if t.get("decidable") else
                f" Not decidable: fewer than {v['min_events']} events or fewer than {v['min_weeks']} weeks, so no row "
                "can win.")
    return (tbl(f"{name}-tail", head, rows, lead + f"{n}; event-window P&L of the seller, bp of notional, net of costs."
                + deci,
                note=f"ES10 = mean of the worst {num(100.0 * alpha, 0)}% of events (k = max(1, ceil(n x "
                     f"{frac.numerator}/{frac.denominator}))); higher is better. p = share of {int(st['bootstrap_n']):,} "
                     "week-clustered bootstrap replicates with ES10(technique) − ES10(base) ≤ 0; Holm p adjusts it "
                     "across the eligible rows. Only rows marked eligible or composite can win; baseline, controls, "
                     "cost levers and the grid points not picked are reported only. Clip days = days inside the event "
                     "windows on which the hedge-ratio cap bound; cost = mean event-window trading cost, bp of "
                     f"notional. Tail test = ES10 higher, worst event not lower and raw p &lt; {num(pmax, 2)}.",
                row_cls=cls)
            + tbl(f"{name}-life", lhead, life,
                  ("Whole life, test runs, all dates, " if test else "Training figures, whole life cut at the split, ")
                  + f"{clock} clock: daily hedged P&L of the unselected notes of the same book "
                  f"({t.get('whole_life', {}).get('n_notes', '–')} notes), day 0 left out.",
                  note=f"Std ratio = daily std of the technique / the baseline's (pass at ≤ {num(v['std_ratio_max'], 2)})."
                       " Terminal − base = mean terminal P&L of completed notes minus the baseline's, bp of notional "
                       f"(pass at ≥ −{num(v['terminal_slack_bp'], 0)} bp)." + (" Win = tail test and life test; robust = a winner that "
                                                              "also passes the tail test under twap and next_open."
                                                              if test else "")))


def _tables_source(model: Mapping[str, Any]) -> Tuple[Optional[Mapping[str, Any]], bool]:
    if model["test_tables"] is not None:
        return model["test_tables"], True
    return model["train_tables"], False


def _no_test_line(model: Mapping[str, Any]) -> str:
    if model["guard"]["test_ok"]:
        return ""
    return ('<p class="note missing">No test number is shown here (see the banner at the top). The rows below are '
            'training figures.</p>')


def _false_winners(t: Optional[Mapping[str, Any]], kind: str, prereg: Mapping[str, Any]) -> str:
    """The multiple-testing statement, with the eligible count of the table shown."""
    pmax = float(prereg["statistics"]["p_max"])
    if not t or not t.get("n_eligible"):
        return ""
    n = int(t["n_eligible"])
    return (f" {n} eligible {kind.upper()} rows are tested at raw p &lt; {num(pmax, 2)}, so about {num(n * pmax, 1)} "
            f"false winners are expected by chance across the {n} eligible {kind.upper()} rows (about one false winner "
            f"per {num(1.0 / pmax, 0)} rows); Holm-adjusted p is printed beside the raw p, and adjacent weeks share "
            "market days, so p is optimistic.")


def _events_shown(model: Mapping[str, Any]) -> Tuple[Optional[Mapping[str, Any]], str]:
    """The run-file block of the slice shown: test when the guard passed and test runs exist, else training."""
    if model["test_events"] is not None:
        return model["test_events"], "Test slice"
    if model["train_events"] is not None and not model["guard"]["test_ok"]:
        return model["train_events"], "Training figures (events dated before the split)"
    return None, ""


def _concentrated(tail: Sequence[Mapping[str, Any]]) -> bool:
    """A tail is concentrated when one event date holds at least a third of it (and more than one event)."""
    if not tail:
        return False
    k = sum(r["events"] for r in tail)
    return tail[0]["events"] > 1 and 3 * tail[0]["events"] >= k


def _tail_dates(model: Mapping[str, Any], kind: str) -> str:
    """The baseline's ES tail grouped by event date, from the run files of the slice shown."""
    block, label = _events_shown(model)
    stats = None if block is None else block["stats"].get(kind)
    if not stats or not stats["tail"]:
        return unavailable(f"no run file of the slice shown holds a {kind.upper()} event")
    limit = {r["date"] for r in model["limit_days"] or []}
    word, hit = ("KO pin events", "fired") if kind == "ko" else ("KI approaches", "knocked in")
    k = sum(r["events"] for r in stats["tail"])
    top = stats["tail"][0]
    rows = [[esc(r["date"]), str(r["events"]), str(r["hits"]), "yes" if r["date"] in limit else ""]
            for r in stats["tail"]]
    if _concentrated(stats["tail"]):
        lead = (f"<p>The baseline's tail is concentrated in single market days: {top['events']} of the {k} {word} in "
                f"its ES10 tail are dated {esc(top['date'])}" + (f", {top['hits']} of them {hit}" if top["hits"] else "")
                + ". Events of one day are one market move, not independent draws.</p>")
    else:
        lead = (f"<p>The {k} {word} in the baseline's ES10 tail are spread over {len(stats['tail'])} event dates; no "
                f"date holds more than {top['events']} of them.</p>")
    return lead + tbl(f"{kind}-tail-dates", ["Event date", "Events in the tail", hit.capitalize(),
                                             "Far IM at its limit"], rows,
                      f"{label}, close clock: the {k} {word} in the baseline's ES10 tail (its worst "
                      f"{num(100.0 * model['alpha'], 0)}%), by event date.")


def _outcomes(model: Mapping[str, Any], t: Optional[Mapping[str, Any]]) -> str:
    """Mean KI event P&L split by how the approach ended, for the baseline, the eligible rows and the controls."""
    block, label = _events_shown(model)
    stats = None if block is None else block["stats"].get("ki")
    if not stats or not stats["outcomes"]:
        return unavailable("no run file of the slice shown holds a KI approach")
    eligible = [tid for tid in (t or {}).get("order", []) if t["rows"][tid].get("eligible")]
    ids = ["base"] + [x for x in ("base_noclip", "ki_wrong") if x in stats["outcomes"]] + eligible
    rows = []
    for tid in dict.fromkeys(ids):
        o = stats["outcomes"].get(tid)
        if o is None:
            continue
        d = None if o["hit"]["mean"] is None or o["miss"]["mean"] is None else o["hit"]["mean"] - o["miss"]["mean"]
        rows.append([code(tid), str(o["hit"]["n"]), num(o["hit"]["mean"], 1), str(o["miss"]["n"]),
                     num(o["miss"]["mean"], 1), signed(d, 1)])
    base = stats["outcomes"]["base"]
    lead = ("<p>Rules that carry less long futures into the KI barrier (<code>ki_wrong</code>, the raised-barrier "
            "control; <code>ki_cap</code>; <code>secant</code> at a wide bump) gain when the approach ends in a "
            "knock-in and pay when the index rebounds, so their mean P&amp;L is partly a directional outcome of how "
            f"many approaches knocked in ({base['hit']['n']} knock-ins and {base['miss']['n']} rebounds here). The "
            "whole-life test catches this only through the std criterion, because the terminal criterion is "
            "one-sided (it only asks that a technique not lose more than the slack).</p>")
    out = lead + tbl("ki-outcome", ["Technique", "Knocked in", "Mean (bp)", "Did not", "Mean (bp)", "Difference"],
                     rows, f"{label}, close clock: mean event-window P&L of the KI approaches, split by whether the "
                           "approach ended in a knock-in, bp of notional.")
    rows_t = (t or {}).get("rows", {})
    if "ki_wrong" in rows_t and rows_t["ki_wrong"].get("es10") is not None:
        w = rows_t["ki_wrong"]
        better = w["es10"] > w["es10_base"] and not is_close(w["es10"], w["es10_base"])
        where = "on the test slice" if model["test_tables"] is not None else "on the training slice"
        out += ("<p>Lowering the booked KI barrier (<code>ki_shift</code>) is the conservative direction for reserving "
                "(it shows a reserve, "
                "section 09) but the wrong direction for hedging a gap down: the lower barrier keeps more long "
                "futures into the knock-in, which the seller then sells after the gap. Hypothesis H5 (the wrong-way "
                "controls are worse than the baseline) "
                + (f"is not supported for <code>ki_wrong</code> {where}: its ES10 is {num(w['es10'], 1)} bp against "
                   f"{num(w['es10_base'], 1)} bp for the baseline, because the raised barrier carries less long "
                   "futures into the knock-ins that make the tail." if better else
                   f"holds for <code>ki_wrong</code> {where}: its ES10 is {num(w['es10'], 1)} bp against "
                   f"{num(w['es10_base'], 1)} bp for the baseline.") + "</p>")
    return out


def _limit_block(model: Mapping[str, Any]) -> str:
    """The days the far IM contract closed at its daily price limit, what each clock does there, and the events of the
    slice shown whose windows hold them."""
    days = model["limit_days"]
    if days is None:
        return unavailable(model["limits_problem"] or "the board and the chain")
    if not days:
        return ("<p>There is no day on which the far IM contract closed at its daily price limit in the data, so every "
                "close-clock fill was at a traded price.</p>")
    block, label = _events_shown(model)
    split = model["prereg"]["split_date"]
    test_rows = model["test_events"] is not None

    def touched(kind: str, d: str) -> str:
        if block is None or (not test_rows and d >= split):
            return "–"
        return str(block["stats"].get(kind, {}).get("touched", {}).get(d, 0))

    rows = [[esc(r["date"]), code(r["code"]), r["side"], "locked" if r["locked"] else "close only",
             num(r["close"], 1), num(r["limit"], 1), "–" if r["vwap_vs_close"] is None else signed(100 * r["vwap_vs_close"], 2),
             "–" if r["next_open_vs_close"] is None else signed(100 * r["next_open_vs_close"], 2),
             touched("ko", r["date"]), touched("ki", r["date"])] for r in days]
    parts = ["<p>On a day the far IM contract closed at its daily price limit, a sale (at a limit down) or a purchase "
             "(at a limit up) at the close was not executable. The close clock books it anyway; the twap clock fills "
             "the close-state size at the day's VWAP, which no desk could have known at the time. The twap "
             "improvement of the KI tail on such days is therefore not an achievable fill."]
    for r in days:
        if r["locked"] and r["vwap_vs_close"] is not None:
            parts.append(f" On {esc(r['date'])} ({r['side']}, locked) the VWAP was {num(100 * abs(r['vwap_vs_close']), 1)}% "
                         f"{'above' if r['vwap_vs_close'] > 0 else 'below'} the close"
                         + ("" if r["next_open_vs_close"] is None else
                            f" and the next open was {num(100 * abs(r['next_open_vs_close']), 1)}% from the limit close")
                         + ".")
    small = [r for r in days if r["locked"] and r["next_open_vs_close"] is not None
             and abs(r["next_open_vs_close"]) < 1.0e-2]
    if small:
        parts.append(" Where the next open was within 1% of the limit close the close figure is about right, by luck: "
                     + ", ".join(esc(r["date"]) for r in small) + ".")
    parts.append("</p>")
    return "".join(parts) + tbl(
        "limit-days", ["Date", "Contract", "Side", "Locked", "Close", "Limit price", "VWAP vs close (%)",
                       "Next open vs close (%)", "KO events touched", "KI events touched"], rows,
        f"Board marks, every day of the data: the far IM contract at its daily price limit (±"
        f"{num(IM_PRICE_LIMIT_PCT, 0)}% of the previous settlement, within one {num(IM_TICK, 1)}-point tick); events "
        f"touched = events of the slice shown ({label or 'none'}) whose window holds the day.",
        note="Locked = the settlement sits at the limit too. A dash: the day lies outside the slice shown.")


def section_design(model: Mapping[str, Any]) -> str:
    p = model["prereg"]
    v, tr, w = p["verdict"], p["tradability"], p["windows"]
    parts = [
        '<p class="lead">The desk sold a 2-year snowball on the CSI 1000 index (000852.SH) at price 0 and hedges it. '
        "Everything on this page is the seller's: P&amp;L, position delta and gamma, and cash are the seller's, in bp "
        "of notional (10 mio CNY per note) unless stated. The primary hedge is the delta in one IM future (the "
        "longest listed contract with at least 7 days left, the central 1% bump delta of the true note); the "
        "secondary layer is MO options near a barrier. Prices and greeks come from the QUAD V2 engine with one flat "
        "volatility per day (the day's far-MO at-the-money vol).</p>",
        h.points([
            "<b>The note.</b> Strike = KO reference = booking-day spot; knock-out at 100% of strike on the monthly "
            "anniversaries from month 3 to month 24, paying the coupon; knock-in at 75% of strike observed on every "
            "daily close; principal excluded. The coupon is solved so the note is worth 0 under hold-forward carry.",
            f"<b>Two books.</b> <code>{esc(model['books']['ko'])}</code> is the 2Y product at real inception dates "
            "(the quarterly headline set plus notes struck so that real KO observations land near the barrier); KO "
            "pin events are measured on it. <code>seas6</code> is the same 75% product in its last 6 months (18 "
            "months elapsed, coupon accrued over the full 24, not knocked in), booked on the first trading day of each "
            "ISO week with its KI barrier 7% below that day's spot; KI approaches are measured on it, because the "
            "knock-in pin is a property of a short remaining life (section 04).",
            f"<b>Events.</b> A KO pin event is a KO observation with the prior close within "
            f"{num(100 * p['sample']['ko_prior_gap'], 0)}% of the barrier. A KI approach is a run of trading days with "
            f"spot within {num(100 * w['ki_band'], 0)}% above the KI barrier that comes within "
            f"{num(100 * p['sample']['ki_near'], 0)}% or ends in a knock-in. Train / test split by event date at "
            f"{esc(p['split_date'])}: parameters were picked on training events, the verdict is read on test events "
            "only.",
            "<b>Windows.</b> KO: the daily rows from the first trading day on which the observation is at most "
            f"{w['ko_days']} trading days ahead through the trading day after it. KI: from the approach's first day "
            "through the trading day after its last day. Each window carries its entry and exit trading costs.",
            f"<b>Costs and limits.</b> IM costs {num(1e4 * tr['im_cost_rate'], 0)} bp of traded notional; MO costs "
            f"{num(tr['mo_cost_pt'], 1)} index points per leg per side. The IM hedge ratio (IM notional / note "
            f"notional) is clipped to [{num(p['im_clip'][0], 2)}, {num(p['im_clip'][1], 1)}]. MO legs are opened only "
            f"with day volume ≥ {tr['min_volume']} and open interest ≥ {tr['min_open_interest']} lots and are checked "
            f"at a desk of {num(tr['desk_notional'] / 1e6, 0)} mio CNY: a trade is cut to "
            f"{num(100 * tr['max_volume_share'], 0)}% of the day's volume and a series position to "
            f"{tr['series_limit_lots']:,} lots per side.",
        ]),
    ]
    if model["sample"]:
        rows = [[code(r["book"]), "KO pin events" if r["kind"] == "ko" else "KI approaches",
                 "training (before split)" if r["slice"] == "train" else "test (on or after split)",
                 str(r["events"]), str(r["weeks"]), str(r["hits"])] for r in model["sample"]]
        parts.append(tbl("design-sample", ["Book", "Barrier type", "Slice", "Events", "Distinct weeks", "Fired / knocked in"],
                         rows, "Sample sizes per book, barrier type and slice: every event of every priced note "
                               "(results/events.json); weeks are ISO weeks of the event date, the bootstrap's clusters.",
                         note="Fired = KO events whose observation redeemed the note; knocked in = KI approaches ending "
                              f"in a knock-in. The verdict needs at least {v['min_events']} test events in at least "
                              f"{v['min_weeks']} weeks."))
    else:
        parts.append(unavailable("results/events.json: the sample sizes"))
    notes = p["clock_notes"]
    parts.append(h.sub("Three execution clocks"))
    parts.append(h.points([
        f"<b>close (decides the verdict).</b> {esc(notes['close'])}. It is the pre-registered primary clock; it lets "
        "the seller trade at the fixing price itself, so it understates the exposure of a true pin.",
        f"<b>twap (a VWAP proxy).</b> {esc(notes['twap'])}. Option legs fill at settlement under this clock too.",
        f"<b>next_open (stress).</b> {esc(notes['next_open'])}. It overstates the exposure of a pin; a fixing on the "
        "last data day has no next open and its event is left out of the next_open tables (counted).",
    ]))
    parts.append(h.sub("The pre-registered verdict rule"))
    parts.append(h.points([
        f"<b>(a) Tail</b>, on test events under close: ES10 of the event-window P&amp;L higher than the baseline's, "
        f"worst event not lower, and week-clustered bootstrap p &lt; {num(p['statistics']['p_max'], 2)} "
        f"({p['statistics']['bootstrap_n']:,} replicates, seed {p['statistics']['seed']}).",
        f"<b>(b) Whole life</b>, on the unselected notes of the same book (KO: the {esc(v['ko_book'])} headline set; "
        f"KI: every {esc(v['ki_book'])} booking), day 0 left out: daily std at most {num(v['std_ratio_max'], 2)} x "
        f"the baseline's and mean terminal P&amp;L at least the baseline's minus {num(v['terminal_slack_bp'], 0)} bp.",
        f"<b>A technique wins</b> with (a) and (b); decidable only with at least {v['min_events']} test events in at least "
        f"{v['min_weeks']} distinct weeks; <b>robust</b> = a winner that also passes (a) under twap and next_open. "
        "Raw p decides; Holm-adjusted p is printed beside it.",
    ]))
    parts.append(tbl("hypotheses", ["Id", "Hypothesis (pre-registered)"],
                     [[esc(x["id"]), esc(x["text"])] for x in p["hypotheses"]],
                     "The hypotheses written down before any ledger run; the conclusions are read against them.",
                     text=True))
    return h.section(1, "design", "TESTED", "What was tested and how", "".join(parts))


def section_ko(model: Mapping[str, Any]) -> str:
    src, test = _tables_source(model)
    clock = model["primary"]["clock"]
    parts = ['<p class="lead">Each technique against the baseline (<code>base</code>: hedge the true note\'s 1% '
             "bump delta with IM at every close) on the window P&amp;L of each KO pin event of the "
             f"<code>{esc(model['books']['ko'])}</code> book."
             + _false_winners(None if src is None else src["tables"][clock]["ko"], "ko", model["prereg"])
             + " Controls "
             "(<code>ko_wrong</code>, the wrong-way shift; <code>base_noclip</code>), the <code>band</code> cost rows "
             "and the grid points that were not picked are shown and never eligible.</p>", _no_test_line(model)]
    if src is None:
        parts.append(unavailable("no current test payload and no current training run: " + (model["train_problem"] or "")))
    else:
        parts.append(_verdict_tables(src["tables"][clock]["ko"], name="ko", test=test, clock=clock,
                                     what=f"KO pin events of {model['books']['ko']}", prereg=model["prereg"]))
        parts.append(_tail_dates(model, "ko"))
    return h.section(2, "ko", "KO VERDICT", "KO pin: does a technique improve the tail?", "".join(parts))


def section_ki(model: Mapping[str, Any]) -> str:
    src, test = _tables_source(model)
    clock = model["primary"]["clock"]
    cap = model["prereg"]["im_clip"][1]
    parts = ['<p class="lead">The same comparison on the KI approaches of the seasoned book '
             f"<code>{esc(model['books']['ki'])}</code>. The pre-registered position limit (the hedge ratio is "
             f"capped at {num(cap, 1)}, i.e. IM notional at most {num(cap, 0)} times the note's) binds on the seasoned "
             "book near its barrier, so the KI baseline is a limit-constrained hedge. <code>base_noclip</code> is the "
             "same hedge without the cap, shown beside <code>base</code> with the number of days the cap bound inside "
             "the windows (clip days); it is a control and never eligible."
             + _false_winners(None if src is None else src["tables"][clock]["ki"], "ki", model["prereg"]) + "</p>",
             _no_test_line(model)]
    if src is None:
        parts.append(unavailable("no current test payload and no current training run: " + (model["train_problem"] or "")))
    else:
        t = src["tables"][clock]
        parts.append(_verdict_tables(t["ki"], name="ki", test=test, clock=clock,
                                     what=f"KI approaches of {model['books']['ki']}", prereg=model["prereg"]))
        rows = t["ki"]["rows"]
        if "base" in rows and "base_noclip" in rows:
            parts.append(f'<p class="note">Clip days inside the windows: base {int(rows["base"].get("clips") or 0)}, '
                         f'base_noclip {int(rows["base_noclip"].get("clips") or 0)}.</p>')
        parts.append(h.sub("Where the KI tail comes from"))
        parts.append(_tail_dates(model, "ki"))
        parts.append(h.sub("Days the hedge could not trade at the close"))
        parts.append(_limit_block(model))
        parts.append(h.sub("Knock-in or rebound: a directional part of the KI P&L"))
        parts.append(_outcomes(model, t["ki"]))
        tr = t.get("ki_transfer")
        parts.append(h.sub("Transfer check on the 2Y product's own approaches"))
        if tr:
            word = "test" if test else "training"
            parts.append(f"<p>The same rows on the KI approaches of the 2Y book ({esc(tr['book'])}): here "
                         f"{tr['n_events']} {word} approaches in {tr['n_weeks']} weeks. Those notes still have a long "
                         "life left, so the gap is small (section 04), and the approaches cluster in few market "
                         "episodes: the rows carry measures only, no p and no verdict.</p>")
            trows = [[code(tid), str(r.get("n_events")), str(r.get("n_weeks")), num(r.get("es10"), 1),
                      num(r.get("es10_base"), 1), num(r.get("worst"), 1), num(r.get("worst_base"), 1),
                      num(r.get("mean"), 1)] for tid, r in ((tid, tr["rows"][tid]) for tid in tr["order"])]
            parts.append(tbl("ki-transfer", ["Technique", "Events", "Weeks", "ES10", "Base ES10", "Worst",
                                             "Base worst", "Mean"], trows,
                             ("Test slice" if test else "Training figures") + f", {clock} clock: KI approaches of "
                             f"{tr['book']}, {tr['n_events']} events in {tr['n_weeks']} weeks; event-window P&L, bp of "
                             "notional, measures only."))
    return h.section(3, "ki", "KI VERDICT", "KI pin on the seasoned book", "".join(parts))


def section_kigap(model: Mapping[str, Any]) -> str:
    probe = model["probe"]
    if not probe:
        return h.section(4, "kigap", "KI GAP", "Why the KI book is seasoned",
                         unavailable("results/ki_gap_probe.json: the KI gap by remaining life and barrier"))
    inp = probe["inputs"]
    rows = []
    for r in sorted(probe["rows"], key=lambda x: (x["ki_pct"], x["months_left"])):
        rows.append([f"{num(100 * r['ki_pct'], 0)}%", str(r["months_left"]), num(r["gap_bp"], 1),
                     num(r["hedge_alive_pct"], 1), num(r["hedge_ki_pct"], 1), num(r["hedge_step_pct"], 1)])
    body = ('<p class="lead">The knock-in pin is the jump in the note\'s value and in its hedge when the KI flag '
            "flips. For a fresh note 3% above its KI barrier it is large only with a short life left: with a long life "
            "left a note near its barrier knocks in later anyway, so the flag is worth little, and a higher barrier "
            "makes that more certain. That is why the KI verdict is read on a 75% note in its last 6 months, not on a "
            "note with a lifted barrier.</p>"
            + tbl("ki-gap", ["KI barrier", "Months left", "Gap (bp)", "Hedge alive (%)", "Hedge knocked in (%)",
                             "Hedge step (%)"], rows,
                  f"Probe: a fresh note 3% above its KI barrier, coupon {num(100 * inp['coupon'], 0)}%, flat vol "
                  f"{num(100 * inp['vol'], 0)}%, q {num(100 * inp['q'], 0)}%, r {num(100 * inp['r'], 1)}%; gap = "
                  "not-knocked-in minus knocked-in value, bp of notional.",
                  note="Hedges are 1% bump deltas as % of notional (IM notional / note notional); hedge step = alive "
                       "minus knocked in, the futures the seller sells when the note knocks in."))
    return h.section(4, "kigap", "KI GAP", "Why the KI book is seasoned", body)


def section_clocks(model: Mapping[str, Any]) -> str:
    src, test = _tables_source(model)
    if src is None:
        return h.section(5, "clocks", "CLOCKS", "The three clocks", unavailable("no current test payload or training run"))
    clocks = [c for c in ("close", "twap", "next_open") if c in src["tables"]]
    rows = []
    for kind in ("ko", "ki"):
        first = src["tables"][clocks[0]][kind]
        for tid in first["order"]:
            r0 = first["rows"][tid]
            if not (r0.get("eligible") or tid == "base"):
                continue
            cells = [kind.upper(), code(tid)]
            for c in clocks:
                r = src["tables"][c][kind]["rows"].get(tid, {})
                d = None if r.get("es10") is None or r.get("es10_base") is None else r["es10"] - r["es10_base"]
                cells += [num(r.get("es10"), 1), signed(d, 1)]
            rows.append(cells)
    head = ["Barrier", "Technique"] + [x for c in clocks for x in (f"ES10 {c}", f"− base {c}")]
    left = src.get("extras", {}).get("left_out", {})
    n_events = {c: (src["tables"][c]["ko"]["n_events"], src["tables"][c]["ki"]["n_events"]) for c in clocks}
    lo = "; ".join(f"{c}: KO {left.get(c, {}).get('ko', 0)}, KI {left.get(c, {}).get('ki', 0)} left out "
                   f"({n_events[c][0]} KO / {n_events[c][1]} KI events used)" for c in clocks)
    body = ('<p class="lead">The verdict is read under close. twap is a VWAP proxy for a TWAP order (the target still '
            "comes from the close state, so its size is correlated with close minus VWAP); next_open is a stress that "
            "moves every fixing-day trade to the next open.</p>" + _no_test_line(model)
            + tbl("clocks", head, rows, ("Test slice" if test else "Training figures")
                  + ": ES10 of the event-window P&L under each clock and its difference from the baseline under the "
                    "same clock, bp of notional; the baseline and the eligible rows.",
                  note=f"Events left out under each clock (a fixing on the last data day has no next open): {lo}."))
    return h.section(5, "clocks", "CLOCKS", "The three clocks", body)


def _config_words(key: str) -> str:
    m = re.match(r"desk(?P<desk>[\d.]+)_cost(?P<cost>[\d.]+)$", key)
    return key if not m else f"desk {m['desk']} mio, MO cost {m['cost']} pt"


def section_sens(model: Mapping[str, Any]) -> str:
    title = "Cost and desk-size sensitivity"
    src = model["test_tables"]
    lead = ('<p class="lead">Close clock only, one number changed at a time against the primary configuration '
            f"({esc(_config_words(model['primary']['config']))}): the MO cost at 1.0 instead of 0.2 index points per "
            "leg per side, and the desk at 100 instead of 500 mio CNY (the MO volume and position screens bind less "
            "on a smaller desk). These runs exist on the test slice only.</p>")
    if src is None or not src.get("sensitivities"):
        return h.section(6, "sens", "SENSITIVITY", title,
                         lead + unavailable("the sensitivities are run with the test slice; no current test payload "
                                            "holds them"))
    clock = model["primary"]["clock"]
    configs = list(src["sensitivities"])
    rows = []
    for kind in ("ko", "ki"):
        prim = src["tables"][clock][kind]
        for tid in prim["order"]:
            if not (prim["rows"][tid].get("eligible") or tid == "base"):
                continue
            cells = [kind.upper(), code(tid), num(prim["rows"][tid].get("es10"), 1)]
            for cfg in configs:
                r = src["sensitivities"][cfg].get(clock, {}).get(kind, {}).get("rows", {}).get(tid, {})
                cells.append(num(r.get("es10"), 1))
            rows.append(cells)
    body = lead + tbl("sensitivity", ["Barrier", "Technique", "ES10 primary"] + [f"ES10 {_config_words(c)}"
                                                                               for c in configs], rows,
                      "Test slice, close clock only: ES10 of the event-window P&L, bp of notional, under the primary "
                      "configuration and each sensitivity (one number changed at a time).")
    return h.section(6, "sens", "SENSITIVITY", title, body)


def _ladder_block(ill: Mapping[str, Any], kind: str, comp_word: str) -> str:
    xs = ill["x_pct"]
    comp = ill["models"][1]
    rel = "KO level" if kind == "ko" else "KI level"
    caption = (f"{ill['note']}, close of {ill['date']}, spot {num(ill['spot'], 1)} ({h.snum(ill['distance_pct'], 3)}% "
               f"from the {rel}); training-dated")
    out = []
    for what, key, unit in (("delta", "position_delta_pct", "POSITION DELTA, % OF NOTIONAL"),
                            ("gamma", "position_gamma_pct", "POSITION GAMMA, % OF NOTIONAL PER 1% MOVE")):
        series = [_series("true note", xs, ill[key][TRUE], "var(--s1)"),
                      _series(comp_word, xs, ill[key][comp], "var(--s2)", "5 4")]
        if not all(s["xs"] for s in series):
            out.append(unavailable(f"the {what} ladder of {ill['note']} on {ill['date']} has no value"))
            continue
        svg = h.xy_chart(series, xdomain=(min(xs), max(xs)), xticks=[t for t in _ticks(min(xs), max(xs), 6) if min(xs) <= t <= max(xs)],
                         xfmt=lambda v: h.snum(v, 0) + "%", yfmt=lambda v: num(v, 0) if abs(v) >= 10 else num(v, 1),
                         ylabel=unit, xlabel=f"SPOT RELATIVE TO THE {rel.upper()}, %",
                         label=f"Seller's position {what} across the spot ladder, {ill['note']} {ill['date']}")
        out.append(chart(f"{what}-{kind}", svg, f"Seller's position {what} · {'KO' if kind == 'ko' else 'KI'} window",
                         caption))
    rows = [[h.snum(x, 1)] + [num(ill["position_delta_pct"][m][i], 1) for m in ill["models"]]
            + [num(ill["position_gamma_pct"][m][i], 2) for m in ill["models"]] for i, x in enumerate(xs)]
    out.append(tbl(f"ladder-{kind}", ["Spot vs level (%)", "Delta true", f"Delta {comp}", "Gamma true",
                                      f"Gamma {comp}"], rows,
                   f"The charts' numbers: {caption}. Delta in % of notional, gamma in % of notional per 1% move."))
    return "".join(out)


def section_gamma(model: Mapping[str, Any]) -> str:
    src, test = _tables_source(model)
    clock = model["primary"]["clock"]
    parts = ['<p class="lead">Futures cannot change gamma; they change only what is carried through it. The table '
             "shows, per technique, the seller's net position gamma at the close one trading day before each KO "
             "observation (the true note plus the Black-76 gamma of the option legs held), the hedge ratio carried "
             "into the fixing and the largest hedge ratio in the window. Only the MO overlays change the gamma "
             "itself.</p>", _no_test_line(model)]
    if src is None:
        parts.append(unavailable("no current test payload or training run"))
    else:
        g = src["gamma_carry"][clock]
        t = src["tables"][clock]["ko"]
        rows = []
        for tid in t["order"]:
            r = g.get(tid)
            if not r:
                continue
            rows.append([code(tid), str(r.get("n_events")), num(r.get("mean_note_gamma"), 2),
                         num(r.get("mean_legs_gamma"), 2), num(r.get("mean_net_gamma"), 2),
                         num(r.get("mean_net_gamma_1pct_bp"), 1), pct(r.get("mean_hedge_into_fixing")),
                         pct(r.get("mean_hedge_max")), pct(r.get("max_hedge_max"))])
        parts.append(tbl("gamma-carry", ["Technique", "Events", "Note Γ", "Legs Γ", "Net Γ", "Net Γ x (1% S)² (bp)",
                                         "Hedge into fixing (%)", "Mean max hedge (%)", "Max hedge (%)"], rows,
                         ("Test slice" if test else "Training figures") + f", {clock} clock: means over the KO pin "
                         f"events of {t['book']} at the T-1 close; gamma in CNY per index point squared (seller's "
                         "position), hedge ratio = IM notional / note notional in %.",
                         note="Net Γ x (1% of spot)² is twice the P&L of a 1% move at that gamma, bp of notional. "
                              "A positive net gamma gains on a move either way."))
    states = model["states"]
    parts.append(h.sub("The seller's position delta and gamma across the spot ladder"))
    if states is None:
        parts.append(unavailable(model["states_problem"] or "no recorded states"))
    else:
        parts.append('<p>Two illustrative days, chosen by a fixed rule from training-dated days (before the split): '
                     "KO, the close one trading day before a KO observation of the 2Y book with the survived branch "
                     "priced, nearest the KO level; KI, a not-knocked-in day of the seasoned book in its KI window, "
                     "nearest the KI level. Ties go to the earlier date. The dashed line is the survived branch (KO) or "
                     "the knocked-in note (KI): the position the seller holds the moment after the fixing goes the "
                     "other way.</p>")
        for kind, word in (("ko", "survived branch"), ("ki", "knocked in")):
            ill = states["illustrative"].get(kind)
            if ill is None:
                parts.append(unavailable(f"no training-dated {kind.upper()}-window day in the recorded states"))
            else:
                parts.append(_ladder_block(ill, kind, word))
    return h.section(7, "gamma", "GAMMA", "Gamma carried into a fixing", "".join(parts))


def _dist_block(block: Mapping[str, Any], kind: str, label: str, alpha: float) -> str:
    data = block["dist"].get(kind) or {}
    if "base" not in data:
        return unavailable(f"no {kind.upper()} event in the {label}")
    rows, trows = [], []
    for tid, by_event in data.items():
        vals = [by_event[e] for e in sorted(by_event)]
        es10 = measures.es(vals, alpha)
        rows.append({"name": tid, "values": vals, "tips": [f"{tid} · {e.split('|')[1]} {e.split('|')[2]} · "
                                                           f"{num(by_event[e], 1)} bp" for e in sorted(by_event)],
                     "colour": "var(--s1)" if tid == "base" else "var(--s2)", "es": es10})
        a = np.asarray(vals)
        trows.append([code(tid), str(len(vals)), num(es10, 1), num(float(a.min()), 1), num(float(np.percentile(a, 10)), 1),
                      num(float(np.median(a)), 1), num(float(a.mean()), 1), num(float(a.max()), 1)])
    word, one = ("KO pin events", "KO pin event") if kind == "ko" else ("KI approaches", "KI approach")
    svg = strip(rows, label=f"Event-window P&L of the baseline and the selected techniques, {word}",
                xlabel="EVENT-WINDOW P&L OF THE SELLER, BP OF NOTIONAL", alpha=alpha)
    cap = f"{label}, close clock: one dot per {one}, the baseline against the training picks and composites"
    return (chart(f"dist-{kind}", svg, f"Event-window P&L · {word}", cap)
            + tbl(f"dist-{kind}", ["Technique", "Events", "ES10", "Worst", "10th pct", "Median", "Mean", "Best"], trows,
                  f"The chart's summary ({label}, close clock): event-window P&L, bp of notional.")
            + _per_event(block, kind, data, label, one))


def _per_event(block: Mapping[str, Any], kind: str, data: Mapping[str, Mapping[str, float]], label: str,
               one: str) -> str:
    """Every plotted dot: one row per event, one column per technique (collapsed under the chart)."""
    hits = block.get("stats", {}).get(kind, {}).get("hits", {})
    hit = "Fired" if kind == "ko" else "Knocked in"
    events = sorted(data["base"], key=lambda e: (e.split("|")[2], e))
    rows = [[esc(e.split("|")[2]), code(e.split("|")[1]), "yes" if hits.get(e) else ""]
            + [num(data[t].get(e), 1) for t in data] for e in events]
    table = tbl(f"dist-{kind}-events", ["Event date", "Note", hit] + list(data), rows,
                f"The chart's dots ({label}, close clock): event-window P&L of every {one}, bp of notional.")
    return f'<details class="dots"><summary>Every {one}: the plotted values ({len(events)} rows)</summary>{table}</details>'


def _path_block(path: Mapping[str, Any], label: str) -> str:
    ids = list(path["days"])
    dates = [d for d, _, _ in path["days"][ids[0]]]
    xs = list(range(len(dates)))
    colours = {"base": "var(--s1)", "sop_futures": "var(--s2)", "sop_overlay": "var(--s3)"}
    cum = {tid: list(np.cumsum([x for _, x, _ in path["days"][tid]])) for tid in ids}
    ticks = xs if len(xs) <= 12 else xs[:: max(1, len(xs) // 8)]
    fmt = (lambda v: dates[int(v)][5:] if 0 <= int(v) < len(dates) else "")
    pnl = h.xy_chart([_series(t, xs, cum[t], colours.get(t, "var(--s4)")) for t in ids], xdomain=(0, max(len(xs) - 1, 1)),
                     xticks=ticks, xfmt=fmt, yfmt=lambda v: num(v, 0), ylabel="CUMULATIVE HEDGED P&L, BP OF NOTIONAL",
                     xlabel="TRADING DAY OF THE WINDOW", label=f"Cumulative hedged P&L through {path['event']}")
    hr = h.xy_chart([_series(t, xs, [100.0 * r for _, _, r in path["days"][t]], colours.get(t, "var(--s4)"))
                     for t in ids], xdomain=(0, max(len(xs) - 1, 1)), xticks=ticks, xfmt=fmt,
                    yfmt=lambda v: num(v, 0), ylabel="HEDGE RATIO, % OF NOTIONAL", xlabel="TRADING DAY OF THE WINDOW",
                    label=f"Hedge ratio through {path['event']}")
    note, obs = path["event"].split("|")[1], path["event"].split("|")[2]
    cap = (f"{label}, close clock: the KO pin event with the worst baseline window P&L, note {note}, observation "
           f"{obs}")
    rows = []
    for i, d in enumerate(dates):
        cells = [esc(d)]
        for t in ids:
            day = path["days"][t][i] if i < len(path["days"][t]) else None
            cells += ["–", "–"] if day is None else [num(cum[t][i], 1), pct(day[2])]
        rows.append(cells)
    head = ["Date"] + [x for t in ids for x in (f"{t} cum. P&L", f"{t} hedge %")]
    return (chart("path-ko", pnl, "An event path · hedged P&L", cap)
            + chart("path-ko-hedge", hr, "An event path · hedge ratio", cap)
            + tbl("path-ko", head, rows, f"The charts' numbers ({cap}): cumulative hedged P&L, bp of "
                                         "notional; hedge ratio, IM notional / note notional in %.",
                  note="A hedge ratio on or after the fixing day of an ended note is the hedge of residual option legs "
                       "left on the book (they could not all be closed that day), within the futures position limit."))


def section_pnl(model: Mapping[str, Any]) -> str:
    parts = ['<p class="lead">The distribution behind the ES10 numbers, and one event followed day by day through '
             "its fixing.</p>"]
    blocks = []
    if model["test_events"] is not None:
        blocks.append((model["test_events"], "Test slice"))
    elif model["train_events"] is not None:
        parts.append(_no_test_line(model))
        blocks.append((model["train_events"], "Training figures (events dated before the split)"))
    if not blocks:
        parts.append(unavailable("no current run files to draw the event P&L from"))
    for block, label in blocks:
        for kind in ("ko", "ki"):
            parts.append(_dist_block(block, kind, label, model["alpha"]))
        if block["path"] is not None:
            parts.append(_path_block(block["path"], label))
        else:
            parts.append(unavailable(f"no KO pin event in the {label}"))
    return h.section(8, "pnl", "EVENT P&L", "Event-window P&L and an event path", "".join(parts))


def _stats_rows(windows: Mapping[str, Any], key: str, d: int) -> List[List[str]]:
    rows = []
    for book, models_ in windows.items():
        for m, cell in models_.items():
            s = cell[key]
            rows.append([code(book), code(m), str(s["n"]), num(s["mean"], d), num(s["median"], d), num(s["p10"], d),
                         num(s["p90"], d), num(s["min"], d), num(s["max"], d)])
    return rows


def section_extras(model: Mapping[str, Any]) -> str:
    parts = ['<p class="lead">Descriptive figures that decide nothing.</p>']
    states = model["states"]
    if states is None:
        parts.append(unavailable(model["states_problem"] or "no recorded states"))
    else:
        span = (f"days before {states['cutoff']} (the split: no test-dated state is read without a current test "
                "payload)" if states["cutoff"] else "every recorded day")
        head = ["Book", "Model", "Windows", "Mean", "Median", "10th pct", "90th pct", "Min", "Max"]
        parts.append(tbl("reserve", head, _stats_rows(states["windows"], "reserve_bp", 1),
                         f"The reserve a managed booking would carry, on the first day of each window of a hedge model, "
                         f"{span}: seller PV on the true note minus seller PV on the model, bp of notional (positive = "
                         "a reserve held against the pin)."))
        parts.append(tbl("step", head, _stats_rows(states["windows"], "step_pct", 1),
                         f"The hedge step on the first day of each window, {span}: d(model) − d(true), 1% bump deltas "
                         "as % of notional (positive = the technique buys IM when it switches on); its cost is inside "
                         "the event window.",
                         note="Models: surv = the next KO observation disabled; ko_sNNN = every future KO barrier "
                              "shifted by NNN/100 % in the over-hedge direction; ko_w100 = the wrong-way 1% shift; "
                              "ki_m100 / ki_m200 = KI barrier 1% / 2% lower; ki_p100 = 1% higher (wrong way); kin = "
                              "the note forced knocked in." + (f" {states['skipped']} windows skipped for a null "
                                                              "ladder price." if states["skipped"] else "")))
    src, test = _tables_source(model)
    label = "Test slice" if test else "Training figures"
    if src is None:
        parts.append(unavailable("no current test payload or training run: overlay fills, flags, gap signs, band rows"))
        return h.section(9, "extras", "EXTRAS", "Descriptive extras", "".join(parts))
    clock = model["primary"]["clock"]
    x = src["extras"]
    ov = x.get("overlay", {}).get(clock, {})
    if ov:
        rows = [[code(t), str(r["leg_days"]), str(r["overlay_days"]), str(r["phi_lt1_days"]), pct(r["phi_mean"]),
                 pct(r["phi_min"])] for t, r in ov.items()]
        parts.append(tbl("overlay", ["Technique", "Days with legs", "Days wanting legs", "Days filled < 100%",
                                     "Mean fill (%)", "Min fill (%)"], rows,
                         f"{label}, {clock} clock, whole lives of every note in the runs: MO overlay fill ratios "
                         "(filled desired lots / desired lots) at the 500 mio desk scale."))
    else:
        parts.append(unavailable("no technique held or wanted an MO leg in these runs"))
    flags = x.get("flags", {}).get(clock, {})
    kinds = {k: src["tables"][clock][k]["rows"] for k in ("ko", "ki")}
    frows = []
    for tid in src.get("order", flags.keys()):
        f = flags.get(tid, {})
        clips = sum(int(kinds[k].get(tid, {}).get("clips") or 0) for k in kinds)
        if not any(f.get(k) for k in EXTRA_FLAGS) and not clips:
            continue
        frows.append([code(tid)] + [str(int(f.get(k, 0))) for k in EXTRA_FLAGS] + [str(clips)])
    if frows:
        parts.append(tbl("flags", ["Technique"] + [f.replace("overlay_", "").replace("_", " ") for f in EXTRA_FLAGS]
                         + ["Clip days (windows)"], frows,
                         f"{label}, {clock} clock: refusals, cuts and position-limit hits, note-days over whole lives; "
                         "the last column counts the days the hedge-ratio cap bound inside the KO and KI event windows.",
                         note="Columns are the ledger flags overlay_no_series, overlay_no_strike, overlay_no_vol, "
                              "overlay_not_openable, volume_cut, series_limit, refused_open. The four overlay "
                              "refusals: no new overlay that day (no "
                              "series ending after the observation, a strike not listed or the barrier more than one "
                              "listed step beyond the listing, a leg without an implied vol, a leg failing the volume "
                              "or open-interest screen); volume_cut: a trade cut to 10% of the day's volume; "
                              "series_limit: a position cut to 1,200 lots; refused_open: an open refused."))
    else:
        parts.append(unavailable("no refusal, cut or position-limit hit in these runs"))
    gaps = x.get("gap_observations", {})
    if gaps:
        rows = [[code(b), str(g["n"]), str(g["positive"]), str(g["negative"]), str(g["zero"]), str(g["non_positive"])]
                for b, g in gaps.items()]
        parts.append(tbl("gap-sign", ["Book", "Observations", "Gap > 0", "Gap < 0", "Gap = 0", "Gap ≤ 0"], rows,
                         f"{label}: the sign of the KO gap on the first KO-window day of each observation (the seller's "
                         "loss if the note is redeemed rather than survives; positive = redemption hurts the seller).",
                         note="A gap ≤ 0 reverses the over-hedge direction of the shift models for that observation "
                              "(zero: no shift model)."))
    band = x.get("band", {})
    if band:
        rows = [[code(b), code(t), str(r.get("n_notes")), num(r.get("mean_cost_bp"), 2), num(r.get("mean_im_lots_traded"), 1),
                 num(r.get("std_bp"), 1), num(r.get("mean_terminal_bp"), 1)] for b, rs in band.items() for t, r in rs.items()]
        parts.append(tbl("band", ["Book", "Technique", "Notes", "Mean cost (bp)", "Mean IM lots traded", "Daily std (bp)",
                                  "Mean terminal (bp)"], rows,
                         f"{label}, {clock} clock, whole lives of the headline notes: the no-trade band rows against "
                         "the baseline (cost lever, no verdict); IM lots per 10 mio note."))
    return h.section(9, "extras", "EXTRAS", "Descriptive extras", "".join(parts))


def section_train(model: Mapping[str, Any]) -> str:
    sel = model["selection"]
    if sel is None:
        return h.section(10, "train", "TRAINING PICKS", "The training selection",
                         unavailable("no current training selection: " + "; ".join(model["guard"]["problems"][:1])))
    rows, cls = [], []
    for kind in ("ko", "ki"):
        base = sel.get("base", {}).get(kind, {})
        rows.append([kind.upper(), code("base"), code("base"), num(base.get("es10"), 1), "–", ""])
        cls.append("")
        for row in sel.get("rows", {}).get(kind, []):
            pick = sel["picks"][kind][row]
            for gid, g in pick.get("grid", {}).items():
                chosen = gid == pick.get("id")
                rows.append([kind.upper(), code(row), code(gid), num(g.get("es10"), 1), num(g.get("cost_bp"), 2),
                             '<b class="hit-yes">pick</b>' if chosen else ""])
                cls.append("vrow" if chosen else "")
    ev = sel.get("events", {})
    cap = ("Training figures, close clock: per row the grid with its training ES10 of the event-window P&L (bp of "
           f"notional) and the grid point picked on the training events (KO: {ev.get('ko', {}).get('n', '–')} events "
           f"in {ev.get('ko', {}).get('weeks', '–')} weeks of {ev.get('ko', {}).get('book', '–')}; KI: "
           f"{ev.get('ki', {}).get('n', '–')} in {ev.get('ki', {}).get('weeks', '–')} weeks of "
           f"{ev.get('ki', {}).get('book', '–')}); ties go to the lower cost.")
    body = ('<p class="lead">The parameters were picked on the training events only: per row the grid point with the '
            "highest training ES10 under close, whether or not it beats the baseline. These are training figures, "
            "optimistic by construction, and decide nothing.</p>"
            + tbl("training-picks", ["Barrier", "Row", "Grid point", "Training ES10", "Mean cost (bp)", "Pick"], rows,
                  cap, row_cls=cls))
    comp = sel.get("composites") or {}
    crow = [[code(c), code(str(m.get("ko"))), code(str(m.get("ki")))] for c, m in comp.items()]
    body += (tbl("composites", ["Composite", "KO member", "KI member"], crow,
                 "Training selection: the composites' members, picked across rows by the same training criterion. "
                 "sop_futures uses its KO member inside the KO window and its KI member inside the KI window; "
                 "sop_overlay adds the two overlay picks on top of sop_futures.")
             if crow else unavailable("no composite: a barrier type had no training event"))
    return h.section(10, "train", "TRAINING PICKS", "The training selection", body)


def _after_end(model: Mapping[str, Any]) -> str:
    """Rows of a run after its note ended (residual option legs still hedged), per technique, for the slice shown."""
    block, label = _events_shown(model)
    after = None if block is None else block.get("after_end")
    cap = model["prereg"]["im_clip"]
    rule = ("<b>The futures position limit binds every futures target</b> (hedge ratio in "
            f"[{num(cap[0], 2)}, {num(cap[1], 1)}]), including the hedge of option legs left on the book after a note "
            "has ended: such a hedge is clipped and counted like any other.")
    if after is None:
        return rule + " The run files of the slice shown are not available, so the rows after a note's end are not counted."
    if not after:
        return rule + f" In the slice shown ({label}) no run continues after its note's end."
    parts = "; ".join(f"<code>{esc(t)}</code> {a['rows']} rows on {a['notes']} notes" for t, a in after.items())
    return (rule + f" Rows after the note's end (the residual legs hedged), {label}, close clock: {parts}. The run "
            "files store the clip count per whole life, not per row, so how many of these rows were clipped is not "
            "shown.")


def _ki_tail_line(model: Mapping[str, Any]) -> str:
    block, label = _events_shown(model)
    tail = [] if block is None else block["stats"].get("ki", {}).get("tail", [])
    if not _concentrated(tail):
        return ""
    k = sum(r["events"] for r in tail)
    return (f" The KI tail is concentrated in single market days: {tail[0]['events']} of the {k} tail approaches "
            f"({label}) are dated {esc(tail[0]['date'])} (section 03).")


def section_limits(model: Mapping[str, Any]) -> str:
    days = model["limit_days"] or []
    limit_line = ("<b>Limit days.</b> On a day the far IM contract closed at its daily price limit the close clock "
                  "books a trade that could not be executed and the twap clock fills it at a VWAP no desk could have "
                  "known: the twap improvement of the KI tail on such days is not an achievable fill (section 03). "
                  + (f"Limit closes in the data: {', '.join(esc(r['date']) + ' (' + r['side'] + (', locked' if r['locked'] else '') + ')' for r in days)}."
                     if days else "No limit close in the data." if model["limit_days"] is not None
                     else "The board could not be read, so limit days are not listed.")
                  + _ki_tail_line(model))
    items = [limit_line, _after_end(model),
             "<b>The carried fill ratio is reset when an overlay closes.</b> A day whose target wants no leg and "
             "carries no refusal sets the carried fill ratio to 0, so the residual of a closed overlay covers nothing "
             "on a later refusal day; that is why the minimum fill ratio of an overlay technique can read 0%.",
        "<b>Scenario sizing mostly hits the 3x cap.</b> Unless the option series ends within about a week of the "
        "observation, the scenario lot count is capped at 3 times the payoff lot count, so "
        "<code>mo_ko@1,scenario</code> is in practice about 3 x the payoff lots.",
        "<b>Overlay strikes.</b> An overlay is refused when the barrier is more than one listed strike step beyond "
        "the listing; within one step it is built, and its payout kink then sits up to one listed strike step from "
        "the barrier.",
        "<b>Carried coverage.</b> On a day an overlay is refused while legs are held, the coverage ratio of the last "
        "sized day is carried; it is not rescaled for expiries or partial unwinds since that day.",
        "<b>freeze inside the overlay composite</b> holds the hedge carried, legs included, which is exact only while "
        "the legs do not change that day.",
        "<b>The seasoned bookings are synthetic bookings on real paths:</b> real spot, vol, IM and MO marks, but "
        "bookings the desk did not make on those dates; weekly bookings overlap in calendar time, so their events are "
        "not independent. The 2Y product's own KI approaches are two episodes with a small gap.",
        "<b>Clocks.</b> close lets the seller trade at the fixing price itself and understates a true pin; next_open "
        "overstates it; twap uses close-state targets with average-price fills, and its trade size is correlated with "
        "close minus VWAP. The option VWAP is not used as a fill (it is outside the day's range on thin contracts).",
        "<b>The seasoned maturity payment</b> is about 2 years of coupon (coupon x (1.5 + T) with T the rolled "
        "maturity in years, up to about 1% away from 2.0), not exactly 2.0.",
        "<b>Intraday rules</b> (an exit at the closing auction, a pin band on the fixing day) are carried from earlier "
        "studies and not re-tested here.",
        "<b>The MO series limit</b> of 1,200 lots per side is the listing-time CFFEX client limit; its current value "
        "was not verified. One flat vol per day: no smile, no vega hedge.",
    ]
    return h.section(11, "limits", "LIMITS", "Limits, stated plainly", h.points(items))


def section_sources(model: Mapping[str, Any]) -> str:
    g = model["guard"]
    sel = model["selection"] or {}
    pay = model["test_tables"] or {}
    ts = g.get("train_stamp") or {}
    short = {"prereg_hash": "pre-registration", "events_hash": "events file", "code_hash": "ledger-side code",
             "inputs_hash": "recorded inputs"}
    rows = [[esc(short[k]), code((g["hashes"].get(k) or "–")[:12]), code((sel.get(k) or "–")[:12]),
             code((ts.get(k) or "–")[:12]), code(((pay.get("stamp") or {}).get(k) or "–")[:12])] for k in HASHES]
    covered = (f"The ledger-side code hash covers the package files {', '.join(measures.CODE_FILES)}, the driver "
               f"{measures.DRIVER_FILE}, the sibling s2ts files {', '.join(measures.S2TS_CODE_FILES)} and the numpy, "
               "scipy and quantark versions; the report itself is not in it (it only displays). The recorded-inputs "
               "hash covers the board, the chain and every state file's pricing fingerprint.")
    body = (tbl("sources", ["Hash", "Current files", "Training selection", "Training runs", "Test payload"], rows,
                "The four hashes the guard compares (first 12 hex digits): a test number is printed only when the "
                "selection and the payload carry the current four.", note=esc(covered))
            + h.points([f"Training run files: {g['train_present']} of {g['train_listed']} note files present"
                        + ("" if g["training_ok"] else " (not shown: " + "; ".join(g["training_problems"]) + ")"),
                        f"Test run files read: {g['test_present']}",
                        *(esc(n) for n in g["notes"])], dim=True))
    return h.section(12, "sources", "SOURCES", "Provenance", body)


def masthead(model: Mapping[str, Any]) -> Tuple[str, str]:
    g = model["guard"]
    data = model["data"] or {}
    sample = model["sample"] or []

    def count(book: str, kind: str, slice_: str) -> str:
        r = next((x for x in sample if x["book"] == book and x["kind"] == kind and x["slice"] == slice_), None)
        return "–" if r is None else f"{r['events']}<small>/{r['weeks']} WK</small>"

    kb, ib = model["books"]["ko"], model["books"]["ki"]
    kpis = [(count(kb, "ko", "train"), f"KO pin events / weeks · training · {kb}"),
            (count(kb, "ko", "test"), f"KO pin events / weeks · test · {kb}"),
            (count(ib, "ki", "test"), f"KI approaches / weeks · test · {ib}"),
            ("SHOWN" if g["test_ok"] else "BLIND", "Test numbers")]
    tiles = "".join(f'<div class="kpi"><div class="kv">{v}</div><div class="kl">{esc(lab)}</div></div>' for v, lab in kpis)
    nav = "".join(f'<a href="#{sid}"><b>{i + 1})</b> {esc(lab)}</a>' for i, (sid, lab) in enumerate(SECTIONS))
    chrome = (f'<div class="top"><header class="fbar"><span class="brand">QUANTARK</span><span class="cmd">SNBL PIN '
              f'&lt;GO&gt;</span><span class="fld">STUDY <b>snowball_2y_gamma_pin_sop</b></span><span class="fld">UL '
              f'<b>000852.SH</b></span><span class="fld">VIEW <b>SELLER</b></span><span class="fld">ENGINE <b>QUAD V2'
              f'</b></span><span class="fld date">{esc(data.get("last", ""))}</span><button class="theme" id="theme" '
              f'type="button" aria-label="Switch colour theme">◐ <b>LIGHT</b></button></header><nav class="menu" '
              f'id="menu">{nav}</nav></div>')
    if g["test_ok"]:
        head = "Gamma pin risk of a short 2Y snowball: the test slice"
        banner = ('<div class="banner ok"><b>Test figures shown.</b> The training selection and the test payload '
                  "carry the current four hashes (pre-registration, events, ledger-side code, recorded inputs).</div>")
    else:
        head = "Gamma pin risk of a short 2Y snowball: research, sample and training figures"
        reasons = "".join(f"<span>· {esc(p)}.</span>" for p in g["problems"]) or "<span>· the guard did not pass.</span>"
        banner = (f'<div class="banner"><b>No test number is shown.</b> {reasons}<span>The page shows the research, '
                  "the sample and the training figures that are current; every training number is labelled as "
                  "such.</span></div>")
    answer = ("How should the seller of a 2Y CSI 1000 snowball hedge the gamma pin at a KO fixing and near the KI "
              "barrier? Futures-only rules, barrier shifts and bends, MO option overlays and calendar SOPs are "
              "compared with the plain delta hedge on real closes from "
              f"{esc(data.get('first', '–'))} to {esc(data.get('last', '–'))}, under a verdict rule written down "
              "before any backtest ran.")
    hero = (f'<div class="hero"><div class="kicker">SNOWBALL · GAMMA PIN · SELLER</div><h1>{esc(head)}</h1>'
            f'<p class="answer">{answer}</p></div>{banner}<div class="kpis">{tiles}</div>')
    return chrome, hero


EXTRA_CSS = """
.banner{margin:28px 0 0;padding:14px 18px;border-left:3px solid var(--amber);background:var(--panel);
  font:13px/1.55 var(--mono);color:var(--ink)}
.banner span{display:block;color:var(--ink-2)}
.banner b{color:var(--amber);font-weight:600}
.banner.ok{border-left-color:var(--up)}.banner.ok b{color:var(--up)}
p.tcap{margin:28px 0 4px;font:12px/1.5 var(--mono);color:var(--ink-2);max-width:none}
@media (max-width:820px){.fbar>span.fld{display:none}}
details.dots{margin:10px 0 0}
details.dots summary{cursor:pointer;font:11px/1.5 var(--mono);letter-spacing:.06em;color:var(--amber)}
details.dots .scroll{max-height:420px;overflow:auto}
tr.vrow td:first-child code{color:var(--amber)}
.chart .esmark{stroke:var(--amber);stroke-width:2}
section .body table{font-size:12px}
th,td{padding-right:10px}
th{white-space:normal;vertical-align:bottom;min-width:4.5em}
table.text td{white-space:normal}
"""


def render(model: Mapping[str, Any]) -> str:
    """One self-contained HTML page from the report model (``build_model``)."""
    chrome, hero = masthead(model)
    content = (hero + section_design(model) + section_ko(model) + section_ki(model) + section_kigap(model)
               + section_clocks(model) + section_sens(model) + section_gamma(model) + section_pnl(model)
               + section_extras(model) + section_train(model) + section_limits(model) + section_sources(model)
               + "<footer>QUANTARK · example/case_studies/snowball_2y_gamma_pin_sop · 00_ki_gap_probe → 01_data → "
                 "02_events → 03_pricing → 04_ledger → 05_report · exchange closes and settlements only, synthetic "
                 "notes, no desk data · seller view</footer>")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>2Y snowball · gamma pin SOP</title>{h.THEME_INIT_JS}<style>{h.CSS}{EXTRA_CSS}</style></head>"
            f"<body>{chrome}<main>{content}</main>{h.PAGE_JS}</body></html>\n")
