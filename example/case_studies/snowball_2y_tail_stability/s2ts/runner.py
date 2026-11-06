"""The paired hedging backtest (design §4.4): one run = one inception × one tail, both books.

Per run, on the inception close (day 0): the tail's curve, the note booked at its
OWN day-0 model PV (the cash account starts at ``PV0_own``), one ``DayGreeks``
shared by both books, both books filled (day-0 trades: costs counted, no P&L).
Each later close ``t`` (per book, in this order):

1. ``extra`` marks from the day's CFFEX rows, only if the book holds something
   the chain cannot mark (``book.unmarkable``);
2. book P&L of the overnight positions (``pnl_split``: IM margin, MO MTM);
3. ``NoteLife.step`` → the day's note cash (paid TO the holder);
4. short-note P&L ``−(PV_t − PV_{t−1}) − cash_t`` with ``PV_t = 0`` once dead;
5. accrual ``r·cash_{t−1}·Δt`` (ACT/365 calendar days);
6. alive → greeks (once, shared) and rebalance; dead or last data day → unwind.

``hedged = note_pnl + book_pnl + accrual`` (gross of costs). The cash accounts
are kept gross of costs too (costs are reported separately and excluded from
both terminals): ``cash_t = cash_{t−1}(1 + rΔt) − MO premium_t − note cash_t +
IM margin_t``. Terminals: own-PV ``= cash_T − PV_T`` (account from ``PV0_own``;
equals Σ hedged), traded-price = the same with the account started at the
traded price 0 (``PV0_hf`` by the coupon solve), i.e. net of the day-0 gap.
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import psutil

from quantark.param import FlatRateCurve
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import safe_log

from . import ACT, ANCHOR_WINDOW, CFFEX_EXTENSION_DIR, CHAIN_PATH, HISTORY_DIR, NOTIONAL, RATE
from .books import Book, BookA, BookB, offchain_marks, total_cost, turnover
from .chain import DayChain, load_chain, load_day_rows
from .note import DayGreeks, NoteLife, build_note, build_terms, daily_greeks, make_env, solve_coupon
from .tails import anchor_series, build_curve, pillar_inputs

#: V1 parity thresholds (Global Constraints).
PARITY_PV_BP = 0.5
PARITY_LOTS = 0.01
RSS_LIMIT_BYTES = 24 * 1024 ** 3
RSS_POLL_S = 30.0
BOOKS = ("a", "b")


# ---------------------------------------------------------------------------
# Engines (one per process, reused)
# ---------------------------------------------------------------------------

_ENGINES: Dict[str, Any] = {}


def v2_engine():
    if "v2" not in _ENGINES:
        from quantark.asset.equity.engine.quad import SnowballQuadEngineV2
        from quantark.asset.equity.param import QuadV2Params

        _ENGINES["v2"] = SnowballQuadEngineV2(params=QuadV2Params())
    return _ENGINES["v2"]


def v1_engine():
    if "v1" not in _ENGINES:
        from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
        from quantark.asset.equity.param import QuadParams

        _ENGINES["v1"] = SnowballQuadEngine(params=QuadParams(readout="transition"))
    return _ENGINES["v1"]


class V1NoLifecycle:
    """V1's ``price(product, env)`` behind the V2 signature; refuses a lifecycle state (parity is day 0, fresh)."""

    def __init__(self, engine) -> None:
        self.engine = engine

    def price(self, product, env, *, lifecycle_state=None) -> float:
        if lifecycle_state is not None:
            raise ValidationError("V1 parity prices the fresh product: no lifecycle state")
        return float(self.engine.price(product, env))


def parity_flag(pv_v2: float, pv_v1: float, lots_v2: Sequence[float], lots_v1: Sequence[float]) -> bool:
    """True when |PV_V2 − PV_V1| > 0.5 bp of notional or any pillar lot differs by > 0.01."""
    if len(lots_v2) != len(lots_v1):
        raise ValidationError(f"lot vectors differ in length: {len(lots_v2)} vs {len(lots_v1)}")
    pv_bp = 1e4 * abs(pv_v2 - pv_v1) / NOTIONAL
    lot_diff = max((abs(a - b) for a, b in zip(lots_v2, lots_v1)), default=0.0)
    return pv_bp > PARITY_PV_BP or lot_diff > PARITY_LOTS


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------


def _basis(f: float, s: float) -> float:
    return float(safe_log(f / s))


def _d_far_basis(prev: DayChain, dc: DayChain) -> Optional[float]:
    """Same far IM contract both days, else ``None`` (its roll day)."""
    if dc.im[-1].code != prev.im[-1].code:
        return None
    return _basis(dc.im[-1].close, dc.spot) - _basis(prev.im[-1].close, prev.spot)


def _d_mo_basis(prev: DayChain, dc: DayChain) -> Optional[float]:
    """Same MO series both days, else ``None`` (the MO roll day)."""
    if dc.mo.series != prev.mo.series:
        return None
    return _basis(dc.mo.forward, dc.spot) - _basis(prev.mo.forward, prev.spot)


def _d_vol(prev: DayChain, dc: DayChain) -> Optional[float]:
    """Change of the flat vol in vol points, same MO expiry both days, else ``None`` (the MO roll day)."""
    if dc.mo.expiry != prev.mo.expiry:
        return None
    return 100.0 * (dc.mo.vol - prev.mo.vol)


def run_one(
    inception: date,
    tail: str,
    chains: Sequence[DayChain],
    anchors: Mapping[date, float],
    coupon: float,
    *,
    parity: bool = True,
    engine=None,
    v1=None,
    max_days: Optional[int] = None,
    rows_loader: Optional[Callable[[date], Mapping[str, Mapping[str, float]]]] = None,
) -> Dict[str, Any]:
    """The daily loop for one inception × tail (see the module docstring); a JSON-ready dict."""
    started = time.perf_counter()
    engine = engine if engine is not None else v2_engine()
    rows_loader = rows_loader or (lambda d: load_day_rows(d, HISTORY_DIR, CFFEX_EXTENSION_DIR))
    days_index = {dc.day: i for i, dc in enumerate(chains)}
    if inception not in days_index:
        raise ValidationError(f"inception {inception} is not a chain day")
    path = list(chains[days_index[inception]:])
    if max_days is not None:
        path = path[: int(max_days) + 1]
    rc = FlatRateCurve(RATE)

    def market(dc: DayChain):
        curve = build_curve(dc, tail, anchors.get(dc.day))
        return curve, make_env(dc, curve.to_dividend_yield(rc))

    def greeks(dc: DayChain, product, curve, env) -> DayGreeks:
        _, mults = pillar_inputs(dc)
        return daily_greeks(product, env, curve, engine, life.lifecycle, mults)

    # ---- day 0
    dc0 = path[0]
    terms = build_terms(inception)
    life = NoteLife(build_note(coupon, dc0.spot, terms), inception)
    curve, env = market(dc0)
    nd = life.step(dc0, env)
    g = g0 = greeks(dc0, nd.product, curve, env)
    priced_days = 1
    pv0 = g.pv
    books: Dict[str, Book] = {"a": BookA(), "b": BookB()}
    last_day0 = len(path) == 1
    trades0 = {k: (b.unwind(dc0) if last_day0 else b.rebalance(dc0, b.target(dc0, g))) for k, b in books.items()}
    cash = {k: pv0 - Book.premium(trades0[k]) for k in BOOKS}         # own: sold at PV0_own
    cash_tr = {k: -Book.premium(trades0[k]) for k in BOOKS}            # traded price: 0
    t_parity = 0.0
    parity_out: Optional[Dict[str, Any]] = None
    if parity:
        tp = time.perf_counter()
        _, mults = pillar_inputs(dc0)
        g1 = daily_greeks(build_note(coupon, dc0.spot, terms), env, curve,
                          V1NoLifecycle(v1 if v1 is not None else v1_engine()), None, mults, with_vega=False)
        parity_out = {
            "pv_v2": g.pv, "pv_v1": g1.pv, "pv_diff_bp": 1e4 * (g.pv - g1.pv) / NOTIONAL,
            "lots_v2": list(g.pillar_lots), "lots_v1": list(g1.pillar_lots),
            "max_lot_diff": max(abs(a - b) for a, b in zip(g.pillar_lots, g1.pillar_lots)),
            "flag": parity_flag(g.pv, g1.pv, g.pillar_lots, g1.pillar_lots),
        }
        t_parity = time.perf_counter() - tp

    rows: List[Dict[str, Any]] = []
    prev, pv_prev = dc0, pv0
    end_reason = "data_end"
    alive = True
    for j, dc in enumerate(path[1:], start=1):
        is_last = j == len(path) - 1
        dt = (dc.day - prev.day).days / ACT
        day_rows: Optional[Mapping] = None
        extra: Dict[str, Dict[str, float]] = {}
        split: Dict[str, Tuple[float, float]] = {}
        for k, b in books.items():
            need = b.unmarkable(dc)
            if need:
                day_rows = day_rows if day_rows is not None else rows_loader(dc.day)
                extra[k] = offchain_marks(day_rows, need)
            split[k] = b.pnl_split(prev, dc, extra.get(k))
        curve, env = market(dc)
        nd = life.step(dc, env)
        alive = nd.alive
        g = greeks(dc, nd.product, curve, env) if alive else None
        if alive:
            priced_days += 1
        pv = g.pv if g is not None else 0.0
        note_pnl = -(pv - pv_prev) - nd.cash
        row: Dict[str, Any] = {
            "date": dc.day.isoformat(), "spot": dc.spot, "alive": alive, "knocked_in": nd.knocked_in,
            "note_cash": nd.cash, "note_pnl": note_pnl, "pv": pv,
            "spot_delta": None if g is None else g.spot_delta,
            "pinned_delta": None if g is None else g.pinned_delta,
            "vega": None if g is None else g.vega, "d_vol": _d_vol(prev, dc),
            "d_far_basis": _d_far_basis(prev, dc), "d_mo_basis": _d_mo_basis(prev, dc),
        }
        for k, b in books.items():
            if alive and not is_last:
                trades = b.rebalance(dc, b.target(dc, g), extra.get(k))
            else:
                trades = b.unwind(dc, extra.get(k))
            im_vm, mo_mtm = split[k]
            accrual = cash[k] * RATE * dt
            cash[k] = cash[k] * (1.0 + RATE * dt) - Book.premium(trades) - nd.cash + im_vm
            cash_tr[k] = cash_tr[k] * (1.0 + RATE * dt) - Book.premium(trades) - nd.cash + im_vm
            book_pnl = im_vm + mo_mtm
            row.update({
                f"pnl_{k}": book_pnl, f"vm_{k}": im_vm, f"mtm_{k}": mo_mtm, f"accrual_{k}": accrual,
                f"hedged_{k}": note_pnl + book_pnl + accrual,
                f"cost_{k}": total_cost(trades), f"turn_{k}": turnover(trades), f"roll_{k}": bool(b.rolled),
            })
        rows.append(row)
        prev, pv_prev = dc, pv
        if not alive:
            end_reason = nd.end_reason
            break

    pv_end = pv_prev if alive else 0.0
    elapsed = time.perf_counter() - started
    out: Dict[str, Any] = {
        "inception": inception.isoformat(), "tail": tail, "coupon": float(coupon),
        "day0_pv": pv0, "day0_spot_delta": g0.spot_delta, "day0_pinned_delta": g0.pinned_delta,
        "day0_lots": list(g0.pillar_lots), "day0_vega": g0.vega,
        "truncated": end_reason == "data_end", "end_reason": end_reason,
        "end_date": prev.day.isoformat(), "live_days": len(rows), "pv_end": pv_end,
        "parity": parity_out,
        "seconds": elapsed, "seconds_parity": t_parity, "priced_days": priced_days,
        "s_per_day": (elapsed - t_parity) / priced_days,
        "days": rows,
    }
    for k in BOOKS:
        out[f"day0_cost_{k}"] = total_cost(trades0[k])
        out[f"day0_turn_{k}"] = turnover(trades0[k])
        out[f"terminal_own_{k}"] = cash[k] - pv_end
        out[f"terminal_{k}"] = cash_tr[k] - pv_end
        out[f"cost_total_{k}"] = out[f"day0_cost_{k}"] + sum(r[f"cost_{k}"] for r in rows)
    return out


# ---------------------------------------------------------------------------
# Coupons (one per inception, hold_forward, cached)
# ---------------------------------------------------------------------------


def write_json_atomic(path: Path, payload: Any) -> None:
    """``path.tmp`` (flushed and fsynced) then ``os.replace``: a reader never sees a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    text = json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def coupons_for(inceptions: Iterable[date], chains: Sequence[DayChain], path: Path,
                *, solver: Optional[Callable[[DayChain], float]] = None) -> Dict[date, float]:
    """``{inception: coupon}``; solved once under hold_forward and cached in ``path``."""
    path = Path(path)
    cached: Dict[str, float] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    by_day = {dc.day: dc for dc in chains}
    solver = solver or (lambda dc: solve_coupon(dc, v2_engine()))
    out: Dict[date, float] = {}
    changed = False
    for inc in inceptions:
        key = inc.isoformat()
        if key not in cached:
            if inc not in by_day:
                raise ValidationError(f"inception {inc} is not a chain day")
            cached[key] = float(solver(by_day[inc]))
            changed = True
        out[inc] = float(cached[key])
    if changed:
        write_json_atomic(path, cached)
    return out


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------


def run_path(out_dir: Path, inception: date, tail: str) -> Path:
    return Path(out_dir) / f"{inception.isoformat()}__{tail}.json"


_WORKER: Dict[str, Any] = {}


def init_worker(chain_path: str) -> None:
    """Spawn-worker initializer: the chain and the 63-day anchors, loaded once per process."""
    chains = load_chain(Path(chain_path))
    _WORKER["chains"] = chains
    _WORKER["anchors"] = anchor_series(chains, ANCHOR_WINDOW)


def worker_job(inception_iso: str, tail: str, coupon: float, max_days: Optional[int], parity: bool) -> Dict[str, Any]:
    """One run in a worker (its engines are built once per process and reused)."""
    if "chains" not in _WORKER:
        init_worker(str(CHAIN_PATH))
    return run_one(date.fromisoformat(inception_iso), tail, _WORKER["chains"], _WORKER["anchors"], coupon,
                   parity=parity, max_days=max_days)


def _kill_children() -> None:
    procs = psutil.Process().children(recursive=True)
    for p in procs:
        try:
            p.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(procs, timeout=5.0)
    for p in alive:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass


class RssGuard(threading.Thread):
    """Polls the RSS of this process plus all its children; above ``limit_bytes`` it trips once:
    sets ``tripped`` and calls ``on_trip`` (default: terminate, then kill, every child process)."""

    def __init__(self, limit_bytes: int = RSS_LIMIT_BYTES, poll_s: float = RSS_POLL_S,
                 on_trip: Optional[Callable[[], None]] = None) -> None:
        super().__init__(name="rss-guard", daemon=True)
        self.limit_bytes = int(limit_bytes)
        self.poll_s = float(poll_s)
        self.on_trip = on_trip or _kill_children
        self.tripped = threading.Event()
        self._halt = threading.Event()  # not ``_stop``: that name is Thread's own
        self.peak = 0

    @staticmethod
    def rss() -> int:
        me = psutil.Process()
        total = me.memory_info().rss
        for child in me.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return int(total)

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        while not self._halt.wait(self.poll_s):
            used = self.rss()
            self.peak = max(self.peak, used)
            if used > self.limit_bytes:
                self.tripped.set()
                self.on_trip()
                return


class RssLimitExceeded(RuntimeError):
    """The group RSS guard terminated the pool."""


class RunsFailed(RuntimeError):
    """Some runs raised; each left ``<inception>__<tail>.error.json`` (every other run completed)."""


class OutDirLocked(RuntimeError):
    """Another live instance holds ``out_dir/.lock``."""


LOCK_NAME = ".lock"


def error_path(out_dir: Path, inception: date, tail: str) -> Path:
    return Path(out_dir) / f"{inception.isoformat()}__{tail}.error.json"


def is_complete_run(path: Path, inception: date, tail: str) -> bool:
    """A run file that parses and is this run's: ``inception``, ``tail`` and a ``days`` list."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (isinstance(data, dict) and data.get("inception") == inception.isoformat()
            and data.get("tail") == tail and isinstance(data.get("days"), list))


def _acquire_lock(out_dir: Path) -> Path:
    """``out_dir/.lock`` via ``O_CREAT|O_EXCL`` holding this pid; a lock whose pid is dead is taken over.

    A lock that cannot be read as a pid is refused (delete it by hand once no instance runs)."""
    lock = Path(out_dir) / LOCK_NAME
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                pid = int(lock.read_text().strip())
            except (OSError, ValueError):
                raise OutDirLocked(f"{lock} exists and holds no pid; remove it if no backtest is running") from None
            if psutil.pid_exists(pid):
                raise OutDirLocked(f"{lock}: process {pid} is running a backtest in {out_dir}") from None
            lock.unlink()                                  # stale: its owner is dead
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(str(os.getpid()))
        return lock
    raise OutDirLocked(f"{lock}: could not take the lock")


def _release_lock(lock: Path) -> None:
    try:
        if lock.read_text().strip() == str(os.getpid()):
            lock.unlink()
    except OSError:
        pass


def _record_failure(out_dir: Path, inc: date, tail: str, err: BaseException, log) -> None:
    tb = "".join(traceback.format_exception(type(err), err, err.__traceback__))
    log(f"  FAILED {inc} {tail}: {err!r}\n{tb}")
    write_json_atomic(error_path(out_dir, inc, tail),
                      {"inception": inc.isoformat(), "tail": tail, "error": repr(err), "traceback": tb})


def run_all(
    inceptions: Sequence[date],
    tails: Sequence[str],
    out_dir: Path,
    *,
    workers: int = 4,
    coupons: Optional[Mapping[date, float]] = None,
    max_days: Optional[int] = None,
    parity: bool = True,
    job: Optional[Callable[[date, str], Dict[str, Any]]] = None,
    rss_limit_bytes: int = RSS_LIMIT_BYTES,
    poll_s: float = RSS_POLL_S,
    log: Callable[[str], None] = print,
) -> List[Path]:
    """Every missing run of ``inceptions × tails`` into ``out_dir``; returns all their paths.

    - One instance per ``out_dir``: ``out_dir/.lock`` (this pid) is taken first, a live
      lock raises ``OutDirLocked`` before anything is touched; a dead pid's lock is
      taken over; the lock is removed on exit.
    - Resumable: a ``<inception>__<tail>.json`` is skipped only if it parses as this
      run (``is_complete_run``), else it is deleted and re-run; stale ``*.json.tmp``
      are deleted unread; a run's ``.error.json`` is deleted before it re-runs; each
      result is written ``.tmp`` (fsynced) then ``os.replace``.
    - A run that raises leaves only ``<inception>__<tail>.error.json`` (traceback) and
      the others go on; after the last run ``RunsFailed`` lists the failures.
    - ``workers=0`` runs in-process (``job(inception, tail)`` when given). Otherwise a
      spawn ``ProcessPoolExecutor`` runs ``worker_job``; launch it from a script file
      (never stdin). An ``RssGuard`` terminates the workers above ``rss_limit_bytes``;
      the queued runs are cancelled and this raises ``RssLimitExceeded``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(out_dir)
    try:
        return _run_all_locked(inceptions, tails, out_dir, workers=workers, coupons=coupons, max_days=max_days,
                               parity=parity, job=job, rss_limit_bytes=rss_limit_bytes, poll_s=poll_s, log=log)
    finally:
        _release_lock(lock)


def _run_all_locked(inceptions, tails, out_dir: Path, *, workers, coupons, max_days, parity, job,
                    rss_limit_bytes, poll_s, log) -> List[Path]:
    for stale in out_dir.glob("*.json.tmp"):
        stale.unlink()
    wanted = [(inc, tail) for inc in inceptions for tail in tails]
    todo = []
    for inc, tail in wanted:
        path = run_path(out_dir, inc, tail)
        if path.exists() and is_complete_run(path, inc, tail):
            continue
        if path.exists():
            log(f"  {path.name} is not a complete run: deleted, re-running")
            path.unlink()
        error_path(out_dir, inc, tail).unlink(missing_ok=True)
        todo.append((inc, tail))
    log(f"{len(wanted)} runs requested, {len(wanted) - len(todo)} complete, {len(todo)} to run")
    failed: List[str] = []
    started = time.perf_counter()

    def done(n: int, inc: date, tail: str, result: Dict[str, Any]) -> None:
        write_json_atomic(run_path(out_dir, inc, tail), result)
        extra = ""
        if "end_reason" in result:
            extra = (f": {result['end_reason']}, {result['live_days']} days, {result['s_per_day']:.2f} s/day")
        log(f"  [{n}/{len(todo)}] {inc} {tail}{extra}, {time.perf_counter() - started:.0f} s elapsed")

    if workers == 0:
        for n, (inc, tail) in enumerate(todo, start=1):
            try:
                if job is not None:
                    result = job(inc, tail)
                else:
                    result = worker_job(inc.isoformat(), tail, float(coupons[inc]), max_days, parity)
            except Exception as err:
                _record_failure(out_dir, inc, tail, err, log)
                failed.append(f"{inc}__{tail}")
                continue
            done(n, inc, tail, result)
    else:
        if job is not None:
            raise ValidationError("a custom job runs in-process only (workers=0)")
        if coupons is None:
            raise ValidationError("coupons are required")
        guard = RssGuard(rss_limit_bytes, poll_s)
        guard.start()
        try:
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=init_worker,
                                     initargs=(str(CHAIN_PATH),)) as pool:
                futures = {
                    pool.submit(worker_job, inc.isoformat(), tail, float(coupons[inc]), max_days, parity): (inc, tail)
                    for inc, tail in todo
                }
                for n, fut in enumerate(as_completed(futures), start=1):
                    inc, tail = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as err:
                        if guard.tripped.is_set():
                            pool.shutdown(wait=False, cancel_futures=True)
                            raise RssLimitExceeded(
                                f"group RSS above {rss_limit_bytes / 1024 ** 3:.0f} GB") from None
                        _record_failure(out_dir, inc, tail, err, log)
                        failed.append(f"{inc}__{tail}")
                        continue
                    done(n, inc, tail, result)
        finally:
            guard.stop()
        if guard.tripped.is_set():
            raise RssLimitExceeded(f"group RSS above {rss_limit_bytes / 1024 ** 3:.0f} GB")
        log(f"peak group RSS {guard.peak / 1024 ** 3:.2f} GB (polled every {poll_s:.0f} s)")
    if failed:
        raise RunsFailed(f"{len(failed)} of {len(todo)} runs failed (see their .error.json): {', '.join(failed)}")
    return [run_path(out_dir, inc, tail) for inc, tail in wanted]
