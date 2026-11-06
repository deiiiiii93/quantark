"""Stage 3: the paired hedging backtest — quarterly inceptions × six tails, Books A and B.

Reads ``OUTPUT_DIR/daily_chain.pkl`` (``01_data.py``). Coupons: one per inception,
solved under hold_forward, cached in ``OUTPUT_DIR/coupons.json``. One JSON per run
in ``OUTPUT_DIR/runs/`` (resumable; ``--quick`` writes ``runs_quick/``, cleared
first). Runs on a spawn pool launched from this file; a group-RSS guard stops the
pool above 24 GB.

Run (from a script file, never stdin):
    python example/case_studies/snowball_2y_tail_stability/03_backtest.py --plan [--s-per-day 5.5]
    python example/case_studies/snowball_2y_tail_stability/03_backtest.py --quick
    python example/case_studies/snowball_2y_tail_stability/03_backtest.py [--workers 4] [--inceptions ..] [--tails ..]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s2ts import ANCHOR_WINDOW, OUTPUT_DIR, RATE  # noqa: E402
from s2ts.chain import load_chain  # noqa: E402
from s2ts.note import NoteLife, build_note, build_terms, inception_dates, make_env  # noqa: E402
from s2ts.runner import RssLimitExceeded, coupons_for, run_all  # noqa: E402
from s2ts.tails import TAILS, anchor_series, build_curve  # noqa: E402

COUPONS_PATH = OUTPUT_DIR / "coupons.json"
RUNS_DIR = OUTPUT_DIR / "runs"
QUICK_DIR = OUTPUT_DIR / "runs_quick"
QUICK_TAILS = ("hold_forward", "converge_hl050")
QUICK_DAYS = 30
BUDGET_HOURS = 10.0


def plan(chains, inceptions, n_tails: int, s_per_day: float, workers: int) -> int:
    """Live days per inception from the spot path alone (KO/KI do not depend on the tail or coupon)."""
    from quantark.param import FlatRateCurve

    rc = FlatRateCurve(RATE)
    by_day = {dc.day: i for i, dc in enumerate(chains)}
    total_priced = 0
    print(f"{'inception':12s} {'end':12s} {'reason':9s} {'live days':>9s} {'priced':>7s}")
    for inc in inceptions:
        life = NoteLife(build_note(0.1, chains[by_day[inc]].spot, build_terms(inc)), inc)
        live, reason, end = 0, "data_end", inc
        for dc in chains[by_day[inc]:]:
            nd = life.step(dc, make_env(dc, build_curve(dc, "hold_forward", None).to_dividend_yield(rc)))
            end = dc.day
            if dc.day == inc:
                continue
            live += 1
            if not nd.alive:
                reason = nd.end_reason
                break
        priced = 1 + live - (0 if reason == "data_end" else 1)   # day 0 + every alive close
        total_priced += priced
        print(f"{inc.isoformat():12s} {end.isoformat():12s} {reason:9s} {live:9d} {priced:7d}")
    hours = total_priced * n_tails * s_per_day / workers / 3600.0
    print(f"\ntotal priced days {total_priced}; budget = {total_priced} x {n_tails} tails x {s_per_day:.2f} s/day "
          f"/ {workers} workers = {hours:.2f} h (limit {BUDGET_HOURS:.0f} h: "
          f"{'OK' if hours <= BUDGET_HOURS else 'OVER'})")
    return 0


def print_quick(paths) -> None:
    for p in paths:
        run = json.loads(Path(p).read_text())
        par = run["parity"]
        print(f"\n{run['inception']} {run['tail']}: coupon {run['coupon']:.6f}, day0 PV {run['day0_pv']:.1f}, "
              f"{run['priced_days']} priced days, {run['s_per_day']:.2f} s/day "
              f"(parity {run['seconds_parity']:.1f} s), end {run['end_reason']}")
        if par is not None:
            print(f"  V1 parity: PV diff {par['pv_diff_bp']:+.4f} bp, max lot diff {par['max_lot_diff']:.4f}, "
                  f"flag {par['flag']}")
        print(f"  {'date':10s} {'note_pnl':>11s} {'pnl_a':>11s} {'pnl_b':>11s} {'hedged_a':>10s} {'hedged_b':>10s} "
              f"{'d_vol':>6s} {'vega_expl':>10s} {'resid_b':>10s}")
        vega_prev = run.get("day0_vega")
        for d in run["days"][:5]:
            # the short's vol P&L: −vega_{t−1}·Δvol_t (holder vega, CNY per vol point)
            expl = None if vega_prev is None or d["d_vol"] is None else -vega_prev * d["d_vol"]
            resid = "" if expl is None else f"{d['hedged_b'] - expl:10.1f}"
            print(f"  {d['date']:10s} {d['note_pnl']:11.1f} {d['pnl_a']:11.1f} {d['pnl_b']:11.1f} "
                  f"{d['hedged_a']:10.1f} {d['hedged_b']:10.1f} "
                  f"{'' if d['d_vol'] is None else format(d['d_vol'], '+6.2f'):>6s} "
                  f"{'' if expl is None else format(expl, '10.1f'):>10s} {resid:>10s}")
            vega_prev = d["vega"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="first inception x (hold_forward, converge_hl050), 30 days")
    ap.add_argument("--plan", action="store_true", help="live days per inception and the run budget; no pricing")
    ap.add_argument("--s-per-day", type=float, default=5.5, help="measured seconds per priced day (--plan)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--inceptions", default="", help="comma-separated ISO dates (default: all)")
    ap.add_argument("--tails", default="", help="comma-separated tail ids (default: all six)")
    ap.add_argument("--no-parity", action="store_true", help="skip the day-0 V1 parity check")
    args = ap.parse_args()

    started = time.time()
    chains = load_chain()
    anchors = anchor_series(chains, ANCHOR_WINDOW)
    inceptions = inception_dates(chains, anchors)
    if args.inceptions:
        wanted = {date.fromisoformat(s.strip()) for s in args.inceptions.split(",")}
        missing = wanted - set(inceptions)
        if missing:
            raise SystemExit(f"not inception dates: {sorted(missing)}; inceptions are {inceptions}")
        inceptions = [d for d in inceptions if d in wanted]
    tails = tuple(t.strip() for t in args.tails.split(",")) if args.tails else TAILS
    unknown = set(tails) - set(TAILS)
    if unknown:
        raise SystemExit(f"unknown tails {sorted(unknown)}; expected {TAILS}")
    print(f"chain {chains[0].day} .. {chains[-1].day}; {len(inceptions)} inceptions "
          f"({inceptions[0]} .. {inceptions[-1]}); tails {', '.join(tails)}")

    if args.plan:
        return plan(chains, inceptions, len(tails), args.s_per_day, args.workers)

    if args.quick:
        inceptions = inceptions[:1]
        tails = QUICK_TAILS
        out_dir, max_days = QUICK_DIR, QUICK_DAYS
        if out_dir.exists():
            shutil.rmtree(out_dir)
        workers = min(args.workers, len(tails))
    else:
        out_dir, max_days, workers = RUNS_DIR, None, args.workers

    coupons = coupons_for(inceptions, chains, COUPONS_PATH)
    for inc in inceptions:
        print(f"  coupon {inc}: {coupons[inc]:.6f}")
    try:
        paths = run_all(inceptions, tails, out_dir, workers=workers, coupons=coupons, max_days=max_days,
                        parity=not args.no_parity)
    except RssLimitExceeded as err:
        print(f"ABORTED: {err}", file=sys.stderr)
        return 3
    if args.quick:
        print_quick(paths)
    flagged = [p.name for p in paths if (json.loads(p.read_text()).get("parity") or {}).get("flag")]
    print(f"\n{len(paths)} runs in {out_dir}; V1 parity flags: {flagged or 'none'}; "
          f"wall {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
