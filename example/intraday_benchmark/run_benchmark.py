"""Gate D: intraday latency and memory at declared accuracy (single trade, 101-point spot curve, 100-item batch).

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
        python example/intraday_benchmark/run_benchmark.py --engine quad_v2 --workload single --repeats 5

There is no SLA: the deliverable is the measured table next to the Gate C accuracy status of each state. Run every
engine x workload back-to-back in one window; never compare arms timed in different sessions.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "test"))

from intraday.conftest import SHANGHAI, dated_phoenix, dated_snowball, digital, flat_env  # noqa: E402
from intraday.controls import fixtures as C  # noqa: E402
from quantark.execution.manifest import platform_tag  # noqa: E402
from quantark.intraday import (EventKind, Fixing, resolve_context, spot_curve, value_intraday,  # noqa: E402
                               value_intraday_many)
from quantark.intraday.capability import capability_evidence  # noqa: E402
from quantark.intraday.request import IntradayValuationRequest  # noqa: E402

GREEKS = ("delta", "gamma", "vega", "rho", "theta")


def engine_factory(name: str, product_kind: str):
    """A fresh engine of the family for one product kind (None when the family has no intraday route for it)."""
    from quantark.asset.equity.engine.analytical import BarrierAnalyticalEngine, DigitalOptionAnalyticalEngine
    from quantark.asset.equity.engine.mc import BarrierOptionMCEngine, DigitalOptionMCEngine, PhoenixMCEngine, SnowballMCEngine
    from quantark.asset.equity.engine.pde import BarrierPDESolver, PhoenixPDESolver, SnowballPDESolver
    from quantark.asset.equity.engine.quad.v2 import PhoenixQuadEngineV2, SnowballQuadEngineV2
    from quantark.asset.equity.param import MCParams, PDEParams
    from quantark.util.enum.engine_enums import MonteCarloMethod

    def mc(cls, **kw):
        return cls(params=MCParams(num_paths=2 ** 14, seed=11), method=MonteCarloMethod.RANDOMIZED_QUASI, **kw)

    table = {
        "quad_v2": {"snowball": SnowballQuadEngineV2, "phoenix": PhoenixQuadEngineV2, "digital": DigitalOptionAnalyticalEngine,
                    "barrier": BarrierAnalyticalEngine},
        "pde": {"snowball": lambda: SnowballPDESolver(PDEParams()), "phoenix": lambda: PhoenixPDESolver(PDEParams()),
                "digital": DigitalOptionAnalyticalEngine, "barrier": lambda: BarrierPDESolver(PDEParams())},
        "mc_rqmc": {"snowball": lambda: mc(SnowballMCEngine), "phoenix": lambda: mc(PhoenixMCEngine),
                    "digital": lambda: mc(DigitalOptionMCEngine), "barrier": lambda: mc(BarrierOptionMCEngine, use_brownian_bridge=True)},
        "analytical": {"digital": DigitalOptionAnalyticalEngine, "barrier": BarrierAnalyticalEngine},
    }
    factory = table[name].get(product_kind)
    return None if factory is None else factory()


def _kos(profile):
    probe = resolve_context(IntradayValuationRequest(product=C.product("snowball_discrete_ki"),
                                                     pricing_env=flat_env(datetime(2026, 9, 1, tzinfo=SHANGHAI)),
                                                     session_calendar=C.sse(), variance_profile=profile))
    return [e for e in probe.timeline.events if e.kind is EventKind.KO]


def gate_c_status(cell_id: str) -> str:
    for row in capability_evidence().get("cells", ()):
        if row["cell"]["id"] == cell_id:
            return row["status"]
    return "n/a"


def timed(fn, repeats):
    samples, result = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), result


def peak_bytes(fn):
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def single(engine_name, repeats):
    rows = []
    profile = C.profile("desk")
    kos = _kos(profile)
    fixings = tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])
    if engine_name == "analytical":
        cases = [("digital", timedelta(hours=1)), ("digital", timedelta(seconds=1))]
    else:
        cases = [("snowball", timedelta(hours=1)), ("snowball", timedelta(seconds=1))]
    for kind, horizon in cases:
        if kind == "snowball":
            ts, product, fx, spot = kos[5].timestamp - horizon, dated_snowball(C.sse().calendar, C.T0), fixings, 102.5
            cell = f"snowball_discrete_ki-{engine_name}-desk-{int(horizon.total_seconds())}s-sd-1-ko"
        else:
            ts, product, fx, spot = datetime(2026, 9, 16, 15, 0, tzinfo=SHANGHAI) - horizon, digital(datetime(2026, 9, 16)), (), 100.5
            cell = f"digital-analytical-desk-{int(horizon.total_seconds())}s-sd+1-strike"
        for convention in ("point", "desk_bump"):
            def request():
                return IntradayValuationRequest(product=product, pricing_env=flat_env(ts, spot=spot), session_calendar=C.sse(),
                                                variance_profile=profile, fixings=fx, greeks=GREEKS, greek_convention=convention)
            warm_engine = engine_factory(engine_name, kind)
            cold, res = timed(lambda: value_intraday(engine_factory(engine_name, kind), request()), repeats)
            value_intraday(warm_engine, request())
            warm, _ = timed(lambda: value_intraday(warm_engine, request()), repeats)
            rows.append({"workload": "single", "engine": engine_name, "state": f"{kind} {horizon} {convention}",
                         "cold_s": cold, "warm_s": warm, "peak_bytes": peak_bytes(lambda: value_intraday(engine_factory(engine_name, kind), request())),
                         "engine_settings": repr(getattr(warm_engine, "params", None))[:200],
                         "greek_status": {g.name: g.status for g in res.greeks}, "accuracy": gate_c_status(cell)})
    return rows


def curve101(engine_name, repeats):
    profile = C.profile("desk")
    kos = _kos(profile)
    fixings = tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])
    rows = []
    kind = "digital" if engine_name == "analytical" else "snowball"
    spots = [95.0 + 0.11 * i for i in range(101)]                     # 95 .. 106 crosses the 103 KO barrier
    for label, horizon in (("1h before a fixing", timedelta(hours=1)), ("1s before a fixing (late state)", timedelta(seconds=1))):
        if kind == "snowball":
            req = IntradayValuationRequest(product=dated_snowball(C.sse().calendar, C.T0), pricing_env=flat_env(kos[5].timestamp - horizon),
                                           session_calendar=C.sse(), variance_profile=profile, fixings=fixings)
        else:
            req = IntradayValuationRequest(product=digital(datetime(2026, 9, 16)),
                                           pricing_env=flat_env(datetime(2026, 9, 16, 15, 0, tzinfo=SHANGHAI) - horizon),
                                           session_calendar=C.sse(), variance_profile=profile)
        prep_engine = engine_factory(engine_name, kind)
        t0 = time.perf_counter()
        spot_curve(prep_engine, req, spots)
        first = time.perf_counter() - t0
        warm, curve = timed(lambda: spot_curve(prep_engine, req, spots), repeats)
        rows.append({"workload": "curve101", "engine": engine_name, "state": f"{kind} {label}", "first_curve_s": first,
                     "warm_curve_s": warm, "peak_bytes": peak_bytes(lambda: spot_curve(engine_factory(engine_name, kind), req, spots)),
                     "points_with_greeks": sum(1 for p in curve if p.status == "ok"),
                     "undefined_points": sum(1 for p in curve if p.status == "undefined"), "accuracy": "n/a"})
    return rows


def batch100(engine_name, repeats):
    profiles = [C.profile(p) for p in C.PROFILES]
    kos = _kos(profiles[1])
    fixings = tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])
    kinds = ("snowball", "phoenix", "digital", "barrier")
    horizons = (timedelta(days=1), timedelta(hours=1), timedelta(minutes=5), timedelta(seconds=10))
    items = []
    for i in range(100):
        kind, profile, horizon = kinds[i % 4], profiles[i % 3], horizons[i % 4]
        engine = engine_factory(engine_name, kind)
        if engine is None:
            continue
        if kind == "snowball":
            product, ts, fx = dated_snowball(C.sse().calendar, C.T0), kos[5].timestamp - horizon, fixings
        elif kind == "phoenix":          # before its first coupon: no coupon replay needed
            product, ts, fx = dated_phoenix(C.sse().calendar, C.T0), datetime(2026, 4, 16, 15, 0, tzinfo=SHANGHAI) - horizon, ()
        elif kind == "digital":
            product, ts, fx = digital(datetime(2026, 9, 16)), datetime(2026, 9, 16, 15, 0, tzinfo=SHANGHAI) - horizon, ()
        else:
            product, ts, fx = C.product("barrier_uo_zero_carry"), datetime(2026, 9, 16, 15, 0, tzinfo=SHANGHAI) - horizon, ()
        r, q = (0.0, 0.0) if kind == "barrier" else (0.03, 0.01)
        items.append((engine, IntradayValuationRequest(product=product, pricing_env=flat_env(ts, spot=100.5, r=r, q=q),
                                                       session_calendar=C.sse(), variance_profile=profile, fixings=fx,
                                                       request_id=f"{kind}-{i}")))
    total, out = timed(lambda: value_intraday_many(items, collect_errors=True), repeats)
    per_item, per_kind = [], {}
    for engine, request in items:
        t0 = time.perf_counter()
        value_intraday_many([(engine, request)], collect_errors=True)
        elapsed = time.perf_counter() - t0
        per_item.append(elapsed)
        per_kind.setdefault(request.request_id.split("-")[0], []).append(elapsed)
    failures = [f"{o.item_id}: {o.error_type}" for o in out if o.__class__.__name__ == "IntradayFailure"]
    kinds = ", ".join(f"{k} {statistics.median(v) * 1e3:.1f} ms" for k, v in sorted(per_kind.items()))
    return [{"workload": "batch100", "engine": engine_name, "state": f"{len(items)} items ({kinds})", "total_s": total,
             "per_item_median_s": statistics.median(per_item), "failure_count": len(failures), "failures": failures,
             "peak_bytes": peak_bytes(lambda: value_intraday_many(items, collect_errors=True)), "accuracy": "n/a"}]


def markdown(rows) -> str:
    keys = [k for k in rows[0] if k not in ("greek_status", "engine_settings", "failures")]
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for row in rows:
        cells = []
        for k in keys:
            v = row[k]
            cells.append(f"{v:.4g}" if isinstance(v, float) else (f"{v / 2**20:.1f} MiB" if k == "peak_bytes" else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", choices=("quad_v2", "pde", "mc_rqmc", "analytical"), required=True)
    parser.add_argument("--workload", choices=("single", "curve101", "batch100"), required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", default=None, help="JSON output path (default: print only)")
    args = parser.parse_args(argv)
    rows = {"single": single, "curve101": curve101, "batch100": batch100}[args.workload](args.engine, args.repeats)
    header = {"platform": platform_tag(), "machine": platform.platform(), "threads_requested": args.threads,
              "thread_env": {k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
              "repeats": args.repeats, "timestamp": datetime.now().isoformat(timespec="seconds")}
    print(json.dumps(header))
    print(markdown(rows))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"header": header, "rows": rows}, handle, indent=1, default=str)


if __name__ == "__main__":
    main()
