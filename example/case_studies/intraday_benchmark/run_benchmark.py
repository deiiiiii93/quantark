"""Intraday latency and memory: single trade, 101-point spot curve, 100-item batch.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \\
        python example/intraday_benchmark/run_benchmark.py --engine quad_v2 --workload single --repeats 5

There is no SLA: the deliverable is the measured table, with each Greek's status. Accuracy is certified offline by
the modelvalidation studies, not here. Run every engine x workload back-to-back in one window; never compare arms
timed in different sessions.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
import tracemalloc
from datetime import datetime, timedelta

from quantark.asset.equity.product.option.barrier_option import BarrierOption
from quantark.asset.equity.product.option.digital_option import CashOrNothingDigitalOption
from quantark.asset.equity.product.option.observation_schedule import ObservationRecord, ObservationSchedule
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.phoenix_option import PhoenixOption
from quantark.asset.equity.product.option.snowball_config import BarrierConfig, PayoffConfig
from quantark.execution.manifest import platform_tag
from quantark.intraday import EventKind, Fixing, resolve_context, spot_curve, value_intraday, value_intraday_many
from quantark.intraday.request import IntradayValuationRequest
from quantark.modelvalidation.builders.intraday_common import (SHANGHAI, exchange_calendar, make_environment,
                                                              make_snowball, session_calendar, variance_profile)
from quantark.util.enum.option_enums import BarrierType, ObservationType, OptionType

GREEKS = ("delta", "gamma", "vega", "rho", "theta")
T0 = datetime(2026, 3, 16)
PROFILES = ("uniform", "desk", "sessions_only")
#: The monthly-KI snowball: KO and KI observed on the twelve monthly dates.
MONTHLY_SNOWBALL = {"initial_date": "2026-03-16", "months": 12, "initial_price": 100.0, "strike": 100.0,
                    "ko_barrier": 103.0, "ki_barrier": 75.0, "ko_rate": 0.12, "rebate_rate": 0.12,
                    "contract_multiplier": 1.0, "ki_observation": "ko_dates"}


def sse():
    return session_calendar("SSE")


def profile(name: str):
    return variance_profile(name, sse())


def market(ts, spot=100.0, r=0.03, q=0.01):
    """A flat market quoted at ``ts``: vol 20%, rate ``r``, dividend yield ``q``."""
    return make_environment({"spot": spot, "vol": 0.20, "rate": r, "div_yield": q}, ts)


def snowball():
    return make_snowball(MONTHLY_SNOWBALL)


def digital(expiry, strike=100.0):
    return CashOrNothingDigitalOption(strike=strike, option_type=OptionType.CALL, payout=1.0, exercise_date=expiry)


def barrier_up_and_out():
    """Continuously monitored up-and-out call, strike 100, barrier 103, expiring on the sixth fixing day."""
    return BarrierOption(strike=100.0, option_type=OptionType.CALL, barrier=103.0, barrier_type=BarrierType.UP_OUT,
                         exercise_date=datetime(2026, 9, 16), observation_type=ObservationType.CONTINUOUS)


def phoenix():
    """Monthly KO/coupon dates with discrete KI on the same dates; memory coupons accrue ACT/365 per period."""
    cal = exchange_calendar("SSE")
    dates = [r.observation_date for r in snowball().barrier_config.ko_observation_schedule.records]
    records = lambda level: ObservationSchedule(records=[ObservationRecord(observation_date=d, barrier=level)  # noqa: E731
                                                         for d in dates])
    assert all(cal.is_business_day(d) for d in dates)
    return PhoenixOption(
        initial_price=100.0, strike=100.0, contract_multiplier=1.0, initial_date=T0, exercise_date=dates[-1],
        barrier_config=BarrierConfig(ko_barrier=103.0, ko_rate=0.0, ko_observation_type=ObservationType.DISCRETE,
                                     ko_observation_schedule=records(103.0), ki_barrier=75.0,
                                     ki_observation_type=ObservationType.DISCRETE, ki_observation_schedule=records(75.0)),
        coupon_config=CouponBarrierConfig(coupon_barrier=80.0, coupon_rate=0.12, memory_coupon=True),
        payoff_config=PayoffConfig(include_principal=True),
    )


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


def _kos(clock_profile):
    probe = resolve_context(IntradayValuationRequest(product=snowball(),
                                                     pricing_env=market(datetime(2026, 9, 1, tzinfo=SHANGHAI)),
                                                     session_calendar=sse(), variance_profile=clock_profile))
    return [e for e in probe.timeline.events if e.kind is EventKind.KO]


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
    desk = profile("desk")
    kos = _kos(desk)
    fixings = tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])
    if engine_name == "analytical":
        cases = [("digital", timedelta(hours=1)), ("digital", timedelta(seconds=1))]
    else:
        cases = [("snowball", timedelta(hours=1)), ("snowball", timedelta(seconds=1))]
    for kind, horizon in cases:
        if kind == "snowball":
            ts, product, fx, spot = kos[5].timestamp - horizon, snowball(), fixings, 102.5
        else:
            ts, product, fx, spot = datetime(2026, 9, 16, 15, 0, tzinfo=SHANGHAI) - horizon, digital(datetime(2026, 9, 16)), (), 100.5
        for convention in ("point", "desk_bump"):
            def request():
                return IntradayValuationRequest(product=product, pricing_env=market(ts, spot=spot), session_calendar=sse(),
                                                variance_profile=desk, fixings=fx, greeks=GREEKS, greek_convention=convention)
            warm_engine = engine_factory(engine_name, kind)
            cold, res = timed(lambda: value_intraday(engine_factory(engine_name, kind), request()), repeats)
            value_intraday(warm_engine, request())
            warm, _ = timed(lambda: value_intraday(warm_engine, request()), repeats)
            rows.append({"workload": "single", "engine": engine_name, "state": f"{kind} {horizon} {convention}",
                         "cold_s": cold, "warm_s": warm, "peak_bytes": peak_bytes(lambda: value_intraday(engine_factory(engine_name, kind), request())),
                         "engine_settings": repr(getattr(warm_engine, "params", None))[:200],
                         "greek_status": {g.name: g.status for g in res.greeks}})
    return rows


def curve101(engine_name, repeats):
    desk = profile("desk")
    kos = _kos(desk)
    fixings = tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])
    rows = []
    kind = "digital" if engine_name == "analytical" else "snowball"
    spots = [95.0 + 0.11 * i for i in range(101)]                     # 95 .. 106 crosses the 103 KO barrier
    for label, horizon in (("1h before a fixing", timedelta(hours=1)), ("1s before a fixing (late state)", timedelta(seconds=1))):
        if kind == "snowball":
            req = IntradayValuationRequest(product=snowball(), pricing_env=market(kos[5].timestamp - horizon),
                                           session_calendar=sse(), variance_profile=desk, fixings=fixings)
        else:
            req = IntradayValuationRequest(product=digital(datetime(2026, 9, 16)),
                                           pricing_env=market(datetime(2026, 9, 16, 15, 0, tzinfo=SHANGHAI) - horizon),
                                           session_calendar=sse(), variance_profile=desk)
        prep_engine = engine_factory(engine_name, kind)
        t0 = time.perf_counter()
        spot_curve(prep_engine, req, spots)
        first = time.perf_counter() - t0
        warm, curve = timed(lambda: spot_curve(prep_engine, req, spots), repeats)
        rows.append({"workload": "curve101", "engine": engine_name, "state": f"{kind} {label}", "first_curve_s": first,
                     "warm_curve_s": warm, "peak_bytes": peak_bytes(lambda: spot_curve(engine_factory(engine_name, kind), req, spots)),
                     "points_with_greeks": sum(1 for p in curve if p.status == "ok"),
                     "undefined_points": sum(1 for p in curve if p.status == "undefined")})
    return rows


def batch100(engine_name, repeats):
    profiles = [profile(p) for p in PROFILES]
    kos = _kos(profiles[1])
    fixings = tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])
    kinds = ("snowball", "phoenix", "digital", "barrier")
    horizons = (timedelta(days=1), timedelta(hours=1), timedelta(minutes=5), timedelta(seconds=10))
    items = []
    for i in range(100):
        kind, clock_profile, horizon = kinds[i % 4], profiles[i % 3], horizons[i % 4]
        engine = engine_factory(engine_name, kind)
        if engine is None:
            continue
        if kind == "snowball":
            product, ts, fx = snowball(), kos[5].timestamp - horizon, fixings
        elif kind == "phoenix":          # before its first coupon: no coupon replay needed
            product, ts, fx = phoenix(), datetime(2026, 4, 16, 15, 0, tzinfo=SHANGHAI) - horizon, ()
        elif kind == "digital":
            product, ts, fx = digital(datetime(2026, 9, 16)), datetime(2026, 9, 16, 15, 0, tzinfo=SHANGHAI) - horizon, ()
        else:
            product, ts, fx = barrier_up_and_out(), datetime(2026, 9, 16, 15, 0, tzinfo=SHANGHAI) - horizon, ()
        r, q = (0.0, 0.0) if kind == "barrier" else (0.03, 0.01)
        items.append((engine, IntradayValuationRequest(product=product, pricing_env=market(ts, spot=100.5, r=r, q=q),
                                                       session_calendar=sse(), variance_profile=clock_profile, fixings=fx,
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
             "peak_bytes": peak_bytes(lambda: value_intraday_many(items, collect_errors=True))}]


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
