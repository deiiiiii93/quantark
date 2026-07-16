"""Phase 4 PDE preparation benchmark (spec section 20 gates 4, 5).

Not pytest-collected (no ``test_`` prefix). Run from the repo root:

    PYTHONPATH=$PWD .venv/bin/python test/execution/benchmark_phase4.py

Protocol per spec section 20: fixed request/plan, warm-up first, >= 5
post-warm-up repetitions, medians with IQR dispersion. Two mechanisms:

1. CRN repricing (gate 4 analog for PDE): 10 identical LV-snowball PDE
   repricings — direct loop (Dupire + grids + factorizations rebuilt every
   call) vs one warm session (artifacts reused). Spec target: >= 2x.
2. Framework overhead (gate 5): single European PDE dispatch, direct vs
   warm-session, cold engine caches. Spec budget: <= 3% median wall-time
   regression (measured here on a developer machine; the release gate runs
   on the scheduled controlled host).

Developer-machine snapshot (2026-07-16, this fixture size):
- CRN x10: 1.40x vs uncached direct. Attribution: the backward marches
  themselves dominate this small fixture's wall time (~70%) and must run
  per repricing; the session recovers essentially the entire build cost
  (Dupire + grids + coefficients + factorizations). The >= 2x claim is the
  production-sized controlled-host gate.
- Overhead: session dispatch was FASTER than direct (-9%) because session
  artifacts outlive the cleared class-level caches; per-dispatch framework
  cost is sub-0.1 ms either way.
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # test/
sys.path.insert(0, str(Path(__file__).resolve().parent))         # execution/

REPS = 5
CRN_CALLS = 10


def _median_iqr(times):
    med = statistics.median(times)
    qs = statistics.quantiles(times, n=4)
    return med, qs[2] - qs[0]


def bench_crn_lv_snowball():
    from execution.matrix_fixtures import FIXTURE_BUILDERS

    from quantark.execution import PricingRequest, PricingSession

    from quantark.asset.equity.engine.pde import LocalVolSnowballPDESolver

    engine, product, env, _ = FIXTURE_BUILDERS["LocalVolSnowballPDESolver"]()
    request = PricingRequest(product=product, pricing_env=env)

    def uncached_loop():
        # The spec gate-4 baseline: UNCACHED serial execution — a fresh
        # engine per repricing with cold class-level caches, as a service
        # without cross-request reuse would run.
        LocalVolSnowballPDESolver.clear_grid_cache()
        out = []
        for _ in range(CRN_CALLS):
            fresh = LocalVolSnowballPDESolver(params=engine.params)
            out.append(fresh.price(product, env))
        return out

    def warm_direct_loop():
        # Attribution comparator: one engine, its own internal caches warm.
        return [engine.price(product, env) for _ in range(CRN_CALLS)]

    def session_loop():
        with PricingSession() as session:
            return [session.execute(engine, request).value
                    for _ in range(CRN_CALLS)]

    uncached_loop(); warm_direct_loop(); session_loop()  # warm-up
    u_times, d_times, s_times = [], [], []
    for _ in range(REPS):
        t0 = time.perf_counter(); u_vals = uncached_loop()
        u_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); d_vals = warm_direct_loop()
        d_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); s_vals = session_loop()
        s_times.append(time.perf_counter() - t0)
        assert s_vals == d_vals == u_vals  # bitwise, every repetition
    u_med, u_iqr = _median_iqr(u_times)
    d_med, d_iqr = _median_iqr(d_times)
    s_med, s_iqr = _median_iqr(s_times)
    print(f"CRN x{CRN_CALLS} LV-snowball PDE:")
    print(f"  uncached direct median {u_med*1e3:9.1f} ms  IQR {u_iqr*1e3:7.1f} ms")
    print(f"  warm direct     median {d_med*1e3:9.1f} ms  IQR {d_iqr*1e3:7.1f} ms")
    print(f"  session         median {s_med*1e3:9.1f} ms  IQR {s_iqr*1e3:7.1f} ms")
    print(f"  speedup vs uncached {u_med/s_med:5.2f}x  (spec gate 4: >= 2x)")
    print(f"  speedup vs warm     {d_med/s_med:5.2f}x  (attribution)")


def bench_overhead_european():
    from execution.matrix_fixtures import FIXTURE_BUILDERS

    from quantark.asset.equity.engine.pde import EuropeanPDESolver

    from quantark.execution import PricingRequest, PricingSession

    engine, product, env, _ = FIXTURE_BUILDERS["EuropeanPDESolver"]()
    request = PricingRequest(product=product, pricing_env=env)

    def direct_once():
        EuropeanPDESolver.clear_grid_cache()
        return engine.price(product, env)

    session = PricingSession()  # dispatch overhead, not construction cost

    def session_once():
        EuropeanPDESolver.clear_grid_cache()
        return session.execute(engine, request).value

    direct_once(); session_once()  # warm-up (imports, registry, leases)
    d_times, s_times = [], []
    for _ in range(REPS):
        t0 = time.perf_counter(); dv = direct_once()
        d_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); sv = session_once()
        s_times.append(time.perf_counter() - t0)
        assert sv == dv
    session.close()
    d_med, d_iqr = _median_iqr(d_times)
    s_med, s_iqr = _median_iqr(s_times)
    print("single European PDE dispatch (cold engine caches, warm session):")
    print(f"  direct  median {d_med*1e3:9.2f} ms  IQR {d_iqr*1e3:7.2f} ms")
    print(f"  session median {s_med*1e3:9.2f} ms  IQR {s_iqr*1e3:7.2f} ms")
    print(f"  overhead {(s_med/d_med - 1.0)*100.0:+6.1f}% at a "
          f"{d_med*1e3:.1f} ms solve (spec budget: <= 3% on the "
          "controlled-host production-sized gate; absolute per-dispatch "
          f"cost {(s_med-d_med)*1e3:+.2f} ms)")


if __name__ == "__main__":
    bench_crn_lv_snowball()
    bench_overhead_european()
