"""WS-3 spike: fused single-pass path build + discrete KO/KI scan (Numba).

Gate (spec docs/superpowers/specs/2026-08-10-mc1d-perf-program-design.md WS-3):
primitives (first_ko_idx, ki_triggered, first_ki_idx, terminal) bit-identical
to the shipped pipeline on the same draws, AND >=1.5x on the fused stage at
100k x 252. Continuous-KI stays out of scope: its rng.random(idx.size) draws
are data-dependent and cannot be ported to numba bit-compatibly.

The hypothesis under test is the one the C3 failure implied: at production
path counts the cost is memory traffic, so a kernel that never materializes
the (n_paths, n_steps) path matrix should win where a full-matrix precompute
lost. The normals matrix is still materialized (bitwise constraint: identical
NumPy stream); only the paths matrix and the scan gathers are removed.

Run:  PYTHONPATH=$PWD <venv>/bin/python docs/mc1d-perf/demo_fused_snowball_kernel.py
"""

import time

import numpy as np
from numba import njit

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams
from quantark.montecarlo.gbm_kernels import gbm_path_tail


@njit(cache=True, fastmath=False)
def _fused_scan(dW, drift_dt, vol, s0, ko_cols, ko_barriers, ki_cols,
                ki_barriers, is_reverse, first_ko, first_ki, terminal):
    n_paths, n_steps = dW.shape
    n_ko = ko_cols.shape[0]
    n_ki = ki_cols.shape[0]
    for p in range(n_paths):
        c = 1.0
        s = s0
        fko = -1
        fki = -1
        iko = 0
        iki = 0
        for k in range(n_steps):
            c = c * np.exp(drift_dt[k] + vol[k] * dW[p, k])
            s = s0 * c
            if iko < n_ko and k == ko_cols[iko]:
                if fko < 0:
                    hit = (s <= ko_barriers[iko]) if is_reverse else (s >= ko_barriers[iko])
                    if hit:
                        fko = iko
                iko += 1
            if iki < n_ki and k == ki_cols[iki]:
                if fki < 0:
                    hit = (s >= ki_barriers[iki]) if is_reverse else (s <= ki_barriers[iki])
                    if hit:
                        fki = iki
                iki += 1
        first_ko[p] = fko
        first_ki[p] = fki
        terminal[p] = s


def shipped_stage(engine, dW, drift_dt, vol, s0, ko_cols, ko_barriers,
                  ki_cols, ki_barriers, is_reverse):
    """Path build (post-Phase-1 shipped path) + the engine's vectorized scans."""
    paths = gbm_path_tail(dW, drift_dt, vol, s0)
    _, first_ko = engine._check_ko_barriers(paths, ko_cols, ko_barriers, is_reverse)
    ki_trig, first_ki = engine._check_ki_barriers(
        paths, ki_cols, ki_barriers, is_reverse
    )
    return first_ko, ki_trig, first_ki, paths[:, -1]


def fused_stage(dW, drift_dt, vol, s0, ko_cols, ko_barriers, ki_cols,
                ki_barriers, is_reverse):
    n = dW.shape[0]
    first_ko = np.empty(n, dtype=np.int64)
    first_ki = np.empty(n, dtype=np.int64)
    terminal = np.empty(n, dtype=float)
    _fused_scan(np.ascontiguousarray(dW), drift_dt, vol, s0, ko_cols,
                ko_barriers, ki_cols, ki_barriers, is_reverse,
                first_ko, first_ki, terminal)
    ki_trig = first_ki >= 0
    return first_ko, ki_trig, first_ki, terminal


def main():
    n_paths, n_steps = 100_000, 252
    rng = np.random.default_rng(42)
    dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(1.0 / n_steps)
    vol = np.full(n_steps, 0.2)
    drift_dt = (0.03 - 0.5 * vol * vol) * (1.0 / n_steps)
    s0 = 100.0
    # Monthly KO on a 252-step grid; the shipped _check_* read paths[:, idx + 1],
    # and the kernel's node after step k is also index k + 1 -- same convention.
    ko_cols = (np.arange(1, 13) * 21 - 1).astype(np.int64)
    ko_barriers = np.full(12, 103.0)
    ki_cols = np.arange(n_steps, dtype=np.int64)   # daily discrete KI
    ki_barriers = np.full(n_steps, 75.0)
    engine = SnowballMCEngine(params=MCParams(num_paths=n_paths, seed=1))

    print("  primitives (shipped pipeline vs fused kernel, same draws):")
    ok = True
    for is_reverse in (False, True):
        a = shipped_stage(engine, dW, drift_dt, vol, s0, ko_cols, ko_barriers,
                          ki_cols, ki_barriers, is_reverse)
        b = fused_stage(dW, drift_dt, vol, s0, ko_cols, ko_barriers,
                        ki_cols, ki_barriers, is_reverse)
        for name, x, y in zip(("first_ko", "ki_trig", "first_ki", "terminal"), a, b):
            xa = np.asarray(x)
            ya = np.asarray(y).astype(xa.dtype)
            same = xa.tobytes() == ya.tobytes()
            ok &= same
            tag = "IDENTICAL" if same else f"MISMATCH ({int(np.sum(xa != ya))} of {xa.size})"
            print(f"    reverse={int(is_reverse)} {name:9}: {tag}")

    t_ship = t_fused = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        shipped_stage(engine, dW, drift_dt, vol, s0, ko_cols, ko_barriers,
                      ki_cols, ki_barriers, False)
        t_ship = min(t_ship, time.perf_counter() - t0)
        t0 = time.perf_counter()
        fused_stage(dW, drift_dt, vol, s0, ko_cols, ko_barriers,
                    ki_cols, ki_barriers, False)
        t_fused = min(t_fused, time.perf_counter() - t0)
    speedup = t_ship / t_fused
    print(f"\n  stage (100k x 252, best of 5): shipped {t_ship * 1e3:7.1f} ms   "
          f"fused {t_fused * 1e3:7.1f} ms   speedup {speedup:.2f}x")
    gate = ok and speedup >= 1.5
    print(f"\nVERDICT: bitwise={'yes' if ok else 'NO'}, speedup={speedup:.2f}x, "
          f"gate(bitwise AND >=1.5x)={'PASS' if gate else 'FAIL'}")


if __name__ == "__main__":
    # warm the JIT outside the timers
    fused_stage(np.zeros((4, 8)), np.zeros(8), np.full(8, 0.2), 100.0,
                np.array([3], dtype=np.int64), np.array([103.0]),
                np.array([1], dtype=np.int64), np.array([75.0]), False)
    main()
