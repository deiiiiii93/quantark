"""Candidate 1: scalar-time fast path for LocalVolSurface.local_vol.

MC kernels call ``local_vol(spot_vector, t_scalar)`` once per step. The shipped
lookup broadcasts the scalar t across all paths, runs an n_paths-long
searchsorted over identical values, and gathers the grid with four 2-D
fancy-index reads — the same pathology the SLV leverage lookup had before
b98a8d9 (measured there: 47% of MC time, fast path 1.79-1.99x bitwise).

Here the share is larger: local_vol is 82% of an LV European run and 54% of an
LV barrier run (docs/mc1d-perf/prof_baseline.py, 2026-08-10).

The fast path computes the time bracket once and uses 1-D row views, keeping
the SAME arithmetic order (strike interpolation first, then the time blend) —
the 2D program proved that blending time rows before the strike interp is NOT
bitwise (4.4e-16). Everything is runtime-patched; no engine code changes.

Run:  PYTHONPATH=$PWD <venv>/bin/python docs/mc1d-perf/demo_lv_scalar_t.py
"""

import time

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.volmodels.localvol.surface import LocalVolSurface
from quantark.volmodels.localvol import mc_kernel


SHIPPED_LOCAL_VOL = LocalVolSurface.local_vol


def local_vol_fast(self, spot, t):
    """Shipped lookup plus a scalar-t fast path (candidate implementation)."""
    s = np.asarray(spot, dtype=float)
    tt = np.asarray(t, dtype=float)
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(tt))):
        raise ValidationError("spot and t must be finite")
    if tt.ndim != 0:
        return SHIPPED_LOCAL_VOL(self, spot, t)

    shape = s.shape
    K = self.strike_grid
    s_flat = np.clip(s.ravel(), K[0], K[-1])
    jK = np.clip(np.searchsorted(K, s_flat, side="right"), 1, K.size - 1)
    j0, j1 = jK - 1, jK
    if self.interp == "linear_logs":
        lnK = np.log(K)
        wK = (np.log(s_flat) - lnK[j0]) / (lnK[j1] - lnK[j0])
    else:
        wK = (s_flat - K[j0]) / (K[j1] - K[j0])

    g = self.lv_grid
    if self.time_grid.size == 1:
        row = g[0]
        vals = row[j0] * (1.0 - wK) + row[j1] * wK
    else:
        Tg = self.time_grid
        t_val = float(np.clip(tt, Tg[0], Tg[-1]))
        iT = int(np.clip(np.searchsorted(Tg, t_val, side="right"), 1, Tg.size - 1))
        i0, i1 = iT - 1, iT
        wT = (t_val - Tg[i0]) / (Tg[i1] - Tg[i0])
        g0, g1 = g[i0], g[i1]  # 1-D row views, not 2-D fancy gathers
        # Same arithmetic ORDER as shipped: strike interp per row, then time blend.
        bottom = g0[j0] * (1.0 - wK) + g0[j1] * wK
        top = g1[j0] * (1.0 - wK) + g1[j1] * wK
        vals = bottom * (1.0 - wT) + top * wT

    result = np.asarray(vals, dtype=float).reshape(shape)
    return result if result.shape else float(result)


def smiled_surface(interp="linear_s"):
    t_grid = np.linspace(0.0, 2.0, 25)
    k_grid = np.exp(np.linspace(np.log(50.0), np.log(200.0), 61))
    logm = np.log(k_grid / 100.0)
    smile = 0.20 + 0.15 * logm**2 - 0.05 * logm
    term = 1.0 + 0.1 * np.sqrt(np.maximum(t_grid, 0.0))[:, None]
    grid = np.clip(smile[None, :] * term, 0.05, 1.5)
    return LocalVolSurface(k_grid, t_grid, grid, interp=interp)


def check_bitwise_lookup():
    """Lookup outputs must be byte-identical across shapes, edges, and interps."""
    rng = np.random.default_rng(7)
    surfaces = {
        "linear_s": smiled_surface("linear_s"),
        "linear_logs": smiled_surface("linear_logs"),
        "single_time": LocalVolSurface(
            np.array([50.0, 100.0, 200.0]), np.array([1.0]),
            np.array([[0.3, 0.2, 0.25]]),
        ),
    }
    t_cases = [0.0, 1e-9, 0.5, 0.083333333, 1.999, 2.0, 5.0, -1.0]  # incl. off-grid clamps
    n_fail = 0
    for name, lv in surfaces.items():
        for t in t_cases:
            for n in (1, 7, 1024, 100_000):
                spots = np.exp(rng.normal(np.log(100.0), 0.6, size=n))
                spots[0] = 10.0   # below strike clamp
                if n > 1:
                    spots[1] = 500.0  # above strike clamp
                a = SHIPPED_LOCAL_VOL(lv, spots, t)
                b = local_vol_fast(lv, spots, t)
                if np.asarray(a).tobytes() != np.asarray(b).tobytes():
                    n_fail += 1
                    print(f"  MISMATCH {name} t={t} n={n} "
                          f"max|d|={np.max(np.abs(np.asarray(a) - np.asarray(b)))}")
        # scalar spot + scalar t must keep returning a float
        fa = SHIPPED_LOCAL_VOL(lv, 100.0, 0.7)
        fb = local_vol_fast(lv, 100.0, 0.7)
        assert isinstance(fb, float) and fa == fb and type(fa) is type(fb)
        # vector t must still route through the shipped path untouched
        tv = np.array([0.1, 0.5, 1.0])
        sv = np.array([90.0, 100.0, 110.0])
        assert SHIPPED_LOCAL_VOL(lv, sv, tv).tobytes() == local_vol_fast(lv, sv, tv).tobytes()
    return n_fail


def bench_lookup(lv):
    print("\n  lookup microbench (252 sequential calls, one per step):")
    for n in (1024, 8192, 100_000):
        rng = np.random.default_rng(3)
        spots = np.exp(rng.normal(np.log(100.0), 0.3, size=n))
        reps = 5
        t_ship = t_fast = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            for k in range(252):
                SHIPPED_LOCAL_VOL(lv, spots, k / 252.0)
            t_ship = min(t_ship, time.perf_counter() - t0)
            t0 = time.perf_counter()
            for k in range(252):
                local_vol_fast(lv, spots, k / 252.0)
            t_fast = min(t_fast, time.perf_counter() - t0)
        print(f"    n_paths={n:>7}: shipped {t_ship * 1e3:8.2f} ms   "
              f"fast {t_fast * 1e3:8.2f} ms   speedup {t_ship / t_fast:5.2f}x")


def bench_end_to_end(lv):
    """Price with both lookups; prices must be bit-equal; report wall times."""
    n = 252
    common = dict(
        s0=100.0, strike=100.0, is_call=True, lv_surface=lv,
        step_dt=np.full(n, 1.0 / n), r_fwd=np.full(n, 0.05),
        carry_fwd=np.full(n, 0.02), disc_factor=float(np.exp(-0.05)),
        num_paths=100_000, seed=42,
    )
    results = {}
    print("\n  end-to-end kernels (100k paths x 252 steps, best of 3):")
    for label, impl in (("shipped", SHIPPED_LOCAL_VOL), ("fast", local_vol_fast)):
        LocalVolSurface.local_vol = impl
        try:
            best = float("inf")
            for _ in range(3):
                t0 = time.perf_counter()
                p_euro = mc_kernel.price_european_lv_mc(**common)
                dt_run = time.perf_counter() - t0
                best = min(best, dt_run)
            t0 = time.perf_counter()
            p_bar = mc_kernel.price_barrier_lv_mc(
                **common, barrier=130.0, is_up=True, is_out=True, rebate=1.0,
                continuous=True,
            )
            t_bar = time.perf_counter() - t0
        finally:
            LocalVolSurface.local_vol = SHIPPED_LOCAL_VOL
        results[label] = (p_euro, best, p_bar, t_bar)
        print(f"    {label:>7}: euro {best:7.3f}s  barrier {t_bar:7.3f}s   "
              f"(prices {p_euro:.17g} / {p_bar:.17g})")
    (pe_a, te_a, pb_a, tb_a), (pe_b, te_b, pb_b, tb_b) = results["shipped"], results["fast"]
    bit_equal = (pe_a == pe_b and pe_a.hex() == pe_b.hex()
                 and pb_a == pb_b and pb_a.hex() == pb_b.hex())
    print(f"    euro speedup {te_a / te_b:.2f}x, barrier speedup {tb_a / tb_b:.2f}x, "
          f"prices bit-equal: {bit_equal}")
    return bit_equal


if __name__ == "__main__":
    print("Candidate 1: LocalVolSurface.local_vol scalar-t fast path")
    fails = check_bitwise_lookup()
    print(f"  lookup bitwise sweep: {'PASS' if fails == 0 else f'{fails} FAILURES'}")
    lv = smiled_surface()
    bench_lookup(lv)
    ok = bench_end_to_end(lv)
    print(f"\nVERDICT: bitwise={'yes' if fails == 0 and ok else 'NO'}")
