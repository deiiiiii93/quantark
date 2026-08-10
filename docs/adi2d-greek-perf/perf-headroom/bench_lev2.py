"""Bitwise-preserving hoist: same arithmetic order, 1-D gathers, scalar time lookup."""
import time, sys
import numpy as np
sys.path.insert(0, 'docs/mc-reference-convergence')
from demo_common import cert

surf = cert.make_leverage_surface(3.0)
rng = np.random.default_rng(5)

def hoisted(surface, spot, t):
    s = np.asarray(spot, dtype=float)
    ln_s = np.log(np.clip(s, surface.strike_grid[0], surface.strike_grid[-1]))
    K = surface._ln_k
    jK = np.clip(np.searchsorted(K, ln_s, side="right"), 1, K.size - 1)
    j0, j1 = jK - 1, jK
    wK = (ln_s - K[j0]) / (K[j1] - K[j0])
    g = surface.leverage_grid
    if surface.time_grid.size == 1:
        return g[0, j0] * (1 - wK) + g[0, j1] * wK
    Tg = surface.time_grid
    tc = min(max(float(t), Tg[0]), Tg[-1])
    iT = int(np.clip(np.searchsorted(Tg, tc, side="right"), 1, Tg.size - 1))
    wT = (tc - Tg[iT - 1]) / (Tg[iT] - Tg[iT - 1])
    r0, r1 = g[iT - 1], g[iT]                     # 1-D views, no 2-D fancy index
    bot = r0[j0] * (1 - wK) + r0[j1] * wK          # SAME order as shipped
    top = r1[j0] * (1 - wK) + r1[j1] * wK
    return bot * (1 - wT) + top * wT

for n in (1024, 8192):
    sp = 100.0 * np.exp(rng.normal(0, 0.25, n))
    t = 1.234
    a, b = surf.leverage(sp, t), hoisted(surf, sp, t)
    def bench(f, reps=300):
        best = 1e9
        for _ in range(reps):
            s = time.perf_counter(); f(); best = min(best, time.perf_counter()-s)
        return best
    t0 = bench(lambda: surf.leverage(sp, t)); t1 = bench(lambda: hoisted(surf, sp, t))
    print(f"n_paths={n:<6} shipped {t0*1e6:7.1f} us  hoisted {t1*1e6:7.1f} us  "
          f"speedup {t0/t1:4.2f}x  BITWISE={np.array_equal(a,b)}")
