"""B05: is the "certain-V0" control certain under Heston?  Independent Heston MC of the knock-out premium.

The control's closed form (110 e^{-rT}) assumes the contract never knocks out. Under BSM (20% vol) that holds to ~1e-8.
Under Heston (v0 = theta = 0.04, kappa 1.5, sigma 0.3, rho -0.5) the left tail is fat: the reverse contract's KO at
20 (S down 80%) and the standard contract's KO at 500 may carry a real premium. This script estimates

    E[ (KO cash x DF(t_KO) - 110 x DF(T)) 1{KO} ]  =  true Heston price - closed form

with a standalone full-truncation log-Euler Heston MC (no library MC engine): 24 monthly KO dates, `sub` Euler
substeps per month, antithetic normals, fixed seeds. It prints the estimate, its standard error and P(KO), at two
substep counts (to show the time-discretization bias is inside the SE).
"""
import math
import sys

import numpy as np

S0, R, Q, T = 100.0, 0.03, 0.0, 2.0
V0, KAPPA, THETA, SIGMA, RHO = 0.04, 1.5, 0.04, 0.3, -0.5
N_OBS = 24
CF = 110.0 * math.exp(-R * T)


def premium(reverse, ko, n_paths, sub, seed):
    rng = np.random.default_rng(seed)
    dt = T / (N_OBS * sub)
    sq = math.sqrt(dt)
    acc, acc2, hits, done = 0.0, 0.0, 0, 0
    chunk = 200_000
    while done < n_paths:
        m = min(chunk, n_paths - done) // 2
        x = np.full(2 * m, math.log(S0))
        v = np.full(2 * m, V0)
        alive = np.ones(2 * m, dtype=bool)
        pay = np.zeros(2 * m)
        for i in range(1, N_OBS + 1):
            for _ in range(sub):
                z1 = rng.standard_normal(m)
                z2 = rng.standard_normal(m)
                z1 = np.concatenate([z1, -z1])
                z2 = np.concatenate([z2, -z2])
                zv = RHO * z1 + math.sqrt(1 - RHO * RHO) * z2
                vp = np.maximum(v, 0.0)
                x += (R - Q - 0.5 * vp) * dt + np.sqrt(vp) * sq * z1
                v = v + KAPPA * (THETA - vp) * dt + SIGMA * np.sqrt(vp) * sq * zv
            t = i * T / N_OBS
            s = np.exp(x)
            hit = alive & ((s <= ko) if reverse else (s >= ko))
            pay[hit] = 100.0 * (1.0 + 0.15 * t) * math.exp(-R * t) - CF
            alive &= ~hit
        # pair the antithetic halves (they are not independent)
        pair = 0.5 * (pay[:m] + pay[m:])
        acc += pair.sum()
        acc2 += (pair * pair).sum()
        hits += int((pay != 0).sum())
        done += 2 * m
    n_pairs = done // 2
    mean = acc / n_pairs
    var = acc2 / n_pairs - mean * mean
    return mean, math.sqrt(var / n_pairs), hits / done


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2_000_000
    for reverse, ko in ((True, 20.0), (False, 500.0), (True, 15.0)):
        for sub in (10, 40):
            m, se, pko = premium(reverse, ko, n, sub, seed=20261009 + sub)
            print(f"{'rev' if reverse else 'std'} KO={ko:5.1f} sub={sub:>2} paths={n}  true-cf = {m:+.4e} "
                  f"(SE {se:.1e})  P(KO)={pko:.2e}", flush=True)
