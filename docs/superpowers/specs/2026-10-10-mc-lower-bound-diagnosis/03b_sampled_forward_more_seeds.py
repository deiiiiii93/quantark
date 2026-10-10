"""Follow-up to 03: the sampled forward with 1 step read -3.0e-7 +- 1.2e-7 (t = -2.5) on 8 seeds (every row of 03
reuses the same 8 seeds, so the 32 rows are ONE draw, not 32). Re-measure on 256 independent seeds, 1 and 100 steps,
pseudo antithetic and plain, T = 1d and 1y. The deterministic forward (03) is exact to 4e-16, so a real bias here
could only come from the random stream itself."""
import math

import numpy as np

from _common import make_engine, make_env, make_option, timing_of
from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs

R = 256
print(f"{'case':22s} {'method':11s} {'steps':>5s} {'paths':>6s} {'R':>4s} {'mean(F_mc/F-1)':>15s} {'se':>9s} {'t':>6s}")
for days in (1, 365):
    T = days / 365
    env = make_env(90.0)
    prod = make_option(T)
    tm = timing_of(prod, env)
    q = env.get_div_yield(T)
    F = 90.0 * math.exp(-q * T) / tm.determination_df
    for method in ("antithetic", "plain"):
        for steps in (1, 100):
            n = 16384
            term = build_mc_term_inputs(env, ref_strike=100.0, maturity=T, time_steps=steps)
            rel = []
            for s in range(1000, 1000 + R):
                g = make_engine(method, n, s, steps)._create_path_generator(90.0, env.get_rate(T), q, 0.2, T,
                                                                             term=term)
                paths, _ = g.generate_paths()
                rel.append(paths[:, -1].mean() / F - 1.0)
            rel = np.asarray(rel)
            se = rel.std(ddof=1) / math.sqrt(R)
            print(f"T={days}d{'':17s} {method:11s} {steps:5d} {n:6d} {R:4d} {rel.mean():+15.3e} {se:9.2e} "
                  f"{rel.mean() / se:+6.2f}", flush=True)
