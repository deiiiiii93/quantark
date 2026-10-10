"""Step 2: isolate the simulated forward, the discount factors and the bound algebra (deterministic, no noise).

(a) The engine's generator is exact log-Euler GBM: with Gaussian increments, E[S_T] = S * exp(sum_k (rrf_k - div_k) dt_k)
    EXACTLY (the -0.5 sigma^2 dt and E[exp(sigma dW)] cancel step by step). So the simulated forward is a
    deterministic function of the generator's drift vector; it is compared with the curve forward
    S * exp(-qT) / P(T) that the bound uses. A sampled check with many antithetic paths is printed alongside.
(b) payment_df / determination_df / delay_df vs exp(-r(T) T) and the MC term grid's node DF.
(c) contract multiplier.
(d) the closed form vs the bound, with and without a settlement lag: the bound is payment_df * (K - F_T) for a put
    exactly when F_T = S exp(-qT) / determination_df.
"""
import math

import numpy as np

from _common import bs_price, make_engine, make_env, make_option, timing_of
from quantark.asset.equity.engine.mc.term_inputs import build_mc_term_inputs
from quantark.util.enum import OptionType

print(f"{'case':30s} {'steps':>5s} {'F_engine/F_curve-1':>19s} {'sampled_F/F_curve-1 (+-se)':>30s} "
      f"{'pay_df':>16s} {'exp(-r(T)T)':>16s} {'node_df[-1]':>16s} {'delay_df':>16s} {'mult':>4s}")
for term_curve in (False, True):
    for days in (1, 5, 30, 365):
        for lag in (0, 2):
            for steps in (1, 100):
                T = days / 365
                env = make_env(90.0, term_curve=term_curve)
                prod = make_option(T, lag_days=lag)
                tm = timing_of(prod, env)
                eng = make_engine("antithetic", 65536, 11, steps)
                q = env.get_div_yield(T)
                term = build_mc_term_inputs(env, ref_strike=100.0, maturity=T, time_steps=steps)
                gen = eng._create_path_generator(90.0, env.get_rate(T), q, 0.2, T, term=term)
                F_engine = 90.0 * math.exp(float(np.sum(gen._drift_vec * gen.dt_vector)))
                F_curve = 90.0 * math.exp(-q * T) / tm.determination_df
                # sampled forward, 64 x 65536 antithetic paths
                means = []
                for s in range(1, 9):
                    g = make_engine("antithetic", 65536, s, steps)._create_path_generator(
                        90.0, env.get_rate(T), q, 0.2, T, term=term)
                    paths, _ = g.generate_paths()
                    means.append(paths[:, -1].mean())
                means = np.asarray(means)
                rel = means / F_curve - 1.0
                name = f"{'term' if term_curve else 'flat'} T={days}d lag{lag}d"
                print(f"{name:30s} {steps:5d} {F_engine / F_curve - 1.0:+19.3e} "
                      f"{rel.mean():+14.3e} +- {rel.std(ddof=1) / math.sqrt(len(rel)):9.2e}    "
                      f"{tm.payment_df:16.13f} {math.exp(-env.get_rate(T) * T):16.13f} {term.node_dfs[-1]:16.13f} "
                      f"{tm.delay_df:16.13f} {prod.contract_multiplier:4g}", flush=True)

print("\n(d) closed form against the bound (the engine's own _european_lower_bound), put K=100 and call K=100")
print(f"{'case':36s} {'BS':>18s} {'bound':>18s} {'BS-bound':>11s} {'parity: P(T+L)*(K-F)':>22s} "
      f"{'bound-parity':>12s}")
for term_curve in (False, True):
    for opt, spot in ((OptionType.PUT, 90.0), (OptionType.PUT, 75.0), (OptionType.CALL, 110.0),
                      (OptionType.PUT, 100.0)):
        for days in (1, 5, 30, 365):
            for lag in (0, 2):
                T = days / 365
                env = make_env(spot, term_curve=term_curve)
                prod = make_option(T, opt, lag_days=lag)
                tm = timing_of(prod, env)
                eng = make_engine()
                q = env.get_div_yield(T)
                bound = eng._european_lower_bound(prod, spot, 100.0, T, q, tm.payment_df, tm.delay_df)
                bs = bs_price(prod, env)
                F = spot * math.exp(-q * T) / tm.determination_df
                parity = tm.payment_df * ((100.0 - F) if opt == OptionType.PUT else (F - 100.0))
                name = f"{'term' if term_curve else 'flat'} {opt.name.lower()} S{spot:g} T={days}d lag{lag}d"
                print(f"{name:36s} {bs:18.12f} {bound:18.12f} {bs - bound:+11.3e} {parity:22.12f} "
                      f"{bound - max(parity, 0.0):+12.3e}")
