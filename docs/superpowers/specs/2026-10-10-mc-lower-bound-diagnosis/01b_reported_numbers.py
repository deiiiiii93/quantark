"""Try to reproduce the brief's figures (9.99399 at 256 antithetic; 9.99415 at 16384 QMC antithetic) on the W13
contract with the plausible seeds / step counts, the bound check off. QMC drops the antithetic flag in the engine."""
from _common import make_env, make_option, mc_raw, bs_price

env, prod = make_env(90.0), make_option(1 / 365)
print(f"BS = {bs_price(prod, env):.8f}")
for method, n in (("antithetic", 256), ("qmc+antithetic", 16384), ("qmc", 16384)):
    for seed in (7, 42):
        for steps in (1, 100, 252):
            p, se, b, _ = mc_raw(prod, env, method=method, num_paths=n, seed=seed, time_steps=steps)
            print(f"{method:15s} n={n:6d} seed={seed:3d} steps={steps:4d}: price={p:.8f} bound={b:.8f} "
                  f"price-bound={p - b:+.3e} eng_se={se:.2e}")
