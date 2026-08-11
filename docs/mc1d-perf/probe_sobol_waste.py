"""Is the desk's 100k QMC path count paying for 131k Sobol points?"""

import time

import numpy as np

from quantark.montecarlo.qmc_sobol import SobolNormalGenerator, _next_power_of_two

DIM = 488
for n in (65_536, 100_000, 131_072):
    gen = SobolNormalGenerator(base_seed=42)
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        z = gen.normal(n_paths=n, dim=DIM, batch_id=1)
        best = min(best, time.perf_counter() - t0)
    total = _next_power_of_two(n)
    waste = 100.0 * (total - n) / total
    print(f"  n_paths={n:>7}  generates {total:>7}  discards {total - n:>6} "
          f"({waste:4.1f}%)  time {best * 1e3:7.1f} ms  shape {z.shape}")
