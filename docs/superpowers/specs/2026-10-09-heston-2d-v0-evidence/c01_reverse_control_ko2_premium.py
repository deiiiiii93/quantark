"""C01 (Q5): is the reverse certain-V0 control with the knock-out at 2 (S down 98%) certain under Heston?

The same independent full-truncation log-Euler Heston MC as ``b05`` (its ``premium`` function, unchanged: no library
MC engine), W3's Heston and market (v0 = theta = 0.04, kappa 1.5, sigma 0.3, rho -0.5; r = 0.03, q = 0, T = 2; 24
monthly KO dates; KO cash 100 (1 + 0.15 t)). It prints  true - closed form  for the reverse contract at KO 2, with
KO 10 and 5 for the trend (KO 20 / 15 are b05's).

With no knock-out on N paths the estimate is 0 with a zero standard error, which is not a bound. The script states
the bound it relies on: P(KO) <= 3 / N at 95% confidence (the rule of three) and |KO cash x DF(t) - 110 DF(T)| <=
max_t |100 (1 + 0.15 t) e^{-rt} - 110 e^{-rT}| = 18.83 (t = 2), so |premium| <= 18.83 x 3 / N.

Control test tolerance (test_snowball_pde_v0_rows.py::TestHeston2DCertainV0Control): 1e-6.
Usage: python c01_reverse_control_ko2_premium.py [n_paths_ko2]   (default 100,000,000)
"""
import math
import sys

from b05_heston_mc_reference import CF, R, T, premium

TOL = 1e-6
MAX_PAY = max(abs(100.0 * (1.0 + 0.15 * (i / 24 * T)) * math.exp(-R * i / 24 * T) - CF) for i in range(1, 25))

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000_000
    print(f"max |KO cash x DF - cf| over the 24 dates = {MAX_PAY:.4f}", flush=True)
    for ko, paths in ((10.0, 20_000_000), (5.0, 20_000_000)):
        m, se, pko = premium(True, ko, paths, 10, seed=20261010 + int(ko))
        print(f"rev KO={ko:5.1f} sub=10 paths={paths}  true-cf = {m:+.4e} (SE {se:.1e})  P(KO)={pko:.2e}", flush=True)
    for sub in (10, 40):
        m, se, pko = premium(True, 2.0, n, sub, seed=20261010 + sub)
        bound = MAX_PAY * 3.0 / n if pko == 0.0 else float("nan")
        print(f"rev KO=  2.0 sub={sub:>2} paths={n}  true-cf = {m:+.4e} (SE {se:.1e})  P(KO)={pko:.2e}  "
              f"95% bound with no knock-out {bound:.2e}  {'BELOW' if bound < TOL else 'NOT below'} tol {TOL:.0e}",
              flush=True)
