"""WS-D7 acceptance artifact: RMSE-vs-paths for pseudo-MC vs Sobol QMC (flat LV).

QMC should show a steeper RMSE decay than pseudo-MC on a smooth European payoff.
"""
import numpy as np

from quantark.montecarlo.qmc_sobol import SobolNormalGenerator
from quantark.volmodels.localvol.mc_kernel import price_european_lv_mc
from quantark.volmodels.localvol.surface import LocalVolSurface
from quantark.volmodels.black_scholes import bs_call_price


def _flat_surface(vol=0.2):
    return LocalVolSurface(strike_grid=np.array([1.0, 1.0e6]),
                           time_grid=np.array([0.0, 100.0]),
                           lv_grid=np.full((2, 2), vol))


def rmse(sampler_factory, n, T, surface, analytic, batches=24):
    s0, k, r, q = 100.0, 100.0, 0.03, 0.0
    dt = np.full(4, T / 4); rf = np.full(4, r); cf = np.full(4, q); df = np.exp(-r * T)
    errs = []
    for b in range(batches):
        p = price_european_lv_mc(s0, k, True, surface, dt, rf, cf, df, num_paths=n,
                                 seed=1000 + b, sampler=sampler_factory(b))
        errs.append((p - analytic) ** 2)
    return np.sqrt(np.mean(errs))


def main():
    T, vol = 1.0, 0.2
    surface = _flat_surface(vol)
    analytic = bs_call_price(100.0, 100.0, T, vol, 0.03, 0.0)
    print("WS-D7  QMC vs pseudo RMSE-vs-paths (flat LV European, RMSE over 24 batches)")
    print(f"{'paths':>8} | {'pseudo RMSE':>12} {'sobol RMSE':>12} {'ratio':>7}")
    for n in [1024, 2048, 4096, 8192, 16384]:
        rp = rmse(lambda b: None, n, T, surface, analytic)
        rq = rmse(lambda b: SobolNormalGenerator(base_seed=1000 + b), n, T, surface, analytic)
        print(f"{n:8d} | {rp:12.5f} {rq:12.5f} {rp / max(rq, 1e-12):7.2f}")


if __name__ == "__main__":
    main()
