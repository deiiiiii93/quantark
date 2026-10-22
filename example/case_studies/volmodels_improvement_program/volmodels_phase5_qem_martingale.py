"""WS-C5 acceptance artifact: QE vs QE-M martingale bias across regimes.

Reports discounted E[S_T]/fwd - 1 in stderr units. QUADEXP shows a measurable
martingale bias at coarse steps / high vol-of-vol; QUADEXP_M's exact K0* removes it.
"""
import numpy as np

from quantark.util.enum.engine_enums import HestonMCScheme
from quantark.volmodels.heston.params import HestonParams
from quantark.volmodels.heston.mc_kernel import price_european_heston_mc


def martingale_bias(scheme, sigma, rho, v0, theta, steps, num_paths=500_000, seed=7):
    s0, T = 100.0, 1.0
    params = HestonParams(kappa=1.0, theta=theta, sigma=sigma, rho=rho, v0=v0)
    dt = np.full(steps, T / steps)
    price, se = price_european_heston_mc(
        s0, 1e-6, True, params, dt, np.zeros(steps), np.zeros(steps), 1.0,
        scheme=scheme, num_paths=num_paths, seed=seed, return_stderr=True,
    )
    return (price - s0), se


def main():
    print("WS-C5  QE-M martingale correction — discounted E[S_T] - s0 (in stderr units)")
    print(f"{'sigma':>5} {'rho':>5} {'v0':>5} {'steps':>5} | {'QE bias(se)':>14} {'QE-M bias(se)':>14}")
    for sigma, rho, v0, theta, steps in [
        (0.5, -0.9, 0.09, 0.09, 8), (1.0, -0.9, 0.09, 0.09, 8),
        (1.0, -0.9, 0.16, 0.16, 2), (1.5, -0.9, 0.16, 0.16, 1),
        (2.0, -0.9, 0.25, 0.25, 2), (1.0, -0.5, 0.09, 0.09, 4),
    ]:
        bq, seq = martingale_bias(HestonMCScheme.QUADEXP, sigma, rho, v0, theta, steps)
        bm, sem = martingale_bias(HestonMCScheme.QUADEXP_M, sigma, rho, v0, theta, steps)
        print(f"{sigma:5.1f} {rho:5.1f} {v0:5.2f} {steps:5d} | "
              f"{bq:+7.4f}({bq/seq:+5.1f}) {bm:+7.4f}({bm/sem:+5.1f})")


if __name__ == "__main__":
    main()
