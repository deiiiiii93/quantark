"""D03 (W7b, ruling 2026-10-10): continuous KI on the production Greek policy path, after the fix -- the numbers behind
test_vol_pde_reverse_continuous_ki.py's slow matrix (which asserts them).

Per (orientation, model, KI): PV on the medium grid, calculate_greeks on its resolved (300, 135, 3200), the MC reference
difference in standard errors (reverse only; d01), and the PV spread over the n_x sweep 192 ... 208. Standard
orientation: KO 103, KI mirrored through the spot (100^2 / KI). SLV: unit leverage.
"""
import math

import test_vol_pde_reverse_continuous_ki as t

if __name__ == "__main__":
    for orientation in ("reverse", "standard"):
        for model in ("heston", "slv"):
            for ki in t.KI_LEVELS:
                product = t._production_product(orientation, ki)
                s = t._production_solver(model)
                pv = s.price(product, t._env())
                g = s.calculate_greeks(product, t._env())
                sweep = [t._production_solver(model, n_x).price(product, t._env()) for n_x in t.PV_SWEEP]
                ki_used = product.barrier_config.ki_barrier
                mc = ""
                if orientation == "reverse":
                    ref, se = t.MC_REFERENCE[ki]
                    mc = f"  PV - MC {(pv - ref) / se:+.2f} s.e., Greek price - MC {(float(g['price']) - ref) / se:+.2f} s.e."
                finite = math.isfinite(pv) and all(math.isfinite(float(v)) for v in g.values()) and all(
                    math.isfinite(v) for v in sweep)
                print(f"{orientation:<8} {model:<6} KI {ki_used:7.2f}  PV {pv:.6f}  Greeks price {float(g['price']):.6f} "
                      f"delta {float(g['delta']):+.6f} gamma {float(g['gamma']):+.6f}  n_x sweep spread "
                      f"{max(sweep) - min(sweep):.2e}  {'finite' if finite else 'NON-FINITE'}{mc}", flush=True)
