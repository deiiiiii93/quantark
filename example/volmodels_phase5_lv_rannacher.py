"""WS-C7 acceptance artifact: LV PDE Rannacher + strike mid-cell grid.

Shows (a) the default (mid-cell + Rannacher) gives oscillation-free near-strike gamma
(min gamma >= 0), and (b) the deliberate golden move on a term-structure LV surface is
CLOSER to the fine-grid reference than the pre-WS-C7 value.
"""
import numpy as np

from quantark.volmodels.localvol.pde_kernel import price_european_lv_pde, _solve_lv_pde
from quantark.volmodels.localvol.surface import LocalVolSurface


def _flat(vol=0.2):
    return LocalVolSurface(strike_grid=np.array([1.0, 1.0e6]),
                           time_grid=np.array([0.0, 100.0]), lv_grid=np.full((2, 2), vol))


def main():
    print("WS-C7  LV Rannacher + strike mid-cell grid")
    print("\n(a) near-strike min-gamma (>=0 = oscillation-free), default (mid-cell+Rannacher):")
    surf = _flat(0.2)
    for T, nt in [(0.05, 3), (0.02, 2), (0.1, 4), (0.5, 8)]:
        dt = np.full(nt, T / nt)
        sg, v = _solve_lv_pde(100.0, 100.0, True, T, surf, dt, np.zeros(nt), np.zeros(nt), n_s=201)
        g = np.gradient(np.gradient(v, sg), sg)
        w = (sg > 0.85 * 100) & (sg < 1.15 * 100)
        print(f"   T={T:5.2f} nt={nt}: min-gamma near strike = {g[w].min():+.3e}")

    print("\n(b) deliberate golden move vs fine-grid reference (term-structure LV):")
    surf2 = LocalVolSurface(strike_grid=np.array([50.0, 100.0, 200.0]),
                            time_grid=np.array([0.0, 1.0]),
                            lv_grid=np.array([[0.30, 0.22, 0.20], [0.32, 0.24, 0.21]]))
    dt = np.full(50, 0.02); rf = np.full(50, 0.03); cf = np.full(50, 0.01)
    new = price_european_lv_pde(100.0, 105.0, True, 1.0, surf2, dt, rf, cf, n_s=200)
    dtf = np.full(400, 1.0 / 400)
    ref = price_european_lv_pde(100.0, 105.0, True, 1.0, surf2, dtf,
                                np.full(400, 0.03), np.full(400, 0.01), n_s=3000)
    old = 7.931287902952514
    print(f"   pre-WS-C7 pin = {old:.6f}  err vs ref = {abs(old - ref):.3e}")
    print(f"   new default   = {new:.6f}  err vs ref = {abs(new - ref):.3e}")
    print(f"   fine-grid ref = {ref:.6f}   -> new is {'CLOSER' if abs(new-ref)<abs(old-ref) else 'FARTHER'}")


if __name__ == "__main__":
    main()
