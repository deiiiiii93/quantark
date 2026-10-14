"""E02: where each DCN level sits on the production log-spot grid, per n_x (no solve; grids only).

For each level b (KI 4500, coupon 4800, KO 6000 = spot, K_loss 6600) and each n_x:
- pre-pin: the fractional offset of ln b inside its cell of the unpinned sinh grid (0 = on the left node,
  0.5 = mid-cell) -- what the pinning has to correct;
- pinned: the node index, the log spacings h_l / h_r on either side and their ratio (pinning moves ONE node, so the
  neighbouring cells become uneven when the pre-pin offset is far from 0);
- the nodal-mask ownership of the barrier node (S = exp(ln b) round-trips 1 ULP off b) for the production masks
  KI: S <= b, coupon / KO: S >= b;
- the effective nodal threshold shift: a nodal mask puts the step at the dual-cell edge on the side the barrier node
  is NOT owned by, i.e. ln b +/- h/2. Reported in log units and as a fraction of the local spacing.
"""
import math

import numpy as np

from common import barriers, core_for, product

LADDER = (301, 401, 501, 601, 801, 1001, 1201, 1601, 2401, 3201, 4801)

if __name__ == "__main__":
    prod = product()
    lv = barriers(prod)
    # inclusive product comparisons: KI is S <= b, coupon / KO are S >= b
    owned_side = {"KI": "le", "CPN": "ge", "KO": "ge", "K_loss": None}
    for n_x in LADDER:
        c = core_for(n_x, 41, prod)
        cu = core_for(n_x, 41, prod, pin=False)
        X, S = c.X_grid, c.S_grid
        print(f"n_x {n_x}: x in [{X[0]:.4f}, {X[-1]:.4f}], S in [{S[0]:.4g}, {S[-1]:.4g}]")
        for name, b in lv.items():
            xb = math.log(b)
            Xu = cu.X_grid
            k = int(np.searchsorted(Xu, xb, side="right")) - 1
            frac = (xb - Xu[k]) / (Xu[k + 1] - Xu[k])
            i = int(np.argmin(np.abs(X - xb)))
            on = X[i] == xb
            h_l, h_r = X[i] - X[i - 1], X[i + 1] - X[i]
            s_node = S[i]
            rel = (s_node - b) / b
            side = owned_side[name]
            if side == "le":
                owned = s_node <= b
                # step between "breached" (node owned) and not: if owned, breach region extends to +h_r/2
                shift = +0.5 * h_r if owned else -0.5 * h_l
            elif side == "ge":
                owned = s_node >= b
                shift = -0.5 * h_l if owned else +0.5 * h_r
            else:
                owned, shift = None, None
            own_txt = "" if owned is None else (
                f" node {'OWNED' if owned else 'NOT owned'} by its mask, effective step at ln b {shift:+.5f}"
                f" ({shift / (0.5 * (h_l + h_r)):+.2f} h)")
            print(f"  {name:6s} b {b:9.3f}: pre-pin offset {frac:5.3f} of cell {Xu[k + 1] - Xu[k]:.5f};"
                  f" pinned node {i} ({'ON' if on else 'off'}), h_l {h_l:.5f} h_r {h_r:.5f} ratio {h_r / h_l:5.2f};"
                  f" S_node - b = {rel:+.1e} b{own_txt}")
