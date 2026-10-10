"""D02 (W7b): what the W7b fix moves. Before / after back to back in one process, compared as float.hex().

"Before" patches ``_ContinuousKIFirstPassage._geometry`` with ``legacy_geometry`` (test/test_vol_pde_reverse_
continuous_ki.py: the pre-W7b form, verbatim); "after" is the library. Run before the fix it must report 0 moved (the
legacy copy is faithful); after it, every moved value. Rows: c03's scan (reverse KO 97 / continuous KI 125 at (n_x, 48,
192), n_x 288 ... 384), n_x = 600 (c05), the production path for KI 125 / 139.80 / 114.45 (c05, c07), and the
inertness set of the regression test.

    env PYTHONPATH=<wt>:<wt>/test <venv>/bin/python d02_w7b_moved_values.py [out.json]
"""
import json
import math
import sys

import test_vol_pde_reverse_continuous_ki as t
from quantark.asset.equity.engine.pde import snowball_pde_solver as sps

LIB = sps._ContinuousKIFirstPassage._geometry


def rows():
    out = {}
    for n_x in range(288, 385):
        out[f"KI 125 ({n_x}, 48, 192)"] = (lambda n_x=n_x: t._price_2d((n_x, 48, 192), 125.0), "")
    for n in ((600, 48, 192), (600, 60, 800)):
        out[f"KI 125 {n}"] = (lambda n=n: t._price_2d(n, 125.0), "")
    for ki in (125.0, 139.80, 114.45):
        out[f"KI {ki:.2f} production PV {t.MEDIUM}"] = (lambda ki=ki: t._price_2d(t.MEDIUM, ki, **t.PRODUCTION), "")

        def greeks(ki=ki):
            s = t.HestonSnowballPDESolver(t.HP, n_x=t.MEDIUM[0], n_v=t.MEDIUM[1], n_t=t.MEDIUM[2], **t.PRODUCTION)
            g = s.calculate_greeks(t._snowball(ki), t._env())
            return tuple(float(g[k]) for k in ("price", "delta", "gamma"))
        out[f"KI {ki:.2f} production Greeks (price, delta, gamma)"] = (greeks, "")
    for name, f in t.INERT.items():
        out[f"inert: {name}"] = (f, "")
    return out


def run(f):
    try:
        v = f()
        vals = v if isinstance(v, tuple) else (v,)
        return [repr(float(x)) for x in vals], [float(x).hex() for x in vals]
    except Exception as exc:  # noqa: BLE001 - probe
        return [f"{type(exc).__name__}"], [f"{type(exc).__name__}"]


if __name__ == "__main__":
    res = {}
    for k, (f, _) in rows().items():
        sps._ContinuousKIFirstPassage._geometry = t.legacy_geometry
        try:
            b = run(f)
        finally:
            sps._ContinuousKIFirstPassage._geometry = LIB
        a = run(f)
        res[k] = {"before": b[0], "after": a[0], "moved": b[1] != a[1]}
        if a[1] == b[1]:
            print(f"{k:<58} bit-identical  {a[0]}", flush=True)
        else:
            try:
                d = " / ".join(f"{float(y) - float(x):+.3e}" for x, y in zip(b[0], a[0]))
            except ValueError:
                d = "n/a"
            print(f"{k:<58} MOVED  {' / '.join(b[0])} -> {' / '.join(a[0])}  (diff {d})", flush=True)
    moved = [k for k, v in res.items() if v["moved"]]
    scan = [k for k in moved if k.endswith(", 48, 192)") and k.startswith("KI 125 (") and "600" not in k]
    print(f"moved: {len(moved)} rows; of the 97-grid scan: {len(scan)}", flush=True)
    if len(sys.argv) > 1:
        json.dump(res, open(sys.argv[1], "w"), indent=1)
