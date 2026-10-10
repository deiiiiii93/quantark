"""Step 2a: the deterministic references. BS SnowballPDESolver(PDEParams()) and SnowballQuadEngineV2() on both forms
of the W12 snowball (and the non-memory Phoenix with PhoenixPDESolver / PhoenixQuadEngineV2), on the W12 days and
on the days where grid_scan.py found a spurious MC node in either form."""
import time

import common as c
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.quad.v2.engine import PhoenixQuadEngineV2, SnowballQuadEngineV2
from quantark.asset.equity.param import PDEParams

DAYS = (10, 14, 41, 45, 53, 90, 91, 92, 93, 105, 122, 170, 183, 200, 248)

ENGINES = {
    "snowball": {"pde": lambda: SnowballPDESolver(PDEParams()), "quad_v2": lambda: SnowballQuadEngineV2()},
    "phoenix": {"pde": lambda: PhoenixPDESolver(PDEParams()), "quad_v2": lambda: PhoenixQuadEngineV2()},
}

if __name__ == "__main__":
    for kind in ("snowball", "phoenix"):
        for day in DAYS:
            vals = {}
            for name, factory in ENGINES[kind].items():
                for form in ("dated", "numeric"):
                    t0 = time.perf_counter()
                    try:
                        v = float(factory().price(c.build(kind, form, day), c.env(100.0, c.on(day))))
                    except Exception as exc:  # noqa: BLE001
                        v = float("nan")
                        print(f"  {kind} {name} {form} day {day} RAISES {type(exc).__name__}: {exc}")
                    vals[(name, form)] = v
            print(f"{kind:8s} day={day:3d} "
                  f"pde dated={vals[('pde', 'dated')]:.10f} numeric={vals[('pde', 'numeric')]:.10f} "
                  f"gap={vals[('pde', 'dated')] - vals[('pde', 'numeric')]:+.2e} | "
                  f"quad dated={vals[('quad_v2', 'dated')]:.10f} numeric={vals[('quad_v2', 'numeric')]:.10f} "
                  f"gap={vals[('quad_v2', 'dated')] - vals[('quad_v2', 'numeric')]:+.2e} | "
                  f"pde-quad(dated)={vals[('pde', 'dated')] - vals[('quad_v2', 'dated')]:+.2e}", flush=True)
