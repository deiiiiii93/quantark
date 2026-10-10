"""Step 2b: ON an observation day (day 91) the dated form keeps the landing record (decided on the known spot at
valuation) while the W12 numeric twin, aged with ``time_shift`` WITHOUT ``keep_landing``, drops it. Both numeric
agings, both forms, BS MC / PDE / QUAD V2, spots 100 (no knock-out) and 104 (knock-out at 103 now)."""
import common as c
from quantark.asset.equity.engine.mc import PhoenixMCEngine, SnowballMCEngine
from quantark.asset.equity.engine.pde.phoenix_pde_solver import PhoenixPDESolver
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.v2.engine import PhoenixQuadEngineV2, SnowballQuadEngineV2
from quantark.asset.equity.param import MCParams, PDEParams

ENGINES = {
    "snowball": {"mc": lambda: SnowballMCEngine(params=MCParams(seed=11, num_paths=4096)),
                 "pde": lambda: SnowballPDESolver(PDEParams()), "quad_v2": SnowballQuadEngineV2},
    "phoenix": {"mc": lambda: PhoenixMCEngine(params=MCParams(seed=11, num_paths=4096)),
                "pde": lambda: PhoenixPDESolver(PDEParams()), "quad_v2": PhoenixQuadEngineV2},
}

if __name__ == "__main__":
    day = 91
    for kind in ("snowball", "phoenix"):
        for spot in (100.0, 104.0):
            e = c.env(spot, c.on(day))
            for name, factory in ENGINES[kind].items():
                d = float(factory().price(c.build(kind, "dated", day), e))
                n0 = float(factory().price(c.build(kind, "numeric", day), e))
                n1 = float(factory().price(c.build(kind, "numeric", day, keep_landing=True), e))
                print(f"{kind:8s} day={day} spot={spot:5.1f} {name:7s} dated={d:.8f} "
                      f"numeric(drop landing)={n0:.8f} gap={d - n0:+.3e} | "
                      f"numeric(keep_landing)={n1:.8f} gap={d - n1:+.3e}", flush=True)
