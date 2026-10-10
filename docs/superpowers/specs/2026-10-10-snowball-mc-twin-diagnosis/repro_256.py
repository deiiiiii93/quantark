"""Step 1a: the W12 measurement as stated (BS SnowballMCEngine, 256 paths), both forms, one process."""
import common as c
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams

if __name__ == "__main__":
    for seed in (7, 11):
        for day in (10, 45, 90, 92, 200):
            out = {}
            for form in ("dated", "numeric"):
                p = c.build("snowball", form, day)
                out[form] = SnowballMCEngine(params=MCParams(seed=seed, num_paths=256)).price(
                    p, c.env(100.0, c.on(day)))
            print(f"seed={seed} day={day:3d} dated={out['dated']:.6f} numeric={out['numeric']:.6f} "
                  f"gap={out['dated'] - out['numeric']:+.6f}")
