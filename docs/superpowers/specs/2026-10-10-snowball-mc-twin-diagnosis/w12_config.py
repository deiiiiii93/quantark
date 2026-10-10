"""Step 1a': find the W12 configuration that gave 12.08 / 11.34 on day 92 (W12 REPORT section 7 item 3 names only
"256 paths"): the 256-path MCParams of the neighbouring test modules, with and without antithetic, and the patched
grid (patch_grid.py) on the same configuration."""
import common as c
import patch_grid
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams

if __name__ == "__main__":
    for seed in (7, 11, 42):
        for anti in (False, True):
            def p(form, day):
                return float(SnowballMCEngine(params=MCParams(seed=seed, num_paths=256, use_antithetic=anti)).price(
                    c.build("snowball", form, day), c.env(100.0, c.on(day))))
            for day in (45, 92):
                d, n = p("dated", day), p("numeric", day)
                undo = patch_grid.install()
                try:
                    pd_, pn = p("dated", day), p("numeric", day)
                finally:
                    undo()
                print(f"seed={seed:2d} antithetic={anti!s:5s} day={day}: dated={d:.6f} numeric={n:.6f} "
                      f"gap={d - n:+.3e} | patched dated={pd_:.6f} numeric={pn:.6f} gap={pd_ - pn:+.3e}")

    # search: which 256-path seed reproduces W12's 12.08 / 11.34 on day 92
    hits = []
    for seed in range(0, 200):
        def p(form):
            return float(SnowballMCEngine(params=MCParams(seed=seed, num_paths=256)).price(
                c.build("snowball", form, 92), c.env(100.0, c.on(92))))
        d, n = p("dated"), p("numeric")
        if round(d, 2) == 12.08 and round(n, 2) == 11.34:
            hits.append((seed, d, n))
    print("seeds 0..199 with dated 12.08 / numeric 11.34 on day 92 (256 paths):", hits)
    for seed, _, _ in hits:
        for form in ("dated", "numeric"):
            eng = SnowballMCEngine(params=MCParams(seed=seed, num_paths=256))
            v = eng.price(c.build("snowball", form, 92), c.env(100.0, c.on(92)))
            print(f"  seed={seed} {form}: {v:.6f}, engine std_error {eng.get_last_std_error():.4f}")
        undo = patch_grid.install()
        try:
            pv = {f: float(SnowballMCEngine(params=MCParams(seed=seed, num_paths=256)).price(
                c.build("snowball", f, 92), c.env(100.0, c.on(92)))) for f in ("dated", "numeric")}
        finally:
            undo()
        print(f"  seed={seed} patched: dated={pv['dated']:.6f} numeric={pv['numeric']:.6f}")
