"""Step 3c: the candidate fix as a RUNTIME patch (no library change). The time grids of the BS snowball / KO-reset /
Phoenix MC engines are the sorted SET of the observation times and the continuous-KI linspace; two spellings of one
instant that differ by an ulp survive as two nodes 1e-16 apart. The patch merges nodes closer than TOL (the cluster
keeps its LAST member, so maturity stays the final node) and re-maps every observation index onto the merged grid.

Measured:
  (a) every valuation day 1..364, W12 snowball and Phoenix, 256 paths seed 7 (the W12 configuration) and 4096 paths
      seed 11: |dated - numeric| unpatched vs patched;
  (b) the moved values: patched vs unpatched price of each form on every day (a value moves only on a day whose
      grid held a sub-TOL step)."""
import sys

import numpy as np

import common as c
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams

TOL = 1e-10          # quantark.util.numerical Tolerance.ZERO: 3 ms of a year; contract nodes are a day apart


def _merge(all_times):
    all_times = np.asarray(all_times, float)
    keep = np.concatenate([np.diff(all_times) > TOL, [True]])
    return all_times[keep]


def _remap(old_times, new_times, idx):
    idx = np.asarray(idx, dtype=int)
    out = idx.copy()
    live = idx >= 0
    out[live] = np.searchsorted(new_times, old_times[idx[live]] - TOL)
    return out


def install():
    snow_orig = SnowballMCEngine._build_time_grid
    phx_orig = PhoenixMCEngine._build_time_grid

    def snow_patched(self, product, pricing_env, T):
        all_times, dt, ko_idx, ki_idx = snow_orig(self, product, pricing_env, T)
        new = _merge(all_times)
        if new.size == all_times.size:
            return all_times, dt, ko_idx, ki_idx
        ki_cont = len(ki_idx) == len(all_times) and np.array_equal(ki_idx, np.arange(len(all_times)))
        new_ki = np.arange(len(new)) if ki_cont else _remap(all_times, new, ki_idx)
        return new, np.diff(np.concatenate([[0.0], new])), _remap(all_times, new, ko_idx), new_ki

    def phx_patched(self, product, pricing_env, T):
        # the Phoenix indices are PATH columns: node index + 1, 0 for an observation at valuation
        all_times, dt, ko_cols, ki_cols = phx_orig(self, product, pricing_env, T)
        new = _merge(all_times)
        if new.size == all_times.size:
            return all_times, dt, ko_cols, ki_cols

        def cols(c_):
            c_ = np.asarray(c_, dtype=int)
            if c_.size == 0:
                return c_
            return np.where(c_ > 0, _remap(all_times, new, c_ - 1) + 1, 0)
        return new, np.diff(np.concatenate([[0.0], new])), cols(ko_cols), cols(ki_cols)

    SnowballMCEngine._build_time_grid = snow_patched
    PhoenixMCEngine._build_time_grid = phx_patched
    return lambda: (setattr(SnowballMCEngine, "_build_time_grid", snow_orig),
                    setattr(PhoenixMCEngine, "_build_time_grid", phx_orig))


def price(kind, form, day, seed, paths):
    eng = (PhoenixMCEngine if kind == "phoenix" else SnowballMCEngine)(params=MCParams(seed=seed, num_paths=paths))
    return float(eng.price(c.build(kind, form, day), c.env(100.0, c.on(day))))


if __name__ == "__main__":
    configs = [(7, 256), (11, 4096)]
    days = range(1, 365) if len(sys.argv) < 2 else [int(a) for a in sys.argv[1:]]
    for kind in ("snowball", "phoenix"):
        for seed, paths in configs:
            worst = {"unpatched": (0.0, None), "patched": (0.0, None)}
            nonzero = {"unpatched": [], "patched": []}
            moved = {"dated": [], "numeric": []}
            for day in days:
                base = {f: price(kind, f, day, seed, paths) for f in ("dated", "numeric")}
                undo = install()
                try:
                    pat = {f: price(kind, f, day, seed, paths) for f in ("dated", "numeric")}
                finally:
                    undo()
                for label, v in (("unpatched", base), ("patched", pat)):
                    g = abs(v["dated"] - v["numeric"])
                    if g > worst[label][0]:
                        worst[label] = (g, day)
                    if g > 1e-10 * max(1.0, abs(v["dated"])):
                        nonzero[label].append((day, round(v["dated"] - v["numeric"], 6)))
                for f in ("dated", "numeric"):
                    if pat[f] != base[f]:
                        moved[f].append((day, round(pat[f] - base[f], 6)))
            print(f"### {kind} seed={seed} paths={paths} days={days[0]}..{days[-1]} ({len(days)} days)")
            for label in ("unpatched", "patched"):
                print(f"  {label:9s}: twin gap > rel 1e-10 on {len(nonzero[label])} days, worst |gap| "
                      f"{worst[label][0]:.3e} (day {worst[label][1]}): {nonzero[label]}")
            for f in ("dated", "numeric"):
                print(f"  moved by the patch, {f:7s}: {len(moved[f])} days {moved[f]}")
            sys.stdout.flush()
