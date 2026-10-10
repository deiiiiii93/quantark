"""Step 3b: scan every valuation day and compare the MC time grid of the two forms (grid building only).

For each day and form: the node count and the smallest step. A step below 1e-12 is a spurious node: two spellings
of one instant (an observation time and a continuous-KI linspace node, or maturity) kept as two set elements. For
each spurious step the two times and where each comes from (KO record / continuous-KI linspace / maturity T).

Products: the W12 snowball and non-memory Phoenix (BS SnowballMCEngine / PhoenixMCEngine ``_build_time_grid``) and
the W11 KO-reset with a CONTINUOUS knock-in (``_build_time_grid_ko_reset``, alive to day 181)."""
import numpy as np

import common as c
from other_engines import ko_reset_continuous
from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.mc.term_inputs import make_df_fn
from quantark.asset.equity.param import MCParams


def grid(kind, form, day):
    e = c.env(100.0, c.on(day))
    if kind == "ko_reset_continuous_ki":
        p = ko_reset_continuous(form, day)
        eng = SnowballMCEngine(params=MCParams(seed=7, num_paths=16))
        eng._term_ctx = (e, p.strike)
        eng._df = make_df_fn(e)
        T = p.get_max_maturity_time(e)
        g = eng._build_time_grid_ko_reset(p, e, T)
        ko = np.concatenate([g["pre_times"], g["post_times"]])
        return T, g["all_times"], g["dt_array"], ko
    p = c.build(kind, form, day)
    eng = (PhoenixMCEngine if kind == "phoenix" else SnowballMCEngine)(params=MCParams(seed=7, num_paths=16))
    eng._term_ctx = (e, p.strike)
    eng._df = make_df_fn(e)
    T = p.get_maturity(e)
    all_times, dt, _, _ = eng._build_time_grid(p, e, T)
    ko = np.asarray(p.get_ko_observation_profile(e)["observation_times"], float)
    return T, all_times, dt, ko


def sources(t, T, ko):
    s = []
    if t == T:
        s.append("T")
    if np.any(ko == t):
        s.append("KO")
    return "+".join(s) if s else "KI-linspace"


if __name__ == "__main__":
    for kind, last in (("snowball", 364), ("phoenix", 364), ("ko_reset_continuous_ki", 181)):
        print(f"### {kind} (days 1..{last})")
        tiny = {"dated": [], "numeric": []}
        node_diff = []
        for day in range(1, last + 1):
            row = {}
            for form in ("dated", "numeric"):
                T, at, dt, ko = grid(kind, form, day)
                row[form] = (len(at), float(dt.min()))
                for i in np.flatnonzero(dt < 1e-12):
                    a, b = at[i - 1], at[i]
                    tiny[form].append(f"d{day}: {a!r}[{sources(a, T, ko)}] / {b!r}[{sources(b, T, ko)}]")
            if row["dated"][0] != row["numeric"][0]:
                node_diff.append((day, row["dated"][0], row["numeric"][0]))
        for form in ("dated", "numeric"):
            print(f"  {form}: {len(tiny[form])} spurious steps < 1e-12")
            for s in tiny[form]:
                print("     ", s)
        print("  days where the node counts differ (day, dated n, numeric n):", node_diff)
