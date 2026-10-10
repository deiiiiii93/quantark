"""Step 3: diff the MC's intermediate objects between the two forms (T, KO profile, accrual payoffs, payment timings,
time grid) on each valuation day. No library change: the engine's own private builders are called directly."""
import numpy as np

import common as c
from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.param import MCParams


def inputs(kind, form, day):
    p = c.build(kind, form, day)
    e = c.env(100.0, c.on(day))
    eng = SnowballMCEngine(params=MCParams(seed=7, num_paths=256))
    T = p.get_maturity(e)
    eng._term_ctx = (e, p.strike)
    from quantark.asset.equity.engine.mc.term_inputs import make_df_fn
    eng._df = make_df_fn(e)
    eng._prepare_payment_timings(p, e)
    prof = p.get_ko_observation_profile(e)
    all_times, dt, ko_idx, ki_idx = eng._build_time_grid(p, e, T)
    tm = eng._payment_timings
    return dict(
        T=T,
        ko_times=np.asarray(prof["observation_times"], float),
        ko_barriers=np.asarray(prof["barriers"], float),
        ko_payoffs=np.asarray(prof["payoffs"], float),
        ko_pay_times=np.asarray(tm.observation_payment_times, float),
        terminal_pay=float(tm.terminal.payment_time),
        n_nodes=len(all_times),
        min_dt=float(dt.min()),
        ko_indices=ko_idx,
        n_ki=len(ki_idx),
        all_times=all_times,
    )


if __name__ == "__main__":
    for day in (45, 90, 91, 92, 93, 100, 121, 122, 123, 200):
        d = inputs("snowball", "dated", day)
        n = inputs("snowball", "numeric", day)
        print(f"=== day {day}")
        for k in d:
            a, b = d[k], n[k]
            if isinstance(a, np.ndarray):
                same_shape = a.shape == b.shape
                if same_shape:
                    diff = float(np.max(np.abs(a - b))) if a.size else 0.0
                    eq = "EQUAL" if np.array_equal(a, b) else f"maxdiff={diff:.3e}"
                else:
                    eq = f"SHAPE {a.shape} vs {b.shape}"
                if k == "all_times":
                    print(f"  {k:14s} {eq}")
                else:
                    print(f"  {k:14s} {eq}" + ("" if np.array_equal(a, b) else f"\n     dated  ={a}\n     numeric={b}"))
            else:
                print(f"  {k:14s} dated={a!r} numeric={b!r}" + ("" if a == b else "  <-- DIFFER"))
