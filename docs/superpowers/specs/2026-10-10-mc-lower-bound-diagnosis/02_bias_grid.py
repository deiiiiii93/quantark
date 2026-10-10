"""Step 1b: MC estimate (bound check off) against the Black-Scholes closed form, replicated over seeds.

For each cell: R independent replications (seed = 1..R, which also re-scrambles the Sobol set under QMC; RQMC
seeds 1, 65, 129, ... because the engine scrambles batch b with seed + b, so consecutive seeds share batches).
bias = mean_r(MC_r - BS) with its standard error std_r / sqrt(R); `emp_sd` = std over replications (the real error
of ONE price); `eng_se` = mean of the engine's own reported std error; `pair_se` = the antithetic std error computed
over pair means (pseudo + antithetic only); `refused` = fraction of replications where the UNPATCHED engine check
(price < bound - 1e-6) fires. All in price units of one contract (multiplier 1).

Part A: the W13 contract (put K=100, S=90, T=1/365, flat r=3% q=1% vol 20%, no lag), paths x methods.
Part B: knobs at 4096 paths: T, settlement lag, term curve, deep-ITM call, ATM controls.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from _common import bs_price, make_env, make_option, mc_raw
from quantark.util.enum import OptionType

PATHS = (256, 1024, 4096, 16384, 65536)
METHODS = ("plain", "antithetic", "qmc", "qmc+antithetic", "rqmc")


def reps_for(method, n):
    if method == "rqmc":
        return 8 if n <= 4096 else 4
    return 64 if n <= 4096 else (32 if n <= 16384 else 16)


def run_cell(product, env, method, n, R=None):
    R = R or reps_for(method, n)
    bs = bs_price(product, env)
    diffs, eng_se, pair_se, refused, total_paths = [], [], [], 0, []
    bound = None
    # RQMC scrambles with base_seed + batch_id (up to 32 batches): consecutive seeds would SHARE batches, so RQMC
    # replications are spaced 64 seeds apart to be independent
    seeds = [1 + 64 * i for i in range(R)] if method == "rqmc" else list(range(1, R + 1))
    for seed in seeds:
        captured = {}
        price, se, bound, engine = None, None, None, None
        # stash payoffs to compute the antithetic pair std error
        from _common import make_engine
        import math, warnings
        engine = make_engine(method, n, seed)
        orig_bound = engine._european_lower_bound
        orig_pay = engine._calculate_payoffs
        stash = []

        def pay(prod, st, _o=orig_pay, _s=stash):
            p = _o(prod, st)
            _s.append(p)
            return p

        engine._calculate_payoffs = pay
        engine._european_lower_bound = lambda *a, _o=orig_bound, _c=captured, **k: (
            _c.setdefault("b", _o(*a, **k)), -math.inf)[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            price = float(engine.price(product, env))
        se = float(engine.get_last_std_error())
        bound = captured["b"]
        diffs.append(price - bs)
        eng_se.append(se)
        if price < bound - 1e-6:
            refused += 1
        total_paths.append(sum(len(s) for s in stash))
        if method == "antithetic":
            p = stash[0]
            h = len(p) // 2
            pm = 0.5 * (p[:h] + p[h:2 * h])
            pair_se.append(float(np.std(pm, ddof=1) / np.sqrt(h)) if h > 1 else float("nan"))
    d = np.asarray(diffs)
    return dict(bs=bs, bound=bound, bias=float(d.mean()), bias_se=float(d.std(ddof=1) / np.sqrt(len(d))),
                emp_sd=float(d.std(ddof=1)), eng_se=float(np.mean(eng_se)),
                pair_se=float(np.mean(pair_se)) if pair_se else float("nan"),
                refused=refused / len(d), R=len(d), paths=int(np.mean(total_paths)))


HEADER = (f"{'cell':34s} {'method':15s} {'paths':>7s} {'R':>3s} {'bias':>11s} {'bias_se':>9s} {'t':>6s} "
          f"{'emp_sd':>9s} {'eng_se':>9s} {'pair_se':>9s} {'bias_bp':>8s} {'refused':>7s}")


def fmt(cell, method, r):
    t = r["bias"] / r["bias_se"] if r["bias_se"] > 0 else float("nan")
    bp = 1e4 * r["bias"] / r["bs"] if r["bs"] > 1e-12 else float("nan")
    return (f"{cell:34s} {method:15s} {r['paths']:7d} {r['R']:3d} {r['bias']:+11.3e} {r['bias_se']:9.2e} {t:+6.2f} "
            f"{r['emp_sd']:9.2e} {r['eng_se']:9.2e} {r['pair_se']:9.2e} {bp:+8.3f} {r['refused']:7.2f}")


def main(part):
    t0 = time.time()
    if part in ("A", "all"):
        print("PART A: W13 contract put K=100 S=90 T=1/365 flat r=3% q=1% vol 20% lag 0")
        env, prod = make_env(90.0), make_option(1 / 365)
        print(f"  BS = {bs_price(prod, env):.12f}")
        print(HEADER)
        for method in METHODS:
            for n in PATHS:
                print(fmt("W13 put S90 1d", method, run_cell(prod, env, method, n)), flush=True)
    if part in ("B", "all"):
        print("\nPART B: knobs at 4096 paths (RQMC: 4096 per batch)")
        cells = []
        for days in (1, 5, 30, 365):
            cells.append((f"put S90 T={days}d flat lag0", make_env(90.0), make_option(days / 365)))
        cells.append(("put S90 T=1d flat lag2d", make_env(90.0), make_option(1 / 365, lag_days=2)))
        cells.append(("put S90 T=1d term lag0", make_env(90.0, term_curve=True), make_option(1 / 365)))
        cells.append(("put S90 T=1d term lag2d", make_env(90.0, term_curve=True), make_option(1 / 365, lag_days=2)))
        cells.append(("put S90 T=1y term lag2d", make_env(90.0, term_curve=True), make_option(1.0, lag_days=2)))
        cells.append(("call S110 T=1d flat lag0", make_env(110.0), make_option(1 / 365, OptionType.CALL)))
        cells.append(("call S110 T=1d term lag2d", make_env(110.0, term_curve=True),
                      make_option(1 / 365, OptionType.CALL, lag_days=2)))
        cells.append(("call S110 T=1y flat lag0", make_env(110.0), make_option(1.0, OptionType.CALL)))
        cells.append(("put S100 ATM T=1d flat", make_env(100.0), make_option(1 / 365)))
        cells.append(("put S100 ATM T=1y flat", make_env(100.0), make_option(1.0)))
        cells.append(("call S100 ATM T=1y term lag2d", make_env(100.0, term_curve=True),
                      make_option(1.0, OptionType.CALL, lag_days=2)))
        print(HEADER)
        for name, env, prod in cells:
            bs = bs_price(prod, env)
            for method in ("plain", "antithetic", "qmc", "rqmc"):
                r = run_cell(prod, env, method, 4096)
                print(fmt(name, method, r) + f"   BS={bs:.10f} BS-bound={bs - r['bound']:+.3e}", flush=True)
    print(f"\nelapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
