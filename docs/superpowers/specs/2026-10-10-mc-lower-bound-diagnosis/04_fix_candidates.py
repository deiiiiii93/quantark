"""Step 3: candidate fixes, measured on replications (runtime patches only).

Rules for the lower-bound check (the estimator unchanged, so no passing price moves):
  current     : refuse if price < bound - 1e-6
  se_z{3,4,5} : refuse if price < bound - max(1e-6, z * std_error), std_error = the engine's own reported error
Alternative estimator (moves values): `parity` = price the out-of-the-money side (by forward moneyness) by MC and the
  in-the-money side as  OTM_MC + payment_df * (+-(F - K)),  F = S exp(-qT) / determination_df (the curve forward the
  generator reproduces exactly, 03_forward_and_discount.out). Then ITM >= bound by construction.

Also: the day-before test cell (test_expiry_carried_knock_in.py, mc / down_in_put) under the se_z4 rule, and the
size of the value move the parity estimator would cause on the default 256-path antithetic test configuration.
"""
import math
import warnings

import numpy as np

from _common import bs_price, make_engine, make_env, make_option, mc_raw, timing_of
from quantark.util.enum import OptionType

ZS = (3, 4, 5)


def parity_price(product, env, method, n, seed):
    """OTM side by the engine's MC (bound check off), ITM side by parity."""
    S, K = env.spot, product.strike
    T = product.get_maturity(env)
    tm = timing_of(product, env)
    F = S * math.exp(-env.get_div_yield(T) * T) / tm.determination_df
    put = not product.is_call()
    itm = (K > F) if put else (F > K)
    if not itm:
        p, se, b, _ = mc_raw(product, env, method=method, num_paths=n, seed=seed)
        return p, se, b
    other = make_option(T, OptionType.CALL if put else OptionType.PUT, K,
                        lag_days=round((tm.payment_time - tm.determination_time) * 365))
    p_otm, se, _, _ = mc_raw(other, env, method=method, num_paths=n, seed=seed)
    fwd_leg = tm.payment_df * ((K - F) if put else (F - K))
    _, _, b, _ = mc_raw(product, env, method=method, num_paths=8, seed=seed)  # the bound only
    return p_otm + fwd_leg, se, b


CELLS = [
    ("W13 put S90 T=1d", 90.0, make_option(1 / 365), False),
    ("put S75 T=1d", 75.0, make_option(1 / 365), False),
    ("put S90 T=5d", 90.0, make_option(5 / 365), False),
    ("put S90 T=30d", 90.0, make_option(30 / 365), False),
    ("call S110 T=1d term lag2d", 110.0, make_option(1 / 365, OptionType.CALL, lag_days=2), True),
    ("put S90 T=1y", 90.0, make_option(1.0), False),
    ("put S100 ATM T=1y", 100.0, make_option(1.0), False),
]

print(f"{'cell':28s} {'method':11s} {'n':>6s} {'R':>3s} | {'refused: cur':>12s} "
      + " ".join(f"{'z=' + str(z):>6s}" for z in ZS)
      + f" | {'parity bias':>11s} {'+-':>8s} {'parity sd':>9s} {'plain sd':>9s} {'parity refused':>14s} "
        f"{'mean|parity-plain|':>18s}")
for name, spot, prod, term in CELLS:
    env = make_env(spot, term_curve=term)
    bs = bs_price(prod, env)
    for method, n in (("plain", 256), ("antithetic", 256), ("antithetic", 4096), ("qmc", 4096), ("rqmc", 1024)):
        R = 8 if method == "rqmc" else 64
        seeds = [1 + 64 * i for i in range(R)] if method == "rqmc" else list(range(1, R + 1))
        cur, zc, pdiff, praw, pref, moves = 0, {z: 0 for z in ZS}, [], [], 0, []
        for s in seeds:
            p, se, b, _ = mc_raw(prod, env, method=method, num_paths=n, seed=s)
            cur += p < b - 1e-6
            for z in ZS:
                zc[z] += p < b - max(1e-6, z * se)
            pp, _, _ = parity_price(prod, env, method, n, s)
            pdiff.append(pp - bs)
            praw.append(p - bs)
            pref += pp < b - 1e-6
            moves.append(abs(pp - p))
        pdiff, praw = np.asarray(pdiff), np.asarray(praw)
        print(f"{name:28s} {method:11s} {n:6d} {R:3d} | {cur / R:12.2f} "
              + " ".join(f"{zc[z] / R:6.2f}" for z in ZS)
              + f" | {pdiff.mean():+11.3e} {pdiff.std(ddof=1) / math.sqrt(R):8.1e} {pdiff.std(ddof=1):9.2e} "
                f"{praw.std(ddof=1):9.2e} {pref / R:14.2f} {np.mean(moves):18.3e}", flush=True)

# the test cell itself under the se_z4 rule: the barrier MC engine -> EuropeanMCEngine, check patched at class level
print("\nDay-before test cell (mc, down_in_put, carried) under refuse-if price < bound - max(1e-6, 4*se):")
from quantark.asset.equity.engine.mc import BarrierOptionMCEngine
from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
from test_expiry_carried_knock_in import BARRIER_PRODUCTS, BEFORE, _mc, env as test_env, one_day_sd

orig_price = EuropeanMCEngine.price


def patched_price(self, product, pricing_env, *, lifecycle_state=None):
    """The engine's price with the check replaced by the z=4 statistical rule (bound captured, then compared)."""
    cap = {}
    ob = self._european_lower_bound
    self._european_lower_bound = lambda *a, **k: (cap.setdefault("b", ob(*a, **k)), -math.inf)[1]
    try:
        p = orig_price(self, product, pricing_env, lifecycle_state=lifecycle_state)
    finally:
        del self._european_lower_bound
    if "b" in cap and p < cap["b"] - max(1e-6, 4 * self.get_last_std_error()):
        raise RuntimeError(f"z=4 rule refuses {p} < {cap['b']}")
    return p


EuropeanMCEngine.price = patched_price
build = dict((k, b) for k, b, _, _ in BARRIER_PRODUCTS)["down_in_put"]
for form in ("dated", "numeric"):
    for spot in (90.0, 75.0):
        settled = max(100.0 - spot, 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            before = BarrierOptionMCEngine(params=_mc()).price(build(form, True, BEFORE), test_env(spot, BEFORE))
        ok = abs(before - settled) <= one_day_sd(spot)
        print(f"  {form:7s} S={spot:g}: before={before:.8f} settled={settled:g} |diff|={abs(before - settled):.3e} "
              f"<= one-day sd {one_day_sd(spot):.4f}: {ok}")
EuropeanMCEngine.price = orig_price
