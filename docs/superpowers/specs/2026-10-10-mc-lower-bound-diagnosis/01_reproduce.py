"""Step 1a: reproduce the refusal on the exact contract of test/test_expiry_carried_knock_in.py.

Contract: down-in put K=100, barrier 80, rebate 5, carried knocked in, flat vol 20%, r=3%, q=1%, valuation
2026-01-04 (the day before the 2026-01-05 exercise date), T = 1/365, no settlement lag; BarrierOptionMCEngine with
MCParams(seed=7, num_paths=256, use_antithetic=True), which prices the knocked-in contract as a EuropeanMCEngine
vanilla. Spots of the test: 90, 110, 75 (110 is out of the money).
"""
import math
import warnings

from test_expiry_carried_knock_in import BEFORE, BARRIER_PRODUCTS, env, _mc
from quantark.asset.equity.engine.mc import BarrierOptionMCEngine
from quantark.asset.equity.engine.mc.euro_mc_engine import EuropeanMCEngine
from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine
from quantark.asset.equity.engine.settlement_support import resolve_terminal_timing
from quantark.asset.equity.product.option import EuropeanVanillaOption

build = dict((k, b) for k, b, _, _ in BARRIER_PRODUCTS)["down_in_put"]
print("MCParams of the test:", _mc())

for form in ("dated", "numeric"):
    for spot in (90.0, 110.0, 75.0):
        product = build(form, True, BEFORE)
        e = env(spot, BEFORE)
        T = product.get_maturity(e)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = BarrierOptionMCEngine(params=_mc()).price(product, e)
            status = f"priced {p:.8f}"
        except Exception as exc:  # noqa: BLE001
            status = f"REFUSED {type(exc).__name__}: {exc}"
        # the engine's estimate with the bound check off (the same vanilla the barrier engine builds)
        vanilla = EuropeanVanillaOption(strike=product.strike, option_type=product.option_type,
                                        maturity=product.maturity, exercise_date=product.exercise_date,
                                        settlement_date=product.settlement_date,
                                        settlement_convention=product.settlement_convention)
        eng = EuropeanMCEngine(params=_mc())
        cap = {}
        orig = eng._european_lower_bound
        eng._european_lower_bound = lambda *a, **k: (cap.setdefault("b", orig(*a, **k)), -math.inf)[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = eng.price(vanilla, e)
            bs = BlackScholesEngine().price(vanilla, e)
        tm = resolve_terminal_timing(vanilla, e)
        se = eng.get_last_std_error()
        print(f"{form:7s} S={spot:5.1f} T={T:.12f} (1/365={1/365:.12f}) r(T)={e.get_rate(T):.6f} "
              f"q={e.get_div_yield(T):.6f} vol={e.get_vol(100.0, T):.4f}")
        print(f"    barrier engine: {status}")
        rel = (raw - cap['b']) / se if se > 0 else float('nan')
        print(f"    EuropeanMC raw = {raw:.8f}  se = {se:.3e}  bound = {cap['b']:.8f}  "
              f"raw-bound = {raw - cap['b']:+.3e} ({rel:+.2f} se)")
        print(f"    BS closed form = {bs:.10f}  BS-bound = {bs - cap['b']:+.3e}  raw-BS = {raw - bs:+.3e}")
        print(f"    timing: payment_df={tm.payment_df:.12f} determination_df={tm.determination_df:.12f} "
              f"delay_df={tm.delay_df:.12f} payment_time={tm.payment_time:.12f}")
