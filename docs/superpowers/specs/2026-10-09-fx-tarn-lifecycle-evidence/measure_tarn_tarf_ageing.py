"""W4b evidence: how FX TARN / TARF behave when aged or rebuilt mid-life, on main dcea77b6.

M1  TARN accrual: rebuilt from remaining fixing times vs aged by time_shift (findings §10).
M2  Pricing a time_shift-ed TARN / TARF whose shift stepped over a fixing.
M3  Theta near a fixing (BaseFxEngine._fdm_theta: time_shift + record_passed_fixings).
M4  Missing realized state: the remaining contract with the FULL target vs the REMAINING target.
M5  FX backtest / dynamic scenario clock: moving env.valuation_date does not age a numeric TARN / TARF.
M6  Parity arithmetic: accrual factors derived from shifted times vs the inception factors (ULPs).

Run from the repo root:  env PYTHONPATH=$PWD:$PWD/test <venv>/bin/python <this file>
"""
import copy
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np

from quantark.asset.fx.engine.mc import FxTargetRedemptionNoteMCEngine, FxTarnForwardMCEngine
from quantark.asset.fx.engine.mc.fx_mc_params import FxMCParams
from quantark.asset.fx.product import CurrencyPair
from quantark.asset.fx.product.option import FxTargetRedemptionForward, FxTargetRedemptionNote
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.priceenv import FxPricingEnvironment

PAIR = CurrencyPair("EUR", "USD")
VAL = datetime(2026, 6, 15)
FIX = [k / 12 for k in range(1, 13)]


def env(spot=1.20, val=VAL):
    return FxPricingEnvironment(valuation_date=val, spot_quote=SpotQuote(spot=spot),
                                domestic_curve=FlatRateCurve(rate=0.05), foreign_curve=FlatRateCurve(rate=0.03),
                                vol_surface=FlatVolSurface(volatility=0.10))


def mc(seed=7, n=200_000):
    return FxMCParams(num_paths=n, seed=seed)


def note(fix=FIX, **kw):
    base = dict(fixing_times=fix, coupon_rate=0.12, notional=100.0, strike=1.10, currency_pair=PAIR)
    base.update(kw)
    return FxTargetRedemptionNote(**base)


def tarf(fix=FIX, **kw):
    base = dict(strike=1.15, fixing_times=fix, notional=1.0, currency_pair=PAIR)
    base.update(kw)
    return FxTargetRedemptionForward(**base)


def try_price(engine, prod, e):
    try:
        return f"{engine.price(prod, e):.6f}"
    except Exception as exc:  # report, do not hide
        return f"RAISES {type(exc).__name__}: {exc}"


def try_theta(engine, prod, e):
    try:
        return f"theta {engine.calculate_greeks(prod, e)['theta']:.6f}"
    except Exception as exc:
        return f"calculate_greeks RAISES {type(exc).__name__}: {exc}"


def m1():
    print("M1 TARN accrual factors, 12 monthly fixings, notional 100 at 12%")
    n0 = note()
    sh = copy.deepcopy(n0)
    sh.time_shift(0.125)
    rem = [t for t in sh.fixing_times if t > 0]
    rb = note(fix=rem)
    print(f"  inception first 2 factors {np.round(n0.accrual_factors[:2], 6).tolist()}")
    print(f"  time_shift(0.125): {len(sh.fixing_times)} fixings kept, first t={sh.fixing_times[0]:.6f}, "
          f"factors {np.round(sh.accrual_factors[:2], 6).tolist()}")
    print(f"  rebuilt from remaining {len(rem)} times: first factor {rb.accrual_factors[0]:.6f} "
          f"-> coupon {rb.full_coupon(0):.4f} (contract {n0.full_coupon(1):.4f})")


def m2():
    print("M2 price after time_shift(0.125) + record_passed_fixings: the shift stepped over the first fixing")
    for name, prod, eng in (("TARN", note(target=3.0), FxTargetRedemptionNoteMCEngine(mc())),
                            ("TARF", tarf(target=0.05), FxTarnForwardMCEngine(mc()))):
        sh = copy.deepcopy(prod)
        sh.time_shift(0.125)
        sh.record_passed_fixings(env())
        print(f"  {name}: {try_price(eng, sh, env())}")


def m3():
    print("M3 theta (theta_days=1) with the first fixing half a day / exactly one day / two days away")
    for label, t1 in (("0.5/365", 0.5 / 365), ("1/365", 1 / 365), ("2/365", 2 / 365)):
        fix = [t1] + FIX[1:]
        for name, prod, eng in (("TARN", note(fix=fix, target=3.0), FxTargetRedemptionNoteMCEngine(mc(n=20_000))),
                                ("TARF", tarf(fix=fix, target=0.05), FxTarnForwardMCEngine(mc(n=20_000)))):
            print(f"  first fixing {label} {name}: {try_theta(eng, prod, env())}")


def m4():
    print("M4 two months aged, both fixings paid; remaining 10 fixings, same seed")
    rem = [t - 2 / 12 for t in FIX[2:]]
    paid = 2 * note().full_coupon(0)
    e = env()
    for incl in (True, False):
        eng = FxTargetRedemptionNoteMCEngine(mc())
        kw = dict(fix=rem, accrual_factors=[1 / 12] * 10, include_principal=incl)
        full = eng.price(note(target=3.0, **kw), e)
        right = eng.price(note(target=3.0 - paid, **kw), e)
        print(f"  TARN include_principal={incl}: target 3.0 (no state) {full:.6f}  remaining target "
              f"{3.0 - paid:.6f} {right:.6f}  diff {full - right:+.6f}")
    eng = FxTarnForwardMCEngine(mc())
    full = eng.price(tarf(fix=rem, target=0.20), e)
    right = eng.price(tarf(fix=rem, target=0.10), e)
    print(f"  TARF target 0.20 (no state) {full:.6f}  remaining target 0.10 {right:.6f}  diff {full - right:+.6f}")


def m5():
    print("M5 env.valuation_date moved 91 days, product unchanged (the FX backtest / dynamic scenario clock)")
    for name, prod, eng in (("TARN", note(target=3.0), FxTargetRedemptionNoteMCEngine(mc())),
                            ("TARF", tarf(target=0.05), FxTarnForwardMCEngine(mc()))):
        e0, e1 = env(), replace(env(), valuation_date=VAL + timedelta(days=91))
        p0, p1 = eng.price(prod, e0), eng.price(prod, e1)
        print(f"  {name}: get_maturity {prod.get_maturity(e0):.6f} -> {prod.get_maturity(e1):.6f}; "
              f"price {p0:.6f} -> {p1:.6f} (identical: {p0 == p1})")


def m6():
    print("M6 factors derived from shifted times vs inception factors (11 surviving periods after the first)")
    kept = note().accrual_factors[1:]
    for tau in (0.125, 0.1, 45 / 365, 91 / 365):
        shifted = [t - tau for t in FIX]
        derived = [b - a for a, b in zip(shifted[:-1], shifted[1:])]
        d = max(abs(x - y) for x, y in zip(derived, kept))
        n_diff = sum(x != y for x, y in zip(derived, kept))
        print(f"  tau={tau:.6f}: max |derived - inception| = {d:.3e}; periods not bitwise equal: {n_diff}/11")


if __name__ == "__main__":
    m1()
    m2()
    m3()
    m4()
    m5()
    m6()
