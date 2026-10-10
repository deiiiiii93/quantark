"""Measure every cell of ``test/test_expiry_carried_knock_in.py`` (and the W9 carried knock-in cells of
``test/test_dated_expiry_settles.py``) on one tree: prices on the exercise date in both forms and every carrier, the PDE
greeks there, and the day-before prices (unchanged code, measured to show they do not move).

Run against any tree with the BRANCH's test directory second on the path (the cells are the branch's):

    env PYTHONPATH=<tree>:<branch>/test <venv>/bin/python measure.py > <out>.txt
"""
from __future__ import annotations

import warnings

warnings.simplefilter("ignore")

import quantark  # noqa: E402
import test_dated_expiry_settles as w9  # noqa: E402
import test_expiry_carried_knock_in as t  # noqa: E402
from quantark.asset.equity.lifecycle.state import AutocallableLifecycleState  # noqa: E402


def attempt(fn):
    try:
        return repr(float(fn()))
    except Exception as exc:  # noqa: BLE001 -- the evidence records the refusal
        return f"RAISES {type(exc).__name__}: {str(exc).splitlines()[0][:100]}"


def row(cell, form, quantity, value):
    print(f"{cell} | {form} | {quantity} | {value}")


print(f"# quantark from {quantark.__file__}")

# -- Part 1: barriers on the exercise date ------------------------------------------------------------------------
for route, factory, state_ok, kind, build, hand, spot in t.BARRIER_SPOT_CASES:
    cell = f"barrier:{kind}-{route}-{spot:g}"
    for form in ("dated", "numeric"):
        row(cell, form, "price_carried_flag", attempt(lambda: t._price(route, factory, build(form, True), spot)))
        if state_ok:
            row(cell, form, "price_carried_state", attempt(lambda: t._price(
                route, factory, build(form, False), spot, lifecycle_state=AutocallableLifecycleState(knocked_in=True))))
        row(cell, form, "price_not_carried", attempt(lambda: t._price(route, factory, build(form, False), spot)))

for route, factory, _ in t._PDE_GREEK_ROUTES:
    build = t.BARRIER_PRODUCTS[0][1]
    for form in ("dated", "numeric"):
        for key in ("price", "delta", "gamma"):
            row(f"greeks:up_in_call-{route}-110", form, key, attempt(
                lambda: t._quiet(lambda: factory().calculate_greeks(build(form, True), t.env(110.0, route=route)))[key]))

# -- W9: the vol-model routes and their BSM counterparts on the carried up-and-in call -------------------------
for route, vol, bsm, kind, build, spot in w9.VOL_CASES:
    if kind != "up_in_call_knocked_in":
        continue
    for form in ("dated", "numeric"):
        row(f"w9:{route}-{kind}-{spot:g}", form, "price", attempt(lambda: vol().price(build(form), w9._vol_env(spot))))
        row(f"w9:{route}-{kind}-{spot:g}", form, "price_bsm",
            attempt(lambda: bsm().price(build(form), w9._vol_env(spot))))
        for key in ("price", "delta", "gamma"):
            row(f"w9:{route}-{kind}-{spot:g}", form, f"greeks_{key}",
                attempt(lambda: vol().calculate_greeks(build(form), w9._vol_env(spot))[key]))

# -- the day before (T = 1/365): unchanged code paths --------------------------------------------------------------
for route, factory, state_ok, kind, build, hand, spot in t._BARRIER_BRANCH_CASES:
    for form in ("dated", "numeric"):
        row(f"before:{kind}-{route}-{spot:g}", form, "price_carried_flag",
            attempt(lambda: t._price(route, factory, build(form, True, t.BEFORE), spot, valuation=t.BEFORE)))

# -- Part 2: autocallables -----------------------------------------------------------------------------------------
for kind, route, factory, hand, spot, carried in t.AUTOCALL_CASES:
    build = t._autocallable(kind)
    cell = f"autocall:{kind}-{route}-{'carried' if carried else 'plain'}-{spot:g}"
    for form in t._forms(kind, route):
        row(cell, form, "price", attempt(lambda: t._price(route, factory, build(form, carried), spot)))
        if route == "quad_v2" and carried:
            row(cell, form, "price_carried_state", attempt(lambda: t._price(
                route, factory, build(form, False), spot, lifecycle_state=AutocallableLifecycleState(knocked_in=True))))
        if carried and spot < 103.0:
            row(cell, form, "price_day_before", attempt(
                lambda: t._price(route, factory, build(form, True, t.BEFORE), spot, valuation=t.BEFORE)))
