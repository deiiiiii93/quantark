"""Measure every W12 cell on one tree: the dated / numeric twins of ``test/test_vol_dated_schedules.py`` (prices,
greeks, the flat-vol-surrogate statistics) and the day-before cells of ``test/test_expiry_carried_knock_in.py`` whose
pinned refusals W12 removed (the Heston / SLV barrier routes on the discrete schedule; the 2D snowball / Phoenix PDE).

Run against any tree with the BRANCH's test directory second on the path (the cells are the branch's):

    env PYTHONPATH=<tree>:<branch>/test <venv>/bin/python measure.py > <out>.txt
"""
from __future__ import annotations

import warnings

warnings.simplefilter("ignore")

import quantark  # noqa: E402
import test_expiry_carried_knock_in as c  # noqa: E402
import test_vol_dated_schedules as t  # noqa: E402


def attempt(fn):
    try:
        return repr(float(fn()))
    except Exception as exc:  # noqa: BLE001 -- the evidence records the refusal
        return f"RAISES {type(exc).__name__}: {str(exc).splitlines()[0][:100]}"


def row(cell, form, quantity, value):
    print(f"{cell} | {form} | {quantity} | {value}")


def one(build, form, run):
    """``run(product)`` for one form, the product built as the test builds it."""
    return attempt(lambda: t._quiet(lambda: run(build(form))))


print(f"# quantark from {quantark.__file__}")

# -- test_vol_dated_schedules: 2D snowball / Phoenix PDE ------------------------------------------------------
for route, kind, day in t.AUTOCALL_CASES:
    cell = f"autocall:{route}-{kind}-d{day}"
    for form in ("dated", "numeric"):
        def build(f, kind=kind, day=day):
            return t._aged(t.AUTOCALL_PRODUCTS[kind](f), f, day, age_dated=kind == "snowball_discrete_ki")
        row(cell, form, "price", one(build, form, lambda p: t.AUTOCALL_ROUTES[route](kind).price(
            p, t.env(100.0, t._on(day)))))
        if (route, kind, day) in t._GREEK_CASES:
            for key in ("price", "delta", "gamma"):
                row(cell, form, f"greeks_{key}", one(build, form, lambda p: t.AUTOCALL_ROUTES[route](
                    kind).calculate_greeks(p, t.env(100.0, t._on(day)))[key]))

for route in t.AUTOCALL_ROUTES:
    for form in ("dated", "numeric"):
        def build(f):
            return t._aged(t.AUTOCALL_PRODUCTS["snowball"](f), f, 45)

        def stats(p, route=route):
            return t.AUTOCALL_ROUTES[route]("snowball", statistics="flat_vol_surrogate").calculate_event_stats(
                p, t.env(100.0, t._on(45)))
        row(f"stats:{route}-snowball-d45", form, "pv", one(build, form, lambda p: stats(p).pv))
        row(f"stats:{route}-snowball-d45", form, "sum_ko_probability",
            one(build, form, lambda p: sum(stats(p).ko_probability)))

# -- test_vol_dated_schedules: vol-model barrier MC / PDE --------------------------------------------------------
for route, kind, day in t.BARRIER_CASES:
    cell = f"barrier:{route}-{kind}-d{day}"
    factory, grid = t.BARRIER_ROUTES[route]
    for form in ("dated", "numeric"):
        def build(f, kind=kind, day=day):
            return t._barrier_aged(kind, f, day)
        e = t.env(100.0, t._on(day), grid=grid)
        row(cell, form, "price", one(build, form, lambda p: factory().price(p, e)))
        if (route, kind, day) in t._BARRIER_GREEK_CASES:
            for key in ("price", "delta", "gamma"):
                row(cell, form, f"greeks_{key}", one(build, form, lambda p: factory().calculate_greeks(p, e)[key]))

# -- test_expiry_carried_knock_in: the day-before cells whose pinned refusal W12 removed -------------------------
for route, factory, state_ok, kind, build, hand, spot in c._BARRIER_BRANCH_CASES:
    if kind != "up_in_call_discrete" or route not in ("heston_mc", "heston_pde", "slv_mc", "slv_pde"):
        continue
    for form in ("dated", "numeric"):
        row(f"carried_before:{kind}-{route}-{spot:g}", form, "price", attempt(
            lambda: c._price(route, factory, build(form, True, c.BEFORE), spot, valuation=c.BEFORE)))

for kind, route, factory, hand, spot, carried in c._AUTOCALL_BRANCH:
    if kind not in ("snowball", "phoenix_plain") or route not in ("heston_pde", "slv_pde"):
        continue
    build = c._autocallable(kind)
    for form in ("dated", "numeric"):
        row(f"carried_before:{kind}-{route}-{spot:g}", form, "price", attempt(
            lambda: c._price(route, factory, build(form, True, c.BEFORE), spot, valuation=c.BEFORE)))
