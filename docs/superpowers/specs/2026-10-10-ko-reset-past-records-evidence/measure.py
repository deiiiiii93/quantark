"""W11 measurement: every KO-reset / snowball / Phoenix twin value the W11 tests touch, on one library tree.

Run with PYTHONPATH=<library root>:<worktree>/test. Prints ``key\tvalue`` (repr of the float, or ``raised: <type>``).
"""
import sys
import warnings
from datetime import timedelta

warnings.simplefilter("ignore")

import quantark
import test_ko_reset_past_records as W
import test_dated_expiry_settles as D
import test_expiry_carried_knock_in as C

out = open(sys.argv[1], "w")
out.write(f"# library: {quantark.__file__}\n")


def emit(key, fn):
    try:
        value = fn()
        if isinstance(value, dict):
            for name in ("price", "delta", "gamma"):
                out.write(f"{key}|{name}\t{float(value[name])!r}\n")
            return
        out.write(f"{key}\t{float(value)!r}\n")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).replace("\n", " ")
        tag = "end_date-after-start_date" if "must be after start_date" in msg else msg[:80]
        out.write(f"{key}\traised: {type(exc).__name__}: {tag}\n")


def greeks(factory, product, e):
    return W._quiet(lambda: factory().calculate_greeks(product, e))


# A. the W11 twins (test_ko_reset_past_records): price, delta, gamma per form
for route, factory in W.KO_RESET_ROUTES:
    for day, ki, spot in W.KO_RESET_STATES:
        for form in ("dated", "numeric"):
            emit(f"A|ko_reset|{route}|d{day}|ki{ki}|{spot:g}|{form}",
                 lambda: greeks(factory, W.ko_reset(form, day, ki), W.env(spot, day, route)))
    for day in (W.SCHEDULE_1_END + 1, 292):
        for form in ("dated", "numeric"):
            emit(f"A|ko_reset|{route}|d{day}|kiNone|100|{form}|price",
                 lambda: W._quiet(lambda: factory().price(W.ko_reset(form, day), W.env(100.0, day, route))))
for kind, routes, build in (("snowball", W.SNOWBALL_ROUTES, W.snowball), ("phoenix", W.PHOENIX_ROUTES, W.phoenix)):
    for route, factory in routes:
        for day, ki, spot in W.AUTOCALL_STATES:
            for form in ("dated", "numeric"):
                emit(f"A|{kind}|{route}|d{day}|ki{ki}|{spot:g}|{form}",
                     lambda: greeks(factory, build(form, day, ki), W.env(spot, day, route)))

# B. test_dated_expiry_settles: the tracker-clock KO-reset (knocked in), exercise date and 292 days in
for name, factory in D.KO_RESET.engines:
    for valuation, label in ((D.VALUATION, "exercise"), (D.INITIAL + timedelta(days=292), "d292")):
        for spot in D.KO_RESET.spots:
            for form in ("dated", "numeric"):
                def run():
                    product = D.KO_RESET.build("dated")
                    if form == "numeric":
                        product = D._autocallable("ko_reset_absolute", knocked_in=True)("numeric") \
                            if label == "exercise" else None
                    if product is None:   # the numeric twin aged to 292 days
                        import pandas as pd
                        from test_tracker_clock_parity import KINDS
                        product = KINDS["ko_reset_absolute"]("numeric", pd.Timestamp(D.INITIAL))
                        product._otc_lifecycle_knocked_in = True
                        product._otc_lifecycle_knock_in_date = D.INITIAL + timedelta(days=152)
                        product.time_shift(292 / 365.0, valuation, D.env(100.0), keep_landing=True)
                    return W._quiet(lambda: factory().price(product, D.env_for(name, spot, valuation)))
                emit(f"B|ko_reset_knocked_in|{name}|{label}|{spot:g}|{form}", run)

# C. test_expiry_carried_knock_in: KO-reset routes, exercise date and the day before, flag and lifecycle state
build = C._autocallable("ko_reset_absolute")
for route, factory in C._KO_RESET_ROUTES:
    for spot in (101.0, 90.0, 104.0):
        for form in ("dated", "numeric"):
            emit(f"C|ko_reset|{route}|exercise|flag|{spot:g}|{form}",
                 lambda: C._price(route, factory, build(form, True), spot))
            emit(f"C|ko_reset|{route}|day_before|flag|{spot:g}|{form}",
                 lambda: C._price(route, factory, build(form, True, C.BEFORE), spot, valuation=C.BEFORE))
            if route == "quad_v2":
                emit(f"C|ko_reset|{route}|exercise|state|{spot:g}|{form}",
                     lambda: C._price(route, factory, build(form, False), spot,
                                      lifecycle_state=C.AutocallableLifecycleState(knocked_in=True)))
out.close()
