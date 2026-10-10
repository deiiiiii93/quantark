"""Measure the W18b knock-in record-payoff cells on one tree: every route of
``test/test_knock_in_record_payoff_refused.py``, both forms, day 10.

Groups (cell prefix):
  kipay    knock-in, records paying 4.5 != product.rebate 3     -- price -> refused (routes off the 1D core)
  kireb    knock-in, records paying product.rebate 3            -- must not move
  kinone   knock-in, records without a payoff                   -- must not move
  kopay    knock-out, records paying 4.5                        -- must not move

    env PYTHONPATH=<tree>:<branch>/test <venv>/bin/python measure_w18b_ki.py > <out>.txt
"""
from __future__ import annotations

import warnings

warnings.simplefilter("ignore")

import quantark  # noqa: E402
import test_knock_in_record_payoff_refused as t  # noqa: E402


def attempt(fn):
    try:
        return repr(float(fn()))
    except Exception as exc:  # noqa: BLE001 -- the evidence records the refusal
        return f"RAISES {type(exc).__name__}: {str(exc).splitlines()[0][:110]}"


print(f"# quantark from {quantark.__file__}")
for route in t.ROUTES:
    for form in ("dated", "numeric"):
        for kind in t._KI:
            for group, payoff in (("kipay", 4.5), ("kireb", 3.0), ("kinone", None)):
                value = attempt(lambda: t._price(route, t._product(kind, form, payoff=payoff)))
                print(f"{group}:{route}-{kind}-d10 | {form} | price | {value}")
        for kind in t._KO:
            value = attempt(lambda: t._price(route, t._product(kind, form, payoff=4.5)))
            print(f"kopay:{route}-{kind}-d10 | {form} | price | {value}")
