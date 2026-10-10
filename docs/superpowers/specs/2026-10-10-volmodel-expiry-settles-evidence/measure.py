"""W9 measurement: every vol-model European / barrier cell on the exercise date, price and greeks, in both forms.

Prints one row per (cell, form, quantity): the value, or ``RAISES <type>: <message>``. Run it on a tree and diff two
runs (``summarize.py``) to list the moved values. The cells are ``test/test_dated_expiry_settles.py``'s: the W8
families' vol-model routes (European call, up-and-out call) and the W9 section's ``VOL_CASES``.

    env PYTHONPATH=<wt>:<wt>/test <venv>/bin/python \
        docs/superpowers/specs/2026-10-10-volmodel-expiry-settles-evidence/measure.py > <out>.txt
"""
from __future__ import annotations

import warnings

from test_dated_expiry_settles import BARRIER, EUROPEAN, VOL_CASES, _vol_env, env_for

warnings.simplefilter("ignore")

#: the W8 families' vol-model routes (pinned by test_dated_contract_on_its_exercise_date_prices_...)
W8_VOL = {"european_call": ("heston_analytical", "lv_mc", "heston_mc", "heston_pde", "slv_mc", "slv_pde"),
          "barrier_up_out_call": ("lv_mc", "heston_mc", "heston_pde", "slv_mc", "slv_pde")}


def _run(fn):
    try:
        out = fn()
    except Exception as exc:  # noqa: BLE001 -- the measurement records the refusal
        return f"RAISES {type(exc).__name__}: {str(exc).splitlines()[0][:90]}"
    if isinstance(out, dict):
        return "{" + ", ".join(f"{k}: {float(v)!r}" for k, v in sorted(out.items())) + "}"
    return repr(float(out))


def main():
    print("cell | form | quantity | value")
    for family in (EUROPEAN, BARRIER):
        for name, factory in family.engines:
            if name not in W8_VOL[family.name]:
                continue
            for spot in family.spots:
                for form in ("dated", "numeric"):
                    cell = f"w8:{family.name}-{name}-{spot:g}"
                    value = _run(lambda: factory().price(family.build(form), env_for(name, spot)))
                    print(f"{cell} | {form} | price | {value}")
    for route, vol, _bsm, kind, build, spot in VOL_CASES:
        for form in ("dated", "numeric"):
            cell = f"w9:{route}-{kind}-{spot:g}"
            print(f"{cell} | {form} | price | {_run(lambda: vol().price(build(form), _vol_env(spot)))}")
            print(f"{cell} | {form} | greeks | {_run(lambda: vol().calculate_greeks(build(form), _vol_env(spot)))}")


if __name__ == "__main__":
    main()
