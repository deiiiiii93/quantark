"""Q7 measurement (plan 2026-10-09-fx-tarn-lifecycle, Task 8): for every BaseFxEngine._fdm_theta call on a NUMERIC FX
product, price theta with the current step (theta_days / 365) and with the Q7 step (the environment's year fraction of
theta_days calendar days) and log every call where the two steps differ. Equal steps give bitwise-equal theta (same
code path), so only differing steps are logged.

Controller run (full suite, before or after the Q7 change - both thetas are computed explicitly):
  env PYTHONPATH=<WT>:<WT>/test:<WT>/docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence \
      Q7_LOG_DIR=<WT>/.superpowers/w4b/q7 /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test \
      -p q7_theta_plugin -n 6 -q
"""
import json
import os
from copy import deepcopy
from datetime import timedelta

import pytest

from quantark.asset.fx.engine.base_fx_engine import BaseFxEngine
from quantark.util.calendar import calculate_year_fraction

_LOG_DIR = os.environ.get("Q7_LOG_DIR")
_ORIG = BaseFxEngine._fdm_theta
_CURRENT = {"nodeid": "<collection>"}


def _theta_with_step(engine, product, fx_env, base_price, dt):
    shifted = deepcopy(product)
    shifted.time_shift(dt)
    shifted.record_passed_fixings(fx_env)
    return (engine.price(shifted, fx_env) - base_price) / engine.params.theta_days


def _spy(self, product, fx_env, base_price):
    result = _ORIG(self, product, fx_env, base_price)
    if _LOG_DIR and product.maturity is not None:
        days = self.params.theta_days
        val = fx_env.valuation_date
        old_dt = days / 365.0
        new_dt = calculate_year_fraction(val, val + timedelta(days=days), fx_env.day_count_convention,
                                         fx_env.bus_days_in_year, calendar=fx_env.calendar)
        if new_dt != old_dt:
            record = {
                "nodeid": _CURRENT["nodeid"], "engine": type(self).__name__, "product": type(product).__name__,
                "convention": fx_env.day_count_convention.name, "valuation_date": str(val),
                "theta_days": days, "old_dt": old_dt, "new_dt": new_dt,
                "old_theta": _theta_with_step(self, product, fx_env, base_price, old_dt),
                "new_theta": _theta_with_step(self, product, fx_env, base_price, new_dt),
            }
            os.makedirs(_LOG_DIR, exist_ok=True)
            with open(os.path.join(_LOG_DIR, f"{os.getpid()}.jsonl"), "a") as fh:
                fh.write(json.dumps(record) + "\n")
    return result


BaseFxEngine._fdm_theta = _spy


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _CURRENT["nodeid"] = item.nodeid
