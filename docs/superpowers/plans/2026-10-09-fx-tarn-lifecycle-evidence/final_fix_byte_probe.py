"""Final-review fix pass (Critical 1) byte probe: date-only dated TARN/TARF scenarios (prices, greeks, resolve views, record events/states, FX backtest
and dynamic rows). Prints one repr line per value; run before and after the Critical 1 fix and compare sha1.
Run: env PYTHONPATH=<wt>:<wt>/test python final_fix_byte_probe.py out.txt"""
import hashlib
import re
import sys
import warnings
from datetime import datetime, timedelta

import pandas as pd

warnings.simplefilter("ignore")
from fx_target_redemption_fixtures import (  # noqa: E402
    FIX_DATES, LAG2, TARGET_CAL, VAL, _numeric_twin, book, dated_note, dated_tarf, engine_for, env, following,
)
from quantark.backtest import FXBacktestConfig, FXBacktestEngine  # noqa: E402
from quantark.backtest.strategy import FXDeltaNeutralStrategy  # noqa: E402
from quantark.backtest.transaction_costs import ZeroCostModel  # noqa: E402
from quantark.dynamicscenario import FXDynamicScenarioConfig, FXDynamicScenarioEngine  # noqa: E402
from quantark.dynamicscenario.path.day_path import DayPath, DayStep, ParameterChange  # noqa: E402
from quantark.stresstest.stress.stress_types import StressType  # noqa: E402
from quantark.util.calendar import DayCountConvention as DC  # noqa: E402

out = []


def emit(*parts):
    out.append(" | ".join(repr(p) for p in parts))


def safe(label, fn):
    try:
        emit(label, fn())
    except Exception as ex:  # refusals are part of the bytes
        emit(label, "RAISE", type(ex).__name__, str(ex))


def make(kind, dates, lag):
    pays = [d + timedelta(days=lag) for d in dates]
    if kind == "tarn":
        return dated_note(fixing_dates=dates, pay_dates=pays, target=3.0)
    if kind == "tarn_np":
        return dated_note(fixing_dates=dates, pay_dates=pays, target=1.5, include_principal=False)
    return dated_tarf(fixing_dates=dates, pay_dates=pays, target=0.20)


def _kw(prod):
    from quantark.asset.fx.product.option import FxTargetRedemptionNote
    if isinstance(prod, FxTargetRedemptionNote):
        return dict(fixing_dates=prod.fixing_dates, pay_dates=prod.pay_dates, coupon_rate=prod.coupon_rate,
                    notional=prod.notional, strike=prod.strike, currency_pair=prod.currency_pair, target=prod.target,
                    include_principal=prod.include_principal, initial_date=VAL, state=prod.state)
    return dict(fixing_dates=prod.fixing_dates, pay_dates=prod.pay_dates, strike=prod.strike, notional=prod.notional,
                currency_pair=prod.currency_pair, target=prod.target, state=prod.state)


DCS = {"cal": (DC.CALENDAR_DAYS, None), "act": (DC.ACT_365, None), "bus": (DC.BUSINESS_DAYS, TARGET_CAL)}
OFFSETS = [-1, 0, 1, 2, 3, 4, 30, 31, 32, 33, 62, 90]
for kind in ["tarn", "tarn_np", "tarf"]:
    for lag in [0, 2]:
        for dname, (dc, cal) in DCS.items():
            dates = [following(d) for d in FIX_DATES] if dname == "bus" else FIX_DATES
            prod = make(kind, dates, lag)
            prev = VAL
            for off in OFFSETS:
                day = dates[0] + timedelta(days=off)
                e = env(val=day, dc=dc, calendar=cal, spot=1.20 + 0.002 * (off % 7) - 0.01 * (off > 30))
                # age day by day through the recorder hook (record_passed_fixings) up to this valuation
                d = prev
                while d < day:
                    d += timedelta(days=1)
                    safe(("rec", kind, lag, dname, str(d)), lambda: prod.record_passed_fixings(
                        env(val=d, dc=dc, calendar=cal, spot=e.spot)))
                prev = day
                tag = (kind, lag, dname, off)
                emit(tag, "state", prod.state)
                emit(tag, "events", prod.lifecycle_events)
                safe((tag, "view"), lambda: prod.resolve(e, warn_missing_fixings=False))
                safe((tag, "price"), lambda: engine_for(prod, n=2_000).price(prod, e))
                if off in (0, 1, 2, 31):
                    safe((tag, "greeks"), lambda: sorted(engine_for(prod, n=2_000).calculate_greeks(prod, e).items()))
                    safe((tag, "twin"), lambda: engine_for(prod, n=2_000).price(_numeric_twin(prod, e), e))
                # a booking rebuilt from the state, priced at the same instant
                safe((tag, "rebook"), lambda: engine_for(prod, n=2_000).price(make(kind, dates, lag).__class__(
                    **_kw(prod)), e))


def _path(index, spots):
    return pd.DataFrame({"EURUSD_spot": spots, "EURUSD_vol": 0.10, "EURUSD_dom_rate": 0.05,
                         "EURUSD_for_rate": 0.03}, index=index)


def backtest(pf, path):
    eng = FXBacktestEngine(FXBacktestConfig(
        portfolio=pf, market_path=path,
        strategy=FXDeltaNeutralStrategy(delta_threshold=1e12, rebalance_frequency="on_threshold"),
        transaction_cost_model=ZeroCostModel(), calculate_greeks=False))
    return eng.run().rows


for dname in ["cal", "bus"]:
    dc, cal = DCS[dname]
    e0 = env(dc=dc, calendar=cal)
    dates = [following(d) for d in FIX_DATES] if dname == "bus" else FIX_DATES
    prods = [make("tarn", dates, 2), make("tarf", dates, 0), dated_note(fixing_dates=dates, target=1.5),
             dated_tarf(fixing_dates=dates, pay_dates=[d + timedelta(days=2) for d in dates], target=0.075,
                        gearing=2.0)]
    prods += [_numeric_twin(p, e0) for p in prods]
    index = pd.date_range(VAL, periods=100, freq="D") if dname == "cal" else pd.bdate_range(VAL, periods=75)
    spots = [1.20 + 0.003 * ((k * 7) % 11 - 5) for k in range(len(index))]
    for p in prods:
        pf, _ = book(p, fx_env=e0)
        try:
            rows = backtest(pf, _path(index, spots))
            for r in rows:
                emit("bt", dname, type(p).__name__, p.is_numeric, r["timestamp"], r["portfolio_value"],
                     r["paid_cash"], r["pending_pv"])
        except Exception as ex:
            emit("bt", dname, "RAISE", type(ex).__name__, str(ex))

    pf, _ = book(*prods[:4], fx_env=e0)
    steps = [DayStep(day_index=i, changes=[ParameterChange("spot", StressType.VALUE, 1.20 + 0.002 * (i % 5))])
             for i in range(70)]
    try:
        res = FXDynamicScenarioEngine(FXDynamicScenarioConfig(calculate_greeks=False)).run(
            pf, DayPath(name="p", steps=steps, start_date=VAL))
        for dr in res.day_results:
            emit("dyn", dname, dr.portfolio_value, dr.paid_cash, dr.pending_receivable_pv, dr.lifecycle_events)
    except Exception as ex:
        emit("dyn", dname, "RAISE", type(ex).__name__, str(ex))

text = re.sub(r"position_id='[0-9a-f-]+'", "position_id=X", "\n".join(out) + "\n")  # uuid4 ids vary per run
open(sys.argv[1], "w").write(text)
print(len(out), "lines sha1", hashlib.sha1(text.encode()).hexdigest())
