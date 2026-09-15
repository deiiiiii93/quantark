"""Reference convergence evidence for the frozen budgets (Plan 2 Task 1, Step 4).

Runs the reference at the design ladder for the fixture snowball, near both the
KO and the KI barrier, under three profiles, and prints the worst reference
uncertainty per horizon against the budgets. Usage (from the repository root,
worktree source first on the path):

    python -m intraday.reference.evidence      # with test/ on sys.path
"""
from __future__ import annotations

import sys
import time as _time
from datetime import datetime, time, timedelta, timezone
from math import exp, sqrt

from quantark.intraday import EventKind, Fixing, TradingSession, TradingSessionCalendar, VarianceProfile, resolve_context
from quantark.intraday.request import IntradayValuationRequest
from quantark.util.calendar import CalendarType, create_calendar

from intraday.conftest import dated_snowball, flat_env
from intraday.reference import budgets
from intraday.reference.budgets import GATE_C_POINTS
from intraday.reference.gaussian_reference import reference_snowball

SHANGHAI = timezone(timedelta(hours=8))
T0 = datetime(2026, 3, 16)
HORIZONS = (timedelta(days=1), timedelta(hours=6), timedelta(hours=1), timedelta(minutes=15), timedelta(minutes=5),
            timedelta(minutes=1), timedelta(seconds=10), timedelta(seconds=1))
NOTIONAL = 100.0


def main(points=GATE_C_POINTS) -> int:
    cal = create_calendar(CalendarType.CHINA_SSE, year_range=(2026, 2028))
    sse = TradingSessionCalendar(name="SSE", tz=SHANGHAI, calendar=cal,
                                 sessions=(TradingSession(time(9, 30), time(11, 30)), TradingSession(time(13, 0), time(15, 0))))
    profiles = {
        "uniform": VarianceProfile.uniform(sse, 244, reference_date=datetime(2026, 9, 15).date()),
        "desk": VarianceProfile("desk", "1", 244, 0.25, (0.35, 0.35), (0.05,)),
        "sessions_only": VarianceProfile.sessions_only(sse, 244),
    }
    product = dated_snowball(cal, T0)
    probe = resolve_context(IntradayValuationRequest(product=product, pricing_env=flat_env(datetime(2026, 9, 1, tzinfo=SHANGHAI)),
                                                     session_calendar=sse, variance_profile=profiles["desk"]))
    kos = [e for e in probe.timeline.events if e.kind is EventKind.KO]
    fixing = kos[5]
    fixings = tuple(Fixing(k.timestamp, 100.0) for k in kos[:5])
    worst = {h: {"price": 0.0, "delta": 0.0, "gamma": 0.0} for h in HORIZONS}
    budget_hit = {h: {"price": 0.0, "delta": 0.0, "gamma": 0.0} for h in HORIZONS}
    started = _time.perf_counter()
    for name, profile in profiles.items():
        for horizon in HORIZONS:
            ts = fixing.timestamp - horizon
            base = resolve_context(IntradayValuationRequest(product=product, pricing_env=flat_env(ts), session_calendar=sse,
                                                            variance_profile=profile, fixings=fixings))
            sw = sqrt(max(float(base.pricing_env.vol_surface.total_variance(100.0, base.numerical.event_taus[fixing.event_id], 100.0)), 0.0))
            for barrier in (103.0, 75.0):
                offsets = [barrier * (1 + e) for e in (-1e-3, -1e-4, 1e-4, 1e-3)]
                offsets += [barrier * exp(k * sw) for k in (-2, -1, -0.5, 0.5, 1, 2)] if sw > 0.0 else []
                for spot in offsets:
                    ctx = resolve_context(IntradayValuationRequest(product=product, pricing_env=flat_env(ts, spot=spot),
                                                                   session_calendar=sse, variance_profile=profile, fixings=fixings))
                    ref = reference_snowball(ctx, points=points)
                    for measure, unc, allowed in (
                            ("price", ref.uncertainty_price, budgets.price_budget(NOTIONAL)),
                            ("delta", ref.uncertainty_delta, budgets.delta_budget(ref.delta, spot, NOTIONAL)),
                            ("gamma", ref.uncertainty_gamma, budgets.gamma_budget(ref.gamma, spot, NOTIONAL))):
                        if unc == unc:   # not NaN
                            worst[horizon][measure] = max(worst[horizon][measure], unc)
                            ratio = budgets.REFERENCE_UNCERTAINTY_MULTIPLIER * unc / allowed
                            budget_hit[horizon][measure] = max(budget_hit[horizon][measure], ratio)
        print(f"profile {name} done ({_time.perf_counter() - started:.0f}s)", file=sys.stderr)
    print("| horizon | max unc price | max unc delta | max unc gamma | worst 3*unc/budget (price, delta, gamma) |")
    print("|---|---|---|---|---|")
    for h in HORIZONS:
        w, r = worst[h], budget_hit[h]
        print(f"| {h} | {w['price']:.2e} | {w['delta']:.2e} | {w['gamma']:.2e} | {r['price']:.3f}, {r['delta']:.3f}, {r['gamma']:.3f} |")
    print(f"\nwall time {_time.perf_counter() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
