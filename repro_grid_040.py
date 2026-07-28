"""Regression check for three quantark 0.4.0 grid-layer defects (all fixed).

Guards the fixes made on 2026-07-28. Exits non-zero if any regresses.

    .venv/bin/python repro_grid_040.py

No external data: a real OTC book position is reconstructed in full — a 2.6y
部分保本 Phoenix on ¥50m notional, 32 discrete KO observations (0.84% instant
coupon, 50% protection), discrete KI at maturity — and priced against a
converged PhoenixQuadEngine reference.

What each check guards, and what it looked like when broken:

  1. Every accuracy profile prices the position sanely.
     Broken: accuracy="standard" (the DEFAULT) returned +165,132 against a
     true -985,774 — a sign flip, +116.75%, gamma 19x too large — while
     `fast` and `high` were both within ~3%. Only a logger.warning fired.

  2. An unreachable critical price does not move the grid domain.
     Broken: _auto_bounds expanded per critical price with no reachability
     test, so a caller's "can never knock out" sentinel at 100x initial
     stretched the upper edge 71,468 -> 844,901 and spent ~37% of the nodes
     above the 4-sigma envelope.

  3. Solve cost is monotone in `points`.
     Broken: cost climbed 8.95s (points=1001) -> 26.71s (points=3000) then
     collapsed to 0.76s (points=3500) — a 35x speedup from asking for MORE
     nodes — with overflow/divide-by-zero RuntimeWarnings out of _ode_f in
     _concentrated_mesh's beta bracket search.
"""
from __future__ import annotations

import datetime
import logging
import sys
import time
import warnings

import numpy as np

logging.basicConfig(level=logging.WARNING, format="    [log] %(message)s")

# --- thresholds ------------------------------------------------------------
# ACC_TOL is the one number here that is a policy call rather than a fact.
# It is set loose enough to survive legitimate numerical drift and tight
# enough to catch a blow-up: the defect it guards was +116.75%, and the
# fixed build lands every profile within 0.17% on this position.
ACC_TOL = 0.01        # max |PV error| vs the quad reference, any profile
COST_SLACK = 2.0      # a smaller grid may not cost more than this x a larger one

_RUNTIME_WARNINGS: list[str] = []


def _record(message, category, filename, lineno, file=None, line=None):
    if issubclass(category, RuntimeWarning):
        _RUNTIME_WARNINGS.append(f"{filename.split('/')[-1]}:{lineno} {message}")


warnings.simplefilter("always", RuntimeWarning)
warnings.showwarning = _record

from quantark.asset.equity.engine.pde import PhoenixPDESolver
from quantark.asset.equity.engine.pde.grid import GridConfig
from quantark.asset.equity.engine.pde.grid.config import resolve_config
from quantark.asset.equity.engine.pde.grid.request import GridRequest, MarketSnapshot
from quantark.asset.equity.engine.pde.grid.space import build_space
from quantark.asset.equity.engine.quad import PhoenixQuadEngine
from quantark.asset.equity.param import PDEParams, QuadParams
from quantark.asset.equity.product.option import PhoenixOption
from quantark.asset.equity.product.option.phoenix_config import CouponBarrierConfig
from quantark.asset.equity.product.option.snowball_config import (
    AccrualConfig,
    BarrierConfig,
    PayoffConfig,
)
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.calendar import CalendarType, create_calendar
from quantark.util.calendar.day_counter import DayCountConvention
from quantark.asset.equity.product.option.observation_schedule import (
    ObservationRecord,
    ObservationSchedule,
)
from quantark.util.enum import ObservationType
from quantark.util.enum.option_enums import CouponPayType, ProtectionType

# ---------------------------------------------------------------------------
# A real book position: 2.6y Phoenix, 32 monthly KO observations, instant
# coupon, discrete KI at maturity only. The FIRST ko_barrier is 100x initial —
# the caller's "this observation already passed, can never knock out" sentinel.
# ---------------------------------------------------------------------------
SPOT, VOL, RATE, DIV = 8601.4075, 0.27334, 0.014, 0.10658216
INITIAL = STRIKE = 8323.41
TAU = 2.6352459016393444
KI_BARRIER = 5410.22
COUPON_BARRIER, COUPON_RATE = 6658.73, 0.0084
NOTIONAL = 50_000_000.0
MULTIPLIER = NOTIONAL / INITIAL  # 6007.153318171278
PROTECTION_RATE = 0.5
VALUATION_DATE = datetime.datetime(2026, 6, 26)
INITIAL_DATE = datetime.datetime(2026, 2, 12)

KO_TIMES = [
    0.045081967213114756, 0.13524590163934427, 0.22950819672131148,
    0.28688524590163933, 0.38114754098360654, 0.4713114754098361,
    0.5532786885245902, 0.6270491803278688, 0.7049180327868853,
    0.7868852459016393, 0.8647540983606558, 0.9549180327868853,
    1.0368852459016393, 1.1311475409836065, 1.221311475409836,
    1.2827868852459017, 1.3770491803278688, 1.4631147540983607,
    1.5491803278688525, 1.6188524590163935, 1.7008196721311475,
    1.7827868852459017, 1.860655737704918, 1.9426229508196722,
    2.0327868852459017, 2.127049180327869, 2.2131147540983607,
    2.2827868852459017, 2.372950819672131, 2.459016393442623,
    2.5491803278688523, TAU,
]
KO_BARRIERS = [
    INITIAL * 100.0,  # <-- unreachable sentinel (already-passed observation)
    8323.41, 8281.79, 8240.18, 8198.56, 8156.94, 8115.32, 8073.71, 8032.09,
    7990.47, 7948.86, 7907.24, 7865.62, 7824.01, 7782.39, 7740.77, 7699.15,
    7657.54, 7615.92, 7574.3, 7532.69, 7491.07, 7449.45, 7407.83, 7366.22,
    7324.6, 7282.98, 7241.37, 7199.75, 7158.13, 7116.52, 5410.22,
]


def _schedule(times, barriers, rate=0.0):
    return ObservationSchedule(
        records=[
            ObservationRecord(observation_time=t, barrier=b, return_rate=rate)
            for t, b in zip(times, barriers)
        ]
    )


def build_product(ko_barriers):
    return PhoenixOption(
        initial_price=INITIAL,
        strike=STRIKE,
        barrier_config=BarrierConfig(
            ko_barrier=list(ko_barriers),
            ko_rate=[0.0] * len(KO_TIMES),
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=list(KO_TIMES),
            ko_observation_schedule=_schedule(KO_TIMES, ko_barriers),
            ki_barrier=KI_BARRIER,
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=[TAU],
            ki_observation_schedule=_schedule([TAU], [KI_BARRIER]),
            ki_continuous=False,
        ),
        coupon_config=CouponBarrierConfig(
            coupon_barrier=[COUPON_BARRIER] * len(KO_TIMES),
            coupon_rate=COUPON_RATE,
            coupon_pay_type=CouponPayType.INSTANT,
            memory_coupon=False,
            day_count_convention=DayCountConvention.BUSINESS_DAYS,
        ),
        payoff_config=PayoffConfig(
            include_principal=False,
            participation_rate=1.0,
            protection_type=ProtectionType.PARTIAL,
            protection_rate=PROTECTION_RATE,
        ),
        accrual_config=AccrualConfig(
            is_annualized=True,
            accrual_factors=[1.0] * len(KO_TIMES),
        ),
        contract_multiplier=MULTIPLIER,
        maturity=TAU,
        initial_date=INITIAL_DATE,
        annualization_day_count=DayCountConvention.BUSINESS_DAYS,
    )


def build_env():
    return PricingEnvironment(
        valuation_date=VALUATION_DATE,
        spot_quote=SpotQuote(spot=SPOT, asset_name="IDX"),
        vol_surface=FlatVolSurface(volatility=VOL),
        rate_curve=FlatRateCurve(rate=RATE),
        div_yield=ContinuousDividendYield(div_yield=DIV),
        day_count_convention=DayCountConvention.BUSINESS_DAYS,
        bus_days_in_year=244,
        calendar=create_calendar(CalendarType.CHINA_SSE, (2020, 2035)),
    )


def pde(product, env, **grid):
    params = PDEParams(
        accuracy=grid.pop("accuracy", "high"),
        grid=GridConfig(**grid) if grid else None,
        bus_days_in_year=244,
        rannacher_steps=2,
    )
    t0 = time.perf_counter()
    pv = float(PhoenixPDESolver(params=params).price(product, env))
    return pv, time.perf_counter() - t0


def main():
    product, env = build_product(KO_BARRIERS), build_env()
    failures: list[str] = []

    print("=" * 78)
    print("REFERENCE — PhoenixQuadEngine")
    print("=" * 78)
    ref = None
    for pts in (8001, 32001):
        v = float(PhoenixQuadEngine(params=QuadParams(grid_points=pts, bus_days_in_year=244))
                  .price(product, env))
        drift = "" if ref is None else f"  ({(v - ref) / abs(ref):+.3%} vs pts=8001)"
        print(f"  grid_points={pts:<6} PV = {v:>15.7g}{drift}")
        ref = v
    print(f"\n  reference PV = {ref:.7g}")
    print("  NOTE: re-derive this if the quad engines change — QuadParams defaults")
    print("        (integration_rule, filter_unreachable_barriers, event_projection)")
    print("        moved it once already.\n")

    print("=" * 78)
    print(f"CHECK 1 — every accuracy profile prices within {ACC_TOL:.1%} of the reference")
    print("=" * 78)
    for acc in ("fast", "standard", "high"):
        pv, dt = pde(product, env, accuracy=acc)
        err = (pv - ref) / abs(ref)
        ok = abs(err) <= ACC_TOL
        print(f"  accuracy={acc:<9} PV = {pv:>15.7g}   err = {err:+8.2%}   {dt:5.2f}s   "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"accuracy={acc!r} off reference by {err:+.2%} (limit {ACC_TOL:.1%})")
    print()

    print("=" * 78)
    print("CHECK 2 — an unreachable critical price does not move the domain")
    print("=" * 78)
    mkt = MarketSnapshot(spot=SPOT, sigma_ref=VOL, r_ref=RATE, q_ref=DIV)
    cfg = resolve_config("standard", None)
    real = tuple(b for b in KO_BARRIERS if b < INITIAL * 10) + (COUPON_BARRIER, KI_BARRIER)
    spans = {}
    for label, crits in (("real barriers only", real),
                         ("+ sentinel 832341.0", (INITIAL * 100.0,) + real)):
        req = GridRequest(tau=TAU, bound_anchors=(SPOT, STRIKE), critical_prices=crits,
                          hard_lower=None, hard_upper=None, event_times=())
        x = build_space(req, mkt, cfg).x
        spans[label] = (float(x[0]), float(x[-1]))
        print(f"  {label:<22} domain = [{np.exp(x[0]):>8.1f}, {np.exp(x[-1]):>10.1f}]  "
              f"log-span = {x[-1] - x[0]:.2f}")
    lo_a, hi_a = spans["real barriers only"]
    lo_b, hi_b = spans["+ sentinel 832341.0"]
    ok = abs(lo_a - lo_b) < 1e-9 and abs(hi_a - hi_b) < 1e-9
    print(f"  -> domains identical: {'PASS' if ok else 'FAIL'}\n")
    if not ok:
        failures.append(
            f"sentinel moved the domain: [{np.exp(lo_a):.1f}, {np.exp(hi_a):.1f}] -> "
            f"[{np.exp(lo_b):.1f}, {np.exp(hi_b):.1f}]"
        )

    print("=" * 78)
    print(f"CHECK 3 — solve cost is monotone in points (slack {COST_SLACK:g}x)")
    print("=" * 78)
    seen: list[tuple[int, float]] = []
    for pts in (1001, 2000, 2500, 3000, 3500, 4000, 6000):
        pv, dt = pde(product, env, points=pts, steps_per_day=8.0, max_points=8000)
        print(f"  points={pts:<6} {dt:>6.2f}s   err = {(pv - ref) / abs(ref):+7.2%}")
        seen.append((pts, dt))
    inversions = [
        f"points={p_small} took {t_small:.2f}s but points={p_big} took only {t_big:.2f}s"
        for p_small, t_small in seen
        for p_big, t_big in seen
        if p_big > p_small and t_small > t_big * COST_SLACK
    ]
    print(f"  -> monotone: {'PASS' if not inversions else 'FAIL'}")
    failures.extend(inversions)

    grid_rw = [w for w in _RUNTIME_WARNINGS
               if "space.py" in w or "overflow" in w or "divide by zero" in w]
    print(f"  -> no overflow/divide-by-zero from the grid layer: "
          f"{'PASS' if not grid_rw else 'FAIL'}")
    for w in dict.fromkeys(grid_rw):
        print(f"       {w}")
    failures.extend(dict.fromkeys(grid_rw))

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED — {len(failures)} regression(s)")
        for f in failures:
            print(f"  - {f}")
    else:
        print("PASSED — all three fixes hold")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
