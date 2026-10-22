"""PnL explain demo: a vanilla, a barrier and a snowball across a Friday -> Monday gap,
a knock-out day with its event row, and a portfolio explain with reconciliation.

Run:  PYTHONPATH=. python example/pnl_explain_demo.py
      PYTHONPATH=. python example/pnl_explain_demo.py --html-output example/data/pnl_explain_lecture_latest.html

Without the flag the script prints the console tables and nothing else. With it,
it also writes a self-contained teaching page -- inline CSS, hand-written SVG, no
image assets -- that walks the module in six steps: the value identity, the two
methods on one day, where Taylor breaks, why the waterfall order is a choice, the
lifecycle event row, and the book-level reconciliation gate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from html import escape
import itertools
from pathlib import Path
import sys
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantark.asset.equity.engine.analytical import (  # noqa: E402
    AmericanOptionAnalyticalEngine, BarrierAnalyticalEngine, DeltaOneEngine,
)
from quantark.asset.equity.engine.analytical.black_scholes_engine import BlackScholesEngine  # noqa: E402
from quantark.asset.equity.engine.pde import SnowballPDESolver  # noqa: E402
from quantark.asset.equity.lifecycle.autocallable import AutocallableLifecycleTracker  # noqa: E402
from quantark.asset.equity.lifecycle.events import LifecycleEventType  # noqa: E402
from quantark.asset.equity.param import PDEParams  # noqa: E402
from quantark.asset.equity.product.deltaone import Futures  # noqa: E402
from quantark.asset.equity.product.option.american_option import AmericanOption  # noqa: E402
from quantark.asset.equity.product.option.barrier_option import BarrierOption  # noqa: E402
from quantark.asset.equity.product.option.european_vanilla_option import EuropeanVanillaOption  # noqa: E402
from quantark.asset.equity.product.option.snowball_config import AccrualConfig, BarrierConfig, PayoffConfig  # noqa: E402
from quantark.asset.equity.product.option.snowball_option import SnowballOption  # noqa: E402
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote  # noqa: E402
from quantark.param.basis.basis_yield import FlatBasisYield  # noqa: E402
from quantark.param.div import ContinuousDividendYield  # noqa: E402
from quantark.pnlexplain import (  # noqa: E402
    MARKET_FACTORS, BookSnapshot, ExplainMethod, Factor, LifecycleTransition, PnLExplainConfig,
    PnLExplainResult, PortfolioExplainResult, PositionSnapshot, RowKind, ValuationSnapshot,
    explain, explain_portfolio,
)
from quantark.priceenv import PricingEnvironment  # noqa: E402
from quantark.util.enum import ObservationType, OptionType  # noqa: E402
from quantark.util.enum.option_enums import BarrierType  # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 80)
COLS = ["method", "kind", "factor", "term", "pnl", "greek", "cash_greek", "spot_return", "vol_pts", "days"]
FRI, MON = datetime(2026, 6, 26), datetime(2026, 6, 29)
DEFAULT_HTML_OUTPUT = Path(__file__).resolve().parent / "data" / "pnl_explain_lecture_latest.html"

# One real risk day on an index structured-products desk, at desk notionals. The index sits
# near a CSI 1000 level, so a barrier 10% out of the money is 600 points away, not 15.
S0, S1 = 6000.0, 5580.0             # -7.0%
V0, V1 = 0.18, 0.26                 # +8 vol points
R0, R1 = 0.0220, 0.0255             # +35 bp
# An index book is hedged with index futures, and the two carry numbers are one fact seen twice.
# The desk observes the futures basis b, annualised; the dividend yield that actually prices the
# product is implied from it by the cost-of-carry identity q = r - b. So q is never moved on its
# own here: move r and b, and q follows. Index futures sit at a discount, so b is negative, and
# the discount widens as the market sells off.
B0, B1 = -0.0300, -0.0450           # -300 bp -> -450 bp: the discount widens on the selloff
Q0, Q1 = R0 - B0, R1 - B1           # 5.20% -> 7.05%, implied, not quoted
FUT_T = 0.25                        # the front quarterly contract the desk hedges in
# The trigger sits just over the first KO barrier. A far bigger rally would knock out just
# as surely, but the alive contract's payoff becomes near-certain and its vol and dividend
# rows collapse to noise -- true, and a confusing first lesson.
KO_QUIET, KO_TRIGGER = 6000.0, 6320.0
# Units of the index. The barrier put is written 10% out of the money and the day takes spot
# to within 3.3% of its barrier, which is where its gamma stops being a rounding error.
QTY = {"vanilla": 2_000.0, "barrier": -1_500.0, "snowball": -3_400.0, "american": 1_200.0}
LABELS = {
    "vanilla": "vanilla call x2,000",
    "barrier": "down-and-out put x-1,500",
    "snowball": "snowball x-3,400",
    "american": "American put x1,200",
    "hedge": "index futures hedge",
}


def env(spot, vol, rate, div, basis, date):
    """One market state. `basis` is carried for the record and for the reader; the products and
    the futures hedge both price off `div`, which is r - b by construction."""
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=spot), vol_surface=FlatVolSurface(volatility=vol),
        rate_curve=FlatRateCurve(rate=rate), div_yield=ContinuousDividendYield(div_yield=div),
        basis_yield=FlatBasisYield(basis_yield=basis), valuation_date=date)


def snowball():
    dates = [0.25, 0.5, 0.75, 1.0]
    return SnowballOption(
        initial_price=6000.0, strike=6000.0,
        barrier_config=BarrierConfig(ko_barrier=[6300.0, 6240.0, 6180.0, 6120.0], ko_rate=0.10,
                                     ko_observation_type=ObservationType.DISCRETE, ko_observation_dates=dates,
                                     ki_barrier=4500.0, ki_observation_type=ObservationType.DISCRETE,
                                     ki_observation_dates=dates, ki_continuous=False),
        payoff_config=PayoffConfig(rebate_rate=0.0, include_principal=True),
        accrual_config=AccrualConfig(is_annualized=True),
        contract_multiplier=1.0, maturity=1.0, is_reverse=False)


def show(title, res):
    print(f"\n=== {title}: total {res.total_pnl:+,.2f}  "
          f"(waterfall gap {res.reconcile(ExplainMethod.WATERFALL):+.2e}, "
          f"taylor gap {res.reconcile(ExplainMethod.TAYLOR):+.2e}, unexplained {res.unexplained:+,.2f})")
    print(res.to_frame()[COLS].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


START = datetime(2024, 1, 1)


def find_event_day(tracker, event_type, spot_quiet, spot_trigger, max_days=400):
    d = START
    for _ in range(max_days):
        d = d + timedelta(days=1)
        probe = deepcopy(tracker)
        ev = probe.observe(pd.Timestamp(d), probe.product_for_lifecycle(),
                           env(spot_trigger, 0.22, 0.0200, 0.0520, -0.0320, d), spot_trigger)
        if any(e.event_type is event_type for e in ev):
            return d - timedelta(days=1), d
        tracker.observe(pd.Timestamp(d), tracker.product_for_lifecycle(),
                        env(spot_quiet, 0.22, 0.0200, 0.0520, -0.0320, d), spot_quiet)
    raise RuntimeError("no event")


def taylor_delta(res: PnLExplainResult) -> float:
    """The position's delta as the Taylor expansion used it: whole position, per index unit."""
    return next(r.greek for r in res.rows_for(ExplainMethod.TAYLOR) if r.term == "delta")


@dataclass
class Cases:
    """Everything the console prints, kept so the lecture page can draw the same numbers."""

    vanilla: PnLExplainResult
    barrier: PnLExplainResult
    snowball: PnLExplainResult
    american: PnLExplainResult
    hedge: PnLExplainResult
    hedge_units: int
    ko: PnLExplainResult
    ko_dates: Tuple[datetime, datetime]
    portfolio: PortfolioExplainResult
    orders: Dict[str, Dict[str, PnLExplainResult]] = field(default_factory=dict)


def build_cases(*, with_orders: bool) -> Cases:
    # --- 1. One risk day across four positions ----------------------------------------------
    e0, e1 = env(S0, V0, R0, Q0, B0, FRI), env(S1, V1, R1, Q1, B1, MON)
    bs, bar = BlackScholesEngine(), BarrierAnalyticalEngine()
    pde = SnowballPDESolver(PDEParams(accuracy="fast"))
    # The desk moved its American book from the BS93 approximation to BAW over the weekend.
    # Two engine objects, different fingerprints: that difference is the model row.
    us0 = AmericanOptionAnalyticalEngine(method="BS93")
    us1 = AmericanOptionAnalyticalEngine(method="BAW")
    days = (MON - FRI).days

    vanilla0 = EuropeanVanillaOption(strike=6000.0, option_type=OptionType.CALL, maturity=1.0)
    vanilla1 = EuropeanVanillaOption(strike=6000.0, option_type=OptionType.CALL, maturity=1.0 - days / 365)
    van = explain(ValuationSnapshot(vanilla0, bs, e0, date=FRI, quantity=QTY["vanilla"]),
                  ValuationSnapshot(vanilla1, bs, e1, date=MON, quantity=QTY["vanilla"]))
    show(LABELS["vanilla"], van)

    barrier0 = BarrierOption(strike=6000.0, option_type=OptionType.PUT, barrier=5400.0,
                             barrier_type=BarrierType.DOWN_OUT, maturity=1.0)
    barrier1 = BarrierOption(strike=6000.0, option_type=OptionType.PUT, barrier=5400.0,
                             barrier_type=BarrierType.DOWN_OUT, maturity=1.0 - days / 365)
    brr = explain(ValuationSnapshot(barrier0, bar, e0, date=FRI, quantity=QTY["barrier"]),
                  ValuationSnapshot(barrier1, bar, e1, date=MON, quantity=QTY["barrier"]))
    show(LABELS["barrier"], brr)

    american0 = AmericanOption(strike=6000.0, option_type=OptionType.PUT, maturity=1.0)
    american1 = AmericanOption(strike=6000.0, option_type=OptionType.PUT, maturity=1.0 - days / 365)
    ame = explain(ValuationSnapshot(american0, us0, e0, date=FRI, quantity=QTY["american"]),
                  ValuationSnapshot(american1, us1, e1, date=MON, quantity=QTY["american"]))
    show(LABELS["american"] + " (BS93 -> BAW)", ame)

    # A snowball's contract rolls schedule and maturity together: use the tracker, never assign maturity.
    tracker = AutocallableLifecycleTracker(product=snowball(), quantity=QTY["snowball"],
                                           start_date=pd.Timestamp(FRI))
    sb1 = tracker.product_for_pricing(pd.Timestamp(MON), e1)
    snb = explain(ValuationSnapshot(snowball(), pde, e0, date=FRI, quantity=QTY["snowball"]),
                  ValuationSnapshot(sb1, pde, e1, date=MON, quantity=QTY["snowball"]),
                  transition=None)
    show(LABELS["snowball"] + " (PDE fast)", snb)

    # The desk hedges the index exposure in the front futures. With q = r - b and no contract
    # basis of its own, the delta-one forward reproduces the discounted futures exactly:
    # F = S*exp((r - q)*T) = S*exp(b*T). Size the hedge on the book's delta at t0, in index
    # units; a desk would trade this as round lots of a 200-unit contract.
    d1 = DeltaOneEngine()
    book_delta = sum(taylor_delta(r) for r in (van, brr, ame, snb))
    hedge_units = -round(book_delta / math.exp((R0 - Q0) * FUT_T))
    fut0 = Futures(underlying="IDX", multiplier=1.0, maturity=FUT_T)
    fut1 = Futures(underlying="IDX", multiplier=1.0, maturity=FUT_T - days / 365)
    hed = explain(ValuationSnapshot(fut0, d1, e0, date=FRI, quantity=float(hedge_units)),
                  ValuationSnapshot(fut1, d1, e1, date=MON, quantity=float(hedge_units)))
    show(f"{LABELS['hedge']} x{hedge_units:,}", hed)

    # --- 2. A knock-out day: the event row ---------------------------------------------------
    ko_tracker = AutocallableLifecycleTracker(product=snowball(), quantity=QTY["snowball"],
                                              start_date=pd.Timestamp(START))
    t0, t1 = find_event_day(ko_tracker, LifecycleEventType.KNOCK_OUT,
                            spot_quiet=KO_QUIET, spot_trigger=KO_TRIGGER)
    # A 10% rally does not leave the rest of the market still: vol bleeds, rates firm.
    # The rally narrows the futures discount, so implied q falls with it.
    ke0 = env(KO_QUIET, 0.22, 0.0200, 0.0520, -0.0320, t0)
    ke1 = env(KO_TRIGGER, 0.19, 0.0215, 0.0455, -0.0240, t1)
    state_before = deepcopy(ko_tracker.lifecycle)
    product_t0 = ko_tracker.product_for_pricing(pd.Timestamp(t0), ke0)
    product_alive = ko_tracker.product_for_pricing(pd.Timestamp(t1), ke1)
    events = ko_tracker.observe(pd.Timestamp(t1), ko_tracker.product_for_lifecycle(), ke1, KO_TRIGGER)
    state_after = deepcopy(ko_tracker.lifecycle)
    transition = LifecycleTransition(product_alive_t1=product_alive, engine_alive_t1=pde,
                                     state_before=state_before, state_after=state_after, events=tuple(events))
    ko0 = ValuationSnapshot(product_t0, pde, ke0, date=t0, quantity=QTY["snowball"],
                            lifecycle_state=state_before, valuation_point=state_before.valuation_point)
    ko1 = ValuationSnapshot(ko_tracker.product_for_pricing(pd.Timestamp(t1), ke1), pde, ke1, date=t1,
                            quantity=QTY["snowball"], lifecycle_state=state_after,
                            valuation_point=state_after.valuation_point)
    ko = explain(ko0, ko1, transition=transition)
    show(f"snowball KO day {t1.date()}", ko)

    # --- 3. The book -------------------------------------------------------------------------
    book0 = BookSnapshot(date=FRI, environments={"IDX": e0}, positions={
        "vanilla": PositionSnapshot("vanilla", "IDX",
                                    ValuationSnapshot(vanilla0, bs, e0, date=FRI, quantity=QTY["vanilla"])),
        "barrier": PositionSnapshot("barrier", "IDX",
                                    ValuationSnapshot(barrier0, bar, e0, date=FRI, quantity=QTY["barrier"])),
        "american": PositionSnapshot("american", "IDX",
                                     ValuationSnapshot(american0, us0, e0, date=FRI, quantity=QTY["american"])),
        "snowball": PositionSnapshot("snowball", "IDX",
                                     ValuationSnapshot(snowball(), pde, e0, date=FRI, quantity=QTY["snowball"])),
        "hedge": PositionSnapshot("hedge", "IDX",
                                  ValuationSnapshot(fut0, d1, e0, date=FRI, quantity=float(hedge_units))),
    })
    book1 = BookSnapshot(date=MON, environments={"IDX": e1}, positions={
        "vanilla": PositionSnapshot("vanilla", "IDX",
                                    ValuationSnapshot(vanilla1, bs, e1, date=MON, quantity=QTY["vanilla"])),
        "barrier": PositionSnapshot("barrier", "IDX",
                                    ValuationSnapshot(barrier1, bar, e1, date=MON, quantity=QTY["barrier"])),
        "american": PositionSnapshot("american", "IDX",
                                     ValuationSnapshot(american1, us1, e1, date=MON, quantity=QTY["american"])),
        "snowball": PositionSnapshot("snowball", "IDX",
                                     ValuationSnapshot(sb1, pde, e1, date=MON, quantity=QTY["snowball"])),
        "hedge": PositionSnapshot("hedge", "IDX",
                                  ValuationSnapshot(fut1, d1, e1, date=MON, quantity=float(hedge_units))),
    })
    port = explain_portfolio(book0, book1)
    print(f"\n=== portfolio: total {port.total_pnl:+,.2f}")
    print(port.to_frame().query("position_id == 'portfolio' and underlying == '*'")[COLS]
          .to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\nreconciliation:", port.metadata["reconciliation"])

    cases = Cases(vanilla=van, barrier=brr, snowball=snb, american=ame, hedge=hed,
                  hedge_units=hedge_units, ko=ko, ko_dates=(t0, t1), portfolio=port)

    # --- 4. Waterfall order, for the lecture page only (six cheap analytical repricings) -----
    if with_orders:
        spot_first = (Factor.SPOT, Factor.VOL, Factor.TIME, Factor.RATE,
                      Factor.DIVIDEND, Factor.BASIS, Factor.MODEL)
        recipes = {
            "time first (default)": PnLExplainConfig(methods=(ExplainMethod.WATERFALL,),
                                                     waterfall_order=MARKET_FACTORS),
            "spot first": PnLExplainConfig(methods=(ExplainMethod.WATERFALL,),
                                           waterfall_order=spot_first),
            "Shapley": PnLExplainConfig(methods=(ExplainMethod.WATERFALL,),
                                        waterfall_order=MARKET_FACTORS, interaction="shapley"),
        }
        pairs = {
            LABELS["barrier"]: (ValuationSnapshot(barrier0, bar, e0, date=FRI, quantity=QTY["barrier"]),
                                ValuationSnapshot(barrier1, bar, e1, date=MON, quantity=QTY["barrier"])),
            LABELS["american"]: (ValuationSnapshot(american0, us0, e0, date=FRI, quantity=QTY["american"]),
                                 ValuationSnapshot(american1, us1, e1, date=MON, quantity=QTY["american"])),
        }
        cases.orders = {
            name: {recipe: explain(s0, s1, config=cfg) for recipe, cfg in recipes.items()}
            for name, (s0, s1) in pairs.items()
        }
    return cases


# =============================================================================
# The visual layer: waterfall bridges as hand-written SVG
# =============================================================================

INK, MUTED, LINE = "#17211f", "#5d6964", "#cfd8d1"
UP, DOWN, ANCHOR, RESIDUAL = "#006d77", "#b23a48", "#3f4f4a", "#9b6a08"
_SEQ = itertools.count()


@dataclass(frozen=True)
class BridgeBar:
    """One bar of a waterfall bridge chart.

    ``start`` and ``end`` are running position values in money, not increments.
    An anchor bar (the opening or closing value) runs from 0 to that value; a
    step bar runs from the running total before the step to the running total
    after it, so consecutive steps chain end-to-start.
    """

    label: str
    start: float
    end: float
    kind: str = "step"          # "anchor" | "step" | "residual"
    note: str = ""

    @property
    def amount(self) -> float:
        return self.end - self.start


def bridge_bars(result: PnLExplainResult, method: ExplainMethod) -> Tuple[BridgeBar, ...]:
    """Project one explain result onto the bars of a waterfall bridge chart.

    In drawing order: an opening anchor at ``result.pv_t0.total``, one step per
    attribution row of ``method``, and a closing anchor at ``result.pv_t1.total``.
    Because a method reconciles exactly, the last step lands on the closing
    anchor; ``validated_bars`` checks that, so a wrong projection fails loudly
    instead of drawing a plausible lie.

    ``rows_for`` already folds in the ``shared`` rows -- the lifecycle event and
    the transaction cost -- that belong to both methods.

    Three judgement calls, and what this projection does with them:

    1. Zero rows are kept. A flat vol day still produces a ``vol`` row worth
       0.0000, and drawing it says the model looked at vol and found it still.
       It also keeps the factor axis identical across charts, which is what
       makes the three panels of section 3 comparable at a glance.
    2. Theta sub-rows are not bars. ``RowKind.INFORMATIONAL`` rows decompose
       theta into carry, r, q and convexity; they do not sum into the total, so
       drawing them would break the bridge. They ride along as a tooltip on the
       time bar of the method that produced them.
    3. Taylor's residual is set apart. ``unexplained`` is an ordinary component
       row and could be drawn as one, but tagging it ``kind="residual"`` renders
       it in hatched amber -- and its growth from vanilla to knock-out is the
       whole argument of section 3.
    """
    subrows = [r for r in result.rows
               if r.kind is RowKind.INFORMATIONAL and r.factor is Factor.TIME
               and r.method in (method, ExplainMethod.SHARED)]
    note = ", ".join(f"{r.term} {r.pnl:+.4f}" for r in subrows)
    bars = [BridgeBar("V(t0)", 0.0, result.pv_t0.total, kind="anchor")]
    running = result.pv_t0.total
    for row in result.rows_for(method, kind=RowKind.COMPONENT):
        bars.append(BridgeBar(
            row.term, running, running + row.pnl,
            kind="residual" if row.factor is Factor.UNEXPLAINED else "step",
            note=note if row.factor is Factor.TIME else ""))
        running += row.pnl
    bars.append(BridgeBar("V(t1)", 0.0, result.pv_t1.total, kind="anchor"))
    return tuple(bars)


def position_bridge_bars(portfolio: PortfolioExplainResult) -> Tuple[BridgeBar, ...]:
    """The book-level bridge: opening book value, one step per position, closing value.

    This is the other projection of the same data -- by position rather than by
    factor -- and it is what a desk head reads first.
    """
    parts = [(pid, pr.instrument) for pid, pr in portfolio.positions.items() if pr.instrument is not None]
    opening = sum(inst.pv_t0.total for _, inst in parts)
    closing = sum(inst.pv_t1.total for _, inst in parts)
    bars = [BridgeBar("book t0", 0.0, opening, kind="anchor")]
    running = opening
    for pid, _ in parts:
        step = portfolio.positions[pid].total_pnl
        bars.append(BridgeBar(pid, running, running + step))
        running += step
    bars.append(BridgeBar("book t1", 0.0, closing, kind="anchor"))
    return tuple(bars)


def validated_bars(result: PnLExplainResult, method: ExplainMethod) -> Tuple[BridgeBar, ...]:
    """Call bridge_bars and refuse a projection that does not close the bridge."""
    bars = tuple(bridge_bars(result, method))
    steps = [b for b in bars if b.kind != "anchor"]
    if len(bars) < 3 or bars[0].kind != "anchor" or bars[-1].kind != "anchor" or not steps:
        raise ValueError("bridge_bars must return an opening anchor, at least one step, a closing anchor")
    tol = 1e-6 * max(1.0, abs(result.pv_t0.total), abs(result.pv_t1.total))
    checks = [
        ("opening anchor", bars[0].end, result.pv_t0.total),
        ("closing anchor", bars[-1].end, result.pv_t1.total),
        ("first step start", steps[0].start, result.pv_t0.total),
        ("last step end", steps[-1].end, result.pv_t1.total),
    ]
    for a, b in zip(steps, steps[1:]):
        checks.append((f"step chain {a.label} -> {b.label}", b.start, a.end))
    for what, got, want in checks:
        if abs(got - want) > tol:
            raise ValueError(f"bridge does not close: {what} is {got:.6f}, expected {want:.6f}")
    return bars


def bridge_domain(*groups: Sequence[BridgeBar]) -> Tuple[float, float]:
    """One value range shared by several charts, so their bars are comparable by eye."""
    values = [0.0]
    for bars in groups:
        for b in bars:
            values += [b.start, b.end]
    return min(values), max(values)


def as_pnl_bars(bars: Sequence[BridgeBar]) -> Tuple[BridgeBar, ...]:
    """The same steps, rebased to start from zero: PnL rather than value.

    A position worth -296 that moves -11 is unreadable as a value bridge -- every
    step is a hairline beside the anchors, which is exactly where the interesting
    rows live. Rebasing keeps every step at full height and drops the level, which
    the tables next to each chart carry instead.
    """
    base = bars[0].end
    steps = [BridgeBar(b.label, b.start - base, b.end - base, kind=b.kind, note=b.note)
             for b in bars if b.kind != "anchor"]
    closing = steps[-1].end if steps else 0.0
    return tuple(steps) + (BridgeBar("total PnL", 0.0, closing, kind="anchor"),)


def _bar_fill(bar: BridgeBar, uid: str) -> str:
    if bar.kind == "anchor":
        return ANCHOR
    if bar.kind == "residual":
        return f"url(#{uid}-hatch)"
    return UP if bar.amount >= 0 else DOWN


def money_label(v: float, span: float) -> str:
    """A chart label in the unit a desk reads: 312k, not 312,431.87.

    The unit follows the chart's own range, and a row that is small against that
    range keeps a decimal -- otherwise every minor row on a million-yuan chart
    reads "-0k" and looks like a bug rather than a small number.
    """
    if v == 0.0:
        return "0"
    for unit, size in (("m", 1e6), ("k", 1e3)):
        if span >= 20 * size:
            scaled = v / size
            return f"{scaled:,.1f}{unit}" if abs(scaled) < 10 else f"{scaled:,.0f}{unit}"
    return f"{v:,.0f}"


def svg_bridge(bars: Sequence[BridgeBar], *, width: int = 560, height: int = 300,
               domain: Optional[Tuple[float, float]] = None, caption: str = "") -> str:
    """A waterfall bridge: anchors from the baseline, signed steps chained between them."""
    uid = f"bg{next(_SEQ)}"
    left, right, top, bottom = 62, 14, 26, 74
    plot_w, plot_h = width - left - right, height - top - bottom
    lo, hi = domain if domain is not None else bridge_domain(bars)
    span = hi - lo
    if span <= 0:
        lo, hi, span = lo - 1.0, hi + 1.0, 2.0
    lo, hi = lo - 0.14 * span, hi + 0.14 * span
    span = hi - lo

    def y(v: float) -> float:
        return top + plot_h * (hi - v) / span

    band = plot_w / len(bars)
    bw = min(band * 0.60, 44.0)
    out = [f'<svg class="bridge" viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="{escape(caption or "waterfall bridge")}">',
           f'<defs><pattern id="{uid}-hatch" width="6" height="6" patternUnits="userSpaceOnUse" '
           f'patternTransform="rotate(45)"><rect width="6" height="6" fill="#f6ecd6"/>'
           f'<line x1="0" y1="0" x2="0" y2="6" stroke="{RESIDUAL}" stroke-width="2.4"/></pattern></defs>']

    for i in range(5):
        v = lo + span * i / 4
        gy = y(v)
        out.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{width - right}" y2="{gy:.1f}" '
                   f'stroke="{LINE}" stroke-width="1" stroke-dasharray="2 4"/>')
        out.append(f'<text x="{left - 8}" y="{gy + 3.5:.1f}" text-anchor="end" '
                   f'class="tick">{money_label(v, span)}</text>')
    if lo < 0 < hi:
        out.append(f'<line x1="{left}" y1="{y(0):.1f}" x2="{width - right}" y2="{y(0):.1f}" '
                   f'stroke="{MUTED}" stroke-width="1.2"/>')

    for i, bar in enumerate(bars):
        cx = left + band * (i + 0.5)
        x = cx - bw / 2
        y_hi, y_lo = y(max(bar.start, bar.end)), y(min(bar.start, bar.end))
        h = max(y_lo - y_hi, 1.6)
        rect = (f'<rect x="{x:.1f}" y="{y_hi:.1f}" width="{bw:.1f}" height="{h:.1f}" '
                f'fill="{_bar_fill(bar, uid)}" rx="1.5"/>')
        out.append(f'<g><title>{escape(bar.note)}</title>{rect}</g>' if bar.note else rect)
        if i and bars[i - 1].kind != "anchor" and bar.kind != "anchor":
            ly = y(bar.start)
            out.append(f'<line x1="{left + band * (i - 0.5) + bw / 2:.1f}" y1="{ly:.1f}" x2="{x:.1f}" '
                       f'y2="{ly:.1f}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="3 3"/>')
        value = bar.end if bar.kind == "anchor" else bar.amount
        label_y = y_hi - 7 if value >= 0 or bar.kind == "anchor" else y_lo + 15
        sign = "+" if bar.kind != "anchor" and value > 0 else ""
        out.append(f'<text x="{cx:.1f}" y="{label_y:.1f}" text-anchor="middle" class="val">'
                   f'{sign}{money_label(value, span)}</text>')
        ty = height - bottom + 16
        out.append(f'<text x="{cx:.1f}" y="{ty:.1f}" text-anchor="end" class="cat" '
                   f'transform="rotate(-32 {cx:.1f} {ty:.1f})">{escape(bar.label)}</text>')
    out.append("</svg>")
    if caption:
        out.append(f'<p class="chart-caption">{escape(caption)}</p>')
    return "\n".join(out)


def paired_bridges(result: PnLExplainResult, *, width: int = 560, height: int = 300) -> str:
    """The same day decomposed both ways, on one shared value axis."""
    wf = validated_bars(result, ExplainMethod.WATERFALL)
    ty = validated_bars(result, ExplainMethod.TAYLOR)
    dom = bridge_domain(wf, ty)
    return ('<div class="pair">'
            f'<figure>{svg_bridge(wf, width=width, height=height, domain=dom)}'
            '<figcaption>waterfall &mdash; full revaluation, one factor at a time</figcaption></figure>'
            f'<figure>{svg_bridge(ty, width=width, height=height, domain=dom)}'
            '<figcaption>Taylor &mdash; greeks times moves, plus what is left over</figcaption></figure>'
            '</div>')


def portfolio_factor_bars(portfolio: PortfolioExplainResult,
                          method: ExplainMethod = ExplainMethod.WATERFALL) -> Tuple[BridgeBar, ...]:
    """The book-level bridge by factor, from the aggregated portfolio rows."""
    insts = [pr.instrument for pr in portfolio.positions.values() if pr.instrument is not None]
    opening = sum(i.pv_t0.total for i in insts)
    closing = sum(i.pv_t1.total for i in insts)
    bars = [BridgeBar("book t0", 0.0, opening, kind="anchor")]
    running = opening
    for row in portfolio.rows:
        if row.kind is not RowKind.COMPONENT or row.method not in (method, ExplainMethod.SHARED):
            continue
        bars.append(BridgeBar(row.term, running, running + row.pnl))
        running += row.pnl
    bars.append(BridgeBar("book t1", 0.0, closing, kind="anchor"))
    return tuple(bars)


# =============================================================================
# The lecture page
# =============================================================================

CSS = """
:root {
  color-scheme: light;
  --paper: #f5f7f4; --ink: #17211f; --muted: #5d6964; --line: #cfd8d1;
  --panel: #ffffff; --panel2: #eef4f0; --teal: #006d77; --red: #b23a48;
  --amber: #9b6a08; --blue: #3559a6; --shadow: 0 18px 50px rgba(23, 33, 31, .08);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background:
    linear-gradient(90deg, rgba(0,109,119,.07) 0 1px, transparent 1px 100%),
    linear-gradient(180deg, rgba(0,109,119,.06) 0 1px, transparent 1px 100%),
    var(--paper);
  background-size: 44px 44px;
  color: var(--ink);
  font-family: "Avenir Next", "Gill Sans", "Segoe UI", sans-serif;
  line-height: 1.55;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 32px 28px 70px; }
h1, h2, h3 { font-family: Georgia, "Times New Roman", serif; line-height: 1.1; }
h1 { margin: 0; font-size: 3.8rem; font-weight: 700; max-width: 900px; }
h2 { margin: 0 0 14px; font-size: 2.05rem; }
h3 { margin: 22px 0 8px; font-size: 1.2rem; }
p { max-width: 860px; margin: 0 0 14px; }
code { background: var(--panel2); padding: 1px 5px; border-radius: 3px; font-size: .88em; }
.eyebrow {
  margin: 0 0 10px; color: var(--teal); font-size: .76rem; font-weight: 800;
  letter-spacing: .13em; text-transform: uppercase;
}
header.hero {
  display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(320px, .9fr);
  gap: 34px; align-items: center; border-bottom: 1px solid var(--line); padding: 26px 0 38px;
}
.lede { margin-top: 20px; color: var(--muted); font-size: 1.12rem; max-width: 740px; }
.chips { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 22px; }
.chip {
  border: 1px solid var(--line); background: rgba(255,255,255,.7);
  padding: 7px 11px; border-radius: 999px; font-size: .86rem;
}
.hero-panel {
  background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow);
  border-radius: 8px; padding: 22px;
}
.hero-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.metric {
  border: 1px solid var(--line); background: var(--panel2);
  border-radius: 7px; padding: 12px; min-height: 78px;
}
.metric small {
  display: block; color: var(--muted); text-transform: uppercase;
  letter-spacing: .08em; font-weight: 700; font-size: .66rem;
}
.metric b { display: block; margin-top: 5px; font-size: 1.2rem; font-variant-numeric: tabular-nums; }
.metric.good b { color: var(--teal); }
.metric.warn b { color: var(--amber); }
section { padding: 42px 0; border-bottom: 1px solid var(--line); }
.section-grid {
  display: grid; grid-template-columns: minmax(240px, .68fr) minmax(0, 1.32fr);
  gap: 30px; align-items: start;
}
.rail { position: sticky; top: 18px; }
.rail p { color: var(--muted); }
.panel {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 8px; padding: 20px; box-shadow: var(--shadow); overflow-x: auto;
}
.panel + .panel { margin-top: 18px; }
.pair { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
.pair figure, figure { margin: 0; }
figcaption, .chart-caption {
  margin: 6px 0 0; color: var(--muted); font-size: .82rem; text-align: center;
}
svg.bridge { width: 100%; height: auto; display: block; }
svg.bridge .tick { fill: #5d6964; font-size: 10px; font-family: inherit; }
svg.bridge .val { fill: #17211f; font-size: 10.5px; font-weight: 700; font-family: inherit;
                  font-variant-numeric: tabular-nums; }
svg.bridge .cat { fill: #5d6964; font-size: 10.5px; font-family: inherit; }
table {
  width: 100%; border-collapse: collapse; margin: 12px 0 4px;
  font-size: .88rem; font-variant-numeric: tabular-nums;
}
th, td { border-bottom: 1px solid var(--line); padding: 7px 9px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
thead th {
  border-bottom: 2px solid var(--line); color: var(--muted);
  text-transform: uppercase; letter-spacing: .06em; font-size: .68rem;
}
tbody tr:hover { background: var(--panel2); }
td.pos { color: var(--teal); }
td.neg { color: var(--red); }
td.flag { color: var(--amber); font-weight: 700; }
.identity { display: flex; flex-wrap: wrap; align-items: stretch; gap: 10px; margin: 14px 0 4px; }
.identity .term {
  flex: 1 1 150px; border: 1px solid var(--line); border-left: 4px solid var(--teal);
  background: #f8fbf9; padding: 10px 12px;
}
.identity .term.cash { border-left-color: var(--amber); }
.identity .term.total { border-left-color: var(--ink); background: var(--panel2); }
.identity .term small { display: block; color: var(--muted); font-size: .68rem;
  text-transform: uppercase; letter-spacing: .07em; font-weight: 700; }
.identity .term b { display: block; margin-top: 4px; font-size: 1.14rem;
  font-variant-numeric: tabular-nums; }
.identity .op { align-self: center; color: var(--muted); font-size: 1.3rem; font-weight: 700; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; color: var(--muted); font-size: .8rem;
  margin-top: 10px; }
.legend span { display: flex; align-items: center; gap: 6px; }
.swatch { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
.callout {
  border-left: 4px solid var(--amber); background: #fdf8ee; padding: 12px 14px;
  margin: 16px 0 0; color: var(--ink);
}
.callout p:last-child { margin-bottom: 0; }
footer { padding: 26px 0 0; color: var(--muted); font-size: .84rem; }
@media (max-width: 900px) {
  header.hero, .section-grid, .pair { grid-template-columns: 1fr; }
  .rail { position: static; }
  h1 { font-size: 2.6rem; }
}
"""

def legend(*, residual: bool = True) -> str:
    """Only name the residual swatch on a chart that can actually show one."""
    parts = [f'<span><i class="swatch" style="background:{ANCHOR}"></i>value anchor</span>',
             f'<span><i class="swatch" style="background:{UP}"></i>gain</span>',
             f'<span><i class="swatch" style="background:{DOWN}"></i>loss</span>']
    if residual:
        parts.append(f'<span><i class="swatch" style="background:{RESIDUAL}"></i>Taylor residual</span>')
    return '<div class="legend">' + "".join(parts) + "</div>"


def money(x: float, digits: int = 0) -> str:
    return f"{x:,.{digits}f}"


def cell(x: float, digits: int = 0, flag: bool = False) -> str:
    cls = "flag" if flag else "pos" if x > 0 else "neg" if x < 0 else ""
    return f'<td class="{cls}">{money(x, digits)}</td>' if cls else f"<td>{money(x, digits)}</td>"


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def section(anchor: str, eyebrow: str, title: str, rail: str, body: str) -> str:
    return (f'<section id="{anchor}"><div class="section-grid">'
            f'<div class="rail"><p class="eyebrow">{escape(eyebrow)}</p><h2>{escape(title)}</h2>{rail}</div>'
            f'<div>{body}</div></div></section>')


def identity_block(label: str, vb) -> str:
    return (f'<h3>{escape(label)}</h3><div class="identity">'
            f'<div class="term"><small>contingent MTM</small><b>{money(vb.contingent_mtm)}</b></div>'
            '<div class="op">+</div>'
            f'<div class="term"><small>pending receivable PV</small><b>{money(vb.pending_receivable_pv)}</b></div>'
            '<div class="op">+</div>'
            f'<div class="term cash"><small>paid cash</small><b>{money(vb.paid_cash)}</b></div>'
            '<div class="op">=</div>'
            f'<div class="term total"><small>position value</small><b>{money(vb.total)}</b></div>'
            '</div>')


def _event_row(result: PnLExplainResult):
    rows = [r for r in result.rows if r.factor is Factor.LIFECYCLE_EVENT and r.kind is RowKind.COMPONENT]
    return rows[0] if rows else None


def render_lecture(cases: Cases) -> str:
    ko_t0, ko_t1 = cases.ko_dates
    port = cases.portfolio
    recon = port.metadata["reconciliation"]
    # Ordered by how smooth the payoff is, which is the order the residual grows in.
    named = ((LABELS["vanilla"], cases.vanilla), (LABELS["american"], cases.american),
             (LABELS["barrier"], cases.barrier),
             (f"{LABELS['snowball']}, KO day {ko_t1.date()}", cases.ko))

    def gross(res: PnLExplainResult) -> float:
        """Total absolute attribution, residual excluded: a denominator that does not vanish."""
        return sum(abs(r.pnl) for r in res.rows_for(ExplainMethod.TAYLOR, kind=RowKind.COMPONENT)
                   if r.factor is not Factor.UNEXPLAINED)

    # --- hero -------------------------------------------------------------------------------
    hero = f"""
<header class="hero">
  <div>
    <p class="eyebrow">quantark.pnlexplain</p>
    <h1>Where did the money go?</h1>
    <p class="lede">One risk day on an index structured-products desk, explained two ways.
      Both must add up to the yuan; only one of them survives a barrier.</p>
    <div class="chips">
      <span class="chip">{FRI:%a %d %b %Y} &rarr; {MON:%a %d %b %Y}</span>
      <span class="chip">index {S0:,.0f} &rarr; {S1:,.0f} ({S1 / S0 - 1:+.1%})</span>
      <span class="chip">vol {V0:.0%} &rarr; {V1:.0%} ({(V1 - V0) * 100:+.0f} pts)</span>
      <span class="chip">rate {R0:.2%} &rarr; {R1:.2%}</span>
      <span class="chip">futures basis {B0:.2%} &rarr; {B1:.2%}</span>
      <span class="chip">implied q = r &minus; b: {Q0:.2%} &rarr; {Q1:.2%}</span>
      <span class="chip">five positions, delta-hedged in the front future</span>
    </div>
  </div>
  <div class="hero-panel">
    <div class="hero-metrics">
      <div class="metric"><small>book PnL on the day</small><b>{port.total_pnl / 1e6:+,.2f}m</b></div>
      <div class="metric good"><small>waterfall reconciles</small>
        <b>{"yes" if recon['waterfall']['ok'] else "NO"}</b></div>
      <div class="metric good"><small>Taylor reconciles</small>
        <b>{"yes" if recon['taylor']['ok'] else "NO"}</b></div>
      <div class="metric warn"><small>Taylor unexplained, KO day</small><b>{cases.ko.unexplained / 1e3:+,.0f}k</b></div>
    </div>
  </div>
</header>
"""

    # --- 1. the value identity --------------------------------------------------------------
    s1 = section(
        "identity", "step one", "The identity everything must satisfy",
        "<p>A position is worth three things at once: what the live contract is worth, the present "
        "value of cash already earned but not yet paid, and cash already banked. PnL explain "
        "attributes the change in that sum, never in the contract price alone.</p>"
        "<p>The vanilla never leaves the first box. The snowball, once it knocks out, empties the "
        "first box into the third. Attribution that watched only the contract price would report a "
        "twenty-million-yuan catastrophe on knock-out day.</p>",
        '<div class="panel">'
        + identity_block(f"{LABELS['vanilla']}, {MON:%d %b}: all contingent", cases.vanilla.pv_t1)
        + identity_block(f"{LABELS['snowball']}, {ko_t1:%d %b %Y}, after knock-out: the cash has moved",
                         cases.ko.pv_t1)
        + '</div>')

    # --- 2. two methods, one day ------------------------------------------------------------
    wf_v = cases.american.by_factor(ExplainMethod.WATERFALL)
    ty_v = cases.american.by_factor(ExplainMethod.TAYLOR)
    factors = ["time", "spot", "vol", "rate", "dividend", "model", "basis", "unexplained"]
    rows = [[f"<td>{escape(f)}</td>", cell(wf_v.get(f, 0.0)), cell(ty_v.get(f, 0.0)),
             cell(ty_v.get(f, 0.0) - wf_v.get(f, 0.0))] for f in factors]
    rows.append(["<td><b>total</b></td>", cell(cases.american.total_pnl),
                 cell(cases.american.total_pnl), "<td>0</td>"])
    s2 = section(
        "methods", "step two", "Two methods, one day",
        "<p>The waterfall reprices the position once per factor, moving one thing at a time from "
        "the old market to the new. Every step is a real price, so the steps add up by "
        "construction.</p>"
        "<p>Taylor multiplies today's greeks by today's moves. It is cheaper, it names the risk in "
        "the language a desk hedges in, and it is an approximation, so it needs a residual row to "
        "close.</p>"
        f"<p>Shown on the American put, because it is the only position that moves the model row: "
        f"the desk repriced its American book from the BS93 approximation to BAW over the weekend. "
        f"That row is worth "
        f"{cases.american.by_factor(ExplainMethod.WATERFALL).get('model', 0.0):,.0f}, which is "
        f"small, and that is the point. Without it the repricing would have leaked into the market "
        f"rows, and nobody could have told a market move from a change of method.</p>"
        "<p>Both charts are value bridges: they start at what the position was worth and end at "
        "what it is worth now.</p>",
        '<div class="panel">' + paired_bridges(cases.american) + legend()
        + table(["factor", "waterfall", "Taylor", "Taylor − waterfall"], rows)
        + f'<div class="callout"><p><b>The dividend row is the carry row.</b> An index desk does '
          f'not quote a dividend yield. It observes the futures basis, and the yield that prices '
          f'the product is implied by cost of carry: <code>q = r &minus; b</code>. This day widens '
          f'the futures discount from {B0:.2%} to {B1:.2%} while the rate firms '
          f'{(R1 - R0) * 1e4:.0f}bp, so implied q rises from {Q0:.2%} to {Q1:.2%}. That is why q is '
          f'never moved on its own in this demo: move the rate and the basis, and q follows.</p>'
          f'<p>The basis row itself stays zero, and structurally so. With q implied that way the '
          f'delta-one forward already reproduces the discounted future, '
          f'<code>F = S&middot;e<sup>(r&minus;q)T</sup> = S&middot;e<sup>bT</sup></code>, so no '
          f'equity engine needs to read the environment&rsquo;s basis yield and none does. The '
          f'factor is still declared and checked on every run; only the stress-test and '
          f'dynamic-scenario engines read that field today.</p></div></div>')

    # --- 3. where Taylor breaks -------------------------------------------------------------
    breaks = "".join(
        '<figure>'
        + svg_bridge(as_pnl_bars(validated_bars(res, ExplainMethod.TAYLOR)), width=760, height=290)
        + f"<figcaption>{escape(name)}</figcaption></figure>" for name, res in named)
    share_rows = []
    for name, res in named:
        g = gross(res)
        share = abs(res.unexplained) / g if g else float("nan")
        share_rows.append([f"<td>{escape(name)}</td>", cell(res.total_pnl), cell(g),
                           cell(res.unexplained),
                           f'<td class="{"flag" if share > 0.05 else ""}">{share:.1%}</td>'])
    s3 = section(
        "breaks", "step three", "Where Taylor breaks",
        "<p>A Taylor expansion assumes the payoff is smooth over the move. A barrier is a "
        "discontinuity, and a knock-out is the discontinuity actually firing.</p>"
        f"<p>Read the amber bar down these four charts. The vanilla and the American are smooth "
        f"payoffs and hold the residual at {abs(cases.vanilla.unexplained) / gross(cases.vanilla):.1%} "
        f"and {abs(cases.american.unexplained) / gross(cases.american):.1%} of the risk they carry. "
        f"The barrier put, which the day pushed to within "
        f"{(S1 / 5400.0 - 1):.1%} of its barrier, reaches "
        f"{abs(cases.barrier.unexplained) / gross(cases.barrier):.1%}, and the snowball on the day "
        f"it terminated reaches {abs(cases.ko.unexplained) / gross(cases.ko):.1%}.</p>"
        f"<p>Measure the residual against the gross attribution, not against the total. The barrier "
        f"put made almost nothing on the day because its spot gain and its vol loss nearly cancelled, "
        f"and dividing by that near-zero total would have called a "
        f"{abs(cases.barrier.unexplained) / gross(cases.barrier):.0%} miss "
        f"{abs(cases.barrier.unexplained) / abs(cases.barrier.total_pnl):.0%}.</p>"
        "<p>The waterfall has no residual on any of the four, because it never approximates. These "
        "charts plot PnL from zero rather than value from the book level, so the small rows stay "
        "legible next to a twenty-million-yuan snowball.</p>",
        f'<div class="panel">{breaks}'
        + legend()
        + table(["case", "total PnL", "gross attribution", "Taylor unexplained", "share of gross"],
                share_rows)
        + '<div class="callout"><p>This is the argument for keeping a full-revaluation method on the '
          'desk even when greeks are already computed. On the day it matters most, the cheap method '
          'is least trustworthy.</p></div></div>')

    # --- 4. order is a choice ---------------------------------------------------------------
    order_panels = []
    for name, by_recipe in cases.orders.items():
        charts = "".join(
            '<figure>'
            + svg_bridge(as_pnl_bars(validated_bars(res, ExplainMethod.WATERFALL)), width=430, height=270)
            + f"<figcaption>{escape(recipe)}</figcaption></figure>" for recipe, res in by_recipe.items())
        rws = []
        for recipe, res in by_recipe.items():
            bf = res.by_factor(ExplainMethod.WATERFALL)
            rws.append([f"<td>{escape(recipe)}</td>", cell(bf.get("time", 0.0)), cell(bf.get("spot", 0.0)),
                        cell(bf.get("vol", 0.0)), cell(res.total_pnl),
                        f"<td>{res.reconcile(ExplainMethod.WATERFALL):+.1e}</td>"])
        order_panels.append(
            f'<div class="panel"><h3>{escape(name)}</h3>'
            f'<div class="pair" style="grid-template-columns:repeat(3,1fr)">{charts}</div>'
            + legend(residual=False)
            + table(["order", "time", "spot", "vol", "total", "gap"], rws) + "</div>")
    s4 = section(
        "order", "step four", "The order is a choice, not a fact",
        "<p>Move time first and the time step is pure theta; the spot step then inherits every "
        "interaction between time and spot. Move spot first and the same interaction lands on "
        "time instead.</p>"
        "<p>No order is more correct. Each is a complete, exactly-reconciling story about a move "
        "that genuinely happened all at once. Shapley averages over every order, which spreads "
        "interaction evenly and costs a factorial.</p>"
        "<p>Notice what does not move: the total. The gap stays zero for the sequential orders and "
        "lands on float noise for Shapley, which averages over many orderings before it closes.</p>",
        "".join(order_panels))

    # --- 5. the lifecycle event row ---------------------------------------------------------
    ev = _event_row(cases.ko)
    ko_bars = validated_bars(cases.ko, ExplainMethod.WATERFALL)
    s5 = section(
        "lifecycle", "step five", "The day the contract ended",
        "<p>Knock-out day is two things happening together: the market moved, and the contract "
        "terminated. The waterfall separates them by construction.</p>"
        "<p>The market rows carry the value from where it opened to what it would have been worth "
        "had nothing fired. That is the alive value. The single lifecycle row carries it the rest "
        "of the way to what the position is actually booked at.</p>"
        "<p>That is the definition of the event row, and the chart is the definition drawn.</p>",
        '<div class="panel">'
        + svg_bridge(as_pnl_bars(ko_bars), width=760, height=330,
                     caption=f"{LABELS['snowball']}, {ko_t0.date()} to {ko_t1.date()}, "
                             f"index {KO_QUIET:,.0f} to {KO_TRIGGER:,.0f} "
                             f"(PnL from zero; the levels are in the table)")
        + legend(residual=False)
        + table(["quantity", "value"], [
            ["<td>value at t0</td>", cell(cases.ko.pv_t0.total)],
            ["<td>alive value at t1 (had nothing fired)</td>", cell(cases.ko.pv_alive_t1.total)],
            ["<td>market rows</td>", cell(cases.ko.pv_alive_t1.total - cases.ko.pv_t0.total)],
            ["<td>lifecycle event row</td>", cell(ev.pnl if ev else 0.0)],
            ["<td>booked value at t1</td>", cell(cases.ko.pv_t1.total)],
            ["<td><b>total PnL</b></td>", cell(cases.ko.total_pnl)],
        ])
        + '<div class="callout"><p>A receivable being paid is not an event. Cash that was already '
          'owed simply moves from the pending box to the paid box inside the time step, which is '
          'why the identity in step one has three terms and not two.</p></div></div>')

    # --- 6. the book ------------------------------------------------------------------------
    pos_bars = position_bridge_bars(port)
    recon_rows = [[f"<td>{escape(method)}</td>", cell(v["expected"]), cell(v["explained"]),
                   f"<td>{v['gap']:+.2e}</td>",
                   f'<td class="{"pos" if v["ok"] else "flag"}">{"ok" if v["ok"] else "FAILED"}</td>']
                  for method, v in recon.items()]
    names = dict(LABELS, hedge=f"{LABELS['hedge']} x{cases.hedge_units:,}")
    pos_rows = [[f"<td>{escape(names.get(pid, pid))}</td>",
                 cell(pr.instrument.pv_t0.total if pr.instrument else 0.0),
                 cell(pr.instrument.pv_t1.total if pr.instrument else 0.0), cell(pr.total_pnl)]
                for pid, pr in port.positions.items()]
    pos_rows.append(["<td><b>book</b></td>",
                     cell(sum(p.instrument.pv_t0.total for p in port.positions.values() if p.instrument)),
                     cell(sum(p.instrument.pv_t1.total for p in port.positions.values() if p.instrument)),
                     cell(port.total_pnl)])
    book_rows = {(r.method, r.term): r.pnl for r in port.rows if r.kind is RowKind.COMPONENT}
    book_delta = book_rows.get((ExplainMethod.TAYLOR, "delta"), 0.0)
    book_spot = book_rows.get((ExplainMethod.WATERFALL, "spot"), 0.0)
    s6 = section(
        "book", "step six", "The book, and the gate",
        "<p>Positions aggregate two ways. By position, which is what a desk head reads. By factor, "
        "which is what a risk manager reads. They are projections of one set of rows and they agree "
        "to the yuan.</p>"
        f"<p>The book is hedged with {cases.hedge_units:,} index units of the front future, sized on "
        "its delta at the open. That hedge is why the book PnL is smaller than the snowball's alone, "
        "and it is where the two methods stop agreeing about what spot did.</p>"
        "<p>The reconciliation dictionary is the gate. It compares what the value identity says the "
        "book made against what the attribution rows explain, per method. The gap is float noise on "
        "a two-million-yuan move, and it is checked on every run rather than assumed.</p>",
        '<div class="panel">'
        + svg_bridge(as_pnl_bars(pos_bars), width=760, height=310, caption="by position")
        + table(["position", "value t0", "value t1", "PnL"], pos_rows) + "</div>"
        '<div class="panel">'
        + svg_bridge(as_pnl_bars(portfolio_factor_bars(port)), width=760, height=310,
                     caption="by factor, waterfall")
        + legend(residual=False)
        + table(["method", "expected", "explained", "gap", "status"], recon_rows)
        + f'<div class="callout"><p><b>Delta-neutral is not spot-neutral.</b> Taylor&rsquo;s delta '
          f'row for the whole book is {book_delta:,.0f}, which is the hedge doing its job: to first '
          f'order the book has no view on the index. The waterfall&rsquo;s spot row is still '
          f'{book_spot:,.0f}.</p><p>The difference is not an error in either method. It is every '
          f'higher-order spot effect on a seven percent move &mdash; the gamma, the vanna, the '
          f'barrier proximity &mdash; which a full revaluation prices and a first-order hedge does '
          f'not remove. A desk reading only the delta row would think it had no spot exposure on '
          f'the day it made {book_spot / 1e3:,.0f}k from spot.</p></div></div>')

    body = hero + s1 + s2 + s3 + s4 + s5 + s6
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Where did the money go? &mdash; quantark.pnlexplain</title>
<style>{CSS}</style>
</head>
<body>
<main class="wrap">
{body}
<footer>Generated by <code>example/pnl_explain_demo.py</code> on {stamp}. Every number on this
page came from the run that wrote it; nothing is hand-entered.</footer>
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--html-output", nargs="?", const=str(DEFAULT_HTML_OUTPUT), default=None,
        help="write the self-contained HTML lecture page here; the bare flag defaults to "
             "example/data/pnl_explain_lecture_latest.html")
    args = parser.parse_args()

    cases = build_cases(with_orders=args.html_output is not None)
    if args.html_output is None:
        return
    path = Path(args.html_output)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_lecture(cases), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
