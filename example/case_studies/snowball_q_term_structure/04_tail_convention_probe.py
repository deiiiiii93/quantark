"""Stage 04 - why the futures-tenor buckets and the hedging delta do not add up.

The hedging delta is dPV/dS at fixed q(T).  A futures-tenor bucket is
dPV/dF_i at fixed spot.  Because a spot bump at fixed q(T) moves every
forward by the same percentage, the two are tied by the exact identity

    dPV/dS |_q  =  dPV/dS |_F  +  sum_i (F_i / S) * dPV/dF_i |_S

where dPV/dS |_F is spot bumped with every listed IM price pinned (q(T)
rebuilt from the same chain).  This stage computes all three terms
DIRECTLY on one grid date under both tail conventions, prints the chain
under a pinned-futures spot bump, and writes the numbers to
``data/tail_convention_probe.json``.  It shows that the residual term is
entirely the flat-q tail: with the forward-carry tail it vanishes.

Usage:
    .venv/bin/python example/snowball_q_term_structure/04_tail_convention_probe.py
    .venv/bin/python example/snowball_q_term_structure/04_tail_convention_probe.py --date 2024-07-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402

from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine  # noqa: E402
from quantark.asset.equity.param import QuadParams  # noqa: E402
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote  # noqa: E402
from quantark.priceenv import PricingEnvironment  # noqa: E402

MODELS = ("term_flat_q", "term_flat_fwd")
SPOT_BUMP_REL = 0.005   # central spot bump, fraction of spot
FUTURES_BUMP = 1.0      # central bump of one IM price, index points
TAIL_TENORS = (0.75,)   # extra tail tenors to print besides maturity


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", type=date.fromisoformat, default=date(2025, 3, 3),
                   help="grid date present in data/static_risk.csv (default 2025-03-03)")
    p.add_argument("--history-dir", type=Path, default=C.DEFAULT_HISTORY_DIR)
    p.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    p.add_argument("--rate", type=float, default=C.FLAT_RATE)
    p.add_argument("--quad-grid", type=int, default=C.DEFAULT_QUAD_GRID)
    return p.parse_args(argv)


def persisted_coupon(data_dir: Path, d: date) -> float:
    """The fair coupon stage 01 solved for this date under the reference model."""
    risk = pd.read_csv(data_dir / "static_risk.csv")
    row = risk[(risk["model"] == C.REFERENCE_MODEL) & (pd.to_datetime(risk["date"]).dt.date == d)]
    if row.empty:
        raise C.StudyDataError(f"{d} is not a grid date in {data_dir / 'static_risk.csv'}; run stage 01")
    return float(row["coupon"].iloc[0])


def main(argv=None) -> int:
    args = parse_args(argv)
    frames = C.load_history(args.history_dir)
    history = C.surface_history(args.history_dir)
    calendar = C.TradingCalendar.from_frames(frames)
    dataset = C.build_market_dataset(frames, history=history, rate=args.rate)
    ts = pd.Timestamp(args.date)
    chain = frames.futures[frames.futures["date"] == ts]
    market = dataset.get_market_row(ts)
    spot, vol = float(market["spot"]), float(market["volatility"])
    coupon = persisted_coupon(args.data_dir, args.date)
    terms = C.build_terms(args.date, calendar)
    product = C.build_product(terms, spot, coupon)
    engine = SnowballQuadEngine(params=QuadParams(grid_points=args.quad_grid))
    quotes = C.curve_quotes(chain, ts)
    mult = float(quotes[0].multiplier)
    rate = float(args.rate)

    def env(s: float, div: Any) -> PricingEnvironment:
        return PricingEnvironment(
            spot_quote=SpotQuote(spot=float(s), asset_name=C.UNDERLYING_NAME),
            vol_surface=FlatVolSurface(volatility=vol),
            rate_curve=FlatRateCurve(rate=rate),
            div_yield=div,
            valuation_date=ts.to_pydatetime(),
        )

    def div_for(model: str, s: float, chain_slice: pd.DataFrame):
        return C.dividend_for(C.Q_MODELS[model], valuation=ts, spot=s, rate=rate, chain_slice=chain_slice)

    def price(s: float, div: Any) -> float:
        return float(engine.price(product, env(s, div)))

    def forward(s: float, div: Any, t: float) -> float:
        return s * float(np.exp((rate - div.get_yield(t)) * t))

    h = SPOT_BUMP_REL * spot
    print(
        f"{args.date}  spot {spot:.2f}  vol {vol:.4f}  coupon {coupon:.4%}  "
        f"maturity {terms.maturity_years:.3f}y  chain: "
        + ", ".join(f"{q.contract}@{q.price:g} (T={q.maturity:.3f})" for q in quotes)
    )
    out: Dict[str, Any] = {
        "schema_version": C.SCHEMA_VERSION,
        "date": args.date.isoformat(),
        "spot": spot, "vol": vol, "coupon": coupon, "rate": rate,
        "maturity_years": terms.maturity_years, "multiplier": mult,
        "spot_bump_rel": SPOT_BUMP_REL, "futures_bump": FUTURES_BUMP,
        "chain": [{"contract": q.contract, "maturity": q.maturity, "price": q.price} for q in quotes],
        "models": {},
    }

    for model in MODELS:
        d0 = div_for(model, spot, chain)
        p0 = price(spot, d0)
        # (a) hedging delta: spot bumped, the q(T) object held
        delta_q = (price(spot + h, d0) - price(spot - h, d0)) / (2 * h)
        # (b) spot bumped with every IM price pinned: q(T) rebuilt from the same chain
        delta_f = (
            price(spot + h, div_for(model, spot + h, chain)) - price(spot - h, div_for(model, spot - h, chain))
        ) / (2 * h)
        # (c) buckets: one IM price bumped, spot held
        buckets: List[Dict[str, Any]] = []
        for q in quotes:
            def bumped(sign: float):
                c = chain.copy()
                c.loc[c["contract"] == q.contract, "futures_price"] += sign * FUTURES_BUMP
                return div_for(model, spot, c)
            b = (price(spot, bumped(+1.0)) - price(spot, bumped(-1.0))) / (2 * FUTURES_BUMP)
            buckets.append({
                "contract": q.contract, "maturity": q.maturity, "price": q.price,
                "price_over_spot": q.price / spot,
                "implied_q": float(d0.get_yield(q.maturity)),
                "delta_bucket": b, "hands": b / mult, "scaled_hands": b * q.price / spot / mult,
                "extrapolated_tail": bool(q is quotes[-1] and terms.maturity_years > q.maturity),
            })
        scaled = sum(row["scaled_hands"] for row in buckets)
        residual = delta_q / mult - scaled - delta_f / mult

        # the curve under spot +1% with the chain pinned
        d_up = div_for(model, spot * 1.01, chain)
        curve_rows = []
        for t in [q.maturity for q in quotes] + list(TAIL_TENORS) + [terms.maturity_years]:
            f0, f1 = forward(spot, d0, t), forward(spot * 1.01, d_up, t)
            curve_rows.append({
                "maturity": t, "q_before": float(d0.get_yield(t)), "q_after": float(d_up.get_yield(t)),
                "forward_before": f0, "forward_after": f1, "forward_change": f1 / f0 - 1.0,
                "listed": t <= quotes[-1].maturity + 1e-12,
            })

        out["models"][model] = {
            "pv_bp": p0 / C.NOTIONAL * 1e4,
            "delta_fixed_q_hands": delta_q / mult,
            "delta_fixed_futures_hands": delta_f / mult,
            "scaled_bucket_sum_hands": scaled,
            "identity_residual_hands": residual,
            "buckets": buckets,
            "spot_up_1pct_chain_pinned": curve_rows,
        }
        print(f"\n=== {model} ===   PV {p0 / C.NOTIONAL * 1e4:+.1f} bp")
        print(f"  delta at fixed q(T)        {delta_q / mult:+8.2f} hands")
        print(f"  delta at fixed futures     {delta_f / mult:+8.2f} hands   (direct)")
        print(f"  sum (F_i/S) * bucket_i     {scaled:+8.2f} hands")
        print(f"  identity residual          {residual:+8.3f} hands")
        print("  buckets (long holder, hands): " + ", ".join(f"{r['contract']} {r['hands']:+.2f}" for r in buckets))
        print("  spot +1% with the IM prices pinned:")
        for r in curve_rows:
            tag = "" if r["listed"] else "  (tail)"
            print(
                f"    T={r['maturity']:.3f}  q {r['q_before']:+.3%} -> {r['q_after']:+.3%}   "
                f"F {r['forward_before']:.1f} -> {r['forward_after']:.1f}  ({r['forward_change']:+.2%}){tag}"
            )

    args.data_dir.mkdir(parents=True, exist_ok=True)
    C.write_json(args.data_dir / "tail_convention_probe.json", out)
    print(f"\nwrote {args.data_dir / 'tail_convention_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
