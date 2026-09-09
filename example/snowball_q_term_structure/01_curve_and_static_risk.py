"""Stage 01 - Curve anatomy and static snowball risk under each q model.

Two cheap, backtest-free views of the question "does the IM chain's q term
structure matter for a CSI 1000 snowball?":

Part A - curve anatomy (every trading day in the history)
    For each day: the implied q of every listed IM contract (continuous
    compounding, signed), the front and far yields, the flat yield the replay
    engine would use for the FRONT-month hedge and for the LONGEST-contract
    hedge (simple compounding, floored at zero, sticky through the roll
    policy), whether the floor bound, whether the hedge rolled, the last
    listed tenor, the q(1Y) each model would price a 1Y product with, and the
    forward-pricing error of each model against the listed marks.

Part B - static risk (first trading day of each month)
    A fresh 1Y snowball on that day, fair coupon solved under the reference
    model (``term_flat_q``), then PV / delta / gamma / rhoq under every q
    model.  Since PV is zero under the reference model by construction, the
    PV under any other model IS the pricing gap in bp of notional.

    ``delta`` here is the HEDGING delta, dPV/dS at fixed q(T), and
    ``delta_hands = delta / multiplier`` is what the stage-02 fleet actually
    trades (rounded to whole contracts).  For the term model the
    futures-tenor delta and rhoq buckets are a SEPARATE diagnostic:
    ``delta_bucket_i = dPV/dF_i`` holding spot fixed, so the only channel is
    the carry node and dPV/dF_i = -(1/(T_i*F_i)) * dPV/dq(T_i) -- carry risk
    in futures-price units.  The buckets say where along the chain that
    carry sensitivity sits and how much of it lies beyond the last listed
    tenor.  They are ANALYSIS ONLY: they are a partial in a different
    coordinate system from the spot delta, they do not decompose it and do
    not sum to it, and nothing in this study hedges them.

    Each grid row also carries the pricing-measure knock-in and knock-out
    probabilities (``p_ki`` = P(the 75% barrier is ever breached), ``p_ko`` =
    P(any monthly observation knocks out)) from the QUAD event recursion, so
    the summary can report how far each carry model's P(KI) sits from the
    reference and how tightly that gap tracks the q(T) gap.

Part C - the q-only knock-in probe (one roll window)
    One product is frozen at the first grid inception (spot, vol, terms and
    fair coupon) and repriced with nothing but the carry read off each
    trading day of the window around the largest day-to-day jump of the
    front contract's annualised basis.  Every move of P(KI) in that table is
    caused by the carry input alone; it isolates the "irregular front-month q
    => wrong knock-in probability" mechanism from spot and vol.

Outputs (persisted, small):
    data/curve_anatomy.csv          one row per trading day
    data/static_risk.csv            one row per grid date x q model
    data/static_buckets.csv         one row per grid date x listed contract
    data/ki_probe_roll_window.csv   one row per probe day x q model
    data/static_summary.json        aggregate statistics for the report

Run:
    .venv/bin/python example/snowball_q_term_structure/01_curve_and_static_risk.py
    .venv/bin/python example/snowball_q_term_structure/01_curve_and_static_risk.py --every-months 3
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C  # noqa: E402

from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine  # noqa: E402
from quantark.asset.equity.param import QuadParams  # noqa: E402
from quantark.asset.equity.riskmeasures.greeks_calculator import GreeksCalculator  # noqa: E402
from quantark.backtest.replay.market import SignedDividendYield  # noqa: E402
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote  # noqa: E402
from quantark.priceenv import PricingEnvironment  # noqa: E402

RHOQ_BUMP = 0.005  # one-sided up-bump, scaled to per +1% q (calculator convention)


# ---------------------------------------------------------------------------
# Part A - curve anatomy
# ---------------------------------------------------------------------------


def _active_contract_walk(frames: C.HistoryFrames, policy) -> Dict[pd.Timestamp, pd.Series]:
    """Replay the roll policy day by day (it is sticky, so it needs state)."""
    by_date = {d: g for d, g in frames.futures.groupby("date")}
    current: Optional[str] = None
    out: Dict[pd.Timestamp, pd.Series] = {}
    for d in frames.dates:
        chain = by_date.get(d)
        if chain is None or chain.empty:
            continue
        row = policy.select_contract(chain, d, current)
        current = str(row["contract"])
        out[d] = row
    return out


def curve_anatomy(
    frames: C.HistoryFrames,
    *,
    rate: float,
    history=None,
    max_contracts: int = 4,
) -> pd.DataFrame:
    by_date = {d: g for d, g in frames.futures.groupby("date")}
    spot_by_date = dict(zip(frames.dates, frames.spot["spot"].astype(float)))
    front_walk = _active_contract_walk(frames, C.HEDGE_POLICIES["front"]())
    far_walk = _active_contract_walk(frames, C.HEDGE_POLICIES["far"]())
    flat = C.Q_MODELS["flat_from_hedge"]
    prev_front = prev_far = None
    rows: List[Dict[str, Any]] = []
    for d in frames.dates:
        chain = by_date.get(d)
        if chain is None:
            continue
        spot = spot_by_date[d]
        live = C.live_chain(chain, d)
        quotes = C.curve_quotes(chain, d)  # the engine's minimum-tenor rule
        if not quotes:
            continue
        # q(T_i) = r - ln(F_i / S) / T_i, continuous, signed (one node is a
        # valid flat point; IndexFuturesCurve itself needs two)
        yields = [rate - math.log(q.price / spot) / q.maturity for q in quotes]
        tenors = [q.maturity for q in quotes]
        row: Dict[str, Any] = {
            "date": d,
            "spot": spot,
            "n_live": len(live),
            "n_eligible": len(quotes),
            "t_first": tenors[0],
            "t_last": tenors[-1],
            "q_front": yields[0],
            "q_far": yields[-1],
            "q_slope": (
                (yields[-1] - yields[0]) / (tenors[-1] - tenors[0]) if len(quotes) > 1 else np.nan
            ),
        }
        for i in range(max_contracts):
            row[f"contract_{i}"] = quotes[i].contract if i < len(yields) else None
            row[f"t_{i}"] = tenors[i] if i < len(yields) else np.nan
            row[f"q_{i}"] = yields[i] if i < len(yields) else np.nan

        # q(1Y) each model would hand a fresh 1Y product
        term_q = C.dividend_for(C.Q_MODELS["term_flat_q"], valuation=d, spot=spot, rate=rate, chain_slice=chain)
        term_fwd = C.dividend_for(C.Q_MODELS["term_flat_fwd"], valuation=d, spot=spot, rate=rate, chain_slice=chain)
        row["q1y_term_flat_q"] = float(term_q.get_yield(1.0))
        row["q1y_term_flat_fwd"] = float(term_fwd.get_yield(1.0))

        for tag, walk, prev in (("front", front_walk, prev_front), ("far", far_walk, prev_far)):
            active = walk.get(d)
            if active is None:
                continue
            ttm = (pd.Timestamp(active["expiry_date"]) - d).days / C.ACT
            basis = (float(active["futures_price"]) - spot) / spot / ttm
            q_flat = max(0.0, rate - basis)
            flat_div = C.dividend_for(flat, valuation=d, spot=spot, rate=rate, chain_slice=chain, active_row=active)
            errors = C.forward_pricing_error_bp(flat_div, valuation=d, spot=spot, rate=rate, chain_slice=chain)
            row[f"active_{tag}"] = str(active["contract"])
            row[f"ttm_active_{tag}"] = ttm
            row[f"q_flat_{tag}"] = q_flat
            row[f"q_flat_unfloored_{tag}"] = rate - basis
            row[f"floored_{tag}"] = bool(rate - basis < 0.0)
            row[f"roll_{tag}"] = bool(prev is not None and prev != str(active["contract"]))
            row[f"fwd_err_rms_flat_{tag}"] = C.rms(errors.values())
            row[f"fwd_err_max_flat_{tag}"] = max(abs(v) for v in errors.values())
            if tag == "front":
                prev_front = str(active["contract"])
            else:
                prev_far = str(active["contract"])

        if history is not None:
            artifact = history.surface_for(d.date())
            surf = C.dividend_for(C.Q_MODELS["surface_fwd"], valuation=d, spot=spot, rate=rate, chain_slice=chain, artifact=artifact)
            errors = C.forward_pricing_error_bp(surf, valuation=d, spot=spot, rate=rate, chain_slice=chain)
            row["q1y_surface_fwd"] = float(surf.get_yield(1.0))
            row["fwd_err_rms_surface"] = C.rms(errors.values())
            row["surface_max_listed_T"] = float(artifact.max_listed_T)
            opt_tail = C.dividend_for(C.Q_MODELS["term_opt_tail"], valuation=d, spot=spot, rate=rate, chain_slice=chain, artifact=artifact)
            row["q1y_term_opt_tail"] = float(opt_tail.get_yield(1.0))
        rows.append(row)
    return pd.DataFrame(rows)


def anatomy_summary(anatomy: pd.DataFrame) -> Dict[str, Any]:
    def stats(col: str) -> Dict[str, Any]:
        s = anatomy[col].astype(float)
        s = s[np.isfinite(s)]
        return {
            "mean": float(s.mean()) if len(s) else None,
            "median": float(s.median()) if len(s) else None,
            "std": float(s.std(ddof=1)) if len(s) > 1 else None,
            "min": float(s.min()) if len(s) else None,
            "max": float(s.max()) if len(s) else None,
        }

    out: Dict[str, Any] = {
        "days": int(len(anatomy)),
        "first_date": anatomy["date"].min().date().isoformat(),
        "last_date": anatomy["date"].max().date().isoformat(),
        "t_last": stats("t_last"),
        "q_front": stats("q_front"),
        "q_far": stats("q_far"),
        "q_slope": stats("q_slope"),
        "q1y_term_flat_q": stats("q1y_term_flat_q"),
        "q1y_term_flat_fwd": stats("q1y_term_flat_fwd"),
        "q_front_minus_far_abs_mean": float((anatomy["q_front"] - anatomy["q_far"]).abs().mean()),
    }
    for tag in ("front", "far"):
        if f"q_flat_{tag}" not in anatomy:
            continue
        dq = anatomy[f"q_flat_{tag}"].astype(float).diff().abs()
        rolls = anatomy[f"roll_{tag}"].astype(bool)
        out[tag] = {
            "q_flat": stats(f"q_flat_{tag}"),
            "floored_days": int(anatomy[f"floored_{tag}"].sum()),
            "floored_share": float(anatomy[f"floored_{tag}"].mean()),
            "roll_days": int(rolls.sum()),
            "q_flat_daily_abs_change_mean": float(dq.mean()),
            "q_flat_abs_change_on_roll_days": float(dq[rolls].mean()) if rolls.any() else None,
            "q_flat_abs_change_on_other_days": float(dq[~rolls].mean()) if (~rolls).any() else None,
            "fwd_err_rms_flat": stats(f"fwd_err_rms_flat_{tag}"),
            "fwd_err_max_flat": stats(f"fwd_err_max_flat_{tag}"),
        }
    if "fwd_err_rms_surface" in anatomy:
        out["surface"] = {
            "q1y_surface_fwd": stats("q1y_surface_fwd"),
            "fwd_err_rms_surface_vs_futures": stats("fwd_err_rms_surface"),
        }
    if "q1y_term_opt_tail" in anatomy:
        out["q1y_term_opt_tail"] = stats("q1y_term_opt_tail")
        out["q1y_term_opt_tail_daily_abs_change_mean"] = float(
            anatomy["q1y_term_opt_tail"].astype(float).diff().abs().mean()
        )
    dq_term = anatomy["q1y_term_flat_q"].astype(float).diff().abs()
    out["q1y_term_flat_q_daily_abs_change_mean"] = float(dq_term.mean())
    return out


# ---------------------------------------------------------------------------
# Part B - static risk on a monthly grid
# ---------------------------------------------------------------------------


def _shifted(div: Any, bump: float):
    if isinstance(div, SignedDividendYield):
        return SignedDividendYield(div.div_yield + bump)
    return div.parallel_shifted(bump)


def _env(spot: float, vol: float, rate: float, div: Any, valuation: date) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=float(spot), asset_name=C.UNDERLYING_NAME),
        vol_surface=FlatVolSurface(volatility=float(vol)),
        rate_curve=FlatRateCurve(rate=float(rate)),
        div_yield=div,
        valuation_date=pd.Timestamp(valuation).to_pydatetime(),
    )


def grid_dates(calendar: C.TradingCalendar, every_months: int, max_dates: Optional[int]) -> List[date]:
    first_of_month: Dict[tuple, date] = {}
    for d in calendar.days:
        first_of_month.setdefault((d.year, d.month), d)
    keys = sorted(first_of_month)
    out = [first_of_month[k] for i, k in enumerate(keys) if i % max(1, every_months) == 0]
    return out[:max_dates] if max_dates else out


PROBE_BEFORE_DAYS = 8  # trading days shown before the largest front-contract q jump
PROBE_AFTER_DAYS = 3   # trading days shown after it


def _event_probabilities(engine: SnowballQuadEngine, product, env: PricingEnvironment) -> Dict[str, float]:
    """PV, P(KO ever) and P(KI ever) under the pricing measure from the QUAD recursion."""
    st = engine.calculate_event_stats(product, env)
    p_ki = st.ki_ever_probability if st.ki_ever_probability is not None else st.ki_probability
    return {
        "pv_bp": float(st.pv) / C.NOTIONAL * 1e4,
        "p_ko": float(np.sum(st.ko_probability)),
        "p_ki": float(p_ki),
    }


def probe_window(anatomy: pd.DataFrame, *, before_days: int = PROBE_BEFORE_DAYS, after_days: int = PROBE_AFTER_DAYS) -> pd.DatetimeIndex:
    """Trading days around the largest day-to-day jump of the unfloored front-contract flat q.

    The jump day is the second of the two days forming the largest |dq|; the
    window is ``before_days`` trading days before it and ``after_days`` after,
    clipped at the history edges.
    """
    idx = pd.DatetimeIndex(anatomy.index)
    jump = anatomy["q_flat_unfloored_front"].diff().abs()
    centre = int(idx.get_loc(jump.idxmax()))
    lo, hi = max(0, centre - before_days), min(len(idx) - 1, centre + after_days)
    return idx[lo : hi + 1]


def ki_probe(
    frames: C.HistoryFrames,
    calendar: C.TradingCalendar,
    dataset,
    anatomy: pd.DataFrame,
    *,
    rate: float,
    models: List[str],
    history,
    coupon: float,
    inception: date,
    quad_grid: int,
    before_days: int = PROBE_BEFORE_DAYS,
    after_days: int = PROBE_AFTER_DAYS,
) -> pd.DataFrame:
    """Hold one product, spot and vol fixed; feed the pricer only the q read off each day of the window.

    Everything except the carry input is pinned at ``inception`` (its spot, vol,
    terms and fair ``coupon``), so the day-to-day movement of P(KI) in the
    output is caused by the carry model alone.
    """
    engine = SnowballQuadEngine(params=QuadParams(grid_points=quad_grid))
    by_date = {d: g for d, g in frames.futures.groupby("date")}
    front_walk = _active_contract_walk(frames, C.HEDGE_POLICIES["front"]())
    anatomy = anatomy.set_index("date") if "date" in anatomy.columns else anatomy
    market = dataset.get_market_row(pd.Timestamp(inception))
    spot, vol = float(market["spot"]), float(market["volatility"])
    terms = C.build_terms(inception, calendar)
    product = C.build_product(terms, spot, coupon)
    rows: List[Dict[str, Any]] = []
    for ts in probe_window(anatomy, before_days=before_days, after_days=after_days):
        chain = by_date[ts]
        artifact = history.surface_for(ts.date()) if history is not None else None
        for name in models:
            div = C.dividend_for(
                C.Q_MODELS[name], valuation=ts, spot=float(anatomy.loc[ts, "spot"]), rate=rate,
                chain_slice=chain, active_row=front_walk[ts], artifact=artifact,
            )
            rows.append(
                {
                    "date": ts.date(),
                    "model": name,
                    "ttm_front": float(anatomy.loc[ts, "ttm_active_front"]),
                    "roll_front": bool(anatomy.loc[ts, "roll_front"]),
                    "q_T": float(div.get_yield(terms.maturity_years)),
                    **_event_probabilities(engine, product, _env(spot, vol, rate, div, inception)),
                }
            )
    return pd.DataFrame(rows)


def ki_probe_summary(probe: pd.DataFrame) -> Dict[str, Any]:
    dates = sorted(pd.to_datetime(probe["date"]).dt.date.unique())
    out: Dict[str, Any] = {
        "n_days": len(dates),
        "first_date": dates[0].isoformat(),
        "last_date": dates[-1].isoformat(),
        "models": {},
    }
    for name, g in probe.groupby("model", sort=False):
        g = g.sort_values("date")
        out["models"][str(name)] = {
            "q_T": C.describe(g["q_T"]),
            "p_ki": C.describe(g["p_ki"]),
            "p_ko": C.describe(g["p_ko"]),
            "pv_bp": C.describe(g["pv_bp"]),
            "p_ki_range": float(g["p_ki"].max() - g["p_ki"].min()),
            "p_ki_daily_abs_change_mean": float(g["p_ki"].diff().abs().mean()),
            "q_T_daily_abs_change_mean": float(g["q_T"].diff().abs().mean()),
        }
    return out


def static_risk(
    frames: C.HistoryFrames,
    calendar: C.TradingCalendar,
    dataset,
    *,
    rate: float,
    models: List[str],
    history,
    every_months: int,
    max_dates: Optional[int],
    quad_grid: int,
) -> tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    engine = SnowballQuadEngine(params=QuadParams(grid_points=quad_grid))
    calc = GreeksCalculator()
    by_date = {d: g for d, g in frames.futures.groupby("date")}
    front_walk = _active_contract_walk(frames, C.HEDGE_POLICIES["front"]())
    rows: List[Dict[str, Any]] = []
    bucket_rows: List[Dict[str, Any]] = []
    coupons: List[Dict[str, Any]] = []
    dates = grid_dates(calendar, every_months, max_dates)
    for n, d in enumerate(dates, 1):
        ts = pd.Timestamp(d)
        chain = by_date[ts]
        market = dataset.get_market_row(ts)
        spot, vol = float(market["spot"]), float(market["volatility"])
        terms = C.build_terms(d, calendar)
        active = front_walk[ts]
        artifact = history.surface_for(d) if history is not None else None

        divs = {
            name: C.dividend_for(
                C.Q_MODELS[name], valuation=ts, spot=spot, rate=rate, chain_slice=chain,
                active_row=active, artifact=artifact,
            )
            for name in models
        }
        started = time.perf_counter()
        ref_env = _env(spot, vol, rate, divs[C.REFERENCE_MODEL], d)
        solution = C.solve_fair_coupon(
            lambda c: engine.price(C.build_product(terms, spot, c), ref_env)
        )
        product = C.build_product(terms, spot, solution.coupon)
        coupons.append({"date": d.isoformat(), "spot": spot, "vol": vol, **solution.summary()})

        quotes = C.curve_quotes(chain, ts)
        curve = C.futures_curve(chain, ts, spot) if len(quotes) >= 2 else None
        t_last = quotes[-1].maturity
        tail_share = max(0.0, terms.maturity_years - t_last) / terms.maturity_years
        ko_beyond = int(sum(1 for t in terms.ko_times if t > t_last))

        for name in models:
            env = _env(spot, vol, rate, divs[name], d)
            greeks = engine.calculate_greeks(product, env)
            price = float(greeks["price"])
            bumped = _env(spot, vol, rate, _shifted(divs[name], RHOQ_BUMP), d)
            rhoq = (float(engine.price(product, bumped)) - price) / RHOQ_BUMP * 0.01
            events = _event_probabilities(engine, product, env)
            rows.append(
                {
                    "date": d,
                    "model": name,
                    "spot": spot,
                    "vol": vol,
                    "coupon": solution.coupon,
                    "t_last": t_last,
                    "tail_time_share": tail_share,
                    "ko_obs_beyond_last": ko_beyond,
                    "q_at_maturity": float(divs[name].get_yield(terms.maturity_years)),
                    "pv_bp": price / C.NOTIONAL * 1e4,
                    "delta": float(greeks["delta"]),
                    "delta_hands": float(greeks["delta"]) / C.FUTURES_MULTIPLIER,
                    "delta_cash_1pct_bp": float(greeks["delta"]) * spot * 0.01 / C.NOTIONAL * 1e4,
                    "gamma_cash_1pct_bp": float(greeks.get("gamma", np.nan)) * spot ** 2 / 100.0 / C.NOTIONAL * 1e4,
                    "rhoq_1pct_bp": rhoq / C.NOTIONAL * 1e4,
                    "p_ki": events["p_ki"],
                    "p_ko": events["p_ko"],
                }
            )
            if name == "term_flat_q" and curve is not None:
                delta_buckets = calc.calculate_futures_delta_buckets(product, env, engine, curve)
                rhoq_buckets = calc.calculate_futures_rhoq_buckets(product, env, engine, curve)
                for db, rb in zip(delta_buckets, rhoq_buckets):
                    bucket_rows.append(
                        {
                            "date": d,
                            "contract": db["contract"],
                            "maturity": db["maturity"],
                            "future_price": db["future_price"],
                            "delta_bucket": db["delta_bucket"],
                            "hedge_hands": db["hedge_hands"],
                            "extrapolated_tail": bool(db["extrapolated_tail"]),
                            "rhoq_bucket_bp": float(rb["rhoq_bucket"]) / C.NOTIONAL * 1e4,
                        }
                    )
        print(
            f"[{n:2d}/{len(dates)}] {d} spot={spot:,.1f} vol={vol:.3f} "
            f"coupon={solution.coupon:.4%} t_last={t_last:.2f} "
            f"({time.perf_counter() - started:.1f}s)",
            flush=True,
        )
    return pd.DataFrame(rows), pd.DataFrame(bucket_rows), coupons


def static_summary(risk: pd.DataFrame, buckets: pd.DataFrame, models: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"grid_dates": int(risk["date"].nunique()), "models": {}}
    ref = risk[risk["model"] == C.REFERENCE_MODEL].set_index("date")
    base = risk[risk["model"] == C.BASELINE_MODEL].set_index("date")
    for name in models:
        sub = risk[risk["model"] == name].set_index("date")
        entry = {
            "pv_bp": C.describe(sub["pv_bp"]),
            "delta_hands": C.describe(sub["delta_hands"]),
            "rhoq_1pct_bp": C.describe(sub["rhoq_1pct_bp"]),
            "q_at_maturity": C.describe(sub["q_at_maturity"]),
            "p_ki": C.describe(sub["p_ki"]),
            "p_ko": C.describe(sub["p_ko"]),
        }
        if name != C.REFERENCE_MODEL:
            entry["pv_gap_vs_reference_bp"] = C.describe(sub["pv_bp"] - ref["pv_bp"])
            entry["delta_hands_gap_vs_reference"] = C.describe(sub["delta_hands"] - ref["delta_hands"])
            ki_gap = sub["p_ki"] - ref["p_ki"]
            q_gap = sub["q_at_maturity"] - ref["q_at_maturity"]
            entry["p_ki_gap_vs_reference"] = C.describe(ki_gap)
            entry["p_ko_gap_vs_reference"] = C.describe(sub["p_ko"] - ref["p_ko"])
            entry["p_ki_gap_vs_q_gap_corr"] = (
                float(np.corrcoef(q_gap, ki_gap)[0, 1]) if len(ki_gap) > 1 and q_gap.std() > 0 and ki_gap.std() > 0 else None
            )
        if name != C.BASELINE_MODEL:
            entry["pv_gap_vs_baseline_bp"] = C.describe(sub["pv_bp"] - base["pv_bp"])
            entry["delta_hands_gap_vs_baseline"] = C.describe(sub["delta_hands"] - base["delta_hands"])
        out["models"][name] = entry
    out["tail_time_share"] = C.describe(ref["tail_time_share"])
    out["ko_obs_beyond_last"] = C.describe(ref["ko_obs_beyond_last"])
    if len(buckets):
        share = (
            buckets.assign(abs_hands=buckets["hedge_hands"].abs())
            .groupby("date")
            .apply(lambda g: g.loc[g["extrapolated_tail"], "abs_hands"].sum() / g["abs_hands"].sum() if g["abs_hands"].sum() > 0 else np.nan)
        )
        out["last_bucket_hands_share"] = C.describe(share)
        rank = buckets.sort_values(["date", "maturity"]).groupby("date").cumcount()
        buckets = buckets.assign(rank=rank.values)
        out["hands_by_rank"] = {
            int(r): C.describe(g["hedge_hands"]) for r, g in buckets.groupby("rank")
        }
    return out


# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--history-dir", type=Path, default=C.DEFAULT_HISTORY_DIR)
    parser.add_argument("--data-dir", type=Path, default=C.DATA_DIR)
    parser.add_argument("--rate", type=float, default=C.FLAT_RATE)
    parser.add_argument("--models", nargs="+", default=list(C.Q_MODEL_ORDER), choices=list(C.Q_MODEL_ORDER))
    parser.add_argument("--every-months", type=int, default=1, help="static grid spacing")
    parser.add_argument("--max-dates", type=int, default=None)
    parser.add_argument("--quad-grid", type=int, default=C.DEFAULT_QUAD_GRID)
    parser.add_argument("--skip-static", action="store_true", help="curve anatomy only")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if C.REFERENCE_MODEL not in args.models:
        args.models = [C.REFERENCE_MODEL] + list(args.models)
    frames = C.load_history(args.history_dir)
    history = C.surface_history(args.history_dir)
    calendar = C.TradingCalendar.from_frames(frames)
    args.data_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    anatomy = curve_anatomy(frames, rate=args.rate, history=history)
    anatomy.to_csv(args.data_dir / "curve_anatomy.csv", index=False)
    summary: Dict[str, Any] = {
        "schema_version": C.SCHEMA_VERSION,
        "rate": args.rate,
        "anatomy": anatomy_summary(anatomy),
    }
    print(f"curve anatomy: {len(anatomy)} days in {time.perf_counter() - started:.1f}s")
    a = summary["anatomy"]
    print(
        f"  q_front mean {a['q_front']['mean']:+.2%} (std {a['q_front']['std']:.2%}), "
        f"q_far mean {a['q_far']['mean']:+.2%} (std {a['q_far']['std']:.2%}), "
        f"last listed tenor median {a['t_last']['median']:.2f}y"
    )
    for tag in ("front", "far"):
        t = a[tag]
        print(
            f"  flat q from {tag}: floored on {t['floored_days']} days, {t['roll_days']} rolls, "
            f"|dq| on roll days {t['q_flat_abs_change_on_roll_days'] or float('nan'):.2%} vs "
            f"{t['q_flat_abs_change_on_other_days'] or float('nan'):.2%} otherwise; "
            f"forward error RMS {t['fwd_err_rms_flat']['mean']:.0f} bp"
        )

    if not args.skip_static:
        dataset = C.build_market_dataset(frames, history=history, rate=args.rate)
        risk, buckets, coupons = static_risk(
            frames, calendar, dataset, rate=args.rate, models=args.models, history=history,
            every_months=args.every_months, max_dates=args.max_dates, quad_grid=args.quad_grid,
        )
        risk.to_csv(args.data_dir / "static_risk.csv", index=False)
        buckets.to_csv(args.data_dir / "static_buckets.csv", index=False)
        summary["static"] = static_summary(risk, buckets, args.models)
        summary["static"]["coupons"] = coupons
        summary["static"]["quad_grid"] = args.quad_grid
        st = summary["static"]
        for name in args.models:
            m = st["models"][name]
            gap = m.get("pv_gap_vs_reference_bp", {}).get("mean")
            dgap = m.get("delta_hands_gap_vs_reference", {}).get("mean")
            print(
                f"  {name:14s} PV {m['pv_bp']['mean']:+8.1f} bp  delta {m['delta_hands']['mean']:8.1f} hands  "
                f"rhoq {m['rhoq_1pct_bp']['mean']:+7.1f} bp/1%  "
                + (f"gap vs ref {gap:+7.1f} bp, {dgap:+6.1f} hands" if gap is not None else "(reference)")
            )
        print(
            f"  tail beyond last listed tenor: {st['tail_time_share']['mean']:.0%} of the life, "
            f"{st['ko_obs_beyond_last']['mean']:.1f} of {C.MATURITY_MONTHS - C.LOCKOUT_MONTHS + 1} KO dates"
        )
        for name in args.models:
            m = st["models"][name]
            gap = m.get("p_ki_gap_vs_reference", {})
            print(
                f"  {name:14s} P(KI) {m['p_ki']['mean']:.3f}  P(KO) {m['p_ko']['mean']:.3f}"
                + (f"  KI gap vs ref {gap['mean']:+.3f} (min {gap['min']:+.3f}, max {gap['max']:+.3f}, "
                   f"corr with q gap {corr:.2f})" if gap and (corr := m.get("p_ki_gap_vs_q_gap_corr")) is not None
                   else f"  KI gap vs ref {gap['mean']:+.3f}" if gap else "  (reference)")
            )
        first = coupons[0]
        probe = ki_probe(
            frames, calendar, dataset, anatomy, rate=args.rate, models=args.models, history=history,
            coupon=float(first["coupon"]), inception=date.fromisoformat(first["date"]), quad_grid=args.quad_grid,
        )
        probe.to_csv(args.data_dir / "ki_probe_roll_window.csv", index=False)
        summary["static"]["ki_probe"] = {"inception": first["date"], "coupon": first["coupon"], **ki_probe_summary(probe)}
        kp = summary["static"]["ki_probe"]
        print(f"  q-only probe {kp['first_date']}..{kp['last_date']} ({kp['n_days']} days), product fixed at {first['date']}:")
        for name in args.models:
            m = kp["models"][name]
            print(
                f"    {name:14s} P(KI) {m['p_ki']['min']:.3f}..{m['p_ki']['max']:.3f}  "
                f"day-to-day |dP(KI)| {m['p_ki_daily_abs_change_mean']:.3f}  q(T) {m['q_T']['min']:+.1%}..{m['q_T']['max']:+.1%}"
            )
    C.write_json(args.data_dir / "static_summary.json", summary)
    print(f"wrote {args.data_dir / 'static_summary.json'} ({time.perf_counter() - started:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
