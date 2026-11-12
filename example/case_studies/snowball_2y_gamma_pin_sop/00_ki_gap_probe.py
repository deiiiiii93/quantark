"""Stage 0: the KI gap probe. Where does the knock-in pin live? (RESEARCH.md section 9, 2026-11-12.)

For a FRESH note with its spot 3% above its knock-in barrier, the engine value of the note with the knock-in flag
off minus the value with the flag on is what the seller GAINS when the note knocks in (the gap, bp of notional), and
the difference of the two hedges (1% bump delta, % of notional) is the futures the seller must sell at the knock-in
(the hedge step). Barriers 75 / 85 / 90% of the strike and 1, 3, 6, 12, 18 and 24 months left.

Fixed inputs: coupon 20% p.a. (a KO cash of ``coupon x t x notional`` at each monthly observation from month 1), flat
vol 25%, dividend yield q 5%, ``r`` = the study's rate, daily KI observations (244 a year), principal excluded,
notional 10 mio, the QUAD V2 engine with its default parameters. The strike and KO level are set so that the booking
spot is 3% above the KI barrier. No market data is read.

The knocked-in side is priced with a lifecycle-state copy forced to knocked-in, the mechanism the engine reads
(``quantark/asset/equity/engine/quad/v2/adapters.py`` takes ``lifecycle_state.knocked_in`` first): the same state
transformation as the ``kin`` hedge model (``gpin.models``), never a product attribute.

Writes ``results/ki_gap_probe.json`` atomically.

Run: python example/case_studies/snowball_2y_gamma_pin_sop/00_ki_gap_probe.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantark.asset.equity.lifecycle.autocallable import AutocallableLifecycleTracker  # noqa: E402
from quantark.asset.equity.product.option.snowball_helpers import create_standard_snowball  # noqa: E402
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote  # noqa: E402
from quantark.param.div.dividend_yield import ContinuousDividendYield  # noqa: E402
from quantark.priceenv.pricing_environment import PricingEnvironment  # noqa: E402
from quantark.util.enum import ObservationType  # noqa: E402

from gpin import RESULTS_DIR, load_prereg  # noqa: E402
from gpin.models import MODELS  # noqa: E402
from s2ts import NOTIONAL, RATE  # noqa: E402
from s2ts.runner import v2_engine  # noqa: E402

OUT_PATH = RESULTS_DIR / "ki_gap_probe.json"

BARRIERS: Tuple[float, ...] = (0.75, 0.85, 0.90)
MONTHS_LEFT: Tuple[int, ...] = (1, 3, 6, 12, 18, 24)
COUPON = 0.20
VOL = 0.25
DIV_YIELD = 0.05
ABOVE_BARRIER = 0.03
SPOT = 6000.0
BUMP = 0.01
KI_DAYS_PER_YEAR = int(load_prereg()["windows"]["days_per_year"])
VALUATION = datetime(2025, 1, 2)
BP = 1.0e4

SELLER_VIEW = ("The gap is the seller's gain when the note knocks in: the not-knocked-in note is the larger liability. "
               "The hedge step is the long futures the seller must sell on the knock-in day.")


def fresh_note(ki_pct: float, months: int):
    """The note of ``months`` left struck so that the booking spot ``SPOT`` is ``ABOVE_BARRIER`` above its KI barrier:
    strike = KO level = ``SPOT / (ki_pct x (1 + above))``, KI = ``ki_pct x strike``, monthly KO from month 1."""
    strike = SPOT / (ki_pct * (1.0 + ABOVE_BARRIER))
    years = months / 12.0
    ko_times = [m / 12.0 for m in range(1, months + 1)]
    n_days = int(round(KI_DAYS_PER_YEAR * years))
    ki_times = [k * years / n_days for k in range(1, n_days + 1)]
    return create_standard_snowball(
        initial_price=strike, strike=strike, maturity=years, contract_multiplier=NOTIONAL / strike,
        ko_barrier=strike, ko_rate=COUPON, ki_barrier=ki_pct * strike, num_observations=len(ko_times),
        is_reverse=False, ko_observation_dates=ko_times, ki_continuous=False,
        ki_observation_type=ObservationType.DISCRETE, ki_observation_dates=ki_times,
        rebate_rate=COUPON, include_principal=False)


def probe_env() -> PricingEnvironment:
    return PricingEnvironment(rate_curve=FlatRateCurve(RATE), valuation_date=VALUATION,
                              spot_quote=SpotQuote(spot=SPOT), vol_surface=FlatVolSurface(volatility=VOL),
                              div_yield=ContinuousDividendYield(DIV_YIELD))


def price_both(engine, ki_pct: float, months: int):
    """``(alive, knocked_in, spots)``: the engine's ``{"price", "delta", ...}`` on the three spots (1 - bump, 1, 1 + bump)
    of the not-knocked-in note and of the same note with a lifecycle state forced to knocked-in."""
    product = fresh_note(ki_pct, months)
    env = probe_env()
    state = AutocallableLifecycleTracker(product=product, quantity=1.0, start_date=pd.Timestamp(VALUATION)).lifecycle
    if state.knocked_in:
        raise RuntimeError("a fresh tracker state is not knocked in")
    spots = np.array([SPOT * (1.0 - BUMP), SPOT, SPOT * (1.0 + BUMP)])
    alive = engine.prepare(product, env, spots, lifecycle_state=state).evaluate(spots)
    forced = MODELS["kin"].lifecycle(state)                    # a copy: ``state`` itself is untouched
    knocked_in = engine.prepare(product, env, spots, lifecycle_state=forced).evaluate(spots)
    return alive, knocked_in, spots


def probe_cell(engine, ki_pct: float, months: int) -> Dict[str, Any]:
    """One row: gap in bp of notional and the hedges (central 1% bump delta, % of notional) of both sides."""
    alive, knocked_in, spots = price_both(engine, ki_pct, months)
    width = float(spots[2] - spots[0])

    def hedge_pct(r) -> float:
        return 100.0 * float((r["price"][2] - r["price"][0]) / width) * SPOT / NOTIONAL

    hedge_alive, hedge_ki = hedge_pct(alive), hedge_pct(knocked_in)
    return {"ki_pct": ki_pct, "months_left": months,
            "gap_bp": BP * float(alive["price"][1] - knocked_in["price"][1]) / NOTIONAL,
            "hedge_alive_pct": hedge_alive, "hedge_ki_pct": hedge_ki, "hedge_step_pct": hedge_alive - hedge_ki}


def build_payload(engine, barriers: Sequence[float] = BARRIERS, months: Sequence[int] = MONTHS_LEFT) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = [probe_cell(engine, k, m) for k in barriers for m in months]
    return {
        "title": "KI gap probe: a fresh note 3% above its knock-in barrier",
        "inputs": {"coupon": COUPON, "vol": VOL, "q": DIV_YIELD, "r": RATE, "above_barrier": ABOVE_BARRIER,
                   "spot": SPOT, "notional": NOTIONAL, "bump": BUMP, "ko": "monthly from month 1, level = strike",
                   "ki": f"daily, {KI_DAYS_PER_YEAR} per year, level = ki_pct x strike", "engine": "SnowballQuadEngineV2 defaults"},
        "barriers": list(barriers), "months_left": list(months),
        "units": {"gap_bp": "bp of notional: not-knocked-in engine price minus knocked-in engine price",
                  "hedge_*_pct": "central 1% bump delta x spot / notional, in percent",
                  "hedge_step_pct": "hedge of the not-knocked-in note minus that of the knocked-in note"},
        "seller_view": SELLER_VIEW,
        "rows": rows,
    }


def write_json_atomic(path: Path, payload: Any) -> None:
    """Sorted keys, indent 2; tmp file (fsynced) then ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def main() -> None:
    payload = build_payload(v2_engine())
    write_json_atomic(OUT_PATH, payload)
    print("fresh note, spot 3% above the KI barrier; gap = the seller's gain on a knock-in (bp of notional);")
    print("hedge = central 1% bump delta, % of notional; step = hedge (not knocked in) - hedge (knocked in)")
    print(f"{'KI':>5} {'months left':>12} {'gap bp':>9} {'hedge not-KI':>13} {'hedge KI':>9} {'step':>7}")
    for r in payload["rows"]:
        print(f"{r['ki_pct']:5.2f} {r['months_left']:12d} {r['gap_bp']:9.1f} {r['hedge_alive_pct']:13.1f} "
              f"{r['hedge_ki_pct']:9.1f} {r['hedge_step_pct']:7.1f}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
