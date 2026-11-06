"""Stage 2: q(2Y) day-to-day stability under the six tails (design §4.3).

Reads ``OUTPUT_DIR/daily_chain.pkl`` (``01_data.py``); writes ``results/stability.json``:
per variant × tail the daily q(2Y) series, the Δq(2Y) summary (bp of yield;
overall and by day type: ordinary / MO roll / ATM switch) and the ATM-switch
noise floor; per variant the tail ranking on std Δq(2Y) and whether it differs
from the base marks on the same dates.

Variants:
- ``base``: IM close, MO settlement ATM pair, 63-day anchor;
- ``multi_strike``: MO forward = mean of ``K + (C−P)/DF`` over the 5 strikes nearest ATM;
- ``all_settlement``: IM at settlement too (anchor from settlement far-IM carry);
- ``anchor_252``: converging tails on the 252-day anchor, on the dates where it
  exists; the three anchor-free tails are summarised on the same dates.

Every variant's tails share one date window (the dates its anchor exists), so
the comparison across tails is paired.

Run:
    python example/case_studies/snowball_2y_tail_stability/02_stability.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s2ts import ANCHOR_WINDOW, ANCHOR_WINDOW_LONG, HALF_LIVES, RATE, RESULTS_DIR  # noqa: E402
from s2ts.chain import DayChain, load_chain  # noqa: E402
from s2ts.stability import (  # noqa: E402
    DAY_TYPES, Q2Y_TENOR, day_types, floor_distribution, q2y_series, settlement_chain, summarize,
    switch_floor,
)
from s2ts.tails import TAILS, anchor_series  # noqa: E402

OUT_PATH = RESULTS_DIR / "stability.json"

VARIANTS: Dict[str, str] = {
    "base": f"IM close, MO settlement ATM pair, {ANCHOR_WINDOW}-day anchor",
    "multi_strike": "MO forward = mean of K+(C-P)/DF over the 5 strikes nearest ATM",
    "all_settlement": "IM and MO at CFFEX settlement; anchor from the settlement far-IM carry",
    "anchor_252": (f"converging tails on the {ANCHOR_WINDOW_LONG}-day anchor; anchor-free tails "
                   "summarised on the same dates"),
}


def write_json(path: Path, payload: Any) -> None:
    """Deterministic JSON (sorted keys, no NaN), written atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
                   encoding="utf-8")
    tmp.replace(path)


def ranking(tails: Mapping[str, Dict]) -> List[str]:
    return sorted(tails, key=lambda t: tails[t]["summary"]["all"]["std"])


def run_variant(chains: Sequence[DayChain], types: Mapping[date, str], anchors: Mapping[date, float],
                multi_strike: bool) -> Dict[str, Any]:
    days = sorted(anchors)
    window = set(days)
    out: Dict[str, Any] = {}
    for tail in TAILS:
        series = q2y_series(chains, tail, anchors, multi_strike=multi_strike)
        floors = [
            switch_floor(prev, dc, tail, anchors[dc.day], multi_strike=multi_strike)
            for prev, dc in zip(chains, chains[1:])
            if dc.day in window and prev.day in window and types[dc.day] == "atm_switch"
        ]
        out[tail] = {
            "summary": summarize(series, types),
            "switch_floor": floor_distribution(floors),
            "q2y": [round(series[d], 12) for d in days],
        }
    return {
        "first_day": days[0].isoformat(),
        "last_day": days[-1].isoformat(),
        "n_days": len(days),
        "dates": [d.isoformat() for d in days],
        "anchor": [round(anchors[d], 12) for d in days],
        "tails": out,
    }


def main() -> int:
    started = time.time()
    chains = load_chain()
    settle_chains = [settlement_chain(dc) for dc in chains]
    types = day_types(chains)
    a63 = anchor_series(chains, ANCHOR_WINDOW)
    a252 = anchor_series(chains, ANCHOR_WINDOW_LONG)

    variants = {
        "base": run_variant(chains, types, a63, multi_strike=False),
        "multi_strike": run_variant(chains, types, a63, multi_strike=True),
        "all_settlement": run_variant(settle_chains, types, anchor_series(settle_chains, ANCHOR_WINDOW),
                                      multi_strike=False),
        "anchor_252": run_variant(chains, types, a252, multi_strike=False),
    }
    base_series = {t: dict(zip(variants["base"]["dates"], variants["base"]["tails"][t]["q2y"])) for t in TAILS}
    for name, v in variants.items():
        v["description"] = VARIANTS[name]
        v["ranking"] = ranking(v["tails"])
        # the base marks on THIS variant's dates: a paired reference for the ranking
        ref = {}
        for t in TAILS:
            s = {date.fromisoformat(d): base_series[t][d] for d in v["dates"] if d in base_series[t]}
            ref[t] = {"summary": summarize(s, types)}
        v["ranking_reference"] = ranking(ref)
        v["ranking_changed"] = v["ranking"] != v["ranking_reference"]

    counts = {k: sum(1 for d in types.values() if d == k) for k in DAY_TYPES}
    payload = {
        "meta": {
            "generated_by": "02_stability.py",
            "chain_first_day": chains[0].day.isoformat(),
            "chain_last_day": chains[-1].day.isoformat(),
            "chain_days": len(chains),
            "day_type_counts": counts,
            "rate": RATE,
            "q2y_tenor_years": Q2Y_TENOR,
            "anchor_window": ANCHOR_WINDOW,
            "anchor_window_long": ANCHOR_WINDOW_LONG,
            "half_lives": list(HALF_LIVES),
            "tails": list(TAILS),
            "units": "daily change of q(2Y) in bp of yield; q2y series as a decimal yield",
            "mad": "median absolute deviation from the median",
            "switch_floor": ("q2y with today's MO pillar at yesterday's ATM strike (multi_strike: "
                             "yesterday's 5-strike set) minus q2y at today's, same day's settlements; "
                             "ATM-switch days whose previous day is in the window"),
        },
        "variants": variants,
    }
    write_json(OUT_PATH, payload)

    print(f"chain {chains[0].day} .. {chains[-1].day}: {len(chains)} days; day types {counts}")
    for name, v in variants.items():
        print(f"  {name:15s} window {v['first_day']} .. {v['last_day']} ({v['n_days']} days)")
    print()
    print("std dq(2Y), bp of yield/day     | base: all   ordin  switch  roll   floor50 floor90 | "
          "multi   settle  anc252")
    for t in TAILS:
        b = variants["base"]["tails"][t]
        s, f = b["summary"], b["switch_floor"]
        row = [s["all"]["std"], s["ordinary"]["std"], s["atm_switch"]["std"], s["mo_roll"]["std"],
               f["median_abs"], f["p90_abs"]]
        other = [variants[n]["tails"][t]["summary"]["all"]["std"] for n in ("multi_strike", "all_settlement",
                                                                             "anchor_252")]
        print(f"  {t:30s} |       " + " ".join(f"{x:6.2f}" for x in row) + " | "
              + " ".join(f"{x:7.2f}" for x in other))
    print()
    for name, v in variants.items():
        flag = "CHANGED" if v["ranking_changed"] else "same"
        print(f"  ranking {name:15s} {' < '.join(v['ranking'])}  [{flag} vs base marks on these dates]")
    print(f"\nwrote {OUT_PATH} in {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
