"""Stage 2: pricing-free lifecycle scan of every inception, the pin-event sample and the train/test split.

Reads ``OUTPUT_DIR/chain.pkl`` (``01_data.py``), scans every trading-day inception, selects the headline and pin
sets (spec D5) and writes ``results/events.json`` atomically. Prints the sample sizes per slice and barrier type.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpin import CHAIN_PATH, RESULTS_DIR, load_prereg, prereg_hash  # noqa: E402
from gpin.events import (  # noqa: E402
    KoEvent, NoteScan, Sample, counts, event_date, scan, select, slice_of, week_key,
)
from s2ts.chain import load_chain  # noqa: E402
from s2ts.tails import anchor_series  # noqa: E402

EVENTS_PATH = RESULTS_DIR / "events.json"
SLICES = ("train", "test")


def write_json_atomic(path: Path, payload: Any) -> None:
    """Sorted keys, indent 2; tmp file (fsynced) then ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _counts_block(scans: Sequence[NoteScan]) -> Dict[str, int]:
    block = counts(scans)
    block["ko_pin_weeks"] = len({week_key(e.obs_date) for s in scans for e in s.ko_events})
    return block


def _event_row(event) -> Dict[str, Any]:
    d = event_date(event)
    row: Dict[str, Any] = {"inception": event.inception.isoformat(), "date": d.isoformat(),
                           "slice": slice_of(d), "week": week_key(d)}
    if isinstance(event, KoEvent):
        row.update(prior_gap=event.prior_gap, fired=event.fired)
    else:
        row.update(start=event.start.isoformat(), knocked_in=event.knocked_in)
    return row


def build_payload(scans: Sequence[NoteScan], sample: Sample) -> Dict[str, Any]:
    """The ``events.json`` document."""
    by_inception = {s.inception: s for s in scans}
    selected = sorted(set(sample.headline) | set(sample.pin))
    ko = [e for e in sample.events if isinstance(e, KoEvent)]
    ki = [e for e in sample.events if not isinstance(e, KoEvent)]
    sizes: Dict[str, Any] = {}
    for sl in SLICES:
        ko_s = [e for e in ko if slice_of(e.obs_date) == sl]
        ki_s = [e for e in ki if slice_of(e.end) == sl]
        sizes[sl] = {"ko_events": len(ko_s), "ko_weeks": len({week_key(e.obs_date) for e in ko_s}),
                     "ko_fired": sum(1 for e in ko_s if e.fired), "ki_approaches": len(ki_s),
                     "ki_knocked_in": sum(1 for a in ki_s if a.knocked_in),
                     "ki_weeks": len({week_key(a.end) for a in ki_s})}
    return {
        "preregistration_hash": prereg_hash(),
        "data": {"first": scans[0].inception.isoformat(), "last": scans[-1].end_date.isoformat(), "days": len(scans)},
        "split_date": load_prereg()["split_date"],
        "unrestricted": _counts_block(scans),
        "restricted": _counts_block([s for s in scans if s.anchored]),
        "headline": [d.isoformat() for d in sample.headline],
        "pin": [{"inception": d.isoformat(), "why": list(sample.reasons[d])} for d in sample.pin],
        "notes": {d.isoformat(): {"ko_date": None if by_inception[d].ko_date is None else by_inception[d].ko_date.isoformat(),
                                  "ki_date": None if by_inception[d].ki_date is None else by_inception[d].ki_date.isoformat(),
                                  "end_reason": by_inception[d].end_reason,
                                  "end_date": by_inception[d].end_date.isoformat(),
                                  "ko_observations": by_inception[d].ko_observations}
                  for d in selected},
        "sample_sizes": {"headline_notes": len(sample.headline), "pin_notes": len(sample.pin),
                         "priced_notes": len(selected), **sizes},
        "ko_events": [_event_row(e) for e in ko],
        "ki_approaches": [_event_row(e) for e in ki],
    }


def main() -> None:
    chains = load_chain(CHAIN_PATH)
    anchors = anchor_series(chains, load_prereg()["data"]["anchor_window"])
    scans = scan(chains, anchors)
    sample = select(scans)
    payload = build_payload(scans, sample)
    write_json_atomic(EVENTS_PATH, payload)

    print(f"scan: {len(scans)} inceptions {scans[0].inception} .. {scans[-1].inception}, "
          f"{sum(s.anchored for s in scans)} anchored")
    for name in ("unrestricted", "restricted"):
        print(f"{name:>12}: {payload[name]}")
    print(f"headline notes {len(sample.headline)}, pin notes {len(sample.pin)}, "
          f"priced notes {payload['sample_sizes']['priced_notes']}")
    print(f"{'slice':>6} {'KO events':>10} {'fired':>6} {'KO weeks':>9} {'KI approaches':>14} {'knocked in':>11} {'KI weeks':>9}")
    for sl in SLICES:
        z = payload["sample_sizes"][sl]
        print(f"{sl:>6} {z['ko_events']:>10} {z['ko_fired']:>6} {z['ko_weeks']:>9} {z['ki_approaches']:>14} "
              f"{z['ki_knocked_in']:>11} {z['ki_weeks']:>9}")
    why = Counter(r.split()[0] for d in sample.pin for r in sample.reasons[d])
    print(f"pin selections by barrier: {dict(why)}")
    print(f"wrote {EVENTS_PATH}")


if __name__ == "__main__":
    main()
