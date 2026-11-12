"""Stage 2: pricing-free lifecycle scan of every inception, the pin-event sample and the train/test split.

Reads ``OUTPUT_DIR/chain.pkl`` (``01_data.py``), scans every trading-day inception, selects the headline and pin
sets (spec D5) and writes ``results/events.json`` atomically. Per book: ``ki75`` the 2Y product (every trading-day
inception scanned, headline and pin sets selected) and ``seas6`` the seasoned book (one booking per ISO week, every
booking priced). Prints, per book and slice, the notes with an approach, the approaches, the knocked-in ones and the
distinct ISO weeks, per barrier type.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gpin import CHAIN_PATH, RESULTS_DIR, load_prereg, prereg_hash  # noqa: E402
from gpin.events import (  # noqa: E402
    BOOKS, PRODUCT_BOOK, KoEvent, NoteScan, Sample, counts, event_date, scan, select, slice_of, week_key,
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


def _ki_sizes(approaches: Sequence) -> Dict[str, Any]:
    """KI approaches, knocked-in ones and distinct ISO weeks, per slice."""
    out: Dict[str, Any] = {}
    for sl in SLICES:
        part = [a for a in approaches if slice_of(a.end) == sl]
        out[sl] = {"ki_approaches": len(part), "ki_knocked_in": sum(1 for a in part if a.knocked_in),
                   "ki_weeks": len({week_key(a.end) for a in part})}
    return out


def _restricted_block(scans: Sequence[NoteScan]) -> Dict[str, Any]:
    """Counts over every anchored inception of a book (the notes that can be priced), not only the selected ones."""
    anchored = [s for s in scans if s.anchored]
    approaches = [a for s in anchored for a in s.ki_approaches]
    return {"inceptions": len(anchored), "ki_approaches": len(approaches),
            "notes_knocked_in": sum(1 for s in anchored if s.ki_date is not None),
            "ki_weeks": len({week_key(a.end) for a in approaches}), **_ki_sizes(approaches)}


def _book_block(book: str, scans: Sequence[NoteScan], sample: Sample) -> Dict[str, Any]:
    by_inception = {s.inception: s for s in scans}
    selected = sorted(set(sample.headline) | set(sample.pin))
    ko = [e for e in sample.events if isinstance(e, KoEvent)]
    ki = [e for e in sample.events if not isinstance(e, KoEvent)]
    sizes: Dict[str, Any] = _ki_sizes(ki)
    if book == PRODUCT_BOOK:
        for sl in SLICES:
            ko_s = [e for e in ko if slice_of(e.obs_date) == sl]
            sizes[sl].update(ko_events=len(ko_s), ko_weeks=len({week_key(e.obs_date) for e in ko_s}),
                             ko_fired=sum(1 for e in ko_s if e.fired))
    block: Dict[str, Any] = {
        "ki_pct": BOOKS[book].ki_pct,
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
        "restricted": _restricted_block(scans),
        "ki_approaches": [_event_row(e) for e in ki],
    }
    if book == PRODUCT_BOOK:                 # KO pin events are measured on the product's book only
        block["ko_events"] = [_event_row(e) for e in ko]
    return block


def build_payload(scans_by_book: Mapping[str, Sequence[NoteScan]], samples: Mapping[str, Sample]) -> Dict[str, Any]:
    """The ``events.json`` document: the product's (``ki75``) unrestricted and restricted counts at the top (the
    anchor-free reproduction of the Decisions block), one block per book below."""
    scans = scans_by_book[PRODUCT_BOOK]
    return {
        "preregistration_hash": prereg_hash(),
        "data": {"first": scans[0].inception.isoformat(), "last": scans[-1].end_date.isoformat(), "days": len(scans)},
        "split_date": load_prereg()["split_date"],
        "unrestricted": _counts_block(scans),
        "restricted": _counts_block([s for s in scans if s.anchored]),
        "books": {b: _book_block(b, scans_by_book[b], samples[b]) for b in BOOKS},
    }


def main() -> None:
    chains = load_chain(CHAIN_PATH)
    anchors = anchor_series(chains, load_prereg()["data"]["anchor_window"])
    scans_by_book = {b: scan(chains, anchors, b) for b in BOOKS}
    samples = {b: select(scans_by_book[b], b) for b in BOOKS}
    payload = build_payload(scans_by_book, samples)
    write_json_atomic(EVENTS_PATH, payload)

    scans = scans_by_book[PRODUCT_BOOK]
    print(f"scan: {len(scans)} inceptions {scans[0].inception} .. {scans[-1].inception}, "
          f"{sum(s.anchored for s in scans)} anchored")
    for name in ("unrestricted", "restricted"):
        print(f"{name:>12} (ki75): {payload[name]}")
    for b in BOOKS:
        blk = payload["books"][b]
        z = blk["sample_sizes"]
        print(f"\nbook {b} (KI {BOOKS[b].ki_pct:.0%}, {BOOKS[b].elapsed_months} months elapsed, "
              f"{BOOKS[b].maturity_months} to run): headline notes {z['headline_notes']}, pin notes {z['pin_notes']}, "
              f"priced notes {z['priced_notes']}")
        r = blk["restricted"]
        print(f"  restricted (all {r['inceptions']} anchored inceptions): {r['ki_approaches']} KI approaches, "
              f"{r['notes_knocked_in']} notes knock in, {r['ki_weeks']} distinct weeks; "
              + "; ".join(f"{sl}: {r[sl]['ki_approaches']} approaches / {r[sl]['ki_knocked_in']} knocked in / "
                          f"{r[sl]['ki_weeks']} weeks" for sl in SLICES))
        head = (f"  sample, selected notes: {'slice':>6} {'notes':>6} {'KI approaches':>14} {'knocked in':>11} "
                f"{'KI weeks':>9}")
        if b == PRODUCT_BOOK:
            head += f" {'KO events':>10} {'fired':>6} {'KO weeks':>9}"
        print(head)
        for sl in SLICES:
            n_notes = len({r["inception"] for r in blk["ki_approaches"] if r["slice"] == sl})
            line = (f"  {'':>23}{sl:>6} {n_notes:>6} {z[sl]['ki_approaches']:>14} {z[sl]['ki_knocked_in']:>11} "
                    f"{z[sl]['ki_weeks']:>9}")
            if b == PRODUCT_BOOK:
                line += f" {z[sl]['ko_events']:>10} {z[sl]['ko_fired']:>6} {z[sl]['ko_weeks']:>9}"
            print(line)
        why = Counter(w.split()[0] for p in blk["pin"] for w in p["why"])
        print(f"  pin selections by barrier: {dict(why)}")
    print(f"wrote {EVENTS_PATH}")


if __name__ == "__main__":
    main()
