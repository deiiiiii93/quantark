"""Regenerate the banked references of the 3C tests (spec 3C §7.2): a provenance act, run by hand.

    PYTHONPATH=$PWD:$PWD/test:<dir holding mpmath> python test/pde_routes/build_reference.py [EVIDENCE_DIR]

`mpmath` must be on PYTHONPATH; it is NOT a project dependency and no test imports it. EVIDENCE_DIR defaults
to the 3C evidence directory of the main checkout (untracked).

- `no_touch_reference.json`: the double no-touch probability at the exact double inputs, from
  `no_touch_series.image`. The `z1`, `stress`, `random_reachable` and `random_any` cells are the evidence's;
  each reference is RECOMPUTED here and must equal the evidence's to the last bit, else the build stops (the
  reference or the input moved). `equity` is rebuilt as `w1d_engines_on_stable.py` builds it (planning finding
  P14), `engine` holds the three cells the analytic engine test reads, `reviewer` the cells of reviews 2 and 3.
- `reference_values.json`: the kernel oracle's Richardson values on the `book` surface (`l2d_final_book.json`)
  and the finest rung of the skewed local-vol DCN ladder (`d6_localvol.json`, contract A).
"""
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/Users/fuxinyao/quant-ark/docs/superpowers/specs/2026-10-04-pde-1d-remaining-routes-evidence")

LO, UP = 80.0, 130.0
RATE, DIV = 0.03, 0.01


def _image(args):
    from pde_routes.no_touch_series import image

    return float(image(*args))


def _recomputed(cells, label):
    out = []
    for c in cells:
        args = [float(a) for a in c["args"]]
        ref = _image(args)
        if ref != float(c["reference"]):
            raise SystemExit(f"{label}: reference moved at {args}: {ref!r} != {c['reference']!r}")
        out.append({"args": args, "reference": ref})
    return out


def _computed(arg_list):
    return [{"args": [float(a) for a in args], "reference": _image(args)} for args in arg_list]


def no_touch_cells():
    w1b = json.loads((E / "w1b_stable_candidate.json").read_text())
    w1c = json.loads((E / "w1c_random_sweep.json").read_text())
    equity = [(spot, days / 365.0, RATE, DIV, vol, LO, UP)
              for vol in (0.22, 0.08)
              for days in (1, 7, 30, 91, 365, 1095)
              for spot in (80.01, 81.0, 85.0, 100.0, 120.0, 127.0, 129.0, 129.9, 129.99)]
    engine = [(spot, 1.0, RATE, DIV, 0.22, LO, UP) for spot in (100.0, 90.0, 120.0)]
    reviewer = [tuple(w1b["reviewer"]["args"]), (129.9, 1.0 / 365.0, 0.5 * 0.01 ** 2, 0.0, 0.01, LO, UP)]
    return {
        "z1": _recomputed(w1b["z1_cells"], "z1"),
        "equity": _computed(equity),
        "stress": _recomputed(w1b["stress"], "stress"),
        "random_reachable": _recomputed(w1c["reachable"], "random_reachable"),
        "random_any": _recomputed(w1c["any"], "random_any"),
        "reviewer": _computed(reviewer),
        "engine": _computed(engine),
    }


def lv_book():
    rows = json.loads((E / "l2d_final_book.json").read_text())["rows"]
    return [{"family": r["family"], "value": r["ref"], "spot": r["meta"].get("spot", 100.0), **r["meta"]} for r in rows]


def dcn_skew_finest():
    skew = json.loads((E / "d6_localvol.json").read_text())["skew"]
    return {"A": [{"moneyness": r["moneyness"], "value": r["reference"], "uncertainty_bp": r["uncertainty_bp"]}
                  for r in skew]}


def provenance(sources):
    import mpmath

    head = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"source": sources, "tree": head, "date": datetime.date.today().isoformat(), "numpy": np.__version__,
            "mpmath": mpmath.__version__}


def main():
    cells = no_touch_cells()
    (HERE / "no_touch_reference.json").write_text(json.dumps(
        {"provenance": provenance(["w1b_stable_candidate.json", "w1c_random_sweep.json", "w1d_engines_on_stable.py"]),
         "cells": cells}, indent=0))
    (HERE / "reference_values.json").write_text(json.dumps(
        {"provenance": provenance(["l2d_final_book.json", "d6_localvol.json"]),
         "lv_book": lv_book(), "dcn_skew_finest": dcn_skew_finest()}, indent=1))
    print({k: len(v) for k, v in cells.items()})


if __name__ == "__main__":
    main()
