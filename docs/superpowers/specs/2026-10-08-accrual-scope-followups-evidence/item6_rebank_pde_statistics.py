"""Item 6 (W2): re-bank the PDE statistics reference cases the rebate rule moved, and prove nothing else moved.

``test/pde_statistics/build_reference.py`` re-derives the whole bank and refuses unless every value matches the
3B evidence (``s1_level1_accuracy.json``) to 1e-12: by construction it cannot bank a deliberate cash change. This
script is the amendment instead:

1. every reference case is recomputed with ``exact_with_error`` (``sb_daily`` too unless ``--skip-daily``);
2. every case but the moved ones must match the bank to 1e-12 on every field;
3. on a moved case every probability (and the KO cash) must match to 1e-12: only
   ``expected_discounted_maturity_cashflow``, ``pv`` and their quadrature errors may move;
4. the move must be the closed form: the rebate falls from ``rebate`` to ``rebate x tenor`` on the paths that
   neither knock out nor knock in, so
   ``new - old = -rebate x (1 - tenor) x e^{-rT} x P(no KO, no KI)``,
   ``P(no KO, no KI) = survival(T) - ki_survive_knocked_in``;
5. the moved fields are written, and the amendment is recorded under ``provenance.amendments``.

Run from the repo root:  env PYTHONPATH=$PWD:$PWD/test <venv>/bin/python <this file> [--write] [--skip-daily]
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "test"))

from pde_statistics.cases import CASES, REFERENCE_CASES, REFERENCE_KWARGS  # noqa: E402
from pde_statistics.reference import exact_with_error  # noqa: E402

BANK_PATH = ROOT / "test" / "pde_statistics" / "reference_values.json"
MOVED = {"sb_cont_3m"}
MAY_MOVE = {"expected_discounted_maturity_cashflow", "pv", "mass_lost", "kie_mass_lost", "quadrature_error"}
MATCH = 1e-12


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x.astype(float).tolist()
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def _diff(a, b) -> float:
    if isinstance(a, dict):
        return max((_diff(a[k], b[k]) for k in a), default=0.0)
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def main(write: bool, skip_daily: bool) -> None:
    bank = json.loads(BANK_PATH.read_text())
    amended = []
    for name in REFERENCE_CASES:
        if skip_daily and name == "sb_daily":
            print(f"{name:22s} skipped")
            continue
        product, env = CASES[name][0](), CASES[name][1]()
        new = _jsonable(exact_with_error(product, env, **REFERENCE_KWARGS.get(name, {})))
        old = bank["cases"][name]["values"]
        moved = sorted(k for k in old if k not in ("mass_lost", "kie_mass_lost", "quadrature_error")
                       and _diff(old[k], new[k]) > MATCH)
        if name not in MOVED:
            assert not moved, f"{name}: unexpected move in {moved}"
            print(f"{name:22s} unchanged (max field diff <= {MATCH})")
            continue
        assert set(moved) <= MAY_MOVE, f"{name}: a field outside {sorted(MAY_MOVE)} moved: {moved}"
        tenor = product.get_contract_tenor(env)
        df = env.rate_curve.get_discount_factor(product.get_maturity(env))
        p_v0 = new["survival_probability"][-1] - new["ki_survive_knocked_in_probability"]
        rebate = product.payoff_config.rebate_rate * product.initial_price * product.contract_multiplier
        closed = -rebate * (1.0 - tenor) * df * p_v0
        for key in ("expected_discounted_maturity_cashflow", "pv"):
            delta = new[key] - old[key]
            print(f"{name:22s} {key:38s} {old[key]!r} -> {new[key]!r}  delta {delta:+.12f}  "
                  f"closed form {closed:+.12f}  (tenor {tenor}, DF {df:.12f}, P(no KO, no KI) {p_v0:.12f})")
            assert math.isclose(delta, closed, rel_tol=1e-9, abs_tol=1e-11), (key, delta, closed)
        bank["cases"][name]["values"] = new
        amended.append({"case": name, "fields": moved, "old": {k: old[k] for k in moved},
                        "new": {k: new[k] for k in moved}})
    if not write:
        print("dry run: nothing written (pass --write)")
        return
    tree = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    bank["provenance"].setdefault("amendments", []).append({
        "date": date.today().isoformat(), "tree": tree,
        "why": "an unset is_annualized_rebate follows is_annualized (accrual follow-ups item 6, W2): the "
               "rebate of a contract whose tenor is not one year is scaled by its tenor",
        "script": "docs/superpowers/specs/2026-10-08-accrual-scope-followups-evidence/"
                  "item6_rebank_pde_statistics.py",
        "cases": amended})
    BANK_PATH.write_text(json.dumps(bank, indent=1, sort_keys=True) + "\n")
    print(f"wrote {BANK_PATH}")


if __name__ == "__main__":
    main(write="--write" in sys.argv, skip_daily="--skip-daily" in sys.argv)
