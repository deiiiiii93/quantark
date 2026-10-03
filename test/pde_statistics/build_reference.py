"""Regenerate ``reference_values.json`` (spec 3B §5.7, decision 12). Run by hand: a provenance act.

    PYTHONPATH=$PWD:$PWD/test python test/pde_statistics/build_reference.py

For each reference case it banks ``exact_with_error`` and the Monte Carlo qualifier: per probability
field ``z = (mc - exact) / se`` with the binomial ``se = sqrt(p (1 - p) / N)``, ``N = 200000``, read from
the evidence's ``s1_level1_accuracy.json`` (the Snowball family runs independent paths whatever
``use_antithetic`` says, so binomial errors are the right ones). It refuses to write when a value
differs from the evidence's ``exact`` rows by more than 1e-12 or a ``|z|`` exceeds 2.9.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pde_statistics.cases import CASES, REFERENCE_CASES, REFERENCE_KWARGS  # noqa: E402
from pde_statistics.reference import exact_with_error  # noqa: E402

EVIDENCE = Path("/Users/fuxinyao/quant-ark/docs/superpowers/specs/"
                "2026-10-03-pde-1d-event-statistics-evidence/s1_level1_accuracy.json")
N_PATHS = 200000
Z_MAX = 2.9
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


def _max_diff(a, b):
    if isinstance(a, dict):
        return max((_max_diff(a[k], b[k]) for k in a if k != "quadrature_error"), default=0.0)
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def main():
    rows = json.loads(EVIDENCE.read_text())
    exact_rows = {r["case"]: r["result"] for r in rows if r["kind"] == "exact"}
    mc_rows = {r["case"]: r["result"] for r in rows if r["kind"] == "mc"}
    cases, worst_z, worst_d = {}, 0.0, 0.0
    for name in REFERENCE_CASES:
        product, env = CASES[name][0](), CASES[name][1]()
        values = _jsonable(exact_with_error(product, env, **REFERENCE_KWARGS.get(name, {})))
        d = _max_diff(exact_rows[name], values)
        worst_d = max(worst_d, d)
        mc = mc_rows[name]
        z = {}
        for field in sorted(mc["se"]):
            p_mc = np.atleast_1d(np.asarray(mc[field], float))
            p_ex = np.atleast_1d(np.asarray(values[field], float))
            se = np.sqrt(p_mc * (1.0 - p_mc) / N_PATHS)
            zz = np.where(se > 0.0, (p_mc - p_ex) / np.where(se > 0.0, se, 1.0), 0.0)
            z[field] = zz.tolist()
            worst_z = max(worst_z, float(np.max(np.abs(zz))))
        cases[name] = {"values": values,
                       "mc": {"num_paths": N_PATHS,
                              "sampling": "independent paths (the Snowball family ignores use_antithetic)",
                              "z": z}}
        print(f"{name:22s} max |exact - evidence| {d:.1e}  max |z| "
              f"{max(max(abs(v) for v in zs) for zs in z.values()):.2f}", flush=True)
    if worst_d > MATCH or worst_z > Z_MAX:
        raise SystemExit(f"refused: max diff {worst_d:.1e} (bound {MATCH}), max |z| {worst_z:.2f} (bound {Z_MAX})")
    tree = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    out = {"provenance": {"source": "spec 3B evidence s1_level1_accuracy.json + exact_stats.py",
                          "tree": tree, "date": date.today().isoformat(), "numpy": np.__version__},
           "cases": cases}
    (HERE / "reference_values.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {HERE / 'reference_values.json'}: max diff {worst_d:.1e}, max |z| {worst_z:.2f}")


if __name__ == "__main__":
    main()
