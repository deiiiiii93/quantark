"""Bank the deterministic references the 3A tests read (spec 3A §10 step 0).

    python test/pde_3a/fixtures/build_fixtures.py $E

The references contain no PDE code, so their values do not depend on the commit. Step 6 of
Task 1 re-runs a sample of them on this machine to confirm they reproduce.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(evidence):
    ev = Path(evidence)
    e13 = json.loads((ev / "e13_base_family_greeks.json").read_text())
    (HERE / "base_family_reference.json").write_text(
        json.dumps({"bump": e13["bump"], "refs": e13["refs"]}, indent=1) + "\n")
    e17 = json.loads((ev / "e17_put_boundary_term_curves.json").read_text())
    rows = [{k: r[k] for k in ("product", "shape", "spot", "lag", "reference")} for r in e17 if "points" not in r]
    (HERE / "term_curve_reference.json").write_text(json.dumps(rows, indent=1) + "\n")
    e14 = json.loads((ev / "e14_continuous_ki_reference.json").read_text())
    keep = [{k: r[k] for k in ("case", "reference", "exact") if k in r} for r in e14]
    (HERE / "continuous_ki_reference.json").write_text(json.dumps(keep, indent=1) + "\n")
    # the 0.7.0 engine values (nodal mask, full-step, 400 points) on the same cells: Task 11's
    # "explicit NODAL + full-step equals 0.7.0" check, at golden tolerance (F5)
    nodal = [{k: r[k] for k in ("product", "horizon", "spot", "pv", "delta", "gamma", "bump_delta", "bump_gamma")}
             for r in e13["pde"] if r.get("mode") == "nodal_today" and r.get("points") == 400 and "error" not in r]
    (HERE / "base_family_nodal_v070.json").write_text(json.dumps(nodal, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
