"""A06: what the 1D Phoenix copy-row fix (a05's runtime patch of PhoenixAdapter.boundaries) would move in the banked
phoenix-flat-bsm certificate (the live anchor files), before / after back to back in one process.

The certificate's cells are continuous KI by default (near_expiry is a 3-month contract). Diagnosis only: nothing is
refreshed, the patch is not applied to the engine.
"""
import json
from pathlib import Path

from a05_phoenix_1d_copy_row import _ORIG, _patched
from quantark.asset.equity.engine.pde.core.adapters.phoenix import PhoenixAdapter
from quantark.modelvalidation import resolve_supersession
from quantark.modelvalidation.anchors import _candidates_by_name, machine_fingerprint
from quantark.modelvalidation.yaml_loader import load_study_text

ROOT = Path(__file__).resolve().parents[4]

if __name__ == "__main__":
    for path in sorted((ROOT / "docs/modelvalidation/certificates/phoenix-flat-bsm").glob("*/anchors.json")):
        if resolve_supersession(path) is not None:
            continue
        anchors = json.loads(path.read_text())
        print(f"{path.relative_to(ROOT)}  banking machine: {anchors['fingerprint'] == machine_fingerprint()}")
        study = load_study_text(anchors["study_source_text"])
        cands = _candidates_by_name(study)
        cases = {c.name: c for c in study.cases}
        for entry in anchors["anchors"]:
            cand, case = cands[entry["candidate"]], cases[entry["case"]]
            ev = getattr(cand, "evaluate_target", cand.evaluate)
            PhoenixAdapter.boundaries = _ORIG
            before = ev(case).values
            PhoenixAdapter.boundaries = _patched
            after = ev(case).values
            PhoenixAdapter.boundaries = _ORIG
            for q, banked in sorted(entry["values"].items()):
                b, a = before[q], after[q]
                flag = "bit-identical" if a == b else f"MOVED {a - b:+.3e}"
                print(f"  {entry['candidate'].split('.')[-1]:<12} {entry['case']:<22} {q:<6} banked==as-is "
                      f"{b == banked}  {flag}", flush=True)
