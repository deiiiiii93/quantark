"""B06: what the proposed v = 0 column fix (b04's runtime patch == adi_core_v0_column.patch) would move in the banked
2D certificate (adi2d-snowball-greeks, the live anchor file), before / after back to back in one process.

Prints every anchored quantity, its banked value, the as-is value (must equal the banked one bit for bit on the
banking machine), the patched value and the move. Diagnosis only: the anchors are NOT refreshed.
"""
import json
from pathlib import Path

from b04_v0_column_a1_patch import _ORIG_A1, _a1_with_v0_column
from quantark.modelvalidation import resolve_supersession
from quantark.modelvalidation.anchors import _candidates_by_name, machine_fingerprint
from quantark.modelvalidation.yaml_loader import load_study_text
from quantark.volmodels import adi_core

ROOT = Path(__file__).resolve().parents[4]


def main():
    paths = sorted((ROOT / "docs/modelvalidation/certificates/adi2d-snowball-greeks").glob("*/anchors.json"))
    live = [p for p in paths if resolve_supersession(p) is None]
    for path in live:
        anchors = json.loads(path.read_text())
        print(f"{path.relative_to(ROOT)}  banking machine: {anchors['fingerprint'] == machine_fingerprint()}")
        study = load_study_text(anchors["study_source_text"])
        cands = _candidates_by_name(study)
        cases = {c.name: c for c in study.cases}
        for entry in anchors["anchors"]:
            cand, case = cands[entry["candidate"]], cases[entry["case"]]
            ev = getattr(cand, "evaluate_target", cand.evaluate)
            adi_core.HestonSLVADICore._A1 = _ORIG_A1
            before = ev(case).values
            adi_core.HestonSLVADICore._A1 = _a1_with_v0_column
            after = ev(case).values
            adi_core.HestonSLVADICore._A1 = _ORIG_A1
            for q, banked in sorted(entry["values"].items()):
                b, a = before[q], after[q]
                print(f"  {entry['candidate'].split('.')[-1]:<15} {entry['case']:<17} {q:<6} banked {banked:+.15e} "
                      f"as-is==banked {b == banked}  patched {a:+.15e}  move {a - b:+.3e} ({(a - b) / abs(b):+.2e} rel)",
                      flush=True)


if __name__ == "__main__":
    main()
