"""Is the new `readout` field the only reason the banked hashes moved?

The certificate records every numerically relevant QuadParams knob, defaults
included, so ANY new field moves the identity hash. If excluding `readout`
restores every banked cell, the move is purely the new field and no priced
number is implicated.
"""
from __future__ import annotations
import json
import pathlib

import quantark.modelvalidation.builders.equity_snowball as B
import quantark.modelvalidation.builders.equity_ko_reset as B_KO
import quantark.modelvalidation.builders.equity_phoenix as B_PH
from quantark.modelvalidation.candidate import candidate_identity
from quantark.modelvalidation.evidence import identity_hash
from quantark.modelvalidation.yaml_loader import load_study

# The ko_reset and phoenix builders do `from ...equity_snowball import
# _QUAD_NON_NUMERIC`, which binds the NAME into their own namespace at import
# time. Rebinding it in equity_snowball alone leaves them untouched -- which
# is exactly the mistake this probe made on its first outing, and it read as
# pre-existing staleness in two candidates that were in fact fine.
_BUILDERS = (B, B_KO, B_PH)


def set_exclusions(names):
    for module in _BUILDERS:
        module._QUAD_NON_NUMERIC = names

STUDY_YAML = {
    "snowball-flat-bsm": "example/modelvalidation/snowball_flat_bsm.yaml",
    "phoenix-flat-bsm": "example/modelvalidation/phoenix_flat_bsm.yaml",
    "ko-reset-flat-bsm": "example/modelvalidation/ko_reset_flat_bsm.yaml",
}
ROOT = pathlib.Path("/Users/fuxinyao/quant-ark/.claude/worktrees/bucket-futures-hedge")


def main():
    certs = sorted(
        (ROOT / "docs" / "modelvalidation" / "certificates").glob("*/*/certificate.json")
    )
    base = B._QUAD_NON_NUMERIC
    for excl in (base, base + ("readout",)):
        set_exclusions(excl)
        ok = tot = 0
        for cert in certs:
            payload = json.loads(cert.read_text())
            if payload["study"]["name"] not in STUDY_YAML:
                continue
            study = load_study(ROOT / STUDY_YAML[payload["study"]["name"]])
            cases = {c.name: c for c in study.cases}
            cands = {c.name(): c for c in study.candidates}
            for cell in payload["cells"]:
                cur = identity_hash(
                    candidate_identity(cands[cell["candidate"]], cases[cell["case"]])
                )
                tot += 1
                ok += cur == cell["identity_hash"]
        print(
            f"exclude readout={'readout' in excl}: {ok}/{tot} banked cells match"
        )


def by_candidate():
    """Which candidates mismatch with `readout` excluded, i.e. before my change."""
    certs = sorted(
        (ROOT / "docs" / "modelvalidation" / "certificates").glob("*/*/certificate.json")
    )
    set_exclusions(B._QUAD_NON_NUMERIC + ("readout",))
    tally: dict[str, list[int]] = {}
    for cert in certs:
        payload = json.loads(cert.read_text())
        if payload["study"]["name"] not in STUDY_YAML:
            continue
        study = load_study(ROOT / STUDY_YAML[payload["study"]["name"]])
        cases = {c.name: c for c in study.cases}
        cands = {c.name(): c for c in study.candidates}
        for cell in payload["cells"]:
            cur = identity_hash(
                candidate_identity(cands[cell["candidate"]], cases[cell["case"]])
            )
            row = tally.setdefault(cell["candidate"], [0, 0])
            row[1] += 1
            row[0] += cur == cell["identity_hash"]
    print("\n# with `readout` excluded (the state before this change):")
    for name, (ok, tot) in sorted(tally.items()):
        flag = "" if ok == tot else "   <-- already stale"
        print(f"  {name:38s} {ok:4d}/{tot:4d}{flag}")


if __name__ == "__main__":
    main()
    by_candidate()
