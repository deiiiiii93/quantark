"""Re-key the banked p17_fixed fleet for the make_decisions reclassification.

Adopting strided pooling as the declared aggregate alignment required two edits
to the stage-16 harness: the alignment change inside ``make_decisions``, and
moving ``make_decisions`` into ``NON_NUMERICAL_SYMBOLS`` (it reads banked
evidence and renders verdicts; it cannot produce a number). The reclassification
changes the numerical projection digest -- not because anything numerical moved,
but because the projection now removes one more symbol's span -- so every banked
cell's ``identity_sha256`` and the anchors checkpoint's configuration hash stop
matching, and ``_load_checkpoint`` correctly refuses them.

This tool re-keys a COPY of the banked fleet to the new digests, but only after
proving, not assuming, that it is entitled to:

1. **Nothing numerical moved.** ``probe_numerical_projection_equivalence``
   projects the banking revision and the live worktree through the CURRENT
   exemption list; the digests must be identical.
2. **The banked identities rebuild bit-for-bit.** Every cell checkpoint's
   recorded ``identity_sha256`` is recomputed from the banked run configuration
   and the banked numerical digest, consumed links included. A single mismatch
   refuses the whole migration: it would mean the reconstruction path is wrong.
3. **The plan is untouched.** Each cell's plan projection under the re-keyed
   configuration must equal its projection under the banked one -- the new
   configuration differs only in the two implementation digests.
4. **The resume run will actually build the re-keyed configuration.** The live
   ``main()`` is replayed with the production argv and aborted at the moment it
   hashes its run configuration; the captured dict must equal the re-keyed one
   exactly, or stamping would strand the copy behind a mismatched gate.

Nothing here weakens the guard: a stamped cell still has to match a freshly
computed identity at resume time. What is being repaired is a digest
re-classification, not a provenance disagreement -- the cells' numbers were
never re-derived.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
      docs/adi2d-greek-perf/probes/restamp_p18_strided.py --dry-run
    PYTHONPATH=$PWD .venv/bin/python \
      docs/adi2d-greek-perf/probes/restamp_p18_strided.py --apply
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
HARNESS_RELATIVE = "example/mo_volmodels/16_adi_greek_certification.py"
SOURCE_DIR = ROOT / "output" / "p17_fixed"
TARGET_DIR = ROOT / "output" / "p18_strided"

BANKING_REVISION = "2003239"
PRODUCTION_ARGV = ["--full-recertification", "--resume"]


class _Captured(Exception):
    def __init__(self, digest: str, configuration: dict) -> None:
        super().__init__(digest)
        self.digest = digest
        self.configuration = configuration


def load_harness():
    spec = importlib.util.spec_from_file_location(
        "s16_restamp", ROOT / HARNESS_RELATIVE
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def capture_configuration(module, scratch: Path) -> tuple[str, dict]:
    """Run main() far enough to hash the run configuration, then abort."""
    genuine = module._canonical_sha256

    def intercept(payload):
        if isinstance(payload, dict) and "certification_mode" in payload:
            raise _Captured(genuine(payload), copy.deepcopy(payload))
        return genuine(payload)

    module._canonical_sha256 = intercept
    try:
        module.main(PRODUCTION_ARGV + ["--output-dir", str(scratch)])
    except _Captured as captured:
        return captured.digest, captured.configuration
    finally:
        module._canonical_sha256 = genuine
    raise RuntimeError("run configuration was never hashed")


def consumed_links(module, variant: str, case_name: str, rc: dict, numerical: str):
    if variant == "heston_slv" and case_name in module.SLV_MULTILEVEL_CASES:
        return {
            f"heston/{case_name}": module.cell_identity(
                "heston", case_name, rc, numerical_sha256=numerical
            )
        }
    return {}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    # ---- precondition 1: nothing numerical moved ----------------------------
    equivalence = subprocess.run(
        [
            sys.executable,
            str(ROOT / "docs/adi2d-greek-perf/probes/probe_numerical_projection_equivalence.py"),
            "--base",
            BANKING_REVISION,
            "--head",
            "worktree",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    projection_ok = equivalence.returncode == 0
    print(f"[1] numerical projection unchanged since {BANKING_REVISION}: {projection_ok}")
    if not projection_ok:
        print(equivalence.stdout[-2000:])
        print("\nREFUSED: something numerical moved; the banked cells are not reusable.")
        return 1

    module = load_harness()
    live_numerical = module.numerical_implementation_sha256()
    live_implementation = module.implementation_sha256()

    evidence = json.loads((SOURCE_DIR / "adi_greek_certification.json").read_text())
    old_rc = evidence["run_configuration"]
    old_rc_sha = evidence["run_configuration_sha256"]
    old_numerical = evidence["numerical_implementation_sha256"]
    if module._canonical_sha256(old_rc) != old_rc_sha:
        print("REFUSED: banked run configuration does not match its recorded hash")
        return 1
    if (
        old_rc.get("numerical_implementation_sha256") != old_numerical
        or old_rc.get("implementation_sha256") != evidence["implementation_sha256"]
    ):
        print("REFUSED: banked configuration and payload digests disagree")
        return 1
    if bool(old_rc.get("quick")):
        print("REFUSED: banked fleet is a quick profile; nothing to re-key")
        return 1
    print(
        f"    banked numerical {old_numerical[:16]} -> live {live_numerical[:16]}\n"
        f"    banked implementation {evidence['implementation_sha256'][:16]} -> "
        f"live {live_implementation[:16]}"
    )

    # ---- precondition 2: banked identities rebuild bit-for-bit --------------
    checkpoints = sorted((SOURCE_DIR / "checkpoints").glob("*.json"))
    cell_paths = [path for path in checkpoints if path.stem != "anchors"]
    anchors_path = SOURCE_DIR / "checkpoints" / "anchors.json"
    anchors_record = json.loads(anchors_path.read_text())
    if anchors_record.get("run_configuration_sha256") != old_rc_sha:
        print("REFUSED: anchors checkpoint was not banked under this configuration")
        return 1
    rebuilt_ok = True
    identities = {}
    for path in cell_paths:
        variant, _, case_name = path.stem.partition("__")
        record = json.loads(path.read_text())
        expected = module.cell_identity(
            variant,
            case_name,
            old_rc,
            numerical_sha256=old_numerical,
            consumed=consumed_links(module, variant, case_name, old_rc, old_numerical),
        )
        banked = record.get("identity_sha256")
        published = evidence.get("cell_identities", {}).get(f"{variant}/{case_name}")
        agrees = expected == banked == published
        rebuilt_ok = rebuilt_ok and agrees
        identities[path.stem] = {"banked": banked, "rebuilt": expected}
        if not agrees:
            print(
                f"    MISMATCH {variant}/{case_name}: rebuilt {expected[:16]} "
                f"banked {str(banked)[:16]} published {str(published)[:16]}"
            )
    print(f"[2] banked identities rebuild bit-for-bit: {rebuilt_ok} "
          f"({len(cell_paths)} cells)")
    if not rebuilt_ok:
        print("\nREFUSED: the identity reconstruction path is wrong; stamping "
              "would assert something unproven.")
        return 1

    # ---- precondition 3: the re-keyed configuration only moves the digests --
    new_rc = copy.deepcopy(old_rc)
    new_rc["implementation_sha256"] = live_implementation
    new_rc["numerical_implementation_sha256"] = live_numerical
    new_rc_sha = module._canonical_sha256(new_rc)
    plans_ok = True
    for path in cell_paths:
        variant, _, case_name = path.stem.partition("__")
        if module.cell_plan_projection(
            variant, case_name, old_rc
        ) != module.cell_plan_projection(variant, case_name, new_rc):
            plans_ok = False
            print(f"    PLAN MOVED {variant}/{case_name}")
    print(f"[3] per-cell plans identical under the re-keyed configuration: {plans_ok}")
    if not plans_ok:
        print("\nREFUSED: the re-keyed configuration changes a declared plan.")
        return 1

    # ---- precondition 4: the resume run builds exactly this configuration ---
    scratch = TARGET_DIR.parent / "p18_strided_capture_scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    captured_sha, captured_rc = capture_configuration(module, scratch)
    replay_ok = captured_sha == new_rc_sha and captured_rc == new_rc
    print(f"[4] live main() rebuilds the re-keyed configuration exactly: {replay_ok}")
    if not replay_ok:
        differing = sorted(
            key
            for key in set(captured_rc) | set(new_rc)
            if captured_rc.get(key) != new_rc.get(key)
        )
        print(f"    captured {captured_sha[:16]} expected {new_rc_sha[:16]}")
        print(f"    differing keys: {', '.join(differing) if differing else '(hash only)'}")
        print("\nREFUSED: stamping would strand the copy behind a mismatched gate.")
        return 1

    # ---- stamp a copy -------------------------------------------------------
    new_identities = {}
    for path in cell_paths:
        variant, _, case_name = path.stem.partition("__")
        new_identities[path.stem] = module.cell_identity(
            variant,
            case_name,
            new_rc,
            numerical_sha256=live_numerical,
            consumed=consumed_links(module, variant, case_name, new_rc, live_numerical),
        )

    migration_note = {
        "banking_revision": BANKING_REVISION,
        "source": str(SOURCE_DIR.relative_to(ROOT)),
        "old_numerical_implementation_sha256": old_numerical,
        "new_numerical_implementation_sha256": live_numerical,
        "reason": (
            "make_decisions reclassified as non-numerical for the strided "
            "aggregate alignment; projection equivalence proven, identities "
            "rebuilt bit-for-bit, cell numbers never re-derived"
        ),
    }
    print(f"\n  target {TARGET_DIR.relative_to(ROOT)}   "
          f"configuration {old_rc_sha[:16]} -> {new_rc_sha[:16]}")
    for stem in sorted(new_identities):
        print(f"    {stem:<34} {identities[stem]['banked'][:16]} -> "
              f"{new_identities[stem][:16]}")

    if args.apply:
        target_checkpoints = TARGET_DIR / "checkpoints"
        target_checkpoints.mkdir(parents=True, exist_ok=True)
        anchors_stamped = dict(anchors_record)
        anchors_stamped["run_configuration_sha256"] = new_rc_sha
        anchors_stamped["identity_migration"] = migration_note
        (target_checkpoints / "anchors.json").write_text(
            json.dumps(anchors_stamped, indent=2, sort_keys=True) + "\n"
        )
        for path in cell_paths:
            record = json.loads(path.read_text())
            record["identity_sha256"] = new_identities[path.stem]
            record["run_configuration_sha256"] = new_rc_sha
            record["identity_migration"] = migration_note
            (target_checkpoints / path.name).write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n"
            )
        print(f"\nstamped anchors + {len(cell_paths)} cells into {target_checkpoints}")
    else:
        print("\ndry run: nothing written (pass --apply to stamp)")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    (TARGET_DIR / "checkpoint_migration.json").write_text(
        json.dumps(
            {
                **migration_note,
                "applied": bool(args.apply),
                "old_run_configuration_sha256": old_rc_sha,
                "new_run_configuration_sha256": new_rc_sha,
                "cells": {
                    stem: {
                        "banked_identity_sha256": identities[stem]["banked"],
                        "restamped_identity_sha256": new_identities[stem],
                    }
                    for stem in sorted(new_identities)
                },
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {TARGET_DIR / 'checkpoint_migration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
