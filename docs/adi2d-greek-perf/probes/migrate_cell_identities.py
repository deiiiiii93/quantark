"""Stamp per-cell identities onto the banked 35.5h fleet, or refuse to.

The fleet in ``output/p14_sequential`` was banked before cell identities existed,
so its checkpoints carry only a fleet-wide configuration hash and
``_load_checkpoint`` refuses them: a checkpoint that cannot state what it depended
on cannot be reused. This tool computes what each cell's identity WOULD have been
and stamps it -- but only after proving, not assuming, that it is entitled to.

Three preconditions, each measured:

1. **The numerical projection is unchanged** between the banking revision and
   HEAD, so nothing that can move a number moved. Delegated to
   ``probe_numerical_projection_equivalence``.
2. **The banked run's declared plan reconstructs exactly.** HEAD's ``main()`` is
   replayed with the banking revision's implementation digest, its old exclusion
   tuple, and without the key that did not exist then; the resulting
   configuration must hash to the digest the fleet actually recorded. A match
   proves the ONLY configuration differences are those three substitutions.
3. **Each cell's plan projection is identical under both configurations.** Cells
   that fail this are refused rather than stamped -- which is precisely what must
   happen to the two ``near_ki`` cells, whose stopping rule genuinely changed.

Nothing here weakens the guard: a stamped cell still has to match a freshly
computed identity at resume time. What is being repaired is a checkpoint format
that predates the field, not a provenance disagreement.

Usage:
    PYTHONPATH=$PWD python docs/adi2d-greek-perf/probes/migrate_cell_identities.py --dry-run
    PYTHONPATH=$PWD python docs/adi2d-greek-perf/probes/migrate_cell_identities.py --apply
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
HARNESS_RELATIVE = "example/mo_volmodels/16_adi_greek_certification.py"
BANKED = ROOT / "output" / "p14_sequential" / "checkpoints"
OUTPUT_DIR = ROOT / "output" / "cell_identity_migration"

BANKING_REVISION = "350d323"
BANKED_CONFIGURATION_PREFIX = "3a1999f05f85f1e1"
BANKED_IMPLEMENTATION_PREFIX = "0a93c4940b1edb98"
# The exclusion tuple as the banking revision declared it: consumer only.
OLD_EXCLUSIONS = ("heston_slv/near_ki",)

FLEET_ARGV = [
    "--full-recertification",
    "--sequential",
    "--sequential-chunk-batches",
    "128",
    "--sequential-margin",
    "0.0",
]


class _Captured(Exception):
    def __init__(self, digest: str, configuration: dict) -> None:
        super().__init__(digest)
        self.digest = digest
        self.configuration = configuration


def _load_harness(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / HARNESS_RELATIVE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _capture_configuration(
    module, *, drop_keys: Sequence[str] = (), scratch: Path
) -> tuple[str, dict]:
    """Run main() far enough to hash the run configuration, then abort."""
    genuine = module._canonical_sha256

    def intercept(payload):
        if isinstance(payload, dict) and "certification_mode" in payload:
            trimmed = {k: v for k, v in payload.items() if k not in drop_keys}
            raise _Captured(genuine(trimmed), copy.deepcopy(payload))
        return genuine(payload)

    module._canonical_sha256 = intercept
    try:
        module.main(FLEET_ARGV + ["--output-dir", str(scratch)])
    except _Captured as captured:
        return captured.digest, captured.configuration
    finally:
        module._canonical_sha256 = genuine
    raise RuntimeError("run configuration was never hashed")


def _banking_implementation_digest(module) -> str:
    blob = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{BANKING_REVISION}:{HARNESS_RELATIVE}"],
        capture_output=True,
        check=True,
    ).stdout
    digest = hashlib.sha256()
    for relative in module.IMPLEMENTATION_INPUTS:
        if relative == "quantark/validation/cell_identity.py":
            # Did not exist at the banking revision, and is outside the numerical
            # projection; the banked digest was taken over the older list.
            continue
        contents = blob if relative == HARNESS_RELATIVE else (ROOT / relative).read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(contents)
        digest.update(b"\0")
    return digest.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true")
    group.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=BANKED,
        help=(
            "checkpoint directory to stamp; default is the banked fleet, but "
            "prefer a copy -- the banked cells are the one artifact here that "
            "cannot be regenerated cheaply"
        ),
    )
    args = parser.parse_args(argv)
    checkpoints = Path(args.checkpoints)
    if not checkpoints.is_dir():
        print(f"no such checkpoint directory: {checkpoints}")
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_DIR / "scratch"

    # ---- precondition 1: the numerical projection has not moved --------------
    equivalence = subprocess.run(
        [
            sys.executable,
            str(ROOT / "docs/adi2d-greek-perf/probes/probe_numerical_projection_equivalence.py"),
            "--base",
            BANKING_REVISION,
            # The working tree, NOT committed HEAD: the source that will consume
            # these stamps is the tree, and asking for HEAD is what let the
            # 2026-08-14 migration clear itself against a revision that excluded
            # the very changes being made.
            "--head",
            "worktree",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    )
    projection_ok = equivalence.returncode == 0
    print(f"[1] numerical projection unchanged since {BANKING_REVISION}: {projection_ok}")
    if not projection_ok:
        print(equivalence.stdout[-2000:])
        print("\nREFUSED: something numerical moved; the banked cells are not reusable.")
        return 1

    module = _load_harness("s16_migrate")
    numerical_hash = module.numerical_implementation_sha256()

    # ---- precondition 2: the banked plan reconstructs exactly ---------------
    head_digest, head_configuration = _capture_configuration(module, scratch=scratch)
    banking_implementation = _banking_implementation_digest(module)
    if not banking_implementation.startswith(BANKED_IMPLEMENTATION_PREFIX):
        print(
            f"[2] REFUSED: reconstructed banking digest {banking_implementation[:16]} "
            f"!= recorded {BANKED_IMPLEMENTATION_PREFIX}"
        )
        return 1

    old_module = _load_harness("s16_migrate_old_plan")
    old_module.implementation_sha256 = lambda: banking_implementation
    old_module.stopping_excluded_variant_cases = lambda: OLD_EXCLUSIONS
    old_digest, old_configuration = _capture_configuration(
        old_module, drop_keys=("numerical_implementation_sha256",), scratch=scratch
    )
    plan_ok = old_digest.startswith(BANKED_CONFIGURATION_PREFIX)
    print(
        f"[2] banked plan reconstructs exactly: {plan_ok}  "
        f"(rebuilt {old_digest[:16]}, recorded {BANKED_CONFIGURATION_PREFIX})"
    )
    if not plan_ok:
        print(
            "\nREFUSED: the banked run's declared plan cannot be reproduced, so what\n"
            "each cell was run under is not established."
        )
        return 1

    # ---- precondition 3: per-cell plan projections agree --------------------
    print("\n[3] per-cell plan projection under both configurations:")
    stamped, refused = [], []
    for path in sorted(checkpoints.glob("*.json")):
        if path.stem == "anchors":
            continue
        variant, _, case_name = path.stem.partition("__")
        under_head = module.cell_plan_projection(variant, case_name, head_configuration)
        under_banked = module.cell_plan_projection(
            variant, case_name, old_configuration
        )
        agrees = under_head == under_banked
        label = f"{variant}/{case_name}"
        if not agrees:
            differing = sorted(
                key
                for key in set(under_head) | set(under_banked)
                if under_head.get(key) != under_banked.get(key)
            )
            refused.append((label, path, differing))
            print(f"    REFUSE  {label:<30} differs in: {', '.join(differing)}")
            continue
        # The consumed control's identity must describe WHAT WAS BANKED, so it is
        # computed under the banked configuration, not HEAD's. Getting this wrong
        # stamps heston_slv/near_ki as though it had consumed the 2048-batch
        # control it will require in future, when what it actually consumed was
        # the gate-truncated 1664-batch one -- papering over the exact defect the
        # consumed link exists to expose. Computed this way the stamp is honest and
        # the cell is refused at resume, which is the correct outcome.
        consumed = {}
        if variant == "heston_slv" and case_name in module.SLV_MULTILEVEL_CASES:
            consumed[f"heston/{case_name}"] = module.cell_identity(
                "heston", case_name, old_configuration,
                numerical_sha256=numerical_hash,
            )
        identity = module.cell_identity(
            variant, case_name, old_configuration,
            numerical_sha256=numerical_hash, consumed=consumed,
        )
        required = module.cell_identity(
            variant, case_name, head_configuration,
            numerical_sha256=numerical_hash,
            consumed=(
                {
                    f"heston/{case_name}": module.cell_identity(
                        "heston", case_name, head_configuration,
                        numerical_sha256=numerical_hash,
                    )
                }
                if consumed
                else {}
            ),
        )
        if identity != required:
            refused.append((label, path, ["consumed cell changed"]))
            print(
                f"    REFUSE  {label:<30} banked {identity[:16]} but HEAD "
                f"requires {required[:16]} (a cell it consumes changed)"
            )
            continue
        stamped.append((label, path, identity))
        print(f"    stamp   {label:<30} {identity[:16]}")

    print(f"\n  stampable {len(stamped)}   refused {len(refused)}")

    if args.apply:
        for label, path, identity in stamped:
            record = json.loads(path.read_text())
            record["identity_sha256"] = identity
            record["identity_migration"] = {
                "banking_revision": BANKING_REVISION,
                "numerical_projection_sha256": numerical_hash,
                "note": (
                    "identity computed from the reconstructed banked plan after "
                    "proving the numerical projection unchanged; the cell's numbers "
                    "were never re-derived"
                ),
            }
            path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"\nstamped {len(stamped)} checkpoints in place")
    else:
        print("\ndry run: nothing written (pass --apply to stamp)")

    (OUTPUT_DIR / "migration.json").write_text(
        json.dumps(
            {
                "banking_revision": BANKING_REVISION,
                "numerical_projection_sha256": numerical_hash,
                "reconstructed_banked_configuration_sha256": old_digest,
                "head_configuration_sha256": head_digest,
                "applied": bool(args.apply),
                "stamped": [
                    {"cell": label, "identity_sha256": identity}
                    for label, _, identity in stamped
                ],
                "refused": [
                    {"cell": label, "differing_plan_keys": keys}
                    for label, _, keys in refused
                ],
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"wrote {OUTPUT_DIR / 'migration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
