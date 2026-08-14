"""Rehearse the publish path over the real gate-driven evidence. NOT OF RECORD.

The 35.5h fleet failed to publish because ``validate_payload`` asserted
``batches_used == expected_batches``, an equality gate-driven stopping breaks by
design. That assertion had never been run against a gate-driven cell: the
``--sequential`` CLI smoke published fine because every quick cell EXHAUSTED its
allocation, so ``banked == declared`` and the short-bank path was never taken.

Fixing one site is not enough -- ``expected_batches`` sizes three different
evidence arrays inside the per-cell loop, and a single-cell replay only reaches
the ones its variant is guarded for. This probe runs the REAL ``main()`` with
``--resume`` over a re-stamped copy of the banked checkpoints, so the real
payload assembly and the real validator see all fourteen real gate-driven cells
in seconds instead of 35.5 hours.

WHAT THIS IS NOT
    The copy's ``run_configuration_sha256`` is rewritten to the live digest so
    ``--resume`` will load it. That defeats the attribution guard, which is
    exactly why the artifacts this writes are NOT a certification of record and
    are never committed: the payload would assert that the live source computed
    numbers produced by an earlier source. This probe tests the VALIDATOR. The
    attribution question is separate and is the user's to decide.

Usage:
    PYTHONPATH=$PWD python docs/adi2d-greek-perf/probes/probe_validator_rehearsal.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
STAGE16 = ROOT / "example" / "mo_volmodels" / "16_adi_greek_certification.py"
BANKED = ROOT / "output" / "p14_sequential"
# Named so nobody can mistake its contents for the certification of record.
SCRATCH = ROOT / "output" / "validator_rehearsal_NOT_OF_RECORD"

FLEET_ARGV = [
    "--full-recertification",
    "--sequential",
    "--sequential-chunk-batches",
    "128",
    "--sequential-margin",
    "0.0",
]


class _Captured(Exception):
    def __init__(self, digest: str) -> None:
        super().__init__(digest)
        self.digest = digest


def _load(intercept_configuration: bool):
    spec = importlib.util.spec_from_file_location("stage16_rehearsal", STAGE16)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage16_rehearsal"] = module
    spec.loader.exec_module(module)
    if intercept_configuration:
        genuine = module._canonical_sha256

        def intercept(payload):
            digest = genuine(payload)
            if isinstance(payload, dict) and "certification_mode" in payload:
                raise _Captured(digest)
            return digest

        module._canonical_sha256 = intercept
    return module


def live_configuration_digest(argv: Sequence[str]) -> str:
    module = _load(intercept_configuration=True)
    try:
        module.main(list(argv) + ["--output-dir", str(SCRATCH)])
    except _Captured as captured:
        return captured.digest
    raise RuntimeError("run_configuration was never hashed")


def restamp(digest: str) -> int:
    """Copy the banked checkpoints and rewrite their configuration digest."""
    destination = SCRATCH / "checkpoints"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    count = 0
    for source in sorted((BANKED / "checkpoints").glob("*.json")):
        record = json.loads(source.read_text())
        record["run_configuration_sha256"] = digest
        record["rehearsal_note"] = (
            "configuration digest rewritten by probe_validator_rehearsal; "
            "this checkpoint is NOT attributable to the live source"
        )
        (destination / source.name).write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        count += 1
    return count


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="leave the re-stamped scratch tree in place for inspection",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        help=(
            "restrict the rehearsal to a case subset, to check whether a known "
            "bad cell is hiding later failures in the others"
        ),
    )
    args = parser.parse_args(argv)
    fleet_argv = list(FLEET_ARGV)
    if args.cases:
        fleet_argv += ["--cases", *args.cases]

    print(__doc__.split("Usage:")[0].strip().splitlines()[0])
    print()

    digest = live_configuration_digest(fleet_argv)
    print(f"live run configuration digest  {digest[:16]}")
    copied = restamp(digest)
    print(f"re-stamped checkpoints         {copied}  -> {SCRATCH}/checkpoints")
    print()

    module = _load(intercept_configuration=False)
    try:
        module.main(fleet_argv + ["--resume", "--output-dir", str(SCRATCH)])
    except Exception as exc:  # noqa: BLE001 - the failure IS the result
        print(f"\nVALIDATOR REJECTED THE GATE-DRIVEN EVIDENCE")
        print(f"  {type(exc).__name__}: {exc}")
        print(
            "\nThis is the publish failure the fleet would hit again, reproduced in\n"
            "seconds. Fix the validator and re-run this probe."
        )
        return 1

    payload_path = SCRATCH / "adi_greek_certification.json"
    payload = json.loads(payload_path.read_text())
    cells = payload["cells"]
    print("\nVALIDATOR ACCEPTED THE GATE-DRIVEN EVIDENCE")
    print(f"  cells               {len(cells)}")
    statuses: dict[str, int] = {}
    for cell in cells:
        statuses[cell["status"]] = statuses.get(cell["status"], 0) + 1
    for status, number in sorted(statuses.items()):
        print(f"  {status:<18}{number}")
    short = [
        (
            f"{cell['variant']}/{cell['case']['name']}",
            cell["sequential_stopping"]["batches_banked"],
            cell["sequential_stopping"]["policy"]["max_batches"],
        )
        for cell in cells
        if cell.get("sequential_stopping")
    ]
    print(f"\n  cells carrying a stopping record: {len(short)}")
    for name, banked, cap in sorted(short):
        marker = "stopped short" if banked < cap else "exhausted"
        print(f"    {name:<34}{banked:>5} / {cap:<5} {marker}")
    print(
        "\nREMINDER: these artifacts are NOT a certification of record -- the\n"
        "checkpoint digests were rewritten to make --resume load them."
    )
    if not args.keep:
        shutil.rmtree(SCRATCH / "checkpoints", ignore_errors=True)
        print(f"removed the re-stamped checkpoints (pass --keep to retain them)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
