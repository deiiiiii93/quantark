"""One command line for every certification.

    python -m quantark.modelvalidation run     <study.yaml> [--quick] [--resume]
    python -m quantark.modelvalidation amend   <study.yaml> --parent <cert.json> --reason TEXT
    python -m quantark.modelvalidation anchors <certificate.json> [--out FILE]
    python -m quantark.modelvalidation list

A REJECTED decision exits zero: the run *succeeded* -- it produced evidence, and
the evidence says no. A non-zero exit means the framework could not produce
evidence at all (an unreadable study, an invalid parent), which is a different
problem and deserves a different signal.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from quantark.util.exceptions import QuantArkException
from quantark.modelvalidation import builders  # noqa: F401 - registers builtin builders
from quantark.modelvalidation.amendment import amend, validate_parent
from quantark.modelvalidation.anchors import extract_anchors
from quantark.modelvalidation.evidence import atomic_write_json
from quantark.modelvalidation.pipeline import certify
from quantark.modelvalidation.registry import list_builders
from quantark.modelvalidation.yaml_loader import load_study, load_study_text

DEFAULT_OUT = "output/modelvalidation"
STUDY_DIR = Path("example/modelvalidation")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m quantark.modelvalidation",
        description="Certify deterministic pricing engines against stochastic benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run a certification from a YAML study")
    run.add_argument("study", help="path to a YAML study file")
    run.add_argument("--out", default=DEFAULT_OUT, help=f"output root (default: {DEFAULT_OUT})")
    run.add_argument(
        "--quick",
        action="store_true",
        help="shrink sampling for a wiring check (never bankable evidence)",
    )
    run.add_argument(
        "--resume", action="store_true", help="reuse checkpoints whose identity matches"
    )

    amend_parser = subparsers.add_parser(
        "amend", help="re-certify what changed since a parent certificate"
    )
    amend_parser.add_argument("study", help="path to a YAML study file")
    amend_parser.add_argument(
        "--parent", required=True, help="path to the parent certificate.json"
    )
    amend_parser.add_argument(
        "--reason", required=True, help="why this amendment exists (recorded in evidence)"
    )
    amend_parser.add_argument("--out", default=DEFAULT_OUT, help="output root")
    amend_parser.add_argument("--quick", action="store_true", help="shrink sampling")
    amend_parser.add_argument("--resume", action="store_true", help="reuse checkpoints")

    anchors = subparsers.add_parser(
        "anchors", help="extract CI anchors from a certificate"
    )
    anchors.add_argument("certificate", help="path to a certificate.json")
    anchors.add_argument(
        "--out", default=None, help="anchor file (default: anchors.json beside the certificate)"
    )

    subparsers.add_parser("list", help="list registered builders and known studies")
    return parser


def _print_decisions(payload: dict) -> None:
    print()
    for name, decision in sorted(payload["decisions"].items()):
        print(f"  {decision:<13} {name}")
    unresolved = sum(1 for cell in payload["cells"] if cell["verdict"] == "UNRESOLVED")
    errored = sum(1 for cell in payload["cells"] if cell["verdict"] == "ERROR")
    if unresolved:
        print(f"\n  {unresolved} cell(s) UNRESOLVED: the benchmark never met its budget.")
    if errored:
        print(f"  {errored} cell(s) ERROR: see the report for the exceptions.")


def _print_artifacts(certificate) -> None:
    directory = certificate.path.parent
    print(f"\nCertificate:  {certificate.path}")
    print(f"Report (md):  {directory / 'report.md'}")
    print(f"Report (html):{directory / 'report.html'}")


def _cmd_run(args: argparse.Namespace) -> int:
    study = load_study(args.study)
    certificate = certify(
        study, out_dir=args.out, quick=args.quick, resume=args.resume
    )
    _print_decisions(certificate.payload)
    _print_artifacts(certificate)
    if certificate.payload["study"]["quick"]:
        print("\nQuick mode: this run is a wiring check, not bankable evidence.")
    return 0


def _cmd_amend(args: argparse.Namespace) -> int:
    study = load_study(args.study)
    certificate = amend(
        study,
        parent=args.parent,
        out_dir=args.out,
        reason=args.reason,
        quick=args.quick,
        resume=args.resume,
    )
    amendment = certificate.payload["amendment"]
    print(
        f"\nAmended {len(amendment['replaced_cells'])} cell(s), carried "
        f"{len(amendment['carried_cells'])} forward from "
        f"{amendment['parent_projected_sha256'][:12]}."
    )
    _print_decisions(certificate.payload)
    _print_artifacts(certificate)
    return 0


def _cmd_anchors(args: argparse.Namespace) -> int:
    certificate_path = Path(args.certificate)
    payload = validate_parent(certificate_path)
    source_text = payload["study"].get("source_text")
    if not source_text:
        raise QuantArkException(
            "This certificate carries no study source text, so anchors cannot be "
            "re-run from the anchor file. Certify a study loaded from YAML."
        )

    study = load_study_text(source_text)
    anchors = extract_anchors(payload, study)
    target = Path(args.out) if args.out else certificate_path.parent / "anchors.json"
    atomic_write_json(target, anchors)
    print(f"Wrote {len(anchors['anchors'])} anchor group(s) to {target}")
    print("Guard it in CI with: assert_anchors(<path>)")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    print("Registered builders:")
    for kind, names in sorted(list_builders().items()):
        print(f"  {kind}:")
        for name in names:
            print(f"    - {name}")
        if not names:
            print("    (none)")

    if STUDY_DIR.is_dir():
        studies = sorted(STUDY_DIR.glob("*.yaml"))
        print(f"\nStudies in {STUDY_DIR}:")
        for study in studies:
            print(f"  - {study}")
        if not studies:
            print("  (none)")
    return 0


_COMMANDS = {
    "run": _cmd_run,
    "amend": _cmd_amend,
    "anchors": _cmd_anchors,
    "list": _cmd_list,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns an exit code; raises SystemExit only on usage errors."""
    args = _build_parser().parse_args(argv)
    try:
        return _COMMANDS[args.command](args)
    except QuantArkException as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
