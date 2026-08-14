"""Is the implementation digest the ONLY drift in the banked run's identity?

``--resume`` compares ``run_configuration_sha256``, which is a canonical hash of
a dict that embeds ``implementation_sha256`` alongside the runtime environment,
seeds, case definitions, sampling plan, worker counts and stopping policy. The
validator fix moved the implementation digest, and the configuration digest moved
with it -- but that does not by itself prove the digest is the ONLY thing that
moved. If the runtime environment also drifted (a numpy upgrade, say), then
scoping the digest to numerical inputs would not restore the match, and any plan
built on "just re-stamp it" would fail later for a second reason.

This probe rebuilds the fleet's run_configuration from the current source with
the OLD implementation digest substituted, and checks whether it reproduces the
banked configuration hash. It intercepts the hash at the point the run prints it
and aborts before any pricing, so it costs seconds.

A match means: one substitution restores the identity, and the only barrier
between the current source and the banked evidence is the whole-file digest.

Usage:
    PYTHONPATH=$PWD python docs/adi2d-greek-perf/probes/probe_run_identity_drift.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
STAGE16 = ROOT / "example" / "mo_volmodels" / "16_adi_greek_certification.py"
OUTPUT_DIR = ROOT / "output" / "hash_reattribution"

BANKED_IMPLEMENTATION = "0a93c4940b1edb98"
BANKED_CONFIGURATION = "3a1999f05f85f1e1"

# The flags the fleet was launched with. Reconstructed, and the probe's whole
# point is that the hash either confirms this reconstruction or refutes it.
#
# --output-dir is deliberately NOT the banked directory. It does not enter the
# run configuration, so it cannot affect the hash under test, and the banked
# checkpoints are the one artifact in this whole exercise that cannot be
# regenerated cheaply -- so no probe of mine gets to point main() at them.
FLEET_ARGV = [
    "--full-recertification",
    "--sequential",
    "--sequential-chunk-batches",
    "128",
    "--sequential-margin",
    "0.0",
    "--output-dir",
    "output/hash_reattribution/identity_probe_scratch",
]


class _Captured(Exception):
    """Carries the run_configuration out of main() before any pricing starts."""

    def __init__(self, digest: str, configuration: dict) -> None:
        super().__init__(digest)
        self.digest = digest
        self.configuration = configuration


def load_stage16(old_implementation_full: Optional[str]):
    spec = importlib.util.spec_from_file_location("stage16_identity", STAGE16)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage16_identity"] = module
    spec.loader.exec_module(module)

    if old_implementation_full is not None:
        module.implementation_sha256 = lambda: old_implementation_full

    genuine = module._canonical_sha256

    def intercept(payload):
        digest = genuine(payload)
        if isinstance(payload, dict) and "certification_mode" in payload:
            raise _Captured(digest, payload)
        return digest

    module._canonical_sha256 = intercept
    return module


def configuration_digest(module, argv: Sequence[str]) -> tuple[str, dict]:
    try:
        module.main(list(argv))
    except _Captured as captured:
        return captured.digest, captured.configuration
    raise RuntimeError("run_configuration was never hashed; the intercept missed")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-implementation-sha256",
        help="full 64-hex digest of the source that produced the banked evidence",
    )
    args = parser.parse_args(argv)

    old_full = args.old_implementation_sha256
    if old_full is None:
        # Recompute it from git rather than trusting a pasted prefix.
        import hashlib
        import subprocess

        harness = "example/mo_volmodels/16_adi_greek_certification.py"
        blob = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"350d323:{harness}"],
            capture_output=True,
            check=True,
        ).stdout
        probe = load_stage16(None)
        digest = hashlib.sha256()
        for relative in probe.IMPLEMENTATION_INPUTS:
            contents = blob if relative == harness else (ROOT / relative).read_bytes()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(contents)
            digest.update(b"\0")
        old_full = digest.hexdigest()

    if not old_full.startswith(BANKED_IMPLEMENTATION):
        print(
            f"refusing: reconstructed implementation digest {old_full[:16]} is not "
            f"the banked {BANKED_IMPLEMENTATION}"
        )
        return 2

    current = load_stage16(None)
    live_digest, _ = configuration_digest(current, FLEET_ARGV)

    substituted = load_stage16(old_full)
    old_digest, configuration = configuration_digest(substituted, FLEET_ARGV)

    print(f"banked configuration            {BANKED_CONFIGURATION}")
    print(f"current source                  {live_digest[:16]}")
    print(f"current source, old impl digest {old_digest[:16]}")
    matched = old_digest.startswith(BANKED_CONFIGURATION)
    print()
    if matched:
        print(
            "VERDICT: substituting the implementation digest alone restores the banked\n"
            "run identity. Nothing else in the run configuration drifted -- not the\n"
            "runtime environment, seeds, cases, sampling plan, workers, or stopping\n"
            "policy. The whole-file digest is the only barrier."
        )
    else:
        print(
            "VERDICT: the identity does NOT reconstruct from the digest alone. Either\n"
            "the reconstructed launch flags are wrong or something else in the run\n"
            "configuration has drifted; both must be resolved before any re-stamp\n"
            "or resume plan is credible."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIR / "run_identity_drift.json"
    destination.write_text(
        json.dumps(
            {
                "banked_configuration_sha256_prefix": BANKED_CONFIGURATION,
                "current_configuration_sha256": live_digest,
                "substituted_configuration_sha256": old_digest,
                "reconstructed_implementation_sha256": old_full,
                "fleet_argv": list(FLEET_ARGV),
                "runtime_environment": configuration["runtime_environment"],
                "identity_reconstructs_from_digest_alone": bool(matched),
            },
            indent=1,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\nwrote {destination}")
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
