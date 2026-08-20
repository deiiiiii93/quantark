"""CI guard for the certifications banked under docs/modelvalidation/certificates/.

An anchor file is the cheap residue of an expensive certification: the
deterministic engines' own outputs at the exact configurations the evidence
describes. Re-running only the deterministic side takes seconds, so every
commit can check that the released engines still produce the numbers their
certificate claims.

A failure here means the banked certificate no longer describes the engine.
Re-certify or amend -- never refresh the anchor file to match the new numbers,
which would silently relabel a numerics change as a no-op.
"""

from pathlib import Path

import pytest

from quantark.modelvalidation import assert_anchors

REPO_ROOT = Path(__file__).resolve().parents[2]
CERTIFICATES = REPO_ROOT / "docs" / "modelvalidation" / "certificates"

BANKED = sorted(CERTIFICATES.glob("*/*/anchors.json"))


def test_at_least_one_certification_is_banked():
    """Guards the guard: a glob that silently matches nothing proves nothing."""
    assert BANKED, f"no anchors.json found under {CERTIFICATES}"


@pytest.mark.parametrize("anchor_path", BANKED, ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_banked_certification_still_describes_its_engines(anchor_path):
    """Exact on the banking machine; rel_tol elsewhere (see the release procedure)."""
    assert_anchors(anchor_path)
