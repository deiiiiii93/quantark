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

from quantark.modelvalidation import assert_anchors, resolve_supersession
from quantark.util.exceptions import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
CERTIFICATES = REPO_ROOT / "docs" / "modelvalidation" / "certificates"

BANKED = sorted(CERTIFICATES.glob("*/*/anchors.json"))


def test_at_least_one_certification_is_banked():
    """Guards the guard: a glob that silently matches nothing proves nothing."""
    assert BANKED, f"no anchors.json found under {CERTIFICATES}"


def _live(anchor_path) -> bool:
    """True when this anchor file is still checked rather than retired."""
    return resolve_supersession(anchor_path) is None


def test_supersession_cannot_retire_a_study_out_of_existence():
    """Guards the guard again: every skip above must leave something checked.

    ``resolve_supersession`` refuses a successor that is missing or that covers
    less than the file it retires. Neither check sees the whole picture: a
    mutual pair that names each other, or a study whose every directory is
    retired in favour of a directory under some *other* study, would satisfy
    both and still leave nothing re-running those engines.

    Requiring every study to keep at least one live file closes all three at
    once: a mutual pair, a cycle, and a cross-study retirement each leave some
    study with nothing live. Genuinely dropping a study's coverage then means
    removing its directories, which is visible in a diff, rather than marking
    them and leaving a green suite that checks nothing.
    """
    by_study = {}
    for anchor_path in BANKED:
        by_study.setdefault(anchor_path.parent.parent.name, []).append(anchor_path)

    for study, paths in sorted(by_study.items()):
        assert any(_live(p) for p in paths), (
            f"every banked certification of {study!r} is marked superseded, so "
            "nothing re-runs its engines"
        )


@pytest.mark.parametrize("anchor_path", BANKED, ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_banked_certification_still_describes_its_engines(anchor_path):
    """Exact on the banking machine; rel_tol elsewhere (see the release procedure)."""
    successor = resolve_supersession(anchor_path)
    if successor is not None:
        # Retired by a later certification of the same study. Its numbers
        # describe engines that have since changed on purpose, so re-running
        # them proves nothing -- but the successor is in BANKED too and is
        # checked in its own right, and resolve_supersession has already
        # refused this skip if it did not cover everything this file anchored.
        pytest.skip(
            f"superseded by {successor.parent.parent.name}/{successor.parent.name}, "
            "which this suite checks in its own right"
        )
    assert_anchors(anchor_path)


# --- the supersession rule itself, tested ----------------------------------
#
# Retiring a directory stops both banked-evidence guards checking it, so the
# conditions under which that is allowed are load-bearing. These build the
# minimal directory shape by hand rather than leaning on banked evidence,
# which cannot express the failure cases without corrupting it.

def _write_anchors(root, study, date, anchors, **extra):
    """One anchor file at the banked layout <root>/<study>/<date>/anchors.json."""
    import json

    directory = root / study / date
    directory.mkdir(parents=True)
    payload = {"schema": 1, "anchors": anchors, **extra}
    (directory / "anchors.json").write_text(json.dumps(payload))
    return directory / "anchors.json"


CELL = [{"candidate": "eq.pde", "case": "ordinary", "values": {"pv": 1.0, "delta": 0.5}}]


def test_a_live_anchor_file_resolves_to_nothing(tmp_path):
    """No marker means check it; that is the path every banked file takes."""
    path = _write_anchors(tmp_path, "study", "2026-01-01", CELL)
    assert resolve_supersession(path) is None


def test_a_successor_that_was_never_banked_is_refused(tmp_path):
    """The common mistake: docs/ is excluded, so the successor is not committed."""
    path = _write_anchors(
        tmp_path, "study", "2026-01-01", CELL, superseded_by="study/2026-02-01"
    )
    with pytest.raises(ValidationError, match="not banked"):
        resolve_supersession(path)


def test_a_file_cannot_supersede_itself(tmp_path):
    """A self-reference would retire the file in favour of nothing."""
    path = _write_anchors(
        tmp_path, "study", "2026-01-01", CELL, superseded_by="study/2026-01-01"
    )
    with pytest.raises(ValidationError, match="its own successor"):
        resolve_supersession(path)


def test_a_successor_that_drops_a_quantity_is_refused(tmp_path):
    """Scope may grow across a supersession; it may never shrink.

    The successor here anchors the same candidate and case but stops anchoring
    delta. Allowing the skip would silently end all coverage of that quantity
    while the suite stayed green.
    """
    _write_anchors(
        tmp_path, "study", "2026-02-01",
        [{"candidate": "eq.pde", "case": "ordinary", "values": {"pv": 2.0}}],
    )
    path = _write_anchors(
        tmp_path, "study", "2026-01-01", CELL, superseded_by="study/2026-02-01"
    )
    with pytest.raises(ValidationError, match="eq.pde/ordinary/delta"):
        resolve_supersession(path)


def test_a_successor_whose_values_moved_is_accepted(tmp_path):
    """The values are SUPPOSED to move -- that is what retired the parent."""
    successor = _write_anchors(
        tmp_path, "study", "2026-02-01",
        [{"candidate": "eq.pde", "case": "ordinary",
          "values": {"pv": 1.0, "delta": 0.9}}],
    )
    path = _write_anchors(
        tmp_path, "study", "2026-01-01", CELL, superseded_by="study/2026-02-01"
    )
    assert resolve_supersession(path) == successor


def test_a_successor_may_add_coverage(tmp_path):
    """Growth is the expected shape: today's study covers more cases."""
    _write_anchors(
        tmp_path, "study", "2026-02-01",
        CELL + [{"candidate": "eq.quad", "case": "near_ki", "values": {"pv": 3.0}}],
    )
    path = _write_anchors(
        tmp_path, "study", "2026-01-01", CELL, superseded_by="study/2026-02-01"
    )
    assert resolve_supersession(path) is not None
