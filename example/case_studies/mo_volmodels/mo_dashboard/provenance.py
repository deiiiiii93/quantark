"""Contract B: is this artifact's verdict still good?

Two verdicts, and the distinction carries weight.  *Stale* means a declared
dependency moved -- re-run to be sure.  *Void* means a declared invalidation
applies: the study spec says this output is not comparable.  Collapsing them
would let void output read as merely old, which is how it gets averaged into
a result.

Neither is proof.  Inferred freshness rests on wall-clock ordering, which is
necessary and not sufficient: a copied, restored or touched artifact reads
fresh while containing stale numbers.  Callers must render the mode.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .registry import Invalidation, parse_iso

FRESH = "fresh"
STALE = "stale"
VOID = "void"

INFERRED = "inferred"

# Directory-level, so a new engine file is covered the day it lands.  The
# whole engine/ directory, not engine/pde/: b6b97f0 changed the facade file
# quantark/asset/equity/engine/pde_engine.py, which a narrower glob misses.
ENGINE_PATHS: Tuple[str, ...] = (
    "quantark/asset/equity/engine/",
    "quantark/volmodels/",
    "quantark/backtest/replay/",
)

# A documentation commit invalidated a numeric artifact (study 5.8 vs the
# live gate_decision.json), so the study spec is a first-class dependency.
STUDY_SPEC = (
    "docs/superpowers/specs/"
    "2026-07-30-snowball-volmodel-backtest-040-rebaseline-design.md"
)

DEPS: Dict[str, Tuple[str, ...]] = {
    "G1": (
        "example/mo_volmodels/13_gate_g1_surface_admission.py",
        "example/mo_volmodels/cohort.py",
        "example/mo_volmodels/data/history/surface_manifest.json",
    ),
    "G4": (
        "example/mo_volmodels/12_snowball_volmodel_backtest.py",
        "quantark/asset/equity/product/option/snowball_option.py",
        *ENGINE_PATHS,
        STUDY_SPEC,
    ),
    "G2": (
        "example/mo_volmodels/11_pde_convergence_gate.py",
        *ENGINE_PATHS,
        STUDY_SPEC,
    ),
    "G5": (
        "example/mo_volmodels/11_pde_convergence_gate.py",
        *ENGINE_PATHS,
    ),
    "FLEET": (
        "example/mo_volmodels/12_snowball_volmodel_backtest.py",
        *ENGINE_PATHS,
        STUDY_SPEC,
    ),
}


@dataclass(frozen=True)
class Commit:
    sha: str
    when: datetime
    subject: str


@dataclass(frozen=True)
class Provenance:
    mode: str = INFERRED
    freshness: str = FRESH
    invalidated_by: Optional[str] = None
    invalidation_reason: str = ""
    superseded_by: Tuple[Commit, ...] = ()
    dirty_deps: Tuple[str, ...] = ()
    missing_deps: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "freshness": self.freshness,
            "invalidated_by": self.invalidated_by,
            "invalidation_reason": self.invalidation_reason,
            "superseded_by": [
                {"sha": c.sha, "when": c.when.isoformat(), "subject": c.subject}
                for c in self.superseded_by
            ],
            "dirty_deps": list(self.dirty_deps),
            "missing_deps": list(self.missing_deps),
        }


def freshness(
    *,
    artifact_mtime: datetime,
    scope: str,
    variant: Optional[str],
    facet: str,
    dep_commits: Sequence[Commit],
    dirty_deps: Mapping[str, datetime],
    missing_deps: Sequence[str],
    invalidations: Sequence[Invalidation],
) -> Provenance:
    """Pure.  All git and filesystem facts arrive pre-collected.

    There is no ``stamped_commit`` parameter and no ``exact`` mode.  An
    earlier draft had one that set the badge to "exact" while still deciding
    from mtime and never validating the SHA -- a badge saying *exact* on
    unvalidated evidence is the one label a reader would trust without
    checking.  It arrives with the rev-list comparison or not at all.
    """
    voided = [
        inv
        for inv in invalidations
        if inv.applies(scope, variant, facet) and inv.landed > artifact_mtime
    ]
    superseded = tuple(c for c in dep_commits if c.when > artifact_mtime)
    dirty = tuple(sorted(p for p, when in dirty_deps.items() if when > artifact_mtime))

    if voided:
        first = min(voided, key=lambda inv: inv.landed)
        state, by, why = VOID, first.commit, first.reason
    elif superseded or dirty or missing_deps:
        # A declared dependency that no longer exists cannot be checked, so
        # the verdict is not fresh.  Carrying it as green metadata is how a
        # renamed engine directory silently certifies everything.
        state, by, why = STALE, None, ""
    else:
        state, by, why = FRESH, None, ""

    return Provenance(
        mode=INFERRED,
        freshness=state,
        invalidated_by=by,
        invalidation_reason=why,
        superseded_by=superseded,
        dirty_deps=dirty,
        missing_deps=tuple(missing_deps),
    )


def mtime_of(path: Path) -> Optional[datetime]:
    """Timezone-aware local mtime, or None when the path is absent."""
    path = Path(path)
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone()


def at_or_below(reported: str, dep: str) -> bool:
    """Is ``reported`` the dep itself, or a leaf inside it?"""
    rep, d = reported.rstrip("/"), dep.rstrip("/")
    return rep == d or rep.startswith(d + "/")


def dep_touched_by(reported: str, dep: str) -> bool:
    """Does a path reported by ``git status`` bear on declared dep ``dep``?

    Containment in BOTH directions.  git collapses untracked trees to a
    parent -- it reports ``?? example/mo_volmodels/data/history/`` and never
    the ``surface_manifest.json`` inside it (verified against this repo) --
    so a one-way test misses every change to a declared untracked dependency.
    """
    rep, d = reported.rstrip("/"), dep.rstrip("/")
    return at_or_below(reported, dep) or d.startswith(rep + "/")


def changed_path(reported: str, dep: str) -> str:
    """Which path is the evidence of change -- the leaf, or the dep?

    Every broad pricing dependency is a DIRECTORY, and editing a file does
    not change its parent directory's mtime (only adding, removing or
    renaming an entry does -- verified).  Stat'ing the declared directory
    when git reported a changed child therefore reads the directory's old
    timestamp, which ``freshness`` then filters out as "older than the
    artifact" -- and an uncommitted engine edit goes undetected.

    So: a reported leaf at or below the dep IS the change.  Only when git
    reported a collapsed parent *above* the dep do we fall back to the dep.
    """
    return reported.rstrip("/") if at_or_below(reported, dep) else dep


def _git(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=str(project_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else ""


def collect_git_facts(
    project_root: Path, deps: Sequence[str]
) -> Tuple[List[Commit], Dict[str, datetime], List[str]]:
    """Impure counterpart to ``freshness``.

    Returns commits touching ``deps`` (newest first), a map of
    modified-uncommitted dep paths to their mtimes, and deps whose path does
    not exist -- the last is an error row, never a skipped check: a renamed
    engine directory must not silently turn every verdict green.
    """
    project_root = Path(project_root)
    missing = [d for d in deps if not (project_root / d).exists()]
    present = [d for d in deps if (project_root / d).exists()]

    commits: List[Commit] = []
    if present:
        out = _git(project_root, "log", "--format=%h\x1f%cI\x1f%s", "--", *present)
        for line in out.splitlines():
            sha, _, rest = line.partition("\x1f")
            when, _, subject = rest.partition("\x1f")
            if sha and when:
                commits.append(Commit(sha, parse_iso(when), subject))

    dirty: Dict[str, datetime] = {}
    status = _git(project_root, "status", "--porcelain")
    for line in status.splitlines():
        rel = line[3:].strip()
        if not rel:
            continue
        for dep in present:
            if dep_touched_by(rel, dep):
                target = changed_path(rel, dep)
                when = mtime_of(project_root / target)
                if when is not None:
                    dirty[target] = when

    # Untracked deps are stat'ed DIRECTLY, never discovered through git
    # status.  git collapses untracked trees to a parent, and an entry in
    # .git/info/exclude suppresses the report entirely -- either way a
    # declared dependency like surface_manifest.json becomes invisible.  It
    # has no commit history, so its mtime is the only evidence there is.
    for dep in _untracked(project_root, present):
        when = mtime_of(project_root / dep)
        if when is not None:
            dirty.setdefault(dep, when)

    return commits, dirty, missing


def _untracked(project_root: Path, deps: Sequence[str]) -> List[str]:
    """Declared deps that git does not track (so have no commit history)."""
    out: List[str] = []
    for dep in deps:
        tracked = _git(project_root, "ls-files", "--", dep).strip()
        if not tracked:
            out.append(dep)
    return out
