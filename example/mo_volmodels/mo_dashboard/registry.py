"""Contract A: which run dirs are the fleet, and which commits void what.

The registry states only what code cannot derive.  Fleet *dimensions* (six
variants, 27 inceptions) come from stage 12 and the G4 artifact -- see
``fleet.py`` -- and are never restated here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

Scoped = Union[Tuple[str, ...], str]  # a tuple of names, or the literal "*"


@dataclass(frozen=True)
class Invalidation:
    """A declared statement that some prior output is not comparable.

    Scoping is the whole point.  An unscoped invalidation applied to the
    2026-08-03 artifacts voids G1, G4 and all 27 ``flat_bsm`` cells on the
    strength of ``f97fba3``, a 2D-PDE Heston delta fix that touches none of
    them (spec 5.2).
    """

    commit: str
    landed: datetime
    spec: str
    scopes: Tuple[str, ...]
    variants: Scoped
    facets: Scoped
    reason: str

    def applies(self, scope: str, variant: Optional[str], facet: str) -> bool:
        if scope not in self.scopes:
            return False
        if self.variants != "*":
            # ``variant is None`` means the row is not variant-specific (a
            # whole-gate verdict), so a variant-scoped invalidation cannot
            # reach it.
            if variant is None or variant not in self.variants:
                return False
        if self.facets != "*" and facet not in self.facets:
            return False
        return True


@dataclass(frozen=True)
class Registry:
    fleet_dirs: Tuple[Path, ...] = ()
    probe_dirs: Tuple[Path, ...] = ()
    invalidations: Tuple[Invalidation, ...] = ()
    errors: Tuple[Dict[str, str], ...] = ()


def _scoped(value: Any, field: str, commit: str) -> Scoped:
    if value == "*":
        return "*"
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    raise ValueError(
        f'{commit}: applies_to.{field} must be a list or "*", got {value!r}'
    )


def norm(path: Path) -> Path:
    """Absolutise without following symlinks.

    ``Path.resolve()`` would follow them, and ``output/`` is a symlink in a
    worktree checkout -- registry entries would resolve into the main
    repository while ``classify_run_dirs`` iterating the same symlink stays
    in worktree space, so the two sides could never match and every
    displayed path would be absolute.  ``normpath`` collapses ``.``/``..``
    and duplicate separators without touching links, keeping both sides in
    one namespace.
    """
    return Path(os.path.normpath(str(path)))


def _dirs(entries: Any, project_root: Path) -> Tuple[Path, ...]:
    out: List[Path] = []
    for entry in entries or []:
        out.append(norm(project_root / str(entry["dir"])))
    return tuple(out)


def load_registry(path: Path, project_root: Path) -> Registry:
    """Parse the registry.  A missing or malformed file is an error row."""
    path = Path(path)
    project_root = Path(project_root).resolve()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 -- surfaced on the page, never raised
        return Registry(
            errors=(
                {
                    "source": "registry",
                    "path": str(path),
                    "message": f"{type(exc).__name__}: {exc}",
                },
            )
        )

    errors: List[Dict[str, str]] = []
    invalidations: List[Invalidation] = []
    for item in raw.get("invalidations") or []:
        commit = str(item.get("commit", "?"))
        try:
            scope_block = item.get("applies_to") or {}
            invalidations.append(
                Invalidation(
                    commit=commit,
                    landed=datetime.fromisoformat(str(item["landed"])),
                    spec=str(item.get("spec", "")),
                    scopes=tuple(str(s) for s in scope_block["scopes"]),
                    variants=_scoped(
                        scope_block.get("variants", "*"), "variants", commit
                    ),
                    facets=_scoped(scope_block.get("facets", "*"), "facets", commit),
                    reason=str(item.get("reason", "")),
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "source": "registry.invalidations",
                    "path": str(path),
                    "message": f"{commit}: {type(exc).__name__}: {exc}",
                }
            )

    try:
        fleet_dirs = _dirs(raw.get("fleet"), project_root)
        probe_dirs = _dirs(raw.get("probes"), project_root)
    except Exception as exc:  # noqa: BLE001
        fleet_dirs, probe_dirs = (), ()
        errors.append(
            {
                "source": "registry.dirs",
                "path": str(path),
                "message": f"{type(exc).__name__}: {exc}",
            }
        )

    return Registry(
        fleet_dirs=fleet_dirs,
        probe_dirs=probe_dirs,
        invalidations=tuple(invalidations),
        errors=tuple(errors),
    )


def classify_run_dirs(registry: Registry, output_root: Path) -> Dict[Path, str]:
    """Role for every run dir on disk.

    A run dir is any immediate child of ``output_root`` holding a
    ``run_manifest.json``.  One absent from the registry is ``unclassified``
    -- a visible gap rather than a silent omission.  This is not
    hypothetical: ``output/volmodel_smoke_gated`` was created 2026-08-03 and
    was missed during the design's own survey of ``output/``.
    """
    output_root = norm(Path(output_root))
    roles: Dict[Path, str] = {}
    for path in registry.fleet_dirs:
        roles[path] = "fleet"
    for path in registry.probe_dirs:
        roles.setdefault(path, "probe")
    if not output_root.is_dir():
        return roles
    for child in sorted(output_root.iterdir()):
        if not child.is_dir() or not (child / "run_manifest.json").exists():
            continue
        roles.setdefault(norm(child), "unclassified")
    return roles
