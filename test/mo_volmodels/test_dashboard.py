"""Unit tests for the snowball study progress dashboard.

Pure functions only, except the integration fixture at the end which reads
the real artifacts and skips when they are absent.
"""
import importlib
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MO_DIR = PROJECT_ROOT / "example/mo_volmodels"
if str(MO_DIR) not in sys.path:
    sys.path.insert(0, str(MO_DIR))

registry = importlib.import_module("mo_dashboard.registry")

CST = timezone(timedelta(hours=8))


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "mo_dashboard.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_registry_reads_dirs_and_invalidations(tmp_path):
    path = _write(tmp_path, """
schema_version: 1
fleet:
  - dir: output/volmodel_backtest
probes:
  - dir: output/timing_on
invalidations:
  - commit: f97fba3
    landed: 2026-08-03T13:39:19+08:00
    spec: "5.6"
    applies_to: {scopes: [G2, FLEET], variants: [heston, heston_slv], facets: "*"}
    reason: "2D PDE Heston delta grid"
""")
    reg = registry.load_registry(path, PROJECT_ROOT)

    assert reg.fleet_dirs == (PROJECT_ROOT / "output/volmodel_backtest",)
    assert reg.probe_dirs == (PROJECT_ROOT / "output/timing_on",)
    assert reg.errors == ()
    (inv,) = reg.invalidations
    assert inv.commit == "f97fba3"
    assert inv.landed == datetime(2026, 8, 3, 13, 39, 19, tzinfo=CST)
    assert inv.scopes == ("G2", "FLEET")
    assert inv.variants == ("heston", "heston_slv")
    assert inv.facets == "*"


def test_invalidation_applies_respects_every_axis(tmp_path):
    inv = registry.Invalidation(
        commit="f97fba3",
        landed=datetime(2026, 8, 3, 13, 39, 19, tzinfo=CST),
        spec="5.6",
        scopes=("G2", "FLEET"),
        variants=("heston", "heston_slv"),
        facets="*",
        reason="",
    )
    assert inv.applies("G2", "heston", "delta")
    assert inv.applies("FLEET", "heston_slv", "all")
    assert not inv.applies("G1", "heston", "all")        # wrong scope
    assert not inv.applies("G2", "flat_bsm", "delta")    # wrong variant
    # A gate row with no variant is not variant-specific, so a
    # variant-scoped invalidation must not reach it.
    assert not inv.applies("G2", None, "all")


def test_invalidation_with_star_variants_reaches_a_variantless_row():
    inv = registry.Invalidation(
        commit="41f2117",
        landed=datetime(2026, 7, 31, 10, 13, 27, tzinfo=CST),
        spec="7A.4",
        scopes=("G2", "G4", "FLEET"),
        variants="*",
        facets="*",
        reason="",
    )
    assert inv.applies("G4", None, "all")


def test_facet_scoped_invalidation_touches_only_that_facet():
    inv = registry.Invalidation(
        commit="3fbbf21",
        landed=datetime(2026, 8, 3, 15, 17, 23, tzinfo=CST),
        spec="5.8",
        scopes=("G2",),
        variants="*",
        facets=("delta",),
        reason="",
    )
    assert inv.applies("G2", "heston", "delta")
    assert not inv.applies("G2", "heston", "pv")


def test_missing_registry_yields_an_error_not_an_exception(tmp_path):
    reg = registry.load_registry(tmp_path / "absent.yaml", PROJECT_ROOT)
    assert reg.fleet_dirs == ()
    assert reg.probe_dirs == ()
    assert reg.invalidations == ()
    assert len(reg.errors) == 1
    assert "absent.yaml" in reg.errors[0]["path"]


def test_unregistered_dir_on_disk_is_unclassified(tmp_path):
    (tmp_path / "output/volmodel_backtest").mkdir(parents=True)
    (tmp_path / "output/volmodel_backtest/run_manifest.json").write_text("{}")
    (tmp_path / "output/mystery_run").mkdir(parents=True)
    (tmp_path / "output/mystery_run/run_manifest.json").write_text("{}")
    path = _write(tmp_path, """
schema_version: 1
fleet:
  - dir: output/volmodel_backtest
""")
    reg = registry.load_registry(path, tmp_path)
    roles = registry.classify_run_dirs(reg, tmp_path / "output")

    assert roles[tmp_path / "output/volmodel_backtest"] == "fleet"
    assert roles[tmp_path / "output/mystery_run"] == "unclassified"


# ---------------------------------------------------------------------------
# Task 2: the freshness rule
# ---------------------------------------------------------------------------

provenance = importlib.import_module("mo_dashboard.provenance")

INV_7A4 = registry.Invalidation(
    commit="41f2117", landed=datetime(2026, 7, 31, 10, 13, 27, tzinfo=CST),
    spec="7A.4", scopes=("G2", "G4", "FLEET"), variants="*", facets="*", reason="",
)
INV_PDEFIX = registry.Invalidation(
    commit="f97fba3", landed=datetime(2026, 8, 3, 13, 39, 19, tzinfo=CST),
    spec="5.6", scopes=("G2", "FLEET"), variants=("heston", "heston_slv"),
    facets="*", reason="",
)
INV_DELTA = registry.Invalidation(
    commit="3fbbf21", landed=datetime(2026, 8, 3, 15, 17, 23, tzinfo=CST),
    spec="5.8", scopes=("G2",), variants="*", facets=("delta",), reason="",
)
ALL_INV = (INV_7A4, INV_PDEFIX, INV_DELTA)


def _fresh(**kw):
    base = dict(
        artifact_mtime=datetime(2026, 8, 3, 14, 39, 38, tzinfo=CST),
        scope="G2", variant="heston", facet="pv",
        dep_commits=(), dirty_deps={}, missing_deps=(), invalidations=ALL_INV,
    )
    base.update(kw)
    return provenance.freshness(**base)


def test_fresh_when_nothing_moved():
    p = _fresh(invalidations=())
    assert p.freshness == "fresh"
    assert p.mode == "inferred"
    assert p.invalidated_by is None


def test_stale_when_a_dependency_commit_landed_after():
    later = provenance.Commit("deadbee", datetime(2026, 8, 3, 16, 0, tzinfo=CST), "fix: x")
    p = _fresh(invalidations=(), dep_commits=(later,))
    assert p.freshness == "stale"
    assert [c.sha for c in p.superseded_by] == ["deadbee"]


def test_dirty_dependency_modified_after_the_artifact_is_stale():
    p = _fresh(
        invalidations=(),
        dirty_deps={"quantark/volmodels/calibration.py": datetime(2026, 8, 3, 15, 0, tzinfo=CST)},
    )
    assert p.freshness == "stale"
    assert p.dirty_deps == ("quantark/volmodels/calibration.py",)


def test_dirty_dependency_modified_before_the_artifact_is_not_counted():
    p = _fresh(
        invalidations=(),
        dirty_deps={"quantark/volmodels/calibration.py": datetime(2026, 8, 3, 10, 0, tzinfo=CST)},
    )
    assert p.freshness == "fresh"
    assert p.dirty_deps == ()


def test_the_live_g2_delta_facet_is_void_by_the_spec_commit():
    """The finding the dashboard exists for.

    gate_decision.json was written 2026-08-03 14:39.  Study section 5.8
    landed at 15:17 and states the delta half of every route decision is
    void.  Nothing on disk says so.
    """
    p = _fresh(facet="delta")
    assert p.freshness == "void"
    assert p.invalidated_by == "3fbbf21"


def test_the_same_artifacts_pv_facet_is_not_void():
    p = _fresh(facet="pv")
    assert p.freshness != "void"


@pytest.mark.parametrize(
    "scope,variant,facet",
    [
        ("G1", None, "all"),           # surface admission, engine-free
        ("G4", None, "all"),           # coupon solve
        ("FLEET", "flat_bsm", "all"),  # the 27 admitted cells
    ],
)
def test_pdefix_does_not_void_unrelated_scopes(scope, variant, facet):
    """Spec section 5.2 regression.

    An UNSCOPED f97fba3 voids G1 (Aug 1 11:35), G4 (Aug 3 01:55) and every
    flat_bsm cell (<= Aug 3 01:55), leaving zero admitted cells and
    contradicting the design's own success criteria.
    """
    p = provenance.freshness(
        artifact_mtime=datetime(2026, 8, 3, 1, 55, 26, tzinfo=CST),
        scope=scope, variant=variant, facet=facet,
        dep_commits=(), dirty_deps={}, missing_deps=(),
        invalidations=(INV_PDEFIX,),
    )
    assert p.freshness == "fresh", f"{scope}/{variant} must not be voided by f97fba3"


def test_pdefix_does_void_a_heston_cell():
    p = provenance.freshness(
        artifact_mtime=datetime(2026, 8, 3, 1, 55, 26, tzinfo=CST),
        scope="FLEET", variant="heston", facet="all",
        dep_commits=(), dirty_deps={}, missing_deps=(), invalidations=(INV_PDEFIX,),
    )
    assert p.freshness == "void"
    assert p.invalidated_by == "f97fba3"


def test_jul27_cells_are_void_by_the_engine_fixes():
    """The eight orphaned ts_bsm / localvol cells (spec section 1.2)."""
    for variant in ("ts_bsm", "localvol"):
        p = provenance.freshness(
            artifact_mtime=datetime(2026, 7, 27, 11, 3, 0, tzinfo=CST),
            scope="FLEET", variant=variant, facet="all",
            dep_commits=(), dirty_deps={}, missing_deps=(), invalidations=ALL_INV,
        )
        assert p.freshness == "void"
        assert p.invalidated_by == "41f2117"


def test_void_beats_stale():
    later = provenance.Commit("deadbee", datetime(2026, 8, 3, 18, 0, tzinfo=CST), "x")
    p = _fresh(facet="delta", dep_commits=(later,))
    assert p.freshness == "void"


def test_there_is_no_exact_mode_to_claim():
    """A badge saying 'exact' on unvalidated evidence is worse than no badge."""
    p = _fresh(invalidations=())
    assert p.mode == "inferred"
    with pytest.raises(TypeError):
        provenance.freshness(
            artifact_mtime=datetime(2026, 8, 3, tzinfo=CST), scope="G2",
            variant=None, facet="all", dep_commits=(), dirty_deps={},
            missing_deps=(), invalidations=(), stamped_commit="f97fba3",
        )


def test_a_missing_dependency_is_never_fresh():
    """Carrying it as green metadata is how a renamed engine directory
    silently certifies every verdict on the page."""
    p = _fresh(invalidations=(), missing_deps=("quantark/asset/equity/engine/",))
    assert p.freshness == "stale"
    assert p.missing_deps == ("quantark/asset/equity/engine/",)


def test_engine_paths_cover_the_facade_files():
    """b6b97f0 touched quantark/asset/equity/engine/pde_engine.py, which a
    narrower engine/pde/ glob would have missed entirely."""
    assert "quantark/asset/equity/engine/" in provenance.ENGINE_PATHS


def test_every_gate_scope_depends_on_the_study_spec_where_it_can_be_invalidated():
    for scope in ("G2", "G4", "FLEET"):
        assert provenance.STUDY_SPEC in provenance.DEPS[scope], scope


def test_a_collapsed_untracked_parent_still_marks_its_declared_child_dirty():
    """git status reports '?? example/.../data/history/', never the
    surface_manifest.json inside it (verified against this repo)."""
    dep = "example/mo_volmodels/data/history/surface_manifest.json"
    reported = "example/mo_volmodels/data/history/"
    assert provenance.dep_touched_by(reported, dep), (
        "the reverse containment test is what catches the collapsed parent"
    )
    assert provenance.dep_touched_by(dep, dep)
    assert not provenance.dep_touched_by("quantark/volmodels/calibration.py", dep)
