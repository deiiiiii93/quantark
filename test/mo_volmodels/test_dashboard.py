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


# ---------------------------------------------------------------------------
# Task 3: gate rows
# ---------------------------------------------------------------------------

gates = importlib.import_module("mo_dashboard.gates")


def test_headline_g1_counts_verified_against_admitted():
    doc = {"n_admitted": 766, "n_verified": 766, "failures": [],
           "min_expiries_seen": 3, "asof": "2026-07-31"}
    h = gates.headline_g1(doc)
    assert h["n_admitted"] == 766
    assert h["n_verified"] == 766
    assert h["n_failures"] == 0
    assert h["satisfied"] is True


def test_headline_g1_is_unsatisfied_when_a_surface_failed():
    doc = {"n_admitted": 766, "n_verified": 765, "failures": [{"date": "20240101"}],
           "min_expiries_seen": 3}
    assert gates.headline_g1(doc)["satisfied"] is False


def test_headline_g4_reports_the_coupon_range_and_solved_count():
    doc = [
        {"inception": "2023-05-04", "coupon": 0.1507, "coupon_solution": {"solved": True}},
        {"inception": "2023-06-01", "coupon": 0.1153, "coupon_solution": {"solved": True}},
    ]
    h = gates.headline_g4(doc)
    assert h["n_solved"] == 2
    assert h["n_inceptions"] == 2
    assert h["coupon_min"] == pytest.approx(0.1153)
    assert h["coupon_max"] == pytest.approx(0.1507)
    assert h["satisfied"] is True


def test_headline_g4_is_unsatisfied_when_one_solve_failed():
    doc = [
        {"inception": "2023-05-04", "coupon": 0.15, "coupon_solution": {"solved": True}},
        {"inception": "2023-06-01", "coupon": None, "coupon_solution": {"solved": False}},
    ]
    assert gates.headline_g4(doc)["satisfied"] is False


def test_headline_g2_splits_pv_from_delta_per_variant():
    doc = {"variants": {
        "flat_bsm": {"route": "pde", "gate": {
            "medium_pass": True, "fine_pass": True, "biased": False,
            "delta_pass": True, "delta_biased": False,
            "delta_info": {"max_abs_contracts": 0.0142, "bound_contracts": 0.1},
        }},
        "heston": {"route": "mc", "gate": {
            "medium_pass": True, "fine_pass": False, "biased": False,
            "delta_pass": False, "delta_biased": True,
            "delta_info": {"max_abs_contracts": 0.9319, "bound_contracts": 0.1},
        }},
    }}
    h = gates.headline_g2(doc)
    assert h["variants"]["flat_bsm"]["pv"]["pass"] is True
    assert h["variants"]["flat_bsm"]["delta"]["pass"] is True
    assert h["variants"]["heston"]["route"] == "mc"
    assert h["variants"]["heston"]["delta"]["pass"] is False
    assert h["variants"]["heston"]["delta"]["max_abs_contracts"] == pytest.approx(0.9319)


def test_g2_is_satisfied_by_routes_not_by_comparison_passes():
    """The real artifact routes localvol/heston/heston_slv to MC *because*
    delta_pass is false.  Treating delta_pass as the predicate would leave G2
    permanently unsatisfiable no matter what the study does."""
    doc = {"variants": {
        "flat_bsm": {"route": "pde", "gate": {"medium_pass": True, "delta_pass": True,
                                              "delta_info": {}}},
        "heston": {"route": "mc", "gate": {"medium_pass": True, "delta_pass": False,
                                           "delta_biased": True, "delta_info": {}}},
    }}
    assert gates.headline_g2(doc)["satisfied"] is True

    no_route = {"variants": {"heston": {"route": None, "gate": {"delta_info": {}}}}}
    assert gates.headline_g2(no_route)["satisfied"] is False


def test_headline_g5_reports_not_run_when_the_artifact_is_absent():
    h = gates.headline_g5(None)
    assert h["satisfied"] is False
    assert h["state"] == "NOT_RUN"


@pytest.mark.parametrize("doc", [
    {},                                                  # empty
    {"n_operating_points": 3},                           # no under_resolved list
    {"under_resolved": []},                              # no point count
    {"n_operating_points": 0, "under_resolved": []},     # zero points swept
])
def test_headline_g5_fails_closed_on_a_partial_document(doc):
    """A truncated write must not clear a mandatory pre-flight.  An earlier
    draft returned satisfied=True for {} because `.get(...) or []` made an
    absent field indistinguishable from an empty one."""
    assert gates.headline_g5(doc)["satisfied"] is False


def test_headline_g5_is_satisfied_only_on_a_complete_clean_sweep():
    h = gates.headline_g5({"n_operating_points": 12, "under_resolved": []})
    assert h["satisfied"] is True
    assert gates.headline_g5(
        {"n_operating_points": 12, "under_resolved": [{"cell": "x"}]}
    )["satisfied"] is False


def test_collect_gates_marks_an_unreadable_artifact(tmp_path):
    (tmp_path / "output").mkdir()
    (tmp_path / "output/gate_g1_admission.json").write_text("{not json", encoding="utf-8")
    rows, errors = gates.collect_gates(tmp_path, registry.Registry())

    g1 = next(r for r in rows if r["id"] == "G1")
    assert g1["status"] == "unreadable"
    assert any("gate_g1_admission" in e["path"] for e in errors)


def test_collect_gates_marks_a_missing_artifact_not_run(tmp_path):
    (tmp_path / "output").mkdir()
    rows, _ = gates.collect_gates(tmp_path, registry.Registry())
    g5 = next(r for r in rows if r["id"] == "G5")
    assert g5["status"] == "missing"
    assert g5["headline"]["state"] == "NOT_RUN"


def test_g2_row_carries_two_facets_and_takes_the_worst():
    assert gates.worst_freshness(["fresh", "void"]) == "void"
    assert gates.worst_freshness(["fresh", "stale"]) == "stale"
    assert gates.worst_freshness(["fresh", "fresh"]) == "fresh"


def test_g2_provenance_is_keyed_by_variant(tmp_path):
    """Without this, f97fba3's heston/heston_slv scope is unreachable and the
    whole scoping mechanism is dead code for the gate it was written for."""
    import os
    out = tmp_path / "output/pde_convergence_gate"
    out.mkdir(parents=True)
    artifact = out / "gate_decision.json"
    artifact.write_text(json.dumps({"variants": {
        "flat_bsm": {"route": "pde", "gate": {"delta_info": {}}},
        "heston": {"route": "mc", "gate": {"delta_info": {}}},
    }}), encoding="utf-8")
    old = datetime(2026, 8, 3, 1, 0, tzinfo=CST).timestamp()
    os.utime(artifact, (old, old))

    reg = registry.Registry(invalidations=(INV_PDEFIX,))
    rows, _ = gates.collect_gates(tmp_path, reg)
    g2 = next(r for r in rows if r["id"] == "G2")

    assert g2["by_variant"]["heston"]["pv"]["freshness"] == "void"
    assert g2["by_variant"]["flat_bsm"]["pv"]["freshness"] != "void"
    assert g2["facets"]["pv"]["freshness"] == "void"   # worst across variants


# ---------------------------------------------------------------------------
# Task 4: fleet dimensions, tree walk, cell state machine
# ---------------------------------------------------------------------------

fleet = importlib.import_module("mo_dashboard.fleet")

FRESH_PROV = provenance.Provenance(freshness="fresh")
STALE_PROV = provenance.Provenance(freshness="stale")
VOID_PROV = provenance.Provenance(freshness="void", invalidated_by="41f2117")


def _facts(**kw):
    base = dict(inception="2023-05-04", variant="flat_bsm", run_dir=Path("/x"),
                summary_mtime=datetime(2026, 8, 3, 1, 55, tzinfo=CST),
                dir_exists=True, summary_readable=True)
    base.update(kw)
    return fleet.CellFacts(**base)


def test_variants_come_from_stage12_not_stage13():
    """Stage 13's VARIANT_ORDER lists five and omits flat_bsm_quad, the
    engine control added by gate-plan Task 3 (spec section 1.3)."""
    assert fleet.VARIANTS == (
        "flat_bsm", "flat_bsm_quad", "ts_bsm", "localvol", "heston", "heston_slv",
    )


def test_cell_state_precedence_is_total():
    assert fleet.cell_state(facts=_facts(summary_readable=False),
                            in_failures=True, prov=VOID_PROV) == "unreadable"
    assert fleet.cell_state(facts=_facts(), in_failures=True, prov=VOID_PROV) == "failed"
    assert fleet.cell_state(facts=_facts(), in_failures=False, prov=VOID_PROV) == "void"
    assert fleet.cell_state(facts=_facts(), in_failures=False, prov=STALE_PROV) == "stale"
    assert fleet.cell_state(facts=_facts(), in_failures=False, prov=FRESH_PROV) == "fresh"
    assert fleet.cell_state(facts=_facts(dir_exists=False, summary_mtime=None),
                            in_failures=False, prov=FRESH_PROV) == "missing"


def test_a_failed_attempt_without_a_directory_is_failed_not_missing():
    state = fleet.cell_state(facts=_facts(dir_exists=False, summary_mtime=None),
                             in_failures=True, prov=FRESH_PROV)
    assert state == "failed"


def test_running_only_when_a_poll_window_is_supplied():
    facts = _facts(summary_mtime=None, dir_exists=True)
    now = datetime(2026, 8, 3, 16, 0, tzinfo=CST)
    assert fleet.cell_state(facts=facts, in_failures=False, prov=FRESH_PROV) == "missing"
    facts_live = _facts(summary_mtime=None, dir_exists=True,
                        dir_mtime=datetime(2026, 8, 3, 15, 59, 55, tzinfo=CST))
    assert fleet.cell_state(facts=facts_live, in_failures=False, prov=FRESH_PROV,
                            poll_window_seconds=30, now=now) == "running"


def test_walk_cells_finds_what_the_manifest_omits(tmp_path):
    """Spec section 1.2: the manifest records only its last invocation."""
    runs = tmp_path / "runs"
    for inception in ("2023-05-04", "2023-06-01"):
        for variant in ("flat_bsm", "ts_bsm"):
            cell = runs / inception / variant
            cell.mkdir(parents=True)
            (cell / "run_summary.json").write_text("{}", encoding="utf-8")
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "config": {"variants": ["flat_bsm"]},
        "counts": {"runs_completed": 2},
        "runs": [{"inception": "2023-05-04", "variant": "flat_bsm"},
                 {"inception": "2023-06-01", "variant": "flat_bsm"}],
        "failures": [],
    }), encoding="utf-8")

    cells = fleet.walk_cells(tmp_path)
    assert len(cells) == 4
    assert ("2023-05-04", "ts_bsm") in cells


def test_manifest_failures_are_keyed_by_pair(tmp_path):
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "failures": [{"inception": "2025-07-01", "variant": "flat_bsm",
                      "error": "event-stats engine returned no stats"}],
    }), encoding="utf-8")
    assert fleet.manifest_failures(tmp_path) == {("2025-07-01", "flat_bsm")}


def test_coverage_counts_fresh_plus_stale_not_fresh_alone():
    """Stale means 're-run to be certain', not 'absent'.  Counting fresh
    alone reads 0/162 on the live tree."""
    counts = fleet.count_states([
        "fresh", "fresh", "void", "void", "stale", "failed", "missing",
    ])
    assert counts["fresh"] == 2
    assert counts["stale"] == 1
    assert counts["void"] == 2
    assert fleet.admitted(counts) == 3


def _seed_g4(tmp_path, tags):
    d = tmp_path / "output/volmodel_backtest"
    d.mkdir(parents=True, exist_ok=True)
    (d / "inceptions.json").write_text(json.dumps(
        [{"inception": t, "coupon": 0.15, "coupon_solution": {"solved": True}}
         for t in tags]), encoding="utf-8")
    return d


def test_a_manifest_failure_with_no_directory_still_renders_failed(tmp_path):
    """Collector-level, not a hand-built cell_state call."""
    run_dir = _seed_g4(tmp_path, ["2023-05-04"])
    (run_dir / "runs").mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "failures": [{"inception": "2023-05-04", "variant": "flat_bsm",
                      "error": "event-stats engine returned no stats"}],
    }), encoding="utf-8")

    reg = registry.Registry(fleet_dirs=(registry.norm(run_dir.absolute()),))
    block = fleet.collect_fleet(tmp_path, reg)
    assert block["grid"]["flat_bsm"]["2023-05-04"]["state"] == "failed"
    assert block["counts"]["failed"] == 1


def test_dimensions_come_from_the_g4_artifact_not_from_running_stage12(tmp_path):
    """Rendering must not import the pricing stack (spec section 5.3)."""
    _seed_g4(tmp_path, ["2023-05-04", "2023-06-01"])

    before = {m for m in sys.modules if m.startswith("quantark.asset")}
    variants, tags = fleet.fleet_dimensions(tmp_path)
    after = {m for m in sys.modules if m.startswith("quantark.asset")}

    assert tags == ["2023-05-04", "2023-06-01"]
    assert len(variants) == 6
    assert after == before, "collecting dimensions imported pricing code"


def test_no_g4_artifact_means_no_defined_fleet(tmp_path):
    (tmp_path / "output").mkdir()
    block = fleet.collect_fleet(tmp_path, registry.Registry())
    assert block["inceptions"] == []
    assert block["expected_cells"] == 0
    assert any(e["source"] == "fleet.dimensions" for e in block["errors"])


def test_collect_fleet_accepts_a_relative_project_root(tmp_path, monkeypatch):
    """load_registry stores absolute dirs; a relative root made
    relative_to() raise ValueError on the first on-grid cell."""
    run_dir = _seed_g4(tmp_path, ["2023-05-04"])
    (run_dir / "runs/2023-05-04/flat_bsm").mkdir(parents=True)
    (run_dir / "runs/2023-05-04/flat_bsm/run_summary.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    reg = registry.Registry(fleet_dirs=(registry.norm(run_dir.absolute()),))
    block = fleet.collect_fleet(Path("."), reg)
    assert block["grid"]["flat_bsm"]["2023-05-04"]["state"] in {"fresh", "stale"}


@pytest.mark.skipif(
    not (PROJECT_ROOT / "example/mo_volmodels/data/history").is_dir(),
    reason="needs the uncommitted history cache",
)
def test_the_artifact_matches_stage12s_schedule():
    """The definition is still enforced -- here, in the test, where paying to
    import stage 12 is fine, rather than on every page render."""
    import pandas as pd
    s12 = fleet._stage12()
    cohort = fleet._cohort()
    history = PROJECT_ROOT / "example/mo_volmodels/data/history"
    spot_csv = history / "csi1000_spot.csv"
    spot = pd.read_csv(spot_csv)
    scheduled = s12.schedule_inceptions(
        calendar=s12.stage11().TradingCalendar.from_spot_csv(spot_csv),
        data_start=pd.Timestamp(spot["date"].iloc[0]).date(),
        data_end=cohort.COHORT_ASOF,
        first_admitted_surface=cohort.admitted_dates(history)[0],
    )
    assert fleet.inception_tags(PROJECT_ROOT) == [d.isoformat() for d in scheduled]
    assert len(scheduled) == 27


# ---------------------------------------------------------------------------
# Task 5: results blocks
# ---------------------------------------------------------------------------

results = importlib.import_module("mo_dashboard.results")


def test_gate_evidence_marks_the_delta_column_void_with_its_citation():
    g2_row = {
        "id": "G2",
        "headline": {"variants": {
            "heston": {"route": "mc", "pv": {"pass": True},
                       "delta": {"pass": False, "max_abs_contracts": 0.9319,
                                 "bound_contracts": 0.1}},
        }},
        "facets": {
            "pv": {"freshness": "stale", "invalidated_by": None},
            "delta": {"freshness": "void", "invalidated_by": "3fbbf21",
                      "invalidation_reason": "reference noise"},
        },
    }
    block = results.gate_evidence_block(g2_row)
    assert block["delta"]["freshness"] == "void"
    assert block["delta"]["invalidated_by"] == "3fbbf21"
    assert block["pv"]["freshness"] == "stale"
    assert block["variants"]["heston"]["delta"]["max_abs_contracts"] == pytest.approx(0.9319)


def test_backtest_block_reconciles_its_denominator_against_the_tree():
    """Panel 2 is manifest-scoped, Panel 3 is tree-scoped, and they
    legitimately differ (spec section 1.2)."""
    block = results.reconcile(manifest_runs=27, tree_fresh=27, tree_total=35)
    assert block["manifest_runs"] == 27
    assert block["tree_total"] == 35
    assert block["unaccounted"] == 8
    assert block["agrees"] is False

    same = results.reconcile(manifest_runs=27, tree_fresh=27, tree_total=27)
    assert same["agrees"] is True


def test_calibration_block_bands_the_feller_ratio():
    records = [
        {"feller_ratio": 0.3}, {"feller_ratio": 0.49},
        {"feller_ratio": 1.0}, {"feller_ratio": 9.9},
        {"feller_ratio": 10.1}, {"feller_ratio": 500.0},
    ]
    bands = results.feller_bands(records)
    assert bands["violated"]["n"] == 2
    assert bands["usable"]["n"] == 2
    assert bands["sigma_collapsed"]["n"] == 2


def test_sigma_collapse_band_label_is_provisional():
    """Study 5.9 (ec20db9) supersedes 7A.11's attribution: those dates fail
    on discretisation, not calibration, and are fixable."""
    bands = results.feller_bands([{"feller_ratio": 50.0}])
    assert bands["sigma_collapsed"]["label"] == "EXCLUDE (provisional)"
    assert "5.9" in bands["sigma_collapsed"]["citation"]


def test_a_corrupt_manifest_is_an_error_not_zero_runs(tmp_path):
    """Fail soft, loud.  A truncated write must not render as a legitimate
    'runs_completed: 0'."""
    (tmp_path / "output/volmodel_backtest").mkdir(parents=True)
    (tmp_path / "output/volmodel_backtest/run_manifest.json").write_text(
        '{"counts": {"runs_comple', encoding="utf-8")

    errors = []
    block = results.backtest_block(tmp_path, {"run_dirs": [], "admitted": 0}, errors)
    assert block["manifest_state"] == "unreadable"
    assert any(e["source"] == "results.backtest" for e in errors)


def test_read_json_separates_missing_from_corrupt(tmp_path):
    assert results.read_json(tmp_path / "nope.json").state == "missing"
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    read = results.read_json(bad)
    assert read.state == "unreadable"
    assert "JSONDecodeError" in read.message or "Expecting" in read.message


# ---------------------------------------------------------------------------
# Task 6: payload, chain, and the integration fixture
# ---------------------------------------------------------------------------

payload_mod = importlib.import_module("mo_dashboard.payload")


def _row(gid, satisfied=True, freshness="fresh", status="ok"):
    return {"id": gid, "status": status, "freshness": freshness,
            "headline": {"satisfied": satisfied},
            "facets": {"all": {"freshness": freshness, "mode": "inferred",
                               "invalidated_by": "3fbbf21" if freshness == "void" else None}}}


def test_chain_includes_g5_before_fleet():
    """Study section 9 requires a grid-resolution sweep over every operating
    point before fleet work, and fdf3a70 made under-resolution a fail-closed
    ValidationError."""
    assert payload_mod.CHAIN == ("G1", "G4", "G2", "G5", "fleet", "aggregate")
    assert payload_mod.CHAIN.index("G5") < payload_mod.CHAIN.index("fleet")


def test_next_action_is_the_first_unsatisfied_node(tmp_path):
    rows = [_row("G1"), _row("G4"), _row("G2", satisfied=False, freshness="void"),
            _row("G5", satisfied=False, status="missing")]
    action = payload_mod.next_action(rows, {"admitted": 27, "expected_cells": 162}, tmp_path)
    assert action["node"] == "G2"
    assert "void" in action["why"].lower()


def test_next_action_carries_the_confidence_of_its_evidence(tmp_path):
    rows = [_row("G1"), _row("G4"), _row("G2", satisfied=False)]
    action = payload_mod.next_action(rows, {"admitted": 0, "expected_cells": 162}, tmp_path)
    assert action["confidence"] in {"exact", "inferred"}


def test_fleet_readiness_needs_fresh_cells_not_merely_admitted(tmp_path):
    """`admitted` (fresh + stale) is the DISPLAY metric; readiness is
    stricter.  162 stale cells are 162 cells whose dependencies moved -- not
    a finished fleet.  Conflating the two let the chain advance past a fleet
    that entirely needed re-running."""
    rows = [_row(g) for g in ("G1", "G4", "G2", "G5")]

    partial = payload_mod.next_action(
        rows,
        {"admitted": 27, "expected_cells": 162, "counts": {"fresh": 0, "stale": 27}},
        tmp_path,
    )
    assert partial["node"] == "fleet"
    assert "0/162 fresh" in partial["why"]
    assert "27 stale" in partial["why"]

    # Fully admitted but entirely stale: still NOT ready.
    all_stale = payload_mod.next_action(
        rows,
        {"admitted": 162, "expected_cells": 162, "counts": {"fresh": 0, "stale": 162}},
        tmp_path,
    )
    assert all_stale["node"] == "fleet"

    # Fully fresh: ready, chain moves on.
    full = payload_mod.next_action(
        rows,
        {"admitted": 162, "expected_cells": 162, "counts": {"fresh": 162, "stale": 0}},
        tmp_path,
    )
    assert full["node"] == "aggregate"


def test_aggregate_node_rejects_a_conclusion_that_predates_its_cells(tmp_path):
    """An aggregate older than the newest cell it summarises is a stale
    conclusion, not a satisfied node."""
    rows = [_row(g) for g in ("G1", "G4", "G2", "G5")]
    fleet_ready = {"admitted": 162, "expected_cells": 162,
                   "counts": {"fresh": 162, "stale": 0},
                   "newest_cell_mtime": "2026-08-03T18:00:00+08:00"}

    (tmp_path / "output/volmodel_backtest").mkdir(parents=True)
    agg = tmp_path / "output/volmodel_backtest/aggregate.json"
    agg.write_text("{}", encoding="utf-8")
    import os
    old = datetime(2026, 8, 3, 12, 0, tzinfo=CST).timestamp()
    os.utime(agg, (old, old))

    stale_agg = payload_mod.next_action(rows, fleet_ready, tmp_path)
    assert stale_agg["node"] == "aggregate"
    assert "predates" in stale_agg["why"]

    new = datetime(2026, 8, 3, 20, 0, tzinfo=CST).timestamp()
    os.utime(agg, (new, new))
    assert payload_mod.next_action(rows, fleet_ready, tmp_path)["node"] is None


def test_a_void_gate_never_renders_as_pass():
    """The page printed 'PASS (inferred)' for G2 while the badge one cell to
    the right read 'void', because display and chain used two predicates."""
    row = {"id": "G2", "status": "ok", "headline": {"satisfied": True},
           "facets": {"pv": {"freshness": "stale"},
                      "delta": {"freshness": "void", "invalidated_by": "3fbbf21"}}}
    ok, label, why = gates.gate_verdict(row)
    assert ok is False
    assert label == "VOID"
    assert "3fbbf21" in why


def test_a_stale_pass_is_labelled_as_such():
    row = {"id": "G1", "status": "ok", "headline": {"satisfied": True},
           "facets": {"all": {"freshness": "stale"}}}
    ok, label, _ = gates.gate_verdict(row)
    assert ok is True
    assert label == "PASS (stale, inferred)"


def test_g2_is_incomplete_when_the_artifact_omits_study_variants():
    """A two-variant artifact previously passed because both had routes --
    and the provenance roll-up then iterated that same short list, so the
    missing variants' invalidations were never evaluated."""
    doc = {"variants": {
        "flat_bsm": {"route": "pde", "gate": {"delta_info": {}}},
        "heston": {"route": "mc", "gate": {"delta_info": {}}},
    }}
    h = gates.headline_g2(doc, fleet.VARIANTS)
    assert h["satisfied"] is False
    assert h["complete"] is False
    assert set(h["missing_variants"]) == set(fleet.VARIANTS) - {"flat_bsm", "heston"}

    whole = {"variants": {v: {"route": "pde", "gate": {"delta_info": {}}}
                          for v in fleet.VARIANTS}}
    assert gates.headline_g2(whole, fleet.VARIANTS)["satisfied"] is True


def test_a_changed_child_under_a_directory_dep_is_detected(tmp_path):
    """Editing a file does not change its parent directory's mtime (verified),
    so stat'ing the declared DIRECTORY read an old timestamp and freshness
    filtered it out -- an uncommitted engine edit went undetected."""
    import os, subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    pkg = tmp_path / "quantark/volmodels"
    pkg.mkdir(parents=True)
    child = pkg / "calibration.py"
    child.write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "seed"], cwd=tmp_path, check=True)

    # Directory predates the artifact; the edited child does not.
    old = datetime(2026, 1, 1, tzinfo=CST).timestamp()
    os.utime(pkg, (old, old))
    child.write_text("x = 2\n", encoding="utf-8")

    dep = "quantark/volmodels/"
    _, dirty, _ = provenance.collect_git_facts(tmp_path, [dep])
    assert dirty, "a changed child under a directory dep must be detected"
    newest = max(dirty.values())
    assert newest > datetime(2026, 6, 1, tzinfo=CST), (
        "the LEAF mtime must be recorded, not the stale directory mtime"
    )


def test_payload_carries_the_schema_version_and_required_keys(tmp_path):
    (tmp_path / "output").mkdir()
    doc = payload_mod.collect(tmp_path, registry_path=tmp_path / "absent.yaml")
    for key in ("schema_version", "generated_at", "mode", "git", "cohort",
                "gates", "chain", "fleet", "results", "errors"):
        assert key in doc, key
    assert doc["schema_version"] == payload_mod.SCHEMA_VERSION
    assert doc["mode"] == "snapshot"
    assert "live" not in doc


def test_serve_mode_adds_a_live_block(tmp_path):
    (tmp_path / "output").mkdir()
    doc = payload_mod.collect(tmp_path, registry_path=tmp_path / "absent.yaml",
                              mode="serve", poll_window_seconds=30)
    assert doc["mode"] == "serve"
    assert "live" in doc


REAL_OUTPUT = PROJECT_ROOT / "output"
REAL_HISTORY = PROJECT_ROOT / "example/mo_volmodels/data/history"

pytestmark_real = pytest.mark.skipif(
    not (REAL_OUTPUT.is_dir() and REAL_HISTORY.is_dir()),
    reason="needs the uncommitted output/ and data/history/ caches",
)


@pytestmark_real
def test_real_artifacts_produce_the_expected_dashboard_state():
    """Pins the whole expected state as of 2026-08-03.

    Every number here is a claim the design makes.  If scoping regresses,
    this fails rather than the page quietly lying.
    """
    doc = payload_mod.collect(PROJECT_ROOT)

    g2 = next(r for r in doc["gates"] if r["id"] == "G2")
    assert g2["facets"]["delta"]["freshness"] == "void"
    assert g2["facets"]["delta"]["invalidated_by"] == "3fbbf21"
    assert g2["facets"]["pv"]["freshness"] != "void"

    g1 = next(r for r in doc["gates"] if r["id"] == "G1")
    assert g1["freshness"] != "void", "f97fba3 must not reach G1"

    assert doc["fleet"]["expected_cells"] == 162
    # 27 STALE, not fresh: every flat_bsm cell is dated no later than
    # 2026-08-03 01:55 while f97fba3 (13:39), 3fbbf21 (15:17) and ec20db9
    # (15:45) all touch declared FLEET dependencies.  Coverage counts
    # fresh + stale precisely so this reads 27 rather than 0.
    assert doc["fleet"]["counts"]["stale"] == 27
    assert doc["fleet"]["counts"]["fresh"] == 0
    assert doc["fleet"]["counts"]["void"] == 8
    assert doc["fleet"]["admitted"] == 27
    assert doc["chain"]["next_action"]["node"] == "G2"


@pytestmark_real
def test_no_run_dir_on_disk_is_unclassified():
    doc = payload_mod.collect(PROJECT_ROOT)
    stragglers = [d for d in doc["fleet"]["run_dirs"] if d["role"] == "unclassified"]
    assert stragglers == [], f"add these to mo_dashboard.yaml: {stragglers}"


@pytestmark_real
def test_rendering_the_real_page_imports_no_pricing_code():
    """Spec criterion 9: the collector must stay out of the pricing stack.

    Run in a CLEAN SUBPROCESS with a real import hook.  A sys.modules delta
    in this interpreter proves nothing: it cannot see an already-cached
    pricing module, it covered only quantark.asset (not volmodels or
    backtest), and this very test module runs a stage-12 test that imports
    the whole stack -- so ordering alone could make it green.
    """
    import subprocess, textwrap
    probe = textwrap.dedent(f"""
        import sys
        FORBIDDEN = ("quantark.asset", "quantark.volmodels",
                     "quantark.backtest", "quantark.montecarlo", "quantark.priceenv")

        class Guard:
            def find_module(self, name, path=None):
                if name.startswith(FORBIDDEN):
                    raise AssertionError("render-time import of " + name)
                return None

            def find_spec(self, name, path=None, target=None):
                if name.startswith(FORBIDDEN):
                    raise AssertionError("render-time import of " + name)
                return None

        sys.meta_path.insert(0, Guard())
        sys.path.insert(0, {str(MO_DIR)!r})
        from mo_dashboard.payload import collect
        from mo_dashboard.render import render
        doc = collect({str(PROJECT_ROOT)!r})
        html = render(doc)
        assert html.startswith("<!doctype html>")
        leaked = [m for m in sys.modules if m.startswith(FORBIDDEN)]
        assert not leaked, leaked
        print("CLEAN")
    """)
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True,
        cwd=str(PROJECT_ROOT), env={"PYTHONPATH": str(PROJECT_ROOT), "PATH": "/usr/bin:/bin"},
    )
    assert "CLEAN" in result.stdout, (
        f"rc={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr[-2000:]}"
    )


# ---------------------------------------------------------------------------
# Tasks 7-8: renderer, server, CLI
# ---------------------------------------------------------------------------

render_mod = importlib.import_module("mo_dashboard.render")
serve_mod = importlib.import_module("mo_dashboard.serve")


def test_render_produces_all_three_panels(tmp_path):
    (tmp_path / "output").mkdir()
    doc = payload_mod.collect(tmp_path, registry_path=tmp_path / "absent.yaml")
    html = render_mod.render(doc)
    assert html.startswith("<!doctype html>")
    for panel_id in render_mod.PANEL_IDS:
        assert f'id="{panel_id}"' in html


def test_render_inlines_the_payload_so_it_works_on_file_urls(tmp_path):
    """A file:// page cannot fetch() a sibling JSON, so a snapshot must carry
    its whole payload inline and issue no network calls."""
    (tmp_path / "output").mkdir()
    doc = payload_mod.collect(tmp_path, registry_path=tmp_path / "absent.yaml")
    html = render_mod.render(doc)
    assert 'id="__DASHBOARD_PAYLOAD__"' in html
    assert "fetch(" not in html
    assert "XMLHttpRequest" not in html
    assert "https://" not in html  # no CDN: the snapshot must render offline


def test_render_never_prints_a_bare_pass():
    assert render_mod.verdict_label(True, "inferred") == "PASS (inferred)"
    assert render_mod.verdict_label(False, "inferred") == "FAIL (inferred)"


def test_render_escapes_artifact_text(tmp_path):
    doc = {"schema_version": 1, "generated_at": "2026-08-03T16:00:00+08:00",
           "mode": "snapshot",
           "git": {"branch": "<script>x</script>", "head": "", "head_subject": "",
                   "dirty_paths": []},
           "cohort": {}, "gates": [], "chain": {"nodes": [], "next_action": {}},
           "fleet": {"grid": {}, "variants": [], "inceptions": [], "counts": {},
                     "run_dirs": [], "expected_cells": 0, "admitted": 0},
           "results": {}, "errors": []}
    html = render_mod.render(doc)
    assert "<script>x</script>" not in html.split("__DASHBOARD_PAYLOAD__")[0]


def test_payload_text_cannot_break_out_of_the_script_element():
    """json.dumps does not escape </script> (verified against this repo), and
    the payload carries log tails and exception text."""
    doc = {"schema_version": 1, "generated_at": "", "mode": "snapshot",
           "git": {"branch": "", "head": "", "head_subject": "", "dirty_paths": []},
           "cohort": {}, "gates": [], "chain": {"nodes": [], "next_action": {}},
           "fleet": {"grid": {}, "variants": [], "inceptions": [], "counts": {},
                     "run_dirs": [], "expected_cells": 0, "admitted": 0},
           "results": {},
           "errors": [{"source": "x", "path": "y",
                       "message": "</script><script>alert(1)</script>"}]}
    html = render_mod.render(doc)
    after_marker = html.split('id="__DASHBOARD_PAYLOAD__"', 1)[1]
    payload_element = after_marker.split("</script>", 1)[0]
    assert "</script><script>" not in payload_element
    assert "\\u003c/script>" in payload_element


def test_api_routes_return_json_slices(tmp_path):
    (tmp_path / "output").mkdir()
    router = serve_mod.Router(tmp_path, tmp_path / "absent.yaml", poll_seconds=10)

    status, ctype, body = router.handle("/api/fleet")
    assert status == 200
    assert ctype == "application/json"
    assert "expected_cells" in json.loads(body)

    status, ctype, body = router.handle("/")
    assert status == 200
    assert ctype == "text/html; charset=utf-8"
    assert body.startswith("<!doctype html>")

    status, _, _ = router.handle("/nope")
    assert status == 404


def test_serve_mode_payload_has_live_block(tmp_path):
    (tmp_path / "output").mkdir()
    router = serve_mod.Router(tmp_path, tmp_path / "absent.yaml", poll_seconds=10)
    _, _, body = router.handle("/api/live")
    assert "log_tails" in json.loads(body)


def _load_cli(name):
    spec = importlib.util.spec_from_file_location(name, MO_DIR / "16_dashboard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_writes_a_snapshot(tmp_path):
    (tmp_path / "output").mkdir()
    module = _load_cli("mo_dash_cli")
    out = tmp_path / "snowball_dashboard_latest.html"
    rc = module.main([
        "--project-root", str(tmp_path),
        "--registry", str(tmp_path / "absent.yaml"),
        "--out", str(out),
    ])
    assert rc == 0
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")


def _tree_state(root: Path):
    """Recursive (path -> mtime_ns, size) over the whole tree."""
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            st = p.stat()
            out[str(p.relative_to(root))] = (st.st_mtime_ns, st.st_size)
    return out


def test_cli_writes_exactly_one_file_and_touches_nothing_else(tmp_path):
    """Read-only contract, checked RECURSIVELY.

    Comparing immediate filenames in output/ still passes if the run
    overwrites an existing artifact, writes inside a run directory, or
    writes elsewhere in the tree.  Snapshot every file's mtime and size.
    """
    project = tmp_path / "proj"
    (project / "output/volmodel_backtest/runs/2023-05-04/flat_bsm").mkdir(parents=True)
    (project / "output/volmodel_backtest/runs/2023-05-04/flat_bsm/run_summary.json").write_text(
        "{}", encoding="utf-8")
    (project / "output/volmodel_backtest/inceptions.json").write_text(
        json.dumps([{"inception": "2023-05-04", "coupon": 0.15,
                     "coupon_solution": {"solved": True}}]), encoding="utf-8")
    (project / "output/gate_g1_admission.json").write_text(
        json.dumps({"n_admitted": 1, "n_verified": 1, "failures": []}), encoding="utf-8")

    target = project / "output/snowball_dashboard_latest.html"
    before = _tree_state(project)
    assert str(target.relative_to(project)) not in before

    module = _load_cli("mo_dash_cli2")
    module.main([
        "--project-root", str(project),
        "--registry", str(project / "absent.yaml"),
        "--out", str(target),
    ])

    after = _tree_state(project)
    created = set(after) - set(before)
    removed = set(before) - set(after)
    modified = {k for k in set(before) & set(after) if before[k] != after[k]}

    assert created == {str(target.relative_to(project))}, created
    assert removed == set(), removed
    assert modified == set(), f"the dashboard modified existing files: {modified}"


# ---------------------------------------------------------------------------
# Gaps found by running the real page: untracked deps, calibration history
# ---------------------------------------------------------------------------

def test_untracked_declared_deps_are_stated_directly(tmp_path):
    """git collapses untracked trees to a parent, and .git/info/exclude
    suppresses the report entirely -- either way a declared dep like
    surface_manifest.json is invisible through git status.  It has no commit
    history, so its mtime is the only evidence there is."""
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    dep = "data/surface_manifest.json"
    target = tmp_path / dep
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    # Suppress it from git status exactly as the real repo does.
    (tmp_path / ".git/info").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".git/info/exclude").write_text("/data\n", encoding="utf-8")
    assert subprocess.run(["git", "status", "--porcelain"], cwd=tmp_path,
                          capture_output=True, text=True).stdout.strip() == ""

    _, dirty, missing = provenance.collect_git_facts(tmp_path, [dep])
    assert dep in dirty, "an untracked dep must be stat'ed, not looked for in git status"
    assert missing == []


def test_calibration_reads_the_manifest_history_not_only_todays_status(tmp_path):
    """status.json holds the latest date only.  Reading it alone computes the
    Feller distribution from a single day."""
    d = tmp_path / "output/mo_daily_calibration"
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps({
        "as_of_date": "2026-08-03",
        "expected_date_records": {"20260731": {"variants": {
            "heston": {"record": {"feller_ratio": 1.0, "cost": 0.001}}}}},
    }), encoding="utf-8")
    (d / "calibration_manifest.json").write_text(json.dumps({
        "records": [
            {"date": "20250731", "variants": {
                "heston": {"record": {"feller_ratio": 0.2, "cost": 0.002}}}},
            {"date": "20250801", "variants": {
                "heston": {"record": {"feller_ratio": 50.0, "cost": 0.003}}}},
        ],
    }), encoding="utf-8")

    errors = []
    block = results.calibration_block(tmp_path, errors)
    assert block["n_records"] == 3, "status-only would report 1"
    assert block["manifest_state"] == "ok"
    assert block["feller"]["violated"]["n"] == 1
    assert block["feller"]["usable"]["n"] == 1
    assert block["feller"]["sigma_collapsed"]["n"] == 1
