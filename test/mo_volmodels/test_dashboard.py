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
