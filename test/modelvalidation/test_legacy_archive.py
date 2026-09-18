"""The archived Gate C evidence matches its manifest and is not a certificate."""
import hashlib
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LEGACY = REPO / "docs" / "modelvalidation" / "legacy" / "intraday-gate-c"
SNAPSHOTS = sorted(LEGACY.glob("*/manifest.json"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_at_least_one_snapshot_is_archived():
    assert SNAPSHOTS, f"no manifest.json under {LEGACY}"


@pytest.mark.parametrize("manifest_path", SNAPSHOTS, ids=lambda p: p.parent.name)
def test_every_archived_file_matches_its_manifest(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "intraday-gate-c-archive/1"
    assert manifest["label"] == "historical Gate C decisions; not a modelvalidation certificate"
    assert manifest["admission_rule"].startswith("|route - ref| <= budget + 3 * ref_uncertainty")
    assert manifest["files"], "an archive with no files archives nothing"
    for entry in manifest["files"]:
        path = manifest_path.parent / entry["path"]
        assert path.is_file(), entry["path"]
        assert _sha256(path) == entry["sha256"], entry["path"]
        assert path.stat().st_size == entry["bytes"], entry["path"]
        assert entry["original_path"]
    assert isinstance(manifest["unretained"], list)


@pytest.mark.parametrize("manifest_path", SNAPSHOTS, ids=lambda p: p.parent.name)
def test_the_archive_is_never_discovered_as_a_certificate(manifest_path):
    assert not (manifest_path.parent / "anchors.json").exists()
    assert not (manifest_path.parent / "certificate.json").exists()
    certificates = REPO / "docs" / "modelvalidation" / "certificates"
    assert manifest_path.parent not in certificates.parents and certificates not in manifest_path.parents


@pytest.mark.parametrize("manifest_path", SNAPSHOTS, ids=lambda p: p.parent.name)
def test_the_inventory_maps_every_gate_c_fixture_to_a_study(manifest_path):
    inventory = (manifest_path.parent / "INVENTORY.md").read_text(encoding="utf-8")
    price = json.loads((manifest_path.parent / "gate_c_results.json").read_text(encoding="utf-8"))
    fixtures = sorted({cell["cell"]["product"] for cell in price["cells"]}) + ["snowball_daily_ki", "snowball_long_gap"]
    for fixture in fixtures:
        assert f"`{fixture}`" in inventory, f"{fixture} has no inventory row"
    assert "Plan 1" in inventory and "Plan 2" in inventory and "uncertified until" in inventory
