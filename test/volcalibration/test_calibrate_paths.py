"""Guards for the failure that shipped: a code path no test ever executed.

The relocation into ``quantark.volcalibration`` moved ``HESTON_PARAMETER_NAMES``
into ``config.py`` and did not import it back, so every Heston and Heston-SLV
calibration raised ``NameError`` at runtime.  206 module tests passed anyway,
because each of them used ``localvol`` or a stubbed calibrator, and every test
that does exercise Heston is gated on a real MO artifact that lives only in the
(git-excluded) history.

So the guards here are deliberately data-free:

* a static scan for unresolved names, which is what actually found the bug, and
* a real Heston fit on an artifact built from a committed CFETS snapshot.
"""

import ast
import builtins
import json
import tempfile
from pathlib import Path

import pytest

from quantark.param.vol.surface_history import IvSurfaceArtifact
from quantark.volcalibration import VolModelCalibrationConfig, VolModelCalibrator
from quantark.volcalibration.normalize.fxdelta import FxDeltaNormalizer
from quantark.volcalibration.snapshot import QuoteSnapshot
from quantark.volcalibration.store import serialize_artifact
from quantark.volcalibration.surface import build_artifact

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "quantark" / "volcalibration"
CFETS = ROOT / "example/fx_volmodels/data/cfets_usdcny_snapshot_20260720.json"


def _unresolved_names(path: Path) -> list:
    """Load-context names with no binding anywhere in the module.

    Deliberately crude -- it over-binds rather than under-binds, so it cannot
    cry wolf: every assignment, argument, import and definition anywhere in the
    file counts as bound.  What survives that is a name with no binding at all,
    which is the bug this exists to catch.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bound = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Global):
            bound.update(node.names)
    used = {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    return sorted(used - bound)


@pytest.mark.parametrize(
    "module", sorted(PACKAGE.rglob("*.py")), ids=lambda p: p.name
)
def test_no_unresolved_names(module):
    """`from __future__ import annotations` hides these from the interpreter
    until the line runs, so import success proves nothing."""
    assert _unresolved_names(module) == []


@pytest.fixture(scope="module")
def data_free_artifact():
    """A real artifact with per-expiry ``points``, from a committed snapshot.

    The replay-golden synthetic surface carries no ``points``, so it cannot
    drive a Heston fit; this can, and needs no private history.
    """
    tmp = Path(tempfile.mkdtemp())
    snapshot = QuoteSnapshot.from_legacy_fx(
        json.loads(CFETS.read_text(encoding="utf-8"))
    )
    quotes = FxDeltaNormalizer().normalize(snapshot)
    path = tmp / "artifact.json"
    path.write_bytes(serialize_artifact(build_artifact(quotes, snapshot)))
    return IvSurfaceArtifact.from_file(path), tmp


def test_heston_calibrates(data_free_artifact):
    """The path that was dead on arrival."""
    artifact, tmp = data_free_artifact
    calibrator = VolModelCalibrator(
        VolModelCalibrationConfig(cache_dir=str(tmp / "cache"), heston_max_nfev=40)
    )
    record = calibrator.calibrate("heston", artifact).record
    for name in ("v0", "kappa", "theta", "sigma", "rho"):
        assert isinstance(record[name], float)
    assert record["feller_ratio"] > 0.0
    assert record["variant"] == "heston"


def test_localvol_calibrates(data_free_artifact):
    artifact, tmp = data_free_artifact
    calibrator = VolModelCalibrator(
        VolModelCalibrationConfig(cache_dir=str(tmp / "cache"))
    )
    record = calibrator.calibrate("localvol", artifact).record
    assert record["lv_min"] > 0.0
    assert record["lv_max"] >= record["lv_min"]


# No heston_slv case here, deliberately.  The Heston stage it shares with
# `test_heston_calibrates` is what carries the parameter-vector code this file
# exists to guard -- the SLV run reaches the Fokker-Planck solve, which proves
# that stage ran.  Beyond it, the "mo_frozen" preset's bounds are tuned for a
# ~20%-vol equity index and pin sigma at its floor on a ~3%-vol FX surface;
# the resulting near-degenerate diffusion trips the FP negative-mass guard
# (0.053 against a 0.05 tolerance at the default grid).  That is the guard
# working.  Making it pass would mean tuning a grid until the number moved,
# which tests nothing -- SLV on a real MO surface is covered by
# test/test_otc_vol_calibrators.py where the preset matches the market.
