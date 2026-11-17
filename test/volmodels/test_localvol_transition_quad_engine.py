"""Engine finalisation (T6): exact surface sampling, auto backend, AC5 against the reference, AC14."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from example.case_studies.localvol_transition_quad.contract import snowball_contract
from example.case_studies.localvol_transition_quad.engine import OUTPUTS, LocalVolQuad
from example.case_studies.localvol_transition_quad.grid import GridConfig, build_grid
from example.case_studies.localvol_transition_quad.kernels import CTMCKernel, SurfaceSampler
from example.case_studies.localvol_transition_quad.market import load_fixture
from example.case_studies.localvol_transition_quad.reference import budget

REFERENCE = Path(__file__).resolve().parents[2] / "docs/research/localvol_transition_quad/reference.json"


def test_sampler_is_bitwise_equal_to_the_surface():
    m = load_fixture()
    spots = np.exp(np.linspace(np.log(m.spot) - 2, np.log(m.spot) + 2, 301))
    sampler = SurfaceSampler(m.surface, spots)
    rng = np.random.default_rng(7)
    for t in list(rng.uniform(-0.1, 2.0, 50)) + list(m.surface.time_grid):
        assert np.array_equal(sampler(float(t)), m.surface.local_vol(spots, float(t)))


def test_auto_backend_matches_both_exact_backends():
    m = load_fixture()
    c = snowball_contract(m.spot, 0.25, 3)
    pvs = {b: LocalVolQuad(CTMCKernel(backend=b), GridConfig(cells=300, half_width_std=8.0)).price(c, m)["values"]
           for b in ("auto", "chebyshev", "contour")}
    for key, tol in (("pv", 1e-11), ("delta", 1e-9), ("gamma", 1e-9)):    # stencil amplifies series noise
        assert pvs["auto"][key] == pytest.approx(pvs["chebyshev"][key], abs=tol)
        assert pvs["auto"][key] == pytest.approx(pvs["contour"][key], abs=tol)


def test_auto_backend_switches_to_the_contour_on_stiff_operators():
    """A fine grid makes the daily series long (> AUTO_MAX_TERMS): auto must use both backends."""
    m = load_fixture()
    c = snowball_contract(m.spot, 0.25, 3)
    kernel = CTMCKernel(backend="auto")
    auto = LocalVolQuad(kernel, GridConfig(cells=300, half_width_std=6.0, subdivide=9)).price(c, m)["values"]
    assert kernel.solves > 0 and kernel.matvecs > 0
    cheb = LocalVolQuad(CTMCKernel(backend="chebyshev"), GridConfig(cells=300, half_width_std=6.0, subdivide=9)
                        ).price(c, m)["values"]
    assert auto["pv"] == pytest.approx(cheb["pv"], abs=1e-11)


@pytest.mark.parametrize("cells", [300, 400, 500])
@pytest.mark.parametrize("half_width", [6.0, 8.0])
def test_no_sliver_cells_on_the_headline_grids(cells, half_width):
    """Regression: 400 cells / 6 sd left a 3.3e-4 cell beside the KO pair (median 6.6e-3); its
    stiffness pushed every operator onto the contour and made the risk run 2.6x slower (0.175 -> 0.068 s per valuation)."""
    m = load_fixture()
    for maturity, months in ((1.0, 12), (2.0, 24)):
        c = snowball_contract(m.spot, maturity, months)
        engine = LocalVolQuad(CTMCKernel(), GridConfig(cells=cells, half_width_std=half_width))
        widths = engine.grid(c, m).widths
        assert np.min(widths) >= np.median(widths) / 3
        inner = widths[1:-1]
        assert not np.any((inner < 0.5 * widths[:-2]) & (inner < 0.5 * widths[2:]))


def test_homogeneous_slab_rates_are_cached_and_identical():
    m = load_fixture()
    c = snowball_contract(m.spot, 1.0, 12)
    grid = build_grid(c, 0.3, GridConfig(cells=200))
    kernel = CTMCKernel()
    first = kernel.rates(grid, m, 0.95)
    again = kernel.rates(grid, m, 0.99)          # beyond the last pillar: same rates object
    assert first[0] is again[0]
    fresh = CTMCKernel().rates(grid, m, 0.99)
    assert np.array_equal(first[0], fresh[0]) and np.array_equal(first[1], fresh[1])


@pytest.mark.slow
@pytest.mark.parametrize("case,months,maturity", [("snowball_1y", 12, 1.0), ("snowball_2y", 24, 2.0)])
def test_ac5_full_risk_within_budget_of_the_reference(case, months, maturity):
    ref = json.loads(REFERENCE.read_text())["cases"][case]["assessment"]["reference"]
    m = load_fixture()
    out = LocalVolQuad(CTMCKernel(), GridConfig(cells=400, half_width_std=8.0)).full_risk(
        snowball_contract(m.spot, maturity, months), m)["values"]
    for key in OUTPUTS:
        assert abs(out[key] - ref[key]) <= budget(key, ref[key]), (key, out[key], ref[key])


# This process's own peak RSS in bytes. Linux keeps ru_maxrss across execve (getrusage(2)): a child spawned
# from a 13 GB xdist worker reports 13 GB before doing anything. VmHWM is the high-water mark of the
# address space, which execve replaces, so it starts fresh. macOS's ru_maxrss is per process (bytes).
_PEAK_BYTES = """
def peak_bytes():
    if sys.platform == "darwin":
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with open("/proc/self/status") as status:
        return next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmHWM:"))
"""

_AC14_CHILD = """
import json, resource, sys
import numpy as np""" + _PEAK_BYTES + """
from example.case_studies.localvol_transition_quad.contract import snowball_contract
from example.case_studies.localvol_transition_quad.engine import OUTPUTS, LocalVolQuad
from example.case_studies.localvol_transition_quad.grid import GridConfig
from example.case_studies.localvol_transition_quad.kernels import CTMCKernel
from example.case_studies.localvol_transition_quad.market import load_fixture
m = load_fixture()
c = snowball_contract(m.spot, 1.0, 12)
engine = LocalVolQuad(CTMCKernel(), GridConfig(cells=400, half_width_std=8.0))
a, b = engine.full_risk(c, m)["values"], engine.full_risk(c, m)["values"]
print(json.dumps({
    "identical": all(np.float64(a[k]).tobytes() == np.float64(b[k]).tobytes() for k in OUTPUTS),
    "peak_bytes": peak_bytes(),
}))
"""


def _run_child(code: str) -> dict:
    """Run ``code`` in a fresh interpreter (this test's import path, repo root as cwd); its last stdout line is
    JSON."""
    import os
    import subprocess

    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    proc = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                          capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_child_peak_excludes_the_parents_high_water_mark():
    # The CI failure mode of AC14 (x86 run 43837324340: 8.7-13 GB reported for a ~110 MB computation): the
    # spawning process's peak must not leak into the child's measurement.
    blob = np.ones(64 * 1024 ** 2)      # 512 MiB touched in this process
    del blob
    report = _run_child("import json, resource, sys" + _PEAK_BYTES + "print(json.dumps({'peak_bytes': peak_bytes()}))")
    assert report["peak_bytes"] < 256 * 1024 ** 2


@pytest.mark.slow
def test_ac14_full_risk_byte_identical_and_memory_bounded():
    # A fresh interpreter, and a peak that is its own (see _PEAK_BYTES): measured inside an xdist worker,
    # ru_maxrss reported the worker's earlier tests (8.6 GB, run 43695238886); measured as ru_maxrss in a
    # child, Linux still reported them, carried across execve (8.7-13 GB, run 43837324340).
    report = _run_child(_AC14_CHILD)
    assert report["identical"]
    assert report["peak_bytes"] < 8 * 1024 ** 3
