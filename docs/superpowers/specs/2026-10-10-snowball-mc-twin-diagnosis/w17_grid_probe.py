"""pytest plugin (``-p w17_grid_probe``): which existing tests price an autocallable MC grid holding a sub-tolerance
step, i.e. which test values the proposed merge would move. Runtime wrap only, no library change.

W17_PROBE_MODE=detect (default): log every grid built by SnowballMCEngine._build_time_grid /
    _build_time_grid_ko_reset and PhoenixMCEngine._build_time_grid whose smallest step is below TOL; the grid is
    returned unchanged.
W17_PROBE_MODE=patch: additionally install patch_grid.install() (snowball + Phoenix merge) so the run shows which
    assertions the moved values break.
Log lines go to $W17_PROBE_LOG (one file per worker, appended)."""
import os

import numpy as np

TOL = 1e-10


def _log(kind, all_times):
    at = np.asarray(all_times, float)
    if at.size < 2:
        return
    steps = np.diff(at)
    if steps.min() >= TOL:
        return
    path = os.environ.get("W17_PROBE_LOG", "w17_probe.log")
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    with open(f"{path}.{worker}", "a") as fh:
        fh.write(f"{os.environ.get('PYTEST_CURRENT_TEST', '?')}\t{kind}\tnodes={at.size}\t"
                 f"min_step={steps.min():.3e}\n")


def pytest_configure(config):
    from quantark.asset.equity.engine.mc.phoenix_mc_engine import PhoenixMCEngine
    from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine

    if os.environ.get("W17_PROBE_MODE", "detect") == "patch":
        import patch_grid
        patch_grid.install()

    snow, ko_reset, phx = (SnowballMCEngine._build_time_grid, SnowballMCEngine._build_time_grid_ko_reset,
                           PhoenixMCEngine._build_time_grid)

    def snow_w(self, product, pricing_env, T):
        out = snow(self, product, pricing_env, T)
        _log(f"snowball:{type(self).__name__}", out[0])
        return out

    def ko_reset_w(self, product, pricing_env, T):
        out = ko_reset(self, product, pricing_env, T)
        _log(f"ko_reset:{type(self).__name__}", out["all_times"])
        return out

    def phx_w(self, product, pricing_env, T):
        out = phx(self, product, pricing_env, T)
        _log(f"phoenix:{type(self).__name__}", out[0])
        return out

    SnowballMCEngine._build_time_grid = snow_w
    SnowballMCEngine._build_time_grid_ko_reset = ko_reset_w
    PhoenixMCEngine._build_time_grid = phx_w
