"""Install the W17 grid probe in EVERY interpreter started with this directory on PYTHONPATH (pytest workers and
any spawned child a certification pipeline starts), not only in the pytest process that loads the plugin.
Active only when W17_PROBE_MODE is set."""
import os

if os.environ.get("W17_PROBE_MODE"):
    try:
        import w17_grid_probe

        w17_grid_probe.pytest_configure(None)
    except Exception as exc:  # noqa: BLE001 -- a probe must never break the run it observes
        import sys

        sys.stderr.write(f"w17 probe not installed: {type(exc).__name__}: {exc}\n")
