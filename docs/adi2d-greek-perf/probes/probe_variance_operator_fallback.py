"""Which certification cells actually trigger the donor-cell fallback?

`adaptive_upwind` keeps the second-order centered row wherever that row stays an
M-matrix and drops to a first-order donor cell only where it does not. So the
upwind truncation error the 2026-08-10 probe measured cannot exist in a cell
whose centered row is monotone everywhere -- there, `adaptive_upwind` *is*
centered. This reads the operator diagnostics per cell to say exactly where the
fallback engages, which is the mechanism behind the cross-scheme gaps in
probe_scheme_cell_matrix.py.

Builds the variance operator only; no time march, so it costs seconds.

Usage:
    PYTHONPATH=$PWD python docs/adi2d-greek-perf/probes/probe_variance_operator_fallback.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_scheme_cell_matrix import OUTPUT_DIR, load_stage16  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    s16 = load_stage16()
    rows = []
    for case in s16.certification_cases(quick=False):
        for variant in ("heston", "heston_slv"):
            product = s16.make_snowball(case, dense_ki=True)
            env = s16.make_environment(
                case.spot, np.sqrt(max(case.params.v0, case.params.theta))
            )
            leverage = (
                s16.make_leverage_surface(case.maturity)
                if variant == "heston_slv"
                else None
            )
            ladders = s16.grid_ladders(
                case.maturity, quick=False, dense_ki_stencil=(case.name == "near_ki")
            )
            engine = s16.make_pde_engine(variant, case, ladders["target"], leverage)
            core = engine._make_core(product, env, case.maturity)
            diagnostics = core.variance_operator_diagnostics()
            rows.append(
                {"variant": variant, "case": case.name, "diagnostics": diagnostics}
            )
            print(
                f"  {variant + '/' + case.name:26s} "
                f"interior={diagnostics['interior_nodes']:4d} "
                f"centered_non_monotone={diagnostics['centered_non_monotone_nodes']:4d} "
                f"fallback={diagnostics['fallback_nodes']:4d} "
                f"max_peclet={diagnostics['max_local_peclet']:9.3f}",
                flush=True,
            )
    path = OUTPUT_DIR / "variance_operator_fallback.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
