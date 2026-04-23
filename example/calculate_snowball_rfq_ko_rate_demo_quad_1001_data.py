"""
Calculate the quad(1001) Snowball RFQ KO-rate demo data payload and CSV export.

This is the expensive step: it builds the exact barrier grid payload used by the
HTML demo, but does not render HTML.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from asset.equity.param import QuadParams
from example import generate_snowball_rfq_ko_rate_demo as base


DATA_OUTPUT_PATH = ROOT / "output" / "snowball_rfq_ko_rate_demo_quad_1001_data.json"
CSV_OUTPUT_PATH = ROOT / "output" / "snowball_rfq_ko_rate_scenarios_quad_1001.csv"
QUAD_GRID_POINTS = 1001


def main() -> None:
    quad_params = QuadParams(grid_points=QUAD_GRID_POINTS)
    cubes, scenario_rows = base.build_cube_with_engines(
        engine=SnowballQuadEngine(params=quad_params),
        bump_engine=SnowballQuadEngine(params=QuadParams(grid_points=QUAD_GRID_POINTS)),
        progress_label=f"SnowballQuadEngine({quad_params.grid_points})",
        exact_barrier_grid=True,
        parallel_workers=max(1, os.cpu_count() or 1),
    )
    data = base.build_demo_data(
        cubes=cubes,
        engine_name="SnowballQuadEngine",
        solver_grid_size=quad_params.grid_points,
        solver_time_steps=None,
        exact_barrier_grid=True,
    )
    base.write_demo_data_json(data, data_output_path=DATA_OUTPUT_PATH)
    base.write_scenario_csv(scenario_rows, csv_output_path=CSV_OUTPUT_PATH)
    print(f"Wrote quad demo data JSON to {DATA_OUTPUT_PATH}")
    print(f"Wrote quad scenario CSV to {CSV_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
