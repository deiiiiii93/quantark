"""
Generate a quad-backed standalone HTML demo for Snowball RFQ KO-rate quoting.

This mirrors `generate_snowball_rfq_ko_rate_demo.py`, but prices the embedded
cube with `SnowballQuadEngine` using a 1001-point quadrature grid.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from asset.equity.param import QuadParams
from example import generate_snowball_rfq_ko_rate_demo as base


OUTPUT_PATH = ROOT / "output" / "snowball_rfq_ko_rate_demo_quad_1001.html"
CSV_OUTPUT_PATH = ROOT / "output" / "snowball_rfq_ko_rate_scenarios_quad_1001.csv"
QUAD_GRID_POINTS = 1001

QUAD_HTML_UI_COPY = {
    "eyebrow_en": "Quad (1001) RFQ explainer",
    "chip_engine_en": "Snowball quadrature cube (1001 points) + interpolation",
    "cube_note_en": "The HTML embeds a quadrature-solved cube using a 1001-point grid and interpolates between nodes in-browser.",
    "eyebrow_cn": "Quad（1001）驱动 RFQ 解释器",
    "chip_engine_cn": "雪球 Quad 立方体（1001 点）+ 插值",
    "cube_note_cn": "页面内嵌基于 1001 点 Quad 求解的立方体，并在浏览器端做插值。",
}


def main() -> None:
    quad_params = QuadParams(grid_points=QUAD_GRID_POINTS)
    bump_params = QuadParams(grid_points=max(101, quad_params.grid_points // 2))
    cubes, scenario_rows = base.build_cube_with_engines(
        engine=SnowballQuadEngine(params=quad_params),
        bump_engine=SnowballQuadEngine(params=bump_params),
        progress_label=f"SnowballQuadEngine({quad_params.grid_points})",
    )
    data = base.build_demo_data(
        cubes=cubes,
        engine_name="SnowballQuadEngine",
        solver_grid_size=quad_params.grid_points,
        solver_time_steps=None,
    )
    base.write_demo_files(
        data,
        scenario_rows,
        output_path=OUTPUT_PATH,
        csv_output_path=CSV_OUTPUT_PATH,
        ui_copy=QUAD_HTML_UI_COPY,
    )
    print(f"Wrote quad demo HTML to {OUTPUT_PATH}")
    print(f"Wrote quad scenario CSV to {CSV_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
