"""
Generate the quad-backed Snowball RFQ KO-rate demo end to end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from example import snowball_rfq_ko_rate_demo_workflow as workflow


def build_quad_ui_copy(grid_points: int) -> dict[str, str]:
    return {
        "eyebrow_en": f"Quad ({grid_points}) risk demo",
        "chip_engine_en": f"Snowball quadrature cube ({grid_points} points)",
        "cube_note_en": f"Embedded exact KO/KI cube solved by quadrature with {grid_points} grid points.",
        "eyebrow_cn": f"Quad（{grid_points}）风控演示",
        "chip_engine_cn": f"雪球 Quad 精确立方体（{grid_points} 点）",
        "cube_note_cn": f"页面内嵌基于 {grid_points} 点 quadrature 求解的精确 KO/KI 立方体。",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the quad Snowball RFQ KO-rate dashboard.",
    )
    parser.add_argument(
        "--grid-points",
        type=int,
        default=1001,
        help="Quadrature grid points.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel worker count for the exact KO/KI cube build.",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=workflow.QUAD_HTML_OUTPUT_PATH,
        help="Output path for the rendered HTML dashboard.",
    )
    parser.add_argument(
        "--data-output",
        type=Path,
        default=workflow.QUAD_DATA_OUTPUT_PATH,
        help="Output path for the saved JSON payload.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=workflow.QUAD_CSV_OUTPUT_PATH,
        help="Output path for the scenario CSV export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, scenario_rows = workflow.calculate_quad_demo_data(
        grid_points=args.grid_points,
        parallel_workers=args.workers,
    )
    workflow.write_demo_artifacts(
        data,
        scenario_rows,
        html_output_path=args.html_output,
        data_output_path=args.data_output,
        csv_output_path=args.csv_output,
        ui_copy=build_quad_ui_copy(args.grid_points),
    )
    print(f"Wrote quad demo HTML to {args.html_output}")
    print(f"Wrote quad demo data JSON to {args.data_output}")
    print(f"Wrote quad scenario CSV to {args.csv_output}")


if __name__ == "__main__":
    main()
