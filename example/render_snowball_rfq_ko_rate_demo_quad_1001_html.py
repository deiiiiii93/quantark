"""
Render the quad-backed Snowball RFQ KO-rate HTML from a saved JSON payload.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from example import snowball_rfq_ko_rate_demo_workflow as workflow


def build_quad_ui_copy(grid_points: int | None) -> dict[str, str]:
    label = grid_points if grid_points is not None else "N/A"
    return {
        "eyebrow_en": f"Quad ({label}) risk demo",
        "chip_engine_en": f"Snowball quadrature cube ({label} points)",
        "cube_note_en": f"Embedded exact KO/KI cube solved by quadrature with {label} grid points.",
        "eyebrow_cn": f"Quad（{label}）风控演示",
        "chip_engine_cn": f"雪球 Quad 精确立方体（{label} 点）",
        "cube_note_cn": f"页面内嵌基于 {label} 点 quadrature 求解的精确 KO/KI 立方体。",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the quad Snowball RFQ KO-rate dashboard from JSON.",
    )
    parser.add_argument(
        "--data-input",
        type=Path,
        default=workflow.QUAD_DATA_OUTPUT_PATH,
        help="Input path for the saved JSON payload.",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=workflow.QUAD_HTML_OUTPUT_PATH,
        help="Output path for the rendered HTML dashboard.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = workflow.read_demo_data_json(args.data_input)
    grid_points = data.get("meta", {}).get("solver_grid_size")
    workflow.write_demo_html(
        data,
        html_output_path=args.html_output,
        ui_copy=build_quad_ui_copy(grid_points),
    )
    print(f"Wrote quad demo HTML to {args.html_output}")


if __name__ == "__main__":
    main()
