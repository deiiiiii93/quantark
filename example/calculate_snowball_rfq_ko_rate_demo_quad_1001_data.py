"""
Calculate the quad-backed Snowball RFQ KO-rate demo payload and CSV export.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


from example import snowball_rfq_ko_rate_demo_workflow as workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate the quad Snowball RFQ KO-rate demo payload.",
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
        data_output_path=args.data_output,
        csv_output_path=args.csv_output,
    )
    print(f"Wrote quad demo data JSON to {args.data_output}")
    print(f"Wrote quad scenario CSV to {args.csv_output}")


if __name__ == "__main__":
    main()
