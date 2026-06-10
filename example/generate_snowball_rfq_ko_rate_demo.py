"""
Generate the PDE-backed Snowball RFQ KO-rate demo end to end.

Workflow:
1. Calculate the embedded pricing payload and scenario CSV.
2. Render the standalone HTML dashboard from that payload.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


from example import snowball_rfq_ko_rate_demo_workflow as workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the PDE Snowball RFQ KO-rate dashboard.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=workflow.PDE_GRID_SIZE,
        help="PDE spatial grid size.",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=workflow.PDE_TIME_STEPS,
        help="PDE time steps.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Parallel worker count for the PDE anchor grid build.",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=workflow.PDE_HTML_OUTPUT_PATH,
        help="Output path for the rendered HTML dashboard.",
    )
    parser.add_argument(
        "--data-output",
        type=Path,
        default=workflow.PDE_DATA_OUTPUT_PATH,
        help="Output path for the saved JSON payload.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=workflow.PDE_CSV_OUTPUT_PATH,
        help="Output path for the scenario CSV export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, scenario_rows = workflow.calculate_pde_demo_data(
        grid_size=args.grid_size,
        time_steps=args.time_steps,
        parallel_workers=args.workers,
    )
    workflow.write_demo_artifacts(
        data,
        scenario_rows,
        html_output_path=args.html_output,
        data_output_path=args.data_output,
        csv_output_path=args.csv_output,
    )
    print(f"Wrote PDE demo HTML to {args.html_output}")
    print(f"Wrote PDE demo data JSON to {args.data_output}")
    print(f"Wrote PDE scenario CSV to {args.csv_output}")


if __name__ == "__main__":
    main()
