"""
Render the PDE-backed Snowball RFQ KO-rate HTML from a saved JSON payload.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from example import snowball_rfq_ko_rate_demo_workflow as workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the PDE Snowball RFQ KO-rate dashboard from JSON.",
    )
    parser.add_argument(
        "--data-input",
        type=Path,
        default=workflow.PDE_DATA_OUTPUT_PATH,
        help="Input path for the saved JSON payload.",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=workflow.PDE_HTML_OUTPUT_PATH,
        help="Output path for the rendered HTML dashboard.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = workflow.read_demo_data_json(args.data_input)
    workflow.write_demo_html(data, html_output_path=args.html_output)
    print(f"Wrote PDE demo HTML to {args.html_output}")


if __name__ == "__main__":
    main()
