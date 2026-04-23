"""
Compatibility wrapper for the quad(1001) Snowball RFQ KO-rate demo.

Runs the expensive data calculation step first, then renders HTML from the
saved JSON payload.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from example.calculate_snowball_rfq_ko_rate_demo_quad_1001_data import main as calculate_main
from example.render_snowball_rfq_ko_rate_demo_quad_1001_html import main as render_main


def main() -> None:
    calculate_main()
    render_main()


if __name__ == "__main__":
    main()
