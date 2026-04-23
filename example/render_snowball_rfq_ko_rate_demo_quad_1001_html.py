"""
Render the quad(1001) Snowball RFQ KO-rate HTML from a saved JSON payload.

This is the cheap step: it reads the precomputed data JSON and writes HTML
without rerunning the full pricing grid.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from example import generate_snowball_rfq_ko_rate_demo as base


DATA_INPUT_PATH = ROOT / "output" / "snowball_rfq_ko_rate_demo_quad_1001_data.json"
OUTPUT_PATH = ROOT / "output" / "snowball_rfq_ko_rate_demo_quad_1001.html"

QUAD_HTML_UI_COPY = {
    "eyebrow_en": "Quad (1001) RFQ explainer",
    "chip_engine_en": "Snowball quadrature cube (1001 points) + interpolation",
    "cube_note_en": "The HTML embeds a quadrature-solved cube using a 1001-point grid and interpolates between nodes in-browser.",
    "eyebrow_cn": "Quad（1001）驱动 RFQ 解释器",
    "chip_engine_cn": "雪球 Quad 立方体（1001 点）+ 插值",
    "cube_note_cn": "页面内嵌基于 1001 点 Quad 求解的立方体，并在浏览器端做插值。",
}


def main() -> None:
    data = base.read_demo_data_json(DATA_INPUT_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        base.render_html(data, ui_copy=QUAD_HTML_UI_COPY),
        encoding="utf-8",
    )
    print(f"Wrote quad demo HTML to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
