"""Tests for the 1001-point quadrature Snowball RFQ KO-rate demo scripts.

The demo family is a set of thin argparse CLIs over
``example.snowball_rfq_ko_rate_demo_workflow``:

- ``generate_…_quad_1001`` calculates the payload and writes all artifacts
  (JSON + CSV + HTML) in one shot;
- ``calculate_…_quad_1001_data`` writes the JSON payload and scenario CSV;
- ``render_…_quad_1001_html`` reads a saved JSON payload and emits HTML only.
"""

from __future__ import annotations

from pathlib import Path

from example import calculate_snowball_rfq_ko_rate_demo_quad_1001_data as calc
from example import generate_snowball_rfq_ko_rate_demo_quad_1001 as demo
from example import render_snowball_rfq_ko_rate_demo_quad_1001_html as render


def test_wrapper_calculates_then_writes_all_artifacts(monkeypatch) -> None:
    """The one-shot wrapper should calculate the cube then write JSON/CSV/HTML."""
    captured: dict[str, object] = {}

    def fake_calculate_quad_demo_data(*, grid_points, parallel_workers):
        captured["grid_points"] = grid_points
        captured["parallel_workers"] = parallel_workers
        return {"meta": {"engine": "SnowballQuadEngine"}}, [{"scenario_id": 1}]

    def fake_write_demo_artifacts(data, scenario_rows, **kwargs):
        captured["data"] = data
        captured["scenario_rows"] = scenario_rows
        captured["artifact_kwargs"] = kwargs

    monkeypatch.setattr(
        demo.workflow, "calculate_quad_demo_data", fake_calculate_quad_demo_data
    )
    monkeypatch.setattr(demo.workflow, "write_demo_artifacts", fake_write_demo_artifacts)
    monkeypatch.setattr("sys.argv", ["generate_snowball_rfq_ko_rate_demo_quad_1001"])

    demo.main()

    artifact_kwargs = captured["artifact_kwargs"]
    assert captured["grid_points"] == 1001
    assert captured["parallel_workers"] is None
    assert captured["data"] == {"meta": {"engine": "SnowballQuadEngine"}}
    assert captured["scenario_rows"] == [{"scenario_id": 1}]
    assert artifact_kwargs["html_output_path"] == demo.workflow.QUAD_HTML_OUTPUT_PATH
    assert artifact_kwargs["data_output_path"] == demo.workflow.QUAD_DATA_OUTPUT_PATH
    assert artifact_kwargs["csv_output_path"] == demo.workflow.QUAD_CSV_OUTPUT_PATH
    assert artifact_kwargs["ui_copy"]["eyebrow_en"] == "Quad (1001) risk demo"


def test_calculate_script_builds_data_and_csv(monkeypatch) -> None:
    """The data script should write the JSON payload and scenario CSV only."""
    captured: dict[str, object] = {}

    def fake_calculate_quad_demo_data(*, grid_points, parallel_workers):
        captured["grid_points"] = grid_points
        captured["parallel_workers"] = parallel_workers
        return {"meta": {"engine": "SnowballQuadEngine"}}, [{"scenario_id": 1}]

    def fake_write_demo_artifacts(data, scenario_rows, **kwargs):
        captured["data"] = data
        captured["scenario_rows"] = scenario_rows
        captured["artifact_kwargs"] = kwargs

    monkeypatch.setattr(
        calc.workflow, "calculate_quad_demo_data", fake_calculate_quad_demo_data
    )
    monkeypatch.setattr(calc.workflow, "write_demo_artifacts", fake_write_demo_artifacts)
    monkeypatch.setattr("sys.argv", ["calculate_snowball_rfq_ko_rate_demo_quad_1001_data"])

    calc.main()

    artifact_kwargs = captured["artifact_kwargs"]
    assert captured["grid_points"] == 1001
    assert artifact_kwargs["data_output_path"] == calc.workflow.QUAD_DATA_OUTPUT_PATH
    assert artifact_kwargs["csv_output_path"] == calc.workflow.QUAD_CSV_OUTPUT_PATH
    assert "html_output_path" not in artifact_kwargs


def test_render_script_reads_json_and_writes_html(monkeypatch, tmp_path) -> None:
    """The render script should consume saved JSON and emit HTML only."""
    captured: dict[str, object] = {}
    data_path = tmp_path / "payload.json"
    html_path = tmp_path / "demo.html"

    def fake_read_demo_data_json(path):
        captured["read_path"] = Path(path)
        return {"meta": {"engine": "SnowballQuadEngine", "solver_grid_size": 1001}}

    def fake_write_demo_html(data, *, html_output_path, ui_copy):
        captured["data"] = data
        captured["html_output_path"] = Path(html_output_path)
        captured["ui_copy"] = ui_copy

    monkeypatch.setattr(render.workflow, "read_demo_data_json", fake_read_demo_data_json)
    monkeypatch.setattr(render.workflow, "write_demo_html", fake_write_demo_html)
    monkeypatch.setattr(
        "sys.argv",
        [
            "render_snowball_rfq_ko_rate_demo_quad_1001_html",
            "--data-input",
            str(data_path),
            "--html-output",
            str(html_path),
        ],
    )

    render.main()

    assert captured["read_path"] == data_path
    assert captured["html_output_path"] == html_path
    assert captured["ui_copy"]["eyebrow_en"] == "Quad (1001) risk demo"
    assert "1001" in captured["ui_copy"]["cube_note_en"]
