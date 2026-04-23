"""Tests for the 1001-point quadrature Snowball RFQ KO-rate demo scripts."""

from __future__ import annotations

from example import calculate_snowball_rfq_ko_rate_demo_quad_1001_data as calc
from example import generate_snowball_rfq_ko_rate_demo_quad_1001 as demo
from example import render_snowball_rfq_ko_rate_demo_quad_1001_html as render


def test_wrapper_runs_calculate_then_render(monkeypatch) -> None:
    """The compatibility wrapper should invoke the two-step workflow."""
    calls: list[str] = []

    monkeypatch.setattr(demo, "calculate_main", lambda: calls.append("calculate"))
    monkeypatch.setattr(demo, "render_main", lambda: calls.append("render"))

    demo.main()

    assert calls == ["calculate", "render"]


def test_calculate_script_builds_data_and_csv(monkeypatch) -> None:
    """The data script should write the JSON payload and scenario CSV."""
    captured: dict[str, object] = {}

    def fake_build_cube_with_engines(
        *,
        engine,
        bump_engine,
        progress_label,
        exact_barrier_grid,
        parallel_workers,
    ):
        captured["engine"] = engine
        captured["bump_engine"] = bump_engine
        captured["progress_label"] = progress_label
        captured["exact_barrier_grid"] = exact_barrier_grid
        captured["parallel_workers"] = parallel_workers
        return {}, []

    def fake_build_demo_data(**kwargs):
        captured["data_kwargs"] = kwargs
        return {"meta": {"engine": kwargs["engine_name"]}}

    monkeypatch.setattr(calc.base, "build_cube_with_engines", fake_build_cube_with_engines)
    monkeypatch.setattr(calc.base, "build_demo_data", fake_build_demo_data)
    monkeypatch.setattr(
        calc.base,
        "write_demo_data_json",
        lambda data, **kwargs: captured.update({"json_data": data, "json_kwargs": kwargs}),
    )
    monkeypatch.setattr(
        calc.base,
        "write_scenario_csv",
        lambda rows, **kwargs: captured.update({"csv_rows": rows, "csv_kwargs": kwargs}),
    )

    calc.main()

    assert captured["engine"].params.grid_points == 1001
    assert captured["bump_engine"].params.grid_points == 1001
    assert captured["exact_barrier_grid"] is True
    assert captured["parallel_workers"] >= 1
    assert captured["data_kwargs"]["engine_name"] == "SnowballQuadEngine"
    assert captured["data_kwargs"]["exact_barrier_grid"] is True
    assert captured["json_kwargs"]["data_output_path"] == calc.DATA_OUTPUT_PATH
    assert captured["csv_kwargs"]["csv_output_path"] == calc.CSV_OUTPUT_PATH


def test_render_script_reads_json_and_writes_html(monkeypatch, tmp_path) -> None:
    """The render script should consume saved JSON and emit HTML only."""
    data_path = tmp_path / "payload.json"
    output_path = tmp_path / "demo.html"
    monkeypatch.setattr(render, "DATA_INPUT_PATH", data_path)
    monkeypatch.setattr(render, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(render.base, "read_demo_data_json", lambda path: {"meta": {"engine": "SnowballQuadEngine"}})
    monkeypatch.setattr(render.base, "render_html", lambda data, ui_copy=None: f"<html>{data['meta']['engine']}</html>")

    render.main()

    assert output_path.read_text(encoding="utf-8") == "<html>SnowballQuadEngine</html>"
