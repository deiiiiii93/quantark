"""Tests for the 1001-point quadrature Snowball RFQ KO-rate demo wrapper."""

from __future__ import annotations

from asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from example import generate_snowball_rfq_ko_rate_demo_quad_1001 as demo


def test_main_wires_quad_1001_demo(monkeypatch) -> None:
    """The wrapper should build and write the demo with a 1001-point quad engine."""
    captured: dict[str, object] = {}

    def fake_build_cube_with_engines(*, engine, bump_engine, progress_label):
        captured["engine"] = engine
        captured["bump_engine"] = bump_engine
        captured["progress_label"] = progress_label
        return {}, []

    def fake_build_demo_data(**kwargs):
        captured["data_kwargs"] = kwargs
        return {"meta": {"engine": kwargs["engine_name"]}}

    def fake_write_demo_files(data, scenario_rows, **kwargs):
        captured["write_data"] = data
        captured["write_rows"] = scenario_rows
        captured["write_kwargs"] = kwargs

    monkeypatch.setattr(demo.base, "build_cube_with_engines", fake_build_cube_with_engines)
    monkeypatch.setattr(demo.base, "build_demo_data", fake_build_demo_data)
    monkeypatch.setattr(demo.base, "write_demo_files", fake_write_demo_files)

    demo.main()

    assert isinstance(captured["engine"], SnowballQuadEngine)
    assert isinstance(captured["bump_engine"], SnowballQuadEngine)
    assert captured["engine"].params.grid_points == 1001
    assert captured["bump_engine"].params.grid_points == 501
    assert captured["data_kwargs"]["engine_name"] == "SnowballQuadEngine"
    assert captured["write_kwargs"]["output_path"] == demo.OUTPUT_PATH
    assert captured["write_kwargs"]["csv_output_path"] == demo.CSV_OUTPUT_PATH
    assert captured["write_kwargs"]["ui_copy"] == demo.QUAD_HTML_UI_COPY
