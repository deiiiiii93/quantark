"""Input module for the bilingual snowball risk comparison report."""

from pathlib import Path

from quantark.asset.equity.report.snowball_risk_comparison_report import (
    SnowballRiskComparisonConfig,
)
from quantark.asset.equity.param import QuadParams
from quantark.util.enum.engine_enums import MonteCarloMethod


def build_config() -> SnowballRiskComparisonConfig:
    return SnowballRiskComparisonConfig(
        output_dir=Path("output/doc/snowball_risk_comparison_example"),
        report_filename="snowball_risk_comparison_bilingual.docx",
        tenor_months=36,
        num_paths=50000,
        mc_method=MonteCarloMethod.QUASI,
        mc_use_parallel=True,
        mc_num_batches=8,
        engine_preference=("quad",),
        quad_params=QuadParams(grid_points=1001),
        generate_plots=True,
    )
