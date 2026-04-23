"""Input module for the bilingual snowball risk comparison report."""

from pathlib import Path

from asset.equity.report.snowball_risk_comparison_report import (
    SnowballRiskComparisonConfig,
)


def build_config() -> SnowballRiskComparisonConfig:
    return SnowballRiskComparisonConfig(
        output_dir=Path("output/doc/snowball_risk_comparison_example"),
        report_filename="snowball_risk_comparison_bilingual.docx",
        num_paths=12000,
        generate_plots=True,
    )
