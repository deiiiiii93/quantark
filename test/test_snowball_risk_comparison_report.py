from pathlib import Path

import pytest
from docx import Document

from asset.equity.report import snowball_risk_comparison_report as report
from util.enum import ObservationType


def _base_config(tmp_path: Path | None = None, *, num_paths: int = 4000):
    return report.SnowballRiskComparisonConfig(
        output_dir=(tmp_path / "risk_report" if tmp_path is not None else Path("output/test/snowball_risk_report")),
        num_paths=num_paths,
        generate_plots=False,
    )


@pytest.fixture(scope="module")
def analysis_bundle(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("snowball-risk-comparison")
    config = _base_config(tmp_path=tmp_path, num_paths=4000)
    (
        calendar,
        _ko_dates,
        ko_times,
        _daily_ki_dates,
        daily_ki_times,
        _maturity_date,
        maturity_time,
    ) = report._build_ko_schedule(config)
    pricing_env = report._build_pricing_environment(config, calendar)
    products = report._build_structures(config, ko_times, daily_ki_times, maturity_time)
    observation_times, _paths, snapshots = report._simulate_common_paths(
        config=config,
        pricing_env=pricing_env,
        products=products,
    )
    deterministic_cases = report._run_deterministic_cases(
        products=products,
        pricing_env=pricing_env,
        observation_times=observation_times,
    )
    return {
        "config": config,
        "pricing_env": pricing_env,
        "products": products,
        "observation_times": observation_times,
        "snapshots": snapshots,
        "deterministic_cases": {case.name: case for case in deterministic_cases},
    }


def test_schedule_builders_daily_vs_european_ki(analysis_bundle):
    products = analysis_bundle["products"]
    pricing_env = analysis_bundle["pricing_env"]

    ppp_dki = products["PPP-DKI"]
    ppp_eki = products["PPP-EKI"]

    assert ppp_dki.barrier_config.ki_observation_type == ObservationType.DISCRETE
    assert ppp_dki.barrier_config.ki_continuous is False
    assert len(ppp_dki.barrier_config.ki_observation_dates) > 200

    assert ppp_eki.barrier_config.ki_observation_type == ObservationType.DISCRETE
    assert ppp_eki.barrier_config.ki_continuous is False
    assert len(ppp_eki.barrier_config.ki_observation_dates) == 1
    assert ppp_eki.barrier_config.ki_observation_dates[0] == pytest.approx(
        ppp_eki.get_maturity(pricing_env)
    )


def test_monitoring_rebound_band_loss_probability_ordering(analysis_bundle):
    snapshots = analysis_bundle["snapshots"]
    assert (
        snapshots["PPP-EKI"].metrics.rebound_band_loss_probability
        < snapshots["PPP-DKI"].metrics.rebound_band_loss_probability
    )


def test_protection_improves_es_metrics(analysis_bundle):
    snapshots = analysis_bundle["snapshots"]
    assert snapshots["NPP-DKI"].metrics.es95 > snapshots["PPP-DKI"].metrics.es95
    assert snapshots["NPP-DKI"].metrics.es99 > snapshots["PPP-DKI"].metrics.es99


def test_monitoring_gap_narrows_in_terminal_below_ki_case(analysis_bundle):
    cases = analysis_bundle["deterministic_cases"]
    rebound_gap = (
        cases["Crash Then Partial Rebound 85"].outcomes["PPP-EKI"].payoff
        - cases["Crash Then Partial Rebound 85"].outcomes["PPP-DKI"].payoff
    )
    deep_bear_gap = (
        cases["Slow Bleed Lower"].outcomes["PPP-EKI"].payoff
        - cases["Slow Bleed Lower"].outcomes["PPP-DKI"].payoff
    )
    assert abs(rebound_gap) > abs(deep_bear_gap)


def test_parachute_dki_value_exceeds_eki_on_74_anchor(analysis_bundle):
    case = analysis_bundle["deterministic_cases"]["Parachute Rescue 74"]
    dki_increment = (
        case.outcomes["PPP-DKI-Parachute"].payoff - case.outcomes["PPP-DKI"].payoff
    )
    eki_increment = (
        case.outcomes["PPP-EKI-Parachute"].payoff - case.outcomes["PPP-EKI"].payoff
    )
    assert dki_increment > eki_increment


def test_greek_curves_cover_requested_structures(analysis_bundle):
    greek_curves = report._compute_greek_curves(
        config=analysis_bundle["config"],
        pricing_env=analysis_bundle["pricing_env"],
        products=analysis_bundle["products"],
    )
    for frame in (
        greek_curves.delta,
        greek_curves.gamma,
        greek_curves.vega,
        greek_curves.rhoq,
        greek_curves.delta_q_curve,
        greek_curves.gamma_q_curve,
        greek_curves.vega_q_curve,
        greek_curves.rhoq_q_curve,
    ):
        first_col = frame.columns[0]
        assert first_col in {"spot", "q"}
        assert list(frame.columns) == [first_col, "PPP-EKI", "PPP-DKI", "NPP-DKI"]
    assert len(greek_curves.delta) == len(analysis_bundle["config"].greek_spot_multipliers)
    assert len(greek_curves.delta_q_curve) == len(analysis_bundle["config"].greek_q_shifts)
    assert greek_curves.q_grid.min() == pytest.approx(0.08)
    assert greek_curves.q_grid.max() == pytest.approx(0.15)
    assert set(greek_curves.key_spot_table["Structure"]) == {
        "PPP-EKI",
        "PPP-DKI",
        "NPP-DKI",
    }
    assert set(greek_curves.key_q_table["Structure"]) == {
        "PPP-EKI",
        "PPP-DKI",
        "NPP-DKI",
    }
    assert set(greek_curves.q_slice_curves.keys()) == {"near_ki", "near_ko"}
    for slice_frames in greek_curves.q_slice_curves.values():
        for greek_name in ("delta", "gamma", "vega", "rhoq"):
            frame = slice_frames[greek_name]
            assert list(frame.columns) == ["q", "PPP-EKI", "PPP-DKI", "NPP-DKI"]


def test_bilingual_docx_smoke(tmp_path):
    config = _base_config(tmp_path=tmp_path, num_paths=512)
    result = report.generate_snowball_risk_comparison_report(config)

    assert result.report_path.exists()
    doc = Document(str(result.report_path))
    texts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    en_idx = texts.index("Snowball Monitoring / Protection / Parachute Risk Comparison")
    zh_idx = texts.index("雪球监控方式 / 保本 / 降落伞特征风险对比报告")
    assert "Contract Terms Comparison" in texts
    assert "Greeks Comparison" in texts
    assert "Key Dividend Yield Greek Levels" in texts
    assert "Near-KI and Near-KO q slices" in texts
    assert "Appendix: Terminology" in texts
    assert "合约条款对比表" in texts
    assert "Greeks 对比" in texts
    assert "关键 q 水平 Greek 对比" in texts
    assert "近 KI 与近 KO 的 q 切片" in texts
    assert "附录：术语说明" in texts
    assert en_idx < zh_idx
