"""The legacy daily files that must stay green through intraday work; a rename must not silently drop one."""
from pathlib import Path

GATE = [
    "test/test_trading_clock_map.py", "test/test_trading_clock_surface.py", "test/test_trading_clock_parity.py",
    "test/test_trading_clock_axis_equivalence.py", "test/test_term_sampling_total_variance.py",
    "test/test_quad_v2_engine.py", "test/test_quad_v2_terms_lifecycle.py", "test/test_quad_v2_coupon_event_stats.py",
    "test/test_observation_schedule.py", "test/test_equity_settlement_resolver.py", "test/test_lifecycle_cashflow_ledger.py",
    "test/test_equity_lifecycle_trackers.py", "test/test_snowball_pde_date_schedule.py", "test/test_digital_option_mc_engine.py",
    "test/execution/test_session_parity.py", "test/execution/test_registry.py",
]


def test_daily_regression_gate_files_exist():
    root = Path(__file__).resolve().parents[2]
    missing = [p for p in GATE if not (root / p).exists()]
    assert not missing, missing
