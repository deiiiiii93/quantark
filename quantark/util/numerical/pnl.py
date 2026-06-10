"""P&L calculation helpers."""


def pnl_pct_of_abs_baseline(pnl: float, baseline_value: float) -> float:
    """
    Calculate P&L percentage relative to absolute baseline exposure.

    Args:
        pnl: Absolute P&L amount.
        baseline_value: Baseline portfolio or position value.

    Returns:
        P&L percentage using ``abs(baseline_value)`` as denominator, or 0.0
        when baseline is zero.
    """
    denominator = abs(baseline_value)
    return (pnl / denominator * 100.0) if denominator else 0.0
