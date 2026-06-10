"""Metrics helpers for FI stress testing."""

from __future__ import annotations

from typing import Dict, List

from portfolio.fi.portfolio import FIPortfolio
from stresstest.fi.config import FIStressConfig


class FIMetricsCalculator:
    """Calculates FI-specific risk metrics used by the stress engine."""

    def __init__(self, config: FIStressConfig):
        self.config = config

    def portfolio_metrics(self, portfolio: FIPortfolio) -> Dict[str, float]:
        metrics = {
            "market_value": portfolio.get_portfolio_value(),
            "dv01": portfolio.get_portfolio_dv01(),
            "convexity": portfolio.get_portfolio_convexity(),
            "modified_duration": portfolio.get_portfolio_duration(),
            "carry": self._estimate_carry(portfolio),
        }

        if self.config.track_key_rate_dv01:
            metrics["key_rate_dv01"] = self._approximate_key_rate_dv01(
                metrics["dv01"]
            )
        else:
            metrics["key_rate_dv01"] = {}

        return metrics

    def _approximate_key_rate_dv01(self, total_dv01: float) -> Dict[str, float]:
        if total_dv01 == 0 or not self.config.key_rate_buckets:
            return {}

        split_value = total_dv01 / len(self.config.key_rate_buckets)
        return {bucket: split_value for bucket in self.config.key_rate_buckets}

    def _estimate_carry(self, portfolio: FIPortfolio) -> float:
        if not self.config.include_carry_metrics:
            return 0.0

        rates: List[float] = []
        for env in portfolio.pricing_environments.values():
            if env.rate_curve:
                try:
                    rates.append(env.rate_curve.get_rate(1.0))
                except Exception:
                    continue

        if not rates:
            return 0.0

        avg_rate = sum(rates) / len(rates)
        portfolio_value = portfolio.get_portfolio_value()
        return portfolio_value * avg_rate / 360.0

