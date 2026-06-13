"""
Credit Value-at-Risk engines (parametric / historical / Monte Carlo).

All three share a finite-difference sensitivity and full-revaluation core over
the CreditPricingEnvironment. Per-entity risk factors:

* ``{entity}_hazard_change`` - absolute hazard-intensity change (Hazard01 factor)
* ``{entity}_rate_shift``    - absolute interest-rate change

Input ``historical_data`` is a DataFrame of *levels* with columns
``{entity}_hazard`` and ``{entity}_rate``; factor changes are first differences.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from quantark.portfolio.credit import CreditPortfolio
from quantark.util.exceptions import MarketDataError, ValidationError
from quantark.util.numerical import is_zero
from quantark.var.config import VaRConfig, VaRMethod
from quantark.var.credit.config import CreditRiskFactorConfig
from quantark.var.credit.revaluation import bump_env, portfolio_value
from quantark.var.results import VaRResult

# Finite-difference bump sizes.
_EPS = {"hazard": 1e-4, "rate": 1e-4}

# (kind, level-column suffix, change-column suffix, include-flag attribute)
_FACTOR_SPECS: List[Tuple[str, str, str, str]] = [
    ("hazard", "_hazard", "_hazard_change", "include_hazard"),
    ("rate", "_rate", "_rate_shift", "include_rate"),
]
_KW = {"hazard": "hazard_change", "rate": "rate_shift"}


@dataclass(frozen=True)
class _Descriptor:
    column: str  # e.g. "ACME_hazard_change"
    entity: str  # e.g. "ACME"
    kind: str    # "hazard" or "rate"


class _BaseCreditVaREngine:
    """Shared credit VaR machinery."""

    def __init__(self, config: Optional[VaRConfig] = None):
        self.config = config if config is not None else VaRConfig()
        self.credit_factors = CreditRiskFactorConfig()

    def supports_portfolio(self, portfolio) -> bool:
        return isinstance(portfolio, CreditPortfolio)

    def _descriptors(
        self, portfolio: CreditPortfolio, df: pd.DataFrame
    ) -> List[_Descriptor]:
        descriptors: List[_Descriptor] = []
        entities = sorted({p.reference_entity for p in portfolio.positions.values()})
        for entity in entities:
            for kind, level_suffix, change_suffix, flag in _FACTOR_SPECS:
                if not getattr(self.credit_factors, flag):
                    continue
                if f"{entity}{level_suffix}" not in df.columns:
                    continue
                descriptors.append(
                    _Descriptor(f"{entity}{change_suffix}", entity, kind)
                )
        if not descriptors:
            raise MarketDataError(
                "No credit risk factors found. Provide level columns like "
                "'ACME_hazard' and 'ACME_rate'."
            )
        return descriptors

    def _factor_changes(
        self, df: pd.DataFrame, descriptors: List[_Descriptor]
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for d in descriptors:
            level_suffix = next(s for k, s, _, _ in _FACTOR_SPECS if k == d.kind)
            out[d.column] = df[f"{d.entity}{level_suffix}"].diff()
        out = out.dropna()
        if len(out) < self.config.lookback_days:
            raise MarketDataError(
                f"Insufficient credit history: {len(out)} rows, "
                f"{self.config.lookback_days} required"
            )
        return out.tail(self.config.lookback_days)

    def _base_envs(self, portfolio: CreditPortfolio) -> Dict[str, object]:
        return dict(portfolio.pricing_environments)

    @staticmethod
    def _bump_one(base_envs, d: _Descriptor, eps: float) -> Dict[str, object]:
        envs = dict(base_envs)
        envs[d.entity] = bump_env(base_envs[d.entity], **{_KW[d.kind]: eps})
        return envs

    def _sensitivities(
        self, portfolio: CreditPortfolio, descriptors: List[_Descriptor]
    ) -> Dict[str, float]:
        base_envs = self._base_envs(portfolio)
        sens: Dict[str, float] = {}
        for d in descriptors:
            eps = _EPS[d.kind]
            up = portfolio_value(portfolio, self._bump_one(base_envs, d, +eps))
            down = portfolio_value(portfolio, self._bump_one(base_envs, d, -eps))
            sens[d.column] = (up - down) / (2.0 * eps)
        return sens

    def _position_sensitivities(
        self, portfolio: CreditPortfolio, descriptors: List[_Descriptor]
    ) -> Dict[str, Dict[str, float]]:
        base_envs = self._base_envs(portfolio)
        result: Dict[str, Dict[str, float]] = {}
        for pid, position in portfolio.positions.items():
            vec: Dict[str, float] = {}
            for d in descriptors:
                if d.entity != position.reference_entity:
                    vec[d.column] = 0.0
                    continue
                eps = _EPS[d.kind]
                up_env = self._bump_one(base_envs, d, +eps)[d.entity]
                down_env = self._bump_one(base_envs, d, -eps)[d.entity]
                vec[d.column] = (
                    position.get_market_value(up_env)
                    - position.get_market_value(down_env)
                ) / (2.0 * eps)
            result[pid] = vec
        return result

    def _scenario_pnls(
        self,
        portfolio: CreditPortfolio,
        changes: pd.DataFrame,
        descriptors: List[_Descriptor],
    ) -> np.ndarray:
        base_envs = self._base_envs(portfolio)
        base_value = portfolio_value(portfolio, base_envs)
        by_entity: Dict[str, List[_Descriptor]] = {}
        for d in descriptors:
            by_entity.setdefault(d.entity, []).append(d)

        pnls = np.empty(len(changes))
        for i, (_, row) in enumerate(changes.iterrows()):
            envs = dict(base_envs)
            for entity, ds in by_entity.items():
                kw = {"hazard_change": 0.0, "rate_shift": 0.0}
                for d in ds:
                    kw[_KW[d.kind]] = float(row[d.column])
                envs[entity] = bump_env(base_envs[entity], **kw)
            pnls[i] = portfolio_value(portfolio, envs) - base_value
        return pnls

    def _z(self) -> float:
        return stats.norm.ppf(self.config.confidence_level)

    def _scale(self) -> float:
        if self.config.holding_period > 1 and self.config.scaling_method == "sqrt_t":
            return np.sqrt(self.config.holding_period)
        return 1.0

    def _finish(self, result: VaRResult, start: float) -> VaRResult:
        result.execution_time_seconds = time.time() - start
        base = {
            "confidence_level": self.config.confidence_level,
            "holding_period": self.config.holding_period,
            "lookback_days": self.config.lookback_days,
            "method": str(self.config.var_method),
        }
        result.config_summary = {**base, **result.config_summary}
        return result

    def _empty_check(self, portfolio) -> None:
        if not isinstance(portfolio, CreditPortfolio):
            raise ValidationError(
                f"Credit VaR engines require a CreditPortfolio, got "
                f"{type(portfolio).__name__}"
            )
        if len(portfolio.positions) == 0:
            raise ValidationError("Cannot calculate VaR for empty portfolio")


class CreditParametricVaREngine(_BaseCreditVaREngine):
    """Variance-covariance credit VaR: ``VaR = z * sqrt(sᵀ Σ s)``."""

    def calculate_var(
        self, portfolio: CreditPortfolio, historical_data: pd.DataFrame
    ) -> VaRResult:
        start = time.time()
        self._empty_check(portfolio)
        descriptors = self._descriptors(portfolio, historical_data)
        changes = self._factor_changes(historical_data, descriptors)
        sens = self._sensitivities(portfolio, descriptors)

        cols = [d.column for d in descriptors]
        s = np.array([sens[c] for c in cols])
        cov = changes[cols].cov().values
        sigma_pnl = float(np.sqrt(max(s @ cov @ s, 0.0))) * self._scale()

        z = self._z()
        var = z * sigma_pnl
        cvar = sigma_pnl * stats.norm.pdf(z) / (1.0 - self.config.confidence_level)
        pv = portfolio.get_portfolio_value()

        result = VaRResult(
            var=abs(var), cvar=abs(cvar),
            confidence_level=self.config.confidence_level,
            holding_period=self.config.holding_period, method=VaRMethod.PARAMETRIC,
            portfolio_value=pv, var_as_pct=abs(var) / pv if not is_zero(pv) else 0.0,
        )
        result.config_summary = {"sensitivities": sens}

        if self.config.calculate_factor_var and not is_zero(sigma_pnl):
            sigma_s = cov @ s
            contrib = z * (s * sigma_s) / (sigma_pnl / self._scale()) * self._scale()
            result.factor_var = {cols[i]: float(contrib[i]) for i in range(len(cols))}

        if self.config.calculate_component_var and not is_zero(sigma_pnl):
            pos_sens = self._position_sensitivities(portfolio, descriptors)
            sigma_s = cov @ s
            comp: Dict[str, float] = {}
            for pid, vec in pos_sens.items():
                s_pos = np.array([vec[c] for c in cols])
                comp[pid] = float(
                    z * (s_pos @ sigma_s) / (sigma_pnl / self._scale()) * self._scale()
                )
            result.component_var = comp

        return self._finish(result, start)


class CreditHistoricalVaREngine(_BaseCreditVaREngine):
    """Historical-simulation credit VaR via full revaluation."""

    def calculate_var(
        self, portfolio: CreditPortfolio, historical_data: pd.DataFrame
    ) -> VaRResult:
        start = time.time()
        self._empty_check(portfolio)
        descriptors = self._descriptors(portfolio, historical_data)
        changes = self._factor_changes(historical_data, descriptors)
        pnls = self._scenario_pnls(portfolio, changes, descriptors) * self._scale()

        var, cvar = _empirical_var_cvar(pnls, self.config.confidence_level)
        pv = portfolio.get_portfolio_value()
        result = VaRResult(
            var=var, cvar=cvar, confidence_level=self.config.confidence_level,
            holding_period=self.config.holding_period, method=VaRMethod.HISTORICAL,
            portfolio_value=pv, var_as_pct=var / pv if not is_zero(pv) else 0.0,
        )
        return self._finish(result, start)


class CreditMonteCarloVaREngine(_BaseCreditVaREngine):
    """Monte-Carlo credit VaR: draw factor changes from N(mean, Σ)."""

    def calculate_var(
        self, portfolio: CreditPortfolio, historical_data: pd.DataFrame
    ) -> VaRResult:
        start = time.time()
        self._empty_check(portfolio)
        descriptors = self._descriptors(portfolio, historical_data)
        changes = self._factor_changes(historical_data, descriptors)
        cols = [d.column for d in descriptors]

        mean = changes[cols].mean().values
        cov = changes[cols].cov().values
        rng = np.random.default_rng(self.config.mc_seed)
        draws = rng.multivariate_normal(mean, cov, size=self.config.mc_num_simulations)
        sim = pd.DataFrame(draws, columns=cols)

        pnls = self._scenario_pnls(portfolio, sim, descriptors) * self._scale()
        var, cvar = _empirical_var_cvar(pnls, self.config.confidence_level)
        pv = portfolio.get_portfolio_value()
        result = VaRResult(
            var=var, cvar=cvar, confidence_level=self.config.confidence_level,
            holding_period=self.config.holding_period, method=VaRMethod.MONTE_CARLO,
            portfolio_value=pv, var_as_pct=var / pv if not is_zero(pv) else 0.0,
        )
        result.config_summary = {"num_simulations": self.config.mc_num_simulations}
        return self._finish(result, start)


def _empirical_var_cvar(
    pnls: np.ndarray, confidence_level: float
) -> Tuple[float, float]:
    losses = -pnls
    var = float(np.quantile(losses, confidence_level))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size else var
    return max(var, 0.0), max(cvar, max(var, 0.0))
