"""
FX Value-at-Risk engines (parametric / historical / Monte Carlo).

All three share a finite-difference sensitivity and full-revaluation core over
the two-rate FxPricingEnvironment. Per-pair risk factors:

* ``{pair}_spot_return`` - relative spot change
* ``{pair}_vol_change``  - absolute vol change
* ``{pair}_dom_shift``   - absolute domestic-rate change
* ``{pair}_for_shift``   - absolute foreign-rate change

Input ``historical_data`` is a DataFrame of *levels* with columns
``{pair}_spot``, ``{pair}_vol``, ``{pair}_dom_rate``, ``{pair}_for_rate``;
factor changes are derived (pct-change for spot, diff for the rest).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from quantark.portfolio.fx import FXPortfolio
from quantark.util.exceptions import MarketDataError, ValidationError
from quantark.util.numerical import is_zero
from quantark.var.config import VaRConfig, VaRMethod
from quantark.var.fx.config import FXRiskFactorConfig
from quantark.var.fx.revaluation import bump_env, portfolio_value
from quantark.var.results import VaRResult

# Finite-difference bump sizes (small, central differences).
_EPS = {"spot": 1e-4, "vol": 1e-4, "dom": 1e-4, "for": 1e-4}

# (level column suffix, change column suffix, include-flag attribute)
_FACTOR_SPECS: List[Tuple[str, str, str, str]] = [
    ("spot", "_spot", "_spot_return", "include_spot"),
    ("vol", "_vol", "_vol_change", "include_vol"),
    ("dom", "_dom_rate", "_dom_shift", "include_domestic_rate"),
    ("for", "_for_rate", "_for_shift", "include_foreign_rate"),
]


@dataclass(frozen=True)
class _Descriptor:
    column: str  # e.g. "EURUSD_spot_return"
    pair: str  # e.g. "EURUSD"
    kind: str  # one of spot / vol / dom / for


class _BaseFXVaREngine:
    """Shared FX VaR machinery."""

    def __init__(self, config: Optional[VaRConfig] = None):
        self.config = config if config is not None else VaRConfig()
        self.fx_factors = FXRiskFactorConfig()

    def supports_portfolio(self, portfolio) -> bool:
        return isinstance(portfolio, FXPortfolio)

    # -- factor plumbing ------------------------------------------------ #
    def _descriptors(self, portfolio: FXPortfolio, df: pd.DataFrame) -> List[_Descriptor]:
        descriptors: List[_Descriptor] = []
        pairs = sorted({p.underlying for p in portfolio.positions.values()})
        for pair in pairs:
            for kind, level_suffix, change_suffix, flag in _FACTOR_SPECS:
                if not getattr(self.fx_factors, flag):
                    continue
                level_col = f"{pair}{level_suffix}"
                if level_col not in df.columns:
                    continue
                descriptors.append(
                    _Descriptor(column=f"{pair}{change_suffix}", pair=pair, kind=kind)
                )
        if not descriptors:
            raise MarketDataError(
                "No FX risk factors found. Provide level columns like "
                "'EURUSD_spot', 'EURUSD_vol', 'EURUSD_dom_rate', 'EURUSD_for_rate'."
            )
        return descriptors

    def _factor_changes(
        self, df: pd.DataFrame, descriptors: List[_Descriptor]
    ) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for d in descriptors:
            level_suffix = next(s for k, s, _, _ in _FACTOR_SPECS if k == d.kind)
            series = df[f"{d.pair}{level_suffix}"]
            change = series.pct_change() if d.kind == "spot" else series.diff()
            out[d.column] = change
        out = out.dropna()
        if len(out) < self.config.lookback_days:
            raise MarketDataError(
                f"Insufficient FX history: {len(out)} rows, "
                f"{self.config.lookback_days} required"
            )
        return out.tail(self.config.lookback_days)

    def _base_envs(self, portfolio: FXPortfolio) -> Dict[str, object]:
        return dict(portfolio.pricing_environments)

    def _sensitivities(
        self, portfolio: FXPortfolio, descriptors: List[_Descriptor]
    ) -> Dict[str, float]:
        """Central finite-difference dollar sensitivity per factor."""
        base_envs = self._base_envs(portfolio)
        sens: Dict[str, float] = {}
        for d in descriptors:
            eps = _EPS[d.kind]
            up = self._bump_one(base_envs, d, +eps)
            down = self._bump_one(base_envs, d, -eps)
            v_up = portfolio_value(portfolio, up)
            v_down = portfolio_value(portfolio, down)
            sens[d.column] = (v_up - v_down) / (2.0 * eps)
        return sens

    def _position_sensitivities(
        self, portfolio: FXPortfolio, descriptors: List[_Descriptor]
    ) -> Dict[str, Dict[str, float]]:
        base_envs = self._base_envs(portfolio)
        result: Dict[str, Dict[str, float]] = {}
        for pid, position in portfolio.positions.items():
            vec: Dict[str, float] = {}
            for d in descriptors:
                if d.pair != position.underlying:
                    vec[d.column] = 0.0
                    continue
                eps = _EPS[d.kind]
                up_env = self._bump_one(base_envs, d, +eps)[d.pair]
                down_env = self._bump_one(base_envs, d, -eps)[d.pair]
                vec[d.column] = (
                    position.get_market_value(up_env)
                    - position.get_market_value(down_env)
                ) / (2.0 * eps)
            result[pid] = vec
        return result

    @staticmethod
    def _bump_one(base_envs, d: _Descriptor, eps: float) -> Dict[str, object]:
        kwargs = {
            "spot": {"spot_return": eps},
            "vol": {"vol_change": eps},
            "dom": {"dom_shift": eps},
            "for": {"for_shift": eps},
        }[d.kind]
        envs = dict(base_envs)
        envs[d.pair] = bump_env(base_envs[d.pair], **kwargs)
        return envs

    def _scenario_pnls(
        self,
        portfolio: FXPortfolio,
        changes: pd.DataFrame,
        descriptors: List[_Descriptor],
    ) -> np.ndarray:
        """Full-revaluation P&L for each row of factor changes."""
        base_envs = self._base_envs(portfolio)
        base_value = portfolio_value(portfolio, base_envs)
        by_pair: Dict[str, List[_Descriptor]] = {}
        for d in descriptors:
            by_pair.setdefault(d.pair, []).append(d)

        pnls = np.empty(len(changes))
        for i, (_, row) in enumerate(changes.iterrows()):
            envs = dict(base_envs)
            for pair, ds in by_pair.items():
                kw = {"spot_return": 0.0, "vol_change": 0.0, "dom_shift": 0.0, "for_shift": 0.0}
                for d in ds:
                    kw[_KW[d.kind]] = float(row[d.column])
                envs[pair] = bump_env(base_envs[pair], **kw)
            pnls[i] = portfolio_value(portfolio, envs) - base_value
        return pnls

    # -- result assembly ------------------------------------------------ #
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
        if not isinstance(portfolio, FXPortfolio):
            raise ValidationError(
                f"FX VaR engines require an FXPortfolio, got {type(portfolio).__name__}"
            )
        if len(portfolio.positions) == 0:
            raise ValidationError("Cannot calculate VaR for empty portfolio")


_KW = {"spot": "spot_return", "vol": "vol_change", "dom": "dom_shift", "for": "for_shift"}


class FXParametricVaREngine(_BaseFXVaREngine):
    """Variance-covariance FX VaR: ``VaR = z * sqrt(sᵀ Σ s)``."""

    def calculate_var(self, portfolio: FXPortfolio, historical_data: pd.DataFrame) -> VaRResult:
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
            portfolio_value=pv,
            var_as_pct=abs(var) / pv if not is_zero(pv) else 0.0,
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
                comp[pid] = float(z * (s_pos @ sigma_s) / (sigma_pnl / self._scale()) * self._scale())
            result.component_var = comp

        return self._finish(result, start)


class FXHistoricalVaREngine(_BaseFXVaREngine):
    """Historical-simulation FX VaR via full revaluation of past factor moves."""

    def calculate_var(self, portfolio: FXPortfolio, historical_data: pd.DataFrame) -> VaRResult:
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


class FXMonteCarloVaREngine(_BaseFXVaREngine):
    """Monte-Carlo FX VaR: draw factor changes from N(mean, Σ), full revaluation."""

    def calculate_var(self, portfolio: FXPortfolio, historical_data: pd.DataFrame) -> VaRResult:
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


def _empirical_var_cvar(pnls: np.ndarray, confidence_level: float) -> Tuple[float, float]:
    """VaR/CVaR as positive losses from a P&L array."""
    losses = -pnls
    var = float(np.quantile(losses, confidence_level))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if tail.size else var
    var = max(var, 0.0)
    cvar = max(cvar, var)
    return var, cvar
