"""HistoricalExposureEngine — non-regulatory, real-world EE/PFE.

Re-uses the canonical exposure machinery (``AnalyticValueSurface`` repricing,
``ExposureGrid``, the as-of roll-down) but swaps the **path generation**: instead
of risk-neutral GBM, state paths come from historical replay/bootstrap
(``HistoricalPathGenerator``), and the output is undiscounted EE + PFE quantiles
rather than a discounted EPE profile.

The single most important property: this engine NEVER produces a regulatory-eligible
profile (``measure=REAL_WORLD``, ``regulatory_eligible=False``, ``epe_discounted=None``),
so it cannot feed the SA-CVA capital path (MAR50.34(1)).

v1 scope mirrors the MC engine's vanilla path: equity / reporting-vs-foreign FX
spot underlyings priced via the analytic value surface. Stateful (snowball) trades
are deferred and raise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.sacva.exposure.engine import ExposureEngine, ExposureProfile, Measure
from quantark.sacva.exposure.grid import ExposureGrid
from quantark.sacva.exposure.value_surface import AnalyticValueSurface
from quantark.sacva.exposure.asof import equity_asof_env
from quantark.sacva.exposure.historical.calibration import DriftMode
from quantark.sacva.exposure.historical.path_generator import HistoricalPathGenerator, PathMode
from quantark.sacva.exposure.historical.resampling import ResamplingScheme
from quantark.sacva.exposure.historical.pfe import PFEProfileAssembler

# tau below this (~1 calendar day) is the terminal node: value by contractual payoff,
# sidestepping the engine's exercise-date guard (mirrors the MC simulator).
_TAU_FLOOR = 1.0 / 365.0


@dataclass
class HistoricalExposureConfig:
    path_mode: str
    drift_modes: dict                       # {underlying_key: DriftMode name}
    scheme: Optional[str] = None
    block_length: Optional[int] = None
    expected_block_length: Optional[float] = None
    n_paths: Optional[int] = None
    seed: int = 0
    n_steps: int = 24
    confidences_bps: tuple = (9500, 9900)
    quantile_method: str = "linear"
    lam: float = 0.94
    m_tail_min: int = 10
    _demo: bool = False

    def __post_init__(self):
        if self.path_mode not in PathMode.__members__:
            raise ValidationError(f"path_mode must be one of {list(PathMode.__members__)}")
        for k, v in self.drift_modes.items():
            if v not in DriftMode.__members__:
                raise ValidationError(f"unknown drift mode {v} for {k}")
        if self.path_mode == "BOOTSTRAP":
            if self.scheme is None or self.scheme not in ResamplingScheme.__members__:
                raise ValidationError("BOOTSTRAP requires a valid scheme (no default)")
            if self.n_paths is None or self.n_paths <= 0:
                raise ValidationError("BOOTSTRAP requires n_paths > 0")
            if self.scheme == "BLOCK_FHS" and self.block_length is None:
                raise ValidationError("BLOCK_FHS requires explicit block_length")
            if self.scheme == "STATIONARY_BLOCK" and self.expected_block_length is None:
                raise ValidationError("STATIONARY_BLOCK requires explicit expected_block_length")

    @classmethod
    def default_block_fhs_for_demo(cls, **kw):
        """NON-PRODUCTION convenience; recorded as non-production in metadata."""
        kw.setdefault("path_mode", "BOOTSTRAP")
        kw.setdefault("scheme", "BLOCK_FHS")
        kw.setdefault("block_length", 10)
        kw.setdefault("n_paths", 5000)
        return cls(_demo=True, **kw)


@dataclass
class HistoricalExposureEngine(ExposureEngine):
    calibration: object
    config: HistoricalExposureConfig

    # -- risk-factor extraction (mirrors the MC engine) -------------------------
    def _underlying_key(self, trade):
        sq = getattr(trade.env, "spot_quote", None)
        key = getattr(sq, "asset_name", None) if sq is not None else None
        if not key:
            raise ValidationError(
                f"{trade.trade_id}: env.spot_quote.asset_name is required to identify "
                "the underlying risk factor")
        return key

    def _trade_maturity(self, trade):
        T = float(trade.product.get_maturity(trade.env))
        if not (T > 0):
            raise ValidationError(f"{trade.trade_id}: non-positive maturity {T}")
        return T

    def compute(self, counterparty) -> ExposureProfile:
        cfg = self.config
        trades = [t for ns in counterparty.netting_sets for t in ns.trades]
        if not trades:
            raise ValidationError(f"{counterparty.name}: no trades")
        for t in trades:
            if getattr(t.engine, "supports_spot_greeks_grid", False):
                raise ValidationError(
                    f"{t.trade_id}: stateful (snowball) trades are deferred in the "
                    "historical engine v1; only vanilla analytic-surface trades are supported")

        # per-underlying spot/key, with market consistency (one spot per underlying)
        keys, today_levels = [], {}
        for t in trades:
            k = self._underlying_key(t)
            spot = float(t.env.spot)
            if k in today_levels:
                if abs(today_levels[k] - spot) > 1e-9:
                    raise ValidationError(f"inconsistent spot for underlying {k} across trades")
            else:
                today_levels[k] = spot
                keys.append(k)

        # factors must exist in the calibration; reconcile today level to history
        missing = set(keys) - set(self.calibration._r.columns)
        if missing:
            raise ValidationError(f"factors not in calibration: {missing}")
        for k in keys:
            if k not in cfg.drift_modes:
                raise ValidationError(f"missing drift mode for factor {k}")
            self.calibration.data.reconcile_today(k, today_levels[k])

        # exposure grid: uniform to the longest maturity (mirrors MC _compute_grid)
        horizon = max(self._trade_maturity(t) for t in trades)
        times = ExposureGrid.build(horizon=horizon, n_steps=cfg.n_steps, event_times=[]).times

        modes = {k: DriftMode[cfg.drift_modes[k]] for k in keys}
        gen = HistoricalPathGenerator(self.calibration, tuple(keys),
                                      {k: today_levels[k] for k in keys}, lam=cfg.lam)
        kw = {}
        if cfg.path_mode == "BOOTSTRAP":
            kw = dict(scheme=ResamplingScheme[cfg.scheme], n_paths=cfg.n_paths, seed=cfg.seed,
                      block_length=cfg.block_length,
                      expected_block_length=cfg.expected_block_length)
        states = gen.generate(PathMode[cfg.path_mode], times, drift_modes=modes, **kw)
        paths = {k: states[:, :, i] for i, k in enumerate(keys)}

        # pathwise undiscounted reporting-currency values per trade, netted per set
        values = {id(t): self._trade_value_array(t, paths, times) for t in trades}
        total = None
        for ns in counterparty.netting_sets:
            arrays = [values[id(t)] for t in ns.trades]
            stacked = np.stack(arrays, axis=0)
            netted = (np.maximum(stacked.sum(axis=0), 0.0) if ns.netting_enforceable
                      else np.maximum(stacked, 0.0).sum(axis=0))
            total = netted if total is None else total + netted

        asm = PFEProfileAssembler(confidences_bps=tuple(cfg.confidences_bps),
                                  quantile_method=cfg.quantile_method, m_tail_min=cfg.m_tail_min)
        out = asm.assemble(total, times)
        meta = {"path_mode": cfg.path_mode, "scheme": cfg.scheme,
                "block_length": cfg.block_length,
                "expected_block_length": cfg.expected_block_length, "seed": cfg.seed,
                "lam": cfg.lam, "drift_modes": dict(cfg.drift_modes), "n_paths": cfg.n_paths,
                "n_steps": cfg.n_steps, "quantile_method": cfg.quantile_method,
                "non_production": cfg._demo}
        return ExposureProfile(times=times, epe_discounted=None, measure=Measure.REAL_WORLD,
                               regulatory_eligible=False, ee_undiscounted=out["ee_undiscounted"],
                               pfe=out["pfe"], epe=out["epe"], metadata=meta)

    def _trade_value_array(self, trade, paths, times):
        """Pathwise UNDISCOUNTED reporting-currency value, shape (num_paths, n_t).
        Identical repricing to the MC engine (analytic value surface + terminal payoff)."""
        key = self._underlying_key(trade)
        spots = paths[key]
        T = self._trade_maturity(trade)
        surface = AnalyticValueSurface(
            engine=trade.engine, product=trade.product, base_env=trade.env,
            as_of_env=equity_asof_env, currency=trade.trade_currency)
        out = np.zeros_like(spots)
        for j, tj in enumerate(times):
            tau = T - float(tj)
            col = spots[:, j]
            if tau >= _TAU_FLOOR:
                out[:, j] = surface.value_at(col, float(tj), None)
            elif tau >= 0.0:
                if not hasattr(trade.product, "get_payoff"):
                    raise ValidationError(
                        f"{trade.trade_id}: product lacks get_payoff for terminal node")
                out[:, j] = np.array([trade.product.get_payoff(float(s)) for s in col])
        return out * float(trade.quantity)
