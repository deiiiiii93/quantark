"""HistoricalExposureEngine — non-regulatory, real-world EE/PFE.

Wires calibrate -> generate -> reprice (provisional scaffold) -> PFE assembly
into an ``ExposureProfile(measure=REAL_WORLD, regulatory_eligible=False)``. The
single most important property: this engine NEVER produces a regulatory-eligible
profile, so it cannot feed the SA-CVA capital path (MAR50.34(1)).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from quantark.util.exceptions import ValidationError
from quantark.sacva.exposure._contract_provisional import (
    ExposureEngine, ExposureProfile, Measure, CONTRACT_VERSION,
)
from quantark.sacva.exposure.historical.calibration import DriftMode
from quantark.sacva.exposure.historical.path_generator import HistoricalPathGenerator, PathMode
from quantark.sacva.exposure.historical.resampling import ResamplingScheme
from quantark.sacva.exposure.historical.pfe import PFEProfileAssembler


@dataclass
class HistoricalExposureConfig:
    path_mode: str
    grid_times: tuple
    factor_keys: tuple
    today_levels: dict
    drift_modes: dict
    scheme: Optional[str] = None
    block_length: Optional[int] = None
    expected_block_length: Optional[float] = None
    n_paths: Optional[int] = None
    seed: int = 0
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
        for k in self.factor_keys:
            if k not in self.drift_modes:
                raise ValidationError(f"missing drift mode for factor {k}")
            if k not in self.today_levels:
                raise ValidationError(f"missing today level for factor {k}")
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

    def _check_scope(self, trade):
        if trade.requires_continuous_barrier:
            raise ValidationError(
                f"trade {trade.trade_id}: continuous barrier not supported (no GBM bridge)")
        if trade.requires_fx_conversion or trade.foreign_underlying:
            raise ValidationError(
                f"trade {trade.trade_id}: FX-conversion/foreign-underlying out of scope")
        if trade.n_state_factors != 1:
            raise ValidationError(f"trade {trade.trade_id}: only single-factor trades supported")

    def _validate(self, counterparty, cfg):
        if not counterparty.netting_sets:
            raise ValidationError("counterparty has no netting sets")
        missing = set(cfg.factor_keys) - set(self.calibration._r.columns)
        if missing:
            raise ValidationError(f"factors not in calibration: {missing}")
        for k in cfg.factor_keys:                       # reconcile today level to data
            self.calibration.data.reconcile_today(k, cfg.today_levels[k])
        col = set(cfg.factor_keys)
        for ns in counterparty.netting_sets:            # scope-check BEFORE path gen
            if not ns.trades:
                raise ValidationError(f"netting set {ns.set_id} has no trades")
            for tr in ns.trades:
                self._check_scope(tr)
                if tr.factor_key not in col:
                    raise ValidationError(
                        f"trade {tr.trade_id}: factor {tr.factor_key} not simulated")
                if not hasattr(tr.surface, "value_at"):
                    raise ValidationError(f"trade {tr.trade_id}: surface has no value_at")
                if not np.isfinite(tr.quantity):
                    raise ValidationError(f"trade {tr.trade_id}: non-finite quantity")

    def compute(self, counterparty):
        cfg = self.config
        self._validate(counterparty, cfg)
        grid = np.array(cfg.grid_times, float)
        modes = {k: DriftMode[v] for k, v in cfg.drift_modes.items()}
        gen = HistoricalPathGenerator(self.calibration, tuple(cfg.factor_keys),
                                      dict(cfg.today_levels), lam=cfg.lam)
        kw = {}
        if cfg.path_mode == "BOOTSTRAP":
            kw = dict(scheme=ResamplingScheme[cfg.scheme], n_paths=cfg.n_paths, seed=cfg.seed,
                      block_length=cfg.block_length,
                      expected_block_length=cfg.expected_block_length)
        states = gen.generate(PathMode[cfg.path_mode], grid, drift_modes=modes, **kw)
        col = {k: i for i, k in enumerate(cfg.factor_keys)}

        total = None
        for ns in counterparty.netting_sets:
            trade_vals = []
            for tr in ns.trades:                        # scope already validated in _validate
                S = states[:, :, col[tr.factor_key]]
                vals = np.empty_like(S)
                for j in range(S.shape[1]):
                    vals[:, j] = tr.surface.value_at(S[:, j], grid[j], None) * tr.quantity
                trade_vals.append(vals)
            stacked = np.stack(trade_vals, axis=0)
            netted = (np.maximum(stacked.sum(axis=0), 0.0) if ns.netting_enforceable
                      else np.maximum(stacked, 0.0).sum(axis=0))
            total = netted if total is None else total + netted

        asm = PFEProfileAssembler(confidences_bps=tuple(cfg.confidences_bps),
                                  quantile_method=cfg.quantile_method, m_tail_min=cfg.m_tail_min)
        out = asm.assemble(total, grid)
        meta = {"contract_version": CONTRACT_VERSION, "path_mode": cfg.path_mode,
                "scheme": cfg.scheme, "block_length": cfg.block_length,
                "expected_block_length": cfg.expected_block_length, "seed": cfg.seed,
                "lam": cfg.lam, "drift_modes": dict(cfg.drift_modes), "n_paths": cfg.n_paths,
                "quantile_method": cfg.quantile_method, "non_production": cfg._demo}
        return ExposureProfile(times=grid, epe_discounted=None, measure=Measure.REAL_WORLD,
                               regulatory_eligible=False, ee_undiscounted=out["ee_undiscounted"],
                               pfe=out["pfe"], epe=out["epe"], metadata=meta)
