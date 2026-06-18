"""Risk-neutral spot path generator (spec §3.2).

Wraps the existing vectorized ``MultiAssetGBMPathGenerator``. v1 simulates spot
factors only (equity / reporting-vs-foreign FX) under deterministic rates. FX is
a GBM with drift ``r_dom - r_for`` (set ``rates=r_domestic``, ``divs=r_foreign``);
equity uses ``rates=r``, ``divs=q``. Seeded via ``PseudoRandomNormalGenerator``
(wraps ``np.random.default_rng(seed)``) for determinism / common random numbers.
"""

from dataclasses import dataclass
from typing import List

import numpy as np

from quantark.asset.equity.process.bsm.qmc_path_generator import (
    MultiAssetGBMPathGenerator,
)
from quantark.asset.equity.process.bsm.qmc_sobol import PseudoRandomNormalGenerator
from quantark.sacva.exposure.correlation import CorrelationModel
from quantark.util.exceptions import ValidationError


@dataclass
class StatePathGenerator:
    keys: List[str]
    spots: List[float]
    vols: List[float]
    rates: List[float]
    divs: List[float]
    corr: object
    grid_times: object
    num_paths: int = 10000
    seed: int = 12345

    def __post_init__(self) -> None:
        self.grid_times = np.asarray(self.grid_times, dtype=float)
        n = len(self.keys)
        if n == 0:
            raise ValidationError("keys must be non-empty")
        if len(set(self.keys)) != n:
            raise ValidationError("keys must be unique (duplicates collide in output)")
        for name, arr in (("spots", self.spots), ("vols", self.vols),
                          ("rates", self.rates), ("divs", self.divs)):
            a = np.asarray(arr, dtype=float)
            if len(arr) != n:
                raise ValidationError(f"{name} length {len(arr)} != keys {n}")
            if not np.all(np.isfinite(a)):
                raise ValidationError(f"{name} must be finite")
        if np.any(np.asarray(self.spots, dtype=float) <= 0):
            raise ValidationError("spots must be positive")
        if np.any(np.asarray(self.vols, dtype=float) < 0):
            raise ValidationError("vols must be non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValidationError("seed must be an int (deterministic CRN)")
        # full correlation validation (shape/symmetry/unit-diagonal/bounds/PD)
        CorrelationModel(keys=list(self.keys), matrix=self.corr).cholesky()
        if isinstance(self.num_paths, bool) or not isinstance(self.num_paths, int):
            raise ValidationError("num_paths must be an int")
        if self.num_paths < 1:
            raise ValidationError("num_paths must be >= 1")
        if self.grid_times.ndim != 1 or self.grid_times.size == 0:
            raise ValidationError("grid_times must be a non-empty 1-D array")
        if not np.all(np.isfinite(self.grid_times)):
            raise ValidationError("grid_times must be finite")
        if self.grid_times[0] != 0.0:
            raise ValidationError("grid_times must start at 0.0")

    def generate(self):
        t = self.grid_times
        dt = np.diff(t)
        if np.any(dt <= 0):
            raise ValidationError("grid_times must be strictly increasing")
        gen = MultiAssetGBMPathGenerator(
            initial_values=np.asarray(self.spots, dtype=float),
            vols=np.asarray(self.vols, dtype=float),
            rrfs=np.asarray(self.rates, dtype=float),
            divs=np.asarray(self.divs, dtype=float),
            correlation_matrix=np.asarray(self.corr, dtype=float),
            maturity=float(t[-1]), time_steps=len(dt), num_paths=self.num_paths,
            model="bsm", dt_array=dt,  # bsm drift = rrfs - divs (risk-neutral r-q)
            random_stream=PseudoRandomNormalGenerator(seed=self.seed),
        )
        paths = gen.generate_paths()  # return_aux defaults False -> ndarray
        if isinstance(paths, tuple):
            paths = paths[0]
        expected = (len(self.keys), self.num_paths, len(t))
        if paths.shape != expected:
            raise ValidationError(
                f"unexpected path shape {paths.shape}; expected {expected}")
        return {k: paths[i] for i, k in enumerate(self.keys)}
