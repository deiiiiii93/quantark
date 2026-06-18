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
    # Term-structure mode (#3): per-(asset, step) forward rates/divs. When provided, the
    # drift over [t_j, t_{j+1}] is the exact integrated forward (rate - div), so the spot
    # distribution at every node matches a deterministic term-structure curve — required
    # for exact IR delta. ``rates``/``divs`` (the flat-drift path) are ignored then.
    step_rates: object = None      # shape (n_assets, n_steps) or None
    step_divs: object = None       # shape (n_assets, n_steps) or None

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
        self._term = self.step_rates is not None or self.step_divs is not None
        if self._term:
            n_steps = self.grid_times.size - 1
            for nm, arr in (("step_rates", self.step_rates), ("step_divs", self.step_divs)):
                if arr is None:
                    raise ValidationError(
                        "term-structure mode requires both step_rates and step_divs")
                a = np.asarray(arr, dtype=float)
                if a.shape != (n, n_steps):
                    raise ValidationError(
                        f"{nm} shape {a.shape} != (n_assets={n}, n_steps={n_steps})")
                if not np.all(np.isfinite(a)):
                    raise ValidationError(f"{nm} must be finite")

    def generate(self):
        t = self.grid_times
        dt = np.diff(t)
        if np.any(dt <= 0):
            raise ValidationError("grid_times must be strictly increasing")
        if self._term:
            return self._generate_term_structure(dt)
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

    def _generate_term_structure(self, dt):
        """Exact deterministic-term-structure GBM: per-step drift = forward(rate-div).

        S_{j+1} = S_j * exp((f_r[a,j] - f_q[a,j] - 0.5 vol_a^2) dt_j + vol_a sqrt(dt_j) Z),
        with Z correlated across assets via the Cholesky factor. Pseudo-random normals are
        seeded deterministically (common random numbers across base/bumped re-runs).
        """
        n = len(self.keys)
        n_steps = dt.shape[0]
        vols = np.asarray(self.vols, dtype=float)              # (n,)
        spots = np.asarray(self.spots, dtype=float)            # (n,)
        sr = np.asarray(self.step_rates, dtype=float)          # (n, n_steps)
        sd = np.asarray(self.step_divs, dtype=float)           # (n, n_steps)
        chol = np.asarray(
            CorrelationModel(keys=list(self.keys), matrix=self.corr).cholesky(), float)
        rng = np.random.default_rng(self.seed)
        z = rng.standard_normal((self.num_paths, n_steps, n))  # (p, step, asset)
        zc = z @ chol.T                                        # correlate across assets
        drift = (sr - sd).T - 0.5 * (vols ** 2)[None, :]       # (n_steps, n)
        log_inc = drift * dt[:, None] + zc * (vols[None, None, :]
                                              * np.sqrt(dt)[None, :, None])
        log_cum = np.cumsum(log_inc, axis=1)                   # (p, step, n)
        out = {}
        for a, k in enumerate(self.keys):
            path = np.empty((self.num_paths, n_steps + 1), dtype=float)
            path[:, 0] = spots[a]
            path[:, 1:] = spots[a] * np.exp(log_cum[:, :, a])
            out[k] = path
        return out
