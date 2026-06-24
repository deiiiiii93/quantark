import numpy as np
import pytest

from quantark.asset.equity.process.sabr import SABRProcess
from quantark.util.enum.engine_enums import SABRMCScheme


def test_martingale_forward_is_preserved():
    # E[F_T] ~= F0 for the driftless SABR forward (log-Euler).
    proc = SABRProcess(f0=100.0, alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
    fT = proc.simulate(T=1.0, n_paths=200_000, n_steps=50, seed=7)
    assert abs(fT.mean() - 100.0) / 100.0 < 0.01


def test_zero_volofvol_reduces_to_lognormal_atm():
    # nu=0, beta=1 => exact GBM; E[F_T]=F0 and dispersion ~ alpha.
    proc = SABRProcess(f0=100.0, alpha=0.2, beta=1.0, rho=0.0, nu=0.0)
    fT = proc.simulate(T=1.0, n_paths=200_000, n_steps=1, seed=3)
    assert abs(np.log(fT / 100.0).std() - 0.2) < 0.01


def test_paths_are_nonnegative():
    proc = SABRProcess(f0=100.0, alpha=0.3, beta=0.5, rho=-0.5, nu=0.8)
    fT = proc.simulate(T=2.0, n_paths=50_000, n_steps=100, seed=1)
    assert np.all(fT >= 0.0)


def test_quadexp_beta1_is_martingale_at_coarse_grid():
    # The whole point of QE: stay accurate with very few steps. beta=1 is exact
    # conditional lognormal, so the forward mean is preserved even at n_steps=4.
    proc = SABRProcess(f0=100.0, alpha=0.2, beta=1.0, rho=-0.4, nu=0.6)
    fT = proc.simulate(T=1.0, n_paths=200_000, n_steps=4, seed=9,
                       scheme=SABRMCScheme.QUADEXP)
    assert abs(fT.mean() - 100.0) / 100.0 < 0.01


def test_quadexp_rejects_non_unit_beta():
    proc = SABRProcess(f0=100.0, alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)
    with pytest.raises(NotImplementedError):
        proc.simulate(T=1.0, n_paths=1000, n_steps=10, scheme=SABRMCScheme.QUADEXP)
