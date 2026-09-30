"""What ``prepare`` sampled from the environment (spec 2 §5.2 step 8, §5.8).

The route's model owns the SAMPLING: the Black-Scholes routes sample forward
rates, carries and a scalar vol per step (``SCALAR``); the local-vol routes add
a vol vector per step at the interval midpoint (``NODAL``). The operator
factory owns the stencil built from these samples.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional

import numpy as np

from quantark.util.exceptions import PricingError, ValidationError

__all__ = ["SampleKind", "MarketSamples", "sample_bsm", "sample_local_vol"]


class SampleKind(Enum):
    SCALAR = "scalar"
    NODAL = "nodal"


@dataclass(frozen=True, eq=False)
class MarketSamples:
    """Forward coefficients per backward step ``j`` (the interval ``[t[j], t[j+1]]``).

    ``r_tau`` / ``q_tau`` / ``sigma_tau`` are the scalar-stencil inputs at
    maturity (the flat-exact replacement); ``None`` where a caller has none.
    """

    kind: SampleKind
    t: np.ndarray
    fwd_rates: np.ndarray
    fwd_carry: np.ndarray
    ref_strike: float
    step_vols: Optional[np.ndarray] = None        # SCALAR
    vol_rows: Optional[np.ndarray] = None         # NODAL, (n_steps, num_x)
    lv_surface: Optional[object] = None           # NODAL
    r_tau: Optional[float] = None
    q_tau: Optional[float] = None
    sigma_tau: Optional[float] = None
    provenance: Mapping[str, Optional[str]] = field(default_factory=dict)

    def first_passage(self, product, barrier, pricing_env):
        """Per-step ``(mu, sigma^2)`` governing the continuous-KI crossing.

        SCALAR: the term vol at the strike, from the same forward coefficients
        the operator uses (``SnowballPDESolver._first_passage_step_coefficients``).
        Reused only when these samples were taken at the product's strike;
        otherwise sampled again at it, as the legacy method did.
        NODAL: the local vol AT THE BARRIER at each step's midpoint
        (``LocalVolBarrierCrossingMixin._first_passage_step_coefficients``).
        """
        from quantark.priceenv.term_sampling import TermCoefficients

        if self.kind is SampleKind.SCALAR:
            if float(product.strike) == self.ref_strike:
                fwd_rates, fwd_carry, step_vols = self.fwd_rates, self.fwd_carry, self.step_vols
            else:
                tc = TermCoefficients.from_env(
                    pricing_env, self.t, ref_strike=float(product.strike)
                )
                fwd_rates, fwd_carry, step_vols = tc.fwd_rates, tc.fwd_carry, tc.step_vols
            sig2 = step_vols * step_vols
            return fwd_rates - fwd_carry - 0.5 * sig2, sig2

        surface = self.lv_surface
        if surface is None:
            raise PricingError("Local-vol surface is not initialized for this solve")
        t = np.asarray(self.t, dtype=float)
        if float(product.strike) == self.ref_strike:
            fwd_rates, fwd_carry = self.fwd_rates, self.fwd_carry
        else:
            tc = TermCoefficients.from_env(pricing_env, t, ref_strike=float(product.strike))
            fwd_rates, fwd_carry = tc.fwd_rates, tc.fwd_carry
        t_mid = 0.5 * (t[:-1] + t[1:])
        sigma = np.asarray(
            surface.local_vol(np.full(t_mid.shape, float(barrier)), t_mid),
            dtype=float,
        )
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValidationError(
                "local-vol surface returned non-positive or non-finite vols "
                "at the KI barrier"
            )
        sig2 = sigma * sigma
        return fwd_rates - fwd_carry - 0.5 * sig2, sig2


def sample_bsm(env, t_vec, *, ref_strike, r_tau=None, q_tau=None, sigma_tau=None,
               provenance=None) -> MarketSamples:
    """Forward rates, carries and a scalar vol per step (``_build_step_coefficients``)."""
    from quantark.priceenv.term_sampling import TermCoefficients

    t = np.asarray(t_vec, dtype=float)
    tc = TermCoefficients.from_env(env, t, ref_strike=float(ref_strike))
    if provenance is None:
        from .identity import market_fingerprints
        provenance = market_fingerprints(env)
    return MarketSamples(
        kind=SampleKind.SCALAR, t=t, fwd_rates=tc.fwd_rates, fwd_carry=tc.fwd_carry,
        ref_strike=float(ref_strike), step_vols=tc.step_vols,
        r_tau=r_tau, q_tau=q_tau, sigma_tau=sigma_tau, provenance=provenance,
    )


def _validated_local_vol_row(sigma_vec, num_x):
    # snowball_vol_pde_solvers.py (_local_vol_coefficients validation), verbatim
    sigma_vec = np.asarray(sigma_vec, dtype=float)
    if sigma_vec.shape[0] != num_x:
        raise ValidationError("local-vol vector length must match PDE spatial grid")
    if not np.all(np.isfinite(sigma_vec)) or np.any(sigma_vec <= 0.0):
        raise ValidationError("local-vol surface returned non-positive or non-finite vols")
    return sigma_vec


def sample_local_vol(env, t_vec, s_vec, surface, *, ref_strike, r_tau=None, q_tau=None,
                     sigma_tau=None) -> MarketSamples:
    """One local-vol row per step at the interval midpoint (the LV ``_build_step_coefficients``)."""
    from quantark.execution.cache.fingerprint import try_fingerprint
    from quantark.priceenv.term_sampling import TermCoefficients

    from .identity import market_fingerprints

    if surface is None or s_vec is None:
        raise PricingError("Local-vol surface is not initialized for this solve")
    t = np.asarray(t_vec, dtype=float)
    tc = TermCoefficients.from_env(env, t, ref_strike=float(ref_strike))
    s_vec = np.asarray(s_vec, dtype=float)
    num_x = len(s_vec)
    dt = np.diff(t)
    rows = []
    for j, dt_j in enumerate(dt):
        t_mid = float(t_vec[j] + 0.5 * dt_j)
        rows.append(_validated_local_vol_row(surface.local_vol(s_vec, t_mid), num_x))
    vol_rows = np.array(rows) if rows else np.empty((0, num_x))
    return MarketSamples(
        kind=SampleKind.NODAL, t=t, fwd_rates=tc.fwd_rates, fwd_carry=tc.fwd_carry,
        ref_strike=float(ref_strike), vol_rows=vol_rows, lv_surface=surface,
        r_tau=r_tau, q_tau=q_tau, sigma_tau=sigma_tau,
        provenance={**market_fingerprints(env), "lv_surface": try_fingerprint(surface)},
    )
