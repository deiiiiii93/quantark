"""Surface admission: the static-arbitrage gate and its machine-readable reasons.

A date that fails admission gets NO artifact and a manifest record naming the
reason.  Nothing is floored, filled or repaired -- the input is fixed upstream
or the date is excluded.
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np

from quantark.param import GridVolSurface
from quantark.param.div import TermStructureDividendYield
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.util.exceptions import NumericalError, QuantArkException
from quantark.util.numerical import fd1_nonuniform, fd2_nonuniform
from quantark.util.numerical.constants import Tolerance
from quantark.volmodels.localvol import build_dupire_local_vol


class AdmissionReason(str, Enum):
    """Stable, machine-readable admission-failure vocabulary."""

    MISSING_SPOT = "missing_spot"
    INVALID_SPOT = "invalid_spot"
    MISSING_SOURCE = "missing_source"
    PARSE_FAILED = "parse_failed"
    INVALID_SNAPSHOT = "invalid_snapshot"
    PARITY_GATING_FAILED = "parity_gating_failed"
    INSUFFICIENT_EXPIRIES = "insufficient_expiries"
    INSUFFICIENT_COMMON_STRIKES = "insufficient_common_strikes"
    DUPLICATE_MATURITY = "duplicate_maturity"
    NON_FINITE_CARRY = "non_finite_carry"
    SABR_SMOOTHING_FAILED = "sabr_smoothing_failed"
    STATIC_ARBITRAGE = "static_arbitrage"
    INVALID_ATM_PILLAR = "invalid_atm_pillar"
    PRICE_FIELD_MISMATCH = "price_field_mismatch"
    # Anything that escapes an AdmissionError.  Present so the vocabulary is
    # complete: the runner uses "is this reason in the enum?" to tell a record
    # the builder wrote from one written out-of-band by a study script, and a
    # missing member would misclassify the builder's own record as foreign.
    UNEXPECTED_ERROR = "unexpected_error"


class AdmissionError(QuantArkException):
    """One date failed admission, with a stable reason code and free-text detail."""

    def __init__(self, reason: AdmissionReason, detail: str) -> None:
        self.reason = AdmissionReason(reason)
        self.detail = str(detail)
        super().__init__(f"{self.reason.value}: {self.detail}")


def validate_static_arbitrage(surface: dict) -> str:
    """Run the suite's LV-input arbitrage validation on the smoothed grid.

    With >= 3 maturities this is exactly ``build_dupire_local_vol``'s default
    ``validate_arbitrage=True`` path (calendar dw/dT|_y >= 0 plus butterfly
    denominator > 0).  With exactly 2 maturities the Dupire builder refuses
    (it needs >= 3), so the same two checks are evaluated in reduced form with
    the same quantark finite differences and ``Tolerance`` thresholds; the
    maturity direction then uses the two-point one-sided stencil.  Returns the
    validation method label for the artifact's admission record.
    """

    s0 = float(surface["s0"])
    pe = surface["per_expiry"]
    ts = [float(p["T"]) for p in pe]
    rate_curve = LinearRateCurve([(float(p["T"]), float(p["r"])) for p in pe])
    div = TermStructureDividendYield(
        times=ts, yields=[float(p["q"]) for p in pe]
    )
    surf = GridVolSurface(
        surface["strikes"], surface["maturities"], np.array(surface["iv_grid"])
    )
    if len(ts) >= 3:
        build_dupire_local_vol(
            surf, spot=s0, rate_curve=rate_curve, div_yield=div.get_yield
        )
        return "build_dupire_local_vol(validate_arbitrage=True)"

    # Reduced-form replica of the Dupire checks for the 2-maturity edge case.
    K = np.asarray(surf.strikes, dtype=float)
    T = np.asarray(surf.maturities, dtype=float)
    iv = np.asarray(surf.iv_grid, dtype=float)
    ln_k = np.log(K)
    r_zero = np.array([rate_curve.get_rate(t) for t in T])
    q_zero = np.array([div.get_yield(t) for t in T])
    fwd = s0 * np.exp((r_zero - q_zero) * T)
    w = iv**2 * T[:, None]
    y = ln_k[None, :] - np.log(fwd)[:, None]
    if np.any(w <= 1e-12):
        raise NumericalError("degenerate total implied variance in 2-maturity grid")
    w_y = fd1_nonuniform(w, ln_k)
    w_yy = fd2_nonuniform(w, ln_k)
    dw_dT = (w[1] - w[0]) / (T[1] - T[0])  # only stencil available with 2 rows
    dlnF_dT = (math.log(fwd[1]) - math.log(fwd[0])) / (T[1] - T[0])
    dw_dT_y = dw_dT + dlnF_dT * w_y
    inv_w = 1.0 / w
    denom = (
        1.0
        - y * inv_w * w_y
        + 0.25 * (-0.25 - inv_w + (y * y) * inv_w * inv_w) * (w_y * w_y)
        + 0.5 * w_yy
    )
    nT, nK = iv.shape
    is_edge = np.zeros((nT, nK), dtype=bool)
    is_edge[0, :] = is_edge[-1, :] = True
    is_edge[:, 0] = is_edge[:, -1] = True
    cal_thresh = np.where(is_edge, 10.0 * Tolerance.PRECISION, Tolerance.PRECISION)
    if np.any(dw_dT_y < -cal_thresh):
        raise NumericalError(
            "calendar arbitrage: dw/dT|_y < 0 (moneyness) in 2-maturity grid"
        )
    bf_thresh = np.where(is_edge, 10.0 * Tolerance.PRECISION, 0.0)
    if np.any(denom < -bf_thresh):
        raise NumericalError(
            "butterfly arbitrage: Dupire denominator < 0 in 2-maturity grid"
        )
    # Dupire's post-check, same semantics as build_dupire_local_vol with
    # vol_floor=None: non-finite or non-positive local variance rejects.
    with np.errstate(divide="ignore", invalid="ignore"):
        lv2 = dw_dT_y / denom
    bad = ~np.isfinite(lv2) | (lv2 <= 0)
    if np.any(bad):
        idx = np.argwhere(bad)
        raise NumericalError(
            "Dupire produced non-positive/instable local variance at nodes "
            f"{idx.tolist()} in 2-maturity grid; the input surface is "
            "inadmissible — fix the input, do not floor"
        )
    return "reduced_form_dupire_checks_2_maturities"


