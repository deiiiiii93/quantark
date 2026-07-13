"""Surface diagnostics and repricing residual reports (spec WP4.3).

Generic over surface/model: the no-arb report needs only
``total_variance(y, T)``; the residual report needs only a
``model_iv_fn(strike, expiry_t) -> iv`` — so the main model AND the
challenger run through the SAME report (problem §7.3: model names are not
calibration evidence).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from quantark.param.vol.marketquotes import black_price, black_vega
from quantark.util.exceptions import ValidationError

BUTTERFLY_TOL = -1e-8      # spec WP4.2 tolerance, reused for the report
CALENDAR_TOL = -1e-8
_DEFAULT_Y = np.arange(-1.5, 1.5 + 1e-12, 0.01)
_FD_H = 1e-3               # dense-grid finite-difference step for g(y)
_MONEYNESS_EDGES = (-0.1, 0.1)   # bucket edges: puts / atm / calls


@dataclass(frozen=True)
class NoArbReport:
    butterfly_min_g: Dict[float, float]
    calendar_min_dw: Dict[Tuple[float, float], float]
    passed: bool
    tolerances: dict

    def to_dict(self) -> dict:
        return {
            "butterfly_min_g": {
                f"{t:g}": v for t, v in self.butterfly_min_g.items()
            },
            "calendar_min_dw": {
                f"{a:g}->{b:g}": v
                for (a, b), v in self.calendar_min_dw.items()
            },
            "passed": self.passed,
            "tolerances": dict(self.tolerances),
        }


def _numerical_g(w_fn: Callable[[np.ndarray], np.ndarray],
                 y: np.ndarray) -> np.ndarray:
    """Gatheral g(y) via 5-point-free central differences of w(y)."""
    w = w_fn(y)
    w_up = w_fn(y + _FD_H)
    w_dn = w_fn(y - _FD_H)
    w1 = (w_up - w_dn) / (2.0 * _FD_H)
    w2 = (w_up - 2.0 * w + w_dn) / (_FD_H * _FD_H)
    term = 1.0 - y * w1 / (2.0 * w)
    return term * term - (w1 * w1 / 4.0) * (1.0 / w + 0.25) + w2 / 2.0


def static_no_arb_report(
    surface,
    expiry_ts: Sequence[float],
    y_grid: Optional[np.ndarray] = None,
    butterfly_tol: float = BUTTERFLY_TOL,
    calendar_tol: float = CALENDAR_TOL,
) -> NoArbReport:
    """Butterfly g(y) minima per slice + calendar Δw minima per adjacent
    pair on a dense grid, with pass/fail vs the tolerances."""
    ts = sorted(float(t) for t in expiry_ts)
    if len(ts) == 0:
        raise ValidationError("expiry_ts must be non-empty")
    y = _DEFAULT_Y if y_grid is None else np.asarray(y_grid, dtype=float)

    butterfly: Dict[float, float] = {}
    for t in ts:
        butterfly[t] = float(
            np.min(_numerical_g(lambda yy: np.asarray(
                surface.total_variance(yy, t), dtype=float), y))
        )
    calendar: Dict[Tuple[float, float], float] = {}
    for t1, t2 in zip(ts, ts[1:]):
        dw = np.asarray(surface.total_variance(y, t2), dtype=float) - \
            np.asarray(surface.total_variance(y, t1), dtype=float)
        calendar[(t1, t2)] = float(dw.min())

    passed = all(v >= butterfly_tol for v in butterfly.values()) and all(
        v >= calendar_tol for v in calendar.values()
    )
    return NoArbReport(
        butterfly_min_g=butterfly,
        calendar_min_dw=calendar,
        passed=passed,
        tolerances={
            "butterfly_tol": butterfly_tol,
            "calendar_tol": calendar_tol,
        },
    )


@dataclass(frozen=True)
class RepricingResidualReport:
    rows: Tuple[dict, ...]
    rmse_iv: float
    max_abs_iv: float
    by_bucket: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "rows": list(self.rows),
            "rmse_iv": self.rmse_iv,
            "max_abs_iv": self.max_abs_iv,
            "by_bucket": dict(self.by_bucket),
        }


def _bucket_label(expiry_t: float, y: float) -> str:
    lo, hi = _MONEYNESS_EDGES
    side = "put_wing" if y < lo else ("call_wing" if y > hi else "atm")
    return f"T={expiry_t:g}|{side}"


def repricing_residual_report(
    cleaned,
    model_iv_fn: Callable[[float, float], float],
) -> RepricingResidualReport:
    """Reprice the CLEANED calibration universe under a model and report
    IV- and price-space residuals as rows + summary stats (spec WP4.3)."""
    rows = []
    bucket_sq: Dict[str, list] = {}
    for expiry_t in sorted(cleaned.slices):
        df = cleaned.dfs[expiry_t]
        fwd = cleaned.forwards[expiry_t]
        for q in cleaned.slices[expiry_t]:
            model_iv = float(model_iv_fn(q.strike, expiry_t))
            iv_error = model_iv - q.iv
            is_call = q.strike >= fwd
            price_error = black_price(
                fwd, q.strike, expiry_t, model_iv, df, is_call
            ) - black_price(fwd, q.strike, expiry_t, q.iv, df, is_call)
            rows.append({
                "expiry_t": expiry_t,
                "strike": q.strike,
                "log_moneyness": q.log_moneyness,
                "market_iv": q.iv,
                "model_iv": model_iv,
                "iv_error": iv_error,
                "price_error": price_error,
                "vega": black_vega(fwd, q.strike, expiry_t, q.iv, df),
            })
            bucket_sq.setdefault(
                _bucket_label(expiry_t, q.log_moneyness), []
            ).append(iv_error * iv_error)
    if not rows:
        raise ValidationError("cleaned quote set is empty")
    errs = np.array([r["iv_error"] for r in rows])
    return RepricingResidualReport(
        rows=tuple(rows),
        rmse_iv=float(np.sqrt(np.mean(errs * errs))),
        max_abs_iv=float(np.max(np.abs(errs))),
        by_bucket={
            k: float(np.sqrt(np.mean(v))) for k, v in bucket_sq.items()
        },
    )
