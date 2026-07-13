"""One-call Heston calibration from a cleaned quote set (spec WP4.6).

Wraps the existing ``calibrate_heston`` (IV target) with stated defaults and
emits the WP4.3 repricing residual report, so the challenger (or main model)
gets a reproducible setup with recorded initial values, bounds, weights, and
stopping criteria.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston.analytical_kernel import heston_implied_vol
from quantark.volmodels.heston.calibration import (
    CalibrationResult,
    MarketOption,
    calibrate_heston,
)
from quantark.volmodels.heston.params import HestonParams

_DEFAULT_BOUNDS = (
    (1e-8, 1e-6, 1e-6, 1e-6, -0.999),
    (5.0, 50.0, 5.0, 5.0, 0.999),
)
_DEFAULT_KAPPA = 1.5
_DEFAULT_VOLVOL = 0.5
_DEFAULT_RHO = -0.5


@dataclass(frozen=True)
class HestonQuoteCalibration:
    params: HestonParams
    result: CalibrationResult
    residual_report: object            # RepricingResidualReport
    config: dict

    def to_dict(self) -> dict:
        return {
            "params": {
                "v0": self.params.v0, "kappa": self.params.kappa,
                "theta": self.params.theta, "sigma": self.params.sigma,
                "rho": self.params.rho,
            },
            "success": self.result.success,
            "cost": self.result.cost,
            "nfev": self.result.nfev,
            "residual_report": self.residual_report.to_dict(),
            "config": dict(self.config),
        }


def _atm_iv(cleaned) -> float:
    """IV of the quote nearest the forward on the shortest expiry."""
    t0 = min(cleaned.slices)
    quotes = cleaned.slices[t0]
    return min(quotes, key=lambda q: abs(q.log_moneyness)).iv


def calibrate_heston_from_quotes(
    cleaned,
    rate_curve,
    carry_curve,
    config: Optional[dict] = None,
) -> HestonQuoteCalibration:
    from quantark.volmodels.diagnostics import repricing_residual_report

    if not cleaned.slices:
        raise ValidationError("cleaned quote set is empty")
    cfg = dict(config or {})
    iv_atm = _atm_iv(cleaned)
    initial = cfg.pop("initial", None) or HestonParams(
        v0=iv_atm * iv_atm,
        kappa=_DEFAULT_KAPPA,
        theta=iv_atm * iv_atm,
        sigma=_DEFAULT_VOLVOL,
        rho=_DEFAULT_RHO,
    )
    bounds = cfg.pop("bounds", _DEFAULT_BOUNDS)
    solver = dict(max_nfev=200, xtol=1e-6, ftol=1e-6, gtol=1e-6)
    solver.update(cfg)

    def r_fn(t: float) -> float:
        return float(rate_curve.get_rate(t))

    def carry_fn(t: float) -> float:
        # q(T) = r(T) - B(T)/T  (cumulative-carry identity)
        return float(rate_curve.get_rate(t) - carry_curve.carry(t) / t)

    options = [
        MarketOption(K=q.strike, T=q.expiry_t, iv=q.iv, weight=q.weight)
        for t in sorted(cleaned.slices)
        for q in cleaned.slices[t]
    ]
    result = calibrate_heston(
        s0=float(cleaned.spot),
        options=options,
        r=r_fn,
        carry=carry_fn,
        initial=initial,
        bounds=bounds,
        target="iv",
        **solver,
    )

    def model_iv_fn(strike: float, expiry_t: float) -> float:
        return float(
            heston_implied_vol(
                float(cleaned.spot), float(strike), float(expiry_t),
                result.params, r_fn(expiry_t), carry_fn(expiry_t),
            )
        )

    report = repricing_residual_report(cleaned, model_iv_fn)
    return HestonQuoteCalibration(
        params=result.params,
        result=result,
        residual_report=report,
        config={
            "initial": {
                "v0": initial.v0, "kappa": initial.kappa,
                "theta": initial.theta, "sigma": initial.sigma,
                "rho": initial.rho,
            },
            "bounds": [list(bounds[0]), list(bounds[1])],
            "target": "iv",
            "weights": "vega-weighted (from quote cleaning)",
            "stopping": {k: solver[k] for k in ("max_nfev", "xtol", "ftol",
                                                "gtol")},
        },
    )
