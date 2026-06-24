"""
SABR-parameterized volatility surface.

Adapts Hagan's SABR Black implied-volatility approximation to the quant-ark
``VolatilitySurface`` interface (``get_vol(strike, time_to_maturity, spot)``)
so a calibrated SABR smile/surface drops directly into a ``PricingEnvironment``.

SABR is parameterized in forward space ``(F, K, T)``. The surface converts the
``spot`` supplied by the interface into a forward via an injectable forward rule
(constant, or a ``spot -> forward`` / ``(spot, ttm) -> forward`` callable), then
evaluates Hagan. Across maturities the calibrated parameters
``(alpha, rho, nu, shift)`` are linearly interpolated between pillars (``beta``
is treated as fixed); this is faithful to the per-slice calibration produced by
:func:`calibrate_sabr_surface`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, ClassVar, Dict, Mapping, Optional, Union

import numpy as np

from quantark.param.vol.vol_surface import VolatilitySurface
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import validate_positive

from .hagan import sabr_implied_vol_black

# A forward rule is either a constant forward, a callable spot -> forward, or a
# callable (spot, ttm) -> forward. ``None`` means "use spot as the forward".
ForwardRule = Optional[Union[float, Callable]]

_PARAM_KEYS = ("alpha", "beta", "rho", "nu", "shift")


def _resolve_forward(forward: ForwardRule, spot: float, ttm: float) -> float:
    """Resolve the forward for the given spot/maturity from the forward rule."""
    if forward is None:
        return float(spot)
    if callable(forward):
        try:
            return float(forward(spot, ttm))
        except TypeError:
            return float(forward(spot))
    return float(forward)


@dataclass
class SABRVolSurface(VolatilitySurface):
    """
    SABR volatility surface implementing the ``VolatilitySurface`` interface.

    Attributes:
        slices: Mapping of maturity (years) -> SABR parameter dict with keys
            ``alpha``, ``beta``, ``rho``, ``nu`` and optional ``shift``. A
            single-pillar mapping yields a maturity-flat smile.
        forward: Forward rule. ``None`` uses spot as the forward; a float is a
            constant forward; a callable is ``spot -> forward`` or
            ``(spot, ttm) -> forward``.
    """

    is_smile: ClassVar[bool] = True

    slices: Dict[float, Dict[str, float]]
    forward: ForwardRule = None
    check_arbitrage: bool = False
    _sorted_t: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.slices:
            raise ValidationError("SABRVolSurface requires at least one maturity slice")

        normalized: Dict[float, Dict[str, float]] = {}
        for t, params in self.slices.items():
            t = float(t)
            validate_positive(t, name="maturity")
            for key in ("alpha", "beta", "rho", "nu"):
                if key not in params:
                    raise ValidationError(
                        f"SABR slice at T={t} missing required parameter '{key}'"
                    )
            validate_positive(float(params["alpha"]), name="alpha")
            validate_positive(float(params["nu"]), name="nu")
            if not -1.0 <= float(params["rho"]) <= 1.0:
                raise ValidationError(f"rho must be in [-1, 1], got {params['rho']}")
            if not 0.0 <= float(params["beta"]) <= 1.0:
                raise ValidationError(f"beta must be in [0, 1], got {params['beta']}")
            normalized[t] = {
                "alpha": float(params["alpha"]),
                "beta": float(params["beta"]),
                "rho": float(params["rho"]),
                "nu": float(params["nu"]),
                "shift": float(params.get("shift", 0.0)),
            }

        self.slices = normalized
        self._sorted_t = np.array(sorted(normalized.keys()), dtype=float)

        if self.check_arbitrage:
            from .diagnostics import check_arbitrage as _check
            ts = list(normalized.keys())
            ref = float(self._sorted_t[len(self._sorted_t) // 2])
            atm = _resolve_forward(self.forward, 1.0, ref) if self.forward else 1.0
            atm = atm if atm > 0 else 1.0
            grid = np.linspace(0.5 * atm, 1.5 * atm, 41)
            report = _check(self, strikes=grid, maturities=ts, spot=atm)
            if not (report.butterfly_ok and report.calendar_ok):
                raise ValidationError(
                    "SABRVolSurface failed arbitrage check: " + "; ".join(report.messages)
                )

    def arbitrage_report(self, strikes, maturities, spot: float):
        """Run no-arbitrage diagnostics on this surface (see sabr.diagnostics)."""
        from .diagnostics import check_arbitrage
        return check_arbitrage(self, strikes=strikes, maturities=maturities, spot=spot)

    @classmethod
    def from_params(
        cls,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        shift: float = 0.0,
        maturity: float = 1.0,
        forward: ForwardRule = None,
    ) -> "SABRVolSurface":
        """Build a maturity-flat surface from a single SABR parameter set."""
        return cls(
            slices={
                float(maturity): {
                    "alpha": alpha,
                    "beta": beta,
                    "rho": rho,
                    "nu": nu,
                    "shift": shift,
                }
            },
            forward=forward,
        )

    @classmethod
    def from_calibration(
        cls,
        calibration: Mapping[float, Mapping[str, float]],
        forward: ForwardRule = None,
    ) -> "SABRVolSurface":
        """Build a surface from :func:`calibrate_sabr_surface` output."""
        return cls(
            slices={float(t): dict(p) for t, p in calibration.items()},
            forward=forward,
        )

    def _params_at(self, ttm: float) -> Dict[str, float]:
        """Linearly interpolate SABR parameters to the requested maturity."""
        ts = self._sorted_t
        if ts.size == 1 or ttm <= ts[0]:
            return self.slices[float(ts[0])]
        if ttm >= ts[-1]:
            return self.slices[float(ts[-1])]
        out: Dict[str, float] = {}
        for key in _PARAM_KEYS:
            values = np.array([self.slices[float(t)][key] for t in ts], dtype=float)
            out[key] = float(np.interp(ttm, ts, values))
        return out

    def get_vol(self, strike: float, time_to_maturity: float, spot: float) -> float:
        """
        SABR Black implied volatility for the given strike, maturity and spot.

        Args:
            strike: Strike price.
            time_to_maturity: Time to maturity in years.
            spot: Current spot price (converted to a forward via the forward rule).

        Returns:
            Black implied volatility (annualized).
        """
        ttm = float(time_to_maturity)
        validate_positive(ttm, name="time_to_maturity")
        params = self._params_at(ttm)
        fwd = _resolve_forward(self.forward, spot, ttm)
        vol = sabr_implied_vol_black(
            fwd,
            float(strike),
            ttm,
            params["alpha"],
            params["beta"],
            params["rho"],
            params["nu"],
            shift=params["shift"],
        )
        return float(np.asarray(vol).reshape(-1)[0])

    def __repr__(self) -> str:
        return f"SABRVolSurface(pillars={list(self.slices.keys())})"


__all__ = ["SABRVolSurface"]
