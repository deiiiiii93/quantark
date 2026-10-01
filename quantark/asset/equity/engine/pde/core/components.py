"""The three component seams and their built-ins (spec 2 §4.1, §4.2, §5.8).

- ``OperatorFactory`` assembles the stencil from the sampled market;
  built-in ``CentralDifference``.
- ``TimeStepper`` owns the θ schedule and advances a column group;
  built-in ``CrankNicolson``, which carries the four scheme settings.
- ``Readout`` reads a price and delta/gamma off a column; built-in
  ``StandardReadout``.

Every built-in body is today's arithmetic, moved verbatim from the solver
method its comment names. A component's identity is its value fingerprint
(``identity.component_fingerprint``); its ``evidence()`` is a human-readable
record and never keys anything (decision 8).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from quantark.asset.equity.engine.pde.grid.request import TimeRefinement
from quantark.util.exceptions import NumericalError, PricingError, ValidationError
from quantark.util.numerical import is_close

from .backends import banded_advance, sparse_lu_advance
from .coefficients import StepCoefficients
from .groups import Backend
from .samples import MarketSamples, SampleKind

__all__ = [
    "OperatorFactory", "TimeStepper", "Readout",
    "CentralDifference", "CrankNicolson", "StandardReadout",
    "resolve_stepper", "scalar_stencil", "nodal_stencil", "flat_exact",
    "interpolate_price", "delta_gamma",
    "validate_schedule", "validate_coefficients", "validate_advanced", "validate_readout",
]


# --- protocols (§5.8) ----------------------------------------------------------

@runtime_checkable
class OperatorFactory(Protocol):
    supports: frozenset

    def assemble(self, samples: MarketSamples, layout) -> StepCoefficients: ...

    def evidence(self) -> dict: ...


@runtime_checkable
class TimeStepper(Protocol):
    def schedule(self, layout) -> np.ndarray: ...

    # Returns the advanced interior columns: §4.1 and the §5.8 validation
    # rules say so ("advanced columns"); §5.8's ``-> None`` contradicts both.
    def step(self, state, s, dt, theta, boundaries) -> np.ndarray: ...

    def evidence(self) -> dict: ...


@runtime_checkable
class Readout(Protocol):
    def price(self, column: np.ndarray, spot: float, layout) -> float: ...

    def delta_gamma(self, column: np.ndarray, spot: float, layout) -> Tuple[float, float]: ...

    def evidence(self) -> dict: ...


# --- the stencil ---------------------------------------------------------------

def scalar_stencil(r, q, sigma, dx_vec, num_x):
    """Coefficients (l, c, u) of the tridiagonal FD system for one (r, q, sigma)."""
    # base_pde_solver.py (_calculate_coefficients), verbatim
    mu = r - q - 0.5 * sigma * sigma
    D = 0.5 * sigma * sigma

    if is_close(float(np.max(dx_vec)), float(np.min(dx_vec))):
        dx = dx_vec[0]
        l = np.full(num_x, D / (dx * dx) - mu / (2.0 * dx))
        c = np.full(num_x, -2.0 * D / (dx * dx) - r)
        u = np.full(num_x, D / (dx * dx) + mu / (2.0 * dx))
    else:
        h_m, h_p = dx_vec[:-1], dx_vec[1:]
        h_sum, h_prod = h_m + h_p, h_m * h_p

        l_diff, c_diff, u_diff = (
            2.0 * D / (h_m * h_sum),
            -2.0 * D / h_prod,
            2.0 * D / (h_p * h_sum),
        )
        l_drift, c_drift, u_drift = (
            -mu * h_p / (h_m * h_sum),
            mu * (h_p - h_m) / h_prod,
            mu * h_m / (h_p * h_sum),
        )

        l, c, u = np.zeros(num_x), np.zeros(num_x), np.zeros(num_x)
        l[1:-1], c[1:-1], u[1:-1] = (
            l_diff + l_drift,
            c_diff + c_drift - r,
            u_diff + u_drift,
        )
        l[0], c[0], u[0] = l[1], c[1], u[1]
        l[-1], c[-1], u[-1] = l[-2], c[-2], u[-2]

    return l, c, u


def nodal_stencil(r, q, sigma_vec, dx_vec, num_x):
    """Coefficients for a per-node vol vector (the local-vol stencil)."""
    # snowball_vol_pde_solvers.py (_local_vol_coefficients), verbatim after its
    # validation, which moved to the sampler (samples._validated_local_vol_row)
    sigma_vec = np.asarray(sigma_vec, dtype=float)
    diff = 0.5 * sigma_vec * sigma_vec
    drift = float(r) - float(q) - diff
    dx_vec = np.asarray(dx_vec, dtype=float)

    if is_close(float(np.max(dx_vec)), float(np.min(dx_vec))):
        dx = float(dx_vec[0])
        l = diff / (dx * dx) - drift / (2.0 * dx)
        c = -2.0 * diff / (dx * dx) - float(r)
        u = diff / (dx * dx) + drift / (2.0 * dx)
        return l, c, u

    h_m, h_p = dx_vec[:-1], dx_vec[1:]
    h_sum, h_prod = h_m + h_p, h_m * h_p
    d_i = diff[1:-1]
    mu_i = drift[1:-1]
    l = np.zeros(num_x)
    c = np.zeros(num_x)
    u = np.zeros(num_x)
    l[1:-1] = 2.0 * d_i / (h_m * h_sum) - mu_i * h_p / (h_m * h_sum)
    c[1:-1] = -2.0 * d_i / h_prod + mu_i * (h_p - h_m) / h_prod - float(r)
    u[1:-1] = 2.0 * d_i / (h_p * h_sum) + mu_i * h_m / (h_p * h_sum)
    l[0], c[0], u[0] = l[1], c[1], u[1]
    l[-1], c[-1], u[-1] = l[-2], c[-2], u[-2]
    return l, c, u


def flat_exact(sc: StepCoefficients, flat) -> StepCoefficients:
    """Replace a single unique set by the stencil of the exact cumulative scalars.

    One unique set means the curves are constant on the grid, so the
    cumulative scalars equal the forwards mathematically; substituting is
    exact (not an approximation) and makes flat envs bit-identical to the
    pre-term code path. Drops ``zero_diffusion_sets``, as it always did.
    """
    # base_pde_solver.py (_flat_exact_step_coefficients), verbatim
    if sc.n_unique != 1:
        return sc
    return StepCoefficients(lcu_sets=[flat], set_index=sc.set_index, n_unique=1)


def _scalar_step_sets(samples: MarketSamples, dx_vec, num_x) -> StepCoefficients:
    # base_pde_solver.py (_build_step_coefficients, from ``triples``), verbatim
    n_steps = len(samples.t) - 1
    # 12 decimals (same precision as the dt cache keys) absorbs DF
    # round-trip ulp noise so flat curves dedupe to ONE set
    triples = np.column_stack(
        (
            np.round(samples.fwd_rates, 12),
            np.round(samples.fwd_carry, 12),
            np.round(samples.step_vols, 12),
        )
    )
    uniq, set_index = np.unique(triples, axis=0, return_inverse=True)
    n_unique = uniq.shape[0]

    # Vectorized _calculate_coefficients over all unique triples at once
    r_u, q_u, sig_u = uniq[:, 0:1], uniq[:, 1:2], uniq[:, 2:3]  # (n_unique, 1)
    mu = r_u - q_u - 0.5 * sig_u * sig_u
    D = 0.5 * sig_u * sig_u
    dx_vec = np.asarray(dx_vec, dtype=float)
    l = np.zeros((n_unique, num_x))
    c = np.zeros((n_unique, num_x))
    u = np.zeros((n_unique, num_x))
    if is_close(float(np.max(dx_vec)), float(np.min(dx_vec))):
        dx = dx_vec[0]
        l[:, :] = D / (dx * dx) - mu / (2.0 * dx)
        c[:, :] = -2.0 * D / (dx * dx) - r_u
        u[:, :] = D / (dx * dx) + mu / (2.0 * dx)
    else:
        h_m, h_p = dx_vec[:-1], dx_vec[1:]
        h_sum, h_prod = h_m + h_p, h_m * h_p
        l[:, 1:-1] = 2.0 * D / (h_m * h_sum) - mu * h_p / (h_m * h_sum)
        c[:, 1:-1] = -2.0 * D / h_prod + mu * (h_p - h_m) / h_prod - r_u
        u[:, 1:-1] = 2.0 * D / (h_p * h_sum) + mu * h_m / (h_p * h_sum)
        l[:, 0], c[:, 0], u[:, 0] = l[:, 1], c[:, 1], u[:, 1]
        l[:, -1], c[:, -1], u[:, -1] = l[:, -2], c[:, -2], u[:, -2]

    # Zero-diffusion sets (sigma_step exactly 0.0 — holiday intervals of a
    # trading-clock surface, spec 2026-09-01 trading-clock-vol §4.5):
    # centered advection has a wrong-sign off-diagonal at D = 0, so those
    # rows are overwritten with first-order upwind selected by sign(mu);
    # the -0.5*sigma^2 drift term vanishes, so mu = r - q exactly.
    sig_zero = uniq[:, 2] == 0.0
    if np.any(sig_zero):
        uniform_dx = is_close(float(np.max(dx_vec)), float(np.min(dx_vec)))
        if uniform_dx:
            h_minus = np.full(num_x - 2, dx_vec[0])
            h_plus = h_minus
        else:
            h_minus, h_plus = dx_vec[:-1], dx_vec[1:]
        for k in np.nonzero(sig_zero)[0]:
            r_k = float(uniq[k, 0])
            mu_k = r_k - float(uniq[k, 1])
            if mu_k >= 0.0:
                l[k, 1:-1] = 0.0
                u[k, 1:-1] = mu_k / h_plus
                c[k, 1:-1] = -mu_k / h_plus - r_k
            else:
                l[k, 1:-1] = -mu_k / h_minus
                u[k, 1:-1] = 0.0
                c[k, 1:-1] = mu_k / h_minus - r_k
            l[k, 0], c[k, 0], u[k, 0] = l[k, 1], c[k, 1], u[k, 1]
            l[k, -1], c[k, -1], u[k, -1] = l[k, -2], c[k, -2], u[k, -2]

    lcu_sets = [(l[k], c[k], u[k]) for k in range(n_unique)]
    return StepCoefficients(
        lcu_sets=lcu_sets,
        set_index=np.asarray(set_index, dtype=int).reshape(n_steps),
        n_unique=n_unique,
        zero_diffusion_sets=frozenset(np.nonzero(sig_zero)[0].tolist()),
    )


def _nodal_step_sets(samples: MarketSamples, dx_vec, num_x) -> StepCoefficients:
    # snowball_vol_pde_solvers.py (LV _build_step_coefficients), verbatim
    lcu_sets = [
        nodal_stencil(
            float(samples.fwd_rates[j]),
            float(samples.fwd_carry[j]),
            samples.vol_rows[j],
            dx_vec,
            num_x,
        )
        for j in range(len(samples.vol_rows))
    ]
    return StepCoefficients(
        lcu_sets=lcu_sets,
        set_index=np.arange(len(lcu_sets), dtype=int),
        n_unique=len(lcu_sets),
    )


@dataclass(frozen=True)
class CentralDifference:
    """The built-in operator: today's central-difference stencil.

    Scalar samples: the per-step triples rounded and deduplicated, first-order
    upwind rows on zero-diffusion sets, then the flat-exact replacement when
    one set remains. Nodal samples: one set per step, no deduplication.
    """

    supports = frozenset({SampleKind.SCALAR, SampleKind.NODAL})

    def assemble_sets(self, samples: MarketSamples, dx_vec, num_x) -> StepCoefficients:
        """The per-step sets BEFORE the flat-exact replacement (the legacy
        ``_build_step_coefficients`` result, which ``_solve`` then replaced)."""
        if samples.kind is SampleKind.NODAL:
            return _nodal_step_sets(samples, dx_vec, num_x)
        return _scalar_step_sets(samples, dx_vec, num_x)

    def flat(self, samples: MarketSamples, dx_vec, num_x):
        return scalar_stencil(samples.r_tau, samples.q_tau, samples.sigma_tau, dx_vec, num_x)

    def assemble_arrays(self, samples: MarketSamples, dx_vec, num_x) -> StepCoefficients:
        sc = self.assemble_sets(samples, dx_vec, num_x)
        if samples.kind is SampleKind.NODAL or sc.n_unique != 1:
            return sc
        return flat_exact(sc, self.flat(samples, dx_vec, num_x))

    def assemble(self, samples: MarketSamples, layout) -> StepCoefficients:
        return self.assemble_arrays(samples, layout.spatial.dx, len(layout.spatial.x))

    def evidence(self) -> dict:
        return {}


# --- the stepper ---------------------------------------------------------------

_BANDED = Backend.BANDED          # read once per group per step

#: The two forms of a damped interval (spec 3A §5.3).
DAMPING_FORMS = ("full_step", "half_step")


@dataclass(frozen=True, kw_only=True)
class CrankNicolson:
    """The built-in stepper: θ-weighted time stepping with terminal and event damping.

    ``damping`` is the form of a damped interval (spec 3A §5.3): ``"full_step"`` takes it as one
    whole step at the damping weight (0.7.0); ``"half_step"`` as two half-steps, and the two counts
    then count intervals, whose midpoints the time builder inserts (``time_refinement``). ``None``
    form or counts are the route's defaults (``grid.config.DAMPING_DEFAULTS``), resolved once by
    ``prepare``.
    """
    theta: float = 0.5
    event_theta: float = 1.0
    terminal_damping_steps: Optional[int] = None
    event_damping_steps: Optional[int] = None
    damping: Optional[str] = None

    def __post_init__(self):
        if not 0.0 <= float(self.theta) <= 1.0:
            raise ValidationError(f"theta must be in [0, 1], got {self.theta}")
        if not 0.0 < float(self.event_theta) <= 1.0:
            raise ValidationError(f"event_theta must be in (0, 1], got {self.event_theta}")
        for name in ("terminal_damping_steps", "event_damping_steps"):
            v = getattr(self, name)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 0):
                raise ValidationError(f"{name} must be a non-negative int or None, got {v!r}")
        if self.damping is not None and self.damping not in DAMPING_FORMS:
            raise ValidationError(f"damping must be None or one of {DAMPING_FORMS}, got {self.damping!r}")

    def resolved(self, damping_defaults: tuple) -> "CrankNicolson":
        form, term, event = damping_defaults
        return replace(self,
                       damping=form if self.damping is None else self.damping,
                       terminal_damping_steps=term if self.terminal_damping_steps is None else self.terminal_damping_steps,
                       event_damping_steps=event if self.event_damping_steps is None else self.event_damping_steps)

    def _require_resolved(self, what: str) -> None:
        if self.damping is None or self.terminal_damping_steps is None or self.event_damping_steps is None:
            raise ValidationError(f"CrankNicolson.{what} needs a resolved stepper; call resolved() first")

    def time_refinement(self) -> TimeRefinement:
        """The intervals the time grid must split for this schedule: the two counts under
        ``half_step``, none under ``full_step``."""
        self._require_resolved("time_refinement")
        if self.damping == "half_step":
            return TimeRefinement(int(self.terminal_damping_steps), int(self.event_damping_steps))
        return TimeRefinement()

    def _damping_sets(self, layout) -> tuple:
        # grid/time.py (build_time), verbatim: damping follows EVENT nodes only. Under half_step
        # each counted interval is two half-steps, exactly the ones the builder created.
        self._require_resolved("schedule")
        per = 2 if self.damping == "half_step" else 1
        step_of = layout.time.step_of
        event_damp = frozenset(
            step_of[e] - j
            for e in layout.request.event_times
            for j in range(1, per * int(self.event_damping_steps) + 1)
            if step_of[e] - j >= 0
        )
        n_steps = len(layout.time.dt)
        term_damp = frozenset(
            n_steps - j
            for j in range(1, per * int(self.terminal_damping_steps) + 1)
            if n_steps - j >= 0
        )
        return event_damp, term_damp

    def schedule(self, layout) -> np.ndarray:
        # the 0.6.0 per-step theta rule (base_pde_solver.py), verbatim order
        event_damp, term_damp = self._damping_sets(layout)
        theta = np.full(layout.time.actual_steps, float(self.theta))
        for k in event_damp:
            theta[k] = float(self.event_theta)
        for k in term_damp:
            theta[k] = 1.0  # terminal wins on overlap
        return theta

    def step(self, state, s, dt, theta, boundaries):
        if state.group.backend is _BANDED:
            return banded_advance(state, s, dt, theta, boundaries)
        return sparse_lu_advance(state, s, dt, theta, boundaries)

    def evidence(self) -> dict:
        out = {"theta": float(self.theta), "event_theta": float(self.event_theta)}
        for name in ("terminal_damping_steps", "event_damping_steps"):
            if getattr(self, name) is not None:
                out[name] = int(getattr(self, name))
        if self.damping is not None:
            out["damping"] = self.damping
        return out


def resolve_stepper(params, damping_defaults):
    """The stepper a route marches with: the built-in with ``None`` damping counts
    resolved to the route's defaults (``grid.config.DAMPING_DEFAULTS``); any
    other stepper as given."""
    stepper = params.stepper
    if type(stepper) is CrankNicolson:
        return stepper.resolved(damping_defaults)
    return stepper


# --- the readout ---------------------------------------------------------------

def interpolate_price(v_vec, x_vec, x_target) -> float:
    """Linear interpolation of the value at the target log-price."""
    # base_pde_solver.py (_interpolate_price), verbatim
    return float(np.interp(x_target, x_vec, v_vec))


def delta_gamma(v_vec, x_vec, x_target, spot) -> Tuple[float, float]:
    """Delta and gamma at the target log-price from a local cubic (or the three-node stencil).

    In log-space: dV/dS = (1/S) * dV/dx; d2V/dS2 = (1/S^2) * (d2V/dx^2 - dV/dx).
    """
    # base_pde_solver.py (_calculate_delta_gamma), verbatim
    # A quadratic's curvature is constant: carrying its first derivative
    # to the query cannot carry its second derivative there. On refinement
    # the nearest node changes sides, producing O(dx) gamma oscillations.
    # Interpolate a local cubic around the QUERY, so both derivatives are
    # evaluated there (second-order curvature on a non-uniform grid).
    if len(x_vec) >= 4:
        start = int(np.clip(np.searchsorted(x_vec, x_target) - 2, 0, len(x_vec) - 4))
        offsets = x_vec[start:start + 4] - float(x_target)
        scale = float(np.max(np.abs(offsets)))
        z = offsets / scale
        values = v_vec[start:start + 4]
        coefficients = np.linalg.solve(np.vander(z, 4, increasing=True), values - values[1])
        dv_dx = float(coefficients[1]) / scale
        d2v_dx2 = 2.0 * float(coefficients[2]) / (scale * scale)
        return dv_dx / spot, (d2v_dx2 - dv_dx) / (spot * spot)

    # A minimal three-node grid has only a quadratic interpolant.
    idx = int(np.searchsorted(x_vec, x_target))
    idx = max(1, min(idx, len(x_vec) - 2))
    if idx > 1 and abs(x_vec[idx - 1] - x_target) < abs(x_vec[idx] - x_target):
        idx -= 1

    # Non-uniform three-point stencil (exact for quadratics on ANY local
    # spacing). The symmetric formulas are only valid when h_m == h_p; on
    # the adaptive grid the asymmetry error is proportional to gamma.
    h_m = x_vec[idx] - x_vec[idx - 1]
    h_p = x_vec[idx + 1] - x_vec[idx]
    h_sum = h_m + h_p

    v_m, v_0, v_p = v_vec[idx - 1], v_vec[idx], v_vec[idx + 1]
    dv_dx = (
        -h_p / (h_m * h_sum) * v_m
        + (h_p - h_m) / (h_m * h_p) * v_0
        + h_m / (h_p * h_sum) * v_p
    )
    d2v_dx2 = 2.0 * (
        v_m / (h_m * h_sum) - v_0 / (h_m * h_p) + v_p / (h_p * h_sum)
    )

    # The stencil's quadratic has a LINEAR first derivative, so carrying
    # dV/dx from the stencil node to x_target is exact for that quadratic
    # (d2V/dx2 is constant on it). Reporting the node's derivative instead
    # makes delta piecewise constant in spot and moves it whenever the
    # mesh moves -- changing a domain bound is enough.
    offset = float(x_target) - float(x_vec[idx])
    if offset != 0.0:
        dv_dx = dv_dx + offset * d2v_dx2

    # Convert to price-space derivatives at the query spot itself.
    delta = dv_dx / spot
    gamma = (d2v_dx2 - dv_dx) / (spot * spot)
    return delta, gamma


@dataclass(frozen=True)
class StandardReadout:
    """The built-in readout: linear price interpolation, delta/gamma from the local cubic."""

    def price(self, column, spot, layout) -> float:
        return interpolate_price(column, layout.spatial.x, np.log(spot))

    def delta_gamma(self, column, spot, layout) -> Tuple[float, float]:
        return delta_gamma(column, layout.spatial.x, np.log(spot), spot)

    def evidence(self) -> dict:
        return {}


# --- validation of component results (§5.8): raise, never fall back ------------

def _fail(component, message):
    raise NumericalError(f"{type(component).__name__}: {message}")


def validate_schedule(schedule, layout, component) -> None:
    ask = getattr(component, "time_refinement", None)
    expected = TimeRefinement() if ask is None else ask()
    got = layout.request.refinement
    if got != expected:
        raise PricingError(
            f"{type(component).__name__} marches a time grid refined {expected}, but this layout "
            f"was built refined {got}: a frozen, session or external layout from another damping "
            "form is refused, never marched (spec 3A §5.3)")
    schedule = np.asarray(schedule)
    n = layout.time.actual_steps
    if schedule.shape != (n,):
        _fail(component, f"schedule has shape {schedule.shape}, the layout has {n} steps")
    if not np.all(np.isfinite(schedule)):
        _fail(component, "schedule is not finite")
    if np.any(schedule < 0.0) or np.any(schedule > 1.0):
        _fail(component, "schedule weights must lie in [0, 1]")


def validate_coefficients(sc, n_steps, num_x, component) -> None:
    for k, lcu in enumerate(sc.lcu_sets):
        for name, arr in zip("lcu", lcu):
            arr = np.asarray(arr)
            if arr.shape != (num_x,):
                _fail(component, f"set {k} {name} has shape {arr.shape}, expected ({num_x},)")
            if not np.all(np.isfinite(arr)):
                _fail(component, f"set {k} {name} is not finite")
    idx = np.asarray(sc.set_index)
    if not np.issubdtype(idx.dtype, np.integer) or idx.shape != (n_steps,):
        _fail(component, f"set_index must be an integer array of length {n_steps}")
    if n_steps and (idx.min() < 0 or idx.max() >= sc.n_unique):
        _fail(component, f"set_index values must lie in [0, {sc.n_unique})")
    if not set(sc.zero_diffusion_sets) <= set(range(sc.n_unique)):
        _fail(component, "zero_diffusion_sets must be a subset of range(n_unique)")


def validate_advanced(out, group, n_int, component, n_cols=None) -> None:
    """``n_cols``: the columns the group handed the stepper (a rolling group's state widths)."""
    out = np.asarray(out)
    if n_cols is None:
        n_cols = 1 if group.one_dimensional_rhs else len(group.surfaces)
    if out.shape != (n_int, n_cols):
        _fail(component, f"advanced columns have shape {out.shape}, expected ({n_int}, {n_cols})")
    if not np.isfinite(out).all():          # the method, not np.all: this runs once per step
        _fail(component, "advanced columns are not finite")


def validate_readout(value, component) -> None:
    if not np.all(np.isfinite(np.asarray(value, dtype=float))):
        _fail(component, f"readout value {value!r} is not finite")
