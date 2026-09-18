"""The deterministic reference of the intraday snowball studies: the Gaussian-transition solver on a nested ladder.

It lives apart from the candidates on purpose. An identity hashes the files an arm depends on, so a change
to this reference's policy must not invalidate a candidate's banked ladders, nor the reverse.

What it states per quantity is a value, an uncertainty radius, and WHICH KIND of radius that is:

* ``analytical``: the quantity has a named exactness basis (a terminated claim, a claim decided at the
  valuation instant, one Gaussian integral taken by adaptive quadrature) and the radius is that basis's
  own error.
* ``calibrated_estimate``: the radius comes from ``intraday_gaussian.ladder_estimate`` -- a numerical
  estimate, calibrated one level down inside the ladder -- plus the errors every level shares
  (``residual_components``). It is never called a bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from quantark.asset.equity.engine.settlement_support import pending_receivable_pv
from quantark.intraday.context import resolve_context
from quantark.intraday.greeks import resolve_theta_step, with_pricing_env
from quantark.intraday.roll import roll_context
from quantark.util.exceptions import ValidationError

from quantark.modelvalidation.builders import intraday_gaussian
from quantark.modelvalidation.builders.intraday_common import (
    COMMON_TREES,
    IntradayArm,
    build_request,
    bumped_env,
    implementation_fingerprint,
    plain,
)
from quantark.modelvalidation.reference import DeterministicResult
from quantark.modelvalidation.registry import register_builder
from quantark.modelvalidation.study import SamplingPolicy

#: What the reference computes for each quantity. Every one is a declared target, the point Greeks included:
#: they are the exact derivatives of the last Gaussian expectation.
_TARGETS: Mapping[str, dict] = {
    "pv": {"estimator": "ladder_estimate_of_the_solve"},
    "desk_delta": {"estimator": "central_difference_of_solves"},
    "desk_gamma": {"estimator": "central_difference_of_solves"},
    "desk_theta": {"estimator": "frozen_market_roll_of_solves",
                   "step": "the runtime's default hour, clamped to the current segment"},
    "point_delta": {"estimator": "analytic_derivative_of_the_last_gaussian_expectation"},
    "point_gamma": {"estimator": "analytic_derivative_of_the_last_gaussian_expectation"},
}
_CANDIDATE_FILE = "quantark/modelvalidation/builders/intraday_snowball.py"
GAUSSIAN_TREES: Tuple[str, ...] = (
    "quantark/modelvalidation/builders/intraday_gaussian.py",
    "quantark/modelvalidation/builders/intraday_gaussian_reference.py",
)
#: Everything this arm depends on: the shared runtime trees and its own two files, never the candidates' file.
REFERENCE_TREES: Tuple[str, ...] = tuple(t for t in COMMON_TREES if t != _CANDIDATE_FILE) + GAUSSIAN_TREES
#: Solved sweeps kept between cases (a few MB each); the oldest leave first.
_MAX_CACHED_SWEEPS = 96
_EPS = float(np.finfo(float).eps)
_EXACTNESS_BASES = {
    "terminated": "no contingent claim remains: the value is the pending receivables, discounted in closed form",
    "decided_at_valuation": "the only remaining instant is the valuation instant: its events are decided at the known spot",
    "single_instant_quadrature": ("one instant remains: value and point derivatives are single Gaussian integrals, taken by "
                                  "adaptive Gauss-Kronrod quadrature split at every jump and kink, with no grid"),
}


@dataclass(frozen=True)
class _Price:
    """One priced context: the whole remaining claim, its point derivatives, and why (if) it is exact."""

    value: float
    delta: float
    gamma: float
    basis: Optional[str]
    errors: Mapping[str, float]              # the exactness basis's own absolute errors; zeros for a ladder price
    note: Mapping[str, Any]                  # what the solve actually discretized


def _nested(points: Sequence[int], name: str, policy: intraday_gaussian.LadderPolicy) -> Tuple[int, ...]:
    levels = tuple(int(p) for p in points)
    if len(levels) < policy.min_levels:
        raise ValidationError(f"{name} needs at least {policy.min_levels} levels (four to estimate, one coarser to "
                              f"calibrate), got {list(levels)}")
    for coarse, fine in zip(levels, levels[1:]):
        if fine - 1 != policy.refinement_ratio * (coarse - 1):
            raise ValidationError(f"{name} must be nested, each level halving the spacing (n -> 2n - 1); got {list(levels)}")
    for level in levels:
        try:
            intraday_gaussian.local_points(level)
        except ValueError as exc:
            raise ValidationError(f"{name}: {exc}") from exc
    return levels


def _common_basis(*prices: _Price) -> Optional[str]:
    """The exactness basis of a quantity built from several prices: every one must be exact."""
    if any(p.basis is None for p in prices):
        return None
    return "+".join(sorted({p.basis for p in prices}))


class IntradaySnowballGaussianReference(IntradayArm):
    """Deterministic reference: the engine-independent Gaussian-transition solver on a nested ladder.

    It shares the resolved context and the product's payoff functions with every arm and nothing else;
    the RQMC arm qualifies it case by case.
    """

    reference_kind = "deterministic"
    policy = intraday_gaussian.LADDER_POLICY

    def __init__(self, sampling: SamplingPolicy, quick: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sampling, self.quick, self._kwargs = sampling, quick, dict(kwargs)
        unknown_params = sorted(set(self._params) - {"points", "quick_points", "width_std"})
        if unknown_params:
            raise ValidationError(f"equity.snowball.intraday.gaussian does not take params {unknown_params}")
        if "points" not in self._params:
            raise ValidationError("equity.snowball.intraday.gaussian needs params.points: the nested refinement ladder")
        self.points = _nested(self._params["points"], "params.points", self.policy)
        self.quick_points = _nested(self._params.get("quick_points", (101, 201, 401, 801, 1601)), "params.quick_points", self.policy)
        self.width_std = float(self._params.get("width_std", 8.0))
        unknown = [q for q in self.quantities if q not in _TARGETS]
        if unknown:
            raise ValidationError(
                f"equity.snowball.intraday.gaussian does not produce {unknown}; it produces {sorted(_TARGETS)}")

    def bind(self, policy: SamplingPolicy, quick: bool = False) -> "IntradaySnowballGaussianReference":
        """The same solver under ``policy`` (it supplies the desk bump); ``quick`` selects the wiring ladder."""
        return type(self)(sampling=policy, quick=quick, **self._kwargs)

    @property
    def levels(self) -> Tuple[int, ...]:
        return self.quick_points if self.quick else self.points

    def targets(self) -> Mapping[str, Optional[dict]]:
        out = {}
        for quantity in self.quantities:
            declared = dict(_TARGETS[quantity])
            if declared["estimator"] == "central_difference_of_solves":
                declared["bump"] = self.sampling.bump
            out[quantity] = declared
        return out

    def error_model(self) -> Mapping[str, Any]:
        """The executed uncertainty policy, structured: the ladder policy object describes itself."""
        return {
            "method": ("backward Gaussian transitions of a piecewise-linear value function of ln S with exact barrier "
                       "jumps; closed-form expectations; an exact last step with analytic first and second derivatives"),
            "levels": list(self.levels),
            "width_std": self.width_std,
            "local_grid": "(points - 1) / 4 + 1 uniform knots over +/- 15 first-interval standard deviations, nested",
            "ladder_policy": self.policy.describe(),
            "exactness_bases": dict(_EXACTNESS_BASES),
            "radius": ("analytical: the exactness basis's own error; calibrated_estimate: the ladder policy's radius plus "
                       "the residual components, scaled to the quantity"),
            "residual_components": {
                "truncation": "analytical bound: value span x reflection-principle mass beyond the flat-extended grids",
                "saturation": "analytical bound: 4 x Phi(-9) x value span x instants",
                "roundoff": "estimate: roundoff_factor x eps x value span x instants",
                "scaling": ("pv 1; desk delta 1/(S h); desk gamma 4/(S h)^2; desk theta 2/step; point delta "
                            "E|Z|/(S sd1); point gamma (E|Z^2-1|/sd1^2 + E|Z|/sd1)/S^2, sd1 the first interval's"),
            },
            "validity_domain": ("deterministic Black-Scholes coefficients, discrete monitoring, snowball payoffs without "
                                "continuous knock-in or Phoenix coupons; every level refines every discretization in play"),
            "calibration_controls": [
                "in-ladder calibration one level down, recorded per quantity",
                "test_intraday_snowball_gaussian_reference.py: closed-form single-event coverage",
                "reference_qualification: independent simulation of every case",
            ],
        }

    def config(self) -> Mapping[str, Any]:
        return {"engine": "intraday_gaussian.solve_snowball", **self.error_model(), "desk_bump": self.sampling.bump,
                "quick_ladder": self.quick}

    def identity(self, case) -> Mapping[str, Any]:
        return {
            "builder": "equity.snowball.intraday.gaussian",
            "case": case.name,
            "inputs": self.resolved_inputs(case),
            "targets": plain(self.targets()),
            "config": plain(dict(self.config())),
            "implementation": implementation_fingerprint(*REFERENCE_TREES),
        }

    # --- pricing ---------------------------------------------------------------------------------------------------
    def _price(self, ctx, points: int) -> _Price:
        """The whole remaining claim: the solve plus the pending receivables ``cell_price`` adds."""
        state = ctx.numerical.lifecycle_state
        pending = float(pending_receivable_pv(state, ctx.pricing_env)) if state is not None else 0.0
        analytic = intraday_gaussian.single_instant_quadrature(ctx)
        if analytic is not None:
            return _Price(analytic["value"] + pending, analytic["delta"], analytic["gamma"], "single_instant_quadrature",
                          analytic["errors"], {"points": points, "instants": 1, "global_spacing": None, "local_spacing": None})
        note: Dict[str, Any] = {}
        value, delta, gamma = intraday_gaussian.solve_snowball(ctx, points, self.width_std, note)
        floating = 4.0 * _EPS * abs(value + pending)
        return _Price(value + pending, delta, gamma, note["basis"], {"value": floating, "delta": 0.0, "gamma": 0.0}, note)

    def _level(self, ctx, points: int) -> Tuple[dict, _Price]:
        """quantity -> (value, exactness basis, basis error) at one ladder level, and the base price."""
        h, spot = self.sampling.bump, ctx.spot
        base = self._price(ctx, points)
        up = self._price(with_pricing_env(ctx, bumped_env(ctx.pricing_env, 1.0 + h), "gaussian_up"), points)
        down = self._price(with_pricing_env(ctx, bumped_env(ctx.pricing_env, 1.0 - h), "gaussian_down"), points)
        move = spot * h
        out = {
            "pv": (base.value, base.basis, base.errors["value"]),
            "desk_delta": ((up.value - down.value) / (2.0 * move), _common_basis(up, down),
                           (up.errors["value"] + down.errors["value"]) / (2.0 * move)),
            "desk_gamma": ((up.value - 2.0 * base.value + down.value) / move ** 2, _common_basis(up, base, down),
                           (up.errors["value"] + 2.0 * base.errors["value"] + down.errors["value"]) / move ** 2),
            "point_delta": (base.delta, base.basis, base.errors["delta"]),
            "point_gamma": (base.gamma, base.basis, base.errors["gamma"]),
        }
        step = resolve_theta_step(ctx, None, "hour")
        if step.actual is None:
            out["desk_theta"] = (float("nan"), None, 0.0)
        else:
            rolled_ctx = roll_context(ctx, ctx.valuation_timestamp + step.actual)
            rolled = self._price(rolled_ctx, points)
            received = float(rolled_ctx.numerical.paid_cash) - float(ctx.numerical.paid_cash)
            out["desk_theta"] = ((rolled.value + received - base.value) / step.divisor, _common_basis(rolled, base),
                                 (rolled.errors["value"] + base.errors["value"]) / abs(step.divisor))
        return out, base

    def _residuals(self, ctx) -> Tuple[dict, dict]:
        """(components, quantity -> allowance) for the errors every level of the ladder shares."""
        components = intraday_gaussian.residual_components(ctx, self.width_std, self.policy)
        eps, sd1, spot, move = components["total"], components["first_sd"], ctx.spot, ctx.spot * self.sampling.bump
        step = resolve_theta_step(ctx, None, "hour")
        abs_z, abs_z2 = 0.7978845608028654, 0.9678828980938468          # E|Z| and E|Z^2 - 1| of a standard normal
        scaled = {"pv": eps, "desk_delta": eps / move, "desk_gamma": 4.0 * eps / move ** 2,
                  "desk_theta": 0.0 if step.actual is None else 2.0 * eps / abs(step.divisor),
                  "point_delta": abs_z * eps / (spot * sd1) if sd1 > 0.0 else 0.0,
                  "point_gamma": (abs_z2 * eps / sd1 ** 2 + abs_z * eps / sd1) / spot ** 2 if sd1 > 0.0 else 0.0}
        return components, scaled

    def _check_refinement(self, notes: Sequence[Mapping[str, Any]], case_name: str) -> None:
        """Every discretization the solve used must be refined by every level: a repeated grid reads as convergence."""
        ratio = float(self.policy.refinement_ratio)
        for which in ("global_spacing", "local_spacing"):
            spacings = [note.get(which) for note in notes]
            used = [s is not None for s in spacings]
            if any(used) and not all(used):
                raise ValidationError(f"{case_name}: the {which} discretization is used on some ladder levels only")
            for coarse, fine in zip(spacings, spacings[1:]):
                if coarse is not None and abs(coarse / fine - ratio) > 1e-9 * ratio:
                    raise ValidationError(
                        f"{case_name}: the ladder does not refine its {which} by {ratio:g} at every level "
                        f"({coarse!r} -> {fine!r}); a repeated or uneven discretization cannot show convergence")

    def solve(self, case) -> DeterministicResult:
        environment, product, context = self.specs(case)
        ctx = resolve_context(build_request(environment, product, context))
        solved = [self._level(ctx, points) for points in self.levels]
        levels, bases = [level for level, _ in solved], [base for _, base in solved]
        self._check_refinement([base.note for base in bases], case.name)
        components, residual = self._residuals(ctx)
        values, radii, kinds, undefined, ladder = {}, {}, {}, {}, {}
        for quantity in self.quantities:
            column = [level[quantity][0] for level in levels]
            missing = [value != value for value in column]              # NaN: no such number at this instant
            if all(missing):
                undefined[quantity] = "valuation is at an event instant: no roll or derivative exists inside the segment"
                continue
            if any(missing):
                raise ValidationError(f"{case.name}: {quantity} is defined on some ladder levels only: {column}")
            basis, basis_error = levels[-1][quantity][1], levels[-1][quantity][2]
            estimate = intraday_gaussian.ladder_estimate(column, self.policy, exact_basis=basis)
            exact = estimate.rule == "exact"
            shared = basis_error if exact else residual[quantity]
            values[quantity], radii[quantity] = estimate.value, estimate.radius + shared
            kinds[quantity] = "analytical" if exact else "calibrated_estimate"
            ladder[quantity] = {"values": column, "extrapolants": list(estimate.extrapolants), "rule": estimate.rule,
                                "observed_order": estimate.observed_order, "basis": estimate.basis,
                                "ladder_radius": estimate.radius, "shared_error": shared,
                                "calibration": estimate.calibration}
        evidence = {
            "levels": list(self.levels),
            "grids": [{"points": base.note.get("points"), "instants": base.note.get("instants"),
                       "global_spacing": base.note.get("global_spacing"), "local_spacing": base.note.get("local_spacing"),
                       "domain": base.note.get("domain")} for base in bases],
            "residual_components": components,
            "quantities": ladder,
        }
        if bases[-1].basis == "single_instant_quadrature":
            # the grid solver is not what prices this case; its finest level is recorded as a consistency check
            solver_value = intraday_gaussian.solve_snowball(ctx, self.levels[-1], self.width_std)[0]
            state = ctx.numerical.lifecycle_state
            pending = float(pending_receivable_pv(state, ctx.pricing_env)) if state is not None else 0.0
            evidence["grid_solver_cross_check"] = {"points": self.levels[-1], "value": solver_value + pending,
                                                   "difference": solver_value + pending - bases[-1].value}
        cache = intraday_gaussian._SWEEP_CACHE
        while len(cache) > _MAX_CACHED_SWEEPS:
            cache.pop(next(iter(cache)))
        assert all(isfinite(v) for v in values.values())
        return DeterministicResult(values=values, radii=radii, undefined=undefined, evidence=evidence, radius_basis=kinds)


@register_builder("equity.snowball.intraday.gaussian", kind="reference")
def build_intraday_snowball_gaussian_reference(
    environment_params: Mapping[str, Any],
    product_params: Mapping[str, Any],
    sampling: SamplingPolicy,
    quantities: Sequence[str],
    params: Mapping[str, Any],
    context_params: Optional[Mapping[str, Any]] = None,
) -> IntradaySnowballGaussianReference:
    if context_params is None:
        raise ValidationError("equity.snowball.intraday.gaussian needs the study's context block (intraday.sse)")
    return IntradaySnowballGaussianReference(
        sampling=sampling, environment_params=environment_params, product_params=product_params,
        context_params=context_params, quantities=quantities, params=params,
    )
