"""The daily-KI snowball intraday study: the paired-RQMC arm and the QUAD V2 and PDE candidates.

Every arm prices the runtime's own resolved context. The RQMC arm reuses ``cell_price`` -- the
price function the desk bumps difference -- with a fresh RQMC engine per batch, so the three
bump arms of one case share one scramble while different cases never do. It is a reference
where it can resolve a study's budgets and the qualifying arm where it cannot (the daily-KI
study: see ``intraday_gaussian_reference``). The candidates call ``value_intraday`` exactly as
a caller would. Nothing here bypasses the runtime.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde import base_pde_solver
from quantark.asset.equity.engine.pde.grid.config import GridConfig, resolve_config
from quantark.asset.equity.engine.pde.snowball_pde_solver import SnowballPDESolver
from quantark.asset.equity.engine.quad.v2.engine import SnowballQuadEngineV2
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.asset.equity.param.quad_v2_params import QuadV2Params
from quantark.intraday.context import resolve_context
from quantark.intraday.greeks import bump_config_for, cell_price, resolve_theta_step, with_pricing_env
from quantark.intraday.roll import roll_context
from quantark.intraday.service import value_intraday
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError
from quantark.util.numerical import is_close

from quantark.modelvalidation.builders.intraday_common import (
    COMMON_TREES,
    MC_TREES,
    PDE_TREES,
    QUAD_V2_TREES,
    IntradayArm,
    build_request,
    bumped_env,
    implementation_fingerprint,
    plain,
)
from quantark.modelvalidation.candidate import CandidateResult, ConvergenceAxis, ConvergenceLevel
from quantark.modelvalidation.engine_config import engine_config
from quantark.modelvalidation.reference import BatchResult
from quantark.modelvalidation.registry import register_builder
from quantark.modelvalidation.study import SamplingPolicy, batch_seed

#: What the reference estimates for each quantity it can produce. ``None`` records the
#: paired-bump proxy without declaring it a point estimator (spec section 6.1).
_REFERENCE_TARGETS: Mapping[str, Optional[dict]] = {
    "pv": {"estimator": "rqmc_replicate_mean"},
    "desk_delta": {"estimator": "paired_central_difference"},
    "desk_gamma": {"estimator": "paired_central_difference"},
    "desk_theta": {"estimator": "frozen_market_roll_same_seed",
                   "step": "the runtime's default hour, clamped to the current segment"},
    "point_delta": None,
    "point_gamma": None,
}


class IntradaySnowballRQMCReference(IntradayArm):
    """Paired-RQMC benchmark on the resolved intraday context.

    One scramble per (case, batch), shared by that case's bump arms and by nothing else.
    """

    def __init__(self, sampling: SamplingPolicy, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sampling = sampling
        self._kwargs = dict(kwargs)
        if self._params:
            raise ValidationError(
                "equity.snowball.intraday.mc_rqmc takes no params: SnowballMCEngine builds its time grid from "
                "the resolved context's events, so there is no sampling knob to record")
        if sampling.seed_scheme != "substream":
            raise ValidationError(
                "equity.snowball.intraday.mc_rqmc needs seed_scheme 'substream': cases must not share scrambles, "
                f"got {sampling.seed_scheme!r}")
        unknown = [q for q in self.quantities if q not in _REFERENCE_TARGETS]
        if unknown:
            raise ValidationError(
                f"equity.snowball.intraday.mc_rqmc does not produce {unknown}; it produces {sorted(_REFERENCE_TARGETS)}")

    def bind(self, policy: SamplingPolicy) -> "IntradaySnowballRQMCReference":
        """The same reference sampling under ``policy`` (the pipeline binds the effective policy, quick or full)."""
        return IntradaySnowballRQMCReference(sampling=policy, **self._kwargs)

    def targets(self) -> Mapping[str, Optional[dict]]:
        out = {}
        for quantity in self.quantities:
            declared = _REFERENCE_TARGETS[quantity]
            if declared is None:
                out[quantity] = None
            elif declared["estimator"] == "paired_central_difference":
                out[quantity] = {**declared, "bump": self.sampling.bump}
            else:
                out[quantity] = dict(declared)
        return out

    def config(self) -> Mapping[str, Any]:
        return {
            "engine": "SnowballMCEngine",
            "method": MonteCarloMethod.RANDOMIZED_QUASI.value,
            "paths_per_batch": self.sampling.paths_per_batch,
            "seed_scheme": self.sampling.seed_scheme,
            "greeks": ("paired central difference at sampling.bump on the resolved intraday context; "
                       "desk theta by a frozen-market roll priced at the same seed"),
            "point_estimator": "none declared: point quantities are uncertified",
        }

    def identity(self, case) -> Mapping[str, Any]:
        return {
            "builder": "equity.snowball.intraday.mc_rqmc",
            "case": case.name,
            "inputs": self.resolved_inputs(case),
            "targets": plain(self.targets()),
            "config": dict(self.config()),
            "sampling": {
                "paths_per_batch": self.sampling.paths_per_batch,
                "min_batches": self.sampling.min_batches,
                "max_batches": self.sampling.max_batches,
                "seed": self.sampling.seed,
                "seed_scheme": self.sampling.seed_scheme,
                "bump": self.sampling.bump,
            },
            "implementation": implementation_fingerprint(*COMMON_TREES, *MC_TREES),
        }

    def run_batch(self, case, batch_index: int) -> BatchResult:
        environment, product, context = self.specs(case)
        ctx = resolve_context(build_request(environment, product, context))
        seed = batch_seed(self.sampling, case.name, batch_index)
        engine = SnowballMCEngine(
            params=MCParams(seed=seed, num_paths=self.sampling.paths_per_batch, use_qmc=True,
                            rqmc_min_batches=1, rqmc_max_batches=1, rqmc_paths_mode="per_batch"),
            method=MonteCarloMethod.RANDOMIZED_QUASI,
        )
        h, spot = self.sampling.bump, ctx.spot
        base = cell_price(ctx, engine)
        up = cell_price(with_pricing_env(ctx, bumped_env(ctx.pricing_env, 1.0 + h), f"reference_up:{batch_index}"), engine)
        down = cell_price(with_pricing_env(ctx, bumped_env(ctx.pricing_env, 1.0 - h), f"reference_down:{batch_index}"), engine)
        delta = (up - down) / (2.0 * spot * h)
        gamma = (up - 2.0 * base + down) / (spot * h) ** 2
        values = {"pv": base, "desk_delta": delta, "desk_gamma": gamma, "point_delta": delta, "point_gamma": gamma}
        if "desk_theta" in self.quantities:
            values["desk_theta"] = self._desk_theta(ctx, engine, base)
        return BatchResult(index=batch_index, seed=seed, values={q: float(values[q]) for q in self.quantities})

    @staticmethod
    def _desk_theta(ctx, engine, base: float) -> float:
        """The runtime's desk roll: default hour clamped to the segment, frozen market, cash received added back."""
        step = resolve_theta_step(ctx, None, "hour")
        if step.actual is None:
            # No roll exists at an event boundary. The candidate reports ``undefined`` there, and the case
            # must carry the matching semantic assertion; this value is recorded, never gated.
            return 0.0
        rolled = roll_context(ctx, ctx.valuation_timestamp + step.actual)
        received = float(rolled.numerical.paid_cash) - float(ctx.numerical.paid_cash)
        return (cell_price(rolled, engine) + received - base) / step.divisor


@register_builder("equity.snowball.intraday.mc_rqmc", kind="reference")
def build_intraday_snowball_mc_reference(
    environment_params: Mapping[str, Any],
    product_params: Mapping[str, Any],
    sampling: SamplingPolicy,
    quantities: Sequence[str],
    params: Mapping[str, Any],
    context_params: Optional[Mapping[str, Any]] = None,
) -> IntradaySnowballRQMCReference:
    if context_params is None:
        raise ValidationError("equity.snowball.intraday.mc_rqmc needs the study's context block (intraday.sse)")
    return IntradaySnowballRQMCReference(
        sampling=sampling, environment_params=environment_params, product_params=product_params,
        context_params=context_params, quantities=quantities, params=params,
    )


_QUAD_V2_APPLICATION_ONLY = ("backend", "max_nodes", "max_events", "max_states", "max_work_bytes", "max_cache_bytes")
_DESK = (("desk_delta", "delta"), ("desk_gamma", "gamma"), ("desk_theta", "theta"))
_POINT = (("point_delta", "delta"), ("point_gamma", "gamma"))
#: Points x time nodes one ladder solve may hold (two-surface solvers keep ~16 bytes per cell, ~1.5 GiB here).
#: A level above it is skipped and recorded, never silently approximated.
LADDER_MAX_GRID_CELLS = 100_000_000
_PLACEMENT_SHIFTS = (1, 2, 3)        # extra points: each moves every barrier's position inside its cell


def _measure(engine, environment, product, context, quantities) -> Tuple[dict, dict, dict, dict]:
    """PV, desk and point quantities through ``value_intraday``, requested exactly as a caller would.

    Returns (values, statuses, reasons, the base price's numerical diagnostics).
    """
    desk_greeks = tuple(g for q, g in _DESK if q in quantities)
    point_greeks = tuple(g for q, g in _POINT if q in quantities)
    desk = value_intraday(engine, build_request(environment, product, context, greeks=desk_greeks,
                                                greek_convention="desk_bump" if desk_greeks else None))
    point = None
    if point_greeks:
        point = value_intraday(engine, build_request(environment, product, context, greeks=point_greeks,
                                                     greek_convention="point"))
        if not is_close(desk.price, point.price, rel_tol=1e-12, abs_tol=1e-12):
            raise ValidationError(f"two conventions priced one context differently: {desk.price!r} vs {point.price!r}")
    values, statuses, reasons = {"pv": float(desk.price)}, {"pv": "ok"}, {"pv": ""}
    for quantity, name in _DESK + _POINT:
        if quantity not in quantities:
            continue
        greek = (desk if quantity.startswith("desk") else point).greek(name)
        statuses[quantity], reasons[quantity] = greek.status, greek.reason or ""
        if greek.status == "ok":
            values[quantity] = float(greek.value)
    return ({q: values[q] for q in quantities if q in values}, {q: statuses[q] for q in quantities},
            {q: reasons[q] for q in quantities}, dict(desk.numerical))


class _IntradayCandidate(IntradayArm):
    base_name = ""
    trees: Tuple[str, ...] = ()

    def name(self) -> str:
        label = self._params.get("label")
        return f"{self.base_name}.{label}" if label else self.base_name

    def _declared(self) -> dict:
        return {k: v for k, v in self._params.items() if k != "label"}

    def fingerprint(self) -> str:
        return implementation_fingerprint(*COMMON_TREES, *self.trees)

    def _target_engine(self):
        raise NotImplementedError

    def anchor_noise_weights(self, case) -> Mapping[str, float]:
        """Per quantity, the L1 weight of the price stencil it is formed from (per unit of the case's PV).

        Off the banking machine an anchor inherits the prices' cross-architecture noise through its stencil
        (``modelvalidation.anchors.anchor_tolerance``). The desk stencils are the runtime's own: the central
        spot difference at the engine's bump and the frozen-market roll over the default theta step. The point
        Greeks come from each engine's internal stencil, never coarser than the desk bump, so the desk weight
        is their floor: it errs tight, never loose.
        """
        ctx = resolve_context(build_request(*self.specs(case)))
        bump = bump_config_for(self._target_engine())
        move_delta = ctx.spot * bump.spot_bump
        move_gamma = ctx.spot * (getattr(bump, "gamma_spot_bump", None) or bump.spot_bump)
        weights = {"pv": 1.0, "desk_delta": 1.0 / move_delta, "desk_gamma": 4.0 / move_gamma ** 2,
                   "point_delta": 1.0 / move_delta, "point_gamma": 4.0 / move_gamma ** 2}
        step = resolve_theta_step(ctx, None, "hour")
        if step.actual is not None:
            weights["desk_theta"] = 2.0 / abs(step.divisor)
        return {q: w for q, w in weights.items() if q in self.quantities}

    def _level(self, label, resolution, engine, specs, settings, *, target=None) -> ConvergenceLevel:
        """One level with the settings it was asked for and the geometry the route actually solved on."""
        if target is not None:
            return ConvergenceLevel(label, float(resolution), target, settings, is_target=True)
        values, _, _, numerical = _measure(engine, *specs, self.quantities)
        achieved = {k: numerical[k] for k in ("points", "steps_per_day", "requested_steps", "nodes", "cells", "resolution")
                    if k in numerical}
        return ConvergenceLevel(label, float(resolution), values, {**settings, "achieved": achieved})


class IntradaySnowballQuadV2Candidate(_IntradayCandidate):
    """QUAD V2 through the intraday route: kernel point Greeks, desk bumps of its own prices."""

    base_name = "equity.snowball.intraday.quad_v2"
    trees = QUAD_V2_TREES

    def _settings(self) -> dict:
        defaults = QuadV2Params()
        return {"cells_per_sd": float(self._params.get("cells_per_sd", defaults.cells_per_sd)),
                "order": int(self._params.get("order", defaults.order)),
                "domain_sd": float(self._params.get("domain_sd", defaults.domain_sd))}

    @staticmethod
    def _engine(settings: Mapping[str, Any]) -> SnowballQuadEngineV2:
        return SnowballQuadEngineV2(QuadV2Params(**settings))

    def _target_engine(self) -> SnowballQuadEngineV2:
        return self._engine(self._settings())

    def _axis_levels(self) -> dict:
        s = self._settings()
        return {"cells_per_sd": [s["cells_per_sd"] * m for m in (1.0, 2.0, 4.0)],
                "order": [s["order"] + k for k in (0, 4, 8)],
                "domain_sd": [s["domain_sd"] + k for k in (0.0, 2.0, 4.0)]}

    def params(self) -> Mapping[str, Any]:
        engine = self._engine(self._settings())
        return {**self._declared(), "engine": "SnowballQuadEngineV2",
                "desk_bump": float(bump_config_for(engine).spot_bump),
                "grid": engine_config(engine.params, exclude=_QUAD_V2_APPLICATION_ONLY),
                "convergence_axes": self._axis_levels()}

    def evaluate_target(self, case) -> CandidateResult:
        values, statuses, reasons, _ = _measure(self._target_engine(), *self.specs(case), self.quantities)
        return CandidateResult(values=values, statuses=statuses, reasons=reasons)

    def evaluate(self, case) -> CandidateResult:
        specs, settings = self.specs(case), self._settings()
        values, statuses, reasons, _ = _measure(self._engine(settings), *specs, self.quantities)
        axes = []
        for name, levels in self._axis_levels().items():
            built = [self._level("target", levels[0], None, specs, dict(settings), target=values)]
            for value in levels[1:]:
                probe = {**settings, name: value}
                built.append(self._level(f"{name}={value:g}", value, self._engine(probe), specs, probe))
            axes.append(ConvergenceAxis(name, "refinement", tuple(built)))
        return CandidateResult(values=values, statuses=statuses, reasons=reasons, convergence=tuple(axes))


class IntradaySnowballPDECandidate(_IntradayCandidate):
    """The two-surface PDE through the intraday route: grid-stencil point Greeks, desk bumps of its prices."""

    base_name = "equity.snowball.intraday.pde"
    trees = PDE_TREES

    def _accuracy(self) -> str:
        return str(self._params.get("accuracy", "standard"))

    def _target_engine(self) -> SnowballPDESolver:
        return SnowballPDESolver(params=PDEParams(accuracy=self._accuracy()))

    def _explicit_engine(self, points: int, steps_per_day: float) -> SnowballPDESolver:
        base = resolve_config(self._accuracy(), None)
        return SnowballPDESolver(params=PDEParams(grid=GridConfig(
            points=int(points), steps_per_day=float(steps_per_day),
            max_points=max(int(points), int(base.max_points)), max_steps=10 ** 8)))

    def params(self) -> Mapping[str, Any]:
        return {**self._declared(), "engine": "SnowballPDESolver",
                "desk_bump": float(bump_config_for(self._target_engine()).spot_bump),
                "grid": engine_config(resolve_config(self._accuracy(), None)),
                "convergence_axes": {"space": "achieved points x (1, 2, 4)", "time": "achieved steps per day x (1, 2, 4)",
                                     "placement": f"achieved points + {list(_PLACEMENT_SHIFTS)}"}}

    def evaluate_target(self, case) -> CandidateResult:
        values, statuses, reasons, _ = _measure(self._target_engine(), *self.specs(case), self.quantities)
        return CandidateResult(values=values, statuses=statuses, reasons=reasons)

    def evaluate(self, case) -> CandidateResult:
        specs = self.specs(case)
        values, statuses, reasons, numerical = _measure(self._target_engine(), *specs, self.quantities)
        base_pde_solver._ENV_STEP_COEFF_MEMO.clear()
        points, spd = numerical.get("points"), numerical.get("steps_per_day")
        if points is None or spd is None:
            # decided without a grid (terminated, or an event at the valuation instant): nothing to refine
            return CandidateResult(values=values, statuses=statuses, reasons=reasons, exact=True)
        points, spd, steps = int(points), float(spd), float(numerical.get("requested_steps") or 0.0)
        solved = {"points": points, "steps_per_day": spd, "resolution": numerical.get("resolution")}

        def level(label, resolution, p, s):
            cells = p * (steps * (s / spd) + 1.0)
            if cells > LADDER_MAX_GRID_CELLS:
                return None                                  # recorded by its absence: the axis stays short
            built = self._level(label, resolution, self._explicit_engine(p, s), specs, {"points": p, "steps_per_day": s})
            base_pde_solver._ENV_STEP_COEFF_MEMO.clear()
            return built

        target = lambda: self._level("target", 1.0, None, specs, dict(solved), target=values)   # noqa: E731
        space = [target()] + [lv for lv in (level(f"points x{m}", m, points * m, spd) for m in (2, 4)) if lv]
        time = [target()] + [lv for lv in (level(f"steps x{m}", m, points, spd * m) for m in (2, 4)) if lv]
        shifts = [ConvergenceLevel("target", 0.0, values, dict(solved), is_target=True)] + [
            lv for lv in (level(f"points +{k}", float(k), points + k, spd) for k in _PLACEMENT_SHIFTS) if lv]
        axes = (ConvergenceAxis("space", "refinement", tuple(space)), ConvergenceAxis("time", "refinement", tuple(time)),
                ConvergenceAxis("placement", "placement", tuple(shifts)))
        return CandidateResult(values=values, statuses=statuses, reasons=reasons, convergence=axes)


def _candidate_builder(cls):
    def build(environment_params: Mapping[str, Any], product_params: Mapping[str, Any], quantities: Sequence[str],
              params: Mapping[str, Any], context_params: Optional[Mapping[str, Any]] = None):
        if context_params is None:
            raise ValidationError(f"{cls.base_name} needs the study's context block (intraday.sse)")
        return cls(environment_params=environment_params, product_params=product_params,
                   context_params=context_params, quantities=quantities, params=params)
    return build


register_builder("equity.snowball.intraday.quad_v2", kind="candidate")(_candidate_builder(IntradaySnowballQuadV2Candidate))
register_builder("equity.snowball.intraday.pde", kind="candidate")(_candidate_builder(IntradaySnowballPDECandidate))
