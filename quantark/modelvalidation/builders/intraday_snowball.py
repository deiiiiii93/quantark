"""The daily-KI snowball intraday study: one paired-RQMC reference, QUAD V2 and PDE candidates.

Every arm prices the runtime's own resolved context through the ordinary APIs. The
reference reuses ``cell_price`` -- the price function the desk bumps difference -- with a
fresh RQMC engine per batch, so the three bump arms of one case share one scramble while
different cases never do. The candidates call ``value_intraday`` exactly as a caller would.
Nothing here bypasses the runtime.
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
