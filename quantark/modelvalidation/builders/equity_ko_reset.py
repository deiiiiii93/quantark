"""Certification builders for the KO-reset snowball family.

The same three arms as the snowball and phoenix studies, over a payoff whose
hard part is a *regime switch*: the trade runs a pre-KI KO schedule to
maturity_pre, and only once it knocks in does it switch to a second, lower KO
schedule running on to maturity_post. Two value surfaces therefore live on
different horizons, which is exactly where the deterministic engines went wrong
before (audit #13: the not-yet-KI surface was matured on the post-KI schedule).

Flat-BSM market data and the central-difference Greek helper are shared with
``equity_snowball``; see the note there. One structural choice is pinned rather
than exposed: the post-KI schedule is ABSOLUTE, because the REBASED mode is
rejected outright by the quadrature engine and so cannot be certified against
the same benchmark.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.pde.grid.config import resolve_config
from quantark.asset.equity.engine.pde.ko_reset_snowball_pde_solver import (
    KOResetSnowballPDESolver,
)
from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import (
    KOResetSnowballQuadEngine,
)
from quantark.asset.equity.param import MCParams, PDEParams, QuadParams
from quantark.asset.equity.product.option import create_ko_reset_snowball
from quantark.asset.equity.product.option.ko_reset_snowball_option import (
    KnockOutResetSnowballOption,
    PostKOScheduleMode,
)
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError

from quantark.modelvalidation.builders.equity_snowball import (
    _COARSER_ACCURACY,
    _QUAD_NON_NUMERIC,
    _central_difference_greeks,
    make_environment,
)
from quantark.modelvalidation.candidate import CandidateResult, LadderRung
from quantark.modelvalidation.engine_config import engine_config
from quantark.modelvalidation.reference import BatchResult
from quantark.modelvalidation.registry import register_builder
from quantark.modelvalidation.study import SamplingPolicy

_PRODUCT_KEYS = (
    "initial_price",
    "strike",
    "maturity_pre",
    "maturity_post",
    "pre_ko_barrier",
    "pre_ko_rate",
    "post_ko_barrier",
    "post_ko_rate",
    "ki_barrier",
    "ki_continuous",
    "contract_multiplier",
)

_REQUIRED_PRODUCT_KEYS = (
    "initial_price",
    "strike",
    "maturity_pre",
    "maturity_post",
    "ki_barrier",
)

#: See ``equity_snowball._REFERENCE_KEYS``.
_REFERENCE_KEYS: frozenset = frozenset()


@register_builder("equity.ko_reset_snowball", kind="product")
def build_ko_reset_product_spec(params: Mapping[str, Any]) -> dict:
    """Validate a KO-reset snowball product spec."""
    spec = dict(params)
    unknown = set(spec) - set(_PRODUCT_KEYS)
    if unknown:
        raise ValidationError(
            f"Unknown equity.ko_reset_snowball product keys: {sorted(unknown)}; "
            f"expected a subset of {_PRODUCT_KEYS}"
        )
    for key in _REQUIRED_PRODUCT_KEYS:
        if key not in spec:
            raise ValidationError(f"equity.ko_reset_snowball product is missing {key!r}")
    if float(spec["maturity_post"]) < float(spec["maturity_pre"]):
        raise ValidationError(
            "maturity_post must not precede maturity_pre: the post-KI schedule "
            "runs on from the pre-KI one"
        )
    return spec


def make_ko_reset(spec: Mapping[str, Any]) -> KnockOutResetSnowballOption:
    """Build a KO-reset snowball with an ABSOLUTE post-KI schedule."""
    kwargs: dict[str, Any] = dict(
        initial_price=float(spec["initial_price"]),
        strike=float(spec["strike"]),
        maturity_pre=float(spec["maturity_pre"]),
        maturity_post=float(spec["maturity_post"]),
        contract_multiplier=float(spec.get("contract_multiplier", 1.0)),
        ki_barrier=float(spec["ki_barrier"]),
        ki_continuous=bool(spec.get("ki_continuous", True)),
        post_ko_mode=PostKOScheduleMode.ABSOLUTE,
    )
    for key in ("pre_ko_barrier", "post_ko_barrier", "pre_ko_rate", "post_ko_rate"):
        if key in spec:
            kwargs[key] = float(spec[key])
    return create_ko_reset_snowball(**kwargs)


class _KOResetArm:
    """Shared spec handling for every arm of this study."""

    def __init__(
        self,
        environment_params: Mapping[str, Any],
        product_params: Mapping[str, Any],
        quantities: Sequence[str],
        params: Mapping[str, Any],
    ) -> None:
        self.environment_params = dict(environment_params)
        self.product_params = dict(product_params)
        self.quantities = tuple(quantities)
        self._params = dict(params)

    def _specs(self, case) -> tuple[dict, dict]:
        environment = dict(self.environment_params)
        environment.update(case.environment_params)
        product = dict(self.product_params)
        product.update(case.product_params)
        return environment, product


class KOResetMCReference(_KOResetArm):
    """Paired-RQMC benchmark: one randomization per batch, shared across bumps."""

    def __init__(self, sampling: SamplingPolicy, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sampling = sampling
        unsupported = set(self._params) - _REFERENCE_KEYS
        if unsupported:
            raise ValidationError(
                f"equity.ko_reset_snowball.mc_rqmc does not support "
                f"{sorted(unsupported)}. SnowballMCEngine builds its time grid "
                "from the observation schedule, so these would be banked as "
                "benchmark settings but never applied."
            )

    def config(self) -> Mapping[str, Any]:
        return {
            "engine": "SnowballMCEngine",
            "method": MonteCarloMethod.RANDOMIZED_QUASI.value,
            "paths_per_batch": self.sampling.paths_per_batch,
            "greeks": "paired central difference (common random numbers)",
        }

    def identity(self, case) -> Mapping[str, Any]:
        environment, product = self._specs(case)
        return {
            "builder": "equity.ko_reset_snowball.mc_rqmc",
            "case": case.name,
            "environment": environment,
            "product": product,
            "quantities": list(self.quantities),
            "params": dict(self._params),
            "config": dict(self.config()),
            "sampling": {
                "paths_per_batch": self.sampling.paths_per_batch,
                "min_batches": self.sampling.min_batches,
                "max_batches": self.sampling.max_batches,
                "seed": self.sampling.seed,
                "bump": self.sampling.bump,
            },
        }

    def run_batch(self, case, batch_index: int) -> BatchResult:
        environment, product_spec = self._specs(case)
        product = make_ko_reset(product_spec)
        seed = self.sampling.seed + batch_index

        def price_at(spot: float) -> float:
            engine = SnowballMCEngine(
                params=MCParams(
                    seed=seed,
                    num_paths=self.sampling.paths_per_batch,
                    use_qmc=True,
                    rqmc_min_batches=1,
                    rqmc_max_batches=1,
                    rqmc_paths_mode="per_batch",
                ),
                method=MonteCarloMethod.RANDOMIZED_QUASI,
            )
            return engine.price(product, make_environment(environment, spot))

        values = _central_difference_greeks(
            price_at, float(environment["spot"]), self.sampling.bump
        )
        return BatchResult(index=batch_index, seed=seed, values=values)


class KOResetPDECandidate(_KOResetArm):
    """Two-surface PDE solver; delta and gamma come from the solver directly."""

    def name(self) -> str:
        return "equity.ko_reset_snowball.pde"

    def params(self) -> Mapping[str, Any]:
        accuracy = str(self._params.get("accuracy", "standard"))
        return {
            **self._params,
            "engine": "KOResetSnowballPDESolver",
            "grid": engine_config(resolve_config(accuracy, None)),
        }

    def _greeks(self, case, accuracy: str) -> dict:
        environment, product_spec = self._specs(case)
        solver = KOResetSnowballPDESolver(params=PDEParams(accuracy=accuracy))
        result = solver.calculate_greeks(
            make_ko_reset(product_spec), make_environment(environment)
        )
        return {
            "pv": result["price"],
            "delta": result["delta"],
            "gamma": result["gamma"],
        }

    def evaluate(self, case) -> CandidateResult:
        accuracy = str(self._params.get("accuracy", "standard"))
        values = self._greeks(case, accuracy)
        coarser = _COARSER_ACCURACY[accuracy]
        rungs = [LadderRung(axis="accuracy", level="target", values=values)]
        if coarser != accuracy:
            rungs.append(
                LadderRung(axis="accuracy", level="medium", values=self._greeks(case, coarser))
            )
        return CandidateResult(values=values, ladders=tuple(rungs))


class KOResetQuadCandidate(_KOResetArm):
    """Quadrature engine; Greeks by central difference on the same bump width."""

    def name(self) -> str:
        return "equity.ko_reset_snowball.quad"

    def params(self) -> Mapping[str, Any]:
        grid_points = int(self._params.get("grid_points", 1001))
        return {
            **self._params,
            "engine": "KOResetSnowballQuadEngine",
            "grid": engine_config(
                QuadParams(grid_points=grid_points), exclude=_QUAD_NON_NUMERIC
            ),
        }

    def _greeks(self, case, grid_points: int) -> dict:
        environment, product_spec = self._specs(case)
        product = make_ko_reset(product_spec)
        bump = float(self._params.get("bump", 0.01))

        def price_at(spot: float) -> float:
            engine = KOResetSnowballQuadEngine(params=QuadParams(grid_points=grid_points))
            return engine.price(product, make_environment(environment, spot))

        return _central_difference_greeks(price_at, float(environment["spot"]), bump)

    def evaluate(self, case) -> CandidateResult:
        grid_points = int(self._params.get("grid_points", 1001))
        values = self._greeks(case, grid_points)
        medium_points = (grid_points - 1) // 2 + 1
        rungs = [LadderRung(axis="grid_points", level="target", values=values)]
        if medium_points >= 11 and medium_points != grid_points:
            rungs.append(
                LadderRung(
                    axis="grid_points", level="medium", values=self._greeks(case, medium_points)
                )
            )
        return CandidateResult(values=values, ladders=tuple(rungs))


@register_builder("equity.ko_reset_snowball.mc_rqmc", kind="reference")
def build_ko_reset_mc_reference(
    environment_params, product_params, sampling, quantities, params
) -> KOResetMCReference:
    return KOResetMCReference(
        sampling=sampling,
        environment_params=environment_params,
        product_params=product_params,
        quantities=quantities,
        params=params,
    )


@register_builder("equity.ko_reset_snowball.pde", kind="candidate")
def build_ko_reset_pde_candidate(
    environment_params, product_params, quantities, params
) -> KOResetPDECandidate:
    return KOResetPDECandidate(
        environment_params=environment_params,
        product_params=product_params,
        quantities=quantities,
        params=params,
    )


@register_builder("equity.ko_reset_snowball.quad", kind="candidate")
def build_ko_reset_quad_candidate(
    environment_params, product_params, quantities, params
) -> KOResetQuadCandidate:
    return KOResetQuadCandidate(
        environment_params=environment_params,
        product_params=product_params,
        quantities=quantities,
        params=params,
    )
