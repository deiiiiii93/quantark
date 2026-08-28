"""Snowball under a real calibrated Dupire local-volatility surface.

The 1-D local-vol counterpart of ``equity_snowball.py``. One deterministic
engine -- ``LocalVolSnowballPDESolver`` -- is certified against a local-vol
Monte-Carlo reference on two real CSI1000 surfaces: the 2024-02-08 crash bottom
and the calm 2023-11-15, chosen as the cohort's steepest and flattest by Dupire
local-vol slope.

Two things differ from the flat-BSM study, and both are deliberate.

**The surfaces are committed data, not generated.** ``example/mo_volmodels/data/
history`` is excluded through ``.git/info/exclude``, a per-clone file that is
never pushed, so a study reading from it would bank a certificate whose CI
anchors fail everywhere but one machine. The artifacts live under
``example/modelvalidation/data/`` and their sha256 enters the identity hash, so
the certificate pins the exact surface bytes it was certified against.

**Levels are declared as moneyness and resolved against each artifact's own
s0.** The two surfaces sit at different index levels (4993.105 and 6207.268), and
``economic_scale`` is a single study-level block. Resolving here lets one set of
case shapes serve both surfaces, and lets ``contract_multiplier`` be *computed*
as ``REFERENCE_SPOT / s0`` rather than transcribed -- uncorrected, every
calm-surface error would be overstated by 1.243x, which risks a false REJECTED
rather than a merely conservative pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from quantark.asset.equity.engine.mc.snowball_vol_mc_engines import (
    LocalVolSnowballMCEngine,
)
from quantark.asset.equity.engine.pde.grid.config import resolve_config
from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    LocalVolSnowballPDESolver,
)
from quantark.asset.equity.param import MCParams, PDEParams
from quantark.param import FlatRateCurve, GridVolSurface, SpotQuote
from quantark.param.div import TermStructureDividendYield
from quantark.param.vol.surface_history import IvSurfaceArtifact
from quantark.priceenv import PricingEnvironment
from quantark.util.enum.engine_enums import MonteCarloMethod
from quantark.util.exceptions import ValidationError
from quantark.volmodels.localvol import LocalVolSurface, build_dupire_local_vol

from quantark.modelvalidation.builders.equity_snowball import (
    _central_difference_greeks,
    build_snowball_product_spec,
    make_snowball,
)
from quantark.modelvalidation.candidate import CandidateResult, LadderRung
from quantark.modelvalidation.engine_config import engine_config
from quantark.modelvalidation.reference import BatchResult
from quantark.modelvalidation.registry import register_builder
from quantark.modelvalidation.study import SamplingPolicy

#: The economic-scale basis. ``notional = 200 * REFERENCE_SPOT`` makes
#: ``delta_quantum`` exactly 1.0, matching the flat-BSM study's normalization,
#: so raw delta reads directly as hedge contracts on both certificates.
REFERENCE_SPOT = 4993.105

#: Repo root, for resolving study-relative artifact paths. The study is run from
#: the repo root, but anchors replay from pytest, so this is derived from the
#: module location rather than from the process working directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]

_ENVIRONMENT_KEYS = ("surface", "rate", "spot_moneyness", "asset_name")

#: Product keys this study accepts. Levels are MONEYNESS; everything else is
#: passed through to the flat-BSM product builder unchanged.
_PRODUCT_KEYS = (
    "strike_moneyness",
    "ko_barrier_moneyness",
    "ki_barrier_moneyness",
    "ko_rate",
    "rebate_rate",
    "months",
    "maturity",
    "ki_monitoring",
    "ko_stepdown",
)

#: One profile coarser than each target, for the refinement ladder.
_COARSER_ACCURACY = {"high": "standard", "standard": "fast", "fast": "fast"}


@dataclass(frozen=True)
class _Surface:
    """A loaded artifact and everything derived from it, built once."""

    artifact: IvSurfaceArtifact
    grid: GridVolSurface
    rate_curve: FlatRateCurve
    div_yield: TermStructureDividendYield
    local_vol: LocalVolSurface


def _resolve_path(surface: str) -> Path:
    path = Path(surface)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    if not path.is_file():
        raise ValidationError(
            f"equity.snowball.localvol_market surface artifact not found: {path}. "
            "Study surfaces live under example/modelvalidation/data/ and are "
            "committed; they are NOT read from example/mo_volmodels/data/history, "
            "which is excluded per-clone and would not exist in CI."
        )
    return path


@lru_cache(maxsize=8)
def load_surface(surface: str, rate: float) -> _Surface:
    """Load an artifact and build its Dupire surface, once per (path, rate).

    The local-vol surface is built at the artifact's OWN s0 and reused across
    spot bumps. Rebuilding it at a bumped spot would make delta a derivative of
    the surface construction as well as of the price.

    The stored ``iv_grid`` is already SABR-smoothed and calendar-projected, so
    it is used as-is: Dupire differentiates total variance twice in strike, and
    smoothing a second time would certify a different surface from the one the
    artifact names.
    """
    artifact = IvSurfaceArtifact.from_file(_resolve_path(surface))
    grid = artifact.grid_vol_surface()
    rate_curve = FlatRateCurve(float(rate))
    div_yield = artifact.term_structure_dividend_yield(float(rate))
    local_vol = build_dupire_local_vol(
        grid,
        spot=float(artifact.s0),
        rate_curve=rate_curve,
        div_yield=div_yield.get_yield,
    )
    return _Surface(artifact, grid, rate_curve, div_yield, local_vol)


@register_builder("equity.snowball.localvol_market", kind="environment")
def build_localvol_market_spec(params: Mapping[str, Any]) -> dict:
    """Validate a local-vol market spec."""
    spec = dict(params)
    unknown = set(spec) - set(_ENVIRONMENT_KEYS)
    if unknown:
        raise ValidationError(
            f"Unknown equity.snowball.localvol_market keys: {sorted(unknown)}; "
            f"expected a subset of {_ENVIRONMENT_KEYS}"
        )
    for key in ("surface", "rate"):
        if key not in spec:
            raise ValidationError(
                f"equity.snowball.localvol_market is missing {key!r}"
            )
    moneyness = float(spec.get("spot_moneyness", 1.0))
    if not moneyness > 0.0:
        raise ValidationError(f"spot_moneyness must be positive, got {moneyness}")
    return spec


def make_localvol_environment(
    spec: Mapping[str, Any], spot: float | None = None
) -> PricingEnvironment:
    """The market one case prices in, optionally at a bumped spot.

    ``valuation_date`` is the artifact's own trade date, so the study is not
    calendar-bound and the surface's maturities mean what they say.
    """
    surface = load_surface(str(spec["surface"]), float(spec["rate"]))
    resolved = (
        float(spec.get("spot_moneyness", 1.0)) * float(surface.artifact.s0)
        if spot is None
        else float(spot)
    )
    trade_date = surface.artifact.trade_date
    return PricingEnvironment(
        rate_curve=surface.rate_curve,
        valuation_date=datetime(trade_date.year, trade_date.month, trade_date.day),
        spot_quote=SpotQuote(
            resolved, asset_name=str(spec.get("asset_name", "CSI1000"))
        ),
        vol_surface=surface.grid,
        div_yield=surface.div_yield,
    )


def resolve_product_spec(
    environment: Mapping[str, Any], product: Mapping[str, Any]
) -> dict:
    """Turn moneyness-declared levels into the absolute spec make_snowball wants.

    ``contract_multiplier`` is COMPUTED, not declared: both surfaces carry the
    same economic notional expressed at their own index level, which is what
    keeps the study-level economic scale honest across two index levels.
    """
    unknown = set(product) - set(_PRODUCT_KEYS)
    if unknown:
        raise ValidationError(
            f"Unknown equity.snowball.localvol product keys: {sorted(unknown)}; "
            f"expected a subset of {_PRODUCT_KEYS}"
        )
    for key in (
        "strike_moneyness",
        "ko_barrier_moneyness",
        "ki_barrier_moneyness",
        "months",
        "maturity",
    ):
        if key not in product:
            raise ValidationError(
                f"equity.snowball.localvol product is missing {key!r}"
            )
    s0 = float(
        load_surface(
            str(environment["surface"]), float(environment["rate"])
        ).artifact.s0
    )
    spec: dict[str, Any] = {
        "initial_price": s0,
        "strike": float(product["strike_moneyness"]) * s0,
        "ko_barrier": float(product["ko_barrier_moneyness"]) * s0,
        "ki_barrier": float(product["ki_barrier_moneyness"]) * s0,
        "months": int(product["months"]),
        "maturity": float(product["maturity"]),
        "contract_multiplier": REFERENCE_SPOT / s0,
    }
    for key in ("ko_rate", "rebate_rate", "ki_monitoring", "ko_stepdown"):
        if key in product:
            spec[key] = product[key]
    return build_snowball_product_spec(spec)


class _LocalVolArm:
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
        build_localvol_market_spec(environment)
        return environment, resolve_product_spec(environment, product)

    def _surface(self, environment: Mapping[str, Any]) -> _Surface:
        return load_surface(str(environment["surface"]), float(environment["rate"]))


class LocalVolPDECandidate(_LocalVolArm):
    """The 1-D two-surface local-vol PDE solver at a declared accuracy profile."""

    def name(self) -> str:
        return "equity.snowball.localvol_pde"

    def params(self) -> Mapping[str, Any]:
        """Declared settings plus the grid the accuracy profile resolves to.

        Recording the resolved grid rather than the profile name is what makes
        the certificate self-describing, and what makes the identity hash move
        if a future release redefines the profile.
        """
        accuracy = str(self._params.get("accuracy", "standard"))
        return {
            **self._params,
            "engine": "LocalVolSnowballPDESolver",
            "grid": engine_config(resolve_config(accuracy, None)),
        }

    def _greeks(self, case, accuracy: str) -> dict:
        environment, product_spec = self._specs(case)
        surface = self._surface(environment)
        solver = LocalVolSnowballPDESolver(
            params=PDEParams(accuracy=accuracy),
            local_vol_surface=surface.local_vol,
        )
        result = solver.calculate_greeks(
            make_snowball(product_spec), make_localvol_environment(environment)
        )
        return {
            "pv": result["price"],
            "delta": result["delta"],
            "gamma": result["gamma"],
        }

    def evaluate(self, case) -> CandidateResult:
        accuracy = str(self._params.get("accuracy", "standard"))
        values = self._greeks(case, accuracy)
        rungs = [LadderRung(axis="accuracy", level="target", values=values)]
        coarser = _COARSER_ACCURACY[accuracy]
        if coarser != accuracy:
            rungs.append(
                LadderRung(
                    axis="accuracy",
                    level="medium",
                    values=self._greeks(case, coarser),
                )
            )
        return CandidateResult(values=values, ladders=tuple(rungs))


#: Knobs the local-vol benchmark can actually honour. Anything else would be
#: banked as a benchmark setting and folded into the identity hash while moving
#: no number, so it is refused rather than ignored.
_REFERENCE_KEYS = frozenset(
    {"substeps_per_interval", "lv_time_sampling", "estimator"}
)

#: The discretization FINDING-2026-08-26 section 5 demonstrated the estimate
#: stops moving at: substeps 8 and 16 differ by 0.04 sigma. PV converged at 2,
#: but delta had not converged at 4 -- reading "PV is flat under refinement" as
#: "the reference is converged" is the inference that produced the defect.
_DEFAULT_SUBSTEPS = 8

#: Exact per-step time-averaged variance instead of the left-endpoint sigma
#: freeze. Exact on time-only surfaces; removes a measured -1.26c daily-grid
#: bias at zero per-step cost (docs/lv-mc-scheme-demos/RESULTS.md).
_DEFAULT_TIME_SAMPLING = "integrated"


class LocalVolMCReference(_LocalVolArm):
    """Paired local-vol MC benchmark: one randomization per batch, shared bumps.

    ONE discretization serves pv, delta and gamma. Running PV at one substep
    level and Greeks at another estimates P(h) and P(h/2) -- different numbers at
    finite h -- so the certified delta would not be the derivative of the
    certified price.
    """

    def __init__(self, sampling: SamplingPolicy, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sampling = sampling
        unsupported = set(self._params) - _REFERENCE_KEYS
        if unsupported:
            raise ValidationError(
                f"equity.snowball.localvol_mc does not support "
                f"{sorted(unsupported)}. Supported knobs: {sorted(_REFERENCE_KEYS)}."
            )
        self.substeps = int(
            self._params.get("substeps_per_interval", _DEFAULT_SUBSTEPS)
        )
        self.time_sampling = str(
            self._params.get("lv_time_sampling", _DEFAULT_TIME_SAMPLING)
        )
        self.estimator = str(self._params.get("estimator", "plain"))
        # one_step_survival rejects RANDOMIZED_QUASI (its control-variate
        # machinery assumes the plain estimator), so an OSS arm runs QUASI.
        # Both scramble: _qmc_normals always builds Sobol(scramble=True,
        # seed=base_seed + batch_id), so batches stay independent and the
        # batch-to-batch standard error remains valid.
        self.method = (
            MonteCarloMethod.QUASI
            if self.estimator == "one_step_survival"
            else MonteCarloMethod.RANDOMIZED_QUASI
        )

    def config(self) -> Mapping[str, Any]:
        """The benchmark's own settings -- it is half of every comparison."""
        return {
            "engine": "LocalVolSnowballMCEngine",
            "method": self.method.value,
            "substeps_per_interval": self.substeps,
            "lv_time_sampling": self.time_sampling,
            "estimator": self.estimator,
            "paths_per_batch": self.sampling.paths_per_batch,
            "greeks": "paired central difference (common random numbers)",
        }

    def identity(self, case) -> Mapping[str, Any]:
        environment, product = self._specs(case)
        return {
            "builder": "equity.snowball.localvol_mc",
            "case": case.name,
            "environment": environment,
            "product": product,
            "surface_sha256": self._surface(environment).artifact.sha256,
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
        surface = self._surface(environment)
        product = make_snowball(product_spec)
        seed = self.sampling.seed + batch_index

        def price_at(spot: float) -> float:
            # A fresh engine per pricing call: engine instances are not safe to
            # reuse across calls, and the shared seed is what pairs the three
            # bump arms onto one set of paths. The local-vol surface is the SAME
            # object across bumps -- it is built at the artifact spot, never
            # rebuilt at a bumped one.
            engine = LocalVolSnowballMCEngine(
                local_vol_surface=surface.local_vol,
                params=MCParams(
                    seed=seed,
                    num_paths=self.sampling.paths_per_batch,
                    use_qmc=True,
                    rqmc_min_batches=1,
                    rqmc_max_batches=1,
                    rqmc_paths_mode="per_batch",
                ),
                method=self.method,
                substeps_per_interval=self.substeps,
                lv_time_sampling=self.time_sampling,
                estimator=self.estimator,
            )
            return engine.price(product, make_localvol_environment(environment, spot))

        base_spot = float(environment.get("spot_moneyness", 1.0)) * float(
            surface.artifact.s0
        )
        values = _central_difference_greeks(price_at, base_spot, self.sampling.bump)
        return BatchResult(index=batch_index, seed=seed, values=values)


@register_builder("equity.snowball.localvol_mc", kind="reference")
def build_localvol_mc_reference(
    environment_params: Mapping[str, Any],
    product_params: Mapping[str, Any],
    sampling: SamplingPolicy,
    quantities: Sequence[str],
    params: Mapping[str, Any],
) -> LocalVolMCReference:
    return LocalVolMCReference(
        sampling=sampling,
        environment_params=environment_params,
        product_params=product_params,
        quantities=quantities,
        params=params,
    )


@register_builder("equity.snowball.localvol_pde", kind="candidate")
def build_localvol_pde_candidate(
    environment_params: Mapping[str, Any],
    product_params: Mapping[str, Any],
    quantities: Sequence[str],
    params: Mapping[str, Any],
) -> LocalVolPDECandidate:
    return LocalVolPDECandidate(
        environment_params=environment_params,
        product_params=product_params,
        quantities=quantities,
        params=params,
    )
