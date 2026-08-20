"""Snowball under Heston and Heston-SLV: the ADI 2D Greek certification.

This family covers the two-dimensional ADI solvers -- one spot axis and one
stochastic-variance axis -- certified for spot **delta and gamma** on a snowball
with a densely monitored (252-per-year) discrete knock-in.  It is the vol-model
counterpart of ``equity_snowball.py``, whose engines run under flat
Black-Scholes.

**This study is archived, not runnable end to end.**  Its stochastic benchmark
is not a plain paired-RQMC average: it is a multilevel control-variate telescope
on independent seed families, with exact conditional integration of the spot
factor, a pilot-frozen Neyman allocation and a substep-refinement bias envelope.
That machinery lives in the certification harness
(``example/mo_volmodels/16_adi_greek_certification.py`` and its stage-17
aggregate amendment), took 28.6 hours of held-out production sampling, and is
banked as evidence rather than reproduced here.  The reference builder below
therefore *declares* the benchmark and refuses to run it, which is the honest
failure: a simplified RQMC average would look runnable while certifying against
a different reference than the evidence describes.

What IS live here is the candidate arm, and that is the half anchors need.  The
deterministic engines are cheap -- about four minutes for all fourteen cells --
so ``assert_anchors`` re-runs them on every commit and fails the moment the
released solvers stop producing the numbers the banked evidence describes.

See ``docs/modelvalidation/certificates/adi2d-snowball-greeks/`` for the banked
evidence and its provenance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np

from quantark.asset.equity.engine.pde.snowball_vol_pde_solvers import (
    HestonSLVSnowballPDESolver,
    HestonSnowballPDESolver,
)
from quantark.asset.equity.param import PDEParams
from quantark.asset.equity.param.engine_params import BumpConfig
from quantark.asset.equity.product.option.snowball_config import (
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.param import (
    ContinuousDividendYield,
    FlatRateCurve,
    FlatVolSurface,
    SpotQuote,
)
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.exceptions import ValidationError
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.slv.leverage import LeverageSurface

from quantark.modelvalidation.candidate import CandidateResult, LadderRung
from quantark.modelvalidation.registry import register_builder
from quantark.modelvalidation.study import SamplingPolicy

#: The certification's valuation date. Fixed, so a re-run is not calendar-bound.
VALUATION_DATE = datetime(2026, 8, 3)

#: Spot bump for the certified central-difference exposure. The certification
#: also solved a 0.02 / 0.005 / 0.0025 ladder; 0.01 is the certified rung.
SPOT_BUMP = 0.01

_ENVIRONMENT_KEYS = ("spot", "rate", "div_yield", "asset_name")

_PRODUCT_KEYS = (
    "initial_price",
    "strike",
    "ko_barrier",
    "ki_barrier",
    "ko_rate",
    "rebate_rate",
    "maturity",
    "contract_multiplier",
    "ki_observations_per_year",
    "ko_first_observation_years",
    "ko_observation_interval_years",
    # Declared per case, never inferred from a case name: the near-KI cell
    # certifies the production Greek policy's own step density, so it solves a
    # different grid from every other cell and has to say so.
    "dense_ki_stencil",
)

_MODEL_KEYS = ("v0", "kappa", "theta", "sigma", "rho")

#: Engine controls the production Greek policy resolves to, as the
#: certification declared them. The three ``*_min_*`` floors are zeroed for a
#: certification solve so the declared grid is solved exactly rather than being
#: raised onto a policy floor -- otherwise two ladder rungs collapse onto one.
_PRODUCTION_ENGINE_CONTROLS: dict[str, Any] = {
    "grid_style": "concentrated",
    "v0_boundary": "degenerate_pde",
    "variance_grid_mode": "auto",
    "v_drift_scheme": "auto",
    "barrier_greek_steps_per_tick": 0,
    "greek_min_n_x": 0,
    "greek_min_n_v": 0,
    "greek_min_steps_per_year": 0,
    "barrier_greek_min_n_x": 0,
}

#: The leverage smile the SLV variant multiplies onto its Heston diffusion.
_LEVERAGE_STRIKES = (40.0, 60.0, 75.0, 90.0, 100.0, 103.0, 115.0, 140.0, 180.0)
_LEVERAGE_SMILE = (1.14, 1.10, 1.06, 1.02, 1.00, 0.99, 0.97, 0.96, 0.95)


@dataclass(frozen=True)
class _Grid:
    n_x: int
    n_v: int
    n_t: int

    def as_dict(self) -> dict:
        return {"n_x": self.n_x, "n_v": self.n_v, "n_t": self.n_t}


def target_grid(maturity: float, dense_ki_stencil: bool) -> _Grid:
    """The certified target grid for one case.

    Resolution is declared as a function of maturity rather than pinned per
    case, because a snowball's time axis has to track its own observation clock:
    a three-year trade at a fixed step count is a coarser solve than a
    three-month one.

    ``dense_ki_stencil`` is the near-KI case, where the production Greek policy's
    sixteen ADI steps per 252-clock KI tick is itself what gets certified, so the
    time axis is expressed in ticks and the spot axis is doubled to resolve a
    barrier the trade is sitting on.
    """
    if dense_ki_stencil:
        ticks = max(1, int(round(252.0 * maturity)))
        return _Grid(n_x=600, n_v=135, n_t=16 * ticks)
    return _Grid(n_x=300, n_v=135, n_t=max(180, int(math.ceil(1600 * maturity))))


def _coarse_time_grid(maturity: float, dense_ki_stencil: bool) -> _Grid:
    """The next-coarser time rung, for the refinement ladder."""
    target = target_grid(maturity, dense_ki_stencil)
    if dense_ki_stencil:
        ticks = max(1, int(round(252.0 * maturity)))
        return _Grid(target.n_x, target.n_v, 8 * ticks)
    return _Grid(
        target.n_x, target.n_v, max(120, int(math.ceil(800 * maturity)))
    )


@register_builder("equity.snowball.vol_dense_ki", kind="product")
def build_vol_snowball_product_spec(params: Mapping[str, Any]) -> dict:
    """Validate a densely-monitored-KI snowball spec."""
    spec = dict(params)
    unknown = set(spec) - set(_PRODUCT_KEYS)
    if unknown:
        raise ValidationError(
            f"Unknown equity.snowball.vol_dense_ki product keys: {sorted(unknown)}; "
            f"expected a subset of {_PRODUCT_KEYS}"
        )
    for key in ("initial_price", "strike", "ko_barrier", "ki_barrier", "maturity"):
        if key not in spec:
            raise ValidationError(
                f"equity.snowball.vol_dense_ki product is missing {key!r}"
            )
    if float(spec["ki_barrier"]) >= float(spec["ko_barrier"]):
        raise ValidationError(
            "ki_barrier must sit below ko_barrier for a standard snowball"
        )
    return spec


@register_builder("equity.snowball.heston_flat_market", kind="environment")
def build_heston_flat_market_spec(params: Mapping[str, Any]) -> dict:
    """Validate the flat rate/dividend market the vol model prices in.

    The vol surface is not an input here: under Heston (and Heston-SLV with a
    precomputed leverage surface) the diffusion comes from the model parameters,
    and the environment's flat vol only supplies the representative level the
    grid is built around.
    """
    spec = dict(params)
    unknown = set(spec) - set(_ENVIRONMENT_KEYS)
    if unknown:
        raise ValidationError(
            f"Unknown equity.snowball.heston_flat_market keys: {sorted(unknown)}; "
            f"expected a subset of {_ENVIRONMENT_KEYS}"
        )
    for key in ("spot", "rate", "div_yield"):
        if key not in spec:
            raise ValidationError(
                f"equity.snowball.heston_flat_market is missing {key!r}"
            )
    return spec


def make_snowball(spec: Mapping[str, Any]) -> SnowballOption:
    """The certified trade: monthly KO observations, a 252-per-year discrete KI.

    Both schedules are built from ONE integer clock. Near-equal but
    non-identical floats would create zero-length intervals after cumulative
    summation once a substep ladder is active, so the KO dates are selected
    *from* the KI grid rather than generated independently.
    """
    maturity = float(spec["maturity"])
    per_year = int(spec.get("ki_observations_per_year", 252))
    n_ki = max(1, int(math.ceil(per_year * maturity)))

    ki_times = np.arange(1, n_ki + 1, dtype=float) * maturity / n_ki
    ki_times[-1] = maturity

    first = float(spec.get("ko_first_observation_years", 0.25))
    interval = float(spec.get("ko_observation_interval_years", 1.0 / 12.0))
    first_ko = max(1, int(round(first * n_ki / maturity)))
    ko_step = max(1, int(round(interval * n_ki / maturity)))
    ko_indices = list(range(first_ko, n_ki + 1, ko_step))
    if not ko_indices or ko_indices[-1] != n_ki:
        ko_indices.append(n_ki)
    ko_times = ki_times[np.asarray(ko_indices, dtype=int) - 1]

    return SnowballOption(
        initial_price=float(spec["initial_price"]),
        strike=float(spec["strike"]),
        maturity=maturity,
        contract_multiplier=float(spec.get("contract_multiplier", 1.0)),
        is_reverse=False,
        payoff_config=PayoffConfig(
            include_principal=False,
            rebate_rate=float(spec.get("rebate_rate", 0.15)),
        ),
        barrier_config=BarrierConfig(
            ko_barrier=float(spec["ko_barrier"]),
            ko_rate=float(spec.get("ko_rate", 0.15)),
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[float(x) for x in ko_times],
            ki_barrier=float(spec["ki_barrier"]),
            ki_observation_type=ObservationType.DISCRETE,
            ki_observation_dates=[float(x) for x in ki_times],
            ki_continuous=False,
        ),
    )


def make_environment(
    spec: Mapping[str, Any], model: Mapping[str, Any]
) -> PricingEnvironment:
    """The market the case prices in, at a representative vol level.

    The representative level is ``sqrt(max(v0, theta))`` -- the model's own
    long-run scale -- so the grid is built around the variance the paths
    actually visit rather than around a quoted number that has nothing to do
    with this model.
    """
    representative_vol = math.sqrt(max(float(model["v0"]), float(model["theta"])))
    return PricingEnvironment(
        rate_curve=FlatRateCurve(float(spec["rate"])),
        valuation_date=VALUATION_DATE,
        spot_quote=SpotQuote(
            float(spec["spot"]), asset_name=str(spec.get("asset_name", "synthetic_index"))
        ),
        vol_surface=FlatVolSurface(representative_vol),
        div_yield=ContinuousDividendYield(float(spec["div_yield"])),
    )


def make_leverage_surface(maturity: float) -> LeverageSurface:
    """The SLV leverage surface: a downside-heavy smile decaying to unity."""
    strikes = np.array(_LEVERAGE_STRIKES, dtype=float)
    times = np.array([0.0, max(0.05, 0.5 * maturity), maturity], dtype=float)
    smile = np.array(_LEVERAGE_SMILE, dtype=float)
    grid = np.vstack([smile, 0.5 * (smile + 1.0), np.ones_like(smile)])
    return LeverageSurface(times, strikes, grid)


def _heston_params(model: Mapping[str, Any]) -> HestonParams:
    missing = [key for key in _MODEL_KEYS if key not in model]
    if missing:
        raise ValidationError(f"Heston model spec is missing {missing}")
    return HestonParams(
        v0=float(model["v0"]),
        kappa=float(model["kappa"]),
        theta=float(model["theta"]),
        sigma=float(model["sigma"]),
        rho=float(model["rho"]),
    )


def central_bump_greeks(engine, product, env) -> dict:
    """One frozen-domain central finite-bump exposure at the certified bump.

    The bump is written onto the engine rather than passed, because the solver
    reuses one spatial domain across the three spot solves: a frozen domain is
    what makes the difference a clean derivative of the same discretization
    instead of a difference between two grids.
    """
    engine.params.bump_config = BumpConfig(
        spot_bump=SPOT_BUMP, gamma_spot_bump=SPOT_BUMP
    )
    native = engine.calculate_greeks(product, env)
    values = {
        "delta": float(native["delta"]),
        "gamma": float(native["gamma"]),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValidationError("ADI central bump returned a non-finite result")
    return values


class _VolSnowballArm:
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
        unsupported = set(self.quantities) - {"delta", "gamma"}
        if unsupported:
            raise ValidationError(
                f"The ADI 2D Greek certification covers delta and gamma only; "
                f"got {sorted(unsupported)}. PV was certified separately by the "
                "stage-11 convergence gate, not by this study."
            )

    def _specs(self, case) -> tuple[dict, dict, dict]:
        environment = dict(self.environment_params)
        environment.update(case.environment_params)
        product = dict(self.product_params)
        product.update(case.product_params)

        # The Heston parameters are a per-case regime, not a study constant:
        # low_feller and sigma_collapse exist precisely to move them.
        model = dict(environment.pop("heston", {}))
        if not model:
            raise ValidationError(
                f"case {case.name!r} declares no 'heston' block; every case in "
                "this study selects its own variance regime"
            )
        build_heston_flat_market_spec(environment)
        build_vol_snowball_product_spec(product)
        return environment, product, model

    def _dense_ki_stencil(self, case) -> bool:
        """Declared by the case, never inferred from its name."""
        return bool(dict(case.product_params).get("dense_ki_stencil", False))


class VolSnowballExternalReference(_VolSnowballArm):
    """The benchmark this study was certified against -- declared, not run.

    A multilevel control-variate telescope on independent seed families, with
    exact conditional integration of the spot factor and a pilot-frozen Neyman
    allocation across cells. Reproducing it takes the certification harness and
    tens of hours; this class records what it was so the certificate is
    self-describing, and refuses to stand in for it.
    """

    def __init__(self, sampling: SamplingPolicy, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sampling = sampling

    def config(self) -> Mapping[str, Any]:
        return {
            "engine": "QESnowballMCEngine / HestonSLVQESnowballMCEngine",
            "method": "randomized_quasi (scrambled Sobol, Brownian bridge)",
            "variance_scheme": "QE-M (martingale-corrected quadratic exponential)",
            "estimator": (
                "multilevel control-variate telescope with exact conditional "
                "integration of the spot factor"
            ),
            "allocation": "pilot-frozen cost-weighted Neyman, no optional stopping",
            "greeks": "paired central difference (common random numbers)",
            "harness": "example/mo_volmodels/16_adi_greek_certification.py",
            "external": True,
        }

    def identity(self, case) -> Mapping[str, Any]:
        environment, product, model = self._specs(case)
        return {
            "builder": "equity.snowball.vol_multilevel_rqmc",
            "case": case.name,
            "environment": environment,
            "product": product,
            "heston": model,
            "quantities": list(self.quantities),
            "config": dict(self.config()),
        }

    def run_batch(self, case, batch_index: int):
        raise ValidationError(
            "equity.snowball.vol_multilevel_rqmc is an EXTERNAL benchmark: its "
            "evidence is banked under docs/modelvalidation/certificates/"
            "adi2d-snowball-greeks/, produced by "
            "example/mo_volmodels/16_adi_greek_certification.py over 28.6 hours "
            "of held-out sampling. Running this study end to end would certify "
            "against a different reference than the evidence describes. Use "
            "`assert_anchors` to check the deterministic arm, or re-run the "
            "harness to re-certify."
        )


class _ADICandidate(_VolSnowballArm):
    """A 2D ADI snowball solver at its certified grid."""

    variant: str = ""

    def _engine(self, model: Mapping[str, Any], grid: _Grid, maturity: float):
        raise NotImplementedError

    def params(self) -> Mapping[str, Any]:
        return {
            **self._params,
            "engine": self.variant,
            "spot_bump": SPOT_BUMP,
            "greeks": "central finite bump on a frozen spatial domain",
            "controls": dict(_PRODUCTION_ENGINE_CONTROLS),
            "grid_policy": {
                "n_x": 300,
                "n_v": 135,
                "steps_per_year": 1600,
                "min_n_t": 180,
                "dense_ki_stencil": {"n_x": 600, "steps_per_ki_tick": 16},
            },
        }

    def _greeks(self, case, grid: _Grid) -> dict:
        environment, product_spec, model = self._specs(case)
        maturity = float(product_spec["maturity"])
        engine = self._engine(model, grid, maturity)
        return central_bump_greeks(
            engine, make_snowball(product_spec), make_environment(environment, model)
        )

    def evaluate(self, case) -> CandidateResult:
        dense = self._dense_ki_stencil(case)
        product = dict(self.product_params)
        product.update(case.product_params)
        maturity = float(product["maturity"])

        target = target_grid(maturity, dense)
        values = self._greeks(case, target)
        rungs = [LadderRung(axis="n_t", level="target", values=values)]

        # Only the time axis is laddered here. The certification's full ladder
        # also coarsens n_x and n_v, and its banked envelope is the sum of all
        # three; re-running the spatial rungs would triple the anchor's cost for
        # a figure the evidence already carries.
        coarse_t = _coarse_time_grid(maturity, dense)
        if coarse_t.n_t != target.n_t:
            rungs.append(
                LadderRung(axis="n_t", level="medium", values=self._greeks(case, coarse_t))
            )
        return CandidateResult(values=values, ladders=tuple(rungs))


class HestonADICandidate(_ADICandidate):
    """Heston ADI: one spot axis, one stochastic-variance axis."""

    variant = "HestonSnowballPDESolver"

    def name(self) -> str:
        return "equity.snowball.heston_pde"

    def _engine(self, model, grid, maturity):
        return HestonSnowballPDESolver(
            _heston_params(model),
            n_x=grid.n_x,
            n_v=grid.n_v,
            n_t=grid.n_t,
            params=PDEParams(cache_enabled=False),
            **_PRODUCTION_ENGINE_CONTROLS,
        )


class HestonSLVADICandidate(_ADICandidate):
    """Heston-SLV ADI: the same operator with a leverage surface on the spot leg."""

    variant = "HestonSLVSnowballPDESolver"

    def name(self) -> str:
        return "equity.snowball.heston_slv_pde"

    def _engine(self, model, grid, maturity):
        return HestonSLVSnowballPDESolver(
            _heston_params(model),
            leverage_surface=make_leverage_surface(maturity),
            n_x=grid.n_x,
            n_v=grid.n_v,
            n_t=grid.n_t,
            params=PDEParams(cache_enabled=False),
            **_PRODUCTION_ENGINE_CONTROLS,
        )


@register_builder("equity.snowball.vol_multilevel_rqmc", kind="reference")
def build_vol_snowball_reference(
    environment_params: Mapping[str, Any],
    product_params: Mapping[str, Any],
    sampling: SamplingPolicy,
    quantities: Sequence[str],
    params: Mapping[str, Any],
) -> VolSnowballExternalReference:
    return VolSnowballExternalReference(
        sampling=sampling,
        environment_params=environment_params,
        product_params=product_params,
        quantities=quantities,
        params=params,
    )


@register_builder("equity.snowball.heston_pde", kind="candidate")
def build_heston_adi_candidate(
    environment_params: Mapping[str, Any],
    product_params: Mapping[str, Any],
    quantities: Sequence[str],
    params: Mapping[str, Any],
) -> HestonADICandidate:
    return HestonADICandidate(
        environment_params=environment_params,
        product_params=product_params,
        quantities=quantities,
        params=params,
    )


@register_builder("equity.snowball.heston_slv_pde", kind="candidate")
def build_heston_slv_adi_candidate(
    environment_params: Mapping[str, Any],
    product_params: Mapping[str, Any],
    quantities: Sequence[str],
    params: Mapping[str, Any],
) -> HestonSLVADICandidate:
    return HestonSLVADICandidate(
        environment_params=environment_params,
        product_params=product_params,
        quantities=quantities,
        params=params,
    )
