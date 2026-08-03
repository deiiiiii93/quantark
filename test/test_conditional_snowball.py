from datetime import datetime

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from quantark.asset.equity.engine.mc.snowball_mc_engine import SnowballMCEngine
from quantark.asset.equity.engine.mc.term_inputs import make_df_fn
from quantark.asset.equity.param import MCParams
from quantark.asset.equity.product.option.snowball_config import (
    BarrierConfig,
    PayoffConfig,
)
from quantark.asset.equity.product.option.snowball_option import SnowballOption
from quantark.montecarlo.conditional_snowball import (
    conditional_standard_snowball_moments,
)
from quantark.param import FlatRateCurve, FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.exceptions import ValidationError


def _environment() -> PricingEnvironment:
    return PricingEnvironment(
        rate_curve=FlatRateCurve(0.02),
        valuation_date=datetime(2026, 8, 3),
        spot_quote=SpotQuote(100.0),
        vol_surface=FlatVolSurface(0.20),
        div_yield=ContinuousDividendYield(0.01),
    )


def _product(*, continuous_ki: bool = False) -> SnowballOption:
    return SnowballOption(
        initial_price=100.0,
        strike=100.0,
        maturity=1.0,
        contract_multiplier=1.0,
        is_reverse=False,
        payoff_config=PayoffConfig(include_principal=False, rebate_rate=0.15),
        barrier_config=BarrierConfig(
            ko_barrier=[104.0, 103.0],
            ko_rate=[0.12, 0.15],
            ko_observation_type=ObservationType.DISCRETE,
            ko_observation_dates=[0.5, 1.0],
            ki_barrier=(75.0 if continuous_ki else [76.0, 75.0]),
            ki_observation_type=(
                ObservationType.CONTINUOUS
                if continuous_ki
                else ObservationType.DISCRETE
            ),
            ki_observation_dates=(None if continuous_ki else [0.5, 1.0]),
            ki_continuous=continuous_ki,
        ),
    )


def test_exact_affine_payoff_matches_native_kernel_quadrature():
    product = _product()
    env = _environment()
    base = np.array(
        [
            [100.0, 96.0, 91.0],
            [100.0, 101.0, 108.0],
        ]
    )
    loadings = np.array(
        [
            [0.0, 0.11, 0.19],
            [0.0, 0.09, 0.17],
        ]
    )
    all_times = np.array([0.5, 1.0])
    ko_indices = np.array([0, 1])
    ki_indices = np.array([0, 1])
    discount = make_df_fn(env)

    moments = conditional_standard_snowball_moments(
        product=product,
        pricing_env=env,
        base_paths=base,
        log_spot_factor_loadings=loadings,
        ko_indices=ko_indices,
        ki_indices=ki_indices,
        maturity=1.0,
        discount_factors=discount,
    )

    native = SnowballMCEngine(MCParams(num_paths=8, seed=17))
    ko_barriers = np.asarray(
        product.get_ko_observation_profile(env)["barriers"], dtype=float
    )
    ki_barriers = np.asarray(
        product.get_ki_observation_profile(env)["barriers"], dtype=float
    )
    for row in range(base.shape[0]):
        split_points = []
        for j, column in enumerate(ko_indices + 1):
            split_points.append(
                (np.log(ko_barriers[j]) - np.log(base[row, column]))
                / loadings[row, column]
            )
        for j, column in enumerate(ki_indices + 1):
            split_points.append(
                (np.log(ki_barriers[j]) - np.log(base[row, column]))
                / loadings[row, column]
            )
        split_points.append(
            (np.log(product.strike) - np.log(base[row, -1]))
            / loadings[row, -1]
        )
        boundaries = [-np.inf, *sorted(set(split_points)), np.inf]

        def discounted_native_payoff(z):
            path = base[row : row + 1] * np.exp(loadings[row : row + 1] * z)
            payoffs, settlement_times, _ = native._compute_payoffs(
                product,
                env,
                path,
                all_times,
                ko_indices,
                ki_indices,
                r=0.02,
                T=1.0,
                sigma=0.20,
                rng_seed=99,
            )
            return float(payoffs[0] * discount(settlement_times)[0])

        expected = 0.0
        for lower, upper in zip(boundaries[:-1], boundaries[1:]):
            integral, _ = quad(
                lambda z: discounted_native_payoff(z) * norm.pdf(z),
                lower,
                upper,
                epsabs=1e-12,
                epsrel=1e-12,
            )
            expected += integral
        assert moments.discounted_payoff[row] == pytest.approx(
            expected, abs=2e-11
        )


def test_affine_payoff_fails_closed_for_continuous_ki():
    product = _product(continuous_ki=True)
    with pytest.raises(ValidationError, match="discrete KI"):
        conditional_standard_snowball_moments(
            product=product,
            pricing_env=_environment(),
            base_paths=np.array([[100.0, 100.0, 100.0]]),
            log_spot_factor_loadings=np.array([[0.0, 0.1, 0.2]]),
            ko_indices=np.array([0, 1]),
            ki_indices=np.array([0, 1]),
            maturity=1.0,
            discount_factors=lambda t: np.exp(-0.02 * np.asarray(t)),
        )


@pytest.mark.parametrize(
    ("base", "loadings", "ko_indices", "message"),
    [
        (
            np.array([[100.0, 100.0, 100.0]]),
            np.array([[0.0, 0.1]]),
            np.array([0, 1]),
            "same-shaped 2D",
        ),
        (
            np.array([[100.0, 100.0, 100.0]]),
            np.array([[0.0, -0.1, 0.2]]),
            np.array([0, 1]),
            "finite and non-negative",
        ),
        (
            np.array([[100.0, 100.0, 100.0]]),
            np.array([[0.0, 0.0, 0.2]]),
            np.array([0, 1]),
            "positive loading",
        ),
        (
            np.array([[100.0, 100.0, 100.0]]),
            np.array([[0.0, 0.1, 0.2]]),
            np.array([0, 2]),
            "outside conditional path columns",
        ),
    ],
)
def test_affine_payoff_rejects_malformed_path_evidence(
    base, loadings, ko_indices, message
):
    with pytest.raises(ValidationError, match=message):
        conditional_standard_snowball_moments(
            product=_product(),
            pricing_env=_environment(),
            base_paths=base,
            log_spot_factor_loadings=loadings,
            ko_indices=ko_indices,
            ki_indices=np.array([0, 1]),
            maturity=1.0,
            discount_factors=lambda t: np.exp(-0.02 * np.asarray(t)),
        )
