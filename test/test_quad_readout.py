"""The QUAD readout mode: how the price is recovered from the nodal values.

The engine diffuses the value surface back to the valuation date on a
log-moneyness grid and then has to report ONE number, at the spot. The
legacy rule interpolates linearly between the two straddling nodes. Because
barrier alignment pins the grid to the barrier rather than to spot, that
makes the price a piecewise-linear function of ``log S`` and its spot
derivative a staircase.

``readout="transition"`` instead evaluates the final backward transition
operator AT the spot, which is what ``QuadratureCore._calculate_final_value``
already does for the product-agnostic path.

Evidence for the defect and the remedy: docs/bucket-futures-hedge/quad-readout/.
"""

from datetime import datetime

import numpy as np
import pytest

from quantark.asset.equity.engine.quad.ko_reset_snowball_quad_engine import (
    KOResetSnowballQuadEngine,
)
from quantark.asset.equity.engine.quad.phoenix_quad_engine import PhoenixQuadEngine
from quantark.asset.equity.engine.quad.snowball_quad_engine import SnowballQuadEngine
from quantark.asset.equity.param import QuadParams
from quantark.asset.equity.product.option.snowball_helpers import (
    create_standard_snowball,
)
from quantark.param import FlatVolSurface, SpotQuote
from quantark.param.div import ContinuousDividendYield
from quantark.param.rrf import FlatRateCurve
from quantark.priceenv import PricingEnvironment
from quantark.util.enum import ObservationType
from quantark.util.exceptions import ValidationError

SPOT = 100.0
KI = 75.0


def _env(spot: float = SPOT) -> PricingEnvironment:
    return PricingEnvironment(
        spot_quote=SpotQuote(spot=float(spot)),
        vol_surface=FlatVolSurface(volatility=0.22),
        rate_curve=FlatRateCurve(rate=0.02),
        div_yield=ContinuousDividendYield(div_yield=0.03),
        valuation_date=datetime(2026, 1, 2),
    )


def _product():
    """A snowball whose KI barrier is the alignment target.

    KI is monitored DISCRETELY: continuous monitoring takes the bridged
    transition, which the pointwise readout refuses.
    """
    return create_standard_snowball(
        initial_price=SPOT,
        strike=SPOT,
        maturity=1.0,
        num_observations=6,
        ko_rate=0.10,
        ki_barrier=KI,
        include_principal=False,
        ki_continuous=False,
        ki_observation_type=ObservationType.DISCRETE,
        ki_observation_dates=[(i + 1) / 12.0 for i in range(12)],
    )


def _engine(readout: str | None = None, grid_points: int = 301):
    kwargs = {"grid_points": grid_points}
    if readout is not None:
        kwargs["readout"] = readout
    return SnowballQuadEngine(params=QuadParams(**kwargs))


def _cell_width(engine, product, spot: float) -> float:
    """Grid spacing at ``spot``, in index points."""
    engine.price(product, _env(spot))
    grid_spots, _ = engine._last_spot_greeks_grid
    j = int(np.argmin(np.abs(grid_spots - spot)))
    return float(grid_spots[j + 1] - grid_spots[j])


def test_the_default_readout_is_the_legacy_linear_one():
    """Existing callers and every stored golden keep the price they had."""
    product = _product()
    assert QuadParams().readout == "legacy_linear"
    default = _engine().price(product, _env())
    explicit = _engine("legacy_linear").price(product, _env())
    assert default == explicit


def test_an_unknown_readout_is_refused():
    with pytest.raises(ValidationError, match="readout"):
        QuadParams(readout="cubic-ish")


def test_the_legacy_price_is_the_linear_interpolant_of_its_own_nodes():
    """Pins the defect: today's price carries no information between nodes."""
    engine = _engine("legacy_linear")
    product = _product()
    engine.price(product, _env())
    grid_spots, _ = engine._last_spot_greeks_grid
    j = int(np.argmin(np.abs(grid_spots - SPOT)))
    lo, hi = float(grid_spots[j]), float(grid_spots[j + 1])

    node_lo = engine.price(product, _env(lo))
    node_hi = engine.price(product, _env(hi))
    mid = float(np.exp(0.5 * (np.log(lo) + np.log(hi))))
    chord = 0.5 * (node_lo + node_hi)

    assert engine.price(product, _env(mid)) == pytest.approx(chord, rel=1e-10)


@pytest.mark.parametrize("fft_filter_alpha", [12.0, 0.0])
@pytest.mark.parametrize("fft_padding_factor", [2, 1])
def test_at_a_node_the_transition_readout_reproduces_the_diffusion(
    fft_filter_alpha, fft_padding_factor
):
    """The two rules must agree exactly where the legacy one is exact.

    This is the correctness check: the transition readout has to BE the
    operator the sweep applies, not a better-behaved substitute for it. It
    catches the spectral filter in particular, which the FFT convolution
    applies and a naive pointwise sum would silently omit.
    """
    kwargs = dict(
        grid_points=301,
        fft_filter_alpha=fft_filter_alpha,
        fft_padding_factor=fft_padding_factor,
    )
    legacy = SnowballQuadEngine(QuadParams(readout="legacy_linear", **kwargs))
    transition = SnowballQuadEngine(QuadParams(readout="transition", **kwargs))
    product = _product()

    legacy.price(product, _env())
    grid_spots, _ = legacy._last_spot_greeks_grid
    node = float(grid_spots[int(np.argmin(np.abs(grid_spots - SPOT)))])

    on_node = legacy.price(product, _env(node))
    assert transition.price(product, _env(node)) == pytest.approx(
        on_node, rel=0.0, abs=1e-12
    )


def test_the_transition_price_is_not_the_linear_interpolant():
    """The remedy's defining behaviour, stated as the negation of the defect."""
    engine = _engine("transition")
    product = _product()
    engine.price(product, _env())
    grid_spots, _ = engine._last_spot_greeks_grid
    j = int(np.argmin(np.abs(grid_spots - SPOT)))
    lo, hi = float(grid_spots[j]), float(grid_spots[j + 1])

    node_lo = engine.price(product, _env(lo))
    node_hi = engine.price(product, _env(hi))
    mid = float(np.exp(0.5 * (np.log(lo) + np.log(hi))))
    chord = 0.5 * (node_lo + node_hi)
    curved = engine.price(product, _env(mid))

    # The value function is convex here, so the true price sits off the chord
    # by a definite amount rather than merely differing in the last bits.
    assert abs(curved - chord) > 1e-6 * abs(chord)


def test_the_transition_readout_removes_the_delta_staircase():
    """A sub-cell bump must give the same delta wherever it is taken.

    Delta genuinely varies across a cell, so the raw spread of a sub-cell
    finite difference is mostly curvature. The staircase is what is left
    once that smooth trend is removed.
    """
    product = _product()

    def staircase(readout: str) -> float:
        engine = _engine(readout)
        cell = _cell_width(engine, product, SPOT)
        bump = 0.02 * cell
        spots = np.linspace(SPOT - 0.4 * cell, SPOT + 0.4 * cell, 9)
        deltas = np.array(
            [
                (
                    engine.price(product, _env(s + bump))
                    - engine.price(product, _env(s - bump))
                )
                / (2.0 * bump)
                for s in spots
            ]
        )
        trend = np.polyval(np.polyfit(spots, deltas, 1), spots)
        return float(np.ptp(deltas - trend))

    assert staircase("transition") < 0.05 * staircase("legacy_linear")


def test_the_two_readouts_agree_to_within_a_few_basis_points():
    """The remedy corrects an interpolation, so it must not move the price far.

    Measured against the contract's notional: a snowball at fair value prices
    near zero, so a ratio to the price itself says nothing.
    """
    product = _product()
    notional = product.contract_multiplier * SPOT
    legacy = _engine("legacy_linear").price(product, _env())
    transition = _engine("transition").price(product, _env())
    assert transition != legacy
    assert abs(transition - legacy) < 1e-3 * notional


def test_the_mode_governs_the_event_decomposition_too():
    """Not just the price.

    The event decomposition is read off the same surfaces by the same rule.
    Were the mode wired to the price alone, the parts would still come from
    the linear readout and would no longer reconcile with the whole they are
    reported beside.
    """
    product = _product()
    legacy = _engine("legacy_linear").calculate_event_stats(product, _env())
    transition = _engine("transition").calculate_event_stats(product, _env())
    assert legacy is not None and transition is not None

    assert transition.pv != legacy.pv
    assert float(np.sum(transition.expected_discounted_ko_cashflow)) != float(
        np.sum(legacy.expected_discounted_ko_cashflow)
    )
    assert float(np.sum(transition.ko_probability)) != float(
        np.sum(legacy.ko_probability)
    )
    # and the decomposition stays a probability distribution
    assert 0.0 <= float(np.sum(transition.ko_probability)) <= 1.0


@pytest.mark.parametrize(
    "engine_cls",
    [PhoenixQuadEngine, KOResetSnowballQuadEngine],
)
def test_engines_that_have_not_implemented_the_mode_refuse_it(engine_cls):
    """A mode that silently does nothing is worse than no mode.

    Both subclasses read the price off their own surfaces and would ignore
    the setting, reporting a legacy-readout price as though it were a
    transition one.
    """
    with pytest.raises(NotImplementedError, match="readout"):
        engine_cls(QuadParams(grid_points=301, readout="transition"))
    engine_cls(QuadParams(grid_points=301))  # the default still constructs


def test_the_transition_readout_refuses_a_continuous_knock_in():
    """The bridged transition is not implemented pointwise; fail closed."""
    product = create_standard_snowball(
        initial_price=SPOT,
        strike=SPOT,
        maturity=1.0,
        num_observations=6,
        ko_rate=0.10,
        ki_barrier=KI,
        ki_continuous=True,
        include_principal=False,
    )
    with pytest.raises(NotImplementedError, match="continuous"):
        _engine("transition").price(product, _env())
