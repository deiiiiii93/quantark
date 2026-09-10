"""Test fixtures for the bucket futures hedge.

The analytic oracles live in
``example/snowball_q_term_structure/_bucket_oracles.py`` and are re-exported
here unchanged, so the tests and the shipped validation CLI check production
code against the SAME independent derivation.  That module imports nothing
from ``quantark.backtest``.

Below the re-exports is the MARKET section: short synthetic replays.  It may
import quantark, because it builds products, markets and configs rather than
reference numbers.  Nothing in it is ever used to check a bucket, a policy
target or a residual.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

_STUDY = Path(__file__).resolve().parents[1] / "example" / "snowball_q_term_structure"


def _load_oracles():
    name = "_bucket_oracles"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, _STUDY / "_bucket_oracles.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ORACLES = _load_oracles()

AnalyticBookRisk = _ORACLES.AnalyticBookRisk
AnalyticQuote = _ORACLES.AnalyticQuote
EarlyDigitalCase = _ORACLES.EarlyDigitalCase
LINEAR_FUTURES_COEFFICIENTS = _ORACLES.LINEAR_FUTURES_COEFFICIENTS
LINEAR_QUOTES = _ORACLES.LINEAR_QUOTES
LINEAR_SPOT = _ORACLES.LINEAR_SPOT
LINEAR_SPOT_COEFFICIENT = _ORACLES.LINEAR_SPOT_COEFFICIENT
early_digital_forward_derivative = _ORACLES.early_digital_forward_derivative
early_digital_price = _ORACLES.early_digital_price
flat_forward_carry_forward = _ORACLES.flat_forward_carry_forward
flat_q_forward = _ORACLES.flat_q_forward
flat_q_yield = _ORACLES.flat_q_yield
forward_claim_price = _ORACLES.forward_claim_price
forward_claim_risk = _ORACLES.forward_claim_risk
implied_yield = _ORACLES.implied_yield
linear_book_pricer = _ORACLES.linear_book_pricer
linear_book_risk = _ORACLES.linear_book_risk
log_forward_elasticities = _ORACLES.log_forward_elasticities
normal_cdf = _ORACLES.normal_cdf
normal_pdf = _ORACLES.normal_pdf
pinned_forward_elasticity = _ORACLES.pinned_forward_elasticity
squared_far_future_price = _ORACLES.squared_far_future_price
squared_far_future_risk = _ORACLES.squared_far_future_risk


# ---------------------------------------------------------------------------
# MARKET section: short synthetic replays
#
# Everything below may import quantark, because it builds products, markets
# and configs rather than reference numbers.  Nothing here is ever used to
# check a bucket, a policy target or a residual: those come from the analytic
# section above, which imports nothing.
# ---------------------------------------------------------------------------

#: A realistic index level.  It matters: the bucket bump is ONE INDEX POINT,
#: so on a spot of 100 a front contract three days from expiry would imply a
#: yield move north of 100%, which the dividend curve rightly refuses rather
#: than clipping.  At 4700 the same bump is 2 basis points of price.
MARKET_SPOT = 4700.0
MARKET_VOL = 0.22
MARKET_RATE = 0.02
MARKET_MULTIPLIER = 200.0

#: Three listed contracts with distinct implied carry (q = 3%, 6%, 9%), so the
#: term structure is unmistakably non-flat.  The front contract expires just
#: after the window: at the production seven-day minimum tenor it RETIRES on
#: 2024-01-06, leaving exactly two later contracts, which is the minimum the
#: primary policy needs.
#:
#: The minimum tenor is the production default on purpose.  A delivery-week
#: contract cannot carry either bump: one index point three days from expiry
#: is a 121% implied yield, and the 1% audit spot bump is worse.  The curve
#: refuses both rather than clipping, which is why the real study drops those
#: contracts instead of shrinking the bump.
MARKET_CHAIN = (
    ("IM2401", "2024-01-12", 0.03),
    ("IM2402", "2024-02-23", 0.06),
    ("IM2403", "2024-03-15", 0.09),
)
MARKET_DATES = (
    "2024-01-02",
    "2024-01-03",
    "2024-01-04",
    "2024-01-05",
    "2024-01-06",
    "2024-01-08",
    "2024-01-09",
)
MARKET_MIN_TENOR_DAYS = 7


def market_dataset(*, dates=MARKET_DATES, chain=MARKET_CHAIN, spots=None):
    """An ``AutocallableMarketDataSet`` over the synthetic chain above."""
    import pandas as pd

    from quantark.backtest.replay import AutocallableMarketDataSet

    stamps = [pd.Timestamp(d) for d in dates]
    if spots is None:
        spots = [MARKET_SPOT] * len(stamps)
    rows = []
    for stamp, spot in zip(stamps, spots):
        for contract, expiry, carry in chain:
            expiry_stamp = pd.Timestamp(expiry)
            tenor = (expiry_stamp - stamp).days / 365.0
            if tenor <= 0:
                continue
            rows.append(
                {
                    "date": stamp,
                    "contract": contract,
                    "futures_price": spot * math.exp((MARKET_RATE - carry) * tenor),
                    "expiry_date": expiry_stamp,
                    "multiplier": MARKET_MULTIPLIER,
                }
            )
    return AutocallableMarketDataSet.from_dataframes(
        spot_data=pd.DataFrame({"date": stamps, "spot": list(spots)}),
        vol_data=pd.DataFrame(
            {"date": stamps, "volatility": [MARKET_VOL] * len(stamps)}
        ),
        rate_data=pd.DataFrame(
            {"date": stamps, "rate": [MARKET_RATE] * len(stamps)}
        ),
        futures_data=pd.DataFrame(rows),
    )


def market_product(
    *,
    maturity_days: float = 60.0,
    ko_barrier: float = MARKET_SPOT * 1.08,
    ki_barrier: float = MARKET_SPOT * 0.92,
    ko_observation_days=(20.0, 50.0),
    ki_observation_days=(10.0, 40.0),
):
    """A short snowball that survives the whole synthetic window by default.

    The observation days are exposed because the window is only a week long:
    a test that needs the book to terminate inside it has to move the first
    KO observation into the window rather than wait for the default.
    """
    from quantark.asset.equity.product.option import create_standard_snowball
    from quantark.util.enum import ObservationType

    return create_standard_snowball(
        initial_price=MARKET_SPOT,
        strike=MARKET_SPOT,
        maturity=maturity_days / 365.0,
        contract_multiplier=200.0,
        ko_barrier=ko_barrier,
        ki_barrier=ki_barrier,
        ko_rate=0.02,
        num_observations=len(ko_observation_days),
        ko_observation_dates=[d / 365.0 for d in ko_observation_days],
        ki_observation_type=ObservationType.DISCRETE,
        ki_continuous=False,
        ki_observation_dates=[d / 365.0 for d in ki_observation_days],
        include_principal=True,
    )


def market_engine_config(extrapolation: str = "flat_q", **kwargs):
    """The engine config a bucket hedge needs: actual quotes, supported tail.

    The PDE runs at ``accuracy="high"`` because the audit is sharp enough to
    see the grid.  On the standard grid the engine's own delta and a central
    difference of its own price disagree by about 0.0094 reference hands --
    a real discretisation gap, not audit truncation: it does NOT shrink as
    the audit bump halves, but it roughly halves on the finer grid.  The
    honest response is resolution, not a wider tolerance.
    """
    from quantark.asset.equity.param import PDEParams
    from quantark.backtest.replay import AutocallableEngineConfig

    options = dict(
        dividend_source="futures_curve",
        futures_curve_extrapolation=extrapolation,
        futures_curve_min_tenor_days=MARKET_MIN_TENOR_DAYS,
        pde_params=PDEParams(accuracy="high"),
    )
    options.update(kwargs)
    return AutocallableEngineConfig(**options)
