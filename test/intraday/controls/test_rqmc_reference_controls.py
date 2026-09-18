"""Independent accuracy controls for the intraday RQMC reference.

1. A single-event snowball in closed form, with the MATCHED finite-bump targets: the reference's
   desk delta and gamma are finite moves, so they are compared with the same moves of the closed
   form, never with its derivatives. The spots sit near the KI level: with KO and rebate both paying
   the full coupon at maturity, the payoff varies only below 75, which from spot 100 one month out is
   a 5-sigma tail no batch samples (every replicate is then the same constant and the SE is exactly 0).
2. A three-event snowball against the independent Gaussian-transition control, same finite moves.
3. Two independently typed economic controls: a pending knock-out coupon worked out by hand, and
   the variance one second of each clock segment carries.
None of them calls a candidate. A disagreement is a finding to investigate, never a tolerance to widen.
"""
from datetime import timedelta
from math import exp, log, sqrt

import numpy as np
import pytest
from scipy.stats import norm

from quantark.intraday.context import resolve_context
from quantark.intraday.events import EventKind
from quantark.intraday.greeks import with_pricing_env
from quantark.modelvalidation.builders.intraday_common import build_request, bumped_env
from quantark.modelvalidation.registry import get_builder
from quantark.modelvalidation.study import CaseSpec, SamplingPolicy
from quantark.modelvalidation.yaml_loader import _ensure_builtin_builders
from intraday.controls.gaussian_control import reference_snowball

ENV = {"spot": 100.0, "vol": 0.20, "rate": 0.03, "div_yield": 0.01}
MONTHLY = {"initial_date": "2026-03-16", "months": 12, "initial_price": 100.0, "strike": 100.0, "ko_barrier": 103.0,
           "ki_barrier": 75.0, "ko_rate": 0.12, "rebate_rate": 0.12, "contract_multiplier": 1.0,
           "ki_observation": "ko_dates"}
CONTEXT = {"valuation": "2026-09-10T14:00:00+08:00", "phase": "before", "profile": "desk", "calendar": "SSE",
           "history_level": 100.0}
QUANTITIES = ("pv", "desk_delta", "desk_gamma")
H = 0.01
SIGMAS = 4.0
#: Replicates per control. Scrambled Sobol puts a near-fixed integer number of points on each side of a payoff
#: jump, so replicate values are quantized: with 8 batches they often coincide and the sample SE understates the
#: error of the mean (observed: a delta off by 8e-4 with SE 6e-7). 32 replicates estimate that SE reliably.
BATCHES = 32


def _after_fixing(index):
    """(context spec, KO events): one hour after monthly fixing ``index`` (0-based), every earlier fixing at 100."""
    probe = resolve_context(build_request(ENV, MONTHLY, CONTEXT))
    kos = [e for e in probe.timeline.events if e.kind is EventKind.KO]
    ts = kos[index].timestamp + timedelta(hours=1)
    return {**CONTEXT, "valuation": ts.isoformat(), "phase": "after"}, kos


def _rqmc(context, spot, product=MONTHLY, quantities=QUANTITIES, paths=8192):
    """Mean and standard error of the mean over independent batches of the production builder."""
    _ensure_builtin_builders()
    sampling = SamplingPolicy(paths_per_batch=paths, min_batches=BATCHES, max_batches=BATCHES, seed=20260915, bump=H,
                              seed_scheme="substream")
    reference = get_builder("equity.snowball.intraday.mc_rqmc", kind="reference")(
        environment_params={**ENV, "spot": spot}, product_params=product, sampling=sampling, quantities=quantities,
        params={}, context_params=context)
    rows = [reference.run_batch(CaseSpec(name="control"), i).values for i in range(BATCHES)]
    mean = {q: float(np.mean([r[q] for r in rows])) for q in quantities}
    se = {q: float(np.std([r[q] for r in rows], ddof=1) / sqrt(BATCHES)) for q in quantities}
    return mean, se


def _moves(price, spot):
    """(PV, finite-move delta, finite-move gamma) of ``price(spot)`` at the reference's relative bump."""
    base, up, down = price(spot), price(spot * (1.0 + H)), price(spot * (1.0 - H))
    return {"pv": base, "desk_delta": (up - down) / (2.0 * spot * H), "desk_gamma": (up - 2.0 * base + down) / (spot * H) ** 2}


# --- 1. single event, closed form --------------------------------------------------------------------------------
@pytest.mark.parametrize("spot", [78.0, 82.0])
def test_single_event_price_and_matched_finite_moves_against_the_closed_form(spot):
    context, kos = _after_fixing(10)                          # only the maturity observation remains
    ctx = resolve_context(build_request({**ENV, "spot": spot}, MONTHLY, context))
    last, prod = kos[11], ctx.numerical.product
    T = ctx.numerical.maturity_tau
    W = float(ctx.pricing_env.vol_surface.total_variance(100.0, T, spot))
    R = -log(ctx.pricing_env.get_discount_factor(T))
    Q = ctx.pricing_env.get_div_yield(T) * T
    B = float(last.barrier)
    # Two grids meeting exactly at the KI level: a trapezoid rule straddling the payoff jump there would carry an
    # O(jump x density x dy) error of its own (checked against scipy.integrate.quad to 1e-9).
    y_in, y_out = np.linspace(log(20.0), log(75.0), 200001), np.linspace(log(75.0), log(B), 200001)
    pay_in = np.array([prod.get_maturity_payoff_v1(exp(v), ctx.pricing_env) for v in y_in])
    pay_out = np.array([prod.get_maturity_payoff_v0(exp(v), ctx.pricing_env) for v in y_out])

    def closed_form(s):
        mu = log(s) + R - Q - 0.5 * W
        knocked_out = norm.cdf((mu - log(B)) / sqrt(W))
        density = lambda y: norm.pdf((y - mu) / sqrt(W)) / sqrt(W)      # noqa: E731
        return exp(-R) * (last.cash * knocked_out + float(np.trapezoid(pay_in * density(y_in), y_in))
                          + float(np.trapezoid(pay_out * density(y_out), y_out)))

    target = _moves(closed_form, spot)
    mean, se = _rqmc(context, spot)
    for quantity in QUANTITIES:
        assert abs(mean[quantity] - target[quantity]) <= SIGMAS * se[quantity] + 1e-9, (quantity, mean[quantity], target[quantity], se[quantity])
    assert se["desk_gamma"] > 0.0                             # a gamma control, not only PV (spec 11)


# --- 2. three events, independent Gaussian control ---------------------------------------------------------------
def test_three_event_price_and_matched_finite_moves_against_the_gaussian_control():
    spot = 101.0
    context, _ = _after_fixing(8)                             # fixings 10, 11 and 12 remain
    ctx = resolve_context(build_request({**ENV, "spot": spot}, MONTHLY, context))

    def control(s):
        bumped = ctx if s == spot else with_pricing_env(ctx, bumped_env(ctx.pricing_env, s / spot), f"control:{s!r}")
        return reference_snowball(bumped).price

    base = reference_snowball(ctx)
    target = _moves(control, spot)
    mean, se = _rqmc(context, spot)
    slack = {"pv": 3.0 * base.uncertainty_price,
             "desk_delta": 3.0 * base.uncertainty_price / (spot * H),
             "desk_gamma": 3.0 * 4.0 * base.uncertainty_price / (spot * H) ** 2}
    for quantity in QUANTITIES:
        assert abs(mean[quantity] - target[quantity]) <= SIGMAS * se[quantity] + slack[quantity], (quantity, mean[quantity], target[quantity])


# --- 3. independently typed economics ---------------------------------------------------------------------------
def test_a_pending_knock_out_coupon_is_the_hand_computed_present_value():
    """Knocked out on 2026-08-17 at 104 >= 103; the coupon settles 30 calendar days later and is still pending
    at the valuation on 2026-09-10 14:00 +08:00.

    Typed from the term sheet, not read from the engine: coupon = 100 x 12% x ACT/365(2026-03-16 -> 2026-08-17)
    = 100 x 0.12 x 154/365; a date-only payment is deemed made at the 15:00 close of 2026-09-16; discounting is
    continuous at 3% on seconds-exact ACT/365. If this fails, find which of those conventions the contract
    intends and type that; do not edit the number to match the engine.
    """
    coupon = 100.0 * 0.12 * 154.0 / 365.0
    seconds = 6 * 86400 + 3600                               # 2026-09-10 14:00 -> 2026-09-16 15:00
    expected = coupon * exp(-0.03 * seconds / (365.0 * 86400.0))
    context = {**CONTEXT, "fixings": [{"date": "2026-08-17", "level": 104.0}]}
    mean, se = _rqmc(context, 100.0, product={**MONTHLY, "settlement_lag_days": 30})
    assert mean["pv"] == pytest.approx(expected, rel=1e-9) and se["pv"] == 0.0
    assert mean["desk_delta"] == 0.0 and mean["desk_gamma"] == 0.0


@pytest.mark.parametrize("profile, at, expected", [
    ("desk", "2026-09-10T12:00:00+08:00", 0.2 ** 2 * 0.05 / 244 / 5400),       # lunch carries 5% of a day over 90 min
    ("desk", "2026-09-10T14:00:00+08:00", 0.2 ** 2 * 0.35 / 244 / 7200),       # a session carries 35% over 2 h
    ("desk", "2026-09-10T09:00:00+08:00", 0.2 ** 2 * 0.25 / 244 / 66600),      # overnight 15:00 -> 09:30 carries 25%
    ("sessions_only", "2026-09-10T12:00:00+08:00", 0.0),                       # the zero-variance control
])
def test_one_second_of_each_clock_segment_carries_its_declared_variance(profile, at, expected):
    ctx = resolve_context(build_request(ENV, MONTHLY, {**CONTEXT, "valuation": at, "profile": profile}))
    one_second = 1.0 / (365.0 * 86400.0)
    variance = float(ctx.pricing_env.vol_surface.total_variance(100.0, one_second, 100.0))
    assert variance == pytest.approx(expected, rel=1e-9, abs=1e-24)
