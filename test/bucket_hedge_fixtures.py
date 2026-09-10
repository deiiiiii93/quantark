"""Independent analytic references for the bucket futures hedge.

Everything in the ANALYTIC section below is written from the revised design
(``docs/superpowers/specs/2026-09-09-bucket-futures-hedge-design-revised.md``)
using only the standard library, so a test that compares a production bucket
against one of these numbers is comparing against an independent derivation
rather than against the code under test.  Nothing here may import
``quantark.backtest.futures_risk``, the bucket strategy, the carry context or
the carry-risk sampler.

The two supported carry builders are re-derived here as closed forms:

``flat_q``
    the zero yield ``q(t)`` is linear in ``t`` between quoted tenors and flat
    outside them, so ``log F(t)`` is log-linear from ``(0, log S)`` to the
    first node and log-linear from ``(0, log S)`` through the last node in the
    tail;
``flat_forward_carry``
    ``log F(t)`` is piecewise linear through ``(0, log S)`` and every node,
    continuing the last segment's slope past the last node.

Both agree on ``(0, T_1)``: the first interval is log-linear from spot to the
first quote under either convention.  That is why the early-monitoring
counterexample of design section 3.3 applies to both builders.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

# ---------------------------------------------------------------------------
# ANALYTIC section: standard library only
# ---------------------------------------------------------------------------

SQRT_2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)


def normal_cdf(x: float) -> float:
    """Standard normal CDF from ``math.erfc``, independent of SciPy/NumPy."""
    return 0.5 * math.erfc(-float(x) / SQRT_2)


def normal_pdf(x: float) -> float:
    """Standard normal density."""
    x = float(x)
    return math.exp(-0.5 * x * x) / SQRT_2PI


@dataclass(frozen=True)
class AnalyticQuote:
    """One listed futures contract in the analytic fixtures."""

    contract: str
    tenor_years: float
    price: float
    multiplier: float = 1.0


# The design's section 3.3 linear counterexample: V = S + F_1 - 0.5 F_2 with
# S = F_1 = F_2 = 100, r = 0, tenors 0.25 / 0.5 and unit multipliers.
LINEAR_SPOT = 100.0
LINEAR_QUOTES: Tuple[AnalyticQuote, ...] = (
    AnalyticQuote("IF2503", 0.25, 100.0, 1.0),
    AnalyticQuote("IF2506", 0.50, 100.0, 1.0),
)
LINEAR_SPOT_COEFFICIENT = 1.0
LINEAR_FUTURES_COEFFICIENTS: Tuple[float, ...] = (1.0, -0.5)


# ---------------------------------------------------------------------------
# Closed-form carry curves
# ---------------------------------------------------------------------------


def _sorted_nodes(quotes: Sequence[AnalyticQuote]) -> Tuple[AnalyticQuote, ...]:
    nodes = tuple(sorted(quotes, key=lambda q: q.tenor_years))
    tenors = [q.tenor_years for q in nodes]
    if any(tenors[i] >= tenors[i + 1] for i in range(len(tenors) - 1)):
        raise ValueError("analytic quotes need strictly increasing tenors")
    return nodes


def implied_yield(spot: float, quote: AnalyticQuote, rate: float) -> float:
    """``q_i = r - log(F_i / S) / T_i``."""
    return float(rate) - math.log(quote.price / float(spot)) / quote.tenor_years


def flat_q_yield(
    spot: float, quotes: Sequence[AnalyticQuote], time: float, rate: float = 0.0
) -> float:
    """Zero yield of the ``flat_q`` builder: linear in ``t``, flat outside."""
    nodes = _sorted_nodes(quotes)
    yields = [implied_yield(spot, q, rate) for q in nodes]
    t = float(time)
    if t <= nodes[0].tenor_years:
        return yields[0]
    if t >= nodes[-1].tenor_years:
        return yields[-1]
    for left, right, y_left, y_right in zip(nodes, nodes[1:], yields, yields[1:]):
        if left.tenor_years <= t <= right.tenor_years:
            w = (t - left.tenor_years) / (right.tenor_years - left.tenor_years)
            return (1.0 - w) * y_left + w * y_right
    raise AssertionError("unreachable: time is bracketed by the node list")


def flat_q_forward(
    spot: float, quotes: Sequence[AnalyticQuote], time: float, rate: float = 0.0
) -> float:
    """``F(t) = S exp((r - q(t)) t)`` for the ``flat_q`` builder."""
    t = float(time)
    if t == 0.0:
        return float(spot)
    q = flat_q_yield(spot, quotes, t, rate)
    return float(spot) * math.exp((float(rate) - q) * t)


def flat_forward_carry_forward(
    spot: float, quotes: Sequence[AnalyticQuote], time: float
) -> float:
    """Piecewise log-linear forward through ``(0, S)`` and every quote.

    Beyond the last node the last segment's forward carry continues; with a
    single node that segment starts at spot, which is the documented one-node
    limit shared with ``flat_q``.
    """
    nodes = _sorted_nodes(quotes)
    t = float(time)
    if t == 0.0:
        return float(spot)
    points = [(0.0, math.log(float(spot)))] + [
        (q.tenor_years, math.log(q.price)) for q in nodes
    ]
    if t >= points[-1][0]:
        (t0, b0), (t1, b1) = points[-2], points[-1]
        slope = (b1 - b0) / (t1 - t0)
        return math.exp(b1 + slope * (t - t1))
    for (t0, b0), (t1, b1) in zip(points, points[1:]):
        if t0 <= t <= t1:
            w = (t - t0) / (t1 - t0)
            return math.exp((1.0 - w) * b0 + w * b1)
    raise AssertionError("unreachable: time is bracketed by the node list")


def pinned_forward_elasticity(
    quotes: Sequence[AnalyticQuote], time: float, *, convention: str
) -> float:
    """``dlog F(t) / dlog S`` holding every listed quote fixed.

    Design section 3.2: ``1 - t/T_1`` on the first interval for both builders,
    ``1 - t[(1-w)/T_i + w/T_(i+1)]`` inside a ``flat_q`` interval, ``1 - t/T_n``
    in the ``flat_q`` tail and exactly zero in the two-node
    ``flat_forward_carry`` tail.
    """
    nodes = _sorted_nodes(quotes)
    t = float(time)
    first, last = nodes[0].tenor_years, nodes[-1].tenor_years
    if t <= first:
        return 1.0 - t / first
    if convention == "flat_forward_carry":
        return 0.0 if len(nodes) >= 2 else 1.0 - t / first
    if convention != "flat_q":
        raise ValueError(f"unknown convention: {convention!r}")
    if t >= last:
        return 1.0 - t / last
    for left, right in zip(nodes, nodes[1:]):
        if left.tenor_years <= t <= right.tenor_years:
            w = (t - left.tenor_years) / (right.tenor_years - left.tenor_years)
            return 1.0 - t * ((1.0 - w) / left.tenor_years + w / right.tenor_years)
    raise AssertionError("unreachable: time is bracketed by the node list")


# ---------------------------------------------------------------------------
# Reference books and their currency Greeks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnalyticBookRisk:
    """Currency Greeks of an analytic reference book.

    ``delta_q`` is ``dV/dS`` at a frozen carry curve, ``buckets`` is the
    ``dV/dF_i`` vector, ``nodal_rhoq`` the ``dV/dq_i`` vector and
    ``delta_f`` the pinned-futures spot derivative.  All are currency
    sensitivities, never hands.
    """

    spot: float
    delta_q: float
    delta_f: float
    buckets: Tuple[float, ...]
    nodal_rhoq: Tuple[float, ...]

    @property
    def parallel_rhoq(self) -> float:
        return sum(self.nodal_rhoq)

    @property
    def gross_nodal_rhoq(self) -> float:
        return sum(abs(r) for r in self.nodal_rhoq)


def linear_book_risk(
    *,
    spot: float = LINEAR_SPOT,
    quotes: Sequence[AnalyticQuote] = LINEAR_QUOTES,
    spot_coefficient: float = LINEAR_SPOT_COEFFICIENT,
    futures_coefficients: Sequence[float] = LINEAR_FUTURES_COEFFICIENTS,
) -> AnalyticBookRisk:
    """``V = a S + sum_i c_i F_i`` differentiated by hand.

    ``B_i = c_i``; the frozen-carry spot derivative adds each future's own
    ``F_i / S``; ``R_i = -c_i F_i T_i``; the pinned-futures spot derivative is
    just ``a`` because every listed quote is held fixed.
    """
    nodes = _sorted_nodes(quotes)
    if len(nodes) != len(tuple(futures_coefficients)):
        raise ValueError("one coefficient per quote is required")
    coefficients = tuple(float(c) for c in futures_coefficients)
    delta_q = float(spot_coefficient) + sum(
        c * q.price / float(spot) for c, q in zip(coefficients, nodes)
    )
    nodal_rhoq = tuple(
        -c * q.price * q.tenor_years for c, q in zip(coefficients, nodes)
    )
    return AnalyticBookRisk(
        spot=float(spot),
        delta_q=delta_q,
        delta_f=float(spot_coefficient),
        buckets=coefficients,
        nodal_rhoq=nodal_rhoq,
    )


def squared_far_future_risk(
    *, spot: float = LINEAR_SPOT, quotes: Sequence[AnalyticQuote] = LINEAR_QUOTES
) -> AnalyticBookRisk:
    """``V = F_n^2``: a carry-curvature reference with no spot coefficient.

    Used to show that a finite joint shock's repricing residual is carry
    curvature, not spot gamma: ``d2V/dF_n^2 = 2`` while ``V`` has no explicit
    dependence on ``S`` at pinned quotes.
    """
    nodes = _sorted_nodes(quotes)
    far = nodes[-1]
    buckets = tuple(
        (2.0 * far.price if q.contract == far.contract else 0.0) for q in nodes
    )
    return AnalyticBookRisk(
        spot=float(spot),
        delta_q=sum(b * q.price / float(spot) for b, q in zip(buckets, nodes)),
        delta_f=0.0,
        buckets=buckets,
        nodal_rhoq=tuple(
            -b * q.price * q.tenor_years for b, q in zip(buckets, nodes)
        ),
    )


def squared_far_future_price(quotes: Sequence[AnalyticQuote]) -> float:
    """``V = F_n^2`` evaluated on a (possibly stressed) quote list."""
    return float(_sorted_nodes(quotes)[-1].price ** 2)


# ---------------------------------------------------------------------------
# Discounted forward claim
# ---------------------------------------------------------------------------


def forward_claim_price(
    *,
    spot: float,
    quotes: Sequence[AnalyticQuote],
    maturity: float,
    rate: float,
    convention: str,
    notional: float = 1.0,
) -> float:
    """PV of a claim paying ``notional * S_T``: ``e^{-rT} N F(T)``."""
    if convention == "flat_q":
        forward = flat_q_forward(spot, quotes, maturity, rate)
    elif convention == "flat_forward_carry":
        forward = flat_forward_carry_forward(spot, quotes, maturity)
    else:
        raise ValueError(f"unknown convention: {convention!r}")
    return float(notional) * math.exp(-float(rate) * float(maturity)) * forward


def forward_claim_risk(
    *,
    spot: float,
    quotes: Sequence[AnalyticQuote],
    maturity: float,
    rate: float,
    convention: str,
    notional: float = 1.0,
) -> AnalyticBookRisk:
    """Currency Greeks of the discounted forward claim.

    ``V = e^{-rT} N F(T)`` so ``B_i = V * dlog F(T)/dlog F_i / F_i``,
    ``D = V/S`` (a proportional scaling of spot and every quote scales ``F``),
    and ``D_F = V * elasticity / S``.
    """
    nodes = _sorted_nodes(quotes)
    price = forward_claim_price(
        spot=spot,
        quotes=quotes,
        maturity=maturity,
        rate=rate,
        convention=convention,
        notional=notional,
    )
    elasticities = _log_forward_elasticities(nodes, maturity, convention=convention)
    buckets = tuple(price * e / q.price for e, q in zip(elasticities, nodes))
    return AnalyticBookRisk(
        spot=float(spot),
        delta_q=price / float(spot),
        delta_f=price
        * pinned_forward_elasticity(nodes, maturity, convention=convention)
        / float(spot),
        buckets=buckets,
        nodal_rhoq=tuple(
            -b * q.price * q.tenor_years for b, q in zip(buckets, nodes)
        ),
    )


def log_forward_elasticities(
    quotes: Sequence[AnalyticQuote], time: float, *, convention: str
) -> Tuple[float, ...]:
    """``dlog F(t) / dlog F_i`` at fixed spot, one entry per node.

    These are LOG elasticities.  A price derivative divides by ``F_i``; the
    tail's ``1 + alpha`` and ``-alpha`` are not themselves ``dF(t)/dF_i``.
    """
    return _log_forward_elasticities(_sorted_nodes(quotes), time, convention=convention)


def _log_forward_elasticities(
    nodes: Tuple[AnalyticQuote, ...], time: float, *, convention: str
) -> Tuple[float, ...]:
    t = float(time)
    first, last = nodes[0].tenor_years, nodes[-1].tenor_years
    out = [0.0] * len(nodes)
    if t <= first:
        out[0] = t / first
        return tuple(out)
    if convention == "flat_forward_carry":
        if t >= last:
            if len(nodes) == 1:
                out[0] = t / first
                return tuple(out)
            alpha = (t - last) / (last - nodes[-2].tenor_years)
            out[-1] = 1.0 + alpha
            out[-2] = -alpha
            return tuple(out)
        for i, (left, right) in enumerate(zip(nodes, nodes[1:])):
            if left.tenor_years <= t <= right.tenor_years:
                w = (t - left.tenor_years) / (right.tenor_years - left.tenor_years)
                out[i] = 1.0 - w
                out[i + 1] = w
                return tuple(out)
    elif convention == "flat_q":
        if t >= last:
            out[-1] = t / last
            return tuple(out)
        for i, (left, right) in enumerate(zip(nodes, nodes[1:])):
            if left.tenor_years <= t <= right.tenor_years:
                w = (t - left.tenor_years) / (right.tenor_years - left.tenor_years)
                out[i] = t * (1.0 - w) / left.tenor_years
                out[i + 1] = t * w / right.tenor_years
                return tuple(out)
    else:
        raise ValueError(f"unknown convention: {convention!r}")
    raise AssertionError("unreachable: time is bracketed by the node list")


# ---------------------------------------------------------------------------
# Early-monitoring down digital (design section 3.3)
# ---------------------------------------------------------------------------


def early_digital_price(forward, barrier, vol, observation, discount, notional):
    """PV of ``notional`` paid at the discount tenor if ``S_t < barrier``.

    ``t`` is the observation tenor and ``forward`` the forward to ``t``.
    """
    width = vol * math.sqrt(observation)
    z = (math.log(barrier / forward) + 0.5 * width * width) / width
    return notional * discount * 0.5 * math.erfc(-z / math.sqrt(2.0))


def early_digital_forward_derivative(
    forward, barrier, vol, observation, discount, notional
) -> float:
    """``dV/dF(t)`` of :func:`early_digital_price`."""
    width = vol * math.sqrt(observation)
    z = (math.log(barrier / forward) + 0.5 * width * width) / width
    return -notional * discount * normal_pdf(z) / (forward * width)


@dataclass(frozen=True)
class EarlyDigitalCase:
    """The design's 29/30 residual-delta counterexample, parameterised."""

    spot: float = 4700.0
    barrier: float = 4650.0
    vol: float = 0.235
    rate: float = 0.02
    notional: float = 50_000_000.0
    observation_days: float = 1.0
    first_expiry_days: float = 30.0
    payment_years: float = 1.0
    day_count: float = 365.0

    @property
    def observation(self) -> float:
        return self.observation_days / self.day_count

    @property
    def first_tenor(self) -> float:
        return self.first_expiry_days / self.day_count

    @property
    def discount(self) -> float:
        return math.exp(-self.rate * self.payment_years)

    @property
    def pinned_fraction(self) -> float:
        """``1 - t/T_1``: the fraction of spot delta the buckets cannot span."""
        return 1.0 - self.observation / self.first_tenor

    def forward(self, quotes: Sequence[AnalyticQuote], convention: str) -> float:
        if convention == "flat_q":
            return flat_q_forward(self.spot, quotes, self.observation, self.rate)
        return flat_forward_carry_forward(self.spot, quotes, self.observation)

    def price(self, quotes: Sequence[AnalyticQuote], convention: str) -> float:
        return early_digital_price(
            self.forward(quotes, convention),
            self.barrier,
            self.vol,
            self.observation,
            self.discount,
            self.notional,
        )

    def frozen_curve_delta(
        self, quotes: Sequence[AnalyticQuote], convention: str
    ) -> float:
        """``dV/dS`` with the carry curve frozen: ``dV/dF(t) * F(t)/S``."""
        forward = self.forward(quotes, convention)
        return (
            early_digital_forward_derivative(
                forward,
                self.barrier,
                self.vol,
                self.observation,
                self.discount,
                self.notional,
            )
            * forward
            / self.spot
        )

    def pinned_futures_delta(
        self, quotes: Sequence[AnalyticQuote], convention: str
    ) -> float:
        """``D_F``: the same derivative times ``F(t)/S * (1 - t/T_1)``."""
        return self.frozen_curve_delta(quotes, convention) * self.pinned_fraction
