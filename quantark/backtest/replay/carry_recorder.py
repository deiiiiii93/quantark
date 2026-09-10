"""Held-book exposure rows, scheduled audits and P&L attribution.

The recorder receives explicit day data.  It never reaches through the
single-leg view, never rebuilds a pricing model of its own, and never
re-sizes a hedge: it reports what the book measured and what the book held.

Three separations are deliberate:

*exposure versus attribution timing*
    exposure columns use TODAY's sensitivities and the holdings AFTER
    trading; the P&L attribution uses YESTERDAY's sensitivities and holdings,
    because that is what the day's move was actually earned on;

*mapped versus direct*
    mapped exposures are reported every day; direct measurements appear only
    on scheduled audit dates, and are NaN with a status elsewhere -- never a
    zero that would read as "checked and neutral";

*numerical validity versus objective achievement*
    a correct ``nodes`` book keeps ``D_F`` of spot delta by design, so the
    audit passes while the objective flag records the intended residual.

Rows cover every eligible curve node PLUS every held or retired leg, zero
holdings included.  Dropping a node the book happens not to hold would let a
single-contract control hide the rhoq it is not hedging.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from quantark.backtest.futures_risk import (
    CarryRiskSettings,
    FuturesBookRisk,
    gross_of,
    held_book_risk,
    parallel_of,
    rhoq_bp_per_1pct,
    spot_delta_hands,
)
from quantark.backtest.replay.carry_risk import (
    CarryAuditResult,
    audit_held_book,
    not_measured_audit,
)
from quantark.backtest.replay.carry_stress import CarryScenario, run_scenario
from quantark.util.exceptions import ValidationError

NAN = float("nan")

#: Why a day's linear attribution could not be formed.  Each one leaves the
#: day's ACTUAL P&L and costs intact; only the decomposition is withheld.
ATTRIBUTION_STATUSES = (
    "ok",
    "first_date",
    "coordinate_set_changed",
    "missing_carried_mark",
    "not_recorded",
)

__all__ = [
    "ATTRIBUTION_STATUSES",
    "CarryDaySnapshot",
    "CarryExposureRecorder",
]


@dataclass
class CarryDaySnapshot:
    """Everything the previous day left behind for today's attribution."""

    date: Any
    spot: float
    coordinates: Tuple[str, ...]
    prices: Dict[str, float]
    multipliers: Dict[str, float]
    delta_f: float
    buckets: Dict[str, float]
    holdings: Dict[str, float]
    gamma: float
    product_pnl: float
    transaction_costs: float


@dataclass
class CarryExposureRecorder:
    """Builds the three carry frames from explicit daily inputs."""

    settings: CarryRiskSettings
    record: bool = True
    audit_mode: str = "none"
    audit_dates: Tuple[Any, ...] = ()
    scenarios: Tuple[CarryScenario, ...] = ()
    stress_dates: Tuple[Any, ...] = ()
    legs: list = field(default_factory=list)
    attribution: list = field(default_factory=list)
    stresses: list = field(default_factory=list)
    _previous: Optional[CarryDaySnapshot] = None
    _node_sets: set = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.record:
            self.settings.require_resolved()

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def audits_today(self, date, coordinates: Sequence[str]) -> bool:
        """Sampled mode always includes inception and node-set changes."""
        if self.audit_mode == "none":
            return False
        if self.audit_mode == "daily":
            return True
        key = tuple(coordinates)
        changed = key not in self._node_sets
        inception = self._previous is None
        return bool(inception or changed or date in self.audit_dates)

    def stresses_today(self, date) -> bool:
        return bool(self.scenarios) and (
            not self.stress_dates or date in self.stress_dates
        )

    # ------------------------------------------------------------------
    # The day
    # ------------------------------------------------------------------

    def record_day(
        self,
        *,
        date,
        risk: Optional[FuturesBookRisk],
        context=None,
        targets,
        carried: Mapping[str, float],
        held: Mapping[str, float],
        chain_prices: Mapping[str, float],
        chain_multipliers: Mapping[str, float],
        chain_expiries: Mapping[str, Any],
        product_pnl: float,
        transaction_costs: float,
        gamma: float = NAN,
        objective: str = "",
        carry_family: str = "futures_curve",
        price_at=None,
        product_maturity: Optional[float] = None,
        flat_parallel_rhoq: Optional[float] = None,
        flat_fit_rmse_bp: float = NAN,
    ) -> None:
        """Append this date's leg rows, attribution row and stress rows."""
        if not self.record:
            return
        settings = self.settings
        notional = float(settings.reference_notional)
        m_ref = float(settings.reference_multiplier)

        audit = self._run_audit(date, risk, context, held, targets, price_at)
        self._append_leg_rows(
            date=date,
            risk=risk,
            targets=targets,
            carried=carried,
            held=held,
            chain_prices=chain_prices,
            chain_multipliers=chain_multipliers,
            chain_expiries=chain_expiries,
            audit=audit,
            notional=notional,
            product_maturity=product_maturity,
        )
        self._append_attribution_row(
            date=date,
            risk=risk,
            targets=targets,
            held=held,
            chain_prices=chain_prices,
            chain_multipliers=chain_multipliers,
            audit=audit,
            product_pnl=product_pnl,
            transaction_costs=transaction_costs,
            gamma=gamma,
            objective=objective,
            carry_family=carry_family,
            notional=notional,
            m_ref=m_ref,
            flat_parallel_rhoq=flat_parallel_rhoq,
            flat_fit_rmse_bp=flat_fit_rmse_bp,
        )
        if (
            price_at is not None
            and risk is not None
            and context is not None
            and self.stresses_today(date)
        ):
            self._append_stress_rows(
                date, context, targets, held, price_at, objective
            )

        if risk is not None:
            self._node_sets.add(tuple(risk.contracts))
            self._previous = CarryDaySnapshot(
                date=date,
                spot=risk.spot,
                coordinates=tuple(risk.contracts),
                prices={c: float(chain_prices[c]) for c in risk.contracts},
                multipliers={
                    c: float(chain_multipliers[c]) for c in risk.contracts
                },
                delta_f=risk.delta_f_derived,
                buckets={b.contract: b.bucket_currency for b in risk.buckets},
                holdings=dict(held),
                gamma=float(gamma),
                product_pnl=float(product_pnl),
                transaction_costs=float(transaction_costs),
            )
        else:
            self._previous = None

    # ------------------------------------------------------------------
    # Audits
    # ------------------------------------------------------------------

    def _run_audit(
        self, date, risk, context, held, targets, price_at
    ) -> CarryAuditResult:
        if risk is None or price_at is None or context is None:
            return not_measured_audit(
                holdings_kind="actual",
                reason="no live carry risk on this date",
                settings=self.settings,
            )
        if not self.audits_today(date, risk.contracts):
            return not_measured_audit(
                holdings_kind="actual",
                reason=f"not a scheduled audit date ({self.audit_mode})",
                settings=self.settings,
            )
        on_curve, off_curve = split_holdings(risk, held)
        if off_curve:
            # Every audit scenario reprices the product on the curve and adds
            # the hedge's scenario P&L over the CURVE's quotes.  A leg outside
            # that universe would be dropped from both sides, so the audit
            # reports that it cannot decide rather than quietly passing.
            return CarryAuditResult(
                status="inconclusive",
                reason=(
                    "held contracts outside the risk universe: "
                    f"{sorted(off_curve)}"
                ),
                holdings_kind="actual",
                spot_step=NAN,
                yield_step=NAN,
                reference_multiplier=float(self.settings.reference_multiplier),
                reference_notional=float(self.settings.reference_notional),
            )
        ideal_delta = getattr(targets, "ideal_net_delta", None)
        ideal_parallel = (
            targets.ideal_net_parallel_rhoq if targets is not None else None
        )
        return audit_held_book(
            price_at,
            context,
            risk,
            on_curve,
            settings=self.settings,
            holdings_kind="actual",
            ideal_net_delta=ideal_delta,
            ideal_net_parallel_rhoq=ideal_parallel,
        )

    # ------------------------------------------------------------------
    # Rows
    # ------------------------------------------------------------------

    def _append_leg_rows(
        self,
        *,
        date,
        risk,
        targets,
        carried,
        held,
        chain_prices,
        chain_multipliers,
        chain_expiries,
        audit,
        notional,
        product_maturity,
    ) -> None:
        nodes = tuple(risk.contracts) if risk is not None else ()
        universe = sorted(set(nodes) | set(carried) | set(held))
        on_curve, off_curve = split_holdings(risk, held)
        mapped_delta, mapped_rho = (
            held_book_risk(risk, on_curve) if risk is not None else (NAN, {})
        )
        ideal_rho = getattr(targets, "ideal_net_rhoq", {}) or {}
        correction = getattr(targets, "correction", {}) or {}
        last_node = nodes[-1] if nodes else None
        for contract in universe:
            bucket = None
            if risk is not None and contract in nodes:
                bucket = risk.bucket(contract)
            price = float(chain_prices.get(contract, NAN))
            multiplier = float(chain_multipliers.get(contract, NAN))
            before = float(carried.get(contract, 0.0))
            after = float(held.get(contract, 0.0))
            scaled = float(getattr(targets, "scaled", {}).get(contract, NAN))
            rounded = float(getattr(targets, "rounded", {}).get(contract, NAN))
            ideal = float(getattr(targets, "ideal", {}).get(contract, NAN))
            product_rhoq = bucket.nodal_rhoq if bucket is not None else NAN
            tenor = (
                bucket.tenor_years
                if bucket is not None
                else _tenor_of(date, chain_expiries.get(contract))
            )
            hedge_rhoq = (
                -after * multiplier * price * tenor
                if math.isfinite(tenor) and math.isfinite(price)
                else NAN
            )
            # An off-curve leg has no product bucket, so its NET carry
            # exposure is its own hedge term alone -- reported, not dropped.
            net_rhoq = mapped_rho.get(
                contract, hedge_rhoq if contract in off_curve else NAN
            )
            direct = (
                audit.direct_nodal_rhoq.get(contract, NAN)
                if audit.measured
                else NAN
            )
            self.legs.append(
                {
                    "date": date,
                    "contract": contract,
                    "maturity": chain_expiries.get(contract),
                    "tenor_years": bucket.tenor_years if bucket else NAN,
                    "price": price,
                    "multiplier": multiplier,
                    "is_curve_node": contract in nodes,
                    "is_correction_leg": bool(correction.get(contract, 0.0)),
                    "is_last_curve_node": contract == last_node,
                    "has_product_tail": (
                        bool(
                            product_maturity is not None
                            and bucket is not None
                            and product_maturity > bucket.tenor_years
                        )
                        if contract == last_node
                        else False
                    ),
                    "bucket_currency": (
                        bucket.bucket_currency if bucket is not None else NAN
                    ),
                    "bucket_contract_equivalent": (
                        bucket.contract_equivalent if bucket is not None else NAN
                    ),
                    "ideal_contracts": ideal,
                    "scaled_contracts": scaled,
                    "rounded_contracts": rounded,
                    "held_before": before,
                    "held_after": after,
                    "trade_contracts": after - before,
                    # rounding + no_trade = actual - scaled, the total
                    # execution error of design section 7.2.
                    "rounding_error_contracts": (
                        rounded - scaled
                        if math.isfinite(rounded) and math.isfinite(scaled)
                        else NAN
                    ),
                    "no_trade_error_contracts": (
                        after - rounded if math.isfinite(rounded) else NAN
                    ),
                    "retired": contract not in nodes,
                    "unrealized_pnl": NAN,
                    "product_rhoq_bp": _bp(product_rhoq, notional),
                    "hedge_rhoq_bp": _bp(hedge_rhoq, notional),
                    "net_rhoq_bp": _bp(net_rhoq, notional),
                    "ideal_net_rhoq_bp": _bp(
                        ideal_rho.get(contract, NAN), notional
                    ),
                    "direct_net_rhoq_bp": _bp(direct, notional),
                    "rhoq_audit_error_bp": (
                        audit.nodal_rhoq_audit_error_bp.get(contract, NAN)
                        if audit.measured
                        else NAN
                    ),
                    "audit_status": audit.status,
                }
            )
        # mapped_delta is reported on the attribution row, not per leg.
        _ = mapped_delta

    def _append_attribution_row(
        self,
        *,
        date,
        risk,
        targets,
        held,
        chain_prices,
        chain_multipliers,
        audit,
        product_pnl,
        transaction_costs,
        gamma,
        objective,
        carry_family,
        notional,
        m_ref,
        flat_parallel_rhoq,
        flat_fit_rmse_bp,
    ) -> None:
        previous = self._previous
        on_curve, off_curve = split_holdings(risk, held)
        mapped_delta, mapped_rho = (
            held_book_risk(risk, on_curve) if risk is not None else (NAN, {})
        )
        off_curve_delta = (
            sum(
                quantity
                * float(chain_multipliers.get(contract, NAN))
                * float(chain_prices.get(contract, NAN))
                / risk.spot
                for contract, quantity in off_curve.items()
            )
            if risk is not None and off_curve
            else 0.0
        )
        if risk is not None and off_curve:
            # A leg outside the risk universe still moves the book.
            mapped_delta += off_curve_delta
        hedge_delta = (
            sum(
                float(on_curve.get(b.contract, 0.0))
                * b.multiplier
                * b.price
                / risk.spot
                for b in risk.buckets
            )
            + off_curve_delta
            if risk is not None
            else NAN
        )
        pair = getattr(targets, "correction_pair", None) or (None, None)

        status = "ok"
        product_dv = NAN
        hedge_price_pnl = NAN
        book_dv = NAN
        linear_spot = NAN
        linear_forwards = NAN
        remainder = NAN
        gamma_diag = NAN
        fees = NAN
        if previous is None:
            status = "first_date"
        elif risk is None or tuple(risk.contracts) != previous.coordinates:
            status = "coordinate_set_changed"
        elif any(c not in chain_prices for c in previous.coordinates):
            status = "missing_carried_mark"

        if status == "ok":
            spot = risk.spot
            fees = float(transaction_costs) - previous.transaction_costs
            product_dv = float(product_pnl) - previous.product_pnl
            hedge_price_pnl = sum(
                previous.holdings.get(c, 0.0)
                * previous.multipliers[c]
                * (float(chain_prices[c]) - previous.prices[c])
                for c in previous.coordinates
            )
            linear_spot = previous.delta_f * (spot - previous.spot)
            linear_forwards = sum(
                (
                    previous.buckets[c]
                    + previous.holdings.get(c, 0.0) * previous.multipliers[c]
                )
                * (float(chain_prices[c]) - previous.prices[c])
                for c in previous.coordinates
            )
            remainder = product_dv + hedge_price_pnl - linear_spot - linear_forwards
            # The replay has no financing account; the pending-receivable
            # decay already belongs to product P&L and is not counted again.
            book_dv = product_dv + hedge_price_pnl - fees + 0.0
            if math.isfinite(previous.gamma):
                gamma_diag = 0.5 * previous.gamma * (spot - previous.spot) ** 2

        self.attribution.append(
            {
                "date": date,
                "carry_family": carry_family,
                "objective": objective,
                "correction_contract_a": pair[0],
                "correction_contract_b": pair[1],
                "reference_notional": notional,
                "reference_multiplier": m_ref,
                "product_delta_hands": (
                    spot_delta_hands(risk.delta_q, m_ref) if risk is not None else NAN
                ),
                "hedge_delta_hands": (
                    spot_delta_hands(hedge_delta, m_ref)
                    if math.isfinite(hedge_delta)
                    else NAN
                ),
                "net_delta_hands": (
                    spot_delta_hands(mapped_delta, m_ref)
                    if math.isfinite(mapped_delta)
                    else NAN
                ),
                "delta_f_derived_hands": (
                    spot_delta_hands(risk.delta_f_derived, m_ref)
                    if risk is not None
                    else NAN
                ),
                "delta_f_direct_hands": (
                    spot_delta_hands(audit.delta_f_direct, m_ref)
                    if audit.measured and math.isfinite(audit.delta_f_direct)
                    else NAN
                ),
                "identity_residual_hands": audit.identity_residual_hands,
                "direct_net_delta_hands": (
                    spot_delta_hands(audit.direct_net_delta, m_ref)
                    if audit.measured and math.isfinite(audit.direct_net_delta)
                    else NAN
                ),
                "net_delta_audit_error_hands": audit.net_delta_audit_error_hands,
                "product_rhoq_bp": _bp(
                    risk.parallel_rhoq if risk is not None else NAN, notional
                ),
                "hedge_rhoq_bp": _bp(
                    (parallel_of(mapped_rho) - risk.parallel_rhoq)
                    if risk is not None
                    else NAN,
                    notional,
                ),
                "net_rhoq_bp": _bp(parallel_of(mapped_rho) if mapped_rho else NAN,
                                   notional),
                "direct_net_parallel_rhoq_bp": _bp(
                    audit.direct_net_parallel_rhoq if audit.measured else NAN,
                    notional,
                ),
                "parallel_rhoq_audit_error_bp": audit.parallel_rhoq_audit_error_bp,
                "product_rhoq_gross_bp": _bp(
                    risk.gross_nodal_rhoq if risk is not None else NAN, notional
                ),
                "net_rhoq_gross_bp": _bp(
                    gross_of(mapped_rho) if mapped_rho else NAN, notional
                ),
                "ideal_net_delta_hands": (
                    spot_delta_hands(targets.ideal_net_delta, m_ref)
                    if targets is not None
                    else NAN
                ),
                "ideal_net_parallel_rhoq_bp": _bp(
                    targets.ideal_net_parallel_rhoq if targets is not None else NAN,
                    notional,
                ),
                "ideal_net_rhoq_gross_bp": _bp(
                    targets.ideal_gross_nodal_rhoq if targets is not None else NAN,
                    notional,
                ),
                "gross_contracts": sum(abs(float(v)) for v in held.values()),
                "gross_futures_notional": sum(
                    abs(float(v))
                    * float(chain_multipliers.get(c, NAN))
                    * float(chain_prices.get(c, NAN))
                    for c, v in held.items()
                ),
                "turnover_contracts": (
                    sum(
                        abs(float(held.get(c, 0.0)) - previous.holdings.get(c, 0.0))
                        for c in set(held) | set(previous.holdings)
                    )
                    if previous is not None
                    else NAN
                ),
                "flat_fit_rmse_bp": flat_fit_rmse_bp,
                "audit_status": audit.status,
                "attribution_status": status,
                "product_dv": product_dv,
                "hedge_price_pnl": hedge_price_pnl,
                "transaction_costs": fees,
                # The replay carries no independent financing account.
                "financing_pnl": 0.0,
                "book_dv": book_dv,
                "linear_spot_pinned": linear_spot,
                "linear_listed_forwards": linear_forwards,
                "linear_product_spot_flat": (
                    NAN if flat_parallel_rhoq is None else _flat_spot(previous, risk)
                ),
                "linear_product_carry_flat": NAN,
                "remainder_after_linear": remainder,
                # A frozen-q gamma is a DIFFERENT directional derivative; it
                # stays a named diagnostic and is never subtracted.
                "spot_gamma_frozen_q_diagnostic": gamma_diag,
            }
        )

    def _append_stress_rows(
        self, date, context, targets, held, price_at, objective
    ) -> None:
        holdings_sets = [("actual", dict(held))]
        if targets is not None:
            holdings_sets.append(("ideal", dict(targets.ideal)))
        for kind, holdings in holdings_sets:
            for scenario in self.scenarios:
                result = run_scenario(
                    price_at, context, scenario, holdings, holdings_kind=kind
                )
                self.stresses.append(
                    {
                        "date": date,
                        "objective": objective,
                        "holdings_kind": kind,
                        "scenario_id": result.scenario_id,
                        "scenario_family": result.family,
                        "shock_definition": _definition(scenario),
                        "bump_metadata": json.dumps(
                            result.metadata, sort_keys=True, default=str
                        ),
                        "product_pnl": result.product_pnl,
                        "hedge_pnl": result.hedge_pnl,
                        "book_pnl": result.book_pnl,
                        "linear_prediction": result.linear_prediction,
                        "repricing_error": result.repricing_error,
                        "source_basis_assumption": result.source_basis_assumption,
                    }
                )


def split_holdings(risk, held) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Separate holdings ON the risk curve from holdings outside it.

    A single-contract control can legitimately hold a contract the CURVE has
    already dropped: the roll policy's window (five days by default) is looser
    than the curve's minimum tenor (seven).  Such a leg has no product bucket,
    so it is not a risk coordinate -- but it does carry spot delta and its own
    carry sensitivity, and it must not silently vanish from either.
    """
    nodes = set(risk.contracts) if risk is not None else set()
    on_curve = {c: float(v) for c, v in held.items() if c in nodes}
    off_curve = {
        c: float(v) for c, v in held.items() if c not in nodes and float(v) != 0.0
    }
    return on_curve, off_curve


def _bp(value, notional) -> float:
    value = float(value)
    if not math.isfinite(value):
        return NAN
    return rhoq_bp_per_1pct(value, notional)


def _tenor_of(date, expiry) -> float:
    """Year fraction from ``date`` to a chain expiry, ACT/365."""
    if expiry is None:
        return NAN
    try:
        days = (expiry - date).days
    except TypeError:
        return NAN
    return float(days) / 365.0


def _flat_spot(previous, risk) -> float:
    if previous is None or risk is None:
        return NAN
    return previous.delta_f * (risk.spot - previous.spot)


def _definition(scenario: CarryScenario) -> str:
    parts = []
    if scenario.spot_shift_rel:
        parts.append(f"spot{scenario.spot_shift_rel:+.4%}")
    if scenario.parallel_yield_shift:
        parts.append(f"parallel_q{scenario.parallel_yield_shift:+.4f}")
    for contract, shift in sorted(scenario.node_yield_shifts.items()):
        parts.append(f"{contract}_q{shift:+.4f}")
    if scenario.tail_rate_shift:
        parts.append(f"tail_lambda{scenario.tail_rate_shift:+.4f}")
    if scenario.shape_interval is not None and scenario.shape_log_forward_shift:
        start, end = scenario.shape_interval
        parts.append(
            f"shape{scenario.shape_log_forward_shift:+.4f}@[{start:g},{end:g}]"
        )
    return "; ".join(parts) or "no shock"
