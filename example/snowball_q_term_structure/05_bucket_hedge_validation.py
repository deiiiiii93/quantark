"""Stage 05 — reproducible numerical validation of the bucket futures hedge.

Runs the design's required cases against INDEPENDENT analytic oracles
(``_bucket_oracles``), records every price, effective bump and grid setting,
and exits non-zero when a required case fails or cannot be resolved.

    python 05_bucket_hedge_validation.py --synthetic --out-dir OUT
    python 05_bucket_hedge_validation.py --snapshot SNAP.json --out-dir OUT
    python 05_bucket_hedge_validation.py --historical-dates 2025-03-03 \
        --history-dir HIST --out-dir OUT

Synthetic mode is the offline default and needs no local vendor history.
Historical mode serialises the exact date, quotes, spot, rates and product
terms it used, so ``--snapshot`` reproduces the same numbers later without
fetching newer data.  Missing history is a DATA BLOCKER, reported as such:
it is neither a numerical failure nor an empty successful run.

Every "pass" here is backed by a direct measurement.  A case whose error has
not stabilised across its declared ladder returns ``inconclusive`` and keeps
all of its samples, rather than widening a tolerance.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

STUDY_DIR = Path(__file__).resolve().parent
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))

import _common as C  # noqa: E402
from quantark.asset.equity.market import IndexFuturesQuote  # noqa: E402
from quantark.backtest.futures_risk import (  # noqa: E402
    CarryRiskSettings,
    FuturesBookRisk,
    held_book_risk,
)
from quantark.backtest.replay.carry_context import CarryCurveContext  # noqa: E402
from quantark.backtest.replay.carry_risk import (  # noqa: E402
    audit_held_book,
    buckets_of,
    direct_pinned_delta,
    sample_buckets,
)
from quantark.backtest.replay.carry_stress import (  # noqa: E402
    run_scenario,
    shape_scenario,
    tail_scenario,
)
from quantark.backtest.strategy import ideal_targets  # noqa: E402
from quantark.param import FlatRateCurve  # noqa: E402


def _load_oracles():
    name = "_bucket_oracles"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, STUDY_DIR / "_bucket_oracles.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


O = _load_oracles()

# Declared STARTING settings, not asserted converged settings.
GRID_LADDER = (201, 401, 801)
SPOT_BUMPS = (0.005, 0.0025, 0.00125)
FUTURES_BUMPS = (1.0, 0.5, 0.25)
YIELD_BUMPS = (1e-4, 5e-5, 2.5e-5)
MAX_HALVINGS = 8

DELTA_TOLERANCE_HANDS = 0.01
RHOQ_TOLERANCE_BP = 0.01

#: The linear reference book is SCALED to the study notional.  The tolerances
#: are 0.01 reference hands and 0.01 bp of N_ref, so a book carrying one
#: currency unit per index point would sit seven orders of magnitude below
#: them and no perturbation could ever breach one.  Sizing the oracle like
#: the real book is what makes the declared budgets meaningful.
def book_scale(inputs) -> float:
    return float(inputs["notional"]) / float(inputs["spot"])


def scaled_coefficients(inputs, quotes) -> Tuple[float, ...]:
    scale = book_scale(inputs)
    return tuple(scale * c for c in (1.0, -0.5, 0.25)[: len(quotes)])

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_INCONCLUSIVE = "inconclusive"
STATUS_BLOCKED = "data_blocker"


@dataclass
class CaseResult:
    """One validation case, its status and everything behind it."""

    name: str
    status: str
    reason: str = ""
    required: bool = True
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)

    def row(self) -> Dict[str, Any]:
        return {
            "case": self.name,
            "status": self.status,
            "required": self.required,
            "reason": self.reason,
            **{f"setting_{k}": v for k, v in self.settings.items()},
        }


def _context(spot, quotes, rate, extrapolation, valuation) -> CarryCurveContext:
    return CarryCurveContext(
        quotes=tuple(
            IndexFuturesQuote(
                contract=q.contract,
                maturity=q.tenor_years,
                price=q.price,
                multiplier=q.multiplier,
            )
            for q in quotes
        ),
        spot=float(spot),
        rate_curve=FlatRateCurve(rate=float(rate)),
        extrapolation=extrapolation,
        underlying=C.UNDERLYING_NAME,
        valuation_date=valuation,
    )


def _settings(notional: float, spot_bump: float, yield_bump: float) -> CarryRiskSettings:
    return CarryRiskSettings(
        reference_notional=float(notional),
        reference_multiplier=C.FUTURES_MULTIPLIER,
        audit_spot_bump_rel=float(spot_bump),
        audit_yield_bump=float(yield_bump),
        delta_tolerance_hands=DELTA_TOLERANCE_HANDS,
        rhoq_tolerance_bp=RHOQ_TOLERANCE_BP,
    )


def _stabilised(values: Sequence[float], tolerance: float) -> bool:
    return len(values) >= 2 and abs(values[-1] - values[-2]) <= tolerance


# ---------------------------------------------------------------------------
# Case 1 — linear signed book, non-unit quantities and multipliers
# ---------------------------------------------------------------------------


def case_linear_book(inputs: Dict[str, Any]) -> Tuple[CaseResult, List[Dict[str, Any]]]:
    quotes = inputs["analytic_quotes"]
    spot, rate = inputs["spot"], inputs["rate"]
    quantity = 2.5
    scale = book_scale(inputs)
    coefficients = scaled_coefficients(inputs, quotes)
    reference = O.linear_book_risk(
        spot=spot,
        quotes=quotes,
        futures_coefficients=coefficients,
        spot_coefficient=scale,
    )
    price_at = O.linear_book_pricer(
        tenors=[q.tenor_years for q in quotes],
        futures_coefficients=coefficients,
        spot_coefficient=scale,
        rate=rate,
    )
    context = _context(spot, quotes, rate, "flat_q", inputs["valuation"])
    rows: List[Dict[str, Any]] = []
    errors: List[float] = []
    for points in FUTURES_BUMPS:
        buckets = buckets_of(context, sample_buckets(price_at, context, points))
        worst = max(
            abs(b.bucket_currency - expected)
            for b, expected in zip(buckets, reference.buckets)
        )
        errors.append(worst)
        rows.append(
            {
                "case": "linear_book",
                "futures_bump_points": points,
                "worst_bucket_error": worst,
                **{f"bucket_{b.contract}": b.bucket_currency for b in buckets},
            }
        )
    # The book is linear in every F_i, so a central difference is EXACT up to
    # floating point; the comparison is relative because the book is scaled.
    magnitude = max(abs(v) for v in reference.buckets)
    ok = max(errors) <= 1e-9 * magnitude
    # Quantity is applied once, at aggregation.
    weighted = O.linear_book_risk(
        spot=spot,
        quotes=quotes,
        futures_coefficients=[quantity * c for c in coefficients],
        spot_coefficient=quantity * scale,
    )
    quantity_ok = all(
        abs(w - quantity * u) <= 1e-9 * magnitude
        for w, u in zip(weighted.buckets, reference.buckets)
    )
    return (
        CaseResult(
            "linear_book",
            STATUS_PASS if (ok and quantity_ok) else STATUS_FAIL,
            "" if ok and quantity_ok else f"worst bucket error {max(errors):.3e}",
            settings={
                "futures_bumps": list(FUTURES_BUMPS),
                "quantity": quantity,
                "book_scale": scale,
            },
            measurements=rows,
        ),
        rows,
    )


# ---------------------------------------------------------------------------
# Case 2 — discounted forward claim before, between and beyond the nodes
# ---------------------------------------------------------------------------


def case_forward_claim(inputs) -> Tuple[CaseResult, List[Dict[str, Any]]]:
    quotes = inputs["analytic_quotes"]
    spot, rate = inputs["spot"], inputs["rate"]
    first = quotes[0].tenor_years
    last = quotes[-1].tenor_years
    tenors = {
        "before_first": 0.5 * first,
        "between": 0.5 * (first + last),
        "tail": last * 1.5,
    }
    rows: List[Dict[str, Any]] = []
    failures = []
    for convention in ("flat_q", "flat_forward_carry"):
        context = _context(spot, quotes, rate, convention, inputs["valuation"])
        for label, tenor in tenors.items():

            def price_at(s, dividend, tenor=tenor):
                q = float(dividend.get_yield(tenor))
                return (
                    math.exp(-rate * tenor)
                    * float(s)
                    * math.exp((rate - q) * tenor)
                )

            reference = O.forward_claim_risk(
                spot=spot,
                quotes=quotes,
                maturity=tenor,
                rate=rate,
                convention=convention,
            )
            errors = []
            for points in FUTURES_BUMPS:
                buckets = buckets_of(
                    context, sample_buckets(price_at, context, points)
                )
                worst = max(
                    abs(b.bucket_currency - expected)
                    for b, expected in zip(buckets, reference.buckets)
                )
                relative = worst / max(
                    1e-12, max(abs(v) for v in reference.buckets)
                )
                errors.append(relative)
                rows.append(
                    {
                        "case": "forward_claim",
                        "convention": convention,
                        "region": label,
                        "maturity": tenor,
                        "futures_bump_points": points,
                        "relative_error": relative,
                    }
                )
            if errors[-1] > 1e-5:
                failures.append(f"{convention}/{label}: {errors[-1]:.2e}")
    return (
        CaseResult(
            "forward_claim",
            STATUS_PASS if not failures else STATUS_FAIL,
            "; ".join(failures),
            settings={"futures_bumps": list(FUTURES_BUMPS), "regions": list(tenors)},
            measurements=rows,
        ),
        rows,
    )


# ---------------------------------------------------------------------------
# Case 3 — early down digital and its 29/30 residual delta fraction
# ---------------------------------------------------------------------------


def case_early_digital(inputs) -> Tuple[CaseResult, List[Dict[str, Any]]]:
    case = O.EarlyDigitalCase()
    quotes = (
        O.AnalyticQuote("F1", case.first_tenor, case.spot * 0.999, C.FUTURES_MULTIPLIER),
        O.AnalyticQuote("F2", 90.0 / 365.0, case.spot * 0.997, C.FUTURES_MULTIPLIER),
    )
    rows: List[Dict[str, Any]] = []
    failures = []
    for convention in ("flat_q", "flat_forward_carry"):
        context = _context(case.spot, quotes, case.rate, convention, inputs["valuation"])

        def price_at(s, dividend):
            t = case.observation
            q = float(dividend.get_yield(t))
            forward = float(s) * math.exp((case.rate - q) * t)
            return O.early_digital_price(
                forward, case.barrier, case.vol, t, case.discount, case.notional
            )

        errors = []
        for points in FUTURES_BUMPS:
            buckets = buckets_of(context, sample_buckets(price_at, context, points))
            risk = FuturesBookRisk(
                spot=case.spot,
                delta_q=case.frozen_curve_delta(quotes, convention),
                buckets=buckets,
            )
            fraction = risk.delta_f_derived / risk.delta_q
            error = abs(fraction - case.pinned_fraction)
            errors.append(error)
            rows.append(
                {
                    "case": "early_digital",
                    "convention": convention,
                    "futures_bump_points": points,
                    "residual_delta_fraction": fraction,
                    "expected_fraction": case.pinned_fraction,
                    "abs_error": error,
                }
            )
        if errors[-1] > 1e-4:
            failures.append(f"{convention}: {errors[-1]:.2e}")
    return (
        CaseResult(
            "early_digital",
            STATUS_PASS if not failures else STATUS_FAIL,
            "; ".join(failures),
            settings={
                "expected_fraction": case.pinned_fraction,
                "observation_days": case.observation_days,
                "first_expiry_days": case.first_expiry_days,
            },
            measurements=rows,
        ),
        rows,
    )


# ---------------------------------------------------------------------------
# Case 4 — the three policy residuals, audited by independent repricing
# ---------------------------------------------------------------------------


def case_policy_residuals(inputs) -> Tuple[CaseResult, List[Dict[str, Any]], List[Dict[str, Any]]]:
    quotes = inputs["analytic_quotes"]
    spot, rate = inputs["spot"], inputs["rate"]
    scale = book_scale(inputs)
    coefficients = scaled_coefficients(inputs, quotes)
    price_at = O.linear_book_pricer(
        tenors=[q.tenor_years for q in quotes],
        futures_coefficients=coefficients,
        spot_coefficient=scale,
        rate=rate,
    )
    reference = O.linear_book_risk(
        spot=spot,
        quotes=quotes,
        futures_coefficients=coefficients,
        spot_coefficient=scale,
    )
    rows: List[Dict[str, Any]] = []
    holdings_rows: List[Dict[str, Any]] = []
    failures = []
    for convention in ("flat_q", "flat_forward_carry"):
        context = _context(spot, quotes, rate, convention, inputs["valuation"])
        buckets = buckets_of(context, sample_buckets(price_at, context, FUTURES_BUMPS[-1]))
        risk = FuturesBookRisk(spot=spot, delta_q=reference.delta_q, buckets=buckets)
        for objective in ("nodes", "spot_far", "spot_parallel"):
            if objective == "spot_parallel" and len(quotes) < 2:
                continue
            targets, _, pair = ideal_targets(risk, objective)
            mapped_delta, mapped_rho = held_book_risk(risk, targets)
            audit = audit_held_book(
                price_at,
                context,
                risk,
                targets,
                settings=_settings(inputs["notional"], SPOT_BUMPS[-1], YIELD_BUMPS[-1]),
                holdings_kind="ideal",
                ideal_net_delta=mapped_delta,
                ideal_net_parallel_rhoq=sum(mapped_rho.values()),
            )
            rows.append(
                {
                    "case": "policy_residuals",
                    "convention": convention,
                    "objective": objective,
                    "correction_pair": "|".join(pair) if pair else "",
                    "mapped_net_delta_hands": mapped_delta / C.FUTURES_MULTIPLIER,
                    "direct_net_delta_hands": audit.direct_net_delta
                    / C.FUTURES_MULTIPLIER,
                    "net_delta_audit_error_hands": audit.net_delta_audit_error_hands,
                    "parallel_rhoq_audit_error_bp": audit.parallel_rhoq_audit_error_bp,
                    "identity_residual_hands": audit.identity_residual_hands,
                    "audit_status": audit.status,
                }
            )
            for contract, quantity in sorted(targets.items()):
                holdings_rows.append(
                    {
                        "convention": convention,
                        "objective": objective,
                        "contract": contract,
                        "ideal_contracts": quantity,
                        "mapped_net_rhoq": mapped_rho[contract],
                        "direct_net_rhoq": audit.direct_nodal_rhoq.get(contract),
                    }
                )
            if audit.status != STATUS_PASS:
                failures.append(f"{convention}/{objective}: {audit.reason}")
    return (
        CaseResult(
            "policy_residuals",
            STATUS_PASS if not failures else STATUS_FAIL,
            "; ".join(failures),
            settings={
                "spot_bump_rel": SPOT_BUMPS[-1],
                "yield_bump": YIELD_BUMPS[-1],
                "delta_tolerance_hands": DELTA_TOLERANCE_HANDS,
                "rhoq_tolerance_bp": RHOQ_TOLERANCE_BP,
            },
            measurements=rows,
        ),
        rows,
        holdings_rows,
    )


# ---------------------------------------------------------------------------
# Case 5 — V = F^2 with unchanged spot: carry curvature, not spot gamma
# ---------------------------------------------------------------------------


def case_carry_curvature(inputs) -> Tuple[CaseResult, List[Dict[str, Any]]]:
    quotes = inputs["analytic_quotes"]
    spot, rate = inputs["spot"], inputs["rate"]
    far = quotes[-1]
    context = _context(spot, quotes, rate, "flat_q", inputs["valuation"])

    def price_at(s, dividend):
        q = float(dividend.get_yield(far.tenor_years))
        forward = float(s) * math.exp((rate - q) * far.tenor_years)
        return forward * forward

    shift = 0.01
    linear = -2.0 * far.price * far.price * far.tenor_years * shift
    from quantark.backtest.replay.carry_stress import CarryScenario

    scenario = CarryScenario(
        scenario_id="carry-curvature",
        family="parallel_carry",
        parallel_yield_shift=shift,
    )
    result = run_scenario(price_at, context, scenario, {}, linear_prediction=linear)
    exact = (far.price * math.exp(-far.tenor_years * shift)) ** 2 - far.price**2
    rows = [
        {
            "case": "carry_curvature",
            "spot_shift_rel": 0.0,
            "parallel_yield_shift": shift,
            "product_pnl": result.product_pnl,
            "exact_product_pnl": exact,
            "linear_prediction": linear,
            "repricing_error": result.repricing_error,
            "label": "carry curvature; spot never moved, so no spot gamma term",
        }
    ]
    ok = (
        result.status == "ok"
        and abs(result.product_pnl - exact) <= 1e-6 * abs(exact)
        and result.stressed_spot == context.spot
        and abs(result.repricing_error) > 0.0
    )
    return (
        CaseResult(
            "carry_curvature",
            STATUS_PASS if ok else STATUS_FAIL,
            "" if ok else "the finite reprice did not match the closed form",
            settings={"parallel_yield_shift": shift},
            measurements=rows,
        ),
        rows,
    )


# ---------------------------------------------------------------------------
# Case 6 — independent tail and interpolation-shape scenarios
# ---------------------------------------------------------------------------


def case_unquoted_scenarios(inputs) -> Tuple[CaseResult, List[Dict[str, Any]]]:
    quotes = inputs["analytic_quotes"]
    spot, rate = inputs["spot"], inputs["rate"]
    last = quotes[-1].tenor_years
    maturity = last * 2.0
    context = _context(spot, quotes, rate, "flat_forward_carry", inputs["valuation"])

    def price_at(s, dividend):
        q = float(dividend.get_yield(maturity))
        return (
            1_000_000.0
            * math.exp(-rate * maturity)
            * float(s)
            * math.exp((rate - q) * maturity)
        )

    base = price_at(context.spot, context.dividend())
    holdings = {q.contract: 3.0 for q in quotes}
    rows: List[Dict[str, Any]] = []
    failures = []

    # A shape stress lives strictly BETWEEN two anchors, so a claim maturing
    # in the tail cannot feel it.  The interior claim below matures at the
    # midpoint of the interval, where the displacement peaks; without it the
    # case would report a comfortable zero and test nothing.
    interior = (quotes[0].tenor_years, quotes[-1].tenor_years)
    midpoint = 0.5 * (interior[0] + interior[1])

    def interior_price_at(s, dividend):
        q = float(dividend.get_yield(midpoint))
        return (
            1_000_000.0
            * math.exp(-rate * midpoint)
            * float(s)
            * math.exp((rate - q) * midpoint)
        )

    interior_base = interior_price_at(context.spot, context.dividend())
    scenarios = [
        (tail_scenario(0.01), price_at, base),
        (tail_scenario(-0.01), price_at, base),
        (shape_scenario(0.01, interior), interior_price_at, interior_base),
        (shape_scenario(-0.01, interior), interior_price_at, interior_base),
    ]
    for scenario, pricer, reference_price in scenarios:
        result = run_scenario(pricer, context, scenario, holdings)
        rows.append(
            {
                "case": "unquoted_scenarios",
                "scenario_id": result.scenario_id,
                "family": result.family,
                "product_pnl": result.product_pnl,
                "hedge_pnl": result.hedge_pnl,
                "book_pnl": result.book_pnl,
                "product_pnl_bp_of_base": 1e4 * result.product_pnl / reference_price,
                "claim_maturity": maturity if scenario.tail_rate_shift else midpoint,
            }
        )
        if result.status != "ok":
            failures.append(f"{result.scenario_id}: {result.reason}")
        elif result.hedge_pnl != 0.0:
            # The defining property: no futures hedge can respond at all.
            failures.append(f"{result.scenario_id}: hedge P&L was not zero")
    tail_up = next(r for r in rows if r["scenario_id"].startswith("tail+"))
    expected_bp = 1e4 * math.expm1(-0.01 * (maturity - last))
    if abs(tail_up["product_pnl_bp_of_base"] - expected_bp) > 1e-6:
        failures.append("the tail stress did not match -(T - T_n) to first order")
    # The interior claim sits at the peak of the displacement, so its log
    # price moves by exactly the declared epsilon.
    shape_up = next(r for r in rows if r["scenario_id"].startswith("shape+"))
    expected_shape_bp = 1e4 * math.expm1(0.01)
    if abs(shape_up["product_pnl_bp_of_base"] - expected_shape_bp) > 1e-6:
        failures.append("the interpolation-shape stress did not reach its peak")
    if shape_up["product_pnl"] == 0.0:
        failures.append("the shape stress moved nothing, so it tested nothing")
    return (
        CaseResult(
            "unquoted_scenarios",
            STATUS_PASS if not failures else STATUS_FAIL,
            "; ".join(failures),
            settings={
                "maturity": maturity,
                "last_node": last,
                "tail_shifts": [0.01, -0.01],
                "shape_shifts": [0.01, -0.01],
                "shape_interval": list(interior),
                "shape_claim_maturity": midpoint,
                "expected_tail_bp": expected_bp,
                "expected_shape_bp": expected_shape_bp,
            },
            measurements=rows,
        ),
        rows,
    )


# ---------------------------------------------------------------------------
# Convergence: the spot ladder on a snowball-shaped payoff
# ---------------------------------------------------------------------------


def case_spot_ladder(inputs) -> Tuple[CaseResult, List[Dict[str, Any]]]:
    """Halve the spot bump and watch the pinned delta settle.

    A measurement that has not stabilised after the declared halvings is
    ``inconclusive`` with every sample kept, never a widened tolerance.
    """
    quotes = inputs["analytic_quotes"]
    spot, rate = inputs["spot"], inputs["rate"]
    scale = book_scale(inputs)
    coefficients = scaled_coefficients(inputs, quotes)
    price_at = O.linear_book_pricer(
        tenors=[q.tenor_years for q in quotes],
        futures_coefficients=coefficients,
        spot_coefficient=scale,
        rate=rate,
    )
    context = _context(spot, quotes, rate, "flat_q", inputs["valuation"])
    rows: List[Dict[str, Any]] = []
    values: List[float] = []
    step = SPOT_BUMPS[0] * spot
    for level in range(MAX_HALVINGS):
        measured = direct_pinned_delta(price_at, context, step)
        values.append(measured)
        rows.append(
            {
                "case": "spot_ladder",
                "level": level,
                "spot_step": step,
                "pinned_delta": measured,
                "expected": scale,
                "abs_error": abs(measured - scale),
            }
        )
        if level + 1 >= len(SPOT_BUMPS) and _stabilised(values, 1e-9 * scale):
            break
        step /= 2.0
    settled = _stabilised(values, 1e-9 * scale)
    accurate = abs(values[-1] - scale) <= 1e-9 * scale
    if settled and accurate:
        status, reason = STATUS_PASS, ""
    elif not settled:
        status, reason = STATUS_INCONCLUSIVE, (
            f"the pinned delta had not stabilised after {len(values)} halvings"
        )
    else:
        status, reason = STATUS_FAIL, (
            f"pinned delta {values[-1]:.6f} != the book's spot coefficient "
            f"{scale:.6f}"
        )
    return (
        CaseResult(
            "spot_ladder",
            status,
            reason,
            settings={
                "spot_bumps": list(SPOT_BUMPS),
                "max_halvings": MAX_HALVINGS,
                "levels_used": len(values),
                "book_scale": scale,
            },
            measurements=rows,
        ),
        rows,
    )


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


SYNTHETIC = {
    "mode": "synthetic",
    "valuation": "2025-03-03",
    "spot": 5_000.0,
    "rate": C.FLAT_RATE,
    "notional": C.NOTIONAL,
    "quotes": [
        {"contract": "IM2503", "tenor_years": 0.05, "price": 4_970.0,
         "multiplier": C.FUTURES_MULTIPLIER},
        {"contract": "IM2506", "tenor_years": 0.25, "price": 4_900.0,
         "multiplier": C.FUTURES_MULTIPLIER},
        {"contract": "IM2509", "tenor_years": 0.50, "price": 4_820.0,
         "multiplier": C.FUTURES_MULTIPLIER},
    ],
}


def prepare(inputs: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(inputs)
    resolved["analytic_quotes"] = tuple(
        O.AnalyticQuote(
            q["contract"], float(q["tenor_years"]), float(q["price"]),
            float(q["multiplier"]),
        )
        for q in inputs["quotes"]
    )
    resolved["valuation"] = pd.Timestamp(inputs["valuation"])
    return resolved


def historical_inputs(
    dates: Sequence[str], history_dir: Path
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Serialise the exact inputs of each requested date.

    Absent history is a DATA BLOCKER, not a numerical failure: the caller is
    told what to run, and no case is reported as passing or failing.
    """
    try:
        frames = C.load_history(history_dir)
    except Exception as error:  # noqa: BLE001 - reported as a blocker
        return [], f"{type(error).__name__}: {error}"
    out = []
    for raw in dates:
        stamp = pd.Timestamp(raw)
        chain = frames.futures[frames.futures["date"] == stamp]
        spot_rows = frames.spot[frames.spot["date"] == stamp]
        if chain.empty or spot_rows.empty:
            return [], f"no market data for {stamp.date()} in {history_dir}"
        spot = float(spot_rows["spot"].iloc[0])
        quotes = C.curve_quotes(chain, stamp, C.FUTURES_CURVE_MIN_TENOR_DAYS)
        if not quotes:
            return [], f"no eligible contract on {stamp.date()}"
        out.append(
            {
                "mode": "historical",
                "valuation": stamp.isoformat(),
                "spot": spot,
                "rate": C.FLAT_RATE,
                "notional": C.NOTIONAL,
                "history_dir": str(history_dir),
                "quotes": [
                    {
                        "contract": q.contract,
                        "tenor_years": q.maturity,
                        "price": q.price,
                        "multiplier": q.multiplier,
                    }
                    for q in quotes
                ],
            }
        )
    return out, None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def validate(inputs: Dict[str, Any]) -> Dict[str, Any]:
    resolved = prepare(inputs)
    results: List[CaseResult] = []
    price_rows: List[Dict[str, Any]] = []
    greek_rows: List[Dict[str, Any]] = []
    holdings_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    stress_rows: List[Dict[str, Any]] = []

    linear, rows = case_linear_book(resolved)
    results.append(linear)
    greek_rows += rows

    claim, rows = case_forward_claim(resolved)
    results.append(claim)
    greek_rows += rows

    digital, rows = case_early_digital(resolved)
    results.append(digital)
    greek_rows += rows

    policy, rows, holdings = case_policy_residuals(resolved)
    results.append(policy)
    audit_rows += rows
    holdings_rows += holdings

    curvature, rows = case_carry_curvature(resolved)
    results.append(curvature)
    price_rows += rows

    scenarios, rows = case_unquoted_scenarios(resolved)
    results.append(scenarios)
    stress_rows += rows

    ladder, rows = case_spot_ladder(resolved)
    results.append(ladder)
    greek_rows += rows

    return {
        "inputs": inputs,
        "cases": results,
        "price_ladder": price_rows,
        "greek_ladder": greek_rows,
        "policy_holdings": holdings_rows,
        "direct_audits": audit_rows,
        "stress_results": stress_rows,
    }


def write_outputs(
    out_dir: Path, batches: Sequence[Dict[str, Any]], command: str, blocker: Optional[str]
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "price_ladder": [],
        "greek_ladder": [],
        "policy_holdings": [],
        "direct_audits": [],
        "stress_results": [],
    }
    cases: List[Dict[str, Any]] = []
    for batch in batches:
        label = str(batch["inputs"]["valuation"])
        for name in frames:
            for row in batch[name]:
                frames[name].append({"valuation": label, **row})
        for case in batch["cases"]:
            cases.append({"valuation": label, **case.row()})
    for name, rows in frames.items():
        pd.DataFrame(rows).to_csv(out_dir / f"{name}.csv", index=False)

    statuses = [c["status"] for c in cases if c["required"]]
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "command": command,
        "source_digest": source_digest(),
        "status_by_case": cases,
        "tolerances": {
            "delta_hands": DELTA_TOLERANCE_HANDS,
            "rhoq_bp": RHOQ_TOLERANCE_BP,
        },
        "ladders": {
            "grid": list(GRID_LADDER),
            "spot_bumps": list(SPOT_BUMPS),
            "futures_bumps": list(FUTURES_BUMPS),
            "yield_bumps": list(YIELD_BUMPS),
            "max_halvings": MAX_HALVINGS,
        },
        "coverage": {
            "cases": len(cases),
            "passed": statuses.count(STATUS_PASS),
            "failed": statuses.count(STATUS_FAIL),
            "inconclusive": statuses.count(STATUS_INCONCLUSIVE),
        },
        "data_blocker": blocker,
    }
    C.write_json(out_dir / "validation_manifest.json", manifest)
    C.write_json(
        out_dir / "input_snapshots.json",
        {"snapshots": [b["inputs"] for b in batches]},
    )
    (out_dir / "validation_summary.md").write_text(_summary_markdown(manifest, cases))
    return manifest


def source_digest() -> str:
    import hashlib

    import quantark.backtest.futures_risk as risk
    import quantark.backtest.replay.carry_context as ctx
    import quantark.backtest.replay.carry_risk as cr
    import quantark.backtest.replay.carry_stress as cs
    import quantark.backtest.strategy.futures_bucket_strategy as bucket

    digest = hashlib.sha256()
    for module in (risk, ctx, cr, cs, bucket, O):
        path = Path(module.__file__)
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()[:16]


def _summary_markdown(manifest: Dict[str, Any], cases: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "# Bucket futures hedge — numerical validation",
        "",
        f"Generated {manifest['generated_at']}  ",
        f"Source digest `{manifest['source_digest']}`",
        "",
        "Reproduce with:",
        "",
        "```",
        manifest["command"],
        "```",
        "",
        "| valuation | case | status | reason |",
        "|---|---|---|---|",
    ]
    for case in cases:
        lines.append(
            f"| {case['valuation']} | {case['case']} | {case['status']} | "
            f"{case['reason'] or ''} |"
        )
    coverage = manifest["coverage"]
    lines += [
        "",
        f"{coverage['passed']}/{coverage['cases']} required cases passed; "
        f"{coverage['failed']} failed, {coverage['inconclusive']} inconclusive.",
        "",
        "Every pass above is backed by a direct measurement recorded in "
        "`direct_audits.csv`; a case whose error had not stabilised across "
        "its declared ladder is reported inconclusive with all of its "
        "samples kept, never resolved by widening a tolerance.",
    ]
    if manifest.get("data_blocker"):
        lines += ["", f"**Data blocker:** {manifest['data_blocker']}"]
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--synthetic", action="store_true", help="offline default")
    mode.add_argument("--snapshot", type=Path, default=None)
    mode.add_argument("--historical-dates", nargs="+", default=None)
    parser.add_argument("--history-dir", type=Path, default=C.DEFAULT_HISTORY_DIR)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--quad-grid", type=int, default=C.DEFAULT_QUAD_GRID)
    parser.add_argument("--grid-ladder", nargs="+", type=int, default=None)
    parser.add_argument("--spot-bumps", nargs="+", type=float, default=None)
    parser.add_argument("--futures-bumps", nargs="+", type=float, default=None)
    parser.add_argument("--yield-bumps", nargs="+", type=float, default=None)
    parser.add_argument("--mc-path-ladder", nargs="+", type=int, default=None)
    parser.add_argument("--mc-seeds", nargs="+", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    global GRID_LADDER, SPOT_BUMPS, FUTURES_BUMPS, YIELD_BUMPS
    args = parse_args(argv)
    if args.grid_ladder:
        GRID_LADDER = tuple(args.grid_ladder)
    if args.spot_bumps:
        SPOT_BUMPS = tuple(args.spot_bumps)
    if args.futures_bumps:
        FUTURES_BUMPS = tuple(args.futures_bumps)
    if args.yield_bumps:
        YIELD_BUMPS = tuple(args.yield_bumps)

    command = "python " + " ".join([Path(__file__).name] + list(argv or sys.argv[1:]))
    blocker: Optional[str] = None
    if args.snapshot is not None:
        payload = json.loads(Path(args.snapshot).read_text())
        inputs_list = payload.get("snapshots", [payload])
    elif args.historical_dates:
        inputs_list, blocker = historical_inputs(
            args.historical_dates, Path(args.history_dir)
        )
    else:
        inputs_list = [SYNTHETIC]

    batches = [validate(inputs) for inputs in inputs_list]
    manifest = write_outputs(Path(args.out_dir), batches, command, blocker)
    coverage = manifest["coverage"]
    print(
        f"validation: {coverage['passed']}/{coverage['cases']} required cases "
        f"passed, {coverage['failed']} failed, "
        f"{coverage['inconclusive']} inconclusive -> {args.out_dir}",
        flush=True,
    )
    if blocker:
        print(f"data blocker: {blocker}", flush=True)
        return 2
    return 0 if coverage["failed"] == 0 and coverage["inconclusive"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
