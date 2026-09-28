"""Pure risk metrics and paired uncertainty for the bucket hedge study.

Nothing here prices, loads or runs anything: it takes recorded frames and
returns statements about them, so the acceptance rules can be tested against
synthetic panels with known answers.

Five conclusions are reported SEPARATELY, because they are different claims
(revised design section 10.2):

``numerical_validity``
    the independent measurements agreed with the predicted residuals;
``objective_achieved``
    the policy's own targets were met;
``joint_mitigation``
    RMS spot-shock AND RMS parallel-rhoq exposure both fell;
``broader_carry_mitigation``
    gross nodal exposure and finite curve-shape losses support it too;
``economic_comparison``
    realised P&L variability, tail loss and costs, paired against controls.

The order matters: a policy can pass the first two and fail the rest.  Zero
parallel rhoq alone never earns the fourth, and none of them is inferred
from another.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

# Research defaults, fixed BEFORE looking at comparative performance.
DEFAULT_REPLICATES = 2_000
DEFAULT_SEED = 9092026
DEFAULT_BLOCK_LENGTH = 20
SENSITIVITY_BLOCK_LENGTHS = (10, 40)
DEFAULT_INTERVAL = (0.025, 0.975)

#: Every conclusion a report may carry.  ``unsupported`` is a real answer;
#: ``inconclusive`` means the evidence could not decide.
STATUSES = ("supported", "unsupported", "inconclusive", "not_available")


@dataclass(frozen=True)
class Conclusion:
    """One claim, its status, why, and the numbers behind it."""

    name: str
    status: str
    reasons: Tuple[str, ...] = ()
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")

    @property
    def supported(self) -> bool:
        return self.status == "supported"


@dataclass(frozen=True)
class RatioResult:
    """An RMS ratio that refuses to divide by a numerically zero denominator."""

    numerator: float
    denominator: float
    ratio: float
    coverage: int
    within_absolute_budget: Optional[bool]
    budget: float

    @property
    def improved(self) -> Optional[bool]:
        if self.within_absolute_budget is not None:
            return self.within_absolute_budget
        if not math.isfinite(self.ratio):
            return None
        return self.ratio < 1.0


def rms(values: Sequence[float]) -> float:
    """Root mean square over the FINITE entries; NaN when there are none."""
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(finite))))


def rms_ratio(
    net: Sequence[float],
    product: Sequence[float],
    *,
    budget: float,
) -> RatioResult:
    """``RMS(net) / RMS(product)``, or an absolute check when it cannot divide.

    An unhedged exposure inside the numerical budget makes the ratio
    meaningless: dividing by it would manufacture an enormous number or an
    accidental pass.  The absolute question is asked instead, and the caller
    is told which one it answered.
    """
    net_values = np.asarray(list(net), dtype=float)
    product_values = np.asarray(list(product), dtype=float)
    if net_values.shape != product_values.shape:
        raise ValueError("net and product exposure series must align")
    mask = np.isfinite(net_values) & np.isfinite(product_values)
    numerator = rms(net_values[mask])
    denominator = rms(product_values[mask])
    coverage = int(mask.sum())
    if coverage == 0:
        return RatioResult(
            float("nan"), float("nan"), float("nan"), 0, None, float(budget)
        )
    if not math.isfinite(denominator) or denominator <= float(budget):
        return RatioResult(
            numerator,
            denominator,
            float("nan"),
            coverage,
            bool(numerator <= float(budget)),
            float(budget),
        )
    return RatioResult(
        numerator,
        denominator,
        numerator / denominator,
        coverage,
        None,
        float(budget),
    )


def exceedances(values: Sequence[float], limit: float) -> Dict[str, Any]:
    """Maximum and date-level breaches, so an average cannot imply neutrality."""
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"max_abs": float("nan"), "exceedances": 0, "dates": 0}
    return {
        "max_abs": float(np.max(np.abs(finite))),
        "exceedances": int(np.sum(np.abs(finite) > float(limit))),
        "dates": int(finite.size),
    }


# ---------------------------------------------------------------------------
# Paired calendar-block resampling
# ---------------------------------------------------------------------------


def moving_block_indices(n_dates: int, block_length: int, rng) -> np.ndarray:
    if not 1 <= block_length <= n_dates:
        raise ValueError("block length must fit the observation window")
    pieces = []
    remaining = n_dates
    while remaining:
        start = int(rng.integers(0, n_dates - block_length + 1))
        take = min(block_length, remaining)
        pieces.append(np.arange(start, start + take))
        remaining -= take
    return np.concatenate(pieces)


def paired_interval(
    candidate: np.ndarray,
    control: np.ndarray,
    statistic,
    *,
    block_length: int,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
    quantiles: Tuple[float, float] = DEFAULT_INTERVAL,
) -> np.ndarray:
    """Percentile interval of ``statistic(candidate) - statistic(control)``.

    The SAME sampled calendar indices are applied to both panels and to every
    inception column, so overlapping runs are resampled together rather than
    treated as independent samples.
    """
    if candidate.shape != control.shape:
        raise ValueError("paired panels must have identical shape")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        idx = moving_block_indices(candidate.shape[0], block_length, rng)
        samples.append(statistic(candidate[idx]) - statistic(control[idx]))
    return np.quantile(samples, list(quantiles))


@dataclass(frozen=True)
class PairedPanel:
    """Matched candidate/control observations on one calendar."""

    dates: Tuple[Any, ...]
    inceptions: Tuple[Any, ...]
    candidate: np.ndarray
    control: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        if self.candidate.shape != self.control.shape:
            raise ValueError("paired panels must have identical shape")
        if self.candidate.shape != self.valid.shape:
            raise ValueError("the validity mask must match the panel shape")
        if self.candidate.shape != (len(self.dates), len(self.inceptions)):
            raise ValueError("panel shape must be dates x inceptions")

    @property
    def matched(self) -> int:
        return int(np.sum(self.valid))

    def masked(self) -> Tuple[np.ndarray, np.ndarray]:
        """Both panels with unmatched observations set to NaN."""
        candidate = np.where(self.valid, self.candidate, np.nan)
        control = np.where(self.valid, self.control, np.nan)
        return candidate, control


def build_panel(
    candidate_rows: Dict[Any, Dict[Any, float]],
    control_rows: Dict[Any, Dict[Any, float]],
    *,
    valid_rows: Optional[Dict[Any, Dict[Any, bool]]] = None,
) -> PairedPanel:
    """A date-by-inception panel from ``{inception: {date: value}}`` maps.

    An observation counts only when BOTH sides have it and the run was
    valid, so a comparison never quietly drops the unfavourable side.
    """
    inceptions = tuple(sorted(set(candidate_rows) & set(control_rows)))
    dates = sorted(
        {d for i in inceptions for d in candidate_rows[i]}
        & {d for i in inceptions for d in control_rows[i]}
    )
    dates = tuple(dates)
    shape = (len(dates), len(inceptions))
    candidate = np.full(shape, np.nan)
    control = np.full(shape, np.nan)
    valid = np.zeros(shape, dtype=bool)
    for column, inception in enumerate(inceptions):
        for row, date in enumerate(dates):
            left = candidate_rows[inception].get(date)
            right = control_rows[inception].get(date)
            if left is None or right is None:
                continue
            ok = True
            if valid_rows is not None:
                ok = bool(valid_rows.get(inception, {}).get(date, True))
            candidate[row, column] = float(left)
            control[row, column] = float(right)
            valid[row, column] = ok and math.isfinite(float(left)) and math.isfinite(
                float(right)
            )
    return PairedPanel(dates, inceptions, candidate, control, valid)


def mean_of_inception_sds(panel: np.ndarray) -> float:
    """Mean over inceptions of each column's sample standard deviation."""
    sds = []
    for column in range(panel.shape[1]):
        values = panel[:, column]
        finite = values[np.isfinite(values)]
        if finite.size > 1:
            sds.append(float(np.std(finite, ddof=1)))
    return float(np.mean(sds)) if sds else float("nan")


def matched_rms(panel: np.ndarray) -> float:
    return rms(panel[np.isfinite(panel)])


def paired_report(
    panel: PairedPanel,
    statistic,
    *,
    block_length: int = DEFAULT_BLOCK_LENGTH,
    replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
    sensitivity: Sequence[int] = SENSITIVITY_BLOCK_LENGTHS,
) -> Dict[str, Any]:
    """The paired difference and its interval, with its settings recorded.

    A window too short for the requested block length reports that the
    interval cannot be estimated, rather than silently shortening the block
    and claiming an answer at a dependence scale it never tested.
    """
    candidate, control = panel.masked()
    point = statistic(candidate) - statistic(control)
    report: Dict[str, Any] = {
        "point": float(point),
        "matched_observations": panel.matched,
        "dates": len(panel.dates),
        "inceptions": len(panel.inceptions),
        "block_length": int(block_length),
        "replicates": int(replicates),
        "seed": int(seed),
        "quantiles": list(DEFAULT_INTERVAL),
        "resampling_unit": "calendar day, blocks shared by every inception",
    }
    if len(panel.dates) < block_length:
        report["interval"] = None
        report["interval_status"] = (
            f"window of {len(panel.dates)} dates is shorter than the requested "
            f"{block_length}-day block"
        )
        return report
    low, high = paired_interval(
        candidate,
        control,
        statistic,
        block_length=block_length,
        replicates=replicates,
        seed=seed,
    )
    report["interval"] = [float(low), float(high)]
    report["interval_status"] = "estimated"
    report["excludes_zero"] = bool(low > 0.0 or high < 0.0)
    sensitivities = {}
    for alternative in sensitivity:
        if alternative > len(panel.dates) or alternative == block_length:
            continue
        low_alt, high_alt = paired_interval(
            candidate,
            control,
            statistic,
            block_length=int(alternative),
            replicates=replicates,
            seed=seed,
        )
        sensitivities[str(alternative)] = [float(low_alt), float(high_alt)]
    report["block_sensitivity"] = sensitivities
    return report


# ---------------------------------------------------------------------------
# The five conclusions
# ---------------------------------------------------------------------------


def numerical_validity(coverage: Dict[str, Any], *, requested_mode: str) -> Conclusion:
    """Did the independent measurements agree, and did they cover the run?"""
    state = coverage.get("audit_coverage")
    if state == "not_available":
        return Conclusion(
            "numerical_validity",
            "not_available",
            ("this run predates carry recording",),
            dict(coverage),
        )
    counts = coverage.get("by_status", {})
    reasons = []
    if state != "measured":
        reasons.append("no audit was measured")
    if counts.get("fail"):
        reasons.append(f"{counts['fail']} failed audit date(s)")
    if counts.get("inconclusive"):
        reasons.append(f"{counts['inconclusive']} inconclusive audit date(s)")
    measured = coverage.get("measured", 0)
    dates = coverage.get("dates", 0)
    if requested_mode == "daily" and dates and measured < dates:
        # A sampled run must not be reported as daily coverage.
        reasons.append(
            f"daily coverage was requested but only {measured}/{dates} dates "
            "were measured"
        )
    if not reasons:
        return Conclusion("numerical_validity", "supported", (), dict(coverage))
    status = "inconclusive" if counts.get("inconclusive") and not counts.get(
        "fail"
    ) else "unsupported"
    return Conclusion("numerical_validity", status, tuple(reasons), dict(coverage))


def objective_achieved(
    *,
    delta_residual_hands: Sequence[float],
    parallel_residual_bp: Sequence[float],
    delta_budget: float,
    rhoq_budget: float,
    ideal_delta_hands: Sequence[float] = (),
    ideal_parallel_bp: Sequence[float] = (),
) -> Conclusion:
    """Were the policy's own targets met, ideally and then actually?"""
    delta = exceedances(delta_residual_hands, delta_budget)
    parallel = exceedances(parallel_residual_bp, rhoq_budget)
    ideal = {
        "delta": exceedances(ideal_delta_hands, delta_budget),
        "parallel": exceedances(ideal_parallel_bp, rhoq_budget),
    }
    evidence = {"actual_delta": delta, "actual_parallel": parallel, "ideal": ideal}
    reasons = []
    if delta["exceedances"]:
        reasons.append(
            f"{delta['exceedances']}/{delta['dates']} dates exceed the delta budget"
        )
    if parallel["exceedances"]:
        reasons.append(
            f"{parallel['exceedances']}/{parallel['dates']} dates exceed the "
            "parallel rhoq budget"
        )
    if not delta["dates"] or not parallel["dates"]:
        return Conclusion(
            "objective_achieved", "not_available", ("no residuals recorded",), evidence
        )
    status = "supported" if not reasons else "unsupported"
    return Conclusion("objective_achieved", status, tuple(reasons), evidence)


def joint_mitigation(spot: RatioResult, parallel: RatioResult) -> Conclusion:
    """Did BOTH the spot-shock and parallel-rhoq exposures fall?"""
    evidence = {"spot_1pct_bp": _ratio_payload(spot), "parallel_rhoq_bp": _ratio_payload(parallel)}
    reasons = []
    for name, result in (("spot", spot), ("parallel rhoq", parallel)):
        improved = result.improved
        if improved is None:
            reasons.append(f"{name} exposure could not be compared")
        elif not improved:
            reasons.append(f"{name} exposure did not fall")
    if any("could not be compared" in r for r in reasons):
        return Conclusion("joint_mitigation", "inconclusive", tuple(reasons), evidence)
    status = "supported" if not reasons else "unsupported"
    return Conclusion("joint_mitigation", status, tuple(reasons), evidence)


def broader_carry_mitigation(
    joint: Conclusion,
    gross: RatioResult,
    *,
    scenario_losses: Dict[str, Dict[str, float]],
) -> Conclusion:
    """An ADDITIONAL claim, never inferred from the parallel column alone."""
    evidence = {
        "joint": joint.status,
        "gross_nodal_rhoq_bp": _ratio_payload(gross),
        "scenarios": scenario_losses,
    }
    reasons = []
    if not joint.supported:
        reasons.append("joint mitigation is not supported")
    improved = gross.improved
    if improved is None:
        reasons.append("gross nodal exposure could not be compared")
    elif not improved:
        reasons.append("gross nodal exposure did not fall")
    for name, payload in sorted(scenario_losses.items()):
        candidate = float(payload.get("candidate_max_loss", float("nan")))
        control = float(payload.get("control_max_loss", float("nan")))
        if not (math.isfinite(candidate) and math.isfinite(control)):
            reasons.append(f"scenario {name} was not measured on both sides")
        elif candidate > control:
            reasons.append(f"scenario {name} loses more than the control")
    if any("could not be compared" in r or "not measured" in r for r in reasons):
        return Conclusion(
            "broader_carry_mitigation", "inconclusive", tuple(reasons), evidence
        )
    status = "supported" if not reasons else "unsupported"
    return Conclusion("broader_carry_mitigation", status, tuple(reasons), evidence)


def economic_comparison(reports: Dict[str, Dict[str, Any]]) -> Conclusion:
    """Paired realised performance: supported, unsupported or inconclusive.

    There is no universal threshold here.  A measure whose interval spans
    zero has not decided, and a candidate can improve variability while
    losing on cost.
    """
    evidence = dict(reports)
    if not reports:
        return Conclusion(
            "economic_comparison", "not_available", ("no paired measures",), evidence
        )
    undecided = []
    worse = []
    for name, report in sorted(reports.items()):
        interval = report.get("interval")
        if interval is None:
            undecided.append(f"{name}: {report.get('interval_status', 'no interval')}")
            continue
        low, high = interval
        if low <= 0.0 <= high:
            undecided.append(f"{name}: interval spans zero")
        elif low > 0.0:
            worse.append(f"{name}: worse than the control")
    if worse:
        return Conclusion(
            "economic_comparison", "unsupported", tuple(worse + undecided), evidence
        )
    if undecided:
        return Conclusion(
            "economic_comparison", "inconclusive", tuple(undecided), evidence
        )
    return Conclusion("economic_comparison", "supported", (), evidence)


def _ratio_payload(result: RatioResult) -> Dict[str, Any]:
    return {
        "numerator": result.numerator,
        "denominator": result.denominator,
        "ratio": result.ratio,
        "coverage": result.coverage,
        "within_absolute_budget": result.within_absolute_budget,
        "budget": result.budget,
        "improved": result.improved,
    }


def conclusions_payload(conclusions: Sequence[Conclusion]) -> Dict[str, Any]:
    """The five results as a serialisable block, in their reporting order."""
    return {
        c.name: {
            "status": c.status,
            "reasons": list(c.reasons),
            "evidence": c.evidence,
        }
        for c in conclusions
    }
