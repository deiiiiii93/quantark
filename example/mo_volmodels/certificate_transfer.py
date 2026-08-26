"""Does the banked ADI 2-D Greek certificate SPAN the states this study visits?

Gate G2 delegates delta admission for ``heston`` and ``heston_slv`` to the
banked certificate (``delta_authority: "stage16"``).  The certificate covers
seven named regime cells.  The study replays 27 inceptions over 590 distinct
calendar dates, each with its own daily Heston fit, so something has to say
whether those fits stay inside the regime span the seven cells were designed
to straddle.

That is what this module is.  It imports nothing from the numbered stages, so
stages 11, 12 and 13 can all use it without a cycle (same rule as cohort.py).

THIS IS A COVERAGE AUDIT, NOT A LICENCE
---------------------------------------
The certificate's admitted verdict is an AGGREGATE: the mean signed delta bias
across the seven cells, against a 0.1-contract economic bound (heston -0.0322,
heston_slv -0.0300).  Individually, each of the seven passed only against a
0.5-contract bound -- five times looser -- with worst single-cell errors of
-0.1066 (heston delta, low_feller), -0.1591 (SLV delta, low_feller) and
+0.1646 (gamma, near_ki).

So the +-0.10 figure that unblocked G2 is a statement about the engine's mean
bias over a designed span of regimes.  It cannot be decomposed into per-date
permissions at that bound.  Nothing here gates pricing: ``audit`` reports, and
an out-of-span state is a study finding to be read alongside the result, not a
hole punched in a daily-rehedged hedge path.  Variant admission stays
whole-variant, where ``apply_adi_greek_admission`` already puts it.

WHY A PARAMETER-CONTAINMENT TEST WOULD LICENSE NOTHING
------------------------------------------------------
Measured over 264 distinct production Heston fits pooled from the gate, fleet
and daily-pipeline caches, against the per-parameter min/max of the seven
certified cells:

    param      study min    study med    study max     certified range
    v0           0.02524      0.06493      0.16489     [0.04000, 0.14027]
    kappa        0.10169      2.01878      3.00000     [0.60000, 3.00000]
    theta        0.00418      0.08706      0.12637     [0.00306, 0.04000]
    sigma        0.00100      0.62168      0.70000     [0.00311, 0.50000]
    rho         -0.81505     -0.20154      0.00000     [-0.70000,-0.50000]

    outside the certified range: theta 97.7%, rho 92.4%, sigma 85.2%
    inside the box on ALL five parameters: 0.0%

The certified cells are regime ARCHETYPES chosen to span failure modes --
``ordinary_full`` is a designed {0.04, 2.0, 0.04, 0.30, -0.55}, not a fit --
so proximity in 5-space is the wrong question.

WHAT THE SPAN IS MEASURED ON
----------------------------
The certification traced the whole admitted bias to ONE mechanism: the order
of the variance-axis drift discretization, with the measured law
E(n_v) ~ 16/n_v.  ``AdiCore._resolve_auto_v_drift_scheme`` selects that scheme
from where the centered row loses monotonicity RELATIVE TO ``min(v0, theta)``
-- "the lowest variance level the CIR state meaningfully occupies" --
separating the harmless coordinate singularity at v=0 from genuine convection
dominance.  The Feller ratio is the study-side observable that orders states
along that axis, and the seven cells were built to straddle the split:
low_feller at 0.192, the ordinary family at 1.778, sigma_collapse at 1898.24.

Stage 11's measured cut points partition the study the same way
(``FELLER_VIOLATED_BELOW`` 0.5, ``FELLER_DEGENERATE_ABOVE`` 10.0) and every
certified cell lands in one of those three buckets, so every bucket the study
visits has a witness.

A state is IN SPAN when its bucket has a certified witness AND its Feller
ratio and remaining maturity both lie within the certified extremes.  Falling
BETWEEN two certified cells is interpolation -- that is what a seven-cell
design is for.  Going PAST the extremes is extrapolation on the one axis the
certification says drives the error, and it is reported.

Measured on the real replay window (2023-05-04 .. 2025-10-09, the last day any
of the 27 inceptions survives -- all 27 knock out, none mature or censor), a
78-fit sample found three degenerate states: 2023-06-08 at 398 and 2025-04-09
at 1901 are inside, and 2025-04-09 is in fact the very fit ``sigma_collapse``
was built from.  2024-10-10 at 35,048 is 18x past the archetype, and is the
kind of state this audit exists to name.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Feller cut points, mirrored from stage 11 so this module stays import-free.
# test_stage13_agrees_with_stage11_on_the_feller_cut_points guards the same
# pair in the aggregator; the transcription here is guarded by
# test_the_certified_envelope_matches_stage_16s_own_case_definitions.
FELLER_VIOLATED_BELOW = 0.5
FELLER_DEGENERATE_ABOVE = 10.0

SCHEMA_VERSION = 1
CERTIFICATE_STUDY = "adi2d-snowball-greeks"


@dataclass(frozen=True)
class CertifiedCell:
    """One archetype the banked certificate actually priced."""

    name: str
    feller_ratio: float
    maturity: float
    bucket: str


def _bucket(ratio: Optional[float]) -> str:
    if ratio is None or not math.isfinite(ratio):
        return "unknown"
    if ratio < FELLER_VIOLATED_BELOW:
        return "violated"
    if ratio > FELLER_DEGENERATE_ABOVE:
        return "degenerate"
    return "boundary"


def _cell(name: str, kappa: float, theta: float, sigma: float, maturity: float):
    ratio = 2.0 * kappa * theta / (sigma * sigma)
    return CertifiedCell(name, ratio, maturity, _bucket(ratio))


# The seven cells, transcribed from 16_adi_greek_certification.py's ORDINARY /
# LOW_FELLER / SIGMA_COLLAPSE parameter sets and certification_cases().  Stage
# 16 is an implementation input to its OWN certification hash, so it is read
# by the test rather than imported here.
CERTIFIED_CELLS: Dict[str, CertifiedCell] = {
    cell.name: cell
    for cell in (
        _cell("ordinary_full", 2.0, 0.04, 0.30, 3.0),
        _cell("ordinary_decayed", 2.0, 0.04, 0.30, 1.50),
        _cell("near_ko", 2.0, 0.04, 0.30, 1.0),
        _cell("near_ki", 2.0, 0.04, 0.30, 1.0),
        _cell("low_feller", 0.60, 0.04, 0.50, 2.0),
        _cell("sigma_collapse", 3.0, 0.00306, 0.00311, 3.0),
        _cell("near_expiry", 2.0, 0.04, 0.30, 0.25),
    )
}

CERTIFIED_RATIO_MIN = min(c.feller_ratio for c in CERTIFIED_CELLS.values())
CERTIFIED_RATIO_MAX = max(c.feller_ratio for c in CERTIFIED_CELLS.values())

# The envelope endpoints are only specified to the precision of the archetype
# parameters they are computed from, so comparing to them at float precision
# is spurious.  ``sigma_collapse`` is the 2025-04-09 production fit quoted to
# three significant figures (design spec section 5.9: "sigma = 0.00311,
# kappa = 3.0, theta = 0.00306, v0 = 0.14027"), and stage 16 hardcodes exactly
# those.  That quote pins its ratio only to [1889.0627, 1907.4735] -- a
# half-width of 0.4849% -- while the fit itself sits at 1901.3981, +0.1662%
# above the 1898.2434 the rounded values give.
#
# Without this the audit would report the very date the archetype was BUILT
# FROM as out of span, which would be a transcription artefact reported as a
# study finding.  0.5% is that measured half-width, rounded up; it is far too
# tight to admit a genuinely different regime -- 2024-10-10 at 35,048 is 18x
# out, and even a ratio 2% past the endpoint stays out.
#
# The maturity axis carries the same kind of imprecision.  An archetype at
# "T = 3.0" means a three-year trade, and a real three-year trade's day-count
# maturity on its inception day is 3*365/365.25 = 2.9979 to 3*366/365.25 =
# 3.0062 depending on leap days -- the 27 inceptions here land at 3.001 to
# 3.006.  Reporting the first day of a three-year trade as outside a span
# whose widest cell IS a three-year trade would be a day-count artefact
# dressed up as a study finding, so the same tolerance applies to both axes.
ENDPOINT_REL_TOL = 0.005
CERTIFIED_MATURITY_MIN = min(c.maturity for c in CERTIFIED_CELLS.values())
CERTIFIED_MATURITY_MAX = max(c.maturity for c in CERTIFIED_CELLS.values())

BUCKETS = ("violated", "boundary", "degenerate", "unknown")


@dataclass(frozen=True)
class SpanVerdict:
    """Whether the certificate's regime span reaches one visited state."""

    in_span: bool
    bucket: str
    feller_ratio: Optional[float]
    witness: Optional[str]  # the nearest certified cell, when in span
    reason: str


def feller_ratio(params: Mapping[str, float]) -> Optional[float]:
    """2*kappa*theta / sigma**2 for a calibrated Heston parameter set.

    Returns None when the state cannot be ranked at all: a missing or
    malformed parameter, or a sigma of zero, which sits outside the
    calibration's own lower bound (0.001 under the mo_frozen preset) and so
    cannot come from a real fit.
    """
    try:
        kappa = float(params["kappa"])
        theta = float(params["theta"])
        sigma = float(params["sigma"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (sigma > 0.0) or not math.isfinite(kappa * theta):
        return None
    ratio = 2.0 * kappa * theta / (sigma * sigma)
    return ratio if math.isfinite(ratio) else None


def _witness(bucket: str, ratio: float) -> Optional[str]:
    """The certified cell in this bucket closest to ``ratio`` in log-ratio."""
    peers = [c for c in CERTIFIED_CELLS.values() if c.bucket == bucket]
    if not peers:
        return None
    return min(
        peers,
        key=lambda c: abs(math.log(max(ratio, 1e-300)) - math.log(c.feller_ratio)),
    ).name


def state_in_span(
    params: Mapping[str, float], *, remaining_years: float
) -> SpanVerdict:
    """Does the certificate's regime span reach this calibrated state?

    ``params`` is one production Heston fit (kappa, theta, sigma at minimum).
    ``remaining_years`` is the trade's remaining maturity on the replay day.

    Fails closed: a state that cannot be ranked comes back OUT of span, never
    in-span-by-default.  Being out of span is a reported finding -- it does
    not exclude the date from pricing, and no MC Greeks are substituted for
    it (see the module docstring on why this is an audit and not a licence).
    """
    ratio = feller_ratio(params)
    bucket = _bucket(ratio)
    if ratio is None:
        return SpanVerdict(
            False, bucket, None, None,
            "Feller ratio is not computable from this fit, so the state "
            "cannot be placed against the certified regimes",
        )

    try:
        remaining = float(remaining_years)
    except (TypeError, ValueError):
        remaining = float("nan")
    if not math.isfinite(remaining) or not (
        CERTIFIED_MATURITY_MIN * (1.0 - ENDPOINT_REL_TOL)
        <= remaining
        <= CERTIFIED_MATURITY_MAX * (1.0 + ENDPOINT_REL_TOL)
    ):
        return SpanVerdict(
            False, bucket, ratio, None,
            f"remaining maturity {remaining:.4g}y is outside the certified "
            f"maturities [{CERTIFIED_MATURITY_MIN:g}, "
            f"{CERTIFIED_MATURITY_MAX:g}]y",
        )

    if ratio > CERTIFIED_RATIO_MAX * (1.0 + ENDPOINT_REL_TOL):
        return SpanVerdict(
            False, bucket, ratio, None,
            f"Feller ratio {ratio:,.4g} is past the most extreme certified "
            f"cell (sigma_collapse at {CERTIFIED_RATIO_MAX:,.4f}); the "
            "certification says this axis drives the error, so beyond it is "
            "extrapolation",
        )
    if ratio < CERTIFIED_RATIO_MIN * (1.0 - ENDPOINT_REL_TOL):
        return SpanVerdict(
            False, bucket, ratio, None,
            f"Feller ratio {ratio:,.4g} is below the lowest certified cell "
            f"(low_feller at {CERTIFIED_RATIO_MIN:,.4f})",
        )

    witness = _witness(bucket, ratio)
    if witness is None:
        return SpanVerdict(
            False, bucket, ratio, None,
            f"no certified cell falls in the {bucket!r} regime",
        )
    return SpanVerdict(
        True, bucket, ratio, witness,
        f"inside the certified span, bracketed by {witness!r} "
        f"({CERTIFIED_CELLS[witness].feller_ratio:,.4f})",
    )


def audit(
    states: Iterable[Tuple[str, Mapping[str, float], float]]
) -> Dict[str, Any]:
    """Coverage report over the (label, fit, remaining_years) the fleet visits.

    Never raises on an out-of-span state -- that is the whole posture.  The
    report names them so the result is read with the caveat attached.
    """
    rows: List[Dict[str, Any]] = []
    ratios: List[float] = []
    buckets = {name: 0 for name in BUCKETS}
    out_of_span: List[Dict[str, Any]] = []

    for label, params, remaining in states:
        verdict = state_in_span(params, remaining_years=remaining)
        buckets[verdict.bucket] += 1
        if verdict.feller_ratio is not None:
            ratios.append(verdict.feller_ratio)
        row = {
            "label": label,
            "bucket": verdict.bucket,
            "feller_ratio": verdict.feller_ratio,
            "remaining_years": remaining,
            "in_span": verdict.in_span,
            "witness": verdict.witness,
            "reason": verdict.reason,
        }
        rows.append(row)
        if not verdict.in_span:
            out_of_span.append(row)

    n = len(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": "certificate-transfer",
        "study": CERTIFICATE_STUDY,
        "posture": "report-only coverage audit; nothing here gates pricing",
        "certificate": {
            "ratio_envelope": [CERTIFIED_RATIO_MIN, CERTIFIED_RATIO_MAX],
            "endpoint_rel_tol": ENDPOINT_REL_TOL,
            "maturity_envelope": [CERTIFIED_MATURITY_MIN, CERTIFIED_MATURITY_MAX],
            "cells": {
                name: {
                    "feller_ratio": cell.feller_ratio,
                    "maturity": cell.maturity,
                    "bucket": cell.bucket,
                }
                for name, cell in CERTIFIED_CELLS.items()
            },
            "aggregate_bound_contracts": 0.1,
            "per_cell_bound_contracts": 0.5,
        },
        "n_states": n,
        "buckets": buckets,
        "n_out_of_span": len(out_of_span),
        "out_of_span": out_of_span,
        "feller_ratio": {
            "n": len(ratios),
            "min": min(ratios) if ratios else None,
            "median": float(statistics.median(ratios)) if ratios else None,
            "max": max(ratios) if ratios else None,
        },
        # None, not True: nothing measured is not the same as nothing wrong.
        "covered": None if n == 0 else not out_of_span,
    }
