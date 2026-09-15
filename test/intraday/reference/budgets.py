"""Frozen Gate C acceptance budgets (Plan 2 Task 1).

Fixed from the design's normalised tolerances BEFORE any intraday route was
tuned, with the reference's own convergence evidence recorded in
docs/superpowers/plans/2026-09-15-intraday-baseline.md. Do not edit: a route
that misses a budget reports ``unqualified`` for that cell.
"""

#: Price: 0.01 bp of initial notional, i.e. this factor x initial_price x contract_multiplier.
PRICE_ABS_PER_NOTIONAL = 1e-6
#: Normalised delta d* = Delta * S_ref / N: absolute and relative tolerance (whichever is larger).
DELTA_ABS, DELTA_REL = 1e-5, 1e-4
#: Normalised gamma g* = Gamma * S_ref^2 / N.
GAMMA_ABS, GAMMA_REL = 1e-4, 1e-3
#: A route passes iff |route - ref| <= budget + REFERENCE_UNCERTAINTY_MULTIPLIER * ref_uncertainty.
REFERENCE_UNCERTAINTY_MULTIPLIER = 3.0


def price_budget(notional: float) -> float:
    return PRICE_ABS_PER_NOTIONAL * float(notional)


def delta_budget(reference_delta: float, spot: float, notional: float) -> float:
    """Budget on Delta (per unit spot) from the normalised tolerance d* = Delta S / N."""
    normalised = abs(reference_delta) * spot / notional
    return max(DELTA_ABS, DELTA_REL * normalised) * notional / spot


def gamma_budget(reference_gamma: float, spot: float, notional: float) -> float:
    """Budget on Gamma (per unit spot^2) from the normalised tolerance g* = Gamma S^2 / N."""
    normalised = abs(reference_gamma) * spot * spot / notional
    return max(GAMMA_ABS, GAMMA_REL * normalised) * notional / (spot * spot)


#: Reference grid levels used by Gate C. (2001, 4001, 8001) left one 6h gamma cell (gamma crossing zero,
#: absolute floor) at 3*unc/budget = 1.29; doubling cut that uncertainty 4x (clean O(h^2)) to 0.32.
GATE_C_POINTS = (4001, 8001, 16001)

#: Horizons (time to fixing) at which the reference's own uncertainty exceeds a budget for a measure;
#: filled from the Step 4 evidence run, read by the Gate C harness ("inconclusive by construction").
REFERENCE_LIMITED = {
    "price": (),
    "delta": (),
    "gamma": (),
}
