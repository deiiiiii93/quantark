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


#: Point vega / rho / dividend rho (per unit vol, rate, yield), frozen with Plan 3 Task 5 BEFORE any greek ladder ran:
#: normalised m* = Measure * 0.01 / N (PnL of a one-point move per unit notional), the price budget's absolute level.
MOVE_ABS, MOVE_REL = 1e-6, 1e-4
POINT_MOVE = 0.01


def move_budget(reference_value: float, notional: float) -> float:
    """Budget on a per-unit-vol / per-unit-rate sensitivity from m* = value * POINT_MOVE / N."""
    normalised = abs(reference_value) * POINT_MOVE / notional
    return max(MOVE_ABS, MOVE_REL * normalised) * notional / POINT_MOVE


#: Theta (PnL per hour), frozen with the 2026-09-16 re-review fixes BEFORE any theta ladder ran: normalised
#: t* = |theta per hour| / N (PnL of a one-hour roll per unit notional), at the move budget's levels. One hour is the
#: default desk roll, so t* is directly comparable with a move of that roll.
THETA_ABS, THETA_REL = 1e-6, 1e-4


def theta_budget(reference_theta_per_hour: float, notional: float) -> float:
    """Budget on a theta per hour from t* = |theta| / N."""
    return max(THETA_ABS, THETA_REL * abs(reference_theta_per_hour) / notional) * notional


def desk_move_budget(reference_move: float, notional: float) -> float:
    """Budget on a desk vega / rho / dividend rho, whose value IS the PnL of a one-point move: m* = |move| / N."""
    return max(MOVE_ABS, MOVE_REL * abs(reference_move) / notional) * notional


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
