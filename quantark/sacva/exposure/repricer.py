"""Value-surface repricing + KO/coupon pending receivables (spec §3.2).

Turns a per-trade ``(ValueSurface, state)`` into pathwise reporting-currency
values across the exposure dates, applying ``quantity`` once and selecting the
state-matching surface. KO/coupon receivables are produced UNDISCOUNTED (the
aggregator applies the discount factor once — returning ``df*amount`` here would
double-discount).
"""

import numpy as np

from quantark.util.exceptions import ValidationError


def reprice_trade(surface, spots, state, times, quantity, exposure_idx,
                  state_labels=(None,)):
    """Pathwise trade values, shape (num_paths, len(exposure_idx)).

    ``state_labels`` must match the surface's keys: ``(None,)`` for a single-state
    (vanilla/grid) surface; ``("alive", "knocked_in")`` for autocallables.
    """
    spots = np.asarray(spots, dtype=float)
    n_paths = spots.shape[0]
    out = np.zeros((n_paths, len(exposure_idx)), dtype=float)
    has_ki = "knocked_in" in state_labels
    for col, j in enumerate(exposure_idx):
        alive = state["alive"][:, j]
        ki = state["knocked_in"][:, j]
        for label in state_labels:
            if has_ki:
                sel = alive & (ki if label == "knocked_in" else ~ki)
            else:
                sel = alive
            if not sel.any():
                continue
            out[sel, col] = surface.value_at(spots[sel, j], float(times[j]), label)
        # dead paths stay 0 (their cashflow is handled by pending_receivable)
    return out * float(quantity)


def pending_receivable_exposure(ko_idx, redemption, n_dates,
                                settlement_idx=None, settlement_offset_steps=None):
    """UNDISCOUNTED receivable exposure, shape (num_paths, n_dates).

    For paths that knocked out (``ko_idx >= 0``), the ``redemption`` is carried at
    exposure dates in ``[ko_idx, settle)`` and 0 from settlement on. Exactly one
    of ``settlement_idx`` (fixed) / ``settlement_offset_steps`` (per-path offset
    from the KO date) must be supplied.
    """
    if (settlement_idx is None) == (settlement_offset_steps is None):
        raise ValidationError(
            "supply exactly one of settlement_idx / settlement_offset_steps")
    ko_idx = np.asarray(ko_idx, dtype=int)
    out = np.zeros((ko_idx.shape[0], n_dates), dtype=float)
    for p, k in enumerate(ko_idx):
        if k < 0:
            continue
        settle = (settlement_idx if settlement_idx is not None
                  else min(k + settlement_offset_steps, n_dates))
        out[p, k:settle] = redemption
    return out
