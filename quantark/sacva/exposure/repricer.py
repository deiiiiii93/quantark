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
    if spots.ndim != 2:
        raise ValidationError("spots must be 2-D (num_paths, n_times)")
    if not np.all(np.isfinite(spots)):
        raise ValidationError("spots must be finite")
    if isinstance(quantity, bool) or not isinstance(quantity, (int, float)) \
            or not np.isfinite(quantity):
        raise ValidationError("quantity must be a finite number")
    times = np.asarray(times, dtype=float)
    if not np.all(np.isfinite(times)):
        raise ValidationError("times must be finite")
    n_paths, n_t = spots.shape
    if times.shape != (n_t,):
        raise ValidationError("times length must match spots' time axis")
    masks = {}
    for key in ("alive", "knocked_in"):
        if key not in state:
            raise ValidationError(f"state must contain '{key}'")
        arr = np.asarray(state[key])
        if arr.dtype != np.bool_ or arr.shape != (n_paths, n_t):
            raise ValidationError(
                f"state['{key}'] must be a boolean array of shape (num_paths, n_times)")
        masks[key] = arr
    for j in exposure_idx:
        if isinstance(j, bool) or not isinstance(j, (int, np.integer)):
            raise ValidationError("exposure_idx entries must be integers")
        if j < 0 or j >= n_t:
            raise ValidationError("exposure_idx out of range")
    out = np.zeros((n_paths, len(exposure_idx)), dtype=float)
    has_ki = "knocked_in" in state_labels
    for col, j in enumerate(exposure_idx):
        alive = masks["alive"][:, j]
        ki = masks["knocked_in"][:, j]
        for label in state_labels:
            if has_ki:
                sel = alive & (ki if label == "knocked_in" else ~ki)
            else:
                sel = alive
            if not sel.any():
                continue
            vals = np.asarray(surface.value_at(spots[sel, j], float(times[j]), label),
                              dtype=float)
            if vals.shape != (int(sel.sum()),) or not np.all(np.isfinite(vals)):
                raise ValidationError(
                    "surface.value_at must return a finite 1-D array of the selected "
                    "path count")
            out[sel, col] = vals
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
    if isinstance(n_dates, bool) or not isinstance(n_dates, int) or n_dates < 1:
        raise ValidationError("n_dates must be a positive int")
    if not np.isfinite(redemption):
        raise ValidationError("redemption must be finite")
    if settlement_idx is not None:
        if isinstance(settlement_idx, bool) or not isinstance(settlement_idx, int) \
                or not (0 <= settlement_idx <= n_dates):
            raise ValidationError("settlement_idx must be an int in [0, n_dates]")
    else:
        if isinstance(settlement_offset_steps, bool) \
                or not isinstance(settlement_offset_steps, int) \
                or settlement_offset_steps < 1:
            raise ValidationError("settlement_offset_steps must be an int >= 1")
    ko_in = np.asarray(ko_idx)
    if ko_in.ndim != 1:
        raise ValidationError("ko_idx must be 1-D")
    if not np.issubdtype(ko_in.dtype, np.integer):  # float KO idx would be truncated
        raise ValidationError("ko_idx must have integer dtype")
    ko_idx = ko_in.astype(int)
    if np.any(ko_idx < -1) or np.any(ko_idx >= n_dates):
        raise ValidationError("ko_idx entries must be -1 or in [0, n_dates)")
    out = np.zeros((ko_idx.shape[0], n_dates), dtype=float)
    for p, k in enumerate(ko_idx):
        if k < 0:
            continue
        if settlement_idx is not None:
            if settlement_idx <= k:
                raise ValidationError(
                    f"settlement_idx {settlement_idx} precedes KO date {k}")
            settle = settlement_idx
        else:
            settle = min(k + settlement_offset_steps, n_dates)
        out[p, k:settle] = redemption
    return out
