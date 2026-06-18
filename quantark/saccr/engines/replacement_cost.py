"""Replacement Cost (RC) calculation for SA-CCR.

Reference: Basel Committee SA-CCR document, paragraphs 130-145.
"""


def calculate_rc_unmargined(v: float, c: float = 0.0) -> float:
    """RC for unmargined transactions: ``max(V - C, 0)`` (paragraph 136)."""
    return max(v - c, 0.0)


def calculate_rc_margined(
    v: float,
    c: float,
    th: float,
    mta: float,
    nica: float,
) -> float:
    """RC for margined transactions: ``max(V - C, TH + MTA - NICA, 0)`` (paragraph 144).

    The largest exposure that would not trigger a variation-margin call.

    Example (Annex 4b, Example 2): V=80, C=79.5, TH=0, MTA=1, NICA=0 ->
    ``max(0.5, 1, 0) = 1``.
    """
    return max(v - c, th + mta - nica, 0.0)
