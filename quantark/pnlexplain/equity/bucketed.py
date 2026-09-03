"""Opt-in tenor-vega / key-rate rows (spec §7.6).

Tenor-vega rows are COMPONENT rows that replace the scalar ``vega`` row: the
node-aligned bumps of the bucketed-greeks machinery sum to the scalar vega, so
``Σ_τ dV/dσ_τ × Δσ_τ`` is the scalar row split by pillar.

Key-rate rows are INFORMATIONAL. The only key-rate machinery in
``riskmeasures`` is carry-invariant (the forward is held fixed, the dividend
yield re-derived), which is a *different* sensitivity from the factor model's
rate step (rate curve replaced, dividend yield held): for a vanilla call the
two even differ in sign. Booking those buckets as the rate component would
mis-attribute a rate move, so the scalar ``rho`` stays the component and the
key-rate rows sit beneath it as a term-structure view, labelled with their
convention.
"""
from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Mapping, Tuple

from quantark.asset.equity.riskmeasures.bucketed_greeks import BucketedGreekCoordinate, BucketedGreeksRequest
from quantark.param import TermStructureVolSurface
from quantark.param.rrf.rate_curve import InterpolatedRateCurve
from quantark.pnlexplain.base import ExplainMethod, ExplainRow, Factor, RowKind
from quantark.pnlexplain.equity.scenario import ScenarioCache
from quantark.util.exceptions import ValidationError

RATE_CONVENTION = "carry_invariant"     # the calculator's key-rate rule: F held fixed, q re-derived


def _check_pair(name: str, a: Any, b: Any, cls) -> bool:
    """True when both snapshots carry a `cls` term structure.

    Rule (spec §7.6): the canonical pillars are the t0 object's; the t1 object
    is sampled at those pillars with its own interpolation, so a differing t1
    grid is allowed. A term structure on one side only is rejected.
    """
    is_a, is_b = isinstance(a, cls), isinstance(b, cls)
    if is_a != is_b:
        raise ValidationError(f"bucketed mode: {name} is a term structure on one side only")
    return is_a


def bucketed_rows(cache: ScenarioCache, calc: Any, bump: Any, level: str
                  ) -> Tuple[Mapping[Factor, Tuple[ExplainRow, ...]], FrozenSet[Factor]]:
    """Rows partitioned by factor + the factors whose scalar row they REPLACE (⊆ {VOL}).

    ``rows[Factor.VOL]`` are the ``vega.<τ>`` COMPONENT rows (t0 pillars).
    ``rows[Factor.RATE]`` are the informational ``rate_keyrate.<τ>`` rows and
    the ``rate_keyrate.parallel`` row; the scalar ``rho`` is kept.
    """
    snap0, snap1, moves = cache.snap0, cache.snap1, cache.moves
    e0, e1 = snap0.pricing_env, snap1.pricing_env
    q, K = snap0.quantity, moves.coordinate.reference_strike
    T = moves.coordinate.tenor_t1
    coords = []
    if _check_pair("vol_surface", e0.vol_surface, e1.vol_surface, TermStructureVolSurface) \
            and Factor.VOL in moves.coordinate.applicable:
        coords.append(BucketedGreekCoordinate.VOL_TENOR_VEGA)
    if _check_pair("rate_curve", e0.rate_curve, e1.rate_curve, InterpolatedRateCurve) \
            and Factor.RATE in moves.coordinate.applicable:
        coords.append(BucketedGreekCoordinate.RATE_KEYRATE)
    if not coords:
        return {}, frozenset()
    request = BucketedGreeksRequest(coordinates=tuple(coords), vol_bump=float(bump.vol_bump),
                                    rate_bump=float(bump.rate_bump))
    result = calc.calculate_bucketed_greeks(snap0.product, e0, cache.bump_engine_t0, request)
    by_factor: Dict[Factor, List[ExplainRow]] = {Factor.VOL: [], Factor.RATE: []}
    for pt in result.points:
        mode = getattr(pt, "difference_mode", None)
        mode = mode.value if hasattr(mode, "value") else mode
        if pt.coordinate is BucketedGreekCoordinate.VOL_TENOR_VEGA:
            tau = float(pt.maturity)
            d = float(e1.get_vol(K, tau)) - float(e0.get_vol(K, tau))
            g = q * float(pt.derivative)
            by_factor[Factor.VOL].append(ExplainRow(
                factor=Factor.VOL, term=f"vega.{tau:g}", method=ExplainMethod.TAYLOR, kind=RowKind.COMPONENT,
                level=level, pnl=g * d, greek=g, cash_greek=g * 0.01,
                moves={"vol_pts": d * 100.0, "tenor": tau},
                metadata={"pillar": tau, "bump_size": float(pt.bump_size), "difference_mode": mode}))
        elif pt.coordinate is BucketedGreekCoordinate.RATE_KEYRATE:
            if pt.name == "rate_keyrate.parallel":
                # the parallel move IS the factor model's scalar rate move, read at the coordinate
                if moves.d_rate is None or T is None:
                    raise ValidationError(
                        "bucketed mode: the parallel key-rate row needs the scalar rate move at the "
                        "product's tenor, which this coordinate does not define"
                    )
                d = float(moves.d_rate)
                g = q * float(pt.derivative)
                by_factor[Factor.RATE].append(ExplainRow(
                    factor=Factor.RATE, term="rate_keyrate.parallel", method=ExplainMethod.TAYLOR,
                    kind=RowKind.INFORMATIONAL, level=level, pnl=g * d, greek=g, cash_greek=g * 0.01,
                    moves={"rate_pct": d * 100.0},
                    metadata={**dict(pt.metadata), "bump_size": float(pt.bump_size), "difference_mode": mode,
                              "convention": RATE_CONVENTION}))
                continue
            tau = float(pt.maturity)
            d = float(e1.get_rate(tau)) - float(e0.get_rate(tau))
            g = q * float(pt.derivative)
            by_factor[Factor.RATE].append(ExplainRow(
                factor=Factor.RATE, term=f"rate_keyrate.{tau:g}", method=ExplainMethod.TAYLOR,
                kind=RowKind.INFORMATIONAL, level=level, pnl=g * d, greek=g, cash_greek=g * 0.01,
                moves={"rate_pct": d * 100.0, "tenor": tau},
                metadata={"pillar": tau, "bump_size": float(pt.bump_size), "difference_mode": mode,
                          "convention": RATE_CONVENTION}))
    rows = {f: tuple(r) for f, r in by_factor.items() if r}
    covered = frozenset({Factor.VOL}) if Factor.VOL in rows else frozenset()
    return rows, covered
