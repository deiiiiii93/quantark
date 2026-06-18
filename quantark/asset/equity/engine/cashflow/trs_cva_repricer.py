"""Markovian as-of repricer for a single-period TRS — the SA-CVA exposure seam.

The SA-CVA Monte-Carlo exposure engine reprices each trade as a *Markovian value
surface*: ``engine.price(product, as_of_env(base_env, spot, t))`` must return the
value of the **remaining** contract at a future time ``t`` and a simulated
``spot``, marginalizing over the path (it cannot pass a per-path history). The
realized-cashflow :class:`TotalReturnSwapEngine` cannot serve this directly: it
reads the *whole observed daily price path* (and raises on any missing pivot), so
it is path-reading, not Markovian.

This module supplies the missing seam for the **single-period** TRS, which *is*
Markovian. The realized engine's valuation-date present value decomposes as

    present_value = accrual_interest_cum + float_interest + cash_div_accrual

and only ``float_interest = float_dir · q · (S − S0)`` depends on the
valuation-date spot. The other two terms — NOTIONAL financing accrual and the
per-share cash-dividend ledger — are **spot-path independent**. So

    V(S, t) = baseline(t) + float_dir · q · (S − S0)

where ``baseline(t)`` is the engine's present value at as-of date ``t`` priced on
a *flat S0* path. The flat path is not an approximation of the real path: at spot
≡ S0 the float term is identically zero, so the engine returns exactly
``accrual_interest_cum + cash_div_accrual`` — the genuine spot-independent
baseline, computed with the engine's own exact accrual/day-count conventions. The
spot term is then added in closed form.

Path-dependent variants (market-value financing accrual, intermediate
redemptions, share dividends/splits, dual-currency) break the decomposition —
``q``, ``S0`` or the baseline itself become path-dependent — and are out of scope
here: :func:`build_trs_cva_components` raises for them (handled on the stateful
exposure path instead), never silently mis-pricing.
"""

from dataclasses import replace
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd

from quantark.asset.equity.product.swap.trs_params import (
    AccrualType,
    TRSParams,
)
from quantark.asset.equity.engine.cashflow.total_return_swap_engine import (
    TotalReturnSwapEngine,
)
from quantark.util.exceptions import MarketDataError, ValidationError

_DATE_FMT = "%Y-%m-%d"
# High extraction precision so the spot-independent baseline is not perturbed by
# the engine's display rounding (the float term is added at full precision).
_PV_PRECISION = 10


def _act365_years(start: str, end: str) -> float:
    """ACT/365 year fraction between two ``YYYY-MM-DD`` dates, floored at 0."""
    a = datetime.strptime(start, _DATE_FMT)
    b = datetime.strptime(end, _DATE_FMT)
    return max((b - a).days / 365.0, 0.0)


class TRSCVARepricer:
    """As-of Markovian repricer for one single-period TRS (unit / per-contract).

    ``price(product, env)`` returns the per-contract mark-to-market of the
    *remaining* swap at ``env.valuation_date`` given ``env.spot`` — the contract is
    delta-one on its float leg, so the spot enters linearly. The SA-CVA layer
    applies the signed trade ``quantity`` once, so this returns the unit value.
    """

    def __init__(self, base_params: TRSParams) -> None:
        if base_params.pricing is None:
            raise ValidationError("TRS pricing parameters must be provided")
        self._base = base_params
        self._engine = TotalReturnSwapEngine()

        self._s0 = float(base_params.asset.asset_initial_price)
        if not (self._s0 > 0.0):
            raise ValidationError("asset_initial_price must be positive")
        self._q = float(base_params.float_leg.initial_notional) / self._s0
        self._float_dir = int(base_params.float_leg.direction)
        self._contract_start = min(
            base_params.fix_leg.start_date, base_params.float_leg.start_date
        )
        self._contract_end = max(
            base_params.fix_leg.end_date, base_params.float_leg.end_date
        )
        # baseline(as_of) is invariant to spot/vol/IR bumps (financing uses the
        # struck fix_leg.rate, dividends are per-share), so cache it by as-of date.
        self._baseline_cache: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Baseline (spot-independent) via the exact realized engine
    # ------------------------------------------------------------------ #
    def _baseline_pv(self, as_of: str) -> float:
        """Engine present value at ``as_of`` on a flat-S0 path.

        At spot ≡ S0 the float term vanishes, so this is exactly the
        spot-independent ``accrual_interest_cum + cash_div_accrual`` evaluated with
        the engine's own accrual conventions.
        """
        cached = self._baseline_cache.get(as_of)
        if cached is not None:
            return cached

        # The flat baseline path spans [contract_start, contract_end]; an as-of
        # before inception is not a path pivot (the engine would raise on the
        # missing date) and a forward-starting swap's pre-start value is a forward
        # exposure this realized repricer cannot express — reject it explicitly.
        if as_of < self._contract_start:
            raise ValidationError(
                f"as-of date {as_of!r} precedes contract start "
                f"{self._contract_start!r}; forward-starting TRS valuation is not "
                "supported by the single-period as-of repricer"
            )

        idx = pd.date_range(self._contract_start, self._contract_end, freq="D")
        flat = pd.Series(self._s0, index=[d.strftime(_DATE_FMT) for d in idx])
        asset = replace(self._base.asset, asset_prices=flat)
        base_pricing = self._base.pricing  # not None (checked in __init__)
        pricing = replace(
            base_pricing, valuation_date=as_of, output_mode="spot"  # type: ignore[type-var]
        )
        clone = replace(self._base, asset=asset, pricing=pricing)

        df = self._engine.price(clone, precision=_PV_PRECISION)
        pv = df.iloc[-1]["present_value"]
        if isinstance(pv, str):
            raise MarketDataError(
                f"TRS baseline present_value is non-numeric at as-of {as_of!r}"
            )
        pv = float(pv)
        self._baseline_cache[as_of] = pv
        return pv

    def value_at(self, spot: float, as_of: str) -> float:
        """Per-contract mark-to-market at ``as_of`` (``YYYY-MM-DD``) and ``spot``."""
        return self._baseline_pv(as_of) + self._float_dir * self._q * (
            float(spot) - self._s0
        )

    # ------------------------------------------------------------------ #
    # SA-CVA engine contract: price(product, env) -> unit value
    # ------------------------------------------------------------------ #
    def price(self, product: "TRSCVAProduct", env) -> float:
        as_of = env.valuation_date.strftime(_DATE_FMT)
        return self.value_at(env.spot, as_of)


class TRSCVAProduct:
    """Product adapter exposing the SA-CVA ``get_maturity`` / ``get_payoff`` API.

    The SA-CVA exposure engine asks the product for its remaining maturity (to size
    the exposure grid) and, at the terminal node, for its contractual payoff. A TRS
    is delta-one, so the terminal payoff is its full as-of value at maturity.
    """

    def __init__(self, repricer: TRSCVARepricer) -> None:
        self._repricer = repricer
        self._contract_end = repricer._contract_end

    def get_maturity(self, env) -> float:
        """Remaining tenor (ACT/365 years) from ``env.valuation_date`` to maturity."""
        as_of = env.valuation_date.strftime(_DATE_FMT)
        return _act365_years(as_of, self._contract_end)

    def get_payoff(self, spot: float) -> float:
        """Terminal (maturity) value: the full as-of value priced at maturity."""
        return self._repricer.value_at(spot, self._contract_end)


def build_trs_cva_components(
    params: TRSParams,
) -> Tuple[TRSCVAProduct, TRSCVARepricer]:
    """Build the ``(product, engine)`` pair for a Markovian single-period TRS.

    Raises for path-dependent variants whose value is not a function of
    ``(spot, t)`` alone — they belong on the stateful exposure path, not the
    single-state analytic value surface:

    * **market-value financing accrual** — financing depends on the spot at every
      accrual pivot (the whole path), not just the current spot;
    * **intermediate redemptions** — the asset quantity / notional changes mid-life
      and realized P&L is booked at fixed dates, so ``q`` is not constant;
    * **share dividends (splits)** — ``q`` and the effective ``S0`` change mid-life.

    Cash dividends (per-share, deterministic), upfront/unwind fees and margin do
    not break the decomposition (they enter the spot-independent baseline or the
    margin ledger, not the spot term) and are accepted.
    """
    if params.fix_leg.accrual_type != AccrualType.NOTIONAL:
        raise ValidationError(
            "single-period TRS CVA repricing supports NOTIONAL financing accrual "
            f"only; {params.fix_leg.accrual_type.value!r} accrual is path-dependent "
            "(financing follows the market-value path) — use the stateful exposure "
            "path"
        )
    events = params.events.events or {}
    if events.get("redm"):
        raise ValidationError(
            "single-period TRS CVA repricing does not support intermediate "
            "redemptions (the asset quantity changes mid-life) — use the stateful "
            "exposure path"
        )
    if events.get("div_share"):
        raise ValidationError(
            "single-period TRS CVA repricing does not support share dividends / "
            "splits (the asset quantity and reset level change mid-life) — use the "
            "stateful exposure path"
        )
    repricer = TRSCVARepricer(params)
    return TRSCVAProduct(repricer), repricer
