# Equity Implied Futures Carry Risk — Design

**Date:** 2026-07-03
**Status:** design draft v2 (amended 2026-07-03 per review: prerequisites
merged, beta constrained to 1.0, extrapolation / MC common-random-numbers /
theoretical-carry rhoq conventions specified)
**Area:** `quantark/asset/equity/`, `quantark/priceenv/`, `quantark/portfolio/equity/`

## Summary

Add an explicit equity **implied-carry risk mode** so listed index futures
tenors (`IC00`, `IC01`, `IC02`, `IC03`, etc.) can be used as both:

1. market calibration instruments for the option's dividend/carry term
   structure; and
2. hedge buckets that convert directly into futures hands.

The core risk output is a futures-tenor delta bucket:

```text
delta_bucket_i = dPV / dF_i
hedge_hands_i = -delta_bucket_i / delta_per_hand_i
```

Bucketed `rhoq` remains a model/carry diagnostic. It is not eliminated on the
standalone option position. The futures hedge offsets `rhoq` only at the whole
portfolio level.

## Prerequisites (merged 2026-07-03)

The original review of this spec found two blockers. Both are resolved by
the engine term-structure upgrade
(`docs/superpowers/specs/2026-07-03-engine-term-structure-upgrade-design.md`,
Phases 0-3 all merged to main):

1. **Signed implied carry.** Contango futures marks imply negative `q(T)` —
   including this spec's own demo marks: spot `5000`, `IC00 = 5008` at
   `T = 0.03` with `r = 3%` gives `q ≈ −2.3%`. `TermStructureDividendYield`
   now accepts signed yields (`|y| ≤ 1.0`), the dividend wrapper clamps are
   removed, and `_build_div_bumped_env` no longer guards `q ≥ 0`. The demo
   marks are therefore valid as written. Bump-direction status quo, stated
   precisely: `calculate_numerical_delta_q` is central, but the scalar
   `calculate_numerical_dividend_rho` remains a **one-sided up-bump**
   (`direction=+1`, scaled to per-1%). This spec's rhoq bucket methods
   follow the same one-sided up-bump convention (see "Rhoq bucket bump")
   so scalar and bucketed rhoq are comparable; neither is central.
2. **Term-aware engines.** All equity MC, PDE, and QUAD engines consume
   forward rates, forward carry, and step vols sampled on their own time
   grids (identity on flat inputs; cross-family term agreement gate green).
   Bumping one futures tenor now produces genuine multi-tenor bucket deltas
   for path-dependent products — snowball/phoenix KO observations respond
   to intermediate forwards — instead of degenerating to
   `[0, 0, ..., everything-at-maturity]` as the pre-upgrade constant-drift
   engines would.

Convention note: `derive_implied_dividend_yield` in
`quantark/backtest/otc/market.py` uses **simple** compounding and clamps
`q = max(0, r − basis)`. This spec's continuous-compounding signed inversion
is a different convention. The backtest helper is intentionally unchanged
and must **not** be reused to build `IndexFuturesCurve` inputs.

## Motivation

Current equity option Greeks expose scalar spot delta, vega, rho, and
`dividend_rho` (`rhoq`). This is not sufficient for index-option hedging where
the desk trades live index futures across tenors.

For example, if the option position has futures-tenor delta buckets:

```text
IC00: 10
IC01: 20
IC02: 30
IC03: 40
```

and one futures hand hedges `10` bucket-delta units, the hedge instruction is:

```text
IC00: -1 hand
IC01: -2 hands
IC02: -3 hands
IC03: -4 hands
```

This example is deliberately small and illustrative. In production,
`delta_bucket_i` is a **per-index-point PV sensitivity** (`dPV/dF_i`) and
`delta_per_hand_i = futures_multiplier_i` (beta is fixed at `1.0` in v1 —
see "Conversion methods"). For real Chinese index futures, one hand commonly
hedges hundreds of currency units per index point (for example multiplier
200 or 300), not `10`. The example only says "if one hand hedges 10
bucket-delta units."

The sign convention is: positive option bucket delta means long index/futures
exposure, so the hedge is short futures.

## Existing gaps

### 1. Flat dividend yield cannot define futures-tenor buckets

A flat dividend input can only create a parallel carry move. It cannot answer
how much risk should be hedged by `IC00` versus `IC01` versus `IC02`.

The option pricing environment needs a term-structured carry input. In the
current QuantArk architecture, the natural representation is
`TermStructureDividendYield`.

### 2. Futures marks need an explicit implied-carry mode

Futures can be viewed in three different modes:

| Mode | Futures price source | Futures rhoq |
|------|----------------------|--------------|
| `market_price` | observed futures mark only | 0 by model convention |
| `theoretical_carry` | generated from `S`, `r`, `q(T)` | non-zero |
| `implied_futures_carry` | observed marks imply `q(T)` for option pricing | non-zero as a portfolio risk coordinate |

The third mode is required. Without an explicit mode, the same futures quote can
be interpreted inconsistently as either an exogenous market price with no model
rhoq or as the calibration instrument for the option's carry curve.

Mode propagation must be explicit. Do **not** add a field to
`PricingEnvironment`; existing environments should remain unchanged. The mode
lives on `IndexFuturesCurve` and is also accepted by the futures-bucket Greek
calculator methods so callers cannot accidentally use an implied-carry curve in
market-price mode.

Mapping to the current `DeltaOneEngine.use_market_price`:

| New mode | Existing futures pricing behavior | Carry/rhoq behavior |
|----------|-----------------------------------|---------------------|
| `market_price` | `DeltaOneEngine(use_market_price=True)` | futures mark is exogenous; model `rhoq = 0` |
| `theoretical_carry` | `DeltaOneEngine(use_market_price=False)` with user-supplied `div_yield` | futures generated from `S`, `r`, `q(T)`; model `rhoq` non-zero |
| `implied_futures_carry` | `DeltaOneEngine(use_market_price=False)` with `div_yield` built from `IndexFuturesCurve` | live futures marks imply `q(T)`; futures/rhoq buckets are portfolio risk coordinates |

### 3. Futures `rhoq` is not exposed correctly today

`DeltaOneEngine._price_futures()` theoretically depends on `q`, but
`DeltaOneEngine.calculate_greeks()` does not return `dividend_rho`, and
`GreeksCalculator.calculate_numerical_greeks()` short-circuits linear products
to `dividend_rho = 0.0`.

For theoretical futures:

```text
F = S * exp((r - q)T) + basis * exp(-lambda T)
dF/dq = -T * S * exp((r - q)T)
rhoq_per_1pct = dF/dq * 0.01
```

Per hand:

```text
futures_rhoq_per_hand = multiplier * rhoq_per_1pct
```

Sign convention: positive `rhoq` means "PV increases when `q` increases."
Theoretical long futures have negative `rhoq` because a higher dividend/carry
yield lowers the forward:

```text
dF/dq < 0
```

This negative futures `rhoq` is the natural offset to option positions whose
portfolio carry exposure has the opposite sign. Reports must keep the sign
unflipped; any desk display convention should be a presentation layer only.

## Financial model

The implied-carry curve is derived from live futures marks:

```text
F(T_i) = S * exp((r(T_i) - q(T_i)) * T_i)
q(T_i) = r(T_i) - ln(F(T_i) / S) / T_i
```

Rates are continuously compounded, matching `RateCurve.get_rate()` and the
existing equity Black-Scholes/Futures conventions.

The formula treats any observed futures basis as part of the implied carry
curve. In v1, `IndexFuturesCurve` does **not** subtract a separate basis before
implying `q(T)`. If a future trades rich/cheap, that richness is intentionally
folded into implied carry because it is also part of the tradable futures risk.
A later extension may add an optional explicit basis field, but it must then
state whether bucket deltas are to futures price, clean forward, or basis.

### Extrapolation beyond the quoted nodes

`to_dividend_yield_curve()` produces a `TermStructureDividendYield` with
nodes exactly at the futures maturities. Interpolation between nodes is
linear in nodal `q` (`linear_q`); extrapolation outside
`[T_first, T_last]` is **flat in nodal `q`**:

```text
q(T) = q(T_first)   for T < T_first
q(T) = q(T_last)    for T > T_last
```

The consequences are intended behavior and must be pinned by test:

- An option maturing at `T* > T_last` prices with its tail carry over
  `(T_last, T*]` driven entirely by the last contract's mark. Bumping the
  last contract therefore moves the option's entire tail forward, and
  `delta_bucket` for the last contract includes this **unspanned tail
  carry**. This is the standard roll-hedge convention: tail carry is hedged
  with the longest listed contract and rolled at expiry.
- For a European option with `T* > T_last`, ALL bucket delta lands on the
  last contract: its PV depends only on the terminal forward, `q(T*)`
  equals the flat-extrapolated `q(T_last)`, and bumping earlier nodes does
  not move `q(T*)`. This concentration is financially correct, not a bug.
- Path-dependent products (snowball/phoenix) with observation dates inside
  `[T_first, T_last]` retain genuine sensitivity to interior nodes even
  when their maturity exceeds `T_last`, because the term-aware engines
  consume forward carry on the observation grid.
- Reporting: when the option maturity exceeds `T_last`, the last contract's
  bucket row must carry an explicit `extrapolated_tail: true` flag so
  spanned and unspanned risk are never silently aggregated into one
  unlabeled number. Symmetrically, `T* < T_first` flags the first
  contract's row.

For each futures contract `i`, the primary hedge risk is:

```text
delta_bucket_i = dPV / dF_i
```

Internally this is implemented by bumping the futures mark and rebuilding the
implied dividend curve:

```text
F_i' = F_i + dF
q_i' = r_i - ln(F_i' / S) / T_i
```

For small bumps:

```text
dq_i ~= -dF / (T_i * F_i)
```

The user-facing output should stay in futures-price units because it converts
directly into hedge hands.

## Risk-coordinate semantics

### Scalar spot delta versus futures-tenor bucket delta

The existing scalar equity delta remains the local spot Greek:

```text
spot_delta = dPV / dS
```

It answers: "How much does PV move for a one-point move in the current index
spot, with the existing Greek convention for how other market inputs move?"

The new futures-tenor bucket delta is a different coordinate:

```text
futures_delta_bucket_i = dPV / dF_i
```

It answers: "How much does PV move for a one-point move in a specific live
index-futures contract, with other futures-tenor marks held fixed?"

These two Greeks are related but not interchangeable. In implied-carry mode,
the futures curve is a term structure. Bumping one tenor changes the implied
carry/dividend node for that tenor, while spot is held fixed. Bumping spot moves
the whole spot anchor and may move forwards differently depending on the chosen
carry convention.

For a complete index-option risk report, show both:

- **spot delta** for local index exposure and traditional delta reporting;
- **futures-tenor bucket delta** for trade sizing across `IC00`, `IC01`,
  `IC02`, `IC03`, etc.

Do not derive hedge hands from scalar spot delta when a futures-tenor bucket
curve is available. Use scalar spot delta as a cross-check: under a simple,
flat, one-contract world (a conceptual limit case — the API itself requires
at least two quotes), futures-bucket delta should reconcile directionally
with spot delta after applying `dF/dS`. Under real multi-tenor carry curves,
bucket deltas provide the hedge allocation.

### Futures-tenor delta bucket

Definition:

```text
delta_bucket_i = [PV(F_i + dF_i) - PV(base)] / dF_i
```

`F_i` is the live futures mark for one tradable tenor bucket. Only that tenor is
bumped.

This is the primary hedging Greek for index futures hedging.

### Bucketed rhoq

Definition:

```text
rhoq_bucket_i = [PV(q_i + dq_i) - PV(base)] * (0.01 / dq_i)
```

This is a carry-coordinate diagnostic. It explains the same risk through the
implied dividend/carry curve, but it is less directly tradeable than
`dPV/dF_i`.

### Bucketed vega

Bucketed vega remains a volatility-tenor risk, separate from implied-carry
futures buckets:

```text
vega_bucket_i = [PV(vol_i + dvol_i) - PV(base)] / dvol_i
```

`vol_i` is the volatility input for one maturity bucket. Only that volatility
bucket is bumped. The default report convention already used elsewhere in
QuantArk is usually "PV change for +1 vol point"; the implementation must make
the scale explicit:

```text
bucket_vega_per_1vol_point = PV(vol_i + 0.01) - PV(base)
```

This is not hedged by index futures. It is hedged by options or volatility
instruments. The futures-tenor delta hedge may reduce carry/rhoq exposure, but
it should not be presented as a vega hedge.

Bucketed vega, bucketed `rhoq`, and futures-tenor bucket delta therefore have
three different hedge meanings:

| Greek | Bumped input | Primary hedge |
|-------|--------------|---------------|
| Bucketed vega | volatility tenor | listed/OTC options or vol instruments |
| Bucketed `rhoq` | dividend/carry tenor | diagnostic; partially spanned by futures |
| Futures-tenor bucket delta | live futures tenor mark | index futures hands |

### Portfolio rhoq after hedging

The option position's standalone `rhoq` remains. The futures hedge adds
offsetting carry exposure. The relevant post-hedge check is portfolio-level:

```text
portfolio_rhoq_i = option_rhoq_i + sum_j hedge_hands_j * futures_rhoq_{j,i}
```

In futures-price coordinates:

```text
portfolio_delta_bucket_i =
    option_delta_bucket_i + hedge_hands_i * delta_per_hand_i
```

This should be close to zero for exact bucket hedges, subject to rounding,
interpolation, and basis risk.

## Proposed API

### Dividend input and futures carry mode enums

Add separate enums for the two different concepts:

```python
from enum import Enum


class EquityDividendInputMode(Enum):
    FLAT_DIVIDEND = "flat_dividend"
    TERM_DIVIDEND = "term_dividend"


class FuturesCarryRiskMode(Enum):
    MARKET_PRICE = "market_price"
    THEORETICAL_CARRY = "theoretical_carry"
    IMPLIED_FUTURES_CARRY = "implied_futures_carry"
```

`EquityDividendInputMode` describes existing option-pricing inputs.
`FuturesCarryRiskMode` describes futures/carry-risk behavior and is the only
mode type allowed on `IndexFuturesCurve`.

### Mode propagation

The clean propagation path is:

1. `IndexFuturesCurve.mode: FuturesCarryRiskMode` stores the market
   interpretation of the futures curve.
2. `GreeksCalculator.calculate_futures_delta_buckets(..., mode=None)` defaults
   to `futures_curve.mode`.
3. `GreeksCalculator.calculate_futures_rhoq_buckets(..., mode=None)` defaults
   to `futures_curve.mode`.
4. `PricingEnvironment` remains unchanged. In implied-carry mode, callers clone
   the environment and set `div_yield` to
   `futures_curve.to_dividend_yield_curve(rate_curve)`.

The calculator must reject incompatible calls:

- `calculate_futures_delta_buckets()` resolves
  `resolved_mode = mode if mode is not None else futures_curve.mode`; the
  resolved mode must be `FuturesCarryRiskMode.IMPLIED_FUTURES_CARRY`.
- `calculate_futures_rhoq_buckets()` supports
  `FuturesCarryRiskMode.THEORETICAL_CARRY` and
  `FuturesCarryRiskMode.IMPLIED_FUTURES_CARRY`; it raises `ValidationError` in
  `MARKET_PRICE` mode because a zero bucket table can look like a real hedge
  result.

In `THEORETICAL_CARRY` mode, `pricing_env.div_yield` is the source of carry for
pricing and rhoq. `IndexFuturesCurve` is used only for contract metadata
(`contract`, `maturity`, `multiplier`, `beta`) and reporting alignment; the
calculator must **not** rebuild `q(T)` from futures marks in this mode.

In `IMPLIED_FUTURES_CARRY` mode, `IndexFuturesCurve` is the source of the carry
curve and the calculator rebuilds `q(T)` from futures marks.

### Futures curve market object

Add a small market object for index futures marks:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from quantark.param.rrf.rate_curve import RateCurve
from quantark.util.exceptions import ValidationError


@dataclass(frozen=True)
class IndexFuturesQuote:
    contract: str
    maturity: float
    price: float
    multiplier: float
    beta: float = 1.0
    expiry_date: datetime | None = None

    def __post_init__(self) -> None:
        if not self.contract:
            raise ValidationError("contract must be non-empty")
        if self.maturity <= 0.0:
            raise ValidationError("maturity must be positive")
        if self.price <= 0.0:
            raise ValidationError("price must be positive")
        if self.multiplier <= 0.0:
            raise ValidationError("multiplier must be positive")
        if self.beta != 1.0:
            raise ValidationError(
                "beta must be 1.0 (cross-hedge beta is not supported in v1)"
            )


@dataclass(frozen=True)
class IndexFuturesCurve:
    underlying: str
    spot: float
    quotes: Sequence[IndexFuturesQuote]
    mode: FuturesCarryRiskMode = FuturesCarryRiskMode.IMPLIED_FUTURES_CARRY
    interpolation: str = "linear_q"
```

Responsibilities:

- validate non-empty underlying, positive spot, unique contracts, sorted
  maturities, and supported interpolation/mode values;
- validate `len(quotes) >= 2`: the conversion target
  `TermStructureDividendYield` requires at least two time nodes, so a
  one-contract curve must fail fast at construction with a clear
  `ValidationError`, not deep inside `to_dividend_yield_curve()`.
  Single-tenor markets are out of scope for v1 (no duplicated-node or
  flat-curve fallback — if support is needed later it must be an explicit
  one-quote conversion mode, not an invented approximation);
- convert futures marks into `TermStructureDividendYield`;
- bump one contract mark and rebuild the implied carry curve;
- compute `delta_per_hand(contract)`.

Field-level quote validation belongs in `IndexFuturesQuote.__post_init__`.
Cross-quote invariants belong in `IndexFuturesCurve.__post_init__`, which must
also snapshot `quotes` into a tuple (`object.__setattr__`) **before**
validating — a frozen market object must not be mutable through the caller's
original list after validation.

### Conversion methods

```python
def to_dividend_yield_curve(self, rate_curve: RateCurve) -> TermStructureDividendYield:
    ...

def bump_contract(self, contract: str, price_bump: float) -> "IndexFuturesCurve":
    ...

def delta_per_hand(self, contract: str) -> float:
    ...
```

Definition:

```text
delta_per_hand = multiplier
```

**Beta is constrained to `1.0` in v1.** The earlier draft used
`delta_per_hand = multiplier * beta`, which double-counts the imperfect-hedge
adjustment:

- `delta_bucket_i = dPV/dF_i` is computed by bumping the futures mark itself
  and repricing, so it is already expressed in the hedge instrument's own
  price coordinates — any relationship between the futures contract and the
  option's underlying is already inside `delta_bucket_i`.
- One hand's realized PnL per index point of `F_i` is `multiplier_i` by
  contract definition, never `multiplier_i * beta_i`. With beta in the
  denominator, the reported `net_delta_bucket` zeroes out as bookkeeping
  while the actual futures position mis-sizes the hedge by `1/beta`.
- A genuine cross-underlying proxy hedge cannot be computed by this
  machinery anyway: the implied-carry inversion divides `F_i` by the option
  underlying's spot, so `IndexFuturesCurve.spot` and the option's underlying
  are the same asset by construction.

`IndexFuturesQuote.beta` is kept as a field for forward compatibility, but
validation rejects any value other than `1.0` (see `__post_init__`). A
future cross-hedge extension must apply beta as an explicitly labeled desk
reporting adjustment **outside** the `delta_bucket`/`hedge_hands` identity —
never inside `delta_per_hand`.

The `delta_bucket` returned by the calculator is in PV currency per one futures
index point. Therefore `hedge_hands = -delta_bucket / multiplier`.

### Greeks calculator extension

Add:

```python
def calculate_futures_delta_buckets(
    self,
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    futures_curve: IndexFuturesCurve,
    *,
    mode: FuturesCarryRiskMode | None = None,
    price_bump: float = 1.0,
) -> list[dict[str, float | str]]:
    ...
```

No `base_price` parameter: the method rebuilds the dividend curve from the
futures marks internally, so the base PV must be computed under that implied
environment. Accepting a caller-supplied PV (typically priced under
`pricing_env.div_yield`) would silently shift every bucket by
`(implied_base − supplied_base) / price_bump` and mis-size the hedge with no
exception. The base PV is always recomputed internally.

Each result row:

```python
{
    "contract": "IC00",
    "maturity": 0.03,
    "future_price": 5200.0,
    "price_bump": 1.0,
    "delta_bucket": 10.0,
    "delta_per_hand": 300.0,
    "hedge_hands": -0.03333333333333333,
    "extrapolated_tail": False,
}
```

Add companion method:

```python
def calculate_futures_rhoq_buckets(
    self,
    product: BaseEquityProduct,
    pricing_env: PricingEnvironment,
    engine: BaseEngine,
    futures_curve: IndexFuturesCurve,
    *,
    mode: FuturesCarryRiskMode | None = None,
    div_bump: float | None = None,
) -> list[dict[str, float | str]]:
    ...
```

(Same rule: no `base_price` parameter — the base PV is computed internally
under the mode's base environment.)

This produces carry-coordinate diagnostics by bumping one implied `q_i` node at
a time.

### Full API flow example

```python
curve = IndexFuturesCurve(
    underlying="IC",
    spot=5000.0,
    quotes=[
        IndexFuturesQuote("IC00", maturity=0.03, price=5008.0, multiplier=200.0),
        IndexFuturesQuote("IC01", maturity=0.10, price=5020.0, multiplier=200.0),
        IndexFuturesQuote("IC02", maturity=0.18, price=5036.0, multiplier=200.0),
        IndexFuturesQuote("IC03", maturity=0.32, price=5064.0, multiplier=200.0),
    ],
    mode=FuturesCarryRiskMode.IMPLIED_FUTURES_CARRY,
)

env = deepcopy(base_env)
env.div_yield = curve.to_dividend_yield_curve(env.rate_curve)

rows = GreeksCalculator().calculate_futures_delta_buckets(
    option,
    env,
    option_engine,
    curve,
    price_bump=1.0,
)
```

The returned `hedge_hands` are fractional. Rounding is deliberately left to the
execution/reporting layer.

## Bump convention

### Futures delta bucket bump

For each futures contract:

1. compute `base_pv`;
2. create `bumped_curve = futures_curve.bump_contract(contract, +price_bump)`;
3. convert `bumped_curve` to `TermStructureDividendYield`;
4. clone the pricing environment and set `div_yield` to the bumped curve;
5. reprice;
6. compute:

```text
delta_bucket = (bumped_pv - base_pv) / price_bump
hedge_hands = -delta_bucket / delta_per_hand
```

Use one-sided bump by default because the hedge instrument is quoted in market
price units and negative futures prices are invalid. Central bump can be added
later behind an option if needed.

### Finite differences on MC engines (common random numbers)

A one-sided finite difference is only usable on an MC engine when the base
and bumped repricings use **common random numbers**: identical `MCParams`
seed and path count for every leg. Note the bump sizes involved — a 1-point
bump on a `T = 0.03` contract implies a carry shift on the order of 67bp,
which is fine analytically but drowns in MC noise without seed reuse. The
calculator must reuse the caller's engine and params unchanged for base and
bumped runs (no reseeding between legs), and any test asserting
finite-difference agreement on an MC engine must pin the seed explicitly.

### Rhoq bucket bump

The node source is mode-specific:

- `IMPLIED_FUTURES_CARRY`: the nodes are the implied `q_i` at the futures
  maturities; bump one node at a time as below.
- `THEORETICAL_CARRY`: `pricing_env.div_yield` may be flat and has no nodes
  to bump. Reuse the existing `BucketedDividendYield` (an interval bump over
  any base curve, already used by the snowball risk report) with bucket
  edges at the futures maturities `[0, T_0], (T_0, T_1], ...`. Do **not**
  invent a parallel node-bump mechanism for this mode.

Bump direction: one-sided **up**-bump (`+div_bump`), matching the existing
scalar `calculate_numerical_dividend_rho` convention so scalar and bucketed
rhoq are directly comparable. Signed carry makes the up-bump always valid
(no `q ≥ 0` constraint to respect).

For each implied carry node (implied-carry mode):

1. build the base implied `q(T)` term structure from futures;
2. bump one node `q_i` by `div_bump`;
3. reprice;
4. compute:

```text
rhoq_bucket_i = (bumped_pv - base_pv) * (0.01 / div_bump)
```

This output is not the primary hedge unit. It is a decomposition check.
`div_bump` defaults to the existing `GreeksCalculator` bump config
`EngineParams.get_effective_bump_config().div_bump` (currently a 1bp absolute
yield bump in the standard configuration) and is always scaled to **per 1%**
yield change by multiplying by `0.01 / div_bump`.

`TermStructureDividendYield` currently supports interpolation and parallel
shifts through existing callers, but it does not expose a single-node bump API.
Single-node bumping is new functionality for this feature. Prefer implementing
the node bump in `IndexFuturesCurve` / a small helper rather than broadening
`TermStructureDividendYield` unless the helper becomes generally useful.

### Interpolation spillover

With the default `linear_q` interpolation, bumping `IC01` leaves the **nodal**
`q` values for `IC00`, `IC02`, and `IC03` unchanged, but interpolated `q(T)`
between neighboring nodes changes. Therefore an option maturing between tenors
can have PV sensitivity to the bumped node even when its maturity is not exactly
equal to that node.

Tests must distinguish:

- **node-locality tests**: assert only that untouched nodal `q_i` values remain
  unchanged after a single-contract futures bump;
- **pricing tests**: accept interpolation spillover as the intended behavior
  under `linear_q`.

Do not claim the PV bump is purely local unless a future `piecewise_flat_q` or
nearest-node interpolation mode is explicitly selected and tested.

## Portfolio aggregation

Add portfolio-level helpers after single-position Greeks exist:

```python
from collections.abc import Mapping, Sequence


FuturesBucketRow = dict[str, float | str]


def aggregate_futures_delta_buckets(
    rows_by_position: Mapping[str, Sequence[FuturesBucketRow]],
) -> list[FuturesBucketRow]:
    ...

def aggregate_futures_rhoq_buckets(
    rows_by_position: Mapping[str, Sequence[FuturesBucketRow]],
) -> list[FuturesBucketRow]:
    ...
```

Aggregation key is `contract`. Position quantity must be applied before rows are
passed into the aggregator. In other words, single-position bucket rows are
already position-level PV sensitivities, not unit-product sensitivities.

For each contract, sum:

```text
delta_bucket
hedge_hands
rhoq_bucket
futures_rhoq_bucket
net_delta_bucket
net_rhoq_bucket
```

Define the futures-hedge rhoq contribution for bucket `i` as:

```text
futures_rhoq_bucket_i =
    sum_j hedge_hands_j * futures_rhoq_per_hand_{j,i}
```

For v1, use the diagonal approximation unless a full cross-tenor Jacobian is
explicitly implemented:

```text
futures_rhoq_per_hand_{j,i} = 0, j != i
futures_rhoq_per_hand_{i,i} = multiplier_i * dF_i/dq_i * 0.01
```

Under `linear_q`, this diagonal approximation is a hedge-reporting convention,
not a full mathematical Jacobian of interpolated carry. If an implementation
later computes cross-tenor effects, the report must label them as full
Jacobian-based bucket rhoq.

Keep non-additive fields (`maturity`, `future_price`, `delta_per_hand`) from
the first row for that contract and assert subsequent rows match within a tight
tolerance. If they do not match, raise a validation error rather than silently
aggregating incompatible futures curves.

Post-hedge diagnostics should report:

```python
{
    "contract": "IC01",
    "option_delta_bucket": 600.0,
    "hedge_hands": -2.0,
    "futures_delta_per_hand": 300.0,
    "net_delta_bucket": 0.0,
    "option_rhoq_bucket": ...,
    "futures_rhoq_bucket": ...,
    "net_rhoq_bucket": ...,
}
```

The standalone option `rhoq` should remain visible. The net portfolio `rhoq`
should be evaluated after adding hedge futures positions.

## Product and engine scope

Initial scope is **Equity only**:

- European vanilla equity/index options;
- existing equity structured products that already price from
  `PricingEnvironment.div_yield`;
- equity `Futures` delta-one product;
- portfolio-level aggregation for equity positions.

Out of scope:

- FX products;
- bond futures;
- stochastic dividend models;
- live market data ingestion;
- optimizer that decides non-diagonal hedge allocation across contracts;
- transaction execution.

## Implementation touch points

Expected files:

- `quantark/util/enum/greeks_enums.py` or a new enum module — add
  `EquityDividendInputMode` and `FuturesCarryRiskMode`.
- `quantark/param/div/dividend_yield.py` — keep `TermStructureDividendYield`;
  add node-bump helper only if it does not make the class too broad.
- New `quantark/asset/equity/market/index_futures_curve.py` — futures quote and
  curve objects.
- `quantark/asset/equity/riskmeasures/greeks_calculator.py` — add futures delta
  and rhoq bucket methods.
- `quantark/asset/equity/engine/analytical/deltaone_engine.py` — add futures
  `dividend_rho` in theoretical carry mode.
- `quantark/portfolio/equity/position.py` or a separate equity risk utility —
  aggregate position-level futures buckets.
- `example/equity_futures_delta_buckets_demo.py` — end-to-end demo that builds
  the implied futures carry curve, computes hedge hands, and shows post-hedge
  net bucket diagnostics.
- Tests under `test/`, likely:
  - `test/test_index_futures_implied_carry.py`
  - `test/test_equity_futures_delta_buckets.py`
  - targeted additions to `test/test_greeks_bump_config.py` or a delta-one test.

## Required demo

Add `example/equity_futures_delta_buckets_demo.py` as a runnable smoke/demo
script. It should be deterministic and use synthetic market data, not live data.

The demo must show this complete flow:

1. Build an `IndexFuturesCurve` for `IC00`, `IC01`, `IC02`, `IC03`, including
   maturity, futures price, and multiplier (beta is fixed at `1.0` in v1).
2. Convert the futures curve to `TermStructureDividendYield` with
   `curve.to_dividend_yield_curve(env.rate_curve)`.
3. Attach the implied dividend term structure to a `PricingEnvironment`.
4. Price a `000905` index Snowball or Snowball-like equity option using an
   existing equity engine that consumes `PricingEnvironment.div_yield`.
5. Run `GreeksCalculator.calculate_futures_delta_buckets()` and print:
   - contract;
   - futures price;
   - bucket delta `dPV/dF_i`;
   - delta per hand;
   - fractional hedge hands.
6. Run `GreeksCalculator.calculate_futures_rhoq_buckets()` and print bucketed
   `rhoq` as diagnostic output.
7. Build a tiny option + futures hedge portfolio from the computed hedge hands.
8. Print post-hedge diagnostics:
   - net futures bucket delta by contract;
   - standalone option `rhoq`;
   - futures hedge `rhoq`;
   - net portfolio `rhoq`.

The demo should make the core convention visible in output:

```text
hedge_hands_i = -delta_bucket_i / multiplier_i
```

The demo should not round hedge hands; if it shows rounded hands for display, it
must also print the fractional hands and post-rounding residual separately.

## Required tests

### 1. Implied carry curve from futures marks

Given `S=100`, flat `r=0.03`, futures marks generated from known `q(T)`, the
curve builder recovers the original dividend yields at each tenor:

```text
q_i = r_i - ln(F_i / S) / T_i
```

### 2. Bumping one futures contract changes only one carry node

Bump `IC01` by `+1.0`. Assert:

- `IC01` implied `q` changes according to the formula;
- `IC00`, `IC02`, `IC03` implied `q` **nodes** are unchanged;
- under `linear_q`, interpolated `q(T)` between neighboring nodes may change;
- validation rejects unknown contracts and non-positive bumped prices;
- constructing an `IndexFuturesCurve` with fewer than two quotes raises
  `ValidationError` at construction (single-tenor markets are out of scope
  in v1).

### 3. Vanilla option futures-delta bucket matches finite difference

For a European call priced under Black-Scholes with implied carry from futures:

- compute `delta_bucket(IC01)` through the new API;
- manually build the bumped `TermStructureDividendYield`;
- reprice;
- assert exact agreement with `(pv_bumped - pv_base) / price_bump`.

### 4. Hedge hands conversion

Given synthetic bucket deltas `[10, 20, 30, 40]` and
`delta_per_hand = 10`, assert hedge hands are `[-1, -2, -3, -4]`.
Also test a realistic multiplier case, e.g. `delta_bucket=600` and
`delta_per_hand=300`, giving `hedge_hands=-2`.

### 5. Futures theoretical rhoq

For theoretical futures:

```text
rhoq = -S * T * exp((r - q)T) * 0.01
```

The engine or calculator should return this value, scaled by multiplier when
computed at position/hand level.

### 6. Market-price mode keeps model rhoq at zero

When futures are valued as observed marks only, `dividend_rho` remains zero.
This prevents accidental mixing of market-price and implied-carry conventions.

### 7. Portfolio net rhoq is offset by futures hedge

Create an option position and hedge futures positions from the computed bucket
hands. Assert:

- option standalone `rhoq` is non-zero and unchanged;
- futures position `rhoq` is non-zero in implied/theoretical carry mode;
- portfolio net bucket delta is near zero;
- portfolio net `rhoq` is reduced relative to standalone option `rhoq`.

### 8. Extrapolated tail concentrates in the last contract

Price a European option with maturity beyond the last futures node
(`T* > T_last`). Bump each contract in turn and assert:

- `delta_bucket` is zero (to bump tolerance) for every contract except the
  last;
- the last contract's `delta_bucket` matches the full finite-difference
  forward delta;
- the last contract's row carries `extrapolated_tail: true` and all other
  rows carry `false`.

Also price a path-dependent product (snowball or phoenix) with KO
observations inside `[T_first, T_last]` and maturity beyond `T_last`, and
assert interior contracts have non-zero `delta_bucket` (the term-aware
engines see the intermediate forwards).

### 9. Beta is rejected when not 1.0

Constructing an `IndexFuturesQuote` with `beta=1.5` (or any value other
than `1.0`) raises `ValidationError`. Hedge hands equal
`-delta_bucket / multiplier` exactly.

## Reporting requirements

Risk reports should separate:

1. **Standalone option Greeks**
   - spot delta;
   - scalar and bucketed vega;
   - scalar and bucketed `rhoq`;
   - futures-tenor delta buckets.

2. **Hedge instruction**
   - contract;
   - bucket delta;
   - delta per hand;
   - hedge hands;
   - rounded hands;
   - post-rounding residual delta bucket.

3. **Post-hedge portfolio diagnostics**
   - net futures bucket delta;
   - option `rhoq`;
   - futures hedge `rhoq`;
   - net portfolio `rhoq`;
   - residuals from rounding, interpolation, and unspanned tenors.

## Open design decisions

1. **Interpolation convention.** Default should be linear interpolation in
   implied `q(T)`. Log-forward interpolation is financially defensible but needs
   explicit tests if chosen.
2. **Bump size.** Default futures bucket bump should be `1.0` index point. Very
   low-priced or non-index contracts may need a relative bump later.
3. **Central versus one-sided bump.** Start one-sided. Add central only if
   numerical noise requires it.
4. **Hedge rounding.** The Greek API returns fractional hands only. Execution
   and reporting can add integer rounding policies later.
5. **Cross-hedging beta.** Removed from the v1 hedge math — `beta` is
   validated to `1.0` (see "Conversion methods" for why any other value
   double-counts). A future extension must define it as a labeled reporting
   adjustment outside `delta_per_hand`.

## Non-goals

- Do not hide the option's own `rhoq` after hedging.
- Do not make futures-tenor delta a replacement for scalar spot delta.
- Do not infer futures quotes from spot when live marks are available.
- Do not silently use implied-carry behavior under `market_price` mode.
- Do not add a general optimizer in the first implementation.
- Do not round hedge hands inside the Greek API. Fractional hands are the API
  output; integer rounding belongs to execution/reporting.
- Do not apply beta inside `delta_per_hand` or `hedge_hands`. One hand's PnL
  per futures index point is `multiplier` by contract definition; the
  imperfect-hedge relationship is already inside `dPV/dF_i`.

## Acceptance criteria

- The option pricer can consume a dividend term structure implied from live
  futures marks.
- The risk API can bump `IC00`, `IC01`, `IC02`, `IC03` independently and return
  `dPV/dF_i`.
- The risk API converts each bucket delta into futures hedge hands using that
  contract's `delta_per_hand`.
- The risk API returns fractional hedge hands and does not apply execution
  rounding.
- Bucketed vega remains available as a separate volatility-tenor risk and is
  not conflated with futures-tenor delta buckets.
- Futures theoretical/implied-carry `rhoq` is available and portfolio-level net
  `rhoq` can be computed.
- Existing scalar Greeks continue to work for flat-dividend and term-dividend
  modes.
- Market-price futures mode remains explicit and does not accidentally report
  model carry Greeks.
- Flat extrapolation beyond the last futures node is pinned by test, and
  bucket rows are flagged `extrapolated_tail` when the option maturity lies
  outside the quoted node range.
- Negative implied carry (contango marks) constructs and prices without
  error.
- MC-engine bucket deltas are computed with common random numbers (same
  seed and path count for base and bumped legs).
