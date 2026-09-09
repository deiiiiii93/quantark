# Bucket-weighted multi-tenor futures hedge — design

Date: 2026-09-09
Status: draft for review
Builds on: `quantark.backtest.replay` (historical product replay),
`quantark.backtest.futures_ledger` (the single-contract hedge position),
`quantark.asset.equity.riskmeasures.bucketed_coordinates` (futures-tenor
buckets), `example/snowball_q_term_structure` (the q term-structure study,
stages 01–04).

## 1. Purpose

Every hedge the replay engine can run today holds one futures contract at a
time. The q term-structure study showed that which contract that is matters
less than which contract the carry is read from: moving the carry read to
the longest listed contract captured 294 of the 299 bp of daily-error
reduction that moving the whole hedge to that contract captured. The
remaining question is whether a hedge that holds several tenors at once,
sized so that the book's exposure to every listed forward is zero, hedges
better than any single contract, and at what cost.

This feature adds that hedge to the library and runs it in the study.

## 2. The mathematics (settled, see stage 04 and the 2026-09-09 probes)

The pricer sees spot `S`, listed marks `F_1..F_n` at tenors `T_1..T_n`, a
flat rate `r`, and turns them into a carry curve with nodes
`q_i = r - ln(F_i/S)/T_i`, interpolated between nodes and continued past
`T_n` by a tail convention. Three first derivatives, all in hands
(multiplier `m` = 200 CNY per index point), long-holder sign:

- `Δ_q = ∂V/∂S` at a frozen carry curve — the delta every existing cell
  hedges (`calculate_greeks`, relative central bump `spot_bump`).
- `Δ_F = ∂V/∂S` with every listed mark pinned and the curve rebuilt — only
  the unquoted tail moves.
- `B_i = ∂V/∂F_i` with spot and the other marks pinned — the bucket.

With `r` fixed, `F_i` and `q_i` are one variable seen two ways, so the bucket
is the carry bucket `ρ_i = ∂V/∂q_i` rescaled: `B_i = -ρ_i / (F_i T_i)`.
Because a frozen curve means `dF_i/dS = F_i/S`, the three are tied exactly:

    Δ_q = Δ_F + Σ_i (F_i/S) B_i                                  (identity)

Stage 04 verifies the identity to 0.001 hands under both tail conventions.

A futures position `h_i` earns `h_i m dF_i`, and `dF_i = (F_i/S) dS - F_i T_i
dq_i`, so every contract carries spot exposure and basis exposure of its own
tenor in a ratio fixed by the tenor. The hedged book's first-order change is

    dBook = Δ_F dS + Σ_i (B_i + m h_i) dF_i

and the residual of each hedge follows by substitution:

| hedge | hands | first-order residual |
|---|---|---|
| single contract `j` (today's cells) | `h_j = -Δ_q/m` | `Δ_F (dS - dF_j) + Σ_i B_i (dF_i - (F_i/S) dF_j)`: the basis moves, plus the `1 - F_j/S` under-hedge |
| bucket-exact | `h_i = -B_i/m` | `Δ_F dS`, whatever the chain does |
| bucket with the tail folded into the far contract | as above, `h_n -= Δ_F S/(m F_n)` | `Δ_F (dS - (S/F_n) dF_n)`: the far basis move, scaled by `Δ_F` |

In `(S, q)` coordinates the hedged book's basis exposure per node is

    ρ_i^book = ρ_i - h_i m F_i T_i          (bp per unit of q_i; ρ_i = -B_i F_i T_i)

which the bucket hedge zeroes node by node by construction, because
`h_i = -B_i/m` gives `h_i m F_i T_i = -B_i F_i T_i = ρ_i`. A single-contract
hedge leaves every node's `ρ_i` untouched except its own tenor, where it
adds `-h_j m F_j T_j`. On the worked date, front-hedged: +3.4 / 0 / +29.4 /
−92.6 bp per 1% on the four nodes; far-hedged: 0 / 0 / +29.4 / −56.3;
bucket-hedged: 0 / 0 / 0 / 0. This is the **hedged carry exposure**, and it
is recorded daily for every cell (section 4.5) so the basis mitigation is
measured directly rather than inferred from P&L.

The bucket hedge zeroes every basis column and leaves `Δ_F` of spot. `Δ_F`
is a property of the tail convention, not of the product:

- flat-forward-carry tail: `F(T) = F_n (F_n/F_{n-1})^{(T-T_n)/(T_n-T_{n-1})}`,
  spot-free, so `Δ_F ≈ 0` and bucket-exact is a complete first-order hedge.
  It gets there by holding a calendar spread geared off the last segment
  (worked date: +40.7 IM2506 / −71.5 IM2509 on a 28-hand delta, 112 gross).
- flat-q tail: `F(T) = S^{1-T/T_n} F_n^{T/T_n}`, so `Δ_F` is large and
  negative (−18 hands on the worked date). Bucket-exact leaves that open and
  the probe shows it hedging worse than the front month. Folding `Δ_F` into
  the far contract halves the front hedge's linear residual and lands, in
  hands, close to the far-contract hedge.

Realized-minus-predicted residuals are the same number across all hedges on
a given day (the `½ Γ ΔS²` term), which confirms the linear algebra is
closed: nothing else is left over at first order.

**Decisions that follow from the math.**

1. The strategy always folds `Δ_F` into the far contract. Under the
   flat-forward-carry tail that is a no-op; under flat-q it is what makes
   the hedge complete against a proportional move. A `tail_residual="none"`
   switch exists for the identity test and diagnostics, and is not in the
   study grid.
2. `Δ_F` is derived from the identity, `Δ_F = Δ_q - Σ (F_i/S) B_i`, not
   measured. That costs no prices. The identity residual is recorded per
   day as a diagnostic so a bump-size mismatch between `Δ_q` (relative,
   `spot_bump`) and `B_i` (one index point) is visible, never hidden.
3. Buckets are central differences of one index point, through the same
   carry builder the engine prices with. This matches stage 04 and the
   probes exactly, and is the only route that produces flat-forward-carry
   buckets: `IndexFuturesCurve.bump_contract` extrapolates flat in nodal `q`
   only, so `calculate_futures_delta_buckets` cannot be used for that
   convention. The study's static stage is re-pointed at the same builder so
   static and replay buckets are one definition (pinned by test).

## 3. Scope and non-goals

In scope:

- A multi-leg futures hedge book in the library, used by every futures
  hedge (single-contract hedges become one-leg books).
- A bucket-weighted hedge strategy that returns a target per listed
  contract, with the tail fold, per-leg no-trade band and per-leg rounding.
- Per-day bucket sensitivities computed inside the replay, summed across a
  multi-product book like delta.
- Two new recorded frames: `hedge_legs` (one row per day per leg) and
  `hedge_attribution` (one row per day, the first-order decomposition of the
  hedged P&L and the hedged carry exposure).
- The hedged carry exposure recorded for EVERY cell, flat and term, behind
  an opt-in flag, so the single-contract cells carry the same measure the
  bucket cells are judged by.
- Two study cells, `term_flat_q__buckets` and `term_flat_fwd__buckets`, and
  the stage 03 measures and README text to read them.

Out of scope:

- Minimum-variance or regularised hedges that trade off basis against
  turnover. The exact bucket hedge is the first thing to measure; a
  regulariser is a separate decision once its cost is known.
- Bucket hedges under the flat carry models. There are no buckets without
  a curve; the strategy rejects a non-term dividend source.
- Any change to the simulation engine (`quantark.backtest.simulation`).
  That engine keeps its single-contract hedge; a bucket strategy handed to
  it is rejected, not approximated.
- Term-structured rates. `r` stays flat; the `ρ_i ↔ B_i` equivalence above
  assumes it.

## 4. Architecture

### 4.1 Ledger: `FuturesHedgeBook` (`quantark/backtest/futures_ledger.py`)

A mapping from contract to an open leg `(quantity, avg_price,
multiplier)`. Realized P&L lives in ONE accumulator on the book, not on the
legs, and every close adds to it in trade order. That is the arithmetic
`FuturesHedgePosition` performs today with a single leg (one running
`realized_pnl`, `mark = realized + unrealized`), so a one-leg book is
bitwise the old position, rolls and partial closes included; a per-leg
accumulator would re-associate the sum `(r1 + r2) + u` as `r1 + (r2 + u)`
and break the golden gate by an ulp. The average-price and close arithmetic
is factored into one shared function used by both classes.
`FuturesHedgePosition` itself is unchanged. The book exposes:

- `trade(contract, quantity_delta, price, multiplier)` — routes to the
  leg, creating it on first trade, dropping it when it goes flat.
- `mark_to_market(prices: Mapping[str, float]) -> float` — sum over legs of
  each leg's `mark_to_market(prices[contract])`; a leg whose contract has
  no price in the mapping raises (fail closed, never carries a stale mark).
- `quantity(contract) -> float`, `contracts() -> tuple[str, ...]`,
  `gross() -> float`, `net_spot_hands(prices, spot) -> float`
  (`Σ h_i F_i/S`).
- `realized_pnl` — the single accumulator.
- `single_leg` — a `FuturesHedgePosition` view (contract, quantity,
  avg_price, multiplier, the book's realized P&L) when at most one leg is
  open, and a `ValidationError` when more than one is.
  `ReplayBacktestEngine.hedge_position` becomes a property returning it, so
  the PnL-explain recorder, which reads `contract`, `quantity`, `avg_price`
  and `multiplier` from it, keeps working unchanged for single-contract
  hedges.

A single-contract hedge is a book with at most one open leg. The engine's
roll becomes two ordinary trades (close the old leg, open the new one) in
the same order and with the same trade rows as today, so the trades frame of
an existing run is byte-identical.

### 4.2 Strategy: `FuturesBucketHedgeStrategy` (`quantark/backtest/strategy/futures_bucket_strategy.py`)

Constructor: `delta_threshold` (per-leg no-trade band in contracts, same
semantics as the delta strategy), `round_contracts` (per leg),
`tail_residual: Literal["far", "none"] = "far"`, `hedge_ratio` (applied to
every leg).

    target_legs(*, buckets, delta_q, spot, product_quantity, multiplier)
        -> dict[str, float]

`buckets` is the day's list of `(contract, maturity, price, B_i)` for the
whole book (already quantity-weighted). Steps:

1. `h_i = -(B_i · q) / m` for every listed node, `q = product_quantity`.
2. `Δ_F = Δ_q·q - Σ (F_i/S) B_i·q`; if `tail_residual == "far"`,
   `h_n -= Δ_F · S / (m F_n)`.
3. Multiply by `hedge_ratio`; round each leg if `round_contracts`.
4. Return the dict, including zero targets for contracts the book holds but
   the curve no longer lists (so they are closed).

`should_rebalance(current, target)` per leg is the delta strategy's rule.

`AutocallableDeltaHedgeStrategy` is unchanged. The engine adapts it to the
same shape: `{selected_contract: target_contracts(...)}`.

### 4.3 Carry context and bucket sensitivities (`quantark/backtest/replay/`)

`term_dividend_yield` currently builds the quotes and the curve and returns
only the dividend object. It gains a companion:

    @dataclass(frozen=True)
    class CarryCurveContext:
        quotes: tuple[IndexFuturesQuote, ...]
        spot: float
        rate_curve: Any
        extrapolation: str
        underlying: str
        artifact: Any | None

        def dividend(self) -> Any                       # the day's object
        def bumped(self, contract: str, points: float) -> Any
            # same builder, one quote's price moved by `points`

`ProductReplay._term_dividend` keeps the context on
`self.last_carry_context` (None under a non-term source).
`ProductReplay.calculate_futures_buckets(product, env, engine)` prices the
product once per node up and once down through `context.bumped(...)` with
spot pinned, and returns `[(contract, maturity, price, B_i)]` in
currency per index point. The engine sums `B_i · quantity` across replays
exactly as it sums delta, on the same day-loop pass, using each replay's
own pricing engine (vol-model days included).

**Carry exposure recording.** A replay-config flag
`record_carry_exposure: bool = False` (both configs, appended last) turns on
the daily measurement; a bucket strategy forces it on because it needs the
buckets anyway. What is computed depends on the carry family:

- term source (`futures_curve`, `surface_forwards`): the buckets above,
  every day, whether or not the strategy uses them. Per node,
  `ρ_i = -B_i F_i T_i`; the product's parallel carry sensitivity is
  `Σ ρ_i`. The book's per-node exposure is `-h_i m F_i T_i` with the
  market `F_i`.
- flat source (`None` / `active_contract`): one central bump of the flat
  yield (two prices) gives the product's `ρ_flat`; the book's exposure to
  the same flat yield is `-Σ_i h_i m F_i T_i` over its legs with market
  `F_i` and each leg's own tenor. The flat yield is a one-parameter fit to
  the chain, and its fit error is basis risk the model cannot see: the RMS
  over listed contracts of `ln(S e^{(r-q)T_i} / F_i)` is recorded as
  `flat_fit_rmse_bp` (stage 01's `fwd_err_rms_flat`, now daily and
  per cell).

Cost: two prices per listed node per day under a term source on top of
today's base, delta and gamma prices, so four nodes take a QUAD study run
from about three prices a day to eleven, roughly 330 s per run against
90 s. Under a flat source it is two prices a day, roughly 150 s per run.
With the flag off nothing changes and the goldens are untouched.

### 4.4 Engine day loop (`engine.py`)

Unchanged for single-contract hedges except that `hedge_position` is now
`hedge_book` and the roll goes through it. When the strategy is a
`FuturesBucketHedgeStrategy`:

1. The roll policy still selects `selected` (the front-month policy by
   default). It drives `basis_yield`, `futures_price`, `futures_ttm`,
   `active_contract` and `multiplier` in the state row, so those scalar
   columns keep their meaning ("the front month") and the real book is in
   `hedge_legs`. No hedge trade is tied to `selected`.
2. After the greeks pass, `buckets` and `delta_q` are handed to
   `target_legs`. Each leg is rebalanced independently through the band,
   trades are executed against that day's chain price and multiplier for
   that contract, and one `rebalances` row per leg is written (same columns
   as today; `active_contract`, `current_contracts` and `target_contracts`
   are that leg's).
3. A leg whose contract is no longer a curve node gets target 0 and is
   closed with `trade_type="hedge_close"`, `reason="bucket_leg_retired"`.
   The curve's minimum tenor (`futures_curve_min_tenor_days`, 7 by default)
   is the bucket hedge's roll rule; no separate roll buffer.
4. `hedge_mtm` in the state row is `hedge_book.mark_to_market(prices)` with
   the day's chain prices, so `total_pnl` and `portfolio_value` keep their
   identities.

### 4.5 Recorded frames (`schema.py`)

`HEDGE_LEG_COLUMNS`: `date, contract, maturity, price, multiplier,
bucket, bucket_hands, target_contracts, held_contracts, trade_contracts,
leg_mtm, is_tail_node, product_rhoq_bp, hedge_rhoq_bp, net_rhoq_bp`. Written for every futures run (one leg for a
single-contract hedge; `bucket` and `bucket_hands` are NaN when no buckets
were computed). `is_tail_node` is true for the last curve node when any
alive product's maturity lies beyond it, the same flag the library bucket
function sets.

`HEDGE_ATTRIBUTION_COLUMNS`: `date, carry_family, product_dv, hedge_pnl,
book_dv, linear_spot, linear_basis, gamma_term, remainder, delta_q_hands,
delta_f_hands, identity_residual_hands, gross_contracts, net_spot_hands,
product_rhoq_bp, hedge_rhoq_bp, net_rhoq_bp, net_rhoq_gross_bp,
flat_fit_rmse_bp`. One row per day whenever `record_carry_exposure` is on;
empty otherwise.

P&L attribution uses the previous day's sensitivities and holdings (what
the book carried overnight), `gamma_term = ½ Γ ΔS²`, and `remainder =
book_dv - linear_spot - linear_basis - gamma_term` (theta, vega, higher
order). The linear terms are defined per carry family, named in
`carry_family`:

- `term`: `linear_spot = Δ_F ΔS`, `linear_basis = Σ (B_i + m h_i) ΔF_i`.
- `flat`: `linear_spot = Δ_q ΔS + hedge_pnl` (the delta-hedged residual)
  and `linear_basis = ρ_flat Δq_flat` (the re-mark from the flat yield
  moving, which is what jumps on roll days).

The first day, and days where the chain's contract set changed, have NaN
linear terms.

Carry exposure, all in bp of notional per one percentage point of yield,
measured on the day's own sensitivities and the book AFTER rebalancing:
`product_rhoq_bp` is `Σ ρ_i` (term) or `ρ_flat` (flat); `hedge_rhoq_bp` is
the book's exposure to the same shift; `net_rhoq_bp` is their sum;
`net_rhoq_gross_bp` is `Σ_i |ρ_i^book|` over nodes (term only, NaN for
flat), which is the number a bucket hedge must drive to zero up to
rounding while a single-contract hedge cannot; `flat_fit_rmse_bp` is the
flat family's chain-fit error (NaN for term). `hedge_legs` carries the
per-node pieces, `product_rhoq_bp`, `hedge_rhoq_bp` and `net_rhoq_bp`, so
the four-node table above is reproducible for any day of any cell.

`STATE_COLUMNS`, `TRADE_COLUMNS`, `REBALANCE_COLUMNS`, `GREEK_COLUMNS`:
unchanged.

### 4.6 Configuration and validation (`config.py`)

No new config fields: the strategy object selects the behaviour. Validation
in both replay configs:

- a `FuturesBucketHedgeStrategy` requires `hedge.kind == "futures"` and
  `engine_config.uses_term_dividend_source()`; otherwise `ValidationError`
  naming the missing piece;
- it is incompatible with `dividend_roll_policy` (already excluded by the
  term-source rule; the message says so);
- it is incompatible with `pnl_explain` in this version: the recorder
  explains one hedge leg, and a multi-leg explain is a separate design.

The simulation engine's config rejects the strategy type outright.

## 5. Study changes (`example/snowball_q_term_structure`)

- `_common.py`: `HEDGE_POLICIES` gains `"buckets"` (returns the front-month
  roll policy for the recorded scalar columns); a new `HEDGE_STRATEGIES`
  map returns the delta strategy for `front`/`far` and
  `FuturesBucketHedgeStrategy(delta_threshold=..., round_contracts=True)`
  for `buckets`. `HEDGE_POLICY_LABELS["buckets"] = "bucket-neutral across
  the listed chain, tail folded into the far contract"`.
- The static bucket computation in stage 01 moves onto `CarryCurveContext`
  so static and replay buckets share one definition, and is computed for
  every term model (today it is `term_flat_q` only through the library
  curve).
- `02_backtest_fleet.py`: `DEFAULT_CELLS` gains `("term_flat_q",
  "buckets")` and `("term_flat_fwd", "buckets")`. `write_run` persists the
  two new frames. The resume fingerprint is unchanged for existing cells.
- `03_aggregate_and_report.py`: new measures from `hedge_attribution` —
  per-run std of `linear_basis`, `linear_spot`, `gamma_term`, `remainder`,
  mean `gross_contracts`, and the carry-exposure summary: mean
  `|net_rhoq_bp|`, mean `net_rhoq_gross_bp` (term cells), mean
  `flat_fit_rmse_bp` (flat cells), each against the unhedged
  `|product_rhoq_bp|` so mitigation reads as a fraction. New pairs: each
  bucket cell against its own model's `__front` and `__far` cells, and
  `term_flat_fwd__buckets - term_flat_q__buckets` (the tail convention
  measured on P&L). Report section and README results text.
- Every cell runs with `record_carry_exposure=True`, so the flag enters
  the resume fingerprint and the whole fleet re-prices: 10 cells × 29
  inceptions = 290 runs, roughly 6.5 hours at four workers (term cells
  ~330 s, flat cells ~150 s). The current 232 runs stay on disk under the
  old fingerprint until the new ones replace them. If that cost is not
  wanted up front, the flag can be restricted to the two bucket cells and
  their four direct comparators (`term_flat_q__front`, `term_flat_q__far`,
  `term_flat_fwd__front`, `flat_from_hedge__front`), 174 runs, at the
  price of the other four cells lacking the measure.

## 6. Testing

Library, all TDD:

- Golden gate: the three replay goldens' existing frames are byte-identical
  after the ledger swap; `hedge_legs` is added to the golden set.
- `FuturesHedgeBook` with one leg equals `FuturesHedgePosition` bitwise
  (`==` on floats, not approx) over a scripted sequence of adds, partial
  closes, a full close and a roll; `mark_to_market` on a missing price
  raises.
- Bucket sensitivities on the test chain: the identity residual is below
  0.01 hands under both extrapolations; the flat-forward-carry buckets
  reproduce the closed-form tail derivatives `1 + (T-T_n)/ΔT` and
  `-(T-T_n)/ΔT` at a tail tenor to three decimals.
- Strategy: with `tail_residual="none"` the book's `net_spot_hands` equals
  `Δ_F`; with `"far"` it is zero to rounding; every node's recorded
  `net_rhoq_bp` is zero before rounding and bounded by one contract's
  `m F_i T_i` after it; a node that leaves the curve gets a zero target.
- Carry exposure under a flat source: `product_rhoq_bp` equals the greeks
  calculator's rhoq on the fixture; `flat_fit_rmse_bp` equals stage 01's
  `fwd_err_rms_flat` on the same chain; a single-contract book's
  `hedge_rhoq_bp` is `-h m F_j T_j` to the last ulp.
- Attribution: on a two-day fixture with a hand-built chain move, the
  recorded `linear_spot + linear_basis` equals the algebraic residual from
  section 2 for the hedge held.
- Validation: bucket strategy with a flat dividend source, with a spot
  hedge, and in the simulation config each raise.

Study:

- `dividend_for` / static buckets pinned against `CarryCurveContext` per
  model; cell grid and pairs tests extended; the study test that
  `flat_from_far__far ≡ flat_from_hedge__far` unchanged.

## 7. Risks and what the numbers will be read for

- **Transaction cost.** The flat-forward-carry hedge traded 2–16 hands a
  day on ~100 gross in the probe, roughly double the front hedge. The
  attribution frame reports cost separately; a bucket cell that wins on
  `linear_basis` and loses on cost is a real finding, not a failure.
- **Whole-contract rounding across four legs** can leave up to two hands
  of net spot open. `net_spot_hands` is recorded so this is visible.
- **The carry-exposure check is the strategy's own audit.** A bucket
  cell whose `net_rhoq_gross_bp` is not within rounding of zero on every
  day has a bug, whatever its P&L says; that check runs before any P&L is
  read.
- **The tail convention.** Under flat-q the bucket cell is expected to
  land near `term_flat_q__far` (the fold makes them nearly the same
  book). Under flat-forward-carry it is expected to remove most of
  `linear_basis`. Whether the remaining daily std is gamma, theta and vega
  is exactly what the decomposition answers; if `remainder` dominates, no
  linear futures hedge can improve on the far contract and the study says
  so.
