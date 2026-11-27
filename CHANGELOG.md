# Changelog

All notable changes to this project will be documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).
During 0.x the public API may still change between minor versions.

## [Unreleased]

## [0.16.0] - unreleased

Follow-up release built from the 0.11.0–0.15.0 open items. Sections are added as each part lands.

Notes:
- This release edits trees fingerprinted by the intraday certificates: `COMMON_TREES`
  (`asset/equity/product/option`, `asset/equity/settlement.py`, `asset/equity/riskmeasures/greeks`, `priceenv`),
  `PDE_TREES` (`engine/pde`), `QUAD_V2_TREES` (`engine/quad/v2`) and `MC_TREES` (`engine/mc`). Their next amendment
  re-runs rather than carrying forward.

### DCN session parity

Every DCN engine serves the `PricingSession` `EVENT_STATS` operation, and a lifecycle state sent
with an `EVENT_STATS` request is honoured or refused, never dropped. Stateless numbers are
unchanged, bit for bit.

#### Added

- **`EVENT_STATS` on all eight DCN engines.** `DCNMCEngine`, `LocalVolDCNMCEngine`,
  `HestonDCNMCEngine`, `QEDCNMCEngine`, `CoupledCoarseHestonDCNMCEngine`, `DCNPDEEngine`,
  `LocalVolDCNPDEEngine` and `HestonDCNPDESolver` return their own `calculate_event_stats` object
  through a session. The value matches the direct call bit for bit, with or without a
  `DCNLifecycleState`, on the final observation day and for terminated notes. On the MC engines
  this also holds with several batches on the serial and threads backends. The MC adapters run the
  same batch plan as `PRICE_DETAILED`. The two 1D PDE engines have new session adapters,
  `EquityDCNPDESessionAdapter` and `EquityLVDCNPDESessionAdapter`, which advertise the operation.
  The `EVENT_STATS` operation accepts only the `PV` output, as on every other adapter. A PRICE
  request with the `EVENT_STATS` output is still refused on the DCN PDE adapters.

#### Fixed

- **A lifecycle state on a session `EVENT_STATS` request is no longer dropped.** Before, the
  legacy statistics call ignored the request's `lifecycle_state`. The Snowball / Phoenix /
  KO-reset MC and PDE engines then returned the statistics of a fresh trade for a mid-life
  request, with no warning (21 inventoried engines). Those engines do not support lifecycle
  states (`supports_lifecycle_state=False`, and their stateful `price()` already raises), so they
  now refuse with `CapabilityError`. On these engines, put the trade's history on the product
  (aged schedule, `knocked_in_at_valuation`). Engines that do accept a state (`HestonDCNPDESolver`
  on the legacy route) now receive it. A fully settled note is refused as before.
- **The same holds for `PRICE_DETAILED` on the legacy route.** A stateful session `PRICE_DETAILED`
  request on `HestonDCNPDESolver` returned the stateless price (a mid-life receivable note:
  -9157.76 instead of 841.27). The state is now forwarded, or refused where unsupported.

#### Changed

- `docs/execution/capability-matrix.md`: the DCN PDE rows give an accurate prepared rationale
  (their statistics are the `EVENT_STATS` operation's forward pass, not an output of the value
  solve).

### QUAD V1 statistics from one forward pass

The QUAD V1 event statistics (`SnowballQuadEngine`, `PhoenixQuadEngine`,
`KOResetSnowballQuadEngine`) read every outcome off one adjoint forward pass of the march. That
covers the knock-out units, the maturity and knocked-in settlement columns, and the Phoenix coupon
columns. Before, they marched one backward column per knock-out record, plus two to eight per
Phoenix observation (two for no-memory instant coupons, up to eight for memory expiry coupons). That is what made 0.13.0's default statistics 41–82× slower on daily-KO
books. Prices and greeks are unchanged, bit for bit.

#### Changed

- **Speed** (back to back on one machine, `calculate_event_stats` wall time including its
  `price()`; base = 0.15.0):

  | contract | 0.15.0 | 0.16.0 | ratio |
  |---|---|---|---|
  | daily KO 1y, KI at maturity | 7.46 s | 0.53 s | 14.1× |
  | daily KO 2y, KI at maturity | 28.46 s | 1.06 s | 26.8× |
  | daily KO 1y, continuous KI | 7.50 s | 0.66 s | 11.4× |
  | monthly KO, daily KI 1y | 1.49 s | 0.83 s | 1.8× |
  | monthly KO, daily KI 2y | 10.47 s | 4.36 s | 2.4× |
  | Phoenix (no memory, instant), daily KI 1y | 2.87 s | 0.87 s | 3.3× |
  | Phoenix daily observations 1y, no memory, instant | 23.34 s | 0.91 s | 25.8× |
  | Phoenix daily observations 1y, no memory, expiry | 38.73 s | 1.52 s | 25.4× |
  | carried KI, monthly KO 1y | 1.47 s | 1.01 s | 1.5× |
  | quarterly KO, continuous KI 2y, `streams={KO}` | 0.054 s | 0.059 s | 0.9× |
  | Phoenix quarterly (memory), continuous KI 1y, `streams={KO}` | 0.058 s | 0.068 s | 0.85× |
  | no KI barrier, monthly KO 1y | 0.037 s | 0.038 s | 1.0× |

  - **Sparse books.** 0.15.0 ran no forward pass on a pruned request (`streams={KO}`) or a note
    without a KI barrier. On a few observations the pass's fixed per-step cost is not repaid, so
    these calls are up to about 10 ms slower (the last three rows). A carried knock-in also ran
    no pass before, but its book is large enough to gain (1.5×).
  - **Memory Phoenix.** A memory Phoenix is capped at 50 observations by `price()`. At 50
    observations its statistics are 2–3× faster, with `price()` itself the remaining cost.
- **Statistics move at round-off.** Forward pass vs the 0.15.0 columns:
  - knock-out probability: at most 3e-16 per record;
  - terminal probabilities: 1e-15;
  - coupon probability and cash: 2e-15;
  - EXPIRY ledger buckets: 3e-13 (100 notional).

  Partition residuals and cash reconciliation stay within their 0.15.0 bounds. The knock-in path
  fields come from the same pass as before and are bitwise.
- Every statistics call that runs the march now runs exactly one forward pass. Before, a pruned
  request, a carried or current knock-in, or a note without a KI barrier ran none. The knock-in
  fields are still reported only when requested.
- A march the adjoint cannot transpose (an event map that is not local in the probe band) falls
  back to the stacked backward columns, which give 0.15.0's numbers.

### Trading-clock follow-ups

Four 0.15.0 open items around TradingClock-wrapped cash curves. Environments with no wrapper and no
`CALENDAR_YEAR` declaration are unchanged, bit for bit (full suite).

- **A dividend created from `None` by a bump follows the declaration.** On a `BUSINESS_DAYS` environment whose
  `div_yield` is `None` and declared `CALENDAR_YEAR`, the dividend-rho bump, the greek-conventions report and the
  carry rho-q bucket bumps used to assign a native curve read per trading year. It is now wrapped on the
  environment's map (`place_created_cash_curve`). Moved: dividend rho on such environments, e.g. a 10-month ATM call
  from 7.624984 to 7.619201 for +1% q. There is no general assignment hook, because a curve built over a wrapper
  would be wrapped twice (0.15.0 D13).
- **Basis/dividend relationship modes across wrappers.** `AUTO_ADJUST_BASIS`, `AUTO_ADJUST_DIVIDEND` and
  `SYNCHRONIZED` were refused on wrapped environments. They now run on the calendar-quoted inners at the mapped
  tenor (`relationship_tenor`), in `StressApplicator` and `DynamicScenarioEngine`. `INDEPENDENT` on a fully wrapped
  environment now reads its source at the mapped tenor too, so a term-structure source gives a different flat value
  than in 0.15.0. A derived curve in an empty slot is wrapped on the sources' clock. Partially wrapped environments,
  a `TRADING_YEAR` target slot, and wrappers on different clocks stay refused.
- **Rate and carry bumps over a wrapper shift the calendar quote.** These sites used to build the bumped curve on
  top of the wrapper, a per-trading-year shift: local-vol rho, Heston and Heston-SLV PDE rho, the convertible
  floor-bond and full DV01/convexity, and the carry rho-q buckets. They now shift the wrapper's inner and re-wrap
  on the same map. Carry rho-q bucket windows are mapped onto the calendar axis. Building the created-dividend
  buckets on the environment axis would have read every window at the wrong time and returned an all-zero table;
  the checkpoint review caught this. Moved: these greeks on wrapped environments only.
- **Taylor PnL explain on wrapped environments.** It used to be refused; parallel Taylor now runs. The greeks are
  forced onto the numerical route, because the analytical formulas read the environment's axis. Gate:
  - delta, vega, rho and rho-q each equal an independent finite difference on the inner-shifted environment, at the
    calculator's bump size, stencil and bump engine (BS on calendar-axis, hand-wrapped and declared environments,
    plain and holiday steps; snowball on PDE);
  - every factor move is the inner quote's move;
  - the per-step theta equals an independent one-step date advance, and the pure-time Taylor row equals the
    waterfall TIME row;
  - halving the moves shrinks the residual at least 3x on a plain step (2x across a holiday, where the cross
    terms with the fixed time step dominate on any environment).

  The informational `r_theta` / `q_theta` rows are repriced (`theta_decomposition_mode="exact"` is forced on
  wrapped environments): the estimate split reads r, q and T on the environment's axis against an inner-unit rho,
  and was off by (T_cal/T_td)^2 at the product's maturity (7.5% on a 2026-12 call across National Day). Bucketed Taylor stays refused on wrapped environments.

Pins moved: the 0.15.0 refusal tests for relationship modes and Taylor on wrapped environments now assert the new
behaviour (`test_curve_quoting_shifts.py`, `test_pnlexplain_trading_clock.py::test_fail_closed_rules`).

Open items:
- Key-rate rho with a `None` dividend declared `CALENDAR_YEAR` next to a native `TRADING_YEAR` rate keeps a native
  carry-invariant dividend. It is built from trading-year rate differences; a wrapped `CALENDAR_YEAR` rate refuses
  key-rate first.
- FX engines' rate bumps (no trading clock on `FxPricingEnvironment`).
- `INDEPENDENT` on a mixed (partly wrapped) environment reads a wrapped source at `T`, not `T_cal`.
- Bucketed Taylor on wrapped environments.
- `carry_curve_buckets` derives a dividend curve from a possibly wrapped rate (`to_dividend_yield`), like the
  futures-implied dividend curve: consistent on the trading axis, not moved onto the calendar quote.
- `clock_parallel_shift_rate` composes `ParallelShiftRateCurve` on the inner; for a `ZeroCurve` inner the
  wrapper's own `parallel_shifted` would keep the anchor (`AnchoredParallelShiftRateCurve`). The numbers agree up
  to rounding.

### Same-instant observation records on dated products

A dated autocallable can carry two observation records that resolve to one environment instant: for
example, on a US `BUSINESS_DAYS` clock with `PRECEDING` holiday placement, a 07-03 record and a
07-04 (holiday) record both sit at the same env time. Before, the engines disagreed on such a
product. The PDE applied the records in a backward march, so the later record overwrote the earlier.
QUAD V1 kept the first record and dropped the rest. QUAD V2 refused. MC tested both. Every product
whose records never share an instant prices and reports statistics bit for bit as in 0.15.0. Pinned
products, with price and full statistics, cover Snowball, Phoenix (every coupon variant, nodal,
valuation instant) and KO-reset on 1D, local-vol and Heston.

#### The zero-gap rule

Records on one env instant are consecutive observations with zero elapsed time: the same spot, in
contract order, under each product's own per-date rules. The desk rule "a knock-out beats a
same-instant discrete knock-in" holds across the whole instant.

- **Snowball.** Both knock-out records are tested at the same spot; the first that triggers pays its
  own coupon. A record whose knock-out region is contained in an earlier one at the instant can
  never fire and drops out (`same_instant_ko_groups` in `observation_schedule.py`), so a pure
  duplicate is exactly the product without it.
- **Phoenix.** No record is dropped. Each record tests its coupon (memory carries) and then its
  knock-out; a knock-out ends the note, and later records at the instant do nothing. Under `EXPIRY`
  payment, a coupon earned by record 1 and then ended by record 2's knock-out at the same instant
  is paid at record 2's settlement.
- **KO-reset.** The instant is one unit: every schedule-1 knock-out record, then the knock-in
  (schedule switch), then every schedule-2 knock-out record.
- **Knock-in pairs.** Two discrete knock-in records at one instant are one knock-in test at the
  widest level, since a knock-in is idempotent at one spot. In the statistics, the instant's
  knock-in probability sits on the first record in contract order whose level is breached; later
  records at the instant carry 0.

#### Changed: moved prices (sanctioned)

US `BUSINESS_DAYS` repro, records 03-03/05-05/07-03/07-04/09-03/11-03, KI 75, 07-03 and 07-04 at
env time 0.5:

| product | engine | 0.15.0 | 0.16.0 | MC |
|---|---|---|---|---|
| Snowball, 07-04 step-down | 1D PDE | -0.445991 | -0.2697 | -0.255036 |
| Snowball, 07-04 step-down | QUAD V1 | -0.150679 (step-down ignored) | -0.2653 | |
| Snowball, 07-04 step-down | QUAD V2 | refused | -0.2647 | |
| Snowball, 07-04 duplicate | 1D / Heston PDE | -0.152781 / -0.316627 (drift) | = the product without the duplicate (-0.155122 / -0.318811) | |
| Phoenix, duplicate | PDE / QUAD V1 | -1.41192 / -1.412 (07-04 coupon dropped) | -1.071155 / -1.070890 | -1.049648 (z -1.26) |
| Phoenix, step-down | PDE / QUAD V1 | -1.42396 / -1.410 | -1.083196 / -1.081637 | -1.059157 (z -1.42) |
| Phoenix, any pair | QUAD V2 | refused | priced (within 1.4 SE of MC) | |
| KO-reset, schedule-1 step-down | PDE | 8.657 | 9.016354 | 9.003973 (z +1.13) |
| KO-reset, knock-in pair 80 → 75 | PDE | 8.8717 | 7.732242 | 7.715935 (z +1.27) |
| KO-reset, schedule-1 step-down | QUAD V1 | the single-record price (step-down ignored) | 9.014574 | 9.003973 (z +0.97) |
| KO-reset, overlapping KI/KO levels at one instant | QUAD V1 | 4.87 / 4.19 | 6.059480 / 4.362742 | 6.043249 / 4.346320 (z +1.03 / +1.01) |

The KO-reset rows are the final 0.16.0 values, measured on the final library (MC: 1M paths, seed 7). They include the
V1 projection from "QUAD numerics" below and the knocked-in put at the exercise date ("Knocked-in KO-reset put at the
exercise date" below). The PDE values equal the ones first measured for this section: run 9 reverted the grid engines'
knocked-in end date, and the projection is V1 only. MC moved: these notes are exercised on 2026-01-05, after schedule
2's last fixing on 2025-12-03.

- **Snowball PDE equal-level knock-in duplicates** move to the product without the duplicate (1D
  2.474447 → 2.476399; local-vol and Heston likewise). The repeated cell-average projection drifted.
- **Valuation-instant decisions.** A same-instant pair decided at the valuation instant pays the
  deciding record. Phoenix V1 valuation-instant coupons are each discounted at their own
  settlement; they were discounted at the last paid record's, a -3.97e-4 error. The Phoenix V1
  memory state at the valuation instant is fixed; the both-missed case was -19.485 vs MC -19.139,
  z -12.7. The Heston KO-reset decision by record 2 now pays 10.0, not record 1's 15.0.

Each engine is gated against MC (3 SE plus 1e-3 relative): 1D, local-vol and Heston PDE, QUAD V1,
QUAD V2, for single, duplicate, step-down and knock-in-pair products
(`test/test_asset/equity/cross_engine/test_same_instant_ko_gate.py`). Exact references: a two-record
Snowball and Phoenix are priced in closed form over the bands of the instant's spot. QUAD V1/V2 are
within 1e-6 relative, and the PDE within 1e-3 relative, converging under refinement.

#### Changed: statistics and readers

- Event statistics report one row per record on every engine, per-record knock-out, coupon and
  knock-in probabilities and cash. Before, they raised `ObservationTable` / `KnockInTable` "duplicate
  rows". The tables accept a repeated time only for dated rows with distinct dates; every other
  repeated-time table still refuses.
- `ProductReplay`'s `next_ko_probability` sums every knock-out record of the first instant. Before,
  it read only the first record's share. On a KO-reset whose two schedules share a date, the sum also covers the
  schedule-1 and schedule-2 rows of that date, so it is the full probability of knocking out there. Before, a
  knocked-in note read the dead schedule-1 row; at inception the move is round-off (0.37514546 → 0.37514552 in one
  probe). The autocallable risk report's barrier watch shows the
  instant's most reachable knock-out level and the widest knock-in level.
- QUAD V1 and PDE KO-reset statistics tables now carry observation `dates` on dated products. This
  is metadata only; prices and rows are bitwise.

#### Changed: cash-leg folding

Folding cash (`FoldOverlay`) onto a same-instant group of two or more effective knock-out records
raises `ValidationError` naming the dates, because the binder is one-to-one by time. A pure
duplicate binds its effective record; cash folded onto the dominated record is never paid.

#### Fixed: holiday-placed pending flows

A date-based pending flow paid after the valuation date, whose env time is 0 (paid on a holiday
placed onto the valuation instant), is timed at 0 (`settlement.resolve_pending`). It used to be
refused with "payment must be strictly after valuation".

#### Added

- A guard test (`test/test_backtest/test_replay_probe_env_clock.py`) keeps `ProductReplay`'s maturity
  probe on the same clock keywords as its day environment.

Pins moved: refusal and old-behaviour pins only. These are the 0.14.0 collapsed-instant refusal in
`test_dcn_collapsed_instants.py`, and the old-behaviour Phoenix / KO-reset pair prices in
`test_same_instant_ko.py`, whose old hex values are kept in a comment.

Open items:
- Resolved later in this release: the KO-reset MC gap (+2 to +3 SE; z −9 with
  `disable_ko_after_ki`) came from the contract reading. MC read a knocked-in note's put at schedule
  2's last fixing; the grid engines read it at the exercise date. The owner ruled for the exercise
  date (2026-11-27), and MC now reads it there ("Knocked-in KO-reset put at the exercise date"
  below). Also, with `disable_ko_after_ki` a schedule-2 knock-out record on the knock-in instant is
  disabled (owner ruling), now on QUAD V2 too ("QUAD numerics" below).
- QUAD V2 KO-reset knock-out pairs still refuse. KO-reset `REBASED` on QUAD V1 and the PDE stays
  refused (pre-existing).
- Heston PDE knock-in statistics rows before a shared instant are 3.5–10 SE off MC, and so is one
  record (route discretisation). 1D nodal knock-in rows are 3.6 SE off with one record.
- A breached knock-in pair at t=0 reports the single "knocked in at valuation" row without naming
  the deciding record.

### QUAD numerics: V1 bump greeks, KO-reset fixes, FFT lengths

Changes to QUAD V1 (`SnowballQuadEngine`, `PhoenixQuadEngine`, `KOResetSnowballQuadEngine`), plus a KO-reset contract
fix on QUAD V2. A second KO-reset change, the knocked-in end date on the grid engines, landed here and is reverted in
this release by owner ruling 2026-11-27 (below, and "Knocked-in KO-reset put at the exercise date"). The KO-reset
changes and the FFT round-off moves are owner-signed (2026-11-27). The KO-reset, Snowball and Phoenix flat-BSM
certificates were amended for the V1 projection and the FFT lengths (see Notes).

#### Changed: V1 bump greeks on the base lattice

V1 sizes its lattice from the inputs: the node count N, the domain (cell width h) and the cell stretch. A bumped leg
used to size its own lattice, so every vega and theta leg was priced on a different grid from the base. At a sizing
switch, the switch's own jump was read as a greek: 1 h of theta across the 5001-node cap gave -4.1/y against a
reference of +3.69/y. On ordinary products the re-meshing showed up as greek error, e.g. a 1-day dated theta 1.7e-2
relative off QUAD V2.

- **`create_bump_context` freezes the base sizing (owner ruling: "freeze sizing at base").** V1 returns a clone that
  carries the base record: N, domain and cell stretch. Every leg priced through it uses that record, whatever the date
  or bumped environment. The time-dependent routing is recomputed on the frozen cell for each leg (the exact-first-step
  decision, the local-step mask and the kernel guard), as the PDE keeps its spatial layout and rebinds time. The record
  binds to the base product or a same-contract copy: same type, strike, initial level and KI level, with a life that
  fits the frozen domain. Any other product sizes itself. A frozen clone's `create_bump_context` returns itself, and
  the auto_converge ladder runs once per clone. If a leg's routing on the frozen cell refuses an under-resolved step,
  that leg alone prices on its own sizing, bit for bit as before (`frozen_sizing_fallbacks` counts these).
- **Moved: V1 bumped greeks** (vega, theta, volga, the cross greeks built on them) on every path through the bump
  context: `GreeksCalculator`, `BaseEngine.calculate_greeks`, carry-curve buckets, trade-value theta, position trade
  risk, execution greeks and pnl-explain `ScenarioCache`. Measured against QUAD V2 (or V1 at 16001 nodes):
  - 1 h theta across the 5001-node cap: 6e-9 abs from the reference; it was -4.1/y against +3.69/y.
  - dated 1-day theta on a mid-week daily-KI, monthly-KO 1y snowball: 7.9e-5 relative; it was 1.7e-2. Its vega:
    1.7e-4; it was 1.0e-3.
  - vega across the exact-first-step threshold: 1.84e-4 per vol point. The leg still switches route there.
  - breadth gate: 16 Snowball / Phoenix / KO-reset products, 32 vega and theta rows. None is worse than with the
    pre-freeze re-sizing legs; the worst ratio of new to old error is 0.96.
- **Unchanged by the freeze:** base price and statistics. On spot-independent vol (flat or term structure), delta,
  gamma and rho are also unchanged, including auto_converge and cell stretch. On a spot-dependent smile surface the
  delta/gamma legs now share the base lattice and move within V1's default-grid error: one product's gamma error goes
  from 27% to 0.1%, another's delta error from 1.5e-4 to 3.0e-4.
- **pnl-explain:** `ScenarioCache` values the t1 endpoint on the t0 bump context, now the frozen V1 lattice. The
  backtest books t1 on a fresh lattice, so `gap_states` on V1 books is no longer exactly zero: it is up to about 1e-6 of
  book value. The reconciliation tests allow 1e-8 of the day's PnL plus 2e-6 of book value. `pnlexplain/README.md`
  says "gap_states is zero" only for engines without a bump context.

#### Changed: KO-reset V1 knock-outs use the cell-average projection (owner: "land it")

KO-reset V1 knock-outs used the nodal cosine smoothing (`_smooth_step_weight`) even under the default cell-average
`event_projection`, unlike Snowball and Phoenix. They now project as Snowball does: single records through
`_project_quad_event`, a same-instant group through the piecewise projection in the instant-unit order, and the
pre-KO weight that narrows a same-instant discrete KI through the breach fractions. One code path serves price,
statistics, the first step and the adjoint. The nodal and width-0 modes are unchanged, bit for bit.

- **Moved: every default KO-reset V1 price, greek and statistic.** On the certification variants at 1001 nodes, the
  error against QUAD V2 falls from +2.0e-3..+1.36e-2 to 3.5e-4..2.1e-3 (3-21x closer); certificate pv anchors move by
  -3.3e-4 to -1.47e-2.
- **Not monotone outside those variants.** The old smoothing's O(h) bias was partly cancelling V1's O(h²) event error.
  On products with a discrete KI on the schedule-1 KO dates the new default-grid error is clean O(h²) but can be larger:
  in a 12-product breadth probe 6 got closer and 6 further (worst -1.04e-3 → -3.47e-3), and the cash-leg bank product
  is erratic in N (401 nodes +5.1e-3 → -1.06e-2; 801 +3.3e-3 → +1.5e-4). Refining the grid converges both.
- Cash-leg baselines record the move as cause "t"; the cross-engine smoke allowance for the KO-reset V1 cell was
  re-banked.

#### Reverted: a knocked-in KO-reset ends at schedule 2's last fixing (D31)

Run 5 staged the knocked-in leg at schedule 2's last fixing on the PDE (1D and local vol), Heston / Heston-SLV ADI,
QUAD V1 and QUAD V2, to match MC's D31 readout. The owner then ruled the other way (2026-11-27): a knocked-in note's
put is read on the spot at the product's exercise date and paid at the exercise settlement. Run 9 reverts the
staging, and MC moves instead ("Knocked-in KO-reset put at the exercise date" below). The grid engines are back on
their 0.15.0 exercise reading; notes whose exercise date is schedule 2's last fixing never moved.

- **The KO-reset MC gap** listed under "Same-instant observation records" (MC +2..+3 SE above the grid engines; z −9
  with `disable_ko_after_ki`) came from the contract reading: MC read the put at schedule 2's end, the grid engines at
  the exercise date. With MC on the exercise reading, the investigation's 12 cases (10, plus 2 settlement variants;
  MC 1M paths, seed 7) agree: every grid engine (PDE, LV-PDE, QUAD V1, QUAD V2) is within −0.53..+1.20 SE of MC.
- Referee (carried knock-in with `disable_ko_after_ki` = a European put to the exercise date, exact
  −19.86527359248963): MC moved from −19.76689 to −19.86353. The grid engines are within 1.47e-6 (PDE, LV-PDE),
  6.3e-6 (V1) and 1.8e-15 (V2) relative.
- Cash-leg folding binds the knocked-in slot at the exercise terminal on every engine, MC included. EXPIRY schedule-2
  knock-outs and the statistics rows are timed there too.

#### Fixed: QUAD V2 KO-reset `disable_ko_after_ki` (owner: "disable that knockout is correct")

With `disable_ko_after_ki`, a schedule-2 knock-out record on the knock-in instant is disabled by that knock-in, as
MC, the PDE and V1 read it. QUAD V2 let it fire: on the repro it priced 3.712 (+11.5 SE against MC); it now prices
3.421 (+0.35 SE), equal to the same note without the record to 1e-10.

#### Changed: V1 convolutions on fast FFT lengths (owner: "approved")

15 of the 33 transform lengths in the synthetic book had a prime factor of 400 or more, so numpy ran them through
Bluestein. V1 now pads each linear convolution to the smallest even 5-smooth length at or above the old one and runs
real FFTs; the adjoint transposes on the same length. A circular convolution at any length of at least 2b-1 is the
exact linear convolution, so with the filter off this is the same operator. The default spectral filter
(`fft_filter_alpha` 12) is the same function of frequency, sampled on the finer grid, so default results agree to
round-off rather than bit for bit. Factor 1 and odd padding factors keep their lengths (`QuadratureCore` and
`DiscreteQuadEngine` are unchanged, bit for bit).

- **Speed** (back to back, separate interpreters): synthetic V1 book 3.50x (47.1 s → 13.5 s), the full
  `GreeksCalculator` set 3.30x, V1 statistics on daily-KO notes 1.34x.
- **Round-off moves, not bitwise:** book prices at most 2.45e-13 relative; statistics of size ≥ 1e-6 at most 1.7e-11;
  finite-difference greeks up to 2.9e-8 relative on rho (1 bp legs that differ by 8.7e-5; each leg moves at most
  4.4e-14).

#### Speed

Overall, back to back: synthetic V1 book `price` 3.50x faster (FFT lengths; the freeze is 1.00x with identical
checksums). Full `GreeksCalculator` set: the freeze alone is 0.93x, 0.78x with auto_converge; the FFT change more than
covers it. KO-reset V1 projection and the knocked-in staging, measured together: 1.00-1.02x (QUAD V2 staged notes
1.09x). The staging is reverted in this release; run 9's timing is in its own section.

#### Notes

- Certificates amended (QUAD candidates re-run against the carried benchmarks, every candidate ADMITTED, parents
  retired in place with `superseded_by`):
  - `ko-reset-flat-bsm/2026-11-27` (D1 projection; parent 2026-11-24), then `ko-reset-flat-bsm/2026-11-27-2` (FFT
    lengths);
  - `snowball-flat-bsm/2026-11-27` and `phoenix-flat-bsm/2026-11-27` (FFT lengths; parents 2026-11-24).

#### Open items

- **Early daily-observation KO probability in V1 statistics.** On a 1y daily-KO note at vol 20%, observation 2 reads
  4.333% against V2's 4.163%. The error is O(h²) and comes from the trapezoid rule across the KO jump on a 1.7-cell
  one-day step. An analytic jump correction improves the statistics 16-110x, but makes some default-grid prices worse.
  QUAD V2 is exact.
- Heston ADI KO-reset is -2.2 SE against MC on one investigation case at its coarse default grid (exercise on schedule
  2's last fixing, so run 9 does not move it).
- **Marginal resolved chains** (0.12.0 open item): already removed by 0.12.1's exact lattice mass, at ≤ 4.9e-6
  relative and flat in step count.
- `greek_conventions_report.build_cash_greeks_report` reprices on the raw engine, outside the bump context (every
  engine family).
- Under `auto_converge`, the frozen V1 clone does not carry the accepted rung's `local_steps_allowed`, so a bumped leg
  may take a local stencil step the base rung refused (inside the local-stencil guard).
- Phoenix V1 KO/coupon blending probe (planned D3) was not run: unmeasured.
- KO-reset V1's projected KI-narrowing weight (`_event_weight`) is inert: a same-date discrete KI forces the
  `pre_ko_last` route, whose cell-average KI blend ignores it.

### Public `knocked_in_at_valuation` on Snowball, Phoenix and KO-reset

A Snowball, Phoenix or KO-reset that is already knocked in can now be priced by setting it on the product, as a DCN
can: `SnowballOption(..., knocked_in_at_valuation=True)`, or `product.knocked_in_at_valuation = True`. Until now this
needed the private lifecycle flag `_otc_lifecycle_knocked_in`. Engines still refuse an `AutocallableLifecycleState`
passed with the product (owner choice, option A).

#### Added

- **`knocked_in_at_valuation`** (read/write property and the last constructor keyword) on `SnowballOption`,
  `PhoenixOption` and `KnockOutResetSnowballOption`. It is the public spelling of `_otc_lifecycle_knocked_in`: both
  read and write one attribute, so the lifecycle stack, the intraday twin, replay and every engine see the same flag.
  A product built with the keyword is the same object, key order and pickle bytes included, as one flagged through the
  private attribute. Prices and statistics are those of the private flag, bit for bit.
- It is not a dataclass field: `==`, `asdict` and `cache_key()` ignore it, as they ignore the private flag. A default
  product's attributes are unchanged.
- A REBASED KO-reset also needs its knock-in date, which only the private `_otc_lifecycle_knock_in_date` /
  `_otc_lifecycle_knock_in_anchor` carry; without it MC and the lifecycle tracker refuse, as before.

#### Changed

- **A flagged Snowball / Phoenix now starts knocked in wherever a lifecycle is built for it.** The flag used to be
  overwritten by a fresh not-knocked-in state, so the contract was priced as not knocked in, with no error:
  - `AutocallableLifecycleTracker` seeds its state from the flag when no state is given, as the KO-reset tracker
    already did (portfolio lifecycle manager);
  - the replay engine (single product and book) no longer hands every Snowball / Phoenix a fresh state, so its tracker
    seeds from the flag too;
  - the intraday reconstruction without a checkpoint starts from the flag.
  Both trackers refuse a flagged contract given an explicit lifecycle state that is not knocked in. The simulated-path
  backtest keeps refusing a flagged contract, as before; its message now names both spellings. Contracts without the
  flag are unchanged.
- The RFQ product registry accepts the new keyword.

#### Open items

- `==` on these products ignores the knock-in flag (as it did for the private attribute); two contracts differing only
  in it compare equal.
- Most library readers still spell the private attribute; they read the same slot.

### Replay on a business-day clock

A replay backtest can now price every day on a `BUSINESS_DAYS` trading clock. It is enabled by one opt-in object,
`AutocallableEngineConfig(trading_clock=TradingClockConfig(calendar=..., bus_days_in_year=...))`, and works for a single
product or a book. It was verified on Snowball and KO-reset (QUAD V1, PDE, MC) and on DCN (MC); the other BSM
engines run on the library's `BUSINESS_DAYS` support without a replay-clock test of their own. Without it (the default) the replay is the calendar-day replay, unchanged: the
replay goldens, env types and checksums are identical, and timing is within noise.

#### Added

- **`TradingClockConfig`** (`quantark.backtest.replay`, `quantark.backtest`) with fields `calendar`,
  `bus_days_in_year` (252), `holiday_time_adjustment` (PRECEDING) and `vol_quoting` (CALENDAR_YEAR).
  - Every env replay builds for a day sits on that clock: the day env, the maturity probe, the scenario and carry
    callbacks, the book's spot-hedge carry and the surface grid.
  - The cash curves (flat rate, implied or term dividend, basis) are declared `CALENDAR_YEAR`, and the env wraps them.
  - `trading_clock_horizon` is the later of the last market date and every product's last contract, settlement or
    payment date, plus 366 days.
- **`CalendarQuotedVolSurface` / `DateKnotTimeMap`** (`quantark.param.vol`) read a calendar-quoted vol on the trading
  axis: σ_td(u) = σ_cal(c(u))·√(c/u).
  - c(u) goes through one knot per business date (the date's env time, then its ACT/365 time). Every business date,
    Mondays and post-holiday dates included, therefore gets exactly the calendar variance.
  - The env re-anchors it.
  - `parallel_shifted`, `bump_unit_vol` and the QUAD V2 deterministic gate support it.
- **Replay's own clock types.** `ClockSignedDividendYield` / `ClockImpliedBasisYield` (`backtest/replay/market.py`)
  are the replay's flat carry types as library curves that the env can wrap; they check finiteness only.
- **Clock-path conversions.**
  - Event dates in the reports map engine times back by business-day count.
  - Pending receivables are discounted at the settlement date's env time.
  - The state row's `pricing_q` stays the calendar quote.
  - `flat_atm_remaining` reads the surface at the calendar tenor.
  - The run info records the clock.

#### Refused on the clock path

These would mix clocks silently, so each one raises with a `# TODO` scope line:
- vol models other than `bsm` (LV/Heston/SLV calibrate in calendar time);
- `trade_value`;
- `pnl_explain`;
- the simulated-path / ensemble engines;
- products with a numeric (float-time) schedule;
- a replay day that is not a business day of the clock's calendar;
- dashboard snapshot repricing of a clock run.

The MC event-statistics fallback lets a `ValidationError` through instead of turning it into an MC row.

#### Verification

- **Independent anchor.** With a calendar on which every day is a business day and `bus_days_in_year=365`, the clock
  path equals the default replay to ≤ 3.3e-13 relative, every day. This holds on QUAD V1, PDE and MC, for a single
  product and for a book.
- **Holiday window.** Over an SSE window with the 2026 New Year closure, each replay day equals a hand-built
  `BUSINESS_DAYS` env to 1e-12. This was checked on a snowball (QUAD V1, PDE, MC) and a KO-reset (QUAD V1).
- **Holiday timing.** DCN holiday timing follows `holiday_time_adjustment` (oracle: business-day counts of the
  adjusted date).
- **Vol wrapper.** The vol wrapper's total variance equals the calendar variance at every business date to 1e-14.
- **Speed.** The clock path is 1.28x the calendar-day replay on a 6-day QUAD window, from the per-day time maps.

#### Notes

- **Greek units.** On the clock path theta is per trading day; vega stays per unit calendar vol (with
  `vol_quoting=CALENDAR_YEAR` the bump moves the calendar quote).
- **PDE grid.** The PDE's default grid has `steps_per_day × bus_days_in_year` steps per env year.
- **Cash-curve lag.** On a Monday or after a holiday, the 0.15.0 cash wrappers read the calendar curve at the day after
  the previous trading day (DF to a Monday = DF_cal(Saturday)). The run measured this on a European call, strike 103 at
  3%: -0.83% of premium on a Monday expiry and -1.73% after Spring Festival. Fixed in this
  release: "Business-day cash curves read the actual date" below (owner ruling).
- **Fingerprinted trees.** This change edits `quantark/param` and `engine/quad/v2` (intraday certificate trees).

#### Open items

- **Late `trading_clock`.** Fixed in this release: the replay engine re-checks the refusals at start.
- **Unchecked hand-built wrappers.** Fixed in this release ("Business-day cash curves read the actual date", guards).

### Business-day cash curves read the actual date

On a `BUSINESS_DAYS` environment, the cash curves declared `CALENDAR_YEAR` (rate, dividend, basis) used to read the
calendar curve one or more days early after a weekend or holiday. They now read it at the date's actual calendar time.
The 0.15.0 wrappers map trading time back to calendar time through `BusinessTimeMap`, whose knots sit at the end of each
trading day. A date that follows a non-trading day therefore came back as the day after the previous trading day: the
discount factor to a Monday was the calendar discount factor to Saturday, so discount factors (at positive rates) were
overstated (owner ruling, 2026-11-27).

#### Changed

- **The environment's map is a `DateKnotTimeMap`.** It has one knot per business date at (the date's env time, its
  ACT/365 calendar time), linear between knots, and follows the environment's `holiday_time_adjustment`. Every
  declared cash curve reads the calendar curve at the actual date, as run 7's `CalendarQuotedVolSurface` does for the
  vol, so vol and cash now agree on every business date.
  - SSE, 244 days, valued Fri 2026-01-16, 3%: the discount-factor error was ≈+1.6e-4 on every Monday (+1.64e-4 the next week,
    +1.60e-4 a year out), +8.20e-4 on 02-24 (after Spring Festival), +2.45e-4 on 04-07 (after Qingming), and 0 after a trading day.
    It is now 0 on every date.
  - A European call (strike 103, 3% rate, 1% dividend) on the business-day clock now equals its calendar-day price to
    ≤ 3.7e-14 at Wednesday, Monday and post-holiday expiries. Before, it was -0.83% of premium on a Monday expiry and
    -1.73% after Spring Festival.
  - A dated one-year snowball (monthly KO, QUAD V1) valued on that environment goes from 99.365176 to 99.363132.
- **Moved:** every price, greek and statistic on a `BUSINESS_DAYS` environment with `CALENDAR_YEAR` cash curves.
  - Flows and observations on a date after a non-trading day move by the amounts above.
  - A `FOLLOWING`-placed holiday flow (DCN) is discounted at the business date it is placed on.
  - Fractional env times between two knots next to a weekend or holiday (engine sub-steps) move too: the midpoint
    between Fri 01-23 and Mon 01-26 read 7.5 calendar days and now reads 8.5.
  - Every other date moves at round-off (≤ 5.1e-15 relative): the old knots were a running sum of 1/D, the new ones
    exact.
  - Unchanged, bit for bit: calendar-day environments, `TRADING_YEAR` curves, undeclared hand-wrapped curves and every
    vol path.
- **A declaration owns the map.** A curve declared `CALENDAR_YEAR` that was hand-wrapped on a `BusinessTimeMap` is moved
  onto the environment's map at construction. Before, it moved only after a valuation-date move. Undeclared
  hand-wrapped curves stay on their own maps and keep the old lag; the docs say so. An environment holding both kinds
  refuses a relationship tenor as "not on one clock".
- **Reach.** The map reaches the first business date on or after `trading_clock_horizon`, so short horizons such as
  "last date + 1 day" keep building. A read past the last knot raises.
- **New refusal.** A `FOLLOWING` environment valued on a holiday of its calendar is refused, at construction and on any
  `valuation_date` move: the next business date would sit at u = 0 with calendar carry already accrued. Paths that can
  reach it on such an environment are a theta calendar-day bump onto a weekend, a pnl-explain t1 or scenario date on a
  non-business day, and any assignment of a holiday valuation date. `PRECEDING`, the default, is unaffected.
- **Guards (run 7 review items).**
  - `TradingClockVolSurface` refuses a `DateKnotTimeMap`, whose slope runs the other way.
  - A `DateKnotTimeMap` on a wrapper must carry the environment's `holiday_time_adjustment`.
  - `CalendarQuotedVolSurface` is refused on a non-`BUSINESS_DAYS` environment.
  - The replay re-checks the trading-clock refusals when the engine starts, since the engine config is mutable.
  - `vol_quoting=TRADING_YEAR` with `vol_source="surface"` is refused: the surface artifacts are calendar-quoted.
- `BusinessTimeMap.u_knots` is now public; carry rho-q bucket bumps read it.
- **Speed.** Building a business-day environment with declared curves takes 0.56 ms (was 0.19 ms). Pricing on it is
  unchanged (1.00–1.03x, back to back).

#### Notes

- Re-pinned tests are listed in the run report, split into lag moves and round-off moves. They include the
  environment-vs-hand-recipe pins, now on a `DateKnotTimeMap` recipe, and run 7's measurement of the lag, now an
  equality. No certificate moved.

### Knocked-in KO-reset put at the exercise date

A knocked-in KO-reset observes schedule 2's knock-outs up to schedule 2's last observation. If none fires, it is a
live put to the product's exercise date: it pays the put on the spot at the exercise date, at the exercise settlement
(an explicit `settlement_date` when booked, else exercise + settlement delay). This holds for ABSOLUTE notes, carried
knock-ins and REBASED notes (owner ruling 2026-11-27: "the final put option value should be determined by the spot at
the exercise date of the product", REBASED too). It replaces D31 (2026-11-14), under which MC read the put at
schedule 2's last fixing. Notes whose exercise date is schedule 2's last observation are unchanged, bit for bit
(pinned on every grid engine; MC on ABSOLUTE layouts).

#### Changed

- **MC reads the knocked-in put at the exercise date** (BSM, local-vol, Heston and Heston-SLV MC share the kernel).
  The exercise date is a node of every MC time grid, and the put is paid at the product terminal. Before, ABSOLUTE and
  carried knock-ins read the spot at schedule 2's last fixing and were paid there; REBASED notes read and paid at each
  path's own schedule-2 end. The cash-leg fold, EXPIRY-paid schedule-2 knock-outs, `terminal_determination` and the
  `ki_settlement` statistics row move to the terminal too. REBASED statistics now have one `ki_settlement` row in
  `termination`, at the terminal. BSM MC, 1M paths, seed 7, on the run-5 investigation notes (exercise 2026-01-05,
  schedule 2 ends 2025-12-03; SE 0.011–0.017):

  | case | before | after | move |
  |---|---|---|---|
  | base, KI 75 | 8.85270 | 8.86101 | +0.76 SE |
  | KI 97 on 07-03, `disable_ko_after_ki` | 3.40294 | 3.32660 | −4.55 SE |
  | KI 85 | 5.79305 | 5.81226 | +1.28 SE |
  | KI 85, exercise 2026-02-05 | 5.79305 | 5.81997 | +1.78 SE |
  | carried knock-in | −18.68669 | −18.59900 | +6.30 SE |
  | carried knock-in, `disable_ko_after_ki` (exact −19.86527) | −19.76689 | −19.86353 | −7.81 SE |

  REBASED, the cash-leg bank note (knock-ins end schedule 2 before exercise 2.0): 94.94324 → 93.46977 at 4096 paths,
  95.81653 → 94.42974 at 200k paths (−24 SE). This includes the accrual change below.
- **The grid engines are back on the exercise reading.** Run 5's staging at schedule 2's end is reverted on the PDE
  (1D and local vol), Heston / Heston-SLV ADI, QUAD V1 and QUAD V2 ("QUAD numerics" above). Run 5's V2
  `disable_ko_after_ki` fix, the V1 projection and the FFT lengths stay. Grid pins taken before run 5 are restored,
  e.g. the base note on the PDE 8.86368327529885 → 8.871664315881244. V1 pins that run 5 moved twice were recaptured;
  each equals its pre-run-5 value within 2.2e-15 relative.
- **Annualized knocked-in amounts accrue to the exercise date** (owner ruling 2026-11-27, "Accrue to exercise"). The
  knocked-in tenor runs from the origin to the exercise date on every engine and the lifecycle tracker. It ran to
  schedule 2's last fixing (ABSOLUTE) or to the knock-in plus schedule 2's last offset (REBASED), so the REBASED tenor
  no longer depends on the knock-in. On the investigation note with annualized knock-in the tenor goes from 0.915068 to
  1.005479 years (334 → 367 days):

  | case | PDE | QUAD V1 | QUAD V2 | MC |
  |---|---|---|---|---|
  | base, KI 75 | 9.01461 → 8.86244 | 9.01307 → 8.86075 | 9.01472 → 8.86256 | 8.99010 → 8.84208 |
  | KI 85 | 6.17256 → 5.80552 | 6.17075 → 5.80356 | 6.17271 → 5.80568 | 6.14863 → 5.79252 |
  | carried, `disable_ko_after_ki` | −18.17811 → −19.97415 | −18.17820 → −19.97425 | −18.17809 → −19.97412 | −18.09512 → −19.97204 |

  MC is 200k paths on the first two rows. On the grid engines the carried row moves by exactly the tenor ratio
  (+9.88e-2 relative); the MC moves also include the readout change above.
- **The lifecycle tracker keeps a knocked-in note alive to the exercise date.** `KOResetLifecycleTracker` books V1 on
  the exercise close's spot at the terminal payment (`resolve_terminal_timing`), not at schedule 2's last observation.
  The calendar placement, the simulated paths' maturity day and `matured_before_run` follow: a carried knock-in
  retires only after its exercise date. `with_absolute_schedule_2` (a REBASED note's ABSOLUTE copy) keeps the original
  exercise date and `settlement_date`; it used to end the copy at the rebased end. Replay and backtest results of
  knocked-in KO-resets whose exercise date follows schedule 2's end move with it. No replay, backtest or execution
  golden moved.

#### Fixed

- The KO-reset MC gap from "Same-instant observation records" above is closed. On the investigation's 12 cases
  (MC 1M paths, seed 7) every grid engine (PDE, LV-PDE, QUAD V1, QUAD V2) is within −0.53..+1.20 SE of MC.
- Referee (carried knock-in with `disable_ko_after_ki` = a European put to the exercise date, −19.86527359248963), also
  with an explicit `settlement_date` and with a settlement lag: PDE and LV-PDE within 1.47e-6 relative, QUAD V1 6.3e-6,
  QUAD V2 1.8e-15, MC within 3 SE. Heston ADI equals the closed-form Heston put to the exercise date within 7.7e-5, and
  agrees with Heston MC within 3 SE (1M paths).
- A knocked-in note valued between schedule 2's last observation and the exercise date prices as the live put to the
  exercise date: every grid engine within 5e-7 relative of the Black-Scholes put, MC within 3 SE. There is no jump at
  schedule 2's end beyond the put's time value. The run-5 open item on such notes is removed.
- EXPIRY-paid schedule-2 knock-outs are paid at the exercise settlement on every engine (PDE, Heston PDE, QUAD V1,
  QUAD V2, MC), as the intraday events and theta already paid them. The run-5 open item on this is removed.
- The QUAD V1 statistics refusal `TODO(D31)` (a carried knock-in whose schedule 2 ends at the valuation instant) is
  gone with the revert.

#### Added: refusals

- **A KO-reset whose exercise date precedes the latest possible end of schedule 2 is refused** at construction
  (`ValidationError`; owner ruling 2026-11-27, "Refuse such notes"). ABSOLUTE: the exercise date (numeric: `maturity`)
  must not precede schedule 2's last booked observation; dated notes compare full datetimes. REBASED, numeric:
  `maturity` must not precede the last knock-in observation plus schedule 2's last offset (dated REBASED notes were
  already checked). Equality is allowed. Under the new reading such a note would settle its put before the schedule
  it follows ends.
  - `create_ko_reset_snowball` builds valid REBASED notes: maturity is at least the last knock-in observation plus
    schedule 2's last offset. ABSOLUTE is unchanged.
  - The tracker's day-grid snapping moves a snapped numeric REBASED copy's maturity onto its snapped latest end when
    rounding left it short (`snapping._keep_exercise_last`). Whole-day contracts are unchanged.
  - Census: no banked, certified, golden or cash-leg baseline contract breaks the rule. The violating test contracts
    were rebuilt. The KO-reset example demo dropped its one schedule-2 date after its exercise date (2030-01-21).
- **The tracker refuses a REBASED copy whose schedule 2 would end after the exercise date** (`ValidationError`,
  `TODO(owner)`). The tracker hangs a REBASED schedule 2 from the knock-in close, and a knock-in record due on a
  non-trading day fires on the next close, so schedule 2 can end past the exercise date. The construction check above
  covers only the booked records.

#### Notes

- Certificates did not move: every certified KO-reset product has its exercise date on schedule 2's end. Full suite
  green on the run's last code commit (31106 passed, 0 failed).
- Cash-leg baselines: the two records of the MC REBASED bank note (`mc_snowball__ko_reset_rebased__-__price` and
  `__event_npv`, 94.94323994203444 → 93.4697686971873) carry the new cause (u). Its check rebuilds both readings from
  the price's own paths: the new reading equals the live price and the old one gives back the banked value, bit for
  bit. Every banked ABSOLUTE KO-reset has its exercise date on schedule 2's end. The D31 class census is replaced by
  this run's.
- `docs/event-statistics-migration.md` item 24 (D31) is marked replaced.
- **Speed.** `price()` wall time, run against the commit before this run, back to back in one window (separate
  interpreters, base and run 9 alternated). Two dated notes: exercise after schedule 2's end (the investigation's base
  note), and the same note with schedule 2's last record on the exercise date. Medians of the per-run medians:

  | engine | exercise after schedule 2's end | exercise on schedule 2's end |
  |---|---|---|
  | PDE | 0.98x | 0.99x |
  | QUAD V1 (1001 nodes) | 1.02x (1.016–1.019x in two windows) | 1.00x |
  | QUAD V2 | 1.03x (1.025–1.036x in three windows) | 1.00x |
  | MC (1M paths) | 0.99x | 0.99x |

  QUAD V2 on the first note is over the 1.02x budget in every window (per-run ratios 1.025–1.039x in the longest
  one). The revert restores the code path from before run 5's staging: against that library (d83256db) QUAD V2 on the
  same note is 0.998x, with the same price bit for bit. Prices on the second note are bitwise equal to the base on
  every engine.

#### Open items

- The tracker's REBASED refusal above. A numeric REBASED note whose maturity equals its latest schedule-2 end (as
  `create_ko_reset_snowball` builds it) refuses tracking or simulation when the last knock-in record falls on a
  non-trading day. The calendar resolution places that day eagerly, so it refuses even if no path knocks in there.
  Owner alternatives: hang REBASED schedule 2 from the record's fixing instead of the close, drop schedule-2 records
  past the exercise date, or extend the exercise date to the copy's end for this edge only.
- REBASED `post_ki_offsets`: the knocked-in (`ki_settlement`) row is still keyed at schedule 2's last offset, though
  that note now settles at the exercise date.

## [0.15.0] - 2026-11-25

A `BUSINESS_DAYS` environment knows how its cash curves are quoted. On a business-day clock every time is a
trading-year fraction, so a rate curve, dividend yield or futures basis quoted per calendar year (the usual ACT/365
desk quote) has to be read through the trading clock. The environment cannot tell a calendar quote from a trading
quote by looking at the curve, so it is now declared per curve. A calendar-quoted curve is wrapped onto the trading
clock by the environment itself, and an undeclared non-zero native curve is refused instead of being silently read
per trading year. Environments on every other convention, and `BUSINESS_DAYS` environments that declare
`TRADING_YEAR`, price bit for bit as in 0.14.0 -- except environments holding TradingClock wrappers (vol surface or
cash curves), whose base (when built with a map anchored off the valuation date), aged and stressed numbers move to
the correct values (see Changed).

### Changed

- **Breaking: an undeclared native cash curve on a `BUSINESS_DAYS` environment is refused.** A `PricingEnvironment`
  with `day_count_convention=BUSINESS_DAYS` whose `rate_curve`, `div_yield` or `basis_yield` is an unwrapped curve
  that is not exactly zero now raises `ValidationError` naming the field and both declarations. To keep 0.14.0
  numbers, declare `rate_curve_quoting=CurveQuotingBasis.TRADING_YEAR` (and `div_yield_quoting` /
  `basis_yield_quoting` for the others); the curve is then read as is. Exactly-zero curves (`FlatRateCurve(0.0)`,
  `ContinuousDividendYield(0.0)`, `NoDividend`, `FlatBasisYield(0.0)`, `ZeroBasis`, `None`) and TradingClock-wrapped
  curves need no declaration. Every `BUSINESS_DAYS` fixture in the test suite now declares `TRADING_YEAR`; no number
  moved.
- **Trading-clock wrappers follow the valuation date (owner rulings; sanctioned price move).** A
  `TradingClockVolSurface` and a hand-wrapped (undeclared) TradingClock cash curve whose map is anchored at another
  date are re-anchored at the environment's valuation date on their own horizon (`BusinessTimeMap.re_anchored`),
  both at construction and whenever `valuation_date` is reassigned; the sticky-moneyness view keeps its spot shock.
  An `IntradayTimeMap` (the intraday route's own clock) is never moved. The ~30
  library sites that age a copy by `env.valuation_date = bumped_date` (numerical theta, product theta helpers, the
  dynamic-scenario day loop, lifecycle, backtests) used to read r and q on a map still anchored at the old date.
  Size of the move, dated European (K = S = 100, sigma 20%, r 3%, q 2%, expiry 2026-12-15): on the trading-clock
  guide's default setup (calendar axis, `TradingClockVolSurface`), a one-day numerical theta starting on a weekend
  or holiday was about 20x too large -- Sat 2026-02-14 -> Sun inside Chinese New Year -0.0185443 in 0.14.0 ->
  -0.0008800 now (carry only: the trading clock is flat). In general the stale map mis-stated the variance to each
  product date X by [d0 is a trading day] - [the day before X is a trading day] trading days, so a step that starts
  on a trading day was wrong too for any date that follows a non-trading day (Monday observations, an expiry right
  after a holiday); for this European, whose expiry follows a trading day, it was right. On a hand-wrapped
  trading-axis environment, numerical theta across Chinese New Year (02-13 -> 02-24) -0.020257 -> -0.027297. Both
  now equal V(d1) - V(d0) on freshly built environments bit for bit. PDE byte book (`test/golden/pde_bitwise`, not
  run by pytest; a base-vs-branch gate): three trading-clock rows move their `bump_theta` -- `3a|tc|snowball_monthly`
  0.046856 -> 0.057386, `3a|tc|phoenix_plain` 0.019096 -> 0.034176, `3b|tc|ko_reset` 0.031928 -> 0.046963 (start
  Mon 2026-02-09; their observations follow non-trading days). A PDE branch cut from 0.15.0 has them in its base. No existing test pinned a hand-wrapped bumped or rolled number (full suite unchanged); the move is pinned by
  `test_curve_quoting_rebuild_sites.py::test_hand_wrapped_numerical_theta_equals_fresh_env_pnl_bitwise`.
- **Cash stresses and scenario moves are per calendar year on wrapped curves.** The built-in rate, key-rate, spread,
  dividend and basis stresses (`StressApplicator`) and the dynamic-scenario rate and dividend changes now move the
  calendar-quoted curve inside a TradingClock wrapper and re-expose it on the same trading clock
  (`quantark.priceenv.clock_validation.calendar_quoted_view`). Before, a wrapped rate curve was refused ("Cannot
  stress curve type"), a wrapped dividend was replaced by a native curve built from a trading-axis reading, and a
  dynamic-scenario change on a wrapped curve was silently skipped; a dynamic-scenario change the native rules cannot
  apply to the wrapped curve's inner type now raises. The `AutocallableRiskReport` q scenarios (spot x q and
  spot x vol surfaces, the q-shift ladders, historical replay) shift a wrapped dividend yield the same way, and a q
  override follows the environment's declaration. Spot, vol and user-registered stress adapters are unchanged.
- **Environment rebuilds carry the calendar.** The stress, dynamic-scenario (equity and FI) and risk-report clones
  dropped `calendar`; they now carry it and the four new fields, and deep-copy curves together with the calendar so
  a wrapper's clock stays the clone's own calendar. The risk-report clone now also carries `basis_yield`. Effects
  on clones that were not built for this: a `CALENDAR_DAYS` clone with a calendar reports event statistics on
  business-day KI rows (`knock_in_day_axis`), and an undeclared `BUSINESS_DAYS` clone resolves auto theta per
  trading day (the per-day divisor of analytical greeks follows). No price that came out before moves: those clones
  either ignored the calendar or raised without it.

### Added

- `quantark.priceenv.CurveQuotingBasis` (`CALENDAR_YEAR`, `TRADING_YEAR`) and four `PricingEnvironment` fields:
  `rate_curve_quoting`, `div_yield_quoting`, `basis_yield_quoting` (default None = undeclared) and
  `trading_clock_horizon`. On `BUSINESS_DAYS`, `CALENDAR_YEAR` makes the environment build one `BusinessTimeMap`
  (its calendar and `bus_days_in_year`, anchored at the valuation date, out to `trading_clock_horizon`) and wrap the
  curve on it -- bit for bit the hand wrapping of `docs/trading-clock.md` on every engine family; the field then
  holds the wrapper. `TRADING_YEAR` reads the curve as is and refuses a wrapper. On every other convention
  `CALENDAR_YEAR` reads the curve as is (an environment-made wrapper is unwrapped) and `TRADING_YEAR` is refused.
  `CALENDAR_YEAR` needs a calendar and a horizon strictly after the valuation date (last date priced + 1 day is
  safe); non-enum values raise `MarketDataError`.
- `quantark.param.basis.TradingClockBasisYield`: the futures-basis wrapper, cumulative-preserving like the dividend
  wrapper (`b_td(u) = b_cal(c)·c/u`, `c = to_calendar(u)`).
- `is_identically_zero()` on `RateCurve`, `DividendYield` and `BasisYield` (default False; exact `== 0.0` on the flat
  classes themselves, never on subclasses).
- `MarketDataConverter.to_pricing_environment` and the builders that forward to it accept `calendar`, the three
  `*_quoting` fields and `trading_clock_horizon`; the RFQ market-kwargs builder passes them through.
- pnlexplain treats a wrapped `basis_yield` as a clock field (re-anchored in the TIME state, compared without its
  anchor).

### Fixed

- The 0.14.0 known issue "a `BUSINESS_DAYS` environment with unwrapped calendar-quoted rate or dividend curves passes
  `validate_trading_clock_configuration`" is resolved: the environment records the quoting basis and refuses the
  undeclared case.
- Numerical theta and every date-aged copy of an environment with a `TradingClockVolSurface` or hand-wrapped cash
  curves read variance, r and q on a map anchored at the old valuation date (see Changed).

### Notes

- `quantark/priceenv` is fingerprinted by the intraday certificate trees: their next amendment re-runs rather than
  carrying forward. The four new dataclass fields change every environment's execution-cache fingerprint (cache keys
  only). An environment holding TradingClock wrappers fingerprints as uncacheable and the execution planner
  invalidates conservatively for it (performance only).
- A `CALENDAR_YEAR` environment builds its trading-clock map at construction and again on every valuation-date move
  (one pass over the days to the horizon).

### Known issues

- Basis/dividend relationship modes (`AUTO_ADJUST_BASIS`, `AUTO_ADJUST_DIVIDEND`, `SYNCHRONIZED`) are refused on an
  environment holding a TradingClock-wrapped cash curve: deriving one curve from the others across the trading clock
  needs the tenor mapped and the derived curve placed on the same clock, which is not built. `INDEPENDENT` works.
- `FxPricingEnvironment` has no quoting declarations and no trading-clock validation.
- pnlexplain still refuses the Taylor method on a trading-clock-wrapped environment; its original reason (the theta
  bump did not re-anchor the map) no longer holds.
- A greek dividend bump of a `None` dividend yield on a `CALENDAR_YEAR` environment creates a native
  `ContinuousDividendYield(shift)` that is read per trading year.

## [0.14.0] - 2026-11-25

The DCN runs on the pricing environment's clock. Every DCN time -- the diffusion grid, payment discount factors,
`get_maturity`, event statistics, receivables and folded cash legs -- is measured with the environment's day-count
convention, calendar and `bus_days_in_year` from the schedule's valuation date, the rule dated snowball records
already follow. `CALENDAR_DAYS` and `ACT_365` environments are unchanged, bit for bit; `ACT_360` and `BUSINESS_DAYS`
DCN prices and statistics move on purpose.

### Changed

- **DCN times on the environment's axis (sanctioned price move; retires the 0.11.0 "D12" rule of a DCN-own ACT/365F
  axis).** `build_dcn_grid_context(product, pricing_env)` now takes the environment (breaking for env-less callers);
  times are `quantark.priceenv.date_axis.env_time(env, date, anchor=schedule.valuation_date)`, so date-rolled theta
  and `theta_shift` (which price a schedule dated away from the env) keep their meaning. `DCNOption.get_maturity(env)`
  is on the same axis (`get_maturity()` without an env keeps the old ACT/365F value for outside callers). Coupon
  accrual (30/360 contract data) does not move. Size of the move, `SAMPLE_DCN_A` (N = 1e7) at inception, PDE: -20,678
  on an ACT_360 environment and +12,476 on an SSE `BUSINESS_DAYS` (244) environment vs its CALENDAR_DAYS price
  (`test_dcn_env_axis_gate.py`).
- **Schedule dates on one instant share one grid column.** When the environment's clock puts two DCN dates at the
  same time (a `BUSINESS_DAYS` environment whose calendar closes on an SSE trading day, e.g. 2025-07-04 on a US
  calendar; 30/360), they are one column (`DCNGridContext.column_dates`); no zero-length step is built. Their
  decisions run in the contract's same-instant order, which equals date order because KI lies strictly below the
  coupon and KO barriers. A future date the environment places at time 0 is decided at the valuation instant on
  today's spot. The KI statistics table has one row per instant, dated by its earliest KI date.
- **DCN receivables on the same axis.** Pending ledger flows (`pending_pv`, the receivables table's decision and
  payment times, and the run ledger's `PortfolioLifecycleManager` revaluation of `dcn:*` flows) are timed by
  `env_time`; a flow paid on an environment holiday right after the valuation date is discounted at DF(0) under
  `PRECEDING` instead of being refused.
- `DCNLifecycleTracker.remaining_maturity(date, pricing_env=None)` measures on the given environment's axis; the
  product replay passes its probe environment (CALENDAR_DAYS: unchanged). The tracker needs no environment
  (`dcn_row_accruals`).

### Added

- `PricingEnvironment.holiday_time_adjustment` (`BusinessDayConvention`, default `PRECEDING`): where a date that is a
  holiday of the environment's calendar sits on a `BUSINESS_DAYS` clock -- `PRECEDING` (plain business-day counting),
  `FOLLOWING`, `MODIFIED_*`. Contract dates never move, only their model time. `UNADJUSTED` and non-convention values
  raise `MarketDataError`. Ignored under calendar-day conventions; in this release only DCN times read it. Every
  field-by-field environment rebuild (stress, dynamic scenario, risk report, intraday snapshot) carries it.
- `quantark.priceenv.date_axis.env_time` / `env_times`: the signed env-axis year fraction from an anchor (past dates
  negative); under `BUSINESS_DAYS` both ends are rolled by the adjustment, so times never decrease in date order.
- DCN refuses an environment convention whose times run backwards in date order (`ValidationError`, e.g. ACT_365L
  across 2024-02-29).

### Fixed

- The 0.11.0 known issue "DCN prices and statistics discount the DCN's ACT/365F grid times on the environment curve,
  which is wrong under `BUSINESS_DAYS` or `ACT_360` environments" is resolved.

### Notes

- `quantark/priceenv` and the DCN engine directories are fingerprinted by the intraday certificate trees
  (`COMMON_TREES`, `PDE_TREES`, `MC_TREES`): their next amendment re-runs rather than carrying forward. A new
  `PricingEnvironment` dataclass field changes every environment's execution-cache fingerprint (cache keys only).
- Still open: the `PricingSession` `EVENT_STATS` operation is accepted only by `HestonDCNPDESolver` among the 8 DCN
  engines -- the 5 DCN MC engines serve statistics as a `PRICE_DETAILED` output, `DCNPDEEngine` and
  `LocalVolDCNPDEEngine` not at all (the next release); a dated
  snowball with two knock-out records on one environment instant is refused by QUAD V2 and priced differently by PDE
  and MC.

### Known issues

- A `BUSINESS_DAYS` environment with unwrapped (native, calendar-quoted) rate or dividend curves passes
  `validate_trading_clock_configuration` (it checks only the wrappers present), so r and q are integrated over
  trading-year time: correct only if the desk quotes them per trading year; otherwise wrap them
  (`TradingClockRateCurve` / `TradingClockDividendYield`, `docs/trading-clock.md`). Library-wide for every dated
  product; the DCN now follows the same convention. The environment does not record the quoting basis; a later
  release decides.

## [0.13.0] - 2026-11-24

QUAD V1 event statistics have one model, riding the price's own operator.

### Removed

- **QUAD V1 `event_stats_mode` and its `"forward_density"` route (breaking).** The opt-in forward
  transition-density statistics (provenance `("quad_v1_forward_density", "quad_v1")`) were the only statistics
  in the library not computed from the pricing operator, and on default grids they were no more accurate than the
  default (1y daily knock-out, vol 20%, second observation: truth 4.163%, default 4.333%, forward 4.062%).
  `QuadParams.event_stats_mode` is gone with it: passing it at all, `"stacked"` included, raises `TypeError` from
  the constructor, and `ValidationError` ("Unknown parameter keys") from `QuadParams.from_config` / a config file.
  KO-reset used to ignore the flag silently and now fails the same way. Prices, greeks and every default
  statistics field are unchanged, bit for bit (31 probes across snowball / Phoenix / KO-reset, discrete and
  continuous KI, KO / KI at valuation, local steps, BGK, auto_converge).
- **Cost for anyone who used it for speed.** The default route keeps one indicator row per knock-out observation,
  so its cost grows with observations x steps. Measured `calculate_event_stats` at default params, 0.12.1,
  interleaved, idle, median of 5. Two sessions: (a) the full request (the default also runs the KI path pass,
  which forward_density never offered) and (b) the KI stream excluded, so both arms do the same work:

  | contract | (a) default | (a) forward | (a) ratio | (b) default | (b) forward | (b) ratio |
  |---|---|---|---|---|---|---|
  | daily KO 1y, KI at maturity | 7.37 s | 0.16 s | 46x | 7.40 s | 0.18 s | 41x |
  | daily KO 2y, KI at maturity | 28.05 s | 0.34 s | 82x | 28.22 s | 0.37 s | 77x |
  | daily KO 1y, continuous KI | 7.73 s | 0.22 s | 36x | | | |
  | monthly KO, daily KI 1y | 1.72 s | 0.29 s | 5.8x | 1.29 s | 0.30 s | 4.3x |
  | monthly KO, daily KI 2y | 12.10 s | 1.95 s | 6.2x | | | |
  | Phoenix (no memory, instant), daily KI 1y | 2.50 s | 0.50 s | 5.0x | 2.29 s | 0.62 s | 3.7x |

  A fast route on the same model (the knock-out table from the exact adjoint forward pass) is a follow-up.

### Notes

- `quantark/asset/equity/param/` is fingerprinted by the intraday certificate trees (`QUAD_V2_TREES`,
  `PDE_TREES`, `MC_TREES`): this edit moves the schema-2 implementation digest of the
  snowball-intraday-daily-ki-bsm arms, so their next amendment re-runs rather than carrying forward. No equity
  snowball certificate identity hash changes (`event_stats_mode` was never recorded).

## [0.12.1] - 2026-11-24

QUAD V1 steps carry the exact transition mass. V1 samples each step's Gaussian kernel at the lattice nodes; by Poisson
summation the samples' trapezoid sum is the exact integral times `Theta = 1 + 2 sum_n exp(-2 pi^2 n^2 (s/h)^2)
cos(2 pi n phi)`. That alias mass multiplied the value on every step, so on schedules with many steps spanning under
~1.4 cells it compounded.

### Fixed

- **QUAD V1 exact-mass step kernel.** Every lattice-sampled step (price, event statistics, continuous-KI bridge,
  forward density, the logged adjoint; snowball, Phoenix, KO-reset) divides its prefactor by the analytic lattice sum
  (`quad_math.lattice_mass_factor`). One scalar per step, so every transpose stays exact. Where Theta rounds to
  exactly 1.0 (s/h >~ 1.38), under the Simpson rule, and on local-stencil steps, nothing changes, bit for bit.
  `SnowballQuadEngine._exact_lattice_mass = False` reproduces 0.12.0.
  - Daily knock-out with sparse knock-in at default params (the lattice is sized from knock-in dates only, so daily
    steps spanned 0.77-0.87 cells), against 32001-node references, relative to notional: 2y vol 10% +2.1e-3 ->
    -1.1e-4 (the remainder is the 1001-node lattice's ordinary O(h^2) error), 3y vol 20% +5.2e-4 -> +4.6e-6, 1y vol 5%
    +6.2e-5 -> -8.4e-6; an explicit 8001-node grid with 64 quarter-hour knock-in steps +3.7e-3 -> -1.2e-7.
  - Event statistics on those contracts lost the same mass: the outcome probabilities summed to 1 - 2.2e-3 (2y vol
    10%; the forward-density route likewise). They now partition to rounding (measured 5e-12 over ~500 steps;
    tested at <= 1e-11 on every contract above).

### Changed (moved values, owner-signed)

- Pinned: `test_quad_v1_local_step::test_resolved_gap_is_bitwise_unchanged` (knock-out and knock-in 8.8 h apart,
  vol 20%, s/h ~1.05) 98.91842991352458 -> 98.91842987745284 (-3.6e-10 relative). Owner ruling: a move far below the
  lattice's own discretization error is judged by single cause (0.12.0 reproduces with the switch off) and the exact
  step mass, since its direction against a reference cannot be measured.
- No golden, certificate anchor, cash-leg baseline or other pinned value moved (full suite).
- Unpinned book benchmark (60 synthetic snowballs whose monthly knock-out dates fall 2-6 h off the daily knock-in
  instants): 120 of 180 prices moved (every discrete-KI trade, up to 2.3e-4 relative). Against 32001-node references,
  36 of 40 trades moved toward the reference; median error 6.5e-3 -> 7.7e-4, max 1.75e-2 -> 3.9e-3. The 4 that moved
  away (trades 19, 37, 38, 56; owner-accepted) had an alias error that happened to cancel the lattice's ordinary error. Cost: 1.001x (interleaved, idle).

### Known issues

- Simpson rule: its half-frequency parity alias is damped by the spectral filter differently at each kernel site, so
  it needs a per-site filtered derivation; not corrected (measured 2.7e-7 per step at s/h 0.75 under the default
  filter). `QuadratureCore` / `DiscreteQuadEngine` (barrier, one-touch): Simpson, unfiltered; not corrected.
- Daily same-barrier knock-out steps narrower than a cell are still refused by QUAD V1 (a uniform lattice cannot hold
  the crossing layer); QUAD V2 prices them.

## [0.12.0] - 2026-11-24

One rule now decides when two numeric event times are the same instant, for every product and engine (contract
clock, spec `docs/superpowers/specs/2026-11-23-contract-clock-design.md`). Before this, QUAD V1, QUAD V2, the PDE,
Monte Carlo and the lifecycle each decided for itself, so one booking could be priced as different contracts.

QUAD V1 and QUAD V2 also price steps shorter than their lattice resolves, including the step from the valuation
instant to an observation minutes away (QUAD sub-resolution steps, spec
`docs/superpowers/specs/2026-11-23-quad-subresolution-step-design.md`). Before, V1 silently mispriced them and V2
refused them.

### Changed

- **One schedule-time generator.** Every factory (`snowball_helpers`, `phoenix_helpers`,
  `range_accrual_helpers`, 11 sites, and `RangeAccrualOption`'s `num_observations` schedule) computes evenly spaced
  times as `(i + 1) * maturity / n` through `contract_clock.schedule_times`. The old `(i + 1) / n * maturity` rounded twice and left equal calendar instants
  one ulp apart (the KO-reset factory's `11/12 * 1.0` vs `11/18 * 1.5`). Maturity-1.0 contracts are bitwise
  unchanged.
- **Canonical contract clock.** At construction (`validate`), `time_shift` and the lifecycle day-snap
  (`snap_contract`), a product's numeric event times (every observation schedule, settlement times, maturity,
  `first_schedule_settlement_time`, legacy time lists, a REBASED schedule 2's offsets) are snapped onto one float per
  instant: times closer than `Tolerance.ZERO` (1e-10 y, ~3 ms) are one instant, the last of a run, a run holding an
  exact 0.0 becomes 0.0 (a theta landing stays at the valuation instant), maturity is pinned. A lone positive time,
  however small, stays ahead (review 2026-09-21 R1). Two distinct observations of one schedule on one instant, and a
  run wider than the tolerance, are refused (`ValidationError`). Covers snowball, Phoenix, KO-reset, barrier, double
  barrier, one-touch, double one-touch, single/double sharkfin, accumulator and range accrual; dated records are
  untouched. KO-reset REBASED schedule-2 times (`t_ki + offset`) snap onto the contract's instants on every path
  (MC rows, REBASED maturity, the knocked-in copy).
- **Engines share the rule.** QUAD V1 (step merge was 1e-6 y, ~31 s), the PDE (relative 1e-9), MC grids (contract
  instants are anchors that engine-added nodes cannot displace), cash legs, theta shift, SA-CVA exposure surfaces,
  the backtest life surface and settlement support all decide time identity with `same_instant` /
  `merge_time_nodes` (new `anchors=` keyword; `None` is bitwise as before). `test/test_contract_clock_sites.py` pins
  77 time-identity sites plus 3 anchored merges. The cash-leg survival axis treats only an exact 0.0 as the
  valuation instant (was 1e-6 y).
- **QUAD V1 prices a step its lattice cannot resolve.** V1 sizes its lattice from the knock-in spacing only, so a
  knock-out minutes from a knock-in, a settlement node or two instants seconds apart left a step narrower than its
  cells. The sampled kernel then lost mass by aliasing (`2 exp(-2 pi^2 (s/h)^2)`, plus a half-frequency term under
  Simpson weights), and V1 silently mispriced (KO 0.5 / KI 0.5+gap, true 98.92: gap 53 min -> 109.33, 5 min ->
  226.05, 35 s -> 573.39). A step whose bound exceeds `QuadParams.convergence_rel_tol` is now applied by a local
  three-point stencil that carries the step's mass, mean and variance (`quad/local_step.py`), on the price, the
  stacked statistics, the Brownian-bridge correction, forward densities and the adjoint statistics pass. A lattice
  too fine for `max_adaptive_grid_points` is sized from the other steps instead of refused. Isolated steps price
  within 2.3e-8 of the same engine with the step resolved. Every contract V1 priced before prices bit for bit as
  before, except through the first step (next item).
- **QUAD V1 prices the first step exactly at the spot.** The step from the valuation instant to the first event is
  applied as a Gaussian quadrature at the spot on the surfaces just before that event, with exact limits at its
  barriers (`quad/first_step.py`), whenever it spans fewer than 13 cells or the lattice cannot resolve it. The
  lattice readout was off there: near a barrier a one-day lead missed the accuracy target in 11 of 30 measured
  configurations (worst 1.1e-3), up to 1.1% just above the old refusal threshold, and 265.92 against the PDE's 98.30
  five minutes ahead (a 14:55 proxy-close revaluation). It covers snowball, Phoenix and KO-reset, every route that
  reads the first step (price, statistics with `pv == price()`, the adjoint, `readout="transition"`,
  `event_stats_mode="forward_density"`, the spot-greeks surface), and every engine's own event semantics. The 13-cell
  switch is measured: beyond it the readout is within half the accuracy target in all 2,298 configurations.
- **QUAD V2 prices intervals its `max_nodes` mesh cannot resolve.** When the mesh sized from every interval would
  exceed `max_nodes`, the shortest intervals are left out of the sizing, the cell edges are aligned to the (at most
  two) value jumps of the event ending each of them, and their kernel is integrated exactly (`Operator(...,
  local_variance=)`). Gaps from 2e-10 y to the old ~7 min threshold price within 3.9e-8 of the resolved mesh at
  default parameters, on at most 784 nodes. Every contract V2 priced before is bit-identical. V2's first step was
  already exact.
- **What is still refused** (`UnderResolvedStepError` on V1, `ResourceLimitError` on V2, both `NumericalError`, each
  with its reason; MC, the PDE and `auto_converge` price them):
  - two observations of the same barrier (within three standard deviations of the variance between them) separated
    only by unresolved steps, whatever other events lie between them: the crossing mass between them lies in a layer narrower than a cell (measured up to 1.07e-3 in
    a knock-out probability);
  - on V1, unresolved steps whose accumulated variance exceeds 8 squared cells (a five-year daily knock-in at vol 5%);
  - on V2, more than two value jumps at the event ending an unresolved interval, or a zero-variance interval beside
    one.
  `UnderResolvedStepError` and V2's `ResourceLimitError` are exported from `quantark.asset.equity.engine.quad`.
- **QUAD V2's resource-limit refusal** names the event times bounding the shortest interval and
  `QuadV2Params.event_time_tolerance` (message only; semantics and defaults unchanged).

### Fixed

- QUAD V2 prices the factory KO-reset (`create_ko_reset_snowball(maturity_pre=1.0, maturity_post=1.5)`) at default
  parameters (it refused with a 36-billion-node resource limit).
- A booking whose KO and KI (or whose two KO-reset schedules) sit one ulp apart prices, reports event statistics,
  raises tracker events and books cash exactly as the coincident booking, on every engine; the knock-out wins the
  tie everywhere.

### Moved values

- **QUAD V1, through the exact first step (owner-signed):**
  - Certificates snowball-, phoenix- and ko-reset-flat-bsm amended and banked as 2026-11-24, every candidate
    ADMITTED; the 2026-11-15 directories are retired in place. Snowball and KO-reset QUAD pv moved by at most 3.7e-5
    relative, delta by 1.5e-2; Phoenix pv (near zero) by at most 2e-3 absolute (~1.6e-5 of notional), delta by
    2.3e-4 absolute; gamma by 2-12%. Against a 16x finer lattice the new values are closer in 30 of 36 checked cells
    (snowball `ordinary` gamma: before -0.04429, now -0.04714, fine lattice -0.04715).
  - Cash-leg baseline: QUAD V1 records move under the new D9 cause (s), checked by recomputing them with the exact
    first step off (price at most 2e-4, legs at most 5e-3 relative).
  - Cross-engine smoke allowances: QUAD V1 on `flat_ko_reset_absolute_dki` (pv 94.54254 -> 94.54341).
  - Futures-carry identity pin: residual .5491942645 -> .5612113718, delta -91.5027056047 -> -91.5114910203.
  - Book backtest single-product P&L -1414.52 -> -1357.60 (trades and days unchanged); the hard-mask NODAL event
    statistics baseline (first knock-out probability 0.35447 -> 0.32354, against 0.32356 on 4001 nodes); the
    empty-side test measures the lattice's own gap on the readout the aged contract uses.
  - Each of these reproduces its old value with the exact first step switched off.
- No other certificate anchor (the PDE cells; adi2d, intraday and local-vol studies) and no replay golden moved. MC
  values moved only on W17a collision grids, by at most 1.6e-15 relative.
- Contracts with event times 1e-10..1e-6 y apart are now two instants on QUAD V1 and the PDE (they were merged).

### Performance

- A 60-trade book of monthly snowballs (price and spot bumps) runs 1.016x its 0.11.3 time on QUAD V1. The exact first
  step's own arithmetic is ~1% of it; allocating and freeing its 0.2-8 MB matrices on every price made numpy's FFTs
  ~13% slower afterwards (macOS heap layout), so those matrices come from a small bounded pool.

### Tests

- `test_contract_clock*.py` (generator, canonical clock, products, barrier family, QUAD V1, PDE, MC, cross-engine
  twins / KO-wins-tie / distinct instants) and the site list `test/test_contract_clock_sites.py`.
- QUAD V2's near-coincident test now prices a 1e-9 y pair locally (within 1e-8 of the coincident booking on under
  20k nodes); a 1e-14 pair is one instant at construction.
- `test_quad_v1_local_step.py`, `test_quad_v1_first_step.py` and `test_quad_v2_local_interval.py`: gap and lead
  ladders (2e-10 y .. 88 h, vol 5/20/60%, spot on, near and away from the barrier) against pinned references,
  continuity of price, delta and theta across every switch, the refusals, and bitwise priority.

## [0.11.3] - 2026-11-17

### Fixed

- **`import quantark.montecarlo` failed when it was the first quantark import.**
  `montecarlo/conditional_snowball.py` imported `SnowballOption` from `quantark.asset` at module load.
  That ran `asset/__init__`, whose equity MC engines import back from the half-initialised
  `conditional_snowball`, raising `ImportError`. Importing `quantark.asset` first hid the cycle.
  `SnowballOption` is now a type-checking-only import, and the one runtime use (the product-type guard)
  imports it locally. `quantark.montecarlo` no longer loads anything under `quantark.asset`.

### Tests

- `test/montecarlo/test_montecarlo_import_order.py` imports both `montecarlo` entry points in a fresh
  interpreter and asserts the package loads nothing under `quantark.asset`.

## [0.11.2] - 2026-11-17

### Fixed

- **The rest of the 0.11.0 performance regression.** 0.11.1 removed the per-record calendar copies in
  `time_shift`; two copy costs remained on every product copy (QUAD V2 `prepare`, theta, greek bumps):
  - The calendar's identity (one string per holiday) sits inside every record's accrual terms key.
    `copy.deepcopy` never memoises a tuple it returns unchanged, so that identity was walked again for
    each distinct key. It is now a tuple subclass that a copy shares; equality and hashing are the
    tuple's, so the stale-terms checks are unchanged.
  - `AccrualPeriod`, `AccrualBasis` and `KnockInAnchor` were rebuilt field by field. A copy now shares
    their immutable members and copies the calendar once through the memo, so it stays independent of
    the original.

  One copy of a 750-record contract with a 1,350-holiday calendar: 13.0 ms (0.11.1) → 7.2 ms. A desk
  benchmark (97 trades with cash legs, full PV and greeks, four workers) is back at 0.5.0's time:
  0.5.0 63.1 s, 0.11.0 124.7 s, 0.11.1 84.2 s, 0.11.2 63.4 s. Prices and greeks are bitwise unchanged.

### Tests

- `test_ac14` (local-vol transition QUAD case study) measures its child process's own peak memory on
  Linux (`VmHWM`). `ru_maxrss` survives `execve` there, so the check reported the spawning xdist
  worker's peak (8.7–13 GB) for a computation that peaks at about 110 MB.

## [0.11.1] - 2026-11-17

### Fixed

- **Performance regression from 0.11.0 accrual periods.** `ObservationSchedule.time_shift` (theta,
  lifecycle ageing) deep-copied each record with its own memo. Every record's `AccrualPeriod`
  shares one accrual basis (with its calendar) and one terms key (with the calendar's full holiday
  identity), so a shift copied that metadata once per record, and every later `deepcopy(product)`
  (engine prepare, greek bumps) paid for all the copies. On a 750-record dated snowball with a
  1,350-holiday calendar, one shift took 1.07 s instead of 10 ms, and QUAD V2 price plus greeks
  roughly doubled. The records of a shifted schedule now share that metadata as the original's do;
  the copy stays independent of the original. Prices and greeks are bitwise unchanged.

## [0.11.0] - 2026-11-14

> **Version numbering.** This release follows 0.5.5. It is numbered 0.11.0, not 0.6.0, because
> 0.6.0 to 0.10.0 are the labels of the unpublished milestones shipped inside 0.5.5, and code
> comments, tests and the README name behaviour changes by them.

Unified event statistics (design: `docs/superpowers/specs/2026-11-11-unified-event-statistics-design.md`).
One result type, `EventStatistics`, replaces the five event-statistics types for Snowball, Phoenix,
KO-reset Snowball and DCN, with one definition per field on every engine (MC, QUAD V1, QUAD V2,
PDE 1D flat and local vol, Heston/SLV ADI, DCN PDE). API:
`quantark/asset/equity/engine/docs/autocallable_event_stats_api.md`. Migration, field by field:
`docs/event-statistics-migration.md`.

### Breaking

- `AutocallableEventStats`, `PhoenixEventStats`, `KOResetEventStats`, `DCNEventStats` and the path
  analyzer's `RiskNeutralSnowballEventStats` are removed, with no aliases. `calculate_event_stats`
  returns `Optional[EventStatistics]` on every engine.
- `EventDistribution.from_autocallable_stats` and `from_dcn_stats` are removed;
  `EventDistribution.from_statistics` is the one converter. `EventDistribution` is keyword-only and
  has one axis per stream: `event_times` / `event_dates` become `stream_times` / `stream_dates`,
  the survival curve gets its own `survival_times` axis starting at 0, and `mc_ko_times` is gone.
- Removed fields with a definition change: `ki_ever_probability` (counted a breach after a
  knock-out; the replacement `knock_in.probability` is P(KI before any KO)), the engine-dependent
  `ki_probability`, and `ki_event_probability` / `ki_survival_probability` (now first KI before any
  KO). `ki_survive_knocked_in_probability` is `ki_settlement.probability`.
- KO cash excludes coupons: a Phoenix coupon and released memory arrears paid with a knock-out move
  from KO cash to `coupons.discounted_cash`.
- `price_with_events(...).npv == price()` bitwise on every engine. MC trade PVs move by MC noise to
  the `price()` value; Heston ADI by 1–2 ULP.
- **KO-reset validity: the second schedule ends last (D35, owner ruling 2026-11-14).**
  `KnockOutResetSnowballOption` raises `ValidationError` at construction, on the numeric and the
  dated clock, when the second schedule's last date falls before the first schedule's last date.
  ABSOLUTE: checked on the booked dates (times). REBASED: the second schedule is placed from the
  knock-in, so the earliest possible knock-in -- the first knock-in observation (a REBASED contract
  monitors its knock-in discretely) -- plus the second schedule's last offset (dated: calendar months,
  `add_months`) must not fall before the first schedule's last date. Equality is allowed; the message
  names the dates. Such a contract left a knock-in with no second schedule in force, which no rule
  covered. A valid REBASED note always converts to a valid ABSOLUTE copy (`with_absolute_schedule_2`,
  the lifecycle tracker; on the tracker's day grid a boundary note booked on off-day times keeps its
  last offset on the next day when rounding would put it one day short). No banked cash-leg case,
  golden, certified case, cross-engine matrix contract or example is affected (census); test
  contracts of that shape were rebuilt or retired. Every valid contract's `price()` is bitwise
  unchanged.
- Replay frames keep their column names; `ki_probability_to_maturity` is now
  `ki_settlement.probability` and `expected_discounted_maturity_cashflow` is
  `maturity.discounted_cash + ki_settlement.discounted_cash`. The KI row the replay used to invent at
  the maturity date is gone. The risk report's "P(KI before maturity)" becomes "P(KI before any KO)"
  plus "P(matured knocked in)".

### Added

- `EventStatistics` with `ObservationTable` (KO/coupon grid, KO-reset `pre`/`post` tags),
  `CouponTable`, `OutcomeRow` (`maturity`, `ki_settlement`), `KnockInStatistics` (P(KI before any
  KO), expected first KI time, expected loss), a per-date `KnockInTable` on the product's KI
  schedule (continuous KI: one row per trading day, interpolated rows labelled), `ReceivablesTable`,
  `CashLedger`, `expected_lifetime`, totals, `provenance` (required), `unavailable` with
  `require(path)` / `flag(path)` and `StatisticUnavailable`, and MC `standard_errors` from
  independent batches.
- Forward (adjoint) passes on PDE 1D, Heston/SLV ADI, DCN PDE, QUAD V2 and QUAD V1, the exact
  transpose of each engine's backward step, for the KI table, P(KI) and the expected KI time. The
  price grid is unchanged; a request without the KI stream skips the pass.
- REBASED KO-reset statistics on MC: `post_ki_offsets` and an absolute `termination` distribution
  on a deterministic candidate axis. `EventDistribution.from_statistics` converts them (it raised
  `NotImplementedError`) and the cash legs value them.
- `calculate_event_stats` on all eight DCN engines, with `lifecycle_state`; receivables-only
  statistics for a terminated note with unpaid cash, `None` when settled, and
  `EventDistribution.ended()` / `ensure_live()` for event-dependent readers. QUAD V2 follows the
  same rule (it raised on terminated states).
- `EventDistribution.probability_residual`, `rebased`, `terminated`, `determination_time(event)`.
- `EventDistribution.survival_after(t)`: survival just after `t`, less the maturity-type terminations
  determined at `t` (a note settled at `t` does not enter a period starting at `t`); ENTER-basis
  accrual legs read it.
- `AccrualLeg(survival_until=SurvivalAnchor.PAYMENT)` (D32, opt-in): a terminated note keeps accruing
  until its cash is actually paid. Every termination -- knock-out, never-knocked-in maturity,
  knocked-in settlement, each REBASED path's own settlement -- is alive until its own payment time
  and gone after it. `EventDistribution.survival_at(t, anchor=...)` / `survival_after(t, anchor=...)`
  read it from the per-stream probabilities and payment times, so it works on every engine and on the
  folded route; it is a step and never revives a paid note. A distribution without a termination
  stream's payment time (path-dependent or absent) raises `ValidationError`. `LegCashBook` truncates a
  knock-out at its settlement time for such a leg. The default `SurvivalAnchor.DETERMINATION` is D29
  unchanged: every price and every cash-leg bank record is bitwise identical. `SurvivalAnchor` is
  exported from `quantark.cashleg`. Under EXPIRY coupon pay a knock-out is paid at maturity, so a
  PAYMENT leg (`TRUNCATE_AT_KO` included) accrues to maturity on knock-out paths.
- `knock_in_was_carried(product, lifecycle_state=None)` (`quantark.cashleg.event_distribution`): the
  carried-KI flag from the product's lifecycle marker or the state's `knocked_in`, feeding
  `from_statistics(..., knock_in_carried=...)`.
- `AutocallablePathAnalyzer.analyze_snowball_risk_neutral` returns `EventStatistics`.
- `KnockOutResetSnowballOption(first_schedule_settlement_time=...)` (numeric clock only): the explicit
  payment time of a note that ends with its first schedule (never knocked in; under EXPIRY also a
  first-schedule knock-out), the numeric counterpart of an explicit `settlement_date` (D33, D34). It
  ages with `time_shift`, and enters `cache_key()` and equality only when set. `validate()` refuses
  it before the first schedule's maturity, and refuses it on a note whose first schedule runs to the
  product maturity unless it equals the terminal payment time (one maturity, one payment). `ContractTimeline.first_schedule_payment`
  is the intraday instant it is resolved from.

- `CarryExtrapolation.FLAT_CARRY_YIELD`: a third carry long-end rule that holds the zero carry yield `B(T)/T` at the
  last node's value (flat `q` under a flat rate curve). `CarryImpliedDividendYield.parallel_shifted` shifts nodes
  exactly under it; the legacy `ForwardCarryCurve` rejects it. Existing rules and defaults are unchanged.
- `calculate_carry_curve_buckets` (`quantark.asset.equity.riskmeasures`, with `CarryCurveBucketReport` /
  `CarryPillarBucket`): pillar-bucket risk on any `CarryCurve` — `∂PV/∂F_i` at fixed spot with optional hedge lots
  per pillar, spot delta (nodes held), pinned-forward delta (forwards held), rho_q per node and parallel, and the
  chain-rule / rho-sum identity residuals. Every price goes through `resolve_bump_engine`. Case study:
  `example/case_studies/snowball_2y_carry_curve/` (2Y CSI 1000 snowball, flat q vs the IM + MO carry curve).

- Curve construction (`quantark.param.curve`, re-exported from `quantark.param`): `ZeroCurve` and `CarryCurve` on one
  shared term interpolator with five schemes (`CurveInterpolation.LINEAR_ZERO`, `LOG_LINEAR_DF`, `CUBIC_ZERO`,
  `LOG_CUBIC_DF`, Hagan–West `MONOTONE_CONVEX`) and the existing `RateExtrapolation` / `CarryExtrapolation` long-end
  rules; instrument quotes (`DepositQuote`, `SwapQuote`, `BondQuote`, `FuturesQuote`, `OptionPairQuote`) in numeric
  (`tenor_years`) or dated (dates + `valuation_date`, calendar ACT/365F) mode, never mixed; an iterative sequential
  bootstrap — `bootstrap_discount_curve` (deposits, par swaps, bonds), `bootstrap_projection_curve` (dual-curve, one
  `RateIndex` off a given discount curve), `build_carry_curve` (index futures or call/put pairs, with a live
  `to_dividend_yield(rate_curve)` view) — and `reprice_quotes` for the round-trip residuals. Dated swaps and bonds are
  built as the repo's own `InterestRateSwap` / `FixedBond`, so bootstrapped curves reprice through
  `IRSDiscountEngine` and `BondDiscountEngine` by construction; a QuantLib cross-check (skipped when QuantLib is
  absent) agrees within 1e-10. `key_rate_bumped_zero_curve` uses `with_pillars` when the curve has it. Guide:
  `docs/curve-construction.md`; demo: `example/param/curve_construction_demo.py`.
- Pillar-curve protocol for the risk stack: `quantark.param.rrf.is_pillar_curve` (`pillars` + `with_pillars`) and
  `is_flat_rate_curve`; `InterpolatedRateCurve.with_pillars`. The equity/FX/credit stress applicators, FX VaR
  revaluation, SA-CVA key-rate IR delta, as-of roll-down and term-structure drift, PnL-explain bucketing and the
  execution greeks fingerprints accept a `ZeroCurve` (or any pillar curve) where they used to require
  `InterpolatedRateCurve`; a stressed or bumped curve keeps its interpolation, extrapolation and clock anchor.
- `simple_forward_rate(curve, t_start, t_end, dcf=None)` (`quantark.param.rrf.projection`): the simple forward over a
  window on a rate curve, with the running-period (`t_start < 0`) rule; the single projection call site for the rate
  engines, the FRN and the bootstrap strips.
- Trade-value portfolios: `EquityPortfolio(trade_value=True)` values every `EquityPosition` at its full trade
  value (product × signed quantity + cash legs), with bump Greeks from `get_trade_risk` and PnL against the new
  `EquityPosition.entry_trade_value`. VaR (historical, Monte Carlo incl. attribution, parametric), SIMM via the
  portfolio adapter, static stress, dynamic scenarios and backtests follow the flag through one shared switch
  (`quantark.portfolio.equity.valuation`). Default off: the product path is bitwise unchanged.
- Multi-day cash-leg booking: `quantark.cashleg.booking.LegCashBook` rolls a position's legs with the product and
  books their flows as paid cash and receivables by explicit per-leg rules on knock-out and maturity. The equity
  backtest (`BacktestConfig.trade_value`), dynamic scenarios and the product-replay engine (`trade_value`,
  per-trade `cash_legs`) use it; leg cash is part of `paid_cash` / `pending_receivable_pv`. Leg times count from
  the contract's `initial_date` (else the run start); unruled leg variants raise `CapabilityError`.
- `time_shift` on `DeterministicLeg`, `AccrualLeg` and `FixedPayoffLeg` (shared landing rule
  `quantark.cashleg.base.roll_payment_time`); PnL explain's `cash_legs` line (`Factor.CASH_LEGS`).

- `ConvertibleBondTFTreeEngine`: the Tsiveriotis-Fernandes model on a CRR binomial tree, a lattice
  twin of the TF PDE. Each node carries the total value and its cash-only component (COCB), which
  discounts at `r + credit_spread`. Time steps are equal in total implied variance, so a vol term
  structure keeps the tree recombining; flat vol gives equal time steps. Select it with
  `ConvertibleBondEngine(env, method=EngineType.TREE(ConvertibleBondMethod.TF))`;
  `EngineType.PDE(TF)`, `ConvertibleBondMethod.TF` and `"tf"` still mean the TF PDE.
  `ConvertibleBondResult.engine_type` says which engine produced a price. Against the TF PDE at
  800×2000 on a widened domain the tree agrees to 0.05 per 100 face on seven cases (puts, soft
  call, term-structure rates and vol).
- `ConvertibleBond.final_coupon_on_conversion` (`FinalCouponOnConversion.KEEP` or `FORFEIT`): whether
  a holder converting at maturity still receives the final coupon. `KEEP`, the default, is unchanged
  (maturity value `max(parity, face) + coupon`; prices byte-identical on all five engines). `FORFEIT`
  gives `max(parity, face + coupon)`, the QuantLib convention: GS, HW, jump-diffusion, the TF PDE and
  the TF tree fold the final coupon into `ConvertibleBond.maturity_redemption`, and
  `ConvertibleBond.coupon_amount` drops it from the in-step coupons (HW therefore survival-weights
  the folded coupon over the last step, about `coupon × hazard × dt`). The setting also accepts
  `"keep"` / `"forfeit"`, and rejects anything else. Against QuantLib 1.44's
  `BinomialConvertibleEngine<CRR>` on the AC7 cases at 2000 steps this convention is the largest
  piece of the GS–QuantLib gap (up to 0.79 per 100 face, on the demo bond); with `FORFEIT` the GS
  tree lands within 0.02 of QuantLib at 500 steps.

- Mid-life valuation of discrete FX barriers and sharkfins: `observation_dates` (the date-clock form of
  `observation_times`) and `observed_fixings` (`{observation date or time: realised spot}`). A recorded breach
  decides the contract (a knock-out keeps only a rebate not yet paid; a knock-in is the vanilla); otherwise only the
  fixings ahead are simulated. A past fixing without a record is decided at the current spot with one
  `MissingFixingWarning`; theta records the fixings it steps over (`record_passed_fixings`), bump Greeks freeze the
  base fill (`with_missing_fixings_filled`).
- Accumulator `past_observations` as `(date, spot)` on a dated contract (a list mixing dates and times is refused).
- `quantark.util.calendar.add_months(date, n, calendar=None, convention=None)`: the calendar's "+nM" (day of month
  kept, clamped at the month end, optional business-day adjustment).

### Changed

#### Prices (eight sanctioned moves; every other `price()` is bitwise unchanged)

- **KO-reset with EXPIRY coupon pay: a first-schedule knock-out pays at the first schedule's
  maturity** (PDE 1D flat and local vol, Heston/SLV ADI, QUAD V1, QUAD V2). See Fixed below for the
  rule and the moves in price units.
- **KO-reset with an explicit `settlement_date`: a note that ends with the first schedule is paid on
  the settlement date** (PDE 1D flat and local vol, Heston/SLV ADI, QUAD V1, QUAD V2). See Fixed
  below for the rule and the moves in price units.
- **KO-reset intraday prices of a dated note with an explicit `settlement_date`: the float-time twin
  carries the first schedule's payment** (intraday PDE and QUAD V2 routes). See Fixed below for the
  rule and the moves in price units.
- **KO-reset Monte Carlo: the knocked-in payoff reads the spot at the second schedule's end**
  (BSM, local-vol, Heston and Heston-SLV MC). See Fixed below for the rule and the moves in price
  units.

- **QUAD V2 memory Phoenix with `initial_coupon_arrears > 0`.** V2 ignored arrears owed at the
  pricing date while QUAD V1, PDE and MC pay them. V2 now releases them at the first remaining
  coupon observation (with a coupon hit or a knock-out there, else carried). The V2 price rises by
  about the arrears times their discount factor: +1.00 to +1.49 on the test contracts (spot 100,
  arrears 1.5). V2 now agrees with V1 (3.3e-6 at 16001 nodes) and with MC (within
  2.6 SE over 8 seeds). Contracts without arrears are bitwise unchanged. V2 theta on a missed
  first coupon without a lifecycle count moves with it (−1.77 to +0.033, in line with V1/PDE/MC).
- **PDE knocked-in edge row, annualized knock-in** (PDE 1D flat and local vol, Heston/SLV ADI).
  The far-edge row of the knocked-in surface (lower edge; upper on a reverse note; the max-loss
  row) omitted the annualized-KI tenor multiplier that the maturity payoff applies (Snowball
  `is_annualized_ki`, KO-reset post-KI tenor). The row now reads slope and intercept from
  `get_maturity_payoff_v1`. Moves only annualized-KI contracts whose edge is reachable. Local-vol
  KO-reset on the default grid: 96.16 → 94.10, confirmed by an independent CTMC QUAD (94.10). The
  width-3 vs width-16 gap falls from 0.14 to 3e-4 (flat), 2.3–3.5 to ≤ 4.5e-3 (local vol) and
  0.7–1.5 to ≤ 1.6e-3 (ADI); local-vol PDE and ADI agree with MC within 2.4 SE. No certified anchor
  moved.
- **PDE knocked-in edge row, airbag with partial protection** (PDE 1D flat and local vol,
  Heston/SLV ADI; Snowball, reverse, Phoenix, KO-reset). The closed-form max-loss edge row ignored
  the airbag. Contracts with PARTIAL protection and an airbag that changes the knocked-in payoff on
  the edge cell now use the payoff-read row; NONE and FULL protection, and partial contracts where
  the airbag does not change the edge, are bitwise unchanged. The width-3 vs width-16 gap falls
  from 2.9e-2–9.2e-2 to ≤ 9.3e-4 (flat), 1.65–1.76 to ≤ 1.4e-2 (local vol; 3.7e-2 on annualized KO-reset, identical with protection NONE), up to 0.63 to
  ≤ 2.3e-3 (Heston ADI) and up to 1.07 to ≤ 4.7e-3 (SLV ADI). Against MC these contracts sat 24 to
  127 SE low; now within 2.9 SE. No certified anchor moved.
- **PDE knocked-in edge row, partial protection without an airbag** (PDE 1D flat and local vol,
  Heston/SLV ADI; Snowball, reverse, Phoenix, KO-reset). See Fixed below for the rule and the moves
  in price units.

#### Statistics

- KI is never triggered after a knock-out, on every engine, and a knock-out wins a same-instant tie
  with a discrete KI. Outcomes partition the paths: KO rows, `maturity` (never KO, never KI) and
  `ki_settlement` (KI, no KO), each computed directly; the maturity is no longer `1 − ΣKO − ki`.
- `probability_residual` and `reconciliation_error` are computed by the type and always reported,
  never absorbed. `from_statistics` absorbs the probability residual into MATURITY_NO_KO and
  carries it as `EventDistribution.probability_residual`.
- `EventStatistics.pv` is `price()`'s value on every engine; MC statistics are recorded on
  `price()`'s own paths.
- The KI stream lives on the KI table's own axis (payment = KI observation; legs add their offset),
  so it is never dropped; AT_KI legs that used to raise now value.
- `cash_to_pay` is computed directly on path-dependent rows (EXPIRY coupons, memory arrears,
  REBASED) instead of discounted / DF.
- EXPIRY coupon rows are path-dependent on every engine.
- Cost of a statistics request against the pre-migration `calculate_event_stats` on the same
  engine: a full request ≤ 2×; a request without the KI streams ≤ 1.10× (QUAD V1 ≤ 1.5×, its direct
  maturity column is never pruned). Accepted exception: PDE 1D memory Phoenix under EXPIRY pay,
  pruned requests up to 1.15× (measured 1.12–1.14×), because its coupon-cash columns cannot be
  pruned. Second accepted exception: QUAD V2 with `backend="fft"` forced on a contract whose auto
  backend is direct, pruned requests up to 1.41× (full 1.88×; snowball, continuous KI, ~9 ms),
  because the FFT now runs each statistics channel on its own so that `pv` equals `price()` bitwise on
  every architecture (the batched einsum was 1 ULP off on x86_64). Routes that choose FFT on auto
  (daily KI) are faster than before (full 0.91×, pruned 0.74×). DCN had no statistics before; its
  full request is ≤ 2× its `price()`.
- MC `calculate_event_stats` on an expired note returns `None` (was `ValueError`).
- QUAD V1 `event_stats_mode="forward_density"` reports no KI path statistics and raises
  `StatisticsRefused` for memory or EXPIRY Phoenix (it returned empty coupon cash).
- KO-reset ABSOLUTE on MC: the never-knocked-in maturity is determined at the first schedule's end
  and leaves the survival there; probabilities unchanged.
- Settled notes leave the survival at their DETERMINATION time (D29). A knock-out leaves at its
  observation row. A maturity-type termination (never-knocked-in maturity, knocked-in settlement,
  each REBASED path's own settlement) is alive at its determination time and gone right after it,
  even while its cash is still unpaid. `survival_at` steps there, interpolation between nodes never
  revives a settled note, and past the last node a settled note stays gone. Cash legs move with it:
  - accrual legs pay nothing past a settled note's end (aged DCN, quarterly ENTER leg running to
    2.0: MC 6598.93 → 3453.33);
  - a KO-reset ABSOLUTE COMPLETE_PERIOD leg now pays its period ending at the first schedule's
    maturity on the notes that mature there;
  - a COMPLETE_PERIOD leg whose last period ends on a settlement date after the determination pays
    nothing on the matured mass (see Known issues).
- A carried knock-in (R1): `EventDistribution.from_statistics(stats, knock_in_carried=True)` gives
  an empty KI stream and `knock_in_carried=True`, so an AT_KI leg on an already-knocked-in note
  values 0. The knock-in is a past event.

#### Other

- **Floating-rate projection convention** (IRS `FloatingLeg`, `FloatingRateBond`, `FRAEngine`, `CapFloorEngine`,
  the `SwaptionEngine` zero-annuity fallback). A projected coupon is now the simple forward on the projection curve
  over the window the index covers, in the leg's own day count: `L = (P(t_a)/P(t_b) − 1)/dcf` over the accrual period
  for in-advance and overnight-compounded flows, over `[t_fix, t_fix + tenor]` for a term index reset in arrears (no
  convexity adjustment). A running period uses the pro-rata spot rate `(1/P(t_e) − 1)/(dcf·t_e/(t_e − t_s))`; a
  still-payable coupon whose accrual has fully elapsed with no stored fixing raises `MarketDataError` instead of being
  valued at today's rate; the `t1 < 0 → 0` clamps in the FRA and cap/floor engines are gone. The previous code plugged
  the continuously-compounded forward over `[t_fix, t_fix + tenor]` (overnight: the zero rate at the fixing date) into
  `N·L·dcf`. **Values move:** on a term index quoted on the axis day count (ACT/365) by the compounding convexity
  `≈ f²τ/2` (≈ 1 bp at 3 %/3M); on an ACT/360 index the day-count basis term `≈ −f·5/365` dominates (−3.8 bp at
  5 %/3M: +3.1 bp convexity, −6.9 bp basis); on an overnight-compounded leg the move is first-order in the curve slope
  (≈ 200 bp on a 10Y quarterly coupon with `z(t) = 2 % + 0.2 %·t`). Equity, FX, credit and bond-discount outputs and
  all goldens have no floating projection and are bitwise unchanged; two rate tests that encoded the shortcut were
  reconciled to the exact formulas.
- Cash-leg times count from the trade's `initial_date` in every consumer, not only in multi-day runs. Point-in-time
  trade valuation (`EquityPosition.get_trade_value` / `get_trade_value_breakdown` / `get_trade_greeks` /
  `get_trade_risk`, hence trade-value portfolios, static stress, VaR and SIMM) rolls the legs from the initial date to
  the valuation date with the booking's own rule (`quantark.cashleg.legs_seen_from`), so an aged trade has one day-0
  value across the static tools, backtests, dynamic scenarios and replay. New `EquityPosition.cash_legs_clock`
  (`LegClock.CONTRACT` default, `LegClock.VALUATION` for legs already rolled) and `on_valuation_clock(env)`.
  **Rebooking needed:** an aged trade (valuation after `initial_date`) whose legs were booked with times counted from
  the valuation date now reprices — rebook its leg times from the initial date, or set
  `cash_legs_clock=LegClock.VALUATION` on it. Trades valued on their initial date, and products without
  `initial_date`, are unchanged.
- **The date / numeric clock rule** (every equity and FX option; DCN, TARF and TARN keep their own, already
  compliant, rules). A product is on the DATE clock (every time a date: `exercise_date` / `maturity_date`, FX
  `expiry_date`) or the NUMERIC clock (every time a float, in years from the day it is booked on), never a mix.
  Both forms complete: the dates win and the floats are dropped at construction (`maturity` beside a date,
  `maturity_date` supplying `exercise_date`, FX `delivery` beside `delivery_date`), so `maturity` + `exercise_date`
  and `maturity` + `expiry_date` are no longer refused. Partial or mixed input is refused with a `ValidationError`
  naming the field: a date field on a numeric contract (`initial_date`, `settlement_date`, `maturity_date`, dated
  records, FX `delivery_date`), or a float a dated contract needs with no date for it (a record with only
  `observation_time`, legacy float observation lists, FX float `observation_times`, FX `delivery` alone, float
  accumulator `past_observations`). Durations and counts stay clock-neutral (`tenor`, settlement lags in any unit,
  accrual factors, schedule generators, REBASED offsets). Detection is `quantark.asset.clock.clock_violations`;
  enforcement `enforce_clock`, first in every validate. Complete-clock contracts are bitwise unchanged.
  This replaces the numeric-note-from-`initial_date` valuation of 2026-10-24: `age_note_to_valuation_date` is gone,
  `EquityPosition.on_valuation_clock` only rolls a dated trade's `CONTRACT`-clock cash legs (it never changes the
  product), and the PnL explain snapshot, trade theta split and SIMM vega tenor read the note as booked.
  **Migration:** a numeric note with an `initial_date` no longer constructs. Book it either dated (give
  `exercise_date` and dated records; keep `initial_date`), or numeric as seen from the day you book it: a note
  struck N days ago is the original contract aged by `product.time_shift(N / 365, valuation_date, env,
  keep_landing=True)` (records already past dropped, the elapsed time banked in `accrued_offset`), the form the
  lifecycle trackers' copies already have. On the numeric clock the tracker's origin is the run's `start_date`.
- A REBASED KO-reset on the date clock places schedule 2 in calendar months: its offsets (booked as `k / 12` years)
  mean `k` months, and each knock-in record date generates schedule 2 as `add_months(date, k)`
  (`rebased_extended_schedule`); the latest end is the `exercise_date` (derived on a `maturity_date` booking; a
  given one may not precede it). Monte Carlo, the lifecycle tracker and the knocked-in conversion read the
  generated dates. **Reprices:** a dated REBASED contract used to read its offsets as continuous years from the
  knock-in time (`t_ki + k / 12`); it now lands on calendar months. Offsets that are not whole months, and
  schedule-2 settlement offsets (no date form), are refused on a dated contract. Numeric REBASED is unchanged.
- A DISCRETE FX barrier / sharkfin is observed only on its fixings: a spot beyond the barrier on a day without a
  fixing no longer knocks it (the engines' inception shortcut used to apply the continuous rule). Continuous
  monitoring is unchanged.
- Asian `observation_dates` (legacy schedule) is now read: it resolves like a record's date and wins over
  `observation_times`. It was validated but ignored (a uniform schedule was generated instead).
- Static equity stress follows `EquityPortfolio.trade_value` (default product mode): a product-mode portfolio is
  valued with `get_market_value` / `get_greeks` again; `native_batch` runs on trade-value portfolios only. New
  result metadata `value_basis`.
- Trade theta now carries the time value of deterministic, accrual and fixed-payoff cash legs (they roll with
  the product; a payment inside the bump is gone at t1, as for other engine-determined cash).
- `EventDistribution.survival_at` at the valuation instant reads the post-observation survival of an event kept
  at 0 (a theta-shifted or knocked-out-at-valuation contract) instead of 1; an `AutocallableCashLeg` keeps its
  terminal branch after its last observation while the terminal settlement is ahead.
- `ConvertibleBond.hazard_rate` now defaults to `None`, which derives the intensity from the credit
  triangle: `ConvertibleBond.effective_hazard_rate = credit_spread / (1 − recovery_rate)` (new
  `quantark.asset.credit.conventions.spread_to_hazard`). Jump-diffusion, the HW trinomial and their
  CS01 read it, so a bond priced on GS/TF (which read `credit_spread`) and on the hazard engines
  carries the same credit. An explicit `hazard_rate`, zero included, still wins, and those prices
  are unchanged; a bond that set a spread but no hazard used to price riskless on the hazard
  engines and now prices its spread. With no hazard and `recovery_rate = 1` the spread does not
  determine the intensity, so the hazard engines raise. The convertible demos drop their
  hand-picked 1% hazard: on the demo bond HW and jump-diffusion move from 138.74 to 136.64–136.65,
  and the five engines now span 136.37–136.80 (they were 2.4 apart).
- On default, the jump-diffusion and HW trinomial convertible engines (all three vol schemes) now
  pay the holder `max(conversion_ratio × (1 − η) × S, recovery_rate × face)` while conversion is
  open: the better of the recovery and conversion into the shares left after the stock drops by
  `stock_jump_on_default` (η), as in Ayache, Forsyth & Vetzal (2003, eq. 4.9). They used to pay the
  recovery alone, so a bond with η < 1 threw away conversion into surviving shares. Jump-diffusion's
  conversion probability counts conversion on default (HW's stays conditional on survival). At
  η = 1 (Hull, Andersen–Buffum: the stock goes to zero) or with conversion closed the payoff is the
  recovery as before, and prices are byte-identical. The product default is η = 0.4, so every
  bond built with default settings, a hazard rate and open conversion reprices upward. The HW
  `recovery_component` stays the recovery floor (default probability × R × face); conversion on
  default lands in `bond_component`. At η = 0.4 with hazard `credit_spread / (1 − R)` the demo bond rises 4.48 per 100 face (132.64 → 137.12; the gap to η = 1 shrinks from 8.1 to
  3.6), and the AC7 cases rise 0.09–4.52. An American zero-coupon convertible (q = 0, where early
  conversion is never optimal) matches its exact value to 0.005 in jump-diffusion and 0.01 in HW.

### Fixed

- Phoenix MC coupon cash: memory arrears released by a knock-out on a missed coupon barrier are
  counted, and EXPIRY coupons are discounted at the actual knock-out or maturity settlement.
- QUAD V1 Phoenix: memory coupon cash is computed (it was missing) and EXPIRY coupons are
  discounted at the knock-out settlement.
- MC exact-affine routes reported KI from a biased Z = 0 slice (off by about 0.06); they now use the
  exact conditional outcome probabilities.
- PDE statistics labelled the not-knocked-in far edge as knocked-in settlement after the last
  discrete KI observation, so P(KI settlement) could exceed P(KI); edge rows now follow the price's
  own edge-row rule.
- **KO-reset with EXPIRY coupon pay: a first-schedule knock-out pays at the first schedule's
  maturity** (owner ruling 2026-11-14). A knock-out on schedule 1 happens only on a path never
  knocked in, and that note ends at schedule 1's maturity, so EXPIRY pay means that date plus the
  terminal settlement delay -- the never-knocked-in maturity's own payment -- not the product
  (schedule-2) maturity. Monte Carlo already paid it there; PDE 1D (flat and local vol), Heston/SLV
  ADI, QUAD V1, QUAD V2 and the KO-reset fold structure now do too, including a schedule-1
  knock-out decided at valuation. Schedule-2 knock-outs, INSTANT pay, carried knock-ins and
  contracts whose schedule 1 runs to the product maturity are bitwise unchanged; probabilities and
  cash amounts are unchanged, only the payment time and its discount factor move (statistics'
  schedule-1 `payment_times` move with the price). Moves on the default engines, with a Monte Carlo
  reference of 8 × 65,536 paths:
  - matrix contract (2-year, schedule 1 to 1 year, discrete KI): PDE / V1 / V2 91.34 → 93.29
    (+1.957); MC 93.288 ± 0.033, from about −59 SE to within 0.3 SE. Local-vol PDE +2.13, Heston
    ADI +2.04;
  - 2-year monthly KO-reset (schedule 1 months 3–12 at KO 100 / 20% p.a., continuous KI 75):
    96.10 → 98.42 (+2.32); MC 98.440 ± 0.014, from −170 SE to −1.1 SE;
  - 1-year quarterly KO-reset (schedule 1 to 6 months): 98.20 → 99.00 (+0.80); MC 98.999 ± 0.011;
  - a schedule-1 knock-out decided at valuation (spot 110): 94.18 → 97.04, equal to MC.

  The Monte Carlo fold route on these contracts, which raised (the fold structure assumed payment at
  the product maturity), now values.
- **KO-reset settlement precedence: an explicit `settlement_date` comes first** (owner ruling
  2026-11-14). The settlement delay applies only when no `settlement_date` is booked. A KO-reset note
  that ends with the first schedule -- the never-knocked-in maturity and, under EXPIRY pay, a
  first-schedule knock-out -- is paid on the product's `settlement_date` when one is booked, else at
  the first schedule's maturity plus the settlement delay. Monte Carlo already paid it there. The grid
  engines paid it at the first schedule's maturity plus (settlement date − product maturity), a year
  too early on a two-year note whose first schedule ends at one year. PDE 1D (flat and local vol),
  Heston/SLV ADI, QUAD V1, QUAD V2, the KO-reset fold structure, the statistics' `payment_times` and
  theta's schedule-1 end now follow it, on INSTANT and EXPIRY pay. One rule in one place:
  `settlement_support.first_schedule_settlement` (dated: `first_schedule_terminal_timing`).
  - Moves only KO-reset contracts with a `settlement_date` whose first schedule ends before the
    product maturity; a numeric contract cannot carry a `settlement_date` (clock rule), so only dated
    notes move.
  - Contracts without a `settlement_date`, or whose first schedule runs to the product maturity, are
    bitwise unchanged, and so is every other `price()`.
  - Probabilities and cash amounts are unchanged; only the payment time and its discount factor move.

  Moves on the default engines (PDE 1D / QUAD V1 / QUAD V2), with a Monte Carlo reference of
  8 × 65,536 paths:
  - the matrix KO-reset on dates (two years, first schedule to one year, discrete KI 80, settled 20
    days after the exercise date), EXPIRY: 93.14 → 90.67 (−2.47); MC 90.723 ± 0.030, from +82 SE to
    −1.9 SE. INSTANT: 94.50 → 93.98 (−0.52), from +14 SE to −1.9 SE;
  - the same aged 100 days: EXPIRY −2.69 (+109 SE → −1.2 to −1.3 SE), INSTANT −0.80 (+28 SE → −1.1 SE);
    continuous KI: EXPIRY −2.21 (+61 SE → ≤ 0.4 SE), INSTANT −0.30 (+8 SE → ≤ 0.4 SE);
  - local-vol PDE −2.49 (EXPIRY) and −0.37 (INSTANT); Heston ADI −2.57 and −0.55; SLV ADI −2.65 and
    −0.63;
  - the one-year monthly note (first schedule to six months, settled two days after the exercise
    date), EXPIRY: 99.43 → 98.03 (−1.41); MC 98.036 ± 0.010, from +140 SE to −1.1 SE. INSTANT:
    100.07 → 99.65 (−0.42), from +41 SE to −0.9 SE;
  - a first-schedule knock-out decided at valuation (EXPIRY), and a note whose first schedule ends at
    the valuation instant: now equal to MC to 1e-14 (−3.12 and −3.10 on the matrix note on dates).

  The Monte Carlo fold route on these contracts, which raised a binding mismatch on both pay types
  (the fold structure assumed the grid's payment), now values and equals the MC event route on the
  same paths. On a carried knock-in, only the zero-probability maturity row's `payment_times` move;
  the price is bitwise.
- **Booking follows the KO-reset payment rule (D30, D33).** The lifecycle tracker and the intraday
  timeline book a first-schedule knock-out under EXPIRY pay, and the never-knocked-in maturity, on the
  payment of the note that ends with the first schedule: the explicit `settlement_date` when booked,
  else the first schedule's maturity plus the settlement convention. The tracker booked such a
  knock-out on its record's own settlement date, and the intraday timeline at the product terminal's
  payment. The simulated-path calendar schedule places the same settlement day and delay, and exact
  mode still equals the replay. Booking only: no price moves (the intraday price follows in the next
  entry).
- **Theta of a KO-reset knocked out at the as-of date under EXPIRY pay (D30 completion).** Theta
  carries a contract ended at `t0` as its `t0` value re-discounted from its payment, and drops it when
  that payment falls within the bump. A first-schedule knock-out under EXPIRY pay is paid with the
  note that ends with the first schedule (`first_schedule_payment_time`: the `settlement_date` when
  booked, else the first schedule's maturity plus the delay), as every engine prices it; theta read
  the knock-out record's own settlement (the observation plus its lag). With no lag, or a
  `settlement_date` the record does not carry, a note whose first schedule runs on past the knock-out
  was booked as paid within the bump: theta was minus the whole value (on a 1-year principal note
  knocked out at `t0` with the first schedule to 6 months, −113.29 → +0.0093 numeric, −113.29 → +0.0093
  dated, −111.54 → +0.0092 with a `settlement_date`; annualized −98.51 → +0.0081). Same move on QUAD
  V1, QUAD V2, PDE 1D and MC. With a positive lag the moves are round-off (2.8e-14). INSTANT pay,
  second-schedule knock-outs and every `price()` are unchanged. Monte Carlo's knock-out decided at zero
  time to maturity now reads the same first-schedule payment; there the first schedule ends at the
  product terminal, so its price is bitwise unchanged.
- **The intraday twin of a dated KO-reset note pays its first schedule on the settlement date
  (D34)** (owner ruling 2026-11-14). The float-time twin clears the contract's dates, so it booked
  the settlement date only as the terminal's year-fraction lag (settlement − maturity), and its
  engines paid the never-knocked-in maturity, and under EXPIRY a first-schedule knock-out, at the
  first schedule's maturity plus that lag: the pre-D33 rule. The twin now carries that payment
  explicitly. `KnockOutResetSnowballOption` takes `first_schedule_settlement_time` (numeric clock
  only; a dated contract books its `settlement_date`): the explicit payment of a note that ends with
  the first schedule. The intraday timeline resolves it once from the dated contract
  (`ContractTimeline.first_schedule_payment`, the settlement date's payment instant), and every
  engine reads it through the D33 helper (`first_schedule_settlement` /
  `first_schedule_terminal_timing`). Monte Carlo's never-knocked-in settlement and EXPIRY
  first-schedule knock-outs, and QUAD V1's matured-at-valuation branch, now resolve through that
  helper too; their requests are unchanged for every contract without the new field.
  - Moves only intraday prices of dated KO-reset notes with a `settlement_date` whose first schedule
    ends before the product maturity, not knocked in. Notes without a `settlement_date`, one-maturity
    notes, knocked-in notes and every non-intraday price are bitwise unchanged.
  - The intraday price now equals the dated note's price on the same engine on the twin's clock, to
    the bit (valuation at the close, same market objects; PDE 1D, QUAD V1, QUAD V2 and MC, fresh and
    aged, INSTANT and EXPIRY).
  - Moves (intraday QUAD V2 / PDE routes, desk variance profile, flat 20%): a 2-year note with
    principal, first schedule to one year, discrete KI 80, settled 20 days after the exercise date:
    EXPIRY 97.47 → 94.91 (−2.55) fresh and 103.23 → 100.40 (−2.83) four months in; INSTANT
    99.33 → 98.99 (−0.34) and 104.37 → 103.71 (−0.67). On the intraday test notes (no principal,
    6-month first schedule of 12, rebate 10%): EXPIRY −0.050 to −0.079, INSTANT −0.023 to −0.073.
  - No banked cash-leg case carries the new field (census under cause (r)); no golden and no
    certified anchor moved.
- **KO-reset Monte Carlo: the knocked-in payoff reads the spot at the second schedule's end**
  (owner ruling 2026-11-14; BSM, local-vol, Heston and Heston-SLV MC, which share the KO-reset payoff
  kernel). A knocked-in KO-reset is a knocked-in Snowball on the second schedule: its redemption is
  decided on the spot at that schedule's last fixing and paid there. On an ABSOLUTE second schedule,
  including a carried knock-in, the kernel paid the redemption at the second schedule's end but read
  its spot at the grid's last node. Where the first schedule's dates run past the second schedule's
  end, that node is the first schedule's end, so the redemption was read on a later spot. It is now
  read at the second schedule's end node, as REBASED MC, PDE and QUAD already did. The REBASED vs
  ABSOLUTE gap listed as a Known issue was this readout, not a tenor difference: both layouts
  annualize the knocked-in redemption over the same tenor (0.5 on the pair test). The paths are
  unchanged. Price moves are confined to MC on ABSOLUTE contracts whose second schedule ends before
  the grid does: chiefly the ABSOLUTE copy that the lifecycle tracker prices for a REBASED note knocked
  in early. REBASED MC, PDE, QUAD, and contracts whose second schedule runs to the end are bitwise
  unchanged. The contracts measured below (the pair and the tracker copies) have a second schedule
  ending before the first and are refused since D35 (see Breaking); on valid contracts the readout
  moves only where the product `maturity` runs past the second schedule's end, and since D35 no path
  knocks in after the second schedule's end, so every knocked-in path reads there. Moves with a Monte
  Carlo reference of 16 × 262,144 paths:
  - the pair test (REBASED knocked in at valuation, second schedule 0.25/0.5, first schedule to 1.0),
    ABSOLUTE: 88.211 → 88.180 (−0.031), from +6.9 SE to −0.6 SE against PDE 88.182. It now equals
    REBASED to round-off on all four MC engines (on the 4,096-path test engines, BSM −0.0252, local
    vol −0.0174, Heston −0.157, SLV −0.200);
  - tracker copy of a 2-year quarterly REBASED note (second schedule offsets to 1 year), knocked in
    on 0.25 and aged 0.3, spot 78: 72.708 → 72.623 (−0.085), from +8.0 SE to −1.0 SE against PDE
    72.631; without annualized KI: −0.068 (+8.0 SE → −1.0 SE);
  - the same note knocked in on 0.5 and aged 0.55, spot 85: −0.038 (+2.5 SE → −1.6 SE);
  - knocked in on the first schedule's last record (the second schedule ends after it): unchanged.

  No banked cash-leg case and no cross-engine matrix contract is in this class (census under cause
  (p)); no golden and no certified anchor moved. The knocked-in settlement cash in the statistics
  moves with the price (`npv == price()` bitwise).
- **PDE knocked-in edge row honours partial protection without an airbag** (owner ruling
  2026-11-14; PDE 1D flat and local vol, Heston/SLV ADI; Snowball, reverse, Phoenix, KO-reset). The
  protected closed-form rows of the knocked-in surface write a constant at the knock-in edge: a
  standard note's maximum loss at S = 0 (floored), a reverse note's floor. That constant is the
  payoff only where the protection floor binds. Where the floor does not bind on the edge cell (a
  deep floor such as a 90% maximum loss, or a reverse floor beyond the upper edge), the row is now
  read off the knocked-in payoff, as for the annualized-KI and airbag rows. Contracts whose floor
  binds on the edge cell (for example a 10–30% maximum loss) are bitwise unchanged, as are NONE and
  FULL protection. The width-3 vs width-16 gap falls from 0.067–0.12 to ≤ 3.3e-4 (flat), 1.7–2.1 to
  ≤ 6.2e-3 (local vol), and 0.02–1.29 to ≤ 4.1e-3 (ADI); the remaining local-vol Phoenix and SLV
  Phoenix gaps equal the unprotected contract's. Moves on the default grids, with Monte Carlo
  references of 16 × 20,000 paths (local vol: 1-day SDE steps; Heston: 32 substeps per interval):
  - 1-year quarterly Snowball (discrete KI 75), 90% maximum loss: local vol 98.622 → 99.321
    (+0.699; MC 99.297 ± 0.027, from −25.1 SE to +0.9 SE); flat +0.0019; Heston +5e-5;
  - KO-reset (2-year, KI 80), 90% maximum loss: local vol 97.897 → 98.528 (+0.631; MC
    98.530 ± 0.028, from −22.8 SE to −0.1 SE); flat +0.0011; Heston +2e-6;
  - 2-year monthly Snowball (continuous KI 75), 90% maximum loss: local vol 97.958 → 98.607
    (+0.648; MC 98.600 ± 0.034, from −18.8 SE to +0.2 SE); flat +0.0011; Heston +2e-6;
  - the same note at a 70% maximum loss: local vol 98.573 → 98.607 (+0.033; MC 98.703 ± 0.026,
    from −5.0 SE to −3.7 SE; see Known issues); flat and Heston unchanged (the floor binds on their
    edge cells);
  - reverse Snowball at half participation, 100% maximum loss (floor above 300): flat +0.0015;
    local vol and Heston unchanged. A reverse note at full participation (floor at most 200) binds
    on every default edge and does not move.
  No banked cash-leg case is partially protected (census under cause (o)) and no certified anchor
  moved. One frozen PDE reference moved and was re-banked: the Phoenix memory fan's partial row
  (90% maximum loss, flat default grid) +0.00207, now equal to the unprotected note; its
  probabilities, times and coupon fields are unchanged.

- A realised receivable due on this close up to rounding (`Tolerance.ZERO`) is paid on it
  (`LifecycleCashflowLedger`): a numeric year-fraction lag added to an aged time could land an ulp past the close,
  stay pending, and then raise "payment must be strictly after valuation".
- Theta of a numeric discrete FX barrier / sharkfin raised "times must be strictly increasing and positive":
  `time_shift` aged the maturity but not the fixing times. The fixing times (and recorded fixings) now move with the
  clock, so numeric theta equals its dated twin's.
- (Superseded by the clock rule under Changed: a numeric note can no longer carry an `initial_date`, so there is
  no aged note to route.) Tools that start from an `EquityPosition` — the static PnL explain's
  `BookSnapshot.from_portfolio`, the trade Greeks' theta split and SIMM's vega option tenor — read the note as
  booked, which is the note the portfolio values.
- Parametric equity VaR understated vol and rate risk 100×, in both portfolio modes. An `EquityPosition`'s vega
  is per vol point and its rho per 1 %, but the `vol_change` / `rate_shift` factors are absolute changes;
  `ParametricVaREngine` now divides vega, rho and `dividend_rho` by 0.01 (TRS rho is per unit of rate already and
  is unchanged). Product mode's `include_div_yield=True` crashed on a Black–Scholes proxy importing the
  nonexistent `quantark.param.div_yield` (and the proxy could price only a European vanilla); it now uses the
  position's own `dividend_rho`, as trade mode does. Parametric vol/rate VaR, factor VaR and incremental VaR
  change; historical and Monte Carlo VaR (full revaluation) do not.
- `KOResetSnowballQuadEngine` (QUAD v1) event distribution at an observation instant. `price` decides a
  knock-out at the valuation instant on the known spot, on the schedule the knock-in state puts in force, but
  `calculate_event_stats` / `price_with_events` skipped it: the decided record got no probability and its value was
  booked as the maturity cash flow. Cash legs on a knock-out-reset were mis-valued on an observation day (a
  trade-mode backtest or replay breaking value + booked cash conservation on the knock-out day). The decided
  knock-out is now certain on its record, as on QUAD v2 and the PDE; every other distribution is bitwise unchanged.
- `KOResetSnowballQuadEngine` (QUAD v1) event distribution once a knock-out schedule has ended. The event
  statistics returned nothing when either schedule had no record left, so `price_with_events` fell back to a
  maturity-only distribution: a knocked-in note past its first schedule had P(maturity, no knock-out) = 1, its
  `AT_KO` legs raised `ValidationError` and its `AT_MATURITY_NO_KO` legs were paid in full, every day for the
  rest of its life; an aged-out second schedule did the same before a knock-in. The recursion now runs with the
  ended side empty and agrees with QUAD v2 to its lattice difference.
- Convertible puts were silently dropped. GS, HW (all three vol schemes), jump-diffusion and the
  TF PDE applied a put only on a step whose whole-day date equalled the put date; at the default
  200 GS steps the demo puttable bond priced the same with or without its put (worth ~2.4). Puts
  now land on the first step at or after their date (a coupon paid on the put date is kept), and a
  put in the last step is exercised at maturity.
- Convertible coupons could fall between two steps and never be paid: the coupon window ended at
  `t + dt` while the next step began at `(i + 1) * dt`, and the last step could end an ulp short of
  maturity. GS, HW, jump-diffusion and the TF PDE priced a 5% semi-annual bond ~2.2–2.5 low on some
  grids. Coupon windows now end on the step grid, whose last edge is maturity exactly; rate lookups
  are unchanged. On a sweep of 300 GS grids, 264 prices are bit-identical and the other 36 rise by
  one discounted coupon (2.21–2.46), each a grid that used to drop it.
- GS and HW (all three vol schemes) converted at maturity even when the conversion window had closed
  before it; the TF tree, TF PDE and jump-diffusion already redeemed. On the demo bond with
  conversion ending seven months early, GS was 2.34 high at the money (now 122.19, next to the TF
  tree's 122.85) and HW 1.23–1.86 high. Bonds convertible at maturity are unchanged.
- A final coupon paid after the maturity date (a weekend maturity rolled forward, or
  `settlement_days > 0`) fell after the last time step and was never paid: every convertible engine
  priced such bonds about one discounted coupon low (0.90–0.95 per 100 face far out of the money on
  the demo bond). `ConvertibleBond.coupon_time` now pays it at maturity; every other coupon keeps
  its time, so other prices are unchanged.
- The stock was not a martingale in the hazard-rate convertible engines. The stock drops by the
  fraction `stock_jump_on_default` (η) on default, so while it survives it must drift at
  `r - q + λη`; jump-diffusion used `r - q - λη` and the HW trinomial `r - q`. The constant-vol HW
  scheme also moved u / 1 / d with weights ½p / ½ / ½(1−p), half the drift and 1.5× the variance,
  so it converted deep in-the-money bonds almost at once even without credit risk (a riskless demo
  bond at parity 300: 300.57, now 311.82 next to GS and the TF tree). Its probabilities now match the
  step's mean and variance exactly and raise instead of clipping. On the demo bond at the default
  hazard 1% and η = 0.4, jump-diffusion rises 1.7–4.4 and HW 0.9–5.0 per 100 face (spot 8–30), and
  the four engines now agree to 0.015 (they were 2.7 apart). A European zero-coupon convertible now
  matches its exact value to 0.002 on all four (and jump-diffusion's conversion probability matches
  `e^{-λT} N(d2)`), and the demo bond matches Hull's tree to 0.0012 (OFOD §27.4: stock to zero,
  σ the vol conditional on no default, `u = e^{σ√Δt}`, Example 27.1 = 107.44; later editions use
  `√(σ² − λ)`). With no hazard, jump-diffusion and the log-price HW schemes are unchanged.
- TF CS01 was identically 0: `ConvertibleBondEngine.cs01` bumped `hazard_rate`, which the TF engine
  never reads. TF (tree and PDE) now bumps `credit_spread`, as GS does. The bumped bonds keep their
  calendar, business-day convention and settlement days, and a down bump floored at zero divides by
  the actual bump width: GS and TF CS01 at `credit_spread` below 1bp, and HW and jump-diffusion CS01
  at `hazard_rate` below the bumped intensity, were half the one-sided value.

### Known issues

- DCN prices and statistics discount the DCN's ACT/365F grid times on the environment curve, which
  is wrong under `BUSINESS_DAYS` or `ACT_360` environments; the fix moves prices and is left to a
  separate release.
- REBASED KO-reset statistics are MC-only; local-vol MC `estimator="one_step_survival"` has no
  statistics; `PricingSession` `EVENT_STATS` is refused for `DCNPDEEngine` and
  `LocalVolDCNPDEEngine`.
- Tracked study outputs computed before this release still show the old KI-ever and maturity
  numbers; see section 11 of the migration note.
- No KI after a certain KO on grid engines holds up to truncated-domain edge mass. When the KO
  barrier lies below the grid (the AC9 test contract), the knocked-in far-edge row counts the
  edge-reaching mass as knocked in. LV PDE: 1.07e-5 at the default width 4, 4.9e-9 at 6, 2.6e-13 at
  8; ADI ≤ 5e-9. P(KI) = 1 − ΣP(KO) in every case. Accepted as a recorded deviation.
- Local-vol MC default stepping (one log-Euler step per contractual interval) is biased on skewed
  surfaces; references and the cross-engine tests use `max_sde_step_days=1`. Defaults unchanged.
- PDE knocked-in edge row when the protection floor's kink lies just beyond the grid edge: neither
  the closed-form row (floor binding on the edge cell) nor the payoff-read row (floor not binding,
  0.11.0) sees a kink outside the domain, so the price keeps a domain-width dependence while the
  kink is within reach of the edge. Seen on skewed local vol at maximum losses of about 50–78%
  (2-year monthly Snowball, default grid vs width 16: up to −0.14 on either row; at 70% the
  payoff-read row leaves −0.133, 3.7 SE below MC, down from −0.167). Flat vol and the Heston/SLV
  ADI default grids are within 2e-3. Pre-existing; a wider domain removes it.
- KO-reset ABSOLUTE contracts whose second schedule ends before the product `maturity`. PDE 1D,
  QUAD V1 and QUAD V2 depend on the product maturity for a carried knock-in, which MC does not. On a
  valid note (both schedules monthly to 6 months, KI 80, spot 90, carried knock-in) the grid engines
  give 94.43 with maturity 1.0 and 95.06 with maturity 0.5; MC gives 95.07 either way. Tracker
  copies end with their second schedule and agree across engines. Pre-existing; not investigated.
- D29 counts a settled note as gone from its determination time, not its payment time. A
  COMPLETE_PERIOD funding leg whose last period ends on a later settlement date pays nothing on the
  matured mass under the default; opt in to `survival_until=SurvivalAnchor.PAYMENT` (D32) to accrue
  to the payment date.
- Theta's at-valuation schedule-2 knock-out of a KO-reset under EXPIRY pay is paid at the record's
  own settlement, while the engines pay it at the product terminal. Pre-existing; unmeasured.
- The realized `LegCashBook` never truncates accrual on a maturity outcome, under either anchor.
  Pricing leaves the matured note at its determination (D29) or, under `SurvivalAnchor.PAYMENT`, at
  its payment, so realized and priced accrual differ past the maturity determination.
- A dated KO-reset with a business-day settlement lag and no `settlement_date`: MC raises (it
  resolves the never-knocked-in payment from a time, and a business-day lag needs a date). The grid
  engines shift the terminal's delay to the first schedule's maturity, while the tracker and the
  intraday timeline apply the convention to the first schedule's dated maturity. Where the
  business-day delays differ between the two maturities, booking and grid pricing disagree.
  Pre-existing; unmeasured.

- TF prices of hard-callable bonds are grid-sensitive in both TF engines (about ±0.45 per 100 face
  on the demo callable bond as N or the PDE grid changes; conversion probability swings between
  about 0.63 and 0.85, and delta with it). Refinement does not cure it: the tree is still bimodal at
  N = 32,000. With the call price above the coupon carry, a hard call happens only at the tie
  parity = call price, where the TF split into cash and shares is undefined; on a lattice the node
  at the tie is called for cash (the call price joins the risky-discounted cash component) or
  converted, depending on the grid. It needs `credit_spread > 0`; the GS tree is not affected.
- QUAD V1 `price()` never applies a knock-in to the EXPIRY coupon column under cell-average
  projection when a discrete KI falls on a KO date and `disable_ko_after_ki` is set. A known price
  defect, pre-existing and not measured; it is left for a price-moving release.
- Dated intraday Snowball: MC gives 0.919 / 0.923 against QUAD V2 0.905 on the same request, beyond
  MC noise. Pre-existing; not investigated.
- KO-reset grid engines without a `settlement_date` seed the never-knocked-in maturity with the
  product terminal's delay factor DF(T + δ) / DF(T) at the first schedule's maturity, not
  DF(pre + δ) / DF(pre). Exact on a flat curve; on a term curve with a settlement delay δ > 0 it is
  off by the forward-rate difference over δ. Pre-existing, kept for bitwise prices.
- QUAD V2 memory Phoenix refuses a lifecycle-tracker missed-coupon count beside a booked
  `initial_coupon_arrears` amount that differs from it, even when the amount covers periods before the
  tracked schedule (two separate debts). No library path builds that combination.
- DCN Heston ADI `price()` is about 5% slower than in 0.5.5 (3.40 s vs 3.23 s on the reference
  note); prices are bitwise unchanged.
- DCN Monte Carlo cash-leg bank records can differ by 1–2 ULP between runs under parallel test
  workers; serial runs are stable.

## [0.5.5] - 2026-10-16

> **Version numbering.** The `[0.6.0]` to `[0.10.0]` sections below were internal milestones and were
> never published. They ship together with everything under this heading as **0.5.5**, the first
> release after 0.5.0. The milestone labels are kept because code comments, tests and the README name
> behaviour changes by them (for example "the 0.7.0 base-family mask").

### Test suite

Full-suite wall-clock 36m 29s to 20m 03s (14 xdist workers, `-n auto --dist worksteal`); no test
changed meaning and every banked anchor still reproduces exactly on the banking machine.

- `assert_anchors` no longer solves a convergence rung it discards: the Schema-1 candidates (ADI
  snowball, snowball, KO-reset, Phoenix, local-vol snowball) gain `evaluate_target`, and
  `assert_anchors(path, only=(candidate, case))` checks one cell. `test_banked_certificates.py` is
  parametrized per cell, so the heavy ADI cells spread across workers (that file: 2,353 CPU-seconds
  with a 1,687 s single-worker tail to 317 s wall-clock).
- The QUAD V2 event-statistics references used by the Phoenix memory-fan, KO-reset native and
  cashleg KO-reset tests are frozen under `test/pde_statistics/v2_reference/` (about 2,500
  CPU-seconds to 4 s) with unchanged tolerances. `build_v2_reference.py` regenerates them;
  `QUANTARK_LIVE_V2=1` recomputes live to confirm a frozen file still matches V2. V2's numerics
  stay pinned by the banked quad certificates.
- `quantark.backtest` imports plotly and matplotlib lazily; `StaticVisualizer`,
  `InteractiveDashboard`, `AutocallableBacktestDashboard` and `AutocallableDashboardConfig` still
  resolve from the same import paths.
- `test/conftest.py` collects cycles and returns freed pages to the OS (`malloc_trim` /
  `malloc_zone_pressure_relief`) after any test that ran 0.5 s or more. Workers kept the memory of
  multi-GB grids, so one reached ~9 GB and four ~24 GB; on the six heaviest files, run serially, peak
  RSS fell 10.2 GB to 4.3 GB at the same wall-clock. CI runs `-n 2` rather than `-n auto`: four
  heavy tests at once exceeded the 16 GB hosted runner, which was shut down mid-suite.
- Removed `test_bond_standalone.py` (no tests) and `test_benchmark_var.py` (module-skipped, vacuous
  assertions), two duplicate QUAD-rejection tests, and the stale "not implemented yet" skips in
  `test_simm_engines_factory.py` (now real assertions).

FX TARN / TARF mid-life booking, dated form and ageing (spec:
`docs/superpowers/specs/2026-10-09-fx-tarn-lifecycle-design.md`). A target-redemption contract carries its realized
state, can be booked by dates, ages through `time_shift` / `record_passed_fixings`, and is aged day by day in the FX
backtest and FX dynamic scenario.

### Added
- `quantark.asset.fx.product.option.target_redemption_state`: `TargetRedemptionState` (amount banked towards the
  target, determined `PendingPayment`s, dated `as_of`), `TargetRedemptionEvent` / `TargetRedemptionEventType`,
  `TerminalBookingWarning`, `ResolvedTargetRedemption`, and `TARGET_HIT_RTOL` (1e-12, moved from the two TARN/TARF
  engines with its value unchanged).
- Dated `FxTargetRedemptionNote` / `FxTargetRedemptionForward`: `fixing_dates` (the whole contractual schedule),
  `pay_dates`, `state.as_of`; the TARN derives its periods from `initial_date` (or explicit `accrual_factors`) in its
  `day_count_convention` (default ACT/365). Times resolve in the environment's convention; a numeric contract and its
  dated twin price identically. A fixing or pay date that resolves to zero time in the environment's day count (a
  weekend date under `BUSINESS_DAYS`, valued on the Friday) is live, as for its numeric twin (user decision OQ-2 b,
  2026-10-10): the fixing is decided at the quoted spot and its cash is due that day. A dated state counts its banked
  fixings (`TargetRedemptionState.fixings_banked`, fixing identity in schedule order; a booking stating only `as_of`
  banks the fixings dated on or before it); `as_of` never passes the valuation date (user ruling OQ-3, 2026-10-10).
- `resolve(fx_env)` (the engines' one input), `lifecycle_events`, `is_redeemed`, `is_terminal`, `hit_tol`,
  `pending_pv(fx_env)` on both products.
- TARN / TARF take part in the missing-fixing fill protocol (`missing_fixing_fills`, `with_fixings_recorded`,
  `with_missing_fixings_filled`, user decision 2026-10-09): an unbanked passed fixing's RATE is recorded once at the
  base environment's quoted spot, on a copy, so `BaseFxEngine.calculate_greeks`, VaR, stress, dynamic scenarios and
  backtests bank it at the base spot under every bump and shock (one `MissingFixingWarning` per call, none per bump).
  Banking (`record_passed_fixings`, `resolve`) uses a recorded rate in place of the quoted spot; a rate recorded for a
  fixing still after valuation is refused with `ValidationError`.
- `FxTargetRedemptionNote(accrual_end_dates=...)` (dated form; user ruling 2026-10-10 on R26, option (b)): each
  period's own accrual end date, separate from its fixing date. Period `i` accrues from `initial_date` (first) or the
  previous period's end to its own end in the contract day count, so periods may share a fixing date without explicit
  `accrual_factors`. One end per fixing, strictly increasing, each between its fixing and pay dates; it requires
  `initial_date` and excludes `accrual_factors`. Without it the ends are the fixing dates: every existing booking keeps
  its factors and value exactly; such a contract still cannot share a fixing date (a zero-length period).
- FX TARN / TARF take part in run fixings (`quantark.util.run_fixings`, W6): `has_run_fixings`,
  `open_fixings(fx_env)`, `with_time_elapsed(dt)` (a numeric copy aged by `time_shift`, times snapped to exactly 0.0,
  nothing banked), `with_history_settled(fx_env)` (recorded history banked and cash due paid, on a copy) and
  `matured_before(fx_env)`. In the FX backtest and FX dynamic scenario they age with the run's clock: a fixing is
  recorded at the spot of the first instant it was live (a fixing between path instants at the previous instant's,
  `gap=True`), banked in order with its cash; a contract whose last fixing is before the run start is removed with
  `MaturedBeforeRunWarning`. A dated booking is validated against the run's environment first (`_check_as_of`), so a
  stale invalid one (a state claiming an `as_of` or banked fixing after the run start, or pending cash already past)
  raises `ValidationError` at registration instead of being dropped with the warning; numeric bookings are unchanged.
  Moved values: none (a refusal replaces a warning on invalid bookings only).
- `RunFixingRecorder` calls an optional product hook `with_history_settled(env)` after the first valuation's fill and
  after each clock move's recording (`settle_history`); products without it are unchanged.
- `target_redemption_state.booked_cash(product_events)`: the quantity-scaled PAYMENT cash of one run step's
  `RunFixingDay.product_events`.

### Changed
- The equity analytical barrier family (`BarrierAnalyticalEngine`, `OneTouchAnalyticalEngine`,
  `DoubleBarrierOptionAnalyticalEngine`, `SingleSharkfinOptionAnalyticalEngine`, `DoubleSharkfinOptionAnalyticalEngine`
  and the barrier legs of `AccumulatorAnalyticalEngine`) values a DISCRETE schedule by one method: with no record
  deciding at the valuation instant and a future observation left, always through the first future observation (the
  exact Gaussian step to it and the barrier test there, `first_observation`), the BGK-shifted closed form only from
  it on -- inside the level as well as beyond it. Before, inside the level the shifted closed form was evaluated at
  today's spot (a hybrid that stepped by up to 0.161 at the level on monthly contracts). No observation left: the
  closed form, as before. Unchanged: a record on the valuation instant that the spot breaches decides at once
  (W18c / A15), CONTINUOUS and EXPIRY monitoring, T = 0, a carried knock-in.
  - A schedule that starts late is a large correction toward QUAD (401 points): the up-and-out put (K 110, L 120,
    vol 0.6, rebate 2 at the hit, five daily records ending at T = 1) at 105: 14.79 -> 26.907 (QUAD 26.908).
  - Far from the level fresh monthly contracts move slightly, sometimes away from QUAD: the up-and-out call
    (K 100, L 120) 4.1453 -> 4.1474 (QUAD 4.0778).
  - The projection needs a first-hit schedule: an ACCUMULATE (or other non-STOP_FIRST_HIT) schedule is refused
    (`PricingError`) inside the level too (main priced the aged up-and-out call at 100 at 2.4865).
  - With no time left after the first observation (it is the maturity: valued on the penultimate record, not
    crossed) the payoff's kinks are the expectation's panel edges; such a contract was refused (`NumericalError`,
    the strike inside a panel) and is now the European payoff decided at T, e.g. the up-and-out call 5.562826
    (closed form 5.562826, main's shifted formula 5.6967), the double knock-out call 5.780781 (main 5.7835). A tiny
    time left (the payoff smoothed over less than the 0.05 panel floor, e.g. 2e-10 <= tau <= 1e-6 after a monthly
    record) grades the panels away from each kink from that scale, and from a region edge that kink's layer reaches
    (a strike just beyond the level); such contracts were refused too and now match the European value over tau
    (within 1e-11, measured <= 1e-13).
  - Performance: the projection evaluates its integrand once on every quadrature node (vectorised closed forms,
    records resolved once per price), bit-exact on ARM64 (last-bit differences possible where NumPy uses its own
    SIMD exp / log / pow, x86 with AVX-512); inside the level the barrier-family prices now take 0.5-2.5x the time of
    the old closed-form path (double sharkfin weekly 2.47x, double knock-out daily 3.1 ms, accumulator daily 0.38 s).
  - Moved values: every DISCRETE cell with no breached record now and the spot inside the level (fresh contracts
    included, and a record day whose record the spot does not cross); beyond the level the values are bitwise
    unchanged. Next to the level the error vs QUAD drops, e.g. the up-and-out call (K 100, L 120, monthly, 0.7 y
    left) at 119.99: 0.7801 -> 0.7227 (QUAD 0.7140), the double knock-in call (75 / 130) at 129.5: 29.1647 ->
    29.3132 (QUAD 29.3007); max |error| within 0.5 of the level 0.004-0.04 (was 0.01-0.16). On a record day (aged
    onto a record the spot does not cross) the max |error| vs QUAD is 0.048 (was 0.24).
- A no-touch has one rebate on every route (user decision 2026-10-15, as a knock-in's since 2026-10-10): it is paid
  only when nothing touches, so a schedule record of a `OneTouchOption` NO_TOUCH or `DoubleOneTouchOption`
  DOUBLE_NO_TOUCH paying anything other than `rebate` is refused (`ValidationError`,
  `barrier_schedule_rules.refuse_no_touch_record_payoff`) by the analytical one-touch engine, `DiscreteQuadEngine`,
  the PDE touch solvers / `PDEEngine`, and a run's `BarrierLifecycleTracker` (the equity backtest and the dynamic
  scenario take the contract there; the cash it books was already right). Before, QUAD subtracted a one-touch
  paying each record's payoff: records paying 3 on a rebate of 1 gave 0.14 where the analytical engine and the PDE
  gave 0.69. A record without a payoff, or one paying the rebate, prices as before; the single sharkfin's no-hit
  leg carries records paying its cash, so sharkfin values are bitwise unchanged.
- **Breaking:** `FxTargetRedemptionNote` and `FxTargetRedemptionForward` take a mandatory keyword `state`
  (`TargetRedemptionState(0.0)` for a contract booked at inception); omitting it raises `ValidationError`.
- **Breaking:** a numeric `FxTargetRedemptionNote` states when its first listed period began: `accrual_start` (years
  from valuation) or explicit `accrual_factors`, exactly one. The old default (`t_0 = 0`, the first period starting at
  the valuation instant) halved the first coupon of a contract rebuilt mid-life. `accrual_start=0.0` reproduces the old
  factors bitwise.
- A non-positive fixing time is accepted (an unbanked past fixing) and an empty `fixing_times` is a terminal booking
  (`TerminalBookingWarning`, valued at the PV of its pending cash).
- FX finite-difference theta (`BaseFxEngine._fdm_theta`) steps a numeric (`maturity=`) product by the environment's
  year fraction of `theta_days` calendar days instead of `theta_days / 365` (user decision Q7). Unchanged to the bit
  under `CALENDAR_DAYS` / `ACT_365`; under `BUSINESS_DAYS` a Friday or holiday theta is 0.0 and `ACT_360` steps 1/360,
  matching the dated product. Values that move are listed in
  `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt`.
  **Breaking:** under a `BUSINESS_DAYS` environment with no calendar, a numeric FX product's theta now raises
  `ValidationError` on every route that steps time (it used to step `theta_days / 365`; a business-day count needs the
  calendar). The raising routes are exactly: every engine whose `calculate_greeks` reaches
  `BaseFxEngine._fdm_theta` (`GarmanKohlhagenQuantoEngine`, `FxQuantoDigitalAnalyticalEngine`, `FxDeltaOneEngine`,
  `VannaVolgaBarrierEngine`, the FX range-accrual analytical engines, and the FX barrier / sharkfin, TARF, TARN and
  range-accrual MC engines); the finite-difference fallbacks of `GarmanKohlhagenEngine` (Vanna-Volga smile, delivery
  differing from expiry, or a market forward set) and of `FxDigitalOptionAnalyticalEngine` (the same three, plus any
  payout not in the domestic currency, i.e. asset-or-nothing); and, through the local-vol / Heston theta change below,
  the FX local-vol, Heston and Heston-SLV MC / PDE routes and the FX Heston analytical engine. The only non-raising
  path is the closed-form branch (`-dV/dtau / 365`, no time step) of `GarmanKohlhagenEngine` and of
  `FxDigitalOptionAnalyticalEngine`, taken only on a flat or term vol surface with delivery equal to expiry, no market
  forward and, for the digital, a domestic-currency payout (cash-or-nothing). Measured 2026-10-14.
- FX local-vol and Heston theta (`fx_local_vol_model_greeks`, `FxHestonPDESolver`, `FxHestonSLVPDESolver`, and the LV/Heston
  analytical and MC engines that share the first) step a numeric (`maturity=`) product by the environment's year
  fraction of `theta_days` calendar days, as `BaseFxEngine._fdm_theta` does, instead of `theta_days / 365`. The
  `min(step, 0.5 * maturity)` cap is kept and theta is rescaled to a per-calendar-day figure; a zero step (BUSINESS_DAYS
  Friday or holiday) gives theta 0.0 and BUSINESS_DAYS without a calendar raises `ValidationError`. Moved values:
  none under `CALENDAR_DAYS` / `ACT_365` (bitwise); on a 1-year ATM EUR/USD call (notional 1e6, flat 10% vol) `ACT_360`
  local-vol theta -97.0526 -> -98.4012 and BUSINESS_DAYS-on-a-Friday -97.0526 -> 0.0 (coarse-grid Heston PDE 434.64 ->
  432.38 and 0.0; Heston-SLV PDE 462.84 -> 460.54 and 0.0).
- FX TARN / TARF cash due on the valuation date stays in the value (user decision OQ-1 b, 2026-10-10, as W8 values a
  dated contract on its date at its settled value): determined cash is booked (a `PAYMENT` event, out of the value)
  once its pay time is strictly past. A numeric pending `pay_time` of 0.0 and a dated pending `pay_date` equal to
  `as_of` are accepted. Moves: the price on a pay date gains the cash due that day, theta the day before a pay date
  gains it and theta on the pay date loses it (values in
  `docs/superpowers/plans/2026-10-09-fx-tarn-lifecycle-evidence/moved_values.txt`); inside runs, cash is booked
  once its pay date is past (on the next path instant). "Due" is decided by resolved TIME in the environment's day
  count, also when a date carries a time of day: under a whole-day count, cash due at 00:00 is still due at 15:00
  that day, and 10:00 cash at the next midnight. `TargetRedemptionState` no longer compares a dated `pay_date` with
  `as_of` (it cannot know the day count); the product refuses pending cash strictly past at `as_of` by resolved time
  when an environment prices, records or settles it (`ValidationError`, the same refusal moved from construction).
- FX TARN / TARF periods may share a fixing date or time (user decision OQ-4 b, 2026-10-10): `fixing_dates` /
  `fixing_times` must be non-decreasing instead of strictly increasing. Both MC engines simulate each distinct fixing
  time once and every period sharing it fixes at that spot; periods are accumulated in schedule order (a period after
  the target hit is dead), each with its own pay date and cash. A dated TARN on the default period derivation
  (`initial_date`, then each previous fixing date) cannot share a fixing date (a zero-length period): it states its own
  accrual. The default derivation refuses any non-positive derived factor (two fixings on one day under ACT/365, at
  construction; a zero-business-day period under a `BUSINESS_DAYS` contract count, at resolve), which a dated contract
  priced before. A dated state's banked count (`fixings_banked`, or derived from `as_of`) may not stop inside the
  periods fixing on one calendar date unless the target was hit. `mean_redemption_fraction` does not count a hit at the final fixing time as early.
- FX TARN / TARF: a fixing at exactly the valuation instant (t = 0) is live (run-fixings D1):
  `missing_fixing_fills` / `with_missing_fixings_filled` list only strictly past fixings, so the Greek freeze leaves
  today's fixing to move with the bumped spot, and pricing decides it at the quoted spot without a
  `MissingFixingWarning`. Numeric protocol indices and event labels count from the booking (fixings banked since
  construction + listed position).
- Run fixings (`quantark.util.run_fixings.step_fixings`): a record that was live is recorded first, at its first live
  instant's spot, in index order; only then are the remaining (gap) records probed at the previous instant's spot. A
  TARN / TARF on a business-day clock whose weekend rows quote a spot other than Friday's now records its Friday and
  Saturday fixings at Friday's spot (was a `KeyError`, a gap at the weekend spot, or a split shared-date refusal).
  W6's range accruals and Asians record exactly as before. `ProductReplay` refuses a TARN / TARF (`ValidationError`):
  it does not settle product-held history.
- Replay settles an Asian / range accrual at maturity (spec 2026-10-13 I7 / A3): `AsianOption` and
  `RangeAccrualOption` gain `settled_maturity_payoff(spot, pricing_env)`, the terminal payoff from the contract's
  recorded fixings with a fixing on the maturity instant (t = 0) decided at `spot`, the spot at maturity (refused
  before the maturity instant and for an unrecorded past fixing; an Asian booked without `observation_records` raises
  `ValidationError` naming the fixings it does not carry, never a `TypeError`). For a run-fixings participant the
  replay steps the day's fixings first and settles on that run-fixed copy (`ProductReplay.settle_run_fixings_maturity`,
  `AutocallableLifecycleTracker.settle_maturity_if_due(..., settled_product=, previous_spot=)`; without the copy a
  participant's maturity raises `ValidationError`), so the maturity day's own fixing is taken at that day's spot. A run
  whose calendar skips the maturity date settles on the first run day after it: the maturity fixing (and a floating
  strike's final spot) at the previous run instant's spot (W6 gap rule), `gap=True` / `maturity_spot` on the MATURITY
  action and its ledger entry, which keeps the contract's own determination and payment timing (a payment date already
  past is paid that day). Every other product keeps the old order (replay goldens unchanged). The replay's analytical
  engine factory builds `RangeAccrualAnalyticalEngine` for a range accrual (was refused). `PortfolioLifecycleManager`
  never tracks an Asian / range accrual, so the equity backtest and dynamic scenario are unchanged. Moved values: none
  (raise -> settled cash).
- `FXBacktestEngine` books TARN / TARF cash once its pay date is past (on the next path instant): rows gain
  `paid_cash` (cumulative, quantity-scaled) and `pending_pv`; `portfolio_value` and `final_value` include paid cash.
  A book without TARN / TARF is unchanged (cash columns 0.0).
- `FXDynamicScenarioEngine` books TARN / TARF cash on a dated `DayPath` (determined cash moves to `paid_cash` /
  `realized_cash` once its pay date is past (on the next path instant); one `FIXING` lifecycle snapshot per banked
  fixing; `pending_receivable_pv` filled). An undated `DayPath` with a TARN / TARF in the book raises
  `ValidationError` (user decision Q8 a). Books without TARN / TARF are unchanged.

### Fixed
- FX TARN / TARF theta within one day of a fixing raised (`Time to maturity must be non-negative` / `times must be
  strictly increasing and positive`); the stepped-over fixing is now banked at the held spot and theta is finite.
- Pricing a TARN / TARF after a bare `time_shift` past a fixing raised; the passed fixing is now filled at the quoted
  spot on a copy with one `MissingFixingWarning`.
- A rebuilt mid-life contract no longer prices with the full target: the engines start from `state.accumulated` and
  add the PV of `state.pending`. Inception prices and Greeks are unchanged to the bit.

Accrual time, separate from observation time (spec: `docs/superpowers/specs/2026-10-08-accrual-time-design.md`).
An autocallable's observation is a trading-day quantity (when the market is checked); its accrual is a calendar
quantity in the contract's interest basis (how much it owes). Every snowball, Phoenix and KO-reset record now
carries its own immutable `AccrualPeriod`, derived once at construction, and every product, engine and intraday
path reads accrual through one resolver (`quantark.asset.equity.product.option.accrual`). Ageing (`time_shift`),
snapping an observation time or rebasing a schedule moves observation timing and never moves cash. A numeric
record's accrual is its quoted fraction (`k/12` accrues exactly 1/12 per period).

### Added
- Run-fixings protocol on `RangeAccrualOption`, `FxRangeAccrualOption` (and its foreign / quanto subclasses) and
  record-schedule `AsianOption`: `has_run_fixings`,
  `open_fixings(env)` and `with_time_elapsed(dt)` (the run clock's step: a numeric contract's record times and
  maturity move by `dt`, snapped to exactly 0.0 when zero; records nothing; a dated contract returns itself).
  `quantark.util.fixings.snap_time`.
- `quantark.util.run_fixings` (spec `docs/superpowers/specs/2026-10-09-run-fixings-design.md`): `RunFixingRecorder`
  (`register` at a run's first valuation = W5b's freeze; `advance` at every later path instant records each fixing
  the clock moved past at the spot of the first instant it was live, or at the previous instant's spot with
  `gap=True` when no instant fell on it; numeric contracts age by the environment's year fraction; time-of-day
  instants and a backwards clock are refused), the stateless `step_fixings`, `update_live`, `participates`,
  `run_step_dt`, `at_spot`, `require_date_instant`, `RecordedFixing`, `RunFixingDay`.
  Optional product hooks `matured_before(env)`, `lifecycle_events` and `pending_pv(env)` feed `RunFixingDay.product_events` /
  `pending_pv` (for FX TARN / TARF, W4b).
- `AccrualPeriod`, `AccrualBasis`, `KnockInAnchor` and the resolver: `ko_accrual_factor`, `coupon_period_fraction`,
  `contract_tenor`, `knock_in_anchor_for`. `ObservationRecord.accrual` holds a record's period.
- `AccrualConfig.accrual_end` (`AccrualEndType.OBSERVATION`, the default, or `SETTLEMENT`: a record accrues to its
  authoritative payment endpoint, resolved from the contract's own terms).
- `AccrualConfig.post_accrual_factors`: KO-reset schedule 2's explicit factors (`accrual_factors` is schedule 1's).
- `AccrualConfig.accrual_calendar` / `accrual_bus_days_in_year`: the frozen terms of a `BUSINESS_DAYS` accrual; no
  accrual reads the pricing environment.
- `AutocallableLifecycleState.knock_in_anchor`, set at the knock-in from the record whose test fired, carried on
  pricing copies as `_otc_lifecycle_knock_in_anchor`.
- `FxRangeAccrualOption(initial_date=..., tenor=...)`, the contract's start and length, as on the equity product.
- `BaseFxProduct.record_passed_fixings(fx_env)`: FX theta fixes the fixings its bump steps over (a no-op by default).

### Changed
- `AsianOption.time_shift` (theta) records, at the held spot, a dated record its bump steps over; the bumped price is
  unchanged (pricing filled the same spot) and theta on a fixing day no longer warns `MissingFixingWarning`.
  On a numeric record schedule it now also moves every record's `observation_time` with the maturity (snapped to
  0.0 when zero, as `with_time_elapsed`), so numeric record Asian theta equals the dated twin's (spec fixture, day 3:
  +0.00109532 -> -0.03081166; it used to age the maturity only).
- A fixing at exactly t = 0 (today's, unprinted) that pricing decides at the current spot no longer counts as
  missing: `MissingFixingWarning` names only strictly past fills (`RangeAccrualOption`, `FxRangeAccrualOption` and
  its variants, `AsianOption`; spec 2026-10-09 run fixings §3.2). Prices are unchanged.
- The equity `BacktestEngine` records a range accrual's / record Asian's fixing once, at the spot of the run day it
  was live, when the clock moves past it (`BacktestEngine.run_fixings`), instead of re-deciding it at every later
  day's spot; numeric contracts age with their records (no more constant-maturity "no roll rule" warning for them).
  A run whose instants carry a time of day (`frequency="H"`) and holds such a product is refused. Measured on the
  spec fixture: a dated range accrual on day 10, 2.744374 -> 4.649441 (= the contract with the day-4 fixing recorded).
- `DynamicScenarioEngine` (equity): the same run-fixings recording on a dated path (`DynamicScenarioEngine.run_fixings`);
  registered on the working copy, so the input portfolio is never touched; numeric range accruals / record Asians now
  age on a dated path (they were frozen). An undated path records nothing; a dated path (or an environment) whose
  dates carry a time of day and that holds such a product is refused (R8), and so is a path whose `start_date`
  precedes the environment's valuation date ("the run clock cannot go back", R9).
- A run holding a NUMERIC range accrual / record Asian / FX range accrual that continues past the contract's maturity
  now raises `ValidationError` on the first instant past it, as the dated twin already did (measured: equity
  backtest, equity dynamic scenario, FX backtest); it used to run on with frozen values. Settlement in runs is a
  follow-up.
- `FXBacktestEngine` and `FXDynamicScenarioEngine` record FX range accrual fixings (domestic, foreign, quanto) on the
  run day they pass, at the quoted spot (`run_fixings`); numeric FX range accruals age in FX runs (they were frozen).
  FX range accrual on the spec fixture, day 10: 11,385.19 -> 19,005.46. The FX dynamic scenario registers on its
  working copy (the input portfolio is never touched) and records nothing on an undated path. New refusals
  (`ValidationError`) for an FX run holding an FX range accrual: a path instant carrying a time of day (an FX backtest
  row stamped 15:00, an FX dynamic path whose `start_date` has a time) is refused at its first step (R8), and an FX
  dynamic path whose `start_date` precedes the environment's valuation date is refused ("the run clock cannot go
  back", R9). A book without one runs as before.
- The replay book records a fixing on the date it passes (at that date's spot) and ages a numeric record Asian's
  records with its maturity (spec fixture day 4: numeric 7.807415 -> 7.686469 = the dated twin).
- PnL explain: a range accrual / Asian fixing that passes inside the step (dated t0, or strictly between t0 and t1)
  is decided once at t0's spot for every revaluation of the pair; the t1 snapshot's own decision (its record, or its
  fill at t1's spot) shows in the `fixing_refill` row, now emitted whenever the pair decided a fixing (label: 'fixing
  decided by the pair at t0's spot, decided differently at t1', was 'missing fixing re-filled'). A t1 contract that
  records a fixing the t0 contract left open is no longer refused (a t0-recorded outcome that changes still is), and
  an aged numeric contract passes the roll check (`check_contract_roll(..., pricing_env_t1=...)`). The equity
  `BacktestEngine` explain recorder no longer declares a run-fixings product unrolled (`contract_roll_days=0`): the
  run clock aged it. An equity backtest or replay with `pnl_explain` holding a range accrual / record Asian whose
  fixing passes mid-run explains again (it raised "contract replacement is not a time step"), with a
  `fixing_refill` of exactly 0.0. Spec probe P1: pv_alive_t1 2.744374 -> 4.649441, `fixing_refill` −1.905067,
  waterfall spot −3.126644 -> −1.221578, total unchanged −2.962925.
- A dated contract whose annualized cash needs an accrual start but has no `initial_date` is refused at
  construction (R5); it used to accrue from the valuation date. A contract with no annualized leg is unchanged.
- A `BUSINESS_DAYS`-annualized contract without `accrual_calendar` / `accrual_bus_days_in_year` is refused.
- A numeric schedule with `initial_date` accrues its quoted fraction instead of YF(`initial_date` -> valuation) plus
  the remaining time.
- Periods are derived from the contract's terms when it is built. Changing an accrual term on a built product
  (`initial_date` of a dated contract, a day count, `accrual_factors`, `fixed_coupon_year_fraction`, `tenor`,
  the accrual end, the settlement convention under `SETTLEMENT`, the accrual calendar) needs `product.validate()`,
  which derives the affected periods again (a re-dated record, a new maturity or a new last fixing too); pricing
  before that raises instead of paying the old cash. A schedule set on a built product likewise needs `validate()`.
  `accrued_offset` and `first_period_start` are ageing bookkeeping, read once when periods are derived.
- The intraday twin carries its source records' periods: fixings at different times of day are no longer refused,
  and the twin no longer back-solves `accrual_factors` or banks one `accrued_offset`.
- Monte Carlo REBASED KO-reset reads each path's knock-in anchor for the knock-out accrual and the knocked-in tenor
  (an annualized V1 used to raise).
- An annualized DATED range accrual with neither `initial_date` nor `tenor` is refused at construction, on both
  asset classes; it used to accrue from the valuation date. So is an annualized NUMERIC one booked mid-life (a
  fixing at a negative `observation_time`) without `tenor`: its `maturity` is only the remaining life. Unannualized
  contracts, and numeric ones aged by `time_shift` (which banks the elapsed time), are unaffected.
- A past fixing without its recorded outcome no longer raises. Equity `RangeAccrualOption` and
  `FxRangeAccrualOption` (and the FX foreign / quanto variants) decide a past record without `observed_in_range`
  at the environment's current spot through the product's own range test (per-record barriers, reverse mode); an
  `AsianOption` past record without `observed_price` takes the current spot. One
  `quantark.util.fixings.MissingFixingWarning` (a `UserWarning`) per pricing call names every fixing filled; the
  product is not changed, so pricing equals pricing the same contract with those fixings recorded at the spot. A
  future fixing that carries an outcome is still refused. `resolve_observations(..., warn_missing_fixings=False)`
  skips the warning for a caller that has already warned.
- **An unset `is_annualized_rebate` now follows `is_annualized`** (default True), as the knock-out coupon always
  did; it used to mean a flat rebate. The knock-in payoff is unchanged: an unset `is_annualized_ki` stays flat. An
  explicit True/False always wins. One resolver, `accrual.leg_annualized(product, "ko" | "rebate" | "knock_in")`,
  answers every cash reader (snowball, Phoenix and KO-reset V0/V1, the terminal periods, QUAD V2 terminal
  breaks, the Monte Carlo REBASED V1 tenor, the snowball PDE V0 rows, the KO-reset lifecycle tenors). Every
  factory default contract whose tenor is not one year moves: the V0 rebate is scaled by the tenor. Measured
  (`create_standard_snowball`, rebate = ko_rate 15%, QUAD at spot 100, vol 20%, r 2%): T=0.5 V0 15 -> 7.5, price
  5.413997 -> 3.107317; T=2 V0 15 -> 30, price 0.810672 -> 1.217247; T=3 V0 15 -> 45, price 1.517037 ->
  1.677312; one-year contracts, Phoenix factory defaults (no rebate) and KO-reset schedule 1 of one year are
  unchanged. A certain-V0 contract prices at its closed form (100 + rebate x tenor) e^{-rT} on QUAD, QUAD V2, MC
  and PDE (`test/accrual_time/test_leg_flags.py`). Re-pinned: the replay goldens (13 frames; a 6-day snowball's
  2% rebate falls from 200 to 3.29 per contract: initial value -10131.703047 -> -9974.988063 on `scalar_bsm`,
  see `test/replay_golden/README.md`) and the carry-identity QUAD pins. The modelvalidation snowball builders
  (`equity_snowball.make_snowball`, `equity_snowball_vol.make_snowball`) state `is_annualized_rebate=False`: the
  banked certificates keep the terms they were certified on (user decision 2026-10-09, no re-certification),
  every anchor reproduces exactly, and the test cases built by them (the `pde_statistics` references, the
  first-passage benchmarks) keep their original banks.
- `SnowballOption` no longer refuses an annualized contract with neither `tenor`, `initial_date` nor
  `settlement_date` at construction. R5 is the one rule, read on the effective flags: a dated contract whose
  annualized cash needs an accrual start without `initial_date` is refused (now including one whose rebate
  annualizes by default, e.g. dated by `exercise_date` alone); a numeric contract accrues on its own clock
  (`accrued_offset` + maturity) and is accepted.
- A filled missing past fixing is history (user decision 2026-10-09): it is decided ONCE, at the base valuation's
  spot, and no bump or scenario shock moves it. Only a STRICTLY past fixing is history
  (`quantark.util.fixings.is_history`); one at exactly t = 0 (today, unprinted) is live and moves with spot bumps
  and shocks as before (pricing still decides it at the current spot). `quantark.util.fixings.freeze_missing_fixings(product, env)`
  returns a copy with those fixings recorded (`with_missing_fixings_filled`; `missing_fixing_fills` /
  `with_fixings_recorded` on `RangeAccrualOption`, `AsianOption`, `FxRangeAccrualOption` and its subclasses) and
  warns once; `freeze_portfolio_fixings(portfolio)` does it per position. Applied at: every public
  `GreeksCalculator` method (so `calculate`, bucketed, SIMM and parametric VaR greeks), `BaseEngine.calculate_greeks`
  and `calculate_spot_greeks_curve` (ladder rows keep the base fill, as the engine-grid ladder does),
  `BaseFxEngine.calculate_greeks` (FX calculator, two-rate greeks), `EquityPosition.get_trade_greeks` /
  `get_trade_risk`, the execution-framework spot cells, equity and FX VaR (parametric, historical, Monte Carlo,
  incremental), equity and FX stress tests, SA-CVA (trades and hedges), the surface-shock pipeline (one warning),
  the replay book's carry reprices, and PnL explain. In PnL explain the waterfall's shocked states, the
  alive-at-t1 contract and the Taylor greeks price t0's fill, while `pv_t1` stays the t1 snapshot's OWN value so
  consecutive daily explains chain; a fixing decided differently at t1 shows as a new `Factor.FIXING_REFILL`
  row ("missing fixing re-filled"), present only when t0 had a missing fixing. Three daily explains of a range
  accrual at 109.5 -> 111 -> 111.5 (quantity 2) sum to the PV change exactly (-3.8965), with a re-fill row of
  -3.8054 on the first day. The roll check runs on the contracts as supplied, before any fill. Multi-day runs (equity and FX dynamic scenarios, equity and
  FX backtests, the replay book) fill once at the FIRST valuation and keep it; a fixing that passes during a run
  is recorded once at the spot of the day it was live (run fixings, below). Delta near a barrier no longer carries the
  fill's jump: a numeric range accrual at spot 109.5 under a 110 barrier, delta -0.8914 -> -0.0227 and gamma
  -1.588 -> -0.00106 (equal to the contract with the fixing recorded); an FX range accrual at 1.2995 under 1.30,
  delta -5.906e6 -> -4.991e4. Prices are unchanged (48 prices bit-identical to main); the warning fires once per
  base valuation instead of once per repricing.
- A dated equity option valued ON its exercise date (the same instant) prices its settled value, identical to its
  numeric twin aged to `maturity == 0.0` (user ruling 2026-10-09, option (a)): `get_maturity` returns 0.0 there
  instead of raising `Valuation date (...) must be before exercise date (...)` (`BaseEquityOption`, `OneTouchOption`,
  `DoubleOneTouchOption`, through the new `time_to_exercise`). The settled value is the payoff at the valuation spot
  with every past fixing or event as recorded and a fixing dated the exercise date live (t = 0, fixed at the spot).
  Measured on 16 families x their engines (188 cells, `docs/superpowers/specs/2026-10-09-dated-expiry-settles-evidence/`):
  on the 154 cells whose numeric twin prices, dated == numeric == the hand settled value to 1e-12, e.g. a range accrual (notional 100, 10%, every fixing in range)
  10.0, a snowball knocked out at 104 15.0. A valuation strictly AFTER the exercise date still raises (now "must be
  on or before exercise date"). Routes with no T = 0 value in either form keep refusing, with `TODO(dated-expiry)`:
  the Heston analytical / MC / PDE and local-vol MC European engines ("maturity must be positive"), the local-vol,
  Heston and SLV barrier MC and Heston / SLV barrier PDE (`t_grid must be strictly increasing`). The replay /
  ensemble dividend-pillar readout on a dated maturity day is unchanged (1e-8).
- QUAD V2 settles on the exercise date (user ruling Q1, 2026-10-10): its compile re-validates its copy with the new
  `validate(allow_zero_maturity=True)` (`BaseEquityOption`, `SnowballOption`, `PhoenixOption`,
  `KnockOutResetSnowballOption`), so a numeric snowball / Phoenix / KO-reset aged to exactly `maturity == 0.0` prices
  its settled value (was `Maturity must be positive, got 0.0`), and so does the dated twin on its exercise date (it
  raised). Snowball KO 103 / continuous KI 75: 15, 15, -30 at spots 104, 90, 70; Phoenix (memory and not): 1, 1, 0,
  -40 at 104, 90, 75, 60. Constructing a contract with maturity 0.0 is still refused, and a negative maturity is
  refused everywhere. The knocked-in KO-reset's numeric twin now settles on V2 (3 at 104, -10 at 90) while the dated
  KO-reset keeps its pre-existing refusal on V2 (a schedule-1 record in the past fails settlement resolution, at any
  valuation date).
- The vol-model European and barrier routes settle on the exercise date (W9, user ruling 2026-10-10): at T = 0 (a
  dated contract on its exercise instant, or the numeric twin at `maturity == 0.0`) each short-circuits before its
  model runs to the expiry code of its Black-Scholes counterpart of the same method, instead of raising `maturity must
  be positive` / `t_grid must be strictly increasing`. Heston analytical, Heston / local-vol / SLV MC European: the
  new `settlement_support.expiry_terminal_value` (the BS analytical and MC engines' expiry branch, now shared);
  Heston / SLV European PDE: `european_expiry_outcome` (the BS European PDE route); local-vol / Heston / SLV barrier
  MC: `barrier_expiry_value` (factored from `BarrierOptionMCEngine._price_expiry_payoff`); Heston / SLV barrier PDE:
  `barrier_expiry_outcome` (the BS barrier PDE route). The nine `TODO(dated-expiry)` markers are gone. European call
  K100: 10 at spot 110, 0 at 90; up-and-out call K100 B130 rebate 5: 10, 0, 5 at 110, 90, 135; a two-day settlement
  lag discounts them (9.99836 at 110). `calculate_greeks` on the exercise date returns the counterpart's keys and
  values (price, delta, gamma; the analytical / MC routes by bump-and-reprice, the PDE routes from the payoff slope)
  where it raised. A valuation after the exercise instant (the next day, or later on the exercise day) and a numeric
  maturity below 0 still refuse. Not changed: a knock-in carried as triggered (`_otc_lifecycle_knocked_in`) is not
  read at T = 0 by any Black-Scholes barrier route (up-and-in call B130 rebate 5 at spot 110: the rebate 5, not the
  payoff 10), so the vol-model barrier routes now return the same (open question in the W9 report; fixed below,
  "A knock-in carried as triggered decides the exercise date"). Evidence:
  `docs/superpowers/specs/2026-10-10-volmodel-expiry-settles-evidence/`.
- A numeric run-fixings participant (range accrual, record Asian) is no longer floored at a 1e-8 maturity in the
  replay (run-fixings plan Ruling 12, superseded now the dated twin settles on its exercise date): on its maturity
  day it is exactly 0.0 and prices its dated twin's settled value (numeric Asian pricing copy on T: 6.9999999965 ->
  7.0, = the dated twin); past it the maturity is negative and pricing refuses, as the dated twin past its date
  does. The replay dividend-pillar readout maps 0.0 / negative to the same 1e-8 pillar (unchanged). PnL explain's
  `check_contract_roll` expects a participant's `with_time_elapsed` maturity (snapped, not floored), so an explain
  step ending on its maturity explains, equal to the dated twin (it raised `alive product maturity 0.0 is not ...
  rolled ...`). A backtest of the twins through T values both at the settled value on T; a run to T + 1 is refused
  for both, as before.
- Asian MC at maturity 0.0 averages the recorded fixings and the live one at the spot (as the analytical engine
  does) instead of the spot alone: 11 fixings 101..111 recorded, spot 110, strike 100: 10.0 -> 6.3333.
- Accumulator analytical and MC at maturity 0.0 settle the observation at the maturity instant on the spot (as at
  every other instant, `settle_valuation_instant`) instead of dropping it: spot 100, strike 95, one share: 0 -> 5;
  `test_accumulator_mc_engine` expiry cases 5.0 -> 10.0 and -5.0 -> -15.0.
- The DCN settles on its final observation date (W10, user rulings 2026-10-10 on
  `docs/superpowers/specs/2026-10-10-dcn-final-observation-proposal.md`): every DCN route (GBM, local-vol, Heston,
  QE and coupled-coarse MC, the execution-session MC adapters, the GBM, local-vol and Heston PDE) prices the settled
  final row from the new neutral `quantark.asset.equity.engine.dcn_valuation` (`settle_final_observation`) instead of
  raising `DCN grid has no steps`, `tau must be positive` or `t_grid must be a 1D array`. The row is decided on
  today's spot, pointwise at barrier ties on every route: knock-in = carried OR spot <= KI; knock-out (spot >= KO,
  either state) pays the KO coupon; the fixed coupon pays when not knocked out, not knocked in and spot >= the
  coupon barrier; the loss leg pays when knocked in and not knocked out; each flow at its own payment-date DF (1 when
  paid that day). DCN_A (N 1e6, S0 6000, lags 2 SSE days): 9996.099391 for a coupon or knock-out, -433164.306946
  knocked in at 4000; the MC result's legs / event statistics are the 0/1 decisions with `std_error` and
  `expected_life_years` 0.0; PDE greeks are the settled row's slope (0 at a tie), gamma 0. Sessions answer the
  final date natively (the kernel's new optional adapter hook `answers_natively(prepared)`: no batch plan). The
  cash-Greeks report's date-rolled `theta_1d` on the penultimate day now prices (settled - today) where it raised.
  The row equals the MC path kernel's one-column row bit for bit and is the limit of both engines as the last step
  shrinks (evidence: `docs/superpowers/specs/2026-10-10-dcn-final-observation-evidence/`). Not changed: decided but
  unpaid coupons are not carried (W14), a knocked-out DCN is assumed alive (W14), `settlement_convention` is ignored
  (W15), a valuation after the final date still raises.
- A DCN's `settlement_convention` times ALL of its cash (W15, spec I2 / amendment A2), where it was ignored. New
  `DCNOption.payment_dates() -> DCNPaymentDates` (`coupon_payment_dates`, `ko_payment_dates` per schedule row,
  `loss_payment_date`; no pricing-env argument) is the single timing source: without a convention it returns the
  schedule's offset dates and `settlement_date`; with one, coupon and KO pay from their observation date and the loss
  leg from the final observation date via the new public `quantark.asset.equity.settlement.payment_date_by_convention
  (convention, determination_date, calendar)`, on the convention's calendar, else the schedule's (any calendar, not
  only SSE). `build_dcn_grid_context` reads it, so every route honours it: GBM / local-vol / Heston / QE /
  coupled-coarse MC, the execution-session adapters, the GBM / local-vol / Heston PDE and the valuation-date row
  (`dcn_valuation`); the MC event statistics' payment dates are C4's. `SettlementResolver._resolve_payment` now
  derives day-lag dates through the same helper (bitwise for every product). A `YEAR_FRACTION` lag on a DCN raises
  ValidationError (in `validate()`, and in `payment_dates()` if set afterwards): a dated DCN pays on dates.
  Moved values: none with `settlement_convention=None` (every DCN route byte-identical before/after, 28 pins);
  `test_autocallable_mc_settlement::test_dcn_explicit_payment_dates_remain_authoritative` (a YEAR_FRACTION
  convention silently ignored) became `test_dcn_year_fraction_convention_is_refused` (ValidationError).
- Every DCN route prices a lifecycle state (W14a, spec I1 / C1 / C2 / A1). New
  `quantark.asset.equity.lifecycle.DCNLifecycleState` (`valuation_point`, `ledger`, `knocked_in`, `knocked_out`,
  `ko_date`, `matured`, `observed_row_indices`, `ki_date`; `alive`, `realized_cashflows`). GBM / local-vol / Heston /
  QE / coupled-coarse MC, the GBM / local-vol / Heston PDE and their execution-session adapters (batch MC: pending PV
  added in the reduction; terminal states and the final date answered natively; PDE session incl. grid rows) declare
  `supports_lifecycle_state` and take `lifecycle_state=`: the engine value + the ledger's pending receivable PV; a
  knocked-out or matured state prices exactly that PV (no simulation / march, greeks 0). Refused (ValidationError):
  `state.knocked_in != product.knocked_in_at_valuation`, a valuation point other than the valuation date, an undated
  flow, a flow determined after the valuation date, and (alive states) a flow determined on a date that still has a
  monthly row in the product's schedule; another state type is a CapabilityError. New
  `DCNSchedule.without_observed_through(d)` is the post-close view (rows and daily-KI dates <= d dropped); the grid
  context's new `valuation_monitored` marks a valuation instant that is no knock-in test (t = 0 prepended; MC, PDE,
  Heston PDE and grid rows then skip today's KI test). Pre-close == post-close (engine + ledger) to 1e-10 on GBM MC
  (same seed, same grid) and PDE on coupon, knock-out, knock-in and final days. A7'(a): `build_dcn_schedule` accepts a
  valuation date that is not a trading day of the schedule calendar (rows / KI dates on or after it, times ACT/365F
  from it, no KI test until the next trading close); the cash-Greeks `theta_1d` rolls to the next trading day from it.
  DCN_A at 5500 Fri..Tue 2024-06-07..11 (PDE standard): -47083.40, -46208.98, -45415.70, -44628.59, -43819.88.
  Moved values: none with `lifecycle_state=None` on a trading day (40 pins byte-identical, every route, greeks, grid
  rows, sessions, cash-Greeks report); closed-day valuations raise -> price;
  `test_dcn_integration_seams::test_schedule_rejects_non_trading_valuation_date` became
  `test_schedule_accepts_non_trading_valuation_date`.
- DCNs age through runs (W14b, spec I1 / C3 / A1 / A4-A6 / A7'). New `quantark.asset.equity.lifecycle.DCNLifecycleTracker`
  (`tracker_for` returns it for a `DCNOption`; `PortfolioLifecycleManager.register_positions` tracks DCN positions in
  the equity backtest and the dynamic scenario; the replay builds it per product). Each close decides the BOOKED rows:
  daily KI, then KO (ends the note), else the coupon, and on the final row the loss leg of a knocked-in survivor; cash
  is a dated `RealizedCashflow` (`dcn:<kind>:<date>`, quantity x signed) payable on `DCNOption.payment_dates()`. Any run
  calendar: every run day prices; a schedule date the run never visits is decided at the previous run instant's spot
  (`gap=True` on the event and the ledger flow, W6 rule); dates before the run start are history (signed rule 3); a
  first run day after an unvisited in-run date refused (`TODO(dcn-run-start)`; superseded by ruling R31, T17 CP6: dates
  before the FIRST RUN INSTANT are history -- the row observed, its coupon not owed, no KI / KO state assumed -- so a
  run whose first market day is after the start completes; raise -> run, no moved values; and a first run instant
  after the final observation retires the note as matured before the run, as W1 -- the warning, matured and settled,
  no cash, valued at zero -- where it stayed alive and its pricing copy refused; ruling R32: before any observation,
  rows before the requested day are history for the pre-close copy). Pricing copies: pre-close
  `product_for_pricing(d, env, initial=True)`, post-close `.without_observed_through(d)`; the runs value the contingent
  claim plus the ledger (MTM + pending PV + paid cash). `DCNLifecycleState` gains `pending_settlement_cashflow`,
  `settlement_date`, `settled`, `settle()`, `advance_to()`, `mark_matured_before_run()`; `ProductReplay` reads a DCN's
  ledger for its receivable PV and payments (A4; the autocallable path unchanged). Replay builds DCN engines
  (`create_autocallable_engine`: `DCNMCEngine` from `MCParams` num_paths / seed / use_qmc / use_antithetic,
  `DCNPDEEngine`; `create_vol_model_engine(product=)`: local-vol / Heston MC and PDE; SLV refused) and its config
  admits a DCN under `vol_model` localvol / heston. Moved values: DCN runs (they priced the booked contract statically
  every day, never deciding a row; the replay raised `PDEEngine settlement support is none`). DCN_A booked 2023-03-27,
  spot 5500 / 4400 on 04-11..12 / 6100 from 05-04, PDE fast, dynamic scenario portfolio value: 04-03 -160297.41 ->
  -156942.55 (coupon 10000 receivable), 04-11 -455784.67 -> -444194.19 (knocked in, coupon paid), 05-04 -10724.37 ->
  19996.10 (knocked out, position closed), 05-12 -10724.37 -> 20000.00; the equity backtest moves identically; the
  replay now runs (05-12 paid cash 20000). Replay goldens unchanged. Refused (ValidationError): a run instant with a
  time of day (signed W6 Q2(a), `TODO(run-fixings)`), PnL explain of a run holding a DCN (`TODO(dcn-explain)`), and a
  pre-close copy while an in-run schedule date before it is undecided; a dead or expired note's pricing copy is
  terminal (`get_maturity` 0.0), never a refusal.

### Fixed
- `HestonDCNPDESolver` events are the 1D DCN route's dual-cell projection on an unpinned grid (T14 / I10). The 2D
  DCN applied coupon / KO / KI as whole-node masks on a grid with a node pinned onto each level, so each trigger sat
  up to half a cell off its level (an O(h) shift whose coefficient follows the level's position in its cell; the
  KI and KO nodes were also left out of their inclusive masks by 1-ULP `exp(log b)` noise), and the price was
  erratic in `n_x`: DCN_A Feller Heston, n_v 101, 301 -> 401 -> 501 moved +4,476 / -940 per 1M, d04's 601 -> 1201
  -> 2401 +2.4 / +414. The 1D DCN route shows the same ladder under its `"nodal"` masks and a ±10 one under its
  default. New `event_projection` (None = CELL_AVERAGE; `"nodal"` is the old discretization bit
  for bit) and `HestonSLVADICore(pin_critical_levels=True)` (False skips the level pinning; the default path is
  unchanged for every other route). After: the nested ladder 301 / 601 / 1201 / 2401 / 4801 is monotone (-56, -87,
  -53, -18), non-nested levels wiggle within ±75; Heston QE-M MC (2^21 paths) -92,980 ± 100 vs the time-converged
  PDE ~ -92,910 (Aitken extrapolation of the e09 time ladder; 0.7 s.e. from the MC). Evidence: `docs/superpowers/specs/2026-10-13-dcn-nx-ladder-evidence/`. Moved values: Heston DCN
  PDE prices only (no anchor or golden prices one): DCN_A Feller at the default (301, 101) -97,286.0115 ->
  -92,971.8705, at (151, 41) -90,111.4178 -> -92,001.9490 per 1M notional.
- The autocallable MC time grid holds one node per instant (W17a, approved 2026-10-10). The snowball, KO-reset and
  Phoenix MC grids (`SnowballMCEngine._build_time_grid` / `_build_time_grid_ko_reset`, `PhoenixMCEngine.
  _build_time_grid`, inherited by the local-vol / Heston / QE / SLV snowball and Phoenix MC) were the sorted Python
  `set` of their times, so one instant spelled two ways an ulp apart -- a knock-out record against a continuous-KI
  `np.linspace` node (a dated record's `(d - a) / 365`, an aged numeric one's `d / 365 - a / 365`, `T * j / n`),
  a REBASED `t_ki + offset`, or KI dates rounded to 10 decimals -- survived as two nodes ~1e-16 years apart. The
  empty step carries no variance but adds a column of random draws and re-deals every draw after it, so a dated
  contract and its aged numeric twin priced on two independent samples (W17 diagnosis: day 92 of a 256-path snowball
  12.08 vs 11.34; at 1M paths the gap is within one standard error, both forms agree with the PDE / QUAD V2).
  New `quantark.util.numerical.merge_time_nodes(times, tol=Tolerance.ZERO, pin=T)` collapses every run of nodes
  closer than `tol` into its LAST member (a maturity within `tol` is pinned back to exactly `T`) and `node_index`
  maps a time to the first merged node strictly after `t - tol`; the three grid builders use them. A grid with no
  such run is bit-identical. Moved: only MC values whose grid held such a run, each a fresh draw of the same law --
  on the W12 snowball / Phoenix, 6 dated and 8 numeric valuation days of 364 (e.g. BS snowball day 92 numeric,
  256 paths seed 7: 10.756027 -> 11.461036 = the dated value), the continuous-KI KO-reset on days 26 / 65 / 78 / 82,
  `test_ko_reset_event_stats_shape[Rebased]` pv 98.1788 -> 97.5565 (shape-only test) and the live gate
  `test/pde_grid/test_anchor_certification.py::test_snowball_pde_vs_mc[False]` MC 96.645369 -> 96.621477 (|PDE - MC|
  0.0213 -> 0.0026, gate 0.2). No golden or certificate anchor moved (probe over `test/modelvalidation`,
  `test/mo_volmodels`, `test/pde_grid` and every autocallable-MC test file). Dated == numeric twin now within
  8.9e-15 on every day and route measured. The range-accrual and Asian MC grids use the same set union but hold no
  such pair (only observation times against `T`, spelled alike in both forms; every day measured): unchanged. New
  `test/test_mc_time_node_merge.py`. Evidence: `docs/superpowers/specs/2026-10-10-snowball-mc-twin-diagnosis/`
  (`moved_values.txt`).
- The local-vol, Heston and SLV snowball MC price a KO-reset with a CONTINUOUS knock-in (W17 side ticket). The
  KO-reset path hands the Brownian bridge its paths cut at the knock-in horizon (schedule 1) while these engines
  record the per-step log-variance over the whole grid, so the shape check raised `PricingError` ("recorded step
  variance ... does not describe these paths") on every such contract. The caller now names the horizon
  (`horizon_steps`) and the record is sliced to it; any other mismatch still raises. Flat-IV local-vol MC agrees with
  the BS MC within 4 standard errors; dated == numeric twin within 1e-10. Moved values: none (raise -> price). New
  `test/test_ko_reset_vol_mc_continuous_ki.py`.
- A dated autocallable whose discrete knock-in records lie in the past prices like its numeric twin (W11). The
  knock-in resolvers (`SnowballOption.resolve_ki_observations`, inherited by `KnockOutResetSnowballOption`, and
  `PhoenixOption.resolve_ki_observations`) resolved EVERY record, so a dated record before the valuation date asked
  the settlement resolver for a year fraction from the valuation date back to it (`calculate_year_fraction`,
  `day_counter.py:86`: "end_date (2025-02-04) must be after start_date (...)"), while the knock-out resolvers already
  kept only the records not before the valuation date and the numeric twin's `time_shift` had dropped them. Every
  engine reads that resolver (QUAD V2's compile, the PDE's valuation decision and knock-in rows, QUAD V1's valuation
  decision, the MC time grid), so a dated KO-reset with schedule-1 records in the past was refused mid-life on all of
  them (QUAD V2 and the PDE also when knocked in, QUAD V2 also on the exercise date), and so was a dated snowball or
  Phoenix with a DISCRETE knock-in schedule (the continuous-KI fixtures never met it). The knock-in resolvers now keep
  the same records as the knock-out ones (`observation_schedule.active_record_indices`; new
  `active_ki_source_indices` on both products): a passed record is history, its outcome the carried knock-in state,
  never resolved forward. The KO-reset MC grid maps each kept record to its source position in the contract's
  schedule (the knock-in anchor's `source_index`). The KO-reset contract rules are unchanged (a past schedule 1 not
  knocked in still refuses, `require_alive`, on both forms). Moved: 363 dated values raised -> price / delta / gamma
  (KO-reset ABSOLUTE at days 100-365 alive and knocked in on MC / PDE / QUAD V1 / V2 / local-vol / Heston / SLV MC,
  e.g. QUAD V2 292 days in knocked in at spot 100: raised -> 2.43400 = the numeric twin; QUAD V2 on the exercise date
  3.0 at 104, -10.0 at 90; discrete-KI snowball and Phoenix on MC / PDE / QUAD V1 / V2 / local-vol MC and PDE /
  Heston / QE / SLV MC); no numeric value and no golden / certificate anchor moved. Dated == numeric twin (price,
  delta, gamma) within 7e-13. Intraday event building (`quantark/intraday/events.py`) now refuses a schedule env that
  drops a knock-in record, as it already refused one that drops a knock-out record (it indexes the source records by
  position). Not changed: the 2D Heston / SLV snowball and Phoenix PDE still refuse a dated knock-out schedule before
  maturity (pre-existing, a different step). New `test/test_ko_reset_past_records.py`; the pinned
  splits are gone (`test_dated_expiry_settles` `DATED_REFUSED_ANY_DATE`, `test_expiry_carried_knock_in` KO-reset
  entries in `AUTOCALL_REFUSED` / `AUTOCALL_BEFORE_REFUSED`). Evidence:
  `docs/superpowers/specs/2026-10-10-ko-reset-past-records-evidence/`.
- `EuropeanMCEngine`'s no-arbitrage lower-bound check is statistical (W13, user decision 2026-10-10; diagnosis
  `docs/superpowers/specs/2026-10-10-mc-lower-bound-diagnosis/`). A deep in-the-money European near expiry is worth
  its discounted intrinsic bound (Black-Scholes minus the bound is 0.0 in double precision one day out), so the
  unbiased estimate fell below the bound about half the time and the hard `price < bound - 1e-6` check refused it
  (38-62 % of seeds at every path count and method; no estimator bias measured, |bias| < 3e-6). The engine now
  refuses only an estimate more than `max(Tolerance.PRECISION, 4 * std_error)` below the bound
  (`EuropeanMCEngine.LOWER_BOUND_STD_ERRORS = 4.0`; the std error it reports, in contract units) and keeps the hard
  `Tolerance.PRECISION` (1e-6) floor when the std error is zero or undefined. The estimator is unchanged: every price
  that passed before is bit-identical (1162 of 1162 grid cells); only refused cells now price (182 of 182, raised ->
  their own estimate; `moved_values.txt`). The BS barrier MC prices the day-before carried knock-in down-in put
  through this engine: 9.993453009252772 at spot 90 and 24.993174367657804 at spot 75 (dated form; the numeric twin
  differs in the last digits), so `("mc", "down_in_put")` leaves `_BEFORE_REFUSED` in
  `test_expiry_carried_knock_in.py` and its day-before continuity assertion runs. Under antithetic and plain QMC the
  reported std error is the iid formula and overstates the real error (up to about 800x measured), so the check is
  loose there; the std error formula, `EuropeanQuadEngine`'s check and RQMC seeding are unchanged (open questions in
  the diagnosis).
- Antithetic MC runs report the pair-mean standard error (W16, signed W13 follow-up 1, spec I3/A9). Within each batch
  the samples are `[z; -z][:n]`; with `m = ceil(n/2)` the pairs are `(i, m+i)`, an odd `n` leaves the unpaired
  `z[m-1]` out of the s.e. only, the pair means of all batches pool into one unit set, and the s.e. is
  `std(units, ddof=1)/sqrt(#units)`. New `quantark.montecarlo.std_error` (`antithetic_pair_means`,
  `mean_and_std_error`, `sample_std_error`) is used by the European, digital, Asian, range-accrual, American, barrier
  (continuous, discrete, expiry), accumulator, single and double sharkfin MC engines, the vol-model (local vol,
  Heston, QE, Heston-SLV) snowball and Phoenix engines, the DCN GBM / local-vol / Heston engines and their execution
  adapters, and the FX barrier, sharkfin, range-accrual, TARF and TARN MC engines. Plain QMC keeps the iid formula,
  now documented as not a valid error bar for deterministic QMC (use `RANDOMIZED_QUASI`). The plain GBM snowball and
  Phoenix engines draw iid paths whatever `use_antithetic` says and keep the iid s.e.; the Dask batch route of the
  snowball and Phoenix engines reduces sum / sum-of-squares and keeps the iid s.e. (`# TODO(w16)`). The W13a lower-bound
  check inherits the corrected s.e.: no refusal flips (0 of 300 seeds refused at spots 90 and 75, 1 and 5 days, 256
  antithetic paths, before and after; the one-day deep-ITM put s.e. falls 0.0636 to 0.0007). Moved values: reported
  standard errors only (examples, 2000 antithetic paths, fixed seed: European call 0.3435 -> 0.2671, cash-or-nothing
  0.1085 -> 0.0193, American put 0.1991 -> 0.1115, FX TARF 0.00722 -> 0.00052, DCN GBM 3220 -> 2680, Phoenix local-vol
  0.01441 -> 0.01022; range accrual rises 353 -> 497 because its payoff is even in z and the iid formula understated
  it); prices bit-identical. New `test/test_mc_std_error_antithetic.py`.
- Proposals (docs only, no code; each ends with open questions for sign-off):
  `docs/superpowers/specs/2026-10-13-last-day-theta-proposal.md`,
  `docs/superpowers/specs/2026-10-13-standardized-pricing-result-proposal.md`,
  `docs/superpowers/specs/2026-10-13-fixing-time-proposal.md`. Moved values: none.
- RQMC seed spacing is documented (W16, signed W13 follow-up 3): batch `b` scrambles with `base_seed + b`, so
  replications whose base seeds sit closer than their batch count share scrambles. `SobolNormalGenerator._make_engine`
  and `docs/execution/policy-guide.md` ("RQMC seeds and replications") state the rule: space base seeds by at least
  the batch count, or derive them from a hash / `SeedSequence`. Seeding unchanged. Moved values: none.
- `EuropeanQuadEngine` lower-bound check scales with the notional (W16, signed W13 follow-up 2: measure first).
  Near-expiry measurement (`docs/superpowers/specs/2026-10-13-quad-near-expiry-evidence/`, 1680 cells to 1e-9 y): no
  refusals at ordinary scale, but the ABSOLUTE `1e-4` tolerance refused correct deep-in-the-money prices once the price
  reached ~1e8 (20 false refusals in the scale probe; relative error <= 2e-11). Tolerance is now
  `max(1e-4, Tolerance.RELATIVE * bound)`: every bound below 1e5 unchanged, a price 1% below the bound still refused.
  Moved values: none (refusal -> price at large notionals). Gauss-Legendre ATM error ~4e-3 (all maturities) logged.
- The DCN PDE routes pay a monthly observation that falls on the valuation date (W10b, user ruling Q5 2026-10-10).
  The GBM, local-vol and Heston DCN PDE dropped it (the march never applies t = 0) while every MC route paid it, so
  the PDE value lost the day's coupon overnight. They now decide the row pointwise at the spot through the W10
  helper (`dcn_valuation.observe_valuation_row`), at every valuation date: a knock-out today settles at the
  knock-out coupon (greeks 0, no march), a coupon today is added to the smooth continuation, a knock-in today reads
  the knocked-in surface (as before). DCN_A on 2024-12-03 (month 23): spot 5500 coupon today 9779.74 -> 19777.79
  (MC 19767.09; the day before 19739.93); carried knock-in at 6300, knock-out today -23212.00 -> 9998.05 (= MC).
  The grid spot curve (`calculate_spot_greeks_curve`, `BranchSolution`) folds the same row per level. 90 values
  move on the measured observation dates; nothing else (evidence `moved_values.txt`).
- The vol-model routes resolve DATED observation schedules (W12, user-approved). (1) The Heston and SLV snowball and
  Phoenix PDE (2D ADI) refused a dated knock-out schedule at any date before maturity ("a date-based knock-out
  observation schedule needs a pricing environment to resolve its observation times"): their S-axis grid request
  (`_layer_x_nodes`, and the bump context's frozen layout) never received the environment the 1D adapter stashes.
  Both now call `_stash_grid_request_env` before `grid_request`, as the 1D adapter does. (2) The local-vol, Heston and
  SLV barrier MC and the Heston and SLV barrier PDE read only the legacy `observation_dates`, which a dated schedule
  leaves unset ("discrete monitoring requires observation_dates"). A dated schedule is now resolved against the
  environment exactly as the Black-Scholes barrier MC resolves it (`schedule.resolve(env, default_barrier=,
  default_payoff=, require_single=True, product=)`, new `engine/vol_barrier_schedule.discrete_observation_times`)
  and the resolved times feed the kernels; a numeric schedule is still read from `observation_dates` verbatim. The
  limits are unchanged and identical for both forms: per-observation varying barriers / payoffs, an observation
  after maturity, a record behind the valuation date (refused by the resolution, as on the Black-Scholes routes),
  a delayed pay-at-hit rebate, a memory Phoenix on the 2D PDE. Moved values: dated forms only, raised -> price;
  every numeric value is unmoved (PDE byte book and the modelvalidation anchors bit-identical). The dated twin
  equals the numeric twin aged to the same date (153 of 201 measured values bitwise, worst 3.5e-12 relative: the
  last ulp of the resolved times). New `test/test_vol_dated_schedules.py`; the W9 / carried-KI pins on these
  refusals are removed (`_BEFORE_REFUSED`, `AUTOCALL_BEFORE_REFUSED` in `test/test_expiry_carried_knock_in.py`).
  Evidence: `docs/superpowers/specs/2026-10-10-vol-dated-schedules-evidence/`.
- The vol-model barrier routes price a discrete schedule at its records' own terms, first-hit only (W18, user rulings
  2026-10-10). The local-vol, Heston and SLV barrier MC and the Heston and SLV barrier PDE accept a discrete schedule
  whose records all share one level and payoff, but handed the kernels `product.barrier` and `product.rebate`: a
  schedule whose common level differs from `product.barrier` was priced at the wrong level, silently, numeric and
  dated alike; so was a knock-out's common record payoff. The kernels now receive the records' common level (a
  record without a level takes `product.barrier`, the product's own normalisation) and, on a knock-out, the
  records' common payoff, exactly as the Black-Scholes barrier MC reads them (`_build_discrete_grid`); a knock-in's
  no-hit rebate stays `product.rebate`, as the Black-Scholes MC pays it. New
  `engine/vol_barrier_schedule.schedule_barrier_terms`. The vol kernels monitor first-hit only: a schedule with
  `aggregation_mode` ACCUMULATE, BEST or WORST, which they silently priced as STOP_FIRST_HIT, is now refused
  (`ValidationError`, "aggregation ... is refused"; new `refuse_non_first_hit`, `# TODO(vol-aggregation)`), on both
  forms and including the Heston / SLV PDE deterministic-variance limit. CONTINUOUS / EXPIRY monitoring is
  unaffected. Moved values: level / payoff-differing schedules value -> value (each equal, bitwise, to the contract
  booked at the records' terms), non-first-hit aggregation value -> refused; every schedule at `product.barrier`
  with STOP_FIRST_HIT is unmoved (PDE byte book and the modelvalidation anchors bit-identical). New
  `test/test_vol_barrier_schedule_level.py`. Evidence: `docs/superpowers/specs/2026-10-10-vol-barrier-schedule-level-evidence/`.
- The Black-Scholes barrier MC decides at the valuation instant on the schedule record's own level (W18b, user
  approval 2026-10-10). Two decisions of `BarrierOptionMCEngine` read `product.barrier` / `product.rebate` while
  its discrete path reads each resolved record: (a) the exercise-date value `barrier_expiry_value` (also the
  local-vol / Heston / SLV barrier MC routes' value at T = 0 since W9) and (b) the "hit at valuation" decision
  before simulating. For a DISCRETE schedule both now read the first record at or after the valuation instant (the
  record on the valuation date, else the next observation, the barrier PDE's `_next_observation` rule -- the
  next-observation reading of (b) is superseded by A15 below: between observations nothing is decided now): its level,
  and on a knock-out its payoff (new `valuation_barrier_terms`; `barrier_expiry_value` now takes the pricing
  environment). A record without a level takes `product.barrier`; CONTINUOUS / EXPIRY monitoring and a knock-in's
  no-hit rebate are unchanged. Example: up-and-out call, `barrier` 120, record at 115 on the exercise date, spot 117:
  17.0 -> 5.0 on the BS MC and the three vol MC routes, equal to the BS PDE, the vol PDE routes and the contract
  booked at 115. A record on a mid-life valuation date whose level the spot has crossed now knocks out / in at once,
  as the PDE does (the MC previously simulated and failed on the zero-length step). Moved values: only contracts
  whose record level differs from `product.barrier` (or a knock-out record paying other than the rebate), spot
  between the levels; plain contracts bit-identical. New `test/test_barrier_mc_record_level_at_valuation.py`.
- The Black-Scholes barrier MC prices a DISCRETE schedule with a record on the valuation instant that the spot has
  NOT crossed (it raised "dt_array must have strictly positive entries"). Such a record is decided "not hit" at the
  valuation instant (W18b) and leaves the simulated grid: the price equals, bitwise, the contract aged past that
  record; a sole such record leaves the never-hit value (knock-out: the vanilla; knock-in: the rebate at the
  terminal). Moved values: none (raise -> price).
- A knock-in whose discrete schedule records pay other than `product.rebate` is refused on every route (W18b, user
  ruling 2026-10-10, option (a)). A knock-in's rebate is paid when no observation hits, so no record's payoff can
  apply. The 1D PDE core already refused it (spec 3C §5.2 (a): the Black-Scholes and local-vol barrier PDE, and the
  Heston / SLV PDE deterministic limit); the Black-Scholes MC, the QUAD barrier, the local-vol / Heston / SLV MC and
  the Heston / SLV 2D PDE priced it. The core's rule and message ("a knock-in has one rebate (R); a schedule record
  pays P (spec 3C §5.2 (a))") move to the neutral `engine/barrier_schedule_rules.refuse_knock_in_record_payoff`,
  which every route now calls, in both forms. Unaffected: knock-ins whose records pay the rebate or nothing, and all
  knock-outs. Moved values: those knock-in cells price -> refused only. New
  `test/test_knock_in_record_payoff_refused.py`.
- The analytical barrier routes read the schedule record's level and payoff on a DISCRETE schedule (W18c, spec
  2026-10-13 I4 / A8): `BarrierAnalyticalEngine`, `OneTouchAnalyticalEngine`, `DoubleBarrierOptionAnalyticalEngine`
  and the single / double sharkfin analytical engines read `product.barrier` (`upper_barrier` / `lower_barrier`) in
  their hit pre-checks, immediate knock-out cash and barrier-shift formula. Now a DISCRETE schedule is hit now only by
  a record at the valuation instant, at its own level(s), and an immediate knock-out pays that record's payoff
  (`valuation_barrier_terms`, moved to the neutral `engine/barrier_schedule_rules.py` and still importable from the
  MC module, plus the new `valuation_double_barrier_terms`); the forward closed form (shift, beyond-the-barrier
  first-observation integral, one-touch volatility) uses the level(s) every remaining record (t > 0) shares
  (`forward_barrier_level` / `forward_double_barrier_levels`). Remaining records with different levels have no
  closed form: `ObservationSchedule.assert_analytical_ready` (new `pricing_env` argument, `remaining_records`,
  `remaining_levels`) raises `ValidationError`, as it does for mixed payoffs; a record on the valuation instant still
  decides there. A future knock-out hit pays the hit record's payoff too (the MC's `payoffs[first_idx]`, the PDE):
  the forward closed form and the first-observation hit value use the remaining records' payoff
  (`forward_knock_out_cash`; one fixed payoff is already required, an unset payoff takes `rebate` /
  `knock_out_rebate`); knock-ins and no-touches keep their rebate. The sharkfins' no-rebate knock-out legs no longer
  carry the record payoffs (their cash legs pay them), and a single sharkfin with `knock_out_rebate` 0 whose records
  pay cash no longer skips its touch leg. CONTINUOUS / EXPIRY monitoring unchanged. Moved values (flat 20% vol, r 3%, q 1%, K 100, T 1,
  monthly records; "now" = aged to the 3-month record): up-and-out call `barrier` 120, records 115, spot 110:
  4.14526 -> 3.95226 (up-and-in 16.16050 -> 16.35350); same, now, spot 117: 4.79158 -> 4.88876 (records paying
  4.5: -> 4.39988); one-touch up 120 / records 115, spot 110: 2.65102 -> 3.37817, now at 117: 3.58628 -> 4.88876;
  double knock-out 130/75, records 125/80, spot 100: 4.52666 -> 3.68421, now at 127: 4.07797 -> 2.0; single
  sharkfin 120 / 115, spot 110: 3.52510 -> 2.89580, now at 117: 3.61756 -> 2.93325; double sharkfin 130/75 /
  125/80, spot 100: 5.49710 -> 4.65465, now at 127: 5.05572 -> 2.93325 -- each equal, bitwise, to the contract
  booked at the records' levels; mixed remaining levels value -> refused. Records paying 4.5 (product cash 5, a
  sharkfin's 3), spot 110 / 100: up-and-out call records at 120: 4.14526 -> 3.88016, records at 115: 4.14526 ->
  3.61444; one-touch: 2.65102 -> 2.38591; double knock-out (rebate 2): 4.52666 -> 5.12593; single sharkfin: 3.52510
  -> 4.32040; double sharkfin: 5.49710 -> 5.85667 -- each equal, bitwise, to the contract booked with cash 4.5.
  Single sharkfin `knock_out_rebate` 0, records paying 2.5: 1.93449 -> 3.25999; records at 115, now at 117: 1.46580
  -> 2.44438 (2.5 at the terminal). Plain contracts (records at `product.barrier`, paying the product's cash)
  bit-identical. New `test/test_barrier_schedule_level_analytical_quad.py`.
  The QUAD barrier, one-touch and double-barrier adapters (`quad_adapters.py`) decide at the valuation instant as
  the Black-Scholes barrier MC does (W18b): the immediate hit / touch and the expired branch read the first record at
  or after the valuation instant (`valuation_barrier_terms` / `valuation_double_barrier_terms`), its level(s), and a
  knock-out (one-touch) pays its payoff (the immediate hit / touch reading the NEXT record when none is on the
  valuation instant is superseded by A15 below, which restores the "beyond the next record" cells' pre-T6 values); the forward lattice was already record-aware and prices mixed levels natively
  (no refusal). The knock-out / one-touch rebate lattice is no longer skipped when the product's own cash is 0 but the
  records pay cash. Moved values (QUAD 401 points; up-and-out call `barrier` 120 / records 115, K 100): spot 117 beyond
  the next record 4.50422 -> 4.85223 (5 at the terminal), records paying 4.5 aged onto a record at 117: raised "t_grid
  must be strictly increasing" -> 4.39988, up-and-in at 117: 21.32214 -> 20.97413 (the vanilla), expired at 117: 17.0
  -> 5.0; rebate 0 with records paying 4.5, spot 110: 1.40591 -> 3.81059; one-touch 120 / records 115 at 117: 4.28985
  -> 4.85223, expired at 117: 0.0 -> 5.0, rebate 0 records paying 4.5: 0.0 -> 2.40468; double knock-out 130/75 /
  records 125/80 at 127: 2.48798 -> 1.94089, expired at 127: 27.0 -> 2.0 -- each equal, bitwise, to the contract
  booked at the records' terms and to the BS MC decision. A knock-in whose records pay other than `product.rebate` is
  now refused on the analytical barrier and double-barrier engines and on the QUAD double barrier too (Ruling R19,
  `refuse_knock_in_record_payoff`, the message every route raises), making the W18 Q3 "every route" rule true: e.g.
  analytical up-and-in, records paying 4.5 (rebate 5): 16.16050 -> refused; double knock-in, records 1.5 (rebate 2):
  analytical 6.24155 / QUAD 5.91733 -> refused. QUAD plain schedules and EXPIRY monitoring bit-identical.
  QUAD prices a DISCRETE schedule with a record on the valuation instant that the spot has NOT crossed (it raised
  "t_grid must be strictly increasing" on the barrier, one-touch, double-barrier and double one-touch adapters, plain
  contracts included): the record is decided "not hit" there and leaves the lattice (I5 / A14, as the BS barrier MC)
  -- bitwise the contract aged past it; a sole such record leaves the never-hit value (the remaining-contract
  replication, as once monitoring has ended); two records on that instant stay refused. The double one-touch /
  no-touch QUAD adapter decides at the valuation instant on the deciding record too (`valuation_double_barrier_terms`;
  a double one-touch pays its payoff; its rebate lattice no longer skipped when only the records pay). No analytical
  engine prices `DoubleOneTouchOption` (`analytical/double_no_touch.py` is a probability function on explicit
  levels), so there is nothing to fix there. Moved values (QUAD 401 points, aged onto the 3-month record, spot inside):
  raised -> priced, e.g. up-and-out call 120: 4.37956, up-and-in: 14.74546, one-touch: 2.40644, no-touch: 2.88655,
  double knock-out: 5.57745, double one-touch: 0.77959, double no-touch: 4.10917; sole record: knock-out 14.23864
  (the vanilla), knock-in / no-touch 4.88876 (the rebate at the terminal), touches 0.0, double knock-out 7.56296.
  Double one-touch 130/75, records 125/80: spot 127 (beyond the next record) 4.31809 -> 4.85223 (hit-paid, at 77:
  4.64745 -> 5.0), expired at 127 0.0 -> 5.0, records paying 2.5 aged onto a record: raised -> 2.44438; rebate 0 with
  records paying 4.5: 0.0 -> 1.77979; double no-touch at 127: 0.53413 -> 0.0, expired 5.0 -> 0.0. Plain double
  one-touch / no-touch (28 cells) bit-identical. (The "beyond the next record" cells above are superseded by A15
  below.)
- User decision 2026-10-14 (A15): a DISCRETE barrier is decided only on an observation ("knock out should only
  happen at the observation date, otherwise price it"). A record ON the valuation instant (dated: the valuation
  date; numeric: `is_zero(observation_time)`) with the spot beyond its level knocks out / in at once, at its level
  and payoff (unchanged); with NO record on the valuation instant nothing is decided now, whatever the spot -- the
  route prices the contract forward and observes each future record with its own machinery. This supersedes the
  W18b / T6 pre-check that read the next record between observations. One shared rule,
  `engine/barrier_schedule_rules.valuation_instant_barrier_terms` / `valuation_instant_double_barrier_terms` (None
  when no record is on the instant), now gates the immediate decision of the Black-Scholes barrier MC (`price`), the
  QUAD barrier / one-touch / double-barrier / double one-touch adapters (`early_price`, new `_single_decided_now` /
  `_double_decided_now`) and the single / double sharkfin MC, which read `product.is_barrier_hit(spot)` on every
  monitoring type and now decide a DISCRETE schedule only on the record on the instant, at its level and payoff
  (`_decided_now`). Already forward, verified and unchanged: the analytical barrier / one-touch / double-barrier /
  sharkfin engines (hit now only by a record on the instant, `observed_at_valuation`), the 1D PDE core (BS and
  local-vol barrier, double barrier, one-touch, double one-touch: `valuation_event` reads only records at t = 0;
  `_next_observation` is the far-field boundary row, not a decision), the Heston / SLV 2D barrier PDE (knock-out
  injected at observation steps only) and the local-vol / Heston / SLV barrier MC (no pre-check; paths start beyond
  the level and are observed at the records). CONTINUOUS / EXPIRY monitoring, the exercise-instant value
  (`barrier_expiry_value`, the QUAD expired branch), a record on the valuation instant and a carried knock-in are
  bit-identical. The price is now continuous across the level between observations on QUAD, the PDE and the MC; the
  analytical engines keep a pre-existing step at the level (their BGK-shifted closed form inside vs the
  first-observation integral beyond: up to 0.15 on the test contracts). New
  `test/test_discrete_barrier_between_observations.py`. Moved values (flat 20% vol, r 3%, q 1%, K 100, T 1, monthly
  records, no record on the valuation instant, spot beyond the next record; QUAD 401 points, BS MC 20000 paths):
  up-and-out call 120 / rebate 5 at inception, spot 121: QUAD 4.85223 -> 4.66393, BS MC 4.85223 -> 4.66792 (s.e.
  0.0144; analytical 4.69051 and BS PDE 4.66385 unmoved); aged 0.30, spot 121: QUAD 4.89609 -> 4.86928, BS MC
  4.89609 -> 4.89024; up-and-in 23.20289 -> 23.22971 (BS MC 23.29391 -> 23.06761); down-and-out put 85, spot 84:
  4.89609 -> 4.71718 (BS MC -> 4.73075); down-and-in 15.86746 -> 16.04638 (BS MC 15.81282 -> 16.08246); hit-paid
  up-and-out 5.0 -> 4.94754; one-touch 120: 4.89609 -> 4.30271; no-touch 0.0 -> 0.59338 (85 down at 84: 0.0 ->
  0.53197); double knock-out 130/75, rebate 2, spot 131: 1.95844 -> 3.07690; double knock-in 32.51122 -> 31.39276;
  double one-touch 4.89609 -> 4.29302 (hit-paid 5.0 -> 4.37090); double no-touch 0.0 -> 0.60308; single sharkfin MC
  120 (knock-out 3, no-hit 1), spot 121: 2.93766 -> 3.28873 (analytical 3.27908); double sharkfin MC 130/75, spot
  131: 2.93766 -> 4.04035 (analytical 3.99354). The W18c QUAD "beyond the next record" cells (records 115 / 125-80,
  `product.barrier` 120 / 130-75, spot 117 / 127 / 77) return to their pre-T6 values: up-and-out 4.85223 ->
  4.50422, up-and-in 20.97413 -> 21.32214, one-touch 4.85223 -> 4.28985, double knock-out 1.94089 -> 2.48798,
  double one-touch 4.85223 -> 4.31809 (hit-paid 5.0 -> 4.64745), double no-touch 0.0 -> 0.53413. BS MC, mid-life
  day 92 (no record), spot 117 between the records' 115 and `product.barrier` 120: up-and-out 4.88906 -> 4.57322,
  up-and-in 19.98757 -> 20.02752, down-and-out (87, records 90 / barrier 85) 4.88906 -> 4.56438, down-and-in
  13.57646 -> 14.00622 (BS PDE 4.56015 / 20.22481 / 4.54902 / 13.96480). QUAD (601 points) up-and-out call K 100,
  quarterly records at 110 paying 5, spot 120: hit-paid 5.0 -> 4.60396, terminal-paid 4.85223 -> 4.50945; K 95, records
  at 100 paying 2, spot 100 (on the level): 2.0 -> 1.50672 (BS PDE within 6.4e-5 of each). No modelvalidation anchor
  or replay golden moved.
  The run lifecycle tracker (`lifecycle/barrier.BarrierLifecycleTracker`: the equity backtest and the dynamic
  scenario via `PortfolioLifecycleManager`, and the intraday provisional replay) tested `product.is_barrier_hit` on
  every daily close whatever the monitoring. A DISCRETE product is now decided only on its schedule's records: each
  record once, on its date at that day's close, against the record's own level and paying its payoff (W18c, times the
  contract multiplier where the product's hit cash carries it). A record date the run never visits is decided at the
  previous run instant's spot with `gap=True` on the event (A7'(c), the signed W6 rule, as `lifecycle/dcn.py`); a
  record before the run is history (signed rule 3: no state assumed, not hit), "before the run" being before the FIRST
  RUN INSTANT (ruling R31, T17 CP6; it first refused a record between the start and the first run instant,
  `TODO(barrier-run-start)`, which broke the intraday provisional replay from a checkpoint: that replay restores the
  checkpoint state and observes only the due instants after it, so the records the checkpoint covers are history --
  restored to the pre-T17 outcome on the CP6 probe, KO paying 1.0 on a supplied 111 and alive on the assumed spot); a
  numeric record falls on `start + round(t * 365)` days (the trackers' run clock); a first run instant after the expiry
  retires the contract as matured before the run (CP6 round 2, every monitoring type: the `MaturedBeforeRunWarning`,
  expired, no cash, and `PortfolioLifecycleManager` drops the position; it booked the expiry payoff at that instant's
  spot after expiry). Between records nothing is decided, and at expiry only a record on the expiry
  date hits (no record there: the no-hit payoff). Events carry `metadata` `monitoring="discrete"`, `gap`, `decided_on`,
  `observation_date`. CONTINUOUS and EXPIRY monitoring unchanged. New
  `test/test_barrier_lifecycle_discrete_observation.py`. Moved values (200 daily GBM paths, 20% vol, one year, monthly
  dated records, per-path tracker from 100): up-and-out call 120 / rebate 5: 66 -> 51 knock-outs, total booked cash
  523.684 -> 595.051; down-and-in put 85: 88 -> 74 knock-ins (cash 1498.738 -> 1444.917); single sharkfin 120
  (knock-out 3): 66 -> 51 knock-outs (cash 391.684 -> 493.051); one-touch 120 (5): 66 -> 51 touches (cash 330.0 ->
  255.0). No existing backtest, dynamic-scenario, pnl-explain or intraday test moved (none books a DISCRETE barrier
  product off its record dates). A one-touch / double one-touch touched on a record books the record's payoff (unset:
  the rebate), no contract multiplier, as `get_payoff(touched=True)` scales it (CP6: a product rebate of 0 with records
  paying 4.5 booked 0 -> 4.5). The intraday refusal of a discrete barrier with no record at expiry keeps its
  `CapabilityError`, reworded for A15 (the expiry fixing settles the no-hit payoff, a path not yet validated there).
  `QuadInputAdapter.build_inputs` may return a sequence of legs (annotation and abstract contract, R30).
  The single and double sharkfin MC price a DISCRETE schedule with a record on the valuation instant that the spot has
  NOT crossed (they raised "dt_array must have strictly positive entries"): the record is decided "not hit" there and
  leaves the simulated grid (I5 / A14, as the BS barrier MC), bitwise the contract aged past it; a sole such record
  leaves the never-hit value; two records on that instant are refused. Moved values: raise -> price only (e.g. single
  sharkfin 120, knock-out 0.5, aged onto its 3-month record at spot 119: 1.32888; double 130/75 at 129: 2.56138).
  The double-barrier analytical engine paid a knock-out decided now undiscounted (`rebate x multiplier`, then only the
  delay to the payment date), while its forward formula, QUAD and the PDE's discrete records pay a double knock-out's
  rebate at the terminal payment time (the product has no hit-paid option): it now pays `rebate x DF(T)` there too,
  CONTINUOUS and on a record on the valuation instant (a record's payoff likewise), continuous at the barrier. Moved
  values (130/75, rebate 2, T 0.75, r 3%): spot 131 or 74, continuous or on a record: 2.0 -> 1.95550 (QUAD and PDE
  1.95550); records paying 2.5 on a record now: 2.5 -> 2.44438. Plain inside cells and every other analytical value
  bit-identical; no anchor moved.
- Double knock-out rebate timing (T17 R30, user authority for bug-fix moves): a `DoubleBarrierOption`'s rebate is
  paid only to knocked-out paths, at the terminal payment time (the product has no hit-paid option). Two routes broke
  it. (a) QUAD: `DoubleBarrierQuadInputAdapter._terminal_payoff_coefficients` carves the vanilla by moving the last
  record's lower bound to the strike (a call; a put: the upper), so a rebate on a record at maturity was paid on
  (0, K] (put: [K, inf)) to paths that never knocked out. That record's rebate on the carved side is now its own leg
  on the records' corridor (`build_inputs` returns two `QuadCoreInputs`; `DiscreteQuadEngine` sums the core over an
  adapter's legs, one leg unchanged); the knock-in parity `vanilla + R DF - core` inherits the fix. Oracle: the rebate
  leg `V(R) - V(0)` equals `R DF(T) P(knock-out)` (knock-in: `P(survive)`, from QUAD's double no-touch) to 1.6e-15,
  and QUAD equals the Black-Scholes PDE within 6.2e-4 (0.75 before). (b) PDE: `DoubleBarrierAdapter._hit_price`
  paid a continuous hit-now at the hit instant (undiscounted) while its forward leg and discrete records pay at the
  terminal payment time; it now reads the terminal payment time (a product with `pay_at_hit` would keep the hit;
  the single-barrier and one-touch adapters already follow `pay_at_hit` / `payment_at_hit` and are unchanged). New
  `test/test_double_knock_out_rebate_timing.py`. Moved values (130/75, rebate 2, K 100, T 1, monthly records, flat 20%
  vol, r 3%, q 1%; QUAD 401 points): knock-out call at inception, spot 100: 5.16249 -> 4.41101 (PDE 4.41066);
  aged 0.30 spot 100: 5.64657 -> 4.78246; EXPIRY monitoring 6.00231 -> 5.17774 (analytical 5.17774); knock-in call
  5.60572 -> 6.35720 (PDE 6.35782), aged 4.46446 from 3.60035, EXPIRY 4.76590 -> 5.59047; knock-out put at inception
  5.11362 -> 4.41915 (PDE 4.41904); aged 0.30 spot 129: call 3.62377 -> 3.54538, put 2.15674 -> 1.87254; spot 80: call
  2.43157 -> 1.74220, put 5.67308 -> 5.51107; the A15 "beyond" cells (aged 0.30, spot 131): knock-out 3.07690 ->
  3.02580, knock-in 31.39276 -> 31.44386. Unmoved: record-day and expired cells (decided before the lattice), every
  contract whose last record is before maturity or pays no rebate, and the strike outside the corridor. PDE
  continuous knock-out hit now (aged 0.30, spot 131 or 74): 2.0 -> 1.95844 (= 2 DF(0.70); just inside, 1.95867);
  rebate 10, T 0.5, r 5%, spot 130 outside 80/120: 10.0 -> 9.75310.
  No modelvalidation anchor or replay golden moved.
- A knock-in carried as triggered decides the exercise date (user ruling 2026-10-10, "check knock in state first").
  At T = 0 (a dated contract on its exercise instant, or the numeric twin at `maturity == 0.0`) the barrier engines
  decided a knock-in from today's spot / records only and paid the rebate to a contract carried as knocked in
  (`lifecycle_state.knocked_in` or the product's `_otc_lifecycle_knocked_in`), while the day before the same state
  priced the knocked-in vanilla. The knock-in decision at T = 0 is now the carried state first
  (`monitoring_ended.knock_in_carried`, the helper the T > 0 path reads), then today's observation: knocked in pays
  the payoff, only a contract never knocked in the rebate. Fixed in the shared expiry code, so every route gets it:
  `BarrierAnalyticalEngine._price_at_maturity`; `barrier_expiry_value` (BS barrier MC, and the local-vol / Heston /
  SLV barrier MC routes); the PDE knock-in adapters' `maturity_value` (BS and local-vol barrier PDE, the Heston / SLV
  barrier PDE routes through `barrier_expiry_outcome`, double barrier PDE; greeks: delta the payoff slope); the QUAD
  barrier and double barrier expired price; `DoubleBarrierOptionAnalyticalEngine._price_zero_maturity`. Up-and-in
  call K100 B130 rebate 5 carried as knocked in: 5.0 -> 10.0 at spot 110, 5.0 -> 0.0 at 90 (both forms, every
  route); down-and-in put B80: 5.0 -> 10.0 at 90, 5.0 -> 0.0 at 110; double knock-in 80/130: 5.0 -> 10.0 at 110.
  The numeric twins moved from the same wrong values on main. A knock-in not carried, a carried knock-out / touch
  (terminal lifecycle state, read before maturity on both sides of T = 0) and every autocallable with a knock-in
  term (snowball, Phoenix, KO-reset, on MC / PDE / QUAD V1 / V2 / local vol / Heston / SLV, which already read the
  carried state at T = 0) are unchanged. New `test/test_expiry_carried_knock_in.py` (hand settled values, both
  forms, both carriers, day-before continuity); W9's `up_in_call_knocked_in` cells are now also pinned to the hand
  value. Evidence: `docs/superpowers/specs/2026-10-10-carried-knock-in-at-expiry-evidence/`.
- Snowball / Phoenix / KO-reset PDE (1D BSM and local vol, 2D Heston and SLV): the not-knocked-in (V0) Dirichlet
  rows are built by edge. The knock-out edge is the upper row on a standard product and the LOWER row on a reverse
  one; it carries the knock-out payoff once it lies beyond every knock-out level. A reverse product used to price
  its upper (knock-in side) row as the knock-out payoff and its lower (knock-out side) row as principal + rebate.
  A call-rebate contract's no-event rows pay principal + the participation's forward intrinsic value (a put on a
  reverse Phoenix) and no fixed rebate; the lower row used to add `rebate_rate x S0` (the factory sets it to the KO
  rate). Under continuous knock-in the knock-in edge copies the knocked-in row only when it lies at or beyond the
  knock-in level; it used to copy it unconditionally, so a knock-in barrier the grid never reaches still knocked in
  the edge. Certain-V0 controls (`docs/superpowers/specs/2026-10-08-accrual-scope-followups-evidence/
  item7_v0_rows_by_edge.py`, PDE - closed form, flat / annualized): reverse fixed rebate +1.1e-3 / +5.0e-4 ->
  6.7e-7 / 6.6e-7; standard call rebate with `rebate_rate` 0.10 +3.4e-4 / +3.5e-4 -> 4.0e-6 / 7.2e-6; reverse call
  rebate -9.2e-4 / -3.6e-3 -> 6.7e-6 / 1.3e-5; continuous KI off the grid -2.7e-3 / -3.0e-3 -> 2.4e-6 / 2.5e-6.
  The earlier "cancellation" on the call-rebate rows was QUAD's: on its default 1001-point grid QUAD overprices the
  call leg by 4.4e-4 / 8.7e-4 against the closed form (2001 points 1.4e-4, 4001 1.4e-5, 8001 7.2e-6), so PDE - QUAD
  read better with the phantom rebate. Every banked certificate anchor, `pde_statistics` reference and replay golden
  reproduces bit for bit (the cert reverse and call-rebate cells' edges lie too far from the spot to reach the price),
  and so does the PDE byte book (`test/pde_bitwise/capture_book.py`, 713 rows, 33201993 vs this branch, same machine:
  0 diffs; the same 10 rows raise on both sides).
- Snowball / Phoenix PDE, 2D Heston and SLV: under continuous knock-in, the not-knocked-in edge that lies beyond the
  knock-in level copies the knocked-in surface's row of the same step. The 2D hooks solve V1 to completion before V0
  and kept no `_grid_v1`, so the edge row was 0.0 (standard lower edge, KO 103 / KI 75: 0.0 -> 10.2004; reverse
  upper edge, KO 97 / KI 125: 0.0 -> -791.999). They now compute V1's rows first, as the 1D adapter does
  (`SnowballPDESolver._set_boundary_conditions_v0_beside_v1_rows`). Prices move at round-off (|move| <= 4e-13) on
  the contracts measured, and by up to 8.1e-5 only when the knock-in level sits just inside the grid's lower edge.
  Discrete knock-in is untouched (`docs/superpowers/specs/2026-10-09-heston-2d-v0-evidence/`).
- 2D Heston / SLV ADI core (`HestonSLVADICore._A1`, `v0_boundary="degenerate_pde"`, the 2D snowball / Phoenix / DCN
  default): the explicit S-operator now carries the degenerate v = 0 column, `(r - q) U_x - r U`. The implicit
  S-solve already carried the S-drift and the -rU reaction on every variance column, so the v = 0 row evolved under
  theta x A1, consistent only on a fully implicit step (W7 ruling Q1, 2026-10-10). The standard certain-V0 control
  under Heston at (96, 24, 48), every damping step off, goes from +2.478e-3 to -9.31e-7 against its closed form; with
  the default damping +3.827e-3 -> +3.761e-3 (the rest is the damped steps' dt^2 term, by design: ruling Q4).
  Every 2D Heston / SLV price on that boundary moves: `adi2d-snowball-greeks` re-certified by amendment
  (2026-10-10; 28 of 28 anchors moved, low_feller delta +2.66e-4 / +2.57e-4 the largest; both candidates ADMITTED,
  the low_feller errors against the banked reference shrink); the reverse continuous-KI Heston Phoenix (KO 97 / KI
  125) at (192, 48, 192) 95.7019 -> 95.5960, now monotone in refinement; the Feller-violating Heston DCN at (151, 41)
  -89176.7 -> -90111.4 per 1,000,000 notional. Values: `docs/superpowers/specs/2026-10-09-heston-2d-v0-evidence/moved_values.txt`.
- Phoenix PDE (1D, and the 1D statistics march of the 2D Heston / SLV Phoenix): under continuous knock-in the
  not-knocked-in edge beyond the knock-in level copies the knocked-in row of the step; `PhoenixAdapter` marched
  every V0 memory state before the knocked-in surfaces and never exposed them, so the row was 0.0 (W7 ruling Q2). A
  V0 group now sets V1's rows of the step first and exposes them for that row, as the snowball adapter does, and the
  Phoenix statistics copy the knocked-in row too (`PhoenixStatistics._alive_ki_edge_copies_ki = True`). Moves short
  continuous-KI Phoenixes whose edge sits near the knock-in level (three months, reverse KI 150: +1.737e-5);
  `phoenix-flat-bsm` re-certified by amendment (2026-10-10; near_expiry pv -2.8e-16, delta +6.9e-17, the other 124
  anchors bit for bit).
- Snowball / Phoenix PDE continuous knock-in on an up (reverse) barrier, FIRST_PASSAGE crossing correction
  (`_ContinuousKIFirstPassage._geometry`, W7b, ruling 2026-10-10). The half-cell floor on the slope sample measured
  the cell at the grid's far edge on an up barrier (a negative "cell": the floor never acted), so a node a few
  thousandths of a cell below the barrier sent the 2D Heston / SLV price off by ~x8 per step; and the reflection
  factor overflowed to inf x 0 = NaN on the 2D v = 0 column. The floor now measures the cell at the barrier, and the
  factor is evaluated in one exponent where it would overflow (`Tolerance.EXP_MAX`). Fixed: reverse KO 97 / KI 125 at
  n_x 312 / 317 / 384 (diverged to -1e30 ... -1e173), the production Greek grid at KI 139.80 (raised `PricingError`),
  the production PV grid at KI 114.45 (NaN); other reverse 2D continuous-KI grids whose nearest node lies within half
  a cell of the barrier move by <= 6.6e-4. Down barriers, discrete knock-in and every 1D layout are bit for bit, and
  so is every certificate anchor and golden (`docs/superpowers/specs/2026-10-09-heston-2d-v0-evidence/moved_values.txt`).
- Dated records without `initial_date` no longer accrue from the valuation date (coupons shrank as the contract
  aged).
- Phoenix `get_ko_payoff` keeps `accrued_offset`.
- The product, QUAD V2 and the intraday path agree on every Phoenix coupon period, the first one included.
- An aged MC REBASED knock-out accrues from the contract's origin (it lost the elapsed part).
- Ageing a KO-reset contract (`time_shift`, a theta bump) now moves an ABSOLUTE schedule 2 a day closer as
  well, and no longer grows its knock-out cash or knocked-in tenor. Schedule 2 used to stay where it was while
  the elapsed time was banked into `accrued_offset` (one day added 1/365 to every post-KI accrual, 0.75 ->
  0.75274), so KO-reset theta was wrong: about 10% low on a live contract, of the wrong sign on a knocked-in one
  (book contract: -0.0181 -> +0.0489). The one-day theta now equals the calculator's theta on the contract aged by
  hand (`test/accrual_time/test_ko_reset_theta.py`). A REBASED schedule 2 holds offsets and stays.

Measured on the PDE byte book (`test/pde_bitwise/capture_book.py`, 721 rows, main e3021668 vs this branch, same
machine): 678 rows bit-identical; in the 43 that moved only `bump_theta` moved, every price and every other Greek
is bit-identical. 16 Phoenix rows move at ulp level (relative 2e-14 to 2.5e-12: an aged numeric accrual now reads
the original quoted end instead of `(offset + bump) + (t - bump)`). 27 KO-reset rows move by the schedule-2 fix
above: relative 2e-2 to 1.6e-1 on the standard alive rows (e.g. `ko_reset|flat|standard` +0.02480 -> +0.02707),
up to a sign change on knocked-in and near-barrier rows (`3a|class|ko_reset|cont|knocked_in` -0.01807 ->
+0.04887, `3a|bumpcross|ko_reset|cont|85.5` -0.14185 -> +0.00369).

- Range accrual (equity `RangeAccrualOption`, FX `FxRangeAccrualOption` and its foreign and quanto variants): an
  annualized coupon accrues over the WHOLE contract -- `tenor`, else `initial_date` -> expiry in
  `range_config.day_count_convention` (which was never read), else a numeric contract's `maturity` plus the time
  `time_shift` has aged it by. It used to accrue over the REMAINING maturity while the in-range ratio counted every
  fixing, past and future, so the coupon shrank as the contract aged: a one-year 10% contract certain to pay 10.0
  priced at 7.34 after 91 days and 0.027 one day before expiry; it now prices at 10 x DF on every date. Inception
  prices are unchanged when the config's day count is the one the environment used.
- Range accrual theta: `time_shift` moves a numeric contract's fixing times with its maturity (they stayed put, so
  past fixings looked future), and a fixing the theta bump steps over is fixed at the held spot (a dated FX one used
  to be refused as an unrecorded past fixing). Theta on a certain coupon is now the discount roll (+8e-4 per day on
  the control) instead of a day's accrual lost (-0.026).
- cashleg `AccrualLeg`: `day_count` is now read. When the `LegSchedule` carries `period_start_dates` /
  `period_end_dates`, each period accrues `calculate_year_fraction(start, end, day_count)`; with times only it keeps
  `period_end - period_start` (bit-identical). The field used to be ignored and the dates never read, so ACT/365,
  ACT/360 and 30/360 US gave the same PV. On a dated quarterly 2026 leg (4% on 1,000,000, r = 5%, no KO): ACT/360
  38773.115547 -> 39306.910836 (+1.38%, = 365/360 x the dated ACT/365 value), ACT/365 -> 38768.460003 (90/91/92/92
  days instead of four quarter-years), 30/360 US unchanged. No library, test or example caller builds a dated
  schedule: every existing leg PV is bit-identical (14 caller configurations measured against dcea77b6). New
  refusals: period dates given for only one side or only some periods, non-`datetime` dates, an end not after its
  start; a dated `BUSINESS_DAYS` leg without the new `calendar` and `bus_days_in_year` fields (and those fields on
  any leg that would not read them); ACT/ACT ISMA/BOND/CHINABOND on a dated leg (no coupon frequency).

### Removed
- `AccrualConfig.accrual_dates` (it had no reader).

The analytic barrier engines' observation rules (the 3C handoff, spec
`docs/superpowers/specs/2026-10-04-pde-1d-remaining-routes-design.md` §12). Prices inside the
barriers away from maturity are bit-identical.

### Fixed
- A spot beyond a discretely monitored barrier with no observation due is no longer a hit in
  `BarrierAnalyticalEngine`, `OneTouchAnalyticalEngine`, `DoubleBarrierOptionAnalyticalEngine` and
  `DoubleSharkfinOptionAnalyticalEngine` (and `SingleSharkfinOptionAnalyticalEngine`, which prices
  through the first two). The value is taken through the next observation
  (`engine/analytical/first_observation.py`): the Gaussian step to it and its barrier test are
  exact, the engine's own barrier-shifted formula applies from it on, and the expectation is a
  two-order Gauss-Legendre rule that raises `NumericalError` when the orders disagree beyond
  `Tolerance.RELATIVE`. Against the exact discrete recursion (`test/pde_routes/barrier_reference`),
  monthly and aged monthly schedules: single barrier 1.41 → 4.4e-2 (74 cells), one-touch /
  no-touch 0.74 → 2.2e-2 (18), double barrier 0.44 → 2.0e-2 (32), double sharkfin 0.58 → 1.3e-2
  (32), each below the shift's own error just inside the barrier (0.24, 0.096). An observation at
  the valuation instant still decides on the spot, at its own level. A schedule other than
  `STOP_FIRST_HIT` beyond its barrier now raises `PricingError` (it was priced as a hit). A price
  beyond the barrier costs 25-30 ms on the single barrier and up to 0.5-0.9 s on a daily-monitored
  double sharkfin (one Ikeda-Kunitomo series per quadrature node); inside, unchanged.
- `DoubleSharkfinOptionAnalyticalEngine`'s continuous paid-at-hit leg integrated the survival curve
  with 48 Gauss-Legendre nodes over (0, T), which miss its fall next to a barrier: off by up to
  2.4e-5 on a rebate of 5. The integral now runs on panels doubling from the time at which a
  barrier comes within nine deviations (exact below it), at two orders, and raises
  `NumericalError` when they disagree; 4e-15 against an adaptive reference on 60 cells.

### Changed
- A no-touch (`OneTouchOption` with `NO_TOUCH`, `DoubleOneTouchOption` with `DOUBLE_NO_TOUCH`) and
  both sharkfins refuse an `ACCUMULATE` observation schedule at construction (`ValidationError`):
  they pay only if the barrier is never hit up to and including expiry, so nothing accumulates
  (user ruling 2026-10-07). The PDE priced an accumulating no-touch as if only its maturity
  observation existed (4.304787 against 4.227537 under first hit, a year out, records at 0.5 and 1);
  the double-sharkfin MC summed every hit's cash. One-touches and barrier knock-outs keep the mode.
  The eight `3c|mat|*no_touch|acc|*` byte-book rows are retired.

### Added
- KO-reset snowball lifecycle tracking (spec `docs/superpowers/specs/2026-10-08-ko-reset-lifecycle-design.md`):
  `KOResetLifecycleTracker` books the schedule-1 knock-out, the knock-in that switches the contract to schedule 2
  for good (tested the same close, inclusive; schedule-2 records due on an earlier close are consumed), V0 at
  schedule 1's end on its tenor (no schedule-2 extension), and V1 at the knocked-in contract's end; a knock-out
  wins a tie. `PortfolioLifecycleManager` (equity backtest, dynamic scenarios), `ProductReplay` and the
  simulated-path backtest track KO-reset positions (the manager used to warn and reprice them as of every day,
  and every engine raised the day after schedule 1 ended un-knocked). Both schedule-2 modes: a knocked-in
  `REBASED` contract is priced as the `ABSOLUTE` contract it has become
  (`KnockOutResetSnowballOption.with_absolute_schedule_2`); an un-knocked `REBASED` contract needs a Monte
  Carlo engine (refused at configuration elsewhere, and in simulation surface mode). The knock-in close
  (`ki_date`) is the reset's timing; its accrual hangs from the persisted `KnockInAnchor` (the knock-in record that
  fired, at its contractual fixing; accrual time spec §5.1), stored in the state beside `ki_date`, carried by every
  pricing copy (`_otc_lifecycle_knock_in_anchor`) and embedded by
  `with_absolute_schedule_2(knock_in_time, *, anchor)` in each converted record's relative period and in the `post`
  terminal (§5.2): knock-in records due on days 91 and 93 observed on day 95 accrue from day 91. A carried `REBASED`
  knock-in needs `_otc_lifecycle_knock_in_date` and, when its knock-in records accrue, its anchor
  (`_otc_lifecycle_knock_in_anchor`); it is never looked up by date. Simulated paths take the lowest-index knock-in
  record due on a path's knock-in close and refuse records of different levels due on one close.

### Changed
- Every autocallable tracker holds its contract with numeric observation times snapped to the day it observes
  them (`round(time * 365) / 365`; whole-day times keep their exact float, dated records are untouched), and
  every consumer (pricing copy, replay, simulation surfaces) builds from it. An off-day numeric record used to
  be observed on its rounded day yet priced as still ahead (a day-1.4 knock-out: 4.9183 on the life surface
  against 0.4381). Snapping moves observation timing only: each record keeps its accrual period, so a `k / 12`
  Phoenix coupon at 12% still pays its quoted 1/12 (1.0) when observed on day 30 (the "cash follows the snapped
  time" amendment is withdrawn, accrual time spec §7). Whole-day and dated products are bit-identical; replay
  goldens unchanged.
- The replay and the simulation price the initial book before the first close (`product_for_pricing(...,
  initial=True)` keeps a record landing that day); the simulation keys such a state apart only where a record
  lands.
- Replay QUAD prices a KO-reset on `KOResetSnowballQuadEngine` (`SnowballQuadEngine` ignored schedule 2: 4.195
  for -1.744). A KO-reset under `vol_model != "bsm"` needs `vol_model_solver="mc"`: the vol-model PDE solvers
  price it on schedule 1 (21-26% off).

### Fixed
- KO-reset lifecycle final review: a `REBASED` schedule 2 hangs from the knock-in close, timed and dated from the
  run start, the clock pricing copies age on (a contract issued before the run placed it early, or its pricing
  copy held it the gap too far ahead) and sits exactly on the day grid (a record ~5.55e-17 years ahead on its own close made QUAD raise); a dated `REBASED` contract's
  knocked-in end is dated (it raised `TypeError` on the knock-in close); `observed_post_ko_indices` is PnL-explain
  bookkeeping; `PortfolioLifecycleManager` values each tracked autocallable from its tracker's initial copy and the
  dynamic scenario and equity backtest price their baseline after registering trackers (a carried knocked-in
  `REBASED` position was priced as booked); `EnsembleConfig` refuses a carried knock-in (it was dropped).
- `SnowballMCEngine`, `REBASED` KO-reset: a post-knock-in knock-out keeps the accrued offset of an aged contract
  without `initial_date` (an annualized control paid 6.5 instead of 7.5); V1 is read at each path's own maturity
  (`ki_time + max(offset)`, not the grid's last node) on that path's tenor (`get_maturity_payoff_v1` takes
  `contract_tenor`; an annualized `REBASED` V1 raised "ki_time is required").

### Removed
- `DoubleSharkfinOptionAnalyticalEngine(quad_points=...)`: the paid-at-hit integral controls its
  own error. Passing it raises `TypeError`.
- The double-sharkfin MC engine's `ACCUMULATE` aggregation branch (unreachable: the product refuses
  the mode).
- At maturity those engines observe only what is observed now: a discrete schedule's records at
  the valuation instant, each at its own level and paying its own cash (none: no hit; it tested
  the spot against the product's barrier, 5 for 35 and 25 for 7), and an untouched single-barrier
  knock-in pays its rebate (it returned 0).



Theta at an observation boundary (spec
`docs/superpowers/specs/2026-10-07-theta-observation-boundary-design.md`). Theta moves the contract
from the as-of date `t0` to the bumped date `t1` in one step shared by every caller
(`riskmeasures/greeks/theta_shift.py`): `GreeksCalculator` and the clock family (charm, color,
vega decay), the execution bump cells, `EquityPosition.get_trade_risk` with cash legs, and the
autocallable risk report with its surfaces. Lifecycle aging keeps its rules (`keep_landing=False`
runs the old comparisons) apart from the KO-reset fix below; the replay goldens are byte-identical.

### Changed
- An observation that lands on `t1` is kept at time 0 and observed at the new as-of instant; it
  was dropped. A daily accumulator one day before a fixing (spot 100, strike 95, 10,000 shares a
  day): theta +19,397 → +69,397, the difference being exactly that fixing's settlement.
- The observation at `t0` is decided on the spot, as the engines decide it at `t0`, and only its
  outcome is carried: a knock-in (on the product, as the autocallable engines read it, and now
  for the barrier family too), an accumulator fixing, a Phoenix's memory arrears, cash paid after
  `t1` as a receivable. A contract `t0` ends is worth its engine's own `t0` value re-discounted to
  `t1`, or 0 once paid within the bump: on an event day theta is about minus the payment; cash
  paid on `t1` itself stays. Theta excludes only cash flows already booked (user ruling 2026-10-07).
  An accumulator's `t0` fixing is that cash (it is no longer added to `past_observations`); a
  TERMINATION breach on it leaves the booked accrual and the rebate receivable; a KO-reset whose
  schedule 1 ends un-knocked matured then. A Phoenix `EXPIRY`-pay coupon decided at `t0` raises
  `CapabilityError`: it rolls up to the note's end, which no state carries yet.
- A carried knock-in is priced through each engine's own triggered knock-in, the path a knock-in it
  decides at the valuation instant takes, so theta differences one method: the PDE composition, the
  MC vanilla, the local-vol European kernel, QUAD's vanilla; the analytic engines' is the
  Black-Scholes vanilla; the Heston and SLV barrier engines (MC and 2D PDE) take it through a
  `knocked_in` flag on their kernels (every path knocked in; the PDE's own vanilla leg), equal to
  their own breached knock-in to 1e-12. A carried knock-in is an event that happened (user
  ruling 2026-10-08); they refused it.
- A schedule the bump empties is `ObservationSchedule.monitoring_ended` and priced as what
  remains (barrier family: vanillas and cash digitals, `engine/monitoring_ended.py`; a triggered
  knock-in the same way); theta there is carry and discounting, not 0. The Heston and SLV barrier
  engines raise `CapabilityError` on it (no remaining-European path under their own model).
- Charm, color and vega decay treat a contract `t0` ended as a fixed receivable (no spot or vol
  greeks at `t1`) and never difference a carried receivable.
- `time_shift(..., keep_landing=False)` on schedules, `BarrierConfig`, products and
  `AutocallableCashLeg`; product `time_shift` always ages and returns `None` (a KO-reset's returns whether
  every observation, schedule 2 included, has passed).
  `RangeAccrualOption.time_shift` now shifts numeric records and decides past fixings on the spot.

### Fixed
- `SnowballQuadEngine`, `PhoenixQuadEngine` and `KOResetSnowballQuadEngine` price an empty
  knock-in or knock-out side (the contract aged past its last date) as QUAD V2 does; they raised
  `PricingError`. A KO-reset whose schedule 1 ended un-knocked still raises: it has matured.
- `KnockOutResetSnowballOption.time_shift` ages an `ABSOLUTE` schedule 2 (the post-knock-in
  knock-out schedule); numeric post dates stayed put. A knocked-in KO-reset's theta now equals the
  knocked-in snowball's on schedule 2 (+0.0166 in the byte book's case, where it was -0.1406); 24
  KO-reset `bump_theta` byte-book rows move, no other output of the 713 rows. (The same fix landed
  independently on the accrual-time branch, entry above; this branch keeps both its `keep_landing`
  and accrual time's schedule-2 factor slicing. Both byte-book counts were measured against main on
  their own branch; the merged tree has not been re-measured.)
- `DiscreteQuadEngine` ignored a knock-in carried on the product (`_otc_lifecycle_knocked_in`) and
  priced a knocked-in contract as alive.
- The accumulator analytic and MC engines settle an observation at the valuation instant on the
  spot; the analytic engine raised on its zero-maturity leg.

## [0.10.0] - 2026-10-06

PDE 1D remaining routes (spec: `docs/superpowers/specs/2026-10-04-pde-1d-remaining-routes-design.md`,
sub-project 3C). The four remaining 1D routes, the local-vol vanilla (`LocalVolPDESolver`), the
local-vol barrier (`LocalVolBarrierPDESolver`), the DCN (`DCNPDEEngine`) and the local-vol DCN
(`LocalVolDCNPDEEngine`), march on the shared PDE core: the shared presets, the cell-average
projection, grid Greeks, bump contexts, `explain` and grid-capable prepared sessions. Their
prices move toward independent references on every measured family. Before they moved, the
barrier family was repaired: the contract multiplier, the decision at the valuation instant and
at maturity, EXPIRY monitoring and the far-field row beyond a discrete barrier, a knock-in's
rebate and session grid, one bump layout per composite part; the analytic double-barrier and
double-sharkfin engines now read one double no-touch probability with error control. The Heston
and SLV barrier solvers' vol-of-vol → 0 limit moved onto the core as well. Last, under
discretely observed knock-in the not-knocked-in edge on the knock-in side takes the knocked-in
row, which moves 31 certified values in 10 cells, each as pre-measured. Every stage was checked
against the PDE byte book (510 rows at the base, 721 at the end) with its declared movers.

### Added
- `BarrierPDESolver` and `LocalVolBarrierPDESolver` accept `ObservationType.EXPIRY`: the product
  is a discretely monitored barrier whose only observation is maturity, its rebate paid at the
  terminal payment time (1.0e-5 from the closed form on the restated 0.7.0 refusal test).
- A knock-in's rebate is priced by `BarrierPDESolver`, `DoubleBarrierPDESolver` and
  `LocalVolBarrierPDESolver` (Fixed, below).
- Bump Greeks of a continuous knock-in through every `create_bump_context` consumer
  (`GreeksCalculator`, the bucketed Greeks, `execution/greeks.py`, portfolio, PnL explain); they
  raised `PricingError`. The bump context freezes one layout per composite part. On 12 cells
  against closed-form Greeks through the same bumps the largest errors go theta 4.6e-5 → 1.0e-6,
  rho 4.6e-4 → 1.8e-5, vega 9.5e-5 → 3.7e-5, delta 1.7e-4 → 6.9e-5.
- `LocalVolPDESolver` and `LocalVolBarrierPDESolver`: `explain`, `create_bump_context`, every
  `PDEParams` field honoured (the kernel's refusals of domain, event, damping and stepper fields
  and of a custom operator are gone), grid artefacts in prepared sessions;
  `LocalVolBarrierPDESolver.calculate_greeks` (`price, delta, gamma` from the grid, composed for
  a knock-in). The barrier route also gains per-observation barrier schedules and a lagged
  pay-at-hit rebate (the kernel raised `CapabilityError`), and both a value at maturity (the
  kernel raised `ValidationError` at `tau = 0`).
- `DCNPDEEngine` and `LocalVolDCNPDEEngine`: `calculate_greeks` (`price, delta, gamma` from the
  grid), `explain`, `create_bump_context`, every `PDEParams` field honoured, prepared sessions
  on the core adapters. Grid delta is within 0.040 % at six spots where the bump delta was up to
  0.85 % off; gamma one point above the knock-in barrier (0.113) is off by −0.0036 where it was
  off by −0.057.
- The DCN's session grid and spot curve, per level (the user's ruling on the plan's P8, in place
  of a fail-closed grid). `BranchSolution`, beside `CompositeSolution`, carries the alive and
  knocked-in surfaces. Each level is read from the surface the base-spot readout would take at
  that level, the knocked-in one at or below the knock-in barrier or when the contract is
  flagged knocked in at valuation, else the alive one, and its delta and gamma are
  differentiated within that surface. A session's grid rows equal `calculate_spot_greeks_curve`
  at the same levels bit for bit; a DCN flagged knocked in reads every row from the knocked-in
  surface. Nothing is refused.
- `localvol_greeks.local_vol_rho_theta`, the rho / theta half of `local_vol_model_greeks`, which
  the local-vol vanilla calls. The helper calls it too, so the Heston analytical, Heston MC,
  Heston-SLV MC and local-vol MC engines are bit-identical.
- A knock-in's session grid and one-solve spot curve, composed from its parts, on
  `BarrierPDESolver`, `DoubleBarrierPDESolver` and `LocalVolBarrierPDESolver`; `CompositeSolution`.
- `analytical/double_no_touch.py`: `double_no_touch`, the double no-touch probability by the
  method of images, written so that two large numbers are never added. It returns a value whose
  error bound is at most `Tolerance.ZERO` (1e-10) or raises `NumericalError` with the bound;
  `double_no_touch_with_bound` returns `(value, bound)` and refuses nothing. Over 10,543 cells no
  error is outside its bound and the largest is 0.46 of it. 1 to 13 image pairs, 4.5 to 35 µs a
  call against 45 to 49 µs for the 80-term series it replaces. The double-barrier and
  double-sharkfin engines read it.
- Core adapters `LocalVolEuropeanAdapter`, `LocalVolBarrierAdapter`, `DCNAdapter`,
  `LocalVolDCNAdapter`; solver mixin `LocalVolSurfaceMixin`, one copy of the surface members of
  the five local-vol solvers (bitwise for the local-vol Snowball and Phoenix).
- `BaseEngine.static_replication(product, env)`: the legs an engine prices a product with when
  it is a static combination of other products (`None` otherwise); `GreeksCalculator`
  computes such a product's greeks as the weighted sum of its own greeks of the legs.
- `quantark.asset.equity.engine.pde.core.adapters.expiry_replication`: EXPIRY monitoring on
  the double-barrier and touch PDE solvers as European vanillas and cash digitals on the
  analytic engines (`FamilyAdapter.replication`; no grid).
- `quantark.volmodels.heston.deterministic_limit.DeterministicVarianceLocalVol`: the
  vol-of-vol → 0 limit of Heston and Heston-SLV as a local-vol surface, `L(S, t) sqrt(v(t))` with
  `v(t) = theta + (v0 − theta) e^(−kappa t)` (`L = 1` under Heston).
- `quantark.modelvalidation amend` keeps an imported certificate's EXTERNAL benchmark: it never
  runs it, re-runs every candidate, carries a bit-identical cell verbatim, translates a moved one
  by the harness's own interval arithmetic, and refuses any translation that would change a
  verdict (`RELEASE_PROCEDURE` section 10). It refused the whole study before.
- `quantark.modelvalidation amend --rerun-candidate NAME` (repeatable) re-runs a candidate whose
  implementation changed although its schema-1 identity (its configuration) did not; the
  amendment records `rerun_candidates` and an unknown name is refused (`RELEASE_PROCEDURE`
  section 6).
- `test/pde_routes/`: the exact DCN and discrete-barrier recursions with their term-curve forms,
  the no-touch series, the local-vol barrier kernel as the test oracle (`lv_barrier_kernel.py`),
  and their guards.

### Changed
- **The four migrated routes' prices and Greeks**, toward the references on every measured
  family. Local-vol vanilla and barrier, maximum |error| at `standard` in price units at spot
  100, kernel → core, on a constant / the byte book's / a steep surface: vanilla 4.6e-3 / 4.8e-3
  / 4.8e-3 → 2.5e-4 / 2.6e-4 / 2.9e-4; continuous barrier 3.5e-3 / 3.8e-3 / 3.2e-3 → 4.1e-4 /
  4.0e-4 / 7.1e-4; monthly 5.9e-3 / 4.7e-3 / 2.2e-2 → 2.5e-4 / 2.5e-4 / 6.2e-4; weekly 1.2e-2 →
  1.6e-4 (constant); daily 1.1e-2 / 8.3e-3 / 9.4e-2 → 1.2e-3 / 1.1e-3 / 2.1e-3. Greeks on the
  constant surface, delta and gamma: vanilla 5.7e-4, 1.3e-3 → 4.7e-6, 1.8e-6; continuous barrier
  7.8e-4, 1.3e-3 → 1.1e-5, 1.3e-6; discrete barrier 1.7e-3, 9.1e-4 → 8.9e-5, 4.4e-5; vanilla rho
  and theta 1.5e-4, 2.6e-5 → 1.1e-5, 1.2e-6. DCN, in bp of notional over 24 cells (four
  contracts, six spots), maximum at fast / standard / high 5.2 / 7.2 / 7.7 → 0.38 / 0.35 / 0.10
  (median 2.1 / 1.4 / 1.8 → 0.12 / 0.11 / 0.03); the points ladder at the money no longer
  wanders (−10.5 to +7.7 bp over 201 to 2,401 points → 0.10 to 0.46 bp); the local-vol DCN on a
  skewed surface −0.9 to −9.8 bp → within 0.18 bp of its finest rung. On kinked rate and dividend
  curves: vanilla 4.5e-3 → 2.6e-4, monthly barriers 5.1e-3 → 4.6e-4, DCN 4.2 to 6.0 bp → 0.22 bp.
- **Their defaults are the shared presets**: 200 / 400 / 800 concentrated points, 4 / 4 / 8 steps
  per day, half-step damping (1, 1), cell-average projection. `GridConfig(time_steps=200)`
  reproduces the old local-vol time budget at a cost of at most 1.4e-5 on a vanilla and 3.4e-5 on
  a continuous barrier, but up to 5.4e-4 monthly, 4.3e-3 weekly and 2.6e-2 daily. Daily
  monitoring at `standard` carries about 1e-3 of time error and wants 8 steps per day (+9.3e-4 /
  +2.1e-4 / +1.1e-5 at 4 / 8 / 16). The DCN's 201-point floor is gone (`fast` is 200 points), and
  a DCN grid of 2,401 points needs `GridConfig(max_points=2401)`. Their wall time: Performance,
  below.
- `LocalVolPDESolver.calculate_greeks`: `delta` and `gamma` are pointwise grid Greeks, not 1 %
  bumps (the convention moves them by at most 2.4e-4 and 2.3e-5 on the measured vanillas); `rho`
  and `theta` keep their definitions and units and re-solve on the bump context with the surface
  held fixed. At maturity it returns `price, delta, gamma`. A numerical vega through
  `GreeksCalculator` re-prices on that context's frozen layout; a fresh grid under the shifted
  surface differs from it by the layout's own movement (6.2e-6, 1.6e-5 relative). The
  structured vol-model risk's local-vol re-price is the solver's price with the surface held
  fixed, and so now includes a lagged product's settlement discount.
- `DCNPDEResult.num_time_steps` is the delivered step count (2,953 at `standard` on contract A,
  where it read 485), `num_space_nodes` the layout's; the DCN's horizon is its final observation
  date.
- A single- or double-barrier knock-in with a rebate (Fixed). A knock-in whose schedule records
  carry a payoff different from its rebate, or whose schedule aggregates other than
  `STOP_FIRST_HIT`, raises `ValidationError`; four 0.7.0 nodal-semantics goldens in
  `test_pde_base_family_projection` (knock-ins with per-record payoffs, two under `ACCUMULATE`)
  now assert that error.
- A discretely monitored single barrier's far-field row on the barrier's side is the certain
  knock-out at the next observation when the edge is beyond that observation's level; with no
  observation ahead, an edge inside the level, on the opposite side and under `ACCUMULATE` it is
  the vanilla asymptote, as before. EXPIRY cells move by −2.8e-4 / −1.2e-4 (constant surface);
  monthly and daily cells do not move to the printed digits; a two-year barrier observed only in
  its second year moves by 1.0e-5 and 1.5e-5. Under `bounds=(None, 140)` on a monthly up-and-out
  call the old row was off by +1.70 / +9.18, the new one by −8.8e-3 / −1.6e-2.
- The European, barrier and double-barrier PDE solvers price per contract (Fixed).
- A discretely monitored barrier, double barrier or touch product valued with the spot beyond a
  barrier and no observation due is solved, not short-circuited (Fixed).
- `calculate_spot_greeks_curve` and the session `GRID` output of a barrier-family product: a
  knock-in's rows are the composition of its parts, at the vanilla part's nodes by default; a
  knock-out's or touch product's rows at hit levels are the hit outcome with zero delta and
  gamma; a product decided at the base spot raises `CapabilityError` for a session grid and
  returns `reprice` rows from the public curve (Fixed).
- `DoubleBarrierOptionAnalyticalEngine`: every price with a rebate under continuous or discrete
  monitoring, and every EXPIRY price with the spot outside the corridor (Fixed). Rebate-free
  prices and EXPIRY inside the corridor are bit-identical.
- `DoubleSharkfinOptionAnalyticalEngine`: the continuous paid-at-hit rebate leg (up to 4.2e-6 at
  vol 22 % and 3.2e-5 at vol 8 % on a rebate of 5); every price where its 80-term series was
  unresolved; `quad_points` is keyword-only. Every other leg moves at round-off (at most 3.6e-15
  at equity parameters).
- A barrier-family product with a record on the valuation date: its decision there carries that
  record's level, cash and payment time and is applied at the spot, not written into the grid.
  At maturity `price`, `calculate_greeks` and the session outputs return one terminal value,
  which carries the participation and the terminal payment discount `DF(T_pay) / DF(T)`, and its
  delta the same factors; a touch product's `calculate_greeks` there agrees with its `price`; an
  expired knock-in prices. `EuropeanPDESolver`'s session outputs and `AmericanPDESolver`'s
  `calculate_greeks` and session outputs at expiry carry their `price`'s payment discount
  (Fixed).
- `DoubleBarrierOptionAnalyticalEngine` and `DoubleSharkfinOptionAnalyticalEngine` raise
  `NumericalError` on an input whose no-touch probability is not resolved to `Tolerance.ZERO`.
  On the double barrier that takes a vol below 1e-4 and a forward carried onto a barrier; inside
  the market objects' limits nothing with a vol of 1e-4 or more is refused, and the sharkfin's
  floor is 1e-3.
- A prepared session asked for more than `PV` with a lifecycle state: the facade's values on an
  engine that supports a state, `CapabilityError` on one that does not (Fixed).
- **The knock-in-side far-field row** under discretely observed knock-in, while a knock-in
  observation remains: every European-knock-in Snowball, Phoenix and KO-reset price, flat,
  local-vol and 2D (Fixed). Under monthly or daily knock-in it moves at round-off.
- Nodal coefficient sets never collapse to the scalar stencil: a one-step local-vol solve no
  longer returns the flat-vol price (5.676243 exactly). Bitwise on every byte-book row.
- Sessions and the registry: `EquityLVPDESessionAdapter` is grid-capable (the artefact-chain
  `prepare` moved up from the local-vol autocallable adapter, bitwise for the local-vol Snowball
  and Phoenix). `LocalVolPDESolver`, `LocalVolBarrierPDESolver` and `LocalVolDCNPDEEngine`
  register on it, `DCNPDEEngine` on `EquityPDE1DSessionAdapter`. A prepared price equals the
  direct one bit for bit for the four, and a `PV + GRID` session's rows equal the public curve.
  `execution/inventory.py` and `docs/execution/capability-matrix.md` list the four as
  prepared-capable.
- `ROUTE_RULES` keeps the 2D row only. Its refusal now ends "These are the 2D route's limits; no
  spec is committed to moving this route onto the shared core".
- Goldens: nine replay frames moved, all one Snowball with discretely observed knock-in in the
  scalar-BSM and local-vol books on its one not-knocked-in day (2024-01-02): price −3.18e-5 /
  −3.28e-5, and every later row's P&L by the same offset. Every `book_*` frame is byte-identical
  and no execution golden moved (`test/replay_golden/README.md`, the 2026-10-06 section).
- Examples, re-run against the 0.9.0 tree: `dcn_gate_report_demo` DCN-A PDE −93,294.8 →
  −93,291.5 and DCN-B −150,768.0 → −150,579.8 (|dPV| against Monte Carlo 115.6 → 112.3 and
  277.0 → 88.7); `vol_models_demo` local-vol PDE 8.8317 → 8.8275, Dupire market-IV vega 38.6555
  → 38.7113; `bucketed_greeks_consolidated_api_demo` Dupire market-IV parallel 0.32127 → 0.32395
  and `strike_column[2]` 0.16597 → 0.20443; `mo_volmodels/07_barrier_exotic` local-vol PDE 23.402
  → 23.338 (gap to Monte Carlo 1.7 % → 1.4 %); `mo_volmodels/09_delta_hedging` local-vol delta0
  0.3736 → 0.3733, gamma0 8.1e-5 → 2.7e-4. `mo_volmodels/03_dupire_localvol`'s overall LV RMSE
  goes 4.519 → 4.872 vol-points, which is not a regression: against the converged price of the
  same Dupire surface the core's 300 × 150 solve errs 0.017 rms (0.043 max) and the kernel's
  0.57 rms (2.05 max), so the kernel's discretization error partly cancelled Dupire's fit error.
  `mo_volmodels/11c_grid_preflight.py` binds on the solver's `grid_binder`.

### Removed
| Removed | Replacement |
|---|---|
| `GridLayerMixin` (and its export from `pde.grid`) | `BasePDESolver` |
| `LV_UNIFORM_ACCURACY_PROFILES`, `DCN_ACCURACY_PROFILES`, `resolve_config(profiles=)`, `GridBinder(profiles=)` | `ACCURACY_PROFILES` |
| `DAMPING_DEFAULTS["local-vol"]`, `DAMPING_DEFAULTS["DCN"]` | `DAMPING_DEFAULTS["core"]` |
| the `"local-vol vanilla"`, `"local-vol barrier"` and `"DCN"` rows of `ROUTE_RULES` | none: the fields are honoured |
| `DCNPDEEngine.price_detailed(layout=)`; the constructor's `points >= 201` check | `PDEParams(grid=…)`; the shared presets |
| `dcn_pde_solver._sinh_grid`, `_operator_coeffs`, `_theta_step`; `LocalVolDCNPDEEngine._step_vol` and its `1e-8` floor | the core march; the mixin's midpoint local-vol rows |
| `price_barrier_lv_pde`, `_barrier_boundary_value`, `_align_discrete_barrier_midcell`, `_solve_ko_lv` (the barrier half of `quantark.volmodels.localvol.pde_kernel`) | `LocalVolBarrierPDESolver`; the functions live on in `test/pde_routes/lv_barrier_kernel.py` as the test oracle |
| `quantark.volmodels.heston.pde_kernel._deterministic_barrier_price`, `quantark.volmodels.slv.slv_pde_kernel._deterministic_slv_barrier` | the engines' deterministic limit on the core (`DeterministicVarianceLocalVol` on `LocalVolBarrierPDESolver`) |
| `DCNLocalVolPDEAdapter`, `_dcn_pde_adapter` | `EquityLVPDESessionAdapter` |
| the `PricingError` on EXPIRY monitoring in `BarrierPDESolver` | EXPIRY monitoring (Added) |
| `DoubleBarrierOptionAnalyticalEngine._prob_inside_at_expiry` | `analytical/double_no_touch.py` |
| `DoubleSharkfinOptionAnalyticalEngine(max_terms=…)`, its `DEFAULT_MAX_TERMS`, `_survival_probability_continuous`, `_weighted_sine_integral` | `analytical/double_no_touch.py`; there is no term count to set |

### Fixed
- **The contract multiplier on the European, barrier and double-barrier PDE solvers.** Their
  value at `contract_multiplier` 10 was their value at 1 (ratio 1 in price, delta and gamma),
  where the analytic engines, the kernel local-vol routes and `AmericanPDESolver` scale by 10. An
  expired product did scale, so the route disagreed with itself, and a knock-in's part products
  were built without the multiplier. The payoff, event and rebate cash, the Dirichlet rows, the
  hit outcomes and the knock-in's parts now carry it, `(participation × option + rebate) ×
  multiplier`. Multiplier 1 is bit-identical.
- **The valuation-date hit test under discrete monitoring.** A spot beyond a discretely
  monitored barrier with no observation due returned the rebate or the vanilla; it is now solved
  from its spot. Error against the exact discrete recursion: monthly up-and-out call, rebate 5,
  spot 135 / 131, −0.202 / −0.346 → +5.3e-5 / +8.3e-5; observed only in months 7 to 12, −1.08 →
  +1.2e-4; monthly down-and-out put 85, rebate 3, spot 80, +3.2e-2 → −2.0e-5; monthly up-and-in
  call at 135 / down-and-in put at 80, +0.596 / +9.8e-2 → −1.7e-5 / −1.9e-5; monthly double
  knock-out 80 / 130, rebate 5, spot 135, −8.6e-2 → +5.8e-6; monthly one-touch paid at expiry,
  spot 135, +7.9e-2 → −3.5e-6; an `ACCUMULATE` schedule paying 5 monthly at spot 135 (closed
  form 72.55), −67.7 → −1.1e-4. The touch solvers move by 0.08 (monthly) to 3.7 (semiannual). The
  double barrier's `calculate_greeks` at a continuous hit took the rebate undiscounted (5.000000
  for 4.962640 under a quarter-year lag at 3 %); it now takes `price`'s value.
- **The knock-in rebate.** The knock-in composite built its knock-out leg with rebate 0 and
  dropped the rebate. On a rebate of 5 the single barrier was off by −3.747 (continuous up-and-in:
  the rebate-free price), −2.588 (continuous down-and-in) and −4.852 (monthly: the whole `R ×
  DF`), the double barrier (corridor 80 / 130) by −2.2205 / −2.2206 (call / put). A knock-in is now
  `participation × vanilla + R × DF(0, T_pay) − leg(R)`: +9e-5 to +4.9e-4 on 8 single-barrier
  rebate cells, the rebate-free cells' own error, and +2.5e-4 / +1.2e-4 on the double barrier. A
  knocked-in product forfeits the rebate; `R = 0` is bit-identical.
- **A knock-in's session grid and spot curve, and a knock-out's rows at hit levels.** A knock-in
  was marched as it stands under `SESSION_OUTPUTS`: on a continuous up-and-in call 130 the public
  curve was off by −4.19 at the money and −51.7 at 150, and a session's `PV + GRID` returned
  2.85597 for 6.93002. The rows are now the composition of the parts, within 5.2e-4 in price,
  6.6e-5 in delta and 7.5e-5 in gamma of the closed forms and the exact recursion. A knock-out's
  or touch product's row at a hit level carried the edge node's value and its one-sided
  gradients; it is now the hit outcome with delta and gamma 0 (from −0.150 on an up-and-out call
  with no rebate, −0.011 with a rebate of 5, +0.028 on a one-touch), and the double knock-out's
  price there goes 4.85223 → 5.00000, the facade's value. A product decided at the base spot
  returned a grid although the facade's hit test never ran; it now raises `CapabilityError` for a
  session grid, and the public curve returns `reprice` rows.
- **The analytic double-barrier engine's rebate terms and its EXPIRY branch outside the
  corridor.** The knock-out added `R × DF` on every path and the knock-in used the probability of
  ending inside the corridor. Corridor 80 / 130, strike 100, one year, rebate 5: continuous
  knock-out +2.221 / +1.533 / +1.125 (spots 100 / 90 / 120) → at most 4e-15, knock-in +1.309 /
  +1.629 / +1.859 → at most 5e-15; monthly knock-out +1.80 to +2.95 → +0.017 to +0.093 and
  knock-in +0.90 to +1.53 → −0.017 to −0.093, the barrier shift's own error. Under EXPIRY a spot
  outside the corridor returned the rebate or the vanilla, −3.350 to +3.497 on eight cells; the
  closed form now applies from any spot, at most 1.1e-14.
- **A record on the valuation date.** The march wrote the valuation-date event into the nodes,
  and the facade tested the product's own barrier. On an up-and-out (barrier 130, rebate 5) with a
  record at 120 paying 7 (worth 6.947696) the spots from 120 outward returned 5.00 for 6.95, and
  just inside the record's level the price was off by −3.7e-2 / −5.2e-2 (delta −0.27). The
  decision now carries the record's level, cash, payment time and operation and is folded at the
  spot onto the continuation: exact where it fires, +5e-6 / +1.2e-4 just inside. A value read
  next to a record crossed the jump, up to +2.52 in price and +13.3 in delta on an up-and-in call
  130 at 129.9 to 129.99; now within 2.0e-4 and 8e-6. On 168 cells (single and double knock-out,
  both one-touches and both no-touches, lags 0 and 0.25, flat and kinked curves) a fired record
  is its cash to the last bit and the rest is within 2.8e-4 in price and 1.0e-5 in delta of the
  term recursion, where the old values were off by −1.2 to −3.8 at the record's lower level, +1.4
  to +4.5 just inside its upper level and −1.7 to −2.0 beyond the product's barrier. Under
  `ACCUMULATE` at and beyond the level: −37.6 to −43.0 → within 1.4e-4.
- **The barrier family at maturity.** At `tau = 0` the expired path returned `product.get_payoff`:
  no payment discount, no participation, and for a touch product a number that ignored the touch;
  `calculate_greeks` disagreed with `price`, and an expired knock-in raised. A value is now the
  terminal payoff × participation × `DF(T_pay) / DF(T)`, then the decision, under every
  operation. A knocked-out product's `calculate_greeks` returns the cash, not the payoff (35 →
  5); an expired knock-in prices and pays its rebate when untriggered (`calculate_greeks` 25 →
  5) and takes a bump context; a maturity record's own level and cash apply (25 → 7); a schedule with no maturity record
  does not test the barrier there (5 → 35). With the terminal payment a quarter-year after
  maturity at 3 %, a surviving knock-out, an untouched no-touch and a triggered or untriggered
  knock-in take the discount (25.000000 → 24.813201, delta 1 → 0.992528), and the value no longer
  jumps by 0.187 at the last instant. Participation 0.5 applies (25.000000 → 12.406601). A touch
  product's `calculate_greeks` there agrees with its `price`: a touched one-touch reports its cash
  (0 → 5), a touched no-touch nothing (5 → 0). `AmericanPDESolver.calculate_greeks` and session
  outputs at expiry under the same lag (25.000000 → 24.813201, delta 1 → 0.992528) and
  `EuropeanPDESolver`'s session outputs (25.000000 → 24.813201) take their `price`'s discount. An
  expired product with no lag, participation 1 and multiplier 1 is bit-identical.
- **The lifecycle state of a `PV + GRID` session on a European.** The session passed the state to
  `price` on the `PV` path and to nothing on the other. A European call with one realized, unpaid
  cashflow of 7 due in 91 days returned the live 9.601810 under `PV + GRID` whether terminated
  (price 6.947839) or alive with the receivable (16.549649). Now a terminated claim's grid fails
  closed (`CapabilityError`), and a live one returns 16.549649 with 6.95 in each row. The barrier
  and Snowball engines, which do not support a state, returned 2.671787 and 97.157139; they now
  raise `CapabilityError`, as for `PV`. A request with no state is bit-identical.
- **The double sharkfin's no-touch probability.** The engine's 80-term eigenfunction series had
  no convergence control: corridor 80 / 130, spot 129.9, vol 1 %, one day returned 0.248354 for
  0.858487, putting a rebate of 5 off by 3.050664, and over 637 cells its worst error was 1.0 at
  dimensionless times up to 1e-3 (two days at vol 20 % on that corridor). It now reads
  `double_no_touch` at its three call sites. Its continuous paid-at-hit factor, which was off by
  9e-8 to 6.3e-6, is within 1.3e-9 at spots 85 and 100 and 6.2e-8 at 127 (one year, vol 22 %).
  The price moves by up to 4.2e-6 (vol 22 %) and 3.2e-5 (vol 8 %) on a rebate of 5, and by up to
  a whole rebate where the series was unresolved.
- **The Heston and SLV barrier solvers' vol-of-vol → 0 limit.** Below `_SIGMA_MIN` (1e-4)
  `HestonBarrierPDESolver` and `HestonSLVBarrierPDESolver` used the 1D local-vol kernel at a
  constant effective vol `sqrt(average variance)` (for SLV, with a 15 × 5 tabulation of the
  leverage). That is exact only for Europeans; for a barrier it was a model error. They now march
  the exact deterministic-variance local vol `L(S, t) sqrt(v(t))`, `v(t) = theta + (v0 − theta)
  e^(−kappa t)` (`DeterministicVarianceLocalVol`), on `LocalVolBarrierPDESolver` at its standard
  preset, after the 2D solvers' own refusals. Measured at the standard preset against a
  3,201-point, 40 steps-per-day reference: with flat variance against the Black–Scholes barrier
  closed form, core 1.84e-4 (continuous up-and-out call 130) / 3.17e-4 (down-and-in put 80),
  old 3.06e-4 / 4.58e-3; monthly up-and-out 125, core 1.27e-4, old 2.08e-3; with decaying
  variance (v0 0.09, theta 0.04, kappa 1.5) the old route was 7.0e-3 to 3.25e-2 from the exact
  limit, the core within 2.0e-4; SLV with a skew leverage, old 1.42e-2 to 5.23e-2, core within
  1.9e-4. Heston and SLV barrier prices at a vol-of-vol below 1e-4 move by these amounts. The
  grid spans the model's vol, the larger of the environment's strike vol and the largest
  `sqrt(v(t))` to maturity (`DeterministicVarianceLocalVol.domain_vol`), so an environment
  quoting less than the model does not narrow it. The
  low-level `price_barrier_heston_pde` / `price_barrier_slv_pde` now raise `ValidationError`
  below `_SIGMA_MIN`, naming the engine.
- **The knock-in-side far-field row.** The not-knocked-in lower row of a standard product was
  "principal plus rebate to maturity", as if the far edge never knocked in; with one knock-in
  date at maturity nothing resets it. Under discretely observed knock-in, European included, that
  row now takes the knocked-in row (the upper row for a reverse product) while a knock-in
  observation remains at or after the row's time; after the last one, on an exhausted schedule,
  or for a KO-reset after its first schedule, it is the old row. Error against the exact recursion
  (QUAD for the reverse product): Snowball with European knock-in, spot 100, standard / high,
  +4.37e-3 / +4.25e-3 → +4.2e-5 / −4.9e-5; spot 86, +4.73e-3 / +4.75e-3 → −1.6e-4 / −8.9e-5; vol
  35 %, +5.03e-3 / +5.10e-3 → −2.0e-4 / −4.4e-5; Phoenix with knock-in at maturity only, spot 100
  / 80, +3.3e-3 / +4.0e-3 → −1.7e-4 / −1.2e-4; reverse Snowball, spots 100 / 114 / 98, +2.96e-3 /
  +3.88e-3 / +1.98e-3 → −1.8e-4 / −1.2e-4 / −2.3e-5; Snowball under `bounds=(60, None)` / `(70,
  None)`, +1.255 / +4.295 → −1.2e-3 / −0.129. On the byte book's flat market against QUAD at
  32,001 points: Snowball price +4.17e-3 → +3.0e-5 (exact recursion +4.17e-3 → +2.8e-5), Phoenix
  with knock-in at maturity +3.03e-3 → −2.2e-4, reverse Snowball +7.61e-3 → −5.3e-4; bump vega
  +5.18e-3 → −2.2e-5, +4.08e-3 → −7.5e-6, +8.90e-3 → −1.2e-5. The reconciliation identity stays
  at round-off, the knock-in probabilities do not move, and the maturity leg's error falls from
  +4.70e-3 to +3.7e-4.
- **EXPIRY monitoring on the double-barrier and touch PDE solvers.** `DoubleBarrierPDESolver`,
  `OneTouchPDESolver` and `DoubleOneTouchPDESolver` tested the barrier during the march under
  `ObservationType.EXPIRY`, close to continuous monitoring: at spot 100, vol 20 %, r 3 %, q 1 %,
  one year, 0.9.0 priced a 80/130 double knock-out call at 3.144 (exact 4.848), an up one-touch
  at 120 paying 10 at 3.428 (exact 1.756) and a 85/120 double one-touch at 7.237 (exact 3.777).
  A barrier observed only at maturity makes the value a function of the terminal spot alone, so
  before maturity these three solvers now price it by static replication: European vanillas and
  cash-or-nothing digitals on the product's own expiry and settlement, priced on
  `BlackScholesEngine` and `DigitalOptionAnalyticalEngine` (a knock-out is the vanilla inside
  the corridor plus the rebate outside it; a knock-in the vanilla outside plus the rebate inside;
  a touch product its digitals). Against quadrature of the terminal value the price is within
  1e-9 relative, with and without a payment lag, and knock-out plus knock-in equals the vanilla
  plus the rebate on a flat and a smile market. Each leg reads its own strike's vol, so under a
  smile the value is the smile's static replication, not the march's single strike-selected vol.
  No PDE grid is requested or bound for such a product: bump contexts return the engine itself,
  a session prepares no artefact, and a session `GRID` must name its levels
  (`grid_spot_levels`; without them `CapabilityError`, as there are no nodes). Greeks follow
  `GreeksCalculator`: the new hook `BaseEngine.static_replication` exposes the legs, and
  `GreeksCalculator.calculate` returns the weighted sum of its own greeks of each leg under the
  caller's method, request and bump configuration (closed forms for the vanillas under `auto` /
  `analytical`, bumps for the digitals, its own refusals per leg). The solver's
  `calculate_greeks` and the session rows (`calculation_mode` `"static_replication"`) are
  `GreeksCalculator(params=solver.params)` on the legs: delta within 1.27e-3 and gamma within
  3.6e-5 of the exact derivatives (relative, 80 cells). The single barrier keeps its EXPIRY
  far-field row; at maturity the terminal value applies as for every barrier product.
- **`GreeksCalculator`'s closed forms under a terminal payment lag.** The analytic route
  (`method="auto"` / `"analytical"` on a European vanilla) ignored `settlement_convention`: at a
  quarter-year lag an at-the-money call returned price 96.0158, delta 5.7385, rho 4.7783 where
  `BlackScholesEngine.price` and the bumped route give 95.2983, 5.6950, 4.5049. The closed forms
  now take the lag as the engine prices it, `V = DF(T, T_pay) V_T`: price, delta, gamma, vega,
  dividend rho, theta (and its three-way split) and the extended closed forms scale by
  `DF(T, T_pay)`, and rho adds the payment's own rate sensitivity, `-(T_pay - T) V_T` (4.5044).
  At expiry the payoff is discounted over the lag, rho is `-lag V` and theta its accretion. A
  vanilla paid at expiry is bit-identical.

### Breaking changes
- `GridLayerMixin` and its export from `pde.grid` are removed; a solver subclasses
  `BasePDESolver`.
- `LV_UNIFORM_ACCURACY_PROFILES` and `DCN_ACCURACY_PROFILES` are removed, and the `profiles=`
  keyword is removed from `resolve_config` and from `GridBinder`: every 1D route takes the shared
  `ACCURACY_PROFILES`.
- `DAMPING_DEFAULTS["local-vol"]` and `DAMPING_DEFAULTS["DCN"]` are removed; the routes use
  `DAMPING_DEFAULTS["core"]`.
- `DCNPDEEngine.price_detailed` no longer takes `layout=` (passing it raises `ValidationError`
  naming `PDEParams(grid=GridConfig(...))`, from `REMOVED_DCN_NAMES`), and the constructor's
  `points >= 201` check is gone. `DCNPDEResult.num_time_steps` is the delivered step count. A 2,401-point DCN
  grid needs `GridConfig(max_points=2401)`.
- `LocalVolPDESolver.calculate_greeks`: `delta` and `gamma` are pointwise, not 1 % bumps.
- The barrier half of the 1D local-vol kernel (`price_barrier_lv_pde` and its helpers) leaves
  the library for `test/pde_routes/lv_barrier_kernel.py`.
- New `ValidationError`s: a knock-in whose schedule records carry a payoff different from its
  rebate, or whose schedule aggregates other than `STOP_FIRST_HIT`; `price_barrier_heston_pde` /
  `price_barrier_slv_pde` below `_SIGMA_MIN` (the 2D barrier kernels refuse the vol-of-vol → 0
  limit and name the engine that prices it).
- New `CapabilityError`s: a session grid for a barrier-family product decided at the base spot;
  a stateful rich-output session request (`PV + GRID`, event statistics) to an engine without
  lifecycle support; a session grid for a terminated European.
- `DoubleSharkfinOptionAnalyticalEngine`: `max_terms` is removed and `quad_points` is
  keyword-only, so a second positional argument written for `max_terms` raises.
- New `NumericalError` from `double_no_touch`, and so from `DoubleBarrierOptionAnalyticalEngine`
  and `DoubleSharkfinOptionAnalyticalEngine`, on an input whose no-touch probability is not
  resolved to `Tolerance.ZERO`.

### Per-stage deltas (PDE byte book)
Stage 1, the barrier family:
- (h) analytic engines: 510 rows, 0 diffs (no PDE file changed).
- (e) multiplier: 510 rows bit-identical (every row has multiplier 1); 5 added (`3c|mult10|*`).
- (f) valuation instant: the record-level event builders bit-identical (515 rows); then exactly
  the eight `3a|farfield|{di_put_85,do_call_85}|*|80|*` rows move, equal to the evidence's
  `r5_book_hit_rule.json` bit for bit, up to 0.5762 in price, 0.1878 in grid delta and 4.306e-2
  in grid gamma (`3a|farfield|do_call_85|kinked|80|lag0`); 21 added.
- (f) maturity: 536 rows bit-identical, both expired barrier rows included; 102 added.
- (b) EXPIRY and the far-field row: discrete-barrier rows at round-off (bump rho 5.3e-13, bump
  vega 4.1e-13); `3c|vdec|uo_late|beyond` 3.39e-5 in price, toward the exact recursion (+6.1e-5
  → +2.7e-5 off), and `3c|vrec|uo|119.99` (+8.6e-5 → +8.0e-5); the `3a|farfield` rows
  bit-identical; 5 added.
- (g) session grid: 643 rows bit-identical; 41 added.
- (g) lifecycle on rich outputs: 684 rows bit-identical; 3 added.
- (a) knock-in rebate: 687 rows bit-identical; 16 added.
- (c) per-part bump layouts: eight `knock_in|*|bump` rows go from `PricingError` to values;
  `knock_in|down_in_put|knocked_in|bump` within 6.1e-5; `3a|base6|di_put|{default,v070}` (a
  monthly down-and-in put) in bump Greeks only, gamma 1.82e-4, delta 1.63e-5, vega 4.24e-6, rho
  9.80e-7, theta 1.07e-7 (the evidence's both-frozen arm); 1 added.
- (d) nodal sets: 704 rows, 0 diffs.

Stage 2, the routes:
- (0) `LocalVolSurfaceMixin`: 704 rows, 0 diffs.
- (a) local-vol vanilla: `lv_vanilla|{default,600x200}`, price up to 2.31e-3, grid gamma 2.08e-3,
  grid delta 8.67e-5; against a fine-grid core ladder the price error goes 2.57e-3 → 2.60e-4 /
  1.14e-3 → 2.79e-4 and delta 8.65e-5 → 2.5e-7 / 5.2e-5 → 5.8e-7; 3 added.
- (b) local-vol barrier: six `lv_barrier_*` rows, price up to 2.01e-3
  (`lv_barrier_discrete|default`). Against the oracle's Richardson references five move toward
  (discrete default 1.75e-3 → 2.5e-4) and `lv_barrier_cont|600x200` away by 1.8e-5 (6.2e-5 →
  8.0e-5). The local-vol autocallables are bit-identical; 2 added.
- (c) DCN: `dcn|default` and `dcn|601`, pv up to 336.1 (`dcn|601`, 3.4 bp of 1e6). Against the
  exact recursion `dcn|601` 3.51 → 0.15 bp and `dcn|default` 0.070 → 0.102 bp (an 801-point plan
  → the 400-point preset; away by 0.03 bp, inside its 1 bp bound); `lv_dcn` bit-identical; 3
  added.
- (d) local-vol DCN: `lv_dcn|default` and `lv_dcn|601`, pv up to 148.6 (`lv_dcn|601`). Against
  the exact recursion on the rows' constant 22 % surface 1.21 → 0.086 bp and 1.52 → 0.038 bp; 1
  added.
- (e) deletions: no value moves; the two declared-error rows
  `3b|{heston,slv}_snowball|half_step|stats` change text; 8 European-knock-in rows added
  (`3c|eki|*`).

Stage 3, the knock-in-side row: 721 rows, 0 unexpected diffs. The 15 rows of the evidence's r4
list equal `r4_book_gated.json` bit for bit, at most 1.11e-6 (`intraday_snowball`). The
`3c|eki` rows move in price by −4.14e-3 (Snowball), −3.25e-3 (Phoenix) and −8.14e-3 (reverse
Snowball), in bump vega by up to 8.91e-3, and the KO-reset by 6.8e-7 (a pre-maturity
statistic); each move was measured toward the independent reference (Fixed). The
`3c|eki|phoenix` row was rebuilt as a knock-in-at-maturity Phoenix; it had been the monthly one.

### Certification
The knock-in-side row moves 31 values in 10 certified cells, each equal bit for bit to its
pre-measured value (`r4_anchors_gated.json`); every other anchored value is unchanged and every
decision is unchanged. The six studies are amended and banked under
`docs/modelvalidation/certificates/{snowball-flat-bsm,phoenix-flat-bsm,ko-reset-flat-bsm,snowball-localvol-1d,adi2d-snowball-greeks,snowball-intraday-daily-ki-bsm}/2026-10-06/`,
their parents retired in place. The four schema-1 studies were amended with `--rerun-candidate`
on their PDE candidate; `adi2d-snowball-greeks` through the external-benchmark path (24 cells
carried bit for bit, 4 translated); the intraday study's reference re-ran value-identical.

| Study | Values / cells | Cells | Largest move |
|---|---|---|---|
| snowball-flat-bsm | 3 / 1 | `european_ki` | 4.33e-3 (pv) |
| snowball-localvol-1d | 12 / 4 | `calm_discrete_ki`, `calm_european_ki`, `crash_discrete_ki`, `crash_european_ki` | 0.165 contracts (`calm_european_ki` pv) |
| ko-reset-flat-bsm | 3 / 1 | `european_ki` | 5.4e-7 |
| snowball-intraday-daily-ki-bsm | 6 / 1 | `near_ko_on_ko_day_1s` | 5.4e-10 |
| phoenix-flat-bsm | 3 / 1 | `ki_stepdown` | 1.9e-15 |
| adi2d-snowball-greeks | 4 / 2 | `heston_pde\|near_expiry`, `heston_slv_pde\|near_expiry` (delta, gamma) | 1.8e-15 |

The local-vol European knock-in pv errors against the certificate's reference go +0.165 →
−0.0002 (calm) and +0.091 → −0.034 (crash) contracts; the flat `european_ki` pv goes from
+4.4e-3 to +4e-5 against the exact value. The adi2d moves are translated by the harness's own
interval arithmetic, shifting its intervals by at most 9.8e-14 contracts. Both `amend` changes
are listed under Added.

### Performance
Timing (`test/pde_bitwise/timing.py`, 0.9.0 and 0.10.0 interleaved, 15 repetitions per arm,
ARM64 macOS; cold medians in ms). A case whose 15 cold times spread by more than 10 % of the
median on either side is inconclusive, never a pass. The four statistics cases were
re-measured one family per process: their joint run spread by 24 to 81 %.

The untouched routes, gated at 1.05×: American 1.007×, barrier 1.013×, Snowball 1.007×,
Phoenix 1.018×, KO-reset 1.010×, local-vol Snowball 1.004×, Snowball statistics 1.015×,
Phoenix statistics 0.996×, KO-reset statistics 1.006×, local-vol Snowball statistics 1.006×:
none fails. The flat European is **inconclusive**: three runs gave 1.008×, 0.983× and 1.007×
with spreads of 10.8 to 28.5 % on a 23 ms single-shot case. Ungated: the memory and EXPIRY
Phoenix statistics, 1.005× and 1.001×.

The four routes moved onto the core are slower by design (the shared presets and the core's
time density, decisions 3 and 4); declared, not gated:

| Route (fast / standard / high) | `price` 0.9.0 | `price` 0.10.0 | `calculate_greeks` 0.9.0 | `calculate_greeks` 0.10.0 |
|---|---|---|---|---|
| Local-vol vanilla | 8.0 / 8.9 / 21.3 | 65.6 / 72.4 / 172 | 47.5 / 52.3 / 128 | 254 / 283 / 679 |
| Local-vol barrier, continuous | 9.4 / 10.3 / 24.4 | 71.1 / 69.0 / 166 | 28.0 / 30.7 / 73.3 | 70.4 / 68.2 / 167 |
| Local-vol barrier, daily | 12.6 / 13.7 / 31.0 | 112 / 122 / 288 | 37.8 / 41.0 / 92.6 | 112 / 123 / 239 |
| DCN | 30.6 / 39.4 / 42.1 | 90.5 / 103 / 219 | 76.5 / 98.0 / 122 | 90.5 / 103 / 218 |
| Local-vol DCN | 44.0 / 54.7 / 61.8 | 234 / 260 / 609 | 117 / 145 / 182 | 232 / 259 / 571 |

The double-barrier and touch products under EXPIRY build no grid (static replication).

### Known issues
- The analytic engines short-circuit a spot beyond a discretely monitored barrier although no
  observation is due: the double barrier returns the rebate (−0.086 on a monthly knock-out at
  spot 135), the single barrier the rebate or the vanilla (−0.202, +0.596, +0.032 on three
  monthly products); the double-sharkfin engine has the same test (read, not measured). At
  maturity the single-barrier analytic engine returns 0 for an untriggered knock-in's rebate,
  tests a schedule with no maturity record by the spot (5 for 35) and ignores a maturity
  record's own level (25 for 7). The PDE is right in all of these.
- Under `ACCUMULATE` a no-touch's interior records add 0, so only its maturity record ends it: a
  year out, records at 0.5 and 1, the PDE gives 4.304787 = `5 DF P[S_1 < 130]`, against 4.227537
  under first hit. Whether the mode has a meaning on a no-touch is a product question.
- `time_shift` drops a record that lands on the valuation instant, and when that empties the
  schedule it returns before shortening the maturity, so the product is not aged at all; a dated
  product valued on its exercise date raises.
- The double-barrier engine's Ikeda–Kunitomo option leg is fixed at 21 images: where the true
  value is below 1e-13 it returns 1.6e-5 (95 / 105, vol 30 %, 3 years) to 4.1e-4 (98 / 102, vol
  50 %, 1 year). It has no error control.
- The double sharkfin's continuous paid-at-hit leg integrates the survival curve with 48
  Gauss–Legendre nodes; next to a barrier (spot 129.9 on 80 / 130) the factor is off by 4.3e-7 to
  1.9e-6 on the exact probability.
- An explicit `GridConfig(bounds=...)` close to a barrier mis-states the far-field row: the
  autocallable row by −0.09 to −0.13 (−0.13 at 15 points below the knock-in barrier, against
  +4.3 before the knock-in-side row), the single-barrier row by 0.1 to 3.3, and the knocked-in
  upper row by −0.73 to −1.74 under a narrow upper bound. The right row there is a knock-in or
  knock-out probability; keep an explicit bound well beyond the barrier.

## [0.9.0] - 2026-10-04

PDE 1D event statistics (spec: `docs/superpowers/specs/2026-10-03-pde-1d-event-statistics-design.md`).
The Snowball, Phoenix and KO-reset statistics (and their local-vol variants) march in the
value's own groups: the value's boundary rows allocated per column, its θ rule, and the
adapter's event stages. The maturity leg is a marched column, so on every native 1D
route `pv = Σ knock-out cash + Σ coupon cash + maturity leg` to round-off and
`reconciliation_error` is a real number. Every stage was checked against the 510-row PDE
byte book with its declared movers. Prices move where a contract was mispriced: the
Phoenix release rule, the contractual coupon periods, the Phoenix knocked out at valuation
on a later observation, and the valuation-date decisions (a carried knock-in under
`disable_ko_after_ki`; a carried-knock-in KO-reset, which raised before). No certificate
identity moves.

### Added
- `AutocallableEventStats.reconciliation_error` is computed:
  `pv − Σ expected_discounted_cashflows`, round-off (≤ 1e-10) on native 1D routes.
- `AutocallableEventStats.provenance: StatisticsProvenance(statistics_model, price_model)`,
  `EventDistribution.provenance`, `EventDistribution.determination_times` (per maturity
  outcome; defaults to the last event time, so existing distributions are unchanged), and
  `StatisticsRefused` (a `PricingError`).
- Phoenix: `expected_discounted_coupon_cashflow` for INSTANT memory coupons and for
  EXPIRY coupons (per earning date, as QUAD V2). EXPIRY cash is also bucketed by payment
  event in the ledger, with `coupon_payment_is_path_dependent = True`.
- `KOResetEventStats`: `expected_discounted_pre_maturity_cashflow` /
  `…_post_maturity_cashflow`, `pre_maturity_probability` / `post_maturity_probability`,
  `pre_maturity_time` / `post_maturity_time`, their payment times, `pre_ko_payment_times` /
  `post_ko_payment_times` and `post_ko_times_are_offsets`. The PDE and QUAD V2 fill the
  buckets; legacy QUAD and Monte Carlo leave them `None`.
- `EventDistribution.from_autocallable_stats` accepts a `KOResetEventStats`.
- `KOResetSnowballPDESolver.calculate_event_stats` is native, and `streams` is honoured as
  on the Snowball.
- `PDEPlan.terminal["event_stats"]` can be `knocked_out_at_valuation`.
- The 2D Heston / SLV solvers' `statistics="refuse" | "flat_vol_surrogate"` option.
- `test/pde_statistics/`: an exact two-state reference for flat GBM, with its guards and
  a Monte Carlo qualifier, banked as JSON; a 32,001-point QUAD reference for KO-reset;
  the identity, column and stage tests.
- `PhoenixOption.active_ko_source_indices(env)`; `AccrualConfig.first_period_start` (where the
  first remaining coupon period started, recorded when an ageing drops an observation).

### Changed
- **Statistics policy.** Boundary rows, θ and event stages are the value's. Snowball
  probabilities move by ≤ 7.3e-8 on the byte book; the coupon-barrier smoothing moves
  Phoenix ones by ≤ 1.8e-6.
- **`expected_discounted_maturity_cashflow` is the marched leg**, not `pv − ΣKO cash`. It
  moves by the 0.8.0 residual (≤ 7.4e-6 on the byte book, 1.2e-4 on the study cases) and,
  on memory, EXPIRY and immediate-knock-out Phoenix, by the coupon cash it no longer holds.
- **2D statistics refuse by default.** `HestonSnowballPDESolver`,
  `HestonSLVSnowballPDESolver` and the Phoenix pair raise `StatisticsRefused` from
  `calculate_event_stats` / `price_with_events` unless built with
  `statistics="flat_vol_surrogate"`. That option returns 0.8.0's flat-vol 1D statistics,
  now with the 2D route check, the residual (the model difference, 0.26 on the book's
  Heston Snowball) and the provenance. The replay's `event_stats_fallback="mc"` no longer
  masks a refusal.
- **Phoenix release rule (decision 4).** A knock-out pays the memory arrears whatever
  the spot, plus the current coupon when its barrier is met, as QUAD V2 always did. This
  applies to the PDE value and statistics, the immediate-knock-out payoff,
  `decide_observations_at_valuation`, the lifecycle tracker and `PhoenixMCEngine`. A
  price moves only under memory with the coupon barrier above the knock-out barrier: on
  the study's dated product (coupon 105, knock-out 103) the PDE goes 97.811385 → 98.384223,
  against V2's 98.384358. Certified products are bit-identical.
- **Valuation-date decisions (decision 15).** The family's own
  `decide_observations_at_valuation`, with the carried knock-in state, decides a
  valuation-date observation on every entry point. The spot-versus-barrier shortcut is
  gone. A carried knock-in with `disable_ko_after_ki` no longer knocks out at valuation:
  the dated case moves 1.019178 → −0.674, against Monte Carlo's −0.664. A
  carried-knock-in KO-reset knocks out on its post schedule.
- `PhoenixOption.get_coupon_period_year_fractions` takes the pricing environment; a list
  or array argument raises `ValidationError` naming the 0.9.0 change. `accrual_factors` and
  a list coupon barrier are read by the observation's ORIGINAL schedule position, in the
  engines and in `decide_observations_at_valuation` (0.8.0 aligned them to the remaining
  records).
- An aged tenor Phoenix (no `initial_date`, a per-annum coupon) pays its first remaining
  period from the last dropped observation (`AccrualConfig.first_period_start`). Where that
  observation carries no time the period is unknown and a per-annum coupon raises
  `ValidationError` naming `fixed_coupon_year_fraction` and `accrual_factors`. A theta clone
  that crosses an observation is ex-coupon: on a 12 % tenor Phoenix one day before an
  observation theta is about −0.98 (the coupon paid); 0.8.0's −0.017 came from an
  over-accrued next period.
- `EventDistribution.from_autocallable_stats` raises `NotImplementedError` on a
  `KOResetEventStats` with `post_ko_times_are_offsets` (Monte Carlo under
  `PostKOScheduleMode.REBASED`) and `ValidationError` on one without its payment times.
- The legacy `ki_probability` on KO-reset PDE statistics is "settles knocked in", as on
  every PDE and QUAD route (the 0.8.0 QUAD delegation reported "ever"). The two explicit
  fields are unambiguous.

### Removed
| Removed | Replacement |
|---|---|
| `STATISTICS_LEGACY`, `GroupPolicy.zero_slope_after_events`, `GroupPolicy.dt_as_python_float`, `BoundaryPolicy.CARRY` | the `VALUE` policy; a rolling group supplies its rows |
| `StatisticsLegacy` and `core/statistics/legacy.py` | `core/statistics/group.py` and the families' groups |
| the Snowball / Phoenix statistics hooks (`_n_extra_event_cols`, `_set_extra_event_indicators`, `_extract_extra_event_stats`, `_t0_extra_indicator_overrides`) | the column layouts of `core/statistics/columns.py`, the adapter's stages, the valuation-date rules |
| the KO-reset solver's QUAD import and its `grid_points=501` delegation | native KO-reset statistics |
| `reconciliation_error = 0.0` hard-coded on the three PDE paths | the computed residual |
| the 2D solvers' silent inherited statistics | `statistics="flat_vol_surrogate"` |
| `_is_knocked_out_at_valuation` on the price and statistics paths | the family's `decide_observations_at_valuation` |
| `PhoenixOption.get_coupon_period_year_fractions(observation_times)` | `get_coupon_period_year_fractions(pricing_env)` |

### Fixed
- **`disable_ko_after_ki` statistics at maturity.** 0.8.0's Snowball statistics let
  knocked-in paths knock out on the final date. The last knock-out probability read
  4.06e-2 instead of 2.26e-3 (continuous knock-in) and 2.42e-2 instead of 4.69e-3
  (discrete), and `ki_survive_knocked_in_probability` was short by the same mass. Prices
  were right. Now within 2.1e-5 / 2.6e-6 (knock-out) and 1.9e-4 / 2.0e-5 (`ki_survive`)
  of QUAD V2 (`test/pde_statistics/test_disable_ko_after_ki.py`). Anyone who used 0.8.0
  (or earlier) PDE statistics on a `disable_ko_after_ki` Snowball for knock-out timing,
  expected life or the knocked-in loss probability should re-run them.
- **KO-reset statistics and conversion.** 0.8.0 ran a 501-point QUAD and folded
  `pde_pv − quad_pv` into the maturity leg. Pre-schedule knock-out probabilities were off
  by 3.8e-3 / 2.0e-3 / 1.4e-3 (spots 100 / 90 / 101) against a 32,001-point QUAD; native,
  they are within 1.3e-5 / 2.0e-5 / 6.9e-6, post-schedule within 3.8e-5, and the identity
  holds to 7e-14. The event distribution read only the pre schedule, carrying a maturity
  mass of 0.294 where 0.200 of the paths survive both. It now has both schedules as
  knock-outs and the two maturity outcomes at their own determination times.
- **Memory Phoenix coupon cash.** INSTANT memory coupons had no coupon cash; it sat in the
  maturity leg. On the study's product the coupon cash went from not reported to
  0.736235 (V2 0.736239) and the leg from −3.34829 to −4.084522 (V2 −4.084004); the identity holds to 9e-16.
- **EXPIRY Phoenix coupon cash.** Deferred coupons are reported per earning date and
  bucketed at each termination's payment time, with the path-dependent flag (no cash
  leg reads the buckets yet; below).
- **Knocked-out distribution.** `price_with_events` on a product knocked out at valuation
  returned `{MATURITY_NO_KO: 1.0}` at maturity, so a cash leg treated the note as alive.
  It now returns the knock-out at its settlement, the same as `calculate_event_stats`.
- **Phoenix knock-out coupon.** On a Phoenix knocked out at valuation, the coupon paid with
  the knock-out (1.019178 on the byte book) was in the maturity leg; it is now coupon cash
  and the leg is 0, as QUAD V2. The price and the statistics took the FIRST period's coupon
  and barrier whatever the observation: on a dated Phoenix knocked out on its second
  observation the PDE paid 1.019178 where the contract pays 1.052055, and with a list
  barrier not met at that observation it paid a coupon the contract does not. Both now take
  the product's own decision (`test_a_later_observation_knock_out_pays_the_decided_coupon`).
- **Lifecycle-knocked-in `ki_survive`.** A product carrying the lifecycle knock-in flag
  reported `ki_survive_knocked_in_probability = 1` although most paths may knock out
  first. It is now the marched value: 0.277176 on the measured Snowball (1 − ΣKO =
  0.277176). With the knock-in streams pruned it reads 0.0, like every pruned field.
- **Contractual coupon periods (decision 13).** A dated Phoenix valued after its start
  gave its first remaining period the fraction from the valuation date (0 on an
  observation date) in the PDE and Monte Carlo, while QUAD V2 used the contract. Byte
  book: observation-date pay +1.019178 and miss +0.7581, ten days in +0.4596 (101.683759 and
  86.114188 against V2's 101.684019 and 86.114162). The deterministic control is exact.
  Tenor Phoenix theta moves by about one day of coupon, because 0.8.0's theta clone
  truncated the first period: up to 1.0e-2 on the byte book, 3.3e-2 on a 12 % tenor
  Phoenix (−0.01685 → +0.01595).
- **Valuation-date regimes.** A carried knock-in's regime decides (above).
- **Dated KO-reset `KeyError`.** A dated KO-reset valued on its first observation date
  raised `KeyError` (the post observation was not a grid node); both schedules' times are
  now in the grid request.

### Known issues
- EXPIRY coupon cash reaches no cash leg: `price_with_events` carries no ledger, and a leg
  that needs path-dependent coupon timing raises as it does for QUAD V2.
- `ki_ever` under continuous knock-in carries the first-passage correction's error
  (1e-4 to 9e-4) and does not converge under time-only refinement.
- `ki_ever` counts a breach after a knock-out, on every engine (a definition).
- The value's far-field rows ignore the certain far-edge knock-in and knock-out, and the
  Phoenix upper row omits the next coupon. None of this is visible at the spot on the
  measured cases.
- The 2D statistics are the flat-vol model's (surrogate only; native 2D is a later spec).
- KO-reset maturity buckets against QUAD V2: the PDE's `M_pre` is −1.5e-2 (cash) from V2's
  at spot 100 (−2.0e-2 at `high`; about 1.5e-4 in probability); the gate is 3e-2.

### Per-stage deltas (PDE byte book, 510 rows)
- (a) Snowball value policy: 97 rows. Leg ≤ 7.4e-6, probabilities ≤ 7.3e-8, except
  `disable_ko_after_ki` (leg 3.45, `ki_survive` 3.1e-2: the fix above) and the 2D
  surrogate residual (0.26).
- (b) Phoenix value policy: 60 rows. Coupon cash out of the leg (up to 6.49 on the
  observation-date memory row), probabilities ≤ 1.8e-6, 2D surrogate residual 0.36.
- (c) native KO-reset: 33 rows. Probabilities up to 1.4e-1 on the legacy `ki_probability`
  (its definition, above), post-schedule cash 5.7e-2, distribution payment times.
- (d) valuation-date rules: 35 rows. `dkak_carried` price 1.693, the lifecycle
  `ki_survive` (up to 0.96), the KO-reset immediate knock-out.
- (e) 2D refusal: 3 rows, now `StatisticsRefused`.
- (f) release rule: 9 rows. `phoenix_c105` +0.5728, `lu_fallback|phoenix` +6.1e-4 (one
  dual cell spans both barriers on that 4-node grid).
- (g) contractual periods: 63 rows. Dated Phoenix prices up to +1.019178, tenor Phoenix
  theta ≤ 1.0e-2, `phoenix_c105` ≤ 1.3e-13.

Goldens: the replay event frames (`scalar_bsm`, `localvol`) and the phase-4 execution
event distributions were regenerated once (`test/replay_golden/README.md`,
`test/execution/goldens/PROVENANCE.md`); the `book` (QUAD) frames are byte-identical.

### Performance
- The statistics block skips its all-zero columns (an observation the backward march has
  not reached) and solves them once as a +0.0 template. The tridiagonal solve calls LAPACK
  `gtsv` directly, without scipy's argument validation. The first-passage correction forms
  only its one row on a block. The value rows are reused, not recomputed. The KO-reset
  terminal resolves the pre-knock-in tenor once per surface, not per node. All are
  bit-identical on the byte book.

Timing gate (`test/pde_bitwise/timing.py`, 0.8.0 and 0.9.0 interleaved, ARM64 macOS with
the `accel` extra; median ms over 15 rounds). The price families come from the full run
(IQR ≤ 3 %). The statistics cases come from a rerun of the gated four alone
(`--families`, three repetitions per process, IQR ≤ 8 %): the full run was inconclusive,
with bimodal efficiency-core times on a loaded machine.

| Case | Warm 0.8.0 | Warm 0.9.0 | Ratio | Cold ratio | Gate |
|---|---|---|---|---|---|
| Snowball price | 42.9 | 34.9 | 0.81× | 0.87× | |
| Phoenix price (memory) | 652 | 455 | 0.70× | 0.71× | |
| KO-reset price | 25.5 | 17.3 | 0.68× | 0.69× | |
| Local-vol Snowball price | 117 | 107 | 0.91× | 0.93× | |
| European / American / barrier price | | | 1.00× | 1.00× | |
| Snowball statistics | 159 | 146 | 0.92× | 0.94× | pass |
| Local-vol Snowball statistics | 248 | 225 | 0.91× | 0.91× | pass |
| Phoenix statistics (no memory, INSTANT) | 230 | 247 | **1.075×** | 1.06× | **missed** (1.05× warm) |
| KO-reset statistics | 40 | 72 | **1.79×** | 1.78× | **not comparable** (0.8.0 was a 501-point QUAD) |
| Phoenix statistics (memory) | 877 | 1,279 | 1.46× | | recorded |
| Phoenix statistics (EXPIRY, memory) | 972 | 2,348 | 2.42× | | recorded |

- **Phoenix statistics miss the 1.05× gate by 2.5 points (+17 ms).** The coupon cash is a
  marched column (spec §5.3) where 0.8.0 multiplied the trigger probability. That is 39
  columns per surface instead of 26; the solve cost is linear in the column count, and
  the faster solve absorbs most but not all of it.
- **KO-reset statistics compare two algorithms.** 0.8.0 ran a 501-point QUAD next to the PDE
  price, with probability errors of up to 3.8e-3. 0.9.0 marches 15 columns per surface on
  the PDE, which cannot cost less than the PDE price it extends.
- Memory and EXPIRY Phoenix carry 183 and 365 columns per surface (12 observations, 12
  memory states); 0.8.0 reported no coupon cash for them. Pruning `streams` does not reduce
  the count.

## [0.8.0] - 2026-10-03

PDE 1D numerical policy (spec: `docs/superpowers/specs/2026-10-01-pde-1d-numerical-policy-design.md`).
Seven deliberate numerics changes, each its own commit checked against the 445-row PDE byte
book with its declared movers: the grid day is the environment's business days, the step
cap raises instead of scaling, the barrier family projects discrete observations, Snowball
`NODAL` means nodal, Phoenix takes θ = 1 on zero-diffusion steps, half-step damping where
monitoring is discrete, and forward far-field rows. No flip touches the damping of a
product with continuous knock-in: on a 252-day environment, inside 0.7.0's 5,000-step cap
and with the default projection, it keeps its 0.7.0 numbers bit for bit (one exception,
below); the grid day (a) and the cap (b) move it like any product. The flat Snowball, Phoenix and
KO-reset and the local-vol Snowball certificates are amended and re-banked (2026-10-02) with
their PDE candidates re-admitted; the intraday daily-KI certificate is re-banked (2026-10-03)
with QUAD V2 admitted and its PDE candidate rejected, as in 0.7.0.

### Added
- `CrankNicolson(damping="full_step" | "half_step")`: the damping form; `None` is the
  route's. Half-step splits each damped interval at its midpoint; both halves take
  `event_theta` below an event and 1.0 at the terminal node.
- `event_projection` is honoured by `BarrierPDESolver`, `DoubleBarrierPDESolver`,
  `OneTouchPDESolver` and `DoubleOneTouchPDESolver`.
- `MarketSnapshot.bus_days_in_year`, `GridRequest.day_count`, `GridRequest.refinement`,
  `TimeRefinement`, `time_fill`, `EventRecord`, `DiscreteEvent`, `apply_discrete_event`
  (`quantark.asset.equity.engine.pde.grid`).
- `DAMPING_DEFAULTS["core-continuous-ki"]`, `["2D"]`, `["local-vol"]` (`pde.grid.config`).
- `grid.mesh_kernels.backend()`: `"numba"` with the `accel` extra, else `"python"`.
- `pde_declared.environment_day_count` / `refuse_case_day_count` (model validation): the
  environment's business days recorded on a PDE candidate's `grid` record.

### Changed (defaults)
- **Core damping:** half-step, one interval at the terminal node and at each event,
  `(half_step, 1, 1)`, for every product without continuous knock-in; a product with
  continuous knock-in keeps full-step `(1, 2)`. The class is read from the contract, never
  from the spot or the knocked-in state.
- **Base-family discrete observations:** dual-cell projection (`event_projection=None` is
  `CELL_AVERAGE` on every core route). `"nodal"` keeps the 0.7.0 masks.
- **`max_steps`:** 12,000; a grid over it raises `ValidationError` naming the count it
  needs, `GridConfig(max_steps=...)` and `steps_per_day`. A grid is never scaled.
- **Snowball `NODAL`:** nodal on interior KO and discrete-KI events too.
- **Phoenix:** θ = 1 on zero-diffusion steps, as every value group.
- **Grid day:** `steps_per_day` counts the pricing environment's `bus_days_in_year`.
- **Far-field rows:** European and Barrier puts' lower row is `K·DF(t,T) − S_min·CF(t,T)`
  (floored at 0); the Barrier call's upper-row carry factor is the forward one
  `exp(−(q(T)T − q(t)t))` (it moves only on a term-dependent carry curve).
- `adapter.stepper(params)` becomes `adapter.stepper(params, product)`.
- `GridRequest(day_count=...)` and `MarketSnapshot(bus_days_in_year=...)` are required: code
  that constructs either directly gets a `TypeError` until it passes them.
- Event statistics (KO/KI probabilities) march the same layout, so they move with the half
  nodes at (f), as the prices do; their conventions are unchanged.
- The step cap holds a little under six years at `high` and twelve at `standard` on a 252-day
  environment; a six-year monthly Snowball at `high` now raises where 0.7.0 scaled it.
- The certificates' `grid` record: `max_steps` 12000, `day_count` from the environment,
  and `discrete_monitoring_damping: ["half_step", 1, 1]`. A full-step `(1, 2)`
  configuration with `max_steps=5000` on a 252-day environment still serializes to the
  0.7.0 record.
- Prepared PDE sessions: artefact builder `pde-prep/v3` (the request carries `refinement`
  and `day_count`); no 0.7.0 artefact is reused.

**Clean break (count meaning).** A caller who sets a count and no form, for example
`CrankNicolson(event_damping_steps=2)`, gets two halved intervals after flip (f) on every
solve that resolves against `"core"`, where 0.7.0 gave two whole steps; on a product with
continuous knock-in the count keeps its 0.7.0 meaning. Nothing can detect the intent, so no
error is possible. `CrankNicolson(damping="full_step", terminal_damping_steps=1,
event_damping_steps=2)` restores 0.7.0 everywhere; `damping="full_step"` alone takes the
route's counts, `(full_step, 1, 1)` on the core. A `CrankNicolson` **subclass** is a custom
stepper, used as given: one that sets counts and no `damping` priced in 0.7.0 and now raises
`ValidationError` naming the fields to set.

### Removed
| Removed | Replacement |
|---|---|
| `GridConfig(day_count=...)` (raises `ValidationError`) | `PricingEnvironment(bus_days_in_year=...)` |
| `TimeLayout.fill_scaled`, `TimeLayout.cap_exceeded`; the same keys in `explain` and the intraday numerical record; `LifeSurfaceSolution.fill_scaled` | none: a grid over `max_steps` raises |
| `groups.VALUE_NO_ZERO_DIFFUSION_OVERRIDE` | `groups.VALUE` |

`GridConfig` has eight fields. `TimeLayout` and `LifeSurfaceSolution` are results the
library constructs, so reading a removed field is a plain `AttributeError`.

### Fixed
- **Put far-field row.** The European and Barrier put's lower row was its `S → 0` limit
  `K·DF` at a grid edge four standard deviations below spot, ~39 too high there. Error
  against Black-Scholes at 400 points (pv / delta / gamma), study environment:

  | European put, strike 100 | 0.7.0 | 0.8.0 | the call, for scale |
  |---|---|---|---|
  | vol 22%, 1y, spot 100 | +2.5e-3 / −4.3e-4 / +9.1e-5 | +1.1e-4 / +4.1e-5 / +2.1e-6 | +2.1e-4 / +4.2e-5 / +2.2e-6 |
  | vol 22%, 1y, spot 80 | +1.9e-3 / −5.4e-4 / +1.1e-4 | −1.4e-5 / −6.6e-5 / +1.7e-6 | +2.6e-5 / −6.6e-5 / +1.7e-6 |
  | vol 22%, 3m, spot 100 | +4.2e-3 / −1.4e-3 / +5.9e-4 | +1.9e-4 / +8.4e-5 / +2.2e-5 | +2.5e-4 / +8.4e-5 / +2.2e-5 |
  | vol 22%, 3y, spot 100 | +1.1e-3 / −1.0e-4 / +1.5e-5 | −7.6e-5 / +2.3e-5 / +1.9e-7 | +2.0e-5 / +2.7e-5 / +2.2e-7 |
  | vol 12%, 1y, spot 100 | +3.9e-3 / −1.2e-3 / +4.6e-4 | +2.0e-4 / +6.2e-5 / +1.6e-5 | +2.8e-4 / +6.2e-5 / +1.6e-5 |
  | vol 40%, 1y, spot 100 | +1.0e-3 / −9.3e-5 / +1.3e-5 | −9.6e-5 / +2.4e-5 / +1.2e-7 | +1.4e-5 / +2.9e-5 / +1.6e-7 |

  The down-and-in put, dominated by that row, now meets its exact reference. The American
  put's lower row is unchanged (spec §9).
- **Snowball `event_projection="nodal"`** projected interior events anyway; it now masks
  them (spec §4.6 cases, NODAL pv 0.7.0 → 0.8.0: ordinary +3.5e-2, spot 102.5 +2.6e-2,
  monthly discrete KI +2.6e-2).
- **Phoenix** took θ = 0.5 on zero-diffusion (trading-clock night and weekend) steps where
  every other family took 1 (trading-clock Phoenix: +1.6e-7).
- **Continuous knock-in, alive boundary:** the knocked-in boundary rows are written before
  the alive rows that read them. Bitwise on every certified cell but one:
  `snowball-localvol-1d/calm_near_expiry` moves by 1.1e-13 (pv) to 3.5e-11 (delta) relative;
  the sparse-LU fallback Snowball (two interior nodes) moves +1.457.

### Performance
- **Mesh build.** The concentrated mesh's RK4 integration (the cold-solve cost, spec §4.9)
  runs on a Python-float kernel, or on a Numba kernel with the `accel` extra
  (`pip install "quantark[accel]"`), both bit-identical to the 0.7.0 NumPy reference on 549
  layouts. A value a kernel does not handle (non-finite, zero denominator, more than eight
  critical points) re-runs that call on the NumPy reference, so every exception and fallback
  is 0.7.0's. On 49 production layouts: 43.1 s (NumPy), 4.06 s (Python kernel), 0.30 s (Numba).
  CI requires the Python kernel on its four Python jobs and Numba in the `accel` job
  (`QUANTARK_REQUIRE_MESH_KERNEL`, a test switch). The
  Numba runtime adds ~30 MB to a process's peak memory.
- **Barrier step events** fold in place on the grid column (no per-step copy), and the
  terminal settlement timing is resolved once per march, not once per step (bitwise).

Timing gate (`test/pde_bitwise/timing.py`, 0.7.0 and 0.8.0 interleaved, two rounds of five
repetitions, ARM64 macOS with the `accel` extra; seconds, better round):

| Family (timing book, `standard`) | Cold 0.7.0 | Cold 0.8.0 | Speed-up | Warm 0.7.0 | Warm 0.8.0 |
|---|---|---|---|---|---|
| European | 0.774 | 0.022 | 35× | 0.0185 | 0.0150 |
| American | 0.765 | 0.026 | 29× | 0.0207 | 0.0207 |
| Snowball | 0.916 | 0.050 | 18× | 0.0434 | 0.0432 |
| Local-vol Snowball | 1.002 | 0.126 | 8.0× | 0.119 | 0.119 |
| Event statistics | 1.04 | 0.165 | 6.3× | 0.158 | 0.158 |
| Phoenix (memory) | 1.542 | 0.668 | 2.3× | 0.654 | 0.658 |
| Barrier (continuous) | 0.030 | 0.021 | 1.4× | 0.0282 | 0.0189 |
| KO-reset | 0.026 | 0.026 | 1.0× | 0.0254 | 0.0255 |

A cold solve is now dominated by its march: the Phoenix's mesh share fell from 0.89 s to
0.01 s, and the barrier and KO-reset cases build no concentrated mesh. Event statistics
from 15-repetition interleaved runs. Peak memory rises ~30 MB with Numba loaded.

### Per-flip deltas (PDE byte book, largest price moves)
- (a) grid day: 244-day environments only — Phoenix memory −2.7e-4, continuous-KI Snowball
  −1.6e-4, European −5.2e-8, intraday Snowball +6.4e-8. The intraday route's grid moved
  because its numerical environment has 244 days (F1).
- (b) cap: five-year standard Snowball −4.9e-7 (5,040 steps, was 4,980), three-year high
  −2.1e-6 (6,048, was 4,968).
- (c) projection: discrete barrier calls up to ±3.8e-2, double knock-out −2.8e-2, the
  down-and-in composite −8.0e-3, touches ~1e-3.
- (d) Snowball NODAL: +1.2e-2 to +1.8e-2 on the book's NODAL rows.
- (e) Phoenix θ: +1.6e-7 on the trading-clock Phoenix only.
- (f) half-step damping: local-vol Phoenix daily KI −3.6e-3, Snowball daily KI −3.3e-3,
  KO-reset discrete −1.6e-3, Snowball monthly −1.3e-4, discrete Snowball/Phoenix ≤ 9e-5,
  barrier/touch ≤ 3.6e-5, European ~1e-6. No continuous-KI row moved.
- (g) far field: puts (above); flat-curve barrier calls bitwise unchanged.

Goldens: replay frames (scalar_bsm, localvol) moved at (f) by ~1.6e-5 of notional;
execution goldens at (d) and (f) (`test/execution/goldens/PROVENANCE.md`).

## [0.7.0] - 2026-10-01

PDE 1D one core (spec: `docs/superpowers/specs/2026-09-30-pde-1d-one-core-design.md`;
architecture: `docs/pde-core.md`). Every shared-grid 1D PDE family (European, American,
the barrier family, Snowball, Phoenix, KO-reset, and the local-vol Snowball and Phoenix)
prices through one `prepare → march → read` path; the four per-family time-stepping loops
and the separate statistics sweep are gone. Numbers are unchanged: a 229-row `float.hex`
byte compare of the 1D book against 0.6.0 shows zero differences, and every banked
certificate keeps its identity. The scheme's settings move onto a stepper component, a
clean break with the old names raising `ValidationError` naming their replacement.

### Added
- `PDEParams.stepper`, `PDEParams.operator`, `PDEParams.readout`: component slots whose
  built-ins are `CrankNicolson()`, `CentralDifference()` and `StandardReadout()`
  (`quantark.asset.equity.engine.pde.core.components`). `CrankNicolson` carries the
  scheme's four settings: `theta`, `event_theta`, `terminal_damping_steps`,
  `event_damping_steps` (`None` = the route's defaults). A custom component runs on the
  core routes; the core validates its results and raises `NumericalError` naming it.
  `from_config` accepts `stepper: {theta: 1.0, ...}` and rejects `operator` / `readout`.
- `explain(product, pricing_env)` on `PDEEngine` and every core solver: the resolved plan
  (route, settings with their source and applicability, space, time, events, scheme,
  outputs, identities, per-operation terminal outcomes) from the same preparation `price`
  runs, without marching.
- `resolved_grid_evidence(params)` (`pde.grid.config`): the certificate `grid` record,
  the nine resolved grid fields plus the stepper's two damping counts under their 0.6.0
  keys. `PDEParams.evidence()` for `engine_config`.
- `OutputRequest`: what one call wants from a preparation (price, Greeks, event
  statistics, grid, life surface).
- `Calendar.identity()`: a value identity (name, sorted holidays, weekend days) for the
  settlement-calendar part of the schedule caches' keys.

### Removed
| Removed name | Replacement |
|---|---|
| `PDEParams(theta=...)` | `stepper=CrankNicolson(theta=...)`; 0.5 is Crank-Nicolson, 1.0 backward Euler |
| `PDEParams(event_theta=...)` | `stepper=CrankNicolson(event_theta=...)` |
| `GridConfig(terminal_damping_steps=...)` | `PDEParams(stepper=CrankNicolson(terminal_damping_steps=N))`; 0 disables |
| `GridConfig(event_damping_steps=...)` | `PDEParams(stepper=CrankNicolson(event_damping_steps=N))`; 0 disables |

`GridConfig` has nine fields and raises `ValidationError` (not `TypeError`) for the two
removed ones. The 0.6.0 entries that pointed at `GridConfig` damping counts
(`use_rannacher`, `rannacher_steps`, `rannacher_at_events`, `event_rannacher_steps`,
`DCNPDEEngine(rannacher_steps=)`) now point at the stepper. `TimeLayout` no longer carries
the damping index sets; `CrankNicolson.schedule` computes them.

### Changed
- The factorization caches key on the exact θ. Two θ agreeing to 6 (LU) or 12 (banded)
  decimals no longer share a matrix. `dt` stays rounded to 12 places, as every banked
  number was computed.
- Event statistics are a column group of the price march. `calculate_event_stats`
  no longer re-prices, and `price_with_events` marches once.
- Prepared sessions store the bound `Layout` (artefact builder `pde-prep/v2`). Bump
  contexts are fresh instances holding a frozen layout; the transient-attribute copy
  machinery is gone (the 2D Heston/SLV Greeks keep a legacy clone).
- Market memos are keyed by value fingerprints, so mutating a market object after a solve
  is no longer silently stale; an unfingerprintable input is sampled fresh.
- Certificate candidates are built from their declared params (`PDEParams.from_config`),
  so a declared setting is the one executed; a key the engine rejects fails the study at
  load.
- The DCN, local-vol vanilla/barrier and 2D Heston/SLV solvers read their damping counts
  and weights from `params.stepper`, and reject a custom component ("no core seam until
  sub-project 3"). `LocalVolPDESolver` checks route applicability on construction.

### Fixed
- Event statistics on a reused solver no longer reuse another market's factorizations
  (measured: a 0.041 KO-probability error on the second market).
- American Greeks on a fresh solver, for exercise with a settlement lag, no longer raise.
- PDE solvers deep-copy and pickle after pricing.
- A session no longer stores a factorization for a stepper without a value fingerprint
  (two such steppers could have shared one).
- Knock-in barriers (single and double) priced through a `PricingSession` now equal the
  direct price bit for bit. 0.6.0 marched the knock-out leg on the knock-in product's own
  spatial grid, masking the barrier instead of using the hard-edged domain (measured:
  up-and-in call −0.9%, double knock-in −2.2%). A session's layout is now refused, not
  reassembled, when the hard bounds differ. The session path was not in the byte book;
  it now has an execution test.
- A session with `cache_enabled=False` no longer warms a factorization pack.

Unchanged and still raising: bump-context Greeks on a knock-in barrier that has not
knocked in (`PricingError`: the frozen layout has the knock-in's bounds, the knock-out
leg needs the barrier's). It is a declared failure in the byte book; freezing the
knock-out leg's layout is sub-project 3 work.

### Sub-project 3
Still planned: the environment's business day as the grid day (full re-certification);
half-step damping as a `CrankNicolson` option; coefficient-break nodes; a budget default
for path-independent routes; failing instead of scaling at the caps; DCN and the
local-vol vanilla/barrier space axis onto the core; native KO-reset statistics. Added by
this release, each its own re-certified commit: statistics Level 2 (the statistics group
on the value policy with a marched maturity leg, memory-state Phoenix coupons); the
zero-diffusion θ override for Phoenix; Snowball `NODAL` on interior schedule events (or
withdrawing `NODAL` from the shared-grid table); native 2D Heston/SLV statistics.

## [0.6.0] - 2026-09-29

PDE 1D surface cleanup (spec: `docs/superpowers/specs/2026-09-28-pde-1d-surface-cleanup-design.md`).
Every 1D PDE route — the shared-grid solvers behind `PDEEngine`, the local-vol vanilla
and barrier solvers, and DCN — is configured by one `PDEParams` and nothing else, and
nothing is silently ignored: a field a route cannot honour raises. This is a clean break:
the removed names raise `ValidationError` naming their replacement. Numbers are unchanged
on every route except the local-vol discretely monitored barrier (see Changed), proven by
a 112-row `float.hex` byte compare of the whole 1D book against 0.5.0.

### Added
- `GridConfig.time_steps`: a total time-step budget over the horizon, spread over the
  mandatory intervals by length (at least one step each), and the alternative to
  `steps_per_day`. Exactly one time basis is in force after resolution; an overlay that
  sets one basis clears the other, and setting both raises.
- `GridConfig.evidence()` (the certificate serialization: unset optionals omitted, so no
  banked identity moves), `GridConfig.with_steps_per_day()` and
  `GridConfig.with_time_steps()`.
- `PDEParams.resolved_grid()`: the requested grid (the `accuracy` preset under the `grid`
  overlay), before the `max_steps` cap that binds once a maturity is known.
- The route applicability table (`quantark.asset.equity.engine.pde.route_applicability`,
  `ROUTE_RULES`): per route, the fields it cannot honour. An inapplicable field accepts its
  preset default and rejects an explicit override; a rejected field rejects any explicit
  value. The error names the field, the route, and sub-project 3.
- `PDEParams.from_config(config_or_path, **overrides)` over field names, with `grid` a
  nested `GridConfig` mapping (a `bounds` list becomes a tuple).

### Removed
`PDEParams`, `PDEEngine` and the DCN constructors take keyword arguments only; each
removed name raises `ValidationError` with its replacement:

| Removed | Replacement |
|---|---|
| `PDEParams(grid_size=...)` | grid=GridConfig(points=...) on every route. |
| `PDEParams(time_steps=...)` | grid=GridConfig(time_steps=N): the same meaning, a total step budget over the horizon. steps_per_day= is the per-day alternative. |
| `PDEParams(s_max=...)` | grid=GridConfig(bounds=(None, hi)) on the shared-grid solvers. On the two local-vol solvers an explicit bound is rejected until sub-project 3: the kernel sets its own domain and aligns the top, so there is no hard edge to honour. |
| `PDEParams(s_min=...)` | removed; it was read by nothing. |
| `PDEParams(use_rannacher=...)` | grid=GridConfig(terminal_damping_steps=N); 0 disables. use_rannacher=False also switched off event damping on the 2D Heston/SLV solvers; add event_damping_steps=0 to keep that |
| `PDEParams(rannacher_steps=...)` | grid=GridConfig(terminal_damping_steps=N); 0 disables |
| `PDEParams(rannacher_at_events=...)` | grid=GridConfig(event_damping_steps=N); 0 disables |
| `PDEParams(event_rannacher_steps=...)` | grid=GridConfig(event_damping_steps=N); 0 disables |
| `PDEParams(boundary_mode=...)` | removed; asymptotic boundaries are the only treatment |
| `PDEParams(continuous_ki_correction=...)` | removed; the first-passage correction always applies to continuously monitored KI |
| `PDEParams(ki_monitoring_mode=...)` | removed from the PDE; each barrier is monitored as the product declares it (discrete dates, continuous, or maturity-only). To price a dense discrete KI as continuous, declare it continuous on the product. QuadParams.ki_monitoring_mode is unchanged. |
| `PDEParams(cache_strategy=...)` | cache_enabled=True/False is the only cache control |
| `PDEParams(grid_cache_max_entries=...)` | cache_enabled=True/False is the only cache control |
| `PDEParams(banded_cache_max_entries=...)` | cache_enabled=True/False is the only cache control |
| `PDEParams(use_banded_solver=...)` | removed; the linear solver is chosen by the solver family |

| Removed | Replacement |
|---|---|
| `PDEEngine(method=...)` | Crank-Nicolson is the only built-in scheme; PDEEngine(params=PDEParams(...)). theta=1.0 gives backward Euler. |

| Removed | Replacement |
|---|---|
| `DCNPDEEngine / LocalVolDCNPDEEngine(num_space_nodes=...)` | DCNPDEEngine(params=PDEParams(grid=GridConfig(points=...))) |
| `DCNPDEEngine / LocalVolDCNPDEEngine(rannacher_steps=...)` | grid=GridConfig(terminal_damping_steps=N, event_damping_steps=N) (DCN applies both in its half-step form) |
| `DCNPDEEngine / LocalVolDCNPDEEngine(s_min_mult=...)` | removed; never read |
| `DCNPDEEngine / LocalVolDCNPDEEngine(s_max_mult=...)` | removed; never read |
| `DCNPDEEngine / LocalVolDCNPDEEngine(concentration=...)` | removed; never read |

Also removed: `BackwardOperator` (`backward_operator.py`), `ContinuousKICorrection`,
`make_pde_params`, `PDE_PARAM_PRESETS`, the `"pde"` branch of `make_engine_params` /
`list_param_profiles` (PDE has no profile tier: use `PDEParams(accuracy=...)`), the
class-level cache setters (`set_cache_enabled`, `set_cache_strategy`,
`set_grid_cache_max_entries`, `clear_grid_cache`), and `reject_legacy_resolution_knobs`.

### Changed
- **Local-vol vanilla and barrier solvers are configured by `PDEParams`.** Previously
  `accuracy` and `grid` were ignored there; they are now honoured or rejected per the
  route table. Their own preset table keeps today's plan at `standard` (400 points, a
  200-step budget); `fast`/`high` are new resolution choices. `theta` reaches both
  kernels; the vanilla's terminal damping is 0 or 1 (an explicit count of 2 or more is
  rejected). Bitwise at the defaults and for continuous and maturity-only monitoring.
- **The local-vol discretely monitored barrier observes on exact event nodes** (the
  declared number change). Each interior observation date is a node of the time layout,
  the step budget spread over the intervals between them, instead of the nearest node of
  a uniform grid. Byte gate: the two `lv_barrier_discrete` rows move by -7.9e-4 and
  -1.0e-3. Against a 1600 x 2400 reference PDE (every date on a node), flat 20% vol:

  | product | grid | base PDE | branch PDE | abs(base-ref) | abs(branch-ref) |
  |---|---|---|---|---|---|
  | UO call B130 monthly | 400x200 | 3.92225 | 3.92154 | 2.47e-03 | 1.75e-03 |
  | UO call B130 monthly | 600x240 | 3.92049 | 3.92049 | 7.03e-04 | 7.03e-04 |
  | UO call B130 monthly | 200x200 | 3.92822 | 3.92752 | 8.44e-03 | 7.74e-03 |
  | UO call B130 monthly | 800x400 | 3.91984 | 3.92013 | 5.49e-05 | 3.44e-04 |
  | UO call B130 monthly | 400x150 | 3.92460 | 3.92149 | 4.81e-03 | 1.70e-03 |
  | UO call B120 monthly | 400x200 | 1.76249 | 1.76185 | 3.05e-03 | 2.41e-03 |
  | UO call B120 monthly | 600x240 | 1.76040 | 1.76040 | 9.57e-04 | 9.57e-04 |
  | UO call B120 monthly | 200x200 | 1.77063 | 1.77001 | 1.12e-02 | 1.06e-02 |
  | UO call B120 monthly | 800x400 | 1.75965 | 1.75991 | 2.13e-04 | 4.74e-04 |
  | UO call B120 monthly | 400x150 | 1.76428 | 1.76179 | 4.84e-03 | 2.35e-03 |
  | UO call B120 quarterly | 400x200 | 2.16190 | 2.16190 | 1.04e-03 | 1.04e-03 |
  | UO call B120 quarterly | 600x240 | 2.16128 | 2.16128 | 4.18e-04 | 4.18e-04 |
  | UO call B120 quarterly | 200x200 | 2.16532 | 2.16532 | 4.46e-03 | 4.46e-03 |
  | UO call B120 quarterly | 800x400 | 2.16107 | 2.16107 | 2.06e-04 | 2.06e-04 |
  | UO call B120 quarterly | 400x150 | 2.16442 | 2.16189 | 3.55e-03 | 1.03e-03 |
  | DO put B85 monthly | 400x200 | 1.20049 | 1.20002 | 4.22e-03 | 3.75e-03 |
  | DO put B85 monthly | 600x240 | 1.19755 | 1.19755 | 1.27e-03 | 1.27e-03 |
  | DO put B85 monthly | 200x200 | 1.21140 | 1.21096 | 1.51e-02 | 1.47e-02 |
  | DO put B85 monthly | 800x400 | 1.19682 | 1.19701 | 5.46e-04 | 7.37e-04 |
  | DO put B85 monthly | 400x150 | 1.20189 | 1.19997 | 5.61e-03 | 3.69e-03 |
  | UI call B130 monthly | 400x200 | 4.90751 | 4.90822 | 1.83e-04 | 5.31e-04 |
  | UI call B130 monthly | 600x240 | 4.90791 | 4.90791 | 2.23e-04 | 2.23e-04 |
  | UI call B130 monthly | 200x200 | 4.90897 | 4.90967 | 1.28e-03 | 1.98e-03 |
  | UI call B130 monthly | 800x400 | 4.90809 | 4.90780 | 4.00e-04 | 1.11e-04 |
  | UI call B130 monthly | 400x150 | 4.90516 | 4.90827 | 2.53e-03 | 5.81e-04 |

  Closer to the reference in 16 of 25 cells; where the old grid misplaced a date by up to
  half a step (400x150) the error falls 2-3x, and elsewhere both placements sit inside
  the grid's own error band (every cell moves by <= 7e-4).
- **DCN is configured by `PDEParams`** with its own `accuracy` presets, which scale
  spatial resolution only (401 / 801 / 1601 points; `fast` and `high` have no banked
  evidence). The observation schedule is the time resolution: an explicit
  `steps_per_day` or `time_steps` is rejected, as is an explicit CELL_AVERAGE. The damping
  counts drive DCN's half-step form; `theta` and `event_theta` reach the loop. Bitwise at
  the defaults.
- **`ki_monitoring_mode` is removed from the PDE.** Each barrier is monitored as the
  product declares it. The two honest alternatives: `QuadParams.ki_monitoring_mode`
  (BGK stays in QUAD), or declare the KI continuous on the product.
- **`event_projection` defaults to `None`**, the route's own treatment (CELL_AVERAGE on
  the shared-grid autocallable solvers, as before).
- **`PDEEngine(params)` is the whole constructor**; Crank-Nicolson is the only built-in
  scheme. `theta=1.0` is backward Euler: first order in time and more diffusive, not
  "monotone" (monotonicity also needs the spatial operator's sign condition).
- The first-passage continuous-KI correction and the asymptotic boundaries are
  unconditional (their opt-outs are removed).
- **2D Heston/SLV damping:** event damping no longer follows the terminal switch
  (0.5's `use_rannacher=False` turned both off; set `event_damping_steps=0` too to keep
  that), and the 2D solvers read `terminal_damping_steps` as on/off (`> 0`).
- A budget-basis `GridConfig` resolves `day_count` to `None` (the day basis is not in
  force), so a resolved grid (`PDEParams.resolved_grid()`) can be fed back as an overlay.
- The grid "day" remains 1/252 of a solver year (`GridConfig.day_count`); it becomes the
  environment's business day in sub-project 3.

## [0.5.0] - 2026-09-28

Intraday pricing: `quantark.intraday` values a contract at a timezone-aware timestamp
instead of a date (spec: `docs/superpowers/specs/2026-09-15-intraday-pricing-design.md`),
and its certification lives in `quantark.modelvalidation`, not in the runtime
(spec: `docs/superpowers/specs/2026-09-18-intraday-modelvalidation-certification-design.md`,
reference revision: `docs/superpowers/specs/2026-09-18-intraday-deterministic-reference-revision.md`).
A certificate constrains the release, never the code. The autocallable engines agree on
the maturity close, on an observation at the valuation instant, on the knock-out-reset
snowball's contract rules and on a knock-out and knock-in at one instant (see Fixed).

The release also carries what `main` gained since 0.4.7: the simulated-path backtest,
futures bucket hedging in the replay engine, the opt-in QUAD V2 engines, PnL explain,
vol-model calibration and the Greeks facade (see "Also in this release").

### Added
- `quantark.intraday`: `value_intraday(request)` and `PricingSession.value_intraday`
  return an `IntradayValuationResult` with PV components, cashflows, a status per
  requested output and the provenance of every assumption. Three clocks: seconds-exact
  calendar time for carry and discounting, a versioned `VarianceProfile` (`desk`,
  `uniform`, `sessions_only`) mapped by `IntradayTimeMap` into `TradingClockVolSurface`
  for variance, and the contract's own day count for accruals. `TradingSessionCalendar`
  declares sessions, breaks, early closes and the payment time; a DST gap or fold is an
  error, never a guess.
- Fixings: confirmed `Fixing`s replay through the lifecycle trackers; a fixing that is due
  but not published is replaced by the latest spot and recorded as an `AssumedFixing`,
  and the flows it can change carry that dependency. `ObservationRecord.observation_timestamp`
  gives a timed observation (intraday only).
- Routes: QUAD V2 (snowball, Phoenix, knock-out-reset snowball), PDE (with an
  `under_resolved` diagnostic when the diffusion layer is not resolved), Monte Carlo,
  analytical digital (exact zero-variance limit) and analytical barrier/one-touch (gated
  by an exact admissibility test), behind an explicit capability matrix with fail-closed
  lookup (`docs/execution/intraday-capability-matrix.md`).
- Greeks: desk-bump Greeks on the frozen float-time twin, point Greeks per route
  (`undefined` at a payoff kink), local theta with a declared step and unit,
  `roll_context` and `roll_through_events`.
- Batch: `value_intraday_many`, prepared `spot_curve`s and provenance-preserving
  `aggregate_intraday`.
- Lifecycle: trackers declare coupon arrears and a settled terminal claim.
- `quantark.modelvalidation` schema 2: a quantity catalogue with per-quantity budgets,
  a `context` block with per-case overrides, semantic `expect:` cells, per-case random
  substreams, convergence axes with at least three levels, three-way verdicts on cells
  and aggregates, and a `contract` block an amendment may not change. Schema-1 studies,
  evidence and wire format are unchanged.
- A deterministic reference kind (`reference_kind = "deterministic"`): a typed record
  with a radius per quantity that declares itself `analytical` or a `calibrated_estimate`,
  consumed whole by the gate (`bounds.radius_budget_fraction`, no interval multiplier) and
  added linearly in the aggregate. `reference_qualification` names a stochastic arm that
  must agree with it case by case; an unqualified case is not decision-eligible, and the
  arm's sampling is frozen in the contract.
- Anchors of quantities that difference prices: a candidate may declare
  `anchor_noise_weights(case)`, the L1 weight of each quantity's price stencil, and off the
  banking machine such an anchor is compared at the prices' tolerance carried through that
  stencil (`modelvalidation.anchors.anchor_tolerance`). The intraday snowball candidates
  declare the runtime's desk bump and theta step. A one-second desk theta had moved 2.2e-8
  relative on x86_64 while the prices it differences agreed within 1e-9; the new tolerance is
  at most 4e-8 of a spot Greek's column scale and 2e-6 of desk theta's. Same-machine
  comparison stays exact.
- `python -m quantark.modelvalidation bank <run dir>`: validates, refuses quick runs,
  extracts anchors and never overwrites a banked directory.
- Study `snowball_intraday_daily_ki_bsm.yaml`: the daily-KI snowball on the intraday
  clock, 23 cases, QUAD V2 and PDE intraday routes against the engine-independent
  Gaussian-transition reference, qualified by paired RQMC. Banked 2026-09-19
  (`docs/modelvalidation/certificates/snowball-intraday-daily-ki-bsm/2026-09-19`, digest
  `a0569e97687cc01c...`): `SnowballQuadEngineV2` ADMITTED, `SnowballPDESolver` REJECTED.

### Removed (breaking)
The import shims deprecated in 0.4.x are removed, as announced:
- `quantark.backtest.otc` and its twelve modules. Import from `quantark.backtest.replay`
  (`config`, `single`, `market`, `engine_factory`, `strategy_state`, `results`,
  `product_replay`, `dashboard`), `quantark.param.vol.surface_history` (was
  `otc.vol_history`) and `quantark.volcalibration.calibrate` (was `otc.vol_calibrators`).
- `quantark.volmodels.calibration`: use `quantark.volcalibration.calibrate` (the same module;
  a monkeypatch on its kernels reaches the calibrator as before).
- `VolModelCalibrationConfig` and `HESTON_PRESETS` are no longer re-exported from
  `quantark.backtest.replay` or `quantark.backtest.replay.config`: import them from
  `quantark.volcalibration.config`.

### Changed
- PDE delta and gamma are read off a local cubic through the four log-spot nodes around
  the spot (`BasePDESolver._calculate_delta_gamma`; a three-node grid keeps the quadratic).
  The nearest-node quadratic carried that node's constant curvature to the spot: an O(dx)
  gamma error whose sign flipped as refinement moved the nearest node across the spot.
  Every PDE engine's delta and gamma move; PV does not. The replay goldens moved by at most
  1.2e-4 of scale in delta and 3.6% in gamma. The four studies that anchor PDE Greeks were
  re-certified and banked 2026-09-28 with every decision unchanged (ADMITTED):
  `snowball-flat-bsm` (`ccba1c32d4055333...`), `phoenix-flat-bsm` (`8cd5584888009924...`),
  `ko-reset-flat-bsm` (`722b7726fadc4d41...`) and `snowball-localvol-1d`
  (`c0d62f9f4c0986a9...`); their predecessors are retired with `superseded_by`. On those
  cells the readout is accuracy-neutral (worst cells move by at most 0.004 of a
  0.5-contract bound).
- `quantark.intraday` returns what it computes. Every requested output carries its own
  status (`ok`, `undefined`, `failed`, `not_requested`) with diagnostics; the `unqualified`
  status, the `certify=` flags, `GreekValue.error_budget` and the packaged Gate C evidence
  (`quantark/intraday/evidence`) are removed. The PDE resolution verdict is `under_resolved`.
  The runtime reads no certificate and does not import `quantark.modelvalidation`.
- The Gate C evidence is archived under `docs/modelvalidation/legacy/intraday-gate-c/2026-09-18/`
  with an inventory of what each fixture's successor study is.
- A knock-out and a knock-in observed at the same instant: the knock-out wins, on every
  autocallable and every engine (desk ruling, 2026-09-21). The knock-in observed with it neither
  disables that knock-out under `disable_ko_after_ki` nor, on a knock-out-reset snowball,
  replaces the first schedule. The rule is for a discretely observed knock-in; a continuously
  monitored level breached at an observation was touched before it and keeps its precedence.
  One spot breaches both only when the knock-out level is at or below the knock-in level, so the
  usual contract is unchanged: 240 prices (three products, Monte Carlo, PDE, QUAD V1 in both
  event projections and QUAD V2, discrete and continuous knock-in, flag on and off) are
  byte-identical before and after. On a tie contract the engines used to disagree (spot 70,
  knock-out at 60 under a knock-in at 75, monthly observations): the snowball under
  `disable_ko_after_ki` was -22.4 on Monte Carlo, PDE and QUAD V1 against 14.0 on QUAD V2; the
  Phoenix 82.1 on Monte Carlo and both QUAD engines against 100.2 on PDE; the knock-out-reset
  snowball -21.6 against 14.0 on QUAD V2, with or without the flag. All four now agree with a
  semi-analytic reference (the snowball under the flag with two observations: 3.0803, where the
  knock-in winning gives -18.1045).
  Changed: `SnowballMCEngine` (both products) and `PhoenixMCEngine` (a knock-out at the knock-in
  instant is valid), `SnowballPDESolver` and `KOResetSnowballPDESolver` (the knock-out jump on
  the not-knocked-in surface goes after the knock-in jump where the knocked-in surface does not
  carry it, at interior observations and in the terminal condition), `SnowballQuadEngine`,
  `KOResetSnowballQuadEngine` and `PhoenixQuadEngine` (the same, in both event projections),
  and QUAD V2's Phoenix, which let the knock-in win under the flag on purpose. `PhoenixPDESolver`
  already let the knock-out win. The event-statistics sweeps of the PDE and QUAD V1 engines are
  not reordered, and the two-dimensional vol-model solvers were not examined.

### Fixed
- The maturity close under `BEFORE`, where every remaining event is decided on the known
  spot. The QUAD V2 intraday route raised there; the Monte Carlo route priced a claim that
  knocks in or out at that instant as if it had survived (12 where the lifecycle pays -30 at
  spot 70), and the PDE route paid a knock-out decided there as a survival (the rebate, not
  the knock-out coupon; its desk theta one hour earlier was wrong by 7 per hour). All three
  routes now resolve the decided claim through the runtime's own event resolution.
- The same defect at its source: the zero-maturity shortcut of `SnowballMCEngine`,
  `SnowballPDESolver` and `SnowballQuadEngine` returned the terminal payoff of the carried
  knock-in state and never applied the observations still pending at that instant, although
  each engine decides such observations correctly whenever more of the claim lies ahead.
  They now share `SnowballOption.decide_observations_at_valuation`. Day-level dated products
  cannot reach this state (valuation on the exercise date is refused); a time-based product
  at zero maturity can.
- The same defect in the other two autocallables. At zero maturity `PhoenixMCEngine`,
  `PhoenixPDESolver` and `PhoenixQuadEngine` returned the redemption alone: the last coupon,
  the memory arrears it releases, a knock-out and a knock-in at the last observation were
  dropped (Monte Carlo and QUAD V1 also dropped a knock-in carried from an earlier day). The
  knock-out-reset snowball's shortcut in `SnowballMCEngine`, `KOResetSnowballPDESolver` and
  `KOResetSnowballQuadEngine` ignored the pending knock-out of the schedule in force (pre-KI
  before a knock-in, post-KI after one), and QUAD V1 the carried knock-in. They now call
  `PhoenixOption.decide_observations_at_valuation` and
  `KnockOutResetSnowballOption.decide_observations_at_valuation`. The Monte Carlo and PDE
  intraday routes resolve the maturity close for all three autocallables; the knock-out-reset
  snowball, which the runtime cannot replay, is decided from its float-time twin, and its QUAD
  V2 route no longer raises there.
- The knock-out-reset snowball's two contract rules, on every engine: a knock-in REPLACES the
  first knock-out schedule by the second and stays to the end, and the second schedule only
  becomes effective through a knock-in DURING the first, so a contract not knocked in when the
  first schedule ends has matured there (`KnockOutResetSnowballOption.first_schedule_ended`,
  `require_alive`). `SnowballMCEngine` kept testing discrete knock-in records after the first
  schedule had ended (3.41 against 5.94 from PDE and QUAD on a contract with monthly knock-in
  records for a year and a six-month first schedule), read a spot-dependent rebate off the end
  of the time grid instead of the end of the first schedule, and ignored a knock-in carried
  into the valuation. `KOResetSnowballQuadEngine` always priced the not-knocked-in state and
  refused a contract whose first schedule had ended; it now reads the knocked-in surface, on
  a lattice pinned to the second schedule's level. Every engine, and the intraday twin, priced
  or refused the wrong way round past the first schedule: a knocked-in contract is now valued
  there on all of them, and one not knocked in is refused as matured instead of being priced
  with the first schedule extended to the final maturity.
- `SnowballMCEngine`, knock-out-reset contracts: the second schedule is in force from the
  observation that knocks the contract in, that observation included, as in the PDE and QUAD
  engines (and in this engine's own REBASED branch). Monte Carlo counted it strictly after the
  knock-in, which only shows when the second schedule's level is at or below the knock-in
  level: 4.60 against 4.75 there, now 4.74 +/- 0.01. The usual contract is unchanged.
- `PhoenixQuadEngine` never read `CouponBarrierConfig.initial_coupon_arrears`: memory coupons
  owed at the pricing date were worth nothing. They are released by the first period that pays,
  as the Monte Carlo and PDE engines release them.
- `AutocallableLifecycleTracker.settle_maturity_if_due` raised `TypeError` for every Phoenix
  alive at maturity: it called `get_payoff` with the snowball's signature. This reached the
  portfolio lifecycle manager and the backtest replay, not only intraday valuation.
- `PhoenixPDESolver`: a knock-out decided at the valuation instant (any instant, not only
  the last) left out the memory coupons in arrears, which Monte Carlo, QUAD V2 and the
  lifecycle tracker release with that period's coupon (101 against 106 with five owed).
- An observation at the valuation instant with more of the claim ahead. `PhoenixQuadEngine` and
  `KOResetSnowballQuadEngine` skipped it (their time grid drops time zero): no knock-out, no
  coupon, no missed-coupon memory and no knock-in from it, so a Phoenix observed now at 104 was
  94.54 against 101 on PDE and QUAD V2, and one at 90 exactly one coupon short.
  `SnowballMCEngine` refused a knock-out-reset contract observed now (`dt_array must have
  strictly positive entries`, in pricing and in the event statistics): its grid kept time zero as
  a simulation node where the plain snowball's grid reads such a record off the known spot. All
  three now decide the observation on the known spot, and each engine satisfies the identities
  that need no second engine: a coupon observed now raises the price by exactly the coupon, a
  memory coupon missed now equals the contract carrying it as arrears, and a knock-in observed
  now equals the contract carrying the knock-in (to 1e-8, on all four engines).
- A Phoenix coupon pays principal x rate x the period's own fraction. The daily lifecycle
  tracker passed no fraction to `get_coupon_payoff` and got a silent 1.0, so a monthly
  product paid about 12x more through the tracker than through any engine. The quotation
  is now declared, `AccrualConfig.is_annualized_coupon` (default annualized, as every engine
  already priced: engine PVs are unchanged), and `get_coupon_payoff` resolves the contract's
  own fraction or fails closed. The tracker also kept missed memory coupons only as a count
  and returned before the coupon loop on a knock-out; both now match the engines.

### Also in this release (merged to `main` since 0.4.7)
- `quantark.backtest.simulation`: simulated-path ensemble backtest for autocallable books.
  `MarketPath` batches from a stationary block bootstrap of the joint spot/vol/carry
  history, GBM or designed stress paths; the CFFEX IM chain read off each path's carry
  curve; a vectorised lifecycle and futures ledger; exact, spot-ladder and PDE
  life-surface repricing providers with in-memory and on-disk state caches; a sampling
  accuracy gate; spawn-pool path batching; a conformance oracle against the replay
  engine; distributions with expected shortfall, paired comparisons on matched paths and
  run persistence. Worked study: `example/case_studies/snowball_simulated_paths`.
- Replay engine, futures carry buckets: immutable carry scenarios, signed bucket
  aggregation, joint delta and parallel carry hedge strategies, a multi-leg futures ledger
  with atomic execution, carry audits by independent repricing, stress of unquoted carry,
  and an opt-in term-structure dividend source.
- QUAD: opt-in QUAD V2 Gaussian autocallable engines (`quantark.asset.equity.engine.quad.v2`:
  `SnowballQuadEngineV2`, `PhoenixQuadEngineV2`, `KOResetSnowballQuadEngineV2`,
  `AutocallableQuadEngineV2`). `QuadParams.readout` selects `"legacy_linear"` (default) or
  `"transition"`, which evaluates the final transition at the spot instead of interpolating
  between nodes (no delta staircase); `QuadParams.align_cell_stretch` (opt-in, `None` by
  default) puts every barrier on a node at once. Defaults leave every price unchanged.
- `quantark.pnlexplain`: waterfall and Taylor attribution of PV changes, lifecycle
  transitions, bucketed factor coordinates, trading-clock theta, and portfolio, position
  and quoted-leg explain with a replay recorder.
- `quantark.volcalibration`: market quotes to IV surface to LV/Heston/SLV calibration,
  with quote normalization, admission checks, a run store and a CLI
  (`python -m quantark.volcalibration`).
- Risk measures: the `GreeksCalculator` facade over a `GreekDef` registry, higher-order
  Greeks (speed, zomma, dividend volga, analytical closed forms) and a dual-clock theta
  suite (charm, color, vega/gamma theta, 1D/1TD, exact mode).
- Changed: scalar rho parallel-shifts the rate curve instead of replacing it with
  `FlatRateCurve(r(T) + bump)`, which measured a curve reshaping on any term structure
  (98x the parallel-shift rho on a 1%/3%/5% curve); flat curves are bitwise unchanged. On
  a trading clock the vol bump unit is one point of the trading-quoted sigma.


## [0.4.7] - 2026-08-25

Settlement-date-aware payoff discounting across the equity option stack
(spec: `docs/superpowers/specs/2026-07-29-equity-option-settlement-date-support-design.md`,
plan: `docs/plans/2026-07-29-equity-option-settlement-date-support.md`).
Determination and payment are now two separate clocks: underlying dynamics,
monitoring and optimal exercise stop at determination; every cashflow
discounts at its own payment time with curve-exact discount factors.

Alongside it, autocallable batch pricing gets its performance pass — the PDE
solve loop no longer re-serializes the product on every time step, and the
QUAD event-stats recursion gains a fused KI walk plus an opt-in forward
transition-density mode (`docs/autocall-engine-perf/`).

### Added
- `quantark.asset.equity.settlement`: shared settlement timing kernel —
  `SettlementConvention` (business-day / calendar-day / year-fraction lag
  with business-day adjustment and calendar), `SettlementRequest`,
  `ResolvedPaymentTiming` (determination/payment times + curve-exact
  `determination_df`/`payment_df`/`delay_df`), `SettlementResolver`
  (`resolve_contingent` / `resolve_pending`). No fabricated dates: a
  date-based convention on a time-only determination raises.
- Product contract: `BaseEquityOption.settlement_convention`; the existing
  terminal `settlement_date` is now honored by engines as the
  terminal-payment override; per-record `ObservationRecord.settlement_date` /
  `settlement_time` drive event-cashflow payment (KO redemptions, coupons,
  hit-paid rebates) across analytical, MC, PDE and QUAD engines for the
  BSM / Local Vol / Heston / SLV families.
- American exercise obstacle reflects delayed cash receipt in PDE and LSMC
  (node-specific `delay_df`); analytical American approximations raise a
  capability error for settlement forms they cannot represent.
- Engines declare settlement/timing capabilities and raise precise
  capability errors before numerical work rather than silently pricing at
  determination.
- Lifecycle: pending-cashflow ledger
  (`quantark.asset.equity.lifecycle.cashflows`) carries
  determined-but-unpaid receivables at `amount * DF(valuation, payment)`
  through payment; `price(..., lifecycle_state=...)` propagates through
  engines, positions, portfolio valuation, Greeks and the consolidated
  replay backtest (settlement decomposition; KO termination waits for
  terminal cash to land).
- Payment-aware event statistics: determination/payment time arrays with
  undiscounted and discounted expected cashflows that reconcile to engine
  PV.
- QUAD autocallable engines: opt-in `QuadParams.event_stats_mode="forward_density"`
  computes the event distribution from a forward transition-density march
  (2–3 surfaces instead of one indicator row per KO observation). `npv` is
  unchanged (always the backward value solve); distribution fields differ
  within the banked validation tolerances
  (`docs/autocall-engine-perf/FORWARD-DENSITY-EVIDENCE-2026-08.md`).
  KO-reset ignores the flag. Default remains `"stacked"`.

### Changed
- Snowball/Phoenix PDE: the product cache token is now memoized per solve
  (`SnowballPDESolver._product_token_memo`, cleared at every solve entry).
  The boundary-condition path previously re-serialized the whole product —
  every KO/KI schedule record — once per time step, >90% of a snowball
  solve; bit-identical output, ~7× faster per solve
  (`docs/autocall-engine-perf/FINDINGS-2026-08-24.md`).
- QUAD stacked event stats: the KI-probability recursion now rides the main
  stacked recursion as fused surfaces (bit-identical output, one fewer full
  time loop per `price_with_events`); identically-zero indicator rows are
  skipped in the diffusion/bridge steps and the row-independent bridge
  kernel stack is cached per engine (64-entry bound). Bit-identical output.
- QUAD `price_with_events` now honors `streams` pruning [§11.1], mirroring
  the PDE contract: KI-probability recursion and Phoenix coupon rows are
  skipped when no requested stream reads them. Callers that pass `streams`
  (the execution framework's greek-bump runner does) get `0`/empty for
  pruned distribution fields where QUAD previously returned computed values;
  `ko_probability`/`survival`/`pv` and all requested fields are unchanged.
  `streams=None` keeps the old full computation.
- **The flat dividend/carry sanity bound moves from 20% to 100%**
  (`ContinuousDividendYield`, the BSM process gate, and the market-data
  point model), matching `TermStructureDividendYield`'s own magnitude
  bound. OTC borrow/lending carry (融券率) legitimately exceeds 20% on
  some trades; the 100% ceiling still rejects unit errors (entering
  `26.89` for `0.2689`). Values between 20% and 100% now price instead of
  raising `ValidationError`.

### Fixed
- Repository tooling only — not shipped in the wheel, which contains
  `quantark/` alone. The `mo_volmodels` dashboard decided whether a results
  aggregate was stale by comparing two ISO-8601 stamps as strings, which
  orders correctly only when both carry the same UTC offset. Read from a
  zone other than the one that wrote the cells, an aggregate *newer* than
  the cells it summarises was reported stale. It now compares instants.
- Repository tooling only. The same dashboard read git commit dates with
  `datetime.fromisoformat`, which rejects an RFC 3339 `Z` suffix on Python
  3.10 (3.11 was the release that learned it). `git log --format=%cI`
  renders a UTC commit date as `...Z` on some git versions, so reading a
  repository committed under UTC raised `Invalid isoformat string` on the
  oldest supported Python. All three readers now share one parser that
  accepts both spellings.

### Compatibility
- Zero-lag identity: with no settlement terms supplied, payment equals
  determination (`delay_df == 1`) and all prices, Greeks, event statistics
  and exercise boundaries are unchanged — verified by the existing suite
  plus the dedicated settlement test files.

### Also in this release
- Constructor normalization forwarding the shared lifecycle fields across
  the concrete equity option products (wip/equity-option-contract).
- Heston temporal regularization in the ADI core
  (wip/heston-temporal-regularization).
- PDE regression coverage ports, per-module reference docs, example suites
  (FX vol models, snowball backtest terms, bucketed-greeks scenarios), and
  modelvalidation certification archival.

## [0.4.6] - 2026-08-20

0.4.5's test suite passed on the ARM64 machine its goldens were frozen on and
failed 28 tests on x86_64 Linux. Chasing that down found one genuine pricing
defect and a set of comparisons that were architecture-dependent by accident.
The published 0.4.5 package is unaffected except as noted below.

### Fixed
- **Parallel-batch pricing of continuously monitored knock-in under Local Vol /
  Heston / SLV Monte Carlo.** The per-step variance the Brownian-bridge crossing
  estimate needs was held in a single engine attribute, written while simulating
  and read while computing payoffs. The dask path (`use_dask=True`) submits every
  batch through `delayed`/`compute` on the threaded scheduler, so batches sat
  between their own generation and their payoffs concurrently on one engine and
  overwrote each other — 20,000 paths over 3 batches is 6667/6667/6666. Now
  scoped per thread, which is where a batch begins and ends. A shape guard made
  this raise rather than price one batch against another's variance, so no wrong
  number could have been returned; serial pricing was never affected.
- `quantark.modelvalidation.anchors`: comparing a banked certificate off the
  banking machine now floors the tolerance at the suite's cross-architecture
  standard (`DEFAULT_REL_TOL` 1e-12 → 1e-9, `DEFAULT_ABS_TOL` 1e-15 → 1e-12).
  Measured across 326 anchors, x86_64-vs-ARM64 drift spans 1e-14 to 2.2e-11
  relative (median 2.0e-12, both signs, no outlier of a different order): last-ULP
  differences in FMA contraction, libm transcendentals and BLAS kernels, amplified
  by an autocallable PDE/QUAD solve marching hundreds of steps through barrier
  events. The bound keeps ~44x margin over that and stays six orders under any
  real numerics change. Applied as a floor on each banked file's own tolerances,
  so already-banked evidence is corrected without its anchored *values* being
  touched. Same-machine comparison remains exact.

### Changed (tests)
- The replay golden gate compares floats against each column's scale rather than
  bitwise. These frames carry bump derivatives, so the rounding noise is set by
  the magnitude of the price being differenced and is near-constant in absolute
  terms down a column: gamma drifts 4.6e-10 against a column max of 43, and the
  same quantity in cash drifts 3.8e-8 against 4,301 — one ratio, ~1e-11. Column
  names and order, row count, and integer, string and date columns still compare
  exactly. New tests pin the tolerance itself: it admits the measured drift,
  rejects 1e-6 of column scale, and every real golden frame is checked against a
  simulated drift locally, because bitwise-identical frames short-circuit before
  any tolerance is consulted and made earlier calibration bugs invisible here.
- Two cross-architecture anchor tests hardcoded the fingerprint
  `{"machine": "x86_64", "system": "Linux"}` — which is the CI machine, so there
  they took the same-machine *exact* branch, the opposite of what they verify.
  They now derive a fingerprint that cannot be the current machine.
- `test_event_stats_pruning` pinned a PV of ~9.9e5 with a `1e-6` absolute
  epsilon, i.e. 1e-12 relative — tighter than the measured drift, passing on
  margin rather than by design. Now on the documented cross-arch tolerance.
- A settlement fixture placed a strike exactly on the parity-implied forward.
  Stage 10 splits OTM puts from calls with `strike < forward` against an
  OLS-regressed forward, so that strike changed side with the last ULP of the
  regression. The grid now keeps every strike clear of the forward.

## [0.4.5] - 2026-08-19

Two programs land together: **model-release certification** — the first
module that decides whether an engine is fit to ship, and the numerics
fixes that decision surfaced — and the **backtest replay consolidation**.

### Model-release certification

#### Added
- `quantark.modelvalidation`: engine-release certification. A study YAML
  declares the product, the market environment, the candidate engines and
  the economic scale; the runner prices every (candidate, case) cell against
  a paired-RQMC benchmark, applies per-cell and aggregate-bias gates, and
  emits a signed certificate, a Markdown/HTML report, and an *anchor file*.
  Anchors are the cheap residue of an expensive run — the deterministic
  engines' own outputs at the certified configurations — so CI can re-check
  in seconds that a released engine still produces what its certificate
  claims (exact on the banking machine, tolerance-based elsewhere).
  `amend` extends a certification's scope while carrying unaffected cells
  forward on identity hashes; scope may grow but never shrink.
  Procedure: `docs/modelvalidation/RELEASE_PROCEDURE.md`.
- Banked flat-BSM certifications for snowball (23 cases), phoenix (14) and
  KO-reset snowball (14), PDE and QUAD both ADMITTED, under
  `docs/modelvalidation/certificates/`, covering the full product feature
  surface: discrete / European / stepping knock-in, step-down and parachute
  knock-out, reverse, airbag, protection types, participation, call rebate,
  `disable_ko_after_ki`, and EXPIRY coupons.

#### Fixed
- **Continuous knock-in was monitored at the time-step width** by the
  local-vol, Heston and SLV solvers, which were gated off the FIRST_PASSAGE
  correction. The correction is barrier-local, so it never needed globally
  constant coefficients — only the dynamics at the barrier. Each solver now
  reports its own: `sigma_loc(barrier, t)` for local vol, one
  `(mu, sigma^2)` per variance column for Heston, and `L(barrier,t)^2 * v`
  for SLV. On a flat surface the local-vol solver had been bitwise identical
  to the *uncorrected* flat-BSM solver (+0.099% of PV). Applies to the
  snowball and phoenix families alike; the phoenix 2-D solvers had never
  built the state at all.
- **The Monte Carlo continuous-KI Brownian bridge used the implied vol at
  the strike** as the variance of every path and every step. The vol-model
  engines now supply the log-variance their own scheme accumulated. Against
  the closed-form first-passage probability the error falls from -10.0 / -8.5
  / -9.2 standard errors (Heston / local vol / SLV) to inside 4.
- The first-passage correction estimated its slope from a node that could sit
  a fraction of a cell above the barrier, where the knock-in mask's own step
  — not the slope — dominates. Sampling is now floored at half a cell.
- `PhoenixOption` never built its knock-in observation schedule, so a
  discretely monitored phoenix raised instead of pricing.
- `disable_ko_after_ki` was ignored by the snowball and phoenix PDE solvers,
  which wrote the knock-out payoff to the knocked-in surface as well.
- Phoenix `coupon_pay_type=EXPIRY` coupons roll up and are paid at
  termination — the knock-out settlement when it knocks out, maturity
  otherwise — and are never forfeited. All three engine families agree;
  PDE and QUAD gained a termination-value surface to price it.

#### Changed
- Continuous knock-in prices move in the local-vol, Heston and SLV engines
  (PDE and Monte Carlo). Discretely monitored, European and no-knock-in
  products are bitwise unchanged, verified across 54 engine/product/
  monitoring combinations; the sole exception is the opt-in
  `KnockInMonitoringMode.BGK_APPROXIMATION`, which replaces a discrete
  schedule with continuous monitoring by design.

### Backtest replay consolidation

(spec: `docs/superpowers/specs/2026-07-30-backtest-replay-consolidation-design.md`).

#### Added
- `quantark.backtest.replay`: canonical home of the product-replay backtest —
  ONE multi-product daily loop (`ReplayBacktestEngine`/`ReplayBacktestConfig`/
  `ReplayProduct`/`ReplayBacktestResults`); the single-product
  `AutocallableBacktestEngine` is now a book-of-one wrapper with byte-identical
  output (golden-gated). Book runs gain per-day vol-model calibration.
- `quantark.backtest.futures_ledger`: shared `FuturesHedgePosition` +
  `FuturesRollPolicy`.
- `quantark.backtest.metrics.CorePerformanceMetrics`: cross-asset metrics over
  the `BaseBacktestResults` protocol; replay results expose `.metrics` and the
  full protocol; equity `PerformanceMetrics` now extends the core.
- `quantark.backtest.replay.schema`: typed row schemas — the single source of
  truth for every record column.
- Pending-settlement KO termination: `terminate_on_lifecycle_end=True`
  (default) ends a replay when terminal cash lands (observation vs settlement
  are separate moments; discounted receivable carried in portfolio value);
  summaries gain `termination_reason` / `days_replayed` / `days_in_contract`.
- `AutocallableEngineConfig.event_stats_fallback` (`"none"` default): the
  silent 5000-path MC event-stats fallback is now opt-in, logged, and recorded
  per row in the new `event_stats_engine` column.
- `BumpConfig.gamma_spot_bump`: engine-side asymmetric gamma bumping;
  replay-side manual bumping removed (greeks are always
  `engine.calculate_greeks`, failures raise — no silent zero-delta days).
- `AutocallableDeltaHedgeStrategy` joined the `BaseStrategy` hierarchy
  (`quantark.backtest.strategy`).

#### Changed
- Relocations (old paths remain as `DeprecationWarning` shims until 0.5.0):
  `quantark.backtest.otc.*` -> `quantark.backtest.replay.*`;
  `otc.vol_calibrators` -> `quantark.volmodels.calibration`;
  `otc.vol_history` -> `quantark.param.vol.surface_history`;
  `_atomic_write_json` -> `quantark.util.io.atomic_write_json`.
- Replay books with `vol_model != "bsm"` reject non-Snowball products at
  config construction (the vol-model engine factory is snowball-only).
- **Heston/Heston-SLV Snowball and Phoenix 2D PDE solvers default
  `v0_boundary="degenerate_pde"`** (was the ADI core's `"neumann"`), and accept
  it as a constructor argument. Smile-calibrated Heston sits near the Feller
  boundary, where the Neumann treatment of the `v = 0` edge mis-prices by
  −0.540% of notional against a QE-M MC reference while the degenerate
  treatment holds +0.156%. `"neumann"` remains selectable for cross-checks.
  **Prices change** for these four solvers at default settings.
- **Heston/Heston-SLV Snowball concentrated PDE grids default to a power-graded
  variance axis** (`v_grid_power=2.5`) to resolve the near-zero layer around the
  Feller boundary. `v_grid_power=0.0` retains the legacy concentrated mesh for
  explicit cross-checks; uniform grids remain ungraded. **Snowball Heston/SLV
  prices and finite-difference Greeks change** at default settings.
- **The `mo_frozen` Heston preset enforces the Feller condition**
  (`enforce_feller=True`; the soft `regularize_feller` penalty is retained).
  Deliberate divergence from the `mo_volmodels` diagnostics scripts, which keep
  the soft-only policy. Feasibility is bought with smile accuracy — measured
  across 762 CFFEX MO settlement surfaces: median 8.4 bp, p90 29.3 bp, max
  217 bp of IV. On 6.6% of dates the constraint is met by driving vol-of-vol to
  its lower bound, which degenerates Heston to deterministic variance; consumers
  should screen on the new `feller_ratio`. Heston and Heston-SLV calibration
  cache entries are invalidated automatically (the fingerprint embeds the
  preset); `localvol` entries are unaffected.

#### Fixed
- `BaseEngine.calculate_greeks` and Gate G2's deterministic delta helper bypassed
  `create_bump_context`, so Heston/SLV central spot bumps rebuilt their S grids even
  though the solvers already provided a frozen-layout context. Base, up, down, and
  separate gamma reprices now all use the resolved base-market context.
- `Heston2DAutocallableSessionAdapter` now preserves the Snowball solver's
  `v_grid_power`; an explicit grading or legacy opt-out no longer disappears when
  the execution session clones the engine. Gate metadata records the selected
  value and Stage 12 forwards it into the production engine options.
- `calibrate_heston(enforce_feller=True)` failed closed on any problem where the
  constraint was active at the optimum and `ftol` was loose. SLSQP satisfies
  constraints only to about its own accuracy tolerance, but the feasibility
  buffer was a constant `1e-8` — measured slack is ~1e-9 at `ftol=1e-8` but
  ~1.6e-7 at `ftol=1e-6`, so the strict post-check rejected every fit. The
  buffer now scales as `max(1e-8, 10 * ftol)`; the post-check is unchanged and
  still fails closed.
- `VolModelCalibrator` Heston records carry `feller_ratio` (`2*kappa*theta /
  sigma**2`) alongside `feller_margin`. The margin is a difference and so is not
  comparable across dates; the ratio is.
- `Heston2DAutocallableSessionAdapter._clone_engine` rebuilds the engine from an
  explicit kwargs list and dropped `v0_boundary`, so a session or prepared-adapter
  run silently priced with the default boundary while the direct call honoured the
  constructor. Any 2D autocallable constructor argument omitted from that list
  fails this way — silently, with no error.

## [0.4.0] - 2026-07-27

Clean rewrite of the PDE grid construction and event application layer
(spec: `docs/superpowers/specs/2026-07-27-pde-grid-redesign-design.md`).

### Added
- `quantark.asset.equity.engine.pde.grid`: declarative grid layer —
  `GridRequest` (frozen geometry), `GridConfig` + accuracy profiles
  (`fast`/`standard`/`high`), ONE time builder (event-aligned, extras-scaled
  cap, damping schedules), ONE spatial builder (multi-point sinh, pinned
  auto-bounds, no barrier snapping), four-stage `EventSchedule`
  (terminal/interior/continuous/valuation-readout, pure block transforms),
  engine-owned `GridBinder` (LRU, `bind_shared` for same-underlying books,
  `rebind_time` for calendar rolls) and `GridLayerMixin` for
  BaseEngine-derived adopters.
- `PDEParams(accuracy=..., grid=GridConfig(...))` tiered front door.
- Frozen-`Layout` bump contexts: spot/vol/rate/div bumps reuse the base
  layout by object identity; calendar rolls rebind time on the same spatial
  object.
- `HestonSLVADICore(x_nodes=...)`: 2D solvers take their S-axis from the
  shared spatial builder.
- Tier-2 anchor certification (`test/pde_grid/test_anchor_certification.py`):
  MC/QUAD/closed-form/smoothness anchors gate every accuracy profile.

### Removed (breaking)
- Legacy grid modules `time_grid.py`, `spatial_grid.py`,
  `event_projection.py` (projection math moved verbatim to `grid/events.py`).
- `PDEParams` knobs: `auto_grid`, `time_grid_type`, `grade_exponent`,
  `event_min_steps_per_interval`, `log_dx_target`,
  `include_spot_in_critical_points`, `frozen_critical_points`,
  `barrier_refine_log_width`, `barrier_refine_levels`,
  `barrier_domain_expand`, `adaptive_grid`, `event_steps_per_day`,
  `max_time_steps`, `max_grid_size` (the last three had no remaining
  consumers — `GridConfig.steps_per_day` / `max_steps` / `max_points` own
  those roles).
- The class-level shared grid cache (`clear_grid_cache` is now a no-op).

### Changed
- All PDE prices reprice slightly (re-certified against MC/QUAD/closed-form
  anchors, goldens re-frozen): the layer builds concentration-based grids
  without barrier snapping — cell-average event projection is placement-
  independent — and event-aligned time grids emit one exact dt per interval
  (operator caches are arithmetic-neutral: cache-on == cache-off bitwise).
- Grid-layer solvers REJECT non-default `grid_size`/`time_steps`/`s_min`/
  `s_max` with a `ValidationError` naming the `accuracy`/`GridConfig`
  replacement (previously the values would have been silently inert); the
  knobs remain live inputs for the standalone vol-model solvers only.
- `make_pde_params` profiles (`fast`/`accurate`/`barrier_sensitive`/
  `reverse_sensitive`) now emit `accuracy`/`grid=GridConfig(...)` instead of
  the legacy knobs (which also fixes a crash: the product-hint path injected
  deleted `barrier_refine_*` keys).
- Bump contexts (`create_bump_context`) clone via `deepcopy`, preserving
  full constructor state (Heston `model_params`, SLV leverage surfaces, ADI
  dimensions); `PDEEngine.create_bump_context` carries the frozen solver
  instance. Frozen layouts fail closed when reused across different hard
  (absorbing-barrier) bounds, and calendar-roll rebinds coverage-validate.
- 2D Heston/SLV snowball/phoenix bump contexts freeze the ADI S-axis at the
  solve configuration (`n_x`, `num_std=8`) and reuse it by identity under
  market bumps (`_layer_x_nodes` routes through `resolve_bound_layout`);
  previously the S-axis was rebuilt per bump, mixing grid movement into 2D
  numerical greeks.
- The scheme knobs are LIVE on the grid layer: `use_rannacher`/
  `rannacher_steps` drive terminal damping and `rannacher_at_events`/
  `event_rannacher_steps` drive event damping (the binder derives the
  damping schedule from them; explicit `GridConfig` fields still win). At
  default knob values the schedule is unchanged.
- Prepared/session PDE pricing resolves solve state
  (`_prepare_for_request`) before fingerprinting grid geometry — BGK
  products no longer trip `DeterminismViolation` against their own
  preparation when an artifact cache is active.

### Fixed
- Autocallable PDE grids ignore out-of-reach critical-price markers (such as
  disabled 100× KO observations) for bounds and concentration while retaining
  the observation date and any coupon event. Dense critical sets now use
  bounded cumulative inversion of the Tavella-Randall monitor, removing the
  overflow-prone shooting path and its non-monotonic construction-time cliff.
- Snowball/Phoenix QUAD event handling now defaults to cell-average projection
  and phase-stable trapezoidal integration. Unreachable KO markers are
  suppressed without deleting their dates, and the legacy nodal/Simpson
  behavior remains available explicitly.
- QUAD can optionally refine nested odd grids until consecutive PV estimates
  satisfy configured absolute/relative tolerances. Fixed
  `grid_points=1001` remains a speed-oriented estimate rather than a universal
  convergence guarantee.

### Deferred (recorded follow-ups)
- Event-aligned 2D ADI time axis (uniform march retained).
- Standalone vol engines (`LocalVolPDESolver`, `HestonPDESolver`,
  `HestonSLVPDESolver`, LV/Heston/SLV barrier engines, `HestonDCNPDESolver`)
  keep bespoke kernels and their `grid_size`/`time_steps`/`s_min`/`s_max`
  inputs; `event_projection` + Rannacher scheme knobs survive for the live
  characterization paths.
- Stencil-delta accuracy on concentrated grids (~2e-3 absolute) —
  provisional 5e-3 anchor band.

## [0.3.0] - 2026-07-20

First release of the `quantark.execution` framework kernel and the
`quantark.execution.greeks` scenario layer. The `test_snowball_quad_flat_identity_golden`
failure predates this program and remains quarantined (reproduces on
unmodified main). The spec §20 controlled-host performance gates in
`docs/execution/README.md` are documented as release-preparation evidence
and were not re-measured on the release host.

### Added
- **`quantark.execution.greeks` — greek bumps as scenario cells** (spec
  2026-07-20): `TradeState` per-trade base, same-type `greek-bump/v1`
  transformer with real spot/vol/rate/div/time mutation attribution,
  `greek-value/v1` float runner, and assemblers mirroring
  `EquityPosition.get_trade_risk` / `GreeksCalculator.
  calculate_numerical_greeks` operation-for-operation (bitwise unit gate).
- **`PricingSession.run_scenario_plans`** — many `(base, specs)` plans
  packed through ONE bounded-window process pool (per-cell worker-spec
  payloads; the portfolio × bumps shape), with a per-plan error boundary
  under `collect_errors=True` (base-resolution/planning failures become
  aligned typed `PricingFailure`s while other plans execute).
- **`quantark.execution` — composable execution kernel** (framework contract
  v1; spec `docs/superpowers/specs/2026-07-15-mc-pde-performance-generalization-design.md`).
  `PricingSession` wraps every exported MC/PDE engine without changing direct
  legacy behavior: adapter registry, resource budgets/leases, prepared-artifact
  and session draw caches, reproducibility manifests and immutable diagnostics.
  Capabilities: fixed-batch MC backends (serial/threads, bit-identical
  reductions in canonical order), adaptive RQMC session mode (bitwise vs
  direct across the 12 autocallable vol engines), PDE preparation artifacts
  (grids, step coefficients, factorization packs; one-solve PV/event/grid
  outputs across 18 prepared engines + FX LV), typed scenario execution with
  verified mutation footprints, spawn-safe `WorkerSpec` process workers, a
  Dask backend over the same plans, `price_many` grouping, and a
  complete-payload scenario validator. Published artifacts: generated
  [capability matrix](docs/execution/capability-matrix.md) (CI-fresh), policy
  guide, legacy-internals rationale, reproducibility JSON Schemas (validated
  against live payloads), and runnable migration examples. Adoption is
  inventory-driven and honest: `temporary_legacy` rows remain, and no
  universal framework support is claimed until the controlled-host gates pass.
- **DCN (Digital Coupon Note)**: product + observation-schedule generator,
  curve-aware MC engine with leg decomposition and event stats, two-surface
  PDE engine with MC↔PDE cross-validation gates, LocalVol/Heston/QE/SLV DCN
  MC engines sharing one payoff kernel, MLMC-style coupled timestep-ladder
  Heston pair, SVI/cleaned-quote calibration layers with no-arbitrage and
  repricing-residual reports, node-bump vega buckets and carry-invariant rate
  bumps.
- **Implied futures carry**: `IndexFuturesCurve` implied q(T), futures-tenor
  delta/rhoq buckets, delta-one futures rhoq, portfolio aggregation and hedge
  integration.
- **Engine term-structure upgrade**: MC, PDE, and quadrature autocallable
  engines consume full r/q/vol term structures with a cross-family term
  agreement gate.
- **Vol-model barrier exotics**: LV/Heston/SLV barrier MC pricers on a shared
  monitoring core, LV 1D barrier PDE, Heston/SLV 2D barrier ADI solvers.
- **Vol-model numerics program (phases 2–5)**: vectorized Lewis Heston
  calibration (~21x), unified Heston/SLV ADI core, opt-in Krylov and TR-BDF2
  Fokker–Planck marches, opt-in QE-M martingale correction, concentrated
  (x,v) grids, degenerate v=0 Feller boundary, LV Rannacher start-up
  (default on), opt-in QMC sampling for Heston/SLV/LV kernels.
- **PDE event-stats API** on Snowball/Phoenix vol solvers with event
  distributions surfaced through session outputs.

### Changed
- DCN MC engine performance: draw cache, batched threads, LV build hoist.
- The legacy autocallable Dask batch loop (Snowball vanilla/KO-reset,
  Phoenix) is consolidated into one shared reducer
  (`autocallable_dask_batch`) with byte-preserved behavior, gated by frozen
  bitwise goldens. The legacy `use_dask` route itself is unchanged and
  preserved (see `docs/execution/internals-and-legacy.md` for the §17.3
  removal preconditions).
- Dev extras now include `dask[distributed]` and `jsonschema`.

### Fixed
- `PhoenixMCEngine` silently disabled its Dask parallel path on modern dask
  (`from dask.compute import compute` no longer resolves); availability now
  matches `SnowballMCEngine`, with a regression test.
- Processes scenario backend: a parent whose `__main__` cannot be
  re-imported by spawn children (stdin/heredoc scripts record
  `__file__ = '<stdin>'`; deleted script files) now fails closed with a
  typed `CapabilityError` before any worker is spawned, instead of every
  child dying at bootstrap and surfacing as an opaque
  `BrokenProcessPool` with zero completed cells.
- Craig–Sneyd ADI corrector order restored (base `Y0` + implicit `-rU`),
  QE sampler sign bug, and the 2026-07 PDE/QUAD audit fixes (13 findings)
  are included via the programs above.

## [0.2.5] - 2026-07-03

### Fixed
- 2026-07 PDE/QUAD audit: 13 numerical findings fixed (discrete-monitoring
  boundary handling, Rannacher terminal off-by-one, BGK state resolution,
  smoothed KI for Phoenix/KO-reset, shared theta damping schedule) with
  solver-family consolidation and regression tests.

## [0.2.4] - 2026-07-02

### Added
- Opt-in BGK continuous-KI mode for Snowball and Phoenix PDE solvers
  (direction-aware shifted barrier, per-product flag).
- `EquityPosition.get_trade_risk` (product + total, frozen one-loop) and
  `AutocallableCashLeg.time_shift`; event streams selected by leg
  requirements.

### Fixed
- Portfolio vega/rho/dividend-rho aligned with the canonical greek
  convention; bump contexts freeze PDE critical points for grid steadiness.

## [0.2.3] - 2026-07-01

### Added
- Cross-engine-consistent `ki_ever` / `ki_survive` event-stat fields.

### Fixed
- Theta advances by business day; quadrature event-stats recursion uses
  smoothed barrier indicators.

## [0.2.2] - 2026-07-01

### Fixed
- PDE numerical Greeks: freeze the base spatial domain for finite-difference
  bump repricing so rho and dividend rho measure market sensitivity without
  contamination from auto-grid/domain movement under bumped rate, dividend,
  volatility, or time inputs.

## [0.2.1] - 2026-06-30

### Fixed
- `PhoenixPDESolver`: apply a KO observation scheduled exactly at maturity.
  The inherited grid builder stores the maturity KO in `_ko_terminal_record`
  (intentionally kept out of `_ko_observation_indices`), but the Phoenix
  `_solve` override looked the terminal KO up in `_ko_observation_indices`
  and therefore dropped it — mispricing products with a terminal KO date by
  several percent versus the quadrature and Monte Carlo engines. Terminal KO
  is now applied after the terminal coupon/KI jumps (matching
  `SnowballPDESolver._solve`), routed through `_apply_ko_jump_vector` so the
  same-date coupon-at-KO payoff is preserved.

## [0.1.2] - 2026-06-13

### Added
- Credit dual-measure framework: a recovery convention layer
  (`quantark.asset.credit.conventions`, `STANDARD_RECOVERY=0.40`) that
  separates the canonical shared-curve **hazard01** factor from the
  recovery-converted **CS01** used by products and SIMM. Curve shocks stay
  in hazard space; spread stresses convert through the recovery convention.
- Single-name CDS **roll-down / as-of pricing** via effective and maturity
  dates (seasoned and forward-start), with `schedule_asof` and a
  total-return coupon cash ledger threaded through the dynamic-scenario and
  backtest engines. SIMM buckets the remaining tenor. (Basket as-of is
  deferred.)

## [0.1.1] - 2026-06-11

### Added
- `SnowballQuadEngine`: explicit `ki_monitoring_mode` on `QuadParams`
  (`KnockInMonitoringMode`). `EXACT_DISCRETE` (default) prices every KI
  observation date exactly with adaptive spatial-grid refinement.
  `BGK_APPROXIMATION` is an opt-in performance mode that replaces a dense
  discrete KI schedule with continuous monitoring at a
  Broadie-Glasserman-Kou shifted barrier; the engine validates approximately
  regular spacing (median-band dispersion test), a constant resolved
  barrier, full-horizon coverage, stable volatility, and a minimum schedule
  density (`bgk_min_ki_observations`), raising `ValidationError` otherwise.
  Converted pricing matches the equivalent shifted-continuous product
  exactly (grid-aligned to the shifted barrier) while the valuation-time KI
  state keeps contractual discrete semantics. A first-order residual bias
  remains (a few bp of PV at daily spacing, growing with observation
  spacing and drift).

### Fixed
- `SnowballQuadEngine`: dense discrete KI schedules now retain their explicit
  observation dates instead of being delegated to continuous monitoring. The
  engine adaptively refines its internal spatial grid to resolve short
  diffusion intervals (accuracy-oriented default of 2.5 cells per interval
  diffusion stddev; lower to 1.25 or opt into BGK for speed), avoiding the
  material fair-KO-rate bias caused by the previous continuous-monitoring
  approximation.

## [0.1.0] - 2026-06-11

### Added
- First public release.
- Equity derivatives: European/American/Asian vanilla options, barrier,
  one-touch, digital, sharkfin, and autocallable products (snowball,
  phoenix, KO-reset snowball, range accrual) with analytical, Monte
  Carlo, PDE, quadrature, and tree engines.
- Fixed income: fixed bonds, FRNs, bond options, bond forwards/futures,
  convertible bonds, interest rate swaps.
- Market data layer (`quantark.param`, `quantark.priceenv`), Greeks
  calculators, portfolio VaR (parametric/historical/Monte Carlo),
  ISDA SIMM v2.6, stress testing, multi-day scenario simulation, and a
  hedging backtest framework.
- Legacy flat-import compatibility shim (`asset`, `util`, …) with
  `DeprecationWarning`; slated for removal in 1.0.
