# KO-reset snowball lifecycle tracking — design

Status: approved in conversation 2026-10-08 (sections 1–4); revised for review findings 1–5 and the user's
rulings on the day rule and the `REBASED` provider matrix (2026-10-08); awaiting review.

## 1. Purpose

`KnockOutResetSnowballOption` has no lifecycle tracking. `PortfolioLifecycleManager` warns "KO-reset snowball
lifecycle ... is not tracked" and reprices the position as of every day, so:

- a backtest or dynamic scenario holding a KO-reset never books its knock-out, knock-in or maturity cashflows;
- the day after schedule 1 ends without a knock-in, every engine raises (`require_alive`: the contract matured),
  because nothing closed the position;
- single-product replay builds the snowball tracker for it (no schedule switch, no schedule-1 maturity), and its
  QUAD route sends it to `SnowballQuadEngine`, which ignores schedule 2: a knocked-in contract prices 4.195 where
  `KOResetSnowballQuadEngine` prices -1.744;
- the simulated-path backtest refuses it.

**Scope (user, 2026-10-08: "(c), we fix it for good").** The lifecycle core (tracker, portfolio manager — the equity
backtest and dynamic scenarios run through it), single-product replay, and the simulated-path backtest. Both
schedule-2 modes, `ABSOLUTE` and `REBASED` ("common", user 2026-10-08).

**Success.** A KO-reset position in any of those consumers books its events and cashflows, is closed when the
contract ends, is never repriced after it matured, and never raises on a live state. Snowball and Phoenix results
stay byte-identical.

## 2. Contract rules (user rulings)

- **R1** (2026-09-21) A knock-in during schedule 1 replaces schedule 1 by schedule 2 for good; schedule 2 is in force
  from the knock-in observation, that observation included.
- **R2** (2026-09-21, restated 2026-10-08) A contract not knocked in when schedule 1 ends has matured there: it pays
  the rebate (V0) on schedule 1's contract length and is never extended to schedule 2. A knock-in is never tested
  after schedule 1 ends.
- **R3** (2026-09-21) A schedule-1 knock-out and a knock-in at the same observation: the knock-out wins; schedule 1 is
  not replaced.
- **R4** (2026-10-08) A carried knock-in is an event that happened; every pricer takes it as such.

These are the rules the engines and the intraday twin already apply; the tracker adopts them, it does not
reinterpret them.

## 3. Approach

A `KOResetLifecycleTracker` subclass of `AutocallableLifecycleTracker` and a KO-reset variant of the simulation's
vector lifecycle (approach B, approved 2026-10-08); the KO-reset rules live in one place. Snowball and Phoenix event
code is not touched. One shared change reaches them on purpose: §4's day rule (user, 2026-10-08: "put the
day-rounding mismatch in this spec"), which moves the base tracker's pricing copy from time-based ageing to
removing observed records by identity (§4a).

Rejected: (A) KO-reset branches inside the existing tracker — interleaves the rules and puts every snowball golden
behind inert branches; (C) swapping the product for a knocked-in `SnowballOption` at the knock-in — loses the
product identity in reports and still needs the pre-knock-in phase logic in the simulation.

## 4. State and event rules (`KOResetLifecycleTracker`)

`AutocallableLifecycleState` gains `knock_in_date: Optional[datetime]` (the close the knock-in was observed on), set
by `mark_ki`. It is the one record of WHEN; on the numeric clock the knock-in time is derived from it exactly as the
tracker derives every valuation point, `knock_in_time = (knock_in_date - start_date).days / 365` (years from the
tracker's `start_date`, the origin of the product's numeric schedule). Snowball and Phoenix never read it. The phase
follows from the state: schedule 1 until a knock-in, schedule 2 after.

**Initial state (review finding 4).** Every consumer hands the tracker its starting state explicitly:
- a `lifecycle` state passed in (replay checkpoints, the dynamic-scenario and backtest managers when they resume) is
  used as given;
- otherwise the state is built from the product: `knocked_in = product._otc_lifecycle_knocked_in` and
  `knock_in_date = product._otc_lifecycle_knock_in_date` (a new carried attribute beside the flag);
- the tracker never overwrites a carried knock-in with a default `False` (the base tracker's `product_for_lifecycle`
  does today; the KO-reset tracker builds its state first and writes the flag from it);
- a carried knock-in on a `REBASED` contract without a `knock_in_date` is refused at construction
  (`ValidationError`: schedule 2 cannot be placed without it). An `ABSOLUTE` contract needs no date.

**One day rule (review finding 2).** A record falls due on the first available close whose day index reaches the
record's due day, where a dated record's due day is its date and a numeric record's due day is
`round(time * 365)` days after `start_date` — the tracker's existing `_record_is_due` rule, kept for every product.
A `REBASED` record's numeric time is `knock_in_time + offset`, so its due day is
`round((knock_in_time + offset) * 365)`. A due day with no market close (holiday, data gap) fires on the next
available close. Ageing never decides by time comparison whether a record is still pending: the pricing copy
removes the records this tracker has OBSERVED, by their stable identity (schedule + index, the `observed_*_indices`
sets), and keeps every other one. So a target at day 1.4, observed on day 1's close, is gone from day 1's pricing
copy; and an unobserved record whose time has passed (the gap case) is still observed at the next close and never
silently dropped.

Each daily close, in this order:

1. **Not knocked in (schedule 1).**
   - Schedule-1 knock-out records due today are tested first. A hit books `KNOCK_OUT` with the record's payoff and
     settlement, terminates the position and ends the day (R3).
   - Knock-in is tested next: discrete knock-in records due today, or the close under continuous monitoring. A hit
     books `KNOCK_IN`, records `knock_in_date`/`knock_in_time` and switches to schedule 2; schedule-2 knock-out
     records due today are then tested the same day (R1, inclusive).
   - Schedule-1 maturity: when schedule 1's last observation was today and neither event fired, the contract matures
     (R2): `MATURITY` with `get_maturity_payoff_v0(close, env, contract_tenor=<schedule-1 tenor>)`, paid with the
     terminal settlement lag after schedule 1's last date (QUAD V2's pre-maturity payment timing), and terminates.
2. **Knocked in (schedule 2).** Only schedule-2 knock-outs are tested, under the day rule above, and none at all
   when `barrier_config.disable_ko_after_ki` is set (review finding 3: the product's decision and every KO-reset
   engine skip schedule 2 under the flag) — for a knock-in fresh today and a carried one alike. With no knock-out the
   contract matures with the knocked-in payoff (`get_maturity_payoff_v1` on schedule 2's tenor) at its final
   maturity: the product's maturity for `ABSOLUTE`; `knock_in_time + max(offsets)` for `REBASED` (review finding 1:
   `get_post_maturity_time(ki_time)`), so a one-year final offset matures at 1.25 years after a knock-in at 0.25 and
   at 1.5 after one at 0.5. Its settlement follows the terminal lag after that maturity, and V1's contract tenor is
   `_resolve_post_contract_tenor` at that maturity.

Event indices name their schedule (`ko1:i`, `ko2:i`, `ki:i`) so ledger cashflow IDs stay unique.

## 4a. The day rule in the base tracker (every autocallable)

The mismatch of §4 is the base tracker's, for snowball and Phoenix too: `_record_is_due` fires a numeric record on
`round(time * 365)` days while `product_for_pricing` ages by `barrier_config.time_shift(elapsed)` and keeps a record
whose time exceeds the elapsed time. A numeric record off a whole day (`k / 12`, `k / 52`, `k / 252` schedules: day
30.42, 7.02, 1.45) is observed on its rounded day but stays in that day's pricing copy as a future observation (or,
rounding up, is dropped from the copy a day before it is observed).

`AutocallableLifecycleTracker.product_for_pricing` removes the records the tracker has observed — `observed_ko_indices`,
`observed_ki_indices`, `observed_coupon_indices` — by identity and keeps the others, with times measured from the
pricing date as today; ageing by time comparison is gone from the pricing copy. Dated records and numeric records on
whole days are removed on the same day as before, so the change moves a price only on a day where the old copy kept an
observed record or dropped an unobserved one. The simulated-path backtest's aged product (`aged_product`) goes through
the same tracker and follows; its `VectorLifecycle` already fires on the tracker's due days.

Acceptance: every replay, simulation and lifecycle golden is byte-identical except on such days; each moving
row is listed with the record that caused it (old, new value); the PDE byte book is untouched (it does not age
through the tracker).

## 5. Pricing copy, portfolio manager, replay

**`product_for_pricing`.** Ages maturity, schedule 1, the knock-in schedule and an `ABSOLUTE` schedule 2; removes the
observed records by identity (§4's day rule), never by time; sets `_otc_lifecycle_knocked_in` and
`_otc_lifecycle_knock_in_date` from the state. A knocked-in `REBASED` contract's copy is converted to the `ABSOLUTE`
contract it has become:
- schedule 2 at `knock_in_time + offset` (measured from the pricing date), the observed records removed, each record
  keeping its barrier, rate and accrual from inception;
- `post_ko_mode = ABSOLUTE`;
- its horizon reset (review finding 1): maturity `knock_in_time + max(offsets)` measured from the pricing date — the
  original contract's maturity is NOT aged on — with the terminal settlement and V1's contract tenor following that
  maturity, so the copy pays and discounts V1 when the knocked-in contract ends.
Every engine then prices an ordinary knocked-in `ABSOLUTE` contract. Before a knock-in a `REBASED` contract stays as
booked (offsets): see §6a for which engines price it. A matured or knocked-out contract is closed and never priced
again.

**MC terminal spot (review finding 1).** `SnowballMCEngine`'s KO-reset path reads V1 off the grid's last node
(`paths[is_v1, -1]`) while it settles V1 at `ki_time + max(offset)`: under `REBASED` a path knocked in early matures
before the grid ends and its V1 is read at the wrong time. V1 is read at each path's own maturity node
(`ki_time + max(offset)` on its grid); the `ABSOLUTE` path is unchanged (its maturity is the last node). Control:
a `REBASED` contract with two knock-in dates against the two `ABSOLUTE` contracts they become, and a grid
extended past a path's maturity leaves that path's V1 unchanged.

**`PortfolioLifecycleManager`** builds a `KOResetLifecycleTracker` for a KO-reset (the warning goes), its initial state
from the product's carried attributes (§4), and processes it as it processes the autocallables. Snowball and Phoenix
keep their tracker.

**Replay.** `ProductReplay` selects the tracker class by product type and passes its `lifecycle` (checkpoint) state
through, or builds it from the product (§4). The engine factory routes a KO-reset to the
KO-reset engines: QUAD → `KOResetSnowballQuadEngine` (MC already dispatches inside `SnowballMCEngine`; PDE through
the facade's adapter). The local-vol / Heston / SLV replay routes build snowball engines only: each is measured on a
KO-reset and refuses it where it would misprice schedule 2. Snowball and Phoenix replay goldens stay byte-identical.

## 6. Simulated-path backtest

- `EnsembleConfig` accepts `KnockOutResetSnowballOption` beside `SnowballOption`; the vol-model gate refuses a
  KO-reset where §5's measurement shows the vol-model engine cannot price schedule 2.
- `KOResetLifecycleTracker.resolve_calendar_schedule` returns a `KOResetCalendarSchedule`: the snowball fields for
  schedule 1 and the knock-in, plus schedule-2 data (absolute due days, barriers, payoffs, settlement days for
  `ABSOLUTE`; offsets and a resolver from (knock-in day, offset) to the due day under §4's day rule for
  `REBASED`), the schedule-1 maturity day with its settlement day and delay, the final maturity day for `ABSOLUTE`
  and, for `REBASED`, a resolver from the knock-in day to the final maturity day, its settlement day and delay, and
  V1's contract tenor (review finding 1), and `disable_ko_after_ki`.
- `KOResetVectorLifecycle` keeps the per-path arrays plus `knock_in_day` (-1 while not knocked in) and applies §4's
  order per day: schedule-1 knock-out; knock-in on paths that did not knock out today (setting `knock_in_day`,
  resolving a `REBASED` path's schedule-2 days, final maturity day, settlement day and tenor); schedule-2 knock-out
  including paths knocked in today, skipped entirely under `disable_ko_after_ki`; schedule-1 maturity (V0 on schedule
  1's tenor, settled with its lag); final maturity of knocked-in paths (V1) on each path's own maturity day — per
  path under `REBASED`. The maturity payoff callback receives the per-path knocked-in flag (V0 or V1) and, under
  `REBASED`, the path's V1 tenor.
- `DayStates`, `StateKey` and the aged-product cache gain `knock_in_day`, set only for a knocked-in `REBASED` path
  (-1 otherwise: snowball and `ABSOLUTE` keys, caches and goldens unchanged). The aged product of such a path is
  §5's pricing copy for that knock-in day. Surface mode builds a knocked-in surface per (bucket, knock-in day)
  visited, lazily; exact mode is unchanged. Which providers can price the un-knocked `REBASED` contract is §6a.
- The conformance oracle (exact mode == replay to the last bit) gains KO-reset cases (§7).

## 6a. Who prices an un-knocked `REBASED` contract (user, 2026-10-08: option (a))

Only `SnowballMCEngine` prices a KO-reset whose `REBASED` schedule 2 is still offsets (it places schedule 2 after each
simulated knock-in); `KOResetSnowballPDESolver` / the facade and `KOResetSnowballQuadEngine` raise, QUAD V2 refuses
relative resets, the intraday twin takes `ABSOLUTE` only. A knocked-in `REBASED` contract is priced by every engine
through §5's `ABSOLUTE` copy. So:

| consumer | un-knocked `REBASED` | knocked-in `REBASED` |
|---|---|---|
| portfolio manager (equity backtest, dynamic scenarios) | MC pricing engine required | any engine |
| single-product replay | MC pricing engine required | any engine |
| simulated-path backtest, exact mode | MC pricing engine required | any engine |
| simulated-path backtest, surface mode | refused | (unreachable: refused) |
| intraday twin | refused (unchanged) | refused (unchanged) |

Each refusal is raised when the configuration is built (`ReplayBacktestConfig`, `EnsembleConfig`, the manager's
position registration), naming the product, the mode and the engine — never mid-run on the first un-knocked day.
`ABSOLUTE` contracts have no restriction.

## 7. Tests and acceptance

- **Tracker rules** (`test/lifecycle_ko_reset/`), one case per rule: schedule-1 knock-out; R3 tie; knock-in → schedule
  2 with a schedule-2 knock-out on the knock-in day; a knock-in on schedule 1's last observation switches instead of
  maturing; un-knocked at schedule 1's end matures with V0 on schedule 1's tenor, equal to the engines' value that
  day (2.5 on the measured case, all four engines), settlement lag honoured; continuous knock-in tested only during
  schedule 1; `REBASED` dates follow the knock-in date; knocked-in final maturity pays V1, under `REBASED` at
  `knock_in_time + max(offset)` (knock-ins at 0.25 and 0.5 years with a one-year final offset mature at 1.25 and 1.5);
  `disable_ko_after_ki`: no schedule-2 knock-out on the knock-in day nor on a later day, fresh and carried knock-in;
  day rule: a numeric target at day 1.4 fires on day 1's close and is absent from day 1's pricing copy, a due day on
  a holiday fires on the next close, an unobserved past record is observed, never dropped; initial state: a product
  carrying a knock-in keeps it through the tracker (the flag is not reset), a carried `REBASED` knock-in without a
  date is refused.
- **Day rule, base tracker (§4a):** a snowball and a Phoenix with `k / 12` numeric schedules: on an observation day the
  observed record is absent from the pricing copy and the next one present (before: present); the replay-golden diff
  lists every moved row with its record; all other rows byte-identical.
- **Provider matrix (§6a):** each refusal raised at configuration with its message; an MC replay of an un-knocked
  `REBASED` contract runs through a knock-in and its knocked-in days price on PDE and QUAD too.
- **Pricing copy:** a knocked-in `REBASED` copy prices as the same contract with an `ABSOLUTE` schedule 2 at
  knock-in + offsets, on every engine, to 1e-12; the tracker's copy agrees with the intraday twin at a matching
  checkpoint.
- **Consumers:** the manager closes a KO-reset at schedule 1's end and the next day does not raise; dynamic scenario
  and equity backtest smoke cases; replay QUAD prices a knocked-in KO-reset on `KOResetSnowballQuadEngine` (the
  4.195 vs -1.744 case as a regression).
- **Simulation:** conformance cases for an `ABSOLUTE` and a `REBASED` contract over paths that knock out on schedule
  1, mature un-knocked, knock in then knock out on schedule 2, and knock in then mature; surface mode's `REBASED`
  per-knock-in-day surfaces within the existing gate against exact mode.
- **Unchanged:** snowball and Phoenix replay goldens and simulation snowball goldens byte-identical apart from §4a's
  listed rows; the PDE byte book byte-identical; full suite `-n 6` under the RAM guard.

## 8. Out of scope

- A PDE / QUAD route for an un-knocked `REBASED` KO-reset (§6a): the knocked-in value depends on the knock-in time,
  so it needs one knocked-in solve per knock-in date folded in at each knock-in observation — its own spec, with its
  own measurements and certification. Until then §6a's matrix stands.

New local-vol / Heston / SLV KO-reset engines. Under those models a KO-reset is refused wherever a snowball engine
would misprice it.
