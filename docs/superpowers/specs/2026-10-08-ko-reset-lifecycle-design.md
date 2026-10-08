# KO-reset snowball lifecycle tracking — design

Status: approved in conversation 2026-10-08 (sections 1–4); spec awaiting review.

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
vector lifecycle (approach B, approved 2026-10-08). Snowball and Phoenix code paths are not touched, so their
outputs are byte-identical by construction; the KO-reset rules live in one place.

Rejected: (A) KO-reset branches inside the existing tracker — interleaves the rules and puts every snowball golden
behind inert branches; (C) swapping the product for a knocked-in `SnowballOption` at the knock-in — loses the
product identity in reports and still needs the pre-knock-in phase logic in the simulation.

## 4. State and event rules (`KOResetLifecycleTracker`)

`AutocallableLifecycleState` gains `knock_in_date: Optional[datetime]` and `knock_in_time: Optional[float]` (the
numeric clock), set by `mark_ki`. Snowball and Phoenix never read them. The phase follows from the state: schedule 1
until a knock-in, schedule 2 after.

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
2. **Knocked in (schedule 2).** Only schedule-2 knock-outs are tested. A `REBASED` record's offset is a
   numeric time (the product refuses dates for it), so it falls due on the first close whose numeric valuation time
   reaches `knock_in_time + offset` — the tracker's existing `_record_is_due` rule on the numeric clock. With no knock-out the contract matures at its final
   maturity with the knocked-in payoff (`get_maturity_payoff_v1` on schedule 2's tenor).

Event indices name their schedule (`ko1:i`, `ko2:i`, `ki:i`) so ledger cashflow IDs stay unique.

## 5. Pricing copy, portfolio manager, replay

**`product_for_pricing`.** Ages maturity, schedule 1, the knock-in schedule and an `ABSOLUTE` schedule 2 as the
snowball tracker ages its schedules; sets `_otc_lifecycle_knocked_in` from the state. A knocked-in `REBASED`
contract's copy carries an `ABSOLUTE` schedule 2 at `knock_in_time + offset`, measured from the pricing date, past
records dropped by the usual ageing rule, each record keeping its barrier, rate and accrual from inception — every
engine then prices an ordinary knocked-in `ABSOLUTE` contract. Before a knock-in a `REBASED` schedule stays offsets;
each engine's ability to price that is measured, and an engine that cannot refuses explicitly. A matured or
knocked-out contract is closed and never priced again.

**`PortfolioLifecycleManager`** builds a `KOResetLifecycleTracker` for a KO-reset (the warning goes) and processes it
as it processes the autocallables. Snowball and Phoenix keep their tracker.

**Replay.** `ProductReplay` selects the tracker class by product type. The engine factory routes a KO-reset to the
KO-reset engines: QUAD → `KOResetSnowballQuadEngine` (MC already dispatches inside `SnowballMCEngine`; PDE through
the facade's adapter). The local-vol / Heston / SLV replay routes build snowball engines only: each is measured on a
KO-reset and refuses it where it would misprice schedule 2. Snowball and Phoenix replay goldens stay byte-identical.

## 6. Simulated-path backtest

- `EnsembleConfig` accepts `KnockOutResetSnowballOption` beside `SnowballOption`; the vol-model gate refuses a
  KO-reset where §5's measurement shows the vol-model engine cannot price schedule 2.
- `KOResetLifecycleTracker.resolve_calendar_schedule` returns a `KOResetCalendarSchedule`: the snowball fields for
  schedule 1 and the knock-in, plus schedule-2 data (absolute due days, barriers, payoffs, settlement days for
  `ABSOLUTE`; offsets and a resolver from (knock-in day, offset) to the due day under the tracker's rule for
  `REBASED`), the schedule-1 maturity day with its settlement day and delay, and the final maturity day.
- `KOResetVectorLifecycle` keeps the per-path arrays plus `knock_in_day` (-1 while not knocked in) and applies §4's
  order per day: schedule-1 knock-out; knock-in on paths that did not knock out today (setting `knock_in_day`,
  resolving a `REBASED` path's schedule-2 days); schedule-2 knock-out including paths knocked in today;
  schedule-1 maturity (V0 on schedule 1's tenor, settled with its lag); final maturity of knocked-in paths (V1). The
  maturity payoff callback's per-path knocked-in flag selects V0 or V1.
- `DayStates`, `StateKey` and the aged-product cache gain `knock_in_day`, set only for a knocked-in `REBASED` path
  (-1 otherwise: snowball and `ABSOLUTE` keys, caches and goldens unchanged). The aged product of such a path is
  §5's pricing copy for that knock-in day. Surface mode builds a knocked-in surface per (bucket, knock-in day)
  visited, lazily; exact mode is unchanged.
- The conformance oracle (exact mode == replay to the last bit) gains KO-reset cases (§7).

## 7. Tests and acceptance

- **Tracker rules** (`test/lifecycle_ko_reset/`), one case per rule: schedule-1 knock-out; R3 tie; knock-in → schedule
  2 with a schedule-2 knock-out on the knock-in day; a knock-in on schedule 1's last observation switches instead of
  maturing; un-knocked at schedule 1's end matures with V0 on schedule 1's tenor, equal to the engines' value that
  day (2.5 on the measured case, all four engines), settlement lag honoured; continuous knock-in tested only during
  schedule 1; `REBASED` dates follow the knock-in date; knocked-in final maturity pays V1.
- **Pricing copy:** a knocked-in `REBASED` copy prices as the same contract with an `ABSOLUTE` schedule 2 at
  knock-in + offsets, on every engine, to 1e-12; the tracker's copy agrees with the intraday twin at a matching
  checkpoint.
- **Consumers:** the manager closes a KO-reset at schedule 1's end and the next day does not raise; dynamic scenario
  and equity backtest smoke cases; replay QUAD prices a knocked-in KO-reset on `KOResetSnowballQuadEngine` (the
  4.195 vs -1.744 case as a regression).
- **Simulation:** conformance cases for an `ABSOLUTE` and a `REBASED` contract over paths that knock out on schedule
  1, mature un-knocked, knock in then knock out on schedule 2, and knock in then mature; surface mode's `REBASED`
  per-knock-in-day surfaces within the existing gate against exact mode.
- **Unchanged:** snowball and Phoenix replay goldens, simulation snowball goldens and the PDE byte book
  byte-identical; full suite `-n 6` under the RAM guard.

## 8. Out of scope

New local-vol / Heston / SLV KO-reset engines. Under those models a KO-reset is refused wherever a snowball engine
would misprice it.
