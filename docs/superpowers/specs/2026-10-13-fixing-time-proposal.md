# Fixing-time instants in runs: proposal (I13)

Status: proposal only, no production code. Follow-up queue item I13 (spec `.ship/2026-10-13-followup-queue/spec.md`).
It is the follow-up named by W6 D6 ("a fixing-time term on the records", run-fixings design §10) and by the W11 ruling
(user, 2026-10-10): a same-day dated KI record is past for a later intraday valuation, as KO is, and "true fixing times"
come in this follow-up. Every file:line below was read in this worktree. The numbers come from a probe on this tree
(session scratchpad `t15/probe_yf.py`, not tracked).

## 1. Problem

Who hits it:
- an equity backtest configured with `frequency="H"`, which is a valid config value
  (`quantark/backtest/equity/config.py:52, 83-85`);
- any replay / dynamic-scenario / FX backtest path whose instants carry a time of day (for example closes stamped
  15:00).

If such a run holds a run-fixings product (Asian with records, equity range accrual, FX range accrual and its
variants, FX TARN/TARF), it is REFUSED. If it holds a DCN or an autocallable, it is NOT refused: the trackers drop the
time of day and decide each schedule date at the first instant of that day, whatever the time.

The underlying cause is that the day-level clock has no notion of a time of day. Measured on
`calculate_year_fraction(…, CALENDAR_DAYS)`, which uses `timedelta.days` (`quantark/util/calendar/day_counter.py:93-94`)
and so floors:

| from | to | year fraction |
|---|---|---|
| D 15:00 | D+1 00:00 (tomorrow's dated record) | **0.0** (resolves to "now", live at today's spot) |
| D 15:00 | D+2 00:00 | 1/365, one day short of the 2/365 a midnight valuation gets |
| D 00:00 | D 15:00 | 0.0 |

Meanwhile `datetime(D) < datetime(D, 15:00)` is True. A record dated D is "strictly before" a 15:00 valuation on D,
even though its year fraction from D 00:00 is 0.

So a time-of-day instant does three wrong things:
- it puts tomorrow's fixing at "now";
- it puts today's fixing in the past;
- it shortens every remaining date-based time by one day.

## 2. Current behaviour

### 2.1 Run fixings refuse (W6 R8 / D6)

- `require_date_instant` (`quantark/util/run_fixings.py:69-77`) raises `ValidationError` when a run instant is not
  midnight, with `# TODO(run-fixings): intraday fixing instants — a fixing-time term on the records (spec §10
  follow-up)` (`:73`).
- It is called from `step_fixings` (`:120-121`), which `ProductReplay` uses (`backtest/replay/product_replay.py:38`,
  `:241`), and from `RunFixingRecorder.advance` (`:243-244`).
- `RunFixingRecorder.advance` is the recorder that runs on every step of:
  - the equity backtest (`backtest/equity/engine.py:23, 159-264`);
  - the FX backtest (`backtest/fx/engine.py:25, 55-74`);
  - the dynamic scenario (`dynamicscenario/engine.py:49, 156-193`);
  - the FX dynamic scenario (`dynamicscenario/fx/engine.py:42, 102-119`).
- PnL explain cannot get there: snapshots must be midnight (`pnlexplain/equity/snapshot.py:54-55`), and the recorder
  normalises trade timestamps (`pnlexplain/equity/recorder.py:42-43`). Its run-fixings step (`explain.py:72-76`) only
  ever sees dates.
- Design: `docs/superpowers/specs/2026-10-09-run-fixings-design.md:263-268` (R8), `:517-519` (D6) and `:526-527` (§10). The moved-values
  table there lists an `H`-frequency backtest holding a covered product as "runs → `ValidationError`".
- Participants (`has_run_fixings`):
  - `AsianOption` (`asset/equity/product/option/asian_option.py:493`);
  - `RangeAccrualOption` (`range_accrual_option.py:663`);
  - `FxRangeAccrualOption` (`asset/fx/product/option/fx_range_accrual_option.py:436`);
  - TARN/TARF (`target_redemption_base.py:613`).
  Their records (`AsianObservationRecord` `asian_option.py:33`, `RangeAccrualObservationRecord`
  `range_accrual_config.py:27`, `FxRangeAccrualObservationRecord` `fx_range_accrual_config.py:26`, TARN
  `fixing_dates`) have no time-of-day field.

### 2.2 Trackers normalise silently

- DCN: `_day = pd.Timestamp(value).normalize()` (`asset/equity/lifecycle/dcn.py:47-48`). In `observe`
  (`:162-199`), an instant on day D decides row D at THAT instant's spot (`:192-193`) and stores `_last = (D, spot)`.
  A later instant on the same day skips row D (`t <= lower`, `:176`).
- In an hourly run, the DCN therefore decides its coupon / KO / KI at the FIRST hourly spot of the day, not at the
  close.
- The autocallable tracker's default date resolver is also `normalize()`
  (`asset/equity/lifecycle/autocallable.py:75-76`). Its per-instant decision order was not traced for this proposal.

### 2.3 Valuation side (W11, W8 Q4)

- `active_record_indices` (`asset/equity/product/option/observation_schedule.py:278-299`) drops a dated record when
  `record.observation_date < pricing_env.valuation_date`. At 10:00 on a KI observation date, that day's KI record is
  therefore already "past" (W11 ruling, the same as KO). Its outcome must come from the carried state, even though
  the close has not printed.
- `time_to_exercise` (`base_equity_option.py:720-741`) gives 0.0 only on exact equality with `exercise_date`. A 15:00
  valuation on the exercise date (booked at midnight) raises (`:730`), which is the W8 Q4 same-instant rule.
  `maturity_reached` (`:744-`) is the run-side test and does not refuse.

### 2.4 Prior art already in the repo

- **Per-record instant.** `ObservationRecord.observation_timestamp` / `settlement_timestamp`: tz-aware, on the same
  local date as `observation_date`. The docstring says "Intraday mode uses it … day-level resolution ignores it"
  (`observation_schedule.py:67-72`, validation `:185-195`). It exists on the autocallable/barrier record type only.
- **Instant resolution.** `intraday/events.py:193-210` `_instant` resolves in this order:
  1. an explicit record timestamp;
  2. the request's `fixing_time_of_day` (`intraday/request.py:39`) on the record date;
  3. the session calendar's close (`TradingSessionCalendar.close_at`, `intraday/session.py:155-156`).
  A non-trading record date is refused.
- **Before/after at the exact instant.** `EventPhase.BEFORE/AFTER` (`intraday/events.py:37-49`) on the intraday
  request.
- **Gap rule.** W6 R4 / D5: a fixing no instant fell on takes the previous instant's spot, `gap=True`
  (`run_fixings.py:9-14`; design `:514-516`). This run's A7'(c) applies the same rule to DCN schedule dates the run
  calendar never visits (`dcn.py:12-18, 186-193`).
- **First live instant.** W6 R3 / Ruling 4: a record live (t = 0) at an instant is recorded at the spot of the FIRST
  instant at which it was live (`run_fixings.py:8-13`).

## 3. Options

All options must keep every midnight-instant run bitwise (replay goldens gate this).

### Option A: a fixing-time term on every record type

Add `fixing_time` (tz-aware or local `time`) to the three run-fixings record types and the TARN dated schedule,
mirroring `ObservationRecord.observation_timestamp`. The recorder and trackers compare full instants. Engines and
day counters stay date-level.

- Cost: high. 4 product families × (validation, `open_fixings`, `with_fixings_recorded`, `time_shift` /
  `with_time_elapsed`), the recorder, `ProductReplay`, both trackers, plus booking adapters that must supply the time.
- Risk: every booking has to state a fixing time, or a default has to come from somewhere, which brings back option B
  anyway.
- What moves: nothing at midnight. Time-stamped runs: refusal → value (participants), and first-instant → fixing-time
  decision (trackers).

### Option B: phase mapping at the run boundary (recommended)

A run declares where fixings print: one fixing time of day per underlying/market, defaulting to the run calendar's
close. Precedence follows the intraday resolver: record `observation_timestamp` > contract fixing time if one
exists > the run's fixing time. Each run instant `τ` on day D maps to `(D, phase)`, with phase = BEFORE if
`τ < fixing time(D)`, else AFTER. Then:

1. **Day-level pricing sees the date.** Engines get `valuation_date = D` at midnight, so the day counter is exact and
   the 1/365 loss in section 1 disappears. Within a day the time value is constant, which is a documented
   limitation. True intraday time value is `quantark.intraday`'s job.
2. **Fixings and decisions happen at the phase change.** D's records stay live (t = 0, priced on the current spot, W6
   D2) while BEFORE. They are recorded/decided at the spot of the first instant at or after the fixing time (R3's
   "first live instant", now ordered within the day). If no instant of D reaches the fixing time, the record is
   decided at the last instant of D (R4 gap: the previous instant's spot) with `gap=True`.
3. **Trackers use the same mapping**: DCN `observe` and the autocallable tracker decide day D on its AFTER instant
   instead of on the first instant of D. `require_date_instant` is replaced by the mapping. The `TODO(run-fixings)`
   goes.
4. **Same-instant rules use the mapped date.** A 15:00 instant on the exercise date maps to `(T, AFTER)`, so the run
   settles via `maturity_reached`, and engines still see `valuation_date = T` (W8 Q4 is kept for direct engine
   calls). For W11, a BEFORE instant on a KI date keeps the day's KI record live. This reverses the W11 rule inside
   runs only (Q5).

- Cost: medium.
  - One mapping module in `quantark/util` (no `quantark.intraday` import; it reuses `close_at` semantics through a
    small calendar protocol).
  - The recorder and `step_fixings` call sites, the two trackers, and the run configs gain a fixing-time setting.
  - No product class changes. The record types get an optional timestamp only if Q2 asks for per-contract times.
- Risk: in a run that stamps only one instant per day after the close (e.g. 15:00 = close), every day is AFTER, which
  is equivalent to today's midnight runs. That is good, but such runs move from refused/first-instant to valued. A
  run with only pre-close instants never reaches AFTER and decides every fixing by the gap rule at the day's last
  instant; this is disclosed through `gap=True`.
- What moves: midnight runs nothing, bitwise. Time-stamped runs holding participants go from raise to value. DCN and
  autocallable decisions in multi-instant days move from the first instant's spot to the fixing-time spot. No golden
  is time-stamped (replay goldens run on dates), and certificates price single dates.

### Option C: make the current policy uniform, no feature

Either the trackers refuse time-of-day instants as well (R8 everywhere), or everything normalises to the date and
decides at the day's LAST instant (the close-equivalent).

- Cost: low (two trackers).
- Risk: the refusal variant blocks hourly hedging backtests for DCNs and autocallables that run today, wrongly. The
  normalise variant still prices with the one-day-short clock unless pricing is also mapped to the date, which is
  most of option B.
- What moves: DCN / autocallable hourly runs go from first-instant decisions to a raise (variant 1) or to
  last-instant decisions (variant 2).

## 4. Recommendation

**Option B**, plus one interim fix that can land on its own: the trackers' first-instant decision (2.2) is a silent
semantic error, not a missing feature. Until B lands it should either refuse like R8 or decide at the day's last
instant (Q6).

Why B:
- It reuses three signed rules unchanged: R3 first live instant, R4/A7'(c) gap at the previous instant, and D2
  t = 0 live.
- It reuses the intraday resolver's precedence, which is already in production for autocallables.
- It fixes the day-counter floor at the boundary instead of in 4000+ date-arithmetic sites.
- It keeps every midnight run bitwise.
- Option A's per-record term is only worth adding when a contract's fixing time differs from its market's close (an
  FX 14:00 ECB fix, for example). B's precedence leaves room for it without redesign.

## 5. Open questions for the user

1. **Scope.** Which runs need time-stamped instants: the `H`-frequency equity backtest, replay with stamped closes,
   dynamic scenarios, the FX backtests? (It sets which consumers B touches first.)
2. **Fixing-time source.** One fixing time per run/underlying (default: the run calendar's close), with a record's
   `observation_timestamp` overriding it. Or also a per-contract fixing time on Asian / range accrual / TARN (option
   A's term)?
3. **Exact tie.** An instant exactly AT the fixing time counts as AFTER (fixing recorded at that instant's spot)?
   Recommended yes.
4. **Day with no instant at or after the fixing time.** Decide the day's records at the day's last instant with
   `gap=True` (R4 analogue)? Or at the next day's first instant (R3 "first live instant")?
5. **W11 inside runs.** A BEFORE instant on a KI/KO observation date keeps that day's dated record live (decided at
   the fixing time), which reverses the W11 rule for run instants only. Direct engine calls with a time-stamped env
   keep W11. Approve?
6. **Interim tracker fix.** Before B, should the DCN / autocallable trackers refuse time-of-day instants (as R8) or
   decide at the day's last instant? (Today: the first instant's spot, silently.)
7. **Pricing clock within a day.** Accept day-level pricing (`valuation_date` = the date at midnight for every instant
   of the day, time value constant intraday), leaving intraday time value to `quantark.intraday`? Yes/no.
