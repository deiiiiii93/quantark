# Gamma pin risk on a short snowball: techniques, conclusions and draft SOP

Written 2026-11-12, **before** any backtest of this study was run. Section 9 is added
afterwards and says what the backtest confirmed or rejected; sections 1–8 are not edited
after the pre-registration commit except to fix errors, which are then listed in section 9.

Position: the **seller** of a 2Y CSI 1000 (000852.SH) snowball. KO 100% of initial, observed
monthly from month 3. KI 75%, observed at every daily close. Delta is hedged with IM futures,
gamma can be mitigated with listed MO options. The hedge is adjusted once a day.

**View.** Everything here is from the side of the snowball **seller**: the desk that sold
the note, is short it, and hedges it. PV is the seller's (a liability is negative), "position
delta" is the seller's unhedged delta (negative: the seller is short the note's upside),
"hedge" is the long IM position that offsets it, and P&L is the seller's.

How to read the citations. **[raw]**: the document's text was read directly. **[fetch]**: the
page was read through a tool that returns extracted quotes, so the quote was not seen in its
surrounding page. **Inference**: reasoning or arithmetic of this study, not a source.
Everything believed but not confirmed from an opened source is in section 8 and nowhere else.

## 1. What a pin is for this note

On a KO observation the seller pays the KO cash if the close is at or above the barrier and
keeps the continuation liability if it is below. The two differ, so the seller's PV has a
step at the barrier and the hedge the model asks for just before the fixing contains a
digital term:

    hedge_true = P(survive) x hedge_surv + dP(KO)/dS x gap_KO        (redeemed branch needs no hedge)

`gap_KO` = what the seller loses if the note is redeemed rather than survives: the KO cash
paid minus the continuation liability released, at the barrier. The second term grows like
`0.4 x gap_KO / (daily vol x sqrt(days to fixing))` and is what explodes (inference; survey
arithmetic at 22% vol gives 0.37x notional at 10 days, 0.53x at 5, 0.68x at 3, 1.19x at 1 for
a 4.2% gap).

Measured on a probe with the study's engine (QUAD V2, note struck 2024-02-29, valued at the
2024-05-28 close, one day before its first KO observation, spot 0.06% above the barrier):

| | value |
| --- | ---: |
| KO cash the seller pays if it fires | 370 bp of notional |
| seller's PV of the survived branch at the barrier | +54 bp (the continuation is an asset to the seller here) |
| `gap_KO` (seller's loss on redemption against survival) | 424 bp |
| hedge the true model asks for at spot (long IM), point delta | 193% of notional |
| the same with the desk's 1% bump (the baseline of this study) | 176% |
| point delta 2% above / 2% below spot | 32% / 120% |
| hedge of the survived branch at spot (point and 1% bump) | 82% |

So one day before a pinned fixing the model asks for 176-193% of notional in futures,
depending on the bump, of which 94-111 points are the digital term; whichever way the fixing
goes, that position is wrong the moment it is known (too long by about 100 points if the note
survives, by all of it if the note is redeemed). That is the pin. The desk's 1% bump already
smooths part of it, because the bump straddles the barrier.

At the KI barrier the same shape appears with a daily observation: the seller gains when the
note knocks in (its liability falls), the hedge the model asks for rises above 1x notional as
spot falls towards the barrier and drops back after the knock-in. In gamma terms the seller
is long gamma over most of the note's life (it buys low and sells high) and turns short gamma
in the strip just above the KI barrier and just below a KO barrier before a fixing. A China sell-side description of
January 2024: "delta 可能会达到 1.2-1.5 ... 而雪球一旦发生敲入，delta 会重新回到 1" (Guojin
Securities, 2024-01-26 [C6, raw]). A 2025 practitioner article adds that the KO step is
larger for a note that has already knocked in: "已敲入合约在敲出障碍价格附近的 Delta 值变化
更为剧烈" (Li and Chen, 2025 [C7, raw]).

## 2. Techniques and the conclusion for each, before the backtest

| # | Technique | What it does | What the sources say | Conclusion carried into the backtest |
| --- | --- | --- | --- | --- |
| 1 | Constant barrier shift | Hedge the note as if the barrier were elsewhere, so the booked product is conservative and the dangerous delta sits beyond the real barrier | Standard practice for digitals and barriers [S1, S2, S4]; "there is no clear procedure for choosing an appropriate value" [S1]; expensive when sized to a delta limit (+27% to +122% in price in the one published example [S2]) | Tested (`ko_shift`, `ki_shift`). A shift moves the delta peak, it does not shrink it (probe, point deltas: a -1% KO shift cuts the hedge at spot from 193% to 122% and raises it to 209% one percent lower). Expected to help little at a daily clock |
| 2 | Digital booked as a spread | Replace the step by a ramp of width e ending at the barrier; equals the average of shifts from 0 to e | "Hedging a short digital with 1/e of the (K - e)/K call spread, so the hedge never pays less than the digital does" [S6, fetch] | Tested (`ko_spread`). Caps the digital delta at gap/e at the fixing: 2.8x notional for e = 1.5%, 1.4x for 3%. At the T-1 close it removes only 16-43% of the digital delta (inference, section 3.3) |
| 3 | Bending | Apply the shift or spread only near the date | A progressive shift is cheaper than a constant one [S4, fetch]; the delta constraint binds "short before maturity and close to the barrier" [S2, raw] | Tested (`ko_bend`, 3 and 5 days). Should cost less than a constant spread for the same tail |
| 4 | Delta-constrained payoff (face-lift) | Price the smallest dominating payoff whose delta respects a limit | Much cheaper than a shift for the same limit (+4% to +35%) [S1, S2] | **Not tested**: needs engine support, out of this study's scope |
| 5 | Delta cap | Clip the hedge at a multiple of notional | "A naive approach"; "it is not clear what the impact is on the hedge performance" [S2, raw] | Tested near KI as a cap at the knocked-in delta (`ki_cap`), the level the Jan 2024 description suggests [C6] |
| 6 | Survived-branch delta | Into a fixing, hold the delta of the note conditional on surviving it (drop the digital term) | Not named in any source opened. Nearest: hedgers "under-hedge" in the run-down to a barrier [S4]; a managed product means selling fewer shares at the lower price [S2]. It is this repo's own intraday finding | Tested (`surv_hold`, `derisk`). Expected to be the best futures-only rule at KO: it removes all of the digital delta |
| 7 | Wide-bump (secant) delta | Finite-difference delta with a bump of a few percent = a centred spread | The centred spread reproduces the digital to second order in width [S7, fetch] | Tested inside the windows only (`secant`). Symmetric, so not conservative |
| 8 | No-trade band | Rebalance only when the held delta leaves a band | Whalley-Wilmott band, valid for small costs and smooth gamma [S5, fetch] | Tested outside the windows only (`band`), as a cost lever. Not a pin technique: at a pinned fixing gamma is not smooth and the band is centred on the wrong delta (inference) |
| 9 | Listed spread overlay | Buy or sell an MO vertical spread whose payoff moves across the barrier by the size of the gap | Digital replication by a spread [S4, S6, S7]; MO grid, expiries, settlement and limits [C1, C2, C3] | Tested (`mo_ko`, `mo_ki`). The only instrument that carries the digital itself. Three frictions: 100-point strikes, third-Friday expiries, 1,200-lot limit |
| 10 | Calendar rules and reserve | Book the managed product, hold the price difference as a reserve, release at the event | Managed booking and provision [S2]; nothing published on schedules or distance-to-barrier limits | Tested as `derisk` and `freeze`; the rest is in the draft SOP as inference |
| 11 | Continuity correction (BGK) | Price a discretely monitored barrier with a continuous formula after moving the barrier by `exp(0.5826 sigma sqrt(dt))` | A pricing equivalence, not an over-hedge [S3; restated in S9, raw] | **Not applied.** The engine monitors the daily close explicitly; applying it would double count |

## 3. Findings by area

### 3.1 Direction of a shift

Four worked cases in the sources (short up-and-out call [S1], short down-and-out put [S2],
long down-and-in put [S4], short digital call [S6]) fit one rule: **move the booked level in
the direction in which crossing it hurts the desk** (inference; no source states it this
way). It is the same as the conservativeness condition in [S2]: the managed product must be
worth more than the original, for a short position, in every market state.

- **KI barrier: shift down.** The desk gains when the note knocks in, so the event is booked
  later. The sign cannot realistically reverse: knocking in adds a put 25% in the money.
- **KO barrier: shift down while `gap_KO > 0`, up if it is negative.** The probe measured a
  positive gap, but the sign is not structural. It is large when continuation is unattractive
  (note already knocked in, or a deep futures discount) and shrinks for a not-knocked-in note
  with a low discount and a large coupon still to accrue. The backtest therefore measures the
  sign every day and counts how often it is not positive.

A practitioner thread makes the same point about what a shift is for: it is "really a PNL
cushion", realised as a windfall when the hedge is liquidated near the shifted level, and
"barrier shifting doesn't work for pure digitals" because the payoff at the shifted level
cannot be replicated (Wilmott forum [S10, fetch]).

### 3.2 Size of a shift

No source gives a procedure. The drivers named are the liquidity of the underlying, the size
of the digital and implied vol at the barrier [S4]; or the shift is solved from a maximum
delta "chosen a-priori according to the liquidity of the underlying ... and the notional of
the trade" [S2]. The one published cost table [S2, raw, their Table 2; short down-and-out
put, spot 76, barrier 62]:

| delta limit | constant shift | price with the shift | price with the delta-constrained payoff |
| --- | ---: | ---: | ---: |
| none | 0% | 6.51 | 6.51 |
| 8 | 3.5% | 8.28 | 6.79 |
| 4 | 11.3% | 12.16 | 7.19 |
| 2 | 16.1% | 14.47 | 8.76 |

The same paper's daily-hedge backtest is the closest published analogue to this study: with
the managed product the 5% VaR was unchanged (-2.2 against -2.2) while the 0.1% expected
shortfall went from -23.9 to -14.6. The gain is entirely in the tail.

Natural units here (inference): one to two daily standard deviations, 1.3-2.6% at 20% vol.
The MO strike step (100 points, 1.5% of spot) is in the same range, so a booked spread can be
matched by a listed one.

### 3.3 Why a few-percent spread is weak at a daily clock

Inference (Black-Scholes digital, 22% vol, spot at the barrier, one day before the fixing):
booking the digital as a spread of width e multiplies the digital delta by
`[N(e/sd) - 0.5] / (0.399 e/sd)`. That is 0.84 for e = 1.5% and 0.57 for 3%. So a 1.5-3%
spread removes 16-43% of the digital delta at the T-1 close, where the survived-branch delta
removes all of it. The shift is a tool for a hedger who trades up to the event; with one
trade a day what matters is the delta carried across the fixing.

### 3.4 Futures-only rules

- A cap alone leaves the price unchanged, "which makes it also difficult to decide how much
  should be provisioned for the gap risk", and cannot help when many barriers are hit together
  [S2, raw]. January 2024 was that case.
- The survived-branch rule is right in the survived branch by construction; in the redeemed
  branch the desk still holds `delta_surv` of futures after an up-move, which offsets part of
  the gap it pays (inference). It leaves the gap itself unhedged.
- **Two rules contradict each other at KI.** The sell-side reading of January 2024 [C6] and
  the gap-risk literature [S2] point to holding less before the knock-in (cap at the
  knocked-in delta, so there is no forced sale into a crowded close). This repo's intraday
  study found the opposite for the fixing itself: the not-knocked-in branch was the desk's
  losing tail, and the best futures-only holding was that branch's high delta. Both are in
  the grid (`ki_cap` against `ki_shift`); the evidence is two episodes.

### 3.5 Listed overlays

- Sign: with `gap_KO > 0` the seller needs a payoff that rises by gap x notional as the index
  rises through the barrier: long a call spread, or by put-call parity short the put spread on
  the same strikes.
- MO terms, read from CFFEX [C1, C2, raw]: multiplier 100; strikes every 100 points for the
  front three months and 200 for the three quarterly months between 5,000 and 10,000; strikes
  listed within +/-10% of the previous close; expiry on the third Friday; final settlement on
  "最后交易日标的指数最后 2 小时的算术平均价", not the close.
- Expiry mismatch (inference): a spread behaves as a digital only when it expires within one
  or two trading days of the observation. Further out it is mainly a delta-and-gamma position,
  and paying the gap at +/-1 daily sigma needs several times more spreads, which over-hedges
  away from the barrier.
- Size: lots = gap x notional / (strike step x 100). For a 4.2% gap and a 100-point step that
  is 419 spreads per CNY 100 mio. The client position limit reported at listing (1,200 lots
  per month per side [C3, fetch]) covers about CNY 286 mio of notes per expiry.
- Liquidity, measured on four sample days from the CFFEX daily files (this study): the front
  series trades 4,500-29,000 lots a day in its busiest strikes, the second series 300-2,700,
  later series mostly under 500. The overlay is realistically a front-month trade.

### 3.6 Calendar and controls

Published guidance is thin: book the managed product and carry the price difference as a
provision [S2]; disperse barrier levels, observation dates and notional so that hedging needs
do not cluster ("对敲入 / 敲出障碍价格、到期时间、挂钩标的及名义本金等集中度实行分散化限额
管理"), and smooth theoretical greeks near barriers "以避免短期巨额调仓" [C7, raw]. No source
opened gives a de-risking schedule, a distance-to-barrier limit or a desk convention for
shift sizes in China. The schedule in section 5 is inference.

### 3.7 January-February 2024

From the Guojin report [C6, raw]: knock-ins began around 2024-01-17 (CSI 1000 near 5,555);
the futures discount deepened "随着收盘时间的临近" on knock-in days; by 2024-01-23 IM2403
traded at an 18% annualised discount against 6-8% the year before; hedgers reduce futures at
knock-in and close them at maturity, both of which "给股指期货带来脉冲式贴水加深". Reading
(inference): the excess delta above 1x is sold at or just after the KI close by every desk
with a barrier in the same neighbourhood, so the loss shows up in the basis as much as in the
gap. A backtest on real IM closes contains that basis move; one on spot would not.

## 4. Hypotheses the backtest tests

Fixed now, so the results cannot be read into them afterwards.

- H1. At the daily clock, `surv_hold` improves the KO event tail by more than any shift,
  spread or bend of up to 3%.
- H2. The techniques that improve the tail (ES10 of event-window P&L) do not lower the
  whole-life daily standard deviation, and may raise it.
- H3. `mo_ko` improves the KO tail by more than any futures-only rule when the series expires
  within two trading days of the observation, and the advantage shrinks with the expiry gap.
  At a 500 mio desk book the volume screen and the 1,200-lot limit leave it partly unfilled.
- H4. At KI, `ki_cap` and `ki_shift` move the tail in opposite directions. No prior on which
  is better.
- H5. The wrong-way controls (`ko_wrong`, `ki_wrong`) are worse than the baseline on the tail.
- H6. The ranking is the same under `close` and `twap` fills; `next_open` widens the gaps
  between techniques without changing their order.
- H7. `gap_KO` is positive on the large majority of KO-window days.
- H8. The ranking of the KI rules on the seasoned book (`seas6`) is the same as on the 2Y
  product's own approaches (`ki75`).

## 5. Draft SOP (v0)

State variables per note, computed at every close: `n` = trading days to the next KO
observation; `z = ln(S/B) / (daily vol x sqrt(n))`; `gap_KO x notional` = the cash at stake;
the hedges `delta_true`, `delta_surv`, `delta_kin` (long IM, in units of notional).

| When | Condition | Action | Evidence |
| --- | --- | --- | --- |
| Every close | always | Delta-neutral in the far IM on `delta_true`. Outside all windows a no-trade band may be used | baseline; band tested |
| T-10 | next KO within 10 days and within 10% of the barrier | Watch list. Record `gap_KO`, its sign, `z`, `delta_true / delta_surv` | descriptive |
| T-5 to T-2 | `abs(z) <= 2` | Switch the futures hedge to the bent spread delta; decide the MO overlay (front series only, strike at or below the barrier) and book the reserve | tested: `ko_bend`, `mo_ko` |
| T-1 close | `abs(z) <= 2` | Hold `delta_surv`, not `delta_true` | tested: `surv_hold`, `derisk` |
| T before 15:00 | outcome decided (spot clear of the barrier by more than a pin band) | KO decided: cut futures to zero before the close. Survival decided: hold `delta_surv` | **intraday, not re-tested here**; from the two intraday studies |
| T before 15:00 | outcome not decided | Hold `delta_surv` through the fixing | **intraday, not re-tested here** |
| T+1 open | always | Move futures to the realised branch's delta, unwind the overlay, release the reserve | tested as the `next_open` clock |
| KI watch | not knocked in, within 10% above the KI barrier | Watch list. Record `delta_true`, `delta_kin` | descriptive |
| KI zone | within 5% above the KI barrier | Futures rule per the backtest (`ki_cap` or `ki_shift`); overlay only if the front series lists strikes at the barrier | tested: `ki_cap`, `ki_shift`, `mo_ki` |
| Knock-in day | close at or below the barrier | Move to `delta_kin` | tested under all three clocks |
| Escalate | `abs(z) < 1` with `n <= 2`; or `gap_KO x notional` above a cash limit; or `delta_true / delta_surv > 1.5` | Desk head decides overlay size and whether to pre-trade | inference; thresholds not tested |
| Book level | always | Limits on the concentration of barrier levels, observation dates and notional | [C7]; not tested |

## 6. What is carried into the backtest

The technique grid, windows, thresholds, the training / test split and the verdict rule are
in `results/preregistration.json`, committed with this file. Not carried: the
delta-constrained payoff (engine work), the intraday rows of the SOP, and the escalation
thresholds.

Three things about how the evidence will be read, also fixed now:

- **Both barriers get a verdict, on different books.** KO: the 2Y product (`ki75`). KI: real
  history holds knock-ins of the 2Y product in two episodes only (January-February 2024 and
  October 2026), about five test weeks, which cannot carry a test, and lifting the barrier
  does not help: the seller's gain on a knock-in (the gap between the not-knocked-in and the
  knocked-in value) is almost nil while the note has a long life left, because a note near
  its barrier knocks in later anyway (section 9, 2026-11-12). The knock-in pin is a property
  of a short remaining life. So the KI sample is the same 75% product in its last six months:
  `seas6` = 18 months elapsed, 6 left, coupon accrued over the full 24 months, not knocked in,
  booked on the first anchored chain day of each ISO week, with its knock-in barrier 7% below
  the booking-day spot. Every booking is priced and none is picked for having a pin. A
  pricing-free scan gives 168 bookings with 97 training approaches in 28 weeks and 79 test
  approaches in 24 weeks. KI rules are picked on the `seas6` training approaches and tested on
  its test approaches, then shown on the `ki75` test approaches as a transfer check without a
  verdict (five weeks of one episode, and a small gap because those notes have 15 to 21 months
  to run). The report states both limits. The `seas6` bookings are notes the desk did not
  write on those dates and weekly bookings overlap in calendar time.
- **Three execution clocks.** `close` (fills at the close, the proxy for a 14:55 day-end
  order) decides. `twap` (fills at the day VWAP) and `next_open` (on a fixing day the
  adjustment waits for the next IM open) are reported beside it.
- **The KO shift direction is measured, not assumed.** It follows the sign of `gap_KO` on
  the first day an observation comes within 10 trading days.

## 7. Costs and contract facts used

| Item | Value used | Source |
| --- | --- | --- |
| IM multiplier | 200 CNY per point | [C4, raw] |
| IM cost | 1 bp of traded notional (fees were 0.23 bp at listing; the rest is slippage) | [C3, fetch]; allowance is this study's |
| MO multiplier, tick | 100 CNY per point, 0.2 point | [C1, raw] |
| MO cost | 0.2 point per leg per side, sensitivity 1.0 point | this study |
| MO position limit | 1,200 lots per month per side | [C3, fetch], listing-time value |
| MO admissibility | day volume >= 100 lots, open interest >= 500 lots, trade <= 10% of day volume | this study |
| Fills | close / day VWAP (turnover / volume / multiplier) / next open | CFFEX daily files |

## 8. Not verified

Nothing below was confirmed from a document that was opened.

1. Taleb, *Dynamic Hedging* (1997); Bouzoubaa and Osseiran, *Exotic Options and Hybrids*
   (2010); De Weert, *Exotic Options Trading* (2008). Books, not opened. De Weert's table of
   contents lists "Factors Influencing the Magnitude of the Barrier Shift" and "Delta Impact
   of a Barrier Shift"; the direction and the factors used here come from [S2] and [S4].
2. Derman, Ergener and Kani (1995) and Carr and Chou (1997) on static replication of
   barriers; Zakamouline (2006) and the original Whalley and Wilmott (1997). Not opened; the
   band formula is as restated in [S5].
3. Broadie, Glasserman and Kou (1997), full text. The abstract [S3] and the restatement in
   [S9] were read.
4. Cui, Li and Zhang, "Pricing and hedging autocallable products by Markov chain
   approximation", *Review of Derivatives Research* 27(3), 2024. Paywalled. Search summaries
   disagree on whether it sizes a barrier shift by minimising the CVaR of the hedging loss.
   The most relevant paper not read.
5. Guillaume, "Autocallable Structured Products", *Journal of Derivatives* 22(3), 2015; Chen,
   Fu, Hull et al., *Journal of Investment Management* 22(4), 2024; Koster and Rehmet; Alm,
   Harrach, Harrach and Keller (2013); Glasserman and Staum (2001). Not opened.
6. Whether the listing-time numbers still apply: MO 1,200-lot limit and CNY 15 fee, IM fees
   and margins.
7. January-February 2024 figures seen only in search snippets (knock-in bands 4,600-5,500,
   amounts knocked in, intraday discount levels).
8. That the 15:00 index close comes from a 14:57-15:00 auction on both exchanges (used only
   for the intraday rows of the SOP).
9. The 22% vol and the Black-Scholes digital behind the arithmetic in sections 1 and 3.3.
   They give orders of magnitude; the backtest measures the real thing.
10. Day VWAP as a stand-in for a TWAP fill. No intraday bars cover the history.

## 9. Corrections, and what the backtest confirmed or rejected

Corrections to sections 1-8 after the pre-registration commit:

- 2026-11-12, before any ledger run. Section 1's probe table quoted the engine's point delta
  (193%) as the hedge; the desk's 1% bump delta, which is this study's baseline, is 176% on
  that date. The gap was quoted as 419 bp from the survived-branch value at spot; at the
  barrier it is 424 bp. Both rows now show the corrected numbers. No grid point, window or
  threshold depended on either number, and `results/preregistration.json` is unchanged.

- 2026-11-12, before any ledger run. Amendment to the pre-registration, at the desk's
  request to make the product easier to knock in: two books with the knock-in barrier at
  85% and 90% were added beside the 75% product, the KI verdict was moved to the 90% book
  (it had been "descriptive only" on the 75% product), the KI window was narrowed from 10%
  to 5% above the barrier, and H8 was added. Counts behind it, from a pricing-free scan:
  75%: 102 training approaches in 3 weeks and 65 test approaches in 5 weeks; 85%: 235 in
  23 weeks and 128 in 9; 90%: 334 in 32 weeks and 182 in 14. Those counts used the 10%
  window. With the 5% window the scan on the anchored inceptions (`results/events.json`,
  `books.<book>.restricted`) gives: 75%: 106 training approaches in 4 weeks and 74 test
  approaches in 5 weeks; 85%: 309 in 21 weeks and 138 in 9; 90%: 404 in 31 weeks and 196
  in 13. The lifted books are sampled 3 notes per ISO week of approach date (not 6 per month, which
  left the 90% test slice at exactly the 10-week minimum), so no week of the universe is dropped: the
  sampled 90% notes carry 121 training and 41 test approaches in 31 and 13 weeks.

- 2026-11-12, before any ledger run. Second amendment to the pre-registration: the lifted
  books `ki85` and `ki90` are dropped and the seasoned book `seas6` replaces them. A probe
  (`00_ki_gap_probe.py`, `results/ki_gap_probe.json`; a fresh note with its spot 3% above the
  knock-in barrier, monthly KO from month 1, coupon 20%, flat vol 25%, q 5%) measured, per
  barrier and remaining life, the gap = the not-knocked-in engine price minus the knocked-in
  engine price (what the seller gains when the note knocks in, bp of notional), and the hedge
  step = the hedge of the not-knocked-in note minus that of the knocked-in note (the long futures
  the seller must sell when the note knocks in, % of notional):

  | KI barrier | 1 month left | 3 | 6 | 12 | 18 | 24 |
  | --- | --- | --- | --- | --- | --- | --- |
  | 75% | 768.2 bp / 177.8 | 384.2 / 96.7 | 222.7 / 57.0 | 81.0 / 20.8 | 27.1 / 6.9 | 8.6 / 2.2 |
  | 85% | 354.3 / 77.9 | 131.5 / 31.3 | 34.5 / 8.2 | 1.9 / 0.5 | 0.1 / 0.0 | 0.0 / 0.0 |
  | 90% | 168.0 / 35.1 | 31.7 / 7.0 | 2.3 / 0.5 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |

  (each cell: gap in bp of notional / hedge step in % of notional). The knock-in pin is a seasoned-note problem: lifting the barrier on a long-dated note
  removes it, a short remaining life creates it. With a long life left, a note near its barrier
  knocks in later anyway, so being knocked in today is worth nothing to the seller, and a higher
  barrier makes that more certain; with three months left the 75% barrier carries a gap of 384
  bp and a hedge step of 97% of notional, and with one month left 768 bp and 178%. The books
  lifted to 85% and 90% were built and measured before this probe; they are removed. Nothing in
  the technique grid, windows or thresholds changed.

Results per hypothesis are added here when the backtest has run.

## References

Read in raw text:

- [C1] 中国金融期货交易所, 中证1000股指期权 product page. http://www.cffex.com.cn/zz1000gzqq/
- [C2] 中国金融期货交易所, 《股指期权合约交易细则》, revised 2022-12-14. http://www.cffex.com.cn/u/cms/www/202212/141851569gwa.pdf
- [C4] 中国金融期货交易所, 中证1000股指期货 product page. http://www.cffex.com.cn/zz1000/
- [C5] 中国金融期货交易所, 《中证1000股指期货合约交易细则》, 2022-07-18. http://www.cffex.com.cn/u/cms/www/202207/182020307zry.pdf
- [C6] 国金证券, 私募基金投资策略报告, 2024-01-26. https://pdf.dfcfw.com/pdf/H301_AP202401291619251910_1.pdf
- [C7] 李滨, 陈春秀, "复杂结构场外衍生品对冲交易风险特征与多维管控", 《金融市场研究》 2025.11, Vol. 162. https://www.nafmii.org.cn/yj/jrscyj/qk/2025/202511/202511/P020251114515007862775.pdf
- [S1] Schmock, U., Shreve, S. E., Wystup, U., "Dealing with Dangerous Digitals", 2001. https://fam.tuwien.ac.at/~schmock/preprints/Dangerous_Digitals.pdf
- [S2] Koster, F., Oeltz, D., "A Generic Approach to Manage Gap Risk", RIVACON GmbH, undated. https://rivacon.com/wp-content/uploads/a-generic-approach-to-manage-gap-risk.pdf
- [S9] Howison, S., Steinberg, M., "A matched asymptotic expansions approach to continuity corrections for discretely sampled options. Part 1: barrier options". https://people.maths.ox.ac.uk/howison/papers/discbarrier.pdf

Read through an extracting fetch tool:

- [C3] 华尔街见闻, report of the CFFEX listing notice for CSI 1000 futures and options, 2022-07-18. https://wallstreetcn.com/articles/3665090
- [S3] Broadie, M., Glasserman, P., Kou, S., "A Continuity Correction for Discrete Barrier Options", *Mathematical Finance* 7(4), 1997 (abstract). https://econpapers.repec.org/RePEc:bla:mathfi:v:7:y:1997:i:4:p:325-349
- [S4] Pandey, V., "Regular and Reverse Barriers", 2023 (partial page). https://www.pandemonium.sg/resource/regular-and-reverse-barriers
- [S5] Arzel, J., Lehdili, N., arXiv:2603.29994, 2026. https://arxiv.org/abs/2603.29994
- [S6] Tradecraft, "Call-spread over-hedge", glossary entry (partly paywalled). https://tradecraftlearning.com/glossary/call-spread-overhedge/
- [S7] options.wiki, "Exotics". https://options.wiki/exotics/
- [S8] Laevitas for Paradigm, "Exotic Options Part 3 - The Greeks", 2023-02-09. https://paradigm.co/blog/exotic-options-part-3-the-greeks
- [S10] Wilmott forum thread on barrier shifts. https://forum.wilmott.com/viewtopic.php?p=345227

Internal: `example/case_studies/intraday_pin_risk/` and `example/case_studies/intraday_ko_pin_risk/`
(the two intraday studies); the literature survey behind this file is kept with the run
records, not in the repository.
