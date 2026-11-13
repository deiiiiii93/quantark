# Gamma pin risk of a short 2Y snowball: techniques and a desk SOP

000852.SH (CSI 1000), monthly KO, daily KI, hedged with IM futures and MO options, priced on
QUAD V2. Everything here is from the side of the snowball **seller**: the desk that sold the
note, is short it and hedges it. P&L is the seller's, in bp of notional, net of costs.

The study first researched how desks handle the pin at a KO fixing and near a KI barrier
(`RESEARCH.md`, written and pre-registered before any backtest), then replayed those
techniques on real closes from 2023-05-04 to 2026-11-06. Parameters were picked on events
before 2025-07-01; the verdict is read on events from 2025-07-01 on, once.

Report: `results/report.html` (offline, light and dark). Numbers: `results/backtest.json`.

## The answer

**No technique robustly beats the plain delta hedge.** Under the rule fixed before the run
(the tail must improve, and the whole life must not get worse), one row out of sixteen
eligible rows passes under the deciding clock, and it does not survive the other two clocks
or a higher option cost. With ten KO rows tested at raw p < 0.10, about one pass is expected
by chance.

What the test slice does show, under the close clock:

- **KO pin (2Y product, 43 fixings in 24 weeks).** The baseline's ES10 (mean of the worst
  10% of event windows) is -424.7 bp and its worst event -516.8 bp.
  - Booking the KO barrier 2% on the losing side (`ko_shift@0.02`) has the best futures-only
    tail: ES10 -390.5 bp (+34.2), worst -457.5 bp. The gain has the same sign under all
    three clocks and on the training slice (+130.6 bp there). It does not pass: p = 0.290,
    and the daily P&L of the unselected notes is 13% more volatile (limit 5%).
  - The MO call-spread overlay (`mo_ko@2,payoff`) is the one row that passes: ES10 -401.0 bp
    (+23.7), p = 0.056 (Holm 0.557), whole-life std ratio 0.99. It fails under `twap` (+3.3,
    p = 0.396) and under `next_open` it is worse than the baseline (ES10 -487.7 against
    -456.7, worst -912.9 against -633.1). At 1.0 index point of option cost it no longer
    passes. At desk scale only 75% of the wanted lots are filled on average.
  - Holding the survived-branch delta into the fixing (`surv_hold@1`), de-risking over three
    days (`derisk@3`) and the wide-bump delta (`secant@0.03`) are all worse than the baseline
    at a daily clock (-16.8, -18.8, -27.0 bp of ES10). The intraday finding behind them does
    not carry over to a desk that trades at the close.
- **KI pin (the same product in its last 6 months, 79 approaches in 24 weeks).** The
  baseline's ES10 is -390.7 bp, worst -468.8 bp. No row passes.
  - Lowering the booked KI barrier (`ki_shift@-0.01`) makes the tail worse (-480.3, a loss of
    89.6 bp), as it did on training. It is the conservative direction for reserving and the
    wrong direction for hedging a gap down: it keeps more long futures into the knock-in.
  - Capping the hedge near the knocked-in delta (`ki_cap@1.25`) is the worst row (-734.6)
    while its mean event P&L is the best (+343.8 against +141.0): it is a bet on a knock-in,
    paid for when the index rebounds. Its whole-life std ratio is 1.21.
  - A 5% bump delta (`secant@0.05`) has ES10 -331.2 (+59.5, p = 0.168) with no whole-life
    cost; the raised-barrier control `ki_wrong` is at -325.7. Not significant.
  - **The futures position limit is the largest KI tail effect measured.** Without the cap
    at 3.0 x notional the baseline's ES10 is -587.0 bp and its worst event -1,357.0 bp.
- **The KI pin is a short-remaining-life problem.** A fresh 75% note 3% above its barrier
  has a knock-in gap of 9 bp with 24 months left, 223 bp with 6 and 768 bp with 1
  (`results/ki_gap_probe.json`). Lifting the barrier removes the pin; a short life creates it.

## What was run

| Item | Value |
| --- | --- |
| Product (`ki75`) | 2Y snowball, KO 100% on month anniversaries 3..24, KI 75% daily, 10 mio CNY, coupon solved to PV 0 at inception |
| Seasoned book (`seas6`) | the same 75% product with 18 months elapsed and 6 left, booked on the first anchored day of each ISO week with its KI barrier 7% below spot: 168 synthetic bookings on real paths |
| Market | daily closes; flat vol = the day's far-MO ATM vol; r 1.4%; carry tail `converge_hl100`; IM and MO marks from CFFEX daily files |
| Sample | 321 priced notes (153 `ki75`, 168 `seas6`), about 83,000 QUAD V2 ladder solves |
| KO events | `ki75` fixings with the prior close within 2% of the barrier: 107 training (52 weeks), 43 test (24 weeks) |
| KI events | `seas6` approaches (runs within 5% above the barrier): 97 training (28 weeks), 79 test (24 weeks); `ki75` approaches as a transfer check only |
| Baseline | far IM contract on the 1% bump delta of the true note at every close, hedge ratio limited to [-0.25, 3.0] |
| Clocks | `close` (decides), `twap` (day VWAP proxy), `next_open` (fixing-day trades wait for the next open) |
| Options | listed MO legs at settlement; a leg may be opened only with volume >= 100 and open interest >= 500; trades cut to 10% of day volume and 1,200 lots per series side at a 500 mio desk; 0.2 index point per leg per side |
| Verdict | (a) ES10 higher, worst event not lower, week-clustered bootstrap p < 0.10; and (b) whole-life daily std <= 1.05 x baseline and mean terminal P&L >= baseline - 10 bp. Decidable with at least 20 events in 10 weeks. "Robust" = also passes (a) under the other two clocks |

Two passes. `03_pricing.py` records, per note and day, a 25-point spot ladder of price,
delta and gamma for the true note and for each hedge model (shifted barriers, the survived
branch, the knocked-in note). `04_ledger.py` replays every technique on those states and
on real IM and MO marks. Techniques never call the engine.

The test slice can only run when four hashes match the committed `results/selection.json`:
the pre-registration, the event file, the ledger-side code and the recorded inputs.

## Headline numbers

Every number in this section is checked against `results/backtest.json` or
`results/selection.json` by `test/test_snowball_2y_gamma_pin_sop_study.py`.

### KO verdict: test slice, close clock, `ki75`, 43 events in 24 weeks

ES10, worst and mean are event-window P&L in bp of notional. "Terminal - base" is the mean
terminal P&L of completed unselected notes minus the baseline's.

<!-- table: verdict close ko -->
| Technique | Role | ES10 | ES10 - base | Worst | Mean | p | Holm p | Std ratio | Terminal - base | Tail | Life | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `base` | baseline | -424.7 | +0.0 | -516.8 | -72.5 | 1.000 | n/a | 1.00 | +0.0 | fail | pass | not eligible |
| `base_noclip` | control | -424.7 | +0.0 | -516.8 | -72.5 | 1.000 | n/a | 1.00 | +0.0 | fail | pass | not eligible |
| `ko_shift@0.02` | eligible | -390.5 | +34.2 | -457.5 | -5.3 | 0.290 | 1.000 | 1.13 | -4.6 | fail | fail | no win |
| `ko_spread@0.03` | eligible | -403.6 | +21.1 | -478.9 | -28.4 | 0.319 | 1.000 | 1.07 | -8.7 | fail | fail | no win |
| `ko_bend@0.03,5` | eligible | -398.7 | +26.0 | -509.4 | -45.0 | 0.302 | 1.000 | 1.07 | -0.3 | fail | fail | no win |
| `ko_wrong` | control | -490.1 | -65.3 | -603.9 | -118.6 | 0.998 | n/a | 1.00 | -17.1 | fail | fail | not eligible |
| `secant@0.03` | eligible | -451.7 | -27.0 | -541.0 | -83.8 | 1.000 | 1.000 | 0.99 | -6.9 | fail | pass | no win |
| `secant@0.05` | candidate | -483.6 | -58.8 | -583.2 | -100.7 | 1.000 | n/a | 1.00 | -6.0 | fail | pass | not eligible |
| `surv_hold@1` | eligible | -441.5 | -16.8 | -546.2 | -65.2 | 0.877 | 1.000 | 1.04 | +4.5 | fail | pass | no win |
| `derisk@3` | eligible | -443.5 | -18.8 | -538.1 | -74.5 | 0.929 | 1.000 | 1.05 | +3.7 | fail | pass | no win |
| `freeze@1` | eligible | -422.3 | +2.4 | -527.5 | -66.5 | 0.343 | 1.000 | 1.07 | +34.3 | fail | fail | no win |
| `mo_ko@2,payoff` | eligible | -401.0 | +23.7 | -502.8 | -71.0 | 0.056 | 0.557 | 0.99 | -2.9 | pass | pass | WIN, not robust |
| `sop_futures` | eligible | -390.5 | +34.2 | -457.5 | -5.3 | 0.290 | 1.000 | 1.13 | -4.4 | fail | fail | no win |
| `sop_overlay` | eligible | -384.1 | +40.6 | -515.9 | -32.5 | 0.130 | 1.000 | 1.13 | +23.3 | fail | fail | no win |
| `band@0.02` | cost | -425.0 | -0.2 | -517.2 | -72.5 | 0.877 | n/a | 1.00 | -0.0 | fail | pass | not eligible |
| `band@0.05` | cost | -424.9 | -0.2 | -519.7 | -72.7 | 0.545 | n/a | 1.00 | +9.0 | fail | pass | not eligible |

### KI verdict: test slice, close clock, `seas6`, 79 approaches in 24 weeks

The baseline here is limit-constrained: the hedge-ratio cap at 3.0 binds on 284 days inside
these windows. `base_noclip` is the same hedge without the cap.

<!-- table: verdict close ki -->
| Technique | Role | ES10 | ES10 - base | Worst | Mean | p | Holm p | Std ratio | Terminal - base | Tail | Life | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `base` | baseline | -390.7 | +0.0 | -468.8 | 141.0 | 1.000 | n/a | 1.00 | +0.0 | fail | pass | not eligible |
| `base_noclip` | control | -587.0 | -196.3 | -1357.0 | -52.4 | 0.915 | n/a | 0.86 | +431.2 | fail | pass | not eligible |
| `ki_shift@-0.01` | eligible | -480.3 | -89.6 | -626.1 | 133.3 | 0.687 | 1.000 | 1.01 | -11.1 | fail | fail | no win |
| `ki_wrong` | control | -325.7 | +64.9 | -378.7 | 152.1 | 0.194 | n/a | 0.99 | +22.5 | fail | pass | not eligible |
| `secant@0.03` | candidate | -360.6 | +30.1 | -411.7 | 142.1 | 0.181 | n/a | 1.00 | +4.6 | fail | pass | not eligible |
| `secant@0.05` | eligible | -331.2 | +59.5 | -378.7 | 150.8 | 0.168 | 0.839 | 1.00 | +17.8 | fail | pass | no win |
| `ki_cap@1.25` | eligible | -734.6 | -343.9 | -897.8 | 343.8 | 1.000 | 1.000 | 1.21 | +86.7 | fail | fail | no win |
| `mo_ki@2,scenario` | eligible | -427.6 | -36.9 | -714.0 | 131.5 | 0.795 | 1.000 | 1.00 | +9.3 | fail | pass | no win |
| `sop_futures` | eligible | -331.2 | +59.5 | -378.7 | 150.8 | 0.168 | 0.839 | 1.01 | +39.9 | fail | pass | no win |
| `sop_overlay` | eligible | -327.7 | +63.0 | -395.9 | 141.6 | 0.123 | 0.737 | 1.00 | +44.2 | fail | pass | no win |
| `band@0.02` | cost | -390.7 | +0.0 | -468.8 | 140.9 | 0.300 | n/a | 1.00 | +3.3 | fail | pass | not eligible |
| `band@0.05` | cost | -389.3 | +1.3 | -468.9 | 141.9 | 0.154 | n/a | 1.00 | +9.6 | fail | pass | not eligible |

### The three clocks: ES10 on the test slice

<!-- table: clocks -->
| Barrier | Technique | close: ES10 | close: - base | twap: ES10 | twap: - base | next_open: ES10 | next_open: - base |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KO | `base` | -424.7 | +0.0 | -498.5 | +0.0 | -456.7 | +0.0 |
| KO | `base_noclip` | -424.7 | +0.0 | -498.5 | +0.0 | -456.7 | +0.0 |
| KO | `ko_shift@0.02` | -390.5 | +34.2 | -472.9 | +25.6 | -394.6 | +62.1 |
| KO | `ko_spread@0.03` | -403.6 | +21.1 | -477.2 | +21.2 | -417.5 | +39.2 |
| KO | `ko_bend@0.03,5` | -398.7 | +26.0 | -465.2 | +33.2 | -410.3 | +46.4 |
| KO | `ko_wrong` | -490.1 | -65.3 | -563.9 | -65.4 | -531.3 | -74.6 |
| KO | `secant@0.03` | -451.7 | -27.0 | -522.0 | -23.6 | -477.7 | -21.0 |
| KO | `surv_hold@1` | -441.5 | -16.8 | -510.1 | -11.6 | -464.3 | -7.6 |
| KO | `derisk@3` | -443.5 | -18.8 | -510.7 | -12.2 | -466.5 | -9.8 |
| KO | `freeze@1` | -422.3 | +2.4 | -483.8 | +14.7 | -445.2 | +11.5 |
| KO | `mo_ko@2,payoff` | -401.0 | +23.7 | -495.2 | +3.3 | -487.7 | -31.0 |
| KO | `sop_futures` | -390.5 | +34.2 | -472.9 | +25.6 | -394.6 | +62.1 |
| KO | `sop_overlay` | -384.1 | +40.6 | -468.7 | +29.7 | -466.6 | -9.9 |
| KI | `base` | -390.7 | +0.0 | -254.5 | +0.0 | -288.7 | +0.0 |
| KI | `base_noclip` | -587.0 | -196.3 | -139.8 | +114.7 | -367.1 | -78.4 |
| KI | `ki_shift@-0.01` | -480.3 | -89.6 | -277.0 | -22.5 | -327.3 | -38.6 |
| KI | `ki_wrong` | -325.7 | +64.9 | -257.4 | -2.9 | -305.2 | -16.5 |
| KI | `secant@0.05` | -331.2 | +59.5 | -261.4 | -6.9 | -312.6 | -23.9 |
| KI | `ki_cap@1.25` | -734.6 | -343.9 | -630.9 | -376.3 | -734.4 | -445.7 |
| KI | `mo_ki@2,scenario` | -427.6 | -36.9 | -211.8 | +42.8 | -299.1 | -10.4 |
| KI | `sop_futures` | -331.2 | +59.5 | -261.4 | -6.9 | -312.6 | -23.9 |
| KI | `sop_overlay` | -327.7 | +63.0 | -207.6 | +46.9 | -289.8 | -1.1 |

Under `twap` two overlay rows pass the KI rule (`mo_ki@2,scenario` p = 0.025,
`sop_overlay` p = 0.005). `twap` does not decide, and its fills use a trade size known only
at the close. Under `next_open` no eligible KI row beats the baseline.

### Sensitivities: ES10 on the test slice, close clock, one number changed at a time

<!-- table: sensitivities -->
| Barrier | Technique | Primary (500 mio, 0.2 pt) | Desk 100 mio | MO cost 1.0 pt |
| --- | --- | ---: | ---: | ---: |
| KO | `base` | -424.7 | -424.7 | -424.7 |
| KO | `ko_shift@0.02` | -390.5 | -390.5 | -390.5 |
| KO | `ko_spread@0.03` | -403.6 | -403.6 | -403.6 |
| KO | `ko_bend@0.03,5` | -398.7 | -398.7 | -398.7 |
| KO | `secant@0.03` | -451.7 | -451.7 | -451.7 |
| KO | `surv_hold@1` | -441.5 | -441.5 | -441.5 |
| KO | `derisk@3` | -443.5 | -443.5 | -443.5 |
| KO | `freeze@1` | -422.3 | -422.3 | -422.3 |
| KO | `mo_ko@2,payoff` | -401.0 | -392.9 | -414.6 |
| KO | `sop_futures` | -390.5 | -390.5 | -390.5 |
| KO | `sop_overlay` | -384.1 | -398.1 | -397.8 |
| KI | `base` | -390.7 | -390.7 | -390.7 |
| KI | `ki_shift@-0.01` | -480.3 | -480.3 | -480.3 |
| KI | `secant@0.05` | -331.2 | -331.2 | -331.2 |
| KI | `ki_cap@1.25` | -734.6 | -734.6 | -734.6 |
| KI | `mo_ki@2,scenario` | -427.6 | -702.5 | -435.5 |
| KI | `sop_futures` | -331.2 | -331.2 | -331.2 |
| KI | `sop_overlay` | -327.7 | -617.9 | -336.1 |

A smaller desk fills more of the KI overlay and the tail gets worse (`mo_ki@2,scenario`
-702.5 bp at 100 mio against -427.6 at 500 mio): a fuller overlay is not a better hedge here.

### What is carried into a KO fixing: test slice, close clock, means over 43 events

Futures cannot change gamma. They change the hedge carried through it. Only the overlay
changes the gamma itself.

<!-- table: gamma -->
| Technique | Net position gamma, bp per (1% move)^2 | Hedge ratio into the fixing | Largest hedge ratio in the window (mean) | Largest (max) |
| --- | ---: | ---: | ---: | ---: |
| `base` | 24.9 | 1.54 | 1.63 | 2.71 |
| `ko_shift@0.02` | 24.9 | 0.89 | 1.44 | 3.00 |
| `ko_spread@0.03` | 24.9 | 1.07 | 1.41 | 3.00 |
| `ko_bend@0.03,5` | 24.9 | 1.07 | 1.42 | 3.00 |
| `secant@0.03` | 24.9 | 1.32 | 1.43 | 2.49 |
| `surv_hold@1` | 24.9 | 1.02 | 1.45 | 2.14 |
| `derisk@3` | 24.9 | 1.02 | 1.37 | 1.93 |
| `freeze@1` | 24.9 | 1.34 | 1.44 | 2.14 |
| `mo_ko@2,payoff` | 20.5 | 1.20 | 1.48 | 2.23 |
| `sop_futures` | 24.9 | 0.89 | 1.44 | 3.00 |
| `sop_overlay` | 20.5 | 1.12 | 1.46 | 3.00 |

### Overlay fills at a 500 mio desk: all test-run days, close clock

<!-- table: overlay -->
| Technique | Days with legs | Mean fill ratio | Days below a full fill | Lowest fill ratio | Volume cuts | Series-limit cuts | Refused: no strike | Refused: not openable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mo_ki@2,scenario` | 1834 | 0.469 | 1186 | 0.006 | 1653 | 697 | 14 | 13 |
| `mo_ko@2,payoff` | 1971 | 0.755 | 1135 | 0.019 | 1430 | 680 | 13 | 14 |
| `sop_overlay` | 3809 | 0.616 | 2325 | 0.000 | 3083 | 1377 | 27 | 27 |

### Training picks (events before 2025-07-01, close clock)

Training figures, not verdicts. Baseline training ES10: KO -531.3 bp on 107 events, KI
-1,000.9 bp on 97 approaches.

<!-- table: training -->
| Barrier | Row | Pick | Training ES10 | Training ES10 - base | Beats base on training |
| --- | --- | --- | ---: | ---: | --- |
| KO | `ko_shift` | `ko_shift@0.02` | -400.7 | +130.6 | yes |
| KO | `ko_spread` | `ko_spread@0.03` | -438.5 | +92.9 | yes |
| KO | `ko_bend` | `ko_bend@0.03,5` | -479.0 | +52.3 | yes |
| KO | `secant` | `secant@0.03` | -540.4 | -9.0 | no |
| KO | `surv_hold` | `surv_hold@1` | -542.2 | -10.9 | no |
| KO | `derisk` | `derisk@3` | -560.9 | -29.6 | no |
| KO | `freeze` | `freeze@1` | -534.9 | -3.6 | no |
| KO | `mo_ko` | `mo_ko@2,payoff` | -512.3 | +19.0 | yes |
| KI | `ki_shift` | `ki_shift@-0.01` | -1099.1 | -98.2 | no |
| KI | `secant` | `secant@0.05` | -849.5 | +151.4 | yes |
| KI | `ki_cap` | `ki_cap@1.25` | -918.5 | +82.4 | yes |
| KI | `mo_ki` | `mo_ki@2,scenario` | -1022.5 | -21.6 | no |

Composites: `sop_futures` = `ko_shift@0.02` in a KO window and `secant@0.05` in a KI window;
`sop_overlay` adds `mo_ko@2,payoff` and `mo_ki@2,scenario` on top.

## The hypotheses, written before the run

| | Hypothesis | Result on the test slice (close clock) |
| --- | --- | --- |
| H1 | `surv_hold` improves the KO tail by more than any shift, spread or bend | **Rejected.** `surv_hold@1` is 16.8 bp worse than the baseline; `ko_shift@0.02` is 34.2 bp better |
| H2 | Techniques that improve the tail do not lower the whole-life std, and may raise it | **Supported for the futures rules.** The shift, spread and bend raise it by 7% to 13%. The overlay does not (0.99) |
| H3 | `mo_ko` beats every futures-only rule when the series expires close to the observation; partly unfilled at desk scale | **Not supported on the point estimate** (+23.7 against +34.2 for the shift). Partly unfilled: yes, mean fill 75%. The split by expiry gap was not computed |
| H4 | At KI, `ki_cap` and `ki_shift` move the tail in opposite directions | **Not supported.** Both are worse than the baseline on test (-343.9 and -89.6). On training they were opposite (+82.4 and -98.2) |
| H5 | The wrong-way controls are worse than the baseline | **Supported at KO** (`ko_wrong` -65.3). **Rejected at KI** (`ki_wrong` +65.0): raising the booked barrier carries less long futures into a gap down |
| H6 | Same ranking under `close` and `twap`; `next_open` widens the gaps without changing the order | **Rejected for the overlays.** The shift family leads the futures rules under all three clocks, but `mo_ko` goes from +23.7 (`close`) to -31.0 (`next_open`), and the KI order changes with the clock |
| H7 | `gap_KO` is positive on the large majority of KO-window observations | **Supported for the 2Y product** (508 of 508). On the seasoned book 107 of 194 are positive |
| H8 | The KI ranking on `seas6` is the same as on the 2Y product's own approaches | **Broadly the same order, tiny effects.** On 26 `ki75` approaches in 5 weeks every row is within 10 bp of the baseline, and `ki_cap` does nothing there |

## The SOP

Each line says what the evidence is. "Tested" means this backtest, daily closes.

| When | Action | Evidence |
| --- | --- | --- |
| Every close | Delta-hedge in the far IM on the true note's 1% bump delta. Keep a hard limit on the hedge ratio (here -0.25 to 3.0 x notional) | Tested. The limit improved the KI tail on both slices (ES10 +196 bp on test, +115 bp on training) |
| T-10, within 10% of the KO barrier | Watch list: record the redemption gap and its sign, the days left, and the hedge of the true note against the survived branch | Descriptive |
| KO window | Default: stay on the baseline. Option for a desk that weights the tail: book the KO barrier 2% on the losing side from T-10. Do not switch to the survived-branch delta at the T-1 close | Tested. The shift improved ES10 under all three clocks and both slices and raised daily std by 13%; not significant (p = 0.290). `surv_hold`, `derisk` and `secant` were worse than the baseline |
| KO overlay | Not established. If a call spread across the barrier is used, size it to the redemption gap over two strike steps, expect partial fills, and unwind it at the fixing close, not the next morning | Tested. Passes under `close` only; under `next_open` its worst event is -912.9 bp against -633.1 |
| KI watch | Flag every note with less than 12 months left that is within 10% of its KI barrier. With more life left the knock-in gap is small | Probe table by remaining life |
| KI zone, last 6 months, within 5% of the barrier | Stay on the limited baseline. Do not lower the booked barrier for hedging. Do not cap the hedge at the knocked-in delta. A wider bump (5%) is a candidate, not a rule | Tested. `ki_shift` -89.6 bp, `ki_cap` -343.9 bp, `secant@0.05` +59.5 bp (p = 0.168) |
| KI overlay | Not established. Put spreads across the barrier did not help under `close`, and a fuller fill was worse | Tested |
| Limit days | On a day the far IM closes at its price limit the hedge cannot trade at the close. Treat the position as unhedged for the gap to the next open | Four such closes in the data (2024-02-06, 2024-09-27, 2024-09-30, 2025-04-07), all before the split |
| Fixing day before 15:00, T+1 open | Rules from the two intraday studies (cut to the decided branch before the close; move to the realised branch at the open) | **Intraday, not re-tested here** |
| Reserve | A lowered KI barrier or a shifted KO barrier is still the conservative booking for the reserve. Keep it separate from the hedge | Reserve table in the report |

## Limits

- **One winner in sixteen rows is what chance gives.** The verdict uses raw p < 0.10; the
  Holm-adjusted p of the one passing row is 0.557. Adjacent weeks share market days, so p is
  optimistic.
- **The seasoned bookings are synthetic.** Real spot, vol, IM and MO marks, but bookings the
  desk did not make on those dates, and weekly bookings overlap in calendar time. The 2Y
  product's own KI approaches are two episodes (26 test approaches in 5 weeks).
- **The KI tail is a few market days.** On the test slice 4 of the 8 tail approaches are
  dated 2026-03-23. On training 7 of 10 are knock-ins on 2025-04-07, a locked limit-down day:
  the `close` clock books a sale at the limit price that could not be executed (the next
  open was 0.3% away), and the `twap` clock sells at a VWAP 2.3% above the close with a size
  known only at the close.
- **The KI baseline is limit-constrained.** The cap at 3.0 binds on 284 window days, so KI
  techniques are compared with a baseline that is already a cap rule. Without the cap the
  tail is worse but, over the whole life of the 168 seasoned bookings, the daily std is 14%
  lower and the mean terminal P&L 431.2 bp higher. This was not decomposed; the seasoned
  book also holds large KO digitals (a knocked-in note recovering to its KO level owes about
  two years of coupon), which are not measured as events here.
- **Clocks.** `close` lets the seller trade at the fixing price itself and understates a
  true pin; `next_open` overstates it; `twap` is a VWAP proxy with close-state sizes.
- **Overlay mechanics.** An overlay is refused when the barrier is more than one listed
  strike step beyond the listing. Scenario sizing mostly hits its 3x cap. The position limit
  also binds the futures hedge of option legs left after a note has ended. The 1,200-lot
  series limit is the listing-time value and was not re-verified.
- **One flat vol per day.** No smile, no vega hedge. Intraday rules were not re-tested.

## Deviations

None after unblinding: the test slice was run once, on 2026-11-14, and no defect has been
found since.

Changes made before any test-slice number existed (all in `RESEARCH.md` section 9 or the
pre-registration's amendments):

- The KI sample moved twice: lifted-barrier books (85%, 90%) were added at the desk's
  request, measured to hold no pin, and replaced by the seasoned book.
- The KI window was narrowed from 10% to 5% above the barrier.
- Review fixes to the ledger and the overlay rule (the overlay is refused far from the
  listing; a refused day keeps the last sized coverage; the data-end flatten is free; the
  position limit binds every futures target). The training selection was re-run after the
  last of these, with the same picks.
- The unblinding guard grew from three hashes to four and its code hash was widened.

## Reproduce

```bash
cd example/case_studies/snowball_2y_gamma_pin_sop
python 00_ki_gap_probe.py            # the KI gap by remaining life and barrier level
python 01_data.py                    # CFFEX board with per-series forwards
python 02_events.py                  # lifecycle scan, sample, events.json
python 03_pricing.py --plan          # then without --plan: about 1.6 h on 4 workers
python 04_ledger.py --slice train --mode all
python 04_ledger.py --slice test --mode all --desk-notional 100000000 --mo-cost-pt 1.0
python 05_report.py
```

Inputs are the shared history cache `example/case_studies/mo_volmodels/data/history/` and
the CFFEX extension of the tail-stability study; nothing is downloaded. Untracked outputs go
to `example/output/snowball_2y_gamma_pin_sop/`. The note, market and carry tail are imported
read-only from the sibling study `snowball_2y_tail_stability`.

## Open items

- H3's split of the overlay result by the gap between series expiry and observation.
- The whole-life effect of the position limit on the seasoned book, by cause (KI zone
  against KO digital).
- A desk's own execution on limit days; the daily data cannot say what was tradable.
- `quantark`: `BarrierConfig.time_shift` leaves a per-observation KO barrier list at full
  length after ageing; this study trims it itself (`gpin.pricing._trim_ko_levels`).
- This study imports four private helpers of the sibling package (`s2ts.note._is_trading`,
  `_next_trading`, `s2ts.runner._acquire_lock`, `_release_lock`).
- The run files store one clip count per note life, so the share of clipped residual-leg
  hedge rows is not reported.
- The current CFFEX position limit and fees for MO and IM.
