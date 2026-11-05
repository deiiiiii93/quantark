# 2Y snowball: flat q vs CarryCurve

How much does a single flat dividend yield misprice and mis-hedge a 2Y CSI 1000
(000852.SH) snowball, against the term structure the IM futures + MO options
chain actually quotes on 2026-11-05?

The chain quotes `q` only out to the MO2709 expiry (T = 0.866Y); the note lives
to 2.005Y, so 57% of its life is unquoted and the curve there is whatever the
long-end convention says. The study therefore runs every comparison under three
tail conventions.

## Setup

| | |
| --- | --- |
| Trade | 2Y standard snowball, notional 10 mio CNY, strike = initial = spot 6650.1383; KO 100% monthly from month 3 (22 SSE dates, last = maturity 2028-11-06); KI 75% discrete daily (486 SSE days); principal excluded |
| Coupon | 31.60% p.a. (= KO coupon = rebate), solved to PV = 0 under the curve with the hold-forward tail, then held fixed in every cell |
| Chain | spot + IM2611/2612/2703/2706 closes (index-dashboard DB, read-only) + MO2709 C/P 6600 at the CFFEX settlement 464.6 / 840.0 |
| Rate, vol | flat r = 1.4%; flat vol 27.62% = Black-76 IV of the MO2709 settlement pair at K = 6200 (nearest the pillar forward) |
| Curve model | `CarryCurve` (LOG_LINEAR_DF) through the five pillars, tail = `FLAT_FORWARD_CARRY` (hold forward carry, default), `FLAT_CARRY_YIELD` (hold zero carry yield) or `ZERO_FORWARD_CARRY` (zero carry past the last node) |
| Flat model | `ContinuousDividendYield(q(T_mat))` read off the same curve and tail, so both models agree on F(T_mat) |
| Engine | `SnowballQuadEngine`, readout `transition`, resolved grid 2075 (doubled to 4151 as a check); PDE accuracy `high` cross-check |

## Headline numbers

From `results/pricing.json` and `results/greeks.json` (bp of notional; `04_report.py`
prints this table):

| tail | curve PV | flat PV | gap | flat-desk coupon | q_flat | Δ curve / flat (CNY/pt) | parallel rho curve / flat (bp per +1% q) | rho beyond last quote |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hold forward carry | 0.00 bp | 44.18 bp | +44.18 bp | −1.30% p.a. | 7.15% | 976.2 / 952.3 | −83.59 / −79.60 | 43.9% |
| hold zero carry yield | −125.46 bp | −117.90 bp | +7.56 bp | −0.23% p.a. | 9.12% | 1,082.6 / 1,079.0 | −85.66 / −84.95 | 41.7% |
| zero carry past last node | 158.37 bp | 228.47 bp | +70.10 bp | −1.95% p.a. | 4.73% | 840.8 / 797.5 | −80.58 / −72.67 | 46.7% |

- **Misprice.** Under the default tail the flat q(T_mat) = 7.15% values the note
  +44.18 bp above the curve; a flat-q desk would solve a coupon 1.30% p.a. lower.
  The gap is positive under all three tails.
- **Why.** The curve puts q in the 2027 H1 dividend window (zero q 9.47% at IM2703,
  10.52% at IM2706), so its first-year forwards are lower: P(KO ≤ 1Y) 64.2% vs 66.6%,
  P(KI) 62.9% vs 61.6%, expected life 0.920Y vs 0.883Y (curve vs flat).
- **Hedge.** Totals are close under the default tail (spot delta 2.5%, parallel rho
  5.0% apart), but 44% of the parallel rho lives beyond the last quote, and the
  pillar hedge is a calendar spread: +11.25 IM2706 lots against −31.26 MO2709 C−P
  pairs, which IM futures alone cannot hold.
- **Model risk.** The tail convention alone moves the curve PV from −125.46 to
  +158.37 bp (a 284 bp range), more than any flat-vs-curve gap.
- **Marks.** MO close instead of settlement, or the all-CFFEX-settlement chain,
  moves the default-tail gap to 38.15 / 59.64 bp.

The full write-up, with every table and chart, is `results/report.html`
(self-contained, opens offline).

## Rerun

```bash
python example/case_studies/snowball_2y_carry_curve/01_snapshot.py   # live: dashboard DB + CFFEX CSV (--db, --cffex-csv)
python example/case_studies/snowball_2y_carry_curve/02_pricing.py    # ~1.5 min -> results/pricing.json
python example/case_studies/snowball_2y_carry_curve/03_greeks.py     # ~29 min  -> results/greeks.json
python example/case_studies/snowball_2y_carry_curve/04_report.py     # results/report.html
```

Stages 02–04 run offline from the committed snapshot. In a worktree, prefix
each command with `env PYTHONPATH=<worktree>` (a `.pth` otherwise imports the
main checkout's `quantark`). Stage tests: `test/test_snowball_2y_carry_curve_study.py`.

## Files

| path | what |
| --- | --- |
| `_common.py` | constants, snapshot loaders, term sheet, product, curves per tail, environments, engines, coupon solve, forward-interval yield bump; loaded by path as `snowball_2y_cc_common` |
| `01_snapshot.py` | dashboard DB + CFFEX CSV → `data/market_20261105.json` (fails closed on a missing row) |
| `02_pricing.py` | coupon, PVs per tail × model, event statistics, grid doubling, PDE cross-check, mark sensitivity → `results/pricing.json` |
| `03_greeks.py` | spot / pillar / pinned-forward delta, pillar and forward-interval rho_q, scenarios → `results/greeks.json` |
| `04_report.py` | → `results/report.html` (Bloomberg × Swiss, inline SVG) |
| `data/market_20261105.json` | frozen market snapshot with DB revision ids / content hashes and the CSV's sha256 |

## Open items

- Historical curve replay / hedging backtest: deferred (static Greeks only).
- Stage 03 takes ~29 min because a QUAD re-price at a new spot rebuilds the
  engine's spot-keyed cache (~13 s vs ~0.8 s at the cached spot). Engine-side,
  out of scope here.
- One flat vol, flat r: smile, local/stochastic vol and a bootstrapped rate
  curve are out of scope.
