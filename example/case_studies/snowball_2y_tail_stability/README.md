# 2Y snowball carry tail: converge-to-anchor vs hold-forward

Does a carry tail that converges to a trailing far-IM anchor give a more stable long-tenor
`q` than hold-forward on 2023-05 → 2026-11 CSI 1000 (000852.SH) history?
"Stable" is measured two ways: (1) the day-to-day change of `q(2Y)` rebuilt daily from the
IM futures + MO options chain, and (2) the daily hedging variance of a short 2Y snowball
hedged every day.

Two hedge books are measured, so we also learn whether the calendar spread mitigates the
IM basis oscillation: **Book A** (single far IM) and **Book B** (pillar lots: IM + MO C−P pair).
The adoption rule needs a lower hedging variance in **both** books, and the terminal P&L
measure must agree with the daily one.

## Setup

| | |
| --- | --- |
| Trade | Short 2Y standard snowball, notional 10 mio CNY, strike = KO ref = inception spot; KO 100% monthly from month 3 to 24; KI 75% discrete daily; coupon = KO coupon = rebate; principal excluded |
| Inceptions | Quarterly from 2023-10 (see `results/backtest.json` for the set); one coupon per inception solved to PV = 0 under hold-forward, shared by all tails |
| Tails | hold forward carry, hold carry yield, zero carry, and `converge_hl025/050/100`: carry converges from the last pillar to a trailing 63-day mean of the far-IM carry, half-life 0.25 / 0.5 / 1.0 Y (252-day anchor as a sensitivity) |
| Marks | spot = close; IM = CFFEX close; MO = CFFEX settlement; flat r = 1.4%, ACT/365F |
| Hedges | Book A follows the longest IM; Book B = IM pillar lots + MO C−P pair (held at its opened strike) + pinned-forward delta in the far IM. Multipliers IM 200, MO pair 100·DF(T_n) |
| Costs | IM 1 bp of traded notional; MO 0.2 pt per option leg per side (40 CNY per pair); variance is gross of costs |
| Engine | `SnowballQuadEngineV2` defaults, flat vol; V1 transition-readout parity check on inception day |
| Verdict | per converging tail and book: (a) pooled daily std lower, (b) paired variance ratio < 1 in a strict majority of inceptions, (c) completed-run terminal std lower and \|terminal\| lower in a strict majority (< 5 completed = not decidable); beats hold-forward only if it passes in both books |

## Headline numbers

Run of 2026-11-06/07: 13 quarterly inceptions (2023-10-09 → 2026-10-08) × 6 tails = 78 runs,
0 failed, 0 V1 parity flags; 11 knocked out, 2 truncated at data end (2026-11-06), so the terminal
measure uses 11 completed runs. Wall 11,598 s on 4 workers, no system sleep in the window; peak group
RSS 4.9 GB. Hedge numbers are daily hedged P&L in bp of notional, short desk.

**Qualifying tail: `converge_hl100`** (half-life 1.0Y) — the only half-life that passes all three
criteria in both books.

| tail | std Δq(2Y), bp/day | Book A pooled std | Book B pooled std | A / B daily lower | A / B terminal std | A / B \|terminal\| lower | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hold forward carry | 68.1 | 66.1 | 90.3 | – | 473 / 400 | – | baseline |
| hold carry yield (ref.) | 61.7 | 43.0 | 56.8 | 13/13 · 13/13 | 367 / 350 | 8/11 · 7/11 | reference (κ → 0 limit) |
| zero carry (ref.) | 28.9 | 35.7 | 40.9 | 13/13 · 13/13 | 579 / 567 | 4/11 · 5/11 | reference; terminal disagrees |
| converge hl 0.25Y | 38.7 | 37.0 | 44.3 | 13/13 · 13/13 | 361 / 356 | 7/11 · **5/11** | Book A only (B fails terminal) |
| converge hl 0.50Y | 45.4 | 38.5 | 47.8 | 13/13 · 13/13 | 361 / 353 | 7/11 · **5/11** | Book A only (B fails terminal) |
| converge hl 1.00Y | 51.6 | 40.1 | 51.2 | 13/13 · 13/13 | 363 / 351 | 8/11 · 7/11 | **beats hold-forward** |

What the table says:

- **The daily measure is not close.** Every non-hold-forward tail cuts the pooled daily std in every
  one of the 13 inceptions in both books (Holm-adjusted p ≈ 1e-6 in Book A, 3e-5 in Book B; optimistic, as the inceptions overlap). Book A
  falls 66 → 37–40 bp, Book B 90 → 44–51 bp. The cause is visible in the basis regression: hold-forward's
  Book A P&L loads |β| ≈ 147 bp on the far-IM basis change, the converging tails ≈ 46.
- **The terminal measure picks the half-life.** Terminal std falls ~110 bp (A) and ~45 bp (B) for all
  three half-lives; the split is the \|terminal\| count in Book B, 5/11 for 0.25Y and 0.5Y vs 7/11 for 1.0Y.
  That is a 2-run margin out of 11 overlapping runs: `converge_hl100` is the adopted tail by the
  pre-registered rule, not a sharply identified optimum.
- **The two-measure rule did its job on `zero_carry`:** lowest daily std in both books, highest terminal
  std (579 / 567 vs hold-forward 473 / 400). Smoother marks there are not a better hedge.
- **`converge_hl100` vs `hold_yield`** (descriptive, not a tested comparison): 40.1 vs 43.0 (A) and 51.2 vs
  56.8 (B) daily, terminal std about level. Most of the gain over hold-forward is already in "stop extending
  the last forward carry"; the anchor adds a modest further cut.
- **Calendar spread (Book B) vs single far IM (Book A):** the pillar book halves the basis loading
  (|β_far| 46 → 32 for hl100, basis R² 0.38 → 0.20) and the roll-day jump (54 → 21 bp), but its total
  daily std is *higher* in every tail (e.g. 51 vs 40 for hl100) and it costs more (12.9 vs 8.1 bp). It
  mitigates the basis oscillation; it does not lower total hedging variance on this history.

Reading notes (fixed before the run):

- `zero_carry` ranks most stable on `q(2Y)` only because `q(2Y) = r − B_n/2` halves the pillar noise;
  that is a property of the formula, not a recommendation.
- The vega-adjusted residual std is a diagnostic only; unhedged vega is common to all tails and
  never enters the verdict.
- Terminal P&L is at the traded price (net of each tail's day-0 model gap); the own-PV version is
  reported alongside.

The full write-up, with every table and chart, is `results/report.html` (self-contained, opens offline).

## Rerun

```bash
python example/case_studies/snowball_2y_tail_stability/01_data.py       # chain from CFFEX CSVs + spot (extensions under OUTPUT_DIR)
python example/case_studies/snowball_2y_tail_stability/02_stability.py  # ~seconds -> results/stability.json
python example/case_studies/snowball_2y_tail_stability/03_backtest.py   # overnight, 4 spawn workers -> OUTPUT_DIR/runs/*.json (resumable)
python example/case_studies/snowball_2y_tail_stability/04_report.py     # results/backtest.json + results/report.html
```

Stage 04 works on a partial set of runs (missing tails or inceptions are listed as missing), so it can
be re-run while the backtest is going; `--runs-dir` and `--out-dir` redirect it (e.g. a smoke render
from `runs_quick`). Long runs: `nohup caffeinate -i -m -s`. In a worktree, prefix each command with
`env PYTHONPATH=<worktree>`. Tests: `test/test_snowball_2y_tail_stability_study.py`.

## Files

| path | what |
| --- | --- |
| `s2ts/` | study package: `chain` (daily chain), `tails` (six tails, `ConvergingCarryCurve`, anchor), `stability`, `note` (lifecycle, daily greeks), `books` (hedge ledgers), `runner` (resumable spawn pool), `measures` (statistics, verdict), `html` (report rendering) |
| `01_data.py` | CFFEX daily CSVs + spot → daily chain pickle (shared history cache is read-only) |
| `02_stability.py` | q(2Y) day-to-day stability per tail × variant, ATM-switch floor → `results/stability.json` |
| `03_backtest.py` | 13 inceptions × 6 tails, one JSON per run under `OUTPUT_DIR/runs/` |
| `04_report.py` | aggregation → `results/backtest.json`, `results/report.html` (Bloomberg × Swiss, inline SVG) |
| `results/` | tracked outputs: `stability.json`, `backtest.json`, `report.html` |

## Open items

- The half-life choice rests on a 2-run margin in Book B's \|terminal\| count (5/11 vs 7/11); re-check after
  more inceptions complete before promoting a half-life.
- `converge_hl100` vs `hold_yield` was not part of the pre-registered tests; a paired test would settle
  whether the anchor earns its parameter.
- Overlapping inceptions make the paired tests optimistic; truncated runs are excluded from the terminal measure.
- One flat vol per day, flat r; vega is unhedged in both books.
- Promotion of the converging tail into `CarryCurve` is a later decision from this study's table.
