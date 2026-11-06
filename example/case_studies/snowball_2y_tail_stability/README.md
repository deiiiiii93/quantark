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

To be filled from `results/stability.json` and `results/backtest.json` after the overnight run
(`04_report.py` prints the run summary and the qualifying tails).

| tail | std Δq(2Y), bp | Book A pooled std, bp | Book B pooled std, bp | A n_lower/n | B n_lower/n | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| hold forward carry | _tbd_ | _tbd_ | _tbd_ | – | – | baseline |
| converge hl 0.25Y | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| converge hl 0.50Y | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| converge hl 1.00Y | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

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

- Headline table and report conclusions: pending the overnight run.
- Overlapping inceptions make the paired tests optimistic; truncated runs are excluded from the terminal measure.
- One flat vol per day, flat r; vega is unhedged in both books.
- Promotion of the converging tail into `CarryCurve` is a later decision from this study's table.
