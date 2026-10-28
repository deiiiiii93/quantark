# Local-vol transition QUAD — report index

The measured report is [`report.html`](report.html) (Bloomberg × Swiss UI language, light/dark, rendered by
`render_report.py` from the JSON files in this folder). Equations, literature review and contract scope:
[`theory.md`](theory.md).

| File | What it holds |
|---|---|
| `reference.json` | D12 references for 1y/2y Snowball, 1y put, 1y up-out: production PDE ladder (to 6400×64) cross-confirmed by refined QUAD; Dupire repricing gap table. Rows are fingerprinted (contract terms, engine parameters, fixture bytes, sources). |
| `bakeoff.json` | Kernel bake-off on the 1y Snowball: CTMC (Chebyshev / Talbot contour) vs moment-matched Gauss + Richardson; decision matrix. |
| `tuning.json` | Matched-accuracy ladders for QUAD, PDE and MC on every case; cheapest configuration meeting the budget and the binding quantity. |
| `benchmark.json` | Cold full-risk timings (3 repeats, alternated, one window, single-threaded BLAS) of the tuned configurations, warm-moved QUAD rows, accuracy at the timed configuration. |
| `narrative.json` | The written verdict, findings and limitations the HTML renders. |

Headline (cold full risk, eight valuations, each engine at its cheapest configuration within budget):

| Case | QUAD | Production PDE | Speedup |
|---|---|---|---|
| Snowball 1y (daily KI, monthly KO) | 0.451 s | 1.927 s | 4.27× |
| Snowball 2y | 0.739 s | 4.335 s | 5.86× |
| Put 1y | 0.220 s | 0.199 s | 0.90× |
| Up-and-out 1y (monthly) | 0.225 s | 0.170 s | 0.76× |

The production LV Monte Carlo does not reach the budget within 2^20 paths (gamma 36–53× over).
