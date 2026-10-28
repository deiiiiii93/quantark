# Local-vol transition QUAD — report index

The measured report is [`report.html`](report.html) (Bloomberg × Swiss UI language, light/dark, rendered by
`render_report.py` from the JSON files in this folder). Equations, literature review and contract scope:
[`theory.md`](theory.md).

| File | What it holds |
|---|---|
| `reference.json` | D12 references for 1y/2y Snowball, 1y put, 1y up-out: production PDE ladder (to 6400×64) cross-confirmed by refined QUAD; Dupire repricing gap table. Rows are fingerprinted (contract terms, engine parameters, fixture bytes, sources). |
| `bakeoff.json` | Kernel bake-off on the 1y Snowball: CTMC (Chebyshev / Talbot contour) vs moment-matched Gauss + Richardson; decision matrix. |
| `tuning.json` | Matched-accuracy ladders for QUAD, PDE and MC on every case (fingerprinted rows); cheapest configuration meeting the budget and the binding quantity. |
| `benchmark.json` | One timing window: cold full-risk medians (3 alternated repeats, single-threaded BLAS) of the tuned QUAD and PDE, QUAD within half the budget, the 2y slab-cache ablation, warm-moved rows for both engines, accuracy at every timed configuration, untimed validation residuals. |
| `narrative.json` | The written verdict, findings and limitations the HTML renders. |

Headline (cold full risk, eight valuations, each engine at its cheapest configuration within budget, one window):

| Case | QUAD | Production PDE | Speedup | QUAD within ½ budget |
|---|---|---|---|---|
| Snowball 1y (daily KI, monthly KO) | 0.460 s | 1.954 s | 4.25× | 4.05× |
| Snowball 2y | 0.756 s | 4.386 s | 5.80× | 4.85× |
| Put 1y | 0.223 s | 0.202 s | 0.90× | 0.68× |
| Up-and-out 1y (monthly) | 0.227 s | 0.176 s | 0.78× | 0.67× |

The production LV Monte Carlo does not reach the budget within 2^20 paths (gamma |bias| + 2 SE is 36–53× the budget).
