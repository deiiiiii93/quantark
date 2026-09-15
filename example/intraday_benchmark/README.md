# Intraday Gate D benchmark

Measures intraday valuation latency and peak traced memory at declared accuracy. There is no SLA: the deliverable
is the measured table next to the Gate C status of each state.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python example/intraday_benchmark/run_benchmark.py --engine quad_v2 --workload single --repeats 5 --out quad_single.json
```

| Workload | What is timed |
|---|---|
| `single` | price + delta, gamma, vega, rho, theta under `point` and `desk_bump`, one hour and one second before a fixing; cold (fresh engine and context) and warm (same engine, re-resolved context) medians |
| `curve101` | `spot_curve` over 101 spots across the 103 KO barrier, one hour and one second before a fixing: first curve (preparation included) and warm curve |
| `batch100` | 100 mixed contracts (snowball, phoenix, digital, barrier; mixed horizons and profiles) through `value_intraday_many`: total, per-item median, failures |

Engines: `quad_v2` (QUAD V2 for autocallables, closed forms for digital/barrier), `pde`, `mc_rqmc` (2^14 paths,
seed 11), `analytical` (digital and barrier only).

Every row records the platform tag, thread environment, engine settings, peak traced allocation (`tracemalloc`,
which sees NumPy buffers), and the Gate C accuracy status of the matching price cell where one exists.

Timing discipline: run every engine x workload back-to-back in one window on an otherwise idle machine; a shared
machine distorts ratios between runs, so never compare arms timed in different sessions. PDE memory scales with
points x time nodes; keep other heavy jobs off the machine while `pde` rows run.
