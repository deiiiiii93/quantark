# Research local-vol transition QUAD engine

A QUAD-style engine for the 1-D local-volatility model on the repository's own `LocalVolSurface`
(bilinear Dupire grid, flat extrapolation). Between observation dates the price moves by an exact
transition operator of a continuous-time Markov chain on a barrier-aligned log-spot grid; events
(daily KI, monthly KO) act on whole cells. Production routing is unchanged.

```python
from example.case_studies.localvol_transition_quad.contract import snowball_contract
from example.case_studies.localvol_transition_quad.engine import LocalVolQuad
from example.case_studies.localvol_transition_quad.grid import GridConfig
from example.case_studies.localvol_transition_quad.kernels import CTMCKernel
from example.case_studies.localvol_transition_quad.market import load_fixture

market = load_fixture()                          # 2024-02-08 CSI1000 Dupire surface, flat r and q
contract = snowball_contract(market.spot, 1.0, 12)   # KO 103%, KI 85% daily, coupon/rebate 15%
engine = LocalVolQuad(CTMCKernel(), GridConfig(cells=300, half_width_std=6.0))
risk = engine.full_risk(contract, market)
print(risk["values"])   # pv, delta, gamma, vega, theta, rho, carry — per unit notional S0
```

Units: PV per unit notional S0; delta = dPV/dS·S0; gamma = d²PV/dS²·S0²; vega per +0.01 absolute
local-vol shift; theta per calendar day (production semantics: the surface is not rolled); rho and
carry per +1bp. Contract semantics mirror the production `SnowballOption` as the model-validation
builders configure it (`adapters.production_snowball` builds the production twin).

| Module | Role |
|---|---|
| `contract.py` | Research contracts (vanilla, discrete up-and-out, two-channel Snowball) and theta shift |
| `market.py` | Fixture market, constant-σ control surface, vega/rho/carry bumps |
| `grid.py` | Finite-volume grid: barriers on edges with symmetric cell pairs, s0 on a centre, odd refinement |
| `kernels.py` | CTMC kernel (exact exponential: Chebyshev–Bessel or Talbot contour, `auto`), Gauss–Richardson alternative, exact BSM control |
| `engine.py` | Backward recursion, events, identity probes on every interval, readout, full risk |
| `adapters.py` | Production PDE / MC baselines through their public API, same units |
| `reference.py`, `bakeoff.py`, `benchmark.py` | Reference ladders, kernel bake-off, matched-accuracy tuning and timing CLIs |

Run from the repository root (worktrees: prefix `env PYTHONPATH=<worktree>:<worktree>/test`):

```sh
.venv/bin/python -m pytest -n0 test/volmodels/test_localvol_transition_quad_*.py

.venv/bin/python -m example.case_studies.localvol_transition_quad.reference --arm all --resume
.venv/bin/python -m example.case_studies.localvol_transition_quad.bakeoff --repeats 3
.venv/bin/python -m example.case_studies.localvol_transition_quad.benchmark tune --resume
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -m example.case_studies.localvol_transition_quad.benchmark time --repeats 3
.venv/bin/python docs/research/localvol_transition_quad/render_report.py
```

Results and the written verdict: [`docs/research/localvol_transition_quad/report.html`](../../../docs/research/localvol_transition_quad/report.html)
(index: `report.md`; equations and literature: `theory.md`). An unresolved grid raises `UnresolvedGrid`;
mass is never renormalised and nothing falls back silently.
