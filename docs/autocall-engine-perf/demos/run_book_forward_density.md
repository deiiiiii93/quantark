# Full-book forward-density A/B — protocol notes (2026-08-24)

The driver lives outside this repo (the adapter repo owns the book and its
venv); these notes make the run reproducible.

## Rig

- One detached worktree at the branch tip (`git worktree add --detach
  <scratch>/wt-fwd HEAD`); the adapter venv's installed quantark wheel is
  PYTHONPATH-shadowed by the worktree source.
- Flag driver: the adapter constructs `QuadParams` internally and knows
  nothing of `event_stats_mode`. A `sitecustomize.py` on the PYTHONPATH
  (workers inherit it) wraps `QuadParams.__post_init__` and forces
  `event_stats_mode = "forward_density"` before validation:

```python
# <scratch>/fwdflag/sitecustomize.py
"""Force forward_density event stats for the book A/B (measurement shim)."""
import quantark.asset.equity.param.engine_params as ep

_orig = ep.QuadParams.__post_init__


def _patched(self):
    self.event_stats_mode = "forward_density"
    _orig(self)


ep.QuadParams.__post_init__ = _patched
```

- Sanity gate before burning a run: `PYTHONPATH=<tree>:<fwdflag> python -c
  "from quantark.asset.equity.param import QuadParams;
  print(QuadParams().event_stats_mode)"` must print `forward_density`, and
  without the fwdflag entry `stacked`; `quantark.__file__` must resolve into
  the worktree.

## Arms

Identical to the 2026-08-24 protocol (`SOLUTIONS-2026-08-24.md`): QUAD config
(`autocallablePriceModel=quad`, `gridXQuad=1001`), `--as-of-date 2026-06-30
--workers 8`, 97-row book, both arms back-to-back in ONE window with a 10 s
load/RSS/swap sampler:

1. `stacked`: `PYTHONPATH=<tree>`
2. `forward`: `PYTHONPATH=<tree>:<fwdflag>`

Re-run any anomalous arm in a fresh window before believing it (shared-host
rule — the first 2026-08-24 fixed-QUAD window read 5,519 s from external
contention).

## Comparison

`compare_fwd.py` (session scratchpad): per-column exactness + max abs/rel
diff over the 19 PV/greek columns × 97 rows. Expectation: `pv` and all greek
columns EXACT (both modes' npv is the backward `price()`); leg-decomposition
columns (`pv_premium`, `pv_interest`, `pv_rebate`, ...) may differ within the
banked tolerances since they read the event distribution.

## Results

Banked in `docs/autocall-engine-perf/FORWARD-DENSITY-EVIDENCE-2026-08.md`
(section "Full-book adapter A/B").
