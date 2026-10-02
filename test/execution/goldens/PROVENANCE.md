# Execution goldens — provenance notes

Frozen with `test/execution/freeze_goldens.py` (`main()` for `phase0_goldens.json`,
`phase4` for `pde_phase4_goldens.json`, `phase6_dask` for
`legacy_dask_phase6_goldens.json`). They are same-machine references (arm64);
other architectures compare through `test/golden_compare.py`. A diff after a
regeneration is a behaviour change to investigate, never a tolerance to widen.

## 2026-10-02 — 0.8.0, the PDE 1D numerical policy (spec 3A)

Spec: `docs/superpowers/specs/2026-10-01-pde-1d-numerical-policy-design.md`.
Regenerated: `pde_phase4_goldens.json` and `phase0_goldens.json`, whose
consumers the spec's flips moved. `legacy_dask_phase6_goldens.json` is
unchanged (its consumers stayed green).

`pde_phase4_goldens.json` (the fixtures pin `event_projection="nodal"`):

| case | before | after | flip |
|---|---|---|---|
| SnowballPDESolver (continuous KI) | 968627.8529297959 | 968776.3703453565 | (d): NODAL masks interior events |
| SnowballPDESolver::refined | 968582.3401649891 | 968599.1436502667 | (d) |
| LocalVolSnowballPDESolver | 977333.5648267246 | 977324.1585073968 | (d) |
| KOResetSnowballPDESolver (discrete KI) | 97.65382706245767 | 97.65346789109894 | (f): half-step damping |
| EuropeanPDESolver | 9.227268060979446 | 9.227268610555406 | (f) |
| EuropeanPDESolver::refined | 9.2271356655672 | 9.227135803060882 | (f) |
| HestonSnowballPDESolver | 973280.2206936385 | 973280.220693638 | none: a stale freeze; 0.7.0 itself gives …638 (bit-identical, checked) |
| PhoenixPDESolver (continuous KI) | unchanged | | |

`phase0_goldens.json`: `equity_pde_european` 10.450850690464524 → 10.450851205998426,
flip (f). No other case moved.

The continuous-KI Snowball and local-vol Snowball moved at (d) only, and are
bit-identical between the (d) and (f) trees: half-step damping does not reach a
product with continuous knock-in (spec 3A decision 12).
