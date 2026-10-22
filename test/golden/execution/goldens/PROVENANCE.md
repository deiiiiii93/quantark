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

## 2026-10-04: PDE 1D event statistics (0.9.0)

Spec: `docs/superpowers/specs/2026-10-03-pde-1d-event-statistics-design.md` §8.
Regenerated: `pde_phase4_goldens.json` only (`freeze_goldens.py phase4`). `phase0_goldens.json` and
`legacy_dask_phase6_goldens.json` are unchanged. No price, Greek or spot curve moved: only the
`price_with_events` event distributions, whose statistics now march as a value-policy column group
(stage 2 (a)-(c)); the Heston case is built with `statistics="flat_vol_surrogate"` (stage 2 (e)).

| case | field | before | after | stage |
|---|---|---|---|---|
| SnowballPDESolver | MATURITY_WITH_KI | 0.13047357797372322 | 0.1304735716186375 | (a) |
| SnowballPDESolver::refined | MATURITY_WITH_KI | 0.1304538691819851 | 0.1304538581729048 | (a) |
| LocalVolSnowballPDESolver | MATURITY_WITH_KI | 0.1405793684576984 | 0.1405793623884317 | (a) |
| HestonSnowballPDESolver (flat-vol surrogate) | MATURITY_WITH_KI | 0.14058071854906753 | 0.14058071248447548 | (a) |
| PhoenixPDESolver | MATURITY_WITH_KI | 0.13045698067871522 | 0.13045697411556476 | (b) |
| KOResetSnowballPDESolver | KO event times | the pre schedule (12) | both schedules merged (pre first at a tie) | (c) |
| KOResetSnowballPDESolver | MATURITY_NO_KO | 0.14853804206324972 | 0.08031493486734011 (the pre-maturity bucket) | (c) |
| KOResetSnowballPDESolver | MATURITY_WITH_KI | 0.1012128770787285 | 0.1010881124915345 (the post-maturity bucket) | (c) |

The Snowball/Phoenix MATURITY splits move by <= 1.1e-8 and the knock-out probabilities by round-off
(<= 2e-16): the knocked-in share now comes from the marched `ki_survive`. The KO-reset distribution
carries both schedules and two maturity outcomes with their own determination times (it was the
pre schedule with the post knock-outs folded into the maturity mass).

## 2026-10-06: PDE 1D remaining routes (0.10.0)

Spec: `docs/superpowers/specs/2026-10-04-pde-1d-remaining-routes-design.md` §8. No execution golden was regenerated:
`test/execution/` passed against the frozen files after stage 2 (full suite at e107b94e) and after stage 3's
knock-in-side row (b0afa9d4), compared through `test/golden_compare.py` as always.
