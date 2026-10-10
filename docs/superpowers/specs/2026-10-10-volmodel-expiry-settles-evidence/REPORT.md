# W9: the vol-model routes settle on the exercise date

User ruling 2026-10-10: a vol-model equity route valued at T = 0 (a dated contract on its exercise instant, or its
numeric twin aged to `maturity == 0.0`) settles: the payoff at today's spot x the settlement discount factor,
short-circuited before the vol model runs. No new payoff, knock or settlement logic: each route calls the expiry
code of its Black-Scholes (BSM) counterpart of the same method. Branch `w9-volmodel-expiry-settles`, from W8 e9daa6ed.

## 1. What changed: the shared code each route calls

| Route | BSM counterpart | Price at T = 0 | Greeks at T = 0 |
|---|---|---|---|
| `HestonAnalyticalEngine` | `BlackScholesEngine` | `settlement_support.expiry_terminal_value` | `BaseEngine.calculate_greeks` (bump-and-reprice, as BS) |
| `HestonMCEngine` | `EuropeanMCEngine` | `expiry_terminal_value` | `BaseEngine.calculate_greeks` |
| `LocalVolMCEngine` | `EuropeanMCEngine` | `expiry_terminal_value` | `BaseEngine.calculate_greeks` |
| `HestonSLVMCEngine` (SLV European; no W8 TODO) | `EuropeanMCEngine` | `expiry_terminal_value` | `BaseEngine.calculate_greeks` |
| `HestonPDESolver` | `EuropeanPDESolver` | `european_pde_solver.european_expiry_outcome` | the same (the BS PDE route's outcome) |
| `HestonSLVPDESolver` (SLV European; no W8 TODO) | `EuropeanPDESolver` | `european_expiry_outcome` | the same |
| `LocalVolBarrierMCEngine` | `BarrierOptionMCEngine` | `barrier_option_mc_engine.barrier_expiry_value` | `BaseEngine.calculate_greeks` (inherited, unchanged) |
| `HestonBarrierMCEngine` | `BarrierOptionMCEngine` | `barrier_expiry_value` | `BaseEngine.calculate_greeks` |
| `HestonSLVBarrierMCEngine` | `BarrierOptionMCEngine` | `barrier_expiry_value` | `BaseEngine.calculate_greeks` |
| `HestonBarrierPDESolver` | `BarrierPDESolver` | `barrier_pde_solver.barrier_expiry_outcome` | the same (new `_Heston2DBarrierBase.calculate_greeks`) |
| `HestonSLVBarrierPDESolver` | `BarrierPDESolver` | `barrier_expiry_outcome` | the same |

Moves (bit-preserving for the BSM engines):

- `expiry_terminal_value(product, env, timing, lifecycle_state)` = `product.get_payoff(spot) * timing.delay_df +
  pending_receivable_pv(...)`: the inline expiry branch duplicated in `BlackScholesEngine.price` and
  `EuropeanMCEngine.price`, which now call it (their `T < 1e-10` condition is untouched).
- `barrier_expiry_value(product, spot, terminal_df)`: the body of `BarrierOptionMCEngine._price_expiry_payoff`
  (private to that engine), now a module function the method delegates to.
- `european_expiry_outcome` / `barrier_expiry_outcome`: thin module functions running `EuropeanPDESolver(PDEParams())`
  / `BarrierPDESolver(PDEParams())`. The PDE expiry code is the core adapters' (`EuropeanAdapter.terminal_outcome`;
  `BarrierAdapter.maturity_value` + `fold_decision`) and needs a 1D solver; `prepare` decides the terminal outcome
  before any grid, market sample or vol is read, so nothing was moved. No MC is imported in a PDE solver.
- `at_exercise_instant(T)` = `T >= 0 and is_zero(T)` (`settlement_support`). A negative numeric maturity still refuses
  with the route's old message; a dated contract after its exercise raises in `get_maturity` first.
- `_Heston2DBarrierBase._admit`: the 2D barrier PDE routes' existing refusals (product check, ended schedule, delayed
  first-hit cash) factored out so `calculate_greeks` runs them before settling.

The short-circuit sits after each route's existing checks (product type, settlement capability, terminal lifecycle
PV, ended schedule, delayed pay-at-hit refusal, the LV barrier MC's ended-schedule replication) and before any model
input (time grid, forward curves, kernels). The local-vol / SLV MC Europeans still build their Dupire surface first
(in `price`, as `LocalVolPDESolver` does), so on a non-grid vol surface they refuse at T = 0 exactly as the LV PDE
does; the local-vol barrier MC builds its surface after the check, so it settles on any surface (the LV barrier PDE
refuses a non-grid surface; pre-existing ordering, unchanged).

All nine `TODO(dated-expiry)` markers are removed (`git grep "TODO(dated-expiry)" -- quantark`: none).

## 2. Tests (`test/test_dated_expiry_settles.py`, written first)

- W8's `REFUSING` loses its 7 vol-model entries (only the pre-existing Heston Phoenix PDE memory refusal stays); the
  SLV European MC / PDE and SLV barrier MC / PDE join W8's `european_call` / `barrier_up_out_call` families. These 27
  cells (11 routes, W8 spots) now run `test_dated_contract_on_its_exercise_date_prices_its_numeric_twins_settled_value`:
  dated == numeric == the hand settled value to 1e-12.
- New `VOL_CASES`, 121 cells over the 11 routes: European call (spots 110 / 90 / 100), put (90 / 110), call with a
  two-day settlement lag (110); barrier up-and-out call (110 / 90 / 130 at the barrier / 135 beyond), up-and-in call
  (110 / 130 / 135), up-and-in call carried as knocked in (110 / 135), down-and-out put (90 / 110 / 80 / 75),
  up-and-out call monitored at expiry (110 / 135), up-and-out call with a two-day lag (110 / 135).
  - `test_vol_model_route_on_its_exercise_date_prices_its_black_scholes_counterparts_settled_value`: vol dated ==
    vol numeric == the BSM counterpart on the same form (asserted to 1e-12; measured equal to the bit).
  - `test_vol_model_route_greeks_on_its_exercise_date_are_its_black_scholes_counterparts`: `calculate_greeks` has the
    counterpart's keys (`price`, `delta`, `gamma`) and values, in both forms.
- `test_vol_model_route_still_refuses_a_valuation_after_the_exercise` (11 routes): the next day, 15:00 on the
  exercise day (ruling Q4, the same instant) and a numeric maturity of -1e-6 all raise `ValidationError`.
- Before the fix: 269 failed (27 moved W8 cells + 121 price + 121 greeks), 195 passed. After: 464 passed.

## 3. Moved values

`moved_values.txt` (`measure.py` on e9daa6ed via `git archive` and on the branch, `summarize.py`): 538 of 538
measured values moved, all "raised -> settled value" (W8 family prices; W9 cell prices and greeks, both forms).
`today_e9daa6ed.txt` / `after_branch.txt` are the raw runs. Nothing else moved:

- The BSM counterparts on every W9 cell (52 cells x 2 forms, price and greeks): base == branch byte for byte.
- `test/pde_bitwise` book (713 rows, base e9daa6ed vs branch, `compare.py --manifest`): 0 diffs, 0 manifest violations.
- No certificate anchor or golden moved; none was touched.

## 4. Runs (`-n 2`, worktree PYTHONPATH)

1. `test_dated_expiry_settles`, every heston / local_vol / slv / volmodels / barrier-vol engine and kernel file,
   `test/volmodels`, BS European / barrier MC (`test_euro_mc_engine`, `test_european_option`,
   `test_european_settlement_matrix`, `test_barrier_option_mc_engine`), `test_engine_settlement_capabilities`,
   `test/theta_boundary`: 1009 passed.
2. Certificates and anchors: `test/modelvalidation/{test_anchors, test_banked_certificates,
   test_anchor_stencil_evidence, test_adi2d_import, test_localvol_study}`, `test_greek_certification`,
   `test/mo_volmodels/{test_adi_greek_certification, test_adi_slv_aggregate_certification}`, `test/pde_bitwise`,
   `test_barrier_vol_pde_engines`: 194 passed, 42 skipped (superseded certificates, skipped by design).
3. `test/pde_bitwise/capture_book.py` base vs branch: identical (above).

No tracked `example/` file changed.

## 5. Left refusing / not covered

- None of the nine sites is left refusing.
- Unchanged, not expiry-caused (W8 §5): Heston Phoenix PDE with memory coupons; QUAD V2 on a dated KO-reset whose
  schedule-1 records are in the past.
- Not in scope, untouched: the FX vol-model engines (`quantark/asset/fx/engine/{mc,pde}/heston*`, `local_vol*`; FX was
  not measured by W8 either); `SABRMCEngine` (not a `BaseEngine`, no settlement support);
  `quantark.asset.vol_model_risk`, which calls the private `_price_with_surface` / `_price_with_artifacts` directly and
  so still meets the kernels' T = 0 refusal (structured vol-model risk on the exercise date).
- Before T = 0 the Heston / LV / SLV European routes return theta and rho from `calculate_greeks`; at T = 0 they
  return the BSM counterpart's three keys (as `LocalVolPDESolver` already did at maturity).

## 6. Open question

**Q-W9-1. A knock-in carried as triggered is not read at T = 0 by any BSM barrier route.** With
`_otc_lifecycle_knocked_in = True` on an up-and-in call K100 B130 rebate 5 at spot 110, the BS analytical, MC, PDE,
QUAD and LV PDE routes all return the rebate 5.0 on the exercise date (both forms, measured on e9daa6ed and the
branch), not the payoff 10.0; before T = 0 a carried knock-in is priced as the knocked-in vanilla (remaining-contract
replication or the engines' own knocked-in path, all gated on T > 0). Per the ruling the vol-model barrier routes now
return the same 5.0 (pinned only as "equal to the BSM counterpart", never as a hand value). Reading the carried flag
at T = 0 would be new knock logic in the shared BSM expiry code (`_price_at_maturity`, `barrier_expiry_value`,
`BarrierAdapter.maturity_value`), so it is left for a ruling.
