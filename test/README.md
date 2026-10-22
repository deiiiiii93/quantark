# Tests

`pytest` runs `test/` (see `pytest.ini`: parallel by default, `-n0` to debug).

## Layout

The folders mirror the `quantark/` package tree, so the tests for a module are where you would
look for the module:

| Folder | Tests for |
| --- | --- |
| `test_asset/{fx,bond,rate,credit}/` | `quantark.asset.*` |
| `test_asset/equity/<layer>/` | `quantark.asset.equity`, by layer: `product_*` (by family), `engine_pde`, `engine_analytical`, `engine_mc`, `engine_quad`, `lifecycle`, `riskmeasures`, `report`, `market`, `param`, `process` |
| `test_backtest/`, `test_var/`, `test_simm/`, `test_stresstest/`, `test_util/`, `test_param/` ... | the like-named packages |
| `volmodels/`, `volcalibration/`, `modelvalidation/`, `sacva/`, `saccr/`, `pnlexplain/`, `montecarlo/`, `validation/`, `execution/`, `intraday/` | the like-named packages |
| `pde_core/`, `pde_grid/`, `pde_routes/`, `pde_statistics/`, `pde_3a/`, `accrual_time/`, `lifecycle_ko_reset/`, `theta_boundary/`, `simulation/` | topic suites that predate the mirror (PDE solver, lifecycle, replay goldens) |
| `golden/` | numeric-correctness tests and their frozen data: a test is golden if it fails when a number produced by `quantark` moves against a frozen value. Holds `pde_bitwise/` (byte-exact PDE book), `replay_golden/` + `replay/` (replay goldens), `modelvalidation/` (certification studies, committed surfaces, banked-certificate replay), `cashleg/`, and the pins moved out of the study tests (`mo_volmodels/`, `snowball_q_term_structure/`). Other tests compare against frozen values inside ordinary files (e.g. the PDE flat-identity golden, the QUAD v2 independent-lognormal checks); moving them whole would split their files, so they stay |
| `mo_volmodels/` | the one end-to-end guard for the `mo_volmodels` reference study; the study's own tests live in `example/case_studies/<study>/tests/` |

### Why some folders start with `test_`

Every test folder is a package and `test/` is on `sys.path`, so a folder's name is also an import
name. `quantark/_compat.py` keeps twelve legacy flat imports alive (`asset`, `backtest`,
`cashleg`, `dynamicscenario`, `param`, `portfolio`, `priceenv`, `rfq`, `simm`, `stresstest`,
`util`, `var`). A test folder with one of those names would shadow `from var import ...`, so
those twelve are prefixed `test_`. Do not rename them.

### Helpers

Shared helper modules (`*_fixtures.py`, `helpers_*.py`, `golden_compare.py`, `mc_se_*.py`,
`term_structure_benchmarks.py`) stay at the `test/` root and import by bare name.
A test that locates the repository root or `test/data` from `__file__` must account for its
folder depth (`Path(__file__).resolve().parents[N]`).

## Golden tests and cross-architecture tolerance

CI runs x86_64; the committed goldens were frozen on ARM64. Use `golden_compare.py` for values that
differ in the last digits across architectures, and keep same-machine invariants exact.

## Running a folder

```bash
python -m pytest test/test_asset/equity/engine_pde
```
