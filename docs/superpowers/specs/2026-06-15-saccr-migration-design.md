# SA-CCR Migration into QuantArk — Design Spec

**Date:** 2026-06-15
**Branch:** `worktree-saccr-migration`
**Status:** Approved at design gate (ZenMux Codex Review, gpt-5.5 xhigh)
**Source:** `/Users/fuxinyao/Documents/sa_ccr_demo` (standalone `saccr` package, ~2,200 LOC, validated vs the 5 Basel Annex 4a worked examples)

## 1. Purpose & Scope

Migrate the standalone Basel **SA-CCR** (Standardised Approach for Counterparty
Credit Risk) calculator into QuantArk as a self-contained supporting module
`quantark/saccr/`, following QuantArk conventions and the SIMM module precedent.

SA-CCR computes counterparty **Exposure at Default**:

```
EAD = alpha * (RC + PFE)          alpha = 1.4
RC  = replacement cost            (unmargined: max(V-C,0); margined: max(V-C, TH+MTA-NICA, 0))
PFE = multiplier * AddOn_aggregate
```

across all five asset classes (Interest Rate, FX, Credit, Equity, Commodity),
for both margined and unmargined netting sets.

**In scope:** full functional migration of the calculator, models, supervisory
parameters, add-on engines, the 5 Basel examples as tests plus component/edge
tests, a methodology doc, a module `CLAUDE.md`, an example script, and root
`CLAUDE.md` registration.

**Out of scope (deferred, documented):** a portfolio adapter mapping QuantArk
`FIPosition`/`EquityPosition` to `SACCRTrade`. No heuristic position→trade
conversion is provided. Jurisdictional SA-CCR variants beyond the Basel text.

## 2. Approach

Three approaches were considered:

- **A. Self-contained module mirroring SIMM (CHOSEN).** Lowest coupling, matches
  the existing regulatory-risk precedent, fast and safe.
- **B. Portfolio-integrated adapter.** Rejected for now: SA-CCR's regulatory trade
  attributes (supervisory-delta inputs, CDO tranches, hedging sets) do not map
  cleanly from existing positions; large scope, high risk. Deferred.
- **C. Thin lift-and-shift.** Rejected: violates QuantArk conventions (raw `math.*`,
  bare `ValueError`, Abramowitz–Stegun CDF approximation, generic `Trade` name).

## 3. Module Structure

```
quantark/saccr/
├── __init__.py             # explicit public API + __all__ (canonical quantark.saccr.* only)
├── calculator.py           # SACCRCalculator orchestrator
├── models/
│   ├── __init__.py
│   ├── enums.py            # AssetClass, Position, CreditRating, IndexGrade,
│   │                       #   CommodityHedgingSet, CommodityType, TransactionType,
│   │                       #   COMMODITY_TYPE_TO_HEDGING_SET  (OptionType reused from util)
│   ├── trade.py            # SACCRTrade  (domain validation -> ValidationError)
│   └── netting_set.py      # SACCRNettingSet
├── parameters/
│   ├── __init__.py
│   └── supervisory.py      # SupervisoryParameters: immutable Basel Table 2 data,
│                           #   SACCR_VERSION metadata, paragraph/table references
├── engines/
│   ├── __init__.py
│   ├── replacement_cost.py # RC unmargined / margined
│   ├── pfe.py              # multiplier (safe_exp) + PFE
│   ├── maths.py            # supervisory_duration, maturity_factor_*, supervisory_delta
│   │                       #   (scipy.stats.norm.cdf; domain-validated)
│   └── addons/
│       ├── __init__.py
│       ├── base.py
│       ├── interest_rate.py
│       ├── fx.py
│       ├── credit.py
│       ├── equity.py
│       └── commodity.py
├── results/
│   ├── __init__.py
│   └── result.py           # SACCRResult (full decomposition)
├── doc/
│   └── saccr_basel.md      # Basel SA-CCR methodology reference (from source reference/full.md)
└── CLAUDE.md               # module developer guide
```

`components/` and `addons/` from the source are folded into `engines/`;
supervisory tables move to `parameters/`; the result object moves to `results/`.

## 4. Public API

`quantark/saccr/__init__.py` exports (and lists in `__all__`):

- `SACCRCalculator`, `SACCRResult`
- `SACCRTrade`, `SACCRNettingSet`
- enums: `AssetClass`, `Position`, `CreditRating`, `IndexGrade`,
  `CommodityHedgingSet`, `CommodityType`, `TransactionType`
- `OptionType` (re-exported from `quantark.util.enum.option_enums`)

All internal imports use canonical `quantark.saccr.*`. No top-level `saccr.*`
imports anywhere; the example script and tests use the public API only. The
package is auto-discovered by existing QuantArk packaging (already under
`quantark/`).

## 5. Conventions Applied

### 5.1 Numerical operations & domain policy
- Replace all raw `math.{log,exp,sqrt}` and ad-hoc tolerances with
  `quantark.util.numerical` (`safe_log`, `safe_exp`, `safe_sqrt`, `safe_divide`,
  `is_zero`, `is_close`, `Tolerance`, `validate_positive`).
- **Domain-first policy:** every formula validates regulatory inputs *before*
  calling any `safe_*` helper. `safe_*` is never used to make an invalid input
  computable. Examples:
  - supervisory delta: require `underlying_price > 0`, `strike_price > 0`,
    `supervisory_volatility > 0`, `exercise_date > 0` → else `ValidationError`,
    *then* `safe_log(P/K)`, `safe_sqrt(T)`.
  - maturity factor / duration: require non-negative/positive maturity and
    `E >= S` → else `ValidationError`, then `safe_sqrt`.
  - multiplier: guard `addon_aggregate <= 0` and use `safe_exp` for the (≤0)
    exponent; `safe_divide` for the denominator.
- Invalid **inputs** → `ValidationError`. Genuine numerical
  overflow/underflow/non-convergence → `NumericalError`. No silent fallbacks.

### 5.2 Normal CDF (exact semantics)
- The supervisory option delta uses `scipy.stats.norm.cdf` exclusively (consistent
  with existing equity analytical engines). The source's Abramowitz–Stegun
  approximation is removed; **no** approximation fallback.
- Migration delta is negligible (A&S absolute error ≈ 7.5e-8) and within the Basel
  examples' tolerances. A regression test asserts `|Phi_scipy - Phi_AS| < 1e-6`
  across a grid to document the change.

### 5.3 Exceptions
- All error paths use `quantark.util.exceptions`
  (`ValidationError`, `NumericalError`). Bare `ValueError`/`KeyError`/
  `ZeroDivisionError`/SciPy domain errors must not escape regulatory paths.

### 5.4 Enums
- Reuse `OptionType` (CALL/PUT) from `quantark.util.enum.option_enums` — audited as
  a pure call/put payoff enum, semantically identical to SA-CCR's need.
- `Position` (LONG/SHORT) and all SA-CCR-specific enums stay local in
  `models/enums.py`.

### 5.5 Supervisory parameters (audit/version)
- `SupervisoryParameters` holds the Basel Table 2 supervisory factors,
  correlations, option volatilities, the maturity-bucket boundaries, alpha, and
  the multiplier floor as immutable class-level data.
- A `SACCR_VERSION` identifier records the implemented rule set; every table
  carries paragraph/table references (matching the SIMM discipline).
- Lookup getters raise `ValidationError` for unknown asset class / commodity type
  / credit grade / index grade — **no silent default bucket**.

### 5.6 Naming
- `Trade` → `SACCRTrade`; `NettingSet` → `SACCRNettingSet`. `SACCRCalculator`,
  `SACCRResult` keep the regulatory prefix for public objects.

## 6. Data Model & Conventions

### 6.1 Time / maturity convention
- Times (`start_date` S, `end_date` E, `maturity` M, `exercise_date` T) are
  **year-fractions** (floats), exactly as in the Basel worked examples.
- MPOR is expressed in **business days**; `mpor_years = mpor_days / 250`.
- Maturity is floored at **10 business days** (`10/250`).
- Validation: `maturity > 0`, `E >= S >= 0`, `exercise_date > 0` for options.
- Rationale: SA-CCR formulas are defined on year-fractions; introducing QuantArk
  calendar/day-count objects would change the regulatory input model with no
  methodological benefit. (Documented limitation.)

### 6.2 Currency convention
- **Single reporting currency:** all monetary values (notional, market value,
  collateral, TH, MTA, ICA) are in one reporting currency (mirrors SIMM's
  calculation-currency convention). No FX conversion is performed and no FX market
  data is consumed.
- `currency` (IR) and `currency_pair` (FX) are **hedging-set keys only**.
- `currency_pair` must use canonical `"CCY1/CCY2"` format; other formats raise
  `ValidationError`.

### 6.3 Collateral / sign conventions (documented + tested)
- `market_value` V: positive = in-the-money to the bank.
- `variation_margin` (VM) and `independent_collateral_received` (ICA received):
  positive increases C.
- `independent_collateral_posted` (unsegregated): positive reduces C and NICA.
- `independent_collateral_posted_segregated`: excluded from C, but treated per
  para 143 (not subtracted from C).
- `NICA = ICA_received - ICA_posted(unsegregated)`.
- Collateral is assumed already haircut-adjusted (cash collateral); non-cash
  haircuts are out of scope (documented).
- `threshold` (TH), `minimum_transfer_amount` (MTA) in reporting currency.
- The margined RC `max(V-C, TH+MTA-NICA, 0)` and the margined-EAD cap are tested
  under these conventions.

## 7. Result Object

`SACCRResult` (dataclass) exposes the full auditable decomposition:

| field | meaning |
|---|---|
| `ead` | final EAD (capped for margined sets) |
| `ead_uncapped` | EAD before the margined cap |
| `ead_capped` | bool, whether the unmargined cap bound the result |
| `rc` | replacement cost |
| `pfe` | potential future exposure |
| `multiplier` | PFE multiplier |
| `alpha` | 1.4 |
| `addon_aggregate` | sum of asset-class add-ons |
| `addon_by_asset_class` | dict `{asset_class_name: addon}` |
| `addon_by_hedging_set` | dict `{hedging_set_label: addon}` |
| `v`, `c`, `nica` | market value, collateral, net independent collateral |
| `is_margined` | bool |

## 8. Testing

Run via the main-repo venv with worktree source shadowing:
`PYTHONPATH=$PWD <mainvenv>/bin/python -m pytest test/test_saccr*.py`.

- **`test/test_saccr.py`** — port all 5 Basel Annex 4a worked examples
  (IR unmargined; Credit unmargined; Commodity unmargined; IR+Credit combined;
  IR+Commodity margined) with the doc's stated tolerances via `pytest.approx`.
- **`test/test_saccr_components.py`** — component & edge tests (exact assertions
  via `is_close`/`Tolerance` where values are hand-computable):
  - RC unmargined & margined (incl. cap)
  - multiplier floor (0.05) & boundary (V−C ≥ 0 ⇒ 1.0)
  - `maturity_factor_unmargined`/`margined`, `supervisory_duration`
  - option `supervisory_delta` vs `scipy.stats.norm.cdf`; A&S regression bound
  - IR 3-bucket partial-offset aggregation
  - FX hedging-set full offset
  - credit single-name vs index; equity single vs index
  - commodity hedging-set mapping + unknown-type rejection
  - CDO tranche delta formula + attachment/detachment validation
  - collateral / NICA sign conventions; MPOR day→year conversion
  - invalid-input rejection (non-positive notional, `E<S`, missing
    asset-class attrs, non-canonical `currency_pair`) → `ValidationError`
  - public-import sanity (`from quantark.saccr import ...`); no flat `saccr.*`

## 9. Documentation

- `quantark/saccr/doc/saccr_basel.md` — methodology reference (ported from source
  `reference/full.md`), referenced by paragraph throughout the code.
- `quantark/saccr/CLAUDE.md` — module developer guide (structure, conventions,
  sign/currency/time conventions, deferred portfolio adapter note).
- Root `CLAUDE.md` — add a row to the supporting-modules table:
  `SA-CCR | quantark/saccr/ | Basel SA-CCR counterparty EAD; see saccr/CLAUDE.md`.
- `example/saccr_demo.py` — runnable demo using the public API.

## 10. Migration Exclusions

Not migrated: `openspec/`, `.cursor/`, `reference/images/` (80 jpgs),
standalone `pyproject.toml` / `requirements.txt` (folds into QuantArk packaging),
`src/saccr.egg-info/`.

## 11. Review Process (per the goal)

Every development gate is reviewed by **ZenMux Codex Review** (`openai/gpt-5.5`,
`xhigh`/`high` reasoning) instead of the user:
1. Design gate — **APPROVED**.
2. Spec gate — this document.
3. Implementation-plan gate.
4. Final-implementation gate (zenmux-codex-review-loop until clean).
