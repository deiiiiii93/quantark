# Change: Add Asian Option Product

## Why

Asian options are widely-used path-dependent derivatives where the payoff depends on the average price of the underlying asset over some period. They are popular for:
- Reducing manipulation risk (harder to manipulate average than spot)
- Lower volatility exposure (averaging smooths price fluctuations)
- Hedging periodic cash flows (e.g., commodity exporters/importers)

The library currently lacks Asian option support despite having other exotic options (barrier, digital, snowball).

## What Changes

- Add `AsianOption` product class extending `BaseEquityOption`
- Support both **arithmetic** and **geometric** averaging
- Support both **fixed strike** (average price) and **floating strike** (average strike) variants
- Support discrete observation schedules (list of dates/times)
- Add enum types for `AveragingType` and `AsianStrikeType`

## Impact

- Affected specs: New `asian-option` specification
- Affected code:
  - `asset/equity/product/option/asian_option.py` (new)
  - `util/enum/option_enums.py` (add enums)
  - `asset/equity/product/option/__init__.py` (export)
  - `asset/equity/product/__init__.py` (export)
- No breaking changes to existing APIs
