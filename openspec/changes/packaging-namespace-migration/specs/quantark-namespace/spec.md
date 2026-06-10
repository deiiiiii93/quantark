## ADDED Requirements

### Requirement: Single canonical namespace
All quantark library code SHALL live under the single top-level package
`quantark`, with the 12 historical top-level packages (`asset`, `backtest`,
`cashleg`, `dynamicscenario`, `param`, `portfolio`, `priceenv`, `rfq`, `simm`,
`stresstest`, `util`, `var`) as its direct subpackages.

#### Scenario: Canonical imports resolve
- **WHEN** `import quantark.asset.equity.product.option`,
  `import quantark.util.enum`, and `import quantark.rfq.service` are executed
- **THEN** each import succeeds and exposes the same public symbols the flat
  packages exposed before the migration

### Requirement: Canonical internal imports
Library code under `quantark/` SHALL reference sibling packages only via
canonical `quantark.*` imports; no flat top-level import of the 12 legacy
names SHALL remain in library code.

#### Scenario: No flat imports in library source
- **WHEN** library source under `quantark/` is searched for `import <flat>` or
  `from <flat>` statements targeting the 12 legacy root names
- **THEN** no matches are found

### Requirement: Test suite uses canonical imports
The test suite under `test/` SHALL import quantark code only via `quantark.*`
names, so the suite validates the canonical import path rather than the
compatibility shim.

#### Scenario: Suite passes on canonical imports
- **WHEN** the full pytest suite runs in an environment with quantark
  installed
- **THEN** all tests pass and no test module imports a flat legacy name

### Requirement: Public API preserved
The namespace move SHALL NOT rename, remove, or change the signature of any
public class or function; the consumer-facing surface (including
`RFQService._normalize_request` and `RFQService._evaluate_candidate`, which a
known consumer calls) SHALL remain available under the new namespace.

#### Scenario: Consumer-facing symbols survive the move
- **WHEN** the symbols listed in the consumer contact surface (see
  `docs/packaging-migration.md`) are imported via their `quantark.*` paths
- **THEN** every symbol resolves, and `RFQService._normalize_request` /
  `RFQService._evaluate_candidate` remain present
