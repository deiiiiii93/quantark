## ADDED Requirements

### Requirement: Legacy flat names remain importable
Each legacy top-level name SHALL remain importable after the namespace
migration, including all of its submodules. The 12 legacy names are `asset`,
`backtest`, `cashleg`, `dynamicscenario`, `param`, `portfolio`, `priceenv`,
`rfq`, `simm`, `stresstest`, `util`, `var`.

#### Scenario: Deep submodule import via legacy name
- **WHEN** `import asset.equity.engine.mc` is executed in an environment with
  quantark installed
- **THEN** the import succeeds

#### Scenario: From-import via legacy name
- **WHEN** `from util.enum import OptionType` is executed
- **THEN** the import succeeds and yields the same class as
  `quantark.util.enum.OptionType`

### Requirement: Module identity preserved across spellings
A legacy flat import and its canonical `quantark.*` counterpart SHALL resolve
to the identical module object, so classes, enums, and module-level state have
a single identity regardless of import spelling.

#### Scenario: Same module object
- **WHEN** `import util.enum as old` and `import quantark.util.enum as new`
  are both executed in either order
- **THEN** `old is new` is true

#### Scenario: Enum and isinstance compatibility
- **WHEN** an enum member obtained via a legacy import is passed to code that
  compares it against the member obtained via the canonical import
- **THEN** identity comparison (`is`), equality, and `isinstance` checks
  behave as if there were a single import path

### Requirement: Import-order independence
The compatibility mechanism SHALL be active from interpreter startup
(registered via a `.pth` file installed with the distribution), so a legacy
flat import succeeds even when it is the first quantark-related import of the
process.

#### Scenario: Legacy import first
- **WHEN** a fresh interpreter executes `import asset` without any prior
  `import quantark`
- **THEN** the import succeeds

### Requirement: Deprecation warning per legacy root
The first import of each legacy top-level name in a process SHALL emit a
`DeprecationWarning` naming the canonical `quantark.*` replacement; subsequent
imports of the same root SHALL NOT emit additional warnings.

#### Scenario: Warn once
- **WHEN** `import util` is executed twice in one process (with warnings
  visible)
- **THEN** exactly one `DeprecationWarning` mentioning `quantark.util` is
  emitted

### Requirement: Installed distributions take precedence
The compatibility mechanism SHALL yield to genuinely installed third-party
distributions: when an installed package's name matches a legacy root (e.g.
HoloViz `param`), importing that name SHALL resolve to the installed package,
and the alias SHALL NOT reach inside such a package for submodule imports.

#### Scenario: Real package shadows the alias
- **WHEN** a distinct third-party package named `param` is installed in the
  environment and `import param` is executed
- **THEN** the third-party package is imported, not `quantark.param`
