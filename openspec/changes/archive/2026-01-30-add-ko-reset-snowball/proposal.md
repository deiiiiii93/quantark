# Change: Add KO-Reset Snowball Option (MC)

## Why
The trading desk needs a KO-reset snowball variant where KO terms switch to a second schedule after a KI event, with optional post-KI tenor extension. This product exists in term sheets (see docx) and requires explicit modeling in QuantArk.

## What Changes
- Add a new equity product class for KO-reset snowball options with dual KO schedules and KI-driven switch logic.
- Support both absolute (calendar) and rebased (KI-anchored) post-KI KO schedule modes.
- Extend the Monte Carlo engine to price the KO-reset product and report event stats.
- Add helper factory function(s), exports, and targeted tests/examples.

## Impact
- Affected specs: `snowball-ko-reset-option`
- Affected code: `asset/equity/product/option/`, `asset/equity/engine/mc/`, `util/enum/`
- Backward compatibility: Existing SnowballOption APIs remain unchanged.
