## ADDED Requirements

### Requirement: Dynamic Transaction Cost Model
The system MUST model transaction costs as a combination of spread and market impact:
- Spread varies by asset liquidity tier and volatility bucket (configurable)
- Impact cost scales non-linearly with trade size (configurable)
- Toggleable via backtest configuration with parameters

#### Scenario: Apply dynamic spread by volatility
- **WHEN** trading in a high-volatility bucket
- **THEN** use the bucket’s spread parameter to compute spread cost

#### Scenario: Apply market impact by trade size
- **WHEN** trade size exceeds the configured threshold
- **THEN** add extra cost per the non-linear impact function

#### Scenario: Backtest configuration toggles model
- **WHEN** the dynamic cost model is enabled in backtest config
- **THEN** the engine uses v2 costs; when disabled, it falls back to the fixed-cost model
