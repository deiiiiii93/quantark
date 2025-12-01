# Change: Dynamic Transaction Cost Model (v2)

## Why
The current backtesting cost model is fixed/linear and fails to capture how market volatility, liquidity, and trade size dynamically affect execution costs. Introducing a dynamic model improves realism and robustness of strategy evaluation.

## What Changes
- Add Dynamic Spread model: compute bid-ask spread by liquidity tier and volatility buckets
- Add Market Impact model: non-linear impact cost driven by trade size
- Add configuration (YAML): enable/parameterize in backtest config
- Default compatibility: when disabled, fallback to legacy fixed-cost logic

## Impact
- Affected specs: backtest-protocols
- Affected code: `backtest/transaction_costs.py`, `backtest/engine.py`, `backtest/config.py`, `backtest/results.py`
