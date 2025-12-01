## 1. Implementation
- [ ] 1.1 Define YAML config structure (volatility buckets, liquidity tiers, impact params)
- [ ] 1.2 Implement dynamic spread and market impact in `backtest/transaction_costs.py`
- [ ] 1.3 Wire cost calculation in `backtest/engine.py` (order/execution path)
- [ ] 1.4 Add unit tests (edges: large/small trades, extreme vol, non-trading days)
- [ ] 1.5 Update examples and reports (`example/` and `reports/`)

## 2. Validation
- [ ] 2.1 Build reproducible market/trade samples and compare with legacy model
- [ ] 2.2 Verify full fallback to legacy logic when disabled (compatibility)
- [ ] 2.3 Pass `openspec validate add-transaction-cost-model-v2 --strict`
