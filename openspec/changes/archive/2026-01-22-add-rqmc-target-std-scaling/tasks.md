## 1. MCParams extensions
- [x] 1.1 Add RQMC fields to MCParams (target std, mode, min/max batches).
- [x] 1.2 Add validation for RQMC fields.
- [x] 1.3 Add helper to resolve target std from notional/relative settings.

## 2. Engine integration
- [x] 2.1 Update SnowballMCEngine RQMC to use MCParams helper.
- [x] 2.2 Update Phoenix/Barrier/Euro/Asian/Digital/American MC engines to use helper.

## 3. Tests and docs
- [x] 3.1 Unit tests for target std scaling.
- [x] 3.2 Update docs or usage notes if needed.
