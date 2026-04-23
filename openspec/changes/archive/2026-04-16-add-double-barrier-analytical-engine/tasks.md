## 1. Product Definition

- [ ] 1.1 Create `DoubleBarrierOption` dataclass in `asset/equity/product/double_barrier_option.py`
- [ ] 1.2 Add `DoubleBarrierObservationType` enum to `util/enum/engine_enums.py` (or adjacent product enum file)
- [ ] 1.3 Wire product into `asset/equity/product/__init__.py`

## 2. Core Engine Implementation

- [ ] 2.1 Create `DoubleBarrierOptionAnalyticalEngine` in `asset/equity/engine/analytical/double_barrier_option_engine.py`
- [ ] 2.2 Implement continuous observation pricing using Ikeda & Kuintomo infinite series (call and put knock-out)
- [ ] 2.3 Implement knock-in pricing via parity (vanilla - knock-out)
- [ ] 2.4 Implement daily observation via barrier shift adjustment
- [ ] 2.5 Implement expiry observation via truncated-domain vanilla payoff
- [ ] 2.6 Add comprehensive input validation and edge-case handling (T=0, S at barriers, strike outside barriers)
- [ ] 2.7 Wire engine into `asset/equity/engine/analytical/__init__.py`

## 3. Testing & Validation

- [ ] 3.1 Create `test/test_double_barrier_option_engine.py`
- [ ] 3.2 Add continuous observation benchmark tests using all Table 4-15 cases
- [ ] 3.3 Add daily observation smoke and monotonicity tests
- [ ] 3.4 Add expiry observation closed-form tests
- [ ] 3.5 Add edge-case tests (T=0, S at/inside/outside barriers, deep ITM/OTM)
- [ ] 3.6 Run full test suite and ensure no regressions
