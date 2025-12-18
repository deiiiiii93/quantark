## 1. Implementation
- [x] Add probability terminal conditions for PDE engines
- [x] Add probability time-stepping PDE solve (same grid/scheme)
- [x] Apply exercise/call/put constraints consistently to probability
- [x] Return `conversion_probability` in PDE engine result objects
- [x] Ensure facade engine propagates PDE conversion probability

## 2. Tests
- [x] Add tests for always-convert → probability ~= 1
- [x] Add tests for never-convert → probability ~= 0
- [x] Add sanity tests: `0 <= p <= 1` for PDE engines

## 3. Validation
- [x] Run `openspec validate update-convertible-pde-conversion-probability --strict`
- [x] Run targeted pytest for convertible bond engines
