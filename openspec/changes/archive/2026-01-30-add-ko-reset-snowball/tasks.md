# Tasks

## 1. Specification
- [ ] Review the KO-reset snowball requirements and finalize the spec scenarios
- [ ] Confirm schedule mode semantics (ABSOLUTE vs REBASED) and constraints for MC

## 2. Product
- [ ] Add `KnockOutResetSnowballOption` product class with dual KO schedules and KI switch logic
- [ ] Add new enum for post-KI schedule mode
- [ ] Add helper factory `create_ko_reset_snowball` and export it
- [ ] Add validation rules and cache key serialization

## 3. Engine (MC)
- [ ] Extend Monte Carlo pricing to support KO-reset product
- [ ] Implement KO schedule switching after KI for ABSOLUTE and REBASED modes
- [ ] Add event stats support for KO-reset product

## 4. Tests & Docs
- [ ] Add unit tests for KO-reset payoff/state transitions
- [ ] Add MC sanity tests (KO probability, KI/KO path selection)
- [ ] Add brief docs/example snippet for the new product
