# Tasks: Add Airbag Snowball Payoff Support

## Implementation Tasks

- [x] 1. Update `SnowballOption.get_maturity_payoff_v1()` to implement airbag logic
  - Check if `airbag_config.airbag_barrier` is not None
  - Standard (`is_reverse=False`)
    - If spot < airbag_barrier: use `airbag_participation_rate` and `airbag_strike`
    - If spot >= airbag_barrier: use standard participation rate (current behavior)
  - Reverse (`is_reverse=True`)
    - If spot > airbag_barrier: use `airbag_participation_rate` and `airbag_strike`
    - If spot <= airbag_barrier: use standard participation rate (current behavior)

- [x] 2. Add unit tests for airbag payoff calculation
  - Test airbag payoff when spot < airbag_barrier
  - Test standard payoff when spot >= airbag_barrier
  - Test airbag with custom airbag_strike
  - Test airbag payoff for reverse snowball when spot > airbag_barrier
  - Test standard payoff for reverse snowball when spot <= airbag_barrier

- [x] 3. Update reverse airbag validation in `create_airbag_snowball()`
  - Standard: require airbag_barrier < ki_barrier (downside airbag below KI barrier)
  - Reverse: require airbag_barrier > ki_barrier (upside airbag above KI barrier)

- [x] 4. Add integration test with MC engine
  - Price airbag snowball with MC engine
  - Verify price differs from standard snowball with same parameters
  - Verify KO/V0/V1 probabilities are reasonable

- [x] 5. Run relevant tests to ensure no regression
  - `pytest test/test_snowball_helpers.py -v` - All 53 tests passed
  - Verified airbag-related tests pass
