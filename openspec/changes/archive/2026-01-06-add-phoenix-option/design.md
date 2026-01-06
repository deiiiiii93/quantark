## Context

Phoenix options are autocallable structured products popular in Asian markets, particularly for income-seeking investors. They provide periodic coupon payments when spot exceeds a coupon barrier, unlike Snowball options which only pay on knock-out events.

The QuantArk codebase already has a sophisticated Snowball option implementation with:
- Modular configuration classes (BarrierConfig, PayoffConfig, AccrualConfig, AirbagConfig)
- ObservationSchedule for discrete/continuous monitoring
- Day count conventions in util/calendar/day_counter.py
- CouponPayType enum for payment timing

## Goals / Non-Goals

### Goals
- Create a dedicated PhoenixOption product class with proper coupon barrier logic
- Support memory coupon feature (accumulate missed coupons)
- Integrate with existing day count conventions for coupon accrual
- Support all variants (standard, reverse, step-down)
- Reuse existing infrastructure where appropriate

### Non-Goals
- Engine implementation (MC/PDE) - deferred to future change
- Greeks calculation for Phoenix options
- Real-time market data integration

## Decisions

### Decision 1: Separate Product Class vs Extending Snowball

**Decision**: Create a separate `PhoenixOption` class rather than extending `SnowballOption`.

**Rationale**:
- Phoenix has fundamentally different coupon mechanics (periodic vs KO-only)
- Cleaner separation of concerns
- Avoids complicating SnowballOption with optional coupon barrier logic
- Both inherit from BaseEquityOption, maintaining consistency

**Alternatives Considered**:
- Extend SnowballOption with optional coupon_barrier: Rejected due to increased complexity and different payoff semantics

### Decision 2: Coupon Barrier Configuration

**Decision**: Create new `CouponBarrierConfig` class for Phoenix-specific settings.

```python
@dataclass(frozen=True)
class CouponBarrierConfig:
    coupon_barrier: Union[float, List[float]]  # Time-varying support
    coupon_rate: float                         # Per-period rate
    coupon_pay_type: CouponPayType             # INSTANT or EXPIRY
    day_count_convention: DayCountConvention   # ACT_365, THIRTY_360_US, etc.
    memory_coupon: bool                        # Accumulate missed coupons
```

**Rationale**:
- Separates coupon-specific settings from barrier settings
- Reuses existing CouponPayType and DayCountConvention enums
- Follows frozen dataclass pattern from SnowballConfig

### Decision 3: Observation Date Sharing

**Decision**: Coupon barrier uses same observation dates as KO barrier.

**Rationale**:
- Matches common market practice
- Simplifies implementation
- Reduces configuration complexity
- Can be extended later if independent schedules needed

### Decision 4: Memory Coupon State Tracking

**Decision**: Track accumulated coupon state as simple float, not per-observation history.

**Rationale**:
- Sufficient for product definition and payoff calculation
- Engines can track detailed history internally
- Keeps product class clean

## Class Hierarchy

```
BaseEquityOption (existing)
    |
    +-- SnowballOption (existing)
    |       - KO barrier, KI barrier
    |       - KO coupon (paid on knockout)
    |       - Rebate (paid at maturity V0)
    |
    +-- PhoenixOption (NEW)
            - KO barrier, KI barrier
            - Coupon barrier (NEW - periodic coupons)
            - Memory coupon (NEW - accumulates missed)
            - Uses day count for coupon calculation
```

## Key Methods

```python
class PhoenixOption(BaseEquityOption):
    # Configuration
    barrier_config: BarrierConfig       # KO/KI (reused from Snowball)
    coupon_config: CouponBarrierConfig  # NEW: Coupon barrier settings
    payoff_config: PayoffConfig         # Maturity payoffs (reused)
    accrual_config: AccrualConfig       # Payment timing (reused)
    airbag_config: AirbagConfig         # Protection (reused)

    # Coupon methods (NEW)
    def is_coupon_triggered(self, spot: float, obs_idx: int) -> bool
    def get_coupon_payoff(self, obs_idx: int) -> float
    def get_coupon_year_fraction(self, start: datetime, end: datetime) -> float

    # Existing methods (similar to Snowball)
    def is_ko_triggered(self, spot: float, obs_idx: int) -> bool
    def is_ki_triggered(self, spot: float, obs_idx: int) -> bool
    def get_maturity_payoff_v0(self, spot: float) -> float
    def get_maturity_payoff_v1(self, spot: float) -> float
```

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Code duplication with Snowball | Extract common barrier logic to shared utilities if needed |
| Memory coupon complexity in engines | Defer engine implementation; product defines semantics only |
| Day count edge cases | Reuse well-tested util/calendar/day_counter.py |

## Migration Plan

1. Create new files without modifying existing Snowball code
2. Export PhoenixOption from option module __init__.py
3. Update snowball-option-helpers spec to reference PhoenixOption for `create_phoenix_snowball()`
4. Add tests and demo script

No breaking changes to existing code.

## Open Questions

None - all clarified with user:
- Observation dates: Same as KO barrier
- Memory coupon: Must-have from start
- Engines: Product only (no engine implementation)
- Variants: All variants (standard, reverse, step-down)
- Payment timing: Support both INSTANT and EXPIRY via CouponPayType
