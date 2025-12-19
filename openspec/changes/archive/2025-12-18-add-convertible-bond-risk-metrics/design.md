# Design: Convertible Bond Risk Metrics

## Overview
This document outlines the technical design for adding DV01, duration, convexity, and floor bond metrics to the `ConvertibleBondEngine`.

## Floor Bond Definition
The **floor bond** (also called "investment value" or "straight bond value") represents the value of a convertible bond assuming:
- No conversion option exercised
- No call option exercised by issuer
- No put option exercised by holder
- Bond pays all scheduled coupons and principal at maturity
- Discounted using the risky rate (risk-free + credit spread)

This gives a lower bound for the convertible's value in scenarios where the stock price is very low.

## Calculation Methods

### Floor Bond Price
```
floor_bond_price = Σ (cashflow_i × e^(-(r + s) × t_i))
```
Where:
- `r` = risk-free rate
- `s` = credit spread from product
- `t_i` = time to cashflow i
- `cashflow_i` = coupon or principal payment

### Floor Bond DV01/CS01/Duration/Convexity
Since the floor bond has no optionality, these can be computed analytically:

```python
# DV01 = Modified Duration × Price × 0.0001
floor_dv01 = floor_mod_duration * floor_price * 0.0001

# CS01 = Modified Duration × Price × 0.0001 (same as DV01 for floor bond)
# Because floor bond discounts at (r + s), a 1bp move in s has the same effect as 1bp in r
floor_cs01 = floor_mod_duration * floor_price * 0.0001

# Modified Duration = Σ(PV_i × t_i) / Price
floor_mod_duration = sum(pv_i * t_i for i in cashflows) / floor_price

# Convexity = Σ(PV_i × t_i²) / Price  
floor_convexity = sum(pv_i * t_i**2 for i in cashflows) / floor_price
```

### Convertible Bond DV01/CS01/Duration/Convexity
For the full convertible (with embedded options), we use numerical bumping:

```python
# DV01 via bump-and-reprice (bump risk-free rate only)
rate_bump = 0.0001  # 1 basis point
price_up = reprice_with_rate(base_rate + rate_bump)
price_down = reprice_with_rate(base_rate - rate_bump)
dv01 = (price_down - price_up) / 2  # Central difference, note: price falls when rate rises

# CS01 via bump-and-reprice (bump credit spread only)
spread_bump = 0.0001  # 1 basis point
price_up_cs = reprice_with_spread(base_spread + spread_bump)
price_down_cs = reprice_with_spread(base_spread - spread_bump)
cs01 = (price_down_cs - price_up_cs) / 2  # Price falls when spread rises

# Duration from DV01
modified_duration = dv01 / (base_price * 0.0001)

# Convexity via central difference
convexity = (price_up - 2*base_price + price_down) / (base_price * rate_bump**2)
```

## API Design

### New Methods on ConvertibleBondEngine

```python
def floor_bond_price(self, bond: ConvertibleBond) -> float:
    """
    Calculate the floor bond (straight bond) price.
    
    The floor bond is the value of the convertible assuming no 
    conversion and no exercise of call/put options.
    
    Args:
        bond: Convertible bond product
        
    Returns:
        Floor bond price (dirty price)
    """

def floor_bond_dv01(self, bond: ConvertibleBond) -> float:
    """
    Calculate DV01 of the floor bond.
    
    Args:
        bond: Convertible bond product
        
    Returns:
        Floor bond DV01 (price change per basis point)
    """

def floor_bond_duration(self, bond: ConvertibleBond) -> float:
    """
    Calculate modified duration of the floor bond.
    
    Args:
        bond: Convertible bond product
        
    Returns:
        Floor bond modified duration
    """

def floor_bond_convexity(self, bond: ConvertibleBond) -> float:
    """
    Calculate convexity of the floor bond.
    
    Args:
        bond: Convertible bond product
        
    Returns:
        Floor bond convexity
    """

def floor_bond_cs01(self, bond: ConvertibleBond) -> float:
    """
    Calculate CS01 of the floor bond.
    
    For floor bond, CS01 equals DV01 since both rate and spread
    affect discounting identically.
    
    Args:
        bond: Convertible bond product
        
    Returns:
        Floor bond CS01 (price change per basis point spread move)
    """

def dv01(self, bond: ConvertibleBond) -> float:
    """
    Calculate DV01 of the convertible bond.
    
    Uses numerical rate bumping since the convertible has embedded options.
    
    Args:
        bond: Convertible bond product
        
    Returns:
        Convertible DV01 (price change per basis point)
    """

def cs01(self, bond: ConvertibleBond) -> float:
    """
    Calculate CS01 of the convertible bond.
    
    Uses numerical credit spread bumping. This measures the sensitivity
    to credit spread changes, which affects both the discount rate and
    hazard rate in credit-adjusted models.
    
    Args:
        bond: Convertible bond product
        
    Returns:
        Convertible CS01 (price change per basis point spread move)
    """

def modified_duration(self, bond: ConvertibleBond) -> float:
    """
    Calculate modified duration of the convertible bond.
    
    Args:
        bond: Convertible bond product
        
    Returns:
        Convertible modified duration
    """

def convexity(self, bond: ConvertibleBond) -> float:
    """
    Calculate convexity of the convertible bond.
    
    Args:
        bond: Convertible bond product
        
    Returns:
        Convertible convexity
    """
```

### Extended ConvertibleBondResult

```python
@dataclass
class ConvertibleBondResult:
    # Existing fields
    price: float
    dirty_price: float
    delta: float = 0.0
    gamma: float = 0.0
    conversion_probability: float = 0.0
    equity_component: float = 0.0
    bond_component: float = 0.0
    default_probability: float = 0.0
    method: str = ""
    
    # New floor bond fields
    floor_bond_price: float = 0.0
    floor_bond_dv01: float = 0.0
    floor_bond_cs01: float = 0.0
    floor_bond_duration: float = 0.0
    floor_bond_convexity: float = 0.0
    
    # New convertible risk fields
    dv01: float = 0.0
    cs01: float = 0.0
    modified_duration: float = 0.0
    convexity: float = 0.0
```

## Implementation Notes

### Rate Bumping Approach
When bumping rates for DV01 calculation:
1. Create a copy of the pricing environment
2. Replace the rate curve with a bumped version
3. Create a new engine instance with the bumped environment
4. Reprice the convertible bond

This approach ensures consistency with the existing pricing logic.

### Credit Spread Handling
- For floor bond: use `bond.credit_spread` from product attributes
- For convertible DV01: bump only the risk-free rate, not the credit spread
- This isolates interest rate risk from credit risk

### Performance Considerations
- Floor bond metrics are cheap (analytical calculation)
- Convertible metrics require 2-3 full repricings (one for each bump direction)
- Consider making risk metrics optional in `price_with_details()` via a flag
- Cache base price to avoid redundant calculations

## Consistency with Existing Code
This design follows the same pattern as:
- `BondDiscountEngine.dv01()`, `modified_duration()`, `convexity()`
- `BondGreeksCalculator._calculate_dv01_fdm()`
- Existing `ConvertibleBondEngine.calculate_delta()` (numerical bumping pattern)
