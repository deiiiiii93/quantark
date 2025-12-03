# Design: SIMM Aggregation Engine

## Context

SIMM margin calculation follows a hierarchical aggregation approach:
1. **Sensitivity Level**: Raw sensitivities weighted by risk weights and concentration factors
2. **Bucket Level**: Aggregate within buckets using intra-bucket correlations
3. **Risk Class Level**: Aggregate across buckets using inter-bucket correlations
4. **Product Class Level**: Aggregate across risk classes using inter-risk-class correlations
5. **Total Level**: Sum across product classes

This design implements all SIMM formulas from Sections B and 5-13 of the specification.

### Stakeholders
- Risk managers calculating regulatory initial margin
- Compliance teams validating SIMM calculations
- Quantitative developers maintaining the calculation engine

### Constraints
- Must implement SIMM formulas exactly as specified
- Must handle all risk classes and margin types
- Must support IR-specific aggregation (different from other risk classes)
- Must be efficient for large portfolios with many sensitivities
- Must track attribution through aggregation levels

## Goals / Non-Goals

### Goals
- Implement all SIMM aggregation formulas per specification
- Support Delta, Vega, Curvature, and Base Correlation margins
- Support all 6 risk classes and 4 product classes
- Provide transparent calculation trace for validation
- Enable what-if analysis (impact of adding/removing trades)

### Non-Goals
- Sensitivity calculation (covered in add-simm-sensitivity-engines)
- Reporting and visualization (covered in add-simm-reporting)
- Parallel processing optimization (future enhancement)

## Decisions

### Decision 1: Module Structure

```
simm/engine/
├── __init__.py               # Public exports
├── concentration.py          # CR, VCR, g_bc calculations
├── weighted_sensitivity.py   # WS calculation
├── bucket_aggregator.py      # K_b calculation
├── risk_class_aggregator.py  # Delta/Vega/CurvatureMargin per risk class
├── product_class_aggregator.py  # SIMM_product calculation
├── addon.py                  # Add-on calculations
└── simm_calculator.py        # Main SIMMCalculator class
```

**Rationale**: Each aggregation level in its own module for testability and clarity.

### Decision 2: Concentration Risk Calculation

```python
def calculate_concentration_risk_factor(
    sensitivities: List[DeltaSensitivity],
    risk_class: RiskClass,
    bucket: Union[int, str],
    threshold: float,
) -> float:
    """
    Calculate concentration risk factor CR.
    
    For IR (paragraph 7):
        CR_b = max(1, sqrt(|Σ s_k,i| / T_b))
        
    For Credit (paragraph 8):
        CR_k = max(1, sqrt(|Σ_j s_j| / T_b))
        where j sums over same issuer/seniority
        
    For Equity/Commodity/FX (paragraph 8):
        CR_k = max(1, sqrt(|s_k| / T_b))
    """
    if risk_class == RiskClass.INTEREST_RATE:
        # Sum all sensitivities in bucket (currency)
        total_sens = sum(s.amount for s in sensitivities)
        return max(1.0, math.sqrt(abs(total_sens) / threshold))
    
    elif risk_class in [RiskClass.CREDIT_QUALIFYING, RiskClass.CREDIT_NON_QUALIFYING]:
        # Group by issuer/seniority, compute CR per group
        grouped = group_by_issuer_seniority(sensitivities)
        cr_values = {}
        for key, group in grouped.items():
            total = sum(s.amount for s in group)
            cr_values[key] = max(1.0, math.sqrt(abs(total) / threshold))
        return cr_values  # Return dict for per-factor CR
    
    else:  # Equity, Commodity, FX
        # CR per individual risk factor
        cr_values = {}
        for s in sensitivities:
            cr_values[s.qualifier] = max(1.0, math.sqrt(abs(s.amount) / threshold))
        return cr_values
```

**Rationale**: Different CR formulas per risk class per SIMM spec.

### Decision 3: Weighted Sensitivity Calculation

```python
def calculate_weighted_sensitivity(
    sensitivity: DeltaSensitivity,
    risk_weight: float,
    concentration_factor: float,
) -> float:
    """
    Calculate weighted sensitivity WS.
    
    WS_k = RW_k × s_k × CR_k
    
    For IR cross-currency basis, CR should not be applied.
    """
    return risk_weight * sensitivity.amount * concentration_factor
```

### Decision 4: Bucket Aggregation (K_b)

```python
def aggregate_bucket(
    weighted_sensitivities: Dict[str, float],  # risk_factor -> WS
    correlation_matrix: Callable[[str, str], float],
    concentration_factors: Dict[str, float],
) -> float:
    """
    Aggregate weighted sensitivities within a bucket.
    
    K_b = sqrt(Σ_k WS_k² + Σ_k Σ_{l≠k} ρ_kl × f_kl × WS_k × WS_l)
    
    where f_kl = min(CR_k, CR_l) / max(CR_k, CR_l)
    """
    risk_factors = list(weighted_sensitivities.keys())
    n = len(risk_factors)
    
    # Diagonal terms
    sum_sq = sum(ws**2 for ws in weighted_sensitivities.values())
    
    # Cross terms
    cross_sum = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rf_i, rf_j = risk_factors[i], risk_factors[j]
            rho = correlation_matrix(rf_i, rf_j)
            
            cr_i = concentration_factors.get(rf_i, 1.0)
            cr_j = concentration_factors.get(rf_j, 1.0)
            f_ij = min(cr_i, cr_j) / max(cr_i, cr_j)
            
            ws_i = weighted_sensitivities[rf_i]
            ws_j = weighted_sensitivities[rf_j]
            
            cross_sum += 2 * rho * f_ij * ws_i * ws_j
    
    return math.sqrt(max(0, sum_sq + cross_sum))
```

**Rationale**: Implements paragraph 8(c) formula with concentration adjustment f_kl.

### Decision 5: Risk Class Aggregation (DeltaMargin)

```python
def aggregate_risk_class(
    bucket_results: Dict[Union[int, str], float],  # bucket -> K_b
    bucket_sums: Dict[Union[int, str], float],     # bucket -> Σ WS
    inter_bucket_correlation: Callable[[Union[int, str], Union[int, str]], float],
    residual_bucket_result: Optional[float] = None,
) -> float:
    """
    Aggregate across buckets within a risk class.
    
    DeltaMargin = sqrt(Σ_b K_b² + Σ_b Σ_{c≠b} γ_bc × S_b × S_c) + K_residual
    
    where S_b = max(min(Σ WS_k, K_b), -K_b)
    """
    buckets = [b for b in bucket_results.keys() if b != "Residual"]
    
    # Calculate S_b for each bucket
    s_values = {}
    for bucket in buckets:
        ws_sum = bucket_sums.get(bucket, 0.0)
        k_b = bucket_results[bucket]
        s_values[bucket] = max(min(ws_sum, k_b), -k_b)
    
    # Diagonal terms
    sum_sq = sum(k**2 for k in bucket_results.values() if k != "Residual")
    
    # Cross terms
    cross_sum = 0.0
    for i, b1 in enumerate(buckets):
        for b2 in buckets[i+1:]:
            gamma = inter_bucket_correlation(b1, b2)
            cross_sum += 2 * gamma * s_values[b1] * s_values[b2]
    
    margin = math.sqrt(max(0, sum_sq + cross_sum))
    
    # Add residual bucket
    if residual_bucket_result:
        margin += residual_bucket_result
    
    return margin
```

**Rationale**: Implements paragraph 8(d) formula.

### Decision 6: IR-Specific Aggregation

Interest Rate aggregation differs from other risk classes:
1. Buckets are currencies, not numbered buckets
2. Concentration risk applies at currency level
3. g_bc factor for inter-currency aggregation

```python
def aggregate_ir_risk_class(
    currency_results: Dict[str, float],    # currency -> K
    currency_cr: Dict[str, float],         # currency -> CR
    currency_ws_sums: Dict[str, float],    # currency -> Σ WS
    inter_currency_correlation: float = 0.32,
) -> float:
    """
    IR-specific aggregation (paragraph 7).
    
    DeltaMargin = sqrt(Σ_b K_b² + Σ_b Σ_{c≠b} γ_bc × g_bc × S_b × S_c)
    
    where g_bc = min(CR_b, CR_c) / max(CR_b, CR_c)
    """
    currencies = list(currency_results.keys())
    
    # Calculate S_b
    s_values = {}
    for ccy in currencies:
        ws_sum = currency_ws_sums.get(ccy, 0.0)
        k = currency_results[ccy]
        s_values[ccy] = max(min(ws_sum, k), -k)
    
    # Diagonal
    sum_sq = sum(k**2 for k in currency_results.values())
    
    # Cross with g_bc
    cross_sum = 0.0
    for i, c1 in enumerate(currencies):
        for c2 in currencies[i+1:]:
            g_bc = min(currency_cr[c1], currency_cr[c2]) / max(currency_cr[c1], currency_cr[c2])
            cross_sum += 2 * inter_currency_correlation * g_bc * s_values[c1] * s_values[c2]
    
    return math.sqrt(max(0, sum_sq + cross_sum))
```

**Rationale**: IR uses different formula per paragraph 7(d).

### Decision 7: Curvature Margin Calculation

Curvature uses squared correlations and theta/lambda factors:

```python
def calculate_curvature_margin(
    cvr_values: Dict[str, Dict[str, float]],  # bucket -> {risk_factor: CVR}
    bucket_correlations: Dict[str, Callable],
    inter_bucket_correlations: Callable,
) -> float:
    """
    Calculate curvature margin per paragraph 11.
    
    θ = min(Σ CVR / Σ|CVR|, 0)
    λ = (Φ^-1(99.5%)² - 1)(1 + θ) - θ
    
    CurvatureMargin_non-res = max(Σ CVR + λ × sqrt(Σ K_b² + Σ γ_bc² × S_b × S_c), 0)
    """
    # Calculate K_b for each bucket (using ρ² not ρ)
    bucket_k = {}
    for bucket, cvrs in cvr_values.items():
        if bucket == "Residual":
            continue
        
        sum_sq = sum(cvr**2 for cvr in cvrs.values())
        cross = 0.0
        factors = list(cvrs.keys())
        for i, f1 in enumerate(factors):
            for f2 in factors[i+1:]:
                rho = bucket_correlations[bucket](f1, f2)
                cross += 2 * (rho**2) * cvrs[f1] * cvrs[f2]
        
        bucket_k[bucket] = math.sqrt(max(0, sum_sq + cross))
    
    # Calculate S_b
    s_values = {}
    for bucket, cvrs in cvr_values.items():
        if bucket == "Residual":
            continue
        cvr_sum = sum(cvrs.values())
        k_b = bucket_k[bucket]
        s_values[bucket] = max(min(cvr_sum, k_b), -k_b)
    
    # Calculate total CVR
    total_cvr = sum(sum(cvrs.values()) for cvrs in cvr_values.values())
    total_abs_cvr = sum(sum(abs(v) for v in cvrs.values()) for cvrs in cvr_values.values())
    
    # θ and λ
    theta = min(total_cvr / total_abs_cvr, 0) if total_abs_cvr > 0 else 0
    phi_inv = 2.576  # Φ^-1(99.5%)
    lambda_val = (phi_inv**2 - 1) * (1 + theta) - theta
    
    # Cross-bucket (using γ²)
    buckets = list(bucket_k.keys())
    sum_k_sq = sum(k**2 for k in bucket_k.values())
    cross = 0.0
    for i, b1 in enumerate(buckets):
        for b2 in buckets[i+1:]:
            gamma = inter_bucket_correlations(b1, b2)
            cross += 2 * (gamma**2) * s_values[b1] * s_values[b2]
    
    margin = max(total_cvr + lambda_val * math.sqrt(max(0, sum_k_sq + cross)), 0)
    
    return margin
```

**Rationale**: Curvature uses ρ² and γ² per paragraph 11.

### Decision 8: Product Class Aggregation

```python
def aggregate_product_class(
    risk_class_margins: Dict[RiskClass, float],
    inter_risk_correlations: np.ndarray,
) -> float:
    """
    Aggregate across risk classes within a product class.
    
    SIMM_product = sqrt(Σ_r IM_r² + Σ_r Σ_{s≠r} ψ_rs × IM_r × IM_s)
    """
    risk_classes = list(risk_class_margins.keys())
    margins = list(risk_class_margins.values())
    
    # Diagonal
    sum_sq = sum(m**2 for m in margins)
    
    # Cross terms
    cross = 0.0
    for i, rc1 in enumerate(risk_classes):
        for j, rc2 in enumerate(risk_classes[i+1:], i+1):
            psi = get_inter_risk_class_correlation(rc1, rc2)
            cross += 2 * psi * margins[i] * margins[j]
    
    return math.sqrt(max(0, sum_sq + cross))
```

### Decision 9: Main Calculator Class

```python
class SIMMCalculator:
    """Main SIMM calculation engine."""
    
    def __init__(self, config: SIMMConfig):
        self.config = config
        self.concentration_calc = ConcentrationCalculator()
        self.bucket_agg = BucketAggregator()
        self.risk_class_agg = RiskClassAggregator()
        self.product_class_agg = ProductClassAggregator()
    
    def calculate(
        self,
        sensitivities: SensitivityCollection,
    ) -> SIMMResult:
        """
        Calculate total SIMM and all components.
        
        Returns detailed result with attribution by:
        - Product class
        - Risk class
        - Margin type
        - Bucket
        """
        results = {}
        
        # Group sensitivities by product class
        by_product = sensitivities.group_by_product_class()
        
        for product_class, pc_sensitivities in by_product.items():
            # Calculate margin for each risk class
            risk_class_margins = {}
            
            for risk_class in RiskClass:
                rc_sensitivities = pc_sensitivities.filter_by_risk_class(risk_class)
                if not rc_sensitivities:
                    continue
                
                # Delta margin
                delta_margin = self._calculate_delta_margin(
                    risk_class, rc_sensitivities.deltas
                )
                
                # Vega margin
                vega_margin = self._calculate_vega_margin(
                    risk_class, rc_sensitivities.vegas
                ) if self.config.calculate_vega else 0
                
                # Curvature margin
                curvature_margin = self._calculate_curvature_margin(
                    risk_class, rc_sensitivities.curvatures
                ) if self.config.calculate_curvature else 0
                
                # Base correlation (Credit Q only)
                base_corr_margin = self._calculate_base_corr_margin(
                    rc_sensitivities.base_corrs
                ) if risk_class == RiskClass.CREDIT_QUALIFYING else 0
                
                risk_class_margins[risk_class] = (
                    delta_margin + vega_margin + curvature_margin + base_corr_margin
                )
            
            # Aggregate across risk classes
            product_simm = self.product_class_agg.aggregate(risk_class_margins)
            
            # Apply multiplier if configured
            multiplier = self._get_multiplier(product_class)
            results[product_class] = product_simm * multiplier
        
        # Sum across product classes
        total_simm = sum(results.values())
        
        # Add add-ons
        addon = self._calculate_addon(sensitivities)
        total_simm += addon
        
        return SIMMResult(
            total=total_simm,
            by_product_class=results,
            addon=addon,
            # ... detailed attribution
        )
```

**Rationale**: Central calculator orchestrates all aggregation levels.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Numerical precision in sqrt | Use max(0, x) before sqrt |
| Large correlation matrices | NumPy for efficiency |
| Complex IR aggregation | Separate IR-specific path |

## Migration Plan

No migration required - new engine module.

