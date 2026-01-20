# Design: SIMM Sensitivity Engines

## Context

SIMM sensitivities are the inputs to margin calculation. Each risk class has specific sensitivity definitions:
- **Delta**: First-order sensitivity to risk factor changes (PV01, CS01, equity delta, etc.)
- **Vega**: Sensitivity to implied volatility changes
- **Curvature**: Second-order volatility sensitivity (captures gamma effect for options)

This design defines how sensitivities are calculated from portfolio positions and how they integrate with the existing QuantArk pricing infrastructure.

### Stakeholders
- Quantitative developers implementing sensitivity calculation
- Risk managers validating sensitivity accuracy
- Trading desks using auto-calculated sensitivities

### Constraints
- Must calculate sensitivities per SIMM Section C definitions
- Must integrate with existing GreeksCalculator for equity derivatives
- Must support both auto-calculation and CRIF import
- Must handle multi-currency portfolios
- Must be efficient for large portfolios (batch calculation)

## Goals / Non-Goals

### Goals
- Implement sensitivity calculation for all 6 risk classes
- Support Delta, Vega, and Curvature margin types
- Integrate with existing portfolio and pricing infrastructure
- Provide clear mapping from products to SIMM risk factors
- Support bucket classification for each risk class

### Non-Goals
- Market data sourcing (uses existing PricingEnvironment)
- Historical sensitivity analysis
- Sensitivity approximation methods (full revaluation only)

## Decisions

### Decision 1: Module Structure

```
simm/sensitivity/
├── __init__.py           # Public exports
├── base.py               # Protocols, base classes
├── ir_engine.py          # Interest Rate sensitivities
├── credit_engine.py      # Credit Q and NQ sensitivities
├── equity_engine.py      # Equity sensitivities
├── commodity_engine.py   # Commodity sensitivities
├── fx_engine.py          # FX sensitivities
├── vega_engine.py        # Cross-risk-class vega
├── curvature_engine.py   # Cross-risk-class curvature
└── portfolio_adapter.py  # Portfolio-to-sensitivity conversion
```

**Rationale**: Separate engine per risk class for focused testing and maintenance.

### Decision 2: Sensitivity Engine Protocol

```python
@runtime_checkable
class SensitivityEngine(Protocol):
    """Protocol for risk-class-specific sensitivity engines."""
    
    risk_class: RiskClass
    
    def calculate_delta(
        self,
        position: Union[EquityPosition, FIPosition],
        pricing_env: PricingEnvironment,
    ) -> List[DeltaSensitivity]:
        """Calculate delta sensitivities for a single position."""
        ...
    
    def calculate_vega(
        self,
        position: Union[EquityPosition, FIPosition],
        pricing_env: PricingEnvironment,
    ) -> List[VegaSensitivity]:
        """Calculate vega sensitivities for a single position."""
        ...
    
    def calculate_curvature(
        self,
        position: Union[EquityPosition, FIPosition],
        pricing_env: PricingEnvironment,
    ) -> List[CurvatureSensitivity]:
        """Calculate curvature sensitivities for a single position."""
        ...
    
    def supports_position(self, position: Any) -> bool:
        """Check if this engine can process the position."""
        ...
```

**Rationale**: Protocol-based design for flexibility. Each engine handles one risk class.

### Decision 3: Sensitivity Dataclasses

```python
@dataclass
class DeltaSensitivity:
    """Delta sensitivity for SIMM calculation."""
    risk_class: RiskClass
    risk_type: SensitivityType
    qualifier: str           # Currency, issuer, equity name, etc.
    bucket: Union[int, str]  # Bucket number or "Residual"
    label1: str              # Tenor for IR/Credit, empty for others
    label2: str              # Sub-curve for IR, empty for others
    amount: float            # Sensitivity value
    amount_currency: str     # Currency of amount
    
    # Optional metadata
    position_id: Optional[str] = None
    trade_id: Optional[str] = None

@dataclass
class VegaSensitivity:
    """Vega sensitivity for SIMM calculation."""
    risk_class: RiskClass
    risk_type: SensitivityType
    qualifier: str
    bucket: Union[int, str]
    option_expiry: float     # Option expiry in years
    amount: float            # Vol-weighted vega (VR_ik)
    amount_currency: str
    
    position_id: Optional[str] = None

@dataclass  
class CurvatureSensitivity:
    """Curvature sensitivity for SIMM calculation."""
    risk_class: RiskClass
    risk_type: SensitivityType
    qualifier: str
    bucket: Union[int, str]
    cvr: float               # Curvature risk (CVR_ik)
    amount_currency: str
    
    position_id: Optional[str] = None
```

**Rationale**: Captures all SIMM-required dimensions. Optional metadata for attribution.

### Decision 4: Interest Rate Sensitivity Calculation

For IR delta (PV01), use the existing pricing infrastructure:

```python
class IRSensitivityEngine:
    """Interest Rate sensitivity engine."""
    
    def calculate_delta(
        self,
        position: FIPosition,
        pricing_env: PricingEnvironment,
    ) -> List[DeltaSensitivity]:
        """
        Calculate IR delta sensitivities by tenor and sub-curve.
        
        For each IR tenor vertex (2w, 1m, ..., 30yr):
        1. Bump the relevant rate by 1bp
        2. Reprice the position
        3. Record sensitivity s = V(r + 1bp) - V(r)
        """
        sensitivities = []
        currency = position.product.currency
        
        for tenor_label, tenor_years in zip(IR_TENOR_LABELS, IR_TENORS):
            # Calculate PV01 at this tenor
            pv01 = self._calculate_pv01_at_tenor(
                position, pricing_env, tenor_years
            )
            
            if abs(pv01) > 1e-10:  # Non-trivial sensitivity
                sensitivities.append(DeltaSensitivity(
                    risk_class=RiskClass.INTEREST_RATE,
                    risk_type=SensitivityType.RISK_IR_CURVE,
                    qualifier=currency,
                    bucket=currency,  # IR buckets by currency
                    label1=tenor_label,
                    label2="OIS",  # Default sub-curve
                    amount=pv01,
                    amount_currency=currency,
                    position_id=position.position_id
                ))
        
        return sensitivities
```

**Rationale**: Bump-and-reprice using existing engines. Matches SIMM Section C.1 definition.

### Decision 5: Equity Sensitivity Calculation

Leverage existing GreeksCalculator:

```python
class EquitySensitivityEngine:
    """Equity sensitivity engine."""
    
    def __init__(self):
        self.greeks_calculator = GreeksCalculator()
    
    def calculate_delta(
        self,
        position: EquityPosition,
        pricing_env: PricingEnvironment,
    ) -> List[DeltaSensitivity]:
        """
        Calculate equity delta sensitivity.
        
        s = V(S + 1%) - V(S)  (per SIMM C.2)
        """
        # Use existing Greeks calculator
        greeks = position.get_greeks(
            pricing_env, 
            self.greeks_calculator,
            use_analytical=True
        )
        
        # Convert delta to 1% shock format
        # Greeks delta is per $1 spot move
        # SIMM wants per 1% spot move
        spot = pricing_env.spot
        simm_delta = greeks['delta'] * spot * 0.01
        
        # Classify into bucket
        bucket = self._classify_equity_bucket(position.underlying, spot)
        
        return [DeltaSensitivity(
            risk_class=RiskClass.EQUITY,
            risk_type=SensitivityType.RISK_EQUITY,
            qualifier=position.underlying,
            bucket=bucket,
            label1="",
            label2="",
            amount=simm_delta,
            amount_currency=pricing_env.calculation_currency or "USD",
            position_id=position.position_id
        )]
    
    def _classify_equity_bucket(self, underlying: str, market_cap: float) -> int:
        """Classify equity into SIMM bucket based on size, region, sector."""
        # Implementation based on SIMM Section G rules
        ...
```

**Rationale**: Reuses existing Greeks infrastructure. Bucket classification per SIMM rules.

### Decision 6: Vega Sensitivity Calculation

SIMM vega is vol-weighted vega:

```python
def calculate_vega(
    self,
    position: EquityPosition,
    pricing_env: PricingEnvironment,
) -> List[VegaSensitivity]:
    """
    Calculate vega sensitivity per SIMM Section C.3.
    
    VR_ik = HVR_c × Σ_j σ_kj × (∂V/∂σ)
    
    where σ_kj = RW_k × sqrt(365/14) / α
    """
    if not self._has_optionality(position):
        return []  # No vega for non-options
    
    greeks = position.get_greeks(pricing_env, self.greeks_calculator)
    raw_vega = greeks.get('vega', 0.0)
    
    if abs(raw_vega) < 1e-10:
        return []
    
    # Get vol estimate from delta risk weight
    bucket = self._classify_equity_bucket(position.underlying, pricing_env.spot)
    rw = get_equity_risk_weight(bucket)
    alpha = stats.norm.ppf(0.99)
    sigma = rw * math.sqrt(365/14) / alpha
    
    # Vol-weighted vega
    hvr = get_hvr(RiskClass.EQUITY)
    vr = hvr * sigma * raw_vega
    
    # Map to expiry tenor bucket
    expiry = position.product.get_maturity(pricing_env)
    expiry_bucket = self._map_to_tenor_bucket(expiry)
    
    return [VegaSensitivity(
        risk_class=RiskClass.EQUITY,
        risk_type=SensitivityType.RISK_EQUITY_VOL,
        qualifier=position.underlying,
        bucket=bucket,
        option_expiry=expiry,
        amount=vr,
        amount_currency=pricing_env.calculation_currency or "USD",
        position_id=position.position_id
    )]
```

**Rationale**: Follows SIMM Section 10 formula exactly.

### Decision 7: Curvature Sensitivity Calculation

Curvature uses scaling function SF(t):

```python
def calculate_curvature(
    self,
    position: EquityPosition,
    pricing_env: PricingEnvironment,
) -> List[CurvatureSensitivity]:
    """
    Calculate curvature sensitivity per SIMM Section 11.
    
    CVR_ik = Σ_j SF(t_kj) × σ_kj × (∂V/∂σ)
    
    where SF(t) = 0.5 × min(1, 14/t)
    """
    if not self._has_optionality(position):
        return []
    
    greeks = position.get_greeks(pricing_env, self.greeks_calculator)
    raw_vega = greeks.get('vega', 0.0)
    
    if abs(raw_vega) < 1e-10:
        return []
    
    # Calculate scaling function
    expiry_days = position.product.get_maturity(pricing_env) * 365
    sf = 0.5 * min(1.0, 14.0 / max(expiry_days, 1))
    
    # Same sigma as vega
    bucket = self._classify_equity_bucket(position.underlying, pricing_env.spot)
    rw = get_equity_risk_weight(bucket)
    alpha = stats.norm.ppf(0.99)
    sigma = rw * math.sqrt(365/14) / alpha
    
    cvr = sf * sigma * raw_vega
    
    return [CurvatureSensitivity(
        risk_class=RiskClass.EQUITY,
        risk_type=SensitivityType.RISK_EQUITY_VOL,
        qualifier=position.underlying,
        bucket=bucket,
        cvr=cvr,
        amount_currency=pricing_env.calculation_currency or "USD",
        position_id=position.position_id
    )]
```

**Rationale**: Implements SIMM Section 11 scaling function.

### Decision 8: Portfolio Adapter

Unified interface to calculate all sensitivities from a portfolio:

```python
class PortfolioSensitivityAdapter:
    """Calculates SIMM sensitivities from portfolio positions."""
    
    def __init__(self, config: SIMMConfig):
        self.config = config
        self.engines = {
            RiskClass.INTEREST_RATE: IRSensitivityEngine(),
            RiskClass.CREDIT_QUALIFYING: CreditSensitivityEngine(),
            RiskClass.CREDIT_NON_QUALIFYING: CreditSensitivityEngine(),
            RiskClass.EQUITY: EquitySensitivityEngine(),
            RiskClass.COMMODITY: CommoditySensitivityEngine(),
            RiskClass.FX: FXSensitivityEngine(),
        }
    
    def calculate_sensitivities(
        self,
        portfolio: Union[EquityPortfolio, FIPortfolio],
        pricing_envs: Dict[str, PricingEnvironment],
    ) -> SensitivityCollection:
        """
        Calculate all SIMM sensitivities for a portfolio.
        
        Args:
            portfolio: Portfolio with positions
            pricing_envs: Map of underlying -> PricingEnvironment
            
        Returns:
            SensitivityCollection with all Delta, Vega, Curvature sensitivities
        """
        all_sensitivities = SensitivityCollection()
        
        for position_id, position in portfolio.positions.items():
            underlying = position.underlying
            pricing_env = pricing_envs.get(underlying)
            
            if pricing_env is None:
                raise MarketDataError(f"No pricing env for {underlying}")
            
            # Determine applicable engines based on position type
            engines = self._get_engines_for_position(position)
            
            for engine in engines:
                if self.config.calculate_delta:
                    deltas = engine.calculate_delta(position, pricing_env)
                    all_sensitivities.add_deltas(deltas)
                
                if self.config.calculate_vega:
                    vegas = engine.calculate_vega(position, pricing_env)
                    all_sensitivities.add_vegas(vegas)
                
                if self.config.calculate_curvature:
                    curvatures = engine.calculate_curvature(position, pricing_env)
                    all_sensitivities.add_curvatures(curvatures)
        
        return all_sensitivities
```

**Rationale**: Central adapter handles portfolio traversal. Per-position calculation for flexibility.

### Decision 9: FX Sensitivity for Non-Base Currency Positions

Positions denominated in non-calculation currencies generate FX delta:

```python
class FXSensitivityEngine:
    """FX sensitivity engine."""
    
    def calculate_delta_from_position_value(
        self,
        position: Union[EquityPosition, FIPosition],
        pricing_env: PricingEnvironment,
        calculation_currency: str,
    ) -> List[DeltaSensitivity]:
        """
        Calculate FX delta from position value in foreign currency.
        
        FX sensitivity = position value in foreign currency × 1%
        """
        position_currency = self._get_position_currency(position)
        
        if position_currency == calculation_currency:
            return []  # No FX risk for same currency
        
        market_value = position.get_market_value(pricing_env)
        fx_delta = market_value * 0.01  # 1% FX shock
        
        return [DeltaSensitivity(
            risk_class=RiskClass.FX,
            risk_type=SensitivityType.RISK_FX,
            qualifier=position_currency,
            bucket="",  # FX has single bucket
            label1="",
            label2="",
            amount=fx_delta,
            amount_currency=calculation_currency,
            position_id=position.position_id
        )]
```

**Rationale**: FX translation risk per SIMM Section C.2.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Slow bump-and-reprice for large portfolios | Batch calculation, caching |
| Bucket classification requires external data | Configurable lookup tables |
| Missing market data for some underlyings | Clear error messages |

## Migration Plan

No migration required - new sensitivity module.

