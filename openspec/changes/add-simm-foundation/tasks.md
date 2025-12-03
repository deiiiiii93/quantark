# Tasks: Add SIMM Foundation Module

## 1. Module Structure Setup
- [ ] 1.1 Create `simm/__init__.py` with public API exports
- [ ] 1.2 Create `simm/crif/__init__.py`

## 2. Taxonomy Implementation
- [ ] 2.1 Create `simm/taxonomy.py` with RiskClass enum
- [ ] 2.2 Add ProductClass enum to taxonomy.py
- [ ] 2.3 Add MarginType enum to taxonomy.py
- [ ] 2.4 Add SensitivityType enum (Risk_IRCurve, Risk_FX, etc.)
- [ ] 2.5 Add IR tenor definitions (IR_TENORS, IR_TENOR_LABELS)
- [ ] 2.6 Add Credit tenor definitions (CREDIT_TENORS, CREDIT_TENOR_LABELS)
- [ ] 2.7 Add currency classifications (regular, low, high volatility)
- [ ] 2.8 Add bucket definitions for IR (IRBucket dataclass)
- [ ] 2.9 Add bucket definitions for Credit Qualifying (12 buckets + residual)
- [ ] 2.10 Add bucket definitions for Credit Non-Qualifying (2 buckets + residual)
- [ ] 2.11 Add bucket definitions for Equity (12 buckets + residual)
- [ ] 2.12 Add bucket definitions for Commodity (17 buckets)
- [ ] 2.13 Add bucket definitions for FX (single bucket)
- [ ] 2.14 Add IR sub-curve definitions (OIS, Libor1m, Libor3m, Libor6m, Libor12m, Prime, Municipal)

## 3. Sensitivity Data Models
- [ ] 3.1 Create `simm/sensitivity.py` with Sensitivity protocol
- [ ] 3.2 Add DeltaSensitivity dataclass
- [ ] 3.3 Add VegaSensitivity dataclass
- [ ] 3.4 Add CurvatureSensitivity dataclass
- [ ] 3.5 Add BaseCorrSensitivity dataclass (Credit Qualifying only)
- [ ] 3.6 Add SensitivityCollection class to hold grouped sensitivities

## 4. CRIF Implementation
- [ ] 4.1 Create `simm/crif/models.py` with CRIFRecord dataclass
- [ ] 4.2 Add CRIFHeader dataclass for file metadata
- [ ] 4.3 Create `simm/crif/parser.py` with CSV parsing
- [ ] 4.4 Add CRIF validation logic (required fields, enum validation)
- [ ] 4.5 Add CRIF-to-Sensitivity conversion functions
- [ ] 4.6 Add Sensitivity-to-CRIF export functions

## 5. Configuration
- [ ] 5.1 Create `simm/config.py` with SIMMConfig dataclass
- [ ] 5.2 Add SIMMVersion enum (v2.5, v2.6 for future versions)
- [ ] 5.3 Add validation methods to SIMMConfig

## 6. Testing
- [ ] 6.1 Create `test/test_simm_taxonomy.py` with enum tests
- [ ] 6.2 Add bucket definition tests
- [ ] 6.3 Create `test/test_simm_crif.py` with parser tests
- [ ] 6.4 Add CRIF validation tests
- [ ] 6.5 Create `test/test_simm_config.py` with config tests

## 7. Documentation
- [ ] 7.1 Add docstrings to all public classes and functions
- [ ] 7.2 Create `simm/README.md` with module overview

