# Tasks: Add SIMM Foundation Module

## 1. Module Structure Setup
- [x] 1.1 Create `simm/__init__.py` with public API exports
- [x] 1.2 Create `simm/crif/__init__.py`

## 2. Taxonomy Implementation
- [x] 2.1 Create `simm/taxonomy.py` with RiskClass enum
- [x] 2.2 Add ProductClass enum to taxonomy.py
- [x] 2.3 Add MarginType enum to taxonomy.py
- [x] 2.4 Add SensitivityType enum (Risk_IRCurve, Risk_FX, etc.)
- [x] 2.5 Add IR tenor definitions (IR_TENORS, IR_TENOR_LABELS)
- [x] 2.6 Add Credit tenor definitions (CREDIT_TENORS, CREDIT_TENOR_LABELS)
- [x] 2.7 Add currency classifications (regular, low, high volatility)
- [x] 2.8 Add bucket definitions for IR (IRBucket dataclass)
- [x] 2.9 Add bucket definitions for Credit Qualifying (12 buckets + residual)
- [x] 2.10 Add bucket definitions for Credit Non-Qualifying (2 buckets + residual)
- [x] 2.11 Add bucket definitions for Equity (12 buckets + residual)
- [x] 2.12 Add bucket definitions for Commodity (17 buckets)
- [x] 2.13 Add bucket definitions for FX (single bucket)
- [x] 2.14 Add IR sub-curve definitions (OIS, Libor1m, Libor3m, Libor6m, Libor12m, Prime, Municipal)

## 3. Sensitivity Data Models
- [x] 3.1 Create `simm/sensitivity.py` with Sensitivity protocol
- [x] 3.2 Add DeltaSensitivity dataclass
- [x] 3.3 Add VegaSensitivity dataclass
- [x] 3.4 Add CurvatureSensitivity dataclass
- [x] 3.5 Add BaseCorrSensitivity dataclass (Credit Qualifying only)
- [x] 3.6 Add SensitivityCollection class to hold grouped sensitivities

## 4. CRIF Implementation
- [x] 4.1 Create `simm/crif/models.py` with CRIFRecord dataclass
- [x] 4.2 Add CRIFHeader dataclass for file metadata
- [x] 4.3 Create `simm/crif/parser.py` with CSV parsing
- [x] 4.4 Add CRIF validation logic (required fields, enum validation)
- [x] 4.5 Add CRIF-to-Sensitivity conversion functions
- [x] 4.6 Add Sensitivity-to-CRIF export functions

## 5. Configuration
- [x] 5.1 Create `simm/config.py` with SIMMConfig dataclass
- [x] 5.2 Add SIMMVersion enum (v2.5, v2.6 for future versions)
- [x] 5.3 Add validation methods to SIMMConfig

## 6. Testing
- [x] 6.1 Create `test/test_simm_taxonomy.py` with enum tests
- [x] 6.2 Add bucket definition tests
- [x] 6.3 Create `test/test_simm_crif.py` with parser tests
- [x] 6.4 Add CRIF validation tests
- [x] 6.5 Create `test/test_simm_config.py` with config tests

## 7. Documentation
- [x] 7.1 Add docstrings to all public classes and functions
- [x] 7.2 Create `simm/README.md` with module overview

