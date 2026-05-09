---
name: code-security-checker
description: |
  Review code for security vulnerabilities, unsafe practices, and potential exploits.
  Focuses on input validation, safe operations, and secure coding patterns.
  Use when the user asks to:
  - Review code for security issues
  - Check for vulnerabilities (OWASP, etc.)
  - Verify input validation
  - Ensure safe math operations
  - Audit code for security compliance
  Triggers: "security review", "vulnerability check", "security audit", "safe coding", "input validation", "OWASP"
---

# Code Security Checker Skill

Review code for security vulnerabilities, unsafe practices, and potential security risks in quantitative finance applications.

## When This Skill Activates

Codex should use this skill when:
- User asks to review code for security issues
- User wants to check for vulnerabilities
- User needs verification of input validation
- Part of a model validation workflow (invoked by orchestrator)
- User mentions security, vulnerabilities, safe coding

## Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    SECURITY REVIEW WORKFLOW                      │
├─────────────────────────────────────────────────────────────────┤
│ Step 1: Static Analysis         → Code pattern scanning         │
│ Step 2: Input Validation Check  → Boundary/type checks          │
│ Step 3: Safe Math Verification  → Overflow/underflow protection │
│ Step 4: Dependency Check        → Known vulnerabilities         │
│ Step 5: Data Handling Review    → Sensitive data protection     │
│ Step 6: Generate Security Report                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Static Analysis

### 1.1 Dangerous Function Patterns

| Pattern | Risk | Mitigation |
|---------|------|------------|
| Dynamic code evaluation | Code injection | Never use with untrusted input |
| Unsafe deserialization | Arbitrary code execution | Use JSON or validate source |
| Shell command execution | Command injection | Use subprocess with array args, shell=False |
| Direct system calls | Command injection | Use subprocess with array args |
| Dynamic imports | Import risks | Whitelist allowed modules |
| Unsafe YAML parsing | Code execution | Use yaml.safe_load() |

### 1.2 Search Patterns

```python
# Patterns to search for dangerous operations:
# - Dynamic code evaluation functions
# - Unsafe deserialization
# - Shell command execution with string interpolation
# - Direct system command calls
# - Dynamic module imports
# - YAML loading without safe loader
# - Format string potential vulnerabilities
```

### 1.3 OWASP Awareness (Where Applicable)

| OWASP Category | Relevance to Quant Code | Check |
|----------------|------------------------|-------|
| Injection | SQL (if DB), command injection | Parameterized queries, no shell=True |
| Broken Auth | API keys, credentials | No hardcoded secrets |
| Sensitive Data | Market data, positions | Encryption, access control |
| Security Misconfiguration | Debug mode, verbose errors | Production settings |
| SSRF | External data fetching | URL validation |

---

## Step 2: Input Validation Check

### 2.1 Validation Requirements

| Input Type | Required Checks |
|------------|-----------------|
| Numerical (float) | Range, NaN, Inf, sign |
| Dates | Valid format, reasonable range |
| Strings | Length, allowed characters |
| Arrays | Shape, dtype, bounds |
| Enums | Member of valid set |

### 2.2 Validation Patterns

```python
# GOOD: Proper validation
def price_option(strike: float, spot: float, vol: float, rate: float):
    """Price an option with validated inputs."""
    # Validate all inputs
    if not isinstance(strike, (int, float)):
        raise ValidationError("strike must be numeric")
    if strike <= 0:
        raise ValidationError("strike must be positive")
    if math.isnan(strike) or math.isinf(strike):
        raise ValidationError("strike must be finite")

    if vol < 0:
        raise ValidationError("volatility cannot be negative")
    if vol > 10:  # 1000% - sanity check
        raise ValidationError("volatility implausibly high")

    # ... proceed with pricing

# BAD: Missing validation
def price_option(strike, spot, vol, rate):
    d1 = (np.log(spot/strike) + (rate + 0.5*vol**2)) / (vol * np.sqrt(T))
    # Crashes if strike=0, vol=0, or any NaN
```

### 2.3 Validation Checklist

- [ ] All public method parameters validated
- [ ] Numerical inputs checked for NaN, Inf
- [ ] Positive values enforced where required (strike, spot, vol)
- [ ] Reasonable bounds checked (vol < 10, rate < 1)
- [ ] Array shapes validated before operations
- [ ] Enum values validated against allowed set
- [ ] Date ranges checked (not in past for pricing, etc.)

---

## Step 3: Safe Math Verification

### 3.1 Numerical Risks

| Operation | Risk | Mitigation |
|-----------|------|------------|
| log(x) | x <= 0 gives error/nan | Check x > 0 or use safe_log |
| sqrt(x) | x < 0 gives nan | Check x >= 0 or use safe_sqrt |
| exp(x) | Large x gives overflow | Cap or use log-space |
| x / y | y = 0 gives inf/nan | Check y != 0 or use safe_divide |
| x ** y | Various edge cases | Validate both operands |

### 3.2 Required Utility Usage

For QuantArk projects, verify use of util.numerical:

```python
# CORRECT: Using safe utilities
from util.numerical import safe_log, safe_exp, safe_sqrt, safe_divide, is_zero

d1 = (safe_log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * safe_sqrt(T))
discount = safe_exp(-r * T)

# WRONG: Raw math operations
import math
d1 = (math.log(S / K) + ...) / (sigma * math.sqrt(T))
```

### 3.3 Safe Math Checklist

- [ ] No raw math.log() or np.log() without guard
- [ ] No raw math.sqrt() or np.sqrt() without guard
- [ ] No raw math.exp() or np.exp() on potentially large values
- [ ] Division protected against zero divisor
- [ ] is_zero() used for near-zero comparisons (not x == 0)
- [ ] Edge cases (T=0, sigma=0) handled explicitly

---

## Step 4: Dependency Check

### 4.1 Known Vulnerability Sources

- Python packages: Check with pip-audit or safety
- NPM packages: Check with npm audit
- Container images: Check with trivy

### 4.2 Dependency Checklist

- [ ] No pinned versions with known CVEs
- [ ] Dependencies from trusted sources (PyPI, npm)
- [ ] No unnecessary dependencies (attack surface)
- [ ] Version constraints reasonable (not >=0.0.0)

### 4.3 Common Vulnerable Patterns

| Package | Issue | Minimum Safe Version |
|---------|-------|----------------------|
| urllib3 | Various CVEs | >= 1.26.18 |
| requests | Redirect handling | >= 2.31.0 |
| numpy | Buffer overflow (rare) | Recent versions |
| cryptography | Various | Check CVE database |

---

## Step 5: Data Handling Review

### 5.1 Sensitive Data Types

| Data Type | Sensitivity | Protection |
|-----------|-------------|------------|
| API keys / credentials | CRITICAL | Never in code, use env vars |
| Market data | MEDIUM | Access control |
| Position data | HIGH | Access control, audit |
| Client information | HIGH | Encryption, compliance |
| Pricing models | MEDIUM | Intellectual property |

### 5.2 Sensitive Data Patterns

```python
# DANGEROUS: Hardcoded credentials
API_KEY = "sk-12345abcdef"
PASSWORD = "admin123"
CONNECTION_STRING = "postgresql://user:pass@host/db"

# SAFE: Environment variables
import os
API_KEY = os.environ.get("API_KEY")
# Or use secrets manager
```

### 5.3 Data Handling Checklist

- [ ] No hardcoded credentials, API keys, passwords
- [ ] No credentials in logs or error messages
- [ ] Sensitive data not in exception messages
- [ ] Config files not containing secrets
- [ ] .env files in .gitignore
- [ ] No PII in logs without redaction

---

## Step 6: Generate Security Report

### Report Template

```markdown
# Security Review Report / 安全审查报告

**Date**: <date>
**Files Reviewed**: <count>
**Risk Level**: CRITICAL / HIGH / MEDIUM / LOW

---

## 1. 执行摘要 / Executive Summary

| Category | Issues Found | Severity |
|----------|--------------|----------|
| Dangerous Functions | X | CRITICAL/HIGH |
| Input Validation | X | MEDIUM/HIGH |
| Safe Math | X | MEDIUM |
| Dependencies | X | varies |
| Data Handling | X | CRITICAL/HIGH |

**Overall Assessment**: PASS / FAIL / NEEDS ATTENTION

---

## 2. 危险函数 / Dangerous Functions

### Critical Issues

| File | Line | Pattern | Risk | Recommendation |
|------|------|---------|------|----------------|
| ... | ... | unsafe function | Code injection | Remove or sandbox |

### Findings

[Detailed findings with code snippets]

---

## 3. 输入验证 / Input Validation

### Missing Validation

| Function | Parameter | Missing Check |
|----------|-----------|---------------|
| ... | strike | Positive check |
| ... | vol | NaN/Inf check |

### Recommendations

- Add validation to function X
- Use util.numerical.validate_positive()
- ...

---

## 4. 安全数学 / Safe Math Operations

### Unsafe Operations Found

| File | Line | Operation | Risk |
|------|------|-----------|------|
| ... | XX | math.log(x) | x <= 0 crash |
| ... | XX | x / y | Division by zero |

### Required Changes

1. Replace math.log(x) with safe_log(x) at line XX
2. Add zero check before division at line XX
3. ...

---

## 5. 依赖安全 / Dependency Security

### Vulnerable Dependencies

| Package | Version | CVE | Severity | Fix Version |
|---------|---------|-----|----------|-------------|
| ... | ... | ... | ... | ... |

### Recommendations

- Upgrade package X to version Y
- ...

---

## 6. 数据处理 / Data Handling

### Sensitive Data Exposure

| Type | Location | Issue |
|------|----------|-------|
| API Key | config.py:XX | Hardcoded |
| Password | ... | In error message |

### Required Actions

1. Move credentials to environment variables
2. Redact sensitive data from logs
3. ...

---

## 7. 安全检查清单 / Security Checklist

### Critical (Must Fix)
- [ ] Remove all hardcoded credentials
- [ ] Fix code injection vulnerabilities
- [ ] Add input validation for public APIs

### High Priority
- [ ] Replace unsafe math operations
- [ ] Upgrade vulnerable dependencies
- [ ] Add bounds checking

### Medium Priority
- [ ] Improve error handling
- [ ] Add rate limiting (if applicable)
- [ ] Review logging for data leakage
```

---

## Integration with Model Validation

When invoked by model-orchestrator:

### Input
- File paths of Developer A's implementation
- Security requirements from spec

### Output
- security-report.md in designated output directory
- Pass/Fail status based on security criteria
- Categorized list of issues with severity

### Quality Gate Criteria

| Severity | Threshold | Pass? |
|----------|-----------|-------|
| CRITICAL | 0 issues | Required |
| HIGH | 0 issues | Required |
| MEDIUM | <= 3 issues | Allowed |
| LOW | No limit | Allowed |

---

## Quant Finance Specific Considerations

### Model Manipulation Risks

```python
# Risk: External data affecting model behavior
def price_with_external_vol(ticker):
    vol = fetch_from_external_api(ticker)  # Could be manipulated
    return price_option(vol=vol, ...)

# Mitigation: Validate and bound external data
def price_with_external_vol(ticker):
    vol = fetch_from_external_api(ticker)
    if vol < 0.01 or vol > 5.0:
        raise ValidationError(f"Implausible vol {vol} for {ticker}")
    return price_option(vol=vol, ...)
```

### Numerical Attack Vectors

```python
# Risk: Adversarial inputs causing incorrect pricing
# Example: Extremely large/small values causing overflow/underflow

# Mitigation: Bounds checking
if T > 100:  # 100 years
    raise ValidationError("Maturity implausibly long")
if spot < 1e-10 or spot > 1e10:
    raise ValidationError("Spot price outside valid range")
```

---

## Principles

1. **Defense in Depth**: Multiple layers of validation
2. **Fail Securely**: Errors should not expose sensitive info
3. **Least Privilege**: Minimal access rights
4. **Validate All Input**: Trust nothing from outside
5. **Log Security Events**: But redact sensitive data
