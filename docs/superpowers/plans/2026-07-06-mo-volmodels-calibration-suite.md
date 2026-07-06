# MO Options Vol-Model Calibration Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 6-stage `example/mo_volmodels/` suite that calibrates Dupire Local Vol, Heston, and Heston-SLV to real CSI 1000 index-option (MO, `000852.SH`) data fetched via AKShare, culminating in an HTML comparison dashboard.

**Architecture:** A snapshot-to-file pipeline bridges the interpreter split — stage 01 fetches with AKShare (`/opt/anaconda3/bin/python`) and writes JSON; stages 02–06 replay offline with quantark (`.venv/bin/python`). A shared `_mo_common.py` holds the tested pure helpers (put-call-parity implication, OTM filter, Black-IV inversion); each stage script is a thin driver over those helpers and existing `quantark.volmodels` kernels.

**Tech Stack:** Python, NumPy, SciPy (Brent/least-squares), matplotlib, `quantark.volmodels` (Dupire / Heston / SLV), `quantark.param` (`GridVolSurface`, `LinearRateCurve`, `TermStructureDividendYield`), AKShare 1.18.64.

## Global Constraints

- **Interpreter split:** stage 01 runs under `/opt/anaconda3/bin/python` (has akshare, no quantark); stages 02–06 and ALL tests run under `.venv/bin/python` (has quantark, no akshare). Never import `akshare` outside stage 01.
- **Numerical ops:** use `quantark.util.numerical` (`is_zero`, `safe_log`, etc.) — never raw float tolerances (per CLAUDE.md).
- **No fabricated math:** on an uninvertible IV, singular PCP fit, or thin expiry, EXCLUDE with a logged reason — never invent a forward/rate/vol (per the user's standing "no stupid fallbacks" rule).
- **Reuse only:** consume existing library kernels; add NO pricing math to `quantark/`.
- **Canonical imports:** `quantark.*` only.
- **Offline reproducibility:** a committed sample snapshot lets stages 02–06 and tests run with no network.
- **Tests:** `test/mo_volmodels/test_*.py`, run with `.venv/bin/python -m pytest`.
- **Learning-mode contribution points** (A, B, C) are left as scaffolds with a `# >>> YOUR CODE HERE` marker and a passing-target test; the user fills the body. Do NOT fill them during subagent execution — stop and hand each back to the user.

---

### Task 1: Suite scaffold + sample snapshot fixture

**Files:**
- Create: `example/mo_volmodels/__init__.py` (empty — makes helpers importable in tests)
- Create: `example/mo_volmodels/data/.gitkeep`
- Create: `example/mo_volmodels/data/plots/.gitkeep`
- Create: `example/mo_volmodels/make_sample_snapshot.py`
- Create: `test/mo_volmodels/__init__.py` (empty)
- Test: `test/mo_volmodels/test_sample_snapshot.py`

**Interfaces:**
- Produces: `data/mo_snapshot_sample.json` with schema
  `{"fetched_at": str, "market_open": bool, "underlying": {"code": str, "spot": float}, "expiries": [{"expiry_date": str, "T_years": float, "quotes": [{"strike": float, "type": "C"|"P", "last": float, "bid": float|null, "ask": float|null, "volume": int, "oi": int}]}]}`.
- The sample is generated from a KNOWN arbitrage-free smile (BS prices under a chosen skew, r=0.02, q=0.01, S0=6000) so downstream stages have a checkable ground truth.

- [ ] **Step 1: Write the failing test**

```python
# test/mo_volmodels/test_sample_snapshot.py
import json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"

def test_sample_snapshot_schema():
    if not SNAP.exists():
        subprocess.run([sys.executable, str(ROOT / "example/mo_volmodels/make_sample_snapshot.py")], check=True)
    snap = json.loads(SNAP.read_text())
    assert set(snap) >= {"fetched_at", "market_open", "underlying", "expiries"}
    assert snap["underlying"]["code"] == "000852.SH"
    assert snap["underlying"]["spot"] > 0
    assert len(snap["expiries"]) >= 3
    for exp in snap["expiries"]:
        assert exp["T_years"] > 0
        types = {q["type"] for q in exp["quotes"]}
        assert types == {"C", "P"}  # both wings present for parity
        assert all(q["strike"] > 0 and q["last"] > 0 for q in exp["quotes"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_sample_snapshot.py -v`
Expected: FAIL (script/file missing).

- [ ] **Step 3: Write `make_sample_snapshot.py`**

```python
"""Generate a deterministic, arbitrage-free MO sample snapshot for offline runs/tests.

Prices a chosen skewed smile with Black-Scholes (r=2%, q=1%, S0=6000) so downstream
stages have a known ground truth. Run: python example/mo_volmodels/make_sample_snapshot.py
"""
import json
from pathlib import Path

import numpy as np

# Self-contained BS (this script may run under any interpreter; no quantark import).
from math import log, sqrt, exp, erf

def _norm_cdf(x): return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _bs(cp, s, k, t, sigma, r, q):
    d1 = (log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)
    if cp == "C":
        return s * exp(-q * t) * _norm_cdf(d1) - k * exp(-r * t) * _norm_cdf(d2)
    return k * exp(-r * t) * _norm_cdf(-d2) - s * exp(-q * t) * _norm_cdf(-d1)

def _smile_iv(k, s, t):
    # Downward equity skew + mild term/convexity, in log-moneyness.
    m = log(k / s)
    return 0.22 - 0.35 * m + 0.6 * m * m + 0.02 * t

def main():
    S0, R, Q = 6000.0, 0.02, 0.01
    expiries = [("2026-08-15", 0.11), ("2026-09-19", 0.20), ("2026-12-19", 0.45), ("2027-03-20", 0.70)]
    strikes = [round(x / 50.0) * 50.0 for x in (S0 * np.exp(np.linspace(-0.35, 0.35, 15)))]
    out = {"fetched_at": "2026-07-06T15:00:00", "market_open": True,
           "underlying": {"code": "000852.SH", "spot": S0}, "expiries": []}
    for date, t in expiries:
        quotes = []
        for k in strikes:
            iv = _smile_iv(k, S0, t)
            for cp in ("C", "P"):
                px = _bs(cp, S0, k, t, iv, R, Q)
                quotes.append({"strike": float(k), "type": cp, "last": round(px, 2),
                               "bid": round(px * 0.995, 2), "ask": round(px * 1.005, 2),
                               "volume": 500, "oi": 2000})
        out["expiries"].append({"expiry_date": date, "T_years": t, "quotes": quotes})
    dest = Path(__file__).resolve().parent / "data/mo_snapshot_sample.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"wrote {dest} ({sum(len(e['quotes']) for e in out['expiries'])} quotes)")

if __name__ == "__main__":
    main()
```

Also create the empty `__init__.py` files and `.gitkeep`s.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_sample_snapshot.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add example/mo_volmodels test/mo_volmodels
git commit -m "feat(mo-volmodels): suite scaffold + deterministic sample snapshot"
```

---

### Task 2: `_mo_common` — snapshot IO

**Files:**
- Create: `example/mo_volmodels/_mo_common.py`
- Test: `test/mo_volmodels/test_snapshot_io.py`

**Interfaces:**
- Produces: `load_snapshot(path) -> dict` (validates required keys, raises `ValueError` on missing key); `Quote`, `ExpirySlice` lightweight dataclasses; `iter_expiries(snapshot) -> list[ExpirySlice]` where `ExpirySlice` has `.expiry_date, .T, .calls: dict[strike->last], .puts: dict[strike->last], .volume: dict[(strike,type)->int]`.

- [ ] **Step 1: Write the failing test**

```python
# test/mo_volmodels/test_snapshot_io.py
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "example/mo_volmodels"))
import _mo_common as mc

SNAP = ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"

def test_load_and_iter():
    snap = mc.load_snapshot(SNAP)
    assert snap["underlying"]["spot"] == 6000.0
    slices = mc.iter_expiries(snap)
    assert len(slices) >= 3
    s0 = slices[0]
    assert s0.T > 0
    # paired strikes present in both call and put maps
    common = set(s0.calls) & set(s0.puts)
    assert len(common) >= 5

def test_load_missing_key(tmp_path):
    import json, pytest
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"underlying": {}}))
    with pytest.raises(ValueError):
        mc.load_snapshot(bad)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_snapshot_io.py -v`
Expected: FAIL (`_mo_common` missing).

- [ ] **Step 3: Implement snapshot IO in `_mo_common.py`**

```python
"""Shared helpers for the MO vol-model suite: snapshot IO, put-call parity, OTM
filtering, Black-IV inversion, and plotting. Pure quantark (no akshare)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_REQUIRED = {"fetched_at", "underlying", "expiries"}

def load_snapshot(path) -> dict:
    snap = json.loads(Path(path).read_text())
    missing = _REQUIRED - set(snap)
    if missing:
        raise ValueError(f"snapshot missing keys: {sorted(missing)}")
    if "spot" not in snap.get("underlying", {}):
        raise ValueError("snapshot underlying missing 'spot'")
    return snap

@dataclass
class ExpirySlice:
    expiry_date: str
    T: float
    calls: Dict[float, float]                 # strike -> last price
    puts: Dict[float, float]
    volume: Dict[Tuple[float, str], int] = field(default_factory=dict)

def iter_expiries(snapshot: dict) -> List[ExpirySlice]:
    out = []
    for exp in snapshot["expiries"]:
        calls, puts, vol = {}, {}, {}
        for q in exp["quotes"]:
            (calls if q["type"] == "C" else puts)[float(q["strike"])] = float(q["last"])
            vol[(float(q["strike"]), q["type"])] = int(q.get("volume", 0))
        out.append(ExpirySlice(exp["expiry_date"], float(exp["T_years"]), calls, puts, vol))
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_snapshot_io.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add example/mo_volmodels/_mo_common.py test/mo_volmodels/test_snapshot_io.py
git commit -m "feat(mo-volmodels): snapshot IO helpers"
```

---

### Task 3: `_mo_common.imply_forward_and_rate` — put-call parity  ⭐ CONTRIBUTION POINT A

**Files:**
- Modify: `example/mo_volmodels/_mo_common.py`
- Test: `test/mo_volmodels/test_parity.py`

**Interfaces:**
- Produces: `imply_forward_and_rate(sl: ExpirySlice, s0: float) -> ParityResult` with fields `r, forward, discount_factor, q, n_pairs`. Uses paired strikes where both C and P exist. Regress `y = C - P` on `x = K`: for each strike, `C - P = DF*(F - K) = DF*F - DF*K`, so slope `= -DF`, intercept `= DF*F`. Then `DF = -slope`, `F = intercept / DF`, `r = -ln(DF)/T`, `q = r - ln(F/s0)/T`.
- Raises `ValueError` if `< 3` paired strikes or `DF <= 0` (ill-posed — logged & excluded upstream).

- [ ] **Step 1: Write the failing test**

The sample was built with r=0.02, q=0.01, S0=6000 → forward `F = S0*exp((r-q)*T)`.

```python
# test/mo_volmodels/test_parity.py
from pathlib import Path
import math, sys, pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "example/mo_volmodels"))
import _mo_common as mc

def _slices():
    return mc.iter_expiries(mc.load_snapshot(ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"))

def test_parity_recovers_rate_and_forward():
    s0 = 6000.0
    for sl in _slices():
        res = mc.imply_forward_and_rate(sl, s0)
        assert res.r == pytest.approx(0.02, abs=1e-3)
        assert res.q == pytest.approx(0.01, abs=1e-3)
        assert res.forward == pytest.approx(s0 * math.exp((0.02 - 0.01) * sl.T), rel=1e-3)
        assert res.n_pairs >= 5

def test_parity_too_few_pairs():
    sl = mc.ExpirySlice("x", 0.2, {6000.0: 10.0}, {6000.0: 9.0})
    with pytest.raises(ValueError):
        mc.imply_forward_and_rate(sl, 6000.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_parity.py -v`
Expected: FAIL (`imply_forward_and_rate` missing).

- [ ] **Step 3: Add the scaffold (leave body for the user)**

```python
from quantark.util.numerical import safe_log

@dataclass
class ParityResult:
    r: float
    forward: float
    discount_factor: float
    q: float
    n_pairs: int

def imply_forward_and_rate(sl: "ExpirySlice", s0: float) -> "ParityResult":
    """Recover (r, forward, DF, q) for one expiry from put-call parity.

    Model: for each paired strike K,  C(K) - P(K) = DF * (F - K),
    which is linear in K with slope = -DF and intercept = DF*F.
    """
    pairs = sorted(set(sl.calls) & set(sl.puts))
    if len(pairs) < 3:
        raise ValueError(f"expiry {sl.expiry_date}: only {len(pairs)} paired strikes (<3)")
    K = np.array(pairs)
    y = np.array([sl.calls[k] - sl.puts[k] for k in pairs])
    # >>> YOUR CODE HERE (CONTRIBUTION POINT A) --------------------------------
    # 1. Least-squares fit y = a*K + b  (np.polyfit(K, y, 1) -> [a, b]).
    # 2. DF = -a ; raise ValueError if DF <= 0 (arbitrage-violating / ill-posed).
    # 3. forward = b / DF
    # 4. r = -safe_log(DF) / sl.T ; q = r - safe_log(forward / s0) / sl.T
    # 5. return ParityResult(r, forward, DF, q, len(pairs))
    raise NotImplementedError("CONTRIBUTION POINT A: implement put-call-parity fit")
    # <<< ----------------------------------------------------------------------
```

STOP here and hand to the user with the guidance in Step 4. Do not fill the body.

- [ ] **Step 4: User writes the regression, then run to verify it passes**

Guidance to user: "This is the load-bearing carry recovery — every downstream model depends on it. Fit `y = aK + b`, read off `DF = -a` and `F = b/DF`, and reject `DF <= 0` rather than fabricating one (our no-fallback rule). ~6 lines."

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_parity.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add example/mo_volmodels/_mo_common.py test/mo_volmodels/test_parity.py
git commit -m "feat(mo-volmodels): put-call-parity forward/rate implication"
```

---

### Task 4: `_mo_common.select_otm` — OTM + liquidity filter  ⭐ CONTRIBUTION POINT B

**Files:**
- Modify: `example/mo_volmodels/_mo_common.py`
- Test: `test/mo_volmodels/test_otm_filter.py`

**Interfaces:**
- Produces: `select_otm(sl: ExpirySlice, forward: float, min_volume: int = 1) -> list[OtmQuote]` where `OtmQuote` has `.strike, .kind ("C"|"P"), .price`. Rule: for `K < forward` keep the PUT, for `K >= forward` keep the CALL; drop quotes with `volume < min_volume`, non-positive price, or below intrinsic.

- [ ] **Step 1: Write the failing test**

```python
# test/mo_volmodels/test_otm_filter.py
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "example/mo_volmodels"))
import _mo_common as mc

def test_otm_selection_splits_at_forward():
    sl = mc.iter_expiries(mc.load_snapshot(ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"))[1]
    fwd = 6000.0
    otm = mc.select_otm(sl, fwd)
    assert otm, "should select some OTM quotes"
    for q in otm:
        if q.strike < fwd:
            assert q.kind == "P"
        else:
            assert q.kind == "C"
        assert q.price > 0

def test_otm_drops_illiquid():
    sl = mc.ExpirySlice("x", 0.2,
                        calls={6100.0: 5.0, 6200.0: 3.0},
                        puts={5900.0: 4.0},
                        volume={(6100.0, "C"): 0, (6200.0, "C"): 100, (5900.0, "P"): 100})
    otm = mc.select_otm(sl, 6000.0, min_volume=1)
    strikes = {q.strike for q in otm}
    assert 6100.0 not in strikes  # zero volume dropped
    assert 6200.0 in strikes and 5900.0 in strikes
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_otm_filter.py -v`
Expected: FAIL.

- [ ] **Step 3: Add the scaffold (leave body for the user)**

```python
@dataclass
class OtmQuote:
    strike: float
    kind: str      # "C" or "P"
    price: float

def select_otm(sl: "ExpirySlice", forward: float, min_volume: int = 1) -> List["OtmQuote"]:
    """Keep only OTM options: puts below the forward, calls at/above it, liquid & sane."""
    out: List[OtmQuote] = []
    strikes = sorted(set(sl.calls) | set(sl.puts))
    for k in strikes:
        # >>> YOUR CODE HERE (CONTRIBUTION POINT B) ----------------------------
        # Decide kind by moneyness: kind = "P" if k < forward else "C".
        # Look up price from sl.puts / sl.calls; skip if that side is absent.
        # Skip if sl.volume.get((k, kind), 0) < min_volume.
        # Skip if price <= 0.
        # Append OtmQuote(k, kind, price).
        raise NotImplementedError("CONTRIBUTION POINT B: implement OTM+liquidity filter")
        # <<< ------------------------------------------------------------------
    return out
```

STOP and hand to the user.

- [ ] **Step 4: User writes the predicate, then run to verify it passes**

Guidance: "This encodes the desk convention that only OTM options carry clean vol information — deep-ITM quotes are intrinsic-dominated and stale. ~8 lines; a `continue` for each drop reason (log the reason in the caller, not here)."

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_otm_filter.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add example/mo_volmodels/_mo_common.py test/mo_volmodels/test_otm_filter.py
git commit -m "feat(mo-volmodels): OTM + liquidity filter"
```

---

### Task 5: `_mo_common` — Black-IV inversion (call-equivalent) + plotting

**Files:**
- Modify: `example/mo_volmodels/_mo_common.py`
- Test: `test/mo_volmodels/test_iv_inversion.py`

**Interfaces:**
- Produces: `otm_implied_vol(q: OtmQuote, s0, r, q_carry, forward, discount_factor, T) -> float | None`. Converts an OTM put to its call-equivalent price `C = P + DF*(F - K)`, then inverts via `volmodels.black_scholes.implied_vol_call`. Returns `None` (never fabricates) when inversion raises `NumericalError` (price outside the no-arb band).
- Produces: `plot_smiles(rows, path)` — matplotlib helper (rows = list of `(label, strikes, ivs)`), saves a PNG. (Not unit-tested; smoke-checked in stage 02.)

- [ ] **Step 1: Write the failing test**

```python
# test/mo_volmodels/test_iv_inversion.py
from pathlib import Path
import math, sys, pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "example/mo_volmodels"))
import _mo_common as mc

def test_call_equiv_inversion_recovers_smile():
    # sample smile: iv = 0.22 - 0.35 m + 0.6 m^2 + 0.02 T ; r=0.02 q=0.01 S0=6000
    s0, r, qc, T = 6000.0, 0.02, 0.01, 0.20
    fwd = s0 * math.exp((r - qc) * T)
    DF = math.exp(-r * T)
    sl = mc.iter_expiries(mc.load_snapshot(ROOT / "example/mo_volmodels/data/mo_snapshot_sample.json"))[1]
    for oq in mc.select_otm(sl, fwd):
        iv = mc.otm_implied_vol(oq, s0, r, qc, fwd, DF, T)
        m = math.log(oq.strike / s0)
        expected = 0.22 - 0.35 * m + 0.6 * m * m + 0.02 * T
        assert iv == pytest.approx(expected, abs=2e-3)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_iv_inversion.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement inversion + plotting**

```python
from quantark.volmodels.black_scholes import implied_vol_call
from quantark.util.exceptions import NumericalError

def otm_implied_vol(oq: "OtmQuote", s0, r, q_carry, forward, discount_factor, T):
    """Invert an OTM quote to Black IV via its call-equivalent price. None if uninvertible."""
    if oq.kind == "P":
        call_equiv = oq.price + discount_factor * (forward - oq.strike)
    else:
        call_equiv = oq.price
    try:
        return implied_vol_call(s0, oq.strike, T, call_equiv, r, q_carry)
    except NumericalError:
        return None  # outside no-arb band -> exclude, never fabricate

def plot_smiles(rows, path, title="MO implied-vol smiles"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, strikes, ivs in rows:
        ax.plot(strikes, np.array(ivs) * 100, marker="o", ms=3, label=label)
    ax.set_xlabel("strike"); ax.set_ylabel("implied vol (%)"); ax.set_title(title)
    ax.legend(fontsize=7); fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120); plt.close(fig)
    return path
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_iv_inversion.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add example/mo_volmodels/_mo_common.py test/mo_volmodels/test_iv_inversion.py
git commit -m "feat(mo-volmodels): call-equivalent Black-IV inversion + smile plot helper"
```

---

### Task 6: Stage 01 — AKShare fetch script

**Files:**
- Create: `example/mo_volmodels/01_fetch_mo_snapshot.py`

**Interfaces:**
- Consumes: AKShare `option_cffex_zz1000_list_sina()`, `option_cffex_zz1000_spot_sina(symbol=...)`, and an index-spot source for `000852`.
- Produces: `data/mo_snapshot_YYYYMMDD.json` in the Task-1 schema, plus copies to `data/mo_snapshot_latest.json`.
- Guard: hard-check `import akshare`; if absent, print the `/opt/anaconda3/bin/python` command and `sys.exit(1)`.

Note: this stage needs the live CFFEX schema, which varies. Verify column names interactively during implementation (`option_cffex_zz1000_spot_sina` returns a DataFrame with strike/last/volume-like columns). Do NOT hardcode until confirmed against a live/`_daily` sample.

- [ ] **Step 1: Write the fetch script**

```python
"""Stage 01 — fetch a CSI 1000 (MO) option snapshot via AKShare.

RUN WITH THE AKSHARE INTERPRETER (quantark's .venv has no akshare):
    /opt/anaconda3/bin/python example/mo_volmodels/01_fetch_mo_snapshot.py

Writes data/mo_snapshot_YYYYMMDD.json and data/mo_snapshot_latest.json.
Stages 02-06 replay that snapshot offline under .venv/bin/python.
"""
import json
import sys
from datetime import datetime, date
from pathlib import Path

try:
    import akshare as ak
except ImportError:
    sys.exit("akshare not found. Run with:\n"
             "  /opt/anaconda3/bin/python example/mo_volmodels/01_fetch_mo_snapshot.py")

DATA = Path(__file__).resolve().parent / "data"

def _expiry_T(expiry_code: str) -> tuple[str, float]:
    # CFFEX month code like 'mo2408' -> third Friday of 2024-08 (verify against calendar).
    yy, mm = 2000 + int(expiry_code[2:4]), int(expiry_code[4:6])
    # third Friday
    d = date(yy, mm, 1)
    fridays = [x for x in range(1, 29) if date(yy, mm, x).weekday() == 4]
    exp = date(yy, mm, fridays[2])
    T = (exp - date.today()).days / 365.0
    return exp.isoformat(), max(T, 1e-6)

def main():
    contracts = ak.option_cffex_zz1000_list_sina()   # {'中证1000指数': ['mo2408', ...]} shape varies
    months = list(contracts.values())[0] if isinstance(contracts, dict) else list(contracts)
    # underlying index spot (verify function name against installed akshare)
    idx = ak.stock_zh_index_spot_sina() if hasattr(ak, "stock_zh_index_spot_sina") else None
    spot = float(idx.loc[idx["代码"].str.contains("000852"), "最新价"].iloc[0]) if idx is not None else float("nan")

    expiries = []
    market_open = False
    for m in months:
        df = ak.option_cffex_zz1000_spot_sina(symbol=m)   # columns: verify live
        # Expect call & put rows keyed by strike; adapt column names after inspecting df.columns.
        exp_date, T = _expiry_T(m)
        quotes = []
        for _, row in df.iterrows():
            # PLACEHOLDER MAPPING — confirm against df.columns during implementation:
            #   strike col, call last, put last, volume, oi.
            pass
        if quotes:
            expiries.append({"expiry_date": exp_date, "T_years": T, "quotes": quotes})

    snap = {"fetched_at": datetime.now().isoformat(timespec="seconds"),
            "market_open": market_open, "underlying": {"code": "000852.SH", "spot": spot},
            "expiries": expiries}
    DATA.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    (DATA / f"mo_snapshot_{stamp}.json").write_text(json.dumps(snap, indent=2))
    (DATA / "mo_snapshot_latest.json").write_text(json.dumps(snap, indent=2))
    print(f"wrote snapshot: {len(expiries)} expiries, spot={spot}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Confirm the live DataFrame schema**

Run: `/opt/anaconda3/bin/python -c "import akshare as ak; print(ak.option_cffex_zz1000_list_sina()); df=ak.option_cffex_zz1000_spot_sina(symbol='<first month>'); print(df.columns.tolist()); print(df.head())"`
Fill the `quotes` mapping (strike/call-last/put-last/volume/oi) and the index-spot lookup from the printed columns. Set `market_open` from whether any volume > 0.
If CFFEX is unreachable (weekend/blocked), skip live run — stages 02–06 use `mo_snapshot_sample.json`. Note this in the run output.

- [ ] **Step 3: Attempt a live fetch (best effort)**

Run: `/opt/anaconda3/bin/python example/mo_volmodels/01_fetch_mo_snapshot.py`
Expected: writes a snapshot, OR prints a clear "CFFEX unreachable, use sample" message. Either is acceptable — offline replay is the committed path.

- [ ] **Step 4: Commit**

```bash
git add example/mo_volmodels/01_fetch_mo_snapshot.py
git commit -m "feat(mo-volmodels): stage 01 AKShare MO snapshot fetch"
```

---

### Task 7: Stage 02 — build IV surface

**Files:**
- Create: `example/mo_volmodels/02_build_iv_surface.py`
- Test: `test/mo_volmodels/test_stage02_surface.py`

**Interfaces:**
- Consumes: `_mo_common` (`load_snapshot`, `iter_expiries`, `imply_forward_and_rate`, `select_otm`, `otm_implied_vol`, `plot_smiles`).
- Produces: `data/mo_iv_surface_latest.json` = `{"s0": float, "strikes": [..], "maturities": [..], "iv_grid": [[..]], "per_expiry": [{"expiry_date","T","r","q","forward","df","points": [[strike, iv], ...]}]}`, and `data/plots/02_smiles.png`.
- Builds a common strike grid (union of surviving strikes, clamped to those present in ≥2 maturities so `GridVolSurface` is rectangular; missing cells filled by that expiry's own smile interp — documented). Drops expiries with `< MIN_STRIKES=5` survivors (logged).

- [ ] **Step 1: Write the failing test**

```python
# test/mo_volmodels/test_stage02_surface.py
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "example/mo_volmodels/data/mo_iv_surface_latest.json"

def test_stage02_builds_surface():
    subprocess.run([sys.executable, str(ROOT / "example/mo_volmodels/02_build_iv_surface.py"),
                    "--snapshot", "sample"], check=True, cwd=ROOT)
    surf = json.loads(OUT.read_text())
    assert surf["s0"] == 6000.0
    assert len(surf["maturities"]) >= 3 and len(surf["strikes"]) >= 5
    import numpy as np
    grid = np.array(surf["iv_grid"])
    assert grid.shape == (len(surf["maturities"]), len(surf["strikes"]))
    assert np.all(grid > 0) and np.all(grid < 2.0)
    # recovered rate/carry close to ground truth
    for pe in surf["per_expiry"]:
        assert abs(pe["r"] - 0.02) < 2e-3 and abs(pe["q"] - 0.01) < 2e-3
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_stage02_surface.py -v`
Expected: FAIL.

- [ ] **Step 3: Write stage 02**

```python
"""Stage 02 — put-call parity + OTM filter + Black-IV inversion -> GridVolSurface.
Run: .venv/bin/python example/mo_volmodels/02_build_iv_surface.py [--snapshot sample|latest]"""
import argparse, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _mo_common as mc

MIN_STRIKES = 5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="sample")
    args = ap.parse_args()
    snap_path = HERE / f"data/mo_snapshot_{args.snapshot}.json"
    snap = mc.load_snapshot(snap_path)
    s0 = float(snap["underlying"]["spot"])
    if not snap.get("market_open", True):
        print("WARNING: snapshot taken while market closed — IVs from last-price mids.")

    per_expiry, smile_rows = [], []
    for sl in mc.iter_expiries(snap):
        try:
            par = mc.imply_forward_and_rate(sl, s0)
        except ValueError as e:
            print(f"  skip {sl.expiry_date}: {e}"); continue
        pts = []
        for oq in mc.select_otm(sl, par.forward):
            iv = mc.otm_implied_vol(oq, s0, par.r, par.q, par.forward, par.discount_factor, sl.T)
            if iv is None or not (0 < iv < 2.0):
                continue
            pts.append((oq.strike, iv))
        if len(pts) < MIN_STRIKES:
            print(f"  skip {sl.expiry_date}: only {len(pts)} usable strikes (<{MIN_STRIKES})"); continue
        pts.sort()
        per_expiry.append({"expiry_date": sl.expiry_date, "T": sl.T, "r": par.r, "q": par.q,
                           "forward": par.forward, "df": par.discount_factor, "points": pts})
        smile_rows.append((f"T={sl.T:.2f}", [k for k, _ in pts], [v for _, v in pts]))

    if len(per_expiry) < 2:
        sys.exit("need >=2 usable expiries to build a surface")

    # Common rectangular strike grid: strikes present in >=2 expiries; fill via per-expiry interp.
    all_strikes = sorted({k for pe in per_expiry for k, _ in pe["points"]})
    def _count(k): return sum(any(abs(k - kk) < 1e-9 for kk, _ in pe["points"]) for pe in per_expiry)
    strikes = [k for k in all_strikes if _count(k) >= 2]
    maturities = [pe["T"] for pe in per_expiry]
    grid = np.empty((len(maturities), len(strikes)))
    for i, pe in enumerate(per_expiry):
        ks = np.array([k for k, _ in pe["points"]]); vs = np.array([v for _, v in pe["points"]])
        grid[i] = np.interp(strikes, ks, vs)  # flat extrapolation at wings

    out = {"s0": s0, "strikes": strikes, "maturities": maturities,
           "iv_grid": grid.tolist(), "per_expiry": per_expiry}
    (HERE / "data/mo_iv_surface_latest.json").write_text(json.dumps(out, indent=2))
    mc.plot_smiles(smile_rows, HERE / "data/plots/02_smiles.png")
    print(f"surface: {len(maturities)} maturities x {len(strikes)} strikes -> data/mo_iv_surface_latest.json")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_stage02_surface.py -v`
Expected: PASS. Also eyeball `data/plots/02_smiles.png` shows a downward skew.

- [ ] **Step 5: Commit**

```bash
git add example/mo_volmodels/02_build_iv_surface.py test/mo_volmodels/test_stage02_surface.py
git commit -m "feat(mo-volmodels): stage 02 build IV surface (PCP + OTM + inversion)"
```

---

### Task 8: Stage 03 — Dupire Local Vol

**Files:**
- Create: `example/mo_volmodels/03_dupire_localvol.py`
- Test: `test/mo_volmodels/test_stage03_localvol.py`

**Interfaces:**
- Consumes: `data/mo_iv_surface_latest.json`; `GridVolSurface`, `LinearRateCurve`, `TermStructureDividendYield`, `PricingEnvironment`, `build_dupire_local_vol`, `LocalVolPDESolver`, `EuropeanVanillaOption`.
- Produces: `data/plots/03_localvol_surface.png`; prints per-expiry IV RMSE; writes `data/mo_reprice_localvol.json` = `{"per_expiry": [{"T", "rmse_iv"}], "overall_rmse_iv"}` (consumed by stage 06).
- Helper for building the env from the surface JSON goes in `_mo_common.build_env(surface_json) -> (env, surf, s0)` (reused by stages 03–06).

- [ ] **Step 1: Write the failing test**

```python
# test/mo_volmodels/test_stage03_localvol.py
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "example/mo_volmodels/data/mo_reprice_localvol.json"

def test_stage03_localvol_reprices_within_tol():
    subprocess.run([sys.executable, str(ROOT / "example/mo_volmodels/03_dupire_localvol.py")],
                   check=True, cwd=ROOT)
    res = json.loads(OUT.read_text())
    # Dupire is built to reprice the market smile: RMSE in IV should be small.
    assert res["overall_rmse_iv"] < 0.01  # < 1 vol point
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_stage03_localvol.py -v`
Expected: FAIL.

- [ ] **Step 3: Add `build_env` to `_mo_common.py`**

```python
def build_env(surface_json: dict):
    """Reconstruct (PricingEnvironment, GridVolSurface, s0) from stage-02 output."""
    from datetime import datetime
    from quantark.param import GridVolSurface, SpotQuote
    from quantark.param.rrf.rate_curve import LinearRateCurve
    from quantark.param.div import TermStructureDividendYield
    from quantark.priceenv import PricingEnvironment
    s0 = float(surface_json["s0"])
    surf = GridVolSurface(surface_json["strikes"], surface_json["maturities"],
                          np.array(surface_json["iv_grid"]))
    pe = surface_json["per_expiry"]
    ts = [p["T"] for p in pe]
    # >=2 pillars required by both curve types:
    rate = LinearRateCurve([(p["T"], p["r"]) for p in pe])
    div = TermStructureDividendYield(times=ts, yields=[p["q"] for p in pe])
    env = PricingEnvironment(rate_curve=rate, valuation_date=datetime(2026, 7, 6),
                             spot_quote=SpotQuote(spot=s0), vol_surface=surf, div_yield=div)
    return env, surf, s0
```

- [ ] **Step 4: Write stage 03**

```python
"""Stage 03 — Dupire local vol: build sigma_LV(K,T) from the market surface, reprice, RMSE.
Run: .venv/bin/python example/mo_volmodels/03_dupire_localvol.py"""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _mo_common as mc
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.black_scholes import implied_vol_call, bs_call_price
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.asset.equity.engine.pde import LocalVolPDESolver
from quantark.asset.equity.param import PDEParams
from quantark.util.enum import OptionType

def main():
    surface = json.loads((HERE / "data/mo_iv_surface_latest.json").read_text())
    env, surf, s0 = mc.build_env(surface)
    lv = build_dupire_local_vol(surf, spot=s0, rate_curve=env.rate_curve, div_yield=env.get_div_yield)

    solver = LocalVolPDESolver(PDEParams(grid_size=300, time_steps=150))
    per_expiry, sq_err = [], []
    for pe in surface["per_expiry"]:
        T, r, q = pe["T"], pe["r"], pe["q"]
        errs = []
        for k, mkt_iv in pe["points"]:
            opt = EuropeanVanillaOption(strike=k, option_type=OptionType.CALL, maturity=T)
            price = solver.price(opt, env)
            try:
                model_iv = implied_vol_call(s0, k, T, price, r, q)
            except Exception:
                continue
            errs.append(model_iv - mkt_iv)
        if errs:
            rmse = float(np.sqrt(np.mean(np.square(errs))))
            per_expiry.append({"T": T, "rmse_iv": rmse}); sq_err.extend(errs)
            print(f"  T={T:.2f}  LV RMSE={rmse*100:.3f} vol-pts  ({len(errs)} strikes)")

    overall = float(np.sqrt(np.mean(np.square(sq_err))))
    (HERE / "data/mo_reprice_localvol.json").write_text(
        json.dumps({"per_expiry": per_expiry, "overall_rmse_iv": overall}, indent=2))
    print(f"overall LV RMSE = {overall*100:.3f} vol-pts")

    # local-vol surface plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Ks = np.array(surface["strikes"]); Ts = np.array(surface["maturities"])
    Z = np.array([[lv.local_vol(k, t) for k in Ks] for t in Ts])
    fig, ax = plt.subplots(figsize=(8, 5))
    c = ax.contourf(Ks, Ts, Z * 100, levels=20)
    fig.colorbar(c, label="local vol (%)"); ax.set_xlabel("strike"); ax.set_ylabel("T")
    ax.set_title("Dupire local volatility surface")
    fig.savefig(HERE / "data/plots/03_localvol_surface.png", dpi=120); plt.close(fig)

if __name__ == "__main__":
    main()
```

Note: confirm the `LocalVolSurface` sampling method name (`local_vol(K, T)`); adjust if the kernel exposes a different accessor (check `quantark/volmodels/localvol/surface.py`).

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_stage03_localvol.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add example/mo_volmodels/03_dupire_localvol.py example/mo_volmodels/_mo_common.py test/mo_volmodels/test_stage03_localvol.py
git commit -m "feat(mo-volmodels): stage 03 Dupire local vol reprice + surface plot"
```

---

### Task 9: Stage 04 — Heston calibration  ⭐ CONTRIBUTION POINT C

**Files:**
- Create: `example/mo_volmodels/04_heston_calibration.py`
- Test: `test/mo_volmodels/test_stage04_heston.py`

**Interfaces:**
- Consumes: `data/mo_iv_surface_latest.json`; `MarketOption`, `calibrate_heston`, `HestonParams`, `HestonAnalyticalEngine`.
- Produces: `data/plots/04_heston_fit.png`; `data/mo_calib_heston.json` = `{"params": {v0,kappa,theta,sigma,rho}, "feller": float, "cost": float, "overall_rmse_iv": float, "per_expiry": [{"T","rmse_iv"}]}`.
- The initial guess + bounds are CONTRIBUTION POINT C.

- [ ] **Step 1: Write the failing test**

```python
# test/mo_volmodels/test_stage04_heston.py
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "example/mo_volmodels/data/mo_calib_heston.json"

def test_stage04_heston_calibrates():
    subprocess.run([sys.executable, str(ROOT / "example/mo_volmodels/04_heston_calibration.py")],
                   check=True, cwd=ROOT)
    res = json.loads(OUT.read_text())
    p = res["params"]
    assert 0 < p["v0"] < 0.5 and p["kappa"] > 0 and p["theta"] > 0 and -1 < p["rho"] < 0
    assert res["overall_rmse_iv"] < 0.03   # Heston fits the whole surface to a few vol pts
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_stage04_heston.py -v`
Expected: FAIL.

- [ ] **Step 3: Write stage 04 with a scaffolded initial guess (CONTRIBUTION POINT C)**

```python
"""Stage 04 — calibrate Heston (v0,kappa,theta,sigma,rho) to the OTM chain.
Run: .venv/bin/python example/mo_volmodels/04_heston_calibration.py"""
import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _mo_common as mc
from quantark.volmodels.heston import HestonParams, MarketOption, calibrate_heston
from quantark.volmodels.black_scholes import bs_call_price, implied_vol_call

def main():
    surface = json.loads((HERE / "data/mo_iv_surface_latest.json").read_text())
    s0 = float(surface["s0"])
    # Term-structure r,q as callables from the per-expiry pillars.
    pe = surface["per_expiry"]
    Ts = np.array([p["T"] for p in pe]); Rs = np.array([p["r"] for p in pe]); Qs = np.array([p["q"] for p in pe])
    r_of = lambda t: float(np.interp(t, Ts, Rs)); q_of = lambda t: float(np.interp(t, Ts, Qs))

    options = []
    for p in pe:
        for k, iv in p["points"]:
            price = bs_call_price(s0, k, p["T"], iv, p["r"], p["q"])  # call-equiv target price
            options.append(MarketOption(K=k, T=p["T"], price=price))

    # >>> YOUR CODE HERE (CONTRIBUTION POINT C) --------------------------------
    # Choose a sensible initial guess and bounds. Hints:
    #   v0 ~ theta ~ (ATM short-T vol)^2  (read an ATM iv near T=Ts[0]);
    #   kappa ~ 1.5 (mean-reversion speed), sigma ~ 0.5 (vol-of-vol), rho ~ -0.6 (equity skew).
    #   Keep the initial guess strictly inside `bounds` or calibrate_heston raises.
    initial = HestonParams(v0=..., kappa=..., theta=..., sigma=..., rho=...)
    bounds = ((1e-6, 1e-3, 1e-4, 1e-3, -0.95), (0.5, 10.0, 0.5, 2.0, 0.0))
    # <<< ----------------------------------------------------------------------

    result = calibrate_heston(s0=s0, options=options, r=r_of, carry=q_of,
                              initial=initial, bounds=bounds, target="price", method="lewis")
    hp = result.params
    feller = 2 * hp.kappa * hp.theta / (hp.sigma ** 2)
    print(f"calibrated: v0={hp.v0:.4f} kappa={hp.kappa:.3f} theta={hp.theta:.4f} "
          f"sigma={hp.sigma:.3f} rho={hp.rho:.3f}  Feller={feller:.2f}  cost={result.cost:.3e}")

    # per-expiry IV RMSE via the Heston analytical pricer
    from quantark.volmodels.heston import heston_call_prices_vectorized  # (s0, strikes, T, params, r, carry)
    per_expiry, sq = [], []
    for p in pe:
        errs = []
        for k, mkt_iv in p["points"]:
            hprice = heston_call_prices_vectorized(s0, np.array([k]), p["T"], hp, r_of(p["T"]), q_of(p["T"]))[0]
            try:
                errs.append(implied_vol_call(s0, k, p["T"], float(hprice), p["r"], p["q"]) - mkt_iv)
            except Exception:
                continue
        if errs:
            rmse = float(np.sqrt(np.mean(np.square(errs))))
            per_expiry.append({"T": p["T"], "rmse_iv": rmse}); sq.extend(errs)
    overall = float(np.sqrt(np.mean(np.square(sq))))
    out = {"params": {"v0": hp.v0, "kappa": hp.kappa, "theta": hp.theta, "sigma": hp.sigma, "rho": hp.rho},
           "feller": feller, "cost": result.cost, "overall_rmse_iv": overall, "per_expiry": per_expiry}
    (HERE / "data/mo_calib_heston.json").write_text(json.dumps(out, indent=2))
    print(f"overall Heston RMSE = {overall*100:.3f} vol-pts")

if __name__ == "__main__":
    main()
```

STOP at the scaffold and hand CONTRIBUTION POINT C to the user. Also confirm the vectorized-pricer export name in `quantark/volmodels/heston/analytical_kernel.py` and fix the import if different.

- [ ] **Step 4: User writes the initial guess/bounds, then run to verify it passes**

Guidance: "The initial guess is where domain knowledge earns its keep — a good `v0≈theta≈ATM²` and `rho<0` start converges fast; a bad one lands in a local min. Read an ATM iv from the shortest expiry for v0/theta. ~2 lines."

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_stage04_heston.py -v`
Expected: PASS.

- [ ] **Step 5: Add the model-vs-market smile plot**

Append to stage 04 before writing JSON: build `smile_rows` of `(f"mkt T={T:.2f}", strikes, mkt_ivs)` and `(f"Heston T={T:.2f}", strikes, model_ivs)` and call `mc.plot_smiles(rows, HERE/"data/plots/04_heston_fit.png", title="Heston fit vs market")`.

- [ ] **Step 6: Commit**

```bash
git add example/mo_volmodels/04_heston_calibration.py test/mo_volmodels/test_stage04_heston.py
git commit -m "feat(mo-volmodels): stage 04 Heston calibration + smile fit"
```

---

### Task 10: Stage 05 — SLV leverage calibration

**Files:**
- Create: `example/mo_volmodels/05_slv_calibration.py`
- Test: `test/mo_volmodels/test_stage05_slv.py`

**Interfaces:**
- Consumes: `data/mo_calib_heston.json` (Heston params), `data/mo_iv_surface_latest.json`; `calibrate_leverage_surface`, `FpCalibrationConfig`, `HestonSLVMCEngine` (or SLV PDE). Grid kept modest (per the approved "runs in seconds" constraint).
- Produces: `data/plots/05_slv_leverage.png`; `data/mo_reprice_slv.json` = `{"overall_rmse_iv", "per_expiry":[{"T","rmse_iv"}]}`.

- [ ] **Step 1: Write the failing test**

```python
# test/mo_volmodels/test_stage05_slv.py
import json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def test_stage05_slv_at_least_as_good_as_heston():
    subprocess.run([sys.executable, str(ROOT / "example/mo_volmodels/05_slv_calibration.py")],
                   check=True, cwd=ROOT)
    slv = json.loads((ROOT / "example/mo_volmodels/data/mo_reprice_slv.json").read_text())
    heston = json.loads((ROOT / "example/mo_volmodels/data/mo_calib_heston.json").read_text())
    # SLV grafts the exact local vol onto Heston -> should reprice at least as well.
    assert slv["overall_rmse_iv"] <= heston["overall_rmse_iv"] + 5e-3
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_stage05_slv.py -v`
Expected: FAIL.

- [ ] **Step 3: Write stage 05**

Use `vol_models_demo.py` (lines building `calibrate_leverage_surface` + `HestonSLVPDESolver`/`HestonSLVMCEngine`) as the reference for exact call signatures. Modest grid (e.g. leverage grid `n≈40`, MC `num_paths≈50_000, time_steps≈50`) so it runs in seconds. Compute per-expiry IV RMSE the same way as stage 03/04, write `data/mo_reprice_slv.json`, and plot the leverage surface `L(S,T)`.

```python
"""Stage 05 — calibrate the SLV leverage surface on the calibrated Heston + Dupire LV.
Run: .venv/bin/python example/mo_volmodels/05_slv_calibration.py"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _mo_common as mc
from quantark.volmodels.heston import HestonParams
from quantark.volmodels.localvol import build_dupire_local_vol
from quantark.volmodels.slv import calibrate_leverage_surface, FpCalibrationConfig
from quantark.volmodels.curves import forward_carry_on_grid
from quantark.asset.equity.engine.pde import HestonSLVPDESolver
from quantark.asset.equity.product.option import EuropeanVanillaOption
from quantark.volmodels.black_scholes import implied_vol_call
from quantark.util.enum import OptionType

def main():
    surface = json.loads((HERE / "data/mo_iv_surface_latest.json").read_text())
    calib = json.loads((HERE / "data/mo_calib_heston.json").read_text())["params"]
    env, surf, s0 = mc.build_env(surface)
    hp = HestonParams(**calib)
    lv = build_dupire_local_vol(surf, spot=s0, rate_curve=env.rate_curve, div_yield=env.get_div_yield)

    # FFP leverage calibration works on a time grid with forward rate/carry per step
    # (signature confirmed against example/vol_models_demo.py). Modest grid -> runs in seconds.
    n = 40
    T_max = max(surface["maturities"])
    t_grid = np.linspace(0.0, T_max, n + 1)
    r_fwd = np.array([env.rate_curve.get_forward_rate(t_grid[i], t_grid[i + 1]) for i in range(n)])
    carry_fwd = forward_carry_on_grid(env.get_div_yield, t_grid)
    leverage = calibrate_leverage_surface(s0, hp, lv, np.diff(t_grid), r_fwd, carry_fwd,
                                          eta=1.0, fp_config=FpCalibrationConfig(n_x=161, n_z=81))
    solver = HestonSLVPDESolver(hp, leverage, eta=1.0, n_x=160, n_v=48, n_t=60)

    per_expiry, sq = [], []
    for pe in surface["per_expiry"]:
        T, r, q = pe["T"], pe["r"], pe["q"]
        errs = []
        for k, mkt_iv in pe["points"]:
            opt = EuropeanVanillaOption(strike=k, option_type=OptionType.CALL, maturity=T)
            price = solver.price(opt, env)
            try:
                errs.append(implied_vol_call(s0, k, T, price, r, q) - mkt_iv)
            except Exception:
                continue
        if errs:
            rmse = float(np.sqrt(np.mean(np.square(errs))))
            per_expiry.append({"T": T, "rmse_iv": rmse}); sq.extend(errs)
            print(f"  T={T:.2f}  SLV RMSE={rmse*100:.3f} vol-pts")
    overall = float(np.sqrt(np.mean(np.square(sq))))
    (HERE / "data/mo_reprice_slv.json").write_text(
        json.dumps({"overall_rmse_iv": overall, "per_expiry": per_expiry}, indent=2))
    print(f"overall SLV RMSE = {overall*100:.3f} vol-pts")

    # leverage surface plot: LeverageSurface.leverage(spot, t) (confirmed against slv/leverage.py)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    Ss = np.array(surface["strikes"]); Ts = np.array(surface["maturities"])
    Z = np.array([[float(leverage.leverage(sx, t)) for sx in Ss] for t in Ts])
    fig, ax = plt.subplots(figsize=(8, 5))
    c = ax.contourf(Ss, Ts, Z, levels=20); fig.colorbar(c, label="leverage L(S,T)")
    ax.set_xlabel("spot"); ax.set_ylabel("T"); ax.set_title("SLV leverage surface")
    fig.savefig(HERE / "data/plots/05_slv_leverage.png", dpi=120); plt.close(fig)

if __name__ == "__main__":
    main()
```

All signatures above are verified against `example/vol_models_demo.py` and `quantark/volmodels/slv/leverage.py`: `calibrate_leverage_surface(spot, heston, lv, dt, r_fwd, carry_fwd, eta, fp_config)`, `HestonSLVPDESolver(hp, leverage, eta, n_x, n_v, n_t)`, and `LeverageSurface.leverage(spot, t)`.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_stage05_slv.py -v`
Expected: PASS (may take ~10–30s).

- [ ] **Step 5: Commit**

```bash
git add example/mo_volmodels/05_slv_calibration.py test/mo_volmodels/test_stage05_slv.py
git commit -m "feat(mo-volmodels): stage 05 SLV leverage calibration + reprice"
```

---

### Task 11: Stage 06 — comparison + HTML dashboard

**Files:**
- Create: `example/mo_volmodels/06_compare_reprice.py`
- Test: `test/mo_volmodels/test_stage06_dashboard.py`

**Interfaces:**
- Consumes: `data/mo_reprice_localvol.json`, `data/mo_calib_heston.json`, `data/mo_reprice_slv.json`, `data/mo_iv_surface_latest.json`, and the PNGs under `data/plots/`.
- Produces: `data/comparison_summary.csv` and a self-contained `data/mo_volmodels_dashboard.html` (inline CSS, base64-embedded PNGs — no external assets, matching `example/simm_portfolio_dashboard.html`).

- [ ] **Step 1: Write the failing test**

```python
# test/mo_volmodels/test_stage06_dashboard.py
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
HTML = ROOT / "example/mo_volmodels/data/mo_volmodels_dashboard.html"

def test_stage06_dashboard():
    subprocess.run([sys.executable, str(ROOT / "example/mo_volmodels/06_compare_reprice.py")],
                   check=True, cwd=ROOT)
    html = HTML.read_text()
    assert "<html" in html.lower()
    for model in ("Local Vol", "Heston", "SLV"):
        assert model in html
    assert "data:image/png;base64," in html  # embedded, self-contained
    assert (ROOT / "example/mo_volmodels/data/comparison_summary.csv").exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_stage06_dashboard.py -v`
Expected: FAIL.

- [ ] **Step 3: Write stage 06**

```python
"""Stage 06 — consolidate BS-flat / LocalVol / Heston / SLV into a CSV + HTML dashboard.
Run: .venv/bin/python example/mo_volmodels/06_compare_reprice.py"""
import base64, csv, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

def _b64(png: Path) -> str:
    if not png.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()

def main():
    surface = json.loads((HERE / "data/mo_iv_surface_latest.json").read_text())
    lv = json.loads((HERE / "data/mo_reprice_localvol.json").read_text())
    he = json.loads((HERE / "data/mo_calib_heston.json").read_text())
    slv = json.loads((HERE / "data/mo_reprice_slv.json").read_text())

    rows = [("Local Vol", lv["overall_rmse_iv"]),
            ("Heston", he["overall_rmse_iv"]),
            ("SLV", slv["overall_rmse_iv"])]
    with (HERE / "data/comparison_summary.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "overall_rmse_iv_volpts"])
        for name, r in rows:
            w.writerow([name, f"{r*100:.4f}"])

    p = he["params"]
    table = "".join(f"<tr><td>{n}</td><td>{r*100:.3f}</td></tr>" for n, r in rows)
    params = " ".join(f"{k}={p[k]:.4f}" for k in ("v0","kappa","theta","sigma","rho"))
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>MO Vol-Model Dashboard</title>
<style>body{{font-family:system-ui,Arial;margin:2rem;color:#1a1a1a}}
h1{{font-size:1.4rem}} table{{border-collapse:collapse;margin:1rem 0}}
td,th{{border:1px solid #ccc;padding:.4rem .8rem;text-align:right}} th{{background:#f0f3f7}}
img{{max-width:100%;border:1px solid #eee;margin:.5rem 0}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:1rem}}</style>
</head><body>
<h1>MO (中证1000 / 000852.SH) Vol-Model Calibration Dashboard</h1>
<p>Underlying spot: <b>{surface['s0']:.1f}</b> &middot; {len(surface['maturities'])} expiries &middot;
Heston params: <code>{params}</code> &middot; Feller 2&kappa;&theta;/&sigma;&sup2; = {he['feller']:.2f}</p>
<h2>Repricing accuracy (overall IV RMSE, vol-pts)</h2>
<table><tr><th>Model</th><th>RMSE (vol-pts)</th></tr>{table}</table>
<div class='grid'>
<div><h3>Market smiles</h3><img src='{_b64(HERE/"data/plots/02_smiles.png")}'></div>
<div><h3>Heston fit</h3><img src='{_b64(HERE/"data/plots/04_heston_fit.png")}'></div>
<div><h3>Dupire local vol</h3><img src='{_b64(HERE/"data/plots/03_localvol_surface.png")}'></div>
<div><h3>SLV leverage</h3><img src='{_b64(HERE/"data/plots/05_slv_leverage.png")}'></div>
</div></body></html>"""
    (HERE / "data/mo_volmodels_dashboard.html").write_text(html)
    print("wrote data/mo_volmodels_dashboard.html and data/comparison_summary.csv")
    for name, r in rows:
        print(f"  {name:10s} RMSE = {r*100:.3f} vol-pts")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_stage06_dashboard.py -v`
Expected: PASS. Open `data/mo_volmodels_dashboard.html` in a browser to eyeball.

- [ ] **Step 5: Commit**

```bash
git add example/mo_volmodels/06_compare_reprice.py test/mo_volmodels/test_stage06_dashboard.py
git commit -m "feat(mo-volmodels): stage 06 comparison CSV + self-contained HTML dashboard"
```

---

### Task 12: README + full-suite smoke test

**Files:**
- Create: `example/mo_volmodels/README.md`
- Test: `test/mo_volmodels/test_suite_smoke.py`

- [ ] **Step 1: Write the end-to-end smoke test**

```python
# test/mo_volmodels/test_suite_smoke.py
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
STAGES = ["02_build_iv_surface.py", "03_dupire_localvol.py",
          "04_heston_calibration.py", "05_slv_calibration.py", "06_compare_reprice.py"]

def test_full_offline_pipeline():
    # 01 is skipped (needs akshare); the committed sample snapshot drives 02-06.
    for stage in STAGES:
        r = subprocess.run([sys.executable, str(ROOT / "example/mo_volmodels" / stage)],
                           cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, f"{stage} failed:\n{r.stdout}\n{r.stderr}"
```

- [ ] **Step 2: Run to verify it passes** (all prior stages must be green)

Run: `.venv/bin/python -m pytest test/mo_volmodels/test_suite_smoke.py -v`
Expected: PASS.

- [ ] **Step 3: Write the README**

Document: the purpose; the **interpreter split** (01 → `/opt/anaconda3/bin/python`, 02–06 → `.venv/bin/python`); how to run each stage; that the committed `mo_snapshot_sample.json` enables offline replay; where outputs land (`data/`, `data/plots/`, the HTML dashboard); the three contribution points; and a pointer to `quantark/volmodels/` and `docs/` theory. Include the exact command sequence:

```
/opt/anaconda3/bin/python example/mo_volmodels/01_fetch_mo_snapshot.py   # optional, live data
.venv/bin/python example/mo_volmodels/02_build_iv_surface.py --snapshot sample
.venv/bin/python example/mo_volmodels/03_dupire_localvol.py
.venv/bin/python example/mo_volmodels/04_heston_calibration.py
.venv/bin/python example/mo_volmodels/05_slv_calibration.py
.venv/bin/python example/mo_volmodels/06_compare_reprice.py
```

- [ ] **Step 4: Run the full suite test again + commit**

Run: `.venv/bin/python -m pytest test/mo_volmodels/ -v`
Expected: ALL PASS.

```bash
git add example/mo_volmodels/README.md test/mo_volmodels/test_suite_smoke.py
git commit -m "docs(mo-volmodels): README + full-suite offline smoke test"
```

---

## Self-Review

**Spec coverage:** §3 layout → Task 1; §4 stage 01 → Task 6; stage 02 → Task 7; stage 03 → Task 8; stage 04 → Task 9; stage 05 → Task 10; stage 06 (HTML dashboard) → Task 11; §5 `_mo_common` → Tasks 2–5; §6 no-fabrication error handling → Tasks 3/5/7 (logged skips, `None` on uninvertible); §7 contribution points A/B/C → Tasks 3/4/9; §9 success criteria → Tasks 7 (surface + rate recovery), 8/9/10 (RMSE), 11 (dashboard), 12 (standalone offline run).

**Placeholder scan:** the only intentional blanks are CONTRIBUTION POINTS A/B/C (marked, with target tests) and the stage-01 live-schema mapping (unavoidable — CFFEX column names must be confirmed against a live DataFrame; the sample-snapshot path is fully specified and testable regardless).

**Type consistency:** `ExpirySlice`(.T/.calls/.puts/.volume), `ParityResult`(.r/.forward/.discount_factor/.q/.n_pairs), `OtmQuote`(.strike/.kind/.price) used consistently across Tasks 2–7. `build_env` returns `(env, surf, s0)` consumed identically in Tasks 8/10. Reprice JSONs use `overall_rmse_iv` + `per_expiry:[{T,rmse_iv}]` consistently, read back in Task 11.

**Library-accessor verification (all confirmed against source before finalizing):**
- `LocalVolSurface.local_vol(spot, t)` — `quantark/volmodels/localvol/surface.py:57` ✓
- `LeverageSurface.leverage(spot, t)` — `quantark/volmodels/slv/leverage.py` ✓ (NOT `.value`)
- `heston_call_prices_vectorized(s0, strikes, T, params, r, carry)` — arg order `params` before `r,carry`, exported from `quantark.volmodels.heston` ✓
- `calibrate_leverage_surface(spot, heston, lv, dt, r_fwd, carry_fwd, eta, fp_config)` + `HestonSLVPDESolver(hp, lev, eta, n_x, n_v, n_t)` — `example/vol_models_demo.py:75–82` ✓

**Remaining unavoidable blank:** only the stage-01 live CFFEX DataFrame column mapping (must be read off a live/`_daily` DataFrame at implementation time). The committed sample-snapshot path is fully specified and drives all tests, so this does not block the suite.
