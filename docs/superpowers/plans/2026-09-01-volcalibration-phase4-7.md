# quantark.volcalibration — Phases 4–7 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `quantark.volcalibration` *operable* — a YAML config plus four CLI verbs that an agent can drive to build and calibrate a day's surface, resume safely, and report state through exit codes; then hand the result to backtest as a `CalibrationSet`, add the FX normalizer, and document the module.

**Architecture:** Phases 1–3 built pure stage functions (snapshot → `QuoteSet` → artifact → calibrated model). This plan adds the two layers above them and one below. Below: `config.py` gains a `RunConfig` tree and `yaml_loader.py` parses it fail-closed. Above: `store.py` gains the on-disk layout and both manifests, `runner.py` orchestrates a per-date loop with an advisory lock and a *checkable* resume rule, and `cli.py` exposes `run | status | show | list`. Then `CalibrationSet` closes the producer/consumer seam to backtest, `normalize/fxdelta.py` proves the `QuoteSet` boundary carries a second quote convention, and the docs land.

**Tech Stack:** Python 3.10+, NumPy, PyYAML, `concurrent.futures.ProcessPoolExecutor`, `fcntl` advisory locking, pytest with `-n auto --dist worksteal`. Library code only — no network, no vendor formats.

**Spec:** `docs/superpowers/specs/2026-09-01-volcalibration-module-design.md` (phases 4–7 of §9)

**Predecessor:** `docs/superpowers/plans/2026-09-01-volcalibration-phase1-3.md` (tasks 1–12, complete, commits `b795db2..5a7cbdf`)

## Global Constraints

Every task's requirements implicitly include all of these. The first six are carried forward verbatim from the phase 1–3 plan and still bind.

- **Artifact bytes must not change.** The artifact sha256 is over raw file bytes and feeds the calibration cache key; 787 admitted MO surfaces and a warm cache depend on it. New provenance goes in manifest records, never the artifact body. (Spec §5.3)
- **Fail-closed, no fallbacks.** A failure raises `ValidationError`, `NumericalError` or `AdmissionError` naming the date and the reason. Never substitute a flat vol, a floored value, or a guessed field. If the maths cannot be implemented correctly, write a `# TODO` and stop.
- **Numerical helpers only.** Use `quantark.util.numerical` (`is_zero`, `is_close`, `is_positive`, `safe_log`, `safe_divide`, `Tolerance`). Never a raw float comparison or a hardcoded tolerance.
- **Canonical imports only.** `quantark.*`. Never the deprecated flat names.
- **Dependency direction.** Nothing under `quantark/volcalibration/` may import `quantark.backtest` or `quantark.asset`. Allowed: `quantark.volmodels`, `quantark.param`, `quantark.priceenv`, `quantark.util`. Task 20 goes the other way — backtest imports volcalibration — which is the point.
- **Never `git add example/`.** The mo_volmodels test runs rewrite two tracked sample files with float churn. Run `git checkout -- example/` before staging. Task 22 adds a *new* file under `example/`, which must be staged by explicit path, never by directory.
- **`docs/` is git-excluded.** New files under `docs/` need `git add -f`.
- **Python is `/Users/fuxinyao/quant-ark/.venv/bin/python`** — the venv lives in the *main repo*; there is no `.venv/` here. That editable install resolves `quantark` to the main repo, so every test command is prefixed `PYTHONPATH=$PWD` to shadow it with worktree source. Run all commands from the worktree root; never `cd` to the main checkout.

**Pre-resolved facts** (verified against the tree and against the live MO store while writing this plan — do not re-derive):

- The live surface manifest at `/Users/fuxinyao/quant-ark/example/mo_volmodels/data/history/surface_manifest.json` has **809 records: 787 `ok`, 22 `excluded`**, and its frozen `config` block is exactly
  `{"artifact_schema_version": 1, "min_common_strikes": 3, "min_expiries": 2, "min_strikes_per_expiry": 5, "sabr_beta": 1.0}`.
  **`min_common_strikes` is 3, not 5.** Spec §6.1's example YAML says 5; that is a spec typo, corrected as step 1 of Task 13.
- Live exclusion reasons: `static_arbitrage` (17), `insufficient_expiries` (2), `insufficient_expiries_for_dupire` (2), `parity_gating_failed` (1).
- `insufficient_expiries_for_dupire` is **not** a builder reason. `example/mo_volmodels/exclude_thin_surfaces.py` writes it out-of-band, keeps the artifact on disk, and records a `study_admission` block at manifest top level. Task 15's resume rule must never resurrect such a record.
- The live calibration manifest is at `/Users/fuxinyao/quant-ark/output/mo_daily_calibration/calibration_manifest.json` — 261 records, top-level `baseline_date`, `bootstrap_policy`, `config`; per-record keys `{date, status, surface_sha, surface_path, started_at, completed_at, elapsed_seconds, config, variants}`. No live record carries `temporal_scheme`, so EWMA smoothing is off in production.
- Store paths in production are **split**: surfaces under `example/mo_volmodels/data/history`, calibration under `output/mo_daily_calibration`. Spec §5.4 draws one root; the layout must express both (Task 14).
- The lock file is `pipeline.lock`, not `.lock`.
- `quantark.util.exceptions` defines `QuantArkException`, `ValidationError`, `NumericalError`, `MarketDataError`, `PricingError`. `ValidationError` is **not** a `ValueError`.
- `AutocallableMarketDataSet` (`quantark/backtest/replay/market.py:132`) already carries `surface_history: Optional[VolSurfaceHistory]`. That is the field Task 20 pairs with.
- `VolSurfaceHistory` (`quantark/param/vol/surface_history.py:316`) already implements manifest loading, admitted-date sorting, sha verification and carry-forward. `CalibrationSet` composes it; it does not reimplement it.
- `build_artifact(quotes, snapshot, *, sabr_beta)` is the phase-2 entry point; `SettlementNormalizer().normalize(snapshot)` produces the `QuoteSet`.
- PyYAML is already a dependency (`quantark/modelvalidation/yaml_loader.py` imports it).
- `QuoteSnapshot` fields are `schema_version, convention, trade_date, symbol, spot, price_field, source, expiries`. The source sha256 lives at `snapshot.source["sha256"]`, not at top level, and the class has no serializer yet (Task 15 adds `.sha256` and `.to_payload()`).
- `snapshot.py` already defines `CONVENTION_LISTED`, `CONVENTION_FX_DELTA`, `CONVENTIONS`, `PRICE_FIELD_SETTLEMENT`, `PRICE_FIELD_MID_OR_LAST`. Import them; do not redeclare them in `config.py`.
- The stored `mo_iv_surface_20260430.json` has `s0 = 8381.947`. Tests that lift the committed sample snapshot must pass that spot, or the rebuilt surface will not match.
- The committed CFETS snapshots are per-date: `example/fx_volmodels/data/cfets_usdcny_snapshot_20260430.json` (and five more).
- `VolSurfaceHistory.admitted_dates` is a public property; `surface_for` handles carry-forward. `test/replay_golden/fixtures.py` provides `DATE_A = date(2024, 1, 2)` and `write_localvol_history(root)`.

---

## File Structure

**Created:**

| File | Responsibility |
|---|---|
| `quantark/volcalibration/yaml_loader.py` | YAML → `RunConfig`, fail-closed, every error names its YAML path |
| `quantark/volcalibration/runner.py` | Lock, per-date surface loop, per-date calibration loop, EWMA, status |
| `quantark/volcalibration/cli.py` | Verbs `run`/`status`/`show`/`list`, `--json`, `--plan`, exit codes |
| `quantark/volcalibration/__main__.py` | `python -m quantark.volcalibration` |
| `quantark/volcalibration/calibration_set.py` | `CalibrationSet` loader and `environment_for` |
| `quantark/volcalibration/normalize/fxdelta.py` | CFETS delta-quoted tenor slices → `QuoteSet` |
| `example/mo_volmodels/export_snapshots.py` | One-way bridge: settlement CSV → canonical snapshot JSON |
| `example/mo_volmodels/mo_calibration.yaml` | The worked MO run config |
| `quantark/volcalibration/CLAUDE.md` | Module guide, including the `BUILDER_SCHEMA_VERSION` obligation |
| `test/volcalibration/test_runconfig.py`, `test_store_layout.py`, `test_runner_surface.py`, `test_runner_calibration.py`, `test_runner_status.py`, `test_cli.py`, `test_calibration_set.py`, `test_fxdelta.py` | Library-level tests |

**Modified:**

| File | Change |
|---|---|
| `quantark/volcalibration/config.py` | Gains `UnderlyingConfig`, `SurfaceBuildConfig`, `CalibrationRunConfig`, `RunConfig` |
| `quantark/volcalibration/store.py` | Gains `StoreLayout`, atomic JSON IO, both manifest readers/writers |
| `quantark/volcalibration/admission.py` | `AdmissionReason` gains `UNEXPECTED_ERROR` |
| `quantark/volcalibration/__init__.py` | Re-exports the new public names |
| `quantark/backtest/replay/market.py` | `AutocallableMarketDataSet` gains `calibration_set` |
| `CLAUDE.md` (root) | One module table row |
| `docs/superpowers/specs/2026-09-01-volcalibration-module-design.md` | `min_common_strikes` typo; the three design decisions this plan resolves |

**Note on `calibration_set.py`:** spec §4.1 places the `CalibrationSet` loader in `store.py`. It goes in its own module instead, because `CalibrationSet` imports `quantark.priceenv` and `quantark.param.rrf` to build environments while `store.py` is deliberately import-light — Task 20 makes `quantark.backtest.replay.market` import from this package, and pulling the pricing-environment stack into every `store.py` consumer would be a gratuitous cost. `store.py` keeps the bytes-and-paths job; `calibration_set.py` keeps the objects-and-environments job.

**Note on the snapshot boundary:** D1 keeps vendor IO in `example/`. The runner therefore reads canonical snapshot envelopes from `<history>/snapshots/{YYYYMMDD}.json` and never parses a CFFEX CSV. `example/mo_volmodels/export_snapshots.py` (Task 22) is the one-way bridge that makes the existing CSV history readable by the CLI. It preserves each snapshot's `source_sha256` verbatim, so exported snapshots agree with the shas already recorded in migrated manifest records and nothing rebuilds.

---

## Design decisions this plan resolves

The spec leaves three things underdetermined. Each is decided here, with its reason, and Task 22 writes the decision back into the spec.

**DP-1. A foreign exclusion is never rebuilt.** The resume rule (§5.4) compares four fields and rebuilds on any mismatch. An `excluded` record has no artifact sha to compare, so a naive rule rebuilds it — which would silently reverse `exclude_thin_surfaces.py`'s study exclusion and re-admit two surfaces that break 20 of 27 `localvol` runs. Decision: **a record whose `reason` is outside `AdmissionReason` was written by something other than the builder, and a normal run leaves it alone.** `--force` still rebuilds it. This is the same bounded-exception shape as `grandfathered`, and it is discriminated by data, not by a hardcoded date list.

**DP-2. The builder fingerprint stays legacy-shaped until a knob actually moves.** `builder_fingerprint` is sha256 over the canonical JSON of the resolved surface config. If `SurfaceBuildConfig` fingerprinted all seven of its fields, every one of the 787 migrated records would mismatch on its first resumed run and rebuild — destroying exactly the bytes §5.3 protects. Decision: **the fingerprint payload is the five frozen legacy keys, plus any additional knob whose value differs from its frozen default.** Default configs therefore fingerprint identically to history; a changed `extrapolation` or parity gate enters the payload and correctly invalidates. No knob is silently ignored.

**DP-3. The trading calendar is optional, and its absence is visible.** Stage 14 derives `expected_trade_date` from a spot CSV that §5.4's layout does not include. Decision: `paths.spot_csv` is an optional config key. With it, `expected` is the last calendar date at or before `--as-of` and the `spot_cache_*` freshness fields are populated. Without it, `expected` is the newest snapshot on disk and those fields are `null` — the status payload says so rather than implying a freshness check that did not run.

---

### Task 13: Run config and YAML loader

**Files:**
- Modify: `quantark/volcalibration/config.py` (append; the existing `VolModelCalibrationConfig` and `HESTON_PRESETS` are untouched)
- Modify: `docs/superpowers/specs/2026-09-01-volcalibration-module-design.md:~395` (the `min_common_strikes` typo)
- Create: `quantark/volcalibration/yaml_loader.py`
- Test: `test/volcalibration/test_runconfig.py`

**Interfaces:**
- Consumes: `quantark.util.exceptions.ValidationError`; `quantark.volcalibration.store.builder_fingerprint` (tests only)
- Produces:
  - `UnderlyingConfig(symbol: str, convention: str, price_field: str)`
  - `SurfaceBuildConfig(sabr_beta=1.0, min_expiries=2, min_strikes_per_expiry=5, min_common_strikes=3, extrapolation="flat_total_variance", max_abs_implied_rate=0.10, max_rmse_over_forward=0.01)` with `.fingerprint_payload() -> Dict[str, Any]`
  - `CalibrationRunConfig(variants=("localvol","heston","heston_slv"), heston_preset="mo_frozen", heston_max_nfev=200, slv_eta=1.0, slv_n_steps=40, slv_n_x=161, slv_n_z=81, temporal_smoothing=False, structural_ewma_span=20, heston_temporal_regularization=0.0)` with `.manifest_payload() -> Dict[str, Any]`
  - `RunConfig(name, underlying, history_dir, runtime_dir, spot_csv, surface, calibration, workers)`
  - `CONVENTION_LISTED_STRIKE = "listed_strike"`, `CONVENTION_FX_DELTA = "fx_delta"`
  - `load_run_config(path) -> RunConfig`, `load_run_config_text(text, *, base_dir) -> RunConfig`

- [ ] **Step 1: Fix the spec typo**

In `docs/superpowers/specs/2026-09-01-volcalibration-module-design.md`, §6.1's YAML block, change `min_common_strikes: 5` to `min_common_strikes: 3` and append to the paragraph below it:

```
The frozen defaults above are the values the 787 admitted MO surfaces were
built with, read back from the live manifest's `config` block on 2026-09-01;
`min_common_strikes` is 3 (the butterfly check needs three grid strikes), not
5 as an earlier draft of this section said.
```

- [ ] **Step 2: Write the failing test**

Create `test/volcalibration/test_runconfig.py`:

```python
import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.config import (
    CalibrationRunConfig,
    RunConfig,
    SurfaceBuildConfig,
    UnderlyingConfig,
)
from quantark.volcalibration.store import builder_fingerprint
from quantark.volcalibration.yaml_loader import load_run_config_text

# Read back from the live surface manifest's "config" block on 2026-09-01.
# 787 admitted artifacts were built with exactly these values; a default
# SurfaceBuildConfig must fingerprint identically or a resumed run rebuilds
# every one of them (spec 5.3, plan DP-2).
FROZEN_CONFIG_BLOCK = {
    "artifact_schema_version": 1,
    "min_common_strikes": 3,
    "min_expiries": 2,
    "min_strikes_per_expiry": 5,
    "sabr_beta": 1.0,
}

MINIMAL_YAML = """
schema_version: 1
name: mo-daily
underlying:
  symbol: "000852.SH"
  convention: listed_strike
  price_field: settlement
paths:
  root: data/history
"""


def test_default_surface_config_fingerprints_as_the_frozen_history():
    assert SurfaceBuildConfig().fingerprint_payload() == FROZEN_CONFIG_BLOCK
    assert builder_fingerprint(
        SurfaceBuildConfig().fingerprint_payload()
    ) == builder_fingerprint(FROZEN_CONFIG_BLOCK)


def test_a_moved_knob_enters_the_fingerprint():
    moved = SurfaceBuildConfig(max_abs_implied_rate=0.05)
    payload = moved.fingerprint_payload()
    assert payload["max_abs_implied_rate"] == 0.05
    assert set(payload) - set(FROZEN_CONFIG_BLOCK) == {"max_abs_implied_rate"}
    assert builder_fingerprint(payload) != builder_fingerprint(FROZEN_CONFIG_BLOCK)


def test_calibration_manifest_payload_matches_the_live_shape():
    # Keys and order-insensitive content of the live calibration manifest's
    # per-record "config" block, so an existing record stays "current".
    assert CalibrationRunConfig().manifest_payload() == {
        "variants": ["localvol", "heston", "heston_slv"],
        "heston_preset": "mo_frozen",
        "heston_max_nfev": 200,
        "slv_eta": 1.0,
        "slv_n_steps": 40,
        "slv_n_x": 161,
        "slv_n_z": 81,
    }


def test_temporal_smoothing_adds_the_scheme_block():
    payload = CalibrationRunConfig(
        temporal_smoothing=True, structural_ewma_span=19
    ).manifest_payload()
    scheme = payload["temporal_scheme"]
    assert scheme["name"] == "daily_v0_structural_ewma"
    assert scheme["structural_ewma_span"] == 19
    assert scheme["structural_ewma_alpha"] == pytest.approx(2.0 / 20.0)


def test_yaml_defaults_reproduce_today(tmp_path):
    config = load_run_config_text(MINIMAL_YAML, base_dir=tmp_path)
    assert config.name == "mo-daily"
    assert config.underlying.price_field == "settlement"
    assert config.history_dir == tmp_path / "data/history"
    # runtime defaults to the surface root: one directory unless told otherwise
    assert config.runtime_dir == config.history_dir
    assert config.spot_csv is None
    assert config.surface.fingerprint_payload() == FROZEN_CONFIG_BLOCK
    assert config.workers == 1


def test_split_roots_are_expressible(tmp_path):
    text = MINIMAL_YAML + "  runtime: ../../output/mo\n  spot_csv: data/history/csi1000_spot.csv\n"
    config = load_run_config_text(text, base_dir=tmp_path)
    assert config.runtime_dir == (tmp_path / "../../output/mo").resolve()
    assert config.spot_csv == tmp_path / "data/history/csi1000_spot.csv"


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("schema_version: 2", "schema_version"),
        ("  price_field: mid\n", "price_field"),
        ("  convention: guess\n", "convention"),
        ("surprise: 1\n", "surprise"),
    ],
)
def test_the_loader_is_fail_closed(tmp_path, mutation, message):
    if mutation.startswith("schema_version"):
        text = MINIMAL_YAML.replace("schema_version: 1", mutation)
    elif mutation.startswith("surprise"):
        text = MINIMAL_YAML + mutation
    else:
        key = mutation.strip().split(":")[0]
        text = "\n".join(
            line for line in MINIMAL_YAML.splitlines() if not line.strip().startswith(key)
        ) + "\n" + mutation
    with pytest.raises(ValidationError) as exc:
        load_run_config_text(text, base_dir=tmp_path)
    assert message in str(exc.value)


def test_run_config_rejects_a_non_positive_worker_count(tmp_path):
    with pytest.raises(ValidationError):
        RunConfig(
            name="x",
            underlying=UnderlyingConfig("S", "listed_strike", "settlement"),
            history_dir=tmp_path,
            runtime_dir=tmp_path,
            spot_csv=None,
            surface=SurfaceBuildConfig(),
            calibration=CalibrationRunConfig(),
            workers=0,
        )
```

- [ ] **Step 3: Run it to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/volcalibration/test_runconfig.py -x -q
```
Expected: FAIL — `ImportError: cannot import name 'SurfaceBuildConfig'`.

- [ ] **Step 4: Append the config tree to `config.py`**

```python
# --- Run configuration -----------------------------------------------------

# CONVENTION_LISTED, CONVENTION_FX_DELTA, CONVENTIONS, PRICE_FIELD_SETTLEMENT
# and PRICE_FIELD_MID_OR_LAST already exist in snapshot.py -- import them, do
# not redeclare them.  Two copies of a vocabulary drift.

TEMPORAL_SCHEME = "daily_v0_structural_ewma"
HESTON_STRUCTURAL_PARAMETER_NAMES = ("kappa", "theta", "sigma", "rho")

# Mirrors surface.ARTIFACT_SCHEMA_VERSION.  Declared as a literal rather than
# imported so that quantark.backtest.replay.config -- which imports this
# module -- does not pull numpy/scipy through surface.py.  test_runconfig.py
# asserts the two agree.
_ARTIFACT_SCHEMA_VERSION_IN_FINGERPRINT = 1


@dataclass(frozen=True)
class UnderlyingConfig:
    """What is being calibrated and which quote convention its snapshots use."""

    symbol: str
    convention: str
    price_field: str

    def __post_init__(self) -> None:
        if not str(self.symbol).strip():
            raise ValidationError("underlying.symbol must be non-empty")
        if self.convention not in CONVENTIONS:
            raise ValidationError(
                f"underlying.convention must be one of {CONVENTIONS}, "
                f"got {self.convention!r}"
            )
        if self.price_field not in (PRICE_FIELD_SETTLEMENT, PRICE_FIELD_MID_OR_LAST):
            raise ValidationError(
                "underlying.price_field must be "
                f"{PRICE_FIELD_SETTLEMENT!r} or {PRICE_FIELD_MID_OR_LAST!r}, "
                f"got {self.price_field!r}"
            )


@dataclass(frozen=True)
class SurfaceBuildConfig:
    """Every value stage 03 hard-coded, with that value as its default."""

    sabr_beta: float = 1.0
    min_expiries: int = 2
    min_strikes_per_expiry: int = 5
    min_common_strikes: int = 3
    extrapolation: str = "flat_total_variance"
    max_abs_implied_rate: float = 0.10
    max_rmse_over_forward: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.sabr_beta) <= 1.0:
            raise ValidationError("surface.sabr_beta must lie in [0, 1]")
        for name in ("min_expiries", "min_strikes_per_expiry", "min_common_strikes"):
            if int(getattr(self, name)) < 2:
                raise ValidationError(f"surface.{name} must be at least 2")
        if float(self.max_abs_implied_rate) <= 0.0:
            raise ValidationError("surface.parity_gate.max_abs_implied_rate must be positive")
        if float(self.max_rmse_over_forward) <= 0.0:
            raise ValidationError("surface.parity_gate.max_rmse_over_forward must be positive")

    def fingerprint_payload(self) -> Dict[str, Any]:
        """The canonical block `builder_fingerprint` hashes.

        The five legacy keys are the ones the 787 admitted artifacts were built
        with; emitting exactly them keeps a default run's fingerprint identical
        to migrated history, so nothing rebuilds (plan DP-2).  A knob moved off
        its frozen default is appended, which changes the fingerprint and
        correctly invalidates -- so no knob is silently ignored.
        """
        payload: Dict[str, Any] = {
            "sabr_beta": float(self.sabr_beta),
            "min_expiries": int(self.min_expiries),
            "min_strikes_per_expiry": int(self.min_strikes_per_expiry),
            "min_common_strikes": int(self.min_common_strikes),
            "artifact_schema_version": _ARTIFACT_SCHEMA_VERSION_IN_FINGERPRINT,
        }
        defaults = SurfaceBuildConfig()
        for name in ("extrapolation", "max_abs_implied_rate", "max_rmse_over_forward"):
            value = getattr(self, name)
            if value != getattr(defaults, name):
                payload[name] = value
        return payload


@dataclass(frozen=True)
class CalibrationRunConfig:
    """Which vol models to fit and with what solver settings."""

    variants: tuple = ("localvol", "heston", "heston_slv")
    heston_preset: str = "mo_frozen"
    heston_max_nfev: int = 200
    slv_eta: float = 1.0
    slv_n_steps: int = 40
    slv_n_x: int = 161
    slv_n_z: int = 81
    temporal_smoothing: bool = False
    structural_ewma_span: int = 20
    heston_temporal_regularization: float = 0.0

    def __post_init__(self) -> None:
        unknown = [v for v in self.variants if v not in VOL_MODEL_VARIANTS]
        if unknown:
            raise ValidationError(
                f"calibration.variants: unknown {unknown}; "
                f"known variants are {list(VOL_MODEL_VARIANTS)}"
            )
        if not self.variants:
            raise ValidationError("calibration.variants must not be empty")
        if self.heston_preset not in HESTON_PRESETS:
            raise ValidationError(
                f"calibration.heston_preset must be one of {sorted(HESTON_PRESETS)}"
            )
        if int(self.structural_ewma_span) < 1:
            raise ValidationError("calibration.structural_ewma_span must be at least 1")

    def manifest_payload(self) -> Dict[str, Any]:
        """The per-record `config` block, byte-shaped like stage 14's."""
        payload: Dict[str, Any] = {
            "variants": list(self.variants),
            "heston_preset": str(self.heston_preset),
            "heston_max_nfev": int(self.heston_max_nfev),
            "slv_eta": float(self.slv_eta),
            "slv_n_steps": int(self.slv_n_steps),
            "slv_n_x": int(self.slv_n_x),
            "slv_n_z": int(self.slv_n_z),
        }
        if self.temporal_smoothing:
            span = int(self.structural_ewma_span)
            payload["temporal_scheme"] = {
                "name": TEMPORAL_SCHEME,
                "structural_ewma_span": span,
                "structural_ewma_alpha": 2.0 / (span + 1.0),
                "heston_temporal_regularization": float(
                    self.heston_temporal_regularization
                ),
                "daily_parameters": ["v0"],
                "structural_parameters": list(HESTON_STRUCTURAL_PARAMETER_NAMES),
            }
        return payload

    def calibrator_config(self, cache_dir) -> "VolModelCalibrationConfig":
        """The engine-level config this run's calibrator is built from."""
        return VolModelCalibrationConfig(
            cache_dir=str(cache_dir),
            heston_preset=self.heston_preset,
            heston_max_nfev=int(self.heston_max_nfev),
            slv_eta=float(self.slv_eta),
            slv_n_steps=int(self.slv_n_steps),
            slv_n_x=int(self.slv_n_x),
            slv_n_z=int(self.slv_n_z),
        )


@dataclass(frozen=True)
class RunConfig:
    """One resolved, versionable calibration run."""

    name: str
    underlying: UnderlyingConfig
    history_dir: Path
    runtime_dir: Path
    spot_csv: Optional[Path]
    surface: SurfaceBuildConfig
    calibration: CalibrationRunConfig
    workers: int = 1

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValidationError("name must be non-empty")
        if int(self.workers) < 1:
            raise ValidationError("run.workers must be at least 1")

    def echo(self) -> Dict[str, Any]:
        """The resolved config, as echoed into both manifests and status."""
        return {
            "name": self.name,
            "underlying": {
                "symbol": self.underlying.symbol,
                "convention": self.underlying.convention,
                "price_field": self.underlying.price_field,
            },
            "surface": self.surface.fingerprint_payload(),
            "calibration": self.calibration.manifest_payload(),
            "workers": int(self.workers),
        }
```

Add the imports this needs at the top of `config.py`: `from pathlib import Path`, `from typing import Tuple`, and
```python
from quantark.volcalibration.snapshot import (
    CONVENTION_FX_DELTA,
    CONVENTION_LISTED,
    CONVENTIONS,
    PRICE_FIELD_MID_OR_LAST,
    PRICE_FIELD_SETTLEMENT,
)
```
`VOL_MODEL_VARIANTS` comes from `quantark.volcalibration.calibrate`; import it lazily inside `CalibrationRunConfig.__post_init__` to avoid `config.py` → `calibrate.py` → `config.py` at import time:
```python
        from quantark.volcalibration.calibrate import VOL_MODEL_VARIANTS
```

- [ ] **Step 5: Write `yaml_loader.py`**

```python
"""YAML -> RunConfig, fail-closed.

A run config is declarative and diffable: one versionable file reproduces a
run.  The loader validates structure and rejects unknown keys, because a typo
that silently falls back to a default would produce a run nobody asked for --
and the resulting artifacts would carry a fingerprint nobody can explain.
Every structural error names the YAML path that caused it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.config import (
    CalibrationRunConfig,
    RunConfig,
    SurfaceBuildConfig,
    UnderlyingConfig,
)

SCHEMA_VERSION = 1

TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "name", "underlying", "paths", "surface", "calibration", "run"}
)
UNDERLYING_KEYS = frozenset({"symbol", "convention", "price_field"})
PATHS_KEYS = frozenset({"root", "runtime", "spot_csv"})
SURFACE_KEYS = frozenset(
    {
        "sabr_beta",
        "min_expiries",
        "min_strikes_per_expiry",
        "min_common_strikes",
        "extrapolation",
        "parity_gate",
    }
)
PARITY_GATE_KEYS = frozenset({"max_abs_implied_rate", "max_rmse_over_forward"})
CALIBRATION_KEYS = frozenset(
    {
        "variants",
        "heston_preset",
        "heston_max_nfev",
        "slv",
        "temporal_smoothing",
        "structural_ewma_span",
        "heston_temporal_regularization",
    }
)
SLV_KEYS = frozenset({"eta", "n_steps", "n_x", "n_z"})
RUN_KEYS = frozenset({"workers"})


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValidationError(f"{path} must be a mapping, got {type(value).__name__}")
    return value


def _reject_unknown(block: Mapping[str, Any], known: frozenset, path: str) -> None:
    unknown = sorted(set(block) - known)
    if unknown:
        raise ValidationError(
            f"{path}: unknown key(s) {unknown}; known keys are {sorted(known)}"
        )


def _require(block: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in block:
        raise ValidationError(f"missing required key {path}.{key}")
    return block[key]


def load_run_config_text(text: str, *, base_dir) -> RunConfig:
    """Parse run-config YAML. Relative paths resolve against ``base_dir``."""
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValidationError(f"run config is not valid YAML: {exc}") from exc
    document = _mapping(document, "run config")
    _reject_unknown(document, TOP_LEVEL_KEYS, "run config")

    version = document.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValidationError(
            f"run config schema_version must be {SCHEMA_VERSION}, got {version!r}"
        )

    underlying_block = _mapping(_require(document, "underlying", "run config"), "underlying")
    _reject_unknown(underlying_block, UNDERLYING_KEYS, "underlying")
    underlying = UnderlyingConfig(
        symbol=str(_require(underlying_block, "symbol", "underlying")),
        convention=str(_require(underlying_block, "convention", "underlying")),
        price_field=str(_require(underlying_block, "price_field", "underlying")),
    )

    paths_block = _mapping(_require(document, "paths", "run config"), "paths")
    _reject_unknown(paths_block, PATHS_KEYS, "paths")
    root = Path(base_dir) / str(_require(paths_block, "root", "paths"))
    runtime = (
        (Path(base_dir) / str(paths_block["runtime"])).resolve()
        if "runtime" in paths_block
        else root
    )
    spot_csv: Optional[Path] = (
        Path(base_dir) / str(paths_block["spot_csv"]) if "spot_csv" in paths_block else None
    )

    surface_block = _mapping(document.get("surface"), "surface")
    _reject_unknown(surface_block, SURFACE_KEYS, "surface")
    gate = _mapping(surface_block.get("parity_gate"), "surface.parity_gate")
    _reject_unknown(gate, PARITY_GATE_KEYS, "surface.parity_gate")
    defaults = SurfaceBuildConfig()
    surface = SurfaceBuildConfig(
        sabr_beta=float(surface_block.get("sabr_beta", defaults.sabr_beta)),
        min_expiries=int(surface_block.get("min_expiries", defaults.min_expiries)),
        min_strikes_per_expiry=int(
            surface_block.get("min_strikes_per_expiry", defaults.min_strikes_per_expiry)
        ),
        min_common_strikes=int(
            surface_block.get("min_common_strikes", defaults.min_common_strikes)
        ),
        extrapolation=str(surface_block.get("extrapolation", defaults.extrapolation)),
        max_abs_implied_rate=float(
            gate.get("max_abs_implied_rate", defaults.max_abs_implied_rate)
        ),
        max_rmse_over_forward=float(
            gate.get("max_rmse_over_forward", defaults.max_rmse_over_forward)
        ),
    )

    calibration_block = _mapping(document.get("calibration"), "calibration")
    _reject_unknown(calibration_block, CALIBRATION_KEYS, "calibration")
    slv = _mapping(calibration_block.get("slv"), "calibration.slv")
    _reject_unknown(slv, SLV_KEYS, "calibration.slv")
    cal_defaults = CalibrationRunConfig()
    calibration = CalibrationRunConfig(
        variants=tuple(calibration_block.get("variants", cal_defaults.variants)),
        heston_preset=str(
            calibration_block.get("heston_preset", cal_defaults.heston_preset)
        ),
        heston_max_nfev=int(
            calibration_block.get("heston_max_nfev", cal_defaults.heston_max_nfev)
        ),
        slv_eta=float(slv.get("eta", cal_defaults.slv_eta)),
        slv_n_steps=int(slv.get("n_steps", cal_defaults.slv_n_steps)),
        slv_n_x=int(slv.get("n_x", cal_defaults.slv_n_x)),
        slv_n_z=int(slv.get("n_z", cal_defaults.slv_n_z)),
        temporal_smoothing=bool(
            calibration_block.get("temporal_smoothing", cal_defaults.temporal_smoothing)
        ),
        structural_ewma_span=int(
            calibration_block.get(
                "structural_ewma_span", cal_defaults.structural_ewma_span
            )
        ),
        heston_temporal_regularization=float(
            calibration_block.get(
                "heston_temporal_regularization",
                cal_defaults.heston_temporal_regularization,
            )
        ),
    )

    run_block = _mapping(document.get("run"), "run")
    _reject_unknown(run_block, RUN_KEYS, "run")

    return RunConfig(
        name=str(_require(document, "name", "run config")),
        underlying=underlying,
        history_dir=root,
        runtime_dir=runtime,
        spot_csv=spot_csv,
        surface=surface,
        calibration=calibration,
        workers=int(run_block.get("workers", 1)),
    )


def load_run_config(path) -> RunConfig:
    """Load a run config from a YAML file."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"cannot read run config {source}: {exc}") from exc
    return load_run_config_text(text, base_dir=source.parent)
```

- [ ] **Step 6: Run the tests**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/volcalibration/test_runconfig.py -q
```
Expected: PASS (10 tests).

- [ ] **Step 7: Commit**

```bash
git add quantark/volcalibration/config.py quantark/volcalibration/yaml_loader.py test/volcalibration/test_runconfig.py
git add -f docs/superpowers/specs/2026-09-01-volcalibration-module-design.md
git commit -m "$(cat <<'EOF'
feat(volcalibration): run config whose defaults fingerprint as frozen history

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YSMxE68spnG5uF97YdTCGU
EOF
)"
```

---

### Task 14: Store layout and both manifests

**Files:**
- Modify: `quantark/volcalibration/store.py`
- Test: `test/volcalibration/test_store_layout.py`

**Interfaces:**
- Consumes: `RunConfig` (Task 13)
- Produces:
  - `StoreLayout(history_dir, runtime_dir)` with properties `snapshots_dir`, `surface_dir`, `surface_manifest`, `calibration_cache`, `calibration_manifest`, `status`, `lock`, and methods `snapshot_path(tag)`, `artifact_path(tag)`
  - `StoreLayout.from_config(config) -> StoreLayout`
  - `read_json(path, *, default)`, `atomic_write_json(path, payload)`, `atomic_write_bytes(path, data)`
  - `load_surface_manifest(layout) -> Tuple[dict, Dict[str, dict]]`
  - `save_surface_manifest(layout, records, *, config, window, price_field, source_class)`
  - `load_calibration_manifest(layout) -> Tuple[dict, Dict[str, dict]]`
  - `save_calibration_manifest(layout, base_payload, records, *, config)`
  - `SURFACE_MANIFEST_SCHEMA_VERSION = 1`, `CALIBRATION_MANIFEST_SCHEMA_VERSION = 1`

- [ ] **Step 1: Write the failing test**

Create `test/volcalibration/test_store_layout.py`:

```python
import json

import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.config import (
    CalibrationRunConfig,
    RunConfig,
    SurfaceBuildConfig,
    UnderlyingConfig,
)
from quantark.volcalibration.store import (
    StoreLayout,
    atomic_write_json,
    load_calibration_manifest,
    load_surface_manifest,
    read_json,
    save_calibration_manifest,
    save_surface_manifest,
    surface_record,
)


def _config(tmp_path, runtime=None):
    return RunConfig(
        name="t",
        underlying=UnderlyingConfig("000852.SH", "listed_strike", "settlement"),
        history_dir=tmp_path / "history",
        runtime_dir=runtime or (tmp_path / "history"),
        spot_csv=None,
        surface=SurfaceBuildConfig(),
        calibration=CalibrationRunConfig(),
        workers=1,
    )


def test_layout_splits_history_from_runtime(tmp_path):
    layout = StoreLayout.from_config(_config(tmp_path, runtime=tmp_path / "out"))
    assert layout.snapshot_path("20260430") == tmp_path / "history/snapshots/20260430.json"
    assert (
        layout.artifact_path("20260430")
        == tmp_path / "history/iv_surface/mo_iv_surface_20260430.json"
    )
    assert layout.surface_manifest == tmp_path / "history/surface_manifest.json"
    assert layout.calibration_manifest == tmp_path / "out/calibration_manifest.json"
    assert layout.calibration_cache == tmp_path / "out/calibration_cache"
    assert layout.status == tmp_path / "out/status.json"
    assert layout.lock == tmp_path / "out/pipeline.lock"


def test_atomic_write_leaves_no_temp_file_and_reads_back(tmp_path):
    target = tmp_path / "deep" / "payload.json"
    atomic_write_json(target, {"b": 2, "a": 1})
    assert json.loads(target.read_text()) == {"a": 1, "b": 2}
    assert sorted(p.name for p in target.parent.iterdir()) == ["payload.json"]
    assert read_json(target, default=None) == {"a": 1, "b": 2}
    assert read_json(tmp_path / "absent.json", default={"records": []}) == {"records": []}


def test_a_corrupt_manifest_raises_rather_than_defaulting(tmp_path):
    layout = StoreLayout.from_config(_config(tmp_path))
    layout.surface_manifest.parent.mkdir(parents=True, exist_ok=True)
    layout.surface_manifest.write_text("{ not json")
    with pytest.raises(ValidationError):
        load_surface_manifest(layout)


def test_saving_the_surface_manifest_preserves_foreign_top_level_blocks(tmp_path):
    layout = StoreLayout.from_config(_config(tmp_path))
    config = _config(tmp_path)
    layout.surface_manifest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        layout.surface_manifest,
        {
            "schema_version": 1,
            "records": [],
            "study_admission": {"vol_model_backtest": {"excluded_dates": ["20240930"]}},
        },
    )
    records = {
        "20260430": surface_record("20260430", status="ok", n_expiries=6, artifact_sha256="a")
    }
    save_surface_manifest(
        layout,
        records,
        config=config.surface.fingerprint_payload(),
        window={"start": "20260430", "end": "20260430"},
        price_field="settlement",
        source_class="official_cffex_eod_settlement",
    )
    payload, by_date = load_surface_manifest(layout)
    # exclude_thin_surfaces.py writes study_admission; the builder must not eat it
    assert payload["study_admission"]["vol_model_backtest"]["excluded_dates"] == ["20240930"]
    assert payload["price_field"] == "settlement"
    assert payload["config"]["min_common_strikes"] == 3
    assert list(by_date) == ["20260430"]


def test_calibration_manifest_round_trips_and_sorts(tmp_path):
    layout = StoreLayout.from_config(_config(tmp_path))
    base, records = load_calibration_manifest(layout)
    assert records == {}
    assert base["bootstrap_policy"] == "latest_admitted_surface_only"
    base["baseline_date"] = "20260430"
    save_calibration_manifest(
        layout,
        base,
        {"20260501": {"date": "20260501"}, "20260430": {"date": "20260430"}},
        config=CalibrationRunConfig().manifest_payload(),
    )
    payload, by_date = load_calibration_manifest(layout)
    assert [r["date"] for r in payload["records"]] == ["20260430", "20260501"]
    assert payload["baseline_date"] == "20260430"
    assert sorted(by_date) == ["20260430", "20260501"]


def test_an_unsupported_calibration_schema_is_refused(tmp_path):
    layout = StoreLayout.from_config(_config(tmp_path))
    atomic_write_json(layout.calibration_manifest, {"schema_version": 99, "records": []})
    with pytest.raises(ValidationError):
        load_calibration_manifest(layout)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/volcalibration/test_store_layout.py -x -q
```
Expected: FAIL — `ImportError: cannot import name 'StoreLayout'`.

- [ ] **Step 3: Extend `store.py`**

Append to `quantark/volcalibration/store.py` (and add `import os`, `import tempfile`, `from dataclasses import dataclass`, `from datetime import datetime, timezone`, `from typing import Dict, Tuple`, `from quantark.util.exceptions import ValidationError` at the top):

```python
SURFACE_MANIFEST_SCHEMA_VERSION = 1
CALIBRATION_MANIFEST_SCHEMA_VERSION = 1
GAP_POLICY = "consumers carry forward previous admitted surface"
BOOTSTRAP_POLICY = "latest_admitted_surface_only"


@dataclass(frozen=True)
class StoreLayout:
    """Where a run's inputs, artifacts and manifests live.

    Two roots, not one: the MO deployment keeps surfaces beside the example
    suite's data and calibration output under ``output/``.  ``runtime_dir``
    defaults to ``history_dir`` so a single-root store stays a single key in
    the YAML.
    """

    history_dir: Path
    runtime_dir: Path

    @classmethod
    def from_config(cls, config) -> "StoreLayout":
        return cls(Path(config.history_dir), Path(config.runtime_dir))

    @property
    def snapshots_dir(self) -> Path:
        return self.history_dir / "snapshots"

    @property
    def surface_dir(self) -> Path:
        return self.history_dir / "iv_surface"

    @property
    def surface_manifest(self) -> Path:
        return self.history_dir / "surface_manifest.json"

    @property
    def calibration_cache(self) -> Path:
        return self.runtime_dir / "calibration_cache"

    @property
    def calibration_manifest(self) -> Path:
        return self.runtime_dir / "calibration_manifest.json"

    @property
    def status(self) -> Path:
        return self.runtime_dir / "status.json"

    @property
    def lock(self) -> Path:
        return self.runtime_dir / "pipeline.lock"

    def snapshot_path(self, trade_date: str) -> Path:
        return self.snapshots_dir / f"{trade_date}.json"

    def artifact_path(self, trade_date: str) -> Path:
        return artifact_path(self.surface_dir, trade_date)

    def available_snapshot_dates(self) -> list:
        """Trade-date tags with a snapshot on disk, ascending."""
        if not self.snapshots_dir.is_dir():
            return []
        return sorted(
            p.stem for p in self.snapshots_dir.glob("*.json") if p.stem.isdigit()
        )


def _iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path, *, default):
    """Read a JSON file, returning ``default`` only when it does not exist.

    An unreadable or malformed file raises: silently falling back to a default
    would rewrite a corrupt manifest as an empty one and orphan every artifact.
    """
    target = Path(path)
    if not target.is_file():
        return default
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read {target}: {exc}") from exc


def atomic_write_bytes(path, data: bytes) -> None:
    """Write bytes through a same-directory temp file and one ``os.replace``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=target.parent, prefix=target.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path, payload) -> None:
    """Deterministic, atomic JSON write: sorted keys, no NaN, trailing newline."""
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=str)
    atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def _records_by_date(payload, source: str) -> Dict[str, dict]:
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValidationError(f"{source}: 'records' must be a list")
    out: Dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict) or "date" not in record:
            raise ValidationError(f"{source}: every record needs a 'date'")
        out[str(record["date"])] = dict(record)
    return out


def load_surface_manifest(layout: StoreLayout) -> Tuple[dict, Dict[str, dict]]:
    """Full manifest payload plus its records keyed by date tag."""
    payload = read_json(
        layout.surface_manifest,
        default={"schema_version": SURFACE_MANIFEST_SCHEMA_VERSION, "records": []},
    )
    if payload.get("schema_version") != SURFACE_MANIFEST_SCHEMA_VERSION:
        raise ValidationError(
            f"{layout.surface_manifest}: unsupported schema_version "
            f"{payload.get('schema_version')!r}"
        )
    return payload, _records_by_date(payload, str(layout.surface_manifest))


def save_surface_manifest(
    layout: StoreLayout,
    records: Mapping[str, Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    window: Mapping[str, str],
    price_field: str,
    source_class: str,
) -> None:
    """Rewrite the surface manifest, preserving foreign top-level blocks.

    ``exclude_thin_surfaces.py`` writes a ``study_admission`` block at top
    level; overwriting the file wholesale would drop the record of why two
    dates are excluded.  Everything not owned by the builder is carried
    forward untouched.
    """
    previous = read_json(layout.surface_manifest, default={})
    payload = dict(previous) if isinstance(previous, dict) else {}
    payload.update(
        {
            "schema_version": SURFACE_MANIFEST_SCHEMA_VERSION,
            "source": source_class,
            "price_field": price_field,
            "generated_at": _iso_utc(),
            "window": dict(window),
            "gap_policy": GAP_POLICY,
            "config": dict(config),
            "records": [dict(records[tag]) for tag in sorted(records)],
        }
    )
    atomic_write_json(layout.surface_manifest, payload)


def load_calibration_manifest(layout: StoreLayout) -> Tuple[dict, Dict[str, dict]]:
    payload = read_json(
        layout.calibration_manifest,
        default={
            "schema_version": CALIBRATION_MANIFEST_SCHEMA_VERSION,
            "records": [],
            "bootstrap_policy": BOOTSTRAP_POLICY,
        },
    )
    if payload.get("schema_version") != CALIBRATION_MANIFEST_SCHEMA_VERSION:
        raise ValidationError(
            f"{layout.calibration_manifest}: unsupported schema_version "
            f"{payload.get('schema_version')!r}"
        )
    return payload, _records_by_date(payload, str(layout.calibration_manifest))


def save_calibration_manifest(
    layout: StoreLayout,
    base_payload: Mapping[str, Any],
    records: Mapping[str, Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> None:
    payload = dict(base_payload)
    payload.update(
        {
            "schema_version": CALIBRATION_MANIFEST_SCHEMA_VERSION,
            "generated_at": _iso_utc(),
            "config": dict(config),
            "records": [dict(records[tag]) for tag in sorted(records)],
        }
    )
    payload.setdefault("bootstrap_policy", BOOTSTRAP_POLICY)
    atomic_write_json(layout.calibration_manifest, payload)
```

- [ ] **Step 4: Run the tests**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/volcalibration/test_store_layout.py test/volcalibration/test_store.py -q
```
Expected: PASS (6 new + 7 existing).

- [ ] **Step 5: Commit**

```bash
git add quantark/volcalibration/store.py test/volcalibration/test_store_layout.py
git commit -m "$(cat <<'EOF'
feat(volcalibration): store layout and both manifests, foreign blocks preserved

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YSMxE68spnG5uF97YdTCGU
EOF
)"
```

---

### Task 15: Runner — the surface stage and its resume rule

**Files:**
- Create: `quantark/volcalibration/runner.py`
- Modify: `quantark/volcalibration/admission.py` (one enum member)
- Test: `test/volcalibration/test_runner_surface.py`

**Interfaces:**
- Consumes: `StoreLayout`, `save_surface_manifest`, `surface_record`, `builder_fingerprint`, `BUILDER_SCHEMA_VERSION`, `atomic_write_bytes`, `serialize_artifact` (Task 14 + phase 2); `QuoteSnapshot.from_payload`; `SettlementNormalizer`, `ListedNormalizer`; `build_artifact`
- Produces:
  - `SurfaceTask` (a plain tuple alias documented in the module)
  - `normalizer_for(config) -> QuoteNormalizer`
  - `surface_record_is_current(record, *, snapshot_sha, price_field, fingerprint) -> bool`
  - `plan_surface_dates(layout, records, config, *, tags, force) -> List[str]`
  - `build_one_surface(task) -> dict` (module-level, picklable)
  - `run_surface_stage(layout, config, *, tags=None, force=False, log=...) -> Dict[str, dict]`

- [ ] **Step 1: Add the missing reason code**

In `quantark/volcalibration/admission.py`, add to `AdmissionReason`:

```python
    UNEXPECTED_ERROR = "unexpected_error"
```

Stage 03 writes this reason for anything that escapes `AdmissionError`; without it the vocabulary is incomplete and `_reason_is_builder_owned` below would treat a builder-written record as foreign.

- [ ] **Step 2: Write the failing test**

Create `test/volcalibration/test_runner_surface.py`:

```python
import json
from datetime import date

import pytest

from quantark.volcalibration.config import (
    CalibrationRunConfig,
    RunConfig,
    SurfaceBuildConfig,
    UnderlyingConfig,
)
from quantark.volcalibration.runner import (
    plan_surface_dates,
    run_surface_stage,
    surface_record_is_current,
)
from quantark.volcalibration.snapshot import QuoteSnapshot
from quantark.volcalibration.store import (
    BUILDER_SCHEMA_VERSION,
    StoreLayout,
    builder_fingerprint,
    load_surface_manifest,
    surface_record,
)

SAMPLE = "example/mo_volmodels/data/mo_settlement_snapshot_20260430.json"
TAG = "20260430"
SPOT = 8381.947  # the s0 the stored 20260430 artifact was built with


def _config(tmp_path, **surface_kwargs):
    return RunConfig(
        name="t",
        underlying=UnderlyingConfig("000852.SH", "listed_strike", "settlement"),
        history_dir=tmp_path,
        runtime_dir=tmp_path,
        spot_csv=None,
        surface=SurfaceBuildConfig(**surface_kwargs),
        calibration=CalibrationRunConfig(),
        workers=1,
    )


@pytest.fixture
def store(tmp_path):
    """A store holding one canonical snapshot for 2026-04-30."""
    config = _config(tmp_path)
    layout = StoreLayout.from_config(config)
    payload = json.loads(open(SAMPLE, encoding="utf-8").read())
    snap = QuoteSnapshot.from_legacy_settlement(
        payload,
        trade_date=date(2026, 4, 30),
        spot=SPOT,
        symbol="000852.SH",
        source_sha256=payload["source_sha256"],
        source_url=payload.get("source_url"),
    )
    layout.snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.snapshot_path(TAG).write_text(json.dumps(snap.to_payload()), encoding="utf-8")
    return config, layout, snap


def test_a_fresh_store_builds_and_records_provenance(store):
    config, layout, snap = store
    records = run_surface_stage(layout, config)
    assert records[TAG]["status"] == "ok"
    assert layout.artifact_path(TAG).is_file()
    assert records[TAG]["snapshot_sha256"] == snap.sha256
    assert records[TAG]["price_field"] == "settlement"
    assert records[TAG]["builder_fingerprint"] == builder_fingerprint(
        config.surface.fingerprint_payload()
    )
    assert records[TAG]["provenance"] == "verified"


def test_a_second_run_skips_and_leaves_the_bytes_identical(store):
    config, layout, _ = store
    run_surface_stage(layout, config)
    before = layout.artifact_path(TAG).read_bytes()
    mtime = layout.artifact_path(TAG).stat().st_mtime_ns
    assert plan_surface_dates(layout, load_surface_manifest(layout)[1], config,
                              tags=[TAG], force=False) == []
    run_surface_stage(layout, config)
    assert layout.artifact_path(TAG).read_bytes() == before
    assert layout.artifact_path(TAG).stat().st_mtime_ns == mtime


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda r: r.update(snapshot_sha256="different"), id="republished_snapshot"),
        pytest.param(lambda r: r.update(price_field="mid_or_last"), id="price_field"),
        pytest.param(lambda r: r.update(builder_fingerprint="stale"), id="builder_config"),
        pytest.param(lambda r: r.update(builder_schema_version=BUILDER_SCHEMA_VERSION + 1),
                     id="builder_code"),
    ],
)
def test_each_invalidation_field_forces_a_rebuild(store, mutate):
    config, layout, _ = store
    records = run_surface_stage(layout, config)
    record = dict(records[TAG])
    mutate(record)
    assert plan_surface_dates(layout, {TAG: record}, config, tags=[TAG], force=False) == [TAG]


def test_a_grandfathered_record_is_trusted_not_rebuilt(store):
    config, layout, _ = store
    record = surface_record(TAG, status="ok", n_expiries=6, artifact_sha256="a",
                            provenance="grandfathered")
    assert surface_record_is_current(record, snapshot_sha="anything",
                                     price_field="settlement", fingerprint="anything")
    assert plan_surface_dates(layout, {TAG: record}, config, tags=[TAG], force=False) == []


def test_a_foreign_exclusion_is_never_resurrected(store):
    """exclude_thin_surfaces.py's record must survive a normal run (plan DP-1)."""
    config, layout, _ = store
    record = surface_record(TAG, status="excluded",
                            reason="insufficient_expiries_for_dupire", n_expiries=2)
    assert plan_surface_dates(layout, {TAG: record}, config, tags=[TAG], force=False) == []
    assert plan_surface_dates(layout, {TAG: record}, config, tags=[TAG], force=True) == [TAG]


def test_a_builder_exclusion_is_retried_when_the_snapshot_changes(store):
    config, layout, _ = store
    record = surface_record(TAG, status="excluded", reason="static_arbitrage",
                            snapshot_sha256="old", price_field="settlement",
                            fingerprint=builder_fingerprint(
                                config.surface.fingerprint_payload()))
    assert plan_surface_dates(layout, {TAG: record}, config, tags=[TAG], force=False) == [TAG]


def test_a_moved_knob_rebuilds_because_the_fingerprint_moves(store):
    config, layout, _ = store
    records = run_surface_stage(layout, config)
    moved = _config(layout.history_dir, max_abs_implied_rate=0.05)
    assert plan_surface_dates(layout, records, moved, tags=[TAG], force=False) == [TAG]
```

- [ ] **Step 3: Run it to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/volcalibration/test_runner_surface.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'quantark.volcalibration.runner'`.

- [ ] **Step 4: Write the surface half of `runner.py`**

```python
"""Orchestration: per-date surface build and per-date calibration.

The runner is a loop, not a framework.  It owns three things the stage
functions deliberately do not: an advisory lock so two invocations cannot
interleave writes, a *checkable* resume rule so "already built" is verified
rather than inferred from a filename, and the freshness status an agent reads
to decide what to do next.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from quantark.util.exceptions import QuantArkException, ValidationError
from quantark.volcalibration.admission import AdmissionError, AdmissionReason
from quantark.volcalibration.config import RunConfig
from quantark.volcalibration.snapshot import CONVENTION_FX_DELTA, CONVENTION_LISTED
from quantark.volcalibration.snapshot import QuoteSnapshot
from quantark.volcalibration.store import (
    BUILDER_SCHEMA_VERSION,
    PROVENANCE_GRANDFATHERED,
    StoreLayout,
    atomic_write_bytes,
    builder_fingerprint,
    load_surface_manifest,
    save_surface_manifest,
    serialize_artifact,
    surface_record,
)
from quantark.volcalibration.surface import build_artifact

Logger = Callable[[str], None]


def _null_log(message: str) -> None:
    return None


def normalizer_for(config: RunConfig):
    """The one pluggable stage: quote convention -> normalizer."""
    convention = config.underlying.convention
    if convention == CONVENTION_LISTED:
        from quantark.volcalibration.normalize.listed import ListedNormalizer
        from quantark.volcalibration.normalize.settlement import SettlementNormalizer
        from quantark.volcalibration.snapshot import PRICE_FIELD_SETTLEMENT

        if config.underlying.price_field == PRICE_FIELD_SETTLEMENT:
            return SettlementNormalizer()
        return ListedNormalizer()
    if convention == CONVENTION_FX_DELTA:
        from quantark.volcalibration.normalize.fxdelta import FxDeltaNormalizer

        return FxDeltaNormalizer()
    raise ValidationError(f"no normalizer for convention {convention!r}")


def _reason_is_builder_owned(reason: Optional[str]) -> bool:
    """Was this exclusion written by the builder, or by something else?

    ``exclude_thin_surfaces.py`` marks two dates excluded for a study-level
    reason the builder has no vocabulary for, and keeps their artifacts on
    disk.  Rebuilding such a record would silently re-admit a surface a human
    decided to exclude, so the discriminator is the reason itself (plan DP-1).
    """
    if reason is None:
        return True
    try:
        AdmissionReason(reason)
    except ValueError:
        return False
    return True


def surface_record_is_current(
    record: Mapping[str, Any],
    *,
    snapshot_sha: Optional[str],
    price_field: str,
    fingerprint: str,
) -> bool:
    """Is this recorded date still valid for the current inputs and config?"""
    if not record:
        return False
    if record.get("provenance") == PROVENANCE_GRANDFATHERED:
        # Trusted as-is: the artifact bytes are the pinned object and a rebuild
        # would destroy them to prove a property nobody doubts (spec 5.4).
        return True
    if record.get("status") == "excluded" and not _reason_is_builder_owned(
        record.get("reason")
    ):
        return True
    return bool(
        record.get("snapshot_sha256") == snapshot_sha
        and record.get("price_field") == price_field
        and record.get("builder_fingerprint") == fingerprint
        and int(record.get("builder_schema_version", -1)) == BUILDER_SCHEMA_VERSION
    )


def _snapshot_sha_on_disk(layout: StoreLayout, tag: str) -> Optional[str]:
    path = layout.snapshot_path(tag)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    source = payload.get("source")
    sha = source.get("sha256") if isinstance(source, Mapping) else None
    return str(sha) if sha else None


def plan_surface_dates(
    layout: StoreLayout,
    records: Mapping[str, Mapping[str, Any]],
    config: RunConfig,
    *,
    tags: Optional[Sequence[str]] = None,
    force: bool = False,
) -> List[str]:
    """Which dates a surface run would build. Writes nothing."""
    candidates = list(tags) if tags is not None else layout.available_snapshot_dates()
    fingerprint = builder_fingerprint(config.surface.fingerprint_payload())
    pending = []
    for tag in sorted(candidates):
        record = records.get(tag, {})
        if force:
            if record.get("status") == "excluded" and not _reason_is_builder_owned(
                record.get("reason")
            ):
                pending.append(tag)  # --force is the documented way to demand this
                continue
            pending.append(tag)
            continue
        if surface_record_is_current(
            record,
            snapshot_sha=_snapshot_sha_on_disk(layout, tag),
            price_field=config.underlying.price_field,
            fingerprint=fingerprint,
        ):
            continue
        pending.append(tag)
    return pending


def build_one_surface(task) -> Dict[str, Any]:
    """Worker: build and atomically write one date's artifact.

    Module-level and tuple-argued so ``ProcessPoolExecutor`` can pickle it.
    Returns a manifest record; never raises for a per-date failure, because one
    inadmissible date must not abort a backfill.
    """
    tag, snapshot_path, artifact_dir, convention, price_field, sabr_beta, fingerprint = task
    record = surface_record(
        tag,
        status="excluded",
        price_field=price_field,
        fingerprint=fingerprint,
    )
    try:
        payload = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
        snapshot = QuoteSnapshot.from_payload(payload)
        record["snapshot_sha256"] = snapshot.sha256
        normalizer = _normalizer_for_convention(convention, price_field)
        quotes = normalizer.normalize(snapshot)
        artifact = build_artifact(quotes, snapshot, sabr_beta=float(sabr_beta))
        data = serialize_artifact(artifact)
    except AdmissionError as exc:
        _remove_stale_artifact(Path(artifact_dir), tag)
        record["reason"] = exc.reason.value
        record["detail"] = exc.detail
        return record
    except QuantArkException as exc:
        _remove_stale_artifact(Path(artifact_dir), tag)
        record["reason"] = AdmissionReason.UNEXPECTED_ERROR.value
        record["detail"] = f"{type(exc).__name__}: {exc}"
        return record

    import hashlib

    atomic_write_bytes(Path(artifact_dir) / f"mo_iv_surface_{tag}.json", data)
    record.update(
        status="ok",
        reason=None,
        detail=None,
        n_expiries=len(artifact["maturities"]),
        artifact_sha256=hashlib.sha256(data).hexdigest(),
    )
    return record


def _normalizer_for_convention(convention: str, price_field: str):
    from quantark.volcalibration.config import UnderlyingConfig

    return normalizer_for(
        _ConventionOnly(UnderlyingConfig("_", convention, price_field))
    )


class _ConventionOnly:
    """Minimal stand-in so a worker can resolve a normalizer without a RunConfig."""

    def __init__(self, underlying) -> None:
        self.underlying = underlying


def _remove_stale_artifact(artifact_dir: Path, tag: str) -> None:
    """A re-excluded date must not leave a stale artifact for globbing consumers."""
    try:
        (artifact_dir / f"mo_iv_surface_{tag}.json").unlink()
    except FileNotFoundError:
        pass


def run_surface_stage(
    layout: StoreLayout,
    config: RunConfig,
    *,
    tags: Optional[Sequence[str]] = None,
    force: bool = False,
    log: Logger = _null_log,
) -> Dict[str, Dict[str, Any]]:
    """Build every pending date, persist the manifest, return all records."""
    _payload, records = load_surface_manifest(layout)
    records = {tag: dict(record) for tag, record in records.items()}
    pending = plan_surface_dates(layout, records, config, tags=tags, force=force)
    if not pending:
        log("surfaces: nothing to build")
        return records

    fingerprint = builder_fingerprint(config.surface.fingerprint_payload())
    layout.surface_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (
            tag,
            str(layout.snapshot_path(tag)),
            str(layout.surface_dir),
            config.underlying.convention,
            config.underlying.price_field,
            float(config.surface.sabr_beta),
            fingerprint,
        )
        for tag in pending
    ]

    results: List[Dict[str, Any]] = []
    if len(tasks) > 1 and config.workers > 1:
        with ProcessPoolExecutor(max_workers=int(config.workers)) as pool:
            for record in pool.map(build_one_surface, tasks):
                log(f"{record['date']}: {record['status']}"
                    + (f" ({record['reason']})" if record["reason"] else ""))
                results.append(record)
    else:
        for task in tasks:
            record = build_one_surface(task)
            log(f"{record['date']}: {record['status']}"
                + (f" ({record['reason']})" if record["reason"] else ""))
            results.append(record)

    for record in results:
        records[record["date"]] = record

    tags_present = sorted(records)
    save_surface_manifest(
        layout,
        records,
        config=config.surface.fingerprint_payload(),
        window={"start": tags_present[0], "end": tags_present[-1]},
        price_field=config.underlying.price_field,
        source_class=_source_class(layout, pending[0]),
    )
    return records


def _source_class(layout: StoreLayout, tag: str) -> str:
    payload = json.loads(layout.snapshot_path(tag).read_text(encoding="utf-8"))
    return str(payload.get("source_class", "unknown"))
```

- [ ] **Step 4b: Give `QuoteSnapshot` the two accessors the runner needs**

`QuoteSnapshot` (`quantark/volcalibration/snapshot.py:48`) has fields
`schema_version, convention, trade_date, symbol, spot, price_field, source, expiries`.
The source sha lives inside `source`, and there is no serializer. Add both —
the runner reads snapshots off disk and the tests write them there:

```python
    @property
    def sha256(self) -> Optional[str]:
        """The source payload's sha256, as recorded by whoever fetched it."""
        value = self.source.get("sha256")
        return str(value) if value else None

    def to_payload(self) -> Dict[str, Any]:
        """The canonical envelope; round-trips through ``from_payload``."""
        return {
            "schema_version": self.schema_version,
            "convention": self.convention,
            "trade_date": self.trade_date.isoformat(),
            "underlying": {"symbol": self.symbol, "spot": self.spot},
            "source": dict(self.source),
            "expiries": [dict(e) for e in self.expiries],
        }
```

Add a round-trip test to `test/volcalibration/test_snapshot.py`:

```python
def test_the_canonical_envelope_round_trips():
    original = QuoteSnapshot.from_payload(CANONICAL_PAYLOAD)
    assert QuoteSnapshot.from_payload(original.to_payload()) == original
```

reusing whatever canonical payload that module already defines.

- [ ] **Step 5: Run the tests**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/volcalibration/test_runner_surface.py -q
```
Expected: PASS (10 tests, counting the 4 parametrized invalidation cases).

- [ ] **Step 6: Commit**

```bash
git add quantark/volcalibration/runner.py quantark/volcalibration/admission.py quantark/volcalibration/snapshot.py test/volcalibration/test_runner_surface.py
git commit -m "$(cat <<'EOF'
feat(volcalibration): surface stage with a checkable resume rule

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YSMxE68spnG5uF97YdTCGU
EOF
)"
```

---

### Task 16: Runner — the calibration stage

**Files:**
- Modify: `quantark/volcalibration/runner.py`
- Test: `test/volcalibration/test_runner_calibration.py`

**Interfaces:**
- Consumes: `load_calibration_manifest`, `save_calibration_manifest` (Task 14); `VolModelCalibrator`, `IvSurfaceArtifact`
- Produces:
  - `heston_feller_diagnostics(params) -> dict`
  - `structural_ewma_before(records, *, before_date, span) -> Optional[dict]`
  - `update_structural_ewma(prior, raw, *, trade_date, span) -> dict`
  - `combine_daily_v0_and_structure(raw, state) -> dict`
  - `calibration_record_is_current(record, surface_record, config_payload, variants) -> bool`
  - `select_calibration_dates(surface_records, calibration_records, *, config, backfill, max_dates, baseline_date, start_date, end_date) -> List[str]`
  - `calibrate_one_surface(tag, *, layout, config, calibration_records) -> dict`
  - `run_calibration_stage(layout, config, *, surface_records, backfill=False, max_dates=None, start_date=None, end_date=None, log=...) -> Dict[str, dict]`

- [ ] **Step 1: Write the failing test**

Create `test/volcalibration/test_runner_calibration.py`. The Heston/SLV calibrators are slow, so the record-shaping tests use a stub calibrator injected through a module attribute; one end-to-end test does a real `localvol` fit only.

```python
import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.config import CalibrationRunConfig
from quantark.volcalibration.runner import (
    calibration_record_is_current,
    combine_daily_v0_and_structure,
    heston_feller_diagnostics,
    select_calibration_dates,
    structural_ewma_before,
    update_structural_ewma,
)

CONFIG = CalibrationRunConfig().manifest_payload()
VARIANTS = ("localvol", "heston", "heston_slv")


def _cal_record(date, sha, status="ok", config=None):
    return {
        "date": date,
        "status": status,
        "surface_sha": sha,
        "config": config or CONFIG,
        "variants": {v: {"status": "ok", "record": {}} for v in VARIANTS},
    }


def test_a_record_is_current_only_when_sha_config_and_every_variant_agree():
    surface = {"artifact_sha256": "abc"}
    assert calibration_record_is_current(_cal_record("20260430", "abc"), surface, CONFIG, VARIANTS)
    assert not calibration_record_is_current(_cal_record("20260430", "xyz"), surface, CONFIG, VARIANTS)
    stale_config = _cal_record("20260430", "abc", config={**CONFIG, "slv_n_x": 81})
    assert not calibration_record_is_current(stale_config, surface, CONFIG, VARIANTS)
    partial = _cal_record("20260430", "abc")
    partial["variants"]["heston_slv"] = {"status": "failed"}
    assert not calibration_record_is_current(partial, surface, CONFIG, VARIANTS)


def test_incremental_selection_does_not_bootstrap_a_multi_year_backfill():
    surfaces = {d: {"status": "ok", "artifact_sha256": d} for d in
                ("20260101", "20260102", "20260103")}
    selected = select_calibration_dates(
        surfaces, {}, config=CONFIG, backfill=False, max_dates=None,
        baseline_date=None, start_date=None, end_date=None, variants=VARIANTS,
    )
    assert selected == ["20260103"], "an empty manifest calibrates only the latest date"


def test_backfill_selects_every_stale_admitted_date():
    surfaces = {d: {"status": "ok", "artifact_sha256": d} for d in
                ("20260101", "20260102", "20260103")}
    selected = select_calibration_dates(
        surfaces, {}, config=CONFIG, backfill=True, max_dates=None,
        baseline_date=None, start_date=None, end_date=None, variants=VARIANTS,
    )
    assert selected == ["20260101", "20260102", "20260103"]


def test_excluded_surfaces_are_never_calibrated():
    surfaces = {"20260101": {"status": "excluded", "reason": "static_arbitrage"}}
    assert select_calibration_dates(
        surfaces, {}, config=CONFIG, backfill=True, max_dates=None,
        baseline_date=None, start_date=None, end_date=None, variants=VARIANTS,
    ) == []


def test_feller_diagnostics_report_the_ratio_and_the_verdict():
    ok = heston_feller_diagnostics({"kappa": 2.0, "theta": 0.04, "sigma": 0.2})
    assert ok["feller_ratio"] == pytest.approx(2 * 2.0 * 0.04 / 0.04)
    assert ok["feller_satisfied"]
    assert not heston_feller_diagnostics(
        {"kappa": 0.1, "theta": 0.01, "sigma": 0.9}
    )["feller_satisfied"]


def test_the_ewma_is_recursive_and_seeds_from_the_first_observation():
    raw = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "sigma": 0.2, "rho": -0.5}
    first = update_structural_ewma(None, raw, trade_date="20260101", span=19)
    assert first["parameters"]["kappa"] == pytest.approx(2.0)
    assert first["observation_count"] == 1
    second = update_structural_ewma(
        first, {**raw, "kappa": 3.0}, trade_date="20260102", span=19
    )
    alpha = 2.0 / 20.0
    assert second["parameters"]["kappa"] == pytest.approx(alpha * 3.0 + (1 - alpha) * 2.0)
    assert second["first_source_date"] == "20260101"
    assert second["last_source_date"] == "20260102"


def test_a_feller_violating_history_seed_fails_closed():
    records = {
        "20260101": {
            "status": "ok",
            "variants": {"heston": {"status": "ok", "record": {
                "v0": 0.04, "kappa": 0.1, "theta": 0.01, "sigma": 0.9, "rho": -0.5}}},
        }
    }
    with pytest.raises(ValidationError):
        structural_ewma_before(records, before_date="20260102", span=19)


def test_the_slv_vector_carries_today_s_v0_over_smoothed_structure():
    raw = {"v0": 0.09, "kappa": 3.0, "theta": 0.05, "sigma": 0.3, "rho": -0.6}
    state = {"parameters": {"kappa": 2.0, "theta": 0.04, "sigma": 0.2, "rho": -0.5}}
    assert combine_daily_v0_and_structure(raw, state) == {
        "v0": 0.09, "kappa": 2.0, "theta": 0.04, "sigma": 0.2, "rho": -0.5
    }
```

Add one slow end-to-end test guarded by a marker:

```python
@pytest.mark.slow
def test_localvol_calibration_writes_a_manifest_record(tmp_path):
    """One real fit through the real calibrator, localvol only (seconds, not minutes)."""
    import json
    from datetime import date

    from quantark.volcalibration.config import (
        RunConfig, SurfaceBuildConfig, UnderlyingConfig,
    )
    from quantark.volcalibration.runner import run_calibration_stage, run_surface_stage
    from quantark.volcalibration.snapshot import QuoteSnapshot
    from quantark.volcalibration.store import StoreLayout

    payload = json.loads(
        open("example/mo_volmodels/data/mo_settlement_snapshot_20260430.json",
             encoding="utf-8").read()
    )
    config = RunConfig(
        name="t",
        underlying=UnderlyingConfig("000852.SH", "listed_strike", "settlement"),
        history_dir=tmp_path, runtime_dir=tmp_path, spot_csv=None,
        surface=SurfaceBuildConfig(),
        calibration=CalibrationRunConfig(variants=("localvol",)),
        workers=1,
    )
    layout = StoreLayout.from_config(config)
    snap = QuoteSnapshot.from_legacy_settlement(
        payload, trade_date=date(2026, 4, 30), spot=8381.947, symbol="000852.SH",
        source_sha256=payload["source_sha256"], source_url=payload.get("source_url"),
    )
    layout.snapshots_dir.mkdir(parents=True, exist_ok=True)
    layout.snapshot_path("20260430").write_text(json.dumps(snap.to_payload()))

    surfaces = run_surface_stage(layout, config)
    records = run_calibration_stage(layout, config, surface_records=surfaces)
    assert records["20260430"]["status"] == "ok"
    assert records["20260430"]["variants"]["localvol"]["status"] == "ok"
    assert records["20260430"]["surface_sha"] == surfaces["20260430"]["artifact_sha256"]
```

- [ ] **Step 2: Run it to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/volcalibration/test_runner_calibration.py -x -q -m "not slow"
```
Expected: FAIL — `ImportError: cannot import name 'calibration_record_is_current'`.

- [ ] **Step 3: Append the calibration half to `runner.py`**

Port `heston_params_payload`, `raw_heston_from_calibration_record`, `structural_ewma_before`, `update_structural_ewma`, `combine_daily_v0_and_structure`, `heston_vector`, `heston_feller_diagnostics`, `calibration_record_is_current`, `select_calibration_dates` and `calibrate_one_surface` from `example/mo_volmodels/14_daily_calibration_pipeline.py:389-812`, with exactly these changes:

1. `PipelineError` → `ValidationError` (`quantark.util.exceptions`).
2. `args: argparse.Namespace` → `config: RunConfig`; every `args.slv_eta` etc. reads `config.calibration.<field>`; `calibration_config_payload(args)` → `config.calibration.manifest_payload()`.
3. `VARIANTS` (module constant) → `config.calibration.variants`, threaded into `calibration_record_is_current` and `select_calibration_dates` as an explicit `variants` keyword — a run that calibrates only `localvol` must not be judged stale for lacking a `heston` record.
4. `paths` → `layout`; `surface_artifact_path(paths, tag)` → `layout.artifact_path(tag)`; `paths.calibration_cache` → `layout.calibration_cache`.
5. `from quantark.backtest.replay.config import VolModelCalibrationConfig` → `config.calibration.calibrator_config(layout.calibration_cache)`. This is the import that made the old pipeline depend on backtest; it is the inversion this module exists to reverse.
6. `except Exception` in the per-variant loop stays as-is: one variant failing must still record the other two, and the record's `status` becomes `failed`, which the CLI maps to exit 1.

Then add the loop:

```python
def run_calibration_stage(
    layout: StoreLayout,
    config: RunConfig,
    *,
    surface_records: Mapping[str, Mapping[str, Any]],
    backfill: bool = False,
    max_dates: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    log: Logger = _null_log,
) -> Dict[str, Dict[str, Any]]:
    """Calibrate every selected admitted surface, persisting after each date."""
    base_payload, records = load_calibration_manifest(layout)
    records = {tag: dict(record) for tag, record in records.items()}
    payload = config.calibration.manifest_payload()
    selected = select_calibration_dates(
        surface_records,
        records,
        config=payload,
        backfill=backfill,
        max_dates=max_dates,
        baseline_date=base_payload.get("baseline_date")
        or (min(records) if records else None),
        start_date=start_date,
        end_date=end_date,
        variants=tuple(config.calibration.variants),
    )
    if selected and not records and not backfill and base_payload.get("baseline_date") is None:
        base_payload["baseline_date"] = selected[0]

    for tag in selected:
        log(f"{tag}: calibrating {', '.join(config.calibration.variants)}")
        record = calibrate_one_surface(
            tag, layout=layout, config=config, calibration_records=records
        )
        records[tag] = record
        # Persist after every date: an interrupted backfill resumes from here.
        save_calibration_manifest(layout, base_payload, records, config=payload)
        log(f"{tag}: {record['status']} [{record['elapsed_seconds']:.2f}s]")
    return records
```

- [ ] **Step 4: Run the fast tests**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/volcalibration/test_runner_calibration.py -q -m "not slow"
```
Expected: PASS (8 tests).

- [ ] **Step 5: Run the slow one once**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/volcalibration/test_runner_calibration.py -q -m slow
```
Expected: PASS (1 test).

- [ ] **Step 6: Commit**

```bash
git add quantark/volcalibration/runner.py test/volcalibration/test_runner_calibration.py
git commit -m "$(cat <<'EOF'
feat(volcalibration): calibration stage, variant-aware and backtest-free

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YSMxE68spnG5uF97YdTCGU
EOF
)"
```

---

### Task 17: Runner — lock, freshness status and exit codes

**Files:**
- Modify: `quantark/volcalibration/runner.py`
- Test: `test/volcalibration/test_runner_status.py`

**Interfaces:**
- Produces:
  - `LockBusy(QuantArkException)`
  - `acquire_lock(path)` — context manager
  - `EXIT_CURRENT = 0`, `EXIT_FAILED = 1`, `EXIT_NON_CURRENT = 2`, `EXIT_LOCKED = 75`
  - `load_trading_dates(spot_csv, *, as_of=None) -> List[str]`
  - `build_status(layout, config, *, as_of, run_id=None, last_error=None) -> dict`
  - `status_exit_code(status) -> int`
  - `run_pipeline(config, *, as_of, backfill=False, ..., force=False, log=...) -> Tuple[int, dict]`

- [ ] **Step 1: Write the failing test**

Create `test/volcalibration/test_runner_status.py`:

```python
import os
import subprocess
import sys
import textwrap

import pytest

from quantark.volcalibration.runner import (
    EXIT_CURRENT,
    EXIT_FAILED,
    EXIT_LOCKED,
    EXIT_NON_CURRENT,
    LockBusy,
    acquire_lock,
    build_status,
    status_exit_code,
)


def test_the_lock_is_exclusive_within_one_process(tmp_path):
    path = tmp_path / "pipeline.lock"
    with acquire_lock(path):
        with pytest.raises(LockBusy):
            with acquire_lock(path):
                pass


def test_the_lock_is_exclusive_across_processes(tmp_path):
    """flock is per-file-description; a same-process re-acquire is not proof."""
    path = tmp_path / "pipeline.lock"
    script = textwrap.dedent(f"""
        import sys
        from quantark.volcalibration.runner import LockBusy, acquire_lock
        try:
            with acquire_lock({str(path)!r}):
                sys.exit(0)
        except LockBusy:
            sys.exit(75)
    """)
    with acquire_lock(path):
        env = dict(os.environ, PYTHONPATH=os.getcwd())
        result = subprocess.run([sys.executable, "-c", script], env=env)
    assert result.returncode == EXIT_LOCKED


@pytest.mark.parametrize(
    "overall,expected",
    [
        ("current", EXIT_CURRENT),
        ("snapshot_pending", EXIT_NON_CURRENT),
        ("surface_excluded", EXIT_NON_CURRENT),
        ("surface_pending", EXIT_NON_CURRENT),
        ("calibration_pending", EXIT_NON_CURRENT),
        ("calibration_failed", EXIT_FAILED),
        ("failed", EXIT_FAILED),
    ],
)
def test_exit_codes_carry_state(overall, expected):
    assert status_exit_code({"overall_status": overall}) == expected


def test_status_without_a_calendar_says_so(tmp_path, monkeypatch):
    from quantark.volcalibration.config import (
        CalibrationRunConfig, RunConfig, SurfaceBuildConfig, UnderlyingConfig,
    )
    from quantark.volcalibration.store import StoreLayout

    config = RunConfig(
        name="t",
        underlying=UnderlyingConfig("S", "listed_strike", "settlement"),
        history_dir=tmp_path, runtime_dir=tmp_path, spot_csv=None,
        surface=SurfaceBuildConfig(), calibration=CalibrationRunConfig(), workers=1,
    )
    layout = StoreLayout.from_config(config)
    layout.snapshots_dir.mkdir(parents=True)
    (layout.snapshots_dir / "20260430.json").write_text("{}")
    from datetime import date

    status = build_status(layout, config, as_of=date(2026, 5, 1))
    assert status["expected_trade_date"] == "20260430"
    assert status["freshness"]["spot_cache_latest"] is None
    assert status["freshness"]["calendar_source"] == "snapshots_on_disk"
    assert status["overall_status"] == "surface_pending"
    assert status_exit_code(status) == EXIT_NON_CURRENT


def test_an_excluded_expected_date_is_non_current_not_failed(tmp_path):
    """A date can be legitimately uncalibratable forever (spec 6.3.2)."""
    from datetime import date

    from quantark.volcalibration.config import (
        CalibrationRunConfig, RunConfig, SurfaceBuildConfig, UnderlyingConfig,
    )
    from quantark.volcalibration.store import (
        StoreLayout, atomic_write_json, surface_record,
    )

    config = RunConfig(
        name="t",
        underlying=UnderlyingConfig("S", "listed_strike", "settlement"),
        history_dir=tmp_path, runtime_dir=tmp_path, spot_csv=None,
        surface=SurfaceBuildConfig(), calibration=CalibrationRunConfig(), workers=1,
    )
    layout = StoreLayout.from_config(config)
    layout.snapshots_dir.mkdir(parents=True)
    (layout.snapshots_dir / "20240930.json").write_text("{}")
    atomic_write_json(
        layout.surface_manifest,
        {
            "schema_version": 1,
            "records": [surface_record("20240930", status="excluded",
                                       reason="insufficient_expiries", n_expiries=1)],
        },
    )
    status = build_status(layout, config, as_of=date(2024, 10, 1))
    assert status["overall_status"] == "surface_excluded"
    assert status_exit_code(status) == EXIT_NON_CURRENT
```

- [ ] **Step 2: Run it to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/volcalibration/test_runner_status.py -x -q
```
Expected: FAIL — `ImportError: cannot import name 'acquire_lock'`.

- [ ] **Step 3: Append lock, status and the top-level pipeline to `runner.py`**

Port `acquire_lock` from `14_daily_calibration_pipeline.py:224-256` verbatim except: `LockBusy` derives from `QuantArkException`, and the recorded owner payload drops `PROJECT_ROOT` for `os.getcwd()`.

Port `load_trading_dates` (`:300-322`) and `trading_day_lag` (`:830-839`) unchanged except `PipelineError` → `ValidationError`.

`build_status` is `build_freshness_status` (`:841-938`) with these changes, and no others:

- The source stage is the snapshot, not the settlement CSV: `settlement_latest`/`settlement_lag_trading_days`/`expected_date_records.settlement` become `snapshot_latest`/`snapshot_lag_trading_days`/`expected_date_records.snapshot`, computed from `layout.available_snapshot_dates()`.
- `overall_status`'s `source_pending` becomes `snapshot_pending`.
- The calendar is optional (plan DP-3):

```python
    if config.spot_csv is not None:
        trading_dates = load_trading_dates(config.spot_csv, as_of=as_of)
        expected = trading_dates[-1]
        calendar_source = str(config.spot_csv)
        spot_cache_latest = trading_dates[-1]
        cache_age_days = (
            as_of - datetime.strptime(trading_dates[-1], "%Y%m%d").date()
        ).days
    else:
        # No calendar: "expected" is the newest snapshot we hold.  The status
        # says which of the two it is rather than implying a freshness check
        # that did not run (plan DP-3).
        snapshots = layout.available_snapshot_dates()
        if not snapshots:
            raise ValidationError(
                f"{layout.snapshots_dir} holds no snapshots and no "
                "paths.spot_csv is configured, so there is no trade date to "
                "report on"
            )
        trading_dates = snapshots
        expected = snapshots[-1]
        calendar_source = "snapshots_on_disk"
        spot_cache_latest = None
        cache_age_days = None
```

with `"calendar_source": calendar_source` and `"spot_cache_latest": spot_cache_latest` in the `freshness` block, and the `market_cache_stale` branch guarded by `cache_age_days is not None`.

- `"pipeline"` becomes `config.name` and a new `"module": "quantark.volcalibration"` key is added, so a library status file is distinguishable from stage 14's.
- `"stages"` (the subprocess `StageResult` list) is dropped: the runner calls functions, so there are no subprocess stages to report. This is the deletion §7.1 calls for.
- The status payload gains `"grandfathered_surface_dates": <count>` — spec §5.4 requires the count to be visible.

`status_exit_code` is `:940-947` unchanged.

Then the transaction:

```python
def run_pipeline(
    config: RunConfig,
    *,
    as_of,
    tags: Optional[Sequence[str]] = None,
    backfill: bool = False,
    max_dates: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    force: bool = False,
    skip_surfaces: bool = False,
    skip_calibration: bool = False,
    log: Logger = _null_log,
):
    """One locked, atomic, resumable run. Returns (exit_code, status)."""
    layout = StoreLayout.from_config(config)
    layout.runtime_dir.mkdir(parents=True, exist_ok=True)
    run_id = _iso_utc()
    with acquire_lock(layout.lock):
        try:
            if skip_surfaces:
                _payload, surface_records = load_surface_manifest(layout)
            else:
                surface_records = run_surface_stage(
                    layout, config, tags=tags, force=force, log=log
                )
            if not skip_calibration:
                run_calibration_stage(
                    layout,
                    config,
                    surface_records=surface_records,
                    backfill=backfill,
                    max_dates=max_dates,
                    start_date=start_date,
                    end_date=end_date,
                    log=log,
                )
            status = build_status(layout, config, as_of=as_of, run_id=run_id)
        except QuantArkException as exc:
            status = build_status(
                layout,
                config,
                as_of=as_of,
                run_id=run_id,
                last_error={"error_type": type(exc).__name__, "message": str(exc)},
            )
        atomic_write_json(layout.status, status)
        return status_exit_code(status), status
```

`LockBusy` propagates out of `run_pipeline` uncaught: the CLI turns it into exit 75, and writing a status file while another process owns the transaction would corrupt its view.

- [ ] **Step 4: Run the tests**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/volcalibration/test_runner_status.py -q
```
Expected: PASS (11 tests).

- [ ] **Step 5: Commit**

```bash
git add quantark/volcalibration/runner.py test/volcalibration/test_runner_status.py
git commit -m "$(cat <<'EOF'
feat(volcalibration): advisory lock, freshness status and exit codes

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YSMxE68spnG5uF97YdTCGU
EOF
)"
```

---

### Task 18: CLI — four verbs and the agent contract

**Files:**
- Create: `quantark/volcalibration/cli.py`, `quantark/volcalibration/__main__.py`
- Test: `test/volcalibration/test_cli.py`

**Interfaces:**
- Produces: `main(argv=None) -> int`; verbs `run`, `status`, `show`, `list`

- [ ] **Step 1: Write the failing test**

Create `test/volcalibration/test_cli.py`. Every test drives `main(argv)` in-process and captures stdout/stderr with `capsys`; the JSON tests parse stdout whole, which is what proves the stdout/stderr split.

```python
import json
from datetime import date

import pytest

from quantark.volcalibration.cli import main
from quantark.volcalibration.snapshot import QuoteSnapshot
from quantark.volcalibration.store import StoreLayout, atomic_write_json, surface_record

SAMPLE = "example/mo_volmodels/data/mo_settlement_snapshot_20260430.json"

CONFIG_YAML = """
schema_version: 1
name: test-mo
underlying:
  symbol: "000852.SH"
  convention: listed_strike
  price_field: settlement
paths:
  root: history
calibration:
  variants: [localvol]
"""


@pytest.fixture
def workspace(tmp_path):
    """A config file plus a store holding one snapshot."""
    config_path = tmp_path / "run.yaml"
    config_path.write_text(CONFIG_YAML, encoding="utf-8")
    payload = json.loads(open(SAMPLE, encoding="utf-8").read())
    snap = QuoteSnapshot.from_legacy_settlement(
        payload, trade_date=date(2026, 4, 30), spot=8381.947, symbol="000852.SH",
        source_sha256=payload["source_sha256"], source_url=payload.get("source_url"),
    )
    snapshots = tmp_path / "history" / "snapshots"
    snapshots.mkdir(parents=True)
    (snapshots / "20260430.json").write_text(json.dumps(snap.to_payload()))
    return config_path, tmp_path / "history"


def test_json_stdout_is_exactly_one_object(workspace, capsys):
    config_path, _ = workspace
    code = main(["status", str(config_path), "--as-of", "2026-04-30", "--json"])
    out, err = capsys.readouterr()
    payload = json.loads(out)  # parses whole: no log line may leak into stdout
    assert payload["module"] == "quantark.volcalibration"
    assert code == 2  # nothing built yet
    assert isinstance(err, str)


def test_plan_writes_nothing(workspace, capsys):
    config_path, history = workspace
    before = {p: p.stat().st_mtime_ns for p in history.rglob("*") if p.is_file()}
    code = main(["run", str(config_path), "--as-of", "2026-04-30", "--plan", "--json"])
    out, _ = capsys.readouterr()
    plan = json.loads(out)
    assert plan["surfaces_to_build"] == ["20260430"]
    assert code == 2
    after = {p: p.stat().st_mtime_ns for p in history.rglob("*") if p.is_file()}
    assert after == before
    assert not (history / "iv_surface").exists()


def test_run_is_idempotent_and_reports_current(workspace, capsys):
    config_path, history = workspace
    first = main(["run", str(config_path), "--as-of", "2026-04-30"])
    capsys.readouterr()
    artifact = history / "iv_surface" / "mo_iv_surface_20260430.json"
    bytes_before = artifact.read_bytes()
    second = main(["run", str(config_path), "--as-of", "2026-04-30", "--json"])
    out, _ = capsys.readouterr()
    status = json.loads(out)
    assert first == 0 and second == 0
    assert status["overall_status"] == "current"
    assert artifact.read_bytes() == bytes_before


def test_an_excluded_expected_date_exits_two_not_one(tmp_path, capsys):
    config_path = tmp_path / "run.yaml"
    config_path.write_text(CONFIG_YAML, encoding="utf-8")
    layout = StoreLayout(tmp_path / "history", tmp_path / "history")
    layout.snapshots_dir.mkdir(parents=True)
    (layout.snapshots_dir / "20240930.json").write_text("{}")
    atomic_write_json(
        layout.surface_manifest,
        {"schema_version": 1,
         "records": [surface_record("20240930", status="excluded",
                                    reason="insufficient_expiries", n_expiries=1)]},
    )
    code = main(["status", str(config_path), "--as-of", "2024-09-30", "--json"])
    out, _ = capsys.readouterr()
    assert json.loads(out)["overall_status"] == "surface_excluded"
    assert code == 2, "an excluded date is 'not yet', not a failure to retry-loop on"


def test_a_locked_store_exits_75(workspace, capsys):
    config_path, history = workspace
    from quantark.volcalibration.runner import acquire_lock

    with acquire_lock(history / "pipeline.lock"):
        code = main(["run", str(config_path), "--as-of", "2026-04-30", "--json"])
    out, err = capsys.readouterr()
    assert code == 75
    assert json.loads(out)["reason"] == "locked"


def test_an_unreadable_config_exits_one(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 9\n", encoding="utf-8")
    code = main(["status", str(bad), "--json"])
    out, err = capsys.readouterr()
    assert code == 1
    assert json.loads(out)["reason"] == "invalid_config"
    assert "schema_version" in err


def test_show_reports_one_date_s_verdict_and_diagnostics(workspace, capsys):
    config_path, _ = workspace
    main(["run", str(config_path), "--as-of", "2026-04-30"])
    capsys.readouterr()
    code = main(["show", str(config_path), "--date", "2026-04-30", "--json"])
    out, _ = capsys.readouterr()
    payload = json.loads(out)
    assert code == 0
    assert payload["surface"]["status"] == "ok"
    assert payload["surface"]["admission"]["validation_method"]
    assert len(payload["surface"]["per_expiry"]) >= 2
    assert payload["surface"]["per_expiry"][0]["parity_rmse_over_forward"] is not None
    assert payload["calibration"]["variants"]["localvol"]["status"] == "ok"


def test_list_without_a_config_reports_discoverable_configs(capsys):
    code = main(["list", "--json"])
    out, _ = capsys.readouterr()
    assert code == 0
    assert isinstance(json.loads(out)["configs"], list)


def test_list_with_a_config_reports_per_date_rows(workspace, capsys):
    config_path, _ = workspace
    main(["run", str(config_path), "--as-of", "2026-04-30"])
    capsys.readouterr()
    code = main(["list", "--config", str(config_path), "--json"])
    out, _ = capsys.readouterr()
    rows = json.loads(out)["dates"]
    assert code == 0
    assert rows[0]["date"] == "20260430"
    assert rows[0]["surface_status"] == "ok"
    assert rows[0]["calibrated_variants"] == ["localvol"]
```

- [ ] **Step 2: Run it to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/volcalibration/test_cli.py -x -q
```
Expected: FAIL — `ModuleNotFoundError: No module named 'quantark.volcalibration.cli'`.

- [ ] **Step 3: Write `cli.py`**

```python
"""One command line for daily calibration.

    python -m quantark.volcalibration run    <config.yaml> [--as-of DATE]
                                             [--backfill] [--from D] [--to D]
                                             [--variants ...] [--workers N]
                                             [--force] [--plan] [--json]
    python -m quantark.volcalibration status <config.yaml> [--json]
    python -m quantark.volcalibration show   <config.yaml> --date D [--json]
    python -m quantark.volcalibration list   [--config <config.yaml>]

The agent contract (spec 6.3):

1.  stdout is data, stderr is narrative.  Under ``--json`` stdout carries
    exactly one JSON object and every progress line goes to stderr.
2.  Exit codes carry state: 0 current, 2 non-current but fail-closed, 1
    pipeline failure, 75 another process holds the lock.  A 2 means the system
    is working and the answer is "not yet" -- do not retry-loop on it.  Two MO
    dates are excluded permanently and will never become admissible.
3.  Re-running a successful config is a no-op reporting ``current``.
4.  Failures are machine-readable: ``reason`` is a stable code, never prose.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from quantark.util.exceptions import QuantArkException
from quantark.volcalibration.config import RunConfig
from quantark.volcalibration.runner import (
    EXIT_CURRENT,
    EXIT_FAILED,
    EXIT_LOCKED,
    EXIT_NON_CURRENT,
    LockBusy,
    build_status,
    load_calibration_manifest,
    load_surface_manifest,
    plan_surface_dates,
    run_pipeline,
    select_calibration_dates,
    status_exit_code,
)
from quantark.volcalibration.store import StoreLayout
from quantark.volcalibration.yaml_loader import load_run_config

CONFIG_SEARCH_DIRS = (Path("example/mo_volmodels"), Path("example/fx_volmodels"))


def _emit(payload: Dict[str, Any], *, as_json: bool, human) -> None:
    if as_json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True, default=str)
        sys.stdout.write("\n")
    else:
        human(payload)


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _parse_date_tag(value: str) -> str:
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y%m%d")
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"invalid date {value!r}; expected YYYY-MM-DD or YYYYMMDD"
    )
```

Then `_build_parser()` with the four subparsers and exactly the flags in the docstring, and the four commands:

```python
def _cmd_run(args, config: RunConfig) -> int:
    if args.variants:
        config = replace(
            config,
            calibration=replace(config.calibration, variants=tuple(args.variants)),
        )
    if args.workers is not None:
        config = replace(config, workers=int(args.workers))

    if args.plan:
        layout = StoreLayout.from_config(config)
        _payload, surface_records = load_surface_manifest(layout)
        _base, calibration_records = load_calibration_manifest(layout)
        surfaces = plan_surface_dates(
            layout, surface_records, config, force=args.force
        )
        # Dates the plan would calibrate assuming every planned surface admits.
        projected = dict(surface_records)
        for tag in surfaces:
            projected.setdefault(tag, {"status": "ok", "artifact_sha256": None})
        calibrations = select_calibration_dates(
            projected,
            calibration_records,
            config=config.calibration.manifest_payload(),
            backfill=bool(args.backfill),
            max_dates=args.max_dates,
            baseline_date=_base.get("baseline_date"),
            start_date=args.date_from,
            end_date=args.date_to,
            variants=tuple(config.calibration.variants),
        )
        payload = {
            "module": "quantark.volcalibration",
            "name": config.name,
            "plan": True,
            "config": config.echo(),
            "surfaces_to_build": surfaces,
            "calibrations_to_run": calibrations,
        }
        _emit(payload, as_json=args.json, human=_print_plan)
        return EXIT_CURRENT if not surfaces and not calibrations else EXIT_NON_CURRENT

    try:
        code, status = run_pipeline(
            config,
            as_of=args.as_of,
            backfill=bool(args.backfill),
            max_dates=args.max_dates,
            start_date=args.date_from,
            end_date=args.date_to,
            force=bool(args.force),
            log=_log,
        )
    except LockBusy as exc:
        _emit(
            {"module": "quantark.volcalibration", "reason": "locked", "detail": str(exc)},
            as_json=args.json,
            human=lambda p: print(f"locked: {p['detail']}"),
        )
        return EXIT_LOCKED
    _emit(status, as_json=args.json, human=_print_status)
    return code
```

`_cmd_status` calls `build_status` and returns `status_exit_code`. `_cmd_show` reads the artifact and the calibration record for one date and emits `{"date", "surface": {...}, "calibration": {...}}` — the surface block carries `status`, `reason`, `admission`, and a per-expiry list of `{T, expiry_date, forward, r, q, n_nodes, parity_rmse_over_forward, sabr_rmse_vol_points}`; the calibration block carries per-variant status, the Heston vector with `feller_ratio`, and the SLV leverage range. `_cmd_list` emits `{"configs": [...]}` or, with `--config`, `{"dates": [{date, surface_status, reason, calibrated_variants}]}`.

`main` wraps everything:

```python
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "list" and args.config is None:
        return _cmd_list(args, None)
    try:
        config = load_run_config(args.config if args.command == "list" else args.study)
    except QuantArkException as exc:
        print(f"error: {exc}", file=sys.stderr)
        _emit(
            {"module": "quantark.volcalibration", "reason": "invalid_config",
             "detail": str(exc)},
            as_json=getattr(args, "json", False),
            human=lambda p: None,
        )
        return EXIT_FAILED
    try:
        return _COMMANDS[args.command](args, config)
    except LockBusy as exc:
        print(f"locked: {exc}", file=sys.stderr)
        return EXIT_LOCKED
    except QuantArkException as exc:
        print(f"error: {exc}", file=sys.stderr)
        _emit(
            {"module": "quantark.volcalibration", "reason": "pipeline_failed",
             "detail": str(exc)},
            as_json=getattr(args, "json", False),
            human=lambda p: None,
        )
        return EXIT_FAILED
```

Note `--json` must emit its object even on the error paths: an agent parsing stdout gets a `reason` code on every exit, never an empty stdout it has to interpret.

- [ ] **Step 4: Write `__main__.py`**

```python
"""``python -m quantark.volcalibration``."""

import sys

from quantark.volcalibration.cli import main

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/volcalibration/test_cli.py -q
```
Expected: PASS (9 tests).

- [ ] **Step 6: Drive it once by hand, exactly as an agent would**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m quantark.volcalibration list --json
```
Expected: exit 0 and one JSON object on stdout.

- [ ] **Step 7: Commit**

```bash
git add quantark/volcalibration/cli.py quantark/volcalibration/__main__.py test/volcalibration/test_cli.py
git commit -m "$(cat <<'EOF'
feat(volcalibration): four CLI verbs with a machine-readable agent contract

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YSMxE68spnG5uF97YdTCGU
EOF
)"
```

---

### Task 19: `CalibrationSet` and `environment_for`

**Files:**
- Create: `quantark/volcalibration/calibration_set.py`
- Modify: `quantark/volcalibration/__init__.py`
- Test: `test/volcalibration/test_calibration_set.py`

**Interfaces:**
- Consumes: `VolSurfaceHistory`, `IvSurfaceArtifact` (`quantark.param.vol.surface_history`); `VolModelCalibrator`, `CalibratedVolModel`; `StoreLayout`, `load_calibration_manifest`
- Produces:
  - `CalibrationSet.open(root, *, runtime=None, config=None) -> CalibrationSet`
  - `.dates() -> List[date]`, `.surface_for(d) -> IvSurfaceArtifact`, `.model_for(d, variant) -> CalibratedVolModel`, `.environment_for(d) -> Tuple[PricingEnvironment, GridVolSurface, float]`, `.status() -> dict`
  - `.surface_history -> VolSurfaceHistory`

- [ ] **Step 1: Write the failing test**

Create `test/volcalibration/test_calibration_set.py`:

```python
import json
from datetime import date

import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.calibration_set import CalibrationSet

# test/replay_golden/fixtures.py already writes a minimal admitted history.
from test.replay_golden.fixtures import DATE_A, write_localvol_history


@pytest.fixture
def history(tmp_path):
    write_localvol_history(tmp_path)
    return tmp_path


def test_open_validates_and_lists_admitted_dates(history):
    cs = CalibrationSet.open(history)
    assert DATE_A in cs.dates()


def test_surface_for_applies_manifest_carry_forward(history):
    cs = CalibrationSet.open(history)
    artifact = cs.surface_for(DATE_A)
    assert artifact.sha256
    # A date after the last admitted one carries the last surface forward.
    later = date(DATE_A.year + 1, DATE_A.month, DATE_A.day)
    assert cs.surface_for(later).sha256 == artifact.sha256


def test_open_is_fail_closed_on_a_missing_manifest(tmp_path):
    with pytest.raises(ValidationError):
        CalibrationSet.open(tmp_path / "nowhere")


def test_environment_for_carries_the_parity_term_structure(history):
    cs = CalibrationSet.open(history)
    env, surface, s0 = cs.environment_for(DATE_A)
    artifact = cs.surface_for(DATE_A)
    assert s0 == pytest.approx(artifact.payload["s0"])
    assert env.valuation_date.date() == DATE_A, (
        "the environment must be dated to the trade date, not a hardcoded day"
    )
    for pillar in artifact.payload["per_expiry"]:
        assert env.rate_curve.get_rate(pillar["T"]) == pytest.approx(pillar["r"])
        assert env.div_yield.get_yield(pillar["T"]) == pytest.approx(pillar["q"])
    assert list(surface.strikes) == list(artifact.payload["strikes"])


def test_model_for_reports_a_missing_calibration_by_name(history):
    cs = CalibrationSet.open(history)
    with pytest.raises(ValidationError) as exc:
        cs.model_for(DATE_A, "heston")
    assert "heston" in str(exc.value)
    assert str(DATE_A) in str(exc.value)
```

**Note on `build_env`'s hardcoded date.** `_mo_common.build_env` passes `valuation_date=datetime(2026, 7, 6)` — a constant, for every date. That is correct enough for a single-date demo and wrong for a history: the environment's valuation date must be the trade date. `environment_for` uses the trade date, and the fourth test above pins that. This is a deliberate behaviour change from the ported function, recorded in the module `CLAUDE.md` (Task 22).

- [ ] **Step 2: Run it to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/volcalibration/test_calibration_set.py -x -q
```
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write `calibration_set.py`**

```python
"""The producer/consumer handover: a directory of artifacts as one object.

A backtest should not glob directories, parse manifests or rebuild a pricing
environment from an artifact's per-expiry block -- three jobs it did before,
each written three times across the example suite.  ``CalibrationSet`` is the
one supported entry point: open a store, get dates, surfaces, calibrated models
and pricing environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from quantark.param import GridVolSurface, SpotQuote
from quantark.param.div import TermStructureDividendYield
from quantark.param.rrf.rate_curve import LinearRateCurve
from quantark.param.vol.surface_history import IvSurfaceArtifact, VolSurfaceHistory
from quantark.priceenv import PricingEnvironment
from quantark.util.exceptions import ValidationError
from quantark.volcalibration.calibrate import CalibratedVolModel, VolModelCalibrator
from quantark.volcalibration.config import CalibrationRunConfig, RunConfig
from quantark.volcalibration.store import (
    StoreLayout,
    load_calibration_manifest,
    read_json,
)


@dataclass(frozen=True)
class CalibrationSet:
    """Read-only view of one calibration store."""

    layout: StoreLayout
    surface_history: VolSurfaceHistory
    calibration: CalibrationRunConfig
    _calibration_records: Dict[str, dict]

    @classmethod
    def open(cls, root, *, runtime=None, calibration=None) -> "CalibrationSet":
        """Validate a store's manifests and return a reader. Fail-closed."""
        layout = StoreLayout(Path(root), Path(runtime) if runtime else Path(root))
        if not layout.surface_manifest.is_file():
            raise ValidationError(
                f"no surface manifest at {layout.surface_manifest}; "
                "this directory is not a calibration store"
            )
        history = VolSurfaceHistory(layout.history_dir)
        _payload, records = load_calibration_manifest(layout)
        return cls(
            layout=layout,
            surface_history=history,
            calibration=calibration or CalibrationRunConfig(),
            _calibration_records=records,
        )

    @classmethod
    def from_config(cls, config: RunConfig) -> "CalibrationSet":
        return cls.open(
            config.history_dir,
            runtime=config.runtime_dir,
            calibration=config.calibration,
        )

    def dates(self) -> List[date]:
        """Admitted trading dates, ascending."""
        return list(self.surface_history.admitted_dates)

    def surface_for(self, when: date) -> IvSurfaceArtifact:
        """The artifact in force on ``when`` (manifest carry-forward applied)."""
        return self.surface_history.surface_for(when)

    def model_for(self, when: date, variant: str) -> CalibratedVolModel:
        """The calibrated model for ``when``'s surface.

        Reads the warm cache keyed on (surface sha, variant, config
        fingerprint), so this is a lookup for any date the runner has already
        calibrated and a fit for any date it has not.
        """
        artifact = self.surface_for(when)
        tag = _tag(artifact)
        record = self._calibration_records.get(tag, {})
        variant_record = record.get("variants", {}).get(variant, {})
        if record and variant_record.get("status") == "failed":
            raise ValidationError(
                f"{when}: calibration of {variant!r} for surface {artifact.sha256[:12]} "
                f"failed: {variant_record.get('error')}"
            )
        if not record:
            raise ValidationError(
                f"{when}: no calibration record for surface {artifact.sha256[:12]}; "
                f"run `python -m quantark.volcalibration run` for {tag} first"
            )
        if variant not in record.get("variants", {}):
            raise ValidationError(
                f"{when}: surface {artifact.sha256[:12]} has no {variant!r} "
                f"calibration; recorded variants are "
                f"{sorted(record.get('variants', {}))}"
            )
        calibrator = VolModelCalibrator(
            self.calibration.calibrator_config(self.layout.calibration_cache)
        )
        return calibrator.calibrate(variant, artifact)

    def environment_for(self, when: date) -> Tuple[PricingEnvironment, GridVolSurface, float]:
        """(PricingEnvironment, GridVolSurface, spot) for ``when``.

        The rate and dividend curves are built from the artifact's own parity
        pillars, so the environment carries the exact term structure the
        surface was calibrated against.  The valuation date is the trade date
        -- ``_mo_common.build_env`` hardcoded one calendar day, which is
        harmless for a single-date demo and wrong for a history.
        """
        artifact = self.surface_for(when)
        payload = artifact.payload
        s0 = float(payload["s0"])
        grid = GridVolSurface(
            payload["strikes"],
            payload["maturities"],
            np.asarray(payload["iv_grid"], dtype=float),
        )
        pillars = payload["per_expiry"]
        if len(pillars) < 2:
            raise ValidationError(
                f"{when}: surface has {len(pillars)} parity pillar(s); both curve "
                "types need at least 2"
            )
        env = PricingEnvironment(
            rate_curve=LinearRateCurve([(float(p["T"]), float(p["r"])) for p in pillars]),
            valuation_date=datetime(when.year, when.month, when.day),
            spot_quote=SpotQuote(spot=s0),
            vol_surface=grid,
            div_yield=TermStructureDividendYield(
                times=[float(p["T"]) for p in pillars],
                yields=[float(p["q"]) for p in pillars],
            ),
        )
        return env, grid, s0

    def status(self) -> Dict[str, Any]:
        """The store's last written status payload, or an empty dict."""
        return read_json(self.layout.status, default={})


def _tag(artifact: IvSurfaceArtifact) -> str:
    return str(artifact.payload["trade_date"]).replace("-", "")
```

`VolSurfaceHistory.admitted_dates` is a public property (`quantark/param/vol/surface_history.py:407`) and `surface_for` is at `:412`; use them, never `_admitted`.

- [ ] **Step 4: Re-export from `__init__.py`**

Add `CalibrationSet` to the imports and `__all__`.

- [ ] **Step 5: Run the tests**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/volcalibration/test_calibration_set.py -q
```
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add quantark/volcalibration/calibration_set.py quantark/volcalibration/__init__.py test/volcalibration/test_calibration_set.py
git commit -m "$(cat <<'EOF'
feat(volcalibration): CalibrationSet, the one supported consumer entry point

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YSMxE68spnG5uF97YdTCGU
EOF
)"
```

---

### Task 20: The backtest handover field

**Files:**
- Modify: `quantark/backtest/replay/market.py:132-167`
- Test: `test/volcalibration/test_calibration_set.py` (append)

**Interfaces:**
- Consumes: `CalibrationSet` (Task 19)
- Produces: `AutocallableMarketDataSet.calibration_set: Optional[CalibrationSet]`

- [ ] **Step 1: Write the failing test**

Append to `test/volcalibration/test_calibration_set.py`:

```python
def test_backtest_accepts_a_calibration_set_instead_of_a_directory(history):
    import pandas as pd

    from quantark.backtest.replay.market import AutocallableMarketDataSet

    cs = CalibrationSet.open(history)
    empty = pd.DataFrame({"date": [], "spot": []})
    data = AutocallableMarketDataSet(
        spot_data=empty.set_index("date"),
        vol_data=empty.rename(columns={"spot": "volatility"}).set_index("date"),
        rate_data=empty.rename(columns={"spot": "rate"}).set_index("date"),
        futures_data=pd.DataFrame(columns=["date", "contract", "expiry_date", "price"]),
        calibration_set=cs,
    )
    # The surface channel is derived, so nothing downstream changes.
    assert data.surface_history is cs.surface_history


def test_supplying_both_channels_is_refused(history):
    import pandas as pd

    from quantark.backtest.replay.market import AutocallableMarketDataSet
    from quantark.util.exceptions import ValidationError as VE

    cs = CalibrationSet.open(history)
    empty = pd.DataFrame({"date": [], "spot": []})
    with pytest.raises(VE):
        AutocallableMarketDataSet(
            spot_data=empty.set_index("date"),
            vol_data=empty.rename(columns={"spot": "volatility"}).set_index("date"),
            rate_data=empty.rename(columns={"spot": "rate"}).set_index("date"),
            futures_data=pd.DataFrame(columns=["date", "contract", "expiry_date", "price"]),
            surface_history=cs.surface_history,
            calibration_set=cs,
        )
```

- [ ] **Step 2: Run it to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/volcalibration/test_calibration_set.py -k backtest -x -q
```
Expected: FAIL — `TypeError: unexpected keyword argument 'calibration_set'`.

- [ ] **Step 3: Add the field**

In `quantark/backtest/replay/market.py`, extend `AutocallableMarketDataSet`:

```python
    surface_history: Optional[VolSurfaceHistory] = None
    calibration_set: Optional[Any] = None

    def __post_init__(self) -> None:
        # A CalibrationSet is the supported way to attach a calibration store:
        # it validates the manifests and owns the paths, so nothing in backtest
        # reaches into a directory.  ``surface_history`` remains for callers
        # who already hold one; supplying both would leave two sources of
        # truth for the same channel.
        if self.calibration_set is not None:
            if self.surface_history is not None:
                raise ValidationError(
                    "supply either surface_history or calibration_set, not both; "
                    "a CalibrationSet already carries its surface history"
                )
            object.__setattr__(
                self, "surface_history", self.calibration_set.surface_history
            )
```

`calibration_set` is typed `Optional[Any]` with the real type named in the docstring: annotating it `Optional[CalibrationSet]` would make `quantark.backtest.replay.market` import `quantark.volcalibration.calibration_set` at module load, which pulls the SABR/Heston calibrator stack into every backtest import. Add the same keyword to `from_dataframes` and update the class docstring.

`AutocallableMarketDataSet` is a plain `@dataclass`, so `__post_init__` may assign directly; use `object.__setattr__` only if it is frozen — check before writing.

- [ ] **Step 4: Run the tests**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/volcalibration/ test/replay_golden -q
```
Expected: PASS, with the replay golden suite unchanged.

- [ ] **Step 5: Commit**

```bash
git add quantark/backtest/replay/market.py test/volcalibration/test_calibration_set.py
git commit -m "$(cat <<'EOF'
feat(backtest): accept a CalibrationSet instead of a surface directory

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YSMxE68spnG5uF97YdTCGU
EOF
)"
```

---

### Task 21: The FX delta-quoted normalizer

**Files:**
- Create: `quantark/volcalibration/normalize/fxdelta.py`
- Modify: `quantark/volcalibration/normalize/__init__.py`
- Test: `test/volcalibration/test_fxdelta.py`

**Interfaces:**
- Consumes: `QuoteSnapshot`, `QuoteSet`, `ExpiryQuotes`, `IvNode`
- Produces:
  - `PILLAR_ORDER`, `PILLAR_DELTA`, `TENOR_ORDER`, `TENOR_SETS`, `normalise_tenor`
  - `strike_from_spot_delta(forward, iv, maturity, foreign_rate, delta) -> float`
  - `spot_delta_from_strike(forward, strike, iv, maturity, foreign_rate, *, is_call) -> float`
  - `FxDeltaNormalizer(tenor_set="core").normalize(snapshot) -> QuoteSet`

- [ ] **Step 1: Write the failing test**

Create `test/volcalibration/test_fxdelta.py`:

```python
import json
from datetime import date
from pathlib import Path

import pytest

from quantark.util.exceptions import ValidationError
from quantark.volcalibration.normalize.fxdelta import (
    FxDeltaNormalizer,
    spot_delta_from_strike,
    strike_from_spot_delta,
)
from quantark.volcalibration.quotes import QuoteSet
from quantark.volcalibration.snapshot import QuoteSnapshot

SAMPLE = Path("example/fx_volmodels/data/cfets_usdcny_snapshot_20260430.json")


def test_delta_and_strike_round_trip():
    forward, iv, maturity, rf = 7.15, 0.045, 0.25, 0.045
    for delta in (-0.10, -0.25, 0.25, 0.10):
        strike = strike_from_spot_delta(forward, iv, maturity, rf, delta)
        recovered = spot_delta_from_strike(
            forward, strike, iv, maturity, rf, is_call=delta > 0.0
        )
        assert recovered == pytest.approx(delta, abs=1e-12)


def test_an_impossible_delta_is_refused():
    with pytest.raises(ValidationError):
        strike_from_spot_delta(7.15, 0.045, 0.25, 0.045, 1.5)


@pytest.mark.skipif(not SAMPLE.is_file(), reason=f"no CFETS sample at {SAMPLE}")
def test_the_fx_normalizer_emits_a_structurally_identical_quote_set():
    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
    snapshot = QuoteSnapshot.from_legacy_fx(payload)
    quotes = FxDeltaNormalizer(tenor_set="core").normalize(snapshot)
    assert isinstance(quotes, QuoteSet)
    assert len(quotes.expiries) == 6
    for slice_ in quotes.expiries:
        assert slice_.expiry_label  # a tenor label, no calendar date
        assert slice_.expiry_date is None
        assert len(slice_.nodes) == 5
        assert slice_.forward > 0.0
        assert [n.strike for n in slice_.nodes] == sorted(n.strike for n in slice_.nodes)


@pytest.mark.skipif(not SAMPLE.is_file(), reason=f"no CFETS sample at {SAMPLE}")
def test_one_end_to_end_fx_surface_builds_and_admits():
    from quantark.volcalibration.surface import build_artifact

    payload = json.loads(SAMPLE.read_text(encoding="utf-8"))
    snapshot = QuoteSnapshot.from_legacy_fx(payload)
    quotes = FxDeltaNormalizer(tenor_set="core").normalize(snapshot)
    artifact = build_artifact(quotes, snapshot)
    assert artifact["admission"]["status"] == "ok"
    assert len(artifact["maturities"]) == 6
```

- [ ] **Step 2: Confirm the sample path**

```bash
ls example/fx_volmodels/data/
```
If the committed CFETS snapshot has a different filename, correct `SAMPLE` before running. If no snapshot is committed, the two skipped tests stay skipped and the plan records it — do not invent a fixture that pretends to be CFETS data.

- [ ] **Step 3: Run it to verify it fails**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/volcalibration/test_fxdelta.py -x -q
```
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 4: Write `normalize/fxdelta.py`**

Port `normalise_tenor`, `strike_from_spot_delta`, `spot_delta_from_strike`, `selected_slices` and `iter_nodes` from `example/fx_volmodels/_fx_common.py:40-224`, changing only:

1. `ValueError` → `ValidationError` (`quantark.util.exceptions`) at every raise.
2. `iter_nodes`' flat list of dicts becomes `ExpiryQuotes` / `IvNode` objects: one `ExpiryQuotes` per tenor with `expiry_label=tenor`, `expiry_date=None`, `T=maturity`, `forward`, `discount_factor=exp(-domestic_rate * maturity)`, `r=domestic_rate`, `q=foreign_rate`, and five `IvNode(strike, mid_iv, weight_hint=None)`.
3. The delta round-trip check that `load_snapshot` performs stays, moved into `FxDeltaNormalizer.normalize` so that the normalizer — not a loader in `example/` — is what refuses a snapshot whose published deltas do not reconstruct.

```python
class FxDeltaNormalizer:
    """CFETS five-delta tenor slices -> QuoteSet.

    The delta convention is CFETS non-premium-adjusted spot delta:
    ``Delta_call = exp(-r_f T) N(d1)``, ``Delta_put = exp(-r_f T) (N(d1)-1)``.
    Strikes are published alongside the deltas, so this normalizer does not
    invert them -- it *verifies* the round trip and refuses a snapshot whose
    published strikes and deltas disagree, which is the FX analogue of the
    listed path's parity quality gate.
    """

    def __init__(self, tenor_set="core") -> None:
        self.tenor_set = tenor_set

    def normalize(self, snapshot: QuoteSnapshot) -> QuoteSet:
        ...
```

`QuoteSnapshot.from_legacy_fx` lifts the CFETS snapshot into the canonical envelope, mirroring the two listed lifters — write it in `snapshot.py` in this task, with `convention="fx_delta"` and `price_field="mid_or_last"` (the CFETS quotes are mid IVs).

- [ ] **Step 5: Register the normalizer**

In `normalize/__init__.py`, add `FxDeltaNormalizer` to the exports so `runner.normalizer_for` resolves it.

- [ ] **Step 6: Run the tests**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/volcalibration/test_fxdelta.py -q
```
Expected: PASS (4 tests, or 2 passed + 2 skipped if no CFETS sample is committed).

- [ ] **Step 7: Commit**

```bash
git add quantark/volcalibration/normalize/fxdelta.py quantark/volcalibration/normalize/__init__.py quantark/volcalibration/snapshot.py test/volcalibration/test_fxdelta.py
git commit -m "$(cat <<'EOF'
feat(volcalibration): FX delta-quoted normalizer converging on the same QuoteSet

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YSMxE68spnG5uF97YdTCGU
EOF
)"
```

---

### Task 22: Docs, the worked MO config, and the snapshot bridge

**Files:**
- Create: `quantark/volcalibration/CLAUDE.md`
- Create: `example/mo_volmodels/mo_calibration.yaml`
- Create: `example/mo_volmodels/export_snapshots.py`
- Modify: `CLAUDE.md` (root), `docs/superpowers/specs/2026-09-01-volcalibration-module-design.md`
- Test: `test/volcalibration/test_cli.py` (append one test)

- [ ] **Step 1: Write the worked MO config**

`example/mo_volmodels/mo_calibration.yaml` — every default written out, so the file *is* the documentation of what runs today:

```yaml
# The MO daily calibration, written down.  Every value here is the value the
# 787 admitted surfaces in data/history were built with; running this config
# rebuilds nothing, because a default surface block fingerprints identically
# to the frozen manifest (see quantark/volcalibration/CLAUDE.md).
schema_version: 1
name: mo-daily
underlying:
  symbol: "000852.SH"
  convention: listed_strike
  price_field: settlement
paths:
  root: data/history
  runtime: ../../output/mo_daily_calibration
  spot_csv: data/history/csi1000_spot.csv
surface:
  sabr_beta: 1.0
  min_expiries: 2
  min_strikes_per_expiry: 5
  min_common_strikes: 3
  extrapolation: flat_total_variance
  parity_gate:
    max_abs_implied_rate: 0.10
    max_rmse_over_forward: 0.01
calibration:
  variants: [localvol, heston, heston_slv]
  heston_preset: mo_frozen
  heston_max_nfev: 200
  slv:
    eta: 1.0
    n_steps: 40
    n_x: 161
    n_z: 81
run:
  workers: 2
```

- [ ] **Step 2: Add a test that the shipped config loads**

Append to `test/volcalibration/test_cli.py`:

```python
def test_the_shipped_mo_config_loads_with_the_frozen_fingerprint():
    from quantark.volcalibration.store import builder_fingerprint
    from quantark.volcalibration.yaml_loader import load_run_config

    config = load_run_config("example/mo_volmodels/mo_calibration.yaml")
    assert config.name == "mo-daily"
    assert builder_fingerprint(config.surface.fingerprint_payload()) == builder_fingerprint(
        {"artifact_schema_version": 1, "min_common_strikes": 3, "min_expiries": 2,
         "min_strikes_per_expiry": 5, "sabr_beta": 1.0}
    ), "the shipped config must not invalidate the 787 admitted artifacts"
```

- [ ] **Step 3: Write the snapshot bridge**

`example/mo_volmodels/export_snapshots.py` — vendor format stays in `example/` (D1), so this is where the CSV history becomes CLI-readable:

```python
"""Export settlement CSVs as canonical volcalibration snapshots.

``quantark.volcalibration`` reads canonical ``QuoteSnapshot`` envelopes from
``<root>/snapshots/{YYYYMMDD}.json`` and never parses a vendor format -- that
boundary is what keeps the library network-free and deterministic.  This script
is the one-way bridge for the existing CFFEX CSV history.

Each exported snapshot carries the source CSV's own ``source_sha256`` verbatim,
so an exported date agrees with the sha already recorded in its migrated
manifest record and a resumed run rebuilds nothing.

Run:  .venv/bin/python example/mo_volmodels/export_snapshots.py [--start D] [--end D]
"""
```

Implementation: glob `settlement_csv/*_1.csv`, `parse_cffex_csv(payload, tag)` from stage 01, look the spot up in `csi1000_spot.csv`, `QuoteSnapshot.from_legacy_settlement(...)`, write `snapshots/{tag}.json` through `atomic_write_json`. Skip a date whose snapshot already exists unless `--force`. Print one line per date and a total.

- [ ] **Step 4: Write the module `CLAUDE.md`**

`quantark/volcalibration/CLAUDE.md`, covering:

- **What this module is:** quotes → IV surface artifact → calibrated LV/Heston/SLV, with the artifact as the frozen contract between producer and consumers.
- **The `BUILDER_SCHEMA_VERSION` obligation**, stated as an obligation and not a note: *bump it whenever a change to normalization, smoothing or admission code alters builder output.* A config change is covered by the fingerprint; a code change is not, and the version bump is the only mechanism that invalidates artifacts after a builder upgrade. Same contract `_CACHE_SCHEMA_VERSION` carries in `calibrate.py`.
- **The artifact-bytes constraint** (§5.3) and why the fingerprint payload is legacy-shaped (DP-2).
- **The two bounded exceptions to "any mismatch rebuilds":** `grandfathered` records and foreign exclusions (DP-1), each with the reason it exists and `--force` as the escape hatch.
- **The agent contract:** the four exit codes, `exit code -> reason code -> act`, and the standing warning that 2 is not retryable.
- **Deliberate divergences from the ported example code**, so nobody "fixes" them back: `environment_for` dates the environment to the trade date where `_mo_common.build_env` hardcoded 2026-07-06; `HESTON_PRESETS["mo_frozen"]` sets `enforce_feller=True` where the diagnostics scripts use the soft penalty.
- **A worked run:**
  ```
  .venv/bin/python example/mo_volmodels/export_snapshots.py
  .venv/bin/python -m quantark.volcalibration run example/mo_volmodels/mo_calibration.yaml --plan --json
  .venv/bin/python -m quantark.volcalibration run example/mo_volmodels/mo_calibration.yaml
  .venv/bin/python -m quantark.volcalibration status example/mo_volmodels/mo_calibration.yaml --json
  ```
- **The snapshot schema**, field by field, with the `price_field` discipline spelled out: a snapshot declaring `settlement` whose quotes carry only `last` is rejected by name, never repaired.
- **What is NOT here:** certification (`quantark.modelvalidation`), vendor fetching (`example/` and the akshare/wind skills), fleet/cohort orchestration.

- [ ] **Step 5: Add the root `CLAUDE.md` row**

One row in the module table, matching the surrounding style:

```
| `quantark/volcalibration/` | Market quotes -> admitted IV surface artifacts -> calibrated LV/Heston/SLV; YAML + CLI (`python -m quantark.volcalibration`), `CalibrationSet` for backtest |
```

- [ ] **Step 6: Write the design decisions back into the spec**

Append a `### 6.4 Decisions resolved during implementation` section to the spec recording DP-1, DP-2 and DP-3 with their reasons, and correct §4.1's layout to show `calibration_set.py` and `runner.py`'s real contents.

- [ ] **Step 7: Run the full volcalibration suite plus the mo suite**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test/volcalibration test/mo_volmodels -q -n auto --dist worksteal
```
Expected: PASS, with only the two pre-existing `test_stage12_backtest_runner` failures (they need the git-excluded `data/history/csi1000_spot.csv`).

- [ ] **Step 8: Restore churned sample data, then commit by explicit path**

```bash
git checkout -- example/mo_volmodels/data/mo_barrier_sample.json example/mo_volmodels/data/mo_calibration_explainer_sample.html
git add example/mo_volmodels/mo_calibration.yaml example/mo_volmodels/export_snapshots.py
git add quantark/volcalibration/CLAUDE.md CLAUDE.md test/volcalibration/test_cli.py
git add -f docs/superpowers/specs/2026-09-01-volcalibration-module-design.md
git status --short
```
Check the `git status --short` output before committing: nothing under `example/mo_volmodels/data/` may be staged.

```bash
git commit -m "$(cat <<'EOF'
docs(volcalibration): module guide, worked MO config and the snapshot bridge

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YSMxE68spnG5uF97YdTCGU
EOF
)"
```

- [ ] **Step 9: Full-tree regression**

```bash
PYTHONPATH=$PWD /Users/fuxinyao/quant-ark/.venv/bin/python -m pytest test -q -n auto --dist worksteal
```
Expected: the same two pre-existing failures as the phase 1–3 baseline (`2 failed, 6685 passed, 120 skipped` plus this plan's new tests), and nothing else.

---

## Self-Review

**Spec coverage.** Walking §9's phases 4–7 and §6:

| Spec requirement | Task |
|---|---|
| §6.1 run config, every stage-03 constant as a defaulted key | 13 |
| §6.1 resolved config echoed into both manifests | 13 (`RunConfig.echo`), 14 (writers) |
| §5.4 store layout, three manifests, atomic writes | 14 |
| §5.4 resume invalidation on all four fields | 15 |
| §8 "resume invalidation" — five cases | 15 (4 rebuild cases + the unchanged-skip case) |
| §7.1 `_build_one` worker pool → `runner.py` | 15 |
| §7.1 stage-14 selection, calibration, EWMA → `runner.py` | 16 |
| §7.1 `acquire_lock`, `build_freshness_status`, `status_exit_code` | 17 |
| §7.1 delete `default_stage_runner` | 17 (the `stages` block goes with it) |
| §6.2 four verbs, `--plan`, `--json` | 18 |
| §6.3 agent contract, all four clauses | 18 |
| §5.5 `CalibrationSet` incl. `environment_for` | 19 |
| §5.5 one backtest config field | 20 |
| §7.1 `_fx_common` functions → `normalize/fxdelta.py` | 21 |
| §8 FX parity + one end-to-end FX surface | 21 |
| §9.7 module `CLAUDE.md`, worked config, snapshot schema, root table row | 22 |

**Gaps, stated rather than hidden:**

- **§8's "legacy migration against a copy of the real manifest" is not a task here.** `migrate_manifest` and its unit tests shipped in phase 2 (`5a7cbdf`). The test that runs it against a *copy of the real 809-record manifest* and asserts zero artifacts modified needs the git-excluded history, so it would skip in CI exactly like `test_artifact_reproduction.py`. Task 15's `test_a_second_run_skips_and_leaves_the_bytes_identical` covers the same invariant on a store the test builds itself, which is the part that can actually run. Running the real-manifest check is a one-command manual step before any production run, documented in the module `CLAUDE.md`.
- **§7.1 puts `build_env` and `build_fx_environment` in `store.py`.** Both land in `calibration_set.py` instead, for the import-weight reason given in the File Structure note. The FX one is folded into Task 21's normalizer work since it consumes the FX slice shape.
- **`_mo_common.build_env` and `_fx_common.build_fx_environment` are not converted to shims.** Phase 1–3 shimmed the functions whose behaviour the 35 mo test modules pin. These two build environments the demos use directly and have no library-equivalent signature (`environment_for` takes a date and a store, not a surface dict). Converting them is a follow-up, not a prerequisite, and the plan does not pretend otherwise.

**Placeholder scan.** No "TBD"/"handle edge cases"/"similar to Task N". One step deliberately says *port this code with these changes* rather than reproducing it inline: Task 16's port of stage 14's calibration half, which is line-referenced (`14_daily_calibration_pipeline.py:389-812`) with all six required changes enumerated. Reproducing 400 lines verbatim into the plan would make it harder to diff against the original, not easier.

**Type consistency.** `variants` is threaded as an explicit keyword through `calibration_record_is_current`, `select_calibration_dates` and `run_calibration_stage` (Task 16) and consumed by `_cmd_run` (Task 18). `StoreLayout` is constructed by `from_config` everywhere except `CalibrationSet.open`, which builds it from a root path. `builder_fingerprint` always takes `config.surface.fingerprint_payload()`, never the `RunConfig`. `surface_record`'s `fingerprint=` keyword (phase 2) is what Tasks 15 and 22 pass.

**Sequencing risk.** Task 20 touches `quantark/backtest/replay/market.py` — the one file in this plan outside `volcalibration/`. Run `test/replay_golden` in that task, not only at the end, so a regression there is attributed to the change that caused it.


---

## Execution Notes (2026-09-01, all 10 tasks complete)

Recorded rather than silently edited into the tasks above, so the next reader
can see where the plan was wrong.

**The plan missed a whole design problem in Task 21.** It assumed the FX
normalizer only had to emit a `QuoteSet` and the existing `build_raw_surface`
would take it. It does not: that function builds its grid from strikes quoted
by >= 2 expiries, and delta-quoted books share no strike at all -- every
tenor's 25-delta strike is its own. The first end-to-end FX test failed with
`insufficient_expiries: < 2 expiries with >= 5 nodes inside the quoted-range
overlap; dropped=[all five]`. Fixed by making the grid rule a field the
normalizer declares in `QuoteSet.universe` (`STRIKE_GRID_SHARED_OBSERVED` vs
`STRIKE_GRID_UNIFORM_OVER_OVERLAP`), which keeps listed artifacts byte-exact
and refuses to auto-detect. This is the one place §4.4's "everything downstream
is convention-blind" needed narrowing, now recorded in spec §6.4.

**The plan missed a second one in the same task.** Delta-quoted books publish
implied vols, not prices, so neither existing `price_field` describes them.
Added `PRICE_FIELD_MID_IV`, made `quote_price` refuse it by name, and made
`UnderlyingConfig` reject a convention/price-field pair no venue produces.

**Three factual errors in the plan's own pre-resolved facts**, all caught by
checking the tree before writing code: the 20260430 artifact's `s0` is
8381.947 (the plan guessed 6167.16); the committed CFETS snapshots are
per-date (`cfets_usdcny_snapshot_20260430.json`, not a single file); and
`snapshot.py` already defined `CONVENTION_LISTED` / `PRICE_FIELD_*`, which
`config.py` was about to redeclare.

**A missing import survived three green test runs.** `runner.run_pipeline` used
`atomic_write_json` without importing it, and only the Task 18 CLI tests
reached that line -- Task 17's status tests never call `run_pipeline`. Worth
remembering: a unit test of the pieces is not a test of the transaction.

**Deviations from the plan as written, each with its reason:**

- `test_calibration_set.py` imports fixtures via `sys.path` + `from
  replay_golden.fixtures import ...`, matching `test/test_calibration_relocation.py`.
  `test/` is not a package, so the plan's `from test.replay_golden...` fails.
- The root `CLAUDE.md` row was **not** applied. That file is git-excluded and
  exists only in the main checkout, so editing it from a worktree would mutate
  shared state that cannot be reviewed on this branch. The exact line to add is
  in the handover.
- `quantark/volcalibration/CLAUDE.md` was force-added despite the repo-wide
  `CLAUDE.md` exclusion. The exclusion targets the root working-notes file; a
  module guide recording the `BUILDER_SCHEMA_VERSION` obligation has to ship
  with the code. One `git rm --cached` reverses it.

**Findings worth carrying forward:**

- The committed settlement snapshots carry `expiry_calendar.frozen_overrides`
  (`{"2606": "2026-06-22"}`), while `normalize/settlement.py` hardcodes the
  same table as `EXPIRY_DATE_OVERRIDES`. The snapshot is the authoritative
  copy and the hardcoded one will go stale. Reading the snapshot's table when
  present (falling back to the frozen one) is strictly better and byte-exact
  on every sample -- not done here because it changes phase-2 code that is
  already certified.
- `show --json` needed no new diagnostics: the artifact and calibration records
  already carry `parity_rmse_points`, `sabr_params.mse`, `feller_ratio`,
  `bound_hits`, `overall_rmse_iv` and `leverage_min/max`. The verb exists so
  nobody reads three JSON files by hand, not to invent a fourth format.
