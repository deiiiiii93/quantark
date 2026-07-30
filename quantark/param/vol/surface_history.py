"""
Per-day IV-surface history channel for OTC autocallable backtests.

Wraps the SABR-smoothed IV-surface artifacts built by the mo_volmodels
history pipeline (``example/mo_volmodels/data/history``) so the backtest
engine can reprice daily against flat-ATM / term-structure / full-surface
vol instead of one scalar vol.

Gap policy (from ``surface_manifest.json``): the manifest admits a subset
of trading dates (``status == "ok"``); excluded dates have no artifact and
consumers carry the previous admitted surface forward.  The carry-forward
is implemented by :meth:`VolSurfaceHistory.surface_for`.

Artifact schema (per admitted date)::

    {
      "trade_date": "YYYY-MM-DD",
      "s0": float,
      "strikes": [float, ...],            # absolute strikes
      "maturities": [float, ...],         # year fractions, increasing
      "iv_grid": [[float, ...], ...],     # maturities x strikes
      "atm_pillars": [{"T", "expiry_date", "atm_vol"}, ...],
      "per_expiry": [{"T", "expiry_date", "r", "q", "forward", "df", ...}, ...],
      "extrapolation_policy": {"beyond_last_listed_expiry": str, "max_listed_T": float},
      "admission": {...},
    }

Loading is fail-closed: any schema violation raises ``ValidationError``
naming the offending artifact and field.
"""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from quantark.param import (
    GridVolSurface,
    TermStructureDividendYield,
    TermStructureVolSurface,
)
from quantark.util.exceptions import ValidationError

DateLike = Union[date, datetime, str]


def _as_date(value: DateLike) -> date:
    """Normalize a date-like value (date, datetime/Timestamp, or str) to ``date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        for fmt in ("%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
    raise ValidationError(f"Cannot interpret {value!r} as a date")


def _positive_finite(values, what: str, where: str) -> List[float]:
    out = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"{where}: {what} must be numeric, got {v!r}") from exc
        if not math.isfinite(f) or f <= 0.0:
            raise ValidationError(f"{where}: {what} must be positive and finite, got {f}")
        out.append(f)
    return out


def _strictly_increasing(values: List[float], what: str, where: str) -> None:
    if len(values) < 2:
        raise ValidationError(f"{where}: {what} must have at least 2 points")
    if any(values[i] >= values[i + 1] for i in range(len(values) - 1)):
        raise ValidationError(f"{where}: {what} must be strictly increasing")


class IvSurfaceArtifact:
    """
    Typed, read-only wrapper over one IV-surface artifact JSON.

    Schema is validated on load; any violation raises ``ValidationError``
    (fail-closed) so a malformed surface can never reach pricing.
    """

    def __init__(self, path: Union[str, Path], payload: Dict[str, Any], sha256: str) -> None:
        self._path = Path(path)
        self._payload = payload
        self._sha256 = sha256
        self._validate()

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "IvSurfaceArtifact":
        """Load and validate an artifact JSON file; sha256 is over the raw bytes."""
        p = Path(path)
        try:
            raw = p.read_bytes()
        except OSError as exc:
            raise ValidationError(f"IV surface artifact not readable: {p}") from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"IV surface artifact {p} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValidationError(f"IV surface artifact {p} must contain a JSON object")
        return cls(p, payload, hashlib.sha256(raw).hexdigest())

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        where = f"IV surface artifact {self._path.name}"
        payload = self._payload
        required = (
            "trade_date",
            "s0",
            "strikes",
            "maturities",
            "iv_grid",
            "atm_pillars",
            "per_expiry",
            "extrapolation_policy",
        )
        for key in required:
            if key not in payload:
                raise ValidationError(f"{where}: missing required key '{key}'")

        self._trade_date = _as_date(payload["trade_date"])
        self._s0 = _positive_finite([payload["s0"]], "s0", where)[0]

        self._strikes = tuple(
            _positive_finite(payload["strikes"], "strikes", where)
        )
        _strictly_increasing(list(self._strikes), "strikes", where)
        self._maturities = tuple(
            _positive_finite(payload["maturities"], "maturities", where)
        )
        _strictly_increasing(list(self._maturities), "maturities", where)

        iv_grid = payload["iv_grid"]
        if not isinstance(iv_grid, list) or len(iv_grid) != len(self._maturities):
            raise ValidationError(
                f"{where}: iv_grid must have one row per maturity "
                f"({len(self._maturities)}), got shape "
                f"({len(iv_grid) if isinstance(iv_grid, list) else '?'}, ...)"
            )
        rows = []
        for row in iv_grid:
            if not isinstance(row, list) or len(row) != len(self._strikes):
                raise ValidationError(
                    f"{where}: iv_grid rows must have one value per strike "
                    f"({len(self._strikes)})"
                )
            rows.append(tuple(_positive_finite(row, "iv_grid", where)))
        self._iv_grid = tuple(rows)

        self._atm_pillars = self._validate_pillars(
            payload["atm_pillars"], ("T", "atm_vol"), "atm_pillars", where
        )
        self._per_expiry = self._validate_pillars(
            payload["per_expiry"], ("T", "forward"), "per_expiry", where
        )

        policy = payload["extrapolation_policy"]
        if not isinstance(policy, dict) or "max_listed_T" not in policy:
            raise ValidationError(
                f"{where}: extrapolation_policy must define max_listed_T"
            )
        self._max_listed_T = _positive_finite(
            [policy["max_listed_T"]], "max_listed_T", where
        )[0]
        self._extrapolation_policy = dict(policy)

    @staticmethod
    def _validate_pillars(
        pillars, positive_keys: Tuple[str, ...], what: str, where: str
    ) -> Tuple[Dict[str, Any], ...]:
        if not isinstance(pillars, list):
            raise ValidationError(f"{where}: {what} must be a list")
        out = []
        times = []
        for pillar in pillars:
            if not isinstance(pillar, dict):
                raise ValidationError(f"{where}: {what} entries must be objects")
            row = dict(pillar)
            for key in positive_keys:
                if key not in row:
                    raise ValidationError(f"{where}: {what} entry missing '{key}'")
                row[key] = _positive_finite([row[key]], f"{what}.{key}", where)[0]
            times.append(row["T"])
            out.append(row)
        _strictly_increasing(times, f"{what}.T", where)
        return tuple(out)

    # ------------------------------------------------------------------
    # Accessors (read-only: sequences are returned as immutable copies)
    # ------------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self._path

    @property
    def trade_date(self) -> date:
        return self._trade_date

    @property
    def s0(self) -> float:
        return self._s0

    @property
    def strikes(self) -> Tuple[float, ...]:
        return self._strikes

    @property
    def maturities(self) -> Tuple[float, ...]:
        return self._maturities

    @property
    def iv_grid(self) -> Tuple[Tuple[float, ...], ...]:
        """Implied vols, shape (n_maturities, n_strikes): axis 0 maturity."""
        return self._iv_grid

    @property
    def atm_pillars(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(dict(p) for p in self._atm_pillars)

    @property
    def per_expiry(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(dict(p) for p in self._per_expiry)

    @property
    def sha256(self) -> str:
        """SHA-256 of the raw artifact file bytes (matches manifest artifact_sha256)."""
        return self._sha256

    @property
    def max_listed_T(self) -> float:
        return self._max_listed_T

    @property
    def extrapolation_policy(self) -> Dict[str, Any]:
        return dict(self._extrapolation_policy)

    # ------------------------------------------------------------------
    # Pricing-object builders (schema knowledge stays in this module)
    # ------------------------------------------------------------------
    def term_structure_vol_surface(self) -> TermStructureVolSurface:
        """ATM term-structure vol surface from the artifact's ``atm_pillars``."""
        return TermStructureVolSurface(
            times=[p["T"] for p in self._atm_pillars],
            vols=[p["atm_vol"] for p in self._atm_pillars],
        )

    def grid_vol_surface(self) -> GridVolSurface:
        """Full smile surface from ``strikes``/``maturities``/``iv_grid``.

        The artifact grid is maturities x strikes with absolute strikes,
        which is exactly the orientation ``GridVolSurface`` expects.
        """
        return GridVolSurface(
            strikes=list(self._strikes),
            maturities=list(self._maturities),
            iv_grid=[list(row) for row in self._iv_grid],
        )

    def implied_q_pillars(self, rate: float) -> Tuple[List[float], List[float]]:
        """
        Dividend-yield pillars implied by the artifact's parity forwards.

        From put-call parity, ``F_i = s0 * exp((r - q_i) * T_i)``, so

            ``q(T_i) = rate - ln(F_i / s0) / T_i``.

        The flat rate channel is kept fixed while the per-expiry carry is
        derived from the option-implied forwards, keeping the dividend curve
        internally consistent with the smile.
        """
        times: List[float] = []
        yields: List[float] = []
        for row in self._per_expiry:
            t = float(row["T"])
            times.append(t)
            yields.append(float(rate) - math.log(float(row["forward"]) / self._s0) / t)
        return times, yields

    def term_structure_dividend_yield(self, rate: float) -> TermStructureDividendYield:
        """
        Dividend term structure from :meth:`implied_q_pillars`.

        ``TermStructureDividendYield`` interpolates linearly between pillars
        and flat-extrapolates (clamps to the endpoint yield) outside the
        pillar range, which is the documented edge behavior beyond
        ``max_listed_T``.
        """
        times, yields = self.implied_q_pillars(rate)
        return TermStructureDividendYield(times=times, yields=yields)

    def __repr__(self) -> str:
        return (
            f"IvSurfaceArtifact(date={self._trade_date.isoformat()}, "
            f"nK={len(self._strikes)}, nT={len(self._maturities)})"
        )


class VolSurfaceHistory:
    """
    Lazy-loading history of admitted IV-surface artifacts with carry-forward.

    Reads ``surface_manifest.json`` from ``history_dir``, keeps the
    ``status == "ok"`` records only, and sorts the admitted dates.
    Artifacts are loaded from
    ``history_dir / artifact_subdir / filename_template`` on first access
    and cached afterwards; every loaded artifact's sha256 is verified
    against its manifest ``artifact_sha256`` record (fail-closed on
    mismatch).

    Memory note: a fully materialized cache costs roughly 40 MB for the
    762-artifact MO history (~50 KB per artifact).  Fleet runs (many
    inceptions over the same calendar) should share ONE instance — attach
    it to the shared ``market_data`` — rather than one per backtest run.

    Args:
        history_dir: Directory containing ``surface_manifest.json``.
        artifact_subdir: Sub-directory holding the per-date artifact files.
        filename_template: Artifact filename template; ``{yyyymmdd}`` is
            replaced by the admitted date in ``%Y%m%d`` format.
    """

    def __init__(
        self,
        history_dir: Union[str, Path],
        *,
        artifact_subdir: str = "iv_surface",
        filename_template: str = "mo_iv_surface_{yyyymmdd}.json",
    ) -> None:
        self._root = Path(history_dir)
        self._artifact_dir = self._root / artifact_subdir
        self._filename_template = filename_template
        self._cache: Dict[date, IvSurfaceArtifact] = {}
        self._expected_sha: Dict[date, Any] = {}
        self._admitted: Tuple[date, ...] = self._load_manifest()

    def _load_manifest(self) -> Tuple[date, ...]:
        manifest_path = self._root / "surface_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text())
        except OSError as exc:
            raise ValidationError(
                f"surface manifest not readable: {manifest_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"surface manifest {manifest_path} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(manifest, dict) or not isinstance(
            manifest.get("records"), list
        ):
            raise ValidationError(
                f"surface manifest {manifest_path} must contain a 'records' list"
            )
        admitted: List[date] = []
        for record in manifest["records"]:
            if not isinstance(record, dict) or "date" not in record or "status" not in record:
                raise ValidationError(
                    f"surface manifest {manifest_path}: every record needs "
                    "'date' and 'status'"
                )
            status = record["status"]
            if status == "excluded":
                continue
            if status != "ok":
                raise ValidationError(
                    f"surface manifest {manifest_path}: unknown status "
                    f"{status!r} for date {record['date']!r}"
                )
            try:
                admitted_date = datetime.strptime(str(record["date"]), "%Y%m%d").date()
            except ValueError as exc:
                raise ValidationError(
                    f"surface manifest {manifest_path}: unparseable date "
                    f"{record['date']!r} (expected YYYYMMDD)"
                ) from exc
            admitted.append(admitted_date)
            self._expected_sha[admitted_date] = record.get("artifact_sha256")
        if not admitted:
            raise ValidationError(
                f"surface manifest {manifest_path} admits no dates"
            )
        admitted.sort()
        if any(admitted[i] == admitted[i + 1] for i in range(len(admitted) - 1)):
            raise ValidationError(
                f"surface manifest {manifest_path} has duplicate admitted dates"
            )
        return tuple(admitted)

    @property
    def admitted_dates(self) -> List[date]:
        """Admitted (``status == "ok"``) surface dates, ascending."""
        return list(self._admitted)

    def surface_for(self, d: DateLike) -> IvSurfaceArtifact:
        """
        Return the artifact of the most recent admitted date on or before ``d``.

        This is the manifest's carry-forward gap policy: excluded dates (and
        any other non-admitted dates) reuse the previous admitted surface.

        Raises:
            ValidationError: If no admitted date exists on or before ``d``.
        """
        d = _as_date(d)
        idx = bisect_right(self._admitted, d) - 1
        if idx < 0:
            raise ValidationError(
                f"No admitted IV surface on or before {d.isoformat()} "
                f"(first admitted: {self._admitted[0].isoformat()})"
            )
        return self._load(self._admitted[idx])

    def sha_for(self, d: DateLike) -> str:
        """SHA-256 of the artifact used for ``d`` (post carry-forward; cache key)."""
        return self.surface_for(d).sha256

    def _load(self, admitted_date: date) -> IvSurfaceArtifact:
        artifact = self._cache.get(admitted_date)
        if artifact is None:
            path = self._artifact_dir / self._filename_template.format(
                yyyymmdd=admitted_date.strftime("%Y%m%d")
            )
            if not path.is_file():
                raise ValidationError(
                    f"Manifest admits {admitted_date.isoformat()} but artifact "
                    f"is missing: {path}"
                )
            artifact = IvSurfaceArtifact.from_file(path)
            expected = self._expected_sha.get(admitted_date)
            if not isinstance(expected, str) or not expected:
                raise ValidationError(
                    f"surface manifest record for {admitted_date.isoformat()} "
                    "lacks an artifact_sha256; refusing to admit an unverified "
                    "artifact"
                )
            if artifact.sha256 != expected:
                raise ValidationError(
                    f"IV surface artifact {path.name} sha256 mismatch for "
                    f"{admitted_date.isoformat()}: manifest artifact_sha256 is "
                    f"{expected}, file hashes to {artifact.sha256} — the "
                    "artifact does not match its admitted record"
                )
            self._cache[admitted_date] = artifact
        return artifact

    def __repr__(self) -> str:
        return (
            f"VolSurfaceHistory(root={self._root}, "
            f"admitted={len(self._admitted)} dates)"
        )
