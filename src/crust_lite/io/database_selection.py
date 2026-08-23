"""Fail-closed selection and verification for the analysis database.

The default is the immutable, independently audited Hi-net-only DuckDB named
by ``configs/analysis_database.json``. Legacy mixed-database runtime access
is retired. Development databases are isolated below ``data/development`` and
require an explicit acknowledgement environment variable.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from crust_lite.paths import ProjectPaths


POINTER_RELATIVE_PATH = Path("configs/analysis_database.json")
MODE_ENV = "CRUST_LITE_DATABASE_MODE"
POINTER_ENV = "CRUST_LITE_DATABASE_POINTER"
OVERRIDE_ENV = "CRUST_LITE_DATABASE_OVERRIDE"
DEVELOPMENT_ACK_ENV = "CRUST_LITE_ENABLE_DEVELOPMENT_DATABASE"
FORMAL_MODE = "formal"
DEVELOPMENT_DUCKDB_MODE = "development-duckdb"
DEVELOPMENT_SQLITE_MODE = "development-sqlite"


@dataclass(frozen=True)
class DatabaseSelection:
    path: Path
    engine: str
    mode: str
    role: str
    read_only: bool
    source_policy: str
    verified: bool
    pointer_manifest: Path | None
    database_sha256: str | None
    audit_path: Path | None
    audit_sha256: str | None
    verification: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("path", "pointer_manifest", "audit_path"):
            value = payload.get(key)
            payload[key] = str(value) if value is not None else None
        return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_under_root(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _pointer_path(paths: ProjectPaths) -> Path:
    override = os.environ.get(POINTER_ENV)
    return _resolve_under_root(paths.root, override) if override else paths.root / POINTER_RELATIVE_PATH


def _load_pointer(paths: ProjectPaths) -> tuple[Path, dict[str, Any]]:
    pointer_path = _pointer_path(paths).resolve()
    if not pointer_path.is_file():
        raise RuntimeError(f"analysis database pointer is missing: {pointer_path}")
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"analysis database pointer is unreadable: {pointer_path}: {exc}") from exc
    if pointer.get("schema") != "crust-lite-analysis-database-pointer-v1":
        raise RuntimeError(f"unsupported analysis database pointer schema: {pointer.get('schema')!r}")
    return pointer_path, pointer


def _require_duckdb() -> Any:
    try:
        import duckdb  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "DuckDB is required for the formal Hi-net-only analysis database; "
            "implicit SQLite fallback is disabled"
        ) from exc
    return duckdb


def _verify_formal(
    paths: ProjectPaths, pointer_path: Path, pointer: dict[str, Any]
) -> DatabaseSelection:
    formal = pointer.get("formal") or {}
    if formal.get("role") != "formal_hinet_only" or formal.get("engine") != "duckdb":
        raise RuntimeError("formal pointer must declare role=formal_hinet_only and engine=duckdb")
    if formal.get("read_only") is not True:
        raise RuntimeError("formal analysis database must be declared read_only=true")
    database = _resolve_under_root(paths.root, str(formal.get("path", "")))
    if not database.is_file():
        raise RuntimeError(f"formal analysis database is missing: {database}")
    mode = stat.S_IMODE(database.stat().st_mode)
    if mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise RuntimeError(f"formal analysis database is writable (mode {oct(mode)}): {database}")
    expected_database_sha = str(formal.get("sha256", ""))
    actual_database_sha = _sha256(database)
    if not expected_database_sha or actual_database_sha != expected_database_sha:
        raise RuntimeError(
            f"formal database SHA-256 mismatch: expected={expected_database_sha}, "
            f"actual={actual_database_sha}"
        )

    audit_spec = formal.get("independent_audit") or {}
    audit_path = _resolve_under_root(paths.root, str(audit_spec.get("path", "")))
    if not audit_path.is_file():
        raise RuntimeError(f"independent PASS audit is missing: {audit_path}")
    expected_audit_sha = str(audit_spec.get("sha256", ""))
    actual_audit_sha = _sha256(audit_path)
    if not expected_audit_sha or actual_audit_sha != expected_audit_sha:
        raise RuntimeError(
            f"independent audit SHA-256 mismatch: expected={expected_audit_sha}, "
            f"actual={actual_audit_sha}"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_checks = audit.get("checks") or {}
    if audit.get("status") != "PASS" or audit.get("failures") not in ([], None):
        raise RuntimeError("independent database audit is not PASS")
    if not audit_checks or not all(value is True for value in audit_checks.values()):
        raise RuntimeError("independent database audit contains a non-PASS check")
    if (audit.get("database") or {}).get("sha256") != actual_database_sha:
        raise RuntimeError("independent audit database SHA-256 does not match selected database")

    policy = str(formal.get("source_policy", ""))
    catalog_source = str(formal.get("catalog_source", ""))
    acquisition_source = str(formal.get("acquisition_source", ""))
    if policy != "NIED_HINET_ONLY" or not catalog_source or not acquisition_source:
        raise RuntimeError("formal pointer has incomplete Hi-net-only source policy")

    duckdb = _require_duckdb()
    con = duckdb.connect(str(database), read_only=True)
    try:
        tables = sorted(row[0] for row in con.execute("SHOW TABLES").fetchall())
        expected_rows = {str(k): int(v) for k, v in (formal.get("expected_rows") or {}).items()}
        actual_rows = {
            table: int(con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
            for table in expected_rows
        }
        if actual_rows != expected_rows:
            raise RuntimeError(
                f"formal database row contract mismatch: expected={expected_rows}, actual={actual_rows}"
            )
        catalog_violations = int(
            con.execute(
                "SELECT count(*) FROM event WHERE catalog_source IS NULL OR catalog_source<>?",
                [catalog_source],
            ).fetchone()[0]
        )
        source_violations: dict[str, int] = {}
        for table in (
            "waveform_feature",
            "waveform_spectrum",
            "phase_spectrum",
            "station_observation",
        ):
            source_violations[table] = int(
                con.execute(
                    f'SELECT count(*) FROM "{table}" WHERE acquisition_source IS NULL '
                    "OR acquisition_source<>?",
                    [acquisition_source],
                ).fetchone()[0]
            )
        policy_rows = con.execute(
            "SELECT policy_id,event_catalog_hinet_only,waveform_hinet_only,"
            "station_hinet_only,external_geology_used_for_model_input "
            "FROM formal_input_policy"
        ).fetchall()
        manifest_rows = con.execute(
            "SELECT source_id,allowed_for_model_input FROM source_manifest ORDER BY source_id"
        ).fetchall()
    finally:
        con.close()
    if catalog_violations or any(source_violations.values()):
        raise RuntimeError(
            f"formal database contains non-Hi-net provenance: catalog={catalog_violations}, "
            f"observations={source_violations}"
        )
    expected_policy_row = [("NIED_HINET_ONLY", True, True, True, False)]
    if policy_rows != expected_policy_row:
        raise RuntimeError(f"formal input policy mismatch: {policy_rows!r}")
    manifest_map = {str(source_id): bool(allowed) for source_id, allowed in manifest_rows}
    if manifest_map.get("nied_hinet_catalog") is not True:
        raise RuntimeError("source_manifest does not allow the Hi-net catalog")
    if manifest_map.get("nied_hinet_waveform") is not True:
        raise RuntimeError("source_manifest does not allow Hi-net waveforms")
    if manifest_map.get("external_geology_reference") is not False:
        raise RuntimeError("external geology must remain posthoc-only")

    verification = {
        "status": "PASS",
        "database_sha256_matches": True,
        "independent_audit_pass": True,
        "independent_audit_sha256_matches": True,
        "source_policy": policy,
        "catalog_source": catalog_source,
        "acquisition_source": acquisition_source,
        "catalog_source_violations": catalog_violations,
        "acquisition_source_violations": source_violations,
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "tables": tables,
    }
    return DatabaseSelection(
        path=database,
        engine="duckdb",
        mode=FORMAL_MODE,
        role="formal_hinet_only",
        read_only=True,
        source_policy=policy,
        verified=True,
        pointer_manifest=pointer_path,
        database_sha256=actual_database_sha,
        audit_path=audit_path,
        audit_sha256=actual_audit_sha,
        verification=verification,
    )


def _development_selection(paths: ProjectPaths, mode: str) -> DatabaseSelection:
    if os.environ.get(DEVELOPMENT_ACK_ENV) != "1":
        raise RuntimeError(
            f"{mode} requires {DEVELOPMENT_ACK_ENV}=1; implicit fallback is disabled"
        )
    development_root = (paths.root / "data" / "development").resolve()
    override_raw = os.environ.get(OVERRIDE_ENV)
    suffix = ".duckdb" if mode == DEVELOPMENT_DUCKDB_MODE else ".sqlite"
    selected = (
        _resolve_under_root(paths.root, override_raw)
        if override_raw
        else development_root / f"crust_lite_development{suffix}"
    )
    try:
        selected.relative_to(development_root)
    except ValueError as exc:
        raise RuntimeError(
            f"development database must stay below {development_root}: {selected}"
        ) from exc
    engine = "duckdb" if mode == DEVELOPMENT_DUCKDB_MODE else "sqlite"
    if engine == "duckdb":
        _require_duckdb()
    return DatabaseSelection(
        path=selected,
        engine=engine,
        mode=mode,
        role="development_only",
        read_only=False,
        source_policy="DEVELOPMENT_ONLY_NOT_FORMAL",
        verified=False,
        pointer_manifest=None,
        database_sha256=_sha256(selected) if selected.is_file() else None,
        audit_path=None,
        audit_sha256=None,
        verification={
            "status": "EXPLICIT_DEVELOPMENT_ONLY",
            "formal_analysis_allowed": False,
            "isolated_root": str(development_root),
        },
    )


def resolve_database(paths: ProjectPaths) -> DatabaseSelection:
    """Resolve one database mode without ever falling back implicitly."""
    mode = os.environ.get(MODE_ENV, FORMAL_MODE).strip().lower() or FORMAL_MODE
    if mode in {DEVELOPMENT_DUCKDB_MODE, DEVELOPMENT_SQLITE_MODE}:
        return _development_selection(paths, mode)
    if mode == "legacy-read-only":
        raise RuntimeError(
            "legacy-read-only runtime selection was retired by NR-RDB-LEGACY-004; "
            "use the isolated Google Drive recovery procedure, never formal analysis"
        )
    pointer_path, pointer = _load_pointer(paths)
    if mode == FORMAL_MODE:
        if os.environ.get(OVERRIDE_ENV):
            raise RuntimeError(
                f"{OVERRIDE_ENV} is forbidden in formal mode; use the signed pointer manifest"
            )
        return _verify_formal(paths, pointer_path, pointer)
    raise RuntimeError(
        f"unsupported {MODE_ENV}={mode!r}; expected one of "
        f"{FORMAL_MODE}, {DEVELOPMENT_DUCKDB_MODE}, {DEVELOPMENT_SQLITE_MODE}"
    )


def database_status(paths: ProjectPaths) -> dict[str, Any]:
    """Return the fully verified resolver state for CLI/report/preflight use."""
    return resolve_database(paths).to_dict()
