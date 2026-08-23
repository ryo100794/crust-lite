from __future__ import annotations

import json
from pathlib import Path

import pytest

from crust_lite.config import load_config
from crust_lite.io.database import connect, database_path, database_status, sqlite_database_path
from crust_lite.paths import ProjectPaths


def _production_paths() -> ProjectPaths:
    config = load_config("configs/east_japan_usgs.yml")
    return ProjectPaths.from_config(config)


def _temporary_paths(root: Path) -> ProjectPaths:
    return ProjectPaths(
        root=root,
        data_raw=root / "data/raw",
        data_interim=root / "data/interim",
        data_processed=root / "data/processed",
        outputs_maps=root / "outputs/maps",
        outputs_tables=root / "outputs/tables",
        outputs_dashboard=root / "outputs/dashboard",
        outputs_reports=root / "outputs/reports",
        outputs_3d=root / "outputs/3d",
    )


def _clear_database_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "CRUST_LITE_DATABASE_MODE",
        "CRUST_LITE_DATABASE_POINTER",
        "CRUST_LITE_DATABASE_OVERRIDE",
        "CRUST_LITE_ENABLE_DEVELOPMENT_DATABASE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_is_verified_hinet_v2_and_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_database_environment(monkeypatch)
    paths = _production_paths()
    status = database_status(paths)
    assert status["path"].endswith("crust_lite_hinet_only_20260823_v2.duckdb")
    assert status["role"] == "formal_hinet_only"
    assert status["source_policy"] == "NIED_HINET_ONLY"
    assert status["read_only"] is True
    assert status["verified"] is True
    with pytest.raises(RuntimeError, match="immutable"):
        connect(paths, read_only=False)
    con = connect(paths)
    try:
        assert con.execute("SELECT count(*) FROM event").fetchone()[0] == 6528
        assert con.execute(
            "SELECT count(*) FROM event WHERE catalog_source <> "
            "'NIED Hi-net authenticated event catalog'"
        ).fetchone()[0] == 0
        with pytest.raises(Exception):
            con.execute("CREATE TABLE forbidden_write(value INTEGER)")
    finally:
        con.close()


def test_formal_pointer_hash_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_database_environment(monkeypatch)
    pointer = json.loads(Path("configs/analysis_database.json").read_text(encoding="utf-8"))
    pointer["formal"]["sha256"] = "0" * 64
    bad_pointer = tmp_path / "bad-pointer.json"
    bad_pointer.write_text(json.dumps(pointer), encoding="utf-8")
    monkeypatch.setenv("CRUST_LITE_DATABASE_POINTER", str(bad_pointer))
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        database_status(_production_paths())


def test_legacy_runtime_selection_is_retired_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_database_environment(monkeypatch)
    pointer = json.loads(Path("configs/analysis_database.json").read_text(encoding="utf-8"))
    assert "legacy_read_only" not in pointer
    assert pointer["legacy_recovery"]["runtime_selection_allowed"] is False
    monkeypatch.setenv("CRUST_LITE_DATABASE_MODE", "legacy-read-only")
    with pytest.raises(RuntimeError, match="retired by NR-RDB-LEGACY-004"):
        database_status(_production_paths())


def test_sqlite_is_explicit_and_isolated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_database_environment(monkeypatch)
    paths = _temporary_paths(tmp_path)
    with pytest.raises(RuntimeError, match="requires"):
        monkeypatch.setenv("CRUST_LITE_DATABASE_MODE", "development-sqlite")
        database_status(paths)
    monkeypatch.setenv("CRUST_LITE_ENABLE_DEVELOPMENT_DATABASE", "1")
    status = database_status(paths)
    assert status["role"] == "development_only"
    assert status["verified"] is False
    assert Path(status["path"]).is_relative_to(tmp_path / "data/development")
    con = connect(paths, read_only=False)
    try:
        con.execute("CREATE TABLE sample_only(value INTEGER)")
        con.commit()
    finally:
        con.close()
    assert sqlite_database_path(paths) == tmp_path / "data/development/crust_lite_development.sqlite"
